/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Bowen Zhu
 *
 * License: BSD-3-Clause-LBNL
 */
#include "HybridMagDiffusion.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Fields.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX_Array.H>
#include <AMReX_Arena.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GMRES.H>
#include <AMReX_Loop.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>


using namespace amrex;

void
HybridMagDiffusion::ReadParameters ()
{
    const ParmParse pp("hybrid_pic_model");

    pp.query("implicit_mag_diffusion", m_enabled);
    utils::parser::queryWithParser(pp, "mag_diff_theta", m_theta);
    utils::parser::queryWithParser(pp, "mag_diff_rtol", m_rtol);
    utils::parser::queryWithParser(pp, "mag_diff_atol", m_atol);
    pp.query("mag_diff_max_iter", m_max_iter);
    pp.query("mag_diff_verbose", m_verbose);
    utils::parser::queryWithParser(pp, "mag_diff_eta_explicit_max", m_eta_explicit_max);
    pp.query("mag_diff_use_variable_eta", m_use_variable_eta);

    pp.query("mag_diff_linear_solver", m_linear_solver);
    pp.query("mag_diff_petsc_pc_type", m_petsc_options.pc_type);
    pp.query("mag_diff_petsc_asm_overlap", m_petsc_options.asm_overlap);
    pp.query("mag_diff_petsc_sub_ksp_type", m_petsc_options.sub_ksp_type);
    pp.query("mag_diff_petsc_sub_pc_type", m_petsc_options.sub_pc_type);
    pp.query("mag_diff_petsc_ilu_factor_levels", m_petsc_options.ilu_factor_levels);

    if (utils::parser::queryWithParser(pp, "mag_diff_constant_eta", m_constant_eta)) {
        m_has_constant_eta = true;
    }

    if (!m_enabled) { return; }

#ifndef AMREX_USE_PETSC
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_linear_solver == MagDiffLinearSolver::amrex_gmres,
        "hybrid_pic_model.mag_diff_linear_solver = petsc requires building WarpX "
        "with PETSc (-DWarpX_PETSC=ON, AMREX_USE_PETSC). The default amrex_gmres "
        "path needs no PETSc.");
#endif

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_theta > 0.0_rt && m_theta <= 1.0_rt,
        "hybrid_pic_model.mag_diff_theta must be in (0,1] "
        "(theta=0 explicit diffusion is not supported)");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(m_rtol) && std::isfinite(m_atol) &&
        m_rtol > 0.0_rt && m_atol >= 0.0_rt,
        "hybrid_pic_model.mag_diff_rtol/atol must be non-negative (rtol > 0)");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_max_iter > 0,
        "hybrid_pic_model.mag_diff_max_iter must be positive");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(m_eta_explicit_max) && m_eta_explicit_max >= 0.0_rt,
        "hybrid_pic_model.mag_diff_eta_explicit_max must be non-negative");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_has_constant_eta ||
            (std::isfinite(m_constant_eta) && m_constant_eta >= 0.0_rt),
        "hybrid_pic_model.mag_diff_constant_eta must be non-negative");

    if (m_linear_solver == MagDiffLinearSolver::petsc) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_petsc_options.asm_overlap >= 0 && m_petsc_options.ilu_factor_levels >= 0,
            "hybrid_pic_model.mag_diff_petsc_asm_overlap and "
            "mag_diff_petsc_ilu_factor_levels must be non-negative");

#ifdef AMREX_USE_GPU
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_petsc_options.pc_type != "lu",
            "hybrid_pic_model.mag_diff_petsc_pc_type = lu is not available on GPUs; "
            "use pc_type = asm with sub_pc_type = lu for local subdomain solves");
#endif
    }

    // theta=1 is backward Euler. For theta in (0,1), the same linear system is
    // used with a modified RHS that includes an explicit curl-curl term.
    amrex::Print() << "HybridMagDiffusion: enabled"
                   << "  theta=" << m_theta
                   << "  rtol=" << m_rtol
                   << "  atol=" << m_atol
                   << "  eta_explicit_max=" << m_eta_explicit_max
                   << "  use_variable_eta=" << m_use_variable_eta
                   << "  linear_solver=" << amrex::getEnumNameString(m_linear_solver);
    if (m_has_constant_eta) {
        amrex::Print() << "  constant_eta=" << m_constant_eta << " Ohm m";
    }
    if (m_linear_solver == MagDiffLinearSolver::petsc &&
        !m_petsc_options.pc_type.empty()) {
        amrex::Print() << "  petsc_pc_type=" << m_petsc_options.pc_type;
        if (m_petsc_options.pc_type == "asm") {
            amrex::Print() << "  petsc_asm_overlap=" << m_petsc_options.asm_overlap
                           << "  petsc_sub_ksp_type=" << m_petsc_options.sub_ksp_type
                           << "  petsc_sub_pc_type=" << m_petsc_options.sub_pc_type;
            if (m_petsc_options.sub_pc_type == "ilu") {
                amrex::Print() << "  petsc_ilu_factor_levels="
                               << m_petsc_options.ilu_factor_levels;
            }
        }
    }
    amrex::Print() << "\n";
}

namespace {

// File-scope device functors (not nested in VariableCoeffMagDiffusionOp).
// nvc++ rejects extended __device__ lambdas whose enclosing parent is a local
// class method ("must allow its address to be taken").

/** Average E/J-centered eta onto a B face for the Jacobi PC. */
struct SampleEtaOntoBFace
{
    Array4<Real> eta_pc;
    Array4<Real const> eta0;
    Array4<Real const> eta1;
    Array4<Real const> eta2;
    GpuArray<int, 3> s0{};
    GpuArray<int, 3> s1{};
    GpuArray<int, 3> s2{};
    GpuArray<int, 3> sb{};
    GpuArray<int, 3> coarsen{};

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void operator() (int i, int j, int k) const noexcept
    {
        Real const e0 = ablastr::coarsen::sample::Interp(
            eta0, s0, sb, coarsen, i, j, k, 0);
        Real const e1 = ablastr::coarsen::sample::Interp(
            eta1, s1, sb, coarsen, i, j, k, 0);
        Real const e2 = ablastr::coarsen::sample::Interp(
            eta2, s2, sb, coarsen, i, j, k, 0);
        eta_pc(i, j, k) = std::max(
            (e0 + e1 + e2) * (1.0_rt / 3.0_rt), Real(0.0));
    }
};

/** Scaled Jacobi diagonal apply using B-centered eta. */
struct MagDiffJacobiPrecond
{
    Array4<Real> dest;
    Array4<Real const> src;
    Array4<Real const> eta;
    // Stair-case EB mask for B (1 = active fluid face, 0 = covered). Default
    // (empty) when EB is off; on covered faces the operator row is identity,
    // so the PC is identity there too. Mirrors ComputeCurlA's mask pattern.
    Array4<int const> mask;
    Real theta_dt = 0.0_rt;
    Real diag_factor = 0.0_rt;   // Cartesian Laplacian diagonal (scalar; cyl unused)
    Real denom = 1.0_rt;
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
    // Cylindrical metric for the scaled-Jacobi approximation of the per-DOF
    // curl-curl diagonal. ishift_r = 1 if the component is NODE in r (Br), 0 if
    // CELL in r (Bt/Bz); is_bt = (idim == 1) -> subtract 1/r^2.
    Real rmin = 0.0_rt;
    Real dr = 1.0_rt;
    Real dr_inv2 = 0.0_rt;
    Real dz_inv2 = 0.0_rt;
    int ishift_r = 1;
    int idim = 0;
#endif

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void operator() (int i, int j, int k) const noexcept
    {
        // Covered B face: operator row is identity (curl is zeroed), so the
        // diagonal PC must be identity too (M^{-1} x = x). Keeps covered B
        // inert and prevents the PC from scaling a nonzero covered residual.
        if (mask && mask(i, j, k) == 0) {
            dest(i, j, k) = src(i, j, k);
            return;
        }
        // Use a precision-safe floor (1e-30 underflows to 0 in single precision).
        Real const tiny = Real(1.e-3) * std::numeric_limits<Real>::min();
        Real const eta_val = std::max(eta(i, j, k), Real(0.0));
        Real const chi_val = eta_val / PhysConst::mu0;
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
        // Exact geometric diagonal of the curl-curl operator at this DOF.
        // Br (idim 0) has only axial self-derivatives: 2/dz^2
        // Bz (idim 2) has only radial self-derivatives: 2/dr^2
        // Bt (idim 1) has both, minus 1/r^2 (applied off-axis to avoid zero/neg diag).
        Real const r = rmin + (i + (ishift_r ? 0.0_rt : 0.5_rt)) * dr;
        Real lap_diag = 0.0_rt;

        if (idim == 1) { // B_theta
            if (r > 0.0_rt) { lap_diag += 2.0_rt * dr_inv2; }
#if defined(WARPX_DIM_RZ)
            lap_diag += 2.0_rt * dz_inv2;
#endif
            if (r > 0.0_rt) {
                Real const inv_r2 = 1.0_rt / (r * r);
                if (lap_diag - inv_r2 > 0.0_rt) { lap_diag -= inv_r2; }
            }
        } else if (idim == 0) { // B_r
#if defined(WARPX_DIM_RZ)
            lap_diag = 2.0_rt * dz_inv2;
#endif
        } else if (idim == 2) { // B_z
            if (r > 0.0_rt) { lap_diag = 2.0_rt * dr_inv2; }
        }
#else
        Real const lap_diag = diag_factor;
#endif
        Real const diag = (1.0_rt + theta_dt * chi_val * lap_diag) / denom;
        Real const inv = Real(1.0) / std::max(diag, tiny);
        dest(i, j, k) = src(i, j, k) * inv;
    }
};

/** Form eta * J on one component (same staggering). */
struct FormEtaTimesJ
{
    Array4<Real> etaJ;
    Array4<Real const> eta;
    Array4<Real const> J;

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void operator() (int i, int j, int k) const noexcept
    {
        etaJ(i, j, k) = eta(i, j, k) * J(i, j, k);
    }
};

class MagDiffVector
{
public:
    using value_type = Real;

    MagDiffVector () = default;
    ~MagDiffVector () = default;
    MagDiffVector (MagDiffVector const&) = delete;
    MagDiffVector& operator= (MagDiffVector const&) = delete;
    MagDiffVector (MagDiffVector&&) noexcept = default;
    MagDiffVector& operator= (MagDiffVector&&) noexcept = default;

    void Define (ablastr::fields::VectorField const& source)
    {
        for (int idim = 0; idim < 3; ++idim) {
            m_fields[idim].define(
                source[idim]->boxArray(), source[idim]->DistributionMap(),
                source[idim]->nComp(), source[idim]->nGrowVect());
        }
        m_is_defined = true;
    }

    void Copy (MagDiffVector const& source)
    {
        AMREX_ALWAYS_ASSERT(m_is_defined && source.m_is_defined);
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Copy(m_fields[idim], source.m_fields[idim], 0, 0,
                           m_fields[idim].nComp(), m_fields[idim].nGrow());
        }
    }

    void CopyFrom (ablastr::fields::VectorField const& source)
    {
        AMREX_ALWAYS_ASSERT(m_is_defined);
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Copy(m_fields[idim], *source[idim], 0, 0,
                           m_fields[idim].nComp(), m_fields[idim].nGrow());
        }
    }

    void setVal (Real value)
    {
        AMREX_ALWAYS_ASSERT(m_is_defined);
        for (auto& field : m_fields) {
            field.setVal(value, field.nGrow());
        }
    }

    // Krylov algebra uses valid cells; apply/precond fill ghosts as needed.
    static bool isFiniteReal (Real v)
    {
        return (v == v) &&
               v <=  std::numeric_limits<Real>::max() &&
               v >= -std::numeric_limits<Real>::max();
    }

    void increment (MagDiffVector const& source, Real scale_factor)
    {
        AMREX_ALWAYS_ASSERT(m_is_defined && source.m_is_defined);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            isFiniteReal(scale_factor),
            "Hybrid magnetic-diffusion GMRES produced a non-finite increment coefficient");
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Saxpy(m_fields[idim], scale_factor, source.m_fields[idim],
                            0, 0, m_fields[idim].nComp(), 0);
        }
    }

    void scale (Real scale_factor)
    {
        AMREX_ALWAYS_ASSERT(m_is_defined);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            isFiniteReal(scale_factor),
            "Hybrid magnetic-diffusion GMRES produced a non-finite scale coefficient");
        for (auto& field : m_fields) {
            field.mult(scale_factor, 0, field.nComp(), 0);
        }
    }

    void linComb (Real a, MagDiffVector const& x, Real b, MagDiffVector const& y)
    {
        AMREX_ALWAYS_ASSERT(m_is_defined && x.m_is_defined && y.m_is_defined);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            isFiniteReal(a) && isFiniteReal(b),
            "Hybrid magnetic-diffusion GMRES produced a non-finite linear-combination coefficient");
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::LinComb(m_fields[idim], a, x.m_fields[idim], 0,
                               b, y.m_fields[idim], 0, 0,
                               m_fields[idim].nComp(), 0);
        }
    }

    [[nodiscard]] Real dotProduct (MagDiffVector const& source) const
    {
        AMREX_ALWAYS_ASSERT(m_is_defined && source.m_is_defined);
        Real result = 0.0_rt;
        for (int idim = 0; idim < 3; ++idim) {
            result += MultiFab::Dot(m_fields[idim], 0, source.m_fields[idim],
                                    0, m_fields[idim].nComp(), 0);
        }
        return result;
    }

    [[nodiscard]] Array<MultiFab,3>& fields () { return m_fields; }
    [[nodiscard]] Array<MultiFab,3> const& fields () const { return m_fields; }

private:
    Array<MultiFab,3> m_fields;
    bool m_is_defined = false;
};

// Zero covered B DOFs (eb_update_B[comp] == 0) in a Krylov vector. B^n is
// already 0 on covered B from the hybrid EM update and the matvec is identity
// there (ComputeCurlA zeros covered B), so this is defensive: it keeps the
// init-guess, RHS, and copy-back robust against any nonzero leaking onto
// covered B (feed offset, restart, single-precision noise). No-op when EB is
// off (returns before dereferencing the mask, which may be null then).
void zeroCoveredB (MagDiffVector& vec,
                   std::array<std::unique_ptr<iMultiFab>,3> const& eb_update_B)
{
    if (!EB::enabled()) { return; }
    auto& f = vec.fields();
    for (int idim = 0; idim < 3; ++idim) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(f[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto arr = f[idim].array(mfi);
            auto const mask_arr = eb_update_B[idim]->const_array(mfi);
            ParallelFor(mfi.tilebox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (mask_arr(i, j, k) == 0) { arr(i, j, k) = 0.0_rt; }
                });
        }
    }
}

class VariableCoeffMagDiffusionOp
{
public:
    using RT = Real;

    VariableCoeffMagDiffusionOp (
        ablastr::fields::VectorField const& Bfield,
        ablastr::fields::VectorField const& eta,
        Real theta_dt, int lev)
        : m_theta_dt(theta_dt), m_lev(lev),
          m_source{Bfield[0], Bfield[1], Bfield[2]},
          m_eta{eta[0], eta[1], eta[2]},
          m_geom(WarpX::GetInstance().Geom(lev))
    {
#if defined(WARPX_DIM_RZ)
        // Lower radial face (r=0) must be None (axis; ApplyFieldBoundaryOnAxis).
        // Upper radial face: None, PEC, or PEC_Insulator (Dirichlet tangential B).
        // Each axial (z) face: Periodic, PEC, or PEC_Insulator.
        // Ghosts are filled through ApplyBfieldBoundary.
        auto const fb_is_radial_face = [] (FieldBoundaryType fb) {
            return fb == FieldBoundaryType::None ||
                   fb == FieldBoundaryType::PEC ||
                   fb == FieldBoundaryType::PEC_Insulator;
        };
        auto const fb_is_axial_face = [] (FieldBoundaryType fb) {
            // None is not allowed on z: edge ghosts would be left undefined.
            return fb == FieldBoundaryType::Periodic ||
                   fb == FieldBoundaryType::PEC ||
                   fb == FieldBoundaryType::PEC_Insulator;
        };
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::field_boundary_lo[0] == FieldBoundaryType::None,
            "RZ matrix-free hybrid magnetic diffusion requires the lower "
            "radial boundary to be None (r=0 axis)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            fb_is_radial_face(WarpX::field_boundary_hi[0]),
            "RZ matrix-free hybrid magnetic diffusion supports None, PEC, or "
            "PEC_Insulator at the upper radial boundary");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            fb_is_axial_face(WarpX::field_boundary_lo[1]) &&
            fb_is_axial_face(WarpX::field_boundary_hi[1]),
            "RZ matrix-free hybrid magnetic diffusion supports Periodic, PEC, "
            "or PEC_Insulator at each axial (z) boundary (None is not "
            "well-posed for the z edge ghosts)");
#elif defined(WARPX_DIM_RCYLINDER)
        // RCYLINDER reuses the cylindrical hybrid-PIC curl and axis handling.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::field_boundary_lo[0] == FieldBoundaryType::None,
            "RCYLINDER matrix-free hybrid magnetic diffusion requires the "
            "lower radial boundary to be None (r=0 axis; "
            "ApplyFieldBoundaryOnAxis)");
        auto const fb_is_outer_radial = [] (FieldBoundaryType fb) {
            return fb == FieldBoundaryType::None ||
                   fb == FieldBoundaryType::PEC ||
                   fb == FieldBoundaryType::PEC_Insulator;
        };
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            fb_is_outer_radial(WarpX::field_boundary_hi[0]),
            "RCYLINDER matrix-free hybrid magnetic diffusion supports None, "
            "PEC, or PEC_Insulator at the outer radial boundary.");
#elif defined(WARPX_DIM_RSPHERE)
        WARPX_ABORT_WITH_MESSAGE(
            "Matrix-free hybrid magnetic diffusion is not yet supported in "
            "RSPHERE geometry");
#else
        auto const fb_is_supported = [] (FieldBoundaryType fb) {
            return fb == FieldBoundaryType::Periodic ||
                   fb == FieldBoundaryType::PEC ||
                   fb == FieldBoundaryType::PEC_Insulator;
        };
        bool cartesian_bcs_supported = true;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            cartesian_bcs_supported = cartesian_bcs_supported &&
                fb_is_supported(WarpX::field_boundary_lo[idim]) &&
                fb_is_supported(WarpX::field_boundary_hi[idim]);
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            cartesian_bcs_supported,
            "Cartesian matrix-free hybrid magnetic diffusion supports Periodic, "
            "PEC, or PEC_Insulator field boundaries");
#endif
        // Covered E faces carry eta=0 and covered B rows are identities. The
        // operator and preconditioners use the same stair-case masks.

        auto& warpx = WarpX::GetInstance();
        m_fdtd = warpx.get_pointer_fdtd_solver_fp(lev);
        m_eb_update_E = &warpx.GetEBUpdateEFlag()[lev];
        m_eb_update_B = &warpx.GetEBUpdateBFlag()[lev];

        // Index types for interpolating E/J-centered eta onto B faces for the PC.
        amrex::GpuArray<int, 3> eta_stag[3];
        amrex::GpuArray<int, 3> B_stag[3];
        for (int idim = 0; idim < 3; ++idim) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                Bfield[idim]->nComp() == 1 && eta[idim]->nComp() == 1,
                "Variable-coefficient hybrid magnetic diffusion supports one "
                "field component per Yee direction");
            m_Bwork[idim].define(Bfield[idim]->boxArray(), Bfield[idim]->DistributionMap(),
                                 1, Bfield[idim]->nGrowVect());
            m_Jwork[idim].define(eta[idim]->boxArray(), eta[idim]->DistributionMap(),
                                 1, eta[idim]->nGrowVect());
            m_etaJ[idim].define(eta[idim]->boxArray(), eta[idim]->DistributionMap(),
                                1, eta[idim]->nGrowVect());
            m_curl_etaJ[idim].define(Bfield[idim]->boxArray(), Bfield[idim]->DistributionMap(),
                                     1, Bfield[idim]->nGrowVect());
            // Diagonal eta estimate lives on B staggering (same BA/DM as LHS).
            m_eta_pc[idim].define(Bfield[idim]->boxArray(), Bfield[idim]->DistributionMap(),
                                  1, 0);
            auto const biv = Bfield[idim]->ixType().toIntVect();
            auto const eiv = eta[idim]->ixType().toIntVect();
            // GpuArray is always 3-wide; pad unused dims nodal (matches hybrid IndexType).
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                B_stag[idim][d] = biv[d];
                eta_stag[idim][d] = eiv[d];
            }
            for (int d = AMREX_SPACEDIM; d < 3; ++d) {
                B_stag[idim][d] = 1;
                eta_stag[idim][d] = 1;
            }
        }

        // Average the three E/J-centered eta components onto each B face.
        amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};
        for (int idim = 0; idim < 3; ++idim) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(m_eta_pc[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                SampleEtaOntoBFace const kernel{
                    m_eta_pc[idim].array(mfi),
                    m_eta[0]->const_array(mfi),
                    m_eta[1]->const_array(mfi),
                    m_eta[2]->const_array(mfi),
                    eta_stag[0], eta_stag[1], eta_stag[2],
                    B_stag[idim], coarsen};
                ParallelFor(mfi.tilebox(), kernel);
            }
        }
    }

    [[nodiscard]] MagDiffVector makeVecRHS () const
    {
        MagDiffVector result;
        result.Define({m_source[0], m_source[1], m_source[2]});
        return result;
    }

    [[nodiscard]] MagDiffVector makeVecLHS () const
    {
        return makeVecRHS();
    }

    void assign (MagDiffVector& destination, MagDiffVector const& source)
    {
        destination.Copy(source);
    }

    void setToZero (MagDiffVector& vector)
    {
        vector.setVal(0.0_rt);
    }

    void increment (MagDiffVector& destination, MagDiffVector const& source, Real scale)
    {
        destination.increment(source, scale);
    }

    void scale (MagDiffVector& vector, Real scale_factor)
    {
        vector.scale(scale_factor);
    }

    void linComb (MagDiffVector& destination, Real a, MagDiffVector const& x,
                  Real b, MagDiffVector const& y)
    {
        destination.linComb(a, x, b, y);
    }

    [[nodiscard]] Real dotProduct (MagDiffVector const& x, MagDiffVector const& y) const
    {
        return x.dotProduct(y);
    }

    [[nodiscard]] Real norm2 (MagDiffVector const& vector) const
    {
        return std::sqrt(vector.dotProduct(vector));
    }

    /**
     * \brief Scaled Jacobi PC for matrix-free curl(eta curl) GMRES.
     *
     * Approximates diag(I + theta_dt * (eta/mu0) * Laplacian) using eta
     * sampled onto B staggering (m_eta_pc). Uniform eta reduces to a pure scale.
     *
     * Critical: do NOT read E/J-centered eta MultiFabs with B (i,j,k). That
     * stagger mismatch is out-of-bounds and was the reason variable-eta GMRES
     * residual went NaN while constant eta looked "stable" (any in-fab index
     * still returned the same constant).
     *
     * Do not call ApplyBfieldBoundary inside the PC (would make it affine
     * under a Dirichlet B_t feed).
     */
    void precond (MagDiffVector& destination, MagDiffVector const& source)
    {
        Geometry const& geom = WarpX::GetInstance().Geom(m_lev);
        auto const * const dx = geom.CellSize();

        // Laplacian-diagonal scale only in the active dimensions.
#if defined(WARPX_DIM_3D)
        Real const dx_inv2 = 1.0_rt / (dx[0]*dx[0]);
        Real const dy_inv2 = 1.0_rt / (dx[1]*dx[1]);
        Real const dz_inv2 = 1.0_rt / (dx[2]*dx[2]);
        Real const diag_factor = 2.0_rt * (dx_inv2 + dy_inv2 + dz_inv2);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        Real const dx_inv2 = 1.0_rt / (dx[0]*dx[0]);
        Real const dz_inv2 = 1.0_rt / (dx[1]*dx[1]);
        Real const diag_factor = 2.0_rt * (dx_inv2 + dz_inv2);
#elif defined(WARPX_DIM_1D_Z)
        Real const dz_inv2 = 1.0_rt / (dx[0]*dx[0]);
        Real const diag_factor = 2.0_rt * dz_inv2;
#elif defined(WARPX_DIM_RCYLINDER)
        // 1D r-only (axisymmetric). diag_factor = 2/dr^2 is the off-axis
        // Laplacian diagonal used as the denom reference scale (uniform eta ->
        // identity after PC). The per-DOF metric-aware diagonal (radial 1/r and
        // the Bt -1/r^2 term) is computed inside MagDiffJacobiPrecond from the
        // geometry passed below, mirroring the assembled PETSc diagonals.
        Real const dr_inv2 = 1.0_rt / (dx[0]*dx[0]);
        Real const diag_factor = 2.0_rt * dr_inv2;
#else
        // RSPHERE (aborts in the ctor) or any other dim: leave the PC diagonal
        // unscaled (diag_factor = 0) so the Jacobi PC reduces to identity rather
        // than reading a bogus scale.
        Real const diag_factor = 0.0_rt;
#endif

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
        // Cylindrical metric for the per-DOF Jacobi diagonal: MagDiffJacobiPrecond
        // computes the r-aware Laplacian diagonal from these; diag_factor above is
        // the off-axis reference used only for the denom scale (uniform eta ->
        // identity after PC, except a negligible scaling on the one near-axis Bt
        // cell whose -1/r^2 is skipped).
        Real const rmin = geom.ProbLo(0);
        Real const dr = dx[0];
#  if defined(WARPX_DIM_RZ)
        Real const kern_dr_inv2 = dx_inv2;
        Real const kern_dz_inv2 = dz_inv2;
#  else // WARPX_DIM_RCYLINDER
        Real const kern_dr_inv2 = dr_inv2;
        Real const kern_dz_inv2 = 0.0_rt;  // no z dimension
#  endif
#endif

        auto& dest_fields = destination.fields();
        auto const& src_fields = source.fields();

        // Reference scale from max eta so stiff vacuum modes are O(1) and
        // uniform eta reduces to the identity map after PC.
        Real max_eta = 0.0_rt;
        for (int idim = 0; idim < 3; ++idim) {
            max_eta = std::max(max_eta, m_eta_pc[idim].max(0));
        }
        // 1e-99 underflows to 0 in single precision; keep a representable floor.
        max_eta = std::max(max_eta, Real(1.e-3) * std::numeric_limits<Real>::min());
        Real const chi_max = max_eta / PhysConst::mu0;
        Real const denom = std::max(
            1.0_rt + m_theta_dt * chi_max * diag_factor,
            Real(1.e-3) * std::numeric_limits<Real>::min());

        bool const eb_on = EB::enabled();
        for (int idim = 0; idim < 3; ++idim) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
            // r-staggering per component: 1 = NODE in r (Br), 0 = CELL in r
            // (Bt/Bz); is_bt -> subtract 1/r^2 (Bt only). See ComputeCurlA.cpp.
            int const ishift_r = dest_fields[idim].ixType().toIntVect()[0];

#endif
            for (MFIter mfi(dest_fields[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                // Covered B faces are identity in the operator; the PC mirrors
                // that (see MagDiffJacobiPrecond). Empty Array4 when EB is off.
                Array4<int const> mask_arr;
                if (eb_on) {
                    mask_arr = (*m_eb_update_B)[idim]->const_array(mfi);
                }
                MagDiffJacobiPrecond const kernel{
                    dest_fields[idim].array(mfi),
                    src_fields[idim].const_array(mfi),
                    m_eta_pc[idim].const_array(mfi),
                    mask_arr,
                    m_theta_dt, diag_factor, denom
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                    , rmin, dr, kern_dr_inv2, kern_dz_inv2, ishift_r, idim
#endif
                };
                ParallelFor(mfi.tilebox(), kernel);
            }
        }
    }

    void apply (MagDiffVector& output, MagDiffVector const& input)
    {
        // Remove the affine feed offset so Krylov solvers see a linear operator.
        computeAFull(output, input, m_theta_dt);
        if (m_has_feed) {
            output.increment(m_feed_offset, -1.0_rt);
        }
    }

    // Compute A_full(x) = x + alpha_dt curl(eta/mu0 curl x). Physical boundary
    // handling is applied by staging through the registered B field.
    void computeAFull (MagDiffVector& output, MagDiffVector const& input, Real alpha_dt)
    {
        auto const& input_fields = input.fields();
        auto& warpx = WarpX::GetInstance();
        // Stage through the registered field to apply physical boundaries.
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Copy(*m_source[idim], input_fields[idim], 0, 0, 1, 0);
        }
        warpx.ApplyBfieldBoundary(
            m_lev, PatchType::fine, SubcyclingHalf::None, warpx.gett_new(m_lev));
        warpx.FillBoundaryB(m_lev, m_source[0]->nGrowVect(), true);

        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Copy(m_Bwork[idim], *m_source[idim], 0, 0, 1,
                           m_Bwork[idim].nGrowVect());
        }


        MultiFab * const Bwork_ptr = m_Bwork.data();
        MultiFab * const Jwork_ptr = m_Jwork.data();
        ablastr::fields::VectorField const Bwork = {
            Bwork_ptr, Bwork_ptr + 1, Bwork_ptr + 2};
        ablastr::fields::VectorField Jwork = {
            Jwork_ptr, Jwork_ptr + 1, Jwork_ptr + 2};
        // Curl kernels do not write every exterior ghost.
        for (int idim = 0; idim < 3; ++idim) {
            m_Jwork[idim].setVal(0.0_rt);
        }
        m_fdtd->CalculateCurrentAmpere(Jwork, Bwork, *m_eb_update_E, m_lev);


        for (int idim = 0; idim < 3; ++idim) {
            // Cover valid and ghost cells needed by curl(eta J).
            for (MFIter mfi(m_etaJ[idim]); mfi.isValid(); ++mfi) {
                FormEtaTimesJ const kernel{
                    m_etaJ[idim].array(mfi),
                    m_eta[idim]->const_array(mfi),
                    m_Jwork[idim].const_array(mfi)};
                ParallelFor(mfi.fabbox(), kernel);
            }
            m_etaJ[idim].FillBoundaryAndSync(m_geom.periodicity());
        }

        MultiFab * const etaJ_ptr = m_etaJ.data();
        MultiFab * const curl_etaJ_ptr = m_curl_etaJ.data();
        ablastr::fields::VectorField const etaJ = {
            etaJ_ptr, etaJ_ptr + 1, etaJ_ptr + 2};
        ablastr::fields::VectorField curl_etaJ = {
            curl_etaJ_ptr, curl_etaJ_ptr + 1, curl_etaJ_ptr + 2};
        // Ampere: J = curl(B)/mu0. ComputeCurlA: out = curl(A) (no 1/mu0).
        // With A = eta*J, curl(A) = curl((eta/mu0) curl B) as required.
        m_fdtd->ComputeCurlA(curl_etaJ, etaJ, *m_eb_update_B, m_lev);

        auto& output_fields = output.fields();
        for (int idim = 0; idim < 3; ++idim) {
            m_curl_etaJ[idim].OverrideSync(m_geom.periodicity());
            MultiFab::Copy(output_fields[idim], input_fields[idim], 0, 0, 1, 0);
            MultiFab::Saxpy(output_fields[idim], alpha_dt, m_curl_etaJ[idim],
                            0, 0, 1, 0);
        }
    }

    // Precompute c=A_full(0) for an inhomogeneous pec_insulator feed.
    void prepareFeed ()
    {
        m_has_feed = false;
        auto& warpx = WarpX::GetInstance();
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            for (int iside = 0; iside < 2; ++iside) {
                const FieldBoundaryType fb = (iside == 0)
                    ? WarpX::field_boundary_lo[idim]
                    : WarpX::field_boundary_hi[idim];
                if (fb != FieldBoundaryType::PEC_Insulator) { continue; }
                for (int ifield = 0; ifield < 3; ++ifield) {
                    if (warpx.GetPECInsulator_IsBSet(idim, iside, ifield)) {
                        m_has_feed = true;
                    }
                }
            }
        }
        if (!m_has_feed) { return; }

        MagDiffVector zero;
        zero.Define({m_source[0], m_source[1], m_source[2]});
        zero.setVal(0.0_rt);
        m_feed_offset.Define({m_source[0], m_source[1], m_source[2]});
        // computeAFull writes only interior (nghost=0), so zero ghosts first to
        // keep the later output -= c (which spans ghosts) well-defined.
        m_feed_offset.setVal(0.0_rt);
        computeAFull(m_feed_offset, zero, m_theta_dt);
    }

    [[nodiscard]] bool hasFeed () const { return m_has_feed; }
    [[nodiscard]] MagDiffVector const& feedOffset () const { return m_feed_offset; }

    // Accessors used by the optional PETSc KSP path. The homogeneous apply() is
    // reused directly. The assembled curl-curl Pmat uses E/J-face eta (m_eta)
    // in every supported geometry to match the matvec face selection. apply()
    // is non-const because it stages through internal work MultiFabs.
    void applyPetsc (MagDiffVector& output, MagDiffVector const& input) { apply(output, input); }
    [[nodiscard]] Real thetaDt () const { return m_theta_dt; }
    [[nodiscard]] Geometry const& geom () const { return m_geom; }
    [[nodiscard]] Array<MultiFab const*,3> const& etaEdge () const { return m_eta; }

private:
    Real m_theta_dt;
    int m_lev;
    Array<MultiFab*,3> m_source;
    Array<MultiFab const*,3> m_eta;
    Geometry m_geom;
    FiniteDifferenceSolver* m_fdtd = nullptr;
    std::array<std::unique_ptr<iMultiFab>,3> const* m_eb_update_E = nullptr;
    std::array<std::unique_ptr<iMultiFab>,3> const* m_eb_update_B = nullptr;
    Array<MultiFab,3> m_Bwork;
    Array<MultiFab,3> m_Jwork;
    Array<MultiFab,3> m_etaJ;
    Array<MultiFab,3> m_curl_etaJ;
    // B-centered eta for Jacobi PC (same BA as each B component). Built once
    // from E/J-centered frozen eta via Interp — never use E indices on B.
    Array<MultiFab,3> m_eta_pc;
    // Nonzero when a pec_insulator Dirichlet tangential-B feed is active.
    // m_feed_offset holds c = A_full(0); apply() subtracts c so GMRES sees a
    // linear operator; AdvanceVariable subtracts c from the RHS.
    bool m_has_feed = false;
    MagDiffVector m_feed_offset;
};

// PETSc glue between flat local vectors and the matrix-free field operator.
#ifdef AMREX_USE_PETSC

// View of an Array<MultiFab,3> as an ablastr VectorField (three MultiFab*), used
// to Define MagDiffVector work buffers from a prototype's fields.
inline ablastr::fields::VectorField
makeVectorFieldView (Array<MultiFab,3>& a)
{
    return {&a[0], &a[1], &a[2]};
}

// Per-call context for the PETSc callbacks: holds the operator and work buffers.
// Created/filled by AdvanceVariable; passed to magdiff_petsc_make as opctx.
struct PetscOpCtx
{
    VariableCoeffMagDiffusionOp* linop = nullptr;
    MagDiffVector U;       // matvec input buffer
    MagDiffVector F;       // matvec output buffer
#ifdef AMREX_USE_GPU
    // Cached pinned host staging for PETSc flat <-> device MultiFab. Allocated
    // once per AdvanceVariable (not per matvec): the naive GPU-safe path that
    // new'd MultiFab every scatter/gather cost ~25-30% wall on CPU/CUDA.
    Array<MultiFab,3> host_U;
    Array<MultiFab,3> host_F;
    bool host_bufs_defined = false;

    void ensureHostBufs (Array<MultiFab,3> const& proto)
    {
        if (host_bufs_defined) { return; }
        MFInfo const info = MFInfo().SetArena(The_Pinned_Arena());
        for (int idim = 0; idim < 3; ++idim) {
            host_U[idim].define(proto[idim].boxArray(),
                                proto[idim].DistributionMap(), 1, 0, info);
            host_F[idim].define(proto[idim].boxArray(),
                                proto[idim].DistributionMap(), 1, 0, info);
        }
        host_bufs_defined = true;
    }
#endif
};

// PETSc Vecs are host-resident, so GPU builds stage through pinned MultiFabs.
// gindex is on The_Pinned_Arena (host-readable). gix < 0 => covered/exterior.
void
petscScatter (MagDiffVector& dst, Real const* flat,
              Array<iMultiFab const*,3> const& gindex, amrex::Long rstart
#ifdef AMREX_USE_GPU
              , Array<MultiFab,3>& host_bufs
#endif
              )
{
    auto& f = dst.fields();
    for (int idim = 0; idim < 3; ++idim) {
#ifdef AMREX_USE_GPU
        MultiFab& host = host_bufs[idim];
        for (MFIter mfi(host); mfi.isValid(); ++mfi) {
            auto arr = host.array(mfi);
#else
        // Only write active gix >= 0; covered DOFs are omitted from the PETSc
        // system (gather skips them). Stale covered values never re-enter KSP.
        for (MFIter mfi(f[idim]); mfi.isValid(); ++mfi) {
            auto arr = f[idim].array(mfi);
#endif
            auto const& gix = gindex[idim]->const_array(mfi);
            Box const& tb = mfi.tilebox();
            LoopOnCpu(lbound(tb), ubound(tb),
                [&] (int i, int j, int k) {
                    auto const gixv = gix(i, j, k);
                    if (gixv < 0) { return; }  // covered B DOF, skip
                    auto const lid = static_cast<amrex::Long>(gixv) - rstart;
                    arr(i, j, k) = flat[lid];
                });
        }
#ifdef AMREX_USE_GPU
        MultiFab::Copy(f[idim], host, 0, 0, 1, 0);
#endif
    }
}

// MagDiffVector interior (nghost=0 cells only) -> PETSc flat local array.
void
petscGather (Real* flat, MagDiffVector const& src,
             Array<iMultiFab const*,3> const& gindex, amrex::Long rstart
#ifdef AMREX_USE_GPU
             , Array<MultiFab,3>& host_bufs
#endif
             )
{
    auto const& f = src.fields();
    for (int idim = 0; idim < 3; ++idim) {
#ifdef AMREX_USE_GPU
        MultiFab& host = host_bufs[idim];
        MultiFab::Copy(host, f[idim], 0, 0, 1, 0);
        for (MFIter mfi(host); mfi.isValid(); ++mfi) {
            auto const& arrf = host.const_array(mfi);
#else
        for (MFIter mfi(f[idim]); mfi.isValid(); ++mfi) {
            auto const& arrf = f[idim].const_array(mfi);
#endif
            auto const& gix = gindex[idim]->const_array(mfi);
            Box const& tb = mfi.tilebox();
            LoopOnCpu(lbound(tb), ubound(tb),
                [&] (int i, int j, int k) {
                    auto const gixv = gix(i, j, k);
                    if (gixv < 0) { return; }  // covered B DOF, skip
                    auto const lid = static_cast<amrex::Long>(gixv) - rstart;
                    flat[lid] = arrf(i, j, k);
                });
        }
    }
}

// Matvec callback: y = A_lin(x) = linop.apply (homogeneous; feed offset is on
// the RHS, handled by AdvanceVariable).
void
petscMatvec (void* ctx, Real const* x, Real* y,
             Array<iMultiFab const*,3> const& gindex, amrex::Long rstart)
{
    auto* c = static_cast<PetscOpCtx*>(ctx);
#ifdef AMREX_USE_GPU
    petscScatter(c->U, x, gindex, rstart, c->host_U);
    c->linop->applyPetsc(c->F, c->U);
    petscGather(y, c->F, gindex, rstart, c->host_F);
#else
    petscScatter(c->U, x, gindex, rstart);
    c->linop->applyPetsc(c->F, c->U);
    petscGather(y, c->F, gindex, rstart);
#endif
}

#endif // AMREX_USE_PETSC

} // namespace

struct HybridMagDiffusionWorkspace
{
    MagDiffVector solution;
    MagDiffVector rhs;
    MagDiffVector Ae;
#ifdef AMREX_USE_PETSC
    PetscOpCtx opctx;
#endif
    BoxArray box_array;
    DistributionMapping distribution_map;
    bool is_defined = false;
};

HybridMagDiffusion::HybridMagDiffusion () = default;
HybridMagDiffusion::~HybridMagDiffusion () = default;

void
HybridMagDiffusion::Advance (
    ablastr::fields::VectorField const& Bfield,
    Real eta_SI,
    Real dt,
    int lev) const
{
    ABLASTR_PROFILE("HybridMagDiffusion::Advance()");

    if (!m_enabled) { return; }
    if (dt <= 0.0_rt) { return; }
    if (eta_SI <= 0.0_rt) { return; }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        lev == 0,
        "HybridMagDiffusion only supports single-level hybrid runs");

    // Constant-η convenience entry: flat η MultiFab → AdvanceVariable (the
    // only mag-diff discrete path).
    const Real chi = eta_SI / PhysConst::mu0;
    if (m_verbose > 0) {
        amrex::Print() << "HybridMagDiffusion::Advance: eta=" << eta_SI
                       << " chi=" << chi
                       << " dt=" << dt
                       << " (matrix-free flat-eta)\n";
    }

    auto& warpx = WarpX::GetInstance();
    ablastr::fields::VectorField const current_layout =
        warpx.m_fields.get_alldirs(
            warpx::fields::FieldType::hybrid_current_fp_plasma, lev);
    Array<MultiFab,3> eta_storage;
    for (int idim = 0; idim < 3; ++idim) {
        eta_storage[idim].define(
            current_layout[idim]->boxArray(), current_layout[idim]->DistributionMap(),
            1, current_layout[idim]->nGrowVect());
        eta_storage[idim].setVal(eta_SI);
    }
    // Zero η on covered E faces (eb_update_E == 0): no diffusion into the
    // solid. Mirrors BuildMagDiffResistivity; matvec already zeros Jwork.
    // No-op when EB is off.
    if (EB::enabled()) {
        auto const& eb_update_E = warpx.GetEBUpdateEFlag()[lev];
        amrex::Periodicity const& eb_period = warpx.Geom(lev).periodicity();
        for (int idim = 0; idim < 3; ++idim) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(eta_storage[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto arr = eta_storage[idim].array(mfi);
                auto const mask_arr = eb_update_E[idim]->const_array(mfi);
                ParallelFor(mfi.tilebox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        if (mask_arr(i, j, k) == 0) { arr(i, j, k) = 0.0_rt; }
                    });
            }
            eta_storage[idim].FillBoundaryAndSync(eb_period);
        }
    }
    MultiFab * const eta_ptr = eta_storage.data();
    ablastr::fields::VectorField const eta_field = {
        eta_ptr, eta_ptr + 1, eta_ptr + 2};
    AdvanceVariable(Bfield, eta_field, dt, lev);
}

void
HybridMagDiffusion::AdvanceVariable (
    ablastr::fields::VectorField const& Bfield,
    ablastr::fields::VectorField const& eta_SI,
    Real dt,
    int lev) const
{
    ABLASTR_PROFILE("HybridMagDiffusion::AdvanceVariable()");

    if (!m_enabled || dt <= 0.0_rt) { return; }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        lev == 0,
        "HybridMagDiffusion only supports single-level hybrid runs");

    // theta in (0,1] is enforced in ReadParameters: theta=1 is backward Euler,
    // theta in (0,1) is Crank-Nicolson / general theta-method. The operator
    // A_lin = I + theta*dt*K_hom (homogeneous; feed carried on the RHS) is the
    // same for every theta; only the RHS gains the explicit (1-theta) term.

    VariableCoeffMagDiffusionOp linop(Bfield, eta_SI, m_theta * dt, lev);

    if (!m_workspace) {
        m_workspace = std::make_unique<HybridMagDiffusionWorkspace>();
    }
    auto& workspace = *m_workspace;
    auto& solution = workspace.solution;
    auto& rhs = workspace.rhs;
    auto& Ae = workspace.Ae;
#ifdef AMREX_USE_PETSC
    auto& opctx = workspace.opctx;
#endif

    bool const layout_changed = !workspace.is_defined
        || !BoxArray::SameRefs(workspace.box_array, Bfield[0]->boxArray())
        || !DistributionMapping::SameRefs(
            workspace.distribution_map, Bfield[0]->DistributionMap());
    if (layout_changed) {
#ifdef AMREX_USE_PETSC
        m_petsc_solver.reset();
#endif
        solution.Define(Bfield);
        rhs.Define(Bfield);
        Ae.Define(Bfield);
#ifdef AMREX_USE_PETSC
        opctx = PetscOpCtx{};
        opctx.U.Define(makeVectorFieldView(solution.fields()));
        opctx.F.Define(makeVectorFieldView(solution.fields()));
#ifdef AMREX_USE_GPU
        opctx.ensureHostBufs(solution.fields());
#endif
#endif
        workspace.box_array = Bfield[0]->boxArray();
        workspace.distribution_map = Bfield[0]->DistributionMap();
        workspace.is_defined = true;
    }
#ifdef AMREX_USE_PETSC
    opctx.linop = &linop;
#endif

    // Capture B^n before prepareFeed: the operator stages its Krylov vectors
    // (and the zero probe for the feed offset) through m_source, which is the
    // registered B field (= Bfield). The true B^n lives in solution/rhs.
    solution.CopyFrom(Bfield);
    rhs.CopyFrom(Bfield);

    // Covered B rows are identities with zero initial values and RHS entries.
    auto const& eb_update_B = WarpX::GetInstance().GetEBUpdateBFlag()[lev];
    zeroCoveredB(solution, eb_update_B);
    zeroCoveredB(rhs, eb_update_B);

    // prepareFeed stages a zero probe through m_source, so B^n must be saved first.
    linop.prepareFeed();
    bool const has_feed = linop.hasFeed();

    // Theta RHS: 2*B^n - A_e(B^n) - c, where
    // A_e=I+(1-theta)dt*K_full and c=theta*dt*K_feed. This injects the feed once.
    if (m_theta < 1.0_rt) {
        const Real alpha_e = (1.0_rt - m_theta) * dt;
        linop.computeAFull(Ae, solution, alpha_e);
        rhs.linComb(2.0_rt, solution, -1.0_rt, Ae);
    }
    if (has_feed) {
        rhs.increment(linop.feedOffset(), -1.0_rt);
    }

    if (m_linear_solver == MagDiffLinearSolver::petsc) {
#ifdef AMREX_USE_PETSC
        auto const& eta_edge = linop.etaEdge();
        amrex::Array<MultiFab const*,3> const B_proto{
            &solution.fields()[0], &solution.fields()[1], &solution.fields()[2]};

        // Pass the EB B-field mask (if any) so covered DOFs are skipped from
        // the PETSc system. Must be nullptr when EB is off (the mask MultiFabs
        // are undefined then), matching the default parameter in the header.
        amrex::Array<iMultiFab const*,3> const eb_update_B_ptrs{
            eb_update_B[0].get(), eb_update_B[1].get(), eb_update_B[2].get()};
        bool const petsc_eb_on = EB::enabled();

        if (!m_petsc_solver) {
            m_petsc_solver.reset(magdiff_petsc_make(
                B_proto, eta_edge, linop.geom(), linop.thetaDt(),
                PhysConst::mu0, m_rtol, m_atol, m_max_iter, m_verbose,
                m_petsc_options,
                &petscMatvec, &opctx,
                petsc_eb_on ? &eb_update_B_ptrs : nullptr));
        } else {
            magdiff_petsc_update(m_petsc_solver.get(), eta_edge, linop.thetaDt(), &opctx);
        }

        const amrex::Long n_local = magdiff_petsc_nlocal(m_petsc_solver.get());
        const amrex::Long rstart = magdiff_petsc_rstart(m_petsc_solver.get());
        auto const gindex = magdiff_petsc_gindex(m_petsc_solver.get());

        // PETSc uses a zero initial guess; gather only the DOF-mapped RHS.
        std::vector<Real> sol_flat(static_cast<std::size_t>(n_local), Real(0.0));
        std::vector<Real> rhs_flat(static_cast<std::size_t>(n_local), Real(0.0));
#ifdef AMREX_USE_GPU
        petscGather(rhs_flat.data(), rhs, gindex, rstart, opctx.host_F);
#else
        petscGather(rhs_flat.data(), rhs, gindex, rstart);
#endif

        Real rnorm = 0.0_rt;
        const int reason = magdiff_petsc_solve(
            m_petsc_solver.get(), rhs_flat.data(), sol_flat.data(), rnorm);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            reason > 0 && std::isfinite(rnorm),
            "Hybrid magnetic-diffusion PETSc solve did not converge");

#ifdef AMREX_USE_GPU
        petscScatter(solution, sol_flat.data(), gindex, rstart, opctx.host_U);
#else
        petscScatter(solution, sol_flat.data(), gindex, rstart);
#endif
#else
        WARPX_ABORT_WITH_MESSAGE(
            "hybrid_pic_model.mag_diff_linear_solver = petsc requires building "
            "WarpX with PETSc (-DWarpX_PETSC=ON, AMREX_USE_PETSC).");
#endif
    } else {
        amrex::GMRES<MagDiffVector, VariableCoeffMagDiffusionOp> solver;
        solver.define(linop);
        solver.setMaxIters(m_max_iter);
        solver.setRestartLength(std::min(m_max_iter, 50));
        solver.setVerbose(m_verbose);
        solver.solve(solution, rhs, m_rtol, m_atol);

        const Real rnorm = solver.getResidualNorm();
        if (solver.getStatus() != 0 || !std::isfinite(rnorm)) {
            std::ostringstream msg;
            msg << "Hybrid magnetic-diffusion GMRES solve failed: status="
                << solver.getStatus() << ", iterations=" << solver.getNumIters()
                << ", residual=" << rnorm;
            WARPX_ABORT_WITH_MESSAGE(msg.str());
        }
        if (m_verbose > 0) {
            amrex::Print() << "HybridMagDiffusion matrix-free GMRES iterations="
                           << solver.getNumIters()
                           << " residual=" << rnorm << "\n";
        }
    }

    auto const& solution_fields = solution.fields();
    // Keep covered degrees of freedom exactly zero before installation.
    zeroCoveredB(solution, eb_update_B);
    auto& warpx = WarpX::GetInstance();
    for (int idim = 0; idim < 3; ++idim) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            solution_fields[idim].is_finite(),
            "Hybrid magnetic-diffusion solve returned a non-finite field");
        MultiFab::Copy(*Bfield[idim], solution_fields[idim], 0, 0, 1, 0);
    }
    // Re-apply physical boundaries through the same path used by the matvec,
    // then exchange periodic and inter-box ghosts.
    warpx.ApplyBfieldBoundary(
        lev, PatchType::fine, SubcyclingHalf::None, warpx.gett_new(lev));
    warpx.FillBoundaryB(lev, Bfield[0]->nGrowVect(), true);
}
