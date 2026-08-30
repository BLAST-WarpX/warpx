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
#include <ablastr/utils/Communication.H>

#include <AMReX_Array.H>
#include <AMReX_Arena.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GMRES.H>
#include <AMReX_Loop.H>
#include <AMReX_MLCurlCurl.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>

#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
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
    pp.query("mag_diff_petsc_pc", m_petsc_pc);

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
        m_rtol > 0.0_rt && m_atol >= 0.0_rt,
        "hybrid_pic_model.mag_diff_rtol/atol must be non-negative (rtol > 0)");

    // theta=1 is backward Euler. For theta in (0,1), the same linear system is
    // used with a modified RHS that includes an explicit curl-curl term.
    amrex::Print() << "HybridMagDiffusion: enabled"
                   << "  theta=" << m_theta
                   << "  rtol=" << m_rtol
                   << "  atol=" << m_atol
                   << "  eta_explicit_max=" << m_eta_explicit_max
                   << "  use_variable_eta=" << m_use_variable_eta
                   << "  linear_solver=" << amrex::getEnumNameString(m_linear_solver)
                   << "  petsc_pc=" << amrex::getEnumNameString(m_petsc_pc);
    if (m_has_constant_eta) {
        amrex::Print() << "  constant_eta=" << m_constant_eta << " Ohm m";
    }
    amrex::Print() << "\n";
}

void
HybridMagDiffusion::GetLinOpBCs (
    Array<LinOpBCType, AMREX_SPACEDIM>& lobc,
    Array<LinOpBCType, AMREX_SPACEDIM>& hibc)
{
    // Map WarpX field BCs to LinOp types (same idea as ProjectionDivCleaner).
    // Unmapped types default to Neumann. PEC_Insulator is treated as Dirichlet
    // for tangential B. On the matrix-free RZ path the feed is enforced via
    // ApplyBfieldBoundary and an affine RHS term (VariableCoeffMagDiffusionOp);
    // this map is used by the Cartesian MLCurlCurl path.
    const std::map<FieldBoundaryType, LinOpBCType> bcmap{
        {FieldBoundaryType::PEC, LinOpBCType::Dirichlet},
        {FieldBoundaryType::Neumann, LinOpBCType::Neumann},
        {FieldBoundaryType::PMC, LinOpBCType::Neumann},
        {FieldBoundaryType::Periodic, LinOpBCType::Periodic},
        {FieldBoundaryType::PEC_Insulator, LinOpBCType::Dirichlet},
        {FieldBoundaryType::None, LinOpBCType::Neumann}
    };

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        auto itlo = bcmap.find(WarpX::field_boundary_lo[idim]);
        auto ithi = bcmap.find(WarpX::field_boundary_hi[idim]);
        lobc[idim] = (itlo != bcmap.end()) ? itlo->second : LinOpBCType::Neumann;
        hibc[idim] = (ithi != bcmap.end()) ? ithi->second : LinOpBCType::Neumann;
    }
}

namespace {

// MLCurlCurl constant-eta path is Cartesian-only; RZ always uses AdvanceVariable.
#if !defined(WARPX_DIM_RZ)
/** Alias WarpX (Bx,By,Bz) MultiFabs into AMReX MLCurlCurl component order. */
Array<MultiFab,3>
MakeCurlCurlAliases (const Array<MultiFab,3>& mf)
{
#if defined(WARPX_DIM_1D_Z)
    // Missing dimensions are x,y in WarpX and y,z in AMReX
    return {
        MultiFab(mf[2], make_alias, 0, 1),
        MultiFab(mf[1], make_alias, 0, 1),
        MultiFab(mf[0], make_alias, 0, 1)
    };
#elif defined(WARPX_DIM_XZ)
    // Missing dimension is y in WarpX and z in AMReX
    return {
        MultiFab(mf[0], make_alias, 0, 1),
        MultiFab(mf[2], make_alias, 0, 1),
        MultiFab(mf[1], make_alias, 0, 1)
    };
#else
    // 3D and RCYLINDER / RSPHERE
    return {
        MultiFab(mf[0], make_alias, 0, 1),
        MultiFab(mf[1], make_alias, 0, 1),
        MultiFab(mf[2], make_alias, 0, 1)
    };
#endif
}
#endif // !WARPX_DIM_RZ

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
    Real diag_factor = 0.0_rt;
    Real denom = 1.0_rt;

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
        Real const diag = (1.0_rt + theta_dt * chi_val * diag_factor) / denom;
        Real const inv = Real(1.0) / std::max(diag, tiny);
        Real const out = src(i, j, k) * inv;
        // Device-safe finite check (avoid std::isfinite on CUDA device).
        dest(i, j, k) = (out == out) ? out : src(i, j, k);
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

    // Krylov vector algebra must use nghost=0 to match norm2/dotProduct.
    // Ghosts are filled inside apply()/precond when needed. Touching ghosts
    // with 1/tiny_residual (or Gram–Schmidt coeffs) overflows under PETSc
    // FPE traps (Azure WarpX_PETSC + WarpX_TEST_FPETRAP).
    static bool isFiniteReal (Real v)
    {
        return (v == v) &&
               v <=  std::numeric_limits<Real>::max() &&
               v >= -std::numeric_limits<Real>::max();
    }

    void increment (MagDiffVector const& source, Real scale_factor)
    {
        AMREX_ALWAYS_ASSERT(m_is_defined && source.m_is_defined);
        if (!isFiniteReal(scale_factor)) { return; }
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Saxpy(m_fields[idim], scale_factor, source.m_fields[idim],
                            0, 0, m_fields[idim].nComp(), 0);
        }
    }

    void scale (Real scale_factor)
    {
        AMREX_ALWAYS_ASSERT(m_is_defined);
        if (!isFiniteReal(scale_factor)) { return; }
        for (auto& field : m_fields) {
            field.mult(scale_factor, 0, field.nComp(), 0);
        }
    }

    void linComb (Real a, MagDiffVector const& x, Real b, MagDiffVector const& y)
    {
        AMREX_ALWAYS_ASSERT(m_is_defined && x.m_is_defined && y.m_is_defined);
        if (!isFiniteReal(a) || !isFiniteReal(b)) { return; }
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
        Real theta_dt, int lev,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& lobc,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& hibc)
        : m_theta_dt(theta_dt), m_lev(lev),
          m_source{Bfield[0], Bfield[1], Bfield[2]},
          m_eta{eta[0], eta[1], eta[2]},
          m_geom(WarpX::GetInstance().Geom(lev))
    {
        amrex::ignore_unused(lobc, hibc);
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
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        WARPX_ABORT_WITH_MESSAGE(
            "Matrix-free hybrid magnetic diffusion is not yet supported in "
            "RCYLINDER or RSPHERE geometries");
#else
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_geom.isAllPeriodic(),
            "Variable-coefficient hybrid magnetic diffusion currently requires "
            "periodic field boundaries");
#endif
        // EB is supported on the matrix-free path via the stair-case masks
        // m_eb_update_E/B (MarkUpdateCellsStairCase): covered E faces carry
        // η=0 (no diffusion into the solid) and covered B DOFs stay inert
        // (identity in apply/precond; zeroed in init-guess/RHS/copy-back). The
        // matvec is already EB-safe — computeAFull zeros m_Jwork before
        // CalculateCurrentAmpere (so J=0 on covered E faces) and ComputeCurlA
        // zeros covered B — so these masks are defensive + PC cleanliness +
        // PETSc-Pmat prep. See notes/2026-07-18_eb_parser_eta.md.

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

        // Sample resistivity onto each B face by averaging the three E/J-centered
        // eta components interpolated to that face. Never index E-staggered eta
        // MultiFabs with B indices (that OOB-read is why variable-eta GMRES
        // residual went NaN while constant eta looked "stable" — any in-fab
        // index still returned the same constant).
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
#else
        Real const diag_factor = 0.0_rt;
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
                    m_theta_dt, diag_factor, denom};
                ParallelFor(mfi.tilebox(), kernel);
            }
        }
    }

    void apply (MagDiffVector& output, MagDiffVector const& input)
    {
        // A_full(x) = (I + theta_dt * curl(eta/mu0 * curl(.))) x, with Dirichlet
        // feed ghosts set to g(t_new) by ApplyBfieldBoundary. With a nonzero
        // feed, A_full is affine: A_full(x) = A_lin(x) + c, where A_lin is the
        // homogeneous operator (zero feed ghost) and c = A_full(0). GMRES needs
        // a linear operator, so this returns A_lin(x) = A_full(x) - c; the RHS
        // carries +c (see AdvanceVariable). c is precomputed in prepareFeed.
        computeAFull(output, input, m_theta_dt);
        if (m_has_feed) {
            output.increment(m_feed_offset, -1.0_rt);
        }
    }

    // Compute A_full(x) = x + alpha_dt * curl(eta/mu0 curl x), staging through
    // the registered B field so ApplyBfieldBoundary imposes RZ axis, PEC walls,
    // and any pec_insulator Dirichlet B_t feed (ghost = g(t_new)). With a nonzero
    // feed, A_full is affine (A_full(x) = A_lin(x) + c); apply() subtracts c so
    // GMRES/PETSc see the homogeneous A_lin, and c is carried on the RHS.
    //
    // alpha_dt is the coefficient on curl(eta J): m_theta_dt (= theta*dt) for the
    // implicit operator A_lin and the feed offset c; (1-theta)*dt for the explicit
    // (1-theta) CN term A_e used to build the theta-method RHS (AdvanceVariable).
    // Staging, BC fill, J = curl(B)/mu0, eta*J, and curl(eta J) all use the frozen
    // eta MultiFabs and are independent of alpha_dt; only the final Saxpy scales
    // by alpha_dt.
    void computeAFull (MagDiffVector& output, MagDiffVector const& input, Real alpha_dt)
    {
        auto const& input_fields = input.fields();
#if defined(WARPX_DIM_RZ)
        auto& warpx = WarpX::GetInstance();
        // Stage valid cells into the registered B field so WarpX applies RZ
        // axis and field boundaries, then pull the filled ghost region into
        // work arrays. (Krylov MagDiffVectors may carry ghost storage, but
        // norms use nghost=0; BC fill goes through registered B.)
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
#else
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Copy(m_Bwork[idim], input_fields[idim], 0, 0, 1, 0);
            m_Bwork[idim].FillBoundaryAndSync(m_geom.periodicity());
        }
#endif


        MultiFab * const Bwork_ptr = m_Bwork.data();
        MultiFab * const Jwork_ptr = m_Jwork.data();
        ablastr::fields::VectorField const Bwork = {
            Bwork_ptr, Bwork_ptr + 1, Bwork_ptr + 2};
        ablastr::fields::VectorField Jwork = {
            Jwork_ptr, Jwork_ptr + 1, Jwork_ptr + 2};
        // Zero J work so any unwritten ghost/edge cell does not retain
        // stale values that would pollute curl(eta J) (especially Jr from
        // DownwardDz(B_t) on the axial path).
        for (int idim = 0; idim < 3; ++idim) {
            m_Jwork[idim].setVal(0.0_rt);
        }
        m_fdtd->CalculateCurrentAmpere(Jwork, Bwork, *m_eb_update_E, m_lev);


        for (int idim = 0; idim < 3; ++idim) {
            // No OpenMP tiling: ParallelFor(fabbox) with TilingIfNotGPU would
            // race if this loop is ever parallelized, and serial fabbox is fine.
            // Cover valid+ghost so curl(eta J) sees domain-edge faces that
            // FillBoundary cannot invent on non-periodic sides; setVal(0) on J
            // (above) keeps unfilled exterior ghosts safe.
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

    // Precompute the affine feed offset c = A_full(0): the curl driven by the
    // Dirichlet feed value g(t_new) on a zero interior (axis/periodic ghosts
    // are homogeneous, so they contribute nothing). Subtracted from the
    // operator in apply() and from the RHS in AdvanceVariable() so GMRES sees
    // the linear homogeneous operator A_lin and rhs = B^n - c. No-op (c=0)
    // when no pec_insulator B_t feed is active.
    void prepareFeed ()
    {
        m_has_feed = false;
#if defined(WARPX_DIM_RZ)
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
#endif
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

    // Accessors used by the optional PETSc KSP path (MagDiffPetscKSP): the
    // homogeneous apply()/precond() are reused directly, and the assembled
    // frozen-eta Laplacian Pmat needs the B-centered eta (m_eta_pc), the
    // theta*dt scale, and the geometry (cell sizes). apply()/precond() are
    // non-const because they stage through internal work MultiFabs.
    void applyPetsc (MagDiffVector& output, MagDiffVector const& input) { apply(output, input); }
    void precondPetsc (MagDiffVector& destination, MagDiffVector const& source)
    { precond(destination, source); }
    [[nodiscard]] Real thetaDt () const { return m_theta_dt; }
    [[nodiscard]] Geometry const& geom () const { return m_geom; }
    [[nodiscard]] Array<MultiFab,3> const& etaPC () const { return m_eta_pc; }

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

// ---- Optional PETSc KSP path (operator/vec glue) ---------------------------
// HybridMagDiffusionPetsc.{H,cpp} own the PETSc objects; this file owns the
// matrix-free operator (apply/precond) and the field MultiFabs. The glue below
// scatters flat local arrays <-> MagDiffVector interior (nghost=0) using the
// shared per-component global-DOF index owned by the PETSc side, and supplies
// the matvec/PC callbacks the PETSc side invokes. The whole block is compiled
// only when PETSc is on; the caller in AdvanceVariable is guarded the same way.
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
    MagDiffVector PCsrc;   // PC input buffer
    MagDiffVector PCdst;   // PC output buffer
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

// PETSc Vec is host memory. On CUDA, MagDiffVector MultiFabs are device:
// LoopOnCpu into Array4(device) SEGVs. Stage via cached pinned host MultiFabs
// (PetscOpCtx), then MultiFab::Copy. On CPU, write MultiFabs directly (baseline
// walltime_W5 C path — no extra alloc/copy).
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

// PC callback: y = M^{-1} x = linop.precond (scaled Jacobi).
void
petscPCApply (void* ctx, Real const* x, Real* y,
              Array<iMultiFab const*,3> const& gindex, amrex::Long rstart)
{
    auto* c = static_cast<PetscOpCtx*>(ctx);
#ifdef AMREX_USE_GPU
    // Reuse host_U / host_F staging (PC is not concurrent with matvec).
    petscScatter(c->PCsrc, x, gindex, rstart, c->host_U);
    c->linop->precondPetsc(c->PCdst, c->PCsrc);
    petscGather(y, c->PCdst, gindex, rstart, c->host_F);
#else
    petscScatter(c->PCsrc, x, gindex, rstart);
    c->linop->precondPetsc(c->PCdst, c->PCsrc);
    petscGather(y, c->PCdst, gindex, rstart);
#endif
}

#endif // AMREX_USE_PETSC

} // namespace

void
HybridMagDiffusion::Advance (
    ablastr::fields::VectorField const& Bfield,
    Real eta_SI,
    Real dt,
    int lev,
    Array<LinOpBCType, AMREX_SPACEDIM> const& lobc,
    Array<LinOpBCType, AMREX_SPACEDIM> const& hibc) const
{
    ABLASTR_PROFILE("HybridMagDiffusion::Advance()");

    if (!m_enabled) { return; }
    if (dt <= 0.0_rt) { return; }
    if (eta_SI <= 0.0_rt) { return; }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        lev == 0,
        "HybridMagDiffusion only supports single-level hybrid runs");

    // Magnetic diffusivity chi = eta / mu0  [m^2/s]
    const Real chi = eta_SI / PhysConst::mu0;
    // MLCurlCurl: curl(alpha curl B) + beta B = rhs
    // with alpha = theta * dt * chi, beta = 1  =>  (I + theta dt chi curlcurl) B
    const Real alpha = m_theta * dt * chi;

    if (m_verbose > 0) {
        amrex::Print() << "HybridMagDiffusion::Advance: eta=" << eta_SI
                       << " chi=" << chi
                       << " dt=" << dt
                       << " alpha=theta*dt*chi=" << alpha << "\n";
    }

    auto& warpx = WarpX::GetInstance();

#if defined(WARPX_DIM_RZ)
    // AMReX MLCurlCurl supports only Cartesian operators in 2D. Build a
    // constant eta field on the E/J staggering and use the matrix-free
    // cylindrical FDTD curl chain instead.
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
    // solid. Mirrors the Cartesian EB branch and BuildMagDiffResistivity; the
    // matvec is already safe (computeAFull zeros Jwork), but this keeps the
    // B-face PC and the frozen-η PETSc Pmat clean near the EB. No-op EB-off.
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
    AdvanceVariable(Bfield, eta_field, dt, lev, lobc, hibc);
#else
    // EB: plain MLCurlCurl has no embedded-boundary API and would silently
    // solve the wrong operator. Route to the matrix-free path (same as RZ),
    // which is EB-aware as of Milestone 1 (stair-case masks: covered E faces
    // carry η=0, covered B inert). See notes/2026-07-18_eb_parser_eta.md.
    if (EB::enabled()) {
        ablastr::fields::VectorField const current_layout =
            warpx.m_fields.get_alldirs(
                warpx::fields::FieldType::hybrid_current_fp_plasma, lev);
        Array<MultiFab,3> eta_storage;
        for (int idim = 0; idim < 3; ++idim) {
            eta_storage[idim].define(
                current_layout[idim]->boxArray(),
                current_layout[idim]->DistributionMap(),
                1, current_layout[idim]->nGrowVect());
            eta_storage[idim].setVal(eta_SI);
        }
        // Zero η on covered E faces (eb_update_E == 0): no diffusion into the
        // solid. The matvec is already safe (computeAFull zeros Jwork, so
        // J = 0 on covered E faces), but this makes the B-face PC
        // (SampleEtaOntoBFace) and the frozen-η PETSc Pmat (M2) see fluid η
        // only near the EB. Mirrors BuildMagDiffResistivity's masking.
        {
            auto const& eb_update_E = warpx.GetEBUpdateEFlag()[lev];
            amrex::Periodicity const& eb_period = warpx.Geom(lev).periodicity();
            for (int idim = 0; idim < 3; ++idim) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(eta_storage[idim], TilingIfNotGPU());
                     mfi.isValid(); ++mfi) {
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
        AdvanceVariable(Bfield, eta_field, dt, lev, lobc, hibc);
        return;
    }

    // Cartesian constant-eta path via AMReX MLCurlCurl.
    Geometry const& geom = warpx.Geom(lev);
    // MLCurlCurl expects cell-centered BoxArray (enclosedCells of edge BA)
    BoxArray ba = Bfield[0]->boxArray();
    ba.enclosedCells();
    DistributionMapping const& dm = Bfield[0]->DistributionMap();

    LPInfo info;
    info.setMaxCoarseningLevel(30);

#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    constexpr int coord = 1;
#else
    constexpr int coord = 0;
#endif

    Vector<Geometry> const geom_v{geom};
    Vector<BoxArray> const grids_v{ba};
    Vector<DistributionMapping> const dmap_v{dm};

    MLCurlCurl linop(geom_v, grids_v, dmap_v, info, coord);
    linop.setDomainBC(lobc, hibc);
    linop.setScalars(alpha, Real(1.0));

    // Working storage in WarpX component order
    Array<MultiFab, 3> sol;
    Array<MultiFab, 3> rhs;
    for (int idim = 0; idim < 3; ++idim) {
        const IntVect ng = Bfield[idim]->nGrowVect();
        sol[idim].define(Bfield[idim]->boxArray(), Bfield[idim]->DistributionMap(), 1, ng);
        rhs[idim].define(Bfield[idim]->boxArray(), Bfield[idim]->DistributionMap(), 1, ng);
        MultiFab::Copy(sol[idim], *Bfield[idim], 0, 0, 1, ng);
        MultiFab::Copy(rhs[idim], *Bfield[idim], 0, 0, 1, ng);
    }

    // For 0 < theta < 1 (e.g. Crank–Nicolson):
    //   rhs = B - (1-theta) dt chi curlcurl B
    //       = 2 B - A_e(B)
    // where A_e uses alpha_e = (1-theta) dt chi and beta = 1.
    if (m_theta < 1.0_rt) {
        for (int idim = 0; idim < 3; ++idim) {
            ablastr::utils::communication::FillBoundary(
                rhs[idim],
                WarpX::do_single_precision_comms,
                geom.periodicity(),
                true);
        }

        const Real alpha_e = (1.0_rt - m_theta) * dt * chi;
        MLCurlCurl linop_e(geom_v, grids_v, dmap_v, info, coord);
        linop_e.setDomainBC(lobc, hibc);
        linop_e.setScalars(alpha_e, Real(1.0));
        linop_e.setLevelBC(0, nullptr);
        linop_e.prepareForSolve();

        Array<MultiFab, 3> Ae_out;
        for (int idim = 0; idim < 3; ++idim) {
            Ae_out[idim].define(
                Bfield[idim]->boxArray(), Bfield[idim]->DistributionMap(), 1, 0);
            Ae_out[idim].setVal(0.0_rt);
        }

        Array<MultiFab, 3> in_arr = MakeCurlCurlAliases(rhs);
        Array<MultiFab, 3> out_arr = MakeCurlCurlAliases(Ae_out);

        using Op = MLLinOpT<Array<MultiFab,3>>;
        linop_e.apply(0, 0, out_arr, in_arr, Op::BCMode::Homogeneous, Op::StateMode::Solution);

        for (int idim = 0; idim < 3; ++idim) {
            // rhs := 2*B - A_e(B)
            MultiFab::LinComb(
                rhs[idim],
                2.0_rt, sol[idim], 0,
                -1.0_rt, Ae_out[idim], 0,
                0, 1, 0);
        }
    }

    Array<MultiFab, 3> solution = MakeCurlCurlAliases(sol);
    Array<MultiFab, 3> rhs_arr = MakeCurlCurlAliases(rhs);

    linop.setLevelBC(0, nullptr);
    linop.prepareRHS({&rhs_arr});

    using MFArr = Array<MultiFab, 3>;
    MLMGT<MFArr> mlmg(linop);
    mlmg.setMaxIter(m_max_iter);
    mlmg.setVerbose(m_verbose);
    mlmg.setBottomVerbose(0);
    const Real residual = mlmg.solve({&solution}, {&rhs_arr}, m_rtol, m_atol);

    if (m_verbose > 0) {
        amrex::Print() << "HybridMagDiffusion MLMG residual=" << residual << "\n";
    }

    for (int idim = 0; idim < 3; ++idim) {
        MultiFab::Copy(*Bfield[idim], sol[idim], 0, 0, 1, 0);
        ablastr::utils::communication::FillBoundary(
            *Bfield[idim],
            WarpX::do_single_precision_comms,
            geom.periodicity(),
            true);
    }
#endif // WARPX_DIM_RZ
}

void
HybridMagDiffusion::AdvanceVariable (
    ablastr::fields::VectorField const& Bfield,
    ablastr::fields::VectorField const& eta_SI,
    Real dt,
    int lev,
    Array<LinOpBCType, AMREX_SPACEDIM> const& lobc,
    Array<LinOpBCType, AMREX_SPACEDIM> const& hibc) const
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

    VariableCoeffMagDiffusionOp linop(Bfield, eta_SI, m_theta * dt, lev, lobc, hibc);

    MagDiffVector solution;
    MagDiffVector rhs;
    solution.Define(Bfield);
    rhs.Define(Bfield);
    // Capture B^n before prepareFeed: the operator stages its Krylov vectors
    // (and the zero probe for the feed offset) through m_source, which is the
    // registered B field (= Bfield). The true B^n lives in solution/rhs.
    solution.CopyFrom(Bfield);
    rhs.CopyFrom(Bfield);

    // Covered B is inert (identity operator row; see VariableCoeffMagDiffusionOp
    // and MagDiffJacobiPrecond). Zero it in the init-guess and RHS so the
    // Krylov solve keeps covered B = 0 even if B^n carried nonzero there
    // (defensive; B^n is normally 0 on covered from the hybrid EM update, and
    // the feed offset c is 0 on covered). Reused before copy-back below.
    auto const& eb_update_B = WarpX::GetInstance().GetEBUpdateBFlag()[lev];
    zeroCoveredB(solution, eb_update_B);
    zeroCoveredB(rhs, eb_update_B);

    // Bake the inhomogeneous Dirichlet B_t feed (pec_insulator) into the RHS.
    // c = A_full(0) = theta*dt*K_feed(g_new) is the constant curl from the feed
    // value g(t_new); the operator seen by GMRES/PETSc is the homogeneous
    // A_lin = A_full - c (see VariableCoeffMagDiffusionOp::apply). prepareFeed
    // stages a zero probe through m_source (clobbering the registered B), so it
    // must run after the B^n copy above. It is a no-op (c = 0) with no feed.
    linop.prepareFeed();
    bool const has_feed = linop.hasFeed();

    // Theta-method RHS for  (I + theta*dt*K) B^{n+1} = rhs, where
    //   rhs = B^n - (1-theta)*dt*K_hom B^n - feed_RHS
    // built in the MLCurlCurl-mirroring form  rhs = 2*B^n - A_e(B^n) - c,
    // with A_e(x) = computeAFull(x, alpha=(1-theta)*dt) = x + (1-theta)*dt*K_full(x)
    // using the same frozen eta and feed ghost g(t_new) as the implicit operator.
    // Expanding (K_full = K_hom + K_feed) gives rhs = B^n - (1-theta)*dt*K_hom B^n
    // - dt*K_feed(g_new): the feed is injected once total (theta*dt implicit +
    // (1-theta)*dt explicit), not double-counted on the explicit L(B^n) term.
    // NOTE: both the implicit and explicit terms sample the pec_insulator feed at
    // g(t_new) (computeAFull calls ApplyBfieldBoundary at t_new). For a TIME-
    // VARYING feed this makes the boundary-driven component first-order in time
    // for CN (the interior stays second-order); backward Euler is unaffected.
    // The full trapezoidal BC blend (g_old + g_new) would need
    // ApplyBfieldBoundary at t_old = t_new - dt and is a documented follow-up
    // (design note Sec. 4); the affine feed split itself is correct here.
    // - theta == 1: A_e uses alpha = 0, so A_e(B^n) = B^n and rhs = B^n - c, the
    //   existing backward-Euler path (skipped here to stay bit-/test-identical
    //   and avoid a wasted matrix-free apply).
    // - theta < 1: one extra matrix-free apply per step (same cost as the
    //   Cartesian MLCurlCurl CN path). c = 0 with no feed -> rhs = 2*B^n - A_e.
    if (m_theta < 1.0_rt) {
        MagDiffVector Ae;
        Ae.Define(Bfield);
        const Real alpha_e = (1.0_rt - m_theta) * dt;
        linop.computeAFull(Ae, solution, alpha_e);
        rhs.linComb(2.0_rt, solution, -1.0_rt, Ae);
    }
    if (has_feed) {
        rhs.increment(linop.feedOffset(), -1.0_rt);
    }

    if (m_linear_solver == MagDiffLinearSolver::petsc) {
#ifdef AMREX_USE_PETSC
        // Optional PETSc KSP path: same matrix-free operator (MATSHELL matvec =
        // linop.apply, the homogeneous A_lin) and same RHS (built above:
        // 2*B^n - A_e(B^n) - c for theta<1, B^n - c for theta=1), only the
        // solver driver differs. PC is the existing scaled Jacobi (shell) by
        // default, or an assembled frozen-eta Laplacian Pmat for algebraic PCs.
        PetscOpCtx opctx;
        opctx.linop = &linop;
        opctx.U.Define(makeVectorFieldView(solution.fields()));
        opctx.F.Define(makeVectorFieldView(solution.fields()));
        opctx.PCsrc.Define(makeVectorFieldView(solution.fields()));
        opctx.PCdst.Define(makeVectorFieldView(solution.fields()));
#ifdef AMREX_USE_GPU
        // One-time pinned host staging for the whole KSP (not per matvec).
        opctx.ensureHostBufs(solution.fields());
#endif

        auto& eta_pc = linop.etaPC();
        amrex::Array<MultiFab const*,3> const B_proto{
            &solution.fields()[0], &solution.fields()[1], &solution.fields()[2]};
        amrex::Array<MultiFab const*,3> const eta_proto{
            &eta_pc[0], &eta_pc[1], &eta_pc[2]};

        // Pass the EB B-field mask (if any) so covered DOFs are skipped from
        // the PETSc system. Must be nullptr when EB is off (the mask MultiFabs
        // are undefined then), matching the default parameter in the header.
        amrex::Array<iMultiFab const*,3> const eb_update_B_ptrs{
            eb_update_B[0].get(), eb_update_B[1].get(), eb_update_B[2].get()};
        bool const petsc_eb_on = EB::enabled();

        MagDiffPetscSolver* petsc_solver = magdiff_petsc_make(
            B_proto, eta_proto, linop.geom(), linop.thetaDt(), PhysConst::mu0,
            m_petsc_pc, m_rtol, m_atol, m_max_iter, m_verbose,
            &petscMatvec, &petscPCApply, &opctx,
            petsc_eb_on ? &eb_update_B_ptrs : nullptr);

        const amrex::Long n_local = magdiff_petsc_nlocal(petsc_solver);
        const amrex::Long rstart = magdiff_petsc_rstart(petsc_solver);
        auto const gindex = magdiff_petsc_gindex(petsc_solver);

        // Initial guess (B^n) and RHS (B^n - feed offset), flat / DOF-mapped.
        std::vector<Real> sol_flat(static_cast<std::size_t>(n_local), Real(0.0));
        std::vector<Real> rhs_flat(static_cast<std::size_t>(n_local), Real(0.0));
#ifdef AMREX_USE_GPU
        // Same host staging as matvec/PC callbacks (CUDA: device MultiFabs).
        petscGather(sol_flat.data(), solution, gindex, rstart, opctx.host_F);
        petscGather(rhs_flat.data(), rhs, gindex, rstart, opctx.host_F);
#else
        petscGather(sol_flat.data(), solution, gindex, rstart);
        petscGather(rhs_flat.data(), rhs, gindex, rstart);
#endif

        Real rnorm = 0.0_rt;
        const int reason = magdiff_petsc_solve(
            petsc_solver, rhs_flat.data(), sol_flat.data(), rnorm);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(rnorm),
            "HybridMagDiffusion PETSc KSP residual is non-finite (NaN/Inf). "
            "Variable-eta operator or preconditioner is broken.");
        amrex::ignore_unused(reason);

#ifdef AMREX_USE_GPU
        petscScatter(solution, sol_flat.data(), gindex, rstart, opctx.host_U);
#else
        petscScatter(solution, sol_flat.data(), gindex, rstart);
#endif
        magdiff_petsc_destroy(petsc_solver);
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

        if (solver.getStatus() != 0) {
            const Real rnorm = solver.getResidualNorm();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::isfinite(rnorm),
                "HybridMagDiffusion GMRES residual is non-finite (NaN/Inf). "
                "Variable-eta operator or preconditioner is broken.");
            // Non-converged but finite: keep last iterate and warn.
            amrex::Print() << "### WARNING: HybridMagDiffusion GMRES did not converge"
                           << " (iters=" << solver.getNumIters()
                           << " residual=" << rnorm
                           << " status=" << solver.getStatus()
                           << "). Keeping last iterate.\n";
        } else if (m_verbose > 0) {
            amrex::Print() << "HybridMagDiffusion matrix-free GMRES iterations="
                           << solver.getNumIters()
                           << " residual=" << solver.getResidualNorm() << "\n";
        }
    }

    auto const& solution_fields = solution.fields();
    // Defensive: ensure covered B stays 0 in the registered field. The solve
    // keeps it 0 when the RHS is 0 there, but force it here so a non-converged
    // last iterate (kept by the GMRES non-finite/failed-conv branch above) can
    // never write a nonzero value into the solid.
    zeroCoveredB(solution, eb_update_B);
    auto& warpx = WarpX::GetInstance();
#if !defined(WARPX_DIM_RZ)
    Geometry const& geom = warpx.Geom(lev);
#endif
    for (int idim = 0; idim < 3; ++idim) {
        MultiFab::Copy(*Bfield[idim], solution_fields[idim], 0, 0, 1, 0);
#if !defined(WARPX_DIM_RZ)
        ablastr::utils::communication::FillBoundary(
            *Bfield[idim],
            WarpX::do_single_precision_comms,
            geom.periodicity(),
            true);
#endif
    }
#if defined(WARPX_DIM_RZ)
    warpx.ApplyBfieldBoundary(
        lev, PatchType::fine, SubcyclingHalf::None, warpx.gett_new(lev));
    warpx.FillBoundaryB(lev, Bfield[0]->nGrowVect(), true);
#endif
}
