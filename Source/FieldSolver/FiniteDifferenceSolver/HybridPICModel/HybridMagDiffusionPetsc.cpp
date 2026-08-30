/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Bowen Zhu
 *
 * License: BSD-3-Clause-LBNL
 *
 * PETSc KSP/PC implementation for the optional matrix-free hybrid
 * magnetic-diffusion curl-curl solve. This translation unit includes `petsc*.h`
 * and therefore MUST NOT include WarpX/ablastr headers (WarpX's own global
 * `ReductionType` enum, WarpXAlgorithmSelection.H, collides with PETSc's global
 * `ReductionType`, petscvec.h). Only pure AMReX headers are used here, mirroring
 * the WarpX_PETSc.cpp isolation precedent.
 *
 * The matrix-free operator (matvec) is supplied as a callback by
 * HybridMagDiffusion.cpp, which owns the Yee/RZ curl-curl apply() and the field
 * MultiFabs. This file owns the PETSc objects (Vec, MatShell, assembled Pmat,
 * KSP, PC), the DOF layout, and the assembled frozen-η curl-curl Pmat.
 */
#include "HybridMagDiffusionPetsc.H"

#include <AMReX.H>
#include <AMReX_Arena.H>
#include <AMReX_Config.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_Loop.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>

#ifdef AMREX_USE_PETSC
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscsys.h>
#include <petscvec.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#ifdef AMREX_USE_PETSC
// PetscCall (PETSc's macro) can only be used inside functions returning
// PetscErrorCode (the static callbacks and assemblePreconditioner). In other
// functions (ctor, make, solve) we abort on a non-zero PETSc error.
#define MAGDIFF_PETSC_CHK(call) do { \
    PetscErrorCode const _magdiff_ierr = (call); \
    if (_magdiff_ierr != 0) { \
        amrex::Abort("HybridMagDiffusion PETSc call failed"); } \
} while (0)
#endif


#ifdef AMREX_USE_PETSC

namespace {

/**
 * \brief PETSc KSP driver for the matrix-free mag-diff operator.
 *
 * DOF layout: component-major (B0, B1, B2), per-rank MFIter order, interior
 * (nghost=0) cells only. A per-component global-index iMultiFab (nghost=1,
 * -1 = exterior) maps each interior cell to its PETSc row/column; the matvec
 * callback (in HybridMagDiffusion.cpp) reads this same index to scatter/gather
 * its field MultiFabs, so the Vec and the assembled Pmat stay
 * self-consistent regardless of tiling. Norms use the nghost=0 interior only
 * (the FPE-trap invariant).
 */
class MagDiffPetscSolverImpl
{
public:
    MagDiffPetscSolverImpl (
        amrex::Array<amrex::MultiFab const*,3> const& B_proto,
        amrex::Array<amrex::MultiFab const*,3> const& eta_edge,
        amrex::Geometry const& geom, amrex::Real theta_dt, amrex::Real mu0,
        amrex::Real rtol, amrex::Real atol, int max_iter, int verbose,
        MagDiffPetscOptions const& options,
        MagDiffMatvecFn matvec, void* opctx,
        amrex::Array<amrex::iMultiFab const*,3> const* eb_update_B)
        : m_geom(geom), m_theta_dt(theta_dt), m_mu0(mu0),
          m_rtol(rtol), m_atol(atol), m_max_iter(max_iter),
          m_verbose(verbose), m_matvec(matvec), m_opctx(opctx)
    {
        // EB masks live on the device under CUDA. Copy to pinned host once so
        // all LoopOnCpu DOF counting / indexing is host-safe (raw device
        // Array4 from eb_update_B in LoopOnCpu SEGVs on GPU builds).
        if (eb_update_B) {
            copyEbMasksToHost(*eb_update_B);
        }

        // Local DOF count over interior (nghost=0) cells, component-major,
        // skipping covered B DOFs when the EB mask is provided.
        for (int idim = 0; idim < 3; ++idim) {
            for (amrex::MFIter mfi(*B_proto[idim]); mfi.isValid(); ++mfi) {
                if (m_eb_mask_host[0]) {
                    auto const& mask_arr = m_eb_mask_host[idim]->const_array(mfi);
                    amrex::Box const& tb = mfi.tilebox();
                    amrex::LoopOnCpu(amrex::lbound(tb), amrex::ubound(tb),
                        [&] (int i, int j, int k) {
                            if (mask_arr(i, j, k) != 0) { ++m_n_local; }
                        });
                } else {
                    m_n_local += mfi.tilebox().numPts();
                }
            }
        }

        MAGDIFF_PETSC_CHK(VecCreateMPI(
            PETSC_COMM_WORLD, m_n_local, PETSC_DETERMINE, &m_x));
        MAGDIFF_PETSC_CHK(VecDuplicate(m_x, &m_b));
        PetscInt rstart = 0;
        MAGDIFF_PETSC_CHK(VecGetOwnershipRange(m_x, &rstart, nullptr));
        m_rstart = rstart;
        PetscInt ng = 0;
        MAGDIFF_PETSC_CHK(VecGetSize(m_x, &ng));
        m_n_global = ng;

        buildGlobalIndex(B_proto);
        // Exact curl-curl rows select η on E/J faces to match the matvec.
        buildEtaGhosts(eta_edge);

        // MatShell operator: matvec = caller's apply (homogeneous A_lin).
        MAGDIFF_PETSC_CHK(MatCreateShell(
            PETSC_COMM_WORLD, m_n_local, m_n_local, m_n_global, m_n_global,
            this, &m_A));
        MAGDIFF_PETSC_CHK(MatShellSetOperation(
            m_A, MATOP_MULT, reinterpret_cast<void(*)(void)>(applyMatOp)));
        MAGDIFF_PETSC_CHK(MatSetUp(m_A));

        MAGDIFF_PETSC_CHK(KSPCreate(PETSC_COMM_WORLD, &m_ksp));
        MAGDIFF_PETSC_CHK(KSPSetOperators(m_ksp, m_A, m_A));

        // Assembled frozen-η curl-curl Pmat (see assemblePreconditioner).
        // Lets runtime -pc_type asm/ilu/hypre factor a real matrix; default PC
        // for an assembled Pmat is block-Jacobi ILU(0). Preallocation follows
        // the maximum exact stencil width in each geometry.
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
        PetscInt const nz = 9;
#elif defined(WARPX_DIM_3D)
        // Four same-component neighbors, eight mixed couplings, and diagonal.
        PetscInt const nz = 13;
#elif defined(WARPX_DIM_XZ)
        PetscInt const nz = 7;
#else
        PetscInt const nz = 3;
#endif
        MAGDIFF_PETSC_CHK(MatCreateAIJ(
            PETSC_COMM_WORLD, m_n_local, m_n_local, m_n_global, m_n_global,
            nz, nullptr, nz, nullptr, &m_P));
        // Variable eta can activate a stencil coefficient that was initially
        // zero. Store those structural zeros so later updates only change
        // values, not the matrix graph.
        MAGDIFF_PETSC_CHK(MatSetOption(
            m_P, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE));
        // Fail loudly if a row exceeds preallocation (heap corruption risk).
        MAGDIFF_PETSC_CHK(MatSetOption(
            m_P, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
        MAGDIFF_PETSC_CHK(assemblePreconditioner());
        MAGDIFF_PETSC_CHK(KSPSetOperators(m_ksp, m_A, m_P));
        MAGDIFF_PETSC_CHK(KSPSetPCSide(m_ksp, PC_RIGHT));
        MAGDIFF_PETSC_CHK(KSPSetType(m_ksp, KSPGMRES));
        MAGDIFF_PETSC_CHK(KSPSetTolerances(
            m_ksp, m_rtol, m_atol, PETSC_DEFAULT, m_max_iter));
        MAGDIFF_PETSC_CHK(KSPSetNormType(m_ksp, KSP_NORM_UNPRECONDITIONED));
        MAGDIFF_PETSC_CHK(KSPGMRESSetRestart(
            m_ksp, std::min(m_max_iter, 50)));
        if (m_verbose > 0) {
            MAGDIFF_PETSC_CHK(KSPMonitorSet(
                m_ksp, monitorResidual, nullptr, nullptr));
        }
        setOptions(options);
        // Let -ksp_type / -pc_type / -ksp_rtol ... from argv or PETSC_OPTIONS win.
        MAGDIFF_PETSC_CHK(KSPSetFromOptions(m_ksp));
    }

    ~MagDiffPetscSolverImpl () {
        // Destroy KSP before Mats/Vecs it references (PETSc refcounts make
        // reverse order usually safe, but KSP-first is the documented pattern).
        if (m_ksp) { KSPDestroy(&m_ksp); m_ksp = nullptr; }
        if (m_A)   { MatDestroy(&m_A);   m_A = nullptr; }
        if (m_P)   { MatDestroy(&m_P);   m_P = nullptr; }
        if (m_x)   { VecDestroy(&m_x);   m_x = nullptr; }
        if (m_b)   { VecDestroy(&m_b);   m_b = nullptr; }
    }

    MagDiffPetscSolverImpl (MagDiffPetscSolverImpl const&) = delete;
    MagDiffPetscSolverImpl& operator= (MagDiffPetscSolverImpl const&) = delete;

    [[nodiscard]] amrex::Long nlocal () const { return m_n_local; }
    [[nodiscard]] amrex::Long rstart () const { return m_rstart; }

    amrex::Array<amrex::iMultiFab const*,3> gindexView () const {
        return {m_gindex[0].get(), m_gindex[1].get(), m_gindex[2].get()};
    }

    int solve (amrex::Real const* rhs, amrex::Real* sol, amrex::Real& rnorm_out) {
        PetscScalar* barr = nullptr;
        MAGDIFF_PETSC_CHK(VecGetArray(m_b, &barr));
        for (amrex::Long i = 0; i < m_n_local; ++i) { barr[i] = rhs[i]; }
        MAGDIFF_PETSC_CHK(VecRestoreArray(m_b, &barr));

        PetscScalar* xarr = nullptr;
        MAGDIFF_PETSC_CHK(VecGetArray(m_x, &xarr));
        for (amrex::Long i = 0; i < m_n_local; ++i) { xarr[i] = sol[i]; }
        MAGDIFF_PETSC_CHK(VecRestoreArray(m_x, &xarr));

        MAGDIFF_PETSC_CHK(KSPSolve(m_ksp, m_b, m_x));

        MAGDIFF_PETSC_CHK(VecGetArray(m_x, &xarr));
        for (amrex::Long i = 0; i < m_n_local; ++i) { sol[i] = static_cast<amrex::Real>(xarr[i]); }
        MAGDIFF_PETSC_CHK(VecRestoreArray(m_x, &xarr));

        PetscInt iters = 0;
        KSPConvergedReason reason = static_cast<KSPConvergedReason>(0);
        PetscReal rnorm = 0.0;
        MAGDIFF_PETSC_CHK(KSPGetIterationNumber(m_ksp, &iters));
        MAGDIFF_PETSC_CHK(KSPGetConvergedReason(m_ksp, &reason));
        MAGDIFF_PETSC_CHK(KSPGetResidualNorm(m_ksp, &rnorm));
        rnorm_out = static_cast<amrex::Real>(rnorm);

        if (reason <= 0) {
            amrex::Print() << "### WARNING: HybridMagDiffusion PETSc KSP did not"
                           << " converge (iters=" << iters
                           << " residual=" << rnorm
                           << " reason=" << reason
                           << "). Keeping last iterate.\n";
        } else if (m_verbose > 0) {
            amrex::Print() << "HybridMagDiffusion PETSc KSP iterations=" << iters
                           << " residual=" << rnorm
                           << " reason=" << reason << "\n";
        }
        return static_cast<int>(reason);
    }

private:
    void setOptions (MagDiffPetscOptions const& options)
    {
        PetscBool pc_specified = PETSC_FALSE;
        MAGDIFF_PETSC_CHK(PetscOptionsHasName(
            nullptr, nullptr, "-pc_type", &pc_specified));
        if (pc_specified || options.pc_type.empty()) { return; }

        MAGDIFF_PETSC_CHK(PetscOptionsSetValue(
            nullptr, "-pc_type", options.pc_type.c_str()));
        if (options.pc_type != "asm") { return; }

        std::string const overlap = std::to_string(options.asm_overlap);
        MAGDIFF_PETSC_CHK(PetscOptionsSetValue(
            nullptr, "-pc_asm_overlap", overlap.c_str()));
        MAGDIFF_PETSC_CHK(PetscOptionsSetValue(
            nullptr, "-sub_ksp_type", options.sub_ksp_type.c_str()));
        MAGDIFF_PETSC_CHK(PetscOptionsSetValue(
            nullptr, "-sub_pc_type", options.sub_pc_type.c_str()));
        if (options.sub_pc_type == "ilu") {
            std::string const levels = std::to_string(options.ilu_factor_levels);
            MAGDIFF_PETSC_CHK(PetscOptionsSetValue(
                nullptr, "-sub_pc_factor_levels", levels.c_str()));
        }
    }

    static PetscErrorCode applyMatOp (Mat A, Vec in, Vec out) {
        PetscFunctionBeginUser;
        MagDiffPetscSolverImpl* self = nullptr;
        PetscCall(MatShellGetContext(A, reinterpret_cast<void**>(&self)));
        const PetscScalar* x = nullptr; PetscScalar* y = nullptr;
        PetscCall(VecGetArrayRead(in, &x));
        PetscCall(VecGetArray(out, &y));
        self->m_matvec(self->m_opctx,
                       reinterpret_cast<amrex::Real const*>(x),
                       reinterpret_cast<amrex::Real*>(y),
                       self->gindexView(), self->m_rstart);
        PetscCall(VecRestoreArrayRead(in, &x));
        PetscCall(VecRestoreArray(out, &y));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    static PetscErrorCode monitorResidual (KSP ksp, PetscInt it,
                                           PetscReal rnorm, void* /*ctx*/) {
        PetscFunctionBeginUser;
        amrex::ignore_unused(ksp);
        amrex::Print() << "  HybridMagDiffusion PETSc KSP: it="
                       << it << " residual=" << rnorm << "\n";
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // Copy device (or host) EB B-masks onto pinned host iMultiFabs for safe
    // LoopOnCpu access. Called once from the ctor when eb_update_B is non-null.
    void copyEbMasksToHost (
        amrex::Array<amrex::iMultiFab const*,3> const& eb_update_B)
    {
        amrex::MFInfo const host_info =
            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
        for (int idim = 0; idim < 3; ++idim) {
            m_eb_mask_host[idim] = std::make_unique<amrex::iMultiFab>(
                eb_update_B[idim]->boxArray(),
                eb_update_B[idim]->DistributionMap(), 1, 0, host_info);
            // iMultiFab::Copy handles device -> pinned host under CUDA.
            amrex::iMultiFab::Copy(*m_eb_mask_host[idim], *eb_update_B[idim],
                                   0, 0, 1, 0);
        }
#ifdef AMREX_USE_GPU
        amrex::Gpu::streamSynchronize();
#endif
    }

    // Per-component global DOF index for interior cells (component-major).
    // nghost=1 so neighbor columns resolve; -1 marks exterior/unknown or covered
    // B DOFs (when an EB mask is provided). Pinned host arena: PETSc scatter/gather
    // and Mat assembly use LoopOnCpu host access; device-arena iMultiFab would SEGV
    // under CUDA WarpX.
    void buildGlobalIndex (amrex::Array<amrex::MultiFab const*,3> const& B_proto) {
        amrex::MFInfo const host_info =
            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
        for (int idim = 0; idim < 3; ++idim) {
            m_gindex[idim] = std::make_unique<amrex::iMultiFab>(
                B_proto[idim]->boxArray(),
                B_proto[idim]->DistributionMap(), 1, amrex::IntVect::Unit,
                host_info);
            m_gindex[idim]->setVal(-1);
        }
        // amrex::Box is BoxND<AMREX_SPACEDIM>, so iterate with the dim-agnostic
        // LoopOnCpu(lbound,ubound,(i,j,k)) (k pads to 0 in 2D/1D), not explicit
        // smallEnd(2)/bigEnd(2) which is out of range for BoxND<2>.
        PetscInt run = static_cast<PetscInt>(m_rstart);
        bool const skip_covered = (m_eb_mask_host[0] != nullptr);
        for (int idim = 0; idim < 3; ++idim) {
            for (amrex::MFIter mfi(*B_proto[idim]); mfi.isValid(); ++mfi) {
                auto gix = m_gindex[idim]->array(mfi);
                amrex::Box const& tb = mfi.tilebox();
                if (skip_covered) {
                    auto const& mask_arr = m_eb_mask_host[idim]->const_array(mfi);
                    amrex::LoopOnCpu(amrex::lbound(tb), amrex::ubound(tb),
                        [&] (int i, int j, int k) {
                            if (mask_arr(i, j, k) == 0) { return; }
                            gix(i, j, k) = static_cast<int>(run);
                            ++run;
                        });
                } else {
                    amrex::LoopOnCpu(amrex::lbound(tb), amrex::ubound(tb),
                        [&] (int i, int j, int k) {
                            gix(i, j, k) = static_cast<int>(run);
                            ++run;
                        });
                }
            }
        }
        for (int idim = 0; idim < 3; ++idim) {
            m_gindex[idim]->FillBoundary(m_geom.periodicity());
        }
    }

    // nghost=1 copy of edge-centered eta for neighbor reads in the assembled Pmat.
    // Pinned host: assemblePreconditioner uses LoopOnCpu host access.
    void buildEtaGhosts (amrex::Array<amrex::MultiFab const*,3> const& eta_edge) {
        amrex::MFInfo const host_info =
            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
        for (int idim = 0; idim < 3; ++idim) {
            if (!m_eta_g[idim].ok()) {
                m_eta_g[idim].define(eta_edge[idim]->boxArray(),
                                     eta_edge[idim]->DistributionMap(), 1,
                                     amrex::IntVect::Unit, host_info);
            }
            m_eta_g[idim].setVal(amrex::Real(0.0));
            // eta_edge may be device memory under CUDA; Copy handles H<->D.
            amrex::MultiFab::Copy(m_eta_g[idim], *eta_edge[idim], 0, 0, 1, 0);
            m_eta_g[idim].FillBoundary(m_geom.periodicity());
        }
    }

    // Assemble frozen-η curl-curl Pmat (P != A; matvec stays
    // matrix-free computeAFull). Exterior neighbors (global index -1) are
    // dropped, matching the homogeneous-operator BC the matvec imposes.
    //
    // RZ / RCYLINDER: discrete curl-curl stencil from the staggered Yee/RZ
    // operator — uncoupled Bt 5-point block; Br and Bz coupled via J_θ mixed
    // derivatives (~7-point). Axis: Br(0,*) is diagonal-only (pin); Bt on-axis
    // uses the 4/dr^2 J_z correction. η is face-selected per term (et_r/t/z).
    // See notes/2026-07-20_exact_curl_curl_pc.md.
    //
    // Cartesian: exact algebraic expansion of the staggered Yee operator,
    // C_up diag(eta/mu0) C_down. Inactive derivatives vanish in XZ and 1D_Z.

    PetscErrorCode assemblePreconditioner () {
        PetscFunctionBeginUser;
        if (m_P) {
            PetscCall(MatZeroEntries(m_P));
        }
        auto const* const dx = m_geom.CellSize();
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
        amrex::Real const rmin = m_geom.ProbLo(0);
        amrex::Real const dr = dx[0];
        amrex::Real const dz = (AMREX_SPACEDIM > 1) ? dx[1] : amrex::Real(1.0);
        amrex::Real const dr2 = dr * dr;
        amrex::Real const dz2 = dz * dz;
        amrex::Real const drdz = dr * dz;
#else
        // Map physical x/y/z derivatives to AMReX index directions. A value
        // of -1 marks an invariant direction that contributes no curl term.
#if defined(WARPX_DIM_3D)
        std::array<int,3> const phys_to_amrex{0, 1, 2};
        std::array<amrex::Real,3> const spacing{dx[0], dx[1], dx[2]};
#elif defined(WARPX_DIM_XZ)
        std::array<int,3> const phys_to_amrex{0, -1, 1};
        std::array<amrex::Real,3> const spacing{dx[0], 1.0, dx[1]};
#elif defined(WARPX_DIM_1D_Z)
        std::array<int,3> const phys_to_amrex{-1, -1, 0};
        std::array<amrex::Real,3> const spacing{1.0, 1.0, dx[0]};
#else
        std::array<int,3> const phys_to_amrex{-1, -1, -1};
        std::array<amrex::Real,3> const spacing{1.0, 1.0, 1.0};
#endif
#endif

        std::vector<PetscInt> cols;
        std::vector<PetscScalar> vals;
        cols.reserve(13);
        vals.reserve(13);

        for (int idim = 0; idim < 3; ++idim) {
            for (amrex::MFIter mfi(*m_gindex[idim]); mfi.isValid(); ++mfi) {
                auto const& gix = m_gindex[idim]->const_array(mfi);
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                auto const& gix_r = m_gindex[0]->const_array(mfi);
                auto const& gix_t = m_gindex[1]->const_array(mfi);
                auto const& gix_z = m_gindex[2]->const_array(mfi);
                auto const& et_r = m_eta_g[0].const_array(mfi);
                auto const& et_t = m_eta_g[1].const_array(mfi);
                auto const& et_z = m_eta_g[2].const_array(mfi);
#else
                amrex::Array<amrex::Array4<int const>,3> const gix_b{
                    m_gindex[0]->const_array(mfi),
                    m_gindex[1]->const_array(mfi),
                    m_gindex[2]->const_array(mfi)};
                amrex::Array<amrex::Array4<amrex::Real const>,3> const eta_j{
                    m_eta_g[0].const_array(mfi),
                    m_eta_g[1].const_array(mfi),
                    m_eta_g[2].const_array(mfi)};
#endif
                amrex::Box const& tb = mfi.tilebox();
                amrex::LoopOnCpu(amrex::lbound(tb), amrex::ubound(tb),
                [&] (int i, int j, int k) {
                    PetscInt const row = gix(i, j, k);
                    if (row < 0) { return; }  // covered B DOF (EB), skip
                    PetscScalar diag = 1.0;
                    cols.clear();
                    vals.clear();

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                    // RZ / RCYLINDER Exact Curl-Curl Assembly
                    amrex::Real const r_node_i = rmin + i * dr;
                    amrex::Real const r_node_ip1 = rmin + (i + 1) * dr;
                    amrex::Real const r_cell_i = rmin + (i + 0.5) * dr;
                    amrex::Real const r_cell_im1 = rmin + (i - 0.5) * dr;
                    amrex::Real const r_cell_ip1 = rmin + (i + 1.5) * dr;

                    if (idim == 1) {
                        // B_theta (idim = 1)
                        // Couples only to B_theta.
                        // r-derivatives use et_z, z-derivatives use et_r.

                        // Radial neighbor i+1
                        PetscInt cp_r = gix_t(i+1, j, k);
                        if (cp_r >= 0) {
                            amrex::Real const chi_n = std::max(et_z(i+1, j, k), amrex::Real(0.0)) / m_mu0;
                            PetscScalar v = -m_theta_dt * chi_n / dr2 * (r_cell_ip1 / r_node_ip1);
                            cols.push_back(cp_r);
                            vals.push_back(v);
                            diag += m_theta_dt * chi_n / dr2 * (r_cell_i / r_node_ip1);
                        }

                        // Radial neighbor i-1
                        PetscInt cm_r = gix_t(i-1, j, k);
                        if (i == 0) {
                            // On-axis correction for J_z at i=0
                            amrex::Real const chi_0 = std::max(et_z(0, j, k), amrex::Real(0.0)) / m_mu0;
                            diag += m_theta_dt * chi_0 * 4.0 / dr2;
                        } else if (cm_r >= 0) {
                            amrex::Real const chi_n = std::max(et_z(i, j, k), amrex::Real(0.0)) / m_mu0;
                            PetscScalar v = -m_theta_dt * chi_n / dr2 * (r_cell_im1 / r_node_i);
                            cols.push_back(cm_r);
                            vals.push_back(v);
                            diag += m_theta_dt * chi_n / dr2 * (r_cell_i / r_node_i);
                        }

                        // Axial neighbors (z direction, only for RZ)
                        if (AMREX_SPACEDIM > 1) {
                            PetscInt cp_z = gix_t(i, j+1, k);
                            if (cp_z >= 0) {
                                amrex::Real const chi_n = std::max(et_r(i, j+1, k), amrex::Real(0.0)) / m_mu0;
                                PetscScalar v = -m_theta_dt * chi_n / dz2;
                                cols.push_back(cp_z);
                                vals.push_back(v);
                                diag -= v;
                            }
                            PetscInt cm_z = gix_t(i, j-1, k);
                            if (cm_z >= 0) {
                                amrex::Real const chi_n = std::max(et_r(i, j, k), amrex::Real(0.0)) / m_mu0;
                                PetscScalar v = -m_theta_dt * chi_n / dz2;
                                cols.push_back(cm_z);
                                vals.push_back(v);
                                diag -= v;
                            }
                        }
                    } else if (idim == 0) {
                        // B_r (idim = 0)
                        if (i == 0) {
                            // On axis, B_r is pinned to 0
                            // Do not add any off-diagonals, diag = 1.0.
                        } else {
                            // Axial neighbors (B_r to B_r)
                            if (AMREX_SPACEDIM > 1) {
                                PetscInt cp_z = gix_r(i, j+1, k);
                                if (cp_z >= 0) {
                                    amrex::Real const chi_n = std::max(et_t(i, j+1, k), amrex::Real(0.0)) / m_mu0;
                                    PetscScalar v = -m_theta_dt * chi_n / dz2;
                                    cols.push_back(cp_z);
                                    vals.push_back(v);
                                    diag -= v;
                                }
                                PetscInt cm_z = gix_r(i, j-1, k);
                                if (cm_z >= 0) {
                                    amrex::Real const chi_n = std::max(et_t(i, j, k), amrex::Real(0.0)) / m_mu0;
                                    PetscScalar v = -m_theta_dt * chi_n / dz2;
                                    cols.push_back(cm_z);
                                    vals.push_back(v);
                                    diag -= v;
                                }

                                // Cross terms to B_z
                                PetscInt cp_z_cp_r = gix_z(i, j+1, k);
                                if (cp_z_cp_r >= 0) {
                                    amrex::Real const chi_n = std::max(et_t(i, j+1, k), amrex::Real(0.0)) / m_mu0;
                                    PetscScalar v = m_theta_dt * chi_n / drdz;
                                    cols.push_back(cp_z_cp_r);
                                    vals.push_back(v);
                                }
                                PetscInt cm_z_cp_r = gix_z(i-1, j+1, k);
                                if (cm_z_cp_r >= 0) {
                                    amrex::Real const chi_n = std::max(et_t(i, j+1, k), amrex::Real(0.0)) / m_mu0;
                                    PetscScalar v = -m_theta_dt * chi_n / drdz;
                                    cols.push_back(cm_z_cp_r);
                                    vals.push_back(v);
                                }
                                PetscInt cp_z_cm_r = gix_z(i, j, k);
                                if (cp_z_cm_r >= 0) {
                                    amrex::Real const chi_n = std::max(et_t(i, j, k), amrex::Real(0.0)) / m_mu0;
                                    PetscScalar v = -m_theta_dt * chi_n / drdz;
                                    cols.push_back(cp_z_cm_r);
                                    vals.push_back(v);
                                }
                                PetscInt cm_z_cm_r = gix_z(i-1, j, k);
                                if (cm_z_cm_r >= 0) {
                                    amrex::Real const chi_n = std::max(et_t(i, j, k), amrex::Real(0.0)) / m_mu0;
                                    PetscScalar v = m_theta_dt * chi_n / drdz;
                                    cols.push_back(cm_z_cm_r);
                                    vals.push_back(v);
                                }
                            }
                        }
                    } else if (idim == 2) {
                        // B_z (idim = 2)
                        // Radial neighbors (B_z to B_z)
                        PetscInt cp_r = gix_z(i+1, j, k);
                        if (cp_r >= 0) {
                            amrex::Real const chi_n = std::max(et_t(i+1, j, k), amrex::Real(0.0)) / m_mu0;
                            PetscScalar v = -m_theta_dt * chi_n / dr2 * (r_node_ip1 / r_cell_i);
                            cols.push_back(cp_r);
                            vals.push_back(v);
                            diag -= v;
                        }
                        PetscInt cm_r = gix_z(i-1, j, k);
                        if (cm_r >= 0 && i > 0) { // If i=0, r_node_0 = 0, so v=0
                            amrex::Real const chi_n = std::max(et_t(i, j, k), amrex::Real(0.0)) / m_mu0;
                            PetscScalar v = -m_theta_dt * chi_n / dr2 * (r_node_i / r_cell_i);
                            cols.push_back(cm_r);
                            vals.push_back(v);
                            diag -= v;
                        }

                        // Cross terms to B_r
                        if (AMREX_SPACEDIM > 1) {
                            PetscInt cp_r_cp_z = gix_r(i+1, j, k);
                            if (cp_r_cp_z >= 0) {
                                amrex::Real const chi_n = std::max(et_t(i+1, j, k), amrex::Real(0.0)) / m_mu0;
                                PetscScalar v = m_theta_dt * chi_n / drdz * (r_node_ip1 / r_cell_i);
                                cols.push_back(cp_r_cp_z);
                                vals.push_back(v);
                            }
                            PetscInt cp_r_cm_z = gix_r(i+1, j-1, k);
                            if (cp_r_cm_z >= 0) {
                                amrex::Real const chi_n = std::max(et_t(i+1, j, k), amrex::Real(0.0)) / m_mu0;
                                PetscScalar v = -m_theta_dt * chi_n / drdz * (r_node_ip1 / r_cell_i);
                                cols.push_back(cp_r_cm_z);
                                vals.push_back(v);
                            }
                            PetscInt cm_r_cp_z = gix_r(i, j, k);
                            if (cm_r_cp_z >= 0 && i > 0) {
                                amrex::Real const chi_n = std::max(et_t(i, j, k), amrex::Real(0.0)) / m_mu0;
                                PetscScalar v = -m_theta_dt * chi_n / drdz * (r_node_i / r_cell_i);
                                cols.push_back(cm_r_cp_z);
                                vals.push_back(v);
                            }
                            PetscInt cm_r_cm_z = gix_r(i, j-1, k);
                            if (cm_r_cm_z >= 0 && i > 0) {
                                amrex::Real const chi_n = std::max(et_t(i, j, k), amrex::Real(0.0)) / m_mu0;
                                PetscScalar v = m_theta_dt * chi_n / drdz * (r_node_i / r_cell_i);
                                cols.push_back(cm_r_cm_z);
                                vals.push_back(v);
                            }
                        }
                    }

#else
                    // Expand C_up diag(eta/mu0) C_down directly. For each
                    // output curl term, q is either p+e_d or p; the nested curl
                    // contributes B(q)-B(q-e_e). This is the exact Yee stencil
                    // in 3D and naturally removes invariant directions in XZ
                    // and 1D_Z.
                    int const curl_component[3][2] = {
                        {2, 1}, {0, 2}, {1, 0}};
                    int const curl_direction[3][2] = {
                        {1, 2}, {2, 0}, {0, 1}};
                    int const curl_sign[2] = {1, -1};

                    auto shift = [] (int direction, int amount,
                                     int& ii, int& jj, int& kk) {
                        if (direction == 0) { ii += amount; }
                        else if (direction == 1) { jj += amount; }
                        else { kk += amount; }
                    };
                    auto add_value = [&] (PetscInt column, PetscScalar value) {
                        if (column < 0) { return; }
                        if (column == row) {
                            diag += value;
                            return;
                        }
                        auto const iter = std::find(cols.begin(), cols.end(), column);
                        if (iter == cols.end()) {
                            cols.push_back(column);
                            vals.push_back(value);
                        } else {
                            auto const index = static_cast<std::size_t>(iter - cols.begin());
                            vals[index] += value;
                        }
                    };

                    for (int outer_term = 0; outer_term < 2; ++outer_term) {
                        int const jcomp = curl_component[idim][outer_term];
                        int const outer_phys = curl_direction[idim][outer_term];
                        int const outer_dir = phys_to_amrex[outer_phys];
                        if (outer_dir < 0) { continue; }

                        for (int outer_side = 0; outer_side < 2; ++outer_side) {
                            int qi = i, qj = j, qk = k;
                            if (outer_side == 0) {
                                shift(outer_dir, 1, qi, qj, qk);
                            }
                            amrex::Real const outer_coeff =
                                curl_sign[outer_term] *
                                ((outer_side == 0) ? amrex::Real(1.0) : amrex::Real(-1.0)) /
                                spacing[outer_phys];
                            amrex::Real const chi =
                                std::max(eta_j[jcomp](qi, qj, qk), amrex::Real(0.0)) /
                                m_mu0;

                            for (int inner_term = 0; inner_term < 2; ++inner_term) {
                                int const bcomp = curl_component[jcomp][inner_term];
                                int const inner_phys = curl_direction[jcomp][inner_term];
                                int const inner_dir = phys_to_amrex[inner_phys];
                                if (inner_dir < 0) { continue; }

                                PetscScalar const value = m_theta_dt * chi * outer_coeff *
                                    curl_sign[inner_term] / spacing[inner_phys];
                                add_value(gix_b[bcomp](qi, qj, qk), value);

                                int mi = qi, mj = qj, mk = qk;
                                shift(inner_dir, -1, mi, mj, mk);
                                add_value(gix_b[bcomp](mi, mj, mk), -value);
                            }
                        }
                    }
#endif
                    cols.push_back(row);
                    vals.push_back(diag);
                    PetscInt const ncol = static_cast<PetscInt>(cols.size());
                    MAGDIFF_PETSC_CHK(MatSetValues(
                        m_P, 1, &row, ncol,
                        cols.data(), vals.data(), INSERT_VALUES));
                });
            }
        }
        PetscCall(MatAssemblyBegin(m_P, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(m_P, MAT_FINAL_ASSEMBLY));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

public:
    void update (
        amrex::Array<amrex::MultiFab const*,3> const& eta_edge,
        amrex::Real theta_dt,
        void* opctx)
    {
        m_theta_dt = theta_dt;
        m_opctx = opctx;

        buildEtaGhosts(eta_edge);

        // Drop stale PC factors before rewriting P (avoids use-after-free when
        // ILU/BJACOBI held internal refs into the previous P values).
        PC pc = nullptr;
        MAGDIFF_PETSC_CHK(KSPGetPC(m_ksp, &pc));
        MAGDIFF_PETSC_CHK(PCReset(pc));
        MAGDIFF_PETSC_CHK(assemblePreconditioner());
        MAGDIFF_PETSC_CHK(KSPSetOperators(m_ksp, m_A, m_P));
    }

    amrex::Geometry m_geom;
    amrex::Real m_theta_dt = amrex::Real(0.0);
    amrex::Real m_mu0 = amrex::Real(0.0);
    amrex::Real m_rtol = amrex::Real(0.0);
    amrex::Real m_atol = amrex::Real(0.0);
    int m_max_iter = 0;
    int m_verbose = 0;
    MagDiffMatvecFn m_matvec = nullptr;
    void* m_opctx = nullptr;

    PetscInt m_n_local = 0;
    PetscInt m_n_global = 0;
    amrex::Long m_rstart = 0;
    Vec m_x = nullptr;
    Vec m_b = nullptr;
    Mat m_A = nullptr;
    Mat m_P = nullptr;
    KSP m_ksp = nullptr;
    std::array<std::unique_ptr<amrex::iMultiFab>,3> m_gindex;
    amrex::Array<amrex::MultiFab,3> m_eta_g;
    // Host (pinned) copy of eb_update_B when EB is on. Empty when EB off.
    // Never LoopOnCpu into the live device eb_update_B MultiFabs.
    std::array<std::unique_ptr<amrex::iMultiFab>,3> m_eb_mask_host;
};

} // namespace

struct MagDiffPetscSolver { std::unique_ptr<MagDiffPetscSolverImpl> impl; };

MagDiffPetscSolver* magdiff_petsc_make (
    amrex::Array<amrex::MultiFab const*,3> const& B_proto,
    amrex::Array<amrex::MultiFab const*,3> const& eta_edge,
    amrex::Geometry const& geom, amrex::Real theta_dt, amrex::Real mu0,
    amrex::Real rtol, amrex::Real atol, int max_iter, int verbose,
    MagDiffPetscOptions const& options,
    MagDiffMatvecFn matvec, void* opctx,
    amrex::Array<amrex::iMultiFab const*,3> const* eb_update_B)
{
    auto* s = new MagDiffPetscSolver;
    s->impl = std::make_unique<MagDiffPetscSolverImpl>(
        B_proto, eta_edge, geom, theta_dt, mu0,
        rtol, atol, max_iter, verbose, options, matvec, opctx, eb_update_B);
    return s;
}

amrex::Long magdiff_petsc_nlocal (MagDiffPetscSolver const* s) {
    return s->impl->nlocal();
}
amrex::Long magdiff_petsc_rstart (MagDiffPetscSolver const* s) {
    return s->impl->rstart();
}
amrex::Array<amrex::iMultiFab const*,3>
magdiff_petsc_gindex (MagDiffPetscSolver const* s) {
    return s->impl->gindexView();
}

int magdiff_petsc_solve (MagDiffPetscSolver* s, amrex::Real const* rhs,
                         amrex::Real* sol, amrex::Real& residual_norm)
{
    return s->impl->solve(rhs, sol, residual_norm);
}

void magdiff_petsc_update (
    MagDiffPetscSolver* s,
    amrex::Array<amrex::MultiFab const*,3> const& eta_edge,
    amrex::Real theta_dt,
    void* opctx)
{
    s->impl->update(eta_edge, theta_dt, opctx);
}

void magdiff_petsc_destroy (MagDiffPetscSolver* s) {
    delete s;
}

#else // !AMREX_USE_PETSC

// Stubs so the translation unit links without PETSc. These are never called:
// HybridMagDiffusion.cpp only reaches them inside its own AMREX_USE_PETSC guard.

MagDiffPetscSolver* magdiff_petsc_make (
    amrex::Array<amrex::MultiFab const*,3> const& /*B_proto*/,
    amrex::Array<amrex::MultiFab const*,3> const& /*eta_edge*/,
    amrex::Geometry const& /*geom*/, amrex::Real /*theta_dt*/, amrex::Real /*mu0*/,
    amrex::Real, amrex::Real, int, int, MagDiffPetscOptions const&,
    MagDiffMatvecFn, void*,
    amrex::Array<amrex::iMultiFab const*,3> const*)
{
    amrex::Abort("magdiff_petsc_make: WarpX was not built with PETSc "
                 "(AMREX_USE_PETSC undefined).");
    return nullptr;
}

amrex::Long magdiff_petsc_nlocal (MagDiffPetscSolver const*) { return 0; }
amrex::Long magdiff_petsc_rstart (MagDiffPetscSolver const*) { return 0; }
amrex::Array<amrex::iMultiFab const*,3>
magdiff_petsc_gindex (MagDiffPetscSolver const*) { return {nullptr, nullptr, nullptr}; }

int magdiff_petsc_solve (MagDiffPetscSolver*, amrex::Real const*, amrex::Real*,
                         amrex::Real&) { return -1; }

void magdiff_petsc_update (
    MagDiffPetscSolver*,
    amrex::Array<amrex::MultiFab const*,3> const&,
    amrex::Real,
    void*) {}

void magdiff_petsc_destroy (MagDiffPetscSolver*) {}

#endif // AMREX_USE_PETSC
