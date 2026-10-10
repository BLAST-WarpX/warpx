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
#include <petscviewer.h>

// PETSc added these convenience macros after the 3.15 release shipped by
// Ubuntu 22.04. Keep the optional solver buildable against that distribution
// package while preserving the newer API when it is available.
#ifndef PETSC_SUCCESS
#define PETSC_SUCCESS 0
#endif
#ifndef PetscCall
#define PetscCall(...) do { \
    PetscErrorCode const ierr_petsc_call = (__VA_ARGS__); \
    CHKERRQ(ierr_petsc_call); \
} while (false)
#endif
#ifndef PetscCallAbort
#define PetscCallAbort(comm, ...) do { \
    PetscErrorCode const ierr_petsc_call_abort = (__VA_ARGS__); \
    CHKERRABORT(comm, ierr_petsc_call_abort); \
} while (false)
#endif
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>

#ifdef AMREX_USE_PETSC
#define MAGDIFF_PETSC_CHK(call) PetscCallAbort(PETSC_COMM_WORLD, call)
#endif


#ifdef AMREX_USE_PETSC

namespace {

static_assert(std::is_same_v<PetscScalar, amrex::Real>,
    "Hybrid magnetic diffusion requires PETSc and AMReX to use the same real precision");

/**
 * \brief PETSc KSP driver for the matrix-free mag-diff operator.
 *
 * DOF layout: component-major (B0, B1, B2), per-rank MFIter order, owned
 * valid (nghost=0) values only. Shared nodal and periodic aliases are one DOF.
 * A per-component global-index iMultiFab (-1 = non-owner/covered) maps owned
 * values to PETSc rows; a synchronized column map resolves aliases; the matvec
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
        amrex::Array<amrex::iMultiFab const*,3> const* eb_update_B,
        amrex::Array<amrex::MultiFab const*,3> const* boundary_response_B)
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
        copyBoundaryMapToHost(boundary_response_B);

        // A nodal value shared by boxes or periodic images is one unknown.
        // Count only AMReX owners; aliases are resolved separately for columns.
        auto const host_info = amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
        for (int idim = 0; idim < 3; ++idim) {
            auto owner = B_proto[idim]->OwnerMask(m_geom.periodicity());
            m_owner_host[idim] = std::make_unique<amrex::iMultiFab>(
                B_proto[idim]->boxArray(), B_proto[idim]->DistributionMap(),
                1, 0, host_info);
            amrex::iMultiFab::Copy(*m_owner_host[idim], *owner, 0, 0, 1, 0);
        }
        amrex::Gpu::synchronize();
        for (int idim = 0; idim < 3; ++idim) {
            for (amrex::MFIter mfi(*B_proto[idim]); mfi.isValid(); ++mfi) {
                auto const owner = m_owner_host[idim]->const_array(mfi);
                auto const mask = m_eb_mask_host[idim]
                    ? m_eb_mask_host[idim]->const_array(mfi)
                    : amrex::Array4<int const>{};
                auto const& box = mfi.validbox();
                amrex::LoopOnCpu(amrex::lbound(box), amrex::ubound(box),
                    [&] (int i, int j, int k) {
                        if (owner(i,j,k) != 0 && (!mask || mask(i,j,k) != 0)) {
                            ++m_n_local;
                        }
                    });
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
        if (m_n_global > 0 &&
            m_n_global - 1 > static_cast<PetscInt>(std::numeric_limits<int>::max()))
        {
            amrex::Abort(
                "Hybrid magnetic diffusion currently stores PETSc global indices in an "
                "AMReX iMultiFab and cannot represent more than INT_MAX + 1 unknowns");
        }

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
        MAGDIFF_PETSC_CHK(KSPSetOptionsPrefix(m_ksp, "magdiff_"));
        MAGDIFF_PETSC_CHK(KSPSetOperators(m_ksp, m_A, m_A));

        // Assembled frozen-η curl-curl Pmat (see assemblePreconditioner).
        // Runtime -magdiff_pc_type can select a PC for the assembled matrix.
        // PETSc otherwise defaults to block-Jacobi ILU(0). Preallocation follows
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
        MAGDIFF_PETSC_CHK(KSPSetInitialGuessNonzero(m_ksp, PETSC_FALSE));
        if (m_verbose > 0) {
            MAGDIFF_PETSC_CHK(KSPMonitorSet(
                m_ksp, monitorResidual, nullptr, nullptr));
        }
        setOptions(options);
        // Let -magdiff_ksp_type, -magdiff_pc_type, and related options win.
        MAGDIFF_PETSC_CHK(KSPSetFromOptions(m_ksp));
        MAGDIFF_PETSC_CHK(PetscOptionsGetBool(
            nullptr, nullptr, "-magdiff_check_finite", &m_check_finite, nullptr));
        if (m_check_finite) {
            amrex::Print() << "PETSC_AUDIT vector checks enabled\n";
        }
        PetscBool inspect_matrix = PETSC_FALSE;
        MAGDIFF_PETSC_CHK(PetscOptionsGetBool(
            nullptr, nullptr, "-magdiff_audit_matrix", &inspect_matrix, nullptr));
        if (inspect_matrix) { MAGDIFF_PETSC_CHK(auditMatrix()); }
        char dump_path[4096] = {};
        PetscBool dump_matrix = PETSC_FALSE;
        MAGDIFF_PETSC_CHK(PetscOptionsGetString(
            nullptr, nullptr, "-magdiff_dump_matrix", dump_path, sizeof(dump_path),
            &dump_matrix));
        if (dump_matrix) {
            PetscViewer viewer = nullptr;
            MAGDIFF_PETSC_CHK(PetscViewerBinaryOpen(
                PETSC_COMM_WORLD, dump_path, FILE_MODE_WRITE, &viewer));
            MAGDIFF_PETSC_CHK(MatView(m_P, viewer));
            MAGDIFF_PETSC_CHK(PetscViewerDestroy(&viewer));
        }
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
        MAGDIFF_PETSC_CHK(auditVector(m_b, "right-hand side"));

        MAGDIFF_PETSC_CHK(KSPSolve(m_ksp, m_b, m_x));

        PetscInt iters = 0;
        KSPConvergedReason reason = static_cast<KSPConvergedReason>(0);
        PetscReal rnorm = 0.0;
        MAGDIFF_PETSC_CHK(KSPGetIterationNumber(m_ksp, &iters));
        MAGDIFF_PETSC_CHK(KSPGetConvergedReason(m_ksp, &reason));
        MAGDIFF_PETSC_CHK(KSPGetResidualNorm(m_ksp, &rnorm));
        rnorm_out = static_cast<amrex::Real>(rnorm);

        if (reason <= 0 || !std::isfinite(rnorm_out)) {
            std::ostringstream msg;
            msg << "Hybrid magnetic-diffusion PETSc solve failed: reason="
                << static_cast<int>(reason) << ", iterations=" << iters
                << ", residual=" << rnorm;
            amrex::Abort(msg.str());
        }

        PetscScalar* xarr = nullptr;
        MAGDIFF_PETSC_CHK(VecGetArray(m_x, &xarr));
        bool solution_is_finite = true;
        for (amrex::Long i = 0; i < m_n_local; ++i) {
            solution_is_finite = solution_is_finite && std::isfinite(xarr[i]);
            sol[i] = static_cast<amrex::Real>(xarr[i]);
        }
        MAGDIFF_PETSC_CHK(VecRestoreArray(m_x, &xarr));
        if (!solution_is_finite) {
            amrex::Abort("Hybrid magnetic-diffusion PETSc solve returned a non-finite solution");
        }

        if (m_verbose > 0) {
            amrex::Print() << "HybridMagDiffusion PETSc KSP iterations=" << iters
                           << " residual=" << rnorm
                           << " reason=" << reason << "\n";
        }
        return static_cast<int>(reason);
    }

private:
    PetscErrorCode auditMatrix () {
        PetscFunctionBeginUser;
        Vec input = nullptr, assembled = nullptr, applied = nullptr;
        PetscCall(VecDuplicate(m_x, &input));
        PetscCall(VecDuplicate(m_x, &assembled));
        PetscCall(VecDuplicate(m_x, &applied));
        PetscScalar* values = nullptr;
        PetscCall(VecGetArray(input, &values));
        for (amrex::Long i = 0; i < m_n_local; ++i) {
            auto const index = static_cast<amrex::Real>(m_rstart + i);
            values[i] = std::sin(index * 0.031) + std::cos(index * 0.017);
        }
        PetscCall(VecRestoreArray(input, &values));
        PetscCall(MatMult(m_P, input, assembled));
        PetscCall(MatMult(m_A, input, applied));
        PetscScalar const *x_values = nullptr, *p_values = nullptr, *a_values = nullptr;
        PetscCall(VecGetArrayRead(input, &x_values));
        PetscCall(VecGetArrayRead(assembled, &p_values));
        PetscCall(VecGetArrayRead(applied, &a_values));
        amrex::Real boundary_p_error = 0.0, boundary_a_error = 0.0;
        amrex::Long clamped_rows = 0;
        for (int idim = 0; idim < 3; ++idim) {
            if (!m_boundary_map_host[idim]) { continue; }
            for (amrex::MFIter mfi(*m_gindex[idim]); mfi.isValid(); ++mfi) {
                auto const indices = m_gindex[idim]->const_array(mfi);
                auto const active = m_boundary_map_host[idim]->const_array(mfi);
                auto const& box = mfi.validbox();
                amrex::LoopOnCpu(amrex::lbound(box), amrex::ubound(box),
                    [&] (int i, int j, int k) {
                        if (indices(i,j,k) < 0 || std::abs(active(i,j,k)) > 0.5) { return; }
                        auto const index = static_cast<amrex::Long>(indices(i,j,k))-m_rstart;
                        boundary_p_error = std::max(boundary_p_error,
                            std::abs(p_values[index]-x_values[index]));
                        boundary_a_error = std::max(boundary_a_error,
                            std::abs(a_values[index]-x_values[index]));
                        ++clamped_rows;
                    });
            }
        }
        PetscCall(VecRestoreArrayRead(input, &x_values));
        PetscCall(VecRestoreArrayRead(assembled, &p_values));
        PetscCall(VecRestoreArrayRead(applied, &a_values));
        amrex::ParallelDescriptor::ReduceRealMax(boundary_p_error);
        amrex::ParallelDescriptor::ReduceRealMax(boundary_a_error);
        amrex::ParallelDescriptor::ReduceLongSum(clamped_rows);
        amrex::Print() << "PETSC_AUDIT clamped_rows=" << clamped_rows
            << " P_error=" << boundary_p_error << " A_error=" << boundary_a_error << "\n";
        if (boundary_p_error > 1.e-12 || boundary_a_error > 1.e-12) {
            amrex::Abort("PETSc boundary identity rows disagree with the native operator");
        }
        PetscReal applied_norm = 0.0, mismatch = 0.0, max_row_sum = 0.0;
        PetscCall(VecNorm(applied, NORM_2, &applied_norm));
        PetscCall(VecAXPY(assembled, -1.0, applied));
        PetscCall(VecNorm(assembled, NORM_2, &mismatch));
        PetscCall(VecAbs(assembled));
        PetscInt worst_index = 0;
        PetscReal worst_difference = 0.0;
        PetscCall(VecMax(assembled, &worst_index, &worst_difference));
        for (int idim = 0; idim < 3; ++idim) {
            for (amrex::MFIter mfi(*m_gindex[idim]); mfi.isValid(); ++mfi) {
                auto const& indices = m_gindex[idim]->const_array(mfi);
                auto const& box = mfi.validbox();
                amrex::LoopOnCpu(amrex::lbound(box), amrex::ubound(box),
                    [&] (int i, int j, int k) {
                        if (indices(i, j, k) == worst_index) {
                            amrex::AllPrint() << "PETSC_AUDIT largest_difference="
                                << worst_difference << " row=" << worst_index
                                << " component=" << idim << " index="
                                << i << "," << j << "," << k << " box=" << box << "\n";
                        }
                    });
            }
        }
        PetscCall(MatNorm(m_P, NORM_INFINITY, &max_row_sum));
        PetscCall(MatGetDiagonal(m_P, assembled));
        PetscReal minimum = 0.0, maximum = 0.0;
        PetscCall(VecMin(assembled, nullptr, &minimum));
        PetscCall(VecMax(assembled, nullptr, &maximum));
        amrex::Print() << "PETSC_AUDIT matrix diagonal_min=" << minimum
            << " diagonal_max=" << maximum << " max_row_sum=" << max_row_sum
            << " relative_Pmat_vs_shell=" << mismatch / applied_norm << "\n";
        char report_path[4096] = {};
        PetscBool report_requested = PETSC_FALSE;
        PetscCall(PetscOptionsGetString(nullptr, nullptr, "-magdiff_audit_file",
            report_path, sizeof(report_path), &report_requested));
        if (report_requested && amrex::ParallelDescriptor::IOProcessor()) {
            std::ofstream report(report_path);
            report.precision(17);
            report << "{\"clamped_rows\":" << clamped_rows
                << ",\"boundary_P_error\":" << boundary_p_error
                << ",\"boundary_A_error\":" << boundary_a_error
                << ",\"relative_Pmat_vs_shell\":" << mismatch/applied_norm
                << ",\"diagonal_min\":" << minimum
                << ",\"diagonal_max\":" << maximum << "}\n";
            if (!report) { amrex::Abort("Cannot write PETSc operator audit report"); }
        }
        PetscCall(VecDestroy(&input));
        PetscCall(VecDestroy(&assembled));
        PetscCall(VecDestroy(&applied));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode auditVector (Vec vector, char const* stage) const {
        PetscFunctionBeginUser;
        if (!m_check_finite) { PetscFunctionReturn(PETSC_SUCCESS); }
        PetscScalar const* values = nullptr;
        PetscCall(VecGetArrayRead(vector, &values));
        amrex::Long first_bad = -1;
        amrex::Real bad_value = 0.0;
        for (amrex::Long i = 0; i < m_n_local; ++i) {
            if (!std::isfinite(values[i])) {
                first_bad = i;
                bad_value = values[i];
                break;
            }
        }
        PetscCall(VecRestoreArrayRead(vector, &values));
        bool invalid = first_bad >= 0;
        amrex::ParallelDescriptor::ReduceBoolOr(invalid);
        if (invalid) {
            if (first_bad >= 0) {
                amrex::AllPrint() << "MAGDIFF_FINITE stage=" << stage
                    << " global_index=" << m_rstart + first_bad
                    << " value=" << bad_value << "\n";
            }
            amrex::Abort(std::string("Non-finite PETSc vector at ") + stage);
        }
        PetscReal norm = 0.0, maximum = 0.0;
        PetscCall(VecNorm(vector, NORM_2, &norm));
        PetscCall(VecNorm(vector, NORM_INFINITY, &maximum));
        if (m_audit_calls++ < 16 || !std::isfinite(norm)) {
            amrex::Print() << "PETSC_AUDIT stage=" << stage
                << " norm=" << norm << " max_abs=" << maximum << "\n";
        }
        if (!std::isfinite(norm)) {
            amrex::Abort(std::string("Non-finite PETSc vector norm at ") + stage);
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    void setOptions (MagDiffPetscOptions const& options)
    {
        auto set_default = [] (char const* name, std::string const& value) {
            PetscBool specified = PETSC_FALSE;
            MAGDIFF_PETSC_CHK(PetscOptionsHasName(nullptr, nullptr, name, &specified));
            if (!specified && !value.empty()) {
                MAGDIFF_PETSC_CHK(PetscOptionsSetValue(nullptr, name, value.c_str()));
            }
        };

        set_default("-magdiff_pc_type", options.pc_type);
        set_default("-magdiff_pc_asm_overlap", std::to_string(options.asm_overlap));
        set_default("-magdiff_sub_ksp_type", options.sub_ksp_type);
        set_default("-magdiff_sub_pc_type", options.sub_pc_type);
        if (options.sub_pc_type == "ilu") {
            set_default("-magdiff_sub_pc_factor_levels",
                        std::to_string(options.ilu_factor_levels));
        }
    }

    static PetscErrorCode applyMatOp (Mat A, Vec in, Vec out) {
        PetscFunctionBeginUser;
        MagDiffPetscSolverImpl* self = nullptr;
        PetscCall(MatShellGetContext(A, reinterpret_cast<void**>(&self)));
        PetscCall(self->auditVector(in, "matrix-free operator input"));
        const PetscScalar* x = nullptr; PetscScalar* y = nullptr;
        PetscCall(VecGetArrayRead(in, &x));
        PetscCall(VecGetArray(out, &y));
        self->m_matvec(self->m_opctx,
                       static_cast<amrex::Real const*>(x),
                       static_cast<amrex::Real*>(y),
                       self->gindexView(), self->m_rstart);
        PetscCall(VecRestoreArrayRead(in, &x));
        PetscCall(VecRestoreArray(out, &y));
        PetscCall(self->auditVector(out, "matrix-free operator output"));
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

    // Retain the native homogeneous boundary response on pinned host memory
    // for safe CPU matrix assembly, including GPU builds.
    int copyBoundaryMapToHost (
        amrex::Array<amrex::MultiFab const*,3> const* masks)
    {
        int changed = 0;
        if (masks == nullptr) {
            for (auto& mask : m_boundary_map_host) {
                if (mask) { changed = 1; }
                mask.reset();
            }
            return changed;
        }
        amrex::Array<std::unique_ptr<amrex::MultiFab>,3> incoming;
        auto const info = amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
        for (int idim = 0; idim < 3; ++idim) {
            incoming[idim] = std::make_unique<amrex::MultiFab>(
                (*masks)[idim]->boxArray(), (*masks)[idim]->DistributionMap(),
                1, 1, info);
            amrex::MultiFab::Copy(*incoming[idim], *(*masks)[idim], 0, 0, 1, 1);
        }
        amrex::Gpu::synchronize();
        for (int idim = 0; idim < 3; ++idim) {
            if (!m_boundary_map_host[idim]) {
                changed = 1;
            } else {
                for (amrex::MFIter mfi(*incoming[idim]); mfi.isValid(); ++mfi) {
                    auto const& previous = (*m_boundary_map_host[idim])[mfi];
                    auto const& current = (*incoming[idim])[mfi];
                    if (std::memcmp(previous.dataPtr(), current.dataPtr(),
                                    static_cast<std::size_t>(current.size())*sizeof(amrex::Real)) != 0) {
                        changed = 1;
                        break;
                    }
                }
            }
            m_boundary_map_host[idim] = std::move(incoming[idim]);
        }
        return changed;
    }

    struct BoundaryColumn
    {
        PetscInt index = -1;
        amrex::Real weight = 0.0;
        operator PetscInt () const { return index; }
    };

    struct BoundaryColumnIndex
    {
        amrex::Array4<int const> indices;
        amrex::Array4<amrex::Real const> response;
        amrex::Box domain;
        BoundaryColumn operator() (int i, int j, int k) const {
            int const column = indices(i,j,k);
            if (column >= 0) {
                if (response && std::abs(response(i,j,k)) < 0.5) { return {}; }
                return {column, 1.0};
            }
            amrex::IntVect point(AMREX_D_DECL(i,j,k));
            if (domain.contains(point) || !response) { return {}; }
            amrex::Real const weight = response(i,j,k);
            if (std::abs(weight) < 1.e-12) { return {}; }
            // Yee curl-curl references exterior B only in a direction where
            // that component is cell centered (tangential to the boundary).
            // The native first guard is either fixed or a signed mirror.
            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
                if (point[direction] < domain.smallEnd(direction)) {
                    AMREX_ALWAYS_ASSERT(!domain.ixType().nodeCentered(direction));
                    point[direction] = 2*domain.smallEnd(direction)-1-point[direction];
                } else if (point[direction] > domain.bigEnd(direction)) {
                    AMREX_ALWAYS_ASSERT(!domain.ixType().nodeCentered(direction));
                    point[direction] = 2*domain.bigEnd(direction)+1-point[direction];
                }
            }
            int const reflected = indices(point[0], AMREX_D_PICK(0,point[1],point[1]), AMREX_D_PICK(0,0,point[2]));
            if (reflected < 0 || std::abs(response(point[0], AMREX_D_PICK(0,point[1],point[1]), AMREX_D_PICK(0,0,point[2]))) < 0.5) {
                return {};
            }
            return {reflected, weight};
        }
    };

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

    // Owner-only global DOF indices (component-major); -1 marks non-owners,
    // exterior values and covered B DOFs. A separate synchronized column map
    // resolves neighboring values and periodic aliases. Pinned host arena: PETSc scatter/gather
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
        for (int idim = 0; idim < 3; ++idim) {
            for (amrex::MFIter mfi(*B_proto[idim]); mfi.isValid(); ++mfi) {
                auto const owner = m_owner_host[idim]->const_array(mfi);
                auto const mask = m_eb_mask_host[idim]
                    ? m_eb_mask_host[idim]->const_array(mfi)
                    : amrex::Array4<int const>{};
                auto const gix = m_gindex[idim]->array(mfi);
                auto const& box = mfi.validbox();
                amrex::LoopOnCpu(amrex::lbound(box), amrex::ubound(box),
                    [&] (int i, int j, int k) {
                        if (owner(i,j,k) == 0 || (mask && mask(i,j,k) == 0)) { return; }
                        gix(i,j,k) = static_cast<int>(run++);
                    });
            }
        }
        AMREX_ALWAYS_ASSERT(run == m_rstart + m_n_local);
        for (int idim = 0; idim < 3; ++idim) {
            m_column_index[idim] = std::make_unique<amrex::iMultiFab>(
                B_proto[idim]->boxArray(), B_proto[idim]->DistributionMap(),
                1, amrex::IntVect::Unit, host_info);
            amrex::iMultiFab::Copy(*m_column_index[idim], *m_gindex[idim], 0, 0, 1, 1);
            // Preserve the owner-only row map for scatter/gather, while the
            // column map resolves all nodal aliases and neighboring ghosts.
            m_column_index[idim]->OverrideSync(m_geom.periodicity());
            m_column_index[idim]->FillBoundary(m_geom.periodicity());
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

    // Assemble frozen-η curl-curl Pmat; the operator stays matrix-free.
    // Native boundary responses clamp valid DOFs, eliminate prescribed guards,
    // or fold reflected guards into their interior columns.
    //
    // RZ / RCYLINDER: discrete curl-curl stencil from the staggered Yee/RZ
    // operator — uncoupled Bt 5-point block; Br and Bz coupled via J_θ mixed
    // derivatives (~7-point). Axis: Br(0,*) is diagonal-only (pin); Bt on-axis
    // uses the 4/dr^2 J_z correction. η is face-selected per term (et_r/t/z).
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
                auto boundary_index = [&] (int component) {
                    auto domain = m_geom.Domain();
                    domain.convert(m_gindex[component]->ixType());
                    return BoundaryColumnIndex{m_column_index[component]->const_array(mfi),
                        m_boundary_map_host[component]
                            ? m_boundary_map_host[component]->const_array(mfi)
                            : amrex::Array4<amrex::Real const>{}, domain};
                };
                auto const gix_r = boundary_index(0);
                auto const gix_t = boundary_index(1);
                auto const gix_z = boundary_index(2);
                auto const& et_r = m_eta_g[0].const_array(mfi);
                auto const& et_t = m_eta_g[1].const_array(mfi);
                auto const& et_z = m_eta_g[2].const_array(mfi);
#else
                auto boundary_index = [&] (int component) {
                    auto domain = m_geom.Domain();
                    domain.convert(m_gindex[component]->ixType());
                    return BoundaryColumnIndex{m_column_index[component]->const_array(mfi),
                        m_boundary_map_host[component]
                            ? m_boundary_map_host[component]->const_array(mfi)
                            : amrex::Array4<amrex::Real const>{}, domain};
                };
                amrex::Array<BoundaryColumnIndex,3> const gix_b{
                    boundary_index(0), boundary_index(1), boundary_index(2)};
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
                    if (m_boundary_map_host[idim] &&
                        std::abs(m_boundary_map_host[idim]->const_array(mfi)(i,j,k)) < 0.5) {
                        // The native PEC map clamps this normal B DOF before
                        // either curl. Its homogeneous diffusion row is zero:
                        // the full theta-method equation is an identity row.
                        MAGDIFF_PETSC_CHK(MatSetValue(m_P, row, row, 1.0, INSERT_VALUES));
                        return;
                    }
                    PetscScalar diag = 1.0;
                    cols.clear();
                    vals.clear();
                    auto add_value = [&] (BoundaryColumn term, PetscScalar value) {
                        PetscInt const column = term.index;
                        value *= term.weight;
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
                            auto const index = static_cast<std::size_t>(iter-cols.begin());
                            vals[index] += value;
                        }
                    };

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
                        auto const cp_r = gix_t(i+1, j, k);
                        {
                            amrex::Real const chi_n = std::max(et_z(i+1, j, k), amrex::Real(0.0)) / m_mu0;
                            PetscScalar v = -m_theta_dt * chi_n / dr2 * (r_cell_ip1 / r_node_ip1);
                            add_value(cp_r, v);
                            diag += m_theta_dt * chi_n / dr2 * (r_cell_i / r_node_ip1);
                        }

                        // Radial neighbor i-1
                        auto const cm_r = gix_t(i-1, j, k);
                        if (i == 0) {
                            // On-axis correction for J_z at i=0
                            amrex::Real const chi_0 = std::max(et_z(0, j, k), amrex::Real(0.0)) / m_mu0;
                            diag += m_theta_dt * chi_0 * 4.0 / dr2;
                        } else {
                            amrex::Real const chi_n = std::max(et_z(i, j, k), amrex::Real(0.0)) / m_mu0;
                            PetscScalar v = -m_theta_dt * chi_n / dr2 * (r_cell_im1 / r_node_i);
                            add_value(cm_r, v);
                            diag += m_theta_dt * chi_n / dr2 * (r_cell_i / r_node_i);
                        }

                        // Axial neighbors (z direction, only for RZ)
                        if (AMREX_SPACEDIM > 1) {
                            auto const cp_z = gix_t(i, j+1, k);
                            {
                                amrex::Real const chi_n = std::max(et_r(i, j+1, k), amrex::Real(0.0)) / m_mu0;
                                PetscScalar v = -m_theta_dt * chi_n / dz2;
                                add_value(cp_z, v);
                                diag -= v;
                            }
                            auto const cm_z = gix_t(i, j-1, k);
                            {
                                amrex::Real const chi_n = std::max(et_r(i, j, k), amrex::Real(0.0)) / m_mu0;
                                PetscScalar v = -m_theta_dt * chi_n / dz2;
                                add_value(cm_z, v);
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
                                auto const cp_z = gix_r(i, j+1, k);
                                {
                                    amrex::Real const chi_n = std::max(et_t(i, j+1, k), amrex::Real(0.0)) / m_mu0;
                                    PetscScalar v = -m_theta_dt * chi_n / dz2;
                                    add_value(cp_z, v);
                                    diag -= v;
                                }
                                auto const cm_z = gix_r(i, j-1, k);
                                {
                                    amrex::Real const chi_n = std::max(et_t(i, j, k), amrex::Real(0.0)) / m_mu0;
                                    PetscScalar v = -m_theta_dt * chi_n / dz2;
                                    add_value(cm_z, v);
                                    diag -= v;
                                }

                                // Cross terms to B_z
                                auto const cp_z_cp_r = gix_z(i, j+1, k);
                                if (cp_z_cp_r >= 0) {
                                    amrex::Real const chi_n = std::max(et_t(i, j+1, k), amrex::Real(0.0)) / m_mu0;
                                    PetscScalar v = m_theta_dt * chi_n / drdz;
                                    add_value(cp_z_cp_r, v);
                                }
                                auto const cm_z_cp_r = gix_z(i-1, j+1, k);
                                if (cm_z_cp_r >= 0) {
                                    amrex::Real const chi_n = std::max(et_t(i, j+1, k), amrex::Real(0.0)) / m_mu0;
                                    PetscScalar v = -m_theta_dt * chi_n / drdz;
                                    add_value(cm_z_cp_r, v);
                                }
                                auto const cp_z_cm_r = gix_z(i, j, k);
                                if (cp_z_cm_r >= 0) {
                                    amrex::Real const chi_n = std::max(et_t(i, j, k), amrex::Real(0.0)) / m_mu0;
                                    PetscScalar v = -m_theta_dt * chi_n / drdz;
                                    add_value(cp_z_cm_r, v);
                                }
                                auto const cm_z_cm_r = gix_z(i-1, j, k);
                                if (cm_z_cm_r >= 0) {
                                    amrex::Real const chi_n = std::max(et_t(i, j, k), amrex::Real(0.0)) / m_mu0;
                                    PetscScalar v = m_theta_dt * chi_n / drdz;
                                    add_value(cm_z_cm_r, v);
                                }
                            }
                        }
                    } else if (idim == 2) {
                        // B_z (idim = 2)
                        // Radial neighbors (B_z to B_z)
                        auto const cp_r = gix_z(i+1, j, k);
                        {
                            amrex::Real const chi_n = std::max(et_t(i+1, j, k), amrex::Real(0.0)) / m_mu0;
                            PetscScalar v = -m_theta_dt * chi_n / dr2 * (r_node_ip1 / r_cell_i);
                            add_value(cp_r, v);
                            diag -= v;
                        }
                        auto const cm_r = gix_z(i-1, j, k);
                        if (i > 0) { // r_node_0 = 0
                            amrex::Real const chi_n = std::max(et_t(i, j, k), amrex::Real(0.0)) / m_mu0;
                            PetscScalar v = -m_theta_dt * chi_n / dr2 * (r_node_i / r_cell_i);
                            add_value(cm_r, v);
                            diag -= v;
                        }

                        // Cross terms to B_r
                        if (AMREX_SPACEDIM > 1) {
                            auto const cp_r_cp_z = gix_r(i+1, j, k);
                            if (cp_r_cp_z >= 0) {
                                amrex::Real const chi_n = std::max(et_t(i+1, j, k), amrex::Real(0.0)) / m_mu0;
                                PetscScalar v = m_theta_dt * chi_n / drdz * (r_node_ip1 / r_cell_i);
                                add_value(cp_r_cp_z, v);
                            }
                            auto const cp_r_cm_z = gix_r(i+1, j-1, k);
                            if (cp_r_cm_z >= 0) {
                                amrex::Real const chi_n = std::max(et_t(i+1, j, k), amrex::Real(0.0)) / m_mu0;
                                PetscScalar v = -m_theta_dt * chi_n / drdz * (r_node_ip1 / r_cell_i);
                                add_value(cp_r_cm_z, v);
                            }
                            auto const cm_r_cp_z = gix_r(i, j, k);
                            if (cm_r_cp_z >= 0 && i > 0) {
                                amrex::Real const chi_n = std::max(et_t(i, j, k), amrex::Real(0.0)) / m_mu0;
                                PetscScalar v = -m_theta_dt * chi_n / drdz * (r_node_i / r_cell_i);
                                add_value(cm_r_cp_z, v);
                            }
                            auto const cm_r_cm_z = gix_r(i, j-1, k);
                            if (cm_r_cm_z >= 0 && i > 0) {
                                amrex::Real const chi_n = std::max(et_t(i, j, k), amrex::Real(0.0)) / m_mu0;
                                PetscScalar v = m_theta_dt * chi_n / drdz * (r_node_i / r_cell_i);
                                add_value(cm_r_cm_z, v);
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
        void* opctx,
        amrex::Array<amrex::MultiFab const*,3> const* boundary_response_B)
    {
        copyBoundaryMapToHost(boundary_response_B);
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
    PetscBool m_check_finite = PETSC_FALSE;
    mutable int m_audit_calls = 0;
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
    std::array<std::unique_ptr<amrex::iMultiFab>,3> m_column_index;
    std::array<std::unique_ptr<amrex::iMultiFab>,3> m_owner_host;
    amrex::Array<amrex::MultiFab,3> m_eta_g;
    // Host (pinned) copy of eb_update_B when EB is on. Empty when EB off.
    // Never LoopOnCpu into the live device eb_update_B MultiFabs.
    std::array<std::unique_ptr<amrex::iMultiFab>,3> m_eb_mask_host;
    std::array<std::unique_ptr<amrex::MultiFab>,3> m_boundary_map_host;
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
    amrex::Array<amrex::iMultiFab const*,3> const* eb_update_B,
    amrex::Array<amrex::MultiFab const*,3> const* boundary_response_B)
{
    auto* s = new MagDiffPetscSolver;
    s->impl = std::make_unique<MagDiffPetscSolverImpl>(
        B_proto, eta_edge, geom, theta_dt, mu0,
        rtol, atol, max_iter, verbose, options, matvec, opctx, eb_update_B, boundary_response_B);
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
    void* opctx,
    amrex::Array<amrex::MultiFab const*,3> const* boundary_response_B)
{
    s->impl->update(eta_edge, theta_dt, opctx, boundary_response_B);
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
    amrex::Array<amrex::iMultiFab const*,3> const*,
    amrex::Array<amrex::MultiFab const*,3> const*)
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
    void*, amrex::Array<amrex::MultiFab const*,3> const*) {}

void magdiff_petsc_destroy (MagDiffPetscSolver*) {}

#endif // AMREX_USE_PETSC
