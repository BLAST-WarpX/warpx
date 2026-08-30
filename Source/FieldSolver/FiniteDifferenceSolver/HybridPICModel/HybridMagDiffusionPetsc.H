/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Bowen Zhu
 *
 * License: BSD-3-Clause-LBNL
 *
 * PETSc-isolated API for the optional KSP/PC path of the matrix-free hybrid
 * magnetic-diffusion curl-curl solve. This header is intentionally pure AMReX
 * (no ablastr/WarpX headers): WarpX's own global `ReductionType` enum
 * (WarpXAlgorithmSelection.H) collides with PETSc's global `ReductionType`
 * (petscvec.h), so `petsc*.h` may only be included in a translation unit that
 * does not also include WarpX/ablastr headers (the WarpX_PETSc.cpp precedent).
 * HybridMagDiffusion.{H,cpp} call this API; the implementation lives in
 * HybridMagDiffusionPetsc.cpp (pure AMReX + PETSc).
 */
#ifndef WARPX_HYBRID_MAG_DIFFUSION_PETSC_H_
#define WARPX_HYBRID_MAG_DIFFUSION_PETSC_H_

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>

#include <string>

/** PETSc preconditioner defaults read from hybrid_pic_model.mag_diff_petsc_*. */
struct MagDiffPetscOptions
{
    // Empty keeps PETSc's default PC unless -pc_type is supplied externally.
    std::string pc_type;
    int asm_overlap = 0;
    std::string sub_ksp_type = "preonly";
    std::string sub_pc_type = "ilu";
    int ilu_factor_levels = 2;
};

/**
 * \brief Opaque PETSc solver handle (defined in HybridMagDiffusionPetsc.cpp).
 *
 * The matrix-free operator is supplied as a callback so the PETSc translation
 * unit never needs the WarpX-internal curl-curl operator. The callback receives
 * flat local arrays (n_local entries, component-major over the 3 B components,
 * interior nghost=0 cells) plus the shared global-DOF index map and row offset
 * so the caller can scatter/gather its field MultiFabs consistently with the
 * PETSc Vec layout.
 */
struct MagDiffPetscSolver;

using MagDiffMatvecFn = void (*)(
    void* opctx, amrex::Real const* x, amrex::Real* y,
    amrex::Array<amrex::iMultiFab const*,3> const& gindex, amrex::Long rstart);

/**
 * \brief Create the PETSc KSP. The 3 B-component MultiFabs (`B_proto`) define the
 * DOF layout (boxArray/DistributionMap/IndexType).
 *
 * `eta_edge` is the E/J-face resistivity used by both the matrix-free operator
 * and the assembled curl-curl Pmat. It is copied to nghost=1 host buffers as
 * needed. `mu0` keeps this header free of WarpX constants.
 *
 * Returns nullptr if built without PETSc (AMREX_USE_PETSC undefined).
 */
MagDiffPetscSolver* magdiff_petsc_make (
    amrex::Array<amrex::MultiFab const*,3> const& B_proto,
    amrex::Array<amrex::MultiFab const*,3> const& eta_edge,
    amrex::Geometry const& geom, amrex::Real theta_dt, amrex::Real mu0,
    amrex::Real rtol, amrex::Real atol, int max_iter, int verbose,
    MagDiffPetscOptions const& options,
    MagDiffMatvecFn matvec, void* opctx,
    amrex::Array<amrex::iMultiFab const*,3> const* eb_update_B = nullptr);

[[nodiscard]] amrex::Long magdiff_petsc_nlocal (MagDiffPetscSolver const* s);
[[nodiscard]] amrex::Long magdiff_petsc_rstart (MagDiffPetscSolver const* s);
[[nodiscard]] amrex::Array<amrex::iMultiFab const*,3>
magdiff_petsc_gindex (MagDiffPetscSolver const* s);

/**
 * \brief Solve A x = b with x and b as flat local arrays (n_local entries).
 * Seeds x with the initial guess and overwrites x with the solution. Returns
 * the PETSc KSP converged reason (positive on success, negative on divergence).
 */
int magdiff_petsc_solve (MagDiffPetscSolver* s,
                         amrex::Real const* rhs, amrex::Real* sol,
                         amrex::Real& residual_norm);

void magdiff_petsc_update (
    MagDiffPetscSolver* s,
    amrex::Array<amrex::MultiFab const*,3> const& eta_edge,
    amrex::Real theta_dt,
    void* opctx);

void magdiff_petsc_destroy (MagDiffPetscSolver* s);

#endif // WARPX_HYBRID_MAG_DIFFUSION_PETSC_H_
