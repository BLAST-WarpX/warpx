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
#include <AMReX_Enum.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>

/**
 * \brief Preconditioner choice for the optional PETSc mag-diffusion KSP path.
 * Declared outside any AMREX_USE_PETSC guard so non-PETSc builds still parse it
 * (same convention as PreconditionerLibrary.H).
 *
 * - shell_jacobi: scaled Jacobi shell PC (callback from HybridMagDiffusion).
 * - exact_curl_curl: assembled frozen-η curl-curl Pmat (RZ/Rcyl exact stencil
 *   with Br–Bz coupling; Cartesian remains the face-averaged Laplacian proxy).
 * - frozen_laplacian: deprecated input alias for exact_curl_curl (same value).
 */
AMREX_ENUM(MagDiffPetscPC,
    shell_jacobi,
    exact_curl_curl,
    frozen_laplacian = 1
);

/**
 * \brief Opaque PETSc solver handle (defined in HybridMagDiffusionPetsc.cpp).
 *
 * Operator (matvec) and preconditioner are supplied as callbacks so the PETSc
 * translation unit never needs the WarpX-internal curl-curl operator. Each
 * callback receives flat local arrays (n_local entries, component-major over
 * the 3 B components, interior nghost=0 cells) plus the shared global-DOF index
 * map and row offset so the caller can scatter/gather its field MultiFabs
 * consistently with the PETSc Vec layout.
 */
struct MagDiffPetscSolver;

using MagDiffMatvecFn = void (*)(
    void* opctx, amrex::Real const* x, amrex::Real* y,
    amrex::Array<amrex::iMultiFab const*,3> const& gindex, amrex::Long rstart);

using MagDiffPCFn = void (*)(
    void* opctx, amrex::Real const* x, amrex::Real* y,
    amrex::Array<amrex::iMultiFab const*,3> const& gindex, amrex::Long rstart);

/**
 * \brief Create the PETSc KSP. The 3 B-component MultiFabs (`B_proto`) define the
 * DOF layout (boxArray/DistributionMap/IndexType).
 *
 * `eta_edge` is the E/J-face resistivity (matvec staggering); on RZ/Rcyl the
 * assembled exact_curl_curl Pmat is built from these faces. `eta_pc` is the
 * B-sampled resistivity used for the Cartesian Laplacian proxy Pmat (and kept
 * for API symmetry with the Jacobi shell path). Both are copied to nghost=1
 * host buffers as needed. `mu0` keeps this header free of WarpX constants.
 *
 * Returns nullptr if built without PETSc (AMREX_USE_PETSC undefined).
 */
MagDiffPetscSolver* magdiff_petsc_make (
    amrex::Array<amrex::MultiFab const*,3> const& B_proto,
    amrex::Array<amrex::MultiFab const*,3> const& eta_edge,
    amrex::Array<amrex::MultiFab const*,3> const& eta_pc,
    amrex::Geometry const& geom, amrex::Real theta_dt, amrex::Real mu0,
    MagDiffPetscPC pc_choice,
    amrex::Real rtol, amrex::Real atol, int max_iter, int verbose,
    MagDiffMatvecFn matvec, MagDiffPCFn pcapply, void* opctx,
    amrex::Array<amrex::iMultiFab const*,3> const* eb_update_B = nullptr);

[[nodiscard]] amrex::Long magdiff_petsc_nlocal (MagDiffPetscSolver const* s);
[[nodiscard]] amrex::Long magdiff_petsc_rstart (MagDiffPetscSolver const* s);
[[nodiscard]] amrex::Array<amrex::iMultiFab const*,3>
magdiff_petsc_gindex (MagDiffPetscSolver const* s);

/**
 * \brief Solve A x = b with x and b as flat local arrays (n_local entries).
 * Seeds x with the initial guess, overwrites x with the solution. Returns the
 * KSP iteration count (>=0) on success, or a negative PETSc converged reason on
 * divergence.
 */
int magdiff_petsc_solve (MagDiffPetscSolver* s,
                         amrex::Real const* rhs, amrex::Real* sol,
                         amrex::Real& residual_norm);

void magdiff_petsc_destroy (MagDiffPetscSolver* s);

#endif // WARPX_HYBRID_MAG_DIFFUSION_PETSC_H_
