/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Bowen Zhu
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_HYBRID_MAG_DIFFUSION_H_
#define WARPX_HYBRID_MAG_DIFFUSION_H_

#include <ablastr/fields/MultiFabRegister.H>

#include "HybridMagDiffusionPetsc.H"  // MagDiffPetscPC (PETSc-free enum)

#include <AMReX_Array.H>
#include <AMReX_Enum.H>
#include <AMReX_Geometry.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

// Solver driver for the matrix-free variable-eta / RZ curl-curl path. Default
// is AMReX GMRES + scaled Jacobi; `petsc` routes the same matrix-free operator
// through a PETSc KSP (MATSHELL matvec) when WarpX is built with PETSc. The enum
// is declared outside the AMREX_USE_PETSC guard so non-PETSc builds still parse
// it (see PreconditionerLibrary.H for the same pattern).
AMREX_ENUM(MagDiffLinearSolver,
    amrex_gmres,
    petsc
);

/**
 * \brief Implicit magnetic diffusion for hybrid-PIC (operator-split).
 *
 * Advances the Yee magnetic field with a theta-method discretization of
 *
 *   dB/dt = - curl( (eta/mu0) curl B )
 *
 * after each hybrid B half-step when enabled. Resistivity may be constant
 * or spatially varying (parser / MultiFab).
 *
 * Implementation: single matrix-free path for all geometries (FDTD
 * curl(eta curl B)/mu0 + AMReX GMRES or PETSc KSP). Constant eta builds a
 * flat eta MultiFab and calls the same AdvanceVariable driver as parser /
 * variable eta.
 *
 * Disabled by default. Enable with hybrid_pic_model.implicit_mag_diffusion = 1.
 */
class HybridMagDiffusion
{
public:
    HybridMagDiffusion () = default;

    void ReadParameters ();

    [[nodiscard]] bool Enabled () const { return m_enabled; }
    [[nodiscard]] amrex::Real Theta () const { return m_theta; }
    [[nodiscard]] amrex::Real EtaExplicitMax () const { return m_eta_explicit_max; }
    [[nodiscard]] bool HasConstantEta () const { return m_has_constant_eta; }
    [[nodiscard]] amrex::Real ConstantEta () const { return m_constant_eta; }
    [[nodiscard]] bool UsesVariableEta () const { return m_use_variable_eta; }
    [[nodiscard]] int Verbose () const { return m_verbose; }
    [[nodiscard]] MagDiffLinearSolver LinearSolver () const { return m_linear_solver; }
    [[nodiscard]] MagDiffPetscPC PetscPC () const { return m_petsc_pc; }

    /**
     * \brief Theta-method curl-curl magnetic diffusion on one level.
     *
     * Solves
     *   curl(alpha curl B) + beta B = rhs
     * with alpha = theta * dt * (eta_SI/mu0), beta = 1.
     * For theta = 1 (backward Euler), rhs = B_in.
     *
     * \param[in,out] Bfield  Yee B components (WarpX ordering)
     * \param[in]     eta_SI  resistivity [Ohm m]
     * \param[in]     dt      diffusion step size [s]
     * \param[in]     lev     AMR level (hybrid is single-level)
     * \param[in]     lobc    low-side linear-operator boundary types
     * \param[in]     hibc    high-side linear-operator boundary types
     */
    void Advance (
        ablastr::fields::VectorField const& Bfield,
        amrex::Real eta_SI,
        amrex::Real dt,
        int lev,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& lobc,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& hibc) const;

    void AdvanceVariable (
        ablastr::fields::VectorField const& Bfield,
        ablastr::fields::VectorField const& eta_SI,
        amrex::Real dt,
        int lev,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& lobc,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& hibc) const;

    /**
     * \brief Map WarpX field boundary types to AMReX LinOpBCType values.
     *
     * Unmapped types default to Neumann. PEC_Insulator maps to Dirichlet
     * (tangential B prescribed). Callers may still fill these arrays even
     * though the matrix-free op primarily uses WarpX field BCs.
     */
    static void GetLinOpBCs (
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>& lobc,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>& hibc);

private:
    bool m_enabled = false;
    amrex::Real m_theta = 1.0;
    /** Relative KSP/GMRES tolerance (FLASH MagDiff-class; CI uses 1e-8). */
    amrex::Real m_rtol = 1.e-8;
    /** Absolute tolerance; 0 = relative-only (FLASH-like). */
    amrex::Real m_atol = 0.0;
    int m_max_iter = 200;
    int m_verbose = 0;
    amrex::Real m_eta_explicit_max = 0.0;
    bool m_has_constant_eta = false;
    amrex::Real m_constant_eta = 0.0;
    bool m_use_variable_eta = false;
    MagDiffLinearSolver m_linear_solver = MagDiffLinearSolver::amrex_gmres;
    MagDiffPetscPC m_petsc_pc = MagDiffPetscPC::shell_jacobi;
};

#endif // WARPX_HYBRID_MAG_DIFFUSION_H_
