/* Copyright 2026 Prabhat Kumar
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef THETA_IMPLICIT_HYBRID_H_
#define THETA_IMPLICIT_HYBRID_H_

#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "ImplicitSolver.H"

#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

class HybridPICModel;

/** @file
 *  Theta-implicit hybrid-PIC time solver class. This is a fully implicit
 *  algorithm where both the fields and particles are treated implicitly.
 *
 *  In the hybrid-PIC model, ions are kinetic particles while electrons are
 *  treated as a massless, charge-neutralizing fluid. The electric field is
 *  determined algebraically from the generalized Ohm's law.
 *
 *  The magnetic field evolves via Faraday's law:
 *  Bg^{n+1} = Bg^n - dt*curl(Eg^{n+theta})
 *
 *  Particles are advanced as:
 *  xp^{n+1} = xp^n + dt*up^{n+1/2}/(0.5*(gammap^n + gammap^{n+1}))
 *  up^{n+1} = up^n + dt*qp/mp*(Ep^{n+theta} + up^{n+1/2}/gammap^{n+1/2} x Bp^{n+theta})
 *
 *  where f^{n+theta} = (1.0-theta)*f^{n} + theta*f^{n+1} with 0.5 <= theta <= 1.0
 *
 *  The key difference from fully electromagnetic implicit solvers is that E is computed from Ohm's law at each
 *  nonlinear iteration. The nonlinear solver finds the self-consistent E^{n+theta}
 *  such that:
 *     E^{n+theta} = Ohm(B^{n+theta}, Ji^{n+1/2}, rho^{n+1/2})
 *  where B^{n+theta} depends on E^{n+theta} through Faraday's law.
 *
 *  The user-specified time-biasing parameter theta is bounded between 0.5 and 1.0.
 *  Significant damping of high-k modes will occur as theta approaches 1.0. Since
 *  there are no light waves in the hybrid model (the displacement current is neglected),
 *  the CFL condition for light waves does not apply. The time step is limited by:
 *  1. Resolving the relevant ion physics (cyclotron motion, etc.)
 *  2. Particles not traveling outside the ghost region (MPI limitation)
 *  3. The need to resolve whistler waves if hyper-resistivity is insufficient
 *
 */

class ThetaImplicitHybrid : public ImplicitSolver
{
public:

    ThetaImplicitHybrid() = default;

    ~ThetaImplicitHybrid() override = default;

    // Prohibit Move and Copy operations
    ThetaImplicitHybrid(const ThetaImplicitHybrid&) = delete;
    ThetaImplicitHybrid& operator=(const ThetaImplicitHybrid&) = delete;
    ThetaImplicitHybrid(ThetaImplicitHybrid&&) = delete;
    ThetaImplicitHybrid& operator=(ThetaImplicitHybrid&&) = delete;

    void Define ( WarpX*  a_WarpX ) override;

    void PrintParameters () const override;

    /**
     * \brief Advances the simulation one time step
     *
     * \param[in] start_time The time at the start of the time step
     * \param[in] a_dt The time step size
     * \param[in] a_step The time step number
     */
    void OneStep ( amrex::Real  start_time,
                   amrex::Real  a_dt,
                   int          a_step ) override;

    void ComputeRHS ( WarpXSolverVec&  a_RHS,
                const WarpXSolverVec&  a_E,
                      amrex::Real      start_time,
                      int              a_nl_iter,
                      bool             a_from_jacobian ) override;

    // Hybrid model has no light waves, so return 0 for the PC theta parameter
    [[nodiscard]] amrex::Real GetThetaForPC () const override { return 0.0_rt; }

private:

    /**
     * \brief Pointer to the hybrid-PIC model that computes E from Ohm's law.
     * This is owned by WarpX, not by this class.
     */
    HybridPICModel* m_hybrid_pic_model = nullptr;

    /**
     * \brief Flag indicating whether external fields from vector potentials
     * should be included. When true, field splitting is used: B_total = B_internal + B_external.
     */
    bool m_add_external_fields = false;

    /**
     * \brief Solver vectors for the electric field.
     * m_E: E field at time n+theta (during solve), then n+1 (after FinishFieldUpdate)
     * m_Eold: E field at time n (saved at start of timestep)
     *
     * m_Eold is needed for:
     * - Initial guess for the nonlinear solver
     * - Reference value for residual computation in ComputeRHS
     * - Extrapolation from t^{n+theta} to t^{n+1} in FinishFieldUpdate
     */
    WarpXSolverVec m_E;
    WarpXSolverVec m_Eold;

    /**
     * \brief Update the E and B fields owned by WarpX. Updates B^{n+theta}
     * from E via Faraday's law, then computes E^{n+theta} from Ohm's law.
     */
    void UpdateWarpXFields ( const WarpXSolverVec&  a_E,
                             amrex::Real            start_time );

    /**
     * \brief Extrapolate fields from t^{n+theta} to t^{n+1} after convergence.
     * Uses: F^{n+1} = (1/theta)*F^{n+theta} + (1 - 1/theta)*F^n
     * This is applied to both E and B fields.
     */
    void FinishFieldUpdate ( amrex::Real end_time );

    /**
     * \brief Add external B field to Bfield_fp: B_fp += B_external
     */
    void AddExternalBfield ();

    /**
     * \brief Subtract external B field from Bfield_fp: B_fp -= B_external
     */
    void SubtractExternalBfield ();

    /**
     * \brief Add external E field to Efield_fp: E_fp += E_external
     */
    void AddExternalEfield ();

    /**
     * \brief Subtract external E field from Efield_fp: E_fp -= E_external
     */
    void SubtractExternalEfield ();

};

#endif
