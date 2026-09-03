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

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <array>
#include <memory>

class HybridPICModel;

/** @file
 *  Theta-implicit hybrid-PIC time solver class. Both the fields and the
 *  particles are treated implicitly.
 *
 *  In the hybrid-PIC model, ions are kinetic particles while electrons are
 *  treated as a massless, charge-neutralizing fluid. The electric field is
 *  determined algebraically from the generalized Ohm's law.
 *
 *  The magnetic field evolves via Faraday's law:
 *  Bg^{n+1} = Bg^n - dt*curl(Eg^{n+theta})
 *
 *  Particles are advanced as:
 *  xp^{n+1} = xp^n + dt*vp^{n+1/2}
 *  vp^{n+1} = vp^n + dt*qp/mp*(Ep^{n+theta} + vp^{n+1/2} x Bp^{n+theta})
 *
 *  where f^{n+theta} = (1.0-theta)*f^{n} + theta*f^{n+1} with 0.5 <= theta <= 1.0
 *
 *  The key difference from the fully electromagnetic implicit solvers is that
 *  E carries no evolution equation of its own: it is computed from Ohm's law
 *  at each nonlinear iteration. The nonlinear solver finds the self-consistent
 *  E^{n+theta} such that:
 *     E^{n+theta} = Ohm(B^{n+theta}, Ji^{n+1/2}, rho^{n+1/2})
 *  where B^{n+theta} depends on E^{n+theta} through Faraday's law and the ion
 *  moments are deposited from the particles re-advanced with the iterate.
 *
 *  Since the displacement current is neglected in the hybrid model there are
 *  no light waves and the speed-of-light CFL condition does not apply. In
 *  particular the time step is not constrained by the whistler-wave CFL
 *  condition of the explicit hybrid advance; it is limited by the ion physics
 *  to be resolved and by particles not traveling beyond the ghost region.
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

    void Define (WarpX* a_WarpX, bool from_restart) override;

    void PrintParameters () const override;

    /**
     * \brief Advances the simulation one time step
     *
     * \param[in] start_time The time at the start of the time step
     * \param[in] a_dt The time step size
     * \param[in] a_step The time step number
     */
    int OneStep ( amrex::Real  start_time,
                  amrex::Real  a_dt,
                  int          a_step ) override;

    void ComputeRHS ( WarpXSolverVec&  a_RHS,
                const WarpXSolverVec&  a_E,
                      amrex::Real      start_time,
                      int              a_nl_iter,
                      bool             a_from_jacobian ) override;

    // The hybrid model has no light waves, so return 0 for the PC theta parameter
    [[nodiscard]] amrex::Real GetThetaForPC () const override { return 0.0; }

private:

    /**
     * \brief Pointer to the hybrid-PIC model that computes E from Ohm's law.
     * This is owned by WarpX, not by this class.
     */
    HybridPICModel* m_hybrid_pic_model = nullptr;

    /** Dissipative part of Ohm's law, D = eta*J_p - eta_h*nabla^2(J_p), captured each
     * residual evaluation (see SubtractDissipativeEFromPushField); m_E_work is the
     * scratch for the two Ohm solves that form it. */
    mutable std::array<std::unique_ptr<amrex::MultiFab>, 3> m_D;
    mutable std::array<std::unique_ptr<amrex::MultiFab>, 3> m_E_work;

    /**
     * \brief Solver vectors for the electric field.
     * m_E: E field at time n+theta (during the solve), then n+1 (after FinishFieldUpdate)
     * m_Eold: E field at time n (saved at the start of the time step), used as the
     * initial guess and as the reference value in ComputeRHS and FinishFieldUpdate.
     */
    WarpXSolverVec m_E;
    WarpXSolverVec m_Eold;

    /**
     * \brief Turn Efield_fp (holding the Newton iterate a_E) into the particle push
     * field E* = a_E - D, where D = eta*J_plasma - eta_h*nabla^2(J_plasma) is the
     * dissipative part of Ohm's law. The dissipative terms are electron-ion friction
     * and electron viscosity: they belong to the electron momentum equation, so the
     * ions must not feel them (Stanier et al., JCP 376 (2019) 597). D is computed as
     * the difference of two Ohm solves (with and without dissipation) from the same
     * state, so it matches HybridPICSolveE exactly by construction. E* = a_E when
     * eta = eta_h = 0.
     */
    void SubtractDissipativeEFromPushField ();

    /**
     * \brief Update the E and B fields owned by WarpX. Sets E^{n+theta}, then
     * updates B^{n+theta} from E via Faraday's law.
     */
    void UpdateWarpXFields ( const WarpXSolverVec&  a_E,
                             amrex::Real            start_time );

    /**
     * \brief Extrapolate the fields from t^{n+theta} to t^{n+1} after convergence:
     * F^{n+1} = (1/theta)*F^{n+theta} + (1 - 1/theta)*F^n, applied to E and B.
     */
    void FinishFieldUpdate ( amrex::Real end_time );

};

#endif
