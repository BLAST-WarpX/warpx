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

    // Hybrid model has no light waves, so return 0 for the PC theta parameter
    [[nodiscard]] amrex::Real GetThetaForPC () const override { return 0.0; }

    /** Capture m_J_base / m_rho_base from the current (scaled, synced) fields */
    void CaptureJRhoBase ();

    /**
     * \brief In-loop electron pressure advance (the implicit implementation of the
     * gamma-law closure): compute
     * pe^{n+theta} from pe^n and the current iterate's electron flow, and write it to
     * hybrid_electron_pressure_fp for the Ohm's-law solve. Public because the CUDA
     * compiler requires the enclosing function of a device lambda to be public.
     */
    void AdvancePeInLoop ( bool a_from_jacobian );

    /**
     * \brief Turn Efield_fp (holding the Newton iterate a_E) into the particle push
     * field E* = a_E - eta*J_plasma. The resistive term is electron-ion friction: it
     * belongs to the electron momentum equation, so the ions must not feel it
     * (Stanier et al. JCP 2019, Eq. (1)). eta is evaluated from the iterate's plasma
     * current exactly as HybridPICSolveE adds it, keeping the push a pure function of
     * the iterate. E* = a_E when eta = 0.
     */
    void SubtractResistiveEFromPushField ();

    /** Advance the stored pe^n to pe^{n+1} after the Newton solve converges */
    void FinishPeUpdate ();

    /** Rebuild rho from the base state and the divergence of the current response.
     * Public because the CUDA compiler requires the enclosing function of an
     * extended device lambda to have public access. */
    void ApplyRhoResponseFromDivJ ();
    [[nodiscard]] const amrex::Vector<amrex::MultiFab*>* GetAlphaCoeff () const override;

    /**
     * \brief Override: skip the c²μ₀θΔt EM scaling from the base class (derived
     * from Ampere's law, physically wrong for the hybrid solver where E comes from
     * Ohm's law).  For the curl-curl MLMG preconditioner, still adds +1 to the
     * diagonal of MassMatrices_PC so that beta >= 1 everywhere, preventing the
     * curl-curl operator from being singular in vacuum cells.
     */
    void SetMassMatricesForPC ( amrex::Real a_theta_dt ) override;

private:

    /**
     * \brief Pointer to the hybrid-PIC model that computes E from Ohm's law.
     * This is owned by WarpX, not by this class.
     */
    HybridPICModel* m_hybrid_pic_model = nullptr;

    /**
     * \brief When true, use Darwin split: JFNK solves only for E_T (transverse
     * part of E_Ohm). E_L = ∇φ (curl-free part) is computed via a Poisson solve
     * (∇²φ = ∇·E_Ohm) once per Newton step and frozen across GMRES matvecs.
     * This decouples the longitudinal (Debye-scale) physics from the transverse
     * Alfvén/Hall physics and improves JFNK conditioning.
     */
    bool m_use_darwin_split = false;

    /**
     * \brief Persistent nodal MultiFab storing the spatially varying alpha
     * coefficient (theta*dt*sqrt((|B|/rho)^2+eta^2)/mu0) for the curl-curl preconditioner.
     * Derived from the hybrid Jacobian eigenvalue radius on the transverse (Er,Etheta) block.
     * Reduces to theta*dt*|B|/(mu0*rho) for eta=0. Refilled each Newton iteration.
     */
    /**
     * Ion current captured at the last nonlinear residual evaluation (same moment as
     * SaveE). During mass-matrix linear-stage evaluations the push field is computed
     * from this frozen J instead of current_fp, which holds the previous evaluation's
     * (perturbation-dependent) current: with a lagged J input, the push field is not a
     * pure function of the Newton variable and the MM linear model J0 + MM*(E - E0)
     * acquires a finite base offset (fatal where 1/rho amplifies it, e.g. vacuum edges).
     */
    mutable std::array<std::unique_ptr<amrex::MultiFab>, 3> m_Ji_save;

    /**
     * J and rho (scaled + synced) captured right after PreRHSOp of the last nonlinear
     * residual evaluation. During mass-matrix linear-stage evaluations rho is rebuilt as
     *   rho = rho_base - (dt/2) * div(J - J_base)
     * (charge continuity applied to the mass-matrix current response): rho responds to E
     * only through particle positions, dx = (dt/2) dv, so the discrete rho response is
     * the divergence of the deposited current response. Without this term the linear
     * model freezes rho entirely, which loses the drho/dE Jacobian channel -- fatal
     * where 1/rho amplifies it (e.g. the plasma-vacuum edge of a compression).
     */
    mutable std::array<std::unique_ptr<amrex::MultiFab>, 3> m_J_base;
    mutable std::unique_ptr<amrex::MultiFab> m_rho_base;

    /** Enable the continuity-based rho response in the MM linear stage (see m_J_base) */
    bool m_use_rho_response_divj = true;

    /**
     * Evolve the electron pressure with its own gamma-law advection/compression
     * equation, (gamma-1)^{-1} [d_t pe + div(ue pe)] + pe div(ue) = 0 (Stanier et al.
     * JCP 2019 Eq. 15 with eta = qe = 0), advanced from pe^n to pe^{n+theta} INSIDE
     * every residual evaluation using the current iterate's ue. This replaces the
     * algebraic closure pe = n^gamma Te0, whose slaving of pe to the deposited
     * (noisy, non-charge-conserving) density is an energy-injection channel for the
     * implicit scheme; it also differs from the operator-split QDSMC energy equation,
     * whose one-step-lagged pe destabilizes noise-driven problems.
     */

    /** pe^n and pe^{n+theta} state for the in-loop pressure advance */
    mutable std::unique_ptr<amrex::MultiFab> m_pe_old;
    mutable std::unique_ptr<amrex::MultiFab> m_pe_theta;
    mutable std::unique_ptr<amrex::MultiFab> m_pe_scratch;




    mutable amrex::Vector<amrex::MultiFab>  m_alpha_mf;
    mutable amrex::Vector<amrex::MultiFab*> m_alpha_mfarrvec;

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

    /** Add E_L (hybrid_E_fp_long) to Efield_fp: E_fp += E_L */
    void AddLongitudinalE ();

    /** Subtract E_L from Efield_fp: E_fp -= E_L */
    void SubtractLongitudinalE ();

    /** Subtract E_L from a solver vector: a_vec -= E_L */
    void SubtractLongitudinalE (WarpXSolverVec& a_vec);

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