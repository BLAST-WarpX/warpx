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

#include <array>
#include <memory>
#include <vector>

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

    /** Lagged mass matrices: re-deposit on Newton iterations 0, n, 2n, ... with
     *  n = implicit_evolve.mass_matrices_deposit_interval (0: iteration 0 only), and at
     *  iteration 0 only on every k-th step, k = mass_matrices_step_interval. */
    [[nodiscard]] bool DepositMassMatricesThisIter (int a_newton_iter) const override
    {
        if (a_newton_iter == 0) { return m_mm_deposit_this_step; }
        if (m_mass_matrices_deposit_interval <= 0) { return false; }
        return a_newton_iter % m_mass_matrices_deposit_interval == 0;
    }

    /** Capture m_J_base / m_rho_base from the current (scaled, synced) fields */
    void CaptureJRhoBase ();

    /**
     * \brief In-loop electron pressure advance (the implicit implementation of the
     * gamma-law closure): compute
     * pe^{n+theta} from pe^n and the current iterate's electron flow, and write it to
     * hybrid_electron_pressure_fp for the Ohm's-law solve. Public because the CUDA
     * compiler requires the enclosing function of a device lambda to be public.
     */
    void AdvanceElectronPressure ( bool a_from_jacobian,
                                   amrex::Real a_theta_time );

    /** pe_wall_mirror = 2: entropy slave of the non-periodic axial face planes applied to the
     *  accepted pe^{n+1} (collocated RZ), see the definition. */
    void SlaveWallPressure ( int lev );

    /** Positive-definite Joule deposit Q = eta |Jp|^2 + eta_H <|grad Jp|^2> of the
     *  collocated in-loop pe advance (include_joule_heating = 1), replacing the pointwise
     *  sign-indefinite -D.Jp pairing there; Yee/RZ keep -D.Jp. */
    mutable std::unique_ptr<amrex::MultiFab> m_Q_diss;

    /** Joule redirect to the ions (implicit_evolve.joule_redirect_to_ions): the fraction of
     *  Q the Te gate withholds from pe, staged per charged species as the (2/3)-scaled
     *  per-ion energy E_s [J] that QDSMCApplyIonHeating deposits (explicit QDSMC
     *  convention), plus the withheld power density (1 - sJ) Q [W/m^3] for the ledger print */
    static constexpr int kMaxRedirectSpecies = 8;
    mutable std::unique_ptr<amrex::MultiFab> m_ion_redirect_E;
    mutable std::unique_ptr<amrex::MultiFab> m_Q_redir;
    mutable int m_n_ion_species = 0;
    mutable amrex::GpuArray<amrex::Real, kMaxRedirectSpecies> m_ion_Z{};

    /**
     * \brief Turn Efield_fp (holding the Newton iterate a_E) into the particle push
     * field E* = a_E - D, where D = eta*J_plasma - eta_h*nabla^2(J_plasma) is the
     * dissipative part of Ohm's law. The dissipative terms are electron-ion friction
     * and electron viscosity: they belong to the electron momentum equation, so the
     * ions must not feel them (Stanier et al. JCP 2019, Eq. (1)). D is computed as
     * the difference of two Ohm solves (with and without dissipation) from the same
     * state, so it matches HybridPICSolveE exactly by construction, and is kept in
     * m_D for the electron work pairing (dU_e gains +D.J_plasma). E* = a_E when
     * eta = eta_h = 0.
     */
    void SubtractDissipativeEFromPushField ();

    /** Advance the field state (E, B and the electron pressure) from t^{n+theta}
     * to t^{n+1}. Public because the CUDA compiler requires the enclosing
     * function of an extended device lambda to have public access. */
    void FinishFieldUpdate ( amrex::Real end_time );

    /** Apply the density weight to the predictor increment (see m_predictor_rho_factor).
     * Public because the CUDA compiler requires the enclosing function of an extended
     * device lambda to have public access. */
    void WeightPredictorIncrement ();

    /** Rebuild rho from the base state and the divergence of the current response.
     * Public because the CUDA compiler requires the enclosing function of an
     * extended device lambda to have public access. */
    void ApplyRhoResponseFromDivJ ();

    /** The ion current deposit consumed by the Ohm solve (current_fp
     *  registry) -- frozen during Jacobian probes, so it is the exact
     *  drift-leg coefficient of the linear solve. */
    amrex::Array<const amrex::MultiFab*, 3> GetIonCurrentForPC (int lev) const override;

    /** The hybrid-PIC model (owned by WarpX, cached in Define): n_floor,
     *  the resistivity and hyper-resistivity parsers, and the Hall flags
     *  a Hall-aware preconditioner needs. */
    const HybridPICModel* GetHybridPICModel () const override { return m_hybrid_pic_model; }

    /** The TOTAL theta-midpoint magnetic field B^{n+theta}. On this branch
     *  the external-field splitting keeps Bfield_fp INTERNAL between
     *  residual evaluations (externals are added and subtracted inside each
     *  evaluation), so with add_external_fields the external contribution
     *  is re-assembled into owned scratch here -- the preconditioner must
     *  linearize the whistler leg at the total field the particles and the
     *  Ohm solve actually see. Valid only after the first residual
     *  evaluation of the iterate. */
    amrex::Array<const amrex::MultiFab*, 3> GetBfieldThetaForPC (int lev) const override;

    /** The charge density holding the midpoint deposit rho^{n+1/2} in
     *  component nComp()/2 (rho_fp registry). */
    const amrex::MultiFab* GetRhoMidForPC (int lev) const override;

    private:

    /**
     * \brief Pointer to the hybrid-PIC model that computes E from Ohm's law.
     * This is owned by WarpX, not by this class.
     */
    HybridPICModel* m_hybrid_pic_model = nullptr;

    /**
     * Ion current captured at the last nonlinear residual evaluation (same moment as
     * SaveE). During mass-matrix linear-stage evaluations the push field is computed
     * from this frozen J instead of current_fp, which holds the previous evaluation's
     * (perturbation-dependent) current: with a lagged J input, the push field is not a
     * pure function of the Newton variable and the MM linear model J0 + MM*(E - E0)
     * acquires a finite base offset (fatal where 1/rho amplifies it, e.g. vacuum edges).
     */
    mutable std::array<std::unique_ptr<amrex::MultiFab>, 3> m_Ji_save;

    /** Dissipative part of Ohm's law, D = eta*J_p - eta_h*nabla^2(J_p), captured each
     * residual evaluation (see SubtractDissipativeEFromPushField); m_E_work is the
     * scratch for the two Ohm solves that form it. */
    mutable std::array<std::unique_ptr<amrex::MultiFab>, 3> m_D;
    mutable std::array<std::unique_ptr<amrex::MultiFab>, 3> m_E_work;
    /** Zeroed stand-in for m_D in the pe work pairing when
     * include_joule_heating is off (the dissipative work then leaves the
     * system instead of heating pe, matching the explicit QDSMC path). */
    mutable std::array<std::unique_ptr<amrex::MultiFab>, 3> m_D_zero;
    /** Total push field E* + E_ext for the pe work pairing when external
     * (vector-potential) fields are on: the ions are pushed with the total
     * field and Ohm's law holds for it, so the electron work E.J_e must be
     * evaluated with it too (see AdvanceElectronPressure). */
    mutable std::array<std::unique_ptr<amrex::MultiFab>, 3> m_E_pair;

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

    /** Owned scratch for GetBfieldThetaForPC's total-field assembly
     *  (internal + external B), allocated lazily per direction. */
    mutable amrex::Array<std::unique_ptr<amrex::MultiFab>, 3> m_B_tot_pc;

    mutable std::unique_ptr<amrex::MultiFab> m_pe_scratch;





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

    /** implicit_evolve.predictor (default 0): start Newton from the extrapolated
     *  guess E^n + (E^{n-1+theta} - E^{n-1}) (the previous step's theta-increment,
     *  = E^n + theta (E^n - E^{n-1}); pe rows included) instead of E^n. Pair it with
     *  newton.absolute_tolerance: the relative tolerance is measured against the
     *  initial residual, which the predictor shrinks. */
    bool m_predictor = false;
    /** the predictor is skipped after a step that needed this many Newton iterations */
    int m_predictor_max_iters = 8;
    /** implicit_evolve.predictor_max_step (default -1 = no limit): the predictor is
     *  used only for steps < this number (a phase where the field evolves smoothly);
     *  later steps start from E^n (the FRC formation halo phase, where extrapolating
     *  particle-noise-set floor-node fields made the first linear solve diverge). */
    int m_predictor_max_step = -1;

    /** implicit_evolve.dt_halving_max_levels (default 0 = off): when the Newton solve
     *  of a step ends unconverged (iteration cap, two rejected directions, divergence),
     *  restore the state at the step start (total fields, electron pressure, particles)
     *  and redo the step as two half-steps, recursively up to this many halvings
     *  (dt/2^levels); the outer time step and the diagnostic cadence are unchanged. */
    int m_dt_halving_max_levels = 0;
    /** implicit_evolve.dt_halving_max_rel (default 0.1): with halving on, a sub-step at
     *  the deepest level that is still unconverged is accepted only if its final relative
     *  residual is below this; otherwise the step gives up (negative exit status: WarpX
     *  aborts, the run can be restarted from a checkpoint). Accepting such a state
     *  silently let the next sub-step start 30x worse and spiral (formation, 7.77 us). */
    amrex::Real m_dt_halving_max_rel = 0.1;
    /** saved total E^n, B^n and pe^n per halving level (see SaveStepStartState) */
    std::vector<std::array<std::unique_ptr<amrex::MultiFab>,3>> m_E_save;
    std::vector<std::array<std::unique_ptr<amrex::MultiFab>,3>> m_B_save;
    std::vector<std::unique_ptr<amrex::MultiFab>> m_pe_save;
    long m_dt_halvings = 0;   //!< number of halvings taken so far (statistics)
    /** set by ComputeRHS when the trial iterate pushed particles beyond the deposition
     *  guard range: the deposits of those tiles were skipped and the RHS holds
     *  m_invalid_rhs_value, a state Newton must reject (see
     *  WarpXParticleContainer::SetDepositOutOfRangeTolerant, ImplicitSolver::LastRHSInvalid) */
    bool m_rhs_invalid = false;
    long m_rhs_invalid_count = 0;   //!< number of such evaluations (statistics)
    static constexpr amrex::Real m_invalid_rhs_value = 1.0e100;
    [[nodiscard]] bool LastRHSInvalid () const override { return m_rhs_invalid; }
    /** implicit_evolve.dt_halving_test_step (default -1 = off; testing only): treat the
     *  first solve of this step as unconverged so the rejection/redo path runs. */
    int m_dt_halving_test_step = -1;

    /** One (sub)step at halving level a_level: solve, and on an unconverged solve
     *  with a level left, restore and recurse with dt/2; otherwise finish the step. */
    int AdvanceStep (amrex::Real start_time, amrex::Real a_dt, int a_step, int a_level);
    /** External-field split, state saves and the Newton solve of one (sub)step.
     *  Returns the nonlinear solver exit status; the state is left at the iterate. */
    int SolveStep (amrex::Real start_time, amrex::Real a_dt, int a_step);
    /** Everything after an accepted solve: fields and particles to t^{n+1}, Q_ei. */
    void FinishStep (amrex::Real start_time, int a_step);
    void SaveStepStartState (int a_level);
    void RestoreStepStartState (int a_level);
    /** implicit_evolve.predictor_rho_factor (default 0 = off): multiply the predictor
     *  increment by 0.5 (1 + tanh((rho - f rho_floor)/rho_floor)), i.e. do not extrapolate
     *  at nodes below ~f x the density floor, where the field is particle-noise dominated
     *  (E ~ J/rho_floor) and its step-to-step increment is not a smooth continuation. */
    amrex::Real m_predictor_rho_factor = 0.0;
    WarpXSolverVec m_dE_prev;
    bool m_have_dE_prev = false;

    /** implicit_evolve.pe_newton_unknown = 1 (in-loop pe closure only): solve for
     *  pe^{n+theta} as a Newton unknown in the solver vector's scalar slot, scaled by
     *  m_pe_scale = 1/(q_e n0_ref dx_min) so the pe row is commensurate with E. */
    bool m_pe_unknown = false;
    amrex::Real m_pe_scale = 1.0;

    /** see DepositMassMatricesThisIter */
    int m_mass_matrices_deposit_interval = 1;
    int m_mass_matrices_step_interval = 1;
    /** implicit_evolve.mass_matrices_rho_response_factor: the linear stage's
     *  density response rho = rho_base - (dt/2) div(J - J_base) is applied only
     *  where rho_base exceeds this factor times the density floor (below it the
     *  1/n factors of Ohm's law amplify the noise of the response); the
     *  Jacobian is frozen-rho in the excluded cells */
    amrex::Real m_mm_rho_response_factor = 10.0;
    bool m_mm_deposit_this_step = true;
    bool m_mm_deposited_once = false;
    /** dt of the last mass-matrix deposit: the kernels carry dt (gyro factors q dt B/(2m)),
     *  so a lagged deposit is re-done whenever the (sub)step size changes (dt halving). */
    amrex::Real m_mm_deposit_dt = -1.0;

    /** pe_newton_unknown mode: one-evaluation theta-update result at the
     *  current iterate, consumed by the residual row in ComputeRHS (the
     *  pressure field itself keeps the iterate for Ohm's law). */
    mutable std::unique_ptr<amrex::MultiFab> m_pe_rhs;

    /** implicit_evolve.pe_advection = central | vanalbada: discretization of the enthalpy
     *  flux div(ue pe) in the in-loop pe advance, 0 = central differences, 1 = van Albada
     *  MUSCL face flux (default on the collocated Cartesian grid; central on Yee/RZ). */
    int m_pe_advection = 1;

    /** implicit_evolve.filter_push_fields (requires warpx.use_filter, a periodic domain
     *  and no external fields): the ions gather the binomial-filtered push field while
     *  Ohm's law and the pe work pairing keep the unfiltered E (conservative smoothing). */
    bool m_filter_push_fields = false;

    /** unfiltered E push-field copies for the filter_push_fields swap */
    mutable std::array<std::unique_ptr<amrex::MultiFab>, 3> m_E_unfiltered;

    /** apply (true) / restore (false) the push-field filter swap */
    void FilterPushFieldsSwap (bool a_apply);

    /** implicit_evolve.pe_ue_cap_fac: multiplier on the marker-CFL u_e cap of the pe
     *  advance, ue_cap = fac * dx_min/(theta dt). */
    amrex::Real m_pe_ue_cap_fac = 1.0;

    /** implicit_evolve.pe_wall_mirror (default 2), collocated RZ grid, non-periodic axial faces:
     *  0 = legacy: the face planes are skipped and stay at pe^n (identity rows);
     *  1 = mirror advance: the face nodes are advanced with the symmetry-plane stencil (axial
     *      index mirrored, axial electron flux odd). Unstable where the column stagnates on
     *      the plane: the folded ion current vanishes on the face node while the electron flow
     *      one plane in is still the full inflow, so the flux convergence lands on the face
     *      node with no density response (formation v2: wall Te 28 keV vs 70 eV one plane in);
     *  2 = mirror advance inside the solve (as 1) + entropy slave of the face planes applied to
     *      the accepted pe^{n+1} after each step, pe = s n^gamma with s the mean pe/n^gamma of
     *      the three planes inside (d s/dz = 0 at a symmetry plane; the RZ axis-row treatment).
     *      Imposing the slave inside the Newton solve instead broke the solve (v3). */
    int m_pe_wall_mirror = 2;

    /** implicit_evolve.joule_Te_cutoff [eV] (< 0: off): the in-loop counterpart of the
     *  explicit QDSMC Te-threshold Joule redirect. The Joule + hyper-resistive deposit
     *  into pe is multiplied by the smooth gate 0.5 (1 - tanh((Te^n - Tc)/(w Tc))),
     *  Te^n = pe^n/(e n) at the node: electrons above Tc stop receiving the
     *  dissipative heat (it leaves the ledger, as with include_joule_heating off,
     *  unless joule_redirect_to_ions hands it to the ions as the explicit path
     *  does). Width w = implicit_evolve.joule_Te_cutoff_width (fraction of Tc).
     *  Collocated grids (positive-definite Joule deposit) only. */
    amrex::Real m_joule_Te_cutoff_eV = -1.0;
    amrex::Real m_joule_Te_cutoff_width = 0.25;

    /** implicit_evolve.joule_redirect_to_ions (default 0): with the Te gate on, deposit the
     *  withheld energy (1 - sJ) Q dt on the ion particles through the QDSMC
     *  Ornstein-Uhlenbeck kernel in FinishStep (per species E_s = (2/3) Z_s (1 - sJ) Q dt /
     *  n_e, the explicit joule_redirect_Te_threshold convention), restoring the energy
     *  ledger. implicit_evolve.joule_redirect_verbose = 1 prints the two Joule powers. */
    bool m_joule_redirect = false;
    int m_joule_redirect_verbose = 0;

    /** implicit_evolve.pe_debug_nodes = i1 j1 [i2 j2 ...]: print the per-term budget of
     *  the in-loop pe advance (pairing field, electron flux, Joule deposit, flux
     *  divergence, conduction, pe^n -> pe^{n+theta}) at these nodes after every
     *  nonlinear residual evaluation (development diagnostic). */
    std::vector<int> m_pe_debug_nodes;

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
