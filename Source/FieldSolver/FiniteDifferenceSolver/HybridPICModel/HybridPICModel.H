/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *          S. Eric Clark (Helion Energy)
 *          Prabhat Kumar (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_HYBRIDPICMODEL_H_
#define WARPX_HYBRIDPICMODEL_H_

#include "HybridPICModel_fwd.H"

#include "Fields.H"
#include "Fluids/QdsmcParticleContainer_fwd.H"

#include "ExternalVectorPotential.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXConst.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/text/IntervalsParser.H>

#include <AMReX_Array.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_DistributionMapping.H>

#include <map>
#include <cmath>
#include <memory>
#include <optional>

/**
 * \brief This class contains the parameters needed to evaluate hybrid field
 * solutions (kinetic ions with fluid electrons).
 */
/** C1 smooth-max blend of the density-floor guard: exact max(rho,
 *  rho_floor) at w_abs = 0 (legacy hard floor, bit-compatible), the
 *  smooth-max 0.5*(rho + rho_floor + sqrt((rho-rho_floor)^2 + w_abs^2))
 *  otherwise. Bounded below by rho_floor everywhere like the hard max
 *  (the halo response stays floored -- this is not a response mask),
 *  but infinitely differentiable, so deposit noise chattering across
 *  the threshold no longer kinks the Ohm residual. w_abs is the
 *  absolute (charge-density) blend width. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real HybridSmoothFloor (amrex::Real rho, amrex::Real rho_floor,
                               amrex::Real w_abs)
{
    if (w_abs <= amrex::Real(0.)) { return amrex::max(rho, rho_floor); }
    const amrex::Real d = rho - rho_floor;
    return amrex::Real(0.5)*(rho + rho_floor + std::sqrt(d*d + w_abs*w_abs));
}

/** Weight of the conditional external-E subtraction across the density
 *  floor: the legacy step (rho >= rho_floor) at w_abs = 0, a tanh blend
 *  of the same width as the smooth floor otherwise (the two share the
 *  hybrid_pic_model.n_floor_smooth_width knob -- they are the same
 *  crossing and must smooth together). */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real HybridExtSubWeight (amrex::Real rho, amrex::Real rho_floor,
                                amrex::Real w_abs)
{
    if (w_abs <= amrex::Real(0.)) {
        return (rho >= rho_floor) ? amrex::Real(1.) : amrex::Real(0.);
    }
    return amrex::Real(0.5)*(amrex::Real(1.)
        + std::tanh((rho - rho_floor)/w_abs));
}

/** Statistics-aware floor of the vacuum weight (see
 *  hybrid_pic_model.vacuum_weight_r_ref): rho_floor * min(max_fac, max(1, r_ref/r)),
 *  r clamped at half a cell so the axis node is finite. r_ref <= 0 returns rho_floor. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real HybridVacuumFloor (amrex::Real rho_floor, amrex::Real r,
                               amrex::Real r_ref, amrex::Real max_fac, amrex::Real dr)
{
    if (r_ref <= amrex::Real(0.)) { return rho_floor; }
    const amrex::Real rc = amrex::max(r, amrex::Real(0.5)*dr);
    return rho_floor * amrex::min(max_fac, amrex::max(amrex::Real(1.), r_ref/rc));
}

class HybridPICModel
{
public:
    HybridPICModel ();
    // Defined out-of-line in the .cpp so the unique_ptr<QdsmcParticleContainer>
    // member can destroy a forward-declared particle container.
    ~HybridPICModel ();

    HybridPICModel ( HybridPICModel const & )             = delete;
    HybridPICModel& operator= ( HybridPICModel const & )  = delete;
    HybridPICModel ( HybridPICModel && )                  = delete;
    HybridPICModel& operator= ( HybridPICModel && )       = delete;

    /** Read user-defined model parameters. Called in constructor. */
    void ReadParameters ();

    /** Allocate hybrid-PIC specific multifabs. Called in constructor. */
    void AllocateLevelMFs (
        ablastr::fields::MultiFabRegister & fields,
        int lev,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        int ncomps,
        const amrex::IntVect& ngJ,
        const amrex::IntVect& ngRho,
        const amrex::IntVect& ngEB,
        const amrex::IntVect& jx_nodal_flag,
        const amrex::IntVect& jy_nodal_flag,
        const amrex::IntVect& jz_nodal_flag,
        const amrex::IntVect& rho_nodal_flag,
        const amrex::IntVect& Ex_nodal_flag,
        const amrex::IntVect& Ey_nodal_flag,
        const amrex::IntVect& Ez_nodal_flag,
        const amrex::IntVect& Bx_nodal_flag,
        const amrex::IntVect& By_nodal_flag,
        const amrex::IntVect& Bz_nodal_flag
    ) const;

    void InitData (const ablastr::fields::MultiFabRegister& fields);

    /**
     * \brief
     * Function to evaluate the external current expressions and populate the
     * external current multifab. Note the external current can be a function
     * of time and therefore this should be re-evaluated at every step.
     */
    void GetCurrentExternal ();

    /**
     * \brief
     * Function to calculate the total plasma current based on Ampere's law while
     * neglecting displacement current (J = curl x B). Any external current is
     * subtracted as well. Used in the Ohm's law solver (kinetic-fluid hybrid model).
     *
     * \param[in] Bfield       Magnetic field from which the current is calculated.
     * \param[in] eb_update_E  Indicate in which cell J should be calculated (related to embedded boundaries).
     */
    void CalculatePlasmaCurrent (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E
    ) const;
    void CalculatePlasmaCurrent (
        ablastr::fields::VectorField const& Bfield,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
        int lev
    ) const;

    /**
     * \brief
     * Function to update the E-field using Ohm's law (hybrid-PIC model).
     */
    void HybridPICSolveE (
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        bool solve_for_Faraday,
        bool keep_grad_pe = false) const;

    void HybridPICSolveE (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& Jfield,
        ablastr::fields::VectorField const& Bfield,
        amrex::MultiFab const& rhofield,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
        int lev, bool solve_for_Faraday,
        bool keep_grad_pe = false) const;

    void HybridPICSolveE (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& Jfield,
        ablastr::fields::VectorField const& Bfield,
        amrex::MultiFab const& rhofield,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
        int lev, PatchType patch_type, bool solve_for_Faraday,
        bool keep_grad_pe = false) const;

    /** Collocated grids, hybrid_pic_model.pec_normal_E_from_interior: overwrite the
     *  electric-field component normal to each perfectly conducting wall at the wall
     *  node with the value at the first interior node (see the member documentation). */
    void ApplyPECNormalEFromInterior (
        ablastr::fields::VectorField const& Efield, int lev) const;

    void BfieldEvolve (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        int step, amrex::Real dt_half, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void BfieldEvolve (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        int step, amrex::Real dt_half, int lev, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void BfieldEvolveRK4 (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        std::array<amrex::MultiFab, 3>& B_old,
        amrex::Real dt, int lev, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    amrex::Real BfieldEvolveRKF45 (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        std::array<amrex::MultiFab, 3>& B_old,
        amrex::Real dt, int lev, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void FieldPush (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        amrex::Real dt, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    /**
     * \brief
     * Function to calculate the electron pressure using the simulation charge
     * density. Used in the Ohm's law solver (kinetic-fluid hybrid model).
     */
    void CalculateElectronPressure () const;
    void CalculateElectronPressure (int lev) const;

    /**
     * \brief Compute the electron fluid velocity V_e on the grid from the
     * deposited ion current and the plasma current: V_e = (J_i - J_plasma)/rho,
     * with rho floored at m_n_floor * q_e. Used by the resistive-drag operator.
     */
    void CalculateElectronFluidVelocity () const;
    void CalculateElectronFluidVelocity (int lev) const;

    /**
     * \brief Compute the per-species ion bulk velocity Vs = Js/rhos on the
     * grid for every charged species. Output is stored in Vs_fp_{species}
     * (face-staggered like J). Used by the resistive-drag operator to
     * apply a per-cell uniform shift (V_s - V_e)(1 - exp(-nu dt)) to every
     * particle, leaving the thermal moment intact.
     */
    void CalculateIonFluidVelocity () const;
    void CalculateIonFluidVelocity (int lev) const;

    /**
     * \brief Fill the electron pressure multifab given the kinetic particle
     * charge density (and assumption of quasi-neutrality) using the user
     * specified electron equation of state.
     *
     * \param[out] Pe_field scalar electron pressure MultiFab at a given level
     * \param[in] rho_field scalar ion charge density Multifab at a given level
     */
    void FillElectronPressureMF (
        amrex::MultiFab& Pe_field,
        amrex::MultiFab const& rho_field ) const;

    /**
     * \brief Fill the nodal V_e = -(J_plasma - J_i) / (q_e n_e) MultiFabs by
     *        interpolating the Yee-staggered total plasma current and ion
     *        current to the nodal grid (using ablastr::coarsen::sample::Interp)
     *        and dividing by rho. Cells with rho <= rho_floor are left at 0.
     *
     *        Reads hybrid_current_fp_plasma (J_plasma), hybrid_current_fp_temp
     *        (J_i at n+1/2), hybrid_rho_fp_temp (rho at n+1/2). Writes
     *        hybrid_electron_velocity_fp (3 directions, nodal).
     */
    void QDSMCInitializeUe (int lev) const;

    /**
     * \brief Fill the nodal K_e = T_e * n_e^(1-gamma) * (k_B / q_e) MultiFab
     *        from the current T_e and rho_fp_temp (= rho at n+1/2). Cells with
     *        rho <= rho_floor are left at 0.
     *
     *        Reads hybrid_electron_temperature_fp (T_e in K) and
     *        hybrid_rho_fp_temp; writes hybrid_entropy_fp.
     */
    void QDSMCInitializeKe (int lev) const;

    /**
     * \brief After the QDSMC scatter, recover T_e^{n+1} from
     *           T_e = (deposited K*N) / (deposited N) / n_e^(1-gamma)
     *                 / (k_B / q_e)
     *        using rho at n+1 (rho_fp), the deposited entropy field
     *        (hybrid_entropy_fp), and the deposited weight field
     *        (hybrid_qdsmc_weights_fp).
     *
     *        Writes hybrid_electron_temperature_fp.
     */
    void QDSMCUpdateTe (int lev) const;

    /** \brief Fill hybrid_electron_temperature_fp with T_e = P_e/(n_e k_B) from the
     * current electron pressure and rho_fp (component 0). */
    void FillTeFromPe (int lev) const;

    /**
     * \brief Symmetric ion-electron collisional energy exchange: the Q_ei relaxation
     * sinks/sources T_e and the stochastic drag-diffusion operator applies the exact
     * conjugate to the ion particles, plus any staged redirected Joule energy.
     * Requires rho_fp, rho_fp_<species> and the per-species T_<species> deposits to
     * be current. Shared by the explicit QDSMC advance and the theta-implicit scheme
     * (whose transport and Joule heating are handled in-loop instead).
     *
     * \param[in] ion_redirect_E per-charged-species staged Joule energy to deliver to
     *            the ions (nullptr when the redirect is off).
     */
    void ApplyIonElectronEnergyExchange (int lev, amrex::Real dt,
                                         amrex::MultiFab* ion_redirect_E = nullptr) const;

    /**
     * \brief Resistive electron-heating source (see Phys. Plasmas 31, 012902 (2024), Eq. 12).
     *        Adds, per cell and per charged ion species,
     *           T_e += dt * (gamma-1) * Z_s e^2 eta n_s |dV|^2 / k_B,
     *        with the e-i relative drift dV = J_plasma/(e n_e) and eta from the
     *        Ohm's-law parser. Summed over species,
     *           S_e = e^2 eta n_e * Sigma_s Z_s n_s |dV|^2,
     *        which reduces to eta J^2 for a single species. n_s is recovered from
     *        the species charge fraction f_s = rho_fp_s / Sigma_t rho_fp_t
     *        = Z_s n_s/n_e (2pi*r-cancelling) -> n_s = f_s n_e/Z_s, with n_e from
     *        the total rho_fp. Deterministic, no per-particle scatter. Called
     *        when m_include_joule_heating is true. When the Te-threshold redirect
     *        is on (\c redirect_E non-null), the contribution from cells with
     *        Te >= m_joule_redirect_Te_eV is written into \c redirect_E for
     *        QDSMCApplyIonHeating to deposit on the ions instead of T_e.
     *
     * \param lev AMR refinement level.
     * \param dt  PIC timestep [s].
     * \param redirect_E nodal output field for the redirected heat, one component
     *        per charged ion species (ordered as in GetSpeciesNames, charged
     *        species only). Component c holds the m_i-independent energy
     *        E_s = (2/3) n_e Z_s e^2 eta |dV|^2 dt [J] (so sig_redir^2 = E_s/m_i).
     *        nullptr when the Joule redirect is off.
     */
    void QDSMCAddJouleHeating (int lev, amrex::Real dt,
                               amrex::MultiFab * redirect_E = nullptr) const;

    /**
     * \brief Add the electron-ion thermal-equilibration sink Q_ei to T_e.
     *
     * Per ion species, cools T_e toward that species' (deposited) T_i:
     *   dT_e = -dt (gamma-1) * 3 * (n_s/n_e) * nu_ei * (T_e - T_i_s),
     * with nu_ei from the electron_ion_relaxation_rate parser. This is the
     * ELECTRON-SIDE sink; QDSMCApplyIonHeating deposits the conjugate ion gain
     * so the pair conserves energy. Called when m_include_temperature_relaxation
     * is true.
     *
     * \param lev AMR refinement level.
     * \param dt  PIC timestep [s].
     * \param Ti_dep_by_species per-charged-species ion temperature [eV], deposited
     *        once by the caller (AdvanceElectronEnergyQDSMC) and shared with
     *        QDSMCApplyIonHeating, which runs immediately after with no intervening
     *        ion motion -- so the deposit is identical for both and is done only once.
     */
    void QDSMCAddTemperatureRelaxation (int lev, amrex::Real dt,
        std::map<std::string, amrex::MultiFab*> const & Ti_dep_by_species) const;

    /**
     * \brief Ion-heating operator: a stochastic drag-diffusion applied to
     *        every ion. It delivers both electron-ion energy channels:
     *
     *   dv_p = -nu_ei (v_p - u_e) dt + sqrt(2 D dt) R,   D = D_qei + D_redir,
     *
     * with R a unit-variance Gaussian vector. The Q_ei conjugate (when
     * m_include_temperature_relaxation) supplies the drag toward the electron
     * fluid u_e and the thermal diffusion D_qei = nu_ei k_B T_e / m_i, relaxing
     * the ions toward a Maxwellian at T_e (T_i -> T_e at rate 2 nu_ei). The
     * Te-threshold Joule redirection (when \c redirect_E is non-null) supplies
     * an additional pure-diffusion heating sig_redir^2 = E_s/m_i that injects,
     * in expectation, the redirected electron energy per cell. Either or both
     * channels may be active. Both are per-species correct (the redirect reads
     * each species' own component of \c redirect_E).
     *
     * \param lev AMR refinement level.
     * \param dt  PIC timestep [s].
     * \param redirect_E per-cell redirected energy [J], one component per charged
     *        ion species (ordered as in QDSMCAddJouleHeating); nullptr when off.
     * \param Ti_dep_by_species per-charged-species ion temperature [eV], deposited
     *        once by the caller and shared with QDSMCAddTemperatureRelaxation
     *        (see that method); nullptr when the Q_ei relaxation channel is off.
     */
    void QDSMCApplyIonHeating (int lev, amrex::Real dt,
                               amrex::MultiFab const * redirect_E = nullptr,
                               std::map<std::string, amrex::MultiFab*> const * Ti_dep_by_species = nullptr) const;

    /**
     * \brief Fill hybrid_electron_pressure_fp from T_e and n_e using the
     *        ideal-gas relation Pe = n_e * k_B * T_e. Called at the end of
     *        each QDSMC step so the existing Ohm's-law E-solver consumes
     *        the QDSMC-updated pressure without modification.
     */
    void FillPeFromTe (int lev) const;

    /** Seed T_e = T_e0 (max(n_e, n_floor)/n0)^(gamma-1): the algebraic
     * adiabat evaluated on the floored density. This is the QDSMC transport's
     * zero-gradient state (K_e uniform everywhere, halo included), so the
     * energy-equation run starts free of spurious entropy mixing and from the
     * same electron pressure as the algebraic closure. */
    void SeedTeAdiabat (int lev) const;

    /**
     * \brief Top-level orchestrator for the QDSMC electron-energy step.
     *
     * Performs the full entropy transport over one PIC step (see Phys. Plasmas 31, 012902 (2024)):
     *   1. CalculatePlasmaCurrent so J_plasma is current
     *   2. QDSMCInitializeUe  (V_e from J_plasma and J_i)
     *   3. QDSMCInitializeKe  (K_e from T_e^n and rho^n)
     *   4. QDSMC particle SetV / SetK / PushX / DepositK / DepositField
     *   5. QDSMCUpdateTe      (recover T_e^{n+1} from K_arr / N_arr / rho^{n+1})
     *   6. QDSMCAddJouleHeating (if m_include_joule_heating is on):
     *      Joule source (Phys. Plasmas 31, 012902 (2024), Eq. 12) Sigma_s nu_{s,e} n_s m_s |V_s - V_e|^2 -> T_e
     *   7. FillPeFromTe (Pe = n_e * k_B * T_e)
     *   8. Reset QDSMC particles to home positions for the next step
     *
     * Called from HybridPICEvolveFields when m_solve_electron_energy_equation is true.
     */
    void AdvanceElectronEnergyQDSMC (amrex::Real dt) const;

    /** Explicit fluid electron-pressure advance (electron_energy_solver = fluid): forward
     *  Euler over one accepted B substep dt_sub with the conservative enthalpy flux, the
     *  u_e.grad(pe) work term, the Joule deposit and the kappa_e conduction; a_Jfield and
     *  a_rho are the ion current and charge density of the half step. Collocated only.
     *  Shares the closure options of the theta-implicit in-loop advance: the Holmstrom
     *  vacuum blend of the advanced pe toward the floored adiabat (holmstrom_vacuum_region),
     *  the Te-gated Joule deposit (joule_redirect_Te_threshold / _width, the gated-out part
     *  staged per charged species in m_pe_redirect_E and handed to the ions by
     *  QDSMCApplyIonHeating in FinishElectronPressureFluid). */
    void AdvanceElectronPressureFluid (int lev, amrex::Real dt_sub, amrex::Real t_eval,
                                       ablastr::fields::VectorField const& a_Jfield,
                                       amrex::MultiFab const& a_rho) const;

    /** filter_push_fields: a_apply = true saves the unfiltered E and binomial-filters
     *  Efield_fp in place for the particle gather; a_apply = false restores it. */
    void FilterPushFieldsSwap (bool a_apply) const;

    /** End-of-step bookkeeping of the explicit fluid pe advance: pin the floored halo,
     *  mirror T_e = pe/(n k_B) and apply the optional Q_ei exchange once per step. */
    void FinishElectronPressureFluid (amrex::Real dt) const;

    /** true when the electron pressure is advanced by the explicit fluid update */
    [[nodiscard]] bool FluidElectronPressure () const {
        return m_solve_electron_energy_equation && (m_electron_energy_solver == 1);
    }

    /**
     * \brief Compute the per-species resistive overlay vector field added to
     *        Ohm's-law E alongside the existing +eta_global * J term:
     *
     *           overlay_d(i,j,k) = Sigma_s [ eta_s_per * rho_s * rho / rho_sum
     *                                       * (V_s_d - V_e_d) ]
     *
     *        summed over charged species that have a per-species
     *        resistivity parser registered. The total friction term that
     *        Ohm's law sees becomes
     *
     *           E_friction = +eta_global * J + overlay
     *                      = +e * Sigma_s Z_s * (eta_global + eta_s_per) * n_s
     *                                       * (V_s - V_e)
     *
     *        i.e. effectively eta_s_eff = eta_global + eta_s_per per species.
     *
     *        Output multifabs are at the Jx/Jy/Jz staggering (== E
     *        component staggering). When m_has_per_species_eta is false
     *        the overlay is set to zero and the function returns early
     *        (single-eta behaviour is exactly recovered when callers add
     *        a field of zeros to E).
     *
     *        Computed once per step (from HybridPICEvolveFields, into the
     *        registered hybrid_eta_overlay_fp fields) and held frozen across
     *        the subcycled E-solves, which only read it. Its inputs (Vs_fp,
     *        Ve_fp, rho_fp_{spec}, T_e) are per-step quantities anyway; only
     *        the |J| and |B| parser arguments would vary per substep, and
     *        those are frozen along with the rest. Caller passes
     *        pre-allocated Yee-staggered MultiFabs (one per direction) sized
     *        to match Jx/Jy/Jz at this level.
     */
    void ComputeResistiveOverlay (
        int lev,
        amrex::MultiFab & overlay_x,
        amrex::MultiFab & overlay_y,
        amrex::MultiFab & overlay_z) const;

    /** Check if rkf45 should be used */
    [[nodiscard]] bool DoRKF45(int step) const { return m_rkf45_intervals.contains(step); }

    // Declare variables to hold hybrid-PIC model parameters
    /** Number of substeps to take when evolving B (also used as initial substep
     *  count guess when RKF45 adaptive stepping is active) */
    int m_substeps = 10;

    /** Intervals to use RKF45 integrator and to update substeps parameter */
    ablastr::utils::text::IntervalsParser m_rkf45_intervals;
    /** Relative tolerance for RKF45 adaptive substep error control */
    amrex::Real m_substep_rtol = 1.e-4;
    /** Absolute tolerance for RKF45 adaptive substep error control */
    amrex::Real m_substep_atol = 1.e-8;
    /** Safety factor for RKF45 step-size adjustment */
    amrex::Real m_substep_safety = 0.9;
    /** Maximum step-size growth factor per accepted step */
    amrex::Real m_substep_max_growth = 5.0;
    /** Maximum number of substep attempts (accepted + rejected) before abort */
    int m_max_substep_attempts = 250;

    bool m_holmstrom_vacuum_region = false;
    /** hybrid_pic_model.pec_normal_E_from_interior (default false; collocated grids):
     *  the electric-field component normal to a perfectly conducting wall is taken
     *  at the wall node from the first interior node instead of Ohm's law at the
     *  wall node. On the collocated grid the wall node carries the normal component,
     *  and Ohm's law there closes through one-sided radial/normal derivatives of the
     *  tangential magnetic field; with plasma on the wall the Hall term then feeds a
     *  wall-node current sheet that steepens along the wall without the dispersion an
     *  interior node has. The staggered (Yee) grid has no normal E on the wall. */
    bool m_pec_normal_E_from_interior = false;
    /** hybrid_pic_model.vacuum_weight_r_ref [m] (default 0 = off; RZ only): the density
     *  floor that enters the vacuum weight is raised to
     *  n_floor * min(vacuum_weight_max_factor, max(1, r_ref / r)), the density at which a
     *  node holds the same number of macro-particles as a node at r_ref does at n_floor
     *  (uniform-weight loading: the count per node scales with n r). Near the axis a cell
     *  at the floor density holds a fraction of a macro-particle and Ohm's law and the
     *  electron pressure advance there are pure noise. */
    amrex::Real m_vacuum_weight_r_ref = 0.0;
    amrex::Real m_vacuum_weight_max_factor = 10.0;

    /** Electron temperature in eV */
    amrex::Real m_elec_temp;
    /** Reference electron density */
    amrex::Real m_n0_ref = 1.0;
    /** Electron pressure scaling exponent */
    amrex::Real m_gamma = 5.0/3.0;

    /** Master gate for the electron-energy equation. When true, K_e is
     * transported each step by QDSMC fictitious particles advecting with
     * V_e, then T_e is recovered from K_e and n_e via the polytropic
     * relation, then the Joule-heating source (if enabled) is added,
     * then Pe = n_e k_B T_e is emitted for the Ohm's-law E-solve. When
     * false (default), Pe is computed from the algebraic polytropic closure
     * (FillElectronPressureMF): T_e is not an evolved state variable, but
     * the implied temperature Pe/(n_e k_B) still varies with density
     * through the closure (with the chosen gamma, which need not be
     * adiabatic). */
    bool m_solve_electron_energy_equation = false;

    /** hybrid_pic_model.electron_energy_solver = qdsmc (default) | fluid, explicit scheme
     *  only: qdsmc transports the electron entropy with fictitious particles, fluid
     *  advances pe with AdvanceElectronPressureFluid inside the B substep loop. */
    int m_electron_energy_solver = 0;

    /** hybrid_pic_model.pe_advection (fluid solver): face reconstruction of the enthalpy
     *  flux, 0 = central, 1 = van Albada limited upwind (default) */
    int m_pe_advection = 1;
    /** hybrid_pic_model.pe_ue_cap (fluid solver): tanh soft cap [m/s] on the flux
     *  velocity u_e, 0 = off */
    amrex::Real m_pe_ue_cap = 0.0;
    /** pe row scale when the theta-implicit scheme evolves pe as a Newton unknown (set by
     *  the scheme, read by the preconditioner); 1.0 otherwise */
    amrex::Real m_pe_newton_scale = 1.0;
    /** scratch copy of pe^s for the explicit fluid update */
    mutable std::unique_ptr<amrex::MultiFab> m_pe_fluid_scratch;
    /** pe at entry of the half step, restored when a NaN in the RK4 B
     *  substepping rewinds the half step to restart it with RKF45 */
    mutable std::unique_ptr<amrex::MultiFab> m_pe_fluid_entry;
    /** fluid solver + Te-threshold redirect: per-ion redirected Joule energy E_s [J]
     *  (one component per charged species, ordered as in QDSMCAddJouleHeating), nodal, no
     *  ghosts, accumulated over the B substeps of the step and applied to the ions once
     *  per step in FinishElectronPressureFluid; m_pe_redirect_entry is its half-step
     *  entry snapshot for the RKF45 rewind. */
    mutable std::unique_ptr<amrex::MultiFab> m_pe_redirect_E;
    mutable std::unique_ptr<amrex::MultiFab> m_pe_redirect_entry;
    static constexpr int kMaxFluidRedirectSpecies = 8;
    mutable int m_pe_redirect_nspec = 0;
    mutable amrex::GpuArray<amrex::Real, kMaxFluidRedirectSpecies> m_pe_redirect_Z{};
    /** hybrid_pic_model.filter_push_fields (requires warpx.use_filter): the ions gather
     *  the binomial-filtered E while Ohm's law and Faraday keep the unfiltered field;
     *  explicit counterpart of implicit_evolve.filter_push_fields. */
    bool m_filter_push_fields = false;
    /** unfiltered E copies for the filter_push_fields swap */
    mutable std::array<std::unique_ptr<amrex::MultiFab>, 3> m_E_unfiltered;

    /** Resistive electron-heating source on T_e (see Phys. Plasmas 31, 012902 (2024), Eq. 12),
     * computed per cell as
     *     S_e = Sigma_s nu_{s,e} n_s m_s |V_s - V_e|^2,
     *         nu_{s,e} = Z_s e^2 eta n_e / m_s,
     * from the gridded Vs_fp_s, Ve_fp, rho_fp(_s) fields and the
     * Ohm's-law eta parser. Single bool: on or off. Reduces exactly to
     * eta J^2 in single species; resolves inter-species drift-dispersion
     * heating in multispecies. Orthogonal to whether HybridResistiveDrag
     * is registered on ions (that controls particle dynamics, not the
     * T_e source). Only consulted when m_solve_electron_energy_equation
     * is also true. */
    bool m_include_joule_heating = false;

    /** hybrid_pic_model.kappa_e: electron thermal conductivity [1/(m s)] in the heat flux
     *  q_e = -kappa_e grad(pe/n) of the energy equation (0 = off); kappa_e(rho,Te) gives
     *  it as a function of the charge density [C/m^3] and Te [eV]. Ignored by QDSMC. */
    amrex::Real m_kappa_e = 0.0;
    std::string m_kappa_e_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_kappa_e_parser;
    /** compiled kappa_e(rho,Te) executor (rho [C/m^3], Te [eV]) */
    amrex::ParserExecutor<2> m_kappa;
    /** a kappa_e(rho,Te) expression was provided (else the numeric m_kappa_e is used) */
    bool m_has_kappa_e_expression = false;
    /** conduction active: numeric kappa_e != 0 or expression provided */
    bool m_has_kappa_e = false;

    /** hybrid_pic_model.te_seed_uniform = 1: seed the energy-equation T_e isothermal
     *  (Te = elec_temp everywhere) instead of on the adiabat Te0 (n/n0_ref)^(gamma-1). */
    bool m_te_seed_uniform = false;

    /** Te-threshold Joule redirection. When active, the eta*J^2 Joule source in
     * QDSMCAddJouleHeating heats electrons in cells where Te < m_joule_redirect_Te_eV
     * and is instead deposited to the ions (as stochastic pure-diffusion heating in
     * QDSMCApplyIonHeating) where Te >= m_joule_redirect_Te_eV. This caps electron
     * heating at the threshold and routes the rest to the ions, so Ti can exceed Te.
     * The redirected energy is bookkept per charged ion species. Requires
     * m_include_joule_heating. Enabled by specifying a threshold >= 0 via
     * joule_redirect_Te_threshold; default off (threshold < 0). The bool is
     * derived from the threshold in ReadParameters, not a user input. */
    bool m_joule_redirect_to_ions = false;
    amrex::Real m_joule_redirect_Te_eV = -1.0;   // threshold [eV]; < 0 -> off
    /** hybrid_pic_model.joule_redirect_Te_width (default 0 = hard step at the threshold):
     *  with w > 0 the electron/ion split of the Joule heat is the smooth gate
     *  sJ = 0.5 (1 - tanh((Te - Tc)/(w Tc))) (electrons receive sJ, the ions 1 - sJ),
     *  the convention of the implicit scheme's implicit_evolve.joule_Te_cutoff_width, so
     *  an explicit run can use the identical closure as its implicit twin. */
    amrex::Real m_joule_redirect_Te_width = 0.0;

    /** Electron-ion thermal equilibration (Q_ei) on T_e:
     *   Q_ei = 3 n_e k_B nu_ei (T_e - T_i),   dU_e/dt += -Q_ei,
     * cooling T_e toward each ion species' T_i at the rate given by the
     * electron_ion_relaxation_rate(rho,Te,Ti,t) parser, with the matching ion
     * heating deposited conservatively (QDSMCApplyIonHeating) so the exchange
     * conserves energy. Enabled by specifying the
     * electron_ion_relaxation_rate expression (the bool is derived in
     * ReadParameters, not a separate user input); only consulted when
     * m_solve_electron_energy_equation is on. */
    bool m_include_temperature_relaxation = false;

    /** Electron-ion energy-equilibration rate nu_ei(rho,Te,Ti,t) [1/s] used by
     * the Q_ei term (rho = charge density [C/m^3]; Te, Ti in eV; t in s). */
    std::string m_nu_ei_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_nu_ei_parser;
    amrex::ParserExecutor<4> m_nu_ei;

    /** Density floor used when dividing by the deposited QDSMC weight to
     * recover K_e^{n+1} from (K*N) / N. Avoids divide-by-zero in cells
     * that no QDSMC particle reached during the push. Defaults to the
     * same floor as the rest of the hybrid solver. */
    amrex::Real m_qdsmc_n_floor = 1.0;

    /** Implicit scheme only: use the algebraic gamma-law closure
     * Pe = n0 kB Te0 (n/n0)^gamma, re-evaluated from the iterate's rho in
     * every residual, instead of the in-loop pe advance -- the exact
     * counterpart of the explicit path's default closure. Sidesteps the
     * in-loop advance's work-pairing artifacts at floored edges (Yee Hall
     * residual, Cartesian m=4 separatrix layer) at the cost of the discrete
     * electron energy ledger. Default off (in-loop advance). */
    bool m_implicit_use_algebraic_closure = false;

    /** Plasma density floor - if n < n_floor it will be set to n_floor */
    amrex::Real m_n_floor = 1.0;
    /** Width of the C1 smooth-max blend of the density floor, in units of
     *  n_floor (0 = exact hard max, bit-compatible legacy behavior). The
     *  hard max(rho, rho_floor) is a C0 kink in the Ohm residual at the
     *  floor crossing (and the conditional external-E subtraction a jump);
     *  cells whose deposit chatters across the threshold stall Newton.
     *  The blend keeps the halo response FLOORED but smooth. */
    amrex::Real m_n_floor_smooth_width = 0.0;

    /** Plasma resistivity */
    std::string m_eta_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_resistivity_parser;
    amrex::ParserExecutor<3> m_eta;
    bool m_resistivity_has_J_dependence = false;

    /** Per-species resistivity overlay (see Phys. Plasmas 31, 012902 (2024), Eq. 10).
     *  Optional. For any charged species {spec} for which the user provides
     *
     *      hybrid_pic_model.plasma_resistivity_{spec}(rho_s,rho,Te,J,J_s,B,t) = "..."
     *
     *  the parser is compiled and stored here. The total per-species
     *  effective resistivity used by Ohm's law, the Joule-heating source,
     *  and the resistive-drag operator is
     *
     *      eta_s_eff = eta_global(rho, |J|, t)              [the global m_eta]
     *                + eta_per_species_s(rho_s, rho, Te,    [this map]
     *                                    |J|, |J_s|, |B|, t)
     *
     *  so the global parser remains the shared anomalous (e.g. Chodura)
     *  layer and the per-species parser overlays species-resolved physics
     *  (e.g. classical Spitzer with Z^2 scaling). Species without a
     *  per-species parser registered fall back to the global parser only.
     *  Parser variables, in order, are:
     *    rho_s : per-species charge density        (C/m^3, unscaled in RZ)
     *    rho   : total plasma charge density       (C/m^3, volume-scaled)
     *    Te    : electron temperature              (Kelvin)
     *    J     : |J_plasma|                        (A/m^2)
     *    J_s   : |J_{species}|                     (A/m^2, unscaled in RZ)
     *    B     : |B|                               (Tesla)
     *    t     : time                              (seconds)
     */
    std::map<std::string, std::unique_ptr<amrex::Parser>> m_per_species_resistivity_parser;
    std::map<std::string, amrex::ParserExecutor<7>>       m_eta_per_species;
    /** Fast-path flag: true if at least one charged species has a
     *  registered per-species resistivity parser. When false the
     *  per-species code paths short-circuit to the global parser only,
     *  preserving the existing single-eta behaviour with zero overhead.
     *  Set from the input deck in ReadParameters (so AllocateLevelMFs can
     *  gate the overlay fields) and confirmed in InitData when the parsers
     *  are compiled. */
    bool m_has_per_species_eta = false;

    /** True when a hybrid_resistive_drag collision is present in the input
     *  deck. Set in ReadParameters; used for the drag/Joule-heating
     *  consistency warnings in InitData. */
    bool m_has_resistive_drag = false;

    /** False until the first AdvanceElectronEnergyQDSMC of the run has
     *  filled hybrid_current_fp_plasma. On later steps J_plasma is already
     *  valid from the previous step's final E-solve (B is unchanged in
     *  between), so the QDSMC entry-point recompute is skipped. Mutable
     *  because it is a lazily-set cache flag inside a const call chain. */
    mutable bool m_qdsmc_J_plasma_valid = false;

    /** True when the gridded fluid velocities Ve_fp / Vs_fp_{spec} are
     *  needed, i.e. when a per-species resistivity parser is registered or
     *  a hybrid_resistive_drag collision is present in the input deck.
     *  Gates their allocation and the once-per-step
     *  CalculateElectronFluidVelocity / CalculateIonFluidVelocity calls.
     *  Set in ReadParameters. */
    bool m_need_fluid_velocities = false;

    /** True when the per-species deposited fields current_fp_{spec} /
     *  rho_fp_{spec} are needed, i.e. when m_need_fluid_velocities is true
     *  or the electron-energy equation is solved (its Joule and Q_ei
     *  sources read the per-species charge densities). Gates their
     *  allocation and the per-species deposition path in
     *  HybridPICDepositRhoAndJ; when false, the deposition falls back to
     *  the single-pass MultiParticleContainer path with zero extra cost.
     *  Set in ReadParameters. */
    bool m_need_per_species_fields = false;

    /** Plasma hyper-resisitivity */
    std::string m_eta_h_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_hyper_resistivity_parser;
    amrex::ParserExecutor<2> m_eta_h;
    bool m_include_hyper_resistivity_term = false;
    bool m_hyper_resistivity_has_B_dependence = false;

    /** External current */
    std::string m_Jx_ext_grid_function = "0.0";
    std::string m_Jy_ext_grid_function = "0.0";
    std::string m_Jz_ext_grid_function = "0.0";
    std::array< std::unique_ptr<amrex::Parser>, 3> m_J_external_parser;
    std::array< amrex::ParserExecutor<4>, 3> m_J_external;
    bool m_has_external_current = true;
    bool m_external_current_has_time_dependence = false;


    /** External E/B fields */
    bool m_add_external_fields = false;
    std::unique_ptr<ExternalVectorPotential> m_external_vector_potential;

    /** Gpu Vector with index type of the Jx multifab */
    amrex::GpuArray<int, 3> Jx_IndexType;
    /** Gpu Vector with index type of the Jy multifab */
    amrex::GpuArray<int, 3> Jy_IndexType;
    /** Gpu Vector with index type of the Jz multifab */
    amrex::GpuArray<int, 3> Jz_IndexType;
    /** Gpu Vector with index type of the Bx multifab */
    amrex::GpuArray<int, 3> Bx_IndexType;
    /** Gpu Vector with index type of the By multifab */
    amrex::GpuArray<int, 3> By_IndexType;
    /** Gpu Vector with index type of the Bz multifab */
    amrex::GpuArray<int, 3> Bz_IndexType;
    /** Gpu Vector with index type of the Ex multifab */
    amrex::GpuArray<int, 3> Ex_IndexType;
    /** Gpu Vector with index type of the Ey multifab */
    amrex::GpuArray<int, 3> Ey_IndexType;
    /** Gpu Vector with index type of the Ez multifab */
    amrex::GpuArray<int, 3> Ez_IndexType;

    /** QDSMC fictitious-particle container used to transport the electron
     *  entropy K_e by V_e over one PIC step. Lazily constructed in InitData
     *  when m_solve_electron_energy_equation is true; otherwise nullptr.
     *  Owned by HybridPICModel
     *  because its lifetime is tied to the electron-energy equation that
     *  the hybrid model is responsible for. */
    std::unique_ptr<QdsmcParticleContainer> m_qdsmc_pc;
};

/**
 * \brief
 * This struct contains only static functions to compute the electron pressure
 * using the particle density at a given point and the user provided reference
 * density and temperatures.
 */
struct ElectronPressure {

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real get_pressure (amrex::Real const n0,
                                     amrex::Real const T0,
                                     amrex::Real const gamma,
                                     amrex::Real const rho) {
        // clamp at zero: the fractional power is NaN for rho < 0, and the
        // theta-implicit mass-matrix continuity response can swing a
        // near-empty cell (e.g. just emptied by an absorbing boundary)
        // transiently negative inside the matvec
        return n0 * T0 * std::pow(
            (amrex::max(rho, amrex::Real(0.0))/PhysConst::q_e)/n0, gamma);
    }
};

#endif // WARPX_HYBRIDPICMODEL_H_
