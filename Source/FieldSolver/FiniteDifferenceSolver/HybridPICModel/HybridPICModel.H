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

#include "ElectronThermodynamics.H"
#include "HybridPICModel_fwd.H"

#include "Fields.H"
#include "Fluids/QdsmcParticleContainer_fwd.H"

#include "ExternalVectorPotential.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/MaterialRegistry_fwd.H"
#include "Utils/WarpXConst.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/text/IntervalsParser.H>

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_DistributionMapping.H>

#include <map>
#include <memory>
#include <optional>

/**
 * \brief This class contains the parameters needed to evaluate hybrid field
 * solutions (kinetic ions with fluid electrons).
 */
class HybridPICModel
{
public:
    explicit HybridPICModel (
        warpx::materials::MaterialRegistry const* material_registry = nullptr);
    // Defined out-of-line in the .cpp so the unique_ptr<QdsmcParticleContainer>
    // member can destroy a forward-declared particle container.
    ~HybridPICModel ();

    HybridPICModel ( HybridPICModel const & )             = delete;
    HybridPICModel& operator= ( HybridPICModel const & )  = delete;
    HybridPICModel ( HybridPICModel && )                  = delete;
    HybridPICModel& operator= ( HybridPICModel && )       = delete;

    /** Read user-defined model parameters. Called in constructor. */
    void ReadParameters (
        warpx::materials::MaterialRegistry const* material_registry);

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

    /** Allocate the particle-gather copy of the isolated pressure electric
     * field after WarpX has established the auxiliary E-field layouts. */
    void AllocateAuxiliaryLevelMFs (
        ablastr::fields::MultiFabRegister& fields,
        int lev) const;

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
        bool solve_for_Faraday) const;

    void HybridPICSolveE (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& Jfield,
        ablastr::fields::VectorField const& Bfield,
        amrex::MultiFab const& rhofield,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
        int lev, bool solve_for_Faraday) const;

    void HybridPICSolveE (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& Jfield,
        ablastr::fields::VectorField const& Bfield,
        amrex::MultiFab const& rhofield,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
        int lev, PatchType patch_type, bool solve_for_Faraday) const;

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
     * Function to calculate the electron pressure, and the closure's electron
     * temperature alongside it, from the simulation charge density. Used in the
     * Ohm's law solver (kinetic-fluid hybrid model).
     *
     * \param[in] floor_density if true, evaluate the closure on the floored
     *            density max(n_e, n_floor) instead of the raw n_e, so T_e
     *            flattens to T_e0 (n_floor/n0)^(gamma-1) below the floor rather
     *            than decaying towards zero with n_e. The QDSMC electron-energy
     *            equation needs that flattened state from its seed: it is the
     *            transport's zero-gradient state (K_e uniform, floored halo
     *            included), whereas a halo whose T_e decays to ~0 acts as an
     *            absorbing K_e = 0 boundary. Off by default, leaving the
     *            algebraic closure path evaluated on the raw density.
     */
    void CalculateElectronPressure (bool floor_density = false) const;
    void CalculateElectronPressure (int lev, bool floor_density = false) const;

    /**
     * \brief Fill the electron pressure multifab given the kinetic particle
     * charge density (and assumption of quasi-neutrality) using the user
     * specified electron equation of state. The closure's electron temperature,
     * T_e = T0 (n_e/n0)^(gamma-1), is filled at the same time (P_e = n_e T_e
     * is formed from it). That T_e is diagnostic-only on this path: with
     * solve_electron_energy_equation on, this function is not called and
     * T_e is owned by the QDSMC entropy transport, which fills Te/Pe at
     * this same point in the field loop.
     *
     * \param[out] Pe_field scalar electron pressure MultiFab at a given level
     * \param[out] Te_field scalar electron temperature MultiFab (in Kelvin) at a given level
     * \param[in] rho_field scalar ion charge density Multifab at a given level
     * \param[in] floor_density evaluate the closure at max(rho, q_e n_floor)
     *            instead of rho
     */
    void FillElectronPressureMF (
        amrex::MultiFab& Pe_field,
        amrex::MultiFab& Te_field,
        amrex::MultiFab const& rho_field,
        bool floor_density = false ) const;

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
     * \brief Initialize the electron-energy transport scratch field.
     *
     * Ideal-gas mode writes the entropy invariant
     * K_e = T_e n_e^(1-gamma) k_B/q_e. Nonlinear thermodynamics write the
     * physical internal-energy density U_e(rho,T_e) consumed directly by the
     * conservative finite-volume transport operator.
     */
    void QDSMCInitializeThermodynamicQuantity (int lev) const;

    /**
     * \brief Advance the thermodynamic state and recover T_e.
     *
     * Ideal-gas mode reconstructs T_e from deposited K_e N_e / N_e and the
     * new density. Supported nonlinear Cartesian modes source-load exact
     * particle-paired pressure work into U_e, advance it with a conservative
     * mass-consistent upwind finite-volume flux, apply the remaining fluid
     * pressure source, and use the configured exact U_e-to-T_e inverse. Ideal
     * cells receiving no marker weight retain their previous temperature.
     */
    void QDSMCUpdateThermodynamics (int lev, amrex::Real dt) const;

    /**
     * \brief Resistive electron-heating source (see Phys. Plasmas 31, 012902 (2024), Eq. 12).
     *        Adds, per cell and per charged ion species,
     *           Delta U_e = dt Z_s e^2 eta n_e n_s |dV|^2,
     *        with the e-i relative drift dV = J_plasma/(e n_e) and eta from the
     *        Ohm's-law parser. Summed over species,
     *           S_e = e^2 eta n_e * Sigma_s Z_s n_s |dV|^2,
     *        which reduces to eta J^2 for a single species. n_s is recovered from
     *        the species charge fraction f_s = rho_fp_s / Sigma_t rho_fp_t
     *        = Z_s n_s/n_e (2pi*r-cancelling) -> n_s = f_s n_e/Z_s, with n_e from
     *        the total rho_fp. The configured caloric inverse recovers T_e,
     *        and the realized energy change is recorded in step and cumulative
     *        ledgers. Deterministic, no per-particle scatter. Called
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
     * Per ion species, exchanges energy between the finite electron EOS heat
     * capacity and C_i=3 n_i k_B/2 using the exact frozen-coefficient
     * two-temperature solution. The realized electron EOS energy change is
     * ledgered explicitly. QDSMCApplyIonHeating consumes each per-species
     * ledger through a strict cell-local moment projection that preserves ion
     * momentum and closes the nonrelativistic pair-energy budget. Called when
     * m_include_temperature_relaxation is true.
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
     * \brief Ion-heating operator: an ion drag-diffusion proposal followed,
     *        for Qei, by a conservative local moment projection.
     *
     *   dv_p = -nu_ei (v_p - u_e) dt + sqrt(2 D dt) R,   D = D_qei + D_redir,
     *
     * with R a unit-variance Gaussian vector. The Q_ei proposal (when
     * Ti_dep_by_species is non-null) supplies the drag toward the electron
     * fluid u_e and the thermal diffusion D_qei = nu_ei k_B T_e / m_i, relaxing
     * ions toward a Maxwellian at T_e. Its realized velocities are recentered
     * and rescaled in every NGP particle cell so that the cell's old ion
     * momentum is preserved and its nonrelativistic thermal-energy change is
     * exactly the negative, physical-corner-volume integral of the electron
     * Qei ledger. Unsupported local requests abort; they are not moved to
     * distant occupied cells. The
     * Te-threshold Joule redirection (when \c redirect_E is non-null) supplies
     * an additional pure-diffusion heating sig_redir^2 = E_s/m_i that injects,
     * in expectation, the redirected electron energy per cell. Qei and Joule
     * redirection are invoked separately so the conservative projection has a
     * single unambiguous source. Both are per-species correct (the redirect
     * reads each species' own component of \c redirect_E).
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
    void QDSMCFillElectronPressureFromTe (int lev) const;

    /** Whether the hybrid electron temperature is an evolved state. */
    [[nodiscard]] bool
    solvesElectronEnergyEquation () const noexcept {
        return m_solve_electron_energy_equation;
    }

    /** Electron number-density floor used by the hybrid material model. */
    [[nodiscard]] amrex::Real
    electronDensityFloor () const noexcept {
        return m_n_floor;
    }

    /**
     * Whether finite-volume electron internal-energy transport uses the raw
     * deposited charge density in its ElectronOnly EOS state.
     *
     * The radiation LTE producer and its material-energy source must use the
     * same density convention. This is true for supported nonlinear
     * finite-volume transport and false for the ideal/entropy path, which
     * retains the configured electron-density floor.
     */
    [[nodiscard]] bool
    transportsElectronInternalEnergyAtRawDensity () const noexcept {
        return m_fv_transport_internal_energy;
    }

    /** Device-callable view of the configured electron thermodynamics. */
    [[nodiscard]] ElectronThermodynamicsExecutor
    electronThermodynamicsExecutor () const noexcept {
        return m_electron_thermodynamics.executor();
    }

    /** Number of fixed-charge ion materials supplying a tabulated electron EOS. */
    [[nodiscard]] int
    electronThermodynamicsNumMaterials () const noexcept {
        return m_electron_thermodynamics.numMaterials();
    }

    /** Particle species that supplies one tabulated material mass density. */
    [[nodiscard]] std::string const&
    electronThermodynamicsMaterialSpeciesName (int const material) const {
        return m_electron_thermodynamics.materialSpeciesName(material);
    }

    /** Whether the configured caloric EOS can accept conservative sources. */
    [[nodiscard]] bool
    electronThermodynamicsSupportsEnergyCoupling () const noexcept {
        return m_electron_thermodynamics.supportsEnergyCoupling();
    }

    /** Whether implicit LTE may use the constant-C_V material residual. */
    [[nodiscard]] bool
    electronThermodynamicsSupportsConstantHeatCapacityLte () const noexcept {
        return m_electron_thermodynamics.supportsConstantHeatCapacityLte();
    }

    /**
     * \brief Top-level orchestrator for the QDSMC electron-energy step.
     *
     * Performs thermodynamic transport over one PIC step. Ideal-gas mode uses
     * the entropy algorithm of Phys. Plasmas 31, 012902 (2024); nonlinear
     * thermodynamics use conservative finite-volume internal-energy transport:
     *   1. CalculatePlasmaCurrent so J_plasma is current
     *   2. QDSMCInitializeUe  (V_e from J_plasma and J_i)
     *   3. Initialize K_e or U_e from T_e^n and rho^n
     *   4. Transport ideal K_e with markers, or source-load paired pressure
     *      work and transport nonlinear U_e with mass-consistent FV fluxes
     *   5. Recover T_e^{n+1}; nonlinear mode includes the remaining
     *      -P_e div(V_e) contribution
     *   6. QDSMCAddJouleHeating (if m_include_joule_heating is on):
     *      Joule source (Phys. Plasmas 31, 012902 (2024), Eq. 12) Sigma_s nu_{s,e} n_s m_s |V_s - V_e|^2 -> U_e -> T_e
     *   7. QDSMCFillElectronPressureFromTe (Pe = n_e * k_B * T_e)
     *   8. Reset QDSMC particles to home positions for the next step
     *
     * Called from HybridPICEvolveFields when m_solve_electron_energy_equation is true.
     */
    void AdvanceElectronEnergyQDSMC (amrex::Real dt) const;

    /**
     * \brief Add a conservative cell-integrated energy source to the nodal
     *        hybrid-electron internal energy.
     *
     * The source is distributed to material corner nodes in proportion to
     * their old-state heat capacities. The configured thermodynamics maps
     * T_e to U_e, receives the energy increment, and inverts U_e back to T_e;
     * P_e is then rebuilt for Ohm's law. The supplied density threshold must
     * match the source producer's material mask. When supplied, the nonlinear
     * LTE remap stores a cell temperature increment and the exact EOS energy
     * represented by it; only its conservative root residual uses the fallback
     * heat-capacity map. This is deliberately separate from QDSMC entropy
     * transport and is called before AdvanceElectronEnergyQDSMC. The mutable
     * cell_integrated_energy enters as the requested radiation-to-material
     * transfer q in J/cell and exits rebased to the EOS-realized material
     * change d in J/cell. The [[nodiscard]] return is the MPI-global residual
     * R = sum(q-d), in joules.
     */
    [[nodiscard]] amrex::Real ApplyElectronEnergySource (
        int lev,
        amrex::MultiFab& cell_integrated_energy,
        amrex::Real minimum_electron_density,
        amrex::MultiFab const* nonlinear_lte_remap = nullptr) const;

    /** Conservatively apply hybrid ionization after the ion charge density
     * changes. The old electron internal-energy density is evaluated at
     * \p old_charge_density and \p electron_temperature, the deposited
     * ionization-potential density is removed, and the EOS is inverted at
     * \p new_charge_density. */
    void ApplyHybridIonizationEnergySource (
        int lev,
        amrex::MultiFab const& old_charge_density,
        amrex::MultiFab const& new_charge_density,
        amrex::MultiFab const& binding_energy_density) const;

    /** Check if rkf45 should be used */
    [[nodiscard]] bool DoRKF45(int step) const { return m_rkf45_intervals.contains(step); }

    // Declare variables to hold hybrid-PIC model parameters
    /** Number of substeps to take when evolving B (also used as initial substep
     *  count guess when RKF45 adaptive stepping is active). May increase under
     *  stress and slowly decay toward 2*n_attempts (absolute minimum 2). */
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

    /** Electron temperature: read from the input in eV, converted to
     * k_B T_e in J at the end of ReadParameters. */
    amrex::Real m_elec_temp;
    /** Reference electron density */
    amrex::Real m_n0_ref = 1.0;
    /** Electron pressure scaling exponent */
    amrex::Real m_gamma = 5.0/3.0;

    /** Electron pressure and caloric equation-of-state interface. */
    ElectronThermodynamics m_electron_thermodynamics;

    /** Master gate for the electron-energy equation. Ideal-gas QDSMC carries
     * K_e; supported nonlinear Cartesian thermodynamics advance internal
     * energy with a conservative finite-volume operator. Particle-paired
     * pressure work is source-loaded before that transport; remaining source
     * terms follow it, then P_e is emitted for Ohm's law. When false
     * (default), the algebraic polytropic closure is used. */
    bool m_solve_electron_energy_equation = false;

    /** Select conservative finite-volume U_e transport instead of ideal-gas
     * entropy markers. Derived from the thermodynamics model and supported
     * geometry/material constraints; not a user-facing approximation switch. */
    bool m_fv_transport_internal_energy = false;

    /** Instrument the full Boris ion push with the isolated pressure electric
     * field so its work can be exchanged conservatively with the nonlinear
     * electron internal-energy update. This experimental path is opt-in until
     * its application-level analytic validation matrix is complete. */
    bool m_conservative_pressure_work = false;

    /** Resistive electron-heating source on T_e (see Phys. Plasmas 31, 012902 (2024), Eq. 12),
     * computed per cell as
     *     S_e = Sigma_s nu_{s,e} n_s m_s |dV|^2,
     *         nu_{s,e} = Z_s e^2 eta n_e / m_s,   dV = J_plasma / (e n_e),
     * from the nodal plasma current, the per-species charge densities and
     * the Ohm's-law eta parser. Single bool: on or off. Reduces exactly to
     * eta J^2 in single species. Only consulted when
     * m_solve_electron_energy_equation is also true. */
    bool m_include_joule_heating = false;

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

    /** Electron-ion thermal equilibration (Q_ei) on T_e:
     *   Q_ei = 3 n_e k_B nu_ei (T_e - T_i),   dU_e/dt += -Q_ei,
     * cooling T_e toward each ion species' T_i at the rate given by the
     * electron_ion_relaxation_rate(rho,Te,Ti,t) parser, with the matching ion
     * electron energy change ledgered exactly. QDSMCApplyIonHeating projects
     * the finite-macroparticle ion proposal onto that ledger while preserving
     * cell ion momentum. Enabled by specifying the
     * electron_ion_relaxation_rate expression (the bool is derived in
     * ReadParameters, not a separate user input); only consulted when
     * m_solve_electron_energy_equation is on. */
    bool m_include_temperature_relaxation = false;

    /** Electron-ion energy-equilibration rate nu_ei(rho,Te,Ti,t) [1/s] used by
     * the Q_ei term (rho = charge density [C/m^3]; Te, Ti in eV; t in s). */
    std::string m_nu_ei_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_nu_ei_parser;
    amrex::ParserExecutor<4> m_nu_ei;

    /** Plasma density floor - if n < n_floor it will be set to n_floor */
    amrex::Real m_n_floor = 1.0;

    /** Plasma resistivity */
    std::string m_eta_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_resistivity_parser;
    amrex::ParserExecutor<3> m_eta;
    bool m_resistivity_has_J_dependence = false;

    /** False until the first AdvanceElectronEnergyQDSMC of the run has
     *  filled hybrid_current_fp_plasma. On later steps J_plasma is already
     *  valid from the previous step's final E-solve (B is unchanged in
     *  between), so the QDSMC entry-point recompute is skipped. Mutable
     *  because it is a lazily-set cache flag inside a const call chain. */
    mutable bool m_qdsmc_J_plasma_valid = false;

    /** True when the per-species deposited charge densities rho_fp_{spec}
     *  are needed, i.e. when the electron-energy equation is solved (its
     *  Joule and Q_ei sources read the per-species charge densities).
     *  Gates their allocation and the per-species deposition path in
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

#endif // WARPX_HYBRIDPICMODEL_H_
