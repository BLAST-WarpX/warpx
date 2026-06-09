/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *          S. Eric Clark (Helion Energy)
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
    HybridPICModel ();
    // Defined out-of-line in the .cpp so the unique_ptr<QdsmcParticleContainer>
    // member can destroy a forward-declared particle container.
    ~HybridPICModel ();

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
        amrex::Real dt_half, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void BfieldEvolve (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        amrex::Real dt_half, int lev, SubcyclingHalf subcycling_half,
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
     * grid for every charged species. Output is stored in Vs_fp_<species>
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

    /**
     * \brief Topanga (Belyaev 2024 Eq. 12) resistive electron-heating source.
     *        Adds, per cell,
     *           T_e += dt * (gamma-1) * S_e / (n_e * k_B)
     *        with
     *           S_e = Σ_s ν_{s,e} n_s m_s |V_s - V_e|^2
     *               = e^2 η n_e · Σ_s Z_s n_s |V_s - V_e|^2
     *        where ν_{s,e} = Z_s e^2 η n_e / m_s is derived from the same
     *        Ohm's-law eta parser. Iterates over every charged ion species.
     *        Inputs per cell:
     *          * V_e: hybrid_electron_velocity_fp (nodal, freshly written
     *            this step by QDSMCInitializeUe).
     *          * V_s: J_s/rho_s computed inline from current_fp_<s> and
     *            rho_fp_<s> (the 2π·r factors cancel in this ratio, so V_s
     *            is in correct m/s without explicit RZ scaling).
     *          * n_s: recovered from the species charge fraction
     *            f_s = rho_fp_<s> / Σ_t rho_fp_<t> = Z_s n_s/n_e (unitless,
     *            2π·r-cancelling) -> n_s = f_s · n_e / Z_s, where n_e is
     *            from the volume-scaled total rho_fp.
     *          * η: Ohm's-law parser, evaluated with rho_fp and |J|.
     *
     *        Reduces algebraically to S = η J² in single species and
     *        resolves inter-species drift-dispersion heating in multispecies
     *        (Cauchy-Schwarz upper bound on η J²). Deterministic, no per-
     *        particle scatter, no accumulator field. Called when
     *        m_include_joule_heating is true.
     */
    void QDSMCAddJouleHeating (int lev, amrex::Real dt) const;

    /**
     * \brief Fill hybrid_electron_pressure_fp from T_e and n_e using the
     *        ideal-gas relation Pe = n_e * k_B * T_e. Called at the end of
     *        each QDSMC step so the existing Ohm's-law E-solver consumes
     *        the QDSMC-updated pressure without modification.
     */
    void QDSMCFillElectronPressureFromTe (int lev) const;

    /**
     * \brief Top-level orchestrator for the QDSMC electron-energy step.
     *
     * Performs the full Belyaev-style entropy transport over one PIC step:
     *   1. CalculatePlasmaCurrent so J_plasma is current
     *   2. QDSMCInitializeUe  (V_e from J_plasma and J_i)
     *   3. QDSMCInitializeKe  (K_e from T_e^n and rho^n)
     *   4. QDSMC particle SetV / SetK / PushX / DepositK / DepositField
     *   5. QDSMCUpdateTe      (recover T_e^{n+1} from K_arr / N_arr / rho^{n+1})
     *   6. QDSMCAddJouleHeating (if m_include_joule_heating is on):
     *      Belyaev Eq. 12 Σ_s ν_{s,e} n_s m_s |V_s - V_e|^2 -> T_e
     *   7. QDSMCFillElectronPressureFromTe (Pe = n_e * k_B * T_e)
     *   8. Reset QDSMC particles to home positions for the next step
     *
     * Called from HybridPICEvolveFields when m_solve_electron_energy_equation is true.
     */
    void AdvanceElectronEnergyQDSMC (amrex::Real dt);

    /**
     * \brief Compute the per-species resistive overlay vector field added to
     *        Ohm's-law E alongside the existing +η_global · J term:
     *
     *           overlay_d(i,j,k) = Σ_s [ η_s_per · ρ_s · ρ / ρ_sum
     *                                       · (V_s_d − V_e_d) ]
     *
     *        summed over charged species that have a per-species
     *        resistivity parser registered. The total friction term that
     *        Ohm's law sees becomes
     *
     *           E_friction = +η_global · J + overlay
     *                      = +e · Σ_s Z_s · (η_global + η_s_per) · n_s
     *                                       · (V_s − V_e)
     *
     *        i.e. effectively η_s_eff = η_global + η_s_per per species.
     *
     *        Output multifabs are at the Jx/Jy/Jz staggering (== E
     *        component staggering). When m_has_per_species_eta is false
     *        the overlay is set to zero and the function returns early
     *        (single-η behaviour is exactly recovered when callers add
     *        a field of zeros to E).
     *
     *        Computed once per E-solve call. Caller passes pre-allocated
     *        Yee-staggered MultiFabs (one per direction) sized to match
     *        Jx/Jy/Jz at this level.
     */
    void ComputeResistiveOverlay (
        int lev,
        amrex::MultiFab & overlay_x,
        amrex::MultiFab & overlay_y,
        amrex::MultiFab & overlay_z) const;

    // Declare variables to hold hybrid-PIC model parameters
    /** Number of substeps to take when evolving B (also used as initial substep
     *  count guess when RKF45 adaptive stepping is active) */
    int m_substeps = 10;

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
    /** If true, use the adaptive RKF45 integrator; if false, use fixed-step RK4 */
    bool m_use_rkf45 = false;

    bool m_holmstrom_vacuum_region = false;

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
     * false (default), Pe is computed from the algebraic adiabatic closure
     * (FillElectronPressureMF) — no temperature evolution. */
    bool m_solve_electron_energy_equation = false;

    /** Topanga resistive electron-heating source on T_e (Belyaev 2024 Eq. 12),
     * computed per cell as
     *     S_e = Σ_s ν_{s,e} n_s m_s |V_s - V_e|^2,
     *         ν_{s,e} = Z_s e^2 η n_e / m_s,
     * from the gridded Vs_fp_<s>, Ve_fp, rho_fp(_s) fields and the
     * Ohm's-law eta parser. Single bool: on or off. Reduces exactly to
     * η J^2 in single species; resolves inter-species drift-dispersion
     * heating in multispecies. Orthogonal to whether HybridResistiveDrag
     * is registered on ions (that controls particle dynamics, not the
     * T_e source). Only consulted when m_solve_electron_energy_equation
     * is also true. */
    bool m_include_joule_heating = false;

    /** Density floor used when dividing by the deposited QDSMC weight to
     * recover K_e^{n+1} from (K*N) / N. Avoids divide-by-zero in cells
     * that no QDSMC particle reached during the push. Defaults to the
     * same floor as the rest of the hybrid solver. */
    amrex::Real m_qdsmc_n_floor = 1.0;

    /** Plasma density floor - if n < n_floor it will be set to n_floor */
    amrex::Real m_n_floor = 1.0;

    /** Plasma resistivity */
    std::string m_eta_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_resistivity_parser;
    amrex::ParserExecutor<3> m_eta;
    bool m_resistivity_has_J_dependence = false;

    /** Per-species resistivity overlay (Topanga-style, Belyaev 2024 Eq. 10).
     *  Optional. For any charged species <spec> for which the user provides
     *
     *      hybrid_pic_model.plasma_resistivity_<spec>(rho_s,rho,Te,J,J_s,B,t) = "..."
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
     *    J_s   : |J_<species>|                     (A/m^2, unscaled in RZ)
     *    B     : |B|                               (Tesla)
     *    t     : time                              (seconds)
     */
    std::map<std::string, std::unique_ptr<amrex::Parser>> m_per_species_resistivity_parser;
    std::map<std::string, amrex::ParserExecutor<7>>       m_eta_per_species;
    /** Fast-path flag: true if at least one charged species has a
     *  registered per-species resistivity parser. When false the
     *  per-species code paths short-circuit to the global parser only,
     *  preserving the existing single-eta behaviour with zero overhead. */
    bool m_has_per_species_eta = false;

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
        return n0 * T0 * std::pow((rho/PhysConst::q_e)/n0, gamma);
    }
};

#endif // WARPX_HYBRIDPICMODEL_H_
