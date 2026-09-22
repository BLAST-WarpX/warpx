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

#include "HybridPICModel.H"

#include "ElectronPressureFlux.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/utils/Communication.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLNodeTensorLaplacian.H>
#include <AMReX_MLMG.H>

#include "EmbeddedBoundary/Enabled.H"
#include "Python/callbacks.H"
#include "Fields.H"
#include "Fluids/QdsmcParticleContainer.H"
#include "Particles/MultiParticleContainer.H"
#include "ExternalVectorPotential.H"
#include "WarpX.H"

#include <AMReX_Random.H>

#include <string>
#include <vector>

using namespace amrex;
using warpx::fields::FieldType;

HybridPICModel::HybridPICModel ()
{
    ReadParameters();
}

HybridPICModel::~HybridPICModel () = default;

void HybridPICModel::ReadParameters ()
{
    const ParmParse pp_hybrid("hybrid_pic_model");

    // The B-field update is subcycled to improve stability - the number
    // of sub steps can be specified by the user.
    utils::parser::queryWithParser(pp_hybrid, "substeps", m_substeps);
    if (m_substeps % 2 != 0) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPIC",
            "hybrid_pic_model.substeps must be divisible by 2. "
            "The value " + std::to_string(m_substeps) + " is not valid. "
            "Automatically adjusting to " + std::to_string(m_substeps + 1) + ".",
            ablastr::warn_manager::WarnPriority::medium);
        m_substeps += 1;
    }

    // read rkf45 intervals
    std::vector<std::string> rkf45_intervals_string_vec = {"0"};
    pp_hybrid.queryarr("use_rkf45", rkf45_intervals_string_vec);
    m_rkf45_intervals = ablastr::utils::text::IntervalsParser(rkf45_intervals_string_vec);
    utils::parser::queryWithParser(pp_hybrid, "substep_rtol", m_substep_rtol);
    utils::parser::queryWithParser(pp_hybrid, "substep_atol", m_substep_atol);
    utils::parser::queryWithParser(pp_hybrid, "substep_safety", m_substep_safety);
    utils::parser::queryWithParser(pp_hybrid, "substep_max_growth", m_substep_max_growth);
    pp_hybrid.query("max_substep_attempts", m_max_substep_attempts);

    utils::parser::queryWithParser(pp_hybrid, "holmstrom_vacuum_region", m_holmstrom_vacuum_region);
    utils::parser::queryWithParser(pp_hybrid, "vacuum_weight_r_ref", m_vacuum_weight_r_ref);
    utils::parser::queryWithParser(pp_hybrid, "vacuum_weight_max_factor", m_vacuum_weight_max_factor);

    // The hybrid model requires an electron temperature, reference density
    // and exponent to be given. These values will be used to calculate the
    // electron pressure according to p = n0 * Te * (n/n0)^gamma
    utils::parser::queryWithParser(pp_hybrid, "gamma", m_gamma);
    if (!utils::parser::queryWithParser(pp_hybrid, "elec_temp", m_elec_temp)) {
        Abort("hybrid_pic_model.elec_temp must be specified when using the hybrid solver");
    }
    const bool n0_ref_given = utils::parser::queryWithParser(pp_hybrid, "n0_ref", m_n0_ref);
    if (m_gamma != 1.0 && !n0_ref_given) {
        Abort("hybrid_pic_model.n0_ref should be specified if hybrid_pic_model.gamma != 1");
    }

    pp_hybrid.query("plasma_resistivity(rho,J,t)", m_eta_expression);
    pp_hybrid.query("plasma_hyper_resistivity(rho,B)", m_eta_h_expression);
    pp_hybrid.query("pec_normal_E_from_interior", m_pec_normal_E_from_interior);

    utils::parser::queryWithParser(pp_hybrid, "n_floor", m_n_floor);
    utils::parser::queryWithParser(pp_hybrid, "n_floor_smooth_width",
                                   m_n_floor_smooth_width);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_n_floor_smooth_width >= 0.,
        "hybrid_pic_model.n_floor_smooth_width must be >= 0");

    // Master gate for the electron-energy equation. When enabled, K_e is
    // advected each step by fictitious Lagrangian particles moving with V_e
    // (see Phys. Plasmas 31, 012902 (2024)); T_e is recovered from K_e and n_e
    // via the polytropic relation; operator-split source terms are added;
    // Pe = n_e k_B T_e is emitted for the Ohm's-law E-solve. Default off
    // preserves the legacy algebraic adiabatic closure.
    pp_hybrid.query("solve_electron_energy_equation",
                    m_solve_electron_energy_equation);
    pp_hybrid.query("qdsmc_n_floor", m_qdsmc_n_floor);
    // Explicit-scheme electron energy solver: QDSMC entropy transport (default)
    // or the fluid pe update sub-stepped with B (see HybridPICModel.H).
    {
        std::string ees = "qdsmc";
        pp_hybrid.query("electron_energy_solver", ees);
        if (ees == "qdsmc") { m_electron_energy_solver = 0; }
        else if (ees == "fluid") { m_electron_energy_solver = 1; }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.electron_energy_solver = " + ees +
                " (expected qdsmc or fluid)");
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_electron_energy_solver == 0 || m_solve_electron_energy_equation,
            "hybrid_pic_model.electron_energy_solver = fluid requires "
            "hybrid_pic_model.solve_electron_energy_equation = 1");
        if (m_electron_energy_solver == 1) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.electron_energy_solver = fluid is implemented for "
                "Cartesian collocated grids only");
#endif
            std::string pe_adv_name = "vanalbada";
            pp_hybrid.query("pe_advection", pe_adv_name);
            if (pe_adv_name == "central") { m_pe_advection = 0; }
            else if (pe_adv_name == "vanalbada") { m_pe_advection = 1; }
            else {
                WARPX_ABORT_WITH_MESSAGE(
                    "hybrid_pic_model.pe_advection = " + pe_adv_name +
                    " (expected central or vanalbada)");
            }
            pp_hybrid.query("pe_ue_cap", m_pe_ue_cap);
        }
    }
    pp_hybrid.query("filter_push_fields", m_filter_push_fields);
    // Implicit scheme: algebraic gamma-law closure instead of the in-loop
    // pe advance (see the member doc in HybridPICModel.H).
    pp_hybrid.query("implicit_use_algebraic_closure",
                    m_implicit_use_algebraic_closure);
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_solve_electron_energy_equation,
        "hybrid_pic_model.solve_electron_energy_equation is not supported in "
        "RCYLINDER/RSPHERE geometries yet.");
#endif

    // Resistive electron-heating source (Phys. Plasmas 31, 012902 (2024), Eq. 12):
    //   S_e = Sigma_s nu_{s,e} n_s m_s |V_s - V_e|^2,  nu_{s,e} = Z_s e^2 eta n_e / m_s
    // added per cell to T_e by QDSMCAddJouleHeating, using the e-i relative
    // drift J_plasma/(e n_e) and rho_fp(_s). Reduces to eta J^2 in single species.
    // Independent of whether HybridResistiveDrag is registered.
    // Default off; only consulted when solve_electron_energy_equation is on.
    pp_hybrid.query("include_joule_heating", m_include_joule_heating);
    pp_hybrid.query("te_seed_uniform", m_te_seed_uniform);
    pp_hybrid.query("kappa_e", m_kappa_e);
    m_has_kappa_e_expression =
        pp_hybrid.query("kappa_e(rho,Te)", m_kappa_e_expression);
    m_has_kappa_e = m_has_kappa_e_expression || (m_kappa_e != 0.0_rt);

    // Te-threshold Joule redirection: heat electrons where Te < threshold,
    // deposit the Joule energy to ions where Te >= threshold. Off by default
    // (threshold < 0); specifying a threshold >= 0 enables the redirect.
    utils::parser::queryWithParser(pp_hybrid, "joule_redirect_Te_threshold", m_joule_redirect_Te_eV);
    m_joule_redirect_to_ions = (m_joule_redirect_Te_eV >= 0._rt);
    // Gate width as a fraction of the threshold (0 = hard step): with w > 0 the split is
    // the smooth gate sJ = 0.5 (1 - tanh((Te - Tc)/(w Tc))) of the implicit scheme
    // (implicit_evolve.joule_Te_cutoff_width), see m_joule_redirect_Te_width.
    utils::parser::queryWithParser(pp_hybrid, "joule_redirect_Te_width", m_joule_redirect_Te_width);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_joule_redirect_Te_width >= 0._rt,
        "hybrid_pic_model.joule_redirect_Te_width must be >= 0");

    // Electron-ion thermal equilibration (Q_ei) on T_e:
    //   Q_ei = 3 n_e k_B nu_ei (T_e - T_i),  applied per ion species weighted by
    //   n_s/n_e, cooling T_e toward T_i. nu_ei[1/s] comes from the
    //   electron_ion_relaxation_rate(rho,Te,Ti,t) parser (rho [C/m^3], Te,Ti [eV]).
    //   The matching ion heating is deposited conservatively, so the exchange
    //   conserves energy. Enabled by specifying the rate expression (only
    //   consulted when solve_electron_energy_equation is on).
    m_include_temperature_relaxation =
        pp_hybrid.query("electron_ion_relaxation_rate(rho,Te,Ti,t)", m_nu_ei_expression);

    // Determine, from the input deck alone, which optional per-species
    // machinery is needed (the particle containers do not exist yet at
    // parameter-parse time, but the species names are available):
    //   * a per-species resistivity overlay parser for any species,
    //   * a hybrid_resistive_drag collision on any species,
    //   * the electron-energy equation (its Joule and Q_ei sources read the
    //     per-species charge densities).
    // These flags gate the per-species field allocations, the per-species
    // rho/J deposition and the fluid-velocity computations, so hybrid-PIC
    // runs that use none of these features carry zero extra cost.
    {
        std::vector<std::string> species_names;
        const ParmParse pp_particles("particles");
        pp_particles.queryarr("species_names", species_names);
        for (auto const & spec_name : species_names) {
            std::string expr;
            if (pp_hybrid.query(
                "plasma_resistivity_" + spec_name + "(rho_s,rho,Te,J,J_s,B,t)",
                expr)) {
                m_has_per_species_eta = true;
            }
        }

        std::vector<std::string> collision_names;
        const ParmParse pp_collisions("collisions");
        pp_collisions.queryarr("collision_names", collision_names);
        for (auto const & coll_name : collision_names) {
            const ParmParse pp_coll(coll_name);
            std::string coll_type;
            pp_coll.query("type", coll_type);
            if (coll_type == "hybrid_resistive_drag") { m_has_resistive_drag = true; }
        }

        m_need_fluid_velocities   = m_has_per_species_eta || m_has_resistive_drag;
        // The QDSMC sources read the per-species charge densities; the fluid
        // pe solver needs them only for the Q_ei exchange.
        m_need_per_species_fields = m_need_fluid_velocities
                                  || (m_solve_electron_energy_equation
                                      && (m_electron_energy_solver == 0
                                          || m_include_temperature_relaxation));
    }
    // (the Te-threshold Joule redirect is supported by both electron energy solvers: the
    // QDSMC path stages it per step in QDSMCAddJouleHeating, the fluid path per B substep
    // in AdvanceElectronPressureFluid)

    // convert electron temperature from eV to J
    m_elec_temp *= PhysConst::q_e;

    // external currents
    pp_hybrid.query("Jx_external_grid_function(x,y,z,t)", m_Jx_ext_grid_function);
    pp_hybrid.query("Jy_external_grid_function(x,y,z,t)", m_Jy_ext_grid_function);
    pp_hybrid.query("Jz_external_grid_function(x,y,z,t)", m_Jz_ext_grid_function);

    // check if external currents are specified
    if ((m_Jx_ext_grid_function == "0.0") &&
        (m_Jy_ext_grid_function == "0.0") &&
        (m_Jz_ext_grid_function == "0.0"))
    {
        m_has_external_current = false;
    }


    // external fields
    pp_hybrid.query("add_external_fields", m_add_external_fields);

    if (m_add_external_fields) {
        m_external_vector_potential = std::make_unique<ExternalVectorPotential>();
    }
}

void HybridPICModel::AllocateLevelMFs (
    ablastr::fields::MultiFabRegister & fields,
    int lev, const BoxArray& ba, const DistributionMapping& dm,
    const int ncomps,
    const IntVect& ngJ, const IntVect& ngRho,
    const IntVect& ngEB,
    const IntVect& jx_nodal_flag,
    const IntVect& jy_nodal_flag,
    const IntVect& jz_nodal_flag,
    const IntVect& rho_nodal_flag,
    const IntVect& Ex_nodal_flag,
    const IntVect& Ey_nodal_flag,
    const IntVect& Ez_nodal_flag,
    const IntVect& Bx_nodal_flag,
    const IntVect& By_nodal_flag,
    const IntVect& Bz_nodal_flag) const
{
    using ablastr::fields::Direction;

    // The "hybrid_electron_pressure_fp" multifab stores the electron pressure
    // consumed by the Ohm's-law E-solve. With solve_electron_energy_equation
    // off, it is computed from the algebraic adiabatic closure each step. With
    // it on, Pe = n_e k_B T_e is emitted by FillPeFromTe
    // at the end of each QDSMC entropy-transport step.
    fields.alloc_init(FieldType::hybrid_electron_pressure_fp,
        lev, amrex::convert(ba, rho_nodal_flag),
        dm, ncomps, ngRho, 0.0_rt);

    // Electron temperature T_e (Kelvin). Allocated unconditionally (one
    // cheap scalar field) so the "Te" diagnostic can always read it: with
    // the energy equation on it is the QDSMC state variable, otherwise it
    // mirrors the closure's implied temperature T_e = P_e / (n_e k_B),
    // filled alongside P_e in CalculateElectronPressure.
    fields.alloc_init(FieldType::hybrid_electron_temperature_fp,
        lev, amrex::convert(ba, rho_nodal_flag),
        dm, ncomps, ngRho, 0.0_rt);

    // QDSMC electron-energy-equation working fields, only touched (and
    // therefore only allocated) when the energy equation is solved by the
    // QDSMC transport (the fluid solver works on pe in place):
    //   * hybrid_entropy_fp              : K_e = T_e * n_e^(1-gamma)
    //   * hybrid_qdsmc_weights_fp        : scratch for deposited N_e
    //   * hybrid_electron_velocity_fp    : three-component V_e on a NODAL
    //     grid, computed each step from V_e = -(J_plasma - J_i)/(q_e n_e)
    //     and consumed by the QDSMC particle SetV step to advect the
    //     entropy carriers.
    if (m_solve_electron_energy_equation && m_electron_energy_solver == 0) {
        fields.alloc_init(FieldType::hybrid_entropy_fp,
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);
        fields.alloc_init(FieldType::hybrid_qdsmc_weights_fp,
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);
    }
    // V_e is also needed by the fluid pe solver when the Q_ei exchange is on (the ion
    // drag toward u_e in QDSMCApplyIonHeating; filled by QDSMCInitializeUe in
    // FinishElectronPressureFluid)
    if (m_solve_electron_energy_equation
        && (m_electron_energy_solver == 0 || m_include_temperature_relaxation)) {
        fields.alloc_init(FieldType::hybrid_electron_velocity_fp, Direction{0},
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);
        fields.alloc_init(FieldType::hybrid_electron_velocity_fp, Direction{1},
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);
        fields.alloc_init(FieldType::hybrid_electron_velocity_fp, Direction{2},
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);
    }

    // The "hybrid_rho_fp_temp" multifab is used to store the ion charge density
    // interpolated or extrapolated to appropriate timesteps.
    fields.alloc_init(FieldType::hybrid_rho_fp_temp,
        lev, amrex::convert(ba, rho_nodal_flag),
        dm, ncomps, ngRho, 0.0_rt);

    // The "hybrid_current_fp_temp" multifab is used to store the ion current density
    // interpolated or extrapolated to appropriate timesteps.
    fields.alloc_init(FieldType::hybrid_current_fp_temp, Direction{0},
        lev, amrex::convert(ba, jx_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_current_fp_temp, Direction{1},
        lev, amrex::convert(ba, jy_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_current_fp_temp, Direction{2},
        lev, amrex::convert(ba, jz_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);

    // The "hybrid_current_fp_plasma" multifab stores the total plasma current calculated
    // as the curl of B minus any external current.
    fields.alloc_init(FieldType::hybrid_current_fp_plasma, Direction{0},
        lev, amrex::convert(ba, jx_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_current_fp_plasma, Direction{1},
        lev, amrex::convert(ba, jy_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_current_fp_plasma, Direction{2},
        lev, amrex::convert(ba, jz_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);

    // Per-species ion fields - one set per charged species. current_fp_s
    // and rho_fp_s are deposited from particles and accumulated into the
    // global current_fp / rho_fp; Vs_fp_s is the bulk velocity J_s/rho_s,
    // used by the resistive-drag operator to shift each species' particles
    // toward V_e without collapsing the thermal moment. Only allocated when
    // a feature that consumes them is active (see m_need_per_species_fields
    // and m_need_fluid_velocities).
    if (m_need_per_species_fields) {
        auto const & mypc = WarpX::GetInstance().GetPartContainer();
        for (auto const & spec : mypc.GetSpeciesNames()) {
            if (mypc.GetParticleContainerFromName(spec).getCharge() == 0._prt) { continue; }
            fields.alloc_init("current_fp_" + spec, Direction{0},
                lev, amrex::convert(ba, jx_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
            fields.alloc_init("current_fp_" + spec, Direction{1},
                lev, amrex::convert(ba, jy_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
            fields.alloc_init("current_fp_" + spec, Direction{2},
                lev, amrex::convert(ba, jz_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
            fields.alloc_init("rho_fp_" + spec,
                lev, amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngRho, 0.0_rt);
            if (m_need_fluid_velocities) {
                fields.alloc_init("Vs_fp_" + spec, Direction{0},
                    lev, amrex::convert(ba, jx_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
                fields.alloc_init("Vs_fp_" + spec, Direction{1},
                    lev, amrex::convert(ba, jy_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
                fields.alloc_init("Vs_fp_" + spec, Direction{2},
                    lev, amrex::convert(ba, jz_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
            }
        }
        // Species-summed raw charge density Sigma_s rho_fp_s, filled once per
        // step in HybridPICDepositRhoAndJ. Unlike rho_fp (which is filtered,
        // boundary-applied and volume-scaled in RZ by SyncCurrentAndRho), this
        // keeps the same raw-deposit form as the rho_fp_s numerators, so the
        // species fraction f_s = rho_s / Sigma_t rho_t is well-defined (the
        // RZ 2pi*r and filter factors cancel). Shared by the Joule, Q_ei and
        // per-species-resistivity consumers.
        fields.alloc_init("hybrid_rho_species_sum_fp",
            lev, amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngRho, 0.0_rt);
    }

    // Electron fluid velocity V_e on the grid, V_e = (J_i - J_plasma)/rho.
    // Face-staggered like J for direct gather by the resistive-drag operator.
    if (m_need_fluid_velocities) {
        fields.alloc_init("Ve_fp", Direction{0},
            lev, amrex::convert(ba, jx_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
        fields.alloc_init("Ve_fp", Direction{1},
            lev, amrex::convert(ba, jy_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
        fields.alloc_init("Ve_fp", Direction{2},
            lev, amrex::convert(ba, jz_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
    }

    // Species-summed resistive overlay added to Ohm's-law E: one vector
    // field holding Sigma_s of the per-species friction contributions (see
    // ComputeResistiveOverlay). Computed once per step and read by every
    // (subcycled) E-solve; staggered like J (== E component staggering).
    if (m_has_per_species_eta) {
        fields.alloc_init("hybrid_eta_overlay_fp", Direction{0},
            lev, amrex::convert(ba, jx_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
        fields.alloc_init("hybrid_eta_overlay_fp", Direction{1},
            lev, amrex::convert(ba, jy_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
        fields.alloc_init("hybrid_eta_overlay_fp", Direction{2},
            lev, amrex::convert(ba, jz_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
    }

    // the external current density multifab matches the current staggering and
    // one ghost cell is used since we interpolate the current to a nodal grid
    if (m_has_external_current) {
        fields.alloc_init(FieldType::hybrid_current_fp_external, Direction{0},
            lev, amrex::convert(ba, jx_nodal_flag),
            dm, ncomps, IntVect(1), 0.0_rt);
        fields.alloc_init(FieldType::hybrid_current_fp_external, Direction{1},
            lev, amrex::convert(ba, jy_nodal_flag),
            dm, ncomps, IntVect(1), 0.0_rt);
        fields.alloc_init(FieldType::hybrid_current_fp_external, Direction{2},
            lev, amrex::convert(ba, jz_nodal_flag),
            dm, ncomps, IntVect(1), 0.0_rt);
    }

    if (m_add_external_fields) {
        m_external_vector_potential->AllocateLevelMFs(
            fields,
            lev, ba, dm,
            ncomps, ngEB,
            Ex_nodal_flag, Ey_nodal_flag, Ez_nodal_flag,
            Bx_nodal_flag, By_nodal_flag, Bz_nodal_flag
        );
    }

#ifdef WARPX_DIM_RZ
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (ncomps == 1),
        "Ohm's law solver only support m = 0 azimuthal mode at present.");
#endif
}

void HybridPICModel::InitData (const ablastr::fields::MultiFabRegister& fields)
{
    m_resistivity_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_eta_expression, {"rho","J","t"}));
    m_eta = m_resistivity_parser->compile<3>();
    const std::set<std::string> resistivity_symbols = m_resistivity_parser->symbols();
    m_resistivity_has_J_dependence += resistivity_symbols.count("J");

    // Electron-ion energy-equilibration rate nu_ei(rho,Te,Ti,t) for the Q_ei term.
    m_nu_ei_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_nu_ei_expression, {"rho","Te","Ti","t"}));
    m_nu_ei = m_nu_ei_parser->compile<4>();

    // kappa_e(rho,Te) [1/(m s)]: always compiled (default "0.0") so the consumers
    // hold a valid executor; with only the numeric kappa_e set they use that.
    m_kappa_e_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_kappa_e_expression, {"rho","Te"}));
    m_kappa = m_kappa_e_parser->compile<2>();

    // --- Per-species resistivity overlay (Phys. Plasmas 31, 012902 (2024), Eq. 10) ---
    // Optional. For any charged species {spec} the user may supply
    //   hybrid_pic_model.plasma_resistivity_{spec}(rho_s,rho,Te,J,J_s,B,t)="..."
    // This parser overlays species-resolved physics on top of the global
    // m_eta parser; the total per-species effective resistivity used by
    // Ohm's law, the Joule-heating source, and the resistive-drag operator
    // is the sum eta_global + eta_per_species_s (so existing single-eta
    // input scripts keep their exact behaviour with eta_per_species_s = 0;
    // species without their own parser simply use eta_global). Done here in
    // InitData rather than ReadParameters because the particle containers
    // (needed to filter out uncharged species) do not exist at
    // parameter-parse time -- only the species names do, which is what the
    // allocation flags in ReadParameters are derived from.
    {
        const amrex::ParmParse pp_hybrid_init("hybrid_pic_model");
        auto const & mypc = WarpX::GetInstance().GetPartContainer();
        std::vector<std::string> species_with_per_eta;
        for (auto const & spec_name : mypc.GetSpeciesNames()) {
            if (mypc.GetParticleContainerFromName(spec_name).getCharge() == 0._prt) {
                continue;
            }
            std::string const param_name =
                "plasma_resistivity_" + spec_name + "(rho_s,rho,Te,J,J_s,B,t)";
            std::string expr;
            if (pp_hybrid_init.query(param_name, expr)) {
                auto parser = std::make_unique<amrex::Parser>(
                    utils::parser::makeParser(
                        expr, {"rho_s","rho","Te","J","J_s","B","t"}));
                m_eta_per_species[spec_name] = parser->compile<7>();
                m_per_species_resistivity_parser[spec_name] = std::move(parser);
                species_with_per_eta.push_back(spec_name);
            }
        }

        // Startup summary on rank 0: which species use only the global
        // eta, and which additionally have a per-species overlay.
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n[HybridPICModel] Resistivity configuration\n";
            amrex::Print() << "  global plasma_resistivity(rho,J,t) = "
                           << m_eta_expression << "\n";
            if (m_has_per_species_eta) {
                amrex::Print() << "  per-species overlays (eta_s_eff = "
                                  "eta_global + eta_per_species):\n";
                for (auto const & spec_name : mypc.GetSpeciesNames()) {
                    if (mypc.GetParticleContainerFromName(spec_name).getCharge() == 0._prt) {
                        continue;
                    }
                    auto it = m_per_species_resistivity_parser.find(spec_name);
                    if (it != m_per_species_resistivity_parser.end()) {
                        std::string expr;
                        pp_hybrid_init.query(
                            "plasma_resistivity_" + spec_name +
                             "(rho_s,rho,Te,J,J_s,B,t)", expr);
                        amrex::Print() << "    " << spec_name
                                       << "  : " << expr << "\n";
                    } else {
                        amrex::Print() << "    " << spec_name
                                       << "  : (global only)\n";
                    }
                }
            } else {
                amrex::Print() << "  per-species overlays              : none "
                                  "(all charged species use the global parser)\n";
            }
            amrex::Print() << "\n";
        }
    }

    // Consistency warnings between the particle-side resistive drag and the
    // electron-side Joule heating: each is allowed on its own, but the
    // resistive energy bookkeeping is only consistent when both are active
    // (the drag removes ion momentum/energy, the Joule source heats the
    // electrons by the corresponding dissipation).
    if (m_has_resistive_drag &&
        !(m_solve_electron_energy_equation && m_include_joule_heating)) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPICModel",
            "A hybrid_resistive_drag collision is registered, but the resistive "
            "(Joule) electron heating is not active (requires both "
            "hybrid_pic_model.solve_electron_energy_equation and "
            "hybrid_pic_model.include_joule_heating). The drag removes ion "
            "momentum/energy without the corresponding electron heating being "
            "tracked.",
            ablastr::warn_manager::WarnPriority::medium);
    }
    if (m_solve_electron_energy_equation && m_include_joule_heating &&
        !m_has_resistive_drag) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPICModel",
            "hybrid_pic_model.include_joule_heating is on, but no "
            "hybrid_resistive_drag collision is registered on the ions, so the "
            "ions feel no resistive friction back-reaction to the eta*J "
            "dissipation heating the electrons.",
            ablastr::warn_manager::WarnPriority::medium);
    }
    // The Te-threshold Joule redirect only acts inside the Joule source.
    if (m_joule_redirect_to_ions &&
        !(m_solve_electron_energy_equation && m_include_joule_heating)) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPICModel",
            "hybrid_pic_model.joule_redirect_Te_threshold is set, but the Joule "
            "heating source is not active (requires both "
            "hybrid_pic_model.solve_electron_energy_equation and "
            "hybrid_pic_model.include_joule_heating), so the redirect has no "
            "effect.",
            ablastr::warn_manager::WarnPriority::medium);
    }

    m_include_hyper_resistivity_term = (m_eta_h_expression != "0.0");
    m_hyper_resistivity_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_eta_h_expression, {"rho","B"}));
    m_eta_h = m_hyper_resistivity_parser->compile<2>();
    const std::set<std::string> hyper_resistivity_symbols = m_hyper_resistivity_parser->symbols();
    m_hyper_resistivity_has_B_dependence += hyper_resistivity_symbols.count("B");

    if (m_has_external_current) {
        m_J_external_parser[0] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_Jx_ext_grid_function,{"x","y","z","t"}));
        m_J_external_parser[1] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_Jy_ext_grid_function,{"x","y","z","t"}));
        m_J_external_parser[2] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_Jz_ext_grid_function,{"x","y","z","t"}));
        m_J_external[0] = m_J_external_parser[0]->compile<4>();
        m_J_external[1] = m_J_external_parser[1]->compile<4>();
        m_J_external[2] = m_J_external_parser[2]->compile<4>();

        // check if the external current parsers depend on time
        for (int i=0; i<3; i++) {
            const std::set<std::string> J_ext_symbols = m_J_external_parser[i]->symbols();
            m_external_current_has_time_dependence += J_ext_symbols.count("t");
        }
    }

    auto& warpx = WarpX::GetInstance();
    using ablastr::fields::Direction;

    // Get the grid staggering of the fields involved in calculating E
    amrex::IntVect Jx_stag = fields.get(FieldType::current_fp, Direction{0}, 0)->ixType().toIntVect();
    amrex::IntVect Jy_stag = fields.get(FieldType::current_fp, Direction{1}, 0)->ixType().toIntVect();
    amrex::IntVect Jz_stag = fields.get(FieldType::current_fp, Direction{2}, 0)->ixType().toIntVect();
    amrex::IntVect Bx_stag = fields.get(FieldType::Bfield_fp, Direction{0}, 0)->ixType().toIntVect();
    amrex::IntVect By_stag = fields.get(FieldType::Bfield_fp, Direction{1}, 0)->ixType().toIntVect();
    amrex::IntVect Bz_stag = fields.get(FieldType::Bfield_fp, Direction{2}, 0)->ixType().toIntVect();
    amrex::IntVect Ex_stag = fields.get(FieldType::Efield_fp, Direction{0}, 0)->ixType().toIntVect();
    amrex::IntVect Ey_stag = fields.get(FieldType::Efield_fp, Direction{1}, 0)->ixType().toIntVect();
    amrex::IntVect Ez_stag = fields.get(FieldType::Efield_fp, Direction{2}, 0)->ixType().toIntVect();

    // copy data to device
    for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        Jx_IndexType[idim]    = Jx_stag[idim];
        Jy_IndexType[idim]    = Jy_stag[idim];
        Jz_IndexType[idim]    = Jz_stag[idim];
        Bx_IndexType[idim]    = Bx_stag[idim];
        By_IndexType[idim]    = By_stag[idim];
        Bz_IndexType[idim]    = Bz_stag[idim];
        Ex_IndexType[idim]    = Ex_stag[idim];
        Ey_IndexType[idim]    = Ey_stag[idim];
        Ez_IndexType[idim]    = Ez_stag[idim];
    }

    // Below we set all the unused dimensions to have nodal values for J, B & E
    // since these values will be interpolated onto a nodal grid - if this is
    // not done the Interp function returns nonsense values.
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_1D_Z) || \
    defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    Jx_IndexType[2]    = 1;
    Jy_IndexType[2]    = 1;
    Jz_IndexType[2]    = 1;
    Bx_IndexType[2]    = 1;
    By_IndexType[2]    = 1;
    Bz_IndexType[2]    = 1;
    Ex_IndexType[2]    = 1;
    Ey_IndexType[2]    = 1;
    Ez_IndexType[2]    = 1;
#endif
#if defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    Jx_IndexType[1]    = 1;
    Jy_IndexType[1]    = 1;
    Jz_IndexType[1]    = 1;
    Bx_IndexType[1]    = 1;
    By_IndexType[1]    = 1;
    Bz_IndexType[1]    = 1;
    Ex_IndexType[1]    = 1;
    Ey_IndexType[1]    = 1;
    Ez_IndexType[1]    = 1;
#endif

    if (m_has_external_current) {
        // Initialize external current - note that this approach skips the check
        // if the current is time dependent which is what needs to be done to
        // write time independent fields on the first step.
        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            warpx.ComputeExternalFieldOnGridUsingParser(
                FieldType::hybrid_current_fp_external,
                m_J_external[0],
                m_J_external[1],
                m_J_external[2],
                lev, PatchType::fine,
                warpx.GetEBUpdateEFlag());
        }
    }

    if (m_add_external_fields) {
        m_external_vector_potential->InitData();
    }

    // Seed T_e with the uniform value parsed from <hybrid>.elec_temp (in
    // Joules after ReadParameters, so dividing by k_B gives Kelvin). The
    // iter-0 diagnostic dump -- which WarpX::InitData() flushes BEFORE the
    // first field-solve -- then sees a meaningful T_e rather than the
    // zero-initialized allocation. With the energy equation on, this is the
    // starting K_e value the QDSMC particles will read on the first step;
    // with it off, CalculateElectronPressure overwrites it each step.
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        amrex::MultiFab & Te_mf = *warpx.m_fields.get(
            FieldType::hybrid_electron_temperature_fp, lev);
        Te_mf.setVal(m_elec_temp / PhysConst::kb);
    }

    // Explicit fluid electron-pressure solver: startup checks that need the
    // fully parsed model.
    if (FluidElectronPressure()) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(warpx.evolve_scheme == EvolveScheme::Explicit,
            "hybrid_pic_model.electron_energy_solver = fluid is for the explicit evolve "
            "scheme (the theta-implicit hybrid scheme advances pe in its Newton loop)");
        // holmstrom_vacuum_region: the advanced pe is blended toward the floored adiabat
        // with the same vacuum weight Ohm's law uses (as the implicit in-loop advance does)
    }
    if (m_filter_push_fields) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::use_filter,
            "hybrid_pic_model.filter_push_fields requires warpx.use_filter = 1");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_add_external_fields,
            "hybrid_pic_model.filter_push_fields: external fields are not supported");
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(warpx.Geom(0).isPeriodic(d),
                "hybrid_pic_model.filter_push_fields requires a fully periodic domain "
                "(binomial-filter self-adjointness at walls is not handled)");
        }
    }

    // QDSMC: lazy-construct the fictitious-particle container and lay one
    // particle per cell (not needed by the fluid electron-pressure solver).
    if (m_solve_electron_energy_equation && m_electron_energy_solver == 0) {
        m_qdsmc_pc = std::make_unique<QdsmcParticleContainer>(&warpx);
        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            m_qdsmc_pc->InitParticles(lev);
        }
    }
}

void HybridPICModel::GetCurrentExternal ()
{
    if (!m_external_current_has_time_dependence) { return; }

    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        warpx.ComputeExternalFieldOnGridUsingParser(
            FieldType::hybrid_current_fp_external,
            m_J_external[0],
            m_J_external[1],
            m_J_external[2],
            lev, PatchType::fine,
            warpx.GetEBUpdateEFlag());
    }
}

void HybridPICModel::CalculatePlasmaCurrent (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E) const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        CalculatePlasmaCurrent(Bfield[lev], eb_update_E[lev], lev);
    }
}

void HybridPICModel::CalculatePlasmaCurrent (
    ablastr::fields::VectorField const& Bfield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
    const int lev) const
{
    ABLASTR_PROFILE("HybridPICModel::CalculatePlasmaCurrent()");

    auto& warpx = WarpX::GetInstance();
    ablastr::fields::VectorField current_fp_plasma = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    warpx.get_pointer_fdtd_solver_fp(lev)->CalculateCurrentAmpere(
        current_fp_plasma, Bfield, eb_update_E, lev
    );

    if (m_has_external_current) {
        // Subtract external current from "Ampere" current calculated above. Note
        // we need to include 1 ghost cell since later we will interpolate the
        // plasma current to a nodal grid.
        ablastr::fields::VectorField current_fp_external = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_external, lev);
        for (int i=0; i<3; i++) {
            current_fp_plasma[i]->minus(*current_fp_external[i], 0, 1, 1);
        }
    }
}

void HybridPICModel::HybridPICSolveE (
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    const bool solve_for_Faraday,
    const bool keep_grad_pe) const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        HybridPICSolveE(
            Efield[lev], Jfield[lev], Bfield[lev], *rhofield[lev],
            eb_update_E[lev], lev, solve_for_Faraday, keep_grad_pe
        );
    }
    // Allow execution of Python callback after E-field push
    ExecutePythonCallback("afterEpush");
}

void HybridPICModel::HybridPICSolveE (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
    const int lev, const bool solve_for_Faraday,
    const bool keep_grad_pe) const
{
    ABLASTR_PROFILE("WarpX::HybridPICSolveE()");

    HybridPICSolveE(
        Efield, Jfield, Bfield, rhofield, eb_update_E, lev,
        PatchType::fine, solve_for_Faraday, keep_grad_pe
    );
    if (lev > 0)
    {
        amrex::Abort(Utils::TextMsg::Err(
        "HybridPICSolveE: Only one level implemented for hybrid-PIC solver."));
    }
}

void HybridPICModel::HybridPICSolveE (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
    const int lev, PatchType patch_type,
    const bool solve_for_Faraday,
    const bool keep_grad_pe) const
{
    auto& warpx = WarpX::GetInstance();

    ablastr::fields::VectorField current_fp_plasma = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    auto* const electron_pressure_fp = warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);

    // Solve E field in regular cells
    warpx.get_pointer_fdtd_solver_fp(lev)->HybridPICSolveE(
        Efield, current_fp_plasma, Jfield, Bfield, rhofield,
        *electron_pressure_fp, eb_update_E, lev, this, solve_for_Faraday, keep_grad_pe
    );
    amrex::Real const time = warpx.gett_old(0) + warpx.getdt(0);
    warpx.ApplyEfieldBoundary(lev, patch_type, time);
    if (m_pec_normal_E_from_interior) {
        ApplyPECNormalEFromInterior(Efield, lev);
    }
}

void HybridPICModel::ApplyPECNormalEFromInterior (
    ablastr::fields::VectorField const& Efield, const int lev) const
{
    // Collocated grids only: there the component normal to a perfectly conducting wall
    // sits on the wall node, and Ohm's law at that node closes through one-sided
    // derivatives of the tangential magnetic field. Replace it by the value at the first
    // interior node (zero normal gradient). On the staggered grid the normal component
    // is cell-centered along its own direction and never lies on the wall.
    auto& warpx = WarpX::GetInstance();
    const amrex::Geometry& geom = warpx.Geom(lev);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (geom.isPeriodic(idim)) { continue; }
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        const int icomp = (idim == 0) ? 0 : 2;
#elif defined(WARPX_DIM_1D_Z)
        const int icomp = 2;
#else
        const int icomp = idim;
#endif
        amrex::MultiFab& E = *Efield[icomp];
        if (!E.ixType().nodeCentered(idim)) { continue; }
        bool any_pec = false;
        for (int side = 0; side < 2; ++side) {
            const FieldBoundaryType bt = (side == 0) ? WarpX::field_boundary_lo[idim]
                                                     : WarpX::field_boundary_hi[idim];
            any_pec = any_pec || (bt == FieldBoundaryType::PEC
                               || bt == FieldBoundaryType::PEC_Insulator);
        }
        if (!any_pec) { continue; }
        // The interior neighbour may belong to another box: make the guard cells current
        E.FillBoundary(geom.periodicity());
        const amrex::Box nodal_domain = amrex::convert(geom.Domain(), E.ixType());
        for (int side = 0; side < 2; ++side) {
            const FieldBoundaryType bt = (side == 0) ? WarpX::field_boundary_lo[idim]
                                                     : WarpX::field_boundary_hi[idim];
            if (bt != FieldBoundaryType::PEC && bt != FieldBoundaryType::PEC_Insulator) {
                continue;
            }
            const int iwall = (side == 0) ? nodal_domain.smallEnd(idim)
                                          : nodal_domain.bigEnd(idim);
            const int shift = (side == 0) ? 1 : -1;
            const int ncomp = E.nComp();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(E, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                amrex::Box bx = mfi.tilebox();
                if (bx.smallEnd(idim) > iwall || bx.bigEnd(idim) < iwall) { continue; }
                bx.setSmall(idim, iwall);
                bx.setBig(idim, iwall);
                amrex::Array4<amrex::Real> const& arr = E.array(mfi);
                amrex::ParallelFor(bx, ncomp,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                    amrex::IntVect iv(AMREX_D_DECL(i, j, k));
                    amrex::IntVect src = iv;
                    src[idim] += shift;
                    arr(iv, n) = arr(src, n);
                });
            }
        }
    }
}

void HybridPICModel::CalculateElectronPressure() const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        CalculateElectronPressure(lev);
    }
}

void HybridPICModel::CalculateElectronPressure(const int lev) const
{
    ABLASTR_PROFILE("WarpX::CalculateElectronPressure()");

    auto& warpx = WarpX::GetInstance();
    ablastr::fields::ScalarField electron_pressure_fp = warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    ablastr::fields::ScalarField rho_fp = warpx.m_fields.get(FieldType::rho_fp, lev);

    // Calculate the electron pressure using rho^{n+1}.
    FillElectronPressureMF(
        *electron_pressure_fp,
        *rho_fp
    );
    warpx.ApplyElectronPressureBoundary(lev, PatchType::fine);

    // Mirror the closure's implied electron temperature into
    // hybrid_electron_temperature_fp so the "Te" diagnostic is meaningful.
    FillTeFromPe(lev);

    ablastr::utils::communication::FillBoundary(
        *electron_pressure_fp,
        WarpX::do_single_precision_comms,
        warpx.Geom(lev).periodicity(),
        true);
}

void HybridPICModel::FillTeFromPe (const int lev) const
{
    // T_e = P_e / (n_e k_B) from the current electron pressure and density.
    auto & warpx = WarpX::GetInstance();
    amrex::MultiFab       & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & Pe  = *warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp, lev);
    auto const rho_floor = PhysConst::q_e * m_n_floor;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real>       const & Te_arr  = Te.array(mfi);
        amrex::Array4<amrex::Real const> const & Pe_arr  = Pe.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);
        amrex::Box const & tbox = mfi.tilebox();
        amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real const rho_val = std::max(rho_arr(i,j,k), rho_floor);
            amrex::Real const ne      = rho_val / PhysConst::q_e;
            Te_arr(i,j,k) = Pe_arr(i,j,k) / (ne * PhysConst::kb);
        });
    }
}

void HybridPICModel::CalculateElectronFluidVelocity () const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        CalculateElectronFluidVelocity(lev);
    }
}

void HybridPICModel::CalculateElectronFluidVelocity (const int lev) const
{
    ABLASTR_PROFILE("WarpX::CalculateElectronFluidVelocity()");
    using namespace ablastr::coarsen::sample;

    auto & warpx = WarpX::GetInstance();
    ablastr::fields::VectorField Ve = warpx.m_fields.get_alldirs("Ve_fp", lev);
    ablastr::fields::VectorField Ji = warpx.m_fields.get_alldirs(FieldType::current_fp, lev);
    ablastr::fields::VectorField J  = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    amrex::MultiFab const & rho_field = *warpx.m_fields.get(FieldType::rho_fp, lev);

    auto const Jx_stag = Jx_IndexType;
    auto const Jy_stag = Jy_IndexType;
    auto const Jz_stag = Jz_IndexType;
    amrex::GpuArray<int, 3> const nodal = {1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};
    auto const rho_floor = m_n_floor * PhysConst::q_e;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Ve[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Vex = Ve[0]->array(mfi);
        Array4<Real> const& Vey = Ve[1]->array(mfi);
        Array4<Real> const& Vez = Ve[2]->array(mfi);
        Array4<Real const> const& Jix = Ji[0]->const_array(mfi);
        Array4<Real const> const& Jiy = Ji[1]->const_array(mfi);
        Array4<Real const> const& Jiz = Ji[2]->const_array(mfi);
        Array4<Real const> const& Jx  = J[0]->const_array(mfi);
        Array4<Real const> const& Jy  = J[1]->const_array(mfi);
        Array4<Real const> const& Jz  = J[2]->const_array(mfi);
        Array4<Real const> const& rho = rho_field.const_array(mfi);

        // Compute Ve in the first ghost layer as well: Ji and rho carry full
        // valid ghost regions here and J_plasma is computed with one valid
        // ghost layer (which sets the limit), so no FillBoundary is needed
        // for the order-1 gather in the drag operator or the default
        // single-pass binomial filter below.
        Box const& tx = mfi.tilebox(Ve[0]->ixType().toIntVect(), IntVect(1));
        Box const& ty = mfi.tilebox(Ve[1]->ixType().toIntVect(), IntVect(1));
        Box const& tz = mfi.tilebox(Ve[2]->ixType().toIntVect(), IntVect(1));

        amrex::ParallelFor(tx, ty, tz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                Real const rho_val = std::max(Interp(rho, nodal, Jx_stag, coarsen, i, j, k, 0), rho_floor);
                Vex(i, j, k) = (Jix(i, j, k) - Jx(i, j, k)) / rho_val;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                Real const rho_val = std::max(Interp(rho, nodal, Jy_stag, coarsen, i, j, k, 0), rho_floor);
                Vey(i, j, k) = (Jiy(i, j, k) - Jy(i, j, k)) / rho_val;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                Real const rho_val = std::max(Interp(rho, nodal, Jz_stag, coarsen, i, j, k, 0), rho_floor);
                Vez(i, j, k) = (Jiz(i, j, k) - Jz(i, j, k)) / rho_val;
            }
        );
    }

    // Apply the same binomial filter used on J (suppresses grid-scale noise
    // that would otherwise be injected into particles by the gather inside
    // the drag operator). The filter only writes valid cells, so refresh the
    // ghosts afterwards; a multi-pass filter also reads ghosts beyond the
    // single computed layer and needs them communicated first.
    if (WarpX::use_filter) {
        if (WarpX::filter_npass_each_dir.max() > 1) {
            for (int idim = 0; idim < 3; ++idim) {
                ablastr::utils::communication::FillBoundary(
                    *Ve[idim], WarpX::do_single_precision_comms,
                    warpx.Geom(lev).periodicity(), true);
            }
        }
        warpx.ApplyFilterMF(
            warpx.m_fields.get_mr_levels_alldirs("Ve_fp", warpx.finestLevel()), lev);
        for (int idim = 0; idim < 3; ++idim) {
            ablastr::utils::communication::FillBoundary(
                *Ve[idim], WarpX::do_single_precision_comms,
                warpx.Geom(lev).periodicity(), true);
        }
    }
}

void HybridPICModel::CalculateIonFluidVelocity () const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        CalculateIonFluidVelocity(lev);
    }
}

void HybridPICModel::CalculateIonFluidVelocity (const int lev) const
{
    ABLASTR_PROFILE("WarpX::CalculateIonFluidVelocity()");
    using namespace ablastr::coarsen::sample;

    auto & warpx = WarpX::GetInstance();
    auto const & mypc = warpx.GetPartContainer();

    auto const Jx_stag = Jx_IndexType;
    auto const Jy_stag = Jy_IndexType;
    auto const Jz_stag = Jz_IndexType;
    amrex::GpuArray<int, 3> const nodal = {1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};
    auto const rho_floor = m_n_floor * PhysConst::q_e;

    for (auto const & spec : mypc.GetSpeciesNames()) {
        if (mypc.GetParticleContainerFromName(spec).getCharge() == 0._prt) { continue; }
        ablastr::fields::VectorField Vs = warpx.m_fields.get_alldirs("Vs_fp_" + spec, lev);
        ablastr::fields::VectorField Js = warpx.m_fields.get_alldirs("current_fp_" + spec, lev);
        amrex::MultiFab const & rho_s = *warpx.m_fields.get("rho_fp_" + spec, lev);

        // Js and rho_s carry full valid ghost regions here (their deposits
        // SumBoundary with dst_ng = nGrowVect()), so compute Vs directly in
        // the ghost cells instead of communicating afterwards. The nodal ->
        // staggered interpolation of rho_s reads one neighbor beyond the
        // written cell, hence the -1 on its ghost extent.
        amrex::IntVect ng_v = Vs[0]->nGrowVect();
        ng_v.min(Js[0]->nGrowVect());
        ng_v.min(rho_s.nGrowVect() - amrex::IntVect(1));

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*Vs[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
            Array4<Real> const& Vsx = Vs[0]->array(mfi);
            Array4<Real> const& Vsy = Vs[1]->array(mfi);
            Array4<Real> const& Vsz = Vs[2]->array(mfi);
            Array4<Real const> const& Jsx = Js[0]->const_array(mfi);
            Array4<Real const> const& Jsy = Js[1]->const_array(mfi);
            Array4<Real const> const& Jsz = Js[2]->const_array(mfi);
            Array4<Real const> const& rho = rho_s.const_array(mfi);

            Box const& tx = mfi.tilebox(Vs[0]->ixType().toIntVect(), ng_v);
            Box const& ty = mfi.tilebox(Vs[1]->ixType().toIntVect(), ng_v);
            Box const& tz = mfi.tilebox(Vs[2]->ixType().toIntVect(), ng_v);

            amrex::ParallelFor(tx, ty, tz,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    Real const rho_val = std::max(Interp(rho, nodal, Jx_stag, coarsen, i, j, k, 0), rho_floor);
                    Vsx(i, j, k) = Jsx(i, j, k) / rho_val;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    Real const rho_val = std::max(Interp(rho, nodal, Jy_stag, coarsen, i, j, k, 0), rho_floor);
                    Vsy(i, j, k) = Jsy(i, j, k) / rho_val;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    Real const rho_val = std::max(Interp(rho, nodal, Jz_stag, coarsen, i, j, k, 0), rho_floor);
                    Vsz(i, j, k) = Jsz(i, j, k) / rho_val;
                }
            );
        }

        // Same J-style binomial filter as in CalculateElectronFluidVelocity;
        // the ghost cells were computed above, so only the post-filter
        // refresh is needed (the filter writes valid cells only).
        if (WarpX::use_filter) {
            warpx.ApplyFilterMF(
                warpx.m_fields.get_mr_levels_alldirs("Vs_fp_" + spec, warpx.finestLevel()),
                lev);
            for (int idim = 0; idim < 3; ++idim) {
                ablastr::utils::communication::FillBoundary(
                    *Vs[idim], WarpX::do_single_precision_comms,
                    warpx.Geom(lev).periodicity(), true);
            }
        }
    }
}

void HybridPICModel::ComputeResistiveOverlay (
    int const lev,
    amrex::MultiFab & overlay_x,
    amrex::MultiFab & overlay_y,
    amrex::MultiFab & overlay_z) const
{
    ABLASTR_PROFILE("HybridPICModel::ComputeResistiveOverlay()");

    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Result is the per-species friction contribution added to Ohm's-law E:
    //   overlay_d(i,j,k) = Sigma_s [ eta_s_per * rho_s * rho / rho_sum * (V_s_d - V_e_d) ]
    // summed over charged species with a registered per-species parser.
    // eta_s_per is evaluated at (rho_s, rho, T_e [K], |J|, |J_s|, |B|, t) per cell
    // at the d-component staggering. We always zero the output first; if
    // no per-species parsers are registered we return immediately (single-eta
    // path is then exactly recovered when the caller adds a field of zeros).

    overlay_x.setVal(0.0_rt);
    overlay_y.setVal(0.0_rt);
    overlay_z.setVal(0.0_rt);

    if (!m_has_per_species_eta) { return; }

    auto & warpx = WarpX::GetInstance();
    auto const & mypc = warpx.GetPartContainer();
    auto const species_names = mypc.GetSpeciesNames();
    auto const t_new = warpx.gett_new(lev);

    amrex::MultiFab const & rho_total =
        *warpx.m_fields.get(FieldType::rho_fp, lev);
    amrex::MultiFab const & Te_K =
        *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    ablastr::fields::VectorField J_plasma =
        warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    ablastr::fields::VectorField B_fp =
        warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev);
    ablastr::fields::VectorField Ve_fp =
        warpx.m_fields.get_alldirs("Ve_fp", lev);

    auto const rho_floor = PhysConst::q_e * m_n_floor;

    // rho_sum = Sigma_t rho_fp_t over all charged species: the unscaled
    // "raw deposit" sum, paired with rho_s_raw in the same form so the
    // 2pi*r RZ factor cancels in the species-fraction ratio. Filled once
    // per step by HybridPICDepositRhoAndJ.
    amrex::MultiFab const & rho_sum =
        *warpx.m_fields.get("hybrid_rho_species_sum_fp", lev);

    amrex::GpuArray<int, 3> const & Jx_stag = Jx_IndexType;
    amrex::GpuArray<int, 3> const & Jy_stag = Jy_IndexType;
    amrex::GpuArray<int, 3> const & Jz_stag = Jz_IndexType;
    amrex::GpuArray<int, 3> const & Bx_stag = Bx_IndexType;
    amrex::GpuArray<int, 3> const & By_stag = By_IndexType;
    amrex::GpuArray<int, 3> const & Bz_stag = Bz_IndexType;
    amrex::GpuArray<int, 3> const nodal   = {1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};

    // Single kernel pass per (species, direction) pair. Each pass reads
    // the species's parser executor by value (amrex::ParserExecutor is
    // device-callable), interpolates the global and per-species scalar
    // fields to the d-component staggering, evaluates the parser, and
    // accumulates the contribution into overlay_<d>. The d-component
    // values of V_s and V_e are read directly because Vs_fp and Ve_fp are
    // both at the d staggering by construction; only the cross components
    // of J, J_s and B need interp to compute magnitudes.
    for (auto const & spec_name : species_names) {
        auto & pc = mypc.GetParticleContainerFromName(spec_name);
        if (pc.getCharge() == 0._prt) { continue; }

        auto eta_s_per_it = m_eta_per_species.find(spec_name);
        if (eta_s_per_it == m_eta_per_species.end()) { continue; }
        auto const eta_s_per = eta_s_per_it->second;

        amrex::MultiFab const & rho_s_mf =
            *warpx.m_fields.get("rho_fp_" + spec_name, lev);
        ablastr::fields::VectorField Vs_fp =
            warpx.m_fields.get_alldirs("Vs_fp_" + spec_name, lev);
        ablastr::fields::VectorField Js_fp =
            warpx.m_fields.get_alldirs("current_fp_" + spec_name, lev);

        // Lambda: accumulate this species's overlay contribution into a
        // single output multifab at the given d staggering.
        auto accumulate_one_direction = [&] (
            amrex::MultiFab       & out,
            amrex::GpuArray<int,3>  const d_stag,
            int                     d_idx)
        {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(out, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Array4<amrex::Real>       const & out_arr     = out.array(mfi);
                amrex::Array4<amrex::Real const> const & rho_arr     = rho_total.const_array(mfi);
                amrex::Array4<amrex::Real const> const & rhos_arr    = rho_s_mf.const_array(mfi);
                amrex::Array4<amrex::Real const> const & rhosum_arr  = rho_sum.const_array(mfi);
                amrex::Array4<amrex::Real const> const & Te_arr      = Te_K.const_array(mfi);
                amrex::Array4<amrex::Real const> const & Jpx_arr     = J_plasma[0]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Jpy_arr     = J_plasma[1]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Jpz_arr     = J_plasma[2]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Jsx_arr     = Js_fp[0]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Jsy_arr     = Js_fp[1]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Jsz_arr     = Js_fp[2]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Bx_arr      = B_fp[0]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & By_arr      = B_fp[1]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Bz_arr      = B_fp[2]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Vsd_arr     = Vs_fp[d_idx]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Ved_arr     = Ve_fp[d_idx]->const_array(mfi);

                amrex::Box const & tbox = mfi.tilebox(out.ixType().toIntVect());
                amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    using ablastr::coarsen::sample::Interp;
                    amrex::Real const rho_val = Interp(rho_arr, nodal, d_stag, coarsen, i, j, k, 0);
                    if (rho_val <= rho_floor) { return; }

                    amrex::Real const rhos_val   = Interp(rhos_arr,   nodal, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const rhosum_val = std::max(
                        Interp(rhosum_arr, nodal, d_stag, coarsen, i, j, k, 0), rho_floor);
                    amrex::Real const Te_val     = Interp(Te_arr,     nodal, d_stag, coarsen, i, j, k, 0);

                    // |J|, |J_s|, |B| at d staggering. Interp every Yee
                    // component (the d-component interp from its own
                    // staggering to itself is identity, so this is safe).
                    amrex::Real const jpx = Interp(Jpx_arr, Jx_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const jpy = Interp(Jpy_arr, Jy_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const jpz = Interp(Jpz_arr, Jz_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const Jmag = std::sqrt(jpx*jpx + jpy*jpy + jpz*jpz);

                    amrex::Real const jsx = Interp(Jsx_arr, Jx_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const jsy = Interp(Jsy_arr, Jy_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const jsz = Interp(Jsz_arr, Jz_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const Jsmag = std::sqrt(jsx*jsx + jsy*jsy + jsz*jsz);

                    amrex::Real const bx = Interp(Bx_arr, Bx_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const by = Interp(By_arr, By_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const bz = Interp(Bz_arr, Bz_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const Bmag = std::sqrt(bx*bx + by*by + bz*bz);

                    amrex::Real const dv_d  = Vsd_arr(i,j,k) - Ved_arr(i,j,k);
                    amrex::Real const eta_s = eta_s_per(rhos_val, rho_val, Te_val,
                                                        Jmag, Jsmag, Bmag, t_new);
                    out_arr(i,j,k) += eta_s * rhos_val * rho_val / rhosum_val * dv_d;
                });
            }
        };

        accumulate_one_direction(overlay_x, Jx_stag, 0);
        accumulate_one_direction(overlay_y, Jy_stag, 1);
        accumulate_one_direction(overlay_z, Jz_stag, 2);
    }
    // No ghost exchange: the E-solve kernels only read the overlay at valid
    // cells (the Yee E updates run on ungrown tileboxes).
}


void HybridPICModel::FillElectronPressureMF (
    amrex::MultiFab& Pe_field,
    amrex::MultiFab const& rho_field
) const
{
    const auto n0_ref = m_n0_ref;
    const auto elec_temp = m_elec_temp;
    const auto gamma = m_gamma;

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(Pe_field, TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {
        // Extract field data for this grid/tile
        Array4<Real const> const& rho = rho_field.const_array(mfi);
        Array4<Real> const& Pe = Pe_field.array(mfi);

        // Extract tileboxes for which to loop
        const Box& tilebox  = mfi.tilebox();

        ParallelFor(tilebox, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            Pe(i, j, k) = ElectronPressure::get_pressure(
                n0_ref, elec_temp, gamma, rho(i, j, k)
            );
        });
    }
}

// =============================================================================
// QDSMC electron-energy-equation orchestration
// =============================================================================
//
// All four methods below are NO-OPs when m_solve_electron_energy_equation is false; they are
// invoked from HybridPICEvolveFields only when QDSMC is enabled. They operate
// on the level-`lev` MultiFabs of WarpX's MultiFabRegister and use the same
// Yee->nodal interpolation (`ablastr::coarsen::sample::Interp`) as the rest
// of the hybrid solver.

void HybridPICModel::QDSMCInitializeUe (int const lev) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCInitializeUe()");

    using ablastr::fields::Direction;

    auto & warpx = WarpX::GetInstance();
    amrex::Geometry const & geom = warpx.Geom(lev);
    amrex::Periodicity const & period = geom.periodicity();

    // V_e and rho live at the nodal grid; J_plasma and J_i are Yee-staggered.
    amrex::MultiFab       & Vex = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{0}, lev);
    amrex::MultiFab       & Vey = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{1}, lev);
    amrex::MultiFab       & Vez = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{2}, lev);

    amrex::MultiFab const & rho_temp = *warpx.m_fields.get(FieldType::hybrid_rho_fp_temp, lev);

    ablastr::fields::VectorField J_plasma =
        warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    ablastr::fields::VectorField J_i =
        warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_temp, lev);

    amrex::Real const rho_floor = PhysConst::q_e * m_n_floor;

    amrex::GpuArray<int, 3> const & Jx_stag = Jx_IndexType;
    amrex::GpuArray<int, 3> const & Jy_stag = Jy_IndexType;
    amrex::GpuArray<int, 3> const & Jz_stag = Jz_IndexType;
    amrex::GpuArray<int, 3> const nodal     = {1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen   = {1, 1, 1};

    Vex.setVal(0.0_rt);
    Vey.setVal(0.0_rt);
    Vez.setVal(0.0_rt);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Vex, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real const> const & rho_arr = rho_temp.const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jpx     = J_plasma[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jpy     = J_plasma[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jpz     = J_plasma[2]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jix     = J_i[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jiy     = J_i[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jiz     = J_i[2]->const_array(mfi);
        amrex::Array4<amrex::Real>       const & Vex_arr = Vex.array(mfi);
        amrex::Array4<amrex::Real>       const & Vey_arr = Vey.array(mfi);
        amrex::Array4<amrex::Real>       const & Vez_arr = Vez.array(mfi);

        amrex::Box const & tbox = mfi.tilebox();

        amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (rho_arr(i,j,k) <= rho_floor) { return; }

            amrex::Real const rho_val = rho_arr(i,j,k);

            auto const jx  = ablastr::coarsen::sample::Interp(Jpx, Jx_stag, nodal, coarsen, i, j, k, 0);
            auto const jy  = ablastr::coarsen::sample::Interp(Jpy, Jy_stag, nodal, coarsen, i, j, k, 0);
            auto const jz  = ablastr::coarsen::sample::Interp(Jpz, Jz_stag, nodal, coarsen, i, j, k, 0);
            auto const jix = ablastr::coarsen::sample::Interp(Jix, Jx_stag, nodal, coarsen, i, j, k, 0);
            auto const jiy = ablastr::coarsen::sample::Interp(Jiy, Jy_stag, nodal, coarsen, i, j, k, 0);
            auto const jiz = ablastr::coarsen::sample::Interp(Jiz, Jz_stag, nodal, coarsen, i, j, k, 0);

            // V_e = -(J_plasma - J_i) / (q_e * n_e) = -(J_plasma - J_i) / rho_val
            Vex_arr(i,j,k) = -(jx - jix) / rho_val;
            Vey_arr(i,j,k) = -(jy - jiy) / rho_val;
            Vez_arr(i,j,k) = -(jz - jiz) / rho_val;
        });
    }

    Vex.FillBoundary(Vex.nGrowVect(), period);
    Vey.FillBoundary(Vey.nGrowVect(), period);
    Vez.FillBoundary(Vez.nGrowVect(), period);
}


void HybridPICModel::QDSMCInitializeKe (int const lev) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCInitializeKe()");

    auto & warpx = WarpX::GetInstance();

    amrex::MultiFab       & Ke  = *warpx.m_fields.get(FieldType::hybrid_entropy_fp,                lev);
    amrex::MultiFab const & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp,   lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::hybrid_rho_fp_temp,               lev);

    Ke.setVal(0.0_rt);

    auto const gamma     = m_gamma;
    auto const rho_floor = PhysConst::q_e * m_n_floor;
    // Scale K_e to eV-equivalent (multiply T_e[K] by k_B/q_e) to keep it
    // numerically O(1) for common plasma parameters.
    auto const kb_over_qe = PhysConst::kb / PhysConst::q_e;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Ke, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real>       const & Ke_arr  = Ke.array(mfi);
        amrex::Array4<amrex::Real const> const & Te_arr  = Te.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);

        amrex::Box const tbox = amrex::convert(mfi.tilebox(), Ke.ixType().toIntVect());
        amrex::Box       box  = tbox;
        box.grow(Ke.nGrowVect());

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Floor the density instead of skipping low-density cells:
            // leaving K_e = 0 in the floored halo turns it into an absorbing
            // boundary that drains the plasma's electron thermal energy via
            // the remap diffusion (global exponential T_e collapse). With the
            // floor, the halo keeps whatever T_e it holds and is insulating.
            amrex::Real const ne =
                amrex::max(rho_arr(i,j,k), rho_floor) / PhysConst::q_e;
            Ke_arr(i,j,k) = Te_arr(i,j,k) * std::pow(ne, 1.0_rt - gamma) * kb_over_qe;
        });
    }
    // No ghost exchange: the kernel runs on the ghost-grown box and its
    // inputs (Te, rho at n) already have valid ghosts, so Ke's ghost cells
    // are consistent with the neighboring boxes' valid values.
}


void HybridPICModel::QDSMCUpdateTe (int const lev) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCUpdateTe()");

    auto & warpx = WarpX::GetInstance();
    amrex::Geometry const & geom = warpx.Geom(lev);

    // After the QDSMC scatter, weights_fp ~= n_e (density) and entropy_fp ~=
    // K_e * N_e (entropy weighted by count, summed). Recover T_e_new:
    //
    //   K_e_new = entropy_fp / (weights_fp * V_cell)
    //   T_e_new = K_e_new / (n_e_new^(1-gamma) * k_B / q_e)
    //
    // n_e_new comes from rho_fp (post-deposit, post-particle-push).

    auto const dx_arr = geom.CellSizeArray();
    amrex::Real cell_volume = 1.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { cell_volume *= dx_arr[d]; }

    amrex::MultiFab       & Te      = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & Ke      = *warpx.m_fields.get(FieldType::hybrid_entropy_fp,              lev);
    amrex::MultiFab const & weights = *warpx.m_fields.get(FieldType::hybrid_qdsmc_weights_fp,        lev);
    amrex::MultiFab const & rho     = *warpx.m_fields.get(FieldType::rho_fp,                         lev);

    // Note: T_e is NOT zeroed here. Cells that received no QDSMC weight or
    // are below the density floor keep their previous T_e -- zeroing them
    // would erase valid state (and seed K_e = 0 into neighbors on the next
    // step) whenever a cell momentarily receives no deposit.

    auto const gamma      = m_gamma;
    // Must match the floor QDSMCInitializeKe applies, so the K -> T_e
    // round-trip is exact for a marker that stayed home.
    auto const n_floor    = m_n_floor;
    auto const kb_over_qe = PhysConst::kb / PhysConst::q_e;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real>       const & Te_arr      = Te.array(mfi);
        amrex::Array4<amrex::Real const> const & Ke_arr      = Ke.const_array(mfi);
        amrex::Array4<amrex::Real const> const & weights_arr = weights.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr     = rho.const_array(mfi);

        amrex::Box const tbox = amrex::convert(mfi.tilebox(), Te.ixType().toIntVect());
        amrex::Box       box  = tbox;
        box.grow(Te.nGrowVect());

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real const w = weights_arr(i,j,k) * cell_volume;
            if (w <= 0.0_rt) { return; }
            // Floored density, mirroring QDSMCInitializeKe: below-floor
            // cells are updated too (insulating halo), and the K <-> T_e
            // conversion uses the same n_e^(gamma-1) factor on both sides of
            // the step, so a cell whose marker did not move keeps its T_e
            // exactly.
            amrex::Real const ne =
                amrex::max(rho_arr(i,j,k) / PhysConst::q_e, n_floor);
            Te_arr(i,j,k) = Ke_arr(i,j,k)
                          / std::pow(ne, 1.0_rt - gamma)
                          / w
                          / kb_over_qe;
        });
    }
    // No ghost exchange: the kernel runs on the ghost-grown box and its
    // inputs already have valid ghosts (the QDSMC deposits SumBoundary with
    // dst_ng = nGrowVect(), and rho_fp was FillBoundary'd after deposition).
}


void HybridPICModel::QDSMCAddJouleHeating (int const lev, amrex::Real const dt,
                                           amrex::MultiFab * const redirect_E) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCAddJouleHeating()");

    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Per-cell resistive electron-heating source (Phys. Plasmas 31, 012902 (2024), Eq. 12).
    // With the e-i relative drift V_s - V_e = J_plasma/(e n_e) and the
    // eta-derived rate nu_{s,e} = Z_s e^2 eta n_e / m_s, the source is
    //
    //   S_e = e^2 eta n_e Sum_s Z_s n_s |J_plasma/(e n_e)|^2
    //
    // which collapses to eta J^2 for a single species. Computed on the grid from
    // rho_fp, rho_fp_s and the plasma current -- no per-particle scatter.

    auto & warpx = WarpX::GetInstance();
    amrex::Periodicity const & period = warpx.Geom(lev).periodicity();

    amrex::MultiFab       & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp,                         lev);
    ablastr::fields::VectorField J_plasma =
        warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    // B field on Yee staggering; needed (as |B|) only when a per-species
    // resistivity parser is registered for a species visited in this loop.
    // The Yee->nodal interp is cheap; we always compute it inside the kernel
    // because the branch fast-paths to skip the parser call when not needed.
    ablastr::fields::VectorField B_fp =
        warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev);

    auto const gamma_minus_1 = m_gamma - 1.0_rt;
    auto const rho_floor     = PhysConst::q_e * m_n_floor;
    auto const eta           = m_eta;
    auto const t_new         = warpx.gett_new(0);

    amrex::GpuArray<int, 3> const & Jx_stag = Jx_IndexType;
    amrex::GpuArray<int, 3> const & Jy_stag = Jy_IndexType;
    amrex::GpuArray<int, 3> const & Jz_stag = Jz_IndexType;
    amrex::GpuArray<int, 3> const & Bx_stag = Bx_IndexType;
    amrex::GpuArray<int, 3> const & By_stag = By_IndexType;
    amrex::GpuArray<int, 3> const & Bz_stag = Bz_IndexType;
    amrex::GpuArray<int, 3> const nodal     = {1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen   = {1, 1, 1};

    // Te-threshold Joule redirection: in cells with Te >= threshold the Joule
    // heat is written into redirect_E (per charged ion species, the m_i-independent
    // energy E_s [J]) for QDSMCApplyIonHeating to deposit on the ions, rather than
    // added to T_e.
    bool const do_redirect = (redirect_E != nullptr);
    auto const K_per_eV    = PhysConst::q_e / PhysConst::kb;        // T[eV]*this = T[K]
    amrex::Real const Te_thresh_K = m_joule_redirect_Te_eV * K_per_eV;
    // smooth-gate width in K (0 = hard step), see m_joule_redirect_Te_width
    amrex::Real const Te_w_K = m_joule_redirect_Te_width * Te_thresh_K;

    auto & mypc = warpx.GetPartContainer();

    // Loop over every charged ion species and accumulate its per-cell
    // contribution to S_e into T_e directly. Each species contributes
    //   dT_e_s = dt (gamma-1) * Z_s e^2 eta n_s |V_s - V_e|^2 / k_B
    // (the n_e factor in nu_{s,e} cancels the 1/n_e from the T_e update).
    // V_s is computed inline from THIS STEP's current_fp_s / rho_fp_s;
    // these are both deposited WITHOUT RZ volume scaling (per
    // HybridPICDepositRhoAndJ -- apply_boundary_and_scale_volume=false),
    // so their ratio V_s = J_s/rho_s gives correct m/s with the 2pi*r
    // factors cancelling.
    //
    // n_s in the heating coefficient, however, must be a TRUE density
    // (1/m^3). Reading rho_fp_s/q_e directly gives (2pi*r) x n_s_true
    // in RZ. We recover the correct n_s from the species charge fraction
    //
    //   f_s = rho_fp_s / Sigma_t rho_fp_t   =   Z_s n_s / n_e   (unitless,
    //                                            2pi*r factor cancels)
    //   n_s = f_s * n_e / Z_s
    //
    // where n_e comes from the volume-scaled total rho_fp. Works in
    // any dimensionality (in Cartesian the 2pi*r is just 1).
    auto const species_names = mypc.GetSpeciesNames();

    // Sigma_t rho_fp_t (unscaled per-species charge densities), used for the
    // species fraction f_s = rho_fp_s / rhos_sum per cell inside the species
    // loop. Filled once per step by HybridPICDepositRhoAndJ.
    amrex::MultiFab const & rhos_sum =
        *warpx.m_fields.get("hybrid_rho_species_sum_fp", lev);

    // Charged-species component index for redirect_E (matches QDSMCApplyIonHeating).
    int ion_comp = -1;
    for (auto const & spec_name : species_names) {
        auto & pc = mypc.GetParticleContainerFromName(spec_name);
        if (pc.getCharge() == 0._prt) { continue; }
        ++ion_comp;

        amrex::Real const Z_s = pc.getCharge() / PhysConst::q_e;

        ablastr::fields::VectorField Js =
            warpx.m_fields.get_alldirs("current_fp_" + spec_name, lev);
        amrex::MultiFab const & rho_s =
            *warpx.m_fields.get("rho_fp_" + spec_name, lev);

        // Per-species resistivity overlay parser, if registered. The
        // captured executor is only called from inside the kernel when
        // has_eta_per is true; default-constructed ParserExecutor is
        // never invoked.
        auto const eta_per_it     = m_eta_per_species.find(spec_name);
        bool const has_eta_per    = (eta_per_it != m_eta_per_species.end());
        amrex::ParserExecutor<7> eta_s_per{};
        if (has_eta_per) { eta_s_per = eta_per_it->second; }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Array4<amrex::Real>       const & Te_arr     = Te.array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr    = rho.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rhos_arr   = rho_s.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rhosum_arr = rhos_sum.const_array(mfi);
            amrex::Array4<amrex::Real const> const & Jpx        = J_plasma[0]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Jpy        = J_plasma[1]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Jpz        = J_plasma[2]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Jsx        = Js[0]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Jsy        = Js[1]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Jsz        = Js[2]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Bx_arr     = B_fp[0]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & By_arr     = B_fp[1]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Bz_arr     = B_fp[2]->const_array(mfi);

            // Redirect output (default Array4 when redirect off -> never indexed
            // because do_redirect gates the write).
            amrex::Array4<amrex::Real> redirect_arr;
            if (do_redirect) { redirect_arr = redirect_E->array(mfi); }

            amrex::Box const & tbox = mfi.tilebox();
            amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real const rho_val = rho_arr(i,j,k);
                if (rho_val <= rho_floor) { return; }
                // n_e (m^-3) from the volume-scaled total rho_fp.
                amrex::Real const ne = rho_val / PhysConst::q_e;
                // Species charge fraction f_s = rho_fp_s / Sigma_t rho_fp_t
                // = Z_s n_s / n_e (unitless; RZ 2pi*r factor cancels). Then
                // the true per-species number density:
                //   n_s = f_s * n_e / Z_s
                amrex::Real const rhos_val_raw  = rhos_arr(i,j,k);
                amrex::Real const rhos_sum_val  = std::max(rhosum_arr(i,j,k), rho_floor);
                amrex::Real const f_s           = rhos_val_raw / rhos_sum_val;
                amrex::Real const ns            = f_s * ne / Z_s;

                // |J| at the nodal grid (where Te lives). Used by the
                // global eta parser and forwarded to per-species parsers.
                auto const jx = ablastr::coarsen::sample::Interp(Jpx, Jx_stag, nodal, coarsen, i, j, k, 0);
                auto const jy = ablastr::coarsen::sample::Interp(Jpy, Jy_stag, nodal, coarsen, i, j, k, 0);
                auto const jz = ablastr::coarsen::sample::Interp(Jpz, Jz_stag, nodal, coarsen, i, j, k, 0);
                amrex::Real const Jmag = std::sqrt(jx*jx + jy*jy + jz*jz);

                // eta_global: same Ohm's-law parser the E-solve uses, evaluated
                // per cell. This makes the per-cell heat reduce to eta J^2
                // exactly in single species (when no per-species overlay).
                amrex::Real eta_s_eff = eta(rho_val, Jmag, t_new);

                // e-i relative drift = J_plasma/(e n_e), from the nodal plasma
                // current and n_e. Energy-consistent with the eta*J dissipation
                // in Ohm's law; reduces to eta*|J|^2 for a single species.
                amrex::Real const inv_ene = 1.0_rt / (PhysConst::q_e * ne);
                amrex::Real const dvx = jx * inv_ene;
                amrex::Real const dvy = jy * inv_ene;
                amrex::Real const dvz = jz * inv_ene;
                amrex::Real const dv2 = dvx*dvx + dvy*dvy + dvz*dvz;

                // eta_per_species overlay (Phys. Plasmas 31, 012902 (2024)). Only evaluated when
                // a per-species parser is registered for this species --
                // for the other species we just see eta_s_eff = eta_global.
                if (has_eta_per) {
                    amrex::Real const Te_K_val = Te_arr(i,j,k);
                    // J_s interpolated to nodal -- only needed for the parser's
                    // |J_s| argument.
                    auto const jsx_nodal = ablastr::coarsen::sample::Interp(Jsx, Jx_stag, nodal, coarsen, i, j, k, 0);
                    auto const jsy_nodal = ablastr::coarsen::sample::Interp(Jsy, Jy_stag, nodal, coarsen, i, j, k, 0);
                    auto const jsz_nodal = ablastr::coarsen::sample::Interp(Jsz, Jz_stag, nodal, coarsen, i, j, k, 0);
                    amrex::Real const Jsmag    = std::sqrt(
                        jsx_nodal*jsx_nodal + jsy_nodal*jsy_nodal + jsz_nodal*jsz_nodal);
                    auto const bx = ablastr::coarsen::sample::Interp(Bx_arr, Bx_stag, nodal, coarsen, i, j, k, 0);
                    auto const by = ablastr::coarsen::sample::Interp(By_arr, By_stag, nodal, coarsen, i, j, k, 0);
                    auto const bz = ablastr::coarsen::sample::Interp(Bz_arr, Bz_stag, nodal, coarsen, i, j, k, 0);
                    amrex::Real const Bmag = std::sqrt(bx*bx + by*by + bz*bz);
                    eta_s_eff += eta_s_per(rhos_val_raw, rho_val, Te_K_val,
                                           Jmag, Jsmag, Bmag, t_new);
                }

                // Per-species contribution to S_e at this cell.
                //   nu_{s,e} n_s m_s |V_s - V_e|^2 = Z_s e^2 eta_s_eff n_e n_s |dV|^2
                // Dividing by (n_e k_B) for the T_e update:
                //   dT_e_s = dt (gamma-1) * Z_s e^2 eta_s_eff n_s |dV|^2 / k_B
                amrex::Real const dTe_s = dt * gamma_minus_1
                               * Z_s * PhysConst::q_e * PhysConst::q_e
                               * eta_s_eff * ns * dv2 / PhysConst::kb;
                // Te-threshold redirection: below threshold heat electrons (the
                // usual Joule deposit); at/above it write this species'
                // m_i-independent redirected energy E_s = (2/3) n_e Z_s e^2 eta
                // |dV|^2 dt [J] into its component for the ion-heating step.
                // With joule_redirect_Te_width w > 0 the step becomes the smooth gate
                // sJ = 0.5 (1 - tanh((Te - Tc)/(w Tc))) of the implicit scheme: the
                // electrons receive sJ of the heat, the ions (1 - sJ) of E_s.
                if (do_redirect && Te_w_K > 0.0_rt) {
                    amrex::Real const sJ = 0.5_rt * (1.0_rt
                        - std::tanh((Te_arr(i,j,k) - Te_thresh_K) / Te_w_K));
                    redirect_arr(i,j,k,ion_comp) = (1.0_rt - sJ) * (2.0_rt/3.0_rt) * ne
                        * Z_s * PhysConst::q_e * PhysConst::q_e * eta_s_eff * dv2 * dt;
                    Te_arr(i,j,k) += sJ * dTe_s;
                } else if (do_redirect && Te_arr(i,j,k) >= Te_thresh_K) {
                    redirect_arr(i,j,k,ion_comp) = (2.0_rt/3.0_rt) * ne
                        * Z_s * PhysConst::q_e * PhysConst::q_e * eta_s_eff * dv2 * dt;
                } else {
                    Te_arr(i,j,k) += dTe_s;
                }
            });
        }
    }

    Te.FillBoundary(Te.nGrowVect(), period);
}


void HybridPICModel::QDSMCAddTemperatureRelaxation (int const lev, amrex::Real const dt,
    std::map<std::string, amrex::MultiFab*> const & Ti_dep_by_species) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCAddTemperatureRelaxation()");

    using warpx::fields::FieldType;

    // Electron-ion thermal-equilibration sink, summed over ion species s:
    //   Q_ei = Sigma_s 3 n_s k_B nu_ei (T_e - T_i_s),    dU_e/dt += -Q_ei.
    // With U_e = n_e k_B T_e/(gamma-1), the per-cell T_e obeys
    //   dT_e/dt = -(gamma-1) 3 Sigma_s (n_s/n_e) nu_ei (T_e - T_i_s),
    // where n_s/n_e = f_s/Z_s, f_s = rho_fp_s/Sigma_t rho_fp_t. T_e is stored in
    // Kelvin; T_i (deposited per species, cell-centered, in eV) is interpolated
    // to the nodal T_e grid and converted to K. This is the electron-side sink;
    // QDSMCApplyIonHeating deposits the matching ion heating so the pair
    // conserves energy.
    auto & warpx = WarpX::GetInstance();
    amrex::Periodicity const & period = warpx.Geom(lev).periodicity();

    amrex::MultiFab       & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp, lev);

    auto const gamma_minus_1 = m_gamma - 1.0_rt;
    auto const rho_floor     = PhysConst::q_e * m_n_floor;
    auto const nu_ei         = m_nu_ei;
    auto const t_new         = warpx.gett_new(0);
    auto const K_per_eV      = PhysConst::q_e / PhysConst::kb;   // T[eV] * this = T[K]
    // Floor on T_e in the nu_ei rate argument so pow(Te,-1.5) stays finite.
    amrex::Real const Te_floor_eV = 1.e-3_rt;

    amrex::GpuArray<int, 3> const nodal   = {1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};

    auto & mypc = warpx.GetPartContainer();
    auto const species_names = mypc.GetSpeciesNames();

    // Sigma_t rho_fp_t (unscaled per-species charge densities) -> species
    // fraction. Filled once per step by HybridPICDepositRhoAndJ.
    amrex::MultiFab const & rhos_sum =
        *warpx.m_fields.get("hybrid_rho_species_sum_fp", lev);

    // Cell-centered field box array (for staging the deposited T_i with a guard
    // cell so it can be interpolated to the nodal T_e grid).
    amrex::BoxArray const cc_ba = amrex::convert(Te.boxArray(), amrex::IntVect::TheCellVector());

    for (auto const & spec_name : species_names) {
        auto & pc = mypc.GetParticleContainerFromName(spec_name);
        if (pc.getCharge() == 0._prt) { continue; }
        amrex::Real const Z_s = pc.getCharge() / PhysConst::q_e;

        amrex::MultiFab const & rho_s = *warpx.m_fields.get("rho_fp_" + spec_name, lev);

        // Per-cell ion temperature [eV] (NGP velocity-variance deposit, done once
        // by the caller and shared via Ti_dep_by_species), moved onto the field's
        // cell-centered grid with one guard cell so the cc->nodal interpolation
        // has its neighbours at box edges.
        amrex::MultiFab const & Ti_dep = *Ti_dep_by_species.at(spec_name);
        amrex::MultiFab Ti_cc(cc_ba, Te.DistributionMap(), 1, 1);
        Ti_cc.setVal(0.0_rt);
        Ti_cc.ParallelCopy(Ti_dep, 0, 0, 1, amrex::IntVect::TheZeroVector(),
                           amrex::IntVect::TheZeroVector());
        Ti_cc.FillBoundary(warpx.Geom(lev).periodicity());
        // Ti_cc is cell-centered in the real dimensions; the unused (2D/1D)
        // dimensions are set NODAL so they match the nodal destination grid in
        // Interp (sf==sc there -> np=1, no out-of-bounds k=-1 read). Mirrors the
        // unused-dimension handling for J/B/E above.
        amrex::GpuArray<int, 3> cc_stag = {0, 0, 0};
        for (int d = AMREX_SPACEDIM; d < 3; ++d) { cc_stag[d] = 1; }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Array4<amrex::Real>       const & Te_arr     = Te.array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr    = rho.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rhos_arr   = rho_s.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rhosum_arr = rhos_sum.const_array(mfi);
            amrex::Array4<amrex::Real const> const & Ti_arr     = Ti_cc.const_array(mfi);

            amrex::Box const & tbox = mfi.tilebox();
            amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real const rho_val = rho_arr(i,j,k);
                if (rho_val <= rho_floor) { return; }
                amrex::Real const rhos_sum_val = std::max(rhosum_arr(i,j,k), rho_floor);
                amrex::Real const f_s = rhos_arr(i,j,k) / rhos_sum_val;   // = Z_s n_s/n_e

                amrex::Real const Ti_eV = ablastr::coarsen::sample::Interp(
                    Ti_arr, cc_stag, nodal, coarsen, i, j, k, 0);
                amrex::Real const Te_K  = Te_arr(i,j,k);
                amrex::Real const Te_eV = Te_K / K_per_eV;
                amrex::Real const Ti_K  = Ti_eV * K_per_eV;

                amrex::Real const nu = nu_ei(rho_val, amrex::max(Te_eV, Te_floor_eV), Ti_eV, t_new);
                // Exact exponential integration of dT_e/dt = -alpha nu (T_e - T_i),
                // with alpha = (gamma-1) 3 n_s/n_e and n_s/n_e = f_s/Z_s.
                amrex::Real const alpha = gamma_minus_1 * 3.0_rt * (f_s / Z_s);
                Te_arr(i,j,k) = Ti_K + (Te_K - Ti_K) * std::exp(-alpha * nu * dt);
            });
        }
    }

    Te.FillBoundary(Te.nGrowVect(), period);
}


void HybridPICModel::QDSMCApplyIonHeating (int const lev, amrex::Real const dt,
                                           amrex::MultiFab const * const redirect_E,
                                           std::map<std::string, amrex::MultiFab*> const * const Ti_dep_by_species) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCApplyIonHeating()");

    using warpx::fields::FieldType;

    // Stochastic Ornstein-Uhlenbeck ion-heating operator delivering both e-i energy
    // channels per particle over dt:
    //   v_p <- u_e + (v_p - u_e) exp(-nu_ei dt) + sig R,   R ~ N(0,1) per component.
    // Q_ei (when do_relax) sets the drag toward the electron fluid u_e and the thermal
    // diffusion sig^2 = k_B T_e/m_i (1 - exp(-2 nu_ei dt)). The Te-threshold redirect
    // (when do_redir) adds pure-diffusion heating E_s/m_i, with the per-species
    // redirected energy E_s [J] read from redirect_E. Both channels are per-species
    // correct (own mass, own T_i, own redirect_E comp).
    auto & warpx = WarpX::GetInstance();

    bool const do_relax = m_include_temperature_relaxation;
    bool const do_redir = (redirect_E != nullptr);
    if (!do_relax && !do_redir) { return; }

    amrex::MultiFab const & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp, lev);
    // u_e is only needed by the Q_ei drag; the fluid pe solver with the redirect alone
    // does not register hybrid_electron_velocity_fp
    ablastr::fields::VectorField Ve{nullptr, nullptr, nullptr};
    if (do_relax) {
        Ve = warpx.m_fields.get_alldirs(FieldType::hybrid_electron_velocity_fp, lev);
    }

    auto const rho_floor = PhysConst::q_e * m_n_floor;
    auto const nu_ei     = m_nu_ei;
    auto const t_new     = warpx.gett_new(0);
    auto const K_per_eV  = PhysConst::q_e / PhysConst::kb;   // T[eV]*this = T[K]
    // Floor on T_e in the nu_ei rate argument so pow(Te,-1.5) stays finite.
    amrex::Real const Te_floor_eV = 1.e-3_rt;

    // Nodal->cc interpolation staggers (unused dims set cc-like).
    amrex::GpuArray<int, 3> nodal_src = {1, 1, 1};
    for (int d = AMREX_SPACEDIM; d < 3; ++d) { nodal_src[d] = 0; }
    amrex::GpuArray<int, 3> const cc_dst  = {0, 0, 0};
    amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};

    amrex::BoxArray const cc_ba = amrex::convert(Te.boxArray(), amrex::IntVect::TheCellVector());

    auto & mypc = warpx.GetPartContainer();
    auto const species_names = mypc.GetSpeciesNames();

    // Charged-species component index for redirect_E (matches QDSMCAddJouleHeating:
    // incremented for every charged species before the mass check).
    int ion_comp = -1;
    for (auto const & spec_name : species_names) {
        auto & pc = mypc.GetParticleContainerFromName(spec_name);
        if (pc.getCharge() == 0._prt) { continue; }
        ++ion_comp;
        auto const m_i = pc.getMass();
        if (m_i <= 0._prt) { continue; }

        // Ion temperature [eV] (NGP) -- only needed as the nu_ei parser argument
        // (Q_ei drag/diffusion). Skipped when only the redirect is active. When
        // relaxation is on, T_i was deposited once by the caller and is shared via
        // Ti_dep_by_species (QDSMCAddTemperatureRelaxation ran just before with no
        // intervening ion motion).
        amrex::MultiFab Ti_cc(cc_ba, Te.DistributionMap(), 1, 0);
        Ti_cc.setVal(0.0_rt);
        if (do_relax) {
            amrex::MultiFab const & Ti_dep = *(Ti_dep_by_species->at(spec_name));
            Ti_cc.ParallelCopy(Ti_dep, 0, 0, 1, amrex::IntVect::TheZeroVector(),
                               amrex::IntVect::TheZeroVector());
        }

        // Per-cell drag-diffusion coefficients on the cc field grid:
        //   0 = nu_ei [1/s], 1-3 = u_e [m/s], 4 = T_e [K], 5 = redirected dTe [K].
        // Defaults (0) leave inactive / below-floor cells as no-ops.
        amrex::MultiFab coef(cc_ba, Te.DistributionMap(), 6, 0);
        coef.setVal(0.0_rt);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(coef, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Array4<amrex::Real>       const & coef_arr = coef.array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr  = rho.const_array(mfi);
            amrex::Array4<amrex::Real const> const & Te_arr   = Te.const_array(mfi);
            amrex::Array4<amrex::Real const> const & Ti_arr   = Ti_cc.const_array(mfi);
            amrex::Array4<amrex::Real const> Vex_arr, Vey_arr, Vez_arr;   // read only when do_relax
            if (do_relax) {
                Vex_arr = Ve[0]->const_array(mfi);
                Vey_arr = Ve[1]->const_array(mfi);
                Vez_arr = Ve[2]->const_array(mfi);
            }
            amrex::Array4<amrex::Real const> redirect_arr;
            if (do_redir) { redirect_arr = redirect_E->const_array(mfi); }

            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real const rho_val = ablastr::coarsen::sample::Interp(
                    rho_arr, nodal_src, cc_dst, coarsen, i, j, k, 0);
                if (rho_val <= rho_floor) { return; }

                amrex::Real const Te_K = ablastr::coarsen::sample::Interp(
                    Te_arr, nodal_src, cc_dst, coarsen, i, j, k, 0);
                coef_arr(i,j,k,4) = Te_K;

                if (do_relax) {
                    amrex::Real const Ti_eV = Ti_arr(i,j,k);
                    coef_arr(i,j,k,0) = nu_ei(rho_val, amrex::max(Te_K / K_per_eV, Te_floor_eV), Ti_eV, t_new);
                    coef_arr(i,j,k,1) = ablastr::coarsen::sample::Interp(
                        Vex_arr, nodal_src, cc_dst, coarsen, i, j, k, 0);
                    coef_arr(i,j,k,2) = ablastr::coarsen::sample::Interp(
                        Vey_arr, nodal_src, cc_dst, coarsen, i, j, k, 0);
                    coef_arr(i,j,k,3) = ablastr::coarsen::sample::Interp(
                        Vez_arr, nodal_src, cc_dst, coarsen, i, j, k, 0);
                }
                if (do_redir) {
                    // E_s for this species = redirect_E component ion_comp.
                    coef_arr(i,j,k,5) = ablastr::coarsen::sample::Interp(
                        redirect_arr, nodal_src, cc_dst, coarsen, i, j, k, ion_comp);
                }
            });
        }

        // Stage the coefficients on the particle grid for NGP lookup.
        auto const & pba = pc.ParticleBoxArray(lev);
        auto const & pdm = pc.ParticleDistributionMap(lev);
        amrex::MultiFab coef_p(pba, pdm, 6, 0);
        coef_p.setVal(0.0_rt);
        coef_p.ParallelCopy(coef, 0, 0, 6);

        // Apply the drag-diffusion update to each ion (NGP cell lookup).
        auto const plo = warpx.Geom(lev).ProbLoArray();
        auto const dxi = warpx.Geom(lev).InvCellSizeArray();
        auto const kb  = PhysConst::kb;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (WarpXParIter pti(pc, lev); pti.isValid(); ++pti)
        {
            long const np = pti.numParticles();
            auto & tile = pti.GetParticleTile();
            auto ptd = tile.getParticleTileData();
            amrex::ParticleReal* AMREX_RESTRICT uxp = pti.GetAttribs(PIdx::ux).dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT uyp = pti.GetAttribs(PIdx::uy).dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT uzp = pti.GetAttribs(PIdx::uz).dataPtr();

            amrex::Array4<amrex::Real const> const & coef_arr = coef_p.const_array(pti);

            amrex::ParallelForRNG(np,
                [=] AMREX_GPU_DEVICE (long ip, amrex::RandomEngine const& engine)
            {
                auto const p = WarpXParticleContainer::ParticleType(ptd, ip);
                const auto [ii, jj, kk] = amrex::getParticleCell(p, plo, dxi).dim3();
                // coef_p has no guard cells: a particle sitting on the tile's upper face
                // (or one not yet redistributed) indexes past the array -- an undefined
                // read that gave one ion a keV kick (3D formation, 40M ions, 2026-09-17:
                // "shape does not fit within guard cells" at the next deposit)
                if (ii < coef_arr.begin.vect[0] || ii >= coef_arr.end.vect[0] ||
                    jj < coef_arr.begin.vect[1] || jj >= coef_arr.end.vect[1] ||
                    kk < coef_arr.begin.vect[2] || kk >= coef_arr.end.vect[2]) { return; }
                amrex::ParticleReal const nu   = coef_arr(ii,jj,kk,0);
                amrex::ParticleReal const Te_K = coef_arr(ii,jj,kk,4);
                amrex::ParticleReal const E_s  = coef_arr(ii,jj,kk,5);

                // Ornstein-Uhlenbeck drag and variance (Q_ei diffusion + redirect E_s).
                amrex::ParticleReal const nu_dt = nu * dt;
                amrex::ParticleReal const drag = -std::expm1(-nu_dt);            // 1 - exp(-nu dt)
                amrex::ParticleReal const sig2 =
                    (-kb * Te_K * std::expm1(-2._prt * nu_dt) + E_s) / m_i;
                if (drag <= 0._prt && sig2 <= 0._prt) { return; }

                amrex::ParticleReal const uex = coef_arr(ii,jj,kk,1);
                amrex::ParticleReal const uey = coef_arr(ii,jj,kk,2);
                amrex::ParticleReal const uez = coef_arr(ii,jj,kk,3);
                amrex::ParticleReal const sig = std::sqrt(amrex::max(0._prt, sig2));
                uxp[ip] += -drag*(uxp[ip]-uex) + sig*amrex::RandomNormal(0._prt, 1._prt, engine);
                uyp[ip] += -drag*(uyp[ip]-uey) + sig*amrex::RandomNormal(0._prt, 1._prt, engine);
                uzp[ip] += -drag*(uzp[ip]-uez) + sig*amrex::RandomNormal(0._prt, 1._prt, engine);
            });
        }
    }
}


void HybridPICModel::FillPeFromTe (int const lev) const
{
    ABLASTR_PROFILE("HybridPICModel::FillPeFromTe()");

    auto & warpx = WarpX::GetInstance();

    amrex::MultiFab       & Pe  = *warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    amrex::MultiFab const & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp, lev);

    auto const rho_floor = PhysConst::q_e * m_n_floor;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Pe, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real>       const & Pe_arr  = Pe.array(mfi);
        amrex::Array4<amrex::Real const> const & Te_arr  = Te.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);

        amrex::Box const & tbox = mfi.tilebox();
        amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real const rho_val = std::max(rho_arr(i,j,k), rho_floor);
            amrex::Real const ne      = rho_val / PhysConst::q_e;
            Pe_arr(i,j,k) = ne * PhysConst::kb * Te_arr(i,j,k);
        });
    }
}



void HybridPICModel::SeedTeAdiabat (int const lev) const
{
    ABLASTR_PROFILE("HybridPICModel::SeedTeAdiabat()");

    auto & warpx = WarpX::GetInstance();

    amrex::MultiFab       & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp, lev);

    auto const n0_ref    = m_n0_ref;
    auto const gamma     = m_gamma;
    auto const rho_floor = PhysConst::q_e * m_n_floor;
    // m_elec_temp is k_B T_e0 [J]; the T_e field is in K.
    auto const Te0_K     = m_elec_temp / PhysConst::kb;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real>       const & Te_arr  = Te.array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);

        amrex::Box const tbox = amrex::convert(mfi.tilebox(), Te.ixType().toIntVect());
        amrex::Box       box  = tbox;
        box.grow(Te.nGrowVect());

        auto const uniform = m_te_seed_uniform;
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (uniform) {
                // te_seed_uniform: isothermal seed Te = Te0 everywhere
                Te_arr(i,j,k) = Te0_K;
                return;
            }
            amrex::Real const ne =
                amrex::max(rho_arr(i,j,k), rho_floor) / PhysConst::q_e;
            Te_arr(i,j,k) = Te0_K * std::pow(ne / n0_ref, gamma - 1.0_rt);
        });
    }
}


void HybridPICModel::ApplyIonElectronEnergyExchange (
    const int lev, amrex::Real const dt, amrex::MultiFab* ion_redirect_E) const
{
    // Symmetric collisional energy exchange between the electron fluid and the
    // ion particles: the Q_ei relaxation sinks/sources T_e and the stochastic
    // drag-diffusion operator applies the exact conjugate to the ions (plus any
    // staged redirected Joule energy). Total energy is conserved between the
    // two sides by construction. Requires rho_fp, rho_fp_<species> and the
    // per-species T_<species> deposits to be current.
    auto & warpx = WarpX::GetInstance();
    amrex::ignore_unused(warpx);
    const bool redirect_active = (ion_redirect_E != nullptr);
    if (!m_include_temperature_relaxation && !redirect_active) { return; }

    // Steps 6b/6c both need each charged species' T_i when Q_ei relaxation is
    // on. Deposit it ONCE here (the expensive per-particle NGP temperature
    // reduction) and share it: the electron sink (6b) and the ion-heating
    // operator (6c) run back-to-back with no intervening ion motion, so the
    // deposited T_i is identical for both.
    std::map<std::string, amrex::MultiFab*> Ti_dep_by_species;
    // Owns the per-species cell-centered scalar T_i built from the shape-aware
    // deposition below; must outlive the QDSMCAddTemperatureRelaxation /
    // QDSMCApplyIonHeating calls that read it through Ti_dep_by_species.
    std::map<std::string, std::unique_ptr<amrex::MultiFab>> Ti_scalar_owned;
    if (m_include_temperature_relaxation) {
        using ablastr::fields::Direction;
        amrex::GpuArray<int, 3> const Tr_stag = Jx_IndexType;
        amrex::GpuArray<int, 3> const Tt_stag = Jy_IndexType;
        amrex::GpuArray<int, 3> const Tz_stag = Jz_IndexType;
        amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};
        // Cell-centered target in the real dimensions; in collapsed dimensions
        // (index >= AMREX_SPACEDIM, e.g. theta in RZ or y in 2D) match the source
        // staggering so Interp does not read the out-of-bounds neighbour there.
        amrex::GpuArray<int, 3> cc_r = {0, 0, 0};
        amrex::GpuArray<int, 3> cc_t = {0, 0, 0};
        amrex::GpuArray<int, 3> cc_z = {0, 0, 0};
        for (int d = AMREX_SPACEDIM; d < 3; ++d) {
            cc_r[d] = Tr_stag[d]; cc_t[d] = Tt_stag[d]; cc_z[d] = Tz_stag[d];
        }

        auto & mpc_ti = warpx.GetPartContainer();
        for (auto const & nm : mpc_ti.GetSpeciesNames()) {
            auto & pc = mpc_ti.GetParticleContainerFromName(nm);
            if (pc.getCharge() == 0._prt) { continue; }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(pc.getTemperatureDepositionFlag(),
                "The Q_ei temperature relaxation requires do_temperature_deposition "
                "on every charged ion species; it is enabled automatically at species "
                "construction, so hitting this indicates the species was created "
                "before the hybrid_pic_model relaxation parameters were readable.");

            // Shape-aware ion temperature (particle-shape order, consistent with
            // charge/current) in the Yee-staggered 3-component T_<nm> vector field
            // (Tr,Tt,Tz), deposited in Kelvin by HybridPICDepositRhoAndJ ->
            // mypc->DepositTemperatures earlier this step and read here. Fill guard
            // cells so the cell-centered interpolation below reads finite
            // neighbours at box/domain edges.
            auto const T_vf = warpx.m_fields.get_mr_levels_alldirs("T_" + nm, warpx.finestLevel());
            for (int idim = 0; idim < 3; ++idim) {
                T_vf[lev][Direction{idim}]->FillBoundary(warpx.Geom(lev).periodicity());
            }

            // Collapse the staggered vector to a cell-centered scalar
            // T_i = (Tr + Tt + Tz)/3 by interpolating each component to CC
            // (Path A: accept the CC interpolation for shape consistency).
            amrex::MultiFab const & Te_ref =
                *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
            amrex::BoxArray const cc_ba =
                amrex::convert(Te_ref.boxArray(), amrex::IntVect::TheCellVector());
            auto Ti_s = std::make_unique<amrex::MultiFab>(
                cc_ba, Te_ref.DistributionMap(), 1, 0);
            Ti_s->setVal(0.0_rt);

            // AccumulateVelocitiesAndComputeTemperature writes T_<nm> in Kelvin;
            // the Q_ei consumers (and the nu_ei parser) expect T_i in eV, matching
            // the previous NGP deposit. Convert K -> eV below.
            amrex::Real const K_per_eV = PhysConst::q_e / PhysConst::kb;

            amrex::MultiFab const & Tr = *T_vf[lev][Direction{0}];
            amrex::MultiFab const & Tt = *T_vf[lev][Direction{1}];
            amrex::MultiFab const & Tz = *T_vf[lev][Direction{2}];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(*Ti_s, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                amrex::Box const & bx = mfi.tilebox();
                amrex::Array4<amrex::Real>       const & Ti_arr = Ti_s->array(mfi);
                amrex::Array4<amrex::Real const> const & Tr_arr = Tr.const_array(mfi);
                amrex::Array4<amrex::Real const> const & Tt_arr = Tt.const_array(mfi);
                amrex::Array4<amrex::Real const> const & Tz_arr = Tz.const_array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    amrex::Real const tr = ablastr::coarsen::sample::Interp(
                        Tr_arr, Tr_stag, cc_r, coarsen, i, j, k, 0);
                    amrex::Real const tt = ablastr::coarsen::sample::Interp(
                        Tt_arr, Tt_stag, cc_t, coarsen, i, j, k, 0);
                    amrex::Real const tz = ablastr::coarsen::sample::Interp(
                        Tz_arr, Tz_stag, cc_z, coarsen, i, j, k, 0);
                    Ti_arr(i, j, k) = (tr + tt + tz) / (3._rt * K_per_eV);  // K -> eV
                });
            }
            Ti_dep_by_species[nm] = Ti_s.get();
            Ti_scalar_owned[nm] = std::move(Ti_s);
        }
    }

    // Step 6b: electron-ion thermal-equilibration (Q_ei) sink on T_e
    // (cools T_e toward each ion species' T_i).
    if (m_include_temperature_relaxation) {
        QDSMCAddTemperatureRelaxation(lev, dt, Ti_dep_by_species);
    }

    // Step 6c: stochastic drag-diffusion ion-heating operator -- delivers the Q_ei
    // conjugate (when relaxation is on) and/or the redirected Joule energy
    // (when the redirect is on), so the ions are heated by one mechanism.
    if (m_include_temperature_relaxation || redirect_active) {
        QDSMCApplyIonHeating(lev, dt, ion_redirect_E,
                             m_include_temperature_relaxation ? &Ti_dep_by_species : nullptr);
    }

}

void HybridPICModel::AdvanceElectronEnergyQDSMC (amrex::Real const dt) const
{
    ABLASTR_PROFILE("HybridPICModel::AdvanceElectronEnergyQDSMC()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_qdsmc_pc != nullptr,
        "AdvanceElectronEnergyQDSMC called with "
        "solve_electron_energy_equation=true but the "
        "QDSMC particle container was not constructed (InitData not run?)");

    auto & warpx = WarpX::GetInstance();

    // J_plasma (at B^n) is needed for V_e. On all but the first step it is
    // already valid: the previous step's final E-solve computed it from B^n
    // (the external-field subtract at the top of this step exactly cancels
    // the re-add at the end of the previous one), and B has not changed
    // since. Only the first step of a run or restart arrives here with an
    // unfilled J_plasma.
    if (!m_qdsmc_J_plasma_valid) {
        CalculatePlasmaCurrent(
            warpx.m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, warpx.finestLevel()),
            warpx.GetEBUpdateEFlag());
        m_qdsmc_J_plasma_valid = true;
    }

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        // Step 1: grid-side initialization at t = n
        QDSMCInitializeUe(lev);
        QDSMCInitializeKe(lev);

        using ablastr::fields::Direction;
        amrex::MultiFab const & Vex = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{0}, lev);
        amrex::MultiFab const & Vey = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{1}, lev);
        amrex::MultiFab const & Vez = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{2}, lev);
        amrex::MultiFab const & Ke  = *warpx.m_fields.get(FieldType::hybrid_entropy_fp,           lev);
        amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::hybrid_rho_fp_temp,          lev);
        amrex::MultiFab       & Karr_out    = *warpx.m_fields.get(FieldType::hybrid_entropy_fp,        lev);
        amrex::MultiFab       & weights_out = *warpx.m_fields.get(FieldType::hybrid_qdsmc_weights_fp, lev);

        // Step 2: load each QDSMC particle with V_e and (K_e * N_e, N_e) from
        // its home cell.
        m_qdsmc_pc->SetV(lev, Vex, Vey, Vez);
        m_qdsmc_pc->SetK(lev, Ke, rho);

        // Step 3: forward-Euler push by dt; redistribute so particles end up
        // in their new tile.
        m_qdsmc_pc->PushX(lev, dt);

        // Step 4: scatter the carried entropy and weight onto the grid (each
        // call zeroes its target field, then deposits, then SumBoundary).
        m_qdsmc_pc->DepositK(lev, Karr_out);
        m_qdsmc_pc->DepositField(lev, weights_out);

        // Step 5: recover T_e^{n+1} from (deposited K*N) / (deposited N) and
        // the updated n_e (from rho_fp = rho^{n+1}).
        QDSMCUpdateTe(lev);

        // Step 6: Joule-heating source on T_e (Phys. Plasmas 31, 012902 (2024), Eq. 12), per-cell from
        // rho_fp(_s), the plasma current, and the Ohm's-law eta parser. With the
        // Te-threshold redirect on, the above-threshold heat is staged in
        // ion_redirect_E (per-charged-species energy, J) for the ion-heating step.
        bool redirect_active = m_include_joule_heating && m_joule_redirect_to_ions;
        int n_ion_species = 0;
        if (redirect_active) {
            auto & mpc = warpx.GetPartContainer();
            for (auto const & nm : mpc.GetSpeciesNames()) {
                if (mpc.GetParticleContainerFromName(nm).getCharge() != 0._prt) { ++n_ion_species; }
            }
            if (n_ion_species == 0) { redirect_active = false; }
        }
        amrex::MultiFab ion_redirect_E;
        if (redirect_active) {
            amrex::MultiFab const & Te_mf =
                *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
            ion_redirect_E.define(Te_mf.boxArray(), Te_mf.DistributionMap(), n_ion_species, 0);
            ion_redirect_E.setVal(0.0_rt);
        }
        if (m_include_joule_heating) {
            QDSMCAddJouleHeating(lev, dt, redirect_active ? &ion_redirect_E : nullptr);
        }

        // Steps 6b/6c: symmetric ion-electron collisional energy exchange
        // (Q_ei electron sink + conjugate ion heating) and/or delivery of the
        // redirected Joule energy. Shared with the implicit scheme, which calls
        // it directly (its transport and Joule heating are handled in-loop).
        ApplyIonElectronEnergyExchange(lev, dt,
            redirect_active ? &ion_redirect_E : nullptr);

        // Step 7: emit P_e = n_e * k_B * T_e for the downstream Ohm's-law
        // solve, with the same boundary treatment the algebraic closure gets
        // in CalculateElectronPressure (grad P_e reads the ghost cells).
        FillPeFromTe(lev);
        warpx.ApplyElectronPressureBoundary(lev, PatchType::fine);
        ablastr::utils::communication::FillBoundary(
            *warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev),
            WarpX::do_single_precision_comms,
            warpx.Geom(lev).periodicity(),
            true);

        // Step 8: reset particles to home positions (and zero velocity /
        // weight / entropy) so the next step starts with a clean grid.
        m_qdsmc_pc->ResetParticles(lev);
    }
}


void HybridPICModel::AdvanceElectronPressureFluid (
    const int lev, amrex::Real const dt_sub, amrex::Real const t_eval,
    ablastr::fields::VectorField const& a_Jfield,
    amrex::MultiFab const& a_rho) const
{
    ABLASTR_PROFILE("HybridPICModel::AdvanceElectronPressureFluid()");
    using namespace amrex::literals;

    auto& warpx = WarpX::GetInstance();
    const amrex::Geometry& geom = warpx.Geom(lev);
    amrex::MultiFab* pe = warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    ablastr::fields::VectorField Jp =
        warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    ablastr::fields::VectorField Bf =
        warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev);

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    amrex::ignore_unused(dt_sub, t_eval, a_Jfield, a_rho, geom, pe, Jp, Bf);
    WARPX_ABORT_WITH_MESSAGE(
        "hybrid_pic_model.electron_energy_solver = fluid is implemented for "
        "Cartesian collocated grids only");
#else
    // Same spatial operators as the theta-implicit in-loop advance on the
    // collocated grid: J, J_plasma, B, rho and pe all nodal.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_Jfield[2]->ixType().nodeCentered() && Jp[2]->ixType().nodeCentered()
        && Bf[2]->ixType().nodeCentered() && pe->ixType().nodeCentered(),
        "hybrid_pic_model.electron_energy_solver = fluid requires the collocated "
        "grid (warpx.grid_type = collocated)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        pe->nGrowVect().min() >= 2 && a_rho.nGrowVect().min() >= 1
        && a_Jfield[0]->nGrowVect().min() >= 1 && Jp[0]->nGrowVect().min() >= 1
        && Bf[0]->nGrowVect().min() >= 1,
        "hybrid_pic_model.electron_energy_solver = fluid needs >= 2 pe ghost "
        "cells and >= 1 on rho, J, J_plasma and B");

    // The face fluxes read the neighbor nodes of the freshly computed plasma
    // current; the ion current and density of this half step arrive with
    // filled ghosts from the deposition.
    for (int d = 0; d < 3; ++d) { Jp[d]->FillBoundary(geom.periodicity()); }

    if (!m_pe_fluid_scratch) {
        m_pe_fluid_scratch = std::make_unique<amrex::MultiFab>(
            pe->boxArray(), pe->DistributionMap(), pe->nComp(), pe->nGrowVect());
    }
    amrex::MultiFab::Copy(*m_pe_fluid_scratch, *pe, 0, 0, pe->nComp(), pe->nGrowVect());

    const auto dxi = geom.InvCellSizeArray();
    const amrex::Real gamma = m_gamma;
    const amrex::Real rho_floor = m_n_floor * PhysConst::q_e;
    const amrex::Real floor_w = m_n_floor_smooth_width * rho_floor;
    const amrex::Real q_e = PhysConst::q_e;
    // width of the C1 positivity floor as a fraction of the floored-adiabat pe
    constexpr amrex::Real pe_floor_eps_fac = 0.01_rt;
    const amrex::Real pe_eps = pe_floor_eps_fac
        * HybridPeFlooredAdiabat(m_n_floor, m_elec_temp, m_n0_ref, gamma);
    // tanh soft cap on the flux velocity (0 = off); the work term uses the
    // uncapped Ohm's-law velocity
    const amrex::Real ue_cap = m_pe_ue_cap;
    const bool jheat = m_include_joule_heating;
    // Te-threshold Joule gate (joule_redirect_Te_threshold Tc, width w Tc): the electrons
    // receive sJ Q with sJ = 0.5 (1 - tanh((Te - Tc)/(w Tc))) (hard step at w = 0), Te^s =
    // pe^s/rho at the node; the gated-out (1 - sJ) Q dt_sub is staged per charged species
    // as the per-ion energy E_s = (2/3) Z_s (1 - sJ) w_v Q dt_sub / n_e [J] for the
    // ion-heating kick of FinishElectronPressureFluid (the implicit scheme's convention,
    // including the Holmstrom weight w_v: in the floored halo the Joule power is that of
    // the vacuum model, not a plasma process)
    const bool jgate = jheat && m_joule_redirect_to_ions;
    const amrex::Real jcut   = m_joule_redirect_Te_eV;
    const amrex::Real jcut_w = m_joule_redirect_Te_width * m_joule_redirect_Te_eV;
    if (jgate && !m_pe_redirect_E) {
        auto& mypc = warpx.GetPartContainer();
        m_pe_redirect_nspec = 0;
        for (auto const& nm : mypc.GetSpeciesNames()) {
            auto& pc = mypc.GetParticleContainerFromName(nm);
            if (pc.getCharge() == 0._prt) { continue; }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_pe_redirect_nspec < kMaxFluidRedirectSpecies,
                "electron_energy_solver = fluid + joule_redirect_Te_threshold: too many "
                "charged species");
            m_pe_redirect_Z[m_pe_redirect_nspec++] =
                static_cast<amrex::Real>(pc.getCharge() / PhysConst::q_e);
        }
        m_pe_redirect_E = std::make_unique<amrex::MultiFab>(
            pe->boxArray(), pe->DistributionMap(), std::max(m_pe_redirect_nspec, 1),
            amrex::IntVect::TheZeroVector());
        m_pe_redirect_E->setVal(0.0_rt);
        m_pe_redirect_entry = std::make_unique<amrex::MultiFab>(
            pe->boxArray(), pe->DistributionMap(), std::max(m_pe_redirect_nspec, 1),
            amrex::IntVect::TheZeroVector());
        m_pe_redirect_entry->setVal(0.0_rt);
    }
    const int n_redir = jgate ? m_pe_redirect_nspec : 0;
    const amrex::GpuArray<amrex::Real, kMaxFluidRedirectSpecies> Zs = m_pe_redirect_Z;
    // Holmstrom vacuum region: blend the advanced pe toward the floored adiabat with the
    // (statistics-aware) vacuum weight Ohm's law uses, as the implicit in-loop advance
    // does (u_e = (J_i - J)/rho is noise in near-empty cells)
    const bool vac_blend = m_holmstrom_vacuum_region;
    const amrex::Real vac_rref = m_vacuum_weight_r_ref;
    const amrex::Real vac_maxf = m_vacuum_weight_max_factor;
    const amrex::Real ad_n0 = m_n0_ref;
    const amrex::Real ad_T0 = m_elec_temp;
    const amrex::Real dx0 = geom.CellSize(0);
    const bool has_kappa = m_has_kappa_e;
    const bool has_kexpr = m_has_kappa_e_expression;
    const amrex::Real kappa_e = m_kappa_e;
    const auto kappa_ex = m_kappa;
    const auto eta_ex  = m_eta;
    const auto etah_ex = m_eta_h;
    const bool inc_hyp = m_include_hyper_resistivity_term;
    const int pe_adv = m_pe_advection;
    const HybridPeFluxParams pfp{m_pe_advection};

    const amrex::Box dom_nodal =
        amrex::convert(geom.Domain(), amrex::IntVect::TheNodeVector());
    const amrex::Dim3 dlo = amrex::lbound(dom_nodal);
    const amrex::Dim3 dhi = amrex::ubound(dom_nodal);
    amrex::GpuArray<bool, 3> is_per = {true, true, true};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { is_per[d] = geom.isPeriodic(d); }

    // EB: covered nodes are frozen at pe^s; the flag is the collocated-nodal
    // component-0 E update flag
    const amrex::iMultiFab* eb_pe_flag = EB::enabled()
        ? warpx.GetEBUpdateEFlag()[lev][0].get() : nullptr;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*pe, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Array4<amrex::Real>       const& pe_arr  = pe->array(mfi);
        amrex::Array4<amrex::Real const> const& pe0     = m_pe_fluid_scratch->const_array(mfi);
        amrex::Array4<amrex::Real const> const& rho_arr = a_rho.const_array(mfi);
        amrex::Array4<int const> ebp;
        if (eb_pe_flag) { ebp = eb_pe_flag->const_array(mfi); }
        amrex::Array4<amrex::Real const> const& Jx  = a_Jfield[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jy  = a_Jfield[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jz  = a_Jfield[2]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jpx = Jp[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jpy = Jp[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jpz = Jp[2]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Bx  = Bf[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& By  = Bf[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Bz  = Bf[2]->const_array(mfi);
        amrex::Array4<amrex::Real> rd;
        if (n_redir > 0) { rd = m_pe_redirect_E->array(mfi); }

        const amrex::Box tb = mfi.tilebox(amrex::IntVect::TheNodeVector());

        amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            amrex::ignore_unused(Jy, Jpy);
            // EB-covered node: frozen at pe^s
            if (ebp && ebp(i,j,k) == 0) {
                pe_arr(i,j,k) = pe0(i,j,k);
                return;
            }
            // non-periodic domain-face nodes are owned by the pressure BC
            if (((i <= dlo.x || i >= dhi.x) && !is_per[0])
#if (AMREX_SPACEDIM >= 2)
                || ((j <= dlo.y || j >= dhi.y) && !is_per[1])
#endif
#if (AMREX_SPACEDIM == 3)
                || ((k <= dlo.z || k >= dhi.z) && !is_per[2])
#endif
               ) { return; }

            // Two electron velocities: the work term uses the uncapped Ohm's-law
            // velocity (smooth density floor), the flux velocity the hard floor
            // and the tanh cap.
            auto ue_raw = [&] (amrex::Array4<amrex::Real const> const& J,
                               amrex::Array4<amrex::Real const> const& JP,
                               int ii, int jj, int kk) {
                return (J(ii,jj,kk) - JP(ii,jj,kk))
                    / HybridSmoothFloor(rho_arr(ii,jj,kk,0), rho_floor, floor_w);
            };
            auto uclamp = [&] (amrex::Real u) {
                return (ue_cap > 0._rt) ? ue_cap * std::tanh(u / ue_cap) : u;
            };
            auto ue_flux = [&] (amrex::Array4<amrex::Real const> const& J,
                                amrex::Array4<amrex::Real const> const& JP,
                                int ii, int jj, int kk) {
                return uclamp((J(ii,jj,kk) - JP(ii,jj,kk))
                              / amrex::max(rho_arr(ii,jj,kk,0), rho_floor));
            };
            auto ue_x = [&] (int ii, int jj, int kk) { return ue_flux(Jx, Jpx, ii, jj, kk); };
            auto ue_z = [&] (int ii, int jj, int kk) { return ue_flux(Jz, Jpz, ii, jj, kk); };
            auto pec = [&] (int ii, int jj, int kk) { return pe0(ii,jj,kk); };
            // electron conduction -div(q_e) = +div(kappa_e grad(pe/n)),
            // conservative face-flux form with face kappa = mean of the nodes
            auto Tn = [&] (int ii, int jj, int kk) {
                return pec(ii,jj,kk) * q_e / amrex::max(rho_arr(ii,jj,kk,0), rho_floor);
            };
            auto kapn = [&] (int ii, int jj, int kk) {
                if (!has_kexpr) { return kappa_e; }
                return kappa_ex(amrex::max(rho_arr(ii,jj,kk,0), rho_floor),
                                Tn(ii,jj,kk) / q_e);   // Te in eV
            };
            auto Fk = [&] (amrex::Real T0, amrex::Real T1,
                           amrex::Real k0, amrex::Real k1, amrex::Real dxi_f) {
                return 0.5_rt*(k0 + k1) * (T1 - T0) * dxi_f;
            };

            amrex::Real divF = 0._rt, W = 0._rt, cond = 0._rt;
#if defined(WARPX_DIM_1D_Z)
            amrex::ignore_unused(ue_x);
            if (pe_adv != 0) {
                auto Fz_face = [&] (int m) {
                    const amrex::Real uf = 0.5_rt*(ue_z(m,j,k) + ue_z(m+1,j,k));
                    return uf * HybridPeFace(pfp, uf, pec(m-1,j,k), pec(m,j,k),
                                             pec(m+1,j,k), pec(m+2,j,k));
                };
                divF = (Fz_face(i) - Fz_face(i-1)) * dxi[0];
            } else {
                divF = (ue_z(i+1,j,k)*pec(i+1,j,k) - ue_z(i-1,j,k)*pec(i-1,j,k)) * 0.5_rt * dxi[0];
            }
            // work term -E*.J_e = -ue.grad(pe) with the nodal Ohm's-law
            // (central) gradient; the Hall term does no work at a node
            W = -ue_raw(Jz, Jpz, i, j, k) * 0.5_rt*dxi[0]*(pec(i+1,j,k) - pec(i-1,j,k));
            if (has_kappa) {
                cond = (Fk(Tn(i,j,k),   Tn(i+1,j,k), kapn(i,j,k),   kapn(i+1,j,k), dxi[0])
                      - Fk(Tn(i-1,j,k), Tn(i,j,k),   kapn(i-1,j,k), kapn(i,j,k),  dxi[0])) * dxi[0];
            }
#elif defined(WARPX_DIM_XZ)
            if (pe_adv != 0) {
                auto Fx_face = [&] (int m) {
                    const amrex::Real uf = 0.5_rt*(ue_x(m,j,k) + ue_x(m+1,j,k));
                    return uf * HybridPeFace(pfp, uf, pec(m-1,j,k), pec(m,j,k),
                                             pec(m+1,j,k), pec(m+2,j,k));
                };
                auto Fz_face = [&] (int m) {
                    const amrex::Real uf = 0.5_rt*(ue_z(i,m,k) + ue_z(i,m+1,k));
                    return uf * HybridPeFace(pfp, uf, pec(i,m-1,k), pec(i,m,k),
                                             pec(i,m+1,k), pec(i,m+2,k));
                };
                divF = (Fx_face(i) - Fx_face(i-1)) * dxi[0]
                     + (Fz_face(j) - Fz_face(j-1)) * dxi[1];
            } else {
                divF = (ue_x(i+1,j,k)*pec(i+1,j,k) - ue_x(i-1,j,k)*pec(i-1,j,k)) * 0.5_rt * dxi[0]
                     + (ue_z(i,j+1,k)*pec(i,j+1,k) - ue_z(i,j-1,k)*pec(i,j-1,k)) * 0.5_rt * dxi[1];
            }
            W = -(ue_raw(Jx, Jpx, i, j, k) * 0.5_rt*dxi[0]*(pec(i+1,j,k) - pec(i-1,j,k))
                + ue_raw(Jz, Jpz, i, j, k) * 0.5_rt*dxi[1]*(pec(i,j+1,k) - pec(i,j-1,k)));
            if (has_kappa) {
                cond = (Fk(Tn(i,j,k),   Tn(i+1,j,k), kapn(i,j,k),   kapn(i+1,j,k), dxi[0])
                      - Fk(Tn(i-1,j,k), Tn(i,j,k),   kapn(i-1,j,k), kapn(i,j,k),   dxi[0])) * dxi[0]
                     + (Fk(Tn(i,j,k),   Tn(i,j+1,k), kapn(i,j,k),   kapn(i,j+1,k), dxi[1])
                      - Fk(Tn(i,j-1,k), Tn(i,j,k),   kapn(i,j-1,k), kapn(i,j,k),  dxi[1])) * dxi[1];
            }
#elif defined(WARPX_DIM_3D)
            auto ue_y = [&] (int ii, int jj, int kk) { return ue_flux(Jy, Jpy, ii, jj, kk); };
            if (pe_adv != 0) {
                auto Fx_face = [&] (int m) {
                    const amrex::Real uf = 0.5_rt*(ue_x(m,j,k) + ue_x(m+1,j,k));
                    return uf * HybridPeFace(pfp, uf, pec(m-1,j,k), pec(m,j,k),
                                             pec(m+1,j,k), pec(m+2,j,k));
                };
                auto Fy_face = [&] (int m) {
                    const amrex::Real uf = 0.5_rt*(ue_y(i,m,k) + ue_y(i,m+1,k));
                    return uf * HybridPeFace(pfp, uf, pec(i,m-1,k), pec(i,m,k),
                                             pec(i,m+1,k), pec(i,m+2,k));
                };
                auto Fz_face = [&] (int m) {
                    const amrex::Real uf = 0.5_rt*(ue_z(i,j,m) + ue_z(i,j,m+1));
                    return uf * HybridPeFace(pfp, uf, pec(i,j,m-1), pec(i,j,m),
                                             pec(i,j,m+1), pec(i,j,m+2));
                };
                divF = (Fx_face(i) - Fx_face(i-1)) * dxi[0]
                     + (Fy_face(j) - Fy_face(j-1)) * dxi[1]
                     + (Fz_face(k) - Fz_face(k-1)) * dxi[2];
            } else {
                divF = (ue_x(i+1,j,k)*pec(i+1,j,k) - ue_x(i-1,j,k)*pec(i-1,j,k)) * 0.5_rt * dxi[0]
                     + (ue_y(i,j+1,k)*pec(i,j+1,k) - ue_y(i,j-1,k)*pec(i,j-1,k)) * 0.5_rt * dxi[1]
                     + (ue_z(i,j,k+1)*pec(i,j,k+1) - ue_z(i,j,k-1)*pec(i,j,k-1)) * 0.5_rt * dxi[2];
            }
            W = -(ue_raw(Jx, Jpx, i, j, k) * 0.5_rt*dxi[0]*(pec(i+1,j,k) - pec(i-1,j,k))
                + ue_raw(Jy, Jpy, i, j, k) * 0.5_rt*dxi[1]*(pec(i,j+1,k) - pec(i,j-1,k))
                + ue_raw(Jz, Jpz, i, j, k) * 0.5_rt*dxi[2]*(pec(i,j,k+1) - pec(i,j,k-1)));
            if (has_kappa) {
                cond = (Fk(Tn(i,j,k),   Tn(i+1,j,k), kapn(i,j,k),   kapn(i+1,j,k), dxi[0])
                      - Fk(Tn(i-1,j,k), Tn(i,j,k),   kapn(i-1,j,k), kapn(i,j,k),   dxi[0])) * dxi[0]
                     + (Fk(Tn(i,j,k),   Tn(i,j+1,k), kapn(i,j,k),   kapn(i,j+1,k), dxi[1])
                      - Fk(Tn(i,j-1,k), Tn(i,j,k),   kapn(i,j-1,k), kapn(i,j,k),   dxi[1])) * dxi[1]
                     + (Fk(Tn(i,j,k),   Tn(i,j,k+1), kapn(i,j,k),   kapn(i,j,k+1), dxi[2])
                      - Fk(Tn(i,j,k-1), Tn(i,j,k),   kapn(i,j,k-1), kapn(i,j,k),  dxi[2])) * dxi[2];
            }
#endif
            // Holmstrom vacuum weight of the node (1 without the vacuum region)
            amrex::Real wv = 1._rt;
            if (vac_blend) {
                const amrex::Real rf = HybridVacuumFloor(rho_floor, 0._rt, vac_rref,
                                                         vac_maxf, dx0);
                wv = HybridExtSubWeight(rho_arr(i,j,k,0), rf, floor_w*(rf/rho_floor));
            }
            // positive-definite Joule deposit (resistive + hyper-resistive), Te-gated
            if (jheat) {
                const amrex::Real Q = HybridPeJouleQ(i, j, k, Jpx, Jpy, Jpz, Bx, By, Bz,
                                                     rho_arr, eta_ex, etah_ex, inc_hyp,
                                                     t_eval, dxi, dlo, dhi, is_per);
                amrex::Real sJ = 1._rt;
                if (jgate) {
                    // Te^s [eV] = pe^s / rho (floored rho)
                    const amrex::Real Te_eV =
                        pe0(i,j,k) / amrex::max(rho_arr(i,j,k,0), rho_floor);
                    sJ = (jcut_w > 0._rt)
                        ? 0.5_rt*(1._rt - std::tanh((Te_eV - jcut)/jcut_w))
                        : ((Te_eV >= jcut) ? 0._rt : 1._rt);
                    if (n_redir > 0) {
                        const amrex::Real ne =
                            amrex::max(rho_arr(i,j,k,0), rho_floor) / q_e;
                        const amrex::Real Wd = wv * (1._rt - sJ) * Q;   // withheld power density
                        for (int s = 0; s < n_redir; ++s) {
                            rd(i,j,k,s) += (2._rt/3._rt) * Zs[s] * Wd * dt_sub / ne;
                        }
                    }
                }
                W -= sJ * Q;
            }
            const amrex::Real pe_new = pe0(i,j,k)
                - dt_sub * (gamma * divF + (gamma - 1._rt) * (W - cond));
            amrex::Real pe_fin = pe_new;
            if (vac_blend) {
                const amrex::Real pe_ad = ElectronPressure::get_pressure(
                    ad_n0, ad_T0, gamma, amrex::max(rho_arr(i,j,k,0), rho_floor));
                pe_fin = wv*pe_new + (1._rt - wv)*pe_ad;
            }
            pe_arr(i,j,k) = HybridPeFloor(pe_fin, pe_eps);
        });
    }

    // Duplicated nodal points on box faces are written redundantly by each
    // box: force them single-valued, then refresh the ghosts read by the next
    // substep's Ohm's-law grad(pe) and by this update's stencils. The
    // non-periodic domain-face nodes were skipped above and stay static, so
    // only the ghost mirrors are rewritten (rewrite_pec_nodes = false).
    pe->OverrideSync(geom.periodicity());
    warpx.ApplyElectronPressureBoundary(lev, PatchType::fine, /*rewrite_pec_nodes=*/false);
    pe->FillBoundary(geom.periodicity());
#endif
}

void HybridPICModel::FinishElectronPressureFluid (amrex::Real const dt) const
{
    ABLASTR_PROFILE("HybridPICModel::FinishElectronPressureFluid()");
    using namespace amrex::literals;
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        // Post-step halo pin: floored cells (rho <= rho_floor) hold the
        // floored-adiabat pe, the rest max(pe, 0), so no electron heat
        // accumulates where there is no electron fluid.
        {
            amrex::MultiFab* pe = warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
            const amrex::MultiFab* rho = warpx.m_fields.get(FieldType::rho_fp, lev);
            const amrex::Real rho_floor = m_n_floor * PhysConst::q_e;
            const amrex::Real pe_vac =
                HybridPeFlooredAdiabat(m_n_floor, m_elec_temp, m_n0_ref, m_gamma);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(*pe, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box bx = mfi.growntilebox();
                auto const& a = pe->array(mfi);
                auto const& r = rho->const_array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    a(i,j,k) = (r(i,j,k,0) <= rho_floor)
                        ? pe_vac : amrex::max(a(i,j,k), 0._rt);
                });
            }
        }
        // T_e = pe/(n k_B) from pe^{n+1} and rho^{n+1} ("Te" diagnostic; also
        // the state the Q_ei exchange relaxes)
        FillTeFromPe(lev);
        // Symmetric Q_ei ion-electron exchange once per step on the t^{n+1} state and/or
        // the Te-threshold Joule redirect: the energy withheld from the electrons over
        // this step's B substeps goes to the ions through the same stochastic heating
        // kick (one NGP lookup per ion; the ions sit in their valid cells since the
        // step's deposit). The Q_ei drag needs u_e (hybrid_electron_velocity_fp).
        const bool redir = (m_pe_redirect_E != nullptr) && (m_pe_redirect_nspec > 0);
        if (m_include_temperature_relaxation || redir) {
            if (m_include_temperature_relaxation) { QDSMCInitializeUe(lev); }
            ApplyIonElectronEnergyExchange(lev, dt, redir ? m_pe_redirect_E.get() : nullptr);
            if (redir) { m_pe_redirect_E->setVal(0.0_rt); }
        }
        if (m_include_temperature_relaxation) {
            // re-emit pe from the relaxed T_e
            FillPeFromTe(lev);
            warpx.ApplyElectronPressureBoundary(lev, PatchType::fine);
            ablastr::utils::communication::FillBoundary(
                *warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev),
                WarpX::do_single_precision_comms,
                warpx.Geom(lev).periodicity(),
                true);
        }
    }
}

void HybridPICModel::FilterPushFieldsSwap (const bool a_apply) const
{
    // Conservative smoothing: the ions gather the binomial-filtered E (the
    // adjoint of the filtered current deposition) while Ohm's law and Faraday
    // keep the unfiltered registry field; B needs no treatment.
    using ablastr::fields::Direction;
    auto& warpx = WarpX::GetInstance();
    const int lev = 0;
    for (int n = 0; n < 3; ++n) {
        amrex::MultiFab& Emf = *warpx.m_fields.get(FieldType::Efield_fp, Direction{n}, lev);
        if (a_apply) {
            if (!m_E_unfiltered[n]) {
                m_E_unfiltered[n] = std::make_unique<amrex::MultiFab>(
                    Emf.boxArray(), Emf.DistributionMap(), Emf.nComp(), Emf.nGrowVect());
            }
            amrex::MultiFab::Copy(*m_E_unfiltered[n], Emf, 0, 0, Emf.nComp(), Emf.nGrowVect());
        } else {
            amrex::MultiFab::Copy(Emf, *m_E_unfiltered[n], 0, 0, Emf.nComp(), Emf.nGrowVect());
        }
    }
    if (a_apply) {
        warpx.ApplyFilterMF(
            warpx.m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, lev), lev);
    }
}

void HybridPICModel::BfieldEvolve (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    int step, amrex::Real dt_half, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        BfieldEvolve(
            Bfield, Efield, Jfield, rhofield, eb_update_E,
            step, dt_half, lev, subcycling_half, ng, nodal_sync
        );
    }
}

void HybridPICModel::BfieldEvolve (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    int step, amrex::Real dt_half, int lev, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    bool use_rkf45 = DoRKF45(step);
    // Make copies of the current B-field multifabs (at t = n) since the
    // starting B-field is needed for the integration logic.
    // We also store the initial B-field from the start of this integration step
    // (i.e., a static copy) in case we need to fully reset the Bfield (needed
    // for RK4).
    std::array< MultiFab, 3 > B_old;
    for (int ii = 0; ii < 3; ii++)
    {
        B_old[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 2, ng
        );
        MultiFab::Copy(B_old[ii], *Bfield[lev][ii], 0, 0, 1, ng);
        // the values at index 1 will be kept static through the integration steps
        MultiFab::Copy(B_old[ii], *Bfield[lev][ii], 0, 1, 1, ng);
    }

    // Fluid electron pressure: forward Euler over each accepted substep with the
    // plasma current of the end-of-substep B. pe is snapshotted so a NaN rewind
    // of the half step (RK4 -> RKF45 restart) does not integrate it twice.
    const bool fluid_pe = FluidElectronPressure();
    amrex::MultiFab* pe_mf = nullptr;
    if (fluid_pe) {
        auto& warpx = WarpX::GetInstance();
        pe_mf = warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
        if (!m_pe_fluid_entry) {
            m_pe_fluid_entry = std::make_unique<MultiFab>(
                pe_mf->boxArray(), pe_mf->DistributionMap(), pe_mf->nComp(), pe_mf->nGrowVect());
        }
        MultiFab::Copy(*m_pe_fluid_entry, *pe_mf, 0, 0, pe_mf->nComp(), pe_mf->nGrowVect());
        if (m_pe_redirect_E) {
            MultiFab::Copy(*m_pe_redirect_entry, *m_pe_redirect_E, 0, 0,
                           m_pe_redirect_E->nComp(), 0);
        }
    }
    auto advance_pe = [&] (amrex::Real dt_pe, amrex::Real t_end) {
        CalculatePlasmaCurrent(Bfield[lev], eb_update_E[lev], lev);
        const amrex::Real t_eval = WarpX::GetInstance().gett_old(0) + t_end
            + ((subcycling_half == SubcyclingHalf::SecondHalf) ? dt_half : 0._rt);
        AdvanceElectronPressureFluid(lev, dt_pe, t_eval, Jfield[lev], *rhofield[lev]);
    };

    amrex::Real dt_sub = dt_half / (m_substeps / 2._rt);
    amrex::Real t = 0._rt;
    int n_attempts = 0;
    int n_accepted = 0;

    // Step the magnetic field forward (from t -> t + dt_half) using the user
    // specified integration scheme. The loop is set up such that the timestep
    // for a given step (dt_sub) can be modified within the loop, i.e.,
    // adaptive timestepping.
    while (t < dt_half)
    {
        // Adjust size of the last substep, so as to land exactly at t+dt_half.
        if (t + dt_sub > dt_half) { dt_sub = dt_half - t; }
        bool step_succeeded = true;
        amrex::Real step_change_factor = 1.0_rt;

        if (use_rkf45) {
            const amrex::Real error = BfieldEvolveRKF45(
                Bfield, Efield, Jfield, rhofield, eb_update_E, B_old,
                dt_sub, lev, subcycling_half, ng, nodal_sync
            );

            step_change_factor = m_substep_safety * std::pow(error + 1.e-10_rt, -0.2_rt);
            step_succeeded = (error <= 1._rt);

        } else {
            BfieldEvolveRK4(
                Bfield, Efield, Jfield, rhofield, eb_update_E, B_old,
                dt_sub, lev, subcycling_half, ng, nodal_sync
            );

            // Check that the B-field does not have nan or inf values
            for (int idim = 0; idim < 3; ++idim) {
                step_succeeded = step_succeeded && Bfield[lev][idim]->is_finite(/*local=*/true);
            }
            amrex::ParallelDescriptor::ReduceBoolAnd(step_succeeded);

            if (!step_succeeded) {
                ablastr::warn_manager::WMRecordWarning(
                    "HybridPIC",
                    "NaN or Inf value encountered in the B-field during RK4 "
                    "substepping. Restarting this step using RKF45.",
                    ablastr::warn_manager::WarnPriority::medium);

                // restart this full step and this time use RKF45
                t = 0._rt;
                n_accepted = 0;
                // reset B_old to original one
                for (int ii = 0; ii < 3; ii++) {
                    MultiFab::Copy(B_old[ii], B_old[ii], 1, 0, 1, ng);
                }
                // and the fluid pe to its entry state (already advanced over
                // the accepted substeps of the abandoned attempt)
                if (fluid_pe) {
                    MultiFab::Copy(*pe_mf, *m_pe_fluid_entry, 0, 0,
                                   pe_mf->nComp(), pe_mf->nGrowVect());
                    if (m_pe_redirect_E) {
                        MultiFab::Copy(*m_pe_redirect_E, *m_pe_redirect_entry, 0, 0,
                                       m_pe_redirect_E->nComp(), 0);
                    }
                }
                use_rkf45 = true;
            }
        }

        if (step_succeeded) {
            // fluid pe: advance over the accepted substep (a rejected or reset
            // B substep never moves pe)
            if (fluid_pe) { advance_pe(dt_sub, t + dt_sub); }
            // update time tracker and accepted steps number
            t += dt_sub;
            ++n_accepted;
            // update B_old to the current Bfield
            for (int ii = 0; ii < 3; ii++) {
                MultiFab::Copy(B_old[ii], *Bfield[lev][ii], 0, 0, 1, ng);
            }
            dt_sub *= std::min(m_substep_max_growth, step_change_factor);
        } else {
            // reset Bfield to B_old before trying the integration again
            for (int ii = 0; ii < 3; ii++) {
                MultiFab::Copy(*Bfield[lev][ii], B_old[ii], 0, 0, 1, ng);
            }
            dt_sub *= std::max(0.1_rt, step_change_factor);
        }

        if (++n_attempts > m_max_substep_attempts) { break; }
    }

    // Adjust the number of substeps. This affects both the next RKF45 or RK4 step.
    // The adjustment is made to jump to more required substeps or slowly decrease
    // if m_substeps is too large (using 95% of the current m_substeps value and
    // 5% of the lower, new value).
    if (m_substeps < 2*n_attempts) {
        m_substeps = 2*n_attempts;
    } else {
        m_substeps = 2 * int(std::ceil(0.475 * m_substeps + 0.05 * n_attempts));
    }

    if (WarpX::GetInstance().Verbose()) {
        amrex::Print() << "B-field update "
            << (subcycling_half == SubcyclingHalf::FirstHalf ? "1st" : "2nd") << " half"
            << ": " << n_accepted << " accepted, "
            << (n_attempts - n_accepted) << " rejected substeps"
            << " (dt_sub_final/dt_half = " << dt_sub / dt_half << ")\n";
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        n_attempts <= m_max_substep_attempts,
        "BfieldEvolve: exceeded max substep attempts;"
        "consider relaxing hybrid_pic_model.substep_rtol/substep_atol."
    );
}

void HybridPICModel::BfieldEvolveRK4 (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    std::array<amrex::MultiFab, 3>& B_old,
    amrex::Real dt, int lev, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    // Create multifabs for each direction to store the Runge-Kutta intermediate terms.
    // Each multifab has 2 components for the different terms that need to be stored.
    std::array< MultiFab, 3 > K;
    for (int ii = 0; ii < 3; ii++)
    {
        K[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 2,
            Bfield[lev][ii]->nGrowVect()
        );
    }

    // The Runge-Kutta scheme begins here.
    // Step 1:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, eb_update_E,
        0.5_rt*dt, subcycling_half, ng, nodal_sync
    );

    // The Bfield is now given by:
    // B_new = B_old + 0.5 * dt * [-curl x E(B_old)] = B_old + 0.5 * dt * K0.
    for (int ii = 0; ii < 3; ii++)
    {
        // Extract 0.5 * dt * K0 for each direction into index 0 of K.
        MultiFab::LinComb(
            K[ii], 1._rt, *Bfield[lev][ii], 0, -1._rt, B_old[ii], 0, 0, 1, ng
        );
    }

    // Step 2:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, eb_update_E,
        0.5_rt*dt, subcycling_half, ng, nodal_sync
    );

    // The Bfield is now given by:
    //   B_new = B_old + 0.5 * dt * K0 + 0.5 * dt * [-curl x E(B_old + 0.5 * dt * K1)]
    //         = B_old + 0.5 * dt * K0 + 0.5 * dt * K1
    //
    // Subtract 0.5 * dt * K0 from the Bfield to get
    //   B_new = B_old + 0.5 * dt * K1.
    // Extract 0.5 * dt * K1 and write into index 1 of K.

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const &Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const &By = Bfield[lev][1]->array(mfi);
        Array4<Real> const &Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const &Kx = K[0].array(mfi);
        Array4<Real> const &Ky = K[1].array(mfi);
        Array4<Real> const &Kz = K[2].array(mfi);
        Array4<Real const> const &Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const &By_old = B_old[1].const_array(mfi);
        Array4<Real const> const &Bz_old = B_old[2].const_array(mfi);

        // Extract tileboxes for which to loop
        Box const& tjx  = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy  = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz  = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);

        amrex::ParallelFor(tjx, tjy, tjz,
            // x calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Bx(i, j, k) -= Kx(i, j, k, 0);
                Kx(i, j, k, 1) = Bx(i, j, k) - Bx_old(i, j, k);
            },

            // y calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                By(i, j, k) -= Ky(i, j, k, 0);
                Ky(i, j, k, 1) = By(i, j, k) - By_old(i, j, k);
            },

            // z calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Bz(i, j, k) -= Kz(i, j, k, 0);
                Kz(i, j, k, 1) = Bz(i, j, k) - Bz_old(i, j, k);
            }
        );
    }

    // Step 3:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, eb_update_E,
        dt, subcycling_half, ng, nodal_sync
    );

    // The Bfield is now given by:
    // B_new = B_old + 0.5 * dt * K1 + dt * [-curl  x E(B_old + 0.5 * dt * K1)]
    //       = B_old + 0.5 * dt * K1 + dt * K2
    for (int ii = 0; ii < 3; ii++)
    {
        // Subtract 0.5 * dt * K1 from the Bfield for each direction to get
        // B_new = B_old + dt * K2.
        MultiFab::Subtract(*Bfield[lev][ii], K[ii], 1, 0, 1, ng);
    }

    // Step 4:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, eb_update_E,
        0.5_rt*dt, subcycling_half, ng, nodal_sync
    );

    // The Bfield is now given by:
    //   B_new = B_old + dt * K2 + 0.5 * dt * [-curl x E(B_old + dt * K2)]
    //         = B_old + dt * K2 + 0.5 * dt * K3
    // and
    //   index 0 of K = 0.5 * dt * K0
    //   index 1 of K = 0.5 * dt * K1
    //
    // We calculate:
    //   K = 0.5 * dt * K0 + dt * K1 + dt * K2 + 0.5 * dt * K3
    // then update B with the Runge-Kutta sum:
    //   B = B_old + 1/3 * K

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const &Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const &By = Bfield[lev][1]->array(mfi);
        Array4<Real> const &Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const &Kx = K[0].array(mfi);
        Array4<Real> const &Ky = K[1].array(mfi);
        Array4<Real> const &Kz = K[2].array(mfi);
        Array4<Real const> const &Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const &By_old = B_old[1].const_array(mfi);
        Array4<Real const> const &Bz_old = B_old[2].const_array(mfi);

        // Extract tileboxes for which to loop
        Box const& tjx  = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy  = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz  = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);

        amrex::ParallelFor(tjx, tjy, tjz,
            // Bx calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Kx(i, j, k, 0) += Bx(i, j, k) - Bx_old(i, j, k) + 2.0_rt * Kx(i, j, k, 1);
                Bx(i, j, k) = Bx_old(i, j, k) + Kx(i, j, k, 0) / 3.0_rt;
            },

            // By calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Ky(i, j, k, 0) += By(i, j, k) - By_old(i, j, k) + 2.0_rt * Ky(i, j, k, 1);
                By(i, j, k) = By_old(i, j, k) + Ky(i, j, k, 0) / 3.0_rt;
            },

            // Bz calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Kz(i, j, k, 0) += Bz(i, j, k) - Bz_old(i, j, k) + 2.0_rt * Kz(i, j, k, 1);
                Bz(i, j, k) = Bz_old(i, j, k) + Kz(i, j, k, 0) / 3.0_rt;
            }
        );
    }
}

amrex::Real HybridPICModel::BfieldEvolveRKF45 (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    std::array<amrex::MultiFab, 3>& B_old,
    amrex::Real dt, int lev, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    // Fehlberg RKF45 Butcher tableau coefficients
    constexpr amrex::Real a21 = 1._rt/4._rt;
    constexpr amrex::Real a31 = 3._rt/32._rt,      a32 = 9._rt/32._rt;
    constexpr amrex::Real a41 = 1932._rt/2197._rt,  a42 = -7200._rt/2197._rt, a43 = 7296._rt/2197._rt;
    constexpr amrex::Real a51 = 439._rt/216._rt,    a52 = -8._rt,
                          a53 = 3680._rt/513._rt,    a54 = -845._rt/4104._rt;
    constexpr amrex::Real a61 = -8._rt/27._rt,      a62 = 2._rt,
                          a63 = -3544._rt/2565._rt,  a64 = 1859._rt/4104._rt,  a65 = -11._rt/40._rt;
    // 4th-order solution weights (k2 and k6 terms are zero in Fehlberg's formula)
    constexpr amrex::Real b1 = 25._rt/216._rt,  b3 = 1408._rt/2565._rt,
                          b4 = 2197._rt/4104._rt, b5 = -1._rt/5._rt;
    // Error = B5 - B4 weights: h*(e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6)
    constexpr amrex::Real e1 =  1._rt/360._rt,    e3 = -128._rt/4275._rt,
                          e4 = -2197._rt/75240._rt, e5 = 1._rt/50._rt, e6 = 2._rt/55._rt;

    // K: 5 components per field direction stored as:
    //   comp 0 = h*k1, comp 1 = h*k2 (overwritten with h*k6 after stage 6),
    //   comp 2 = h*k3, comp 3 = h*k4, comp 4 = h*k5
    std::array<MultiFab, 3> K;
    std::array<MultiFab, 3> err_scratch;
    for (int ii = 0; ii < 3; ii++)
    {
        K[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 5,
            Bfield[lev][ii]->nGrowVect()
        );
        err_scratch[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 1,
            amrex::IntVect(0)
        );
    }

    // ---- Stage 1: B = B_old, FieldPush, K[comp0] = h*k1 fused with Stage 2 B-update ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx, tjy, tjz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Bx(i, j, k) - Bx_old(i, j, k);
                Kx(i, j, k, 0) = k1;
                Bx(i, j, k) = Bx_old(i, j, k) + a21*k1;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = By(i, j, k) - By_old(i, j, k);
                Ky(i, j, k, 0) = k1;
                By(i, j, k) = By_old(i, j, k) + a21*k1;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Bz(i, j, k) - Bz_old(i, j, k);
                Kz(i, j, k, 0) = k1;
                Bz(i, j, k) = Bz_old(i, j, k) + a21*k1;
            }
        );
    }

    // ---- Stage 2: FieldPush, K[comp1] = h*k2 fused with Stage 3 B-update ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
    // Stage 2 K[1]-readback fused with Stage 3 B-update.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx, tjy, tjz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kx(i, j, k, 0);
                amrex::Real const k2 = Bx(i, j, k) - Bx_old(i, j, k) - a21*k1;
                Kx(i, j, k, 1) = k2;
                Bx(i, j, k) = Bx_old(i, j, k) + a31*k1 + a32*k2;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Ky(i, j, k, 0);
                amrex::Real const k2 = By(i, j, k) - By_old(i, j, k) - a21*k1;
                Ky(i, j, k, 1) = k2;
                By(i, j, k) = By_old(i, j, k) + a31*k1 + a32*k2;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kz(i, j, k, 0);
                amrex::Real const k2 = Bz(i, j, k) - Bz_old(i, j, k) - a21*k1;
                Kz(i, j, k, 1) = k2;
                Bz(i, j, k) = Bz_old(i, j, k) + a31*k1 + a32*k2;
            }
        );
    }

    // ---- Stage 3: FieldPush, then K[comp2] = h*k3 fused with Stage 4 B-update ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx, tjy, tjz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kx(i, j, k, 0);
                amrex::Real const k2 = Kx(i, j, k, 1);
                amrex::Real const k3 = Bx(i, j, k) - Bx_old(i, j, k) - a31*k1 - a32*k2;
                Kx(i, j, k, 2) = k3;
                Bx(i, j, k) = Bx_old(i, j, k) + a41*k1 + a42*k2 + a43*k3;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Ky(i, j, k, 0);
                amrex::Real const k2 = Ky(i, j, k, 1);
                amrex::Real const k3 = By(i, j, k) - By_old(i, j, k) - a31*k1 - a32*k2;
                Ky(i, j, k, 2) = k3;
                By(i, j, k) = By_old(i, j, k) + a41*k1 + a42*k2 + a43*k3;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kz(i, j, k, 0);
                amrex::Real const k2 = Kz(i, j, k, 1);
                amrex::Real const k3 = Bz(i, j, k) - Bz_old(i, j, k) - a31*k1 - a32*k2;
                Kz(i, j, k, 2) = k3;
                Bz(i, j, k) = Bz_old(i, j, k) + a41*k1 + a42*k2 + a43*k3;
            }
        );
    }

    // ---- Stage 4: FieldPush, then K[comp3] = h*k4 fused with Stage 5 B-update ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx, tjy, tjz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kx(i, j, k, 0);
                amrex::Real const k2 = Kx(i, j, k, 1);
                amrex::Real const k3 = Kx(i, j, k, 2);
                amrex::Real const k4 = Bx(i, j, k) - Bx_old(i, j, k)
                                        - a41*k1 - a42*k2 - a43*k3;
                Kx(i, j, k, 3) = k4;
                Bx(i, j, k) = Bx_old(i, j, k) + a51*k1 + a52*k2 + a53*k3 + a54*k4;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Ky(i, j, k, 0);
                amrex::Real const k2 = Ky(i, j, k, 1);
                amrex::Real const k3 = Ky(i, j, k, 2);
                amrex::Real const k4 = By(i, j, k) - By_old(i, j, k)
                                        - a41*k1 - a42*k2 - a43*k3;
                Ky(i, j, k, 3) = k4;
                By(i, j, k) = By_old(i, j, k) + a51*k1 + a52*k2 + a53*k3 + a54*k4;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kz(i, j, k, 0);
                amrex::Real const k2 = Kz(i, j, k, 1);
                amrex::Real const k3 = Kz(i, j, k, 2);
                amrex::Real const k4 = Bz(i, j, k) - Bz_old(i, j, k)
                                        - a41*k1 - a42*k2 - a43*k3;
                Kz(i, j, k, 3) = k4;
                Bz(i, j, k) = Bz_old(i, j, k) + a51*k1 + a52*k2 + a53*k3 + a54*k4;
            }
        );
    }

    // ---- Stage 5: FieldPush, then K[comp4] = h*k5 fused with Stage 6 B-update ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx, tjy, tjz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kx(i, j, k, 0);
                amrex::Real const k2 = Kx(i, j, k, 1);
                amrex::Real const k3 = Kx(i, j, k, 2);
                amrex::Real const k4 = Kx(i, j, k, 3);
                amrex::Real const k5 = Bx(i, j, k) - Bx_old(i, j, k)
                                        - a51*k1 - a52*k2 - a53*k3 - a54*k4;
                Kx(i, j, k, 4) = k5;
                Bx(i, j, k) = Bx_old(i, j, k)
                            + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Ky(i, j, k, 0);
                amrex::Real const k2 = Ky(i, j, k, 1);
                amrex::Real const k3 = Ky(i, j, k, 2);
                amrex::Real const k4 = Ky(i, j, k, 3);
                amrex::Real const k5 = By(i, j, k) - By_old(i, j, k)
                                        - a51*k1 - a52*k2 - a53*k3 - a54*k4;
                Ky(i, j, k, 4) = k5;
                By(i, j, k) = By_old(i, j, k)
                            + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kz(i, j, k, 0);
                amrex::Real const k2 = Kz(i, j, k, 1);
                amrex::Real const k3 = Kz(i, j, k, 2);
                amrex::Real const k4 = Kz(i, j, k, 3);
                amrex::Real const k5 = Bz(i, j, k) - Bz_old(i, j, k)
                                        - a51*k1 - a52*k2 - a53*k3 - a54*k4;
                Kz(i, j, k, 4) = k5;
                Bz(i, j, k) = Bz_old(i, j, k)
                            + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5;
            }
        );
    }

    // ---- Stage 6: FieldPush, then K[comp1] = h*k6 (overwrites h*k2) fused with B4 + error ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
    // K[comp1] is overwritten here: reads h*k2 (old value) then writes h*k6 in each cell.
    // k6, B4 assembly (b2=0, so k2 is not needed for B4), and error assembly are fused into
    // one ParallelFor per direction. B4 is updated over ghost+valid cells; error is written
    // only for valid cells (err_scratch has no ghost), guarded by a box check.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real> const& error_x = err_scratch[0].array(mfi);
        Array4<Real> const& error_y = err_scratch[1].array(mfi);
        Array4<Real> const& error_z = err_scratch[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect());
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect());
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect());
        Box const& tjx_ng = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy_ng = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz_ng = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx_ng, tjy_ng, tjz_ng,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kx(i, j, k, 0);
                amrex::Real const k2 = Kx(i, j, k, 1);
                amrex::Real const k3 = Kx(i, j, k, 2);
                amrex::Real const k4 = Kx(i, j, k, 3);
                amrex::Real const k5 = Kx(i, j, k, 4);
                amrex::Real const k6 = Bx(i, j, k) - Bx_old(i, j, k)
                                        - a61*k1 - a62*k2 - a63*k3 - a64*k4 - a65*k5;
                Kx(i, j, k, 1) = k6;
                Bx(i, j, k) = Bx_old(i, j, k) + b1*k1 + b3*k3 + b4*k4 + b5*k5;
                if (tjx.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
                    error_x(i, j, k) = e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6;
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Ky(i, j, k, 0);
                amrex::Real const k2 = Ky(i, j, k, 1);
                amrex::Real const k3 = Ky(i, j, k, 2);
                amrex::Real const k4 = Ky(i, j, k, 3);
                amrex::Real const k5 = Ky(i, j, k, 4);
                amrex::Real const k6 = By(i, j, k) - By_old(i, j, k)
                                        - a61*k1 - a62*k2 - a63*k3 - a64*k4 - a65*k5;
                Ky(i, j, k, 1) = k6;
                By(i, j, k) = By_old(i, j, k) + b1*k1 + b3*k3 + b4*k4 + b5*k5;
                if (tjy.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
                    error_y(i, j, k) = e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6;
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kz(i, j, k, 0);
                amrex::Real const k2 = Kz(i, j, k, 1);
                amrex::Real const k3 = Kz(i, j, k, 2);
                amrex::Real const k4 = Kz(i, j, k, 3);
                amrex::Real const k5 = Kz(i, j, k, 4);
                amrex::Real const k6 = Bz(i, j, k) - Bz_old(i, j, k)
                                        - a61*k1 - a62*k2 - a63*k3 - a64*k4 - a65*k5;
                Kz(i, j, k, 1) = k6;
                Bz(i, j, k) = Bz_old(i, j, k) + b1*k1 + b3*k3 + b4*k4 + b5*k5;
                if (tjz.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
                    error_z(i, j, k) = e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6;
                }
            }
        );
    }

    // ---- Error norm and adaptive step control ----
    // Compute local maxima first, then one combined AllReduce for both norms.
    amrex::Real err_norm = 0._rt;
    amrex::Real B4_norm  = 0._rt;
    for (int ii = 0; ii < 3; ii++) {
        err_norm = std::max(err_norm, err_scratch[ii].norm0(/*comp=*/0, /*nghost=*/0, /*local=*/true));
        B4_norm  = std::max(B4_norm,  Bfield[lev][ii]->norm0(/*comp=*/0, /*nghost=*/0, /*local=*/true));
    }
    amrex::ParallelDescriptor::ReduceRealMax({err_norm, B4_norm});
    return err_norm / (m_substep_atol + m_substep_rtol * B4_norm);
}


void HybridPICModel::FieldPush (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    amrex::Real dt, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    auto& warpx = WarpX::GetInstance();

    amrex::Real const t_old = warpx.gett_old(0);

    // Calculate J = curl x B / mu0 - J_ext
    CalculatePlasmaCurrent(Bfield, eb_update_E);
    // Calculate the E-field from Ohm's law; with an evolved electron pressure
    // grad(pe)/(e n) stays in the Faraday E, since its curl no longer vanishes.
    HybridPICSolveE(Efield, Jfield, Bfield, rhofield, eb_update_E, true,
                    FluidElectronPressure());
    // Call FillBoundary if a collocated grid is used
    if (Bz_IndexType[0] == Ez_IndexType[0]) {
        warpx.FillBoundaryE(ng, nodal_sync);
    }

    // Push forward the B-field using Faraday's law
    warpx.EvolveB(dt, subcycling_half, t_old);
    warpx.FillBoundaryB(ng, nodal_sync);
}
