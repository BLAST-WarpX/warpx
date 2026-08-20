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

#include <ablastr/coarsen/sample.H>
#include <ablastr/utils/Communication.H>
#include <ablastr/warn_manager/WarnManager.H>

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

    utils::parser::queryWithParser(pp_hybrid, "n_floor", m_n_floor);

    // Master gate for the electron-energy equation. When enabled, K_e is
    // advected each step by fictitious Lagrangian particles moving with V_e
    // (see Phys. Plasmas 31, 012902 (2024)); T_e is recovered from K_e and n_e
    // via the polytropic relation; operator-split source terms are added;
    // Pe = n_e k_B T_e is emitted for the Ohm's-law E-solve. Default off
    // preserves the legacy algebraic adiabatic closure.
    pp_hybrid.query("solve_electron_energy_equation",
                    m_solve_electron_energy_equation);
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
    // Default off; only consulted when solve_electron_energy_equation is on.
    pp_hybrid.query("include_joule_heating", m_include_joule_heating);

    // Te-threshold Joule redirection: heat electrons where Te < threshold,
    // deposit the Joule energy to ions where Te >= threshold. Off by default
    // (threshold < 0); specifying a threshold >= 0 enables the redirect.
    utils::parser::queryWithParser(pp_hybrid, "joule_redirect_Te_threshold", m_joule_redirect_Te_eV);
    m_joule_redirect_to_ions = (m_joule_redirect_Te_eV >= 0._rt);

    // Electron-ion thermal equilibration (Q_ei) on T_e:
    //   Q_ei = 3 n_e k_B nu_ei (T_e - T_i),  applied per ion species weighted by
    //   n_s/n_e, cooling T_e toward T_i. nu_ei[1/s] comes from the
    //   electron_ion_relaxation_rate(rho,Te,Ti,t) parser (rho [C/m^3], Te,Ti [eV]).
    //   The matching ion heating is deposited conservatively, so the exchange
    //   conserves energy. Enabled by specifying the rate expression (only
    //   consulted when solve_electron_energy_equation is on).
    m_include_temperature_relaxation =
        pp_hybrid.query("electron_ion_relaxation_rate(rho,Te,Ti,t)", m_nu_ei_expression);

    // The electron-energy equation's Joule and Q_ei sources read the
    // per-species charge densities; this flag gates their allocation and
    // the per-species deposition path, so hybrid-PIC runs without the
    // energy equation carry zero extra cost.
    m_need_per_species_fields = m_solve_electron_energy_equation;

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
    // it on, Pe = n_e k_B T_e is emitted by QDSMCFillElectronPressureFromTe
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
    // therefore only allocated) when the energy equation is solved:
    //   * hybrid_entropy_fp              : K_e = T_e * n_e^(1-gamma)
    //   * hybrid_qdsmc_weights_fp        : scratch for deposited N_e
    //   * hybrid_electron_velocity_fp    : three-component V_e on a NODAL
    //     grid, computed each step from V_e = -(J_plasma - J_i)/(q_e n_e)
    //     and consumed by the QDSMC particle SetV step to advect the
    //     entropy carriers.
    if (m_solve_electron_energy_equation) {
        fields.alloc_init(FieldType::hybrid_entropy_fp,
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);
        fields.alloc_init(FieldType::hybrid_qdsmc_weights_fp,
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);
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

    // Per-species charge densities - one per charged species, deposited
    // from particles and accumulated into the global rho_fp. Only
    // allocated when a feature that consumes them is active (see
    // m_need_per_species_fields).
    if (m_need_per_species_fields) {
        auto const & mypc = WarpX::GetInstance().GetPartContainer();
        for (auto const & spec : mypc.GetSpeciesNames()) {
            if (mypc.GetParticleContainerFromName(spec).getCharge() == 0._prt) { continue; }
            fields.alloc_init("rho_fp_" + spec,
                lev, amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngRho, 0.0_rt);
        }
        // Species-summed physical charge density Sigma_s rho_fp_s, filled
        // once per step in HybridPICDepositRhoAndJ (volume-scaled in radial
        // geometries like the totals, but unfiltered: the same processing as
        // the rho_fp_s numerators, so the species fraction
        // f_s = rho_s / Sigma_t rho_t is well-defined and the physical
        // rho_floor applies to it). Shared by the Joule and Q_ei consumers.
        fields.alloc_init("hybrid_rho_species_sum_fp",
            lev, amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngRho, 0.0_rt);
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
    // zero-initialized allocation. This value does not survive into the
    // solve: CalculateElectronPressure overwrites T_e from the closure, both
    // each step on the algebraic path and once from HybridPICInitializeRhoJandB
    // (on the floored density) to seed the energy-equation path.
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        amrex::MultiFab & Te_mf = *warpx.m_fields.get(
            FieldType::hybrid_electron_temperature_fp, lev);
        Te_mf.setVal(m_elec_temp / PhysConst::kb);
    }

    // QDSMC: lazy-construct the fictitious-particle container and lay one
    // particle per cell.
    if (m_solve_electron_energy_equation) {
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
    const bool solve_for_Faraday) const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        HybridPICSolveE(
            Efield[lev], Jfield[lev], Bfield[lev], *rhofield[lev],
            eb_update_E[lev], lev, solve_for_Faraday
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
    const int lev, const bool solve_for_Faraday) const
{
    ABLASTR_PROFILE("WarpX::HybridPICSolveE()");

    HybridPICSolveE(
        Efield, Jfield, Bfield, rhofield, eb_update_E, lev,
        PatchType::fine, solve_for_Faraday
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
    const bool solve_for_Faraday) const
{
    auto& warpx = WarpX::GetInstance();

    ablastr::fields::VectorField current_fp_plasma = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    auto* const electron_pressure_fp = warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);

    // Solve E field in regular cells
    warpx.get_pointer_fdtd_solver_fp(lev)->HybridPICSolveE(
        Efield, current_fp_plasma, Jfield, Bfield, rhofield,
        *electron_pressure_fp, eb_update_E, lev, this, solve_for_Faraday
    );
    amrex::Real const time = warpx.gett_old(0) + warpx.getdt(0);
    warpx.ApplyEfieldBoundary(lev, patch_type, time);
}

void HybridPICModel::CalculateElectronPressure(bool const floor_density) const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        CalculateElectronPressure(lev, floor_density);
    }
}

void HybridPICModel::CalculateElectronPressure(const int lev, bool const floor_density) const
{
    ABLASTR_PROFILE("WarpX::CalculateElectronPressure()");

    auto& warpx = WarpX::GetInstance();
    ablastr::fields::ScalarField electron_temperature_fp = warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    ablastr::fields::ScalarField electron_pressure_fp = warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    ablastr::fields::ScalarField rho_fp = warpx.m_fields.get(FieldType::rho_fp, lev);

    // Calculate the electron pressure (and its implied temperature) using rho^{n+1}.
    FillElectronPressureMF(
        *electron_pressure_fp,
        *electron_temperature_fp,
        *rho_fp,
        floor_density
    );
    warpx.ApplyElectronPressureBoundary(lev, PatchType::fine);
    ablastr::utils::communication::FillBoundary(
        *electron_pressure_fp,
        WarpX::do_single_precision_comms,
        warpx.Geom(lev).periodicity(),
        true);
}

void HybridPICModel::FillElectronPressureMF (
    amrex::MultiFab& Pe_field,
    amrex::MultiFab& Te_field,
    amrex::MultiFab const& rho_field,
    bool const floor_density
) const
{
    const auto n0_ref = m_n0_ref;
    const auto elec_temp = m_elec_temp;
    const auto gamma_minus_1 = m_gamma - 1.0_rt;
    // Only bites when floor_density is set: max(rho, 0) leaves every physical
    // rho >= 0 bit-for-bit alone, so the algebraic-closure path is unchanged.
    const auto rho_floor =
        floor_density ? PhysConst::q_e * m_n_floor : amrex::Real(0.0);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(Pe_field, TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {
        // Extract field data for this grid/tile
        Array4<Real const> const& rho = rho_field.const_array(mfi);
        Array4<Real> const& Te = Te_field.array(mfi);
        Array4<Real> const& Pe = Pe_field.array(mfi);

        // Extract tileboxes for which to loop
        Box tilebox = mfi.tilebox();
        // Cover the ghosts too.
        // QDSMCInitializeKe reads T_e over its own ghost-grown box
        // so the seed has to leave T_e's ghosts valid itself.
        // Out-of-domain ghosts are handled at the density floor.
        tilebox.grow(Pe_field.nGrowVect());

        ParallelFor(tilebox, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Polytropic closure: T_e = T0 (n_e/n0)^(gamma-1), in the units of
            // elec_temp (Joules), with P_e = n_e T_e following from it. The
            // "Te" diagnostic wants Kelvin. Flooring n_e once here keeps P_e
            // and T_e consistent with each other.
            const Real ne = std::max(rho(i, j, k), rho_floor) / PhysConst::q_e;
            const Real Te_joule = elec_temp * std::pow(ne/n0_ref, gamma_minus_1);
            Pe(i, j, k) = ne * Te_joule;
            Te(i, j, k) = Te_joule / PhysConst::kb;
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
    // After the QDSMC scatter, weights_fp holds the deposited extensive
    // electron count and entropy_fp holds K_e times that count. Recover T_e:
    //
    //   K_e_new = entropy_fp / weights_fp
    //   T_e_new = K_e_new / (n_e_new^(1-gamma) * k_B / q_e)
    //
    // n_e_new comes from rho_fp (post-deposit, post-particle-push).

    amrex::MultiFab       & Te      = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & Ke      = *warpx.m_fields.get(FieldType::hybrid_entropy_fp,              lev);
    amrex::MultiFab const & weights = *warpx.m_fields.get(FieldType::hybrid_qdsmc_weights_fp,        lev);
    amrex::MultiFab const & rho     = *warpx.m_fields.get(FieldType::rho_fp,                         lev);

    // Note: T_e is NOT zeroed here. Cells that received no QDSMC weight
    // keep their previous T_e -- zeroing them would erase valid state (and
    // seed a wrong K_e into neighbors on the next step) whenever a cell
    // momentarily receives no deposit.

    auto const gamma      = m_gamma;
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
            // Guard the division: a cell no QDSMC marker reached has exactly
            // zero deposited weight and keeps its previous T_e. Cells that did
            // receive weight are all updated, however small the deposit -- the
            // (K*N)/N ratio is well conditioned there because numerator and
            // denominator carry the same small factor.
            if (weights_arr(i,j,k) <= 0.0_rt) { return; }
            amrex::Real const w = weights_arr(i,j,k);
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

    auto const gamma_minus_1 = m_gamma - 1.0_rt;
    auto const rho_floor     = PhysConst::q_e * m_n_floor;
    auto const eta           = m_eta;
    auto const t_new         = warpx.gett_new(0);

    amrex::GpuArray<int, 3> const & Jx_stag = Jx_IndexType;
    amrex::GpuArray<int, 3> const & Jy_stag = Jy_IndexType;
    amrex::GpuArray<int, 3> const & Jz_stag = Jz_IndexType;
    amrex::GpuArray<int, 3> const nodal     = {1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen   = {1, 1, 1};

    // Te-threshold Joule redirection: in cells with Te >= threshold the Joule
    // heat is written into redirect_E (per charged ion species, the m_i-independent
    // energy E_s [J]) for QDSMCApplyIonHeating to deposit on the ions, rather than
    // added to T_e.
    bool const do_redirect = (redirect_E != nullptr);
    auto const K_per_eV    = PhysConst::q_e / PhysConst::kb;        // T[eV]*this = T[K]
    amrex::Real const Te_thresh_K = m_joule_redirect_Te_eV * K_per_eV;

    auto & mypc = warpx.GetPartContainer();

    // Loop over every charged ion species and accumulate its per-cell
    // contribution to S_e into T_e directly. Each species contributes
    //   dT_e_s = dt (gamma-1) * Z_s e^2 eta n_s |dV|^2 / k_B
    // (the n_e factor in nu_{s,e} cancels the 1/n_e from the T_e update).
    //
    // n_s is recovered from the species charge fraction rather than from
    // rho_fp_s/q_e directly: the per-species deposits are physical (volume-
    // scaled in radial geometries) but unfiltered and not boundary-treated,
    // while n_e comes from the fully processed total rho_fp used by the
    // E-solve. Taking
    //
    //   f_s = rho_fp_s / Sigma_t rho_fp_t   =   Z_s n_s / n_e   (unitless)
    //   n_s = f_s * n_e / Z_s
    //
    // keeps n_s consistent with that n_e in any dimensionality (numerator
    // and denominator of f_s share identical processing).
    auto const species_names = mypc.GetSpeciesNames();

    // Sigma_t rho_fp_t (physical per-species charge densities), used for the
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

        amrex::MultiFab const & rho_s =
            *warpx.m_fields.get("rho_fp_" + spec_name, lev);

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
                // = Z_s n_s / n_e (unitless; both sides physical and
                // identically processed). Then the per-species number
                // density: n_s = f_s * n_e / Z_s
                amrex::Real const rhos_val      = rhos_arr(i,j,k);
                amrex::Real const rhos_sum_val  = std::max(rhosum_arr(i,j,k), rho_floor);
                amrex::Real const f_s           = rhos_val / rhos_sum_val;
                amrex::Real const ns            = f_s * ne / Z_s;

                // |J| at the nodal grid (where Te lives), for the eta parser.
                auto const jx = ablastr::coarsen::sample::Interp(Jpx, Jx_stag, nodal, coarsen, i, j, k, 0);
                auto const jy = ablastr::coarsen::sample::Interp(Jpy, Jy_stag, nodal, coarsen, i, j, k, 0);
                auto const jz = ablastr::coarsen::sample::Interp(Jpz, Jz_stag, nodal, coarsen, i, j, k, 0);
                amrex::Real const Jmag = std::sqrt(jx*jx + jy*jy + jz*jz);

                // eta: same Ohm's-law parser the E-solve uses, evaluated
                // per cell. This makes the per-cell heat reduce to eta J^2
                // exactly in single species.
                amrex::Real const eta_s_eff = eta(rho_val, Jmag, t_new);

                // e-i relative drift = J_plasma/(e n_e), from the nodal plasma
                // current and n_e. Energy-consistent with the eta*J dissipation
                // in Ohm's law; reduces to eta*|J|^2 for a single species.
                amrex::Real const inv_ene = 1.0_rt / (PhysConst::q_e * ne);
                amrex::Real const dvx = jx * inv_ene;
                amrex::Real const dvy = jy * inv_ene;
                amrex::Real const dvz = jz * inv_ene;
                amrex::Real const dv2 = dvx*dvx + dvy*dvy + dvz*dvz;

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
                if (do_redirect && Te_arr(i,j,k) >= Te_thresh_K) {
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

    // Sigma_t rho_fp_t (physical per-species charge densities) -> species
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
    ablastr::fields::VectorField Ve =
        warpx.m_fields.get_alldirs(FieldType::hybrid_electron_velocity_fp, lev);

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
            amrex::Array4<amrex::Real const> const & Vex_arr  = Ve[0]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Vey_arr  = Ve[1]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Vez_arr  = Ve[2]->const_array(mfi);
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


void HybridPICModel::QDSMCFillElectronPressureFromTe (int const lev) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCFillElectronPressureFromTe()");

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
            QDSMCApplyIonHeating(lev, dt, redirect_active ? &ion_redirect_E : nullptr,
                                 m_include_temperature_relaxation ? &Ti_dep_by_species : nullptr);
        }

        // Step 7: emit P_e = n_e * k_B * T_e for the downstream Ohm's-law
        // solve, with the same boundary treatment the algebraic closure gets
        // in CalculateElectronPressure (grad P_e reads the ghost cells).
        QDSMCFillElectronPressureFromTe(lev);
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
                use_rkf45 = true;
            }
        }

        if (step_succeeded) {
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

    // Adjust the number of substeps for the next RKF45/RK4 half-step.
    // Jump up immediately when this half needed more attempts; otherwise
    // slowly relax toward target = 2*n_attempts.
    // Blend on the half-step counts M = m_substeps/2 and N = n_attempts via
    // integer arithmetic: relaxed = 2*((19*M + N)/20). That is exactly the
    // 95/5 blend, stays even, holds when N == M, and actually decays when
    // N < M (e.g. m=40, n_attempts=10 → 38 → … → 20). Floating-point
    // 0.95*M+0.05*N can undershoot M slightly so floor would leak even at
    // equilibrium.
    {
        const int target = 2 * n_attempts;
        if (m_substeps < target) {
            m_substeps = target;
        } else {
            const int M = m_substeps / 2;
            const int N = n_attempts;
            const int relaxed = 2 * ((19 * M + N) / 20);
            m_substeps = std::max(relaxed, 2);
        }
        // Stay within the abort budget so the controller cannot request more
        // substeps than max_substep_attempts allows.
        if (m_substeps > m_max_substep_attempts) {
            m_substeps = m_max_substep_attempts - (m_max_substep_attempts % 2);
            m_substeps = std::max(m_substeps, 2);
        }
    }

    if (WarpX::GetInstance().Verbose()) {
        amrex::Print() << "B-field update "
            << (subcycling_half == SubcyclingHalf::FirstHalf ? "1st" : "2nd") << " half"
            << ": " << n_accepted << " accepted, "
            << (n_attempts - n_accepted) << " rejected substeps"
            << " (dt_sub_final/dt_half = " << dt_sub / dt_half
            << ", m_substeps = " << m_substeps << ")\n";
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
    // Calculate the E-field from Ohm's law
    HybridPICSolveE(Efield, Jfield, Bfield, rhofield, eb_update_E, true);
    // Call FillBoundary if a collocated grid is used
    if (Bz_IndexType[0] == Ez_IndexType[0]) {
        warpx.FillBoundaryE(ng, nodal_sync);
    }

    // Push forward the B-field using Faraday's law
    warpx.EvolveB(dt, subcycling_half, t_old);
    warpx.FillBoundaryB(ng, nodal_sync);
}
