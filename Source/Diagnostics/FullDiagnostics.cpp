#include "FullDiagnostics.H"

#include "ComputeDiagFunctors/CellCenterFunctor.H"
#include "ComputeDiagFunctors/DivBFunctor.H"
#include "ComputeDiagFunctors/DivEFunctor.H"
#include "ComputeDiagFunctors/JFunctor.H"
#include "ComputeDiagFunctors/JdispFunctor.H"
#include "ComputeDiagFunctors/PartPerCellFunctor.H"
#include "ComputeDiagFunctors/PartPerGridFunctor.H"
#include "ComputeDiagFunctors/ParticleReductionFunctor.H"
#include "ComputeDiagFunctors/TemperatureFunctor.H"
#include "ComputeDiagFunctors/RhoFunctor.H"
#include "Diagnostics/Diagnostics.H"
#include "Diagnostics/ParticleDiag/ParticleDiag.H"
#include "Fields.H"
#include "FlushFormats/FlushFormat.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/Algorithms/IsIn.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MakeType.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

using namespace amrex::literals;
using warpx::fields::FieldType;

FullDiagnostics::FullDiagnostics (int i, const std::string& name, DiagTypes diag_type):
    Diagnostics{i, name, diag_type},
    m_solver_deposits_current{
        (WarpX::electromagnetic_solver_id != ElectromagneticSolverAlgo::None) ||
        (WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameElectroMagnetostatic)}
{
    ReadParameters();
    BackwardCompatibility();
}

void
FullDiagnostics::DerivedInitData() {
    if (m_diag_type == DiagTypes::TimeAveraged) {
        auto & warpx = WarpX::GetInstance();
        if (m_time_average_mode == TimeAverageType::Dynamic) {

            // already checked in ReadParameters that only one of the parameters is set
            // calculate the other averaging period parameter from the other one, respectively
            if (m_average_period_steps > 0) {
                m_average_period_time = m_average_period_steps * warpx.getdt(0);
            } else if (m_average_period_time > 0) {
                m_average_period_steps = static_cast<int> (std::round(m_average_period_time / warpx.getdt(0)));
            }
            amrex::Print() << Utils::TextMsg::Info(
                "Initializing TimeAveragedDiagnostics " + m_diag_name
                + " with an averaging period of " + std::to_string(m_average_period_steps) + " steps"
                );
        }
    }
}

void
FullDiagnostics::InitializeParticleBuffer ()
{
    // When particle buffers are included, the vector of particle containers
    // must be allocated in this function.
    // Initialize data in the base class Diagnostics
    auto & warpx = WarpX::GetInstance();

    const MultiParticleContainer& mpc = warpx.GetPartContainer();
    // If not specified, dump all species
    if (m_output_species_names.empty()) {
        if (m_format == "checkpoint") {
            m_output_species_names = mpc.GetSpeciesAndLasersNames();
        } else {
            m_output_species_names = mpc.GetSpeciesNames();
        }
    }
    // Initialize one ParticleDiag per species requested
    for (int i_buffer = 0; i_buffer < m_num_buffers; ++i_buffer) {
        for (auto const& species : m_output_species_names){
            const int idx = mpc.getSpeciesID(species);
            m_output_species[i_buffer].push_back(ParticleDiag(m_diag_name, species, mpc.GetParticleContainerPtr(idx)));
        }
    }
}

void
FullDiagnostics::ReadParameters ()
{
    // Read list of full diagnostics fields requested by the user.
    const bool checkpoint_compatibility = BaseReadParameters();
    const amrex::ParmParse pp_diag_name(m_diag_name);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_format == "plotfile" || m_format == "openpmd" ||
        m_format == "checkpoint" || m_format == "ascent" ||
        m_format == "sensei" || m_format == "catalyst",
        "<diag>.format must be plotfile or openpmd or checkpoint or ascent or catalyst or sensei");
    std::vector<std::string> intervals_string_vec = {"0"};
    pp_diag_name.getarr("intervals", intervals_string_vec);
    m_intervals = utils::parser::IntervalsParser(intervals_string_vec);
    const bool plot_raw_fields_specified = pp_diag_name.query("plot_raw_fields", m_plot_raw_fields);
    const bool plot_raw_fields_guards_specified = pp_diag_name.query("plot_raw_fields_guards", m_plot_raw_fields_guards);
    const bool raw_specified = plot_raw_fields_specified || plot_raw_fields_guards_specified;

    if (m_diag_type == DiagTypes::TimeAveraged) {
        std::string m_time_average_mode_str = "none";
        /** Whether the diagnostics are averaging data over time or not
         * Valid options are "fixed_start" and "dynamic_start".
         */
        pp_diag_name.get("time_average_mode", m_time_average_mode_str);

        const amrex::ParmParse pp_warpx("warpx");
        std::vector<std::string> dt_interval_vec = {"-1"};
        const bool timestep_may_vary = pp_warpx.queryarr("dt_update_interval", dt_interval_vec);
        amrex::Print() << Utils::TextMsg::Warn("Time step varies?" + std::to_string(timestep_may_vary));
        if (timestep_may_vary) {
            WARPX_ABORT_WITH_MESSAGE(
                    "Time-averaged diagnostics (encountered in: "
                    + m_diag_name + ") are currently not supported with adaptive time-stepping"
                    );
        }

        if (m_time_average_mode_str == "fixed_start") {
            m_time_average_mode = TimeAverageType::Static;
            } else if (m_time_average_mode_str == "dynamic_start") {
            m_time_average_mode = TimeAverageType::Dynamic;
            } else if (m_time_average_mode_str == "none") {
            m_time_average_mode = TimeAverageType::None;
            } else {
            WARPX_ABORT_WITH_MESSAGE(
                    "Unknown time averaging mode. Valid entries are: none, fixed_start, dynamic_start"
                    );
        }

        const bool averaging_period_steps_specified = pp_diag_name.query(
                "average_period_steps", m_average_period_steps
        );
        const bool averaging_period_time_specified = pp_diag_name.queryWithParser(
                "average_period_time", m_average_period_time
        );

        if (m_time_average_mode == TimeAverageType::Static) {
            // This fails if users do not specify a start.
            pp_diag_name.get("average_start_step", m_average_start_step);
            if (m_average_start_step == 0) {
                WARPX_ABORT_WITH_MESSAGE(
                    "Static-start time-averaged diagnostic " + m_diag_name + " requires a positive (non-zero) value "
                    "for the 'average_start_step' parameter."
                );
            }

            if (averaging_period_time_specified || averaging_period_steps_specified) {
                const std::string period_spec_warn_msg = "An averaging period was specified for the 'static_start' averaging mode " \
                                                         "but will be IGNORED. Averaging will be performed between step " \
                                                         + std::to_string(m_average_start_step) \
                                                         + " and the specified intervals.";
                ablastr::warn_manager::WMRecordWarning(
                        "Diagnostics",
                        period_spec_warn_msg,
                        ablastr::warn_manager::WarnPriority::medium
                );
            }

        }

        if (m_time_average_mode == TimeAverageType::Dynamic) {
            // one of the two averaging period options must be set but neither none nor both
            if (
                    (averaging_period_steps_specified && averaging_period_time_specified)
                    || !(averaging_period_steps_specified || averaging_period_time_specified)
                    ) {
                WARPX_ABORT_WITH_MESSAGE("Please specify either 'average_period_steps' or 'average_period_time', not both.");
            }

            int unused_start_step = -1;
            const bool averaging_start_on_dynamic_period_specified = pp_diag_name.query("average_start_step", unused_start_step);
            if (averaging_start_on_dynamic_period_specified) {
                const std::string start_spec_warn_msg = "An averaging start step was specified for the 'dynamic_start'" \
                                                         "time-averaged diagnostic " + m_diag_name + " but will be IGNORED. " \
                                                         "Averaging will begin with the first averaging period.";
                ablastr::warn_manager::WMRecordWarning(
                        "Diagnostics",
                        start_spec_warn_msg,
                        ablastr::warn_manager::WarnPriority::medium
                );
            }
        }
    }


#ifdef WARPX_DIM_RZ
    pp_diag_name.query("dump_rz_modes", m_dump_rz_modes);
#else
    amrex::ignore_unused(m_dump_rz_modes);
#endif

    if (m_format == "checkpoint"){
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            raw_specified == false &&
            checkpoint_compatibility == true,
            "For a checkpoint output, cannot specify these parameters as all data must be dumped "
            "to file for a restart");
    }
    // Number of buffers = 1 for FullDiagnostics.
    // It is used to allocate the number of output multi-level MultiFab, m_mf_output
    m_num_buffers = 1;
}

void
FullDiagnostics::BackwardCompatibility ()
{
    const amrex::ParmParse pp_diag_name(m_diag_name);
    std::vector<std::string> backward_strings;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_diag_name.queryarr("period", backward_strings),
        "<diag_name>.period is no longer a valid option. "
        "Please use the renamed option <diag_name>.intervals instead."
    );
}

void
FullDiagnostics::Flush ( int i_buffer, bool /* force_flush */ )
{
    // This function should be moved to Diagnostics when plotfiles/openpmd format
    // is supported for BackTransformed Diagnostics, in BTDiagnostics class.
    auto & warpx = WarpX::GetInstance();

    // Get the time step on coarsest level.
    const int step = warpx.getistep(0);
    // For time-averaged diagnostics, we still write out an instantaneous diagnostic on step 0
    // to accommodate a user workflow that only uses that type of diagnostic.
    // This allows for quicker turnaround in setup by avoiding having to set an additional instantaneous diagnostic.
    if (m_diag_type == DiagTypes::TimeAveraged && step > 0) {
        if (m_time_average_mode == TimeAverageType::Static || m_time_average_mode == TimeAverageType::Dynamic) {
            // Loop over the output levels and divide by the number of steps in the averaging period
            for (int lev = 0; lev < nlev_output; ++lev) {
                m_sum_mf_output.at(i_buffer).at(lev).mult(1._rt/static_cast<amrex::Real>(m_average_period_steps));
            }

            m_flush_format->WriteToFile(
                    m_varnames, m_sum_mf_output.at(i_buffer), m_geom_output.at(i_buffer), warpx.getistep(),
                    warpx.gett_new(0),
                    m_output_species.at(i_buffer), nlev_output, m_file_prefix,
                    m_file_min_digits, m_plot_raw_fields, m_plot_raw_fields_guards,
                    m_verbose);

            // Reset the values in the dynamic start time-averaged diagnostics after flush
            if (m_time_average_mode == TimeAverageType::Dynamic) {
                for (int lev = 0; lev < nlev_output; ++lev) {
                    m_sum_mf_output.at(i_buffer).at(lev).setVal(0.);
                }
            }
        }
    } else {
        if (m_diag_type == DiagTypes::TimeAveraged && step == 0) {
            // For both dynamic_start and fixed_start at step 0 we prepare an instantaneous output
            amrex::Print() << Utils::TextMsg::Info("Time-averaged diagnostic " + m_diag_name
                                                   + " is preparing an instantaneous output during step " + std::to_string(step));
        }

        m_flush_format->WriteToFile(
            m_varnames, m_mf_output.at(i_buffer), m_geom_output.at(i_buffer), warpx.getistep(),
            warpx.gett_new(0),
            m_output_species.at(i_buffer), nlev_output, m_file_prefix,
            m_file_min_digits, m_plot_raw_fields, m_plot_raw_fields_guards,
            m_verbose);
    }

    FlushRaw();
}

void
FullDiagnostics::FlushRaw () {}


bool
FullDiagnostics::DoDump (int step, int /*i_buffer*/, bool force_flush)
{
    if (m_already_done) { return false; }
    if ( force_flush || (m_intervals.contains(step+1)) ){
        m_already_done = true;
        return true;
    }
    return false;
}

bool
FullDiagnostics::DoComputeAndPack (int step, bool force_flush)
{
    // Start averaging at output step (from diag.intervals) - period + 1
    bool in_averaging_period = false;
    if (m_diag_type == DiagTypes::TimeAveraged) {

        if (step > 0) {

            if (m_time_average_mode == TimeAverageType::Dynamic) {
                m_average_start_step = m_intervals.nextContains(step) - m_average_period_steps;
                // check that the periods do not overlap and that the start step is not negative
                if (m_average_start_step > 0) {
                    // The start step cannot be on an interval step because then we would begin a new period and also output the old one
                    if (m_average_start_step < m_intervals.previousContains(step)) {
                        WARPX_ABORT_WITH_MESSAGE(
                                "Averaging periods may not overlap within a single diagnostic. "
                                "Please create a second diagnostic for overlapping time averaging periods "
                                "and account for the increased memory consumption."
                        );
                    }
                } else {
                    WARPX_ABORT_WITH_MESSAGE(
                            "The step to begin time averaging ("
                            + std::to_string(m_average_start_step)
                            + ") for diagnostic " + m_diag_name + " must be a positive number."
                    );
                }

                if (step >= m_average_start_step && step <= m_intervals.nextContains(step)) {
                    in_averaging_period = true;

                    if (m_time_average_mode == TimeAverageType::Static) {
                        // Update time averaging period to current step
                        m_average_period_steps = step - m_average_start_step;
                    }
                }
                // Print information on when time-averaging is active
                if ((m_verbose > 1) && in_averaging_period) {
                    if (step == m_average_start_step) {
                        amrex::Print() << Utils::TextMsg::Info(
                                "Begin time averaging for " + m_diag_name + " and output at step "
                                + std::to_string(m_intervals.nextContains(step))
                        );
                    } else {
                        amrex::Print()
                                << Utils::TextMsg::Info(
                                        "Time-averaging during this step for diagnostic: " + m_diag_name);
                    }
                }
            }
        }
    }
    // Data must be computed and packed for full diagnostics
    // whenever the data needs to be flushed.
    return (force_flush || m_intervals.contains(step+1) || in_averaging_period);

}

void
FullDiagnostics::InitializeFieldFunctorsRZopenPMD (int lev)
{
#ifdef WARPX_DIM_RZ
    using ablastr::fields::Direction;

    auto & warpx = WarpX::GetInstance();
    const int ncomp_multimodefab = warpx.m_fields.get(FieldType::Efield_aux, Direction{0}, 0)->nComp();
    // Make sure all multifabs have the same number of components
    for (int dim=0; dim<3; dim++){
        AMREX_ALWAYS_ASSERT(
            warpx.m_fields.get(FieldType::Efield_aux, Direction{dim}, lev)->nComp() == ncomp_multimodefab );
        AMREX_ALWAYS_ASSERT(
            warpx.m_fields.get(FieldType::Bfield_aux, Direction{dim}, lev)->nComp() == ncomp_multimodefab );
        AMREX_ALWAYS_ASSERT(
            warpx.m_fields.get(FieldType::current_fp, Direction{dim}, lev)->nComp() == ncomp_multimodefab );
    }

    // Species index to loop over species that dump rho per species
    int i = 0;
    // Species index to loop over species that dump temperature per species
    int i_T_species = 0;
    const int ncomp = ncomp_multimodefab;
    // This function is called multiple times, for different values of `lev`
    // but the `varnames` need only be updated once.

    const bool update_varnames = (lev==0);
    if (update_varnames) {
        m_varnames.clear();
        const auto n_rz = ncomp * static_cast<int>(m_varnames.size());
        m_varnames.reserve(n_rz);
    }

    // Add functors for average particle data for each species
    const auto nvar = static_cast<int>(m_varnames_fields.size());
    const auto nspec = static_cast<int>(m_pfield_species.size());
    const auto ntot = static_cast<int>(nvar + m_pfield_varnames.size() * nspec);

    // Reset field functors
    m_all_field_functors[lev].clear();
    m_all_field_functors[lev].resize(ntot);

    // Boolean flag for whether the current density should be deposited before
    // diagnostic output
    bool deposit_current = !m_solver_deposits_current;

    std::vector<std::string> field_names = {"r", "t", "z"};

    // Fill vector of functors for all components except individual cylindrical modes.
    const auto m_varname_fields_size = static_cast<int>(m_varnames_fields.size());
    for (int comp=0; comp<m_varname_fields_size; comp++){
        for (int idir=0; idir < 3; idir++) {
            if        ( m_varnames_fields[comp] == "E"+field_names[idir] ){
                m_all_field_functors[lev][comp] = std::make_unique<CellCenterFunctor>(warpx.m_fields.get(FieldType::Efield_aux,
                                                            Direction{idir}, lev), lev, m_crse_ratio, false, ncomp);
                if (update_varnames) {
                    AddRZModesToOutputNames(std::string("E"+field_names[idir]), ncomp);
                }
            } else if ( m_varnames_fields[comp] == "B"+field_names[idir] ){
                m_all_field_functors[lev][comp] = std::make_unique<CellCenterFunctor>(warpx.m_fields.get(FieldType::Bfield_aux,
                                                            Direction{idir}, lev), lev, m_crse_ratio, false, ncomp);
                if (update_varnames) {
                    AddRZModesToOutputNames(std::string("B"+field_names[idir]), ncomp);
                }
            } else if ( m_varnames_fields[comp] == "j"+field_names[idir] ){
                m_all_field_functors[lev][comp] = std::make_unique<JFunctor>(idir, lev, m_crse_ratio,
                                                            false, deposit_current, ncomp);
                deposit_current = false;
                if (update_varnames) {
                    AddRZModesToOutputNames(std::string("j"+field_names[idir]), ncomp);
                }
            } else if ( m_varnames_fields[comp] == "j"+field_names[idir]+"_displacement" ){
                m_all_field_functors[lev][comp] = std::make_unique<JdispFunctor>(idir, lev, m_crse_ratio,
                                                            false, ncomp);
                if (update_varnames) {
                    AddRZModesToOutputNames(std::string("j"+field_names[idir]+"_displacement"), ncomp);
                }
            }
        }
        // Check if comp was found above
        if (m_all_field_functors[lev][comp]) {continue;}

        if ( m_varnames_fields[comp] == "rho" ){
            // Initialize rho functor to dump total rho
            m_all_field_functors[lev][comp] = std::make_unique<RhoFunctor>(lev, m_crse_ratio, true, -1,
                                                        false, ncomp);
            if (update_varnames) {
                AddRZModesToOutputNames(std::string("rho"), ncomp);
            }
        } else if ( m_varnames_fields[comp].rfind("rho_", 0) == 0 ){
            // Initialize rho functor to dump rho per species
            m_all_field_functors[lev][comp] = std::make_unique<RhoFunctor>(lev, m_crse_ratio, true, m_rho_per_species_index[i],
                                                        false, ncomp);
            if (update_varnames) {
                AddRZModesToOutputNames(std::string("rho_") + m_all_species_names[m_rho_per_species_index[i]], ncomp);
            }
            i++;
        } else if ( m_varnames_fields[comp].rfind("T_", 0) == 0 ){
            // Initialize temperature functor to dump temperature per species
            m_all_field_functors[lev][comp] = std::make_unique<TemperatureFunctor>(lev, m_crse_ratio, m_T_per_species_index[i_T_species]);
            if (update_varnames) {
                AddRZModesToOutputNames(std::string("T_") + m_all_species_names[m_T_per_species_index[i_T_species]], ncomp);
            }
            i_T_species++;
        } else if ( m_varnames_fields[comp] == "F" ){
            m_all_field_functors[lev][comp] = std::make_unique<CellCenterFunctor>(warpx.m_fields.get(FieldType::F_fp, lev), lev, m_crse_ratio,
                                                        false, ncomp);
            if (update_varnames) {
                AddRZModesToOutputNames(std::string("F"), ncomp);
            }
        } else if ( m_varnames_fields[comp] == "G" ){
            m_all_field_functors[lev][comp] = std::make_unique<CellCenterFunctor>( warpx.m_fields.get(FieldType::G_fp, lev), lev, m_crse_ratio,
                                                        false, ncomp);
            if (update_varnames) {
                AddRZModesToOutputNames(std::string("G"), ncomp);
            }
        } else if ( m_varnames_fields[comp] == "phi" ){
            m_all_field_functors[lev][comp] = std::make_unique<CellCenterFunctor>(warpx.m_fields.get(FieldType::phi_fp, lev), lev, m_crse_ratio,
                                                        false, ncomp);
            if (update_varnames) {
                AddRZModesToOutputNames(std::string("phi"), ncomp);
            }
        } else if ( m_varnames_fields[comp] == "part_per_cell" ){
            m_all_field_functors[lev][comp] = std::make_unique<PartPerCellFunctor>(nullptr, lev, m_crse_ratio);
            if (update_varnames) {
                m_varnames.push_back(std::string("part_per_cell"));
            }
        } else if ( m_varnames_fields[comp] == "part_per_grid" ){
            m_all_field_functors[lev][comp] = std::make_unique<PartPerGridFunctor>(nullptr, lev, m_crse_ratio);
            if (update_varnames) {
                m_varnames.push_back(std::string("part_per_grid"));
            }
        } else if ( m_varnames_fields[comp] == "divB" ){
            m_all_field_functors[lev][comp] = std::make_unique<DivBFunctor>(
                warpx.m_fields.get_alldirs(FieldType::Bfield_aux, lev),
                lev, m_crse_ratio, false, ncomp);
            if (update_varnames) {
                AddRZModesToOutputNames(std::string("divB"), ncomp);
            }
        } else if ( m_varnames_fields[comp] == "divE" ){
            m_all_field_functors[lev][comp] = std::make_unique<DivEFunctor>(
                warpx.m_fields.get_alldirs(FieldType::Efield_aux, lev),
                lev, m_crse_ratio, false, ncomp);
            if (update_varnames) {
                AddRZModesToOutputNames(std::string("divE"), ncomp);
            }
        }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "Error: " + m_varnames_fields[comp] + " is not a known field output type in RZ geometry");
        }
    }

    // Generate field functors for every particle field diagnostic for every species in m_pfield_species.
    // The names of the diagnostics are output in the `[varname]_[species]` format.
    for (int pcomp=0; pcomp<int(m_pfield_varnames.size()); pcomp++) {
        for (int ispec=0; ispec<int(m_pfield_species.size()); ispec++) {
            m_all_field_functors[lev][nvar + pcomp * nspec + ispec] = std::make_unique<ParticleReductionFunctor>(nullptr,
                    lev, m_crse_ratio, m_pfield_strings[pcomp], m_pfield_species_index[ispec], m_pfield_do_average[pcomp],
                    m_pfield_dofilter[pcomp], m_pfield_filter_strings[pcomp]);
            if (update_varnames) {
                AddRZModesToOutputNames(std::string(m_pfield_varnames[pcomp]) + "_" + std::string(m_pfield_species[ispec]), ncomp);
            }
        }
    }

    // Sum the number of components in input vector m_all_field_functors
    // and check that it corresponds to the number of components in m_varnames
    // and m_mf_output
    int ncomp_from_src = 0;
    for (int jj=0; jj<m_all_field_functors[0].size(); jj++){
        ncomp_from_src += m_all_field_functors[lev][jj]->nComp();
    }

    AMREX_ALWAYS_ASSERT( ncomp_from_src == m_varnames.size() );
#else
    amrex::ignore_unused(lev);
#endif
}

void
FullDiagnostics::AddRZModesToDiags (int lev)
{
#ifdef WARPX_DIM_RZ
    using ablastr::fields::Direction;

    if (!m_dump_rz_modes) { return; }

    auto & warpx = WarpX::GetInstance();
    const int ncomp_multimodefab = warpx.m_fields.get(FieldType::Efield_aux, Direction{0}, 0)->nComp();
    // Make sure all multifabs have the same number of components
    for (int dim=0; dim<3; dim++){
        AMREX_ALWAYS_ASSERT(
            warpx.m_fields.get(FieldType::Efield_aux, Direction{dim}, lev)->nComp() == ncomp_multimodefab );
        AMREX_ALWAYS_ASSERT(
            warpx.m_fields.get(FieldType::Bfield_aux, Direction{dim}, lev)->nComp() == ncomp_multimodefab );
        AMREX_ALWAYS_ASSERT(
            warpx.m_fields.get(FieldType::current_fp, Direction{dim}, lev)->nComp() == ncomp_multimodefab );
    }

    // Check if divE is requested
    // If so, all components will be written out
    bool divE_requested = false;
    for (int comp = 0; comp < m_varnames.size(); comp++) {
        if ( m_varnames[comp] == "divE" ) {
            divE_requested = true;
        }
    }

    // If rho is requested, all components will be written out
    const bool rho_requested = utils::algorithms::is_in( m_varnames, "rho" );

    // Boolean flag for whether the current density should be deposited before
    // diagnostic output
    bool deposit_current = !m_solver_deposits_current;

    const std::array<std::string, 3> coord {"r", "theta", "z"};

    // Er, Etheta, Ez, Br, Btheta, Bz, jr, jtheta, jz
    // Each of them being a multi-component multifab
    int n_new_fields = 9;
    if (divE_requested) {
        n_new_fields += 1;
    }
    if (rho_requested) {
        n_new_fields += 1;
    }
    m_all_field_functors[lev].reserve( m_all_field_functors[0].size() + n_new_fields );
    // E
    for (int dim=0; dim<3; dim++){
        // 3 components, r theta z
        m_all_field_functors[lev].push_back(std::make_unique<CellCenterFunctor>(
                warpx.m_fields.get(FieldType::Efield_aux, Direction{dim}, lev), lev,
                    m_crse_ratio, false, ncomp_multimodefab));
        AddRZModesToOutputNames(std::string("E") + coord[dim],
                warpx.m_fields.get(FieldType::Efield_aux, Direction{0}, 0)->nComp());
    }
    // B
    for (int dim=0; dim<3; dim++){
        // 3 components, r theta z
        m_all_field_functors[lev].push_back(std::make_unique<CellCenterFunctor>(
                warpx.m_fields.get(FieldType::Bfield_aux, Direction{dim}, lev), lev,
                    m_crse_ratio, false, ncomp_multimodefab));
        AddRZModesToOutputNames(std::string("B") + coord[dim],
                warpx.m_fields.get(FieldType::Bfield_aux, Direction{0}, 0)->nComp());
    }
    // j
    for (int dim=0; dim<3; dim++){
        // 3 components, r theta z
        m_all_field_functors[lev].push_back(std::make_unique<JFunctor>(
            dim, lev, m_crse_ratio, false, deposit_current, ncomp_multimodefab));
        deposit_current = false;
        AddRZModesToOutputNames(std::string("J") + coord[dim],
                warpx.m_fields.get(FieldType::current_fp,Direction{0},0)->nComp());
    }
    // divE
    if (divE_requested) {
        m_all_field_functors[lev].push_back(std::make_unique<DivEFunctor>(
            warpx.m_fields.get_alldirs(FieldType::Efield_aux, lev),
            lev, m_crse_ratio, false, ncomp_multimodefab));
        AddRZModesToOutputNames(std::string("divE"), ncomp_multimodefab);
    }
    // rho
    if (rho_requested) {
        m_all_field_functors[lev].push_back(std::make_unique<RhoFunctor>(
            lev, m_crse_ratio, true, -1, false, ncomp_multimodefab));
        AddRZModesToOutputNames(std::string("rho"), ncomp_multimodefab);
    }
    // Sum the number of components in input vector m_all_field_functors
    // and check that it corresponds to the number of components in m_varnames
    // and m_mf_output
    int ncomp_from_src = 0;
    for (int i=0; i<m_all_field_functors[0].size(); i++){
        ncomp_from_src += m_all_field_functors[lev][i]->nComp();
    }
    AMREX_ALWAYS_ASSERT( ncomp_from_src == m_varnames.size() );
#else
    amrex::ignore_unused(lev);
#endif
}

void
FullDiagnostics::AddRZModesToOutputNames (const std::string& field, int ncomp){
#ifdef WARPX_DIM_RZ
    // In cylindrical geometry, real and imag part of each mode are also
    // dumped to file separately, so they need to be added to m_varnames
    m_varnames.push_back( field + "_0_real" );
    for (int ic=1 ; ic < (ncomp+1)/2 ; ic += 1) {
        m_varnames.push_back( field + "_" + std::to_string(ic) + "_real" );
        m_varnames.push_back( field + "_" + std::to_string(ic) + "_imag" );
    }
#else
    amrex::ignore_unused(field, ncomp);
#endif
}


void
FullDiagnostics::InitializeBufferData (int i_buffer, int lev, bool restart ) {
    amrex::ignore_unused(restart);
    auto & warpx = WarpX::GetInstance();
    amrex::RealBox diag_dom;
    bool use_warpxba = true;
    const amrex::IntVect blockingFactor = warpx.blockingFactor( lev );

    // Default BoxArray and DistributionMap for initializing the output MultiFab, m_mf_output.
    amrex::BoxArray ba = warpx.boxArray(lev);
    amrex::DistributionMapping dmap = warpx.DistributionMap(lev);
    // Check if warpx BoxArray is coarsenable.
    if (warpx.get_numprocs() == 0)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE (
            ba.coarsenable(m_crse_ratio), "Invalid coarsening ratio for field diagnostics."
            "Must be an integer divisor of the blocking factor.");
    }
    else
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE (
            ba.coarsenable(m_crse_ratio), "Invalid coarsening ratio for field diagnostics."
            "The total number of cells must be a multiple of the coarsening ratio multiplied by numprocs.");
    }

    // Find if user-defined physical dimensions are different from the simulation domain.
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
         // To ensure that the diagnostic lo and hi are within the domain defined at level, lev.
        diag_dom.setLo(idim, std::max(m_lo[idim],warpx.Geom(lev).ProbLo(idim)) );
        diag_dom.setHi(idim, std::min(m_hi[idim],warpx.Geom(lev).ProbHi(idim)) );
        if ( std::fabs(warpx.Geom(lev).ProbLo(idim) - diag_dom.lo(idim))
                               >  warpx.Geom(lev).CellSize(idim) ) {
             use_warpxba = false;
        }
        if ( std::fabs(warpx.Geom(lev).ProbHi(idim) - diag_dom.hi(idim))
                               > warpx.Geom(lev).CellSize(idim) ) {
             use_warpxba = false;
        }

        // User-defined value for coarsening should be an integer divisor of
        // blocking factor at level, lev. This assert is not relevant and thus
        // removed if warpx.numprocs is used for the domain decomposition.
        if (warpx.get_numprocs() == 0)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( blockingFactor[idim] % m_crse_ratio[idim]==0,
                           " coarsening ratio must be integer divisor of blocking factor");
        }
    }

    if (!use_warpxba) {
        // Following are the steps to compute the lo and hi index corresponding to user-defined
        // m_lo and m_hi using the same resolution as the simulation at level, lev.
        amrex::IntVect lo(0);
        amrex::IntVect hi(1);
        for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
            // lo index with same cell-size as simulation at level, lev.
            lo[idim] = std::max( static_cast<int>( std::floor (
                          ( diag_dom.lo(idim) - warpx.Geom(lev).ProbLo(idim)) /
                            warpx.Geom(lev).CellSize(idim)) ), 0 );
            // hi index with same cell-size as simulation at level, lev.
            hi[idim] = std::max( static_cast<int> ( std::ceil (
                          ( diag_dom.hi(idim) - warpx.Geom(lev).ProbLo(idim)) /
                            warpx.Geom(lev).CellSize(idim) ) ), 0) - 1 ;
            // if hi<=lo, then hi = lo + 1, to ensure one cell in that dimension
            if ( hi[idim] <= lo[idim] ) {
                 hi[idim]  = lo[idim] + 1;
                 WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    m_crse_ratio[idim]==1, "coarsening ratio in reduced dimension must be 1."
                 );
            }
        }

        // Box for the output MultiFab corresponding to the user-defined physical co-ordinates at lev.
        const amrex::Box diag_box( lo, hi );
        // Define box array
        amrex::BoxArray diag_ba;
        diag_ba.define(diag_box);
        ba = diag_ba.maxSize( warpx.maxGridSize( lev ) );
        // At this point in the code, the BoxArray, ba, is defined with the same index space and
        // resolution as the simulation, at level, lev.
        // Coarsen and refine so that the new BoxArray is coarsenable.
        ba.coarsen(m_crse_ratio).refine(m_crse_ratio);

        // Update the physical co-ordinates m_lo and m_hi using the final index values
        // from the coarsenable, cell-centered BoxArray, ba.
        for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            diag_dom.setLo( idim, warpx.Geom(lev).ProbLo(idim) +
                ba.getCellCenteredBox(0).smallEnd(idim) * warpx.Geom(lev).CellSize(idim));
            diag_dom.setHi( idim, warpx.Geom(lev).ProbLo(idim) +
                (ba.getCellCenteredBox( static_cast<int>(ba.size())-1 ).bigEnd(idim) + 1) * warpx.Geom(lev).CellSize(idim));
        }

    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_crse_ratio.min() > 0, "Coarsening ratio must be non-zero.");
    // The BoxArray is coarsened based on the user-defined coarsening ratio.
    ba.coarsen(m_crse_ratio);
    // Generate a new distribution map if the physical m_lo and m_hi for the output
    // is different from the lo and hi physical co-ordinates of the simulation domain.
    if (!use_warpxba) { dmap = amrex::DistributionMapping{ba}; }
    // Allocate output MultiFab for diagnostics. The data will be stored at cell-centers.
    const int ngrow = (m_format == "sensei" || m_format == "ascent") ? 1 : 0;
    int const ncomp = static_cast<int>(m_varnames.size());
    m_mf_output[i_buffer][lev] = amrex::MultiFab(ba, dmap, ncomp, ngrow);

    if (m_diag_type == DiagTypes::TimeAveraged) {
        // Allocate MultiFab for cell-centered field output accumulation. The data will be averaged before flushing.
        m_sum_mf_output[i_buffer][lev] = amrex::MultiFab(ba, dmap, ncomp, ngrow);
        // Initialize to zero because we add data.
        m_sum_mf_output[i_buffer][lev].setVal(0.);
    }

    if (lev == 0) {
        // The extent of the domain covered by the diag multifab, m_mf_output
        //default non-periodic geometry for diags
        amrex::Vector<int> diag_periodicity(AMREX_SPACEDIM,0);
        // Box covering the extent of the user-defined diagnostic domain
        const amrex::Box domain = ba.minimalBox();
        // define geom object
        m_geom_output[i_buffer][lev].define( domain, &diag_dom,
                                             amrex::CoordSys::cartesian,
                                             diag_periodicity.data() );
    } else if (lev > 0) {
        // Take the geom object of previous level and refine it.
        m_geom_output[i_buffer][lev] = amrex::refine( m_geom_output[i_buffer][lev-1],
                                                      WarpX::RefRatio(lev-1) );
    }
}


void
FullDiagnostics::InitializeFieldFunctors (int lev)
{
#ifdef WARPX_DIM_RZ
    // For RZ, with openPMD, we need a special initialization instead
    if (m_format == "openpmd") {
        InitializeFieldFunctorsRZopenPMD(lev);
        return; // We skip the rest of this function
    }
#endif

    auto & warpx = WarpX::GetInstance();

    // Clear any pre-existing vector to release stored data.
    m_all_field_functors[lev].clear();

    // Species index to loop over species that dump rho per species
    int i = 0;

    // Species index to loop over species that dump temperature per species
    int i_T_species = 0;

    const auto nvar = static_cast<int>(m_varnames_fields.size());
    const auto nspec = static_cast<int>(m_pfield_species.size());
    const auto ntot = static_cast<int>(nvar + m_pfield_varnames.size() * nspec);

    // Boolean flag for whether the current density should be deposited before
    // diagnostic output
    bool deposit_current = !m_solver_deposits_current;

    using ablastr::fields::Direction;

#if defined(WARPX_DIM_RZ)
    std::vector<std::string> field_names = {"r", "t", "z"};
#else
    std::vector<std::string> field_names = {"x", "y", "z"};
#endif

    m_all_field_functors[lev].resize(ntot);
    // Fill vector of functors for all components except individual cylindrical modes.
    for (int comp=0; comp<nvar; comp++){
        for (int idir=0; idir < 3; idir++) {
            if        ( m_varnames[comp] == "E"+field_names[idir] ){
                m_all_field_functors[lev][comp] = std::make_unique<CellCenterFunctor>(warpx.m_fields.get(FieldType::Efield_aux, Direction{idir}, lev), lev, m_crse_ratio);
            } else if ( m_varnames[comp] == "B"+field_names[idir] ){
                m_all_field_functors[lev][comp] = std::make_unique<CellCenterFunctor>(warpx.m_fields.get(FieldType::Bfield_aux, Direction{idir}, lev), lev, m_crse_ratio);
            } else if ( m_varnames[comp] == "j"+field_names[idir] ){
                m_all_field_functors[lev][comp] = std::make_unique<JFunctor>(idir, lev, m_crse_ratio, true, deposit_current);
                deposit_current = false;
            } else if ( m_varnames[comp] == "j"+field_names[idir]+"_displacement" ) {
                    m_all_field_functors[lev][comp] = std::make_unique<JdispFunctor>(idir, lev, m_crse_ratio, true);
            } else if ( m_varnames[comp] == "A"+field_names[idir] ){
                m_all_field_functors[lev][comp] = std::make_unique<CellCenterFunctor>(warpx.m_fields.get(FieldType::vector_potential_fp_nodal, Direction{idir}, lev), lev, m_crse_ratio);
            }
        }
        // Check if comp was found above
        if (m_all_field_functors[lev][comp]) {continue;}

        if ( m_varnames[comp] == "rho" ){
            // Initialize rho functor to dump total rho
            m_all_field_functors[lev][comp] = std::make_unique<RhoFunctor>(lev, m_crse_ratio, true);
        } else if ( m_varnames[comp].rfind("rho_", 0) == 0 ){
            // Initialize rho functor to dump rho per species
            m_all_field_functors[lev][comp] = std::make_unique<RhoFunctor>(lev, m_crse_ratio, true, m_rho_per_species_index[i]);
            i++;
        } else if ( m_varnames[comp].rfind("T_", 0) == 0 ){
            // Initialize temperature functor to dump temperature per species
            m_all_field_functors[lev][comp] = std::make_unique<TemperatureFunctor>(lev, m_crse_ratio, m_T_per_species_index[i_T_species]);
            i_T_species++;
        } else if ( m_varnames[comp] == "F" ){
            m_all_field_functors[lev][comp] = std::make_unique<CellCenterFunctor>(warpx.m_fields.get(FieldType::F_fp, lev), lev, m_crse_ratio);
        } else if ( m_varnames[comp] == "G" ){
            m_all_field_functors[lev][comp] = std::make_unique<CellCenterFunctor>(warpx.m_fields.get(FieldType::G_fp, lev), lev, m_crse_ratio);
        } else if ( m_varnames[comp] == "phi" ){
            m_all_field_functors[lev][comp] = std::make_unique<CellCenterFunctor>(warpx.m_fields.get(FieldType::phi_fp, lev), lev, m_crse_ratio);
        } else if ( m_varnames[comp] == "part_per_cell" ){
            m_all_field_functors[lev][comp] = std::make_unique<PartPerCellFunctor>(nullptr, lev, m_crse_ratio);
        } else if ( m_varnames[comp] == "part_per_grid" ){
            m_all_field_functors[lev][comp] = std::make_unique<PartPerGridFunctor>(nullptr, lev, m_crse_ratio);
        } else if ( m_varnames[comp] == "divB" ){
            m_all_field_functors[lev][comp] = std::make_unique<DivBFunctor>(warpx.m_fields.get_alldirs(FieldType::Bfield_aux, lev), lev, m_crse_ratio);
        } else if ( m_varnames[comp] == "divE" ){
            m_all_field_functors[lev][comp] = std::make_unique<DivEFunctor>(warpx.m_fields.get_alldirs(FieldType::Efield_aux, lev), lev, m_crse_ratio);
        } else {
            std::cout << "Error on component " << m_varnames[comp] << std::endl;
            WARPX_ABORT_WITH_MESSAGE(m_varnames[comp] + " is not a known field output type for this geometry");
        }
    }
    // Add functors for average particle data for each species
    for (int pcomp=0; pcomp<int(m_pfield_varnames.size()); pcomp++) {
        for (int ispec=0; ispec<int(m_pfield_species.size()); ispec++) {
            m_all_field_functors[lev][nvar + pcomp * nspec + ispec] = std::make_unique<ParticleReductionFunctor>(nullptr,
                    lev, m_crse_ratio, m_pfield_strings[pcomp], m_pfield_species_index[ispec], m_pfield_do_average[pcomp],
                    m_pfield_dofilter[pcomp], m_pfield_filter_strings[pcomp]);
        }
    }
    AddRZModesToDiags( lev );
}


void
FullDiagnostics::PrepareFieldDataForOutput ()
{
    // First, make sure all guard cells are properly filled
    // Probably overkill/unnecessary, but safe and shouldn't happen often !!
    auto & warpx = WarpX::GetInstance();
    warpx.FillBoundaryE(warpx.getngEB());
    warpx.FillBoundaryB(warpx.getngEB());
    warpx.UpdateAuxilaryData();
    warpx.FillBoundaryAux(warpx.getngUpdateAux());

    // Update the RealBox used for the geometry filter in particle diags
    // Note that full diagnostics every diag has only one buffer. (m_num_buffers = 1).
    // For m_geom_output[i_buffer][lev], the first element is the buffer index, and
    // second is level=0
    // The level is set to 0, because the whole physical domain of the simulation is used
    // to set the domain dimensions for the output particle container.
    for (int i_buffer = 0; i_buffer < m_num_buffers; ++i_buffer) {
        for (int i = 0; i < m_output_species.at(i_buffer).size(); ++i) {
            m_output_species[i_buffer][i].m_diag_domain = m_geom_output[i_buffer][0].ProbDomain();
        }
    }
}

void
FullDiagnostics::MovingWindowAndGalileanDomainShift (int step)
{
    auto & warpx = WarpX::GetInstance();

    // Get current finest level available
    const int finest_level = warpx.finestLevel();

    // Account for galilean shift
    amrex::Real new_lo[AMREX_SPACEDIM];
    amrex::Real new_hi[AMREX_SPACEDIM];
    // Note that Full diagnostics has only one snapshot, m_num_buffers = 1
    // m_geom_output[i_buffer][lev] below have values 0 and 0, respectively, because
    // we need the physical extent from mesh-refinement level = 0,
    // and only for the 0th snapshot, since full diagnostics has only one snapshot.
    const amrex::Real* current_lo = m_geom_output[0][0].ProbLo();
    const amrex::Real* current_hi = m_geom_output[0][0].ProbHi();

#if defined(WARPX_DIM_3D)
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        new_lo[idim] = current_lo[idim] + warpx.m_galilean_shift[idim];
        new_hi[idim] = current_hi[idim] + warpx.m_galilean_shift[idim];
    }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    {
        new_lo[0] = current_lo[0] + warpx.m_galilean_shift[0];
        new_hi[0] = current_hi[0] + warpx.m_galilean_shift[0];
        new_lo[1] = current_lo[1] + warpx.m_galilean_shift[2];
        new_hi[1] = current_hi[1] + warpx.m_galilean_shift[2];
    }
#elif defined(WARPX_DIM_1D_Z)
    {
        new_lo[0] = current_lo[0] + warpx.m_galilean_shift[2];
        new_hi[0] = current_hi[0] + warpx.m_galilean_shift[2];
    }
#endif
    // Update RealBox of geometry with galilean-shifted boundary.
    for (int lev = 0; lev <= finest_level; ++lev) {
        // Note that Full diagnostics has only one snapshot, m_num_buffers = 1
        // Thus here we set the prob domain for the 0th snapshot only.
        m_geom_output[0][lev].ProbDomain( amrex::RealBox(new_lo, new_hi) );
    }
    // For Moving Window Shift
    if (WarpX::moving_window_active(step+1)) {
        const int moving_dir = WarpX::moving_window_dir;
        const amrex::Real moving_window_x = warpx.getmoving_window_x();
        // Get the updated lo and hi of the geom domain
        const amrex::Real* cur_lo = m_geom_output[0][0].ProbLo();
        const amrex::Real* cur_hi = m_geom_output[0][0].ProbHi();
        const amrex::Real* geom_dx = m_geom_output[0][0].CellSize();
        const auto num_shift_base = static_cast<int>((moving_window_x - cur_lo[moving_dir])
                                              / geom_dx[moving_dir]);
        // Update the diagnostic geom domain. Note that this is done only for the
        // base level 0 because m_geom_output[0][lev] share the same static RealBox
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            new_lo[idim] = cur_lo[idim];
            new_hi[idim] = cur_hi[idim];
        }
        new_lo[moving_dir] = cur_lo[moving_dir] + num_shift_base*geom_dx[moving_dir];
        new_hi[moving_dir] = cur_hi[moving_dir] + num_shift_base*geom_dx[moving_dir];
        // Update RealBox of geometry with shifted domain geometry for moving-window
        for (int lev = 0; lev < nmax_lev; ++lev) {
            // Note that Full diagnostics has only one snapshot, m_num_buffers = 1
            // Thus here we set the prob domain for the 0th snapshot only.
            m_geom_output[0][lev].ProbDomain( amrex::RealBox(new_lo, new_hi) );
        }
    }


}
