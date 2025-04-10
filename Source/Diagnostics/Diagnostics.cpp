#include "Diagnostics.H"

#include "Diagnostics/ComputeDiagFunctors/ComputeDiagFunctor.H"
#include "ComputeDiagFunctors/BackTransformParticleFunctor.H"
#include "Diagnostics/FlushFormats/FlushFormat.H"
#include "Diagnostics/ParticleDiag/ParticleDiag.H"
#include "FlushFormats/FlushFormatCatalyst.H"
#include "FlushFormats/FlushFormatAscent.H"
#include "FlushFormats/FlushFormatCheckpoint.H"
#ifdef WARPX_USE_OPENPMD
#   include "FlushFormats/FlushFormatOpenPMD.H"
#endif
#include "FlushFormats/FlushFormatPlotfile.H"
#include "FlushFormats/FlushFormatSensei.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/Algorithms/IsIn.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

#include <ablastr/utils/Communication.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_BLassert.H>
#include <AMReX_Config.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <string>

using namespace amrex::literals;

Diagnostics::Diagnostics (int i, std::string name, DiagTypes diag_type)
    : m_diag_type(diag_type), m_diag_name(std::move(name)), m_diag_index(i)
{
}

Diagnostics::~Diagnostics () = default;

bool
Diagnostics::BaseReadParameters ()
{
    auto & warpx = WarpX::GetInstance();

    const amrex::ParmParse pp_diag_name(m_diag_name);
    m_file_prefix = "diags/" + m_diag_name;
    pp_diag_name.query("file_prefix", m_file_prefix);
    utils::parser::queryWithParser(
        pp_diag_name, "file_min_digits", m_file_min_digits);
    pp_diag_name.query("format", m_format);
    pp_diag_name.query("dump_last_timestep", m_dump_last_timestep);

    const amrex::ParmParse pp_geometry("geometry");
    std::string dims;
    pp_geometry.get("dims", dims);

    const amrex::ParmParse pp_warpx("warpx");
    pp_warpx.query("verbose", m_verbose);

    // Query list of grid fields to write to output
    const bool varnames_specified = pp_diag_name.queryarr("fields_to_plot", m_varnames_fields);
    if (!varnames_specified){
        if( dims == "RZ" ) {
            m_varnames_fields = {"Er", "Et", "Ez", "Br", "Bt", "Bz", "jr", "jt", "jz"};
        }
        else {
            m_varnames_fields = {"Ex", "Ey", "Ez", "Bx", "By", "Bz", "jx", "jy", "jz"};
        }
    }

    // Sanity check if user requests to plot phi
    if (utils::algorithms::is_in(m_varnames_fields, "phi")){
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            warpx.electrostatic_solver_id==ElectrostaticSolverAlgo::LabFrame ||
            warpx.electrostatic_solver_id==ElectrostaticSolverAlgo::LabFrameElectroMagnetostatic ||
            warpx.electrostatic_solver_id==ElectrostaticSolverAlgo::LabFrameEffectivePotential,
            "plot phi only works if do_electrostatic = labframe, do_electrostatic = labframe-electromagnetostatic or do_electrostatic = labframe-effective-potential");
    }

    // Sanity check if user requests to plot A
    if (utils::algorithms::any_of_is_in(m_varnames_fields, amrex::Vector<std::string>({"Ax", "Ay", "Az", "Ar", "At"}))){
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            warpx.electrostatic_solver_id==ElectrostaticSolverAlgo::LabFrameElectroMagnetostatic,
            "plot A only works if do_electrostatic = labframe-electromagnetostatic");
    }

    // Sanity check if user requests to plot F
    if (utils::algorithms::is_in(m_varnames_fields, "F")){
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            warpx.do_dive_cleaning,
            "plot F only works if warpx.do_dive_cleaning = 1");
    }

    // G can be written to file only if WarpX::do_divb_cleaning = 1
    if (utils::algorithms::is_in(m_varnames_fields, "G"))
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            warpx.do_divb_cleaning,
            "G can be written to file only if warpx.do_divb_cleaning = 1");
    }

    // If user requests to plot proc_number for a serial run,
    // delete proc_number from fields_to_plot
    if (amrex::ParallelDescriptor::NProcs() == 1){
        m_varnames_fields.erase(
            std::remove(m_varnames_fields.begin(), m_varnames_fields.end(), "proc_number"),
            m_varnames_fields.end());
    }

    // Get names of particle field diagnostic quantities to calculate at each grid point
    const bool pfield_varnames_specified = pp_diag_name.queryarr("particle_fields_to_plot", m_pfield_varnames);
    if (!pfield_varnames_specified){
        m_pfield_varnames = {};
    }

#ifdef WARPX_DIM_RZ
    if (pfield_varnames_specified){
        ablastr::warn_manager::WMRecordWarning(
            "Diagnostics",
            "Particle field diagnostics will output the 0th mode only.",
            ablastr::warn_manager::WarnPriority::low
        );
    }
#endif

    // Get parser strings for particle fields and generate map of parsers
    std::string parser_str;
    std::string filter_parser_str;
    const amrex::ParmParse pp_diag_pfield(m_diag_name + ".particle_fields");
    for (const auto& var : m_pfield_varnames) {

        bool do_average = true;
        pp_diag_pfield.query((var + ".do_average").c_str(), do_average);
        m_pfield_do_average.push_back(do_average);
        utils::parser::Store_parserString(
            pp_diag_pfield, (var + "(x,y,z,ux,uy,uz)"), parser_str);

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !parser_str.empty(),
            std::string("Input error: cannot find parser string for ").append(var).append(" in file. ").append(
                m_diag_name).append(".particle_fields.").append(var).append("(x,y,z,ux,uy,uz) is required"));

        m_pfield_strings.push_back(parser_str);

        // Look for and record filter functions. If one is not found, the empty string will be
        // stored as the filter string, and will be ignored.
        const bool do_parser_filter = pp_diag_pfield.query((var + ".filter(x,y,z,ux,uy,uz)").c_str(), filter_parser_str);
        m_pfield_dofilter.push_back(do_parser_filter);
        m_pfield_filter_strings.push_back(filter_parser_str);
    }

    // Names of all species in the simulation
    m_all_species_names = warpx.GetPartContainer().GetSpeciesNames();

    // Get names of species to average at each grid point
    const bool pfield_species_specified = pp_diag_name.queryarr("particle_fields_species", m_pfield_species);
    if (!pfield_species_specified){
        m_pfield_species = m_all_species_names;
    }

    // Check that species names specified in m_pfield_species are valid
    // Loop over all species specified above
    for (const auto& species : m_pfield_species) {
        // Boolean used to check if species name was misspelled
        bool p_species_name_is_wrong = true;
        // Loop over all species
        for (int i = 0, n = int(m_all_species_names.size()); i < n; i++) {
            if (species == m_all_species_names[i]) {
                // Store species index: will be used in ParticleReductionFunctor to calculate
                // averages for this species
                m_pfield_species_index.push_back(i);
                p_species_name_is_wrong = false;
            }
        }
        // If species name was misspelled, abort with error message
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !p_species_name_is_wrong,
            "Input error: string " + species + " in " + m_diag_name
            + ".particle_fields_species does not match any species"
        );
    }

    if (utils::algorithms::is_in(m_varnames_fields, "none")){
        m_varnames_fields.clear();
    }

    m_varnames = m_varnames_fields;
    // Generate names of averaged particle fields and append to m_varnames
    for (const auto& fname : m_pfield_varnames) {
        for (const auto& sname : m_pfield_species) {
            auto varname = fname;
            varname.append("_").append(sname);
            m_varnames.push_back(varname);
        }
    }

    // Read user-defined physical extents for the output and store in m_lo and m_hi.
    m_lo.resize(AMREX_SPACEDIM);
    m_hi.resize(AMREX_SPACEDIM);

    const bool lo_specified = utils::parser::queryArrWithParser(
        pp_diag_name, "diag_lo", m_lo, 0, AMREX_SPACEDIM);

    if (!lo_specified) {
       for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
            m_lo[idim] = warpx.Geom(0).ProbLo(idim);
       }
    }
    const bool hi_specified = utils::parser::queryArrWithParser(
        pp_diag_name, "diag_hi", m_hi, 0, AMREX_SPACEDIM);
    if (!hi_specified) {
       for (int idim =0; idim < AMREX_SPACEDIM; ++idim) {
            m_hi[idim] = warpx.Geom(0).ProbHi(idim);
       }
    }
    // For a moving window simulation, the user-defined m_lo and m_hi must be converted.
    if (WarpX::do_moving_window) {
#if defined(WARPX_DIM_3D)
    amrex::Vector<int> dim_map {0, 1, 2};
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    amrex::Vector<int> dim_map {0, 2};
#else
    amrex::Vector<int> dim_map {2};
#endif
       if (WarpX::boost_direction[ dim_map[WarpX::moving_window_dir] ] == 1) {
           // Convert user-defined lo and hi for diagnostics to account for boosted-frame
           // simulations with moving window
           const amrex::Real beta_window = WarpX::moving_window_v / PhysConst::c;
           const amrex::Real convert_factor = 1._rt/(
               WarpX::gamma_boost * (1._rt - WarpX::beta_boost * beta_window) );
           m_lo[WarpX::moving_window_dir] *= convert_factor;
           m_hi[WarpX::moving_window_dir] *= convert_factor;
       }
    }

    // Initialize cr_ratio with default value of 1 for each dimension.
    amrex::Vector<int> cr_ratio(AMREX_SPACEDIM, 1);
    // Read user-defined coarsening ratio for the output MultiFab.
    const bool cr_specified =
        utils::parser::queryArrWithParser(
            pp_diag_name, "coarsening_ratio", cr_ratio, 0, AMREX_SPACEDIM);
    if (cr_specified) {
       for (int idim =0; idim < AMREX_SPACEDIM; ++idim) {
           m_crse_ratio[idim] = cr_ratio[idim];
       }
    }

    // Names of species to write to output
    const bool species_specified =
        pp_diag_name.queryarr("species", m_output_species_names);


    // Loop over all fields stored in m_varnames
    for (const auto& var : m_varnames) {
        // Check if m_varnames contains a string of the form rho_<species_name>
        if (var.rfind("rho_", 0) == 0) {
            // Extract species name from the string rho_<species_name>
            const std::string species = var.substr(var.find("rho_") + 4);
            // Boolean used to check if species name was misspelled
            bool species_name_is_wrong = true;
            // Loop over all species
            for (int i = 0, n = int(m_all_species_names.size()); i < n; i++) {
                // Check if species name extracted from the string rho_<species_name>
                // matches any of the species in the simulation
                if (species == m_all_species_names[i]) {
                    // Store species index: will be used in RhoFunctor to dump
                    // rho for this species
                    m_rho_per_species_index.push_back(i);
                    species_name_is_wrong = false;
                }
            }
            // If species name was misspelled, abort with error message
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !species_name_is_wrong,
                "Input error: string " + var + " in " + m_diag_name
                + ".fields_to_plot does not match any species"
            );
        }
        // Check if m_varnames contains a string of the form T_<species_name>
        if (var.rfind("T_", 0) == 0) {
            // Extract species name from the string T_<species_name>
            const std::string species = var.substr(var.find("T_") + 2);
            // Boolean used to check if species name was misspelled
            bool species_name_is_wrong = true;
            // Loop over all species
            for (int i = 0, n = int(m_all_species_names.size()); i < n; i++) {
                // Check if species name extracted from the string T_<species_name>
                // matches any of the species in the simulation
                if (species == m_all_species_names[i]) {
                    // Store species index: will be used in TemperatureFunctor to dump
                    // T for this species
                    m_T_per_species_index.push_back(i);
                    species_name_is_wrong = false;
                }
            }
            // If species name was misspelled, abort with error message
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !species_name_is_wrong,
                "Input error: string " + var + " in " + m_diag_name
                + ".fields_to_plot does not match any species"
            );
        }
    }

    const bool checkpoint_compatibility = (
        m_format == "checkpoint" &&
        !varnames_specified &&
        !pfield_varnames_specified &&
        !pfield_species_specified &&
        !lo_specified &&
        !hi_specified &&
        !cr_specified &&
        !species_specified );

    return checkpoint_compatibility;
}


void
Diagnostics::InitDataBeforeRestart ()
{
    // initialize member variables and arrays in base class::Diagnostics
    InitBaseData();
    // initialize member variables and arrays specific to each derived class
    // (FullDiagnostics, BTDiagnostics, etc.)
    DerivedInitData();
}

void
Diagnostics::InitDataAfterRestart ()
{
    for (int i_buffer = 0; i_buffer < m_num_buffers; ++i_buffer) {
        // loop over all levels
        // This includes full diagnostics and BTD as well as cell-center functors for BTD.
        // Note that the cell-centered data for BTD is computed for all levels and hence
        // the corresponding functor is also initialized for all the levels
        for (int lev = 0; lev < nmax_lev; ++lev) {
            // allocate and initialize m_all_field_functors depending on diag type
            InitializeFieldFunctors(lev);
        }
        // loop over the levels selected for output
        // This includes all the levels for full diagnostics
        // and only the coarse level (mother grid) for BTD
        for (int lev = 0; lev < nlev_output; ++lev) {
            // Initialize buffer data required for particle and/or fields
            InitializeBufferData(i_buffer, lev, true);
        }
    }

    const amrex::ParmParse pp_diag_name(m_diag_name);
    // default for writing species output is 1
    int write_species = 1;
    pp_diag_name.query("write_species", write_species);
    if (write_species == 1) {
        // When particle buffers, m_particle_boundary_buffer are included,
        // they will be initialized here
        InitializeParticleBuffer();
        InitializeParticleFunctors();
    }
    if (write_species == 0) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_format != "checkpoint",
            "For checkpoint format, write_species flag must be 1."
        );
        // if user-defined value for write_species == 0, then clear species vector
        for (int i_buffer = 0; i_buffer < m_num_buffers; ++i_buffer ) {
            m_output_species.at(i_buffer).clear();
        }
        m_output_species_names.clear();
    } else {
        amrex::Vector <amrex::Real> dummy_val(AMREX_SPACEDIM);
        if ( utils::parser::queryArrWithParser(
                pp_diag_name, "diag_lo", dummy_val, 0, AMREX_SPACEDIM) ||
             utils::parser::queryArrWithParser(
                pp_diag_name, "diag_hi", dummy_val, 0, AMREX_SPACEDIM) ) {
            // set geometry filter for particle-diags to true when the diagnostic domain-extent
            // is specified by the user.
            // Note that the filter is set for every ith snapshot, and the number of snapshots
            // for full diagnostics is 1, while for BTD it is user-defined.
            for (int i_buffer = 0; i_buffer < m_num_buffers; ++i_buffer ) {
                for (auto& v : m_output_species.at(i_buffer)) {
                    v.m_do_geom_filter = true;
                }
                // Disabling particle-io for reduced domain diagnostics by reducing
                // the particle-diag vector to zero.
                // This is a temporary fix until particle_buffer is supported in diagnostics.
                m_output_species.at(i_buffer).clear();
            }
            std::string warnMsg = "For full diagnostics on a reduced domain, particle I/O is not ";
            warnMsg += "supported, yet! Therefore, particle I/O is disabled for this diagnostics: ";
            warnMsg += m_diag_name;
            ablastr::warn_manager::WMRecordWarning("Diagnostics", warnMsg);
        }
    }
}


void
Diagnostics::InitData ()
{
    auto& warpx = WarpX::GetInstance();

    // Get current finest level available
    const int finest_level = warpx.finestLevel();

    // initialize member variables and arrays in base class::Diagnostics
    InitBaseData();
    // initialize member variables and arrays specific to each derived class
    // (FullDiagnostics, BTDiagnostics, etc.)
    DerivedInitData();
    for (int i_buffer = 0; i_buffer < m_num_buffers; ++i_buffer) {
        // loop over all levels
        // This includes full diagnostics and BTD as well as cell-center functors for BTD.
        // Note that the cell-centered data for BTD is computed for all levels and hence
        // the corresponding functor is also initialized for all the levels
        for (int lev = 0; lev <= finest_level; ++lev) {
            // allocate and initialize m_all_field_functors depending on diag type
            InitializeFieldFunctors(lev);
        }
        // loop over the levels selected for output
        // This includes all the levels for full diagnostics
        // and only the coarse level (mother grid) for BTD
        for (int lev = 0; lev < nlev_output; ++lev) {
            // Initialize buffer data required for particle and/or fields
            InitializeBufferData(i_buffer, lev);
        }
    }

    const amrex::ParmParse pp_diag_name(m_diag_name);
    // default for writing species output is 1
    int write_species = 1;
    pp_diag_name.query("write_species", write_species);
    if (write_species == 1) {
        // When particle buffers, m_particle_boundary_buffer are included,
        // they will be initialized here
        InitializeParticleBuffer();
        InitializeParticleFunctors();
    }

    if (write_species == 0) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_format != "checkpoint",
            "For checkpoint format, write_species flag must be 1."
        );
        // if user-defined value for write_species == 0, then clear species vector
        for (int i_buffer = 0; i_buffer < m_num_buffers; ++i_buffer ) {
            m_output_species.at(i_buffer).clear();
        }
        m_output_species_names.clear();
    } else {
        amrex::Vector <amrex::Real> dummy_val(AMREX_SPACEDIM);
        if ( utils::parser::queryArrWithParser(
                pp_diag_name, "diag_lo", dummy_val, 0, AMREX_SPACEDIM) ||
             utils::parser::queryArrWithParser(
                pp_diag_name, "diag_hi", dummy_val, 0, AMREX_SPACEDIM) ) {
            // set geometry filter for particle-diags to true when the diagnostic domain-extent
            // is specified by the user.
            // Note that the filter is set for every ith snapshot, and the number of snapshots
            // for full diagnostics is 1, while for BTD it is user-defined.
            for (int i_buffer = 0; i_buffer < m_num_buffers; ++i_buffer ) {
                for (auto& v : m_output_species.at(i_buffer)) {
                    v.m_do_geom_filter = true;
                }
                // Disabling particle-io for reduced domain diagnostics by reducing
                // the particle-diag vector to zero.
                // This is a temporary fix until particle_buffer is supported in diagnostics.
                m_output_species.at(i_buffer).clear();
            }
            std::string warnMsg = "For full diagnostics on a reduced domain, particle I/O is not ";
            warnMsg += "supported, yet! Therefore, particle I/O is disabled for this diagnostics: ";
            warnMsg += m_diag_name;
            ablastr::warn_manager::WMRecordWarning("Diagnostics", warnMsg);
        }
    }
}


void
Diagnostics::InitBaseData ()
{
    auto & warpx = WarpX::GetInstance();
    // Number of levels in the simulation at the current timestep
    nlev = warpx.finestLevel() + 1;
    // default number of levels to be output = nlev
    nlev_output = nlev;
    // Maximum number of levels that will be allocated in the simulation
    nmax_lev = warpx.maxLevel() + 1;
    m_all_field_functors.resize( nmax_lev );

    // For restart, move the m_lo and m_hi of the diag consistent with the
    // current moving_window location
    if (WarpX::do_moving_window) {
        const int moving_dir = WarpX::moving_window_dir;
        const int shift_num_base = static_cast<int>((warpx.getmoving_window_x() - m_lo[moving_dir]) / warpx.Geom(0).CellSize(moving_dir) );
        m_lo[moving_dir] += shift_num_base * warpx.Geom(0).CellSize(moving_dir);
        m_hi[moving_dir] += shift_num_base * warpx.Geom(0).CellSize(moving_dir);
    }
    // Construct Flush class.
    if        (m_format == "plotfile"){
        m_flush_format = std::make_unique<FlushFormatPlotfile>() ;
    } else if (m_format == "checkpoint"){
        // creating checkpoint format
        m_flush_format = std::make_unique<FlushFormatCheckpoint>() ;
    } else if (m_format == "ascent"){
        m_flush_format = std::make_unique<FlushFormatAscent>();
    } else if (m_format == "catalyst") {
        m_flush_format = std::make_unique<FlushFormatCatalyst>();
    } else if (m_format == "sensei") {
#ifdef AMREX_USE_SENSEI_INSITU
        m_flush_format = std::make_unique<FlushFormatSensei>(
            dynamic_cast<amrex::AmrMesh*>(const_cast<WarpX*>(&warpx)),
            m_diag_name);
#else
        WARPX_ABORT_WITH_MESSAGE(
            "To use SENSEI in situ, compile with USE_SENSEI=TRUE");
#endif
    } else if (m_format == "openpmd"){
#ifdef WARPX_USE_OPENPMD
        m_flush_format = std::make_unique<FlushFormatOpenPMD>(m_diag_name);
#else
        WARPX_ABORT_WITH_MESSAGE(
            "To use openpmd output format, need to compile with USE_OPENPMD=TRUE");
#endif
    } else {
        WARPX_ABORT_WITH_MESSAGE(
            "unknown output format");
    }

    // allocate vector of buffers then allocate vector of levels for each buffer
    m_mf_output.resize( m_num_buffers );
    for (int i = 0; i < m_num_buffers; ++i) {
        m_mf_output[i].resize( nmax_lev );
    }

    // allocate vector of buffers and vector of levels for each buffer for summation multifab for TimeAveragedDiagnostics
    if (m_diag_type == DiagTypes::TimeAveraged) {
        m_sum_mf_output.resize(m_num_buffers);
        for (int i = 0; i < m_num_buffers; ++i) {
            m_sum_mf_output[i].resize( nmax_lev );
        }
    }

    // allocate vector of geometry objects corresponding to each output multifab.
    m_geom_output.resize( m_num_buffers );
    for (int i = 0; i < m_num_buffers; ++i) {
        m_geom_output[i].resize( nmax_lev );
    }

    // allocate vector of particle buffers
    m_output_species.resize(m_num_buffers);
}

void
Diagnostics::ComputeAndPack ()
{
    PrepareBufferData();
    // prepare the field-data necessary to compute output data
    PrepareFieldDataForOutput();
    // Prepare the particle data necessary to compute output data
    // Field-data is called first for BTD, since the z-slice location is used to prepare particle data
    // to determine if the transform is to be done this step.
    PrepareParticleDataForOutput();

    auto & warpx = WarpX::GetInstance();

    // compute the necessary fields and store result in m_mf_output.
    for (int i_buffer = 0; i_buffer < m_num_buffers; ++i_buffer) {
        for(int lev=0; lev<nlev_output; lev++){
            int icomp_dst = 0;
            const auto n = static_cast<int>(m_all_field_functors[lev].size());
            for (int icomp=0; icomp<n; icomp++){
                // Call all functors in m_all_field_functors[lev]. Each of them computes
                // a diagnostics and writes in one or more components of the output
                // multifab m_mf_output[lev].
                m_all_field_functors[lev][icomp]->operator()(m_mf_output[i_buffer][lev], icomp_dst, i_buffer);
                // update the index of the next component to fill
                icomp_dst += m_all_field_functors[lev][icomp]->nComp();
            }
            // Check that the proper number of components of mf_avg were updated.
            AMREX_ALWAYS_ASSERT( icomp_dst == m_varnames.size() );

            if (m_diag_type == DiagTypes::TimeAveraged) {

                const amrex::Real real_a = 1.0;
                // Compute m_sum_mf_output += real_a*m_mf_output
                amrex::MultiFab::Saxpy(
                        m_sum_mf_output[i_buffer][lev], real_a, m_mf_output[i_buffer][lev],
                        0, 0, m_mf_output[i_buffer][lev].nComp(), m_mf_output[i_buffer][lev].nGrowVect());
            }

            // needed for contour plots of rho, i.e. ascent/sensei
            if (m_format == "sensei" || m_format == "ascent") {
                ablastr::utils::communication::FillBoundary(m_mf_output[i_buffer][lev], WarpX::do_single_precision_comms,
                                                            warpx.Geom(lev).periodicity());
            }
        }
        // Call Particle functor
        for (int isp = 0; isp < m_all_particle_functors.size(); ++isp) {
            m_all_particle_functors[isp]->operator()(*m_particles_buffer[i_buffer][isp], m_totalParticles_in_buffer[i_buffer][isp], i_buffer);
        }
    }

    UpdateBufferData();
}


void
Diagnostics::FilterComputePackFlush (int step, bool force_flush)
{
    WARPX_PROFILE("Diagnostics::FilterComputePackFlush()");
    MovingWindowAndGalileanDomainShift (step);

    if ( DoComputeAndPack (step, force_flush) ) {
        ComputeAndPack();
    }

    for (int i_buffer = 0; i_buffer < m_num_buffers; ++i_buffer) {
        if ( !DoDump (step, i_buffer, force_flush) ) { continue; }
        Flush(i_buffer, force_flush);
    }
}
