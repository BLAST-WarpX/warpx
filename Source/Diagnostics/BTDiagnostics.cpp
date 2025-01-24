/* Copyright 2021 Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "BTDiagnostics.H"
#include "BTD_Plotfile_Header_Impl.H"
#include "ComputeDiagFunctors/BackTransformFunctor.H"
#include "ComputeDiagFunctors/CellCenterFunctor.H"
#include "ComputeDiagFunctors/ComputeDiagFunctor.H"
#include "ComputeDiagFunctors/RhoFunctor.H"
#include "Diagnostics/Diagnostics.H"
#include "Diagnostics/FlushFormats/FlushFormat.H"
#include "ComputeDiagFunctors/BackTransformParticleFunctor.H"
#include "Fields.H"
#include "Utils/Algorithms/IsIn.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/coarsen/sample.H>
#include <ablastr/utils/Communication.H>
#include <ablastr/utils/SignalHandling.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_Algorithm.H>
#include <AMReX_BLassert.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FileSystem.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <sstream>
#include <vector>

using namespace amrex::literals;
using warpx::fields::FieldType;

namespace
{
    constexpr int permission_flag_rwxrxrx = 0755;
}

BTDiagnostics::BTDiagnostics (int i, const std::string& name, DiagTypes diag_type)
    : Diagnostics{i, name, diag_type},
      m_cell_centered_data_name("BTD_cell_centered_data_" + name)
{
    ReadParameters();
}


void BTDiagnostics::DerivedInitData ()
{
    auto & warpx = WarpX::GetInstance();
    m_gamma_boost = WarpX::gamma_boost;
    m_beta_boost = std::sqrt( 1._rt - 1._rt/( m_gamma_boost * m_gamma_boost) );
    m_moving_window_dir = WarpX::moving_window_dir;
    m_moving_window_beta = WarpX::moving_window_v/PhysConst::c;
    // Currently, for BTD, all the data is averaged+coarsened to coarsest level
    // and then sliced+back-transformed+filled_to_buffer.
    // The number of levels to be output is nlev_output.
    nlev_output = 1;

    m_t_lab.resize(m_num_buffers);
    m_snapshot_domain_lab.resize(m_num_buffers);
    m_buffer_domain_lab.resize(m_num_buffers);
    m_snapshot_box.resize(m_num_buffers);
    m_buffer_box.resize(m_num_buffers);
    m_current_z_lab.resize(m_num_buffers);
    m_current_z_boost.resize(m_num_buffers);
    m_old_z_boost.resize(m_num_buffers);
    m_buffer_counter.resize(m_num_buffers);
    m_snapshot_ncells_lab.resize(m_num_buffers);
    m_cell_center_functors.resize(nmax_lev);
    m_max_buffer_multifabs.resize(m_num_buffers);
    m_buffer_flush_counter.resize(m_num_buffers);
    m_geom_snapshot.resize( m_num_buffers );
    m_snapshot_full.resize( m_num_buffers );
    m_lastValidZSlice.resize( m_num_buffers );
    m_buffer_k_index_hi.resize(m_num_buffers);
    m_first_flush_after_restart.resize(m_num_buffers);
    m_snapshot_geometry_defined.resize(m_num_buffers);
    m_field_buffer_multifab_defined.resize(m_num_buffers);
    for (int i = 0; i < m_num_buffers; ++i) {
        m_geom_snapshot[i].resize(nmax_lev);
        m_snapshot_full[i] = 0;
        m_lastValidZSlice[i] = 0;
        m_buffer_flush_counter[i] = 0;
        m_first_flush_after_restart[i] = 1;
        m_snapshot_geometry_defined[i] = 0;
        m_field_buffer_multifab_defined[i] = 0;
    }
    for (int lev = 0; lev < nmax_lev; ++lev) {
        // Define cell-centered multifab over the whole domain with
        // user-defined crse_ratio for nlevels
        DefineCellCenteredMultiFab(lev);
    }

    /* Allocate vector of particle buffer vectors for each snapshot */
    MultiParticleContainer& mpc = warpx.GetPartContainer();
    // If not specified, and write species is not 0, dump all species
    const amrex::ParmParse pp_diag_name(m_diag_name);
    int write_species = 1;
    pp_diag_name.query("write_species", write_species);
    if ((m_output_species_names.empty()) && (write_species == 1)) {
        m_output_species_names = mpc.GetSpeciesNames();
    }

    m_do_back_transformed_particles =
        ((!m_output_species_names.empty()) && (write_species == 1));

    // Turn on do_back_transformed_particles in the particle containers so that
    // the tmp_particle_data is allocated and the data of the corresponding species is
    // copied and stored in tmp_particle_data before particles are pushed.
    if (m_do_back_transformed_particles) {
        mpc.SetDoBackTransformedParticles(m_do_back_transformed_particles);
        for (auto const& species : m_output_species_names){
            mpc.SetDoBackTransformedParticles(species, m_do_back_transformed_particles);
        }
    }
    m_particles_buffer.resize(m_num_buffers);
    m_totalParticles_in_buffer.resize(m_num_buffers);

    // check that simulation can fill all BTD snapshots
    const int lev = 0;
    const amrex::Real dt_boosted_frame = warpx.getdt(lev);
    const int moving_dir = WarpX::moving_window_dir;
    const amrex::Real Lz_lab = warpx.Geom(lev).ProbLength(moving_dir) * WarpX::gamma_boost * (1._rt - WarpX::beta_boost*m_moving_window_beta);
    const int ref_ratio = 1;
    const amrex::Real dz_snapshot_grid = dz_lab(dt_boosted_frame, ref_ratio);
    // Need enough buffers so the snapshot length is longer than the lab frame length
    // num_buffers * m_buffer_size * dz_snapshot_grid >= Lz
    const int num_buffers = static_cast<int>(std::ceil(Lz_lab / m_buffer_size / dz_snapshot_grid));
    const int final_snapshot_iteration = m_intervals.GetFinalIteration();

    // the final snapshot starts filling when the
    // right edge of the moving window intersects the final snapshot
    // time of final snapshot : t_sn = t0 + i*dt_snapshot
    // where t0 is the time of first BTD snapshot, t0 = zmax / c  * beta / (1-beta*beta_mw)
    //
    // the right edge of the moving window at the time of the final snapshot
    // has space time coordinates
    // time t_intersect = t_sn, position  z_intersect=zmax + v_mw*t_sn
    // the boosted time of this space time pair is
    // t_intersect_boost = gamma * (t_intersect - beta * z_intersect_boost/c)
    //                   = gamma * (t_sn * (1 - beta*beta_mw) - beta * zmax / c)
    //                   = gamma * (zmax*beta/c + i*dt_snapshot*(1-beta*beta_mw) - beta*zmax/c)
    //                   = gamma * (1-beta*beta_mw) * i * dt_snapshot
    //
    // if j = final snapshot starting step, then we want to solve
    // j dt_boosted_frame >= t_intersect_boost = i * gamma * (1-beta*beta_mw) * dt_snapshot
    // j >= i * gamma * (1-beta*beta_mw) * dt_snapshot / dt_boosted_frame
    const int final_snapshot_starting_step = static_cast<int>(std::ceil(final_snapshot_iteration * WarpX::gamma_boost * (1._rt - WarpX::beta_boost*m_moving_window_beta) * m_dt_snapshots_lab / dt_boosted_frame));
    const int final_snapshot_fill_iteration = final_snapshot_starting_step + num_buffers * m_buffer_size - 1;
    const amrex::Real final_snapshot_fill_time = final_snapshot_fill_iteration * dt_boosted_frame;
    if (WarpX::compute_max_step_from_btd) {
        if (final_snapshot_fill_iteration > warpx.maxStep()) {
            warpx.updateMaxStep(final_snapshot_fill_iteration);
            amrex::Print()<<"max_step insufficient to fill all BTD snapshots. Automatically increased to: "
                << final_snapshot_fill_iteration << "\n";

        }
        if (final_snapshot_fill_time > warpx.stopTime()) {
            warpx.updateStopTime(final_snapshot_fill_time);
            amrex::Print()<<"stop_time insufficient to fill all BTD snapshots. Automatically increased to: "
                << final_snapshot_fill_time << "\n";

        }
        if (warpx.maxStep() == std::numeric_limits<int>::max() && warpx.stopTime() == std::numeric_limits<amrex::Real>::max()) {
            amrex::Print()<<"max_step unspecified and stop time unspecified.  Setting max step to "
                <<final_snapshot_fill_iteration<< " to fill all BTD snapshots.\n";
            warpx.updateMaxStep(final_snapshot_fill_iteration);
        }

    } else if (final_snapshot_fill_iteration > warpx.maxStep() || final_snapshot_fill_time > warpx.stopTime()) {
        std::stringstream warn_string;
            warn_string << "\nSimulation might not run long enough to fill all BTD snapshots.\n"
            << "Final step: " << warpx.maxStep() << "\n"
            <<"Stop time: " << warpx.stopTime() << "\n"
            <<"Last BTD snapshot fills around step: " << final_snapshot_fill_iteration << "\n"
            <<" or time: " << final_snapshot_fill_time << "\n";
        ablastr::warn_manager::WMRecordWarning(
            "BTD", warn_string.str(),
            ablastr::warn_manager::WarnPriority::low);
    }

#ifdef WARPX_DIM_RZ
    UpdateVarnamesForRZopenPMD();
#endif
}

void
BTDiagnostics::ReadParameters ()
{
    BaseReadParameters();

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( WarpX::gamma_boost > 1.0_rt,
        "gamma_boost must be > 1 to use the back-transformed diagnostics");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( WarpX::boost_direction[2] == 1,
        "The back transformed diagnostics currently only works if the boost is in the z-direction");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( WarpX::do_moving_window,
           "The moving window should be on if using the boosted frame diagnostic.");
    // The next two asserts could be relaxed with respect to check to current step
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( WarpX::end_moving_window_step < 0,
        "The moving window must not stop when using the boosted frame diagnostic.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( WarpX::start_moving_window_step == 0,
        "The moving window must start at step zero for the boosted frame diagnostic.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( WarpX::moving_window_dir == WARPX_ZINDEX,
           "The boosted frame diagnostic currently only works if the moving window is in the z direction.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_format == "plotfile" || m_format == "openpmd",
        "<diag>.format must be plotfile or openpmd for back transformed diagnostics");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_crse_ratio == amrex::IntVect(1),
        "Only support for coarsening ratio of 1 in all directions is included for BTD\n"
        );
    // Read list of back-transform diag parameters requested by the user //
    const amrex::ParmParse pp_diag_name(m_diag_name);

    m_file_prefix = "diags/" + m_diag_name;
    pp_diag_name.query("file_prefix", m_file_prefix);
    pp_diag_name.query("do_back_transformed_fields", m_do_back_transformed_fields);
    pp_diag_name.query("do_back_transformed_particles", m_do_back_transformed_particles);
    AMREX_ALWAYS_ASSERT(m_do_back_transformed_fields or m_do_back_transformed_particles);
    if (!m_do_back_transformed_fields) { m_varnames.clear(); }


    std::vector<std::string> intervals_string_vec = {"0"};
    bool const num_snapshots_specified = utils::parser::queryWithParser(
        pp_diag_name, "num_snapshots_lab", m_num_snapshots_lab);
    bool const intervals_specified = pp_diag_name.queryarr("intervals", intervals_string_vec);
    if (num_snapshots_specified)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!intervals_specified,
            "For back-transformed diagnostics, user should specify either num_snapshots_lab or intervals, not both");
        intervals_string_vec = {":" + std::to_string(m_num_snapshots_lab-1)};
    }
    m_intervals = utils::parser::BTDIntervalsParser(intervals_string_vec);
    m_num_buffers = m_intervals.NumSnapshots();

    // Read either dz_snapshots_lab or dt_snapshots_lab
    bool snapshot_interval_is_specified = utils::parser::queryWithParser(
        pp_diag_name, "dt_snapshots_lab", m_dt_snapshots_lab);
    if ( utils::parser::queryWithParser(pp_diag_name, "dz_snapshots_lab", m_dz_snapshots_lab) ) {
        m_dt_snapshots_lab = m_dz_snapshots_lab/WarpX::moving_window_v;
        snapshot_interval_is_specified = true;
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(snapshot_interval_is_specified,
        "For back-transformed diagnostics, user should specify either dz_snapshots_lab or dt_snapshots_lab");

    utils::parser::queryWithParser(pp_diag_name, "buffer_size", m_buffer_size);
#ifdef WARPX_DIM_RZ
    const amrex::Vector< std::string > BTD_varnames_supported = {"Er", "Et", "Ez",
                                                           "Br", "Bt", "Bz",
                                                           "jr", "jt", "jz", "rho"};
#else
    const amrex::Vector< std::string > BTD_varnames_supported = {"Ex", "Ey", "Ez",
                                                           "Bx", "By", "Bz",
                                                           "jx", "jy", "jz", "rho"};
#endif

    for (const auto& var : m_varnames) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (utils::algorithms::is_in(BTD_varnames_supported, var )),
            "Input error: field variable " + var + " in " + m_diag_name
            + ".fields_to_plot is not supported for BackTransformed diagnostics."
            + " Currently supported field variables for BackTransformed diagnostics "
            + "include Ex, Ey, Ez, Bx, By, Bz, jx, jy, jz, and rho in Cartesian coordinates and "
            + "Er, Et, Ez, Br, Bt, Bz, jr, jt, jz, and rho in cylindrical (RZ coordinates)");
    }

    const bool particle_fields_to_plot_specified = pp_diag_name.queryarr("particle_fields_to_plot", m_pfield_varnames);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!particle_fields_to_plot_specified, "particle_fields_to_plot is currently not supported for BackTransformed Diagnostics");
    if (m_varnames.empty()) {
        m_do_back_transformed_fields = false;
    }

}

bool
BTDiagnostics::DoDump (int step, int i_buffer, bool force_flush)
{

    // Do not call dump if timestep < 0, i.e., at initialization time when step == -1
    // or if the snapshot is already full and the files are closed.
    if (step >= 0 && (m_snapshot_full[i_buffer] != 1)){

        // If buffer for this lab snapshot is full then dump it and continue to collect
        // slices afterwards
        const auto is_buffer_full = buffer_full(i_buffer);
        // or
        // If last z-slice in the lab-frame snapshot is filled, call dump to
        // write the buffer and close the file.
        const auto last_z_slice_filled = (m_lastValidZSlice[i_buffer] == 1);
        // or
        // Do a forced dump at the end of the simulation, unless lab snapshot
        // was already fully written and buffer was reset to zero size or that
        // lab snapshot was not even started to be backtransformed yet
        const auto do_forced_flush = (force_flush && !buffer_empty(i_buffer));

        return is_buffer_full || last_z_slice_filled || do_forced_flush;
    }

    return false;
}


bool
BTDiagnostics::DoComputeAndPack (int step, bool force_flush)
{
    // always set to true for BTDiagnostics since back-transform buffers are potentially
    // computed and packed every timstep, except at initialization when step == -1, or when
    // force_flush is set to true, because we dont need to redundantly re-compute
    // buffers when force_flush = true. We only need to dump the buffers when
    // force_flush=true. Note that the BTD computation is performed every timestep (step>=0)
    return ( (step >=0 ) && (!force_flush) );
}

void
BTDiagnostics::InitializeBufferData ( int i_buffer , int lev, bool restart)
{
    auto & warpx = WarpX::GetInstance();

    // When restarting boosted simulations, the code below needs to take
    // into account the fact that the position of the box at the beginning
    // of the simulation, is not the one that we had at t=0 (because of the moving window)
    const amrex::Real boosted_moving_window_v = (m_moving_window_beta - m_beta_boost)
                                        / (1._rt - m_beta_boost*m_moving_window_beta);
    // Lab-frame time for the i^th snapshot
    if (!restart) {
        const amrex::Real zmax_boost = warpx.Geom(lev).ProbHi(m_moving_window_dir);
        m_t_lab.at(i_buffer) = m_intervals.GetBTDIteration(i_buffer) * m_dt_snapshots_lab
            + m_gamma_boost*m_beta_boost*zmax_boost/PhysConst::c;
            // Note: gamma_boost*beta_boost*zmax_boost is equal to
            // beta_boost*zmax_lab/(1-beta_boost*beta_moving_window)
    }

    // Define buffer domain in boosted frame at level, lev, with user-defined lo and hi
    amrex::RealBox diag_dom;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim ) {
        // Setting lo-coordinate for the diag domain by taking the max of user-defined
        // lo-cordinate and lo-coordinate of the simulation domain at level, lev
        diag_dom.setLo(idim, std::max(m_lo[idim],warpx.Geom(lev).ProbLo(idim)) );
        // Setting hi-coordinate for the diag domain by taking the max of user-defined
        // hi-cordinate and hi-coordinate of the simulation domain at level, lev
        diag_dom.setHi(idim, std::min(m_hi[idim],warpx.Geom(lev).ProbHi(idim)) );
    }
    // Initializing the m_buffer_box for the i^th snapshot.
    // At initialization, the Box has the same index space as the boosted-frame
    // As time-progresses, the z-dimension indices will be modified based on
    // current_z_lab
    amrex::IntVect lo(0);
    amrex::IntVect hi(1);
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
        // lo index with same cell-size as simulation at level, lev.
        const int lo_index = static_cast<int>( std::floor(
                ( diag_dom.lo(idim) - warpx.Geom(lev).ProbLo(idim) ) /
                  warpx.Geom(lev).CellSize(idim) ) );
        // Taking max of (0,lo_index) because lo_index must always be >=0
        lo[idim] = std::max( 0, lo_index );
        // hi index with same cell-size as simulation at level, lev.
        const int hi_index =  static_cast<int>( std::ceil(
                ( diag_dom.hi(idim) - warpx.Geom(lev).ProbLo(idim) ) /
                  warpx.Geom(lev).CellSize(idim) ) );
        // Taking max of (0,hi_index) because hi_index must always be >=0
        // Subtracting by 1 because lo,hi indices are set to cell-centered staggering.
        hi[idim] = std::max( 0, hi_index) - 1;
        // if hi<=lo, then hi = lo + 1, to ensure one cell in that dimension
        if ( hi[idim] <= lo[idim] ) {
             hi[idim]  = lo[idim] + 1;
             WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_crse_ratio[idim]==1, "coarsening ratio in reduced dimension must be 1."
             );
        }
    }
    const amrex::Box diag_box( lo, hi );
    m_buffer_box[i_buffer] = diag_box;
    m_snapshot_box[i_buffer] = diag_box;
    // Define box array
    amrex::BoxArray diag_ba(diag_box);
    diag_ba.maxSize( warpx.maxGridSize( lev ) );
    // Update the physical co-ordinates m_lo and m_hi using the final index values
    // from the coarsenable, cell-centered BoxArray, ba.
    for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        diag_dom.setLo( idim, warpx.Geom(lev).ProbLo(idim) +
            diag_ba.getCellCenteredBox(0).smallEnd(idim) * warpx.Geom(lev).CellSize(idim));
        diag_dom.setHi( idim, warpx.Geom(lev).ProbLo(idim) +
            (diag_ba.getCellCenteredBox( static_cast<int>(diag_ba.size()-1) ).bigEnd(idim) + 1) * warpx.Geom(lev).CellSize(idim));
    }

    // Define buffer_domain in lab-frame for the i^th snapshot.
    // Replace z-dimension with lab-frame co-ordinates.
    const amrex::Real zmin_buffer_lab = ( diag_dom.lo(m_moving_window_dir) - boosted_moving_window_v * warpx.gett_new(0) )
                                * (1.0_rt - m_beta_boost*m_moving_window_beta) * m_gamma_boost;
    const amrex::Real zmax_buffer_lab = ( diag_dom.hi(m_moving_window_dir) - boosted_moving_window_v * warpx.gett_new(0) )
                                * (1.0_rt - m_beta_boost*m_moving_window_beta) * m_gamma_boost;

    // Initialize buffer counter and z-positions of the  i^th snapshot in
    // boosted-frame and lab-frame
    m_buffer_counter[i_buffer] = 0;
    m_current_z_lab[i_buffer] = 0._rt;
    m_current_z_boost[i_buffer] = 0._rt;
    // store old z boost before updated zboost position
    m_old_z_boost[i_buffer] = m_current_z_boost[i_buffer];
    // Now Update Current Z Positions
    m_current_z_boost[i_buffer] = UpdateCurrentZBoostCoordinate(m_t_lab[i_buffer],
                                                              warpx.gett_new(lev) );
    m_current_z_lab[i_buffer] = UpdateCurrentZLabCoordinate(m_t_lab[i_buffer],
                                                              warpx.gett_new(lev) );

    // Compute number of cells in lab-frame required for writing Header file
    // and potentially to generate Back-Transform geometry to ensure
    // compatibility with plotfiles.
    // For the z-dimension, number of cells in the lab-frame is
    // computed using the coarsened cell-size in the lab-frame obtained using
    // the ref_ratio at level, lev-1.
    auto ref_ratio = amrex::IntVect(1);
    if (lev > 0 ) { ref_ratio = WarpX::RefRatio(lev-1); }
    // Number of lab-frame cells in z-direction at level, lev
    const int num_zcells_lab = static_cast<int>( std::floor (
                                   ( zmax_buffer_lab - zmin_buffer_lab)
                                   / dz_lab(warpx.getdt(lev), ref_ratio[m_moving_window_dir])));
    // Take the max of 0 and num_zcells_lab
    const int Nz_lab = std::max( 0, num_zcells_lab );
#if (AMREX_SPACEDIM >= 2)
    // Number of lab-frame cells in x-direction at level, lev
    const int num_xcells_lab = static_cast<int>( std::floor (
                                  ( diag_dom.hi(0) - diag_dom.lo(0) )
                                  / warpx.Geom(lev).CellSize(0)
                              ) );
    // Take the max of 0 and num_ycells_lab
    const int Nx_lab = std::max( 0, num_xcells_lab);
#endif
#if defined(WARPX_DIM_3D)
    // Number of lab-frame cells in the y-direction at level, lev
    const int num_ycells_lab = static_cast<int>( std::floor (
                                   ( diag_dom.hi(1) - diag_dom.lo(1) )
                                   / warpx.Geom(lev).CellSize(1)
                               ) );
    // Take the max of 0 and num_xcells_lab
    const int Ny_lab = std::max( 0, num_ycells_lab );
    m_snapshot_ncells_lab[i_buffer] = {Nx_lab, Ny_lab, Nz_lab};
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    m_snapshot_ncells_lab[i_buffer] = {Nx_lab, Nz_lab};
#else
    m_snapshot_ncells_lab[i_buffer] = amrex::IntVect(Nz_lab);
#endif

    // Box covering the extent of the user-defined diag in the back-transformed frame
    // for the ith snapshot
    // estimating the maximum number of buffer multifabs needed to obtain the
    // full lab-frame snapshot
    m_max_buffer_multifabs[i_buffer] = static_cast<int>( std::ceil (
        amrex::Real(m_snapshot_ncells_lab[i_buffer][m_moving_window_dir]) /
        amrex::Real(m_buffer_size) ) );
    // number of cells in z is modified since each buffer multifab always
    // contains a minimum m_buffer_size=256 cells
    const int num_z_cells_in_snapshot = m_max_buffer_multifabs[i_buffer] * m_buffer_size;
    if (!restart) {
        m_snapshot_domain_lab[i_buffer] = diag_dom;
        m_snapshot_domain_lab[i_buffer].setLo(m_moving_window_dir,
                                      zmin_buffer_lab + WarpX::moving_window_v * m_t_lab[i_buffer]);
        m_snapshot_domain_lab[i_buffer].setHi(m_moving_window_dir,
                                      zmax_buffer_lab + WarpX::moving_window_v * m_t_lab[i_buffer]);
        // To prevent round off errors, moving the snapshot domain by half a cell so that all the slices
        // lie close to the cell-centers in the lab-frame grid instead of on the edge of cell.
        const amrex::Real new_hi = m_snapshot_domain_lab[i_buffer].hi(m_moving_window_dir)
                           + 0.5_rt * dz_lab(warpx.getdt(lev), ref_ratio[m_moving_window_dir]);
        m_snapshot_domain_lab[i_buffer].setHi(m_moving_window_dir,new_hi);
        const amrex::Real new_lo = m_snapshot_domain_lab[i_buffer].hi(m_moving_window_dir) -
                             num_z_cells_in_snapshot *
                             dz_lab(warpx.getdt(lev), ref_ratio[m_moving_window_dir]);
        m_snapshot_domain_lab[i_buffer].setLo(m_moving_window_dir, new_lo);
    }

    // cell-centered index that corresponds to the hi-end of the lab-frame in the z-direction
    // Adding 0.5 dz_lab so that we obtain the cell-centered index consistent to the hi-end
    const int snapshot_kindex_hi = static_cast<int>(floor(
                             ( m_snapshot_domain_lab[i_buffer].hi(m_moving_window_dir)
                               - (m_snapshot_domain_lab[i_buffer].lo(m_moving_window_dir)
                                 + 0.5*dz_lab(warpx.getdt(lev), ref_ratio[m_moving_window_dir])
                                 )
                             ) / dz_lab(warpx.getdt(lev), ref_ratio[m_moving_window_dir]) ));
    m_snapshot_box[i_buffer].setBig( m_moving_window_dir, snapshot_kindex_hi);
    m_snapshot_box[i_buffer].setSmall( m_moving_window_dir,
                                       snapshot_kindex_hi - (num_z_cells_in_snapshot-1) );
    // Setting hi k-index for the first buffer
    if (!restart) {
        m_buffer_k_index_hi[i_buffer] = m_snapshot_box[i_buffer].bigEnd(m_moving_window_dir);
    }
}

void
BTDiagnostics::DefineCellCenteredMultiFab(int lev)
{
    if (!m_do_back_transformed_fields) { return; }
    // Creating MultiFab to store cell-centered data in boosted-frame for the entire-domain
    // This MultiFab will store all the user-requested fields in the boosted-frame
    auto & warpx = WarpX::GetInstance();
    // The BoxArray is coarsened based on the user-defined coarsening ratio
    amrex::BoxArray ba = warpx.boxArray(lev);
    ba.coarsen(m_crse_ratio);
    const amrex::DistributionMapping dmap = warpx.DistributionMap(lev);
    const int ngrow = 1;
#ifdef WARPX_DIM_RZ
    const int ncomps = WarpX::ncomps * static_cast<int>(m_cellcenter_varnames.size());
#else
    const int ncomps = static_cast<int>(m_cellcenter_varnames.size());
#endif
    bool const remake = false;
    bool const redistribute_on_remake = false;
    warpx.m_fields.alloc_init(m_cell_centered_data_name, lev, ba, dmap, ncomps, amrex::IntVect(ngrow), 0.0_rt,
                              remake, redistribute_on_remake);

}

void
BTDiagnostics::InitializeFieldFunctors (int lev)
{
    using ablastr::fields::Direction;

    // Initialize fields functors only if do_back_transformed_fields is selected
    if (!m_do_back_transformed_fields) { return; }

#ifdef WARPX_DIM_RZ
    // For RZ, initialize field functors RZ for openpmd
    // This is a specialized call for intializing cell-center functors
    // such that, all modes of a field component are stored contiguously
    // For example, Er0, Er1_real, Er1_imag, etc
    InitializeFieldFunctorsRZopenPMD(lev);
#else

    auto & warpx = WarpX::GetInstance();
    auto & fields = warpx.m_fields;

    // Clear any pre-existing vector to release stored data
    // This ensures that when domain is load-balanced, the functors point
    // to the correct field-data pointers
    m_all_field_functors[lev].clear();
    // For back-transformed data, all the components are cell-centered and stored
    // in a single multifab.
    // Therefore, size of functors at all levels is 1.
    const int num_BT_functors = 1;
    m_all_field_functors[lev].resize(num_BT_functors);
    m_cell_center_functors[lev].clear();
    m_cell_center_functors[lev].resize( m_cellcenter_varnames.size() );
    // Create an object of class BackTransformFunctor
    for (int i = 0; i < num_BT_functors; ++i)
    {
        // coarsening ratio is not provided since the source MultiFab
        // is coarsened based on the user-defined m_crse_ratio
        const int nvars = static_cast<int>(m_varnames.size());
        m_all_field_functors[lev][i] = std::make_unique<BackTransformFunctor>(
                  fields.get(m_cell_centered_data_name, lev), lev,
                  nvars, m_num_buffers, m_varnames, m_varnames_fields);
    }

    // Define all cell-centered functors required to compute cell-centere data
    // Fill vector of cell-center functors for all field-components, namely,
    // Ex, Ey, Ez, Bx, By, Bz, jx, jy, jz, and rho are included in the
    // cell-center functors for BackTransform Diags
    const auto m_cell_center_functors_at_lev_size = static_cast<int>(
        m_cell_center_functors.at(lev).size());
    for (int comp=0; comp<m_cell_center_functors_at_lev_size; comp++){
        if        ( m_cellcenter_varnames[comp] == "Ex" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::Efield_aux, Direction{0}, lev), lev, m_crse_ratio);
        } else if ( m_cellcenter_varnames[comp] == "Ey" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::Efield_aux, Direction{1}, lev), lev, m_crse_ratio);
        } else if ( m_cellcenter_varnames[comp] == "Ez" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::Efield_aux, Direction{2}, lev), lev, m_crse_ratio);
        } else if ( m_cellcenter_varnames[comp] == "Bx" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::Bfield_aux, Direction{0}, lev), lev, m_crse_ratio);
        } else if ( m_cellcenter_varnames[comp] == "By" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::Bfield_aux, Direction{1}, lev), lev, m_crse_ratio);
        } else if ( m_cellcenter_varnames[comp] == "Bz" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::Bfield_aux, Direction{2}, lev), lev, m_crse_ratio);
        } else if ( m_cellcenter_varnames[comp] == "jx" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::current_fp,Direction{0}, lev), lev, m_crse_ratio);
        } else if ( m_cellcenter_varnames[comp] == "jy" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::current_fp,Direction{1}, lev), lev, m_crse_ratio);
        } else if ( m_cellcenter_varnames[comp] == "jz" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::current_fp,Direction{2}, lev), lev, m_crse_ratio);
        } else if ( m_cellcenter_varnames[comp] == "rho" ){
            m_cell_center_functors[lev][comp] = std::make_unique<RhoFunctor>(lev, m_crse_ratio);
        }
    }

#endif
}


void
BTDiagnostics::UpdateVarnamesForRZopenPMD ()
{
#ifdef WARPX_DIM_RZ
    auto & warpx = WarpX::GetInstance();
    auto & fields = warpx.m_fields;
    using ablastr::fields::Direction;
    const int ncomp_multimodefab = fields.get(FieldType::Efield_aux, Direction{0}, 0)->nComp();
    const int ncomp = ncomp_multimodefab;


    const bool update_varnames = true;
    if (update_varnames) {
        const auto n_rz = ncomp * static_cast<int>(m_varnames_fields.size());
        m_varnames.clear();
        m_varnames.reserve(n_rz);
    }
    // AddRZ modes to output names for the back-transformed data
    if (update_varnames) {
        const auto m_varnames_fields_size = static_cast<int>(m_varnames_fields.size());
        for (int comp=0; comp<m_varnames_fields_size; comp++)
        {
            if (m_varnames_fields[comp] == "Er") {  AddRZModesToOutputNames(std::string("Er"), ncomp, false); }
            if (m_varnames_fields[comp] == "Et") {  AddRZModesToOutputNames(std::string("Et"), ncomp, false); }
            if (m_varnames_fields[comp] == "Ez") {  AddRZModesToOutputNames(std::string("Ez"), ncomp, false); }
            if (m_varnames_fields[comp] == "Br") {  AddRZModesToOutputNames(std::string("Br"), ncomp, false); }
            if (m_varnames_fields[comp] == "Bt") {  AddRZModesToOutputNames(std::string("Bt"), ncomp, false); }
            if (m_varnames_fields[comp] == "Bz") {  AddRZModesToOutputNames(std::string("Bz"), ncomp, false); }
            if (m_varnames_fields[comp] == "jr") {  AddRZModesToOutputNames(std::string("jr"), ncomp, false); }
            if (m_varnames_fields[comp] == "jt") {  AddRZModesToOutputNames(std::string("jt"), ncomp, false); }
            if (m_varnames_fields[comp] == "jz") {  AddRZModesToOutputNames(std::string("jz"), ncomp, false); }
            if (m_varnames_fields[comp] == "rho") { AddRZModesToOutputNames(std::string("rho"),ncomp, false); }
        }
    }

    // This function may be called multiple times, for different values of `lev`
    // but the `varnames` need only be updated once.
    const bool update_cellcenter_varnames = true;
    if (update_cellcenter_varnames) {
        const auto n_rz = ncomp * static_cast<int>(m_cellcenter_varnames.size());
        m_cellcenter_varnames.clear();
        m_cellcenter_varnames.reserve(n_rz);
        const auto m_cellcenter_varnames_fields_size = static_cast<int>(m_cellcenter_varnames_fields.size());
        for (int comp=0; comp<m_cellcenter_varnames_fields_size; comp++)
        {
            if ( m_cellcenter_varnames_fields[comp] == "Er" ) { AddRZModesToOutputNames(std::string("Er"), ncomp, true); }
            if ( m_cellcenter_varnames_fields[comp] == "Et" ) { AddRZModesToOutputNames(std::string("Et"), ncomp, true); }
            if ( m_cellcenter_varnames_fields[comp] == "Ez" ) { AddRZModesToOutputNames(std::string("Ez"), ncomp, true); }
            if ( m_cellcenter_varnames_fields[comp] == "Br" ) { AddRZModesToOutputNames(std::string("Br"), ncomp, true); }
            if ( m_cellcenter_varnames_fields[comp] == "Bt" ) { AddRZModesToOutputNames(std::string("Bt"), ncomp, true); }
            if ( m_cellcenter_varnames_fields[comp] == "Bz" ) { AddRZModesToOutputNames(std::string("Bz"), ncomp, true); }
            if ( m_cellcenter_varnames_fields[comp] == "jr" ) { AddRZModesToOutputNames(std::string("jr"), ncomp, true); }
            if ( m_cellcenter_varnames_fields[comp] == "jt" ) { AddRZModesToOutputNames(std::string("jt"), ncomp, true); }
            if ( m_cellcenter_varnames_fields[comp] == "jz" ) { AddRZModesToOutputNames(std::string("jz"), ncomp, true); }
            if ( m_cellcenter_varnames_fields[comp] == "rho" ) { AddRZModesToOutputNames(std::string("rho"), ncomp, true); }
        }
    }

#endif
}

void
BTDiagnostics::InitializeFieldFunctorsRZopenPMD (int lev)
{
#ifdef WARPX_DIM_RZ
    using ablastr::fields::Direction;

    auto & warpx = WarpX::GetInstance();
    auto & fields = warpx.m_fields;
    const int ncomp_multimodefab = fields.get(FieldType::Efield_aux, Direction{0}, 0)->nComp();
    const int ncomp = ncomp_multimodefab;
    // Clear any pre-existing vector to release stored data
    // This ensures that when domain is load-balanced, the functors point
    // to the correct field-data pointers
    m_all_field_functors[lev].clear();
    // For back-transformed data, all the components are cell-centered and stored
    // in a single MultiFab.
    // Therefore, size of functors at all levels is 1
    const int num_BT_functors = 1;
    m_all_field_functors[lev].resize(num_BT_functors);
    for (int i = 0; i < num_BT_functors; ++i) {
        const int nvars = static_cast<int>(m_varnames.size());
        m_all_field_functors[lev][i] = std::make_unique<BackTransformFunctor>(
                                       fields.get(m_cell_centered_data_name, lev), lev,
                                       nvars, m_num_buffers, m_varnames,
                                       m_varnames_fields);
    }

    // Reset field functors for cell-center multifab
    m_cell_center_functors[lev].clear();
    m_cell_center_functors[lev].resize(m_cellcenter_varnames_fields.size());

    const auto m_cell_center_functors_at_lev_size = static_cast<int>(m_cell_center_functors.at(lev).size());
    for (int comp=0; comp<m_cell_center_functors_at_lev_size; comp++){
        if        ( m_cellcenter_varnames_fields[comp] == "Er" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::Efield_aux, Direction{0}, lev), lev, m_crse_ratio, false, ncomp);
        } else if ( m_cellcenter_varnames_fields[comp] == "Et" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::Efield_aux, Direction{1}, lev), lev, m_crse_ratio, false, ncomp);
        } else if ( m_cellcenter_varnames_fields[comp] == "Ez" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::Efield_aux, Direction{2}, lev), lev, m_crse_ratio, false, ncomp);
        } else if ( m_cellcenter_varnames_fields[comp] == "Br" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::Bfield_aux, Direction{0}, lev), lev, m_crse_ratio, false, ncomp);
        } else if ( m_cellcenter_varnames_fields[comp] == "Bt" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::Bfield_aux, Direction{1}, lev), lev, m_crse_ratio, false, ncomp);
        } else if ( m_cellcenter_varnames_fields[comp] == "Bz" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::Bfield_aux, Direction{2}, lev), lev, m_crse_ratio, false, ncomp);
        } else if ( m_cellcenter_varnames_fields[comp] == "jr" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::current_fp, Direction{0}, lev), lev, m_crse_ratio, false, ncomp);
        } else if ( m_cellcenter_varnames_fields[comp] == "jt" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::current_fp, Direction{1}, lev), lev, m_crse_ratio, false, ncomp);
        } else if ( m_cellcenter_varnames_fields[comp] == "jz" ){
            m_cell_center_functors[lev][comp] = std::make_unique<CellCenterFunctor>(fields.get(FieldType::current_fp, Direction{2}, lev), lev, m_crse_ratio, false, ncomp);
        } else if ( m_cellcenter_varnames_fields[comp] == "rho" ){
            m_cell_center_functors[lev][comp] = std::make_unique<RhoFunctor>(lev, m_crse_ratio, false, -1, false, ncomp);
        }
    }

#endif
    amrex::ignore_unused(lev);
}

void
BTDiagnostics::AddRZModesToOutputNames (const std::string& field, const int ncomp, bool cellcenter_data)
{
#ifdef WARPX_DIM_RZ
    // In cylindrical geometry, real and imag part of each mode are also
    // dumped to file separately, so they need to be added to m_varnames
    // we number modes from 0 to (nmodes-1);
    // mode 0 is purely real, all higher modes are complex
    int const nmodes = (ncomp+1)/2;

    if (cellcenter_data) {
        m_cellcenter_varnames.push_back( field + "_0_real" );
        for (int ic=1; ic < nmodes; ic++) {
            m_cellcenter_varnames.push_back( field + "_" + std::to_string(ic) + "_real" );
            m_cellcenter_varnames.push_back( field + "_" + std::to_string(ic) + "_imag" );
        }
    } else {
        m_varnames.push_back(field + "_0_real");
        for (int ic=1; ic < nmodes; ic++) {
            m_varnames.push_back( field + "_" + std::to_string(ic) + "_real" );
            m_varnames.push_back( field + "_" + std::to_string(ic) + "_imag" );
        }
    }
#else
    amrex::ignore_unused(field, ncomp, cellcenter_data);
#endif
}

void
BTDiagnostics::PrepareBufferData ()
{
    auto & warpx = WarpX::GetInstance();
    const int num_BT_functors = 1;

    for (int lev = 0; lev < nlev_output; ++lev)
    {
        for (int i = 0; i < num_BT_functors; ++i)
        {
            for (int i_buffer = 0; i_buffer < m_num_buffers; ++i_buffer )
            {
                m_old_z_boost[i_buffer] = m_current_z_boost[i_buffer];
                // Update z-boost and z-lab positions
                m_current_z_boost[i_buffer] = UpdateCurrentZBoostCoordinate(m_t_lab[i_buffer],
                                                                      warpx.gett_new(lev) );
                m_current_z_lab[i_buffer] = UpdateCurrentZLabCoordinate(m_t_lab[i_buffer],
                                                                      warpx.gett_new(lev) );
            }
        }
    }
}

void
BTDiagnostics::UpdateBufferData ()
{
    const int num_BT_functors = 1;

    for (int lev = 0; lev < nlev_output; ++lev)
    {
        for (int i = 0; i < num_BT_functors; ++i)
        {
            for (int i_buffer = 0; i_buffer < m_num_buffers; ++i_buffer )
            {
                const bool ZSliceInDomain = GetZSliceInDomainFlag (i_buffer, lev);
                if (ZSliceInDomain) { ++m_buffer_counter[i_buffer]; }
                // when the z-index is equal to the smallEnd of the snapshot box, then set lastValidZSlice to 1
                if (k_index_zlab(i_buffer, lev) == m_snapshot_box[i_buffer].smallEnd(m_moving_window_dir)) {
                    m_lastValidZSlice[i_buffer] = 1;
                }
            }
        }
   }
}

void
BTDiagnostics::PrepareFieldDataForOutput ()
{
    // Initialize fields functors only if do_back_transformed_fields is selected
    if (!m_do_back_transformed_fields) { return; }

    auto & warpx = WarpX::GetInstance();
    auto & fields = warpx.m_fields;

    // In this function, we will get cell-centered data for every level, lev,
    // using the cell-center functors and their respective operators()
    // Call m_cell_center_functors->operator
    for (int lev = 0; lev < nmax_lev; ++lev) {
        int icomp_dst = 0;
        const auto m_cell_center_fuctors_at_lev_size = static_cast<int>(m_cell_center_functors.at(lev).size());
        for (int icomp = 0; icomp<m_cell_center_fuctors_at_lev_size; ++icomp) {
            // Call all the cell-center functors in m_cell_center_functors.
            // Each of them computes cell-centered data for a field and
            // stores it in cell-centered MultiFab.
            m_cell_center_functors[lev][icomp]->operator()(*fields.get(m_cell_centered_data_name, lev), icomp_dst);
            icomp_dst += m_cell_center_functors[lev][icomp]->nComp();
        }
        // Check that the proper number of user-requested components are cell-centered
        AMREX_ALWAYS_ASSERT( icomp_dst == m_cellcenter_varnames.size() );
        // fill boundary call is required to average_down (flatten) data to
        // the coarsest level.
        ablastr::utils::communication::FillBoundary(*fields.get(m_cell_centered_data_name, lev),
                                                    WarpX::do_single_precision_comms,
                                                    warpx.Geom(lev).periodicity());
    }
    // Flattening out MF over levels

    for (int lev = warpx.finestLevel(); lev > 0; --lev) {
        ablastr::coarsen::sample::Coarsen(*fields.get(m_cell_centered_data_name, lev - 1),
                                          *fields.get(m_cell_centered_data_name, lev), 0, 0,
                                          static_cast<int>(m_cellcenter_varnames.size()), 0, WarpX::RefRatio(lev-1) );
    }

    const int num_BT_functors = 1;
    for (int lev = 0; lev < nlev_output; ++lev)
    {
        for (int i = 0; i < num_BT_functors; ++i)
        {
            for (int i_buffer = 0; i_buffer < m_num_buffers; ++i_buffer )
            {
                // Check if the zslice is in domain
                const bool ZSliceInDomain = GetZSliceInDomainFlag (i_buffer, lev);
                // Initialize and define field buffer multifab if buffer is empty
                const bool kindexInSnapshotBox = GetKIndexInSnapshotBoxFlag (i_buffer, lev);
                if (kindexInSnapshotBox) {
                    if ( buffer_empty(i_buffer) ) {
                        if ( m_buffer_flush_counter[i_buffer] == 0 || m_first_flush_after_restart[i_buffer] == 1) {
                            // Compute the geometry, snapshot lab-domain extent
                            // and box-indices
                            DefineSnapshotGeometry(i_buffer, lev);
                        }
                        DefineFieldBufferMultiFab(i_buffer, lev);
                    }
                }
                if (ZSliceInDomain) {
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        m_current_z_lab[i_buffer] >= m_buffer_domain_lab[i_buffer].lo(m_moving_window_dir) and
                        m_current_z_lab[i_buffer] <= m_buffer_domain_lab[i_buffer].hi(m_moving_window_dir),
                        "z-slice in lab-frame (" +
                        std::to_string(m_current_z_lab[i_buffer]) +
                        ") is outside the buffer domain physical extent (" +
                        std::to_string(m_buffer_domain_lab[i_buffer].lo(m_moving_window_dir)) +
                        " to " +
                        std::to_string(m_buffer_domain_lab[i_buffer].hi(m_moving_window_dir)) +
                        ")."
                    );
                }
                m_all_field_functors[lev][i]->PrepareFunctorData (
                                             i_buffer, ZSliceInDomain,
                                             m_current_z_boost[i_buffer],
                                             m_buffer_box[i_buffer],
                                             k_index_zlab(i_buffer, lev),
                                             m_snapshot_full[i_buffer] );

            }
        }
    }
}


amrex::Real
BTDiagnostics::dz_lab (amrex::Real dt, amrex::Real ref_ratio) const
{
    return PhysConst::c * dt * 1._rt/m_beta_boost * 1._rt/m_gamma_boost * 1._rt/ref_ratio;
}


int
BTDiagnostics::k_index_zlab (int i_buffer, int lev) const
{
    auto & warpx = WarpX::GetInstance();
    const amrex::Real prob_domain_zmin_lab = m_snapshot_domain_lab[i_buffer].lo( m_moving_window_dir );
    auto ref_ratio = amrex::IntVect(1);
    if (lev > 0 ) { ref_ratio = WarpX::RefRatio(lev-1); }
    const int k_lab = static_cast<int>(std::floor (
                          ( m_current_z_lab[i_buffer]
                            - (prob_domain_zmin_lab  ) )
                          / dz_lab( warpx.getdt(lev), ref_ratio[m_moving_window_dir] )
                      ) ) + m_snapshot_box[i_buffer].smallEnd(m_moving_window_dir);
    return k_lab;
}

void
BTDiagnostics::SetSnapshotFullStatus (const int i_buffer)
{
    if (m_snapshot_full[i_buffer] == 1) { return; }
    // if the last valid z-index of the snapshot, which is 0, is filled, then
    // set the snapshot full integer to 1
    if (m_lastValidZSlice[i_buffer] == 1) { m_snapshot_full[i_buffer] = 1; }
}

void
BTDiagnostics::DefineFieldBufferMultiFab (const int i_buffer, const int lev)
{
    if (m_field_buffer_multifab_defined[i_buffer] == 1) { return; }
    auto & warpx = WarpX::GetInstance();

    const int hi_k_lab = m_buffer_k_index_hi[i_buffer];
    m_buffer_box[i_buffer].setSmall( m_moving_window_dir, hi_k_lab - m_buffer_size + 1);
    m_buffer_box[i_buffer].setBig( m_moving_window_dir, hi_k_lab );
    const amrex::BoxArray buffer_ba( m_buffer_box[i_buffer] );
    // Generate a new distribution map for the back-transformed buffer multifab
    const amrex::DistributionMapping buffer_dmap(buffer_ba);
    // Number of guard cells for the output buffer is zero.
    // Unlike FullDiagnostics, "m_format == sensei" option is not included here.
    const int ngrow = 0;
    m_mf_output[i_buffer][lev] = amrex::MultiFab( buffer_ba, buffer_dmap,
                                              static_cast<int>(m_varnames.size()), ngrow );
    m_mf_output[i_buffer][lev].setVal(0.);

    auto ref_ratio = amrex::IntVect(1);
    if (lev > 0 ) { ref_ratio = WarpX::RefRatio(lev-1); }
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        const amrex::Real cellsize = (idim < WARPX_ZINDEX)?
            warpx.Geom(lev).CellSize(idim):
            dz_lab(warpx.getdt(lev), ref_ratio[m_moving_window_dir]);
        const amrex::Real buffer_lo = m_snapshot_domain_lab[i_buffer].lo(idim)
                                + ( buffer_ba.getCellCenteredBox(0).smallEnd(idim)
                                  - m_snapshot_box[i_buffer].smallEnd(idim)
                                  ) * cellsize;
        const amrex::Real buffer_hi = m_snapshot_domain_lab[i_buffer].lo(idim)
                                + ( buffer_ba.getCellCenteredBox( static_cast<int>(buffer_ba.size()-1) ).bigEnd(idim)
                                  - m_snapshot_box[i_buffer].smallEnd(idim)
                                  + 1 ) * cellsize;
        m_buffer_domain_lab[i_buffer].setLo(idim, buffer_lo);
        m_buffer_domain_lab[i_buffer].setHi(idim, buffer_hi);
    }

    // Define the geometry object at level, lev, for the ith buffer.
    if (lev == 0) {
        // The extent of the physical domain covered by the ith buffer mf, m_mf_output
        // Default non-periodic geometry for diags
        amrex::Vector<int> BTdiag_periodicity(AMREX_SPACEDIM, 0);
        // Box covering the extent of the user-defined diag in the back-transformed frame
        const amrex::Box domain = buffer_ba.minimalBox();
        // define the geometry object for the ith buffer using Physical co-ordinates
        // of m_buffer_domain_lab[i_buffer].
        m_geom_output[i_buffer][lev].define( domain, &m_buffer_domain_lab[i_buffer],
                                             amrex::CoordSys::cartesian,
                                             BTdiag_periodicity.data() );
    } else if (lev > 0 ) {
        // Refine the geometry object defined at the previous level, lev-1
        m_geom_output[i_buffer][lev] = amrex::refine( m_geom_output[i_buffer][lev-1],
                                                      WarpX::RefRatio(lev-1) );
    }
    m_field_buffer_multifab_defined[i_buffer] = 1;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( m_mf_output[i_buffer][lev].boxArray().size() == 1,
        "BoxArray size must be 1 for back-transformed diagnostics multifab that stores buffers");
}


void
BTDiagnostics::DefineSnapshotGeometry (const int i_buffer, const int lev)
{
    if (m_snapshot_geometry_defined[i_buffer] == 1) { return; }

    if (lev == 0) {
        // Default non-periodic geometry for diags
        amrex::Vector<int> BTdiag_periodicity(AMREX_SPACEDIM, 0);
        // Define the geometry object for the ith snapshot using Physical co-ordinates
        // of m_snapshot_domain_lab[i_buffer], that corresponds to the full snapshot
        // in the back-transformed frame
        m_geom_snapshot[i_buffer][lev].define( m_snapshot_box[i_buffer],
                                               &m_snapshot_domain_lab[i_buffer],
                                               amrex::CoordSys::cartesian,
                                               BTdiag_periodicity.data() );
    } else if (lev > 0) {
        // Refine the geometry object defined at the previous level, lev-1
        m_geom_snapshot[i_buffer][lev] = amrex::refine( m_geom_snapshot[i_buffer][lev-1],
                                                        WarpX::RefRatio(lev-1) );
    }
    m_snapshot_geometry_defined[i_buffer] = 1;
}

bool
BTDiagnostics::GetZSliceInDomainFlag (const int i_buffer, const int lev)
{
    auto & warpx = WarpX::GetInstance();
    const amrex::RealBox& boost_domain = warpx.Geom(lev).ProbDomain();
    const amrex::Real boost_cellsize = warpx.Geom(lev).CellSize(m_moving_window_dir);
    const amrex::Real buffer_zmin_lab = m_snapshot_domain_lab[i_buffer].lo( m_moving_window_dir );
    const amrex::Real buffer_zmax_lab = m_snapshot_domain_lab[i_buffer].hi( m_moving_window_dir );

    // Exclude 0.5*boost_cellsize from the edge, to avoid that the interpolation to
    // cell centers uses data from the guard cells.
    const bool slice_not_in_domain =
        ( m_current_z_boost[i_buffer] <= boost_domain.lo(m_moving_window_dir) + 0.5_rt*boost_cellsize) ||
        ( m_current_z_boost[i_buffer] >= boost_domain.hi(m_moving_window_dir) - 0.5_rt*boost_cellsize) ||
        ( m_current_z_lab[i_buffer] <= buffer_zmin_lab ) ||
        ( m_current_z_lab[i_buffer] >= buffer_zmax_lab );

    return !slice_not_in_domain;
}


bool
BTDiagnostics::GetKIndexInSnapshotBoxFlag (const int i_buffer, const int lev)
{
    return (k_index_zlab(i_buffer, lev) >= m_snapshot_box[i_buffer].smallEnd(m_moving_window_dir) &&
        k_index_zlab(i_buffer, lev) <= m_snapshot_box[i_buffer].bigEnd(m_moving_window_dir));
}

void
BTDiagnostics::Flush (int i_buffer, bool force_flush)
{
    auto & warpx = WarpX::GetInstance();
    std::string file_name = m_file_prefix;
    if (m_format=="plotfile") {
        file_name = amrex::Concatenate(m_file_prefix, i_buffer, m_file_min_digits);
        file_name = file_name+"/buffer";
    }
    SetSnapshotFullStatus(i_buffer);
    const bool isLastBTDFlush = ( m_snapshot_full[i_buffer] == 1 ) || force_flush;
    bool const use_pinned_pc = true;
    bool const isBTD = true;
    double const labtime = m_t_lab[i_buffer];

    amrex::Vector<amrex::BoxArray> vba;
    amrex::Vector<amrex::DistributionMapping> vdmap;
    amrex::Vector<amrex::Geometry> vgeom;
    amrex::Vector<amrex::IntVect> vrefratio;
    if (!m_particles_buffer.at(i_buffer).empty()) {
        const int nlevels = m_particles_buffer[i_buffer][0]->numLevels();
        for (int lev = 0 ; lev < nlevels; ++lev) {
            // Store BoxArray, dmap, geometry, and refratio for every level
            vba.push_back(m_particles_buffer[i_buffer][0]->ParticleBoxArray(lev));
            vdmap.push_back(m_particles_buffer[i_buffer][0]->ParticleDistributionMap(lev));
            vgeom.push_back(m_particles_buffer[i_buffer][0]->ParticleGeom(lev));
            if (lev < nlevels - 1) {
                vrefratio.push_back(m_particles_buffer[i_buffer][0]->GetParGDB()->refRatio(lev));
            }
        }
        // Redistribute particles in the lab frame box arrays that correspond to the buffer
        // Prior to redistribute, increase buffer box and Box in ParticleBoxArray by 1 index in the
        // lo and hi-end, so particles can be binned in the boxes correctly.
        // For BTD, we may have particles that are out of the domain by half a cell-size or one cell size.
        // As a result, the index they correspond to may be out of the box by one index
        // As a work around to the locateParticle error in Redistribute, we increase the box size before
        // redistribute and shrink it after the call to redistribute.
        m_buffer_box[i_buffer].setSmall(m_moving_window_dir, (m_buffer_box[i_buffer].smallEnd(m_moving_window_dir) - 1) );
        m_buffer_box[i_buffer].setBig(m_moving_window_dir, (m_buffer_box[i_buffer].bigEnd(m_moving_window_dir) + 1) );
        const amrex::Box particle_buffer_box = m_buffer_box[i_buffer];
        const amrex::BoxArray buffer_ba( particle_buffer_box );
        m_particles_buffer[i_buffer][0]->SetParticleBoxArray(0, buffer_ba);
        for (int isp = 0; isp < m_particles_buffer.at(i_buffer).size(); ++isp) {
            // BTD output is single level. Setting particle geometry, dmap, boxarray to level0
            m_particles_buffer[i_buffer][isp]->SetParGDB(vgeom[0], vdmap[0], buffer_ba);
        }
    }
    RedistributeParticleBuffer(i_buffer);

    // Reset buffer box and particle box array
    if (m_format == "openpmd") {
        if (!m_particles_buffer.at(i_buffer).empty()) {
            m_buffer_box[i_buffer].setSmall(m_moving_window_dir, (m_buffer_box[i_buffer].smallEnd(m_moving_window_dir) + 1) );
            m_buffer_box[i_buffer].setBig(m_moving_window_dir, (m_buffer_box[i_buffer].bigEnd(m_moving_window_dir) - 1) );
            m_particles_buffer[i_buffer][0]->SetParticleBoxArray(0,vba.back());
            for (int isp = 0; isp < m_particles_buffer.at(i_buffer).size(); ++isp) {
                // BTD output is single level. Setting particle geometry, dmap, boxarray to level0
                m_particles_buffer[i_buffer][isp]->SetParGDB(vgeom[0], vdmap[0], vba.back());
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE( m_particles_buffer[i_buffer][isp]->ParticleBoxArray(0).size() == 1,
                    "ParticleBoxArray size must be 1 for back-transformed diagnostic particle buffer");
            }
        }
    }
    m_flush_format->WriteToFile(
        m_varnames, m_mf_output.at(i_buffer), m_geom_output.at(i_buffer), warpx.getistep(),
        labtime,
        m_output_species.at(i_buffer), nlev_output, file_name, m_file_min_digits,
        m_plot_raw_fields, m_plot_raw_fields_guards, m_verbose,
        use_pinned_pc, isBTD, i_buffer, m_buffer_flush_counter.at(i_buffer),
        m_max_buffer_multifabs.at(i_buffer), m_geom_snapshot.at(i_buffer).at(0), isLastBTDFlush);

    // Rescaling the box for plotfile after WriteToFile. This is because, for plotfiles, when writing particles, amrex checks if the particles are within the bounds defined by the box. However, in BTD, particles can be (at max) 1 cell outside the bounds of the geometry. So we keep a one-cell bigger box for plotfile when writing out the particle data and rescale after.
    if (m_format == "plotfile") {
        if (!m_particles_buffer.at(i_buffer).empty()) {
            m_buffer_box[i_buffer].setSmall(m_moving_window_dir, (m_buffer_box[i_buffer].smallEnd(m_moving_window_dir) + 1) );
            m_buffer_box[i_buffer].setBig(m_moving_window_dir, (m_buffer_box[i_buffer].bigEnd(m_moving_window_dir) - 1) );
            m_particles_buffer[i_buffer][0]->SetParticleBoxArray(0,vba.back());
            for (int isp = 0; isp < m_particles_buffer.at(i_buffer).size(); ++isp) {
                // BTD output is single level. Setting particle geometry, dmap, boxarray to level0
                m_particles_buffer[i_buffer][isp]->SetParGDB(vgeom[0], vdmap[0], vba.back());
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE( m_particles_buffer[i_buffer][isp]->ParticleBoxArray(0).size() == 1,
                    "ParticleBoxArray size must be 1 for back-transformed diagnostic particle buffer");
            }
        }
    }

    for (int isp = 0; isp < m_particles_buffer.at(i_buffer).size(); ++isp) {
        // Buffer particle container reset to include geometry, dmap, Boxarray, and refratio
        // so that particles from finest level can also be selected and transformed
        m_particles_buffer[i_buffer][isp]->SetParGDB(vgeom, vdmap, vba, vrefratio);
    }

    if (m_format == "plotfile") {
        MergeBuffersForPlotfile(i_buffer);
    }

    // Reset the buffer counter to zero after flushing out data stored in the buffer.
    ResetBufferCounter(i_buffer);
    m_field_buffer_multifab_defined[i_buffer] = 0;
    IncrementBufferFlushCounter(i_buffer);
    NullifyFirstFlush(i_buffer);
    // if particles are selected for output then update and reset counters
    if (!m_output_species_names.empty()) {
        ResetTotalParticlesInBuffer(i_buffer);
        ClearParticleBuffer(i_buffer);
    }
    // Setting hi k-index for the next buffer, such that, the index is one less than the lo-index of previous buffer
    // For example, for buffer size of 256, if the first buffer extent was [256,511]
    // then the next buffer will be from [0,255]. That is, the hi-index of the following buffer is 256-1
    m_buffer_k_index_hi[i_buffer] = m_buffer_box[i_buffer].smallEnd(m_moving_window_dir) - 1;
}

void BTDiagnostics::RedistributeParticleBuffer (const int i_buffer)
{
    for (int isp = 0; isp < m_particles_buffer.at(i_buffer).size(); ++isp) {
        m_particles_buffer[i_buffer][isp]->Redistribute();
    }
}

void BTDiagnostics::MergeBuffersForPlotfile (int i_snapshot)
{
    // Make sure all MPI ranks wrote their files and closed it
    // Note: additionally, since a Barrier does not guarantee a FS sync
    //       on a parallel FS, we might need to add timeouts and retries
    //       to the open calls below when running at scale.
    amrex::ParallelDescriptor::Barrier();

    auto & warpx = WarpX::GetInstance();
    const amrex::Vector<int> iteration = warpx.getistep();
    // number of digits for plotfile containing multifab data (Cell_D_XXXXX)
    // the digits here are "multifab ids" (independent of the step) and thus always small
    const int amrex_fabfile_digits = 5;
    // number of digits for plotfile containing particle data (DATA_XXXXX)
    // the digits here are fab ids that the particles belong to (independent of the step) and thus always small
    const int amrex_partfile_digits = 5;
    if (amrex::ParallelContext::IOProcessorSub()) {
        // Path to final snapshot plotfiles
        const std::string snapshot_path = amrex::Concatenate(m_file_prefix, i_snapshot, m_file_min_digits);
        // BTD plotfile have only one level, Level0.
        const std::string snapshot_Level0_path = snapshot_path + "/Level_0";
        const std::string snapshot_Header_filename = snapshot_path + "/Header";
        // Path of the buffer recently flushed
        const std::string BufferPath_prefix = snapshot_path + "/buffer";
        const std::string recent_Buffer_filepath = amrex::Concatenate(BufferPath_prefix,iteration[0], m_file_min_digits);
        // Header file of the recently flushed buffer
        const std::string recent_Header_filename = recent_Buffer_filepath+"/Header";
        const std::string recent_Buffer_Level0_path = recent_Buffer_filepath + "/Level_0";
        const std::string recent_Buffer_FabHeaderFilename = recent_Buffer_Level0_path + "/Cell_H";
        // Create directory only when the first buffer is flushed out.
        if (m_buffer_flush_counter[i_snapshot] == 0 || m_first_flush_after_restart[i_snapshot] == 1) {
            // Create Level_0 directory to store all Cell_D and Cell_H files
            if (!amrex::UtilCreateDirectory(snapshot_Level0_path, permission_flag_rwxrxrx) ) {
                amrex::CreateDirectoryFailed(snapshot_Level0_path);
            }
            // Create directory for each species selected for diagnostic
            for (int i = 0; i < m_particles_buffer[i_snapshot].size(); ++i) {
                const std::string snapshot_species_path = snapshot_path + "/" + m_output_species_names[i];
                if ( !amrex::UtilCreateDirectory(snapshot_species_path, permission_flag_rwxrxrx)) {
                    amrex::CreateDirectoryFailed(snapshot_species_path);
                }
                // Create Level_0 directory for particles to store Particle_H and DATA files
                const std::string species_Level0_path = snapshot_species_path + "/Level_0";
                if ( !amrex::UtilCreateDirectory(species_Level0_path, permission_flag_rwxrxrx)) {
                    amrex::CreateDirectoryFailed(species_Level0_path);
                }
            }
            const std::string buffer_WarpXHeader_path = recent_Buffer_filepath + "/WarpXHeader";
            const std::string snapshot_WarpXHeader_path = snapshot_path + "/WarpXHeader";
            const std::string buffer_job_info_path = recent_Buffer_filepath + "/warpx_job_info";
            const std::string snapshot_job_info_path = snapshot_path + "/warpx_job_info";
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::rename(buffer_WarpXHeader_path.c_str(), snapshot_WarpXHeader_path.c_str()) == 0,
                std::string("Renaming ").append(buffer_WarpXHeader_path).append(" to ").append(snapshot_WarpXHeader_path).append(" has failed"));
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::rename(buffer_job_info_path.c_str(), snapshot_job_info_path.c_str()) == 0,
                std::string("Renaming ").append(buffer_job_info_path).append(" to ").append(snapshot_job_info_path).append(" has failed"));
        }

        if (m_do_back_transformed_fields) {
            // Read the header file to get the fab on disk string
            BTDMultiFabHeaderImpl Buffer_FabHeader(recent_Buffer_FabHeaderFilename);
            Buffer_FabHeader.ReadMultiFabHeader();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                Buffer_FabHeader.ba_size() <= 1,
                "BTD Buffer has more than one fabs."
            );
            // Every buffer that is flushed only has a single fab.
            const std::string recent_Buffer_FabFilename = recent_Buffer_Level0_path + "/"
                                                  + Buffer_FabHeader.FabName(0);
            // Existing snapshot Fab Header Filename
            // Cell_D_<number> is padded with 5 zeros as that is the default AMReX output
            // The number is the multifab ID here.
            const std::string snapshot_FabHeaderFilename = snapshot_Level0_path + "/Cell_H";
            const std::string snapshot_FabFilename = amrex::Concatenate(snapshot_Level0_path+"/Cell_D_", m_buffer_flush_counter[i_snapshot], amrex_fabfile_digits);
            // Name of the newly appended fab in the snapshot
            // Cell_D_<number> is padded with 5 zeros as that is the default AMReX output
            const std::string new_snapshotFabFilename = amrex::Concatenate("Cell_D_", m_buffer_flush_counter[i_snapshot], amrex_fabfile_digits);

            if (m_buffer_flush_counter[i_snapshot] == 0 || m_first_flush_after_restart[i_snapshot] == 1) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    std::rename(recent_Header_filename.c_str(), snapshot_Header_filename.c_str()) == 0,
                    std::string("Renaming ").append(recent_Header_filename).append(" to ").append(snapshot_Header_filename).append(" has failed"));
                Buffer_FabHeader.SetFabName(0, Buffer_FabHeader.fodPrefix(0),
                                            new_snapshotFabFilename,
                                            Buffer_FabHeader.FabHead(0));
                Buffer_FabHeader.WriteMultiFabHeader();
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    std::rename(recent_Buffer_FabHeaderFilename.c_str(), snapshot_FabHeaderFilename.c_str()) == 0,
                    std::string("Renaming ").append(recent_Buffer_FabHeaderFilename).append(" to ").append(snapshot_FabHeaderFilename).append(" has failed"));
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    std::rename(recent_Buffer_FabFilename.c_str(), snapshot_FabFilename.c_str()) == 0,
                    std::string("Renaming ").append(recent_Buffer_FabFilename).append(" to ").append(snapshot_FabFilename).append(" has failed"));
            } else {
                // Interleave Header file
                InterleaveBufferAndSnapshotHeader(recent_Header_filename,
                                                  snapshot_Header_filename);
                InterleaveFabArrayHeader(recent_Buffer_FabHeaderFilename,
                                         snapshot_FabHeaderFilename,
                                         new_snapshotFabFilename);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    std::rename(recent_Buffer_FabFilename.c_str(), snapshot_FabFilename.c_str()) == 0,
                    std::string("Renaming ").append(recent_Buffer_FabFilename).append(" to ").append(snapshot_FabFilename).append(" has failed"));
            }
        }
        for (int i = 0; i < m_particles_buffer[i_snapshot].size(); ++i) {
            // species filename of recently flushed buffer
            const std::string recent_species_prefix = recent_Buffer_filepath+"/"+m_output_species_names[i];
            const std::string recent_species_Header = recent_species_prefix + "/Header";
            const std::string recent_ParticleHdrFilename = recent_species_prefix + "/Level_0/Particle_H";
            BTDSpeciesHeaderImpl BufferSpeciesHeader(recent_species_Header,
                                                     m_output_species_names[i]);
            BufferSpeciesHeader.ReadHeader();
            // only one box is flushed out at a time
            // DATA_<number> is padded with 5 zeros as that is the default AMReX output for plotfile
            // The number is the ID of the multifab that the particles belong to.
            const std::string recent_ParticleDataFilename = amrex::Concatenate(
                recent_species_prefix + "/Level_0/DATA_",
                BufferSpeciesHeader.m_which_data[0][0],
                amrex_partfile_digits);
            // Path to snapshot particle files
            const std::string snapshot_species_path = snapshot_path + "/" + m_output_species_names[i];
            const std::string snapshot_species_Level0path = snapshot_species_path + "/Level_0";
            const std::string snapshot_species_Header = snapshot_species_path + "/Header";
            const std::string snapshot_ParticleHdrFilename = snapshot_species_Level0path + "/Particle_H";
            const std::string snapshot_ParticleDataFilename = amrex::Concatenate(
                snapshot_species_Level0path + "/DATA_",
                m_buffer_flush_counter[i_snapshot],
                amrex_partfile_digits);

            if (m_buffer_flush_counter[i_snapshot] == 0 || m_first_flush_after_restart[i_snapshot] == 1) {
                BufferSpeciesHeader.set_DataIndex(0,0,m_buffer_flush_counter[i_snapshot]);
                BufferSpeciesHeader.WriteHeader();

                // copy Header file for the species
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    std::rename(recent_species_Header.c_str(), snapshot_species_Header.c_str()) == 0,
                    std::string("Renaming ").append(recent_species_Header).append(" to ").append(snapshot_species_Header).append(" has failed"));
                if (BufferSpeciesHeader.m_total_particles == 0) { continue; }
                // if finite number of particles in the output, copy ParticleHdr and Data file
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    std::rename(recent_ParticleHdrFilename.c_str(), snapshot_ParticleHdrFilename.c_str()) == 0,
                    std::string("Renaming ").append(recent_ParticleHdrFilename).append(" to ").append(snapshot_ParticleHdrFilename).append(" has failed"));
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    std::rename(recent_ParticleDataFilename.c_str(), snapshot_ParticleDataFilename.c_str()) == 0,
                    std::string("Renaming ").append(recent_ParticleDataFilename).append(" to ").append(snapshot_ParticleDataFilename).append(" has failed"));
            } else {
                InterleaveSpeciesHeader(recent_species_Header,snapshot_species_Header,
                                        m_output_species_names[i], m_buffer_flush_counter[i_snapshot]);
                if (BufferSpeciesHeader.m_total_particles == 0) { continue; }
                if (!amrex::FileExists(snapshot_ParticleHdrFilename)) {
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        std::rename(recent_ParticleHdrFilename.c_str(), snapshot_ParticleHdrFilename.c_str()) == 0,
                        std::string("Renaming ").append(recent_ParticleHdrFilename).append(" to ").append(snapshot_ParticleHdrFilename).append(" has failed"));
                } else {
                    InterleaveParticleDataHeader(recent_ParticleHdrFilename,
                                                 snapshot_ParticleHdrFilename);
                }
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    std::rename(recent_ParticleDataFilename.c_str(), snapshot_ParticleDataFilename.c_str()) == 0,
                    std::string("Renaming ").append(recent_ParticleDataFilename).append(" to ").append(snapshot_ParticleDataFilename).append(" has failed"));
            }
        }
        // Destroying the recently flushed buffer directory since it is already merged.
        amrex::FileSystem::RemoveAll(recent_Buffer_filepath);

    } // ParallelContext if ends
    amrex::ParallelDescriptor::Barrier();
}

void
BTDiagnostics::InterleaveBufferAndSnapshotHeader ( const std::string& buffer_Header_path,
                                                   const std::string& snapshot_Header_path)
{
    BTDPlotfileHeaderImpl snapshot_HeaderImpl(snapshot_Header_path);
    snapshot_HeaderImpl.ReadHeaderData();

    BTDPlotfileHeaderImpl buffer_HeaderImpl(buffer_Header_path);
    buffer_HeaderImpl.ReadHeaderData();

    // Update step_scraped of snapshot with recently flushed buffer
    snapshot_HeaderImpl.set_time( buffer_HeaderImpl.time() );
    snapshot_HeaderImpl.set_timestep( buffer_HeaderImpl.timestep() );

    const amrex::Box snapshot_Box = snapshot_HeaderImpl.probDomain();
    const amrex::Box buffer_Box = buffer_HeaderImpl.probDomain();
    amrex::IntVect box_lo(0);
    amrex::IntVect box_hi(1);
    // Update prob_lo with min of buffer and snapshot
    for (int idim = 0; idim < snapshot_HeaderImpl.spaceDim(); ++idim) {
        const amrex::Real min_prob_lo = amrex::min(buffer_HeaderImpl.problo(idim),
                                             snapshot_HeaderImpl.problo(idim));
        const amrex::Real max_prob_hi = amrex::max(buffer_HeaderImpl.probhi(idim),
                                             snapshot_HeaderImpl.probhi(idim));
        snapshot_HeaderImpl.set_problo(idim, min_prob_lo);
        snapshot_HeaderImpl.set_probhi(idim, max_prob_hi);
        // Update prob_hi with max of buffer and snapshot
        box_lo[idim] = amrex::min(buffer_Box.smallEnd(idim),
                                  snapshot_Box.smallEnd(idim));
        box_hi[idim] = amrex::max(buffer_Box.bigEnd(idim),
                                  snapshot_Box.bigEnd(idim));
    }
    const amrex::Box domain_box(box_lo, box_hi);
    snapshot_HeaderImpl.set_probDomain(domain_box);

    // Increment numFabs
    snapshot_HeaderImpl.IncrementNumFabs();
    // The number of fabs in the recently written buffer is always 1.
    snapshot_HeaderImpl.AppendNewFabLo( buffer_HeaderImpl.FabLo(0));
    snapshot_HeaderImpl.AppendNewFabHi( buffer_HeaderImpl.FabHi(0));

    snapshot_HeaderImpl.WriteHeader();
}


void
BTDiagnostics::InterleaveFabArrayHeader(const std::string& Buffer_FabHeader_path,
                                        const std::string& snapshot_FabHeader_path,
                                        const std::string& newsnapshot_FabFilename)
{
    BTDMultiFabHeaderImpl snapshot_FabHeader(snapshot_FabHeader_path);
    snapshot_FabHeader.ReadMultiFabHeader();

    BTDMultiFabHeaderImpl Buffer_FabHeader(Buffer_FabHeader_path);
    Buffer_FabHeader.ReadMultiFabHeader();

    // Increment existing fabs in snapshot with the number of fabs in the buffer
    snapshot_FabHeader.IncreaseMultiFabSize( Buffer_FabHeader.ba_size() );
    snapshot_FabHeader.ResizeFabData();

    for (int ifab = 0; ifab < Buffer_FabHeader.ba_size(); ++ifab) {
        const int new_ifab = snapshot_FabHeader.ba_size() - 1 + ifab;
        snapshot_FabHeader.SetBox(new_ifab, Buffer_FabHeader.ba_box(ifab) );
        // Set Name of the new fab using newsnapshot_FabFilename.
        snapshot_FabHeader.SetFabName(new_ifab, Buffer_FabHeader.fodPrefix(ifab),
                                                newsnapshot_FabFilename,
                                                Buffer_FabHeader.FabHead(ifab) );
        snapshot_FabHeader.SetMinVal(new_ifab, Buffer_FabHeader.minval(ifab));
        snapshot_FabHeader.SetMaxVal(new_ifab, Buffer_FabHeader.maxval(ifab));
    }

    snapshot_FabHeader.WriteMultiFabHeader();

}

void
BTDiagnostics::InterleaveSpeciesHeader(const std::string& buffer_species_Header_path,
                                       const std::string& snapshot_species_Header_path,
                                       const std::string& species_name, const int new_data_index)
{
    BTDSpeciesHeaderImpl BufferSpeciesHeader(buffer_species_Header_path,
                                             species_name);
    BufferSpeciesHeader.ReadHeader();

    BTDSpeciesHeaderImpl SnapshotSpeciesHeader(snapshot_species_Header_path,
                                               species_name);
    SnapshotSpeciesHeader.ReadHeader();
    SnapshotSpeciesHeader.AddTotalParticles( BufferSpeciesHeader.m_total_particles);

    SnapshotSpeciesHeader.IncrementParticleBoxArraySize();
    const int buffer_finestLevel = BufferSpeciesHeader.m_finestLevel;
    const int buffer_boxId = BufferSpeciesHeader.m_particleBoxArray_size[buffer_finestLevel]-1;
    SnapshotSpeciesHeader.AppendParticleInfoForNewBox(
                              new_data_index,
                              BufferSpeciesHeader.m_particles_per_box[buffer_finestLevel][buffer_boxId],
                              BufferSpeciesHeader.m_offset_per_box[buffer_finestLevel][buffer_boxId]);
    SnapshotSpeciesHeader.WriteHeader();
}

void
BTDiagnostics::InterleaveParticleDataHeader(const std::string& buffer_ParticleHdrFilename,
                                            const std::string& snapshot_ParticleHdrFilename)
{
    BTDParticleDataHeaderImpl BufferParticleHeader(buffer_ParticleHdrFilename);
    BufferParticleHeader.ReadHeader();

    BTDParticleDataHeaderImpl SnapshotParticleHeader(snapshot_ParticleHdrFilename);
    SnapshotParticleHeader.ReadHeader();

    // Increment BoxArraySize
    SnapshotParticleHeader.IncreaseBoxArraySize( BufferParticleHeader.ba_size() );
    // Append New box in snapshot
    for (int ibox = 0; ibox < BufferParticleHeader.ba_size(); ++ibox) {
        const int new_ibox = SnapshotParticleHeader.ba_size() - 1 + ibox;
        SnapshotParticleHeader.ResizeBoxArray();
        SnapshotParticleHeader.SetBox(new_ibox, BufferParticleHeader.ba_box(ibox) );
    }
    SnapshotParticleHeader.WriteHeader();
}

void
BTDiagnostics::InitializeParticleFunctors ()
{
    auto & warpx = WarpX::GetInstance();
    const MultiParticleContainer& mpc = warpx.GetPartContainer();
    // allocate with total number of species diagnostics
    m_all_particle_functors.resize(m_output_species_names.size());
    // Create an object of class BackTransformParticleFunctor
    for (int i = 0; i < m_all_particle_functors.size(); ++i)
    {
        // species id corresponding to ith diag species
        const int idx = mpc.getSpeciesID(m_output_species_names[i]);
        m_all_particle_functors[i] = std::make_unique<BackTransformParticleFunctor>(mpc.GetParticleContainerPtr(idx), m_output_species_names[i], m_num_buffers);
    }

}

void
BTDiagnostics::InitializeParticleBuffer ()
{
    auto& warpx = WarpX::GetInstance();
    const MultiParticleContainer& mpc = warpx.GetPartContainer();
    for (int i = 0; i < m_num_buffers; ++i) {
        m_particles_buffer[i].resize(m_output_species_names.size());
        m_totalParticles_in_buffer[i].resize(m_output_species_names.size());
        for (int isp = 0; isp < m_particles_buffer[i].size(); ++isp) {
            m_totalParticles_in_buffer[i][isp] = 0;
            m_particles_buffer[i][isp] = std::make_unique<PinnedMemoryParticleContainer>(WarpX::GetInstance().GetParGDB());
            const int idx = mpc.getSpeciesID(m_output_species_names[isp]);
            m_output_species[i].push_back(ParticleDiag(m_diag_name,
                                                       m_output_species_names[isp],
                                                       mpc.GetParticleContainerPtr(idx),
                                                       m_particles_buffer[i][isp].get()));
        }
    }

}

void
BTDiagnostics::PrepareParticleDataForOutput()
{
    for (int lev = 0; lev < nlev_output; ++lev) {
        for (int i = 0; i < m_all_particle_functors.size(); ++i)
        {
            for (int i_buffer = 0; i_buffer < m_num_buffers; ++i_buffer )
            {
                // Check if the zslice is in domain
                const bool ZSliceInDomain = GetZSliceInDomainFlag (i_buffer, lev);
                const bool kindexInSnapshotBox = GetKIndexInSnapshotBoxFlag (i_buffer, lev);
                if (kindexInSnapshotBox) {
                    if ( buffer_empty(i_buffer) ) {
                        if (!m_do_back_transformed_fields || m_varnames_fields.empty()) {
                            if ( m_buffer_flush_counter[i_buffer] == 0) {
                                DefineSnapshotGeometry(i_buffer, lev);
                            }
                            DefineFieldBufferMultiFab(i_buffer, lev);
                        }
                        const amrex::Box particle_buffer_box = m_buffer_box[i_buffer];
                        const amrex::BoxArray buffer_ba( particle_buffer_box );
                        const amrex::DistributionMapping buffer_dmap(buffer_ba);
                        m_particles_buffer[i_buffer][i]->SetParticleBoxArray(lev, buffer_ba);
                        m_particles_buffer[i_buffer][i]->SetParticleDistributionMap(lev, buffer_dmap);
                        m_particles_buffer[i_buffer][i]->SetParticleGeometry(lev, m_geom_snapshot[i_buffer][lev]);
                        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( m_particles_buffer[i_buffer][i]->ParticleBoxArray(lev).size() == 1,
                            "ParticleBoxArray size must be 1 for back-transformed diagnostic particle buffer");
                    }
                }
                m_all_particle_functors[i]->PrepareFunctorData (
                                             i_buffer, ZSliceInDomain, m_old_z_boost[i_buffer],
                                             m_current_z_boost[i_buffer], m_t_lab[i_buffer],
                                             m_snapshot_full[i_buffer]);
            }
        }
    }
}

void
BTDiagnostics::ResetTotalParticlesInBuffer(int i_buffer)
{
    std::fill(
        m_totalParticles_in_buffer[i_buffer].begin(),
        m_totalParticles_in_buffer[i_buffer].end(),
        0);
}

void
BTDiagnostics::ClearParticleBuffer(int i_buffer)
{
    std::for_each(
        m_particles_buffer[i_buffer].begin(),
        m_particles_buffer[i_buffer].end(),
        [](auto& pb){pb->clearParticles();});
}
