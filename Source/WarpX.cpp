/* Copyright 2016-2020 Andrew Myers, Ann Almgren, Aurore Blelly
 * Axel Huebl, Burlen Loring, David Grote
 * Glenn Richardson, Jean-Luc Vay, Junmin Gu
 * Mathieu Lobet, Maxence Thevenet, Michael Rowan
 * Remi Lehe, Revathi Jambunathan, Weiqun Zhang
 * Yinjian Zhao, levinem
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#include "BoundaryConditions/PEC_Insulator.H"
#include "BoundaryConditions/PML.H"
#include "Diagnostics/MultiDiagnostics.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "EmbeddedBoundary/Enabled.H"
#include "EmbeddedBoundary/WarpXFaceInfoBox.H"
#include "FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.H"
#include "FieldSolver/ElectrostaticSolvers/LabFrameExplicitES.H"
#include "FieldSolver/ElectrostaticSolvers/RelativisticExplicitES.H"
#include "FieldSolver/ElectrostaticSolvers/EffectivePotentialES.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#ifdef WARPX_USE_FFT
#   include "FieldSolver/SpectralSolver/SpectralKSpace.H"
#   ifdef WARPX_DIM_RZ
#       include "FieldSolver/SpectralSolver/SpectralSolverRZ.H"
#       include "BoundaryConditions/PML_RZ.H"
#   else
#       include "FieldSolver/SpectralSolver/SpectralSolver.H"
#   endif // RZ ifdef
#endif // use PSATD ifdef
#include "FieldSolver/WarpX_FDTD.H"
#include "Filter/NCIGodfreyFilter.H"
#include "Initialization/ExternalField.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Fluids/MultiFluidContainer.H"
#include "Fluids/WarpXFluidContainer.H"
#include "Particles/ParticleBoundaryBuffer.H"
#include "AcceleratorLattice/AcceleratorLattice.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "Utils/WarpXUtil.H"

#include "FieldSolver/ImplicitSolvers/ImplicitSolverLibrary.H"

#include <ablastr/utils/SignalHandling.H>
#include <ablastr/warn_manager/WarnManager.H>

#ifdef AMREX_USE_SENSEI_INSITU
#   include <AMReX_AmrMeshInSituBridge.H>
#endif
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Dim3.H>
#ifdef AMREX_USE_EB
#   include <AMReX_EBFabFactory.H>
#   include <AMReX_EBSupport.H>
#endif
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_FabFactory.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MFIter.H>
#include <AMReX_MakeType.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
#include <AMReX_SPACE.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

using namespace amrex;
using warpx::fields::FieldType;

int WarpX::do_moving_window = 0;
int WarpX::start_moving_window_step = 0;
int WarpX::end_moving_window_step = -1;
int WarpX::moving_window_dir = -1;
Real WarpX::moving_window_v = std::numeric_limits<amrex::Real>::max();

bool WarpX::fft_do_time_averaging = false;

amrex::IntVect WarpX::m_fill_guards_fields  = amrex::IntVect(0);
amrex::IntVect WarpX::m_fill_guards_current = amrex::IntVect(0);

Real WarpX::gamma_boost = 1._rt;
Real WarpX::beta_boost = 0._rt;
Vector<int> WarpX::boost_direction = {0,0,0};
bool WarpX::compute_max_step_from_btd = false;
Real WarpX::zmin_domain_boost_step_0 = 0._rt;

int WarpX::max_particle_its_in_implicit_scheme = 21;
ParticleReal WarpX::particle_tol_in_implicit_scheme = 1.e-10;
bool WarpX::do_dive_cleaning = false;
bool WarpX::do_divb_cleaning = false;
bool WarpX::do_single_precision_comms = false;

bool WarpX::do_shared_mem_charge_deposition = false;
bool WarpX::do_shared_mem_current_deposition = false;
#if defined(WARPX_DIM_3D)
amrex::IntVect WarpX::shared_tilesize(AMREX_D_DECL(6,6,8));
#elif (AMREX_SPACEDIM == 2)
amrex::IntVect WarpX::shared_tilesize(AMREX_D_DECL(14,14,0));
#else
//Have not experimented with good tilesize here because expect use case to be low
amrex::IntVect WarpX::shared_tilesize(AMREX_D_DECL(1,1,1));
#endif
int WarpX::shared_mem_current_tpb = 128;

int WarpX::n_rz_azimuthal_modes = 1;
int WarpX::ncomps = 1;

// This will be overwritten by setting nox = noy = noz = algo.particle_shape
int WarpX::nox = 0;
int WarpX::noy = 0;
int WarpX::noz = 0;

// Order of finite-order centering of fields (staggered to nodal)
int WarpX::field_centering_nox = 2;
int WarpX::field_centering_noy = 2;
int WarpX::field_centering_noz = 2;

// Order of finite-order centering of currents (nodal to staggered)
int WarpX::current_centering_nox = 2;
int WarpX::current_centering_noy = 2;
int WarpX::current_centering_noz = 2;

bool WarpX::use_fdtd_nci_corr = false;
bool WarpX::galerkin_interpolation = true;

bool WarpX::use_filter = true;
bool WarpX::use_kspace_filter       = true;
bool WarpX::use_filter_compensation = false;

bool WarpX::serialize_initial_conditions = false;
bool WarpX::refine_plasma     = false;

utils::parser::IntervalsParser WarpX::sort_intervals;
amrex::IntVect WarpX::sort_bin_size(AMREX_D_DECL(1,1,1));

#if defined(AMREX_USE_CUDA)
bool WarpX::sort_particles_for_deposition = true;
#else
bool WarpX::sort_particles_for_deposition = false;
#endif

amrex::IntVect WarpX::sort_idx_type(AMREX_D_DECL(0,0,0));

bool WarpX::do_dynamic_scheduling = true;

bool WarpX::do_multi_J = false;
int WarpX::do_multi_J_n_depositions;

std::map<std::string, amrex::iMultiFab *> WarpX::imultifab_map;

IntVect WarpX::filter_npass_each_dir(1);

int WarpX::n_field_gather_buffer = -1;
int WarpX::n_current_deposition_buffer = -1;

amrex::IntVect m_rho_nodal_flag;

WarpX* WarpX::m_instance = nullptr;

namespace
{

    [[nodiscard]] bool
    isAnyBoundaryPML(
        const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_lo,
        const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_hi)
    {
        constexpr auto is_pml = [](const FieldBoundaryType fbt) {return (fbt == FieldBoundaryType::PML);};
        const auto is_any_pml =
            std::any_of(field_boundary_lo.begin(), field_boundary_lo.end(), is_pml) ||
            std::any_of(field_boundary_hi.begin(), field_boundary_hi.end(), is_pml);
        return is_any_pml;
    }

    /**
     * \brief Re-orders the Fornberg coefficients so that they can be used more conveniently for
     * finite-order centering operations. For example, for finite-order centering of order 6,
     * the Fornberg coefficients \c (c_0,c_1,c_2) are re-ordered as \c (c_2,c_1,c_0,c_0,c_1,c_2).
     *
     * \param[in,out] ordered_coeffs host vector where the re-ordered Fornberg coefficients will be stored
     * \param[in] unordered_coeffs host vector storing the original sequence of Fornberg coefficients
     * \param[in] order order of the finite-order centering along a given direction
     */
    void ReorderFornbergCoefficients (
        amrex::Vector<amrex::Real>& ordered_coeffs,
        const amrex::Vector<amrex::Real>& unordered_coeffs,
        const int order)
        {
            const int n = order / 2;
            for (int i = 0; i < n; i++) {
                ordered_coeffs[i] = unordered_coeffs[n-1-i];
            }
            for (int i = n; i < order; i++) {
                ordered_coeffs[i] = unordered_coeffs[i-n];
            }
        }
}

void WarpX::MakeWarpX ()
{
    warpx::initialization::check_dims();

    ReadMovingWindowParameters(
        do_moving_window, start_moving_window_step, end_moving_window_step,
        moving_window_dir, moving_window_v);

    ConvertLabParamsToBoost();
    ReadBCParams();

#ifdef WARPX_DIM_RZ
    CheckGriddingForRZSpectral();
#endif

    m_instance = new WarpX();
}

WarpX&
WarpX::GetInstance ()
{
    if (!m_instance) {
        MakeWarpX();
    }
    return *m_instance;
}

void
WarpX::ResetInstance ()
{
    if (m_instance){
        delete m_instance;
        m_instance = nullptr;
    }
}

void
WarpX::Finalize()
{
    WarpX::ResetInstance();
}

WarpX::WarpX ()
{
    ReadParameters();

    BackwardCompatibility();

    if (EB::enabled()) { InitEB(); }

    ablastr::utils::SignalHandling::InitSignalHandling();

    // Geometry on all levels has been defined already.
    // No valid BoxArray and DistributionMapping have been defined.
    // But the arrays for them have been resized.

    const int nlevs_max = maxLevel() + 1;

    istep.resize(nlevs_max, 0);
    nsubsteps.resize(nlevs_max, 1);

    t_new.resize(nlevs_max, 0.0);
    t_old.resize(nlevs_max, std::numeric_limits<Real>::lowest());
    dt.resize(nlevs_max, std::numeric_limits<Real>::max());

    mypc = std::make_unique<MultiParticleContainer>(this);

    // Loop over species (particles and lasers)
    // and set current injection position per species
    if (do_moving_window){
        const int n_containers = mypc->nContainers();
        for (int i=0; i<n_containers; i++)
        {
            WarpXParticleContainer& pc = mypc->GetParticleContainer(i);

            // Storing injection position for all species, regardless of whether
            // they are continuously injected, since it makes looping over the
            // elements of current_injection_position easier elsewhere in the code.
            if (moving_window_v > 0._rt)
            {
                // Inject particles continuously from the right end of the box
                pc.m_current_injection_position = geom[0].ProbHi(moving_window_dir);
            }
            else if (moving_window_v < 0._rt)
            {
                // Inject particles continuously from the left end of the box
                pc.m_current_injection_position = geom[0].ProbLo(moving_window_dir);
            }
        }
    }

    // Particle Boundary Buffer (i.e., scraped particles on boundary)
    m_particle_boundary_buffer = std::make_unique<ParticleBoundaryBuffer>();

    // Fluid Container
    if (do_fluid_species) {
        myfl = std::make_unique<MultiFluidContainer>();
    }

    Efield_dotMask.resize(nlevs_max);
    Bfield_dotMask.resize(nlevs_max);
    Afield_dotMask.resize(nlevs_max);
    phi_dotMask.resize(nlevs_max);

    m_eb_update_E.resize(nlevs_max);
    m_eb_update_B.resize(nlevs_max);
    m_eb_reduce_particle_shape.resize(nlevs_max);

    m_flag_info_face.resize(nlevs_max);
    m_flag_ext_face.resize(nlevs_max);
    m_borrowing.resize(nlevs_max);

    // Create Electrostatic Solver object if needed
    if ((WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrame)
        || (WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameElectroMagnetostatic))
    {
        m_electrostatic_solver = std::make_unique<LabFrameExplicitES>(nlevs_max);
    }
    // Initialize the effective potential electrostatic solver if required
    else if (electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameEffectivePotential)
    {
        m_electrostatic_solver = std::make_unique<EffectivePotentialES>(nlevs_max);
    }
    else
    {
        m_electrostatic_solver = std::make_unique<RelativisticExplicitES>(nlevs_max);
    }

    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC)
    {
        // Create hybrid-PIC model object if needed
        m_hybrid_pic_model = std::make_unique<HybridPICModel>();
    }

    current_buffer_masks.resize(nlevs_max);
    gather_buffer_masks.resize(nlevs_max);

    pml.resize(nlevs_max);
#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
    pml_rz.resize(nlevs_max);
#endif

    do_pml_Lo.resize(nlevs_max);
    do_pml_Hi.resize(nlevs_max);

    costs.resize(nlevs_max);
    load_balance_efficiency.resize(nlevs_max);

    m_field_factory.resize(nlevs_max);

    if (em_solver_medium == MediumForEM::Macroscopic) {
        // create object for macroscopic solver
        m_macroscopic_properties = std::make_unique<MacroscopicProperties>();
    }

    // Set default values for particle and cell weights for costs update;
    // Default values listed here for the case AMREX_USE_GPU are determined
    // from single-GPU tests on Summit.
    if (costs_heuristic_cells_wt<=0. && costs_heuristic_particles_wt<=0.
        && WarpX::load_balance_costs_update_algo==LoadBalanceCostsUpdateAlgo::Heuristic)
    {
#ifdef AMREX_USE_GPU
        if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
            switch (WarpX::nox)
            {
                case 1:
                    costs_heuristic_cells_wt = 0.575_rt;
                    costs_heuristic_particles_wt = 0.425_rt;
                    break;
                case 2:
                    costs_heuristic_cells_wt = 0.405_rt;
                    costs_heuristic_particles_wt = 0.595_rt;
                    break;
                case 3:
                    costs_heuristic_cells_wt = 0.250_rt;
                    costs_heuristic_particles_wt = 0.750_rt;
                    break;
                case 4:
                    // this is only a guess
                    costs_heuristic_cells_wt = 0.200_rt;
                    costs_heuristic_particles_wt = 0.800_rt;
                    break;
            }
        } else { // FDTD
            switch (WarpX::nox)
            {
                case 1:
                    costs_heuristic_cells_wt = 0.401_rt;
                    costs_heuristic_particles_wt = 0.599_rt;
                    break;
                case 2:
                    costs_heuristic_cells_wt = 0.268_rt;
                    costs_heuristic_particles_wt = 0.732_rt;
                    break;
                case 3:
                    costs_heuristic_cells_wt = 0.145_rt;
                    costs_heuristic_particles_wt = 0.855_rt;
                    break;
                case 4:
                    // this is only a guess
                    costs_heuristic_cells_wt = 0.100_rt;
                    costs_heuristic_particles_wt = 0.900_rt;
                    break;
            }
        }
#else // CPU
        costs_heuristic_cells_wt = 0.1_rt;
        costs_heuristic_particles_wt = 0.9_rt;
#endif // AMREX_USE_GPU
    }

    // Allocate field solver objects
#ifdef WARPX_USE_FFT
    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
        spectral_solver_fp.resize(nlevs_max);
        spectral_solver_cp.resize(nlevs_max);
    }
#endif
    if (WarpX::electromagnetic_solver_id != ElectromagneticSolverAlgo::PSATD) {
        m_fdtd_solver_fp.resize(nlevs_max);
        m_fdtd_solver_cp.resize(nlevs_max);
    }

    // NCI Godfrey filters can have different stencils
    // at different levels (the stencil depends on c*dt/dz)
    nci_godfrey_filter_exeybz.resize(nlevs_max);
    nci_godfrey_filter_bxbyez.resize(nlevs_max);

    // Sanity checks. Must be done after calling the MultiParticleContainer
    // constructor, as it reads additional parameters
    // (e.g., use_fdtd_nci_corr)
    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
        AMREX_ALWAYS_ASSERT(use_fdtd_nci_corr == 0);
        AMREX_ALWAYS_ASSERT(m_do_subcycling == 0);
    }

    if (WarpX::current_deposition_algo != CurrentDepositionAlgo::Esirkepov) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            use_fdtd_nci_corr == 0,
            "The NCI corrector should only be used with Esirkepov deposition");
    }

    m_accelerator_lattice.resize(nlevs_max);

}

WarpX::~WarpX ()
{
    const int nlevs_max = maxLevel() +1;
    for (int lev = 0; lev < nlevs_max; ++lev) {
        ClearLevel(lev);
    }
}

void
WarpX::ReadParameters ()
{
    {
        const ParmParse pp;// Traditionally, max_step and stop_time do not have prefix.
        utils::parser::queryWithParser(pp, "max_step", max_step);
        utils::parser::queryWithParser(pp, "stop_time", stop_time);
        pp.query("authors", m_authors);
    }

    {
        const ParmParse pp_amr("amr");

        pp_amr.query("restart", restart_chkfile);
    }

    {
        const ParmParse pp_algo("algo");
        pp_algo.query_enum_sloppy("maxwell_solver", electromagnetic_solver_id, "-_");
        if (electromagnetic_solver_id == ElectromagneticSolverAlgo::ECT && !EB::enabled()) {
            throw std::runtime_error("ECP Solver requires to enable embedded boundaries at runtime.");
        }
#ifdef WARPX_DIM_RZ
        if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(Geom(0).ProbLo(0) == 0.,
                "Lower bound of radial coordinate (prob_lo[0]) with RZ PSATD solver must be zero");
        }
        else
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(Geom(0).ProbLo(0) >= 0.,
            "Lower bound of radial coordinate (prob_lo[0]) with RZ FDTD solver must be non-negative");
        }
#endif

        pp_algo.query_enum_sloppy("evolve_scheme", evolve_scheme, "-_");
    }

    {
        ParmParse const pp_warpx("warpx");

        //"Synthetic" warning messages may be injected in the Warning Manager via
        // inputfile for debug&testing purposes.
        ablastr::warn_manager::GetWMInstance().debug_read_warnings_from_input(pp_warpx);

        // Set the flag to control if WarpX has to emit a warning message as soon as a warning is recorded
        bool always_warn_immediately = false;
        pp_warpx.query("always_warn_immediately", always_warn_immediately);
        ablastr::warn_manager::GetWMInstance().SetAlwaysWarnImmediately(always_warn_immediately);

        // Set the WarnPriority threshold to decide if WarpX has to abort when a warning is recorded
        if(std::string str_abort_on_warning_threshold;
            pp_warpx.query("abort_on_warning_threshold", str_abort_on_warning_threshold)){
            std::optional<ablastr::warn_manager::WarnPriority> abort_on_warning_threshold = std::nullopt;
            if (str_abort_on_warning_threshold == "high") {
                abort_on_warning_threshold = ablastr::warn_manager::WarnPriority::high;
            } else if (str_abort_on_warning_threshold == "medium" ) {
                abort_on_warning_threshold = ablastr::warn_manager::WarnPriority::medium;
            } else if (str_abort_on_warning_threshold == "low") {
                abort_on_warning_threshold = ablastr::warn_manager::WarnPriority::low;
            } else {
                WARPX_ABORT_WITH_MESSAGE(str_abort_on_warning_threshold
                    +"is not a valid option for warpx.abort_on_warning_threshold (use: low, medium or high)");
            }
            ablastr::warn_manager::GetWMInstance().SetAbortThreshold(abort_on_warning_threshold);
        }

        std::vector<int> numprocs_in;
        utils::parser::queryArrWithParser(
            pp_warpx, "numprocs", numprocs_in, 0, AMREX_SPACEDIM);

        if (not numprocs_in.empty()) {
#ifdef WARPX_DIM_RZ
            if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(numprocs_in[0] == 1,
                    "Domain decomposition in RZ with spectral solvers works only along z direction");
            }
#endif
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE
                (numprocs_in.size() == AMREX_SPACEDIM,
                 "warpx.numprocs, if specified, must have AMREX_SPACEDIM numbers");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE
                (ParallelDescriptor::NProcs() == AMREX_D_TERM(numprocs_in[0],
                                                             *numprocs_in[1],
                                                             *numprocs_in[2]),
                 "warpx.numprocs, if specified, its product must be equal to the number of processes");
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                numprocs[idim] = numprocs_in[idim];
            }
        }

        using ablastr::utils::SignalHandling;
        std::vector<std::string> signals_in;
        pp_warpx.queryarr("break_signals", signals_in);

#if defined(__linux__) || defined(__APPLE__)
        for (const std::string &str : signals_in) {
            const int sig = SignalHandling::parseSignalNameToNumber(str);
            SignalHandling::signal_conf_requests[SignalHandling::SIGNAL_REQUESTS_BREAK][sig] = true;
        }
        signals_in.clear();
#else
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(signals_in.empty(),
                                         "Signal handling requested in input, but is not supported on this platform");
#endif

        bool have_checkpoint_diagnostic = false;

        const ParmParse pp("diagnostics");
        std::vector<std::string> diags_names;
        pp.queryarr("diags_names", diags_names);

        for (const auto &diag : diags_names) {
            const ParmParse dd(diag);
            std::string format;
            dd.query("format", format);
            if (format == "checkpoint") {
                have_checkpoint_diagnostic = true;
                break;
            }
        }

        pp_warpx.query("write_diagnostics_on_restart", write_diagnostics_on_restart);

        pp_warpx.queryarr("checkpoint_signals", signals_in);
#if defined(__linux__) || defined(__APPLE__)
        for (const std::string &str : signals_in) {
            const int sig = SignalHandling::parseSignalNameToNumber(str);
            SignalHandling::signal_conf_requests[SignalHandling::SIGNAL_REQUESTS_CHECKPOINT][sig] = true;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(have_checkpoint_diagnostic,
                                             "Signal handling was requested to checkpoint, but no checkpoint diagnostic is configured");
        }
#else
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(signals_in.empty(),
                                         "Signal handling requested in input, but is not supported on this platform");
#endif

        // set random seed
        std::string random_seed = "default";
        pp_warpx.query("random_seed", random_seed);
        if ( random_seed != "default" ) {
            const unsigned long myproc_1 = ParallelDescriptor::MyProc() + 1;
            if ( random_seed == "random" ) {
                std::random_device rd;
                std::uniform_int_distribution<int> dist(2, INT_MAX);
                const unsigned long cpu_seed = myproc_1 * dist(rd);
                const unsigned long gpu_seed = myproc_1 * dist(rd);
                ResetRandomSeed(cpu_seed, gpu_seed);
            } else if ( std::stoi(random_seed) > 0 ) {
                const unsigned long nprocs = ParallelDescriptor::NProcs();
                const unsigned long seed_long = std::stoul(random_seed);
                const unsigned long cpu_seed = myproc_1 * seed_long;
                const unsigned long gpu_seed = (myproc_1 + nprocs) * seed_long;
                ResetRandomSeed(cpu_seed, gpu_seed);
            } else {
                WARPX_ABORT_WITH_MESSAGE(
                    "warpx.random_seed must be \"default\", \"random\" or an integer > 0.");
            }
        }

        utils::parser::queryWithParser(pp_warpx, "cfl", cfl);
        pp_warpx.query("verbose", verbose);
        utils::parser::queryWithParser(pp_warpx, "regrid_int", regrid_int);
        pp_warpx.query("do_subcycling", m_do_subcycling);
        pp_warpx.query("do_multi_J", do_multi_J);
        if (do_multi_J)
        {
            utils::parser::getWithParser(
                pp_warpx, "do_multi_J_n_depositions", do_multi_J_n_depositions);
        }
        pp_warpx.query("use_hybrid_QED", use_hybrid_QED);
        pp_warpx.query("safe_guard_cells", m_safe_guard_cells);
        std::vector<std::string> override_sync_intervals_string_vec = {"1"};
        pp_warpx.queryarr("override_sync_intervals", override_sync_intervals_string_vec);
        override_sync_intervals =
            utils::parser::IntervalsParser(override_sync_intervals_string_vec);

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_do_subcycling != 1 || max_level <= 1,
                                         "Subcycling method 1 only works for 2 levels.");

        ReadBoostedFrameParameters(gamma_boost, beta_boost, boost_direction);

        // queryWithParser returns 1 if argument zmax_plasma_to_compute_max_step is
        // specified by the user, 0 otherwise.
        if(auto temp = 0.0_rt; utils::parser::queryWithParser(pp_warpx, "zmax_plasma_to_compute_max_step",temp)){
            m_zmax_plasma_to_compute_max_step = temp;
        }

        pp_warpx.query("compute_max_step_from_btd",
            compute_max_step_from_btd);

        if (do_moving_window) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                Geom(0).isPeriodic(moving_window_dir) == 0,
                "The problem must be non-periodic in the moving window direction");
            moving_window_x = geom[0].ProbLo(moving_window_dir);
        }

        m_p_ext_field_params = std::make_unique<ExternalFieldParams>(pp_warpx);
        if (m_p_ext_field_params->B_ext_grid_type == ExternalFieldType::read_from_file ||
            m_p_ext_field_params->E_ext_grid_type == ExternalFieldType::read_from_file){
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(max_level == 0,
                "External field reading is not implemented for more than one level");
        }

        maxlevel_extEMfield_init = maxLevel();
        pp_warpx.query("maxlevel_extEMfield_init", maxlevel_extEMfield_init);

        pp_warpx.query_enum_sloppy("do_electrostatic", electrostatic_solver_id, "-_");
        // if an electrostatic solver is used, set the Maxwell solver to None
        if (electrostatic_solver_id != ElectrostaticSolverAlgo::None) {
            electromagnetic_solver_id = ElectromagneticSolverAlgo::None;
        }

        pp_warpx.query_enum_sloppy("poisson_solver", poisson_solver_id, "-_");
#ifndef WARPX_DIM_3D
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        poisson_solver_id!=PoissonSolverAlgo::IntegratedGreenFunction,
        "The FFT Poisson solver only works in 3D.");
#endif
#ifndef WARPX_USE_FFT
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        poisson_solver_id!=PoissonSolverAlgo::IntegratedGreenFunction,
        "To use the FFT Poisson solver, compile with WARPX_USE_FFT=ON.");
#endif
        utils::parser::queryWithParser(pp_warpx, "self_fields_max_iters", magnetostatic_solver_max_iters);
        utils::parser::queryWithParser(pp_warpx, "self_fields_verbosity", magnetostatic_solver_verbosity);

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (
            electrostatic_solver_id!=ElectrostaticSolverAlgo::LabFrameElectroMagnetostatic ||
            poisson_solver_id!=PoissonSolverAlgo::IntegratedGreenFunction
        ),
        "The FFT Poisson solver is not implemented in labframe-electromagnetostatic mode yet."
        );

        [[maybe_unused]] bool const eb_enabled = EB::enabled();
#if !defined(AMREX_USE_EB)
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !eb_enabled,
            "Embedded boundaries are requested via warpx.eb_enabled but were not compiled!"
        );
#endif

#ifdef WARPX_DIM_RZ
        const ParmParse pp_boundary("boundary");
        pp_boundary.query("verboncoeur_axis_correction", m_verboncoeur_axis_correction);
#endif

        // Read timestepping options
        utils::parser::queryWithParser(pp_warpx, "const_dt", m_const_dt);
        utils::parser::queryWithParser(pp_warpx, "max_dt", m_max_dt);
        std::vector<std::string> dt_interval_vec = {"-1"};
        pp_warpx.queryarr("dt_update_interval", dt_interval_vec);
        m_dt_update_interval = utils::parser::IntervalsParser(dt_interval_vec);

        // Filter defaults to true for the explicit scheme, and false for the implicit schemes
        if (evolve_scheme != EvolveScheme::Explicit) {
            use_filter = false;
        }

        // Filter currently not working with FDTD solver in RZ geometry: turn OFF by default
        // (see https://github.com/ECP-WarpX/WarpX/issues/1943)
#ifdef WARPX_DIM_RZ
        if (WarpX::electromagnetic_solver_id != ElectromagneticSolverAlgo::PSATD) { WarpX::use_filter = false; }
#endif

        // Read filter and fill IntVect filter_npass_each_dir with
        // proper size for AMREX_SPACEDIM
        pp_warpx.query("use_filter", use_filter);
        pp_warpx.query("use_filter_compensation", use_filter_compensation);
        Vector<int> parse_filter_npass_each_dir(AMREX_SPACEDIM,1);
        utils::parser::queryArrWithParser(
            pp_warpx, "filter_npass_each_dir", parse_filter_npass_each_dir, 0, AMREX_SPACEDIM);
        filter_npass_each_dir[0] = parse_filter_npass_each_dir[0];
#if (AMREX_SPACEDIM >= 2)
        filter_npass_each_dir[1] = parse_filter_npass_each_dir[1];
#endif
#if defined(WARPX_DIM_3D)
        filter_npass_each_dir[2] = parse_filter_npass_each_dir[2];
#endif

        // TODO When k-space filtering will be implemented also for Cartesian geometries,
        // this code block will have to be applied in all cases (remove #ifdef condition)
#ifdef WARPX_DIM_RZ
        if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
            // With RZ spectral, only use k-space filtering
            use_kspace_filter = use_filter;
            use_filter = false;
        }
        else // FDTD
        {
            // Filter currently not working with FDTD solver in RZ geometry along R
            // (see https://github.com/ECP-WarpX/WarpX/issues/1943)
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!use_filter || filter_npass_each_dir[0] == 0,
                "In RZ geometry with FDTD, filtering can only be apply along z. This can be controlled by setting warpx.filter_npass_each_dir");
        }
#endif

        utils::parser::queryWithParser(
            pp_warpx, "num_mirrors", m_num_mirrors);
        if (m_num_mirrors>0){
            m_mirror_z.resize(m_num_mirrors);
            utils::parser::getArrWithParser(
                pp_warpx, "mirror_z", m_mirror_z, 0, m_num_mirrors);
            m_mirror_z_width.resize(m_num_mirrors);
            utils::parser::getArrWithParser(
                pp_warpx, "mirror_z_width", m_mirror_z_width, 0, m_num_mirrors);
            m_mirror_z_npoints.resize(m_num_mirrors);
            utils::parser::getArrWithParser(
                pp_warpx, "mirror_z_npoints", m_mirror_z_npoints, 0, m_num_mirrors);
        }

        pp_warpx.query("do_single_precision_comms", do_single_precision_comms);
#ifdef AMREX_USE_FLOAT
        if (do_single_precision_comms) {
            do_single_precision_comms = false;
            ablastr::warn_manager::WMRecordWarning(
                "comms",
                "Overwrote warpx.do_single_precision_comms to be 0, since WarpX was built in single precision.",
                ablastr::warn_manager::WarnPriority::low);
        }
#endif
        pp_warpx.query("do_shared_mem_charge_deposition", do_shared_mem_charge_deposition);
        pp_warpx.query("do_shared_mem_current_deposition", do_shared_mem_current_deposition);
#if !(defined(AMREX_USE_HIP) || defined(AMREX_USE_CUDA))
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!do_shared_mem_current_deposition,
                "requested shared memory for current deposition, but shared memory is only available for CUDA or HIP");
#endif
        pp_warpx.query("shared_mem_current_tpb", shared_mem_current_tpb);

        // initialize the shared tilesize
        Vector<int> vect_shared_tilesize(AMREX_SPACEDIM, 1);
        const bool shared_tilesize_is_specified = utils::parser::queryArrWithParser(pp_warpx, "shared_tilesize",
                                                            vect_shared_tilesize, 0, AMREX_SPACEDIM);
        if (shared_tilesize_is_specified){
            for (int i=0; i<AMREX_SPACEDIM; i++) {
                shared_tilesize[i] = vect_shared_tilesize[i];
            }
        }

        pp_warpx.query("serialize_initial_conditions", serialize_initial_conditions);
        pp_warpx.query("refine_plasma", refine_plasma);
        pp_warpx.query("do_dive_cleaning", do_dive_cleaning);
        pp_warpx.query("do_divb_cleaning", do_divb_cleaning);

        utils::parser::queryWithParser(
            pp_warpx, "n_field_gather_buffer", n_field_gather_buffer);
        utils::parser::queryWithParser(
            pp_warpx, "n_current_deposition_buffer", n_current_deposition_buffer);

        //Default value for the quantum parameter used in Maxwell’s QED equations
        m_quantum_xi_c2 = PhysConst::xi_c2;

        amrex::Real quantum_xi_tmp;
        const auto quantum_xi_is_specified =
            utils::parser::queryWithParser(pp_warpx, "quantum_xi", quantum_xi_tmp);
        if (quantum_xi_is_specified) {
            double const quantum_xi = quantum_xi_tmp;
            m_quantum_xi_c2 = static_cast<amrex::Real>(quantum_xi * PhysConst::c * PhysConst::c);
        }

        const auto at_least_one_boundary_is_pml =
            (std::any_of(WarpX::field_boundary_lo.begin(), WarpX::field_boundary_lo.end(),
                [](const auto& cc){return cc == FieldBoundaryType::PML;})
            ||
            std::any_of(WarpX::field_boundary_hi.begin(), WarpX::field_boundary_hi.end(),
                [](const auto& cc){return cc == FieldBoundaryType::PML;})
            );

        const auto at_least_one_boundary_is_silver_mueller =
            (std::any_of(WarpX::field_boundary_lo.begin(), WarpX::field_boundary_lo.end(),
                [](const auto& cc){return cc == FieldBoundaryType::Absorbing_SilverMueller;})
            ||
            std::any_of(WarpX::field_boundary_hi.begin(), WarpX::field_boundary_hi.end(),
                [](const auto& cc){return cc == FieldBoundaryType::Absorbing_SilverMueller;})
            );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !(at_least_one_boundary_is_pml && at_least_one_boundary_is_silver_mueller),
            "PML and Silver-Mueller boundary conditions cannot be activated at the same time.");

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (!at_least_one_boundary_is_silver_mueller) ||
            (electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee),
            "The Silver-Mueller boundary condition can only be used with the Yee solver.");

        utils::parser::queryWithParser(pp_warpx, "pml_ncell", pml_ncell);
        utils::parser::queryWithParser(pp_warpx, "pml_delta", pml_delta);
        pp_warpx.query("pml_has_particles", pml_has_particles);
        pp_warpx.query("do_pml_j_damping", do_pml_j_damping);
        pp_warpx.query("do_pml_in_domain", do_pml_in_domain);
        pp_warpx.query("do_similar_dm_pml", do_similar_dm_pml);
        // Read `v_particle_pml` in units of the speed of light
        v_particle_pml = 1._rt;
        utils::parser::queryWithParser(pp_warpx, "v_particle_pml", v_particle_pml);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(0._rt < v_particle_pml && v_particle_pml <= 1._rt,
            "Input value for the velocity warpx.v_particle_pml of the macroparticle must be in (0,1] (in units of c).");
        // Scale by the speed of light
        v_particle_pml = v_particle_pml * PhysConst::c;

        // Default values of WarpX::do_pml_dive_cleaning and WarpX::do_pml_divb_cleaning:
        // true for Cartesian PSATD solver, false otherwise
        do_pml_dive_cleaning = false;
        do_pml_divb_cleaning = false;
#ifndef WARPX_DIM_RZ
        if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD)
        {
            do_pml_dive_cleaning = true;
            do_pml_divb_cleaning = true;
        }

        // If WarpX::do_dive_cleaning = true, set also WarpX::do_pml_dive_cleaning = true
        // (possibly overwritten by users in the input file, see query below)
        if (do_dive_cleaning) { do_pml_dive_cleaning = true; }

        // If WarpX::do_divb_cleaning = true, set also WarpX::do_pml_divb_cleaning = true
        // (possibly overwritten by users in the input file, see query below)
        // TODO Implement div(B) cleaning in PML with FDTD and remove second if condition
        if (do_divb_cleaning && electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) { do_pml_divb_cleaning = true; }
#endif

        // Query input parameters to use div(E) and div(B) cleaning in PMLs
        pp_warpx.query("do_pml_dive_cleaning", do_pml_dive_cleaning);
        pp_warpx.query("do_pml_divb_cleaning", do_pml_divb_cleaning);

        // TODO Implement div(B) cleaning in PML with FDTD and remove ASSERT
        if (electromagnetic_solver_id != ElectromagneticSolverAlgo::PSATD)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                do_pml_divb_cleaning == false,
                "warpx.do_pml_divb_cleaning = true not implemented for FDTD solver");
        }

        // Divergence cleaning in PMLs for PSATD solver implemented only
        // for both div(E) and div(B) cleaning
        if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                do_pml_dive_cleaning == do_pml_divb_cleaning,
                "warpx.do_pml_dive_cleaning = "
                + std::to_string(do_pml_dive_cleaning)
                +" and warpx.do_pml_divb_cleaning = "
                + std::to_string(do_pml_divb_cleaning)
                + ": this case is not implemented yet,"
                + " please set both parameters to the same value"
            );
        }

#ifdef WARPX_DIM_RZ
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( ::isAnyBoundaryPML(field_boundary_lo, field_boundary_hi) == false || electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD,
            "PML are not implemented in RZ geometry with FDTD; please set a different boundary condition using boundary.field_lo and boundary.field_hi.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( field_boundary_lo[1] != FieldBoundaryType::PML && field_boundary_hi[1] != FieldBoundaryType::PML,
            "PML are not implemented in RZ geometry along z; please set a different boundary condition using boundary.field_lo and boundary.field_hi.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( (do_pml_dive_cleaning == false && do_pml_divb_cleaning == false),
            "do_pml_dive_cleaning and do_pml_divb_cleaning are not implemented in RZ geometry." );
#endif

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (do_pml_j_damping==0)||(do_pml_in_domain==1),
            "J-damping can only be done when PML are inside simulation domain (do_pml_in_domain=1)"
        );

        {
            // Parameters below control all plotfile diagnostics
            bool plotfile_min_max = true;
            pp_warpx.query("plotfile_min_max", plotfile_min_max);
            if (plotfile_min_max) {
                plotfile_headerversion = amrex::VisMF::Header::Version_v1;
            } else {
                plotfile_headerversion = amrex::VisMF::Header::NoFabHeader_v1;
            }
            pp_warpx.query("usesingleread", use_single_read);
            pp_warpx.query("usesinglewrite", use_single_write);
            ParmParse pp_vismf("vismf");
            pp_vismf.add("usesingleread", use_single_read);
            pp_vismf.add("usesinglewrite", use_single_write);
            utils::parser::queryWithParser(pp_warpx, "mffile_nstreams", mffile_nstreams);
            VisMF::SetMFFileInStreams(mffile_nstreams);
            utils::parser::queryWithParser(pp_warpx, "field_io_nfiles", field_io_nfiles);
            VisMF::SetNOutFiles(field_io_nfiles);
            utils::parser::queryWithParser(pp_warpx, "particle_io_nfiles", particle_io_nfiles);
            ParmParse pp_particles("particles");
            pp_particles.add("particles_nfiles", particle_io_nfiles);
        }

        if (maxLevel() > 0) {
            Vector<Real> lo, hi;
            const bool fine_tag_lo_specified = utils::parser::queryArrWithParser(pp_warpx, "fine_tag_lo", lo);
            const bool fine_tag_hi_specified = utils::parser::queryArrWithParser(pp_warpx, "fine_tag_hi", hi);
            std::string ref_patch_function;
            const bool parser_specified = pp_warpx.query("ref_patch_function(x,y,z)",ref_patch_function);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( ((fine_tag_lo_specified && fine_tag_hi_specified) ||
                                                parser_specified ),
                                                "For max_level > 0, you need to either set\
                                                warpx.fine_tag_lo and warpx.fine_tag_hi\
                                                or warpx.ref_patch_function(x,y,z)");

            if ( (fine_tag_lo_specified && fine_tag_hi_specified) && parser_specified) {
               ablastr::warn_manager::WMRecordWarning("Refined patch", "Both fine_tag_lo,fine_tag_hi\
                   and ref_patch_function(x,y,z) are provided. Note that fine_tag_lo/fine_tag_hi will\
                   override the ref_patch_function(x,y,z) for defining the refinement patches");
            }
            if (fine_tag_lo_specified && fine_tag_hi_specified) {
                fine_tag_lo = RealVect{lo};
                fine_tag_hi = RealVect{hi};
            } else {
                utils::parser::Store_parserString(pp_warpx, "ref_patch_function(x,y,z)", ref_patch_function);
                ref_patch_parser = std::make_unique<amrex::Parser>(
                    utils::parser::makeParser(ref_patch_function,{"x","y","z"}));
            }
        }

        pp_warpx.query("do_dynamic_scheduling", do_dynamic_scheduling);

        // Integer that corresponds to the type of grid used in the simulation
        // (collocated, staggered, hybrid)
        pp_warpx.query_enum_sloppy("grid_type", grid_type, "-_");

        // Use same shape factors in all directions, for gathering
        if (grid_type == GridType::Collocated) { galerkin_interpolation = false; }

#ifdef WARPX_DIM_RZ
        // Only needs to be set with WARPX_DIM_RZ, otherwise defaults to 1
        utils::parser::queryWithParser(pp_warpx, "n_rz_azimuthal_modes", n_rz_azimuthal_modes);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( n_rz_azimuthal_modes > 0,
            "The number of azimuthal modes (n_rz_azimuthal_modes) must be at least 1");
#endif

        // Check whether fluid species will be used
        {
            const ParmParse pp_fluids("fluids");
            std::vector<std::string> fluid_species_names = {};
            pp_fluids.queryarr("species_names", fluid_species_names);
            do_fluid_species = !fluid_species_names.empty();
            if (do_fluid_species) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(max_level <= 1,
                    "Fluid species cannot currently be used with mesh refinement.");
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    electrostatic_solver_id != ElectrostaticSolverAlgo::Relativistic,
                    "Fluid species cannot currently be used with the relativistic electrostatic solver.");
#ifdef WARPX_DIM_RZ
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE( n_rz_azimuthal_modes <= 1,
                    "Fluid species cannot be used with more than 1 azimuthal mode.");
#endif
            }
        }

        // Set default parameters with hybrid grid (parsed later below)
        if (grid_type == GridType::Hybrid)
        {
            // Finite-order centering of fields (staggered to nodal)
            // Default field gathering algorithm will be set below
            field_centering_nox = 8;
            field_centering_noy = 8;
            field_centering_noz = 8;
            // Finite-order centering of currents (nodal to staggered)
            do_current_centering = true;
            current_centering_nox = 8;
            current_centering_noy = 8;
            current_centering_noz = 8;
        }

#ifdef WARPX_DIM_RZ
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            grid_type != GridType::Hybrid,
            "warpx.grid_type=hybrid is not implemented in RZ geometry");
#endif

        // Update default to external projection divb cleaner if external fields are loaded,
        // the grids are staggered, and the solver is compatible with the cleaner
        if (!do_divb_cleaning
            && m_p_ext_field_params->B_ext_grid_type != ExternalFieldType::default_zero
            && m_p_ext_field_params->B_ext_grid_type != ExternalFieldType::constant
            && grid_type != GridType::Collocated
            && (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee
            ||  WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC
            ||  ( (WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrame
                || WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameElectroMagnetostatic)
                && WarpX::poisson_solver_id == PoissonSolverAlgo::Multigrid)))
        {
            m_do_divb_cleaning_external = true;
        }
        pp_warpx.query("do_divb_cleaning_external", m_do_divb_cleaning_external);

        // If true, the current is deposited on a nodal grid and centered onto
        // a staggered grid. Setting warpx.do_current_centering=1 makes sense
        // only if warpx.grid_type=hybrid. Instead, if warpx.grid_type=nodal or
        // warpx.grid_type=staggered, Maxwell's equations are solved either on a
        // collocated grid or on a staggered grid without current centering.
        pp_warpx.query("do_current_centering", do_current_centering);
        if (do_current_centering)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                grid_type == GridType::Hybrid,
                "warpx.do_current_centering=1 can be used only with warpx.grid_type=hybrid");

            utils::parser::queryWithParser(
                pp_warpx, "current_centering_nox", current_centering_nox);
            utils::parser::queryWithParser(
                pp_warpx, "current_centering_noy", current_centering_noy);
            utils::parser::queryWithParser(
                pp_warpx, "current_centering_noz", current_centering_noz);

            AllocateCenteringCoefficients(device_current_centering_stencil_coeffs_x,
                                          device_current_centering_stencil_coeffs_y,
                                          device_current_centering_stencil_coeffs_z,
                                          current_centering_nox,
                                          current_centering_noy,
                                          current_centering_noz,
                                          grid_type);
        }

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            maxLevel() == 0 || !do_current_centering,
            "Finite-order centering of currents is not implemented with mesh refinement"
        );
    }

    {
        const ParmParse pp_algo("algo");
#ifdef WARPX_DIM_RZ
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( electromagnetic_solver_id != ElectromagneticSolverAlgo::CKC,
            "algo.maxwell_solver = ckc is not (yet) available for RZ geometry");
#endif
#ifndef WARPX_USE_FFT
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( electromagnetic_solver_id != ElectromagneticSolverAlgo::PSATD,
            "algo.maxwell_solver = psatd is not supported because WarpX was built without spectral solvers");
#endif
#if defined(WARPX_DIM_1D_Z) && defined(WARPX_USE_FFT)
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( electromagnetic_solver_id != ElectromagneticSolverAlgo::PSATD,
            "algo.maxwell_solver = psatd is not available for 1D geometry");
#endif
#ifdef WARPX_DIM_RZ
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(Geom(0).isPeriodic(0) == 0,
            "The problem must not be periodic in the radial direction");

        // Ensure code aborts if PEC is specified at r=0 for RZ
        if (Geom(0).ProbLo(0) == 0){
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                WarpX::field_boundary_lo[0] == FieldBoundaryType::None,
                "Error : Field boundary at r=0 must be ``none``. \n");

            const ParmParse pp_boundary("boundary");
            if (pp_boundary.contains("particle_lo")) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    WarpX::particle_boundary_lo[0] == ParticleBoundaryType::None,
                    "Error : Particle boundary at r=0 must be ``none``. \n");
            }

        }

        if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
            // Force grid_type=collocated (neither staggered nor hybrid)
            // and use same shape factors in all directions for gathering
            grid_type = GridType::Collocated;
            galerkin_interpolation = false;
        }
#endif

        // note: current_deposition must be set after maxwell_solver (electromagnetic_solver_id) or
        //       do_electrostatic (electrostatic_solver_id) are already determined,
        //       because its default depends on the solver selection
        if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD ||
            electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC ||
            electrostatic_solver_id != ElectrostaticSolverAlgo::None) {
            current_deposition_algo = CurrentDepositionAlgo::Direct;
        }
        pp_algo.query_enum_sloppy("current_deposition", current_deposition_algo, "-_");
        pp_algo.query_enum_sloppy("charge_deposition", charge_deposition_algo, "-_");
        pp_algo.query_enum_sloppy("particle_pusher", particle_pusher_algo, "-_");

        // check for implicit evolve scheme
        if (evolve_scheme == EvolveScheme::SemiImplicitEM) {
            m_implicit_solver = std::make_unique<SemiImplicitEM>();
        }
        else if (evolve_scheme == EvolveScheme::ThetaImplicitEM) {
            m_implicit_solver = std::make_unique<ThetaImplicitEM>();
        }
        else if (evolve_scheme == EvolveScheme::StrangImplicitSpectralEM) {
            m_implicit_solver = std::make_unique<StrangImplicitSpectralEM>();
        }

        // implicit evolve schemes not setup to use mirrors
        if (evolve_scheme == EvolveScheme::SemiImplicitEM ||
            evolve_scheme == EvolveScheme::ThetaImplicitEM) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( m_num_mirrors == 0,
                "Mirrors cannot be used with Implicit evolve schemes.");
        }

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            current_deposition_algo != CurrentDepositionAlgo::Esirkepov ||
            !do_current_centering,
            "Current centering (nodal deposition) cannot be used with Esirkepov deposition."
            "Please set warpx.do_current_centering = 0 or algo.current_deposition = direct.");

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            current_deposition_algo != CurrentDepositionAlgo::Villasenor ||
            !do_current_centering,
            "Current centering (nodal deposition) cannot be used with Villasenor deposition."
            "Please set warpx.do_current_centering = 0 or algo.current_deposition = direct.");

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::current_deposition_algo != CurrentDepositionAlgo::Vay ||
            !do_current_centering,
            "Vay deposition not implemented with current centering");

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::current_deposition_algo != CurrentDepositionAlgo::Vay ||
            maxLevel() <= 0,
            "Vay deposition not implemented with mesh refinement");

        if (WarpX::current_deposition_algo == CurrentDepositionAlgo::Vay) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD,
                "Vay deposition is implemented only for PSATD");
        }

        if (WarpX::current_deposition_algo == CurrentDepositionAlgo::Vay) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                do_multi_J == false,
                "Vay deposition not implemented with multi-J algorithm");
        }

        if (current_deposition_algo == CurrentDepositionAlgo::Villasenor) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                evolve_scheme == EvolveScheme::SemiImplicitEM ||
                evolve_scheme == EvolveScheme::ThetaImplicitEM ||
                evolve_scheme == EvolveScheme::StrangImplicitSpectralEM,
                "Villasenor current deposition can only"
                "be used with Implicit evolve schemes.");
        }

        // Query algo.field_gathering from input, set field_gathering_algo to
        // "default" if not found (default defined in Utils/WarpXAlgorithmSelection.cpp)
        pp_algo.query_enum_sloppy("field_gathering", field_gathering_algo, "-_");

        // Set default field gathering algorithm for hybrid grids (momentum-conserving)
        std::string tmp_algo;
        // - algo.field_gathering not found in the input
        // - field_gathering_algo set to "default" above
        //   (default defined in Utils/WarpXAlgorithmSelection.cpp)
        // - reset default value here for hybrid grids
        if (!pp_algo.query("field_gathering", tmp_algo))
        {
            if (grid_type == GridType::Hybrid)
            {
                field_gathering_algo = GatheringAlgo::MomentumConserving;
            }
        }
        // - algo.field_gathering found in the input
        // - field_gathering_algo read above and set to user-defined value
        else
        {
            if (grid_type == GridType::Hybrid)
            {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    field_gathering_algo == GatheringAlgo::MomentumConserving,
                    "Hybrid grid (warpx.grid_type=hybrid) should be used only with "
                    "momentum-conserving field gathering algorithm "
                    "(algo.field_gathering=momentum-conserving)");
            }
        }

        // Use same shape factors in all directions
        // - with momentum-conserving field gathering
        if (field_gathering_algo == GatheringAlgo::MomentumConserving) {galerkin_interpolation = false;}
        // - with direct current deposition and the EM solver
        if( electromagnetic_solver_id != ElectromagneticSolverAlgo::None &&
            electromagnetic_solver_id != ElectromagneticSolverAlgo::HybridPIC ) {
            if (current_deposition_algo == CurrentDepositionAlgo::Direct) {
                galerkin_interpolation = false;
            }
        }

        {
            const ParmParse pp_interpolation("interpolation");
            pp_interpolation.query("galerkin_scheme",galerkin_interpolation);
        }

        // With the PSATD solver, momentum-conserving field gathering
        // combined with mesh refinement does not seem to work correctly
        // TODO Needs debugging
        if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD &&
            field_gathering_algo == GatheringAlgo::MomentumConserving &&
            maxLevel() > 0)
        {
            WARPX_ABORT_WITH_MESSAGE(
                "With the PSATD solver, momentum-conserving field gathering"
                " combined with mesh refinement is currently not implemented");
        }

        pp_algo.query_enum_sloppy("em_solver_medium", em_solver_medium, "-_");
        if (em_solver_medium == MediumForEM::Macroscopic ) {
            pp_algo.query_enum_sloppy("macroscopic_sigma_method",
                                      macroscopic_solver_algo, "-_");
        }

        if (evolve_scheme == EvolveScheme::SemiImplicitEM ||
            evolve_scheme == EvolveScheme::ThetaImplicitEM ||
            evolve_scheme == EvolveScheme::StrangImplicitSpectralEM) {

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                current_deposition_algo == CurrentDepositionAlgo::Esirkepov ||
                current_deposition_algo == CurrentDepositionAlgo::Villasenor ||
                current_deposition_algo == CurrentDepositionAlgo::Direct,
                "Only Esirkepov, Villasenor, or Direct current deposition supported with the implicit and semi-implicit schemes");

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee ||
                electromagnetic_solver_id == ElectromagneticSolverAlgo::CKC ||
                electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD,
                "Only the Yee, CKC, and PSATD EM solvers are supported with the implicit and semi-implicit schemes");

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                particle_pusher_algo == ParticlePusherAlgo::Boris ||
                particle_pusher_algo == ParticlePusherAlgo::HigueraCary,
                "Only the Boris and Higuera particle pushers are supported with the implicit and semi-implicit schemes");

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                field_gathering_algo != GatheringAlgo::MomentumConserving,
                    "With implicit and semi-implicit schemes, the momentum conserving field gather is not supported as it would not conserve energy");
        }
        if (evolve_scheme == EvolveScheme::StrangImplicitSpectralEM) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD,
                "With the strang_implicit_spectral_em evolve scheme, the algo.maxwell_solver must be psatd");
        }

        // Load balancing parameters
        std::vector<std::string> load_balance_intervals_string_vec = {"0"};
        pp_algo.queryarr("load_balance_intervals", load_balance_intervals_string_vec);
        load_balance_intervals = utils::parser::IntervalsParser(
            load_balance_intervals_string_vec);
        pp_algo.query("load_balance_with_sfc", load_balance_with_sfc);
        // Knapsack factor only used with non-SFC strategy
        if (!load_balance_with_sfc) {
            pp_algo.query("load_balance_knapsack_factor", load_balance_knapsack_factor);
        }
        utils::parser::queryWithParser(pp_algo, "load_balance_efficiency_ratio_threshold",
                        load_balance_efficiency_ratio_threshold);
        pp_algo.query_enum_sloppy("load_balance_costs_update", load_balance_costs_update_algo, "-_");
        if (WarpX::load_balance_costs_update_algo==LoadBalanceCostsUpdateAlgo::Heuristic) {
            utils::parser::queryWithParser(
                pp_algo, "costs_heuristic_cells_wt", costs_heuristic_cells_wt);
            utils::parser::queryWithParser(
                pp_algo, "costs_heuristic_particles_wt", costs_heuristic_particles_wt);
        }

        // Parse algo.particle_shape and check that input is acceptable
        // (do this only if there is at least one particle or laser species)
        const ParmParse pp_particles("particles");
        std::vector<std::string> species_names;
        pp_particles.queryarr("species_names", species_names);

        const ParmParse pp_lasers("lasers");
        std::vector<std::string> lasers_names;
        pp_lasers.queryarr("names", lasers_names);

#ifdef WARPX_DIM_RZ
        // Here we check if the simulation includes laser and the number of
        // azimuthal modes is less than 2.
        // In that case we should throw a specific warning since
        // representation of a laser pulse in cylindrical coordinates
        // requires at least 2 azimuthal modes
        if (!lasers_names.empty() && n_rz_azimuthal_modes < 2) {
            ablastr::warn_manager::WMRecordWarning("Laser",
            "Laser pulse representation in RZ requires at least 2 azimuthal modes",
            ablastr::warn_manager::WarnPriority::high);
        }
#endif

        std::vector<std::string> sort_intervals_string_vec = {"-1"};
        int particle_shape;
        if (!species_names.empty() || !lasers_names.empty()) {
            if (utils::parser::queryWithParser(pp_algo, "particle_shape", particle_shape)){
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    (particle_shape >= 1) && (particle_shape <=4),
                    "algo.particle_shape can be only 1, 2, 3, or 4"
                );

                nox = particle_shape;
                noy = particle_shape;
                noz = particle_shape;
            }
            else{
                WARPX_ABORT_WITH_MESSAGE(
                    "algo.particle_shape must be set in the input file:"
                    " please set algo.particle_shape to 1, 2, 3, or 4");
            }

            if ((maxLevel() > 0) && (particle_shape > 1) && (do_pml_j_damping == 1))
            {
                ablastr::warn_manager::WMRecordWarning("Particles",
                    "When algo.particle_shape > 1,"
                    "some numerical artifact will be present at the interface between coarse and fine patch."
                    "We recommend setting algo.particle_shape = 1 in order to avoid this issue");
            }

            // default sort interval for particles if species or lasers vector is not empty
#ifdef AMREX_USE_GPU
            sort_intervals_string_vec = {"4"};
#else
            sort_intervals_string_vec = {"-1"};
#endif
        }

        const amrex::ParmParse pp_warpx("warpx");
        pp_warpx.queryarr("sort_intervals", sort_intervals_string_vec);
        sort_intervals = utils::parser::IntervalsParser(sort_intervals_string_vec);

        Vector<int> vect_sort_bin_size(AMREX_SPACEDIM,1);
        const bool sort_bin_size_is_specified =
            utils::parser::queryArrWithParser(
                pp_warpx, "sort_bin_size",
                vect_sort_bin_size, 0, AMREX_SPACEDIM);
        if (sort_bin_size_is_specified){
            for (int i=0; i<AMREX_SPACEDIM; i++) {
                sort_bin_size[i] = vect_sort_bin_size[i];
            }
        }

        pp_warpx.query("sort_particles_for_deposition",sort_particles_for_deposition);
        Vector<int> vect_sort_idx_type(AMREX_SPACEDIM,0);
        const bool sort_idx_type_is_specified =
            utils::parser::queryArrWithParser(
                pp_warpx, "sort_idx_type",
                vect_sort_idx_type, 0, AMREX_SPACEDIM);
        if (sort_idx_type_is_specified){
            for (int i=0; i<AMREX_SPACEDIM; i++) {
                sort_idx_type[i] = vect_sort_idx_type[i];
            }
        }

    }

    {
        const ParmParse pp_warpx("warpx");

        // If warpx.grid_type=staggered or warpx.grid_type=hybrid,
        // and algo.field_gathering=momentum-conserving, the fields are solved
        // on a staggered grid and centered onto a nodal grid for gathering.
        // Instead, if warpx.grid_type=collocated, the momentum-conserving and
        // energy conserving field gathering algorithms are equivalent (forces
        // gathered from the collocated grid) and no fields centering occurs.
        if ((WarpX::field_gathering_algo == GatheringAlgo::MomentumConserving
            && WarpX::grid_type != GridType::Collocated)
            || WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameElectroMagnetostatic)
        {
            utils::parser::queryWithParser(
                pp_warpx, "field_centering_nox", field_centering_nox);
            utils::parser::queryWithParser(
                pp_warpx, "field_centering_noy", field_centering_noy);
            utils::parser::queryWithParser(
                pp_warpx, "field_centering_noz", field_centering_noz);

            AllocateCenteringCoefficients(device_field_centering_stencil_coeffs_x,
                                          device_field_centering_stencil_coeffs_y,
                                          device_field_centering_stencil_coeffs_z,
                                          field_centering_nox,
                                          field_centering_noy,
                                          field_centering_noz,
                                          grid_type);
        }

        // Finite-order centering is not implemented with mesh refinement
        // (note that when warpx.grid_type=collocated, finite-order centering is not used anyways)
        if (maxLevel() > 0 && WarpX::grid_type != GridType::Collocated)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                field_centering_nox == 2 && field_centering_noy == 2 && field_centering_noz == 2,
                "High-order centering of fields (order > 2) is not implemented with mesh refinement");
        }
    }

    if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD)
    {
        const ParmParse pp_psatd("psatd");
        pp_psatd.query("periodic_single_box_fft", fft_periodic_single_box);

        std::string nox_str;
        std::string noy_str;
        std::string noz_str;

        pp_psatd.query("nox", nox_str);
        pp_psatd.query("noy", noy_str);
        pp_psatd.query("noz", noz_str);

        if(nox_str == "inf") {
            nox_fft = -1;
        } else {
            utils::parser::queryWithParser(pp_psatd, "nox", nox_fft);
        }
        if(noy_str == "inf") {
            noy_fft = -1;
        } else {
            utils::parser::queryWithParser(pp_psatd, "noy", noy_fft);
        }
        if(noz_str == "inf") {
            noz_fft = -1;
        } else {
            utils::parser::queryWithParser(pp_psatd, "noz", noz_fft);
        }

        if (!fft_periodic_single_box) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(nox_fft > 0, "PSATD order must be finite unless psatd.periodic_single_box_fft is used");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(noy_fft > 0, "PSATD order must be finite unless psatd.periodic_single_box_fft is used");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(noz_fft > 0, "PSATD order must be finite unless psatd.periodic_single_box_fft is used");
        }

        // Integer that corresponds to the order of the PSATD solution
        // (whether the PSATD equations are derived from first-order or
        // second-order solution)
        pp_psatd.query_enum_sloppy("solution_type", m_psatd_solution_type, "-_");

        // Integers that correspond to the time dependency of J (constant, linear)
        // and rho (linear, quadratic) for the PSATD algorithm
        pp_psatd.query_enum_sloppy("J_in_time", J_in_time, "-_");
        pp_psatd.query_enum_sloppy("rho_in_time", rho_in_time, "-_");

        if (m_psatd_solution_type != PSATDSolutionType::FirstOrder || !do_multi_J)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                rho_in_time == RhoInTime::Linear,
                "psatd.rho_in_time=constant not yet implemented, "
                "except for psatd.solution_type=first-order and warpx.do_multi_J=1");
        }

        // Current correction activated by default, unless a charge-conserving
        // current deposition (Esirkepov, Vay) or the div(E) cleaning scheme
        // are used
        if (WarpX::current_deposition_algo == CurrentDepositionAlgo::Esirkepov ||
            WarpX::current_deposition_algo == CurrentDepositionAlgo::Villasenor ||
            WarpX::current_deposition_algo == CurrentDepositionAlgo::Vay ||
            WarpX::do_dive_cleaning)
        {
            current_correction = false;
        }

        // TODO Remove this default when current correction will
        // be implemented for the multi-J algorithm as well.
        if (do_multi_J) { current_correction = false; }

        pp_psatd.query("current_correction", current_correction);

        if (!current_correction &&
            current_deposition_algo != CurrentDepositionAlgo::Esirkepov &&
            current_deposition_algo != CurrentDepositionAlgo::Villasenor &&
            current_deposition_algo != CurrentDepositionAlgo::Vay)
        {
            ablastr::warn_manager::WMRecordWarning(
                "Algorithms",
                "The chosen current deposition algorithm does not guarantee"
                " charge conservation, and no additional current correction"
                " algorithm is activated in order to compensate for that."
                " Lack of charge conservation may negatively affect the"
                " results of the simulation.",
                ablastr::warn_manager::WarnPriority::low);
        }

        pp_psatd.query("do_time_averaging", fft_do_time_averaging);

        if (WarpX::current_deposition_algo == CurrentDepositionAlgo::Vay)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !fft_periodic_single_box,
                "Option algo.current_deposition=vay must be used with psatd.periodic_single_box_fft=0.");
        }

        if (current_deposition_algo == CurrentDepositionAlgo::Vay)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !current_correction,
                "Options algo.current_deposition=vay and psatd.current_correction=1 cannot be combined together.");
        }

        // Auxiliary: boosted_frame = true if WarpX::gamma_boost is set in the inputs
        const amrex::ParmParse pp_warpx("warpx");
        const bool boosted_frame = pp_warpx.query("gamma_boost", gamma_boost);

        // Check whether the default Galilean velocity should be used
        bool use_default_v_galilean = false;
        pp_psatd.query("use_default_v_galilean", use_default_v_galilean);

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !use_default_v_galilean || boosted_frame,
            "psatd.use_default_v_galilean = 1 can be used only if WarpX::gamma_boost is also set"
        );

        if (use_default_v_galilean && boosted_frame)
        {
            m_v_galilean[2] = -std::sqrt(1._rt - 1._rt / (gamma_boost * gamma_boost));
        }
        else
        {
            utils::parser::queryArrWithParser(
                pp_psatd, "v_galilean", m_v_galilean, 0, 3);
        }

        // Check whether the default comoving velocity should be used
        bool use_default_v_comoving = false;
        pp_psatd.query("use_default_v_comoving", use_default_v_comoving);

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !use_default_v_comoving || boosted_frame,
            "psatd.use_default_v_comoving = 1 can be used only if WarpX::gamma_boost is also set"
        );

        if (use_default_v_comoving && boosted_frame)
        {
            m_v_comoving[2] = -std::sqrt(1._rt - 1._rt / (gamma_boost * gamma_boost));
        }
        else
        {
            utils::parser::queryArrWithParser(
                pp_psatd, "v_comoving", m_v_comoving, 0, 3);
        }

        // Scale the Galilean/comoving velocity by the speed of light
        for (auto& vv : m_v_galilean) { vv*= PhysConst::c; }
        for (auto& vv : m_v_comoving) { vv*= PhysConst::c; }

        const auto v_galilean_is_zero =
            std::all_of(m_v_galilean.begin(), m_v_galilean.end(),
            [](const auto& val){return val == 0.;});

        const auto v_comoving_is_zero =
            std::all_of(m_v_comoving.begin(), m_v_comoving.end(),
            [](const auto& val){return val == 0.;});


        // Galilean and comoving algorithms should not be used together
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            v_galilean_is_zero || v_comoving_is_zero,
            "Galilean and comoving algorithms should not be used together"
        );


        if (current_deposition_algo == CurrentDepositionAlgo::Esirkepov ||
            current_deposition_algo == CurrentDepositionAlgo::Villasenor) {

            // The comoving PSATD algorithm is not implemented nor tested with Esirkepov current deposition
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(v_comoving_is_zero,
                "charge-conserving current depositions (Esirkepov and Villasenor) cannot be used with the comoving PSATD algorithm");

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(v_galilean_is_zero,
                "charge-conserving current depositions (Esirkepov and Villasenor) cannot be used with the Galilean algorithm.");
        }

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (current_deposition_algo != CurrentDepositionAlgo::Vay) ||
            v_galilean_is_zero,
            "Vay current deposition not implemented for Galilean algorithms"
        );

#   ifdef WARPX_DIM_RZ
        update_with_rho = true;
#   else
        if (m_v_galilean[0] == 0. && m_v_galilean[1] == 0. && m_v_galilean[2] == 0. &&
            m_v_comoving[0] == 0. && m_v_comoving[1] == 0. && m_v_comoving[2] == 0.) {
            update_with_rho = do_dive_cleaning; // standard PSATD
        }
        else {
            update_with_rho = true;  // Galilean PSATD or comoving PSATD
        }
#   endif

        // Overwrite update_with_rho with value set in input file
        pp_psatd.query("update_with_rho", update_with_rho);

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (!do_dive_cleaning) || update_with_rho,
            "warpx.do_dive_cleaning = 1 not implemented with psatd.update_with_rho = 0"
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            v_comoving_is_zero || update_with_rho,
            "psatd.update_with_rho must be equal to 1 for comoving PSATD"
        );

        if (do_multi_J)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                v_galilean_is_zero,
                "Multi-J algorithm not implemented with Galilean PSATD"
            );
        }

        if (J_in_time == JInTime::Linear)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                update_with_rho,
                "psatd.update_with_rho must be set to 1 when psatd.J_in_time=linear");

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                v_galilean_is_zero,
                "psatd.J_in_time=linear not implemented with Galilean PSATD");

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                v_comoving_is_zero,
                "psatd.J_in_time=linear not implemented with comoving PSATD");

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !current_correction,
                "psatd.current_correction=1 not implemented with psatd.J_in_time=linear");

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                current_deposition_algo != CurrentDepositionAlgo::Vay,
                "algo.current_deposition=vay not implemented with psatd.J_in_time=linear");
        }

        for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
        {
            if (WarpX::field_boundary_lo[dir] == FieldBoundaryType::Damped ||
                WarpX::field_boundary_hi[dir] == FieldBoundaryType::Damped ) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    WarpX::field_boundary_lo[dir] == WarpX::field_boundary_hi[dir],
                    "field boundary in both lo and hi must be set to Damped for PSATD"
                );
            }
        }

        // Fill guard cells with backward FFTs in directions with field damping
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
        {
            if (WarpX::field_boundary_lo[dir] == FieldBoundaryType::Damped ||
                WarpX::field_boundary_hi[dir] == FieldBoundaryType::Damped)
            {
                WarpX::m_fill_guards_fields[dir] = 1;
            }
        }

        // Without periodic single box, fill guard cells with backward FFTs,
        // with current correction or Vay deposition
        if (!fft_periodic_single_box)
        {
            if (current_correction ||
                current_deposition_algo == CurrentDepositionAlgo::Vay)
            {
                WarpX::m_fill_guards_current = amrex::IntVect(1);
            }
        }
    }

    if (electromagnetic_solver_id != ElectromagneticSolverAlgo::PSATD ) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (WarpX::field_boundary_lo[idim] != FieldBoundaryType::Damped) &&
                (WarpX::field_boundary_hi[idim] != FieldBoundaryType::Damped),
                "FieldBoundaryType::Damped is only supported for PSATD"
            );
        }
    }

    // Setup pec_insulator boundary conditions
    pec_insulator_boundary = std::make_unique<PEC_Insulator>();

    // for slice generation //
    {
        const ParmParse pp_slice("slice");
        amrex::Vector<Real> slice_lo(AMREX_SPACEDIM);
        amrex::Vector<Real> slice_hi(AMREX_SPACEDIM);
        Vector<int> slice_crse_ratio(AMREX_SPACEDIM);
        // set default slice_crse_ratio //
        for (int idim=0; idim < AMREX_SPACEDIM; ++idim )
        {
            slice_crse_ratio[idim] = 1;
        }
        utils::parser::queryArrWithParser(
        pp_slice, "dom_lo", slice_lo, 0, AMREX_SPACEDIM);
        utils::parser::queryArrWithParser(
        pp_slice, "dom_hi", slice_hi, 0, AMREX_SPACEDIM);
        utils::parser::queryArrWithParser(
        pp_slice, "coarsening_ratio",slice_crse_ratio,0,AMREX_SPACEDIM);
        utils::parser::queryWithParser(
        pp_slice, "plot_int",slice_plot_int);
        slice_realbox.setLo(slice_lo);
        slice_realbox.setHi(slice_hi);
        slice_cr_ratio = IntVect(AMREX_D_DECL(1,1,1));
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            if (slice_crse_ratio[idim] > 1 ) {
                slice_cr_ratio[idim] = slice_crse_ratio[idim];
            }
        }
    }
}

void
WarpX::BackwardCompatibility ()
{
    // Auxiliary variables
    int backward_int;
    bool backward_bool;
    std::string backward_str;
    amrex::Real backward_Real;

    const ParmParse pp_amr("amr");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_amr.query("plot_int", backward_int),
        "amr.plot_int is not supported anymore. Please use the new syntax for diagnostics:\n"
        "diagnostics.diags_names = my_diag\n"
        "my_diag.intervals = 10\n"
        "for output every 10 iterations. See documentation for more information"
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_amr.query("plot_file", backward_str),
        "amr.plot_file is not supported anymore. "
        "Please use the new syntax for diagnostics, see documentation."
    );

    const ParmParse pp_warpx("warpx");
    std::vector<std::string> backward_strings;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.queryarr("fields_to_plot", backward_strings),
        "warpx.fields_to_plot is not supported anymore. "
        "Please use the new syntax for diagnostics, see documentation."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.query("plot_finepatch", backward_int),
        "warpx.plot_finepatch is not supported anymore. "
        "Please use the new syntax for diagnostics, see documentation."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.query("plot_crsepatch", backward_int),
        "warpx.plot_crsepatch is not supported anymore. "
        "Please use the new syntax for diagnostics, see documentation."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.queryarr("load_balance_int", backward_strings),
        "warpx.load_balance_int is no longer a valid option. "
        "Please use the renamed option algo.load_balance_intervals instead."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.queryarr("load_balance_intervals", backward_strings),
        "warpx.load_balance_intervals is no longer a valid option. "
        "Please use the renamed option algo.load_balance_intervals instead."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.query("load_balance_efficiency_ratio_threshold", backward_Real),
        "warpx.load_balance_efficiency_ratio_threshold is not supported anymore. "
        "Please use the renamed option algo.load_balance_efficiency_ratio_threshold."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.query("load_balance_with_sfc", backward_int),
        "warpx.load_balance_with_sfc is not supported anymore. "
        "Please use the renamed option algo.load_balance_with_sfc."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.query("load_balance_knapsack_factor", backward_Real),
        "warpx.load_balance_knapsack_factor is not supported anymore. "
        "Please use the renamed option algo.load_balance_knapsack_factor."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.queryarr("override_sync_int", backward_strings),
        "warpx.override_sync_int is no longer a valid option. "
        "Please use the renamed option warpx.override_sync_intervals instead."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.queryarr("sort_int", backward_strings),
        "warpx.sort_int is no longer a valid option. "
        "Please use the renamed option warpx.sort_intervals instead."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.query("do_nodal", backward_int),
        "warpx.do_nodal is not supported anymore. "
        "Please use the flag warpx.grid_type instead."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.query("use_kspace_filter", backward_int),
        "warpx.use_kspace_filter is not supported anymore. "
        "Please use the flag use_filter, see documentation."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.query("do_pml", backward_int),
        "do_pml is not supported anymore. Please use boundary.field_lo and boundary.field_hi"
        " to set the boundary conditions."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.query("serialize_ics", backward_bool),
        "warpx.serialize_ics is no longer a valid option. "
        "Please use the renamed option warpx.serialize_initial_conditions instead."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.query("do_back_transformed_diagnostics", backward_int),
        "Legacy back-transformed diagnostics are not supported anymore. "
        "Please use the new syntax for back-transformed diagnostics, see documentation."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.query("lab_data_directory", backward_str),
        "Legacy back-transformed diagnostics are not supported anymore. "
        "Please use the new syntax for back-transformed diagnostics, see documentation."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.query("num_snapshots_lab", backward_int),
        "Legacy back-transformed diagnostics are not supported anymore. "
        "Please use the new syntax for back-transformed diagnostics, see documentation."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.query("dt_snapshots_lab", backward_Real),
        "Legacy back-transformed diagnostics are not supported anymore. "
        "Please use the new syntax for back-transformed diagnostics, see documentation."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.query("dz_snapshots_lab", backward_Real),
        "Legacy back-transformed diagnostics are not supported anymore. "
        "Please use the new syntax for back-transformed diagnostics, see documentation."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.query("do_back_transformed_fields", backward_int),
        "Legacy back-transformed diagnostics are not supported anymore. "
        "Please use the new syntax for back-transformed diagnostics, see documentation."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.query("buffer_size", backward_int),
        "Legacy back-transformed diagnostics are not supported anymore. "
        "Please use the new syntax for back-transformed diagnostics, see documentation."
    );

    const ParmParse pp_slice("slice");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_slice.query("num_slice_snapshots_lab", backward_int),
        "Legacy back-transformed diagnostics are not supported anymore. "
        "Please use the new syntax for back-transformed diagnostics, see documentation."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_slice.query("dt_slice_snapshots_lab", backward_Real),
        "Legacy back-transformed diagnostics are not supported anymore. "
        "Please use the new syntax for back-transformed diagnostics, see documentation."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_slice.query("particle_slice_width_lab", backward_Real),
        "Legacy back-transformed diagnostics are not supported anymore. "
        "Please use the new syntax for back-transformed diagnostics, see documentation."
    );

    const ParmParse pp_interpolation("interpolation");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_interpolation.query("nox", backward_int) &&
        !pp_interpolation.query("noy", backward_int) &&
        !pp_interpolation.query("noz", backward_int),
        "interpolation.nox (as well as .noy, .noz) are not supported anymore."
        " Please use the new syntax algo.particle_shape instead"
    );

    const ParmParse pp_algo("algo");
    int backward_mw_solver;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_algo.query("maxwell_fdtd_solver", backward_mw_solver),
        "algo.maxwell_fdtd_solver is not supported anymore. "
        "Please use the renamed option algo.maxwell_solver");

    const ParmParse pp_particles("particles");
    int nspecies;
    if (pp_particles.query("nspecies", nspecies)){
        ablastr::warn_manager::WMRecordWarning("Species",
            "particles.nspecies is ignored. Just use particles.species_names please.",
            ablastr::warn_manager::WarnPriority::low);
    }

    std::vector<std::string> backward_sp_names;
    pp_particles.queryarr("species_names", backward_sp_names);
    for(const std::string& speciesiter : backward_sp_names){
        const ParmParse pp_species(speciesiter);
        std::vector<amrex::Real> backward_vel;
        std::stringstream ssspecies;

        ssspecies << "'" << speciesiter << ".multiple_particles_vel_<x,y,z>'";
        ssspecies << " are not supported anymore. ";
        ssspecies << "Please use the renamed variables ";
        ssspecies << "'" << speciesiter << ".multiple_particles_u<x,y,z>' .";
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_species.queryarr("multiple_particles_vel_x", backward_vel) &&
            !pp_species.queryarr("multiple_particles_vel_y", backward_vel) &&
            !pp_species.queryarr("multiple_particles_vel_z", backward_vel),
            ssspecies.str());

        ssspecies.str("");
        ssspecies.clear();
        ssspecies << "'" << speciesiter << ".single_particle_vel'";
        ssspecies << " is not supported anymore. ";
        ssspecies << "Please use the renamed variable ";
        ssspecies << "'" << speciesiter << ".single_particle_u' .";
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_species.queryarr("single_particle_vel", backward_vel),
            ssspecies.str());
    }

    const ParmParse pp_collisions("collisions");
    int ncollisions;
    if (pp_collisions.query("ncollisions", ncollisions)){
        ablastr::warn_manager::WMRecordWarning("Collisions",
            "collisions.ncollisions is ignored. Just use particles.collision_names please.",
            ablastr::warn_manager::WarnPriority::low);
    }

    const ParmParse pp_lasers("lasers");
    int nlasers;
    if (pp_lasers.query("nlasers", nlasers)){
        ablastr::warn_manager::WMRecordWarning("Laser",
            "lasers.nlasers is ignored. Just use lasers.names please.",
            ablastr::warn_manager::WarnPriority::low);
    }
}

// This is a virtual function.
void
WarpX::MakeNewLevelFromScratch (int lev, Real time, const BoxArray& new_grids,
                                const DistributionMapping& new_dmap)
{
    AllocLevelData(lev, new_grids, new_dmap);
    InitLevelData(lev, time);
}

// This is a virtual function.
void
WarpX::MakeNewLevelFromCoarse (int /*lev*/, amrex::Real /*time*/, const amrex::BoxArray& /*ba*/,
                                         const amrex::DistributionMapping& /*dm*/)
{
    WARPX_ABORT_WITH_MESSAGE("MakeNewLevelFromCoarse: To be implemented");
}

void
WarpX::ClearLevel (int lev)
{
    m_fields.clear_level(lev);

    for (int i = 0; i < 3; ++i) {
        Efield_dotMask [lev][i].reset();
        Bfield_dotMask [lev][i].reset();
        Afield_dotMask [lev][i].reset();
    }

    phi_dotMask[lev].reset();

#ifdef WARPX_USE_FFT
    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
        spectral_solver_fp[lev].reset();
        spectral_solver_cp[lev].reset();
    }
#endif

    costs[lev].reset();
    load_balance_efficiency[lev] = -1;
}

void
WarpX::AllocLevelData (int lev, const BoxArray& ba, const DistributionMapping& dm)
{
    const bool aux_is_nodal = (field_gathering_algo == GatheringAlgo::MomentumConserving);

    const Real* dx = Geom(lev).CellSize();

    // Initialize filter before guard cells manager
    // (needs info on length of filter's stencil)
    if (use_filter)
    {
        InitFilter();
    }

    guard_cells.Init(
        dt[lev],
        dx,
        m_do_subcycling,
        WarpX::use_fdtd_nci_corr,
        grid_type,
        do_moving_window,
        moving_window_dir,
        WarpX::nox,
        nox_fft, noy_fft, noz_fft,
        NCIGodfreyFilter::m_stencil_width,
        electromagnetic_solver_id,
        maxLevel(),
        WarpX::m_v_galilean,
        WarpX::m_v_comoving,
        m_safe_guard_cells,
        WarpX::do_multi_J,
        WarpX::fft_do_time_averaging,
        ::isAnyBoundaryPML(field_boundary_lo, field_boundary_hi),
        WarpX::do_pml_in_domain,
        WarpX::pml_ncell,
        this->refRatio(),
        use_filter,
        bilinear_filter.stencil_length_each_dir);

#ifdef AMREX_USE_EB
    bool const eb_enabled = EB::enabled();
    if (eb_enabled) {
        int const max_guard = guard_cells.ng_FieldSolver.max();
        m_field_factory[lev] = amrex::makeEBFabFactory(Geom(lev), ba, dm,
                                                       {max_guard, max_guard, max_guard},
                                                       amrex::EBSupport::full);
    } else
#endif
    {
        m_field_factory[lev] = std::make_unique<FArrayBoxFactory>();
    }


    if (mypc->nSpeciesDepositOnMainGrid() && n_current_deposition_buffer == 0) {
        n_current_deposition_buffer = 1;
        // This forces the allocation of buffers and allows the code associated
        // with buffers to run. But the buffer size of `1` is in fact not used,
        // `deposit_on_main_grid` forces all particles (whether or not they
        // are in buffers) to deposition on the main grid.
    }

    if (n_current_deposition_buffer < 0) {
        n_current_deposition_buffer = guard_cells.ng_alloc_J.max();
    }
    if (n_field_gather_buffer < 0) {
        // Field gather buffer should be larger than current deposition buffers
        n_field_gather_buffer = n_current_deposition_buffer + 1;
    }

    AllocLevelMFs(lev, ba, dm, guard_cells.ng_alloc_EB, guard_cells.ng_alloc_J,
                  guard_cells.ng_alloc_Rho, guard_cells.ng_alloc_F, guard_cells.ng_alloc_G, aux_is_nodal);

    m_accelerator_lattice[lev] = std::make_unique<AcceleratorLattice>();
    m_accelerator_lattice[lev]->InitElementFinder(lev, gamma_boost, ba, dm);

}

void
WarpX::AllocLevelMFs (int lev, const BoxArray& ba, const DistributionMapping& dm,
                      const IntVect& ngEB, IntVect& ngJ, const IntVect& ngRho,
                      const IntVect& ngF, const IntVect& ngG, const bool aux_is_nodal)
{
    using ablastr::fields::Direction;

    // Declare nodal flags
    IntVect Ex_nodal_flag, Ey_nodal_flag, Ez_nodal_flag;
    IntVect Bx_nodal_flag, By_nodal_flag, Bz_nodal_flag;
    IntVect jx_nodal_flag, jy_nodal_flag, jz_nodal_flag;
    IntVect rho_nodal_flag;
    IntVect phi_nodal_flag;
    amrex::IntVect F_nodal_flag, G_nodal_flag;

    // Set nodal flags
#if   defined(WARPX_DIM_1D_Z)
    // AMReX convention: x = missing dimension, y = missing dimension, z = only dimension
    Ex_nodal_flag = IntVect(1);
    Ey_nodal_flag = IntVect(1);
    Ez_nodal_flag = IntVect(0);
    Bx_nodal_flag = IntVect(0);
    By_nodal_flag = IntVect(0);
    Bz_nodal_flag = IntVect(1);
    jx_nodal_flag = IntVect(1);
    jy_nodal_flag = IntVect(1);
    jz_nodal_flag = IntVect(0);
#elif   defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    // AMReX convention: x = first dimension, y = missing dimension, z = second dimension
    Ex_nodal_flag = IntVect(0,1);
    Ey_nodal_flag = IntVect(1,1);
    Ez_nodal_flag = IntVect(1,0);
    Bx_nodal_flag = IntVect(1,0);
    By_nodal_flag = IntVect(0,0);
    Bz_nodal_flag = IntVect(0,1);
    jx_nodal_flag = IntVect(0,1);
    jy_nodal_flag = IntVect(1,1);
    jz_nodal_flag = IntVect(1,0);
#elif defined(WARPX_DIM_3D)
    Ex_nodal_flag = IntVect(0,1,1);
    Ey_nodal_flag = IntVect(1,0,1);
    Ez_nodal_flag = IntVect(1,1,0);
    Bx_nodal_flag = IntVect(1,0,0);
    By_nodal_flag = IntVect(0,1,0);
    Bz_nodal_flag = IntVect(0,0,1);
    jx_nodal_flag = IntVect(0,1,1);
    jy_nodal_flag = IntVect(1,0,1);
    jz_nodal_flag = IntVect(1,1,0);
#endif
    if (electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameElectroMagnetostatic)
    {
        jx_nodal_flag  = IntVect::TheNodeVector();
        jy_nodal_flag  = IntVect::TheNodeVector();
        jz_nodal_flag  = IntVect::TheNodeVector();
        ngJ = ngRho;
    }
    rho_nodal_flag = IntVect( AMREX_D_DECL(1,1,1) );
    phi_nodal_flag = IntVect::TheNodeVector();
    F_nodal_flag = amrex::IntVect::TheNodeVector();
    G_nodal_flag = amrex::IntVect::TheCellVector();

    // Overwrite nodal flags if necessary
    if (grid_type == GridType::Collocated) {
        Ex_nodal_flag  = IntVect::TheNodeVector();
        Ey_nodal_flag  = IntVect::TheNodeVector();
        Ez_nodal_flag  = IntVect::TheNodeVector();
        Bx_nodal_flag  = IntVect::TheNodeVector();
        By_nodal_flag  = IntVect::TheNodeVector();
        Bz_nodal_flag  = IntVect::TheNodeVector();
        jx_nodal_flag  = IntVect::TheNodeVector();
        jy_nodal_flag  = IntVect::TheNodeVector();
        jz_nodal_flag  = IntVect::TheNodeVector();
        rho_nodal_flag = IntVect::TheNodeVector();
        G_nodal_flag = amrex::IntVect::TheNodeVector();
    }
#ifdef WARPX_DIM_RZ
    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
        // Force cell-centered IndexType in r and z
        Ex_nodal_flag  = IntVect::TheCellVector();
        Ey_nodal_flag  = IntVect::TheCellVector();
        Ez_nodal_flag  = IntVect::TheCellVector();
        Bx_nodal_flag  = IntVect::TheCellVector();
        By_nodal_flag  = IntVect::TheCellVector();
        Bz_nodal_flag  = IntVect::TheCellVector();
        jx_nodal_flag  = IntVect::TheCellVector();
        jy_nodal_flag  = IntVect::TheCellVector();
        jz_nodal_flag  = IntVect::TheCellVector();
        rho_nodal_flag = IntVect::TheCellVector();
        F_nodal_flag = IntVect::TheCellVector();
        G_nodal_flag = IntVect::TheCellVector();
    }

    // With RZ multimode, there is a real and imaginary component
    // for each mode, except mode 0 which is purely real
    // Component 0 is mode 0.
    // Odd components are the real parts.
    // Even components are the imaginary parts.
    ncomps = n_rz_azimuthal_modes*2 - 1;
#endif

    // Set global rho nodal flag to know about rho index type when rho MultiFab is not allocated
    m_rho_nodal_flag = rho_nodal_flag;

    //
    // The fine patch
    //
    const std::array<Real,3> dx = CellSize(lev);

    m_fields.alloc_init(FieldType::Bfield_fp, Direction{0}, lev, amrex::convert(ba, Bx_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
    m_fields.alloc_init(FieldType::Bfield_fp, Direction{1}, lev, amrex::convert(ba, By_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
    m_fields.alloc_init(FieldType::Bfield_fp, Direction{2}, lev, amrex::convert(ba, Bz_nodal_flag), dm, ncomps, ngEB, 0.0_rt);

    m_fields.alloc_init(FieldType::Efield_fp, Direction{0}, lev, amrex::convert(ba, Ex_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
    m_fields.alloc_init(FieldType::Efield_fp, Direction{1}, lev, amrex::convert(ba, Ey_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
    m_fields.alloc_init(FieldType::Efield_fp, Direction{2}, lev, amrex::convert(ba, Ez_nodal_flag), dm, ncomps, ngEB, 0.0_rt);

    m_fields.alloc_init(FieldType::current_fp, Direction{0}, lev, amrex::convert(ba, jx_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
    m_fields.alloc_init(FieldType::current_fp, Direction{1}, lev, amrex::convert(ba, jy_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
    m_fields.alloc_init(FieldType::current_fp, Direction{2}, lev, amrex::convert(ba, jz_nodal_flag), dm, ncomps, ngJ, 0.0_rt);

    if (do_current_centering)
    {
        amrex::BoxArray const& nodal_ba = amrex::convert(ba, amrex::IntVect::TheNodeVector());
        m_fields.alloc_init(FieldType::current_fp_nodal, Direction{0}, lev, nodal_ba, dm, ncomps, ngJ, 0.0_rt);
        m_fields.alloc_init(FieldType::current_fp_nodal, Direction{1}, lev, nodal_ba, dm, ncomps, ngJ, 0.0_rt);
        m_fields.alloc_init(FieldType::current_fp_nodal, Direction{2}, lev, nodal_ba, dm, ncomps, ngJ, 0.0_rt);
    }

    if (WarpX::current_deposition_algo == CurrentDepositionAlgo::Vay)
    {
        m_fields.alloc_init(FieldType::current_fp_vay, Direction{0}, lev, amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
        m_fields.alloc_init(FieldType::current_fp_vay, Direction{1}, lev, amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
        m_fields.alloc_init(FieldType::current_fp_vay, Direction{2}, lev, amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
    }

    if (electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameElectroMagnetostatic)
    {
        m_fields.alloc_init(FieldType::vector_potential_fp_nodal, Direction{0}, lev, amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngRho, 0.0_rt);
        m_fields.alloc_init(FieldType::vector_potential_fp_nodal, Direction{1}, lev, amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngRho, 0.0_rt);
        m_fields.alloc_init(FieldType::vector_potential_fp_nodal, Direction{2}, lev, amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngRho, 0.0_rt);

        // Memory buffers for computing magnetostatic fields
        // Vector Potential A and previous step.  Time buffer needed for computing dA/dt to first order
        m_fields.alloc_init(FieldType::vector_potential_grad_buf_e_stag, Direction{0}, lev, amrex::convert(ba, Ex_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::vector_potential_grad_buf_e_stag, Direction{1}, lev, amrex::convert(ba, Ey_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::vector_potential_grad_buf_e_stag, Direction{2}, lev, amrex::convert(ba, Ez_nodal_flag), dm, ncomps, ngEB, 0.0_rt);

        m_fields.alloc_init(FieldType::vector_potential_grad_buf_b_stag, Direction{0}, lev, amrex::convert(ba, Bx_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::vector_potential_grad_buf_b_stag, Direction{1}, lev, amrex::convert(ba, By_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::vector_potential_grad_buf_b_stag, Direction{2}, lev, amrex::convert(ba, Bz_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
    }

    // Allocate extra multifabs needed by the kinetic-fluid hybrid algorithm.
    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC)
    {
        m_hybrid_pic_model->AllocateLevelMFs(
            m_fields,
            lev, ba, dm, ncomps, ngJ, ngRho, jx_nodal_flag, jy_nodal_flag,
            jz_nodal_flag, rho_nodal_flag
        );
    }

    // Allocate extra multifabs needed for fluids
    if (do_fluid_species) {
        myfl->AllocateLevelMFs(m_fields, ba, dm, lev);
        auto & warpx = GetInstance();
        const amrex::Real cur_time = warpx.gett_new(lev);
        myfl->InitData(m_fields, geom[lev].Domain(), cur_time, lev);
    }

    // Allocate extra multifabs for macroscopic properties of the medium
    if (em_solver_medium == MediumForEM::Macroscopic) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( lev==0,
            "Macroscopic properties are not supported with mesh refinement.");
        m_macroscopic_properties->AllocateLevelMFs(ba, dm, ngEB);
    }

    if (fft_do_time_averaging)
    {
        m_fields.alloc_init(FieldType::Bfield_avg_fp, Direction{0}, lev, amrex::convert(ba, Bx_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Bfield_avg_fp, Direction{1}, lev, amrex::convert(ba, By_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Bfield_avg_fp, Direction{2}, lev, amrex::convert(ba, Bz_nodal_flag), dm, ncomps, ngEB, 0.0_rt);

        m_fields.alloc_init(FieldType::Efield_avg_fp, Direction{0}, lev, amrex::convert(ba, Ex_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Efield_avg_fp, Direction{1}, lev, amrex::convert(ba, Ey_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Efield_avg_fp, Direction{2}, lev, amrex::convert(ba, Ez_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
    }

    if (EB::enabled()) {
        constexpr int nc_ls = 1;
        amrex::IntVect const ng_ls(2);
        //EB level set
        m_fields.alloc_init(FieldType::distance_to_eb, lev, amrex::convert(ba, IntVect::TheNodeVector()), dm, nc_ls, ng_ls, 0.0_rt);
        // Whether to reduce the particle shape to order 1 when close to the EB
        AllocInitMultiFab(m_eb_reduce_particle_shape[lev], amrex::convert(ba, IntVect::TheCellVector()), dm, ncomps,
            ngRho, lev, "m_eb_reduce_particle_shape");

        // EB info are needed only at the finest level
        if (lev == maxLevel()) {

            if (WarpX::electromagnetic_solver_id != ElectromagneticSolverAlgo::PSATD) {

                AllocInitMultiFab(m_eb_update_E[lev][0], amrex::convert(ba, Ex_nodal_flag), dm, ncomps,
                                  guard_cells.ng_FieldSolver, lev, "m_eb_update_E[x]");
                AllocInitMultiFab(m_eb_update_E[lev][1], amrex::convert(ba, Ey_nodal_flag), dm, ncomps,
                                  guard_cells.ng_FieldSolver, lev, "m_eb_update_E[y]");
                AllocInitMultiFab(m_eb_update_E[lev][2], amrex::convert(ba, Ez_nodal_flag), dm, ncomps,
                                  guard_cells.ng_FieldSolver, lev, "m_eb_update_E[z]");

                AllocInitMultiFab(m_eb_update_B[lev][0], amrex::convert(ba, Bx_nodal_flag), dm, ncomps,
                                  guard_cells.ng_FieldSolver, lev, "m_eb_update_B[x]");
                AllocInitMultiFab(m_eb_update_B[lev][1], amrex::convert(ba, By_nodal_flag), dm, ncomps,
                                  guard_cells.ng_FieldSolver, lev, "m_eb_update_B[y]");
                AllocInitMultiFab(m_eb_update_B[lev][2], amrex::convert(ba, Bz_nodal_flag), dm, ncomps,
                                  guard_cells.ng_FieldSolver, lev, "m_eb_update_B[z]");
            }
            if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::ECT) {

                //! EB: Lengths of the mesh edges
                m_fields.alloc_init(FieldType::edge_lengths, Direction{0}, lev, amrex::convert(ba, Ex_nodal_flag),
                    dm, ncomps, guard_cells.ng_FieldSolver, 0.0_rt);
                m_fields.alloc_init(FieldType::edge_lengths, Direction{1}, lev, amrex::convert(ba, Ey_nodal_flag),
                    dm, ncomps, guard_cells.ng_FieldSolver, 0.0_rt);
                m_fields.alloc_init(FieldType::edge_lengths, Direction{2}, lev, amrex::convert(ba, Ez_nodal_flag),
                    dm, ncomps, guard_cells.ng_FieldSolver, 0.0_rt);

                //! EB: Areas of the mesh faces
                m_fields.alloc_init(FieldType::face_areas, Direction{0}, lev, amrex::convert(ba, Bx_nodal_flag),
                    dm, ncomps, guard_cells.ng_FieldSolver, 0.0_rt);
                m_fields.alloc_init(FieldType::face_areas, Direction{1}, lev, amrex::convert(ba, By_nodal_flag),
                    dm, ncomps, guard_cells.ng_FieldSolver, 0.0_rt);
                m_fields.alloc_init(FieldType::face_areas, Direction{2}, lev, amrex::convert(ba, Bz_nodal_flag),
                    dm, ncomps, guard_cells.ng_FieldSolver, 0.0_rt);

                AllocInitMultiFab(m_flag_info_face[lev][0], amrex::convert(ba, Bx_nodal_flag), dm, ncomps,
                                  guard_cells.ng_FieldSolver, lev, "m_flag_info_face[x]");
                AllocInitMultiFab(m_flag_info_face[lev][1], amrex::convert(ba, By_nodal_flag), dm, ncomps,
                                  guard_cells.ng_FieldSolver, lev, "m_flag_info_face[y]");
                AllocInitMultiFab(m_flag_info_face[lev][2], amrex::convert(ba, Bz_nodal_flag), dm, ncomps,
                                  guard_cells.ng_FieldSolver, lev, "m_flag_info_face[z]");
                AllocInitMultiFab(m_flag_ext_face[lev][0], amrex::convert(ba, Bx_nodal_flag), dm, ncomps,
                                  guard_cells.ng_FieldSolver, lev, "m_flag_ext_face[x]");
                AllocInitMultiFab(m_flag_ext_face[lev][1], amrex::convert(ba, By_nodal_flag), dm, ncomps,
                                  guard_cells.ng_FieldSolver, lev, "m_flag_ext_face[y]");
                AllocInitMultiFab(m_flag_ext_face[lev][2], amrex::convert(ba, Bz_nodal_flag), dm, ncomps,
                                  guard_cells.ng_FieldSolver, lev, "m_flag_ext_face[z]");

                /** EB: area_mod contains the modified areas of the mesh faces, i.e. if a face is enlarged it
                * contains the area of the enlarged face
                * This is only used for the ECT solver.*/
                m_fields.alloc_init(FieldType::area_mod, Direction{0}, lev, amrex::convert(ba, Bx_nodal_flag),
                    dm, ncomps, guard_cells.ng_FieldSolver, 0.0_rt);
                m_fields.alloc_init(FieldType::area_mod, Direction{1}, lev, amrex::convert(ba, By_nodal_flag),
                    dm, ncomps, guard_cells.ng_FieldSolver, 0.0_rt);
                m_fields.alloc_init(FieldType::area_mod, Direction{2}, lev, amrex::convert(ba, Bz_nodal_flag),
                    dm, ncomps, guard_cells.ng_FieldSolver, 0.0_rt);

                m_borrowing[lev][0] = std::make_unique<amrex::LayoutData<FaceInfoBox>>(
                        amrex::convert(ba, Bx_nodal_flag), dm);
                m_borrowing[lev][1] = std::make_unique<amrex::LayoutData<FaceInfoBox>>(
                        amrex::convert(ba, By_nodal_flag), dm);
                m_borrowing[lev][2] = std::make_unique<amrex::LayoutData<FaceInfoBox>>(
                        amrex::convert(ba, Bz_nodal_flag), dm);

                /** Venl contains the electromotive force for every mesh face, i.e. every entry is
                * the corresponding entry in ECTRhofield multiplied by the total area (possibly with enlargement)
                * This is only used for the ECT solver.*/
                m_fields.alloc_init(FieldType::Venl, Direction{0}, lev, amrex::convert(ba, Bx_nodal_flag),
                    dm, ncomps, guard_cells.ng_FieldSolver, 0.0_rt);
                m_fields.alloc_init(FieldType::Venl, Direction{1}, lev, amrex::convert(ba, By_nodal_flag),
                    dm, ncomps, guard_cells.ng_FieldSolver, 0.0_rt);
                m_fields.alloc_init(FieldType::Venl, Direction{2}, lev, amrex::convert(ba, Bz_nodal_flag),
                    dm, ncomps, guard_cells.ng_FieldSolver, 0.0_rt);

                /** ECTRhofield is needed only by the ect
                * solver and it contains the electromotive force density for every mesh face.
                * The name ECTRhofield has been used to comply with the notation of the paper
                * https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4463918 (page 9, equation 4
                * and below).
                * Although it's called rho it has nothing to do with the charge density!
                * This is only used for the ECT solver.*/
                m_fields.alloc_init(FieldType::ECTRhofield, Direction{0}, lev, amrex::convert(ba, Bx_nodal_flag),
                    dm, ncomps, guard_cells.ng_FieldSolver, 0.0_rt);
                m_fields.alloc_init(FieldType::ECTRhofield, Direction{1}, lev, amrex::convert(ba, By_nodal_flag),
                    dm, ncomps, guard_cells.ng_FieldSolver, 0.0_rt);
                m_fields.alloc_init(FieldType::ECTRhofield, Direction{2}, lev, amrex::convert(ba, Bz_nodal_flag),
                    dm, ncomps, guard_cells.ng_FieldSolver, 0.0_rt);
            }
        }
    }

    int rho_ncomps = 0;
    if( (electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrame) ||
        (electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameElectroMagnetostatic) ||
        (electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameEffectivePotential) ||
        (electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC) ) {
        rho_ncomps = ncomps;
    }
    if (do_dive_cleaning) {
        rho_ncomps = 2*ncomps;
    }
    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
        if (do_dive_cleaning || update_with_rho || current_correction) {
            // For the multi-J algorithm we can allocate only one rho component (no distinction between old and new)
            rho_ncomps = (WarpX::do_multi_J) ? ncomps : 2*ncomps;
        }
    }
    if (rho_ncomps > 0)
    {
        m_fields.alloc_init(FieldType::rho_fp,
            lev, amrex::convert(ba, rho_nodal_flag), dm,
            rho_ncomps, ngRho, 0.0_rt);
    }

    if (electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrame ||
        electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameElectroMagnetostatic ||
        electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameEffectivePotential )
    {
        const IntVect ngPhi = IntVect( AMREX_D_DECL(1,1,1) );
        m_fields.alloc_init(FieldType::phi_fp, lev, amrex::convert(ba, phi_nodal_flag), dm,
                             ncomps, ngPhi, 0.0_rt );
    }

    if (m_do_subcycling && lev == 0)
    {
        m_fields.alloc_init(FieldType::current_store, Direction{0}, lev, amrex::convert(ba,jx_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
        m_fields.alloc_init(FieldType::current_store, Direction{1}, lev, amrex::convert(ba,jy_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
        m_fields.alloc_init(FieldType::current_store, Direction{2}, lev, amrex::convert(ba,jz_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
    }

    if (do_dive_cleaning)
    {
        m_fields.alloc_init(FieldType::F_fp,
            lev, amrex::convert(ba, F_nodal_flag), dm,
            ncomps, ngF, 0.0_rt);
    }

    if (do_divb_cleaning)
    {
        m_fields.alloc_init(FieldType::G_fp,
            lev, amrex::convert(ba, G_nodal_flag), dm,
            ncomps, ngG, 0.0_rt);
    }

    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD)
    {
        // Allocate and initialize the spectral solver
#ifndef WARPX_USE_FFT
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( false,
            "WarpX::AllocLevelMFs: PSATD solver requires WarpX build with spectral solver support.");
#else

        // Check whether the option periodic, single box is valid here
        if (fft_periodic_single_box) {
#   ifdef WARPX_DIM_RZ
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                geom[0].isPeriodic(1)          // domain is periodic in z
                && ba.size() == 1 && lev == 0, // domain is decomposed in a single box
                "The option `psatd.periodic_single_box_fft` can only be used for a periodic domain, decomposed in a single box");
#   else
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                geom[0].isAllPeriodic()        // domain is periodic in all directions
                && ba.size() == 1 && lev == 0, // domain is decomposed in a single box
                "The option `psatd.periodic_single_box_fft` can only be used for a periodic domain, decomposed in a single box");
#   endif
        }
        // Get the cell-centered box
        BoxArray realspace_ba = ba;  // Copy box
        realspace_ba.enclosedCells(); // Make it cell-centered
        // Define spectral solver
#   ifdef WARPX_DIM_RZ
        if ( !fft_periodic_single_box ) {
            realspace_ba.grow(1, ngEB[1]); // add guard cells only in z
        }
        if (field_boundary_hi[0] == FieldBoundaryType::PML && !do_pml_in_domain) {
            // Extend region that is solved for to include the guard cells
            // which is where the PML boundary is applied.
            realspace_ba.growHi(0, pml_ncell);
        }
        AllocLevelSpectralSolverRZ(spectral_solver_fp,
                                   lev,
                                   realspace_ba,
                                   dm,
                                   dx);
#   else
        if ( !fft_periodic_single_box ) {
            realspace_ba.grow(ngEB);   // add guard cells
        }
        bool const pml_flag_false = false;
        AllocLevelSpectralSolver(spectral_solver_fp,
                                 lev,
                                 realspace_ba,
                                 dm,
                                 dx,
                                 pml_flag_false);
#   endif
#endif
    } // ElectromagneticSolverAlgo::PSATD
    else {
        m_fdtd_solver_fp[lev] = std::make_unique<FiniteDifferenceSolver>(electromagnetic_solver_id, dx, grid_type);
    }

    //
    // The Aux patch (i.e., the full solution)
    //
    if (aux_is_nodal and grid_type != GridType::Collocated)
    {
        // Create aux multifabs on Nodal Box Array
        BoxArray const nba = amrex::convert(ba,IntVect::TheNodeVector());

        m_fields.alloc_init(FieldType::Bfield_aux, Direction{0}, lev, nba, dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Bfield_aux, Direction{1}, lev, nba, dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Bfield_aux, Direction{2}, lev, nba, dm, ncomps, ngEB, 0.0_rt);

        m_fields.alloc_init(FieldType::Efield_aux, Direction{0}, lev, nba, dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Efield_aux, Direction{1}, lev, nba, dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Efield_aux, Direction{2}, lev, nba, dm, ncomps, ngEB, 0.0_rt);
    } else if (lev == 0) {
        if (WarpX::fft_do_time_averaging) {
            m_fields.alias_init(FieldType::Bfield_aux, FieldType::Bfield_avg_fp, Direction{0}, lev, 0.0_rt);
            m_fields.alias_init(FieldType::Bfield_aux, FieldType::Bfield_avg_fp, Direction{1}, lev, 0.0_rt);
            m_fields.alias_init(FieldType::Bfield_aux, FieldType::Bfield_avg_fp, Direction{2}, lev, 0.0_rt);

            m_fields.alias_init(FieldType::Efield_aux, FieldType::Efield_avg_fp, Direction{0}, lev, 0.0_rt);
            m_fields.alias_init(FieldType::Efield_aux, FieldType::Efield_avg_fp, Direction{1}, lev, 0.0_rt);
            m_fields.alias_init(FieldType::Efield_aux, FieldType::Efield_avg_fp, Direction{2}, lev, 0.0_rt);
        } else {
            if (mypc->m_B_ext_particle_s == "read_from_file") {
                m_fields.alloc_init(FieldType::Bfield_aux, Direction{0}, lev, amrex::convert(ba, Bx_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
                m_fields.alloc_init(FieldType::Bfield_aux, Direction{1}, lev, amrex::convert(ba, By_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
                m_fields.alloc_init(FieldType::Bfield_aux, Direction{2}, lev, amrex::convert(ba, Bz_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
            } else {
                // In this case, the aux grid is simply an alias of the fp grid (most common case in WarpX)
                m_fields.alias_init(FieldType::Bfield_aux, FieldType::Bfield_fp, Direction{0}, lev, 0.0_rt);
                m_fields.alias_init(FieldType::Bfield_aux, FieldType::Bfield_fp, Direction{1}, lev, 0.0_rt);
                m_fields.alias_init(FieldType::Bfield_aux, FieldType::Bfield_fp, Direction{2}, lev, 0.0_rt);
            }
            if (mypc->m_E_ext_particle_s == "read_from_file") {
                m_fields.alloc_init(FieldType::Efield_aux, Direction{0}, lev, amrex::convert(ba, Ex_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
                m_fields.alloc_init(FieldType::Efield_aux, Direction{1}, lev, amrex::convert(ba, Ey_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
                m_fields.alloc_init(FieldType::Efield_aux, Direction{2}, lev, amrex::convert(ba, Ez_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
            } else {
                // In this case, the aux grid is simply an alias of the fp grid (most common case in WarpX)
                m_fields.alias_init(FieldType::Efield_aux, FieldType::Efield_fp, Direction{0}, lev, 0.0_rt);
                m_fields.alias_init(FieldType::Efield_aux, FieldType::Efield_fp, Direction{1}, lev, 0.0_rt);
                m_fields.alias_init(FieldType::Efield_aux, FieldType::Efield_fp, Direction{2}, lev, 0.0_rt);
            }
        }
    } else {
        m_fields.alloc_init(FieldType::Bfield_aux, Direction{0}, lev, amrex::convert(ba, Bx_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Bfield_aux, Direction{1}, lev, amrex::convert(ba, By_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Bfield_aux, Direction{2}, lev, amrex::convert(ba, Bz_nodal_flag), dm, ncomps, ngEB, 0.0_rt);

        m_fields.alloc_init(FieldType::Efield_aux, Direction{0}, lev, amrex::convert(ba, Ex_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Efield_aux, Direction{1}, lev, amrex::convert(ba, Ey_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Efield_aux, Direction{2}, lev, amrex::convert(ba, Ez_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
    }

    // The external fields that are read from file
    if (m_p_ext_field_params->B_ext_grid_type != ExternalFieldType::default_zero && m_p_ext_field_params->B_ext_grid_type != ExternalFieldType::constant) {
        // These fields will be added directly to the grid, i.e. to fp, and need to match the index type
        m_fields.alloc_init(FieldType::Bfield_fp_external, Direction{0}, lev,
            amrex::convert(ba, m_fields.get(FieldType::Bfield_fp,Direction{0},lev)->ixType()),
            dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Bfield_fp_external, Direction{1}, lev,
            amrex::convert(ba, m_fields.get(FieldType::Bfield_fp,Direction{1},lev)->ixType()),
            dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Bfield_fp_external, Direction{2}, lev,
            amrex::convert(ba, m_fields.get(FieldType::Bfield_fp,Direction{2},lev)->ixType()),
            dm, ncomps, ngEB, 0.0_rt);
    }
    if (mypc->m_B_ext_particle_s == "read_from_file") {
        //  These fields will be added to the fields that the particles see, and need to match the index type
        auto *Bfield_aux_levl_0 = m_fields.get(FieldType::Bfield_aux, Direction{0}, lev);
        auto *Bfield_aux_levl_1 = m_fields.get(FieldType::Bfield_aux, Direction{1}, lev);
        auto *Bfield_aux_levl_2 = m_fields.get(FieldType::Bfield_aux, Direction{2}, lev);

        // Same as Bfield_fp for reading external field data
        m_fields.alloc_init(FieldType::B_external_particle_field, Direction{0}, lev, amrex::convert(ba, Bfield_aux_levl_0->ixType()),
            dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::B_external_particle_field, Direction{1}, lev, amrex::convert(ba, Bfield_aux_levl_1->ixType()),
            dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::B_external_particle_field, Direction{2}, lev, amrex::convert(ba, Bfield_aux_levl_2->ixType()),
            dm, ncomps, ngEB, 0.0_rt);
    }
    if (m_p_ext_field_params->E_ext_grid_type != ExternalFieldType::default_zero && m_p_ext_field_params->E_ext_grid_type != ExternalFieldType::constant) {
        // These fields will be added directly to the grid, i.e. to fp, and need to match the index type
        m_fields.alloc_init(FieldType::Efield_fp_external, Direction{0}, lev, amrex::convert(ba, m_fields.get(FieldType::Efield_fp, Direction{0}, lev)->ixType()),
            dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Efield_fp_external, Direction{1}, lev, amrex::convert(ba, m_fields.get(FieldType::Efield_fp, Direction{1}, lev)->ixType()),
            dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Efield_fp_external, Direction{2}, lev, amrex::convert(ba, m_fields.get(FieldType::Efield_fp, Direction{2}, lev)->ixType()),
            dm, ncomps, ngEB, 0.0_rt);
    }
    if (mypc->m_E_ext_particle_s == "read_from_file") {
        //  These fields will be added to the fields that the particles see, and need to match the index type
        auto *Efield_aux_levl_0 = m_fields.get(FieldType::Efield_aux, Direction{0}, lev);
        auto *Efield_aux_levl_1 = m_fields.get(FieldType::Efield_aux, Direction{1}, lev);
        auto *Efield_aux_levl_2 = m_fields.get(FieldType::Efield_aux, Direction{2}, lev);

        // Same as Efield_fp for reading external field data
        m_fields.alloc_init(FieldType::E_external_particle_field, Direction{0}, lev, amrex::convert(ba, Efield_aux_levl_0->ixType()),
            dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::E_external_particle_field, Direction{1}, lev, amrex::convert(ba, Efield_aux_levl_1->ixType()),
            dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::E_external_particle_field, Direction{2}, lev, amrex::convert(ba, Efield_aux_levl_2->ixType()),
            dm, ncomps, ngEB, 0.0_rt);
    }

    //
    // The coarse patch
    //
    if (lev > 0)
    {
        BoxArray cba = ba;
        cba.coarsen(refRatio(lev-1));
        const std::array<Real,3> cdx = CellSize(lev-1);

        // Create the MultiFabs for B
        m_fields.alloc_init(FieldType::Bfield_cp, Direction{0}, lev, amrex::convert(cba, Bx_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Bfield_cp, Direction{1}, lev, amrex::convert(cba, By_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Bfield_cp, Direction{2}, lev, amrex::convert(cba, Bz_nodal_flag), dm, ncomps, ngEB, 0.0_rt);

        // Create the MultiFabs for E
        m_fields.alloc_init(FieldType::Efield_cp, Direction{0}, lev, amrex::convert(cba, Ex_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Efield_cp, Direction{1}, lev, amrex::convert(cba, Ey_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        m_fields.alloc_init(FieldType::Efield_cp, Direction{2}, lev, amrex::convert(cba, Ez_nodal_flag), dm, ncomps, ngEB, 0.0_rt);

        if (fft_do_time_averaging)
        {
            m_fields.alloc_init(FieldType::Bfield_avg_cp, Direction{0}, lev, amrex::convert(cba, Bx_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
            m_fields.alloc_init(FieldType::Bfield_avg_cp, Direction{1}, lev, amrex::convert(cba, By_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
            m_fields.alloc_init(FieldType::Bfield_avg_cp, Direction{2}, lev, amrex::convert(cba, Bz_nodal_flag), dm, ncomps, ngEB, 0.0_rt);

            m_fields.alloc_init(FieldType::Efield_avg_cp, Direction{0}, lev, amrex::convert(cba, Ex_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
            m_fields.alloc_init(FieldType::Efield_avg_cp, Direction{1}, lev, amrex::convert(cba, Ey_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
            m_fields.alloc_init(FieldType::Efield_avg_cp, Direction{2}, lev, amrex::convert(cba, Ez_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        }

        // Create the MultiFabs for the current
        m_fields.alloc_init(FieldType::current_cp, Direction{0}, lev, amrex::convert(cba, jx_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
        m_fields.alloc_init(FieldType::current_cp, Direction{1}, lev, amrex::convert(cba, jy_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
        m_fields.alloc_init(FieldType::current_cp, Direction{2}, lev, amrex::convert(cba, jz_nodal_flag), dm, ncomps, ngJ, 0.0_rt);

        if (rho_ncomps > 0) {
            m_fields.alloc_init(FieldType::rho_cp,
                lev, amrex::convert(cba, rho_nodal_flag), dm,
                rho_ncomps, ngRho, 0.0_rt);
        }

        if (do_dive_cleaning)
        {
            m_fields.alloc_init(FieldType::F_cp,
                lev, amrex::convert(cba, IntVect::TheUnitVector()), dm,
                ncomps, ngF, 0.0_rt);
        }

        if (do_divb_cleaning)
        {
            if (grid_type == GridType::Collocated)
            {
                m_fields.alloc_init(FieldType::G_cp,
                    lev, amrex::convert(cba, IntVect::TheUnitVector()), dm,
                    ncomps, ngG, 0.0_rt);
            }
            else // grid_type=staggered or grid_type=hybrid
            {
                m_fields.alloc_init(FieldType::G_cp,
                    lev, amrex::convert(cba, IntVect::TheZeroVector()), dm,
                    ncomps, ngG, 0.0_rt);
            }
        }

        if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD)
        {
            // Allocate and initialize the spectral solver
#ifndef WARPX_USE_FFT
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( false,
                "WarpX::AllocLevelMFs: PSATD solver requires WarpX build with spectral solver support.");
#else

            // Get the cell-centered box, with guard cells
            BoxArray c_realspace_ba = cba;// Copy box
            c_realspace_ba.enclosedCells(); // Make it cell-centered
            // Define spectral solver
#ifdef WARPX_DIM_RZ
            c_realspace_ba.grow(1, ngEB[1]); // add guard cells only in z
            if (field_boundary_hi[0] == FieldBoundaryType::PML && !do_pml_in_domain) {
                // Extend region that is solved for to include the guard cells
                // which is where the PML boundary is applied.
                c_realspace_ba.growHi(0, pml_ncell);
            }
            AllocLevelSpectralSolverRZ(spectral_solver_cp,
                                       lev,
                                       c_realspace_ba,
                                       dm,
                                       cdx);
#   else
            c_realspace_ba.grow(ngEB);
            bool const pml_flag_false = false;
            AllocLevelSpectralSolver(spectral_solver_cp,
                                     lev,
                                     c_realspace_ba,
                                     dm,
                                     cdx,
                                     pml_flag_false);
#   endif
#endif
        } // ElectromagneticSolverAlgo::PSATD
        else {
            m_fdtd_solver_cp[lev] = std::make_unique<FiniteDifferenceSolver>(electromagnetic_solver_id, cdx,
                                                                             grid_type);
        }
    }

    //
    // Copy of the coarse aux
    //
    if (lev > 0 && (n_field_gather_buffer > 0 || n_current_deposition_buffer > 0 ||
                    mypc->nSpeciesGatherFromMainGrid() > 0))
    {
        BoxArray cba = ba;
        cba.coarsen(refRatio(lev-1));

        if (n_field_gather_buffer > 0 || mypc->nSpeciesGatherFromMainGrid() > 0) {
            if (aux_is_nodal) {
                BoxArray const& cnba = amrex::convert(cba,IntVect::TheNodeVector());
                // Create the MultiFabs for B
                m_fields.alloc_init(FieldType::Bfield_cax, Direction{0}, lev, cnba, dm, ncomps, ngEB, 0.0_rt);
                m_fields.alloc_init(FieldType::Bfield_cax, Direction{1}, lev, cnba, dm, ncomps, ngEB, 0.0_rt);
                m_fields.alloc_init(FieldType::Bfield_cax, Direction{2}, lev, cnba, dm, ncomps, ngEB, 0.0_rt);

                // Create the MultiFabs for E
                m_fields.alloc_init(FieldType::Efield_cax, Direction{0}, lev, cnba, dm, ncomps, ngEB, 0.0_rt);
                m_fields.alloc_init(FieldType::Efield_cax, Direction{1}, lev, cnba, dm, ncomps, ngEB, 0.0_rt);
                m_fields.alloc_init(FieldType::Efield_cax, Direction{2}, lev, cnba, dm, ncomps, ngEB, 0.0_rt);
            } else {
                // Create the MultiFabs for B
                m_fields.alloc_init(FieldType::Bfield_cax, Direction{0}, lev, amrex::convert(cba, Bx_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
                m_fields.alloc_init(FieldType::Bfield_cax, Direction{1}, lev, amrex::convert(cba, By_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
                m_fields.alloc_init(FieldType::Bfield_cax, Direction{2}, lev, amrex::convert(cba, Bz_nodal_flag), dm, ncomps, ngEB, 0.0_rt);

                // Create the MultiFabs for E
                m_fields.alloc_init(FieldType::Efield_cax, Direction{0}, lev, amrex::convert(cba,Ex_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
                m_fields.alloc_init(FieldType::Efield_cax, Direction{1}, lev, amrex::convert(cba,Ey_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
                m_fields.alloc_init(FieldType::Efield_cax, Direction{2}, lev, amrex::convert(cba,Ez_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
            }

            AllocInitMultiFab(gather_buffer_masks[lev], ba, dm, ncomps, amrex::IntVect(1), lev, "gather_buffer_masks");
            // Gather buffer masks have 1 ghost cell, because of the fact
            // that particles may move by more than one cell when using subcycling.
        }

        if (n_current_deposition_buffer > 0) {
            m_fields.alloc_init(FieldType::current_buf, Direction{0}, lev, amrex::convert(cba,jx_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
            m_fields.alloc_init(FieldType::current_buf, Direction{1}, lev, amrex::convert(cba,jy_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
            m_fields.alloc_init(FieldType::current_buf, Direction{2}, lev, amrex::convert(cba,jz_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
            if (m_fields.has(FieldType::rho_cp, lev)) {
                m_fields.alloc_init(FieldType::rho_buf, lev, amrex::convert(cba,rho_nodal_flag), dm, 2*ncomps, ngRho, 0.0_rt);
            }
            AllocInitMultiFab(current_buffer_masks[lev], ba, dm, ncomps, amrex::IntVect(1), lev, "current_buffer_masks");
            // Current buffer masks have 1 ghost cell, because of the fact
            // that particles may move by more than one cell when using subcycling.
        }
    }

    if (load_balance_intervals.isActivated())
    {
        costs[lev] = std::make_unique<LayoutData<Real>>(ba, dm);
        load_balance_efficiency[lev] = -1;
    }
}

#ifdef WARPX_USE_FFT
#   ifdef WARPX_DIM_RZ
/* \brief Allocate spectral Maxwell solver (RZ dimensions) at a level
 *
 * \param[in, out] spectral_solver Vector of pointer to SpectralSolver, to point to allocated spectral Maxwell
 *                                 solver at a given level
 * \param[in] lev                  Level at which to allocate spectral Maxwell solver
 * \param[in] realspace_ba         Box array that corresponds to the decomposition of the fields in real space
 *                                 (cell-centered; includes guard cells)
 * \param[in] dm                   Indicates which MPI proc owns which box, in realspace_ba
 * \param[in] dx                   Cell size along each dimension
 */
void WarpX::AllocLevelSpectralSolverRZ (amrex::Vector<std::unique_ptr<SpectralSolverRZ>>& spectral_solver,
                                        const int lev,
                                        const amrex::BoxArray& realspace_ba,
                                        const amrex::DistributionMapping& dm,
                                        const std::array<Real,3>& dx)
{
    const RealVect dx_vect(dx[0], dx[2]);

    amrex::Real solver_dt = dt[lev];
    if (WarpX::do_multi_J) { solver_dt /= static_cast<amrex::Real>(WarpX::do_multi_J_n_depositions); }
    if (evolve_scheme == EvolveScheme::StrangImplicitSpectralEM) {
        // The step is Strang split into two half steps
        solver_dt /= 2.;
    }

    auto pss = std::make_unique<SpectralSolverRZ>(lev,
                                                  realspace_ba,
                                                  dm,
                                                  n_rz_azimuthal_modes,
                                                  noz_fft,
                                                  grid_type,
                                                  m_v_galilean,
                                                  dx_vect,
                                                  solver_dt,
                                                  ::isAnyBoundaryPML(field_boundary_lo, field_boundary_hi),
                                                  update_with_rho,
                                                  fft_do_time_averaging,
                                                  J_in_time,
                                                  rho_in_time,
                                                  do_dive_cleaning,
                                                  do_divb_cleaning);
    spectral_solver[lev] = std::move(pss);

    if (use_kspace_filter) {
        spectral_solver[lev]->InitFilter(filter_npass_each_dir,
                                         use_filter_compensation);
    }
}
#   else
/* \brief Allocate spectral Maxwell solver at a level
 *
 * \param[in, out] spectral_solver  Vector of pointer to SpectralSolver, to point to allocated spectral Maxwell
 *                                  solver at a given level
 * \param[in] lev                   Level at which to allocate spectral Maxwell solver
 * \param[in] realspace_ba          Box array that corresponds to the decomposition of the fields in real space
 *                                  (cell-centered; includes guard cells)
 * \param[in] dm                    Indicates which MPI proc owns which box, in realspace_ba
 * \param[in] dx                    Cell size along each dimension
 * \param[in] pml_flag              Whether the boxes in which the solver is applied are PML boxes
 */
void WarpX::AllocLevelSpectralSolver (amrex::Vector<std::unique_ptr<SpectralSolver>>& spectral_solver,
                                      const int lev,
                                      const amrex::BoxArray& realspace_ba,
                                      const amrex::DistributionMapping& dm,
                                      const std::array<Real,3>& dx,
                                      const bool pml_flag)
{
#if defined(WARPX_DIM_3D)
    const RealVect dx_vect(dx[0], dx[1], dx[2]);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    const RealVect dx_vect(dx[0], dx[2]);
#elif defined(WARPX_DIM_1D_Z)
    const RealVect dx_vect(dx[2]);
#endif

    amrex::Real solver_dt = dt[lev];
    if (WarpX::do_multi_J) { solver_dt /= static_cast<amrex::Real>(WarpX::do_multi_J_n_depositions); }
    if (evolve_scheme == EvolveScheme::StrangImplicitSpectralEM) {
        // The step is Strang split into two half steps
        solver_dt /= 2.;
    }

    auto pss = std::make_unique<SpectralSolver>(lev,
                                                realspace_ba,
                                                dm,
                                                nox_fft,
                                                noy_fft,
                                                noz_fft,
                                                grid_type,
                                                m_v_galilean,
                                                m_v_comoving,
                                                dx_vect,
                                                solver_dt,
                                                pml_flag,
                                                fft_periodic_single_box,
                                                update_with_rho,
                                                fft_do_time_averaging,
                                                m_psatd_solution_type,
                                                J_in_time,
                                                rho_in_time,
                                                do_dive_cleaning,
                                                do_divb_cleaning);
    spectral_solver[lev] = std::move(pss);
}
#   endif
#endif

std::array<Real,3>
WarpX::CellSize (int lev)
{
    const amrex::Geometry& gm = GetInstance().Geom(lev);
    const Real* dx = gm.CellSize();
#if defined(WARPX_DIM_3D)
    return { dx[0], dx[1], dx[2] };
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    return { dx[0], 1.0, dx[1] };
#else
    return { 1.0, 1.0, dx[0] };
#endif
}

amrex::XDim3
WarpX::InvCellSize (int lev)
{
    std::array<Real,3> dx = WarpX::CellSize(lev);
    return {1._rt/dx[0], 1._rt/dx[1], 1._rt/dx[2]};
}

amrex::RealBox
WarpX::getRealBox(const Box& bx, int lev)
{
    const amrex::Geometry& gm = GetInstance().Geom(lev);
    const RealBox grid_box{bx, gm.CellSize(), gm.ProbLo()};
    return( grid_box );
}

amrex::XDim3
WarpX::LowerCorner(const Box& bx, const int lev, const amrex::Real time_shift_delta)
{
    auto & warpx = GetInstance();
    const RealBox grid_box = getRealBox( bx, lev );

    const Real* grid_min = grid_box.lo();

    const amrex::Real cur_time = warpx.gett_new(lev);
    const amrex::Real time_shift = (cur_time + time_shift_delta - warpx.time_of_last_gal_shift);
    amrex::Array<amrex::Real,3> galilean_shift = { warpx.m_v_galilean[0]*time_shift,
                                                   warpx.m_v_galilean[1]*time_shift,
                                                   warpx.m_v_galilean[2]*time_shift };

#if defined(WARPX_DIM_3D)
    return { grid_min[0] + galilean_shift[0], grid_min[1] + galilean_shift[1], grid_min[2] + galilean_shift[2] };

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    return { grid_min[0] + galilean_shift[0], std::numeric_limits<Real>::lowest(), grid_min[1] + galilean_shift[2] };

#elif defined(WARPX_DIM_1D_Z)
    return { std::numeric_limits<Real>::lowest(), std::numeric_limits<Real>::lowest(), grid_min[0] + galilean_shift[2] };
#endif
}

amrex::XDim3
WarpX::UpperCorner(const Box& bx, const int lev, const amrex::Real time_shift_delta)
{
    auto & warpx = GetInstance();
    const RealBox grid_box = getRealBox( bx, lev );

    const Real* grid_max = grid_box.hi();

    const amrex::Real cur_time = warpx.gett_new(lev);
    const amrex::Real time_shift = (cur_time + time_shift_delta - warpx.time_of_last_gal_shift);
    amrex::Array<amrex::Real,3> galilean_shift = { warpx.m_v_galilean[0]*time_shift,
                                                   warpx.m_v_galilean[1]*time_shift,
                                                   warpx.m_v_galilean[2]*time_shift };

#if defined(WARPX_DIM_3D)
    return { grid_max[0] + galilean_shift[0], grid_max[1] + galilean_shift[1], grid_max[2] + galilean_shift[2] };

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    return { grid_max[0] + galilean_shift[0], std::numeric_limits<Real>::max(), grid_max[1] + galilean_shift[1] };

#elif defined(WARPX_DIM_1D_Z)
    return { std::numeric_limits<Real>::max(), std::numeric_limits<Real>::max(), grid_max[0] + galilean_shift[0] };
#endif
}

IntVect
WarpX::RefRatio (int lev)
{
    return GetInstance().refRatio(lev);
}

void
WarpX::ComputeDivB (amrex::MultiFab& divB, int const dcomp,
                    ablastr::fields::VectorField const& B,
                    const std::array<amrex::Real,3>& dx)
{
    ComputeDivB(divB, dcomp, B, dx, IntVect::TheZeroVector());
}

void
WarpX::ComputeDivB (amrex::MultiFab& divB, int const dcomp,
                    ablastr::fields::VectorField const& B,
                    const std::array<amrex::Real,3>& dx, IntVect const ngrow)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(grid_type != GridType::Collocated,
        "ComputeDivB not implemented with warpx.grid_type=collocated.");

    const Real dxinv = 1._rt/dx[0], dyinv = 1._rt/dx[1], dzinv = 1._rt/dx[2];

#ifdef WARPX_DIM_RZ
    const Real rmin = GetInstance().Geom(0).ProbLo(0);
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(divB, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box bx = mfi.growntilebox(ngrow);
        amrex::Array4<const amrex::Real> const& Bxfab = B[0]->array(mfi);
        amrex::Array4<const amrex::Real> const& Byfab = B[1]->array(mfi);
        amrex::Array4<const amrex::Real> const& Bzfab = B[2]->array(mfi);
        amrex::Array4<amrex::Real> const& divBfab = divB.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            warpx_computedivb(i, j, k, dcomp, divBfab, Bxfab, Byfab, Bzfab, dxinv, dyinv, dzinv
#ifdef WARPX_DIM_RZ
                              ,rmin
#endif
                              );
        });
    }
}

void
WarpX::ComputeDivE(amrex::MultiFab& divE, const int lev)
{
    if ( WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD ) {
#ifdef WARPX_USE_FFT
        const ablastr::fields::VectorField Efield_aux_lev = m_fields.get_alldirs(FieldType::Efield_aux, lev);
        spectral_solver_fp[lev]->ComputeSpectralDivE(lev, Efield_aux_lev, divE);
#else
        WARPX_ABORT_WITH_MESSAGE(
            "ComputeDivE: PSATD requested but not compiled");
#endif
    } else {
        const ablastr::fields::VectorField Efield_aux_lev = m_fields.get_alldirs(FieldType::Efield_aux, lev);
        m_fdtd_solver_fp[lev]->ComputeDivE(Efield_aux_lev, divE);
    }
}

#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
PML_RZ*
WarpX::GetPML_RZ (int lev)
{
    if (pml_rz[lev]) {
        // This should check if pml was initialized
        return pml_rz[lev].get();
    } else {
        return nullptr;
    }
}
#endif

PML*
WarpX::GetPML (int lev)
{
    if (do_pml) {
        // This should check if pml was initialized
        return pml[lev].get();
    } else {
        return nullptr;
    }
}

std::vector< bool >
WarpX::getPMLdirections() const
{
    std::vector< bool > dirsWithPML( 6, false );
#if AMREX_SPACEDIM!=3
    dirsWithPML.resize( 4 );
#endif
    if( do_pml )
    {
        for( int i = 0; i < static_cast<int>(dirsWithPML.size()) / 2; ++i )
        {
            dirsWithPML.at( 2u*i      ) = bool(do_pml_Lo[0][i]); // on level 0
            dirsWithPML.at( 2u*i + 1u ) = bool(do_pml_Hi[0][i]); // on level 0
        }
    }
    return dirsWithPML;
}

amrex::LayoutData<amrex::Real>*
WarpX::getCosts (int lev)
{
    if (m_instance)
    {
        return m_instance->costs[lev].get();
    } else
    {
        return nullptr;
    }
}

void
WarpX::setLoadBalanceEfficiency (const int lev, const amrex::Real efficiency)
{
    if (m_instance)
    {
        m_instance->load_balance_efficiency[lev] = efficiency;
    } else
    {
        return;
    }
}

amrex::Real
WarpX::getLoadBalanceEfficiency (const int lev)
{
    if (m_instance)
    {
        return m_instance->load_balance_efficiency[lev];
    } else
    {
        return -1;
    }
}

void
WarpX::BuildBufferMasks ()
{
    for (int lev = 1; lev <= maxLevel(); ++lev)
    {
        for (int ipass = 0; ipass < 2; ++ipass)
        {
            const int ngbuffer = (ipass == 0) ? n_current_deposition_buffer : n_field_gather_buffer;
            iMultiFab* bmasks = (ipass == 0) ? current_buffer_masks[lev].get() : gather_buffer_masks[lev].get();
            if (bmasks)
            {
                const IntVect ngtmp = ngbuffer + bmasks->nGrowVect();
                iMultiFab tmp(bmasks->boxArray(), bmasks->DistributionMap(), 1, ngtmp);
                const int covered = 1;
                const int notcovered = 0;
                const int physbnd = 1;
                const int interior = 1;
                const Box& dom = Geom(lev).Domain();
                const amrex::Periodicity& period = Geom(lev).periodicity();
                tmp.BuildMask(dom, period, covered, notcovered, physbnd, interior);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(*bmasks, true); mfi.isValid(); ++mfi)
                {
                    const Box tbx = mfi.growntilebox();
                    BuildBufferMasksInBox( tbx, (*bmasks)[mfi], tmp[mfi], ngbuffer );
                }
            }
        }
    }
}

/**
 * \brief Build buffer mask within given FArrayBox
 *
 * \param tbx         Current FArrayBox
 * \param buffer_mask Buffer mask to be set
 * \param guard_mask  Guard mask used to set buffer_mask
 * \param ng          Number of guard cells
 */
void
WarpX::BuildBufferMasksInBox ( const amrex::Box tbx, amrex::IArrayBox &buffer_mask,
                               const amrex::IArrayBox &guard_mask, const int ng )
{
    auto const& msk = buffer_mask.array();
    auto const& gmsk = guard_mask.const_array();
    const amrex::Dim3 ng3 = amrex::IntVect(ng).dim3();
    amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
    {
        for         (int kk = k-ng3.z; kk <= k+ng3.z; ++kk) {
            for     (int jj = j-ng3.y; jj <= j+ng3.y; ++jj) {
                for (int ii = i-ng3.x; ii <= i+ng3.x; ++ii) {
                    if ( gmsk(ii,jj,kk) == 0 ) {
                        msk(i,j,k) = 0;
                        return;
                    }
                }
            }
        }
        msk(i,j,k) = 1;
    });
}

amrex::Vector<amrex::Real> WarpX::getFornbergStencilCoefficients (const int n_order, ablastr::utils::enums::GridType a_grid_type)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_order % 2 == 0, "n_order must be even");

    const int m = n_order / 2;
    amrex::Vector<amrex::Real> coeffs;
    coeffs.resize(m);

    // There are closed-form formula for these coefficients, but they result in
    // an overflow when evaluated numerically. One way to avoid the overflow is
    // to calculate the coefficients by recurrence.

    // Coefficients for collocated (nodal) finite-difference approximation
    if (a_grid_type == GridType::Collocated)
    {
        // First coefficient
        coeffs.at(0) = m * 2._rt / (m+1);
        // Other coefficients by recurrence
        for (int n = 1; n < m; n++)
        {
            coeffs.at(n) = - (m-n) * 1._rt / (m+n+1) * coeffs.at(n-1);
        }
    }
    // Coefficients for staggered finite-difference approximation
    else
    {
        Real prod = 1.;
        for (int k = 1; k < m+1; k++)
        {
            prod *= (m + k) / (4._rt * k);
        }
        // First coefficient
        coeffs.at(0) = 4_rt * m * prod * prod;
        // Other coefficients by recurrence
        for (int n = 1; n < m; n++)
        {
            coeffs.at(n) = - ((2_rt*n-1) * (m-n)) * 1._rt / ((2_rt*n+1) * (m+n)) * coeffs.at(n-1);
        }
    }

    return coeffs;
}

void WarpX::AllocateCenteringCoefficients (amrex::Gpu::DeviceVector<amrex::Real>& device_centering_stencil_coeffs_x,
                                           amrex::Gpu::DeviceVector<amrex::Real>& device_centering_stencil_coeffs_y,
                                           amrex::Gpu::DeviceVector<amrex::Real>& device_centering_stencil_coeffs_z,
                                           const int centering_nox,
                                           const int centering_noy,
                                           const int centering_noz,
                                           ablastr::utils::enums::GridType a_grid_type)
{
    // Vectors of Fornberg stencil coefficients
    amrex::Vector<amrex::Real> Fornberg_stencil_coeffs_x;
    amrex::Vector<amrex::Real> Fornberg_stencil_coeffs_y;
    amrex::Vector<amrex::Real> Fornberg_stencil_coeffs_z;

    // Host vectors of stencil coefficients used for finite-order centering
    amrex::Vector<amrex::Real> host_centering_stencil_coeffs_x;
    amrex::Vector<amrex::Real> host_centering_stencil_coeffs_y;
    amrex::Vector<amrex::Real> host_centering_stencil_coeffs_z;

    Fornberg_stencil_coeffs_x = getFornbergStencilCoefficients(centering_nox, a_grid_type);
    Fornberg_stencil_coeffs_y = getFornbergStencilCoefficients(centering_noy, a_grid_type);
    Fornberg_stencil_coeffs_z = getFornbergStencilCoefficients(centering_noz, a_grid_type);

    host_centering_stencil_coeffs_x.resize(centering_nox);
    host_centering_stencil_coeffs_y.resize(centering_noy);
    host_centering_stencil_coeffs_z.resize(centering_noz);

    // Re-order Fornberg stencil coefficients:
    // example for order 6: (c_0,c_1,c_2) becomes (c_2,c_1,c_0,c_0,c_1,c_2)
    ::ReorderFornbergCoefficients(
        host_centering_stencil_coeffs_x,
        Fornberg_stencil_coeffs_x,
        centering_nox
    );
    ::ReorderFornbergCoefficients(
        host_centering_stencil_coeffs_y,
        Fornberg_stencil_coeffs_y,
        centering_noy
    );
    ::ReorderFornbergCoefficients(
        host_centering_stencil_coeffs_z,
        Fornberg_stencil_coeffs_z,
        centering_noz
    );

    // Device vectors of stencil coefficients used for finite-order centering

    device_centering_stencil_coeffs_x.resize(host_centering_stencil_coeffs_x.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                          host_centering_stencil_coeffs_x.begin(),
                          host_centering_stencil_coeffs_x.end(),
                          device_centering_stencil_coeffs_x.begin());

    device_centering_stencil_coeffs_y.resize(host_centering_stencil_coeffs_y.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                          host_centering_stencil_coeffs_y.begin(),
                          host_centering_stencil_coeffs_y.end(),
                          device_centering_stencil_coeffs_y.begin());

    device_centering_stencil_coeffs_z.resize(host_centering_stencil_coeffs_z.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                          host_centering_stencil_coeffs_z.begin(),
                          host_centering_stencil_coeffs_z.end(),
                          device_centering_stencil_coeffs_z.begin());

    amrex::Gpu::synchronize();
}

const iMultiFab*
WarpX::CurrentBufferMasks (int lev)
{
    return GetInstance().getCurrentBufferMasks(lev);
}

const iMultiFab*
WarpX::GatherBufferMasks (int lev)
{
    return GetInstance().getGatherBufferMasks(lev);
}

void
WarpX::StoreCurrent (int lev)
{
    using ablastr::fields::Direction;
    for (int idim = 0; idim < 3; ++idim) {
        if (m_fields.has(FieldType::current_store, Direction{idim},lev)) {
            MultiFab::Copy(*m_fields.get(FieldType::current_store, Direction{idim}, lev),
                           *m_fields.get(FieldType::current_fp, Direction{idim}, lev),
                           0, 0, 1, m_fields.get(FieldType::current_store, Direction{idim}, lev)->nGrowVect());
        }
    }
}

void
WarpX::RestoreCurrent (int lev)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    for (int idim = 0; idim < 3; ++idim) {
        if (m_fields.has(FieldType::current_store, Direction{idim}, lev)) {
            std::swap(
                *m_fields.get(FieldType::current_fp, Direction{idim}, lev),
                *m_fields.get(FieldType::current_store, Direction{idim}, lev)
            );
        }
    }
}

bool
WarpX::isAnyParticleBoundaryThermal ()
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (WarpX::particle_boundary_lo[idim] == ParticleBoundaryType::Thermal) {return true;}
        if (WarpX::particle_boundary_hi[idim] == ParticleBoundaryType::Thermal) {return true;}
    }
    return false;
}

void
WarpX::AllocInitMultiFab (
    std::unique_ptr<amrex::iMultiFab>& mf,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm,
    const int ncomp,
    const amrex::IntVect& ngrow,
    const int level,
    const std::string& name,
    std::optional<const int> initial_value)
{
    // Add the suffix "[level=level]"
    const auto name_with_suffix = name + "[level=" + std::to_string(level) + "]";
    const auto tag = amrex::MFInfo().SetTag(name_with_suffix);
    mf = std::make_unique<amrex::iMultiFab>(ba, dm, ncomp, ngrow, tag);
    if (initial_value) {
        mf->setVal(*initial_value);
    }
    imultifab_map[name_with_suffix] = mf.get();
}

amrex::DistributionMapping
WarpX::MakeDistributionMap (int lev, amrex::BoxArray const& ba)
{
    bool roundrobin_sfc = false;
    const ParmParse pp("warpx");
    pp.query("roundrobin_sfc", roundrobin_sfc);

    // If this is true, AMReX's RRSFC strategy is used to make
    // DistributionMapping. Note that the DistributionMapping made by the
    // here could still be overridden by load balancing. In the RRSFC
    // strategy, the Round robin method is used to distribute Boxes orderd
    // by the space filling curve. This might help avoid some processes
    // running out of memory due to having too many particles during
    // initialization.

    if (roundrobin_sfc) {
        auto old_strategy = amrex::DistributionMapping::strategy();
        amrex::DistributionMapping::strategy(amrex::DistributionMapping::RRSFC);
        amrex::DistributionMapping dm(ba);
        amrex::DistributionMapping::strategy(old_strategy);
        return dm;
    } else {
        return amrex::AmrCore::MakeDistributionMap(lev, ba);
    }
}

const amrex::iMultiFab*
WarpX::getFieldDotMaskPointer ( FieldType field_type, int lev, int dir ) const
{
    switch(field_type)
    {
        case FieldType::Efield_fp :
            SetDotMask( Efield_dotMask[lev][dir], "Efield_fp", lev, dir );
            return Efield_dotMask[lev][dir].get();
        case FieldType::Bfield_fp :
            SetDotMask( Bfield_dotMask[lev][dir], "Bfield_fp", lev, dir );
            return Bfield_dotMask[lev][dir].get();
        case FieldType::vector_potential_fp :
            SetDotMask( Afield_dotMask[lev][dir], "vector_potential_fp", lev, dir );
            return Afield_dotMask[lev][dir].get();
        case FieldType::phi_fp :
            SetDotMask( phi_dotMask[lev], "phi_fp", lev, 0 );
            return phi_dotMask[lev].get();
        default:
            WARPX_ABORT_WITH_MESSAGE("Invalid field type for dotMask");
            return Efield_dotMask[lev][dir].get();
    }
}

void WarpX::SetDotMask( std::unique_ptr<amrex::iMultiFab>& field_dotMask,
                        std::string const & field_name, int lev, int dir ) const
{
    // Define the dot mask for this field_type needed to properly compute dotProduct()
    // for field values that have shared locations on different MPI ranks
    if (field_dotMask != nullptr) { return; }

    ablastr::fields::ConstVectorField const& this_field = m_fields.get_alldirs(field_name,lev);
    const amrex::BoxArray& this_ba = this_field[dir]->boxArray();
    const amrex::MultiFab tmp( this_ba, this_field[dir]->DistributionMap(),
                               1, 0, amrex::MFInfo().SetAlloc(false) );
    const amrex::Periodicity& period = Geom(lev).periodicity();
    field_dotMask = tmp.OwnerMask(period);

}
