/* Copyright 2020 Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Initialization/WarpXAMReXInit.H"

#include "BoundaryConditions/FieldBoundaries.H"
#include "Particles/ParticleBoundaries.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ccse-mpi.H>
#include <AMReX_Vector.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <string>

// for MPI_COMM_WORLD in non-MPI build
using namespace amrex;

namespace {

#ifdef AMREX_USE_GPU
        constexpr auto amrex_use_gpu = true;
#else
        constexpr auto amrex_use_gpu = false;
#endif

    void override_default_abort_on_out_of_gpu_memory ()
    {
        // https://amrex-codes.github.io/amrex/docs_html/GPU.html#inputs-parameters
        auto pp_amrex = amrex::ParmParse{"amrex"};
        bool abort_on_out_of_gpu_memory = true; // AMReX's default: false
        pp_amrex.queryAdd("abort_on_out_of_gpu_memory", abort_on_out_of_gpu_memory);
    }

    void override_default_the_arena_is_managed ()
    {
        auto pp_amrex = amrex::ParmParse{"amrex"};
        bool the_arena_is_managed = false; // AMReX's default: true
        pp_amrex.queryAdd("the_arena_is_managed", the_arena_is_managed);
    }

    void override_default_omp_threads ()
    {
        // https://amrex-codes.github.io/amrex/docs_html/InputsComputeBackends.html
        auto pp_amrex = amrex::ParmParse{"amrex"};
        std::string omp_threads = "nosmt"; // AMReX's default: system
        pp_amrex.queryAdd("omp_threads", omp_threads);
    }

    void set_device_synchronization ()
    {
        //See https://github.com/AMReX-Codes/amrex/pull/3763
        auto warpx_do_device_synchronize = amrex_use_gpu;

        auto pp_warpx = amrex::ParmParse{"warpx"};
        pp_warpx.query("do_device_synchronize", warpx_do_device_synchronize);
        bool do_device_synchronize = warpx_do_device_synchronize;

        auto pp_tiny_profiler = amrex::ParmParse{"tiny_profiler"};
        if (pp_tiny_profiler.queryAdd("device_synchronize_around_region", do_device_synchronize) )
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                do_device_synchronize == warpx_do_device_synchronize,
                "tiny_profiler.device_synchronize_around_region overrides warpx.do_device_synchronize.");
        }

    }

    void apply_workaround_for_warpx_numprocs ()
    {
        // Work-around:
        // If warpx.numprocs is used for the domain decomposition, we will not use blocking factor
        // to generate grids. Nonetheless, AMReX has asserts in place that validate that the
        // number of cells is a multiple of blocking factor. We set the blocking factor to 1 so those
        // AMReX asserts will always pass.
        const auto pp_warpx = amrex::ParmParse{"warpx"};
        if (pp_warpx.contains("numprocs"))
        {
            amrex::ParmParse pp_amr("amr");
            pp_amr.add("blocking_factor", 1);
        }
    }

    void override_default_tiling_option_for_particles ()
    {
        // Here we override the default tiling option for particles, which is always
        // "false" in AMReX, to "false" if compiling for GPU execution and "true"
        // if compiling for CPU.
        auto pp_particles = amrex::ParmParse{"particles"};
        auto do_tiling = !amrex_use_gpu; // By default, tiling is off on GPU
        pp_particles.queryAdd("do_tiling", do_tiling);
    }

    void set_periodicity_according_to_boundary_types ()
    {
        auto pp_geometry = amrex::ParmParse{"geometry"};
        if (pp_geometry.contains("is_periodic")){
            std::string const warnMsg =
                "geometry.is_periodic is only used internally. Please use `boundary.field_lo`,"
                " `boundary.field_hi` to specifiy field boundary conditions and"
                " 'boundary.particle_lo', 'boundary.particle_hi'  to specify particle"
                " boundary conditions.";
            ablastr::warn_manager::WMRecordWarning("Input", warnMsg);
        }

        const auto [field_boundary_lo, field_boundary_hi] =
            warpx::boundary_conditions::parse_field_boundaries();

        const auto is_field_boundary_periodic =
            warpx::boundary_conditions::get_periodicity_array(field_boundary_lo, field_boundary_hi);

        const auto [particle_boundary_lo, particle_boundary_hi] =
            warpx::particles::parse_particle_boundaries(is_field_boundary_periodic);

        amrex::Vector<int> geom_periodicity(AMREX_SPACEDIM,0);

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            if (field_boundary_lo[idim] == FieldBoundaryType::Periodic ||
                field_boundary_hi[idim] == FieldBoundaryType::Periodic ||
                particle_boundary_lo[idim] == ParticleBoundaryType::Periodic ||
                particle_boundary_hi[idim] == ParticleBoundaryType::Periodic ) {
                    geom_periodicity[idim] = 1;
            }
        }

        // Appending periodicity information to input so that it can be used by amrex
        // to set parameters necessary to define geometry and perform communication
        // such as FillBoundary. The periodicity is 1 if user-define boundary condition is
        // periodic else it is set to 0.
        pp_geometry.addarr("is_periodic", geom_periodicity);
    }

    void add_constants ()
    {
        amrex::ParmParse::SetParserPrefix("my_constants");
        amrex::ParmParse pp_constants("my_constants");
        // Add constants only if it's not defined already.
        amrex::Real tmp = PhysConst::c;
        pp_constants.queryAdd("clight", tmp);
        tmp =       PhysConst::ep0;
        pp_constants.queryAdd("epsilon0", tmp);
        tmp =       PhysConst::mu0;
        pp_constants.queryAdd("mu0", tmp);
        tmp =       PhysConst::q_e;
        pp_constants.queryAdd("q_e", tmp);
        tmp =       PhysConst::m_e;
        pp_constants.queryAdd("m_e", tmp);
        tmp =       PhysConst::m_p;
        pp_constants.queryAdd("m_p", tmp);
        tmp =       PhysConst::m_u;
        pp_constants.queryAdd("m_u", tmp);
        tmp =       PhysConst::kb;
        pp_constants.queryAdd("kb", tmp);
        tmp =       MathConst::pi;
        pp_constants.queryAdd("pi", tmp);
    }

    /** Overwrite defaults in AMReX Inputs
     *
     * This overwrites defaults in amrex::ParmParse for inputs.
     */
    void
    overwrite_amrex_parser_defaults ()
    {
        add_constants();
        override_default_abort_on_out_of_gpu_memory();
        override_default_the_arena_is_managed();
        override_default_omp_threads();
        apply_workaround_for_warpx_numprocs();
        set_device_synchronization();
        override_default_tiling_option_for_particles();
        set_periodicity_according_to_boundary_types();
    }

    /** Parse prob_lo and hi
     *
     * Parse prob_lo and hi evaluating any expressions since geometry
     * does not parse its input. Note that this operation has to be
     * performed after having initialized AMReX
     */
    void parse_geometry_input ()
    {
        auto pp_geometry = amrex::ParmParse {"geometry"};

        auto prob_lo = amrex::Vector<amrex::Real>(AMREX_SPACEDIM);
        auto prob_hi = amrex::Vector<amrex::Real>(AMREX_SPACEDIM);

        utils::parser::getArrWithParser(
            pp_geometry, "prob_lo", prob_lo, 0, AMREX_SPACEDIM);
        utils::parser::getArrWithParser(
            pp_geometry, "prob_hi", prob_hi, 0, AMREX_SPACEDIM);

        AMREX_ALWAYS_ASSERT(prob_lo.size() == AMREX_SPACEDIM);
        AMREX_ALWAYS_ASSERT(prob_hi.size() == AMREX_SPACEDIM);

        pp_geometry.addarr("prob_lo", prob_lo);
        pp_geometry.addarr("prob_hi", prob_hi);

        // Parse amr input, evaluating any expressions since amr does not parse its input
        auto pp_amr = amrex::ParmParse{"amr"};

        // Note that n_cell is replaced so that only the parsed version is written out to the
        // warpx_job_info file. This must be done since yt expects to be able to parse
        // the value of n_cell from that file. For the rest, this doesn't matter.
        auto preparse_amrex_input_int_array =
            [&pp_amr](const std::string& input_str, const bool replace = false)
            {
                const auto *const c_input_str = input_str.c_str();
                if (pp_amr.contains(c_input_str)) {
                    amrex::Vector<int> input_array;
                    utils::parser::getArrWithParser(pp_amr,c_input_str, input_array);
                    if (replace) {
                        pp_amr.remove(c_input_str);
                    }
                    pp_amr.addarr(c_input_str, input_array);
                }
            };

        preparse_amrex_input_int_array("n_cell", true);

        const auto params_to_parse = std::vector<std::string>{
            "max_grid_size", "max_grid_size_x", "max_grid_size_y", "max_grid_size_z",
            "blocking_factor", "blocking_factor_x", "blocking_factor_y", "blocking_factor_z"};
        std::for_each(params_to_parse.begin(), params_to_parse.end(), preparse_amrex_input_int_array);
    }

    /** This method groups calls to functions related to the initialization of AMReX
     * that can run only after having called amrex::Initialize
     */
    void amrex_post_initialize ()
    {
        parse_geometry_input();
    }
}

namespace warpx::initialization
{

    amrex::AMReX*
    amrex_init (int& argc, char**& argv, bool build_parm_parse)
    {
        amrex::AMReX* amrex =
            amrex::Initialize(
                argc,
                argv,
                build_parm_parse,
                MPI_COMM_WORLD,
                ::overwrite_amrex_parser_defaults
            );

        ::amrex_post_initialize();

        return amrex;
    }

}
