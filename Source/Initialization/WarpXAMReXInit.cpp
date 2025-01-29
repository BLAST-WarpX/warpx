/* Copyright 2020 Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Initialization/WarpXAMReXInit.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ccse-mpi.H>
#include <AMReX_Vector.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <string>

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

    /** Overwrite defaults in AMReX Inputs
     *
     * This overwrites defaults in amrex::ParmParse for inputs.
     */
    void
    overwrite_amrex_parser_defaults ()
    {
        override_default_abort_on_out_of_gpu_memory();
        override_default_the_arena_is_managed();
        override_default_omp_threads();
        apply_workaround_for_warpx_numprocs();
        set_device_synchronization();
        override_default_tiling_option_for_particles();
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
