/* Copyright 2020 Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Initialization/WarpXAMReXInit.H"

#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_ccse-mpi.H>
#include <AMReX_ParmParse.H>
#include <AMReX_BLProfiler.H>

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

    void add_constants ()
    {
        amrex::ParmParse::SetParserPrefix("my_constants");
        amrex::ParmParse pp_constants("my_constants");
        pp_constants.add("clight", PhysConst::c);
        pp_constants.add("epsilon0", PhysConst::ep0);
        pp_constants.add("mu0", PhysConst::mu0);
        pp_constants.add("q_e", PhysConst::q_e);
        pp_constants.add("m_e", PhysConst::m_e);
        pp_constants.add("m_p", PhysConst::m_p);
        pp_constants.add("m_u", PhysConst::m_u);
        pp_constants.add("kb", PhysConst::kb);
        pp_constants.add("pi", MathConst::pi);
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
        add_constants();
    }
}

namespace warpx::initialization
{

    amrex::AMReX*
    amrex_init (int& argc, char**& argv, bool build_parm_parse)
    {
        return amrex::Initialize(
            argc,
            argv,
            build_parm_parse,
            MPI_COMM_WORLD,
            ::overwrite_amrex_parser_defaults
        );
    }

}
