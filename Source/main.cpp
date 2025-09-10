/* Copyright 2016-2020 Andrew Myers, Ann Almgren, Axel Huebl
 *                     David Grote, Jean-Luc Vay, Remi Lehe
 *                     Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#include "Initialization/WarpXInit.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <ablastr/utils/timer/Timer.H>

#include <AMReX_Print.H>

#ifdef AMREX_USE_PETSC
#include <petscsys.h>
#endif

int
main (int argc, char* argv[]) {
    warpx::initialization::initialize_external_libraries(argc, argv);
#ifdef AMREX_USE_PETSC
    PETSC_COMM_WORLD = amrex::ParallelContext::CommunicatorSub();
    PetscInitialize(&argc, &argv, nullptr, "WarpX with PETSc");
    amrex::Print() << "Initialized PETSc.\n";
#endif
    {
        WARPX_PROFILE_VAR("main()", pmain);

        auto timer = ablastr::utils::timer::Timer{};
        timer.record_start_time();

        auto& warpx = WarpX::GetInstance();
        warpx.InitData();
        warpx.Evolve();
        const auto is_warpx_verbose = warpx.Verbose();
        WarpX::Finalize();

        timer.record_stop_time();
        if (is_warpx_verbose) {
            amrex::Print() << "Total Time                     : "
                           << timer.get_global_duration() << '\n';
        }

        WARPX_PROFILE_VAR_STOP(pmain);
    }
#ifdef AMREX_USE_PETSC
    PetscFinalize();
    amrex::Print() << "Finalized PETSc.\n";
#endif
    warpx::initialization::finalize_external_libraries();
}
