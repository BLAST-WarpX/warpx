/* This file is part of ABLASTR.
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "MPIInitHelpers.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_Config.H>
#include <AMReX_ParallelDescriptor.H>

#if defined(AMREX_USE_MPI)
#   include <mpi.h>
#endif

// OLCFDEV-1655: Segfault during MPI_Init & in PMI_Allgather
// https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html#olcfdev-1655-occasional-seg-fault-during-mpi-init
#if defined(AMREX_USE_HIP)
#include <hip/hip_runtime.h>
#endif

#include <iostream>
#include <string>
#include <utility>
#include <stdexcept>
#include <sstream>


namespace ablastr::parallelization
{
    constexpr int
    mpi_thread_required ()
    {
#ifdef AMREX_USE_MPI
#   ifdef AMREX_MPI_THREAD_MULTIPLE  // i.e. for async_io
        return MPI_THREAD_MULTIPLE;
#   elif defined(AMREX_USE_OMP)
        return MPI_THREAD_FUNNELED;
#   else
        return MPI_THREAD_SINGLE; // equiv. to MPI_Init
#   endif
#else
        return -1;
#endif
    }

    std::pair< int, int >
    mpi_init (int argc, char* argv[])
    {
        // OLCFDEV-1655: Segfault during MPI_Init & in PMI_Allgather
        // https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html#olcfdev-1655-occasional-seg-fault-during-mpi-init
#if defined(AMREX_USE_HIP) && defined(AMREX_USE_MPI)
        hipError_t hip_ok = hipInit(0);
        if (hip_ok != hipSuccess) {
            std::cerr << "hipInit failed with error code " << hip_ok << "! Aborting now.\n";
            throw std::runtime_error("hipInit failed. Did not proceeding with MPI_Init_thread.");
        }
#endif

        constexpr int thread_required = mpi_thread_required();
#ifdef AMREX_USE_MPI
        int thread_provided = -1;
        MPI_Init_thread(&argc, &argv, thread_required, &thread_provided);
#else
        amrex::ignore_unused(argc, argv);
        const int thread_provided = -1;
#endif
        return std::make_pair(thread_required, thread_provided);
    }


    void
    mpi_finalize ()
    {
#ifdef AMREX_USE_MPI
        MPI_Finalize();
#endif
    }

    void
    check_mpi_thread_level ()
    {
#ifdef AMREX_USE_MPI
        constexpr int thread_required = mpi_thread_required();
        int thread_provided = -1;
        MPI_Query_thread(&thread_provided);
        auto mtn = amrex::ParallelDescriptor::mpi_level_to_string;

        std::stringstream ss;
        if( thread_provided < thread_required ){
            ss << "WARNING: Provided MPI thread safety level ("
                           << mtn(thread_provided) << ") is LOWER than requested "
                           << mtn(thread_required) << "). This might lead to undefined "
                           << "results in asynchronous operations (e.g. async_io).";
            ablastr::warn_manager::WMRecordWarning(
                    "MPI", ss.str(), ablastr::warn_manager::WarnPriority::high);
        }
        if( thread_provided > thread_required ){
            ss << "NOTE: Provided MPI thread safety level ("
                           << mtn(thread_provided) << ") is stricter than requested "
                           << mtn(thread_required) << "). This might reduce multi-node "
                           << "communication performance.";
            ablastr::warn_manager::WMRecordWarning(
                    "MPI", ss.str());
        }
#endif
    }

} // namespace ablastr::parallelization
