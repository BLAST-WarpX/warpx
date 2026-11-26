#include "HostRandomState.H"

#include "Utils/TextMsg.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_OpenMP.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>
#include <AMReX_Utility.H>

#include <fstream>
#include <string>

namespace warpx::checkpoint
{
    namespace
    {
        bool host_random_state_enabled ()
        {
            bool enabled = false;
            amrex::ParmParse("warpx").query("checkpoint_host_random_state", enabled);
            return enabled;
        }

        std::string host_stream_file (std::string const& directory)
        {
            return amrex::Concatenate(directory + "/Rank_", amrex::ParallelDescriptor::MyProc(), 6);
        }

        void warn_host_stream_restart (std::string const& reason)
        {
            if (amrex::ParallelDescriptor::IOProcessor()) {
                ablastr::warn_manager::WMRecordWarning("Restart",
                    reason + " Continuing with the host random streams from initialization.",
                    ablastr::warn_manager::WarnPriority::medium);
            }
        }
    }

    void write_host_random_state (std::string const& checkpoint)
    {
        if (!host_random_state_enabled()) { return; }

        namespace pd = amrex::ParallelDescriptor;
        std::string const directory = checkpoint + "/HostRandomState";
        if (pd::IOProcessor()) {
            if (!amrex::UtilCreateDirectory(directory, 0755)) {
                amrex::CreateDirectoryFailed(directory);
            }
            std::ofstream header(directory + "/Header");
            header << "WarpXHostRandomState-v1 " << pd::NProcs() << '\n';
            header.close();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(header, "Could not write " + directory + "/Header");
        }
        pd::Barrier();

        auto const file = host_stream_file(directory);
        std::ofstream out(file);
        out << amrex::OpenMP::get_max_threads() << '\n';
        amrex::SaveRandomState(out);
        out.close();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(out, "Could not write " + file);
    }

    void restore_host_random_state (std::string const& checkpoint, int const step)
    {
        if (!host_random_state_enabled() || checkpoint.empty()) { return; }

        namespace pd = amrex::ParallelDescriptor;
        std::string const directory = checkpoint + "/HostRandomState";
        if (!amrex::FileExists(directory)) {
            warn_host_stream_restart("The checkpoint has no saved host random state.");
            return;
        }

        std::ifstream header(directory + "/Header");
        std::string version;
        int ranks = 0;
        header >> version >> ranks;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(header && version == "WarpXHostRandomState-v1" && ranks > 0,
            "Invalid host random state header in " + directory);
        if (ranks != pd::NProcs()) {
            warn_host_stream_restart("The checkpoint used a different number of MPI ranks.");
            return;
        }

        auto const file = host_stream_file(directory);
        std::ifstream in(file);
        int threads = 0;
        in >> threads;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(in && threads > 0, "Invalid host random state in " + file);
        int different_threads = (threads != amrex::OpenMP::get_max_threads());
        pd::ReduceIntMax(different_threads);
        if (different_threads) {
            warn_host_stream_restart("The checkpoint used a different number of host threads.");
            return;
        }

        amrex::RestoreRandomState(in, threads, step);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(in, "Incomplete host random state in " + file);
    }
}
