/* Copyright 2024 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "WarpXInit.H"

#include "Initialization/WarpXAMReXInit.H"
#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include <ablastr/math/fft/AnyFFT.H>
#include <ablastr/parallelization/MPIInitHelpers.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <optional>
#include <string>

void warpx::initialization::initialize_external_libraries (int argc, char* argv[])
{
    ablastr::parallelization::mpi_init(argc, argv);
    warpx::initialization::amrex_init(argc, argv);
    ablastr::math::anyfft::setup();
}

void warpx::initialization::finalize_external_libraries ()
{
    ablastr::math::anyfft::cleanup();
    amrex::Finalize();
    ablastr::parallelization::mpi_finalize();
}

void warpx::initialization::initialize_warning_manager ()
{
    const auto pp_warpx = amrex::ParmParse{"warpx"};

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
}
