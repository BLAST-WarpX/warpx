/* Copyright 2024 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "WarpXInit.H"

#include "Initialization/WarpXAMReXInit.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXConst.H"
#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include <ablastr/math/fft/AnyFFT.H>
#include <ablastr/parallelization/MPIInitHelpers.H>
#include <ablastr/warn_manager/WarnManager.H>

#ifdef AMREX_USE_PETSC
#include <petscsys.h>
#endif


#include <optional>
#include <string>

void warpx::initialization::initialize_external_libraries(int argc, char* argv[])
{
    ablastr::parallelization::mpi_init(argc, argv);
    warpx::initialization::amrex_init(argc, argv);
    ablastr::math::anyfft::setup();
#ifdef AMREX_USE_PETSC
    PETSC_COMM_WORLD = amrex::ParallelContext::CommunicatorSub();
    PetscInitialize(&argc, &argv, nullptr, "WarpX with PETSc");
    amrex::Print() << "Initialized PETSc.\n";
#endif
}

void warpx::initialization::finalize_external_libraries ()
{
#ifdef AMREX_USE_PETSC
    PetscFinalize();
    amrex::Print() << "Finalized PETSc.\n";
#endif
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

void warpx::initialization::check_dims()
{
    // Ensure that geometry.dims is set properly.
#if defined(WARPX_DIM_3D)
    std::string const dims_compiled = "3";
#elif defined(WARPX_DIM_XZ)
    std::string const dims_compiled = "2";
#elif defined(WARPX_DIM_1D_Z)
    std::string const dims_compiled = "1";
#elif defined(WARPX_DIM_RZ)
    std::string const dims_compiled = "RZ";
#elif defined(WARPX_DIM_RCYLINDER)
    std::string const dims_compiled = "RCYLINDER";
#elif defined(WARPX_DIM_RSPHERE)
    std::string const dims_compiled = "RSPHERE";
#endif
    const amrex::ParmParse pp_geometry("geometry");
    std::string dims;
    std::string dims_error = "The selected WarpX executable was built as '";
    dims_error.append(dims_compiled).append("'-dimensional, but the ");
    if (pp_geometry.contains("dims")) {
        pp_geometry.get("dims", dims);
        dims_error.append("inputs file declares 'geometry.dims = ").append(dims).append("'.\n");
        dims_error.append("Please re-compile with a different WarpX_DIMS option or select the right executable name.");
    } else {
        dims = "Not specified";
        dims_error.append("inputs file does not declare 'geometry.dims'. Please add 'geometry.dims = ");
        dims_error.append(dims_compiled).append("' to inputs file.");
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(dims == dims_compiled, dims_error);
}

void warpx::initialization::read_moving_window_parameters(
    int& do_moving_window, int& start_moving_window_step, int& end_moving_window_step,
    [[maybe_unused]] int& moving_window_dir, amrex::Real& moving_window_v)
{
    const amrex::ParmParse pp_warpx("warpx");
    pp_warpx.query("do_moving_window", do_moving_window);
    if (do_moving_window) {
        utils::parser::queryWithParser(
            pp_warpx, "start_moving_window_step", start_moving_window_step);
        utils::parser::queryWithParser(
            pp_warpx, "end_moving_window_step", end_moving_window_step);
        std::string s;
        pp_warpx.get("moving_window_dir", s);

        if (s == "z" || s == "Z") {
#ifdef WARPX_ZINDEX
            moving_window_dir = WARPX_ZINDEX;
#endif
        }
#if defined(WARPX_DIM_3D)
        else if (s == "y" || s == "Y") {
            moving_window_dir = 1;
        }
#endif
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D)
        else if (s == "x" || s == "X") {
            moving_window_dir = 0;
        }
#endif
        else {
            WARPX_ABORT_WITH_MESSAGE("Unknown moving_window_dir: "+s);
        }

        utils::parser::getWithParser(
            pp_warpx, "moving_window_v", moving_window_v);
        moving_window_v *= PhysConst::c;
    }
}
