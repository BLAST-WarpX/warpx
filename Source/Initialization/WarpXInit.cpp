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

#include <string>

void warpx::initialization::initialize_external_libraries(int argc, char* argv[])
{
    ablastr::parallelization::mpi_init(argc, argv);
    warpx::initialization::amrex_init(argc, argv);
    ablastr::math::anyfft::setup();
}

void warpx::initialization::finalize_external_libraries()
{
    ablastr::math::anyfft::cleanup();
    amrex::Finalize();
    ablastr::parallelization::mpi_finalize();
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
