/* Copyright 2024 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_INIT_H_
#define WARPX_INIT_H_

#include <AMReX_REAL.H>

namespace warpx::initialization
{
    /** Initializes, in the following order:
     * - the MPI library through the mpi_init helper function in ablastr
     * - the AMReX library
     * - the FFT library through the anyfft::setup() function in ablastr
     *
     * @param[in] argc number of arguments from main()
     * @param[in] argv argument strings from main()
     */
    void initialize_external_libraries (int argc, char* argv[]);

    /** Initializes, in the following order:
     * - the FFT library through the anyfft::cleanup() function in ablastr
     * - the AMReX library
     * - the MPI library through the mpi_finalize helper function in ablastr
     */
    void finalize_external_libraries ();

    /**
     * Initializes the Warning manager in ablastr
     */
    void initialize_warning_manager ();

    /** Check that warpx.dims matches the binary name
    */
    void check_dims ();

    /** Reads the moving window parameters
     *
     * @param[out] do_moving_window          if the moving window is enabled
     * @param[out] start_moving_window_step  step at which the moving window is started
     * @param[out] end_moving_window_step    step at which the moving window is stopped
     * @param[out] moving_window_dir         direction of the moving window
     * @param[out] moving_window_v           velocity of the moving  window
    */
    void read_moving_window_parameters (
        int& do_moving_window, int& start_moving_window_step, int& end_moving_window_step,
        int& moving_window_dir, amrex::Real& moving_window_v);
}

#endif //WARPX_INIT_H_
