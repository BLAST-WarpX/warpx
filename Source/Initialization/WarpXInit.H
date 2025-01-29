/* Copyright 2024 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_INIT_H_
#define WARPX_INIT_H_

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
    void initialize_external_libraries(int argc, char* argv[]);

    /** Initializes, in the following order:
     * - the FFT library through the anyfft::cleanup() function in ablastr
     * - the AMReX library
     * - the MPI library through the mpi_finalize helper function in ablastr
     */
    void finalize_external_libraries();

    /** Check that warpx.dims matches the binary name
    */
    void check_dims ();
}

#endif //WARPX_INIT_H_
