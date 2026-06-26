/* Copyright 2020 Axel Huebl
 *
 * This file is part of ABLASTR.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_MPI_INIT_HELPERS_H_
#define ABLASTR_MPI_INIT_HELPERS_H_

#include <utility>

namespace ablastr::parallelization
{
    /** Return the required MPI threading
     *
     * @return the MPI_THREAD_* level required for MPI_Init_thread
     */
    constexpr int
    mpi_thread_required ();

    /** Initialize MPI
     *
     * @return pair(required, provided) of MPI thread level from MPI_Init_thread
     */
    std::pair< int, int >
    mpi_init (int argc, char* argv[]);

    /** Finalize MPI
     *
     * This function is simply a wrapper around MPI_Finalize(). It is a no-op if
     * ABLASTR is compiled without MPI support.
     */
    void
    mpi_finalize ();

    /** Check if the requested MPI thread level is valid
     *
     * Prints warnings and notes otherwise.
     */
    void
    check_mpi_thread_level ();

} // namespace ablastr::parallelization

#endif // ABLASTR_MPI_INIT_HELPERS_H_
