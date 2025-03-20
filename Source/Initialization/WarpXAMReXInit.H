/* Copyright 2020 Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_AMREX_INIT_H_
#define WARPX_AMREX_INIT_H_

#include <AMReX_BaseFwd.H>

namespace warpx::initialization
{

    /** Call amrex::Initialize
     *
     * This function calls amrex::Initialize and overwrites AMReX' defaults.
     * Note: AMReX defines a placeholder/"mock-up" for MPI_Comm and
     * MPI_COMM_WORLD in serial builds
     *
     * @param[in] argc number of arguments from main()
     * @param[in] argv argument strings from main()
     * @param[in] build_parm_parse build the input file parser (AMReX' default: true)
     * @param[in] mpi_comm the MPI communicator to use (AMReX' default: MPI_COMM_WORLD)
     * @returns pointer to an AMReX* object, forwarded from amrex::Initialize
     */
    amrex::AMReX*
    amrex_init(
        int& argc,
        char**& argv,
        bool build_parm_parse = true
        /*  MPI_Comm mpi_comm = MPI_COMM_WORLD */
    );

}
#endif //WARPX_AMREX_INIT_H_
