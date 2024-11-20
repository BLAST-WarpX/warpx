/* Copyright 2019-2020 Michael Rowan, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_LOADBALANCECOSTS_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_LOADBALANCECOSTS_H_

#include "ReducedDiags.H"

#include <AMReX_Vector.H>

#include <string>
#include <vector>

/**
 *  This class mainly contains a function that update the
 *  costs (used in load balance) for writing to output.
 */
class LoadBalanceCosts : public ReducedDiags
{
public:

    /** stores the host identifiers */
    amrex::Vector<char> m_data_string_recvbuf;

    /** length of the total string to be collected to IOProc  */
    int m_data_string_recvbuf_length = 0;

    /** easier storage of strings after the MPI Gatherv */
    std::vector<std::string> m_data_string;

    /** stores information needed for MPI Gatherv */
    amrex::Vector<int> m_data_string_recvcount; // array of size N_procs, how many message root recv from sender
    amrex::Vector<int> m_data_string_disp;      // array of size N_procs, where to place data in IOProc

    /** number of data fields we save for each box
     *  (cost, processor, level, i_low, j_low, k_low, gpu_ID [if GPU run], num_cells, num_macro_particles
     * note: the hostname per box is stored separately (in m_data_string) */
#ifdef AMREX_USE_GPU
    const int m_nDataFields = 9;
#else
    const int m_nDataFields = 8;
#endif

    /** used to keep track of max number of boxes over all timesteps; this allows
     *  to compute the number of NaNs required to fill jagged array into a
     *  rectangular one */
    int m_nBoxesMax = -1;

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    LoadBalanceCosts(const std::string& rd_name);

    /**
     * This function updates the costs, given the current distribution mapping,
     * according to the number of particles and cells on the box
     *
     * @param[in] step current time step
     */
    void ComputeDiags(int step) final;

    /**
     * write to file function for costs;  this differs from the base class
     * `ReducedDiags` in that it will fill in blank entries with NaN at the
     * final timestep, ensuring that the data array is not jagged
     *
     * @param[in] step current time step
     */
    void WriteToFile(int step) const final;

};

#endif
