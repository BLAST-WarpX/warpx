/* Copyright 2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Arianna Formenti
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_DIFFERENTIALLUMINOSITY_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_DIFFERENTIALLUMINOSITY_H_

#include "ReducedDiags.H"
#include <AMReX_GpuContainers.H>

#include <map>
#include <string>
#include <vector>

/**
 *  This class contains the differential luminosity diagnostics.
 */
class DifferentialLuminosity : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    DifferentialLuminosity(const std::string& rd_name);

    /// name of the two colliding species
    std::vector<std::string> m_beam_name;

    /// number of bins
    int m_bin_num;

    /// max and min bin values
    amrex::Real m_bin_max;
    amrex::Real m_bin_min;

    /// bin size
    amrex::Real m_bin_size;

    void ComputeDiags(int step) final;

private:
    /// auxiliary structure to store headers and indices of the reduced diagnostics
    struct aux_header_index
    {
        std::string header;
        int idx;
    };

    /// map to store header texts and indices of the reduced diagnostics
    std::map<std::string, aux_header_index> m_headers_indices;

    // Array in which to accumulate the luminosity across timesteps
    amrex::Gpu::DeviceVector< amrex::Real > d_data;
};

#endif  // WARPX_DIAGNOSTICS_REDUCEDDIAGS_DIFFERENTIALLUMINOSITY_H_
