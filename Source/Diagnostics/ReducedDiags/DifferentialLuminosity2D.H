/* Copyright 2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Arianna Formenti, Remi Lehe
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_DIFFERENTIALLUMINOSITY2D_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_DIFFERENTIALLUMINOSITY2D_H_

#include "ReducedDiags.H"
#include <AMReX_GpuContainers.H>
#include <AMReX_TableData.H>

#include <map>
#include <string>
#include <vector>

/**
 *  This class contains the differential luminosity diagnostics.
 */
class DifferentialLuminosity2D : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    DifferentialLuminosity2D(const std::string& rd_name);

    /// File type
    std::string m_openpmd_backend {"default"};

    /// minimum number of digits for file suffix (file-based only supported for now) */
    int m_file_min_digits = 6;

    /// name of the two colliding species
    std::vector<std::string> m_beam_name;

    /// number of bins for the c.o.m. energy of the 2 species
    int m_bin_num_1;
    int m_bin_num_2;

    /// max and min bin values
    amrex::Real m_bin_max_1;
    amrex::Real m_bin_min_1;
    amrex::Real m_bin_max_2;
    amrex::Real m_bin_min_2;

    /// bin size
    amrex::Real m_bin_size_1;
    amrex::Real m_bin_size_2;

    /// output data
    amrex::TableData<amrex::Real,2> m_h_data_2D;

    void ComputeDiags(int step) final;

    void WriteToFile (int step) const final;

private:

    /// output table in which to accumulate the luminosity across timesteps
    amrex::TableData<amrex::Real,2> m_d_data_2D;

};

#endif  // WARPX_DIAGNOSTICS_REDUCEDDIAGS_DIFFERENTIALLUMINOSITY2D_H_
