/* Copyright 2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Juliette Pech, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_PARTICLEHISTOGRAM2D_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_PARTICLEHISTOGRAM2D_H_

#include "ReducedDiags.H"

#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_TableData.H>

#include <memory>
#include <string>

/**
 * Reduced diagnostics that computes a histogram over particles
 * for a quantity specified by the user in the input file using the parser.
 */
class ParticleHistogram2D : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    ParticleHistogram2D(const std::string& rd_name);

    /// File type
    std::string m_openpmd_backend {"default"};

    /// minimum number of digits for file suffix (file-based only supported now for now) */
    int m_file_min_digits = 6;

    /// number of bins on the abscissa and ordinate
    int m_bin_num_abs;
    int m_bin_num_ord;

    /// output data
    amrex::TableData<amrex::Real,2> m_h_data_2D;

    /// selected species index
    int m_selected_species_id = -1;

    /// max and min bin values
    amrex::Real m_bin_max_abs;
    amrex::Real m_bin_min_abs;
    amrex::Real m_bin_max_ord;
    amrex::Real m_bin_min_ord;

    /// bin sizes
    amrex::Real m_bin_size_abs;
    amrex::Real m_bin_size_ord;

    /// functions
    std::string function_string_abs;
    std::string function_string_ord;
    std::string filter_string;
    std::string value_string;

/// Parser to read expression for particle quantity from the input file.
    /// 7 elements are t, x, y, z, ux, uy, uz, w
    static constexpr int m_nvars = 8;
    std::unique_ptr<amrex::Parser> m_parser_abs;
    std::unique_ptr<amrex::Parser> m_parser_ord;

    /// Optional parser to filter particles before doing the histogram
    std::unique_ptr<amrex::Parser> m_parser_filter;

    /// Whether the filter is activated
    bool m_do_parser_filter = false;

    /// Optional parser to filter particles before doing the histogram
    std::unique_ptr<amrex::Parser> m_parser_value;

    /// Whether the filter is activated
    bool m_do_parser_value;

    /**
     * This function computes a histogram of user defined quantity.
     *
     * @param[in] step current time step
     */
    void ComputeDiags(int step) final;

    /**
     * write to file function
     *
     * @param[in] step current time step
     */
    void WriteToFile (int step) const final;

};

#endif
