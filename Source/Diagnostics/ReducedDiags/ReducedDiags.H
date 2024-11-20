/* Copyright 2019-2020 Maxence Thevenet, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_REDUCEDDIAGS_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_REDUCEDDIAGS_H_

#include "Utils/Parser/IntervalsParser.H"

#include <AMReX_REAL.H>

#include <string>
#include <vector>

/**
 *  Base class for reduced diagnostics. Each type of reduced diagnostics is
 *  implemented in a derived class, and must override the (pure virtual)
 *  function ComputeDiags.
 */
class ReducedDiags
{
public:

    /// output path (default)
    std::string m_path = "./diags/reducedfiles/";

    /// output extension (default)
    std::string m_extension = "txt";

    /// diags name
    std::string m_rd_name;

    /// output intervals
    utils::parser::IntervalsParser m_intervals;

    /// check if header should be written
    bool m_write_header = false;

    /// separator in the output file
    std::string m_sep = " ";

    /// precision for data in the output file
    int m_precision = 14;

    /// output data
    std::vector<amrex::Real> m_data;

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    ReducedDiags (const std::string& rd_name);

    /**
     * Virtual destructor for polymorphism
     */
    virtual ~ReducedDiags () = default;

    // Default move and copy operations
    ReducedDiags(const ReducedDiags&) = default;
    ReducedDiags& operator=(const ReducedDiags&) = default;
    ReducedDiags(ReducedDiags&&) = default;
    ReducedDiags& operator=(ReducedDiags&&) = default;


    /**
     * function to initialize data after amr
     * levels are initialized.
     */
    virtual void InitData ();

    /** Redistribute parallel data structures during load balance
     */
    virtual void LoadBalance ();

    /**
     * function to compute diags
     *
     * @param[in] step current time step
     */
    virtual void ComputeDiags (int step) = 0;

    /**
     * write to file function
     *
     * @param[in] step current time step
     */
    virtual void WriteToFile (int step) const;

    /**
     * This function queries deprecated input parameters and aborts
     * the run if one of them is specified.
     */
    void BackwardCompatibility () const;

};

#endif
