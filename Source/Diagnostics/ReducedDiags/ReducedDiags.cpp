/* Copyright 2019-2020 Maxence Thevenet, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ReducedDiags.H"

#include "WarpX.H"
#include "Utils/Parser/IntervalsParser.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>

#include <fstream>
#include <iomanip>

using namespace amrex;

// constructor
ReducedDiags::ReducedDiags (const std::string& rd_name):
m_rd_name{rd_name}
{
    BackwardCompatibility();

    const ParmParse pp_rd("reduced_diags");
    const ParmParse pp_rd_name(m_rd_name);

    // read path
    pp_rd.query("path", m_path);
    pp_rd_name.query("path", m_path);

    // read extension
    pp_rd.query("extension", m_extension);
    pp_rd_name.query("extension", m_extension);

    // check if it is a restart run
    std::string restart_chkfile;
    const ParmParse pp_amr("amr");
    pp_amr.query("restart", restart_chkfile);
    const bool IsNotRestart = restart_chkfile.empty();

    if (ParallelDescriptor::IOProcessor())
    {
        // create folder
        constexpr int permission_flag_rwxrxrx = 0755;
        if (!amrex::UtilCreateDirectory(m_path, permission_flag_rwxrxrx))
        { amrex::CreateDirectoryFailed(m_path); }

        // replace / create output file
        const std::string rd_full_file_name = m_path + m_rd_name + "." + m_extension;
        m_write_header = IsNotRestart || !amrex::FileExists(rd_full_file_name); // not a restart or file doesn't exist
        if (m_write_header)
        {
            std::ofstream ofs{rd_full_file_name, std::ios::trunc};
            ofs.close();
        }
    }

    // read reduced diags intervals
    std::vector<std::string> intervals_string_vec = {"1"};
    pp_rd.queryarr("intervals", intervals_string_vec);
    pp_rd_name.queryarr("intervals", intervals_string_vec);
    m_intervals = utils::parser::IntervalsParser(intervals_string_vec);

    // read separator
    pp_rd.query("separator", m_sep);
    pp_rd_name.query("separator", m_sep);

    // precision of data in the output file
    utils::parser::queryWithParser(pp_rd, "precision", m_precision);
    utils::parser::queryWithParser(pp_rd_name, "precision", m_precision);
}
// end constructor

void ReducedDiags::InitData ()
{
    // Defines an empty function InitData() to be overwritten if needed.
    // Function used to initialize data of the diagnostics after the WarpX
    // data structures are all set up
}

void ReducedDiags::LoadBalance ()
{
    // Defines an empty function LoadBalance() to be overwritten if needed.
    // Function used to redistribute parallel data of the diagnostics in
    // load balancing operations
}

void ReducedDiags::BackwardCompatibility () const
{
    const amrex::ParmParse pp_rd_name(m_rd_name);
    std::vector<std::string> backward_strings;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_rd_name.queryarr("frequency", backward_strings),
        "<reduced_diag_name>.frequency is no longer a valid option. "
        "Please use the renamed option <reduced_diag_name>.intervals instead."
    );
}

// write to file function
void ReducedDiags::WriteToFile (int step) const
{
    // open file
    std::ofstream ofs{m_path + m_rd_name + "." + m_extension,
        std::ofstream::out | std::ofstream::app};

    // write step
    ofs << step+1;

    ofs << m_sep;

    // set precision
    ofs << std::fixed << std::setprecision(m_precision) << std::scientific;

    // write time
    ofs << WarpX::GetInstance().gett_new(0);

    // loop over data size and write
    for (const auto& item : m_data) { ofs << m_sep << item; }

    // end loop over data size

    // end line
    ofs << "\n";

    // close file
    ofs.close();
}
// end ReducedDiags::WriteToFile
