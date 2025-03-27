/* Copyright 2025 Marco Garten
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "MemoryPerRank.H"

#include "WarpX.H"

#include <AMReX_Arena.H>
#include <AMReX_ParmParse.H>

#include <utility>
#include <iomanip>
#include <sstream>


namespace {
    constexpr int permission_flag_rwxrxrx = 0755;
}

// Constructor
MemoryPerRank::MemoryPerRank (const std::string& rd_name)
        : ReducedDiags{std::move(rd_name)}
{
    // Initialize filename for memory usage files
    m_filename = m_rd_name;

    // Get parameters from input file
    amrex::ParmParse pp_name(m_rd_name);

    // Allow customizing the output filename
    pp_name.queryAdd("file_prefix", m_filename);

    // We don't need the WriteToFile functionality as AMReX::Arena::PrintUsageToFiles
    // handles the file output directly
    m_write_header = false;

    // Create the output directory if it doesn't exist
    if (amrex::ParallelDescriptor::IOProcessor()) {
        if (!amrex::UtilCreateDirectory(m_path, permission_flag_rwxrxrx)) {
            amrex::CreateDirectoryFailed(m_path);
        }
    }

}

// Compute diagnostics - this calls AMReX's memory usage reporting function
void MemoryPerRank::ComputeDiags (int step)
{
    // Only proceed if this step is included in our intervals
    if (!m_intervals.contains(step+1)) { return; }

    // Create a message with the current step and time (with scientific notation)
    std::stringstream time_ss;
    time_ss << std::scientific << std::setprecision(14) << WarpX::GetInstance().gett_new(0);

    std::string message = std::string("Memory usage") +
                          std::string(" at step ") + std::to_string(step+1) +
                          std::string(", time = ") + time_ss.str();

    // Call AMReX's memory usage reporting function
    // This will create one file per MPI rank in the specified directory
    amrex::Arena::PrintUsageToFiles(m_path + m_filename, message);
}
