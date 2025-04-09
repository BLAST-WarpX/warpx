/* Copyright 2022 Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "UsedInputsFile.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <ostream>
#include <string>


void
ablastr::utils::write_used_inputs_file (std::string const & filename, bool verbose)
{
    if (filename.empty() || filename == "/dev/null") {
        return;
    }

    if (verbose) {
        amrex::Print() << "For full input parameters, see the file: " << filename << "\n\n";
    }

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream jobInfoFile;
        jobInfoFile.open(filename.c_str(), std::ios::out);
        amrex::ParmParse::prettyPrintTable(jobInfoFile);
        jobInfoFile.close();
    }
}
