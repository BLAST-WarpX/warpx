/* Copyright 2022 Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_USED_INPUTS_FILE_H
#define ABLASTR_USED_INPUTS_FILE_H

#include <string>


namespace ablastr::utils
{
    /** Write a file that record all inputs: inputs file + command line options
     *
     * This uses the same syntax as amrex::ParmParse inputs files.
     * Only the AMReX IOProcessor writes.
     *
     * @param filename the name of the text file to write
     * @param verbose print information about the file location to stdout
     */
    void
    write_used_inputs_file (std::string const & filename, bool verbose=true);
}

#endif // ABLASTR_USED_INPUTS_FILE_H
