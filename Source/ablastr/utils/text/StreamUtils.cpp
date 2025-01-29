/* Copyright 2023
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "StreamUtils.H"

#include <limits>

void
ablastr::utils::text::goto_next_line (std::istream& is)
{
    is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}
