/* Copyright 2023
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_UTILS_TEXT_STREAMUTILS_H_
#define ABLASTR_UTILS_TEXT_STREAMUTILS_H_

#include <istream>

namespace ablastr::utils::text
{
    /**
    * \brief This function drops the rest of the current line of
    * the input stream "is"
    *
    * @param[in,out] is the input stream
    */
    void goto_next_line (std::istream& is);
}

#endif //ABLASTR_UTILS_TEXT_STREAMUTILS_H_
