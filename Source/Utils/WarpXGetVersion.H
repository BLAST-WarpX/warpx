/* Copyright 2022 Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include <string>

namespace warpx
{
    [[nodiscard]] std::string Version () noexcept; //!< Version of WarpX executable

    [[nodiscard]] std::string PicsarVersion () noexcept; //!< Version of PICSAR dependency
}
