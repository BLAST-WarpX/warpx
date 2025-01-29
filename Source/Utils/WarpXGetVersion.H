/* Copyright 2025 Axel Huebl, Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include <string>

#ifndef WARPX_GET_VERSION_H_
#define WARPX_GET_VERSION_H_

namespace warpx
{
    [[nodiscard]] std::string Version () noexcept; //!< Version of WarpX executable

    [[nodiscard]] std::string PicsarVersion () noexcept; //!< Version of PICSAR dependency
}

#endif //WARPX_GET_VERSION_H_
