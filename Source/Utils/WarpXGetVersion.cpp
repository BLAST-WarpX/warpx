/* Copyright 2025 Axel Huebl, Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Utils/WarpXGetVersion.H"

#include "Utils/WarpXVersion.H"

std::string warpx::Version () noexcept
{
    std::string version;
#ifdef WARPX_GIT_VERSION
    version = std::string(WARPX_GIT_VERSION);
#endif
    if( version.empty() ) {
        return {"Unknown"};
    } else {
        return version;
    }
}

std::string warpx::PicsarVersion () noexcept
{
    std::string version;
#ifdef PICSAR_GIT_VERSION
    version = std::string(PICSAR_GIT_VERSION);
#endif
    if( version.empty() ) {
        return {"Unknown"};
    } else {
        return version;
    }
}
