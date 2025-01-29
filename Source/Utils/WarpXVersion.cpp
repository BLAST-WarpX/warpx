/* Copyright 2022 Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"
#include "Utils/WarpXVersion.H"

#include <string>


std::string
WarpX::Version ()
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

std::string
WarpX::PicsarVersion ()
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
