/* Copyright 2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Juliette Pech, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "OpenPMDHelpFunction.H"

#include "Utils/TextMsg.H"

#include <map>

std::string
WarpXOpenPMDFileType ()
{
    std::string openPMDFileType;
#ifdef WARPX_USE_OPENPMD
#if openPMD_HAVE_ADIOS2==1
    openPMDFileType = "bp";
#elif openPMD_HAVE_ADIOS1==1
    openPMDFileType = "bp";
#elif openPMD_HAVE_HDF5==1
    openPMDFileType = "h5";
#else
    openPMDFileType = "json";
#endif
#else
    WARPX_ABORT_WITH_MESSAGE("openPMD-api cannot be used!");
#endif // WARPX_USE_OPENPMD
    return openPMDFileType;
}

#ifdef WARPX_USE_OPENPMD
unsigned long
num_already_flushed (openPMD::ParticleSpecies & currSpecies)
{
    const auto *const scalar = openPMD::RecordComponent::SCALAR;

    unsigned long ParticleFlushOffset = 0;

    if (currSpecies.contains("id")) {
        if (currSpecies["id"].contains(scalar)) {
            if (!currSpecies["id"][scalar].empty()) {
                ParticleFlushOffset = currSpecies["id"][scalar].getExtent().at(0);
            }
        }
    }

    return ParticleFlushOffset;
}
#endif
