/* Copyright 2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Juliette Pech, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_OPENPMDHELPFUNCTION_H_
#define WARPX_OPENPMDHELPFUNCTION_H_

#ifdef WARPX_USE_OPENPMD
#   include <openPMD/openPMD.hpp>
#endif
#include <string>


/** Determine the preferred file ending if unspecified
 *
 * @return file ending without the "."
 */
std::string
WarpXOpenPMDFileType ();

#ifdef WARPX_USE_OPENPMD
/** Determine how many particles were already written in this species and step
 *
 * This checks for a particle species the current size of the id attribute, if it exists,
 * and if it does it takes its extent as the number of particles already on disk.
 *
 * Note that this checks declared size, not necessarily written size.
 *
 * @return exisitng extent of the "id" attribute or zero.
 */
unsigned long
num_already_flushed (openPMD::ParticleSpecies & currSpecies);
#endif

#endif  // WARPX_OPENPMDHELPFUNCTION_H_
