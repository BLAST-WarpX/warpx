/* Copyright 2021 Luca Fedeli, Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_WarpXParticleContainer_fwd_H_
#define WARPX_WarpXParticleContainer_fwd_H_

#include <AMReX_GpuAllocators.H>


enum struct ParticleBC;
enum struct ConvertDirection;

struct PIdx;
struct DiagIdx;

class WarpXParIter;

class WarpXParticleContainer;

struct DiagIdx
{
    enum {
        w = 0,
        x, y, z, ux, uy, uz,
        nattribs
    };
};

#endif /* WARPX_WarpXParticleContainer_fwd_H_ */
