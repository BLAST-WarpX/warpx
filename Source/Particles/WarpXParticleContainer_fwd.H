/* Copyright 2021 Luca Fedeli, Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_WarpXParticleContainer_fwd_H_
#define WARPX_WarpXParticleContainer_fwd_H_

enum struct ParticleBC;
enum struct ConvertDirection;

struct PIdx;
struct DiagIdx;
struct TmpIdx;

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

struct TmpIdx
{
    enum {
        xold = 0,
        yold, zold, uxold, uyold, uzold,
        nattribs
    };
};

#endif /* WARPX_WarpXParticleContainer_fwd_H_ */
