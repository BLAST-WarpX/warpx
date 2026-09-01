/* Copyright 2019-2020 Andrew Myers, Axel Huebl, David Grote
 * Jean-Luc Vay, Junmin Gu, Luca Fedeli
 * Maxence Thevenet, Remi Lehe, Revathi Jambunathan
 * Weiqun Zhang, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PIDX_H_
#define WARPX_PARTICLES_PIDX_H_

#include <initializer_list>

/** Real Particle Attributes stored in amrex::ParticleContainer's struct of array
 *
 *  x, y, z  : position components
 *  ux uy uz : momentum components, stored as proper velocity, i.e. gamma*v
 *  w        : weight
 *  theta    : azimuthal angle
 *  phi      : polar angle, relative to the x-y plane
 *  nattribs : number of compile-time attributes
 */
struct PIdx
{
#if defined(WARPX_DIM_RUNTIME)
    // Unified runtime-dimensionality layout: pure-SoA particle containers
    // require the positions to be the first AMREX_SPACEDIM real components
    // and AMReX is always built 3D, so positions are always (x, y, z).
    // Position components of collapsed dimensions are 0 and never updated.
    enum                          { x ,  y,   z,   w,   ux,   uy,   uz,            nattribs};
    static constexpr auto names = {"x", "y", "z", "w", "ux", "uy", "uz"};
#elif defined(WARPX_DIM_1D_Z)
    enum                          { z ,  w ,  ux ,  uy ,  uz,                      nattribs};
    static constexpr auto names = {"z", "w", "ux", "uy", "uz"};
#elif defined(WARPX_DIM_XZ)
    enum                          { x ,  z ,  w ,  ux ,  uy ,  uz,                 nattribs};
    static constexpr auto names = {"x", "z", "w", "ux", "uy", "uz"};
#elif defined(WARPX_DIM_RZ)
    enum                          { r ,  z ,  w ,  ux ,  uy ,  uz ,  theta,        nattribs};
    static constexpr auto names = {"r", "z", "w", "ux", "uy", "uz", "theta"};
#elif defined(WARPX_DIM_RCYLINDER)
    enum                          { r ,  w ,  ux ,  uy ,  uz ,  theta,             nattribs};
    static constexpr auto names = {"r", "w", "ux", "uy", "uz", "theta"};
#elif defined(WARPX_DIM_RSPHERE)
    enum                          { r ,  w ,  ux ,  uy ,  uz ,  theta ,  phi,      nattribs};
    static constexpr auto names = {"r", "w", "ux", "uy", "uz", "theta", "phi"};
#elif defined(WARPX_DIM_3D)
    enum                          { x ,  y,   z,   w,   ux,   uy,   uz,            nattribs};
    static constexpr auto names = {"x", "y", "z", "w", "ux", "uy", "uz"};
#else
  #error WarpX dimension is not correctly set!
#endif

    static_assert(names.size() == nattribs);
};

#endif // WARPX_PARTICLES_PIDX_H_
