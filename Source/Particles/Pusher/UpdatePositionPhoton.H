/* Copyright 2019 David Grote, Luca Fedeli, Maxence Thevenet
 * Remi Lehe, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_UPDATEPOSITIONPHOTON_H_
#define WARPX_PARTICLES_PUSHER_UPDATEPOSITIONPHOTON_H_

#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>

/**
 * \brief Push the position of a photon particle over one timestep,
 *    given the value of its momenta `ux`, `uy`, `uz`
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdatePositionPhoton(
    amrex::ParticleReal& x, amrex::ParticleReal& y, amrex::ParticleReal& z,
    const amrex::ParticleReal ux, const amrex::ParticleReal uy, const amrex::ParticleReal uz,
    const amrex::Real dt )
{
    using namespace amrex::literals;

    // Compute speed of light over inverse of momentum modulus, avoiding a division by zero in the
    // case where the photon has exactly zero momentum
    const amrex::ParticleReal u_norm = std::sqrt(ux*ux + uy*uy + uz*uz);
    const amrex::ParticleReal c_over_umod = (u_norm == 0._prt) ? 0._prt: PhysConst::c/u_norm;

    // Update positions over one time step
#if !defined(WARPX_DIM_1D_Z)
    x += ux * c_over_umod * dt;
#else
    amrex::ignore_unused(x);
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    // cylindrical and spherical push particles in 3D
    y += uy * c_over_umod * dt;
#else
    amrex::ignore_unused(y);
#endif
#if !defined(WARPX_DIM_RCYLINDER)
    z += uz * c_over_umod * dt;
#else
    amrex::ignore_unused(z);
#endif
}

#endif // WARPX_PARTICLES_PUSHER_UPDATEPOSITION_H_
