/* Copyright 2022 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ACCELERATORLATTICE_LATTICEELEMENTS_HARDEDGED_K_H_
#define WARPX_ACCELERATORLATTICE_LATTICEELEMENTS_HARDEDGED_K_H_

#include "Utils/WarpXConst.H"

#include <AMReX_REAL.H>

/**
 * \brief Calculate the residence correction, the fraction of the time step the particle
 * spends inside of the hard edge lattice element
 *
 * @param[in] z the current location of the particle
 * @param[in] zpvdt the estimated future location of the particle, z + v*dt
 * @param[in] zs the start of the lattice element
 * @param[in] ze the end of the lattice element
 * @return the fraction is returned
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::ParticleReal hard_edged_fraction(const amrex::ParticleReal z,
                                        const amrex::ParticleReal zpvdt,
                                        const amrex::ParticleReal zs,
                                        const amrex::ParticleReal ze)
{

    using namespace amrex::literals;

    // This allows vz to be positive or negative
    amrex::ParticleReal const zl = std::min(z, zpvdt);
    amrex::ParticleReal const zr = std::max(z, zpvdt);

    // Calculate the residence correction
    // frac will be 1 if the step is completely inside the lens, between 0 and 1
    // when entering or leaving the lens, and otherwise 0.
    // This accounts for the case when particles step over the element without landing in it.
    // This assumes that vzp != 0.
    amrex::ParticleReal const zl_bounded = std::min(std::max(zl, zs), ze);
    amrex::ParticleReal const zr_bounded = std::min(std::max(zr, zs), ze);
    const amrex::ParticleReal frac = (zr_bounded - zl_bounded)/(zr - zl);

    return frac;
}

#endif // WARPX_ACCELERATORLATTICE_LATTICEELEMENTS_HARDEDGED_K_H_
