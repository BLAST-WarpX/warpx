/* Copyright 2019 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_amrex_qed_chi_functions_h_
#define WARPX_amrex_qed_chi_functions_h_

/**
 * This header contains wrappers around functions provided by
 * the PICSAR QED library to calculate the 'chi' parameter
 * for photons or electrons and positrons.
 */

#include "QedWrapperCommons.H"

#include <picsar_qed/physics/chi_functions.hpp>

namespace QedUtils{
    /**
    * Function to calculate the 'chi' parameter for photons.
    * Suitable for GPU kernels.
    * @param[in] px,py,pz components of momentum (SI units)
    * @param[in] ex,ey,ez components of electric field (SI units)
    * @param[in] bx,by,bz components of magnetic field (SI units)
    * @return chi parameter
    */
    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real chi_photon(
        const amrex::ParticleReal px, const amrex::ParticleReal py,
        const amrex::ParticleReal pz, const amrex::ParticleReal ex,
        const amrex::ParticleReal ey, const amrex::ParticleReal ez,
        const amrex::ParticleReal bx, const amrex::ParticleReal by,
        const amrex::ParticleReal bz)
    {
        namespace pxr_p = picsar::multi_physics::phys;
        return pxr_p::chi_photon<amrex::ParticleReal, pxr_p::unit_system::SI>(
            px, py, pz, ex, ey, ez, bx, by, bz);
    }

    /**
    * Function to calculate the 'chi' parameter for electrons or positrons.
    * Suitable for GPU kernels.
    * @param[in] px,py,pz components of momentum (SI units)
    * @param[in] ex,ey,ez components of electric field (SI units)
    * @param[in] bx,by,bz components of magnetic field (SI units)
    * @return chi parameter
    */
    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real chi_ele_pos(
        const amrex::ParticleReal px, const amrex::ParticleReal py,
        const amrex::ParticleReal pz, const amrex::ParticleReal ex,
        const amrex::ParticleReal ey, const amrex::ParticleReal ez,
        const amrex::ParticleReal bx, const amrex::ParticleReal by,
        const amrex::ParticleReal bz)
    {
            namespace pxr_p = picsar::multi_physics::phys;
            return pxr_p::chi_ele_pos<amrex::ParticleReal, pxr_p::unit_system::SI>(
                px, py, pz, ex, ey, ez, bx, by, bz);
    }
    //_________
}

#endif //WARPX_amrex_qed_chi_functions_h_
