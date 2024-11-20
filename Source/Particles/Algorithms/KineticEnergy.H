/* Copyright 2021 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_PARTICLES_KINETIC_ENERGY_H_
#define WARPX_PARTICLES_KINETIC_ENERGY_H_

#include "Utils/WarpXConst.H"

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include <cmath>

namespace Algorithms{

    /**
     * \brief Computes the kinetic energy of a particle.
     * This method should not be used with photons.
     *
     * @param[in] ux x component of the particle momentum (code units)
     * @param[in] uy y component of the particle momentum (code units)
     * @param[in] uz z component of the particle momentum (code units)
     * @param[in] mass mass of the particle (in S.I. units)
     *
     * @return the kinetic energy of the particle (in S.I. units)
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    amrex::ParticleReal KineticEnergy(
        const amrex::ParticleReal ux, const amrex::ParticleReal uy, const amrex::ParticleReal uz,
        const amrex::ParticleReal mass)
    {
        using namespace amrex;

        constexpr auto inv_c2 = 1.0_prt/(PhysConst::c * PhysConst::c);

        // The expression used is derived by reducing the expression
        // (gamma - 1)*(gamma + 1)/(gamma + 1)

        const auto u2 = ux*ux + uy*uy + uz*uz;
        const auto gamma = std::sqrt(1.0_prt + u2*inv_c2);
        return 1.0_prt/(1.0_prt + gamma)*mass*u2;
    }

    /**
     * \brief Computes the kinetic energy of a photon.
     *
     * @param[in] ux x component of the particle momentum (code units)
     * @param[in] uy y component of the particle momentum (code units)
     * @param[in] uz z component of the particle momentum (code units)
     *
     * @return the kinetic energy of the photon (in S.I. units)
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    amrex::ParticleReal KineticEnergyPhotons(
        const amrex::ParticleReal ux, const amrex::ParticleReal uy, const amrex::ParticleReal uz)
    {
        // Photons have zero mass, but ux, uy and uz are calculated assuming a mass equal to the
        // electron mass. Hence, photons need a special treatment to calculate the total energy.
        constexpr auto me_c = PhysConst::m_e * PhysConst::c;

        return me_c * std::sqrt(ux*ux + uy*uy + uz*uz);
    }

}

#endif // WARPX_PARTICLES_KINETIC_ENERGY_H_
