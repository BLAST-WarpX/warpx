/* Copyright 2020 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_PUSHSELECTOR_H_
#define WARPX_PARTICLES_PUSHER_PUSHSELECTOR_H_

// Import low-level single-particle kernels
#include "Particles/Pusher/UpdateMomentumBoris.H"
#include "Particles/Pusher/UpdateMomentumBorisWithRadiationReaction.H"
#include "Particles/Pusher/UpdateMomentumHigueraCary.H"
#include "Particles/Pusher/UpdateMomentumVay.H"
#include "Particles/Pusher/UpdateMomentumBlended.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_REAL.H>

#include <limits>

/**
 * \brief Push momentum for a single particle
 *
 * \tparam do_sync                  Whether to include quantum synchrotron radiation (QSR)
 * \param ux, uy, uz                Particle momentum
 * \param Ex, Ey, Ez                Electric field on particles.
 * \param Bx, By, Bz                Magnetic field on particles.
 * \param ion_lev                   Ionization level of this particle (0 if ionization not on)
 * \param m                         Mass of this species.
 * \param a_q                       Charge of this species.
 * \param pusher_algo               0: Boris, 1: Vay, 2: HigueraCary, 3: Blended
 * \param do_crr                    Whether to do the classical radiation reaction
 * \param t_chi_max                 Cutoff chi for QSR
 * \param dt                        Time step size
 * \param gradBx, gradBy, gradBz    Magnetic field gradient (dB/dx,dB/dy,dB/dz) on particles - only used for Blended pusher.
 */

template <int do_sync>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void doParticleMomentumPush(amrex::ParticleReal& ux,
                            amrex::ParticleReal& uy,
                            amrex::ParticleReal& uz,
                            const amrex::ParticleReal Ex,
                            const amrex::ParticleReal Ey,
                            const amrex::ParticleReal Ez,
                            const amrex::ParticleReal Bx,
                            const amrex::ParticleReal By,
                            const amrex::ParticleReal Bz,
                            const amrex::ParticleReal gradBx,
                            const amrex::ParticleReal gradBy,
                            const amrex::ParticleReal gradBz,
                            const int ion_lev,
                            const amrex::ParticleReal m,
                            const amrex::ParticleReal a_q,
                            const ParticlePusherAlgo pusher_algo,
                            const int do_crr,
                            const amrex::ParticleReal alpha_gc,
                            const amrex::ParticleReal mu_gc,
#ifdef WARPX_QED
                            const amrex::Real t_chi_max,
#endif
                            const amrex::Real dt)
{
    amrex::ParticleReal qp = a_q;
    qp *= ion_lev;

    // Blended pusher not supported with QED / radiation reaction
    if (do_crr && pusher_algo == ParticlePusherAlgo::Blended) {
        amrex::Abort("Blended particle pusher is not supported with QED / "
                     "radiation reaction (do_crr=1).");
    }

    if (do_crr) {
#ifdef WARPX_QED
        amrex::ignore_unused(t_chi_max);
        if constexpr (do_sync) {
            auto chi = QedUtils::chi_ele_pos(m*ux, m*uy, m*uz,
                                            Ex, Ey, Ez,
                                            Bx, By, Bz);
            if (chi < t_chi_max) {
                UpdateMomentumBorisWithRadiationReaction(ux, uy, uz,
                                                         Ex, Ey, Ez, Bx,
                                                         By, Bz, qp, m, dt);
            }
            else {
                UpdateMomentumBoris( ux, uy, uz,
                                     Ex, Ey, Ez, Bx,
                                     By, Bz, qp, m, dt);
            }
        } else
#endif
        {

            UpdateMomentumBorisWithRadiationReaction(ux, uy, uz,
                                                     Ex, Ey, Ez, Bx,
                                                     By, Bz, qp, m, dt);
        }
    } else if (pusher_algo == ParticlePusherAlgo::Boris) {
        UpdateMomentumBoris( ux, uy, uz,
                             Ex, Ey, Ez, Bx,
                             By, Bz, qp, m, dt);
    } else if (pusher_algo == ParticlePusherAlgo::Vay) {
        UpdateMomentumVay( ux, uy, uz,
                           Ex, Ey, Ez, Bx,
                           By, Bz, qp, m, dt);
    } else if (pusher_algo == ParticlePusherAlgo::HigueraCary) {
        UpdateMomentumHigueraCary( ux, uy, uz,
                                   Ex, Ey, Ez, Bx,
                                   By, Bz, qp, m, dt);    
    } else if (pusher_algo == ParticlePusherAlgo::Blended) {
        UpdateMomentumBlended( ux, uy, uz,
                               Ex, Ey, Ez, Bx,
                               By, Bz, qp, m, dt,
                               gradBx, gradBy, gradBz,
                               alpha_gc, mu_gc);
    }     
    //else {
//        amrex::Abort("Unknown particle pusher");
//    }
}



// --------------------------------------------------------------------
// Preserve the old interface with a wrapper that sets gradBx=gradBy=gradBz=0, alpha_gc=1, mu_gc=0
// --------------------------------------------------------------------
template <int do_sync>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void doParticleMomentumPush(
    amrex::ParticleReal& ux,
    amrex::ParticleReal& uy,
    amrex::ParticleReal& uz,
    const amrex::ParticleReal Ex,
    const amrex::ParticleReal Ey,
    const amrex::ParticleReal Ez,
    const amrex::ParticleReal Bx,
    const amrex::ParticleReal By,
    const amrex::ParticleReal Bz,
    const int ion_lev,
    const amrex::ParticleReal m,
    const amrex::ParticleReal a_q,
    const ParticlePusherAlgo pusher_algo,
    const int do_crr,
#ifdef WARPX_QED
    const amrex::Real t_chi_max,
#endif
    const amrex::Real dt)
{
    // Forward with dBdx=dBdy=dBdz=0
    // and with alpha_gc=1 and mu_gc=0
    doParticleMomentumPush<do_sync>(
        ux, uy, uz,
        Ex, Ey, Ez,
        Bx, By, Bz,
        0._prt, 0._prt, 0._prt,
        ion_lev,
        m,
        a_q,
        pusher_algo,
        do_crr,
        1._prt, 0._prt,
#ifdef WARPX_QED
        t_chi_max,
#endif
        dt);
}


#endif // WARPX_PARTICLES_PUSHER_SELECTOR_H_
