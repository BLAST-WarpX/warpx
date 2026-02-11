/* Copyright 2021 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLEBOUNDARYPROCESS_H_
#define WARPX_PARTICLEBOUNDARYPROCESS_H_

#include "Particles/WarpXParticleContainer.H"

#include <AMReX_Particle.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>
#include <AMReX_Random.H>

#include <cmath>


namespace ParticleBoundaryProcess {

struct NoOp {
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (const PData& /*ptd*/, int /*i*/,
                     const amrex::RealVect& /*pos*/, const amrex::RealVect& /*normal*/,
                     amrex::Real /*phi_value*/,
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {}
};

struct Absorb {
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (PData& ptd, int i,
                     const amrex::RealVect& /*pos*/, const amrex::RealVect& /*normal*/,
                     amrex::Real /*phi_value*/,
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {
        amrex::ParticleIDWrapper{ptd.m_idcpu[i]}.make_invalid();
    }
};

/**
 * \brief Reflect particles off embedded boundary surfaces.
 *
 *  Specular reflection: the position is mirrored across the EB surface
 *  using the signed distance (phi_value), and the velocity component
 *  normal to the surface is reversed.
 */
struct Reflect {
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (PData& ptd, int i,
                     const amrex::RealVect& /*pos*/, const amrex::RealVect& normal,
                     amrex::Real phi_value,
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {
        if (i == 0) {
            printf("[ParticleBoundaryProcess::Reflect] EB reflection active  "
                   "phi=%e  normal=(%e, %e)\n",
                   phi_value, double(normal[0]), double(normal[1]));
        }

        // --- Reflect position across EB surface ---
        // phi_value < 0 inside EB. The reflected position is
        // x_new = x - 2 * phi * n  (moves particle to the mirror point outside).
#if defined(WARPX_DIM_3D)
        ptd.m_rdata[PIdx::x][i] -= amrex::ParticleReal(2.0 * phi_value * normal[0]);
        ptd.m_rdata[PIdx::y][i] -= amrex::ParticleReal(2.0 * phi_value * normal[1]);
        ptd.m_rdata[PIdx::z][i] -= amrex::ParticleReal(2.0 * phi_value * normal[2]);
#elif defined(WARPX_DIM_XZ)
        ptd.m_rdata[PIdx::x][i] -= amrex::ParticleReal(2.0 * phi_value * normal[0]);
        ptd.m_rdata[PIdx::z][i] -= amrex::ParticleReal(2.0 * phi_value * normal[1]);
#elif defined(WARPX_DIM_RZ)
        ptd.m_rdata[PIdx::x][i] -= amrex::ParticleReal(2.0 * phi_value * normal[0]);
        ptd.m_rdata[PIdx::z][i] -= amrex::ParticleReal(2.0 * phi_value * normal[1]);
#elif defined(WARPX_DIM_1D_Z)
        ptd.m_rdata[PIdx::z][i] -= amrex::ParticleReal(2.0 * phi_value * normal[0]);
#endif

        // --- Reflect velocity: v_new = v - 2*(v . n_hat)*n_hat ---
#if defined(WARPX_DIM_3D)
        amrex::ParticleReal const v_dot_n =
            ptd.m_rdata[PIdx::ux][i] * amrex::ParticleReal(normal[0]) +
            ptd.m_rdata[PIdx::uy][i] * amrex::ParticleReal(normal[1]) +
            ptd.m_rdata[PIdx::uz][i] * amrex::ParticleReal(normal[2]);
        ptd.m_rdata[PIdx::ux][i] -= amrex::ParticleReal(2.0) * v_dot_n * amrex::ParticleReal(normal[0]);
        ptd.m_rdata[PIdx::uy][i] -= amrex::ParticleReal(2.0) * v_dot_n * amrex::ParticleReal(normal[1]);
        ptd.m_rdata[PIdx::uz][i] -= amrex::ParticleReal(2.0) * v_dot_n * amrex::ParticleReal(normal[2]);

#elif defined(WARPX_DIM_XZ)
        // normal is (n_x, n_z). uy is out-of-plane, unaffected.
        amrex::ParticleReal const v_dot_n =
            ptd.m_rdata[PIdx::ux][i] * amrex::ParticleReal(normal[0]) +
            ptd.m_rdata[PIdx::uz][i] * amrex::ParticleReal(normal[1]);
        ptd.m_rdata[PIdx::ux][i] -= amrex::ParticleReal(2.0) * v_dot_n * amrex::ParticleReal(normal[0]);
        ptd.m_rdata[PIdx::uz][i] -= amrex::ParticleReal(2.0) * v_dot_n * amrex::ParticleReal(normal[1]);

#elif defined(WARPX_DIM_RZ)
        // normal is (n_r, n_z). Velocity is Cartesian (ux, uy, uz).
        // Project velocity to cylindrical (v_r, v_theta, v_z), reflect
        // the (v_r, v_z) components, then project back.
        amrex::ParticleReal const theta_i = ptd.m_rdata[PIdx::theta][i];
        amrex::ParticleReal const cos_t = std::cos(theta_i);
        amrex::ParticleReal const sin_t = std::sin(theta_i);

        amrex::ParticleReal v_r =  ptd.m_rdata[PIdx::ux][i]*cos_t + ptd.m_rdata[PIdx::uy][i]*sin_t;
        amrex::ParticleReal v_t = -ptd.m_rdata[PIdx::ux][i]*sin_t + ptd.m_rdata[PIdx::uy][i]*cos_t;
        amrex::ParticleReal v_z =  ptd.m_rdata[PIdx::uz][i];

        amrex::ParticleReal const v_dot_n =
            v_r * amrex::ParticleReal(normal[0]) +
            v_z * amrex::ParticleReal(normal[1]);
        v_r -= amrex::ParticleReal(2.0) * v_dot_n * amrex::ParticleReal(normal[0]);
        v_z -= amrex::ParticleReal(2.0) * v_dot_n * amrex::ParticleReal(normal[1]);

        // Back to Cartesian
        ptd.m_rdata[PIdx::ux][i] = v_r*cos_t - v_t*sin_t;
        ptd.m_rdata[PIdx::uy][i] = v_r*sin_t + v_t*cos_t;
        ptd.m_rdata[PIdx::uz][i] = v_z;

#elif defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal const v_dot_n =
            ptd.m_rdata[PIdx::uz][i] * amrex::ParticleReal(normal[0]);
        ptd.m_rdata[PIdx::uz][i] -= amrex::ParticleReal(2.0) * v_dot_n * amrex::ParticleReal(normal[0]);
#endif
    }
};

}

#endif //WARPX_PARTICLEBOUNDARYPROCESS_H_
