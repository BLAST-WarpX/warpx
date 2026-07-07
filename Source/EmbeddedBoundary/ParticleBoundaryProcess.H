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
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {}
};

struct Absorb {
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (PData& ptd, int i,
                     const amrex::RealVect& /*pos*/, const amrex::RealVect& /*normal*/,
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {
        amrex::ParticleIDWrapper{ptd.m_idcpu[i]}.make_invalid();
    }
};

// Specular reflection of a particle off the embedded-boundary surface: the
// particle is kept, and the component of its velocity along the EB surface
// normal is reversed (u -> u - 2 (u.n) n), but only when it is moving INTO the
// conductor (u.n < 0; the signed-distance normal points into the fluid). This
// gives a conducting-for-fields / reflecting-for-particles wall, avoiding the
// charge depletion + sheath E-field that absorption creates at a dense surface.
struct Reflect {
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (PData& ptd, int i,
                     const amrex::RealVect& /*pos*/, const amrex::RealVect& normal,
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {
        using namespace amrex::literals;
        amrex::ParticleReal& ux = ptd.m_rdata[PIdx::ux][i];
        amrex::ParticleReal& uy = ptd.m_rdata[PIdx::uy][i];
        amrex::ParticleReal& uz = ptd.m_rdata[PIdx::uz][i];
#if defined(WARPX_DIM_RZ)
        // momentum (ux,uy,uz) is Cartesian; the EB normal is (n_r, n_z). Rotate
        // the normal to the particle's azimuth to build the Cartesian normal.
        amrex::ParticleReal const theta = ptd.m_rdata[PIdx::theta][i];
        amrex::Real const ct = std::cos(theta);
        amrex::Real const st = std::sin(theta);
        amrex::Real const nx = normal[0]*ct;
        amrex::Real const ny = normal[0]*st;
        amrex::Real const nz = normal[1];
#elif defined(WARPX_DIM_3D)
        amrex::Real const nx = normal[0];
        amrex::Real const ny = normal[1];
        amrex::Real const nz = normal[2];
#elif defined(WARPX_DIM_XZ)
        amrex::Real const nx = normal[0];
        amrex::Real const ny = 0._rt;
        amrex::Real const nz = normal[1];
#elif defined(WARPX_DIM_1D_Z)
        amrex::Real const nx = 0._rt;
        amrex::Real const ny = 0._rt;
        amrex::Real const nz = normal[0];
#elif defined(WARPX_DIM_RCYLINDER)
        // 1D radial: only a radial normal; rotate to the particle's azimuth.
        amrex::ParticleReal const theta = ptd.m_rdata[PIdx::theta][i];
        amrex::Real const nx = normal[0]*std::cos(theta);
        amrex::Real const ny = normal[0]*std::sin(theta);
        amrex::Real const nz = 0._rt;
#elif defined(WARPX_DIM_RSPHERE)
        // 1D radial (spherical): radial unit vector uses azimuth theta and polar phi.
        amrex::ParticleReal const theta = ptd.m_rdata[PIdx::theta][i];
        amrex::ParticleReal const phi   = ptd.m_rdata[PIdx::phi][i];
        amrex::Real const cphi = std::cos(phi);
        amrex::Real const nx = normal[0]*cphi*std::cos(theta);
        amrex::Real const ny = normal[0]*cphi*std::sin(theta);
        amrex::Real const nz = normal[0]*std::sin(phi);
#endif
        amrex::ParticleReal const vn = ux*nx + uy*ny + uz*nz;
        if (vn < 0._prt) {
            ux -= 2._prt*vn*nx;
            uy -= 2._prt*vn*ny;
            uz -= 2._prt*vn*nz;
        }
    }
};
}

#endif //WARPX_PARTICLEBOUNDARYPROCESS_H_
