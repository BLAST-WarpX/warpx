/* Copyright 2019-2020 Neil Zaim, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLE_UTILS_H_
#define WARPX_PARTICLE_UTILS_H_

#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXConst.H"

#include <AMReX_DenseBins.H>
#include <AMReX_Particles.H>

#include <AMReX_BaseFwd.H>

namespace ParticleUtils {

    /**
     * \brief Find the particles and count the particles that are in each cell. More specifically
     * this function returns an amrex::DenseBins object containing an offset array and a permutation
     * array which can be used to loop over all the cells in a tile and apply an algorithm to
     * particles of a given species present in each cell.
     * Note that this does *not* rearrange particle arrays.
     *
     * @param[in] lev the index of the refinement level.
     * @param[in] mfi the MultiFAB iterator.
     * @param[in] ptile the particle tile.
     */
    amrex::DenseBins<typename WarpXParticleContainer::ParticleTileType::ParticleTileDataType>
    findParticlesInEachCell (int lev,
                             amrex::MFIter const & mfi,
                             WarpXParticleContainer::ParticleTileType & ptile);

    /**
     * \brief Return (relativistic) particle energy given velocity and mass.
     * Note the use of `double` since this calculation is prone to error with
     * single precision.
     *
     * @param[in] u2 square of particle speed (i.e. u dot u where u = gamma*v)
     * @param[in] mass particle mass
     * @param[out] energy particle energy in eV
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void getEnergy ( amrex::ParticleReal const u2, double const mass,
                     double& energy )
    {
        using std::sqrt;
        using namespace amrex::literals;

        constexpr auto c2 = PhysConst::c * PhysConst::c;
        energy = mass * u2 / (sqrt(1.0_rt + u2 / c2) + 1.0_rt) / PhysConst::q_e;
    }

    /**
     * \brief Return (relativistic) collision energy assuming the target (with
     * mass M) is stationary and the projectile is approaching with the
     * the given speed and mass m. Note the use of `double` since this
     * calculation is prone to error with single precision.
     *
     * @param[in] u2 square of particle speed (i.e. u dot u where u = gamma*v)
     * @param[in] m, M mass of projectile and target, respectively
     * @param[out] gamma relativistic factor of the particle
     * @param[out] energy particle energy in eV
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void getCollisionEnergy ( amrex::ParticleReal const u2, double const m,
                              double const M, double& gamma, double& energy )
    {
        using std::sqrt;
        using namespace amrex::literals;

        constexpr auto c2 = PhysConst::c * PhysConst::c;

        gamma = sqrt(1.0_rt + u2 / c2);
        energy = (
            2.0_rt * m * M * u2 / (gamma + 1.0_rt)
            / (M + m + sqrt(m*m + M*M + 2.0_rt * m * M * gamma))
        ) / PhysConst::q_e;
    }

    /**
     * \brief Perform a Lorentz transformation of the given velocity
     * to a frame moving with velocity (Vx, Vy, Vz) relative to the present one.
     *
     * @param[in,out] ux,uy,uz components of velocity vector in the current
                      frame - importantly these quantities are gamma * velocity
     * @param[in] Vx,Vy,Vz velocity of the new frame relative to the current one,
                      NOT gamma*velocity!
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void doLorentzTransform ( amrex::ParticleReal& ux, amrex::ParticleReal& uy,
                              amrex::ParticleReal& uz,
                              amrex::ParticleReal const Vx, amrex::ParticleReal const Vy,
                              amrex::ParticleReal const Vz )
    {
        using namespace amrex::literals;

        // precompute repeatedly used quantities
        constexpr auto c2 = PhysConst::c * PhysConst::c;
        const auto V2 = (Vx*Vx + Vy*Vy + Vz*Vz);
        const auto gamma_V = 1.0_prt / std::sqrt(1.0_prt - V2 / c2);
        const auto gamma_u = std::sqrt(1.0_prt + (ux*ux + uy*uy + uz*uz) / c2);

        // copy velocity vector values
        const auto vx = ux;
        const auto vy = uy;
        const auto vz = uz;

        ux = vx * (1.0_prt + (gamma_V - 1.0_prt) * Vx*Vx/V2)
             + vy * (gamma_V - 1.0_prt) * Vx*Vy/V2
             + vz * (gamma_V - 1.0_prt) * Vx*Vz/V2
             - gamma_V * Vx * gamma_u;

        uy = vy * (1.0_prt + (gamma_V - 1.0_prt) * Vy*Vy/V2)
             + vx * (gamma_V - 1.0_prt) * Vx*Vy/V2
             + vz * (gamma_V - 1.0_prt) * Vy*Vz/V2
             - gamma_V * Vy * gamma_u;

        uz = vz * (1.0_prt + (gamma_V - 1.0_prt) * Vz*Vz/V2)
             + vx * (gamma_V - 1.0_prt) * Vx*Vz/V2
             + vy * (gamma_V - 1.0_prt) * Vy*Vz/V2
             - gamma_V * Vz * gamma_u;
    }

    /**
     * \brief Generate random unit vector in 3 dimensions
     * https://mathworld.wolfram.com/SpherePointPicking.html
     *
     * @param[out] x x-component of resulting random vector
     * @param[out] y y-component of resulting random vector
     * @param[out] z z-component of resulting random vector
     * @param[in] engine the random-engine
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void getRandomVector ( amrex::ParticleReal& x, amrex::ParticleReal& y,
                           amrex::ParticleReal& z, amrex::RandomEngine const& engine )
    {
        using std::sqrt;
        using std::cos;
        using std::sin;
        using namespace amrex::literals;

        auto const theta = amrex::Random(engine) * 2.0_prt * MathConst::pi;
        z = 2.0_prt * amrex::Random(engine) - 1.0_prt;
        auto const xy = sqrt(1_prt - z*z);
        x = xy * cos(theta);
        y = xy * sin(theta);
    }


    /** \brief Function to perform scattering of a particle that results in a
     * random velocity vector with given magnitude. This is used in isotropic
     * collision events.
     *
     * @param[in,out] ux, uy, uz colliding particle's velocity
     * @param[in] vp velocity magnitude of the colliding particle after collision.
     * @param[in] engine the random-engine
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void RandomizeVelocity ( amrex::ParticleReal& ux, amrex::ParticleReal& uy,
                             amrex::ParticleReal& uz,
                             const amrex::ParticleReal vp,
                             amrex::RandomEngine const& engine )
    {
        amrex::ParticleReal x, y, z;
        // generate random unit vector for the new velocity direction
        getRandomVector(x, y, z, engine);

        // scale new vector to have the desired magnitude
        ux = x * vp;
        uy = y * vp;
        uz = z * vp;
    }

    /* \brief Determines whether the point is within the tilebox, inclusive of the boundaries.
     *        Note that this routine is needed since tilebox.contains excludes the boundaries.
     * \param[in] tilebox The tilebox being checked
     * \param[in] point The point being checked
     * \result true if the point with within the boundary, otherwise false
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    bool containsInclusive (amrex::RealBox const& tilebox, amrex::XDim3 const point) {
        const auto *const xlo = tilebox.lo();
        const auto *const xhi = tilebox.hi();
        return  AMREX_D_TERM((xlo[0] <= point.x) && (point.x <= xhi[0]),
                          && (xlo[1] <= point.y) && (point.y <= xhi[1]),
                          && (xlo[2] <= point.z) && (point.z <= xhi[2]));
    }

}

#endif // WARPX_PARTICLE_UTILS_H_
