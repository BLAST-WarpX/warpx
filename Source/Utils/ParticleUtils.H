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
#include <AMReX_Geometry.H>
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
     * @param[in] geom_lev the geometry of the current refinement level.
     * @param[in] mfi the MultiFAB iterator.
     * @param[in] ptile the particle tile.
     */
    amrex::DenseBins<typename WarpXParticleContainer::ParticleTileType::ParticleTileDataType>
    findParticlesInEachCell (amrex::Geometry const& geom_lev,
                             amrex::MFIter const & mfi,
                             WarpXParticleContainer::ParticleTileType & ptile);

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


    /* \brief method to add/subtract a small percent of energy from a
     *        pair of particles such that momentum is not disturbed
     *
     * This is done by treating it as an inelastic scattering event
     * with potential U = a_deltaE and zero scattering angle
     * a_deltaE > 0 ==> need to take energy away
     * a_deltaE < 0 ==> need to add energy
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void ModifyEnergyPairwise( amrex::ParticleReal& ux1, amrex::ParticleReal& uy1, amrex::ParticleReal& uz1,
                               amrex::ParticleReal& ux2, amrex::ParticleReal& uy2, amrex::ParticleReal& uz2,
                               amrex::ParticleReal const a_wpmp1,
                               amrex::ParticleReal const a_wpmp2,
                               amrex::ParticleReal const a_Erel_frac,
                               amrex::ParticleReal & a_Erel_cumm,
                               amrex::ParticleReal & a_deltaE )
    {

        using namespace amrex::literals;
        constexpr auto c2 = PhysConst::c * PhysConst::c;

        int const sign = (a_deltaE < 0.0 ? -1 : 1);

        const amrex::ParticleReal ux = ux1 - ux2;
        const amrex::ParticleReal uy = uy1 - uy2;
        const amrex::ParticleReal uz = uz1 - uz2;
        const long double usq = (ux*ux + uy*uy + uz*uz)/c2;
        const long double gbsq1 = (ux1*ux1 + uy1*uy1 + uz1*uz1)/c2;
        const long double gbsq2 = (ux2*ux2 + uy2*uy2 + uz2*uz2)/c2;
        const long double gamma1 = std::sqrt(1.0_prt + gbsq1);
        const long double gamma2 = std::sqrt(1.0_prt + gbsq2);
        const long double E1 = a_wpmp1*gamma1;
        const long double E2 = a_wpmp2*gamma2;
        const long double Etot = E1 + E2;
        const long double pxtot = a_wpmp1*ux1 + a_wpmp2*ux2;
        const long double pytot = a_wpmp1*uy1 + a_wpmp2*uy2;
        const long double pztot = a_wpmp1*uz1 + a_wpmp2*uz2;
        const long double Ecm = std::sqrt(Etot*Etot - (pxtot*pxtot + pytot*pytot + pztot*pztot)/c2);
        const long double Erel = Ecm - a_wpmp1 - a_wpmp2;
        if (Erel <= 0.0) { return; }

        long double deltaE = sign*a_Erel_frac*Erel;
        if (std::abs(deltaE) > std::abs(a_deltaE)) {
            deltaE = a_deltaE;
            a_deltaE = 0.0_prt;
        } else {
            a_deltaE -= deltaE;
        }
        a_Erel_cumm += Erel - deltaE;

        const long double A = Etot - deltaE;
        const long double D = A*A + E2*E2 - E1*E1;
        const long double p2dotu = a_wpmp2*(ux2*ux + uy2*uy + uz2*uz)/c2;
        const long double ptdotu = (pxtot*ux + pytot*uy + pztot*uz)/c2;

        // compute coefficients for quadratic equation for alpha
        const long double a = A*A*usq - ptdotu*ptdotu;
        const long double b = D*ptdotu - 2.0_prt*A*A*p2dotu;
        const long double c = A*A*E2*E2 - D*D/4.0_prt;

        const long double root = b*b - 4.0_prt*a*c;
        if (root < 0.0 || a == 0.0) { return; }
        const long double alpha = (-b + std::sqrt(root))/(2.0_prt*a);

        // update particle velocities
        const long double ratio1 = alpha/a_wpmp1;
        const long double ratio2 = alpha/a_wpmp2;
        ux1 += ratio1*ux;
        uy1 += ratio1*uy;
        uz1 += ratio1*uz;
        ux2 -= ratio2*ux;
        uy2 -= ratio2*uy;
        uz2 -= ratio2*uz;

    }
}

#endif // WARPX_PARTICLE_UTILS_H_
