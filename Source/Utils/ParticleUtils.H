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

    // Define shortcuts for frequently-used type names
    using ParticleTileType = typename WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = typename ParticleTileType::ParticleTileDataType;

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
    amrex::DenseBins<ParticleTileDataType>
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
     * \brief Perform a Lorentz transformation of the given velocity
     * to a frame moving with gamma*velocity (Ux, Uy, Uz) relative to the present one.
     *
     * @param[in,out] ux,uy,uz components of velocity vector in the current
                      frame - importantly these quantities are gamma * velocity
     * @param[in] Ux,Uy,Uz velocity of the new frame relative to the current one,
                      importantly these quantities are gamma * velocity
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void doLorentzTransformWithU ( amrex::ParticleReal& ux, amrex::ParticleReal& uy,
                                   amrex::ParticleReal& uz,
                                   amrex::ParticleReal const Ux, amrex::ParticleReal const Uy,
                                   amrex::ParticleReal const Uz )
    {
        using namespace amrex::literals;

        constexpr auto c2 = PhysConst::c * PhysConst::c;

        amrex::ParticleReal const Usq = Ux*Ux + Uy*Uy + Uz*Uz;
        amrex::ParticleReal const U = std::sqrt(Usq);
        if (U > 0._prt) {
            amrex::ParticleReal const gamma = std::sqrt(1._prt + Usq/c2);
            // A nice way of calculating (gamma - 1) when it is small
            amrex::ParticleReal const gammam1 = Usq/c2/(gamma + 1.0_prt);
            amrex::ParticleReal const Ux_hat = Ux/U;
            amrex::ParticleReal const Uy_hat = Uy/U;
            amrex::ParticleReal const Uz_hat = Uz/U;
            amrex::ParticleReal const P0 = std::sqrt(1._prt + (ux*ux + uy*uy + uz*uz)/c2);
            amrex::ParticleReal const udotn = ux*Ux_hat + uy*Uy_hat + uz*Uz_hat;
            ux = ux + gammam1*udotn*Ux_hat - Ux*P0;
            uy = uy + gammam1*udotn*Uy_hat - Uy*P0;
            uz = uz + gammam1*udotn*Uz_hat - Uz*P0;
        }

    }

    /**
     * \brief Perform a Lorentz transformation of the given momentum
     * to a frame moving with gamma*velocity (Ux, Uy, Uz) relative to the present one.
     *
     * @param[in,out] px,py,pz components of momentum vector in the current frame
     * @param[in] mass the particle mass
     * @param[in] Ux,Uy,Uz velocity of the new frame relative to the current one,
                      importantly these quantities are gamma * velocity
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void doLorentzTransformWithP ( amrex::ParticleReal& px, amrex::ParticleReal& py,
                                   amrex::ParticleReal& pz, amrex::ParticleReal mass,
                                   amrex::ParticleReal const Ux, amrex::ParticleReal const Uy,
                                   amrex::ParticleReal const Uz )
    {
        using namespace amrex::literals;

        constexpr auto c2 = PhysConst::c * PhysConst::c;

        amrex::ParticleReal const Usq = Ux*Ux + Uy*Uy + Uz*Uz;
        amrex::ParticleReal const U = std::sqrt(Usq);
        if (U > 0._prt) {
            amrex::ParticleReal const gamma = std::sqrt(1._prt + Usq/c2);
            // A nice way of calculating (gamma - 1) when it is small
            amrex::ParticleReal const gammam1 = Usq/c2/(gamma + 1.0_prt);
            amrex::ParticleReal const Ux_hat = Ux/U;
            amrex::ParticleReal const Uy_hat = Uy/U;
            amrex::ParticleReal const Uz_hat = Uz/U;
            amrex::ParticleReal const P0 = std::sqrt(px*px + py*py + pz*pz + mass*mass*c2);
            amrex::ParticleReal const pdotn = px*Ux_hat + py*Uy_hat + pz*Uz_hat;
            px = px + gammam1*pdotn*Ux_hat - Ux*P0/PhysConst::c;
            py = py + gammam1*pdotn*Uy_hat - Uy*P0/PhysConst::c;
            pz = pz + gammam1*pdotn*Uz_hat - Uz*P0/PhysConst::c;
        }

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


    /* \brief Add or subtract a small percent of energy from a
     *        pair of particles such that momentum is not disturbed.
     *
     * This is done by treating it as an inelastic scattering event
     * with potential U = deltaE and zero scattering angle
     * deltaE > 0 ==> need to take energy away
     * deltaE < 0 ==> need to add energy
     *
     * \param[in/out] uxp, uyp, uzp momenta of the particles
     * \param[in] w weight of the particles
     * \param[in] indices indices of particles, listed in cell order
     * \param[in] cell_start start index of particles in the cell
     * \param[in] cell_stop start index of particles in the next cell
     * \param[in] mass mass of the particles
     * \param[in] energy_fraction fraction of pair's relative energy to change
     * \param[in] energy_fraction_max max fracton of total relative energy that can be changed
     * \param[in/out] deltaE amount of energy to be distributed in this species
     */
    template <typename T_index>
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    bool ModifyEnergyPairwise (amrex::ParticleReal * const AMREX_RESTRICT uxp,
                               amrex::ParticleReal * const AMREX_RESTRICT uyp,
                               amrex::ParticleReal * const AMREX_RESTRICT uzp,
                               amrex::ParticleReal * const AMREX_RESTRICT w,
                               T_index const * const AMREX_RESTRICT indices,
                               T_index const cell_start,
                               T_index const cell_stop,
                               amrex::ParticleReal const mass,
                               amrex::ParticleReal const energy_fraction,
                               amrex::ParticleReal const energy_fraction_max,
                               amrex::ParticleReal & deltaE )
    {
        using namespace amrex::literals;

        constexpr auto c2 = PhysConst::c * PhysConst::c;

        int const numCell = (cell_stop - cell_start);

        // Adjust particle's momentum to absorb deltaE.
        // This loops over the particles, two at a time, distributing
        // the energy until all of it has been distributed or failing if not.
        // If deltaE < 0, it should never fail since energy can always be added
        // to the particles.
        // if deltaE > 0, it will sometimes fail because energy cannot always be
        // removed from the particles without changing the momentum.
        // The loop is setup to switch between even first and odd first pairs
        // each loop over the particles.
        int loop_count = 0;
        int const max_loop_count = 10;
        amrex::ParticleReal Erel_cumm = 0.0_prt;
        amrex::ParticleReal fmult_fact = 1.0_prt;
        amrex::ParticleReal energy_frac_eff = 0.0_prt;
        T_index i1 = cell_start;
        bool correction_failed = false;
        bool deltaE_is_zero = false;
        while (!deltaE_is_zero && !correction_failed && loop_count <= max_loop_count) {

            T_index i2 = i1 + 1;
            if (i2 == cell_stop) {
                // If i1 is the last particle, set the other particle, i2, to be the first particle
                i2 = cell_start;
            }

            int const sign = (deltaE < 0.0 ? -1 : 1);

            const amrex::ParticleReal ux1 = uxp[ indices[i1] ];
            const amrex::ParticleReal uy1 = uyp[ indices[i1] ];
            const amrex::ParticleReal uz1 = uzp[ indices[i1] ];
            const amrex::ParticleReal ux2 = uxp[ indices[i2] ];
            const amrex::ParticleReal uy2 = uyp[ indices[i2] ];
            const amrex::ParticleReal uz2 = uzp[ indices[i2] ];
            const amrex::ParticleReal wpmp1 = mass*w[ indices[i1] ];
            const amrex::ParticleReal wpmp2 = mass*w[ indices[i2] ];

            // For the calculations below, since it involves taking differences
            // between similar numbers, which can lose precision, to achieve machine
            // level accuracy on the energy conservation long double precision would
            // be required. However, long double is implemented inconsistently across
            // platforms, so standard precision is used here with the accordant loss
            // of precision. For practical purposes however, the loss of precision is negligible.

            const amrex::ParticleReal ux = ux1 - ux2;
            const amrex::ParticleReal uy = uy1 - uy2;
            const amrex::ParticleReal uz = uz1 - uz2;
            const amrex::ParticleReal dbetasq = (ux*ux + uy*uy + uz*uz)/c2;
            const amrex::ParticleReal gbsq1 = (ux1*ux1 + uy1*uy1 + uz1*uz1)/c2;
            const amrex::ParticleReal gbsq2 = (ux2*ux2 + uy2*uy2 + uz2*uz2)/c2;
            const amrex::ParticleReal gamma1 = std::sqrt(1.0_prt + gbsq1);
            const amrex::ParticleReal gamma2 = std::sqrt(1.0_prt + gbsq2);
            const amrex::ParticleReal E1 = wpmp1*gamma1*c2;
            const amrex::ParticleReal E2 = wpmp2*gamma2*c2;
            const amrex::ParticleReal Etot = E1 + E2;
            const amrex::ParticleReal pxtot = wpmp1*ux1 + wpmp2*ux2;
            const amrex::ParticleReal pytot = wpmp1*uy1 + wpmp2*uy2;
            const amrex::ParticleReal pztot = wpmp1*uz1 + wpmp2*uz2;
            const amrex::ParticleReal Ecm = std::sqrt(Etot*Etot - (pxtot*pxtot + pytot*pytot + pztot*pztot)*c2);
            const amrex::ParticleReal Erel = Ecm - wpmp1*c2 - wpmp2*c2;
            if (Erel <= 0.0) { continue; }

            // Set the amount of energy change for this pair based on the
            // relative energy in the center of momentum.
            // If deltaE > 0, no more that Erel can be removed from this pair.
            // If deltaE < 0, limit the energy added by Erel at first, but
            // after the first loop, add whatever is left (using the adjusted fmult_fact).
            amrex::ParticleReal deltaEpair = static_cast<amrex::ParticleReal>(sign*energy_fraction*fmult_fact)*Erel;
            if (std::abs(deltaEpair) >= std::abs(deltaE)) {
                deltaEpair = deltaE;
                deltaE = 0.0_prt;
                deltaE_is_zero = true;
            } else {
                deltaE -= deltaEpair;
            }

            const amrex::ParticleReal A = Etot - deltaEpair;
            const amrex::ParticleReal D = A*A + E2*E2 - E1*E1;
            const amrex::ParticleReal p2dotu = static_cast<amrex::ParticleReal>(wpmp2*(ux2*ux + uy2*uy + uz2*uz));
            const amrex::ParticleReal ptdotu = (pxtot*ux + pytot*uy + pztot*uz);

            // Compute coefficients for quadratic equation for alpha.
            const amrex::ParticleReal a = A*A*dbetasq - ptdotu*ptdotu;
            const amrex::ParticleReal b = D*ptdotu - 2.0_prt*A*A*p2dotu;
            const amrex::ParticleReal c = A*A*E2*E2 - D*D/4.0_prt;

            const amrex::ParticleReal root = b*b - 4.0_prt*a*c;
            if (root < 0.0 || a == 0.0) { continue; }
            const amrex::ParticleReal alpha = (-b + std::sqrt(root))/(2.0_prt*a);

            // Update particle velocities.
            const amrex::ParticleReal ratio1 = alpha/(wpmp1*c2);
            const amrex::ParticleReal ratio2 = alpha/(wpmp2*c2);
            uxp[ indices[i1] ] += ratio1*ux;
            uyp[ indices[i1] ] += ratio1*uy;
            uzp[ indices[i1] ] += ratio1*uz;
            uxp[ indices[i2] ] -= ratio2*ux;
            uyp[ indices[i2] ] -= ratio2*uy;
            uzp[ indices[i2] ] -= ratio2*uz;

            Erel_cumm += Erel - deltaEpair;

            if (i1 < cell_stop - 2) {
                // If not done with the loop, increment to the next pair of particles
                i1 = i2 + 1;
            } else {
                loop_count++;
                // On next time through the loop, use a different set of pairs.
                // What index to start with depends on whether the number of particles
                // is even or odd.
                if (i1 == cell_stop - 2) {
                    if ((numCell % 2) == 0) { i1 = cell_start + 1; }
                    if ((numCell % 2) != 0) { i1 = cell_start; }
                } else {
                    if ((numCell % 2) == 0) { i1 = cell_start; }
                    if ((numCell % 2) != 0) { i1 = cell_start + 1; }
                }
                // Compute the energy fraction to correct with one more loop.
                // This is the remaining energy correction divided by the cummulated Erel,
                // the amount of "free" energy available.
                // If deltaE < 0, this check always passes since the amount of energy that
                // can be added is unlimited.
                energy_frac_eff = std::abs(deltaE)/Erel_cumm;
                if (energy_frac_eff > energy_fraction_max || loop_count > max_loop_count) {
                    correction_failed = true;
                    break;
                } else if (deltaE < 0.) {
                    // If after the first time around, there is energy left to distribute,
                    // increase the multiplier to distribute the rest faster.
                    fmult_fact = std::max(1._prt, energy_frac_eff/energy_fraction);
                }
            }

        }

        return correction_failed;
    }
}

#endif // WARPX_PARTICLE_UTILS_H_
