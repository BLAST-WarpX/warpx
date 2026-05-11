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
#include "ablastr/warn_manager/WarnManager.H"

#include <AMReX_DenseBins.H>
#include <AMReX_Geometry.H>
#include <AMReX_Particles.H>
#include <AMReX_Math.H>

#include <AMReX_BaseFwd.H>

#include <cmath>

namespace ScatteringUtils {

    struct AnisotropicCoefficientTable
    {
        amrex::ParticleReal const* m_energies = nullptr;
        amrex::ParticleReal const* m_coefficients = nullptr;
        int m_num_energies = 0;
        int m_num_coefficients = 0;

        [[nodiscard]]
        AMREX_GPU_HOST_DEVICE AMREX_INLINE
        bool isValid () const
        {
            return (m_energies != nullptr) && (m_coefficients != nullptr)
                && (m_num_energies > 0) && (m_num_coefficients > 0);
        }

        /**
         * \brief Return a linearly interpolated coefficient for the given energy.
         *
         * When \p energy_MeV is below the minimum or above the maximum table energy,
         * the coefficient at the nearest endpoint row is returned unchanged (constant
         * extrapolation / clamping). This is a deliberate, conservative choice that
         * keeps the angular distribution physical, but it silently discards any
         * energy dependence outside the table range. A warning is emitted (once, on
         * the host) the first time an out-of-range energy is encountered so that
         * users can detect tables that do not cover their simulated energy regime.
         *
         * @param[in] energy_MeV centre-of-mass energy in MeV
         * @param[in] coefficient_index column index into the coefficient table
         * @return interpolated (or clamped) coefficient value
         */
        [[nodiscard]]
        AMREX_GPU_HOST_DEVICE AMREX_INLINE
        amrex::ParticleReal getCoefficient (
            amrex::ParticleReal const energy_MeV,
            int const coefficient_index) const
        {
            using namespace amrex::literals;

            if (energy_MeV <= m_energies[0]) {
                AMREX_IF_ON_HOST((
                    ablastr::warn_manager::WMRecordWarning(
                        "FusionAngularDistributionTable",
                        "A particle energy is below the minimum energy in the "
                        "fusion_angular_distribution_coefficients table. "
                        "Endpoint angular distribution coefficients will be used. "
                        "Verify that the table covers the simulated energy regime.",
                        ablastr::warn_manager::WarnPriority::medium);
                ))
                return m_coefficients[coefficient_index];
            }
            if (energy_MeV >= m_energies[m_num_energies - 1]) {
                AMREX_IF_ON_HOST((
                    ablastr::warn_manager::WMRecordWarning(
                        "FusionAngularDistributionTable",
                        "A particle energy is above the maximum energy in the "
                        "fusion_angular_distribution_coefficients table. "
                        "Endpoint angular distribution coefficients will be used. "
                        "Verify that the table covers the simulated energy regime.",
                        ablastr::warn_manager::WarnPriority::medium);
                ))
                return m_coefficients[(m_num_energies - 1) * m_num_coefficients
                                      + coefficient_index];
            }

            int hi = 1;
            while ((hi < m_num_energies - 1) && (energy_MeV > m_energies[hi])) {
                ++hi;
            }
            int const lo = hi - 1;

            amrex::ParticleReal const energy_lo = m_energies[lo];
            amrex::ParticleReal const energy_hi = m_energies[hi];
            amrex::ParticleReal const coeff_lo =
                m_coefficients[lo * m_num_coefficients + coefficient_index];
            amrex::ParticleReal const coeff_hi =
                m_coefficients[hi * m_num_coefficients + coefficient_index];
            amrex::ParticleReal const interp =
                (energy_MeV - energy_lo) / (energy_hi - energy_lo);

            return coeff_lo + (coeff_hi - coeff_lo) * interp;
        }
    };

    /**
     * \brief Sample the cosine of the fusion scattering angle from a coefficient
     * table, linearly interpolating the coefficients in energy.
     *
     * @param[in] a_An table of anisotropic polynomial coefficients
     * @param[in] a_energy_MeV center-of-mass kinetic energy in MeV
     * @param[in] a_U uniform random number in [0,1]
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    amrex::ParticleReal getRandomCosThetaAnisotropic (
        AnisotropicCoefficientTable const& a_An,
        amrex::ParticleReal const a_energy_MeV,
        amrex::ParticleReal const a_U)
    {
        using namespace amrex::literals;

        amrex::ParticleReal x = 0.0_prt;
        amrex::ParticleReal f, dfdx, sum_f, sum_dfdx;
        constexpr amrex::ParticleReal rel_tol = 1.0e-8_prt;
        constexpr int iter_max = 100;

        amrex::ParticleReal alpha_even = 0.0_prt;
        amrex::ParticleReal alpha_odd = 0.0_prt;
        for (int i = 0; i < a_An.m_num_coefficients; ++i) {
            auto const coeff = a_An.getCoefficient(a_energy_MeV, i);
            auto const denom = static_cast<amrex::ParticleReal>(i + 1);
            if (i % 2 == 0) {
                alpha_even += coeff / denom;
            } else {
                alpha_odd += coeff / denom;
            }
        }

        amrex::ParticleReal rel_diff = 1.0_prt;
        int iter = 0;
        while (rel_diff > rel_tol && iter < iter_max) {
            sum_f = 0.0_prt;
            sum_dfdx = 0.0_prt;

            amrex::ParticleReal x_power_f = x;
            amrex::ParticleReal x_power_dfdx = 1.0_prt;
            for (int i = 0; i < a_An.m_num_coefficients; ++i) {
                auto const coeff = a_An.getCoefficient(a_energy_MeV, i);
                sum_f += coeff / static_cast<amrex::ParticleReal>(i + 1) * x_power_f;
                sum_dfdx += coeff * x_power_dfdx;
                x_power_f *= x;
                x_power_dfdx *= x;
            }

            f = a_U - 0.5_prt * (1.0_prt + alpha_odd / alpha_even)
                + 0.5_prt * sum_f / alpha_even;
            dfdx = 0.5_prt * sum_dfdx / alpha_even;
            if (std::abs(dfdx) <= rel_tol) {
                break;
            }

            amrex::ParticleReal const delta = f / dfdx;
            x -= delta;
            x = amrex::max(-1.0_prt + rel_tol, amrex::min(1.0_prt - rel_tol, x));
            rel_diff = std::abs(delta);
            ++iter;
        }

        return x;
    }
}

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

        gamma = sqrt(1.0_rt + u2 * PhysConst::inv_c2);
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
        const auto V2 = (Vx*Vx + Vy*Vy + Vz*Vz);
        const auto gamma_V = 1.0_prt / std::sqrt(1.0_prt - V2 * PhysConst::inv_c2);
        const auto gamma_u = std::sqrt(1.0_prt + (ux*ux + uy*uy + uz*uz)* PhysConst::inv_c2);

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

        constexpr auto c2 = PhysConst::c2;

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

        constexpr auto c2 = PhysConst::c2;

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
     * \brief Generate a random angle with isotropic or anisotropic distribution,
     * following the method described in equations (27)-(31) or (36)-(42) of
     * Higginson et al., J. Comput. Phys. 388 (2019), doi.org/10.1016/j.jcp.2019.03.020.
     *
     * @param[in] engine the random engine
     * @param[in] scattering_angle_model the scattering angle distribution to use
     * @param[in] anisotropic_coefficients table of coefficients for anisotropic scattering
     * @param[in] collision_energy_MeV center-of-mass collision energy in MeV
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    amrex::ParticleReal getRandomCosTheta (
        amrex::RandomEngine const& engine,
        ScatteringAngleModel scattering_angle_model,
        ScatteringUtils::AnisotropicCoefficientTable const& anisotropic_coefficients,
        amrex::ParticleReal const collision_energy_MeV)
    {
        using namespace amrex::literals;
        amrex::ParticleReal cos_theta = 1.0_prt;
        if (scattering_angle_model == ScatteringAngleModel::Isotropic) {
            // Isotropic scattering: the angle of emission of the products in the center of mass frame is random
            cos_theta = 2.0_prt * amrex::Random(engine) - 1.0_prt;  // opposite sign of that in Higginson's paper
        }
        else if (scattering_angle_model == ScatteringAngleModel::Anisotropic) {
            cos_theta = -ScatteringUtils::getRandomCosThetaAnisotropic(
                anisotropic_coefficients, collision_energy_MeV, amrex::Random(engine));
        }
        return cos_theta;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    amrex::ParticleReal getRandomCosTheta (
        amrex::RandomEngine const& engine,
        ScatteringAngleModel scattering_angle_model)
    {
        return getRandomCosTheta(
            engine, scattering_angle_model, ScatteringUtils::AnisotropicCoefficientTable{},
            amrex::ParticleReal(0.0));
    }

    /**
     * \brief Generate a random unit vector with isotropic or anisotropic distribution,
     * following the method described in equations (27)-(31) or (36)-(42) of
     * Higginson et al., J. Comput. Phys. 388 (2019), doi.org/10.1016/j.jcp.2019.03.020.
     *
     * @param[out] x x-component of the random unit vector
     * @param[out] y y-component of the random unit vector
     * @param[out] z z-component of the random unit vector
     * @param[in] engine the random engine
     * @param[in] scattering_angle_model the scattering angle distribution to use
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void getRandomVector ( amrex::ParticleReal& x, amrex::ParticleReal& y,
                           amrex::ParticleReal& z, amrex::RandomEngine const& engine,
                           ScatteringAngleModel scattering_angle_model,
                           ScatteringUtils::AnisotropicCoefficientTable const&
                               anisotropic_coefficients,
                           amrex::ParticleReal const collision_energy_MeV)
    {
        using namespace amrex::literals;

        auto const phi = 2.0_prt * MathConst::pi * amrex::Random(engine);
        auto const cos_theta = getRandomCosTheta(
            engine, scattering_angle_model, anisotropic_coefficients, collision_energy_MeV);
        auto const sin_theta = std::sqrt(1.0_prt - cos_theta*cos_theta);
        x = sin_theta * std::cos(phi);
        y = sin_theta * std::sin(phi);
        z = cos_theta;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void getRandomVector ( amrex::ParticleReal& x, amrex::ParticleReal& y,
                           amrex::ParticleReal& z, amrex::RandomEngine const& engine,
                           ScatteringAngleModel scattering_angle_model)
    {
        getRandomVector(
            x, y, z, engine, scattering_angle_model,
            ScatteringUtils::AnisotropicCoefficientTable{}, amrex::ParticleReal(0.0));
    }


    /** \brief Function to perform scattering of a particle that results in a
     * random velocity vector with given magnitude. This is used in isotropic
     * collision events.
     *
     * @param[in,out] ux, uy, uz colliding particle's velocity
     * @param[in] vp velocity magnitude of the colliding particle after collision.
     * @param[in] engine the random-engine
     * @param[in] scattering_angle_model the scattering angle distribution to use
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void RandomizeVelocity ( amrex::ParticleReal& ux, amrex::ParticleReal& uy,
                             amrex::ParticleReal& uz,
                             const amrex::ParticleReal vp,
                             amrex::RandomEngine const& engine,
                             ScatteringAngleModel scattering_angle_model,
                             ScatteringUtils::AnisotropicCoefficientTable const&
                                 anisotropic_coefficients,
                             amrex::ParticleReal const collision_energy_MeV)
    {
        amrex::ParticleReal x, y, z;
        // generate random unit vector for the new velocity direction
        getRandomVector(
            x, y, z, engine, scattering_angle_model, anisotropic_coefficients,
            collision_energy_MeV);

        // scale new vector to have the desired magnitude
        ux = x * vp;
        uy = y * vp;
        uz = z * vp;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void RandomizeVelocity ( amrex::ParticleReal& ux, amrex::ParticleReal& uy,
                             amrex::ParticleReal& uz,
                             const amrex::ParticleReal vp,
                             amrex::RandomEngine const& engine,
                             ScatteringAngleModel scattering_angle_model)
    {
        RandomizeVelocity(
            ux, uy, uz, vp, engine, scattering_angle_model,
            ScatteringUtils::AnisotropicCoefficientTable{}, amrex::ParticleReal(0.0));
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

    /* \brief Crops the position at the specified boundary if do_cropping is true, 1D version
     * \param[in] x0 The initial particle position
     * \param[in/out] x1 The final particle position, with the returned value being cropped
     * \param[in] xmin The lower boundary of the domain in grid units
     * \param[in] xmax The upper boundary of the domain in grid units
     * \param[in] do_cropping Whether to crop at each of the boundaries
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void crop_at_boundary(double & x1,
                          double xmin, double xmax, amrex::GpuArray<bool,2> const& do_cropping) {
        if (do_cropping[0] && x1 < xmin) {
            x1 = xmin;
        }
        else if (do_cropping[1] && x1 > xmax) {
            x1 = xmax;
        }
    }

    /* \brief Crops the position at the specified boundary if do_cropping is true, 2D version
     * \param[in] x0, z0 The initial particle position
     * \param[in/out] x1, z1 The final particle position, with the returned value being cropped
     * \param[in] xmin The lower boundary of the domain in grid units
     * \param[in] xmax The upper boundary of the domain in grid units
     * \param[in] do_cropping Whether to crop at each of the boundaries
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void crop_at_boundary(double x0, double z0,
                          double & x1, double & z1,
                          double xmin, double xmax, amrex::GpuArray<bool,2> const& do_cropping) {
        if (do_cropping[0] && x1 < xmin) {
            double const fx = ((x0 - x1) != 0. ? (x0 - xmin)/(x0 - x1) : 0.);
            x1 = xmin;
            z1 = z0 + fx*(z1 - z0);
        }
        else if (do_cropping[1] && x1 > xmax) {
            double const fx = ((x0 - x1) != 0. ? (xmax - x0)/(x1 - x0) : 0.);
            x1 = xmax;
            z1 = z0 + fx*(z1 - z0);
        }
    }

    /* \brief Crops the position at the specified boundary if do_cropping is true, 3D version
     * \param[in] x0, y0, z0 The initial particle position
     * \param[in/out] x1, y1, z1 The final particle position, with the returned value being cropped
     * \param[in] xmin The lower boundary of the domain in grid units
     * \param[in] xmax The upper boundary of the domain in grid units
     * \param[in] do_cropping Whether to crop at each of the boundaries
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void crop_at_boundary(double x0, double y0, double z0,
                          double & x1, double & y1, double & z1,
                          double xmin, double xmax, amrex::GpuArray<bool,2> const& do_cropping) {
        if (do_cropping[0] && x1 < xmin) {
            double const fx = ((x0 - x1) != 0. ? (x0 - xmin)/(x0 - x1) : 0.);
            x1 = xmin;
            y1 = y0 + fx*(y1 - y0);
            z1 = z0 + fx*(z1 - z0);
        }
        else if (do_cropping[1] && x1 > xmax) {
            double const fx = ((x0 - x1) != 0. ? (xmax - x0)/(x1 - x0) : 0.);
            x1 = xmax;
            y1 = y0 + fx*(y1 - y0);
            z1 = z0 + fx*(z1 - z0);
        }
    }

    /* \brief Check if the particle position is out of bounds at a boundary
     *        where orbit cropping is being performed.
     * \param[in] xp, yp, zp      The particle position
     * \param[in] dinv            3D cell size inverse
     * \param[in] xyzmin          Physical lower bounds of domain
     * \param[in] domain_double   The domain boundary indicies cast to double values
     * \param[in] do_cropping     Whether to crop particle orbits at domain boundaries
     * return whether position is out of bounds
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool is_out_of_bounds ([[maybe_unused]] double xp,
                           [[maybe_unused]] double yp,
                           [[maybe_unused]] double zp,
                           const amrex::XDim3 & dinv,
                           const amrex::XDim3 & xyzmin,
                           const amrex::GpuArray<amrex::GpuArray<double,2>, AMREX_SPACEDIM> & domain_double,
                           const amrex::GpuArray<amrex::GpuArray<bool,2>, AMREX_SPACEDIM> & do_cropping)
    {

        // Store logical particle position (in grid units) in temporary array
        amrex::GpuArray<double,AMREX_SPACEDIM> xp_grid;
#if defined(WARPX_DIM_3D)
        xp_grid[0] = (xp - xyzmin.x)*dinv.x;
        xp_grid[1] = (yp - xyzmin.y)*dinv.y;
        xp_grid[2] = (zp - xyzmin.z)*dinv.z;
#elif defined(WARPX_DIM_XZ)
        xp_grid[0] = (xp - xyzmin.x)*dinv.x;
        xp_grid[1] = (zp - xyzmin.z)*dinv.z;
#elif defined(WARPX_DIM_1D_Z)
        xp_grid[0] = (zp - xyzmin.z)*dinv.z;
#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
        const double rp = std::sqrt(xp*xp + yp*yp);
        xp_grid[0] = (rp - xyzmin.x)*dinv.x;
#if defined(WARPX_DIM_RZ)
        xp_grid[1] = (zp - xyzmin.z)*dinv.z;
#endif
#elif defined(WARPX_DIM_RSPHERE)
        const double rp = std::sqrt(xp*xp + yp*yp + zp*zp);
        xp_grid[0] = (rp - xyzmin.x)*dinv.x;
#endif

        // Check if particle's position is out of bounds
        for (int dim = 0; dim < AMREX_SPACEDIM; dim++) {
            if ( (do_cropping[dim][0] && xp_grid[dim] <= domain_double[dim][0]) ||
                 (do_cropping[dim][1] && xp_grid[dim] >= domain_double[dim][1])) {
                return true;
            }
        }
        return false;

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
     * \param[in/out] deltaE amount of energy to be distributed in this species
     */
    template <typename T_index>
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    bool ModifyEnergyPairwise (amrex::ParticleReal * const AMREX_RESTRICT uxp, //NOLINT(readability-non-const-parameter)
                               amrex::ParticleReal * const AMREX_RESTRICT uyp, //NOLINT(readability-non-const-parameter)
                               amrex::ParticleReal * const AMREX_RESTRICT uzp, //NOLINT(readability-non-const-parameter)
                               const amrex::ParticleReal * const AMREX_RESTRICT w,
                               T_index const * const AMREX_RESTRICT indices,
                               T_index const cell_start,
                               T_index const cell_stop,
                               amrex::ParticleReal const mass,
                               amrex::ParticleReal const energy_fraction,
                               amrex::ParticleReal & deltaE )
    {
        using namespace amrex::literals;

        constexpr auto c2 = PhysConst::c2;

        int const numCell = (cell_stop - cell_start);

        if (numCell < 2) {
            return true;
        }

        // Adjust particle's momentum to absorb deltaE.
        // This loops over the particles, two at a time, distributing
        // the energy until all of it has been distributed or failing if not.
        // The loop is setup to switch between even first and odd first pairs
        // each loop over the particles.
        int loop_count = 0;
        int const max_loop_count = 10;
        int pair_offset = 0;
        T_index i1 = cell_start + pair_offset;
        bool correction_failed = false;
        bool deltaE_is_zero = false;
        while (!deltaE_is_zero) {

            T_index i2 = i1 + 1;
            if (i2 == cell_stop) {
                // If i1 is the last particle, set the other particle, i2, to be the first particle
                i2 = cell_start;
            }

            int const sign = (deltaE < 0.0_prt ? -1 : 1);

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

            if (Erel > 0.0_prt) {

                // Set the amount of energy change for this pair based on the
                // relative energy in the center of momentum.
                bool deltaE_finishes = false;
                amrex::ParticleReal deltaEpair = static_cast<amrex::ParticleReal>(sign*energy_fraction)*Erel;
                if (std::abs(deltaEpair) >= std::abs(deltaE)) {
                    deltaEpair = deltaE;
                    deltaE_finishes = true;
                }

                const amrex::ParticleReal A = Etot - deltaEpair;
                const amrex::ParticleReal D = A*A + E2*E2 - E1*E1;
                const amrex::ParticleReal p2dotu = wpmp2*(ux2*ux + uy2*uy + uz2*uz);
                const amrex::ParticleReal ptdotu = pxtot*ux + pytot*uy + pztot*uz;

                // Compute coefficients for quadratic equation for alpha.
                const amrex::ParticleReal a = A*A*dbetasq - ptdotu*ptdotu;
                const amrex::ParticleReal b = D*ptdotu - 2.0_prt*A*A*p2dotu;
                const amrex::ParticleReal c = A*A*E2*E2 - D*D/4.0_prt;

                const amrex::ParticleReal root = b*b - 4.0_prt*a*c;
                if (root >= 0.0_prt && amrex::Math::abs(a) > 0.0_prt) {

                    // Update particle velocities.
                    const amrex::ParticleReal alpha = (-b + std::sqrt(root))/(2.0_prt*a);
                    const amrex::ParticleReal ratio1 = alpha/(wpmp1*c2);
                    const amrex::ParticleReal ratio2 = alpha/(wpmp2*c2);
                    uxp[ indices[i1] ] += ratio1*ux;
                    uyp[ indices[i1] ] += ratio1*uy;
                    uzp[ indices[i1] ] += ratio1*uz;
                    uxp[ indices[i2] ] -= ratio2*ux;
                    uyp[ indices[i2] ] -= ratio2*uy;
                    uzp[ indices[i2] ] -= ratio2*uz;

                    // Update energy violation.
                    if (deltaE_finishes) {
                        deltaE = 0.0_prt;
                        deltaE_is_zero = true;
                    }
                    else {
                        deltaE -= deltaEpair;
                    }

                }

            }

            // Update particle 1 index.
            if (i1 < cell_stop - 2) {
                i1 = i2 + 1;
            } else {
                loop_count++;
                if (loop_count >= max_loop_count) {
                    correction_failed = true;
                    break;
                }

                pair_offset = 1 - pair_offset; // 0, 1, 0, 1, ...
                i1 = cell_start + pair_offset;
            }
        }

        return correction_failed;
    }

    /**
     * \brief A GPU compartible sort of a index vector vector
     * based on another GPU vector's values, sorting to decreasing order.
     * The heap-sort functions below were obtained from
     * https://www.geeksforgeeks.org/iterative-heap-sort/ and
     * modified for the current purpose. It achieves the same as
     * ```
     * std::sort(
     *    sorted_indices_data + cell_start, sorted_indices_data + cell_stop,
     *    [&momentum_bin_number_data](size_t i1, size_t i2) {
     *        return momentum_bin_number_data[i1] > momentum_bin_number_data[i2];
     *    }
     * );
     * ```
     * but with support for device execution.
    */
    struct HeapSortDecreasing {

        template <typename T_index>
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void operator() (T_index index_array[], const amrex::ParticleReal bin_array[], const T_index start, const T_index n) const
        {
            // sort index_array into a max heap structure
            for (T_index i = 1; i < n; i++)
            {
                auto j = i;
                // move child through heap if it is less than its parent
                while (j > 0 && bin_array[index_array[j+start]] < bin_array[index_array[(j - 1)/2 + start]]) {
                    // swap child and parent until branch is properly ordered
                    amrex::Swap(index_array[j+start], index_array[(j - 1)/2 + start]);
                    j = (j - 1) / 2;
                }
            }

            for (T_index i = n - 1; i > 0; i--)
            {
                // swap value of first (now the smallest value) to the new end point
                amrex::Swap(index_array[start], index_array[i+start]);

                // remake the max heap
                T_index j = 0, index;
                while (j < i) {
                    index = 2 * j + 1;

                    // if left child is larger than right child, point index variable to right child
                    if (index + 1 < i && bin_array[index_array[index+start]] > bin_array[index_array[index+1+start]]) {
                        index++;
                    }
                    // if parent is larger than child, swap parent with child having lower value
                    if (index < i && bin_array[index_array[j+start]] > bin_array[index_array[index+start]]) {
                        amrex::Swap(index_array[j+start], index_array[index+start]);
                    }
                    j = index;
                }
            }
        }
    };
}

#endif // WARPX_PARTICLE_UTILS_H_
