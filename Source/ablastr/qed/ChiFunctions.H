/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 * Based on PICSAR QED.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_QED_CHI_FUNCTIONS_H_
#define ABLASTR_QED_CHI_FUNCTIONS_H_

#include "ablastr/constant.H"
#include "ablastr/math/Vec3.H"
#include "ablastr/qed/UnitConversion.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_Random.H>

#include <cmath>

namespace ablastr::qed
{
    /**
    * This function returns the chi parameter for a photon
    *
    * @tparam RealType the floating point type of the result
    * @tparam UnitSystem unit system to be used for input&output [default is SI]
    * @param[in] px a RealType containing the x component of the momentum of the photon
    * @param[in] py a RealType containing the y component of the momentum of the photon
    * @param[in] pz a RealType containing the z component of the momentum of the photon
    * @param[in] ex a RealType containing the x component of the electric field
    * @param[in] ey a RealType containing the y component of the electric field
    * @param[in] ez a RealType containing the z component of the electric field
    * @param[in] bx a RealType containing the x component of the magnetic field
    * @param[in] by a RealType containing the y component of the magnetic field
    * @param[in] bz a RealType containing the z component of the magnetic field
    * @return the chi parameter
    */
    template<typename RealType, unit_system UnitSystem = unit_system::SI>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType chi_photon(
        const RealType px, const RealType py, const RealType pz,
        const RealType ex, const RealType ey, const RealType ez,
        const RealType bx, const RealType by, const RealType bz) noexcept
    {
        const auto p =
            convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::momentum>(
                math::vec3<RealType>{px, py, pz});
        const auto em_e =
            convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::E>(
                math::vec3<RealType>{ex, ey, ez});
        const auto em_b =
            convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::B>(
                math::vec3<RealType>{bx, by, bz});

        const auto norm_p = math::norm(p);

        constexpr auto zero = static_cast<RealType>(0.0);
        if(norm_p == zero){
            return zero;
        }

        const auto gamma_phot = norm_p/heaviside_lorentz_electron_rest_energy_v<RealType>;

        const auto p_unit = p / norm_p;
        const auto em_eperp = em_e - math::dot(p_unit,em_e)*p_unit;
        const auto field = math::norm(em_eperp + math::cross(p_unit, em_b));

        constexpr auto one_over_schwinger = static_cast<RealType>(1.0/
            heaviside_lorentz_schwinger_field_v<double>);

        return gamma_phot*field*one_over_schwinger;
    }


    /**
    * This function returns the chi parameter for either an electron or a positron
    *
    * @tparam RealType the floating point type of the result
    * @tparam UnitSystem unit system to be used for input&output [default is SI]
    * @param[in] px a RealType containing the x component of the momentum of the electron (or the positron)
    * @param[in] py a RealType containing the y component of the momentum of the electron (or the positron)
    * @param[in] pz a RealType containing the z component of the momentum of the electron (or the positron)
    * @param[in] ex a RealType containing the x component of the electric field
    * @param[in] ey a RealType containing the y component of the electric field
    * @param[in] ez a RealType containing the z component of the electric field
    * @param[in] bx a RealType containing the x component of the magnetic field
    * @param[in] by a RealType containing the y component of the magnetic field
    * @param[in] bz a RealType containing the z component of the magnetic field
    * @return the chi parameter
    */
    template<typename RealType, unit_system UnitSystem = unit_system::SI>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType chi_ele_pos(
        const RealType px, const RealType py, const RealType pz,
        const RealType ex, const RealType ey, const RealType ez,
        const RealType bx, const RealType by, const RealType bz) noexcept
    {
        using namespace ablastr::math;

        const auto p =
            convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::momentum>(
                math::vec3<RealType>{px, py, pz});
        const auto em_e =
            convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::E>(
                math::vec3<RealType>{ex, ey, ez});
        const auto em_b =
            convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::B>(
                math::vec3<RealType>{bx, by, bz});

        const RealType norm_p = math::norm(p);

        constexpr auto zero = static_cast<RealType>(0.0);
        if(norm_p == zero){
            return zero;
        }

        constexpr auto one = static_cast<RealType>(1.0);
        const auto p_unit = p/norm_p;

        constexpr auto one_over_m2 = static_cast<RealType>(1.0/
            heaviside_lorentz_electron_rest_energy_v<double>/
            heaviside_lorentz_electron_rest_energy_v<double>);

        const auto gamma_2 = one + norm_p*norm_p*one_over_m2;
        const auto gamma   = std::sqrt(gamma_2);
        const auto beta    = std::sqrt((gamma_2-one)/gamma_2);

        const auto beta_vec     = beta * p_unit;
        const auto beta_dot_e   = math::dot(beta_vec, em_e);
        const auto beta_dot_e_2 = beta_dot_e*beta_dot_e;

        const auto e_plus_beta_cross_b_2 =
            math::norm_square(em_e + math::cross(beta_vec, em_b));

        const auto field = std::sqrt(std::fabs(beta_dot_e_2-e_plus_beta_cross_b_2));

        constexpr auto one_over_schwinger = static_cast<RealType>(1.0/
            heaviside_lorentz_schwinger_field_v<double>);

        return gamma*field*one_over_schwinger;
    }
}

#endif // ABLASTR_QED_CHI_FUNCTIONS_H_
