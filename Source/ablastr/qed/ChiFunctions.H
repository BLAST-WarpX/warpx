/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 * Based on PICSAR QED.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_QED_CHI_FUNCTIONS_
#define ABLASTR_QED_CHI_FUNCTIONS_

#include "ablastr/math/Vec3.H"
#include "ablastr/qed/GammaFunctions.H"
#include "ablastr/qed/UnitConversion.H"

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

#include <cmath>

namespace ablastr::qed
{

    /**
    * This function returns the chi parameter for a photon
    *
    * @tparam RealType the floating point type of the result
    * @tparam unit_system unit system to be used for input&output [default is SI]
    * @param[in] t_p a vec3<RealType> containing the momentum of the photon
    * @param[in] t_em_e a vec3<RealType> containing the electric field
    * @param[in] t_em_b a vec3<RealType> containing the magnetic field
    * @return the chi parameter
    */
    template<typename RealType, unit_system UnitSystem = unit_system::SI>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType chi_photon(
        const math::vec3<RealType>& t_p,
        const math::vec3<RealType>& t_em_e,
        const math::vec3<RealType>& t_em_b) noexcept
    {
        const auto p    = convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::momentum>(t_p);
        const auto em_e = convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::E>       (t_em_e);
        const auto em_b = convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::B>       (t_em_b);

        const auto norm_p = math::norm(p);

        if(norm_p == RealType(0.0)) AMREX_UNLIKELY
        {
            return RealType(0.0);
        }

        const auto gamma_phot =
            compute_gamma_photon<RealType, unit_system::heaviside_lorentz>(p);

        const auto p_unit = p / norm_p;
        const auto em_eperp = em_e - math::dot(p_unit,em_e)*p_unit;
        const auto field = math::norm(em_eperp + math::cross(p_unit, em_b));

        constexpr auto one_over_schwinger = static_cast<RealType>(1.0/
            heaviside_lorentz_schwinger_field<double>);

        return gamma_phot*field*one_over_schwinger;
    }

    /**
    * This function returns the chi parameter for a photon
    *
    * @tparam RealType the floating point type of the result
    * @tparam unit_system unit system to be used for input&output [default is SI]
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
    template<typename RealType, unit_system UnitSystem>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType chi_photon(
        const RealType px, const RealType py, const RealType pz,
        const RealType ex, const RealType ey, const RealType ez,
        const RealType bx, const RealType by, const RealType bz) noexcept
    {
        const auto p = math::vec3<RealType>{px, py, pz};
        const auto em_e = math::vec3<RealType>{ex, ey, ez};
        const auto em_b = math::vec3<RealType>{bx, by, bz};
        return chi_photon<RealType, UnitSystem>(p, em_e, em_b);
    }

    /**
    * This function returns the chi parameter for either an electron or a positron
    *
    * @tparam RealType the floating point type of the result
    * @tparam unit_system unit system to be used for input&output [default is SI]
    * @param[in] t_p a vec3<RealType> containing the momentum of the electron or the positrion
    * @param[in] t_em_e a vec3<RealType> containing the electric field
    * @param[in] t_em_b a vec3<RealType> containing the magnetic field
    * @return the chi parameter
    */
    template<typename RealType, unit_system UnitSystem>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType chi_ele_pos(
        const math::vec3<RealType>& t_p,
        const math::vec3<RealType>& t_em_e,
        const math::vec3<RealType>& t_em_b,
        const RealType reference_quantity = math::one<RealType>) noexcept
    {
        const auto p    = convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::momentum>(t_p);
        const auto em_e = convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::E>       (t_em_e);
        const auto em_b = convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::B>       (t_em_b);

        const auto norm_p = norm(p);

        if(norm_p == RealType(0.0)) AMREX_UNLIKELY
        {
            return RealType(0.0);
        }

        const auto p_unit = p / norm_p;

        const auto gamma =
            compute_gamma_ele_pos<RealType, unit_system::heaviside_lorentz>(p);

        const auto gamma_2 = gamma*gamma;

        const auto beta = std::sqrt((gamma_2-RealType(1.0))/gamma_2);
        const auto beta_vec = beta * p_unit;

        const auto beta_dot_e = math::dot(beta_vec, em_e);
        const auto beta_dot_e_2 = beta_dot_e*beta_dot_e;
        const auto e_plus_beta_cross_b_2 =
            math::norm_square(em_e + math::cross(beta_vec, em_b));
        const auto field = std::sqrt(std::fabs(beta_dot_e_2-e_plus_beta_cross_b_2));

        constexpr auto one_over_schwinger = static_cast<RealType>(1.0/
            heaviside_lorentz_schwinger_field<double>);

        return gamma*field*one_over_schwinger;
    }


    /**
    * This function returns the chi parameter for either an electron or a positron
    *
    * @tparam RealType the floating point type of the result
    * @tparam unit_system unit system to be used for input&output [default is SI]
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
    template<typename RealType, unit_system UnitSystem>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType chi_ele_pos(
        const RealType px, const RealType py, const RealType pz,
        const RealType ex, const RealType ey, const RealType ez,
        const RealType bx, const RealType by, const RealType bz) noexcept
    {
        const auto p = math::vec3<RealType>{px, py, pz};
        const auto em_e = math::vec3<RealType>{ex, ey, ez};
        const auto em_b = math::vec3<RealType>{bx, by, bz};
        return chi_ele_pos<RealType, UnitSystem>(p, em_e, em_b);
    }
}

#endif //ABLASTR_QED_CHI_FUNCTIONS_
