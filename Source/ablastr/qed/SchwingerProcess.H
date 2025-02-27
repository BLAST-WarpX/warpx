/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 * Based on PICSAR QED.
 *
 * License: BSD-3-Clause-LBNL
 */

//This file contains the implementation of the
//core functions of the Schwinger pair engine.
//Please have a look at the jupyter notebook in QED_tests/validation
//for a more in-depth discussion.
//
// References:
// 1) Schwinger. Phys. Rev. 82, 5 (1951)
// 2) Nikishov. Sov. Phys. JETP 30, 660 (1970)
// 3) Narozhny et al. Phys. LEtt. A, 330, 1-2 (2004)
// 4) Bulanov et al. Phys. Rev. Lett. 104, 220404 (2010)


#ifndef ABLASTR_QED_SCHWINGER_PROCESS
#define ABLASTR_QED_SCHWINGER_PROCESS

#include "ablastr/constant.H"
#include "ablastr/math/Vec3.H"
#include "ablastr/qed/UnitConversion.H"

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>

#include <cmath>

namespace ablastr::qed::schwinger
{

    /**
    * This function computes the Schwinger pair production rate
    * using the Nikishov formula.
    *
    * @tparam RealType the floating point type to be used
    * @tparam UnitSystem unit system to be used for inputs & outputs
    * @param[in] t_em_e a vec3<RealType> containing the electric field
    * @param[in] t_em_b a vec3<RealType> containing the magnetic field
    *
    * @return the pair production rate
    */
    template<typename RealType, unit_system UnitSystem = unit_system::SI>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType pair_production_rate(
        const math::vec3<RealType>& t_em_e,
        const math::vec3<RealType>& t_em_b) noexcept
    {
        const auto em_e = convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::E>(t_em_e);
        const auto em_b = convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::B>(t_em_b);

        constexpr auto one_over_schwinger = static_cast<RealType>(
            1.0/heaviside_lorentz_schwinger_field<double>);

        const auto ff =(math::norm_square(em_e) - math::norm_square(em_b))*RealType(0.5);
        const auto gg = math::dot(em_e, em_b);

        const auto inner = std::sqrt(ff*ff+ gg*gg);

        const auto epsi = std::sqrt(std::fabs(inner + ff))*one_over_schwinger;
        const auto eta = std::sqrt(std::fabs(inner - ff))*one_over_schwinger;

        constexpr const auto coeff = static_cast<RealType>(
            amrex::Math::powi<2>(
                heaviside_lorentz_elementary_charge<double>*
                heaviside_lorentz_schwinger_field<double>*
                2.0*ablastr::constant::math::variable_templates::pi<double>));

        constexpr auto rate_conv =
            convert_from_to<unit_system::heaviside_lorentz, UnitSystem, quantity::rate>(RealType(1.0));

        constexpr auto zero = RealType(0.0);
        constexpr auto rtp  = ablastr::constant::math::variable_templates::pi<RealType>;
        if(epsi != zero && eta != zero) AMREX_LIKELY
        {
            return coeff*rate_conv*epsi*eta*std::exp(-rtp/epsi)/std::tanh(rtp*eta/epsi);
        }
        else if(epsi == zero)
        {
            return zero;
        }
        else
        {
            return coeff*rate_conv*epsi*epsi*std::exp(-rtp/epsi)/rtp;
        }
    }

    /**
    * This function computes the Schwinger pair production rate
    * using the Nikishov formula.
    *
    * @tparam RealType the floating point type to be used
    * @tparam UnitSystem unit system to be used for inputs & outputs
    * @param[in] ex a RealType containing the x component of the electric field
    * @param[in] ey a RealType containing the y component of the electric field
    * @param[in] ez a RealType containing the z component of the electric field
    * @param[in] bx a RealType containing the x component of the magnetic field
    * @param[in] by a RealType containing the y component of the magnetic field
    * @param[in] bz a RealType containing the z component of the magnetic field
    *
    * @return the pair production rate
    */
    template<typename RealType, unit_system UnitSystem = unit_system::SI>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType pair_production_rate(
        const RealType ex, const RealType ey, const RealType ez,
        const RealType bx, const RealType by, const RealType bz) noexcept
    {
        const auto em_e = math::vec3<RealType>{ex, ey, ez};
        const auto em_b = math::vec3<RealType>{bx, by, bz};
        return pair_production_rate<RealType, UnitSystem>(em_e, em_b);
    }

    /**
    * This function computes the number of expected Schwinger pairs
    * in a given volume for a given timestep using the Nikishov formula.
    *
    * @tparam RealType the floating point type to be used
    * @tparam UnitSystem unit system to be used for inputs & outputs
    * @param[in] t_em_e a vec3<RealType> containing the electric field
    * @param[in] t_em_b a vec3<RealType> containing the magnetic field
    * @param[in] t_volume the volume of the physical region
    * @param[in] t_dt the temporal step
    *
    * @return the expected number of generated pairs
    */
    template<typename RealType, unit_system UnitSystem = unit_system::SI>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType expected_pair_number(
        const math::vec3<RealType> t_em_e,
        const math::vec3<RealType> t_em_b,
        const RealType t_volume, const RealType t_dt) noexcept
    {
        using namespace picsar::multi_physics::math;

        const auto em_e = convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::E>(t_em_e);
        const auto em_b = convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::B>(t_em_b);

        const auto schwinger_pair_production_rate = pair_production_rate<
            RealType, unit_system::heaviside_lorentz>(em_e, em_b);

        const auto dt     = convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::time>  (t_dt);
        const auto volume = convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::volume>(t_volume);

        return schwinger_pair_production_rate*dt*volume;
    }

    /**
    * This function computes the number of expected Schwinger pairs
    * in a given volume for a given timestep using the Nikishov formula.
    *
    * @tparam RealType the floating point type to be used
    * @tparam UnitSystem unit system to be used for inputs & outputs
    * @param[in] ex a RealType containing the x component of the electric field
    * @param[in] ey a RealType containing the y component of the electric field
    * @param[in] ez a RealType containing the z component of the electric field
    * @param[in] bx a RealType containing the x component of the magnetic field
    * @param[in] by a RealType containing the y component of the magnetic field
    * @param[in] bz a RealType containing the z component of the magnetic field
    * @param[in] t_volume the volume of the physical region
    * @param[in] t_dt the temporal step
    *
    * @return the expected number of generated pairs
    */
    template<typename RealType, unit_system UnitSystem>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType expected_pair_number(
        const RealType ex, const RealType ey, const RealType ez,
        const RealType bx, const RealType by, const RealType bz,
        const RealType t_volume, const RealType t_dt) noexcept
    {
        const auto em_e = math::vec3<RealType>{ex, ey, ez};
        const auto em_b = math::vec3<RealType>{bx, by, bz};
        return expected_pair_number<RealType, UnitSystem>(em_e, em_b, t_volume, t_dt);
    }

}

#endif //ABLASTR_QED_SCHWINGER_PROCESS
