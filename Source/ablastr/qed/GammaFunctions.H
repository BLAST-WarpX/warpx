/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 * Based on PICSAR QED.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_QED_GAMMA_FUNCTIONS_
#define ABLASTR_QED_GAMMA_FUNCTIONS_

#include "ablastr/math/Vec3.H"
#include "ablastr/qed/UnitConversion.H"

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

#include <cmath>

namespace ablastr::qed
{

    /**
    * This function returns the energy of a photon normalized with respect to the rest mass of an electron
    *
    * @tparam RealType the floating point type of the result
    * @tparam unit_system unit system to be used for input [default is SI]
    * @param[in] t_p a vec3<RealType> containing the momentum of the photon
    * @return the normalized energy of the photon
    */
    template<typename RealType, unit_system UnitSystem = unit_system::SI>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType compute_gamma_photon (const math::vec3<RealType>& t_p) noexcept
    {
        const auto p = from_to<UnitSystem, unit_system::heaviside_lorentz>::convert<quantity::momentum>(t_p);
        return math::norm(p)/heaviside_lorentz_electron_rest_energy<RealType>;
    }

    /**
    * This function returns the energy of a photon normalized with respect to the rest mass of an electron
    *
    * @tparam RealType the floating point type of the result
    * @tparam unit_system unit system to be used for input [default is SI]
    * @param[in] px a RealType containing the x component of the momentum of the photon
    * @param[in] py a RealType containing the y component of the momentum of the photon
    * @param[in] pz a RealType containing the z component of the momentum of the photon
    * @return the normalized energy of the photon
    */
    template<typename RealType, unit_system UnitSystem = unit_system::SI>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType compute_gamma_photon (
        const RealType px, const RealType py, const RealType pz) noexcept
    {
        const auto p = math::vec3<RealType>{px, py, pz};
        return compute_gamma_photon<RealType, UnitSystem>(p);
    }

    /**
    * This function returns the Lorentz factor for an electron or a positron
    *
    * @tparam RealType the floating point type of the result
    * @tparam unit_system unit system to be used for input [default is SI]
    * @param[in] t_p a vec3<RealType> containing the momentum of the electron (or positron)
    * @return the Lorentz factor
    */
    template<typename RealType, unit_system UnitSystem = unit_system::SI>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType compute_gamma_ele_pos (const math::vec3<RealType> t_p) noexcept
    {
        const auto p = from_to<UnitSystem, unit_system::heaviside_lorentz>::convert<quantity::momentum>(t_p);

        constexpr auto one_over_m2 = static_cast<RealType>(1.0/
            heaviside_lorentz_electron_rest_energy<double>/
            heaviside_lorentz_electron_rest_energy<double>);

        const auto gamma_2 = RealType(1.0) + math::norm_square(p)*one_over_m2;

        return std::sqrt(gamma_2);
    }

    /**
    * This function returns the Lorentz factor for an electron or a positron
    *
    * @tparam RealType the floating point type of the result
    * @tparam unit_system unit system to be used for input [default is SI]
    * @param[in] px a RealType containing the x component of the momentum of the electron (or the positron)
    * @param[in] py a RealType containing the y component of the momentum of the electron (or the positron)
    * @param[in] pz a RealType containing the z component of the momentum of the electron (or the positron)
    * @return the Lorentz factor
    */
    template<typename RealType, unit_system UnitSystem = unit_system::SI>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType compute_gamma_ele_pos (
        const RealType px, const RealType py, const RealType pz) noexcept
    {
        const auto p = math::vec3<RealType>{px, py, pz};
        return compute_gamma_ele_pos<RealType, UnitSystem>(p);
    }
}

#endif //ABLASTR_QED_GAMMA_FUNCTIONS_
