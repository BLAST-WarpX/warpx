/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 * Based on PICSAR QED.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_UNIT_CONVERSION_H_
#define ABLASTR_UNIT_CONVERSION_H_

#include "ablastr/constant.H"
#include "ablastr/utils/TextMsg.H"

#include <AMReX_Math.H>
#include <AMReX_REAL.H>

/**
 * It is sometimes convenient to carry out numerical calculations using
 * unit systems different from SI. It is the case, for instance, for the
 * ABLASTR QED library: internally all the calculations are performed in
 * Natural Heaviside-Lorentz units, while the interface supports the two
 * unit systems SI and Natural Heaviside Lorentz units (these unit systems
 * are precisely defined below).
 *
 * The helper functions provided by this module can be used to convert
 * physical quantities between different unit systems.
 *
 * Inside the helper functions, calculations of constexpr constants are
 * carried out in double precision.
 *
 * ## Supported unit systems:
 *
 * > SI units (International System of Units)
 *
 * > Natural Heaviside Lorentz units, a unit system frequently used in
 *   particle phyics where c (speed of light) = hbar (reduced Plank constant)
 *   = epsilon_0 (vacuum permittivity) = mu_0 (vacuum_permeability) = 1.
 *   In addition, we choose to measure the energy in MeV units
 *   (1 MeV = 1.0e6 * e0 * 1 V (Volt) = 1.60218e-13 J ).
 *   As a consequence:
 *   - mass is in MeV units (m --> mc^2 [MeV])
 *   - length is in 1/MeV units (l --> l / (hbar*c) [MeV^-1])
 *   - time is in 1/MeV units (t --> t / (hbar) [MeV^-1])
 *   - momentum is in MeV units (p --> p*c [MeV])
 *   - energy is simply expressed in MeV
 *   - velocity is normalized with respect to the speed of light
 *   - electric charge is dimensionless (q --> q*sqrt(4.0*pi*fine_structure)/e0)
 *   - electric field is in MeV^2 units (E --> e0*hbar*c*E/sqrt(4*pi*fine_structure) [MeV^2])
 *   - magnetic field is in MeV^2 (B --> e0*hbar*c^2*E/sqrt(4*pi*fine_structure) [MeV^2])
 *
 */

namespace ablastr
{

    /**
    * Supported unit systems
    */
    enum class unit_system
    {
        SI,
        heaviside_lorentz
    };

    /**
    * Reference energy used for the heaviside_lorentz units
    */
    template <typename RealType>
    constexpr RealType heaviside_lorentz_reference_energy =
        ablastr::constant::SI::variable_templates::MeV<RealType>;

    /**
    * Physical quantities
    */
    enum class quantity
    {
        mass,
        charge,
        velocity,
        momentum,
        energy,
        length,
        area,
        volume,
        time,
        rate,
        E,
        B
    };


    /**
    * Conversion factors
    */
    namespace detail::SI_to_HL_factors
    {
        template <quantity Quantity, typename T>
        constexpr T fact;

        template <typename T>
        constexpr auto fact<quantity::mass, T> = static_cast<T>(
            amrex::Math::powi<2>(ablastr::constant::SI::variable_templates::c<double>)/
            heaviside_lorentz_reference_energy<double>);

        template <typename T>
        constexpr auto fact<quantity::charge, T>  = static_cast<T>(
            ablastr::constant::SI::variable_templates::sqrt_4_pi_fine_structure<double>/
            ablastr::constant::SI::variable_templates::q_e<double>);

        template <typename T>
        constexpr auto fact<quantity::velocity, T>  = static_cast<T>(
            1.0/ablastr::constant::SI::variable_templates::c<double>);

        template <typename T>
        constexpr auto fact<quantity::momentum, T>  = static_cast<T>(
            ablastr::constant::SI::variable_templates::c<double>/
            heaviside_lorentz_reference_energy<double>);

        template <typename T>
        constexpr auto fact<quantity::energy, T>  = static_cast<T>(
            1.0/heaviside_lorentz_reference_energy<double>);

        template <typename T>
        constexpr auto fact<quantity::length, T>  = static_cast<T>(
            heaviside_lorentz_reference_energy<double>/
            (ablastr::constant::SI::variable_templates::hbar<double>*ablastr::constant::SI::variable_templates::c<double>));

        template <typename T>
        constexpr auto fact<quantity::area, T>  = static_cast<T>(
            amrex::Math::powi<2>(fact<quantity::length, double>));

        template <typename T>
        constexpr auto fact<quantity::volume, T>  = static_cast<T>(
            amrex::Math::powi<3>(fact<quantity::length, double>));

        template <typename T>
        constexpr auto fact<quantity::time, T>  = static_cast<T>(
            heaviside_lorentz_reference_energy<double>/
            ablastr::constant::SI::variable_templates::hbar<double>);

        template <typename T>
        constexpr auto fact<quantity::rate, T>  = static_cast<T>(
            1.0/fact<quantity::time, double>);

        template <typename T>
        constexpr auto fact<quantity::E, T>  = static_cast<T>(
            (
                ablastr::constant::SI::variable_templates::q_e<double>*
                ablastr::constant::SI::variable_templates::hbar<double>*
                ablastr::constant::SI::variable_templates::c<double>
            )/
            (amrex::Math::powi<2>(heaviside_lorentz_reference_energy<double>))/
            (ablastr::constant::SI::variable_templates::sqrt_4_pi_fine_structure<double>));

        template <typename T>
        constexpr auto fact<quantity::B, T>  = static_cast<T>(
            fact<quantity::E, double>*ablastr::constant::SI::variable_templates::c<double>);
    }

    /**
    * \brief This function converts the value of a physical quantity from
    * one unit system to another.
    *
    * @tparam From     the unit system from which the quantity has to be converted
    * @tparam To       the unit system to which the quantity has to be converted
    * @tparam Quantity the quantity to convert from one unit system to another
    * @tparam RealType the floating point type of the quantity to convert

    * @param[in] value the value of the quantity to convert
    * @return          the converted value of the quantity
    *
    */
    template
    <
        unit_system From,
        unit_system To,
        quantity Quantity,
        typename RealType
    >
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    [[nodiscard]]
    constexpr RealType
    convert_from_to (const RealType value) noexcept
    {
        if constexpr (
            (From == unit_system::SI) &&
            (To == unit_system::heaviside_lorentz))
        {
            return value*detail::SI_to_HL_factors::fact<Quantity, RealType>;
        }
        else if constexpr (
            (From == unit_system::heaviside_lorentz) &&
            (To == unit_system::SI))
        {
            constexpr auto fact = static_cast<RealType>(
                1.0/detail::SI_to_HL_factors::fact<Quantity, double>);
            return value*fact;
        }
        else
        {
            ABLASTR_ABORT_WITH_MESSAGE ("Unit conversion not yet iimplemented");
        }
    }
}

#endif //ABLASTR_UNIT_CONVERSION_H_
