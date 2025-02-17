/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_QED_PHYSICAL_QUANTITIES_H_
#define ABLASTR_QED_PHYSICAL_QUANTITIES_H_

#include "ablastr/constant.H"

#include <AMReX_Math.H>
#include <AMReX_REAL.H>

    /**
     * The interface of the ABLASTR QED library supports 2 unit systems
     * (Internally all the calculations are performed in Natural
     * Heaviside-Lorentz units):
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
     * The helper functions provided by this module are used internally by the
     * library to convert the input values of a function to Natural units and to
     * convert the output from Natural units to the desired unit system.
     *
     * Inside these helper functions, calculations of constexpr constants are
     * carried out in double precision.
     */

namespace ablastr::qed
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

    template
    <
        quantity Quantity,
        unit_system From,
        unit_system To,
        typename RealType = amrex::Real
    >
    AMREX_FORCE_INLINE
    constexpr RealType
    convert (RealType value) noexcept ;


    template
    <
        quantity Quantity,
        typename RealType = amrex::Real
    >
    AMREX_FORCE_INLINE
    constexpr RealType
    convert <
        Quantity,
        unit_system::SI,
        unit_system::SI,
        RealType> (RealType value) noexcept
    {
        return value;
    }

    template
    <
        quantity Quantity,
        typename RealType = amrex::Real
    >
    AMREX_FORCE_INLINE
    constexpr RealType
    convert <
        Quantity,
        unit_system::heaviside_lorentz,
        unit_system::heaviside_lorentz,
        RealType> (RealType value) noexcept
    {
        return value;
    }

    template
    <
        quantity Quantity,
        typename RealType = amrex::Real
    >
    AMREX_FORCE_INLINE
    constexpr RealType
    convert <
        Quantity,
        unit_system::SI,
        unit_system::heaviside_lorentz,
        RealType> (RealType value) noexcept
    {
        return value*
            detail::SI_to_HL_factors::fact<Quantity, RealType>;
    }

    template
    <
        quantity Quantity,
        typename RealType = amrex::Real
    >
    AMREX_FORCE_INLINE
    constexpr auto
    convert <
        Quantity,
        unit_system::heaviside_lorentz,
        unit_system::SI,
        RealType> (RealType value) noexcept
    {
        constexpr auto fact = static_cast<RealType>(
            1.0/detail::SI_to_HL_factors::fact<Quantity, double>);
        return value*fact;
    }
}

#endif //ABLASTR_PHYSICAL_QUANTITIES_H_
