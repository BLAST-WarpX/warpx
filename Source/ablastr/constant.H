/* Copyright 2019-2022 Andrew Myers, Luca Fedeli, Maxence Thevenet,
 *                     Weiqun Zhang, Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_CONSTANT_H_
#define ABLASTR_CONSTANT_H_

#include <AMReX_REAL.H>


/** Numerical compile-time constants */
namespace ablastr::constant
{
    /** Mathematical constants */
    namespace math
    {
        namespace constant_templates
        {
            //! ratio of a circle's circumference to its diameter (variable template)
            template<typename T>
            constexpr auto pi = static_cast<T>(3.14159265358979323846);

            //! https://tauday.com/tau-manifesto (variable template)
            template<typename T>
            constexpr auto tau = static_cast<T>(2.0 * pi<double>);

        } // namespace constant_templates

        //! ratio of a circle's circumference to its diameter
        constexpr auto pi = constant_templates::pi<amrex::Real>;

        //! https://tauday.com/tau-manifesto
        constexpr auto tau = constant_templates::tau<amrex::Real>;

    } // namespace math

    /** Physical constants
     *
     * Values are the 2022 CODATA recommended values
     * https://physics.nist.gov/cuu/Constants/index.html
     *
     * New additions here should also be considered for addition to
     * `warpx_constants` in WarpXUtil.cpp's `makeParser`, so that they're
     * available in parsing and evaluation of PICMI expressions, as well
     * as the corresponding Python definitions
     */
    namespace SI
    {
        namespace constant_templates
        {
            //! vacuum speed of light [m/s] (variable template)
            template<typename T>
            constexpr auto c = static_cast<T>(299'792'458);

            //! vacuum permittivity: dielectric permittivity of vacuum [F/m] (variable template)
            template<typename T>
            constexpr auto ep0 = static_cast<T>(8.8541878188e-12);

            //! vacuum permeability: magnetic permeability of vacuum = 4.0e-7 * pi [H/m] (variable template)
            //! Note that this is adjusted from the CODATA2022 value, 1.25663706127e-06
            //! So that the relation between mu0, ep0, and c is exact
            template<typename T>
            constexpr auto mu0 = static_cast<T>(1.2566370612685e-06);

            //! elementary charge [C] (variable template)
            template<typename T>
            constexpr auto q_e = static_cast<T>(1.602176634e-19);

            //! electron mass [kg] (variable template)
            template<typename T>
            constexpr auto m_e = static_cast<T>(9.1093837139e-31);

            //! proton mass [kg] (variable template)
            template<typename T>
            constexpr auto m_p = static_cast<T>(1.67262192595e-27);

            //! dalton: unified atomic mass unit [kg] (variable template)
            template<typename T>
            constexpr auto m_u = static_cast<T>(1.66053906892e-27);

            //! reduced Planck Constant = h / tau [J*s] (variable template)
            template<typename T>
            constexpr auto hbar = static_cast<T>(1.0545718176461565e-34);

            //! fine-structure constant = mu0/(4*pi)*q_e*q_e*c/hbar [dimensionless] (variable template)
            //! The value is calculated from the expression. This differs
            //! slightly from the CODATA2022 value 0.0072973525643.
            template<typename T>
            constexpr auto alpha = static_cast<T>(0.0072973525643330135);

            //! classical electron radius = 1./(4*pi*ep0) * q_e*q_e/(m_e*c*c) [m] (variable template)
            template<typename T>
            constexpr auto r_e = static_cast<T>(2.8179403205e-15);

            //! xi times c2 = xi*c*c. This should be usable for single precision instead of xi; very close to smallest float32 number possible (1.2e-38) (variable template)
            template<typename T>
            constexpr auto xi_c2 = static_cast<T>(1.1728865075613984e-35);

            //! Schwinger field [V/m] (variable template)
            template<typename T>
            constexpr auto schwinger_field = static_cast<T>(
                (m_e<double>*m_e<double>)*(c<double>*c<double>*c<double>)/(hbar<double>*q_e<double>));

            //! Boltzmann constant (exact) [J/K] (variable template)
            template<typename T>
            constexpr auto kb = static_cast<T>(1.380649e-23);

            //! 1 eV in [J] (variable template)
            template<typename T>
            constexpr auto eV = q_e<T>;

            //! 1 MeV in [J] (variable template)
            template<typename T>
            constexpr auto MeV = static_cast<T>(q_e<double> * 1e6);

            //! 1 eV/c in [kg*m/s] (variable template)
            template<typename T>
            constexpr auto eV_invc = static_cast<T>(eV<double> / c<double>);

            //! 1 MeV/c in [kg*m/s] (variable template)
            template<typename T>
            constexpr auto MeV_invc = static_cast<T>(MeV<double>  / c<double>);

            //! 1 eV/c^2 in [kg] (variable template)
            template<typename T>
            constexpr auto eV_invc2 = static_cast<T>(eV<double> / (c<double> * c<double>));

            //! 1 MeV/c^2 in [kg] (variable template)
            template<typename T>
            constexpr auto MeV_invc2 = static_cast<T>(MeV<double> / (c<double> * c<double>));

            //! sqrt(4*pi*alpha) is a constant used for the Heaviside Lorentz unit system (variable template)
            template<typename T>
            constexpr auto sqrt_4_pi_fine_structure = static_cast<T>(0.3028221207690299);

            //! square of the vacuum speed of light [m^2/s^2] (variable template)
            template<typename T>
            constexpr auto c2 = static_cast<T>(c<double>*c<double>);

            //! inverse of the vacuum speed of light [s/m] (variable template)
            template<typename T>
            constexpr auto invc = static_cast<T>(1.0/c<double>);

            //! inverse of the square of the vacuum speed of light [s^2/m^2] (variable template)
            template<typename T>
            constexpr auto invc2 = static_cast<T>(1.0/c2<double>);

        } // namespace constant_templates

        //! vacuum speed of light [m/s]
        constexpr auto c = constant_templates::c<amrex::Real>;

        //! vacuum permittivity: dielectric permittivity of vacuum [F/m]
        constexpr auto ep0 = constant_templates::ep0<amrex::Real>;

        //! vacuum permeability: magnetic permeability of vacuum = 4.0e-7 * pi [H/m]
        constexpr auto mu0 = constant_templates::mu0<amrex::Real>;

        //! elementary charge [C]
        constexpr auto q_e = constant_templates::q_e<amrex::Real>;

        //! electron mass [kg]
        constexpr auto m_e = constant_templates::m_e<amrex::Real>;

        //! proton mass [kg]
        constexpr auto m_p = constant_templates::m_p<amrex::Real>;

        //! dalton: unified atomic mass unit [kg]
        constexpr auto m_u = constant_templates::m_u<amrex::Real>;

        //! reduced Planck Constant = h / tau [J*s]
        constexpr auto hbar = constant_templates::hbar<amrex::Real>;

        //! fine-structure constant = mu0/(4*pi)*q_e*q_e*c/hbar [dimensionless]
        constexpr auto alpha = constant_templates::alpha<amrex::Real>;

        //! classical electron radius = 1./(4*pi*ep0) * q_e*q_e/(m_e*c*c) [m]
        constexpr auto r_e = constant_templates::r_e<amrex::Real>;

        //! xi: nonlinearity parameter of Heisenberg-Euler effective theory = (2.*alpha*alpha*ep0*ep0*hbar*hbar*hbar)/(45.*m_e*m_e*m_e*m_e*c*c*c*c*c)
        constexpr double xi = 1.3050122383827227e-52;

        //! xi times c2 = xi*c*c. This should be usable for single precision instead of xi; very close to smallest float32 number possible (1.2e-38)
        constexpr auto xi_c2 = constant_templates::xi_c2<amrex::Real>;

        //! Schwinger field [V/m] (variable template)
        template<typename T>
        constexpr auto schwinger_field = constant_templates::schwinger_field<amrex::Real>;

        //! Boltzmann constant (exact) [J/K]
        constexpr auto kb = constant_templates::kb<amrex::Real>;

        //! 1 eV in [J]
        constexpr auto eV = constant_templates::eV<amrex::Real>;

        //! 1 MeV in [J]
        constexpr auto MeV = constant_templates::MeV<amrex::Real>;

        //! 1 eV/c in [kg*m/s]
        constexpr auto eV_invc = constant_templates::eV_invc<amrex::Real>;

        //! 1 MeV/c in [kg*m/s]
        constexpr auto MeV_invc = constant_templates::MeV_invc<amrex::Real>;

        //! 1 eV/c^2 in [kg]
        constexpr auto eV_invc2 = constant_templates::eV_invc2<amrex::Real>;

        //! 1 MeV/c^2 in [kg]
        constexpr auto MeV_invc2 = constant_templates::MeV_invc2<amrex::Real>;

        //! square of the vacuum speed of light [m^2/s^2]
        constexpr auto c2 = constant_templates::c2<amrex::Real>;

        //! inverse of the vacuum speed of light [s/m] (variable template)
        constexpr auto invc = constant_templates::invc<amrex::Real>;

        //! inverse of the square of the vacuum speed of light [s^2/m^2] (variable template)
        constexpr auto invc2 = constant_templates::invc2<amrex::Real>;

    } // namespace SI

} // namespace ablastr::constant

#endif // ABLASTR_CONSTANT_H_
