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
        using namespace amrex::literals;

        //! ratio of a circle's circumference to its diameter
        static constexpr amrex::Real pi = 3.14159265358979323846_rt;

        //! https://tauday.com/tau-manifesto
        static constexpr amrex::Real tau = 2.0_rt * pi;
    } // namespace math

    /** Physical constants
     *
     * Values are the 2018 CODATA recommended values
     * https://physics.nist.gov/cuu/Constants/index.html
     *
     * New additions here should also be considered for addition to
     * `warpx_constants` in WarpXUtil.cpp's `makeParser`, so that they're
     * available in parsing and evaluation of PICMI expressions, as well
     * as the corresponding Python definitions
     */
    namespace SI
    {
        using namespace amrex::literals;

        namespace variable_templates
        {
            //! vacuum speed of light [m/s] (variable template)
            template<typename T>
            constexpr auto c = static_cast<T>(299'792'458);

            //! vacuum permittivity: dielectric permittivity of vacuum [F/m] (variable template)
            template<typename T>
            constexpr auto ep0 = static_cast<T>(8.8541878128e-12);

            //! vacuum permeability: magnetic permeability of vacuum = 4.0e-7 * pi [H/m] (variable template)
            template<typename T>
            constexpr auto mu0 = static_cast<T>(1.25663706212e-06);

            //! elementary charge [C] (variable template)
            template<typename T>
            constexpr auto q_e = static_cast<T>(1.602176634e-19);

            //! electron mass [kg] (variable template)
            template<typename T>
            constexpr auto m_e = static_cast<T>(9.1093837015e-31);

            //! proton mass [kg] (variable template)
            template<typename T>
            constexpr auto m_p = static_cast<T>(1.67262192369e-27);

            //! dalton: unified atomic mass unit [kg] (variable template)
            template<typename T>
            constexpr auto m_u = static_cast<T>(1.66053906660e-27);

            //! reduced Planck Constant = h / tau [J*s] (variable template)
            template<typename T>
            constexpr auto hbar = static_cast<T>(1.054571817e-34);

            //! fine-structure constant = mu0/(4*pi)*q_e*q_e*c/hbar [dimensionless] (variable template)
            template<typename T>
            constexpr auto alpha = static_cast<T>(0.007297352573748943);

            //! classical electron radius = 1./(4*pi*ep0) * q_e*q_e/(m_e*c*c) [m] (variable template)
            template<typename T>
            constexpr auto r_e = static_cast<T>(2.817940326204929e-15);

            //! xi times c2 = xi*c*c. This should be usable for single precision instead of xi; very close to smallest float32 number possible (1.2e-38) (variable template)
            template<typename T>
            constexpr auto xi_c2 = static_cast<T>(1.1728865132395492e-35);

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
        }

        //! vacuum speed of light [m/s]
        static constexpr auto c = variable_templates::c<amrex::Real>;
        //! vacuum permittivity: dielectric permittivity of vacuum [F/m]
        static constexpr auto ep0 = variable_templates::ep0<amrex::Real>;
        //! vacuum permeability: magnetic permeability of vacuum = 4.0e-7 * pi [H/m]
        static constexpr auto mu0 = variable_templates::mu0<amrex::Real>;
        //! elementary charge [C]
        static constexpr auto q_e = variable_templates::q_e<amrex::Real>;
        //! electron mass [kg]
        static constexpr auto m_e = variable_templates::m_e<amrex::Real>;
        //! proton mass [kg]
        static constexpr auto m_p = variable_templates::m_p<amrex::Real>;
        //! dalton: unified atomic mass unit [kg]
        static constexpr auto m_u = variable_templates::m_u<amrex::Real>;

        //! reduced Planck Constant = h / tau [J*s]
        static constexpr auto hbar = variable_templates::hbar<amrex::Real>;
        //! fine-structure constant = mu0/(4*pi)*q_e*q_e*c/hbar [dimensionless]
        static constexpr auto alpha = variable_templates::alpha<amrex::Real>;
        //! classical electron radius = 1./(4*pi*ep0) * q_e*q_e/(m_e*c*c) [m]
        static constexpr auto r_e = variable_templates::r_e<amrex::Real>;
        //! xi: nonlinearity parameter of Heisenberg-Euler effective theory = (2.*alpha*alpha*ep0*ep0*hbar*hbar*hbar)/(45.*m_e*m_e*m_e*m_e*c*c*c*c*c)
        static constexpr double xi = 1.3050122447005176e-52;
        //! xi times c2 = xi*c*c. This should be usable for single precision instead of xi; very close to smallest float32 number possible (1.2e-38)
        static constexpr auto xi_c2 = variable_templates::xi_c2<amrex::Real>;
        //! Schwinger field [V/m]
        static constexpr auto schwinger_field = variable_templates::schwinger_field<amrex::Real>;

        //! Boltzmann constant (exact) [J/K]
        static constexpr auto kb = variable_templates::kb<amrex::Real>;

        //! 1 eV in [J]
        static constexpr auto eV = variable_templates::eV<amrex::Real>;
        //! 1 MeV in [J]
        static constexpr auto MeV = variable_templates::MeV<amrex::Real>;
        //! 1 eV/c in [kg*m/s]
        static constexpr auto eV_invc = variable_templates::eV_invc<amrex::Real>;
        //! 1 MeV/c in [kg*m/s]
        static constexpr auto MeV_invc = variable_templates::MeV_invc<amrex::Real>;
        //! 1 eV/c^2 in [kg]
        static constexpr auto eV_invc2 = variable_templates::eV_invc2<amrex::Real>;
        //! 1 MeV/c^2 in [kg]
        static constexpr auto MeV_invc2 = variable_templates::MeV_invc2<amrex::Real>;
    } // namespace SI
} // namespace ablastr::constant

#endif // ABLASTR_CONSTANT_H_
