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
        //! ratio of a circle's circumference to its diameter (variable template)
        template<typename T>
        constexpr auto pi_v = static_cast<T>(3.14159265358979323846);

        //! https://tauday.com/tau-manifesto (variable template)
        template<typename T>
        constexpr auto tau_v = static_cast<T>(2.0 * pi_v<double>);

        //! ratio of a circle's circumference to its diameter
        constexpr auto pi = pi_v<amrex::Real>;

        //! https://tauday.com/tau-manifesto
        constexpr auto tau = tau_v<amrex::Real>;

    } // namespace math

    /** Physical constants
     *
     * Values are the 2022 CODATA recommended values
     * https://physics.nist.gov/cuu/Constants/index.html
     *
     * New additions here should also be considered for addition to
     * `warpx_constants` in WarpXUtil.cpp's `makeParser`, so that they're
     * available in parsing and evaluation of PICMI expressions, as well
     * as the corresponding Python definitions.
     *
     * Note: _v suffix is used for variable templates (as in C++ 20's numbers library)
     */
    namespace SI
    {

        //! vacuum speed of light [m/s] (variable template)
        template<typename T>
        constexpr auto c_v = static_cast<T>(299'792'458);

        //! vacuum permittivity: dielectric permittivity of vacuum [F/m] (variable template)
        template<typename T>
        constexpr auto epsilon_0_v = static_cast<T>(8.8541878188e-12);

        //! vacuum permeability: magnetic permeability of vacuum = 4.0e-7 * pi [H/m] (variable template)
        //! Note that this is adjusted from the CODATA2022 value, 1.25663706127e-06
        //! So that the relation between mu0, epsilon_0, and c is exact
        template<typename T>
        constexpr auto mu0_v = static_cast<T>(1.2566370612685e-06);

        //! elementary charge [C] (variable template)
        template<typename T>
        constexpr auto q_e_v = static_cast<T>(1.602176634e-19);

        //! electron mass [kg] (variable template)
        template<typename T>
        constexpr auto m_e_v = static_cast<T>(9.1093837139e-31);

        //! proton mass [kg] (variable template)
        template<typename T>
        constexpr auto m_p_v = static_cast<T>(1.67262192595e-27);

        //! dalton: unified atomic mass unit [kg] (variable template)
        template<typename T>
        constexpr auto m_u_v = static_cast<T>(1.66053906892e-27);

        //! reduced Planck Constant = h / tau [J*s] (variable template)
        template<typename T>
        constexpr auto hbar_v = static_cast<T>(1.0545718176461565e-34);

        //! fine-structure constant = mu0/(4*pi)*q_e*q_e*c/hbar [dimensionless] (variable template)
        //! The value is calculated from the expression. This differs
        //! slightly from the CODATA2022 value 0.0072973525643.
        template<typename T>
        constexpr auto alpha_v = static_cast<T>(0.0072973525643330135);

        //! classical electron radius = 1./(4*pi*epsilon_0) * q_e*q_e/(m_e*c*c) [m] (variable template)
        template<typename T>
        constexpr auto r_e_v = static_cast<T>(2.8179403205e-15);

        //! xi times c2 = xi*c*c. This should be usable for single precision instead of xi; very close to smallest float32 number possible (1.2e-38) (variable template)
        template<typename T>
        constexpr auto xi_c2_v = static_cast<T>(1.1728865075613984e-35);

        //! Schwinger electric field [V/m] (variable template)
        template<typename T>
        constexpr auto schwinger_field_v = static_cast<T>(
            (m_e_v<double>*m_e_v<double>)*(c_v<double>*c_v<double>*c_v<double>)/(hbar_v<double>*q_e_v<double>));

        //! Boltzmann constant (exact) [J/K] (variable template)
        template<typename T>
        constexpr auto kb_v = static_cast<T>(1.380649e-23);

        //! 1 eV in [J] (variable template)
        template<typename T>
        constexpr auto eV_v = q_e_v<T>;

        //! 1 MeV in [J] (variable template)
        template<typename T>
        constexpr auto MeV_v = static_cast<T>(q_e_v<double> * 1e6);

        //! 1 eV/c in [kg*m/s] (variable template)
        template<typename T>
        constexpr auto eV_inv_c_v = static_cast<T>(eV_v<double> / c_v<double>);

        //! 1 MeV/c in [kg*m/s] (variable template)
        template<typename T>
        constexpr auto MeV_inv_c_v = static_cast<T>(MeV_v<double>  / c_v<double>);

        //! 1 eV/c^2 in [kg] (variable template)
        template<typename T>
        constexpr auto eV_inv_c2_v = static_cast<T>(eV_v<double> / (c_v<double> * c_v<double>));

        //! 1 MeV/c^2 in [kg] (variable template)
        template<typename T>
        constexpr auto MeV_inv_c2_v = static_cast<T>(MeV_v<double> / (c_v<double> * c_v<double>));

        //! sqrt(4*pi*alpha) is a constant used for the Heaviside Lorentz unit system (variable template)
        template<typename T>
        constexpr auto sqrt_4_pi_fine_structure_v = static_cast<T>(0.3028221207690299);

        //! square of the vacuum speed of light [m^2/s^2] (variable template)
        template<typename T>
        constexpr auto c2_v = static_cast<T>(c_v<double>*c_v<double>);

        //! inverse of the vacuum speed of light [s/m] (variable template)
        template<typename T>
        constexpr auto inv_c_v = static_cast<T>(1.0/c_v<double>);

        //! inverse of the square of the vacuum speed of light [s^2/m^2] (variable template)
        template<typename T>
        constexpr auto inv_c2_v = static_cast<T>(1.0/c2_v<double>);

        //! vacuum speed of light [m/s]
        constexpr auto c = c_v<amrex::Real>;

        //! vacuum permittivity: dielectric permittivity of vacuum [F/m]
        constexpr auto epsilon_0 = epsilon_0_v<amrex::Real>;

        //! vacuum permeability: magnetic permeability of vacuum = 4.0e-7 * pi [H/m]
        constexpr auto mu0 = mu0_v<amrex::Real>;

        //! elementary charge [C]
        constexpr auto q_e = q_e_v<amrex::Real>;

        //! electron mass [kg]
        constexpr auto m_e = m_e_v<amrex::Real>;

        //! proton mass [kg]
        constexpr auto m_p = m_p_v<amrex::Real>;

        //! dalton: unified atomic mass unit [kg]
        constexpr auto m_u = m_u_v<amrex::Real>;

        //! reduced Planck Constant = h / tau [J*s]
        constexpr auto hbar = hbar_v<amrex::Real>;

        //! fine-structure constant = mu0/(4*pi)*q_e*q_e*c/hbar [dimensionless]
        constexpr auto alpha = alpha_v<amrex::Real>;

        //! classical electron radius = 1./(4*pi*epsilon_0) * q_e*q_e/(m_e*c*c) [m]
        constexpr auto r_e = r_e_v<amrex::Real>;

        //! xi: nonlinearity parameter of Heisenberg-Euler effective theory = (2.*alpha**2*epsilon_0**2*hbar**3)/(45.*m_e**4*c**5)
        constexpr double xi = 1.3050122383827227e-52;

        //! xi times c2 = xi*c*c. This should be usable for single precision instead of xi; very close to smallest float32 number possible (1.2e-38)
        constexpr auto xi_c2 = xi_c2_v<amrex::Real>;

        //! Schwinger electric field [V/m]
        template<typename T>
        constexpr auto schwinger_field = schwinger_field_v<amrex::Real>;

        //! Boltzmann constant (exact) [J/K]
        constexpr auto kb = kb_v<amrex::Real>;

        //! 1 eV in [J]
        constexpr auto eV = eV_v<amrex::Real>;

        //! 1 MeV in [J]
        constexpr auto MeV = MeV_v<amrex::Real>;

        //! 1 eV/c in [kg*m/s]
        constexpr auto eV_inv_c = eV_inv_c_v<amrex::Real>;

        //! 1 MeV/c in [kg*m/s]
        constexpr auto MeV_inv_c = MeV_inv_c_v<amrex::Real>;

        //! 1 eV/c^2 in [kg]
        constexpr auto eV_inv_c2 = eV_inv_c2_v<amrex::Real>;

        //! 1 MeV/c^2 in [kg]
        constexpr auto MeV_inv_c2 = MeV_inv_c2_v<amrex::Real>;

        //! square of the vacuum speed of light [m^2/s^2]
        constexpr auto c2 = c2_v<amrex::Real>;

        //! inverse of the vacuum speed of light [s/m]
        constexpr auto inv_c = inv_c_v<amrex::Real>;

        //! inverse of the square of the vacuum speed of light [s^2/m^2]
        constexpr auto inv_c2 = inv_c2_v<amrex::Real>;

    } // namespace SI

} // namespace ablastr::constant

#endif // ABLASTR_CONSTANT_H_
