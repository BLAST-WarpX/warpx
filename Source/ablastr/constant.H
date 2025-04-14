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
        using namespace amrex::literals;

        //! vacuum speed of light [m/s]
        static constexpr auto c = 299'792'458._rt;
        //! vacuum permittivity: dielectric permittivity of vacuum [F/m]
        static constexpr auto ep0 = 8.8541878188e-12_rt;
        //! vacuum permeability: magnetic permeability of vacuum = 4.0e-7 * pi [H/m]
        //! Note that this is adjusted from the CODATA2022 value, 1.25663706127e-06
        //! So that the relation between mu0, ep0, and c is exact.
        static constexpr auto mu0 = 1.2566370612685e-06_rt;
        //! elementary charge [C]
        static constexpr auto q_e = 1.602176634e-19_rt;
        //! electron mass [kg]
        static constexpr auto m_e = 9.1093837139e-31_rt;
        //! proton mass [kg]
        static constexpr auto m_p = 1.67262192595e-27_rt;
        //! dalton: unified atomic mass unit [kg]
        static constexpr auto m_u = 1.66053906892e-27_rt;

        //! reduced Planck Constant = h / tau [J*s]
        static constexpr auto hbar = 1.0545718176461565e-34_rt;
        //! fine-structure constant = mu0/(4*pi)*q_e*q_e*c/hbar [dimensionless]
        //! The value is calculated from the expression. This differs
        //! slightly from the CODATA2022 value 0.0072973525643.
        static constexpr auto alpha = 0.0072973525643330135_rt;
        //! classical electron radius = 1./(4*pi*ep0) * q_e*q_e/(m_e*c*c) [m]
        static constexpr auto r_e = 2.8179403205e-15_rt;
        //! xi: nonlinearity parameter of Heisenberg-Euler effective theory = (2.*alpha*alpha*ep0*ep0*hbar*hbar*hbar)/(45.*m_e*m_e*m_e*m_e*c*c*c*c*c)
        static constexpr double xi = 1.3050122383827227e-52;
        //! xi times c2 = xi*c*c. This should be usable for single precision instead of xi; very close to smallest float32 number possible (1.2e-38)
        static constexpr auto xi_c2 = 1.1728865075613984e-35_rt;

        //! Boltzmann constant (exact) [J/K]
        static constexpr auto kb = 1.380649e-23_rt;

        //! 1 eV in [J]
        static constexpr auto eV = q_e;
        //! 1 MeV in [J]
        static constexpr auto MeV = q_e * 1e6_rt;
        //! 1 eV/c in [kg*m/s]
        static constexpr auto eV_invc = eV / c;
        //! 1 MeV/c in [kg*m/s]
        static constexpr auto MeV_invc = MeV / c;
        //! 1 eV/c^2 in [kg]
        static constexpr auto eV_invc2 = eV / (c * c);
        //! 1 MeV/c^2 in [kg]
        static constexpr auto MeV_invc2 = MeV / (c * c);
    } // namespace SI
} // namespace ablastr::constant

#endif // ABLASTR_CONSTANT_H_
