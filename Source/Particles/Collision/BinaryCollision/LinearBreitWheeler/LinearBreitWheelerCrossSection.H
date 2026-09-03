/* Copyright 2023 Arianna Formenti
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef LINEAR_BREIT_WHEELER_CROSS_SECTION_H
#define LINEAR_BREIT_WHEELER_CROSS_SECTION_H

#include "Utils/WarpXConst.H"

#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

/**
 * \brief Computes the linear Breit-Wheeler cross section (refer to, for example,
 * [Gould and Schreder Phys. Rev. 155, 1404 (1967)]) as a function of E_star.
 * E_star is the kinetic energy of each photon in the center of momentum frame.
 * If E_star < m_e * c^2, then the total kinetic energy in the center of momentum
 * frame is not enough to produce a pair, hence the cross-section is zero.
 *
 * @param[in] E_star the kinetic energy of each photon involved in the collision in
 * the center of momentum frame, in SI units.
 * @return The total cross section in SI units (square meters).
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::ParticleReal LinearBreitWheelerCrossSection (const amrex::ParticleReal& E_star)
{
    using namespace amrex::literals;

    constexpr auto c2 = PhysConst::c2_v<amrex::ParticleReal>;
    constexpr amrex::ParticleReal me = PhysConst::m_e;

    if (E_star < me*c2) {return 0._prt;}

    constexpr amrex::ParticleReal pi = MathConst::pi;
    constexpr amrex::ParticleReal re2 = PhysConst::r_e * PhysConst::r_e;

    const amrex::ParticleReal s = amrex::Math::powi<2>(E_star/(me * c2));
    const amrex::ParticleReal beta = std::sqrt(1._prt-1._prt/s);

    const amrex::ParticleReal term1 = (3._prt-amrex::Math::powi<4>(beta))*std::log((1._prt+beta)/(1._prt-beta));
    const amrex::ParticleReal term2 = 2._prt*beta*(amrex::Math::powi<2>(beta)-2._prt);
    const amrex::ParticleReal cross_section = 0.5_prt*pi*re2*(1._prt-amrex::Math::powi<2>(beta))*(term1+term2);

    return cross_section;
}

/**
 * \brief Indefinite integral of the unnormalized BW differential cross section,
 * evaluated in the transformed coordinate y = atanh(beta*cos(theta)).
 *
 * This form avoids loss of precision near cos(theta) = +/-1 for highly relativistic
 * products. It corresponds to the untransformed indefinite integral
 * x / (1 - beta^2 x^2) * (-1 + 2 beta^2 - beta^4) - x
 * + atanh(beta * x) / beta * (3 - beta^4), evaluated with x = tanh(y)/beta.
 *
 * @param[in] beta velocity of the produced leptons in the CM frame (dimensionless)
 * @param[in] one_minus_beta2 1 - beta^2, evaluated without subtractive cancellation
 * @param[in] y transformed angular coordinate atanh(beta*cos(theta))
 * @return the indefinite integral evaluated at y
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::ParticleReal LBWIntegralOfDifferentialCrossSectionTransformed (
    const amrex::ParticleReal& beta,
    const amrex::ParticleReal& one_minus_beta2,
    const amrex::ParticleReal& y)
{
    using namespace amrex::literals;

    const amrex::ParticleReal tanh_y = std::tanh(y);
    const amrex::ParticleReal cosh_y = std::cosh(y);
    const amrex::ParticleReal x = tanh_y / beta;
    const amrex::ParticleReal one_minus_beta2_cosh_y = one_minus_beta2 * cosh_y;
    const amrex::ParticleReal coefficient =
        2._prt + 2._prt * one_minus_beta2 - one_minus_beta2 * one_minus_beta2;
    return -x * one_minus_beta2_cosh_y * one_minus_beta2_cosh_y
        - x + coefficient * y / beta;
}

/**
 * \brief Derivative with respect to y of
 * LBWIntegralOfDifferentialCrossSectionTransformed.
 *
 * @param[in] beta velocity of the produced leptons in the CM frame (dimensionless)
 * @param[in] one_minus_beta2 1 - beta^2, evaluated without subtractive cancellation
 * @param[in] y transformed angular coordinate atanh(beta*cos(theta))
 * @return the unnormalized differential cross section with respect to y
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::ParticleReal LBWDifferentialCrossSectionTransformed (
    const amrex::ParticleReal& beta,
    const amrex::ParticleReal& one_minus_beta2,
    const amrex::ParticleReal& y)
{
    using namespace amrex::literals;

    const amrex::ParticleReal tanh_y = std::tanh(y);
    const amrex::ParticleReal cosh_y = std::cosh(y);
    const amrex::ParticleReal sech2_y = 1._prt / (cosh_y * cosh_y);
    return (1._prt + tanh_y * tanh_y
        + 2._prt * one_minus_beta2 * (1._prt - one_minus_beta2 / sech2_y)) / beta;
}
#endif // LINEAR_BREIT_WHEELER_CROSS_SECTION_H
