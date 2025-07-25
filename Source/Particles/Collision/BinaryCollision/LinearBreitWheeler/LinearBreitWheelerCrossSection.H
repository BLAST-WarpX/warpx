/* Copyright 2023 Arianna Formenti
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef LINEAR_BREIT_WHEELER_CROSS_SECTION_H
#define LINEAR_BREIT_WHEELER_CROSS_SECTION_H

#include "Utils/WarpXConst.H"

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

    constexpr amrex::ParticleReal c_sq = PhysConst::c * PhysConst::c;
    constexpr amrex::ParticleReal me = PhysConst::m_e;

    if (E_star < me*c_sq) {return 0._prt;}

    constexpr auto one_half_pr = amrex::ParticleReal(1./2.);
    constexpr amrex::ParticleReal pi = MathConst::pi;
    constexpr amrex::ParticleReal re2 = PhysConst::r_e * PhysConst::r_e;
    auto constexpr pow2 = [](auto const x) { return x*x; };
    auto constexpr pow4 = [](auto const x) { return x*x*x*x; };

    const amrex::ParticleReal s = pow2(E_star/(me * c_sq));
    const amrex::ParticleReal beta = std::sqrt(1._prt-1._prt/s);

    const amrex::ParticleReal term1 = (3._prt-pow4(beta))*std::log((1._prt+beta)/(1._prt-beta));
    const amrex::ParticleReal term2 = 2._prt*beta*(pow2(beta)-2._prt);
    const amrex::ParticleReal cross_section = one_half_pr*pi*re2*(1._prt-pow2(beta))*(term1+term2);

    return cross_section;
}

#endif // LINEAR_BREIT_WHEELER_CROSS_SECTION_H
