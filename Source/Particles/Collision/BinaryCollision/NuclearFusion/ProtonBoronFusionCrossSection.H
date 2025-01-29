/* Copyright 2021 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_PROTON_BORON_FUSION_CROSS_SECTION_H
#define WARPX_PROTON_BORON_FUSION_CROSS_SECTION_H

#include "Utils/WarpXConst.H"

#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

/**
 * \brief Computes the total proton-boron fusion cross section in the range 0 < E < 9.76 MeV using
 * the analytical fits given in A. Tentori & F. Belloni, Nuclear Fusion, 63, 086001 (2023).
 *
 * @param[in] E_keV the kinetic energy of the proton-boron pair in its center of mass frame, in
 * keV.
 * @return The total cross section in barn.
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::ParticleReal ProtonBoronFusionCrossSectionTentori (const amrex::ParticleReal& E_keV)
{
    using namespace amrex::literals;
    using namespace amrex::Math;

    // If kinetic energy is 0, return a 0 cross section and avoid later division by 0.
    if (E_keV == 0._prt) {return 0._prt;}

    // Fits also use energy in MeV
    const amrex::ParticleReal E_MeV = E_keV*1.e-3_prt;
    constexpr amrex::ParticleReal joule_to_MeV = 1.e-6_prt/PhysConst::q_e;

    // Compute Gamow factor, in MeV
    constexpr auto one_pr = 1._prt;
    constexpr auto Z_boron = 5._prt;
    constexpr amrex::ParticleReal m_boron = 11.00930536_prt * PhysConst::m_u;
    constexpr amrex::ParticleReal m_hydrogen = 1.00782503223 * PhysConst::m_u;
    constexpr amrex::ParticleReal m_reduced = m_boron / (one_pr + m_boron/m_hydrogen);
    constexpr amrex::ParticleReal gamow_factor = m_reduced / 2._prt *
                        (PhysConst::q_e*PhysConst::q_e * Z_boron /
                           (2._prt*PhysConst::ep0*PhysConst::hbar)) *
                        (PhysConst::q_e*PhysConst::q_e * Z_boron /
                           (2._prt*PhysConst::ep0*PhysConst::hbar)) *
                        joule_to_MeV;

    // Compute astrophysical factor, in MeV barn, using the fits
    constexpr auto E_lim1 = 400._prt; // Limits between the different fit regions
    constexpr auto E_lim2 = 668._prt;
    amrex::ParticleReal astrophysical_factor;
    if (E_keV < E_lim1)
    {
        constexpr auto C0 = 197._prt;
        constexpr auto C1 = 0.269_prt;
        constexpr auto C2 = 2.54e-4_prt;
        constexpr auto AL = 1.82e4_prt;
        constexpr auto EL = 148._prt;
        constexpr auto dEL_sq = 2.35_prt*2.35_prt;
        astrophysical_factor = C0 + C1*E_keV + C2*powi<2>(E_keV) +
                                               AL/((E_keV - EL)*(E_keV - EL) + dEL_sq);
    }
    else if (E_keV < E_lim2)
    {
        constexpr auto D0 = 346._prt;
        constexpr auto D1 = 150._prt;
        constexpr auto D2 = -59.9_prt;
        constexpr auto D5 = -0.460_prt;
        const amrex::ParticleReal E_norm = (E_keV-400._prt) * 1.e-2_prt;
        astrophysical_factor = D0 + D1*E_norm + D2*powi<2>(E_norm) + D5*powi<5>(E_norm);
    }
    else
    {
        constexpr auto A0 = 1.98e6_prt;
        constexpr auto A1 = 3.89e6_prt;
        constexpr auto A2 = 1.36e6_prt;
        constexpr auto A3 = 3.71e6_prt;
        constexpr auto E0 = 640.9_prt;
        constexpr auto E1 = 1211._prt;
        constexpr auto E2 = 2340._prt;
        constexpr auto E3 = 3294._prt;
        constexpr auto dE0_sq = 85.5_prt*85.5_prt;
        constexpr auto dE1_sq = 414._prt*414._prt;
        constexpr auto dE2_sq = 221._prt*221._prt;
        constexpr auto dE3_sq = 351._prt*351._prt;
        constexpr auto B = 0.381_prt;
        astrophysical_factor =  A0 / ((E_keV-E0)*(E_keV-E0) + dE0_sq) +
                                A1 / ((E_keV-E1)*(E_keV-E1) + dE1_sq) +
                                A2 / ((E_keV-E2)*(E_keV-E2) + dE2_sq) +
                                A3 / ((E_keV-E3)*(E_keV-E3) + dE3_sq) + B;
    }

    // Compute cross section, in barn
    return astrophysical_factor/E_MeV*std::exp(-std::sqrt(gamow_factor/E_MeV));
}

/**
 * \brief Computes the total proton-boron fusion cross section in the range E > 9.76 MeV using a
 * simple power law fit of the data presented in Buck et al., Nuclear Physics A, 398(2), 189-202
 * (1983) (data can also be found in the EXFOR database). Note: the fit in Buck et al. started
 * from 3.5 MeV. The same exponent power has been used here however the cross_section_start_fit
 * has been modified to ensure exact continuity with the fit used for lower energies.

 * @param[in] E_keV the kinetic energy of the proton-boron pair in its center of mass frame, in
 * keV.
 * @return The total cross section in barn.
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::ParticleReal ProtonBoronFusionCrossSectionBuck (const amrex::ParticleReal& E_keV)
{
    using namespace amrex::literals;

    constexpr amrex::ParticleReal E_start_fit = 9760._prt; // Fit starts at 9.76 MeV
    // cross section at E = E_start_fit, in barn
    constexpr amrex::ParticleReal cross_section_start_fit = 0.01277998_prt;
    constexpr amrex::ParticleReal slope_fit = -2.661840717596765_prt;

    // Compute fitted value
    return cross_section_start_fit*std::pow(E_keV/E_start_fit, slope_fit);
}

/**
 * \brief Computes the total proton-boron fusion cross section using the analytical fit described
 * in A. Tentori & F. Belloni, Nuclear Fusion, 63, 086001 (2023). Includes the Breit-Wigner term
 * to reconstruct the 148 keV resonance, which is missing from the Sikora, Wellar dataset. When
 * E_kin_star > 9.76 MeV, we use a simple power law fit of the data presented in Buck et al.,
 * Nuclear Physics A, 398(2), 189-202 (1983). Both fits return the same value for
 * E_kin_star = 9.76 MeV.

 * @param[in] E_kin_star the kinetic energy of the proton-boron pair in its center of mass frame,
 * in SI units.
 * @return The total cross section in SI units (square meters).
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::ParticleReal ProtonBoronFusionCrossSection (const amrex::ParticleReal& E_kin_star)
{
    using namespace amrex::literals;

    // Fits use energy in keV
    constexpr amrex::ParticleReal joule_to_keV = 1.e-3_prt/PhysConst::q_e;
    const amrex::ParticleReal E_keV = E_kin_star*joule_to_keV;
    constexpr amrex::ParticleReal E_threshold = 9760._prt;

    const amrex::ParticleReal cross_section_b = (E_keV <= E_threshold) ?
                                                ProtonBoronFusionCrossSectionTentori(E_keV) :
                                                ProtonBoronFusionCrossSectionBuck(E_keV);

    // Convert cross section to SI units: barn to square meter
    constexpr auto barn_to_sqm = 1.e-28_prt;
    return cross_section_b*barn_to_sqm;
}

#endif // WARPX_PROTON_BORON_FUSION_CROSS_SECTION_H
