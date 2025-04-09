/* Copyright 2022 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_BOSCH_HALE_FUSION_CROSS_SECTION_H
#define WARPX_BOSCH_HALE_FUSION_CROSS_SECTION_H

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Utils/WarpXConst.H"

#include <AMReX_REAL.H>

#include <cmath>

/**
 * \brief Computes the fusion cross section, using the analytical fits given in
 * H.-S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611
 *
 * @param[in] E_kin_star the kinetic energy of the reactants in their center of mass frame, in SI units.
 * @param[in] fusion_type indicates which fusion reaction to calculate the cross-section for
 * @param[in] m1 mass of the incoming particle
 * @param[in] m2 mass of the target particle
 * @return The total cross section in SI units (square meters).
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::ParticleReal BoschHaleFusionCrossSection (
    const amrex::ParticleReal& E_kin_star,
    const NuclearFusionType& fusion_type,
    const amrex::ParticleReal& m1,
    const amrex::ParticleReal& m2 )
{
    using namespace amrex::literals;

    constexpr amrex::ParticleReal joule_to_keV = 1.e-3_prt/PhysConst::q_e;
    const amrex::ParticleReal E_keV = E_kin_star*joule_to_keV;

    // If kinetic energy is 0, return a 0 cross section and avoid later division by 0.
    if (E_keV == 0._prt) {return 0._prt;}

    // Compute the Gamow constant B_G (in keV^{1/2})
    // (See Eq. 3 in H.-S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611)
    const amrex::ParticleReal m_reduced = m1 / (1._prt + m1/m2);
    // The formula for `B_G` below assumes that both reactants have Z=1
    // When one of the reactants is helium (Z=2), this formula is corrected further below.
    amrex::ParticleReal B_G = MathConst::pi * PhysConst::alpha
         * std::sqrt( 2._prt*m_reduced*PhysConst::c*PhysConst::c * joule_to_keV );
    if (fusion_type == NuclearFusionType::DeuteriumHeliumToProtonHelium) {
        // Take into account the fact that Z=2 for one of the reactant (helium) in this case
        B_G *= 2;
    }

    // Compute astrophysical_factor
    // (See Eq. 9 and Table IV in H.-S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611)
    amrex::ParticleReal A1=0_prt, A2=0_prt, A3=0_prt, A4=0_prt, A5=0_prt, B1=0_prt, B2=0_prt, B3=0_prt, B4=0_prt;
    if (fusion_type == NuclearFusionType::DeuteriumTritiumToNeutronHelium) {
        A1 = 6.927e4_prt;
        A2 = 7.454e8_prt;
        A3 = 2.050e6_prt;
        A4 = 5.2002e4_prt;
        A5 = 0_prt;
        B1 = 6.38e1_prt;
        B2 = -9.95e-1_prt;
        B3 = 6.981e-5_prt;
        B4 = 1.728e-4_prt;
    }
    else if (fusion_type == NuclearFusionType::DeuteriumDeuteriumToProtonTritium) {
        A1 = 5.5576e4_prt;
        A2 = 2.1054e2_prt;
        A3 = -3.2638e-2_prt;
        A4 = 1.4987e-6_prt;
        A5 = 1.8181e-10_prt;
        B1 = 0_prt;
        B2 = 0_prt;
        B3 = 0_prt;
        B4 = 0_prt;
    }
    else if (fusion_type == NuclearFusionType::DeuteriumDeuteriumToNeutronHelium) {
        A1 = 5.3701e4_prt;
        A2 = 3.3027e2_prt;
        A3 = -1.2706e-1_prt;
        A4 = 2.9327e-5_prt;
        A5 = -2.5151e-9_prt;
        B1 = 0_prt;
        B2 = 0_prt;
        B3 = 0_prt;
        B4 = 0_prt;
    }
    else if (fusion_type == NuclearFusionType::DeuteriumHeliumToProtonHelium) {
        A1 = 5.7501e6_prt;
        A2 = 2.5226e3_prt;
        A3 = 4.5566e1_prt;
        A4 = 0_prt;
        A5 = 0_prt;
        B1 = -3.1995e-3_prt;
        B2 = -8.5530e-6_prt;
        B3 = 5.9014e-8_prt;
        B4 = 0_prt;
    }

    const amrex::ParticleReal astrophysical_factor =
        (A1 + E_keV*(A2 + E_keV*(A3 + E_keV*(A4 + E_keV*A5)))) /
        (1_prt + E_keV*(B1 + E_keV*(B2 + E_keV*(B3 + E_keV*B4))));

    // Compute cross-section in SI units
    // (See Eq. 8 in H.-S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611)
    constexpr amrex::ParticleReal millibarn_to_sqm = 1.e-31_prt;
    return millibarn_to_sqm * astrophysical_factor/E_keV * std::exp(-B_G/std::sqrt(E_keV));
}

#endif // WARPX_BOSCH_HALE_FUSION_CROSS_SECTION_H
