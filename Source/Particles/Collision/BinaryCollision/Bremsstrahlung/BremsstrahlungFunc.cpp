/* Copyright 2024 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "BremsstrahlungFunc.H"

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

using namespace amrex::literals;

BremsstrahlungFunc::BremsstrahlungFunc (std::string const& collision_name, MultiParticleContainer const * const mypc,
                                        bool isSameSpecies)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!isSameSpecies,
                                     "BremsstrahlungFunc: The two colliding species must be different");

    const amrex::ParmParse pp_collision_name(collision_name);

    // Read in the number of electrons on the target
    int Z;
    pp_collision_name.get("Z", Z);

    std::string product_species_name;
    pp_collision_name.get("product_species", product_species_name);
    auto& product_species = mypc->GetParticleContainerFromName(product_species_name);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(product_species.AmIA<PhysicalSpecies::photon>(),
                                     "BremsstrahlungFunc: The product species must be photons");

    bool create_photons = true;
    pp_collision_name.query("create_photons", create_photons);
    m_exe.m_create_photons = create_photons;

    amrex::ParticleReal multiplier = 1._prt;
    pp_collision_name.query("multiplier", multiplier);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(multiplier >= 1.,
                                     "BremsstrahlungFunc: The multiplier must be greater than or equal to one");
    m_exe.m_multiplier = multiplier;

    // Fill in the m_kdsigdk array
    UploadCrossSection(Z);

}

void
BremsstrahlungFunc::UploadCrossSection (int Z)
{

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_kdsigdk_map.count(Z) == 1,
        "Bremsstrahlung cross section not available for Z = " + std::to_string(Z) + "!");

    constexpr auto m_e_eV = PhysConst::m_e*PhysConst::c*PhysConst::c/PhysConst::q_e; // 0.511e6

    std::vector<std::vector<amrex::ParticleReal>> & kdsigdk = m_kdsigdk_map.at(Z);

    // Convert Seltzer and Berger energy-weighted differential cross section to units of [m^2]
    for (int iee=0; iee < Executor::nKE; iee++) {
        amrex::ParticleReal const E = m_exe.m_KEgrid_eV[iee]/m_e_eV;
        amrex::ParticleReal const gamma = 1.0_prt + E;
        /* betaSq = 1.0 - 1.0/gamma/gamma */
        amrex::ParticleReal const betaSq = (E*E + 2._prt*E)/gamma/gamma;
        // 1.0e-31 converts mBarn to m**2
        amrex::ParticleReal const scale_factor = 1.0e-31_prt*Z*Z/betaSq;
        for (int iep=0; iep < Executor::nkoT1; iep++) {
            m_exe.m_kdsigdk[iee][iep] = kdsigdk[iee][iep]*scale_factor;
        }
    }

}
