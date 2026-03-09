/* Copyright 2021 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "BinaryCollisionUtils.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <string>
#include <sstream>

#include "Utils/TextMsg.H"

namespace BinaryCollisionUtils{

    CollisionType get_collision_type (const std::string& collision_name,
                                      MultiParticleContainer const * const mypc)
    {
        const amrex::ParmParse pp_collision_name(collision_name);
        // For legacy, pairwisecoulomb is the default
        std::string type = "pairwisecoulomb";
        pp_collision_name.query("type", type);
        if (type == "pairwisecoulomb") {
            return CollisionType::PairwiseCoulomb;
        }
        else if (type == "nuclearfusion") {
            const NuclearFusionType fusion_type = get_nuclear_fusion_type(collision_name, mypc);
            return nuclear_fusion_type_to_collision_type(fusion_type);
        }
        else if (type == "bremsstrahlung") {
            return CollisionType::Bremsstrahlung;
        }
        else if (type == "dsmc") {
            return CollisionType::DSMC;
        }
        else if (type == "linear_breit_wheeler") {
            amrex::Vector<std::string> species_name;
            // Check that incoming species are photons
            pp_collision_name.getarr("species", species_name);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                species_name.size() == 2u,
                "Linear Breit-Wheeler collisions must involve exactly two species");
            auto& species1 = mypc->GetParticleContainerFromName(species_name[0]);
            auto& species2 = mypc->GetParticleContainerFromName(species_name[1]);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                species1.AmIA<PhysicalSpecies::photon>() && species2.AmIA<PhysicalSpecies::photon>(),
                "Species involved in linear Breit-Wheeler collisions must be of type photon.");
            // Check that product species are electron and positron
            amrex::Vector<std::string> product_species_name;
            pp_collision_name.getarr("product_species", product_species_name);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species_name.size() == 2u,
                "Linear Breit-Wheeler collisions must contain exactly two product species");
            auto& product_species1 = mypc->GetParticleContainerFromName(product_species_name[0]);
            auto& product_species2 = mypc->GetParticleContainerFromName(product_species_name[1]);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (product_species1.AmIA<PhysicalSpecies::electron>() && product_species2.AmIA<PhysicalSpecies::positron>())
                ||
                (product_species1.AmIA<PhysicalSpecies::positron>() && product_species2.AmIA<PhysicalSpecies::electron>()),
                "Product species of linear Breit-Wheeler collisions must be of type electron and positron");
            return CollisionType::LinearBreitWheeler;
        }
        else if (type == "linear_compton") {
            amrex::Vector<std::string> species_name;
            // Check that the first incoming species is a photon and the second is an electron/positron
            pp_collision_name.getarr("species", species_name);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                species_name.size() == 2u,
                "Linear Compton collisions must involve exactly two species");
            auto& species1 = mypc->GetParticleContainerFromName(species_name[0]);
            auto& species2 = mypc->GetParticleContainerFromName(species_name[1]);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( species1.AmIA<PhysicalSpecies::photon>(),
                "The first species in linear Compton collisions must be a photon");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( species2.AmIA<PhysicalSpecies::electron>() || species2.AmIA<PhysicalSpecies::positron>(),
                "The second species in linear Compton collisions must be an electron or positron");
            // Check that first product species is photon and second is electron/positron
            amrex::Vector<std::string> product_species_name;
            pp_collision_name.getarr("product_species", product_species_name);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species_name.size() == 2u,
                "Linear Compton collisions must contain exactly two product species");
            auto& product_species1 = mypc->GetParticleContainerFromName(product_species_name[0]);
            auto& product_species2 = mypc->GetParticleContainerFromName(product_species_name[1]);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( product_species1.AmIA<PhysicalSpecies::photon>(),
                "The first product species in linear Compton collisions must be a photon");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( product_species2.AmIA<PhysicalSpecies::electron>() || product_species2.AmIA<PhysicalSpecies::positron>(),
                "The second product species in linear Compton collisions must be an electron or positron");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( species2.getSpeciesTypeName() == product_species2.getSpeciesTypeName(),
                "The second incident species and the second product species in linear Compton collisions must be of the same type");
            return CollisionType::LinearCompton;
        }
        return CollisionType::Undefined;
    }

    NuclearFusionType get_nuclear_fusion_type (const std::string& collision_name,
                                               MultiParticleContainer const * const mypc)
    {
        using namespace amrex::literals;

        const amrex::ParmParse pp_collision_name(collision_name);
        amrex::Vector<std::string> species_names;
        pp_collision_name.getarr("species", species_names);
        auto& species1 = mypc->GetParticleContainerFromName(species_names[0]);
        auto& species2 = mypc->GetParticleContainerFromName(species_names[1]);

        // Compute the total mass before and after collision to check the fusion energy
        const amrex::ParticleReal mass_before = species1.getMass() + species2.getMass();
        amrex::ParticleReal mass_after = 0.0_prt;

        amrex::Vector<std::string> product_species_name;
        pp_collision_name.getarr("product_species", product_species_name);

        NuclearFusionType fusion_type = NuclearFusionType::Undefined;
        if ((species1.AmIA<PhysicalSpecies::hydrogen2>() && species2.AmIA<PhysicalSpecies::hydrogen3>())
            ||
            (species1.AmIA<PhysicalSpecies::hydrogen3>() && species2.AmIA<PhysicalSpecies::hydrogen2>())
            )
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species_name.size() == 2u,
                "ERROR: Deuterium-tritium fusion must contain exactly two product species");
            auto& product_species1 = mypc->GetParticleContainerFromName(product_species_name[0]);
            auto& product_species2 = mypc->GetParticleContainerFromName(product_species_name[1]);
            mass_after = product_species1.getMass() + product_species2.getMass();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (product_species1.AmIA<PhysicalSpecies::helium4>() && product_species2.AmIA<PhysicalSpecies::neutron>())
                ||
                (product_species1.AmIA<PhysicalSpecies::neutron>() && product_species2.AmIA<PhysicalSpecies::helium4>()),
                "ERROR: Product species of deuterium-tritium fusion must be of type neutron and helium4");
            fusion_type = NuclearFusionType::DeuteriumTritiumToNeutronHelium;
        }
        else if (species1.AmIA<PhysicalSpecies::hydrogen2>() && species2.AmIA<PhysicalSpecies::hydrogen2>())
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species_name.size() == 2u,
                "ERROR: Deuterium-deuterium fusion must contain exactly two product species");
            auto& product_species1 = mypc->GetParticleContainerFromName(product_species_name[0]);
            auto& product_species2 = mypc->GetParticleContainerFromName(product_species_name[1]);
            mass_after = product_species1.getMass() + product_species2.getMass();
            if (
                (product_species1.AmIA<PhysicalSpecies::helium3>() && product_species2.AmIA<PhysicalSpecies::neutron>())
                ||(product_species1.AmIA<PhysicalSpecies::neutron>() && product_species2.AmIA<PhysicalSpecies::helium3>())){
                fusion_type = NuclearFusionType::DeuteriumDeuteriumToNeutronHelium;
            } else if (
                (product_species1.AmIA<PhysicalSpecies::hydrogen3>() && product_species2.AmIA<PhysicalSpecies::proton>())
                ||(product_species1.AmIA<PhysicalSpecies::proton>() && product_species2.AmIA<PhysicalSpecies::hydrogen3>())){
                fusion_type = NuclearFusionType::DeuteriumDeuteriumToProtonTritium;
            } else {
                WARPX_ABORT_WITH_MESSAGE("ERROR: Product species of deuterium-deuterium fusion must be of type helium3 and neutron, or tritium/hydrogen3 and proton/hydrogen1");
            }
        }
        else if ((species1.AmIA<PhysicalSpecies::hydrogen2>() && species2.AmIA<PhysicalSpecies::helium3>())
            ||
            (species1.AmIA<PhysicalSpecies::helium3>() && species2.AmIA<PhysicalSpecies::hydrogen2>())
            )
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species_name.size() == 2u,
                "ERROR: Deuterium-helium fusion must contain exactly two product species");
            auto& product_species1 = mypc->GetParticleContainerFromName(product_species_name[0]);
            auto& product_species2 = mypc->GetParticleContainerFromName(product_species_name[1]);
            mass_after = product_species1.getMass() + product_species2.getMass();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (product_species1.AmIA<PhysicalSpecies::helium4>() && product_species2.AmIA<PhysicalSpecies::proton>())
                ||
                (product_species1.AmIA<PhysicalSpecies::proton>() && product_species2.AmIA<PhysicalSpecies::helium4>()),
                "ERROR: Product species of deuterium-helium fusion must be of type proton/hydrogen1 and alpha/helium4");
            fusion_type = NuclearFusionType::DeuteriumHeliumToProtonHelium;
        }
        else if ((species1.AmIA<PhysicalSpecies::proton>() && species2.AmIA<PhysicalSpecies::boron11>())
            ||
            (species1.AmIA<PhysicalSpecies::boron11>() && species2.AmIA<PhysicalSpecies::proton>())
            )
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species_name.size() == 1,
                "ERROR: Proton-boron must contain exactly one product species");
            auto& product_species = mypc->GetParticleContainerFromName(product_species_name[0]);
            mass_after = 3.0_prt*product_species.getMass();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species.AmIA<PhysicalSpecies::helium4>(),
                "ERROR: Product species of proton-boron fusion must be of type helium4");
            fusion_type = NuclearFusionType::ProtonBoronToAlphas;
        }
        else {
            WARPX_ABORT_WITH_MESSAGE("Binary nuclear fusion not implemented between species " +
                            species_names[0] + " of type " + species1.getSpeciesTypeName() +
                            " and species " + species_names[1] + " of type " +
                            species2.getSpeciesTypeName());
        }

        CheckFusionEnergy(fusion_type, mass_before, mass_after);

        return fusion_type;
    }

    void CheckFusionEnergy (const NuclearFusionType fusion_type,
                            const amrex::ParticleReal mass_before,
                            const amrex::ParticleReal mass_after)
    {
        // Compute the fusion energy
        amrex::ParticleReal fusion_energy = (mass_before - mass_after)*PhysConst::c2;

        // Verify that the fusion energy is close to what is expected.
        // The expected fusion energies are computed using fully ionized species mass
        // computed as M = A*m_u - Z*m_e, with the atomic mass numbers A found in
        // SpeciesPhysicalProperties.cpp
        std::ostringstream error_msg;
        amrex::ParticleReal expected_fusion_energy = 0.0_prt;
        if (fusion_type == NuclearFusionType::DeuteriumTritiumToNeutronHelium) {
            expected_fusion_energy = 17.58929696e6_prt * PhysConst::q_e;
            error_msg << "Fusion energy mismatch in D + T -> He4 + n\n";
        }
        if (fusion_type == NuclearFusionType::DeuteriumDeuteriumToProtonTritium) {
            expected_fusion_energy = 4.032653948e6_prt * PhysConst::q_e;
            error_msg << "Fusion energy mismatch in D + D -> T + p\n";
        }
        if (fusion_type == NuclearFusionType::DeuteriumDeuteriumToNeutronHelium) {
            expected_fusion_energy = 3.26891111e6_prt * PhysConst::q_e;
            error_msg << "Fusion energy mismatch in D + D -> He3 + n\n";
        }
        if (fusion_type == NuclearFusionType::DeuteriumHeliumToProtonHelium) {
            expected_fusion_energy = 18.35303980e6_prt * PhysConst::q_e;
            error_msg << "Fusion energy mismatch in D + He3 -> He3 + p\n";
        }
        if (fusion_type == NuclearFusionType::ProtonBoronToAlphas) {
            expected_fusion_energy = 8.68212502e6_prt * PhysConst::q_e;
            error_msg << "Fusion energy mismatch in p + B11 -> 3 He4\n";
        }

        const amrex::ParticleReal energy_error = amrex::Math::abs(fusion_energy - expected_fusion_energy);
        const amrex::ParticleReal energy_rel_tol = 0.01_prt;

        error_msg << "  energy error [eV]           = " << energy_error / PhysConst::q_e << "\n"
                  << "  energy tolerance (rel.)     = " << energy_rel_tol << "\n"
                  << "  expected fusion energy [eV] = " << expected_fusion_energy / PhysConst::q_e << "\n"
                  << "  computed fusion energy [eV] = " << fusion_energy / PhysConst::q_e<< "\n"
                  << "Check that species masses are set correctly.";

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            energy_error < energy_rel_tol*expected_fusion_energy,
            error_msg.str()
        );

    }
}
