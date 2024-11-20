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
        else if (type == "dsmc") {
            return CollisionType::DSMC;
        }
        return CollisionType::Undefined;
    }

    NuclearFusionType get_nuclear_fusion_type (const std::string& collision_name,
                                               MultiParticleContainer const * const mypc)
    {
        const amrex::ParmParse pp_collision_name(collision_name);
        amrex::Vector<std::string> species_names;
        pp_collision_name.getarr("species", species_names);
        auto& species1 = mypc->GetParticleContainerFromName(species_names[0]);
        auto& species2 = mypc->GetParticleContainerFromName(species_names[1]);
        amrex::Vector<std::string> product_species_name;
        pp_collision_name.getarr("product_species", product_species_name);

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
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (product_species1.AmIA<PhysicalSpecies::helium4>() && product_species2.AmIA<PhysicalSpecies::neutron>())
                ||
                (product_species1.AmIA<PhysicalSpecies::neutron>() && product_species2.AmIA<PhysicalSpecies::helium4>()),
                "ERROR: Product species of deuterium-tritium fusion must be of type neutron and helium4");
            return NuclearFusionType::DeuteriumTritiumToNeutronHelium;
        }
        else if (species1.AmIA<PhysicalSpecies::hydrogen2>() && species2.AmIA<PhysicalSpecies::hydrogen2>())
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species_name.size() == 2u,
                "ERROR: Deuterium-deuterium fusion must contain exactly two product species");
            auto& product_species1 = mypc->GetParticleContainerFromName(product_species_name[0]);
            auto& product_species2 = mypc->GetParticleContainerFromName(product_species_name[1]);
            if (
                (product_species1.AmIA<PhysicalSpecies::helium3>() && product_species2.AmIA<PhysicalSpecies::neutron>())
                ||(product_species1.AmIA<PhysicalSpecies::neutron>() && product_species2.AmIA<PhysicalSpecies::helium3>())){
                return NuclearFusionType::DeuteriumDeuteriumToNeutronHelium;
            } else if (
                (product_species1.AmIA<PhysicalSpecies::hydrogen3>() && product_species2.AmIA<PhysicalSpecies::hydrogen1>())
                ||(product_species1.AmIA<PhysicalSpecies::hydrogen1>() && product_species2.AmIA<PhysicalSpecies::hydrogen3>())){
                return NuclearFusionType::DeuteriumDeuteriumToProtonTritium;
            } else {
                WARPX_ABORT_WITH_MESSAGE("ERROR: Product species of deuterium-deuterium fusion must be of type helium3 and neutron, or hydrogen3 and hydrogen1");
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
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (product_species1.AmIA<PhysicalSpecies::helium4>() && product_species2.AmIA<PhysicalSpecies::hydrogen1>())
                ||
                (product_species1.AmIA<PhysicalSpecies::hydrogen1>() && product_species2.AmIA<PhysicalSpecies::helium4>()),
                "ERROR: Product species of deuterium-helium fusion must be of type hydrogen1 and helium4");
            return NuclearFusionType::DeuteriumHeliumToProtonHelium;
        }
        else if ((species1.AmIA<PhysicalSpecies::hydrogen1>() && species2.AmIA<PhysicalSpecies::boron11>())
            ||
            (species1.AmIA<PhysicalSpecies::boron11>() && species2.AmIA<PhysicalSpecies::hydrogen1>())
            )
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species_name.size() == 1,
                "ERROR: Proton-boron must contain exactly one product species");
            auto& product_species = mypc->GetParticleContainerFromName(product_species_name[0]);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species.AmIA<PhysicalSpecies::helium4>(),
                "ERROR: Product species of proton-boron fusion must be of type helium4");
            return NuclearFusionType::ProtonBoronToAlphas;
        }
        WARPX_ABORT_WITH_MESSAGE("Binary nuclear fusion not implemented between species " +
                        species_names[0] + " of type " + species1.getSpeciesTypeName() +
                        " and species " + species_names[1] + " of type " +
                        species2.getSpeciesTypeName());
        return NuclearFusionType::Undefined;
    }

    CollisionType nuclear_fusion_type_to_collision_type (const NuclearFusionType fusion_type)
    {
        if (fusion_type == NuclearFusionType::DeuteriumTritiumToNeutronHelium) {
            return CollisionType::DeuteriumTritiumToNeutronHeliumFusion;
        }
        if (fusion_type == NuclearFusionType::DeuteriumDeuteriumToProtonTritium) {
            return CollisionType::DeuteriumDeuteriumToProtonTritiumFusion;
        }
        if (fusion_type == NuclearFusionType::DeuteriumDeuteriumToNeutronHelium) {
            return CollisionType::DeuteriumDeuteriumToNeutronHeliumFusion;
        }
        if (fusion_type == NuclearFusionType::DeuteriumHeliumToProtonHelium) {
            return CollisionType::DeuteriumHeliumToProtonHeliumFusion;
        }
        if (fusion_type == NuclearFusionType::ProtonBoronToAlphas) {
            return CollisionType::ProtonBoronToAlphasFusion;
        }
        WARPX_ABORT_WITH_MESSAGE("Invalid nuclear fusion type");
        return CollisionType::Undefined;
    }
}
