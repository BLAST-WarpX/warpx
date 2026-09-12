/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "MaterialRegistry.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"

#include <AMReX_ParmParse.H>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <set>
#include <string>
#include <utility>

using namespace amrex::literals;

namespace warpx::materials
{
    namespace
    {
        [[nodiscard]] std::optional<MaterialTableHandle>
        read_table_handle (
            amrex::ParmParse const& pp,
            std::string const& prefix,
            std::string const& default_material_key)
        {
            std::string file;
            long long material_id = -1;
            bool const file_is_set = pp.query(prefix + "_table_file", file);
            bool const id_is_set = pp.query(prefix + "_material_id", material_id);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                file_is_set == id_is_set,
                pp.prefixedName(prefix + "_table_file") + " and "
                    + pp.prefixedName(prefix + "_material_id")
                    + " must be specified together.");
            if (!file_is_set) { return std::nullopt; }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !file.empty() && material_id >= 0,
                pp.prefixedName(prefix)
                    + " table file must be nonempty and material ID must be "
                      "non-negative.");
            std::string material_key = default_material_key;
            pp.query(prefix + "_material_key", material_key);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !material_key.empty(),
                pp.prefixedName(prefix + "_material_key")
                    + " must be nonempty.");
            return MaterialTableHandle{
                std::move(file), static_cast<std::int64_t>(material_id),
                std::move(material_key)};
        }

        [[nodiscard]] std::string normalize_path (std::string const& path)
        {
            return std::filesystem::path(path).lexically_normal().string();
        }
    }

    void MaterialRegistry::ReadParameters ()
    {
        m_materials.clear();
        m_selector = {};
        amrex::ParmParse const pp("materials");
        std::vector<std::string> names;
        pp.queryarr("names", names);
        if (names.empty()) { return; }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            names.size() <= static_cast<std::size_t>(max_materials),
            "materials.names supports at most "
                + std::to_string(max_materials) + " registered materials.");
        m_selector.num_materials = static_cast<int>(names.size());

        std::string mixture_policy = "resolved_single_material";
        pp.query("mixture_policy", mixture_policy);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            mixture_policy == "resolved_single_material",
            "materials.mixture_policy currently supports only "
            "resolved_single_material. WarpX does not yet combine separately "
            "group-averaged Rosseland or pure-material EOS tables in one cell.");
        pp.query("vacuum_mass_density", m_selector.vacuum_mass_density);
        pp.query(
            "mixed_cell_absolute_tolerance",
            m_selector.mixed_cell_absolute_tolerance);
        pp.query(
            "mixed_cell_relative_tolerance",
            m_selector.mixed_cell_relative_tolerance);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(m_selector.vacuum_mass_density)
                && m_selector.vacuum_mass_density >= 0.0_rt,
            "materials.vacuum_mass_density must be finite and non-negative.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(m_selector.mixed_cell_absolute_tolerance)
                && m_selector.mixed_cell_absolute_tolerance >= 0.0_rt,
            "materials.mixed_cell_absolute_tolerance must be finite and "
            "non-negative.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(m_selector.mixed_cell_relative_tolerance)
                && m_selector.mixed_cell_relative_tolerance >= 0.0_rt
                && m_selector.mixed_cell_relative_tolerance < 0.5_rt,
            "materials.mixed_cell_relative_tolerance must be finite and in "
            "[0,0.5) so a resolved cell has a unique dominant material.");

        std::set<std::string> unique_names;
        std::set<std::string> assigned_species;
        m_materials.reserve(names.size());
        for (std::string const& name : names) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !name.empty() && unique_names.insert(name).second,
                "materials.names entries must be nonempty and unique; duplicate '"
                    + name + "'.");
            amrex::ParmParse const pp_material("materials." + name);
            MaterialDefinition definition;
            definition.name = name;
            pp_material.getarr("species", definition.species);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !definition.species.empty(),
                "materials." + name + ".species must be nonempty.");
            for (std::string const& species : definition.species) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    !species.empty() && assigned_species.insert(species).second,
                    "Material carrier species must be nonempty and belong to "
                    "exactly one named material; duplicate mapping for species '"
                        + species + "'.");
            }
            definition.electron_eos = read_table_handle(
                pp_material, "electron_eos", name);
            definition.opacity = read_table_handle(
                pp_material, "opacity", name);
            m_materials.push_back(std::move(definition));
        }
        std::sort(
            m_materials.begin(), m_materials.end(),
            [] (MaterialDefinition const& lhs, MaterialDefinition const& rhs)
            { return lhs.name < rhs.name; });
    }

    void MaterialRegistry::ValidateSpecies (
        MultiParticleContainer const& particles) const
    {
        if (!enabled()) { return; }
        std::set<std::string> registered_species;
        auto const& particle_species = particles.GetSpeciesNames();
        for (MaterialDefinition const& definition : m_materials) {
            for (std::string const& species_name : definition.species) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    std::find(
                        particle_species.begin(), particle_species.end(),
                        species_name) != particle_species.end(),
                    "Named material '" + definition.name
                        + "' refers to unknown particle species '"
                        + species_name + "'.");
                auto const& species =
                    particles.GetParticleContainerFromName(species_name);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    species.getMass() > 0.0_prt
                        && species.getCharge() > 0.0_prt
                        && !species.do_not_deposit
                        && !species.HasEvolvingChargeState(),
                    "Named material carrier species '" + species_name
                        + "' must be a depositing, massive, positively charged "
                          "particle species with fixed charge state.");
                registered_species.insert(species_name);
            }
        }

        for (std::string const& species_name : particle_species) {
            auto const& species =
                particles.GetParticleContainerFromName(species_name);
            if (species.getMass() <= 0.0_prt
                || species.getCharge() <= 0.0_prt
                || species.do_not_deposit)
            {
                continue;
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                registered_species.count(species_name) == 1,
                "Every depositing massive positive particle species must "
                "belong to exactly one materials.names entry; unregistered '"
                    + species_name + "'.");
        }
    }

    MaterialDefinition const* MaterialRegistry::findMaterial (
        std::string const& name) const noexcept
    {
        auto const it = std::lower_bound(
            m_materials.begin(), m_materials.end(), name,
            [] (MaterialDefinition const& material, std::string const& key)
            { return material.name < key; });
        return it != m_materials.end() && it->name == name ? &*it : nullptr;
    }

    MaterialDefinition const* MaterialRegistry::findMaterialForSpecies (
        std::string const& species) const noexcept
    {
        for (MaterialDefinition const& definition : m_materials) {
            if (std::find(
                    definition.species.begin(), definition.species.end(),
                    species) != definition.species.end())
            {
                return &definition;
            }
        }
        return nullptr;
    }

    int MaterialRegistry::classifyResolvedCell (
        std::span<amrex::Real const> const material_mass_densities) const noexcept
    {
        if (material_mass_densities.size() != m_materials.size()
            || material_mass_densities.size()
                > static_cast<std::size_t>(max_materials))
        {
            return static_cast<int>(ResolvedCell::Invalid);
        }
        amrex::GpuArray<amrex::Real, max_materials> densities{};
        for (std::size_t material = 0;
             material < material_mass_densities.size(); ++material)
        {
            densities[static_cast<int>(material)] =
                material_mass_densities[material];
        }
        return m_selector(densities);
    }

    bool MaterialRegistry::sameTableHandle (
        MaterialTableHandle const& lhs,
        MaterialTableHandle const& rhs) noexcept
    {
        return normalize_path(lhs.file) == normalize_path(rhs.file)
            && lhs.material_id == rhs.material_id
            && lhs.material_key == rhs.material_key;
    }
}
