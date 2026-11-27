/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "MaterialRegistry.H"
#include "MaterialTableIdentity.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"

#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <set>
#include <string>
#include <tuple>
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

        [[nodiscard]] std::string
        material_binding (std::string const& name,
                          std::vector<std::string> const& species) {
            std::ostringstream output;
            output << "material:" << std::quoted(name) << " species:";
            for (std::size_t i = 0; i < species.size(); ++i) {
                if (i != 0U) {
                    output << ',';
                }
                output << std::quoted(species[i]);
            }
            return output.str();
        }

        [[nodiscard]] std::string
        eos_interpretation () {
            std::string interpretation =
                "singularity_spiner_rho_temp_electron_only_v2"
                "|units=cgs_to_si"
                "|energy_reference=table_absolute"
                "|density_tail=evaluate_rho_min_scale_pressure"
                "|temperature_bounds=table_intersection"
                "|inverse=bracket_64";
#ifdef SINGULARITY_USE_TRUE_LOG_GRIDDING
            interpretation += "|gridding=log10"
#ifdef SINGULARITY_USE_SINGLE_LOGS
                              "|log_precision=single"
#else
                              "|log_precision=double"
#endif
#else
            interpretation += "|gridding=nqt"
#ifdef SINGULARITY_NQT_ORDER_1
                              "|nqt_order=1"
#else
                              "|nqt_order=2"
#endif
#ifdef SINGULARITY_NQT_PORTABLE
                              "|nqt_implementation=portable"
#else
                              "|nqt_implementation=bitwise"
#endif
#endif
                ;
            return interpretation;
        }

        [[nodiscard]] std::string
        opacity_interpretation (std::string const& interpolation) {
            return "warpx_opacity_multigroup_hdf5_v1"
                   "|units=si_mass_opacity"
                   "|array_order=rho_temperature_group"
                   "|interpolation=" +
                   interpolation + "|density_conversion=linear" +
                   "|groups=table_order" + "|range=reject";
        }

        [[nodiscard]] std::optional<MaterialTableHandle>
        read_legacy_opacity_handle (amrex::ParmParse const& pp)
        {
            std::string file;
            long long material_id = -1;
            bool const file_is_set =
                pp.query("material_opacity_table_file", file);
            bool const id_is_set =
                pp.query("material_opacity_table_id", material_id);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                file_is_set == id_is_set,
                "radiation_transport.material_opacity_table_file and "
                "radiation_transport.material_opacity_table_id must be "
                "specified together.");
            if (!file_is_set) { return std::nullopt; }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !file.empty() && material_id >= 0,
                "The legacy material opacity table file must be nonempty and "
                "its table ID must be non-negative.");
            std::string material_key = "legacy-material-opacity";
            pp.query("material_opacity_material_key", material_key);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !material_key.empty(),
                "radiation_transport.material_opacity_material_key must be "
                "nonempty.");
            return MaterialTableHandle{
                std::move(file), static_cast<std::int64_t>(material_id),
                std::move(material_key)};
        }

        void capture_table (
            MaterialTableHandle& handle,
            std::string const& role,
            std::string const& binding,
            std::string const& interpretation,
            std::vector<MaterialTableFingerprint>& fingerprints,
            std::vector<std::string>& staged_files,
            std::string& error)
        {
            try {
                std::ifstream input{handle.file, std::ios::binary};
                if (!input) {
                    throw std::runtime_error(
                        "could not open externally supplied table '"
                        + handle.file + "'");
                }
                std::vector<char> bytes;
                std::array<char, 64U * 1024U> buffer{};
                while (input) {
                    input.read(
                        buffer.data(),
                        static_cast<std::streamsize>(buffer.size()));
                    auto const count = input.gcount();
                    if (count > 0) {
                        bytes.insert(
                            bytes.end(), buffer.data(), buffer.data() + count);
                    }
                }
                if (input.bad()) {
                    throw std::runtime_error(
                        "could not read externally supplied table '"
                        + handle.file + "'");
                }

                static std::atomic<unsigned long> sequence{0};
                auto const suffix =
                    std::filesystem::path(handle.file).extension().string();
                auto const timestamp = std::chrono::steady_clock::now()
                    .time_since_epoch().count();
                auto const staged = std::filesystem::temp_directory_path()
                    / ("warpx-material-table-"
                       + std::to_string(
                           amrex::ParallelDescriptor::MyProc())
                       + "-" + std::to_string(timestamp)
                       + "-" + std::to_string(sequence.fetch_add(1))
                       + "-" + std::to_string(bytes.size()) + suffix);
                std::ofstream output{staged, std::ios::binary | std::ios::trunc};
                if (!output) {
                    throw std::runtime_error(
                        "could not stage externally supplied table '"
                        + handle.file + "'");
                }
                output.write(
                    bytes.data(), static_cast<std::streamsize>(bytes.size()));
                output.close();
                if (!output) {
                    throw std::runtime_error(
                        "could not write exact-byte table staging file for '"
                        + handle.file + "'");
                }

                std::uint64_t size = 0;
                auto const digest =
                    MaterialTableIdentity::HashFile(staged.string(), size);
                if (size != bytes.size()) {
                    throw std::runtime_error(
                        "exact-byte table staging size changed for '"
                        + handle.file + "'");
                }
                std::string const logical_file = handle.logical_file();
                handle.load_file = staged.string();
                handle.content_sha256 = digest;
                handle.content_size = size;
                staged_files.push_back(handle.load_file);
                MaterialTableIdentity::RegisterLoadAlias(
                    logical_file, role, handle.material_id,
                    handle.material_key, handle.load_file);
                handle.source_file = logical_file;
                handle.file = handle.load_file;
                fingerprints.push_back({
                    role, binding, handle.material_id, handle.material_key,
                    digest, size, interpretation});
            } catch (std::exception const& exception) {
                if (!error.empty()) { error += "; "; }
                error += exception.what();
            }
        }

        [[nodiscard]] std::string real_text (amrex::Real const value)
        {
            std::ostringstream output;
            output << std::setprecision(
                std::numeric_limits<amrex::Real>::max_digits10)
                   << value;
            return output.str();
        }

        [[nodiscard]] std::string selection_contract (
            MaterialRegistry::ResolvedCellSelector const& selector,
            std::string const& opacity_interpolation)
        {
            std::ostringstream output;
            output << "mixture_policy=resolved_single_material"
                   << ";vacuum_mass_density="
                   << real_text(selector.vacuum_mass_density)
                   << ";mixed_cell_absolute_tolerance="
                   << real_text(selector.mixed_cell_absolute_tolerance)
                   << ";mixed_cell_relative_tolerance="
                   << real_text(selector.mixed_cell_relative_tolerance)
                   << ";fixed_charge=true"
                   << ";electron_model=singularity_spiner"
                   << ";electron_table_split=electron_only"
                   << ";opacity_backend=hdf5_multigroup"
                   << ";opacity_interpolation=" << opacity_interpolation
                   << ";hash_mode=complete_file_sha256";

            amrex::ParmParse const pp_rad("radiation_transport");
            std::vector<amrex::Real> boundaries;
            std::vector<amrex::Real> photon_energies;
            if (pp_rad.queryarr("energy_group_boundaries", boundaries)) {
                output << ";energy_group_boundaries=";
                for (std::size_t i = 0; i < boundaries.size(); ++i) {
                    if (i != 0U) { output << ','; }
                    output << real_text(boundaries[i]);
                }
            }
            if (pp_rad.queryarr("group_photon_energies", photon_energies)) {
                output << ";group_photon_energies=";
                for (std::size_t i = 0; i < photon_energies.size(); ++i) {
                    if (i != 0U) { output << ','; }
                    output << real_text(photon_energies[i]);
                }
            }
            return output.str();
        }
    }

    MaterialRegistry::~MaterialRegistry ()
    {
        std::error_code error;
        for (auto const& file : m_staged_table_files) {
            std::filesystem::remove(file, error);
            error.clear();
        }
    }

    void MaterialRegistry::ReadParameters ()
    {
        for (auto const& file : m_staged_table_files) {
            std::error_code error;
            std::filesystem::remove(file, error);
        }
        m_staged_table_files.clear();
        MaterialTableIdentity::ResetForTesting();
        m_materials.clear();
        m_selector = {};

        amrex::ParmParse const pp("materials");
        amrex::ParmParse const pp_hybrid("hybrid_pic_model");
        amrex::ParmParse const pp_rad("radiation_transport");
        std::vector<std::string> names;
        pp.queryarr("names", names);
        if (!names.empty()) {
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
                        !species.empty()
                            && assigned_species.insert(species).second,
                        "Material carrier species must be nonempty and belong "
                        "to exactly one named material; duplicate mapping for "
                        "species '" + species + "'.");
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

        std::string electron_model = "ideal_gas";
        pp_hybrid.query("electron_thermodynamics", electron_model);
        bool const consume_eos = electron_model == "singularity_spiner";
        bool radiation_enabled = false;
        pp_rad.query("enabled", radiation_enabled);
        std::string opacity_interpolation = "linear";
        pp_rad.query(
            "opacity_table_interpolation", opacity_interpolation);
        bool has_opacity_input = pp_rad.contains("material_opacity_table_file");
        for (auto const& definition : m_materials) {
            amrex::ParmParse const pp_material(
                "materials." + definition.name);
            has_opacity_input =
                has_opacity_input || pp_material.contains("opacity_table_file");
        }
        bool const consume_opacity =
            radiation_enabled && has_opacity_input;

        std::vector<MaterialTableFingerprint> fingerprints;
        std::string identity_error;
        for (auto& definition : m_materials) {
            auto const binding =
                material_binding(definition.name, definition.species);
            if (consume_eos && definition.electron_eos.has_value()) {
                capture_table(
                    *definition.electron_eos, "electron_eos", binding,
                    eos_interpretation(), fingerprints,
                    m_staged_table_files, identity_error);
            }
            if (consume_opacity && definition.opacity.has_value()) {
                capture_table(
                    *definition.opacity, "opacity", binding,
                    opacity_interpretation(opacity_interpolation),
                    fingerprints, m_staged_table_files, identity_error);
            }
        }

        if (consume_eos) {
            std::vector<std::string> eos_species;
            pp_hybrid.queryarr("electron_eos_species", eos_species);
            for (auto const& species : eos_species) {
                auto handle = read_table_handle(
                    pp_hybrid, "electron_eos_" + species, species);
                if (!handle.has_value()) { continue; }
                bool const registered = std::any_of(
                    m_materials.begin(), m_materials.end(),
                    [&] (MaterialDefinition const& definition)
                    {
                        return definition.electron_eos.has_value()
                            && MaterialRegistry::sameTableHandle(
                                *definition.electron_eos, *handle);
                    });
                if (registered) { continue; }
                std::ostringstream binding;
                binding << "species:" << std::quoted(species);
                capture_table(
                    *handle, "electron_eos", binding.str(),
                    eos_interpretation(), fingerprints,
                    m_staged_table_files, identity_error);
            }
        }
        if (consume_opacity) {
            auto handle = read_legacy_opacity_handle(pp_rad);
            if (handle.has_value()) {
                bool const registered = std::any_of(
                    m_materials.begin(), m_materials.end(),
                    [&] (MaterialDefinition const& definition)
                    {
                        return definition.opacity.has_value()
                            && MaterialRegistry::sameTableHandle(
                                *definition.opacity, *handle);
                    });
                if (!registered) {
                    capture_table(
                        *handle, "opacity",
                        "legacy:radiation_transport.material_opacity_table_file",
                        opacity_interpretation(opacity_interpolation),
                        fingerprints, m_staged_table_files, identity_error);
                }
            }
        }

        int local_failure = identity_error.empty() ? 0 : 1;
        amrex::ParallelDescriptor::ReduceIntMax(&local_failure, 1);
        if (local_failure != 0) {
            if (!identity_error.empty()) {
                amrex::Print()
                    << Utils::TextMsg::Err(
                           "Material table identity capture failed: "
                           + identity_error)
                    << '\n';
            }
            amrex::Abort(
                "Material table identity capture failed on at least one MPI "
                "rank; no table was consumed.");
        }

        MaterialTableIdentityContract contract;
        contract.selection_contract =
            selection_contract(m_selector, opacity_interpolation);
        contract.tables = std::move(fingerprints);
        MaterialTableIdentity::Install(std::move(contract));
        MaterialTableIdentity::ValidateCurrentAcrossRanks();

        std::string restart;
        amrex::ParmParse const pp_amr("amr");
        pp_amr.query("restart", restart);
        if (!restart.empty()) {
            MaterialTableIdentity::ValidateRestart(restart);
        }
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
        return normalize_path(lhs.logical_file())
                == normalize_path(rhs.logical_file())
            && lhs.material_id == rhs.material_id
            && lhs.material_key == rhs.material_key;
    }
}
