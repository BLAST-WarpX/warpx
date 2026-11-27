/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_UTILS_MATERIALREGISTRY_H_
#define WARPX_UTILS_MATERIALREGISTRY_H_

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <cstdint>
#include <vector>

class MultiParticleContainer;

namespace warpx::materials
{
    /** Immutable reference to one externally owned material table. */
    struct MaterialTableHandle
    {
        MaterialTableHandle () = default;
        MaterialTableHandle (
            std::string file_in,
            std::int64_t material_id_in,
            std::string material_key_in)
            : file(std::move(file_in)),
              source_file(file),
              material_id(material_id_in),
              material_key(std::move(material_key_in))
        {}

        /** Path consumed by the table loader. */
        std::string file;
        /** User-configured logical path. */
        std::string source_file;
        /** Exact-byte staging path supplied to the table loader. */
        std::string load_file;
        std::int64_t material_id = -1;
        std::string material_key;
        std::string content_sha256;
        std::uint64_t content_size = 0;

        [[nodiscard]] bool configured () const noexcept
        {
            return !file.empty();
        }

        [[nodiscard]] std::string logical_file () const
        {
            return source_file.empty() ? file : source_file;
        }
    };

    /** Host-side identity and metadata shared by material-physics adapters. */
    struct MaterialDefinition
    {
        std::string name;
        std::vector<std::string> species;
        std::optional<MaterialTableHandle> electron_eos;
        std::optional<MaterialTableHandle> opacity;
    };

    /**
     * Opt-in named-material registry.
     *
     * Definitions are sorted by name so their indices are independent of input
     * order. Particle validation is deliberately separate from parsing, which
     * keeps the configuration and cell-classification invariants unit-testable.
     */
    class MaterialRegistry
    {
    public:
        static constexpr int max_materials = 8;

        enum class ResolvedCell : int
        {
            Invalid = -3,
            Mixed = -2,
            Vacuum = -1
        };

        /** Device-safe resolved-single-material cell classifier.
         *
         * A non-vacuum cell resolves to its largest mass-density component
         * only when the sum of every other component is no larger than the
         * configured absolute-or-relative mixed-cell tolerance. Thus, no
         * additive EOS or opacity mixture is implied by a successful result.
         */
        struct ResolvedCellSelector
        {
            int num_materials = 0;
            amrex::Real vacuum_mass_density = amrex::Real(0.0);
            amrex::Real mixed_cell_absolute_tolerance = amrex::Real(0.0);
            amrex::Real mixed_cell_relative_tolerance = amrex::Real(1.0e-12);

            [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
            int operator() (
                amrex::GpuArray<amrex::Real, max_materials> const&
                    material_mass_densities) const noexcept
            {
                if (num_materials < 0 || num_materials > max_materials
                    || !amrex::Math::isfinite(vacuum_mass_density)
                    || vacuum_mass_density < amrex::Real(0.0)
                    || !amrex::Math::isfinite(
                        mixed_cell_absolute_tolerance)
                    || mixed_cell_absolute_tolerance < amrex::Real(0.0)
                    || !amrex::Math::isfinite(
                        mixed_cell_relative_tolerance)
                    || mixed_cell_relative_tolerance < amrex::Real(0.0)
                    || mixed_cell_relative_tolerance >= amrex::Real(0.5))
                {
                    return static_cast<int>(ResolvedCell::Invalid);
                }

                auto total_density = amrex::Real(0.0);
                auto dominant_density = amrex::Real(-1.0);
                int dominant_material = static_cast<int>(ResolvedCell::Vacuum);
                bool dominant_is_tied = false;
                for (int material = 0; material < num_materials; ++material) {
                    amrex::Real const density =
                        material_mass_densities[material];
                    if (!amrex::Math::isfinite(density)
                        || density < amrex::Real(0.0))
                    {
                        return static_cast<int>(ResolvedCell::Invalid);
                    }
                    total_density += density;
                    if (!amrex::Math::isfinite(total_density)) {
                        return static_cast<int>(ResolvedCell::Invalid);
                    }
                    if (density > dominant_density) {
                        dominant_density = density;
                        dominant_material = material;
                        dominant_is_tied = false;
                    } else if (density == dominant_density) {
                        dominant_is_tied = true;
                    }
                }

                if (total_density <= vacuum_mass_density) {
                    return static_cast<int>(ResolvedCell::Vacuum);
                }
                if (dominant_material < 0) {
                    return static_cast<int>(ResolvedCell::Invalid);
                }
                if (dominant_is_tied) {
                    return static_cast<int>(ResolvedCell::Mixed);
                }

                amrex::Real const contaminant_density =
                    total_density - dominant_density;
                amrex::Real const mixed_cell_tolerance =
                    mixed_cell_absolute_tolerance
                        > mixed_cell_relative_tolerance * total_density
                    ? mixed_cell_absolute_tolerance
                    : mixed_cell_relative_tolerance * total_density;
                return contaminant_density <= mixed_cell_tolerance
                    ? dominant_material
                    : static_cast<int>(ResolvedCell::Mixed);
            }
        };

        MaterialRegistry () = default;
        ~MaterialRegistry ();

        MaterialRegistry (MaterialRegistry const&) = delete;
        MaterialRegistry& operator= (MaterialRegistry const&) = delete;
        MaterialRegistry (MaterialRegistry&&) = delete;
        MaterialRegistry& operator= (MaterialRegistry&&) = delete;

        /** Read ``materials.*`` inputs. An absent names list disables it. */
        void ReadParameters ();

        /** Validate exclusive fixed-charge particle-carrier membership. */
        void ValidateSpecies (MultiParticleContainer const& particles) const;

        [[nodiscard]] bool enabled () const noexcept
        {
            return !m_materials.empty();
        }

        [[nodiscard]] int size () const noexcept
        {
            return static_cast<int>(m_materials.size());
        }

        [[nodiscard]] MaterialDefinition const& material (int index) const
        {
            return m_materials.at(static_cast<std::size_t>(index));
        }

        [[nodiscard]] MaterialDefinition const* findMaterial (
            std::string const& name) const noexcept;

        [[nodiscard]] MaterialDefinition const* findMaterialForSpecies (
            std::string const& species) const noexcept;

        /** Return a device-safe copy of the configured cell selector. */
        [[nodiscard]] ResolvedCellSelector selector () const noexcept
        {
            return m_selector;
        }

        /**
         * Return the one resolved material index, or a negative ResolvedCell
         * sentinel, using the configured selector contract.
         */
        [[nodiscard]] int classifyResolvedCell (
            std::span<amrex::Real const> material_mass_densities) const noexcept;

        [[nodiscard]] amrex::Real mixedCellRelativeTolerance () const noexcept
        {
            return m_selector.mixed_cell_relative_tolerance;
        }

        /** Compare normalized file names and exact material metadata. */
        [[nodiscard]] static bool sameTableHandle (
            MaterialTableHandle const& lhs,
            MaterialTableHandle const& rhs) noexcept;

    private:
        std::vector<MaterialDefinition> m_materials;
        ResolvedCellSelector m_selector;
        std::vector<std::string> m_staged_table_files;
    };

    static_assert(
        std::is_trivially_copyable_v<MaterialRegistry::ResolvedCellSelector>);
}

#endif // WARPX_UTILS_MATERIALREGISTRY_H_
