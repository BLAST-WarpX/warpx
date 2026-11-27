/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_OPACITYEVALUATOR_H_
#define WARPX_RADIATION_OPACITYEVALUATOR_H_

#include "EnergyGroups.H"
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
#   include "MaterialOpacityTable.H"
#   include "Utils/MaterialRegistry.H"
#endif
#include "OpacityTable.H"

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

namespace warpx::radiation::detail
{
using namespace amrex::literals;

#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
/** Evaluate one fixed-composition material table from the total ion mass density.
 *
 * The table already represents the complete mixture. In particular, its
 * Rosseland mean is not added species-by-species. Packets are assigned to a
 * discrete group from their actual photon energy; using that group's Planck
 * mean for packet attenuation is an intentional group-mean approximation.
 */
template <unsigned int N>
struct MaterialOpacityEvaluator
{
    bool enabled = false;
    int num_species = 0;
    amrex::GpuArray<amrex::Real, N> species_masses{};
    warpx::radiation::MaterialOpacityTableExecutor table;
    warpx::radiation::EnergyGroupsExecutor energy_groups;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    warpx::radiation::MaterialOpacityCoefficients operator() (
        amrex::GpuArray<amrex::Real, N> const& number_densities,
        amrex::Real const photon_energy,
        amrex::Real const electron_temperature) const noexcept
    {
        if (photon_energy < 0.0_rt
            || !amrex::Math::isfinite(photon_energy))
        {
            return {-1.0_rt, -1.0_rt, -1.0_rt, -1.0_rt};
        }
        amrex::Real mass_density = 0.0_rt;
        for (int species = 0; species < num_species; ++species) {
            amrex::Real const number_density = number_densities[species];
            if (number_density < 0.0_rt
                || !amrex::Math::isfinite(number_density))
            {
                return {-1.0_rt, -1.0_rt, -1.0_rt, -1.0_rt};
            }
            mass_density += number_density * species_masses[species];
        }
        if (!amrex::Math::isfinite(mass_density)) {
            return {-1.0_rt, -1.0_rt, -1.0_rt, -1.0_rt};
        }

        int const group = energy_groups.index(photon_energy);
        return table(mass_density, electron_temperature, group);
    }
};

/** Select one pure-material table from registry-resolved mass densities.
 *
 * Carrier number densities are grouped by immutable material identity.  A
 * mixed or otherwise invalid cell returns invalid coefficients; callers must
 * nevertheless reject such owned cells in the preflight pass before this
 * evaluator can participate in any radiation or material mutation.
 */
template <unsigned int N, unsigned int M>
struct RegisteredMaterialOpacityEvaluator
{
    bool enabled = false;
    int num_species = 0;
    int num_materials = 0;
    amrex::GpuArray<amrex::Real, N> species_masses{};
    amrex::GpuArray<int, N> species_material_indices{};
    amrex::GpuArray<warpx::radiation::MaterialOpacityTableExecutor, M> tables{};
    warpx::materials::MaterialRegistry::ResolvedCellSelector selector;
    warpx::radiation::EnergyGroupsExecutor energy_groups;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int resolvedMaterial (
        amrex::GpuArray<amrex::Real, N> const& number_densities,
        amrex::GpuArray<amrex::Real, M>& material_mass_densities) const noexcept
    {
        if (num_species < 0 || num_species > static_cast<int>(N)
            || num_materials < 0
            || num_materials > static_cast<int>(M)
            || num_materials > warpx::materials::MaterialRegistry::max_materials)
        {
            return static_cast<int>(
                warpx::materials::MaterialRegistry::ResolvedCell::Invalid);
        }
        for (int species = 0; species < num_species; ++species) {
            amrex::Real const number_density = number_densities[species];
            int const material = species_material_indices[species];
            if (number_density < 0.0_rt
                || !amrex::Math::isfinite(number_density)
                || material < 0 || material >= num_materials)
            {
                return static_cast<int>(
                    warpx::materials::MaterialRegistry::ResolvedCell::Invalid);
            }
            material_mass_densities[material] +=
                number_density * species_masses[species];
            if (!amrex::Math::isfinite(material_mass_densities[material])) {
                return static_cast<int>(
                    warpx::materials::MaterialRegistry::ResolvedCell::Invalid);
            }
        }
        amrex::GpuArray<
            amrex::Real, warpx::materials::MaterialRegistry::max_materials>
            selector_densities{};
        for (int material = 0; material < num_materials; ++material) {
            selector_densities[material] = material_mass_densities[material];
        }
        return selector(selector_densities);
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    warpx::radiation::MaterialOpacityCoefficients operator() (
        amrex::GpuArray<amrex::Real, N> const& number_densities,
        amrex::Real const photon_energy,
        amrex::Real const electron_temperature) const noexcept
    {
        if (photon_energy < 0.0_rt
            || !amrex::Math::isfinite(photon_energy))
        {
            return {-1.0_rt, -1.0_rt, -1.0_rt, -1.0_rt};
        }
        amrex::GpuArray<amrex::Real, M> material_mass_densities{};
        int const material =
            resolvedMaterial(number_densities, material_mass_densities);
        if (material == static_cast<int>(
                warpx::materials::MaterialRegistry::ResolvedCell::Vacuum))
        {
            return {};
        }
        if (material < 0 || material >= num_materials) {
            return {-1.0_rt, -1.0_rt, -1.0_rt, -1.0_rt};
        }
        int const group = energy_groups.index(photon_energy);
        return tables[material](
            material_mass_densities[material], electron_temperature, group);
    }
};
#endif

/** Sum non-negative partial extinction coefficients for the local mixture. */
enum class SpeciesOpacityBackend : int
{
    Parser,
    Table2D,
    Table3D
};

template <unsigned int N>
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
SpeciesOpacity (
    amrex::GpuArray<amrex::ParserExecutor<8>, N> const& coefficients,
    amrex::GpuArray<int, N> const& backends,
    amrex::GpuArray<
        warpx::radiation::OpacityTable2DExecutor, N> const& tables,
    amrex::GpuArray<
        warpx::radiation::OpacityTable3DExecutor, N> const& spectral_tables,
    int const num_species,
    amrex::GpuArray<amrex::Real, N> const& number_densities,
    amrex::Real const x,
    amrex::Real const y,
    amrex::Real const z,
    amrex::Real const t,
    amrex::Real const photon_energy,
    amrex::Real const electron_density,
    amrex::Real const electron_temperature) noexcept
{
    amrex::Real total = 0.0_rt;
    for (int species = 0; species < num_species; ++species) {
        amrex::Real const ion_density = number_densities[species];
        if (ion_density < 0.0_rt || !amrex::Math::isfinite(ion_density)) {
            return -1.0_rt;
        }
        if (ion_density == 0.0_rt) { continue; }
        amrex::Real partial = -1.0_rt;
        int const backend = backends[species];
        if (backend == static_cast<int>(SpeciesOpacityBackend::Parser)) {
            partial = coefficients[species](
                x, y, z, t, photon_energy, ion_density,
                electron_density, electron_temperature);
        } else if (
            backend == static_cast<int>(SpeciesOpacityBackend::Table2D))
        {
            partial = tables[species](ion_density, electron_temperature);
        } else if (
            backend == static_cast<int>(SpeciesOpacityBackend::Table3D))
        {
            partial = spectral_tables[species](
                ion_density, electron_temperature, photon_energy);
        }
        if (partial < 0.0_rt || !amrex::Math::isfinite(partial)) {
            return -1.0_rt;
        }
        total += partial;
    }
    return total;
}

/** Device-callable selector for one complete opacity channel. */
template <unsigned int N>
struct OpacityEvaluator
{
    bool use_species = false;
    int num_species = 0;
    amrex::GpuArray<amrex::ParserExecutor<8>, N> species_coefficients;
    amrex::GpuArray<int, N> species_backends;
    amrex::GpuArray<
        warpx::radiation::OpacityTable2DExecutor, N> species_tables;
    amrex::GpuArray<
        warpx::radiation::OpacityTable3DExecutor, N> species_spectral_tables;
    bool use_table = false;
    bool use_spectral_table = false;
    warpx::radiation::OpacityTable2DExecutor table;
    warpx::radiation::OpacityTable3DExecutor spectral_table;
    amrex::ParserExecutor<7> coefficient;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator() (
        amrex::GpuArray<amrex::Real, N> const& number_densities,
        amrex::Real const x,
        amrex::Real const y,
        amrex::Real const z,
        amrex::Real const t,
        amrex::Real const photon_energy,
        amrex::Real const electron_density,
        amrex::Real const electron_temperature) const noexcept
    {
        if (use_species) {
            return SpeciesOpacity(
                species_coefficients, species_backends, species_tables,
                species_spectral_tables, num_species, number_densities,
                x, y, z, t, photon_energy,
                electron_density, electron_temperature);
        }
        if (use_spectral_table) {
            return spectral_table(
                electron_density, electron_temperature, photon_energy);
        }
        if (use_table) {
            return table(electron_density, electron_temperature);
        }
        return coefficient(
            x, y, z, t, photon_energy,
            electron_density, electron_temperature);
    }
};
} // namespace warpx::radiation::detail

#endif
