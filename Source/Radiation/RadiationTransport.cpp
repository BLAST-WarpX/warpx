/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "RadiationTransport.H"
#include "PlanckExchange.H"

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "Particles/Algorithms/KineticEnergy.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhotonParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/UpdatePosition.H"
#include "Particles/SpeciesPhysicalProperties.H"
#include "Particles/WarpXParticleContainer.H"
#include "RadialFaceMarching.H"
#include "RadiationEnergyUpdate.H"
#include "RadiationKineticEnergyUpdate.H"
#include "Utils/MaterialRegistry.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/Communication.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_Array4.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_FabArrayUtility.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuControl.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Reduce.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace amrex::literals;
using warpx::fields::FieldType;
using warpx::radiation::EvaluatePlanckExchange;
using warpx::radiation::PlanckExchangeResult;
using warpx::radiation::ApplyRadiationEnergyUpdate;
using warpx::radiation::RadiationEnergyUpdateResult;
using warpx::radiation::EvaluateKineticEnergyUpdate;
using warpx::radiation::KineticEnergyUpdateResult;
using warpx::radiation::KineticPrecisionEpsilon;

namespace
{
int
ParseDiffusionBoundary (std::string const& name, std::string const& key)
{
    if (name == "reflecting" || name == "zero_flux") {
        return static_cast<int>(RadiationTransport::DiffusionBoundary::Reflecting);
    }
    if (name == "vacuum" || name == "free_streaming") {
        return static_cast<int>(RadiationTransport::DiffusionBoundary::Vacuum);
    }
    if (name == "marshak") {
        return static_cast<int>(RadiationTransport::DiffusionBoundary::Marshak);
    }
    WARPX_ABORT_WITH_MESSAGE(
        "Unknown " + key + "='" + name + "'. Valid values are reflecting "
        "(zero normal flux), vacuum (free-streaming F=cE), and marshak "
        "(P1 vacuum F=cE/2). A zero-energy Dirichlet face is not offered: "
        "in the diffusion limit it implies a mesh- and opacity-dependent "
        "flux that can exceed cE and destablize the explicit update.");
    return static_cast<int>(RadiationTransport::DiffusionBoundary::Reflecting);
}

void
ParseDiffusionBoundaryArray (
    amrex::ParmParse const& pp,
    char const* key,
    amrex::GpuArray<int, 3>& boundary)
{
    std::vector<std::string> names;
    if (!pp.queryarr(key, names) || names.empty()) { return; }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        names.size() == 1 || names.size() == AMREX_SPACEDIM,
        std::string("radiation_transport.") + key
            + " must contain one value or one value per dimension.");
    if (names.size() == 1) {
        int const value = ParseDiffusionBoundary(names[0], key);
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            boundary[direction] = value;
        }
        return;
    }
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
        boundary[direction] = ParseDiffusionBoundary(names[direction], key);
    }
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
DiffusionEscapeFactor (int const boundary) noexcept
{
    if (boundary == static_cast<int>(
            RadiationTransport::DiffusionBoundary::Marshak))
    {
        return 0.5_rt;
    }
    if (boundary == static_cast<int>(
            RadiationTransport::DiffusionBoundary::Vacuum))
    {
        return 1.0_rt;
    }
    return 0.0_rt;
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
DiffusionEscapeMomentumFactor (int const boundary) noexcept
{
    if (boundary == static_cast<int>(
            RadiationTransport::DiffusionBoundary::Marshak))
    {
        // The P1 half-isotropic boundary has F=cE/2 and normal pressure E/3.
        return 1.0_rt / 3.0_rt;
    }
    if (boundary == static_cast<int>(
            RadiationTransport::DiffusionBoundary::Vacuum))
    {
        return 1.0_rt;
    }
    return 0.0_rt;
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int
RadiationMomentumComponent (int const direction) noexcept
{
#if defined(WARPX_DIM_1D_Z)
    return direction == 0 ? 2 : direction;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    return direction == 1 ? 2 : direction;
#else
    return direction;
#endif
}

#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
[[nodiscard]] bool
OpacityGroupValuesAgree (
    amrex::Real const configured, amrex::Real const tabulated) noexcept
{
    if (configured == tabulated) { return true; }
    if (!amrex::Math::isfinite(configured)
        || !amrex::Math::isfinite(tabulated))
    {
        return false;
    }
    amrex::Real const scale = amrex::max(
        std::abs(configured), std::abs(tabulated));
    return std::abs(configured - tabulated)
        <= 64.0_rt * std::numeric_limits<amrex::Real>::epsilon() * scale;
}

[[nodiscard]] bool
OpacityGroupArraysAgree (
    amrex::Vector<amrex::Real> const& first,
    amrex::Vector<amrex::Real> const& second) noexcept
{
    return first.size() == second.size()
        && std::equal(
            first.begin(), first.end(), second.begin(),
            [] (amrex::Real const lhs, amrex::Real const rhs) noexcept
            { return OpacityGroupValuesAgree(lhs, rhs); });
}

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
        return selector(material_mass_densities);
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

struct HybridCellMaterialState
{
    amrex::Real electron_density = 0.0_rt;
    amrex::Real electron_temperature = 0.0_rt;
    amrex::Real internal_energy = 0.0_rt;
    amrex::Real heat_capacity = 0.0_rt;
    amrex::Real available_energy = 0.0_rt;
    bool valid = true;
};

/** Physical part of one cell represented by one of its nodal corners. */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
HybridCellCornerVolume (
    int const i,
    int const j,
    int const k,
    int const di,
    int const dj,
    int const dk,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& cell_size,
    amrex::Dim3 const domain_lo) noexcept
{
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE) \
    || defined(WARPX_DIM_RZ)
    amrex::Real const r_lo =
        problo[0] + (i - domain_lo.x) * cell_size[0];
    amrex::Real const r_mid = r_lo + 0.5_rt * cell_size[0];
    amrex::Real const r_hi = r_lo + cell_size[0];
    amrex::Real const corner_r_lo = di == 0 ? r_lo : r_mid;
    amrex::Real const corner_r_hi = di == 0 ? r_mid : r_hi;
#endif
#if defined(WARPX_DIM_RCYLINDER)
    amrex::ignore_unused(j, k, dj, dk);
    return MathConst::pi
        * (corner_r_hi * corner_r_hi - corner_r_lo * corner_r_lo);
#elif defined(WARPX_DIM_RSPHERE)
    amrex::ignore_unused(j, k, dj, dk);
    return 4.0_rt / 3.0_rt * MathConst::pi
        * (corner_r_hi * corner_r_hi * corner_r_hi
           - corner_r_lo * corner_r_lo * corner_r_lo);
#elif defined(WARPX_DIM_RZ)
    amrex::ignore_unused(j, k, dj, dk);
    return MathConst::pi
        * (corner_r_hi * corner_r_hi - corner_r_lo * corner_r_lo)
        * 0.5_rt * cell_size[1];
#else
    amrex::ignore_unused(i, j, k, di, dj, dk, problo, domain_lo);
    return AMREX_D_TERM(
        0.5_rt * cell_size[0],
        * 0.5_rt * cell_size[1],
        * 0.5_rt * cell_size[2]);
#endif
}

/** Fixed-size old nodal state used by a nonlinear cell LTE solve. */
struct HybridNonlinearCellState
{
    static constexpr int max_corners = 1 << AMREX_SPACEDIM;
    HybridCellMaterialState material;
    int num_active_corners = 0;
    amrex::GpuArray<int, max_corners> active{};
    amrex::GpuArray<amrex::Real, max_corners> charge_density{};
    amrex::GpuArray<amrex::Real, max_corners> temperature{};
    amrex::GpuArray<amrex::Real, max_corners> corner_volume{};
    amrex::GpuArray<amrex::Real, max_corners> internal_energy_density{};
    amrex::GpuArray<
        ElectronThermodynamicsExecutor::MaterialMassDensities,
        max_corners> material_mass_density{};
};

/** Gather the complete old nodal caloric state for a nonlinear cell solve. */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HybridNonlinearCellState
GatherHybridNonlinearCellState (
    int const i,
    int const j,
    int const k,
    amrex::Real const material_density_threshold,
    amrex::Real const thermodynamic_density_floor,
    ElectronThermodynamicsExecutor const thermodynamics,
    amrex::Array4<amrex::Real const> const& rho,
    amrex::Array4<amrex::Real const> const& temperature,
    ElectronThermodynamicsExecutor::MaterialChargeDensityArrays const&
        material_charge_density,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& cell_size,
    amrex::Dim3 const domain_lo) noexcept
{
    HybridNonlinearCellState result;
    amrex::Real density_sum = 0.0_rt;
    amrex::Real internal_energy_sum = 0.0_rt;
    amrex::Real heat_capacity_sum = 0.0_rt;
    amrex::Real capacity_temperature_sum = 0.0_rt;
    amrex::Real available_energy_sum = 0.0_rt;

    for (int dk = 0; dk < (AMREX_SPACEDIM == 3 ? 2 : 1); ++dk) {
        for (int dj = 0; dj < (AMREX_SPACEDIM >= 2 ? 2 : 1); ++dj) {
            for (int di = 0; di < 2; ++di) {
                int const corner = di + 2 * (dj + 2 * dk);
                int const ni = i + di;
                int const nj = j + dj;
                int const nk = k + dk;
                amrex::Real const node_density = rho(ni, nj, nk);
                density_sum += node_density;
                if (node_density <=
                    material_density_threshold * PhysConst::q_e)
                {
                    continue;
                }

                amrex::Real const node_temperature = temperature(ni, nj, nk);
                amrex::Real const volume = HybridCellCornerVolume(
                    i, j, k, di, dj, dk, problo, cell_size, domain_lo);
                auto const mass_density = thermodynamics
                    .materialMassDensitiesFromChargeDensityArrays(
                        material_charge_density, ni, nj, nk);
                amrex::Real const eos_charge_density = amrex::max(
                    node_density, thermodynamic_density_floor * PhysConst::q_e);
                ElectronThermodynamicState const node_state = thermodynamics
                    .stateFromMaterialMassDensitiesTemperature(
                        eos_charge_density, mass_density, node_temperature);
                ElectronThermodynamicState const minimum_node_state =
                    thermodynamics.stateFromMaterialMassDensitiesTemperature(
                        eos_charge_density, mass_density,
                        thermodynamics.minimumTemperature());
                amrex::Real const available_node_energy_density =
                    node_state.internal_energy_density
                    - minimum_node_state.internal_energy_density;
                amrex::Real const energy_tolerance = 256.0_rt
                    * std::numeric_limits<amrex::Real>::epsilon()
                    * amrex::max(
                        amrex::max(std::abs(available_node_energy_density),
                            node_state.heat_capacity_density * amrex::max(
                                std::abs(node_temperature), 1.0_rt)),
                        std::numeric_limits<amrex::Real>::min());
                if (!(volume > 0.0_rt)
                    || !amrex::Math::isfinite(volume)
                    || node_temperature < thermodynamics.minimumTemperature()
                    || node_temperature > thermodynamics.maximumTemperature()
                    || !amrex::Math::isfinite(node_temperature)
                    || node_state.pressure < 0.0_rt
                    || !(node_state.heat_capacity_density > 0.0_rt)
                    || !amrex::Math::isfinite(node_state.pressure)
                    || !amrex::Math::isfinite(
                        node_state.internal_energy_density)
                    || !amrex::Math::isfinite(
                        node_state.heat_capacity_density)
                    || !amrex::Math::isfinite(
                        minimum_node_state.internal_energy_density)
                    || available_node_energy_density < -energy_tolerance)
                {
                    result.material.valid = false;
                    continue;
                }

                result.active[corner] = 1;
                result.charge_density[corner] = eos_charge_density;
                result.temperature[corner] = node_temperature;
                result.corner_volume[corner] = volume;
                result.internal_energy_density[corner] =
                    node_state.internal_energy_density;
                result.material_mass_density[corner] = mass_density;
                ++result.num_active_corners;
                internal_energy_sum +=
                    node_state.internal_energy_density * volume;
                heat_capacity_sum +=
                    node_state.heat_capacity_density * volume;
                capacity_temperature_sum +=
                    node_state.heat_capacity_density * volume
                    * node_temperature;
                available_energy_sum +=
                    amrex::max(available_node_energy_density, 0.0_rt) * volume;
            }
        }
    }

    result.material.electron_density = density_sum
        / (HybridNonlinearCellState::max_corners * PhysConst::q_e);
    if (heat_capacity_sum > 0.0_rt) {
        result.material.electron_temperature =
            capacity_temperature_sum / heat_capacity_sum;
        result.material.internal_energy = internal_energy_sum;
        result.material.heat_capacity = heat_capacity_sum;
        result.material.available_energy = available_energy_sum;
    }
    return result;
}

/** Evaluate the corner-quadrature material-energy change after one delta-T.
 *
 * Forming the difference per corner makes the nonlinear residual independent
 * of an EOS table's arbitrary absolute internal-energy reference.  Summing two
 * absolute cell energies before subtracting would unnecessarily amplify
 * cancellation when that reference is large.
 */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
EvaluateHybridNonlinearMaterialEnergyChange (
    HybridNonlinearCellState const& state,
    ElectronThermodynamicsExecutor const thermodynamics,
    amrex::Real const temperature_increment,
    bool& valid) noexcept
{
    amrex::Real material_energy_change = 0.0_rt;
    for (int corner = 0;
         corner < HybridNonlinearCellState::max_corners; ++corner)
    {
        if (state.active[corner] == 0) { continue; }
        amrex::Real const trial_temperature =
            state.temperature[corner] + temperature_increment;
        ElectronThermodynamicState const trial_state = thermodynamics
            .stateFromMaterialMassDensitiesTemperature(
                state.charge_density[corner],
                state.material_mass_density[corner], trial_temperature);
        if (trial_temperature < thermodynamics.minimumTemperature()
            || trial_temperature > thermodynamics.maximumTemperature()
            || !amrex::Math::isfinite(trial_temperature)
            || trial_state.pressure < 0.0_rt
            || !(trial_state.heat_capacity_density > 0.0_rt)
            || !amrex::Math::isfinite(trial_state.pressure)
            || !amrex::Math::isfinite(
                trial_state.internal_energy_density)
            || !amrex::Math::isfinite(
                trial_state.heat_capacity_density))
        {
            valid = false;
            return 0.0_rt;
        }
        material_energy_change += (
            trial_state.internal_energy_density
            - state.internal_energy_density[corner])
            * state.corner_volume[corner];
    }
    valid = valid && amrex::Math::isfinite(material_energy_change);
    return material_energy_change;
}

/** Gather the nodal hybrid-electron state onto one radiation cell. */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HybridCellMaterialState
GatherHybridCellMaterialState (
    int const i,
    int const j,
    int const k,
    amrex::Real const material_density_threshold,
    amrex::Real const thermodynamic_density_floor,
    ElectronThermodynamicsExecutor const thermodynamics,
    amrex::Array4<amrex::Real const> const& rho,
    amrex::Array4<amrex::Real const> const& temperature,
    ElectronThermodynamicsExecutor::MaterialChargeDensityArrays const&
        material_charge_density,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& cell_size,
    amrex::Dim3 const domain_lo) noexcept
{
    constexpr int max_adjacent_cells = 1 << AMREX_SPACEDIM;
    amrex::Real density_sum = 0.0_rt;
    amrex::Real internal_energy_sum = 0.0_rt;
    amrex::Real heat_capacity_sum = 0.0_rt;
    amrex::Real capacity_temperature_sum = 0.0_rt;
    amrex::Real minimum_available_energy_per_capacity =
        std::numeric_limits<amrex::Real>::max();
    HybridCellMaterialState state;

    for (int dk = 0; dk < (AMREX_SPACEDIM == 3 ? 2 : 1); ++dk) {
        for (int dj = 0; dj < (AMREX_SPACEDIM >= 2 ? 2 : 1); ++dj) {
            for (int di = 0; di < 2; ++di) {
                int const ni = i + di;
                int const nj = j + dj;
                int const nk = k + dk;
                amrex::Real const node_density = rho(ni, nj, nk);
                density_sum += node_density;
                if (node_density <=
                    material_density_threshold * PhysConst::q_e) {
                    continue;
                }

                amrex::Real const node_volume = HybridCellCornerVolume(
                    i, j, k, di, dj, dk, problo, cell_size, domain_lo);
                amrex::Real const node_temperature = temperature(ni, nj, nk);
                if (node_temperature < thermodynamics.minimumTemperature()
                    || !amrex::Math::isfinite(node_temperature))
                {
                    state.valid = false;
                    continue;
                }
                auto const material_mass_density = thermodynamics
                    .materialMassDensitiesFromChargeDensityArrays(
                        material_charge_density, ni, nj, nk);
                ElectronThermodynamicState const node_state = thermodynamics
                    .stateFromMaterialMassDensitiesTemperature(
                        amrex::max(node_density, thermodynamic_density_floor *
                                                     PhysConst::q_e),
                        material_mass_density, node_temperature);
                ElectronThermodynamicState const minimum_node_state =
                    thermodynamics.stateFromMaterialMassDensitiesTemperature(
                        amrex::max(node_density, thermodynamic_density_floor *
                                                     PhysConst::q_e),
                        material_mass_density,
                        thermodynamics.minimumTemperature());
                amrex::Real const available_node_energy_density =
                    node_state.internal_energy_density
                    - minimum_node_state.internal_energy_density;
                amrex::Real const energy_tolerance = 256.0_rt
                    * std::numeric_limits<amrex::Real>::epsilon()
                    * amrex::max(
                        amrex::max(std::abs(available_node_energy_density),
                            node_state.heat_capacity_density * amrex::max(
                                std::abs(node_temperature), 1.0_rt)),
                        std::numeric_limits<amrex::Real>::min());
                if (node_state.pressure < 0.0_rt ||
                    !(node_state.heat_capacity_density > 0.0_rt) ||
                    !amrex::Math::isfinite(node_state.pressure) ||
                    !amrex::Math::isfinite(
                        node_state.internal_energy_density) ||
                    !amrex::Math::isfinite(node_state.heat_capacity_density) ||
                    !amrex::Math::isfinite(
                        minimum_node_state.internal_energy_density) ||
                    available_node_energy_density < -energy_tolerance) {
                    state.valid = false;
                    continue;
                }
                amrex::Real const node_heat_capacity =
                    node_state.heat_capacity_density * node_volume;
                amrex::Real const node_internal_energy =
                    node_state.internal_energy_density * node_volume;
                internal_energy_sum += node_internal_energy;
                heat_capacity_sum += node_heat_capacity;
                capacity_temperature_sum +=
                    node_heat_capacity * node_temperature;
                minimum_available_energy_per_capacity = amrex::min(
                    minimum_available_energy_per_capacity,
                    amrex::max(available_node_energy_density, 0.0_rt)
                        * node_volume / node_heat_capacity);
            }
        }
    }

    state.electron_density =
        density_sum / (max_adjacent_cells * PhysConst::q_e);
    if (heat_capacity_sum > 0.0_rt) {
        // Match the ideal-electron heat-capacity weights used by the inverse
        // cell-to-node source map. Cap emission before any participating node
        // reaches zero electron temperature.
        state.electron_temperature =
            capacity_temperature_sum / heat_capacity_sum;
        state.internal_energy = internal_energy_sum;
        state.heat_capacity = heat_capacity_sum;
        state.available_energy = heat_capacity_sum
            * minimum_available_energy_per_capacity;
    }
    return state;
}

template <unsigned int N> struct ImplicitLteCellContext {
    bool enable_diffusion = false;
    int num_groups = 1;
    int max_iterations = 64;
    amrex::Real tolerance = 1.0e-10_rt;
    amrex::Real current_time = 0.0_rt;
    amrex::Real dt = 0.0_rt;
    OpacityEvaluator<N> planck_evaluator;
    OpacityEvaluator<N> rosseland_evaluator;
    warpx::radiation::EnergyGroupsExecutor energy_groups;
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
    MaterialOpacityEvaluator<N> material_evaluator;
    RegisteredMaterialOpacityEvaluator<
        N, warpx::materials::MaterialRegistry::max_materials>
        registered_material_evaluator;
#endif

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real planckAbsorption (
        amrex::GpuArray<amrex::Real, N> const& number_densities,
        amrex::Real const x, amrex::Real const y, amrex::Real const z,
        amrex::Real const photon_energy, amrex::Real const electron_density,
        amrex::Real const electron_temperature) const noexcept
    {
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
        if (registered_material_evaluator.enabled) {
            return registered_material_evaluator(
                number_densities, photon_energy, electron_temperature)
                    .planck_absorption;
        }
        if (material_evaluator.enabled) {
            // Only true absorption heats matter. The table's scattering
            // channel is not sampled and deposits no event momentum.
            return material_evaluator(
                number_densities, photon_energy, electron_temperature)
                    .planck_absorption;
        }
#endif
        return planck_evaluator(
            number_densities, x, y, z, current_time, photon_energy,
            electron_density, electron_temperature);
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real planckEmission (
        amrex::GpuArray<amrex::Real, N> const& number_densities,
        amrex::Real const x, amrex::Real const y, amrex::Real const z,
        amrex::Real const photon_energy, amrex::Real const electron_density,
        amrex::Real const electron_temperature) const noexcept
    {
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
        if (registered_material_evaluator.enabled) {
            return registered_material_evaluator(
                number_densities, photon_energy, electron_temperature)
                    .planck_emission;
        }
        if (material_evaluator.enabled) {
            return material_evaluator(
                number_densities, photon_energy, electron_temperature)
                    .planck_emission;
        }
#endif
        // Legacy analytic and additive-species backends obey Kirchhoff's law.
        // Schema-1 native material tables are LTE and canonicalize emission to
        // the absorption mean during loading.
        return planck_evaluator(
            number_densities, x, y, z, current_time, photon_energy,
            electron_density, electron_temperature);
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real rosselandOpacity (
        amrex::GpuArray<amrex::Real, N> const& number_densities,
        amrex::Real const x, amrex::Real const y, amrex::Real const z,
        amrex::Real const photon_energy, amrex::Real const electron_density,
        amrex::Real const electron_temperature) const noexcept
    {
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
        if (registered_material_evaluator.enabled) {
            return registered_material_evaluator(
                number_densities, photon_energy, electron_temperature)
                    .rosseland_transport;
        }
        if (material_evaluator.enabled) {
            return material_evaluator(
                number_densities, photon_energy, electron_temperature)
                    .rosseland_transport;
        }
#endif
        return rosseland_evaluator(
            number_densities, x, y, z, current_time, photon_energy,
            electron_density, electron_temperature);
    }
};

constexpr int nonlinear_lte_temperature_increment_comp = 0;
constexpr int nonlinear_lte_represented_energy_comp = 1;
constexpr int nonlinear_lte_remap_components = 2;

struct HybridNonlinearLteTrial
{
    amrex::Real material_energy_change = 0.0_rt;
    amrex::Real radiation_exchange = 0.0_rt;
    amrex::Real residual = 0.0_rt;
    amrex::Real radiation_temperature = 0.0_rt;
    bool valid = true;
};

/** Evaluate the conservative nonlinear LTE residual without changing fields. */
template <unsigned int N>
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HybridNonlinearLteTrial
EvaluateHybridNonlinearLteTrial (
    int const i,
    int const j,
    int const k,
    HybridNonlinearCellState const& state,
    amrex::Real const temperature_increment,
    amrex::Real const pending_material_energy,
    amrex::Real const cell_volume,
    amrex::Real const x,
    amrex::Real const y,
    amrex::Real const z,
    amrex::GpuArray<amrex::Real, N> const& opacity_number_density,
    amrex::Array4<amrex::Real> const& radiation,
    ElectronThermodynamicsExecutor const thermodynamics,
    ImplicitLteCellContext<N> const& context) noexcept
{
    constexpr amrex::Real radiation_constant = 7.565733250280007e-16_rt;
    HybridNonlinearLteTrial trial;
    trial.valid = state.material.valid
        && state.num_active_corners > 0
        && pending_material_energy >= 0.0_rt
        && amrex::Math::isfinite(pending_material_energy)
        && cell_volume > 0.0_rt
        && amrex::Math::isfinite(cell_volume);
    if (!trial.valid) { return trial; }

    trial.radiation_temperature =
        state.material.electron_temperature + temperature_increment;
    if (trial.radiation_temperature < thermodynamics.minimumTemperature()
        || trial.radiation_temperature > thermodynamics.maximumTemperature()
        || !amrex::Math::isfinite(trial.radiation_temperature))
    {
        trial.valid = false;
        return trial;
    }
    trial.material_energy_change = EvaluateHybridNonlinearMaterialEnergyChange(
        state, thermodynamics, temperature_increment, trial.valid);
    if (!trial.valid) { return trial; }

    amrex::Real const temperature_squared =
        trial.radiation_temperature * trial.radiation_temperature;
    amrex::Real const equilibrium_total_energy = radiation_constant
        * temperature_squared * temperature_squared * cell_volume;
    if (equilibrium_total_energy < 0.0_rt
        || !amrex::Math::isfinite(equilibrium_total_energy))
    {
        trial.valid = false;
        return trial;
    }

    for (int group = 0; group < context.num_groups; ++group) {
        amrex::Real const group_energy =
            context.energy_groups.representativeEnergy(group);
        amrex::Real const planck_absorption = context.planckAbsorption(
            opacity_number_density, x, y, z, group_energy,
            state.material.electron_density, trial.radiation_temperature);
        amrex::Real const planck_emission = context.planckEmission(
            opacity_number_density, x, y, z, group_energy,
            state.material.electron_density, trial.radiation_temperature);
        amrex::Real const old_radiation_energy = radiation(i, j, k, group);
        amrex::Real const equilibrium_energy = equilibrium_total_energy
            * context.energy_groups.planckFraction(
                group, PhysConst::kb * trial.radiation_temperature);
        PlanckExchangeResult const exchange = EvaluatePlanckExchange(
            planck_absorption, planck_emission, old_radiation_energy,
            equilibrium_energy, context.dt);
        if (!exchange.valid) {
            trial.valid = false;
            return trial;
        }
        trial.radiation_exchange += exchange.exchange_energy;
    }
    trial.residual = trial.material_energy_change
        + trial.radiation_exchange - pending_material_energy;
    trial.valid = amrex::Math::isfinite(trial.material_energy_change)
        && amrex::Math::isfinite(trial.radiation_exchange)
        && amrex::Math::isfinite(trial.residual);
    return trial;
}

/** Conservative implicit LTE exchange for a nonlinear hybrid-electron EOS. */
template <unsigned int N>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
ApplyHybridNonlinearImplicitLteCellExchange (
    int const i,
    int const j,
    int const k,
    HybridNonlinearCellState const& state,
    amrex::Real const pending_material_energy,
    amrex::Real const cell_volume,
    amrex::Real const x,
    amrex::Real const y,
    amrex::Real const z,
    amrex::GpuArray<amrex::Real, N> const& opacity_number_density,
    amrex::Array4<amrex::Real> const& radiation,
    amrex::Array4<amrex::Real> const& material,
    amrex::Array4<amrex::Real> const& nonlinear_remap,
    amrex::Array4<amrex::Real> const& rosseland_opacity,
    ElectronThermodynamicsExecutor const thermodynamics,
    ImplicitLteCellContext<N> const& context,
    amrex::Array4<int> const& cell_status) noexcept
{
    if (!state.material.valid || state.num_active_corners <= 0
        || !(state.material.heat_capacity > 0.0_rt)
        || !amrex::Math::isfinite(state.material.heat_capacity)
        || !amrex::Math::isfinite(state.material.internal_energy)
        || pending_material_energy < 0.0_rt
        || !amrex::Math::isfinite(pending_material_energy))
    {
        cell_status(i, j, k, 0) = 1;
        return;
    }

    amrex::Real minimum_old_temperature =
        std::numeric_limits<amrex::Real>::max();
    amrex::Real maximum_old_temperature = 0.0_rt;
    for (int corner = 0;
         corner < HybridNonlinearCellState::max_corners; ++corner)
    {
        if (state.active[corner] == 0) { continue; }
        minimum_old_temperature = amrex::min(
            minimum_old_temperature, state.temperature[corner]);
        maximum_old_temperature = amrex::max(
            maximum_old_temperature, state.temperature[corner]);
    }
    // Opacity tables are often a stricter temperature domain than the caloric
    // EOS. A residual trial at the EOS floor/ceiling is unevaluable if it
    // leaves the opacity table, and that must not be reported as a local
    // Planck/Rosseland failure.
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
    amrex::Real opacity_minimum_temperature = 0.0_rt;
    amrex::Real opacity_maximum_temperature =
        std::numeric_limits<amrex::Real>::infinity();
    if (context.material_evaluator.enabled) {
        auto const& table = context.material_evaluator.table;
        if (table.m_num_temperature > 0 && table.m_log_temperature != nullptr)
        {
            opacity_minimum_temperature =
                std::exp(table.m_log_temperature[0]);
            opacity_maximum_temperature = std::exp(
                table.m_log_temperature[table.m_num_temperature - 1]);
        }
    } else if (context.registered_material_evaluator.enabled) {
        amrex::GpuArray<
            amrex::Real, warpx::materials::MaterialRegistry::max_materials>
            material_mass_densities{};
        int const material_index = context.registered_material_evaluator
            .resolvedMaterial(
                opacity_number_density, material_mass_densities);
        if (material_index >= 0
            && material_index
                < context.registered_material_evaluator.num_materials)
        {
            auto const& table =
                context.registered_material_evaluator.tables[material_index];
            if (table.m_num_temperature > 0
                && table.m_log_temperature != nullptr)
            {
                opacity_minimum_temperature =
                    std::exp(table.m_log_temperature[0]);
                opacity_maximum_temperature = std::exp(
                    table.m_log_temperature[
                        table.m_num_temperature - 1]);
            }
        }
    }
#else
    amrex::Real const opacity_minimum_temperature = 0.0_rt;
    amrex::Real const opacity_maximum_temperature =
        std::numeric_limits<amrex::Real>::infinity();
#endif
    amrex::Real const minimum_evaluable_temperature = amrex::max(
        thermodynamics.minimumTemperature(), opacity_minimum_temperature);
    amrex::Real const maximum_evaluable_temperature = amrex::min(
        thermodynamics.maximumTemperature(), opacity_maximum_temperature);
    amrex::Real const minimum_increment =
        minimum_evaluable_temperature - minimum_old_temperature;
    amrex::Real const maximum_increment =
        maximum_evaluable_temperature - maximum_old_temperature;
    amrex::Real const temperature_scale = amrex::max(
        state.material.electron_temperature, 1.0_rt);

    amrex::Real energy_scale = state.material.available_energy
        + pending_material_energy;
    for (int group = 0; group < context.num_groups; ++group) {
        amrex::Real const old_radiation_energy = radiation(i, j, k, group);
        if (old_radiation_energy < 0.0_rt
            || !amrex::Math::isfinite(old_radiation_energy))
        {
            cell_status(i, j, k, 0) = 1;
            return;
        }
        energy_scale += old_radiation_energy;
    }
    energy_scale = amrex::max(
        energy_scale, std::numeric_limits<amrex::Real>::min());
    if (!amrex::Math::isfinite(energy_scale)) {
        cell_status(i, j, k, 0) = 1;
        return;
    }

    HybridNonlinearLteTrial const zero_trial =
        EvaluateHybridNonlinearLteTrial(
            i, j, k, state, 0.0_rt, pending_material_energy, cell_volume,
            x, y, z, opacity_number_density, radiation, thermodynamics,
            context);
    if (!zero_trial.valid) {
        cell_status(i, j, k, 0) = 1;
        return;
    }

    amrex::Real final_increment = 0.0_rt;
    bool floor_limited = false;
    bool converged = zero_trial.residual == 0.0_rt;
    HybridNonlinearLteTrial final_trial = zero_trial;
    amrex::Real bracket_lo = 0.0_rt;
    amrex::Real bracket_hi = 0.0_rt;
    HybridNonlinearLteTrial lo_trial = zero_trial;
    HybridNonlinearLteTrial hi_trial = zero_trial;

    if (!converged && zero_trial.residual > 0.0_rt) {
        // Emission: expand a cooling bracket from ΔT=0. Jumping to the EOS
        // floor first evaluates opacity at a temperature the table does not
        // cover even when the physical root is a tiny decrement.
        bracket_hi = 0.0_rt;
        hi_trial = zero_trial;
        amrex::Real last_valid_increment = 0.0_rt;
        HybridNonlinearLteTrial last_valid = zero_trial;
        amrex::Real step = amrex::max(1.0e-6_rt * temperature_scale, 1.0_rt);
        bool found_lo = false;
        for (int iteration = 0;
             iteration < context.max_iterations; ++iteration)
        {
            amrex::Real const candidate =
                amrex::max(minimum_increment, -step);
            HybridNonlinearLteTrial const trial =
                EvaluateHybridNonlinearLteTrial(
                    i, j, k, state, candidate, pending_material_energy,
                    cell_volume, x, y, z, opacity_number_density, radiation,
                    thermodynamics, context);
            if (trial.valid) {
                last_valid_increment = candidate;
                last_valid = trial;
                if (trial.residual <= 0.0_rt) {
                    bracket_lo = candidate;
                    lo_trial = trial;
                    found_lo = true;
                    break;
                }
                if (candidate <= minimum_increment) {
                    floor_limited = true;
                    converged = true;
                    final_increment = candidate;
                    final_trial = trial;
                    found_lo = true;
                    break;
                }
                step *= 2.0_rt;
                continue;
            }
            if (!(candidate < last_valid_increment)) {
                cell_status(i, j, k, 0) = 1;
                return;
            }
            amrex::Real domain_warm = last_valid_increment;
            amrex::Real domain_cold = candidate;
            HybridNonlinearLteTrial domain_trial = last_valid;
            for (int shrink = 0;
                 shrink < context.max_iterations; ++shrink)
            {
                amrex::Real const midpoint =
                    0.5_rt * (domain_warm + domain_cold);
                HybridNonlinearLteTrial const mid =
                    EvaluateHybridNonlinearLteTrial(
                        i, j, k, state, midpoint, pending_material_energy,
                        cell_volume, x, y, z, opacity_number_density,
                        radiation, thermodynamics, context);
                if (mid.valid) {
                    domain_warm = midpoint;
                    domain_trial = mid;
                    if (mid.residual <= 0.0_rt) {
                        bracket_lo = midpoint;
                        lo_trial = mid;
                        found_lo = true;
                        break;
                    }
                } else {
                    domain_cold = midpoint;
                }
                if (domain_warm - domain_cold
                    <= context.tolerance * temperature_scale)
                {
                    break;
                }
            }
            if (found_lo) { break; }
            if (domain_trial.valid && domain_trial.residual > 0.0_rt) {
                floor_limited = true;
                converged = true;
                final_increment = domain_warm;
                final_trial = domain_trial;
                found_lo = true;
                break;
            }
            cell_status(i, j, k, 0) = 1;
            return;
        }
        if (!found_lo && !converged) {
            cell_status(i, j, k, 1) = 1; // cooling bracket not found
            return;
        }
    } else if (!converged) {
        bracket_lo = 0.0_rt;
        lo_trial = zero_trial;
        amrex::Real last_valid_increment = 0.0_rt;
        amrex::Real step = amrex::max(1.0e-6_rt * temperature_scale, 1.0_rt);
        bool found_hi = false;
        for (int iteration = 0;
             iteration < context.max_iterations; ++iteration)
        {
            amrex::Real candidate = step;
            if (amrex::Math::isfinite(maximum_increment)) {
                candidate = amrex::min(maximum_increment, step);
            }
            HybridNonlinearLteTrial const trial =
                EvaluateHybridNonlinearLteTrial(
                    i, j, k, state, candidate, pending_material_energy,
                    cell_volume, x, y, z, opacity_number_density, radiation,
                    thermodynamics, context);
            if (trial.valid) {
                last_valid_increment = candidate;
                if (trial.residual >= 0.0_rt) {
                    bracket_hi = candidate;
                    hi_trial = trial;
                    found_hi = true;
                    break;
                }
                if (amrex::Math::isfinite(maximum_increment)
                    && candidate >= maximum_increment)
                {
                    // Absorption that would exceed the evaluable temperature
                    // ceiling is a configuration error, never a clamp.
                    cell_status(i, j, k, 0) = 1;
                    return;
                }
                step *= 2.0_rt;
                if (!amrex::Math::isfinite(step)) { break; }
                continue;
            }
            if (!(candidate > last_valid_increment)) {
                cell_status(i, j, k, 0) = 1;
                return;
            }
            amrex::Real domain_cool = last_valid_increment;
            amrex::Real domain_hot = candidate;
            for (int shrink = 0;
                 shrink < context.max_iterations; ++shrink)
            {
                amrex::Real const midpoint =
                    0.5_rt * (domain_cool + domain_hot);
                HybridNonlinearLteTrial const mid =
                    EvaluateHybridNonlinearLteTrial(
                        i, j, k, state, midpoint, pending_material_energy,
                        cell_volume, x, y, z, opacity_number_density,
                        radiation, thermodynamics, context);
                if (mid.valid) {
                    domain_cool = midpoint;
                    if (mid.residual >= 0.0_rt) {
                        bracket_hi = midpoint;
                        hi_trial = mid;
                        found_hi = true;
                        break;
                    }
                } else {
                    domain_hot = midpoint;
                }
                if (domain_hot - domain_cool
                    <= context.tolerance * temperature_scale)
                {
                    break;
                }
            }
            if (found_hi) { break; }
            cell_status(i, j, k, 0) = 1;
            return;
        }
        if (!found_hi) {
            cell_status(i, j, k, 1) = 2; // heating bracket not found
            return;
        }
    }

    if (!converged) {
        if (!(lo_trial.residual <= 0.0_rt)
            || !(hi_trial.residual >= 0.0_rt))
        {
            cell_status(i, j, k, 1) = 3; // residual signs do not bracket a root
            return;
        }
        for (int iteration = 0;
             iteration < context.max_iterations; ++iteration)
        {
            amrex::Real const midpoint =
                0.5_rt * (bracket_lo + bracket_hi);
            HybridNonlinearLteTrial const midpoint_trial =
                EvaluateHybridNonlinearLteTrial(
                    i, j, k, state, midpoint, pending_material_energy,
                    cell_volume, x, y, z, opacity_number_density, radiation,
                    thermodynamics, context);
            if (!midpoint_trial.valid) {
                cell_status(i, j, k, 0) = 1;
                return;
            }
            if (midpoint_trial.residual > 0.0_rt) {
                bracket_hi = midpoint;
                hi_trial = midpoint_trial;
            } else {
                bracket_lo = midpoint;
                lo_trial = midpoint_trial;
            }
            amrex::Real const trial_temperature =
                state.material.electron_temperature + midpoint;
            if (bracket_hi - bracket_lo
                <= context.tolerance * amrex::max(
                    std::abs(trial_temperature), 1.0_rt))
            {
                converged = true;
                break;
            }
        }
        if (!converged) {
            cell_status(i, j, k, 1) = 4; // temperature bisection did not converge
            return;
        }
        // Keep residual <= 0 so emission is funded by the EOS energy change
        // without a roundoff-only positive-exchange clip.
        final_increment = bracket_lo;
        final_trial = lo_trial;
        if (hi_trial.valid
            && std::abs(hi_trial.residual) < std::abs(lo_trial.residual)
            && hi_trial.residual
                <= 10.0_rt * context.tolerance * energy_scale)
        {
            final_increment = bracket_hi;
            final_trial = hi_trial;
        }
    }

    constexpr amrex::Real radiation_constant = 7.565733250280007e-16_rt;
    amrex::Real const final_temperature = final_trial.radiation_temperature;
    amrex::Real const temperature_squared =
        final_temperature * final_temperature;
    amrex::Real const equilibrium_total_energy = radiation_constant
        * temperature_squared * temperature_squared * cell_volume;
    amrex::Real positive_exchange = 0.0_rt;
    amrex::Real negative_exchange = 0.0_rt;
    for (int group = 0; group < context.num_groups; ++group) {
        amrex::Real const group_energy =
            context.energy_groups.representativeEnergy(group);
        amrex::Real const planck_absorption = context.planckAbsorption(
            opacity_number_density, x, y, z, group_energy,
            state.material.electron_density, final_temperature);
        amrex::Real const planck_emission = context.planckEmission(
            opacity_number_density, x, y, z, group_energy,
            state.material.electron_density, final_temperature);
        amrex::Real rosseland_opacity_value = 0.0_rt;
        if (context.enable_diffusion) {
            rosseland_opacity_value = context.rosselandOpacity(
                opacity_number_density, x, y, z, group_energy,
                state.material.electron_density, final_temperature);
        }
        amrex::Real const old_radiation_energy = radiation(i, j, k, group);
        amrex::Real const equilibrium_energy = equilibrium_total_energy
            * context.energy_groups.planckFraction(
                group, PhysConst::kb * final_temperature);
        PlanckExchangeResult const exchange = EvaluatePlanckExchange(
            planck_absorption, planck_emission, old_radiation_energy,
            equilibrium_energy, context.dt);
        if (!exchange.valid || rosseland_opacity_value < 0.0_rt
            || !amrex::Math::isfinite(rosseland_opacity_value)
            || !amrex::Math::isfinite(exchange.exchange_energy))
        {
            cell_status(i, j, k, 0) = 1;
            return;
        }
        if (context.enable_diffusion) {
            rosseland_opacity(i, j, k, group) = rosseland_opacity_value;
        }
        if (exchange.exchange_energy > 0.0_rt) {
            positive_exchange += exchange.exchange_energy;
        } else {
            negative_exchange += exchange.exchange_energy;
        }
    }

    amrex::Real positive_scale = 1.0_rt;
    amrex::Real const available_exchange = pending_material_energy
        - final_trial.material_energy_change;
    if (positive_exchange + negative_exchange > available_exchange
        && positive_exchange > 0.0_rt)
    {
        amrex::Real const overflow = positive_exchange + negative_exchange
            - available_exchange;
        if (!floor_limited
            && overflow > 10.0_rt * context.tolerance * energy_scale)
        {
            cell_status(i, j, k, 1) = 5; // emission exceeds EOS energy change
            return;
        }
        positive_scale = amrex::max(
            0.0_rt,
            (available_exchange - negative_exchange) / positive_exchange);
    }

    amrex::Real total_exchange = 0.0_rt;
    for (int group = 0; group < context.num_groups; ++group) {
        amrex::Real const group_energy =
            context.energy_groups.representativeEnergy(group);
        amrex::Real const planck_absorption = context.planckAbsorption(
            opacity_number_density, x, y, z, group_energy,
            state.material.electron_density, final_temperature);
        amrex::Real const planck_emission = context.planckEmission(
            opacity_number_density, x, y, z, group_energy,
            state.material.electron_density, final_temperature);
        amrex::Real const old_radiation_energy = radiation(i, j, k, group);
        amrex::Real const equilibrium_energy = equilibrium_total_energy
            * context.energy_groups.planckFraction(
                group, PhysConst::kb * final_temperature);
        PlanckExchangeResult const exchange = EvaluatePlanckExchange(
            planck_absorption, planck_emission, old_radiation_energy,
            equilibrium_energy, context.dt);
        if (!exchange.valid) {
            cell_status(i, j, k, 0) = 1;
            return;
        }
        amrex::Real exchange_energy = exchange.exchange_energy;
        if (exchange_energy > 0.0_rt) {
            exchange_energy *= positive_scale;
        }
        amrex::Real const new_radiation_energy =
            old_radiation_energy + exchange_energy;
        if (new_radiation_energy < 0.0_rt
            || !amrex::Math::isfinite(new_radiation_energy))
        {
            cell_status(i, j, k, 0) = 1;
            return;
        }
        radiation(i, j, k, group) = new_radiation_energy;
        total_exchange += exchange_energy;
    }

    amrex::Real const material_ledger =
        pending_material_energy - total_exchange;
    amrex::Real const remap_residual = material_ledger
        - final_trial.material_energy_change;
    if (!amrex::Math::isfinite(material_ledger)
        || !amrex::Math::isfinite(remap_residual)
        || std::abs(remap_residual)
            > 10.0_rt * context.tolerance * energy_scale)
    {
        cell_status(i, j, k, 1) = 6; // remap residual exceeds conservation gate
        return;
    }
    material(i, j, k) = material_ledger;
    nonlinear_remap(
        i, j, k, nonlinear_lte_temperature_increment_comp) =
            final_increment;
    nonlinear_remap(i, j, k, nonlinear_lte_represented_energy_comp) =
        final_trial.material_energy_change;
}

/** Advance one cell with conservative implicit-temperature LTE exchange.
 *
 * This helper deliberately implements the constant-heat-capacity residual
 * U(T)=C_V*T. Constructor validation restricts this path to thermodynamics
 * backends that provide that exact relation.
 */
template <unsigned int N>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
ApplyImplicitLteCellExchange (
    int const i, int const j, int const k,
    HybridCellMaterialState const& material_state,
    amrex::Real const pending_material_energy, amrex::Real const cell_volume,
    amrex::Real const x, amrex::Real const y, amrex::Real const z,
    amrex::GpuArray<amrex::Real, N> const& opacity_number_density,
    amrex::Array4<amrex::Real> const& radiation,
    amrex::Array4<amrex::Real> const& material,
    amrex::Array4<amrex::Real> const& rosseland_opacity,
    ImplicitLteCellContext<N> const& context,
    amrex::Array4<int> const& cell_status) noexcept {
    constexpr amrex::Real radiation_constant = 7.565733250280007e-16_rt;
    amrex::Real const heat_capacity = material_state.heat_capacity;
    amrex::Real const final_material_energy_before_lte =
        material_state.internal_energy + pending_material_energy;
    amrex::Real total_energy = final_material_energy_before_lte;
    bool valid = heat_capacity > 0.0_rt &&
                 amrex::Math::isfinite(heat_capacity) &&
                 material_state.internal_energy >= 0.0_rt &&
                 amrex::Math::isfinite(material_state.internal_energy) &&
                 pending_material_energy >= 0.0_rt &&
                 amrex::Math::isfinite(pending_material_energy) &&
                 final_material_energy_before_lte >= 0.0_rt &&
                 amrex::Math::isfinite(final_material_energy_before_lte);
    for (int group = 0; group < context.num_groups; ++group) {
        amrex::Real const old_radiation_energy = radiation(i, j, k, group);
        valid = valid && old_radiation_energy >= 0.0_rt &&
                amrex::Math::isfinite(old_radiation_energy);
        total_energy += old_radiation_energy;
    }
    valid =
        valid && total_energy >= 0.0_rt && amrex::Math::isfinite(total_energy);
    if (!valid) {
        cell_status(i, j, k, 0) = 1;
        return;
    }

    amrex::Real const available_material_energy =
        material_state.available_energy +
        amrex::max(0.0_rt, pending_material_energy);
    amrex::Real temperature_lo =
        amrex::max(0.0_rt, (material_state.internal_energy -
                            material_state.available_energy) /
                               heat_capacity);
    amrex::Real temperature_hi = total_energy / heat_capacity;
    if (!amrex::Math::isfinite(temperature_hi)) {
        cell_status(i, j, k, 0) = 1;
        return;
    }

    for (int iteration = 0; iteration < context.max_iterations; ++iteration) {
        if (temperature_hi - temperature_lo <=
            context.tolerance *
                amrex::max(temperature_hi,
                           std::numeric_limits<amrex::Real>::min())) {
            break;
        }
        amrex::Real const trial_temperature =
            0.5_rt * (temperature_lo + temperature_hi);
        amrex::Real const temperature_squared =
            trial_temperature * trial_temperature;
        amrex::Real const equilibrium_total_energy =
            radiation_constant * temperature_squared * temperature_squared *
            cell_volume;
        amrex::Real energy_residual =
            heat_capacity * trial_temperature - total_energy;

        for (int group = 0; group < context.num_groups; ++group) {
            amrex::Real const group_energy =
                context.energy_groups.representativeEnergy(group);
            amrex::Real const planck_absorption = context.planckAbsorption(
                opacity_number_density, x, y, z, group_energy,
                material_state.electron_density,
                trial_temperature);
            amrex::Real const planck_emission = context.planckEmission(
                opacity_number_density, x, y, z, group_energy,
                material_state.electron_density, trial_temperature);
            amrex::Real const old_radiation_energy = radiation(i, j, k, group);
            amrex::Real const equilibrium_energy =
                equilibrium_total_energy *
                context.energy_groups.planckFraction(
                    group, PhysConst::kb * trial_temperature);
            PlanckExchangeResult const exchange = EvaluatePlanckExchange(
                planck_absorption, planck_emission, old_radiation_energy,
                equilibrium_energy, context.dt);
            if (!exchange.valid) {
                valid = false;
                break;
            }
            energy_residual += exchange.new_radiation_energy;
        }
        if (!valid || !amrex::Math::isfinite(energy_residual)) {
            cell_status(i, j, k, 0) = 1;
            return;
        }
        if (energy_residual > 0.0_rt) {
            temperature_hi = trial_temperature;
        } else {
            temperature_lo = trial_temperature;
        }
    }

    if (temperature_hi - temperature_lo >
        context.tolerance *
            amrex::max(temperature_hi,
                       std::numeric_limits<amrex::Real>::min())) {
        cell_status(i, j, k, 1) = 1;
        return;
    }

    amrex::Real const final_temperature = temperature_lo;
    amrex::Real const final_temperature_squared =
        final_temperature * final_temperature;
    amrex::Real const equilibrium_total_energy =
        radiation_constant * final_temperature_squared *
        final_temperature_squared * cell_volume;
    amrex::Real positive_exchange = 0.0_rt;
    amrex::Real negative_exchange = 0.0_rt;
    for (int group = 0; group < context.num_groups; ++group) {
        amrex::Real const group_energy =
            context.energy_groups.representativeEnergy(group);
        amrex::Real const planck_absorption = context.planckAbsorption(
            opacity_number_density, x, y, z, group_energy,
            material_state.electron_density, final_temperature);
        amrex::Real const planck_emission = context.planckEmission(
            opacity_number_density, x, y, z, group_energy,
            material_state.electron_density, final_temperature);
        amrex::Real rosseland_opacity_value = 0.0_rt;
        if (context.enable_diffusion) {
            rosseland_opacity_value = context.rosselandOpacity(
                opacity_number_density, x, y, z, group_energy,
                material_state.electron_density,
                final_temperature);
        }
        amrex::Real const old_radiation_energy = radiation(i, j, k, group);
        amrex::Real const equilibrium_energy =
            equilibrium_total_energy *
            context.energy_groups.planckFraction(group, PhysConst::kb *
                                                            final_temperature);
        PlanckExchangeResult const exchange = EvaluatePlanckExchange(
            planck_absorption, planck_emission, old_radiation_energy,
            equilibrium_energy, context.dt);
        if (!exchange.valid || rosseland_opacity_value < 0.0_rt ||
            !amrex::Math::isfinite(rosseland_opacity_value) ||
            !amrex::Math::isfinite(exchange.exchange_energy)) {
            cell_status(i, j, k, 0) = 1;
            return;
        }
        if (context.enable_diffusion) {
            rosseland_opacity(i, j, k, group) = rosseland_opacity_value;
        }
        if (exchange.exchange_energy > 0.0_rt) {
            positive_exchange += exchange.exchange_energy;
        } else {
            negative_exchange += exchange.exchange_energy;
        }
    }

    amrex::Real positive_scale = 1.0_rt;
    if (positive_exchange + negative_exchange > available_material_energy &&
        positive_exchange > 0.0_rt) {
        positive_scale =
            amrex::max(0.0_rt, (available_material_energy - negative_exchange) /
                                   positive_exchange);
    }

    amrex::Real total_exchange = 0.0_rt;
    for (int group = 0; group < context.num_groups; ++group) {
        amrex::Real const group_energy =
            context.energy_groups.representativeEnergy(group);
        amrex::Real const planck_absorption = context.planckAbsorption(
            opacity_number_density, x, y, z, group_energy,
            material_state.electron_density, final_temperature);
        amrex::Real const planck_emission = context.planckEmission(
            opacity_number_density, x, y, z, group_energy,
            material_state.electron_density, final_temperature);
        amrex::Real const old_radiation_energy = radiation(i, j, k, group);
        amrex::Real const equilibrium_energy =
            equilibrium_total_energy *
            context.energy_groups.planckFraction(group, PhysConst::kb *
                                                            final_temperature);
        PlanckExchangeResult const exchange = EvaluatePlanckExchange(
            planck_absorption, planck_emission, old_radiation_energy,
            equilibrium_energy, context.dt);
        if (!exchange.valid) {
            cell_status(i, j, k, 0) = 1;
            return;
        }
        amrex::Real exchange_energy = exchange.exchange_energy;
        if (exchange_energy > 0.0_rt) {
            exchange_energy *= positive_scale;
        }
        radiation(i, j, k, group) = old_radiation_energy + exchange_energy;
        total_exchange += exchange_energy;
    }

    amrex::Real const energy_residual = heat_capacity * final_temperature +
                                        total_exchange -
                                        final_material_energy_before_lte;
    amrex::Real const energy_scale =
        amrex::max(total_energy, std::numeric_limits<amrex::Real>::min());
    if (!amrex::Math::isfinite(energy_residual) ||
        std::abs(energy_residual) >
            10.0_rt * context.tolerance * energy_scale) {
        cell_status(i, j, k, 1) = 1;
        return;
    }
    material(i, j, k) -= total_exchange;
}

[[nodiscard]]
std::vector<std::unique_ptr<amrex::MultiFab>>
GetOpacityNumberDensities (
    MultiParticleContainer& particles,
    std::vector<std::string> const& opacity_species,
    int const level,
    int const radiation_finest_level,
    amrex::MultiFab const& material_energy)
{
    std::vector<std::unique_ptr<amrex::MultiFab>> number_densities;
    number_densities.reserve(opacity_species.size());
    for (std::string const& species_name : opacity_species) {
        auto& species = particles.GetParticleContainerFromName(species_name);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            species.finestLevel() == radiation_finest_level,
            "Radiation photons and opacity material species must use the same "
            "AMR levels.");
        auto deposited_number_density = species.GetNumberDensity(level);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            deposited_number_density->boxArray() == material_energy.boxArray()
                && deposited_number_density->DistributionMap()
                    == material_energy.DistributionMap(),
            "Opacity-species number-density grids must match the radiation "
            "material grid.");
        auto number_density = std::make_unique<amrex::MultiFab>(
            deposited_number_density->boxArray(),
            deposited_number_density->DistributionMap(), 1, 1);
        number_density->setVal(0.0_rt);
        amrex::MultiFab::Copy(
            *number_density, *deposited_number_density, 0, 0, 1, 0);
        number_densities.push_back(std::move(number_density));
    }
    return number_densities;
}

void
FillOpacityNumberDensityBoundaries (
    std::vector<std::unique_ptr<amrex::MultiFab>> const& number_densities,
    int const level)
{
    for (auto const& number_density : number_densities) {
        number_density->FillBoundary(
            WarpX::GetInstance().Geom(level).periodicity());
    }
}

#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
template <unsigned int N, unsigned int M>
void
PreflightRegisteredOpacityCells (
    std::vector<std::unique_ptr<amrex::MultiFab>> const& number_densities,
    int const num_species,
    amrex::GpuArray<amrex::Real, N> const& species_masses,
    amrex::GpuArray<int, N> const& species_material_indices,
    warpx::materials::MaterialRegistry::ResolvedCellSelector const selector)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        num_species > 0
            && number_densities.size()
                == static_cast<std::size_t>(num_species),
        "Internal registry opacity preflight error: carrier-density count "
        "does not match the configured species map.");
    // Each owned cell writes only its own two flags. The subsequent iMultiFab
    // sums provide device and MPI reductions without shared kernel atomics.
    amrex::iMultiFab classification_status(
        number_densities.front()->boxArray(),
        number_densities.front()->DistributionMap(), 2, 0);
    classification_status.setVal(0);
    auto const masses = species_masses;
    auto const material_indices = species_material_indices;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(
             classification_status, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi)
    {
        amrex::GpuArray<amrex::Array4<amrex::Real const>, N>
            number_density_arrays{};
        for (int species = 0; species < num_species; ++species) {
            number_density_arrays[species] =
                number_densities[species]->const_array(mfi);
        }
        amrex::Array4<int> const status = classification_status.array(mfi);
        amrex::Box const box = mfi.tilebox();
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (
            int i, int j, int k) noexcept
        {
            amrex::GpuArray<amrex::Real, M> material_mass_densities{};
            for (int species = 0; species < num_species; ++species) {
                int const material = material_indices[species];
                amrex::Real const number_density =
                    number_density_arrays[species](i, j, k);
                if (material < 0 || material >= selector.num_materials
                    || number_density < 0.0_rt
                    || !amrex::Math::isfinite(number_density))
                {
                    status(i, j, k, 1) = 1;
                    return;
                }
                material_mass_densities[material] +=
                    number_density * masses[species];
            }
            int const classification = selector(material_mass_densities);
            if (classification == static_cast<int>(
                    warpx::materials::MaterialRegistry::ResolvedCell::Mixed))
            {
                status(i, j, k, 0) = 1;
            } else if (classification == static_cast<int>(
                           warpx::materials::MaterialRegistry::ResolvedCell::Invalid))
            {
                status(i, j, k, 1) = 1;
            }
        });
    }

    amrex::Long const global_mixed_cells =
        classification_status.sum(0, 0, /*local=*/false);
    amrex::Long const global_invalid_cells =
        classification_status.sum(1, 0, /*local=*/false);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        global_mixed_cells == 0 && global_invalid_cells == 0,
        "Registry-native opacity preflight rejected the mesh before radiation "
        "or material mutation: " + std::to_string(global_mixed_cells)
        + " owned mixed cell(s), " + std::to_string(global_invalid_cells)
        + " owned invalid cell(s). Refine carrier initialization/deposition "
        "or explicitly adjust the resolved-cell tolerances; pure-material "
        "opacity tables are not mixed.");
}
#endif

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
AddDiffusionEscape (
    int const boundary,
    amrex::Real const energy_density,
    amrex::Real const face_area,
    amrex::Real const diffusion_dt,
    int const direction,
    int const side,
    amrex::Real& energy_rate,
    amrex::Real* escaped_energy,
    amrex::GpuArray<amrex::Real*, 3> const& escaped_momentum) noexcept
{
    amrex::Real const escape_factor = DiffusionEscapeFactor(boundary);
    if (escape_factor <= 0.0_rt || energy_density <= 0.0_rt
        || face_area <= 0.0_rt)
    {
        return;
    }
    amrex::Real const outward_flux =
        escape_factor * PhysConst::c * energy_density;
    energy_rate -= outward_flux * face_area;
    amrex::HostDevice::Atomic::Add(
        escaped_energy, outward_flux * face_area * diffusion_dt);
    amrex::Real const outward_momentum =
        DiffusionEscapeMomentumFactor(boundary)
        * energy_density * face_area * diffusion_dt;
    amrex::HostDevice::Atomic::Add(
        escaped_momentum[RadiationMomentumComponent(direction)],
        side * outward_momentum);
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
std::uint64_t
KineticThermalSeedHash (std::uint64_t value) noexcept
{
    // SplitMix64 is a counter-based integer permutation, so each particle's
    // seed is independent of execution order. Floating atomic accumulation of
    // the cell mean is conserved to the checked tolerance, not bitwise across
    // backends.
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
KineticThermalSeed (
    std::uint64_t const particle_identity, int const component) noexcept
{
    std::uint64_t const key = particle_identity
        ^ (0xd1b54a32d192ed03ULL
            * static_cast<std::uint64_t>(component + 1));
    // Use the upper 53 bits to form a deterministic value in [-c,c).
    amrex::Real const unit = static_cast<amrex::Real>(
        KineticThermalSeedHash(key) >> 11U)
        * 1.11022302462515654042e-16_rt;
    return PhysConst::c * (2.0_rt * unit - 1.0_rt);
}

/** Proper velocity represented in the local kinetic-electron moment frame.
 *
 * Components 0, 1, and 2 are (r, theta, z) in RCYLINDER/RZ, (r, theta, phi)
 * in RSPHERE, and (x, y, z) in Cartesian geometries.  The Cartesian
 * components are the stored particle components; radial geometry stores
 * particle momenta in Cartesian coordinates but deposits cell moments in the
 * local curvilinear frame.
 */
struct KineticLocalVelocity
{
    amrex::Real component_0;
    amrex::Real component_1;
    amrex::Real component_2;
};

// Shared layout for kinetic_moments in Advance and the thermal-energy update.
// Components 2--4 are local momentum components; component 5 is the frame-
// invariant weighted |u| scale used by the roundoff-scaled momentum check.
constexpr int local_momentum_0_comp = 2;
constexpr int local_momentum_1_comp = 3;
constexpr int local_momentum_2_comp = 4;
constexpr int local_momentum_scale_comp = 5;

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
KineticLocalVelocity
KineticCartesianToLocal (
    amrex::Real const ux_cartesian,
    amrex::Real const uy_cartesian,
    amrex::Real const uz_cartesian,
    amrex::Real const theta,
    amrex::Real const phi) noexcept
{
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
    amrex::ignore_unused(phi);
    amrex::Real const cos_theta = std::cos(theta);
    amrex::Real const sin_theta = std::sin(theta);
    return {
        ux_cartesian * cos_theta + uy_cartesian * sin_theta,
        -ux_cartesian * sin_theta + uy_cartesian * cos_theta,
        uz_cartesian};
#elif defined(WARPX_DIM_RSPHERE)
    amrex::Real const cos_theta = std::cos(theta);
    amrex::Real const sin_theta = std::sin(theta);
    amrex::Real const cos_phi = std::cos(phi);
    amrex::Real const sin_phi = std::sin(phi);
    return {
        ux_cartesian * cos_theta * cos_phi
            + uy_cartesian * sin_theta * cos_phi
            + uz_cartesian * sin_phi,
        -ux_cartesian * sin_theta + uy_cartesian * cos_theta,
        -ux_cartesian * cos_theta * sin_phi
            - uy_cartesian * sin_theta * sin_phi
            + uz_cartesian * cos_phi};
#else
    amrex::ignore_unused(theta, phi);
    return {ux_cartesian, uy_cartesian, uz_cartesian};
#endif
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
KineticLocalVelocity
KineticLocalToCartesian (
    amrex::Real const component_0,
    amrex::Real const component_1,
    amrex::Real const component_2,
    amrex::Real const theta,
    amrex::Real const phi) noexcept
{
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
    amrex::ignore_unused(phi);
    amrex::Real const cos_theta = std::cos(theta);
    amrex::Real const sin_theta = std::sin(theta);
    return {
        component_0 * cos_theta - component_1 * sin_theta,
        component_0 * sin_theta + component_1 * cos_theta,
        component_2};
#elif defined(WARPX_DIM_RSPHERE)
    amrex::Real const cos_theta = std::cos(theta);
    amrex::Real const sin_theta = std::sin(theta);
    amrex::Real const cos_phi = std::cos(phi);
    amrex::Real const sin_phi = std::sin(phi);
    return {
        component_0 * cos_theta * cos_phi
            - component_1 * sin_theta
            - component_2 * cos_theta * sin_phi,
        component_0 * sin_theta * cos_phi
            + component_1 * cos_theta
            - component_2 * sin_theta * sin_phi,
        component_0 * sin_phi + component_2 * cos_phi};
#else
    amrex::ignore_unused(theta, phi);
    return {component_0, component_1, component_2};
#endif
}

/** Apply a cell-integrated thermal-energy source to kinetic electrons.
 *
 * In each NGP cell, particle proper velocities are written as
 *
 *     u_p^{local} = u_bar^{local} + delta_u_p^{local},
 *
 * where the local frame is (r,theta,z) in RCYLINDER/RZ, (r,theta,phi) in
 * RSPHERE, and (x,y,z) in Cartesian geometries.  Particle momenta are rotated
 * into this frame before the update and rotated back before they are written.
 * The update scales only delta_u^{local}, so it preserves the cell-weighted
 * local electron momentum while a bracketed Newton solve chooses the scale that
 * gives the requested relativistic kinetic energy. For an exactly cold cell
 * with at least two particles, deterministic seeds keyed by the packed AMReX
 * particle identity (ID and birth CPU) are centered by their weighted cell mean
 * and provide a zero-momentum thermal direction. A one-particle/degenerate cell
 * cannot accept heat, and cooling below the fixed-momentum cold state is
 * impossible; both are rejected explicitly.
 */
amrex::Real
ApplyKineticElectronThermalEnergy (
    WarpXParticleContainer& electrons,
    int const lev,
    amrex::MultiFab& cell_integrated_energy_source,
    amrex::MultiFab const& initial_moments,
    amrex::Geometry const& geometry,
    amrex::MFItInfo info)
{
    constexpr int weight_comp = 0;
    constexpr int energy_comp = 1;

    constexpr int local_bulk_0_comp = 0;
    constexpr int local_bulk_1_comp = 1;
    constexpr int local_bulk_2_comp = 2;
    constexpr int target_energy_comp = 3;
    constexpr int lower_scale_comp = 4;
    constexpr int upper_scale_comp = 5;
    constexpr int trial_scale_comp = 6;
    constexpr int trial_energy_comp = 7;
    constexpr int trial_derivative_comp = 8;
    constexpr int local_seed_mean_0_comp = 9;
    constexpr int local_seed_mean_1_comp = 10;
    constexpr int local_seed_mean_2_comp = 11;
    constexpr int seed_reference_energy_comp = 12;
    constexpr int seeded_mode_comp = 13;
    constexpr int num_state_components = 14;

    amrex::MultiFab thermal_state(
        initial_moments.boxArray(), initial_moments.DistributionMap(),
        num_state_components, 0);
    thermal_state.setVal(0.0_rt);

    amrex::ParticleReal const electron_mass = electrons.getMass();
    auto const electron_mass_real =
        static_cast<amrex::Real>(electrons.getMass());
    auto const plo = geometry.ProbLoArray();
    auto const dxi = geometry.InvCellSizeArray();

    // Accumulate and center deterministic seeds from AMReX's packed (id, cpu)
    // particle identity. They are only used when the physical distribution is
    // exactly cold, but constructing them unconditionally avoids
    // data-dependent launch topology.
#ifdef AMREX_USE_OMP
    info.SetDynamic(true);
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi = electrons.MakeMFIter(lev, info);
         mfi.isValid(); ++mfi)
    {
        auto& tile = electrons.ParticlesAt(lev, mfi);
        long const np = tile.numParticles();
        auto const ptd = tile.getParticleTileData();
        auto const* const AMREX_RESTRICT wp = ptd.m_rdata[PIdx::w];
        amrex::Array4<amrex::Real> const state_arr = thermal_state.array(mfi);
        amrex::For(np, [=] AMREX_GPU_DEVICE (long ip) noexcept
        {
            auto const p = WarpXParticleContainer::ParticleType(ptd, ip);
            auto const [i, j, k] =
                amrex::getParticleCell(p, plo, dxi).dim3();
            auto const weight = static_cast<amrex::Real>(wp[ip]);
            amrex::Gpu::Atomic::AddNoRet(
                &state_arr(i, j, k, local_seed_mean_0_comp),
                weight * KineticThermalSeed(p.idcpu(), 0));
            amrex::Gpu::Atomic::AddNoRet(
                &state_arr(i, j, k, local_seed_mean_1_comp),
                weight * KineticThermalSeed(p.idcpu(), 1));
            amrex::Gpu::Atomic::AddNoRet(
                &state_arr(i, j, k, local_seed_mean_2_comp),
                weight * KineticThermalSeed(p.idcpu(), 2));
        });
    }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(thermal_state, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi)
    {
        amrex::Box const& box = mfi.tilebox();
        amrex::Array4<amrex::Real const> const moment_arr =
            initial_moments.const_array(mfi);
        amrex::Array4<amrex::Real> const state_arr = thermal_state.array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (
            int i, int j, int k) noexcept
        {
            amrex::Real const weight = moment_arr(i, j, k, weight_comp);
            if (weight <= 0.0_rt) { return; }
            state_arr(i, j, k, local_seed_mean_0_comp) /= weight;
            state_arr(i, j, k, local_seed_mean_1_comp) /= weight;
            state_arr(i, j, k, local_seed_mean_2_comp) /= weight;
        });
    }

    // Measure the relativistic energy at unit seeded scale after centering.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi = electrons.MakeMFIter(lev, info);
         mfi.isValid(); ++mfi)
    {
        auto& tile = electrons.ParticlesAt(lev, mfi);
        long const np = tile.numParticles();
        auto const ptd = tile.getParticleTileData();
        auto const* const AMREX_RESTRICT wp = ptd.m_rdata[PIdx::w];
        amrex::Array4<amrex::Real const> const moment_arr =
            initial_moments.const_array(mfi);
        amrex::Array4<amrex::Real const> const source_arr =
            cell_integrated_energy_source.const_array(mfi);
        amrex::Array4<amrex::Real> const state_arr = thermal_state.array(mfi);
        amrex::For(np, [=] AMREX_GPU_DEVICE (long ip) noexcept
        {
            auto const p = WarpXParticleContainer::ParticleType(ptd, ip);
            auto const [i, j, k] =
                amrex::getParticleCell(p, plo, dxi).dim3();
            if (source_arr(i, j, k) == 0.0_rt) { return; }
            amrex::Real const weight_sum = moment_arr(i, j, k, weight_comp);
            if (weight_sum <= 0.0_rt) { return; }
            amrex::Real const local_bulk_0 =
                moment_arr(i, j, k, local_momentum_0_comp) / weight_sum;
            amrex::Real const local_bulk_1 =
                moment_arr(i, j, k, local_momentum_1_comp) / weight_sum;
            amrex::Real const local_bulk_2 =
                moment_arr(i, j, k, local_momentum_2_comp) / weight_sum;
            amrex::Real const trial_0 = local_bulk_0
                + KineticThermalSeed(p.idcpu(), 0)
                - state_arr(i, j, k, local_seed_mean_0_comp);
            amrex::Real const trial_1 = local_bulk_1
                + KineticThermalSeed(p.idcpu(), 1)
                - state_arr(i, j, k, local_seed_mean_1_comp);
            amrex::Real const trial_2 = local_bulk_2
                + KineticThermalSeed(p.idcpu(), 2)
                - state_arr(i, j, k, local_seed_mean_2_comp);
            amrex::Gpu::Atomic::AddNoRet(
                &state_arr(i, j, k, seed_reference_energy_comp),
                static_cast<amrex::Real>(wp[ip])
                    * static_cast<amrex::Real>(Algorithms::KineticEnergy(
                        trial_0, trial_1, trial_2, electron_mass)));
        });
    }

    amrex::ReduceOps<amrex::ReduceOpMax> infeasible_reduce_ops;
    amrex::ReduceData<int> infeasible_reduce_data(infeasible_reduce_ops);
    using InfeasibleReduceTuple =
        typename decltype(infeasible_reduce_data)::Type;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(thermal_state, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi)
    {
        amrex::Box const& box = mfi.tilebox();
        amrex::Array4<amrex::Real const> const moment_arr =
            initial_moments.const_array(mfi);
        amrex::Array4<amrex::Real const> const source_arr =
            cell_integrated_energy_source.const_array(mfi);
        amrex::Array4<amrex::Real> const state_arr = thermal_state.array(mfi);
        infeasible_reduce_ops.eval(box, infeasible_reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> InfeasibleReduceTuple
        {
            amrex::Real const source = source_arr(i, j, k);
            amrex::Real const weight = moment_arr(i, j, k, weight_comp);
            if (source == 0.0_rt) {
                state_arr(i, j, k, trial_scale_comp) = 1.0_rt;
                return {0};
            }
            if (!amrex::Math::isfinite(source) || weight <= 0.0_rt
                || !amrex::Math::isfinite(weight))
            {
                return {1};
            }

            amrex::Real const local_bulk_0 =
                moment_arr(i, j, k, local_momentum_0_comp) / weight;
            amrex::Real const local_bulk_1 =
                moment_arr(i, j, k, local_momentum_1_comp) / weight;
            amrex::Real const local_bulk_2 =
                moment_arr(i, j, k, local_momentum_2_comp) / weight;
            amrex::Real const old_energy =
                moment_arr(i, j, k, energy_comp);
            amrex::Real const cold_energy = weight
                * static_cast<amrex::Real>(Algorithms::KineticEnergy(
                    local_bulk_0, local_bulk_1, local_bulk_2, electron_mass));
            amrex::Real const target_energy = old_energy + source;
            amrex::Real const energy_scale = amrex::max(
                std::numeric_limits<amrex::Real>::min(),
                amrex::max(
                    std::abs(old_energy),
                    amrex::max(std::abs(cold_energy), std::abs(source))));
            amrex::Real const tolerance =
                512.0_rt * std::numeric_limits<amrex::Real>::epsilon()
                * energy_scale;
            amrex::Real const old_thermal_energy = old_energy - cold_energy;
            amrex::Real const target_thermal_energy =
                target_energy - cold_energy;

            if (!amrex::Math::isfinite(local_bulk_0)
                || !amrex::Math::isfinite(local_bulk_1)
                || !amrex::Math::isfinite(local_bulk_2)
                || !amrex::Math::isfinite(old_energy)
                || !amrex::Math::isfinite(cold_energy)
                || !amrex::Math::isfinite(target_energy)
                || old_thermal_energy < -tolerance
                || target_thermal_energy < -tolerance)
            {
                return {1};
            }

            state_arr(i, j, k, local_bulk_0_comp) = local_bulk_0;
            state_arr(i, j, k, local_bulk_1_comp) = local_bulk_1;
            state_arr(i, j, k, local_bulk_2_comp) = local_bulk_2;
            state_arr(i, j, k, target_energy_comp) = target_energy;

            if (target_thermal_energy <= tolerance) {
                state_arr(i, j, k, trial_scale_comp) = 0.0_rt;
                return {0};
            }
            amrex::Real reference_thermal_energy = old_thermal_energy;
            if (old_thermal_energy <= tolerance) {
                state_arr(i, j, k, seeded_mode_comp) = 1.0_rt;
                reference_thermal_energy =
                    state_arr(i, j, k, seed_reference_energy_comp)
                    - cold_energy;
                if (reference_thermal_energy <= tolerance
                    || !amrex::Math::isfinite(reference_thermal_energy))
                {
                    return {1};
                }
            }

            amrex::Real const ratio =
                target_thermal_energy / reference_thermal_energy;
            state_arr(i, j, k, lower_scale_comp) = 0.0_rt;
            // Convexity guarantees E(s)-E(0) >= s*(E(1)-E(0)) for s>=1,
            // so max(1,ratio) is a valid upper bracket even relativistically.
            state_arr(i, j, k, upper_scale_comp) = amrex::max(1.0_rt, ratio);
            state_arr(i, j, k, trial_scale_comp) = amrex::min(
                amrex::max(0.0_rt, std::sqrt(ratio)),
                state_arr(i, j, k, upper_scale_comp));
            return {0};
        });
    }
    int infeasible_cell = amrex::get<0>(infeasible_reduce_data.value());
    amrex::ParallelDescriptor::ReduceIntMax(&infeasible_cell, 1);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        infeasible_cell == 0,
        "Kinetic-electron radiation coupling requested an infeasible thermal "
        "update. Cooling cannot go below the cold kinetic energy at fixed local "
        "cell momentum, and heating requires at least two nondegenerate "
        "particles in every affected NGP cell.");

    // The nonrelativistic square-root estimate is normally already close. A
    // fixed number of bracketed Newton iterations keeps the implementation
    // GPU deterministic at the kernel-launch level and retains a safe bracket
    // for relativistic distributions.
    constexpr int root_iterations = 40;
    for (int iteration = 0; iteration < root_iterations; ++iteration) {
        thermal_state.setVal(
            0.0_rt, trial_energy_comp, 2, amrex::IntVect::TheZeroVector());
#ifdef AMREX_USE_OMP
        info.SetDynamic(true);
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (WarpXParIter mfi(electrons, lev, info);
             mfi.isValid(); ++mfi)
        {
            auto& tile = electrons.ParticlesAt(lev, mfi);
            long const np = tile.numParticles();
            auto const ptd = tile.getParticleTileData();
            auto const* const AMREX_RESTRICT wp = ptd.m_rdata[PIdx::w];
            auto const* const AMREX_RESTRICT uxp = ptd.m_rdata[PIdx::ux];
            auto const* const AMREX_RESTRICT uyp = ptd.m_rdata[PIdx::uy];
            auto const* const AMREX_RESTRICT uzp = ptd.m_rdata[PIdx::uz];
            amrex::Array4<amrex::Real> const state_arr =
                thermal_state.array(mfi);
            amrex::Array4<amrex::Real const> const source_arr =
                cell_integrated_energy_source.const_array(mfi);
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ) \
    || defined(WARPX_DIM_RSPHERE)
            auto const get_position = GetParticlePosition<PIdx>(mfi);
#endif
            amrex::For(np, [=] AMREX_GPU_DEVICE (long ip) noexcept
            {
                auto const p = WarpXParticleContainer::ParticleType(ptd, ip);
                auto const [i, j, k] =
                    amrex::getParticleCell(p, plo, dxi).dim3();
                if (source_arr(i, j, k) == 0.0_rt) { return; }
                amrex::Real const local_bulk_0 =
                    state_arr(i, j, k, local_bulk_0_comp);
                amrex::Real const local_bulk_1 =
                    state_arr(i, j, k, local_bulk_1_comp);
                amrex::Real const local_bulk_2 =
                    state_arr(i, j, k, local_bulk_2_comp);
                amrex::Real const scale = state_arr(i, j, k, trial_scale_comp);
                bool const seeded = state_arr(i, j, k, seeded_mode_comp) > 0.5_rt;
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ) \
    || defined(WARPX_DIM_RSPHERE)
                amrex::ParticleReal radius;
                amrex::ParticleReal stored_theta;
                amrex::ParticleReal stored_phi;
                get_position.AsStored(ip, radius, stored_theta, stored_phi);
                (void)radius;
                auto const particle_theta =
                    static_cast<amrex::Real>(stored_theta);
                auto const particle_phi =
                    static_cast<amrex::Real>(stored_phi);
#else
                amrex::Real const particle_theta = 0.0_rt;
                amrex::Real const particle_phi = 0.0_rt;
#endif
                auto const local_velocity = KineticCartesianToLocal(
                    static_cast<amrex::Real>(uxp[ip]),
                    static_cast<amrex::Real>(uyp[ip]),
                    static_cast<amrex::Real>(uzp[ip]),
                    particle_theta, particle_phi);
                amrex::Real const delta_0 = seeded
                    ? KineticThermalSeed(p.idcpu(), 0)
                        - state_arr(i, j, k, local_seed_mean_0_comp)
                    : local_velocity.component_0 - local_bulk_0;
                amrex::Real const delta_1 = seeded
                    ? KineticThermalSeed(p.idcpu(), 1)
                        - state_arr(i, j, k, local_seed_mean_1_comp)
                    : local_velocity.component_1 - local_bulk_1;
                amrex::Real const delta_2 = seeded
                    ? KineticThermalSeed(p.idcpu(), 2)
                        - state_arr(i, j, k, local_seed_mean_2_comp)
                    : local_velocity.component_2 - local_bulk_2;
                amrex::Real const trial_0 = local_bulk_0 + scale * delta_0;
                amrex::Real const trial_1 = local_bulk_1 + scale * delta_1;
                amrex::Real const trial_2 = local_bulk_2 + scale * delta_2;
                amrex::Real const gamma = std::sqrt(
                    1.0_rt + (trial_0 * trial_0 + trial_1 * trial_1
                        + trial_2 * trial_2) / PhysConst::c2);
                auto const physical_weight =
                    static_cast<amrex::Real>(wp[ip]);
                amrex::Gpu::Atomic::AddNoRet(
                    &state_arr(i, j, k, trial_energy_comp),
                    physical_weight * static_cast<amrex::Real>(
                        Algorithms::KineticEnergy(
                            trial_0, trial_1, trial_2, electron_mass)));
                amrex::Gpu::Atomic::AddNoRet(
                    &state_arr(i, j, k, trial_derivative_comp),
                    physical_weight * electron_mass_real / gamma
                        * (trial_0 * delta_0 + trial_1 * delta_1
                            + trial_2 * delta_2));
            });
        }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(thermal_state, amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi)
        {
            amrex::Box const& box = mfi.tilebox();
            amrex::Array4<amrex::Real> const state_arr = thermal_state.array(mfi);
            amrex::Array4<amrex::Real const> const source_arr =
                cell_integrated_energy_source.const_array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (
                int i, int j, int k) noexcept
            {
                if (source_arr(i, j, k) == 0.0_rt) { return; }
                amrex::Real lower = state_arr(i, j, k, lower_scale_comp);
                amrex::Real upper = state_arr(i, j, k, upper_scale_comp);
                amrex::Real const scale = state_arr(i, j, k, trial_scale_comp);
                amrex::Real const residual =
                    state_arr(i, j, k, trial_energy_comp)
                    - state_arr(i, j, k, target_energy_comp);
                if (residual < 0.0_rt) {
                    lower = scale;
                } else {
                    upper = scale;
                }
                state_arr(i, j, k, lower_scale_comp) = lower;
                state_arr(i, j, k, upper_scale_comp) = upper;

                amrex::Real candidate = 0.5_rt * (lower + upper);
                amrex::Real const derivative =
                    state_arr(i, j, k, trial_derivative_comp);
                if (derivative > 0.0_rt && amrex::Math::isfinite(derivative)) {
                    amrex::Real const newton = scale - residual / derivative;
                    if (newton > lower && newton < upper
                        && amrex::Math::isfinite(newton))
                    {
                        candidate = newton;
                    }
                }
                state_arr(i, j, k, trial_scale_comp) = candidate;
            });
        }
    }

#ifdef AMREX_USE_OMP
    info.SetDynamic(true);
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (WarpXParIter mfi(electrons, lev, info);
         mfi.isValid(); ++mfi)
    {
        auto& tile = electrons.ParticlesAt(lev, mfi);
        long const np = tile.numParticles();
        auto const ptd = tile.getParticleTileData();
        auto* const AMREX_RESTRICT uxp = ptd.m_rdata[PIdx::ux];
        auto* const AMREX_RESTRICT uyp = ptd.m_rdata[PIdx::uy];
        auto* const AMREX_RESTRICT uzp = ptd.m_rdata[PIdx::uz];
        amrex::Array4<amrex::Real const> const source_arr =
            cell_integrated_energy_source.const_array(mfi);
        amrex::Array4<amrex::Real const> const state_arr =
            thermal_state.const_array(mfi);
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ) \
    || defined(WARPX_DIM_RSPHERE)
        auto const get_position = GetParticlePosition<PIdx>(mfi);
#endif
        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip) noexcept
        {
            auto const p = WarpXParticleContainer::ParticleType(ptd, ip);
            auto const [i, j, k] =
                amrex::getParticleCell(p, plo, dxi).dim3();
            if (source_arr(i, j, k) == 0.0_rt) { return; }
            amrex::Real const local_bulk_0 =
                state_arr(i, j, k, local_bulk_0_comp);
            amrex::Real const local_bulk_1 =
                state_arr(i, j, k, local_bulk_1_comp);
            amrex::Real const local_bulk_2 =
                state_arr(i, j, k, local_bulk_2_comp);
            amrex::Real const scale = state_arr(i, j, k, trial_scale_comp);
            bool const seeded = state_arr(i, j, k, seeded_mode_comp) > 0.5_rt;
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ) \
    || defined(WARPX_DIM_RSPHERE)
            amrex::ParticleReal radius;
            amrex::ParticleReal stored_theta;
            amrex::ParticleReal stored_phi;
            get_position.AsStored(ip, radius, stored_theta, stored_phi);
            (void)radius;
            auto const particle_theta =
                static_cast<amrex::Real>(stored_theta);
            auto const particle_phi =
                static_cast<amrex::Real>(stored_phi);
#else
            amrex::Real const particle_theta = 0.0_rt;
            amrex::Real const particle_phi = 0.0_rt;
#endif
            auto const local_velocity = KineticCartesianToLocal(
                static_cast<amrex::Real>(uxp[ip]),
                static_cast<amrex::Real>(uyp[ip]),
                static_cast<amrex::Real>(uzp[ip]),
                particle_theta, particle_phi);
            amrex::Real const delta_0 = seeded
                ? KineticThermalSeed(p.idcpu(), 0)
                    - state_arr(i, j, k, local_seed_mean_0_comp)
                : local_velocity.component_0 - local_bulk_0;
            amrex::Real const delta_1 = seeded
                ? KineticThermalSeed(p.idcpu(), 1)
                    - state_arr(i, j, k, local_seed_mean_1_comp)
                : local_velocity.component_1 - local_bulk_1;
            amrex::Real const delta_2 = seeded
                ? KineticThermalSeed(p.idcpu(), 2)
                    - state_arr(i, j, k, local_seed_mean_2_comp)
                : local_velocity.component_2 - local_bulk_2;
            amrex::Real const local_velocity_0 = local_bulk_0 + scale * delta_0;
            amrex::Real const local_velocity_1 = local_bulk_1 + scale * delta_1;
            amrex::Real const local_velocity_2 = local_bulk_2 + scale * delta_2;
            auto const cartesian_velocity = KineticLocalToCartesian(
                local_velocity_0, local_velocity_1, local_velocity_2,
                particle_theta, particle_phi);
            uxp[ip] = static_cast<amrex::ParticleReal>(
                cartesian_velocity.component_0);
            uyp[ip] = static_cast<amrex::ParticleReal>(
                cartesian_velocity.component_1);
            uzp[ip] = static_cast<amrex::ParticleReal>(
                cartesian_velocity.component_2);
        });
    }

    // Re-deposit the achieved energy and momentum. This catches insufficient
    // root convergence and particle-precision loss instead of silently
    // attributing an unrepresented source to the material ledger.
    amrex::MultiFab achieved(
        initial_moments.boxArray(), initial_moments.DistributionMap(), 5, 0);
    achieved.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (WarpXParIter mfi(electrons, lev, info);
         mfi.isValid(); ++mfi)
    {
        auto& tile = electrons.ParticlesAt(lev, mfi);
        long const np = tile.numParticles();
        auto const ptd = tile.getParticleTileData();
        auto const* const AMREX_RESTRICT wp = ptd.m_rdata[PIdx::w];
        auto const* const AMREX_RESTRICT uxp = ptd.m_rdata[PIdx::ux];
        auto const* const AMREX_RESTRICT uyp = ptd.m_rdata[PIdx::uy];
        auto const* const AMREX_RESTRICT uzp = ptd.m_rdata[PIdx::uz];
        amrex::Array4<amrex::Real> const achieved_arr = achieved.array(mfi);
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ) \
    || defined(WARPX_DIM_RSPHERE)
        auto const get_position = GetParticlePosition<PIdx>(mfi);
#endif
        amrex::For(np, [=] AMREX_GPU_DEVICE (long ip) noexcept
        {
            auto const p = WarpXParticleContainer::ParticleType(ptd, ip);
            auto const [i, j, k] =
                amrex::getParticleCell(p, plo, dxi).dim3();
            auto const weight = static_cast<amrex::Real>(wp[ip]);
            amrex::Gpu::Atomic::AddNoRet(
                &achieved_arr(i, j, k, 0),
                weight * static_cast<amrex::Real>(Algorithms::KineticEnergy(
                    uxp[ip], uyp[ip], uzp[ip], electron_mass)));
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ) \
    || defined(WARPX_DIM_RSPHERE)
            amrex::ParticleReal radius;
            amrex::ParticleReal particle_theta;
            amrex::ParticleReal particle_phi;
            get_position.AsStored(
                ip, radius, particle_theta, particle_phi);
            (void)radius;
            auto const local_velocity = KineticCartesianToLocal(
                static_cast<amrex::Real>(uxp[ip]),
                static_cast<amrex::Real>(uyp[ip]),
                static_cast<amrex::Real>(uzp[ip]),
                static_cast<amrex::Real>(particle_theta),
                static_cast<amrex::Real>(particle_phi));
#else
            auto const local_velocity = KineticCartesianToLocal(
                static_cast<amrex::Real>(uxp[ip]),
                static_cast<amrex::Real>(uyp[ip]),
                static_cast<amrex::Real>(uzp[ip]),
                0.0_rt, 0.0_rt);
#endif
            amrex::Gpu::Atomic::AddNoRet(
                &achieved_arr(i, j, k, 1),
                weight * local_velocity.component_0);
            amrex::Gpu::Atomic::AddNoRet(
                &achieved_arr(i, j, k, 2),
                weight * local_velocity.component_1);
            amrex::Gpu::Atomic::AddNoRet(
                &achieved_arr(i, j, k, 3),
                weight * local_velocity.component_2);
            amrex::Gpu::Atomic::AddNoRet(
                &achieved_arr(i, j, k, 4), weight * std::sqrt(
                    static_cast<amrex::Real>(uxp[ip] * uxp[ip]
                        + uyp[ip] * uyp[ip] + uzp[ip] * uzp[ip])));
        });
    }

    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpMax,
                     amrex::ReduceOpMax>
        realization_reduce_ops;
    amrex::ReduceData<amrex::Real, int, int> realization_reduce_data(
        realization_reduce_ops);
    using RealizationReduceTuple =
        typename decltype(realization_reduce_data)::Type;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(achieved, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi)
    {
        amrex::Box const& box = mfi.tilebox();
        amrex::Array4<amrex::Real const> const achieved_arr =
            achieved.const_array(mfi);
        amrex::Array4<amrex::Real const> const moment_arr =
            initial_moments.const_array(mfi);
        amrex::Array4<amrex::Real const> const state_arr =
            thermal_state.const_array(mfi);
        amrex::Array4<amrex::Real const> const source_arr =
            cell_integrated_energy_source.const_array(mfi);
        amrex::Array4<amrex::Real> const material_arr =
            cell_integrated_energy_source.array(mfi);
        realization_reduce_ops.eval(box, realization_reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
                -> RealizationReduceTuple
        {
            if (source_arr(i, j, k) == 0.0_rt) {
                return {0.0_rt, 0, 0};
            }
            amrex::Real const target = state_arr(i, j, k, target_energy_comp);
            amrex::Real const old_energy =
                moment_arr(i, j, k, energy_comp);
            amrex::Real const momentum_tolerance =
                1024.0_rt * KineticPrecisionEpsilon()
                * amrex::max(
                    moment_arr(i, j, k, local_momentum_scale_comp),
                    achieved_arr(i, j, k, 4));
            // Normalize before forming the L2 norm: squaring weighted
            // momenta can overflow single precision even when their relative
            // error is within the roundoff-scaled tolerance.
            amrex::Real momentum_error_ratio_squared = 0.0_rt;
            int momentum_invalid =
                (!amrex::Math::isfinite(momentum_tolerance)
                    || momentum_tolerance < 0.0_rt)
                ? 1
                : 0;
            if (momentum_invalid == 0) {
                for (int component = 0; component < 3; ++component) {
                    amrex::Real const difference =
                        achieved_arr(i, j, k, component + 1)
                        - moment_arr(
                            i, j, k, local_momentum_0_comp + component);
                    amrex::Real const absolute_difference =
                        amrex::Math::abs(difference);
                    if (!amrex::Math::isfinite(difference)
                        || absolute_difference > momentum_tolerance) {
                        momentum_invalid = 1;
                        break;
                    }
                    if (momentum_tolerance == 0.0_rt) { continue; }
                    amrex::Real const ratio = difference / momentum_tolerance;
                    momentum_error_ratio_squared += ratio * ratio;
                }
            }
            if (!amrex::Math::isfinite(momentum_error_ratio_squared)
                || momentum_error_ratio_squared > 1.0_rt) {
                momentum_invalid = 1;
            }
            KineticEnergyUpdateResult const energy_update =
                EvaluateKineticEnergyUpdate(
                    source_arr(i, j, k), old_energy, target,
                    achieved_arr(i, j, k, 0));
            int const energy_invalid = energy_update.valid ? 0 : 1;
            if (energy_invalid != 0 || momentum_invalid != 0) {
                return {0.0_rt, energy_invalid, momentum_invalid};
            }
            material_arr(i, j, k) = energy_update.realized_energy_change;
            return {energy_update.residual, 0, 0};
        });
    }
    auto const realization_reduction = realization_reduce_data.value();
    amrex::Real residual = amrex::get<0>(realization_reduction);
    int unresolved_energy = amrex::get<1>(realization_reduction);
    int unresolved_momentum = amrex::get<2>(realization_reduction);
    amrex::ParallelDescriptor::ReduceRealSum(residual);
    amrex::ParallelDescriptor::ReduceIntMax(&unresolved_energy, 1);
    amrex::ParallelDescriptor::ReduceIntMax(&unresolved_momentum, 1);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        unresolved_energy == 0,
        "Kinetic-electron radiation thermalization failed to reproduce the "
        "requested cell energy within the fixed particle-precision-scaled "
        "acceptance bound. Increase electron particles per cell or inspect the "
        "source magnitude and particle precision.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        unresolved_momentum == 0,
        "Kinetic-electron radiation thermalization failed to preserve cell "
        "local momentum to roundoff-scaled accuracy. Increase electron particles "
        "per cell or inspect particle precision.");
    return residual;
}
}

RadiationTransport::RadiationTransport (
    MultiParticleContainer& particles,
    HybridPICModel const* const hybrid_model,
    warpx::materials::MaterialRegistry const* const material_registry)
    : m_hybrid_model(hybrid_model)
{
    amrex::ParmParse const pp("radiation_transport");
    pp.query("enabled", m_enabled);
    if (!m_enabled) { return; }

    pp.get("photon_species", m_photon_species);
    std::vector<std::string> reduced_diagnostic_names;
    amrex::ParmParse const pp_warpx("warpx");
    pp_warpx.queryarr("reduced_diags_names", reduced_diagnostic_names);
    for (auto const& diagnostic_name : reduced_diagnostic_names) {
        std::string diagnostic_type;
        amrex::ParmParse const pp_diagnostic(diagnostic_name);
        if (pp_diagnostic.query("type", diagnostic_type)
            && (diagnostic_type == "RadiationEnergy"
                || diagnostic_type == "RadiationMomentum"))
        {
            m_track_energy_balance = true;
            break;
        }
    }
    std::string table_interpolation = "linear";
    pp.query("opacity_table_interpolation", table_interpolation);
    warpx::radiation::OpacityTableInterpolation interpolation_mode =
        warpx::radiation::OpacityTableInterpolation::Linear;
    if (table_interpolation == "linear") {
        interpolation_mode =
            warpx::radiation::OpacityTableInterpolation::Linear;
    } else if (table_interpolation == "log_log") {
        interpolation_mode =
            warpx::radiation::OpacityTableInterpolation::LogLog;
    } else {
        WARPX_ABORT_WITH_MESSAGE(
            "Unknown radiation_transport.opacity_table_interpolation='"
            + table_interpolation + "'. Valid values are linear and log_log.");
    }

    std::vector<amrex::Real> energy_group_boundaries;
    utils::parser::queryArrWithParser(
        pp, "energy_group_boundaries", energy_group_boundaries);
    m_num_groups = static_cast<int>(energy_group_boundaries.size()) + 1;
    m_energy_group_boundaries_h.resize(energy_group_boundaries.size());
    std::copy(
        energy_group_boundaries.begin(), energy_group_boundaries.end(),
        m_energy_group_boundaries_h.begin());
    for (int i = 0; i < m_num_groups - 1; ++i) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            amrex::Math::isfinite(m_energy_group_boundaries_h[i])
                && m_energy_group_boundaries_h[i] > 0.0_rt
                && (i == 0
                    || m_energy_group_boundaries_h[i]
                        > m_energy_group_boundaries_h[i - 1]),
            "radiation_transport.energy_group_boundaries must be finite, "
            "positive and strictly increasing.");
    }

    std::vector<amrex::Real> group_photon_energies;
    bool const group_photon_energies_are_set = utils::parser::queryArrWithParser(
        pp, "group_photon_energies", group_photon_energies);
    if (group_photon_energies_are_set) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<int>(group_photon_energies.size()) == m_num_groups,
            "radiation_transport.group_photon_energies must contain exactly "
            "one representative photon energy per energy group.");
        m_group_photon_energies_h.resize(group_photon_energies.size());
        std::copy(
            group_photon_energies.begin(), group_photon_energies.end(),
            m_group_photon_energies_h.begin());
    } else {
        m_group_photon_energies_h.resize(m_num_groups, 0.0_rt);
        if (m_num_groups > 1) {
            m_group_photon_energies_h[0] =
                0.5_rt * m_energy_group_boundaries_h[0];
            for (int group = 1; group < m_num_groups - 1; ++group) {
                m_group_photon_energies_h[group] = std::sqrt(
                    m_energy_group_boundaries_h[group - 1]
                    * m_energy_group_boundaries_h[group]);
            }
            m_group_photon_energies_h[m_num_groups - 1] =
                2.0_rt * m_energy_group_boundaries_h[m_num_groups - 2];
        }
    }

    std::string material_opacity_table_file;
    bool const material_opacity_table_file_is_set = pp.query(
        "material_opacity_table_file", material_opacity_table_file);
    long long material_opacity_table_id = 0;
    bool const material_opacity_table_id_is_set = pp.query(
        "material_opacity_table_id", material_opacity_table_id);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        material_opacity_table_file_is_set
            == material_opacity_table_id_is_set,
        "radiation_transport.material_opacity_table_file and "
        "radiation_transport.material_opacity_table_id must be specified "
        "together.");
    bool registry_opacity_handles_are_set = false;
    warpx::materials::MaterialDefinition const*
        redundant_registry_opacity_material = nullptr;
    pp.queryarr("opacity_species", m_opacity_species);
    if (material_registry != nullptr && material_registry->enabled()) {
        int num_registry_opacity_handles = 0;
        std::vector<std::string> registry_species;
        for (int material = 0; material < material_registry->size(); ++material) {
            auto const& definition = material_registry->material(material);
            num_registry_opacity_handles +=
                static_cast<int>(definition.opacity.has_value());
            registry_species.insert(
                registry_species.end(), definition.species.begin(),
                definition.species.end());
        }
        registry_opacity_handles_are_set = num_registry_opacity_handles != 0;
        if (registry_opacity_handles_are_set) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                num_registry_opacity_handles == material_registry->size(),
                "When any materials.names entry configures native opacity, "
                "every registered non-vacuum material must configure an "
                "opacity table handle.");
            if (material_opacity_table_file_is_set) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    material_registry->size() == 1,
                    "A redundant legacy material opacity file/ID may accompany "
                    "materials.names only for one registered material.");
                redundant_registry_opacity_material =
                    &material_registry->material(0);
                warpx::materials::MaterialTableHandle const configured{
                    material_opacity_table_file,
                    static_cast<std::int64_t>(material_opacity_table_id),
                    redundant_registry_opacity_material->opacity->material_key};
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    warpx::materials::MaterialRegistry::sameTableHandle(
                        configured,
                        *redundant_registry_opacity_material->opacity),
                    "Legacy native opacity file/material ID does not match "
                    "the sole materials.names opacity handle.");
            }
            std::sort(registry_species.begin(), registry_species.end());
            registry_species.erase(
                std::unique(registry_species.begin(), registry_species.end()),
                registry_species.end());
            if (!m_opacity_species.empty()) {
                auto configured_species = m_opacity_species;
                std::sort(
                    configured_species.begin(), configured_species.end());
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    configured_species == registry_species,
                    "radiation_transport.opacity_species must exactly match "
                    "the sorted union of every registered material carrier "
                    "species when registry-native opacity is enabled.");
            }
            // Canonicalize evaluator and deposition order independently of
            // both material and opacity_species declaration order.
            m_opacity_species = std::move(registry_species);
        }
    }
    m_use_material_opacity_table =
        material_opacity_table_file_is_set
        || registry_opacity_handles_are_set;
#ifndef WARPX_USE_MATERIAL_OPACITY_HDF5
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_use_material_opacity_table,
        "Native material opacity tables require a WarpX build configured "
        "with WarpX_MATERIAL_OPACITY_HDF5=ON.");
#else
    m_use_registered_material_opacity_tables =
        registry_opacity_handles_are_set
        && !material_opacity_table_file_is_set;
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_use_material_opacity_table || !m_opacity_species.empty(),
        "Native material opacity requires at least one carrier species to "
        "compute material mass density.");
    // With a native table, opacity_species supplies density only. It must not
    // select the legacy additive per-species opacity backend.
    m_use_species_opacity = !m_opacity_species.empty()
        && !m_use_material_opacity_table;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<int>(m_opacity_species.size()) <= max_opacity_species,
        "radiation_transport.opacity_species supports at most "
        + std::to_string(max_opacity_species) + " material species.");
    for (std::size_t i = 0; i < m_opacity_species.size(); ++i) {
        std::string const& species_name = m_opacity_species[i];
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::find(m_opacity_species.begin(), m_opacity_species.begin() + i,
                      species_name) == m_opacity_species.begin() + i,
            "radiation_transport.opacity_species contains duplicate species '"
            + species_name + "'.");
        auto& species = particles.GetParticleContainerFromName(species_name);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !species.AmIA<PhysicalSpecies::photon>()
                && species.getMass() > 0.0_prt
                && species.getCharge() > 0.0_prt,
            "radiation_transport.opacity_species entry '" + species_name
                + "' must be a massive, positively charged ion species.");
        m_opacity_species_masses[static_cast<int>(i)] =
            static_cast<amrex::Real>(species.getMass());
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
        if (m_use_registered_material_opacity_tables) {
            int material_index = -1;
            for (int material = 0; material < material_registry->size();
                 ++material)
            {
                auto const& carriers =
                    material_registry->material(material).species;
                if (std::find(
                        carriers.begin(), carriers.end(), species_name)
                    != carriers.end())
                {
                    material_index = material;
                    break;
                }
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                material_index >= 0,
                "Internal registry opacity error: carrier species '"
                    + species_name + "' has no material index.");
            m_opacity_species_material_indices[i] = material_index;
        }
#endif
    }

    if (m_use_material_opacity_table) {
        bool legacy_backend_is_set = false;
        for (char const* key : {
                 "absorption_coefficient",
                 "absorption_coefficient(x,y,z,t,photon_energy,ne,Te)",
                 "absorption_coefficient_table_file",
                 "planck_absorption_coefficient",
                 "planck_absorption_coefficient(x,y,z,t,ne,Te)",
                 "planck_absorption_coefficient(x,y,z,t,photon_energy,ne,Te)",
                 "planck_absorption_coefficient_table_file",
                 "planck_absorption_coefficient_spectral_table_file",
                 "rosseland_transport_coefficient",
                 "rosseland_transport_coefficient(x,y,z,t,ne,Te)",
                 "rosseland_transport_coefficient(x,y,z,t,photon_energy,ne,Te)",
                 "rosseland_transport_coefficient_table_file",
                 "rosseland_transport_coefficient_spectral_table_file"})
        {
            legacy_backend_is_set = legacy_backend_is_set || pp.contains(key);
        }
        for (std::string const& species_name : m_opacity_species) {
            for (std::string const& key : {
                     "absorption_coefficient_" + species_name
                         + "(x,y,z,t,photon_energy,ni,ne,Te)",
                     "absorption_coefficient_table_file_" + species_name,
                     "planck_absorption_coefficient_" + species_name
                         + "(x,y,z,t,photon_energy,ni,ne,Te)",
                     "planck_absorption_coefficient_table_file_" + species_name,
                     "planck_absorption_coefficient_spectral_table_file_"
                         + species_name,
                     "rosseland_transport_coefficient_" + species_name
                         + "(x,y,z,t,photon_energy,ni,ne,Te)",
                     "rosseland_transport_coefficient_table_file_" + species_name,
                     "rosseland_transport_coefficient_spectral_table_file_"
                         + species_name})
            {
                legacy_backend_is_set = legacy_backend_is_set || pp.contains(key);
            }
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !legacy_backend_is_set,
            "radiation_transport.material_opacity_table_file is mutually "
            "exclusive with every legacy global and per-species absorption, "
            "Planck, and Rosseland opacity backend.");
    }

#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
    if (m_use_material_opacity_table) {
        auto validate_configured_groups = [&] (
            warpx::radiation::MaterialOpacityTable const& table,
            std::string const& description)
        {
            auto const& table_edges = table.groupEdges();
            auto const& table_representatives =
                table.representativeEnergies();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<int>(table_representatives.size()) == m_num_groups
                && table_edges.size() == table_representatives.size() + 1U,
            description + " must contain exactly the number of groups "
            "configured by "
            "radiation_transport.energy_group_boundaries.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            table_edges.front() == 0.0_rt
                && std::isinf(table_edges.back())
                && table_edges.back() > 0.0_rt,
            description + " must have exact 0 and +inf outer group edges.");
        for (int group = 0; group < m_num_groups - 1; ++group) {
            amrex::Real const table_edge = table_edges[group + 1];
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                table_edge > 0.0_rt && amrex::Math::isfinite(table_edge)
                    && (group == 0
                        || table_edge > table_edges[group])
                    && OpacityGroupValuesAgree(
                        m_energy_group_boundaries_h[group], table_edge),
                "The finite internal edges in " + description
                + " must agree with "
                "radiation_transport.energy_group_boundaries.");
        }
        if (group_photon_energies_are_set) {
            for (int group = 0; group < m_num_groups; ++group) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    OpacityGroupValuesAgree(
                        m_group_photon_energies_h[group],
                        table_representatives[group]),
                    "radiation_transport.group_photon_energies must agree "
                    "with the representative energies in " + description
                    + ".");
            }
        } else {
            m_group_photon_energies_h.resize(table_representatives.size());
            std::copy(
                table_representatives.begin(), table_representatives.end(),
                m_group_photon_energies_h.begin());
        }
        };

        if (m_use_registered_material_opacity_tables) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                material_registry != nullptr && material_registry->enabled(),
                "Internal registry opacity error: registry is unavailable.");
            m_registered_material_selector = material_registry->selector();
            std::set<std::string> material_keys;
            std::set<std::pair<std::string, std::int64_t>> selected_tables;
            amrex::Vector<amrex::Real> canonical_edges;
            amrex::Vector<amrex::Real> canonical_representatives;
            for (int material = 0; material < material_registry->size();
                 ++material)
            {
                auto const& definition = material_registry->material(material);
                auto const& handle = *definition.opacity;
                std::string const normalized_file =
                    std::filesystem::path(handle.file)
                        .lexically_normal().string();
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    material_keys.insert(handle.material_key).second,
                    "Registered opacity material_key values must be unique; "
                    "duplicate '" + handle.material_key + "'.");
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    selected_tables.emplace(
                        normalized_file, handle.material_id).second,
                    "Each registered material must select a unique opacity "
                    "file/material-ID pair; duplicate '" + normalized_file
                    + "' ID " + std::to_string(handle.material_id) + ".");

                auto& table =
                    m_registered_material_opacity_tables[material];
                table.load(handle.file, handle.material_id);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    table.materialId() == handle.material_id
                        && table.materialKey() == handle.material_key,
                    "Registered opacity table metadata does not match named "
                    "material '" + definition.name + "'.");
                std::string const description =
                    "Registered opacity table for material '"
                    + definition.name + "'";
                validate_configured_groups(table, description);

                if (material == 0) {
                    canonical_edges = table.groupEdges();
                    canonical_representatives =
                        table.representativeEnergies();
                } else {
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        OpacityGroupArraysAgree(
                            table.groupEdges(), canonical_edges)
                            && OpacityGroupArraysAgree(
                                table.representativeEnergies(),
                                canonical_representatives),
                        "All registered opacity tables must have identical "
                        "group edges and representative energies; mismatch "
                        "for material '" + definition.name + "'.");
                }
            }
        } else {
            m_material_opacity_table.load(
                material_opacity_table_file,
                static_cast<std::int64_t>(material_opacity_table_id));
            if (redundant_registry_opacity_material != nullptr) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    m_material_opacity_table.materialKey()
                        == redundant_registry_opacity_material->opacity
                            ->material_key,
                    "Legacy native opacity table material_key does not match "
                    "the sole materials.names opacity handle.");
            }
            validate_configured_groups(
                m_material_opacity_table,
                "The selected legacy material opacity table");
        }
    }
#endif

    auto compile_species_parser = [&] (
        std::string const& key,
        std::vector<amrex::Parser>& parsers,
        auto& executors,
        int const species_index)
    {
        std::string expression;
        utils::parser::Store_parserString(pp, key, expression);
        parsers.emplace_back(utils::parser::makeParser(
            expression,
            {"x", "y", "z", "t", "photon_energy", "ni", "ne", "Te"}));
        executors[species_index] = parsers.back().template compile<8>();
    };

    auto configure_species_absorption = [&]
    {
        m_species_absorption_coefficient_parsers.reserve(
            m_opacity_species.size());
        int const num_species = static_cast<int>(m_opacity_species.size());
        for (int i = 0; i < num_species; ++i) {
            auto const species_index = static_cast<std::size_t>(i);
            std::string const& species_name =
                m_opacity_species[species_index];
            std::string parser_key = "absorption_coefficient_";
            parser_key += species_name;
            parser_key += "(x,y,z,t,photon_energy,ni,ne,Te)";
            std::string table_key = "absorption_coefficient_table_file_";
            table_key += species_name;
            bool const parser_is_set = pp.contains(parser_key);
            std::string table_file;
            bool const table_is_set = pp.query(table_key, table_file);
            std::string selection_error =
                "Species-aware opacity requires exactly one of radiation_transport.";
            selection_error += parser_key;
            selection_error += " and radiation_transport.";
            selection_error += table_key;
            selection_error += ".";
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                static_cast<int>(parser_is_set)
                        + static_cast<int>(table_is_set)
                    == 1,
                selection_error);
            if (parser_is_set) {
                m_species_absorption_backends[i] = static_cast<int>(
                    SpeciesOpacityBackend::Parser);
                compile_species_parser(
                    parser_key,
                    m_species_absorption_coefficient_parsers,
                    m_species_absorption_coefficients, i);
                continue;
            }
            m_species_absorption_backends[i] = static_cast<int>(
                SpeciesOpacityBackend::Table3D);
            m_species_absorption_tables[species_index].load(
                table_file, interpolation_mode);
            m_species_absorption_table_executors[i] =
                m_species_absorption_tables[species_index].executor();
        }
    };

    auto configure_species_mean_opacity = [&] (
        std::string const& coefficient_name,
        std::vector<amrex::Parser>& parsers,
        auto& parser_executors,
        auto& backends,
        auto& tables,
        auto& table_executors,
        auto& spectral_tables,
        auto& spectral_table_executors)
    {
        parsers.reserve(m_opacity_species.size());
        int const num_species = static_cast<int>(m_opacity_species.size());
        for (int i = 0; i < num_species; ++i) {
            auto const species_index = static_cast<std::size_t>(i);
            std::string const& species_name =
                m_opacity_species[species_index];
            std::string parser_key = coefficient_name;
            parser_key += "_";
            parser_key += species_name;
            parser_key += "(x,y,z,t,photon_energy,ni,ne,Te)";
            std::string table_key = coefficient_name;
            table_key += "_table_file_";
            table_key += species_name;
            std::string spectral_table_key = coefficient_name;
            spectral_table_key += "_spectral_table_file_";
            spectral_table_key += species_name;
            bool const parser_is_set = pp.contains(parser_key);
            std::string table_file;
            std::string spectral_table_file;
            bool const table_is_set = pp.query(table_key, table_file);
            bool const spectral_table_is_set = pp.query(
                spectral_table_key, spectral_table_file);
            std::string selection_error =
                "Species-aware opacity requires exactly one of radiation_transport.";
            selection_error += parser_key;
            selection_error += ", radiation_transport.";
            selection_error += table_key;
            selection_error += " and radiation_transport.";
            selection_error += spectral_table_key;
            selection_error += ".";
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                static_cast<int>(parser_is_set)
                        + static_cast<int>(table_is_set)
                        + static_cast<int>(spectral_table_is_set)
                    == 1,
                selection_error);
            if (parser_is_set) {
                backends[i] = static_cast<int>(SpeciesOpacityBackend::Parser);
                compile_species_parser(
                    parser_key, parsers, parser_executors, i);
            } else if (table_is_set) {
                backends[i] = static_cast<int>(SpeciesOpacityBackend::Table2D);
                tables[species_index].load(table_file, interpolation_mode);
                table_executors[i] = tables[species_index].executor();
            } else {
                backends[i] = static_cast<int>(SpeciesOpacityBackend::Table3D);
                spectral_tables[species_index].load(
                    spectral_table_file, interpolation_mode);
                spectral_table_executors[i] =
                    spectral_tables[species_index].executor();
            }
        }
    };

    std::string absorption_table_file;
    m_use_absorption_table =
        pp.query("absorption_coefficient_table_file", absorption_table_file);
    bool const absorption_scalar_is_set =
        pp.contains("absorption_coefficient");
    bool const absorption_function_is_set = pp.contains(
        "absorption_coefficient(x,y,z,t,photon_energy,ne,Te)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<int>(m_use_absorption_table)
                + static_cast<int>(absorption_scalar_is_set)
                + static_cast<int>(absorption_function_is_set)
                + static_cast<int>(m_use_species_opacity)
                + static_cast<int>(m_use_material_opacity_table)
            == 1,
        "Specify exactly one of radiation_transport.absorption_coefficient, "
        "radiation_transport.absorption_coefficient(x,y,z,t,photon_energy,ne,Te), "
        "radiation_transport.absorption_coefficient_table_file, and the "
        "per-species backends selected by radiation_transport.opacity_species, "
        "or radiation_transport.material_opacity_table_file.");
    if (m_use_species_opacity) {
        configure_species_absorption();
    } else if (m_use_absorption_table) {
        m_absorption_table.load(absorption_table_file, interpolation_mode);
    } else if (!m_use_material_opacity_table) {
        amrex::Real constant_absorption_coefficient = 0.0_rt;
        if (utils::parser::queryWithParser(
                pp, "absorption_coefficient", constant_absorption_coefficient))
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                constant_absorption_coefficient >= 0.0_rt,
                "radiation_transport.absorption_coefficient must be non-negative.");
            m_absorption_coefficient_parser = utils::parser::makeParser(
                std::to_string(constant_absorption_coefficient),
                {"x", "y", "z", "t", "photon_energy", "ne", "Te"});
        } else {
            std::string absorption_coefficient_function;
            utils::parser::Store_parserString(
                pp,
                "absorption_coefficient(x,y,z,t,photon_energy,ne,Te)",
                absorption_coefficient_function);
            m_absorption_coefficient_parser = utils::parser::makeParser(
                absorption_coefficient_function,
                {"x", "y", "z", "t", "photon_energy", "ne", "Te"});
        }
        m_absorption_coefficient = m_absorption_coefficient_parser.compile<7>();
    }
    pp.query("enable_lte_exchange", m_enable_lte_exchange);
    std::string lte_exchange_solver = "frozen_temperature";
    pp.query("lte_exchange_solver", lte_exchange_solver);
    if (lte_exchange_solver == "frozen_temperature") {
        m_lte_exchange_solver = LteExchangeSolver::FrozenTemperature;
    } else if (lte_exchange_solver == "implicit_temperature") {
        m_lte_exchange_solver = LteExchangeSolver::ImplicitTemperature;
    } else {
        WARPX_ABORT_WITH_MESSAGE(
            "Unknown radiation_transport.lte_exchange_solver='"
            + lte_exchange_solver
            + "'. Valid values are frozen_temperature and implicit_temperature.");
    }
    utils::parser::queryWithParser(
        pp, "lte_exchange_max_iterations", m_lte_exchange_max_iterations);
    utils::parser::queryWithParser(
        pp, "lte_exchange_tolerance", m_lte_exchange_tolerance);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_lte_exchange_max_iterations > 0,
        "radiation_transport.lte_exchange_max_iterations must be positive.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_lte_exchange_tolerance > 0.0_rt
            && m_lte_exchange_tolerance < 1.0_rt,
        "radiation_transport.lte_exchange_tolerance must be in (0, 1).");
    bool spectral_planck_function_is_set = false;
    if (m_enable_lte_exchange) {
        std::string planck_table_file;
        m_use_planck_table = pp.query(
            "planck_absorption_coefficient_table_file", planck_table_file);
        std::string spectral_planck_table_file;
        m_use_spectral_planck_table = pp.query(
            "planck_absorption_coefficient_spectral_table_file",
            spectral_planck_table_file);
        bool const planck_scalar_is_set =
            pp.contains("planck_absorption_coefficient");
        bool const planck_function_is_set = pp.contains(
            "planck_absorption_coefficient(x,y,z,t,ne,Te)");
        spectral_planck_function_is_set = pp.contains(
            "planck_absorption_coefficient(x,y,z,t,photon_energy,ne,Te)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<int>(m_use_planck_table)
                    + static_cast<int>(m_use_spectral_planck_table)
                    + static_cast<int>(planck_scalar_is_set)
                    + static_cast<int>(planck_function_is_set)
                    + static_cast<int>(spectral_planck_function_is_set)
                    + static_cast<int>(m_use_species_opacity)
                    + static_cast<int>(m_use_material_opacity_table)
                == 1,
            "When LTE exchange is enabled, specify exactly one of "
            "radiation_transport.planck_absorption_coefficient, its parser "
            "functions, planck_absorption_coefficient_table_file and "
            "planck_absorption_coefficient_spectral_table_file, or the "
            "per-species backends selected by opacity_species, or "
            "material_opacity_table_file.");
        bool const use_planck_parser = !m_use_species_opacity
            && !m_use_material_opacity_table
            && !m_use_planck_table && !m_use_spectral_planck_table;
        if (m_use_species_opacity) {
            configure_species_mean_opacity(
                "planck_absorption_coefficient",
                m_species_planck_absorption_coefficient_parsers,
                m_species_planck_absorption_coefficients,
                m_species_planck_backends,
                m_species_planck_tables,
                m_species_planck_table_executors,
                m_species_spectral_planck_tables,
                m_species_spectral_planck_table_executors);
        }
        if (m_use_planck_table) {
            m_planck_table.load(planck_table_file, interpolation_mode);
        }
        if (m_use_spectral_planck_table) {
            m_spectral_planck_table.load(
                spectral_planck_table_file, interpolation_mode);
        }
        if (use_planck_parser) {
            amrex::Real constant_planck_coefficient = 0.0_rt;
            if (utils::parser::queryWithParser(
                    pp, "planck_absorption_coefficient",
                    constant_planck_coefficient))
            {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    constant_planck_coefficient >= 0.0_rt,
                    "radiation_transport.planck_absorption_coefficient must be "
                    "non-negative.");
                m_planck_absorption_coefficient_parser =
                    utils::parser::makeParser(
                        std::to_string(constant_planck_coefficient),
                        {"x", "y", "z", "t", "photon_energy", "ne", "Te"});
            } else {
                std::string planck_coefficient_function;
                if (planck_function_is_set) {
                    utils::parser::Store_parserString(
                        pp,
                        "planck_absorption_coefficient(x,y,z,t,ne,Te)",
                        planck_coefficient_function);
                } else {
                    utils::parser::Store_parserString(
                        pp,
                        "planck_absorption_coefficient(x,y,z,t,photon_energy,ne,Te)",
                        planck_coefficient_function);
                }
                m_planck_absorption_coefficient_parser =
                    utils::parser::makeParser(
                        planck_coefficient_function,
                        {"x", "y", "z", "t", "photon_energy", "ne", "Te"});
            }
            m_planck_absorption_coefficient =
                m_planck_absorption_coefficient_parser.compile<7>();
        }
    }
    pp.query("enable_diffusion", m_enable_diffusion);
    bool spectral_rosseland_function_is_set = false;
    if (m_enable_diffusion) {
        std::string rosseland_table_file;
        m_use_rosseland_table = pp.query(
            "rosseland_transport_coefficient_table_file", rosseland_table_file);
        std::string spectral_rosseland_table_file;
        m_use_spectral_rosseland_table = pp.query(
            "rosseland_transport_coefficient_spectral_table_file",
            spectral_rosseland_table_file);
        bool const rosseland_scalar_is_set =
            pp.contains("rosseland_transport_coefficient");
        bool const rosseland_function_is_set = pp.contains(
            "rosseland_transport_coefficient(x,y,z,t,ne,Te)");
        spectral_rosseland_function_is_set = pp.contains(
            "rosseland_transport_coefficient(x,y,z,t,photon_energy,ne,Te)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<int>(m_use_rosseland_table)
                    + static_cast<int>(m_use_spectral_rosseland_table)
                    + static_cast<int>(rosseland_scalar_is_set)
                    + static_cast<int>(rosseland_function_is_set)
                    + static_cast<int>(spectral_rosseland_function_is_set)
                    + static_cast<int>(m_use_species_opacity)
                    + static_cast<int>(m_use_material_opacity_table)
                == 1,
            "When diffusion is enabled, specify exactly one of "
            "radiation_transport.rosseland_transport_coefficient, its parser "
            "functions, rosseland_transport_coefficient_table_file and "
            "rosseland_transport_coefficient_spectral_table_file, or the "
            "per-species backends selected by opacity_species, or "
            "material_opacity_table_file.");
        bool const use_rosseland_parser = !m_use_species_opacity
            && !m_use_material_opacity_table
            && !m_use_rosseland_table && !m_use_spectral_rosseland_table;
        if (m_use_species_opacity) {
            configure_species_mean_opacity(
                "rosseland_transport_coefficient",
                m_species_rosseland_transport_coefficient_parsers,
                m_species_rosseland_transport_coefficients,
                m_species_rosseland_backends,
                m_species_rosseland_tables,
                m_species_rosseland_table_executors,
                m_species_spectral_rosseland_tables,
                m_species_spectral_rosseland_table_executors);
        }
        if (m_use_rosseland_table) {
            m_rosseland_table.load(rosseland_table_file, interpolation_mode);
        }
        if (m_use_spectral_rosseland_table) {
            m_spectral_rosseland_table.load(
                spectral_rosseland_table_file, interpolation_mode);
        }
        if (use_rosseland_parser) {
            amrex::Real constant_rosseland_coefficient = 0.0_rt;
            if (utils::parser::queryWithParser(
                    pp, "rosseland_transport_coefficient",
                    constant_rosseland_coefficient))
            {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    constant_rosseland_coefficient >= 0.0_rt,
                    "radiation_transport.rosseland_transport_coefficient must be "
                    "non-negative.");
                m_rosseland_transport_coefficient_parser =
                    utils::parser::makeParser(
                        std::to_string(constant_rosseland_coefficient),
                        {"x", "y", "z", "t", "photon_energy", "ne", "Te"});
            } else {
                std::string rosseland_coefficient_function;
                if (rosseland_function_is_set) {
                    utils::parser::Store_parserString(
                        pp,
                        "rosseland_transport_coefficient(x,y,z,t,ne,Te)",
                        rosseland_coefficient_function);
                } else {
                    utils::parser::Store_parserString(
                        pp,
                        "rosseland_transport_coefficient(x,y,z,t,photon_energy,ne,Te)",
                        rosseland_coefficient_function);
                }
                m_rosseland_transport_coefficient_parser =
                    utils::parser::makeParser(
                        rosseland_coefficient_function,
                        {"x", "y", "z", "t", "photon_energy", "ne", "Te"});
            }
            m_rosseland_transport_coefficient =
                m_rosseland_transport_coefficient_parser.compile<7>();
        }
    }
    pp.query("enable_particle_conversion", m_enable_particle_conversion);
    pp.query("enable_momentum_coupling", m_enable_momentum_coupling);
#if defined(WARPX_DIM_RSPHERE)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_enable_momentum_coupling,
        "radiation_transport.enable_momentum_coupling=1 is not supported in "
        "RSPHERE until Cartesian-to-spherical particle initialization is "
        "corrected upstream and component-wise radiation-momentum validation "
        "passes.");
#endif
    if (m_enable_momentum_coupling) {
        pp.getarr("momentum_species", m_momentum_species);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_momentum_species.empty(),
            "radiation_transport.enable_momentum_coupling=1 requires at least "
            "one massive ion in radiation_transport.momentum_species.");
    }
    if (m_enable_particle_conversion) {
        if (m_num_groups == 1 && !group_photon_energies_are_set
            && !m_use_material_opacity_table)
        {
            utils::parser::getWithParser(
                pp, "emission_photon_energy", m_emission_photon_energy);
            m_group_photon_energies_h[0] = m_emission_photon_energy;
        } else {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                group_photon_energies_are_set
                    || m_use_material_opacity_table,
                "Multigroup particle conversion requires "
                "radiation_transport.group_photon_energies or representative "
                "energies supplied by material_opacity_table_file.");
            if (pp.contains("emission_photon_energy")) {
                ablastr::warn_manager::WMRecordWarning(
                    "Radiation transport",
                    "Ignoring radiation_transport.emission_photon_energy because "
                    "group representative photon energies are already set.",
                    ablastr::warn_manager::WarnPriority::low);
            }
            m_emission_photon_energy = m_group_photon_energies_h[0];
        }
        utils::parser::queryWithParser(
            pp, "particle_conversion_energy_threshold",
            m_particle_conversion_energy_threshold);
    }
    utils::parser::queryWithParser(pp, "path_cell_fraction", m_path_cell_fraction);
    pp.query(
        "require_cell_interface_exact_streaming",
        m_require_cell_interface_exact_streaming);
#if defined(WARPX_DIM_RCYLINDER)
    if (m_require_cell_interface_exact_streaming) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::particle_boundary_hi[0] == ParticleBoundaryType::Open,
            "radiation_transport.require_cell_interface_exact_streaming=1 "
            "requires an Open outer radial particle boundary.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_enable_momentum_coupling,
            "radiation_transport.require_cell_interface_exact_streaming=1 "
            "is incompatible with radiation_transport.enable_momentum_coupling=1.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_enable_particle_conversion,
            "radiation_transport.require_cell_interface_exact_streaming=1 "
            "is incompatible with radiation_transport.enable_particle_conversion=1.");
    } else {
        ablastr::warn_manager::WMRecordWarning(
            "Radiation transport",
            "Streaming attenuation in radial geometry still samples and deposits "
            "each bounded path segment in its starting radial cell. Set "
            "radiation_transport.require_cell_interface_exact_streaming=1 to use "
            "face-exact streaming in the supported RCYLINDER subset.",
            ablastr::warn_manager::WarnPriority::low);
    }
#elif defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_require_cell_interface_exact_streaming,
        "radiation_transport.require_cell_interface_exact_streaming=1 is not "
        "supported in RZ geometry.");
    ablastr::warn_manager::WMRecordWarning(
        "Radiation transport",
        "Streaming attenuation in radial geometry still samples and deposits "
        "each bounded path segment in its starting radial cell. Set "
        "radiation_transport.require_cell_interface_exact_streaming=1 to make "
        "this unsupported approximation a startup error.",
        ablastr::warn_manager::WarnPriority::low);
#elif defined(WARPX_DIM_RSPHERE)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_require_cell_interface_exact_streaming,
        "radiation_transport.require_cell_interface_exact_streaming=1 is not "
        "supported in RSPHERE geometry.");
    ablastr::warn_manager::WMRecordWarning(
        "Radiation transport",
        "Streaming attenuation in radial geometry still samples and deposits "
        "each bounded path segment in its starting radial cell. Set "
        "radiation_transport.require_cell_interface_exact_streaming=1 to make "
        "this unsupported approximation a startup error.",
        ablastr::warn_manager::WarnPriority::low);
#endif
    utils::parser::queryWithParser(pp, "diffusion_cfl", m_diffusion_cfl);
    utils::parser::queryWithParser(
        pp, "minimum_diffusion_optical_depth",
        m_minimum_diffusion_optical_depth);
    pp.query("max_transport_substeps", m_max_transport_substeps);
    pp.query("max_diffusion_substeps", m_max_diffusion_substeps);
    bool const minimum_density_is_set = utils::parser::queryWithParser(
        pp, "minimum_electron_density", m_minimum_electron_density);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_minimum_electron_density >= 0.0_rt,
        "radiation_transport.minimum_electron_density must be non-negative.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_path_cell_fraction > 0.0_rt && m_path_cell_fraction <= 1.0_rt,
        "radiation_transport.path_cell_fraction must be in (0, 1].");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_max_transport_substeps > 0,
        "radiation_transport.max_transport_substeps must be positive.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_diffusion_cfl > 0.0_rt && m_diffusion_cfl <= 0.5_rt,
        "radiation_transport.diffusion_cfl must be in (0, 0.5].");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_minimum_diffusion_optical_depth > 0.0_rt,
        "radiation_transport.minimum_diffusion_optical_depth must be positive.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_max_diffusion_substeps > 0,
        "radiation_transport.max_diffusion_substeps must be positive.");
    ParseDiffusionBoundaryArray(
        pp, "diffusion_boundary_lo", m_diffusion_boundary_lo);
    ParseDiffusionBoundaryArray(
        pp, "diffusion_boundary_hi", m_diffusion_boundary_hi);
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ) \
    || defined(WARPX_DIM_RSPHERE)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_diffusion_boundary_lo[0] == static_cast<int>(
            DiffusionBoundary::Reflecting),
        "The radial lower face is a zero-area symmetry/axis boundary. "
        "radiation_transport.diffusion_boundary_lo must be reflecting.");
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_enable_particle_conversion || m_emission_photon_energy > 0.0_rt,
        "Every photon energy used for particle conversion must be positive.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_particle_conversion_energy_threshold >= 0.0_rt,
        "radiation_transport.particle_conversion_energy_threshold must be "
        "non-negative.");

    bool const representative_energies_are_required =
        m_num_groups > 1 || group_photon_energies_are_set
        || m_enable_particle_conversion || m_use_spectral_planck_table
        || m_use_spectral_rosseland_table || spectral_planck_function_is_set
        || spectral_rosseland_function_is_set
        || m_use_material_opacity_table
        || (m_use_species_opacity
            && (m_enable_lte_exchange || m_enable_diffusion));
    for (int group = 0;
         representative_energies_are_required && group < m_num_groups; ++group)
    {
        amrex::Real const energy = m_group_photon_energies_h[group];
        bool in_group = amrex::Math::isfinite(energy) && energy > 0.0_rt;
        if (group > 0) {
            in_group = in_group
                && energy >= m_energy_group_boundaries_h[group - 1];
        }
        if (group < m_num_groups - 1) {
            in_group = in_group
                && energy < m_energy_group_boundaries_h[group];
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            in_group,
            "Each radiation_transport.group_photon_energies entry must be "
            "finite, positive and lie inside its energy group.");
    }

    m_energy_groups = {
        m_energy_group_boundaries_h.data(), m_group_photon_energies_h.data(),
        m_num_groups};
#ifdef AMREX_USE_GPU
    m_energy_group_boundaries_d.resize(m_energy_group_boundaries_h.size());
    m_group_photon_energies_d.resize(m_group_photon_energies_h.size());
    amrex::Gpu::copyAsync(
        amrex::Gpu::hostToDevice,
        m_energy_group_boundaries_h.begin(), m_energy_group_boundaries_h.end(),
        m_energy_group_boundaries_d.begin());
    amrex::Gpu::copyAsync(
        amrex::Gpu::hostToDevice,
        m_group_photon_energies_h.begin(), m_group_photon_energies_h.end(),
        m_group_photon_energies_d.begin());
    amrex::Gpu::streamSynchronize();
    m_energy_groups.m_boundaries = m_energy_group_boundaries_d.data();
    m_energy_groups.m_representative_energies =
        m_group_photon_energies_d.data();
#endif

    auto const& photons = particles.GetParticleContainerFromName(m_photon_species);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        photons.AmIA<PhysicalSpecies::photon>(),
        "radiation_transport.photon_species must name a photon species.");
    auto* const photon_container =
        dynamic_cast<PhotonParticleContainer*>(
            &particles.GetParticleContainerFromName(m_photon_species));
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        photon_container != nullptr,
        "Radiation transport requires a PhotonParticleContainer.");
    photon_container->setRadiationTransportManaged(true);

    std::string material_coupling = "none";
    pp.query("material_coupling", material_coupling);
    if (material_coupling == "none") {
        m_material_coupling = MaterialCoupling::None;
    } else if (material_coupling == "hybrid_electrons") {
        m_material_coupling = MaterialCoupling::HybridElectrons;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC,
            "radiation_transport.material_coupling=hybrid_electrons requires "
            "algo.maxwell_solver=hybrid.");

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_hybrid_model != nullptr,
            "Hybrid radiation coupling requires a HybridPICModel instance.");
        if (!minimum_density_is_set) {
            m_minimum_electron_density =
                m_hybrid_model->electronDensityFloor();
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_hybrid_model->solvesElectronEnergyEquation(),
            "radiation_transport.material_coupling=hybrid_electrons requires "
            "hybrid_pic_model.solve_electron_energy_equation=1. An algebraic "
            "electron-pressure closure cannot retain deposited radiation energy.");
    } else if (material_coupling == "kinetic_electrons") {
        m_material_coupling = MaterialCoupling::KineticElectrons;
        pp.get("electron_species", m_electron_species);
        auto const& electrons =
            particles.GetParticleContainerFromName(m_electron_species);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            electrons.getCharge() < 0.0_prt && electrons.getMass() > 0.0_prt,
            "radiation_transport.electron_species must name a massive, "
            "negatively charged kinetic species.");
    } else {
        WARPX_ABORT_WITH_MESSAGE(
            "Unknown radiation_transport.material_coupling='" + material_coupling
            + "'. Valid values are none, hybrid_electrons and kinetic_electrons.");
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_enable_lte_exchange || m_material_coupling != MaterialCoupling::None,
        "radiation_transport.enable_lte_exchange requires hybrid_electrons or "
        "kinetic_electrons material coupling.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_use_material_opacity_table
            || m_material_coupling != MaterialCoupling::None,
        "radiation_transport.material_opacity_table_file requires "
        "material_coupling=hybrid_electrons or kinetic_electrons to supply "
        "the local electron temperature.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_enable_diffusion || m_enable_lte_exchange,
        "radiation_transport.enable_diffusion requires enable_lte_exchange=1.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_enable_particle_conversion || m_enable_diffusion,
        "radiation_transport.enable_particle_conversion requires "
        "enable_diffusion=1.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_enable_particle_conversion || !m_enable_momentum_coupling,
        "radiation_transport.enable_particle_conversion is not compatible with "
        "enable_momentum_coupling=1 because the scalar diffusion field does not "
        "retain the converted packet momentum.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_material_coupling != MaterialCoupling::HybridElectrons
            || (m_hybrid_model != nullptr
                && m_hybrid_model
                    ->electronThermodynamicsSupportsEnergyCoupling()),
        "Hybrid radiation coupling requires configured electron thermodynamics "
        "with a finite positive caloric heat capacity.");
    m_use_nonlinear_hybrid_lte_remap = m_enable_lte_exchange
        && m_material_coupling == MaterialCoupling::HybridElectrons
        && m_lte_exchange_solver == LteExchangeSolver::ImplicitTemperature
        && !m_hybrid_model
            ->electronThermodynamicsSupportsConstantHeatCapacityLte();

    if (m_enable_momentum_coupling) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::numeric_limits<amrex::ParticleReal>::digits >= 53,
            "Radiation momentum coupling currently requires DOUBLE particle "
            "precision so small impulses on a moving liner are not rounded away.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_material_coupling == MaterialCoupling::HybridElectrons
                || m_material_coupling == MaterialCoupling::KineticElectrons,
            "Conservative radiation-momentum coupling requires "
            "radiation_transport.material_coupling=hybrid_electrons or "
            "kinetic_electrons. The ion impulse adapter is shared; kinetic "
            "electrons receive the remaining cell energy after ion work is "
            "debited.");
        for (std::size_t i = 0; i < m_momentum_species.size(); ++i) {
            std::string const& species_name = m_momentum_species[i];
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::find(
                    m_momentum_species.begin(),
                    m_momentum_species.begin() + static_cast<std::ptrdiff_t>(i),
                    species_name) == m_momentum_species.begin()
                        + static_cast<std::ptrdiff_t>(i),
                "Duplicate species '" + species_name
                    + "' in radiation_transport.momentum_species.");
            auto const& species =
                particles.GetParticleContainerFromName(species_name);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !species.AmIA<PhysicalSpecies::photon>()
                    && species.getMass() > 0.0_prt
                    && species.getCharge() > 0.0_prt,
                "radiation_transport.momentum_species entry '" + species_name
                    + "' must be a massive, positively charged ion species.");
        }
        m_track_energy_balance = true;
    }
}

void
RadiationTransport::RecordMaterialEnergyRealizationResidual (
    amrex::Real const residual) const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        amrex::Math::isfinite(residual),
        "Radiation material-energy realization residual must be finite.");
    m_last_numerical_energy_residual += residual;
    m_cumulative_numerical_energy_residual += residual;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        amrex::Math::isfinite(m_last_numerical_energy_residual)
            && amrex::Math::isfinite(m_cumulative_numerical_energy_residual),
        "Radiation material-energy realization residual overflowed its "
        "diagnostic ledger.");
}

void
RadiationTransport::WriteCheckpointData (std::string const& dir) const
{
    if (!m_enabled) { return; }
    // This counter feeds the live circuit load and is independent of output
    // cadence.
    std::ofstream checkpoint{
        dir + "/RadiationTransport_data.txt", std::ofstream::out};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        checkpoint.good(),
        "RadiationTransport could not write its checkpoint state.");
    checkpoint.precision(17);
    checkpoint << m_cumulative_boundary_energy_loss << "\n"
               << m_cumulative_numerical_energy_residual << "\n";
}

void
RadiationTransport::ReadCheckpointData (std::string const& dir)
{
    if (!m_enabled) { return; }
    std::ifstream checkpoint{
        dir + "/RadiationTransport_data.txt", std::ifstream::in};
    if (!checkpoint.good()) {
        m_cumulative_boundary_energy_loss = 0.0_rt;
        m_cumulative_numerical_energy_residual = 0.0_rt;
        m_last_numerical_energy_residual = 0.0_rt;
        ablastr::warn_manager::WMRecordWarning(
            "Radiation transport",
            "The restart checkpoint predates RadiationTransport cumulative "
            "boundary accounting. The live cumulative escaped-energy counter "
            "will restart from zero.",
            ablastr::warn_manager::WarnPriority::low);
        return;
    }
    amrex::Real cumulative_boundary_energy_loss = 0.0_rt;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<bool>(checkpoint >> cumulative_boundary_energy_loss)
            && amrex::Math::isfinite(cumulative_boundary_energy_loss)
            && cumulative_boundary_energy_loss >= 0.0_rt,
        "RadiationTransport checkpoint state is truncated or invalid.");
    m_cumulative_boundary_energy_loss = cumulative_boundary_energy_loss;

    std::string numerical_energy_residual_token;
    if (!(checkpoint >> numerical_energy_residual_token)) {
        if (checkpoint.eof()) {
            checkpoint.clear();
            m_cumulative_numerical_energy_residual = 0.0_rt;
            ablastr::warn_manager::WMRecordWarning(
                "Radiation transport",
                "The restart checkpoint predates signed numerical-energy "
                "residual accounting. The cumulative numerical-energy residual "
                "will restart from zero.",
                ablastr::warn_manager::WarnPriority::low);
        } else {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                false,
                "RadiationTransport checkpoint numerical-energy residual is "
                "malformed.");
        }
    } else {
        amrex::Real cumulative_numerical_energy_residual = 0.0_rt;
        std::istringstream numerical_energy_residual_stream{
            numerical_energy_residual_token};
        std::string residual_trailing_token;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<bool>(numerical_energy_residual_stream
                >> cumulative_numerical_energy_residual)
                && !(numerical_energy_residual_stream >> residual_trailing_token),
            "RadiationTransport checkpoint numerical-energy residual is "
            "malformed.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            amrex::Math::isfinite(cumulative_numerical_energy_residual),
            "RadiationTransport checkpoint numerical-energy residual is "
            "non-finite.");
        std::string trailing_token;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !(checkpoint >> trailing_token),
            "RadiationTransport checkpoint state has unexpected trailing data.");
        m_cumulative_numerical_energy_residual =
            cumulative_numerical_energy_residual;
    }
    m_last_numerical_energy_residual = 0.0_rt;
}

void
RadiationTransport::AllocateLevelMFs (
    ablastr::fields::MultiFabRegister& fields,
    int const lev,
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm) const
{
    if (!m_enabled) { return; }

    // The ledger stores energy, rather than energy density, so every material
    // adapter can remap it conservatively to its own degrees of freedom.
    fields.alloc_init(
        FieldType::radiation_material_energy, lev, ba, dm, 1,
        amrex::IntVect(1), 0.0_rt);
    fields.alloc_init(
        FieldType::radiation_material_kinetic_energy, lev, ba, dm, 1,
        amrex::IntVect(1), 0.0_rt);
    fields.alloc_init(
        FieldType::radiation_material_momentum, lev, ba, dm, 3,
        amrex::IntVect(1), 0.0_rt);
    if (m_use_nonlinear_hybrid_lte_remap) {
        fields.alloc_init(
            FieldType::radiation_hybrid_lte_remap, lev, ba, dm,
            nonlinear_lte_remap_components, amrex::IntVect(1), 0.0_rt,
            /*remake=*/true,
            /*redistribute_on_remake=*/true,
            /*checkpoint_restart=*/false);
    }
    if (m_enable_lte_exchange) {
        fields.alloc_init(
            FieldType::radiation_diffusion_energy, lev, ba, dm, m_num_groups,
            amrex::IntVect(1), 0.0_rt,
            /*remake=*/true,
            /*redistribute_on_remake=*/true,
            /*checkpoint_restart=*/true);
    }
}

namespace
{
/** Apply a cell-integrated impulse to the configured massive material species.
 *
 * The same proper-velocity increment is applied to every selected particle in
 * a cell. The returned field stores the actual relativistic kinetic-energy
 * change, so the caller can make the radiation/material energy update exactly
 * paired with the momentum update.
 *
 * Keep this as a free implementation function: NVCC does not permit extended
 * device lambdas in a private or protected member function.
 */
void
ApplyMaterialImpulse (
    MultiParticleContainer& particles,
    std::vector<std::string> const& momentum_species,
    amrex::MultiFab const& cell_integrated_impulse,
    amrex::MultiFab& cell_integrated_applied_impulse,
    amrex::MultiFab& cell_integrated_kinetic_energy_change)
{
    ABLASTR_PROFILE("RadiationTransport::ApplyMaterialImpulse()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        cell_integrated_impulse.nComp() == 3
            && cell_integrated_impulse.ixType().cellCentered(),
        "Radiation material impulse must be a three-component cell-centered field.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        cell_integrated_applied_impulse.nComp() == 3
            && cell_integrated_applied_impulse.ixType().cellCentered(),
        "Applied radiation impulse must be a three-component cell-centered field.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        cell_integrated_kinetic_energy_change.nComp() == 1
            && cell_integrated_kinetic_energy_change.ixType().cellCentered(),
        "Radiation kinetic-work ledger must be a scalar cell-centered field.");

    cell_integrated_applied_impulse.setVal(0.0_rt);
    cell_integrated_kinetic_energy_change.setVal(0.0_rt);
    amrex::MultiFab cell_material_mass(
        cell_integrated_impulse.boxArray(),
        cell_integrated_impulse.DistributionMap(), 1, 0);
    cell_material_mass.setVal(0.0_rt);

    auto& warpx = WarpX::GetInstance();
    constexpr int lev = 0;
    auto const plo = warpx.Geom(lev).ProbLoArray();
    auto const dxi = warpx.Geom(lev).InvCellSizeArray();

    amrex::MFItInfo info;
    if (amrex::Gpu::notInLaunchRegion()) {
        info.EnableTiling(WarpXParticleContainer::tile_size);
    }

    // First form the physical material mass in every NGP cell. The impulse is
    // then shared as one common proper-velocity increment, so the selected ion
    // mixture receives exactly the requested cell momentum without changing
    // its relative drifts or thermal spread intentionally.
    for (std::string const& species_name : momentum_species) {
        auto& species = particles.GetParticleContainerFromName(species_name);
        amrex::ParticleReal const species_mass = species.getMass();
#ifdef AMREX_USE_OMP
        info.SetDynamic(true);
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (WarpXParIter mfi(species, lev, info); mfi.isValid(); ++mfi) {
            auto& tile = mfi.GetParticleTile();
            long const np = mfi.numParticles();
            auto const ptd = tile.getParticleTileData();
            auto const* const AMREX_RESTRICT wp = ptd.m_rdata[PIdx::w];
            amrex::Array4<amrex::Real> const mass_arr =
                cell_material_mass.array(mfi);

            // Different particles can deposit into the same cell.
            amrex::For(np, [=] AMREX_GPU_DEVICE (long ip) noexcept
            {
                auto const p = WarpXParticleContainer::ParticleType(ptd, ip);
                auto const [i, j, k] =
                    amrex::getParticleCell(p, plo, dxi).dim3();
                amrex::Gpu::Atomic::AddNoRet(
                    &mass_arr(i, j, k),
                    static_cast<amrex::Real>(wp[ip] * species_mass));
            });
        }
    }

    amrex::Gpu::DeviceScalar<int> missing_material_mass(0);
    int* const missing_material_mass_ptr = missing_material_mass.dataPtr();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(cell_material_mass, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi)
    {
        amrex::Box const& box = mfi.tilebox();
        amrex::Array4<amrex::Real const> const mass_arr =
            cell_material_mass.const_array(mfi);
        amrex::Array4<amrex::Real const> const impulse_arr =
            cell_integrated_impulse.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (
            int i, int j, int k) noexcept
        {
            amrex::Real const impulse_norm = std::sqrt(
                impulse_arr(i, j, k, 0) * impulse_arr(i, j, k, 0)
                + impulse_arr(i, j, k, 1) * impulse_arr(i, j, k, 1)
                + impulse_arr(i, j, k, 2) * impulse_arr(i, j, k, 2));
            if (impulse_norm > 0.0_rt && mass_arr(i, j, k) <= 0.0_rt) {
                amrex::HostDevice::Atomic::Add(
                    missing_material_mass_ptr, 1);
            }
        });
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        missing_material_mass.dataValue() == 0,
        "Radiation deposited momentum in a cell containing no configured "
        "radiation_transport.momentum_species mass.");

    for (std::string const& species_name : momentum_species) {
        auto& species = particles.GetParticleContainerFromName(species_name);
        amrex::ParticleReal const species_mass = species.getMass();
#ifdef AMREX_USE_OMP
        info.SetDynamic(true);
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (WarpXParIter mfi(species, lev, info); mfi.isValid(); ++mfi) {
            auto& tile = mfi.GetParticleTile();
            long const np = mfi.numParticles();
            auto const ptd = tile.getParticleTileData();
            auto const* const AMREX_RESTRICT wp = ptd.m_rdata[PIdx::w];
            auto* const AMREX_RESTRICT uxp = ptd.m_rdata[PIdx::ux];
            auto* const AMREX_RESTRICT uyp = ptd.m_rdata[PIdx::uy];
            auto* const AMREX_RESTRICT uzp = ptd.m_rdata[PIdx::uz];
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ) \
    || defined(WARPX_DIM_RSPHERE)
            auto const get_position = GetParticlePosition<PIdx>(mfi);
#endif
            amrex::Array4<amrex::Real const> const impulse_arr =
                cell_integrated_impulse.const_array(mfi);
            amrex::Array4<amrex::Real const> const mass_arr =
                cell_material_mass.const_array(mfi);
            amrex::Array4<amrex::Real> const kinetic_arr =
                cell_integrated_kinetic_energy_change.array(mfi);
            amrex::Array4<amrex::Real> const applied_impulse_arr =
                cell_integrated_applied_impulse.array(mfi);

            // Different particles can contribute kinetic work to the same cell.
            amrex::For(np, [=] AMREX_GPU_DEVICE (long ip) noexcept
            {
                auto const p = WarpXParticleContainer::ParticleType(ptd, ip);
                auto const [i, j, k] =
                    amrex::getParticleCell(p, plo, dxi).dim3();
                amrex::Real const material_mass = mass_arr(i, j, k);
                if (material_mass <= 0.0_rt) { return; }

                auto const old_ux = static_cast<amrex::Real>(uxp[ip]);
                auto const old_uy = static_cast<amrex::Real>(uyp[ip]);
                auto const old_uz = static_cast<amrex::Real>(uzp[ip]);
                auto const delta_u_r =
                    static_cast<amrex::ParticleReal>(
                        impulse_arr(i, j, k, 0) / material_mass);
                auto const delta_u_theta =
                    static_cast<amrex::ParticleReal>(
                        impulse_arr(i, j, k, 1) / material_mass);
                auto const delta_u_z =
                    static_cast<amrex::ParticleReal>(
                        impulse_arr(i, j, k, 2) / material_mass);

#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
                amrex::ParticleReal radius;
                amrex::ParticleReal position_theta;
                amrex::ParticleReal z;
                get_position.AsStored(ip, radius, position_theta, z);
                (void)radius;
                (void)z;
                amrex::ParticleReal const cos_theta =
                    std::cos(position_theta);
                amrex::ParticleReal const sin_theta =
                    std::sin(position_theta);
                uxp[ip] += delta_u_r * cos_theta
                    - delta_u_theta * sin_theta;
                uyp[ip] += delta_u_r * sin_theta
                    + delta_u_theta * cos_theta;
                uzp[ip] += delta_u_z;
#elif defined(WARPX_DIM_RSPHERE)
                amrex::ParticleReal radius;
                amrex::ParticleReal position_theta;
                amrex::ParticleReal position_phi;
                get_position.AsStored(
                    ip, radius, position_theta, position_phi);
                (void)radius;
                amrex::ParticleReal const cos_theta =
                    std::cos(position_theta);
                amrex::ParticleReal const sin_theta =
                    std::sin(position_theta);
                amrex::ParticleReal const cos_phi =
                    std::cos(position_phi);
                amrex::ParticleReal const sin_phi =
                    std::sin(position_phi);
                auto const delta_u_phi = delta_u_z;
                uxp[ip] += delta_u_r * cos_theta * cos_phi
                    - delta_u_theta * sin_theta
                    - delta_u_phi * cos_theta * sin_phi;
                uyp[ip] += delta_u_r * sin_theta * cos_phi
                    + delta_u_theta * cos_theta
                    - delta_u_phi * sin_theta * sin_phi;
                uzp[ip] += delta_u_r * sin_phi
                    + delta_u_phi * cos_phi;
#else
                uxp[ip] += delta_u_r;
                uyp[ip] += delta_u_theta;
                uzp[ip] += delta_u_z;
#endif

                amrex::Real const actual_delta_ux =
                    static_cast<amrex::Real>(uxp[ip]) - old_ux;
                amrex::Real const actual_delta_uy =
                    static_cast<amrex::Real>(uyp[ip]) - old_uy;
                amrex::Real const actual_delta_uz =
                    static_cast<amrex::Real>(uzp[ip]) - old_uz;
                amrex::Real const gamma_old = std::sqrt(
                    1.0_rt + (old_ux * old_ux + old_uy * old_uy
                        + old_uz * old_uz) / PhysConst::c2);
                amrex::Real const new_ux = old_ux + actual_delta_ux;
                amrex::Real const new_uy = old_uy + actual_delta_uy;
                amrex::Real const new_uz = old_uz + actual_delta_uz;
                amrex::Real const gamma_new = std::sqrt(
                    1.0_rt + (new_ux * new_ux + new_uy * new_uy
                        + new_uz * new_uz) / PhysConst::c2);
                amrex::Real const delta_u_squared =
                    actual_delta_ux * actual_delta_ux
                    + actual_delta_uy * actual_delta_uy
                    + actual_delta_uz * actual_delta_uz;
                amrex::Real const old_u_dot_delta =
                    old_ux * actual_delta_ux + old_uy * actual_delta_uy
                    + old_uz * actual_delta_uz;
                auto const weighted_mass =
                    static_cast<amrex::Real>(wp[ip] * species_mass);
                amrex::Gpu::Atomic::AddNoRet(
                    &kinetic_arr(i, j, k),
                    weighted_mass
                        * (2.0_rt * old_u_dot_delta + delta_u_squared)
                        / (gamma_old + gamma_new));

#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
                amrex::Real const actual_delta_u_r =
                    actual_delta_ux * cos_theta
                    + actual_delta_uy * sin_theta;
                amrex::Real const actual_delta_u_theta =
                    -actual_delta_ux * sin_theta
                    + actual_delta_uy * cos_theta;
                amrex::Gpu::Atomic::AddNoRet(
                    &applied_impulse_arr(i, j, k, 0),
                    weighted_mass * actual_delta_u_r);
                amrex::Gpu::Atomic::AddNoRet(
                    &applied_impulse_arr(i, j, k, 1),
                    weighted_mass * actual_delta_u_theta);
                amrex::Gpu::Atomic::AddNoRet(
                    &applied_impulse_arr(i, j, k, 2),
                    weighted_mass * actual_delta_uz);
#elif defined(WARPX_DIM_RSPHERE)
                amrex::Real const actual_delta_u_r =
                    actual_delta_ux * cos_theta * cos_phi
                    + actual_delta_uy * sin_theta * cos_phi
                    + actual_delta_uz * sin_phi;
                amrex::Real const actual_delta_u_theta =
                    -actual_delta_ux * sin_theta
                    + actual_delta_uy * cos_theta;
                amrex::Real const actual_delta_u_phi =
                    -actual_delta_ux * cos_theta * sin_phi
                    - actual_delta_uy * sin_theta * sin_phi
                    + actual_delta_uz * cos_phi;
                amrex::Gpu::Atomic::AddNoRet(
                    &applied_impulse_arr(i, j, k, 0),
                    weighted_mass * actual_delta_u_r);
                amrex::Gpu::Atomic::AddNoRet(
                    &applied_impulse_arr(i, j, k, 1),
                    weighted_mass * actual_delta_u_theta);
                amrex::Gpu::Atomic::AddNoRet(
                    &applied_impulse_arr(i, j, k, 2),
                    weighted_mass * actual_delta_u_phi);
#else
                amrex::Gpu::Atomic::AddNoRet(
                    &applied_impulse_arr(i, j, k, 0),
                    weighted_mass * actual_delta_ux);
                amrex::Gpu::Atomic::AddNoRet(
                    &applied_impulse_arr(i, j, k, 1),
                    weighted_mass * actual_delta_uy);
                amrex::Gpu::Atomic::AddNoRet(
                    &applied_impulse_arr(i, j, k, 2),
                    weighted_mass * actual_delta_uz);
#endif
            });
        }
    }

    amrex::Gpu::DeviceScalar<int> unresolved_impulse(0);
    int* const unresolved_impulse_ptr = unresolved_impulse.dataPtr();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(
             cell_integrated_impulse, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi)
    {
        amrex::Box const& box = mfi.tilebox();
        amrex::Array4<amrex::Real const> const requested_arr =
            cell_integrated_impulse.const_array(mfi);
        amrex::Array4<amrex::Real const> const applied_arr =
            cell_integrated_applied_impulse.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (
            int i, int j, int k) noexcept
        {
            amrex::Real requested_norm_squared = 0.0_rt;
            amrex::Real difference_norm_squared = 0.0_rt;
            for (int component = 0; component < 3; ++component) {
                amrex::Real const requested =
                    requested_arr(i, j, k, component);
                amrex::Real const difference =
                    applied_arr(i, j, k, component) - requested;
                requested_norm_squared += requested * requested;
                difference_norm_squared += difference * difference;
            }
            if (requested_norm_squared > 0.0_rt
                && difference_norm_squared
                    > 1.0e-20_rt * requested_norm_squared)
            {
                amrex::HostDevice::Atomic::Add(
                    unresolved_impulse_ptr, 1);
            }
        });
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        unresolved_impulse.dataValue() == 0,
        "A radiation impulse could not be represented in the configured ion "
        "particle momenta to 1e-10 relative accuracy. Increase the radiation "
        "timestep or macroparticle impulse, or inspect the material state.");
}

/** Apply radiation momentum to material and debit the paired kinetic work.
 *
 * The energy reservoir is the absorbed-packet material energy for streaming
 * recoil and the diffusion-radiation energy for FLD recoil. Keeping both paths
 * here guarantees that the actual particle kinetic-energy change, including
 * finite particle-precision rounding, is paired with the energy update. For a
 * multigroup reservoir, the measured total work is distributed in proportion
 * to the surviving group energies. This closes total energy exactly, but is a
 * group-integrated work approximation rather than a frequency-resolved work
 * source.
 */
amrex::Real
ApplyRadiationMomentumWork (
    MultiParticleContainer& particles,
    std::vector<std::string> const& momentum_species,
    amrex::MultiFab const& requested_material_momentum,
    amrex::MultiFab& accumulated_material_momentum,
    amrex::MultiFab& accumulated_material_kinetic_energy,
    amrex::MultiFab& energy_reservoir,
    char const* const insufficient_energy_message)
{
    amrex::MultiFab applied_material_momentum(
        accumulated_material_momentum.boxArray(),
        accumulated_material_momentum.DistributionMap(), 3, 0);
    amrex::MultiFab material_kinetic_energy_change(
        accumulated_material_kinetic_energy.boxArray(),
        accumulated_material_kinetic_energy.DistributionMap(), 1, 0);
    ApplyMaterialImpulse(
        particles, momentum_species, requested_material_momentum,
        applied_material_momentum, material_kinetic_energy_change);
    amrex::MultiFab::Add(
        accumulated_material_momentum, applied_material_momentum,
        0, 0, 3, 0);
    amrex::MultiFab::Add(
        accumulated_material_kinetic_energy, material_kinetic_energy_change,
        0, 0, 1, 0);

    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpMax>
        residual_reduce_ops;
    amrex::ReduceData<amrex::Real, int> residual_reduce_data(
        residual_reduce_ops);
    using ResidualReduceTuple = typename decltype(residual_reduce_data)::Type;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(energy_reservoir, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi)
    {
        amrex::Box const& box = mfi.tilebox();
        amrex::Array4<amrex::Real> const energy_arr =
            energy_reservoir.array(mfi);
        amrex::Array4<amrex::Real const> const work_arr =
            material_kinetic_energy_change.const_array(mfi);
        int const num_energy_components = energy_reservoir.nComp();
        residual_reduce_ops.eval(box, residual_reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ResidualReduceTuple
        {
            amrex::Real old_energy = 0.0_rt;
            for (int group = 0; group < num_energy_components; ++group) {
                old_energy += energy_arr(i, j, k, group);
            }
            amrex::Real const work = work_arr(i, j, k);
            RadiationEnergyUpdateResult const update =
                ApplyRadiationEnergyUpdate(old_energy, -work);
            if (!update.valid) {
                return {0.0_rt, 1};
            }
            if (num_energy_components == 1 || !(old_energy > 0.0_rt)) {
                energy_arr(i, j, k, 0) = update.stored_energy;
                for (int group = 1; group < num_energy_components; ++group) {
                    energy_arr(i, j, k, group) = 0.0_rt;
                }
                return {update.residual, 0};
            }
            amrex::Real const scale = amrex::max(
                0.0_rt, update.stored_energy / old_energy);
            for (int group = 0; group < num_energy_components; ++group) {
                energy_arr(i, j, k, group) *= scale;
            }
            return {update.residual, 0};
        });
    }
    auto const reduction = residual_reduce_data.value();
    amrex::Real residual = amrex::get<0>(reduction);
    int const invalid_work = amrex::get<1>(reduction);
    amrex::ParallelDescriptor::ReduceRealSum(residual);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        invalid_work == 0, insufficient_energy_message);
    return residual;
}

/** Accumulate the four-momentum carried out by streaming photon packets.
 *
 * Conversion-invalid packets have zero weight. Therefore, immediately after
 * particle boundary conditions and before redistribution, an invalid packet
 * with positive residual weight is exactly a packet removed by an absorbing
 * particle boundary in this transport substep.
 */
void
AccumulateStreamingBoundaryLoss (
    PhotonParticleContainer& photons,
    int const lev,
    amrex::Real* const streaming_boundary_energy,
    amrex::GpuArray<amrex::Real*, 3> const& streaming_boundary_momentum)
{
    amrex::MFItInfo info;
    if (amrex::Gpu::notInLaunchRegion()) {
        info.EnableTiling(WarpXParticleContainer::tile_size);
    }
#ifdef AMREX_USE_OMP
    info.SetDynamic(true);
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (WarpXParIter mfi(photons, lev, info); mfi.isValid(); ++mfi)
    {
        auto& tile = mfi.GetParticleTile();
        long const np = mfi.numParticles();
        auto const ptd = tile.getParticleTileData();
        auto const* const AMREX_RESTRICT idcpu = ptd.m_idcpu;
        auto const* const AMREX_RESTRICT wp = ptd.m_rdata[PIdx::w];
        auto const* const AMREX_RESTRICT uxp = ptd.m_rdata[PIdx::ux];
        auto const* const AMREX_RESTRICT uyp = ptd.m_rdata[PIdx::uy];
        auto const* const AMREX_RESTRICT uzp = ptd.m_rdata[PIdx::uz];
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ) \
    || defined(WARPX_DIM_RSPHERE)
        auto const get_position = GetParticlePosition<PIdx>(mfi);
#endif

        amrex::For(np, [=] AMREX_GPU_DEVICE (long ip) noexcept
        {
            if (amrex::ParticleIDWrapper{idcpu[ip]}.is_valid()
                || wp[ip] <= 0.0_prt)
            {
                return;
            }
            amrex::ParticleReal const weight = wp[ip];
            amrex::ParticleReal const energy = weight
                * Algorithms::KineticEnergyPhotons(
                    uxp[ip], uyp[ip], uzp[ip]);
            amrex::HostDevice::Atomic::Add(
                streaming_boundary_energy, static_cast<amrex::Real>(energy));

            amrex::ParticleReal const momentum_scale =
                weight * PhysConst::m_e;
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
            amrex::ParticleReal radius;
            amrex::ParticleReal theta;
            amrex::ParticleReal position_z;
            get_position.AsStored(ip, radius, theta, position_z);
            (void)radius;
            (void)position_z;
            amrex::ParticleReal const cos_theta = std::cos(theta);
            amrex::ParticleReal const sin_theta = std::sin(theta);
            amrex::HostDevice::Atomic::Add(
                streaming_boundary_momentum[0],
                static_cast<amrex::Real>(
                    momentum_scale * (uxp[ip] * cos_theta
                        + uyp[ip] * sin_theta)));
            amrex::HostDevice::Atomic::Add(
                streaming_boundary_momentum[1],
                static_cast<amrex::Real>(
                    momentum_scale * (-uxp[ip] * sin_theta
                        + uyp[ip] * cos_theta)));
            amrex::HostDevice::Atomic::Add(
                streaming_boundary_momentum[2], static_cast<amrex::Real>(
                    momentum_scale * uzp[ip]));
#elif defined(WARPX_DIM_RSPHERE)
            amrex::ParticleReal radius;
            amrex::ParticleReal theta;
            amrex::ParticleReal phi;
            get_position.AsStored(ip, radius, theta, phi);
            (void)radius;
            amrex::ParticleReal const cos_theta = std::cos(theta);
            amrex::ParticleReal const sin_theta = std::sin(theta);
            amrex::ParticleReal const cos_phi = std::cos(phi);
            amrex::ParticleReal const sin_phi = std::sin(phi);
            amrex::HostDevice::Atomic::Add(
                streaming_boundary_momentum[0],
                static_cast<amrex::Real>(
                    momentum_scale * (uxp[ip] * cos_theta * cos_phi
                        + uyp[ip] * sin_theta * cos_phi
                        + uzp[ip] * sin_phi)));
            amrex::HostDevice::Atomic::Add(
                streaming_boundary_momentum[1],
                static_cast<amrex::Real>(
                    momentum_scale * (-uxp[ip] * sin_theta
                        + uyp[ip] * cos_theta)));
            amrex::HostDevice::Atomic::Add(
                streaming_boundary_momentum[2],
                static_cast<amrex::Real>(
                    momentum_scale * (-uxp[ip] * cos_theta * sin_phi
                        - uyp[ip] * sin_theta * sin_phi
                        + uzp[ip] * cos_phi)));
#else
            amrex::HostDevice::Atomic::Add(
                streaming_boundary_momentum[0], static_cast<amrex::Real>(
                    momentum_scale * uxp[ip]));
            amrex::HostDevice::Atomic::Add(
                streaming_boundary_momentum[1], static_cast<amrex::Real>(
                    momentum_scale * uyp[ip]));
            amrex::HostDevice::Atomic::Add(
                streaming_boundary_momentum[2], static_cast<amrex::Real>(
                    momentum_scale * uzp[ip]));
#endif
        });
    }
}

[[nodiscard]]
amrex::Real
TotalRadiationEnergy (
    PhotonParticleContainer& photons,
    ablastr::fields::MultiFabRegister& fields,
    int const level,
    int const num_groups)
{
    amrex::Real total = photons.sumParticleEnergy(/*local=*/false);
    if (fields.has(FieldType::radiation_diffusion_energy, level)) {
        auto const* diffusion = fields.get(
            FieldType::radiation_diffusion_energy, level);
        for (int group = 0; group < num_groups; ++group) {
            total += diffusion->sum(group, /*local=*/false);
        }
    }
    return total;
}
}

void
// Advance coordinates the mutually exclusive transport backends and ledgers.
// NOLINTNEXTLINE(readability-function-size)
RadiationTransport::Advance (
    MultiParticleContainer& particles,
    ablastr::fields::MultiFabRegister& fields,
    amrex::Real const current_time,
    amrex::Real const dt) const
{
    ABLASTR_PROFILE("RadiationTransport::Advance()");
    if (!m_enabled) { return; }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        dt >= 0.0_rt, "RadiationTransport requires a non-negative timestep.");

    auto& warpx = WarpX::GetInstance();
#if defined(WARPX_DIM_RCYLINDER)
    if (m_require_cell_interface_exact_streaming) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            warpx.Geom(0).ProbLo(0) == 0.0_rt,
            "radiation_transport.require_cell_interface_exact_streaming=1 "
            "requires an axis-containing RCYLINDER domain with "
            "geometry.prob_lo[0]==0.");
    }
#endif

    m_last_boundary_energy_loss = 0.0_rt;
    m_last_diffusion_boundary_energy_loss = 0.0_rt;
    m_last_streaming_boundary_energy_loss = 0.0_rt;
    m_last_numerical_energy_residual = 0.0_rt;
    m_last_diffusion_boundary_momentum_loss = {0.0_rt, 0.0_rt, 0.0_rt};
    m_last_streaming_boundary_momentum_loss = {0.0_rt, 0.0_rt, 0.0_rt};

    auto* const photon_container =
        dynamic_cast<PhotonParticleContainer*>(
            &particles.GetParticleContainerFromName(m_photon_species));
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        photon_container != nullptr,
        "Radiation transport requires a PhotonParticleContainer.");
    auto& photons = *photon_container;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        warpx.finestLevel() == 0 && photons.finestLevel() == 0,
        "Radiation transport does not yet support mesh refinement.");

    amrex::Real initial_radiation_energy = 0.0_rt;
    if (m_track_energy_balance) {
        initial_radiation_energy =
            TotalRadiationEnergy(photons, fields, 0, m_num_groups);
    }

    auto const density_floor = m_minimum_electron_density;
    bool const gate_on_hybrid_density = couplesToHybridElectrons();
    bool const gate_on_kinetic_density = couplesToKineticElectrons();
    ElectronThermodynamicsExecutor hybrid_thermodynamics;
    amrex::Real hybrid_thermodynamic_density_floor = 0.0_rt;
    if (gate_on_hybrid_density) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_hybrid_model != nullptr,
            "Hybrid radiation coupling lost its HybridPICModel owner.");
        // Fetch this view after HybridPICModel::InitData. Future table-backed
        // executors can then refer to material data finalized during InitData
        // without caching a stale constructor-time view.
        hybrid_thermodynamics =
            m_hybrid_model->electronThermodynamicsExecutor();
        hybrid_thermodynamic_density_floor =
            m_hybrid_model->transportsElectronInternalEnergyAtRawDensity()
                ? 0.0_rt
                : m_hybrid_model->electronDensityFloor();
    }
    WarpXParticleContainer* kinetic_electrons = nullptr;
    amrex::ParticleReal kinetic_electron_mass = 0.0_prt;
    if (gate_on_kinetic_density) {
        kinetic_electrons =
            &particles.GetParticleContainerFromName(m_electron_species);
        kinetic_electron_mass = kinetic_electrons->getMass();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            kinetic_electrons->finestLevel() == photons.finestLevel(),
            "Radiation photons and kinetic electrons must use the same AMR levels.");
    }

    amrex::MFItInfo info;
    if (amrex::Gpu::notInLaunchRegion()) {
        info.EnableTiling(WarpXParticleContainer::tile_size);
    }

    constexpr int lev = 0;
    amrex::MultiFab& material_energy =
        *fields.get(FieldType::radiation_material_energy, lev);
    int const num_opacity_species =
        static_cast<int>(m_opacity_species.size());
    auto const opacity_number_densities = GetOpacityNumberDensities(
        particles, m_opacity_species, lev, photons.finestLevel(),
        material_energy);
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
    if (m_use_registered_material_opacity_tables) {
        PreflightRegisteredOpacityCells<
            max_opacity_species, max_opacity_materials>(
            opacity_number_densities, num_opacity_species,
            m_opacity_species_masses,
            m_opacity_species_material_indices,
            m_registered_material_selector);
    }
#endif
    // Ghost values are needed by streaming paths only after every owned cell
    // has satisfied the resolved-single-material contract.
    FillOpacityNumberDensityBoundaries(opacity_number_densities, lev);

    material_energy.setVal(0.0_rt);
    amrex::MultiFab& material_kinetic_energy =
        *fields.get(FieldType::radiation_material_kinetic_energy, lev);
    material_kinetic_energy.setVal(0.0_rt);
    amrex::MultiFab& material_momentum =
        *fields.get(FieldType::radiation_material_momentum, lev);
    material_momentum.setVal(0.0_rt);
    amrex::MultiFab* nonlinear_lte_remap = nullptr;
    if (m_use_nonlinear_hybrid_lte_remap) {
        nonlinear_lte_remap =
            fields.get(FieldType::radiation_hybrid_lte_remap, lev);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            nonlinear_lte_remap->ixType().cellCentered()
                && nonlinear_lte_remap->nComp()
                    == nonlinear_lte_remap_components
                && nonlinear_lte_remap->nGrowVect().allGE(1)
                && nonlinear_lte_remap->boxArray()
                    == material_energy.boxArray()
                && nonlinear_lte_remap->DistributionMap()
                    == material_energy.DistributionMap(),
            "Nonlinear hybrid LTE remap and material-energy fields must "
            "share one cell-centered layout, distribution map, and ghost.");
        nonlinear_lte_remap->setVal(0.0_rt);
    }

    OpacityEvaluator<max_opacity_species> const absorption_evaluator{
        m_use_species_opacity,
        num_opacity_species,
        m_species_absorption_coefficients,
        m_species_absorption_backends,
        {},
        m_species_absorption_table_executors,
        false,
        m_use_absorption_table,
        {},
        m_absorption_table.executor(),
        m_absorption_coefficient};
    OpacityEvaluator<max_opacity_species> const planck_evaluator{
        m_use_species_opacity,
        num_opacity_species,
        m_species_planck_absorption_coefficients,
        m_species_planck_backends,
        m_species_planck_table_executors,
        m_species_spectral_planck_table_executors,
        m_use_planck_table,
        m_use_spectral_planck_table,
        m_planck_table.executor(),
        m_spectral_planck_table.executor(),
        m_planck_absorption_coefficient};
    OpacityEvaluator<max_opacity_species> const rosseland_evaluator{
        m_use_species_opacity,
        num_opacity_species,
        m_species_rosseland_transport_coefficients,
        m_species_rosseland_backends,
        m_species_rosseland_table_executors,
        m_species_spectral_rosseland_table_executors,
        m_use_rosseland_table,
        m_use_spectral_rosseland_table,
        m_rosseland_table.executor(),
        m_spectral_rosseland_table.executor(),
        m_rosseland_transport_coefficient};
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
    MaterialOpacityEvaluator<max_opacity_species> material_opacity_evaluator;
    if (m_use_material_opacity_table
        && !m_use_registered_material_opacity_tables)
    {
        material_opacity_evaluator.enabled = true;
        material_opacity_evaluator.num_species = num_opacity_species;
        material_opacity_evaluator.species_masses = m_opacity_species_masses;
        material_opacity_evaluator.table = m_material_opacity_table.executor();
        material_opacity_evaluator.energy_groups = m_energy_groups;
    }
    RegisteredMaterialOpacityEvaluator<
        max_opacity_species, max_opacity_materials>
        registered_material_opacity_evaluator;
    if (m_use_registered_material_opacity_tables) {
        registered_material_opacity_evaluator.enabled = true;
        registered_material_opacity_evaluator.num_species =
            num_opacity_species;
        registered_material_opacity_evaluator.num_materials =
            m_registered_material_selector.num_materials;
        registered_material_opacity_evaluator.species_masses =
            m_opacity_species_masses;
        registered_material_opacity_evaluator.species_material_indices =
            m_opacity_species_material_indices;
        for (int material = 0;
             material < m_registered_material_selector.num_materials;
             ++material)
        {
            registered_material_opacity_evaluator.tables[material] =
                m_registered_material_opacity_tables[material].executor();
        }
        registered_material_opacity_evaluator.selector =
            m_registered_material_selector;
        registered_material_opacity_evaluator.energy_groups = m_energy_groups;
    }
#endif

    amrex::MultiFab packet_material_momentum;
    amrex::MultiFab diffusion_material_momentum;
    amrex::MultiFab diffusion_group_material_momentum;
    if (m_enable_momentum_coupling) {
        packet_material_momentum.define(
            material_momentum.boxArray(), material_momentum.DistributionMap(),
            3, 1);
        packet_material_momentum.setVal(0.0_rt);
        if (m_enable_diffusion) {
            diffusion_material_momentum.define(
                material_momentum.boxArray(),
                material_momentum.DistributionMap(), 3, 0);
            diffusion_material_momentum.setVal(0.0_rt);
            // Each (cell, group) kernel writes its own three components. A
            // separate deterministic reduction over groups below avoids a
            // concurrent scatter into the group-integrated impulse.
            diffusion_group_material_momentum.define(
                material_momentum.boxArray(),
                material_momentum.DistributionMap(), 3 * m_num_groups, 0);
            diffusion_group_material_momentum.setVal(0.0_rt);
        }
    }

    amrex::MultiFab kinetic_moments;
    if (gate_on_kinetic_density) {
        kinetic_moments.define(material_energy.boxArray(),
                               material_energy.DistributionMap(), 6, 1);
        kinetic_moments.setVal(0.0_rt);
    }

    amrex::MultiFab* rho = nullptr;
    amrex::MultiFab* hybrid_temperature = nullptr;
    int hybrid_num_eos_materials = 0;
    amrex::GpuArray<amrex::MultiFab const*,
                    ElectronThermodynamicsExecutor::max_materials>
        hybrid_material_charge_density_fields{};
    if (gate_on_hybrid_density) {
        rho = fields.get(FieldType::rho_fp, lev);
        hybrid_temperature =
            fields.get(FieldType::hybrid_electron_temperature_fp, lev);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            rho->ixType().nodeCentered()
                && hybrid_temperature->ixType() == rho->ixType()
                && hybrid_temperature->boxArray() == rho->boxArray()
                && hybrid_temperature->DistributionMap()
                    == rho->DistributionMap()
                && rho->nGrowVect().allGE(1)
                && hybrid_temperature->nGrowVect().allGE(1)
                && amrex::convert(rho->boxArray(), material_energy.ixType())
                    == material_energy.boxArray()
                && material_energy.DistributionMap()
                    == rho->DistributionMap(),
            "Hybrid LTE coupling requires rho and Te on one nodal layout, "
            "with matching cell-centered radiation/material fields and one "
            "nodal ghost in every active dimension.");
        rho->FillBoundary(
            rho->nGrowVect(), warpx.Geom(lev).periodicity());
        hybrid_temperature->FillBoundary(
            hybrid_temperature->nGrowVect(),
            warpx.Geom(lev).periodicity());
        hybrid_num_eos_materials =
            m_hybrid_model->electronThermodynamicsNumMaterials();
        for (int material = 0;
             material < hybrid_num_eos_materials; ++material)
        {
            auto* const material_field = fields.get(
                "ni_charge_fp_"
                + m_hybrid_model
                    ->electronThermodynamicsMaterialSpeciesName(material),
                lev);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                material_field->ixType() == rho->ixType()
                    && material_field->boxArray() == rho->boxArray()
                    && material_field->DistributionMap()
                        == rho->DistributionMap()
                    && material_field->nGrowVect().allGE(
                        rho->nGrowVect()),
                "Tabulated electron-EOS material charge densities must share "
                "the hybrid rho/Te nodal layout and ghost extent.");
            material_field->FillBoundary(
                material_field->nGrowVect(),
                warpx.Geom(lev).periodicity());
            hybrid_material_charge_density_fields[material] = material_field;
        }
    }

    amrex::MultiFab* kinetic_temperature = nullptr;
    amrex::MultiFab kinetic_temperature_with_ghost;
    if (gate_on_kinetic_density) {
        kinetic_electrons->DepositTotalNGPTemperature(lev);
        auto const* const deposited_kinetic_temperature =
            fields.get("T_" + m_electron_species, lev);
        kinetic_temperature_with_ghost.define(
            deposited_kinetic_temperature->boxArray(),
            deposited_kinetic_temperature->DistributionMap(), 1, 1);
        kinetic_temperature_with_ghost.setVal(0.0_rt);
        amrex::MultiFab::Copy(
            kinetic_temperature_with_ghost, *deposited_kinetic_temperature,
            0, 0, 1, 0);
        kinetic_temperature_with_ghost.FillBoundary(
            warpx.Geom(lev).periodicity());
        kinetic_temperature = &kinetic_temperature_with_ghost;
    }

    auto const plo = warpx.Geom(lev).ProbLoArray();
    auto const dxi = warpx.Geom(lev).InvCellSizeArray();
    auto const dx = warpx.Geom(lev).CellSizeArray();
    auto const domain_lo = amrex::lbound(warpx.Geom(lev).Domain());
#if !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RZ) \
    && !defined(WARPX_DIM_RSPHERE)
    auto const streaming_domain_hi = amrex::ubound(warpx.Geom(lev).Domain());
    amrex::GpuArray<int, AMREX_SPACEDIM> const domain_lo_index{
        AMREX_D_DECL(domain_lo.x, domain_lo.y, domain_lo.z)};
    amrex::GpuArray<int, AMREX_SPACEDIM> const domain_hi_index{
        AMREX_D_DECL(
            streaming_domain_hi.x, streaming_domain_hi.y,
            streaming_domain_hi.z)};
    amrex::GpuArray<int, AMREX_SPACEDIM> periodic_direction{};
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
        periodic_direction[direction] =
            warpx.Geom(lev).isPeriodic(direction) ? 1 : 0;
    }
#elif defined(WARPX_DIM_RCYLINDER)
    auto const radial_domain_hi = amrex::ubound(warpx.Geom(lev).Domain());
#endif

    if (gate_on_kinetic_density) {
#ifdef AMREX_USE_OMP
        info.SetDynamic(true);
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (WarpXParIter mfi(*kinetic_electrons, lev, info);
             mfi.isValid(); ++mfi)
        {
            auto& tile = kinetic_electrons->ParticlesAt(lev, mfi);
            long const np = tile.numParticles();
            auto const ptd = tile.getParticleTileData();
            auto const* const AMREX_RESTRICT wp = ptd.m_rdata[PIdx::w];
            auto const* const AMREX_RESTRICT uxp = ptd.m_rdata[PIdx::ux];
            auto const* const AMREX_RESTRICT uyp = ptd.m_rdata[PIdx::uy];
            auto const* const AMREX_RESTRICT uzp = ptd.m_rdata[PIdx::uz];
            amrex::ParticleReal const electron_mass = kinetic_electrons->getMass();
            amrex::Array4<amrex::Real> const moment_arr = kinetic_moments.array(mfi);
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ) \
    || defined(WARPX_DIM_RSPHERE)
            auto const get_position = GetParticlePosition<PIdx>(mfi);
#endif

            // amrex::For is required: electrons scatter-add their weights by cell.
            amrex::For(np, [=] AMREX_GPU_DEVICE (long ip) noexcept
            {
                auto const p = WarpXParticleContainer::ParticleType(ptd, ip);
                auto const [i, j, k] = amrex::getParticleCell(p, plo, dxi).dim3();
                auto const physical_weight =
                    static_cast<amrex::Real>(wp[ip]);
                amrex::Gpu::Atomic::AddNoRet(
                    &moment_arr(i, j, k, 0), physical_weight);
                amrex::Gpu::Atomic::AddNoRet(
                    &moment_arr(i, j, k, 1),
                    physical_weight * static_cast<amrex::Real>(
                        Algorithms::KineticEnergy(
                            uxp[ip], uyp[ip], uzp[ip], electron_mass)));
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ) \
    || defined(WARPX_DIM_RSPHERE)
                amrex::ParticleReal radius;
                amrex::ParticleReal particle_theta;
                amrex::ParticleReal particle_phi;
                get_position.AsStored(
                    ip, radius, particle_theta, particle_phi);
                (void)radius;
                auto const local_velocity = KineticCartesianToLocal(
                    static_cast<amrex::Real>(uxp[ip]),
                    static_cast<amrex::Real>(uyp[ip]),
                    static_cast<amrex::Real>(uzp[ip]),
                    static_cast<amrex::Real>(particle_theta),
                    static_cast<amrex::Real>(particle_phi));
#else
                auto const local_velocity = KineticCartesianToLocal(
                    static_cast<amrex::Real>(uxp[ip]),
                    static_cast<amrex::Real>(uyp[ip]),
                    static_cast<amrex::Real>(uzp[ip]),
                    0.0_rt, 0.0_rt);
#endif
                amrex::Gpu::Atomic::AddNoRet(
                    &moment_arr(i, j, k, local_momentum_0_comp),
                    physical_weight * local_velocity.component_0);
                amrex::Gpu::Atomic::AddNoRet(
                    &moment_arr(i, j, k, local_momentum_1_comp),
                    physical_weight * local_velocity.component_1);
                amrex::Gpu::Atomic::AddNoRet(
                    &moment_arr(i, j, k, local_momentum_2_comp),
                    physical_weight * local_velocity.component_2);
                amrex::Gpu::Atomic::AddNoRet(
                    &moment_arr(i, j, k, local_momentum_scale_comp),
                    physical_weight * std::sqrt(
                        static_cast<amrex::Real>(uxp[ip] * uxp[ip]
                            + uyp[ip] * uyp[ip] + uzp[ip] * uzp[ip])));
            });
        }
        kinetic_moments.FillBoundary(warpx.Geom(lev).periodicity());
    }

    amrex::Real min_cell_size = dx[0];
    for (int d = 1; d < AMREX_SPACEDIM; ++d) {
        min_cell_size = amrex::min(min_cell_size, dx[d]);
    }
    amrex::Real const requested_substeps = amrex::max(
        1.0_rt,
        std::ceil(PhysConst::c * dt / (m_path_cell_fraction * min_cell_size)));
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        requested_substeps <= static_cast<amrex::Real>(m_max_transport_substeps),
        "Radiation photon path requires more than "
        "radiation_transport.max_transport_substeps. Increase that limit or "
        "reduce the WarpX timestep.");
    int const transport_substeps = static_cast<int>(requested_substeps);
    amrex::Real const transport_dt = dt / transport_substeps;
    auto const energy_groups = m_energy_groups;
    bool const convert_particles_to_diffusion =
        m_enable_diffusion && m_enable_particle_conversion;
    bool const enable_momentum_coupling = m_enable_momentum_coupling;
    amrex::MultiFab packet_diffusion_energy;
    amrex::MultiFab* diffusion_energy_for_particles = nullptr;
    amrex::MultiFab* persistent_diffusion_energy = nullptr;
    if (convert_particles_to_diffusion) {
        persistent_diffusion_energy =
            fields.get(FieldType::radiation_diffusion_energy, lev);
        packet_diffusion_energy.define(
            persistent_diffusion_energy->boxArray(),
            persistent_diffusion_energy->DistributionMap(), m_num_groups, 1);
        packet_diffusion_energy.setVal(0.0_rt);
        diffusion_energy_for_particles = &packet_diffusion_energy;
    }
    amrex::Real const minimum_diffusion_optical_depth =
        m_minimum_diffusion_optical_depth;
    amrex::Gpu::DeviceScalar<int> invalid_opacity(0);
    int* const invalid_opacity_ptr = invalid_opacity.dataPtr();
    amrex::Gpu::DeviceScalar<int> incomplete_path(0);
    int* const incomplete_path_ptr = incomplete_path.dataPtr();
#if defined(WARPX_DIM_RCYLINDER)
    amrex::Gpu::DeviceScalar<int> invalid_radial_face_input(0);
    int* const invalid_radial_face_input_ptr =
        invalid_radial_face_input.dataPtr();
#endif
    amrex::Gpu::DeviceScalar<amrex::Real> streaming_boundary_energy(0.0_rt);
    amrex::Gpu::DeviceScalar<amrex::Real> streaming_boundary_momentum_0(0.0_rt);
    amrex::Gpu::DeviceScalar<amrex::Real> streaming_boundary_momentum_1(0.0_rt);
    amrex::Gpu::DeviceScalar<amrex::Real> streaming_boundary_momentum_2(0.0_rt);
    amrex::Real* const streaming_boundary_energy_ptr =
        streaming_boundary_energy.dataPtr();
    amrex::GpuArray<amrex::Real*, 3> const streaming_boundary_momentum_ptr{
        streaming_boundary_momentum_0.dataPtr(),
        streaming_boundary_momentum_1.dataPtr(),
        streaming_boundary_momentum_2.dataPtr()};

    // Advance photons in paths no longer than one cell. Cartesian paths are
    // split exactly at every crossed mesh face. In exact RCYLINDER mode, the
    // one-cell bound c*transport_dt <= dx[0] permits a finite local face
    // handoff loop; a packet can therefore write at most one radial guard cell
    // before the existing sums and redistribution transfer the deposit.
#if defined(WARPX_DIM_RCYLINDER)
    bool const exact_rcyl_streaming =
        m_require_cell_interface_exact_streaming;
#endif
    for (int transport_step = 0; transport_step < transport_substeps; ++transport_step)
    {
        amrex::Real const transport_time =
            current_time + transport_step * transport_dt;
#ifdef AMREX_USE_OMP
        info.SetDynamic(true);
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (WarpXParIter mfi(photons, lev, info); mfi.isValid(); ++mfi)
        {
            auto& tile = mfi.GetParticleTile();
            long const np = mfi.numParticles();
            auto const ptd = tile.getParticleTileData();

            auto* const AMREX_RESTRICT idcpu = ptd.m_idcpu;
            auto* const AMREX_RESTRICT wp = ptd.m_rdata[PIdx::w];
            auto const* const AMREX_RESTRICT uxp = ptd.m_rdata[PIdx::ux];
            auto const* const AMREX_RESTRICT uyp = ptd.m_rdata[PIdx::uy];
            auto const* const AMREX_RESTRICT uzp = ptd.m_rdata[PIdx::uz];
            auto const get_position = GetParticlePosition<PIdx>(mfi);
            auto set_position = SetParticlePosition<PIdx>(mfi);

            amrex::Array4<amrex::Real> const energy_arr = material_energy.array(mfi);
            amrex::Array4<amrex::Real> momentum_arr;
            if (enable_momentum_coupling) {
                momentum_arr = packet_material_momentum.array(mfi);
            }
            amrex::Array4<amrex::Real> diffusion_arr;
            if (diffusion_energy_for_particles != nullptr) {
                diffusion_arr = diffusion_energy_for_particles->array(mfi);
            }
            amrex::Array4<amrex::Real const> rho_arr;
            if (rho != nullptr) { rho_arr = rho->const_array(mfi); }
            amrex::Array4<amrex::Real const> hybrid_temperature_arr;
            if (hybrid_temperature != nullptr) {
                hybrid_temperature_arr = hybrid_temperature->const_array(mfi);
            }
            amrex::Array4<amrex::Real const> electron_weight_arr;
            if (gate_on_kinetic_density) {
                electron_weight_arr = kinetic_moments.const_array(mfi);
            }
            amrex::Array4<amrex::Real const> kinetic_temperature_arr;
            if (kinetic_temperature != nullptr) {
                kinetic_temperature_arr = kinetic_temperature->const_array(mfi);
            }
            amrex::GpuArray<amrex::Array4<amrex::Real const>,
                            max_opacity_species> opacity_number_density_arr{};
            for (int species = 0; species < num_opacity_species; ++species) {
                opacity_number_density_arr[species] =
                    opacity_number_densities[species]->const_array(mfi);
            }

            // amrex::For is required: different photons can scatter-add to one cell.
            amrex::For(np, [=] AMREX_GPU_DEVICE (long ip) noexcept
            {
                auto const p = WarpXParticleContainer::ParticleType(ptd, ip);
                auto const initial_cell = amrex::getParticleCell(p, plo, dxi);
                auto const [initial_i, initial_j, initial_k] = initial_cell.dim3();
#if defined(WARPX_DIM_RCYLINDER)
                int i = initial_i;
                int const j = initial_j;
                int const k = initial_k;
#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RSPHERE)
                int const i = initial_i;
                int const j = initial_j;
                int const k = initial_k;
#else
                int i = initial_i;
                int j = initial_j;
                int k = initial_k;
#endif
                amrex::ParticleReal x;
                amrex::ParticleReal y;
                amrex::ParticleReal z;
                get_position(ip, x, y, z);
                amrex::ParticleReal const photon_energy =
                    Algorithms::KineticEnergyPhotons(uxp[ip], uyp[ip], uzp[ip]);

                amrex::Real remaining_transport_dt = transport_dt;
#if defined(WARPX_DIM_RCYLINDER)
                amrex::ParticleReal nx = 0.0_prt;
                amrex::ParticleReal ny = 0.0_prt;
                amrex::ParticleReal nz = 0.0_prt;
                if (exact_rcyl_streaming) {
                    amrex::ParticleReal const momentum_norm = std::sqrt(
                        uxp[ip] * uxp[ip] + uyp[ip] * uyp[ip]
                        + uzp[ip] * uzp[ip]);
                    if (!(momentum_norm > 0.0_prt)
                        || !amrex::Math::isfinite(momentum_norm)) {
                        amrex::HostDevice::Atomic::Add(
                            invalid_radial_face_input_ptr, 1);
                        return;
                    }
                    nx = uxp[ip] / momentum_norm;
                    ny = uyp[ip] / momentum_norm;
                    nz = uzp[ip] / momentum_norm;
                    if (!amrex::Math::isfinite(nx)
                        || !amrex::Math::isfinite(ny)
                        || !amrex::Math::isfinite(nz)) {
                        amrex::HostDevice::Atomic::Add(
                            invalid_radial_face_input_ptr, 1);
                        return;
                    }
                }
#endif
#if !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RZ) \
    && !defined(WARPX_DIM_RSPHERE)
                amrex::GpuArray<amrex::ParticleReal, AMREX_SPACEDIM>
                    grid_position{};
                amrex::GpuArray<amrex::ParticleReal, AMREX_SPACEDIM>
                    grid_velocity{};
                amrex::ParticleReal const momentum_norm = std::sqrt(
                    uxp[ip] * uxp[ip] + uyp[ip] * uyp[ip]
                    + uzp[ip] * uzp[ip]);
                amrex::ParticleReal const velocity_scale =
                    momentum_norm > 0.0_prt
                    ? PhysConst::c / momentum_norm
                    : 0.0_prt;
#if defined(WARPX_DIM_1D_Z)
                grid_position[0] = z;
                grid_velocity[0] = uzp[ip] * velocity_scale;
#elif defined(WARPX_DIM_XZ)
                grid_position[0] = x;
                grid_position[1] = z;
                grid_velocity[0] = uxp[ip] * velocity_scale;
                grid_velocity[1] = uzp[ip] * velocity_scale;
#else
                grid_position[0] = x;
                grid_position[1] = y;
                grid_position[2] = z;
                grid_velocity[0] = uxp[ip] * velocity_scale;
                grid_velocity[1] = uyp[ip] * velocity_scale;
                grid_velocity[2] = uzp[ip] * velocity_scale;
#endif
#endif

                // c*transport_dt <= min(dx), so no Cartesian component can
                // cross more than one full cell. The larger fixed bound also
                // covers a zero-length crossing when a negative-going packet
                // starts exactly on a face and simultaneous corner crossings.
#if defined(WARPX_DIM_RCYLINDER)
                constexpr int max_exact_radial_segments = 4;
                int const max_segments = exact_rcyl_streaming
                    ? max_exact_radial_segments
                    : 2 * AMREX_SPACEDIM + 2;
#else
                constexpr int max_segments = 2 * AMREX_SPACEDIM + 2;
#endif
                int segment = 0;
                while (remaining_transport_dt > 0.0_rt
                       && segment < max_segments)
                {
                    ++segment;
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ) \
    || defined(WARPX_DIM_RSPHERE)
#if defined(WARPX_DIM_RCYLINDER)
                    amrex::Real segment_dt = remaining_transport_dt;
#else
                    amrex::Real const segment_dt = remaining_transport_dt;
#endif
#if defined(WARPX_DIM_RCYLINDER)
                    bool radial_face_crossed = false;
                    bool radial_face_outside = false;
                    int radial_next_cell = i;
                    if (exact_rcyl_streaming) {
                        auto const radial_face =
                            warpx::radiation::FindNextRadialFace(
                                x, y, nx, ny, nz, i, domain_lo.x,
                                radial_domain_hi.x, plo[0], dx[0],
                                static_cast<amrex::ParticleReal>(PhysConst::c)
                                    * static_cast<amrex::ParticleReal>(
                                        remaining_transport_dt));
                        if (!radial_face.valid_input) {
                            amrex::HostDevice::Atomic::Add(
                                invalid_radial_face_input_ptr, 1);
                            return;
                        }
                        radial_face_crossed = radial_face.crossed;
                        radial_face_outside = radial_face.outside_domain;
                        radial_next_cell = radial_face.next_radial_cell;
                        if (radial_face_crossed) {
                            amrex::ParticleReal const segment_distance =
                                radial_face.distance;
                            auto const speed =
                                static_cast<amrex::ParticleReal>(PhysConst::c);
                            if (!amrex::Math::isfinite(segment_distance)
                                || segment_distance < 0.0_prt
                                || !amrex::Math::isfinite(speed)
                                || !(speed > 0.0_prt)) {
                                amrex::HostDevice::Atomic::Add(
                                    invalid_radial_face_input_ptr, 1);
                                return;
                            }
                            segment_dt = static_cast<amrex::Real>(
                                segment_distance / speed);
                            if (!amrex::Math::isfinite(segment_dt)
                                || segment_dt < 0.0_rt) {
                                amrex::HostDevice::Atomic::Add(
                                    invalid_radial_face_input_ptr, 1);
                                return;
                            }
                            // The primitive enforces distance <= c*dt.  The
                            // final conversion can round upward when the
                            // direction is stored in ParticleReal, so keep
                            // the time segment within the caller's budget.
                            segment_dt = amrex::min(
                                segment_dt, remaining_transport_dt);
                        }
                    }
#endif
#else
                    amrex::Real segment_dt = remaining_transport_dt;
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
                        face_crossing_dt{};
                    int const cell_index[3] = {i, j, k};
                    for (int direction = 0; direction < AMREX_SPACEDIM;
                         ++direction)
                    {
                        face_crossing_dt[direction] =
                            std::numeric_limits<amrex::Real>::max();
                        amrex::ParticleReal const velocity =
                            grid_velocity[direction];
                        if (velocity == 0.0_prt) { continue; }
                        amrex::Real const cell_lo = plo[direction]
                            + (cell_index[direction]
                               - domain_lo_index[direction])
                                * dx[direction];
                        amrex::Real const face = velocity > 0.0_prt
                            ? cell_lo + dx[direction]
                            : cell_lo;
                        face_crossing_dt[direction] = amrex::max(
                            0.0_rt,
                            static_cast<amrex::Real>(
                                (face - grid_position[direction]) / velocity));
                        segment_dt = amrex::min(
                            segment_dt, face_crossing_dt[direction]);
                    }
#endif

                    amrex::ParticleReal sample_x = x;
                    amrex::ParticleReal sample_y = y;
                    amrex::ParticleReal sample_z = z;
                    UpdatePosition(
                        sample_x, sample_y, sample_z, uxp[ip], uyp[ip], uzp[ip],
                        0.5_rt * segment_dt, 0.0_prt);
                    amrex::Real const sample_time = transport_time
                        + (transport_dt - remaining_transport_dt)
                        + 0.5_rt * segment_dt;
                    bool absorb_here = true;
                    amrex::Real electron_density = 0.0_rt;
                    amrex::Real electron_temperature = 0.0_rt;

                    if (gate_on_hybrid_density) {
#if AMREX_SPACEDIM == 1
                        amrex::Real const rho_cell =
                            0.5_rt * (rho_arr(i, j, k) + rho_arr(i + 1, j, k));
                        amrex::Real const temperature_cell = 0.5_rt * (
                            hybrid_temperature_arr(i, j, k)
                            + hybrid_temperature_arr(i + 1, j, k));
#elif AMREX_SPACEDIM == 2
                        amrex::Real const rho_cell = 0.25_rt * (
                            rho_arr(i, j, k) + rho_arr(i + 1, j, k)
                            + rho_arr(i, j + 1, k) + rho_arr(i + 1, j + 1, k));
                        amrex::Real const temperature_cell = 0.25_rt * (
                            hybrid_temperature_arr(i, j, k)
                            + hybrid_temperature_arr(i + 1, j, k)
                            + hybrid_temperature_arr(i, j + 1, k)
                            + hybrid_temperature_arr(i + 1, j + 1, k));
#else
                        amrex::Real const rho_cell = 0.125_rt * (
                            rho_arr(i, j, k) + rho_arr(i + 1, j, k)
                            + rho_arr(i, j + 1, k) + rho_arr(i + 1, j + 1, k)
                            + rho_arr(i, j, k + 1) + rho_arr(i + 1, j, k + 1)
                            + rho_arr(i, j + 1, k + 1)
                            + rho_arr(i + 1, j + 1, k + 1));
                        amrex::Real const temperature_cell = 0.125_rt * (
                            hybrid_temperature_arr(i, j, k)
                            + hybrid_temperature_arr(i + 1, j, k)
                            + hybrid_temperature_arr(i, j + 1, k)
                            + hybrid_temperature_arr(i + 1, j + 1, k)
                            + hybrid_temperature_arr(i, j, k + 1)
                            + hybrid_temperature_arr(i + 1, j, k + 1)
                            + hybrid_temperature_arr(i, j + 1, k + 1)
                            + hybrid_temperature_arr(i + 1, j + 1, k + 1));
#endif
                        electron_density = rho_cell / PhysConst::q_e;
                        electron_temperature = temperature_cell;
                        absorb_here = electron_density > density_floor;
                    }

                    if (absorb_here && gate_on_kinetic_density) {
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
                        amrex::Real const r_lo =
                            plo[0] + (i - domain_lo.x) * dx[0];
                        amrex::Real const r_hi = r_lo + dx[0];
#if defined(WARPX_DIM_RCYLINDER)
                        amrex::Real const cell_volume =
                            MathConst::pi * (r_hi * r_hi - r_lo * r_lo);
#else
                        amrex::Real const cell_volume = MathConst::pi
                            * (r_hi * r_hi - r_lo * r_lo) * dx[1];
#endif
#elif defined(WARPX_DIM_RSPHERE)
                        amrex::Real const r_lo =
                            plo[0] + (i - domain_lo.x) * dx[0];
                        amrex::Real const r_hi = r_lo + dx[0];
                        amrex::Real const cell_volume =
                            4.0_rt / 3.0_rt * MathConst::pi
                            * (r_hi * r_hi * r_hi - r_lo * r_lo * r_lo);
#else
                        amrex::Real const cell_volume =
                            AMREX_D_TERM(dx[0], * dx[1], * dx[2]);
#endif
                        electron_density =
                            electron_weight_arr(i, j, k, 0) / cell_volume;
                        electron_temperature = kinetic_temperature_arr(i, j, k)
                            * PhysConst::q_e / PhysConst::kb;
                        absorb_here = electron_density > density_floor;
                    }

                    amrex::GpuArray<amrex::Real, max_opacity_species>
                        opacity_number_density{};
                    for (int species = 0; species < num_opacity_species; ++species) {
                        opacity_number_density[species] =
                            opacity_number_density_arr[species](i, j, k);
                    }

#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
                    warpx::radiation::MaterialOpacityCoefficients
                        material_opacity_coefficients;
                    bool const use_native_material_opacity =
                        material_opacity_evaluator.enabled
                        || registered_material_opacity_evaluator.enabled;
                    if (absorb_here && use_native_material_opacity) {
                        material_opacity_coefficients =
                            registered_material_opacity_evaluator.enabled
                            ? registered_material_opacity_evaluator(
                                opacity_number_density, photon_energy,
                                electron_temperature)
                            : material_opacity_evaluator(
                                opacity_number_density, photon_energy,
                                electron_temperature);
                    }
#endif
                    if (absorb_here && convert_particles_to_diffusion) {
                        amrex::Real rosseland_opacity_value;
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
                        if (use_native_material_opacity) {
                            rosseland_opacity_value =
                                material_opacity_coefficients.rosseland_transport;
                        } else
#endif
                        {
                            rosseland_opacity_value = rosseland_evaluator(
                                opacity_number_density, sample_x, sample_y,
                                sample_z, sample_time, photon_energy,
                                electron_density, electron_temperature);
                        }
                        if (rosseland_opacity_value < 0.0_rt
                            || !amrex::Math::isfinite(rosseland_opacity_value))
                        {
                            amrex::HostDevice::Atomic::Add(
                                invalid_opacity_ptr, 1);
                        } else if (rosseland_opacity_value * min_cell_size
                                   >= minimum_diffusion_optical_depth)
                        {
                            // Once a streaming packet enters an optically thick
                            // cell, hand its complete radiation energy to the
                            // diffusion representation. LTE exchange below can
                            // then couple that energy to matter in the same step.
                            int const group = energy_groups.index(photon_energy);
                            amrex::Gpu::Atomic::AddNoRet(
                                &diffusion_arr(i, j, k, group),
                                static_cast<amrex::Real>(wp[ip] * photon_energy));
                            wp[ip] = 0.0_prt;
                            idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                            return;
                        }
                    }

                    if (absorb_here) {
                        amrex::Real alpha;
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
                        if (use_native_material_opacity) {
                            // Packet attenuation uses the group's Planck true
                            // absorption. Scattering neither heats nor deposits
                            // an event momentum in this first table backend.
                            alpha = material_opacity_coefficients.planck_absorption;
                        } else
#endif
                        {
                            alpha = absorption_evaluator(
                                opacity_number_density, sample_x, sample_y,
                                sample_z, sample_time, photon_energy,
                                electron_density, electron_temperature);
                        }
                        if (alpha < 0.0_rt || !amrex::Math::isfinite(alpha)) {
                            amrex::HostDevice::Atomic::Add(
                                invalid_opacity_ptr, 1);
                            alpha = 0.0_rt;
                        }
                        auto const absorbed_fraction =
                            static_cast<amrex::ParticleReal>(-std::expm1(
                                -alpha * PhysConst::c * segment_dt));
                        amrex::ParticleReal const old_weight = wp[ip];
                        amrex::ParticleReal const removed_weight =
                            old_weight * absorbed_fraction;
                        wp[ip] = old_weight - removed_weight;

                        auto const removed_energy =
                            static_cast<amrex::Real>(
                                removed_weight * photon_energy);
#if defined(WARPX_DIM_RCYLINDER)
                        if (exact_rcyl_streaming) {
                            // Face handoff can scatter from one CPU particle
                            // tile into a cell concurrently visited by another
                            // tile.  This must be atomic on both host and device.
                            amrex::HostDevice::Atomic::Add(
                                &energy_arr(i, j, k), removed_energy);
                        } else {
                            amrex::Gpu::Atomic::AddNoRet(
                                &energy_arr(i, j, k), removed_energy);
                        }
#else
                        amrex::Gpu::Atomic::AddNoRet(
                            &energy_arr(i, j, k), removed_energy);
#endif
                        if (enable_momentum_coupling && removed_weight > 0.0_prt) {
                            amrex::ParticleReal const momentum_scale =
                                removed_weight * PhysConst::m_e;
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
                            amrex::ParticleReal radius;
                            amrex::ParticleReal position_theta;
                            amrex::ParticleReal position_z;
                            get_position.AsStored(
                                ip, radius, position_theta, position_z);
                            (void)radius;
                            (void)position_z;
                            amrex::ParticleReal const cos_theta =
                                std::cos(position_theta);
                            amrex::ParticleReal const sin_theta =
                                std::sin(position_theta);
                            amrex::Gpu::Atomic::AddNoRet(
                                &momentum_arr(i, j, k, 0),
                                static_cast<amrex::Real>(
                                    momentum_scale * (uxp[ip] * cos_theta
                                                      + uyp[ip] * sin_theta)));
                            amrex::Gpu::Atomic::AddNoRet(
                                &momentum_arr(i, j, k, 1),
                                static_cast<amrex::Real>(
                                    momentum_scale * (-uxp[ip] * sin_theta
                                                      + uyp[ip] * cos_theta)));
                            amrex::Gpu::Atomic::AddNoRet(
                                &momentum_arr(i, j, k, 2),
                                static_cast<amrex::Real>(
                                    momentum_scale * uzp[ip]));
#elif defined(WARPX_DIM_RSPHERE)
                            amrex::ParticleReal radius;
                            amrex::ParticleReal position_theta;
                            amrex::ParticleReal position_phi;
                            get_position.AsStored(
                                ip, radius, position_theta, position_phi);
                            (void)radius;
                            amrex::ParticleReal const cos_theta =
                                std::cos(position_theta);
                            amrex::ParticleReal const sin_theta =
                                std::sin(position_theta);
                            amrex::ParticleReal const cos_phi =
                                std::cos(position_phi);
                            amrex::ParticleReal const sin_phi =
                                std::sin(position_phi);
                            amrex::Gpu::Atomic::AddNoRet(
                                &momentum_arr(i, j, k, 0),
                                static_cast<amrex::Real>(
                                    momentum_scale
                                    * (uxp[ip] * cos_theta * cos_phi
                                       + uyp[ip] * sin_theta * cos_phi
                                       + uzp[ip] * sin_phi)));
                            amrex::Gpu::Atomic::AddNoRet(
                                &momentum_arr(i, j, k, 1),
                                static_cast<amrex::Real>(
                                    momentum_scale
                                    * (-uxp[ip] * sin_theta
                                       + uyp[ip] * cos_theta)));
                            amrex::Gpu::Atomic::AddNoRet(
                                &momentum_arr(i, j, k, 2),
                                static_cast<amrex::Real>(
                                    momentum_scale
                                    * (-uxp[ip] * cos_theta * sin_phi
                                       - uyp[ip] * sin_theta * sin_phi
                                       + uzp[ip] * cos_phi)));
#else
                            amrex::Gpu::Atomic::AddNoRet(
                                &momentum_arr(i, j, k, 0),
                                static_cast<amrex::Real>(
                                    momentum_scale * uxp[ip]));
                            amrex::Gpu::Atomic::AddNoRet(
                                &momentum_arr(i, j, k, 1),
                                static_cast<amrex::Real>(
                                    momentum_scale * uyp[ip]));
                            amrex::Gpu::Atomic::AddNoRet(
                                &momentum_arr(i, j, k, 2),
                                static_cast<amrex::Real>(
                                    momentum_scale * uzp[ip]));
#endif
                        }
                    }

                    UpdatePosition(
                        x, y, z, uxp[ip], uyp[ip], uzp[ip], segment_dt, 0.0_prt);
                    remaining_transport_dt = amrex::max(
                        0.0_rt, remaining_transport_dt - segment_dt);

#if !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RZ) \
    && !defined(WARPX_DIM_RSPHERE)
                    for (int direction = 0; direction < AMREX_SPACEDIM;
                         ++direction)
                    {
                        grid_position[direction] +=
                            grid_velocity[direction] * segment_dt;
                    }
                    if (remaining_transport_dt > 0.0_rt) {
                        amrex::Real const crossing_tolerance =
                            32.0_rt * std::numeric_limits<amrex::Real>::epsilon()
                            * amrex::max(transport_dt, segment_dt);
                        int* const mutable_cell_index[3] = {&i, &j, &k};
                        bool outside_nonperiodic_domain = false;
                        for (int direction = 0; direction < AMREX_SPACEDIM;
                             ++direction)
                        {
                            if (std::abs(
                                    face_crossing_dt[direction] - segment_dt)
                                > crossing_tolerance)
                            {
                                continue;
                            }
                            int const side =
                                grid_velocity[direction] > 0.0_prt ? 1 : -1;
                            *mutable_cell_index[direction] += side;
                            if (*mutable_cell_index[direction]
                                    >= domain_lo_index[direction]
                                && *mutable_cell_index[direction]
                                    <= domain_hi_index[direction])
                            {
                                continue;
                            }
                            // A periodic neighbor is represented by this FAB's
                            // single exterior guard cell. FillBoundary supplied
                            // its material state, SumBoundary transfers its
                            // deposits, and the particle boundary operation wraps
                            // the final position before redistribution.
                            if (periodic_direction[direction] == 0) {
                                outside_nonperiodic_domain = true;
                            }
                        }
                        if (outside_nonperiodic_domain) {
                            UpdatePosition(
                                x, y, z, uxp[ip], uyp[ip], uzp[ip],
                                remaining_transport_dt, 0.0_prt);
                            remaining_transport_dt = 0.0_rt;
                        }
                    }
#else
#if defined(WARPX_DIM_RCYLINDER)
                    if (exact_rcyl_streaming) {
                        if (radial_face_crossed) {
                            if (radial_face_outside) {
                                // The open outer radial boundary owns the
                                // positive-weight packet at the face, including
                                // an exact substep endpoint.  Mark it here
                                // because the generic boundary operation tests
                                // strictly outside ProbHi.  The existing loss
                                // accumulator below observes this invalid packet
                                // before redistribution.
                                idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                                remaining_transport_dt = 0.0_rt;
                            } else {
                                // Preserve the returned handoff, including a
                                // zero-length departure from an exact face.
                                i = radial_next_cell;
                            }
                        } else {
                            remaining_transport_dt = 0.0_rt;
                        }
                    } else {
                        remaining_transport_dt = 0.0_rt;
                    }
#else
                    remaining_transport_dt = 0.0_rt;
#endif
#endif
                }
                if (remaining_transport_dt > 0.0_rt) {
                    amrex::HostDevice::Atomic::Add(incomplete_path_ptr, 1);
                    UpdatePosition(
                        x, y, z, uxp[ip], uyp[ip], uzp[ip],
                        remaining_transport_dt, 0.0_prt);
                }
                set_position(ip, x, y, z);
            });
        }

        photons.ApplyBoundaryConditions();
        if (m_track_energy_balance) {
            AccumulateStreamingBoundaryLoss(
                photons, lev, streaming_boundary_energy_ptr,
                streaming_boundary_momentum_ptr);
        }
        photons.Redistribute();
    }

    auto const one_guard = amrex::IntVect(1);
    auto const zero_guard = amrex::IntVect::TheZeroVector();
    auto const& periodicity = warpx.Geom(lev).periodicity();
    ablastr::utils::communication::SumBoundary(
        material_energy, 0, 1, one_guard, zero_guard,
        WarpX::do_single_precision_comms, periodicity);
    if (m_enable_momentum_coupling) {
        ablastr::utils::communication::SumBoundary(
            packet_material_momentum, 0, 3, one_guard, zero_guard,
            WarpX::do_single_precision_comms, periodicity);
    }
    if (convert_particles_to_diffusion) {
        ablastr::utils::communication::SumBoundary(
            packet_diffusion_energy, 0, m_num_groups, one_guard, zero_guard,
            WarpX::do_single_precision_comms, periodicity);
        amrex::MultiFab::Add(
            *persistent_diffusion_energy, packet_diffusion_energy,
            0, 0, m_num_groups, 0);
    }

    if (m_track_energy_balance) {
        m_last_streaming_boundary_energy_loss =
            streaming_boundary_energy.dataValue();
        amrex::ParallelDescriptor::ReduceRealSum(
            m_last_streaming_boundary_energy_loss);
        m_last_streaming_boundary_momentum_loss = {
            streaming_boundary_momentum_0.dataValue(),
            streaming_boundary_momentum_1.dataValue(),
            streaming_boundary_momentum_2.dataValue()};
        amrex::ParallelDescriptor::ReduceRealSum(
            m_last_streaming_boundary_momentum_loss.data(), 3);
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        invalid_opacity.dataValue() == 0,
        "A streaming absorption or Rosseland transport coefficient evaluated "
        "to a negative or non-finite value. Radiation opacity expressions and "
        "tables must be non-negative and finite for every transported photon state.");
#if defined(WARPX_DIM_RCYLINDER)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        invalid_radial_face_input.dataValue() == 0,
        "Exact RCYLINDER radiation face streaming received invalid particle "
        "direction or face-march data.");
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        incomplete_path.dataValue() == 0,
        "The radiation face marcher exhausted its bounded segment count. This "
        "violates the one-cell transport-substep invariant.");

    if (m_enable_momentum_coupling) {
        m_last_numerical_energy_residual += ApplyRadiationMomentumWork(
            particles, m_momentum_species, packet_material_momentum,
            material_momentum, material_kinetic_energy, material_energy,
            "Streaming radiation recoil requires more bulk kinetic work than "
            "the absorbed packet energy supplies. Reduce the timestep or inspect "
            "the configured material momentum species.");
    }

    if (m_enable_lte_exchange) {
        bool const enable_diffusion = m_enable_diffusion;
        amrex::MultiFab& diffusion_energy =
            *fields.get(FieldType::radiation_diffusion_energy, lev);
        amrex::MultiFab rosseland_opacity;
        if (enable_diffusion) {
            rosseland_opacity.define(
                diffusion_energy.boxArray(), diffusion_energy.DistributionMap(),
                m_num_groups, 1);
            rosseland_opacity.setVal(0.0_rt);
        }
        int const num_groups = m_num_groups;
        bool const implicit_lte_temperature =
            m_lte_exchange_solver == LteExchangeSolver::ImplicitTemperature;
        bool const nonlinear_hybrid_lte =
            m_use_nonlinear_hybrid_lte_remap;
        int const lte_exchange_max_iterations = m_lte_exchange_max_iterations;
        amrex::Real const lte_exchange_tolerance = amrex::max(
            m_lte_exchange_tolerance,
            10.0_rt * std::numeric_limits<amrex::Real>::epsilon());
        constexpr amrex::Real radiation_constant = 7.565733250280007e-16_rt;
        // Each cell owns its validation flags. This avoids shared writes from
        // OpenMP/GPU kernels; the iMultiFab maxima below perform the reduction.
        amrex::iMultiFab lte_cell_status(
            diffusion_energy.boxArray(), diffusion_energy.DistributionMap(),
            2, 0);
        lte_cell_status.setVal(0);
        ImplicitLteCellContext<max_opacity_species> const implicit_lte_context{
            enable_diffusion,
            num_groups,
            lte_exchange_max_iterations,
            lte_exchange_tolerance,
            current_time,
            dt,
            planck_evaluator,
            rosseland_evaluator,
            energy_groups
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
            , material_opacity_evaluator
            , registered_material_opacity_evaluator
#endif
        };
        auto const domain_hi = amrex::ubound(warpx.Geom(lev).Domain());
        amrex::GpuArray<int, 3> periodic{0, 0, 0};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            periodic[d] = warpx.Geom(lev).isPeriodic(d) ? 1 : 0;
        }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(diffusion_energy, amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi)
        {
            amrex::Box const& box = mfi.tilebox();
            amrex::Array4<amrex::Real> const radiation_arr =
                diffusion_energy.array(mfi);
            amrex::Array4<amrex::Real> const material_arr =
                material_energy.array(mfi);
            amrex::Array4<amrex::Real> nonlinear_remap_arr;
            if (nonlinear_hybrid_lte) {
                nonlinear_remap_arr = nonlinear_lte_remap->array(mfi);
            }
            amrex::Array4<int> const lte_status_arr =
                lte_cell_status.array(mfi);
            amrex::Array4<amrex::Real> rosseland_arr;
            if (enable_diffusion) {
                rosseland_arr = rosseland_opacity.array(mfi);
            }
            amrex::Array4<amrex::Real const> rho_arr;
            amrex::Array4<amrex::Real const> hybrid_temperature_arr;
            if (gate_on_hybrid_density) {
                rho_arr = rho->const_array(mfi);
                hybrid_temperature_arr = hybrid_temperature->const_array(mfi);
            }
            ElectronThermodynamicsExecutor::MaterialChargeDensityArrays
                hybrid_material_charge_density{};
            for (int material = 0;
                 material < hybrid_num_eos_materials; ++material)
            {
                hybrid_material_charge_density[material] =
                    hybrid_material_charge_density_fields[material]
                        ->const_array(mfi);
            }
            amrex::Array4<amrex::Real const> kinetic_moment_arr;
            amrex::Array4<amrex::Real const> kinetic_temperature_arr;
            if (gate_on_kinetic_density) {
                kinetic_moment_arr = kinetic_moments.const_array(mfi);
                kinetic_temperature_arr = kinetic_temperature->const_array(mfi);
            }
            amrex::GpuArray<amrex::Array4<amrex::Real const>,
                            max_opacity_species> opacity_number_density_arr{};
            for (int species = 0; species < num_opacity_species; ++species) {
                opacity_number_density_arr[species] =
                    opacity_number_densities[species]->const_array(mfi);
            }

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (
                int i, int j, int k) noexcept
            {
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
                amrex::Real const r_lo =
                    plo[0] + (i - domain_lo.x) * dx[0];
                amrex::Real const r_hi = r_lo + dx[0];
#if defined(WARPX_DIM_RCYLINDER)
                amrex::Real const cell_volume =
                    MathConst::pi * (r_hi * r_hi - r_lo * r_lo);
#else
                amrex::Real const cell_volume = MathConst::pi
                    * (r_hi * r_hi - r_lo * r_lo) * dx[1];
#endif
#elif defined(WARPX_DIM_RSPHERE)
                amrex::Real const r_lo =
                    plo[0] + (i - domain_lo.x) * dx[0];
                amrex::Real const r_hi = r_lo + dx[0];
                amrex::Real const cell_volume =
                    4.0_rt / 3.0_rt * MathConst::pi
                    * (r_hi * r_hi * r_hi - r_lo * r_lo * r_lo);
#else
                amrex::Real const cell_volume =
                    AMREX_D_TERM(dx[0], * dx[1], * dx[2]);
#endif

                HybridCellMaterialState material_state;
                HybridNonlinearCellState nonlinear_material_state;
                if (gate_on_hybrid_density) {
                    if (nonlinear_hybrid_lte) {
                        nonlinear_material_state =
                            GatherHybridNonlinearCellState(
                                i, j, k, density_floor,
                                hybrid_thermodynamic_density_floor,
                                hybrid_thermodynamics, rho_arr,
                                hybrid_temperature_arr,
                                hybrid_material_charge_density,
                                plo, dx, domain_lo);
                        material_state = nonlinear_material_state.material;
                    } else {
                        material_state = GatherHybridCellMaterialState(
                            i, j, k, density_floor,
                            hybrid_thermodynamic_density_floor,
                            hybrid_thermodynamics, rho_arr,
                            hybrid_temperature_arr,
                            hybrid_material_charge_density,
                            plo, dx, domain_lo);
                    }
                    if (!material_state.valid) {
                        lte_status_arr(i, j, k, 0) = 1;
                        return;
                    }
                } else {
                    amrex::Real const electron_weight =
                        kinetic_moment_arr(i, j, k, 0);
                    material_state.electron_density = electron_weight / cell_volume;
                    material_state.electron_temperature =
                        kinetic_temperature_arr(i, j, k)
                        * PhysConst::q_e / PhysConst::kb;
                    amrex::Real cold_energy = 0.0_rt;
                    if (electron_weight > 0.0_rt) {
                        amrex::Real const bulk_x =
                            kinetic_moment_arr(i, j, k, 2) / electron_weight;
                        amrex::Real const bulk_y =
                            kinetic_moment_arr(i, j, k, 3) / electron_weight;
                        amrex::Real const bulk_z =
                            kinetic_moment_arr(i, j, k, 4) / electron_weight;
                        cold_energy = electron_weight
                            * static_cast<amrex::Real>(Algorithms::KineticEnergy(
                                bulk_x, bulk_y, bulk_z,
                                kinetic_electron_mass));
                    }
                    // Only the energy above the minimum compatible with the
                    // deposited cell momentum is available to LTE emission.
                    // Treating total kinetic energy as thermal would let
                    // radiation remove the bulk drift.
                    material_state.available_energy = amrex::max(
                        0.0_rt,
                        kinetic_moment_arr(i, j, k, 1) - cold_energy);
                    material_state.internal_energy =
                        material_state.available_energy;
                    if (material_state.electron_temperature > 0.0_rt &&
                        material_state.internal_energy > 0.0_rt) {
                        material_state.heat_capacity =
                            material_state.internal_energy /
                            material_state.electron_temperature;
                    } else {
                        material_state.heat_capacity =
                            1.5_rt * material_state.electron_density *
                            PhysConst::kb * cell_volume;
                    }
                }

                amrex::Real const electron_density =
                    material_state.electron_density;
                amrex::Real const electron_temperature =
                    material_state.electron_temperature;
                amrex::Real available_material_energy =
                    material_state.available_energy;
                amrex::Real const pending_material_energy =
                    material_arr(i, j, k);

                // Streaming absorption was accumulated in this same ledger
                // before LTE exchange. It is immediately available to fund
                // emission in the operator-split material state; adding only
                // its positive part cannot weaken the non-negative-energy
                // bound on the final, net material source.
                available_material_energy +=
                    amrex::max(0.0_rt, pending_material_energy);

                if (electron_density <= density_floor) { return; }

                amrex::GpuArray<amrex::Real, max_opacity_species>
                    opacity_number_density{};
                for (int species = 0; species < num_opacity_species; ++species) {
                    opacity_number_density[species] =
                        opacity_number_density_arr[species](i, j, k);
                }

#if defined(WARPX_DIM_3D)
                amrex::Real const x =
                    plo[0] + (i - domain_lo.x + 0.5_rt) * dx[0];
                amrex::Real const y =
                    plo[1] + (j - domain_lo.y + 0.5_rt) * dx[1];
                amrex::Real const z =
                    plo[2] + (k - domain_lo.z + 0.5_rt) * dx[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                amrex::Real const x =
                    plo[0] + (i - domain_lo.x + 0.5_rt) * dx[0];
                amrex::Real const y = 0.0_rt;
                amrex::Real const z =
                    plo[1] + (j - domain_lo.y + 0.5_rt) * dx[1];
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                amrex::Real const x =
                    plo[0] + (i - domain_lo.x + 0.5_rt) * dx[0];
                amrex::Real const y = 0.0_rt;
                amrex::Real const z = 0.0_rt;
#else
                amrex::Real const x = 0.0_rt;
                amrex::Real const y = 0.0_rt;
                amrex::Real const z =
                    plo[0] + (i - domain_lo.x + 0.5_rt) * dx[0];
#endif
                if (electron_temperature < 0.0_rt
                    || !amrex::Math::isfinite(electron_temperature)) {
                    lte_status_arr(i, j, k, 0) = 1;
                    return;
                }

                if (implicit_lte_temperature) {
                    if (nonlinear_hybrid_lte) {
                        ApplyHybridNonlinearImplicitLteCellExchange(
                            i, j, k, nonlinear_material_state,
                            pending_material_energy, cell_volume,
                            x, y, z, opacity_number_density,
                            radiation_arr, material_arr, nonlinear_remap_arr,
                            rosseland_arr, hybrid_thermodynamics,
                            implicit_lte_context, lte_status_arr);
                    } else {
                        ApplyImplicitLteCellExchange(
                            i, j, k, material_state, pending_material_energy,
                            cell_volume, x, y, z, opacity_number_density,
                            radiation_arr, material_arr, rosseland_arr,
                            implicit_lte_context, lte_status_arr);
                    }
                } else {
                    amrex::Real const temperature_squared =
                        electron_temperature * electron_temperature;
                    amrex::Real const equilibrium_total_energy = radiation_constant
                        * temperature_squared * temperature_squared * cell_volume;
                    amrex::Real positive_exchange = 0.0_rt;
                    amrex::Real negative_exchange = 0.0_rt;

                    // First determine the conservative material-energy limiter
                    // shared by all groups. Absorption in one group can fund
                    // emission in another within this local LTE operator.
                    for (int group = 0;
                         group < implicit_lte_context.num_groups; ++group)
                    {
                        amrex::Real const group_energy =
                            implicit_lte_context.energy_groups
                                .representativeEnergy(group);
                        amrex::Real const planck_absorption =
                            implicit_lte_context.planckAbsorption(
                                opacity_number_density, x, y, z, group_energy,
                                electron_density, electron_temperature);
                        amrex::Real const planck_emission =
                            implicit_lte_context.planckEmission(
                                opacity_number_density, x, y, z, group_energy,
                                electron_density, electron_temperature);
                        amrex::Real rosseland_opacity_value = 0.0_rt;
                        if (implicit_lte_context.enable_diffusion) {
                            rosseland_opacity_value =
                                implicit_lte_context.rosselandOpacity(
                                opacity_number_density, x, y, z, group_energy,
                                electron_density, electron_temperature);
                        }
                        amrex::Real const old_radiation_energy =
                            radiation_arr(i, j, k, group);
                        amrex::Real const equilibrium_energy =
                            equilibrium_total_energy
                            * implicit_lte_context.energy_groups.planckFraction(
                                group, PhysConst::kb * electron_temperature);
                        PlanckExchangeResult const exchange =
                            EvaluatePlanckExchange(
                                planck_absorption, planck_emission,
                                old_radiation_energy, equilibrium_energy,
                                implicit_lte_context.dt);
                        if (!exchange.valid || rosseland_opacity_value < 0.0_rt
                            || !amrex::Math::isfinite(rosseland_opacity_value)
                            || !amrex::Math::isfinite(
                                exchange.exchange_energy))
                        {
                            lte_status_arr(i, j, k, 0) = 1;
                            return;
                        }
                        if (implicit_lte_context.enable_diffusion) {
                            rosseland_arr(i, j, k, group) =
                                rosseland_opacity_value;
                        }
                        if (exchange.exchange_energy > 0.0_rt) {
                            positive_exchange += exchange.exchange_energy;
                        } else {
                            negative_exchange += exchange.exchange_energy;
                        }
                    }

                    amrex::Real positive_scale = 1.0_rt;
                    if (positive_exchange + negative_exchange
                            > available_material_energy
                        && positive_exchange > 0.0_rt)
                    {
                        positive_scale = amrex::max(
                            0.0_rt,
                            (available_material_energy - negative_exchange)
                                / positive_exchange);
                    }

                    amrex::Real total_exchange = 0.0_rt;
                    for (int group = 0;
                         group < implicit_lte_context.num_groups; ++group)
                    {
                        amrex::Real const group_energy =
                            implicit_lte_context.energy_groups
                                .representativeEnergy(group);
                        amrex::Real const planck_absorption =
                            implicit_lte_context.planckAbsorption(
                                opacity_number_density, x, y, z, group_energy,
                                electron_density, electron_temperature);
                        amrex::Real const planck_emission =
                            implicit_lte_context.planckEmission(
                                opacity_number_density, x, y, z, group_energy,
                                electron_density, electron_temperature);
                        amrex::Real const old_radiation_energy =
                            radiation_arr(i, j, k, group);
                        amrex::Real const equilibrium_energy =
                            equilibrium_total_energy
                            * implicit_lte_context.energy_groups.planckFraction(
                                group, PhysConst::kb * electron_temperature);
                        PlanckExchangeResult const exchange =
                            EvaluatePlanckExchange(
                                planck_absorption, planck_emission,
                                old_radiation_energy, equilibrium_energy,
                                implicit_lte_context.dt);
                        if (!exchange.valid) {
                            lte_status_arr(i, j, k, 0) = 1;
                            return;
                        }
                        amrex::Real exchange_energy =
                            exchange.exchange_energy;
                        if (exchange_energy > 0.0_rt) {
                            exchange_energy *= positive_scale;
                        }
                        radiation_arr(i, j, k, group) =
                            old_radiation_energy + exchange_energy;
                        total_exchange += exchange_energy;
                    }
                    material_arr(i, j, k) -= total_exchange;
                }
            });
        }

        int const lte_invalid = lte_cell_status.max(0);
        int const lte_unconverged = lte_cell_status.max(1);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            lte_invalid == 0,
            "LTE radiation exchange encountered a negative/non-finite Planck or "
            "Rosseland coefficient, radiation energy, or material temperature.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            lte_unconverged == 0,
            "The implicit LTE temperature solve did not converge (reason code "
            + std::to_string(lte_unconverged)
            + ": 1=cooling bracket, 2=heating bracket, 3=residual signs, "
              "4=bisection, 5=unfunded emission, 6=remap residual). Increase "
            "radiation_transport.lte_exchange_max_iterations or relax "
            "radiation_transport.lte_exchange_tolerance.");

        if (enable_diffusion) {
            rosseland_opacity.FillBoundary(warpx.Geom(lev).periodicity());
            // Positivity bound for an explicit d-dimensional diffusion
            // stencil. At a thick/thin seam the arithmetic face opacity can
            // be half the threshold-cell opacity, doubling the largest D;
            // the factor 4*d accounts for both neighbors per dimension and
            // that worst-case interface. The first spherical shell has a
            // coefficient 3 instead of the Cartesian 2, hence d_eff=3/2.
#if defined(WARPX_DIM_RSPHERE)
            amrex::Real const diffusion_stability_dimension = 1.5_rt;
#else
            auto const diffusion_stability_dimension =
                static_cast<amrex::Real>(AMREX_SPACEDIM);
#endif
            amrex::Real requested_diffusion_substeps = amrex::max(
                1.0_rt,
                std::ceil(
                    4.0_rt * diffusion_stability_dimension
                        * PhysConst::c * dt
                    / (3.0_rt * m_minimum_diffusion_optical_depth
                       * m_diffusion_cfl * min_cell_size)));
            amrex::Real max_escape_factor = 0.0_rt;
            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
                max_escape_factor = amrex::max(
                    max_escape_factor,
                    amrex::max(
                        DiffusionEscapeFactor(m_diffusion_boundary_lo[direction]),
                        DiffusionEscapeFactor(
                            m_diffusion_boundary_hi[direction])));
            }
            if (max_escape_factor > 0.0_rt) {
                // Vacuum/Marshak faces remove alpha*c*E*A. A/V <= 2/dx covers
                // the outer RCYLINDER shell, where A/V = 2R/(2R dr - dr^2).
                requested_diffusion_substeps = amrex::max(
                    requested_diffusion_substeps,
                    std::ceil(
                        2.0_rt * max_escape_factor * PhysConst::c * dt
                        / (m_diffusion_cfl * min_cell_size)));
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                requested_diffusion_substeps
                    <= static_cast<amrex::Real>(m_max_diffusion_substeps),
                "Radiation diffusion requires more than "
                "radiation_transport.max_diffusion_substeps. Increase that limit, "
                "increase minimum_diffusion_optical_depth, or reduce the timestep.");
            int const diffusion_substeps =
                static_cast<int>(requested_diffusion_substeps);
            amrex::Real const diffusion_dt = dt / diffusion_substeps;
            amrex::Real const minimum_optical_depth =
                m_minimum_diffusion_optical_depth;
            amrex::MultiFab next_diffusion_energy(
                diffusion_energy.boxArray(), diffusion_energy.DistributionMap(),
                num_groups, 0);
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpMax>
                diffusion_residual_reduce_ops;
            amrex::ReduceData<amrex::Real, int>
                diffusion_residual_reduce_data(diffusion_residual_reduce_ops);
            using DiffusionResidualReduceTuple =
                typename decltype(diffusion_residual_reduce_data)::Type;
            amrex::Gpu::DeviceScalar<amrex::Real> escaped_energy(0.0_rt);
            amrex::Real* const escaped_energy_ptr = escaped_energy.dataPtr();
            amrex::Gpu::DeviceScalar<amrex::Real> escaped_momentum_0(0.0_rt);
            amrex::Gpu::DeviceScalar<amrex::Real> escaped_momentum_1(0.0_rt);
            amrex::Gpu::DeviceScalar<amrex::Real> escaped_momentum_2(0.0_rt);
            amrex::GpuArray<amrex::Real*, 3> const escaped_momentum_ptr{
                escaped_momentum_0.dataPtr(), escaped_momentum_1.dataPtr(),
                escaped_momentum_2.dataPtr()};
            auto const diffusion_boundary_lo = m_diffusion_boundary_lo;
            auto const diffusion_boundary_hi = m_diffusion_boundary_hi;
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ) \
    || defined(WARPX_DIM_RSPHERE)
            bool const allow_radial_interface_escape =
                !m_enable_particle_conversion;
#else
            bool const allow_radial_interface_escape = false;
#endif

            for (int diffusion_step = 0;
                 diffusion_step < diffusion_substeps; ++diffusion_step)
            {
                diffusion_energy.FillBoundary(warpx.Geom(lev).periodicity());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (amrex::MFIter mfi(diffusion_energy, amrex::TilingIfNotGPU());
                     mfi.isValid(); ++mfi)
                {
                    amrex::Box const& box = mfi.tilebox();
                    amrex::Array4<amrex::Real const> const old_arr =
                        diffusion_energy.const_array(mfi);
                    amrex::Array4<amrex::Real> const new_arr =
                        next_diffusion_energy.array(mfi);
                    amrex::Array4<amrex::Real const> const opacity_arr =
                        rosseland_opacity.const_array(mfi);
                    amrex::Array4<amrex::Real> diffusion_group_momentum_arr;
                    if (enable_momentum_coupling) {
                        diffusion_group_momentum_arr =
                            diffusion_group_material_momentum.array(mfi);
                    }

                    diffusion_residual_reduce_ops.eval(
                        box, num_groups, diffusion_residual_reduce_data,
                        [=] AMREX_GPU_DEVICE (
                            int i, int j, int k, int group)
                            -> DiffusionResidualReduceTuple
                    {
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
                        amrex::Real const r_lo =
                            plo[0] + (i - domain_lo.x) * dx[0];
                        amrex::Real const r_hi = r_lo + dx[0];
#if defined(WARPX_DIM_RCYLINDER)
                        amrex::Real const cell_volume = MathConst::pi
                            * (r_hi * r_hi - r_lo * r_lo);
#else
                        amrex::Real const cell_volume = MathConst::pi
                            * (r_hi * r_hi - r_lo * r_lo) * dx[1];
#endif
#elif defined(WARPX_DIM_RSPHERE)
                        amrex::Real const r_lo =
                            plo[0] + (i - domain_lo.x) * dx[0];
                        amrex::Real const r_hi = r_lo + dx[0];
                        amrex::Real const cell_volume =
                            4.0_rt / 3.0_rt * MathConst::pi
                            * (r_hi * r_hi * r_hi - r_lo * r_lo * r_lo);
#else
                        amrex::Real const cell_volume =
                            AMREX_D_TERM(dx[0], * dx[1], * dx[2]);
#endif
                        amrex::Real const old_energy = old_arr(i, j, k, group);
                        amrex::Real const energy_density = old_energy / cell_volume;
                        amrex::Real energy_rate = 0.0_rt;
                        amrex::GpuArray<amrex::Real, 3> cell_flux{
                            0.0_rt, 0.0_rt, 0.0_rt};

                        for (int direction = 0;
                             direction < AMREX_SPACEDIM; ++direction)
                        {
                            for (int side = -1; side <= 1; side += 2) {
                                int const ni = i + (direction == 0 ? side : 0);
                                int const nj = j + (direction == 1 ? side : 0);
                                int const nk = k + (direction == 2 ? side : 0);
                                amrex::GpuArray<int, AMREX_SPACEDIM> const
                                    neighbor_indices{
                                        AMREX_D_DECL(ni, nj, nk)};
                                amrex::GpuArray<int, AMREX_SPACEDIM> const
                                    domain_lower{
                                        AMREX_D_DECL(
                                            domain_lo.x, domain_lo.y, domain_lo.z)};
                                amrex::GpuArray<int, AMREX_SPACEDIM> const
                                    domain_upper{
                                        AMREX_D_DECL(
                                            domain_hi.x, domain_hi.y, domain_hi.z)};
                                int const index = neighbor_indices[direction];
                                int const index_lo = domain_lower[direction];
                                int const index_hi = domain_upper[direction];
                                int const side_boundary = side < 0
                                    ? diffusion_boundary_lo[direction]
                                    : diffusion_boundary_hi[direction];
#if defined(WARPX_DIM_RCYLINDER)
                                amrex::Real const face_radius =
                                    plo[0]
                                    + (i - domain_lo.x
                                       + (side > 0 ? 1.0_rt : 0.0_rt)) * dx[0];
                                amrex::Real const face_area =
                                    2.0_rt * MathConst::pi * face_radius;
#elif defined(WARPX_DIM_RSPHERE)
                                amrex::Real const face_radius =
                                    plo[0]
                                    + (i - domain_lo.x
                                       + (side > 0 ? 1.0_rt : 0.0_rt)) * dx[0];
                                amrex::Real const face_area = 4.0_rt * MathConst::pi
                                    * face_radius * face_radius;
#elif defined(WARPX_DIM_RZ)
                                amrex::Real const face_radius =
                                    plo[0]
                                    + (i - domain_lo.x
                                       + (side > 0 ? 1.0_rt : 0.0_rt)) * dx[0];
                                amrex::Real const face_area = direction == 0
                                    ? 2.0_rt * MathConst::pi * face_radius * dx[1]
                                    : MathConst::pi
                                        * (r_hi * r_hi - r_lo * r_lo);
#else
                                amrex::Real const face_area =
                                    cell_volume / dx[direction];
#endif
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ) \
    || defined(WARPX_DIM_RSPHERE)
                                // Axis / origin: zero area and a symmetry plane.
                                if (direction == 0 && side < 0 && i == domain_lo.x)
                                {
                                    continue;
                                }
#endif
                                bool const neighbor_outside =
                                    (index < index_lo || index > index_hi)
                                    && periodic[direction] == 0;
                                if (neighbor_outside) {
                                    cell_flux[direction] += 0.5_rt * side
                                        * DiffusionEscapeFactor(side_boundary)
                                        * PhysConst::c * energy_density;
                                    AddDiffusionEscape(
                                        side_boundary, energy_density,
                                        face_area, diffusion_dt, direction, side,
                                        energy_rate, escaped_energy_ptr,
                                        escaped_momentum_ptr);
                                    continue;
                                }

#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
                                amrex::Real const neighbor_r_lo =
                                    plo[0] + (ni - domain_lo.x) * dx[0];
                                amrex::Real const neighbor_r_hi =
                                    neighbor_r_lo + dx[0];
#if defined(WARPX_DIM_RCYLINDER)
                                amrex::Real const neighbor_volume = MathConst::pi
                                    * (neighbor_r_hi * neighbor_r_hi
                                       - neighbor_r_lo * neighbor_r_lo);
#else
                                amrex::Real const neighbor_volume = MathConst::pi
                                    * (neighbor_r_hi * neighbor_r_hi
                                       - neighbor_r_lo * neighbor_r_lo) * dx[1];
#endif
#elif defined(WARPX_DIM_RSPHERE)
                                amrex::Real const neighbor_r_lo =
                                    plo[0] + (ni - domain_lo.x) * dx[0];
                                amrex::Real const neighbor_r_hi =
                                    neighbor_r_lo + dx[0];
                                amrex::Real const neighbor_volume =
                                    4.0_rt / 3.0_rt * MathConst::pi
                                    * (neighbor_r_hi * neighbor_r_hi
                                           * neighbor_r_hi
                                       - neighbor_r_lo * neighbor_r_lo
                                           * neighbor_r_lo);
#else
                                amrex::Real const neighbor_volume =
                                    AMREX_D_TERM(dx[0], * dx[1], * dx[2]);
#endif
                                amrex::Real const neighbor_density =
                                    old_arr(ni, nj, nk, group) / neighbor_volume;
                                amrex::Real const cell_opacity =
                                    opacity_arr(i, j, k, group);
                                amrex::Real const neighbor_opacity =
                                    opacity_arr(ni, nj, nk, group);
                                bool const cell_is_thick =
                                    cell_opacity * min_cell_size
                                    >= minimum_optical_depth;
                                bool const neighbor_is_thick =
                                    neighbor_opacity * min_cell_size
                                    >= minimum_optical_depth;
                                // Radial coordinates have an unambiguous
                                // outward direction, so an open radial-high
                                // condition can approximate escape directly
                                // at a thick shell's outer surface. Cartesian
                                // faces cannot distinguish the outer surface
                                // from a cavity surface. Particle conversion
                                // transports both geometries through the thin
                                // region to the actual domain boundary.
                                if (allow_radial_interface_escape
                                    && direction == 0
                                    && cell_is_thick && !neighbor_is_thick
                                    && DiffusionEscapeFactor(side_boundary)
                                        > 0.0_rt)
                                {
                                    cell_flux[direction] += 0.5_rt * side
                                        * DiffusionEscapeFactor(side_boundary)
                                        * PhysConst::c * energy_density;
                                    AddDiffusionEscape(
                                        side_boundary, energy_density,
                                        face_area, diffusion_dt, direction, side,
                                        energy_rate, escaped_energy_ptr,
                                        escaped_momentum_ptr);
                                    continue;
                                }
                                if (allow_radial_interface_escape
                                    && direction == 0
                                    && !cell_is_thick && neighbor_is_thick)
                                {
                                    int const face_boundary = side < 0
                                        ? diffusion_boundary_hi[direction]
                                        : diffusion_boundary_lo[direction];
                                    if (DiffusionEscapeFactor(face_boundary)
                                        > 0.0_rt)
                                    {
                                        continue;
                                    }
                                }
                                bool const touches_diffusion_region =
                                    cell_is_thick || neighbor_is_thick;
                                amrex::Real const face_opacity =
                                    0.5_rt * (cell_opacity + neighbor_opacity);
                                if (!touches_diffusion_region
                                    || face_opacity <= 0.0_rt)
                                {
                                    continue;
                                }

                                amrex::Real const gradient =
                                    (neighbor_density - energy_density) / dx[direction];
                                amrex::Real const face_density = amrex::max(
                                    0.0_rt,
                                    0.5_rt * (energy_density + neighbor_density));
                                if (face_density == 0.0_rt) { continue; }
                                amrex::Real const dimensionless_gradient =
                                    std::abs(gradient)
                                    / (face_opacity * face_density);
                                amrex::Real const flux_limiter =
                                    (2.0_rt + dimensionless_gradient)
                                    / (6.0_rt + 3.0_rt * dimensionless_gradient
                                       + dimensionless_gradient
                                           * dimensionless_gradient);
                                amrex::Real const diffusion_coefficient =
                                    PhysConst::c * flux_limiter / face_opacity;
                                energy_rate += diffusion_coefficient * face_area
                                    * gradient;
                                cell_flux[direction] -= 0.5_rt * side
                                    * diffusion_coefficient * gradient;
                            }
                        }

                        if (enable_momentum_coupling) {
                            amrex::Real const cell_opacity =
                                opacity_arr(i, j, k, group);
                            for (int direction = 0;
                                 direction < AMREX_SPACEDIM; ++direction)
                            {
                                int const component =
                                    RadiationMomentumComponent(direction);
                                diffusion_group_momentum_arr(
                                    i, j, k, 3 * group + component) +=
                                    diffusion_dt * cell_volume * cell_opacity
                                    * cell_flux[direction] / PhysConst::c;
                            }
                        }

                        RadiationEnergyUpdateResult const update =
                            ApplyRadiationEnergyUpdate(
                                old_energy, diffusion_dt * energy_rate);
                        if (!update.valid) {
                            new_arr(i, j, k, group) = update.stored_energy;
                            return DiffusionResidualReduceTuple{0.0_rt, 1};
                        }
                        new_arr(i, j, k, group) = update.stored_energy;
                        return DiffusionResidualReduceTuple{update.residual, 0};
                    });
                }
                amrex::MultiFab::Copy(
                    diffusion_energy, next_diffusion_energy,
                    0, 0, num_groups, 0);
            }

            auto const diffusion_reduction =
                diffusion_residual_reduce_data.value();
            amrex::Real diffusion_numerical_energy_residual =
                amrex::get<0>(diffusion_reduction);
            amrex::ParallelDescriptor::ReduceRealSum(
                diffusion_numerical_energy_residual);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                amrex::get<1>(diffusion_reduction) == 0,
                "Radiation diffusion produced a negative or non-finite cell energy. "
                "Reduce radiation_transport.diffusion_cfl.");
            m_last_numerical_energy_residual +=
                diffusion_numerical_energy_residual;
            if (enable_momentum_coupling) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (amrex::MFIter mfi(
                         diffusion_group_material_momentum,
                         amrex::TilingIfNotGPU());
                     mfi.isValid(); ++mfi)
                {
                    amrex::Box const& box = mfi.tilebox();
                    amrex::Array4<amrex::Real const> const group_momentum_arr =
                        diffusion_group_material_momentum.const_array(mfi);
                    amrex::Array4<amrex::Real> const momentum_arr =
                        diffusion_material_momentum.array(mfi);
                    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (
                        int i, int j, int k) noexcept
                    {
                        for (int component = 0; component < 3; ++component) {
                            amrex::Real group_sum = 0.0_rt;
                            for (int group = 0; group < num_groups; ++group) {
                                group_sum += group_momentum_arr(
                                    i, j, k, 3 * group + component);
                            }
                            momentum_arr(i, j, k, component) = group_sum;
                        }
                    });
                }
            }
            amrex::Real escaped = escaped_energy.dataValue();
            amrex::ParallelDescriptor::ReduceRealSum(escaped);
            m_last_diffusion_boundary_energy_loss += escaped;
            amrex::GpuArray<amrex::Real, 3> escaped_momentum{
                escaped_momentum_0.dataValue(),
                escaped_momentum_1.dataValue(),
                escaped_momentum_2.dataValue()};
            amrex::ParallelDescriptor::ReduceRealSum(
                escaped_momentum.data(), 3);
            for (int component = 0; component < 3; ++component) {
                m_last_diffusion_boundary_momentum_loss[component] +=
                    escaped_momentum[component];
            }
            diffusion_energy.FillBoundary(warpx.Geom(lev).periodicity());

            if (m_enable_particle_conversion) {
                amrex::MFInfo const host_info =
                    amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
                amrex::MultiFab host_energy(
                    diffusion_energy.boxArray(), diffusion_energy.DistributionMap(),
                    num_groups, 1, host_info);
                amrex::MultiFab host_opacity(
                    rosseland_opacity.boxArray(), rosseland_opacity.DistributionMap(),
                    num_groups, 1, host_info);
                amrex::dtoh_memcpy(host_energy, diffusion_energy);
                amrex::dtoh_memcpy(host_opacity, rosseland_opacity);
                amrex::Gpu::streamSynchronize();

                amrex::Vector<amrex::ParticleReal> photon_x;
                amrex::Vector<amrex::ParticleReal> photon_y;
                amrex::Vector<amrex::ParticleReal> photon_z;
                amrex::Vector<amrex::ParticleReal> photon_ux;
                amrex::Vector<amrex::ParticleReal> photon_uy;
                amrex::Vector<amrex::ParticleReal> photon_uz;
                amrex::Vector<amrex::ParticleReal> photon_weight;
                std::uint64_t const time_key = std::hash<double>{}(
                    static_cast<double>(current_time));

                for (amrex::MFIter mfi(host_energy); mfi.isValid(); ++mfi) {
                    amrex::Box const& box = mfi.validbox();
                    amrex::Array4<amrex::Real> const host_energy_arr =
                        host_energy.array(mfi);
                    amrex::Array4<amrex::Real const> const host_opacity_arr =
                        host_opacity.const_array(mfi);
                    for (amrex::BoxIterator bit(box); bit.ok(); ++bit) {
                        auto const [i, j, k] = bit().dim3();
                        for (int group = 0; group < num_groups; ++group) {
                            amrex::Real const cell_energy =
                                host_energy_arr(i, j, k, group);
                            bool const optically_thin =
                                host_opacity_arr(i, j, k, group) * min_cell_size
                                < minimum_optical_depth;
                            if (!optically_thin
                                || cell_energy
                                    <= m_particle_conversion_energy_threshold)
                            {
                                continue;
                            }

                            amrex::Real const emission_photon_energy =
                                m_group_photon_energies_h[group];
                            auto const photon_momentum_magnitude =
                                static_cast<amrex::ParticleReal>(
                                    emission_photon_energy
                                    / (PhysConst::m_e * PhysConst::c));

                            std::uint64_t random_state = time_key
                                ^ (static_cast<std::uint64_t>(
                                       static_cast<std::int64_t>(i))
                                   * 0x9e3779b97f4a7c15ULL)
                                ^ (static_cast<std::uint64_t>(
                                       static_cast<std::int64_t>(j))
                                   * 0xbf58476d1ce4e5b9ULL)
                                ^ (static_cast<std::uint64_t>(
                                       static_cast<std::int64_t>(k))
                                   * 0x94d049bb133111ebULL)
                                ^ (static_cast<std::uint64_t>(group)
                                   * 0xd6e8feb86659fd93ULL);
                            auto next_uniform = [&random_state] () noexcept {
                                random_state += 0x9e3779b97f4a7c15ULL;
                                std::uint64_t value = random_state;
                                value = (value ^ (value >> 30U))
                                    * 0xbf58476d1ce4e5b9ULL;
                                value = (value ^ (value >> 27U))
                                    * 0x94d049bb133111ebULL;
                                value ^= value >> 31U;
                                return static_cast<amrex::ParticleReal>(value >> 11U)
                                    * 0x1.0p-53_prt;
                            };

#if defined(WARPX_DIM_3D)
                            amrex::ParticleReal const x =
                                plo[0] + (i - domain_lo.x + 0.5_prt) * dx[0];
                            amrex::ParticleReal const y =
                                plo[1] + (j - domain_lo.y + 0.5_prt) * dx[1];
                            amrex::ParticleReal const z =
                                plo[2] + (k - domain_lo.z + 0.5_prt) * dx[2];
#elif defined(WARPX_DIM_XZ)
                            amrex::ParticleReal const x =
                                plo[0] + (i - domain_lo.x + 0.5_prt) * dx[0];
                            amrex::ParticleReal const y = 0.0_prt;
                            amrex::ParticleReal const z =
                                plo[1] + (j - domain_lo.y + 0.5_prt) * dx[1];
#elif defined(WARPX_DIM_RZ)
                            amrex::ParticleReal const radius =
                                plo[0] + (i - domain_lo.x + 0.5_prt) * dx[0];
                            amrex::ParticleReal const position_angle =
                                MathConst::tau * next_uniform();
                            amrex::ParticleReal const x =
                                radius * std::cos(position_angle);
                            amrex::ParticleReal const y =
                                radius * std::sin(position_angle);
                            amrex::ParticleReal const z =
                                plo[1] + (j - domain_lo.y + 0.5_prt) * dx[1];
#elif defined(WARPX_DIM_RCYLINDER)
                            amrex::ParticleReal const radius =
                                plo[0] + (i - domain_lo.x + 0.5_prt) * dx[0];
                            amrex::ParticleReal const position_angle =
                                MathConst::tau * next_uniform();
                            amrex::ParticleReal const x =
                                radius * std::cos(position_angle);
                            amrex::ParticleReal const y =
                                radius * std::sin(position_angle);
                            amrex::ParticleReal const z = 0.0_prt;
#elif defined(WARPX_DIM_RSPHERE)
                            amrex::ParticleReal const radius =
                                plo[0] + (i - domain_lo.x + 0.5_prt) * dx[0];
                            amrex::ParticleReal const position_mu =
                                2.0_prt * next_uniform() - 1.0_prt;
                            amrex::ParticleReal const position_angle =
                                MathConst::tau * next_uniform();
                            amrex::ParticleReal const position_sin =
                                std::sqrt(1.0_prt - position_mu * position_mu);
                            amrex::ParticleReal const x =
                                radius * position_sin * std::cos(position_angle);
                            amrex::ParticleReal const y =
                                radius * position_sin * std::sin(position_angle);
                            amrex::ParticleReal const z = radius * position_mu;
#else
                            amrex::ParticleReal const x = 0.0_prt;
                            amrex::ParticleReal const y = 0.0_prt;
                            amrex::ParticleReal const z =
                                plo[0] + (i - domain_lo.x + 0.5_prt) * dx[0];
#endif
                            amrex::ParticleReal const direction_mu =
                                2.0_prt * next_uniform() - 1.0_prt;
                            amrex::ParticleReal const direction_angle =
                                MathConst::tau * next_uniform();
                            amrex::ParticleReal const direction_sin =
                                std::sqrt(1.0_prt - direction_mu * direction_mu);

                            photon_x.push_back(x);
                            photon_y.push_back(y);
                            photon_z.push_back(z);
                            photon_ux.push_back(
                                photon_momentum_magnitude * direction_sin
                                * std::cos(direction_angle));
                            photon_uy.push_back(
                                photon_momentum_magnitude * direction_sin
                                * std::sin(direction_angle));
                            photon_uz.push_back(
                                photon_momentum_magnitude * direction_mu);
                            photon_weight.push_back(
                                static_cast<amrex::ParticleReal>(
                                    cell_energy / emission_photon_energy));
                            host_energy_arr(i, j, k, group) = 0.0_rt;
                        }
                    }
                }

                amrex::htod_memcpy(diffusion_energy, host_energy);
                amrex::Gpu::streamSynchronize();
                amrex::Vector<amrex::Vector<amrex::ParticleReal>> const attributes{
                    photon_weight};
                amrex::Vector<amrex::Vector<int>> const integer_attributes{};
                photons.AddNParticles(
                    lev, static_cast<long>(photon_x.size()),
                    photon_x, photon_y, photon_z,
                    photon_ux, photon_uy, photon_uz,
                    1, attributes, 0, integer_attributes,
                    /*uniqueparticles=*/1);
                diffusion_energy.FillBoundary(warpx.Geom(lev).periodicity());
            }

            if (m_enable_momentum_coupling) {
                m_last_numerical_energy_residual += ApplyRadiationMomentumWork(
                    particles, m_momentum_species, diffusion_material_momentum,
                    material_momentum, material_kinetic_energy,
                    diffusion_energy,
                    "Radiation diffusion recoil requires more bulk kinetic work "
                    "than the local diffusion energy supplies. Reduce the "
                    "timestep or inspect the configured momentum species.");
                diffusion_energy.FillBoundary(
                    warpx.Geom(lev).periodicity());
            }
        }
    }

    material_energy.FillBoundary(warpx.Geom(lev).periodicity());
    if (nonlinear_lte_remap != nullptr) {
        nonlinear_lte_remap->FillBoundary(warpx.Geom(lev).periodicity());
    }
    material_kinetic_energy.FillBoundary(warpx.Geom(lev).periodicity());
    material_momentum.FillBoundary(warpx.Geom(lev).periodicity());

    if (gate_on_kinetic_density) {
        m_last_numerical_energy_residual += ApplyKineticElectronThermalEnergy(
            *kinetic_electrons, lev, material_energy, kinetic_moments,
            warpx.Geom(lev), info);
        material_energy.FillBoundary(warpx.Geom(lev).periodicity());
    }

    if (m_track_energy_balance) {
        amrex::Real const final_radiation_energy =
            TotalRadiationEnergy(photons, fields, lev, m_num_groups);
        amrex::Real const material_exchange =
            material_energy.sum(0, /*local=*/false)
            + material_kinetic_energy.sum(0, /*local=*/false);
        amrex::Real const residual_boundary_energy_loss =
            initial_radiation_energy
            - final_radiation_energy - material_exchange;
        m_last_boundary_energy_loss =
            m_last_streaming_boundary_energy_loss
            + m_last_diffusion_boundary_energy_loss;
        amrex::Real const roundoff_tolerance = 1000.0_rt
            * amrex::max(
                std::numeric_limits<amrex::Real>::epsilon(),
                static_cast<amrex::Real>(
                    std::numeric_limits<amrex::ParticleReal>::epsilon()))
            * amrex::max(
                std::abs(material_exchange),
                amrex::max(
                    std::abs(initial_radiation_energy),
                    std::abs(final_radiation_energy)));
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::abs(residual_boundary_energy_loss
                - m_last_boundary_energy_loss
                - m_last_numerical_energy_residual) <= roundoff_tolerance,
            "Explicit streaming-packet and diffusion-face boundary losses do "
            "not close the radiation/material energy balance. This indicates "
            "an untracked radiation representation change or boundary path.");
        m_cumulative_boundary_energy_loss += m_last_boundary_energy_loss;
    }
    m_cumulative_numerical_energy_residual +=
        m_last_numerical_energy_residual;
}
