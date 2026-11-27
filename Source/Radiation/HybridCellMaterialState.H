/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_HYBRIDCELLMATERIALSTATE_H_
#define WARPX_RADIATION_HYBRIDCELLMATERIALSTATE_H_

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronThermodynamics.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

namespace warpx::radiation::detail
{
    using namespace amrex::literals;

    struct HybridCellMaterialState
    {
        amrex::Real electron_density = 0.0_rt;
        amrex::Real electron_temperature = 0.0_rt;
        amrex::Real internal_energy = 0.0_rt;
        amrex::Real heat_capacity = 0.0_rt;
        amrex::Real available_energy = 0.0_rt;
        bool valid = true;
    };

    struct SignedMaterialAvailability
    {
        amrex::Real remaining_energy = 0.0_rt;
        amrex::Real tolerance = 0.0_rt;
        bool valid = true;
    };

    /** Remaining material energy above its floor after a signed source. */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    SignedMaterialAvailability
    EvaluateSignedMaterialAvailability (
        amrex::Real const available_energy,
        amrex::Real const pending_energy) noexcept
    {
        SignedMaterialAvailability result;
        amrex::Real const scale = amrex::max(
            amrex::max(std::abs(available_energy), std::abs(pending_energy)),
            std::numeric_limits<amrex::Real>::min());
        result.tolerance = 256.0_rt
            * std::numeric_limits<amrex::Real>::epsilon() * scale;
        amrex::Real const remaining_energy = available_energy + pending_energy;
        result.valid = available_energy >= 0.0_rt
            && amrex::Math::isfinite(available_energy)
            && amrex::Math::isfinite(pending_energy)
            && amrex::Math::isfinite(remaining_energy)
            && amrex::Math::isfinite(result.tolerance)
            && remaining_energy >= -result.tolerance;
        result.remaining_energy = result.valid
            ? amrex::max(0.0_rt, remaining_energy)
            : 0.0_rt;
        return result;
    }

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
} // namespace warpx::radiation::detail
#endif
