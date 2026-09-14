/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronThermodynamics.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/TwoTemperatureExchange.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>

using namespace amrex::literals;

namespace
{
    struct TestState
    {
        ElectronThermodynamicsExecutor thermodynamics;
        ElectronThermodynamicsExecutor::MaterialMassDensities mass_density{};
        amrex::Real charge_density;
        amrex::Real temperature;
    };

    struct TestResult
    {
        amrex::Real pressure;
        amrex::Real legacy_pressure;
        amrex::Real thermal_energy;
        amrex::Real heat_capacity;
        ElectronThermodynamicState combined_state;
        amrex::Real recovered_temperature;
        amrex::Real heated_temperature;
        ElectronEnergySourceUpdate source_update;
    };

    bool close (
        amrex::Real const actual,
        amrex::Real const expected,
        amrex::Real const epsilon_multiplier = 64.0_rt)
    {
        amrex::Real const scale = amrex::max(
            std::abs(expected), std::numeric_limits<amrex::Real>::min());
        return std::abs(actual - expected)
            <= epsilon_multiplier * std::numeric_limits<amrex::Real>::epsilon()
                * scale;
    }

    bool same_bits (amrex::Real const left, amrex::Real const right)
    {
        using Bits = std::conditional_t<
            sizeof(amrex::Real) == sizeof(float), std::uint32_t, std::uint64_t>;
        static_assert(sizeof(Bits) == sizeof(amrex::Real));
        return std::bit_cast<Bits>(left) == std::bit_cast<Bits>(right);
    }
}

int main (int argc, char* argv[])
{
    static_assert(
        std::is_trivially_copyable_v<ElectronThermodynamicsExecutor>,
        "The electron thermodynamics executor must be safe to capture by value "
        "in device kernels.");
    static_assert(
        std::is_trivially_copyable_v<ElectronThermodynamicState>,
        "Electron thermodynamic states must be safe device-kernel values.");
    static_assert(
        std::is_trivially_copyable_v<ElectronEnergySourceUpdate>,
        "Electron energy-source updates must be safe device-kernel values.");

    amrex::Initialize(argc, argv);
    bool pass = true;
    {
        constexpr int num_states = 6;
        std::array<TestState, num_states> host_states{};
        std::array<amrex::Real, num_states> const number_densities{
            1.0e24_rt, 0.0_rt, 1.0e10_rt, 1.0e20_rt, 1.0e30_rt, 4.0e28_rt};
        std::array<amrex::Real, num_states> const temperatures{
            10.0_rt * PhysConst::q_e / PhysConst::kb,
            2.0_rt, 3.0e2_rt, 1.0e6_rt, 1.0e9_rt, 2.5e7_rt};
        std::array<amrex::Real, num_states> const gammas{
            5.0_rt / 3.0_rt, 1.4_rt, 5.0_rt / 3.0_rt,
            1.4_rt, 5.0_rt / 3.0_rt, 1.2_rt};

        // Exercise both the omitted/default and explicit ideal_gas input
        // paths. CTest registers this executable once for each form, and the
        // configured executor is used by the first device-kernel state.
        amrex::ParmParse const pp_hybrid("hybrid_pic_model");
        ElectronThermodynamics configured;
        configured.ReadParameters(pp_hybrid, gammas[0]);

        for (int i = 0; i < num_states; ++i) {
            if (i == 0) {
                host_states[i].thermodynamics = configured.executor();
            } else {
                host_states[i].thermodynamics.m_gamma = gammas[i];
            }
            host_states[i].charge_density =
                number_densities[i] * PhysConst::q_e;
            host_states[i].temperature = temperatures[i];
        }

        // Standard zero-dimensional two-temperature relaxation.  This is the
        // exact frozen-coefficient source solution used as the conservative
        // contract between a grid electron EOS and kinetic ion thermal energy.
        amrex::Real const electron_temperature = 1200.0_rt;
        amrex::Real const ion_temperature = 200.0_rt;
        amrex::Real const electron_capacity = 3.0_rt;
        amrex::Real const ion_capacity = 2.0_rt;
        amrex::Real const conductance = 4.0_rt;
        amrex::Real const exchange_dt = 0.25_rt;
        TwoTemperatureExchangeState const exchange =
            exactTwoTemperatureExchange(
                electron_temperature, ion_temperature,
                electron_capacity, ion_capacity, conductance, exchange_dt);
        amrex::Real const expected_decay = std::exp(
            -conductance
                * (1.0_rt / electron_capacity + 1.0_rt / ion_capacity)
                * exchange_dt);
        pass = pass && exchange.valid
            && close(
                exchange.electron_temperature - exchange.ion_temperature,
                (electron_temperature - ion_temperature) * expected_decay)
            && close(
                electron_capacity * exchange.electron_temperature
                    + ion_capacity * exchange.ion_temperature,
                electron_capacity * electron_temperature
                    + ion_capacity * ion_temperature)
            && close(
                electron_capacity
                    * (exchange.electron_temperature - electron_temperature),
                exchange.electron_energy_change_density)
            && close(
                ion_capacity * (exchange.ion_temperature - ion_temperature),
                -exchange.electron_energy_change_density);
        bool const has_composition = configured.numMaterials() > 0;
        constexpr amrex::Real represented_mean_ionization = 20.0_rt;
        if (has_composition) {
            host_states[0].mass_density[0] =
                number_densities[0] / represented_mean_ionization
                * configured.executor().m_atomic_mass[0] * PhysConst::m_u;
        }

        amrex::Gpu::DeviceVector<TestState> device_states(num_states);
        amrex::Gpu::DeviceVector<TestResult> device_results(num_states);
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice,
            host_states.begin(), host_states.end(), device_states.begin());

        TestState const* const states = device_states.data();
        TestResult* const results = device_results.data();
        amrex::ParallelFor(num_states, [=] AMREX_GPU_DEVICE (int i) noexcept
        {
            TestState const state = states[i];
            amrex::Real const electron_density =
                state.charge_density / PhysConst::q_e;
            amrex::Real const pressure = state.thermodynamics
                .pressureFromChargeDensityTemperature(
                    state.charge_density, state.temperature);
            ElectronThermodynamicState const combined_state =
                state.thermodynamics
                    .stateFromMaterialMassDensitiesTemperature(
                        state.charge_density, state.mass_density,
                        state.temperature);
            amrex::Real const energy =
                combined_state.internal_energy_density;
            results[i].pressure = pressure;
            results[i].legacy_pressure =
                electron_density * PhysConst::kb * state.temperature;
            results[i].thermal_energy = energy;
            results[i].heat_capacity =
                combined_state.heat_capacity_density;
            results[i].combined_state = combined_state;
            results[i].recovered_temperature = state.thermodynamics
                .temperatureFromMaterialMassDensitiesThermalEnergyDensity(
                    state.charge_density, state.mass_density, energy);
            results[i].heated_temperature = state.thermodynamics
                .temperatureFromMaterialMassDensitiesThermalEnergyDensity(
                    state.charge_density, state.mass_density,
                    state.thermodynamics
                        .stateFromMaterialMassDensitiesTemperature(
                            state.charge_density, state.mass_density,
                            state.temperature + 0.25_rt)
                        .internal_energy_density);
            ElectronThermodynamicState const heated_state =
                state.thermodynamics
                    .stateFromMaterialMassDensitiesTemperature(
                        state.charge_density, state.mass_density,
                        state.temperature + 0.25_rt);
            results[i].source_update = state.thermodynamics
                .applyEnergyDensityIncrement(
                    state.charge_density, state.mass_density,
                    state.temperature,
                    heated_state.internal_energy_density - energy);
        });

        std::array<TestResult, num_states> host_results{};
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost,
            device_results.begin(), device_results.end(), host_results.begin());

        for (int i = 0; i < num_states; ++i) {
            TestResult const result = host_results[i];
            amrex::Real const gamma_minus_one = gammas[i] - 1.0_rt;
            amrex::Real expected_energy =
                result.legacy_pressure / gamma_minus_one;
            amrex::Real expected_capacity = number_densities[i]
                * PhysConst::kb / gamma_minus_one;
            bool const analytic_state =
                i == 0 && configured.isFixedChargeLatentEnergy();
            if (analytic_state) {
                ElectronThermodynamicsExecutor const executor =
                    configured.executor();
                for (int transition = 0;
                     transition < executor.m_num_latent_transitions;
                     ++transition)
                {
                    amrex::Real const transition_temperature =
                        executor.m_latent_transition_temperature[transition];
                    amrex::Real const sharpness =
                        executor.m_latent_transition_sharpness[transition];
                    amrex::Real const ratio = std::pow(
                        temperatures[i] / transition_temperature, sharpness);
                    amrex::Real const fraction = ratio / (1.0_rt + ratio);
                    amrex::Real const latent_energy_density =
                        number_densities[i]
                        * executor.m_latent_energy_per_electron[transition];
                    expected_energy += latent_energy_density * fraction;
                    expected_capacity += latent_energy_density * sharpness
                        * fraction * (1.0_rt - fraction) / temperatures[i];
                }
            }

            bool const state_pass =
                same_bits(result.pressure, result.legacy_pressure)
                && close(result.thermal_energy, expected_energy)
                && close(result.heat_capacity, expected_capacity)
                && same_bits(
                    result.combined_state.pressure, result.pressure)
                && same_bits(
                    result.combined_state.internal_energy_density,
                    result.thermal_energy)
                && same_bits(
                    result.combined_state.heat_capacity_density,
                    result.heat_capacity)
                && close(
                    result.combined_state.ion_number_density,
                    (has_composition && i == 0)
                        ? number_densities[i]
                            / represented_mean_ionization
                        : number_densities[i])
                && close(
                    result.combined_state.mean_atomic_number,
                    (has_composition && i == 0)
                        ? configured.executor().m_atomic_number[0]
                        : (number_densities[i] > 0.0_rt ? 1.0_rt : 0.0_rt))
                && close(
                    result.combined_state.mean_ionization,
                    (has_composition && i == 0)
                        ? represented_mean_ionization
                        : (number_densities[i] > 0.0_rt ? 1.0_rt : 0.0_rt))
                && (number_densities[i] > 0.0_rt
                    ? close(result.recovered_temperature, temperatures[i])
                        && close(
                            result.heated_temperature,
                            temperatures[i] + 0.25_rt)
                        && result.source_update.valid
                        && close(
                            result.source_update.temperature,
                            temperatures[i] + 0.25_rt)
                        && std::abs(
                            result.source_update.energy_residual_density)
                            <= 2048.0_rt
                                * std::numeric_limits<amrex::Real>::epsilon()
                                * amrex::max(
                                    std::abs(result.thermal_energy),
                                    std::abs(result.source_update
                                        .energy_change_density))
                    : result.recovered_temperature == 0.0_rt
                        && result.heated_temperature == 0.0_rt
                        && !result.source_update.valid);
            pass = pass && state_pass;
            std::cout << "state " << i
                      << ": P=" << result.pressure
                      << " U=" << result.thermal_energy
                      << " Cv=" << result.heat_capacity
                      << " T(U)=" << result.recovered_temperature
                      << (state_pass ? " PASS\n" : " FAIL\n");
        }

        ElectronThermodynamicsExecutor const configured_executor =
            configured.executor();
        pass = pass && configured_executor.m_gamma == gammas[0];
        pass = pass && configured.supportsEnergyCoupling();
        pass = pass
            && configured.supportsConstantHeatCapacityLte()
                == !configured.isFixedChargeLatentEnergy();
        pass = pass
            && configured_executor.m_num_latent_transitions
                == (configured.isFixedChargeLatentEnergy() ? 2 : 0);

        ElectronThermodynamics nonfinite_gamma;
        nonfinite_gamma.ReadParameters(
            pp_hybrid, std::numeric_limits<amrex::Real>::infinity());
        pass = pass && !nonfinite_gamma.supportsEnergyCoupling();
        pass = pass && !amrex::ParmParse::QueryUnusedInputs();
    }
    amrex::Finalize();

    std::cout << (pass ? "PASS\n" : "FAIL\n");
    return pass ? 0 : 1;
}
