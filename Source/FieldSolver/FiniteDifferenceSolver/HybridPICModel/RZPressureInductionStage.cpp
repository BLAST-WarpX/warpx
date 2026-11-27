/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "RZPressureInductionStage.H"

#include "RZPressureInduction.H"
#include "Utils/TextMsg.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <array>
#include <cmath>
#include <limits>

namespace warpx::hybrid {
using namespace amrex::literals;

namespace {
constexpr amrex::Real temperature_tolerance_multiplier = amrex::Real(256.0);
constexpr amrex::Real energy_tolerance_multiplier = amrex::Real(2048.0);

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
finiteReal (amrex::Real const value) {
    return amrex::Math::isfinite(value);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
positiveFinite (amrex::Real const value) {
    return finiteReal(value) && value > amrex::Real(0.0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
near (amrex::Real const value, amrex::Real const expected,
      amrex::Real const multiplier) {
    amrex::Real const scale = amrex::max(
        amrex::max(amrex::Math::abs(value), amrex::Math::abs(expected)),
        std::numeric_limits<amrex::Real>::min());
    return amrex::Math::abs(value - expected) <=
           multiplier * std::numeric_limits<amrex::Real>::epsilon() * scale;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
admissibleState (ElectronThermodynamicsExecutor const& eos,
                 amrex::Real const charge, amrex::Real const U,
                 amrex::Real const T, amrex::Real const temperature_floor,
                 ElectronThermodynamicState& state) {
    if (!positiveFinite(charge) || !positiveFinite(U) || !positiveFinite(T)) {
        return false;
    }

    amrex::Real const inverted_temperature =
        eos.temperatureFromChargeDensityThermalEnergyDensity(charge, U);
    if (!positiveFinite(inverted_temperature) ||
        inverted_temperature <
            amrex::max(eos.minimumTemperature(), temperature_floor) ||
        inverted_temperature > eos.maximumTemperature() ||
        !near(T, inverted_temperature, temperature_tolerance_multiplier)) {
        return false;
    }

    state = eos.stateFromChargeDensityTemperature(charge, inverted_temperature);
    amrex::Real const energy_tolerance =
        energy_tolerance_multiplier *
        std::numeric_limits<amrex::Real>::epsilon() *
        amrex::max(amrex::Math::abs(U),
                   std::numeric_limits<amrex::Real>::min());
    return positiveFinite(state.pressure) &&
           positiveFinite(state.internal_energy_density) &&
           positiveFinite(state.heat_capacity_density) &&
           amrex::Math::abs(state.internal_energy_density - U) <=
               energy_tolerance;
}

int
globalInvalid (int invalid) {
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    return invalid;
}
} // namespace

bool
tryAdvanceRZPressureInductionStage (
    amrex::MultiFab& B_theta, amrex::MultiFab& U, amrex::MultiFab& T,
    amrex::MultiFab const& charge, amrex::MultiFab const& caloric_volume,
    amrex::Geometry const& geometry, amrex::Real dt,
    ElectronThermodynamicsExecutor eos, amrex::Real const temperature_floor) {
    // The first bounded stage deliberately supports the charge-only ideal EOS.
    // Tables and analytic latent reservoirs have distinct admissibility and
    // composition rules, so they are rejected rather than approximated here.
    if (!eos.isIdealGas() || eos.m_num_latent_transitions != 0 ||
        eos.m_num_materials != 0 || eos.m_use_resolved_material_selection) {
        return false;
    }

    if (!finiteReal(temperature_floor) ||
        temperature_floor < amrex::Real(0.0) ||
        temperature_floor > eos.maximumTemperature()) {
        return false;
    }
    amrex::Real const stage_minimum_temperature =
        amrex::max(eos.minimumTemperature(), temperature_floor);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        &B_theta != &U && &B_theta != &T && &U != &T && geometry.Coord() == 1 &&
            geometry.ProbLo(0) == 0 && !geometry.isPeriodic(0) &&
            geometry.isPeriodic(1) && B_theta.ixType().nodeCentered() &&
            finiteReal(dt) && dt > 0,
        "RZ pressure induction requires distinct nodal state arrays, an "
        "axis/PEC, periodic-z domain, and positive finite dt.");

    // Object identity is necessary but not sufficient: AMReX aliases can have
    // distinct C++ objects backed by the same allocation.  This private helper
    // therefore documents an independently-owned, nonaliasing caller contract.
    constexpr int field_count = 5;
    std::array<amrex::MultiFab const*, field_count> const fields{
        &B_theta, &U, &T, &charge, &caloric_volume};
    for (auto const* field : fields) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            field->nComp() == 1 && field->boxArray() == B_theta.boxArray() &&
                field->DistributionMap() == B_theta.DistributionMap() &&
                field->ixType() == B_theta.ixType(),
            "RZ pressure-induction stage fields must have matching scalar "
            "nodal layouts.");
    }
    for (int left = 0; left < field_count; ++left) {
        for (int right = left + 1; right < field_count; ++right) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                fields[left] != fields[right],
                "RZ pressure-induction stage fields must be distinct objects; "
                "their backing stores must also be nonaliasing.");
        }
    }

    amrex::BoxArray const& boxes = B_theta.boxArray();
    amrex::DistributionMapping const& mapping = B_theta.DistributionMap();
    amrex::MultiFab candidate_B(boxes, mapping, 1, 0);
    amrex::MultiFab candidate_U(boxes, mapping, 1, 0);
    amrex::MultiFab candidate_T(boxes, mapping, 1, 0);
    amrex::MultiFab pressure(boxes, mapping, 1, 0);

    auto const nodal_domain =
        amrex::convert(geometry.Domain(), B_theta.ixType());
    int const axis = nodal_domain.smallEnd(0);

    amrex::ReduceOps<amrex::ReduceOpMax> initial_ops;
    amrex::ReduceData<int> initial_data(initial_ops);
    for (amrex::MFIter it(pressure); it.isValid(); ++it) {
        auto const B = B_theta.const_array(it);
        auto const input_U = U.const_array(it);
        auto const input_T = T.const_array(it);
        auto const rho = charge.const_array(it);
        auto const volume = caloric_volume.const_array(it);
        auto const output_pressure = pressure.array(it);
        initial_ops.eval(
            it.validbox(), initial_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int> {
                ElectronThermodynamicState state{};
                bool const valid =
                    admissibleState(eos, rho(i, j, k), input_U(i, j, k),
                                    input_T(i, j, k), stage_minimum_temperature,
                                    state) &&
                    positiveFinite(volume(i, j, k)) && finiteReal(B(i, j, k)) &&
                    !(i == axis && B(i, j, k) != 0);
                output_pressure(i, j, k) =
                    valid ? state.pressure
                          : std::numeric_limits<amrex::Real>::quiet_NaN();
                return {!valid};
            });
    }
    if (globalInvalid(amrex::get<0>(initial_data.value())) != 0) {
        return false;
    }

    amrex::MultiFab::Copy(candidate_B, B_theta, 0, 0, 1, 0);
    amrex::MultiFab::Copy(candidate_U, U, 0, 0, 1, 0);
    if (!tryAdvanceRZPressureInduction(candidate_B, candidate_U, pressure,
                                       charge, caloric_volume, geometry, dt)) {
        return false;
    }

    amrex::ReduceOps<amrex::ReduceOpMax> candidate_ops;
    amrex::ReduceData<int> candidate_data(candidate_ops);
    for (amrex::MFIter it(candidate_T); it.isValid(); ++it) {
        auto const candidate_magnetic = candidate_B.const_array(it);
        auto const rho = charge.const_array(it);
        auto const updated_U = candidate_U.const_array(it);
        auto const updated_T = candidate_T.array(it);
        candidate_ops.eval(
            it.validbox(), candidate_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int> {
                amrex::Real const temperature =
                    eos.temperatureFromChargeDensityThermalEnergyDensity(
                        rho(i, j, k), updated_U(i, j, k));
                ElectronThermodynamicState state{};
                bool const valid =
                    admissibleState(eos, rho(i, j, k), updated_U(i, j, k),
                                    temperature, stage_minimum_temperature,
                                    state) &&
                    finiteReal(candidate_magnetic(i, j, k)) &&
                    !(i == axis && candidate_magnetic(i, j, k) != 0);
                updated_T(i, j, k) =
                    valid ? temperature
                          : std::numeric_limits<amrex::Real>::quiet_NaN();
                return {!valid};
            });
    }
    if (globalInvalid(amrex::get<0>(candidate_data.value())) != 0) {
        return false;
    }

    amrex::MultiFab::Copy(B_theta, candidate_B, 0, 0, 1, 0);
    amrex::MultiFab::Copy(U, candidate_U, 0, 0, 1, 0);
    amrex::MultiFab::Copy(T, candidate_T, 0, 0, 1, 0);
    return true;
}
} // namespace warpx::hybrid
