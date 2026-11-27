/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_NATIVE_RZ_HISTORY_ACCEPTANCE_BUDGET_H_
#define WARPX_NATIVE_RZ_HISTORY_ACCEPTANCE_BUDGET_H_

#include <AMReX_REAL.H>
#include <AMReX_INT.H>

#include <array>
#include <cmath>
#include <limits>

namespace warpx::radiation
{
/** Exact existing history acceptance law. Physical tolerance scales only
 * qualified radiation/ACTUAL source changes, never background material/EOS.
 * E/cPz are J, physical cLz is J m. The native caller retains these immutable
 * inputs so a private weight-only phase can minimize the UNMODIFIED law.
 */
struct NativeRZHistoryAcceptanceInputs
{
    amrex::Long material_records = 0, radiation_before_records = 0;
    amrex::Long radiation_after_records = 0, cells = 0;
    int subintervals = 0, ranks = 0;
    std::array<long double, 3> before_operands{}, after_operands{}, material_operands{};
    std::array<amrex::Real, 3> step_operands{}, step_bounds{};
    std::array<long double, 3> wall_operands{};
    std::array<long double, 3> reported_scale{}, accumulated_operands{};
    long double eos_scale = 0, eos_bound = 0, physical_source_budget = 0;
    amrex::Real physical_energy_scale = 0, radial_lever = 0, energy_tolerance = 0;
};

struct NativeRZHistoryCountBudget
{
    bool valid = false;
    long double count = 0, product = 0, gamma = 0;
};

inline NativeRZHistoryCountBudget EvaluateNativeRZHistoryCountBudget (
    NativeRZHistoryAcceptanceInputs const& input)
{
    NativeRZHistoryCountBudget out;
    if (input.material_records < 0 || input.radiation_before_records < 0 ||
        input.radiation_after_records < 0 || input.cells <= 0 ||
        input.subintervals <= 0 || input.ranks <= 0 ||
        input.radiation_before_records > std::numeric_limits<amrex::Long>::max() -
                                        input.radiation_after_records) { return out; }
    // Keep the established operation order and double-precision epsilon.
    out.count = 128.L * input.material_records +
        64.L * (input.radiation_before_records + input.radiation_after_records) +
        128.L * input.subintervals * input.cells + 8.L * input.ranks;
    out.product = out.count * std::numeric_limits<amrex::Real>::epsilon();
    out.valid = std::isfinite(out.product) && out.product >= 0 && out.product < .5L;
    if (out.valid) { out.gamma = out.product / (1 - out.product); }
    return out;
}

struct NativeRZHistoryAcceptanceLimits
{
    bool valid = false;
    NativeRZHistoryCountBudget arithmetic;
    std::array<long double, 3> endpoint{}, material_report{};
    long double heat_report = 0;
};

inline NativeRZHistoryAcceptanceLimits EvaluateNativeRZHistoryAcceptanceBudget (
    NativeRZHistoryAcceptanceInputs const& input)
{
    NativeRZHistoryAcceptanceLimits out;
    out.arithmetic = EvaluateNativeRZHistoryCountBudget(input);
    auto finite_nonnegative = [] (long double x) { return std::isfinite(x) && x >= 0; };
    bool valid = out.arithmetic.valid && finite_nonnegative(input.eos_scale) &&
        finite_nonnegative(input.eos_bound) && finite_nonnegative(input.physical_source_budget) &&
        finite_nonnegative(input.physical_energy_scale) && finite_nonnegative(input.energy_tolerance) &&
        std::isfinite(input.radial_lever) && input.radial_lever > 0;
    auto const gamma = out.arithmetic.gamma;
    for (int d = 0; d < 3; ++d)
    {
        for (auto value : {input.before_operands[d], input.after_operands[d],
             input.material_operands[d], static_cast<long double>(input.step_operands[d]),
             static_cast<long double>(input.step_bounds[d]),
             input.wall_operands[d], input.reported_scale[d], input.accumulated_operands[d]})
        { valid = valid && finite_nonnegative(value); }
        auto const scale = input.before_operands[d] + input.after_operands[d] +
            input.material_operands[d] + input.step_operands[d] +
            (d == 0 ? input.eos_scale : 0) + input.wall_operands[d];
        auto const physical_scale = input.physical_energy_scale *
                                    (d == 2 ? input.radial_lever : 1);
        out.endpoint[d] = input.step_bounds[d] + (d == 0 ? input.eos_bound : 0) +
            gamma / (1 - gamma) * scale + input.energy_tolerance * physical_scale;
        out.material_report[d] = input.step_bounds[d] + gamma / (1 - gamma) *
            (input.material_operands[d] + input.reported_scale[d] + input.accumulated_operands[d]);
        valid = valid && finite_nonnegative(out.endpoint[d]) &&
                        finite_nonnegative(out.material_report[d]);
    }
    out.heat_report = input.step_bounds[0] + input.eos_bound +
        gamma / (1 - gamma) * (input.eos_scale + input.physical_source_budget) +
        input.energy_tolerance * input.physical_source_budget;
    out.valid = valid && finite_nonnegative(out.heat_report);
    return out;
}

struct NativeRZHistoryAcceptanceWitness
{
    bool valid = false;
    NativeRZHistoryAcceptanceInputs original;
    std::array<long double, 3> residual{}, work_report_residual{}, carry_report_residual{};
    long double heat_report_residual = 0;
    // Immutable D-cell records/absolute operands survive every photon-only
    // coarsening outcome. Native capture supplies downward-rounded operands.
    amrex::Long minimum_after_records = 0;
    std::array<long double, 3> minimum_after_operands{};
};

inline NativeRZHistoryAcceptanceLimits NativeRZHistoryAcceptanceLowerBound (
    NativeRZHistoryAcceptanceWitness const& witness)
{
    if (!witness.valid || witness.minimum_after_records < 0 ||
        witness.minimum_after_records > witness.original.radiation_after_records) { return {}; }
    auto const old = EvaluateNativeRZHistoryAcceptanceBudget(witness.original);
    if (!old.valid) { return {}; }
    auto floor = witness.original;
    floor.radiation_after_records = witness.minimum_after_records;
    for (int d = 0; d < 3; ++d)
    {
        auto const operand = witness.minimum_after_operands[d];
        if (!std::isfinite(operand) || operand < 0 || operand > floor.after_operands[d] ||
            !std::isfinite(witness.residual[d]) || !std::isfinite(witness.work_report_residual[d]) ||
            !std::isfinite(witness.carry_report_residual[d])) { return {}; }
        floor.after_operands[d] = operand;
    }
    if (!std::isfinite(witness.heat_report_residual)) { return {}; }
    auto lower = EvaluateNativeRZHistoryAcceptanceBudget(floor);
    if (!lower.valid) { return {}; }
    // gamma/(1-gamma)=product/(1-2*product) is monotone ONLY in this
    // validated product<1/2 domain. Fixed reporting checks must fit the FLOOR.
    for (int d = 0; d < 3; ++d)
    {
        if (std::abs(witness.residual[d]) > old.endpoint[d] ||
            std::abs(witness.work_report_residual[d]) > lower.material_report[d] ||
            std::abs(witness.carry_report_residual[d]) > lower.material_report[d]) { return {}; }
        if (lower.endpoint[d] > 0)
        { lower.endpoint[d] = std::nextafter(lower.endpoint[d], 0.L); }
    }
    if (std::abs(witness.heat_report_residual) > lower.heat_report) { return {}; }
    return lower;
}
} // namespace warpx::radiation
#endif
