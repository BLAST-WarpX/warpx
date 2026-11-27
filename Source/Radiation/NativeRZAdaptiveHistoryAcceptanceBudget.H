/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_NATIVE_RZ_ADAPTIVE_HISTORY_ACCEPTANCE_BUDGET_H_
#define WARPX_NATIVE_RZ_ADAPTIVE_HISTORY_ACCEPTANCE_BUDGET_H_

#include "AdaptivePacketDiffusionTransition.H"
#include "NativeRZHistoryAcceptanceBudget.H"

#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace warpx::radiation
{
/** Independently measured native material and physical caloric endpoint.
 * Counts/currencies are GLOBAL once; frozen positions/schema/IDs are required.
 * This is additional owner readback, not another finite material source. */
struct NativeRZAdaptiveHistoryOwnerWitness
{
    bool valid = false;
    amrex::Long material_records = 0, caloric_cells = 0;
    int ranks = 0;
    std::array<long double, 3> material_change{}, material_carry_change{}, material_operands{};
    long double eos_heat = 0, eos_scale = 0, eos_roundoff_bound = 0;
    std::array<long double, 3> work_report_residual{}, carry_report_residual{};
    long double heat_report_residual = 0;
};

struct NativeRZAdaptiveHistoryAcceptanceLimits
{
    bool valid = false;
    NativeRZHistoryAcceptanceLimits inner;
    long double extra_operations = 0, extra_gamma = 0;
    std::array<long double, 3> representation_operands{}, representation_roundoff{};
    std::array<long double, 3> endpoint{};
};

/** NEW outer arithmetic calculator, not a change to the inner fixed history.
 * Consume GLOBAL converter witnesses ONCE; physical currency map [0,3,2].
 * Keep the EXACT original inner physical/source/report/heat law. Only declared
 * existing conversion arithmetic plus explicitly counted additional readback
 * and scalar composition arithmetic is added. Carry operands condition ONLY
 * arithmetic, never physical tolerance; no wall slot or measured-defect budget.
 * Every input term is immutable before outer acceptance. In-progress clocks
 * and photon-coarsening floors are unrelated to this calculator.
 */
[[nodiscard]] inline NativeRZAdaptiveHistoryAcceptanceLimits
EvaluateNativeRZAdaptiveHistoryAcceptanceBudget (
    NativeRZHistoryAcceptanceWitness const& history,
    AdaptiveRepresentationConversionWitness const& entry,
    AdaptiveRepresentationConversionWitness const& endpoint,
    NativeRZAdaptiveHistoryOwnerWitness const& owners)
{
    NativeRZAdaptiveHistoryAcceptanceLimits out;
    auto upward = [] (long double value)
    { return value == 0 ? 0.L : std::nextafter(value,
        std::numeric_limits<long double>::infinity()); };
    out.inner = EvaluateNativeRZHistoryAcceptanceBudget(history.original);
    bool valid = history.valid && out.inner.valid && entry.valid && endpoint.valid && owners.valid &&
        owners.material_records == history.original.material_records &&
        owners.material_records >= 0 && owners.caloric_cells == history.original.cells &&
        owners.caloric_cells > 0 && owners.ranks == history.original.ranks && owners.ranks > 0 &&
        std::isfinite(owners.eos_heat) && std::isfinite(owners.eos_scale) &&
        owners.eos_scale >= std::abs(owners.eos_heat) &&
        std::isfinite(owners.eos_roundoff_bound) && owners.eos_roundoff_bound >= 0;
    for (int d = 0; d < 3; ++d)
    { valid = valid && std::isfinite(history.residual[d]) &&
        std::isfinite(history.work_report_residual[d]) &&
        std::isfinite(history.carry_report_residual[d]); }
    valid = valid && std::isfinite(history.heat_report_residual);
    // 32 scalar operations/currency covers combining six stored participants
    // at each boundary, subtracting the two GLOBAL transfers and endpoint
    // native work/carry/heat, and summing nonnegative arithmetic operands.
    // Independent repeated material readback <=128/native record (the SAME
    // paired finite-KE and physical Cartesian/carry moment formulas); <=32/cell
    // and32/rank for the repeated certified caloric readback/positive sums.
    // Existing caloric storage bounds are retained by inner, never invented.
    out.extra_operations = 3*32 + 128.L*owners.material_records +
        32.L*owners.caloric_cells + 32.L*owners.ranks;
    for (int d = 0; d < 3; ++d)
    { valid = valid && std::isfinite(owners.material_change[d]) &&
        std::isfinite(owners.material_carry_change[d]) &&
        std::isfinite(owners.material_operands[d]) && owners.material_operands[d] >=
            std::abs(owners.material_change[d])+std::abs(owners.material_carry_change[d]) &&
        std::isfinite(owners.work_report_residual[d]) &&
        std::isfinite(owners.carry_report_residual[d]) &&
        std::abs(owners.work_report_residual[d]) <= out.inner.material_report[d] &&
        std::abs(owners.carry_report_residual[d]) <= out.inner.material_report[d]; }
    valid = valid && std::isfinite(owners.heat_report_residual) &&
        std::abs(owners.heat_report_residual) <= out.inner.heat_report;
    for (auto const* conversion : {&entry, &endpoint})
    {
        for (int n = 0; n < 4; ++n)
        {
            valid = valid && std::isfinite(conversion->carried_residual[n]) &&
                std::isfinite(conversion->declared_conversion_roundoff_bound[n]) &&
                conversion->declared_conversion_roundoff_bound[n] >= 0;
        }
        valid = valid && conversion->carried_residual[1] == 0 &&
            conversion->declared_conversion_roundoff_bound[1] == 0;
        for (auto const* state : {&conversion->before, &conversion->after})
        {
            valid = valid && state->valid && state->packet_records >= 0 &&
                state->field_records >= 0 && state->carry_records >= 0 &&
                state->ranks == history.original.ranks && state->ranks > 0 &&
                state->field_records == history.original.cells &&
                state->carry_records == history.original.cells &&
                std::isfinite(state->readback_operations) && state->readback_operations ==
                    64.L*state->packet_records +
                    32.L*(static_cast<long double>(state->field_records)+state->carry_records) +
                    32.L*state->ranks;
            for (auto const* values : {&state->packet, &state->field, &state->carry})
            { for (auto value : *values) { valid = valid && std::isfinite(value); } }
            for (int n = 0; n < 4; ++n)
            {
                for (auto pair : {std::pair{state->packet[n], state->packet_operands[n]},
                                  std::pair{state->field[n], state->field_operands[n]},
                                  std::pair{state->carry[n], state->carry_operands[n]}})
                { valid = valid && std::isfinite(pair.second) && pair.second >= std::abs(pair.first); }
            }
            valid = valid && state->packet[1] == 0 && state->field[1] == 0 && state->carry[1] == 0;
            out.extra_operations += state->readback_operations;
        }
    }
    auto const product = out.extra_operations*std::numeric_limits<amrex::Real>::epsilon();
    valid = valid && std::isfinite(product) && product >= 0 && product < .25L;
    if (!valid) { return out; }
    out.extra_gamma = upward(product/(1-product));
    std::array<int, 3> const component{0,3,2};
    for (int d = 0; d < 3; ++d)
    {
        auto const c = component[d];
        for (auto const* conversion : {&entry, &endpoint})
        {
            auto const bound = conversion->declared_conversion_roundoff_bound[c];
            valid = valid && std::isfinite(bound) && bound >= 0 &&
                std::isfinite(conversion->carried_residual[c]);
            out.representation_roundoff[d] = upward(out.representation_roundoff[d] + bound);
            for (auto const* state : {&conversion->before, &conversion->after})
            {
                for (auto value : {state->packet_operands[c], state->field_operands[c],
                                   state->carry_operands[c]})
                {
                    valid = valid && std::isfinite(value) && value >= 0;
                    out.representation_operands[d] = upward(out.representation_operands[d] + value);
                }
            }
        }
        // The inner uncancelled native material/EOS operands also bound the
        // outer subtraction's operands, without counting material work twice.
        auto composition_scale = out.representation_operands[d];
        for (auto value : {history.original.material_operands[d],
             owners.material_operands[d], history.original.accumulated_operands[d],
             d == 0 ? history.original.eos_scale : 0.L, d == 0 ? owners.eos_scale : 0.L})
        { composition_scale = upward(composition_scale + value); }
        // gamma is already an upper endpoint. Round its positive denominator
        // DOWN before dividing, then round the quotient UP: a nearest-rounded
        // denominator alone would not be an interval enclosure.
        auto const gamma_denominator = std::nextafter(1-out.extra_gamma, 0.L);
        auto const arithmetic_factor = upward(out.extra_gamma/gamma_denominator);
        out.representation_roundoff[d] = upward(out.representation_roundoff[d] +
            upward(arithmetic_factor*composition_scale));
        out.endpoint[d] = upward(out.inner.endpoint[d] + out.representation_roundoff[d]);
        valid = valid && std::isfinite(out.endpoint[d]) && out.endpoint[d] >= 0;
    }
    out.valid = valid;
    return out;
}
} // namespace warpx::radiation
#endif
