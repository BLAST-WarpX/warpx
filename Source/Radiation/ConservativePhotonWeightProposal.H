/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CONSERVATIVE_PHOTON_WEIGHT_PROPOSAL_H_
#define WARPX_CONSERVATIVE_PHOTON_WEIGHT_PROPOSAL_H_

#include "ConservativePhotonWeightOriginalRowBasis.H"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace warpx::radiation
{
/** Host-only, double-precision proposal primitive. It has no native owner,
 * geometry, ID allocator, carry, global RNG, field, or publication access.
 * Features are per unit statistical weight; feature 0 must be exactly one.
 * A native adapter must supply physical E [J], Cartesian cP [J], cLz [J m]
 * from actual positions/momenta, and any required pressure/census features.
 * In particular cLz is not the conjugate RZ rate velocity or cLz/rbar.
 * Surviving native phase-space, spectral, key and carry state must remain
 * unchanged. Native integration, feature building and clock policy are NOT
 * implemented or activated here.
 */
struct PhotonWeightRecord
{
    // Caller-supplied stable draw identity: native fallback or explicit key.
    // This is NOT a provisional placeholder native ID.
    std::pair<std::uint64_t, std::uint64_t> key{};
    double weight = 0;
    std::vector<double> features;
};

struct PhotonWeightProposalOptions
{
    std::size_t target = 1;
    std::uint64_t seed = 0;
    std::uint64_t outer_interval = 0;
    std::uint64_t canonical_bucket = 0;
    // Private original-row preparation; Legacy preserves the established path.
    // The alternative is bounded to <=13 features, <=14 rays and one attempt.
    PhotonWeightDirectionMethod direction_method = PhotonWeightDirectionMethod::Legacy;
    // Optional restrictive budget from an UNCHANGED caller acceptance bound.
    // It is checked against the whole outcome envelope BEFORE any draw.
    std::vector<long double> unchanged_moment_headroom{};
};

struct PhotonWeightProposal
{
    bool valid = false;
    bool applied = false;
    bool target_met = false;
    std::string reason;
    // In ORIGINAL caller order, including zero rows. A stop before the first
    // draw is an exact no-op. Later stops retain the accepted prefix, because
    // outcome-dependent rollback would bias expected original-ray weights.
    std::vector<double> weights;
    std::vector<long double> moment_change;
    std::vector<long double> arithmetic_bound;
    // All-outcome upper bound for finite-precision/53-bit-draw deviation from
    // ideal per-original-ray expectation. No native selection/fallback allowed.
    long double weight_expectation_roundoff_bound = 0;
    std::size_t survivors = 0;
    std::size_t eliminations = 0;
};

namespace photon_weight_detail
{
inline std::uint64_t Mix (std::uint64_t value) noexcept
{
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

struct CounterDraw
{
    std::uint64_t state = 0;
    double operator() (std::size_t elimination) const noexcept
    {
        auto const word = Mix(state ^ Mix(static_cast<std::uint64_t>(elimination)));
        return static_cast<double>(word >> 11U) * 0x1.0p-53;
    }
};

// Row-scaled, fixed-column Gaussian elimination. Column order comes only
// from canonical tagged keys; largest-row pivots have deterministic tie order.
// The rank threshold is a numerical heuristic, NOT a nullsolve condition
// bound. BOTH represented endpoints must pass explicit moment guards before
// any draw, including when this heuristic omits a near-dependent constraint.
struct NullDirectionPivots
{
    // Original scaled constraint rows and canonical active columns retained by
    // the unchanged Gaussian preparation. Optional diagnostic/certificate only.
    std::vector<std::size_t> rows, columns;
};

inline std::vector<long double> NullDirection (
    std::vector<PhotonWeightRecord> const& records, std::vector<std::size_t> const& block,
    NullDirectionPivots* retained = nullptr,
    std::vector<std::size_t> const* selected_original_rows = nullptr)
{
    auto const original_rows = records.front().features.size();
    auto const rows = selected_original_rows ? selected_original_rows->size() : original_rows;
    auto const columns = block.size();
    if (rows == 0) { return {}; }
    std::vector<std::size_t> row_indices;
    if (retained)
    {
        *retained = {};
        row_indices.resize(rows);
        if (selected_original_rows) { row_indices = *selected_original_rows; }
        else { std::iota(row_indices.begin(), row_indices.end(), 0); }
    }
    std::vector<std::vector<long double>> matrix(rows, std::vector<long double>(columns));
    for (std::size_t r = 0; r < rows; ++r)
    {
        auto const original = selected_original_rows ? (*selected_original_rows)[r] : r;
        if (original >= original_rows || (selected_original_rows &&
            std::find(selected_original_rows->begin(), selected_original_rows->begin() + r,
                      original) != selected_original_rows->begin() + r)) { return {}; }
        long double scale = 0;
        for (std::size_t c = 0; c < columns; ++c)
        {
            matrix[r][c] = records[block[c]].features[original];
            scale = std::max(scale, std::abs(matrix[r][c]));
        }
        if (scale > 0)
        {
            for (auto& value : matrix[r]) { value /= scale; }
        }
    }
    auto const rank_budget = 32.L * static_cast<long double>((original_rows + 1) * (columns + 1)) *
                             std::numeric_limits<double>::epsilon();
    std::vector<std::size_t> pivots;
    for (std::size_t c = 0; c < columns && pivots.size() < rows; ++c)
    {
        auto const next = pivots.size();
        auto pivot = next;
        for (std::size_t r = next + 1; r < rows; ++r)
        {
            if (std::abs(matrix[r][c]) > std::abs(matrix[pivot][c])) { pivot = r; }
        }
        if (std::abs(matrix[pivot][c]) <= rank_budget) { continue; }
        std::swap(matrix[next], matrix[pivot]);
        if (retained) { std::swap(row_indices[next], row_indices[pivot]); }
        auto const diagonal = matrix[next][c];
        for (std::size_t j = c; j < columns; ++j) { matrix[next][j] /= diagonal; }
        for (std::size_t r = 0; r < rows; ++r)
        {
            if (r == next) { continue; }
            auto const factor = matrix[r][c];
            for (std::size_t j = c; j < columns; ++j)
            { matrix[r][j] -= factor * matrix[next][j]; }
        }
        pivots.push_back(c);
        if (retained)
        {
            retained->rows.push_back(row_indices[next]);
            retained->columns.push_back(c);
        }
    }
    auto free_column = columns;
    for (std::size_t c = 0; c < columns; ++c)
    {
        if (std::find(pivots.begin(), pivots.end(), c) == pivots.end())
        { free_column = c; break; }
    }
    if (free_column == columns) { return {}; }
    std::vector<long double> direction(columns, 0);
    direction[free_column] = 1;
    for (std::size_t r = 0; r < pivots.size(); ++r)
    { direction[pivots[r]] = -matrix[r][free_column]; }
    long double magnitude = 0;
    for (auto value : direction) { magnitude = std::max(magnitude, std::abs(value)); }
    if (!std::isfinite(magnitude) || !(magnitude > 0)) { return {}; }
    for (auto& value : direction) { value /= magnitude; }
    return direction;
}
} // namespace photon_weight_detail

/** For Ah=0 choose w-a+*h with probability a-/(a++a-), otherwise w+a-*h.
 * Both endpoints are nonnegative and eliminate at least one existing row;
 * each original row keeps its ideal conditional expected weight. BOTH stored
 * endpoints pass the same fixed component guards BEFORE drawing. Accumulated
 * all-outcome bounds prove each accepted prefix's drift; there is no selected-
 * outcome final rejection. A later rank/numeric stop retains that prefix.
 * Floating storage/counter discretization bias has an explicit separate bound.
 * No measured defect supplies an allowance and no user tolerance is exposed.
 * A native adapter must preflight its unchanged physical budget against the
 * returned all-outcome bounds. It cannot conditionally drop an outcome just
 * because target_met is false or a final native remeasurement fails.
 * The draw overload supports independent finite-outcome test oracles.
 */
template <typename Draw>
PhotonWeightProposal ProposeConservativePhotonWeights (
    std::vector<PhotonWeightRecord> const& records,
    PhotonWeightProposalOptions const& options, Draw draw)
{
    PhotonWeightProposal out;
    for (auto const& record : records)
    {
        out.weights.push_back(record.weight);
        out.survivors += record.weight > 0;
    }
    if (options.target == 0) { out.reason = "zero population target"; return out; }
    if (options.direction_method != PhotonWeightDirectionMethod::Legacy &&
        options.direction_method != PhotonWeightDirectionMethod::RankRevealingOriginalRows)
    { out.reason = "invalid direction method"; return out; }
    if (records.empty()) { out.valid = true; return out; }
    auto const features = records.front().features.size();
    if (features == 0 || features > 32)
    { out.reason = "unsupported feature count"; return out; }
    if (options.direction_method != PhotonWeightDirectionMethod::Legacy &&
        (options.direction_method != PhotonWeightDirectionMethod::RankRevealingOriginalRows ||
         features > 13 || out.survivors > features + 1 || out.survivors > 14 ||
         (out.survivors > options.target && out.survivors - options.target != 1)))
    { out.reason = "unsupported original-row basis scope"; return out; }
    std::vector<std::size_t> canonical(records.size());
    std::iota(canonical.begin(), canonical.end(), 0);
    std::sort(canonical.begin(), canonical.end(), [&] (auto a, auto b)
    { return records[a].key < records[b].key; });
    for (std::size_t p = 0; p < canonical.size(); ++p)
    {
        auto const& record = records[canonical[p]];
        if (!std::isfinite(record.weight) || record.weight < 0 || record.key.first > 1 ||
            record.key.second == 0 || record.features.size() != features ||
            record.features[0] != 1 ||
            (p > 0 && record.key == records[canonical[p - 1]].key) ||
            std::any_of(record.features.begin(), record.features.end(),
                        [] (auto value) { return !std::isfinite(value); }))
        { out.reason = "invalid weight, key or feature schema"; return out; }
    }
    out.valid = true;
    out.target_met = out.survivors <= options.target;
    if (out.target_met) { return out; }
    auto const maximum_steps = out.survivors - 1;
    auto const step_operations = 64.L * static_cast<long double>((features + 1) *
                                                               (features + 1)) + 32;
    auto const step_product = step_operations * std::numeric_limits<double>::epsilon();
    auto const step_gamma = step_product / (1 - step_product);
    if (!(step_gamma * maximum_steps < .25L))
    { out.reason = "arithmetic operation budget"; return out; }
    long double initial_weight = 0;
    std::vector<long double> maximum_feature(features, 0);
    for (auto index : canonical)
    {
        initial_weight += records[index].weight;
        for (std::size_t r = 0; r < features; ++r)
        {
            maximum_feature[r] = std::max(maximum_feature[r],
                std::abs(static_cast<long double>(records[index].features[r])));
        }
    }
    auto const sum_product = 4.L * records.size() *
                             std::numeric_limits<long double>::epsilon();
    auto const initial_upper = initial_weight / (1 - sum_product);
    auto const worst_weight = initial_upper / (1 - step_gamma * maximum_steps);
    if (!std::isfinite(worst_weight))
    { out.reason = "unrepresentable all-outcome weight bound"; return out; }
    std::vector<long double> step_bound(features, 0);
    for (std::size_t r = 0; r < features; ++r)
    { step_bound[r] = step_gamma * maximum_feature[r] * worst_weight; }
    auto const audit_product = 16.L * (records.size() + maximum_steps + 1) *
                               std::numeric_limits<long double>::epsilon();
    auto const audit_gamma = audit_product / (1 - audit_product);
    out.arithmetic_bound.resize(features);
    for (std::size_t r = 0; r < features; ++r)
    {
        out.arithmetic_bound[r] = maximum_steps * step_bound[r] +
            audit_gamma * maximum_feature[r] * (initial_upper + worst_weight);
    }
    out.weight_expectation_roundoff_bound = maximum_steps * worst_weight *
        (8 * std::numeric_limits<double>::epsilon() +
         16 * std::numeric_limits<long double>::epsilon() + 0x1.0p-53L);
    if (!options.unchanged_moment_headroom.empty())
    {
        if (options.unchanged_moment_headroom.size() != features ||
            std::any_of(options.unchanged_moment_headroom.begin(),
                        options.unchanged_moment_headroom.end(),
                        [] (auto value) { return !std::isfinite(value) || value < 0; }))
        { out.valid = false; out.reason = "invalid unchanged moment headroom"; return out; }
        for (std::size_t r = 0; r < features; ++r)
        {
            if (out.arithmetic_bound[r] > options.unchanged_moment_headroom[r])
            { out.reason = "whole-outcome envelope exceeds unchanged headroom"; return out; }
        }
    }
    auto staged = out.weights;
    std::size_t steps = 0;
    for (;;)
    {
        std::vector<std::size_t> active;
        for (auto index : canonical)
        { if (staged[index] > 0) { active.push_back(index); } }
        if (active.size() <= options.target) { break; }
        active.resize(std::min(active.size(), features + 1));
        PhotonWeightOriginalRowBasis basis;
        if (options.direction_method == PhotonWeightDirectionMethod::RankRevealingOriginalRows)
        {
            basis = SelectPhotonWeightOriginalRowBasis(records, active);
            if (!basis.valid) { out.reason = "invalid original-row basis"; break; }
        }
        auto const direction = photon_weight_detail::NullDirection(records, active, nullptr,
            options.direction_method == PhotonWeightDirectionMethod::Legacy ?
                nullptr : &basis.rows);
        if (direction.empty()) { out.reason = "irreducible feature rank"; break; }
        long double plus = std::numeric_limits<long double>::infinity();
        long double minus = plus;
        std::size_t plus_zero = active.size(), minus_zero = active.size();
        for (std::size_t p = 0; p < active.size(); ++p)
        {
            auto const h = direction[p];
            if (h > 0 && staged[active[p]] / h < plus)
            { plus = staged[active[p]] / h; plus_zero = p; }
            if (h < 0 && -staged[active[p]] / h < minus)
            { minus = -staged[active[p]] / h; minus_zero = p; }
        }
        if (!std::isfinite(plus) || !std::isfinite(minus) || !(plus > 0 && minus > 0) ||
            !std::isfinite(plus + minus))
        { out.reason = "invalid feasible endpoint"; break; }
        std::vector<std::vector<double>> endpoints(2, std::vector<double>(active.size()));
        bool safe_pair = true;
        for (int side = 0; side < 2; ++side)
        {
            auto const displacement = side == 0 ? -plus : minus;
            auto const eliminated = side == 0 ? plus_zero : minus_zero;
            for (std::size_t p = 0; p < active.size(); ++p)
            {
                auto const index = active[p];
                long double const increment = displacement * direction[p];
                auto value = static_cast<long double>(staged[index]) + increment;
                auto const cancellation_bound = 4 * std::numeric_limits<long double>::epsilon() *
                    (std::abs(static_cast<long double>(staged[index])) + std::abs(increment));
                if (p == eliminated || (value < 0 && -value <= cancellation_bound)) { value = 0; }
                endpoints[side][p] = static_cast<double>(value);
                safe_pair = safe_pair && std::isfinite(endpoints[side][p]) &&
                            endpoints[side][p] >= 0;
            }
            if (!safe_pair) { break; }
            for (std::size_t r = 0; r < features; ++r)
            {
                long double difference = 0;
                for (std::size_t p = 0; p < active.size(); ++p)
                {
                    difference += (static_cast<long double>(endpoints[side][p]) -
                                   staged[active[p]]) * records[active[p]].features[r];
                }
                safe_pair = safe_pair && std::isfinite(difference) &&
                            std::abs(difference) <= step_bound[r];
            }
        }
        if (!safe_pair) { out.reason = "unsafe represented endpoint pair"; break; }
        // Only now can a stochastic choice occur. Rejecting either represented
        // endpoint must stop the pair as a whole, independent of the draw.
        double const uniform = draw(steps);
        if (!(uniform >= 0 && uniform < 1) || !std::isfinite(uniform))
        { out.valid = false; out.reason = "invalid counter draw"; return out; }
        auto const side = uniform < minus / (plus + minus) ? 0 : 1;
        for (std::size_t p = 0; p < active.size(); ++p)
        { staged[active[p]] = endpoints[side][p]; }
        ++steps;
    }
    if (steps == 0) { return out; }
    std::vector<long double> change(features, 0);
    for (auto index : canonical)
    {
        for (std::size_t r = 0; r < features; ++r)
        {
            auto const before = static_cast<long double>(records[index].weight) *
                                records[index].features[r];
            auto const after = static_cast<long double>(staged[index]) *
                               records[index].features[r];
            change[r] += after - before;
        }
    }
    // Each accepted local delta was checked for BOTH outcomes. The triangle
    // inequality plus long-double audit arithmetic bounds every possible
    // prefix, without a conditional selected-outcome final fallback.
    out.weights = std::move(staged);
    out.moment_change = std::move(change);
    out.survivors = static_cast<std::size_t>(std::count_if(
        out.weights.begin(), out.weights.end(), [] (auto value) { return value > 0; }));
    out.eliminations = steps;
    out.applied = true;
    out.target_met = out.survivors <= options.target;
    return out;
}

inline PhotonWeightProposal ProposeConservativePhotonWeights (
    std::vector<PhotonWeightRecord> const& records, PhotonWeightProposalOptions const& options)
{
    using photon_weight_detail::Mix;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> keys;
    for (auto const& record : records) { keys.push_back(record.key); }
    std::sort(keys.begin(), keys.end());
    auto state = Mix(options.seed ^ UINT64_C(0x706f70756c617469));
    state = Mix(state ^ Mix(options.outer_interval ^ UINT64_C(0xa14587125bf62687)));
    state = Mix(state ^ Mix(options.canonical_bucket ^ UINT64_C(0x87f4c9b62704a145)));
    for (auto const& key : keys)
    {
        state = Mix(state ^ Mix(key.first));
        state = Mix(state ^ Mix(key.second));
    }
    return ProposeConservativePhotonWeights(
        records, options, photon_weight_detail::CounterDraw{state});
}
} // namespace warpx::radiation

#endif
