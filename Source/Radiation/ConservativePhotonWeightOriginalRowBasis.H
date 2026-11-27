/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CONSERVATIVE_PHOTON_WEIGHT_ORIGINAL_ROW_BASIS_H_
#define WARPX_CONSERVATIVE_PHOTON_WEIGHT_ORIGINAL_ROW_BASIS_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace warpx::radiation
{
enum class PhotonWeightDirectionMethod : std::uint64_t
{
    Legacy = 0,
    RankRevealingOriginalRows = 1
};

struct PhotonWeightOriginalRowBasis
{
    bool valid = false;
    // Indices into the UNTOUCHED original feature rows, in selection order.
    // No transformed coefficient or numerical rank residual is exported.
    std::vector<std::size_t> rows;
};

/** Private host-only, bounded row selector: column-pivoted Householder QR of
 * the transpose of the original row-max-scaled features. QR is ONLY a discrete
 * selection device. Its temporary coefficients never enter a null direction,
 * endpoint, moment account, certificate or allowance. The caller must reread
 * selected ORIGINAL rows and use its unchanged Gaussian/certificate/full-row
 * BOTH-endpoint guards before any draw. A mistaken numerical rank decision can
 * therefore only change which candidate basis is checked, not grant admission.
 *
 * For F<=13 original features and N<=min(F+1,14) rays, at most min(F,N)
 * reflectors are formed. A conservative work bound is
 * 16*F*N*(min(F,N)+1)+8*N*(min(F,N)+1) scalar operations, distinct from the
 * moment-affecting solve budget: only integer row indices cross this boundary.
 * Fixed original-row tie order and the ORIGINAL F,N rank threshold are used.
 * This is not a condition certificate, a complete rank proof or a population cap.
 */
template <typename Record>
PhotonWeightOriginalRowBasis SelectPhotonWeightOriginalRowBasis (
    std::vector<Record> const& records, std::vector<std::size_t> const& block)
{
    PhotonWeightOriginalRowBasis out;
    if (records.empty() || block.empty()) { return out; }
    auto const features = records.front().features.size();
    auto const columns = block.size();
    if (features == 0 || features > 13 || columns > features + 1 || columns > 14)
    { return out; }
    std::vector<std::vector<long double>> transpose(columns,
        std::vector<long double>(features));
    for (std::size_t r = 0; r < features; ++r)
    {
        long double scale = 0;
        for (std::size_t c = 0; c < columns; ++c)
        {
            if (block[c] >= records.size() || records[block[c]].features.size() != features ||
                std::find(block.begin(), block.begin() + c, block[c]) != block.begin() + c)
            { return out; }
            auto const value = records[block[c]].features[r];
            if (!std::isfinite(value)) { return out; }
            transpose[c][r] = value;
            scale = std::max(scale, std::abs(transpose[c][r]));
        }
        if (scale > 0)
        { for (auto& row : transpose) { row[r] /= scale; } }
    }
    std::vector<std::size_t> original(features);
    std::iota(original.begin(), original.end(), 0);
    auto const rank_budget = 32.L * static_cast<long double>((features + 1) * (columns + 1)) *
                             std::numeric_limits<double>::epsilon();
    for (std::size_t k = 0; k < std::min(features, columns); ++k)
    {
        auto pivot = k;
        long double largest = -1;
        for (std::size_t j = k; j < features; ++j)
        {
            long double norm_squared = 0;
            for (std::size_t i = k; i < columns; ++i)
            { norm_squared += transpose[i][j] * transpose[i][j]; }
            if (!std::isfinite(norm_squared)) { return {}; }
            if (norm_squared > largest ||
                (norm_squared == largest && original[j] < original[pivot]))
            { largest = norm_squared; pivot = j; }
        }
        auto const magnitude = std::sqrt(largest);
        if (!(magnitude > rank_budget)) { break; }
        for (auto& row : transpose) { std::swap(row[k], row[pivot]); }
        std::swap(original[k], original[pivot]);
        std::vector<long double> reflector(columns - k);
        for (std::size_t i = k; i < columns; ++i) { reflector[i - k] = transpose[i][k]; }
        reflector.front() += std::copysign(magnitude, reflector.front());
        long double norm_squared = 0;
        for (auto value : reflector) { norm_squared += value * value; }
        if (!std::isfinite(norm_squared) || !(norm_squared > 0)) { return {}; }
        auto const factor = 2 / norm_squared;
        for (std::size_t j = k; j < features; ++j)
        {
            long double projection = 0;
            for (std::size_t i = k; i < columns; ++i)
            { projection += reflector[i - k] * transpose[i][j]; }
            projection *= factor;
            for (std::size_t i = k; i < columns; ++i)
            {
                transpose[i][j] -= reflector[i - k] * projection;
                if (!std::isfinite(transpose[i][j])) { return {}; }
            }
        }
        out.rows.push_back(original[k]);
    }
    out.valid = true;
    return out;
}
} // namespace warpx::radiation
#endif
