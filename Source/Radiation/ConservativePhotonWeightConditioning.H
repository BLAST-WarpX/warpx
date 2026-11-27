/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CONSERVATIVE_PHOTON_WEIGHT_CONDITIONING_H_
#define WARPX_CONSERVATIVE_PHOTON_WEIGHT_CONDITIONING_H_

#include "ConservativePhotonWeightProposal.H"

namespace warpx::radiation
{
struct PhotonWeightDirectionCertificate
{
    bool valid = false;
    bool admitted = false;
    long double inverse_residual_upper = 0;
    long double inverse_norm_upper = 0;
    long double condition_upper = 0;
    long double gamma_upper = 0;
    long double half_resolution_lower = 0;
};

namespace photon_condition_detail
{
using Matrix = std::vector<std::vector<long double>>;

inline long double Up (long double value)
{ return std::nextafter(value, std::numeric_limits<long double>::infinity()); }
inline long double Down (long double value)
{ return std::nextafter(value, -std::numeric_limits<long double>::infinity()); }
inline long double AddUp (long double a, long double b) { return Up(a + b); }
inline long double MultiplyUp (long double a, long double b) { return Up(a * b); }
inline long double DivideUp (long double a, long double b) { return Up(a / b); }

inline long double NormUpper (Matrix const& matrix)
{
    long double result = 0;
    for (auto const& row : matrix)
    {
        long double sum = 0;
        for (auto value : row) { sum = AddUp(sum, std::abs(value)); }
        result = std::max(result, sum);
    }
    return result;
}

// An approximate inverse is only a candidate witness. Its construction does
// NOT grant admission; the independent outward residual below must prove it.
inline Matrix ApproximateInverse (Matrix matrix)
{
    auto const size = matrix.size();
    Matrix inverse(size, std::vector<long double>(size));
    for (std::size_t p = 0; p < size; ++p) { inverse[p][p] = 1; }
    for (std::size_t c = 0; c < size; ++c)
    {
        auto pivot = c;
        for (std::size_t r = c + 1; r < size; ++r)
        { if (std::abs(matrix[r][c]) > std::abs(matrix[pivot][c])) { pivot = r; } }
        auto const diagonal = matrix[pivot][c];
        if (!std::isfinite(diagonal) || diagonal == 0) { return {}; }
        std::swap(matrix[c], matrix[pivot]);
        std::swap(inverse[c], inverse[pivot]);
        for (std::size_t j = 0; j < size; ++j)
        {
            matrix[c][j] /= diagonal;
            inverse[c][j] /= diagonal;
        }
        for (std::size_t r = 0; r < size; ++r)
        {
            if (r == c) { continue; }
            auto const factor = matrix[r][c];
            for (std::size_t j = 0; j < size; ++j)
            {
                matrix[r][j] -= factor * matrix[c][j];
                inverse[r][j] -= factor * inverse[c][j];
            }
        }
    }
    return inverse;
}

// For EACH dot product, enclose every represented product and partial sum by
// nextafter in both directions. This verifies ||I-RB||_inf, not a measured
// selected endpoint residual. IEEE basic-operation rounding is assumed; no
// fast-math, reassociation, global rounding-mode change or observed tolerance.
inline long double InverseResidualUpper (Matrix const& inverse, Matrix const& system)
{
    long double residual = 0;
    for (std::size_t i = 0; i < system.size(); ++i)
    {
        long double row = 0;
        for (std::size_t j = 0; j < system.size(); ++j)
        {
            long double low = 0, high = 0;
            for (std::size_t p = 0; p < system.size(); ++p)
            {
                auto const product = inverse[i][p] * system[p][j];
                low = Down(low + Down(product));
                high = Up(high + Up(product));
            }
            auto const identity = static_cast<long double>(i == j);
            auto const below = Down(identity - high), above = Up(identity - low);
            if (!std::isfinite(below) || !std::isfinite(above))
            { return std::numeric_limits<long double>::infinity(); }
            row = AddUp(row, std::max(std::abs(below), std::abs(above)));
        }
        residual = std::max(residual, row);
    }
    return residual;
}
} // namespace photon_condition_detail

/** Optional one-attempt pre-draw resolution admission, NOT a new moment
 * tolerance or a universal per-ray/pathwise continuation guarantee.
 * Rebuild ONLY the actual scaled retained system selected by the unchanged
 * direction preparation. A verified inverse residual nu<1 proves
 * ||B^-1||_inf <= ||R||_inf/(1-nu). All certificate bounds round outward.
 * gamma uses exactly the existing pure-core operation envelope and binary64
 * epsilon. Require kappa_upper*gamma <= sqrt(gamma): the declared retained
 * equation resolution loses at most half of this operation-budget precision.
 * The whole pair stays unchanged on failure, BEFORE any counter draw. It
 * neither changes a direction, rank threshold, endpoint nor physical guard.
 */
inline PhotonWeightDirectionCertificate CertifyPhotonWeightDirection (
    std::vector<PhotonWeightRecord> const& records, std::vector<std::size_t> const& active,
    photon_weight_detail::NullDirectionPivots const& retained)
{
    using namespace photon_condition_detail;
    PhotonWeightDirectionCertificate out;
    if (records.empty() || active.empty() || retained.rows.empty() ||
        retained.rows.size() != retained.columns.size() ||
        retained.columns.size() >= active.size()) { return out; }
    auto const features = records.front().features.size();
    if (features == 0 || features > 32 || active.size() > features + 1) { return out; }
    Matrix scaled(features, std::vector<long double>(active.size()));
    for (std::size_t r = 0; r < features; ++r)
    {
        long double scale = 0;
        for (std::size_t c = 0; c < active.size(); ++c)
        {
            if (active[c] >= records.size() || records[active[c]].features.size() != features)
            { return out; }
            auto const value = records[active[c]].features[r];
            if (!std::isfinite(value)) { return out; }
            scaled[r][c] = value;
            scale = std::max(scale, std::abs(scaled[r][c]));
        }
        if (scale > 0) { for (auto& value : scaled[r]) { value /= scale; } }
    }
    auto const size = retained.rows.size();
    Matrix system(size, std::vector<long double>(size));
    for (std::size_t i = 0; i < size; ++i)
    for (std::size_t j = 0; j < size; ++j)
    {
        if (retained.rows[i] >= features || retained.columns[j] >= active.size()) { return out; }
        system[i][j] = scaled[retained.rows[i]][retained.columns[j]];
    }
    auto const inverse = ApproximateInverse(system);
    if (inverse.empty()) { return out; }
    out.inverse_residual_upper = InverseResidualUpper(inverse, system);
    if (!(out.inverse_residual_upper >= 0 && out.inverse_residual_upper < 1)) { return out; }
    auto const margin_lower = Down(1 - out.inverse_residual_upper);
    if (!(margin_lower > 0)) { return out; }
    out.inverse_norm_upper = DivideUp(NormUpper(inverse), margin_lower);
    out.condition_upper = MultiplyUp(out.inverse_norm_upper, NormUpper(system));
    auto const operations = 64.L * static_cast<long double>((features + 1) * (features + 1)) + 32;
    auto const product = operations * std::numeric_limits<double>::epsilon();
    if (!(product > 0 && product < .25L)) { return out; }
    out.gamma_upper = DivideUp(product, Down(1 - product));
    auto const gamma_lower = Down(product / Up(1 - product));
    out.half_resolution_lower = Down(std::sqrt(gamma_lower));
    out.valid = std::isfinite(out.condition_upper) && std::isfinite(out.gamma_upper) &&
        out.half_resolution_lower > 0;
    out.admitted = out.valid && MultiplyUp(out.condition_upper, out.gamma_upper) <=
                                  out.half_resolution_lower;
    return out;
}
} // namespace warpx::radiation
#endif
