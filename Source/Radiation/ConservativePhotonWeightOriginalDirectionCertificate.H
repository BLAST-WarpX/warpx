/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CONSERVATIVE_PHOTON_WEIGHT_ORIGINAL_DIRECTION_CERTIFICATE_H_
#define WARPX_CONSERVATIVE_PHOTON_WEIGHT_ORIGINAL_DIRECTION_CERTIFICATE_H_

#include "ConservativePhotonWeightConditioning.H"

#include <array>
#include <cfenv>
#include <string>

namespace warpx::radiation {
/** Additive, host-only original-equation certificate. No direction, rank,
 * weights, endpoint, draw, native owner or old admission predicate is changed.
 * The caller must still check ALL original13 BOTH represented endpoints and
 * its unchanged actual-native whole-candidate acceptance law before drawing.
 */
struct PhotonWeightOriginalDirectionCertificate {
    bool valid = false;
    bool admitted = false;
    std::string reason;
    long double inverse_residual_upper = 0;
    long double inverse_norm_upper = 0;
    long double retained_residual_upper = 0;
    long double solve_error_upper = 0;
    long double perturbation_eta_upper = 0;
    long double data_error_upper = 0;
    long double normalized_total_error_upper = 0;
    long double half_resolution_lower = 0;
    // Original row-max units; includes every omitted and retained row.
    std::array<long double, 13> original_row_residual_upper{};
};

namespace photon_original_direction_detail {
struct Interval {
    long double low = 0, high = 0;
};
inline Interval
Add (Interval a, Interval b) {
    return {photon_condition_detail::Down(a.low + b.low),
            photon_condition_detail::Up(a.high + b.high)};
}
inline Interval
Multiply (Interval a, Interval b) {
    std::array<long double, 4> const products{a.low * b.low, a.low * b.high,
                                              a.high * b.low, a.high * b.high};
    return {photon_condition_detail::Down(
                *std::min_element(products.begin(), products.end())),
            photon_condition_detail::Up(
                *std::max_element(products.begin(), products.end()))};
}
inline long double
Absolute (Interval value) {
    return std::max(std::abs(value.low), std::abs(value.high));
}
inline bool
Finite (Interval value) {
    return std::isfinite(value.low) && std::isfinite(value.high) &&
           value.low <= value.high;
}
inline bool
UniqueRange (std::vector<std::size_t> const& values, std::size_t limit) {
    for (std::size_t n = 0; n < values.size(); ++n) {
        if (values[n] >= limit || std::find(values.begin(), values.begin() + n,
                                            values[n]) != values.begin() + n) {
            return false;
        }
    }
    return true;
}
} // namespace photon_original_direction_detail

/** Certify the ACTUAL unchanged Gaussian direction against exact original
 * binary64 coefficient ratios, and a caller-owned original-coefficient box.
 * coefficient_radius[record][feature] is an ABSOLUTE, outward uncertainty in
 * ORIGINAL unscaled units. It is neither measured endpoint defect nor a user
 * tolerance. Feature0 is exactly1 and its radius MUST be0. Native formation
 * radii/libm assumptions belong to the caller; this helper assumes no sin/cos/
 * sqrt bound or physical-input continuation radius. No hardcoded ULP radius.
 *
 * Fixed original row maxima s are represented binary64 values, exact in LD.
 * Enclose a/s by adjacent LD values. With retained square B, approximate R
 * and nu>=||I-RB||inf<1, K=||R||inf/(1-nu) bounds ||B^-1||inf. For represented
 * h (||h||inf=1), rho>=||B h_p+C h_f||inf gives e=K*rho. For the coefficient
 * box, beta>=||dB||inf, omega>=||dC h_f||inf, eta=K*beta<1 and
 * d=K/(1-eta)*(omega+beta*(||h_p||inf+e)) enclose exact fixed-free solution
 * changes. Normalization then gives 2*(e+d)/(1-e-d), rounded outward.
 * Require that bound <= the ORIGINAL lower sqrt(gamma) resolution floor.
 * All original rows are reported; omitted-row/full13 moment acceptance is
 * STILL the separate mandatory BOTH-endpoint guard, never inferred from B.
 *
 * This is a fixed-coordinate represented-input proof, not stability of row/
 * pivot/cohort/eligibility selection or universal pathwise continuation.
 * New native/runtime use requires an explicit policy and independent existing
 * reorder/decomposition/restart gates. Old certificate/defaults remain intact.
 */
[[nodiscard]] inline PhotonWeightOriginalDirectionCertificate
CertifyPhotonWeightOriginalDirection (
    std::vector<PhotonWeightRecord> const& records,
    std::vector<std::size_t> const& active,
    photon_weight_detail::NullDirectionPivots const& retained,
    std::vector<long double> const& direction,
    std::vector<std::vector<long double>> const& coefficient_radius) {
    using namespace photon_condition_detail;
    using namespace photon_original_direction_detail;
    PhotonWeightOriginalDirectionCertificate out;
    out.reason = "unsupported original-direction arithmetic or shape";
#ifdef __FAST_MATH__
    return out;
#endif
    bool const arithmetic =
        std::numeric_limits<double>::is_iec559 &&
        std::numeric_limits<double>::radix == 2 &&
        std::numeric_limits<double>::digits == 53 &&
        std::numeric_limits<long double>::is_iec559 &&
        std::numeric_limits<long double>::radix == 2 &&
        std::numeric_limits<long double>::digits >= 64 &&
        std::numeric_limits<long double>::min_exponent <= -8192 &&
        std::numeric_limits<long double>::max_exponent >= 8192 &&
        std::fegetround() == FE_TONEAREST;
    if (!arithmetic || records.empty() || records.size() > 14 ||
        active.size() < 2 || active.size() > 14 ||
        direction.size() != active.size() ||
        coefficient_radius.size() != records.size() || retained.rows.empty() ||
        retained.rows.size() != retained.columns.size() ||
        retained.rows.size() >= active.size() ||
        !UniqueRange(active, records.size()) ||
        !UniqueRange(retained.rows, 13) ||
        !UniqueRange(retained.columns, active.size())) {
        return out;
    }
    out.reason =
        "invalid original coefficient, radius, weight or tagged identity";
    std::size_t positives = 0;
    for (std::size_t p = 0; p < records.size(); ++p) {
        auto const& record = records[p];
        auto const& radius = coefficient_radius[p];
        if (record.features.size() != 13 || radius.size() != 13 ||
            record.features[0] != 1 || radius[0] != 0 ||
            !std::isfinite(record.weight) || record.weight < 0 ||
            record.key.first > 1 || record.key.second == 0) {
            return out;
        }
        for (std::size_t q = 0; q < p; ++q) {
            if (records[q].key == record.key) {
                return out;
            }
        }
        positives += record.weight > 0;
        for (std::size_t r = 0; r < 13; ++r) {
            if (!std::isfinite(record.features[r]) ||
                !std::isfinite(radius[r]) || radius[r] < 0) {
                return out;
            }
        }
    }
    if (positives != active.size()) {
        return out;
    }
    for (auto p : active) {
        if (!(records[p].weight > 0)) {
            return out;
        }
    }
    out.reason = "invalid represented normalization or free coordinates";
    long double norm = 0, pivot_norm = 0;
    std::size_t nonzero_free = 0;
    for (std::size_t c = 0; c < active.size(); ++c) {
        if (!std::isfinite(direction[c])) {
            return out;
        }
        norm = std::max(norm, std::abs(direction[c]));
        if (std::find(retained.columns.begin(), retained.columns.end(), c) !=
            retained.columns.end()) {
            pivot_norm = std::max(pivot_norm, std::abs(direction[c]));
        } else if (direction[c] != 0) {
            if (!(direction[c] > 0)) {
                return out;
            }
            ++nonzero_free;
        }
    }
    if (norm != 1 || nonzero_free != 1) {
        return out;
    }
    Matrix represented(13, std::vector<long double>(active.size()));
    Matrix radius(13, std::vector<long double>(active.size()));
    std::vector<std::vector<Interval>> exact(
        13, std::vector<Interval>(active.size()));
    out.reason = "original coefficient scaling or radius overflow";
    for (std::size_t r = 0; r < 13; ++r) {
        long double scale = 0;
        for (auto p : active) {
            scale = std::max(scale, std::abs(static_cast<long double>(
                                        records[p].features[r])));
        }
        for (std::size_t c = 0; c < active.size(); ++c) {
            auto const numerator =
                static_cast<long double>(records[active[c]].features[r]);
            auto const uncertainty = coefficient_radius[active[c]][r];
            if (scale == 0) {
                if (uncertainty != 0) {
                    return out;
                }
                continue;
            }
            auto const quotient = numerator / scale;
            represented[r][c] = quotient;
            exact[r][c] = {Down(quotient), Up(quotient)};
            radius[r][c] =
                uncertainty == 0 ? 0.L : DivideUp(uncertainty, scale);
            if (!Finite(exact[r][c]) || !std::isfinite(radius[r][c])) {
                return out;
            }
        }
    }
    auto const rank = retained.rows.size();
    Matrix system(rank, std::vector<long double>(rank));
    for (std::size_t i = 0; i < rank; ++i)
        for (std::size_t j = 0; j < rank; ++j) {
            system[i][j] = represented[retained.rows[i]][retained.columns[j]];
        }
    auto const inverse = ApproximateInverse(system);
    out.reason = "original inverse witness unavailable";
    if (inverse.empty()) {
        return out;
    }
    for (auto const& row : inverse) {
        for (auto value : row) {
            if (!std::isfinite(value)) {
                return out;
            }
        }
    }
    out.reason = "original inverse residual uncertified";
    for (std::size_t i = 0; i < rank; ++i) {
        long double row = 0;
        for (std::size_t j = 0; j < rank; ++j) {
            Interval dot;
            for (std::size_t k = 0; k < rank; ++k) {
                dot =
                    Add(dot,
                        Multiply({inverse[i][k], inverse[i][k]},
                                 exact[retained.rows[k]][retained.columns[j]]));
            }
            auto const identity = static_cast<long double>(i == j);
            Interval const defect{Down(identity - dot.high),
                                  Up(identity - dot.low)};
            if (!Finite(defect)) {
                return out;
            }
            row = AddUp(row, Absolute(defect));
        }
        out.inverse_residual_upper = std::max(out.inverse_residual_upper, row);
    }
    if (!(out.inverse_residual_upper >= 0 && out.inverse_residual_upper < 1)) {
        return out;
    }
    out.inverse_norm_upper =
        DivideUp(NormUpper(inverse), Down(1 - out.inverse_residual_upper));
    if (!std::isfinite(out.inverse_norm_upper)) {
        return out;
    }
    for (std::size_t r = 0; r < 13; ++r) {
        Interval residual;
        for (std::size_t c = 0; c < active.size(); ++c) {
            residual = Add(residual,
                           Multiply(exact[r][c], {direction[c], direction[c]}));
        }
        if (!Finite(residual)) {
            return out;
        }
        out.original_row_residual_upper[r] = Absolute(residual);
    }
    for (auto r : retained.rows) {
        out.retained_residual_upper = std::max(
            out.retained_residual_upper, out.original_row_residual_upper[r]);
    }
    out.solve_error_upper =
        MultiplyUp(out.inverse_norm_upper, out.retained_residual_upper);
    long double beta = 0, omega = 0;
    for (auto r : retained.rows) {
        long double pivot_row = 0, free_row = 0;
        for (std::size_t c = 0; c < active.size(); ++c) {
            if (std::find(retained.columns.begin(), retained.columns.end(),
                          c) != retained.columns.end()) {
                pivot_row = AddUp(pivot_row, radius[r][c]);
            } else {
                free_row = AddUp(
                    free_row, MultiplyUp(radius[r][c], std::abs(direction[c])));
            }
        }
        beta = std::max(beta, pivot_row);
        omega = std::max(omega, free_row);
    }
    out.perturbation_eta_upper = MultiplyUp(out.inverse_norm_upper, beta);
    out.reason = "original coefficient box or normalization unresolved";
    if (!(out.perturbation_eta_upper >= 0 && out.perturbation_eta_upper < 1) ||
        !(out.solve_error_upper >= 0 && out.solve_error_upper < 1)) {
        return out;
    }
    auto const rhs = AddUp(
        omega, MultiplyUp(beta, AddUp(pivot_norm, out.solve_error_upper)));
    out.data_error_upper = MultiplyUp(
        DivideUp(out.inverse_norm_upper, Down(1 - out.perturbation_eta_upper)),
        rhs);
    auto const total = AddUp(out.solve_error_upper, out.data_error_upper);
    if (!(total >= 0 && total < 1)) {
        return out;
    }
    out.normalized_total_error_upper =
        DivideUp(MultiplyUp(2, total), Down(1 - total));
    // EXACT existing 13-feature operation envelope and binary64 epsilon.
    auto const product =
        (64.L * 14 * 14 + 32) * std::numeric_limits<double>::epsilon();
    out.half_resolution_lower =
        Down(std::sqrt(Down(product / Up(1 - product))));
    out.valid = std::isfinite(out.normalized_total_error_upper) &&
                out.half_resolution_lower > 0;
    out.admitted = out.valid && out.normalized_total_error_upper <=
                                    out.half_resolution_lower;
    out.reason =
        out.admitted
            ? "certified original-equation direction and coefficient box"
            : "original-equation direction resolution refused";
    return out;
}
} // namespace warpx::radiation
#endif
