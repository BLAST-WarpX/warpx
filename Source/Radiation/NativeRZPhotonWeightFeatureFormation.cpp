/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativeRZPhotonWeightFeatureFormation.H"

#include "Utils/WarpXConst.H"

#include <AMReX_Particle.H>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>

namespace warpx::radiation {
namespace {
struct Interval {
    long double low = 0, high = 0;
};
long double
Down (long double value) {
    return std::nextafter(value, -std::numeric_limits<long double>::infinity());
}
long double
Up (long double value) {
    return std::nextafter(value, std::numeric_limits<long double>::infinity());
}
Interval
Point (long double value) {
    return {value, value};
}
bool
Zero (Interval value) {
    return value.low == 0 && value.high == 0;
}
bool
Finite (Interval value) {
    return std::isfinite(value.low) && std::isfinite(value.high) &&
           value.low <= value.high;
}

Interval
Add (Interval a, Interval b) {
    if (Zero(a)) {
        return b;
    }
    if (Zero(b)) {
        return a;
    }
    return {Down(a.low + b.low), Up(a.high + b.high)};
}
Interval
Subtract (Interval a, Interval b) {
    if (Zero(b)) {
        return a;
    }
    return {Down(a.low - b.high), Up(a.high - b.low)};
}
Interval
Multiply (Interval a, Interval b) {
    if (Zero(a) || Zero(b)) {
        return {};
    } // Exact algebraic zero, not a fitted radius.
    std::array<long double, 4> const products{a.low * b.low, a.low * b.high,
                                              a.high * b.low, a.high * b.high};
    return {Down(*std::min_element(products.begin(), products.end())),
            Up(*std::max_element(products.begin(), products.end()))};
}
Interval
Square (Interval value) {
    if (Zero(value)) {
        return {};
    }
    auto const low = value.low * value.low, high = value.high * value.high;
    return {value.low <= 0 && value.high >= 0 ? 0 : Down(std::min(low, high)),
            Up(std::max(low, high))};
}
Interval
DividePositive (Interval a, Interval b) {
    if (Zero(a)) {
        return {};
    }
    std::array<long double, 4> const quotients{a.low / b.low, a.low / b.high,
                                               a.high / b.low, a.high / b.high};
    return {Down(*std::min_element(quotients.begin(), quotients.end())),
            Up(*std::max_element(quotients.begin(), quotients.end()))};
}
Interval
LibmPoint (long double value) {
    // Conditional one-ULP libm error; two neighbors cover a binade boundary.
    return {Down(Down(value)), Up(Up(value))};
}
} // namespace

NativePhotonWeightFeatureFormation
BoundNativeRZPhotonWeightFeatureFormation (
    NativePhotonWeightUnitPose const& pose, PhotonWeightRecord const& row) {
    NativePhotonWeightFeatureFormation out;
    out.reason = "unsupported native feature formation precision/rounding";
#if defined(__FAST_MATH__)
    return out;
#else
    if (std::numeric_limits<amrex::Real>::digits != 53 ||
        std::numeric_limits<amrex::ParticleReal>::digits != 53 ||
        !std::numeric_limits<amrex::Real>::is_iec559 ||
        !std::numeric_limits<amrex::ParticleReal>::is_iec559 ||
        std::numeric_limits<amrex::Real>::radix != 2 ||
        std::numeric_limits<amrex::ParticleReal>::radix != 2 ||
        !std::numeric_limits<double>::is_iec559 ||
        std::numeric_limits<double>::radix != 2 ||
        !std::numeric_limits<long double>::is_iec559 ||
        std::numeric_limits<long double>::radix != 2 ||
        std::numeric_limits<long double>::digits < 64 ||
        std::numeric_limits<long double>::min_exponent > -8192 ||
        std::numeric_limits<long double>::max_exponent < 8192 ||
        std::fegetround() != FE_TONEAREST) {
        return out;
    }
    out.reason = "invalid native feature formation pose/record";
    if (row.features.size() != 13 || row.features[0] != 1 ||
        !std::isfinite(row.weight) || row.weight < 0 || pose[0] < 0) {
        return out;
    }
    for (auto value : pose) {
        if (!std::isfinite(value)) {
            return out;
        }
    }
    for (auto value : row.features) {
        if (!std::isfinite(value)) {
            return out;
        }
    }
    auto const r = Point(pose[0]), x = Point(pose[2]), y = Point(pose[3]),
               z = Point(pose[4]);
    auto const length_squared = Add(Add(Square(x), Square(y)), Square(z));
    if (!Finite(length_squared) || !(length_squared.low > 0)) {
        return out;
    }
    Interval const length{LibmPoint(std::sqrt(length_squared.low)).low,
                          LibmPoint(std::sqrt(length_squared.high)).high};
    if (!Finite(length) || !(length.low > 0)) {
        return out;
    }
    auto const cosine = std::cos(static_cast<long double>(pose[1]));
    auto const sine = std::sin(static_cast<long double>(pose[1]));
    // Exact analytic identities for stored +/-0, never a small-angle threshold.
    if (pose[1] == 0 && (cosine != 1 || sine != 0)) {
        return out;
    }
    auto const c = pose[1] == 0 ? Point(1) : LibmPoint(cosine);
    auto const s = pose[1] == 0 ? Point(0) : LibmPoint(sine);
    auto const ur = Add(Multiply(c, x), Multiply(s, y));
    auto const ut = Subtract(Multiply(c, y), Multiply(s, x));
    auto const f = Multiply(Point(PhysConst::m_e), Point(PhysConst::c));
    auto const energy = Multiply(f, length);
    auto const nr = DividePositive(ur, length), nt = DividePositive(ut, length);
    auto const nz = DividePositive(z, length);
    // Keep actual unit NativeValues grouping: cLz=(f*r)*ut, Pij=(E*ni)*nj.
    std::array<Interval, 13> const features{Point(1),
                                            energy,
                                            Multiply(f, x),
                                            Multiply(f, y),
                                            Multiply(f, z),
                                            Multiply(Multiply(f, r), ut),
                                            Multiply(f, ur),
                                            Multiply(Multiply(energy, nr), nr),
                                            Multiply(Multiply(energy, nt), nt),
                                            Multiply(Multiply(energy, nz), nz),
                                            Multiply(Multiply(energy, nr), nt),
                                            Multiply(Multiply(energy, nr), nz),
                                            Multiply(Multiply(energy, nt), nz)};
    for (std::size_t d = 0; d < features.size(); ++d) {
        if (!Finite(features[d])) {
            return out;
        }
        if (d == 0) {
            out.absolute_radius[d] = 0;
            continue;
        }
        auto const represented = static_cast<long double>(row.features[d]);
        if (features[d].low == features[d].high &&
            features[d].low == represented) {
            out.absolute_radius[d] = 0;
            continue;
        } // Exact singleton, including algebraic zero.
        // Directed subtraction and absolute-value maximum enclose final cast.
        auto const below = Down(features[d].low - represented);
        auto const above = Up(features[d].high - represented);
        out.absolute_radius[d] = Up(std::max(std::abs(below), std::abs(above)));
        if (!std::isfinite(out.absolute_radius[d])) {
            return out;
        }
    }
    out.valid = true;
    out.reason = "conditional native unit-pose formation enclosure";
    return out;
#endif
}
} // namespace warpx::radiation
