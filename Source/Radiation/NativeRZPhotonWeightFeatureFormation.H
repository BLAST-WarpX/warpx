/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_NATIVE_RZ_PHOTON_WEIGHT_FEATURE_FORMATION_H_
#define WARPX_NATIVE_RZ_PHOTON_WEIGHT_FEATURE_FORMATION_H_

#include "ConservativePhotonWeightProposal.H"

#include <array>
#include <string>

namespace warpx::radiation {
// Exact stored binary64 r, theta, ux, uy, uz. No position/trajectory
// uncertainty.
using NativePhotonWeightUnitPose = std::array<double, 5>;

struct NativePhotonWeightFeatureFormation {
    bool valid = false;
    std::array<long double, 13> absolute_radius{}; // ORIGINAL feature units.
    std::string reason;
};

/** Pure CPU-host coefficient formation enclosure; no native owner or RNG.
 * Treat stored pose and binary64 PhysConst values as exact. Outward basic LD
 * operations enclose the NativeValues(unit-weight) formula, including norm,
 * local-basis rotation and all13 pressure/account features. Sqrt/sin/cos use
 * an EXPLICIT CONDITIONAL <=1-ULP LD-libm accuracy assumption, not a C++
 * standard guarantee; two neighboring LD values cover binade spacing. Stored
 * +/-0 angles use the exact analytic sin(0)=0 and cos(0)=1 identities.
 * Radius encloses each true given-pose coefficient relative to the authentic
 * represented binary64 feature, so it includes formation and final cast.
 * Number's coefficient is exactly one and its radius is exactly zero.
 * Requires nearest IEEE radix-2 binary64 fields/particles, LD with at least
 * 64 significand bits and extended exponent range, and no fast math. Invalid
 * domains are refused; no radius is fitted from a residual.
 * This does not bound physical trajectory or restart input perturbations.
 * The caller supplies actual NativeValues(unit-weight) record and stored pose.
 * This routine is never called by established/default proposal paths.
 */
[[nodiscard]] NativePhotonWeightFeatureFormation
BoundNativeRZPhotonWeightFeatureFormation (NativePhotonWeightUnitPose const&,
                                           PhotonWeightRecord const&);
} // namespace warpx::radiation
#endif
