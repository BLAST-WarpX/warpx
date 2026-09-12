/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_MOMENTBOUNDARYFLUX_H_
#define WARPX_RADIATION_MOMENTBOUNDARYFLUX_H_

#include "MomentClosure.H"
#include "Utils/WarpXConst.H"

#include <cmath>

namespace warpx::radiation {
struct MomentBoundaryFluxResult {
    // Outward flux of (E,q=F/c); divide components 1:3 by c for momentum.
    FourVector flux{};
    bool valid = false;
};

/** Outgoing half-space flux of the normalized gray M1 angular distribution.
 * normal is a Cartesian component; side is the outward sign. This is not
 * an AP boundary prescription or a time integrator. No input is modified.
 * See radiation_moment_boundary_plan.md for integration obligations.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE MomentBoundaryFluxResult
EvaluateM1OutgoingFlux (FourVector const& state, int normal,
                        int side) noexcept {
    MomentBoundaryFluxResult result;
    if (normal < 0 || normal > 2 || (side != -1 && side != 1)) {
        return result;
    }
    auto const closure = EvaluateM1Closure(state);
    if (!closure.valid) {
        return result;
    }
    if (state[0] == 0) {
        result.valid = true;
        return result;
    }
    auto const norm = std::hypot(std::hypot(state[1], state[2]), state[3]);
    auto const f = norm / state[0];
    if (f >= 1) {
        // Use the existing closure's precision collar without changing E/q.
        // Exact grazing beams have zero outward flux, not an undefined 0/0.
        if (side * state[normal + 1] > 0) {
            result.flux[0] = PhysConst::c * (side * state[normal + 1]);
            for (int d = 0; d < 3; ++d) {
                result.flux[d + 1] =
                    PhysConst::c * (side * closure.tensor[d + 1][normal + 1]);
            }
        }
    } else {
        auto const root = std::sqrt(4 - 3 * f * f);
        auto const coefficient = 3 / (2 + root);
        amrex::GpuArray<amrex::Real, 3> b{};
        for (int d = 0; d < 3; ++d) {
            b[d] = coefficient * (state[d + 1] / state[0]);
        }
        auto const magnitude = coefficient * f;
        // Stable even when the radiation angular distribution is nearly a beam.
        auto const delta = 12 * (1 - f) * (1 + f) / ((root + 1) * (root + 2));
        auto const bn = side * b[normal];
        auto const distance = delta + bn * bn;
        auto const r = std::sqrt(distance);
        auto const a = -bn / r;
        auto const s = bn >= 0 ? 1 + bn / r : delta / (r * (r - bn));
        auto const denominator = 3 + magnitude * magnitude;
        auto const energy_fraction =
            r * s * s * s * (3 + a) / (4 * denominator);
        auto const normal_fraction = distance * s * s * s / (2 * denominator);
        result.flux[0] = PhysConst::c * (state[0] * energy_fraction);
        for (int d = 0; d < 3; ++d) {
            result.flux[d + 1] = b[d] * result.flux[0];
        }
        result.flux[normal + 1] =
            side * (PhysConst::c * (state[0] * normal_fraction));
    }
    for (auto value : result.flux) {
        if (!amrex::Math::isfinite(value)) {
            return {};
        }
    }
    result.valid = true;
    return result;
}

/** Fixed specular optical wall: zero energy/tangential exchange, normal
 * pressure twice the outgoing normal momentum flux. No material-wall or
 * pending-particle-impulse boundary rule is supplied by this helper. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE MomentBoundaryFluxResult
EvaluateM1MirrorFlux (FourVector const& state, int normal, int side) noexcept {
    auto const outgoing = EvaluateM1OutgoingFlux(state, normal, side);
    if (!outgoing.valid) {
        return {};
    }
    MomentBoundaryFluxResult result;
    result.flux[normal + 1] = 2 * outgoing.flux[normal + 1];
    result.valid = amrex::Math::isfinite(result.flux[normal + 1]);
    return result.valid ? result : MomentBoundaryFluxResult{};
}

struct MomentMirrorJacobianResult {
    // Derivatives of the Cartesian normal q-flux with respect to (E,qx,qy,qz).
    // All other mirror flux rows vanish identically.
    FourVector derivative{};
    bool valid = false;
};

/** Analytic linearization of the same half-space pressure used by the mirror.
 * At vacuum use the isotropic limiting Jacobian, as for the M1 closure; no
 * direction-independent Frechet derivative exists at vacuum. At an exact
 * grazing beam use the symmetric angular limit. These choices do not alter
 * the physical flux or clip the supplied moments.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE MomentMirrorJacobianResult
EvaluateM1MirrorJacobian (FourVector const& state, int normal, int side) noexcept
{
    MomentMirrorJacobianResult result;
    if (!EvaluateM1MirrorFlux(state, normal, side).valid) {
        return result;
    }
    auto const norm = std::hypot(std::hypot(state[1], state[2]), state[3]);
    auto const f = state[0] > 0 ? norm / state[0] : amrex::Real(0);
    if (f >= 1) {
        auto const normal_q = side * state[normal + 1];
        auto const multiplier = normal_q > 0 ? 2 : (normal_q < 0 ? 0 : 1);
        for (int column = 0; column < 4; ++column) {
            result.derivative[column] = side * PhysConst::c * multiplier *
                M1ClosureDerivative(state, column)[normal + 1][normal + 1];
        }
    } else {
        auto const root = std::sqrt(4 - 3 * f * f);
        auto const coefficient = 3 / (2 + root);
        auto const curvature = coefficient * coefficient / root;
        amrex::GpuArray<amrex::Real, 3> reduced{}, b{}, gradient{};
        for (int d = 0; d < 3; ++d) {
            reduced[d] = state[0] > 0 ? state[d + 1] / state[0] : amrex::Real(0);
            b[d] = coefficient * reduced[d];
        }
        auto const delta = 12 * (1 - f) * (1 + f) / ((root + 1) * (root + 2));
        auto const bn = side * b[normal];
        auto const r = std::sqrt(delta + bn * bn);
        auto const s = bn >= 0 ? 1 + bn / r : delta / (r * (r - bn));
        auto const denominator = 3 + coefficient * coefficient * f * f;
        auto const h = r * r * s * s * s / denominator;
        for (int d = 0; d < 3; ++d) {
            gradient[d] = b[d] * s * s * (s - 3) / denominator -
                          2 * b[d] * h / denominator;
        }
        gradient[normal] = side * (3 * r * s * s / denominator -
                                   2 * bn * h / denominator);
        // E*db/dq and E*db/dE are evaluated without forming 1/E, preserving
        // the derivative for scale-extreme but otherwise identical states.
        auto energy_derivative = h;
        for (int d = 0; d < 3; ++d) {
            energy_derivative -= gradient[d] * (coefficient + curvature * f * f) * reduced[d];
        }
        result.derivative[0] = side * PhysConst::c * energy_derivative;
        for (int column = 0; column < 3; ++column) {
            auto value = coefficient * gradient[column];
            for (int d = 0; d < 3; ++d) {
                value += gradient[d] * curvature * reduced[d] * reduced[column];
            }
            result.derivative[column + 1] = side * PhysConst::c * value;
        }
    }
    for (auto value : result.derivative) {
        if (!amrex::Math::isfinite(value)) {
            return {};
        }
    }
    result.valid = true;
    return result;
}
} // namespace warpx::radiation
#endif
