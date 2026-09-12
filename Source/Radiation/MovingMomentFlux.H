/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_MOVINGMOMENTFLUX_H_
#define WARPX_RADIATION_MOVINGMOMENTFLUX_H_

#include "MomentClosure.H"
#include "Utils/WarpXConst.H"

#include <cmath>

namespace warpx::radiation
{
    struct MovingMomentFluxResult {
        // Conservative fluxes of energy density and q=F/c. Divide the latter
        // by c to obtain physical momentum flux. Not cell-integrated values.
        FourVector flux{};
        amrex::Real projected_flux = 0;
        // Split for checking weak diffusion underneath a large advective flux.
        amrex::Real projected_nonequilibrium_flux = 0;
        amrex::Real asymptotic_weight = 0;
        bool valid = false;
    };

    namespace detail
    {
        // Also used on signed Krylov vectors with a frozen constitutive pressure.
        // Physical callers must validate the M1 states through the public wrapper.
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE MovingMomentFluxResult
        MovingMomentFluxWithPressure (
            FourVector const& left, FourVector const& right, StressEnergy const& left_pressure,
            StressEnergy const& right_pressure, amrex::GpuArray<amrex::Real, 3> const& beta,
            int normal, amrex::Real kappa, amrex::Real spacing,
            amrex::GpuArray<amrex::Real, 3> const& transverse_gradient = {}) noexcept
        {
            using namespace amrex::literals;
            MovingMomentFluxResult result;
            if (normal < 0 || normal > 2 || !(kappa >= 0) || !(spacing > 0) ||
                !amrex::Math::isfinite(kappa) || !amrex::Math::isfinite(spacing)) {
                return result;
            }
            amrex::Real b2 = 0;
            for (int d = 0; d < 3; ++d) {
                if (!amrex::Math::isfinite(beta[d]) ||
                    !amrex::Math::isfinite(transverse_gradient[d])) {
                    return result;
                }
                b2 += beta[d] * beta[d];
            }
            if (!(b2 < 1)) {
                return result;
            }
            auto const gamma = 1 / std::sqrt(1 - b2);
            auto const bn = beta[normal];
            auto const length = 1.5_rt * gamma * spacing / (1 - bn * bn);
            if (!amrex::Math::isfinite(length)) {
                return result;
            }
            auto const tau = kappa * length;
            auto const alpha = amrex::Math::isfinite(tau) ? 1 / (1 + tau) : (1 / kappa) / length;
            // Stable for transparent cells: do not form 1-alpha and lose a weak
            // optical depth, or divide a transverse coefficient by zero opacity.
            auto const blend = amrex::Math::isfinite(tau) ? tau / (1 + tau) : amrex::Real(1);
            amrex::Real rl = left[0];
            amrex::Real rr = right[0];
            amrex::Real sl = left[normal + 1];
            amrex::Real sr = right[normal + 1];
            for (int d = 0; d < 3; ++d) {
                rl -= beta[d] * left[d + 1];
                rr -= beta[d] * right[d + 1];
                sl -= beta[d] * left_pressure[d + 1][normal + 1];
                sr -= beta[d] * right_pressure[d + 1][normal + 1];
                result.flux[d + 1] =
                    PhysConst::c *
                    (0.5_rt * (left_pressure[d + 1][normal + 1] + right_pressure[d + 1][normal + 1]) -
                     0.5_rt * alpha * (right[d + 1] - left[d + 1]));
            }
            auto const average_r = 0.5_rt * (rl + rr);
            auto const jump_r = rr - rl;
            auto nonequilibrium_flux =
                PhysConst::c * alpha * (0.5_rt * (sl + sr) - bn * average_r - 0.5_rt * jump_r);
            // Divide before the second blend multiplication to avoid underflow
            // of blend^2 at weak opacity and overflow of kappa*spacing at large tau.
            auto const transverse_coefficient =
                kappa > 0 ? PhysConst::c * (blend * (blend / kappa)) / (3 * gamma) : amrex::Real(0);
            for (int d = 0; d < 3; ++d) {
                if (d != normal) {
                    nonequilibrium_flux +=
                        transverse_coefficient * bn * beta[d] * transverse_gradient[d];
                }
            }
            auto const slow_flux =
                PhysConst::c * (bn * average_r - 0.5_rt * blend * std::abs(bn) * jump_r) +
                nonequilibrium_flux;
            result.projected_nonequilibrium_flux = nonequilibrium_flux;
            result.projected_flux = slow_flux;
            result.flux[0] = slow_flux;
            for (int d = 0; d < 3; ++d) {
                result.flux[0] += beta[d] * result.flux[d + 1];
            }
            for (auto value : result.flux) {
                if (!amrex::Math::isfinite(value)) {
                    return MovingMomentFluxResult{};
                }
            }
            result.asymptotic_weight = alpha;
            result.valid = amrex::Math::isfinite(slow_flux);
            return result;
        }
    } // namespace detail

    /** Candidate source-aware LLF face flux. One shared face beta must be used
     * for both states and all four components. The thick-limit target is
     * D=c*(I-beta*beta^T)/(3*kappa*gamma) for constant-velocity gray scattering.
     * transverse_gradient holds derivatives of E-beta.q using the SAME face
     * beta; its normal entry is ignored. This kernel provides no time advance,
     * boundary prescription, positivity guarantee or finite-group transform.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE MovingMomentFluxResult
    EvaluateMovingMomentFlux (
        FourVector const& left, FourVector const& right,
        amrex::GpuArray<amrex::Real, 3> const& beta, int normal, amrex::Real kappa,
        amrex::Real spacing,
        amrex::GpuArray<amrex::Real, 3> const& transverse_gradient = {}) noexcept
    {
        auto const l = EvaluateM1Closure(left);
        auto const r = EvaluateM1Closure(right);
        if (!l.valid || !r.valid) {
            return {};
        }
        return detail::MovingMomentFluxWithPressure(left, right, l.tensor, r.tensor, beta, normal,
                                                    kappa, spacing, transverse_gradient);
    }
} // namespace warpx::radiation
#endif
