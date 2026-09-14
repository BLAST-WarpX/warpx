/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_MOMENTCLOSURE_H_
#define WARPX_RADIATION_MOMENTCLOSURE_H_

#include "FrameTransform.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

namespace warpx::radiation
{
    struct MomentClosureResult
    {
        StressEnergy tensor{};
        bool valid = false;
    };

    /** Lab-frame M1 closure for (E,F/c), all in energy-density units.
     * Radiation momentum density is F/c^2, not F/c. The closure represents a
     * boosted isotropic angular distribution, not arbitrary crossing beams.
     * No source, update, or positivity repair is performed here. Non-realizable
     * candidates must be rejected by the integrator; silently clipping flux
     * would destroy its momentum conservation. A precision-sized boundary
     * collar accommodates rounded beam moments and host/device norm arithmetic.
     * Within that collar the input energy and flux remain exactly unchanged;
     * only the constitutive square root uses its limiting beam value.
     *
     * chi = (3+4*f^2)/(5+2*sqrt(4-3*f^2)). The equivalent tensor form below
     * avoids division by f^2 and loss of anisotropy near isotropic equilibrium.
     * This gray tensor may not be boosted group by group as though finite
     * frequency integration boundaries were Lorentz invariant.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE MomentClosureResult
    EvaluateM1Closure (FourVector const& moments) noexcept
    {
        MomentClosureResult result;
        for (auto value : moments)
        {
            if (!amrex::Math::isfinite(value))
            {
                return result;
            }
        }
        amrex::Real const energy = moments[0];
        if (energy < 0)
        {
            return result;
        }
        if (energy == 0)
        {
            result.valid = moments[1] == 0 && moments[2] == 0 && moments[3] == 0;
            return result;
        }
        amrex::GpuArray<amrex::Real, 3> reduced_flux{};
        for (int d = 0; d < 3; ++d)
        {
            reduced_flux[d] = moments[d + 1] / energy;
        }
        // A scaled hypot norm avoids an intermediate overflow and avoids the
        // extra square/sum rounding that can reject a represented oblique beam.
        amrex::Real const f = std::hypot(
            std::hypot(moments[1], moments[2]), moments[3]) / energy;
        constexpr amrex::Real roundoff = 8 * std::numeric_limits<amrex::Real>::epsilon();
        if (!(f <= 1 + roundoff))
        {
            return result;
        }
        amrex::Real const f2 = f * f;
        amrex::Real const root = std::sqrt(amrex::max(amrex::Real(1), 4 - 3 * f2));
        amrex::Real const isotropic = (root - 1) / 3;
        amrex::Real const anisotropic = 3 / (2 + root);
        result.tensor[0][0] = energy;
        for (int d = 0; d < 3; ++d)
        {
            result.tensor[0][d + 1] = moments[d + 1];
            result.tensor[d + 1][0] = moments[d + 1];
            for (int e = 0; e < 3; ++e)
            {
                result.tensor[d + 1][e + 1] = energy *
                    ((d == e ? isotropic : 0) + anisotropic * reduced_flux[d] * reduced_flux[e]);
            }
        }
        result.valid = true;
        return result;
    }

    /** Derivative dT/d(E,F/c)[component] at a previously validated M1 state.
     * At vacuum the isotropic tangent supplies an initial Newton linearization;
     * every nonzero trial must recompute its actual angular closure. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE StressEnergy
    M1ClosureDerivative (FourVector const& moments, int component) noexcept
    {
        StressEnergy derivative{};
        derivative[0][component] = 1;
        derivative[component][0] = 1;
        amrex::GpuArray<amrex::Real, 3> f{};
        amrex::Real f2 = 0;
        if (moments[0] > 0)
        {
            for (int d = 0; d < 3; ++d)
            {
                f[d] = moments[d + 1] / moments[0];
                f2 += f[d] * f[d];
            }
        }
        amrex::Real const s = std::sqrt(amrex::max(amrex::Real(1), 4 - 3 * f2));
        amrex::Real const a = (s - 1) / 3;
        amrex::Real const b = 3 / (2 + s);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                amrex::Real value = 0;
                if (component == 0)
                {
                    value = (i == j ? a + f2 / s : 0)
                        - (b + b * b * f2 / s) * f[i] * f[j];
                }
                else
                {
                    int const k = component - 1;
                    value = -(i == j ? f[k] / s : 0)
                        + b * b * f[k] / s * f[i] * f[j]
                        + b * ((i == k ? f[j] : 0) + (j == k ? f[i] : 0));
                }
                derivative[i + 1][j + 1] = value;
            }
        }
        return derivative;
    }
}
#endif
