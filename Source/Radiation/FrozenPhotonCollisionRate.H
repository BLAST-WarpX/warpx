/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_FROZENPHOTONCOLLISIONRATE_H_
#define WARPX_RADIATION_FROZENPHOTONCOLLISIONRATE_H_

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

namespace warpx::radiation
{
    struct FrozenPhotonCollisionRate
    {
        amrex::Real inverse_length = 0;
        amrex::Real maximum_inverse_length = 0;
        bool valid = false;
    };

    /** Lab event rate is c*inverse_length for a prescribed proper opacity chi0.
     * The material frame is frozen at interval start. Updating the photon
     * direction after an event changes its waiting rate even in that frame.
     * This supplies a local source rate, not a moving transport discretization.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE FrozenPhotonCollisionRate
    EvaluateFrozenPhotonCollisionRate (
        amrex::Real chi0, amrex::GpuArray<amrex::Real, 3> const& beta,
        amrex::GpuArray<amrex::Real, 3> const& laboratory_direction) noexcept
    {
        FrozenPhotonCollisionRate result;
        if (!amrex::Math::isfinite(chi0) || chi0 < 0) { return result; }
        amrex::Real b2 = 0;
        amrex::Real n2 = 0;
        amrex::Real projection = 0;
        for (int d = 0; d < 3; ++d)
        {
            if (!amrex::Math::isfinite(beta[d]) ||
                !amrex::Math::isfinite(laboratory_direction[d])) { return result; }
            b2 += beta[d]*beta[d];
            n2 += laboratory_direction[d]*laboratory_direction[d];
            projection += beta[d]*laboratory_direction[d];
        }
        if (!(b2 < 1) || !amrex::Math::isfinite(n2) ||
            std::abs(n2-1) > 256*std::numeric_limits<amrex::Real>::epsilon()) { return result; }
        auto const gamma = 1/std::sqrt(1-b2);
        result.inverse_length = chi0*gamma*(1-projection);
        result.maximum_inverse_length = chi0*gamma*(1+std::sqrt(b2));
        result.valid = amrex::Math::isfinite(result.inverse_length) &&
            amrex::Math::isfinite(result.maximum_inverse_length) &&
            (chi0 == 0 || result.inverse_length > 0);
        return result;
    }
}

#endif
