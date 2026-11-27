/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CHARGE_CONTINUITY_ROUNDOFF_H_
#define WARPX_CHARGE_CONTINUITY_ROUNDOFF_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <limits>

namespace warpx::hybrid
{
    /** Backward-error check for cancellation of the six Cartesian face fluxes.
     * Both arguments have charge-density units. The transfer is dt times the
     * sum of absolute face fluxes per control volume, not the signed divergence.
     * A purely relative endpoint check is singular when both endpoints are
     * vacuum: any nonzero rounded divergence then has relative error one.
     *
     * 64 epsilon is a conservative allowance for face interpolation, metric
     * scaling and the short divergence sum. It is NOT a density floor or an
     * adjustable physics tolerance. The original relative check remains in use;
     * this supplementary check neither modifies rho nor corrects any flux.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool chargeFluxRoundoffConsistent (
        amrex::Real const absolute_error, amrex::Real const absolute_transfer) noexcept
    {
        return amrex::Math::isfinite(absolute_error)
            && amrex::Math::isfinite(absolute_transfer)
            && absolute_error >= amrex::Real(0)
            && absolute_transfer >= amrex::Real(0)
            && absolute_error <= amrex::Real(64)
                * std::numeric_limits<amrex::Real>::epsilon() * absolute_transfer;
    }
}
#endif
