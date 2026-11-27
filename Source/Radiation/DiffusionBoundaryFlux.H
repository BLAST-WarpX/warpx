/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_DIFFUSIONBOUNDARYFLUX_H_
#define WARPX_RADIATION_DIFFUSIONBOUNDARYFLUX_H_

#include "RadiationBoundary.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace warpx::radiation
{
using namespace amrex::literals;

// Explicit-FLD boundary factors. Parsing and accounting stay with the caller.
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
DiffusionEscapeFactor (int const boundary) noexcept
{
    if (boundary == static_cast<int>(
            DiffusionBoundary::Marshak))
    {
        return 0.5_rt;
    }
    if (boundary == static_cast<int>(
            DiffusionBoundary::Vacuum))
    {
        return 1.0_rt;
    }
    return 0.0_rt;
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
DiffusionEscapeMomentumFactor (int const boundary) noexcept
{
    if (boundary == static_cast<int>(
            DiffusionBoundary::Marshak))
    {
        // The P1 half-isotropic boundary has F=cE/2 and normal pressure E/3.
        return 1.0_rt / 3.0_rt;
    }
    if (boundary == static_cast<int>(
            DiffusionBoundary::Vacuum))
    {
        return 1.0_rt;
    }
    return 0.0_rt;
}

/** Authoritative normal current at a scalar diffusion face [W m^-2].
 *
 * Reflecting means exactly zero normal energy current (both diffusive and
 * advective parts). The caller supplies the interior candidate only so the
 * same policy can later be used by non-reflecting escape implementations.
 */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
DiffusionBoundaryNormalEnergyFlux (int const boundary,
                                   amrex::Real const interior_flux) noexcept
{
    return boundary == static_cast<int>(DiffusionBoundary::Reflecting)
               ? amrex::Real(0.0)
               : interior_flux;
}

/** Signed stationary-wall reaction accumulated by the radiation field [kg m/s].
 *
 * `normal_sign` is +1 on a domain-high face and -1 on a domain-low face.
 * A reflecting wall receives `dt*A*P*n`; it is external to material recoil.
 */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
DiffusionBoundaryWallMomentum (int const boundary,
                               amrex::Real const face_pressure,
                               amrex::Real const face_area,
                               amrex::Real const diffusion_dt,
                               int const normal_sign) noexcept
{
    if (boundary != static_cast<int>(DiffusionBoundary::Reflecting))
    {
        return amrex::Real(0.0);
    }
    return normal_sign * face_pressure * face_area * diffusion_dt;
}

} // namespace warpx::radiation

#endif
