/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>

namespace warpx::hybrid
{
using namespace amrex::literals;

/** Metric-weighted divergence of a face flux density. */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real metricFluxDivergence (
    amrex::Real const flux_lo,
    amrex::Real const flux_hi,
    amrex::Real const metric_lo,
    amrex::Real const metric_hi,
    amrex::Real const inverse_volume) noexcept
{
    return (flux_hi * metric_hi - flux_lo * metric_lo) * inverse_volume;
}

/** Full cylindrical nodal control volume used by inverse-volume-scaled
 * charge and current deposits. */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cylindricalTransportNodeVolume (
    amrex::Real const radius,
    amrex::Real const dr,
    amrex::Real const dz,
    amrex::Real const axis_volume_factor = 1.0_rt / 3.0_rt) noexcept
{
    amrex::Real const r_lo = std::max(0.0_rt, radius - 0.5_rt * dr);
    amrex::Real const r_hi = radius + 0.5_rt * dr;
    constexpr amrex::Real pi = 3.14159265358979323846_rt;
    if (radius == 0.0_rt) {
        return pi * dr * dr * axis_volume_factor * dz;
    }
    return pi * (r_hi * r_hi - r_lo * r_lo) * dz;
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cylindricalRadialFaceArea (
    amrex::Real const radius, amrex::Real const dz) noexcept
{
    constexpr amrex::Real pi = 3.14159265358979323846_rt;
    return 2.0_rt * pi * radius * dz;
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cylindricalAxialFaceArea (
    amrex::Real const r_lo, amrex::Real const r_hi) noexcept
{
    constexpr amrex::Real pi = 3.14159265358979323846_rt;
    return pi * (r_hi * r_hi - r_lo * r_lo);
}

/** Interpolate radial velocity by making r*V linear. This preserves both a
 * constant field and the divergence-free field V=C/r at a cell face. */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cylindricalRadialFaceVelocity (
    amrex::Real const velocity_lo,
    amrex::Real const velocity_hi,
    amrex::Real const radius_lo,
    amrex::Real const radius_hi,
    amrex::Real const face_radius) noexcept
{
    return 0.5_rt
        * (radius_lo * velocity_lo + radius_hi * velocity_hi)
        / face_radius;
}

} // namespace warpx::hybrid
