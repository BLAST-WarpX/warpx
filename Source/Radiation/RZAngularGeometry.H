/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_RZANGULARGEOMETRY_H_
#define WARPX_RADIATION_RZANGULARGEOMETRY_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace warpx::radiation
{
/** Angular-conservative radial face weights for a valid annular cell.
 * Radiation stores integrated Qtheta=Ftheta/c. Its discrete axial angular
 * inventory is mean_radius*Qtheta/c, not Qtheta/c. The caller supplies physical
 * face radii/areas with 0 <= lower_radius < upper_radius. No material torque
 * assignment or native swirl capability is implied by this geometry helper.
 */
struct RZAngularGeometry
{
    amrex::Real mean_radius;
    amrex::Real lower_weight;
    amrex::Real upper_weight;
    amrex::Real lower_geometric_weight;
    amrex::Real upper_geometric_weight;

    AMREX_GPU_HOST_DEVICE
    RZAngularGeometry (amrex::Real lower_radius, amrex::Real upper_radius,
                       amrex::Real lower_area, amrex::Real upper_area) noexcept
        : mean_radius(MeanRadius(lower_radius, upper_radius)),
          lower_weight(lower_area * (lower_radius / mean_radius)),
          upper_weight(upper_area * (upper_radius / mean_radius))
    {
        amrex::Real const midpoint = (lower_radius + upper_radius) / 2;
        amrex::Real const width = upper_radius - lower_radius;
        amrex::Real const offset = width * width / (12 * midpoint);
        // Direct local geometry, without subtracting globally accumulated
        // flux divergences. The offsets also avoid cancellation at large r/dr.
        lower_geometric_weight = lower_area * ((width / 2 + offset) / mean_radius);
        upper_geometric_weight = upper_area * ((-width / 2 + offset) / mean_radius);
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE static amrex::Real
    MeanRadius (amrex::Real lower_radius, amrex::Real upper_radius) noexcept
    {
        amrex::Real const midpoint = (lower_radius + upper_radius) / 2;
        amrex::Real const width = upper_radius - lower_radius;
        return midpoint + width * width / (12 * midpoint);
    }

    /** dt*divergence of the angular-conservative theta row, in Qtheta units. */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE amrex::Real
    Divergence (amrex::Real lower_flux, amrex::Real upper_flux, amrex::Real dt) const noexcept
    {
        return dt * (upper_weight * upper_flux - lower_weight * lower_flux);
    }

    /** Gtheta = ordinary theta divergence - angular-conservative divergence.
     * The caloric projected row E-beta.Q receives +beta_theta*Gtheta. */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE amrex::Real
    Geometric (amrex::Real lower_flux, amrex::Real upper_flux, amrex::Real dt) const noexcept
    {
        return dt * (upper_geometric_weight * upper_flux - lower_geometric_weight * lower_flux);
    }

    /** Uncancelled angular face operands for arithmetic error conditioning. */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE amrex::Real
    Magnitude (amrex::Real lower_flux, amrex::Real upper_flux, amrex::Real dt) const noexcept
    {
        return std::abs(dt) * (std::abs(upper_weight * upper_flux) +
                               std::abs(lower_weight * lower_flux));
    }
};
} // namespace warpx::radiation
#endif
