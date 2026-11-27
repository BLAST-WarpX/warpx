/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include "RadialFaceMarching.H"

namespace warpx::radiation
{
struct RZFaceMarchResult
{
    amrex::ParticleReal distance;
    int next_radial_cell;
    int next_axial_cell;
    bool crossed;
    bool outside_domain;
    bool valid_input;
};

/** Nearest cylindrical or axial cell face along a straight photon path.
 * The axis is not a face. Exact departures have zero length and change
 * ownership once; simultaneous corner crossings change both indices.
 * Periodic z crossings retain the adjacent ghost-cell index until the
 * caller's ordinary particle redistribution wraps the final position.
 * Runtime substeps must span at most one mesh spacing in any direction.
 */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE RZFaceMarchResult
FindNextRZFace (amrex::ParticleReal const x, amrex::ParticleReal const y,
                amrex::ParticleReal const z, amrex::ParticleReal const nx,
                amrex::ParticleReal const ny, amrex::ParticleReal const nz, int const radial_cell,
                int const axial_cell, int const radial_lo, int const radial_hi, int const axial_lo,
                int const axial_hi, amrex::ParticleReal const prob_rlo,
                amrex::ParticleReal const prob_zlo, amrex::ParticleReal const dr,
                amrex::ParticleReal const dz, bool const periodic_z,
                amrex::ParticleReal const max_distance) noexcept
{
    RZFaceMarchResult result{};
    result.next_radial_cell = radial_cell;
    result.next_axial_cell = axial_cell;
    auto const radial = FindNextRadialFace(x, y, nx, ny, nz, radial_cell, radial_lo, radial_hi,
                                           prob_rlo, dr, max_distance);
    if (!radial.valid_input || !amrex::Math::isfinite(z) || !amrex::Math::isfinite(prob_zlo) ||
        !amrex::Math::isfinite(dz) || !(dz > amrex::ParticleReal(0)) || axial_lo > axial_hi ||
        axial_cell < axial_lo - static_cast<int>(periodic_z) ||
        axial_cell > axial_hi + static_cast<int>(periodic_z))
    {
        return result;
    }
    result.distance = radial.crossed ? radial.distance : max_distance;
    amrex::ParticleReal axial_distance = max_distance;
    bool axial_crossed = false;
    if (nz != amrex::ParticleReal(0))
    {
        int const side = nz > amrex::ParticleReal(0) ? 1 : -1;
        amrex::ParticleReal const offset =
            static_cast<amrex::ParticleReal>(axial_cell) -
            static_cast<amrex::ParticleReal>(axial_lo) +
            (side > 0 ? amrex::ParticleReal(1) : amrex::ParticleReal(0));
        amrex::ParticleReal const face = prob_zlo + offset * dz;
        amrex::ParticleReal const signed_gap = (face - z) * static_cast<amrex::ParticleReal>(side);
        amrex::ParticleReal const tolerance =
            amrex::ParticleReal(64) * detail::PrecisionEpsilon() *
            amrex::max(amrex::Math::abs(face), amrex::max(amrex::Math::abs(z), dz));
        if (!amrex::Math::isfinite(face) || !amrex::Math::isfinite(signed_gap) ||
            signed_gap < -tolerance)
        {
            return result;
        }
        amrex::ParticleReal const gap = amrex::max(amrex::ParticleReal(0), signed_gap);
        // Test the bounded displacement before division: an almost
        // transverse ray can have a finite gap/nz quotient that would
        // overflow even though no axial face is reachable this substep.
        axial_crossed = gap <= amrex::Math::abs(nz) * max_distance;
        if (axial_crossed)
        {
            axial_distance = gap / amrex::Math::abs(nz);
            result.distance = amrex::min(result.distance, axial_distance);
        }
    }
    amrex::ParticleReal const crossing_tolerance =
        amrex::ParticleReal(8) * detail::PrecisionEpsilon() * max_distance;
    if (radial.crossed && amrex::Math::abs(radial.distance - result.distance) <= crossing_tolerance)
    {
        result.next_radial_cell = radial.next_radial_cell;
        result.crossed = true;
        result.outside_domain = radial.outside_domain;
    }
    if (axial_crossed && amrex::Math::abs(axial_distance - result.distance) <= crossing_tolerance)
    {
        result.next_axial_cell += nz > amrex::ParticleReal(0) ? 1 : -1;
        result.crossed = true;
        result.outside_domain =
            result.outside_domain || (!periodic_z && (result.next_axial_cell < axial_lo ||
                                                      result.next_axial_cell > axial_hi));
    }
    result.valid_input = true;
    return result;
}
} // namespace warpx::radiation
