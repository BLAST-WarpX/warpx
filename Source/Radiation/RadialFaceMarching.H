/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_RADIALFACEMARCHING_H_
#define WARPX_RADIATION_RADIALFACEMARCHING_H_

#include <AMReX_Algorithm.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

namespace warpx::radiation
{
    /** Result of one cylindrical radial-face query.
     *
     * Callers must check `valid_input` before using any other field.  A false
     * value means that the query rejected its geometry or direction and did
     * not produce a transport handoff.
     */
    struct RadialFaceMarchResult
    {
        amrex::ParticleReal distance;
        int next_radial_cell;
        bool crossed;
        bool outside_domain;
        bool valid_input;
    };

    namespace detail
    {
        struct RadialFaceRoots
        {
            amrex::ParticleReal first;
            amrex::ParticleReal second;
            amrex::ParticleReal derivative_tolerance;
            bool has_roots;
        };

        [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::ParticleReal PrecisionEpsilon () noexcept
        {
            return amrex::max(
                static_cast<amrex::ParticleReal>(
                    std::numeric_limits<amrex::Real>::epsilon()),
                static_cast<amrex::ParticleReal>(
                    std::numeric_limits<amrex::ParticleReal>::epsilon()));
        }

        [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::ParticleReal RadialLengthScale (
            amrex::ParticleReal const x,
            amrex::ParticleReal const y,
            amrex::ParticleReal const face_radius,
            amrex::ParticleReal const cell_width) noexcept
        {
            amrex::ParticleReal scale = amrex::Math::abs(x);
            scale = amrex::max(scale, amrex::Math::abs(y));
            scale = amrex::max(scale, amrex::Math::abs(face_radius));
            scale = amrex::max(scale, amrex::Math::abs(cell_width));
            return scale;
        }

        [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        RadialFaceRoots RadialCircleRoots (
            amrex::ParticleReal const x,
            amrex::ParticleReal const y,
            amrex::ParticleReal const nx,
            amrex::ParticleReal const ny,
            amrex::ParticleReal const face_radius,
            amrex::ParticleReal const cell_width) noexcept
        {
            RadialFaceRoots result{};
            amrex::ParticleReal const scale = RadialLengthScale(
                x, y, face_radius, cell_width);
            if (!(scale > amrex::ParticleReal(0.0))
                || !amrex::Math::isfinite(scale))
            {
                return result;
            }

            // Normalize the quadratic before forming its discriminant.  This
            // avoids overflow for large physical coordinates while preserving
            // the dimensions of the resulting path distance through `scale`.
            amrex::ParticleReal const xs = x / scale;
            amrex::ParticleReal const ys = y / scale;
            amrex::ParticleReal const rs = face_radius / scale;
            amrex::ParticleReal const nx2 = nx * nx;
            amrex::ParticleReal const ny2 = ny * ny;
            amrex::ParticleReal const a = nx2 + ny2;
            if (!(a > amrex::ParticleReal(0.0))
                || !amrex::Math::isfinite(a))
            {
                return result;
            }

            amrex::ParticleReal const b = xs * nx + ys * ny;
            amrex::ParticleReal const c = xs * xs + ys * ys - rs * rs;
            amrex::ParticleReal const b2 = b * b;
            amrex::ParticleReal const ac = a * c;
            amrex::ParticleReal const discriminant = b2 - ac;
            if (!amrex::Math::isfinite(discriminant)) {
                return result;
            }

            amrex::ParticleReal const epsilon = PrecisionEpsilon();
            // Every term that contributes to the discriminant is scaled by
            // the transverse magnitude.  In particular, do not use rs^2 by
            // itself: that would classify a legitimate near-axial crossing
            // as a tangent when a is small.
            amrex::ParticleReal scale_squared = b2;
            scale_squared = amrex::max(scale_squared, amrex::Math::abs(ac));
            amrex::ParticleReal const position_squared = xs * xs + ys * ys;
            amrex::ParticleReal const geometry_squared = amrex::max(
                position_squared, rs * rs);
            scale_squared = amrex::max(scale_squared, a * geometry_squared);
            amrex::ParticleReal const discriminant_tolerance =
                amrex::ParticleReal(64.0) * epsilon * scale_squared;
            // A zero or near-zero discriminant is a tangent.  It touches a
            // circle but never leaves the current radial interval.
            if (!(discriminant > discriminant_tolerance)) {
                return result;
            }

            amrex::ParticleReal const root_discriminant =
                std::sqrt(discriminant);
            // q = -(b + sign(b)*sqrt(D)) is the cancellation-safe quadratic
            // branch.  The other root is c/q, so one root remains accurate
            // when the direct (-b +/- sqrt(D)) subtraction loses digits.
            amrex::ParticleReal const q = b >= amrex::ParticleReal(0.0)
                ? -b - root_discriminant
                : -b + root_discriminant;
            if (q == amrex::ParticleReal(0.0)
                || !amrex::Math::isfinite(q))
            {
                return result;
            }
            amrex::ParticleReal root_a = q / a;
            amrex::ParticleReal root_b = c / q;
            if (!amrex::Math::isfinite(root_a)
                || !amrex::Math::isfinite(root_b))
            {
                return result;
            }
            if (root_a > root_b) {
                amrex::ParticleReal const temporary = root_a;
                root_a = root_b;
                root_b = temporary;
            }

            result.first = root_a * scale;
            result.second = root_b * scale;
            result.derivative_tolerance = std::sqrt(discriminant_tolerance)
                * scale;
            result.has_roots = amrex::Math::isfinite(result.first)
                && amrex::Math::isfinite(result.second)
                && amrex::Math::isfinite(result.derivative_tolerance);
            return result;
        }

        [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        bool IsLeavingFace (
            amrex::ParticleReal const derivative,
            amrex::ParticleReal const derivative_tolerance,
            bool const is_inner_face) noexcept
        {
            if (amrex::Math::abs(derivative) <= derivative_tolerance) {
                return false;
            }
            return is_inner_face
                ? derivative < amrex::ParticleReal(0.0)
                : derivative > amrex::ParticleReal(0.0);
        }
    }

    /** Find the next positive-area radial face reached by a Cartesian path.
     *
     * The path is `(x,y) + s (nx,ny)` for `0 <= s <= max_distance`, with a
     * normalized Cartesian/axial direction `(nx,ny,nz)`.  The radial cells are
     * `[prob_lo + (i-domain_lo) dr, prob_lo + (i+1-domain_lo) dr)`.
     * The caller must keep `current_radial_cell` consistent with the
     * half-open ownership of `(x,y)` and must apply each returned handoff
     * before querying the next segment.  `outside_domain` reports only that
     * the returned radial index is outside `[domain_lo, domain_hi]`; it does
     * not report box, tile, or MPI-rank ownership.
     * Positive-radius faces are owned by the interval being left according to
     * the sign of the radial derivative at the root.  Consequently a path
     * arriving on a face from the neighboring interval does not repeat that
     * transition, while an exact outward/inward departure returns distance
     * zero.  The radius-zero face is omitted because its cylindrical area is
     * zero and an axis-containing domain has no radial exit there.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RadialFaceMarchResult FindNextRadialFace (
        amrex::ParticleReal const x,
        amrex::ParticleReal const y,
        amrex::ParticleReal const nx,
        amrex::ParticleReal const ny,
        amrex::ParticleReal const nz,
        int const current_radial_cell,
        int const domain_lo,
        int const domain_hi,
        amrex::ParticleReal const prob_lo,
        amrex::ParticleReal const dr,
        amrex::ParticleReal const max_distance) noexcept
    {
        RadialFaceMarchResult result{};
        result.distance = amrex::ParticleReal(0.0);
        result.next_radial_cell = current_radial_cell;

        if (!amrex::Math::isfinite(x) || !amrex::Math::isfinite(y)
            || !amrex::Math::isfinite(nx) || !amrex::Math::isfinite(ny)
            || !amrex::Math::isfinite(nz)
            || !amrex::Math::isfinite(prob_lo)
            || !(prob_lo >= amrex::ParticleReal(0.0))
            || !amrex::Math::isfinite(dr)
            || !(dr > amrex::ParticleReal(0.0))
            || !(max_distance >= amrex::ParticleReal(0.0))
            || !amrex::Math::isfinite(max_distance)
            || current_radial_cell < domain_lo
            || current_radial_cell > domain_hi
            || domain_lo > domain_hi)
        {
            return result;
        }

        amrex::ParticleReal const norm_squared = nx * nx + ny * ny + nz * nz;
        amrex::ParticleReal const epsilon = detail::PrecisionEpsilon();
        amrex::ParticleReal const norm_tolerance =
            amrex::ParticleReal(64.0) * epsilon;
        if (!amrex::Math::isfinite(norm_squared)
            || amrex::Math::abs(norm_squared - amrex::ParticleReal(1.0))
                   > norm_tolerance)
        {
            return result;
        }

        amrex::ParticleReal const radial_cell_offset =
            static_cast<amrex::ParticleReal>(current_radial_cell)
            - static_cast<amrex::ParticleReal>(domain_lo);
        amrex::ParticleReal const inner_radius =
            prob_lo + radial_cell_offset * dr;
        amrex::ParticleReal const outer_radius = inner_radius + dr;
        if (!amrex::Math::isfinite(inner_radius)
            || !amrex::Math::isfinite(outer_radius)
            || !(outer_radius > inner_radius))
        {
            return result;
        }

        result.valid_input = true;
        result.distance = max_distance;

        amrex::ParticleReal const path_scale = detail::RadialLengthScale(
            x, y, outer_radius, dr);
        amrex::ParticleReal const distance_tolerance = path_scale > 0.0
            ? amrex::ParticleReal(64.0) * epsilon * path_scale
            : amrex::ParticleReal(0.0);

        bool have_candidate = false;
        amrex::ParticleReal candidate_distance = result.distance;
        int candidate_step = 0;

        // The zero-radius face is a zero-area coordinate singularity, not a
        // physical radial boundary.  It is intentionally never considered.
        for (int face_number = 0; face_number < 2; ++face_number) {
            bool const is_inner_face = face_number == 0;
            amrex::ParticleReal const face_radius = is_inner_face
                ? inner_radius
                : outer_radius;
            if (!(face_radius > amrex::ParticleReal(0.0))) {
                continue;
            }

            auto const roots = detail::RadialCircleRoots(
                x, y, nx, ny, face_radius, dr);
            if (!roots.has_roots) {
                continue;
            }

            for (int root_number = 0; root_number < 2; ++root_number) {
                amrex::ParticleReal const root = root_number == 0
                    ? roots.first
                    : roots.second;
                amrex::ParticleReal const clamped_root =
                    root < amrex::ParticleReal(0.0)
                    && root >= -distance_tolerance
                    ? amrex::ParticleReal(0.0)
                    : root;
                if (clamped_root < amrex::ParticleReal(0.0)
                    || clamped_root > max_distance)
                {
                    continue;
                }

                // The derivative is evaluated at this root, not inferred
                // from nx.  A line can move inward in x while moving
                // outward in radius after passing its closest approach.
                amrex::ParticleReal const path_x =
                    x + nx * clamped_root;
                amrex::ParticleReal const path_y =
                    y + ny * clamped_root;
                amrex::ParticleReal const derivative =
                    path_x * nx + path_y * ny;
                if (!detail::IsLeavingFace(
                        derivative, roots.derivative_tolerance,
                        is_inner_face))
                {
                    continue;
                }

                int const step = is_inner_face ? -1 : 1;
                if (!have_candidate || clamped_root < candidate_distance) {
                    have_candidate = true;
                    candidate_distance = clamped_root;
                    candidate_step = step;
                }
            }
        }

        if (!have_candidate) {
            return result;
        }

        result.distance = candidate_distance;
        result.next_radial_cell = current_radial_cell + candidate_step;
        result.crossed = true;
        result.outside_domain = result.next_radial_cell < domain_lo
            || result.next_radial_cell > domain_hi;
        return result;
    }
}

#endif // WARPX_RADIATION_RADIALFACEMARCHING_H_
