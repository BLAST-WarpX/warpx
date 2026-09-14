/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_RZSPECULARTRAJECTORY_H_
#define WARPX_RADIATION_RZSPECULARTRAJECTORY_H_

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

namespace warpx::radiation
{
struct RZSpecularTrajectory
{
    amrex::GpuArray<amrex::Real, 2> position{};
    // Row-major Cartesian reflection product, also applied to pending carry.
    amrex::GpuArray<amrex::Real, 4> reflection{1, 0, 0, 1};
    int collisions = 0;
    bool valid = false;
};

/** Specular continuation of a straight transverse drift inside a circle.
 * Input endpoints are Cartesian; the trial endpoint may lie outside.
 * Each remaining segment reflects about its actual collision normal.
 * No velocity, particle, or ledger is mutated. More than max_collisions
 * rejects; the caller must not commit the partial candidate.
 */
[[nodiscard]] AMREX_GPU_HOST_DEVICE inline RZSpecularTrajectory
ReflectRZTrajectory (amrex::GpuArray<amrex::Real, 2> const& previous,
                     amrex::GpuArray<amrex::Real, 2> const& trial, amrex::Real radius,
                     int max_collisions = 8) noexcept
{
    RZSpecularTrajectory result;
    if (!(radius > 0) || !std::isfinite(radius) || max_collisions < 0)
    {
        return result;
    }
    amrex::GpuArray<amrex::Real, 2> start{}, end{};
    for (int d = 0; d < 2; ++d)
    {
        start[d] = previous[d] / radius;
        end[d] = trial[d] / radius;
        if (!std::isfinite(start[d]) || !std::isfinite(end[d]))
        {
            return result;
        }
    }
    amrex::Real const rounding = 64 * std::numeric_limits<amrex::Real>::epsilon();
    if (start[0] * start[0] + start[1] * start[1] > 1 + rounding)
    {
        return result;
    }
    for (;;)
    {
        amrex::Real const end_square = end[0] * end[0] + end[1] * end[1];
        if (!std::isfinite(end_square))
        {
            return result;
        }
        if (end_square <= 1 + rounding)
        {
            // Only an explicitly bounded surface-rounding correction.
            amrex::Real const scale = end_square > 1 ? 1 / std::sqrt(end_square) : 1;
            result.position = {radius * end[0] * scale, radius * end[1] * scale};
            result.valid = true;
            return result;
        }
        if (result.collisions == max_collisions)
        {
            return result;
        }
        amrex::Real const dx = end[0] - start[0], dy = end[1] - start[1];
        amrex::Real const a = dx * dx + dy * dy;
        amrex::Real const b = start[0] * dx + start[1] * dy;
        amrex::Real const raw_c = start[0] * start[0] + start[1] * start[1] - 1;
        amrex::Real const c = raw_c > 0 && raw_c <= rounding ? 0 : raw_c;
        amrex::Real const discriminant = b * b - a * c;
        if (!(a > 0) || !(discriminant >= 0) || !std::isfinite(discriminant))
        {
            return result;
        }
        amrex::Real const root = std::sqrt(discriminant);
        amrex::Real const fraction = b >= 0 ? -c / (b + root) : (root - b) / a;
        if (fraction < 0 || fraction > 1 || !std::isfinite(fraction))
        {
            return result;
        }
        amrex::Real const hx = start[0] + fraction * dx, hy = start[1] + fraction * dy;
        amrex::Real const norm = std::sqrt(hx * hx + hy * hy);
        if (!(norm > 0))
        {
            return result;
        }
        amrex::Real const nx = hx / norm, ny = hy / norm;
        amrex::Real const rx = (1 - fraction) * dx, ry = (1 - fraction) * dy;
        amrex::Real const projection = nx * rx + ny * ry;
        start = {nx, ny};
        end = {nx + rx - 2 * projection * nx, ny + ry - 2 * projection * ny};
        auto const old = result.reflection;
        for (int column = 0; column < 2; ++column)
        {
            amrex::Real const product = nx * old[column] + ny * old[2 + column];
            result.reflection[column] = old[column] - 2 * nx * product;
            result.reflection[2 + column] = old[2 + column] - 2 * ny * product;
        }
        ++result.collisions;
    }
}
} // namespace warpx::radiation
#endif
