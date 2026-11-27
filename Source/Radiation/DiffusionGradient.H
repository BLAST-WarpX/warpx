/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_DIFFUSIONGRADIENT_H_
#define WARPX_RADIATION_DIFFUSIONGRADIENT_H_

#include "CellVolume.H"

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace warpx::radiation
{
    using namespace amrex::literals;

    template <bool CellIntegrated> struct DiffusionDensityView
    {
        amrex::Array4<amrex::Real const> data;
        int group;
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx;
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lower;
        amrex::Dim3 domain_lo;

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
        operator()(int i, int j, int k) const noexcept
        {
            if constexpr (!CellIntegrated) {
                return data(i, j, k, group);
            } else {
                return data(i, j, k, group) / CellVolume(i, dx, lower, domain_lo);
            }
        }
    };

    /** Gradient at an interior face: two-point normal derivative plus transverse
     * derivatives of the face-averaged density. Physical transverse boundaries
     * use one-sided derivatives, never unfilled physical guard cells. */
    template <typename Density, unsigned int PeriodicDimensions>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    FaceGradientMagnitude (Density const& density, int i, int j, int k, int direction, int side,
                           amrex::Real normal_gradient,
                           amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
                           amrex::GpuArray<int, PeriodicDimensions> const& periodic,
                           amrex::Dim3 const& domain_lo, amrex::Dim3 const& domain_hi) noexcept
    {
        static_assert(PeriodicDimensions >= AMREX_SPACEDIM);
        amrex::Real magnitude = std::abs(normal_gradient);
        amrex::GpuArray<int, AMREX_SPACEDIM> const cell{AMREX_D_DECL(i, j, k)};
        amrex::GpuArray<int, AMREX_SPACEDIM> const lo{
            AMREX_D_DECL(domain_lo.x, domain_lo.y, domain_lo.z)};
        amrex::GpuArray<int, AMREX_SPACEDIM> const hi{
            AMREX_D_DECL(domain_hi.x, domain_hi.y, domain_hi.z)};
        int const ni = i + (direction == 0 ? side : 0);
        int const nj = j + (direction == 1 ? side : 0);
        int const nk = k + (direction == 2 ? side : 0);
        for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
            if (axis == direction) {
                continue;
            }
            int const low = periodic[axis] != 0 || cell[axis] > lo[axis] ? -1 : 0;
            int const high = periodic[axis] != 0 || cell[axis] < hi[axis] ? 1 : 0;
            if (low == high) {
                continue;
            }
            int const li = axis == 0 ? low : 0, lj = axis == 1 ? low : 0, lk = axis == 2 ? low : 0;
            int const ui = axis == 0 ? high : 0, uj = axis == 1 ? high : 0,
                      uk = axis == 2 ? high : 0;
            amrex::Real const lower_face = 0.5_rt * density(i + li, j + lj, k + lk) +
                                           0.5_rt * density(ni + li, nj + lj, nk + lk);
            amrex::Real const upper_face = 0.5_rt * density(i + ui, j + uj, k + uk) +
                                           0.5_rt * density(ni + ui, nj + uj, nk + uk);
            amrex::Real const transverse = (upper_face - lower_face) / ((high - low) * dx[axis]);
            magnitude = std::hypot(magnitude, transverse);
        }
        return magnitude;
    }
} // namespace warpx::radiation
#endif
