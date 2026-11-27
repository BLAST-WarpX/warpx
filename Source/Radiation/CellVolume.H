/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_CELLVOLUME_H_
#define WARPX_RADIATION_CELLVOLUME_H_

#include "Utils/WarpXConst.H"

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace warpx::radiation
{
    /** Physical volume of a valid radiation cell; Cartesian inactive lengths
     * are unity. The same measure converts every source and transport term. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    CellVolume (int i, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& lower,
                amrex::Dim3 const& domain_lo) noexcept
    {
        using namespace amrex::literals;
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RSPHERE)
        amrex::Real const lo = lower[0] + (i - domain_lo.x) * dx[0];
        amrex::Real const hi = lo + dx[0];
#if defined(WARPX_DIM_RSPHERE)
        return 4.0_rt / 3.0_rt * MathConst::pi * (hi * hi * hi - lo * lo * lo);
#elif defined(WARPX_DIM_RZ)
        return MathConst::pi * (hi * hi - lo * lo) * dx[1];
#else
        return MathConst::pi * (hi * hi - lo * lo);
#endif
#else
        amrex::ignore_unused(i, lower, domain_lo);
        return AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
#endif
    }
} // namespace warpx::radiation
#endif
