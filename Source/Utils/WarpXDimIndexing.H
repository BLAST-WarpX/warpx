/* Copyright 2026 Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_UTILS_WARPXDIMINDEXING_H_
#define WARPX_UTILS_WARPXDIMINDEXING_H_

#include "Utils/WarpXDim.H"

#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

namespace warpx
{
    /** Whether fields use the unified (degenerate-3D) index layout
     *
     * In unified builds (WARPX_DIM_RUNTIME, AMReX built with
     * AMREX_SPACEDIM=3), fields of all dimensionalities are stored in 3D
     * MultiFabs with collapsed dimensions of extent 1: 1D uses (1,1,nz), 2D
     * and RZ use (nx,1,nz), so that z is always at index 2. In per-dimension
     * builds (legacy), fields use the compact layout of the corresponding
     * AMReX dimensionality, e.g., z at index 0 (1D), 1 (2D, RZ) or 2 (3D).
     */
#if defined(WARPX_DIM_RUNTIME)
    inline constexpr bool unified_indexing = true;
#else
    inline constexpr bool unified_indexing = false;
#endif

    /** Array4/IntVect slot of each logical axis
     *
     * Maps the logical axes x (or r), y and z to their index slot in field
     * arrays, replacing the WARPX_ZINDEX macro. Axes absent from the grid of
     * dimensionality D use slot -1 in legacy builds (never dereferenced);
     * in unified builds, every axis has a slot and absent axes have extent 1.
     */
    template <Dim D>
    struct IdxMap
    {
        static constexpr int x = unified_indexing ? 0 : (has_x<D> ? 0 : -1);
        static constexpr int y = unified_indexing ? 1 : (D == Dim::XYZ ? 1 : -1);
        static constexpr int z = unified_indexing ? 2 :
            (D == Dim::Z ? 0 :
            (D == Dim::XZ || D == Dim::RZ ? 1 :
            (D == Dim::XYZ ? 2 : -1)));
    };

    /** Access a field array at the logical index (ix,iy,iz)
     *
     * Collapses the indices of absent axes, so that one dimension-templated
     * kernel body produces the compact indexing of today's per-dimension
     * code in legacy builds, e.g., a(ix,iz,0) in 2D, and degenerate-3D
     * indexing a(ix,iy,iz) in unified builds. The index of an absent axis
     * must be 0.
     */
    template <Dim D, typename T>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T&
    field_at (amrex::Array4<T> const& a, int const ix, int const iy, int const iz,
              int const comp = 0)
    {
        if constexpr (unified_indexing)
        {
            return a(ix, iy, iz, comp);
        }
        else
        {
            int idx[3] = {0, 0, 0};
            if constexpr (IdxMap<D>::x >= 0) { idx[IdxMap<D>::x] = ix; }
            if constexpr (IdxMap<D>::y >= 0) { idx[IdxMap<D>::y] = iy; }
            if constexpr (IdxMap<D>::z >= 0) { idx[IdxMap<D>::z] = iz; }
            return a(idx[0], idx[1], idx[2], comp);
        }
    }

    /** Map an array-layout amrex::Dim3 (e.g., a box lower bound) into logical (x,y,z) space
     *
     * Components of absent axes are set to 0.
     */
    template <Dim D>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    constexpr amrex::Dim3
    logical_lo (amrex::Dim3 const& lo)
    {
        auto const get = [&lo] (int const slot) constexpr {
            return slot == 0 ? lo.x : (slot == 1 ? lo.y : (slot == 2 ? lo.z : 0));
        };
        return {get(IdxMap<D>::x), get(IdxMap<D>::y), get(IdxMap<D>::z)};
    }

    // Sanity checks of the index mappings, in both legacy and unified layouts
    static_assert(IdxMap<Dim::XYZ>::x == 0 && IdxMap<Dim::XYZ>::y == 1 && IdxMap<Dim::XYZ>::z == 2);
    static_assert(IdxMap<Dim::XZ>::x == 0 && IdxMap<Dim::XZ>::z == (unified_indexing ? 2 : 1));
    static_assert(IdxMap<Dim::RZ>::x == 0 && IdxMap<Dim::RZ>::z == (unified_indexing ? 2 : 1));
    static_assert(IdxMap<Dim::Z>::z == (unified_indexing ? 2 : 0));
    static_assert(IdxMap<Dim::XZ>::y == (unified_indexing ? 1 : -1));
    static_assert(IdxMap<Dim::Z>::x == (unified_indexing ? 0 : -1));
    static_assert(IdxMap<Dim::RCylinder>::x == 0 && IdxMap<Dim::RSphere>::x == 0);
    static_assert(IdxMap<Dim::RCylinder>::z == (unified_indexing ? 2 : -1));
} // namespace warpx

#endif // WARPX_UTILS_WARPXDIMINDEXING_H_
