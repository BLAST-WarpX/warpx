/* Copyright 2022-2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_MATH_LINEAR_INTERPOLATION_H_
#define ABLASTR_MATH_LINEAR_INTERPOLATION_H_

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

namespace ablastr::math
{
    /** \brief Performs a linear interpolation
     *
     * Performs a linear interpolation at x given the 2 points
     * (x0, f0) and (x1, f1)
     */
    template<typename TCoord, typename TVal>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    constexpr auto linear_interp(
        TCoord x0, TCoord x1,
        TVal f0, TVal f1,
        TCoord x)
    {
        return ((x1-x)*f0 + (x-x0)*f1)/(x1-x0);
    }

    /** \brief Performs a bilinear interpolation
     *
     * Performs a bilinear interpolation at (x,y) given the 4 points
     * (x0, y0, f00), (x0, y1, f01), (x1, y0, f10), (x1, y1, f11).
     */
    template<typename TCoord, typename TVal>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    constexpr auto bilinear_interp(
        TCoord x0, TCoord x1, TCoord y0, TCoord y1,
        TVal f00, TVal f01, TVal f10, TVal f11,
        TCoord x, TCoord y)
    {
        const auto fx0 = linear_interp(x0, x1, f00, f10, x);
        const auto fx1 = linear_interp(x0, x1, f01, f11, x);
        return linear_interp(y0, y1, fx0, fx1, y);
    }

    /** \brief Performs a trilinear interpolation
     *
     * Performs a trilinear interpolation at (x,y,z) given the 8 points
     * (x0, y0, z0, f000), (x0, y0, z1, f001), (x0, y1, z0, f010), (x0, y1, z1, f011),
     * (x1, y0, z0, f100), (x1, y0, z1, f101), (x1, y1, z0, f110), (x1, y1, z1, f111)
     */
    template<typename TCoord, typename TVal>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    constexpr auto trilinear_interp(
        TCoord x0, TCoord x1, TCoord y0, TCoord y1, TCoord z0, TCoord z1,
        TVal f000, TVal f001, TVal f010, TVal f011, TVal f100, TVal f101, TVal f110, TVal f111,
        TCoord x, TCoord y, TCoord z)
    {
        const auto fxy0 = bilinear_interp(
            x0, x1, y0, y1, f000, f010, f100, f110, x, y);
        const auto fxy1 = bilinear_interp(
            x0, x1, y0, y1, f001, f011, f101, f111, x, y);
        return linear_interp(z0, z1, fxy0, fxy1, z);
    }
}

#endif //ABLASTR_MATH_LINEAR_INTERPOLATION_H_
