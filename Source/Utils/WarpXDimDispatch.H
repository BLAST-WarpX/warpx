/* Copyright 2026 Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_UTILS_WARPXDIMDISPATCH_H_
#define WARPX_UTILS_WARPXDIMDISPATCH_H_

#include "Utils/WarpXDim.H"

#include <AMReX.H>

#include <type_traits>
#include <utility>

namespace warpx
{
    /** Dispatch the runtime dimensionality to a compile-time Dim
     *
     * Calls f with a std::integral_constant<Dim, D> matching the runtime
     * dimension, so that f can instantiate Dim-templated kernels. This
     * mirrors the runtime particle-shape-order dispatch, e.g., in
     * WarpXParticleContainer::DepositCurrent.
     *
     * In per-dimension builds (WARPX_DIM_* defined), only the compiled
     * dimension is instantiated and any other runtime dimension aborts. In
     * unified builds (WARPX_DIM_RUNTIME), all supported dimensions are
     * instantiated and selected at runtime.
     *
     * Note: f must not *define* extended device lambdas itself (an nvcc
     * restriction on generic lambdas); it instead calls named function
     * templates, which may.
     *
     * The dispatched dimensions are currently limited to the ones supported
     * by the runtime-dimensionality prototype (1D, 2D, 3D); more are added as
     * the migration proceeds.
     *
     * @param dim the runtime dimensionality
     * @param f a callable taking a std::integral_constant<Dim, D>
     * @return the value returned by f
     */
    template <typename F>
    decltype(auto)
    dim_dispatch (Dim const dim, F&& f)
    {
        if constexpr (CompileDim != Dim::Runtime)
        {
            if (dim != CompileDim) {
                amrex::Abort("dim_dispatch: this binary was compiled for geometry.dims = "
                             + geometry_dims_name(CompileDim)
                             + ", but the runtime dimensionality is geometry.dims = "
                             + geometry_dims_name(dim));
            }
            return std::forward<F>(f)(std::integral_constant<Dim, CompileDim>{});
        }
        else
        {
            if (dim == Dim::Z)  { return std::forward<F>(f)(std::integral_constant<Dim, Dim::Z>{}); }
            if (dim == Dim::XZ) { return std::forward<F>(f)(std::integral_constant<Dim, Dim::XZ>{}); }
            if (dim != Dim::XYZ) {
                amrex::Abort("dim_dispatch: geometry.dims = " + geometry_dims_name(dim)
                             + " is not compiled into this binary");
            }
            return std::forward<F>(f)(std::integral_constant<Dim, Dim::XYZ>{});
        }
    }
} // namespace warpx

#endif // WARPX_UTILS_WARPXDIMDISPATCH_H_
