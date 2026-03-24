/* Copyright 2019-2025 Maxence Thevenet, Remi Lehe, Alexander Sinn
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

/* Table-driven shape factor evaluation using Horner's method.
 * Polynomial coefficients are stored in constant memory for fast GPU access.
 * Inspired by Hi-PACE++ (https://github.com/Hi-PACE/hipace/pull/1328).
 */
#ifndef WARPX_SHAPEFACTORSTABLEDRIVEN_H_
#define WARPX_SHAPEFACTORSTABLEDRIVEN_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Tuple.H>
#include <AMReX_Array.H>
#include <AMReX_Extension.H>

#include <cmath>

/** Polynomial coefficients for Horner evaluation of shape factors.
 *  For each order N, there are N+1 segments, each with N+1 coefficients.
 *  Coefficients are ordered for Horner evaluation: s[0]*x^N + s[1]*x^(N-1) + ... + s[N],
 *  evaluated as res = (...((s[0]*x + s[1])*x + s[2])*x + ... + s[N]).
 *
 *  All orders use a unified xint in [0,1) convention:
 *    - Even orders: xmid += 0.5 before floor
 *    - Odd orders: xmid used directly before floor
 *  This differs from WarpX's original convention where even-order xint is in [-0.5, 0.5).
 *
 *  Uses double precision to match WarpX's requirement for accurate deposition
 *  when particles move small distances.
 */
inline static constexpr amrex::GpuTuple<
    amrex::GpuArray<amrex::GpuArray<double, 1>, 1>,    // order 0: 1 segment,  1 coeff
    amrex::GpuArray<amrex::GpuArray<double, 2>, 2>,    // order 1: 2 segments, 2 coeffs
    amrex::GpuArray<amrex::GpuArray<double, 3>, 3>,    // order 2: 3 segments, 3 coeffs
    amrex::GpuArray<amrex::GpuArray<double, 4>, 4>,    // order 3: 4 segments, 4 coeffs
    amrex::GpuArray<amrex::GpuArray<double, 5>, 5>     // order 4: 5 segments, 5 coeffs
> depos_coefficients {
    // Order 0: S_0 = 1
    {{
        {       1.}
    }},
    // Order 1: S_0 = 1-t, S_1 = t
    {{
        {      -1.,       1.},
        {       1.,       0.}
    }},
    // Order 2: S_0 = (1-t)^2/2, S_1 = -t^2+t+1/2, S_2 = t^2/2
    {{
        {    1./2.,      -1.,    1./2.},
        {      -1.,       1.,    1./2.},
        {    1./2.,       0.,       0.}
    }},
    // Order 3: S_0 = (1-t)^3/6, S_1 = t^3/2-t^2+2/3,
    //          S_2 = -t^3/2+t^2/2+t/2+1/6, S_3 = t^3/6
    {{
        {   -1./6.,    1./2.,   -1./2.,    1./6.},
        {    1./2.,      -1.,       0.,    2./3.},
        {   -1./2.,    1./2.,    1./2.,    1./6.},
        {    1./6.,       0.,       0.,       0.}
    }},
    // Order 4: derived from WarpX order-4 formulas with t = xint_original + 0.5
    {{
        {   1./24.,   -1./6.,    1./4.,   -1./6.,   1./24.},
        {  -1./6.,     1./2.,   -1./4.,   -1./2.,  11./24.},
        {    1./4.,   -1./2.,   -1./4.,    1./2.,  11./24.},
        {  -1./6.,     1./6.,    1./4.,    1./6.,   1./24.},
        {   1./24.,       0.,       0.,       0.,       0.}
    }}
};

#ifdef AMREX_USE_CUDA
AMREX_GPU_CONSTANT inline static constexpr auto depos_coefficients_g = depos_coefficients;
#endif

/** Accessor for depos_coefficients that uses GPU constant memory on CUDA devices. */
template<int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto get_depos_coefficients (int ix)
{
#ifdef AMREX_USE_CUDA
    AMREX_IF_ON_DEVICE((return amrex::get<depos_order>(depos_coefficients_g)[ix];))
    AMREX_IF_ON_HOST((return amrex::get<depos_order>(depos_coefficients)[ix];))
#else
    return amrex::get<depos_order>(depos_coefficients)[ix];
#endif
}

/** Precompute the fractional position xint and leftmost cell index j_base
 *  for a given normalized particle position xmid.
 *
 *  \tparam depos_order deposition order (0 to 4)
 *  \param[in]  xmid   particle position in grid units (cell-centered or node-centered)
 *  \param[out] xint   fractional position in [0,1) for Horner evaluation
 *  \param[out] j_base index of the leftmost cell where the particle deposits
 */
template<int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void compute_xint_and_jbase (double xmid, double& xint, int& j_base) noexcept
{
    if constexpr (depos_order % 2 == 0) {
        xmid += 0.5;
    }
    const double xfloor = std::floor(xmid);
    j_base = static_cast<int>(xfloor) - depos_order / 2;
    xint = xmid - xfloor;
}

/** Evaluate a single shape factor element using Horner's method.
 *
 *  \tparam depos_order deposition order (0 to 4)
 *  \param[in] xint fractional position in [0,1), as computed by compute_xint_and_jbase
 *  \param[in] ix   index of the shape factor element (0 to depos_order)
 *  \return    the shape factor value (in double precision)
 */
template<int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double eval_shape_factor (double xint, int ix) noexcept
{
    auto const s = get_depos_coefficients<depos_order>(ix);
    double res = s[0];
    for (int i = 1; i <= depos_order; ++i) {
        res = res * xint + s[i];
    }
    return res;
}

#endif // WARPX_SHAPEFACTORSTABLEDRIVEN_H_
