/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "EBJBoundary.H"

#include "EmbeddedBoundary/DistanceToEB.H"

#include <ablastr/particles/NodalFieldGather.H>
#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_RealVect.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace amrex;

namespace
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    /** Trilinear (bilinear in 2D) gather of component n of a staggered field
     *  at an arbitrary position (physical coordinates), clamping the stencil
     *  to the array bounds (constant extrapolation past the available ghosts).
     *
     *  TODO(PR#6994): replace with a library interp (ablastr) call. The existing
     *  ablastr::particles::compute_weights / interp_field_nodal pair only supports
     *  a single NODE or CELL centering (one half-cell offset applied to every
     *  direction) and does no bounds clamping, so it cannot express a general
     *  Yee-staggered field (per-direction stag[d]) gathered with the
     *  ghost-clamped, constant-extrapolation behavior this helper requires.
     *  ablastr::math::{bi,tri}linear_interp are pure value kernels that would
     *  still leave the staggered indexing and the clamp hand-rolled here. Adding
     *  a staggered, arbitrary-position, ghost-clamped gather to ablastr is a
     *  follow-up; until then the hand-rolled stencil below stays. */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real gather_staggered (
        amrex::Array4<amrex::Real const> const& a,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& pos,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& stag,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
        int n) noexcept
    {
        using namespace amrex::literals;

        amrex::Real const lx = (pos[0] - plo[0])*dxi[0] - stag[0];
        int i0 = static_cast<int>(std::floor(lx));
        i0 = amrex::min(amrex::max(i0, a.begin[0]), a.end[0] - 2);
        amrex::Real const wx = amrex::min(amrex::max(lx - i0, 0._rt), 1._rt);

#if defined(WARPX_DIM_3D)
        amrex::Real const ly = (pos[1] - plo[1])*dxi[1] - stag[1];
        amrex::Real const lz = (pos[2] - plo[2])*dxi[2] - stag[2];
        int j0 = static_cast<int>(std::floor(ly));
        int k0 = static_cast<int>(std::floor(lz));
        j0 = amrex::min(amrex::max(j0, a.begin[1]), a.end[1] - 2);
        k0 = amrex::min(amrex::max(k0, a.begin[2]), a.end[2] - 2);
        amrex::Real const wy = amrex::min(amrex::max(ly - j0, 0._rt), 1._rt);
        amrex::Real const wz = amrex::min(amrex::max(lz - k0, 0._rt), 1._rt);

        return (1._rt-wx)*(1._rt-wy)*(1._rt-wz)*a(i0  , j0  , k0  , n)
             +        wx *(1._rt-wy)*(1._rt-wz)*a(i0+1, j0  , k0  , n)
             + (1._rt-wx)*       wy *(1._rt-wz)*a(i0  , j0+1, k0  , n)
             +        wx *       wy *(1._rt-wz)*a(i0+1, j0+1, k0  , n)
             + (1._rt-wx)*(1._rt-wy)*       wz *a(i0  , j0  , k0+1, n)
             +        wx *(1._rt-wy)*       wz *a(i0+1, j0  , k0+1, n)
             + (1._rt-wx)*       wy *       wz *a(i0  , j0+1, k0+1, n)
             +        wx *       wy *       wz *a(i0+1, j0+1, k0+1, n);
#else
        amrex::Real const lz = (pos[1] - plo[1])*dxi[1] - stag[1];
        int j0 = static_cast<int>(std::floor(lz));
        j0 = amrex::min(amrex::max(j0, a.begin[1]), a.end[1] - 2);
        amrex::Real const wz = amrex::min(amrex::max(lz - j0, 0._rt), 1._rt);

        return (1._rt-wx)*(1._rt-wz)*a(i0  , j0  , 0, n)
             +        wx *(1._rt-wz)*a(i0+1, j0  , 0, n)
             + (1._rt-wx)*       wz *a(i0  , j0+1, 0, n)
             +        wx *       wz *a(i0+1, j0+1, 0, n);
#endif
    }

    /** Like gather_staggered, but using only stencil points accepted by the
     *  \c fluid predicate: rejected points get zero weight and the result is
     *  renormalized by the remaining weight, which is also returned (zero if
     *  the entire stencil is rejected). */
    template <typename FluidPred>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::GpuTuple<amrex::Real, amrex::Real> gather_staggered_pred (
        amrex::Array4<amrex::Real const> const& a,
        FluidPred const& fluid,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& pos,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& stag,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
        int n) noexcept
    {
        using namespace amrex::literals;

        amrex::Real const lx = (pos[0] - plo[0])*dxi[0] - stag[0];
        int i0 = static_cast<int>(std::floor(lx));
        i0 = amrex::min(amrex::max(i0, a.begin[0]), a.end[0] - 2);
        amrex::Real const wx = amrex::min(amrex::max(lx - i0, 0._rt), 1._rt);

        amrex::Real vsum = 0._rt;
        amrex::Real wsum = 0._rt;
#if defined(WARPX_DIM_3D)
        amrex::Real const ly = (pos[1] - plo[1])*dxi[1] - stag[1];
        amrex::Real const lz = (pos[2] - plo[2])*dxi[2] - stag[2];
        int j0 = static_cast<int>(std::floor(ly));
        int k0 = static_cast<int>(std::floor(lz));
        j0 = amrex::min(amrex::max(j0, a.begin[1]), a.end[1] - 2);
        k0 = amrex::min(amrex::max(k0, a.begin[2]), a.end[2] - 2);
        amrex::Real const wy = amrex::min(amrex::max(ly - j0, 0._rt), 1._rt);
        amrex::Real const wz = amrex::min(amrex::max(lz - k0, 0._rt), 1._rt);

        for (int dk = 0; dk < 2; ++dk) {
            for (int dj = 0; dj < 2; ++dj) {
                for (int di = 0; di < 2; ++di) {
                    amrex::Real const w = (di ? wx : 1._rt-wx)
                                        * (dj ? wy : 1._rt-wy)
                                        * (dk ? wz : 1._rt-wz);
                    if (fluid(i0+di, j0+dj, k0+dk)) {
                        vsum += w*a(i0+di, j0+dj, k0+dk, n);
                        wsum += w;
                    }
                }
            }
        }
#else
        amrex::Real const lz = (pos[1] - plo[1])*dxi[1] - stag[1];
        int j0 = static_cast<int>(std::floor(lz));
        j0 = amrex::min(amrex::max(j0, a.begin[1]), a.end[1] - 2);
        amrex::Real const wz = amrex::min(amrex::max(lz - j0, 0._rt), 1._rt);

        for (int dj = 0; dj < 2; ++dj) {
            for (int di = 0; di < 2; ++di) {
                amrex::Real const w = (di ? wx : 1._rt-wx) * (dj ? wz : 1._rt-wz);
                if (fluid(i0+di, j0+dj, 0)) {
                    vsum += w*a(i0+di, j0+dj, 0, n);
                    wsum += w;
                }
            }
        }
#endif
        amrex::Real const v = (wsum > 0._rt) ? vsum/wsum : 0._rt;
        return {v, wsum};
    }

    /** Predicate-renormalized gather using only unmasked (update mask != 0)
     *  points. */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::GpuTuple<amrex::Real, amrex::Real> gather_staggered_masked (
        amrex::Array4<amrex::Real const> const& a,
        amrex::Array4<int const> const& msk,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& pos,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& stag,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
        int n) noexcept
    {
        return gather_staggered_pred(a,
            [&] (int i, int j, int k) { return msk(i, j, k) != 0; },
            pos, stag, plo, dxi, n);
    }

    // Number of terms in the local quadratic basis used by the 2nd-order gather
    // (1, u, v[, w], u^2, v^2[, w^2], uv[, uw, vw]); u,v,w are per-direction cell
    // offsets, so the basis is naturally non-dimensional and aspect-aware.
#if defined(WARPX_DIM_3D)
    constexpr int QUAD_NB = 10;
#else
    constexpr int QUAD_NB = 6;
#endif

    // Reciprocal-condition gate for the quadratic moving-least-squares solve
    // (the "Phoenix corner-trim"). At a Cartesian staircase cut of a curved
    // wall the surviving fluid taps in the +/-2 window can be nearly collinear,
    // so the normal-equations matrix M = sum p p^T is strictly SPD (it passes
    // the Cholesky positivity test) yet cond(M) >> 1e6. The resulting fit has
    // huge oscillatory tap weights that overshoot the covered-B value and feed
    // an enormous curl(B) -> J -> E, which crashes the stiff RKF45 substepper at
    // step 1. We reject the solve when the Cholesky pivot ratio (dmin/dmax)^2 --
    // a cheap proxy for 1/cond(M) (the true reciprocal condition number of M is
    // bounded by this ratio for an SPD matrix) -- drops below rcond_min, so the
    // caller demotes the degenerate cut-cell to its bounded linear gather. This
    // is the load-bearing fix; the ridge below is only a secondary regularizer.
    // Tighter (smaller) rcond_min trims fewer cells; looser (larger) trims more.
    constexpr amrex::Real rcond_min = 1.e-6_rt;

    /** In-place Cholesky solve of a small symmetric positive-definite system
     *  M x = b (result returned in b). Returns false if M is not SPD, OR if M is
     *  too ill-conditioned (reciprocal-condition proxy below rcond_min); in
     *  either case the caller falls back to the bounded linear gather. A small
     *  ridge (added to the diagonal AFTER the conditioning gate) keeps the solve
     *  of the accepted, well-conditioned systems numerically clean -- it is a
     *  secondary regularizer; the rcond gate is the load-bearing corner-trim.
     *  Fixed size, no allocation -> GPU friendly.
     *
     *  IMPORTANT: the gate must see the RAW (un-ridged) matrix. A ridge added
     *  before factoring inflates the smallest Cholesky pivot toward sqrt(ridge),
     *  which masks the true ill-conditioning of a collinear-tap corner and lets
     *  it slip past the gate -- so the ridge is applied here, internally, only
     *  after the raw conditioning has been measured and accepted. */
    template <int N>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    bool solve_spd (amrex::Real M[N][N], amrex::Real b[N], amrex::Real ridge = 0._rt) noexcept
    {
        using namespace amrex::literals;
        amrex::Real L[N][N];
        for (int i = 0; i < N; ++i) { for (int j = 0; j < N; ++j) { L[i][j] = 0._rt; } }
        // Track the extreme Cholesky pivots so we can gate on conditioning. For
        // an SPD M the pivots L[i][i] are real and positive, and (dmin/dmax)^2
        // lower-bounds 1/cond(M); rejecting on it costs no extra work. This first
        // factorization is of the RAW matrix so the gate is not fooled by the
        // ridge. (N is tiny -- 6 in 2D, 10 in 3D -- so the second, ridged
        // factorization below for the accepted systems is negligible.)
        amrex::Real dmin =  std::numeric_limits<amrex::Real>::max();
        amrex::Real dmax = -std::numeric_limits<amrex::Real>::max();
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j <= i; ++j) {
                amrex::Real sum = M[i][j];
                for (int k = 0; k < j; ++k) { sum -= L[i][k]*L[j][k]; }
                if (i == j) {
                    if (!(sum > 0._rt)) { return false; }
                    amrex::Real const d = std::sqrt(sum);
                    L[i][j] = d;
                    dmin = amrex::min(dmin, d);
                    dmax = amrex::max(dmax, d);
                } else {
                    L[i][j] = sum / L[j][j];
                }
            }
        }
        // Phoenix corner-trim: reject ill-conditioned (near-degenerate) systems
        // so collinear-tap cut-cells take the bounded linear fallback instead of
        // producing oscillatory, overshooting quadratic weights.
        if (!(dmin*dmin >= rcond_min * dmax*dmax)) { return false; }

        // Accepted system: refactor with the secondary ridge on the diagonal for
        // a numerically clean solve (no-op when ridge == 0).
        if (ridge != 0._rt) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) { L[i][j] = 0._rt; }
            }
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j <= i; ++j) {
                    amrex::Real sum = M[i][j] + ((i == j) ? ridge : 0._rt);
                    for (int k = 0; k < j; ++k) { sum -= L[i][k]*L[j][k]; }
                    if (i == j) {
                        if (!(sum > 0._rt)) { return false; }
                        L[i][j] = std::sqrt(sum);
                    } else {
                        L[i][j] = sum / L[j][j];
                    }
                }
            }
        }
        for (int i = 0; i < N; ++i) {  // forward: L y = b
            amrex::Real s = b[i];
            for (int k = 0; k < i; ++k) { s -= L[i][k]*b[k]; }
            b[i] = s / L[i][i];
        }
        for (int i = N-1; i >= 0; --i) {  // back: L^T x = y
            amrex::Real s = b[i];
            for (int k = i+1; k < N; ++k) { s -= L[k][i]*b[k]; }
            b[i] = s / L[i][i];
        }
        return true;
    }

    /** 2nd-order moving-least-squares gather of component n at an arbitrary
     *  position, using ONLY stencil points accepted by the \c fluid predicate.
     *  A local quadratic (QUAD_NB terms) is fit to the fluid cells in a +/-2-cell
     *  window by regularized normal equations and evaluated at \c pos. This is
     *  O(h^3) in the value (vs O(h^2) for the (bi/tri)linear gather), so the
     *  curl(B) it feeds is 2nd order at a curved wall. The ridge regularization
     *  lets a degenerate one-sided near-wall stencil degrade gracefully instead
     *  of overshooting. Returns (value, fluid-point count); count < QUAD_NB or a
     *  non-SPD system signals the caller to fall back to the linear gather. */
    template <typename FluidPred>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::GpuTuple<amrex::Real, amrex::Real> gather_quadratic_pred (
        amrex::Array4<amrex::Real const> const& a,
        FluidPred const& fluid,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& pos,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& stag,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
        int n,
        amrex::Real* g_io = nullptr, bool use_cached = false) noexcept
    {
        using namespace amrex::literals;
        constexpr int W = 2;  // +/- window (5 cells per direction)

        amrex::Real lc[AMREX_SPACEDIM];
        int ic[AMREX_SPACEDIM];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            lc[d] = (pos[d] - plo[d])*dxi[d] - stag[d];
            ic[d] = static_cast<int>(std::floor(lc[d] + 0.5_rt));
        }

        // The fit value is g . b with g = M^{-1} e0 (M = sum p p^T over the fluid
        // window, b = sum p*field). g is geometry-only, so when use_cached it is
        // read from g_io and the matrix build + solve are skipped; otherwise it is
        // solved here and (if g_io) written out for the cache. Either way b is
        // accumulated over the window (it carries the field values).
        amrex::Real M[QUAD_NB][QUAD_NB];
        amrex::Real bb[QUAD_NB];
        for (int r = 0; r < QUAD_NB; ++r) {
            bb[r] = 0._rt;
            if (!use_cached) { for (int c = 0; c < QUAD_NB; ++c) { M[r][c] = 0._rt; } }
        }
        int count = 0;

        auto accumulate = [&] (int ii, int jj, int kk) {
            amrex::Real const u = ii - lc[0];
#if defined(WARPX_DIM_3D)
            amrex::Real const v = jj - lc[1];
            amrex::Real const w = kk - lc[2];
            amrex::Real const p[QUAD_NB] =
                {1._rt, u, v, w, u*u, v*v, w*w, u*v, u*w, v*w};
#else
            amrex::Real const v = jj - lc[1];
            amrex::Real const p[QUAD_NB] = {1._rt, u, v, u*u, u*v, v*v};
            amrex::ignore_unused(kk);
#endif
            amrex::Real const f = a(ii, jj, kk, n);
            for (int r = 0; r < QUAD_NB; ++r) {
                bb[r] += p[r]*f;
                if (!use_cached) { for (int c = 0; c <= r; ++c) { M[r][c] += p[r]*p[c]; } }
            }
            ++count;
        };

#if defined(WARPX_DIM_3D)
        for (int dk = -W; dk <= W; ++dk) {
            int const kk = ic[2] + dk;
            if (kk < a.begin[2] || kk >= a.end[2]) { continue; }
            for (int dj = -W; dj <= W; ++dj) {
                int const jj = ic[1] + dj;
                if (jj < a.begin[1] || jj >= a.end[1]) { continue; }
                for (int di = -W; di <= W; ++di) {
                    int const ii = ic[0] + di;
                    if (ii < a.begin[0] || ii >= a.end[0]) { continue; }
                    if (fluid(ii, jj, kk)) { accumulate(ii, jj, kk); }
                }
            }
        }
#else
        for (int dj = -W; dj <= W; ++dj) {
            int const jj = ic[1] + dj;
            if (jj < a.begin[1] || jj >= a.end[1]) { continue; }
            for (int di = -W; di <= W; ++di) {
                int const ii = ic[0] + di;
                if (ii < a.begin[0] || ii >= a.end[0]) { continue; }
                if (fluid(ii, jj, 0)) { accumulate(ii, jj, 0); }
            }
        }
#endif
        if (count < QUAD_NB) { return {0._rt, 0._rt}; }

        amrex::Real g[QUAD_NB];
        if (use_cached) {
            for (int r = 0; r < QUAD_NB; ++r) { g[r] = g_io[r]; }
        } else {
            // symmetrize, then solve M g = e0 (g = M^{-1} e0). solve_spd gates on
            // the conditioning of the RAW matrix (Phoenix corner-trim) and only
            // then applies the small ridge as a secondary regularizer -- passing
            // the ridge in (rather than pre-adding it here) keeps the gate from
            // being fooled into accepting a near-collinear corner stencil.
            amrex::Real const ridge = 1.e-6_rt * M[0][0];
            for (int r = 0; r < QUAD_NB; ++r) {
                for (int c = r+1; c < QUAD_NB; ++c) { M[r][c] = M[c][r]; }
                g[r] = (r == 0) ? 1._rt : 0._rt;
            }
            if (!solve_spd<QUAD_NB>(M, g, ridge)) { return {0._rt, 0._rt}; }
            if (g_io != nullptr) { for (int r = 0; r < QUAD_NB; ++r) { g_io[r] = g[r]; } }
        }
        amrex::Real value = 0._rt;
        for (int r = 0; r < QUAD_NB; ++r) { value += g[r]*bb[r]; }
        return {value, static_cast<amrex::Real>(count)};  // fit value at pos (u=0)
    }

    // --- compact per-tap weight cache for the quadratic gather --------------
    // The fit value g.b is linear in the field: value = sum_tap (g.p(tap))*f_tap.
    // Caching the collapsed per-tap weight w_tap = g.p(tap) and the tap location
    // turns every later apply into a sparse dot product -- no window matrix build,
    // Cholesky, or basis evaluation. Tap locations are stored as the neighbour
    // offset (di,dj,dk) from the fill cell, kept as three contiguous ints (one
    // TAP_NCOMP-wide stride per tap) so the apply forms the gather address with a
    // plain add instead of a per-tap mask/shift/subtract unpack. TAP_HALF bounds
    // the representable offset so the build-time range check is unchanged.
    constexpr int TAP_HALF = 32;
    constexpr int TAP_NCOMP = 3;  // di, dj, dk stored contiguously per tap

    // Count the fluid taps a quadratic gather at `pos` would use (the +/-2 window
    // around the staggered image cell). Used once at build time to size the cache.
    template <typename F>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    int count_fluid_window (
        amrex::Array4<amrex::Real const> const& a, F const& fluid,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& pos,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& stag,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi) noexcept
    {
        using namespace amrex::literals;
        constexpr int W = 2;
        int ic[AMREX_SPACEDIM];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            ic[d] = static_cast<int>(std::floor((pos[d]-plo[d])*dxi[d] - stag[d] + 0.5_rt));
        }
        int count = 0;
#if defined(WARPX_DIM_3D)
        for (int dk = -W; dk <= W; ++dk) { int const kk = ic[2]+dk; if (kk < a.begin[2] || kk >= a.end[2]) { continue; }
        for (int dj = -W; dj <= W; ++dj) { int const jj = ic[1]+dj; if (jj < a.begin[1] || jj >= a.end[1]) { continue; }
        for (int di = -W; di <= W; ++di) { int const ii = ic[0]+di; if (ii < a.begin[0] || ii >= a.end[0]) { continue; }
            if (fluid(ii, jj, kk)) { ++count; } } } }
#else
        for (int dj = -W; dj <= W; ++dj) { int const jj = ic[1]+dj; if (jj < a.begin[1] || jj >= a.end[1]) { continue; }
        for (int di = -W; di <= W; ++di) { int const ii = ic[0]+di; if (ii < a.begin[0] || ii >= a.end[0]) { continue; }
            if (fluid(ii, jj, 0)) { ++count; } } }
#endif
        return count;
    }

    // Build-time: given the solved geometry weights g = M^{-1} e0, re-traverse the
    // window and store, per fluid tap, the packed offset from the fill cell and the
    // collapsed weight w_tap = g.p(tap). Writes up to `maxtap` taps and sets
    // *ntap_out; on overflow (or an out-of-range offset) it sets *ntap_out = 0 so
    // the apply falls back to the live gather for that target.
    template <typename F>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void emit_taps (
        amrex::Real const* g, F const& fluid,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& pos,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& stag,
        int ifill, int jfill, int kfill,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
        amrex::Array4<amrex::Real const> const& a,
        int* tap_off, amrex::Real* tap_w, int* ntap_out, int maxtap) noexcept
    {
        using namespace amrex::literals;
        constexpr int W = 2;
        amrex::Real lc[AMREX_SPACEDIM];
        int ic[AMREX_SPACEDIM];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            lc[d] = (pos[d]-plo[d])*dxi[d] - stag[d];
            ic[d] = static_cast<int>(std::floor(lc[d] + 0.5_rt));
        }
        int t = 0;
        bool ok = true;
        auto emit = [&] (int ii, int jj, int kk) {
            amrex::Real const u = ii - lc[0];
#if defined(WARPX_DIM_3D)
            amrex::Real const v = jj - lc[1];
            amrex::Real const w = kk - lc[2];
            amrex::Real const p[QUAD_NB] =
                {1._rt, u, v, w, u*u, v*v, w*w, u*v, u*w, v*w};
#else
            amrex::Real const v = jj - lc[1];
            amrex::Real const p[QUAD_NB] = {1._rt, u, v, u*u, u*v, v*v};
            amrex::ignore_unused(kk);
#endif
            amrex::Real wt = 0._rt;
            for (int r = 0; r < QUAD_NB; ++r) { wt += g[r]*p[r]; }
            int const di = ii - ifill, dj = jj - jfill, dk = kk - kfill;
            if (t >= maxtap || di <= -TAP_HALF || di >= TAP_HALF
                || dj <= -TAP_HALF || dj >= TAP_HALF
                || dk <= -TAP_HALF || dk >= TAP_HALF) { ok = false; return; }
            tap_off[t*TAP_NCOMP + 0] = di;
            tap_off[t*TAP_NCOMP + 1] = dj;
            tap_off[t*TAP_NCOMP + 2] = dk;
            tap_w[t] = wt;
            ++t;
        };
#if defined(WARPX_DIM_3D)
        for (int dk = -W; dk <= W; ++dk) { int const kk = ic[2]+dk; if (kk < a.begin[2] || kk >= a.end[2]) { continue; }
        for (int dj = -W; dj <= W; ++dj) { int const jj = ic[1]+dj; if (jj < a.begin[1] || jj >= a.end[1]) { continue; }
        for (int di = -W; di <= W; ++di) { int const ii = ic[0]+di; if (ii < a.begin[0] || ii >= a.end[0]) { continue; }
            if (fluid(ii, jj, kk)) { emit(ii, jj, kk); } } } }
#else
        for (int dj = -W; dj <= W; ++dj) { int const jj = ic[1]+dj; if (jj < a.begin[1] || jj >= a.end[1]) { continue; }
        for (int di = -W; di <= W; ++di) { int const ii = ic[0]+di; if (ii < a.begin[0] || ii >= a.end[0]) { continue; }
            if (fluid(ii, jj, 0)) { emit(ii, jj, 0); } } }
#endif
        *ntap_out = ok ? t : 0;
    }

    // Apply: collapsed sparse dot value = sum_tap w_tap * field(fill_cell + offset).
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real gather_tap_apply (
        amrex::Array4<amrex::Real const> const& a, int n,
        int ifill, int jfill, int kfill,
        int const* tap_off, amrex::Real const* tap_w, int ntap) noexcept
    {
        using namespace amrex::literals;
        amrex::Real value = 0._rt;
        for (int t = 0; t < ntap; ++t) {
            int const di = tap_off[t*TAP_NCOMP + 0];
            int const dj = tap_off[t*TAP_NCOMP + 1];
            int const dk = tap_off[t*TAP_NCOMP + 2];
            value += tap_w[t] * a(ifill+di, jfill+dj, kfill+dk, n);
        }
        return value;
    }

    // Classification of every staggered point for the embedded-boundary fill
    constexpr int S_SOLUTION = 0;  //!< solution domain: gather source, never written
    constexpr int S_FILL     = 1;  //!< fill target with a well-posed image stencil
    constexpr int S_PENDING  = 2;  //!< fill target with an ill-posed image stencil
    constexpr int S_DEEP     = 3;  //!< deep inside the conductor (or degenerate normal): zeroed
    constexpr int S_RESOLVED = 4;  //!< well-posed target written and locked during this call
    constexpr int S_JUSTDONE = 5;  //!< pending target written in the current cascade sweep
    constexpr int S_RESOLVED_P = 6; //!< pending target written and locked during this call
    constexpr int S_CORNER   = 7;  //!< re-entrant corner: covered-B replaced by the diagonal-cut chamfer

    // Concave re-entrant-corner detector thresholds (corner CHAMFER, opt-in).
    // Clause (i): the level-set normal must bend by more than this against a
    // +/-1-cell wall neighbour to count as a surface bend (dot below cos 30deg).
    constexpr amrex::Real S_CORNER_COS_BEND = 0.866_rt;  // cos(30 degrees)
    // Clause (ii), LOAD-BEARING: the discrete Laplacian of the signed distance,
    // made dimensionless as |lap(phi)| * h_max, must exceed this. WarpX feeds a
    // UNIT-GRADIENT signed distance (amrex::FillSignedDistance, |grad phi| ~ 1):
    // there lap(phi) = -kappa (minus the surface curvature) away from the medial
    // axis, so a SMOOTH cylinder gives a small bounded |lap*h| ~ 2 h / R (~0.08
    // at the test resolutions), while a re-entrant CORNER puts the medial-axis
    // kink (a delta-like ridge) right in the covered band and |lap*h| spikes to
    // O(1) (measured 0.6 at the 90th pct, ~2 at the 99th on the stepdown).
    // 0.30 separates them with zero smooth-wall false positives across radius /
    // resolution / orientation (prototype_corner_detector.py). This REPLACES the
    // old "phi_nbr - phi_cell > 1.5 h" radius-jump test, which is UNSATISFIABLE
    // on a 1-Lipschitz signed distance (one-cell |dphi| <= h always) and fired
    // n_corner = 0 on the real stepdown.
    constexpr amrex::Real S_CORNER_LAP = 0.30_rt;

    // A fill target is well-posed when at least this fraction of every
    // component's image-interpolation weight lies in the solution domain
    constexpr amrex::Real W_MIN = 0.5;

    /** Mirror-image geometry of one staggered point: signed distance,
     *  image-point position and its distance from the surface. */
    struct MirrorGeom
    {
        bool band;        //!< in the fill band with a usable boundary normal
        amrex::Real s;    //!< signed distance of the point (< 0 in the conductor)
        amrex::Real d_im; //!< distance of the image point from the surface
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xe;  //!< point position
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xim; //!< image position
        amrex::RealVect nv; //!< unit boundary normal (toward the plasma)
    };

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    MirrorGeom mirror_geom (
        int i, int j, int k,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& stag_own,
        amrex::Array4<amrex::Real const> const& phi,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx_arr,
        amrex::Real d_band, amrex::Real d_img_min, amrex::Real h_max) noexcept
    {
        using namespace amrex::literals;

        MirrorGeom g{};

        // point position in physical coordinates
        auto& xe = g.xe;
#if defined(WARPX_DIM_3D)
        xe[0] = plo[0] + (i + stag_own[0])*dx_arr[0];
        xe[1] = plo[1] + (j + stag_own[1])*dx_arr[1];
        xe[2] = plo[2] + (k + stag_own[2])*dx_arr[2];
        amrex::Real const yq = xe[1];
        amrex::Real const zq = xe[2];
#else
        xe[0] = plo[0] + (i + stag_own[0])*dx_arr[0];
        xe[1] = plo[1] + (j + stag_own[1])*dx_arr[1];
        amrex::Real const yq = 0._rt;
        amrex::Real const zq = xe[1];
#endif

        // signed distance at the point (< 0 in the conductor)
        int ii, jj, kk;
        amrex::Real W[AMREX_SPACEDIM][2];
        ablastr::particles::compute_weights<amrex::IndexType::NODE>(
            xe[0], yq, zq, plo, dxi, ii, jj, kk, W);
        g.s = ablastr::particles::interp_field_nodal(ii, jj, kk, W, phi);

        if (g.s < -d_band) {
            g.band = false;
            return g;
        }

        // boundary normal (toward the plasma) from the level set
        int ic, jc, kc;
        amrex::Real Wc[AMREX_SPACEDIM][2];
        ablastr::particles::compute_weights<amrex::IndexType::CELL>(
            xe[0], yq, zq, plo, dxi, ic, jc, kc, Wc);
        g.nv = DistanceToEB::interp_normal(ii, jj, kk, W, ic, jc, kc, Wc, phi, dxi);
        amrex::Real const nv2 = DistanceToEB::dot_product(g.nv, g.nv);
        if (!(nv2 > 0._rt) || !std::isfinite(nv2)) {
            // degenerate level-set gradient: treat as deep interior
            g.band = false;
            return g;
        }
        DistanceToEB::normalize(g.nv);

        // image point in the plasma, at least one cell away from this point
        // so that the interpolation stencil decouples from the fill band;
        // d_im is its distance from the surface (linear tangential profile)
        amrex::Real const offset =
            amrex::max(amrex::max(std::abs(g.s), d_img_min) - g.s, h_max);
        g.d_im = g.s + offset;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            g.xim[d] = xe[d] + offset*g.nv[d];
        }
        g.band = true;
        return g;
    }

    /** Unit boundary normal (toward the plasma) of the level set at an arbitrary
     *  physical position \c xq. Returns the zero vector if the gradient is
     *  degenerate. Used by the sharp-corner detector to compare the wall normal
     *  at the fill cell against the normal in the region its mirror image lands
     *  in: at a smooth wall they agree; at a sharp concave corner the image
     *  crosses to the other facet and the two normals disagree. */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::RealVect normal_at (
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& xq,
        amrex::Array4<amrex::Real const> const& phi,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi) noexcept
    {
        using namespace amrex::literals;
#if defined(WARPX_DIM_3D)
        amrex::Real const yq = xq[1];
        amrex::Real const zq = xq[2];
#else
        amrex::Real const yq = 0._rt;
        amrex::Real const zq = xq[1];
#endif
        int ii, jj, kk;
        amrex::Real W[AMREX_SPACEDIM][2];
        ablastr::particles::compute_weights<amrex::IndexType::NODE>(
            xq[0], yq, zq, plo, dxi, ii, jj, kk, W);
        int ic, jc, kc;
        amrex::Real Wc[AMREX_SPACEDIM][2];
        ablastr::particles::compute_weights<amrex::IndexType::CELL>(
            xq[0], yq, zq, plo, dxi, ic, jc, kc, Wc);
        amrex::RealVect nv = DistanceToEB::interp_normal(ii, jj, kk, W, ic, jc, kc, Wc, phi, dxi);
        amrex::Real const nv2 = DistanceToEB::dot_product(nv, nv);
        if (!(nv2 > 0._rt) || !std::isfinite(nv2)) { return amrex::RealVect{AMREX_D_DECL(0._rt, 0._rt, 0._rt)}; }
        DistanceToEB::normalize(nv);
        return nv;
    }

    /** Radial metric Jacobian lambda = r_image / r_fill for a wall that is a
     *  surface of revolution about axis \c cyl_ax (the cylinder is centered on
     *  the transverse origin). Returns 1 in 2D / for a flat wall. */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real cyl_lambda (
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& xe,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& xim,
        int cyl_ax) noexcept
    {
        using namespace amrex::literals;
#if defined(WARPX_DIM_3D)
        amrex::Real re2 = 0._rt;
        amrex::Real ri2 = 0._rt;
        for (int d = 0; d < 3; ++d) {
            if (d == cyl_ax) { continue; }
            re2 += xe[d]*xe[d];
            ri2 += xim[d]*xim[d];
        }
        amrex::Real const re = std::sqrt(re2);
        return std::sqrt(ri2) / amrex::max(re, std::numeric_limits<amrex::Real>::min());
#else
        amrex::ignore_unused(xe, xim, cyl_ax);
        return 1._rt;
#endif
    }

    /** Parity-correct mirror value of component \c c built from the image-point
     *  field (Jx_im, Jy_im, Jz_im) and the unit boundary normal \c nv, with the
     *  normal weight \c w_n and the tangential weight \c w_t. With \c cyl the
     *  wall is a surface of revolution about axis \c cyl_ax: the radial and
     *  azimuthal parts of the reflected field are scaled by the radial metric
     *  Jacobian \c lambda (the axial part is not) -- the leading O(h/R)
     *  curved-wall correction. With cyl=false (or lambda=1) this reduces exactly
     *  to the planar reflection w_n*(n.J)n_c + w_t*(J_c - (n.J)n_c). */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real mirror_combine (
        int c,
        amrex::Real Jx_im, amrex::Real Jy_im, amrex::Real Jz_im,
        amrex::RealVect const& nv, amrex::Real w_n, amrex::Real w_t,
        bool cyl, int cyl_ax, amrex::Real lambda) noexcept
    {
        using namespace amrex::literals;
#if defined(WARPX_DIM_3D)
        amrex::Real const ndotJ = nv[0]*Jx_im + nv[1]*Jy_im + nv[2]*Jz_im;
        amrex::Real const J_im_e = (c == 0) ? Jx_im : ((c == 1) ? Jy_im : Jz_im);
        amrex::Real const rad_c = ndotJ * nv[c];   // radial (normal) part along c
        if (!cyl) {
            return w_n*rad_c + w_t*(J_im_e - rad_c);
        }
        // surface of revolution about cyl_ax: split the tangential part into a
        // purely axial piece (no radial metric) and the azimuthal remainder
        amrex::Real const J_ax = (cyl_ax == 0) ? Jx_im : ((cyl_ax == 1) ? Jy_im : Jz_im);
        amrex::Real const ax_c = (c == cyl_ax) ? J_ax : 0._rt;   // axial part along c
        amrex::Real const azi_c = J_im_e - rad_c - ax_c;         // azimuthal part along c
        return w_n*lambda*rad_c + w_t*(lambda*azi_c + ax_c);
#else
        amrex::ignore_unused(cyl, cyl_ax, lambda, Jy_im);
        amrex::Real const ndotJ = nv[0]*Jx_im + nv[1]*Jz_im;
        amrex::Real const e_dot_n = (c == 0) ? nv[0] : ((c == 2) ? nv[1] : 0._rt);
        amrex::Real const J_im_e = (c == 0) ? Jx_im : ((c == 1) ? Jy_im : Jz_im);
        amrex::Real const rad_c = ndotJ * e_dot_n;
        return w_n*rad_c + w_t*(J_im_e - rad_c);
#endif
    }

    /** Build the fill classification of every staggered point of the three
     *  field components (see the status codes above). Depends only on the
     *  embedded-boundary geometry and the update masks, so the result is
     *  cacheable until the grids change. */
    void build_fill_status (
        warpx::hybrid::EBFillStatus& st,
        ablastr::fields::VectorField const& field,
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update,
        amrex::MultiFab const& distance_to_eb,
        amrex::Geometry const& geom,
        std::array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, 3> const& stag,
        amrex::Real d_band, amrex::Real d_img_min, amrex::Real h_max,
        bool fill_covered_centers, bool corner_skip)
    {
        using namespace amrex::literals;

        auto const plo = geom.ProbLoArray();
        auto const dxi = geom.InvCellSizeArray();
        auto const dx_arr = geom.CellSizeArray();

        for (int c = 0; c < 3; ++c) {
            st.status[c] = std::make_unique<amrex::iMultiFab>(
                field[c]->boxArray(), field[c]->DistributionMap(), 1,
                field[c]->nGrowVect());
        }

        // First pass: solution domain vs fill target vs deep interior
        for (int c = 0; c < 3; ++c) {
            auto const stag_own = stag[c];
            for (amrex::MFIter mfi(*st.status[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                amrex::Box const tb = mfi.tilebox(field[c]->ixType().toIntVect());
                auto const& stat = st.status[c]->array(mfi);
                auto const& mask = eb_update[c]->const_array(mfi);
                auto const& phi = distance_to_eb.const_array(mfi);

                amrex::ParallelFor(tb,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    auto const g = ::mirror_geom(i, j, k, stag_own, phi,
                        plo, dxi, dx_arr, d_band, d_img_min, h_max);
                    if (mask(i, j, k) != 0) {
                        // Updated by the solver. With fill_covered_centers, cut
                        // points whose centers are on or inside the surface are
                        // fill targets anyway (the solver evaluates them there).
                        stat(i, j, k) = (fill_covered_centers && g.s <= 0._rt)
                            ? (g.band ? S_FILL : S_DEEP) : S_SOLUTION;
                    } else {
                        stat(i, j, k) = g.band ? S_FILL : S_DEEP;
                    }
                });
            }
            st.status[c]->FillBoundary(geom.periodicity());
        }

        // Corner sweep (corner CHAMFER, opt-in): flag the re-entrant-corner fill
        // targets so the direct pass replaces their sharp-facet covered-B mirror
        // with a DIAGONAL-CUT (chamfer) value -- the smooth fluid-side B carried
        // across the corner -- which removes the cross-wall B jump whose Ampere
        // curl (1/(mu0 h) x dB) crashes the substepper. Runs AFTER the
        // S_FILL/S_DEEP split and BEFORE the well-posedness pass, so a flagged
        // cell is no longer S_FILL and is never reclassified S_PENDING (its
        // pointwise-mirror cascade is exactly the spike-injecting path we must
        // avoid). A cell is S_CORNER only if BOTH geometry clauses hold:
        //   (i)   the level-set normal bends > 30 deg vs a +/-1-cell wall
        //         neighbour's normal (corroborates a surface bend);
        //   (ii)  LOAD-BEARING Laplacian spike: |lap(phi)| * h_max > S_CORNER_LAP.
        //         On the unit-gradient signed distance this is the corner kink
        //         (medial-axis ridge); a smooth wall stays well below the gate.
        //         This REPLACES the old radius-jump test that was unsatisfiable on
        //         a 1-Lipschitz signed distance (n_corner = 0 on the real stepdown).
        // Clause (iii) mirror-reach is folded into (ii): the kink ridge only
        // reaches the covered band within ~2 cells of the corner edge.
        if (corner_skip) {
            amrex::ReduceOps<amrex::ReduceOpSum> creduce_op;
            amrex::ReduceData<int> creduce_data(creduce_op);
            using CornerTuple = typename decltype(creduce_data)::Type;

            for (int c = 0; c < 3; ++c) {
                auto const stag_own = stag[c];
                for (amrex::MFIter mfi(*st.status[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    amrex::Box const tb = mfi.tilebox(field[c]->ixType().toIntVect());
                    auto const& stat = st.status[c]->array(mfi);
                    auto const& phi = distance_to_eb.const_array(mfi);

                    creduce_op.eval(tb, creduce_data,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> CornerTuple
                    {
                        if (stat(i, j, k) != S_FILL) { return {0}; }

                        auto const g = ::mirror_geom(i, j, k, stag_own, phi,
                            plo, dxi, dx_arr, d_band, d_img_min, h_max);
                        if (!g.band) { return {0}; }

                        // Clause (i): wall-normal bend > 30 deg vs a +/-1-cell
                        // wall neighbour. nmin is the min dot of nv(xe) against
                        // the wall normal at each valid neighbour (smooth wall
                        // ~1; a sharp facet-change corner << 1).
                        amrex::Real nmin = 2._rt;
                        for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                            for (int sgn = -1; sgn <= 1; sgn += 2) {
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xq = g.xe;
                                xq[dd] += sgn*dx_arr[dd];
                                amrex::RealVect const nn = ::normal_at(xq, phi, plo, dxi);
                                amrex::Real nn2 = 0._rt;
                                amrex::Real dt = 0._rt;
                                for (int e = 0; e < AMREX_SPACEDIM; ++e) {
                                    nn2 += nn[e]*nn[e];
                                    dt  += g.nv[e]*nn[e];
                                }
                                if (nn2 > 0.5_rt) { nmin = amrex::min(nmin, dt); }
                            }
                        }
                        bool const fiducial = (nmin < S_CORNER_COS_BEND);
                        if (!fiducial) { return {0}; }

                        // Clause (ii): LOAD-BEARING Laplacian spike. Build the
                        // discrete Laplacian of the (unit-gradient) signed
                        // distance at the staggered point xe by a centered 2nd
                        // difference of the nodally-interpolated phi at
                        // xe +/- h_max e_d. On a true signed distance lap(phi) =
                        // -kappa away from the medial axis (bounded ~1/R for a
                        // smooth cylinder); at a re-entrant corner the medial-axis
                        // kink makes |lap(phi)| spike to O(1/h). The dimensionless
                        // |lap(phi)| * h_max therefore separates the corner band
                        // (O(1)) from the smooth wall (~2 h / R << 1) with zero
                        // smooth-wall false positives (prototype-validated).
                        amrex::Real lap = 0._rt;
                        for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                            amrex::Real phi_pm[2];
                            for (int sgn = -1, si = 0; sgn <= 1; sgn += 2, ++si) {
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xq = g.xe;
                                xq[dd] += sgn*h_max;
#if defined(WARPX_DIM_3D)
                                amrex::Real const yq = xq[1];
                                amrex::Real const zq = xq[2];
#else
                                amrex::Real const yq = 0._rt;
                                amrex::Real const zq = xq[1];
#endif
                                int iq, jq, kq;
                                amrex::Real Wq[AMREX_SPACEDIM][2];
                                ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                                    xq[0], yq, zq, plo, dxi, iq, jq, kq, Wq);
                                phi_pm[si] =
                                    ablastr::particles::interp_field_nodal(iq, jq, kq, Wq, phi);
                            }
                            lap += (phi_pm[0] + phi_pm[1] - 2._rt*g.s) / (h_max*h_max);
                        }
                        bool const laplacian_spike =
                            (std::abs(lap) * h_max > S_CORNER_LAP);
                        if (!laplacian_spike) { return {0}; }

                        stat(i, j, k) = S_CORNER;
                        return {1};
                    });
                }
                st.status[c]->FillBoundary(geom.periodicity());
            }

            auto const cresult = creduce_data.value(creduce_op);
            int n_corner = amrex::get<0>(cresult);
            amrex::ParallelDescriptor::ReduceIntSum(n_corner);
            st.n_corner = n_corner;
            amrex::Print() << "[EBJBoundary] S_CORNER detector: n_corner = "
                           << n_corner << " cells flagged (diagonal-cut chamfer)\n";
        }

        // Second pass: a fill target is well-posed only if every component
        // of its mirror-image interpolation keeps at least W_MIN of its
        // stencil weight in the solution domain
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<int> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (int c = 0; c < 3; ++c) {
            auto const stag_own = stag[c];
            auto const stag_x = stag[0];
            auto const stag_y = stag[1];
            auto const stag_z = stag[2];
            for (amrex::MFIter mfi(*st.status[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                amrex::Box const tb = mfi.tilebox(field[c]->ixType().toIntVect());
                auto const& stat = st.status[c]->array(mfi);
                auto const& stat_x = st.status[0]->const_array(mfi);
                auto const& stat_y = st.status[1]->const_array(mfi);
                auto const& stat_z = st.status[2]->const_array(mfi);
                auto const& Jx_l = field[0]->const_array(mfi);
                auto const& Jy_l = field[1]->const_array(mfi);
                auto const& Jz_l = field[2]->const_array(mfi);
                auto const& phi = distance_to_eb.const_array(mfi);

                reduce_op.eval(tb, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (stat(i, j, k) != S_FILL) { return {0}; }

                    auto const g = ::mirror_geom(i, j, k, stag_own, phi,
                        plo, dxi, dx_arr, d_band, d_img_min, h_max);
                    auto const in_solution_x =
                        [&] (int ig, int jg, int kg) { return stat_x(ig, jg, kg) == S_SOLUTION; };
                    auto const in_solution_y =
                        [&] (int ig, int jg, int kg) { return stat_y(ig, jg, kg) == S_SOLUTION; };
                    auto const in_solution_z =
                        [&] (int ig, int jg, int kg) { return stat_z(ig, jg, kg) == S_SOLUTION; };
                    auto const [vx, wx] = gather_staggered_pred(Jx_l, in_solution_x, g.xim, stag_x, plo, dxi, 0);
                    auto const [vy, wy] = gather_staggered_pred(Jy_l, in_solution_y, g.xim, stag_y, plo, dxi, 0);
                    auto const [vz, wz] = gather_staggered_pred(Jz_l, in_solution_z, g.xim, stag_z, plo, dxi, 0);
                    amrex::ignore_unused(vx, vy, vz);

                    if (amrex::min(wx, amrex::min(wy, wz)) < W_MIN) {
                        stat(i, j, k) = S_PENDING;
                        return {1};
                    }
                    return {0};
                });
            }
        }

        auto const result = reduce_data.value(reduce_op);
        int n_pending = amrex::get<0>(result);
        amrex::ParallelDescriptor::ReduceIntSum(n_pending);
        st.n_pending = n_pending;

        for (int c = 0; c < 3; ++c) {
            st.status[c]->FillBoundary(geom.periodicity());
        }
    }
#endif

}

void warpx::hybrid::ApplyPECBoundaryToField (
    ablastr::fields::VectorField const& field,
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update,
    amrex::MultiFab const& distance_to_eb,
    amrex::Geometry const& geom,
    amrex::Real rtol,
    int max_iters,
    bool direct_fill,
    bool normal_odd,
    bool fill_covered_centers,
    EBFillStatus* status_cache,
    amrex::Real band_cells,
    bool eb_cyl,
    int eb_cyl_axis,
    bool quadratic_gather,
    amrex::Real normal_weight_floor,
    amrex::Real cut_blend,
    amrex::Real cut_clamp,
    bool corner_skip)
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    using namespace amrex::literals;

    ABLASTR_PROFILE("warpx::hybrid::ApplyPECBoundaryToField()");

    // Near-wall stability blend toward the conformal-ECT cut-face B (only the
    // covered-B quadratic_gather fill consumes it). Clamp to [0,1]; cut_blend=0
    // is the byte-identical full-mirror behavior.
    amrex::Real const cut_blend_w =
        quadratic_gather ? amrex::min(amrex::max(cut_blend, 0._rt), 1._rt) : 0._rt;
    // Cut-face overshoot clamp (relative cap on |B_mirror - B_ECT|); 0 = off.
    amrex::Real const cut_clamp_rel =
        quadratic_gather ? amrex::max(cut_clamp, 0._rt) : 0._rt;
    // Re-entrant-corner CHAMFER: where the two-clause geometric detector (see
    // build_fill_status: a wall-normal bend AND a Laplacian-of-signed-distance
    // spike) flags an S_CORNER cell, the sharp-facet covered-B mirror is replaced
    // by a diagonal-cut value (the smooth fluid-side B carried across the corner)
    // in the post-cascade chamfer pass. This removes the cross-wall B jump whose
    // Ampere curl crashes the substepper. Only the covered-B quadratic_gather fill
    // reaches the covered-center write, so the chamfer is wired through the fill
    // classification and is a no-op otherwise. corner_skip=false is byte-identical
    // (the detector sweep and the chamfer pass are both skipped).
    bool const corner_skip_on = quadratic_gather && corner_skip;

    // DIAGNOSTIC ONLY (Step-1 face-class A/B): WARPX_BCURL_DIAG_SCOPE selects
    // which face class the quadratic_gather mirror writes; the other class keeps
    // its pre-fill value (the conformal-ECT value at cut faces, the previous
    // value at fully-covered faces). 0/unset = both (production). 1 = covered
    // only (cut faces keep ECT). 2 = cut only (covered faces keep prev). Read
    // once per call from the environment so a quick A/B needs no rebuild.
    int diag_scope = 0;
    if (quadratic_gather) {
        if (char const* e = std::getenv("WARPX_BCURL_DIAG_SCOPE")) {
            diag_scope = std::atoi(e);
        }
    }
    // DIAGNOSTIC ONLY (Step-1 corner probe): WARPX_BCURL_DIAG_CORNER=1 dumps, on
    // the quadratic_gather covered-B fill, the geometry (signed distance, normal,
    // image point), the fluid tap count and the mirror value at every S_FILL cell
    // whose image point lands near the concave step-down corner ring (r~R2, z~Z0).
    // Zero cost in production (unset).
    int diag_corner = 0;
    if (quadratic_gather) {
        if (char const* e = std::getenv("WARPX_BCURL_DIAG_CORNER")) {
            diag_corner = std::atoi(e);
        }
    }
    // Full cut-face blend (cut_blend >= 1) is the validated stable configuration:
    // route it through the covered-only path (diag_scope = 1), which SKIPS the
    // cut-face mirror write so each cut face keeps its conformal-ECT value
    // untouched. (Empirically a per-cell self-assign Jc = Jc_prefill at the cut
    // face does NOT reproduce this stability -- the skip is load-bearing -- so the
    // full blend reuses the skip path rather than writing the blended value.)
    if (cut_blend_w >= 1._rt && diag_scope == 0) { diag_scope = 1; }

    if (!eb_update[0]) { return; }

    auto const plo = geom.ProbLoArray();
    auto const dxi = geom.InvCellSizeArray();
    auto const dx_arr = geom.CellSizeArray();

#if defined(WARPX_DIM_3D)
    amrex::Real const h_max = std::max({dx_arr[0], dx_arr[1], dx_arr[2]});
#else
    amrex::Real const h_max = std::max(dx_arr[0], dx_arr[1]);
#endif
    // Mirror-fill masked edges within band_cells of the surface and zero
    // everything deeper. The default one-cell band (band_cells = 1) covers the
    // axis-aligned stencils that straddle the wall; an isotropic stencil also
    // reaches the diagonal neighbors (sqrt(2)*h in plane, sqrt(3)*h at a 3D
    // cube corner), so a consumer of those stencils widens the band to
    // sqrt(AMREX_SPACEDIM) so the corner edges are mirror-filled rather than
    // left in the zeroed deep interior. The minimum image distance keeps the
    // interpolation point in the plasma for edges that sit very close to the
    // surface.
    amrex::Real const d_band = band_cells * h_max;
    amrex::Real const d_img_min = 0.5_rt * h_max;

    // Staggering offsets in grid coordinates for each field component (0.5 in
    // directions where the component is cell-centered)
    std::array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, 3> stag{};
    for (int c = 0; c < 3; ++c) {
        auto const t = field[c]->ixType();
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            stag[c][d] = t.nodeCentered(d) ? 0.0_rt : 0.5_rt;
        }
    }

    if (direct_fill) {
        warpx::hybrid::EBFillStatus local_status;
        warpx::hybrid::EBFillStatus& st = status_cache ? *status_cache : local_status;
        if (st.empty()) {
            ::build_fill_status(st, field, eb_update, distance_to_eb, geom, stag,
                                d_band, d_img_min, h_max, fill_covered_centers,
                                corner_skip_on);
        }

        for (int c = 0; c < 3; ++c) {
            field[c]->FillBoundary(geom.periodicity());
        }

        // 2nd-order gather weight cache: the weights are geometry-only, so on the
        // first call (per cached status) assign every S_FILL target a compact
        // band-sparse slot and size the weight buffer; the direct pass below then
        // computes the weights once (build mode) and reuses them every later call
        // (apply mode). quadratic_gather is used only for the single-component B
        // fill, so the per-component weights do not depend on the field index n.
        bool const cache_weights = quadratic_gather && (status_cache != nullptr);
        if (cache_weights) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(field[0]->nComp() == 1,
                "quadratic_gather weight cache assumes a single-component field");
        }
        if (cache_weights && !st.weights_built) {
            // Pass 1: assign every S_FILL target a compact band-sparse slot.
            for (int c = 0; c < 3; ++c) {
                st.gslot[c] = std::make_unique<amrex::iMultiFab>(
                    st.status[c]->boxArray(), st.status[c]->DistributionMap(),
                    1, st.status[c]->nGrowVect());
                st.gslot[c]->setVal(-1);
                amrex::Gpu::DeviceScalar<int> counter(0);
                int* const cptr = counter.dataPtr();
                for (amrex::MFIter mfi(*st.status[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    amrex::Box const tb = mfi.tilebox(field[c]->ixType().toIntVect());
                    auto const& stat = st.status[c]->const_array(mfi);
                    auto const& slotarr = st.gslot[c]->array(mfi);
                    amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        if (stat(i, j, k) == S_FILL) {
                            slotarr(i, j, k) = amrex::Gpu::Atomic::Add(cptr, 1);
                        }
                    });
                }
                st.n_slots[c] = counter.dataValue();
                st.gtap_n[c].resize(std::size_t(st.n_slots[c]) * 3);
            }
            // Pass 2: count the fluid taps per (slot, gathered component) and find
            // the global max, so the compact (offset, weight) buffers are sized
            // exactly (no per-cell padding to the full window).
            amrex::Gpu::DeviceScalar<int> maxtap_s(0);
            int* const maxtap_ptr = maxtap_s.dataPtr();
            for (int c = 0; c < 3; ++c) {
                auto const stag_own = stag[c];
                auto const stag_x = stag[0];
                auto const stag_y = stag[1];
                auto const stag_z = stag[2];
                int* const gtn = st.gtap_n[c].dataPtr();
                for (amrex::MFIter mfi(*field[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    amrex::Box const tb = mfi.tilebox(field[c]->ixType().toIntVect());
                    auto const& stat = st.status[c]->const_array(mfi);
                    auto const& stat_x = st.status[0]->const_array(mfi);
                    auto const& stat_y = st.status[1]->const_array(mfi);
                    auto const& stat_z = st.status[2]->const_array(mfi);
                    auto const& Jx_l = field[0]->const_array(mfi);
                    auto const& Jy_l = field[1]->const_array(mfi);
                    auto const& Jz_l = field[2]->const_array(mfi);
                    auto const& phi = distance_to_eb.const_array(mfi);
                    auto const& slotarr = st.gslot[c]->const_array(mfi);
                    amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        if (stat(i, j, k) != S_FILL) { return; }
                        auto const gm = ::mirror_geom(i, j, k, stag_own, phi,
                            plo, dxi, dx_arr, d_band, d_img_min, h_max);
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xim;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            xim[d] = gm.xe[d] - 2._rt*gm.s*gm.nv[d];
                        }
                        auto const in_sol_x = [&] (int ig, int jg, int kg) { return stat_x(ig, jg, kg) == S_SOLUTION; };
                        auto const in_sol_y = [&] (int ig, int jg, int kg) { return stat_y(ig, jg, kg) == S_SOLUTION; };
                        auto const in_sol_z = [&] (int ig, int jg, int kg) { return stat_z(ig, jg, kg) == S_SOLUTION; };
                        int const slot = slotarr(i, j, k);
                        int const nx = count_fluid_window(Jx_l, in_sol_x, xim, stag_x, plo, dxi);
                        int const ny = count_fluid_window(Jy_l, in_sol_y, xim, stag_y, plo, dxi);
                        int const nz = count_fluid_window(Jz_l, in_sol_z, xim, stag_z, plo, dxi);
                        gtn[slot*3 + 0] = nx;
                        gtn[slot*3 + 1] = ny;
                        gtn[slot*3 + 2] = nz;
                        amrex::Gpu::Atomic::Max(maxtap_ptr, amrex::max(nx, amrex::max(ny, nz)));
                    });
                }
            }
            st.maxtap = maxtap_s.dataValue();
            for (int c = 0; c < 3; ++c) {
                std::size_t const sz = std::size_t(st.n_slots[c]) * 3 * std::size_t(st.maxtap);
                st.gtap_off[c].resize(sz * TAP_NCOMP);  // di,dj,dk per tap
                st.gtap_w[c].resize(sz);
                st.ggeom[c].resize(std::size_t(st.n_slots[c])
                                   * warpx::hybrid::EBFillStatus::geom_nr);
            }
        }
        bool const use_cached_weights = cache_weights && st.weights_built;

        // The cascade runs only when ill-posed targets exist; it is the only
        // path that mutates (and afterwards restores) the cached status arrays.
        bool const cascade = (st.n_pending > 0);

        // Direct pass: deterministic mirror fill of the well-posed targets,
        // gathering only from solution-domain values so that no stale fill
        // or covered point contaminates the image; deep points are zeroed.
        for (int c = 0; c < 3; ++c) {
            auto const stag_own = stag[c];
            auto const stag_x = stag[0];
            auto const stag_y = stag[1];
            auto const stag_z = stag[2];
            int* const gtn_ptr = cache_weights ? st.gtap_n[c].dataPtr() : nullptr;
            int* const goff_ptr = cache_weights ? st.gtap_off[c].dataPtr() : nullptr;
            amrex::Real* const gtw_ptr = cache_weights ? st.gtap_w[c].dataPtr() : nullptr;
            amrex::Real* const ggeom_ptr = cache_weights ? st.ggeom[c].dataPtr() : nullptr;
            int const maxtap = st.maxtap;
            constexpr int geom_nr = warpx::hybrid::EBFillStatus::geom_nr;

            for (amrex::MFIter mfi(*field[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                amrex::Box const tb = mfi.tilebox(field[c]->ixType().toIntVect());
                int const ncomp = field[c]->nComp();

                auto const& Jc = field[c]->array(mfi);
                auto const& Jx_l = field[0]->const_array(mfi);
                auto const& Jy_l = field[1]->const_array(mfi);
                auto const& Jz_l = field[2]->const_array(mfi);
                auto const& stat = st.status[c]->const_array(mfi);
                auto const& stat_x = st.status[0]->const_array(mfi);
                auto const& stat_y = st.status[1]->const_array(mfi);
                auto const& stat_z = st.status[2]->const_array(mfi);
                auto const& phi = distance_to_eb.const_array(mfi);
                amrex::Array4<int const> const slotarr =
                    cache_weights ? st.gslot[c]->const_array(mfi) : amrex::Array4<int const>{};
                // Update mask: nonzero on a cut/fluid face (the conformal-ECT B
                // push computed a value there). Used by the near-wall cut_blend
                // to keep part of that stabler ECT B instead of the full mirror,
                // and by the Step-1 A/B diagnostic scope.
                auto const& mask = eb_update[c]->const_array(mfi);
                amrex::Real const blend_w = cut_blend_w;
                amrex::Real const clamp_rel = cut_clamp_rel;
                int const dscope = diag_scope;
                int const dcorner = diag_corner;
                int const c_diag = c;

                amrex::ParallelFor(tb, ncomp,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                    int const s0 = stat(i, j, k);
                    if (s0 == S_DEEP) {
                        Jc(i, j, k, n) = 0._rt;
                        return;
                    }
                    if (s0 != S_FILL) { return; }

                    // mirror_geom is geometry-only (it reads the static level-set
                    // phi -- a second scattered field, ~32 reads/cell, plus a sqrt
                    // and the normal-gradient stencil -- and returns the signed
                    // distance, image position and boundary normal, none of which
                    // depend on the field values). On the cached apply path it is
                    // therefore read back from the per-slot ggeom cache built once
                    // alongside the tap weights, so the apply never touches phi.
                    MirrorGeom g{};
                    if (use_cached_weights) {
                        int const slot = slotarr(i, j, k);
                        amrex::Real const* const gp = ggeom_ptr + std::size_t(slot)*geom_nr;
                        g.band = true;
                        g.s = gp[0];
                        g.d_im = gp[1];
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            g.nv[d]  = gp[2 + d];
                            g.xe[d]  = gp[2 + AMREX_SPACEDIM + d];
                            g.xim[d] = gp[2 + 2*AMREX_SPACEDIM + d];
                        }
                    } else {
                        g = ::mirror_geom(i, j, k, stag_own, phi,
                            plo, dxi, dx_arr, d_band, d_img_min, h_max);
                    }

                    auto const in_sol_x =
                        [&] (int ig, int jg, int kg) { return stat_x(ig, jg, kg) == S_SOLUTION; };
                    auto const in_sol_y =
                        [&] (int ig, int jg, int kg) { return stat_y(ig, jg, kg) == S_SOLUTION; };
                    auto const in_sol_z =
                        [&] (int ig, int jg, int kg) { return stat_z(ig, jg, kg) == S_SOLUTION; };
                    // Image point and its surface distance. The default linear
                    // gather uses the band-decoupled image (>= h_max into the
                    // plasma). The 2nd-order gather instead uses the TRUE even
                    // reflection (image at the reflected distance |s|, d_im=|s|)
                    // and a quadratic fluid-only fit, so the curl(B) it feeds is
                    // 2nd order at a curved wall; the linear image's >= h_max
                    // offset is itself an O(h) value error there.
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xim = g.xim;
                    amrex::Real d_im = g.d_im;
                    if (quadratic_gather) {
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            xim[d] = g.xe[d] - 2._rt*g.s*g.nv[d];
                        }
                        d_im = -g.s;
                    }

                    amrex::Real Jx_im, Jy_im, Jz_im;
                    // Step-1 corner diagnostic: per-component fluid tap counts of
                    // the quadratic gather (-1 = not the quadratic build path).
                    amrex::Real cnt_x = -1._rt, cnt_y = -1._rt, cnt_z = -1._rt;
                    if (quadratic_gather && use_cached_weights) {
                        // Apply: collapsed sparse dot per gathered component from
                        // the prebuilt per-tap (offset, weight) cache; no matrix
                        // build, Cholesky, or basis evaluation. Fall back to the
                        // linear gather where the cache is empty (degenerate fit).
                        int const slot = slotarr(i, j, k);
                        int const* const off = goff_ptr;
                        amrex::Real const* const wt = gtw_ptr;
                        int const nx = gtn_ptr[slot*3 + 0];
                        int const ny = gtn_ptr[slot*3 + 1];
                        int const nz = gtn_ptr[slot*3 + 2];
                        std::size_t const bx = (std::size_t(slot)*3 + 0)*maxtap;
                        std::size_t const by = (std::size_t(slot)*3 + 1)*maxtap;
                        std::size_t const bz = (std::size_t(slot)*3 + 2)*maxtap;
                        // the offset stream stores TAP_NCOMP ints per tap
                        Jx_im = (nx > 0) ? gather_tap_apply(Jx_l, n, i, j, k, off+bx*TAP_NCOMP, wt+bx, nx)
                                         : amrex::get<0>(gather_staggered_pred(Jx_l, in_sol_x, xim, stag_x, plo, dxi, n));
                        Jy_im = (ny > 0) ? gather_tap_apply(Jy_l, n, i, j, k, off+by*TAP_NCOMP, wt+by, ny)
                                         : amrex::get<0>(gather_staggered_pred(Jy_l, in_sol_y, xim, stag_y, plo, dxi, n));
                        Jz_im = (nz > 0) ? gather_tap_apply(Jz_l, n, i, j, k, off+bz*TAP_NCOMP, wt+bz, nz)
                                         : amrex::get<0>(gather_staggered_pred(Jz_l, in_sol_z, xim, stag_z, plo, dxi, n));
                    } else if (quadratic_gather) {
                        // Build (or non-cached live): solve the geometry weights g,
                        // and on the cache build also emit the collapsed per-tap
                        // (offset, weight) so later substages take the apply path.
                        amrex::Real gx[QUAD_NB], gy[QUAD_NB], gz[QUAD_NB];
                        auto const [vx, cx] = gather_quadratic_pred(Jx_l, in_sol_x, xim, stag_x, plo, dxi, n, gx, false);
                        auto const [vy, cy] = gather_quadratic_pred(Jy_l, in_sol_y, xim, stag_y, plo, dxi, n, gy, false);
                        auto const [vz, cz] = gather_quadratic_pred(Jz_l, in_sol_z, xim, stag_z, plo, dxi, n, gz, false);
                        if (cache_weights) {
                            int const slot = slotarr(i, j, k);
                            std::size_t const bx = (std::size_t(slot)*3 + 0)*maxtap;
                            std::size_t const by = (std::size_t(slot)*3 + 1)*maxtap;
                            std::size_t const bz = (std::size_t(slot)*3 + 2)*maxtap;
                            // the offset stream stores TAP_NCOMP ints per tap
                            if (cx > 0._rt) { emit_taps(gx, in_sol_x, xim, stag_x, i, j, k, plo, dxi, Jx_l, goff_ptr+bx*TAP_NCOMP, gtw_ptr+bx, gtn_ptr+slot*3+0, maxtap); }
                            else { gtn_ptr[slot*3 + 0] = 0; }
                            if (cy > 0._rt) { emit_taps(gy, in_sol_y, xim, stag_y, i, j, k, plo, dxi, Jy_l, goff_ptr+by*TAP_NCOMP, gtw_ptr+by, gtn_ptr+slot*3+1, maxtap); }
                            else { gtn_ptr[slot*3 + 1] = 0; }
                            if (cz > 0._rt) { emit_taps(gz, in_sol_z, xim, stag_z, i, j, k, plo, dxi, Jz_l, goff_ptr+bz*TAP_NCOMP, gtw_ptr+bz, gtn_ptr+slot*3+2, maxtap); }
                            else { gtn_ptr[slot*3 + 2] = 0; }
                            // cache the geometry-only mirror outputs for this slot
                            // so later (apply) calls skip the mirror_geom recompute
                            amrex::Real* const gp = ggeom_ptr + std::size_t(slot)*geom_nr;
                            gp[0] = g.s;
                            gp[1] = g.d_im;
                            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                                gp[2 + d]                  = g.nv[d];
                                gp[2 + AMREX_SPACEDIM + d] = g.xe[d];
                                gp[2 + 2*AMREX_SPACEDIM + d] = g.xim[d];
                            }
                        }
                        Jx_im = (cx > 0._rt) ? vx : amrex::get<0>(gather_staggered_pred(Jx_l, in_sol_x, xim, stag_x, plo, dxi, n));
                        Jy_im = (cy > 0._rt) ? vy : amrex::get<0>(gather_staggered_pred(Jy_l, in_sol_y, xim, stag_y, plo, dxi, n));
                        Jz_im = (cz > 0._rt) ? vz : amrex::get<0>(gather_staggered_pred(Jz_l, in_sol_z, xim, stag_z, plo, dxi, n));
                        cnt_x = cx; cnt_y = cy; cnt_z = cz;
                    } else {
                        Jx_im = amrex::get<0>(gather_staggered_pred(Jx_l, in_sol_x, xim, stag_x, plo, dxi, n));
                        Jy_im = amrex::get<0>(gather_staggered_pred(Jy_l, in_sol_y, xim, stag_y, plo, dxi, n));
                        Jz_im = amrex::get<0>(gather_staggered_pred(Jz_l, in_sol_z, xim, stag_z, plo, dxi, n));
                    }

                    // edge fields (E, J): normal even / tangential odd;
                    // magnetic field: normal odd / tangential even. The
                    // magnetic normal weight is optionally clamped from below
                    // (default -1e30 = no clamp): a floor of 0 drives the
                    // covered B_normal toward 0 (PEC B_normal -> 0) instead of
                    // an odd sign reversal; the tangential weight is untouched.
                    amrex::Real const w_n = normal_odd
                        ? amrex::max(g.s/d_im, normal_weight_floor) : 1._rt;
                    amrex::Real const w_t = normal_odd ? 1._rt : g.s/d_im;
                    // optional surface-of-revolution radial metric Jacobian
                    amrex::Real const lambda =
                        eb_cyl ? ::cyl_lambda(g.xe, xim, eb_cyl_axis) : 1._rt;
                    amrex::Real const b_mirror = ::mirror_combine(
                        c, Jx_im, Jy_im, Jz_im, g.nv,
                        w_n, w_t, eb_cyl, eb_cyl_axis, lambda);

                    // Step-1 corner diagnostic (WARPX_BCURL_DIAG_CORNER=1): dump
                    // the geometry, the fluid tap count and the mirror value at
                    // every covered-B fill cell near the concave step-down corner
                    // ring (test geometry: r ~ R2 = 0.5, z ~ Z0 = 1.0). Reads the
                    // fill cell position g.xe; gated so it is zero-cost otherwise.
                    if (dcorner != 0 && n == 0) {
#if defined(WARPX_DIM_3D)
                        amrex::Real const r_xe = std::sqrt(g.xe[0]*g.xe[0] + g.xe[1]*g.xe[1]);
                        amrex::Real const z_xe = g.xe[2];
                        amrex::Real const r_im = std::sqrt(xim[0]*xim[0] + xim[1]*xim[1]);
                        amrex::Real const z_im = xim[2];
                        if (std::abs(r_xe - 0.5_rt) < 0.12_rt && std::abs(z_xe - 1.0_rt) < 0.12_rt) {
                            amrex::Real const b_im_mag = std::sqrt(
                                Jx_im*Jx_im + Jy_im*Jy_im + Jz_im*Jz_im);
                            // sharp-corner detector signal: normal at the image
                            // region vs at the fill cell (1 = smooth, <~0.5 = the
                            // image crossed to the other facet of a sharp corner)
                            amrex::RealVect const nv_im = ::normal_at(xim, phi, plo, dxi);
                            amrex::Real const ndot =
                                g.nv[0]*nv_im[0] + g.nv[1]*nv_im[1] + g.nv[2]*nv_im[2];
                            // neighbor-normal disagreement: min dot of nv(xe) against
                            // the wall normal at the +/-1-cell face neighbors (smooth
                            // wall ~1, sharp facet-change corner << 1)
                            amrex::Real nmin = 2._rt;
                            for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                                for (int sgn = -1; sgn <= 1; sgn += 2) {
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xq = g.xe;
                                    xq[dd] += sgn*dx_arr[dd];
                                    amrex::RealVect const nn = ::normal_at(xq, phi, plo, dxi);
                                    amrex::Real const nn2 = nn[0]*nn[0] + nn[1]*nn[1] + nn[2]*nn[2];
                                    if (nn2 > 0.5_rt) {
                                        amrex::Real const dt =
                                            g.nv[0]*nn[0] + g.nv[1]*nn[1] + g.nv[2]*nn[2];
                                        nmin = amrex::min(nmin, dt);
                                    }
                                }
                            }
                            std::printf(
                                "[BCURL c=%d i=%d j=%d k=%d] r_xe=%.4f z_xe=%.4f s=%.5f "
                                "nv=(%.3f,%.3f,%.3f) r_im=%.4f z_im=%.4f ndot=%.3f nmin=%.3f taps=(%.0f,%.0f,%.0f) "
                                "Jim=(%.3e,%.3e,%.3e) |Jim|=%.3e w_n=%.3f w_t=%.3f "
                                "b_mirror=%.4e cut=%d\n",
                                c_diag, i, j, k, r_xe, z_xe, g.s,
                                g.nv[0], g.nv[1], g.nv[2], r_im, z_im, ndot, nmin,
                                cnt_x, cnt_y, cnt_z, Jx_im, Jy_im, Jz_im, b_im_mag,
                                w_n, w_t, b_mirror, int(mask(i, j, k) != 0));
                        }
#endif
                    }

                    // Step-1 A/B face-class diagnostic: skip the mirror write at
                    // the deselected class so it keeps its pre-fill value (cut
                    // faces -> the conformal-ECT B; fully-covered faces -> their
                    // previous value). dscope 1 = covered only, 2 = cut only.
                    bool const is_cut = (mask(i, j, k) != 0);
                    if ((dscope == 1 && is_cut) || (dscope == 2 && !is_cut)) {
                        return;  // leave Jc at its pre-fill value
                    }

                    // Near-wall stability blend toward the conformal-ECT cut-face
                    // B. The face-class A/B diagnostic (WARPX_BCURL_DIAG_SCOPE)
                    // shows the stiffness driver is the CUT faces (mask != 0): the
                    // mirror discards the stabler ECT B the masked Faraday push
                    // computed there and writes a steeper near-wall extrapolation
                    // (which can overshoot wildly -- the quadratic fluid gather has
                    // oscillatory tap weights) that collapses the RKF45 substep on
                    // the dense-shell liftoff. The fully-covered faces' mirror is
                    // harmless, so the blend is applied ONLY at cut faces, where the
                    // pre-fill Jc is the true ECT value. The blend keeps the ECT
                    // value and adds back only the (1-blend) fraction of the mirror
                    // CORRECTION (b_mirror - b_ect), so at blend = 1 the cut face is
                    // exactly the ECT value (the mirror -- and any overshoot in it --
                    // is fully discarded, identical to the stable diagnostic scope),
                    // and at blend = 0 it is exactly the full mirror (byte-identical).
                    // Writing it as b_ect + (1-blend)*(b_mirror - b_ect) rather than
                    // (1-blend)*b_mirror + blend*b_ect avoids a 0*overshoot residual
                    // at blend = 1. The gentle regime (eb_diffusion) runs at blend =
                    // 0, so its full 2nd-order mirror is preserved.
                    if (blend_w >= 1._rt && is_cut) {
                        // Full blend: keep the ECT cut-face value untouched (skip
                        // the mirror write entirely), bit-identical to leaving Jc
                        // at its pre-fill value -- the stable configuration.
                        return;
                    }
                    if (is_cut && (blend_w > 0._rt || clamp_rel > 0._rt)) {
                        amrex::Real const b_ect = Jc(i, j, k, n);
                        // Optional cut-face overshoot CLAMP: the quadratic mirror
                        // can overshoot the ECT value wildly on the stiff dense
                        // shell (oscillatory gather weights) while staying close to
                        // it in the gentle regime. clamp_rel > 0 caps the mirror's
                        // deviation from the ECT value to clamp_rel times the local
                        // reference magnitude, so the gentle 2nd-order correction
                        // passes unclamped while the stiff overshoot is bounded.
                        amrex::Real b_use = b_mirror;
                        if (clamp_rel > 0._rt) {
                            amrex::Real const b_im = std::sqrt(
                                Jx_im*Jx_im + Jy_im*Jy_im + Jz_im*Jz_im);
                            amrex::Real const cap =
                                clamp_rel * amrex::max(std::abs(b_ect), b_im);
                            amrex::Real const dev = b_mirror - b_ect;
                            b_use = b_ect + amrex::min(amrex::max(dev, -cap), cap);
                        }
                        Jc(i, j, k, n) = b_ect + (1._rt - blend_w)*(b_use - b_ect);
                    } else {
                        // Fully-covered center (and any cut face whose blend/clamp
                        // are off): the full mirror is the only write this branch
                        // reaches. The re-entrant-corner CHAMFER is handled
                        // separately: conformal_b_curl_fill_corner_skip flags those
                        // cells S_CORNER in the classification, so they never reach
                        // this kernel as S_FILL; the post-cascade chamfer pass then
                        // overwrites them with the diagonal-cut fluid-side B (see
                        // build_fill_status and the S_CORNER detector / chamfer pass).
                        Jc(i, j, k, n) = b_mirror;
                    }
                });
            }
        }

        // The direct pass has now computed every well-posed target's gather
        // weights (build mode); flag the cache so later calls skip the matrix
        // build + Cholesky and just do the cached mat-vec (apply mode).
        if (cache_weights) { st.weights_built = true; }

        // Resolve the ill-posed targets by a deterministic cascade: lock the
        // direct-pass values and, sweep by sweep, fill the pending points
        // whose image stencils reach already locked values
        if (cascade) {
            for (int c = 0; c < 3; ++c) {
                for (amrex::MFIter mfi(*st.status[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    amrex::Box const tb = mfi.tilebox(field[c]->ixType().toIntVect());
                    auto const& stat = st.status[c]->array(mfi);
                    amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        if (stat(i, j, k) == S_FILL) { stat(i, j, k) = S_RESOLVED; }
                    });
                }
            }

            int n_left = st.n_pending;
            for (int sweep = 0; sweep < max_iters && n_left > 0; ++sweep) {
                for (int c = 0; c < 3; ++c) {
                    field[c]->FillBoundary(geom.periodicity());
                    st.status[c]->FillBoundary(geom.periodicity());
                }

                amrex::ReduceOps<amrex::ReduceOpSum> rop;
                amrex::ReduceData<int> rdata(rop);
                using SweepTuple = typename decltype(rdata)::Type;

                for (int c = 0; c < 3; ++c) {
                    auto const stag_own = stag[c];
                    auto const stag_x = stag[0];
                    auto const stag_y = stag[1];
                    auto const stag_z = stag[2];
                    for (amrex::MFIter mfi(*st.status[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                        amrex::Box const tb = mfi.tilebox(field[c]->ixType().toIntVect());
                        int const ncomp = field[c]->nComp();
                        auto const& Jc = field[c]->array(mfi);
                        auto const& Jx_l = field[0]->const_array(mfi);
                        auto const& Jy_l = field[1]->const_array(mfi);
                        auto const& Jz_l = field[2]->const_array(mfi);
                        auto const& stat = st.status[c]->array(mfi);
                        auto const& stat_x = st.status[0]->const_array(mfi);
                        auto const& stat_y = st.status[1]->const_array(mfi);
                        auto const& stat_z = st.status[2]->const_array(mfi);
                        auto const& phi = distance_to_eb.const_array(mfi);

                        rop.eval(tb, rdata,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> SweepTuple
                        {
                            if (stat(i, j, k) != S_PENDING) { return {0}; }

                            auto const g = ::mirror_geom(i, j, k, stag_own, phi,
                                plo, dxi, dx_arr, d_band, d_img_min, h_max);

                            auto const locked_x = [&] (int ig, int jg, int kg) {
                                int const sl = stat_x(ig, jg, kg);
                                return sl == S_SOLUTION || sl == S_RESOLVED || sl == S_RESOLVED_P;
                            };
                            auto const locked_y = [&] (int ig, int jg, int kg) {
                                int const sl = stat_y(ig, jg, kg);
                                return sl == S_SOLUTION || sl == S_RESOLVED || sl == S_RESOLVED_P;
                            };
                            auto const locked_z = [&] (int ig, int jg, int kg) {
                                int const sl = stat_z(ig, jg, kg);
                                return sl == S_SOLUTION || sl == S_RESOLVED || sl == S_RESOLVED_P;
                            };

                            int const ncomp_l = ncomp;
                            // Fill only once every component stencil reaches a
                            // locked value; the weights do not depend on n, so
                            // probing n = 0 suffices.
                            bool reached = true;
                            {
                                auto const [v0x, w0x] = gather_staggered_pred(Jx_l, locked_x, g.xim, stag_x, plo, dxi, 0);
                                auto const [v0y, w0y] = gather_staggered_pred(Jy_l, locked_y, g.xim, stag_y, plo, dxi, 0);
                                auto const [v0z, w0z] = gather_staggered_pred(Jz_l, locked_z, g.xim, stag_z, plo, dxi, 0);
                                amrex::ignore_unused(v0x, v0y, v0z);
                                reached = (w0x > 0._rt) && (w0y > 0._rt) && (w0z > 0._rt);
                            }
                            if (!reached) { return {0}; }

                            for (int n = 0; n < ncomp_l; ++n) {
                                auto const [Jx_im, wx_im] = gather_staggered_pred(Jx_l, locked_x, g.xim, stag_x, plo, dxi, n);
                                auto const [Jy_im, wy_im] = gather_staggered_pred(Jy_l, locked_y, g.xim, stag_y, plo, dxi, n);
                                auto const [Jz_im, wz_im] = gather_staggered_pred(Jz_l, locked_z, g.xim, stag_z, plo, dxi, n);
                                amrex::ignore_unused(wx_im, wy_im, wz_im);
                                amrex::Real const w_n = normal_odd
                                    ? amrex::max(g.s/g.d_im, normal_weight_floor) : 1._rt;
                                amrex::Real const w_t = normal_odd ? 1._rt : g.s/g.d_im;
                                amrex::Real const lambda =
                                    eb_cyl ? ::cyl_lambda(g.xe, g.xim, eb_cyl_axis) : 1._rt;
                                Jc(i, j, k, n) = ::mirror_combine(c, Jx_im, Jy_im, Jz_im,
                                    g.nv, w_n, w_t, eb_cyl, eb_cyl_axis, lambda);
                            }
                            stat(i, j, k) = S_JUSTDONE;
                            return {1};
                        });
                    }
                }

                auto const sweep_result = rdata.value(rop);
                int n_done = amrex::get<0>(sweep_result);
                amrex::ParallelDescriptor::ReduceIntSum(n_done);

                // Promote this sweep's results to locked values only now, so
                // pending points never gather from values of the same sweep.
                for (int c = 0; c < 3; ++c) {
                    for (amrex::MFIter mfi(*st.status[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                        amrex::Box const tb = mfi.tilebox(field[c]->ixType().toIntVect());
                        auto const& stat = st.status[c]->array(mfi);
                        amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        {
                            if (stat(i, j, k) == S_JUSTDONE) { stat(i, j, k) = S_RESOLVED_P; }
                        });
                    }
                }

                if (n_done == 0) { break; }
                n_left -= n_done;
            }

            if (n_left > 0) {
                // Targets still pending are fully enclosed by other pending
                // points; no meaningful mirror value exists, so zero them.
                for (int c = 0; c < 3; ++c) {
                    for (amrex::MFIter mfi(*st.status[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                        amrex::Box const tb = mfi.tilebox(field[c]->ixType().toIntVect());
                        int const ncomp = field[c]->nComp();
                        auto const& Jc = field[c]->array(mfi);
                        auto const& stat = st.status[c]->const_array(mfi);
                        amrex::ParallelFor(tb, ncomp,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                        {
                            if (stat(i, j, k) == S_PENDING) { Jc(i, j, k, n) = 0._rt; }
                        });
                    }
                }
            }

            // restore the cached classification for the next call
            for (int c = 0; c < 3; ++c) {
                for (amrex::MFIter mfi(*st.status[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    amrex::Box const tb = mfi.tilebox(field[c]->ixType().toIntVect());
                    auto const& stat = st.status[c]->array(mfi);
                    amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        if (stat(i, j, k) == S_RESOLVED) { stat(i, j, k) = S_FILL; }
                        else if (stat(i, j, k) == S_RESOLVED_P) { stat(i, j, k) = S_PENDING; }
                    });
                }
            }
        }

        // CHAMFER pass (opt-in): at each S_CORNER cell, replace the (skipped)
        // sharp-facet mirror with a DIAGONAL-CUT value -- the smooth fluid-side B
        // carried across the corner. The verified abort is a one-cell cross-wall
        // B jump (covered tangential B on one side, fluid B = 0 across the step)
        // that the Ampere curl amplifies by 1/(mu0 h). Setting the covered corner
        // B to the average of its in-component FLUID (S_SOLUTION) +/-1
        // curl-neighbours makes dB ~ 0 across the step (a 45-degree chamfer of the
        // 90-degree facet) so curl(B) stays bounded. The field has been
        // FillBoundary'd by the cascade (or by the FillBoundary above), so the
        // fluid neighbour values are current. Runs only where corner_skip flagged
        // S_CORNER cells (none when the flag is off => byte-identical).
        if (corner_skip_on) {
            for (int c = 0; c < 3; ++c) {
                field[c]->FillBoundary(geom.periodicity());
            }
            for (int c = 0; c < 3; ++c) {
                for (amrex::MFIter mfi(*st.status[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    amrex::Box const tb = mfi.tilebox(field[c]->ixType().toIntVect());
                    int const ncomp = field[c]->nComp();
                    auto const& Jc = field[c]->array(mfi);
                    auto const& stat = st.status[c]->const_array(mfi);
                    amrex::ParallelFor(tb, ncomp,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                    {
                        if (stat(i, j, k) != S_CORNER) { return; }
                        // Diagonal-cut value: average the in-component B over the
                        // fluid (S_SOLUTION) +/-1 curl-difference neighbours, so
                        // the covered corner B equals the smooth fluid B carried
                        // across the step (no cross-wall jump). Falls back to the
                        // stable pre-fill / OFF value if no fluid neighbour exists.
                        amrex::Real acc = 0._rt;
                        int cnt = 0;
                        for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                            int di = (dd == 0) ? 1 : 0;
                            int dj = (dd == 1) ? 1 : 0;
                            int dk = (dd == 2) ? 1 : 0;
                            if (stat(i+di, j+dj, k+dk) == S_SOLUTION) {
                                acc += Jc(i+di, j+dj, k+dk, n); ++cnt;
                            }
                            if (stat(i-di, j-dj, k-dk) == S_SOLUTION) {
                                acc += Jc(i-di, j-dj, k-dk, n); ++cnt;
                            }
                        }
                        if (cnt > 0) { Jc(i, j, k, n) = acc / amrex::Real(cnt); }
                    });
                }
            }
        }

        // Leave ghost edges consistent for the stencils that consume the field
        for (int c = 0; c < 3; ++c) {
            field[c]->FillBoundary(geom.periodicity());
        }
        return;
    }

    // Scratch copies holding the previous Jacobi iterate
    std::array<amrex::MultiFab, 3> Jold;
    for (int c = 0; c < 3; ++c) {
        Jold[c].define(field[c]->boxArray(), field[c]->DistributionMap(),
                       field[c]->nComp(), field[c]->nGrowVect());
    }

    int iter = 0;
    amrex::Real resid = std::numeric_limits<amrex::Real>::max();
    while (iter < max_iters && resid > rtol) {

        // Refresh ghost edges (image stencils can reach into neighbor boxes)
        // and snapshot the previous iterate for a deterministic Jacobi sweep
        for (int c = 0; c < 3; ++c) {
            field[c]->FillBoundary(geom.periodicity());
            amrex::MultiFab::Copy(Jold[c], *field[c], 0, 0,
                                  field[c]->nComp(), field[c]->nGrowVect());
        }

        // Track the largest change and the largest band value so convergence
        // is measured relative to the boundary-band field magnitude; a per-edge
        // relative criterion can never be met by edges holding near-zero values.
        amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> reduce_op;
        amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (int c = 0; c < 3; ++c) {
            auto const stag_own = stag[c];
            auto const stag_x = stag[0];
            auto const stag_y = stag[1];
            auto const stag_z = stag[2];

            for (amrex::MFIter mfi(*field[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                amrex::Box const tb = mfi.tilebox(field[c]->ixType().toIntVect());
                int const ncomp = field[c]->nComp();

                auto const& Jc = field[c]->array(mfi);
                auto const& Jx_o = Jold[0].const_array(mfi);
                auto const& Jy_o = Jold[1].const_array(mfi);
                auto const& Jz_o = Jold[2].const_array(mfi);
                auto const& mask = eb_update[c]->const_array(mfi);
                auto const& phi = distance_to_eb.const_array(mfi);

                reduce_op.eval(tb, ncomp, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) -> ReduceTuple
                {
                    if (mask(i, j, k) != 0) { return {0._rt, 0._rt}; }

                    // edge-center position in physical coordinates
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xe;
#if defined(WARPX_DIM_3D)
                    xe[0] = plo[0] + (i + stag_own[0])*dx_arr[0];
                    xe[1] = plo[1] + (j + stag_own[1])*dx_arr[1];
                    xe[2] = plo[2] + (k + stag_own[2])*dx_arr[2];
                    amrex::Real const yq = xe[1];
                    amrex::Real const zq = xe[2];
#else
                    xe[0] = plo[0] + (i + stag_own[0])*dx_arr[0];
                    xe[1] = plo[1] + (j + stag_own[1])*dx_arr[1];
                    amrex::Real const yq = 0._rt;
                    amrex::Real const zq = xe[1];
#endif

                    // signed distance at the edge center (< 0 in the conductor)
                    int ii, jj, kk;
                    amrex::Real W[AMREX_SPACEDIM][2];
                    ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                        xe[0], yq, zq, plo, dxi, ii, jj, kk, W);
                    amrex::Real const s = ablastr::particles::interp_field_nodal(ii, jj, kk, W, phi);

                    amrex::Real const v_old = Jc(i, j, k, n);

                    if (s < -d_band) {
                        // deep inside the conductor: the field vanishes
                        Jc(i, j, k, n) = 0._rt;
                        return {std::abs(v_old), 0._rt};
                    }

                    // boundary normal (toward the plasma) from the level set
                    int ic, jc, kc;
                    amrex::Real Wc[AMREX_SPACEDIM][2];
                    ablastr::particles::compute_weights<amrex::IndexType::CELL>(
                        xe[0], yq, zq, plo, dxi, ic, jc, kc, Wc);
                    amrex::RealVect nv = DistanceToEB::interp_normal(ii, jj, kk, W, ic, jc, kc, Wc, phi, dxi);
                    amrex::Real const nv2 = DistanceToEB::dot_product(nv, nv);
                    if (!(nv2 > 0._rt) || !std::isfinite(nv2)) {
                        // degenerate level-set gradient (e.g. saturated far
                        // field): treat as deep interior
                        Jc(i, j, k, n) = 0._rt;
                        return {std::abs(v_old), 0._rt};
                    }
                    DistanceToEB::normalize(nv);

                    // Image point at least one cell into the plasma so its
                    // stencil decouples from the boundary band (fast Jacobi
                    // convergence); d_im is its distance from the surface.
                    amrex::Real const offset =
                        amrex::max(amrex::max(std::abs(s), d_img_min) - s, h_max);
                    amrex::Real const d_im = s + offset;
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xim;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        xim[d] = xe[d] + offset*nv[d];
                    }

                    amrex::Real const Jx_im = gather_staggered(Jx_o, xim, stag_x, plo, dxi, n);
                    amrex::Real const Jy_im = gather_staggered(Jy_o, xim, stag_y, plo, dxi, n);
                    amrex::Real const Jz_im = gather_staggered(Jz_o, xim, stag_z, plo, dxi, n);

                    // edge fields (E, J): normal even / tangential odd;
                    // magnetic field: normal odd / tangential even. The
                    // magnetic normal weight is optionally clamped from below
                    // (default -1e30 = no clamp); see the direct-fill path.
                    amrex::Real const w_n = normal_odd
                        ? amrex::max(s/d_im, normal_weight_floor) : 1._rt;
                    amrex::Real const w_t = normal_odd ? 1._rt : s/d_im;
                    amrex::Real const lambda =
                        eb_cyl ? ::cyl_lambda(xe, xim, eb_cyl_axis) : 1._rt;
                    amrex::Real const v_new = ::mirror_combine(c, Jx_im, Jy_im, Jz_im,
                        nv, w_n, w_t, eb_cyl, eb_cyl_axis, lambda);

                    Jc(i, j, k, n) = v_new;

                    return {std::abs(v_new - v_old), std::abs(v_new)};
                });
            }
        }

        auto const result = reduce_data.value(reduce_op);
        amrex::Real dmax = amrex::get<0>(result);
        amrex::Real smax = amrex::get<1>(result);
        amrex::ParallelDescriptor::ReduceRealMax(dmax);
        amrex::ParallelDescriptor::ReduceRealMax(smax);
        resid = dmax / amrex::max(smax, std::numeric_limits<amrex::Real>::min());
        ++iter;
    }

    if (resid > rtol) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPIC",
            "ApplyPECBoundaryToField: the embedded-boundary field band "
            "relaxation did not reach the requested tolerance "
            "(hybrid_pic_model.eb_bc_rtol) within "
            "hybrid_pic_model.eb_bc_max_iters sweeps.",
            ablastr::warn_manager::WarnPriority::low);
    }

    // Leave ghost edges consistent with the final band values for the
    // stencils that consume J (nodal interpolation, hyper-resistivity)
    for (int c = 0; c < 3; ++c) {
        field[c]->FillBoundary(geom.periodicity());
    }
#else
    amrex::ignore_unused(field, eb_update, distance_to_eb, geom, rtol, max_iters,
                         direct_fill, normal_odd, fill_covered_centers, status_cache,
                         band_cells, eb_cyl, eb_cyl_axis);
#endif
}

// NOTE: This folder is collocated/nodal-specific by design: it assumes a fully
// nodal scalar (asserted below), uses zero staggering (stag0) and NODE-centered
// interpolation weights, and samples node coordinates without a half-cell
// offset. It is NOT Yee/staggered-aware. The staggered (Yee) deposit is folded
// by the separate FoldEBDepositToField, which carries a per-component staggering
// vector. A Yee-specific scalar folder is a follow-up if a staggered scalar
// deposit ever needs this treatment.
void warpx::hybrid::FoldEBDepositToNodalScalar (
    amrex::MultiFab& field,
    amrex::MultiFab const& distance_to_eb,
    amrex::Geometry const& geom,
    bool pec_images)
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    using namespace amrex::literals;

    ABLASTR_PROFILE("warpx::hybrid::FoldEBDepositToNodalScalar()");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(field.ixType().nodeCentered(),
        "FoldEBDepositToNodalScalar requires a fully nodal field");

    auto const plo = geom.ProbLoArray();
    auto const dxi = geom.InvCellSizeArray();
    auto const dx_arr = geom.CellSizeArray();

#if defined(WARPX_DIM_3D)
    amrex::Real const h_max = std::max({dx_arr[0], dx_arr[1], dx_arr[2]});
#else
    amrex::Real const h_max = std::max(dx_arr[0], dx_arr[1]);
#endif
    // deposit shape functions reach one cell past the surface; fold targets
    // mirror that reach on the fluid side
    amrex::Real const fold_band = 1.5_rt * h_max;
    amrex::Real const fold_sign = pec_images ? -1.0_rt : 1.0_rt;

    // covered-side ghosts must hold the guard-summed deposit
    field.FillBoundary(geom.periodicity());

    // The mirror gather reaches up to 2*fold_band past the surface plus one
    // stencil cell, which can exceed the field's own ghost width at box
    // seams near the wall (the stencil clamp would then silently read the
    // wrong cells). Gather from ghost-extended scratch copies instead; the
    // level-set scratch is initialized to a large positive (fluid) value so
    // unfilled physical-boundary ghosts never enter the covered set.
    int const ng_gather = 4;
    amrex::MultiFab f_src(field.boxArray(), field.DistributionMap(), field.nComp(),
                          amrex::IntVect(ng_gather));
    f_src.setVal(0.0_rt);
    amrex::MultiFab::Copy(f_src, field, 0, 0, field.nComp(), 0);
    f_src.FillBoundary(geom.periodicity());
    amrex::MultiFab phi_src(distance_to_eb.boxArray(), distance_to_eb.DistributionMap(),
                            1, amrex::IntVect(ng_gather));
    phi_src.setVal(std::numeric_limits<amrex::Real>::max());
    amrex::MultiFab::Copy(phi_src, distance_to_eb, 0, 0, 1, 0);
    phi_src.FillBoundary(geom.periodicity());

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const stag0{};

    for (amrex::MFIter mfi(field, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const tb = mfi.tilebox();
        int const ncomp = field.nComp();

        auto const& f = field.array(mfi);
        auto const& f_r = f_src.const_array(mfi);
        auto const& phi = distance_to_eb.const_array(mfi);
        auto const& phi_g = phi_src.const_array(mfi);

        amrex::ParallelFor(tb, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
        {
            // fluid points within the fold reach of the surface receive the
            // image of the deposit collected by the covered points (write
            // set phi > 0 and gather set phi <= 0 are disjoint)
            amrex::Real const s = phi(i, j, k);
            if (s <= 0._rt || s > fold_band) { return; }

            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xe;
#if defined(WARPX_DIM_3D)
            xe[0] = plo[0] + i*dx_arr[0];
            xe[1] = plo[1] + j*dx_arr[1];
            xe[2] = plo[2] + k*dx_arr[2];
            amrex::Real const yq = xe[1];
            amrex::Real const zq = xe[2];
#else
            xe[0] = plo[0] + i*dx_arr[0];
            xe[1] = plo[1] + j*dx_arr[1];
            amrex::Real const yq = 0._rt;
            amrex::Real const zq = xe[1];
#endif

            int ii, jj, kk;
            amrex::Real W[AMREX_SPACEDIM][2];
            ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                xe[0], yq, zq, plo, dxi, ii, jj, kk, W);
            int ic, jc, kc;
            amrex::Real Wc[AMREX_SPACEDIM][2];
            ablastr::particles::compute_weights<amrex::IndexType::CELL>(
                xe[0], yq, zq, plo, dxi, ic, jc, kc, Wc);
            amrex::RealVect nv = DistanceToEB::interp_normal(ii, jj, kk, W, ic, jc, kc, Wc, phi, dxi);
            amrex::Real const nv2 = DistanceToEB::dot_product(nv, nv);
            if (!(nv2 > 0._rt) || !std::isfinite(nv2)) { return; }
            DistanceToEB::normalize(nv);

            // exact mirror image of this point inside the conductor
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xm;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                xm[d] = xe[d] - 2._rt*s*nv[d];
            }

            // raw (un-renormalized) covered-only interpolation: fluid
            // stencil points contribute nothing, so only the deposit that
            // actually landed on covered points is folded
            auto const [v, w] = gather_staggered_pred(
                f_r,
                [&] (int ig, int jg, int kg) { return phi_g(ig, jg, kg) <= 0._rt; },
                xm, stag0, plo, dxi, n);

            // PEC image charge has the opposite sign (matches the domain
            // treatment in PEC::ApplyReflectiveBoundarytoRhofield); the
            // reflecting-wall parity adds the deposit back instead
            f(i, j, k, n) += fold_sign*v*w;
        });
    }
#else
    amrex::ignore_unused(field, distance_to_eb, geom, pec_images);
#endif
}

void warpx::hybrid::FoldEBDepositToField (
    ablastr::fields::VectorField const& field,
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update,
    amrex::MultiFab const& distance_to_eb,
    amrex::Geometry const& geom,
    EBFillStatus* status_cache,
    bool pec_images)
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    using namespace amrex::literals;

    ABLASTR_PROFILE("warpx::hybrid::FoldEBDepositToField()");

    if (!eb_update[0]) { return; }

    auto const plo = geom.ProbLoArray();
    auto const dxi = geom.InvCellSizeArray();
    auto const dx_arr = geom.CellSizeArray();

#if defined(WARPX_DIM_3D)
    amrex::Real const h_max = std::max({dx_arr[0], dx_arr[1], dx_arr[2]});
#else
    amrex::Real const h_max = std::max(dx_arr[0], dx_arr[1]);
#endif
    amrex::Real const d_band = h_max;
    amrex::Real const d_img_min = 0.5_rt * h_max;
    amrex::Real const fold_band = 1.5_rt * h_max;
    amrex::Real const fold_sign = pec_images ? -1.0_rt : 1.0_rt;

    std::array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, 3> stag{};
    for (int c = 0; c < 3; ++c) {
        auto const t = field[c]->ixType();
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            stag[c][d] = t.nodeCentered(d) ? 0.0_rt : 0.5_rt;
        }
    }

    // the covered set of the fold is exactly the write set of the fill with
    // covered-center cut edges included
    warpx::hybrid::EBFillStatus local_status;
    warpx::hybrid::EBFillStatus& st = status_cache ? *status_cache : local_status;
    if (st.empty()) {
        ::build_fill_status(st, field, eb_update, distance_to_eb, geom, stag,
                            d_band, d_img_min, h_max,
                            /*fill_covered_centers=*/true,
                            /*corner_skip=*/false);
    }

    for (int c = 0; c < 3; ++c) {
        field[c]->FillBoundary(geom.periodicity());
    }

    // Ghost-extended scratch copies for the mirror gather (its reach can
    // exceed the field/status ghost widths at box seams near the wall; the
    // stencil clamp would then silently read the wrong cells). The status
    // scratch is initialized to S_SOLUTION so unfilled physical-boundary
    // ghosts never enter the covered set.
    int const ng_gather = 4;
    std::array<amrex::MultiFab, 3> J_src;
    std::array<amrex::iMultiFab, 3> stat_src;
    for (int c = 0; c < 3; ++c) {
        J_src[c].define(field[c]->boxArray(), field[c]->DistributionMap(),
                        field[c]->nComp(), amrex::IntVect(ng_gather));
        J_src[c].setVal(0.0_rt);
        amrex::MultiFab::Copy(J_src[c], *field[c], 0, 0, field[c]->nComp(), 0);
        J_src[c].FillBoundary(geom.periodicity());
        stat_src[c].define(st.status[c]->boxArray(), st.status[c]->DistributionMap(),
                           1, amrex::IntVect(ng_gather));
        stat_src[c].setVal(S_SOLUTION);
        amrex::iMultiFab::Copy(stat_src[c], *st.status[c], 0, 0, 1, 0);
        stat_src[c].FillBoundary(geom.periodicity());
    }

    for (int c = 0; c < 3; ++c) {
        auto const stag_own = stag[c];
        auto const stag_x = stag[0];
        auto const stag_y = stag[1];
        auto const stag_z = stag[2];

        for (amrex::MFIter mfi(*field[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box const tb = mfi.tilebox(field[c]->ixType().toIntVect());
            int const ncomp = field[c]->nComp();

            auto const& Jc = field[c]->array(mfi);
            auto const& Jx_l = J_src[0].const_array(mfi);
            auto const& Jy_l = J_src[1].const_array(mfi);
            auto const& Jz_l = J_src[2].const_array(mfi);
            auto const& stat = st.status[c]->const_array(mfi);
            auto const& stat_x = stat_src[0].const_array(mfi);
            auto const& stat_y = stat_src[1].const_array(mfi);
            auto const& stat_z = stat_src[2].const_array(mfi);
            auto const& phi = distance_to_eb.const_array(mfi);

            amrex::ParallelFor(tb, ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                // fluid (solution) points near the surface receive the fold;
                // covered points are read only (disjoint sets, race-free)
                if (stat(i, j, k) != S_SOLUTION) { return; }

                auto const g = ::mirror_geom(i, j, k, stag_own, phi,
                    plo, dxi, dx_arr, d_band, d_img_min, h_max);
                if (g.s <= 0._rt || g.s > fold_band || !g.band) { return; }

                // exact mirror image of this point inside the conductor
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xm;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    xm[d] = g.xe[d] - 2._rt*g.s*g.nv[d];
                }

                // raw covered-only interpolation of the deposit per component
                auto const cov_x = [&] (int ig, int jg, int kg) { return stat_x(ig, jg, kg) != S_SOLUTION; };
                auto const cov_y = [&] (int ig, int jg, int kg) { return stat_y(ig, jg, kg) != S_SOLUTION; };
                auto const cov_z = [&] (int ig, int jg, int kg) { return stat_z(ig, jg, kg) != S_SOLUTION; };
                auto const [vx, wx] = gather_staggered_pred(Jx_l, cov_x, xm, stag_x, plo, dxi, n);
                auto const [vy, wy] = gather_staggered_pred(Jy_l, cov_y, xm, stag_y, plo, dxi, n);
                auto const [vz, wz] = gather_staggered_pred(Jz_l, cov_z, xm, stag_z, plo, dxi, n);
                amrex::Real const gx = vx*wx;
                amrex::Real const gy = vy*wy;
                amrex::Real const gz = vz*wz;

#if defined(WARPX_DIM_3D)
                amrex::Real const ndotg = g.nv[0]*gx + g.nv[1]*gy + g.nv[2]*gz;
                amrex::Real const e_dot_n = g.nv[c];
#else
                amrex::Real const ndotg = g.nv[0]*gx + g.nv[1]*gz;
                amrex::Real const e_dot_n = (c == 0) ? g.nv[0] : ((c == 2) ? g.nv[1] : 0._rt);
#endif
                amrex::Real const g_e = (c == 0) ? gx : ((c == 1) ? gy : gz);

                // PEC image current: normal part added, tangential part
                // subtracted (matches PEC::ApplyReflectiveBoundarytoJfield);
                // the reflecting-wall parities are the exact opposite
                Jc(i, j, k, n) += fold_sign*((g_e - ndotg*e_dot_n) - ndotg*e_dot_n);
            });
        }
    }
#else
    amrex::ignore_unused(field, eb_update, distance_to_eb, geom, status_cache, pec_images);
#endif
}

void warpx::hybrid::ApplyEBBoundaryToNodalScalar (
    amrex::MultiFab& field,
    amrex::MultiFab const& distance_to_eb,
    amrex::Geometry const& geom,
    bool odd)
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    using namespace amrex::literals;

    ABLASTR_PROFILE("warpx::hybrid::ApplyEBBoundaryToNodalScalar()");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(field.ixType().nodeCentered(),
        "ApplyEBBoundaryToNodalScalar requires a fully nodal field");

    auto const plo = geom.ProbLoArray();
    auto const dxi = geom.InvCellSizeArray();
    auto const dx_arr = geom.CellSizeArray();

#if defined(WARPX_DIM_3D)
    amrex::Real const h_max = std::max({dx_arr[0], dx_arr[1], dx_arr[2]});
#else
    amrex::Real const h_max = std::max(dx_arr[0], dx_arr[1]);
#endif
    amrex::Real const d_band = h_max;
    amrex::Real const d_img_min = 0.5_rt * h_max;

    // Image-point gathers can reach fluid nodes owned by neighboring boxes.
    // The write set (nodes on or inside the surface) and the gather set
    // (fluid nodes) are disjoint, so a single deterministic pass is exact.
    field.FillBoundary(geom.periodicity());

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const stag0{};

    for (amrex::MFIter mfi(field, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const tb = mfi.tilebox();
        int const ncomp = field.nComp();

        auto const& f = field.array(mfi);
        auto const& f_r = field.const_array(mfi);
        auto const& phi = distance_to_eb.const_array(mfi);

        amrex::ParallelFor(tb, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
        {
            // The nodal level set gives the signed distance directly (< 0 in
            // the conductor). Fluid nodes are never modified; on-surface nodes
            // (s == 0) are written so the odd parity pins them to zero.
            amrex::Real const s = phi(i, j, k);
            if (s > 0._rt) { return; }

            if (s < -d_band) {
                f(i, j, k, n) = 0._rt;
                return;
            }

            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xe;
#if defined(WARPX_DIM_3D)
            xe[0] = plo[0] + i*dx_arr[0];
            xe[1] = plo[1] + j*dx_arr[1];
            xe[2] = plo[2] + k*dx_arr[2];
            amrex::Real const yq = xe[1];
            amrex::Real const zq = xe[2];
#else
            xe[0] = plo[0] + i*dx_arr[0];
            xe[1] = plo[1] + j*dx_arr[1];
            amrex::Real const yq = 0._rt;
            amrex::Real const zq = xe[1];
#endif

            // boundary normal (toward the plasma) from the level set
            int ii, jj, kk;
            amrex::Real W[AMREX_SPACEDIM][2];
            ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                xe[0], yq, zq, plo, dxi, ii, jj, kk, W);
            int ic, jc, kc;
            amrex::Real Wc[AMREX_SPACEDIM][2];
            ablastr::particles::compute_weights<amrex::IndexType::CELL>(
                xe[0], yq, zq, plo, dxi, ic, jc, kc, Wc);
            amrex::RealVect nv = DistanceToEB::interp_normal(ii, jj, kk, W, ic, jc, kc, Wc, phi, dxi);
            amrex::Real const nv2 = DistanceToEB::dot_product(nv, nv);
            if (!(nv2 > 0._rt) || !std::isfinite(nv2)) {
                // degenerate level-set gradient: treat as deep interior
                f(i, j, k, n) = 0._rt;
                return;
            }
            DistanceToEB::normalize(nv);

            // Image point at the exact mirror distance, regularized near the
            // surface so the stencil retains fluid nodes; the gather never
            // reads written nodes, so no decoupling offset is needed.
            amrex::Real const d_im = amrex::max(std::abs(s), d_img_min);
            amrex::Real const offset = d_im - s;
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xim;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                xim[d] = xe[d] + offset*nv[d];
            }

            // field value at the image point from fluid nodes only; if the
            // whole stencil is covered (thin gap, sharp corner) this is zero
            auto const [f_im, w_im] = gather_staggered_pred(
                f_r,
                [&] (int ig, int jg, int kg) { return phi(ig, jg, kg) > 0._rt; },
                xim, stag0, plo, dxi, n);
            amrex::ignore_unused(w_im);

            // odd: value vanishes at the surface (Dirichlet 0, ghost values
            // change sign across the wall); even: zero normal gradient
            f(i, j, k, n) = odd ? (s/d_im)*f_im : f_im;
        });
    }

    // leave ghost nodes consistent for the stencils that consume the field
    field.FillBoundary(geom.periodicity());
#else
    amrex::ignore_unused(field, distance_to_eb, geom, odd);
#endif
}
