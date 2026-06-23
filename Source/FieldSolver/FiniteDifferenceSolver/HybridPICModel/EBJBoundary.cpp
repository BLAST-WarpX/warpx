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
#include <limits>

using namespace amrex;

namespace
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    /** Trilinear (bilinear in 2D) gather of component n of a staggered field
     *  at an arbitrary position (physical coordinates), clamping the stencil
     *  to the array bounds (constant extrapolation past the available ghosts). */
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

    /** In-place Cholesky solve of a small symmetric positive-definite system
     *  M x = b (result returned in b). Returns false if M is not SPD (caller
     *  falls back). Fixed size, no allocation -> GPU friendly. */
    template <int N>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    bool solve_spd (amrex::Real M[N][N], amrex::Real b[N]) noexcept
    {
        using namespace amrex::literals;
        amrex::Real L[N][N];
        for (int i = 0; i < N; ++i) { for (int j = 0; j < N; ++j) { L[i][j] = 0._rt; } }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j <= i; ++j) {
                amrex::Real sum = M[i][j];
                for (int k = 0; k < j; ++k) { sum -= L[i][k]*L[j][k]; }
                if (i == j) {
                    if (!(sum > 0._rt)) { return false; }
                    L[i][j] = std::sqrt(sum);
                } else {
                    L[i][j] = sum / L[j][j];
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
        int n) noexcept
    {
        using namespace amrex::literals;
        constexpr int W = 2;  // +/- window (5 cells per direction)

        amrex::Real lc[AMREX_SPACEDIM];
        int ic[AMREX_SPACEDIM];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            lc[d] = (pos[d] - plo[d])*dxi[d] - stag[d];
            ic[d] = static_cast<int>(std::floor(lc[d] + 0.5_rt));
        }

        amrex::Real M[QUAD_NB][QUAD_NB];
        amrex::Real bb[QUAD_NB];
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
                for (int c = 0; c <= r; ++c) { M[r][c] += p[r]*p[c]; }
            }
            ++count;
        };

        // Grow the window (+/-2 -> +/-4) until the quadratic is comfortably
        // overdetermined. A one-sided near-wall stencil (a covered node can sit
        // on the surface, so its fluid neighbors are all to one side) needs more
        // points than the minimum to stay well-conditioned -- the fixed small
        // window is what limited the nodal grid.
        constexpr int MIN_PTS = 2*QUAD_NB;
        for (int W = 2; W <= 4; ++W) {
            count = 0;
            for (int r = 0; r < QUAD_NB; ++r) {
                bb[r] = 0._rt;
                for (int c = 0; c < QUAD_NB; ++c) { M[r][c] = 0._rt; }
            }
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
            if (count >= MIN_PTS) { break; }
        }
        if (count < QUAD_NB) { return {0._rt, 0._rt}; }

        // symmetrize and add a per-diagonal (relative) ridge so an
        // under-determined one-sided stencil regularizes toward its lower-order
        // fit instead of overshooting (the device analog of SVD truncation)
        amrex::Real const ridge = 1.e-6_rt * M[0][0];
        for (int r = 0; r < QUAD_NB; ++r) {
            for (int c = r+1; c < QUAD_NB; ++c) { M[r][c] = M[c][r]; }
            M[r][r] += ridge;
        }
        if (!solve_spd<QUAD_NB>(M, bb)) { return {0._rt, 0._rt}; }
        return {bb[0], static_cast<amrex::Real>(count)};  // fit value at pos (u=0)
    }

    // Classification of every staggered point for the embedded-boundary fill
    constexpr int S_SOLUTION = 0;  //!< solution domain: gather source, never written
    constexpr int S_FILL     = 1;  //!< fill target with a well-posed image stencil
    constexpr int S_PENDING  = 2;  //!< fill target with an ill-posed image stencil
    constexpr int S_DEEP     = 3;  //!< deep inside the conductor (or degenerate normal): zeroed
    constexpr int S_RESOLVED = 4;  //!< well-posed target written and locked during this call
    constexpr int S_JUSTDONE = 5;  //!< pending target written in the current cascade sweep
    constexpr int S_RESOLVED_P = 6; //!< pending target written and locked during this call

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
        bool fill_covered_centers)
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
    bool quadratic_gather)
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    using namespace amrex::literals;

    ABLASTR_PROFILE("warpx::hybrid::ApplyPECBoundaryToField()");

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
                                d_band, d_img_min, h_max, fill_covered_centers);
        }

        for (int c = 0; c < 3; ++c) {
            field[c]->FillBoundary(geom.periodicity());
        }

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

                amrex::ParallelFor(tb, ncomp,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                    int const s0 = stat(i, j, k);
                    if (s0 == S_DEEP) {
                        Jc(i, j, k, n) = 0._rt;
                        return;
                    }
                    if (s0 != S_FILL) { return; }

                    auto const g = ::mirror_geom(i, j, k, stag_own, phi,
                        plo, dxi, dx_arr, d_band, d_img_min, h_max);

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
                    if (quadratic_gather) {
                        auto const [vx, cx] = gather_quadratic_pred(Jx_l, in_sol_x, xim, stag_x, plo, dxi, n);
                        auto const [vy, cy] = gather_quadratic_pred(Jy_l, in_sol_y, xim, stag_y, plo, dxi, n);
                        auto const [vz, cz] = gather_quadratic_pred(Jz_l, in_sol_z, xim, stag_z, plo, dxi, n);
                        // fall back to the linear gather where the quadratic
                        // stencil is degenerate (too few fluid points / non-SPD)
                        Jx_im = (cx > 0._rt) ? vx : amrex::get<0>(gather_staggered_pred(Jx_l, in_sol_x, xim, stag_x, plo, dxi, n));
                        Jy_im = (cy > 0._rt) ? vy : amrex::get<0>(gather_staggered_pred(Jy_l, in_sol_y, xim, stag_y, plo, dxi, n));
                        Jz_im = (cz > 0._rt) ? vz : amrex::get<0>(gather_staggered_pred(Jz_l, in_sol_z, xim, stag_z, plo, dxi, n));
                    } else {
                        Jx_im = amrex::get<0>(gather_staggered_pred(Jx_l, in_sol_x, xim, stag_x, plo, dxi, n));
                        Jy_im = amrex::get<0>(gather_staggered_pred(Jy_l, in_sol_y, xim, stag_y, plo, dxi, n));
                        Jz_im = amrex::get<0>(gather_staggered_pred(Jz_l, in_sol_z, xim, stag_z, plo, dxi, n));
                    }

                    // edge fields (E, J): normal even / tangential odd;
                    // magnetic field: normal odd / tangential even
                    amrex::Real const w_n = normal_odd ? g.s/d_im : 1._rt;
                    amrex::Real const w_t = normal_odd ? 1._rt : g.s/d_im;
                    // optional surface-of-revolution radial metric Jacobian
                    amrex::Real const lambda =
                        eb_cyl ? ::cyl_lambda(g.xe, xim, eb_cyl_axis) : 1._rt;
                    Jc(i, j, k, n) = ::mirror_combine(c, Jx_im, Jy_im, Jz_im, g.nv,
                        w_n, w_t, eb_cyl, eb_cyl_axis, lambda);
                });
            }
        }

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
                                amrex::Real const w_n = normal_odd ? g.s/g.d_im : 1._rt;
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
                    // magnetic field: normal odd / tangential even
                    amrex::Real const w_n = normal_odd ? s/d_im : 1._rt;
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
                            /*fill_covered_centers=*/true);
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
