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
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_RealVect.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <array>
#include <cmath>

using namespace amrex;

namespace
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    /** Trilinear (bilinear in 2D) gather of component n of a staggered field
     *  at an arbitrary position (grid coordinates), clamping the stencil to
     *  the array bounds (constant extrapolation past the available ghosts). */
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

    /** Like gather_staggered, but using only unmasked (solution-domain)
     *  points: masked stencil points get zero weight and the result is
     *  renormalized by the remaining weight, which is also returned (zero if
     *  the entire stencil is masked). */
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
                    if (msk(i0+di, j0+dj, k0+dk) != 0) {
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
                if (msk(i0+di, j0+dj, 0) != 0) {
                    vsum += w*a(i0+di, j0+dj, 0, n);
                    wsum += w;
                }
            }
        }
#endif
        amrex::Real const v = (wsum > 0._rt) ? vsum/wsum : 0._rt;
        return {v, wsum};
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
    bool normal_odd)
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
    // Mirror-fill masked edges within one cell of the surface (the only ones
    // reached by stencils that straddle the wall) and zero everything deeper.
    // The minimum image distance keeps the interpolation point in the plasma
    // for edges that sit very close to the surface.
    amrex::Real const d_band = h_max;
    amrex::Real const d_img_min = 0.5_rt * h_max;

    // Staggering offsets in grid coordinates for each J component (0.5 in
    // directions where the component is cell-centered)
    std::array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, 3> stag{};
    for (int c = 0; c < 3; ++c) {
        auto const t = field[c]->ixType();
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            stag[c][d] = t.nodeCentered(d) ? 0.0_rt : 0.5_rt;
        }
    }

    if (direct_fill) {
        // Single deterministic pass: the mirrored interpolation uses only
        // unmasked (solution-domain) values, with the stencil weights
        // renormalized over the unmasked points, so the ghost values never
        // depend on each other and no relaxation is required
        for (int c = 0; c < 3; ++c) {
            field[c]->FillBoundary(geom.periodicity());
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
                auto const& Jx_l = field[0]->const_array(mfi);
                auto const& Jy_l = field[1]->const_array(mfi);
                auto const& Jz_l = field[2]->const_array(mfi);
                auto const& mask = eb_update[c]->const_array(mfi);
                auto const& mask_x = eb_update[0]->const_array(mfi);
                auto const& mask_y = eb_update[1]->const_array(mfi);
                auto const& mask_z = eb_update[2]->const_array(mfi);
                auto const& phi = distance_to_eb.const_array(mfi);

                amrex::ParallelFor(tb, ncomp,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                    if (mask(i, j, k) != 0) { return; }

                    // edge-center position in grid coordinates
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

                    if (s < -d_band) {
                        // deep inside the conductor: no volume current
                        Jc(i, j, k, n) = 0._rt;
                        return;
                    }

                    // boundary normal (toward the plasma) from the level set
                    int ic, jc, kc;
                    amrex::Real Wc[AMREX_SPACEDIM][2];
                    ablastr::particles::compute_weights<amrex::IndexType::CELL>(
                        xe[0], yq, zq, plo, dxi, ic, jc, kc, Wc);
                    amrex::RealVect nv = DistanceToEB::interp_normal(ii, jj, kk, W, ic, jc, kc, Wc, phi, dxi);
                    amrex::Real const nv2 = DistanceToEB::dot_product(nv, nv);
                    if (!(nv2 > 0._rt) || !std::isfinite(nv2)) {
                        Jc(i, j, k, n) = 0._rt;
                        return;
                    }
                    DistanceToEB::normalize(nv);

                    // image point in the plasma; d_im is its distance from
                    // the surface, used for the linear tangential profile
                    amrex::Real const offset =
                        amrex::max(amrex::max(std::abs(s), d_img_min) - s, h_max);
                    amrex::Real const d_im = s + offset;
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xim;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        xim[d] = xe[d] + offset*nv[d];
                    }

                    // field vector at the image point from unmasked values
                    auto const [Jx_im, wx_im] = gather_staggered_masked(Jx_l, mask_x, xim, stag_x, plo, dxi, n);
                    auto const [Jy_im, wy_im] = gather_staggered_masked(Jy_l, mask_y, xim, stag_y, plo, dxi, n);
                    auto const [Jz_im, wz_im] = gather_staggered_masked(Jz_l, mask_z, xim, stag_z, plo, dxi, n);
                    amrex::ignore_unused(wx_im, wy_im, wz_im);

#if defined(WARPX_DIM_3D)
                    amrex::Real const ndotJ = nv[0]*Jx_im + nv[1]*Jy_im + nv[2]*Jz_im;
                    amrex::Real const e_dot_n = nv[c];
#else
                    // out-of-plane component (y in XZ, theta in RZ) is purely tangential
                    amrex::Real const ndotJ = nv[0]*Jx_im + nv[1]*Jz_im;
                    amrex::Real const e_dot_n = (c == 0) ? nv[0] : ((c == 2) ? nv[1] : 0._rt);
#endif
                    amrex::Real const J_im_e = (c == 0) ? Jx_im : ((c == 1) ? Jy_im : Jz_im);

                    // edge fields (E, J): normal even / tangential odd;
                    // magnetic field: normal odd / tangential even
                    amrex::Real const w_n = normal_odd ? s/d_im : 1._rt;
                    amrex::Real const w_t = normal_odd ? 1._rt : s/d_im;
                    Jc(i, j, k, n) = w_n*ndotJ*e_dot_n + w_t*(J_im_e - ndotJ*e_dot_n);
                });
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

        // Track both the largest change and the largest band value so that
        // convergence is measured relative to the boundary-band field
        // magnitude (a per-edge relative criterion can never be met by edges
        // holding near-zero values)
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

                    // edge-center position in grid coordinates
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
                        // deep inside the conductor: no volume current
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

                    // Image point in the plasma, at least one cell away from
                    // this edge so that the interpolation stencil decouples
                    // from the boundary band (the Jacobi relaxation then
                    // converges quickly); d_im is its distance from the
                    // surface, used for the linear tangential profile
                    amrex::Real const offset =
                        amrex::max(amrex::max(std::abs(s), d_img_min) - s, h_max);
                    amrex::Real const d_im = s + offset;
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xim;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        xim[d] = xe[d] + offset*nv[d];
                    }

                    // full current vector at the image point
                    amrex::Real const Jx_im = gather_staggered(Jx_o, xim, stag_x, plo, dxi, n);
                    amrex::Real const Jy_im = gather_staggered(Jy_o, xim, stag_y, plo, dxi, n);
                    amrex::Real const Jz_im = gather_staggered(Jz_o, xim, stag_z, plo, dxi, n);

#if defined(WARPX_DIM_3D)
                    amrex::Real const ndotJ = nv[0]*Jx_im + nv[1]*Jy_im + nv[2]*Jz_im;
                    amrex::Real const e_dot_n = nv[c];
#else
                    // out-of-plane component (y in XZ, theta in RZ) is purely tangential
                    amrex::Real const ndotJ = nv[0]*Jx_im + nv[1]*Jz_im;
                    amrex::Real const e_dot_n = (c == 0) ? nv[0] : ((c == 2) ? nv[1] : 0._rt);
#endif
                    amrex::Real const J_im_e = (c == 0) ? Jx_im : ((c == 1) ? Jy_im : Jz_im);

                    // edge fields (E, J): normal even / tangential odd;
                    // magnetic field: normal odd / tangential even
                    amrex::Real const w_n = normal_odd ? s/d_im : 1._rt;
                    amrex::Real const w_t = normal_odd ? 1._rt : s/d_im;
                    amrex::Real const v_new = w_n*ndotJ*e_dot_n + w_t*(J_im_e - ndotJ*e_dot_n);

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
                         direct_fill, normal_odd);
#endif
}
