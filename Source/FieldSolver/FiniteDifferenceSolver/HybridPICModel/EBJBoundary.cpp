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
                        // updated by the solver; with fill_covered_centers,
                        // cut points whose centers are on or inside the
                        // surface are fill targets anyway (the solver
                        // evaluates them at their covered centers)
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
    EBFillStatus* status_cache)
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
        warpx::hybrid::EBFillStatus local_status;
        warpx::hybrid::EBFillStatus& st = status_cache ? *status_cache : local_status;
        if (st.empty()) {
            ::build_fill_status(st, field, eb_update, distance_to_eb, geom, stag,
                                d_band, d_img_min, h_max, fill_covered_centers);
        }

        for (int c = 0; c < 3; ++c) {
            field[c]->FillBoundary(geom.periodicity());
        }

        // Whether the cascade below runs (the status arrays are only mutated
        // when locked values must be distinguished from pending ones)
        bool const cascade = (st.n_pending > 0);

        // Direct pass: deterministic mirror fill of the well-posed targets,
        // gathering only from solution-domain values (so neither other fill
        // targets nor covered-center cut points contaminate the image);
        // deep points are zeroed
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

                    // field vector at the image point from solution values
                    auto const in_sol_x =
                        [&] (int ig, int jg, int kg) { return stat_x(ig, jg, kg) == S_SOLUTION; };
                    auto const in_sol_y =
                        [&] (int ig, int jg, int kg) { return stat_y(ig, jg, kg) == S_SOLUTION; };
                    auto const in_sol_z =
                        [&] (int ig, int jg, int kg) { return stat_z(ig, jg, kg) == S_SOLUTION; };
                    auto const [Jx_im, wx_im] = gather_staggered_pred(Jx_l, in_sol_x, g.xim, stag_x, plo, dxi, n);
                    auto const [Jy_im, wy_im] = gather_staggered_pred(Jy_l, in_sol_y, g.xim, stag_y, plo, dxi, n);
                    auto const [Jz_im, wz_im] = gather_staggered_pred(Jz_l, in_sol_z, g.xim, stag_z, plo, dxi, n);
                    amrex::ignore_unused(wx_im, wy_im, wz_im);

#if defined(WARPX_DIM_3D)
                    amrex::Real const ndotJ = g.nv[0]*Jx_im + g.nv[1]*Jy_im + g.nv[2]*Jz_im;
                    amrex::Real const e_dot_n = g.nv[c];
#else
                    // out-of-plane component (y in XZ, theta in RZ) is purely tangential
                    amrex::Real const ndotJ = g.nv[0]*Jx_im + g.nv[1]*Jz_im;
                    amrex::Real const e_dot_n = (c == 0) ? g.nv[0] : ((c == 2) ? g.nv[1] : 0._rt);
#endif
                    amrex::Real const J_im_e = (c == 0) ? Jx_im : ((c == 1) ? Jy_im : Jz_im);

                    // edge fields (E, J): normal even / tangential odd;
                    // magnetic field: normal odd / tangential even
                    amrex::Real const w_n = normal_odd ? g.s/g.d_im : 1._rt;
                    amrex::Real const w_t = normal_odd ? 1._rt : g.s/g.d_im;
                    Jc(i, j, k, n) = w_n*ndotJ*e_dot_n + w_t*(J_im_e - ndotJ*e_dot_n);
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
                            // require every component stencil to reach at
                            // least one locked value
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
#if defined(WARPX_DIM_3D)
                                amrex::Real const ndotJ = g.nv[0]*Jx_im + g.nv[1]*Jy_im + g.nv[2]*Jz_im;
                                amrex::Real const e_dot_n = g.nv[c];
#else
                                amrex::Real const ndotJ = g.nv[0]*Jx_im + g.nv[1]*Jz_im;
                                amrex::Real const e_dot_n = (c == 0) ? g.nv[0] : ((c == 2) ? g.nv[1] : 0._rt);
#endif
                                amrex::Real const J_im_e = (c == 0) ? Jx_im : ((c == 1) ? Jy_im : Jz_im);
                                amrex::Real const w_n = normal_odd ? g.s/g.d_im : 1._rt;
                                amrex::Real const w_t = normal_odd ? 1._rt : g.s/g.d_im;
                                Jc(i, j, k, n) = w_n*ndotJ*e_dot_n + w_t*(J_im_e - ndotJ*e_dot_n);
                            }
                            stat(i, j, k) = S_JUSTDONE;
                            return {1};
                        });
                    }
                }

                auto const sweep_result = rdata.value(rop);
                int n_done = amrex::get<0>(sweep_result);
                amrex::ParallelDescriptor::ReduceIntSum(n_done);

                // promote this sweep's results to locked values
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
                // fully enclosed by other pending points: no meaningful
                // mirror value exists, zero them
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
                         direct_fill, normal_odd, fill_covered_centers, status_cache);
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
                [&] (int ig, int jg, int kg) { return phi(ig, jg, kg) <= 0._rt; },
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
            // The level set is collocated with the field, so the signed
            // distance at this node is just the nodal value (< 0 in the
            // conductor). Fluid nodes are never modified; on-surface nodes
            // (s == 0) are written so the odd parity pins them to zero.
            amrex::Real const s = phi(i, j, k);
            if (s > 0._rt) { return; }

            if (s < -d_band) {
                // deep inside the conductor
                f(i, j, k, n) = 0._rt;
                return;
            }

            // node position
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

            // image point in the plasma at the exact mirror distance
            // (regularized very close to the surface so the interpolation
            // stencil retains fluid nodes); unlike the staggered vector
            // fill, the gather below never touches written nodes, so the
            // image needs no additional decoupling offset
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
