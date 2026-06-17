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

#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
    // Collocated-ECT classification of a node's cut dual face
    constexpr int C_INTERIOR = 0;  //!< all four dual-face corners in the fluid
    constexpr int C_BAND     = 1;  //!< the level set cuts the dual face
    constexpr int C_DEEP     = 2;  //!< all four corners inside the conductor

    /** Cut geometry and circulation of one node's dual face for the C-ECT
     *  conformal B correction (one B component, in-plane axes a and b).
     *
     *  The four dual-face corners (CCW: SW, SE, NE, NW, at +/- h/2 about the
     *  node in the (a, b) plane) carry the level-set values phi[4]. The cut is
     *  the marching-squares clip of the dual square against the fluid half-space
     *  {phi >= 0}: cut edge-lengths l = h*(phi>0 fraction), cut area S_cut from
     *  a Sutherland-Hodgman/shoelace clip, ECT stable area S^stb = 0.5*h*max(l),
     *  effective area S_eff = max(S_cut, S^stb). The circulation V_cut sums the
     *  midpoint-averaged in-plane E along the cut edges, oriented (per the
     *  full-cell identity, see ApplyConformalBCorrection) so that
     *  dt*V_cut/S_eff -> dt*(UpwardD_b(E_a) - UpwardD_a(E_b)) when uncut.
     *
     *  The edge-midpoint E samples are: E_a along the bottom/top edges
     *  (b = -/+ h/2), averaged between the node and its +/-1 neighbor in b;
     *  E_b along the left/right edges (a = -/+ h/2), averaged between the node
     *  and its +/-1 neighbor in a. */
    struct CutResult
    {
        int klass;          //!< C_INTERIOR / C_BAND / C_DEEP
        amrex::Real dB_fv;  //!< dt*V_cut/S_eff (valid only for C_BAND)
    };

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real cut_edge_len (amrex::Real pa, amrex::Real pb, amrex::Real h) noexcept
    {
        using namespace amrex::literals;
        if (pa > 0._rt && pb > 0._rt) { return h; }
        if (pa <= 0._rt && pb <= 0._rt) { return 0._rt; }
        amrex::Real const pos = (pa > 0._rt) ? pa : pb;
        amrex::Real const neg = (pa > 0._rt) ? pb : pa;
        return h * pos / (pos - neg);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    CutResult cut_cell (
        amrex::Real const phi[4],   // CCW corners: SW, SE, NE, NW
        amrex::Real const Ea[3],    // E_a at b = {-1, 0, +1} neighbors of the node
        amrex::Real const Eb[3],    // E_b at a = {-1, 0, +1} neighbors of the node
        amrex::Real ha, amrex::Real hb, amrex::Real dt) noexcept
    {
        using namespace amrex::literals;

        amrex::Real phi_min = phi[0];
        amrex::Real phi_max = phi[0];
        for (int n = 1; n < 4; ++n) {
            phi_min = amrex::min(phi_min, phi[n]);
            phi_max = amrex::max(phi_max, phi[n]);
        }
        if (phi_min > 0._rt) { return {C_INTERIOR, 0._rt}; }
        if (phi_max <= 0._rt) { return {C_DEEP, 0._rt}; }

        // Cut edge-lengths. Edge order matches the corner pairs:
        //   bottom  SW->SE (b = -h/2, runs in a), top   NW->NE (b = +h/2),
        //   left    SW->NW (a = -h/2, runs in b), right SE->NE (a = +h/2).
        amrex::Real const l_bot   = cut_edge_len(phi[0], phi[1], ha);  // a-edge
        amrex::Real const l_top   = cut_edge_len(phi[3], phi[2], ha);  // a-edge
        amrex::Real const l_left  = cut_edge_len(phi[0], phi[3], hb);  // b-edge
        amrex::Real const l_right = cut_edge_len(phi[1], phi[2], hb);  // b-edge

        // Cut area: Sutherland-Hodgman clip of the unit dual square against
        // {phi >= 0} (phi linear along each edge), shoelace of the clipped
        // polygon. Local corner coordinates are +/- h/2 in (a, b).
        amrex::Real const cx[4] = {-0.5_rt*ha,  0.5_rt*ha, 0.5_rt*ha, -0.5_rt*ha};
        amrex::Real const cy[4] = {-0.5_rt*hb, -0.5_rt*hb, 0.5_rt*hb,  0.5_rt*hb};
        amrex::Real px[8];
        amrex::Real py[8];
        int np = 0;
        for (int e = 0; e < 4; ++e) {
            int const f = (e + 1) % 4;
            amrex::Real const pe = phi[e];
            amrex::Real const pf = phi[f];
            if (pe >= 0._rt) {
                px[np] = cx[e];
                py[np] = cy[e];
                ++np;
            }
            if ((pe >= 0._rt) != (pf >= 0._rt)) {
                amrex::Real const t = pe / (pe - pf);
                px[np] = cx[e] + t*(cx[f] - cx[e]);
                py[np] = cy[e] + t*(cy[f] - cy[e]);
                ++np;
            }
        }
        amrex::Real shoelace = 0._rt;
        for (int e = 0; e < np; ++e) {
            int const f = (e + 1) % np;
            shoelace += px[e]*py[f] - px[f]*py[e];
        }
        amrex::Real const S_cut = 0.5_rt * std::abs(shoelace);

        // ECT stable area (bounds dt*V/S, no small-cut blow-up). Per-axis form
        // of the paper's max(side)*h/2: an a-edge of length l backs an area
        // l*hb, a b-edge backs l*ha; reduces to 0.5*h*max(l) for square cells.
        amrex::Real const S_eff = amrex::max(S_cut,
            0.5_rt * amrex::max(amrex::max(l_bot, l_top) * hb,
                                amrex::max(l_left, l_right) * ha));

        // Midpoint-averaged in-plane E along the cut edges (uncut: l = h)
        amrex::Real const Ea_bot = 0.5_rt*(Ea[1] + Ea[0]);  // node and -b neighbor
        amrex::Real const Ea_top = 0.5_rt*(Ea[1] + Ea[2]);  // node and +b neighbor
        amrex::Real const Eb_left  = 0.5_rt*(Eb[1] + Eb[0]); // node and -a neighbor
        amrex::Real const Eb_right = 0.5_rt*(Eb[1] + Eb[2]); // node and +a neighbor

        // Oriented circulation (full-cell limit equals UpwardD_b(E_a) -
        // UpwardD_a(E_b); orientation verified per component).
        amrex::Real const V_cut = Ea_top*l_top - Ea_bot*l_bot
                                + Eb_left*l_left - Eb_right*l_right;

        return {C_BAND, dt * V_cut / S_eff};
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

void warpx::hybrid::ApplyConformalBCorrection (
    ablastr::fields::VectorField const& Bfield,
    ablastr::fields::VectorField const& Efield,
    amrex::MultiFab const& distance_to_eb,
    amrex::Geometry const& geom,
    amrex::Real dt)
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
    using namespace amrex::literals;

    ABLASTR_PROFILE("warpx::hybrid::ApplyConformalBCorrection()");

    auto const dx_arr = geom.CellSizeArray();

    // Each B component's cut geometry uses the two in-plane cell sizes
    // (ha, hb) = (dx[grid_a], dx[grid_b]) of its Faraday face, so anisotropic
    // (e.g. dx = dy != dz) collocated grids are supported per face; the
    // full-cell identity dB_fv == dB_nodal holds for any (ha, hb).

    // The dual-corner average and the E-edge samples reach one node past the
    // node loop in each in-plane direction; the caller FillBoundary's the level
    // set (>= 2 valid ghosts) and the E field before this call.

    // Per B component, the in-plane axes (a, b) of its Faraday curl
    // UpwardD_b(E_a) - UpwardD_a(E_b) (matching EvolveBCartesian, e.g.
    // Bx += dt*(UpwardDz(Ey) - UpwardDy(Ez))):
    //   Bx -> (y, z), By -> (z, x), Bz -> (x, y).
    // Two index spaces are needed and they differ in 2D XZ:
    //   - the field-component index into Efield (Ex = 0, Ey = 1, Ez = 2);
    //   - the grid-axis index into dx and the node-offset IntVect (XZ: x = 0,
    //     z = 1; the y axis is absent).
    constexpr int comp_a[3] = {1, 2, 0};  // Ey, Ez, Ex
    constexpr int comp_b[3] = {2, 0, 1};  // Ez, Ex, Ey
#if defined(WARPX_DIM_3D)
    constexpr int grid_a[3] = {1, 2, 0};  // y, z, x
    constexpr int grid_b[3] = {2, 0, 1};  // z, x, y
    constexpr int comp_lo = 0;
    constexpr int comp_hi = 3;
#else
    // 2D XZ: correct only the out-of-plane By (in-plane curl circ(Ez, Ex),
    // axes a = z, b = x). The in-plane Bx, Bz keep the core nodal push (their
    // curl uses the out-of-plane Ey through a 1-D derivative).
    constexpr int grid_a[3] = {0, 1, 0};  // a = z -> grid axis 1 (for By)
    constexpr int grid_b[3] = {0, 0, 0};  // b = x -> grid axis 0 (for By)
    constexpr int comp_lo = 1;
    constexpr int comp_hi = 2;
#endif

    for (int c = comp_lo; c < comp_hi; ++c) {
        int const ca = comp_a[c];
        int const cb = comp_b[c];
        int const ga = grid_a[c];
        int const gb = grid_b[c];

        // Unit node offsets along the in-plane grid axes (the level set is
        // nodal and collocated with B on this grid).
        amrex::IntVect da(0);
        amrex::IntVect db(0);
        da[ga] = 1;
        db[gb] = 1;
        auto const da0 = da[0];
        auto const db0 = db[0];
#if defined(WARPX_DIM_3D)
        auto const da1 = da[1];
        auto const db1 = db[1];
        auto const da2 = da[2];
        auto const db2 = db[2];
#else
        auto const da1 = da[1];
        auto const db1 = db[1];
#endif
        amrex::Real const ha = dx_arr[ga];
        amrex::Real const hb = dx_arr[gb];

        for (amrex::MFIter mfi(*Bfield[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box const tb = mfi.tilebox(Bfield[c]->ixType().toIntVect());

            auto const& Bc = Bfield[c]->array(mfi);
            auto const& Ea = Efield[ca]->const_array(mfi);
            auto const& Eb = Efield[cb]->const_array(mfi);
            auto const& phi = distance_to_eb.const_array(mfi);

            amrex::ParallelFor(tb,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // Neighbor indices along the in-plane axes a and b.
#if defined(WARPX_DIM_3D)
                int const ima = i - da0, jma = j - da1, kma = k - da2;
                int const ipa = i + da0, jpa = j + da1, kpa = k + da2;
                int const imb = i - db0, jmb = j - db1, kmb = k - db2;
                int const ipb = i + db0, jpb = j + db1, kpb = k + db2;
#else
                int const ima = i - da0, jma = j - da1, kma = k;
                int const ipa = i + da0, jpa = j + da1, kpa = k;
                int const imb = i - db0, jmb = j - db1, kmb = k;
                int const ipb = i + db0, jpb = j + db1, kpb = k;
#endif

                // Dual-face corners (CCW: SW, SE, NE, NW) at +/- h/2 in (a, b),
                // each the bilinear cell-center average of the four nodal
                // level-set values bracketing it in the (a, b) plane. The four
                // bracketing nodes per corner are {node, +/-a, +/-b, +/-a+/-b}.
#if defined(WARPX_DIM_3D)
                amrex::Real const p_c  = phi(i,   j,   k  );
                amrex::Real const p_ma = phi(ima, jma, kma);
                amrex::Real const p_pa = phi(ipa, jpa, kpa);
                amrex::Real const p_mb = phi(imb, jmb, kmb);
                amrex::Real const p_pb = phi(ipb, jpb, kpb);
                amrex::Real const p_mamb = phi(ima-db0, jma-db1, kma-db2);
                amrex::Real const p_pamb = phi(ipa-db0, jpa-db1, kpa-db2);
                amrex::Real const p_pamb2 = phi(ipa+db0, jpa+db1, kpa+db2);
                amrex::Real const p_mapb = phi(ima+db0, jma+db1, kma+db2);
#else
                amrex::Real const p_c  = phi(i,   j,   k);
                amrex::Real const p_ma = phi(ima, jma, k);
                amrex::Real const p_pa = phi(ipa, jpa, k);
                amrex::Real const p_mb = phi(imb, jmb, k);
                amrex::Real const p_pb = phi(ipb, jpb, k);
                amrex::Real const p_mamb = phi(ima-db0, jma-db1, k);
                amrex::Real const p_pamb = phi(ipa-db0, jpa-db1, k);
                amrex::Real const p_pamb2 = phi(ipa+db0, jpa+db1, k);
                amrex::Real const p_mapb = phi(ima+db0, jma+db1, k);
#endif
                amrex::Real const cphi[4] = {
                    0.25_rt*(p_c + p_ma + p_mb + p_mamb),   // SW (-a, -b)
                    0.25_rt*(p_c + p_pa + p_mb + p_pamb),   // SE (+a, -b)
                    0.25_rt*(p_c + p_pa + p_pb + p_pamb2),  // NE (+a, +b)
                    0.25_rt*(p_c + p_ma + p_pb + p_mapb)};  // NW (-a, +b)

                // E along the cut edges: E_a at b = {-1, 0, +1}, E_b at
                // a = {-1, 0, +1} (component 0 of each collocated field).
                amrex::Real const Ea_s[3] = {
                    Ea(imb, jmb, kmb), Ea(i, j, k), Ea(ipb, jpb, kpb)};
                amrex::Real const Eb_s[3] = {
                    Eb(ima, jma, kma), Eb(i, j, k), Eb(ipa, jpa, kpa)};

                auto const cut = ::cut_cell(cphi, Ea_s, Eb_s, ha, hb, dt);
                if (cut.klass == ::C_INTERIOR) {
                    return;  // dB_corr is machine-zero away from the wall
                }
                if (cut.klass == ::C_DEEP) {
                    Bc(i, j, k) = 0._rt;  // no field deep inside the conductor
                    return;
                }

                // Band node: the core nodal push already added dB_nodal_c, so
                // add dB_corr = dB_fv - dB_nodal_c to leave the net at dB_fv.
                // dB_nodal_c is the EvolveBCartesian curl of this component,
                //   dt*(UpwardD_b(E_a) - UpwardD_a(E_b)),
                // with Ea_s = {Ea(-b), Ea(0), Ea(+b)} along the a edges and
                // Eb_s = {Eb(-a), Eb(0), Eb(+a)} along the b edges. The uncut
                // dB_fv (l = h, S_eff = h^2) equals this exactly (verified).
                amrex::Real const dB_nodal =
                    dt * (0.5_rt/hb * (Ea_s[2] - Ea_s[0])
                        - 0.5_rt/ha * (Eb_s[2] - Eb_s[0]));
                Bc(i, j, k) += cut.dB_fv - dB_nodal;
            });
        }
    }
#else
    amrex::ignore_unused(Bfield, Efield, distance_to_eb, geom, dt);
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
