/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "EBJBoundary.H"

#include "EBJBoundary_K.H"
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
#include <cstdlib>
#include <limits>

using namespace amrex;


using namespace warpx::hybrid::detail;

void warpx::hybrid::ApplyPECBoundaryToField (
    ablastr::fields::VectorField const& field,
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update,
    amrex::MultiFab const& distance_to_eb,
    amrex::Geometry const& geom,
    bool normal_odd,
    bool fill_covered_centers,
    EBFillStatus* status_cache,
    amrex::Real band_cells)
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

    // Single-pass deterministic mirror fill (the only strategy): well-posed
    // targets in one pass, ill-posed ones via the bounded cascade below.
    {
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

                    MirrorGeom const g = ::mirror_geom(i, j, k, stag_own, phi,
                        plo, dxi, dx_arr, d_band, d_img_min, h_max);

                    auto const in_sol_x =
                        [&] (int ig, int jg, int kg) { return stat_x(ig, jg, kg) == S_SOLUTION; };
                    auto const in_sol_y =
                        [&] (int ig, int jg, int kg) { return stat_y(ig, jg, kg) == S_SOLUTION; };
                    auto const in_sol_z =
                        [&] (int ig, int jg, int kg) { return stat_z(ig, jg, kg) == S_SOLUTION; };
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const xim = g.xim;
                    amrex::Real const d_im = g.d_im;

                    amrex::Real const Jx_im = amrex::get<0>(gather_staggered_pred(Jx_l, in_sol_x, xim, stag_x, plo, dxi, n));
                    amrex::Real const Jy_im = amrex::get<0>(gather_staggered_pred(Jy_l, in_sol_y, xim, stag_y, plo, dxi, n));
                    amrex::Real const Jz_im = amrex::get<0>(gather_staggered_pred(Jz_l, in_sol_z, xim, stag_z, plo, dxi, n));

                    // edge fields (E, J): normal even / tangential odd;
                    // magnetic field: normal odd / tangential even.
                    amrex::Real const w_n = normal_odd ? g.s/d_im : 1._rt;
                    amrex::Real const w_t = normal_odd ? 1._rt : g.s/d_im;
                    Jc(i, j, k, n) = ::mirror_combine(
                        c, Jx_im, Jy_im, Jz_im, g.nv, w_n, w_t);
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

            // Sweep cap of the ill-posed resolution cascade: each sweep
            // locks at least one point or stalls, so this is a backstop,
            // not a knob.
            constexpr int max_cascade_sweeps = 10;
            int n_left = st.n_pending;
            for (int sweep = 0; sweep < max_cascade_sweeps && n_left > 0; ++sweep) {
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
                                Jc(i, j, k, n) = ::mirror_combine(c, Jx_im, Jy_im, Jz_im,
                                    g.nv, w_n, w_t);
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

#else
    amrex::ignore_unused(field, eb_update, distance_to_eb, geom, normal_odd,
                         fill_covered_centers, status_cache, band_cells);
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
            auto const n3 =
                DistanceToEB::interp_normal(xe[0], yq, zq, plo, dxi, phi);
#if defined(WARPX_DIM_3D)
            amrex::RealVect const nv{
                amrex::Real(n3[0]), amrex::Real(n3[1]), amrex::Real(n3[2])};
#else
            amrex::RealVect const nv{
                amrex::Real(n3[0]), amrex::Real(n3[2])};
#endif
            amrex::Real const nv2 = DistanceToEB::dot_product(nv, nv);
            if (!std::isfinite(nv2) || !(nv2 > 0._rt)) {
                // degenerate level-set gradient (interp_normal normalizes
                // internally, so a vanishing gradient arrives non-finite):
                // treat as deep interior
                f(i, j, k, n) = 0._rt;
                return;
            }

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
