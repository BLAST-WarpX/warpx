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

// Nodal-only by design: assumes a fully nodal scalar (asserted below) with
// zero staggering. A staggered scalar would need per-component half-cell
// offsets, as FoldEBDepositToField derives them.
void warpx::hybrid::FoldEBDepositToNodalScalar (
    amrex::MultiFab& field,
    amrex::MultiFab const& distance_to_eb,
    amrex::Geometry const& geom)
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
    amrex::Real constexpr fold_sign = -1.0_rt;  // PEC image parity (the EB is always a PEC)

    // covered-side ghosts must hold the guard-summed deposit
    field.FillBoundary(geom.periodicity());

    // The mirror gather reaches up to 2*fold_band past the surface plus one
    // stencil cell, which can exceed the field's own ghost width at box
    // seams near the wall (the stencil clamp would then silently read the
    // wrong cells). Gather from ghost-extended scratch copies instead; the
    // level-set scratch is initialized to a large positive (fluid) value so
    // unfilled physical-boundary ghosts never enter the covered set.
    int const ng_gather = 4;
    amrex::MultiFab f_scratch(field.boxArray(), field.DistributionMap(), field.nComp(),
                          amrex::IntVect(ng_gather));
    f_scratch.setVal(0.0_rt);
    amrex::MultiFab::Copy(f_scratch, field, 0, 0, field.nComp(), 0);
    f_scratch.FillBoundary(geom.periodicity());
    amrex::MultiFab phi_scratch(distance_to_eb.boxArray(), distance_to_eb.DistributionMap(),
                            1, amrex::IntVect(ng_gather));
    phi_scratch.setVal(std::numeric_limits<amrex::Real>::max());
    amrex::MultiFab::Copy(phi_scratch, distance_to_eb, 0, 0, 1, 0);
    phi_scratch.FillBoundary(geom.periodicity());

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const stag0{};

    for (amrex::MFIter mfi(field, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const tb = mfi.tilebox();
        int const ncomp = field.nComp();

        auto const& f = field.array(mfi);
        auto const& f_r = f_scratch.const_array(mfi);
        auto const& phi = distance_to_eb.const_array(mfi);
        auto const& phi_g = phi_scratch.const_array(mfi);

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
            if (!std::isfinite(nv2) || !(nv2 > 0._rt)) { return; }

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
    amrex::ignore_unused(field, distance_to_eb, geom);
#endif
}

void warpx::hybrid::FoldEBDepositToField (
    ablastr::fields::VectorField const& field,
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update,
    amrex::MultiFab const& distance_to_eb,
    amrex::Geometry const& geom,
    EBFillStatus* status_cache)
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
    amrex::Real constexpr fold_sign = -1.0_rt;  // PEC image parity (the EB is always a PEC)

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
    amrex::ignore_unused(field, eb_update, distance_to_eb, geom, status_cache);
#endif
}
