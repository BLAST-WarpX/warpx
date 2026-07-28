/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "HybridPICModel.H"

#include "EBJBoundary.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "WarpX.H"

#include <ablastr/utils/Communication.H>

#include <AMReX_Array4.H>
#include <AMReX_GpuControl.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#include <algorithm>
#include <climits>
#include <limits>

using namespace amrex;
using warpx::fields::FieldType;

void HybridPICModel::MarderCleanDivergence (
    ablastr::fields::VectorField const& field,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update,
    warpx::hybrid::EBFillStatus* status_cache,
    const bool normal_odd,
    const bool fill_covered_centers,
    const amrex::Real alpha,
    const int max_iters,
    const amrex::Real clean_band_cells,
    const amrex::Real fill_band_cells,
    const int lev,
    const bool cut_metric) const
{
// 3D/XZ Cartesian only: the gradient below is the discrete adjoint of the
// Cartesian ComputeDivE, which is what makes the update strictly dissipative
// of the divergence norm. In RZ ComputeDivE is the cylindrical (1/r)
// divergence and the plain Cartesian gradient is NOT its adjoint, so the
// dissipativity and curl-preservation guarantees would not hold there.
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
    if (alpha <= 0.0_rt || max_iters <= 0 || !EB::enabled()) { return; }

    auto& warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom(lev);
    const auto dx = geom.CellSizeArray();
    const bool nodal_grid =
        (WarpX::grid_type == ablastr::utils::enums::GridType::Collocated);

    // CFL-stable explicit grad(div) step: the discrete Laplacian peaks at ~1/h^2
    // (collocated wide) or ~4/h^2 (Yee compact), so alpha (a fraction of the CFL
    // cap) maps to the same damping on both grids via this scaling -- mirrors the
    // E-field Marder.
    Real h_min = dx[0], h_max = dx[0];
    for (int d = 1; d < AMREX_SPACEDIM; ++d) {
        h_min = std::min(h_min, dx[d]);
        h_max = std::max(h_max, dx[d]);
    }
    const Real alpha_scaled = alpha * h_min * h_min * (nodal_grid ? 4.0_rt : 1.0_rt);

    amrex::MultiFab const* phi_mf = warpx.m_fields.get(FieldType::distance_to_eb, lev);

    // Matched cut-metric mode (staggered conformal/ECT path, 3D only): the
    // divergence is the signed sum of OPEN-FACE fluxes, D = sum_d [frac_+ F_+
    // - frac_- F_-]/h_d with frac = face_areas / full face area -- the
    // invariant the ECT Faraday actually conserves -- and the correction is
    // its exact negative adjoint, F_f += alpha * frac_f * grad_f(div). The
    // frac weight appears once in each operator, so G = -D^T and the sweep is
    // strictly dissipative of the FLUX-metric divergence norm; since
    // frac <= 1 the explicit CFL cap is the same as the raw operator's (no
    // implicit solve needed for the diffusive damper -- the CG/agglomeration
    // requirement applies to the exact projection, not to Marder damping).
    // Covered faces (frac = 0) are never read or written; live faces are
    // gated on frac > 0 rather than the staircase update mask, which zeroes
    // CUT faces and would otherwise skip exactly the flux DOFs the clean is
    // for. The per-sweep level-set mirror re-fill is skipped too: the ECT arm
    // imposes no mirror on B, and re-imposing one here would overwrite the
    // ECT-advanced cut faces with raw-metric mirror values (the two-metric
    // fight measured on the liftoff ect_clean arm).
#if defined(WARPX_DIM_3D)
    const bool use_cut = cut_metric && !nodal_grid;
#else
    const bool use_cut = false;
    amrex::ignore_unused(cut_metric);
#endif
    ablastr::fields::VectorField frac_mf;
    if (use_cut) {
        frac_mf = warpx.m_fields.get_alldirs(FieldType::face_areas, lev);
    }
#if defined(WARPX_DIM_3D)
    const amrex::GpuArray<Real, 3> inv_full_area = use_cut
        ? amrex::GpuArray<Real, 3>{1.0_rt/(dx[1]*dx[2]),
                                   1.0_rt/(dx[0]*dx[2]),
                                   1.0_rt/(dx[0]*dx[1])}
        : amrex::GpuArray<Real, 3>{0.0_rt, 0.0_rt, 0.0_rt};
#else
    const amrex::GpuArray<Real, 3> inv_full_area{0.0_rt, 0.0_rt, 0.0_rt};
    amrex::ignore_unused(inv_full_area);
#endif

    // band_cells <= 0 selects the UNBOUNDED mode: the correction applies on
    // every uncovered node. The pure-gradient update is strictly dissipative
    // of the global divergence norm (d/dt int(div^2) = -2 alpha int|grad div|^2)
    // and is a round-off no-op wherever div is already ~0 (the bulk), so the
    // only cost is the sweep itself. A HARD outer cutoff instead transports
    // divergence to the band edge and piles it up just outside, where nothing
    // damps it (measured on the annulus: d=3 divergence 14x the baseline).
    // In the banded mode, cap the outer cutoff half a cell below the measured
    // level-set roof: FillSignedDistance clamps every deep-fluid node to a
    // constant roof of a few cells, so a phi-only band test would otherwise
    // classify the ENTIRE deep interior as in-band once the band reaches the
    // roof.
    const bool unbounded = (clean_band_cells <= 0.0_rt);
    const Real phi_roof = phi_mf->max(0, 0, false);
    const Real d_clean = unbounded
        ? std::numeric_limits<Real>::max()
        : std::min(clean_band_cells * h_max, phi_roof - 0.5_rt * h_max);
    // EB-aware band cutoffs, exposed as knobs. The defaults (inner_div=1,
    // inner_corr=2) trust the divergence only where its own +/-1 stencil is in
    // the fluid and apply the correction only where the full +/-2 grad(div)
    // stencil is fluid, leaving the immediate wall layer to the level-set
    // mirror fill -- that preserves the near-wall order on a smooth wall but
    // cannot reach a divergence mode living in the first two fluid layers.
    // Setting both to 0 extends the clean over every uncovered band node (the
    // wall-layer mode): the stencils then read mirror-filled covered nodes,
    // trading an O(h) near-wall error for dissipation exactly where a sharp
    // re-entrant corner (e.g. a wall radius step) pumps an exponentially
    // growing div(B). phi is the true signed distance, so these cutoffs hold
    // for any wall orientation. (No clean if d_clean <= d_stencil.)
    const Real d_div_keep = m_divb_clean_inner_div_cells * h_max;
    const Real d_stencil = m_divb_clean_inner_corr_cells * h_max;
    amrex::GpuArray<Real, AMREX_SPACEDIM> inv_dx{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { inv_dx[d] = 1.0_rt/dx[d]; }

    // Exclude the first two node layers at every NON-PERIODIC physical domain
    // boundary from both the kept divergence and the correction. There the
    // stencil goes asymmetric (the div ghost beyond the boundary is 0, and the
    // boundary B is owned by the domain BC), and at a junction of the EB wall
    // with the domain boundary that asymmetry PUMPS a spurious divergence
    // instead of damping it (measured on the annulus throat at z=0: baseline
    // machine-zero there, damped run grew 0.12 -> 24 T/m over 5000 steps,
    // e-fold ~940). Nothing needs cleaning there anyway.
    const amrex::Box nodal_domain = amrex::convert(geom.Domain(), IntVect::TheNodeVector());
    amrex::GpuArray<int, 3> dom_lo{INT_MIN, INT_MIN, INT_MIN};
    amrex::GpuArray<int, 3> dom_hi{INT_MAX, INT_MAX, INT_MAX};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (!geom.isPeriodic(d)) {
            dom_lo[d] = nodal_domain.smallEnd(d) + 2;
            dom_hi[d] = nodal_domain.bigEnd(d) - 2;
        }
    }

    auto* fdtd = warpx.get_pointer_fdtd_solver_fp(lev);

    // Nodal divergence scratch with one ghost on EVERY grid type: the centered
    // (collocated) grad reads d(i-1..i+1) and the Yee forward difference reads
    // d(i+1), and a valid face at the high box edge puts that read one node
    // past the valid nodal range on both. Zero-initialize so ghost nodes at
    // non-periodic physical domain boundaries (which FillBoundary does not
    // fill) read as div=0 rather than uninitialized memory.
    const IntVect div_ng = IntVect(1);
    auto const& nodal_ba = amrex::convert(field[0]->boxArray(), IntVect::TheNodeVector());
    MultiFab div_f(nodal_ba, field[0]->DistributionMap(), 1, div_ng);
    div_f.setVal(0.0_rt);

    for (int it = 0; it < max_iters; ++it) {
        for (int c = 0; c < 3; ++c) {
            ablastr::utils::communication::FillBoundary(
                *field[c], IntVect(1), WarpX::do_single_precision_comms,
                geom.periodicity());
        }
        if (use_cut) {
#if defined(WARPX_DIM_3D)
            // Flux divergence with the same DownwardD index arithmetic as
            // ComputeDivE, weighted by the open-face fraction.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(div_f, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                Array4<Real> const& d = div_f.array(mfi);
                Array4<Real const> const& Fx = field[0]->const_array(mfi);
                Array4<Real const> const& Fy = field[1]->const_array(mfi);
                Array4<Real const> const& Fz = field[2]->const_array(mfi);
                Array4<Real const> const& sx = frac_mf[0]->const_array(mfi);
                Array4<Real const> const& sy = frac_mf[1]->const_array(mfi);
                Array4<Real const> const& sz = frac_mf[2]->const_array(mfi);
                const auto ifa = inv_full_area;
                const auto idx = inv_dx;
                amrex::ParallelFor(mfi.tilebox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k){
                        d(i, j, k) =
                            (sx(i,j,k)*ifa[0]*Fx(i,j,k) - sx(i-1,j,k)*ifa[0]*Fx(i-1,j,k))*idx[0]
                          + (sy(i,j,k)*ifa[1]*Fy(i,j,k) - sy(i,j-1,k)*ifa[1]*Fy(i,j-1,k))*idx[1]
                          + (sz(i,j,k)*ifa[2]*Fz(i,j,k) - sz(i,j,k-1)*ifa[2]*Fz(i,j,k-1))*idx[2];
                    });
            }
#endif
        } else {
            fdtd->ComputeDivE(field, div_f);
        }

        // Keep the divergence only in the fluid-clean near-wall band
        // [d_div_keep, d_clean]: drop the bulk (already solenoidal) and, unless
        // the wall-layer mode is selected, the wall layer / covered nodes
        // (phi < d_div_keep), where the centered divergence would read the
        // mirror-filled covered nodes across the wall.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(div_f, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Array4<Real> const& d = div_f.array(mfi);
            Array4<Real const> const& phi = phi_mf->const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    if (phi(i, j, k) < d_div_keep || phi(i, j, k) > d_clean
                        || i < dom_lo[0] || i > dom_hi[0]
                        || j < dom_lo[1] || j > dom_hi[1]
                        || k < dom_lo[2] || k > dom_hi[2]) { d(i, j, k) = 0.0_rt; }
                });
        }
        ablastr::utils::communication::FillBoundary(
            div_f, div_ng, WarpX::do_single_precision_comms, geom.periodicity());

        // field += alpha_scaled * grad(div), on solution nodes inside the band.
        // grad of a scalar is curl-free, so curl(field) (= J for B) is unchanged
        // away from the band cutoffs (the band masking perturbs the curl at the
        // band edges; in the wall-layer mode the mirror re-fill below re-imposes
        // the wall parity on top each sweep).
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*field[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Array4<Real> const& Fx = field[0]->array(mfi);
            Array4<Real> const& Fy = field[1]->array(mfi);
            Array4<Real> const& Fz = field[2]->array(mfi);
            Array4<Real const> const& d = div_f.const_array(mfi);
            Array4<Real const> const& phi = phi_mf->const_array(mfi);
            Array4<int const> const& ux = eb_update[0]->const_array(mfi);
            Array4<int const> const& uy = eb_update[1]->const_array(mfi);
            Array4<int const> const& uz = eb_update[2]->const_array(mfi);

            Box const& tx = mfi.tilebox(field[0]->ixType().toIntVect());
            Box const& ty = mfi.tilebox(field[1]->ixType().toIntVect());
            Box const& tz = mfi.tilebox(field[2]->ixType().toIntVect());

#if defined(WARPX_DIM_3D)
            if (use_cut) {
                // Adjoint (frac-weighted) gradient: F_f += alpha * frac_f *
                // grad_f(div). Gated on frac > 0 (a live flux DOF), NOT on
                // the staircase update mask, which zeroes cut faces.
                Array4<Real const> const& sx = frac_mf[0]->const_array(mfi);
                Array4<Real const> const& sy = frac_mf[1]->const_array(mfi);
                Array4<Real const> const& sz = frac_mf[2]->const_array(mfi);
                const auto ifa = inv_full_area;
                amrex::ParallelFor(tx, ty, tz,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k){
                        const Real fr = sx(i,j,k)*ifa[0];
                        if (fr <= 0.0_rt || phi(i,j,k)<d_stencil || phi(i,j,k)>d_clean
                            || i < dom_lo[0] || i > dom_hi[0]
                            || j < dom_lo[1] || j > dom_hi[1]
                            || k < dom_lo[2] || k > dom_hi[2]) { return; }
                        Fx(i,j,k) += alpha_scaled*fr*(d(i+1,j,k)-d(i,j,k))*inv_dx[0];
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k){
                        const Real fr = sy(i,j,k)*ifa[1];
                        if (fr <= 0.0_rt || phi(i,j,k)<d_stencil || phi(i,j,k)>d_clean
                            || i < dom_lo[0] || i > dom_hi[0]
                            || j < dom_lo[1] || j > dom_hi[1]
                            || k < dom_lo[2] || k > dom_hi[2]) { return; }
                        Fy(i,j,k) += alpha_scaled*fr*(d(i,j+1,k)-d(i,j,k))*inv_dx[1];
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k){
                        const Real fr = sz(i,j,k)*ifa[2];
                        if (fr <= 0.0_rt || phi(i,j,k)<d_stencil || phi(i,j,k)>d_clean
                            || i < dom_lo[0] || i > dom_hi[0]
                            || j < dom_lo[1] || j > dom_hi[1]
                            || k < dom_lo[2] || k > dom_hi[2]) { return; }
                        Fz(i,j,k) += alpha_scaled*fr*(d(i,j,k+1)-d(i,j,k))*inv_dx[2];
                    });
                continue;
            }
            amrex::ParallelFor(tx, ty, tz,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    if (ux(i,j,k)==0 || phi(i,j,k)<d_stencil || phi(i,j,k)>d_clean
                        || i < dom_lo[0] || i > dom_hi[0]
                        || j < dom_lo[1] || j > dom_hi[1]
                        || k < dom_lo[2] || k > dom_hi[2]) { return; }
                    Fx(i,j,k) += nodal_grid
                        ? alpha_scaled*(d(i+1,j,k)-d(i-1,j,k))*(0.5_rt*inv_dx[0])
                        : alpha_scaled*(d(i+1,j,k)-d(i,j,k))*inv_dx[0];
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    if (uy(i,j,k)==0 || phi(i,j,k)<d_stencil || phi(i,j,k)>d_clean
                        || i < dom_lo[0] || i > dom_hi[0]
                        || j < dom_lo[1] || j > dom_hi[1]
                        || k < dom_lo[2] || k > dom_hi[2]) { return; }
                    Fy(i,j,k) += nodal_grid
                        ? alpha_scaled*(d(i,j+1,k)-d(i,j-1,k))*(0.5_rt*inv_dx[1])
                        : alpha_scaled*(d(i,j+1,k)-d(i,j,k))*inv_dx[1];
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    if (uz(i,j,k)==0 || phi(i,j,k)<d_stencil || phi(i,j,k)>d_clean
                        || i < dom_lo[0] || i > dom_hi[0]
                        || j < dom_lo[1] || j > dom_hi[1]
                        || k < dom_lo[2] || k > dom_hi[2]) { return; }
                    Fz(i,j,k) += nodal_grid
                        ? alpha_scaled*(d(i,j,k+1)-d(i,j,k-1))*(0.5_rt*inv_dx[2])
                        : alpha_scaled*(d(i,j,k+1)-d(i,j,k))*inv_dx[2];
                });
#elif defined(WARPX_DIM_XZ)
            // In-plane components only (out-of-plane has no divergence contribution).
            amrex::ignore_unused(Fy, uy, ty);
            amrex::ParallelFor(tx, tz,
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    if (ux(i,j,0)==0 || phi(i,j,0)<d_stencil || phi(i,j,0)>d_clean
                        || i < dom_lo[0] || i > dom_hi[0]
                        || j < dom_lo[1] || j > dom_hi[1]) { return; }
                    Fx(i,j,0) += nodal_grid
                        ? alpha_scaled*(d(i+1,j,0)-d(i-1,j,0))*(0.5_rt*inv_dx[0])
                        : alpha_scaled*(d(i+1,j,0)-d(i,j,0))*inv_dx[0];
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    if (uz(i,j,0)==0 || phi(i,j,0)<d_stencil || phi(i,j,0)>d_clean
                        || i < dom_lo[0] || i > dom_hi[0]
                        || j < dom_lo[1] || j > dom_hi[1]) { return; }
                    Fz(i,j,0) += nodal_grid
                        ? alpha_scaled*(d(i,j+1,0)-d(i,j-1,0))*(0.5_rt*inv_dx[1])
                        : alpha_scaled*(d(i,j+1,0)-d(i,j,0))*inv_dx[1];
                });
#endif
        }

        // Re-impose the EB PEC condition with the field's own parity, using its
        // own fill band, so the covered band stays consistent with the main fill.
        // Skipped in the cut-metric mode: covered faces (frac = 0) were never
        // touched, and the ECT arm imposes no level-set mirror on B -- re-filling
        // here would overwrite the ECT-advanced cut faces with raw-metric mirror
        // values every sweep.
        if (!use_cut) {
            warpx::hybrid::ApplyPECBoundaryToField(
                field, eb_update, *phi_mf, geom,
                normal_odd, fill_covered_centers, status_cache, fill_band_cells);
        }
    }

    for (int c = 0; c < 3; ++c) {
        ablastr::utils::communication::FillBoundary(
            *field[c], field[c]->nGrowVect(), WarpX::do_single_precision_comms,
            geom.periodicity());
    }
#else
    amrex::ignore_unused(field, eb_update, status_cache, normal_odd,
        fill_covered_centers, alpha, max_iters, clean_band_cells, fill_band_cells,
        lev, cut_metric);
#endif
}

void HybridPICModel::MarderCleanFieldsPerStep () const
{
    if (m_divb_clean_alpha <= 0.0_rt && m_divj_clean_alpha <= 0.0_rt) { return; }

    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        // B (magnetic parity). On the staggered conformal (ECT) path the clean
        // uses the matched cut-metric operators (see MarderCleanDivergence).
        if (m_divb_clean_alpha > 0.0_rt) {
            if (static_cast<int>(m_eb_bc_status_B.size()) <= lev) { m_eb_bc_status_B.resize(lev+1); }
            const bool cut_metric = m_divb_clean_cut_metric && m_use_conformal_eb
                && WarpX::grid_type != ablastr::utils::enums::GridType::Collocated;
            MarderCleanDivergence(
                warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev),
                warpx.GetEBUpdateBFlag()[lev], &m_eb_bc_status_B[lev],
                /*normal_odd=*/true, /*fill_covered_centers=*/false,
                m_divb_clean_alpha, m_divb_clean_iters,
                m_divb_clean_band_cells, m_eb_b_fill_band_cells, lev, cut_metric);
        }
        // Total Ampere current (electric parity), never an ion species.
        if (m_divj_clean_alpha > 0.0_rt) {
            if (static_cast<int>(m_eb_bc_status_Jplasma.size()) <= lev) { m_eb_bc_status_Jplasma.resize(lev+1); }
            MarderCleanDivergence(
                warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev),
                warpx.GetEBUpdateEFlag()[lev], &m_eb_bc_status_Jplasma[lev],
                /*normal_odd=*/false, /*fill_covered_centers=*/true,
                m_divj_clean_alpha, m_divb_clean_iters,
                m_divb_clean_band_cells, m_eb_fill_band_cells, lev);
        }
    }
}

void HybridPICModel::MarderCleanESeam (
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    amrex::IntVect ng, std::optional<bool> nodal_sync) const
{
// 3D/XZ Cartesian, staggered grids only (the caller gates on grid type): the
// nodal divergence below is the discrete adjoint of the edge gradient, which
// makes the update strictly dissipative of the divergence norm; the masked
// (density-banded, EB-masked, domain-excluded) variant stays dissipative since
// every mask is a nonnegative diagonal weight.
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
    if (m_dive_seam_alpha <= 0.0_rt || m_dive_seam_iters <= 0) { return; }

    auto& warpx = WarpX::GetInstance();
    const Real rho_floor = m_n_floor * PhysConst::q_e;
    const Real rho_hi = m_dive_seam_band * rho_floor;
    const bool eb = EB::enabled();

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            Efield[lev][0]->nGrowVect().min() >= 1,
            "MarderCleanESeam: the nodal divergence at box seams needs at "
            "least one valid E ghost cell.");

        const auto& geom = warpx.Geom(lev);
        const auto dx = geom.CellSizeArray();
        Real h_min = dx[0];
        for (int d = 1; d < AMREX_SPACEDIM; ++d) { h_min = std::min(h_min, dx[d]); }
        // Explicit grad(div) sweep: the compact Yee operator peaks at
        // 4*sum_d(1/dx_d^2) = 12/h^2 (cubic 3D), so stability requires
        // alpha <= 1/6 (asserted at parse time); alpha is the fraction of that
        // cap actually applied.
        const Real alpha_scaled = m_dive_seam_alpha * h_min * h_min;
        GpuArray<Real, 3> inv_dx{0.0_rt, 0.0_rt, 0.0_rt};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { inv_dx[d] = 1.0_rt/dx[d]; }

        // Exclude two layers at every non-periodic physical domain face, per
        // component staggering: the asymmetric stencil at a domain/EB junction
        // was measured to PUMP divergence in the B cleaner (e-fold ~940), and
        // this cleaner runs every substage. Nothing needs cleaning there.
        GpuArray<int, 3> ex_lo[3], ex_hi[3];
        for (int c = 0; c < 3; ++c) {
            const amrex::Box comp_domain =
                amrex::convert(geom.Domain(), Efield[lev][c]->ixType());
            for (int d = 0; d < 3; ++d) { ex_lo[c][d] = INT_MIN; ex_hi[c][d] = INT_MAX; }
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (!geom.isPeriodic(d)) {
                    ex_lo[c][d] = comp_domain.smallEnd(d) + 2;
                    ex_hi[c][d] = comp_domain.bigEnd(d) - 2;
                }
            }
        }

        // Nodal divergence scratch, zero-initialized: the valid region is
        // recomputed each sweep and the (never-written) ghosts read as div=0.
        auto const& nodal_ba = amrex::convert(
            rhofield[lev]->boxArray(), IntVect::TheNodeVector());
        MultiFab div_mf(nodal_ba, rhofield[lev]->DistributionMap(), 1, IntVect(1));
        div_mf.setVal(0.0_rt);

        auto const& Ex_stag = Ex_IndexType;
        auto const& Ey_stag = Ey_IndexType;
        auto const& Ez_stag = Ez_IndexType;

        for (int it = 0; it < m_dive_seam_iters; ++it) {

            // Fresh ghosts: on the staggered path the Ohm solve fills valid
            // tileboxes only, and the update below invalidates them each sweep.
            warpx.FillBoundaryE(lev, ng, nodal_sync);

            // 1) nodal divergence of the edge E on the valid nodal boxes. Each
            // component contributes the difference along its cell-centered
            // direction; fully nodal components (the out-of-plane E in 2D)
            // have no divergence role.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(div_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box tbx = mfi.tilebox();
                Array4<Real> const& div_arr = div_mf.array(mfi);
                Array4<Real const> const& Ex = Efield[lev][0]->const_array(mfi);
                Array4<Real const> const& Ey = Efield[lev][1]->const_array(mfi);
                Array4<Real const> const& Ez = Efield[lev][2]->const_array(mfi);
                amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    Real dsum = 0.0_rt;
                    if (Ex_stag[0] == 0) { dsum += (Ex(i,j,k) - Ex(i-1,j,k))*inv_dx[0]; }
                    else if (Ex_stag[1] == 0) { dsum += (Ex(i,j,k) - Ex(i,j-1,k))*inv_dx[1]; }
                    else if (Ex_stag[2] == 0) { dsum += (Ex(i,j,k) - Ex(i,j,k-1))*inv_dx[2]; }
                    if (Ey_stag[0] == 0) { dsum += (Ey(i,j,k) - Ey(i-1,j,k))*inv_dx[0]; }
                    else if (Ey_stag[1] == 0) { dsum += (Ey(i,j,k) - Ey(i,j-1,k))*inv_dx[1]; }
                    else if (Ey_stag[2] == 0) { dsum += (Ey(i,j,k) - Ey(i,j,k-1))*inv_dx[2]; }
                    if (Ez_stag[0] == 0) { dsum += (Ez(i,j,k) - Ez(i-1,j,k))*inv_dx[0]; }
                    else if (Ez_stag[1] == 0) { dsum += (Ez(i,j,k) - Ez(i,j-1,k))*inv_dx[1]; }
                    else if (Ez_stag[2] == 0) { dsum += (Ez(i,j,k) - Ez(i,j,k-1))*inv_dx[2]; }
                    div_arr(i,j,k) = dsum;
                });
            }

            // 2) masked gradient update: E += alpha * grad(div E) on edges that
            // (a) sit at or below the seam window (endpoint-averaged
            // rho <= dive_seam_band * rho_floor), (b) are LIVE per the EB
            // update masks (the wall/covered E is owned by the EB PEC fill the
            // Ohm solve just applied -- do not rewrite it), and (c) are at
            // least two layers from every non-periodic domain face. The bulk
            // plasma is untouched; curl(E) is preserved to round-off away from
            // the window edge.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(*Efield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                Array4<Real> const& Ex = Efield[lev][0]->array(mfi);
                Array4<Real> const& Ey = Efield[lev][1]->array(mfi);
                Array4<Real> const& Ez = Efield[lev][2]->array(mfi);
                Array4<Real const> const& div_arr = div_mf.const_array(mfi);
                Array4<Real const> const& rho = rhofield[lev]->const_array(mfi);
                Array4<int const> ux, uy, uz;
                if (eb) {
                    ux = eb_update_E[lev][0]->const_array(mfi);
                    uy = eb_update_E[lev][1]->const_array(mfi);
                    uz = eb_update_E[lev][2]->const_array(mfi);
                }
                const Box tex = mfi.tilebox(Efield[lev][0]->ixType().toIntVect());
                const Box tey = mfi.tilebox(Efield[lev][1]->ixType().toIntVect());
                const Box tez = mfi.tilebox(Efield[lev][2]->ixType().toIntVect());
                const auto exl0 = ex_lo[0]; const auto exh0 = ex_hi[0];
                const auto exl1 = ex_lo[1]; const auto exh1 = ex_hi[1];
                const auto exl2 = ex_lo[2]; const auto exh2 = ex_hi[2];
                amrex::ParallelFor(tex, tey, tez,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        if (ux && ux(i,j,k) == 0) { return; }
                        if (i < exl0[0] || i > exh0[0] || j < exl0[1] || j > exh0[1]
                            || k < exl0[2] || k > exh0[2]) { return; }
                        Real g = 0.0_rt; Real r = rho(i,j,k);
                        if (Ex_stag[0] == 0) { g = (div_arr(i+1,j,k)-div_arr(i,j,k))*inv_dx[0]; r = 0.5_rt*(rho(i,j,k)+rho(i+1,j,k)); }
                        else if (Ex_stag[1] == 0) { g = (div_arr(i,j+1,k)-div_arr(i,j,k))*inv_dx[1]; r = 0.5_rt*(rho(i,j,k)+rho(i,j+1,k)); }
                        else if (Ex_stag[2] == 0) { g = (div_arr(i,j,k+1)-div_arr(i,j,k))*inv_dx[2]; r = 0.5_rt*(rho(i,j,k)+rho(i,j,k+1)); }
                        if (r <= rho_hi) { Ex(i,j,k) += alpha_scaled * g; }
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        if (uy && uy(i,j,k) == 0) { return; }
                        if (i < exl1[0] || i > exh1[0] || j < exl1[1] || j > exh1[1]
                            || k < exl1[2] || k > exh1[2]) { return; }
                        Real g = 0.0_rt; Real r = rho(i,j,k);
                        if (Ey_stag[0] == 0) { g = (div_arr(i+1,j,k)-div_arr(i,j,k))*inv_dx[0]; r = 0.5_rt*(rho(i,j,k)+rho(i+1,j,k)); }
                        else if (Ey_stag[1] == 0) { g = (div_arr(i,j+1,k)-div_arr(i,j,k))*inv_dx[1]; r = 0.5_rt*(rho(i,j,k)+rho(i,j+1,k)); }
                        else if (Ey_stag[2] == 0) { g = (div_arr(i,j,k+1)-div_arr(i,j,k))*inv_dx[2]; r = 0.5_rt*(rho(i,j,k)+rho(i,j,k+1)); }
                        if (r <= rho_hi) { Ey(i,j,k) += alpha_scaled * g; }
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        if (uz && uz(i,j,k) == 0) { return; }
                        if (i < exl2[0] || i > exh2[0] || j < exl2[1] || j > exh2[1]
                            || k < exl2[2] || k > exh2[2]) { return; }
                        Real g = 0.0_rt; Real r = rho(i,j,k);
                        if (Ez_stag[0] == 0) { g = (div_arr(i+1,j,k)-div_arr(i,j,k))*inv_dx[0]; r = 0.5_rt*(rho(i,j,k)+rho(i+1,j,k)); }
                        else if (Ez_stag[1] == 0) { g = (div_arr(i,j+1,k)-div_arr(i,j,k))*inv_dx[1]; r = 0.5_rt*(rho(i,j,k)+rho(i,j+1,k)); }
                        else if (Ez_stag[2] == 0) { g = (div_arr(i,j,k+1)-div_arr(i,j,k))*inv_dx[2]; r = 0.5_rt*(rho(i,j,k)+rho(i,j,k+1)); }
                        if (r <= rho_hi) { Ez(i,j,k) += alpha_scaled * g; }
                    });
            }
        }

        // Leave the ghosts fresh for the Faraday push that follows.
        warpx.FillBoundaryE(lev, ng, nodal_sync);
    }
#else
    amrex::ignore_unused(Efield, rhofield, eb_update_E, ng, nodal_sync);
#endif
}
