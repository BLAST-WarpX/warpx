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
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/utils/Communication.H>

#include <AMReX_Array4.H>
#include <AMReX_GpuControl.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace amrex;
using warpx::fields::FieldType;

void HybridPICModel::MarderCorrectE (
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    MarderSite site)
{
    if (m_marder_alpha <= 0.0_rt) { return; }

    const bool site_matches =
        (site == MarderSite::Substep  && m_marder_level == MarderLevel::AllSubsteps) ||
        (site == MarderSite::HalfStep && m_marder_level == MarderLevel::HalfSteps) ||
        (site == MarderSite::FullStep && m_marder_level == MarderLevel::FullSteps);
    if (!site_matches) { return; }

    if (site == MarderSite::Substep) {
        // Reduced cadence: apply only on every m_marder_substep_interval-th
        // substep E evaluation. The counter is reset at the start of each step.
        ++m_marder_substep_counter;
        if (m_marder_substep_counter % m_marder_substep_interval != 0) { return; }
    }

    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        ApplyMarderCorrection(
            Efield[lev], Jfield[lev], Bfield[lev], *rhofield[lev],
            eb_update_E[lev], lev);
    }
}

std::pair<int, amrex::Real> HybridPICModel::ApplyMarderCorrection (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
    const int lev,
    std::optional<amrex::Real> alpha,
    std::optional<int> max_iterations,
    std::optional<amrex::Real> rtol,
    std::optional<amrex::Real> atol,
    std::optional<MarderTarget> target,
    const bool allow_target_cache) const
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    using namespace ablastr::coarsen::sample;

    const Real alpha_v = alpha.value_or(m_marder_alpha);
    if (alpha_v <= 0.0_rt) { return {0, 0.0_rt}; }
    const int max_iters_v = max_iterations.value_or(m_marder_max_iterations);
    const Real rtol_v = rtol.value_or(m_marder_rtol);
    const Real atol_v = atol.value_or(m_marder_atol);
    const MarderTarget target_v = target.value_or(m_marder_target);

    auto& warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom(lev);
    const auto dx = geom.CellSizeArray();
    const Real rho_floor = m_n_floor * PhysConst::q_e;

    // On a collocated (nodal) grid every field component lives at the mesh node,
    // so the discrete grad(Pe) building E_target and the grad(div_err) of the
    // fixed-point update are centered node-to-node differences rather than the
    // one-sided node-to-edge differences of the Yee staggering. ComputeDivE
    // already selects the centered nodal divergence for this grid type, and the
    // centered gradient is its adjoint, so grad(div) stays a convergent Laplacian.
    const bool nodal_grid =
        (WarpX::grid_type == ablastr::utils::enums::GridType::Collocated);

    // The explicit grad(div) update is CFL-limited by the largest eigenvalue of
    // the discrete operator. The Yee compact Laplacian peaks at 4*D/h^2 while the
    // collocated centered (wide) Laplacian peaks at D/h^2 -- four times smaller --
    // so the nodal grid admits (and needs, for the same damping per sweep) four
    // times the step. Scaling alpha_scaled by that ratio makes the user's
    // marder_alpha mean the same fraction of the CFL limit on both grids.
    Real h_min = dx[0];
    for (int d = 1; d < AMREX_SPACEDIM; ++d) { h_min = std::min(h_min, dx[d]); }
    const Real alpha_scaled =
        alpha_v * h_min * h_min * (nodal_grid ? 4.0_rt : 1.0_rt);

    amrex::GpuArray<Real, AMREX_SPACEDIM> inv_dx{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { inv_dx[d] = 1.0_rt/dx[d]; }

    amrex::GpuArray<int, 3> const& Ex_stag = Ex_IndexType;
    amrex::GpuArray<int, 3> const& Ey_stag = Ey_IndexType;
    amrex::GpuArray<int, 3> const& Ez_stag = Ez_IndexType;
    amrex::GpuArray<int, 3> const nodal{1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen{1, 1, 1};

    const bool eb_enabled = EB::enabled();
    amrex::MultiFab const* phi_mf = eb_enabled
        ? warpx.m_fields.get(FieldType::distance_to_eb, lev) : nullptr;

    auto* fdtd = warpx.get_pointer_fdtd_solver_fp(lev);

    // Nodal scratch for the divergence target and the masked divergence error.
    // The Yee edge update at a valid edge reads only valid nodal values, so it
    // needs no ghosts; the collocated update is a centered node-to-node gradient
    // that reaches one node past a box boundary, so div_err carries one ghost
    // layer (filled below) on a collocated grid.
    const IntVect div_err_ng = nodal_grid ? IntVect(1) : IntVect::TheZeroVector();
    auto const& nodal_ba = amrex::convert(rhofield.boxArray(), IntVect::TheNodeVector());
    MultiFab div_target(nodal_ba, rhofield.DistributionMap(), 1, IntVect::TheZeroVector());
    MultiFab div_err(nodal_ba, rhofield.DistributionMap(), 1, div_err_ng);

    // ---------------------------------------------------------------------
    // Assemble the divergence target
    // ---------------------------------------------------------------------
    // The grad_pe_only target depends only on Pe and the rho passed by the
    // call site, both constant between field-epoch bumps of the hybrid
    // advance, so its assembly is cached per level. The ohm target reads J
    // and B, which change every stage, and is never cached. The Python
    // unit-test binding passes allow_target_cache=false since it pokes Pe
    // directly through the field wrappers.
    const bool use_cache = allow_target_cache
        && (target_v == MarderTarget::GradPeOnly);
    bool cache_hit = false;
    if (use_cache) {
        if (static_cast<int>(m_marder_target_cache.size()) <= lev) {
            m_marder_target_cache.resize(lev+1);
            m_marder_target_cache_epoch.resize(lev+1, -1);
            m_marder_target_cache_rho.resize(lev+1, nullptr);
        }
        if (m_marder_target_cache_epoch[lev] == m_marder_field_epoch
            && m_marder_target_cache_rho[lev] == static_cast<const void*>(&rhofield)
            && m_marder_target_cache[lev].boxArray() == nodal_ba) {
            MultiFab::Copy(div_target, m_marder_target_cache[lev], 0, 0, 1, 0);
            cache_hit = true;
        }
    }

    if (cache_hit) {
        // div_target already holds the cached grad_pe_only assembly.
    } else if (target_v == MarderTarget::Zero) {
        div_target.setVal(0.0_rt);
    } else {
        const bool with_jxb = (target_v == MarderTarget::Ohm);

        // E_target = (enE - grad Pe) / max(rho, rho_floor) on the E
        // staggering, with enE = (J_plasma - J_i) x B for the "ohm" target
        // and zero for "grad_pe_only" (pressure smoothing only). This is the
        // bare Hall/pressure Ohm's-law field: deliberately not a call to
        // HybridPICSolveE, which zeroes the band when the holmstrom vacuum
        // treatment is on and adds the resistive/external terms.
        ablastr::fields::VectorField current_fp_plasma;
        if (with_jxb) {
            current_fp_plasma = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
        }
        auto const* electron_pressure_fp =
            warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);

        const bool include_external_fields = m_add_external_fields;
        ablastr::fields::VectorField Bfield_external;
        if (with_jxb && include_external_fields) {
            Bfield_external = warpx.m_fields.get_alldirs(FieldType::hybrid_B_fp_external, 0); // lev=0
        }

        amrex::GpuArray<int, 3> const& Jx_stag = Jx_IndexType;
        amrex::GpuArray<int, 3> const& Jy_stag = Jy_IndexType;
        amrex::GpuArray<int, 3> const& Jz_stag = Jz_IndexType;
        amrex::GpuArray<int, 3> const& Bx_stag = Bx_IndexType;
        amrex::GpuArray<int, 3> const& By_stag = By_IndexType;
        amrex::GpuArray<int, 3> const& Bz_stag = Bz_IndexType;

        // Nodal (J_plasma - J_i) x B; zero for the grad_pe_only target.
        MultiFab enE_nodal_mf(nodal_ba, rhofield.DistributionMap(), 3, IntVect::TheZeroVector());
        if (!with_jxb) {
            enE_nodal_mf.setVal(0.0_rt);
        } else {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(enE_nodal_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                Array4<Real> const& enE_nodal = enE_nodal_mf.array(mfi);
                Array4<Real const> const& Jx = current_fp_plasma[0]->const_array(mfi);
                Array4<Real const> const& Jy = current_fp_plasma[1]->const_array(mfi);
                Array4<Real const> const& Jz = current_fp_plasma[2]->const_array(mfi);
                Array4<Real const> const& Jix = Jfield[0]->const_array(mfi);
                Array4<Real const> const& Jiy = Jfield[1]->const_array(mfi);
                Array4<Real const> const& Jiz = Jfield[2]->const_array(mfi);
                Array4<Real const> const& Bx = Bfield[0]->const_array(mfi);
                Array4<Real const> const& By = Bfield[1]->const_array(mfi);
                Array4<Real const> const& Bz = Bfield[2]->const_array(mfi);

                Array4<Real const> Bx_ext, By_ext, Bz_ext;
                if (include_external_fields) {
                    Bx_ext = Bfield_external[0]->const_array(mfi);
                    By_ext = Bfield_external[1]->const_array(mfi);
                    Bz_ext = Bfield_external[2]->const_array(mfi);
                }

                amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    auto const jx_interp = Interp(Jx, Jx_stag, nodal, coarsen, i, j, k, 0);
                    auto const jy_interp = Interp(Jy, Jy_stag, nodal, coarsen, i, j, k, 0);
                    auto const jz_interp = Interp(Jz, Jz_stag, nodal, coarsen, i, j, k, 0);

                    auto const jix_interp = Interp(Jix, Jx_stag, nodal, coarsen, i, j, k, 0);
                    auto const jiy_interp = Interp(Jiy, Jy_stag, nodal, coarsen, i, j, k, 0);
                    auto const jiz_interp = Interp(Jiz, Jz_stag, nodal, coarsen, i, j, k, 0);

                    auto Bx_interp = Interp(Bx, Bx_stag, nodal, coarsen, i, j, k, 0);
                    auto By_interp = Interp(By, By_stag, nodal, coarsen, i, j, k, 0);
                    auto Bz_interp = Interp(Bz, Bz_stag, nodal, coarsen, i, j, k, 0);

                    if (include_external_fields) {
                        Bx_interp += Interp(Bx_ext, Bx_stag, nodal, coarsen, i, j, k, 0);
                        By_interp += Interp(By_ext, By_stag, nodal, coarsen, i, j, k, 0);
                        Bz_interp += Interp(Bz_ext, Bz_stag, nodal, coarsen, i, j, k, 0);
                    }

                    enE_nodal(i, j, k, 0) = (jy_interp - jiy_interp)*Bz_interp
                                          - (jz_interp - jiz_interp)*By_interp;
                    enE_nodal(i, j, k, 1) = (jz_interp - jiz_interp)*Bx_interp
                                          - (jx_interp - jix_interp)*Bz_interp;
                    enE_nodal(i, j, k, 2) = (jx_interp - jix_interp)*By_interp
                                          - (jy_interp - jiy_interp)*Bx_interp;
                });
            }
        }

        // E_target on the E staggering, with the same ghosts as E so the EB
        // fill and the divergence stencil have the layers they need.
        std::array<MultiFab, 3> E_target_mf;
        ablastr::fields::VectorField E_target;
        for (int c = 0; c < 3; ++c) {
            E_target_mf[c].define(
                Efield[c]->boxArray(), Efield[c]->DistributionMap(), 1,
                Efield[c]->nGrowVect());
            E_target_mf[c].setVal(0.0_rt);
            E_target[c] = &E_target_mf[c];
        }

#if defined(WARPX_DIM_RZ)
        const Real dr = dx[0];
        const Real rmin = geom.ProbLo(0);
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Array4<Real> const& Etx = E_target_mf[0].array(mfi);
            Array4<Real> const& Ety = E_target_mf[1].array(mfi);
            Array4<Real> const& Etz = E_target_mf[2].array(mfi);
            Array4<Real const> const& enE = enE_nodal_mf.const_array(mfi);
            Array4<Real const> const& rho = rhofield.const_array(mfi);
            Array4<Real const> const& Pe = electron_pressure_fp->const_array(mfi);

            Box const& tex = mfi.tilebox(Efield[0]->ixType().toIntVect());
            Box const& tey = mfi.tilebox(Efield[1]->ixType().toIntVect());
            Box const& tez = mfi.tilebox(Efield[2]->ixType().toIntVect());

#if defined(WARPX_DIM_3D)
            amrex::ParallelFor(tex, tey, tez,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    const Real rho_val = Interp(rho, nodal, Ex_stag, coarsen, i, j, k, 0);
                    const Real rho_lim = std::max(rho_val, rho_floor);
                    const Real grad_Pe = nodal_grid
                        ? (Pe(i+1, j, k) - Pe(i-1, j, k))*(0.5_rt*inv_dx[0])
                        : (Pe(i+1, j, k) - Pe(i, j, k))*inv_dx[0];
                    const Real enE_x = Interp(enE, nodal, Ex_stag, coarsen, i, j, k, 0);
                    Etx(i, j, k) = (enE_x - grad_Pe)/rho_lim;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    const Real rho_val = Interp(rho, nodal, Ey_stag, coarsen, i, j, k, 0);
                    const Real rho_lim = std::max(rho_val, rho_floor);
                    const Real grad_Pe = nodal_grid
                        ? (Pe(i, j+1, k) - Pe(i, j-1, k))*(0.5_rt*inv_dx[1])
                        : (Pe(i, j+1, k) - Pe(i, j, k))*inv_dx[1];
                    const Real enE_y = Interp(enE, nodal, Ey_stag, coarsen, i, j, k, 1);
                    Ety(i, j, k) = (enE_y - grad_Pe)/rho_lim;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    const Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, k, 0);
                    const Real rho_lim = std::max(rho_val, rho_floor);
                    const Real grad_Pe = nodal_grid
                        ? (Pe(i, j, k+1) - Pe(i, j, k-1))*(0.5_rt*inv_dx[2])
                        : (Pe(i, j, k+1) - Pe(i, j, k))*inv_dx[2];
                    const Real enE_z = Interp(enE, nodal, Ez_stag, coarsen, i, j, k, 2);
                    Etz(i, j, k) = (enE_z - grad_Pe)/rho_lim;
                });
#elif defined(WARPX_DIM_RZ)
            amrex::ParallelFor(tex, tey, tez,
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    const Real rho_val = Interp(rho, nodal, Ex_stag, coarsen, i, j, 0, 0);
                    const Real rho_lim = std::max(rho_val, rho_floor);
                    const Real grad_Pe = (Pe(i+1, j, 0) - Pe(i, j, 0))*inv_dx[0];
                    const Real enE_r = Interp(enE, nodal, Ex_stag, coarsen, i, j, 0, 0);
                    Etx(i, j, 0) = (enE_r - grad_Pe)/rho_lim;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    // Etheta is nodal in r; for m=0 the axis row must stay zero.
                    const Real r = rmin + i*dr;
                    if (r < 0.5_rt*dr) { Ety(i, j, 0) = 0.0_rt; return; }
                    const Real rho_val = Interp(rho, nodal, Ey_stag, coarsen, i, j, 0, 0);
                    const Real rho_lim = std::max(rho_val, rho_floor);
                    const Real enE_t = Interp(enE, nodal, Ey_stag, coarsen, i, j, 0, 1);
                    Ety(i, j, 0) = enE_t/rho_lim;  // no theta pressure gradient for m=0
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    const Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, 0, 0);
                    const Real rho_lim = std::max(rho_val, rho_floor);
                    const Real grad_Pe = nodal_grid
                        ? (Pe(i, j+1, 0) - Pe(i, j-1, 0))*(0.5_rt*inv_dx[1])
                        : (Pe(i, j+1, 0) - Pe(i, j, 0))*inv_dx[1];
                    const Real enE_z = Interp(enE, nodal, Ez_stag, coarsen, i, j, 0, 2);
                    Etz(i, j, 0) = (enE_z - grad_Pe)/rho_lim;
                });
#elif defined(WARPX_DIM_XZ)
            amrex::ParallelFor(tex, tey, tez,
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    const Real rho_val = Interp(rho, nodal, Ex_stag, coarsen, i, j, 0, 0);
                    const Real rho_lim = std::max(rho_val, rho_floor);
                    const Real grad_Pe = nodal_grid
                        ? (Pe(i+1, j, 0) - Pe(i-1, j, 0))*(0.5_rt*inv_dx[0])
                        : (Pe(i+1, j, 0) - Pe(i, j, 0))*inv_dx[0];
                    const Real enE_x = Interp(enE, nodal, Ex_stag, coarsen, i, j, 0, 0);
                    Etx(i, j, 0) = (enE_x - grad_Pe)/rho_lim;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    const Real rho_val = Interp(rho, nodal, Ey_stag, coarsen, i, j, 0, 0);
                    const Real rho_lim = std::max(rho_val, rho_floor);
                    const Real enE_y = Interp(enE, nodal, Ey_stag, coarsen, i, j, 0, 1);
                    Ety(i, j, 0) = enE_y/rho_lim;  // out-of-plane: no y pressure gradient
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    const Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, 0, 0);
                    const Real rho_lim = std::max(rho_val, rho_floor);
                    const Real grad_Pe = nodal_grid
                        ? (Pe(i, j+1, 0) - Pe(i, j-1, 0))*(0.5_rt*inv_dx[1])
                        : (Pe(i, j+1, 0) - Pe(i, j, 0))*inv_dx[1];
                    const Real enE_z = Interp(enE, nodal, Ez_stag, coarsen, i, j, 0, 2);
                    Etz(i, j, 0) = (enE_z - grad_Pe)/rho_lim;
                });
#endif
        }

        // Make the target boundary-consistent before taking its divergence:
        // enforce the embedded-boundary PEC condition on the target itself
        // and fill the box ghosts. No status cache is passed because the
        // persistent one belongs to Efield.
        if (eb_enabled) {
            warpx::hybrid::ApplyPECBoundaryToField(
                E_target, eb_update_E, *phi_mf, geom,
                m_eb_bc_rtol, m_eb_bc_max_iters, m_eb_bc_direct_fill,
                /*normal_odd=*/false, /*fill_covered_centers=*/true,
                nullptr);
        }
        for (int c = 0; c < 3; ++c) {
            ablastr::utils::communication::FillBoundary(
                E_target_mf[c], E_target_mf[c].nGrowVect(),
                WarpX::do_single_precision_comms, geom.periodicity());
        }

        fdtd->ComputeDivE(E_target, div_target);

        if (use_cache) {
            m_marder_target_cache[lev].define(
                nodal_ba, rhofield.DistributionMap(), 1, IntVect::TheZeroVector());
            MultiFab::Copy(m_marder_target_cache[lev], div_target, 0, 0, 1, 0);
            m_marder_target_cache_epoch[lev] = m_marder_field_epoch;
            m_marder_target_cache_rho[lev] = static_cast<const void*>(&rhofield);
        }
    }

    // Owner mask for the residual norm: nodal points shared between boxes
    // must be counted once (same semantics as MultiFab::norm2 with
    // periodicity). It is built once per application and reused every iteration.
    auto owner_mask = amrex::OwnerMask(div_err, geom.periodicity());

    // ---------------------------------------------------------------------
    // Fixed-point iteration: E += alpha_scaled * mask * grad(div(E) - div_target)
    // ---------------------------------------------------------------------
    int n_iter = 0;
    Real resid0 = 0.0_rt;
    Real final_resid = std::numeric_limits<Real>::infinity();

    for (n_iter = 1; n_iter <= max_iters_v; ++n_iter) {
        // The nodal divergence at box faces reads one E ghost layer.
        for (int c = 0; c < 3; ++c) {
            ablastr::utils::communication::FillBoundary(
                *Efield[c], IntVect(1),
                WarpX::do_single_precision_comms, geom.periodicity());
        }

        fdtd->ComputeDivE(Efield, div_err);

        // Restrict the error to the transition band (0 < rho <= rho_floor),
        // excluding nodes inside the conductor (the mirrored rho can be
        // positive there when the Neumann rho fill is selected), and
        // accumulate the owner-masked L2 residual in the same sweep.
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<Real> reduce_data(reduce_op);
        for (MFIter mfi(div_err, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Array4<Real> const& derr = div_err.array(mfi);
            Array4<Real const> const& dtar = div_target.const_array(mfi);
            Array4<Real const> const& rho = rhofield.const_array(mfi);
            Array4<int const> const& own = owner_mask->const_array(mfi);
            Array4<Real const> phi;
            if (eb_enabled) { phi = phi_mf->const_array(mfi); }
            reduce_op.eval(mfi.tilebox(), reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> amrex::GpuTuple<Real>
            {
                const bool in_band = (rho(i, j, k) > 0.0_rt)
                    && (rho(i, j, k) <= rho_floor)
                    && (!phi || phi(i, j, k) > 0.0_rt);
                const Real err = in_band ? (derr(i, j, k) - dtar(i, j, k)) : 0.0_rt;
                derr(i, j, k) = err;
                return {own(i, j, k) ? err*err : 0.0_rt};
            });
        }
        Real sumsq = amrex::get<0>(reduce_data.value(reduce_op));
        amrex::ParallelDescriptor::ReduceRealSum(sumsq);
        const Real resid = std::sqrt(sumsq);
        if (n_iter == 1) { resid0 = std::max(resid, 1.e-30_rt); }
        final_resid = resid;
        if (resid < atol_v || resid < rtol_v*resid0) { break; }

        // The collocated update's centered grad(div_err) reads one node past a
        // box boundary, so propagate the masked error into the ghost layer.
        if (nodal_grid) {
            ablastr::utils::communication::FillBoundary(
                div_err, div_err_ng, WarpX::do_single_precision_comms,
                geom.periodicity());
        }

        // Update restricted to nodes/edges that lie in the transition band and
        // that the EB mask flags as solution points.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Array4<Real> const& Ex = Efield[0]->array(mfi);
            Array4<Real> const& Ey = Efield[1]->array(mfi);
            Array4<Real> const& Ez = Efield[2]->array(mfi);
            Array4<Real const> const& derr = div_err.const_array(mfi);
            Array4<Real const> const& rho = rhofield.const_array(mfi);

            amrex::Array4<int> update_Ex_arr, update_Ey_arr, update_Ez_arr;
            if (eb_enabled) {
                update_Ex_arr = eb_update_E[0]->array(mfi);
                update_Ey_arr = eb_update_E[1]->array(mfi);
                update_Ez_arr = eb_update_E[2]->array(mfi);
            }

            Box const& tex = mfi.tilebox(Efield[0]->ixType().toIntVect());
            Box const& tey = mfi.tilebox(Efield[1]->ixType().toIntVect());
            Box const& tez = mfi.tilebox(Efield[2]->ixType().toIntVect());

#if defined(WARPX_DIM_3D)
            amrex::ParallelFor(tex, tey, tez,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    if (update_Ex_arr && update_Ex_arr(i, j, k) == 0) { return; }
                    const Real rho_val = Interp(rho, nodal, Ex_stag, coarsen, i, j, k, 0);
                    if (rho_val <= 0.0_rt || rho_val > rho_floor) { return; }
                    Ex(i, j, k) += nodal_grid
                        ? alpha_scaled*(derr(i+1, j, k) - derr(i-1, j, k))*(0.5_rt*inv_dx[0])
                        : alpha_scaled*(derr(i+1, j, k) - derr(i, j, k))*inv_dx[0];
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    if (update_Ey_arr && update_Ey_arr(i, j, k) == 0) { return; }
                    const Real rho_val = Interp(rho, nodal, Ey_stag, coarsen, i, j, k, 0);
                    if (rho_val <= 0.0_rt || rho_val > rho_floor) { return; }
                    Ey(i, j, k) += nodal_grid
                        ? alpha_scaled*(derr(i, j+1, k) - derr(i, j-1, k))*(0.5_rt*inv_dx[1])
                        : alpha_scaled*(derr(i, j+1, k) - derr(i, j, k))*inv_dx[1];
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    if (update_Ez_arr && update_Ez_arr(i, j, k) == 0) { return; }
                    const Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, k, 0);
                    if (rho_val <= 0.0_rt || rho_val > rho_floor) { return; }
                    Ez(i, j, k) += nodal_grid
                        ? alpha_scaled*(derr(i, j, k+1) - derr(i, j, k-1))*(0.5_rt*inv_dx[2])
                        : alpha_scaled*(derr(i, j, k+1) - derr(i, j, k))*inv_dx[2];
                });
#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
            // In-plane components only: the out-of-plane component (Etheta
            // for RZ m=0, Ey for XZ) has no divergence contribution and
            // receives no correction.
            amrex::ignore_unused(Ey, update_Ey_arr, tey);
            amrex::ParallelFor(tex, tez,
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    if (update_Ex_arr && update_Ex_arr(i, j, 0) == 0) { return; }
                    const Real rho_val = Interp(rho, nodal, Ex_stag, coarsen, i, j, 0, 0);
                    if (rho_val <= 0.0_rt || rho_val > rho_floor) { return; }
                    Ex(i, j, 0) += nodal_grid
                        ? alpha_scaled*(derr(i+1, j, 0) - derr(i-1, j, 0))*(0.5_rt*inv_dx[0])
                        : alpha_scaled*(derr(i+1, j, 0) - derr(i, j, 0))*inv_dx[0];
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    if (update_Ez_arr && update_Ez_arr(i, j, 0) == 0) { return; }
                    const Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, 0, 0);
                    if (rho_val <= 0.0_rt || rho_val > rho_floor) { return; }
                    Ez(i, j, 0) += nodal_grid
                        ? alpha_scaled*(derr(i, j+1, 0) - derr(i, j-1, 0))*(0.5_rt*inv_dx[1])
                        : alpha_scaled*(derr(i, j+1, 0) - derr(i, j, 0))*inv_dx[1];
                });
#endif
        }

        // Re-apply the domain and embedded-boundary E conditions so the next
        // iteration's stencils (and the consumer of the corrected E) read
        // boundary-consistent values.
        const Real time = warpx.gett_old(0) + warpx.getdt(0);
        warpx.ApplyEfieldBoundary(lev, PatchType::fine, time);
        if (eb_enabled) {
            if (static_cast<int>(m_eb_bc_status_E.size()) <= lev) { m_eb_bc_status_E.resize(lev+1); }
            warpx::hybrid::ApplyPECBoundaryToField(
                Efield, eb_update_E, *phi_mf, geom,
                m_eb_bc_rtol, m_eb_bc_max_iters, m_eb_bc_direct_fill,
                /*normal_odd=*/false, /*fill_covered_centers=*/true,
                &m_eb_bc_status_E[lev]);
        }
    }
    n_iter = std::min(n_iter, max_iters_v);

    for (int c = 0; c < 3; ++c) {
        ablastr::utils::communication::FillBoundary(
            *Efield[c], Efield[c]->nGrowVect(),
            WarpX::do_single_precision_comms, geom.periodicity());
    }

    return {n_iter, final_resid};
#else
    amrex::ignore_unused(Efield, Jfield, Bfield, rhofield, eb_update_E, lev,
        alpha, max_iterations, rtol, atol, target);
    WARPX_ABORT_WITH_MESSAGE(
        "The transitional Marder correction is only supported in 3D Cartesian, "
        "2D Cartesian (XZ) and RZ geometry");
    return {0, 0.0_rt};
#endif
}
