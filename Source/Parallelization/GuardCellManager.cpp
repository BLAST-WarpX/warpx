/* Copyright 2019-2020 Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "GuardCellManager.H"

#ifndef WARPX_DIM_RZ
#    include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#    include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#    include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#else
#    include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#endif
#include "Filter/NCIGodfreyFilter.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Config.H>
#include <AMReX_INT.H>
#include <AMReX_Math.H>
#include <AMReX_ParmParse.H>
#include <AMReX_SPACE.H>

#include <algorithm>

using namespace amrex;

void
guardCellManager::Init (
    const amrex::Real dt,
    const amrex::Real *dx,
    const bool do_subcycling,
    const bool do_fdtd_nci_corr,
    ablastr::utils::enums::GridType grid_type,
    const bool do_moving_window,
    const int moving_window_dir,
    const int nox,
    const int nox_fft, const int noy_fft, const int noz_fft,
    const int nci_corr_stencil,
    const ElectromagneticSolverAlgo electromagnetic_solver_id,
    const int max_level,
    const amrex::Vector<amrex::Real>& v_galilean,
    const amrex::Vector<amrex::Real>& v_comoving,
    const bool safe_guard_cells,
    const int do_multi_J,
    const bool fft_do_time_averaging,
    const bool do_pml,
    const int do_pml_in_domain,
    const int pml_ncell,
    const amrex::Vector<amrex::IntVect>& ref_ratios,
    const bool use_filter,
    const amrex::IntVect& bilinear_filter_stencil_length)
{
    // When using subcycling, the particles on the fine level perform two pushes
    // before being redistributed ; therefore, we need one extra guard cell
    // (the particles may move by 2*c*dt)
    int ngx_tmp = (max_level > 0 && do_subcycling) ? nox+1 : nox;
    int ngy_tmp = (max_level > 0 && do_subcycling) ? nox+1 : nox;
    int ngz_tmp = (max_level > 0 && do_subcycling) ? nox+1 : nox;

    const bool galilean = (v_galilean[0] != 0. || v_galilean[1] != 0. || v_galilean[2] != 0.);
    const bool comoving = (v_comoving[0] != 0. || v_comoving[1] != 0. || v_comoving[2] != 0.);

    // Add one guard cell in the case of the Galilean or comoving algorithms
    if (galilean || comoving)
    {
      ngx_tmp += 1;
      ngy_tmp += 1;
      ngz_tmp += 1;
    }

    // Ex, Ey, Ez, Bx, By, and Bz have the same number of ghost cells.
    // jx, jy, jz and rho have the same number of ghost cells.
    // E and B have the same number of ghost cells as j and rho if NCI filter is not used,
    // but different number of ghost cells in z-direction if NCI filter is used.
    // The number of cells should be even, in order to easily perform the
    // interpolation from coarse grid to fine grid.
    int ngx = (ngx_tmp % 2) ? ngx_tmp+1 : ngx_tmp;  // Always even number
    int ngy = (ngy_tmp % 2) ? ngy_tmp+1 : ngy_tmp;  // Always even number
    const int ngz_nonci = (ngz_tmp % 2) ? ngz_tmp+1 : ngz_tmp;  // Always even number
    int ngz;
    if (do_fdtd_nci_corr) {
        const int ng = ngz_tmp + nci_corr_stencil;
        ngz = (ng % 2) ? ng+1 : ng;
    } else {
        ngz = ngz_nonci;
    }

    // J is only interpolated from fine to coarse (not coarse to fine)
    // and therefore does not need to be even.
    int ngJx = ngx_tmp;
    int ngJy = ngy_tmp;
    int ngJz = ngz_tmp;

    // When calling the moving window (with one level of refinement), we shift
    // the fine grid by a number of cells equal to the ref_ratio in the moving
    // window direction.
    if (do_moving_window) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(max_level <= 1,
            "The number of grow cells for the moving window currently assumes 2 levels max.");
        const auto nlevs = static_cast<int>(max_level+1);
        const int max_r = (nlevs > 1) ? ref_ratios[0][moving_window_dir] : 2;

        ngx = std::max(ngx,max_r);
        ngy = std::max(ngy,max_r);
        ngz = std::max(ngz,max_r);
        ngJx = std::max(ngJx,max_r);
        ngJy = std::max(ngJy,max_r);
        ngJz = std::max(ngJz,max_r);
    }

#if defined(WARPX_DIM_3D)
    ng_alloc_EB = IntVect(ngx,ngy,ngz);
    ng_alloc_J = IntVect(ngJx,ngJy,ngJz);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    ng_alloc_EB = IntVect(ngx,ngz);
    ng_alloc_J = IntVect(ngJx,ngJz);
    amrex::ignore_unused(ngy, ngJy);
#elif defined(WARPX_DIM_1D_Z)
    ng_alloc_EB = IntVect(ngz);
    ng_alloc_J = IntVect(ngJz);
    amrex::ignore_unused(ngx, ngJx, ngy, ngJy);
#endif

    // TODO Adding one cell for rho should not be necessary, given that the number of guard cells
    // now takes into account the time step (see code block below). However, this does seem to be
    // necessary in order to avoid some remaining instances of out-of-bound array access in
    // simulations with large time steps (revealed by building WarpX with BOUND_CHECK = TRUE).
    ng_alloc_Rho = ng_alloc_J+1;

    // Electromagnetic simulations: account for change in particle positions within half a time step
    // for current deposition and within one time step for charge deposition (since rho is needed
    // both at the beginning and at the end of the PIC iteration).
    // For the hybrid-PIC solver, the same number of guard cells are used as for
    // the electrostatic solver.
    if (electromagnetic_solver_id != ElectromagneticSolverAlgo::None &&
        electromagnetic_solver_id != ElectromagneticSolverAlgo::HybridPIC)
    {
        for (int i = 0; i < AMREX_SPACEDIM; i++)
        {
            amrex::Real dt_Rho = dt;
            amrex::Real dt_J = 0.5_rt*dt;
            if (do_multi_J) {
                // With multi_J + time averaging, particles can move during 2*dt per PIC cycle.
                if (fft_do_time_averaging){
                    dt_Rho = 2._rt*dt;
                    dt_J = 2._rt*dt;
                }
                // With multi_J but without time averaging, particles can move during dt per PIC
                // cycle for the current deposition as well.
                else {
                    dt_J = dt;
                }
            }
            ng_alloc_Rho[i] += static_cast<int>(std::ceil(PhysConst::c * dt_Rho / dx[i]));
            ng_alloc_J[i]   += static_cast<int>(std::ceil(PhysConst::c * dt_J / dx[i]));
        }
    }

    // Number of guard cells for local deposition of J and rho
    ng_depos_J   = ng_alloc_J;
    ng_depos_rho = ng_alloc_Rho;

    if (use_filter)
    {
        ng_alloc_J += bilinear_filter_stencil_length - amrex::IntVect(1);
    }

    // After pushing particle
    int ng_alloc_F_int = (do_moving_window) ? 2 : 0;
    // CKC solver requires one additional guard cell
    if (electromagnetic_solver_id == ElectromagneticSolverAlgo::CKC) { ng_alloc_F_int = std::max( ng_alloc_F_int, 1 ); }
    ng_alloc_F = IntVect(AMREX_D_DECL(ng_alloc_F_int, ng_alloc_F_int, ng_alloc_F_int));

    // Used if warpx.do_divb_cleaning = 1
    int ng_alloc_G_int = (do_moving_window) ? 2 : 1;
    // TODO Does the CKC solver require one additional guard cell (as for F)?
    ng_alloc_G = IntVect(AMREX_D_DECL(ng_alloc_G_int, ng_alloc_G_int, ng_alloc_G_int));

    if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD)
    {
        // The number of guard cells should be enough to contain the stencil of the FFT solver.
        //
        // Here, this number (ngFFT) is determined empirically to be the order of the solver or
        // half the order of the solver, depending on other various numerical parameters.
        //
        // With the standard PSATD algorithm, simulations on staggered grids usually work fine
        // with a number of guard cells equal to half the number of guard cells that would be
        // used on nodal grids, in all directions x, y and z.
        //
        // On the other hand, with the Galilean PSATD or averaged Galilean PSATD algorithms,
        // with a Galilean coordinate transformation directed only in z, it seems more robust
        // to set the same number of guard cells in z, irrespective of whether the simulation
        // runs on nodal grids or staggered grids (typically with centering of fields and/or
        // currents in the latter case). This does not seem to be necessary in x and y,
        // where it still seems fine to set half the number of guard cells of the nodal case.

        using namespace ablastr::utils::enums;
        int ngFFt_x = (grid_type == GridType::Collocated) ? nox_fft : nox_fft / 2;
        int ngFFt_y = (grid_type == GridType::Collocated) ? noy_fft : noy_fft / 2;
        int ngFFt_z = (grid_type == GridType::Collocated || galilean) ? noz_fft : noz_fft / 2;

        const ParmParse pp_psatd("psatd");
        utils::parser::queryWithParser(pp_psatd, "nx_guard", ngFFt_x);
        utils::parser::queryWithParser(pp_psatd, "ny_guard", ngFFt_y);
        utils::parser::queryWithParser(pp_psatd, "nz_guard", ngFFt_z);

#if defined(WARPX_DIM_3D)
        auto ngFFT = IntVect(ngFFt_x, ngFFt_y, ngFFt_z);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        auto ngFFT = IntVect(ngFFt_x, ngFFt_z);
#elif defined(WARPX_DIM_1D_Z)
        auto ngFFT = IntVect(ngFFt_z);
#endif

#ifdef WARPX_DIM_RZ
        if (do_pml) {
            if (!do_pml_in_domain) {
                ngFFT[0] = std::max(ngFFT[0], pml_ncell);
            }
        }
#else
        amrex::ignore_unused(do_pml, do_pml_in_domain, pml_ncell);
#endif

        // All boxes should have the same number of guard cells, to avoid temporary parallel copies:
        // thus we take the maximum of the required number of guard cells over all available fields.
        for (int i_dim = 0; i_dim < AMREX_SPACEDIM; i_dim++) {
            int ng_required = ngFFT[i_dim];
            // Get the max
            ng_required = std::max(ng_required, ng_alloc_EB[i_dim]);
            ng_required = std::max(ng_required, ng_alloc_J[i_dim]);
            ng_required = std::max(ng_required, ng_alloc_Rho[i_dim]);
            ng_required = std::max(ng_required, ng_alloc_F[i_dim]);
            // Set the guard cells to this max
            ng_alloc_EB[i_dim] = ng_required;
            ng_alloc_J[i_dim] = ng_required;
            ng_alloc_F[i_dim] = ng_required;
            ng_alloc_Rho[i_dim] = ng_required;
            ng_alloc_F_int = ng_required;
            ng_alloc_G_int = ng_required;
        }
        ng_alloc_F = IntVect(AMREX_D_DECL(ng_alloc_F_int, ng_alloc_F_int, ng_alloc_F_int));
        ng_alloc_G = IntVect(AMREX_D_DECL(ng_alloc_G_int, ng_alloc_G_int, ng_alloc_G_int));
    }

    // Compute number of cells required for Field Solver
    if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
        ng_FieldSolver = ng_alloc_EB;
        ng_FieldSolverF = ng_alloc_EB;
        ng_FieldSolverG = ng_alloc_EB;
    }
#ifdef WARPX_DIM_RZ
    else if (electromagnetic_solver_id == ElectromagneticSolverAlgo::None ||
             electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee ||
             electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC ) {
        ng_FieldSolver  = CylindricalYeeAlgorithm::GetMaxGuardCell();
        ng_FieldSolverF = CylindricalYeeAlgorithm::GetMaxGuardCell();
        ng_FieldSolverG = CylindricalYeeAlgorithm::GetMaxGuardCell();
    }
#else
    else {
        if (grid_type == GridType::Collocated) {
            ng_FieldSolver  = CartesianNodalAlgorithm::GetMaxGuardCell();
            ng_FieldSolverF = CartesianNodalAlgorithm::GetMaxGuardCell();
            ng_FieldSolverG = CartesianNodalAlgorithm::GetMaxGuardCell();
        } else if (electromagnetic_solver_id == ElectromagneticSolverAlgo::None ||
                   electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee ||
                   electromagnetic_solver_id == ElectromagneticSolverAlgo::ECT ||
                   electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC ) {
            ng_FieldSolver  = CartesianYeeAlgorithm::GetMaxGuardCell();
            ng_FieldSolverF = CartesianYeeAlgorithm::GetMaxGuardCell();
            ng_FieldSolverG = CartesianYeeAlgorithm::GetMaxGuardCell();
        } else if (electromagnetic_solver_id == ElectromagneticSolverAlgo::CKC) {
            ng_FieldSolver  = CartesianCKCAlgorithm::GetMaxGuardCell();
            ng_FieldSolverF = CartesianCKCAlgorithm::GetMaxGuardCell();
            ng_FieldSolverG = CartesianCKCAlgorithm::GetMaxGuardCell();
        }
    }
#endif

    // Number of guard cells is the max of that determined by particle shape factor and
    // the stencil used in the field solve
    ng_alloc_EB.max( ng_FieldSolver );
    ng_alloc_F.max( ng_FieldSolverF );
    ng_alloc_G.max( ng_FieldSolverG );

    if (do_moving_window && electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
        ng_afterPushPSATD = ng_alloc_EB;
    }

    if (safe_guard_cells){
        // Run in safe mode: exchange all allocated guard cells at each
        // call of FillBoundary
        ng_FieldSolver = ng_alloc_EB;
        ng_FieldSolverF = ng_alloc_F;
        ng_FieldSolverG = ng_alloc_G;
        ng_FieldGather = ng_alloc_EB;
        ng_UpdateAux = ng_alloc_EB;
        ng_afterPushPSATD = ng_alloc_EB;
        if (do_moving_window){
            ng_MovingWindow = ng_alloc_EB;
        }
    } else {
        // Compute number of cells required for Field Gather:
        // When increasing the shape by order by one, support of the shape
        // factor grows symmetrically by half a cell on each side. So every
        // +2 orders, one touches one more cell point.
        int const FGcell = (nox + 1) / 2;  // integer division
        auto ng_FieldGather_noNCI = IntVect(AMREX_D_DECL(FGcell,FGcell,FGcell));
        ng_FieldGather_noNCI = ng_FieldGather_noNCI.min(ng_alloc_EB);

        // If NCI filter, add guard cells in the z direction
        IntVect ng_NCIFilter = IntVect::TheZeroVector();
        if (do_fdtd_nci_corr) {
            ng_NCIFilter[WARPX_ZINDEX] = NCIGodfreyFilter::m_stencil_width;
        }

        // Note: communications of guard cells for bilinear filter are handled
        // separately.
        ng_FieldGather = ng_FieldGather_noNCI + ng_NCIFilter;

        // Guard cells for auxiliary grid.
        // Not sure why there is a 2* here...
        ng_UpdateAux = 2*ng_FieldGather_noNCI + ng_NCIFilter;

        // Make sure we do not exchange more guard cells than allocated.
        ng_FieldGather = ng_FieldGather.min(ng_alloc_EB);
        ng_UpdateAux = ng_UpdateAux.min(ng_alloc_EB);
        // Only FillBoundary(ng_FieldGather) is called between consecutive
        // field solves. So ng_FieldGather must have enough cells
        // for the field solve too.
        ng_FieldGather = ng_FieldGather.max(ng_FieldSolver);

        if (do_moving_window){
            ng_MovingWindow[moving_window_dir] = 1;
        }
    }
}
