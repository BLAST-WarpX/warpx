/* Copyright 2019-2020 Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_GUARDCELLMANAGER_H_
#define WARPX_GUARDCELLMANAGER_H_

#include <Utils/WarpXAlgorithmSelection.H>
#include <ablastr/utils/Enums.H>

#include <AMReX_Array.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>
#include <AMReX_Vector.H>

/**
 * \brief This class computes and stores the number of guard cells needed for
 * the allocation of the MultiFabs and required for each part of the PIC loop.
 */
class guardCellManager{

public:

    /**
     * \brief Initialize number of guard cells depending on the options used.
     *
     * \param dt time step
     * \param dx cell spacing
     * \param do_subcycling bool, whether to use subcycling
     * \param do_fdtd_nci_corr bool, whether to use Godfrey NCI corrector
     * \param grid_type integer, whether the grid is collocated or staggered
     * \param do_moving_window bool, whether to use moving window
     * \param moving_window_dir direction of moving window
     * \param nox order of current deposition
     * \param nox_fft order of PSATD in x direction
     * \param noy_fft order of PSATD in y direction
     * \param noz_fft order of PSATD in z direction
     * \param nci_corr_stencil stencil of NCI corrector
     * \param electromagnetic_solver_id Integer corresponding to the type of Maxwell solver
     * \param max_level max level of the simulation
     * \param v_galilean Velocity used in the Galilean PSATD scheme
     * \param v_comoving Velocity used in the comoving PSATD scheme
     * \param safe_guard_cells Run in safe mode, exchanging more guard cells, and more often in the PIC loop (for debugging).
     * \param do_multi_J Whether to use the multi-J PSATD scheme
     * \param fft_do_time_averaging Whether to average the E and B field in time (with PSATD) before interpolating them onto the macro-particles
     * \param do_pml whether pml is turned on (only used by RZ PSATD)
     * \param do_pml_in_domain whether pml is done in the domain (only used by RZ PSATD)
     * \param pml_ncell number of cells on the pml layer (only used by RZ PSATD)
     * \param ref_ratios mesh refinement ratios between mesh-refinement levels
     * \param use_filter whether filtering will be done
     * \param bilinear_filter_stencil_length the size of the stencil for filtering
     */
    void Init(
        amrex::Real dt,
        const amrex::Real * dx,
        bool do_subcycling,
        bool do_fdtd_nci_corr,
        ablastr::utils::enums::GridType grid_type,
        bool do_moving_window,
        int moving_window_dir,
        int nox,
        int nox_fft, int noy_fft, int noz_fft,
        int nci_corr_stencil,
        ElectromagneticSolverAlgo electromagnetic_solver_id,
        int max_level,
        const amrex::Vector<amrex::Real>& v_galilean,
        const amrex::Vector<amrex::Real>& v_comoving,
        bool safe_guard_cells,
        int do_multi_J,
        bool fft_do_time_averaging,
        bool do_pml,
        int do_pml_in_domain,
        int pml_ncell,
        const amrex::Vector<amrex::IntVect>& ref_ratios,
        bool use_filter,
        const amrex::IntVect& bilinear_filter_stencil_length);

    // Guard cells allocated for MultiFabs E and B
    amrex::IntVect ng_alloc_EB = amrex::IntVect::TheZeroVector();
    // Guard cells allocated for MultiFab J
    amrex::IntVect ng_alloc_J = amrex::IntVect::TheZeroVector();
    // Guard cells allocated for MultiFab Rho
    amrex::IntVect ng_alloc_Rho = amrex::IntVect::TheZeroVector();
    // Guard cells allocated for MultiFab F
    amrex::IntVect ng_alloc_F = amrex::IntVect::TheZeroVector();
    // Guard cells allocated for MultiFab G
    amrex::IntVect ng_alloc_G = amrex::IntVect::TheZeroVector();

    // Guard cells exchanged for specific parts of the PIC loop

    // Number of guard cells of E and B that must exchanged before Field Solver
    amrex::IntVect ng_FieldSolver = amrex::IntVect::TheZeroVector();
    // Number of guard cells of F that must exchanged before Field Solver
    amrex::IntVect ng_FieldSolverF = amrex::IntVect::TheZeroVector();
    // Number of guard cells of G that must be exchanged before Field Solver
    amrex::IntVect ng_FieldSolverG = amrex::IntVect::TheZeroVector();
    // Number of guard cells of E and B that must exchanged before Field Gather
    amrex::IntVect ng_FieldGather = amrex::IntVect::TheZeroVector();
    // Number of guard cells of E and B that must exchanged before updating the Aux grid
    amrex::IntVect ng_UpdateAux = amrex::IntVect::TheZeroVector();
    // Number of guard cells of all MultiFabs that must exchanged before moving window
    amrex::IntVect ng_MovingWindow = amrex::IntVect::TheZeroVector();
    // Number of guard cells of E and B that are exchanged immediately after the main PSATD push
    amrex::IntVect ng_afterPushPSATD = amrex::IntVect::TheZeroVector();

    // Number of guard cells for local deposition of J and rho
    amrex::IntVect ng_depos_J   = amrex::IntVect::TheZeroVector();
    amrex::IntVect ng_depos_rho = amrex::IntVect::TheZeroVector();
};

#endif // WARPX_GUARDCELLMANAGER_H_
