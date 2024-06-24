/* Copyright 2019-2020 Andrew Myers, Ann Almgren, Aurore Blelly
 *                     Axel Huebl, Burlen Loring, David Grote
 *                     Glenn Richardson, Jean-Luc Vay, Luca Fedeli
 *                     Maxence Thevenet, Remi Lehe, Revathi Jambunathan
 *                     Weiqun Zhang, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#include "BoundaryConditions/PML.H"
#include "Diagnostics/MultiDiagnostics.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "Evolve/WarpXDtType.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#ifdef WARPX_USE_FFT
#   ifdef WARPX_DIM_RZ
#       include "FieldSolver/SpectralSolver/SpectralSolverRZ.H"
#   else
#       include "FieldSolver/SpectralSolver/SpectralSolver.H"
#   endif
#endif
#include "Parallelization/GuardCellManager.H"
#include "Particles/MultiParticleContainer.H"
#include "Fluids/MultiFluidContainer.H"
#include "Fluids/WarpXFluidContainer.H"
#include "Particles/ParticleBoundaryBuffer.H"
#include "Python/callbacks.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXUtil.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <ablastr/utils/SignalHandling.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>
#include <memory>
#include <ostream>
#include <vector>

using namespace amrex;
using ablastr::utils::SignalHandling;

void
WarpX::Evolve (int numsteps)
{
    WARPX_PROFILE_REGION("WarpX::Evolve()");
    WARPX_PROFILE("WarpX::Evolve()");

    Real cur_time = t_new[0];

    // Note that the default argument is numsteps = -1
    const int numsteps_max = (numsteps < 0)?(max_step):(istep[0] + numsteps);

    // check typos in inputs after step 1
    bool early_params_checked = false;

    static Real evolve_time = 0;

    const int step_begin = istep[0];
    for (int step = istep[0]; step < numsteps_max && cur_time < stop_time; ++step)
    {
        WARPX_PROFILE("WarpX::Evolve::step");
        const auto evolve_time_beg_step = static_cast<Real>(amrex::second());

        // Check and clear signal flags and asynchronously broadcast them from process 0
        SignalHandling::CheckSignals();

        multi_diags->NewIteration();

        // Start loop on time steps
        if (verbose) {
            amrex::Print() << "STEP " << step+1 << " starts ...\n";
        }
        ExecutePythonCallback("beforestep");

        CheckLoadBalance(step);

        if (evolve_scheme == EvolveScheme::Explicit)
        {
            ExplicitFillBoundaryEBUpdateAux();
        }

        // If needed, deposit the initial ion charge and current densities that
        // will be used to update the E-field in Ohm's law.
        if (step == step_begin &&
            electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC
        ) {
            HybridPICDepositInitialRhoAndJ();
        }

        // Run multi-physics modules:
        // ionization, Coulomb collisions, QED
        doFieldIonization();

        ExecutePythonCallback("beforecollisions");
        mypc->doCollisions( cur_time, dt[0] );
        ExecutePythonCallback("aftercollisions");

#ifdef WARPX_QED
        doQEDEvents();
        mypc->doQEDSchwinger();
#endif

        // Main PIC operation:
        // gather fields, push particles, deposit sources, update fields

        ExecutePythonCallback("particleinjection");

        if (m_implicit_solver) {
            m_implicit_solver->OneStep(cur_time, dt[0], step);
        }
        else if ( electromagnetic_solver_id == ElectromagneticSolverAlgo::None ||
             electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC )
        {
            // Electrostatic or hybrid-PIC case: only gather fields and push
            // particles, deposition and calculation of fields done further below
            const bool skip_deposition = true;
            PushParticlesandDeposit(cur_time, skip_deposition);
        }
        // Electromagnetic case: multi-J algorithm
        else if (do_multi_J)
        {
            OneStep_multiJ(cur_time);
        }
        // Electromagnetic case: no subcycling or no mesh refinement
        else if (do_subcycling == 0 || finest_level == 0)
        {
            OneStep_nosub(cur_time);
            // E: guard cells are up-to-date
            // B: guard cells are NOT up-to-date
            // F: guard cells are NOT up-to-date
        }
        // Electromagnetic case: subcycling with one level of mesh refinement
        else if (do_subcycling == 1 && finest_level == 1)
        {
            OneStep_sub1(cur_time);
        }
        else
        {
            WARPX_ABORT_WITH_MESSAGE(
                "do_subcycling = " + std::to_string(do_subcycling)
                + " is an unsupported do_subcycling type.");
        }

        // Resample particles
        // +1 is necessary here because value of step seen by user (first step is 1) is different than
        // value of step in code (first step is 0)
        mypc->doResampling(istep[0]+1, verbose);

        if (evolve_scheme == EvolveScheme::Explicit) {
            applyMirrors(cur_time);
            // E : guard cells are NOT up-to-date
            // B : guard cells are NOT up-to-date
        }

        // TODO: move out
        if (evolve_scheme == EvolveScheme::Explicit) {
            if (cur_time + dt[0] >= stop_time - 1.e-3*dt[0] || step == numsteps_max-1) {
                // At the end of last step, push p by 0.5*dt to synchronize
                FillBoundaryE(guard_cells.ng_FieldGather);
                FillBoundaryB(guard_cells.ng_FieldGather);
                if (fft_do_time_averaging)
                {
                    FillBoundaryE_avg(guard_cells.ng_FieldGather);
                    FillBoundaryB_avg(guard_cells.ng_FieldGather);
                }
                UpdateAuxilaryData();
                FillBoundaryAux(guard_cells.ng_UpdateAux);
                for (int lev = 0; lev <= finest_level; ++lev) {
                    mypc->PushP(lev, 0.5_rt*dt[lev],
                                *Efield_aux[lev][0],*Efield_aux[lev][1],
                                *Efield_aux[lev][2],
                                *Bfield_aux[lev][0],*Bfield_aux[lev][1],
                                *Bfield_aux[lev][2]);
                }
                is_synchronized = true;
            }
        }

        for (int lev = 0; lev <= max_level; ++lev) {
            ++istep[lev];
        }

        cur_time += dt[0];

        ShiftGalileanBoundary();

        // sync up time
        for (int i = 0; i <= max_level; ++i) {
            t_new[i] = cur_time;
        }
        multi_diags->FilterComputePackFlush( step, false, true );

        const bool move_j = is_synchronized;
        // If is_synchronized we need to shift j too so that next step we can evolve E by dt/2.
        // We might need to move j because we are going to make a plotfile.
        const int num_moved = MoveWindow(step+1, move_j);

        // Update the accelerator lattice element finder if the window has moved,
        // from either a moving window or a boosted frame
        if (num_moved != 0 || gamma_boost > 1) {
            for (int lev = 0; lev <= finest_level; ++lev) {
                m_accelerator_lattice[lev]->UpdateElementFinder(lev);
            }
        }

        HandleParticlesAtBoundaries(step, cur_time, num_moved);

        // Field solve step for electrostatic or hybrid-PIC solvers
        if( electrostatic_solver_id != ElectrostaticSolverAlgo::None ||
            electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC )
        {
            ExecutePythonCallback("beforeEsolve");

            if (electrostatic_solver_id != ElectrostaticSolverAlgo::None) {
                // Electrostatic solver:
                // For each species: deposit charge and add the associated space-charge
                // E and B field to the grid ; this is done at the end of the PIC
                // loop (i.e. immediately after a `Redistribute` and before particle
                // positions are next pushed) so that the particles do not deposit out of bounds
                // and so that the fields are at the correct time in the output.
                bool const reset_fields = true;
                ComputeSpaceChargeField( reset_fields );
                if (electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameElectroMagnetostatic) {
                    // Call Magnetostatic Solver to solve for the vector potential A and compute the
                    // B field.  Time varying A contribution to E field is neglected.
                    // This is currently a lab frame calculation.
                    ComputeMagnetostaticField();
                }
                AddExternalFields();
            } else if (electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC) {
                // Hybrid-PIC case:
                // The particles are now at p^{n+1/2} and x^{n+1}. The fields
                // are updated according to the hybrid-PIC scheme (Ohm's law
                // and Ampere's law).
                HybridPICEvolveFields();
            }
            ExecutePythonCallback("afterEsolve");
        }

        // afterstep callback runs with the updated global time. It is included
        // in the evolve timing.
        ExecutePythonCallback("afterstep");

        /// reduced diags
        if (reduced_diags->m_plot_rd != 0)
        {
            reduced_diags->LoadBalance();
            reduced_diags->ComputeDiags(step);
            reduced_diags->WriteToFile(step);
        }
        multi_diags->FilterComputePackFlush( step );

        // execute afterdiagnostic callbacks
        ExecutePythonCallback("afterdiagnostics");

        // inputs: unused parameters (e.g. typos) check after step 1 has finished
        if (!early_params_checked) {
            checkEarlyUnusedParams();
            early_params_checked = true;
        }

        // create ending time stamp for calculating elapsed time each iteration
        const auto evolve_time_end_step = static_cast<Real>(amrex::second());
        evolve_time += evolve_time_end_step - evolve_time_beg_step;

        HandleSignals();

        if (verbose) {
            amrex::Print()<< "STEP " << step+1 << " ends." << " TIME = " << cur_time
                        << " DT = " << dt[0] << "\n";
            amrex::Print()<< "Evolve time = " << evolve_time
                      << " s; This step = " << evolve_time_end_step-evolve_time_beg_step
                      << " s; Avg. per step = " << evolve_time/(step-step_begin+1) << " s\n\n";
        }

        if (checkStopSimulation(cur_time)) {
            break;
        }
    } // End loop on time steps

    // This if statement is needed for PICMI, which allows the Evolve routine to be
    // called multiple times, otherwise diagnostics will be done at every call,
    // regardless of the diagnostic period parameter provided in the inputs.
    if (istep[0] == max_step || (stop_time - 1.e-3*dt[0] <= cur_time && cur_time < stop_time + dt[0])
        || m_exit_loop_due_to_interrupt_signal) {
        multi_diags->FilterComputePackFlushLastTimestep( istep[0] );
        if (m_exit_loop_due_to_interrupt_signal) { ExecutePythonCallback("onbreaksignal"); }
    }

    amrex::Print() <<
        ablastr::warn_manager::GetWMInstance().PrintGlobalWarnings("THE END");
}

/* /brief Perform one PIC iteration, without subcycling
*  i.e. all levels/patches use the same timestep (that of the finest level)
*  for the field advance and particle pusher.
*/
void
WarpX::OneStep_nosub (Real cur_time)
{
    WARPX_PROFILE("WarpX::OneStep_nosub()");

    // Push particle from x^{n} to x^{n+1}
    //               from p^{n-1/2} to p^{n+1/2}
    // Deposit current j^{n+1/2}
    // Deposit charge density rho^{n}

    ExecutePythonCallback("particlescraper");
    ExecutePythonCallback("beforedeposition");

    PushParticlesandDeposit(cur_time);

    ExecutePythonCallback("afterdeposition");

    // Synchronize J and rho:
    // filter (if used), exchange guard cells, interpolate across MR levels
    // and apply boundary conditions
    SyncCurrentAndRho();

    // At this point, J is up-to-date inside the domain, and E and B are
    // up-to-date including enough guard cells for first step of the field
    // solve.

    // For extended PML: copy J from regular grid to PML, and damp J in PML
    if (do_pml && pml_has_particles) { CopyJPML(); }
    if (do_pml && do_pml_j_damping) { DampJPML(); }

    ExecutePythonCallback("beforeEsolve");

    // Push E and B from {n} to {n+1}
    // (And update guard cells immediately afterwards)
    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
        if (use_hybrid_QED)
        {
            WarpX::Hybrid_QED_Push(dt);
            FillBoundaryE(guard_cells.ng_alloc_EB);
        }
        PushPSATD();

        if (do_pml) {
            DampPML();
        }

        if (use_hybrid_QED) {
            FillBoundaryE(guard_cells.ng_alloc_EB);
            FillBoundaryB(guard_cells.ng_alloc_EB, WarpX::sync_nodal_points);
            WarpX::Hybrid_QED_Push(dt);
            FillBoundaryE(guard_cells.ng_afterPushPSATD, WarpX::sync_nodal_points);
        }
        else {
            FillBoundaryE(guard_cells.ng_afterPushPSATD, WarpX::sync_nodal_points);
            FillBoundaryB(guard_cells.ng_afterPushPSATD, WarpX::sync_nodal_points);
            if (WarpX::do_dive_cleaning || WarpX::do_pml_dive_cleaning) {
                FillBoundaryF(guard_cells.ng_alloc_F, WarpX::sync_nodal_points);
            }
            if (WarpX::do_divb_cleaning || WarpX::do_pml_divb_cleaning) {
                FillBoundaryG(guard_cells.ng_alloc_G, WarpX::sync_nodal_points);
            }
        }
    } else {
        EvolveF(0.5_rt * dt[0], DtType::FirstHalf);
        EvolveG(0.5_rt * dt[0], DtType::FirstHalf);
        FillBoundaryF(guard_cells.ng_FieldSolverF);
        FillBoundaryG(guard_cells.ng_FieldSolverG);

        EvolveB(0.5_rt * dt[0], DtType::FirstHalf); // We now have B^{n+1/2}
        FillBoundaryB(guard_cells.ng_FieldSolver, WarpX::sync_nodal_points);

        if (WarpX::em_solver_medium == MediumForEM::Vacuum) {
            // vacuum medium
            EvolveE(dt[0]); // We now have E^{n+1}
        } else if (WarpX::em_solver_medium == MediumForEM::Macroscopic) {
            // macroscopic medium
            MacroscopicEvolveE(dt[0]); // We now have E^{n+1}
        } else {
            WARPX_ABORT_WITH_MESSAGE("Medium for EM is unknown");
        }
        FillBoundaryE(guard_cells.ng_FieldSolver, WarpX::sync_nodal_points);

        EvolveF(0.5_rt * dt[0], DtType::SecondHalf);
        EvolveG(0.5_rt * dt[0], DtType::SecondHalf);
        EvolveB(0.5_rt * dt[0], DtType::SecondHalf); // We now have B^{n+1}

        if (do_pml) {
            DampPML();
            FillBoundaryE(guard_cells.ng_MovingWindow, WarpX::sync_nodal_points);
            FillBoundaryB(guard_cells.ng_MovingWindow, WarpX::sync_nodal_points);
            FillBoundaryF(guard_cells.ng_MovingWindow, WarpX::sync_nodal_points);
            FillBoundaryG(guard_cells.ng_MovingWindow, WarpX::sync_nodal_points);
        }

        // E and B are up-to-date in the domain, but all guard cells are
        // outdated.
        if (safe_guard_cells) {
            FillBoundaryB(guard_cells.ng_alloc_EB);
        }
    } // !PSATD

    ExecutePythonCallback("afterEsolve");
}

bool WarpX::checkStopSimulation (amrex::Real cur_time)
{
    m_exit_loop_due_to_interrupt_signal = SignalHandling::TestAndResetActionRequestFlag(SignalHandling::SIGNAL_REQUESTS_BREAK);
    return (cur_time >= stop_time - 1.e-3*dt[0])  ||
        m_exit_loop_due_to_interrupt_signal;
}

void WarpX::checkEarlyUnusedParams ()
{
    amrex::Print() << "\n"; // better: conditional \n based on return value
    amrex::ParmParse::QueryUnusedInputs();

    // Print the warning list right after the first step.
    amrex::Print() << ablastr::warn_manager::GetWMInstance().PrintGlobalWarnings("FIRST STEP");
}

void WarpX::ExplicitFillBoundaryEBUpdateAux ()
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(evolve_scheme == EvolveScheme::Explicit,
        "Cannot call WarpX::ExplicitFillBoundaryEBUpdateAux wihtout Explicit evolve scheme set!");

    // At the beginning, we have B^{n} and E^{n}.
    // Particles have p^{n} and x^{n}.
    // is_synchronized is true.

    if (is_synchronized) {
        // Not called at each iteration, so exchange all guard cells
        FillBoundaryE(guard_cells.ng_alloc_EB);
        FillBoundaryB(guard_cells.ng_alloc_EB);

        UpdateAuxilaryData();
        FillBoundaryAux(guard_cells.ng_UpdateAux);
        // on first step, push p by -0.5*dt
        for (int lev = 0; lev <= finest_level; ++lev)
        {
            mypc->PushP(lev, -0.5_rt*dt[lev],
                        *Efield_aux[lev][0],*Efield_aux[lev][1],*Efield_aux[lev][2],
                        *Bfield_aux[lev][0],*Bfield_aux[lev][1],*Bfield_aux[lev][2]);
        }
        is_synchronized = false;

    } else {
        // Beyond one step, we have E^{n} and B^{n}.
        // Particles have p^{n-1/2} and x^{n}.
        // E and B: enough guard cells to update Aux or call Field Gather in fp and cp
        // Need to update Aux on lower levels, to interpolate to higher levels.

        // E and B are up-to-date inside the domain only
        FillBoundaryE(guard_cells.ng_FieldGather);
        FillBoundaryB(guard_cells.ng_FieldGather);
        if (electrostatic_solver_id == ElectrostaticSolverAlgo::None) {
            if (fft_do_time_averaging)
            {
                FillBoundaryE_avg(guard_cells.ng_FieldGather);
                FillBoundaryB_avg(guard_cells.ng_FieldGather);
            }
            // TODO Remove call to FillBoundaryAux before UpdateAuxilaryData?
            if (WarpX::electromagnetic_solver_id != ElectromagneticSolverAlgo::PSATD) {
                FillBoundaryAux(guard_cells.ng_UpdateAux);
            }
        }
        UpdateAuxilaryData();
        FillBoundaryAux(guard_cells.ng_UpdateAux);
    }
}

void WarpX::HandleParticlesAtBoundaries (int step, amrex::Real cur_time, int num_moved)
{
    mypc->ContinuousFluxInjection(cur_time, dt[0]);

    mypc->ApplyBoundaryConditions();
    m_particle_boundary_buffer->gatherParticlesFromDomainBoundaries(*mypc);

    // Non-Maxwell solver: particles can move by an arbitrary number of cells
    if( electromagnetic_solver_id == ElectromagneticSolverAlgo::None ||
        electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC )
    {
        mypc->Redistribute();
    }
    else
    {
        // Electromagnetic solver: due to CFL condition, particles can
        // only move by one or two cells per time step
        if (max_level == 0) {
            int num_redistribute_ghost = num_moved;
            if ((m_v_galilean[0]!=0) or (m_v_galilean[1]!=0) or (m_v_galilean[2]!=0)) {
                // Galilean algorithm ; particles can move by up to 2 cells
                num_redistribute_ghost += 2;
            } else {
                // Standard algorithm ; particles can move by up to 1 cell
                num_redistribute_ghost += 1;
            }
            mypc->RedistributeLocal(num_redistribute_ghost);
        }
        else {
            mypc->Redistribute();
        }
    }

    // interact the particles with EB walls (if present)
#ifdef AMREX_USE_EB
    mypc->ScrapeParticlesAtEB(amrex::GetVecOfConstPtrs(m_distance_to_eb));
    m_particle_boundary_buffer->gatherParticlesFromEmbeddedBoundaries(*mypc, amrex::GetVecOfConstPtrs(m_distance_to_eb));
    mypc->deleteInvalidParticles();
#endif

    if (sort_intervals.contains(step+1)) {
        if (verbose) {
            amrex::Print() << Utils::TextMsg::Info("re-sorting particles");
        }
        mypc->SortParticlesByBin(sort_bin_size);
    }
}

void WarpX::SyncCurrentAndRho ()
{
    if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD)
    {
        if (fft_periodic_single_box)
        {
            // With periodic single box, synchronize J and rho here,
            // even with current correction or Vay deposition
            if (current_deposition_algo == CurrentDepositionAlgo::Vay)
            {
                // TODO Replace current_cp with current_cp_vay once Vay deposition is implemented with MR
                SyncCurrent(current_fp_vay, current_cp, current_buf);
                SyncRho(rho_fp, rho_cp, charge_buf);
            }
            else
            {
                SyncCurrent(current_fp, current_cp, current_buf);
                SyncRho(rho_fp, rho_cp, charge_buf);
            }
        }
        else // no periodic single box
        {
            // Without periodic single box, synchronize J and rho here,
            // except with current correction or Vay deposition:
            // in these cases, synchronize later (in WarpX::PushPSATD)
            if (!current_correction &&
                current_deposition_algo != CurrentDepositionAlgo::Vay)
            {
                SyncCurrent(current_fp, current_cp, current_buf);
                SyncRho(rho_fp, rho_cp, charge_buf);
            }

            if (current_deposition_algo == CurrentDepositionAlgo::Vay)
            {
                // TODO This works only without mesh refinement
                const int lev = 0;
                if (use_filter) { ApplyFilterJ(current_fp_vay, lev); }
            }
        }
    }
    else // FDTD
    {
        SyncCurrent(current_fp, current_cp, current_buf);
        SyncRho(rho_fp, rho_cp, charge_buf);
    }

    // Reflect charge and current density over PEC boundaries, if needed.
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (rho_fp[lev]) {
            ApplyRhofieldBoundary(lev, rho_fp[lev].get(), PatchType::fine);
        }
        ApplyJfieldBoundary(
            lev, current_fp[lev][0].get(), current_fp[lev][1].get(),
            current_fp[lev][2].get(), PatchType::fine
        );
        if (lev > 0) {
            if (rho_cp[lev]) {
                ApplyRhofieldBoundary(lev, rho_cp[lev].get(), PatchType::coarse);
            }
            ApplyJfieldBoundary(
                lev, current_cp[lev][0].get(), current_cp[lev][1].get(),
                current_cp[lev][2].get(), PatchType::coarse
            );
        }
    }
}

void
WarpX::OneStep_multiJ (const amrex::Real cur_time)
{
#ifdef WARPX_USE_FFT

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD,
        "multi-J algorithm not implemented for FDTD"
    );

    const int rho_mid = spectral_solver_fp[0]->m_spectral_index.rho_mid;
    const int rho_new = spectral_solver_fp[0]->m_spectral_index.rho_new;

    // Push particle from x^{n} to x^{n+1}
    //               from p^{n-1/2} to p^{n+1/2}
    const bool skip_deposition = true;
    PushParticlesandDeposit(cur_time, skip_deposition);

    // Initialize multi-J loop:

    // 1) Prepare E,B,F,G fields in spectral space
    PSATDForwardTransformEB(Efield_fp, Bfield_fp, Efield_cp, Bfield_cp);
    if (WarpX::do_dive_cleaning) { PSATDForwardTransformF(); }
    if (WarpX::do_divb_cleaning) { PSATDForwardTransformG(); }

    // 2) Set the averaged fields to zero
    if (WarpX::fft_do_time_averaging) { PSATDEraseAverageFields(); }

    // 3) Deposit rho (in rho_new, since it will be moved during the loop)
    //    (after checking that pointer to rho_fp on MR level 0 is not null)
    if (rho_fp[0] && rho_in_time == RhoInTime::Linear)
    {
        // Deposit rho at relative time -dt
        // (dt[0] denotes the time step on mesh refinement level 0)
        mypc->DepositCharge(rho_fp, -dt[0]);
        // Filter, exchange boundary, and interpolate across levels
        SyncRho(rho_fp, rho_cp, charge_buf);
        // Forward FFT of rho
        PSATDForwardTransformRho(rho_fp, rho_cp, 0, rho_new);
    }

    // 4) Deposit J at relative time -dt with time step dt
    //    (dt[0] denotes the time step on mesh refinement level 0)
    if (J_in_time == JInTime::Linear)
    {
        auto& current = (do_current_centering) ? current_fp_nodal : current_fp;
        mypc->DepositCurrent(current, dt[0], -dt[0]);
        // Synchronize J: filter, exchange boundary, and interpolate across levels.
        // With current centering, the nodal current is deposited in 'current',
        // namely 'current_fp_nodal': SyncCurrent stores the result of its centering
        // into 'current_fp' and then performs both filtering, if used, and exchange
        // of guard cells.
        SyncCurrent(current_fp, current_cp, current_buf);
        // Forward FFT of J
        PSATDForwardTransformJ(current_fp, current_cp);
    }

    // Number of depositions for multi-J scheme
    const int n_deposit = WarpX::do_multi_J_n_depositions;
    // Time sub-step for each multi-J deposition
    const amrex::Real sub_dt = dt[0] / static_cast<amrex::Real>(n_deposit);
    // Whether to perform multi-J depositions on a time interval that spans
    // one or two full time steps (from n*dt to (n+1)*dt, or from n*dt to (n+2)*dt)
    const int n_loop = (WarpX::fft_do_time_averaging) ? 2*n_deposit : n_deposit;

    // Loop over multi-J depositions
    for (int i_deposit = 0; i_deposit < n_loop; i_deposit++)
    {
        // Move J from new to old if J is linear in time
        if (J_in_time == JInTime::Linear) { PSATDMoveJNewToJOld(); }

        const amrex::Real t_deposit_current = (J_in_time == JInTime::Linear) ?
            (i_deposit-n_deposit+1)*sub_dt : (i_deposit-n_deposit+0.5_rt)*sub_dt;

        const amrex::Real t_deposit_charge = (rho_in_time == RhoInTime::Linear) ?
            (i_deposit-n_deposit+1)*sub_dt : (i_deposit-n_deposit+0.5_rt)*sub_dt;

        // Deposit new J at relative time t_deposit_current with time step dt
        // (dt[0] denotes the time step on mesh refinement level 0)
        auto& current = (do_current_centering) ? current_fp_nodal : current_fp;
        mypc->DepositCurrent(current, dt[0], t_deposit_current);
        // Synchronize J: filter, exchange boundary, and interpolate across levels.
        // With current centering, the nodal current is deposited in 'current',
        // namely 'current_fp_nodal': SyncCurrent stores the result of its centering
        // into 'current_fp' and then performs both filtering, if used, and exchange
        // of guard cells.
        SyncCurrent(current_fp, current_cp, current_buf);
        // Forward FFT of J
        PSATDForwardTransformJ(current_fp, current_cp);

        // Deposit new rho
        // (after checking that pointer to rho_fp on MR level 0 is not null)
        if (rho_fp[0])
        {
            // Move rho from new to old if rho is linear in time
            if (rho_in_time == RhoInTime::Linear) { PSATDMoveRhoNewToRhoOld(); }

            // Deposit rho at relative time t_deposit_charge
            mypc->DepositCharge(rho_fp, t_deposit_charge);
            // Filter, exchange boundary, and interpolate across levels
            SyncRho(rho_fp, rho_cp, charge_buf);
            // Forward FFT of rho
            const int rho_idx = (rho_in_time == RhoInTime::Linear) ? rho_new : rho_mid;
            PSATDForwardTransformRho(rho_fp, rho_cp, 0, rho_idx);
        }

        if (WarpX::current_correction)
        {
            WARPX_ABORT_WITH_MESSAGE(
                "Current correction not implemented for multi-J algorithm.");
        }

        // Advance E,B,F,G fields in time and update the average fields
        PSATDPushSpectralFields();

        // Transform non-average fields E,B,F,G after n_deposit pushes
        // (the relative time reached here coincides with an integer full time step)
        if (i_deposit == n_deposit-1)
        {
            PSATDBackwardTransformEB(Efield_fp, Bfield_fp, Efield_cp, Bfield_cp);
            if (WarpX::do_dive_cleaning) { PSATDBackwardTransformF(); }
            if (WarpX::do_divb_cleaning) { PSATDBackwardTransformG(); }
        }
    }

    // Transform fields back to real space
    if (WarpX::fft_do_time_averaging)
    {
        // We summed the integral of the field over 2*dt
        PSATDScaleAverageFields(1._rt / (2._rt*dt[0]));
        PSATDBackwardTransformEBavg(Efield_avg_fp, Bfield_avg_fp, Efield_avg_cp, Bfield_avg_cp);
    }

    // Evolve fields in PML
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (do_pml && pml[lev]->ok())
        {
            pml[lev]->PushPSATD(lev);
        }
        ApplyEfieldBoundary(lev, PatchType::fine);
        if (lev > 0) { ApplyEfieldBoundary(lev, PatchType::coarse); }
        ApplyBfieldBoundary(lev, PatchType::fine, DtType::FirstHalf);
        if (lev > 0) { ApplyBfieldBoundary(lev, PatchType::coarse, DtType::FirstHalf); }
    }

    // Damp fields in PML before exchanging guard cells
    if (do_pml)
    {
        DampPML();
    }

    // Exchange guard cells and synchronize nodal points
    FillBoundaryE(guard_cells.ng_alloc_EB, WarpX::sync_nodal_points);
    FillBoundaryB(guard_cells.ng_alloc_EB, WarpX::sync_nodal_points);
    if (WarpX::do_dive_cleaning || WarpX::do_pml_dive_cleaning) {
        FillBoundaryF(guard_cells.ng_alloc_F, WarpX::sync_nodal_points);
    }
    if (WarpX::do_divb_cleaning || WarpX::do_pml_divb_cleaning) {
        FillBoundaryG(guard_cells.ng_alloc_G, WarpX::sync_nodal_points);
    }

#else
    amrex::ignore_unused(cur_time);
    WARPX_ABORT_WITH_MESSAGE(
        "multi-J algorithm not implemented for FDTD");
#endif // WARPX_USE_FFT
}

/* /brief Perform one PIC iteration, with subcycling
*  i.e. The fine patch uses a smaller timestep (and steps more often)
*  than the coarse patch, for the field advance and particle pusher.
*
* This version of subcycling only works for 2 levels and with a refinement
* ratio of 2.
* The particles and fields of the fine patch are pushed twice
* (with dt[coarse]/2) in this routine.
* The particles of the coarse patch and mother grid are pushed only once
* (with dt[coarse]). The fields on the coarse patch and mother grid
* are pushed in a way which is equivalent to pushing once only, with
* a current which is the average of the coarse + fine current at the 2
* steps of the fine grid.
*
*/
void
WarpX::OneStep_sub1 (Real cur_time)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        electrostatic_solver_id == ElectrostaticSolverAlgo::None,
        "Electrostatic solver cannot be used with sub-cycling."
    );

    // TODO: we could save some charge depositions

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(finest_level == 1, "Must have exactly two levels");
    const int fine_lev = 1;
    const int coarse_lev = 0;

    // i) Push particles and fields on the fine patch (first fine step)
    PushParticlesandDeposit(fine_lev, cur_time, DtType::FirstHalf);
    RestrictCurrentFromFineToCoarsePatch(current_fp, current_cp, fine_lev);
    RestrictRhoFromFineToCoarsePatch(rho_fp, rho_cp, fine_lev);
    if (use_filter) { ApplyFilterJ(current_fp, fine_lev); }
    SumBoundaryJ(current_fp, fine_lev, Geom(fine_lev).periodicity());
    ApplyFilterandSumBoundaryRho(rho_fp, rho_cp, fine_lev, PatchType::fine, 0, 2*ncomps);

    EvolveB(fine_lev, PatchType::fine, 0.5_rt*dt[fine_lev], DtType::FirstHalf);
    EvolveF(fine_lev, PatchType::fine, 0.5_rt*dt[fine_lev], DtType::FirstHalf);
    FillBoundaryB(fine_lev, PatchType::fine, guard_cells.ng_FieldSolver,
                  WarpX::sync_nodal_points);
    FillBoundaryF(fine_lev, PatchType::fine, guard_cells.ng_alloc_F,
                  WarpX::sync_nodal_points);

    EvolveE(fine_lev, PatchType::fine, dt[fine_lev]);
    FillBoundaryE(fine_lev, PatchType::fine, guard_cells.ng_FieldGather);

    EvolveB(fine_lev, PatchType::fine, 0.5_rt*dt[fine_lev], DtType::SecondHalf);
    EvolveF(fine_lev, PatchType::fine, 0.5_rt*dt[fine_lev], DtType::SecondHalf);

    if (do_pml) {
        FillBoundaryF(fine_lev, PatchType::fine, guard_cells.ng_alloc_F);
        DampPML(fine_lev, PatchType::fine);
        FillBoundaryE(fine_lev, PatchType::fine, guard_cells.ng_FieldGather);
    }

    FillBoundaryB(fine_lev, PatchType::fine, guard_cells.ng_FieldGather);

    // ii) Push particles on the coarse patch and mother grid.
    // Push the fields on the coarse patch and mother grid
    // by only half a coarse step (first half)
    PushParticlesandDeposit(coarse_lev, cur_time, DtType::Full);
    StoreCurrent(coarse_lev);
    AddCurrentFromFineLevelandSumBoundary(current_fp, current_cp, current_buf, coarse_lev);
    AddRhoFromFineLevelandSumBoundary(rho_fp, rho_cp, charge_buf, coarse_lev, 0, ncomps);

    EvolveB(fine_lev, PatchType::coarse, dt[fine_lev], DtType::FirstHalf);
    EvolveF(fine_lev, PatchType::coarse, dt[fine_lev], DtType::FirstHalf);
    FillBoundaryB(fine_lev, PatchType::coarse, guard_cells.ng_FieldGather);
    FillBoundaryF(fine_lev, PatchType::coarse, guard_cells.ng_FieldSolverF);

    EvolveE(fine_lev, PatchType::coarse, dt[fine_lev]);
    FillBoundaryE(fine_lev, PatchType::coarse, guard_cells.ng_FieldGather);

    EvolveB(coarse_lev, PatchType::fine, 0.5_rt*dt[coarse_lev], DtType::FirstHalf);
    EvolveF(coarse_lev, PatchType::fine, 0.5_rt*dt[coarse_lev], DtType::FirstHalf);
    FillBoundaryB(coarse_lev, PatchType::fine, guard_cells.ng_FieldGather,
                    WarpX::sync_nodal_points);
    FillBoundaryF(coarse_lev, PatchType::fine, guard_cells.ng_FieldSolverF,
                    WarpX::sync_nodal_points);

    EvolveE(coarse_lev, PatchType::fine, 0.5_rt*dt[coarse_lev]);
    FillBoundaryE(coarse_lev, PatchType::fine, guard_cells.ng_FieldGather);

    // TODO Remove call to FillBoundaryAux before UpdateAuxilaryData?
    FillBoundaryAux(guard_cells.ng_UpdateAux);
    // iii) Get auxiliary fields on the fine grid, at dt[fine_lev]
    UpdateAuxilaryData();
    FillBoundaryAux(guard_cells.ng_UpdateAux);

    // iv) Push particles and fields on the fine patch (second fine step)
    PushParticlesandDeposit(fine_lev, cur_time + dt[fine_lev], DtType::SecondHalf);
    RestrictCurrentFromFineToCoarsePatch(current_fp, current_cp, fine_lev);
    RestrictRhoFromFineToCoarsePatch(rho_fp, rho_cp, fine_lev);
    if (use_filter) { ApplyFilterJ(current_fp, fine_lev); }
    SumBoundaryJ(current_fp, fine_lev, Geom(fine_lev).periodicity());
    ApplyFilterandSumBoundaryRho(rho_fp, rho_cp, fine_lev, PatchType::fine, 0, ncomps);

    EvolveB(fine_lev, PatchType::fine, 0.5_rt*dt[fine_lev], DtType::FirstHalf);
    EvolveF(fine_lev, PatchType::fine, 0.5_rt*dt[fine_lev], DtType::FirstHalf);
    FillBoundaryB(fine_lev, PatchType::fine, guard_cells.ng_FieldSolver);
    FillBoundaryF(fine_lev, PatchType::fine, guard_cells.ng_FieldSolverF);

    EvolveE(fine_lev, PatchType::fine, dt[fine_lev]);
    FillBoundaryE(fine_lev, PatchType::fine, guard_cells.ng_FieldSolver,
                    WarpX::sync_nodal_points);

    EvolveB(fine_lev, PatchType::fine, 0.5_rt*dt[fine_lev], DtType::SecondHalf);
    EvolveF(fine_lev, PatchType::fine, 0.5_rt*dt[fine_lev], DtType::SecondHalf);

    if (do_pml) {
        DampPML(fine_lev, PatchType::fine);
        FillBoundaryE(fine_lev, PatchType::fine, guard_cells.ng_FieldSolver);
    }

    if ( safe_guard_cells ) {
        FillBoundaryF(fine_lev, PatchType::fine, guard_cells.ng_FieldSolver);
    }
    FillBoundaryB(fine_lev, PatchType::fine, guard_cells.ng_FieldSolver);

    // v) Push the fields on the coarse patch and mother grid
    // by only half a coarse step (second half)
    RestoreCurrent(coarse_lev);
    AddCurrentFromFineLevelandSumBoundary(current_fp, current_cp, current_buf, coarse_lev);
    AddRhoFromFineLevelandSumBoundary(rho_fp, rho_cp, charge_buf, coarse_lev, ncomps, ncomps);

    EvolveE(fine_lev, PatchType::coarse, dt[fine_lev]);
    FillBoundaryE(fine_lev, PatchType::coarse, guard_cells.ng_FieldSolver,
                  WarpX::sync_nodal_points);

    EvolveB(fine_lev, PatchType::coarse, dt[fine_lev], DtType::SecondHalf);
    EvolveF(fine_lev, PatchType::coarse, dt[fine_lev], DtType::SecondHalf);

    if (do_pml) {
        FillBoundaryF(fine_lev, PatchType::fine, guard_cells.ng_FieldSolverF);
        DampPML(fine_lev, PatchType::coarse); // do it twice
        DampPML(fine_lev, PatchType::coarse);
        FillBoundaryE(fine_lev, PatchType::coarse, guard_cells.ng_alloc_EB);
    }

    FillBoundaryB(fine_lev, PatchType::coarse, guard_cells.ng_FieldSolver,
                  WarpX::sync_nodal_points);
    FillBoundaryF(fine_lev, PatchType::coarse, guard_cells.ng_FieldSolverF,
                  WarpX::sync_nodal_points);

    EvolveE(coarse_lev, PatchType::fine, 0.5_rt*dt[coarse_lev]);
    FillBoundaryE(coarse_lev, PatchType::fine, guard_cells.ng_FieldSolver,
                  WarpX::sync_nodal_points);

    EvolveB(coarse_lev, PatchType::fine, 0.5_rt*dt[coarse_lev], DtType::SecondHalf);
    EvolveF(coarse_lev, PatchType::fine, 0.5_rt*dt[coarse_lev], DtType::SecondHalf);

    if (do_pml) {
        if (moving_window_active(istep[0]+1)){
            // Exchange guard cells of PMLs only (0 cells are exchanged for the
            // regular B field MultiFab). This is required as B and F have just been
            // evolved.
            FillBoundaryB(coarse_lev, PatchType::fine, IntVect::TheZeroVector(),
                          WarpX::sync_nodal_points);
            FillBoundaryF(coarse_lev, PatchType::fine, IntVect::TheZeroVector(),
                          WarpX::sync_nodal_points);
        }
        DampPML(coarse_lev, PatchType::fine);
        if ( safe_guard_cells ) {
            FillBoundaryE(coarse_lev, PatchType::fine, guard_cells.ng_FieldSolver,
                          WarpX::sync_nodal_points);
        }
    }
    if ( safe_guard_cells ) {
        FillBoundaryB(coarse_lev, PatchType::fine, guard_cells.ng_FieldSolver,
                      WarpX::sync_nodal_points);
    }
}

void
WarpX::doFieldIonization ()
{
    for (int lev = 0; lev <= finest_level; ++lev) {
        doFieldIonization(lev);
    }
}

void
WarpX::doFieldIonization (int lev)
{
    mypc->doFieldIonization(lev,
                            *Efield_aux[lev][0],*Efield_aux[lev][1],*Efield_aux[lev][2],
                            *Bfield_aux[lev][0],*Bfield_aux[lev][1],*Bfield_aux[lev][2]);
}

#ifdef WARPX_QED
void
WarpX::doQEDEvents ()
{
    for (int lev = 0; lev <= finest_level; ++lev) {
        doQEDEvents(lev);
    }
}

void
WarpX::doQEDEvents (int lev)
{
    mypc->doQedEvents(lev,
                      *Efield_aux[lev][0],*Efield_aux[lev][1],*Efield_aux[lev][2],
                      *Bfield_aux[lev][0],*Bfield_aux[lev][1],*Bfield_aux[lev][2]);
}
#endif

void
WarpX::PushParticlesandDeposit (amrex::Real cur_time, bool skip_current, PushType push_type)
{
    // Evolve particles to p^{n+1/2} and x^{n+1}
    // Deposit current, j^{n+1/2}
    for (int lev = 0; lev <= finest_level; ++lev) {
        PushParticlesandDeposit(lev, cur_time, DtType::Full, skip_current, push_type);
    }
}

void
WarpX::PushParticlesandDeposit (int lev, amrex::Real cur_time, DtType a_dt_type, bool skip_current,
                               PushType push_type)
{
    amrex::MultiFab* current_x = nullptr;
    amrex::MultiFab* current_y = nullptr;
    amrex::MultiFab* current_z = nullptr;

    if (WarpX::do_current_centering)
    {
        current_x = current_fp_nodal[lev][0].get();
        current_y = current_fp_nodal[lev][1].get();
        current_z = current_fp_nodal[lev][2].get();
    }
    else if (WarpX::current_deposition_algo == CurrentDepositionAlgo::Vay)
    {
        // Note that Vay deposition is supported only for PSATD and the code currently aborts otherwise
        current_x = current_fp_vay[lev][0].get();
        current_y = current_fp_vay[lev][1].get();
        current_z = current_fp_vay[lev][2].get();
    }
    else
    {
        current_x = current_fp[lev][0].get();
        current_y = current_fp[lev][1].get();
        current_z = current_fp[lev][2].get();
    }

    mypc->Evolve(lev,
                 *Efield_aux[lev][0], *Efield_aux[lev][1], *Efield_aux[lev][2],
                 *Bfield_aux[lev][0], *Bfield_aux[lev][1], *Bfield_aux[lev][2],
                 *current_x, *current_y, *current_z,
                 current_buf[lev][0].get(), current_buf[lev][1].get(), current_buf[lev][2].get(),
                 rho_fp[lev].get(), charge_buf[lev].get(),
                 Efield_cax[lev][0].get(), Efield_cax[lev][1].get(), Efield_cax[lev][2].get(),
                 Bfield_cax[lev][0].get(), Bfield_cax[lev][1].get(), Bfield_cax[lev][2].get(),
                 cur_time, dt[lev], a_dt_type, skip_current, push_type);
    if (! skip_current) {
#ifdef WARPX_DIM_RZ
        // This is called after all particles have deposited their current and charge.
        ApplyInverseVolumeScalingToCurrentDensity(current_fp[lev][0].get(), current_fp[lev][1].get(), current_fp[lev][2].get(), lev);
        if (current_buf[lev][0].get()) {
            ApplyInverseVolumeScalingToCurrentDensity(current_buf[lev][0].get(), current_buf[lev][1].get(), current_buf[lev][2].get(), lev-1);
        }
        if (rho_fp[lev]) {
            ApplyInverseVolumeScalingToChargeDensity(rho_fp[lev].get(), lev);
            if (charge_buf[lev]) {
                ApplyInverseVolumeScalingToChargeDensity(charge_buf[lev].get(), lev-1);
            }
        }
// #else
        // I left this comment here as a reminder that currently the
        // boundary handling for cartesian grids are not matching the RZ handling
        // (done in the ApplyInverseScalingToChargeDensity function). The
        // Cartesian grid code had to be moved from here to after the application
        // of the filter to avoid incorrect results (moved to `SyncCurrentAndRho()`).
        // Might this be related to issue #1943?
#endif
        if (do_fluid_species) {
            myfl->Evolve(lev,
                         *Efield_aux[lev][0], *Efield_aux[lev][1], *Efield_aux[lev][2],
                         *Bfield_aux[lev][0], *Bfield_aux[lev][1], *Bfield_aux[lev][2],
                         rho_fp[lev].get(), *current_x, *current_y, *current_z, cur_time, skip_current);
        }
    }
}

/* \brief Apply perfect mirror condition inside the box (not at a boundary).
 * In practice, set all fields to 0 on a section of the simulation domain
 * (as for a perfect conductor with a given thickness).
 * The mirror normal direction has to be parallel to the z axis.
 */
void
WarpX::applyMirrors (Real time)
{
    // something to do?
    if (num_mirrors == 0) {
        return;
    }

    // Loop over the mirrors
    for(int i_mirror=0; i_mirror<num_mirrors; ++i_mirror)
    {
        // Get mirror properties (lower and upper z bounds)
        amrex::Real z_min = mirror_z[i_mirror];
        amrex::Real z_max_tmp = z_min + mirror_z_width[i_mirror];

        // Boost quantities for boosted frame simulations
        if (gamma_boost>1)
        {
            z_min = z_min/gamma_boost - PhysConst::c*beta_boost*time;
            z_max_tmp = z_max_tmp/gamma_boost - PhysConst::c*beta_boost*time;
        }

        // Loop over levels
        for(int lev=0; lev<=finest_level; lev++)
        {
            // Mirror must contain at least mirror_z_npoints[i_mirror] cells
            const amrex::Real dz = WarpX::CellSize(lev)[2];
            const amrex::Real z_max = std::max(z_max_tmp, z_min+mirror_z_npoints[i_mirror]*dz);

            // Get fine patch field MultiFabs
            amrex::MultiFab& Ex = *Efield_fp[lev][0].get();
            amrex::MultiFab& Ey = *Efield_fp[lev][1].get();
            amrex::MultiFab& Ez = *Efield_fp[lev][2].get();
            amrex::MultiFab& Bx = *Bfield_fp[lev][0].get();
            amrex::MultiFab& By = *Bfield_fp[lev][1].get();
            amrex::MultiFab& Bz = *Bfield_fp[lev][2].get();

            // Set each field to zero between z_min and z_max
            NullifyMF(Ex, lev, z_min, z_max);
            NullifyMF(Ey, lev, z_min, z_max);
            NullifyMF(Ez, lev, z_min, z_max);
            NullifyMF(Bx, lev, z_min, z_max);
            NullifyMF(By, lev, z_min, z_max);
            NullifyMF(Bz, lev, z_min, z_max);

            // If div(E)/div(B) cleaning are used, set F/G field to zero
            if (F_fp[lev]) { NullifyMF(*F_fp[lev], lev, z_min, z_max); }
            if (G_fp[lev]) { NullifyMF(*G_fp[lev], lev, z_min, z_max); }

            if (lev>0)
            {
                // Get coarse patch field MultiFabs
                amrex::MultiFab& cEx = *Efield_cp[lev][0].get();
                amrex::MultiFab& cEy = *Efield_cp[lev][1].get();
                amrex::MultiFab& cEz = *Efield_cp[lev][2].get();
                amrex::MultiFab& cBx = *Bfield_cp[lev][0].get();
                amrex::MultiFab& cBy = *Bfield_cp[lev][1].get();
                amrex::MultiFab& cBz = *Bfield_cp[lev][2].get();

                // Set each field to zero between z_min and z_max
                NullifyMF(cEx, lev, z_min, z_max);
                NullifyMF(cEy, lev, z_min, z_max);
                NullifyMF(cEz, lev, z_min, z_max);
                NullifyMF(cBx, lev, z_min, z_max);
                NullifyMF(cBy, lev, z_min, z_max);
                NullifyMF(cBz, lev, z_min, z_max);

                // If div(E)/div(B) cleaning are used, set F/G field to zero
                if (F_cp[lev]) { NullifyMF(*F_cp[lev], lev, z_min, z_max); }
                if (G_cp[lev]) { NullifyMF(*G_cp[lev], lev, z_min, z_max); }
            }
        }
    }
}

void
WarpX::HandleSignals()
{
    SignalHandling::WaitSignals();

    // SIGNAL_REQUESTS_BREAK is handled directly in WarpX::Evolve

    if (SignalHandling::TestAndResetActionRequestFlag(SignalHandling::SIGNAL_REQUESTS_CHECKPOINT)) {
        multi_diags->FilterComputePackFlushLastTimestep( istep[0] );
        ExecutePythonCallback("oncheckpointsignal");
    }
}
