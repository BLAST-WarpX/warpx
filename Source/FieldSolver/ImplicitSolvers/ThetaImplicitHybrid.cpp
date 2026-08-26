/* Copyright 2026 Prabhat Kumar
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Fields.H"
#include "ThetaImplicitHybrid.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "WarpX.H"

using warpx::fields::FieldType;
using namespace amrex::literals;

void ThetaImplicitHybrid::Define (WarpX* const a_WarpX, bool /*from_restart*/)
{
    BL_PROFILE("ThetaImplicitHybrid::Define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_is_defined,
        "ThetaImplicitHybrid object is already defined!");

    // Retain a pointer back to main WarpX class
    m_WarpX = a_WarpX;
    m_num_amr_levels = 1;

    m_hybrid_pic_model = m_WarpX->get_pointer_HybridPICModel();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_pic_model != nullptr,
        "ThetaImplicitHybrid solver requires the hybrid-PIC model to be defined");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_hybrid_pic_model->m_solve_electron_energy_equation,
        "hybrid_pic_model.solve_electron_energy_equation is not yet supported with "
        "algo.evolve_scheme = theta_implicit_hybrid");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_hybrid_pic_model->m_add_external_fields,
        "External fields from vector potentials are not yet supported with "
        "algo.evolve_scheme = theta_implicit_hybrid");

    // Define E and Eold vectors
    m_E.Define( m_WarpX, "Efield_fp" );
    m_Eold.Define( m_E );

    // Define B_old MultiFabs
    using ablastr::fields::Direction;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const auto& dm = m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, lev)->DistributionMap();
        const amrex::IntVect ngb = m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, lev)->nGrowVect();
        for (int dir = 0; dir < 3; ++dir) {
            const auto& ba = m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{dir}, lev)->boxArray();
            m_WarpX->m_fields.alloc_init(FieldType::B_old, Direction{dir}, lev, ba, dm, 1, ngb, 0.0_rt);
        }
    }

    // Parse theta-implicit solver specific parameters
    const amrex::ParmParse pp("implicit_evolve");
    pp.query("theta", m_theta);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_theta >= 0.5 && m_theta <= 1.0,
        "theta parameter for theta implicit time solver must be between 0.5 and 1.0");

    // Parse nonlinear solver parameters
    parseNonlinearSolverParams( pp );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_use_mass_matrices,
        "Mass matrices are not yet supported with algo.evolve_scheme = theta_implicit_hybrid");

    // Define the nonlinear solver
    m_nlsolver->Define(m_E, this);

    m_is_defined = true;
}

void ThetaImplicitHybrid::PrintParameters () const
{
    BL_PROFILE("ThetaImplicitHybrid::PrintParameters()");

    if (!m_WarpX->Verbose()) { return; }
    amrex::Print() << "\n";
    amrex::Print() << "-----------------------------------------------------------\n";
    amrex::Print() << "-------- THETA IMPLICIT HYBRID PIC SOLVER PARAMETERS ------\n";
    amrex::Print() << "-----------------------------------------------------------\n";
    amrex::Print() << "Time-bias parameter theta:           " << m_theta << "\n";
    PrintBaseImplicitSolverParameters();
    m_nlsolver->PrintParams();
    amrex::Print() << "-----------------------------------------------------------\n\n";
}

int ThetaImplicitHybrid::OneStep ( const amrex::Real  start_time,
                                   const amrex::Real  a_dt,
                                   const int          a_step )
{
    BL_PROFILE("ThetaImplicitHybrid::OneStep()");

    // Fields have Eg^{n} and Bg^{n}
    // Particles have vp^{n} and xp^{n}.

    // Set the member time step
    m_dt = a_dt;

    // Save vp and xp at the start of the time step
    m_WarpX->SaveParticlesAtImplicitStepStart();

    // Save Eg at the start of the time step
    m_Eold.Copy(FieldType::Efield_fp);

    // Save Bg at the start of the time step
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const ablastr::fields::VectorField Bfp = m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        ablastr::fields::VectorField B_old = m_WarpX->m_fields.get_alldirs(FieldType::B_old, lev);
        for (int n = 0; n < 3; ++n) {
            amrex::MultiFab::Copy(*B_old[n], *Bfp[n], 0, 0,
                                  B_old[n]->nComp(), B_old[n]->nGrowVect());
        }
    }

    // Initial guess for Eg^{n+theta} is Eg^{n}
    m_E.Copy(m_Eold);

    // Solve the nonlinear system for Eg at t_{n+theta}
    // Particles will be advanced to t_{n+1/2}
    m_nlsolver->Solve( m_E, m_Eold, start_time, m_dt, a_step );

    const int exit_status = m_nlsolver->GetExitStatus();
    if (exit_status < 0) { return exit_status; }

    // Update WarpX owned Efield_fp and Bfield_fp to t_{n+theta}
    UpdateWarpXFields( m_E, start_time );
    m_WarpX->reduced_diags->ComputeDiagsMidStep(a_step);

    // Advance particles from t_{n+1/2} to t_{n+1}
    m_WarpX->FinishImplicitParticleUpdate( start_time + m_dt );

    // Advance Eg and Bg from t_{n+theta} to t_{n+1}
    FinishFieldUpdate( start_time + m_dt );

    return exit_status;
}

void ThetaImplicitHybrid::ComputeRHS ( WarpXSolverVec&        a_RHS,
                                       const WarpXSolverVec&  a_E,
                                       amrex::Real            start_time,
                                       int                    a_nl_iter,
                                       bool                   a_from_jacobian )
{
    BL_PROFILE("ThetaImplicitHybrid::ComputeRHS()");

    // Set Eg^{n+theta} = a_E and update Bg^{n+theta} via Faraday's law
    UpdateWarpXFields( a_E, start_time );

    const amrex::Real theta_time = start_time + m_theta * m_dt;

    ablastr::fields::MultiLevelVectorField Efield_fp =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, m_num_amr_levels - 1);
    ablastr::fields::MultiLevelVectorField Bfield_fp =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1);
    ablastr::fields::MultiLevelVectorField current_fp =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, m_num_amr_levels - 1);
    ablastr::fields::MultiLevelScalarField rho_fp =
        m_WarpX->m_fields.get_mr_levels(FieldType::rho_fp, m_num_amr_levels - 1);

    // Total (plasma) current from Ampere's law, mu0*J = curl(B^{n+theta})
    m_hybrid_pic_model->CalculatePlasmaCurrent(Bfield_fp, m_WarpX->GetEBUpdateEFlag());

    // The particles are pushed with the Newton iterate itself, minus the
    // dissipative part of Ohm's law: E* = a_E - D, with
    // D = eta*J_p - eta_h*nabla^2(J_p) (E* = a_E for eta = eta_h = 0).
    // Pushing with the iterate gives the residual a true Jacobian through the
    // particle response -- in particular the electrostatic limit (B = 0) is
    // degenerate with any recomputed push field, which would not depend on
    // the solver variable at all.
    SubtractDissipativeEFromPushField();

    m_WarpX->ApplyFillBoundaryE();

    // Advance the particles with E* and Bg^{n+theta} and deposit Ji^{n+1/2}
    // and rho from the time-centered particle state
    PreRHSOp( theta_time, a_nl_iter, a_from_jacobian );

    {
        // Make the Ohm's-law rho a pure function of the Newton iterate: component 0
        // (deposited at the entry positions) carries the previous evaluation's particle
        // state, which pollutes the finite-difference Jacobian matvec at O(hysteresis/eps)
        // in problems where E is deposition-noise dominated (e.g. the electrostatic
        // limit). The post-push component is deposited fresh from x^n each evaluation.
        // At the nonlinear fixed point the two components coincide.
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            amrex::MultiFab* rho = m_WarpX->m_fields.get(FieldType::rho_fp, lev);
            const int nc = rho->nComp()/2;
            const int c_new = rho->nComp() - nc;
            // NOTE: not MultiFab::Copy(*rho, *rho, ...) -- self-copy is UB in AMReX
            for (amrex::MFIter mfi(*rho); mfi.isValid(); ++mfi) {
                const amrex::Box bx = mfi.growntilebox();
                auto const& a = rho->array(mfi);
                amrex::ParallelFor(bx, nc, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                    a(i,j,k,n) = a(i,j,k,c_new + n);
                });
            }
            // The deposit+SumBoundary leaves the rho ghosts stale; Ohm's law reads
            // rho through ghosts at box faces, so refresh them here.
            rho->FillBoundary(m_WarpX->Geom(lev).periodicity());
        }
    }

    // Electron pressure from the algebraic gamma-law closure, re-evaluated
    // from the iterate's rho
    m_hybrid_pic_model->CalculateElectronPressure();

    // Full Ohm's-law E for the Faraday update and the residual. grad(Pe) is
    // curl-free so it does not affect the Faraday update, but it is retained
    // (solve_for_implicit = true) for a self-consistent Newton residual.
    m_hybrid_pic_model->HybridPICSolveE(
        Efield_fp, current_fp, Bfield_fp, rho_fp,
        m_WarpX->GetEBUpdateEFlag(),
        true, true
    );

    m_WarpX->ApplyFillBoundaryE();

    // The nonlinear problem is posed in the fixed-point form
    // E^{n+theta} = E^n + RHS, so RHS = E_Ohm - E^n
    a_RHS.Copy(FieldType::Efield_fp);
    a_RHS.linComb(1.0, a_RHS, -1.0, m_Eold);
}

void ThetaImplicitHybrid::SubtractDissipativeEFromPushField ()
{
    // D = E_Ohm(with dissipation) - E_Ohm(without) evaluated from the same
    // (B^{n+theta}, rho, pe) state: the ideal, Hall and grad-pe parts cancel
    // exactly, leaving eta*J_p - eta_h*nabla^2(J_p) with the identical stencils,
    // interpolations and floors as the residual's Ohm solve -- by construction,
    // for any resistivity model. The FD-solver-level entry is used so no
    // boundary condition is applied to Efield_fp as a side effect.
    if (m_hybrid_pic_model->m_eta_expression == "0.0" &&
        !m_hybrid_pic_model->m_include_hyper_resistivity_term) { return; }

    const int lev = 0;
    const ablastr::fields::VectorField E =
        m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev);
    const ablastr::fields::VectorField Ji =
        m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev);
    const amrex::MultiFab* rho = m_WarpX->m_fields.get(FieldType::rho_fp, lev);
    const amrex::MultiFab* pe =
        m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    const ablastr::fields::VectorField B =
        m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);

    for (int n = 0; n < 3; ++n) {
        if (!m_D[n]) {
            m_D[n] = std::make_unique<amrex::MultiFab>(
                E[n]->boxArray(), E[n]->DistributionMap(), E[n]->nComp(), E[n]->nGrowVect());
            m_E_work[n] = std::make_unique<amrex::MultiFab>(
                E[n]->boxArray(), E[n]->DistributionMap(), E[n]->nComp(), E[n]->nGrowVect());
            // The masked Ohm solves below skip EB-covered locations, which
            // therefore retain their allocation-time content forever: without
            // this initialization, D at covered nodes is uninitialized arena
            // memory that pollutes E* = a_E - D and the Newton residual.
            m_D[n]->setVal(0.0_rt);
            m_E_work[n]->setVal(0.0_rt);
        }
    }
    const ablastr::fields::VectorField D     = {m_D[0].get(), m_D[1].get(), m_D[2].get()};
    const ablastr::fields::VectorField Ework = {m_E_work[0].get(), m_E_work[1].get(), m_E_work[2].get()};

    ablastr::fields::VectorField Jp =
        m_WarpX->m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    auto& eb_update_E = m_WarpX->GetEBUpdateEFlag()[lev];
    auto* fdtd = m_WarpX->get_pointer_fdtd_solver_fp(lev);
    // with dissipation (solve_for_Faraday = true), into D
    fdtd->HybridPICSolveE(D, Jp, Ji, B, *rho, *pe, eb_update_E, lev,
                          m_hybrid_pic_model, true, true);
    // without dissipation, into Ework
    fdtd->HybridPICSolveE(Ework, Jp, Ji, B, *rho, *pe, eb_update_E, lev,
                          m_hybrid_pic_model, false, true);

    for (int n = 0; n < 3; ++n) {
        amrex::MultiFab::Subtract(*m_D[n], *m_E_work[n], 0, 0, m_D[n]->nComp(), 0);
        m_D[n]->FillBoundary(m_WarpX->Geom(lev).periodicity());
        // E* = a_E - D (the particle push reads this)
        amrex::MultiFab::Subtract(*E[n], *m_D[n], 0, 0, E[n]->nComp(), 0);
    }
}

void ThetaImplicitHybrid::UpdateWarpXFields ( const WarpXSolverVec&  a_E,
                                              amrex::Real start_time )
{
    BL_PROFILE("ThetaImplicitHybrid::UpdateWarpXFields()");

    const amrex::Real theta_time = start_time + m_theta * m_dt;

    // Set E^{n+theta} in WarpX
    m_WarpX->SetElectricFieldAndApplyBCs( a_E, theta_time );

    // Compute B^{n+theta} = B^n - theta*dt*curl(E^{n+theta}) via Faraday's law
    ablastr::fields::MultiLevelVectorField const& B_old =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, m_num_amr_levels - 1);
    m_WarpX->UpdateMagneticFieldAndApplyBCs( B_old, m_theta * m_dt, start_time );
}

void ThetaImplicitHybrid::FinishFieldUpdate ( amrex::Real end_time )
{
    BL_PROFILE("ThetaImplicitHybrid::FinishFieldUpdate()");

    // Extrapolate from t^{n+theta} to t^{n+1}:
    // F^{n+1} = (1/theta)*F^{n+theta} + (1 - 1/theta)*F^n
    const amrex::Real c0 = 1.0_rt / m_theta;
    const amrex::Real c1 = 1.0_rt - c0;

    // E^{n+1}
    m_E.linComb( c0, m_E, c1, m_Eold );
    m_WarpX->SetElectricFieldAndApplyBCs( m_E, end_time );

    // B^{n+1}
    ablastr::fields::MultiLevelVectorField const& B_old =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, m_num_amr_levels - 1);
    m_WarpX->FinishMagneticFieldAndApplyBCs( B_old, m_theta, end_time );
}
