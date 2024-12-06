/* Copyright 2024 Justin Angus
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Fields.H"
#include "ThetaImplicitEM.H"
#include "WarpX.H"

using warpx::fields::FieldType;
using namespace amrex::literals;

void ThetaImplicitEM::Define ( WarpX* const  a_WarpX )
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_is_defined,
        "ThetaImplicitEM object is already defined!");

    // Retain a pointer back to main WarpX class
    m_WarpX = a_WarpX;
    m_num_amr_levels = 1;

    // Define E and Eold vectors
    m_E.Define( m_WarpX, "Efield_fp" );
    m_Eold.Define( m_E );

    // Define B_old MultiFabs
    using ablastr::fields::Direction;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const auto& ba_Bx = m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, lev)->boxArray();
        const auto& ba_By = m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{1}, lev)->boxArray();
        const auto& ba_Bz = m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{2}, lev)->boxArray();
        const auto& dm = m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, lev)->DistributionMap();
        const amrex::IntVect ngb = m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, lev)->nGrowVect();
        m_WarpX->m_fields.alloc_init(FieldType::B_old, Direction{0}, lev, ba_Bx, dm, 1, ngb, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::B_old, Direction{1}, lev, ba_By, dm, 1, ngb, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::B_old, Direction{2}, lev, ba_Bz, dm, 1, ngb, 0.0_rt);
    }

    // Parse theta-implicit solver specific parameters
    const amrex::ParmParse pp("implicit_evolve");
    pp.query("theta", m_theta);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_theta>=0.5 && m_theta<=1.0,
        "theta parameter for theta implicit time solver must be between 0.5 and 1.0");

    // Parse nonlinear solver parameters
    parseNonlinearSolverParams( pp );

    // Define the nonlinear solver
    m_nlsolver->Define(m_E, this);
    m_is_defined = true;

}

void ThetaImplicitEM::PrintParameters () const
{
    if (!m_WarpX->Verbose()) { return; }
    amrex::Print() << "\n";
    amrex::Print() << "-----------------------------------------------------------\n";
    amrex::Print() << "----------- THETA IMPLICIT EM SOLVER PARAMETERS -----------\n";
    amrex::Print() << "-----------------------------------------------------------\n";
    amrex::Print() << "Time-bias parameter theta:  " << m_theta << "\n";
    amrex::Print() << "max particle iterations:    " << m_max_particle_iterations << "\n";
    amrex::Print() << "particle tolerance:         " << m_particle_tolerance << "\n";
    if (m_nlsolver_type==NonlinearSolverType::Picard) {
        amrex::Print() << "Nonlinear solver type:      Picard\n";
    }
    else if (m_nlsolver_type==NonlinearSolverType::Newton) {
        amrex::Print() << "Nonlinear solver type:      Newton\n";
    }
    m_nlsolver->PrintParams();
    amrex::Print() << "-----------------------------------------------------------\n\n";
}

void ThetaImplicitEM::OneStep ( const amrex::Real  start_time,
                                const amrex::Real  a_dt,
                                const int          a_step )
{
    amrex::ignore_unused(a_step);

    // Fields have Eg^{n} and Bg^{n}
    // Particles have up^{n} and xp^{n}.

    // Set the member time step
    m_dt = a_dt;

    // Save up and xp at the start of the time step
    m_WarpX->SaveParticlesAtImplicitStepStart ( );

    // Save Eg at the start of the time step
    m_Eold.Copy( FieldType::Efield_fp );

    const int num_levels = 1;
    for (int lev = 0; lev < num_levels; ++lev) {
        const ablastr::fields::VectorField Bfp = m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        ablastr::fields::VectorField B_old = m_WarpX->m_fields.get_alldirs(FieldType::B_old, lev);
        for (int n = 0; n < 3; ++n) {
            amrex::MultiFab::Copy(*B_old[n], *Bfp[n], 0, 0, B_old[n]->nComp(),
                                  B_old[n]->nGrowVect() );
        }
    }

    // Solve nonlinear system for Eg at t_{n+theta}
    // Particles will be advanced to t_{n+1/2}
    m_E.Copy(m_Eold); // initial guess for Eg^{n+theta}
    m_nlsolver->Solve( m_E, m_Eold, start_time, m_theta*m_dt );

    // Update WarpX owned Efield_fp and Bfield_fp to t_{n+theta}
    UpdateWarpXFields( m_E, start_time );

    // Advance particles from time n+1/2 to time n+1
    m_WarpX->FinishImplicitParticleUpdate();

    // Advance Eg and Bg from time n+theta to time n+1
    const amrex::Real end_time = start_time + m_dt;
    FinishFieldUpdate( end_time );

}

void ThetaImplicitEM::ComputeRHS ( WarpXSolverVec&  a_RHS,
                             const WarpXSolverVec&  a_E,
                                   amrex::Real      start_time,
                                   int              a_nl_iter,
                                   bool             a_from_jacobian )
{
    // Update WarpX-owned Efield_fp and Bfield_fp using current state of
    // Eg from the nonlinear solver at time n+theta
    UpdateWarpXFields( a_E, start_time );

    // Update particle positions and velocities using the current state
    // of Eg and Bg. Deposit current density at time n+1/2
    const amrex::Real theta_time = start_time + m_theta*m_dt;
    m_WarpX->ImplicitPreRHSOp( theta_time, m_dt, a_nl_iter, a_from_jacobian );

    // RHS = cvac^2*m_theta*dt*( curl(Bg^{n+theta}) - mu0*Jg^{n+1/2} )
    m_WarpX->ImplicitComputeRHSE( m_theta*m_dt, a_RHS);
}

void ThetaImplicitEM::UpdateWarpXFields ( const WarpXSolverVec&  a_E,
                                          amrex::Real start_time )
{

    // Update Efield_fp owned by WarpX
    const amrex::Real theta_time = start_time + m_theta*m_dt;
    m_WarpX->SetElectricFieldAndApplyBCs( a_E, theta_time );

    // Update Bfield_fp owned by WarpX
    ablastr::fields::MultiLevelVectorField const& B_old = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, 0);
    m_WarpX->UpdateMagneticFieldAndApplyBCs( B_old, m_theta*m_dt, start_time );

}

void ThetaImplicitEM::FinishFieldUpdate ( amrex::Real end_time )
{

    // Eg^{n+1} = (1/theta)*Eg^{n+theta} + (1-1/theta)*Eg^n
    // Bg^{n+1} = (1/theta)*Bg^{n+theta} + (1-1/theta)*Bg^n

    const amrex::Real c0 = 1._rt/m_theta;
    const amrex::Real c1 = 1._rt - c0;
    m_E.linComb( c0, m_E, c1, m_Eold );
    m_WarpX->SetElectricFieldAndApplyBCs( m_E, end_time );
    ablastr::fields::MultiLevelVectorField const & B_old = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, 0);
    m_WarpX->FinishMagneticFieldAndApplyBCs( B_old, m_theta, end_time );

}
