/* Copyright 2024 Justin Angus
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SemiImplicitEM.H"
#include "WarpX.H"

using warpx::fields::FieldType;
using namespace amrex::literals;

void SemiImplicitEM::Define ( WarpX*  a_WarpX )
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_is_defined,
        "SemiImplicitEM object is already defined!");

    // Retain a pointer back to main WarpX class
    m_WarpX = a_WarpX;

    // Define E and Eold vectors
    m_E.Define( m_WarpX, "Efield_fp" );
    m_Eold.Define( m_E );

    // Parse implicit solver parameters
    const amrex::ParmParse pp("implicit_evolve");
    parseNonlinearSolverParams( pp );

    // Define the nonlinear solver
    m_nlsolver->Define(m_E, this);
    m_is_defined = true;

}

void SemiImplicitEM::PrintParameters () const
{
    if (!m_WarpX->Verbose()) { return; }
    amrex::Print() << "\n";
    amrex::Print() << "-----------------------------------------------------------\n";
    amrex::Print() << "----------- SEMI IMPLICIT EM SOLVER PARAMETERS ------------\n";
    amrex::Print() << "-----------------------------------------------------------\n";
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

void SemiImplicitEM::OneStep ( amrex::Real  start_time,
                               amrex::Real  a_dt,
                               int          a_step )
{
    amrex::ignore_unused(a_step);

    // Set the member time step
    m_dt = a_dt;

    // Fields have Eg^{n}, Bg^{n}
    // Particles have up^{n} and xp^{n}.

    // Save up and xp at the start of the time step
    m_WarpX->SaveParticlesAtImplicitStepStart ( );

    // Save Eg at the start of the time step
    m_Eold.Copy( FieldType::Efield_fp );

    // Advance WarpX owned Bfield_fp from t_{n} to t_{n+1/2}
    m_WarpX->EvolveB(0.5_rt*m_dt, DtType::FirstHalf, start_time);
    m_WarpX->FillBoundaryB(m_WarpX->getngEB(), true);

    const amrex::Real half_time = start_time + 0.5_rt*m_dt;

    // Solve nonlinear system for Eg at t_{n+1/2}
    // Particles will be advanced to t_{n+1/2}
    m_E.Copy(m_Eold); // initial guess for Eg^{n+1/2}
    m_nlsolver->Solve( m_E, m_Eold, half_time, 0.5_rt*m_dt );

    // Update WarpX owned Efield_fp to t_{n+1/2}
    m_WarpX->SetElectricFieldAndApplyBCs( m_E, half_time );

    // Advance particles from time n+1/2 to time n+1
    m_WarpX->FinishImplicitParticleUpdate();

    // Advance Eg from time n+1/2 to time n+1
    // Eg^{n+1} = 2.0*Eg^{n+1/2} - Eg^n
    m_E.linComb( 2._rt, m_E, -1._rt, m_Eold );
    const amrex::Real new_time = start_time + m_dt;
    m_WarpX->SetElectricFieldAndApplyBCs( m_E, new_time );

    // Advance WarpX owned Bfield_fp from t_{n+1/2} to t_{n+1}
    m_WarpX->EvolveB(0.5_rt*m_dt, DtType::SecondHalf, half_time);
    m_WarpX->FillBoundaryB(m_WarpX->getngEB(), true);

}

void SemiImplicitEM::ComputeRHS ( WarpXSolverVec&  a_RHS,
                            const WarpXSolverVec&  a_E,
                                  amrex::Real      half_time,
                                  int              a_nl_iter,
                                  bool             a_from_jacobian )
{
    // Update WarpX-owned Efield_fp using current state of Eg from
    // the nonlinear solver at time n+theta
    m_WarpX->SetElectricFieldAndApplyBCs( a_E, half_time );

    // Update particle positions and velocities using the current state
    // of Eg and Bg. Deposit current density at time n+1/2
    m_WarpX->ImplicitPreRHSOp( half_time, m_dt, a_nl_iter, a_from_jacobian );

    // RHS = cvac^2*0.5*dt*( curl(Bg^{n+1/2}) - mu0*Jg^{n+1/2} )
    m_WarpX->ImplicitComputeRHSE(0.5_rt*m_dt, a_RHS);
}
