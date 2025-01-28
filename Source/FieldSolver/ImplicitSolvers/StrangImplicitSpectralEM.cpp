/* Copyright 2024 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Fields.H"
#include "StrangImplicitSpectralEM.H"
#include "WarpX.H"

using namespace warpx::fields;
using namespace amrex::literals;

void StrangImplicitSpectralEM::Define ( WarpX* const a_WarpX )
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_is_defined,
        "StrangImplicitSpectralEM object is already defined!");

    // Retain a pointer back to main WarpX class
    m_WarpX = a_WarpX;

    // Define E and Eold vectors
    m_E.Define( m_WarpX, "Efield_fp" );
    m_Eold.Define( m_E );


    // Parse nonlinear solver parameters
    const amrex::ParmParse pp_implicit_evolve("implicit_evolve");
    parseNonlinearSolverParams( pp_implicit_evolve );

    // Define the nonlinear solver
    m_nlsolver->Define(m_E, this);
    m_is_defined = true;

}

void StrangImplicitSpectralEM::PrintParameters () const
{
    if (!m_WarpX->Verbose()) { return; }
    amrex::Print() << "\n";
    amrex::Print() << "------------------------------------------------------------------------" << "\n";
    amrex::Print() << "----------- STRANG SPLIT IMPLICIT SPECTRAL EM SOLVER PARAMETERS --------" << "\n";
    amrex::Print() << "------------------------------------------------------------------------" << "\n";
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

void StrangImplicitSpectralEM::OneStep ( amrex::Real start_time,
                                         amrex::Real a_dt,
                                         int a_step )
{
    amrex::ignore_unused(a_step);

    // Fields have E^{n} and B^{n}
    // Particles have p^{n} and x^{n}.

    // Set the member time step
    m_dt = a_dt;

    // Save the values at the start of the time step,
    m_WarpX->SaveParticlesAtImplicitStepStart();

    // Advance the fields to time n+1/2 source free
    m_WarpX->SpectralSourceFreeFieldAdvance(start_time);

    // Save the fields at the start of the step
    m_Eold.Copy( FieldType::Efield_fp );
    m_E.Copy(m_Eold); // initial guess for E

    amrex::Real const half_time = start_time + 0.5_rt*m_dt;

    // Solve nonlinear system for E at t_{n+1/2}
    // Particles will be advanced to t_{n+1/2}
    m_nlsolver->Solve( m_E, m_Eold, half_time, 0.5_rt*m_dt );

    // Update WarpX owned Efield_fp and Bfield_fp to t_{n+1/2}
    UpdateWarpXFields( m_E, half_time );

    // Advance particles from time n+1/2 to time n+1
    m_WarpX->FinishImplicitParticleUpdate();

    // Advance E and B fields from time n+1/2 to time n+1
    amrex::Real const new_time = start_time + m_dt;
    FinishFieldUpdate( new_time );

    // Advance the fields to time n+1 source free
    m_WarpX->SpectralSourceFreeFieldAdvance(half_time);

}

void StrangImplicitSpectralEM::ComputeRHS ( WarpXSolverVec& a_RHS,
                                            WarpXSolverVec const & a_E,
                                            amrex::Real half_time,
                                            int a_nl_iter,
                                            bool a_from_jacobian )
{
    // Update WarpX-owned Efield_fp and Bfield_fp using current state of
    // E from the nonlinear solver at time n+1/2
    UpdateWarpXFields( a_E, half_time );

    // Self consistently update particle positions and velocities using the
    // current state of the fields E and B. Deposit current density at time n+1/2.
    m_WarpX->ImplicitPreRHSOp( half_time, m_dt, a_nl_iter, a_from_jacobian );

    // For Strang split implicit PSATD, the RHS = -dt*mu*c**2*J
    bool const allow_type_mismatch = true;
    a_RHS.Copy(FieldType::current_fp, warpx::fields::FieldType::None, allow_type_mismatch);
    amrex::Real constexpr coeff = PhysConst::c * PhysConst::c * PhysConst::mu0;
    a_RHS.scale(-coeff * 0.5_rt*m_dt);

}

void StrangImplicitSpectralEM::UpdateWarpXFields (WarpXSolverVec const & a_E,
                                                  amrex::Real half_time )
{

    // Update Efield_fp owned by WarpX
    m_WarpX->SetElectricFieldAndApplyBCs( a_E, half_time );

}

void StrangImplicitSpectralEM::FinishFieldUpdate ( amrex::Real a_new_time )
{
    // Eg^{n+1} = 2*E_g^{n+1/2} - E_g^n
    amrex::Real const c0 = 1._rt/0.5_rt;
    amrex::Real const c1 = 1._rt - c0;
    m_E.linComb( c0, m_E, c1, m_Eold );
    m_WarpX->SetElectricFieldAndApplyBCs( m_E, a_new_time );

}
