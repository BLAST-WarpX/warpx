/* Copyright 2024 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Fields.H"
#include "StrangImplicitSpectralEM.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "WarpX.H"

using namespace warpx::fields;
using namespace amrex::literals;

void StrangImplicitSpectralEM::Define (WarpX* const a_WarpX, bool a_from_restart)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_is_defined,
        "StrangImplicitSpectralEM object is already defined!");

    // Retain a pointer back to main WarpX class
    m_WarpX = a_WarpX;

    // Define E and Eold vectors
    m_E.Define(m_WarpX, "Efield_fp");
    m_Eold.Define(m_E);

    // Set initial values for E and Eold vectors
    m_E.Copy(FieldType::Efield_fp);
    m_Eold.Copy(a_from_restart ? FieldType::E_old : FieldType::Efield_fp, FieldType::None, true);

    // Initialize the midpoint guess by averaging the available fields (E^0 on a fresh start).
    // On restart, E_old and Efield_fp straddle the final source-free half step,
    // so this does not exactly recover the previous implicit midpoint.
    m_E.linComb(1.0_rt - m_theta, m_Eold, m_theta, m_E);

    // Parse nonlinear solver parameters
    const amrex::ParmParse pp_implicit_evolve("implicit_evolve");
    parseNonlinearSolverParams( pp_implicit_evolve );

    // Define the nonlinear solver
    m_nlsolver->Define(m_E, this);

    // Initialize the mass matrices for plasma response
    if (m_use_mass_matrices) { InitializeMassMatrices(); }

    m_is_defined = true;

}

void StrangImplicitSpectralEM::PrintParameters () const
{
    if (!m_WarpX->Verbose()) { return; }
    amrex::Print() << "\n";
    amrex::Print() << "------------------------------------------------------------------------" << "\n";
    amrex::Print() << "----------- STRANG SPLIT IMPLICIT SPECTRAL EM SOLVER PARAMETERS --------" << "\n";
    amrex::Print() << "------------------------------------------------------------------------" << "\n";
    PrintBaseImplicitSolverParameters();
    m_nlsolver->PrintParams();
    amrex::Print() << "-----------------------------------------------------------\n\n";
}

int StrangImplicitSpectralEM::OneStep (amrex::Real start_time,
                                       amrex::Real a_dt,
                                       int a_step,
                                       bool verbose_step)
{
    // Fields have E^{n} and B^{n}
    // Particles have p^{n} and x^{n}.

    // Set the member time step
    m_dt = a_dt;

    // Save the values at the start of the time step,
    m_WarpX->SaveParticlesAtImplicitStepStart();

    // Advance the fields to time n+1/2 source free
    m_WarpX->SpectralSourceFreeFieldAdvance(start_time);

    // Save E at start of implicit substep, after the first source-free half step
    SaveEoldMultifab(); // Copy Efield_fp into E_old
    m_Eold.Copy(FieldType::Efield_fp); // Copy Efield_fp into m_Eold

    amrex::Real const half_time = start_time + 0.5_rt*m_dt;

    // Solve nonlinear system for E at t_{n+1/2}
    // Particles will be advanced to t_{n+1/2}
    m_nlsolver->Solve(m_E, m_Eold, start_time, m_dt, a_step, verbose_step);

    const int exit_status = m_nlsolver->GetExitStatus();
    if (exit_status < 0) { return exit_status; }

    // Copy the converged implicit midpoint E into WarpX-owned Efield_fp
    UpdateWarpXFields(m_E, half_time);
    m_WarpX->reduced_diags->ComputeDiagsMidStep(a_step);

    amrex::Real const end_time = start_time + m_dt;

    // Advance particles from time n+1/2 to time n+1
    FinishImplicitParticleUpdate(end_time, a_step);

    // Finish the implicit E update before the second source-free half step
    FinishFieldUpdate(end_time);

    // Advance the fields to time n+1 source free
    m_WarpX->SpectralSourceFreeFieldAdvance(half_time);

    return exit_status;
}

void StrangImplicitSpectralEM::ComputeRHS ( WarpXSolverVec& a_RHS,
                                            WarpXSolverVec const & a_E,
                                            amrex::Real start_time,
                                            int a_nl_iter,
                                            bool a_from_jacobian )
{
    // Update WarpX-owned Efield_fp and Bfield_fp using current state of
    // E from the nonlinear solver at time n+1/2
    const amrex::Real half_time = start_time + 0.5_rt*m_dt;
    UpdateWarpXFields( a_E, half_time );

    // Self consistently update particle positions and velocities using the
    // current state of the fields E and B. Deposit current density at time n+1/2.
    PreRHSOp( half_time, a_nl_iter, a_from_jacobian );

    // For Strang split implicit PSATD, the RHS = -dt*mu*c**2*J
    bool const allow_type_mismatch = true;
    a_RHS.Copy(FieldType::current_fp, warpx::fields::FieldType::None, allow_type_mismatch);
    amrex::Real constexpr coeff = PhysConst::c2 * PhysConst::mu0;
    a_RHS.scale(-coeff * 0.5_rt*m_dt);

}

void StrangImplicitSpectralEM::UpdateWarpXFields (WarpXSolverVec const & a_E,
                                                  amrex::Real half_time )
{

    // Update Efield_fp owned by WarpX
    m_WarpX->SetElectricFieldAndApplyBCs( a_E, half_time );

}

void StrangImplicitSpectralEM::FinishFieldUpdate (amrex::Real end_time)
{

    // Finish the implicit substep: E_after = 2*E_midpoint - E_before,
    // where E_before is saved in E_old after the first source-free half step.
    // Preserve m_E at the implicit midpoint for the next nonlinear solve.
    // The second source-free half step then advances Efield_fp to E^{n+1}.
    m_WarpX->FinishElectricFieldAndApplyBCs(m_theta, end_time);

}
