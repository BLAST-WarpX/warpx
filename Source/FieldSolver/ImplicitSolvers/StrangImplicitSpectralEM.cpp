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

void StrangImplicitSpectralEM::Define (WarpX* const a_WarpX, bool  a_from_restart)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_is_defined,
        "StrangImplicitSpectralEM object is already defined!");

    // Retain a pointer back to main WarpX class
    m_WarpX = a_WarpX;

    // Define E and Eold vectors
    m_E.Define(m_WarpX, "Efield_fp");
    m_Eold.Define(m_E);

    // Define E_old MultiFab
    using ablastr::fields::Direction;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const auto& ba_Ex = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, lev)->boxArray();
        const auto& ba_Ey = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{1}, lev)->boxArray();
        const auto& ba_Ez = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{2}, lev)->boxArray();
        const auto& dmE = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, lev)->DistributionMap();
        const amrex::IntVect ngE = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, lev)->nGrowVect();
        m_WarpX->m_fields.alloc_init(FieldType::E_old, Direction{0}, lev, ba_Ex, dmE, 1, ngE, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::E_old, Direction{1}, lev, ba_Ey, dmE, 1, ngE, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::E_old, Direction{2}, lev, ba_Ez, dmE, 1, ngE, 0.0_rt);
    }

    m_Eold.Copy(a_from_restart ? FieldType::E_old : FieldType::Efield_fp);

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

    // Initial guess for Eg^{n+theta} is Eg^{n-1+theta}
    // (i.e., Eg used to advance the system from step n-1 to step n)
    m_E.linComb(1.0_rt - m_theta, m_Eold, m_theta, m_E);

    // Save Eg at start of time step
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const ablastr::fields::VectorField Efp = m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev);
        ablastr::fields::VectorField E_old = m_WarpX->m_fields.get_alldirs(FieldType::E_old, lev);
        for (int n = 0; n < 3; ++n) {
            // E_old is needed for diagnostics and saving at checkpoints
            amrex::MultiFab::Copy(*E_old[n], *Efp[n], 0, 0, E_old[n]->nComp(), E_old[n]->nGrowVect());
        }
    }
    m_Eold.Copy(FieldType::E_old);

    amrex::Real const half_time = start_time + 0.5_rt*m_dt;

    // Solve nonlinear system for E at t_{n+1/2}
    // Particles will be advanced to t_{n+1/2}
    m_nlsolver->Solve(m_E, m_Eold, start_time, m_dt, a_step);

    // Update WarpX owned Efield_fp and Bfield_fp to t_{n+1/2}
    UpdateWarpXFields(m_E, half_time);
    m_WarpX->reduced_diags->ComputeDiagsMidStep(a_step);

    // Advance particles from time n+1/2 to time n+1
    m_WarpX->FinishImplicitParticleUpdate();

    // Advance E and B fields from time n+1/2 to time n+1
    amrex::Real const new_time = start_time + m_dt;
    FinishFieldUpdate(new_time);

    // Advance the fields to time n+1 source free
    m_WarpX->SpectralSourceFreeFieldAdvance(half_time);

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
