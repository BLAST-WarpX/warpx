/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Fields.H"
#include "SemiImplicitDarwin.H"
#include "Python/callbacks.H"
#include "WarpX.H"

using warpx::fields::FieldType;
using namespace amrex::literals;

void SemiImplicitDarwin::Define ( WarpX*  a_WarpX )
{
    BL_PROFILE("SemiImplicitDarwin::Define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_is_defined,
        "SemiImplicitDarwin object is already defined!");

    // Retain a pointer back to main WarpX class
    m_WarpX = a_WarpX;

    // Define dA and xi MultiFabs
    using ablastr::fields::Direction;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const auto& ba_Ex = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, lev)->boxArray();
        const auto& ba_Ey = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{1}, lev)->boxArray();
        const auto& ba_Ez = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{2}, lev)->boxArray();
        const auto& dm = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, lev)->DistributionMap();
        const amrex::IntVect nge = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, lev)->nGrowVect();
        m_WarpX->m_fields.alloc_init(FieldType::dA_fp, Direction{0}, lev, ba_Ex, dm, 1, nge, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::dA_fp, Direction{1}, lev, ba_Ey, dm, 1, nge, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::dA_fp, Direction{2}, lev, ba_Ez, dm, 1, nge, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::vector_potential_fp, Direction{0}, lev, ba_Ex, dm, 1, nge, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::vector_potential_fp, Direction{1}, lev, ba_Ey, dm, 1, nge, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::vector_potential_fp, Direction{2}, lev, ba_Ez, dm, 1, nge, 0.0_rt);
        const auto& ba_phi = m_WarpX->m_fields.get(FieldType::phi_fp, lev)->boxArray();
        m_WarpX->m_fields.alloc_init(FieldType::xi_fp, lev, ba_phi, dm, 1, amrex::IntVect(AMREX_D_DECL(1,1,1)), 0.0_rt );
    }

    // Define WarpXSolverVec instances for the MS equation solution (dA) and
    // source
    m_dA.Define( m_WarpX, "dA_fp", "xi_fp" );
    m_dA.zero();
    m_source.Define(m_dA);
    m_source.zero();

    // Parse implicit solver parameters
    // const amrex::ParmParse pp("implicit_evolve");
    // parseNonlinearSolverParams( pp );
    m_use_mass_matrices = true;
    // m_use_mass_matrices_jacobian = true;
    m_nlsolver_type = NonlinearSolverType::None;
    m_max_particle_iterations = 1;
    m_particle_tolerance = 0.0;

    // Get the linear solver input parameters
    const amrex::ParmParse pp_l(amrex::getEnumNameString(m_linear_solver_type));
    pp_l.query("verbose_int",         m_linsol_verbose_int);
    pp_l.query("restart_length",      m_linsol_restart_length);
    pp_l.query("absolute_tolerance",  m_linsol_atol);
    pp_l.query("relative_tolerance",  m_linsol_rtol);
    pp_l.query("max_iterations",      m_linsol_maxits);

    // Define the linear function - Note we could use JacobianFunctionMF if we
    // write ComputeRHS appropriately, this will add some extra overhead in MF operations
    // but would reduce code.
    m_linear_function = std::make_unique<LinearFunctionMF<WarpXSolverVec,SemiImplicitDarwin>>();
    m_linear_function->define(m_dA, this, PreconditionerType::none);

    // Define the nonlinear solver
    // m_nlsolver->Define(m_dA, this);

    // Define the linear solver
    m_linear_solver = std::make_unique<AMReXGMRES<WarpXSolverVec,LinearFunctionMF<WarpXSolverVec,SemiImplicitDarwin>>>();
    m_linear_solver->define(*m_linear_function);
    m_linear_solver->setVerbose( m_linsol_verbose_int );
    m_linear_solver->setRestartLength( m_linsol_restart_length );
    m_linear_solver->setMaxIters( m_linsol_maxits );

    // Initialize the mass matrices for plasma response
    InitializeMassMatrices();

    m_is_defined = true;
}

void SemiImplicitDarwin::PrintParameters () const
{
    if (!m_WarpX->Verbose()) { return; }
    amrex::Print() << "\n";
    amrex::Print() << "-----------------------------------------------------------\n";
    amrex::Print() << "--------- SEMI IMPLICIT DARWIN SOLVER PARAMETERS ----------\n";
    amrex::Print() << "-----------------------------------------------------------\n";
    //PrintBaseImplicitSolverParameters();
    //m_nlsolver->PrintParams();
    auto linsol_name = amrex::getEnumNameString(m_linear_solver_type);
    amrex::Print()     << "Linear solver (" << linsol_name << ") verbose:            " << m_linsol_verbose_int << "\n";
    amrex::Print()     << "Linear solver (" << linsol_name << ") restart length:     " << m_linsol_restart_length << "\n";
    amrex::Print()     << "Linear solver (" << linsol_name << ") max iterations:     " << m_linsol_maxits << "\n";
    amrex::Print()     << "Linear solver (" << linsol_name << ") relative tolerance: " << m_linsol_rtol << "\n";
    amrex::Print()     << "Linear solver (" << linsol_name << ") absolute tolerance: " << m_linsol_atol << "\n";
    amrex::Print() << "-----------------------------------------------------------\n\n";
}

void SemiImplicitDarwin::OneStep ( amrex::Real  start_time,
                                   amrex::Real  a_dt,
                                   int          a_step )
{
    BL_PROFILE("SemiImplicitDarwin::OneStep()");

    amrex::ignore_unused(a_step);

    // Set the member time step
    m_dt = a_dt;

    amrex::Print() << "taking one step in Darwin solver..." << std::endl;

    // Fields have E^{n} (from phi^n only), B^{n-1/2}
    // Particles have u^{n-1/2} and x^{n}.

    // Save u and x at the start of the time step
    // TODO: only save u since we don't need to keep x
    m_WarpX->SaveParticlesAtImplicitStepStart();

    // Push particles with E_fp (which currently just contains -grad phi since
    // the E-field was cleared during the last Poisson solve) and deposit current
    // at the velocity mid-point (should actually be deposited onto the nodes and
    // then averaged to the edges, but is that equivalent to directly depositing
    // on the edges?)
    //PredictorVelocityPush();

    // Accumulate current* and susceptibility (mass matrix)
    //AccumulateCurrentAndMassMatrices();

    // Populate the source vector
    //CalculateSourceVector();

    {
        // TEMPORARY HACK TO TEST SOLVER - THIS CALLBACK IS NOT OTHERWISE USED
        // WITHIN THE CALLBACK Python POPULATES THE dA_fp AND xi_fp
        // MULTIFABS WITH THE SOURCE VALUES
        ExecutePythonCallback("beforedeposition");

        // COPY SOURCE VALUES TO SOURCE VEC
        m_source.Copy( FieldType::dA_fp, FieldType::xi_fp );
    }

    // Solve MS equation
    m_linear_solver->solve(m_dA, m_source, m_linsol_rtol, m_linsol_atol);

    {
        // TEMPORARY HACK TO TEST SOLVER - THIS CALLBACK IS NOT OTHERWISE USED
        auto dA_fp = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::dA_fp, 0);
        const auto& dAvec = m_dA.getArrayVec();

        // copy test values from a_dA to dA_fp and xi_fp
        amrex::MultiFab::Copy(*dA_fp[0][0], *dAvec[0][0], 0, 0, 1, dAvec[0][0]->nGrowVect());
        amrex::MultiFab::Copy(*dA_fp[0][1], *dAvec[0][1], 0, 0, 1, dAvec[0][1]->nGrowVect());
        amrex::MultiFab::Copy(*dA_fp[0][2], *dAvec[0][2], 0, 0, 1, dAvec[0][2]->nGrowVect());

        ExecutePythonCallback("particlescraper");
    }

    // add remainder of algorithm steps...

}

void SemiImplicitDarwin::ComputeRHS ( WarpXSolverVec&  a_RHS,
                                      const WarpXSolverVec&  a_dA,
                                      amrex::Real      start_time,
                                      int              a_nl_iter,
                                      bool             a_from_jacobian )
{
    BL_PROFILE("SemiImplicitDarwin::ComputeRHS()");

    // Evaluate the Darwin MS operator with the given input (a_dA) and
    // write results into a_RHS

    // HACK TO JUST TEST SOLVER
    {
        auto dA_fp = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::dA_fp, 0);
        const auto& dAvec = a_dA.getArrayVec();
        auto xi_fp = m_WarpX->m_fields.get_mr_levels(FieldType::xi_fp, 0);
        const auto& xivec = a_dA.getScalarVec();

        // copy test values from a_dA to dA_fp and xi_fp
        amrex::MultiFab::Copy(*dA_fp[0][0], *dAvec[0][0], 0, 0, 1, dAvec[0][0]->nGrowVect());
        amrex::MultiFab::Copy(*dA_fp[0][1], *dAvec[0][1], 0, 0, 1, dAvec[0][1]->nGrowVect());
        amrex::MultiFab::Copy(*dA_fp[0][2], *dAvec[0][2], 0, 0, 1, dAvec[0][2]->nGrowVect());
        amrex::MultiFab::Copy(*xi_fp[0], *xivec[0], 0, 0, 1, xivec[0]->nGrowVect());

        // Use the Python code to apply the MS operator to the dA_fp and xi_fp
        // vectors, writing the new RHS values in their place
        ExecutePythonCallback("afterdeposition");

        // Copy new RHS values calculated in Python to the a_RHS vector
        a_RHS.Copy( FieldType::dA_fp, FieldType::xi_fp );
    }
}

void SemiImplicitDarwin::PredictorVelocityPush ()
{
    BL_PROFILE("SemiImplicitDarwin::PredictorPush()");

    using warpx::fields::FieldType;

    if (m_WarpX->use_filter) {
        int finest_level = 0;
        m_WarpX->ApplyFilterMF(m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level), 0);
    }

    // Set the implict solver options for particle push, use explicit push
    // to respect the PositionPushType::None option (i.e. no spatial advance) - but then the current is not the average...
    // ImplicitOptions options;
    // options.linear_stage_of_jfnk = a_from_jacobian;
    // options.evolve_suborbit_particles_only = false;
    // options.max_particle_iterations = 1;
    // options.particle_tolerance = 0.0;
    // options.deposit_mass_matrices = m_use_mass_matrices;
    // options.use_mass_matrices_pc = false; // m_use_mass_matrices_pc;
    // options.use_explicit_push = true;

    // const bool skip_current = false;
    // // Advance the particle velocities by dt, then take average of old and new v,
    // // deposit currents, giving J at t=n
    // // This uses Efield_fp and Bfield_fp, where Efield_fp holds the E-field due
    // // to phi^n and Bfield_fp holds the B-field at t = n-1/2.
    // m_WarpX->PushParticlesandDeposit(a_cur_time, skip_current, PositionPushType::None, MomentumPushType::Full, &options);

    // NEED TO USE THIS APPROACH TO GET POSITION AND CURRENT RIGHT (also to keep partially updated velocity value)
    // Could just use PushP and do all the deposition calls ourselves...
    // TODO: make a new WarpX function to PushP all species with _fp fields.
    // using ablastr::fields::Direction;
    // const int lev = 0;
    // PushP(
    //     lev, m_dt,
    //     m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, lev), ...
    // );


}

void AccumulateCurrentAndMassMatrices ()
{
    const int lev = 0;

    // amrex::MultiFab * jx = fields.get(FieldType::current_fp, Direction{0}, lev);
    // amrex::MultiFab * jy = fields.get(FieldType::current_fp, Direction{1}, lev);
    // amrex::MultiFab * jz = fields.get(FieldType::current_fp, Direction{2}, lev);
    // amrex::MultiFab * Sxx = fields.get(FieldType::MassMatrices_X, Direction{0}, lev);
    // amrex::MultiFab * Sxy = fields.get(FieldType::MassMatrices_X, Direction{1}, lev);
    // amrex::MultiFab * Sxz = fields.get(FieldType::MassMatrices_X, Direction{2}, lev);
    // amrex::MultiFab * Syx = fields.get(FieldType::MassMatrices_Y, Direction{0}, lev);
    // amrex::MultiFab * Syy = fields.get(FieldType::MassMatrices_Y, Direction{1}, lev);
    // amrex::MultiFab * Syz = fields.get(FieldType::MassMatrices_Y, Direction{2}, lev);
    // amrex::MultiFab * Szx = fields.get(FieldType::MassMatrices_Z, Direction{0}, lev);
    // amrex::MultiFab * Szy = fields.get(FieldType::MassMatrices_Z, Direction{1}, lev);
    // amrex::MultiFab * Szz = fields.get(FieldType::MassMatrices_Z, Direction{2}, lev);

    // clear MultiFabs in preparation for new deposit
    // jx->setVal(0.0);
    // jy->setVal(0.0);
    // jz->setVal(0.0);
    // Sxx->setVal(0.0);
    // Sxy->setVal(0.0);
    // Sxz->setVal(0.0);
    // Syx->setVal(0.0);
    // Syy->setVal(0.0);
    // Syz->setVal(0.0);
    // Szx->setVal(0.0);
    // Szy->setVal(0.0);
    // Szz->setVal(0.0);

    // loop over particle containers
    //      loop over particles
    //           Get average velocity * 2 and write new velocity into "stored" index

            // DepositCurrentAndMassMatrices(pti, wp, uxp, uyp, uzp, jx, jy, jz,
            //                 Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz,
            //                 bxfab, byfab, bzfab, 0, np_to_deposit, thread_num, lev, lev, dt);

}

void SemiImplicitDarwin::CalculateSourceVector ()
{
    // Divergence clean J

    // Add solenoidal J to source (with mu0 factor)

    // Calculate 2 nabla^2 A and add to source vector


}
