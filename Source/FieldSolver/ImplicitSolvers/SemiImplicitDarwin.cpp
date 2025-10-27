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

    // Initialize a projection divergence cleaner for the current density
    // m_div_cleaner("current_fp", true);

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

    using ablastr::fields::Direction;

    // Set the member time step
    m_dt = a_dt;

    const int finest_level = 0;

    amrex::Print() << "taking one step in Darwin solver..." << std::endl;

    // Fields have E^{n} (from phi^n only), B^{n-1/2}
    // Particles have u^{n-1/2} and x^{n}.

    // Save u and x at the start of the time step
    // TODO: only save u since we don't need to keep x
    m_WarpX->SaveParticlesAtImplicitStepStart();

    // Push particle velocities with E_fp (which currently just contains -grad phi since
    // the E-field was cleared during the last Poisson solve)
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        m_WarpX->GetPartContainer().PushP(
            lev,
            m_dt,
            *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, lev),
            *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{1}, lev),
            *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{2}, lev),
            *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, lev),
            *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{1}, lev),
            *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{2}, lev)
        );
    }

    // Prepare current deposition by setting particle velocities to twice the
    // t = n velocity values
    PrepareCurrentDeposition();

    // Accumulate current* and susceptibility (mass matrix)
    //AccumulateCurrentAndMassMatrices();

    {
        // TEMPORARY HACK TO TEST SOLVER - THIS CALLBACK IS NOT OTHERWISE USED
        // WITHIN THE CALLBACK Python GETS THE CURRENT DENSITY AND MASS MATRICES
        ExecutePythonCallback("beforedeposition");
    }

    // Populate the source vector
    CalculateSourceVector();

    // Solve MS equation
    m_linear_solver->solve(m_dA, m_source, m_linsol_rtol, m_linsol_atol);

    {
        // TEMPORARY HACK TO TEST SOLVER - THIS CALLBACK IS NOT OTHERWISE USED
        if (IsPythonCallbackInstalled("particlescraper")) {
            auto dA_fp = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::dA_fp, 0);
            const auto& source_vec = m_source.getArrayVec();

            // copy test values from m_dA to dA_fp
            amrex::MultiFab::Copy(*dA_fp[0][0], *source_vec[0][0], 0, 0, 1, source_vec[0][0]->nGrowVect());
            amrex::MultiFab::Copy(*dA_fp[0][1], *source_vec[0][1], 0, 0, 1, source_vec[0][1]->nGrowVect());
            amrex::MultiFab::Copy(*dA_fp[0][2], *source_vec[0][2], 0, 0, 1, source_vec[0][2]->nGrowVect());

            ExecutePythonCallback("particlescraper");

            // Copy solution calculated in Python to the dA vector
            m_dA.Copy( FieldType::dA_fp, FieldType::xi_fp );

            amrex::Print() << "MS solve overwritten from Python." << std::endl;
        }
    }

    // Update E to E = -dA/dt and A to A += dA (recall that B is updated after Poisson solve)
    UpdateEandAfromdA(a_step);

    // Set particle velocities to 0 since the push below is just calculating
    // the acceleratio due to the inductive E-field
    ClearParticleVelocities();

    // Push particle velocities (E-field now only includes the inductive component)
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        m_WarpX->GetPartContainer().PushP(
            lev,
            m_dt,
            *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, lev),
            *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{1}, lev),
            *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{2}, lev),
            *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, lev),
            *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{1}, lev),
            *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{2}, lev)
        );
    }

    // Update particle velocities to include acceleration from both
    // electrostatic and inductive electric field components
    FinishVelocityUpdate();

    // Push particle positions forward (velocities are already updated)
    m_WarpX->GetPartContainer().PushX(m_dt);
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

void SemiImplicitDarwin::PrepareCurrentDeposition ()
{
    BL_PROFILE("SemiImplicitDarwin::PrepareCurrentDeposition()");
    // This function sets the particle velocity pids to twice the intermediate
    // velocity values, i.e., the sum of the values currently stored in u and u_n

    for (auto const& pc : m_WarpX->GetPartContainer()) {

        // for (int lev = 0; lev <= finest_level; ++lev)
        const int lev = 0;
        {
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
            auto particle_comps = pc->GetRealSoANames();

            for (WarpXParIter pti(*pc, lev); pti.isValid(); ++pti) {

                auto& attribs = pti.GetAttribs();
                amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
                amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
                amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();

                amrex::ParticleReal* ux_n = pti.GetAttribs("ux_n").dataPtr();
                amrex::ParticleReal* uy_n = pti.GetAttribs("uy_n").dataPtr();
                amrex::ParticleReal* uz_n = pti.GetAttribs("uz_n").dataPtr();

                const long np = pti.numParticles();

                amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
                {
                    // swap old and new values then add new value to old
                    std::swap(ux[ip], ux_n[ip]);
                    ux[ip] += ux_n[ip];

                    std::swap(uy[ip], uy_n[ip]);
                    uy[ip] += uy_n[ip];

                    std::swap(uz[ip], uz_n[ip]);
                    uz[ip] += uz_n[ip];
                });
            }
        }
    }
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
    // This function calculates the "b" vector for the linear MS equation,
    // i.e., the source vector.
    BL_PROFILE("SemiImplicitDarwin::CalculateSourceVector()");

    const int lev = 0;

    // Zero out existing source values
    m_source.zero();

    // Divergence clean J
    // Initialize a projection divergence cleaner for the current density
    warpx::initialization::ProjectionDivCleaner m_div_cleaner("current_fp", true);
    m_div_cleaner.setSourceFromField();
    m_div_cleaner.solve();
    m_div_cleaner.correctField();

    // Grab the vector potential
    ablastr::fields::MultiLevelVectorField Afield = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::vector_potential_fp, lev);

    // Use the dA_fp MultiFabs as temporary storage for the vector Laplacian of A
    ablastr::fields::MultiLevelVectorField dAfield = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::dA_fp, lev);

    // Calculate the vector Laplacian of A and write result into dA
    m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeVectorLaplacian(
        dAfield[lev], Afield[lev], m_WarpX->GetEBUpdateBFlag()[lev], lev
    );

    // Calculate 2 * nabla^2 A + mu_0 J and write result in dA_fp MF for temporary storage
    const auto& jfield = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, lev);
    for (int ii = 0; ii < 3; ii++)
    {
        amrex::MultiFab::LinComb(
            *dAfield[lev][ii], PhysConst::mu0, *jfield[lev][ii], 0, 2.0, *dAfield[lev][ii], 0, 0, 1, 0
        );
    }

    // Copy calculated source to m_source
    m_source.Copy( FieldType::dA_fp, FieldType::None, true);
}

void SemiImplicitDarwin::UpdateEandAfromdA ( int astep )
{
    // This function updates the Efield_fp MF to hold the new inductive E-field.
    // And updates the vector potential to A^n+1/2 = A^n-1/2 + dA^n
    BL_PROFILE("SemiImplicitDarwin::UpdateEandAfromdA()");

    const int lev = 0;

    // Grab the E-field MultiFabs
    ablastr::fields::MultiLevelVectorField Efield = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, lev);

    // Grab the vector potential
    ablastr::fields::MultiLevelVectorField Afield = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::vector_potential_fp, lev);

    // Grab dA_fp MultiFabs (change in vector potential)
    const auto& dAfield = m_dA.getArrayVec();

    const auto prefac = -1.0_rt / m_dt;
    for (int ii = 0; ii < 3; ii++)
    {
        // Copy dA values to E-field then scale by -1/dt
        amrex::MultiFab::Copy( *Efield[lev][ii], *dAfield[lev][ii], 0, 0, 1,
                                dAfield[lev][ii]->nGrowVect() );
        Efield[lev][ii]->mult(prefac, 0); // use zero ghost cells since FillBoundary is called below

        // Update vector potential
        amrex::MultiFab::Add(*Afield[lev][ii], *dAfield[lev][ii], 0, 0, 1, 0);
        // Fill guard cell values
        Afield[lev][ii]->FillBoundary(m_WarpX->Geom(lev).periodicity());
    }

    // Apply E-field boundary
    m_WarpX->FillBoundaryE(Efield[lev][0]->nGrowVect(), true);
    m_WarpX->ApplyEfieldBoundary(0, PatchType::fine, astep*m_dt);

    // if (m_WarpX->use_filter) {
    //     m_WarpX->ApplyFilterMF(Efield, lev);
    // }
}

void SemiImplicitDarwin::ClearParticleVelocities ()
{
    BL_PROFILE("SemiImplicitDarwin::ClearParticleVelocities()");
    // This function sets the particle velocities to zero since the "corrector"
    // velocity push only calculate the velocity due to acceleration from
    // the inductive E-field. The actual velocities are still stored in u_n.

    for (auto const& pc : m_WarpX->GetPartContainer()) {

        // for (int lev = 0; lev <= finest_level; ++lev)
        const int lev = 0;
        {
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
            auto particle_comps = pc->GetRealSoANames();

            for (WarpXParIter pti(*pc, lev); pti.isValid(); ++pti) {

                auto& attribs = pti.GetAttribs();
                amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
                amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
                amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();

                const long np = pti.numParticles();

                amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
                {
                    ux[ip] = 0.0;
                    uy[ip] = 0.0;
                    uz[ip] = 0.0;
                });
            }
        }
    }
}

void SemiImplicitDarwin::FinishVelocityUpdate ()
{
    BL_PROFILE("SemiImplicitDarwin::FinishVelocityUpdate()");
    // This function sets the particle velocities to include the acceleration
    // from both the electrostatic field (currently held in u_n) and the
    // inductive field (currently held in u)

    for (auto const& pc : m_WarpX->GetPartContainer()) {

        // for (int lev = 0; lev <= finest_level; ++lev)
        const int lev = 0;
        {
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
            auto particle_comps = pc->GetRealSoANames();

            for (WarpXParIter pti(*pc, lev); pti.isValid(); ++pti) {

                auto& attribs = pti.GetAttribs();
                amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
                amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
                amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();

                amrex::ParticleReal* ux_n = pti.GetAttribs("ux_n").dataPtr();
                amrex::ParticleReal* uy_n = pti.GetAttribs("uy_n").dataPtr();
                amrex::ParticleReal* uz_n = pti.GetAttribs("uz_n").dataPtr();

                const long np = pti.numParticles();

                amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
                {
                    ux[ip] += ux_n[ip];
                    uy[ip] += uy_n[ip];
                    uz[ip] += uz_n[ip];
                });
            }
        }
    }
}
