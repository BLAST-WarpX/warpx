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
    m_use_mass_matrices_pc = false;
    m_use_mass_matrices_jacobian = true;
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

void SemiImplicitDarwin::OneStep ( [[maybe_unused]] amrex::Real  start_time,
                                                    amrex::Real  a_dt,
                                                    int          a_step )
{
    BL_PROFILE("SemiImplicitDarwin::OneStep()");

    using ablastr::fields::Direction;

    // Set the member time step
    m_dt = a_dt;

    const int finest_level = 0;

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
    // t = n velocity values (with just the ES acceleration applied for the
    // advanced velocity)
    PrepareCurrentDeposition();

    // Accumulate current* and susceptibility (mass matrices)
    AccumulateCurrentAndSusceptibility();

    // Python callback insertion
    ExecutePythonCallback("afterdeposition");

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
    // the acceleration due to the inductive E-field
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

void SemiImplicitDarwin::ComputeRHS ( WarpXSolverVec& a_RHS,
                                      const WarpXSolverVec& a_dA,
                                      [[maybe_unused]] amrex::Real start_time,
                                      [[maybe_unused]] int a_nl_iter,
                                      [[maybe_unused]] bool a_from_jacobian )
{
    BL_PROFILE("SemiImplicitDarwin::ComputeRHS()");

    const int lev = 0;
    const int ncomps = 1;

    // Evaluate the Darwin MS operator with the given input (a_dA) and
    // write results into a_RHS.
    const auto& dAvec = a_dA.getArrayVec();
    const auto& xivec = a_dA.getScalarVec();

    auto& rhs_vec = a_RHS.getArrayVec();
    auto& rhs_scalar = a_RHS.getScalarVec();

    // The dA_fp and xi_fp MultiFabs are used to store intermediate
    // calculations.
    auto dA_fp = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::dA_fp, lev);
    auto xi_fp = m_WarpX->m_fields.get_mr_levels(FieldType::xi_fp, 0);

    auto E_temp = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, lev);

    // Copy a_dA vector to dA_fp so that guard cell values are included (this
    // is necessary for the vector Laplacian calculation)
    for (int ii = 0; ii < 3; ii++)
    {
        amrex::MultiFab::Copy(*dA_fp[lev][ii], *dAvec[lev][ii], 0, 0, ncomps, 0);
        // Fill guard cell values for vector Laplacian calculation
        dA_fp[lev][ii]->FillBoundary(m_WarpX->Geom(lev).periodicity());
    }

    // Evaluation of first field equation:
    // Calculate the vector Laplacian of the candidate dA and write result into rhs
    m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeVectorLaplacian(
        rhs_vec[lev], dA_fp[lev], m_WarpX->GetEBUpdateEFlag()[lev], lev
    );
    // Calculate gradient of candidate xi and write result into dA_fp
    m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeGradient(
        dA_fp[lev], xivec[lev], m_WarpX->GetEBUpdateEFlag()[lev], lev
    );
    // Write -nabla^2 dA - mu_0 nabla xi into rhs_vec
    for (int ii = 0; ii < 3; ii++)
    {
        amrex::MultiFab::LinComb(
            *rhs_vec[lev][ii], -PhysConst::mu0, *dA_fp[lev][ii], 0, -1.0, *rhs_vec[lev][ii], 0, 0, ncomps, 0
        );
        // clear dA_fp to prepare for susceptibility application
        dA_fp[lev][ii]->setVal(0);

        // use the E-field MFs FOR NOW to include guard cells in the candidate dA
        E_temp[lev][ii]->setVal(0);
        amrex::MultiFab::Copy(*E_temp[lev][ii], *dAvec[lev][ii], 0, 0, ncomps, 0);
        E_temp[lev][ii]->FillBoundary(m_WarpX->Geom(lev).periodicity());
    }
    // Calculate chi dA and write into dA_fp
    ApplySusceptibility(dA_fp, E_temp);

    for (int ii = 0; ii < 3; ii++)
    {
        amrex::MultiFab::Add(*rhs_vec[lev][ii], *dA_fp[lev][ii], 0, 0, ncomps, 0);
    }

    // Evaluation of second field equation:
    // Similarly as before, copy candidate xi to xi_fp and fill boundary for
    // the Laplacian calculation
    amrex::MultiFab::Copy(*xi_fp[lev], *xivec[lev], 0, 0, ncomps, 0);
    xi_fp[lev]->FillBoundary(m_WarpX->Geom(lev).periodicity());
    m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeLaplacian(
        rhs_scalar[lev], xi_fp[lev], m_WarpX->GetEBUpdateEFlag()[lev], lev
    );

    // Calculate divergence of chi dA (currently stored in dA_fp) and write
    // result to xivec - ComputeDivE is used since the staggering matches dA
    xi_fp[lev]->setVal(0);
    m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeDivE(dA_fp[lev], *xi_fp[lev]);

    // Get div chi dA - \mu_0 nabla^2 xi
    amrex::MultiFab::LinComb(
        *rhs_scalar[lev], -PhysConst::mu0, *rhs_scalar[lev], 0, 1.0, *xi_fp[lev], 0, 0, ncomps, 0
    );

    // ------- HACK TO JUST TEST SOLVER -------
    if (IsPythonCallbackInstalled("beforedeposition")) {
        // copy test values from a_dA to dA_fp and xi_fp
        amrex::MultiFab::Copy(*dA_fp[lev][0], *dAvec[lev][0], 0, 0, ncomps, dAvec[lev][0]->nGrowVect());
        amrex::MultiFab::Copy(*dA_fp[lev][1], *dAvec[lev][1], 0, 0, ncomps, dAvec[lev][1]->nGrowVect());
        amrex::MultiFab::Copy(*dA_fp[lev][2], *dAvec[lev][2], 0, 0, ncomps, dAvec[lev][2]->nGrowVect());
        amrex::MultiFab::Copy(*xi_fp[lev], *xivec[lev], 0, 0, ncomps, xivec[lev]->nGrowVect());

        /* ----------------------------------------------------------  */
        // copy C++ calculated RHS values from a_RHS to Efield_fp and rho_fp
        // -- this is just to compare to expected values in Python and is okay
        // since the E-field and rho are overwritten later.
        auto E_fp = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, lev);
        auto rho_fp = m_WarpX->m_fields.get_mr_levels(FieldType::rho_fp, 0);
        amrex::MultiFab::Copy(*E_fp[lev][0], *rhs_vec[lev][0], 0, 0, 1, rhs_vec[lev][0]->nGrowVect());
        amrex::MultiFab::Copy(*E_fp[lev][1], *rhs_vec[lev][1], 0, 0, 1, rhs_vec[lev][1]->nGrowVect());
        amrex::MultiFab::Copy(*E_fp[lev][2], *rhs_vec[lev][2], 0, 0, 1, rhs_vec[lev][2]->nGrowVect());
        amrex::MultiFab::Copy(*rho_fp[lev], *rhs_scalar[lev], 0, 0, 1, rhs_scalar[lev]->nGrowVect());
        /* ----------------------------------------------------------  */

        // Use the Python code to apply the MS operator to the dA_fp and xi_fp
        // vectors, writing the new RHS values in their place
        ExecutePythonCallback("beforedeposition");

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

void SemiImplicitDarwin::AccumulateCurrentAndSusceptibility ()
{
    /*
        Note: The functionality here deposits current to the Yee grid (and
        accumulates the susceptibility to a staggered grid). The prototype
        Darwin solver does the depositions to nodal grids!
        This should maybe be fixed for > 1d!!
        (In 1d the z-current component is basically divergence cleaned away.)

        Note: There is an outstanding issue with this function - the
        `WarpXParticleContainer::DepositCurrentAndMassMatrices` calls
        `doDirectJandSigmaDeposition` which uses `GetImplicitGammaInverse` to
        get the Lorentz factor used in the current deposition. That function
        is not appropriate for the Darwin model since it is hard coded for the
        electromagnetic implicit methods (it uses u and u_n to get a time
        centered gamma).
    */

    BL_PROFILE("SemiImplicitDarwin::AccumulateCurrentAndSusceptibility()");

    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    const int lev = 0;

    amrex::MultiFab * jx = m_WarpX->m_fields.get(FieldType::current_fp, Direction{0}, lev);
    amrex::MultiFab * jy = m_WarpX->m_fields.get(FieldType::current_fp, Direction{1}, lev);
    amrex::MultiFab * jz = m_WarpX->m_fields.get(FieldType::current_fp, Direction{2}, lev);
    amrex::MultiFab * Sxx = m_WarpX->m_fields.get(FieldType::MassMatrices_X, Direction{0}, lev);
    amrex::MultiFab * Sxy = m_WarpX->m_fields.get(FieldType::MassMatrices_X, Direction{1}, lev);
    amrex::MultiFab * Sxz = m_WarpX->m_fields.get(FieldType::MassMatrices_X, Direction{2}, lev);
    amrex::MultiFab * Syx = m_WarpX->m_fields.get(FieldType::MassMatrices_Y, Direction{0}, lev);
    amrex::MultiFab * Syy = m_WarpX->m_fields.get(FieldType::MassMatrices_Y, Direction{1}, lev);
    amrex::MultiFab * Syz = m_WarpX->m_fields.get(FieldType::MassMatrices_Y, Direction{2}, lev);
    amrex::MultiFab * Szx = m_WarpX->m_fields.get(FieldType::MassMatrices_Z, Direction{0}, lev);
    amrex::MultiFab * Szy = m_WarpX->m_fields.get(FieldType::MassMatrices_Z, Direction{1}, lev);
    amrex::MultiFab * Szz = m_WarpX->m_fields.get(FieldType::MassMatrices_Z, Direction{2}, lev);

    // clear MultiFabs in preparation for new deposit
    jx->setVal(0.0);
    jy->setVal(0.0);
    jz->setVal(0.0);
    Sxx->setVal(0.0);
    Sxy->setVal(0.0);
    Sxz->setVal(0.0);
    Syx->setVal(0.0);
    Syy->setVal(0.0);
    Syz->setVal(0.0);
    Szx->setVal(0.0);
    Szy->setVal(0.0);
    Szz->setVal(0.0);

    // Grab B-field MFs since it is needed for the susceptibility
    amrex::MultiFab & Bx = *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, lev);
    amrex::MultiFab & By = *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{1}, lev);
    amrex::MultiFab & Bz = *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{2}, lev);

    // loop over particle containers
    for (auto const& pc : m_WarpX->GetPartContainer()) {

        // TODO: Add omp support
        const int thread_num = 0;

        for (WarpXParIter pti(*pc, lev); pti.isValid(); ++pti)
        {
            // Extract particle data
            auto& attribs = pti.GetAttribs();
            auto&  wp = attribs[PIdx::w];
            auto& uxp = attribs[PIdx::ux];
            auto& uyp = attribs[PIdx::uy];
            auto& uzp = attribs[PIdx::uz];

            const long np = pti.numParticles();

            // Data on the grid
            amrex::FArrayBox const* bxfab = &Bx[pti];
            amrex::FArrayBox const* byfab = &By[pti];
            amrex::FArrayBox const* bzfab = &Bz[pti];

            pc->DepositCurrentAndMassMatrices(pti, wp, uxp, uyp, uzp, jx, jy, jz,
                            Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz,
                            bxfab, byfab, bzfab, 0, np, thread_num, lev, lev, m_dt);
        }
    }

    // Sum boundaries for current
    auto const& period = m_WarpX->Geom(lev).periodicity();
    m_WarpX->SumBoundaryJ(m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, lev), lev, period);

    // TODO: Handle boundary conditions for mass matrices...
    // m_WarpX->SumBoundaryJ(m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::MassMatrices_X, lev), lev, period);
    // m_WarpX->SumBoundaryJ(m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::MassMatrices_Y, lev), lev, period);
    // m_WarpX->SumBoundaryJ(m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::MassMatrices_Y, lev), lev, period);
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
        dAfield[lev], Afield[lev], m_WarpX->GetEBUpdateEFlag()[lev], lev
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

    // Grab m_dA MultiFabs (change in vector potential)
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

void SemiImplicitDarwin::ApplySusceptibility (
    ablastr::fields::MultiLevelVectorField& rhs,
    const ablastr::fields::MultiLevelVectorField& dA )
{
    BL_PROFILE("SemiImplicitDarwin::ApplySusceptibility()");
    // This function applies the susceptibility matrices to the given dA.
    // The functionality is copied from the ``ImplicitSolver::ComputeJfromMassMatrices``
    // function.

    using namespace amrex::literals;

    using warpx::fields::FieldType;
    using ablastr::fields::Direction;

    const int ncomps = 1;
    const int finest_level = 0;

    for (int lev = 0; lev <= finest_level; ++lev) {

        ablastr::fields::VectorField SX = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_X, lev);
        ablastr::fields::VectorField SY = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_Y, lev);
        ablastr::fields::VectorField SZ = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_Z, lev);

        const amrex::IntVect dAx_nodal = dA[lev][0]->ixType().toIntVect();
        const amrex::IntVect dAy_nodal = dA[lev][1]->ixType().toIntVect();
        const amrex::IntVect dAz_nodal = dA[lev][2]->ixType().toIntVect();

        // Compute the component offset in each direction (careful with staggering)
        amrex::IntVect offset_xx, offset_xy, offset_xz;
        amrex::IntVect offset_yx, offset_yy, offset_yz;
        amrex::IntVect offset_zx, offset_zy, offset_zz;
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
            offset_xx[dir] = (m_ncomp_xx[dir]-1)/2;
            offset_xy[dir] = (dAx_nodal[dir] > dAy_nodal[dir]) ?  (m_ncomp_xy[dir]/2)
                                                               : ((m_ncomp_xy[dir]-1)/2);
            offset_xz[dir] = (dAx_nodal[dir] > dAz_nodal[dir]) ?  (m_ncomp_xz[dir]/2)
                                                               : ((m_ncomp_xz[dir]-1)/2);
            offset_yx[dir] = (dAy_nodal[dir] > dAx_nodal[dir]) ?  (m_ncomp_yx[dir]/2)
                                                               : ((m_ncomp_yx[dir]-1)/2);
            offset_yy[dir] = (m_ncomp_yy[dir]-1)/2;
            offset_yz[dir] = (dAy_nodal[dir] > dAz_nodal[dir]) ?  (m_ncomp_yz[dir]/2)
                                                               : ((m_ncomp_yz[dir]-1)/2);
            offset_zx[dir] = (dAz_nodal[dir] > dAx_nodal[dir]) ?  (m_ncomp_zx[dir]/2)
                                                               : ((m_ncomp_zx[dir]-1)/2);
            offset_zy[dir] = (dAz_nodal[dir] > dAy_nodal[dir]) ?  (m_ncomp_zy[dir]/2)
                                                               : ((m_ncomp_zy[dir]-1)/2);
            offset_zz[dir] = (m_ncomp_zz[dir]-1)/2;
        }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( amrex::MFIter mfi(*dA[lev][0], false); mfi.isValid(); ++mfi )
        {
            amrex::Array4<amrex::Real> const& Fx = rhs[lev][0]->array(mfi);
            amrex::Array4<amrex::Real> const& Fy = rhs[lev][1]->array(mfi);
            amrex::Array4<amrex::Real> const& Fz = rhs[lev][2]->array(mfi);

            amrex::Array4<const amrex::Real> const& dAx = dA[lev][0]->array(mfi);
            amrex::Array4<const amrex::Real> const& dAy = dA[lev][1]->array(mfi);
            amrex::Array4<const amrex::Real> const& dAz = dA[lev][2]->array(mfi);

            amrex::Array4<const amrex::Real> const& Sxx = SX[0]->array(mfi);
            amrex::Array4<const amrex::Real> const& Sxy = SX[1]->array(mfi);
            amrex::Array4<const amrex::Real> const& Sxz = SX[2]->array(mfi);

            amrex::Array4<const amrex::Real> const& Syx = SY[0]->array(mfi);
            amrex::Array4<const amrex::Real> const& Syy = SY[1]->array(mfi);
            amrex::Array4<const amrex::Real> const& Syz = SY[2]->array(mfi);

            amrex::Array4<const amrex::Real> const& Szx = SZ[0]->array(mfi);
            amrex::Array4<const amrex::Real> const& Szy = SZ[1]->array(mfi);
            amrex::Array4<const amrex::Real> const& Szz = SZ[2]->array(mfi);

            // Use grown boxes here with all dA guard cells
            amrex::Box dAbx = amrex::convert(mfi.validbox(),dA[lev][0]->ixType());
            amrex::Box dAby = amrex::convert(mfi.validbox(),dA[lev][1]->ixType());
            amrex::Box dAbz = amrex::convert(mfi.validbox(),dA[lev][2]->ixType());
            dAbx.grow(dA[lev][0]->nGrowVect());
            dAby.grow(dA[lev][1]->nGrowVect());
            dAbz.grow(dA[lev][2]->nGrowVect());

            const amrex::IntVect ncomp_xx = m_ncomp_xx;
            const amrex::IntVect ncomp_xy = m_ncomp_xy;
            const amrex::IntVect ncomp_xz = m_ncomp_xz;
            const amrex::IntVect ncomp_yx = m_ncomp_yx;
            const amrex::IntVect ncomp_yy = m_ncomp_yy;
            const amrex::IntVect ncomp_yz = m_ncomp_yz;
            const amrex::IntVect ncomp_zx = m_ncomp_zx;
            const amrex::IntVect ncomp_zy = m_ncomp_zy;
            const amrex::IntVect ncomp_zz = m_ncomp_zz;

            amrex::ParallelFor(
            dAbx, ncomps, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                const int idx[3] = {i, j, k};
                amrex::GpuArray<int, 3> index_min = {0, 0, 0};
                amrex::GpuArray<int, 3> index_max = {0, 0, 0};

                // Compute Sxx*dAx
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_xx[dim],dAbx.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_xx[dim]-1-offset_xx[dim],dAbx.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SxxdAx = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_xx[0],
                                   + ncomp_xx[0]*( jj+offset_xx[1] ),
                                   + ncomp_xx[0]*ncomp_xx[1]*( kk+offset_xx[2] ) );
                            SxxdAx += Sxx(i,j,k,Nc)*dAx(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                // Compute Sxy*dAy
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_xy[dim],dAby.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_xy[dim]-1-offset_xy[dim],dAby.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SxydAy = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_xy[0],
                                   + ncomp_xy[0]*( jj+offset_xy[1] ),
                                   + ncomp_xy[0]*ncomp_xy[1]*( kk+offset_xy[2] ) );
                            SxydAy += Sxy(i,j,k,Nc)*dAy(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                // Compute Sxz*dAz
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_xz[dim],dAbz.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_xz[dim]-1-offset_xz[dim],dAbz.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SxzdAz = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_xz[0],
                                   + ncomp_xz[0]*( jj+offset_xz[1] ),
                                   + ncomp_xz[0]*ncomp_xz[1]*( kk+offset_xz[2] ) );
                            SxzdAz += Sxz(i,j,k,Nc)*dAz(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                Fx(i,j,k,n) += 2._prt * PhysConst::mu0 / m_dt * (SxxdAx + SxydAy + SxzdAz);
            });
            amrex::ParallelFor(
            dAby, ncomps, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                const int idx[3] = {i, j, k};
                amrex::GpuArray<int, 3> index_min = {0, 0, 0};
                amrex::GpuArray<int, 3> index_max = {0, 0, 0};

                // Compute Syx*dAx
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_yx[dim],dAbx.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_yx[dim]-1-offset_yx[dim],dAbx.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SyxdAx = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_yx[0],
                                   + ncomp_yx[0]*( jj+offset_yx[1] ),
                                   + ncomp_yx[0]*ncomp_yx[1]*( kk+offset_yx[2] ) );
                            SyxdAx += Syx(i,j,k,Nc)*dAx(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                // Compute Syy*dAy
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_yy[dim],dAby.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_yy[dim]-1-offset_yy[dim],dAby.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SyydAy = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_yy[0],
                                   + ncomp_yy[0]*( jj+offset_yy[1] ),
                                   + ncomp_yy[0]*ncomp_yy[1]*( kk+offset_yy[2] ) );
                            SyydAy += Syy(i,j,k,Nc)*dAy(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                // Compute Syz*dAz
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_yz[dim],dAbz.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_yz[dim]-1-offset_yz[dim],dAbz.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SyzdAz = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_yz[0],
                                   + ncomp_yz[0]*( jj+offset_yz[1] ),
                                   + ncomp_yz[0]*ncomp_yz[1]*( kk+offset_yz[2] ) );
                            SyzdAz += Syz(i,j,k,Nc)*dAz(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                Fy(i,j,k,n) += 2._prt * PhysConst::mu0 / m_dt * (SyxdAx + SyydAy + SyzdAz);
            });
            amrex::ParallelFor(
            dAbz, ncomps, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                const int idx[3] = {i, j, k};
                amrex::GpuArray<int, 3> index_min = {0, 0, 0};
                amrex::GpuArray<int, 3> index_max = {0, 0, 0};

                // Compute Szx*dAx
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_zx[dim],dAbx.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_zx[dim]-1-offset_zx[dim],dAbx.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SzxdAx = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_zx[0],
                                   + ncomp_zx[0]*( jj+offset_zx[1] ),
                                   + ncomp_zx[0]*ncomp_zx[1]*( kk+offset_zx[2] ) );
                            SzxdAx += Szx(i,j,k,Nc)*dAx(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                // Compute Szy*dAy
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_zy[dim],dAby.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_zy[dim]-1-offset_zy[dim],dAby.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SzydAy = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_zy[0],
                                   + ncomp_zy[0]*( jj+offset_zy[1] ),
                                   + ncomp_zy[0]*ncomp_zy[1]*( kk+offset_zy[2] ) );
                            SzydAy += Szy(i,j,k,Nc)*dAy(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                // Compute Szz*dAz
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_zz[dim],dAbz.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_zz[dim]-1-offset_zz[dim],dAbz.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SzzdAz = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_zz[0],
                                   + ncomp_zz[0]*( jj+offset_zz[1] ),
                                   + ncomp_zz[0]*ncomp_zz[1]*( kk+offset_zz[2] ) );
                            SzzdAz += Szz(i,j,k,Nc)*dAz(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                Fz(i,j,k,n) += 2._prt * PhysConst::mu0 / m_dt * (SzxdAx + SzydAy + SzzdAz);
            });
        }

        // Apply boundary conditions
        rhs[lev][0]->FillBoundary(m_WarpX->Geom(lev).periodicity());
        rhs[lev][1]->FillBoundary(m_WarpX->Geom(lev).periodicity());
        rhs[lev][2]->FillBoundary(m_WarpX->Geom(lev).periodicity());
    }
}
