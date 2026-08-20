/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (Realta Fusion)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Fields.H"
#include "SemiImplicitDarwin.H"
#include "Python/callbacks.H"
#include "WarpX.H"

#include <ablastr/warn_manager/WarnManager.H>

using warpx::fields::FieldType;
using namespace amrex::literals;

void SemiImplicitDarwin::Define ( WarpX*  a_WarpX, bool from_restart)
{
    amrex::ignore_unused(from_restart);
    BL_PROFILE("SemiImplicitDarwin::Define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_is_defined,
        "SemiImplicitDarwin object is already defined!");

    // Retain a pointer back to main WarpX class
    m_WarpX = a_WarpX;

    // The guard-cell handling throughout this solver (SumBoundaryJ and
    // FillBoundaryAndSync calls using the domain periodicity) and the GMRES
    // operator in DarwinLinearFieldOperator assume periodic boundaries; with conducting
    // (PEC) walls the run would proceed but give wrong results near the walls.
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_WarpX->Geom(lev).isAllPeriodic(),
            "The semi-implicit Darwin solver requires periodic field boundary "
            "conditions in all directions.");
    }

    // Define dA MultiFabs
    using ablastr::fields::Direction;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const auto& ba_Ex = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, lev)->boxArray();
        const auto& ba_Ey = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{1}, lev)->boxArray();
        const auto& ba_Ez = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{2}, lev)->boxArray();
        const auto& dm_E = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, lev)->DistributionMap();
        const amrex::IntVect nge = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, lev)->nGrowVect();
        m_WarpX->m_fields.alloc_init(FieldType::dA_fp, Direction{0}, lev, ba_Ex, dm_E, 1, nge, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::dA_fp, Direction{1}, lev, ba_Ey, dm_E, 1, nge, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::dA_fp, Direction{2}, lev, ba_Ez, dm_E, 1, nge, 0.0_rt);
    }

    // Define WarpXSolverVec instances for the magnetoinductive equation solution (Z) and source
    m_Z.Define( m_WarpX, "Bfield_fp");
    m_Z.zero();
    m_source.Define(m_Z);
    m_source.zero();

    // Set parameters used by `InitializeMassMatrices`
    m_use_mass_matrices = true;
    m_use_mass_matrices_pc = false;
    m_use_mass_matrices_jacobian = true;

    // Get the linear solver input parameters
    const amrex::ParmParse pp_l(amrex::getEnumNameString(m_linear_solver_type));
    pp_l.query("verbose_int",         m_linsol_verbose_int);
    pp_l.query("restart_length",      m_linsol_restart_length);
    pp_l.query("absolute_tolerance",  m_linsol_atol);
    pp_l.query("relative_tolerance",  m_linsol_rtol);
    pp_l.query("max_iterations",      m_linsol_maxits);

    // Define the linear operator (this also allocates the scratch space it
    // uses to evaluate the operator on each GMRES iteration)
    m_linear_function = std::make_unique<DarwinLinearFieldOperator>();
    m_linear_function->define(m_Z, this, PreconditionerType::none);

    // Define the linear solver
    if (m_linear_solver_type == LinearSolverType::amrex_gmres) {
        m_linear_solver = std::make_unique<AMReXGMRES<WarpXSolverVec,DarwinLinearFieldOperator>>();
    }
    else {
        amrex::Abort("Darwin linear solver: unknown type");
    }
    m_linear_solver->define(*m_linear_function);
    m_linear_solver->setVerbose( m_linsol_verbose_int );
    m_linear_solver->setRestartLength( m_linsol_restart_length );
    m_linear_solver->setMaxIters( m_linsol_maxits );

    // Initialize the mass matrices for plasma response
    InitializeMassMatrices();

    // The predictor velocity push in OneStep() temporarily overrides the
    // global galerkin_interpolation flag to true, to gather with the same
    // shape-factor order used for deposition. Skip that override, and warn,
    // if the user has explicitly selected momentum-conserving gathering,
    // since forcing Galerkin gathering there would silently negate that choice.
    m_predictor_use_galerkin = (WarpX::field_gathering_algo != GatheringAlgo::MomentumConserving);
    if (!m_predictor_use_galerkin) {
        ablastr::warn_manager::WMRecordWarning("Semi-implicit Darwin solver",
            "algo.field_gathering = momentum_conserving is set; the predictor "
            "velocity push will keep using momentum-conserving gathering "
            "rather than switching to the Galerkin scheme.",
            ablastr::warn_manager::WarnPriority::medium);
    }

    m_is_defined = true;
}

void SemiImplicitDarwin::PrintParameters () const
{
    amrex::Print() << "\n";
    amrex::Print() << "-----------------------------------------------------------\n";
    amrex::Print() << "--------- SEMI IMPLICIT DARWIN SOLVER PARAMETERS ----------\n";
    amrex::Print() << "-----------------------------------------------------------\n";

    auto linsol_name = amrex::getEnumNameString(m_linear_solver_type);
    amrex::Print()     << "Linear solver (" << linsol_name << ") verbose:            " << m_linsol_verbose_int << "\n";
    amrex::Print()     << "Linear solver (" << linsol_name << ") restart length:     " << m_linsol_restart_length << "\n";
    amrex::Print()     << "Linear solver (" << linsol_name << ") max iterations:     " << m_linsol_maxits << "\n";
    amrex::Print()     << "Linear solver (" << linsol_name << ") relative tolerance: " << m_linsol_rtol << "\n";
    amrex::Print()     << "Linear solver (" << linsol_name << ") absolute tolerance: " << m_linsol_atol << "\n";
    amrex::Print() << "-----------------------------------------------------------\n\n";
}

int SemiImplicitDarwin::OneStep ( [[maybe_unused]] amrex::Real  start_time,
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

    // Push particle velocities with E_fp (which currently just contains -grad(phi) since
    // the E-field was cleared during the last Poisson solve). Temporarily force
    // Galerkin gathering for this predictor push (skipped if the user explicitly
    // requested momentum-conserving gathering - see the warning issued in Define()).
    const bool save_galerkin_interpolation = WarpX::galerkin_interpolation;
    if (m_predictor_use_galerkin) { WarpX::galerkin_interpolation = true; }

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
            *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{2}, lev),
            MomentumPushType::Full
        );
    }

    WarpX::galerkin_interpolation = save_galerkin_interpolation;

    // Prepare current deposition: the velocities are time centered with
    // u -> (u^{n+1/2} + u^{n-1/2}) / 2.0 (with just the ES acceleration applied
    // for the advanced velocity), and the advanced velocity is saved to u_n
    PrepareVelocitiesForCurrentDeposition();

    // Accumulate current* and the mass matrices
    AccumulateCurrentAndMassMatrices();

    // Python callback insertion
    ExecutePythonCallback("afterdeposition");

    // Populate the source vector
    // i.e. fill m_source with `2 * laplacian(B) + 2 * mu_0 curl(J)`
    CalculateSourceVector();

    // Solve the magnetoinductive equation:
    // bilaplacian(Z) + curl(chi curl(Z)) = 2 * laplacian(B) + 2 * mu_0 curl(J)
    // where chi is the mass matrix scaled by 2 * mu_0 / dt (see
    // ApplyScaledMassMatrices), i.e. the linear response of the deposited
    // current to the inductive E-field that this solve produces.
    m_linear_solver->solve(m_Z, m_source, m_linsol_rtol, m_linsol_atol);

    // AMReX's GMRES::getStatus() returns 0 on convergence and a positive
    // value (e.g. 1 if the iteration count was exceeded) otherwise. Map
    // that onto the negative-means-failure convention used by the caller.
    const int exit_status = (m_linear_solver->getStatus() == 0) ? 0 : -1;
    if (exit_status < 0) {
        return exit_status;
    }

    // Set E = -dA/dt (B is updated after the corrector push below)
    ComputeInductiveEfromdA(a_step);

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
            *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{2}, lev),
            MomentumPushType::Full
        );
    }

    // Update particle velocities to include acceleration from both
    // electrostatic and inductive electric field components
    FinishVelocityUpdate();

    // Push particle positions forward (velocities are already updated)
    m_WarpX->GetPartContainer().PushX(m_dt);

    // Update magnetic field using dB/dt = -curl(E)
    m_WarpX->EvolveB(m_dt, SubcyclingHalf::None, a_step*m_dt);
    m_WarpX->FillBoundaryB(m_WarpX->getngEB(), true);

    return exit_status;
}

void SemiImplicitDarwin::ComputeRHS ( [[maybe_unused]] WarpXSolverVec& a_RHS,
                                      [[maybe_unused]] const WarpXSolverVec& a_Z,
                                      [[maybe_unused]] amrex::Real start_time,
                                      [[maybe_unused]] int a_nl_iter,
                                      [[maybe_unused]] bool a_from_jacobian )
{
    // The Darwin scheme is linear in its unknown and never installs a
    // nonlinear solver, so it has no nonlinear residual to compute. This
    // override only exists because ImplicitSolver::ComputeRHS() is pure
    // virtual. The linear operator that GMRES applies each iteration is
    // DarwinLinearFieldOperator::apply() instead.
    WARPX_ABORT_WITH_MESSAGE(
        "SemiImplicitDarwin::ComputeRHS() is not implemented: the semi-implicit "
        "Darwin solver is linear and uses DarwinLinearFieldOperator::apply() instead.");
}

void SemiImplicitDarwin::PrepareVelocitiesForCurrentDeposition ()
{
    BL_PROFILE("SemiImplicitDarwin::PrepareVelocitiesForCurrentDeposition()");
    // On entry, u holds the velocity after the electrostatic-only push
    // (PushP in OneStep()) and u_n holds the velocity saved at the start of
    // the step (SaveParticlesAtImplicitStepStart()). This function sets u to
    // the time-centered average of the two, which is what
    // GetImplicitGammaInverse() and setMassMatricesKernels() (shared with
    // the electromagnetic implicit schemes) expect as the deposition-time
    // velocity to compute a correct relativistic gamma factor from.
    // u_n is left holding the electrostatic-only velocity (the u value at the
    // start of this function) rather than the step-start value, since
    // FinishVelocityUpdate() later reads u_n to recombine the electrostatic
    // and inductive velocity contributions; GetImplicitGammaInverse()'s
    // reconstruction is symmetric under swapping which of the two sampled
    // velocities is treated as "u_n" vs "u_nph", so this substitution does
    // not affect the deposition-time physics.

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
                    const amrex::ParticleReal ux_es = ux[ip];
                    ux[ip] = 0.5_prt*(ux_es + ux_n[ip]);
                    ux_n[ip] = ux_es;

                    const amrex::ParticleReal uy_es = uy[ip];
                    uy[ip] = 0.5_prt*(uy_es + uy_n[ip]);
                    uy_n[ip] = uy_es;

                    const amrex::ParticleReal uz_es = uz[ip];
                    uz[ip] = 0.5_prt*(uz_es + uz_n[ip]);
                    uz_n[ip] = uz_es;
                });
            }
        }
    }
}

void SemiImplicitDarwin::AccumulateCurrentAndMassMatrices ()
{

    BL_PROFILE("SemiImplicitDarwin::AccumulateCurrentAndMassMatrices()");

    using warpx::fields::FieldType;

    const int lev = 0;

    // Deposit the current density from all species, using the time-centered
    // particle velocities as appropriate for the implicit push. This also
    // resets the current MultiFabs before depositing.
    m_WarpX->GetPartContainer().DepositCurrent(
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, lev),
        m_dt, 0.0_rt, PushType::Implicit);

    // Zero and accumulate the mass matrices from all species. This shares the
    // zero-then-deposit machinery with the electromagnetic implicit solvers
    // (see ImplicitSolver::PreLinearSolve), which drive the same
    // WarpX::DepositMassMatrices() -> MultiParticleContainer::DepositMassMatrices().
    m_WarpX->DepositMassMatrices();

    // Sync current (filter and sum boundaries)
    m_WarpX->SyncCurrent("current_fp");

    // Sum boundaries for mass matrices
    m_WarpX->SyncMassMatrices();

    // The deposit routine only fills half of each diagonal mass matrix's
    // band (exploiting symmetry); mirror the other half back in now that
    // deposition and boundary summation are complete.
    FinishMassMatrices();
}

void SemiImplicitDarwin::CalculateSourceVector ()
{
    // Compute the right-hand side of the magnetoinductive equation
    // bilaplacian(Z) + curl(chi curl(Z)) = 2 * laplacian(B) + 2 * mu_0 curl(J)
    // where chi is the mass matrix scaled by 2 * mu_0 / dt (see
    // ApplyScaledMassMatrices).
    BL_PROFILE("SemiImplicitDarwin::CalculateSourceVector()");

    const int lev = 0;

    // Zero out existing source values
    m_source.zero();

    // Grab the magnetic field and current density
    ablastr::fields::MultiLevelVectorField Bfield = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, lev);
    ablastr::fields::MultiLevelVectorField jfield = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, lev);

    // Ensure guard cells are valid before differentiating these fields below -
    // this function doesn't otherwise control when Bfield_fp/current_fp were
    // last synced, so don't rely on that happening elsewhere.
    for (int ii = 0; ii < 3; ii++)
    {
        Bfield[lev][ii]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
        jfield[lev][ii]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
    }

    // Create temporary multifabs with B-staggering for storage
    amrex::MultiFab lapB_x(Bfield[lev][0]->boxArray(), Bfield[lev][0]->DistributionMap(),
                           Bfield[lev][0]->nComp(), Bfield[lev][0]->nGrowVect());
    amrex::MultiFab lapB_y(Bfield[lev][1]->boxArray(), Bfield[lev][1]->DistributionMap(),
                           Bfield[lev][1]->nComp(), Bfield[lev][1]->nGrowVect());
    amrex::MultiFab lapB_z(Bfield[lev][2]->boxArray(), Bfield[lev][2]->DistributionMap(),
                           Bfield[lev][2]->nComp(), Bfield[lev][2]->nGrowVect());
    ablastr::fields::VectorField lapB = {&lapB_x, &lapB_y, &lapB_z};

    amrex::MultiFab curlJ_x(Bfield[lev][0]->boxArray(), Bfield[lev][0]->DistributionMap(),
                            Bfield[lev][0]->nComp(), Bfield[lev][0]->nGrowVect());
    amrex::MultiFab curlJ_y(Bfield[lev][1]->boxArray(), Bfield[lev][1]->DistributionMap(),
                            Bfield[lev][1]->nComp(), Bfield[lev][1]->nGrowVect());
    amrex::MultiFab curlJ_z(Bfield[lev][2]->boxArray(), Bfield[lev][2]->DistributionMap(),
                            Bfield[lev][2]->nComp(), Bfield[lev][2]->nGrowVect());
    ablastr::fields::VectorField curlJ = {&curlJ_x, &curlJ_y, &curlJ_z};

    // Calculate the vector Laplacian of B and write result into first temporary MF
    m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeVectorLaplacian(
        lapB, Bfield[lev], m_WarpX->GetEBUpdateBFlag()[lev], lev
    );

    // Calculate the curl of J and write result into second temporary MF
    m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(
        curlJ, jfield[lev], m_WarpX->GetEBUpdateBFlag()[lev], lev
    );

    // Calculate 2 * laplacian(B) + 2 * mu_0 curl(J) and write result in m_source
    const auto& b = m_source.getArrayVec();
    for (int ii = 0; ii < 3; ii++)
    {
        amrex::MultiFab::LinComb(
            *b[lev][ii], 2.0*PhysConst::mu0, *curlJ[ii], 0, 2.0, *lapB[ii], 0, 0, 1, 0
        );
    }

    // This is the RHS GMRES solves against for the entire step.
    for (int ii = 0; ii < 3; ii++)
    {
        b[lev][ii]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
    }
}

void SemiImplicitDarwin::ComputeInductiveEfromdA ( int astep )
{
    // This function updates the Efield_fp MF to hold the new inductive E-field.
    BL_PROFILE("SemiImplicitDarwin::ComputeInductiveEfromdA()");

    const int lev = 0;

    // Grab the E-field MultiFabs
    ablastr::fields::MultiLevelVectorField Efield = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, lev);

    // Grab the dA_fp MultiFabs to store dA = curl(Z) (the solved-for Z lives
    // on B's staggering; dA lives on A/E's staggering)
    ablastr::fields::MultiLevelVectorField dAfield = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::dA_fp, lev);

    // Grab m_Z MultiFabs (the solved-for Z). Z's valid region is sound at this
    // point - it is a linear combination of vectors that were themselves
    // consistent, and GMRES's arithmetic is element-wise - but WarpXSolverVec
    // always allocates with zero guard cells (see its Define()), so there are
    // none to fill in place, while the ComputeCurlB stencil below reads i-1.
    // Hence the copy into a local scratch one cell wider, and the boundary fill
    // on that, same as in the linear operator.
    const auto& Zfield = m_Z.getArrayVec();
    const amrex::IntVect curl_ng = amrex::IntVect(1);
    amrex::MultiFab Zscratch_x(Zfield[lev][0]->boxArray(), Zfield[lev][0]->DistributionMap(),
                               Zfield[lev][0]->nComp(), curl_ng);
    amrex::MultiFab Zscratch_y(Zfield[lev][1]->boxArray(), Zfield[lev][1]->DistributionMap(),
                               Zfield[lev][1]->nComp(), curl_ng);
    amrex::MultiFab Zscratch_z(Zfield[lev][2]->boxArray(), Zfield[lev][2]->DistributionMap(),
                               Zfield[lev][2]->nComp(), curl_ng);
    ablastr::fields::VectorField Zscratch = {&Zscratch_x, &Zscratch_y, &Zscratch_z};
    for (int ii = 0; ii < 3; ii++)
    {
        amrex::MultiFab::Copy(*Zscratch[ii], *Zfield[lev][ii], 0, 0, 1, 0);
        // Z's transverse components are nodal, so transverse end points are
        // *valid* cells possibly representing the same periodic-wrapped point.
        // Plain FillBoundary only reconciles true ghost cells, not two
        // overlapping valid cells - use FillBoundaryAndSync instead.
        Zscratch[ii]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
    }

    // Calculate dA = curl(Z)
    m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeCurlB(
        dAfield[lev], Zscratch, m_WarpX->GetEBUpdateEFlag()[lev], lev
    );
    for (int ii = 0; ii < 3; ii++)
    {
        dAfield[lev][ii]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
    }

    const auto prefac = -1.0_rt / m_dt;
    for (int ii = 0; ii < 3; ii++)
    {
        // Copy dA values to E-field then scale by -1/dt
        amrex::MultiFab::Copy( *Efield[lev][ii], *dAfield[lev][ii], 0, 0, 1,
                                dAfield[lev][ii]->nGrowVect() );
        Efield[lev][ii]->mult(prefac, 0); // use zero ghost cells since FillBoundary is called below
    }

    // Apply E-field boundary
    m_WarpX->ApplyEfieldBoundary(0, PatchType::fine, astep*m_dt);
    m_WarpX->FillBoundaryE(m_WarpX->getngEB(), true);
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

void SemiImplicitDarwin::ApplyScaledMassMatrices (
    ablastr::fields::MultiLevelVectorField& rhs,
    const ablastr::fields::MultiLevelVectorField& dA )
{
    BL_PROFILE("SemiImplicitDarwin::ApplyScaledMassMatrices()");
    using namespace amrex::literals;

    const amrex::Real scale = 2._prt * PhysConst::mu0 / m_dt;

    ApplyMassMatrices(
        /* a_out           = */ rhs,
        /* a_in            = */ dA,
        /* a_in_ref        = */ nullptr,
        /* a_baseline      = */ nullptr,
        /* a_scale         = */ scale,
        /* a_zero_out_first = */ false);

    for (int lev = 0; lev < static_cast<int>(rhs.size()); ++lev) {
        // Fill and sync guard cells & edges
        rhs[lev][0]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
        rhs[lev][1]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
        rhs[lev][2]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
    }
}
