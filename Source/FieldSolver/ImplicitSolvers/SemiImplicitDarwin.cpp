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
#include "BoundaryConditions/WarpX_PEC.H"
#include "Python/callbacks.H"
#include "WarpX.H"

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

#if defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_XZ)
    // Conducting (PEC) walls are supported in 1D and 2D: the wall treatment
    // (image-symmetry ghost fills on Z/dA, guard-cell folds of J and of the
    // chi*dA product, and projection of the constrained wall DOFs) is applied
    // wherever m_has_pec is set below. Any other non-periodic boundary type
    // is not handled by this solver.
    {
        const auto& fbl = m_WarpX->GetFieldBoundaryLo();
        const auto& fbh = m_WarpX->GetFieldBoundaryHi();
        // Map each spatial dimension to the vector component normal to its
        // boundary planes, to identify the exact constant null vectors of the
        // wall-projected operator (see ProjectOutNullConstants).
#if defined(WARPX_DIM_1D_Z)
        const std::array<int, 1> wall_normal_comp = {2};
#else
        const std::array<int, 2> wall_normal_comp = {0, 2};
#endif
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (fbl[idim] == FieldBoundaryType::Periodic || fbl[idim] == FieldBoundaryType::PEC) &&
                (fbh[idim] == FieldBoundaryType::Periodic || fbh[idim] == FieldBoundaryType::PEC),
                "The semi-implicit Darwin solver only supports periodic and PEC "
                "field boundary conditions.");
            m_has_pec = m_has_pec ||
                (fbl[idim] == FieldBoundaryType::PEC) || (fbh[idim] == FieldBoundaryType::PEC);
        }
        if (m_has_pec) {
            // A constant Z component is an exact null vector unless some PEC
            // wall is normal to it (the wall projection zeroes those DOFs).
            m_null_constant_comp = {true, true, true};
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                if (fbl[idim] == FieldBoundaryType::PEC || fbh[idim] == FieldBoundaryType::PEC) {
                    m_null_constant_comp[wall_normal_comp[idim]] = false;
                }
            }
        }
    }
    if (m_has_pec) {
        // Orbit cropping at PEC walls is only implemented in the Villasenor
        // deposition kernels; with direct deposition the guard-cell deposits
        // are folded back into the domain instead (see
        // AccumulateCurrentAndSusceptibility and ApplySusceptibility).
        bool crop_on_pec = false;
        const amrex::ParmParse pp_particles("particles");
        pp_particles.query("crop_on_PEC_boundary", crop_on_pec);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !(crop_on_pec && WarpX::current_deposition_algo == CurrentDepositionAlgo::Direct),
            "particles.crop_on_PEC_boundary is not supported with direct current "
            "deposition in the semi-implicit Darwin solver.");
    }
#else
    // The wall treatment below is only implemented (and verified) for the
    // 1D and 2D Cartesian geometries so far.
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_WarpX->Geom(lev).isAllPeriodic(),
            "The semi-implicit Darwin solver requires periodic field boundary "
            "conditions in all directions in this geometry.");
    }
#endif

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

    // Define WarpXSolverVec instances for the MS equation solution (Z) and source
    m_Z.Define( m_WarpX, "Bfield_fp");
    m_Z.zero();
    m_source.Define(m_Z);
    m_source.zero();

    // Scratch space used by ComputeRHS(), allocated once here (every
    // iterate of m_Z shares this same layout) rather than on every
    // GMRES iteration.
    {
        const auto& Zvec = m_Z.getArrayVec();
        const int lev = 0;
        m_lapZ_x.define(Zvec[lev][0]->boxArray(), Zvec[lev][0]->DistributionMap(),
                        Zvec[lev][0]->nComp(), Zvec[lev][0]->nGrowVect());
        m_lapZ_y.define(Zvec[lev][1]->boxArray(), Zvec[lev][1]->DistributionMap(),
                        Zvec[lev][1]->nComp(), Zvec[lev][1]->nGrowVect());
        m_lapZ_z.define(Zvec[lev][2]->boxArray(), Zvec[lev][2]->DistributionMap(),
                        Zvec[lev][2]->nComp(), Zvec[lev][2]->nGrowVect());

        const amrex::IntVect biharmonic_ng =
            amrex::elemwiseMax(Zvec[lev][0]->nGrowVect(), amrex::IntVect(2));
        m_Zscratch_x.define(Zvec[lev][0]->boxArray(), Zvec[lev][0]->DistributionMap(),
                            Zvec[lev][0]->nComp(), biharmonic_ng);
        m_Zscratch_y.define(Zvec[lev][1]->boxArray(), Zvec[lev][1]->DistributionMap(),
                            Zvec[lev][1]->nComp(), biharmonic_ng);
        m_Zscratch_z.define(Zvec[lev][2]->boxArray(), Zvec[lev][2]->DistributionMap(),
                            Zvec[lev][2]->nComp(), biharmonic_ng);

        // div(Z) of the B-staggered Z is fully cell-centered in every
        // Cartesian geometry. One guard layer: AddGradDivZTerm() evaluates
        // div(Z) on each FAB's grown box so the gradient can consume it
        // without an intermediate boundary fill.
        m_divZ.define(amrex::convert(Zvec[lev][0]->boxArray(), amrex::IndexType::TheCellType()),
                      Zvec[lev][0]->DistributionMap(), 1, amrex::IntVect(1));
    }

    // Parse implicit solver parameters
    m_use_mass_matrices = true;
    m_use_mass_matrices_pc = false;
    m_use_mass_matrices_jacobian = true;
    m_nlsolver_type = NonlinearSolverType::none;
    m_max_particle_iterations = 1;

    // Get the linear solver input parameters
    const amrex::ParmParse pp_l(amrex::getEnumNameString(m_linear_solver_type));
    pp_l.query("verbose_int",         m_linsol_verbose_int);
    pp_l.query("restart_length",      m_linsol_restart_length);
    pp_l.query("absolute_tolerance",  m_linsol_atol);
    pp_l.query("relative_tolerance",  m_linsol_rtol);
    pp_l.query("max_iterations",      m_linsol_maxits);

    // Multiplier on the grad-div (Coulomb-gauge penalty) stabilization
    // coefficient (see AddGradDivZTerm); 0 disables the term.
    const amrex::ParmParse pp_darwin("semi_implicit_darwin");
    pp_darwin.query("graddiv_factor", m_graddiv_factor);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_graddiv_factor >= 0.0_rt,
        "semi_implicit_darwin.graddiv_factor must be >= 0 (a negative value "
        "would make the grad-div penalty term indefinite).");

    // Define the linear function - Note we could use JacobianFunctionMF if we
    // write ComputeRHS appropriately, this will add some extra overhead in MF operations
    // but would reduce code.
    m_linear_function = std::make_unique<LinearFunctionMF<WarpXSolverVec,SemiImplicitDarwin>>();
    m_linear_function->define(m_Z, this, PreconditionerType::none);

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
    amrex::Print()     << "Grad-div (gauge penalty) factor:                    " << m_graddiv_factor << "\n";
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

    // Push particle velocities with E_fp (which currently just contains -∇ phi since
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
            *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{2}, lev),
            MomentumPushType::Full
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

    // Project the warm-start initial guess (the previous step's Z) at
    // conducting (PEC) walls: the wall-normal rows of the projected system
    // have identically zero residual, so a nonzero wall value in the initial
    // guess would persist through the solve and contaminate dA at the wall.
    ApplyPECtoZ(m_Z.getArrayVec()[0], 0, amrex::IntVect(0));

    // Solve MS equation
    m_linear_solver->solve(m_Z, m_source, m_linsol_rtol, m_linsol_atol);

    // AMReX's GMRES::getStatus() returns 0 on convergence and a positive
    // value (e.g. 1 if the iteration count was exceeded) otherwise. Map
    // that onto the negative-means-failure convention used by the caller.
    const int exit_status = (m_linear_solver->getStatus() == 0) ? 0 : -1;
    if (exit_status < 0) {
        return exit_status;
    }

    // Update E to E = -dA/dt (B is updated after the corrector push below)
    UpdateEfromdA(a_step);

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

    // Update magnetic field using dB/dt = -∇ x E
    m_WarpX->EvolveB(m_dt, SubcyclingHalf::None, a_step*m_dt);
    m_WarpX->FillBoundaryB(m_WarpX->getngEB(), true);

    return exit_status;
}

void SemiImplicitDarwin::ComputeRHS ( WarpXSolverVec& a_RHS,
                                      const WarpXSolverVec& a_Z,
                                      [[maybe_unused]] amrex::Real start_time,
                                      [[maybe_unused]] int a_nl_iter,
                                      [[maybe_unused]] bool a_from_jacobian )
{
    BL_PROFILE("SemiImplicitDarwin::ComputeRHS()");

    const int lev = 0;
    const int ncomps = 1;

    // Evaluate the Darwin MS operator with the given input (a_Z) and
    // write results into a_RHS.
    const auto& Zvec = a_Z.getArrayVec();
    auto& rhs_vec = a_RHS.getArrayVec();

    // The dA_fp and Efield_fp MultiFabs are used to store intermediate calculations.
    auto dA_fp = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::dA_fp, lev);
    auto E_temp = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, lev);

    // Scratch space (allocated once in Define()), reused below to hold ∇ x (chi ∇ x Z).
    ablastr::fields::VectorField lapZ = {&m_lapZ_x, &m_lapZ_y, &m_lapZ_z};

    // GMRES builds intermediate Krylov candidates via WarpXSolverVec's
    // arithmetic (linComb/increment/Saxpy), which only ever touch the valid
    // region (nghost=0), so a_Z's own guard cells cannot be trusted here.
    // Copy the candidate into a B-staggered scratch (allocated once in
    // Define(), with >=2 ghost cells for the nabla^4 stencil below, which
    // reads i-2..i+2) and FillBoundary on that scratch, which derives its
    // guard cells from its own (just-copied) valid-region data via the
    // periodic halo exchange.
    ablastr::fields::VectorField Zscratch = {&m_Zscratch_x, &m_Zscratch_y, &m_Zscratch_z};
    for (int ii = 0; ii < 3; ii++)
    {
        amrex::MultiFab::Copy(*Zscratch[ii], *Zvec[lev][ii], 0, 0, ncomps, 0);
        // Plain FillBoundary only reconciles true ghost cells, not two
        // overlapping valid cells - use FillBoundaryAndSync instead (harmless
        // no-op for the transverse, cell-centered components, which have no
        // such duplication).
        Zscratch[ii]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
    }

    // At conducting (PEC) walls, fill Zscratch's guard cells with the B-type
    // image symmetry (2 layers, covering the nabla^4 stencil below) and zero
    // the wall-normal DOFs of the iterate - the projection that, together
    // with the corresponding zeroing of the operator output and source, makes
    // GMRES solve the reflected half-domain problem. Must come after the
    // periodic fill above so mirror reads (and 2D corner ghosts) see valid
    // data.
    ApplyPECtoZ(Zscratch, lev, m_Zscratch_x.nGrowVect());

    // Evaluation of the (single) 4th-order field equation:
    // ∇^4(Z), discretized directly in a single pass. Composing two
    // separate ComputeVectorLaplacian calls (with an intermediate boundary
    // fill in between) introduces a parasitic, sign-alternating mode in Z's
    // nodal component, since each application's guard cells are filled
    // independently rather than being derived from a single consistent
    // wide-stencil read of Z.
    m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeVectorBiLaplacian(
        rhs_vec[lev], Zscratch, m_WarpX->GetEBUpdateBFlag()[lev], lev
    );

    // Constrain Z's divergence with a grad-div (Coulomb-gauge penalty) term.
    // Deriving nabla^4(Z) + curl(chi(curl(Z))) from the validated 2nd-order
    // (dA, xi) equation relies on curl(curl(Z)) = -nabla^2(Z), which only
    // holds if div(Z)=0. Nothing else in the operator constrains div(Z): the
    // curl(chi curl) term annihilates curl-free modes, leaving them
    // controlled by the bare biharmonic k^4 alone - orders of magnitude more
    // singular than the solenoidal modes' k^4 + chi*k^2 in domains large
    // compared to the electron skin depth, which makes GMRES grind on any
    // divergence-carrying source content (wall deposits, PIC noise). Adding
    // -grad(alpha div(Z)) with alpha = mean(chi) lifts curl-free modes to
    // the same k^4 + alpha*k^2 scale while leaving solenoidal modes (and,
    // since curl(grad(.)) = 0 discretely, dA = curl(Z)) exactly unchanged.
    // Note this is deliberately NOT the exact-identity correction
    // -grad(nabla^2(div Z)) (which would cancel the biharmonic on curl-free
    // modes exactly, making the null space exact and requiring the source to
    // be divergence-cleaned); the penalty sign regularizes instead.
    AddGradDivZTerm(rhs_vec[lev], Zscratch, lev);

    // Calculate dA = ∇ x Z (ComputeCurlB resets dA_fp to zero internally).
    // Use Zscratch (guard cells already filled above) rather than Zvec directly.
    m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeCurlB(
        dA_fp[lev], Zscratch, m_WarpX->GetEBUpdateEFlag()[lev], lev
    );

    // include guard cells. dA_fp is E-staggered: use FillBoundaryAndSync so
    // periodically wrapped cells agree before ApplySusceptibility reads dA_fp below.
    for (int ii = 0; ii < 3; ii++)
    {
        dA_fp[lev][ii]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
        // clear E_temp since ApplySusceptibility accumulates into its rhs argument
        E_temp[lev][ii]->setVal(0);
    }

    // At conducting (PEC) walls, fill dA's guard cells with the E-type image
    // symmetry before the band product below reads them: the guard-cell S
    // deposits multiply these beyond-wall dA values, and the mixed-staggering
    // blocks (S_xz etc.) only fold with the correct sign when the ghost dA
    // holds the image values.
    ApplyPECtodA(dA_fp[lev], lev, dA_fp[lev][0]->nGrowVect());

    // Calculate chi dA and write into E_temp
    ApplySusceptibility(E_temp, dA_fp);

    // E_temp (Efield_fp) shares dA_fp's staggering (nodal transverse
    // components) - sync it too before ComputeCurlA reads it with a stencil.
    for (int ii = 0; ii < 3; ii++)
    {
        E_temp[lev][ii]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
    }

    // Reuse lapZ as a temporary storage location for the ∇ x E_temp = ∇ x (chi ∇ x Z_vec)
    m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(
        lapZ, E_temp[lev], m_WarpX->GetEBUpdateBFlag()[lev], lev
    );

    for (int ii = 0; ii < 3; ii++)
    {
        amrex::MultiFab::Add(*rhs_vec[lev][ii], *lapZ[ii], 0, 0, ncomps, 0);
    }

    // rhs_vec is the operator's own output (B-staggered, matching Z).
    // Nothing guarantees the stencil evaluations above produced identical
    // values at the duplicate periodic-image cells of the nodal
    // component(s), and GMRES's own linComb/increment arithmetic (used to
    // build every subsequent Krylov vector from this result) is
    // element-wise and has no notion of that duplication - so reconcile it
    // here before handing the result back.
    for (int ii = 0; ii < 3; ii++)
    {
        rhs_vec[lev][ii]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
    }

    // Project the operator output at conducting (PEC) walls: zero the
    // constrained wall-normal rows so they stay trivially satisfied for
    // every Krylov vector (rhs_vec has no guard cells, so ng must be 0
    // here - only the on-wall zeroing is performed).
    ApplyPECtoZ(rhs_vec[lev], lev, amrex::IntVect(0));

    // Remove the (analytically zero, numerically roundoff-level) projection
    // of the operator output onto the exact constant null vectors, so the
    // Krylov space stays orthogonal to them (their component in the source
    // is removed in CalculateSourceVector and can never be reduced).
    ProjectOutNullConstants(rhs_vec[lev]);
}

void SemiImplicitDarwin::PrepareCurrentDeposition ()
{
    BL_PROFILE("SemiImplicitDarwin::PrepareCurrentDeposition()");
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
        centered gamma) - was fixed in 89a8fc3.
    */

    BL_PROFILE("SemiImplicitDarwin::AccumulateCurrentAndSusceptibility()");

    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    const int lev = 0;

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
    Sxx->setVal(0.0);
    Sxy->setVal(0.0);
    Sxz->setVal(0.0);
    Syx->setVal(0.0);
    Syy->setVal(0.0);
    Syz->setVal(0.0);
    Szx->setVal(0.0);
    Szy->setVal(0.0);
    Szz->setVal(0.0);

    // Deposit the current density from all species, using the time-centered
    // particle velocities as appropriate for the implicit push. This also
    // resets the current MultiFabs before depositing.
    m_WarpX->GetPartContainer().DepositCurrent(
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, lev),
        m_dt, 0.0_rt, PushType::Implicit);

    for (auto const& pc : m_WarpX->GetPartContainer()) {
        pc->DepositMassMatrices(m_WarpX->m_fields, lev, m_dt);
    }

    // Sync current (filter and sum boundaries)
    m_WarpX->SyncCurrent("current_fp");

    // At conducting (PEC) walls, fold the guard-cell current deposited by
    // particles near the wall back into the domain (the image-charge
    // contribution) and set image guard values, which curl(J) in
    // CalculateSourceVector() reads. No-op for periodic boundaries.
    m_WarpX->ApplyJfieldBoundary(
        lev,
        m_WarpX->m_fields.get(FieldType::current_fp, Direction{0}, lev),
        m_WarpX->m_fields.get(FieldType::current_fp, Direction{1}, lev),
        m_WarpX->m_fields.get(FieldType::current_fp, Direction{2}, lev),
        PatchType::fine);

    // Sum boundaries for mass matrices.
    // NOTE: at conducting (PEC) walls the mass matrices are deliberately NOT
    // folded or image-filled: SyncMassMatrices() leaves the raw guard-cell S
    // deposits in place (guards at physical walls have no communication
    // partner), and ApplySusceptibility() consumes them directly by
    // evaluating chi*dA over the grown boxes and folding the resulting
    // E-staggered product with ApplyJfieldBoundary. Folding S itself would
    // require remapping its band component indices under reflection.
    m_WarpX->SyncMassMatrices();

    // The deposit routine only fills half of each diagonal mass matrix's
    // band (exploiting symmetry); mirror the other half back in now that
    // deposition and boundary summation are complete.
    FinishMassMatrices();

    // Refresh the grad-div penalty coefficient (see AddGradDivZTerm) from
    // the freshly deposited mass matrices: the diagonal band row-sum is the
    // local chi the curl(chi curl Z) term applies to a uniform dA, so its
    // domain mean (averaged over the three diagonal blocks) scaled by the
    // same 2 mu0/dt prefactor gives the mean solenoidal-mode chi scale.
    // The coefficient must be spatially constant: a varying alpha would
    // couple the curl-free and solenoidal subspaces and change dA.
    {
        const amrex::MultiFab* Sdiag[3] = {Sxx, Syy, Szz};
        amrex::Real chi_sum[3];
        for (int d = 0; d < 3; ++d) {
            amrex::Real s_sum = 0.0_rt;
            for (int c = 0; c < Sdiag[d]->nComp(); ++c) {
                s_sum += Sdiag[d]->sum(c, true);
            }
            chi_sum[d] = s_sum;
        }
        amrex::ParallelDescriptor::ReduceRealSum(chi_sum, 3);
        amrex::Real chi_mean = 0.0_rt;
        for (int d = 0; d < 3; ++d) {
            chi_mean += chi_sum[d] / static_cast<amrex::Real>(Sdiag[d]->boxArray().numPts());
        }
        m_graddiv_alpha =
            m_graddiv_factor * 2.0_rt * PhysConst::mu0 / m_dt * chi_mean / 3.0_rt;
    }
}

void SemiImplicitDarwin::CalculateSourceVector ()
{
    // This function calculates the "b" vector for the linear MS equation,
    // i.e., the source vector.
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

    // Calculate 2 * ∇^2 B + 2 * mu_0 ∇ x J and write result in m_source
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

    // Project the source at conducting (PEC) walls: zero the constrained
    // wall-normal rows so b lies exactly in the range of the projected
    // operator applied by ComputeRHS (deposition noise would otherwise
    // leave a residual there that GMRES can never reduce).
    ApplyPECtoZ(b[lev], lev, amrex::IntVect(0));

    // Remove the source's projection onto the exact constant null vectors
    // of the wall-projected operator (with PEC walls, curl(J)'s domain sum
    // telescopes to the tangential wall currents, which are physically
    // nonzero with reflecting particles): that component is irreducible
    // residual GMRES can never remove, and being constant it has zero curl,
    // so removing it cannot change dA = curl(Z).
    ProjectOutNullConstants(b[lev]);
}

void SemiImplicitDarwin::UpdateEfromdA ( int astep )
{
    // This function updates the Efield_fp MF to hold the new inductive E-field.
    BL_PROFILE("SemiImplicitDarwin::UpdateEfromdA()");

    const int lev = 0;

    // Grab the E-field MultiFabs
    ablastr::fields::MultiLevelVectorField Efield = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, lev);

    // Grab the dA_fp MultiFabs to store dA = ∇ x Z (the solved-for Z lives
    // on B's staggering; dA lives on A/E's staggering)
    ablastr::fields::MultiLevelVectorField dAfield = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::dA_fp, lev);

    // Grab m_Z MultiFabs (the solved-for Z). GMRES builds its final answer via
    // linComb/increment-style arithmetic, which only touches the valid region,
    // so Zfield's own guard cells cannot be trusted here either - copy into a
    // scratch and FillBoundary on that, same as in ComputeRHS.
    // Zscratch is a local MultiFab, not m_Z's own storage, so its ghost width
    // isn't tied to m_Z's own native ghost width (which is 0, by design - see
    // note above) - the ComputeCurlB stencil below reads i-1, so at least 1
    // ghost cell is requested here regardless.
    const auto& Zfield = m_Z.getArrayVec();
    const amrex::IntVect curl_ng = amrex::elemwiseMax(Zfield[lev][0]->nGrowVect(), amrex::IntVect(1));
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

    // At conducting (PEC) walls, fill Zscratch's guard cells with the B-type
    // image symmetry so the curl below produces dA with tangential
    // components that vanish at the wall (the solved-for Z was projected,
    // but its guard cells must hold the image values for the stencil).
    ApplyPECtoZ(Zscratch, lev, curl_ng);

    // Calculate dA = curl(Z)
    m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeCurlB(
        dAfield[lev], Zscratch, m_WarpX->GetEBUpdateEFlag()[lev], lev
    );
    for (int ii = 0; ii < 3; ii++)
    {
        dAfield[lev][ii]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
    }

    // At conducting (PEC) walls, set dA's image guard values (and re-zero
    // the on-wall tangential values against stencil roundoff) so the
    // ghost-inclusive copy into Efield_fp below carries consistent data.
    ApplyPECtodA(dAfield[lev], lev, dAfield[lev][0]->nGrowVect());

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

void SemiImplicitDarwin::ApplySusceptibility (
    ablastr::fields::MultiLevelVectorField& rhs,
    const ablastr::fields::MultiLevelVectorField& dA )
{
    BL_PROFILE("SemiImplicitDarwin::ApplySusceptibility()");
    // This function applies the susceptibility matrices to the given dA.
    // The functionality is copied from the ``ImplicitSolver::ComputeJfromMassMatrices``
    // function rather than calling it directly, since that function is hardcoded to
    // the registered current_fp/Efield_fp/Efield_fp_save fields, whereas this one
    // must operate on the caller-supplied rhs/dA vectors used inside the GMRES matvec.

    using namespace amrex::literals;

    using warpx::fields::FieldType;
    using ablastr::fields::Direction;

    const int ncomps = 1;
    const int finest_level = 0;
    const auto dt = m_dt;

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

            // The outer loop below reads Sxx/Sxy/Sxz (etc.) directly at (i,j,k),
            // so it must stay within the mass matrices' own ghost region - grow
            // by the min of dA's and the mass matrices' ghost widths (dA_fp can
            // have more ghost cells than current_fp/the mass matrices, since
            // Efield's ghost width only has to be >= current's, not equal).
            amrex::Box dAbx = amrex::convert(mfi.validbox(),dA[lev][0]->ixType());
            amrex::Box dAby = amrex::convert(mfi.validbox(),dA[lev][1]->ixType());
            amrex::Box dAbz = amrex::convert(mfi.validbox(),dA[lev][2]->ixType());
            dAbx.grow(amrex::elemwiseMin(dA[lev][0]->nGrowVect(), SX[0]->nGrowVect()));
            dAby.grow(amrex::elemwiseMin(dA[lev][1]->nGrowVect(), SY[1]->nGrowVect()));
            dAbz.grow(amrex::elemwiseMin(dA[lev][2]->nGrowVect(), SZ[2]->nGrowVect()));

            // The inner stencil reads of dAx/dAy/dAz, however, are bounded by
            // dA's own (potentially wider) ghost region, which holds correct
            // periodic-wrapped data via FillBoundaryAndSync. Clamping those
            // reads to dAbx/dAby/dAbz above (the S-limited box) would silently
            // truncate the stencil near the domain edges whenever dA has more
            // ghost cells than the mass matrices - dropping legitimate
            // wraparound contributions there, not just avoiding OOB reads.
            amrex::Box dA_fullbx = amrex::convert(mfi.validbox(),dA[lev][0]->ixType());
            amrex::Box dA_fullby = amrex::convert(mfi.validbox(),dA[lev][1]->ixType());
            amrex::Box dA_fullbz = amrex::convert(mfi.validbox(),dA[lev][2]->ixType());
            dA_fullbx.grow(dA[lev][0]->nGrowVect());
            dA_fullby.grow(dA[lev][1]->nGrowVect());
            dA_fullbz.grow(dA[lev][2]->nGrowVect());

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
                    index_min[dim] = std::max(-offset_xx[dim],dA_fullbx.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_xx[dim]-1-offset_xx[dim],dA_fullbx.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SxxdAx = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_xx[0],
                                   + ncomp_xx[0]*( jj+offset_xx[1] ),
                                   + ncomp_xx[0]*ncomp_xx[1]*( kk+offset_xx[2] ) );
                            SxxdAx += Sxx(i,j,k,Nc)*dAx(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                // Compute Sxy*dAy
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_xy[dim],dA_fullby.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_xy[dim]-1-offset_xy[dim],dA_fullby.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SxydAy = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_xy[0],
                                   + ncomp_xy[0]*( jj+offset_xy[1] ),
                                   + ncomp_xy[0]*ncomp_xy[1]*( kk+offset_xy[2] ) );
                            SxydAy += Sxy(i,j,k,Nc)*dAy(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                // Compute Sxz*dAz
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_xz[dim],dA_fullbz.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_xz[dim]-1-offset_xz[dim],dA_fullbz.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SxzdAz = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_xz[0],
                                   + ncomp_xz[0]*( jj+offset_xz[1] ),
                                   + ncomp_xz[0]*ncomp_xz[1]*( kk+offset_xz[2] ) );
                            SxzdAz += Sxz(i,j,k,Nc)*dAz(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                Fx(i,j,k,n) += 2._prt * PhysConst::mu0 / dt * (SxxdAx + SxydAy + SxzdAz);
            });
            amrex::ParallelFor(
            dAby, ncomps, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                const int idx[3] = {i, j, k};
                amrex::GpuArray<int, 3> index_min = {0, 0, 0};
                amrex::GpuArray<int, 3> index_max = {0, 0, 0};

                // Compute Syx*dAx
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_yx[dim],dA_fullbx.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_yx[dim]-1-offset_yx[dim],dA_fullbx.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SyxdAx = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_yx[0],
                                   + ncomp_yx[0]*( jj+offset_yx[1] ),
                                   + ncomp_yx[0]*ncomp_yx[1]*( kk+offset_yx[2] ) );
                            SyxdAx += Syx(i,j,k,Nc)*dAx(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                // Compute Syy*dAy
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_yy[dim],dA_fullby.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_yy[dim]-1-offset_yy[dim],dA_fullby.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SyydAy = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_yy[0],
                                   + ncomp_yy[0]*( jj+offset_yy[1] ),
                                   + ncomp_yy[0]*ncomp_yy[1]*( kk+offset_yy[2] ) );
                            SyydAy += Syy(i,j,k,Nc)*dAy(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                // Compute Syz*dAz
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_yz[dim],dA_fullbz.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_yz[dim]-1-offset_yz[dim],dA_fullbz.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SyzdAz = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_yz[0],
                                   + ncomp_yz[0]*( jj+offset_yz[1] ),
                                   + ncomp_yz[0]*ncomp_yz[1]*( kk+offset_yz[2] ) );
                            SyzdAz += Syz(i,j,k,Nc)*dAz(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                Fy(i,j,k,n) += 2._prt * PhysConst::mu0 / dt * (SyxdAx + SyydAy + SyzdAz);
            });
            amrex::ParallelFor(
            dAbz, ncomps, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                const int idx[3] = {i, j, k};
                amrex::GpuArray<int, 3> index_min = {0, 0, 0};
                amrex::GpuArray<int, 3> index_max = {0, 0, 0};

                // Compute Szx*dAx
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_zx[dim],dA_fullbx.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_zx[dim]-1-offset_zx[dim],dA_fullbx.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SzxdAx = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_zx[0],
                                   + ncomp_zx[0]*( jj+offset_zx[1] ),
                                   + ncomp_zx[0]*ncomp_zx[1]*( kk+offset_zx[2] ) );
                            SzxdAx += Szx(i,j,k,Nc)*dAx(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                // Compute Szy*dAy
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_zy[dim],dA_fullby.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_zy[dim]-1-offset_zy[dim],dA_fullby.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SzydAy = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_zy[0],
                                   + ncomp_zy[0]*( jj+offset_zy[1] ),
                                   + ncomp_zy[0]*ncomp_zy[1]*( kk+offset_zy[2] ) );
                            SzydAy += Szy(i,j,k,Nc)*dAy(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                // Compute Szz*dAz
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_zz[dim],dA_fullbz.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_zz[dim]-1-offset_zz[dim],dA_fullbz.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SzzdAz = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_zz[0],
                                   + ncomp_zz[0]*( jj+offset_zz[1] ),
                                   + ncomp_zz[0]*ncomp_zz[1]*( kk+offset_zz[2] ) );
                            SzzdAz += Szz(i,j,k,Nc)*dAz(i+ii,j+jj,k+kk,n);
                        }
                    }
                }

                Fz(i,j,k,n) += 2._prt * PhysConst::mu0 / dt * (SzxdAx + SzydAy + SzzdAz);
            });
        }

        // At conducting (PEC) walls, fold the guard-cell chi*dA products
        // (accumulated above from the raw guard-cell S deposits and the
        // image-filled dA values) back into the domain and set image guard
        // values. This must precede the FillBoundaryAndSync below so
        // neighboring boxes' guard cells pick up the post-fold valid values.
        // No-op for periodic boundaries.
        //
        // Deliberately call PEC::ApplyReflectiveBoundarytoJfield directly
        // here instead of the WarpX::ApplyJfieldBoundary wrapper used for the
        // real deposited current below: that wrapper reads the *particle*
        // boundary type and switches to a PMC-like fold sign whenever
        // particles are Reflecting/Thermal (correct for actual deposited
        // current, which really does come from bouncing particles). The
        // chi*dA product folded here is a purely field-space intermediate in
        // the GMRES matvec with no notion of particle reflection, so it must
        // always use the field-boundary-only PEC convention - pass
        // Absorbing unconditionally so the sign is never coupled to whatever
        // particle boundary the user configured for real particles.
        {
            const amrex::Array<ParticleBoundaryType, AMREX_SPACEDIM> no_particle_reflection =
                {{AMREX_D_DECL(ParticleBoundaryType::Absorbing,
                               ParticleBoundaryType::Absorbing,
                               ParticleBoundaryType::Absorbing)}};
            PEC::ApplyReflectiveBoundarytoJfield(
                rhs[lev][0], rhs[lev][1], rhs[lev][2],
                m_WarpX->GetFieldBoundaryLo(), m_WarpX->GetFieldBoundaryHi(),
                no_particle_reflection, no_particle_reflection,
                m_WarpX->Geom(lev), lev, PatchType::fine, m_WarpX->refRatio());
        }

        // Fill and sync guard cells & edges
        rhs[lev][0]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
        rhs[lev][1]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
        rhs[lev][2]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
    }
}

void SemiImplicitDarwin::ApplyPECtoZ ( const ablastr::fields::VectorField& a_field,
                                       int lev, const amrex::IntVect& ng )
{
    if (!m_has_pec) { return; }
    // Z shares B's staggering, so the standard B-type PEC image symmetry
    // (tangential components even-mirrored, normal components odd-mirrored
    // and zeroed on the wall plane) is exactly what makes dA = curl(Z)
    // satisfy the E-type PEC symmetry (tangential dA = 0 at the wall).
    PEC::ApplyPECtoBfield(
        a_field,
        m_WarpX->GetFieldBoundaryLo(), m_WarpX->GetFieldBoundaryHi(),
        FieldBoundaryType::PEC, ng,
        m_WarpX->Geom(lev), lev, PatchType::fine, m_WarpX->refRatio());
}

void SemiImplicitDarwin::ApplyPECtodA ( const ablastr::fields::VectorField& a_field,
                                        int lev, const amrex::IntVect& ng )
{
    if (!m_has_pec) { return; }
    // dA shares E's staggering (E = -dA/dt), so the standard E-type PEC
    // image symmetry applies directly.
    PEC::ApplyPECtoEfield(
        a_field,
        m_WarpX->GetFieldBoundaryLo(), m_WarpX->GetFieldBoundaryHi(),
        FieldBoundaryType::PEC, ng,
        m_WarpX->Geom(lev), lev, PatchType::fine, m_WarpX->refRatio());
}

void SemiImplicitDarwin::AddGradDivZTerm ( const ablastr::fields::VectorField& a_rhs,
                                           const ablastr::fields::VectorField& a_Z,
                                           int lev )
{
    BL_PROFILE("SemiImplicitDarwin::AddGradDivZTerm()");

    if (m_graddiv_alpha == 0.0_rt) { return; }

#if defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D)
    // The B-staggered Z is nodal exactly in each component's own direction,
    // so div(Z) built from the "Upward" (node-pair) difference of each
    // component lands fully cell-centered, and the "Downward" (cell-pair)
    // gradient of that scalar lands back on Z's staggering. This matched
    // Upward/Downward pair makes the term symmetric positive semidefinite
    // (<-grad(alpha div Z), Z> = alpha ||div Z||^2) and, since the Downward
    // gradient commutes with the curl stencils, exactly curl-free.
    const auto dxinv = m_WarpX->Geom(lev).InvCellSizeArray();
    const amrex::Real alpha = m_graddiv_alpha;

    // No tiling: div(Z) is evaluated on each FAB's grown box (into m_divZ's
    // own guard cells, reading a_Z's image/periodic-filled guards at
    // distance <= 2) and immediately consumed by the gradient within the
    // same FAB, avoiding an intermediate boundary fill. Growing a *tile*
    // box instead would make neighboring tiles write overlapping cells.
    // At conducting walls a_Z's guards hold the B-type image values, so the
    // guard div(Z) is automatically the even mirror of the interior value
    // and the wall-normal gradient vanishes on the wall plane.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(m_divZ, false); mfi.isValid(); ++mfi )
    {
        amrex::Array4<amrex::Real> const& divZ = m_divZ.array(mfi);

        amrex::Array4<const amrex::Real> const& Zx = a_Z[0]->const_array(mfi);
        amrex::Array4<const amrex::Real> const& Zy = a_Z[1]->const_array(mfi);
        amrex::Array4<const amrex::Real> const& Zz = a_Z[2]->const_array(mfi);

        amrex::Array4<amrex::Real> const& Fx = a_rhs[0]->array(mfi);
        amrex::Array4<amrex::Real> const& Fy = a_rhs[1]->array(mfi);
        amrex::Array4<amrex::Real> const& Fz = a_rhs[2]->array(mfi);

        const amrex::Box gbx = amrex::grow(mfi.validbox(), 1);

#if defined(WARPX_DIM_1D_Z)
        const amrex::Real dzi = dxinv[0];

        amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            divZ(i,j,k) = (Zz(i+1,j,k) - Zz(i,j,k))*dzi;
        });

        const amrex::Box tbz = amrex::convert(mfi.validbox(), a_rhs[2]->ixType());
        amrex::ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Fz(i,j,k) -= alpha*(divZ(i,j,k) - divZ(i-1,j,k))*dzi;
        });
        amrex::ignore_unused(Zx, Zy, Fx, Fy);
#elif defined(WARPX_DIM_XZ)
        const amrex::Real dxi = dxinv[0];
        const amrex::Real dzi = dxinv[1];

        amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            divZ(i,j,k) = (Zx(i+1,j,k) - Zx(i,j,k))*dxi
                        + (Zz(i,j+1,k) - Zz(i,j,k))*dzi;
        });

        const amrex::Box tbx = amrex::convert(mfi.validbox(), a_rhs[0]->ixType());
        const amrex::Box tbz = amrex::convert(mfi.validbox(), a_rhs[2]->ixType());
        amrex::ParallelFor(tbx, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Fx(i,j,k) -= alpha*(divZ(i,j,k) - divZ(i-1,j,k))*dxi;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Fz(i,j,k) -= alpha*(divZ(i,j,k) - divZ(i,j-1,k))*dzi;
            }
        );
        // The out-of-plane (y) component contributes nothing to div(Z) and
        // receives no in-plane gradient component.
        amrex::ignore_unused(Zy, Fy);
#elif defined(WARPX_DIM_3D)
        const amrex::Real dxi = dxinv[0];
        const amrex::Real dyi = dxinv[1];
        const amrex::Real dzi = dxinv[2];

        amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            divZ(i,j,k) = (Zx(i+1,j,k) - Zx(i,j,k))*dxi
                        + (Zy(i,j+1,k) - Zy(i,j,k))*dyi
                        + (Zz(i,j,k+1) - Zz(i,j,k))*dzi;
        });

        const amrex::Box tbx = amrex::convert(mfi.validbox(), a_rhs[0]->ixType());
        const amrex::Box tby = amrex::convert(mfi.validbox(), a_rhs[1]->ixType());
        const amrex::Box tbz = amrex::convert(mfi.validbox(), a_rhs[2]->ixType());
        amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Fx(i,j,k) -= alpha*(divZ(i,j,k) - divZ(i-1,j,k))*dxi;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Fy(i,j,k) -= alpha*(divZ(i,j,k) - divZ(i,j-1,k))*dyi;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Fz(i,j,k) -= alpha*(divZ(i,j,k) - divZ(i,j,k-1))*dzi;
            }
        );
#endif
    }
#else
    amrex::ignore_unused(a_rhs, a_Z, lev);
    WARPX_ABORT_WITH_MESSAGE(
        "SemiImplicitDarwin::AddGradDivZTerm is only implemented for the "
        "Cartesian 1D/2D/3D geometries.");
#endif
}

void SemiImplicitDarwin::ProjectOutNullConstants ( const ablastr::fields::VectorField& a_field )
{
    BL_PROFILE("SemiImplicitDarwin::ProjectOutNullConstants()");

    if (!m_has_pec) { return; }

    for (int comp = 0; comp < 3; ++comp)
    {
        if (!m_null_constant_comp[comp]) { continue; }
        amrex::MultiFab& mf = *a_field[comp];
        // Both sum() and numPts() count nodal points shared between boxes
        // once per box, consistently; the residual projection error from
        // that shared-point weighting is O(1/N) of the removed mean.
        const amrex::Real mean =
            mf.sum(0, false) / static_cast<amrex::Real>(mf.boxArray().numPts());
        mf.plus(-mean, 0, 1, 0);
    }
}
