/* Copyright 2024 Justin Angus
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Fields.H"
#include "ThetaImplicitEM.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "WarpX.H"

using warpx::fields::FieldType;
using namespace amrex::literals;

void ThetaImplicitEM::Define ( WarpX* const  a_WarpX )
{
    BL_PROFILE("ThetaImplicitEM::Define()");

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

    // Initialize the mass matrices for plasma response
    if (m_use_mass_matrices) { InitializeMassMatrices(); }

    const PreconditionerType pc_type = m_nlsolver->GetPreconditionerType();
    if (pc_type == PreconditionerType::pc_petsc) { InitializeCurlCurlBCMasks(); }

    m_is_defined = true;

}

void ThetaImplicitEM::PrintParameters () const
{
    BL_PROFILE("ThetaImplicitEM::PrintParameters()");

    if (!m_WarpX->Verbose()) { return; }
    amrex::Print() << "\n";
    amrex::Print() << "-----------------------------------------------------------\n";
    amrex::Print() << "----------- THETA IMPLICIT EM SOLVER PARAMETERS -----------\n";
    amrex::Print() << "-----------------------------------------------------------\n";
    amrex::Print() << "Time-bias parameter theta:           " << m_theta << "\n";
    PrintBaseImplicitSolverParameters();
    m_nlsolver->PrintParams();
    amrex::Print() << "-----------------------------------------------------------\n\n";
}

void ThetaImplicitEM::OneStep ( const amrex::Real  start_time,
                                const amrex::Real  a_dt,
                                const int          a_step )
{
    BL_PROFILE("ThetaImplicitEM::OneStep()");

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
    m_nlsolver->Solve( m_E, m_Eold, start_time, m_dt, a_step );

    // Update WarpX owned Efield_fp and Bfield_fp to t_{n+theta}
    UpdateWarpXFields( m_E, start_time );
    m_WarpX->reduced_diags->ComputeDiagsMidStep(a_step);

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
    BL_PROFILE("ThetaImplicitEM::ComputeRHS()");

    // Update WarpX-owned Efield_fp and Bfield_fp using current state of
    // Eg from the nonlinear solver at time n+theta
    UpdateWarpXFields( a_E, start_time );

    // Update particle positions and velocities using the current state
    // of Eg and Bg. Deposit current density at time n+1/2
    const amrex::Real theta_time = start_time + m_theta*m_dt;
    PreRHSOp( theta_time, a_nl_iter, a_from_jacobian );

    // RHS = cvac^2*m_theta*dt*( curl(Bg^{n+theta}) - mu0*Jg^{n+1/2} )
    m_WarpX->ImplicitComputeRHSE( m_theta*m_dt, a_RHS);
}

void ThetaImplicitEM::UpdateWarpXFields ( const WarpXSolverVec&  a_E,
                                          amrex::Real start_time )
{
    BL_PROFILE("ThetaImplicitEM::UpdateWarpXFields()");

    // Update Efield_fp owned by WarpX
    const amrex::Real theta_time = start_time + m_theta*m_dt;
    m_WarpX->SetElectricFieldAndApplyBCs( a_E, theta_time );

    // Update Bfield_fp owned by WarpX
    ablastr::fields::MultiLevelVectorField const& B_old = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, 0);
    m_WarpX->UpdateMagneticFieldAndApplyBCs( B_old, m_theta*m_dt, start_time );

}

void ThetaImplicitEM::FinishFieldUpdate ( amrex::Real end_time )
{
    BL_PROFILE("ThetaImplicitEM::FinishFieldUpdate()");

    // Eg^{n+1} = (1/theta)*Eg^{n+theta} + (1-1/theta)*Eg^n
    // Bg^{n+1} = (1/theta)*Bg^{n+theta} + (1-1/theta)*Bg^n

    const amrex::Real c0 = 1._rt/m_theta;
    const amrex::Real c1 = 1._rt - c0;
    m_E.linComb( c0, m_E, c1, m_Eold );
    m_WarpX->SetElectricFieldAndApplyBCs( m_E, end_time );
    ablastr::fields::MultiLevelVectorField const & B_old = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, 0);
    m_WarpX->FinishMagneticFieldAndApplyBCs( B_old, m_theta, end_time );

}

const amrex::MultiFab* ThetaImplicitEM::GetCurl2BCmask (const int lev, const int field_dir) const
{
    using ablastr::fields::Direction;
    const amrex::MultiFab* mask = m_WarpX->m_fields.get(FieldType::curl2_BC_mask, Direction{field_dir}, lev);
    return mask;
}

void ThetaImplicitEM::InitializeCurlCurlBCMasks ()
{

    // Define masks to incorporate boundary conditions into the curl curl operator matrix
    /*
    The curl curl operator:
    3D: xhat\cdot[\nabla\times\nabla E] = d/dx[dEy/dy + dEz/dz] - [d2/dy2 + d2/dz2]Ex
        yhat\cdot[\nabla\times\nabla E] = d/dy[dEx/dx + dEz/dz] - [d2/dx2 + d2/dz2]Ey
        zhat\cdot[\nabla\times\nabla E] = d/dz[dEx/dx + dEy/dy] - [d2/dx2 + d2/dy2]Ez
    2D: xhat\cdot[\nabla\times\nabla E] = d/dx[dEz/dz] - [d2/dz2]Ex
        yhat\cdot[\nabla\times\nabla E] = - [d2/dx2 + d2/dz2]Ey
        zhat\cdot[\nabla\times\nabla E] = d/dz[dEx/dx] - [d2/dx2]Ez
    1D: xhat\cdot[\nabla\times\nabla E] = - [d2/dz2]Ex
        yhat\cdot[\nabla\times\nabla E] = - [d2/dz2]Ey
        zhat\cdot[\nabla\times\nabla E] = 0

    In general, one mask is needed for each derivative in each component of the operator.
    However, for a second order Yee grid where E lives on cell edges, no masks are
    required for terms that do not use ghost cell values, such as dEi/di for i = x, y, z.
    1D: The out-of-line x- and y-components have two masks. The first is for the diagonal term of
        the 2nd derivative operator and the second mask is for the off-diagonal term.
        No masks are needed for the in-line z-component of the curl curl operator.
    2D: There are three masks for each of the in-plane x- and z-components. The first two
        are for the diagonal, and off-diagonal terms of the 2nd derivative operator, respectively.
        The third term is for the cross term. The out-of-plane y-component requires four masks.
        The first two are for the d2/dx2 operator, and the last two are for the d2/dz2 operator.
    3D: Six masks are required for each component of the curl-curl operator in 3D.
        For the x-component, the first two are for the d2/dy2 operator, the second two are for
        the d2/dz2 operator, the fifth term is for the d/dx[Ey] operator, and the sixth term
        is for the d/dx[Ez] operator. For the y- and z-components, just permutate the indices
        to the right in a cyclic fashion.

    Example: Consider the second derivative operator at a symmetry boundary at the lower-boundary.
        In this case, the operator transforms as [1 -2 1] ==> [0 -2 2]. In terms of the masks,
        the mask for the diagonal term is 1 and the mask for the off-diagonal term is 2.
    */

    using ablastr::fields::Direction;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const amrex::IntVect ghosts = amrex::IntVect{0};
        const auto& dm = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{1}, lev)->DistributionMap();
        const auto& ba_Ex = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, lev)->boxArray();
        const auto& ba_Ey = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{1}, lev)->boxArray();
        const auto& ba_Ez = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{2}, lev)->boxArray();
#if defined(WARPX_DIM_1D_Z)
        const int ncomps_Ex = 2;
        const int ncomps_Ey = 2;
        const int ncomps_Ez = 0;
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        const int ncomps_Ex = 0;
        const int ncomps_Ey = 2;
        const int ncomps_Ez = 2;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        const int ncomps_Ex = 3;
        const int ncomps_Ey = 4;
        const int ncomps_Ez = 3;
#elif defined(WARPX_DIM_3D)
        const int ncomps_Ex = 6;
        const int ncomps_Ey = 6;
        const int ncomps_Ez = 6;
#endif
        m_WarpX->m_fields.alloc_init(FieldType::curl2_BC_mask, Direction{0}, lev, ba_Ex, dm, ncomps_Ex, ghosts, 1.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::curl2_BC_mask, Direction{1}, lev, ba_Ey, dm, ncomps_Ey, ghosts, 1.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::curl2_BC_mask, Direction{2}, lev, ba_Ez, dm, ncomps_Ez, ghosts, 1.0_rt);
    }

    const int lev = 0;
    const amrex::Geometry& geom = GetGeometry(lev);
    amrex::Box domain_box = geom.Domain();
    domain_box.convert(amrex::IntVect::TheNodeVector());
    const amrex::IntVect domain_lo = domain_box.smallEnd();
    const amrex::IntVect domain_hi = domain_box.bigEnd();

    const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& bc_type_lo = GetFieldBoundaryLo();
    const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& bc_type_hi = GetFieldBoundaryHi();

    ablastr::fields::VectorField curl2_BC_mask = m_WarpX->m_fields.get_alldirs(FieldType::curl2_BC_mask, lev);
#if AMREX_SPACEDIM < 3
    // Set the BC masks for the out-of-plane components of the curl curl E operator
    for (amrex::MFIter mfi(*curl2_BC_mask[1], false); mfi.isValid(); ++mfi) {

        // Get nodal box that does not include ghost cells
        const amrex::Box node_box = amrex::convert(mfi.validbox(),amrex::IntVect::TheNodeVector());

        for (int bdry_dir = 0; bdry_dir < AMREX_SPACEDIM; ++bdry_dir) {

            if (bc_type_lo[bdry_dir] == FieldBoundaryType::Periodic) { continue; }

            for (int bdry_side = 0; bdry_side < 2; ++bdry_side) {

                // Check if the box touches the boundary
                if (bdry_side == 0 && node_box.smallEnd()[bdry_dir] != domain_lo[bdry_dir]) {
                    continue;
                }
                if (bdry_side == 1 && node_box.bigEnd()[bdry_dir] != domain_hi[bdry_dir]) {
                    continue;
                }

                // Create a node box that only contains locations right on the boundary
                amrex::Box bdry_box = node_box;
                if (bdry_side == 0) { bdry_box.setBig(bdry_dir,domain_lo[bdry_dir]); }
                if (bdry_side == 1) { bdry_box.setSmall(bdry_dir,domain_hi[bdry_dir]); }

                // Set the BC-dependent mask values
                amrex::Real val0 = 1.0;
                amrex::Real val1 = 1.0;
                const FieldBoundaryType bc_type = (bdry_side == 0) ? bc_type_lo[bdry_dir]:bc_type_hi[bdry_dir];
                if (bc_type == FieldBoundaryType::PEC){
                    val0 = 0.0;
                    val1 = 0.0;
                }
                if (bc_type == FieldBoundaryType::PMC){
                    val0 = 1.0;
                    val1 = 2.0;
                }
                if (bc_type == FieldBoundaryType::Absorbing_SilverMueller) {
                    val0 = 0.5;
                    val1 = 1.0;
                }
                if (bc_type == FieldBoundaryType::PECInsulator) {
                    const int voltage_driven = m_WarpX->GetPECInsulator_IsESet(bdry_dir,bdry_side);
                    if (voltage_driven) { // Dirichlet boundary for E
                        val0 = 0.0;
                        val1 = 0.0;
                    }
                    else { // Dirichlet boundary for B
                        val0 = 0.5;
                        val1 = 1.0;
                    }
                }

                // Set mask values on the boundary cells (same for Ex and Ey for 1D_Z)
#if defined(WARPX_DIM_1D_Z)
                amrex::Array4<amrex::Real> const& maskEx_arr = curl2_BC_mask[0]->array(mfi);
#endif
                amrex::Array4<amrex::Real> const& maskEy_arr = curl2_BC_mask[1]->array(mfi);
                amrex::ParallelFor(bdry_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
#if defined(WARPX_DIM_1D_Z)
                    maskEx_arr(i,j,k,2*bdry_dir  ) = val0;
                    maskEx_arr(i,j,k,2*bdry_dir+1) = val1;
#endif
                    maskEy_arr(i,j,k,2*bdry_dir  ) = val0;
                    maskEy_arr(i,j,k,2*bdry_dir+1) = val1;
                });

            } // end loop over boundary sides

        } // end loop over boundary dirs

    } // end loop over boxes
#endif

#if AMREX_SPACEDIM > 1
    // Set the BC masks for the in-plane components of the curl curl E operator
    for (amrex::MFIter mfi(*curl2_BC_mask[0], false); mfi.isValid(); ++mfi) {

        for (int bdry_dir = 0; bdry_dir < AMREX_SPACEDIM; ++bdry_dir) {

            if (bc_type_lo[bdry_dir] == FieldBoundaryType::Periodic) { continue; }

            for (int field_dir = 0; field_dir < 3; ++field_dir) {

#if AMREX_SPACEDIM == 3
                if (field_dir == bdry_dir) { continue; }
                const int tdir1 = field_dir + 1 % AMREX_SPACEDIM; // next direction after field_dir
#else
                if (field_dir == 1) { continue; } // this is out-of-plane E in 2D
                if (bdry_dir == 0 && field_dir == 0) { continue; } // Ex is centered in bdry_dir = 0
                if (bdry_dir == 1 && field_dir == 2) { continue; } // Ez is centered in bdry_dir = 1
                const int tdir1 = bdry_dir;
#endif
                // Get edge box for Ecomp (nodal in bdry-direction) that does not include ghost cells
                const amrex::IntVect Edir_nodal = curl2_BC_mask[field_dir]->ixType().toIntVect();
                const amrex::Box edge_box = amrex::convert(mfi.validbox(),Edir_nodal);

                for (int bdry_side = 0; bdry_side < 2; ++bdry_side) {

                    // Check if the box touches the boundary
                    if (bdry_side == 0 && edge_box.smallEnd()[bdry_dir] != domain_lo[bdry_dir]) {
                        continue;
                    }
                    if (bdry_side == 1 && edge_box.bigEnd()[bdry_dir] != domain_hi[bdry_dir]) {
                        continue;
                    }

                    // Create a edge box that only contains locations right on the boundary
                    amrex::Box bdry_box = edge_box;
                    if (bdry_side == 0) { bdry_box.setBig(bdry_dir,domain_lo[bdry_dir]); }
                    if (bdry_side == 1) { bdry_box.setSmall(bdry_dir,domain_hi[bdry_dir]); }

                    // Set the BC-dependent mask values
                    amrex::Real val0 = 1.0;
                    amrex::Real val1 = 1.0;
                    amrex::Real val2 = 1.0;
                    const FieldBoundaryType bc_type = (bdry_side == 0) ? bc_type_lo[bdry_dir]:bc_type_hi[bdry_dir];
                    if (bc_type == FieldBoundaryType::PEC){
                        val0 = 0.0;
                        val1 = 0.0;
                        val2 = 0.0;
                    }
                    if (bc_type == FieldBoundaryType::PMC){
                        val0 = 1.0;
                        val1 = 2.0;
                        val2 = 2.0;
                    }
                    if (bc_type == FieldBoundaryType::PECInsulator) {
                        const int voltage_driven = m_WarpX->GetPECInsulator_IsESet(bdry_dir,bdry_side);
                        if (voltage_driven) { // Dirichlet boundary for E
                            val0 = 0.0;
                            val1 = 0.0;
                            val2 = 0.0;
                        }
                        else { // Dirichlet boundary for B
                            val0 = 0.5;
                            val1 = 1.0;
                            val2 = 1.0;
                        }
                    }

                    // Set mask values on the boundary cells
                    const int comp_shift = (tdir1 == bdry_dir) ? 0:3;
                    amrex::Array4<amrex::Real> const& mask_arr = curl2_BC_mask[field_dir]->array(mfi);
                    amrex::ParallelFor(bdry_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        mask_arr(i,j,k,comp_shift+0) = val0;
                        mask_arr(i,j,k,comp_shift+1) = val1;
                        mask_arr(i,j,k,comp_shift+2) = val2;
                    });

                } // end loop over boundary sides

            } // end loop over field dirs

        } // end loop over boundary dirs

    } // end loop over boxes
#endif

}
