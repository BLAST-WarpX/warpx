/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald, Arianna Formenti, Revathi Jambunathan
 *
 * License: BSD-3-Clause-LBNL
 */
#include "LabFrameExplicitES.H"
#include "Fluids/MultiFluidContainer_fwd.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "Particles/MultiParticleContainer_fwd.H"
#include "Python/callbacks.H"
#include "WarpX.H"

using namespace amrex;

void LabFrameExplicitES::InitData() {
    auto & warpx = WarpX::GetInstance();
    m_poisson_boundary_handler->DefinePhiBCs(warpx.Geom(0));
}

void LabFrameExplicitES::ComputeSpaceChargeField (
    ablastr::fields::MultiFabRegister& fields,
    MultiParticleContainer& mpc,
    MultiFluidContainer* mfl,
    int max_level)
{
    using ablastr::fields::MultiLevelScalarField;
    using ablastr::fields::MultiLevelVectorField;
    using warpx::fields::FieldType;

    bool const skip_lev0_coarse_patch = true;

    const MultiLevelScalarField rho_fp = fields.get_mr_levels(FieldType::rho_fp, max_level);
    const MultiLevelScalarField rho_cp = fields.get_mr_levels(FieldType::rho_cp, max_level, skip_lev0_coarse_patch);
    const MultiLevelScalarField phi_fp = fields.get_mr_levels(FieldType::phi_fp, max_level);
    const MultiLevelVectorField Efield_fp = fields.get_mr_levels_alldirs(FieldType::Efield_fp, max_level);

    mpc.DepositCharge(rho_fp, 0.0_rt);
    if (mfl) {
        const int lev = 0;
        mfl->DepositCharge(fields, *rho_fp[lev], lev);
    }

    // Apply filter, perform MPI exchange, interpolate across levels
    const Vector<std::unique_ptr<MultiFab> > rho_buf(num_levels);
    auto & warpx = WarpX::GetInstance();
    warpx.SyncRho( rho_fp, rho_cp, amrex::GetVecOfPtrs(rho_buf) );

#ifndef WARPX_DIM_RZ
    for (int lev = 0; lev < num_levels; lev++) {
        // Reflect density over PEC boundaries, if needed.
        warpx.ApplyRhofieldBoundary(lev, rho_fp[lev], PatchType::fine);
    }
#endif
    // beta is zero in lab frame
    // Todo: use simpler finite difference form with beta=0
    const std::array<Real, 3> beta = {0._rt};

    // set the boundary potentials appropriately
    setPhiBC(phi_fp, warpx.gett_new(0));

    // Compute the potential phi, by solving the Poisson equation
    if (IsPythonCallbackInstalled("poissonsolver")) {

        // Use the Python level solver (user specified)
        ExecutePythonCallback("poissonsolver");

    } else {

#if defined(WARPX_DIM_1D_Z)
        // Use the tridiag solver with 1D
        computePhiTriDiagonal(rho_fp, phi_fp);
#else
        // Use the AMREX MLMG or the FFT (IGF) solver otherwise
        computePhi(rho_fp, phi_fp, beta, self_fields_required_precision,
                   self_fields_absolute_tolerance, self_fields_max_iters,
                   self_fields_verbosity, is_igf_2d_slices, Efield_fp);
#endif

    }

    // Compute the electric field. Note that if an EB is used the electric
    // field will be calculated in the computePhi call.
    if (!EB::enabled()) { computeE( Efield_fp, phi_fp, beta ); }
    else {
        if (IsPythonCallbackInstalled("poissonsolver")) { computeE(Efield_fp, phi_fp, beta); }
    }
}

/* \brief Compute the potential by solving Poisson's equation with
          a 1D tridiagonal solve.

   \param[in] rho The charge density a given species
   \param[out] phi The potential to be computed by this function
*/
void LabFrameExplicitES::computePhiTriDiagonal (
    const ablastr::fields::MultiLevelScalarField& rho,
    const ablastr::fields::MultiLevelScalarField& phi)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(num_levels == 1,
    "The tridiagonal solver cannot be used with mesh refinement");

    auto field_boundary_lo0 = WarpX::field_boundary_lo[0];
    auto field_boundary_hi0 = WarpX::field_boundary_hi[0];

    if (field_boundary_lo0 == FieldBoundaryType::Periodic) {
        computePhiTriDiagonal_periodic(rho, phi);
        return;
    }

    const int lev = 0;
    auto & warpx = WarpX::GetInstance();

    const amrex::Real* dx = warpx.Geom(lev).CellSize();
    const amrex::Real xmin = warpx.Geom(lev).ProbLo(0);
    const amrex::Real xmax = warpx.Geom(lev).ProbHi(0);
    const int nx_full_domain = static_cast<int>( (xmax - xmin)/dx[0] + 0.5_rt );

    int nx_solve_min = 1;
    int nx_solve_max = nx_full_domain - 1;

    if (field_boundary_lo0 == FieldBoundaryType::Neumann) {
        // Solve for the point on the lower boundary
        nx_solve_min = 0;
    }
    if (field_boundary_hi0 == FieldBoundaryType::Neumann) {
        // Solve for the point on the upper boundary
        nx_solve_max = nx_full_domain;
    }

    // Create a 1-D MultiFab that covers all of x.
    // The tridiag solve will be done in this MultiFab and then copied out afterwards.
    const amrex::IntVect lo_full_domain(AMREX_D_DECL(0,0,0));
    const amrex::IntVect hi_full_domain(AMREX_D_DECL(nx_full_domain,0,0));
    const amrex::Box box_full_domain_node(lo_full_domain, hi_full_domain, amrex::IntVect::TheNodeVector());
    const BoxArray ba_full_domain_node(box_full_domain_node);
    const amrex::Vector<int> pmap = {0}; // The data will only be on processor 0
    const amrex::DistributionMapping dm_full_domain(pmap);

    // Put the data in the pinned arena since the tridiag solver will be done on the CPU, but have
    // the data readily accessible from the GPU.
    auto phi1d_mf = MultiFab(ba_full_domain_node, dm_full_domain, 1, 0, MFInfo().SetArena(The_Pinned_Arena()));
    auto zwork1d_mf = MultiFab(ba_full_domain_node, dm_full_domain, 1, 0, MFInfo().SetArena(The_Pinned_Arena()));
    auto rho1d_mf = MultiFab(ba_full_domain_node, dm_full_domain, 1, 0, MFInfo().SetArena(The_Pinned_Arena()));

    if (field_boundary_lo0 == FieldBoundaryType::PEC || field_boundary_hi0 == FieldBoundaryType::PEC) {
        // Copy from phi to get the boundary values
        phi1d_mf.ParallelCopy(*phi[lev], 0, 0, 1);
    }
    rho1d_mf.ParallelCopy(*rho[lev], 0, 0, 1);

    // Multiplier on the charge density
    const amrex::Real norm = dx[0]*dx[0]/PhysConst::epsilon_0;
    rho1d_mf.mult(norm);

    // Use the MFIter loop since when parallel, only process zero has a FAB.
    // This skips the loop on all other processors.
    for (MFIter mfi(phi1d_mf); mfi.isValid(); ++mfi) {

        const auto& phi1d_arr = phi1d_mf[mfi].array();
        const auto& zwork1d_arr = zwork1d_mf[mfi].array();
        const auto& rho1d_arr = rho1d_mf[mfi].array();

        // The loops are always performed on the CPU

        amrex::Real diag = 2._rt;

        // The initial values depend on the boundary condition
        if (field_boundary_lo0 == FieldBoundaryType::PEC) {

            phi1d_arr(1,0,0) = (phi1d_arr(0,0,0) + rho1d_arr(1,0,0))/diag;

        } else if (field_boundary_lo0 == FieldBoundaryType::Neumann) {

            // Neumann boundary condition
            phi1d_arr(0,0,0) = rho1d_arr(0,0,0)/diag;

            zwork1d_arr(1,0,0) = 2._rt/diag;
            diag = 2._rt - zwork1d_arr(1,0,0);
            phi1d_arr(1,0,0) = (rho1d_arr(1,0,0) - (-1._rt)*phi1d_arr(1-1,0,0))/diag;

        }

        // Loop upward, calculating the Gaussian elimination multipliers and right hand sides
        for (int i_up = 2 ; i_up < nx_solve_max ; i_up++) {

            zwork1d_arr(i_up,0,0) = 1._rt/diag;
            diag = 2._rt - zwork1d_arr(i_up,0,0);
            phi1d_arr(i_up,0,0) = (rho1d_arr(i_up,0,0) - (-1._rt)*phi1d_arr(i_up-1,0,0))/diag;

        }

        // The last value depend on the boundary condition
        if (field_boundary_hi0 == FieldBoundaryType::PEC) {

            int const nxm1 = nx_full_domain - 1;
            zwork1d_arr(nxm1,0,0) = 1._rt/diag;
            diag = 2._rt - zwork1d_arr(nxm1,0,0);
            phi1d_arr(nxm1,0,0) = (phi1d_arr(nxm1+1,0,0) + rho1d_arr(nxm1,0,0) - (-1._rt)*phi1d_arr(nxm1-1,0,0))/diag;

        } else if (field_boundary_hi0 == FieldBoundaryType::Neumann) {

            // Neumann boundary condition
            zwork1d_arr(nx_full_domain,0,0) = 1._rt/diag;
            diag = 2._rt - 2._rt*zwork1d_arr(nx_full_domain,0,0);
            if (diag == 0._rt) {
                // This happens if the lower boundary is also Neumann.
                // It this case, the potential is relative to an arbitrary constant,
                // so set the upper boundary to zero to force a value.
                phi1d_arr(nx_full_domain,0,0) = 0.;
            } else {
                phi1d_arr(nx_full_domain,0,0) = (rho1d_arr(nx_full_domain,0,0) - (-1._rt)*phi1d_arr(nx_full_domain-1,0,0))/diag;
            }

        }


        for (int i_down = nx_solve_max-1 ; i_down >= nx_solve_min ; i_down--) {
            phi1d_arr(i_down,0,0) = phi1d_arr(i_down,0,0) + zwork1d_arr(i_down+1,0,0)*phi1d_arr(i_down+1,0,0);
        }

    }

    // Copy phi1d to phi
    phi[lev]->ParallelCopy(phi1d_mf, 0, 0, 1);
}

/* \brief Compute the potential by solving Poisson's equation
 *        with periodic boundaries using
          a 1D tridiagonal solve.
          This makes use of the Sherman–Morrison formula.
          The code is based on the code given in the Wikipedia page,
          https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm#Variants.

   \param[in] rho The charge density a given species
   \param[out] phi The potential to be computed by this function
*/
void LabFrameExplicitES::computePhiTriDiagonal_periodic (
    const ablastr::fields::MultiLevelScalarField& rho,
    const ablastr::fields::MultiLevelScalarField& phi)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(num_levels == 1,
    "The tridiagonal solver cannot be used with mesh refinement");

    const int lev = 0;
    auto & warpx = WarpX::GetInstance();

    const amrex::Real* dx = warpx.Geom(lev).CellSize();
    const amrex::Real xmin = warpx.Geom(lev).ProbLo(0);
    const amrex::Real xmax = warpx.Geom(lev).ProbHi(0);
    const int nx = static_cast<int>( (xmax - xmin)/dx[0] + 0.5_rt );


    // Create a 1-D MultiFab that covers all of x.
    // The tridiag solve will be done in this MultiFab and then copied out afterwards.
    const amrex::IntVect lo_full_domain(AMREX_D_DECL(0,0,0));
    const amrex::IntVect hi_full_domain(AMREX_D_DECL(nx,0,0));
    const amrex::Box box_full_domain_node(lo_full_domain, hi_full_domain, amrex::IntVect::TheNodeVector());
    const BoxArray ba_full_domain_node(box_full_domain_node);
    const amrex::Vector<int> pmap = {0}; // The data will only be on processor 0
    const amrex::DistributionMapping dm_full_domain(pmap);

    // Put the data in the pinned arena since the tridiag solver will be done on the CPU, but have
    // the data readily accessible from the GPU.
    auto phi1d_mf = MultiFab(ba_full_domain_node, dm_full_domain, 1, 0, MFInfo().SetArena(The_Pinned_Arena()));

    // Work arrays
    auto cmod_mf = MultiFab(ba_full_domain_node, dm_full_domain, 1, 0, MFInfo().SetArena(The_Pinned_Arena()));
    auto u_mf = MultiFab(ba_full_domain_node, dm_full_domain, 1, 0, MFInfo().SetArena(The_Pinned_Arena()));

    // Copy rho into phi1d_mf to start
    phi1d_mf.ParallelCopy(*rho[lev], 0, 0, 1);

    // Multiplier on the charge density
    const amrex::Real norm = dx[0]*dx[0]/PhysConst::epsilon_0;
    phi1d_mf.mult(norm);

    // Use the MFIter loop since when parallel, only process zero has a FAB.
    // This skips the loop on all other processors.
    for (MFIter mfi(phi1d_mf); mfi.isValid(); ++mfi) {

        const auto& x = phi1d_mf[mfi].array();
        const auto& cmod = cmod_mf[mfi].array();
        const auto& u = u_mf[mfi].array();

        // The loops are always performed on the CPU

        {

        // This code is adapted from the Wikipedia page on the Tridiagonal matrix algorithm
        // https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm#Variants.
        // Licensed under CC BY-SA 4.0
        // Modifications: The a, b, and c inputs are replaced with the fixed values.

        const amrex::Real alpha = -1.0_rt;
        const amrex::Real beta = -1.0_rt;

        /* arbitrary, but chosen such that division by zero is avoided */
        const amrex::Real gamma = -2.0_rt;

        cmod(0,0,0) = -1.0_rt / (2.0_rt - gamma);
        u(0,0,0) = gamma / (2.0_rt - gamma);
        x(0,0,0) /= (2.0_rt - gamma);

        /* loop from 1 to nx - 2 inclusive */
        for (int ix = 1; ix + 1 < nx; ix++) {
            const amrex::Real m = 1.00_rt / (2.0_rt - -1.0_rt * cmod(ix - 1,0,0));
            cmod(ix,0,0) = -1.0_rt * m;
            u(ix,0,0) = (0.0f  - -1.0_rt * u(ix - 1,0,0)) * m;
            x(ix,0,0) = (x(ix,0,0) - -1.0_rt * x(ix - 1,0,0)) * m;
        }

        /* handle nx - 1 */
        const amrex::Real m = 1.00_rt / (2.0_rt - alpha * beta / gamma - -1.0_rt * cmod(nx - 2,0,0));
        u(nx - 1,0,0) = (alpha    - -1.0_rt * u(nx - 2,0,0)) * m;
        x(nx - 1,0,0) = (x(nx - 1,0,0) - -1.0_rt * x(nx - 2,0,0)) * m;

        /* loop from nx - 2 to 0 inclusive */
        for (int ix = nx - 2; ix >= 0; ix--) {
            u(ix,0,0) -= cmod(ix,0,0) * u(ix + 1,0,0);
            x(ix,0,0) -= cmod(ix,0,0) * x(ix + 1,0,0);
        }

        const amrex::Real fact = (x(0,0,0) + x(nx - 1,0,0) * alpha / gamma) / (1.00 + u(0,0,0) + u(nx - 1,0,0) * alpha / gamma);

        /* loop from 0 to nx - 1 inclusive */
        for (int ix = 0; ix < nx; ix++)
            x(ix,0,0) -= fact * u(ix,0,0);

        }

        x(nx,0,0) = x(0,0,0);

        // In a test case, this was giving an relative residual of around 1.e-10.
        // A dozen or so SOR iterations could improve that by a factor of 10.
        // Is it worth it?

    }

    // Copy phi1d to phi
    phi[lev]->ParallelCopy(phi1d_mf, 0, 0, 1);
}
