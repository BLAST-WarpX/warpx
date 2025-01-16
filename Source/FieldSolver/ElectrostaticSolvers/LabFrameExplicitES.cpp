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

    const MultiLevelScalarField rho_fp = fields.get_mr_levels(FieldType::rho_fp, max_level);
    const MultiLevelScalarField rho_cp = fields.get_mr_levels(FieldType::rho_cp, max_level);
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
                   self_fields_verbosity, is_igf_2d_slices);
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

    const int lev = 0;
    auto & warpx = WarpX::GetInstance();

    const amrex::Real* dx = warpx.Geom(lev).CellSize();
    const amrex::Real xmin = warpx.Geom(lev).ProbLo(0);
    const amrex::Real xmax = warpx.Geom(lev).ProbHi(0);
    const int nx_full_domain = static_cast<int>( (xmax - xmin)/dx[0] + 0.5_rt );

    int nx_solve_min = 1;
    int nx_solve_max = nx_full_domain - 1;

    auto field_boundary_lo0 = WarpX::field_boundary_lo[0];
    auto field_boundary_hi0 = WarpX::field_boundary_hi[0];
    if (field_boundary_lo0 == FieldBoundaryType::Neumann || field_boundary_lo0 == FieldBoundaryType::Periodic) {
        // Neumann or periodic boundary condition
        // Solve for the point on the lower boundary
        nx_solve_min = 0;
    }
    if (field_boundary_hi0 == FieldBoundaryType::Neumann || field_boundary_hi0 == FieldBoundaryType::Periodic) {
        // Neumann or periodic boundary condition
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
    const amrex::Real norm = dx[0]*dx[0]/PhysConst::ep0;
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

        } else if (field_boundary_lo0 == FieldBoundaryType::Periodic) {

            phi1d_arr(0,0,0) = rho1d_arr(0,0,0)/diag;

            zwork1d_arr(1,0,0) = 1._rt/diag;
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
        amrex::Real zwork_product = 1.; // Needed for parallel boundaries
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

        } else if (field_boundary_hi0 == FieldBoundaryType::Periodic) {

            zwork1d_arr(nx_full_domain,0,0) = 1._rt/diag;

            for (int i = 1 ; i <= nx_full_domain ; i++) {
                zwork_product *= zwork1d_arr(i,0,0);
            }

            diag = 2._rt - zwork1d_arr(nx_full_domain,0,0) - zwork_product;
            // Note that rho1d_arr(0,0,0) is used to ensure that the same value is used
            // on both boundaries.
            phi1d_arr(nx_full_domain,0,0) = (rho1d_arr(0,0,0) - (-1._rt)*phi1d_arr(nx_full_domain-1,0,0))/diag;

        }

        // Loop downward to calculate the phi
        if (field_boundary_lo0 == FieldBoundaryType::Periodic) {

            // With periodic, the right hand column adds an extra term for all rows
            for (int i_down = nx_full_domain-1 ; i_down >= 0 ; i_down--) {
                zwork_product /= zwork1d_arr(i_down+1,0,0);
                phi1d_arr(i_down,0,0) = phi1d_arr(i_down,0,0) + zwork1d_arr(i_down+1,0,0)*phi1d_arr(i_down+1,0,0) + zwork_product*phi1d_arr(nx_full_domain,0,0);
            }

        } else {

            for (int i_down = nx_solve_max-1 ; i_down >= nx_solve_min ; i_down--) {
                phi1d_arr(i_down,0,0) = phi1d_arr(i_down,0,0) + zwork1d_arr(i_down+1,0,0)*phi1d_arr(i_down+1,0,0);
            }

        }

    }

    // Copy phi1d to phi
    phi[lev]->ParallelCopy(phi1d_mf, 0, 0, 1);
}
