/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald, Arianna Formenti, Revathi Jambunathan
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ElectrostaticSolver.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"

#include <ablastr/fields/PoissonSolver.H>


using namespace amrex;
using warpx::fields::FieldType;

ElectrostaticSolver::ElectrostaticSolver (int nlevs_max) : num_levels{nlevs_max}
{
    // Create an instance of the boundary handler to properly set boundary
    // conditions
    m_poisson_boundary_handler = std::make_unique<PoissonBoundaryHandler>();
}

ElectrostaticSolver::~ElectrostaticSolver () = default;

void ElectrostaticSolver::ReadParameters () {

    ParmParse const pp_warpx("warpx");

    // Note that with the relativistic version, these parameters would be
    // input for each species.
    utils::parser::queryWithParser(
        pp_warpx, "self_fields_required_precision", self_fields_required_precision);
    utils::parser::queryWithParser(
        pp_warpx, "self_fields_absolute_tolerance", self_fields_absolute_tolerance);
    utils::parser::queryWithParser(
        pp_warpx, "self_fields_max_iters", self_fields_max_iters);
   utils::parser::queryWithParser(
        pp_warpx, "self_fields_verbosity", self_fields_verbosity);

    // FFT solver flags
   utils::parser::queryWithParser(
        pp_warpx, "use_2d_slices_fft_solver", is_igf_2d_slices);
}

void
ElectrostaticSolver::setPhiBC (
    ablastr::fields::MultiLevelScalarField const& phi,
    amrex::Real t
) const
{
    // check if any dimension has non-periodic boundary conditions
    if (!m_poisson_boundary_handler->has_non_periodic) { return; }

    // get the boundary potentials at the current time
    amrex::Array<amrex::Real,AMREX_SPACEDIM> phi_bc_values_lo;
    amrex::Array<amrex::Real,AMREX_SPACEDIM> phi_bc_values_hi;
    phi_bc_values_lo[WARPX_ZINDEX] = m_poisson_boundary_handler->potential_zlo(t);
    phi_bc_values_hi[WARPX_ZINDEX] = m_poisson_boundary_handler->potential_zhi(t);
#ifndef WARPX_DIM_1D_Z
    phi_bc_values_lo[0] = m_poisson_boundary_handler->potential_xlo(t);
    phi_bc_values_hi[0] = m_poisson_boundary_handler->potential_xhi(t);
#endif
#if defined(WARPX_DIM_3D)
    phi_bc_values_lo[1] = m_poisson_boundary_handler->potential_ylo(t);
    phi_bc_values_hi[1] = m_poisson_boundary_handler->potential_yhi(t);
#endif

    auto dirichlet_flag = m_poisson_boundary_handler->dirichlet_flag;

    auto & warpx = WarpX::GetInstance();

    // loop over all mesh refinement levels and set the boundary values
    for (int lev=0; lev < num_levels; lev++) {

        amrex::Box domain = warpx.Geom(lev).Domain();
        domain.surroundingNodes();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*phi[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
            // Extract the potential
            auto phi_arr = phi[lev]->array(mfi);
            // Extract tileboxes for which to loop
            const Box& tb  = mfi.tilebox( phi[lev]->ixType().toIntVect() );

            // loop over dimensions
            for (int idim=0; idim<AMREX_SPACEDIM; idim++){
                // check if neither boundaries in this dimension should be set
                if (!(dirichlet_flag[2*idim] || dirichlet_flag[2*idim+1])) { continue; }

                // a check can be added below to test if the boundary values
                // are already correct, in which case the ParallelFor over the
                // cells can be skipped

                if (!domain.strictly_contains(tb)) {
                    amrex::ParallelFor( tb,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                            IntVect iv(AMREX_D_DECL(i,j,k));

                            if (dirichlet_flag[2*idim] && iv[idim] == domain.smallEnd(idim)){
                                phi_arr(i,j,k) = phi_bc_values_lo[idim];
                            }
                            if (dirichlet_flag[2*idim+1] && iv[idim] == domain.bigEnd(idim)) {
                                phi_arr(i,j,k) = phi_bc_values_hi[idim];
                            }

                        } // loop ijk
                    );
                }
            } // idim
    }} // lev & MFIter
}


void
ElectrostaticSolver::computePhi (
    ablastr::fields::MultiLevelScalarField const& rho,
    ablastr::fields::MultiLevelScalarField const& phi,
    std::array<Real, 3> const beta,
    Real const required_precision,
    Real absolute_tolerance,
    int const max_iters,
    int const verbosity,
    bool const is_igf_2d
) const
{
    using ablastr::fields::Direction;

    // create a vector to our fields, sorted by level
    amrex::Vector<amrex::MultiFab *> sorted_rho;
    amrex::Vector<amrex::MultiFab *> sorted_phi;
    for (int lev = 0; lev < num_levels; ++lev) {
        sorted_rho.emplace_back(rho[lev]);
        sorted_phi.emplace_back(phi[lev]);
    }

    std::optional<EBCalcEfromPhiPerLevel> post_phi_calculation;
#ifdef AMREX_USE_EB
    // TODO: double check no overhead occurs on "m_eb_enabled == false"
    std::optional<amrex::Vector<amrex::EBFArrayBoxFactory const *> > eb_farray_box_factory;
#else
    std::optional<amrex::Vector<amrex::FArrayBoxFactory const *> > const eb_farray_box_factory;
#endif
    auto & warpx = WarpX::GetInstance();
    if (EB::enabled())
    {
        if (WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrame ||
            WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameElectroMagnetostatic)
        {
            // EB: use AMReX to directly calculate the electric field since with EB's the
            // simple finite difference scheme in WarpX::computeE sometimes fails

            // TODO: maybe make this a helper function or pass Efield_fp directly
            amrex::Vector<
                amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM>
            > e_field;
            for (int lev = 0; lev < num_levels; ++lev) {
                e_field.push_back(
#if defined(WARPX_DIM_1D_Z)
                    amrex::Array<amrex::MultiFab*, 1>{
                        warpx.m_fields.get(FieldType::Efield_fp, Direction{2}, lev)
                    }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    amrex::Array<amrex::MultiFab*, 2>{
                        warpx.m_fields.get(FieldType::Efield_fp, Direction{0}, lev),
                        warpx.m_fields.get(FieldType::Efield_fp, Direction{2}, lev)
                    }
#elif defined(WARPX_DIM_3D)
                    amrex::Array<amrex::MultiFab *, 3>{
                        warpx.m_fields.get(FieldType::Efield_fp, Direction{0}, lev),
                        warpx.m_fields.get(FieldType::Efield_fp, Direction{1}, lev),
                        warpx.m_fields.get(FieldType::Efield_fp, Direction{2}, lev)
                    }
#endif
                );
            }
            post_phi_calculation = EBCalcEfromPhiPerLevel(e_field);
        }
#ifdef AMREX_USE_EB
        amrex::Vector<
            amrex::EBFArrayBoxFactory const *
        > factories;
        for (int lev = 0; lev < num_levels; ++lev) {
            factories.push_back(&warpx.fieldEBFactory(lev));
        }
        eb_farray_box_factory = factories;
#endif
    }

    bool const is_solver_igf_on_lev0 =
        WarpX::poisson_solver_id == PoissonSolverAlgo::IntegratedGreenFunction;

    ablastr::fields::computePhi(
        sorted_rho,
        sorted_phi,
        beta,
        required_precision,
        absolute_tolerance,
        max_iters,
        verbosity,
        warpx.Geom(),
        warpx.DistributionMap(),
        warpx.boxArray(),
        WarpX::grid_type,
        is_solver_igf_on_lev0,
        is_igf_2d,
        EB::enabled(),
        WarpX::do_single_precision_comms,
        warpx.refRatio(),
        post_phi_calculation,
        *m_poisson_boundary_handler,
        warpx.gett_new(0),
        eb_farray_box_factory
    );

}

void
ElectrostaticSolver::computeE (
    ablastr::fields::MultiLevelVectorField const& E,
    ablastr::fields::MultiLevelScalarField const& phi,
    std::array<amrex::Real, 3> beta ) const
{
    auto & warpx = WarpX::GetInstance();
    for (int lev = 0; lev < num_levels; lev++) {

        const Real* dx = warpx.Geom(lev).CellSize();

#ifdef AMREX_USE_OMP
#    pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*phi[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi )
        {
#if defined(WARPX_DIM_3D)
            const Real inv_dx = 1._rt/dx[0];
            const Real inv_dy = 1._rt/dx[1];
            const Real inv_dz = 1._rt/dx[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            const Real inv_dx = 1._rt/dx[0];
            const Real inv_dz = 1._rt/dx[1];
#else
            const Real inv_dz = 1._rt/dx[0];
#endif
            const amrex::IntVect ex_type = E[lev][0]->ixType().toIntVect();
            const amrex::IntVect ey_type = E[lev][1]->ixType().toIntVect();
            const amrex::IntVect ez_type = E[lev][2]->ixType().toIntVect();

            const amrex::Box& tbx = mfi.tilebox(ex_type);
            const amrex::Box& tby = mfi.tilebox(ey_type);
            const amrex::Box& tbz = mfi.tilebox(ez_type);

            const auto& phi_arr = phi[lev]->array(mfi);
            const auto& Ex_arr = (*E[lev][0])[mfi].array();
            const auto& Ey_arr = (*E[lev][1])[mfi].array();
            const auto& Ez_arr = (*E[lev][2])[mfi].array();

            const Real beta_x = beta[0];
            const Real beta_y = beta[1];
            const Real beta_z = beta[2];

            // Calculate the electric field
            // Use discretized derivative that matches the staggering of the grid.
            // Nodal solver
            if (ex_type == amrex::IntVect::TheNodeVector() &&
                ey_type == amrex::IntVect::TheNodeVector() &&
                ez_type == amrex::IntVect::TheNodeVector())
            {
#if defined(WARPX_DIM_3D)
                amrex::ParallelFor( tbx, tby, tbz,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Ex_arr(i,j,k) +=
                            +(beta_x*beta_x-1._rt)*0.5_rt*inv_dx*(phi_arr(i+1,j  ,k  )-phi_arr(i-1,j  ,k  ))
                            + beta_x*beta_y       *0.5_rt*inv_dy*(phi_arr(i  ,j+1,k  )-phi_arr(i  ,j-1,k  ))
                            + beta_x*beta_z       *0.5_rt*inv_dz*(phi_arr(i  ,j  ,k+1)-phi_arr(i  ,j  ,k-1));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Ey_arr(i,j,k) +=
                            + beta_y*beta_x       *0.5_rt*inv_dx*(phi_arr(i+1,j  ,k  )-phi_arr(i-1,j  ,k  ))
                            +(beta_y*beta_y-1._rt)*0.5_rt*inv_dy*(phi_arr(i  ,j+1,k  )-phi_arr(i  ,j-1,k  ))
                            + beta_y*beta_z       *0.5_rt*inv_dz*(phi_arr(i  ,j  ,k+1)-phi_arr(i  ,j  ,k-1));                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Ez_arr(i,j,k) +=
                            + beta_z*beta_x       *0.5_rt*inv_dx*(phi_arr(i+1,j  ,k  )-phi_arr(i-1,j  ,k  ))
                            + beta_z*beta_y       *0.5_rt*inv_dy*(phi_arr(i  ,j+1,k  )-phi_arr(i  ,j-1,k  ))
                            +(beta_z*beta_z-1._rt)*0.5_rt*inv_dz*(phi_arr(i  ,j  ,k+1)-phi_arr(i  ,j  ,k-1));
                    }
                );
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                amrex::ParallelFor( tbx, tby, tbz,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Ex_arr(i,j,k) +=
                            +(beta_x*beta_x-1._rt)*0.5_rt*inv_dx*(phi_arr(i+1,j  ,k)-phi_arr(i-1,j  ,k))
                            + beta_x*beta_z       *0.5_rt*inv_dz*(phi_arr(i  ,j+1,k)-phi_arr(i  ,j-1,k));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Ey_arr(i,j,k) +=
                            +beta_x*beta_y*0.5_rt*inv_dx*(phi_arr(i+1,j  ,k)-phi_arr(i-1,j  ,k))
                            +beta_y*beta_z*0.5_rt*inv_dz*(phi_arr(i  ,j+1,k)-phi_arr(i  ,j-1,k));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Ez_arr(i,j,k) +=
                            + beta_z*beta_x       *0.5_rt*inv_dx*(phi_arr(i+1,j  ,k)-phi_arr(i-1,j  ,k))
                            +(beta_z*beta_z-1._rt)*0.5_rt*inv_dz*(phi_arr(i  ,j+1,k)-phi_arr(i  ,j-1,k));
                    }
                );
#else
                amrex::ParallelFor( tbx, tby, tbz,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Ex_arr(i,j,k) +=
                            +(beta_x*beta_z-1._rt)*0.5_rt*inv_dz*(phi_arr(i+1,j,k)-phi_arr(i-1,j,k));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Ey_arr(i,j,k) +=
                            +beta_y*beta_z*0.5_rt*inv_dz*(phi_arr(i+1,j,k)-phi_arr(i-1,j,k));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Ez_arr(i,j,k) +=
                            +(beta_z*beta_z-1._rt)*0.5_rt*inv_dz*(phi_arr(i+1,j,k)-phi_arr(i-1,j,k));
                    }
                );
#endif
            }
            else // Staggered solver
            {
#if defined(WARPX_DIM_3D)
                amrex::ParallelFor( tbx, tby, tbz,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Ex_arr(i,j,k) +=
                            +(beta_x*beta_x-1._rt) *inv_dx*(phi_arr(i+1,j  ,k  )-phi_arr(i  ,j  ,k  ))
                            + beta_x*beta_y*0.25_rt*inv_dy*(phi_arr(i  ,j+1,k  )-phi_arr(i  ,j-1,k  )
                                                        + phi_arr(i+1,j+1,k  )-phi_arr(i+1,j-1,k  ))
                            + beta_x*beta_z*0.25_rt*inv_dz*(phi_arr(i  ,j  ,k+1)-phi_arr(i  ,j  ,k-1)
                                                        + phi_arr(i+1,j  ,k+1)-phi_arr(i+1,j  ,k-1));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Ey_arr(i,j,k) +=
                            + beta_y*beta_x*0.25_rt*inv_dx*(phi_arr(i+1,j  ,k  )-phi_arr(i-1,j  ,k  )
                                                        + phi_arr(i+1,j+1,k  )-phi_arr(i-1,j+1,k  ))
                            +(beta_y*beta_y-1._rt) *inv_dy*(phi_arr(i  ,j+1,k  )-phi_arr(i  ,j  ,k  ))
                            + beta_y*beta_z*0.25_rt*inv_dz*(phi_arr(i  ,j  ,k+1)-phi_arr(i  ,j  ,k-1)
                                                        + phi_arr(i  ,j+1,k+1)-phi_arr(i  ,j+1,k-1));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Ez_arr(i,j,k) +=
                            + beta_z*beta_x*0.25_rt*inv_dx*(phi_arr(i+1,j  ,k  )-phi_arr(i-1,j  ,k  )
                                                        + phi_arr(i+1,j  ,k+1)-phi_arr(i-1,j  ,k+1))
                            + beta_z*beta_y*0.25_rt*inv_dy*(phi_arr(i  ,j+1,k  )-phi_arr(i  ,j-1,k  )
                                                        + phi_arr(i  ,j+1,k+1)-phi_arr(i  ,j-1,k+1))
                            +(beta_z*beta_z-1._rt) *inv_dz*(phi_arr(i  ,j  ,k+1)-phi_arr(i  ,j  ,k  ));
                    }
                );
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                amrex::ParallelFor( tbx, tbz,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Ex_arr(i,j,k) +=
                            +(beta_x*beta_x-1._rt)*inv_dx*(phi_arr(i+1,j  ,k)-phi_arr(i  ,j  ,k))
                            +beta_x*beta_z*0.25_rt*inv_dz*(phi_arr(i  ,j+1,k)-phi_arr(i  ,j-1,k)
                                                        + phi_arr(i+1,j+1,k)-phi_arr(i+1,j-1,k));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Ez_arr(i,j,k) +=
                            +beta_z*beta_x*0.25_rt*inv_dx*(phi_arr(i+1,j  ,k)-phi_arr(i-1,j  ,k)
                                                        + phi_arr(i+1,j+1,k)-phi_arr(i-1,j+1,k))
                            +(beta_z*beta_z-1._rt)*inv_dz*(phi_arr(i  ,j+1,k)-phi_arr(i  ,j  ,k));
                    }
                );
                ignore_unused(beta_y);
#else
                amrex::ParallelFor( tbz,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Ez_arr(i,j,k) +=
                            +(beta_z*beta_z-1._rt)*inv_dz*(phi_arr(i+1,j,k)-phi_arr(i,j,k));
                    }
                );
                ignore_unused(beta_x,beta_y);
#endif
            }
        }
    }
}


void ElectrostaticSolver::computeB (
        ablastr::fields::MultiLevelVectorField const& B,
        ablastr::fields::MultiLevelScalarField const& phi,
        std::array<amrex::Real, 3> beta) const
{
    // return early if beta is 0 since there will be no B-field
    if ((beta[0] == 0._rt) && (beta[1] == 0._rt) && (beta[2] == 0._rt)) { return; }

    auto & warpx = WarpX::GetInstance();
    for (int lev = 0; lev < num_levels; lev++) {

        const Real* dx = warpx.Geom(lev).CellSize();

#ifdef AMREX_USE_OMP
#    pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*phi[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi )
        {
#if defined(WARPX_DIM_3D)
            const Real inv_dx = 1._rt/dx[0];
            const Real inv_dy = 1._rt/dx[1];
            const Real inv_dz = 1._rt/dx[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            const Real inv_dx = 1._rt/dx[0];
            const Real inv_dz = 1._rt/dx[1];
#else
            const Real inv_dz = 1._rt/dx[0];
#endif
            const amrex::IntVect bx_type = B[lev][0]->ixType().toIntVect();
            const amrex::IntVect by_type = B[lev][1]->ixType().toIntVect();
            const amrex::IntVect bz_type = B[lev][2]->ixType().toIntVect();

            const amrex::Box& tbx = mfi.tilebox(bx_type);
            const amrex::Box& tby = mfi.tilebox(by_type);
            const amrex::Box& tbz = mfi.tilebox(bz_type);

            const auto& phi_arr = phi[lev]->array(mfi);
            const auto& Bx_arr = (*B[lev][0])[mfi].array();
            const auto& By_arr = (*B[lev][1])[mfi].array();
            const auto& Bz_arr = (*B[lev][2])[mfi].array();

            const Real beta_x = beta[0];
            const Real beta_y = beta[1];
            const Real beta_z = beta[2];

            constexpr Real inv_c = 1._rt/PhysConst::c;

            // Calculate the magnetic field
            // Use discretized derivative that matches the staggering of the grid.
            // Nodal solver
            if (bx_type == amrex::IntVect::TheNodeVector() &&
                by_type == amrex::IntVect::TheNodeVector() &&
                bz_type == amrex::IntVect::TheNodeVector())
            {
#if defined(WARPX_DIM_3D)
                amrex::ParallelFor( tbx, tby, tbz,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Bx_arr(i,j,k) += inv_c * (
                            -beta_y*inv_dz*0.5_rt*(phi_arr(i,j  ,k+1)-phi_arr(i,j  ,k-1))
                            +beta_z*inv_dy*0.5_rt*(phi_arr(i,j+1,k  )-phi_arr(i,j-1,k  )));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        By_arr(i,j,k) += inv_c * (
                            -beta_z*inv_dx*0.5_rt*(phi_arr(i+1,j,k  )-phi_arr(i-1,j,k  ))
                            +beta_x*inv_dz*0.5_rt*(phi_arr(i  ,j,k+1)-phi_arr(i  ,j,k-1)));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Bz_arr(i,j,k) += inv_c * (
                            -beta_x*inv_dy*0.5_rt*(phi_arr(i  ,j+1,k)-phi_arr(i  ,j-1,k))
                            +beta_y*inv_dx*0.5_rt*(phi_arr(i+1,j  ,k)-phi_arr(i-1,j  ,k)));
                    }
                );
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                amrex::ParallelFor( tbx, tby, tbz,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Bx_arr(i,j,k) += inv_c * (
                            -beta_y*inv_dz*0.5_rt*(phi_arr(i,j+1,k)-phi_arr(i,j-1,k)));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        By_arr(i,j,k) += inv_c * (
                            -beta_z*inv_dx*0.5_rt*(phi_arr(i+1,j  ,k)-phi_arr(i-1,j  ,k))
                            +beta_x*inv_dz*0.5_rt*(phi_arr(i  ,j+1,k)-phi_arr(i  ,j-1,k)));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Bz_arr(i,j,k) += inv_c * (
                            +beta_y*inv_dx*0.5_rt*(phi_arr(i+1,j,k)-phi_arr(i-1,j,k)));
                    }
                );
#else
                amrex::ParallelFor( tbx, tby,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Bx_arr(i,j,k) += inv_c * (
                            -beta_y*inv_dz*(phi_arr(i+1,j,k)-phi_arr(i,j,k)));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        By_arr(i,j,k) += inv_c * (
                            +beta_x*inv_dz*(phi_arr(i+1,j,k)-phi_arr(i,j,k)));
                    }
                );
                ignore_unused(beta_z,tbz,Bz_arr);
#endif
            }
            else // Staggered solver
            {
#if defined(WARPX_DIM_3D)
                amrex::ParallelFor( tbx, tby, tbz,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Bx_arr(i,j,k) += inv_c * (
                            -beta_y*inv_dz*0.5_rt*(phi_arr(i,j  ,k+1)-phi_arr(i,j  ,k  )
                                                + phi_arr(i,j+1,k+1)-phi_arr(i,j+1,k  ))
                            +beta_z*inv_dy*0.5_rt*(phi_arr(i,j+1,k  )-phi_arr(i,j  ,k  )
                                                + phi_arr(i,j+1,k+1)-phi_arr(i,j  ,k+1)));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        By_arr(i,j,k) += inv_c * (
                            -beta_z*inv_dx*0.5_rt*(phi_arr(i+1,j,k  )-phi_arr(i  ,j,k  )
                                                + phi_arr(i+1,j,k+1)-phi_arr(i  ,j,k+1))
                            +beta_x*inv_dz*0.5_rt*(phi_arr(i  ,j,k+1)-phi_arr(i  ,j,k  )
                                                + phi_arr(i+1,j,k+1)-phi_arr(i+1,j,k  )));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Bz_arr(i,j,k) += inv_c * (
                            -beta_x*inv_dy*0.5_rt*(phi_arr(i  ,j+1,k)-phi_arr(i  ,j  ,k)
                                                + phi_arr(i+1,j+1,k)-phi_arr(i+1,j  ,k))
                            +beta_y*inv_dx*0.5_rt*(phi_arr(i+1,j  ,k)-phi_arr(i  ,j  ,k)
                                                + phi_arr(i+1,j+1,k)-phi_arr(i  ,j+1,k)));
                    }
                );
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                amrex::ParallelFor( tbx, tby, tbz,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Bx_arr(i,j,k) += inv_c * (
                            -beta_y*inv_dz*(phi_arr(i,j+1,k)-phi_arr(i,j,k)));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        By_arr(i,j,k) += inv_c * (
                            -beta_z*inv_dx*0.5_rt*(phi_arr(i+1,j  ,k)-phi_arr(i  ,j  ,k)
                                                + phi_arr(i+1,j+1,k)-phi_arr(i  ,j+1,k))
                            +beta_x*inv_dz*0.5_rt*(phi_arr(i  ,j+1,k)-phi_arr(i  ,j  ,k)
                                                + phi_arr(i+1,j+1,k)-phi_arr(i+1,j  ,k)));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Bz_arr(i,j,k) += inv_c * (
                            +beta_y*inv_dx*(phi_arr(i+1,j,k)-phi_arr(i,j,k)));
                    }
                );
#else
                amrex::ParallelFor( tbx, tby,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        Bx_arr(i,j,k) += inv_c * (
                            -beta_y*inv_dz*(phi_arr(i+1,j,k)-phi_arr(i,j,k)));
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        By_arr(i,j,k) += inv_c * (
                            +beta_x*inv_dz*(phi_arr(i+1,j,k)-phi_arr(i,j,k)));
                    }
                );
                ignore_unused(beta_z,tbz,Bz_arr);
#endif
            }
        }
    }
}
