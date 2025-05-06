/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FiniteDifferenceSolver.H"

#include "EmbeddedBoundary/WarpXFaceInfoBox.H"
#include "Fields.H"
#ifndef WARPX_DIM_RZ
#   include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianCustomCoefAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#else
#   include "FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#endif
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <memory>

using namespace amrex;

/**
 * \brief Update the B field, over one timestep
 */
void FiniteDifferenceSolver::EvolveB (
    ablastr::fields::MultiFabRegister& fields,
    int lev,
    PatchType patch_type,
    [[maybe_unused]] std::array< std::unique_ptr<amrex::iMultiFab>, 3 >& flag_info_cell,
    [[maybe_unused]] std::array< std::unique_ptr<amrex::LayoutData<FaceInfoBox> >, 3 >& borrowing,
    [[maybe_unused]] amrex::Real const dt )
{

    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    const ablastr::fields::VectorField Bfield = patch_type == PatchType::fine ?
        fields.get_alldirs(FieldType::Bfield_fp, lev) : fields.get_alldirs(FieldType::Bfield_cp, lev);
    const ablastr::fields::VectorField Efield = patch_type == PatchType::fine ?
        fields.get_alldirs(FieldType::Efield_fp, lev) : fields.get_alldirs(FieldType::Efield_cp, lev);

    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
#ifdef WARPX_DIM_RZ
    if ((m_fdtd_algo == ElectromagneticSolverAlgo::Yee)||
        (m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC)){
        EvolveBCylindrical <CylindricalYeeAlgorithm> ( Bfield, Efield, lev, dt );
#else

    amrex::MultiFab const * Gfield = nullptr;
    if (fields.has(FieldType::G_fp, lev)) {
        Gfield = patch_type == PatchType::fine ?
            fields.get(FieldType::G_fp, lev) : fields.get(FieldType::G_cp, lev);
    }
    ablastr::fields::VectorField face_areas;
    if (fields.has_vector(FieldType::face_areas, lev)) {
        face_areas = fields.get_alldirs(FieldType::face_areas, lev);
    }
    ablastr::fields::VectorField area_mod;
    if (fields.has_vector(FieldType::area_mod, lev)) {
        area_mod = fields.get_alldirs(FieldType::area_mod, lev);
    }
    ablastr::fields::VectorField ECTRhofield;
    if (fields.has_vector(FieldType::ECTRhofield, lev)) {
        ECTRhofield = fields.get_alldirs(FieldType::ECTRhofield, lev);
    }
    ablastr::fields::VectorField Venl;
    if (fields.has_vector(FieldType::Venl, lev)) {
        Venl = fields.get_alldirs(FieldType::Venl, lev);
    }

    if (m_grid_type == GridType::Collocated) {

        EvolveBCartesian <CartesianNodalAlgorithm> ( Bfield, Efield, Gfield, lev, dt );

    } else if ((m_fdtd_algo == ElectromagneticSolverAlgo::Yee) ||
               (m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC)) {

        EvolveBCartesian <CartesianYeeAlgorithm> ( Bfield, Efield, Gfield, lev, dt );

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::CKC) {

        EvolveBCartesian <CartesianCKCAlgorithm> ( Bfield, Efield, Gfield, lev, dt );

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::CustomCoef) {

        EvolveBCartesian <CartesianCustomCoefAlgorithm> ( Bfield, Efield, Gfield, lev, dt );

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::ECT) {
        EvolveBCartesianECT(Bfield, face_areas, area_mod, ECTRhofield, Venl, flag_info_cell,
                            borrowing, lev, dt);
#endif
    } else {
        WARPX_ABORT_WITH_MESSAGE("EvolveB: Unknown algorithm");
    }
}


#ifndef WARPX_DIM_RZ

template<typename T_Algo>
void FiniteDifferenceSolver::EvolveBCartesian (
    ablastr::fields::VectorField const& Bfield,
    ablastr::fields::VectorField const& Efield,
    amrex::MultiFab const * Gfield,
    int lev, amrex::Real const dt ) {

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const& Bx = Bfield[0]->array(mfi);
        Array4<Real> const& By = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);
        Array4<Real> const& Ex = Efield[0]->array(mfi);
        Array4<Real> const& Ey = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract tileboxes for which to loop
        Box const& tbx  = mfi.tilebox(Bfield[0]->ixType().toIntVect());
        Box const& tby  = mfi.tilebox(Bfield[1]->ixType().toIntVect());
        Box const& tbz  = mfi.tilebox(Bfield[2]->ixType().toIntVect());

        // Loop over the cells and update the fields
        amrex::ParallelFor(tbx, tby, tbz,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                Bx(i, j, k) += dt * T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k)
                             - dt * T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k);

            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                By(i, j, k) += dt * T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k)
                             - dt * T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k);

            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                Bz(i, j, k) += dt * T_Algo::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k)
                             - dt * T_Algo::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k);

            }
        );

        // div(B) cleaning correction for errors in magnetic Gauss law (div(B) = 0)
        if (Gfield)
        {
            // Extract field data for this grid/tile
            Array4<Real const> const G = Gfield->array(mfi);

            // Loop over cells and update G
            amrex::ParallelFor(tbx, tby, tbz,

                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    Bx(i,j,k) += dt * T_Algo::DownwardDx(G, coefs_x, n_coefs_x, i, j, k);
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    By(i,j,k) += dt * T_Algo::DownwardDy(G, coefs_y, n_coefs_y, i, j, k);
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    Bz(i,j,k) += dt * T_Algo::DownwardDz(G, coefs_z, n_coefs_z, i, j, k);
                }
            );
        }

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}


void FiniteDifferenceSolver::EvolveBCartesianECT (
    ablastr::fields::VectorField const& Bfield,
    ablastr::fields::VectorField const& face_areas,
    ablastr::fields::VectorField const& area_mod,
    ablastr::fields::VectorField const& ECTRhofield,
    ablastr::fields::VectorField const& Venl,
    std::array< std::unique_ptr<amrex::iMultiFab>, 3 >& flag_info_cell,
    std::array< std::unique_ptr<amrex::LayoutData<FaceInfoBox> >, 3 >& borrowing,
    const int lev, amrex::Real const dt ) {
#ifdef AMREX_USE_EB

#if !(defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ))
    WARPX_ABORT_WITH_MESSAGE(
        "EvolveBCartesianECT: Embedded Boundaries are only implemented in 2D3V and 3D3V");
#endif

    amrex::LayoutData<amrex::Real> *cost = WarpX::getCosts(lev);

    Venl[0]->setVal(0.);
    Venl[1]->setVal(0.);
    Venl[2]->setVal(0.);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*Bfield[0]); mfi.isValid(); ++mfi) {

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers) {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            // Extract field data for this grid/tile
            Array4<Real> const &B = Bfield[idim]->array(mfi);
            Array4<Real> const &Rho = ECTRhofield[idim]->array(mfi);
            Array4<Real> const &Venl_dim = Venl[idim]->array(mfi);

            amrex::Array4<int> const &flag_info_cell_dim = flag_info_cell[idim]->array(mfi);
            amrex::Array4<Real> const &S = face_areas[idim]->array(mfi);
            amrex::Array4<Real> const &S_mod = area_mod[idim]->array(mfi);

            auto & borrowing_dim = (*borrowing[idim])[mfi];
            auto * borrowing_dim_neigh_faces = borrowing_dim.neigh_faces.data();
            auto * borrowing_dim_area = borrowing_dim.area.data();

            auto const &borrowing_inds = (*borrowing[idim])[mfi].inds.data();
            auto const &borrowing_size = (*borrowing[idim])[mfi].size.array();
            auto const &borrowing_inds_pointer = (*borrowing[idim])[mfi].inds_pointer.array();

            // Extract tileboxes for which to loop
            Box const &tb = mfi.tilebox(Bfield[idim]->ixType().toIntVect());

            //Take care of the unstable cells
            amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                if (S(i, j, k) <= 0) { return; }

                if (!(flag_info_cell_dim(i, j, k) == 0)) { return; }

                Venl_dim(i, j, k) = Rho(i, j, k) * S(i, j, k);
                amrex::Real rho_enl;

                // First we compute the rho of the enlarged face
                for (int offset = 0; offset<borrowing_size(i, j, k); offset++) {
                    int const ind = borrowing_inds[*borrowing_inds_pointer(i, j, k) + offset];
                    auto vec = FaceInfoBox::uint8_to_inds(borrowing_dim_neigh_faces[ind]);
                    int ip, jp, kp;
                    if (idim == 0) {
                        ip = i;
                        jp = j + vec(0);
                        kp = k + vec(1);
                    } else if (idim == 1) {  // NOLINT(bugprone-branch-clone)
#ifdef WARPX_DIM_XZ
                        ip = i + vec(0);
                        jp = j + vec(1);
                        kp = k;
#elif defined(WARPX_DIM_3D)
                        ip = i + vec(0);
                        jp = j;
                        kp = k + vec(1);
#else
                        WARPX_ABORT_WITH_MESSAGE("EvolveBCartesianECT: Embedded Boundaries are only implemented in 2D3V and 3D3V");
#endif
                    } else {
                        ip = i + vec(0);
                        jp = j + vec(1);
                        kp = k;
                    }

                    Venl_dim(i, j, k) += Rho(ip, jp, kp) * borrowing_dim_area[ind];

                }

                rho_enl = Venl_dim(i, j, k) / S_mod(i, j, k);

                for (int offset = 0; offset < borrowing_size(i, j, k); offset++) {
                    int const ind = borrowing_inds[*borrowing_inds_pointer(i, j, k) + offset];
                    auto vec = FaceInfoBox::uint8_to_inds(borrowing_dim_neigh_faces[ind]);
                    int ip, jp, kp;
                    if (idim == 0) {
                        ip = i;
                        jp = j + vec(0);
                        kp = k + vec(1);
                    } else if (idim == 1) {  // NOLINT(bugprone-branch-clone)
#ifdef WARPX_DIM_XZ
                        ip = i + vec(0);
                        jp = j + vec(1);
                        kp = k;
#elif defined(WARPX_DIM_3D)
                        ip = i + vec(0);
                        jp = j;
                        kp = k + vec(1);
#else
                        WARPX_ABORT_WITH_MESSAGE("EvolveBCartesianECT: Embedded Boundaries are only implemented in 2D3V and 3D3V");
#endif
                    } else {
                        ip = i + vec(0);
                        jp = j + vec(1);
                        kp = k;
                    }

                    Venl_dim(ip, jp, kp) += rho_enl * borrowing_dim_area[ind];

                }

                B(i, j, k) = B(i, j, k) - dt * rho_enl;

            });

            //Take care of the stable cells
            amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (S(i, j, k) <= 0) { return; }

                if (flag_info_cell_dim(i, j, k) == 0) {
                    return;
                }
                else if (flag_info_cell_dim(i, j, k) == 1) {
                    //Stable cell which hasn't been intruded
                    B(i, j, k) = B(i, j, k) - dt * Rho(i, j, k);
                } else if (flag_info_cell_dim(i, j, k) == 2) {
                    //Stable cell which has been intruded
                    Venl_dim(i, j, k) += Rho(i, j, k) * S_mod(i, j, k);
                    B(i, j, k) = B(i, j, k) - dt * Venl_dim(i, j, k) / S(i, j, k);
                }

            });

        }
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
#else
    amrex::ignore_unused(Bfield, face_areas, area_mod, ECTRhofield, Venl, flag_info_cell, borrowing,
                         lev, dt);
#endif
}

#else // corresponds to ifndef WARPX_DIM_RZ

template<typename T_Algo>
void FiniteDifferenceSolver::EvolveBCylindrical (
    ablastr::fields::VectorField const& Bfield,
    ablastr::fields::VectorField const& Efield,
    int lev, amrex::Real const dt ) {

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const& Br = Bfield[0]->array(mfi);
        Array4<Real> const& Bt = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);
        Array4<Real> const& Er = Efield[0]->array(mfi);
        Array4<Real> const& Et = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
        auto const n_coefs_r = static_cast<int>(m_stencil_coefs_r.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract cylindrical specific parameters
        Real const dr = m_dr;
        int const nmodes = m_nmodes;
        Real const rmin = m_rmin;

        // Extract tileboxes for which to loop
        Box const& tbr  = mfi.tilebox(Bfield[0]->ixType().toIntVect());
        Box const& tbt  = mfi.tilebox(Bfield[1]->ixType().toIntVect());
        Box const& tbz  = mfi.tilebox(Bfield[2]->ixType().toIntVect());

        // Loop over the cells and update the fields
        amrex::ParallelFor(tbr, tbt, tbz,

            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                Real const r = rmin + i*dr; // r on nodal point (Br is nodal in r)
                if (r != 0) { // Off-axis, regular Maxwell equations
                    Br(i, j, 0, 0) += dt * T_Algo::UpwardDz(Et, coefs_z, n_coefs_z, i, j, 0, 0); // Mode m=0
                    for (int m=1; m<nmodes; m++) { // Higher-order modes
                        Br(i, j, 0, 2*m-1) += dt*(
                            T_Algo::UpwardDz(Et, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                            - m * Ez(i, j, 0, 2*m  )/r );  // Real part
                        Br(i, j, 0, 2*m  ) += dt*(
                            T_Algo::UpwardDz(Et, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                            + m * Ez(i, j, 0, 2*m-1)/r ); // Imaginary part
                    }
                } else { // r==0: On-axis corrections
                    // Ensure that Br remains 0 on axis (except for m=1)
                    Br(i, j, 0, 0) = 0.; // Mode m=0
                    for (int m=1; m<nmodes; m++) { // Higher-order modes
                        if (m == 1){
                            // For m==1, Ez is linear in r, for small r
                            // Therefore, the formula below regularizes the singularity
                            Br(i, j, 0, 2*m-1) += dt*(
                                T_Algo::UpwardDz(Et, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                                - m * Ez(i+1, j, 0, 2*m  )/dr );  // Real part
                            Br(i, j, 0, 2*m  ) += dt*(
                                T_Algo::UpwardDz(Et, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                                + m * Ez(i+1, j, 0, 2*m-1)/dr ); // Imaginary part
                        } else {
                            Br(i, j, 0, 2*m-1) = 0.;
                            Br(i, j, 0, 2*m  ) = 0.;
                        }
                    }
                }
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                Bt(i, j, 0, 0) += dt*(
                    T_Algo::UpwardDr(Ez, coefs_r, n_coefs_r, i, j, 0, 0)
                    - T_Algo::UpwardDz(Er, coefs_z, n_coefs_z, i, j, 0, 0)); // Mode m=0
                for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                    Bt(i, j, 0, 2*m-1) += dt*(
                        T_Algo::UpwardDr(Ez, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                        - T_Algo::UpwardDz(Er, coefs_z, n_coefs_z, i, j, 0, 2*m-1)); // Real part
                    Bt(i, j, 0, 2*m  ) += dt*(
                        T_Algo::UpwardDr(Ez, coefs_r, n_coefs_r, i, j, 0, 2*m  )
                        - T_Algo::UpwardDz(Er, coefs_z, n_coefs_z, i, j, 0, 2*m  )); // Imaginary part
                }
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                Real const r = rmin + (i + 0.5_rt)*dr; // r on a cell-centered grid (Bz is cell-centered in r)
                Bz(i, j, 0, 0) += dt*( - T_Algo::UpwardDrr_over_r(Et, r, dr, coefs_r, n_coefs_r, i, j, 0, 0));
                for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                    Bz(i, j, 0, 2*m-1) += dt*( m * Er(i, j, 0, 2*m  )/r
                        - T_Algo::UpwardDrr_over_r(Et, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m-1)); // Real part
                    Bz(i, j, 0, 2*m  ) += dt*(-m * Er(i, j, 0, 2*m-1)/r
                        - T_Algo::UpwardDrr_over_r(Et, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m  )); // Imaginary part
                }
            }

        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

#endif // corresponds to ifndef WARPX_DIM_RZ
