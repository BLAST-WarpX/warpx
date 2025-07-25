/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FiniteDifferenceSolver.H"

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#elif defined(WARPX_DIM_RSPHERE)
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/SphericalYeeAlgorithm.H"
#else
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#endif
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
#   include "WarpX.H"
#endif

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <memory>

using namespace amrex;

/**
 * \brief Update the F field, over one timestep
 */
void FiniteDifferenceSolver::EvolveF (
    amrex::MultiFab* Ffield,
    ablastr::fields::VectorField const& Efield,
    amrex::MultiFab* const rhofield,
    int const rhocomp,
    amrex::Real const dt ) {

    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::Yee){

        EvolveFCylindrical <CylindricalYeeAlgorithm> ( Ffield, Efield, rhofield, rhocomp, dt );

#elif defined(WARPX_DIM_RSPHERE)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::Yee){

        EvolveFSpherical <SphericalYeeAlgorithm> ( Ffield, Efield, rhofield, rhocomp, dt );

#else
    if (m_grid_type == GridType::Collocated) {

        EvolveFCartesian <CartesianNodalAlgorithm> ( Ffield, Efield, rhofield, rhocomp, dt );

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::Yee) {

        EvolveFCartesian <CartesianYeeAlgorithm> ( Ffield, Efield, rhofield, rhocomp, dt );

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::CKC) {

        EvolveFCartesian <CartesianCKCAlgorithm> ( Ffield, Efield, rhofield, rhocomp, dt );

#endif
    } else {
        WARPX_ABORT_WITH_MESSAGE("EvolveF: Unknown algorithm");
    }

}


#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)

template<typename T_Algo>
void FiniteDifferenceSolver::EvolveFCartesian (
    amrex::MultiFab* Ffield,
    ablastr::fields::VectorField const Efield,
    amrex::MultiFab* const rhofield,
    int const rhocomp,
    amrex::Real const dt ) {

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Ffield, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const& F = Ffield->array(mfi);
        Array4<Real> const& Ex = Efield[0]->array(mfi);
        Array4<Real> const& Ey = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real> const& rho = rhofield->array(mfi);

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z =static_cast<int>(m_stencil_coefs_z.size());

        // Extract tileboxes for which to loop
        Box const& tf  = mfi.tilebox(Ffield->ixType().toIntVect());

        Real constexpr inv_epsilon0 = 1._rt/PhysConst::ep0;

        // Loop over the cells and update the fields
        amrex::ParallelFor(tf,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                F(i, j, k) += dt * (
                    - rho(i, j, k, rhocomp) * inv_epsilon0
                    + T_Algo::DownwardDx(Ex, coefs_x, n_coefs_x, i, j, k)
                    + T_Algo::DownwardDy(Ey, coefs_y, n_coefs_y, i, j, k)
                    + T_Algo::DownwardDz(Ez, coefs_z, n_coefs_z, i, j, k) );
            }

        );

    }

}

#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)

template<typename T_Algo>
void FiniteDifferenceSolver::EvolveFCylindrical (
    amrex::MultiFab* Ffield,
    ablastr::fields::VectorField const & Efield,
    amrex::MultiFab* const rhofield,
    int const rhocomp,
    amrex::Real const dt ) {

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Ffield, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        const Array4<Real> F = Ffield->array(mfi);
        Array4<Real> const& Er = Efield[0]->array(mfi);
        Array4<Real> const& Et = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real> const& rho = rhofield->array(mfi);

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
        Box const& tf  = mfi.tilebox(Ffield->ixType().toIntVect());

        Real constexpr inv_epsilon0 = 1./PhysConst::ep0;

        // Use the right shift in components:
        // - the first WarpX::ncomps (2*n_rz_azimuthal_modes-1) components correspond to rho old (i.e. rhocomp=0)
        // - the next WarpX::ncomps (2*n_rz_azimuthal_modes-1) components correspond to rho new (i.e. rhocomp=1)
        int rho_shift = 0;
        if (rhocomp == 1) {
            rho_shift = WarpX::ncomps;
        }

        // Loop over the cells and update the fields
        amrex::ParallelFor(tf,

            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                Real const r = rmin + i*dr; // r on a nodal grid (F is nodal in r)
                if (r != 0) { // Off-axis, regular equations
                    F(i, j, 0, 0) += dt * (
                        - rho(i, j, 0, rho_shift) * inv_epsilon0
                        + T_Algo::DownwardDrr_over_r(Er, r, dr, coefs_r, n_coefs_r, i, j, 0, 0)
                        + T_Algo::DownwardDz(Ez, coefs_z, n_coefs_z, i, j, 0, 0) );
                    for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                        F(i, j, 0, 2*m-1) += dt * (
                            - rho(i, j, 0, rho_shift + 2*m-1) * inv_epsilon0
                            + T_Algo::DownwardDrr_over_r(Er, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                            + m * Et( i, j, 0, 2*m )/r
                            + T_Algo::DownwardDz(Ez, coefs_z, n_coefs_z, i, j, 0, 2*m-1) ); // Real part
                        F(i, j, 0, 2*m  ) += dt *(
                            - rho(i, j, 0, rho_shift + 2*m-1) * inv_epsilon0
                            + T_Algo::DownwardDrr_over_r(Er, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                            - m * Et( i, j, 0, 2*m-1 )/r
                            + T_Algo::DownwardDz(Ez, coefs_z, n_coefs_z, i, j, 0, 2*m  ) ); // Imaginary part
                    }
                } else { // r==0: on-axis corrections
                    // For m==0, Er is linear in r, for small r
                    // Therefore, the formula below regularizes the singularity
                    F(i, j, 0, 0) += dt * (
                        - rho(i, j, 0, rho_shift) * inv_epsilon0
                         + 4._rt*Er(i, j, 0, 0)/dr // regularization
                         + T_Algo::DownwardDz(Ez, coefs_z, n_coefs_z, i, j, 0, 0) );
                    // Ensure that F remains 0 for higher-order modes
                    for (int m=1; m<nmodes; m++) {
                        F(i, j, 0, 2*m-1) = 0._rt;
                        F(i, j, 0, 2*m  ) = 0._rt;
                    }
                }
            }

        ); // end of loop over cells

    } // end of loop over grid/tiles

}

#elif defined(WARPX_DIM_RSPHERE)

template<typename T_Algo>
void FiniteDifferenceSolver::EvolveFSpherical (
    amrex::MultiFab* Ffield,
    ablastr::fields::VectorField const & Efield,
    amrex::MultiFab* const rhofield,
    int const rhocomp,
    amrex::Real const dt ) {

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Ffield, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        const Array4<Real> F = Ffield->array(mfi);
        Array4<Real> const& Er = Efield[0]->array(mfi);
        Array4<Real> const& rho = rhofield->array(mfi);

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
        auto const n_coefs_r = static_cast<int>(m_stencil_coefs_r.size());

        // Extract spherical specific parameters
        Real const dr = m_dr;
        Real const rmin = m_rmin;

        // Extract tileboxes for which to loop
        Box const& tf  = mfi.tilebox(Ffield->ixType().toIntVect());

        Real constexpr inv_epsilon0 = 1./PhysConst::ep0;

        // Use the right shift in components:
        // - the first component corresponds to rho old (i.e. rhocomp=0)
        // - the next component corresponds to rho new (i.e. rhocomp=1)
        int rho_shift = 0;
        if (rhocomp == 1) {
            rho_shift = WarpX::ncomps;
        }

        // Loop over the cells and update the fields
        amrex::ParallelFor(tf,

            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                Real const r = rmin + i*dr; // r on a nodal grid (F is nodal in r)
                if (r != 0) { // Off-axis, regular equations
                    F(i, j, 0, 0) += dt * (
                        - rho(i, j, 0, rho_shift) * inv_epsilon0
                        + T_Algo::DownwardDrr2_over_r2(Er, r, dr, coefs_r, n_coefs_r, i, j, 0, 0) );
                } else { // r==0: on-axis corrections
                    // Er is linear in r, for small r
                    // Therefore, the formula below regularizes the singularity
                    F(i, j, 0, 0) += dt * (
                        - rho(i, j, 0, rho_shift) * inv_epsilon0
                         + 6._rt*Er(i, j, 0, 0)/dr); // regularization
                }
            }

        ); // end of loop over cells

    } // end of loop over grid/tiles

}

#endif
