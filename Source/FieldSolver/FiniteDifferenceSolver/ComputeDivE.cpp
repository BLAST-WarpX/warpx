/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FiniteDifferenceSolver.H"

#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#ifndef WARPX_DIM_RZ
#   include "FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#else
#   include "FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#endif

#include <ablastr/fields/MultiFabRegister.H>

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
void FiniteDifferenceSolver::ComputeDivE (
    ablastr::fields::VectorField const & Efield,
    amrex::MultiFab& divEfield
)
{
    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
#ifdef WARPX_DIM_RZ
    if (m_fdtd_algo == ElectromagneticSolverAlgo::Yee ||
        m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC){

        ComputeDivECylindrical <CylindricalYeeAlgorithm> ( Efield, divEfield );

#else
    if (m_grid_type == ablastr::utils::enums::GridType::Collocated) {

        ComputeDivECartesian <CartesianNodalAlgorithm> ( Efield, divEfield );

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::Yee ||
               m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {

        ComputeDivECartesian <CartesianYeeAlgorithm> ( Efield, divEfield );

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::CKC) {

        ComputeDivECartesian <CartesianCKCAlgorithm> ( Efield, divEfield );

#endif
    } else {
        WARPX_ABORT_WITH_MESSAGE("ComputeDivE: Unknown algorithm");
    }

}


#ifndef WARPX_DIM_RZ

template<typename T_Algo>
void FiniteDifferenceSolver::ComputeDivECartesian (
    ablastr::fields::VectorField const & Efield,
    amrex::MultiFab& divEfield ) {

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(divEfield, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const& divE = divEfield.array(mfi);
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
        Box const& tdive = mfi.tilebox(divEfield.ixType().toIntVect());

        // Loop over the cells and update the fields
        amrex::ParallelFor(tdive,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                divE(i, j, k) =
                      T_Algo::DownwardDx(Ex, coefs_x, n_coefs_x, i, j, k)
                    + T_Algo::DownwardDy(Ey, coefs_y, n_coefs_y, i, j, k)
                    + T_Algo::DownwardDz(Ez, coefs_z, n_coefs_z, i, j, k);
            }

        );

    }

}

#else // corresponds to ifndef WARPX_DIM_RZ

template<typename T_Algo>
void FiniteDifferenceSolver::ComputeDivECylindrical (
    ablastr::fields::VectorField const & Efield,
    amrex::MultiFab& divEfield
)
{
    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(divEfield, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        const Array4<Real> divE = divEfield.array(mfi);
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
        Box const& tdive  = mfi.tilebox(divEfield.ixType().toIntVect());

        // Loop over the cells and update the fields
        amrex::ParallelFor(tdive,

            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                Real const r = rmin + i*dr; // r on a nodal grid (F is nodal in r)
                if (r != 0) { // Off-axis, regular equations
                    divE(i, j, 0, 0) =
                          T_Algo::DownwardDrr_over_r(Er, r, dr, coefs_r, n_coefs_r, i, j, 0, 0)
                        + T_Algo::DownwardDz(Ez, coefs_z, n_coefs_z, i, j, 0, 0);
                    for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                        divE(i, j, 0, 2*m-1) =
                              T_Algo::DownwardDrr_over_r(Er, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                            + m * Et( i, j, 0, 2*m )/r
                            + T_Algo::DownwardDz(Ez, coefs_z, n_coefs_z, i, j, 0, 2*m-1); // Real part
                        divE(i, j, 0, 2*m  ) =
                              T_Algo::DownwardDrr_over_r(Er, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m)
                            - m * Et( i, j, 0, 2*m-1 )/r
                            + T_Algo::DownwardDz(Ez, coefs_z, n_coefs_z, i, j, 0, 2*m  ); // Imaginary part
                    }
                } else { // r==0: on-axis corrections
                    // For m==0, Er is linear in r, for small r
                    // Therefore, the formula below regularizes the singularity
                    divE(i, j, 0, 0) =
                           4._rt*Er(i, j, 0, 0)/dr // regularization
                         + T_Algo::DownwardDz(Ez, coefs_z, n_coefs_z, i, j, 0, 0);
                    // Ensure that divE remains 0 for higher-order modes
                    for (int m=1; m<nmodes; m++) {
                        divE(i, j, 0, 2*m-1) = 0._rt;
                        divE(i, j, 0, 2*m  ) = 0._rt;
                    }
                }
            }

        ); // end of loop over cells

    } // end of loop over grid/tiles

}

#endif // corresponds to ifndef WARPX_DIM_RZ
