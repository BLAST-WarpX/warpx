/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"

#include "BoundaryConditions/PMLComponent.H"
#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#endif
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"

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

/**
 * \brief Update the split magnetic-divergence cleaning field over one timestep
 */
void FiniteDifferenceSolver::EvolveGPML (
    amrex::MultiFab* Gfield,
    ablastr::fields::VectorField const Bfield,
    amrex::Real const dt )
{

    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    amrex::ignore_unused(Gfield, Bfield, dt);
    WARPX_ABORT_WITH_MESSAGE(
        "PML are only implemented in Cartesian geometry.");
#else
    if (m_grid_type == GridType::Collocated) {

        EvolveGPMLCartesian <CartesianNodalAlgorithm> ( Gfield, Bfield, dt );

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::Yee) {

        EvolveGPMLCartesian <CartesianYeeAlgorithm> ( Gfield, Bfield, dt );

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::CKC) {

        EvolveGPMLCartesian <CartesianCKCAlgorithm> ( Gfield, Bfield, dt );

    } else {
        WARPX_ABORT_WITH_MESSAGE("EvolveGPML: Unknown algorithm");
    }
#endif
}


#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)

template<typename T_Algo>
void FiniteDifferenceSolver::EvolveGPMLCartesian (
    amrex::MultiFab* Gfield,
    ablastr::fields::VectorField const Bfield,
    amrex::Real const dt )
{

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(*Gfield, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        amrex::Array4<amrex::Real> const& G = Gfield->array(mfi);
        amrex::Array4<amrex::Real const> const Bx = Bfield[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const By = Bfield[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const Bz = Bfield[2]->const_array(mfi);

        // Extract stencil coefficients
        amrex::Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        amrex::Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        amrex::Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract tileboxes for which to loop
        amrex::Box const& tf  = mfi.tilebox(Gfield->ixType().ixType());

        // Loop over the cells and update the fields
        amrex::ParallelFor(tf,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                G(i, j, k, PMLComp::x) += PhysConst::c2 * dt * (
                      T_Algo::UpwardDx(Bx, coefs_x, n_coefs_x, i, j, k, PMLComp::xx)
                    + T_Algo::UpwardDx(Bx, coefs_x, n_coefs_x, i, j, k, PMLComp::xy)
                    + T_Algo::UpwardDx(Bx, coefs_x, n_coefs_x, i, j, k, PMLComp::xz) );

                G(i, j, k, PMLComp::y) += PhysConst::c2 * dt * (
                      T_Algo::UpwardDy(By, coefs_y, n_coefs_y, i, j, k, PMLComp::yx)
                    + T_Algo::UpwardDy(By, coefs_y, n_coefs_y, i, j, k, PMLComp::yy)
                    + T_Algo::UpwardDy(By, coefs_y, n_coefs_y, i, j, k, PMLComp::yz) );

                G(i, j, k, PMLComp::z) += PhysConst::c2 * dt * (
                      T_Algo::UpwardDz(Bz, coefs_z, n_coefs_z, i, j, k, PMLComp::zx)
                    + T_Algo::UpwardDz(Bz, coefs_z, n_coefs_z, i, j, k, PMLComp::zy)
                    + T_Algo::UpwardDz(Bz, coefs_z, n_coefs_z, i, j, k, PMLComp::zz) );

            }

        );

    }

}

#endif // corresponds to if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_RCYLINDER)
