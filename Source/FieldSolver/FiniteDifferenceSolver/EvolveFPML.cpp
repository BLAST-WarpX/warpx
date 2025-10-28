/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"

#include "BoundaryConditions/PMLComponent.H"
#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
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

using namespace amrex;

/**
 * \brief Update the E field, over one timestep
 */
void FiniteDifferenceSolver::EvolveFPML (
    amrex::MultiFab* Ffield,
    ablastr::fields::VectorField const Efield,
    amrex::Real const dt ) {

    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    amrex::ignore_unused(Ffield, Efield, dt);
    WARPX_ABORT_WITH_MESSAGE(
        "PML are only implemented in Cartesian geometry.");
#else
    if (m_grid_type == GridType::Collocated) {

        EvolveFPMLCartesian <CartesianNodalAlgorithm> ( Ffield, Efield, dt );

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::Yee) {

        EvolveFPMLCartesian <CartesianYeeAlgorithm> ( Ffield, Efield, dt );

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::CKC) {

        EvolveFPMLCartesian <CartesianCKCAlgorithm> ( Ffield, Efield, dt );

    } else {
        WARPX_ABORT_WITH_MESSAGE("EvolveFPML: Unknown algorithm");
    }
#endif
}


#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)

template<typename T_Algo>
void FiniteDifferenceSolver::EvolveFPMLCartesian (
    amrex::MultiFab* Ffield,
    ablastr::fields::VectorField const Efield,
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

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract tileboxes for which to loop
        Box const& tf  = mfi.tilebox(Ffield->ixType().ixType());

        // Loop over the cells and update the fields
        amrex::ParallelFor(tf,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                F(i, j, k, PMLComp::x) += dt * (
                      T_Algo::DownwardDx(Ex, coefs_x, n_coefs_x, i, j, k, PMLComp::xx)
                    + T_Algo::DownwardDx(Ex, coefs_x, n_coefs_x, i, j, k, PMLComp::xy)
                    + T_Algo::DownwardDx(Ex, coefs_x, n_coefs_x, i, j, k, PMLComp::xz) );

                F(i, j, k, PMLComp::y) += dt * (
                      T_Algo::DownwardDy(Ey, coefs_y, n_coefs_y, i, j, k, PMLComp::yx)
                    + T_Algo::DownwardDy(Ey, coefs_y, n_coefs_y, i, j, k, PMLComp::yy)
                    + T_Algo::DownwardDy(Ey, coefs_y, n_coefs_y, i, j, k, PMLComp::yz) );

                F(i, j, k, PMLComp::z) += dt * (
                      T_Algo::DownwardDz(Ez, coefs_z, n_coefs_z, i, j, k, PMLComp::zx)
                    + T_Algo::DownwardDz(Ez, coefs_z, n_coefs_z, i, j, k, PMLComp::zy)
                    + T_Algo::DownwardDz(Ez, coefs_z, n_coefs_z, i, j, k, PMLComp::zz) );

            }

        );

    }

}

#endif // corresponds to if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_RCYLINDER)
