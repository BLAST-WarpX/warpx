/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FiniteDifferenceSolver.H"

#include "EmbeddedBoundary/Enabled.H"
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
#   include "FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#elif defined(WARPX_DIM_RSPHERE)
#   include "FiniteDifferenceAlgorithms/SphericalYeeAlgorithm.H"
#else
#   include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#endif

#include "Utils/TextMsg.H"
#include "WarpX.H"

using namespace amrex;

void FiniteDifferenceSolver::ComputeLaplacian (
    ablastr::fields::ScalarField& out_field,
    ablastr::fields::ScalarField const& in_field,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update,
    int lev )
{
    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::Yee) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
        ComputeLaplacianCylindrical <CylindricalYeeAlgorithm> (
            out_field, in_field, eb_update, lev
        );

#elif defined(WARPX_DIM_RSPHERE)
        ComputeLaplacianSpherical <SphericalYeeAlgorithm> (
            out_field, in_field, eb_update, lev
        );

#else
    ComputeLaplacianCartesian <CartesianYeeAlgorithm> (
        out_field, in_field, eb_update, lev
    );

#endif
    } else {
        amrex::Abort(Utils::TextMsg::Err(
            "ComputeLaplacian: Unsupported FDTD algorithm choice."));
    }
}

/**
     * \brief Calculation of the Laplacian of the given scalar field.
     *
     * \param[out] out_field  output MultiFab at a given level
     * \param[in] in_field   input MultiFab at a given level
     * \param[in] eb_update  array indicating where the field should be updated with respect to the position of the embedded boundary
     * \param[in] lev  level number for the calculation
     */
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
template<typename T_Algo>
void FiniteDifferenceSolver::ComputeLaplacianCylindrical (
    ablastr::fields::ScalarField& out_field,
    ablastr::fields::ScalarField const& in_field,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update,
    int lev )
{
    amrex::ignore_unused(out_field, in_field, eb_update, lev);
    WARPX_ABORT_WITH_MESSAGE("ComputeLaplacianCylindrical not fully implemented");
}

#elif defined(WARPX_DIM_RSPHERE)
template<typename T_Algo>
void FiniteDifferenceSolver::ComputeLaplacianSpherical (
    ablastr::fields::ScalarField& out_field,
    ablastr::fields::ScalarField const& in_field,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update,
    int lev )
{
    amrex::ignore_unused(out_field, in_field, eb_update, lev);
    WARPX_ABORT_WITH_MESSAGE("ComputeLaplacianSpherical not fully implemented");
}

#else
template<typename T_Algo>
void FiniteDifferenceSolver::ComputeLaplacianCartesian (
    ablastr::fields::ScalarField& out_field,
    ablastr::fields::ScalarField const& in_field,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update,
    int lev )
{
    using ablastr::fields::Direction;

    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // reset output field
    out_field->setVal(0);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*in_field, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers) {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const &F = out_field->array(mfi);
        Array4<Real const> const &S = in_field->const_array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries.
        amrex::Array4<int> update_x_arr, update_y_arr, update_z_arr;
        if (EB::enabled()) {
            update_x_arr = eb_update[0]->array(mfi);
            update_y_arr = eb_update[1]->array(mfi);
            update_z_arr = eb_update[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract tileboxes for which to loop
        Box const& tb  = mfi.tilebox(out_field->ixType().toIntVect());

        // Calculate the Laplacian of the input scalar field (S)
        amrex::ParallelFor(tb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_x_arr) {
                    if (update_x_arr(i, j, k) == 0 || update_x_arr(i+1, j, k) == 0) { return; }
                    if (update_y_arr(i, j, k) == 0 || update_y_arr(i, j+1, k) == 0) { return; }
                    if (update_z_arr(i, j, k) == 0 || update_z_arr(i, j, k+1) == 0) { return; }
                }

                F(i, j, k) = (
                        T_Algo::Dxx(S, coefs_x, n_coefs_x, i, j, k) +
                        T_Algo::Dyy(S, coefs_y, n_coefs_y, i, j, k) +
                        T_Algo::Dzz(S, coefs_z, n_coefs_z, i, j, k)
                );
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
#endif


void FiniteDifferenceSolver::ComputeVectorLaplacian (
    ablastr::fields::VectorField& out_field,
    ablastr::fields::VectorField const& in_field,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update,
    int lev )
{
    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::Yee) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
        ComputeVectorLaplacianCylindrical <CylindricalYeeAlgorithm> (
            out_field, in_field, eb_update, lev
        );

#elif defined(WARPX_DIM_RSPHERE)
        ComputeVectorLaplacianSpherical <SphericalYeeAlgorithm> (
            out_field, in_field, eb_update, lev
        );

#else
    ComputeVectorLaplacianCartesian <CartesianYeeAlgorithm> (
        out_field, in_field, eb_update, lev
    );

#endif
    } else {
        amrex::Abort(Utils::TextMsg::Err(
            "ComputeVectorLaplacian: Unsupported FDTD algorithm choice."));
    }
}

/**
     * \brief Calculation of the vector Laplacian of the given vector field.
     *
     * \param[out] out_field  vector of output MultiFabs at a given level
     * \param[in] in_field   vector of input MultiFabs at a given level
     * \param[in] eb_update  array indicating where the field should be updated with respect to the position of the embedded boundary
     * \param[in] lev  level number for the calculation
     */
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
template<typename T_Algo>
void FiniteDifferenceSolver::ComputeVectorLaplacianCylindrical (
    ablastr::fields::VectorField& out_field,
    ablastr::fields::VectorField const& in_field,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update,
    int lev )
{
    amrex::ignore_unused(out_field, in_field, eb_update, lev);
    WARPX_ABORT_WITH_MESSAGE("ComputeVectorLaplacianCylindrical not fully implemented");
}

#elif defined(WARPX_DIM_RSPHERE)
template<typename T_Algo>
void FiniteDifferenceSolver::ComputeVectorLaplacianSpherical (
    ablastr::fields::VectorField& out_field,
    ablastr::fields::VectorField const& in_field,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update,
    int lev )
{
    amrex::ignore_unused(out_field, in_field, eb_update, lev);
    WARPX_ABORT_WITH_MESSAGE("ComputeVectorLaplacianSpherical not fully implemented");
}

#else
template<typename T_Algo>
void FiniteDifferenceSolver::ComputeVectorLaplacianCartesian (
    ablastr::fields::VectorField& out_field,
    ablastr::fields::VectorField const& in_field,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update,
    int lev )
{
    using ablastr::fields::Direction;

    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // reset output field
    out_field[0]->setVal(0);
    out_field[1]->setVal(0);
    out_field[2]->setVal(0);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*in_field[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers) {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const &Fx = out_field[0]->array(mfi);
        Array4<Real> const &Fy = out_field[1]->array(mfi);
        Array4<Real> const &Fz = out_field[2]->array(mfi);
        Array4<Real const> const &Gx = in_field[0]->const_array(mfi);
        Array4<Real const> const &Gy = in_field[1]->const_array(mfi);
        Array4<Real const> const &Gz = in_field[2]->const_array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries.
        amrex::Array4<int> update_x_arr, update_y_arr, update_z_arr;
        if (EB::enabled()) {
            update_x_arr = eb_update[0]->array(mfi);
            update_y_arr = eb_update[1]->array(mfi);
            update_z_arr = eb_update[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract tileboxes for which to loop
        Box const& tbx  = mfi.tilebox(out_field[0]->ixType().toIntVect());
        Box const& tby  = mfi.tilebox(out_field[1]->ixType().toIntVect());
        Box const& tbz  = mfi.tilebox(out_field[2]->ixType().toIntVect());

        // Calculate the vector Laplacian of the input field (G)
        amrex::ParallelFor(tbx, tby, tbz,

            // x calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_x_arr && update_x_arr(i, j, k) == 0) { return; }

                Fx(i, j, k) = (
                        T_Algo::Dxx(Gx, coefs_x, n_coefs_x, i, j, k) +
                        T_Algo::Dyy(Gx, coefs_y, n_coefs_y, i, j, k) +
                        T_Algo::Dzz(Gx, coefs_z, n_coefs_z, i, j, k)
                );
            },

            // y calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_y_arr && update_y_arr(i, j, k) == 0) { return; }

                Fy(i, j, k) = (
                        T_Algo::Dxx(Gy, coefs_x, n_coefs_x, i, j, k) +
                        T_Algo::Dyy(Gy, coefs_y, n_coefs_y, i, j, k) +
                        T_Algo::Dzz(Gy, coefs_z, n_coefs_z, i, j, k)
                );
            },

            // z calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_z_arr && update_z_arr(i, j, k) == 0) { return; }

                Fz(i, j, k) = (
                        T_Algo::Dxx(Gz, coefs_x, n_coefs_x, i, j, k) +
                        T_Algo::Dyy(Gz, coefs_y, n_coefs_y, i, j, k) +
                        T_Algo::Dzz(Gz, coefs_z, n_coefs_z, i, j, k)
                );
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
#endif
