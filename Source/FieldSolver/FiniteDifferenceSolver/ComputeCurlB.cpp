/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
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
#   include "FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#endif

#include "Utils/TextMsg.H"
#include "WarpX.H"

using namespace amrex;

void FiniteDifferenceSolver::ComputeCurlB (
    ablastr::fields::VectorField& Efield,
    ablastr::fields::VectorField const& Bfield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update_E,
    int lev )
{
    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::Yee ||
        m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
        ComputeCurlBCylindrical <CylindricalYeeAlgorithm> (
            Efield, Bfield, eb_update_E, lev
        );

#elif defined(WARPX_DIM_RSPHERE)
        ComputeCurlBSpherical <SphericalYeeAlgorithm> (
            Efield, Bfield, eb_update_E, lev
        );

#else
    if (WarpX::grid_type == GridType::Staggered)
    {
        ComputeCurlBCartesian <CartesianYeeAlgorithm> (
            Efield, Bfield, eb_update_E, lev
        );
    } else {
        ComputeCurlBCartesian <CartesianNodalAlgorithm> (
            Efield, Bfield, eb_update_E, lev
        );
    }

#endif
    } else {
        amrex::Abort(Utils::TextMsg::Err(
            "ComputeCurlB: Unknown algorithm choice."));
    }
}

// /**
//   * \brief Calculate curl(B), output field on E/A/J mesh staggering
//   *
//   * \param[out] Efield  output of curl operation
//   * \param[in] Bfield   input staggered field, should be on B mesh staggering
//   * \param[in] eb_update_E specifies where the field should be updated
//   * \param[in] lev refinement level
//   */
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
template<typename T_Algo>
void FiniteDifferenceSolver::ComputeCurlBCylindrical (
    ablastr::fields::VectorField& Efield,
    ablastr::fields::VectorField const& Bfield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update_E,
    int lev
)
{
    amrex::ignore_unused(Efield, Bfield, eb_update_E, lev);
    WARPX_ABORT_WITH_MESSAGE("ComputeCurlBCylindrical not fully implemented");
}

#elif defined(WARPX_DIM_RSPHERE)
template<typename T_Algo>
void FiniteDifferenceSolver::ComputeCurlBSpherical (
    ablastr::fields::VectorField& Efield,
    ablastr::fields::VectorField const& Bfield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update_E,
    int lev
)
{
    amrex::ignore_unused(Efield, Bfield, eb_update_E, lev);
    WARPX_ABORT_WITH_MESSAGE("ComputeCurlBSpherical not fully implemented");
}

#else

template<typename T_Algo>
void FiniteDifferenceSolver::ComputeCurlBCartesian (
    ablastr::fields::VectorField & Efield,
    ablastr::fields::VectorField const& Bfield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update_E,
    int lev
)
{
    using ablastr::fields::Direction;

    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // reset Efield
    Efield[0]->setVal(0);
    Efield[1]->setVal(0);
    Efield[2]->setVal(0);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers) {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const &Ex = Efield[0]->array(mfi);
        Array4<Real> const &Ey = Efield[1]->array(mfi);
        Array4<Real> const &Ez = Efield[2]->array(mfi);
        Array4<Real const> const &Bx = Bfield[0]->const_array(mfi);
        Array4<Real const> const &By = Bfield[1]->const_array(mfi);
        Array4<Real const> const &Bz = Bfield[2]->const_array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries.
        amrex::Array4<int> update_Ex_arr, update_Ey_arr, update_Ez_arr;
        if (EB::enabled()) {
            update_Ex_arr = eb_update_E[0]->array(mfi);
            update_Ey_arr = eb_update_E[1]->array(mfi);
            update_Ez_arr = eb_update_E[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract tileboxes for which to loop
        Box const& tex  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tey  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Calculate the curl of B
        amrex::ParallelFor(tex, tey, tez,

            // Ex calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_Ex_arr && update_Ex_arr(i, j, k) == 0) { return; }

                Ex(i, j, k) = (
                    - T_Algo::DownwardDz(By, coefs_z, n_coefs_z, i, j, k)
                    + T_Algo::DownwardDy(Bz, coefs_y, n_coefs_y, i, j, k)
                );
            },

            // Ey calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_Ey_arr && update_Ey_arr(i, j, k) == 0) { return; }

                Ey(i, j, k) = (
                    - T_Algo::DownwardDx(Bz, coefs_x, n_coefs_x, i, j, k)
                    + T_Algo::DownwardDz(Bx, coefs_z, n_coefs_z, i, j, k)
                );
            },

            // Ez calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_Ez_arr && update_Ez_arr(i, j, k) == 0) { return; }

                Ez(i, j, k) = (
                    - T_Algo::DownwardDy(Bx, coefs_y, n_coefs_y, i, j, k)
                    + T_Algo::DownwardDx(By, coefs_x, n_coefs_x, i, j, k)
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
