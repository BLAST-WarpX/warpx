/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *          S. Eric Clark (Helion Energy)
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
#include "HybridPICModel/HybridPICModel.H"
#include "IsotropicOperators.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

#include <type_traits>

using namespace amrex;
using warpx::fields::FieldType;

namespace {
    /** EB-aware staggered->nodal interpolation for the hybrid Hall term.
     *
     * The plain ablastr::coarsen::Interp (coarsening ratio 1) averages the source
     * edge/face values around a node with equal weights. At the embedded boundary that
     * pulls COVERED-cell field values into the nodal J x B, polluting the near-wall Hall
     * term (a covered edge/face carries no physical current/field, but the average reads
     * it anyway). This variant averages only the UNCOVERED neighbours -- those whose
     * update mask (same staggering as the source) is nonzero -- and renormalizes by the
     * number kept, so covered edges/faces never enter the interpolation. Returns 0 only
     * if every neighbour is covered. Mirrors Interp's stencil for coarsening ratio 1. */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real InterpMasked (
        amrex::Array4<amrex::Real const> const& arr,
        amrex::Array4<int const> const& mask,
        amrex::GpuArray<int,3> const& sf,
        amrex::GpuArray<int,3> const& sc,
        int i, int j, int k, int comp)
    {
        using namespace amrex::literals;
        int const ic[3] = {i, j, k};
        int np[3], idx_min[3];
        for (int l = 0; l < 3; ++l) {
            np[l] = 1 + amrex::Math::abs(sf[l]-sc[l]);
            idx_min[l] = ic[l] - sc[l]*(1-sf[l]);
        }
        amrex::Real c = 0.0_rt, w = 0.0_rt;
        for (int kr = 0; kr < np[2]; ++kr) {
        for (int jr = 0; jr < np[1]; ++jr) {
        for (int ir = 0; ir < np[0]; ++ir) {
            int const ii = idx_min[0]+ir, jj = idx_min[1]+jr, kk = idx_min[2]+kr;
            amrex::Real const ww = (mask(ii,jj,kk) != 0) ? 1.0_rt : 0.0_rt;
            c += ww * arr(ii,jj,kk,comp);
            w += ww;
        }}}
        return (w > 0.0_rt) ? (c / w) : 0.0_rt;
    }
}

void FiniteDifferenceSolver::CalculateCurrentAmpere (
    ablastr::fields::VectorField & Jfield,
    ablastr::fields::VectorField const& Bfield,
    [[maybe_unused]]std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev )
{
    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
        CalculateCurrentAmpereCylindrical <CylindricalYeeAlgorithm> (
            Jfield, Bfield, eb_update_E, lev
        );

#elif defined(WARPX_DIM_RSPHERE)
        CalculateCurrentAmpereSpherical <SphericalYeeAlgorithm> (
            Jfield, Bfield, lev
        );

#else
    if (WarpX::grid_type == GridType::Staggered)
    {
        CalculateCurrentAmpereCartesian <CartesianYeeAlgorithm> (
            Jfield, Bfield, eb_update_E, lev
        );
    } else {
        CalculateCurrentAmpereCartesian <CartesianNodalAlgorithm> (
            Jfield, Bfield, eb_update_E, lev
        );
    }

#endif
    } else {
        amrex::Abort(Utils::TextMsg::Err(
            "CalculateCurrentAmpere: Unknown algorithm choice."));
    }
}

void FiniteDifferenceSolver::CalculateCurrentAmpereECT (
    ablastr::fields::VectorField & Jfield,
    ablastr::fields::VectorField const& Bfield,
    [[maybe_unused]] ablastr::fields::VectorField const& face_areas,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev )
{
    // The flux-weighted ECT curl is defined for the staggered Cartesian (Yee) grid
    // only; the caller (HybridPICModel::CalculatePlasmaCurrent) gates it on a
    // staggered embedded-boundary run. For any other configuration fall back to the
    // standard masked curl so the entry point stays safe.
    if (m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        CalculateCurrentAmpere(Jfield, Bfield, eb_update_E, lev);
#else
        if (WarpX::grid_type == GridType::Staggered)
        {
            CalculateCurrentAmpereCartesianECT <CartesianYeeAlgorithm> (
                Jfield, Bfield, face_areas, eb_update_E, lev
            );
        } else {
            CalculateCurrentAmpere(Jfield, Bfield, eb_update_E, lev);
        }
#endif
    } else {
        amrex::Abort(Utils::TextMsg::Err(
            "CalculateCurrentAmpereECT: Unknown algorithm choice."));
    }
}

// /**
//   * \brief Calculate total current from Ampere's law without displacement
//   * current i.e. J = 1/mu_0 curl x B.
//   *
//   * \param[out] Jfield  vector of total current MultiFabs at a given level
//   * \param[in] Bfield   vector of magnetic field MultiFabs at a given level
//   * \param[in] eb_update_E specifies where the plasma current should be calculated.
//   * \param[in] lev refinement level
//   */
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
template<typename T_Algo>
void FiniteDifferenceSolver::CalculateCurrentAmpereCylindrical (
    ablastr::fields::VectorField& Jfield,
    ablastr::fields::VectorField const& Bfield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev
)
{
    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Jfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const& Jr = Jfield[0]->array(mfi);
        Array4<Real> const& Jtheta = Jfield[1]->array(mfi);
        Array4<Real> const& Jz = Jfield[2]->array(mfi);
        Array4<Real> const& Br = Bfield[0]->array(mfi);
        Array4<Real> const& Btheta = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries.
        // The plasma current is stored at the same locations as the E-field,
        // therefore the `eb_update_E` multifab also appropriately specifies
        // where the plasma current should be calculated.
        amrex::Array4<int> update_Jr_arr, update_Jtheta_arr, update_Jz_arr;
        if (EB::enabled()) {
            update_Jr_arr = eb_update_E[0]->array(mfi);
            update_Jtheta_arr = eb_update_E[1]->array(mfi);
            update_Jz_arr = eb_update_E[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
        int const n_coefs_r = static_cast<int>(m_stencil_coefs_r.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract cylindrical specific parameters
        Real const dr = m_dr;
        int const nmodes = m_nmodes;
        Real const rmin = m_rmin;

        // Extract tileboxes for which to loop with 1 guard cell included
        Box const& tjr  = mfi.tilebox(Jfield[0]->ixType().toIntVect(), IntVect(1));
        Box const& tjtheta  = mfi.tilebox(Jfield[1]->ixType().toIntVect(), IntVect(1));
        Box const& tjz  = mfi.tilebox(Jfield[2]->ixType().toIntVect(), IntVect(1));

        Real const one_over_mu0 = 1._rt / PhysConst::mu0;

        // Calculate the total current, using Ampere's law, on the same grid
        // as the E-field
        amrex::ParallelFor(tjr, tjtheta, tjz,

            // Jr calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Jr_arr && update_Jr_arr(i, j, 0) == 0) { return; }

                // Mode m=0
                Jr(i, j, 0, 0) = one_over_mu0 * (
                    - T_Algo::DownwardDz(Btheta, coefs_z, n_coefs_z, i, j, 0, 0)
                );

                // Higher-order modes
                // r on cell-centered point (Jr is cell-centered in r)
                Real const r = rmin + (i + 0.5_rt)*dr;
                for (int m=1; m<nmodes; m++) {
                    Jr(i, j, 0, 2*m-1) = one_over_mu0 * (
                        - T_Algo::DownwardDz(Btheta, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                        + m * Bz(i, j, 0, 2*m  ) / r
                    );  // Real part
                    Jr(i, j, 0, 2*m  ) = one_over_mu0 * (
                        - T_Algo::DownwardDz(Btheta, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                        - m * Bz(i, j, 0, 2*m-1) / r
                    ); // Imaginary part
                }
            },

            // Jtheta calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Jtheta_arr && update_Jtheta_arr(i, j, 0) == 0) { return; }

                // r on a nodal point (Jtheta is nodal in r)
                Real const r = rmin + i*dr;
                // Off-axis, regular curl
                if (r > 0.5_rt*dr) {
                    // Mode m=0
                    Jtheta(i, j, 0, 0) = one_over_mu0 * (
                        - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 0)
                        + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 0)
                    );

                    // Higher-order modes
                    for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                        Jtheta(i, j, 0, 2*m-1) = one_over_mu0 * (
                            - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                            + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                        ); // Real part
                        Jtheta(i, j, 0, 2*m  ) = one_over_mu0 * (
                            - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 2*m  )
                            + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                        ); // Imaginary part
                    }
                // r==0: on-axis corrections
                } else {
                    // Ensure that Jtheta remains 0 on axis (except for m=1)
                    // Mode m=0
                    Jtheta(i, j, 0, 0) = 0.;
                    // Higher-order modes
                    for (int m=1; m<nmodes; m++) {
                        if (m == 1){
                            // The same logic as is used in the E-field update for the fully
                            // electromagnetic FDTD case is used here.
                            Jtheta(i,j,0,2*m-1) =  Jr(i,j,0,2*m  );
                            Jtheta(i,j,0,2*m  ) = -Jr(i,j,0,2*m-1);
                        } else {
                            Jtheta(i, j, 0, 2*m-1) = 0.;
                            Jtheta(i, j, 0, 2*m  ) = 0.;
                        }
                    }
                }
            },

            // Jz calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Jz_arr && update_Jz_arr(i, j, 0) == 0) { return; }

                // r on a nodal point (Jz is nodal in r)
                Real const r = rmin + i*dr;
                // Off-axis, regular curl
                if (r > 0.5_rt*dr) {
                    // Mode m=0
                    Jz(i, j, 0, 0) = one_over_mu0 * (
                       T_Algo::DownwardDrr_over_r(Btheta, r, dr, coefs_r, n_coefs_r, i, j, 0, 0)
                    );
                    // Higher-order modes
                    for (int m=1 ; m<nmodes ; m++) {
                        Jz(i, j, 0, 2*m-1) = one_over_mu0 * (
                            - m * Br(i, j, 0, 2*m  ) / r
                            + T_Algo::DownwardDrr_over_r(Btheta, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                        ); // Real part
                        Jz(i, j, 0, 2*m  ) = one_over_mu0 * (
                            m * Br(i, j, 0, 2*m-1) / r
                            + T_Algo::DownwardDrr_over_r(Btheta, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m  )
                        ); // Imaginary part
                    }
                // r==0: on-axis corrections
                } else {
                    // For m==0, Btheta is linear in r, for small r
                    // Therefore, the formula below regularizes the singularity
                    Jz(i, j, 0, 0) = one_over_mu0 * 4 * Btheta(i, j, 0, 0) / dr;
                    // Ensure that Jz remains 0 for higher-order modes
                    for (int m=1; m<nmodes; m++) {
                        Jz(i, j, 0, 2*m-1) = 0.;
                        Jz(i, j, 0, 2*m  ) = 0.;
                    }
                }
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

#elif defined(WARPX_DIM_RSPHERE)
template<typename T_Algo>
void FiniteDifferenceSolver::CalculateCurrentAmpereSpherical (
    ablastr::fields::VectorField& Jfield,
    ablastr::fields::VectorField const& Bfield,
    int lev
)
{
    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Jfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const& Jr = Jfield[0]->array(mfi);
        Array4<Real> const& Jtheta = Jfield[1]->array(mfi);
        Array4<Real> const& Jphi = Jfield[2]->array(mfi);
        Array4<Real> const& Btheta = Bfield[1]->array(mfi);
        Array4<Real> const& Bphi = Bfield[2]->array(mfi);

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
        int const n_coefs_r = static_cast<int>(m_stencil_coefs_r.size());

        // Extract cylindrical specific parameters
        Real const dr = m_dr;
        Real const rmin = m_rmin;

        // Extract tileboxes for which to loop with 1 guard cell included
        Box const& tjr  = mfi.tilebox(Jfield[0]->ixType().toIntVect(), IntVect(1));
        Box const& tjtheta  = mfi.tilebox(Jfield[1]->ixType().toIntVect(), IntVect(1));
        Box const& tjphi  = mfi.tilebox(Jfield[2]->ixType().toIntVect(), IntVect(1));

        Real const one_over_mu0 = 1._rt / PhysConst::mu0;

        // Calculate the total current, using Ampere's law, on the same grid
        // as the E-field
        amrex::ParallelFor(tjr, tjtheta, tjphi,

            // Jr calculation
            [=] AMREX_GPU_DEVICE (int i, int /*j*/, int /*k*/){
                Jr(i, 0, 0, 0) = 0._rt;
            },

            // Jtheta calculation
            [=] AMREX_GPU_DEVICE (int i, int /*j*/, int /*k*/){
                // r on a nodal point (Jtheta is nodal in r)
                Real const r = rmin + i*dr;
                // Off-axis, regular curl
                if (r > 0.5_rt*dr) {
                    // Mode m=0
                    Jtheta(i, 0, 0, 0) = one_over_mu0 * (
                        - T_Algo::DownwardDrr_over_r(Bphi, r, dr, coefs_r, n_coefs_r, i, 0, 0, 0));
                } else { // r==0: on-axis corrections
                    // Ensure that Jtheta remains 0 on axis
                    Jtheta(i, 0, 0, 0) = 0.;
                }
            },

            // Jphi calculation
            [=] AMREX_GPU_DEVICE (int i, int /*j*/, int /*k*/){
                // r on a nodal point (Jphi is nodal in r)
                Real const r = rmin + i*dr;
                // Off-axis, regular curl
                if (r > 0.5_rt*dr) {
                    Jphi(i, 0, 0, 0) = one_over_mu0 * (
                       T_Algo::DownwardDrr_over_r(Btheta, r, dr, coefs_r, n_coefs_r, i, 0, 0, 0)
                    );
                // r==0: on-axis corrections
                } else {
                    // Btheta is linear in r, for small r
                    // Therefore, the formula below regularizes the singularity
                    Jphi(i, 0, 0, 0) = one_over_mu0 * 4 * Btheta(i, 0, 0, 0) / dr;
                }
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

#else

template<typename T_Algo>
void FiniteDifferenceSolver::CalculateCurrentAmpereCartesian (
    ablastr::fields::VectorField& Jfield,
    ablastr::fields::VectorField const& Bfield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev
)
{
    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Jfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers) {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const &Jx = Jfield[0]->array(mfi);
        Array4<Real> const &Jy = Jfield[1]->array(mfi);
        Array4<Real> const &Jz = Jfield[2]->array(mfi);
        Array4<Real const> const &Bx = Bfield[0]->const_array(mfi);
        Array4<Real const> const &By = Bfield[1]->const_array(mfi);
        Array4<Real const> const &Bz = Bfield[2]->const_array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries.
        // The plasma current is stored at the same locations as the E-field,
        // therefore the `eb_update_E` multifab also appropriately specifies
        // where the plasma current should be calculated.
        amrex::Array4<int> update_Jx_arr, update_Jy_arr, update_Jz_arr;
        if (EB::enabled()) {
            update_Jx_arr = eb_update_E[0]->array(mfi);
            update_Jy_arr = eb_update_E[1]->array(mfi);
            update_Jz_arr = eb_update_E[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract tileboxes for which to loop with 1 guard cell included
        Box const& tjx = mfi.tilebox(Jfield[0]->ixType().toIntVect(), IntVect(1));
        Box const& tjy = mfi.tilebox(Jfield[1]->ixType().toIntVect(), IntVect(1));
        Box const& tjz = mfi.tilebox(Jfield[2]->ixType().toIntVect(), IntVect(1));

        Real const one_over_mu0 = 1._rt / PhysConst::mu0;

        // Calculate the total current, using Ampere's law, on the same grid
        // as the E-field
        amrex::ParallelFor(tjx, tjy, tjz,

            // Jx calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Zero the current in fully-covered cells (eb_update flag == 0):
                // no plasma current inside the conductor (follows the update_E flag).
                if (update_Jx_arr && update_Jx_arr(i, j, k) == 0) { Jx(i, j, k) = 0._rt; return; }

                Jx(i, j, k) = one_over_mu0 * (
                    - T_Algo::DownwardDz(By, coefs_z, n_coefs_z, i, j, k)
                    + T_Algo::DownwardDy(Bz, coefs_y, n_coefs_y, i, j, k)
                );
            },

            // Jy calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Zero the current in fully-covered cells (eb_update flag == 0).
                if (update_Jy_arr && update_Jy_arr(i, j, k) == 0) { Jy(i, j, k) = 0._rt; return; }

                Jy(i, j, k) = one_over_mu0 * (
                    - T_Algo::DownwardDx(Bz, coefs_x, n_coefs_x, i, j, k)
                    + T_Algo::DownwardDz(Bx, coefs_z, n_coefs_z, i, j, k)
                );
            },

            // Jz calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Zero the current in fully-covered cells (eb_update flag == 0).
                if (update_Jz_arr && update_Jz_arr(i, j, k) == 0) { Jz(i, j, k) = 0._rt; return; }

                Jz(i, j, k) = one_over_mu0 * (
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

template<typename T_Algo>
void FiniteDifferenceSolver::CalculateCurrentAmpereCartesianECT (
    ablastr::fields::VectorField& Jfield,
    ablastr::fields::VectorField const& Bfield,
    [[maybe_unused]] ablastr::fields::VectorField const& face_areas,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev
)
{
    // Flux-weighted ("Form A") Ampere curl for the conformal embedded boundary.
    // Each B value entering the standard Yee curl is scaled by the open-fluid
    // fraction of the cut face it lives on, frac = face_areas / (full face area),
    // so curl(B) = J is a signed sum of open-face fluxes. Covered faces have
    // face_areas = 0 -> frac = 0 -> they drop out, so no separate covered-B fill
    // is needed. On interior edges every frac = 1 and each weighted difference
    // reduces termwise to the standard Yee stencil (inv_d * (B_a - B_b)), so this
    // matches CalculateCurrentAmpereCartesian byte-for-byte away from the wall.
    //
    // The open-face-area weighting is only well defined with the face_areas field
    // in 3D, where face_areas[0/1/2] are the genuine cut areas of the Bx/By/Bz
    // faces. In 2D (WARPX_DIM_XZ) only face_areas[1] (the out-of-plane By face) is
    // an area; the in-plane Bx/Bz live on edges whose open fraction lives in
    // edge_lengths, not face_areas. Since the validated Form A scheme is 3D
    // (Docs/eb_fill_review/cut_circulation_3d_poc.py) and the spec passes only
    // face_areas, the weighted path is restricted to 3D; in 2D we fall back to the
    // standard masked Yee curl so nothing spurious is emitted. See report notes.
#if defined(WARPX_DIM_3D)
    using namespace amrex::literals;

    // Full (uncut) face areas for the open-fraction normalization, from the cell
    // sizes at this level: Bx face = dy*dz, By face = dx*dz, Bz face = dx*dy.
    amrex::GpuArray<amrex::Real, 3> dx_lev{};
    {
        const auto cs = WarpX::GetInstance().Geom(lev).CellSizeArray();
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { dx_lev[d] = cs[d]; }
    }
    amrex::Real const inv_full_area_x = 1._rt / (dx_lev[1]*dx_lev[2]); // 1/(dy*dz)
    amrex::Real const inv_full_area_y = 1._rt / (dx_lev[0]*dx_lev[2]); // 1/(dx*dz)
    amrex::Real const inv_full_area_z = 1._rt / (dx_lev[0]*dx_lev[1]); // 1/(dx*dy)

    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Jfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers) {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const &Jx = Jfield[0]->array(mfi);
        Array4<Real> const &Jy = Jfield[1]->array(mfi);
        Array4<Real> const &Jz = Jfield[2]->array(mfi);
        Array4<Real const> const &Bx = Bfield[0]->const_array(mfi);
        Array4<Real const> const &By = Bfield[1]->const_array(mfi);
        Array4<Real const> const &Bz = Bfield[2]->const_array(mfi);

        // Open cut-face areas (SI m^2), B-staggered like the B components above.
        Array4<Real const> const &Sx = face_areas[0]->const_array(mfi);
        Array4<Real const> const &Sy = face_areas[1]->const_array(mfi);
        Array4<Real const> const &Sz = face_areas[2]->const_array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries.
        // The plasma current is stored at the same locations as the E-field,
        // therefore the `eb_update_E` multifab also appropriately specifies
        // where the plasma current should be calculated.
        amrex::Array4<int> update_Jx_arr, update_Jy_arr, update_Jz_arr;
        if (EB::enabled()) {
            update_Jx_arr = eb_update_E[0]->array(mfi);
            update_Jy_arr = eb_update_E[1]->array(mfi);
            update_Jz_arr = eb_update_E[2]->array(mfi);
        }

        // Inverse cell sizes (match the inv_d used by the Yee DownwardD stencils,
        // i.e. coefs_*[0], so the unweighted reduction is bit-identical).
        Real const inv_dx = 1._rt / dx_lev[0];
        Real const inv_dy = 1._rt / dx_lev[1];
        Real const inv_dz = 1._rt / dx_lev[2];

        // Extract tileboxes for which to loop with 1 guard cell included
        Box const& tjx = mfi.tilebox(Jfield[0]->ixType().toIntVect(), IntVect(1));
        Box const& tjy = mfi.tilebox(Jfield[1]->ixType().toIntVect(), IntVect(1));
        Box const& tjz = mfi.tilebox(Jfield[2]->ixType().toIntVect(), IntVect(1));

        Real const one_over_mu0 = 1._rt / PhysConst::mu0;

        // Calculate the total current, using the flux-weighted Ampere's law, on the
        // same grid as the E-field.
        amrex::ParallelFor(tjx, tjy, tjz,

            // Jx calculation: circulates the open-face fluxes of Bz (along y) and
            // By (along z). frac_z = Sz/(dx*dy), frac_y = Sy/(dx*dz).
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Zero the current in fully-covered cells (eb_update flag == 0):
                // no plasma current inside the conductor (follows the update_E flag).
                if (update_Jx_arr && update_Jx_arr(i, j, k) == 0) { Jx(i, j, k) = 0._rt; return; }

                Jx(i, j, k) = one_over_mu0 * (
                      inv_dy * ( Sz(i, j  , k) * inv_full_area_z * Bz(i, j  , k)
                               - Sz(i, j-1, k) * inv_full_area_z * Bz(i, j-1, k) )
                    - inv_dz * ( Sy(i, j, k  ) * inv_full_area_y * By(i, j, k  )
                               - Sy(i, j, k-1) * inv_full_area_y * By(i, j, k-1) )
                );
            },

            // Jy calculation: circulates the open-face fluxes of Bx (along z) and
            // Bz (along x). frac_x = Sx/(dy*dz), frac_z = Sz/(dx*dy).
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Zero the current in fully-covered cells (eb_update flag == 0).
                if (update_Jy_arr && update_Jy_arr(i, j, k) == 0) { Jy(i, j, k) = 0._rt; return; }

                Jy(i, j, k) = one_over_mu0 * (
                      inv_dz * ( Sx(i, j, k  ) * inv_full_area_x * Bx(i, j, k  )
                               - Sx(i, j, k-1) * inv_full_area_x * Bx(i, j, k-1) )
                    - inv_dx * ( Sz(i  , j, k) * inv_full_area_z * Bz(i  , j, k)
                               - Sz(i-1, j, k) * inv_full_area_z * Bz(i-1, j, k) )
                );
            },

            // Jz calculation: circulates the open-face fluxes of By (along x) and
            // Bx (along y). frac_y = Sy/(dx*dz), frac_x = Sx/(dy*dz).
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Zero the current in fully-covered cells (eb_update flag == 0).
                if (update_Jz_arr && update_Jz_arr(i, j, k) == 0) { Jz(i, j, k) = 0._rt; return; }

                Jz(i, j, k) = one_over_mu0 * (
                      inv_dx * ( Sy(i  , j, k) * inv_full_area_y * By(i  , j, k)
                               - Sy(i-1, j, k) * inv_full_area_y * By(i-1, j, k) )
                    - inv_dy * ( Sx(i, j  , k) * inv_full_area_x * Bx(i, j  , k)
                               - Sx(i, j-1, k) * inv_full_area_x * Bx(i, j-1, k) )
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
#else
    // 2D (XZ): face_areas alone does not carry the in-plane open fractions, so use
    // the standard masked Yee curl (see comment above).
    CalculateCurrentAmpereCartesian<T_Algo>(Jfield, Bfield, eb_update_E, lev);
#endif
}
#endif


void FiniteDifferenceSolver::HybridPICSolveE (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField& Jfield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    [[maybe_unused]]std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev, HybridPICModel const* hybrid_model,
    const bool solve_for_Faraday)
{
    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)

        HybridPICSolveECylindrical <CylindricalYeeAlgorithm> (
            Efield, Jfield, Jifield, Bfield, rhofield, Pefield,
            eb_update_E, lev, hybrid_model, solve_for_Faraday
        );

#elif defined(WARPX_DIM_RSPHERE)

        HybridPICSolveESpherical <SphericalYeeAlgorithm> (
            Efield, Jfield, Jifield, Bfield, rhofield, Pefield,
            lev, hybrid_model, solve_for_Faraday
        );

#else
    if (WarpX::grid_type == GridType::Staggered)
    {
        HybridPICSolveECartesian <CartesianYeeAlgorithm> (
            Efield, Jfield, Jifield, Bfield, rhofield, Pefield,
            eb_update_E, lev, hybrid_model, solve_for_Faraday
        );
    } else {
        HybridPICSolveECartesian <CartesianNodalAlgorithm> (
            Efield, Jfield, Jifield, Bfield, rhofield, Pefield,
            eb_update_E, lev, hybrid_model, solve_for_Faraday
        );
    }
#endif
    } else {
        amrex::Abort(Utils::TextMsg::Err(
            "HybridSolveE: The hybrid-PIC electromagnetic solver algorithm must be used"));
    }
}

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
template<typename T_Algo>
void FiniteDifferenceSolver::HybridPICSolveECylindrical (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev, HybridPICModel const* hybrid_model,
    const bool solve_for_Faraday )
{
    // Both steps below do not currently support m > 0 and should be
    // modified if such support wants to be added
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_nmodes == 1),
        "Ohm's law solver only support m = 0 azimuthal mode at present.");

    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    using namespace ablastr::coarsen::sample;

    // get hybrid model parameters
    const auto eta = hybrid_model->m_eta;
    const auto eta_h = hybrid_model->m_eta_h;
    const auto rho_floor = hybrid_model->m_n_floor * PhysConst::q_e;
    const auto resistivity_has_J_dependence = hybrid_model->m_resistivity_has_J_dependence;
    const auto hyper_resistivity_has_B_dependence = hybrid_model->m_hyper_resistivity_has_B_dependence;
    const bool include_hyper_resistivity_term = hybrid_model->m_include_hyper_resistivity_term;

    const bool include_external_fields = hybrid_model->m_add_external_fields;

    const bool holmstrom_vacuum_region = hybrid_model->m_holmstrom_vacuum_region;

    auto & warpx = WarpX::GetInstance();
    const amrex::Real t_new = warpx.gett_new(lev);
    ablastr::fields::VectorField Bfield_external, Efield_external;
    if (include_external_fields) {
        Bfield_external = warpx.m_fields.get_alldirs(FieldType::hybrid_B_fp_external, 0); // lev=0
        Efield_external = warpx.m_fields.get_alldirs(FieldType::hybrid_E_fp_external, 0); // lev=0
    }

    // Index type required for interpolating fields from their respective
    // staggering to the Ex, Ey, Ez locations
    amrex::GpuArray<int, 3> const& Er_stag = hybrid_model->Ex_IndexType;
    amrex::GpuArray<int, 3> const& Etheta_stag = hybrid_model->Ey_IndexType;
    amrex::GpuArray<int, 3> const& Ez_stag = hybrid_model->Ez_IndexType;
    amrex::GpuArray<int, 3> const& Jr_stag = hybrid_model->Jx_IndexType;
    amrex::GpuArray<int, 3> const& Jtheta_stag = hybrid_model->Jy_IndexType;
    amrex::GpuArray<int, 3> const& Jz_stag = hybrid_model->Jz_IndexType;
    amrex::GpuArray<int, 3> const& Br_stag = hybrid_model->Bx_IndexType;
    amrex::GpuArray<int, 3> const& Btheta_stag = hybrid_model->By_IndexType;
    amrex::GpuArray<int, 3> const& Bz_stag = hybrid_model->Bz_IndexType;

    // Parameters for `interp` that maps from Yee to nodal mesh and back
    amrex::GpuArray<int, 3> const& nodal = {1, 1, 1};
    // The "coarsening is just 1 i.e. no coarsening"
    amrex::GpuArray<int, 3> const& coarsen = {1, 1, 1};

    // The E-field calculation is done in 2 steps:
    // 1) The J x B term is calculated on a nodal mesh in order to ensure
    //    energy conservation.
    // 2) The nodal E-field values are averaged onto the Yee grid and the
    //    electron pressure & resistivity terms are added (these terms are
    //    naturally located on the Yee grid).

    // Create a temporary multifab to hold the nodal E-field values
    // Note the multifab has 3 values for Ex, Ey and Ez which we can do here
    // since all three components will be calculated on the same grid.
    // Also note that enE_nodal_mf does not need to have any guard cells since
    // these values will be interpolated to the Yee mesh which is contained
    // by the nodal mesh.
    auto const& ba = convert(rhofield.boxArray(), IntVect::TheNodeVector());
    MultiFab enE_nodal_mf(ba, rhofield.DistributionMap(), 3, IntVect::TheZeroVector());

    // Loop through the grids, and over the tiles within each grid for the
    // initial, nodal calculation of E
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(enE_nodal_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        Array4<Real> const& enE_nodal = enE_nodal_mf.array(mfi);
        Array4<Real const> const& Jr = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jtheta = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& Jir = Jifield[0]->const_array(mfi);
        Array4<Real const> const& Jit = Jifield[1]->const_array(mfi);
        Array4<Real const> const& Jiz = Jifield[2]->const_array(mfi);
        Array4<Real const> const& Br = Bfield[0]->const_array(mfi);
        Array4<Real const> const& Btheta = Bfield[1]->const_array(mfi);
        Array4<Real const> const& Bz = Bfield[2]->const_array(mfi);

        Array4<Real> Br_ext, Btheta_ext, Bz_ext;
        if (include_external_fields) {
            Br_ext = Bfield_external[0]->array(mfi);
            Btheta_ext = Bfield_external[1]->array(mfi);
            Bz_ext = Bfield_external[2]->array(mfi);
        }

        // Loop over the cells and update the nodal E field
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

            // interpolate the total current to a nodal grid
            auto const jr_interp = Interp(Jr, Jr_stag, nodal, coarsen, i, j, 0, 0);
            auto const jtheta_interp = Interp(Jtheta, Jtheta_stag, nodal, coarsen, i, j, 0, 0);
            auto const jz_interp = Interp(Jz, Jz_stag, nodal, coarsen, i, j, 0, 0);

            // interpolate the ion current to a nodal grid
            auto const jir_interp = Interp(Jir, Jr_stag, nodal, coarsen, i, j, 0, 0);
            auto const jit_interp = Interp(Jit, Jtheta_stag, nodal, coarsen, i, j, 0, 0);
            auto const jiz_interp = Interp(Jiz, Jz_stag, nodal, coarsen, i, j, 0, 0);

            // interpolate the B field to a nodal grid
            auto Br_interp = Interp(Br, Br_stag, nodal, coarsen, i, j, 0, 0);
            auto Btheta_interp = Interp(Btheta, Btheta_stag, nodal, coarsen, i, j, 0, 0);
            auto Bz_interp = Interp(Bz, Bz_stag, nodal, coarsen, i, j, 0, 0);

            if (include_external_fields) {
                Br_interp += Interp(Br_ext, Br_stag, nodal, coarsen, i, j, 0, 0);
                Btheta_interp += Interp(Btheta_ext, Btheta_stag, nodal, coarsen, i, j, 0, 0);
                Bz_interp += Interp(Bz_ext, Bz_stag, nodal, coarsen, i, j, 0, 0);
            }

            // calculate enE = (J - Ji) x B
            enE_nodal(i, j, 0, 0) = (
                (jtheta_interp - jit_interp) * Bz_interp
                - (jz_interp - jiz_interp) * Btheta_interp
            );
            enE_nodal(i, j, 0, 1) = (
                (jz_interp - jiz_interp) * Br_interp
                - (jr_interp - jir_interp) * Bz_interp
            );
            enE_nodal(i, j, 0, 2) = (
                (jr_interp - jir_interp) * Btheta_interp
                - (jtheta_interp - jit_interp) * Br_interp
            );
        });

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    // Loop through the grids, and over the tiles within each grid again
    // for the Yee grid calculation of the E field
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const& Er = Efield[0]->array(mfi);
        Array4<Real> const& Etheta = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real const> const& Jr = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jtheta = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& enE = enE_nodal_mf.const_array(mfi);
        Array4<Real const> const& rho = rhofield.const_array(mfi);
        Array4<Real const> const& Pe = Pefield.const_array(mfi);
        Array4<Real> const& Br = Bfield[0]->array(mfi);
        Array4<Real> const& Btheta = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries
        amrex::Array4<int> update_Er_arr, update_Etheta_arr, update_Ez_arr;
        if (EB::enabled()) {
            update_Er_arr = eb_update_E[0]->array(mfi);
            update_Etheta_arr = eb_update_E[1]->array(mfi);
            update_Ez_arr = eb_update_E[2]->array(mfi);
        }

        Array4<Real> Er_ext, Etheta_ext, Ez_ext;
        if (include_external_fields) {
            Er_ext = Efield_external[0]->array(mfi);
            Etheta_ext = Efield_external[1]->array(mfi);
            Ez_ext = Efield_external[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
        int const n_coefs_r = static_cast<int>(m_stencil_coefs_r.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract cylindrical specific parameters
        Real const dr = m_dr;
        Real const rmin = m_rmin;

        Box const& ter  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tet  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Loop over the cells and update the E field
        amrex::ParallelFor(ter, tet, tez,

            // Er calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Er_arr && update_Er_arr(i, j, 0) == 0) { return; }

                // Interpolate to get the appropriate charge density in space
                const Real rho_val = Interp(rho, nodal, Er_stag, coarsen, i, j, 0, 0);

                if (rho_val < rho_floor && holmstrom_vacuum_region) {
                    Er(i, j, 0) = 0._rt;
                } else {
                    // Get the gradient of the electron pressure if the longitudinal part of
                    // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                    const Real grad_Pe = (!solve_for_Faraday) ?
                        T_Algo::UpwardDr(Pe, coefs_r, n_coefs_r, i, j, 0, 0)
                        : 0._rt;

                    // interpolate the nodal neE values to the Yee grid
                    const auto enE_r = Interp(enE, nodal, Er_stag, coarsen, i, j, 0, 0);

                    // safety condition since we divide by rho
                    const auto rho_val_limited = std::max(rho_val, rho_floor);

                    Er(i, j, 0) = (enE_r - grad_Pe) / rho_val_limited;
                }

                // Add resistivity only if E field value is used to update B
                if (solve_for_Faraday) {
                    // the embedded-boundary Dirichlet mirror of rho is
                    // negative inside the conductor: keep the resistivity
                    // parsers on their physical domain
                    // |rho|: eta on the covered/mirror side uses the reflected
                    // plasma density and is never driven negative (see 3D notes).
                    const Real rho_val_eta = std::abs(rho_val);
                    Real jtot_val = 0._rt;
                    if (resistivity_has_J_dependence) {
                        // Interpolate current to appropriate staggering to match E field
                        const Real jr_val = Jr(i, j, 0);
                        const Real jtheta_val = Interp(Jtheta, Jtheta_stag, Er_stag, coarsen, i, j, 0, 0);
                        const Real jz_val = Interp(Jz, Jz_stag, Er_stag, coarsen, i, j, 0, 0);
                        jtot_val = std::sqrt(jr_val*jr_val + jtheta_val*jtheta_val + jz_val*jz_val);
                    }

                    Er(i, j, 0) += eta(rho_val_eta, jtot_val, t_new) * Jr(i, j, 0);

                    if (include_hyper_resistivity_term) {

                        // Interpolate B field to appropriate staggering to match E field
                        Real btot_val = 0._rt;
                        if (hyper_resistivity_has_B_dependence) {
                            const Real br_val = Interp(Br, Br_stag, Er_stag, coarsen, i, j, 0, 0);
                            const Real bt_val = Interp(Btheta, Btheta_stag, Er_stag, coarsen, i, j, 0, 0);
                            const Real bz_val = Interp(Bz, Bz_stag, Er_stag, coarsen, i, j, 0, 0);
                            btot_val = std::sqrt(br_val*br_val + bt_val*bt_val + bz_val*bz_val);
                        }

                        // r on cell-centered point (Jr is cell-centered in r)
                        const Real r = rmin + (i + 0.5_rt)*dr;
                        auto nabla2Jr = T_Algo::Dr_rDr_over_r(Jr, r, dr, coefs_r, n_coefs_r, i, j, 0, 0)
                            + T_Algo::Dzz(Jr, coefs_z, n_coefs_z, i, j, 0, 0) - Jr(i, j, 0)/(r*r);

                        Er(i, j, 0) -= eta_h(rho_val_eta, btot_val) * nabla2Jr;
                    }
                }

                if (include_external_fields && (rho_val >= rho_floor)) {
                    Er(i, j, 0) -= Er_ext(i, j, 0);
                }
            },

            // Etheta calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Etheta_arr && update_Etheta_arr(i, j, 0) == 0) { return; }

                // r on a nodal grid (Etheta is nodal in r)
                Real const r = rmin + i*dr;
                // Mode m=0: // Ensure that Etheta remains 0 on axis
                if (r < 0.5_rt*dr) {
                    Etheta(i, j, 0, 0) = 0.;
                    return;
                }

                // Interpolate to get the appropriate charge density in space
                const Real rho_val = Interp(rho, nodal, Etheta_stag, coarsen, i, j, 0, 0);

                if (rho_val < rho_floor && holmstrom_vacuum_region) {
                    Etheta(i, j, 0) = 0._rt;
                } else {
                    // Get the gradient of the electron pressure
                    // -> d/dt = 0 for m = 0
                    const auto grad_Pe = 0.0_rt;

                    // interpolate the nodal neE values to the Yee grid
                    const auto enE_t = Interp(enE, nodal, Etheta_stag, coarsen, i, j, 0, 1);

                    // safety condition since we divide by rho
                    const auto rho_val_limited = std::max(rho_val, rho_floor);

                    Etheta(i, j, 0) = (enE_t - grad_Pe) / rho_val_limited;
                }

                // Add resistivity only if E field value is used to update B
                if (solve_for_Faraday) {
                    // the embedded-boundary Dirichlet mirror of rho is
                    // negative inside the conductor: keep the resistivity
                    // parsers on their physical domain
                    // |rho|: eta on the covered/mirror side uses the reflected
                    // plasma density and is never driven negative (see 3D notes).
                    const Real rho_val_eta = std::abs(rho_val);
                    Real jtot_val = 0._rt;
                    if(resistivity_has_J_dependence) {
                        // Interpolate current to appropriate staggering to match E field
                        const Real jr_val = Interp(Jr, Jr_stag, Etheta_stag, coarsen, i, j, 0, 0);
                        const Real jtheta_val = Jtheta(i, j, 0);
                        const Real jz_val = Interp(Jz, Jz_stag, Etheta_stag, coarsen, i, j, 0, 0);
                        jtot_val = std::sqrt(jr_val*jr_val + jtheta_val*jtheta_val + jz_val*jz_val);
                    }

                    Etheta(i, j, 0) += eta(rho_val_eta, jtot_val, t_new) * Jtheta(i, j, 0);

                    if (include_hyper_resistivity_term) {

                        // Interpolate B field to appropriate staggering to match E field
                        Real btot_val = 0._rt;
                        if (hyper_resistivity_has_B_dependence) {
                            const Real br_val = Interp(Br, Br_stag, Etheta_stag, coarsen, i, j, 0, 0);
                            const Real bt_val = Interp(Btheta, Btheta_stag, Etheta_stag, coarsen, i, j, 0, 0);
                            const Real bz_val = Interp(Bz, Bz_stag, Etheta_stag, coarsen, i, j, 0, 0);
                            btot_val = std::sqrt(br_val*br_val + bt_val*bt_val + bz_val*bz_val);
                        }

                        // Special handling of the hyper-resistivity term on axis to avoid division by zero
                        // and ensure that Etheta remains 0 on axis for m=0 mode
                        auto nabla2Jtheta = 0.0_rt;
                        if (r > 0.0_rt) {
                            nabla2Jtheta = T_Algo::Dr_rDr_over_r(Jtheta, r, dr, coefs_r, n_coefs_r, i, j, 0, 0)
                                + T_Algo::Dzz(Jtheta, coefs_z, n_coefs_z, i, j, 0, 0) - Jtheta(i, j, 0)/(r*r);
                        }

                        Etheta(i, j, 0) -= eta_h(rho_val_eta, btot_val) * nabla2Jtheta;
                    }
                }

                if (include_external_fields && (rho_val >= rho_floor)) {
                    Etheta(i, j, 0) -= Etheta_ext(i, j, 0);
                }
            },

            // Ez calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Ez_arr && update_Ez_arr(i, j, 0) == 0) { return; }

                // Interpolate to get the appropriate charge density in space
                const Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, 0, 0);

                if (rho_val < rho_floor && holmstrom_vacuum_region) {
                    Ez(i, j, 0) = 0._rt;
                } else {
                    // Get the gradient of the electron pressure if the longitudinal part of
                    // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                    const Real grad_Pe = (!solve_for_Faraday) ?
                        T_Algo::UpwardDz(Pe, coefs_z, n_coefs_z, i, j, 0, 0)
                        : 0._rt;

                    // interpolate the nodal neE values to the Yee grid
                    const auto enE_z = Interp(enE, nodal, Ez_stag, coarsen, i, j, 0, 2);

                    // safety condition since we divide by rho
                    const auto rho_val_limited = std::max(rho_val, rho_floor);

                    Ez(i, j, 0) = (enE_z - grad_Pe) / rho_val_limited;
                }

                // Add resistivity only if E field value is used to update B
                if (solve_for_Faraday) {
                    // the embedded-boundary Dirichlet mirror of rho is
                    // negative inside the conductor: keep the resistivity
                    // parsers on their physical domain
                    // |rho|: eta on the covered/mirror side uses the reflected
                    // plasma density and is never driven negative (see 3D notes).
                    const Real rho_val_eta = std::abs(rho_val);
                    Real jtot_val = 0._rt;
                    if (resistivity_has_J_dependence) {
                        // Interpolate current to appropriate staggering to match E field
                        const Real jr_val = Interp(Jr, Jr_stag, Ez_stag, coarsen, i, j, 0, 0);
                        const Real jtheta_val = Interp(Jtheta, Jtheta_stag, Ez_stag, coarsen, i, j, 0, 0);
                        const Real jz_val = Jz(i, j, 0);
                        jtot_val = std::sqrt(jr_val*jr_val + jtheta_val*jtheta_val + jz_val*jz_val);
                    }

                    Ez(i, j, 0) += eta(rho_val_eta, jtot_val, t_new) * Jz(i, j, 0);

                    if (include_hyper_resistivity_term) {

                        // Interpolate B field to appropriate staggering to match E field
                        Real btot_val = 0._rt;
                        if (hyper_resistivity_has_B_dependence) {
                            const Real br_val = Interp(Br, Br_stag, Ez_stag, coarsen, i, j, 0, 0);
                            const Real bt_val = Interp(Btheta, Btheta_stag, Ez_stag, coarsen, i, j, 0, 0);
                            const Real bz_val = Interp(Bz, Bz_stag, Ez_stag, coarsen, i, j, 0, 0);
                            btot_val = std::sqrt(br_val*br_val + bt_val*bt_val + bz_val*bz_val);
                        }

                        // r on nodal point (Jz is nodal in r)
                        const Real r = rmin + i*dr;

                        auto nabla2Jz = T_Algo::Dzz(Jz, coefs_z, n_coefs_z, i, j, 0, 0);
                        if (r > 0.5_rt*dr) {
                            nabla2Jz += T_Algo::Dr_rDr_over_r(Jz, r, dr, coefs_r, n_coefs_r, i, j, 0, 0);
                        } else {
                            // On axis the geometric term does not cancel: by
                            // L'Hopital, (1/r) d/dr(r dJz/dr) -> 2 d2Jz/dr2
                            // as r -> 0 (dJz/dr vanishes on axis for m=0), so
                            // the radial part is twice the second derivative.
                            // The even axis parity Jz(-dr) = Jz(dr) gives the
                            // ghost-free form 2 * 2*(Jz(dr) - Jz(0))/dr^2.
                            const Real inv_dr2 = coefs_r[0]*coefs_r[0];
                            nabla2Jz += 4._rt*(Jz(i+1, j, 0) - Jz(i, j, 0))*inv_dr2;
                        }

                        Ez(i, j, 0) -= eta_h(rho_val_eta, btot_val) * nabla2Jz;
                    }
                }

                if (include_external_fields && (rho_val >= rho_floor)) {
                    Ez(i, j, 0) -= Ez_ext(i, j, 0);
                }
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

#elif defined(WARPX_DIM_RSPHERE)
template<typename T_Algo>
void FiniteDifferenceSolver::HybridPICSolveESpherical (
    ablastr::fields::VectorField const& /*Efield*/,
    ablastr::fields::VectorField const& /*Jfield*/,
    ablastr::fields::VectorField const& /*Jifield*/,
    ablastr::fields::VectorField const& /*Bfield*/,
    amrex::MultiFab const& /*rhofield*/,
    amrex::MultiFab const& /*Pefield*/,
    int /*lev*/, HybridPICModel const* /*hybrid_model*/,
    const bool /*solve_for_Faraday*/ )
{
    WARPX_ABORT_WITH_MESSAGE("HybridPICSolveESphrical not fully implemented");
}
#else

template<typename T_Algo>
void FiniteDifferenceSolver::HybridPICSolveECartesian (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev, HybridPICModel const* hybrid_model,
    const bool solve_for_Faraday )
{
    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    using namespace ablastr::coarsen::sample;

    // get hybrid model parameters
    const auto eta = hybrid_model->m_eta;
    const auto eta_h = hybrid_model->m_eta_h;
    const auto rho_floor = hybrid_model->m_n_floor * PhysConst::q_e;
    const auto resistivity_has_J_dependence = hybrid_model->m_resistivity_has_J_dependence;
    const auto hyper_resistivity_has_B_dependence = hybrid_model->m_hyper_resistivity_has_B_dependence;
    const bool include_hyper_resistivity_term = hybrid_model->m_include_hyper_resistivity_term;

    const bool include_external_fields = hybrid_model->m_add_external_fields;

    const bool holmstrom_vacuum_region = hybrid_model->m_holmstrom_vacuum_region;

    // GOL masking (Lever 2): in partially-covered EB cells (eb_update_E flag == 2)
    // drop the stiff 1/n Hall + electron-pressure terms and the hyper-resistivity /
    // corner-curl corrections, leaving the well-posed resistive E = eta*J.
    const bool eb_resistive_only_partial = hybrid_model->m_eb_resistive_only_partial;

    // isotropized hyper-resistivity Laplacian (Mehrstellen / Patra-Karttunen).
    // LaplacianIsotropic is a centered compact stencil keyed off the field's own
    // neighbours, so it is correct for both the Yee (edge-centered) and the
    // collocated (nodal) staggerings with no change.
    const bool iso_hyper = hybrid_model->m_isotropic_hyper_resistivity;
    // isotropized resistive diffusion (corner-curl E correction). Two stencils:
    // the Yee corner (one-sided Faraday UpwardD) and the nodal corner (wide centered
    // Faraday UpwardD); the call sites below select via T_Algo. Both isotropize the
    // in-plane resistive diffusion of the out-of-plane B and preserve div(B) exactly.
    const bool iso_resistivity = hybrid_model->m_isotropic_resistivity;
    const amrex::Real inv_mu0 = 1._rt/PhysConst::mu0;
    amrex::GpuArray<amrex::Real, 3> dx_arr{};
    amrex::GpuArray<amrex::Real, 3> h2{};
    amrex::GpuArray<amrex::Real, 3> inv_h2{};
    {
        const auto dx_lev = WarpX::GetInstance().Geom(lev).CellSizeArray();
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            dx_arr[d] = dx_lev[d];
            h2[d] = dx_lev[d]*dx_lev[d];
            inv_h2[d] = 1._rt/h2[d];
        }
    }

    auto & warpx = WarpX::GetInstance();
    const amrex::Real t_new = warpx.gett_new(lev);
    ablastr::fields::VectorField Bfield_external, Efield_external;
    if (include_external_fields) {
        Bfield_external = warpx.m_fields.get_alldirs(FieldType::hybrid_B_fp_external, 0); // lev=0
        Efield_external = warpx.m_fields.get_alldirs(FieldType::hybrid_E_fp_external, 0); // lev=0
    }

    // Index type required for interpolating fields from their respective
    // staggering to the Ex, Ey, Ez locations
    amrex::GpuArray<int, 3> const& Ex_stag = hybrid_model->Ex_IndexType;
    amrex::GpuArray<int, 3> const& Ey_stag = hybrid_model->Ey_IndexType;
    amrex::GpuArray<int, 3> const& Ez_stag = hybrid_model->Ez_IndexType;
    amrex::GpuArray<int, 3> const& Jx_stag = hybrid_model->Jx_IndexType;
    amrex::GpuArray<int, 3> const& Jy_stag = hybrid_model->Jy_IndexType;
    amrex::GpuArray<int, 3> const& Jz_stag = hybrid_model->Jz_IndexType;
    amrex::GpuArray<int, 3> const& Bx_stag = hybrid_model->Bx_IndexType;
    amrex::GpuArray<int, 3> const& By_stag = hybrid_model->By_IndexType;
    amrex::GpuArray<int, 3> const& Bz_stag = hybrid_model->Bz_IndexType;

    // Parameters for `interp` that maps from Yee to nodal mesh and back
    amrex::GpuArray<int, 3> const& nodal = {1, 1, 1};
    // The "coarsening is just 1 i.e. no coarsening"
    amrex::GpuArray<int, 3> const& coarsen = {1, 1, 1};

    // The E-field calculation is done in 2 steps:
    // 1) The J x B term is calculated on a nodal mesh in order to ensure
    //    energy conservation.
    // 2) The nodal E-field values are averaged onto the Yee grid and the
    //    electron pressure & resistivity terms are added (these terms are
    //    naturally located on the Yee grid).

    // Create a temporary multifab to hold the nodal E-field values
    // Note the multifab has 3 values for Ex, Ey and Ez which we can do here
    // since all three components will be calculated on the same grid.
    // Also note that enE_nodal_mf does not need to have any guard cells since
    // these values will be interpolated to the Yee mesh which is contained
    // by the nodal mesh.
    auto const& ba = convert(rhofield.boxArray(), IntVect::TheNodeVector());
    MultiFab enE_nodal_mf(ba, rhofield.DistributionMap(), 3, IntVect::TheZeroVector());

    // Loop through the grids, and over the tiles within each grid for the
    // initial, nodal calculation of E
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(enE_nodal_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        Array4<Real> const& enE_nodal = enE_nodal_mf.array(mfi);
        Array4<Real const> const& Jx = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jy = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& Jix = Jifield[0]->const_array(mfi);
        Array4<Real const> const& Jiy = Jifield[1]->const_array(mfi);
        Array4<Real const> const& Jiz = Jifield[2]->const_array(mfi);
        Array4<Real const> const& Bx = Bfield[0]->const_array(mfi);
        Array4<Real const> const& By = Bfield[1]->const_array(mfi);
        Array4<Real const> const& Bz = Bfield[2]->const_array(mfi);

        Array4<Real> Bx_ext, By_ext, Bz_ext;
        if (include_external_fields) {
            Bx_ext = Bfield_external[0]->array(mfi);
            By_ext = Bfield_external[1]->array(mfi);
            Bz_ext = Bfield_external[2]->array(mfi);
        }

        // EB-aware nodal interpolation of the Hall J x B: exclude covered edges/faces
        // (update mask == 0) so covered-cell J/B values do not pollute the near-wall
        // nodal J x B (see InterpMasked). eb_update_E masks the currents (edge), the
        // B-update flags mask the self magnetic field (face).
        const bool hall_eb = EB::enabled() && hybrid_model->m_eb_hall_mask;
        amrex::Array4<int const> jxm, jym, jzm, bxm, bym, bzm;
        if (hall_eb) {
            jxm = eb_update_E[0]->const_array(mfi);
            jym = eb_update_E[1]->const_array(mfi);
            jzm = eb_update_E[2]->const_array(mfi);
            auto const& eb_update_B = WarpX::GetInstance().GetEBUpdateBFlag()[lev];
            bxm = eb_update_B[0]->const_array(mfi);
            bym = eb_update_B[1]->const_array(mfi);
            bzm = eb_update_B[2]->const_array(mfi);
        }

        // Loop over the cells and update the nodal E field
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k){

            // interpolate the total plasma current to a nodal grid
            auto const jx_interp = hall_eb ? InterpMasked(Jx, jxm, Jx_stag, nodal, i, j, k, 0)
                                           : Interp(Jx, Jx_stag, nodal, coarsen, i, j, k, 0);
            auto const jy_interp = hall_eb ? InterpMasked(Jy, jym, Jy_stag, nodal, i, j, k, 0)
                                           : Interp(Jy, Jy_stag, nodal, coarsen, i, j, k, 0);
            auto const jz_interp = hall_eb ? InterpMasked(Jz, jzm, Jz_stag, nodal, i, j, k, 0)
                                           : Interp(Jz, Jz_stag, nodal, coarsen, i, j, k, 0);

            // interpolate the ion current to a nodal grid
            auto const jix_interp = hall_eb ? InterpMasked(Jix, jxm, Jx_stag, nodal, i, j, k, 0)
                                            : Interp(Jix, Jx_stag, nodal, coarsen, i, j, k, 0);
            auto const jiy_interp = hall_eb ? InterpMasked(Jiy, jym, Jy_stag, nodal, i, j, k, 0)
                                            : Interp(Jiy, Jy_stag, nodal, coarsen, i, j, k, 0);
            auto const jiz_interp = hall_eb ? InterpMasked(Jiz, jzm, Jz_stag, nodal, i, j, k, 0)
                                            : Interp(Jiz, Jz_stag, nodal, coarsen, i, j, k, 0);

            // interpolate the (self) B field to a nodal grid
            auto Bx_interp = hall_eb ? InterpMasked(Bx, bxm, Bx_stag, nodal, i, j, k, 0)
                                     : Interp(Bx, Bx_stag, nodal, coarsen, i, j, k, 0);
            auto By_interp = hall_eb ? InterpMasked(By, bym, By_stag, nodal, i, j, k, 0)
                                     : Interp(By, By_stag, nodal, coarsen, i, j, k, 0);
            auto Bz_interp = hall_eb ? InterpMasked(Bz, bzm, Bz_stag, nodal, i, j, k, 0)
                                     : Interp(Bz, Bz_stag, nodal, coarsen, i, j, k, 0);

            if (include_external_fields) {
                Bx_interp += Interp(Bx_ext, Bx_stag, nodal, coarsen, i, j, k, 0);
                By_interp += Interp(By_ext, By_stag, nodal, coarsen, i, j, k, 0);
                Bz_interp += Interp(Bz_ext, Bz_stag, nodal, coarsen, i, j, k, 0);
            }

            // calculate enE = (J - Ji) x B
            enE_nodal(i, j, k, 0) = (
                (jy_interp - jiy_interp) * Bz_interp
                - (jz_interp - jiz_interp) * By_interp
            );
            enE_nodal(i, j, k, 1) = (
                (jz_interp - jiz_interp) * Bx_interp
                - (jx_interp - jix_interp) * Bz_interp
            );
            enE_nodal(i, j, k, 2) = (
                (jx_interp - jix_interp) * By_interp
                - (jy_interp - jiy_interp) * Bx_interp
            );
        });

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    // Loop through the grids, and over the tiles within each grid again
    // for the Yee grid calculation of the E field
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const& Ex = Efield[0]->array(mfi);
        Array4<Real> const& Ey = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real const> const& Jx = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jy = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& enE = enE_nodal_mf.const_array(mfi);
        Array4<Real const> const& rho = rhofield.const_array(mfi);
        Array4<Real const> const& Pe = Pefield.array(mfi);
        Array4<Real> const& Bx = Bfield[0]->array(mfi);
        Array4<Real> const& By = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries
        amrex::Array4<int> update_Ex_arr, update_Ey_arr, update_Ez_arr;
        if (EB::enabled()) {
            update_Ex_arr = eb_update_E[0]->array(mfi);
            update_Ey_arr = eb_update_E[1]->array(mfi);
            update_Ez_arr = eb_update_E[2]->array(mfi);
        }

        Array4<Real> Ex_ext, Ey_ext, Ez_ext;
        if (include_external_fields) {
            Ex_ext = Efield_external[0]->array(mfi);
            Ey_ext = Efield_external[1]->array(mfi);
            Ez_ext = Efield_external[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        Box const& tex  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tey  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Loop over the cells and update the E field
        // Ex calculation
        amrex::ParallelFor(tex, [=] AMREX_GPU_DEVICE (int i, int j, int k){

            // Skip fully-covered edges (flag 0); flag 2 = partially covered ->
            // resistive-only (GOL mask); flag 1 = regular.
            const int ebf_x = update_Ex_arr ? update_Ex_arr(i, j, k) : 1;
            if (ebf_x == 0) { return; }
            const bool partial_resistive_only = eb_resistive_only_partial
                && (ebf_x == 2
                    || warpx::hybrid_isotropic::StencilTouchesNonregular(update_Ex_arr, i, j, k));

            // Interpolate to get the appropriate charge density in space
            const Real rho_val = Interp(rho, nodal, Ex_stag, coarsen, i, j, k, 0);

            // Drop the stiff 1/n Hall + electron-pressure terms in partial cells
            // (E = eta*J there) or in the holmstrom vacuum region below the floor.
            const bool drop_hall_pressure =
                partial_resistive_only || (rho_val < rho_floor && holmstrom_vacuum_region);
            if (drop_hall_pressure) {
                Ex(i, j, k) = 0._rt;
            } else {
                // Get the gradient of the electron pressure if the longitudinal part of
                // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                const Real grad_Pe = (!solve_for_Faraday) ?
                    T_Algo::UpwardDx(Pe, coefs_x, n_coefs_x, i, j, k)
                    : 0._rt;

                // interpolate the nodal neE values to the Yee grid
                const auto enE_x = Interp(enE, nodal, Ex_stag, coarsen, i, j, k, 0);

                // safety condition since we divide by rho
                const auto rho_val_limited = std::max(rho_val, rho_floor);

                Ex(i, j, k) = (enE_x - grad_Pe) / rho_val_limited;
            }

            // Add resistivity only if E field value is used to update B
            if (solve_for_Faraday) {
                // The EB Dirichlet mirror of rho is negative inside the conductor.
                // Feed |rho| (not max(rho,0)) to the resistivity parser so eta is
                // evaluated at the reflected PLASMA density on the covered side --
                // the physically correct eta for E = eta*J in the mirror region --
                // and eta is never driven negative (which would invert the resistive
                // term and pump energy into the wall).
                const Real rho_val_eta = std::abs(rho_val);
                Real jtot_val = 0._rt;
                if (resistivity_has_J_dependence) {
                    // Interpolate current to appropriate staggering to match E field
                    const Real jx_val = Jx(i, j, k);
                    const Real jy_val = Interp(Jy, Jy_stag, Ex_stag, coarsen, i, j, k, 0);
                    const Real jz_val = Interp(Jz, Jz_stag, Ex_stag, coarsen, i, j, k, 0);
                    jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);
                }

                // Evaluate the resistivity parser once: the same eta(rho,jtot,t)
                // is reused below by the corner-curl (iso_resistivity) term.
                const amrex::Real eta_val = eta(rho_val_eta, jtot_val, t_new);

                Ex(i, j, k) += eta_val * Jx(i, j, k);

                if (include_hyper_resistivity_term && !partial_resistive_only) {

                    // Interpolate B field to appropriate staggering to match E field
                    Real btot_val = 0._rt;
                    if (hyper_resistivity_has_B_dependence) {
                        const Real bx_val = Interp(Bx, Bx_stag, Ex_stag, coarsen, i, j, k, 0);
                        const Real by_val = Interp(By, By_stag, Ex_stag, coarsen, i, j, k, 0);
                        const Real bz_val = Interp(Bz, Bz_stag, Ex_stag, coarsen, i, j, k, 0);
                        btot_val = std::sqrt(bx_val*bx_val + by_val*by_val + bz_val*bz_val);
                    }

                    const Real nabla2Jx = iso_hyper
                        ? warpx::hybrid_isotropic::LaplacianIsotropic(Jx, i, j, k, h2, inv_h2)
                        : T_Algo::Dxx(Jx, coefs_x, n_coefs_x, i, j, k)
                          + T_Algo::Dyy(Jx, coefs_y, n_coefs_y, i, j, k)
                          + T_Algo::Dzz(Jx, coefs_z, n_coefs_z, i, j, k);

                    Ex(i, j, k) -= eta_h(rho_val_eta, btot_val) * nabla2Jx;
                }

                // Isotropize the in-plane resistive diffusion of the
                // out-of-plane B (Bz in 3D, By in 2D XZ) via the corner-curl
                // correction; div(B) preserved (added through E).
                if (iso_resistivity && !partial_resistive_only) {
                    // Runtime if (not if constexpr): nvcc forbids an extended
                    // __device__ lambda from first-capturing a variable inside a
                    // constexpr-if. The condition is still compile-time constant,
                    // so the dead branch is pruned.
                    const bool nodal_grid =
                        std::is_same_v<T_Algo, CartesianNodalAlgorithm>;
#if defined(WARPX_DIM_3D)
                    if (nodal_grid) {
                        Ex(i, j, k) += eta_val
                            * warpx::hybrid_isotropic::CornerResistiveEx_3D_Nodal(Bz, i, j, k, dx_arr, inv_mu0);
                    } else {
                        Ex(i, j, k) += eta_val
                            * warpx::hybrid_isotropic::CornerResistiveEx_3D(Bz, i, j, k, h2, inv_h2, dx_arr, inv_mu0);
                    }
#elif defined(WARPX_DIM_XZ)
                    if (nodal_grid) {
                        Ex(i, j, k) += eta_val
                            * warpx::hybrid_isotropic::CornerResistiveEx_XZ_Nodal(By, i, j, k, dx_arr, inv_mu0);
                    } else {
                        Ex(i, j, k) += eta_val
                            * warpx::hybrid_isotropic::CornerResistiveEx_XZ(By, i, j, k, h2, inv_h2, dx_arr, inv_mu0);
                    }
#endif
                }
            }

            if (include_external_fields && (rho_val >= rho_floor)) {
                Ex(i, j, k) -= Ex_ext(i, j, k);
            }
        });

        // Ey calculation
        amrex::ParallelFor(tey, [=] AMREX_GPU_DEVICE (int i, int j, int k) {

            // Skip fully-covered edges (flag 0); flag 2 = partially covered ->
            // resistive-only (GOL mask); flag 1 = regular.
            const int ebf_y = update_Ey_arr ? update_Ey_arr(i, j, k) : 1;
            if (ebf_y == 0) { return; }
            const bool partial_resistive_only = eb_resistive_only_partial
                && (ebf_y == 2
                    || warpx::hybrid_isotropic::StencilTouchesNonregular(update_Ey_arr, i, j, k));

            // Interpolate to get the appropriate charge density in space
            const Real rho_val = Interp(rho, nodal, Ey_stag, coarsen, i, j, k, 0);

            // Drop the stiff 1/n Hall + electron-pressure terms in partial cells
            // (E = eta*J there) or in the holmstrom vacuum region below the floor.
            const bool drop_hall_pressure =
                partial_resistive_only || (rho_val < rho_floor && holmstrom_vacuum_region);
            if (drop_hall_pressure) {
                Ey(i, j, k) = 0._rt;
            } else {
                // Get the gradient of the electron pressure if the longitudinal part of
                // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                const Real grad_Pe = (!solve_for_Faraday) ?
                    T_Algo::UpwardDy(Pe, coefs_y, n_coefs_y, i, j, k)
                    : 0._rt;

                // interpolate the nodal neE values to the Yee grid
                const auto enE_y = Interp(enE, nodal, Ey_stag, coarsen, i, j, k, 1);

                // safety condition since we divide by rho
                const auto rho_val_limited = std::max(rho_val, rho_floor);

                Ey(i, j, k) = (enE_y - grad_Pe) / rho_val_limited;
            }

            // Add resistivity only if E field value is used to update B
            if (solve_for_Faraday) {
                // The EB Dirichlet mirror of rho is negative inside the conductor.
                // Feed |rho| (not max(rho,0)) to the resistivity parser so eta is
                // evaluated at the reflected PLASMA density on the covered side --
                // the physically correct eta for E = eta*J in the mirror region --
                // and eta is never driven negative (which would invert the resistive
                // term and pump energy into the wall).
                const Real rho_val_eta = std::abs(rho_val);
                Real jtot_val = 0._rt;
                if (resistivity_has_J_dependence) {
                    // Interpolate current to appropriate staggering to match E field
                    const Real jx_val = Interp(Jx, Jx_stag, Ey_stag, coarsen, i, j, k, 0);
                    const Real jy_val = Jy(i, j, k);
                    const Real jz_val = Interp(Jz, Jz_stag, Ey_stag, coarsen, i, j, k, 0);
                    jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);
                }

                // Evaluate the resistivity parser once: the same eta(rho,jtot,t)
                // is reused below by the corner-curl (iso_resistivity) term.
                const amrex::Real eta_val = eta(rho_val_eta, jtot_val, t_new);

                Ey(i, j, k) += eta_val * Jy(i, j, k);

                if (include_hyper_resistivity_term && !partial_resistive_only) {

                    // Interpolate B field to appropriate staggering to match E field
                    Real btot_val = 0._rt;
                    if (hyper_resistivity_has_B_dependence) {
                        const Real bx_val = Interp(Bx, Bx_stag, Ey_stag, coarsen, i, j, k, 0);
                        const Real by_val = Interp(By, By_stag, Ey_stag, coarsen, i, j, k, 0);
                        const Real bz_val = Interp(Bz, Bz_stag, Ey_stag, coarsen, i, j, k, 0);
                        btot_val = std::sqrt(bx_val*bx_val + by_val*by_val + bz_val*bz_val);
                    }

                    const Real nabla2Jy = iso_hyper
                        ? warpx::hybrid_isotropic::LaplacianIsotropic(Jy, i, j, k, h2, inv_h2)
                        : T_Algo::Dxx(Jy, coefs_x, n_coefs_x, i, j, k)
                          + T_Algo::Dyy(Jy, coefs_y, n_coefs_y, i, j, k)
                          + T_Algo::Dzz(Jy, coefs_z, n_coefs_z, i, j, k);

                    Ey(i, j, k) -= eta_h(rho_val_eta, btot_val) * nabla2Jy;
                }

                // Bz corner-curl correction, second (Ey) half (3D only; in
                // 2D XZ the out-of-plane corner is delivered via Ex and Ez).
#if defined(WARPX_DIM_3D)
                if (iso_resistivity && !partial_resistive_only) {
                    if (std::is_same_v<T_Algo, CartesianNodalAlgorithm>) {
                        Ey(i, j, k) += eta_val
                            * warpx::hybrid_isotropic::CornerResistiveEy_3D_Nodal(Bz, i, j, k, dx_arr, inv_mu0);
                    } else {
                        Ey(i, j, k) += eta_val
                            * warpx::hybrid_isotropic::CornerResistiveEy_3D(Bz, i, j, k, h2, inv_h2, dx_arr, inv_mu0);
                    }
                }
#endif
            }

            if (include_external_fields && (rho_val >= rho_floor)) {
                Ey(i, j, k) -= Ey_ext(i, j, k);
            }
        });

        // Ez calculation
        amrex::ParallelFor(tez, [=] AMREX_GPU_DEVICE (int i, int j, int k){

            // Skip fully-covered edges (flag 0); flag 2 = partially covered ->
            // resistive-only (GOL mask); flag 1 = regular.
            const int ebf_z = update_Ez_arr ? update_Ez_arr(i, j, k) : 1;
            if (ebf_z == 0) { return; }
            const bool partial_resistive_only = eb_resistive_only_partial
                && (ebf_z == 2
                    || warpx::hybrid_isotropic::StencilTouchesNonregular(update_Ez_arr, i, j, k));

            // Interpolate to get the appropriate charge density in space
            const Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, k, 0);

            // Drop the stiff 1/n Hall + electron-pressure terms in partial cells
            // (E = eta*J there) or in the holmstrom vacuum region below the floor.
            const bool drop_hall_pressure =
                partial_resistive_only || (rho_val < rho_floor && holmstrom_vacuum_region);
            if (drop_hall_pressure) {
                Ez(i, j, k) = 0._rt;
            } else {
                // Get the gradient of the electron pressure if the longitudinal part of
                // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                const Real grad_Pe = (!solve_for_Faraday) ?
                    T_Algo::UpwardDz(Pe, coefs_z, n_coefs_z, i, j, k)
                    : 0._rt;

                // interpolate the nodal neE values to the Yee grid
                const auto enE_z = Interp(enE, nodal, Ez_stag, coarsen, i, j, k, 2);

                // safety condition since we divide by rho
                const auto rho_val_limited = std::max(rho_val, rho_floor);

                Ez(i, j, k) = (enE_z - grad_Pe) / rho_val_limited;
            }

            // Add resistivity only if E field value is used to update B
            if (solve_for_Faraday) {
                // The EB Dirichlet mirror of rho is negative inside the conductor.
                // Feed |rho| (not max(rho,0)) to the resistivity parser so eta is
                // evaluated at the reflected PLASMA density on the covered side --
                // the physically correct eta for E = eta*J in the mirror region --
                // and eta is never driven negative (which would invert the resistive
                // term and pump energy into the wall).
                const Real rho_val_eta = std::abs(rho_val);
                Real jtot_val = 0._rt;
                if (resistivity_has_J_dependence) {
                    // Interpolate current to appropriate staggering to match E field
                    const Real jx_val = Interp(Jx, Jx_stag, Ez_stag, coarsen, i, j, k, 0);
                    const Real jy_val = Interp(Jy, Jy_stag, Ez_stag, coarsen, i, j, k, 0);
                    const Real jz_val = Jz(i, j, k);
                    jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);
                }

                // Evaluate the resistivity parser once: the same eta(rho,jtot,t)
                // is reused below by the corner-curl (iso_resistivity) term.
                const amrex::Real eta_val = eta(rho_val_eta, jtot_val, t_new);

                Ez(i, j, k) += eta_val * Jz(i, j, k);

                if (include_hyper_resistivity_term && !partial_resistive_only) {

                    // Interpolate B field to appropriate staggering to match E field
                    Real btot_val = 0._rt;
                    if (hyper_resistivity_has_B_dependence) {
                        const Real bx_val = Interp(Bx, Bx_stag, Ez_stag, coarsen, i, j, k, 0);
                        const Real by_val = Interp(By, By_stag, Ez_stag, coarsen, i, j, k, 0);
                        const Real bz_val = Interp(Bz, Bz_stag, Ez_stag, coarsen, i, j, k, 0);
                        btot_val = std::sqrt(bx_val*bx_val + by_val*by_val + bz_val*bz_val);
                    }

                    const Real nabla2Jz = iso_hyper
                        ? warpx::hybrid_isotropic::LaplacianIsotropic(Jz, i, j, k, h2, inv_h2)
                        : T_Algo::Dxx(Jz, coefs_x, n_coefs_x, i, j, k)
                          + T_Algo::Dyy(Jz, coefs_y, n_coefs_y, i, j, k)
                          + T_Algo::Dzz(Jz, coefs_z, n_coefs_z, i, j, k);

                    Ez(i, j, k) -= eta_h(rho_val_eta, btot_val) * nabla2Jz;
                }

                // By corner-curl correction, second (Ez) half (2D XZ only;
                // in 3D the axial Bz corner is delivered via Ex and Ey).
#if defined(WARPX_DIM_XZ)
                if (iso_resistivity && !partial_resistive_only) {
                    if (std::is_same_v<T_Algo, CartesianNodalAlgorithm>) {
                        Ez(i, j, k) += eta_val
                            * warpx::hybrid_isotropic::CornerResistiveEz_XZ_Nodal(By, i, j, k, dx_arr, inv_mu0);
                    } else {
                        Ez(i, j, k) += eta_val
                            * warpx::hybrid_isotropic::CornerResistiveEz_XZ(By, i, j, k, h2, inv_h2, dx_arr, inv_mu0);
                    }
                }
#endif
            }

            if (include_external_fields && (rho_val >= rho_floor)) {
                Ez(i, j, k) -= Ez_ext(i, j, k);
            }
        });

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}
#endif
