/* Copyright 2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FiniteDifferenceSolver.H"

#include "EmbeddedBoundary/Enabled.H"
#ifdef WARPX_DIM_RZ
#   include "FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#else
#   include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#endif
#include "HybridPICModel/HybridPICModel.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

using namespace amrex;

void FiniteDifferenceSolver::CalculateCurrentAmpere (
    ablastr::fields::VectorField & Jfield,
    ablastr::fields::VectorField const& Bfield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev )
{
    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {
#ifdef WARPX_DIM_RZ
        CalculateCurrentAmpereCylindrical <CylindricalYeeAlgorithm> (
            Jfield, Bfield, eb_update_E, lev
        );

#else
        CalculateCurrentAmpereCartesian <CartesianYeeAlgorithm> (
            Jfield, Bfield, eb_update_E, lev
        );

#endif
    } else {
        amrex::Abort(Utils::TextMsg::Err(
            "CalculateCurrentAmpere: Unknown algorithm choice."));
    }
}

// /**
//   * \brief Calculate total current from Ampere's law without displacement
//   * current i.e. J = 1/mu_0 curl x B.
//   *
//   * \param[out] Jfield  vector of total current MultiFabs at a given level
//   * \param[in] Bfield   vector of magnetic field MultiFabs at a given level
//   */
#ifdef WARPX_DIM_RZ
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

    // reset Jfield
    Jfield[0]->setVal(0);
    Jfield[1]->setVal(0);
    Jfield[2]->setVal(0);

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
        Array4<Real> const& Jt = Jfield[1]->array(mfi);
        Array4<Real> const& Jz = Jfield[2]->array(mfi);
        Array4<Real> const& Br = Bfield[0]->array(mfi);
        Array4<Real> const& Bt = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries.
        // The plasma current is stored at the same locations as the E-field,
        // therefore the `eb_update_E` multifab also appropriately specifies
        // where the plasma current should be calculated.
        amrex::Array4<int> update_Jr_arr, update_Jt_arr, update_Jz_arr;
        if (EB::enabled()) {
            update_Jr_arr = eb_update_E[0]->array(mfi);
            update_Jt_arr = eb_update_E[1]->array(mfi);
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

        // Extract tileboxes for which to loop
        Box const& tjr  = mfi.tilebox(Jfield[0]->ixType().toIntVect());
        Box const& tjt  = mfi.tilebox(Jfield[1]->ixType().toIntVect());
        Box const& tjz  = mfi.tilebox(Jfield[2]->ixType().toIntVect());

        Real const one_over_mu0 = 1._rt / PhysConst::mu0;

        // Calculate the total current, using Ampere's law, on the same grid
        // as the E-field
        amrex::ParallelFor(tjr, tjt, tjz,

            // Jr calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Jr_arr && update_Jr_arr(i, j, 0) == 0) { return; }

                // Mode m=0
                Jr(i, j, 0, 0) = one_over_mu0 * (
                    - T_Algo::DownwardDz(Bt, coefs_z, n_coefs_z, i, j, 0, 0)
                );

                // Higher-order modes
                // r on cell-centered point (Jr is cell-centered in r)
                Real const r = rmin + (i + 0.5_rt)*dr;
                for (int m=1; m<nmodes; m++) {
                    Jr(i, j, 0, 2*m-1) = one_over_mu0 * (
                        - T_Algo::DownwardDz(Bt, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                        + m * Bz(i, j, 0, 2*m  ) / r
                    );  // Real part
                    Jr(i, j, 0, 2*m  ) = one_over_mu0 * (
                        - T_Algo::DownwardDz(Bt, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                        - m * Bz(i, j, 0, 2*m-1) / r
                    ); // Imaginary part
                }
            },

            // Jt calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Jt_arr && update_Jt_arr(i, j, 0) == 0) { return; }

                // r on a nodal point (Jt is nodal in r)
                Real const r = rmin + i*dr;
                // Off-axis, regular curl
                if (r > 0.5_rt*dr) {
                    // Mode m=0
                    Jt(i, j, 0, 0) = one_over_mu0 * (
                        - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 0)
                        + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 0)
                    );

                    // Higher-order modes
                    for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                        Jt(i, j, 0, 2*m-1) = one_over_mu0 * (
                            - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                            + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                        ); // Real part
                        Jt(i, j, 0, 2*m  ) = one_over_mu0 * (
                            - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 2*m  )
                            + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                        ); // Imaginary part
                    }
                // r==0: on-axis corrections
                } else {
                    // Ensure that Jt remains 0 on axis (except for m=1)
                    // Mode m=0
                    Jt(i, j, 0, 0) = 0.;
                    // Higher-order modes
                    for (int m=1; m<nmodes; m++) {
                        if (m == 1){
                            // The same logic as is used in the E-field update for the fully
                            // electromagnetic FDTD case is used here.
                            Jt(i,j,0,2*m-1) =  Jr(i,j,0,2*m  );
                            Jt(i,j,0,2*m  ) = -Jr(i,j,0,2*m-1);
                        } else {
                            Jt(i, j, 0, 2*m-1) = 0.;
                            Jt(i, j, 0, 2*m  ) = 0.;
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
                       T_Algo::DownwardDrr_over_r(Bt, r, dr, coefs_r, n_coefs_r, i, j, 0, 0)
                    );
                    // Higher-order modes
                    for (int m=1 ; m<nmodes ; m++) {
                        Jz(i, j, 0, 2*m-1) = one_over_mu0 * (
                            - m * Br(i, j, 0, 2*m  ) / r
                            + T_Algo::DownwardDrr_over_r(Bt, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                        ); // Real part
                        Jz(i, j, 0, 2*m  ) = one_over_mu0 * (
                            m * Br(i, j, 0, 2*m-1) / r
                            + T_Algo::DownwardDrr_over_r(Bt, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m  )
                        ); // Imaginary part
                    }
                // r==0: on-axis corrections
                } else {
                    // For m==0, Bt is linear in r, for small r
                    // Therefore, the formula below regularizes the singularity
                    Jz(i, j, 0, 0) = one_over_mu0 * 4 * Bt(i, j, 0, 0) / dr;
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

    // reset Jfield
    Jfield[0]->setVal(0);
    Jfield[1]->setVal(0);
    Jfield[2]->setVal(0);

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

        // Extract tileboxes for which to loop
        Box const& tjx  = mfi.tilebox(Jfield[0]->ixType().toIntVect());
        Box const& tjy  = mfi.tilebox(Jfield[1]->ixType().toIntVect());
        Box const& tjz  = mfi.tilebox(Jfield[2]->ixType().toIntVect());

        Real const one_over_mu0 = 1._rt / PhysConst::mu0;

        // Calculate the total current, using Ampere's law, on the same grid
        // as the E-field
        amrex::ParallelFor(tjx, tjy, tjz,

            // Jx calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Skip field update in the embedded boundaries
                if (update_Jx_arr && update_Jx_arr(i, j, k) == 0) { return; }

                Jx(i, j, k) = one_over_mu0 * (
                    - T_Algo::DownwardDz(By, coefs_z, n_coefs_z, i, j, k)
                    + T_Algo::DownwardDy(Bz, coefs_y, n_coefs_y, i, j, k)
                );
            },

            // Jy calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Skip field update in the embedded boundaries
                if (update_Jy_arr && update_Jy_arr(i, j, k) == 0) { return; }

                Jy(i, j, k) = one_over_mu0 * (
                    - T_Algo::DownwardDx(Bz, coefs_x, n_coefs_x, i, j, k)
                    + T_Algo::DownwardDz(Bx, coefs_z, n_coefs_z, i, j, k)
                );
            },

            // Jz calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Skip field update in the embedded boundaries
                if (update_Jz_arr && update_Jz_arr(i, j, k) == 0) { return; }

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
#endif


void FiniteDifferenceSolver::HybridPICSolveE (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField& Jfield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev, HybridPICModel const* hybrid_model,
    const bool solve_for_Faraday)
{
    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {
#ifdef WARPX_DIM_RZ

        HybridPICSolveECylindrical <CylindricalYeeAlgorithm> (
            Efield, Jfield, Jifield, Bfield, rhofield, Pefield,
            eb_update_E, lev, hybrid_model, solve_for_Faraday
        );

#else

        HybridPICSolveECartesian <CartesianYeeAlgorithm> (
            Efield, Jfield, Jifield, Bfield, rhofield, Pefield,
            eb_update_E, lev, hybrid_model, solve_for_Faraday
        );

#endif
    } else {
        amrex::Abort(Utils::TextMsg::Err(
            "HybridSolveE: The hybrid-PIC electromagnetic solver algorithm must be used"));
    }
}

#ifdef WARPX_DIM_RZ
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

    const bool include_hyper_resistivity_term = (eta_h > 0.0) && solve_for_Faraday;

    // Index type required for interpolating fields from their respective
    // staggering to the Ex, Ey, Ez locations
    amrex::GpuArray<int, 3> const& Er_stag = hybrid_model->Ex_IndexType;
    amrex::GpuArray<int, 3> const& Et_stag = hybrid_model->Ey_IndexType;
    amrex::GpuArray<int, 3> const& Ez_stag = hybrid_model->Ez_IndexType;
    amrex::GpuArray<int, 3> const& Jr_stag = hybrid_model->Jx_IndexType;
    amrex::GpuArray<int, 3> const& Jt_stag = hybrid_model->Jy_IndexType;
    amrex::GpuArray<int, 3> const& Jz_stag = hybrid_model->Jz_IndexType;
    amrex::GpuArray<int, 3> const& Br_stag = hybrid_model->Bx_IndexType;
    amrex::GpuArray<int, 3> const& Bt_stag = hybrid_model->By_IndexType;
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
        Array4<Real const> const& Jt = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& Jir = Jifield[0]->const_array(mfi);
        Array4<Real const> const& Jit = Jifield[1]->const_array(mfi);
        Array4<Real const> const& Jiz = Jifield[2]->const_array(mfi);
        Array4<Real const> const& Br = Bfield[0]->const_array(mfi);
        Array4<Real const> const& Bt = Bfield[1]->const_array(mfi);
        Array4<Real const> const& Bz = Bfield[2]->const_array(mfi);

        // Loop over the cells and update the nodal E field
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

            // interpolate the total current to a nodal grid
            auto const jr_interp = Interp(Jr, Jr_stag, nodal, coarsen, i, j, 0, 0);
            auto const jt_interp = Interp(Jt, Jt_stag, nodal, coarsen, i, j, 0, 0);
            auto const jz_interp = Interp(Jz, Jz_stag, nodal, coarsen, i, j, 0, 0);

            // interpolate the ion current to a nodal grid
            auto const jir_interp = Interp(Jir, Jr_stag, nodal, coarsen, i, j, 0, 0);
            auto const jit_interp = Interp(Jit, Jt_stag, nodal, coarsen, i, j, 0, 0);
            auto const jiz_interp = Interp(Jiz, Jz_stag, nodal, coarsen, i, j, 0, 0);

            // interpolate the B field to a nodal grid
            auto const Br_interp = Interp(Br, Br_stag, nodal, coarsen, i, j, 0, 0);
            auto const Bt_interp = Interp(Bt, Bt_stag, nodal, coarsen, i, j, 0, 0);
            auto const Bz_interp = Interp(Bz, Bz_stag, nodal, coarsen, i, j, 0, 0);

            // calculate enE = (J - Ji) x B
            enE_nodal(i, j, 0, 0) = (
                (jt_interp - jit_interp) * Bz_interp
                - (jz_interp - jiz_interp) * Bt_interp
            );
            enE_nodal(i, j, 0, 1) = (
                (jz_interp - jiz_interp) * Br_interp
                - (jr_interp - jir_interp) * Bz_interp
            );
            enE_nodal(i, j, 0, 2) = (
                (jr_interp - jir_interp) * Bt_interp
                - (jt_interp - jit_interp) * Br_interp
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
        Array4<Real> const& Et = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real const> const& Jr = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jt = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& enE = enE_nodal_mf.const_array(mfi);
        Array4<Real const> const& rho = rhofield.const_array(mfi);
        Array4<Real const> const& Pe = Pefield.const_array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries
        amrex::Array4<int> update_Er_arr, update_Et_arr, update_Ez_arr;
        if (EB::enabled()) {
            update_Er_arr = eb_update_E[0]->array(mfi);
            update_Et_arr = eb_update_E[1]->array(mfi);
            update_Ez_arr = eb_update_E[2]->array(mfi);
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
                Real rho_val = Interp(rho, nodal, Er_stag, coarsen, i, j, 0, 0);

                // Interpolate current to appropriate staggering to match E field
                Real jtot_val = 0._rt;
                if (solve_for_Faraday && resistivity_has_J_dependence) {
                    const Real jr_val = Interp(Jr, Jr_stag, Er_stag, coarsen, i, j, 0, 0);
                    const Real jt_val = Interp(Jt, Jt_stag, Er_stag, coarsen, i, j, 0, 0);
                    const Real jz_val = Interp(Jz, Jz_stag, Er_stag, coarsen, i, j, 0, 0);
                    jtot_val = std::sqrt(jr_val*jr_val + jt_val*jt_val + jz_val*jz_val);
                }

                // safety condition since we divide by rho_val later
                if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Get the gradient of the electron pressure if the longitudinal part of
                // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                Real grad_Pe = 0._rt;
                if (!solve_for_Faraday) { grad_Pe = T_Algo::UpwardDr(Pe, coefs_r, n_coefs_r, i, j, 0, 0); }

                // interpolate the nodal neE values to the Yee grid
                auto enE_r = Interp(enE, nodal, Er_stag, coarsen, i, j, 0, 0);

                Er(i, j, 0) = (enE_r - grad_Pe) / rho_val;

                // Add resistivity only if E field value is used to update B
                if (solve_for_Faraday) { Er(i, j, 0) += eta(rho_val, jtot_val) * Jr(i, j, 0); }

                if (include_hyper_resistivity_term) {
                    // r on cell-centered point (Jr is cell-centered in r)
                    Real const r = rmin + (i + 0.5_rt)*dr;

                    auto nabla2Jr = T_Algo::Dr_rDr_over_r(Jr, r, dr, coefs_r, n_coefs_r, i, j, 0, 0);
                    Er(i, j, 0) -= eta_h * nabla2Jr;
                }
            },

            // Et calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Et_arr && update_Et_arr(i, j, 0) == 0) { return; }

                // r on a nodal grid (Et is nodal in r)
                Real const r = rmin + i*dr;
                // Mode m=0: // Ensure that Et remains 0 on axis
                if (r < 0.5_rt*dr) {
                    Et(i, j, 0, 0) = 0.;
                    return;
                }

                // Interpolate to get the appropriate charge density in space
                Real rho_val = Interp(rho, nodal, Er_stag, coarsen, i, j, 0, 0);

                // Interpolate current to appropriate staggering to match E field
                Real jtot_val = 0._rt;
                if (solve_for_Faraday && resistivity_has_J_dependence) {
                    const Real jr_val = Interp(Jr, Jr_stag, Et_stag, coarsen, i, j, 0, 0);
                    const Real jt_val = Interp(Jt, Jt_stag, Et_stag, coarsen, i, j, 0, 0);
                    const Real jz_val = Interp(Jz, Jz_stag, Et_stag, coarsen, i, j, 0, 0);
                    jtot_val = std::sqrt(jr_val*jr_val + jt_val*jt_val + jz_val*jz_val);
                }

                // safety condition since we divide by rho_val later
                if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Get the gradient of the electron pressure
                // -> d/dt = 0 for m = 0
                auto grad_Pe = 0.0_rt;

                // interpolate the nodal neE values to the Yee grid
                auto enE_t = Interp(enE, nodal, Et_stag, coarsen, i, j, 0, 1);

                Et(i, j, 0) = (enE_t - grad_Pe) / rho_val;

                // Add resistivity only if E field value is used to update B
                if (solve_for_Faraday) { Et(i, j, 0) += eta(rho_val, jtot_val) * Jt(i, j, 0); }

                // Note: Hyper-resisitivity should be revisited here when modal decomposition is implemented
            },

            // Ez calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Ez_arr && update_Ez_arr(i, j, 0) == 0) { return; }

                // Interpolate to get the appropriate charge density in space
                Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, 0, 0);

                // Interpolate current to appropriate staggering to match E field
                Real jtot_val = 0._rt;
                if (solve_for_Faraday && resistivity_has_J_dependence) {
                    const Real jr_val = Interp(Jr, Jr_stag, Ez_stag, coarsen, i, j, 0, 0);
                    const Real jt_val = Interp(Jt, Jt_stag, Ez_stag, coarsen, i, j, 0, 0);
                    const Real jz_val = Interp(Jz, Jz_stag, Ez_stag, coarsen, i, j, 0, 0);
                    jtot_val = std::sqrt(jr_val*jr_val + jt_val*jt_val + jz_val*jz_val);
                }

                // safety condition since we divide by rho_val later
                if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Get the gradient of the electron pressure if the longitudinal part of
                // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                Real grad_Pe = 0._rt;
                if (!solve_for_Faraday) { grad_Pe = T_Algo::UpwardDz(Pe, coefs_z, n_coefs_z, i, j, 0, 0); }

                // interpolate the nodal neE values to the Yee grid
                auto enE_z = Interp(enE, nodal, Ez_stag, coarsen, i, j, 0, 2);

                Ez(i, j, 0) = (enE_z - grad_Pe) / rho_val;

                // Add resistivity only if E field value is used to update B
                if (solve_for_Faraday) { Ez(i, j, 0) += eta(rho_val, jtot_val) * Jz(i, j, 0); }

                if (include_hyper_resistivity_term) {
                    auto nabla2Jz = T_Algo::Dzz(Jz, coefs_z, n_coefs_z, i, j, 0, 0);
                    Ez(i, j, 0) -= eta_h * nabla2Jz;
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

    const bool include_hyper_resistivity_term = (eta_h > 0.) && solve_for_Faraday;

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

        // Loop over the cells and update the nodal E field
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k){

            // interpolate the total plasma current to a nodal grid
            auto const jx_interp = Interp(Jx, Jx_stag, nodal, coarsen, i, j, k, 0);
            auto const jy_interp = Interp(Jy, Jy_stag, nodal, coarsen, i, j, k, 0);
            auto const jz_interp = Interp(Jz, Jz_stag, nodal, coarsen, i, j, k, 0);

            // interpolate the ion current to a nodal grid
            auto const jix_interp = Interp(Jix, Jx_stag, nodal, coarsen, i, j, k, 0);
            auto const jiy_interp = Interp(Jiy, Jy_stag, nodal, coarsen, i, j, k, 0);
            auto const jiz_interp = Interp(Jiz, Jz_stag, nodal, coarsen, i, j, k, 0);

            // interpolate the B field to a nodal grid
            auto const Bx_interp = Interp(Bx, Bx_stag, nodal, coarsen, i, j, k, 0);
            auto const By_interp = Interp(By, By_stag, nodal, coarsen, i, j, k, 0);
            auto const Bz_interp = Interp(Bz, Bz_stag, nodal, coarsen, i, j, k, 0);

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

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries
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

        Box const& tex  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tey  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Loop over the cells and update the E field
        amrex::ParallelFor(tex, tey, tez,

            // Ex calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Skip field update in the embedded boundaries
                if (update_Ex_arr && update_Ex_arr(i, j, k) == 0) { return; }

                // Interpolate to get the appropriate charge density in space
                Real rho_val = Interp(rho, nodal, Ex_stag, coarsen, i, j, k, 0);

                // Interpolate current to appropriate staggering to match E field
                Real jtot_val = 0._rt;
                if (solve_for_Faraday && resistivity_has_J_dependence) {
                    const Real jx_val = Interp(Jx, Jx_stag, Ex_stag, coarsen, i, j, k, 0);
                    const Real jy_val = Interp(Jy, Jy_stag, Ex_stag, coarsen, i, j, k, 0);
                    const Real jz_val = Interp(Jz, Jz_stag, Ex_stag, coarsen, i, j, k, 0);
                    jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);
                }

                // safety condition since we divide by rho_val later
                if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Get the gradient of the electron pressure if the longitudinal part of
                // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                Real grad_Pe = 0._rt;
                if (!solve_for_Faraday) { grad_Pe = T_Algo::UpwardDx(Pe, coefs_x, n_coefs_x, i, j, k); }

                // interpolate the nodal neE values to the Yee grid
                auto enE_x = Interp(enE, nodal, Ex_stag, coarsen, i, j, k, 0);

                Ex(i, j, k) = (enE_x - grad_Pe) / rho_val;

                // Add resistivity only if E field value is used to update B
                if (solve_for_Faraday) { Ex(i, j, k) += eta(rho_val, jtot_val) * Jx(i, j, k); }

                if (include_hyper_resistivity_term) {
                    auto nabla2Jx = T_Algo::Dxx(Jx, coefs_x, n_coefs_x, i, j, k);
                    Ex(i, j, k) -= eta_h * nabla2Jx;
                }
            },

            // Ey calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                // Skip field update in the embedded boundaries
                if (update_Ey_arr && update_Ey_arr(i, j, k) == 0) { return; }

                // Interpolate to get the appropriate charge density in space
                Real rho_val = Interp(rho, nodal, Ey_stag, coarsen, i, j, k, 0);

                // Interpolate current to appropriate staggering to match E field
                Real jtot_val = 0._rt;
                if (solve_for_Faraday && resistivity_has_J_dependence) {
                    const Real jx_val = Interp(Jx, Jx_stag, Ey_stag, coarsen, i, j, k, 0);
                    const Real jy_val = Interp(Jy, Jy_stag, Ey_stag, coarsen, i, j, k, 0);
                    const Real jz_val = Interp(Jz, Jz_stag, Ey_stag, coarsen, i, j, k, 0);
                    jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);
                }

                // safety condition since we divide by rho_val later
                if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Get the gradient of the electron pressure if the longitudinal part of
                // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                Real grad_Pe = 0._rt;
                if (!solve_for_Faraday) { grad_Pe = T_Algo::UpwardDy(Pe, coefs_y, n_coefs_y, i, j, k); }

                // interpolate the nodal neE values to the Yee grid
                auto enE_y = Interp(enE, nodal, Ey_stag, coarsen, i, j, k, 1);

                Ey(i, j, k) = (enE_y - grad_Pe) / rho_val;

                // Add resistivity only if E field value is used to update B
                if (solve_for_Faraday) { Ey(i, j, k) += eta(rho_val, jtot_val) * Jy(i, j, k); }

                if (include_hyper_resistivity_term) {
                    auto nabla2Jy = T_Algo::Dyy(Jy, coefs_y, n_coefs_y, i, j, k);
                    Ey(i, j, k) -= eta_h * nabla2Jy;
                }
            },

            // Ez calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Skip field update in the embedded boundaries
                if (update_Ez_arr && update_Ez_arr(i, j, k) == 0) { return; }

                // Interpolate to get the appropriate charge density in space
                Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, k, 0);

                // Interpolate current to appropriate staggering to match E field
                Real jtot_val = 0._rt;
                if (solve_for_Faraday && resistivity_has_J_dependence) {
                    const Real jx_val = Interp(Jx, Jx_stag, Ez_stag, coarsen, i, j, k, 0);
                    const Real jy_val = Interp(Jy, Jy_stag, Ez_stag, coarsen, i, j, k, 0);
                    const Real jz_val = Interp(Jz, Jz_stag, Ez_stag, coarsen, i, j, k, 0);
                    jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);
                }

                // safety condition since we divide by rho_val later
                if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Get the gradient of the electron pressure if the longitudinal part of
                // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                Real grad_Pe = 0._rt;
                if (!solve_for_Faraday) { grad_Pe = T_Algo::UpwardDz(Pe, coefs_z, n_coefs_z, i, j, k); }

                // interpolate the nodal neE values to the Yee grid
                auto enE_z = Interp(enE, nodal, Ez_stag, coarsen, i, j, k, 2);

                Ez(i, j, k) = (enE_z - grad_Pe) / rho_val;

                // Add resistivity only if E field value is used to update B
                if (solve_for_Faraday) { Ez(i, j, k) += eta(rho_val, jtot_val) * Jz(i, j, k); }

                if (include_hyper_resistivity_term) {
                    auto nabla2Jz = T_Algo::Dzz(Jz, coefs_z, n_coefs_z, i, j, k);
                    Ez(i, j, k) -= eta_h * nabla2Jz;
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
#endif
