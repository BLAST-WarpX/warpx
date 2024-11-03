/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FiniteDifferenceSolver.H"

#include "Fields.H"
#ifndef WARPX_DIM_RZ
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#else
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#endif
#include "EmbeddedBoundary/Enabled.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

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
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <memory>

using namespace amrex;
using namespace ablastr::fields;

/**
 * \brief Update the E field, over one timestep
 */
void FiniteDifferenceSolver::EvolveE (
    ablastr::fields::MultiFabRegister & fields,
    int lev,
    PatchType patch_type,
    ablastr::fields::VectorField const& Efield,
    amrex::Real const dt
)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    const ablastr::fields::VectorField Bfield = patch_type == PatchType::fine ?
        fields.get_alldirs(FieldType::Bfield_fp, lev) : fields.get_alldirs(FieldType::Bfield_cp, lev);
    const ablastr::fields::VectorField Jfield = patch_type == PatchType::fine ?
        fields.get_alldirs(FieldType::current_fp, lev) : fields.get_alldirs(FieldType::current_cp, lev);

    amrex::MultiFab* Ffield = nullptr;
    if (fields.has(FieldType::F_fp, lev)) {
        Ffield = patch_type == PatchType::fine ?
                 fields.get(FieldType::F_fp, lev) : fields.get(FieldType::F_cp, lev);
    }

    ablastr::fields::VectorField edge_lengths;
    if (fields.has_vector(FieldType::edge_lengths, lev)) {
        edge_lengths = fields.get_alldirs(FieldType::edge_lengths, lev);
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

    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
#ifdef WARPX_DIM_RZ
    if (m_fdtd_algo == ElectromagneticSolverAlgo::Yee){
        EvolveECylindrical <CylindricalYeeAlgorithm> ( Efield, Bfield, Jfield, edge_lengths, Ffield, lev, dt );
#else
    if (m_grid_type == GridType::Collocated) {

        EvolveECartesian <CartesianNodalAlgorithm> ( Efield, Bfield, Jfield, edge_lengths, Ffield, lev, dt );

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::Yee || m_fdtd_algo == ElectromagneticSolverAlgo::ECT) {

        EvolveECartesian <CartesianYeeAlgorithm> ( Efield, Bfield, Jfield, edge_lengths, Ffield, lev, dt );

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::CKC) {

        EvolveECartesian <CartesianCKCAlgorithm> ( Efield, Bfield, Jfield, edge_lengths, Ffield, lev, dt );

#endif
    } else {
        WARPX_ABORT_WITH_MESSAGE("EvolveE: Unknown algorithm");
    }

}


#ifndef WARPX_DIM_RZ

template<typename T_Algo>
void FiniteDifferenceSolver::EvolveECartesian (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Bfield,
    ablastr::fields::VectorField const& Jfield,
    VectorField const& edge_lengths,
    amrex::MultiFab const* Ffield,
    int lev, amrex::Real const dt ) {

#ifndef AMREX_USE_EB
    amrex::ignore_unused(edge_lengths);
#endif

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);
    Real constexpr c2 = PhysConst::c * PhysConst::c;

    // Loop through the grids, and over the tiles within each grid
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
        Array4<Real> const& Bx = Bfield[0]->array(mfi);
        Array4<Real> const& By = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);
        Array4<Real> const& jx = Jfield[0]->array(mfi);
        Array4<Real> const& jy = Jfield[1]->array(mfi);
        Array4<Real> const& jz = Jfield[2]->array(mfi);

        amrex::Array4<amrex::Real> lx, ly, lz;
        if (EB::enabled()) {
            lx = edge_lengths[0]->array(mfi);
            ly = edge_lengths[1]->array(mfi);
            lz = edge_lengths[2]->array(mfi);
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

        // Loop over the cells and update the fields
        amrex::ParallelFor(tex, tey, tez,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field push if this cell is fully covered by embedded boundaries
                if (lx && lx(i, j, k) <= 0) { return; }

                Ex(i, j, k) += c2 * dt * (
                    - T_Algo::DownwardDz(By, coefs_z, n_coefs_z, i, j, k)
                    + T_Algo::DownwardDy(Bz, coefs_y, n_coefs_y, i, j, k)
                    - PhysConst::mu0 * jx(i, j, k) );
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field push if this cell is fully covered by embedded boundaries
#ifdef WARPX_DIM_3D
                if (ly && ly(i,j,k) <= 0) { return; }
#elif defined(WARPX_DIM_XZ)
                //In XZ Ey is associated with a mesh node, so we need to check if the mesh node is covered
                amrex::ignore_unused(ly);
                if (lx && (lx(i, j, k)<=0 || lx(i-1, j, k)<=0 || lz(i, j-1, k)<=0 || lz(i, j, k)<=0)) { return; }
#endif

                Ey(i, j, k) += c2 * dt * (
                    - T_Algo::DownwardDx(Bz, coefs_x, n_coefs_x, i, j, k)
                    + T_Algo::DownwardDz(Bx, coefs_z, n_coefs_z, i, j, k)
                    - PhysConst::mu0 * jy(i, j, k) );
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field push if this cell is fully covered by embedded boundaries
                if (lz && lz(i,j,k) <= 0) { return; }
                Ez(i, j, k) += c2 * dt * (
                    - T_Algo::DownwardDy(Bx, coefs_y, n_coefs_y, i, j, k)
                    + T_Algo::DownwardDx(By, coefs_x, n_coefs_x, i, j, k)
                    - PhysConst::mu0 * jz(i, j, k) );
            }

        );

        // If F is not a null pointer, further update E using the grad(F) term
        // (hyperbolic correction for errors in charge conservation)
        if (Ffield) {

            // Extract field data for this grid/tile
            const Array4<Real const> F = Ffield->array(mfi);

            // Loop over the cells and update the fields
            amrex::ParallelFor(tex, tey, tez,

                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    Ex(i, j, k) += c2 * dt * T_Algo::UpwardDx(F, coefs_x, n_coefs_x, i, j, k);
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    Ey(i, j, k) += c2 * dt * T_Algo::UpwardDy(F, coefs_y, n_coefs_y, i, j, k);
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    Ez(i, j, k) += c2 * dt * T_Algo::UpwardDz(F, coefs_z, n_coefs_z, i, j, k);
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

#else // corresponds to ifndef WARPX_DIM_RZ

template<typename T_Algo>
void FiniteDifferenceSolver::EvolveECylindrical (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Bfield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& edge_lengths,
    amrex::MultiFab const* Ffield,
    int lev, amrex::Real const dt ) {

#ifndef AMREX_USE_EB
    amrex::ignore_unused(edge_lengths);
#endif

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Loop through the grids, and over the tiles within each grid
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
        Array4<Real> const& Er = Efield[0]->array(mfi);
        Array4<Real> const& Et = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real> const& Br = Bfield[0]->array(mfi);
        Array4<Real> const& Bt = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);
        Array4<Real> const& jr = Jfield[0]->array(mfi);
        Array4<Real> const& jt = Jfield[1]->array(mfi);
        Array4<Real> const& jz = Jfield[2]->array(mfi);

        amrex::Array4<amrex::Real> lr, lz;
        if (EB::enabled()) {
            lr = edge_lengths[0]->array(mfi);
            lz = edge_lengths[2]->array(mfi);
        }

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
        Box const& ter  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tet  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        Real const c2 = PhysConst::c * PhysConst::c;

        // Loop over the cells and update the fields
        amrex::ParallelFor(ter, tet, tez,

            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                // Skip field push if this cell is fully covered by embedded boundaries
                if (lr && lr(i, j, 0) <= 0) { return; }

                Real const r = rmin + (i + 0.5_rt)*dr; // r on cell-centered point (Er is cell-centered in r)
                Er(i, j, 0, 0) +=  c2 * dt*(
                    - T_Algo::DownwardDz(Bt, coefs_z, n_coefs_z, i, j, 0, 0)
                    - PhysConst::mu0 * jr(i, j, 0, 0) ); // Mode m=0
                for (int m=1; m<nmodes; m++) { // Higher-order modes
                    Er(i, j, 0, 2*m-1) += c2 * dt*(
                        - T_Algo::DownwardDz(Bt, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                        + m * Bz(i, j, 0, 2*m  )/r
                        - PhysConst::mu0 * jr(i, j, 0, 2*m-1) );  // Real part
                    Er(i, j, 0, 2*m  ) += c2 * dt*(
                        - T_Algo::DownwardDz(Bt, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                        - m * Bz(i, j, 0, 2*m-1)/r
                        - PhysConst::mu0 * jr(i, j, 0, 2*m  ) ); // Imaginary part
                }
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                // Skip field push if this cell is fully covered by embedded boundaries
                // The Et field is at a node, so we need to check if the node is covered
                if (lr && (lr(i, j, 0)<=0 || lr(i-1, j, 0)<=0 || lz(i, j-1, 0)<=0 || lz(i, j, 0)<=0)) { return; }

                Real const r = rmin + i*dr; // r on a nodal grid (Et is nodal in r)
                if (r != 0) { // Off-axis, regular Maxwell equations
                    Et(i, j, 0, 0) += c2 * dt*(
                        - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 0)
                        + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 0)
                        - PhysConst::mu0 * jt(i, j, 0, 0 ) ); // Mode m=0
                    for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                        Et(i, j, 0, 2*m-1) += c2 * dt*(
                            - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                            + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                            - PhysConst::mu0 * jt(i, j, 0, 2*m-1) ); // Real part
                        Et(i, j, 0, 2*m  ) += c2 * dt*(
                            - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 2*m  )
                            + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                            - PhysConst::mu0 * jt(i, j, 0, 2*m  ) ); // Imaginary part
                    }
                } else { // r==0: on-axis corrections
                    // Ensure that Et remains 0 on axis (except for m=1)
                    Et(i, j, 0, 0) = 0.; // Mode m=0
                    for (int m=1; m<nmodes; m++) { // Higher-order modes
                        if (m == 1){
                            // The bulk equation could in principle be used here since it does not diverge
                            // on axis. However, it typically gives poor results e.g. for the propagation
                            // of a laser pulse (the field is spuriously reduced on axis). For this reason
                            // a modified on-axis condition is used here: we use the fact that
                            // Etheta(r=0,m=1) should equal -iEr(r=0,m=1), for the fields Er and Et to be
                            // independent of theta at r=0. Now with linear interpolation:
                            // Er(r=0,m=1) = 0.5*[Er(r=dr/2,m=1) + Er(r=-dr/2,m=1)]
                            // And using the rule applying for the guards cells
                            // Er(r=-dr/2,m=1) = Er(r=dr/2,m=1). Thus: Et(i,j,m) = -i*Er(i,j,m)
                            Et(i,j,0,2*m-1) =  Er(i,j,0,2*m  );
                            Et(i,j,0,2*m  ) = -Er(i,j,0,2*m-1);
                        } else {
                            Et(i, j, 0, 2*m-1) = 0.;
                            Et(i, j, 0, 2*m  ) = 0.;
                        }
                    }
                }
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                // Skip field push if this cell is fully covered by embedded boundaries
                if (lz && lz(i, j, 0) <= 0) { return; }

                Real const r = rmin + i*dr; // r on a nodal grid (Ez is nodal in r)
                if (r != 0) { // Off-axis, regular Maxwell equations
                    Ez(i, j, 0, 0) += c2 * dt*(
                       T_Algo::DownwardDrr_over_r(Bt, r, dr, coefs_r, n_coefs_r, i, j, 0, 0)
                        - PhysConst::mu0 * jz(i, j, 0, 0  ) ); // Mode m=0
                    for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                        Ez(i, j, 0, 2*m-1) += c2 * dt *(
                            - m * Br(i, j, 0, 2*m  )/r
                            + T_Algo::DownwardDrr_over_r(Bt, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                            - PhysConst::mu0 * jz(i, j, 0, 2*m-1) ); // Real part
                        Ez(i, j, 0, 2*m  ) += c2 * dt *(
                            m * Br(i, j, 0, 2*m-1)/r
                            + T_Algo::DownwardDrr_over_r(Bt, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m  )
                            - PhysConst::mu0 * jz(i, j, 0, 2*m  ) ); // Imaginary part
                    }
                } else { // r==0: on-axis corrections
                    // For m==0, Bt is linear in r, for small r
                    // Therefore, the formula below regularizes the singularity
                    Ez(i, j, 0, 0) += c2 * dt*(
                         4*Bt(i, j, 0, 0)/dr // regularization
                         - PhysConst::mu0 * jz(i, j, 0, 0  ) );
                    // Ensure that Ez remains 0 for higher-order modes
                    for (int m=1; m<nmodes; m++) {
                        Ez(i, j, 0, 2*m-1) = 0.;
                        Ez(i, j, 0, 2*m  ) = 0.;
                    }
                }
            }

        ); // end of loop over cells

        // If F is not a null pointer, further update E using the grad(F) term
        // (hyperbolic correction for errors in charge conservation)
        if (Ffield) {

            // Extract field data for this grid/tile
            const Array4<Real const> F = Ffield->array(mfi);

            // Loop over the cells and update the fields
            amrex::ParallelFor(ter, tet, tez,

                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    Er(i, j, 0, 0) += c2 * dt * T_Algo::UpwardDr(F, coefs_r, n_coefs_r, i, j, 0, 0);
                    for (int m=1; m<nmodes; m++) { // Higher-order modes
                        Er(i, j, 0, 2*m-1) += c2 * dt * T_Algo::UpwardDr(F, coefs_r, n_coefs_r, i, j, 0, 2*m-1); // Real part
                        Er(i, j, 0, 2*m  ) += c2 * dt * T_Algo::UpwardDr(F, coefs_r, n_coefs_r, i, j, 0, 2*m  ); // Imaginary part
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    // Mode m=0: no update
                    Real const r = rmin + i*dr; // r on a nodal grid (Et is nodal in r)
                    if (r != 0){ // Off-axis, regular Maxwell equations
                        for (int m=1; m<nmodes; m++) { // Higher-order modes
                            Et(i, j, 0, 2*m-1) += c2 * dt *  m * F(i, j, 0, 2*m  )/r; // Real part
                            Et(i, j, 0, 2*m  ) += c2 * dt * -m * F(i, j, 0, 2*m-1)/r; // Imaginary part
                        }
                    } else { // r==0: on-axis corrections
                        // For m==1, F is linear in r, for small r
                        // Therefore, the formula below regularizes the singularity
                        if (nmodes >= 2) { // needs to have at least m=0 and m=1
                            int const m=1;
                            Et(i, j, 0, 2*m-1) += c2 * dt *  m * F(i+1, j, 0, 2*m  )/dr; // Real part
                            Et(i, j, 0, 2*m  ) += c2 * dt * -m * F(i+1, j, 0, 2*m-1)/dr; // Imaginary part
                        }
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    Ez(i, j, 0, 0) += c2 * dt * T_Algo::UpwardDz(F, coefs_z, n_coefs_z, i, j, 0, 0);
                    for (int m=1; m<nmodes; m++) { // Higher-order modes
                        Ez(i, j, 0, 2*m-1) += c2 * dt * T_Algo::UpwardDz(F, coefs_z, n_coefs_z, i, j, 0, 2*m-1); // Real part
                        Ez(i, j, 0, 2*m  ) += c2 * dt * T_Algo::UpwardDz(F, coefs_z, n_coefs_z, i, j, 0, 2*m  ); // Imaginary part
                    }
                }

            ); // end of loop over cells

        } // end of if condition for F

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    } // end of loop over grid/tiles

}

#endif // corresponds to ifndef WARPX_DIM_RZ
