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
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();

        // Extract tileboxes for which to loop
        Box const& tex  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tey  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Loop over the cells and update the fields
        amrex::ParallelFor(

            tex, m_ncomps,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n){
                // Skip field push if this cell is fully covered by embedded boundaries
                if (lx && lx(i, j, k) <= 0) { return; }

                Ex(i, j, k, n) += c2 * dt * (
                    + T_Algo::Curl_Nodal_0(By, Bz, coefs_y, coefs_z, i, j, k, n)
                    - PhysConst::mu0 * jx(i, j, k, n) );
            },

            tey, m_ncomps,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n){
                // Skip field push if this cell is fully covered by embedded boundaries
#ifdef WARPX_DIM_3D
                if (ly && ly(i,j,k) <= 0) { return; }
#elif defined(WARPX_DIM_XZ)
                //In XZ Ey is associated with a mesh node, so we need to check if the mesh node is covered
                amrex::ignore_unused(ly);
                if (lx && (lx(i, j, k)<=0 || lx(i-1, j, k)<=0 || lz(i, j-1, k)<=0 || lz(i, j, k)<=0)) { return; }
#endif

                Ey(i, j, k, n) += c2 * dt * (
                    + T_Algo::Curl_Nodal_1(Bz, Bx, coefs_z, coefs_x, i, j, k, n)
                    - PhysConst::mu0 * jy(i, j, k) );
            },

            tez, m_ncomps,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n){
                // Skip field push if this cell is fully covered by embedded boundaries
                if (lz && lz(i,j,k) <= 0) { return; }
                Ez(i, j, k, n) += c2 * dt * (
                    + T_Algo::Curl_Nodal_2(Bx, By, coefs_x, coefs_y, i, j, k, n)
                    - PhysConst::mu0 * jz(i, j, k) );
            }

        );

        // If F is not a null pointer, further update E using the grad(F) term
        // (hyperbolic correction for errors in charge conservation)
        if (Ffield) {

            // Extract field data for this grid/tile
            const Array4<Real const> F = Ffield->array(mfi);

            // Loop over the cells and update the fields
            amrex::ParallelFor(

                tex, m_ncomps,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n){
                    Ex(i, j, k, n) += c2 * dt * T_Algo::Grad_cell_0(F, coefs_x, i, j, k, n);
                },

                tey, m_ncomps,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n){
                    Ey(i, j, k, n) += c2 * dt * T_Algo::Grad_cell_1(F, coefs_y, i, j, k, n);
                },

                tez, m_ncomps,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n){
                    Ez(i, j, k, n) += c2 * dt * T_Algo::Grad_cell_2(F, coefs_z, i, j, k, n);
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
        Real const * const AMREX_RESTRICT coefs_t = m_stencil_coefs_t.dataPtr();
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();

        // Extract tileboxes for which to loop
        Box const& ter  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tet  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        Real const c2 = PhysConst::c * PhysConst::c;

        // Loop over the cells and update the fields
        amrex::ParallelFor(

            ter, m_ncomps,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n){
                // Skip field push if this cell is fully covered by embedded boundaries
                if (lr && lr(i, j, 0) <= 0) { return; }

                Er(i, j, k, n) += c2 * dt * (
                    + T_Algo::Curl_Nodal_0(Bt, Bz, coefs_t, coefs_z, i, j, k, n)
                    - PhysConst::mu0 * jr(i, j, k, n) );

            },

            tet, m_ncomps,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n){
                // Skip field push if this cell is fully covered by embedded boundaries
                // The Et field is at a node, so we need to check if the node is covered
                if (lr && (lr(i, j, 0)<=0 || lr(i-1, j, 0)<=0 || lz(i, j-1, 0)<=0 || lz(i, j, 0)<=0)) { return; }

                Et(i, j, k, n) += c2 * dt*(
                    + T_Algo::Curl_Nodal_1(Bz, Br, coefs_z, coefs_r, i, j, k, n)
                    - PhysConst::mu0 * jt(i, j, 0, n) );
            },

            tez, m_ncomps,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n){
                // Skip field push if this cell is fully covered by embedded boundaries
                if (lz && lz(i, j, 0) <= 0) { return; }

                Ez(i, j, k, n) += c2 * dt*(
                    + T_Algo::Curl_Nodal_2(Br, Bt, coefs_r, coefs_t, i, j, k, n)
                    - PhysConst::mu0 * jz(i, j, 0, n) );
            }

        ); // end of loop over cells

        // Place holder
        amrex::ParallelFor(tet, m_ncomps,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                T_Algo::UpdateEthetaOnAxis(Et, Er, coefs_t, i, j, k, n);
            }
        );

        // If F is not a null pointer, further update E using the grad(F) term
        // (hyperbolic correction for errors in charge conservation)
        if (Ffield) {

            // Extract field data for this grid/tile
            const Array4<Real const> F = Ffield->array(mfi);

            // Loop over the cells and update the fields
            amrex::ParallelFor(

                ter, m_ncomps,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n){
                    Er(i, j, k, n) += c2 * dt * T_Algo::Grad_cell_0(F, coefs_r, i, j, k, n);
                },

                tet, m_ncomps,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n){
                    Et(i, j, k, n) += c2 * dt * T_Algo::Grad_cell_1(F, coefs_t, i, j, k, n);
                },

                tez, m_ncomps,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n){
                    Ez(i, j, k, n) += c2 * dt * T_Algo::Grad_cell_2(F, coefs_z, i, j, k, n);
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
