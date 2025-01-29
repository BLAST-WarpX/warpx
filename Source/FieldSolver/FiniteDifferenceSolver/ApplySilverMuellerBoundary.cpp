/* Copyright 2021
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FiniteDifferenceSolver.H"

#ifdef WARPX_DIM_RZ
#   include "FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#endif
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <array>
#include <memory>

using namespace amrex;

/**
 * \brief Update the B field at the boundary, using the Silver-Mueller condition
 */
void FiniteDifferenceSolver::ApplySilverMuellerBoundary (
    ablastr::fields::VectorField& Efield,
    ablastr::fields::VectorField& Bfield,
    amrex::Box domain_box,
    amrex::Real const dt,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> field_boundary_lo,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> field_boundary_hi) {

    using ablastr::fields::Direction;

    // Ensure that we are using the Yee solver
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_fdtd_algo == ElectromagneticSolverAlgo::Yee,
        "The Silver-Mueller boundary conditions can only be used with the Yee solver."
    );

    // Ensure that we are using the cells the domain
    domain_box.enclosedCells();

#ifdef WARPX_DIM_RZ
    // Calculate relevant coefficients
    amrex::Real const cdt = PhysConst::c*dt;
    amrex::Real const cdt_over_dr = cdt*m_h_stencil_coefs_r[0];
    amrex::Real const coef1_r = (1._rt - cdt_over_dr)/(1._rt + cdt_over_dr);
    amrex::Real const coef2_r = 2._rt*cdt_over_dr/(1._rt + cdt_over_dr) / PhysConst::c;
    amrex::Real const coef3_r = cdt/(1._rt + cdt_over_dr) / PhysConst::c;
    amrex::Real const cdt_over_dz = cdt*m_h_stencil_coefs_z[0];
    amrex::Real const coef1_z = (1._rt - cdt_over_dz)/(1._rt + cdt_over_dz);
    amrex::Real const coef2_z = 2._rt*cdt_over_dz/(1._rt + cdt_over_dz) / PhysConst::c;

    // Extract stencil coefficients
    Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
    auto const n_coefs_z = static_cast<int>(m_h_stencil_coefs_z.size());

    // Extract cylindrical specific parameters
    Real const dr = m_dr;
    int const nmodes = m_nmodes;
    Real const rmin = m_rmin;

    // Infer whether the Silver-Mueller needs to be applied in each direction
    bool const apply_hi_r = (field_boundary_hi[0] == FieldBoundaryType::Absorbing_SilverMueller);
    bool const apply_lo_z = (field_boundary_lo[1] == FieldBoundaryType::Absorbing_SilverMueller);
    bool const apply_hi_z = (field_boundary_hi[1] == FieldBoundaryType::Absorbing_SilverMueller);

    // tiling is usually set by TilingIfNotGPU()
    // but here, we set it to false because of potential race condition,
    // since we grow the tiles by one guard cell after creating them.
    for ( MFIter mfi(*Efield[Direction{0}], false); mfi.isValid(); ++mfi ) {
        // Extract field data for this grid/tile
        Array4<Real> const& Er = Efield[Direction{0}]->array(mfi);
        Array4<Real> const& Et = Efield[Direction{1}]->array(mfi);
        Array4<Real> const& Ez = Efield[Direction{2}]->array(mfi);
        Array4<Real> const& Br = Bfield[Direction{0}]->array(mfi);
        Array4<Real> const& Bt = Bfield[Direction{1}]->array(mfi);
        Array4<Real> const& Bz = Bfield[Direction{2}]->array(mfi);

        // Extract tileboxes for which to loop
        Box tbr  = mfi.tilebox(Bfield[0]->ixType().toIntVect());
        Box tbt  = mfi.tilebox(Bfield[1]->ixType().toIntVect());
        Box tbz  = mfi.tilebox(Bfield[2]->ixType().toIntVect());

        // We will modify the first (i.e. innermost) guard cell
        // (if it is outside of the physical domain)
        // Thus, the tileboxes here are grown by 1 guard cell
        tbr.grow(1);
        tbt.grow(1);
        tbz.grow(1);

        // Loop over the cells
        amrex::ParallelFor(tbr, tbt, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // At the +z boundary (innermost guard cell)
                if ( apply_hi_z && (j==domain_box.bigEnd(1)+1) ){
                    for (int m=0; m<2*nmodes-1; m++) {
                        Br(i,j,0,m) = coef1_z*Br(i,j,0,m) - coef2_z*Et(i,j,0,m);
                    }
                }
                // At the -z boundary (innermost guard cell)
                if ( apply_lo_z && (j==domain_box.smallEnd(1)-1) ){
                    for (int m=0; m<2*nmodes-1; m++) {
                        Br(i,j,0,m) = coef1_z*Br(i,j,0,m) + coef2_z*Et(i,j+1,0,m);
                    }
                }

            },
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // At the +z boundary (innermost guard cell)
                if ( apply_hi_z && (j==domain_box.bigEnd(1)+1) ){
                    for (int m=0; m<2*nmodes-1; m++) {
                        Bt(i,j,0,m) = coef1_z*Bt(i,j,0,m) + coef2_z*Er(i,j,0,m);
                    }
                }
                // At the -z boundary (innermost guard cell)
                if ( apply_lo_z && (j==domain_box.smallEnd(1)-1) ){
                    for (int m=0; m<2*nmodes-1; m++) {
                        Bt(i,j,0,m) = coef1_z*Bt(i,j,0,m) - coef2_z*Er(i,j+1,0,m);
                    }
                }
                // At the +r boundary (innermost guard cell)
                if ( apply_hi_r && (i==domain_box.bigEnd(0)+1) ){
                    // Mode 0
                    Bt(i,j,0,0) = coef1_r*Bt(i,j,0,0) - coef2_r*Ez(i,j,0,0)
                        + coef3_r*CylindricalYeeAlgorithm::UpwardDz(Er, coefs_z, n_coefs_z, i, j, 0, 0);
                    for (int m=1; m<nmodes; m++) { // Higher-order modes
                        // Real part
                        Bt(i,j,0,2*m-1) = coef1_r*Bt(i,j,0,2*m-1) - coef2_r*Ez(i,j,0,2*m-1)
                            + coef3_r*CylindricalYeeAlgorithm::UpwardDz(Er, coefs_z, n_coefs_z, i, j, 0, 2*m-1);
                        // Imaginary part
                        Bt(i,j,0,2*m) = coef1_r*Bt(i,j,0,2*m) - coef2_r*Ez(i,j,0,2*m)
                            + coef3_r*CylindricalYeeAlgorithm::UpwardDz(Er, coefs_z, n_coefs_z, i, j, 0, 2*m);
                    }
                }

            },
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // At the +r boundary (innermost guard cell)
                if ( apply_hi_r && (i==domain_box.bigEnd(0)+1) ){
                    Real const r = rmin + (i + 0.5_rt)*dr; // r on nodal point (Bz is cell-centered in r)
                    // Mode 0
                    Bz(i,j,0,0) = coef1_r*Bz(i,j,0,0) + coef2_r*Et(i,j,0,0) - coef3_r*Et(i,j,0,0)/r;
                    for (int m=1; m<nmodes; m++) { // Higher-order modes
                        // Real part
                        Bz(i,j,0,2*m-1) = coef1_r*Bz(i,j,0,2*m-1) + coef2_r*Et(i,j,0,2*m-1)
                            - coef3_r/r*(Et(i,j,0,2*m-1) - m*Er(i,j,0,2*m));
                        // Imaginary part
                        Bz(i,j,0,2*m) = coef1_r*Bz(i,j,0,2*m) + coef2_r*Et(i,j,0,2*m)
                            - coef3_r/r*(Et(i,j,0,2*m) + m*Er(i,j,0,2*m-1));
                    }
                }

            }
        );
    }
#else

    // Calculate relevant coefficients
#if (defined WARPX_DIM_3D || WARPX_DIM_XZ)
    amrex::Real const cdt_over_dx = PhysConst::c*dt*m_h_stencil_coefs_x[0];
    amrex::Real const coef1_x = (1._rt - cdt_over_dx)/(1._rt + cdt_over_dx);
    amrex::Real const coef2_x = 2._rt*cdt_over_dx/(1._rt + cdt_over_dx) / PhysConst::c;
#endif
#ifdef WARPX_DIM_3D
    amrex::Real const cdt_over_dy = PhysConst::c*dt*m_h_stencil_coefs_y[0];
    amrex::Real const coef1_y = (1._rt - cdt_over_dy)/(1._rt + cdt_over_dy);
    amrex::Real const coef2_y = 2._rt*cdt_over_dy/(1._rt + cdt_over_dy) / PhysConst::c;
#endif
    amrex::Real const cdt_over_dz = PhysConst::c*dt*m_h_stencil_coefs_z[0];
    amrex::Real const coef1_z = (1._rt - cdt_over_dz)/(1._rt + cdt_over_dz);
    amrex::Real const coef2_z = 2._rt*cdt_over_dz/(1._rt + cdt_over_dz) / PhysConst::c;

#if (defined WARPX_DIM_3D || WARPX_DIM_XZ)
    bool const apply_lo_x = (field_boundary_lo[0] == FieldBoundaryType::Absorbing_SilverMueller);
    bool const apply_hi_x = (field_boundary_hi[0] == FieldBoundaryType::Absorbing_SilverMueller);
#endif
#ifdef WARPX_DIM_3D
    bool const apply_lo_y = (field_boundary_lo[1] == FieldBoundaryType::Absorbing_SilverMueller);
    bool const apply_hi_y = (field_boundary_hi[1] == FieldBoundaryType::Absorbing_SilverMueller);
#endif
    bool const apply_lo_z = (field_boundary_lo[WARPX_ZINDEX] == FieldBoundaryType::Absorbing_SilverMueller);
    bool const apply_hi_z = (field_boundary_hi[WARPX_ZINDEX] == FieldBoundaryType::Absorbing_SilverMueller);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    // tiling is usually set by TilingIfNotGPU()
    // but here, we set it to false because of potential race condition,
    // since we grow the tiles by one guard cell after creating them.
    for ( MFIter mfi(*Efield[Direction{0}], false); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const& Ex = Efield[Direction{0}]->array(mfi);
        Array4<Real> const& Ey = Efield[Direction{1}]->array(mfi);
#ifndef WARPX_DIM_1D_Z
        Array4<Real> const& Ez = Efield[Direction{2}]->array(mfi);
#endif
        Array4<Real> const& Bx = Bfield[Direction{0}]->array(mfi);
        Array4<Real> const& By = Bfield[Direction{1}]->array(mfi);
#ifndef WARPX_DIM_1D_Z
        Array4<Real> const& Bz = Bfield[Direction{2}]->array(mfi);
#endif

        // Extract the tileboxes for which to loop
        Box tbx  = mfi.tilebox(Bfield[0]->ixType().toIntVect());
        Box tby  = mfi.tilebox(Bfield[1]->ixType().toIntVect());
        Box tbz  = mfi.tilebox(Bfield[2]->ixType().toIntVect());

        // We will modify the first (i.e. innermost) guard cell
        // (if it is outside of the physical domain)
        // Thus, the tileboxes here are grown by 1 guard cell
        tbx.grow(1);
        tby.grow(1);
        tbz.grow(1);

        // Loop over cells
        amrex::ParallelFor(tbx, tby, tbz,

            // Apply Boundary condition to Bx
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

#ifdef WARPX_DIM_3D
                // At the +y boundary (innermost guard cell)
                if ( apply_hi_y && ( j==domain_box.bigEnd(1)+1 ) ) {
                    Bx(i,j,k) = coef1_y * Bx(i,j,k) + coef2_y * Ez(i,j,k);
                }
                // At the -y boundary (innermost guard cell)
                if ( apply_lo_y && ( j==domain_box.smallEnd(1)-1 ) ) {
                    Bx(i,j,k) = coef1_y * Bx(i,j,k) - coef2_y * Ez(i,j+1,k);
                }
                // At the +z boundary (innermost guard cell)
                if ( apply_hi_z && ( k==domain_box.bigEnd(2)+1 ) ) {
                    Bx(i,j,k) = coef1_z * Bx(i,j,k) - coef2_z * Ey(i,j,k);
                }
                // At the -z boundary (innermost guard cell)
                if ( apply_lo_z && ( k==domain_box.smallEnd(2)-1 ) ) {
                    Bx(i,j,k) = coef1_z * Bx(i,j,k) + coef2_z * Ey(i,j,k+1);
                }
#elif WARPX_DIM_XZ
                // At the +z boundary (innermost guard cell)
                if ( apply_hi_z && ( j==domain_box.bigEnd(1)+1 ) ) {
                    Bx(i,j,k) = coef1_z * Bx(i,j,k) - coef2_z * Ey(i,j,k);
                }
                // At the -z boundary (innermost guard cell)
                if ( apply_lo_z && ( j==domain_box.smallEnd(1)-1 ) ) {
                    Bx(i,j,k) = coef1_z * Bx(i,j,k) + coef2_z * Ey(i,j+1,k);
                }
#elif WARPX_DIM_1D_Z
                // At the +z boundary (innermost guard cell)
                if ( apply_hi_z && ( i==domain_box.bigEnd(0)+1 ) ) {
                    Bx(i,j,k) = coef1_z * Bx(i,j,k) - coef2_z * Ey(i,j,k);
                }
                // At the -z boundary (innermost guard cell)
                if ( apply_lo_z && ( i==domain_box.smallEnd(0)-1 ) ) {
                    Bx(i,j,k) = coef1_z * Bx(i,j,k) + coef2_z * Ey(i+1,j,k);
                }
#endif
            },

            // Apply Boundary condition to By
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

#if (defined WARPX_DIM_3D || WARPX_DIM_XZ)
                // At the +x boundary (innermost guard cell)
                if ( apply_hi_x && ( i==domain_box.bigEnd(0)+1 ) ) {
                    By(i,j,k) = coef1_x * By(i,j,k) - coef2_x * Ez(i,j,k);
                }
                // At the -x boundary (innermost guard cell)
                if ( apply_lo_x && ( i==domain_box.smallEnd(0)-1 ) ) {
                    By(i,j,k) = coef1_x * By(i,j,k) + coef2_x * Ez(i+1,j,k);
                }
#endif
#ifdef WARPX_DIM_3D
                // At the +z boundary (innermost guard cell)
                if ( apply_hi_z && ( k==domain_box.bigEnd(2)+1 ) ) {
                    By(i,j,k) = coef1_z * By(i,j,k) + coef2_z * Ex(i,j,k);
                }
                // At the -z boundary (innermost guard cell)
                if ( apply_lo_z && ( k==domain_box.smallEnd(2)-1 ) ) {
                    By(i,j,k) = coef1_z * By(i,j,k) - coef2_z * Ex(i,j,k+1);
                }
#elif WARPX_DIM_XZ
                // At the +z boundary (innermost guard cell)
                if ( apply_hi_z && ( j==domain_box.bigEnd(1)+1 ) ) {
                    By(i,j,k) = coef1_z * By(i,j,k) + coef2_z * Ex(i,j,k);
                }
                // At the -z boundary (innermost guard cell)
                if ( apply_lo_z && ( j==domain_box.smallEnd(1)-1 ) ) {
                    By(i,j,k) = coef1_z * By(i,j,k) - coef2_z * Ex(i,j+1,k);
                }
#elif WARPX_DIM_1D_Z
                // At the +z boundary (innermost guard cell)
                if ( apply_hi_z && ( i==domain_box.bigEnd(0)+1 ) ) {
                    By(i,j,k) = coef1_z * By(i,j,k) + coef2_z * Ex(i,j,k);
                }
                // At the -z boundary (innermost guard cell)
                if ( apply_lo_z && ( i==domain_box.smallEnd(0)-1 ) ) {
                    By(i,j,k) = coef1_z * By(i,j,k) - coef2_z * Ex(i+1,j,k);
                }
#endif
            },

            // Apply Boundary condition to Bz
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

#if (defined WARPX_DIM_3D || WARPX_DIM_XZ)
                // At the +x boundary (innermost guard cell)
                if ( apply_hi_x && ( i==domain_box.bigEnd(0)+1 ) ) {
                    Bz(i,j,k) = coef1_x * Bz(i,j,k) + coef2_x * Ey(i,j,k);
                }
                // At the -x boundary (innermost guard cell)
                if ( apply_lo_x && ( i==domain_box.smallEnd(0)-1 ) ) {
                    Bz(i,j,k) = coef1_x * Bz(i,j,k) - coef2_x * Ey(i+1,j,k);
                }
#endif
#ifdef WARPX_DIM_3D
                // At the +y boundary (innermost guard cell)
                if ( apply_hi_y && ( j==domain_box.bigEnd(1)+1 ) ) {
                    Bz(i,j,k) = coef1_y * Bz(i,j,k) - coef2_y * Ex(i,j,k);
                }
                // At the -y boundary (innermost guard cell)
                if ( apply_lo_y && ( j==domain_box.smallEnd(1)-1 ) ) {
                    Bz(i,j,k) = coef1_y * Bz(i,j,k) + coef2_y * Ex(i,j+1,k);
                }
#elif WARPX_DIM_1D_Z
                ignore_unused(i,j,k);
#endif
            }
        );

    }
#endif // WARPX_DIM_RZ
}
