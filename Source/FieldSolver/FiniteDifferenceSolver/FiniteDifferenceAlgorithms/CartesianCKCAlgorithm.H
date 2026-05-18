/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_CKC_H_
#define WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_CKC_H_

#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <array>


/**
 * This struct contains only static functions to initialize the stencil coefficients
 * and to compute finite-difference derivatives for the Cartesian CKC algorithm.
 */
struct CartesianCKCAlgorithm {

    static void InitializeStencilCoefficients (
        std::array<amrex::Real, 3>& cell_size,
        amrex::Vector<amrex::Real>& stencil_coefs_x,
        amrex::Vector<amrex::Real>& stencil_coefs_y,
        amrex::Vector<amrex::Real>& stencil_coefs_z ) {

        using namespace amrex;

        // Compute Cole-Karkkainen-Cowan coefficients according
        // to Cowan - PRST-AB 16, 041303 (2013)
        Real const inv_dx = 1._rt/cell_size[0];
        Real const inv_dy = 1._rt/cell_size[1];
        Real const inv_dz = 1._rt/cell_size[2];
#if defined WARPX_DIM_3D
        Real const delta = std::max( { inv_dx,inv_dy,inv_dz } );
        Real const rx = (inv_dx/delta)*(inv_dx/delta);
        Real const ry = (inv_dy/delta)*(inv_dy/delta);
        Real const rz = (inv_dz/delta)*(inv_dz/delta);
        Real const beta = 0.125_rt*(1._rt - rx*ry*rz/(ry*rz + rz*rx + rx*ry));
        Real const betaxy = ry*beta*inv_dx;
        Real const betaxz = rz*beta*inv_dx;
        Real const betayx = rx*beta*inv_dy;
        Real const betayz = rz*beta*inv_dy;
        Real const betazx = rx*beta*inv_dz;
        Real const betazy = ry*beta*inv_dz;
        Real const inv_r_fac = (1._rt/(ry*rz + rz*rx + rx*ry));
        Real const gammax = ry*rz*(0.0625_rt - 0.125_rt*ry*rz*inv_r_fac);
        Real const gammay = rx*rz*(0.0625_rt - 0.125_rt*rx*rz*inv_r_fac);
        Real const gammaz = rx*ry*(0.0625_rt - 0.125_rt*rx*ry*inv_r_fac);
        Real const alphax = (1._rt - 2._rt*ry*beta - 2._rt*rz*beta - 4._rt*gammax)*inv_dx;
        Real const alphay = (1._rt - 2._rt*rx*beta - 2._rt*rz*beta - 4._rt*gammay)*inv_dy;
        Real const alphaz = (1._rt - 2._rt*rx*beta - 2._rt*ry*beta - 4._rt*gammaz)*inv_dz;
#elif defined WARPX_DIM_XZ
        Real const delta = std::max(inv_dx,inv_dz);
        Real const rx = (inv_dx/delta)*(inv_dx/delta);
        Real const rz = (inv_dz/delta)*(inv_dz/delta);
        constexpr Real beta = 0.125_rt;
        Real const betaxz = beta*rz*inv_dx;
        Real const betazx = beta*rx*inv_dz;
        Real const alphax = (1._rt - 2._rt*rz*beta)*inv_dx;
        Real const alphaz = (1._rt - 2._rt*rx*beta)*inv_dz;
        // Other coefficients are 0 in 2D Cartesian
        // (and will actually not be used in the stencil)
        constexpr Real gammax=0._rt, gammay=0._rt, gammaz=0._rt;
        constexpr Real betaxy=0._rt, betazy=0._rt, betayx=0._rt, betayz=0._rt;
        constexpr Real alphay=0._rt;
#elif defined WARPX_DIM_1D_Z
        Real const alphaz = inv_dz;
        // Other coefficients are 0 in 1D Cartesian
        // (and will actually not be used in the stencil)
        constexpr Real gammax=0._rt, gammay=0._rt, gammaz=0._rt;
        constexpr Real betaxy=0._rt, betazy=0._rt, betayx=0._rt, betayz=0._rt, betaxz=0._rt, betazx=0._rt;
        constexpr Real alphay=0._rt, alphax=0._rt;
#endif

        // Store the coefficients in array `stencil_coefs`, in prescribed order
        stencil_coefs_x.resize(6);
        stencil_coefs_x[0] = inv_dx;
        stencil_coefs_x[1] = alphax;
        stencil_coefs_x[2] = betaxy;
        stencil_coefs_x[3] = betaxz;
        stencil_coefs_x[4] = gammax*inv_dx;
        stencil_coefs_y.resize(6);
        stencil_coefs_y[0] = inv_dy;
        stencil_coefs_y[1] = alphay;
        stencil_coefs_y[2] = betayz;
        stencil_coefs_y[3] = betayx;
        stencil_coefs_y[4] = gammay*inv_dy;
        stencil_coefs_z.resize(6);
        stencil_coefs_z[0] = inv_dz;
        stencil_coefs_z[1] = alphaz;
        stencil_coefs_z[2] = betazx;
        stencil_coefs_z[3] = betazy;
        stencil_coefs_z[4] = gammaz*inv_dz;
    }

    /**
     * Compute the maximum timestep, for which the scheme remains stable
     * (Courant-Friedrichs-Levy limit) */
    static amrex::Real ComputeMaxDt ( amrex::Real const * const dx ) {
#if (defined WARPX_DIM_1D_Z)
            amrex::Real const delta_t = dx[0]/PhysConst::c;
#elif (defined WARPX_DIM_XZ)
            // - In Cartesian 2D geometry: determined by the minimum cell size in all direction
            amrex::Real const delta_t = std::min( dx[0], dx[1] )/PhysConst::c;
#else
            // - In Cartesian 3D geometry: determined by the minimum cell size in all direction
            amrex::Real const delta_t = std::min( dx[0], std::min( dx[1], dx[2] ) ) / PhysConst::c;
#endif
        return delta_t;
    }

    /**
     * \brief Returns maximum number of guard cells required by the field-solve
     */
    static amrex::IntVect GetMaxGuardCell () {
        // The ckc solver requires one guard cell in each dimension
        return amrex::IntVect{AMREX_D_DECL(1,1,1)};
    }

    /**
     * Perform derivative along x on a cell-centered grid, from a nodal field `F` */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real UpwardDx (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_x, int const /*n_coefs_x*/,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
#if (defined WARPX_DIM_3D || WARPX_DIM_XZ)
        amrex::Real const alphax = coefs_x[1];
        amrex::Real const betaxz = coefs_x[3];
#endif
#if defined WARPX_DIM_3D
        amrex::Real const betaxy = coefs_x[2];
        amrex::Real const gammax = coefs_x[4];
#endif

#if defined WARPX_DIM_3D
        return alphax * (F(i+1,j  ,k  ,ncomp) - F(i,  j,  k  ,ncomp))
             + betaxy * (F(i+1,j+1,k  ,ncomp) - F(i  ,j+1,k  ,ncomp)
                      +  F(i+1,j-1,k  ,ncomp) - F(i  ,j-1,k  ,ncomp))
             + betaxz * (F(i+1,j  ,k+1,ncomp) - F(i  ,j  ,k+1,ncomp)
                      +  F(i+1,j  ,k-1,ncomp) - F(i  ,j  ,k-1,ncomp))
             + gammax * (F(i+1,j+1,k+1,ncomp) - F(i  ,j+1,k+1,ncomp)
                      +  F(i+1,j-1,k+1,ncomp) - F(i  ,j-1,k+1,ncomp)
                      +  F(i+1,j+1,k-1,ncomp) - F(i  ,j+1,k-1,ncomp)
                      +  F(i+1,j-1,k-1,ncomp) - F(i  ,j-1,k-1,ncomp));
#elif (defined WARPX_DIM_XZ)
        return alphax * (F(i+1,j  ,k  ,ncomp) - F(i,  j,  k  ,ncomp))
             + betaxz * (F(i+1,j+1,k  ,ncomp) - F(i  ,j+1,k  ,ncomp)
                      +  F(i+1,j-1,k  ,ncomp) - F(i  ,j-1,k  ,ncomp));
#elif (defined WARPX_DIM_1D_Z)
        amrex::ignore_unused(F, coefs_x, i, j, k, ncomp);
        return 0._rt; // 1D Cartesian: derivative along x is 0
#endif
    }

    /**
     * Perform derivative along x on a nodal grid, from a cell-centered field `F` */
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real DownwardDx (
        T_Field const& F,
        amrex::Real const * const coefs_x, int const /*n_coefs_x*/,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
#if (defined WARPX_DIM_1D_Z)
        amrex::ignore_unused(F, coefs_x, i, j, k, ncomp);
        return 0._rt; // 1D Cartesian: derivative along x is 0
#else
        amrex::Real const inv_dx = coefs_x[0];
        return inv_dx*( F(i,j,k,ncomp) - F(i-1,j,k,ncomp) );
#endif
    }

    /**
     * Perform derivative along y on a cell-centered grid, from a nodal field `F` */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real UpwardDy (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_y, int const n_coefs_y,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
#if defined WARPX_DIM_3D
        Real const alphay = coefs_y[1];
        Real const betayz = coefs_y[2];
        Real const betayx = coefs_y[3];
        Real const gammay = coefs_y[4];
        amrex::ignore_unused(n_coefs_y);
        return alphay * (F(i  ,j+1,k  ,ncomp) - F(i  ,j  ,k  ,ncomp))
             + betayx * (F(i+1,j+1,k  ,ncomp) - F(i+1,j  ,k  ,ncomp)
                      +  F(i-1,j+1,k  ,ncomp) - F(i-1,j  ,k  ,ncomp))
             + betayz * (F(i  ,j+1,k+1,ncomp) - F(i  ,j  ,k+1,ncomp)
                      +  F(i  ,j+1,k-1,ncomp) - F(i  ,j  ,k-1,ncomp))
             + gammay * (F(i+1,j+1,k+1,ncomp) - F(i+1,j  ,k+1,ncomp)
                      +  F(i-1,j+1,k+1,ncomp) - F(i-1,j  ,k+1,ncomp)
                      +  F(i+1,j+1,k-1,ncomp) - F(i+1,j  ,k-1,ncomp)
                       +  F(i-1,j+1,k-1,ncomp) - F(i-1,j  ,k-1,ncomp));
#elif (defined WARPX_DIM_XZ || WARPX_DIM_1D_Z)
        amrex::ignore_unused(F, coefs_y, n_coefs_y,
            i, j, k, ncomp);
        return 0._rt; // 1D and 2D Cartesian: derivative along y is 0
#else
        amrex::ignore_unused(F, coefs_y, n_coefs_y,
            i, j, k, ncomp);
#endif
    }

    /**
     * Perform derivative along y on a nodal grid, from a cell-centered field `F` */
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real DownwardDy (
        T_Field const& F,
        amrex::Real const * const coefs_y, int const n_coefs_y,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
#if defined WARPX_DIM_3D
        amrex::Real const inv_dy = coefs_y[0];
        amrex::ignore_unused(n_coefs_y);
        return inv_dy*( F(i,j,k,ncomp) - F(i,j-1,k,ncomp) );
#elif (defined WARPX_DIM_XZ || WARPX_DIM_1D_Z)
        amrex::ignore_unused(F, coefs_y, n_coefs_y,
            i, j, k, ncomp);
        return 0._rt; // 2D Cartesian: derivative along y is 0
#else
        amrex::ignore_unused(F, coefs_y, n_coefs_y,
            i, j, k, ncomp);
#endif
    }

    /**
     * Perform derivative along z on a cell-centered grid, from a nodal field `F` */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real UpwardDz (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_z, int const n_coefs_z,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;

        amrex::ignore_unused(n_coefs_z);

        Real const alphaz = coefs_z[1];
#if (defined WARPX_DIM_3D || WARPX_DIM_XZ)
        Real const betazx = coefs_z[2];
#endif
#if defined WARPX_DIM_3D
        Real const betazy = coefs_z[3];
        Real const gammaz = coefs_z[4];
#endif
#if defined WARPX_DIM_3D
        return alphaz * (F(i  ,j  ,k+1,ncomp) - F(i  ,j  ,k  ,ncomp))
             + betazx * (F(i+1,j  ,k+1,ncomp) - F(i+1,j  ,k  ,ncomp)
                      +  F(i-1,j  ,k+1,ncomp) - F(i-1,j  ,k  ,ncomp))
             + betazy * (F(i  ,j+1,k+1,ncomp) - F(i  ,j+1,k  ,ncomp)
                      +  F(i  ,j-1,k+1,ncomp) - F(i  ,j-1,k  ,ncomp))
             + gammaz * (F(i+1,j+1,k+1,ncomp) - F(i+1,j+1,k  ,ncomp)
                      +  F(i-1,j+1,k+1,ncomp) - F(i-1,j+1,k  ,ncomp)
                      +  F(i+1,j-1,k+1,ncomp) - F(i+1,j-1,k  ,ncomp)
                      +  F(i-1,j-1,k+1,ncomp) - F(i-1,j-1,k  ,ncomp));
#elif (defined WARPX_DIM_XZ)
        return alphaz * (F(i  ,j+1,k  ,ncomp) - F(i  ,j  ,k  ,ncomp))
             + betazx * (F(i+1,j+1,k  ,ncomp) - F(i+1,j  ,k  ,ncomp)
                      +  F(i-1,j+1,k  ,ncomp) - F(i-1,j  ,k  ,ncomp));
#elif (defined WARPX_DIM_1D_Z)
        return alphaz * (F(i+1  ,j,k  ,ncomp) - F(i  ,j  ,k  ,ncomp));
#endif
    }

    /**
     * Perform derivative along z on a nodal grid, from a cell-centered field `F` */
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real DownwardDz (
        T_Field const& F,
        amrex::Real const * const coefs_z, int const /*n_coefs_z*/,
        int const i, int const j, int const k, int const ncomp=0) {

        amrex::Real const inv_dz = coefs_z[0];
#if defined WARPX_DIM_3D
        return inv_dz*( F(i,j,k,ncomp) - F(i,j,k-1,ncomp) );
#elif (defined WARPX_DIM_XZ)
        return inv_dz*( F(i,j,k,ncomp) - F(i,j-1,k,ncomp) );
#elif (defined WARPX_DIM_1D_Z)
        return inv_dz*( F(i,j,k,ncomp) - F(i-1,j,k,ncomp) );
#endif
    }

};

#endif // WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_CKC_H_
