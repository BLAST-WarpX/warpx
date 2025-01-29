/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_NODAL_H_
#define WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_NODAL_H_

#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

#include <array>
#include <cmath>


/**
 * This struct contains only static functions to initialize the stencil coefficients
 * and to compute finite-difference derivatives for the Cartesian nodal algorithm.
 */
struct CartesianNodalAlgorithm {

    static void InitializeStencilCoefficients (
        std::array<amrex::Real,3>& cell_size,
        amrex::Vector<amrex::Real>& stencil_coefs_x,
        amrex::Vector<amrex::Real>& stencil_coefs_y,
        amrex::Vector<amrex::Real>& stencil_coefs_z ) {

        using namespace amrex;

        // Store the inverse cell size along each direction in the coefficients
        stencil_coefs_x.resize(1);
        stencil_coefs_x[0] = 1._rt/cell_size[0];
        stencil_coefs_y.resize(1);
        stencil_coefs_y[0] = 1._rt/cell_size[1];
        stencil_coefs_z.resize(1);
        stencil_coefs_z[0] = 1._rt/cell_size[2];
    }

    /**
     * Compute the maximum timestep, for which the scheme remains stable
     * (Courant-Friedrichs-Levy limit) */
    static amrex::Real ComputeMaxDt ( amrex::Real const * const dx ) {
        using namespace amrex::literals;
        amrex::Real const delta_t  = 1._rt / ( std::sqrt( AMREX_D_TERM(
                                           1._rt/(dx[0]*dx[0]),
                                         + 1._rt/(dx[1]*dx[1]),
                                         + 1._rt/(dx[2]*dx[2])
                                     ) ) * PhysConst::c );
        return delta_t;
    }

    /**
     * \brief Returns maximum number of guard cells required by the field-solve
     */
    static amrex::IntVect GetMaxGuardCell () {
        // The nodal solver requires one guard cell in each dimension
        return amrex::IntVect{AMREX_D_DECL(1,1,1)};
    }

    /**
     * Perform derivative along x
     * (For a solver on a staggered grid, `UpwardDx` and `DownwardDx` take into
     * account the staggering; but for `CartesianNodalAlgorithm`, they are equivalent) */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real UpwardDx (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_x, int const /*n_coefs_x*/,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
#if (defined WARPX_DIM_1D_Z)
        ignore_unused(i, j, k, coefs_x, ncomp, F);
        return 0._rt; // 1D Cartesian: derivative along x is 0
#else
        Real const inv_dx = coefs_x[0];
        return 0.5_rt*inv_dx*( F(i+1,j,k,ncomp) - F(i-1,j,k,ncomp) );
#endif
    }

    /**
     * Perform derivative along x
     * (For a solver on a staggered grid, `UpwardDx` and `DownwardDx` take into
     * account the staggering; but for `CartesianNodalAlgorithm`, they are equivalent) */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real DownwardDx (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_x, int const n_coefs_x,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
#if (defined WARPX_DIM_1D_Z)
        ignore_unused(i, j, k, coefs_x, n_coefs_x, ncomp, F);
        return 0._rt; // 1D Cartesian: derivative along x is 0
#else
        return UpwardDx( F, coefs_x, n_coefs_x, i, j, k ,ncomp);
        // For CartesianNodalAlgorithm, UpwardDx and DownwardDx are equivalent
#endif
    }

    /**
     * Perform derivative along y
     * (For a solver on a staggered grid, `UpwardDy` and `DownwardDy` take into
     * account the staggering; but for `CartesianNodalAlgorithm`, they are equivalent) */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real UpwardDy (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_y, int const /*n_coefs_y*/,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
#if defined WARPX_DIM_3D
        Real const inv_dy = coefs_y[0];
        return 0.5_rt*inv_dy*( F(i,j+1,k,ncomp) - F(i,j-1,k,ncomp) );
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_1D_Z)
        ignore_unused(i, j, k, coefs_y, ncomp, F);
        return 0._rt; // 1D and 2D Cartesian: derivative along y is 0
#endif
    }

    /**
     * Perform derivative along y
     * (For a solver on a staggered grid, `UpwardDy` and `DownwardDy` take into
     * account the staggering; but for `CartesianNodalAlgorithm`, they are equivalent) */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real DownwardDy (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_y, int const n_coefs_y,
        int const i, int const j, int const k, int const ncomp=0 ) {

        return UpwardDy( F, coefs_y, n_coefs_y, i, j, k ,ncomp);
        // For CartesianNodalAlgorithm, UpwardDy and DownwardDy are equivalent
    }

    /**
     * Perform derivative along z
     * (For a solver on a staggered grid, `UpwardDz` and `DownwardDz` take into
     * account the staggering; but for `CartesianNodalAlgorithm`, they are equivalent) */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real UpwardDz (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_z, int const /*n_coefs_z*/,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
        Real const inv_dz = coefs_z[0];
#if defined WARPX_DIM_3D
        return 0.5_rt*inv_dz*( F(i,j,k+1,ncomp) - F(i,j,k-1,ncomp) );
#elif (defined WARPX_DIM_XZ)
        return 0.5_rt*inv_dz*( F(i,j+1,k,ncomp) - F(i,j-1,k,ncomp) );
#elif (defined WARPX_DIM_1D_Z)
        return 0.5_rt*inv_dz*( F(i+1,j,k,ncomp) - F(i-1,j,k,ncomp) );
#endif
    }

    /**
     * Perform derivative along z
     * (For a solver on a staggered grid, `UpwardDz` and `DownwardDz` take into
     * account the staggering; but for `CartesianNodalAlgorithm`, they are equivalent) */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real DownwardDz (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_z, int const n_coefs_z,
        int const i, int const j, int const k, int const ncomp=0 ) {

        return UpwardDz( F, coefs_z, n_coefs_z, i, j, k ,ncomp);
        // For CartesianNodalAlgorithm, UpwardDz and DownwardDz are equivalent
    }

};

#endif // WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_NODAL_H_
