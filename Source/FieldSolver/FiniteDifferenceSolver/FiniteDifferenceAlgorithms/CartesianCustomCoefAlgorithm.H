/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_CUSTOM_COEF_H_
#define WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_CUSTOM_COEF_H_

#include "Utils/WarpXConst.H"
#include "Utils/WarpXUtil.H"
#include "ablastr/warn_manager/WarnManager.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>

#include <algorithm>
#include <array>

/**
 * This struct contains only static functions to initialize the stencil coefficients
 * and to compute finite-difference derivatives for a stencil with user-defined coefficents algorithm.
 */
struct CartesianCustomCoefAlgorithm {
    static void InitializeStencilCoefficients (
        std::array<amrex::Real, 3>& cell_size,
        amrex::Vector<amrex::Real>& stencil_coefs_x,
        amrex::Vector<amrex::Real>& stencil_coefs_y,
        amrex::Vector<amrex::Real>& stencil_coefs_z) {

        using namespace amrex;
        
        Real const inv_dx = 1._rt/cell_size[0];
        Real const inv_dy = 1._rt/cell_size[1];
        Real const inv_dz = 1._rt/cell_size[2];

        // Initialize to Yee single coefficient by default
        stencil_coefs_x.resize(1);
        stencil_coefs_x[0] = 1._rt * inv_dx;
        stencil_coefs_y.resize(1);
        stencil_coefs_y[0] = 1._rt * inv_dy;
        stencil_coefs_z.resize(1);
        stencil_coefs_z[0] = 1._rt * inv_dz;

        amrex::ParmParse pp_algo("algo");
        Vector<Real> upward_coefs_x, downward_coefs_x;
        Vector<Real> upward_coefs_y, downward_coefs_y;
        Vector<Real> upward_coefs_z, downward_coefs_z;

        // Check for custom coefficients and override defaults if found
        if (pp_algo.queryarr("custom_coef_upward_stencil_x", upward_coefs_x) && 
            pp_algo.queryarr("custom_coef_downward_stencil_x", downward_coefs_x)) {
            
            if (upward_coefs_x.size() != downward_coefs_x.size()) {
                amrex::Abort("Upward and downward stencils must have the same size");
            }

            int stencil_size = upward_coefs_x.size();
            stencil_coefs_x.resize(2 * stencil_size);
            
            // Upward coefficients in first half
            for (int i = 0; i < stencil_size; ++i) {
                stencil_coefs_x[i] = upward_coefs_x[i] * inv_dx;
            }
            // Downward coefficients in second half
            for (int i = 0; i < stencil_size; ++i) {
                stencil_coefs_x[i + stencil_size] = downward_coefs_x[i] * inv_dx;
            }
        }

        if (pp_algo.queryarr("custom_coef_upward_stencil_y", upward_coefs_y) &&
            pp_algo.queryarr("custom_coef_downward_stencil_y", downward_coefs_y)) {
            
            if (upward_coefs_y.size() != downward_coefs_y.size()) {
                amrex::Abort("Upward and downward stencils must have the same size");
            }

            int stencil_size = upward_coefs_y.size();
            stencil_coefs_y.resize(2 * stencil_size);
            
            for (int i = 0; i < stencil_size; ++i) {
                stencil_coefs_y[i] = upward_coefs_y[i] * inv_dy;
            }
            for (int i = 0; i < stencil_size; ++i) {
                stencil_coefs_y[i + stencil_size] = downward_coefs_y[i] * inv_dy;
            }
        }

        if (pp_algo.queryarr("custom_coef_upward_stencil_z", upward_coefs_z) &&
            pp_algo.queryarr("custom_coef_downward_stencil_z", downward_coefs_z)) {
            
            if (upward_coefs_z.size() != downward_coefs_z.size()) {
                amrex::Abort("Upward and downward stencils must have the same size");
            }

            int stencil_size = upward_coefs_z.size();
            stencil_coefs_z.resize(2 * stencil_size);
            
            for (int i = 0; i < stencil_size; ++i) {
                stencil_coefs_z[i] = upward_coefs_z[i] * inv_dz;
            }
            for (int i = 0; i < stencil_size; ++i) {
                stencil_coefs_z[i + stencil_size] = downward_coefs_z[i] * inv_dz;
            }
        }

        // Print coefficients for debugging
        amrex::Print() << "X stencil coefficients: ";
        for (auto coef : stencil_coefs_x) amrex::Print() << coef << " ";
        amrex::Print() << "\n";
    }

    static amrex::Real ComputeMaxDt(amrex::Real const * const dx) {
        using namespace amrex::literals;
        amrex::Real const delta_t = 1._rt / (std::sqrt(AMREX_D_TERM(
            1._rt / (dx[0]*dx[0]),
            + 1._rt / (dx[1]*dx[1]),
            + 1._rt / (dx[2]*dx[2])
        )) * PhysConst::c);
        return delta_t;
    }

    static amrex::IntVect GetMaxGuardCell() {
        return amrex::IntVect{AMREX_D_DECL(1,1,1)};
    }

    /**
     * Perform derivative along x on a cell-centered grid, from a nodal field `F` */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real UpwardDx(
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_x, int const n_coefs_x,
        int const i, int const j, int const k, int const ncomp=0) {

        using namespace amrex;
#if defined WARPX_DIM_3D
        if (n_coefs_x == 1) {
            return coefs_x[0] * (F(i+1,j,k,ncomp) - F(i,j,k,ncomp));
        }
        
        amrex::Real derivative = 0;
        int stencil_size = n_coefs_x / 2;
        for (int l = 0; l < stencil_size; l++) {
            derivative += coefs_x[l] * (F(i+1,j,k,ncomp) - F(i,j,k,ncomp));
        }
        return derivative;
#elif (defined WARPX_DIM_1D_Z)
        amrex::ignore_unused(F, coefs_x, n_coefs_x, i, j, k, ncomp);
        return 0._rt;
#endif
    }

    /**
     * Perform derivative along x on a nodal grid, from a cell-centered field `F` */
    template<typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real DownwardDx(
        T_Field const& F,
        amrex::Real const * const coefs_x, int const n_coefs_x,
        int const i, int const j, int const k, int const ncomp=0) {

        using namespace amrex;
#if defined WARPX_DIM_3D
        if (n_coefs_x == 1) {
            return coefs_x[0] * (F(i,j,k,ncomp) - F(i-1,j,k,ncomp));
        }
        
        amrex::Real derivative = 0;
        int stencil_size = n_coefs_x / 2;
        for (int l = 0; l < stencil_size; l++) {
            derivative += coefs_x[l + stencil_size] * (F(i,j,k,ncomp) - F(i-1,j,k,ncomp));
        }
        return derivative;
#elif (defined WARPX_DIM_1D_Z)
        amrex::ignore_unused(F, coefs_x, n_coefs_x, i, j, k, ncomp);
        return 0._rt;
#endif
    }

    /**
     * Perform derivative along y on a cell-centered grid, from a nodal field `F` */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real UpwardDy(
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_y, int const n_coefs_y,
        int const i, int const j, int const k, int const ncomp=0) {

        using namespace amrex;
#if defined WARPX_DIM_3D
        if (n_coefs_y == 1) {
            return coefs_y[0] * (F(i,j+1,k,ncomp) - F(i,j,k,ncomp));
        }
        
        amrex::Real derivative = 0;
        int stencil_size = n_coefs_y / 2;
        for (int l = 0; l < stencil_size; l++) {
            derivative += coefs_y[l] * (F(i,j+1,k,ncomp) - F(i,j,k,ncomp));
        }
        return derivative;
#elif (defined WARPX_DIM_XZ || WARPX_DIM_1D_Z)
        amrex::ignore_unused(F, coefs_y, n_coefs_y, i, j, k, ncomp);
        return 0._rt;
#endif
    }

    /**
     * Perform derivative along y on a nodal grid, from a cell-centered field `F` */
    template<typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real DownwardDy(
        T_Field const& F,
        amrex::Real const * const coefs_y, int const n_coefs_y,
        int const i, int const j, int const k, int const ncomp=0) {

        using namespace amrex;
#if defined WARPX_DIM_3D
        if (n_coefs_y == 1) {
            return coefs_y[0] * (F(i,j,k,ncomp) - F(i,j-1,k,ncomp));
        }
        
        amrex::Real derivative = 0;
        int stencil_size = n_coefs_y / 2;
        for (int l = 0; l < stencil_size; l++) {
            derivative += coefs_y[l + stencil_size] * (F(i,j,k,ncomp) - F(i,j-1,k,ncomp));
        }
        return derivative;
#elif (defined WARPX_DIM_XZ || WARPX_DIM_1D_Z)
        amrex::ignore_unused(F, coefs_y, n_coefs_y, i, j, k, ncomp);
        return 0._rt;
#endif
    }

    /**
     * Perform derivative along z on a cell-centered grid, from a nodal field `F` */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real UpwardDz(
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_z, int const n_coefs_z,
        int const i, int const j, int const k, int const ncomp=0) {

        using namespace amrex;
#if defined WARPX_DIM_3D
        if (n_coefs_z == 1) {
            return coefs_z[0] * (F(i,j,k+1,ncomp) - F(i,j,k,ncomp));
        }
        
        amrex::Real derivative = 0;
        int stencil_size = n_coefs_z / 2;
        for (int l = 0; l < stencil_size; l++) {
            derivative += coefs_z[l] * (F(i,j,k+1,ncomp) - F(i,j,k,ncomp));
        }
        return derivative;
#elif (defined WARPX_DIM_XZ)
        if (n_coefs_z == 1) {
            return coefs_z[0] * (F(i,j+1,k,ncomp) - F(i,j,k,ncomp));
        }
        amrex::Real derivative = 0;
        int stencil_size = n_coefs_z / 2;
        for (int l = 0; l < stencil_size; l++) {
            derivative += coefs_z[l] * (F(i,j+1,k,ncomp) - F(i,j,k,ncomp));
        }
        return derivative;
#elif (defined WARPX_DIM_1D_Z)
        if (n_coefs_z == 1) {
            return coefs_z[0] * (F(i+1,j,k,ncomp) - F(i,j,k,ncomp));
        }
        amrex::Real derivative = 0;
        int stencil_size = n_coefs_z / 2;
        for (int l = 0; l < stencil_size; l++) {
            derivative += coefs_z[l] * (F(i+1,k,j,ncomp) - F(i,j,k,ncomp));
        }
        return derivative;
#endif
    }

    /**
     * Perform derivative along z on a nodal grid, from a cell-centered field `F` */
    template<typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real DownwardDz(
        T_Field const& F,
        amrex::Real const * const coefs_z, int const n_coefs_z,
        int const i, int const j, int const k, int const ncomp=0) {

        using namespace amrex;
#if defined WARPX_DIM_3D
        if (n_coefs_z == 1) {
            return coefs_z[0] * (F(i,j,k,ncomp) - F(i,j,k-1,ncomp));
        }

        amrex::Real derivative = 0;
        int stencil_size = n_coefs_z / 2;
        for (int l = 0; l < stencil_size; l++) {
            derivative += coefs_z[l + stencil_size] * (F(i,j,k,ncomp) - F(i,j,k-1,ncomp));
        }
        return derivative;
#elif (defined WARPX_DIM_XZ)
        if (n_coefs_z == 1) {
            return coefs_z[0] * (F(i,j,k,ncomp) - F(i,j-1,k,ncomp));
        }
        amrex::Real derivative = 0;
        int stencil_size = n_coefs_z / 2;
        for (int l = 0; l < stencil_size; l++) {
            derivative += coefs_z[l + stencil_size] * (F(i,j,k,ncomp) - F(i,j-1,k,ncomp));
        }
        return derivative;
#elif (defined WARPX_DIM_1D_Z)
        if (n_coefs_z == 1) {
            return coefs_z[0] * (F(i,j,k,ncomp) - F(i-1,j,k,ncomp));
        }
        amrex::Real derivative = 0;
        int stencil_size = n_coefs_z / 2;
        for (int l = 0; l < stencil_size; l++) {
            derivative += coefs_z[l + stencil_size] * (F(i,j,k,ncomp) - F(i-1,j,k,ncomp));
        }
        return derivative;
#endif
    }
};

#endif // WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_CUSTOM_COEF_H_
