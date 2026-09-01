/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_YEE_H_
#define WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_YEE_H_

#include "FieldAccessorFunctors.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXDim.H"
#include "Utils/WarpXDimIndexing.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

#include <array>
#include <cmath>

/**
 * This struct contains only static functions to initialize the stencil coefficients
 * and to compute finite-difference derivatives for the Cartesian Yee algorithm.
 *
 * \tparam D dimensionality of the simulation
 */
template <warpx::Dim D>
struct CartesianYeeAlgorithmND {

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
     * (Courant-Friedrichs-Levy limit)
     *
     * The sum extends over the axes present on the grid only, so that the
     * unit cell sizes of collapsed dimensions in unified
     * (runtime-dimensionality) builds do not perturb the timestep.
     */
    static amrex::Real ComputeMaxDt ( amrex::Real const * const dx ) {
        using namespace amrex::literals;
        amrex::Real inv_dx2_sum = 0._rt;
        if constexpr (warpx::has_x<D>) {
            inv_dx2_sum += 1._rt / (dx[warpx::IdxMap<D>::x]*dx[warpx::IdxMap<D>::x]);
        }
        if constexpr (warpx::has_y<D>) {
            inv_dx2_sum += 1._rt / (dx[warpx::IdxMap<D>::y]*dx[warpx::IdxMap<D>::y]);
        }
        if constexpr (warpx::has_z<D>) {
            inv_dx2_sum += 1._rt / (dx[warpx::IdxMap<D>::z]*dx[warpx::IdxMap<D>::z]);
        }
        amrex::Real const delta_t = 1._rt / ( std::sqrt( inv_dx2_sum ) * PhysConst::c );
        return delta_t;
    }

    /**
     * \brief Returns maximum number of guard cells required by the field-solve
     */
    static amrex::IntVect GetMaxGuardCell () {
        // The yee solver requires one guard cell along each axis present on the grid
        amrex::IntVect ng(0);
        if constexpr (warpx::has_x<D>) { ng[warpx::IdxMap<D>::x] = 1; }
        if constexpr (warpx::has_y<D>) { ng[warpx::IdxMap<D>::y] = 1; }
        if constexpr (warpx::has_z<D>) { ng[warpx::IdxMap<D>::z] = 1; }
        return ng;
    }

    /**
     * Perform an upward (cell-centered, from a nodal field `F`) first
     * derivative along the array slot `slot` */
    template <int slot, typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real UpwardD (
        T_Field const& F,
        amrex::Real const * const coefs,
        int const i, int const j, int const k, int const ncomp ) {

        using namespace amrex::literals;
        static_assert(0 <= slot && slot <= 2);
        amrex::Real const inv_d = coefs[0];
        if constexpr (slot == 0) {
            return inv_d*( F(i+1,j,k,ncomp) - F(i,j,k,ncomp) );
        } else if constexpr (slot == 1) {
            return inv_d*( F(i,j+1,k,ncomp) - F(i,j,k,ncomp) );
        } else {
            return inv_d*( F(i,j,k+1,ncomp) - F(i,j,k,ncomp) );
        }
    }

    /**
     * Perform a downward (nodal, from a cell-centered field `F`) first
     * derivative along the array slot `slot` */
    template <int slot, typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real DownwardD (
        T_Field const& F,
        amrex::Real const * const coefs,
        int const i, int const j, int const k, int const ncomp ) {

        using namespace amrex::literals;
        static_assert(0 <= slot && slot <= 2);
        amrex::Real const inv_d = coefs[0];
        if constexpr (slot == 0) {
            return inv_d*( F(i,j,k,ncomp) - F(i-1,j,k,ncomp) );
        } else if constexpr (slot == 1) {
            return inv_d*( F(i,j,k,ncomp) - F(i,j-1,k,ncomp) );
        } else {
            return inv_d*( F(i,j,k,ncomp) - F(i,j,k-1,ncomp) );
        }
    }

    /**
     * Perform a centered second derivative along the array slot `slot` */
    template <int slot, typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real DD (
        T_Field const& F,
        amrex::Real const * const coefs,
        int const i, int const j, int const k, int const ncomp ) {

        using namespace amrex::literals;
        static_assert(0 <= slot && slot <= 2);
        amrex::Real const inv_d2 = coefs[0]*coefs[0];
        if constexpr (slot == 0) {
            return inv_d2*( F(i-1,j,k,ncomp) - 2._rt*F(i,j,k,ncomp) + F(i+1,j,k,ncomp) );
        } else if constexpr (slot == 1) {
            return inv_d2*( F(i,j-1,k,ncomp) - 2._rt*F(i,j,k,ncomp) + F(i,j+1,k,ncomp) );
        } else {
            return inv_d2*( F(i,j,k-1,ncomp) - 2._rt*F(i,j,k,ncomp) + F(i,j,k+1,ncomp) );
        }
    }

    /**
     * Perform derivative along x on a cell-centered grid, from a nodal field `F`*/
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real UpwardDx (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_x, int const n_coefs_x,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
        amrex::ignore_unused(n_coefs_x);
        if constexpr (warpx::has_x<D>) {
            return UpwardD<warpx::IdxMap<D>::x>(F, coefs_x, i, j, k, ncomp);
        } else {
            amrex::ignore_unused(F, coefs_x, i, j, k, ncomp);
            return 0._rt; // 1D Cartesian: derivative along x is 0
        }
    }

    /**
     * Perform derivative along x on a nodal grid, from a cell-centered field `F`*/
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real DownwardDx (
        T_Field const& F,
        amrex::Real const * const coefs_x, int const n_coefs_x,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
        amrex::ignore_unused(n_coefs_x);
        if constexpr (warpx::has_x<D>) {
            return DownwardD<warpx::IdxMap<D>::x>(F, coefs_x, i, j, k, ncomp);
        } else {
            amrex::ignore_unused(F, coefs_x, i, j, k, ncomp);
            return 0._rt; // 1D Cartesian: derivative along x is 0
        }
    }

    /**
     * Perform second derivative along x on a cell-centered grid, from a cell-centered field `F`*/
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real Dxx (
        T_Field const& F,
        amrex::Real const * const coefs_x, int const n_coefs_x,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
        amrex::ignore_unused(n_coefs_x);
        if constexpr (warpx::has_x<D>) {
            return DD<warpx::IdxMap<D>::x>(F, coefs_x, i, j, k, ncomp);
        } else {
            amrex::ignore_unused(F, coefs_x, i, j, k, ncomp);
            return 0._rt; // 1D Cartesian: derivative along x is 0
        }
    }

    /**
     * Perform derivative along y on a cell-centered grid, from a nodal field `F`*/
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real UpwardDy (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_y, int const n_coefs_y,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
        amrex::ignore_unused(n_coefs_y);
        if constexpr (warpx::has_y<D>) {
            return UpwardD<warpx::IdxMap<D>::y>(F, coefs_y, i, j, k, ncomp);
        } else {
            amrex::ignore_unused(F, coefs_y, i, j, k, ncomp);
            return 0._rt; // 1D and 2D Cartesian: derivative along y is 0
        }
    }

    /**
     * Perform derivative along y on a nodal grid, from a cell-centered field `F`*/
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real DownwardDy (
        T_Field const& F,
        amrex::Real const * const coefs_y, int const n_coefs_y,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
        amrex::ignore_unused(n_coefs_y);
        if constexpr (warpx::has_y<D>) {
            return DownwardD<warpx::IdxMap<D>::y>(F, coefs_y, i, j, k, ncomp);
        } else {
            amrex::ignore_unused(F, coefs_y, i, j, k, ncomp);
            return 0._rt; // 1D and 2D Cartesian: derivative along y is 0
        }
    }

    /**
     * Perform second derivative along y on a cell-centered grid, from a cell-centered field `F`*/
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real Dyy (
        T_Field const& F,
        amrex::Real const * const coefs_y, int const n_coefs_y,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
        amrex::ignore_unused(n_coefs_y);
        if constexpr (warpx::has_y<D>) {
            return DD<warpx::IdxMap<D>::y>(F, coefs_y, i, j, k, ncomp);
        } else {
            amrex::ignore_unused(F, coefs_y, i, j, k, ncomp);
            return 0._rt; // 1D and 2D Cartesian: derivative along y is 0
        }
    }

    /**
     * Perform derivative along z on a cell-centered grid, from a nodal field `F`*/
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real UpwardDz (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_z, int const n_coefs_z,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
        amrex::ignore_unused(n_coefs_z);
        if constexpr (warpx::has_z<D>) {
            return UpwardD<warpx::IdxMap<D>::z>(F, coefs_z, i, j, k, ncomp);
        } else {
            amrex::ignore_unused(F, coefs_z, i, j, k, ncomp);
            return 0._rt; // 1D radial: derivative along z is 0
        }
    }

    /**
     * Perform derivative along z on a nodal grid, from a cell-centered field `F`*/
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real DownwardDz (
        T_Field const& F,
        amrex::Real const * const coefs_z, int const n_coefs_z,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
        amrex::ignore_unused(n_coefs_z);
        if constexpr (warpx::has_z<D>) {
            return DownwardD<warpx::IdxMap<D>::z>(F, coefs_z, i, j, k, ncomp);
        } else {
            amrex::ignore_unused(F, coefs_z, i, j, k, ncomp);
            return 0._rt; // 1D radial: derivative along z is 0
        }
    }

    /**
     * Perform second derivative along z on a cell-centered field `F`*/
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real Dzz (
        T_Field const& F,
        amrex::Real const * const coefs_z, int const n_coefs_z,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
        amrex::ignore_unused(n_coefs_z);
        if constexpr (warpx::has_z<D>) {
            return DD<warpx::IdxMap<D>::z>(F, coefs_z, i, j, k, ncomp);
        } else {
            amrex::ignore_unused(F, coefs_z, i, j, k, ncomp);
            return 0._rt; // 1D radial: derivative along z is 0
        }
    }

    /**
     * Perform fourth derivative along x on a cell-centered grid, from a cell-centered field `F`*/
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real Dxxxx (
        T_Field const& F,
        amrex::Real const * const coefs_x, int const /*n_coefs_x*/,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
#if (defined WARPX_DIM_1D_Z)
        amrex::ignore_unused(F, coefs_x, i, j, k, ncomp);
        return 0._rt; // 1D Cartesian: derivative along x is 0
#else
        amrex::Real const inv_dx4 = amrex::Math::powi<4>(coefs_x[0]);
        return inv_dx4*( F(i-2,j,k,ncomp) - 4._rt*F(i-1,j,k,ncomp) + 6._rt*F(i,j,k,ncomp)
                          - 4._rt*F(i+1,j,k,ncomp) + F(i+2,j,k,ncomp) );
#endif
    }

    /**
     * Perform fourth derivative along y on a cell-centered grid, from a cell-centered field `F`*/
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real Dyyyy (
        T_Field const& F,
        amrex::Real const * const coefs_y, int const /*n_coefs_y*/,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
#if defined WARPX_DIM_3D
        amrex::Real const inv_dy4 = amrex::Math::powi<4>(coefs_y[0]);
        return inv_dy4*( F(i,j-2,k,ncomp) - 4._rt*F(i,j-1,k,ncomp) + 6._rt*F(i,j,k,ncomp)
                          - 4._rt*F(i,j+1,k,ncomp) + F(i,j+2,k,ncomp) );
#else
        amrex::ignore_unused(F, coefs_y, i, j, k, ncomp);
        return 0._rt; // 1D and 2D Cartesian: derivative along y is 0
#endif
    }

    /**
     * Perform fourth derivative along z on a cell-centered field `F`*/
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real Dzzzz (
        T_Field const& F,
        amrex::Real const * const coefs_z, int const /*n_coefs_z*/,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
        amrex::Real const inv_dz4 = amrex::Math::powi<4>(coefs_z[0]);
#if defined WARPX_DIM_3D
        return inv_dz4*( F(i,j,k-2,ncomp) - 4._rt*F(i,j,k-1,ncomp) + 6._rt*F(i,j,k,ncomp)
                          - 4._rt*F(i,j,k+1,ncomp) + F(i,j,k+2,ncomp) );
#elif (defined WARPX_DIM_XZ)
        return inv_dz4*( F(i,j-2,k,ncomp) - 4._rt*F(i,j-1,k,ncomp) + 6._rt*F(i,j,k,ncomp)
                          - 4._rt*F(i,j+1,k,ncomp) + F(i,j+2,k,ncomp) );
#elif (defined WARPX_DIM_1D_Z)
        return inv_dz4*( F(i-2,j,k,ncomp) - 4._rt*F(i-1,j,k,ncomp) + 6._rt*F(i,j,k,ncomp)
                          - 4._rt*F(i+1,j,k,ncomp) + F(i+2,j,k,ncomp) );
#endif
    }

    /**
     * Perform mixed fourth derivative d^4/(dx^2 dy^2) on a cell-centered grid, from a
     * cell-centered field `F` (only nonzero in 3D)*/
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real Dxxyy (
        T_Field const& F,
        amrex::Real const * const coefs_x, int const /*n_coefs_x*/,
        amrex::Real const * const coefs_y, int const /*n_coefs_y*/,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
#if defined WARPX_DIM_3D
        amrex::Real const inv_dx2 = coefs_x[0]*coefs_x[0];
        amrex::Real const inv_dy2 = coefs_y[0]*coefs_y[0];
        amrex::Real result = 0._rt;
        for (int di = -1; di <= 1; ++di) {
            amrex::Real const cx = (di == 0) ? -2._rt : 1._rt;
            for (int dj = -1; dj <= 1; ++dj) {
                amrex::Real const cy = (dj == 0) ? -2._rt : 1._rt;
                result += cx*cy*F(i+di,j+dj,k,ncomp);
            }
        }
        return inv_dx2*inv_dy2*result;
#else
        amrex::ignore_unused(F, coefs_x, coefs_y, i, j, k, ncomp);
        return 0._rt;
#endif
    }

    /**
     * Perform mixed fourth derivative d^4/(dy^2 dz^2) on a cell-centered grid, from a
     * cell-centered field `F` (only nonzero in 3D)*/
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real Dyyzz (
        T_Field const& F,
        amrex::Real const * const coefs_y, int const /*n_coefs_y*/,
        amrex::Real const * const coefs_z, int const /*n_coefs_z*/,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
#if defined WARPX_DIM_3D
        amrex::Real const inv_dy2 = coefs_y[0]*coefs_y[0];
        amrex::Real const inv_dz2 = coefs_z[0]*coefs_z[0];
        amrex::Real result = 0._rt;
        for (int dj = -1; dj <= 1; ++dj) {
            amrex::Real const cy = (dj == 0) ? -2._rt : 1._rt;
            for (int dk = -1; dk <= 1; ++dk) {
                amrex::Real const cz = (dk == 0) ? -2._rt : 1._rt;
                result += cy*cz*F(i,j+dj,k+dk,ncomp);
            }
        }
        return inv_dy2*inv_dz2*result;
#else
        amrex::ignore_unused(F, coefs_y, coefs_z, i, j, k, ncomp);
        return 0._rt;
#endif
    }

    /**
     * Perform mixed fourth derivative d^4/(dx^2 dz^2) on a cell-centered grid, from a
     * cell-centered field `F` (nonzero in XZ and 3D; zero in 1D)*/
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real Dxxzz (
        T_Field const& F,
        amrex::Real const * const coefs_x, int const /*n_coefs_x*/,
        amrex::Real const * const coefs_z, int const /*n_coefs_z*/,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
#if defined WARPX_DIM_3D
        amrex::Real const inv_dx2 = coefs_x[0]*coefs_x[0];
        amrex::Real const inv_dz2 = coefs_z[0]*coefs_z[0];
        amrex::Real result = 0._rt;
        for (int di = -1; di <= 1; ++di) {
            amrex::Real const cx = (di == 0) ? -2._rt : 1._rt;
            for (int dk = -1; dk <= 1; ++dk) {
                amrex::Real const cz = (dk == 0) ? -2._rt : 1._rt;
                result += cx*cz*F(i+di,j,k+dk,ncomp);
            }
        }
        return inv_dx2*inv_dz2*result;
#elif (defined WARPX_DIM_XZ)
        // In XZ, x maps to the i index and z maps to the j index
        amrex::Real const inv_dx2 = coefs_x[0]*coefs_x[0];
        amrex::Real const inv_dz2 = coefs_z[0]*coefs_z[0];
        amrex::Real result = 0._rt;
        for (int di = -1; di <= 1; ++di) {
            amrex::Real const cx = (di == 0) ? -2._rt : 1._rt;
            for (int dj = -1; dj <= 1; ++dj) {
                amrex::Real const cz = (dj == 0) ? -2._rt : 1._rt;
                result += cx*cz*F(i+di,j+dj,k,ncomp);
            }
        }
        return inv_dx2*inv_dz2*result;
#else
        amrex::ignore_unused(F, coefs_x, coefs_z, i, j, k, ncomp);
        return 0._rt;
#endif
    }

};

/** In per-dimension builds, the unqualified name refers to the algorithm of
 *  the compiled dimensionality, so that existing use sites stay unchanged. */
using CartesianYeeAlgorithm = CartesianYeeAlgorithmND<warpx::CompileDim>;

#endif // WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_YEE_H_
