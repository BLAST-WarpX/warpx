#ifndef WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_CUSTOM_COEF_H_
#define WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_CUSTOM_COEF_H_

#include "Utils/WarpXConst.H"
#include "Utils/WarpXUtil.H"
#include "ablastr/warn_manager/WarnManager.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>
#include <AMReX_ParmParse.H> // Added this include
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

        amrex::ParmParse pp_algo("algo"); // Moved ParmParse here
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

        // --- ADDED PRINT STATEMENTS HERE ---
        amrex::Print() << "Custom Stencil Coefficients (x, upward):\n";
        for (int i = 0; i < stencil_coefs_x.size() / 2; ++i) {
            amrex::Print() << stencil_coefs_x[i] << " ";
        }
        amrex::Print() << "\n";

        amrex::Print() << "Custom Stencil Coefficients (x, downward):\n";
        for (int i = stencil_coefs_x.size() / 2; i < stencil_coefs_x.size(); ++i) {
            amrex::Print() << stencil_coefs_x[i] << " ";
        }
        amrex::Print() << "\n";

        amrex::Print() << "Custom Stencil Coefficients (y, upward):\n";
        for (int i = 0; i < stencil_coefs_y.size() / 2; ++i) {
             amrex::Print() << stencil_coefs_y[i] << " ";
        }
        amrex::Print() << "\n";

        amrex::Print() << "Custom Stencil Coefficients (y, downward):\n";
        for (int i = stencil_coefs_y.size() / 2; i < stencil_coefs_y.size(); ++i) {
            amrex::Print() << stencil_coefs_y[i] << " ";
        }
        amrex::Print() << "\n";

        amrex::Print() << "Custom Stencil Coefficients (z, upward):\n";
        for (int i = 0; i < stencil_coefs_z.size() / 2; ++i) {
            amrex::Print() << stencil_coefs_z[i] << " ";
        }
        amrex::Print() << "\n";

        amrex::Print() << "Custom Stencil Coefficients (z, downward):\n";
        for (int i = stencil_coefs_z.size() / 2; i < stencil_coefs_z.size(); ++i) {
            amrex::Print() << stencil_coefs_z[i] << " ";
        }
        amrex::Print() << "\n";
        // --- END OF ADDED PRINT STATEMENTS ---

    }

    static amrex::Real ComputeMaxDt(amrex::Real const * const dx) {
        using namespace amrex::literals;

        // Get the stencil coefficients
        amrex::ParmParse pp_algo("algo");
        amrex::Vector<amrex::Real> upward_coefs_x, downward_coefs_x;
        amrex::Vector<amrex::Real> upward_coefs_y, downward_coefs_y;
        amrex::Vector<amrex::Real> upward_coefs_z, downward_coefs_z;

        // Sum of absolute coefficients for each direction
        amrex::Real sum_abs_coefs_x = 1.0_rt;
        amrex::Real sum_abs_coefs_y = 1.0_rt;
        amrex::Real sum_abs_coefs_z = 1.0_rt;

        // If custom coefficients exist, calculate their absolute value sums
        if (pp_algo.queryarr("custom_coef_upward_stencil_x", upward_coefs_x) &&
            pp_algo.queryarr("custom_coef_downward_stencil_x", downward_coefs_x)) {
            sum_abs_coefs_x = 0.0_rt;
            for (auto const& coef : upward_coefs_x) sum_abs_coefs_x += std::abs(coef);
            for (auto const& coef : downward_coefs_x) sum_abs_coefs_x += std::abs(coef);
        }

        if (pp_algo.queryarr("custom_coef_upward_stencil_y", upward_coefs_y) &&
            pp_algo.queryarr("custom_coef_downward_stencil_y", downward_coefs_y)) {
            sum_abs_coefs_y = 0.0_rt;
            for (auto const& coef : upward_coefs_y) sum_abs_coefs_y += std::abs(coef);
            for (auto const& coef : downward_coefs_y) sum_abs_coefs_y += std::abs(coef);
        }

        if (pp_algo.queryarr("custom_coef_upward_stencil_z", upward_coefs_z) &&
            pp_algo.queryarr("custom_coef_downward_stencil_z", downward_coefs_z)) {
            sum_abs_coefs_z = 0.0_rt;
            for (auto const& coef : upward_coefs_z) sum_abs_coefs_z += std::abs(coef);
            for (auto const& coef : downward_coefs_z) sum_abs_coefs_z += std::abs(coef);
        }

        // Implementing equation (22) from the paper
        amrex::Real const delta_t = 1._rt / (PhysConst::c * std::sqrt(
            AMREX_D_TERM(
                (sum_abs_coefs_x * sum_abs_coefs_x) / (dx[0] * dx[0]),
                + (sum_abs_coefs_y * sum_abs_coefs_y) / (dx[1] * dx[1]),
                + (sum_abs_coefs_z * sum_abs_coefs_z) / (dx[2] * dx[2])
            )));

        // Print CFL number (dt*c/dx for the smallest dx)
        amrex::Real min_dx = std::min({dx[0], dx[1], dx[2]});
        amrex::Print() << "Computed CFL number: " << delta_t * PhysConst::c / min_dx << "\n";

        return delta_t;
    }




    static amrex::IntVect GetMaxGuardCell()
    {
        amrex::ParmParse pp_algo("algo");
        amrex::Vector<amrex::Real> upward_coefs_x, downward_coefs_x;
        amrex::Vector<amrex::Real> upward_coefs_y, downward_coefs_y;
        amrex::Vector<amrex::Real> upward_coefs_z, downward_coefs_z;

        // We only care about whether the arrays exist, and their sizes.  The
        // actual values don't matter for GetMaxGuardCell. We use dummy values
        // if they aren't in the input file
        int stencil_size_x = 1;
        if (pp_algo.queryarr("custom_coef_upward_stencil_x", upward_coefs_x) &&
            pp_algo.queryarr("custom_coef_downward_stencil_x", downward_coefs_x)) {
            stencil_size_x = std::max(upward_coefs_x.size(), downward_coefs_x.size());  //could be different
        }
        int stencil_size_y = 1;
        if (pp_algo.queryarr("custom_coef_upward_stencil_y", upward_coefs_y) &&
             pp_algo.queryarr("custom_coef_downward_stencil_y", downward_coefs_y)) {
            stencil_size_y = std::max(upward_coefs_y.size(), downward_coefs_y.size());
        }
        int stencil_size_z = 1;
         if (pp_algo.queryarr("custom_coef_upward_stencil_z", upward_coefs_z) &&
             pp_algo.queryarr("custom_coef_downward_stencil_z", downward_coefs_z)) {
            stencil_size_z = std::max(upward_coefs_z.size(), downward_coefs_z.size());
        }

        return amrex::IntVect{AMREX_D_DECL(stencil_size_x, stencil_size_y, stencil_size_z)};
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

        amrex::Real derivative = 0;
        int stencil_size = n_coefs_x / 2;
        for (int l = 0; l < stencil_size; l++) {
            // --- (Commented Out) Debugging Print Statement ---
            //amrex::Print() << "UpwardDx: i=" << i << " j=" << j << " k=" << k << " l=" << l << " coef=" << coefs_x[l] << " F(i+1+l)=" << F(i+1+l,j,k,ncomp) << " F(i-l)=" << F(i-l,j,k,ncomp) << "\n";
            derivative += coefs_x[l] * (F(i+1+l,j,k,ncomp) - F(i-l,j,k,ncomp));
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

        amrex::Real derivative = 0;
        int stencil_size = n_coefs_x / 2;
        for (int l = 0; l < stencil_size; l++) {
            // --- (Commented Out) Debugging Print Statement ---
            //amrex::Print() << "DownwardDx: i=" << i << " j=" << j << " k=" << k << " l=" << l << " coef=" << coefs_x[l + stencil_size] << " F(i+l)=" << F(i+l,j,k,ncomp) << " F(i-l-1)=" << F(i-l-1,j,k,ncomp) << "\n";
            derivative += coefs_x[l + stencil_size] * (F(i+l,j,k,ncomp) - F(i-l-1,j,k,ncomp));
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

        amrex::Real derivative = 0;
        int stencil_size = n_coefs_y / 2;
        for (int l = 0; l < stencil_size; l++) {
            // --- (Commented Out) Debugging Print Statement ---
            //amrex::Print() << "UpwardDy: i=" << i << " j=" << j << " k=" << k << " l=" << l << " coef=" << coefs_y[l] << " F(i,j+1+l)=" << F(i,j+1+l,k,ncomp) << " F(i,j-l)=" << F(i,j-l,k,ncomp) << "\n";
            derivative += coefs_y[l] * (F(i,j+1+l,k,ncomp) - F(i,j-l,k,ncomp));
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

        int stencil_size = n_coefs_y / 2;
        amrex::Real derivative = 0;
        for (int l = 0; l < stencil_size; l++) {
            // --- (Commented Out) Debugging Print Statement ---
            //amrex::Print() << "DownwardDy: i=" << i << " j=" << j << " k=" << k << " l=" << l << " coef=" << coefs_y[l + stencil_size] << " F(i,j+l)=" << F(i,j+l,k,ncomp) << " F(i,j-1-l)=" << F(i,j-1-l,k,ncomp) << "\n";
            derivative += coefs_y[l + stencil_size] * (F(i,j+l,k,ncomp) - F(i,j-1-l,k,ncomp));
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
        int stencil_size = n_coefs_z / 2;
        amrex::Real derivative = 0;
        for (int l = 0; l < stencil_size; l++) {
            // --- (Commented Out) Debugging Print Statement ---
            //amrex::Print() << "UpwardDz: i=" << i << " j=" << j << " k=" << k << " l=" << l << " coef=" << coefs_z[l] << " F(i,j,k+1+l)=" << F(i,j,k+1+l,ncomp) << " F(i,j,k-l)=" << F(i,j,k-l,ncomp) << "\n";
            derivative += coefs_z[l] * (F(i,j,k+1+l,ncomp) - F(i,j,k-l,ncomp));
        }
        return derivative;
#elif (defined WARPX_DIM_1D_Z)
        amrex::ignore_unused(F, i, j, k, ncomp);
        return 0._rt;
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

        amrex::Real derivative = 0;
        int stencil_size = n_coefs_z / 2;
        for (int l = 0; l < stencil_size; l++) {
            // --- (Commented Out) Debugging Print Statement ---
            //amrex::Print() << "DownwardDz: i=" << i << " j=" << j << " k=" << k << " l=" << l << " coef=" << coefs_z[l + stencil_size] << " F(i,j,k+l)=" << F(i,j,k+l,ncomp) << " F(i,j,k-1-l)=" << F(i,j,k-1-l,ncomp) << "\n";
            derivative += coefs_z[l + stencil_size] * (F(i,j,k+l,ncomp) - F(i,j,k-1-l,ncomp));
        }
        return derivative;
#elif (defined WARPX_DIM_1D_Z)
        amrex::ignore_unused(F, coefs_z, n_coefs_z, i, j, k, ncomp);
        return 0._rt;
#endif
    }
};

#endif // WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_CUSTOM_COEF_H_
