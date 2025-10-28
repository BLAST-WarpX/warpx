/* Copyright 2021-2025 Edoardo Zoni, Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_MATH_FINITE_DIFFERENCE_H_
#define ABLASTR_MATH_FINITE_DIFFERENCE_H_

#include "ablastr/utils/Enums.H"

#include <AMReX_Vector.H>
#include <AMReX_REAL.H>

namespace ablastr::math
{
    /**
     * \brief Returns an array of coefficients (Fornberg coefficients), corresponding
     * to the weight of each point in a finite-difference approximation of a derivative
     * (up to order \c n_order).
     *
     * \param[in] n_order order of the finite-difference approximation
     * \param[in] a_grid_type type of grid (collocated or not)
     */
    [[nodiscard]] amrex::Vector<amrex::Real>
    getFornbergStencilCoefficients (
        int n_order, ablastr::utils::enums::GridType a_grid_type);

    /**
     * \brief Re-orders the Fornberg coefficients so that they can be used more conveniently for
     * finite-order centering operations. For example, for finite-order centering of order 6,
     * the Fornberg coefficients \c (c_0,c_1,c_2) are re-ordered as \c (c_2,c_1,c_0,c_0,c_1,c_2).
     *
     * \param[in,out] ordered_coeffs host vector where the re-ordered Fornberg coefficients will be stored
     * \param[in] unordered_coeffs host vector storing the original sequence of Fornberg coefficients
     * \param[in] order order of the finite-order centering along a given direction
     */
    void
    ReorderFornbergCoefficients (
        amrex::Vector<amrex::Real>& ordered_coeffs,
        const amrex::Vector<amrex::Real>& unordered_coeffs, int order);
}

#endif //ABLASTR_MATH_FINITE_DIFFERENCE_H_
