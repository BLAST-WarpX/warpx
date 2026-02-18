/* Copyright 2021-2025 Edoardo Zoni, Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FiniteDifference.H"

#include "ablastr/utils/TextMsg.H"

using namespace ablastr::utils::enums;
using namespace amrex;

namespace ablastr::math
{

    amrex::Vector<amrex::Real>
    getFornbergStencilCoefficients (const int n_order, GridType a_grid_type)
    {
        ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(n_order % 2 == 0, "n_order must be even");

        const int m = n_order / 2;
        amrex::Vector<amrex::Real> coeffs;
        coeffs.resize(m);

        // There are closed-form formula for these coefficients, but they result in
        // an overflow when evaluated numerically. One way to avoid the overflow is
        // to calculate the coefficients by recurrence.

        // Coefficients for collocated (nodal) finite-difference approximation
        if (a_grid_type == GridType::Collocated)
        {
            // First coefficient
            coeffs.at(0) = m * 2._rt / (m+1);
            // Other coefficients by recurrence
            for (int n = 1; n < m; n++)
            {
                coeffs.at(n) = - (m-n) * 1._rt / (m+n+1) * coeffs.at(n-1);
            }
        }
        // Coefficients for staggered finite-difference approximation
        else
        {
            amrex::Real prod = 1.;
            for (int k = 1; k < m+1; k++)
            {
                prod *= (m + k) / (4._rt * k);
            }
            // First coefficient
            coeffs.at(0) = 4_rt * m * prod * prod;
            // Other coefficients by recurrence
            for (int n = 1; n < m; n++)
            {
                coeffs.at(n) = - ((2_rt*n-1) * (m-n)) * 1._rt / ((2_rt*n+1) * (m+n)) * coeffs.at(n-1);
            }
        }

        return coeffs;
    }

    void
    ReorderFornbergCoefficients (
        amrex::Vector<amrex::Real>& ordered_coeffs,
        const amrex::Vector<amrex::Real>& unordered_coeffs,
        const int order)
    {
        const int n = order / 2;
        for (int i = 0; i < n; i++) {
            ordered_coeffs[i] = unordered_coeffs[n-1-i];
        }
        for (int i = n; i < order; i++) {
            ordered_coeffs[i] = unordered_coeffs[i-n];
        }
    }

}
