/* Copyright 2019-2022 Axel Huebl, Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_INTERPOLATE_H
#define ABLASTR_INTERPOLATE_H

#include <AMReX_Array4.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFInterp_C.H>

#include <ablastr/fields/Interpolate.H>

#include <array>
#include <optional>


namespace ablastr::fields::details {

    /** Local interpolation from phi_cp to phi[lev+1]
     *
     * This is needed to work-around an NVCC limitation in downstream code (ImpactX),
     * when nesting lambdas. Otherwise this could be written directly into the
     * ParallelFor.
     *
     * @param[out] phi_fp_arr phi on the fine level
     * @param[in] phi_cp_arr phi on the coarse level
     * @param[in] refratio refinement ration
     */
    struct PoissonInterpCPtoFP
    {
        PoissonInterpCPtoFP(
                amrex::Array4<amrex::Real> const phi_fp_arr,
                amrex::Array4<amrex::Real const> const phi_cp_arr,
                amrex::IntVect const refratio)
        : m_phi_fp_arr(phi_fp_arr), m_phi_cp_arr(phi_cp_arr), m_refratio(refratio)
        {}

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        void
        operator() (int i, int j, int k) const noexcept
        {
            amrex::mf_nodebilin_interp(i, j, k, 0, m_phi_fp_arr, 0, m_phi_cp_arr,
                                       0, m_refratio);
        }

        amrex::Array4<amrex::Real> const m_phi_fp_arr;
        amrex::Array4<amrex::Real const> const m_phi_cp_arr;
        amrex::IntVect const m_refratio;
    };
} // namespace ablastr::fields::details

#endif // ABLASTR_INTERPOLATE_H
