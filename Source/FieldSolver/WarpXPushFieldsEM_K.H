/* Copyright 2019-2020
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_PushFieldsEM_K_h
#define WARPX_PushFieldsEM_K_h

#include "Utils/WarpXConst.H"

#include <AMReX.H>

/*
 * \brief Return a tilebox that only covers the outer half of the guard cells.
 *        For tileboxes that don't include cells beyond the whole domain,
 *        an empty box is returned.
 *
 * \param[in,out] input_tb tilebox to be modified
 * \param[in] dir direction where the tilebox smallEnd/bigEnd is modified
 * \param[in] n_domain number of valid cells in the whole simulation domain
 * \param[in] tb_smallEnd the lowest index of the tilebox, including guard cells
 * \param[in] tb_bigEnd the highest index of the tilebox, including guard cells
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Box constrain_tilebox_to_guards(
    const amrex::Box& input_tb,
    const int dir,
    const int iside,
    const int n_domain,
    const int tb_smallEnd,
    const int tb_bigEnd)
{
    using namespace amrex;

    amrex::Box constrained_tb = input_tb;

    // If the input_tb does not overlap either the lower or upper guard,
    // an empty box is returned.

    if (iside == 0)
    {
        // Lower guard
        const int n_guard = -tb_smallEnd;
        const int upper_bound = (tb_smallEnd < 0 ? -n_guard/2 : tb_smallEnd);
        constrained_tb.setBig(dir, upper_bound - 1);
    }
    else if (iside == 1)
    {
        // Upper guard
        const int n_guard = tb_bigEnd - n_domain;
        const int lower_bound = (tb_bigEnd > n_domain ? n_domain + n_guard/2 : tb_bigEnd);
        constrained_tb.setSmall(dir, lower_bound + 1);
    }

    return constrained_tb;
}

/*
 * \brief Damp a given field in the guard cells along a given direction
 *
 * \param[in,out] mf_arr array that contains the field values to be damped
 * \oaram[in] i index along x
 * \oaram[in] j index along y (in 3D) or z (in 2D/RZ)
 * \oaram[in] k index along z (in 3D, \c k = 0 in 2D/RZ)
 * \param[in] icomp index along the fourth component of the array
 * \param]in] dir direction where the field will be damped
 * \param[in] n_domain number of valid cells in the whole simulation domain
 * \param[in] tb_smallEnd the lowest index of the tilebox, including guard cells
 * \param[in] tb_bigEnd the highest index of the tilebox, including guard cells
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void damp_field_in_guards(
    amrex::Array4<amrex::Real> const& mf_arr,
    const int i,
    const int j,
    const int k,
    const int icomp,
    const int dir,
    const int n_domain,
    const int tb_smallEnd,
    const int tb_bigEnd)
{
    using namespace amrex;

    // dir = 0: idx = i (x)
    // dir = 1: idx = j (y in 3D, z in 2D/RZ)
    // dir = 2: idx = k (z in 3D)
    const int idx = ((dir == 0) ? i : ((dir == 1) ? j : k));

    if (idx < 0)
    {
        // Apply damping factor in guards cells below the lower end of the domain
        const int n_guard = -tb_smallEnd;

        const auto cell = static_cast<amrex::Real>(idx + n_guard);

        const amrex::Real phase = MathConst::pi * cell / n_guard;
        const amrex::Real sin_phase = std::sin(phase);
        const amrex::Real damp_factor = sin_phase * sin_phase;

        mf_arr(i,j,k,icomp) *= damp_factor;
    }
    else if (idx > n_domain)
    {
        // Apply damping factor in guards cells above the upper end of the domain
        const int n_guard = tb_bigEnd - n_domain;

        const auto cell = static_cast<amrex::Real>(tb_bigEnd - idx);

        const amrex::Real phase = MathConst::pi * cell / n_guard;
        const amrex::Real sin_phase = std::sin(phase);
        const amrex::Real damp_factor = sin_phase * sin_phase;

        mf_arr(i,j,k,icomp) *= damp_factor;
    }
}

#endif //WARPX_PushFieldsEM_K_h
