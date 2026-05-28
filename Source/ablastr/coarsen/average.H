/* Copyright 2022 Edoardo Zoni, Remi Lehe, Prabhat Kumar, Axel Huebl
 *
 * This file is part of ABLASTR.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_COARSEN_AVERAGE_H_
#define ABLASTR_COARSEN_AVERAGE_H_

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>
#include <AMReX_BaseFwd.H>

#include <cstdlib>


/** Mesh Coarsening by Averaging
 *
 * These methods are mostly used for mesh-refinement.
 */
namespace ablastr::coarsen::average
{
    AMREX_FORCE_INLINE
    void
    CalculateCoarseningData (
        amrex::GpuArray<int, 3> &idx_min_src,
        amrex::GpuArray<int, 3> &np_src,
        amrex::GpuArray<bool, 3> &use_half_weight, // Use int instead bool?
        amrex::Real &crx_cry_crz_inv,
        amrex::GpuArray<int, 3> const &stag_src,
        amrex::GpuArray<int, 3> const &stag_des,
        amrex::GpuArray<int, 3> const &crse_ratio)
    {
        using namespace amrex::literals;

        for (int l = 0; l < 3; ++l) {
            int stag_src_minus_cr_stag_des = stag_src[l] - crse_ratio[l]*stag_des[l];
            if (stag_src_minus_cr_stag_des & 1) {
                // stag_src - crse_ratio*stag_des is odd
                idx_min_src[l] = (stag_src_minus_cr_stag_des - 1)/2;
                np_src[l] = crse_ratio[l] + 1;
                use_half_weight[l] = true; // Use half weight at end points
            } else {
                // stag_src - crse_ratio*stag_des is even
                idx_min_src[l] = stag_src_minus_cr_stag_des/2;
                np_src[l] = crse_ratio[l];
                use_half_weight[l] = false; // Don't use half weight at end points
            }
        }

        crx_cry_crz_inv = 1.0_rt/static_cast<amrex::Real>(crse_ratio[0]*crse_ratio[1]*crse_ratio[2]);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real
    InterpWithCoarseningData (
        amrex::Array4<amrex::Real const> const &arr_src,
        amrex::GpuArray<int, 3> const &idx_min_src,
        amrex::GpuArray<int, 3> const &np_src,
        amrex::GpuArray<bool, 3> const &use_half_weight,
        amrex::Real crx_cry_crz_inv,
        amrex::GpuArray<int, 3> const &crse_ratio,
        int i, int j, int k, int comp)
    {
        using namespace amrex::literals;

        // Auxiliary integer variables
        int const krefmax = np_src[0] - 1;
        int const jrefmax = np_src[1] - 1;
        int const irefmax = np_src[2] - 1;
        int const iimin = i * crse_ratio[0] + idx_min_src[0];
        int const jjmin = j * crse_ratio[1] + idx_min_src[1];
        int const kkmin = k * crse_ratio[2] + idx_min_src[2];

        // Add neutral elements (=0) beyond guard cells in source array (fine)
        auto const arr_src_safe = [arr_src]
                AMREX_GPU_DEVICE(int const ix, int const iy, int const iz, int const n) noexcept {
            return arr_src.contains(ix, iy, iz) ? arr_src(ix, iy, iz, n) : 0.0_rt;
        };

        // Interpolate over points computed above. Weights are computed in order
        // to guarantee total charge conservation for any staggering.
        // Python script Source/Utils/check_interp_points_and_weights.py can be
        // used to check interpolation points and weights in 1D.
        amrex::Real c = 0.0_rt;
        for (int kref = 0; kref <= irefmax; ++kref) {
            int kk = kkmin + kref;
            bool use_half_kk = use_half_weight[2] && (kref == 0 || kref == irefmax);

            for (int jref = 0; jref <= jrefmax; ++jref) {
                int jj = jjmin + jref;
                bool use_half_jj = use_half_weight[1] && (jref == 0 || jref == jrefmax);

                for (int iref = 0; iref <= krefmax; ++iref) {
                    int ii = iimin + iref;
                    bool use_half_ii = use_half_weight[0] && (iref == 0 || iref == krefmax);

                    amrex::Real c_iijjkk = arr_src_safe(ii, jj, kk, comp);
                    if (use_half_kk) { c_iijjkk *= 0.5_rt; }
                    if (use_half_jj) { c_iijjkk *= 0.5_rt; }
                    if (use_half_ii) { c_iijjkk *= 0.5_rt; }
                    c += c_iijjkk;
                }
            }
        }

        c *= crx_cry_crz_inv;

        return c;
    }

    /**
     * \brief Interpolates the floating point data contained in the source Array4
     *        \c arr_src, extracted from a fine MultiFab, with weights defined in
     *        such a way that the total charge is preserved.
     *
     * Works for any staggering of the fine source and coarse destination.
     *
     * \param[in] arr_src    source fine floating point data to be interpolated
     * \param[in] stag_src   staggering of the source fine MultiFab
     * \param[in] stag_des   staggering of the destination coarsened MultiFab
     * \param[in] crse_ratio coarsening ratio along each spatial direction
     * \param[in] i          index along x of the coarsened Array4 to be filled
     * \param[in] j          index along y of the coarsened Array4 to be filled
     * \param[in] k          index along z of the coarsened Array4 to be filled
     * \param[in] comp       index along the fourth component of the Array4 \c arr_src
     *                       containing the data to be interpolated
     *
     * \return interpolated field at cell (i,j,k) of a coarsened Array4
     */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real
    Interp (
        amrex::Array4<amrex::Real const> const &arr_src,
        amrex::GpuArray<int, 3> const &stag_src,
        amrex::GpuArray<int, 3> const &stag_des,
        amrex::GpuArray<int, 3> const &crse_ratio,
        int const i, int const j, int const k, int const comp)
    {
        using namespace amrex::literals;

        // Number of points and starting indices of source array (fine)
        amrex::GpuArray<int, 3> np_src, idx_min_src;
        amrex::GpuArray<bool, 3> use_half_weight;
        amrex::Real crx_cry_crz_inv;

        // The following is intended to be an exact copy of CalculateCoarseningData
        // This is needed to calculate the coarsening data on the GPU device instead of the host

        for (int l = 0; l < 3; ++l) {
            int stag_src_minus_cr_stag_des = stag_src[l] - crse_ratio[l]*stag_des[l];
            if (stag_src_minus_cr_stag_des & 1) {
                // stag_src - crse_ratio*stag_des is odd
                idx_min_src[l] = (stag_src_minus_cr_stag_des - 1)/2;
                np_src[l] = crse_ratio[l] + 1;
                use_half_weight[l] = true; // Use half weight at end points
            } else {
                // stag_src - crse_ratio*stag_des is even
                idx_min_src[l] = stag_src_minus_cr_stag_des/2;
                np_src[l] = crse_ratio[l];
                use_half_weight[l] = false; // Don't use half weight at end points
            }
        }

        crx_cry_crz_inv = 1.0_rt/static_cast<amrex::Real>(crse_ratio[0]*crse_ratio[1]*crse_ratio[2]);

        // End copy of CalculateCoarseningData

        return InterpWithCoarseningData(
            arr_src,idx_min_src,np_src,use_half_weight,crx_cry_crz_inv,crse_ratio,i,j,k,comp);
    }

    /**
     * \brief Loops over the boxes of the coarsened MultiFab \c mf_dst and fills
     *        them by interpolating the data contained in the fine MultiFab \c mf_src.
     *
     * \param[in,out] mf_dst     coarsened MultiFab containing the floating point data
     *                           to be filled by interpolating the source fine MultiFab
     * \param[in]     mf_src     fine MultiFab containing the floating point data to be interpolated
     * \param[in]     ncomp      number of components to loop over for the coarsened
     *                           Array4 extracted from the coarsened MultiFab \c mf_dst
     * \param[in]     ngrow      number of guard cells to fill along each spatial direction
     * \param[in]     crse_ratio coarsening ratio between the fine MultiFab \c mf_src
     *                           and the coarsened MultiFab \c mf_dst along each spatial direction
     */
    void
    Loop (
        amrex::MultiFab & mf_dst,
        amrex::MultiFab const & mf_src,
        int ncomp,
        amrex::IntVect ngrow,
        amrex::IntVect crse_ratio
    );

    /**
     * \brief Stores in the coarsened MultiFab \c mf_dst the values obtained by
     *        interpolating the data contained in the fine MultiFab \c mf_src.
     *
     * \param[in,out] mf_dst     coarsened MultiFab containing the floating point data
     *                           to be filled by interpolating the fine MultiFab \c mf_src
     * \param[in]     mf_src     fine MultiFab containing the floating point data to be interpolated
     * \param[in]     crse_ratio coarsening ratio between the fine MultiFab \c mf_src
     *                           and the coarsened MultiFab \c mf_dst along each spatial direction
     */
    void
    Coarsen (
        amrex::MultiFab & mf_dst,
        amrex::MultiFab const & mf_src,
        amrex::IntVect crse_ratio
    );

} // namespace ablastr::coarsen::average

#endif // ABLASTR_COARSEN_AVERAGE_H_
