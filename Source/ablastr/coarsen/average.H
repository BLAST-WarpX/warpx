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
    /**
     * \brief Interpolates the floating point data contained in the source Array4
     *        \c arr_src, extracted from a fine MultiFab, with weights defined in
     *        such a way that the total charge is preserved.
     *
     * The input (sf) and output (sc) staggering need to be the same.
     *
     * \param[in] arr_src floating point data to be interpolated
     * \param[in] sf      staggering of the source fine MultiFab
     * \param[in] sc      staggering of the destination coarsened MultiFab
     * \param[in] cr      coarsening ratio along each spatial direction
     * \param[in] i       index along x of the coarsened Array4 to be filled
     * \param[in] j       index along y of the coarsened Array4 to be filled
     * \param[in] k       index along z of the coarsened Array4 to be filled
     * \param[in] comp    index along the fourth component of the Array4 \c arr_src
     *                    containing the data to be interpolated
     *
     * \return interpolated field at cell (i,j,k) of a coarsened Array4
     */
    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real
    Interp (
        amrex::Array4<amrex::Real const> const &arr_src,
        amrex::GpuArray<int, 3> const &sf,
        amrex::GpuArray<int, 3> const &sc,
        amrex::GpuArray<int, 3> const &cr,
        int const i,
        int const j,
        int const k,
        int const comp
    )
    {
        using namespace amrex::literals;

        AMREX_ASSERT_WITH_MESSAGE(sf[0] == sc[0], "Interp: Staggering for component 0 does not match!");
        AMREX_ASSERT_WITH_MESSAGE(sf[1] == sc[1], "Interp: Staggering for component 1 does not match!");
        AMREX_ASSERT_WITH_MESSAGE(sf[2] == sc[2], "Interp: Staggering for component 2 does not match!");

        // Indices of destination array (coarse)
        int const ic[3] = {i, j, k};

        // Number of points and starting indices of source array (fine)
        int np[3], idx_min[3];

        // Compute number of points
        for (int l = 0; l < 3; ++l) {
            if (cr[l] == 1) {
                np[l] = 1; // no coarsening
            } else {
                np[l] = cr[l] * (1 - sf[l]) * (1 - sc[l])        // cell-centered
                        + (2 * (cr[l] - 1) + 1) * sf[l] * sc[l]; // nodal
            }
        }

        // Compute starting indices of source array (fine)
        for (int l = 0; l < 3; ++l) {
            if (cr[l] == 1) {
                idx_min[l] = ic[l]; // no coarsening
            } else {
                idx_min[l] = ic[l] * cr[l] * (1 - sf[l]) * (1 - sc[l])      // cell-centered
                             + (ic[l] * cr[l] - cr[l] + 1) * sf[l] * sc[l]; // nodal
            }
        }

        // Auxiliary integer variables
        int const numx = np[0];
        int const numy = np[1];
        int const numz = np[2];
        int const imin = idx_min[0];
        int const jmin = idx_min[1];
        int const kmin = idx_min[2];
        int const sfx = sf[0];
        int const sfy = sf[1];
        int const sfz = sf[2];
        int const scx = sc[0];
        int const scy = sc[1];
        int const scz = sc[2];
        int const crx = cr[0];
        int const cry = cr[1];
        int const crz = cr[2];

        // Add neutral elements (=0) beyond guard cells in source array (fine)
        auto const arr_src_safe = [arr_src]
                AMREX_GPU_DEVICE(int const ix, int const iy, int const iz, int const n) noexcept {
            return arr_src.contains(ix, iy, iz) ? arr_src(ix, iy, iz, n) : 0.0_rt;
        };

        // Interpolate over points computed above. Weights are computed in order
        // to guarantee total charge conservation for both cell-centered data
        // (equal weights) and nodal data (weights depend on distance between
        // points on fine and coarse grids). Terms multiplied by (1-sf)*(1-sc)
        // are ON for cell-centered data and OFF for nodal data, while terms
        // multiplied by sf*sc are ON for nodal data and OFF for cell-centered data.
        // Python script Source/Utils/check_interp_points_and_weights.py can be
        // used to check interpolation points and weights in 1D.
        amrex::Real c = 0.0_rt;
        for (int kref = 0; kref < numz; ++kref) {
            for (int jref = 0; jref < numy; ++jref) {
                for (int iref = 0; iref < numx; ++iref) {
                    const int ii = imin + iref;
                    const int jj = jmin + jref;
                    const int kk = kmin + kref;
                    const amrex::Real wx = (1.0_rt / static_cast<amrex::Real>(numx)) * (1 - sfx) * (1 - scx) // if cell-centered
                         + ((amrex::Math::abs(crx - amrex::Math::abs(ii - i * crx))) /
                            static_cast<amrex::Real>(crx * crx)) * sfx * scx; // if nodal
                    const amrex::Real wy = (1.0_rt / static_cast<amrex::Real>(numy)) * (1 - sfy) * (1 - scy) // if cell-centered
                         + ((amrex::Math::abs(cry - amrex::Math::abs(jj - j * cry))) /
                            static_cast<amrex::Real>(cry * cry)) * sfy * scy; // if nodal
                    const amrex::Real wz = (1.0_rt / static_cast<amrex::Real>(numz)) * (1 - sfz) * (1 - scz) // if cell-centered
                         + ((amrex::Math::abs(crz - amrex::Math::abs(kk - k * crz))) /
                            static_cast<amrex::Real>(crz * crz)) * sfz * scz; // if nodal
                    c += wx * wy * wz * arr_src_safe(ii, jj, kk, comp);
                }
            }
        }
        return c;
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
