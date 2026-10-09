/* Copyright 2022 Edoardo Zoni, Remi Lehe, David Grote, Axel Huebl
 *
 * This file is part of ABLASTR.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_COARSEN_SAMPLE_H_
#define ABLASTR_COARSEN_SAMPLE_H_


#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <AMReX_BaseFwd.H>

#include <cstdlib>


/** Mesh Coarsening by Sampling
 *
 * These methods are mostly used for I/O.
 */
namespace ablastr::coarsen::sample
{
    /**
     * \brief Interpolates the floating point data contained in the source Array4
     *        \c arr_src, extracted from a fine MultiFab, by averaging over either
     *        1 point or 2 equally distant points.
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
    amrex::Real Interp (
        amrex::Array4<amrex::Real const> const& arr_src,
        amrex::GpuArray<int,3> const& sf,
        amrex::GpuArray<int,3> const& sc,
        amrex::GpuArray<int,3> const& cr,
        const int i,
        const int j,
        const int k,
        const int comp
    )
    {
        using namespace amrex::literals;

        // Indices of destination array (coarse)
        const int ic[3] = { i, j, k };

        // Number of points and starting indices of source array (fine)
        int np[3], idx_min[3];

        // Compute number of points
        for ( int l = 0; l < 3; ++l ) {
            if ( cr[l] == 1 ) { np[l] = 1+amrex::Math::abs(sf[l]-sc[l]); // no coarsening
            } else {            np[l] = 2-sf[l]; }
        }

        // Compute starting indices of source array (fine)
        for ( int l = 0; l < 3; ++l ) {
            if ( cr[l] == 1 ) { idx_min[l] = ic[l]-sc[l]*(1-sf[l]); // no coarsening
            } else {            idx_min[l] = ic[l]*cr[l]+static_cast<int>(cr[l]/2)*(1-sc[l])-(1-sf[l]); }
        }

        // Auxiliary integer variables
        const int numx = np[0];
        const int numy = np[1];
        const int numz = np[2];
        const int imin = idx_min[0];
        const int jmin = idx_min[1];
        const int kmin = idx_min[2];
        amrex::Real const wx = 1.0_rt / static_cast<amrex::Real>(numx);
        amrex::Real const wy = 1.0_rt / static_cast<amrex::Real>(numy);
        amrex::Real const wz = 1.0_rt / static_cast<amrex::Real>(numz);

        // Interpolate over points computed above
        amrex::Real c = 0.0_rt;
        for         (int kref = 0; kref < numz; ++kref) {
            for     (int jref = 0; jref < numy; ++jref) {
                for (int iref = 0; iref < numx; ++iref) {
                    const int ii = imin+iref;
                    const int jj = jmin+jref;
                    const int kk = kmin+kref;
                    c += wx*wy*wz*arr_src(ii,jj,kk,comp);
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
     * \param[in]     dcomp      offset for the fourth component of the coarsened Array4
     *                           object, extracted from its MultiFab, where the interpolated
     *                           values will be stored
     * \param[in]     scomp      offset for the fourth component of the fine Array4
     *                           object, extracted from its MultiFab, containing the
     *                           data to be interpolated
     * \param[in]     ncomp      number of components to loop over for the coarsened
     *                           Array4 extracted from the coarsened MultiFab \c mf_dst
     * \param[in]     ngrow      number of guard cells to fill
     * \param[in]     crse_ratio coarsening ratio between the fine MultiFab \c mf_src
     *                           and the coarsened MultiFab \c mf_dst along each spatial direction
     */
    void Loop ( amrex::MultiFab& mf_dst,
                const amrex::MultiFab& mf_src,
                int dcomp,
                int scomp,
                int ncomp,
                amrex::IntVect ngrow,
                amrex::IntVect crse_ratio=amrex::IntVect(1) );

    /**
     * \brief Stores in the coarsened MultiFab \c mf_dst the values obtained by
     *        interpolating the data contained in the fine MultiFab \c mf_src.
     *
     * \param[in,out] mf_dst     coarsened MultiFab containing the floating point data
     *                           to be filled by interpolating the fine MultiFab \c mf_src
     * \param[in]     mf_src     fine MultiFab containing the floating point data to be interpolated
     * \param[in]     dcomp      offset for the fourth component of the coarsened Array4
     *                           object, extracted from its MultiFab, where the interpolated
     *                           values will be stored
     * \param[in]     scomp      offset for the fourth component of the fine Array4
     *                           object, extracted from its MultiFab, containing the
     *                           data to be interpolated
     * \param[in]     ncomp      number of components to loop over for the coarsened
     *                           Array4 extracted from the coarsened MultiFab \c mf_dst
     * \param[in]     ngrow      number of guard cells to fill
     * \param[in]     crse_ratio coarsening ratio between the fine MultiFab \c mf_src
     *                           and the coarsened MultiFab \c mf_dst along each spatial direction
     */
    void Coarsen ( amrex::MultiFab& mf_dst,
                   const amrex::MultiFab& mf_src,
                   int dcomp,
                   int scomp,
                   int ncomp,
                   int ngrow,
                   amrex::IntVect crse_ratio=amrex::IntVect(1) );
    void Coarsen ( amrex::MultiFab& mf_dst,
                   const amrex::MultiFab& mf_src,
                   int dcomp,
                   int scomp,
                   int ncomp,
                   amrex::IntVect ngrowvect,
                   amrex::IntVect crse_ratio=amrex::IntVect(1) );

} // namespace ablastr::coarsen::sample

#endif // ABLASTR_COARSEN_SAMPLE_H_
