/* Copyright 2020 Luca Fedeli, Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_QED_SCHWINGER_PROCESS_H_
#define WARPX_QED_SCHWINGER_PROCESS_H_

#include "Particles/ElementaryProcess/QEDInternals/SchwingerProcessWrapper.H"
#include "Utils/TextMsg.H"

/**
 * This structure is a functor which calls getSchwingerProductionNumber to
 * calculate the number of pairs created during a given timestep at a given cell.
 */
struct SchwingerFilterFunc
{
    const int m_threshold_poisson_gaussian;
    const amrex::Real m_dV;
    const amrex::Real m_dt;

    /** Get the number of created pairs in a given cell at a given timestep.
     *
     * \tparam FABs the src array of Array4 type
     *
     * @param[in] src_FABs A class with 6 named Array4 that contain the EM field in the tile.
     * @param[in] i index of the cell in the first direction.
     * @param[in] j index of the cell in the second direction.
     * @param[in] k index of the cell in the third direction.
     * @param[in] engine the random number state and factory
     * @return the number of pairs created in the cell.
     */
    template <typename FABs>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator() (const FABs& src_FABs,
                            const int i, const int j, const int k,
                            amrex::RandomEngine const& engine) const noexcept
    {
        const auto& arrEx = src_FABs.Ex;
        const auto& arrEy = src_FABs.Ey;
        const auto& arrEz = src_FABs.Ez;
        const auto& arrBx = src_FABs.Bx;
        const auto& arrBy = src_FABs.By;
        const auto& arrBz = src_FABs.Bz;

        return getSchwingerProductionNumber( m_dV, m_dt,
                    arrEx(i,j,k),arrEy(i,j,k),arrEz(i,j,k),
                    arrBx(i,j,k),arrBy(i,j,k),arrBz(i,j,k),
                    m_threshold_poisson_gaussian,engine);
    }
};


/**
 * This structure is a functor which assigns a weight to particles created via
 * the Schwinger process.
 */
struct SchwingerTransformFunc
{
    const amrex::Real m_y_size;
    const int m_weight_index;

    /** Assign a weight to particles created via the Schwinger process.
     *
     * \tparam DstTile the dst particle tile type
     *
     * @param[in,out] dst1 Target electron species.
     * @param[in,out] dst2 Target photon species.
     * @param[in] i_dst1 Particle index of target electron species.
     * @param[in] i_dst2 Particle index of target positron species.
     * @param[in] N Number of particles created per cell per species.
     * @param[in] total_weight Number of physical pairs created in the
     *            considered cell.
     */
    template <typename DstTile>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (DstTile& dst1, DstTile& dst2, const int i_dst1,
                       const int i_dst2, const int N,
                        const amrex::Real total_weight) const noexcept
    {
        for (int n = 0; n < N; ++n){
#if defined(WARPX_DIM_1D_Z)
        amrex::ignore_unused(dst1, dst2, i_dst1, i_dst2, N, total_weight);
        amrex::Abort("SchwingerTransformFunc not implemented in 1D");
#elif defined(WARPX_DIM_3D)
            dst1.m_rdata[m_weight_index][i_dst1+n] = total_weight/N;
            dst2.m_rdata[m_weight_index][i_dst2+n] = total_weight/N;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            dst1.m_rdata[m_weight_index][i_dst1+n] = total_weight/N/m_y_size;
            dst2.m_rdata[m_weight_index][i_dst2+n] = total_weight/N/m_y_size;
#endif
        }
    }
};

#endif // WARPX_QED_SCHWINGER_PROCESS_H_
