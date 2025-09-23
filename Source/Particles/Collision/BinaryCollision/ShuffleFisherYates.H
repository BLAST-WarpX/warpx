/* Copyright 2019 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_SHUFFLE_FISHER_YATES_H_
#define WARPX_PARTICLES_COLLISION_SHUFFLE_FISHER_YATES_H_

#include <AMReX_Random.H>
#include <AMReX_Scan.H>

/* \brief Shuffle array according to Fisher-Yates algorithm.
 *        Only shuffle the part between is <= i < ie, n = ie-is.
 *        T_index shall be
 *        amrex::DenseBins<WarpXParticleContainer::ParticleTileType::ParticleTileDataType>::index_type
*/
template <typename T_index>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void ShuffleFisherYates (T_index *array, T_index const is, T_index const ie,
                         amrex::RandomEngine const& engine)
{
    T_index buf;
    for (int i = ie-1; i >= static_cast<int>(is+1); --i)
    {
        // get random number j: is <= j <= i
        const int j = amrex::Random_int(i-is+1, engine) + is;
        // swap the ith array element with the jth
        buf      = array[i];
        array[i] = array[j];
        array[j] = buf;
    }
}

/* \brief A helper class for computing and looping over the independent pairs of
          macroparticles in each cell, for the purpose of processing collisions.
      It also has the ability to shuffle the particles in each cell for both
      species according to the Fisher-Yates algorithm.
      The number of independent pairs is equivalent to the number of particles per cell in
      whichever species has the fewer number of macroparticles.
*/
template <typename index_type>
struct IndependentPairHelper
{
    int m_n_cells;
    index_type const* AMREX_RESTRICT m_cell_offsets_1;
    index_type const* AMREX_RESTRICT m_cell_offsets_2;
    amrex::Gpu::DeviceVector<index_type> m_n_ind_pairs_in_each_cell;
    amrex::Gpu::DeviceVector<index_type> m_coll_offsets;
    int m_n_independent_pairs;

    /**
     * \brief Constructor
     *
     * @param[in] a_n_cells the number of cells in this tile
     * @param[in] a_cell_offsets_1 the offset array storing the number of particles per cell for species 1
     * @param[in] a_cell_offsets_2 the offset array storing the number of particles per cell for species 2
     *
     */
    IndependentPairHelper (int a_n_cells,
               index_type const* AMREX_RESTRICT a_cell_offsets_1,
               index_type const* AMREX_RESTRICT a_cell_offsets_2)
    : m_n_cells(a_n_cells), m_cell_offsets_1(a_cell_offsets_1), m_cell_offsets_2(a_cell_offsets_2)
    {
        m_n_ind_pairs_in_each_cell.resize(m_n_cells+1);
        m_coll_offsets.resize(m_n_cells+1);
        Initialize();
    }

    /* \brief Compute the number of independent pairs in each cell. This is equal to
     * the number of particles in whichever species has fewer
    */
    void Initialize () {
        // Compute the number of independent pairs in each cell. This is equal to
        // the number of particles in whichever species has fewer
        index_type* AMREX_RESTRICT p_n_ind_pairs_in_each_cell = m_n_ind_pairs_in_each_cell.dataPtr();
        int n_cells = m_n_cells;
        index_type const* AMREX_RESTRICT cell_offsets_1 = m_cell_offsets_1;
        index_type const* AMREX_RESTRICT cell_offsets_2 = m_cell_offsets_2;
        amrex::ParallelFor( n_cells+1, [=] AMREX_GPU_DEVICE (int i_cell) noexcept
        {
            if (i_cell < n_cells)
            {
                const auto n_part_in_cell_1 = cell_offsets_1[i_cell+1] - cell_offsets_1[i_cell];
                const auto n_part_in_cell_2 = cell_offsets_2[i_cell+1] - cell_offsets_2[i_cell];
                p_n_ind_pairs_in_each_cell[i_cell] = amrex::min(n_part_in_cell_1, n_part_in_cell_2);
            }
            else
                {
                    p_n_ind_pairs_in_each_cell[i_cell] = 0;
                }
        });

        // number of total independent collision pairs
        m_n_independent_pairs = (int) amrex::Scan::ExclusiveSum(m_n_cells+1,
            p_n_ind_pairs_in_each_cell, m_coll_offsets.data(), amrex::Scan::RetSum{true});
    }

    /**
     * \brief Shuffle the particles in each cell using the Fisher-Yates algorithm, for both species.
     *
     * @param[inout] a_indices_1 the indices of the particles sorted by cell for species 1
     * @param[inout] a_indices_2 the indices of the particles sorted by cell for species 2
     *
     */
    void shuffle (index_type* AMREX_RESTRICT a_indices_1,
          index_type* AMREX_RESTRICT a_indices_2)
    {
    // shuffle each species.
    // we launch 2*n_cells threads to process both species simultaneously
    int n_cells = m_n_cells;
    index_type const* AMREX_RESTRICT cell_offsets_1 = m_cell_offsets_1;
    index_type const* AMREX_RESTRICT cell_offsets_2 = m_cell_offsets_2;
    amrex::ParallelForRNG( 2*m_n_cells,
        [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
    {
        int i_cell = i < n_cells ? i : i - n_cells;

        // The particles from species1 that are in the cell `i_cell` are
        // given by the `indices_1[cell_start_1:cell_stop_1]`
        index_type const cell_start_1 = cell_offsets_1[i_cell];
        index_type const cell_stop_1  = cell_offsets_1[i_cell+1];

        // Same for species 2
        index_type const cell_start_2 = cell_offsets_2[i_cell];
        index_type const cell_stop_2  = cell_offsets_2[i_cell+1];

        // Do not collide if one species is missing in the cell
        if ( cell_stop_1 - cell_start_1 < 1 ||
         cell_stop_2 - cell_start_2 < 1 ) { return; }

        if (i < n_cells) {
        ShuffleFisherYates(a_indices_1, cell_start_1, cell_stop_1, engine);
        } else {
        ShuffleFisherYates(a_indices_2, cell_start_2, cell_stop_2, engine);
        }
    });
    }

    int numIndependentPairs () const {return m_n_independent_pairs;}
    const index_type* collisionOffsetsPtr () const {return m_coll_offsets.dataPtr();}
};

#endif // WARPX_PARTICLES_COLLISION_SHUFFLE_FISHER_YATES_H_
