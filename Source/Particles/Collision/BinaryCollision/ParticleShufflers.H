/* Copyright 2019 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_PARTICLE_SHUFFLERS_H_
#define WARPX_PARTICLES_COLLISION_PARTICLE_SHUFFLERS_H_

#include <AMReX_Random.H>
#include <AMReX_Scan.H>

/* \brief Calculate the greatest common denominator of the two integers.
 *        This does the same thing as std::gcd, but that routine is not
 *        reliably available on GPU.
 *        As a comment, this is a fast operation with roughly
 *        the complexity of log(max(a,b)).
 * \param a, b Two integers, both greater than zero.
 * \result The greatest common denominator
 */
template <typename T_index>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
T_index gcd_int(T_index a, T_index b) {
    while (b != 0) {
        T_index r = a % b;
        a = b;
        b = r;
    }
    return a;
}

/* \brief Generate the step for the modulus shuffle
 *        In order for the shuffle to include all particles, the step
 *        and the number of particles must not have any common factors
 *        except 1, they are coprime.
 *        As a comment, the likelyhood that two integers are coprime
 *        is roughly 60%, so the while loop should succeed in a small
 *        number of attempts.
 *
 * \param n The number of particles
 * \param engine The random number generator engine
 * \result The generated step value
 */
template <typename T_index>
T_index generate_modulus_step(T_index n, amrex::RandomEngine const& engine)
{
    T_index step = 1;
    if (n > 2) {
        while (true) {
            // The step will be greater than 0 and less than n
            step = 1 + amrex::Random_int(n - 2, engine);
            if (gcd_int(n, step) == 1) {
                break;
            }
        }
    }
    return step;
}

/* \brief Shuffles the indices using a modulus shuffle.
 *        For each particle i, move it to (i*step + offset) % n,
 *        where i and n are relative to the grid cell.
 *        This will work for all particles if step and n are coprime.
 *
 * \param n_cell The number of grid cells
 * \param np The number of particles (the length of indices)
 * \param cell_offsets The particle number offset for each cell
 *                     (The cumulative integral of the number of particles in each cell.)
 * \param indices The indices to the particles that is to be shuffled
 */
template <typename T_index>
void ModulusShuffle (int n_cells, T_index np, T_index const * cell_offsets, T_index * indices)
{
#ifdef AMREX_USE_GPU

    // Doing the ParallelFor loops over the particles is substantially faster
    // than looping over cells on the GPU.

    amrex::Gpu::DeviceVector<T_index> indices_copy_vector(np);
    amrex::Gpu::DeviceVector<int> indices_step_vector(n_cells);
    amrex::Gpu::DeviceVector<int> indices_offset_vector(n_cells);
    T_index* AMREX_RESTRICT indices_copy = indices_copy_vector.dataPtr();
    T_index* AMREX_RESTRICT indices_step = indices_step_vector.dataPtr();
    T_index* AMREX_RESTRICT indices_offset = indices_offset_vector.dataPtr();

    // For each grid cell, calculates the step size and offset
    // that are used in the suffle.
    amrex::ParallelForRNG(n_cells,
        [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
        {
            T_index const cell_start = cell_offsets[i_cell];
            T_index const cell_stop  = cell_offsets[i_cell+1];

            T_index const n = cell_stop - cell_start;
            indices_step[i_cell] = generate_modulus_step(n, engine);
            indices_offset[i_cell] = amrex::Random_int(n, engine);
        });

    // A copy is needed of the indices
    amrex::ParallelFor(np,
        [=] AMREX_GPU_DEVICE (int ip) noexcept
        {
            indices_copy[ip] = indices[ip];
        });

    // Do the shuffle.
    amrex::ParallelFor(np,
        [=] AMREX_GPU_DEVICE (int ip) noexcept
        {
            T_index u_ip = (T_index)(ip);
            int const i_cell = amrex::bisect( cell_offsets, 0, n_cells, u_ip );
            T_index const cell_start = cell_offsets[i_cell];
            T_index const cell_stop  = cell_offsets[i_cell+1];

            T_index const n = cell_stop - cell_start;
            T_index const i = u_ip - cell_start;
            T_index const step = indices_step[i_cell];
            T_index const offset = indices_offset[i_cell];

            // A long is needed since the product i*step can exceed the range of
            // standard ints
            T_index const i2 = (int)(((amrex::Long)i*step + offset) % n);
            indices[ip] = indices_copy[cell_start + i2];
        }
    );

#else

    // This is faster on CPU, putting it all in a single loop

    amrex::Gpu::DeviceVector<T_index> indices_copy_vector(np);
    T_index* AMREX_RESTRICT indices_copy = indices_copy_vector.dataPtr();

    amrex::ParallelForRNG(n_cells,
        [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
        {
            T_index const cell_start = cell_offsets[i_cell];
            T_index const cell_stop  = cell_offsets[i_cell+1];
            T_index const n = cell_stop - cell_start;

            for (T_index i = cell_start ; i < cell_stop ; i++) {
                indices_copy[i] = indices[i];
            }

            T_index const step = generate_modulus_step(n, engine);
            T_index const offset = amrex::Random_int(n, engine);
            for (T_index i = 0 ; i < n ; i++) {
                T_index const i2 = (int)(((amrex::Long)i*step + offset) % n);
                indices[cell_start + i] = indices_copy[cell_start + i2];
            }
    });

#endif
}

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

/* \brief Shuffle array according to Fisher-Yates algorithm.
 *
 * \param n_cell The number of grid cells
 * \param cell_offsets The particle number offset for each cell
 *                     (The cumulative integral of the number of particles in each cell.)
 * \param indices The indices to the particles that is to be shuffled
*/
template <typename T_index>
void FisherYatesShuffle (int n_cells, T_index const * cell_offsets, T_index *indices)
{
    amrex::ParallelForRNG(n_cells,
        [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
        {
            // The particles from species1 that are in the cell `i_cell` are
            // given by the `indices[cell_start:cell_stop]`
            T_index const cell_start = cell_offsets[i_cell];
            T_index const cell_stop  = cell_offsets[i_cell+1];

            // Do not shuffle if there is only one particle in the cell
            if ( cell_stop - cell_start <= 1 ) { return; }

            ShuffleFisherYates(indices, cell_start, cell_stop, engine);
        }
    );
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

#endif // WARPX_PARTICLES_COLLISION_PARTICLE_SHUFFLERS_H_
