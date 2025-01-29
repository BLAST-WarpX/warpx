/* Copyright 2019-2020 Andrew Myers, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_SORTING_SORTINGUTILS_H_
#define WARPX_PARTICLES_SORTING_SORTINGUTILS_H_

#include "Particles/WarpXParticleContainer.H"

#include <AMReX_Gpu.H>
#include <AMReX_Partition.H>
#include <AMReX_ParticleUtil.H>


/** \brief Fill the elements of the input vector with consecutive integer,
 *        starting from 0
 *
 * \param[inout] v Vector of integers, to be filled by this routine
 */
void fillWithConsecutiveIntegers( amrex::Gpu::DeviceVector<int>& v );

/** \brief Find the indices that would reorder the elements of `predicate`
 * so that the elements with non-zero value precede the other elements
 *
 * \tparam ForwardIterator An iterator that supports std::advance
 * \param[in, out] index_begin Point to the beginning of the vector which is
 *            to be filled with these indices
 * \param[in, out] index_end Point to the end of the vector which is
 *            to be filled with these indices
 * \param[in] predicate that indicates the elements that need to be reordered first
 */
template< typename ForwardIterator >
ForwardIterator stablePartition(ForwardIterator const index_begin,
                                ForwardIterator const index_end,
                                amrex::Gpu::DeviceVector<int> const& predicate)
{
#ifdef AMREX_USE_GPU
    // On GPU: Use amrex
    int const* AMREX_RESTRICT predicate_ptr = predicate.dataPtr();
    int N = static_cast<int>(std::distance(index_begin, index_end));
    auto num_true = amrex::StablePartition(&(*index_begin), N,
        [predicate_ptr] AMREX_GPU_DEVICE (int i) { return predicate_ptr[i]; });

    ForwardIterator sep = index_begin;
    std::advance(sep, num_true);
#else
    // On CPU: Use std library
    ForwardIterator const sep = std::stable_partition(
        index_begin, index_end,
        [&predicate](int i) { return predicate[i]; }
    );
#endif
    return sep;
}

/** \brief Return the number of elements between `first` and `last`
 *
 * \tparam ForwardIterator An iterator that supports std::distance
 * \param[in] first Points to a position in a vector
 * \param[in] last Points to another position in a vector
 * \return The number of elements between `first` and `last`
 */
template< typename ForwardIterator >
int iteratorDistance(ForwardIterator const first,
                     ForwardIterator const last)
{
    return std::distance( first, last );
}

/** \brief Functor that fills the elements of the particle array `inexflag`
 *  with the value of the spatial array `bmasks`, at the corresponding particle position.
 *
 * \param[in] pti Contains information on the particle positions
 * \param[in] bmasks Spatial array, that contains a flag indicating
 *         whether each cell is part of the gathering/deposition buffers
 * \param[out] inexflag Vector to be filled with the value of `bmasks`
 * \param[in] geom Geometry object, necessary to locate particles within the array `bmasks`
 *
 */
class fillBufferFlag
{
    public:
        fillBufferFlag( WarpXParIter const& pti, amrex::iMultiFab const* bmasks,
                        amrex::Gpu::DeviceVector<int>& inexflag,
                        amrex::Geometry const& geom ):
            // Extract simple structure that can be used directly on the GPU
            m_domain{geom.Domain()},
            m_inexflag_ptr{inexflag.dataPtr()},
            m_ptd{pti.GetParticleTile().getConstParticleTileData()},
            m_buffer_mask{(*bmasks)[pti].array()}
        {
            for (int idim=0; idim<AMREX_SPACEDIM; idim++) {
                m_prob_lo[idim] = geom.ProbLo(idim);
                m_inv_cell_size[idim] = geom.InvCellSize(idim);
            }
        }


        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void operator()( const int i ) const {
            // Find the index of the cell where this particle is located
            amrex::IntVect const iv = amrex::getParticleCell( m_ptd, i,
                                m_prob_lo, m_inv_cell_size, m_domain );
            // Find the value of the buffer flag in this cell and
            // store it at the corresponding particle position in the array `inexflag`
            m_inexflag_ptr[i] = m_buffer_mask(iv);
        }

    private:
        amrex::Box m_domain;
        int* m_inexflag_ptr;
        WarpXParticleContainer::ParticleTileType::ConstParticleTileDataType m_ptd;
        amrex::Array4<int const> m_buffer_mask;
        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_prob_lo;
        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_inv_cell_size;
};

/** \brief Functor that fills the elements of the particle array `inexflag`
 *  with the value of the spatial array `bmasks`, at the corresponding particle position.
 *
 * Contrary to `fillBufferFlag`, here this is done only for the particles that
 * the last elements of `particle_indices` point to (from the element at
 * index `start_index` in `particle_indices`, to the last element of `particle_indices`)
 *
 * \param[in] pti Contains information on the particle positions
 * \param[in] bmasks Spatial array, that contains a flag indicating
 *         whether each cell is part of the gathering/deposition buffers
 * \param[out] inexflag Vector to be filled with the value of `bmasks`
 * \param[in] geom Geometry object, necessary to locate particles within the array `bmasks`
 * \param[in] start_index Index that which elements start to be modified
 */
class fillBufferFlagRemainingParticles
{
    public:
        fillBufferFlagRemainingParticles(
                        WarpXParIter const& pti,
                        amrex::iMultiFab const* bmasks,
                        amrex::Gpu::DeviceVector<int>& inexflag,
                        amrex::Geometry const& geom,
                        amrex::Gpu::DeviceVector<int> const& particle_indices,
                        int start_index ) :
            m_domain{geom.Domain()},
            // Extract simple structure that can be used directly on the GPU
            m_inexflag_ptr{inexflag.dataPtr()},
            m_ptd{pti.GetParticleTile().getConstParticleTileData()},
            m_buffer_mask{(*bmasks)[pti].array()},
            m_start_index{start_index},
            m_indices_ptr{particle_indices.dataPtr()}
        {
            for (int idim=0; idim<AMREX_SPACEDIM; idim++) {
                m_prob_lo[idim] = geom.ProbLo(idim);
                m_inv_cell_size[idim] = geom.InvCellSize(idim);
            }
        }


        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void operator()( const int i ) const {
            // Select a particle
            auto const j = m_indices_ptr[i+m_start_index];
            // Find the index of the cell where this particle is located
            amrex::IntVect const iv = amrex::getParticleCell( m_ptd, j,
                                m_prob_lo, m_inv_cell_size, m_domain );
            // Find the value of the buffer flag in this cell and
            // store it at the corresponding particle position in the array `inexflag`
            m_inexflag_ptr[m_indices_ptr[i+m_start_index]] = m_buffer_mask(iv);
        }

    private:
        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_prob_lo;
        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_inv_cell_size;
        amrex::Box m_domain;
        int* m_inexflag_ptr;
        WarpXParticleContainer::ParticleTileType::ConstParticleTileDataType const m_ptd;
        amrex::Array4<int const> m_buffer_mask;
        int const m_start_index;
        int const* m_indices_ptr;
};

/** \brief Functor that copies the elements of `src` into `dst`,
 *       while reordering them according to `indices`
 *
 * \param[in] src Source vector
 * \param[out] dst Destination vector
 * \param[in] indices Array of indices that indicate how to reorder elements
 */
template <typename T>
class copyAndReorder
{
    public:
        copyAndReorder(
            amrex::Gpu::DeviceVector<T> const& src,
            amrex::Gpu::DeviceVector<T>& dst,
            amrex::Gpu::DeviceVector<int> const& indices ):
                // Extract simple structure that can be used directly on the GPU
                m_src_ptr{src.dataPtr()},
                m_dst_ptr{dst.dataPtr()},
                m_indices_ptr{indices.dataPtr()}
        {}

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        void operator()( const int ip ) const {
            m_dst_ptr[ip] = m_src_ptr[ m_indices_ptr[ip] ];
        }

    private:
        T const* m_src_ptr;
        T* m_dst_ptr;
        int const* m_indices_ptr;
};

#endif // WARPX_PARTICLES_SORTING_SORTINGUTILS_H_
