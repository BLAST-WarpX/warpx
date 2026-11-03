/* Copyright 2022 Noah Kaplan, Andrew Myers, Phil Miller
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SHAREDDEPOSITIONUTILS_H_
#define WARPX_SHAREDDEPOSITIONUTILS_H_

#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/ShapeFactors.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#ifdef WARPX_DIM_RZ
#   include "Utils/WarpX_Complex.H"
#endif

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_ConstexprFor.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_IntVect.H>

#include <cstddef>

/*
 * \brief gets the maximum width, height, or length of a tilebox. In number of cells.
 * \param nCells : Number of cells in the direction to be considered
 * \param tilesize : The 1D tilesize in the direction to be considered
 */
AMREX_FORCE_INLINE
int getMaxTboxAlongDim (int nCells, int tilesize){
    int maxTilesize = 0;
    const int nTiles = nCells / tilesize;
    const int remainder = nCells % tilesize;
    maxTilesize = tilesize + int(std::ceil((amrex::Real) remainder / nTiles));
    return maxTilesize;
}

/*
 * \brief Box of the tile with a given index, for the tiles of a box numbered as by
 *        amrex::getTileIndex (with tiling). This is the inverse of amrex::getTileIndex.
 * \param tile_index : Index of the tile
 * \param box : The box that is divided into tiles
 * \param tile_size : The tile size
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Box getTileBoxFromIndex (int tile_index, const amrex::Box& box,
                                const amrex::IntVect& tile_size) noexcept
{
    amrex::IntVect tile_lo, tile_hi;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        // Must be consistent with the tiling_1d function in amrex::getTileIndex
        const int lo = box.smallEnd(idim);
        const int ncells = box.length(idim);
        const int ntile = amrex::max(ncells/tile_size[idim], 1);
        const int itile = tile_index % ntile;
        tile_index /= ntile;
        const int ts_right = ncells/ntile;
        const int ts_left = ts_right+1;
        const int nleft = ncells - ntile*ts_right;
        if (itile < nleft) {
            // tiles on the left have size of ts_left
            tile_lo[idim] = lo + itile*ts_left;
            tile_hi[idim] = tile_lo[idim] + ts_left - 1;
        } else {
            // tiles on the right have size of ts_right
            tile_lo[idim] = lo + itile*ts_right + nleft;
            tile_hi[idim] = tile_lo[idim] + ts_right - 1;
        }
    }
    return amrex::Box(tile_lo, tile_hi);
}

/*
 * \brief atomically add the values from the local deposition buffer back to the global array.
 * \param bx : Box defining the index space of the local buffer
 * \param global : The global array
 * \param local : The local array
 */
#if defined(AMREX_USE_HIP) || defined(AMREX_USE_CUDA)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void addLocalToGlobal (const amrex::Box& bx,
                       const amrex::Array4<amrex::Real>& global,
                       const amrex::Array4<amrex::Real>& local) noexcept
{
    using namespace amrex::literals;

    const auto lo  = amrex::lbound(bx);
    const auto len = amrex::length(bx);
    for (int icell = threadIdx.x; icell < bx.numPts(); icell += blockDim.x)
    {
        int k =  icell / (len.x*len.y);
        int j = (icell - k*(len.x*len.y)) /   len.x;
        int i = (icell - k*(len.x*len.y)) - j*len.x;
        i += lo.x;
        j += lo.y;
        k += lo.z;
        if (amrex::Math::abs(local(i, j, k)) > 0.0_rt) {
            amrex::Gpu::Atomic::AddNoRet( &global(i, j, k), local(i, j, k));
        }
    }
}
#endif

#if (defined(AMREX_USE_HIP) || defined(AMREX_USE_CUDA)) && \
    !(defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE))
/**
 * \brief Implementation of doSharedDeposition, for a given choice of the use of shared
 *        memory: launches one thread block per bin, which deposits the particles of the bin
 *        into shared memory buffers, then adds the buffers to the global arrays.
 *
 * \tparam fused  if true, deposit into all the arrays in a single pass, with one shared memory
 *                buffer per array; otherwise, deposit into them one at a time using a single
 *                buffer, which requires less shared memory but deposits each particle N times.
 * \tparam N      number of arrays to deposit into
 * \tparam DepositParticle  type of deposit_particle, see doSharedDeposition
 * \tparam Bins   type of the bins (amrex::DenseBins)
 * \param deposit_particle  functor that deposits one particle, see doSharedDeposition
 * \param arrays            the global arrays to deposit into
 * \param types             index types of the shared memory buffers of the arrays
 * \param a_bins            bins used for particle sorting
 * \param box               the current box
 * \param threads_per_block number of threads to use per block in shared deposition
 * \param bin_size          tileSize to use for shared current deposition operations
 * \param ngrow             number of cells by which the tiles are grown for the buffers
 * \param npts              number of points of the largest buffer, for the unfused version
 * \param shared_mem_bytes  size of shared memory per block
 */
template <bool fused, int N, typename DepositParticle, typename Bins>
void doSharedDepositionImpl (const DepositParticle& deposit_particle,
                             const amrex::GpuArray<amrex::Array4<amrex::Real>, N>& arrays,
                             const amrex::GpuArray<amrex::IntVect, N>& types,
                             const Bins& a_bins,
                             const amrex::Box& box,
                             const int threads_per_block,
                             const amrex::IntVect bin_size,
                             const int ngrow,
                             const amrex::Long npts,
                             const std::size_t shared_mem_bytes)
{
    using namespace amrex::literals;

    auto permutation = a_bins.permutationPtr();
    const int nblocks = a_bins.numBins();
    const auto offsets_ptr = a_bins.offsetsPtr();

    // Launch one thread-block per bin
    amrex::launch(
            nblocks, threads_per_block, shared_mem_bytes, amrex::Gpu::gpuStream(),
            [=] AMREX_GPU_DEVICE () noexcept {
        const int bin_id = blockIdx.x;
        const unsigned int bin_start = offsets_ptr[bin_id];
        const unsigned int bin_stop = offsets_ptr[bin_id+1];

        if (bin_start == bin_stop) { return; /*this bin has no particles*/ }

        // These boxes define the index space for the shared memory buffers.
        // Tile of this bin (the bins are numbered as by amrex::getTileIndex)
        amrex::Box buffer_box = getTileBoxFromIndex(bin_id, box, bin_size);
        buffer_box.grow(ngrow);
        amrex::GpuArray<amrex::Box, N> tbox;
        for (int n = 0; n < N; ++n) {
            tbox[n] = amrex::convert(buffer_box, types[n]);
        }

        amrex::Gpu::SharedMemory<amrex::Real> gsm;
        amrex::Real* const shared = gsm.dataPtr();

        // Note: not `if constexpr`, since nvcc does not allow an extended __device__ lambda
        // to first-capture variables in a constexpr-if context. `fused` is a template
        // parameter, so the branch is still resolved at compile time.
        if (fused) {
            // Deposit into one buffer per array, then add them to the global arrays
            amrex::GpuArray<amrex::Array4<amrex::Real>, N> buffers;
            int nshared = 0;
            for (int n = 0; n < N; ++n) {
                buffers[n] = amrex::Array4<amrex::Real>(shared + nshared,
                    amrex::begin(tbox[n]), amrex::end(tbox[n]), 1);
                nshared += static_cast<int>(tbox[n].numPts());
            }
            for (int i = threadIdx.x; i < nshared; i += blockDim.x) {
                shared[i] = 0.0_rt;
            }
            __syncthreads();
            for (unsigned int ip_orig = bin_start+threadIdx.x; ip_orig<bin_stop;
                 ip_orig += blockDim.x)
            {
                const unsigned int ip = permutation[ip_orig];
                deposit_particle.template deposit<(1 << N) - 1>(ip, buffers);
            }
            __syncthreads();
            for (int n = 0; n < N; ++n) {
                addLocalToGlobal(tbox[n], arrays[n], buffers[n]);
            }
        } else {
            // Deposit into one array at a time: deposit into the buffer, then add the
            // buffer to the global array
            amrex::constexpr_for<0, N>([&] (auto n) {
                amrex::Array4<amrex::Real> const buffer(shared,
                    amrex::begin(tbox[n]), amrex::end(tbox[n]), 1);
                amrex::GpuArray<amrex::Array4<amrex::Real>, N> buffers;
                for (int m = 0; m < N; ++m) { buffers[m] = buffer; }
                for (int i = threadIdx.x; i < npts; i += blockDim.x) {
                    shared[i] = 0.0_rt;
                }
                __syncthreads();
                for (unsigned int ip_orig = bin_start+threadIdx.x; ip_orig<bin_stop;
                     ip_orig += blockDim.x)
                {
                    const unsigned int ip = permutation[ip_orig];
                    deposit_particle.template deposit<(1 << decltype(n)::value)>(ip, buffers);
                }
                __syncthreads();
                addLocalToGlobal(tbox[n], arrays[n], buffer);
                __syncthreads();
            });
        }
    });
}

/**
 * \brief Deposition using shared memory buffers. Particles are binned into tiles of bin_size
 *        cells, and each tile is processed by one GPU thread block, which accumulates the
 *        deposited quantities in shared memory before adding them to the global arrays.
 *        Only available for CUDA and HIP.
 *
 * The deposition of each particle is done by deposit_particle, with a member function
 * template deposit, templated on a bitmask of the arrays to deposit into (bit n for array n),
 * that takes the index of the particle and an amrex::GpuArray of N Array4. The arrays that
 * are not in the bitmask must not be accessed. The deposition is done in one pass if the
 * buffers of all the arrays fit in shared memory, otherwise one array at a time.
 *
 * \tparam N      number of arrays to deposit into
 * \tparam DepositParticle  type of deposit_particle
 * \tparam Bins   type of the bins (amrex::DenseBins)
 * \param deposit_particle  functor that deposits one particle
 * \param arrays            the global arrays to deposit into
 * \param types             index types of the shared memory buffers of the arrays
 * \param ngrow             number of cells by which the tiles are grown for the buffers, so
 *                          that they contain the deposition stencils of the particles
 * \param a_bins            bins used for particle sorting
 * \param box               the current box
 * \param a_tbox_max_size   the maximum size of a tilebox
 * \param threads_per_block number of threads to use per block in shared deposition
 * \param bin_size          tileSize to use for shared current deposition operations
 */
template <int N, typename DepositParticle, typename Bins>
void doSharedDeposition (const DepositParticle& deposit_particle,
                         const amrex::GpuArray<amrex::Array4<amrex::Real>, N>& arrays,
                         const amrex::GpuArray<amrex::IntVect, N>& types,
                         const int ngrow,
                         const Bins& a_bins,
                         const amrex::Box& box,
                         const amrex::IntVect& a_tbox_max_size,
                         const int threads_per_block,
                         const amrex::IntVect bin_size)
{
    amrex::Box sample_tbox(amrex::IntVect(AMREX_D_DECL(0,0,0)), a_tbox_max_size - 1);
    sample_tbox.grow(ngrow);
    amrex::Long npts = 0;
    amrex::Long npts_fused = 0;
    for (int n = 0; n < N; ++n) {
        const amrex::Long np = amrex::convert(sample_tbox, types[n]).numPts();
        npts = amrex::max(npts, np);
        npts_fused += np;
    }

    const std::size_t max_shared_mem_bytes = amrex::Gpu::Device::sharedMemPerBlock();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(npts*sizeof(amrex::Real) <= max_shared_mem_bytes,
                                     "Tile size too big for GPU shared memory current deposition");

    // Deposit into all the arrays in a single pass if their buffers fit in shared memory
    const bool fused = (npts_fused*sizeof(amrex::Real) <= max_shared_mem_bytes);
    const std::size_t shared_mem_bytes = (fused ? npts_fused : npts)*sizeof(amrex::Real);
    if (fused) {
        doSharedDepositionImpl<true, N>(deposit_particle, arrays, types, a_bins, box,
            threads_per_block, bin_size, ngrow, npts, shared_mem_bytes);
    } else {
        doSharedDepositionImpl<false, N>(deposit_particle, arrays, types, a_bins, box,
            threads_per_block, bin_size, ngrow, npts, shared_mem_bytes);
    }
}
#endif

#endif // WARPX_SHAREDDEPOSITIONUTILS_H_
