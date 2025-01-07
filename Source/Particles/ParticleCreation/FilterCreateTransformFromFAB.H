/* Copyright 2020 Luca Fedeli, Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_FILTER_CREATE_TRANSFORM_FROM_FAB_H_
#define WARPX_FILTER_CREATE_TRANSFORM_FROM_FAB_H_

#include "Particles/ParticleCreation/DefaultInitialization.H"

#include <AMReX_REAL.H>
#include <AMReX_TypeTraits.H>

/**
 * \brief Apply a filter on a list of FABs, then create and apply a transform
 * operation to the particles depending on the output of the filter.
 *
 * This version of the function takes as inputs a mask and a FAB that can be
 * used in the transform function, both of which can be obtained using another version
 * of filterCreateTransformFromFAB that takes a filter function as input.
 *
 * \tparam N number of particles created in the dst(s) in each cell
 * \tparam DstTile the dst particle tile type
 * \tparam FAB the src FAB type
 * \tparam Index the index type, e.g. unsigned int
 * \tparam CreateFunc1 the create function type for dst1
 * \tparam CreateFunc2 the create function type for dst2
 * \tparam TransFunc the transform function type
 *
 * \param[in,out] dst1 the first destination tile
 * \param[in,out] dst2 the second destination tile
 * \param[in] box the box where the particles are created
 * \param[in] src_FAB A FAB defined on box that is used in the transform function
 * \param[in] mask pointer to the mask: 1 means create, 0 means don't create
 * \param[in] dst1_index the location at which to starting writing the result to dst1
 * \param[in] dst2_index the location at which to starting writing the result to dst2
 * \param[in] create1 callable that defines what will be done for the create step for dst1.
 * \param[in] create2 callable that defines what will be done for the create step for dst2.
 * \param[in] transform callable that defines the transformation to apply on dst1 and dst2.
 * \param[in] geom_lev_zero the geometry object associated to level zero
 *
 * \return num_added the number of particles that were written to dst1 and dst2.
 */
template <int N, typename DstPC, typename DstTile, typename FAB, typename Index,
          typename CreateFunc1, typename CreateFunc2, typename TransFunc,
          amrex::EnableIf_t<std::is_integral_v<Index>, int> foo = 0>
Index filterCreateTransformFromFAB (DstPC& pc1, DstPC& pc2,
                                    DstTile& dst1, DstTile& dst2, const amrex::Box box,
                                    const FAB *src_FAB, const Index* mask,
                                    const Index dst1_index, const Index dst2_index,
                                    CreateFunc1&& create1, CreateFunc2&& create2,
                                    TransFunc&& transform, const amrex::Geometry& geom_lev_zero) noexcept
{
    using namespace amrex;

    const auto ncells = box.volume();
    if (ncells == 0) { return 0; }

    constexpr int spacedim = AMREX_SPACEDIM;

#if defined(WARPX_DIM_1D_Z)
    const Real zlo_global = geom_lev_zero.ProbLo(0);
    const Real dz         = geom_lev_zero.CellSize(0);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    const Real xlo_global = geom_lev_zero.ProbLo(0);
    const Real dx         = geom_lev_zero.CellSize(0);
    const Real zlo_global = geom_lev_zero.ProbLo(1);
    const Real dz         = geom_lev_zero.CellSize(1);
#elif defined(WARPX_DIM_3D)
    const Real xlo_global = geom_lev_zero.ProbLo(0);
    const Real dx         = geom_lev_zero.CellSize(0);
    const Real ylo_global = geom_lev_zero.ProbLo(1);
    const Real dy         = geom_lev_zero.CellSize(1);
    const Real zlo_global = geom_lev_zero.ProbLo(2);
    const Real dz         = geom_lev_zero.CellSize(2);
#endif

    const auto arrNumPartCreation = src_FAB->array();
    Gpu::DeviceVector<Index> offsets(ncells);
    auto total = amrex::Scan::ExclusiveSum(ncells, mask, offsets.data());
    const Index num_added = N*total;
    auto old_np1 = dst1.size();
    auto new_np1 = std::max(dst1_index + num_added, dst1.numParticles());
    dst1.resize(new_np1);

    auto old_np2 = dst2.size();
    auto new_np2 = std::max(dst2_index + num_added, dst2.numParticles());
    dst2.resize(new_np2);

    auto *p_offsets = offsets.dataPtr();

    const auto dst1_data = dst1.getParticleTileData();
    const auto dst2_data = dst2.getParticleTileData();

    // For loop over all cells in the box. If mask is true in the given cell,
    // we create the particles in the cell and apply a transform function to the
    // created particles.
    amrex::ParallelForRNG(ncells,
    [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
    {
        if (mask[i])
        {
            const IntVect iv = box.atOffset(i);
            const int j = iv[0];
            const int k = (spacedim >= 2) ? iv[1] : 0;
            const int l = (spacedim == 3) ? iv[2] : 0;

            // Currently all particles are created on nodes. This makes it useless
            // to use N>1 (for now).
#if defined(WARPX_DIM_1D_Z)
            Real const x = 0.0;
            Real const y = 0.0;
            Real const z = zlo_global + j*dz;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            Real const x = xlo_global + j*dx;
            Real const y = 0.0;
            Real const z = zlo_global + k*dz;
#elif defined(WARPX_DIM_3D)
            Real const x = xlo_global + j*dx;
            Real const y = ylo_global + k*dy;
            Real const z = zlo_global + l*dz;
#endif

            for (int n = 0; n < N; ++n)
            {
                create1(dst1_data, N*p_offsets[i] + dst1_index + n, engine, x, y, z);
                create2(dst2_data, N*p_offsets[i] + dst2_index + n, engine, x, y, z);
            }
            transform(dst1_data, dst2_data, N*p_offsets[i] + dst1_index,
                    N*p_offsets[i] + dst2_index, N, arrNumPartCreation(j,k,l));
        }
    });

    ParticleCreation::DefaultInitializeRuntimeAttributes(dst1,
                                       0, 0,
                                       pc1.getUserRealAttribs(), pc1.getUserIntAttribs(),
                                       pc1.getParticleComps(), pc1.getParticleiComps(),
                                       pc1.getUserRealAttribParser(),
                                       pc1.getUserIntAttribParser(),
#ifdef WARPX_QED
                                       false, // do not initialize QED quantities, since they were initialized
                                              // when calling the CreateFunc functor
                                       pc1.get_breit_wheeler_engine_ptr(),
                                       pc1.get_quantum_sync_engine_ptr(),
#endif
                                       pc1.getIonizationInitialLevel(),
                                       old_np1, new_np1);
    ParticleCreation::DefaultInitializeRuntimeAttributes(dst2,
                                       0, 0,
                                       pc2.getUserRealAttribs(), pc2.getUserIntAttribs(),
                                       pc2.getParticleComps(), pc2.getParticleiComps(),
                                       pc2.getUserRealAttribParser(),
                                       pc2.getUserIntAttribParser(),
#ifdef WARPX_QED
                                       false, // do not initialize QED quantities, since they were initialized
                                              // when calling the CreateFunc functor
                                       pc2.get_breit_wheeler_engine_ptr(),
                                       pc2.get_quantum_sync_engine_ptr(),
#endif
                                       pc2.getIonizationInitialLevel(),
                                       old_np2, new_np2);

    Gpu::synchronize();
    return num_added;
}

/**
 * \brief Apply a filter on a list of FABs, then create and apply a transform
 * operation to the particles depending on the output of the filter.
 *
 * This version of the function takes as input a filter functor (and an array of FABs that
 * can be used in the filter functor), uses it to obtain a mask and a FAB and then calls another
 * version of filterCreateTransformFromFAB that takes the mask and the FAB as inputs.
 *
 * \tparam N number of particles created in the dst(s) in each cell
 * \tparam DstTile the dst particle tile type
 * \tparam FABs the src array of Array4 type
 * \tparam Index the index type, e.g. unsigned int
 * \tparam FilterFunc the filter function type
 * \tparam CreateFunc1 the create function type for dst1
 * \tparam CreateFunc2 the create function type for dst2
 * \tparam TransFunc the transform function type
 *
 * \param[in,out] dst1 the first destination tile
 * \param[in,out] dst2 the second destination tile
 * \param[in] box the box where the particles are created
 * \param[in] src_FABs A collection of source data, e.g. a class with Array4 to the EM fields,
 *            defined on box on which the filter operation is applied
 * \param[in] dst1_index the location at which to starting writing the result to dst1
 * \param[in] dst2_index the location at which to starting writing the result to dst2
 * \param[in] filter a callable returning a value > 0 if particles are to be created
 *            in the considered cell.
 * \param[in] create1 callable that defines what will be done for the create step for dst1.
 * \param[in] create2 callable that defines what will be done for the create step for dst2.
 * \param[in] transform callable that defines the transformation to apply on dst1 and dst2.
 * \param[in] geom_lev_zero the geometry object associated to level zero
 *
 * \return num_added the number of particles that were written to dst1 and dst2.
 */
template <int N, typename DstPC, typename DstTile, typename FABs, typename Index,
          typename FilterFunc, typename CreateFunc1, typename CreateFunc2,
           typename TransFunc>
Index filterCreateTransformFromFAB (DstPC& pc1, DstPC& pc2, DstTile& dst1, DstTile& dst2, const amrex::Box box,
                                const FABs& src_FABs, const Index dst1_index,
                                const Index dst2_index, FilterFunc&& filter,
                                CreateFunc1&& create1, CreateFunc2&& create2,
                                TransFunc && transform, const amrex::Geometry& geom_lev_zero) noexcept
{
    using namespace amrex;

    FArrayBox NumPartCreation(box, 1);
    // This may be unnecessary because of the Gpu::streamSynchronize() in
    // filterCreateTransformFromFAB called below, but let us keep it for safety.
    const Elixir tmp_eli = NumPartCreation.elixir();
    auto arrNumPartCreation = NumPartCreation.array();

    const auto ncells = box.volume();
    if (ncells == 0) { return 0; }

    Gpu::DeviceVector<Index> mask(ncells);

    auto *p_mask = mask.dataPtr();

    // for loop over all cells in the box. We apply the filter function to each cell
    // and store the result in arrNumPartCreation. If the result is strictly greater than
    // 0, the mask is set to true at the given cell position.
    amrex::ParallelForRNG(box,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine){
        arrNumPartCreation(i,j,k) = filter(src_FABs,i,j,k,engine);
        const IntVect iv(AMREX_D_DECL(i,j,k));
        const auto mask_position = box.index(iv);
        p_mask[mask_position] = (arrNumPartCreation(i,j,k) > 0);
    });

    return filterCreateTransformFromFAB<N>(pc1, pc2, dst1, dst2, box, &NumPartCreation,
                                        mask.dataPtr(), dst1_index, dst2_index,
                                        std::forward<CreateFunc1>(create1),
                                        std::forward<CreateFunc2>(create2),
                                        std::forward<TransFunc>(transform),
                                        geom_lev_zero);
}

#endif // WARPX_FILTER_CREATE_TRANSFORM_FROM_FAB_H_
