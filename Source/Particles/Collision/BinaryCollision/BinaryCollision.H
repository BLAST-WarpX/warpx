/* Copyright 2020-2021 Yinjian Zhao, David Grote, Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_BINARYCOLLISION_H_
#define WARPX_PARTICLES_COLLISION_BINARYCOLLISION_H_

#include "Particles/Collision/BinaryCollision/Coulomb/PairWiseCoulombCollisionFunc.H"
#include "Particles/Collision/BinaryCollision/Coulomb/ComputeTemperature.H"
#include "Particles/Collision/BinaryCollision/DSMC/DSMCFunc.H"
#include "Particles/Collision/BinaryCollision/NuclearFusion/NuclearFusionFunc.H"
#include "Particles/Collision/BinaryCollision/ParticleCreationFunc.H"
#include "Particles/Collision/BinaryCollision/ShuffleFisherYates.H"
#include "Particles/Collision/CollisionBase.H"
#include "Particles/ParticleCreation/SmartCopy.H"
#include "Particles/ParticleCreation/SmartUtils.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/ParticleUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "WarpX.H"

#include "Particles/MultiParticleContainer_fwd.H"
#include "Particles/WarpXParticleContainer_fwd.H"

#include <AMReX.H>
#include <AMReX_Algorithm.H>
#include <AMReX_BLassert.H>
#include <AMReX_Config.H>
#include <AMReX_DenseBins.H>
#include <AMReX_Extension.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MFIter.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>
#include <AMReX_Scan.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

#include <cmath>
#include <string>

/**
 * \brief This class performs generic binary collisions.
 *
 * \tparam CollisionFunctor the specific binary collision functor that acts on a
 *         single cell
 * \tparam CopyTransformFunctor the second functor used in the case of
 *         particle creation
 *
 */
template <typename CollisionFunctor,
          typename CopyTransformFunctor = NoParticleCreationFunc>
class BinaryCollision final
    : public CollisionBase
{
    // Define shortcuts for frequently-used type names
    using ParticleType = WarpXParticleContainer::ParticleType;
    using ParticleTileType = WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = ParticleBins::index_type;

public:
    /**
     * \brief Constructor of the BinaryCollision class.
     *
     * @param[in] collision_name the name of the collision
     * @param[in] mypc Container of species involved
     *
     */
    BinaryCollision (std::string collision_name, MultiParticleContainer const * const mypc)
        : CollisionBase(collision_name)
    {
        if(m_species_names.size() != 2) {
            WARPX_ABORT_WITH_MESSAGE("Binary collision " + collision_name + " must have exactly two species.");
        }

        const CollisionType collision_type = BinaryCollisionUtils::get_collision_type(collision_name, mypc);

        m_isSameSpecies = (m_species_names[0] == m_species_names[1]);

        m_binary_collision_functor = CollisionFunctor(collision_name, mypc, m_isSameSpecies);

        const amrex::ParmParse pp_collision_name(collision_name);
        pp_collision_name.queryarr("product_species", m_product_species);

        // if DSMC the colliding species are also product species
        // Therefore, we insert the colliding species at the beginning of `m_product_species`
        if (collision_type == CollisionType::DSMC) {
            m_product_species.insert( m_product_species.begin(), m_species_names.begin(), m_species_names.end() );
        }
        m_have_product_species = !m_product_species.empty();

        if ((std::is_same_v<CopyTransformFunctor, NoParticleCreationFunc>) && (m_have_product_species)) {
            WARPX_ABORT_WITH_MESSAGE( "Binary collision " + collision_name +
                " does not produce species. Thus, `product_species` should not be specified in the input script." );
        }
        m_copy_transform_functor = CopyTransformFunctor(collision_name, mypc);
    }

    ~BinaryCollision () override = default;

    BinaryCollision ( BinaryCollision const &)             = default;
    BinaryCollision& operator= ( BinaryCollision const & ) = default;

    BinaryCollision ( BinaryCollision&& )                  = delete;
    BinaryCollision& operator= ( BinaryCollision&& )       = delete;

    /** Perform the collisions
     *
     * @param cur_time Current time
     * @param dt Time step size
     * @param mypc Container of species involved
     *
     */
    void doCollisions (amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc) override
    {
        amrex::ignore_unused(cur_time);

        auto& species1 = mypc->GetParticleContainerFromName(m_species_names[0]);
        auto& species2 = mypc->GetParticleContainerFromName(m_species_names[1]);

        // In case of particle creation, create the necessary vectors
        const int n_product_species = m_product_species.size();
        amrex::Vector<WarpXParticleContainer*> product_species_vector;
        amrex::Vector<SmartCopyFactory> copy_factory_species1;
        amrex::Vector<SmartCopyFactory> copy_factory_species2;
        amrex::Vector<SmartCopy> copy_species1;
        amrex::Vector<SmartCopy> copy_species2;
        for (int i = 0; i < n_product_species; i++)
        {
            auto& product = mypc->GetParticleContainerFromName(m_product_species[i]);
            product.defineAllParticleTiles();
            product_species_vector.push_back(&product);
            // Although the copy factories are not explicitly reused past this point, we need to
            // store them in vectors so that the data that they own, which is used by the smart
            // copy functors, does not go out of scope at the end of this for loop.
            copy_factory_species1.push_back(SmartCopyFactory(species1, product));
            copy_factory_species2.push_back(SmartCopyFactory(species2, product));
            copy_species1.push_back(copy_factory_species1[i].getSmartCopy());
            copy_species2.push_back(copy_factory_species2[i].getSmartCopy());
        }
#ifdef AMREX_USE_GPU
        amrex::Gpu::DeviceVector<SmartCopy> device_copy_species1(n_product_species);
        amrex::Gpu::DeviceVector<SmartCopy> device_copy_species2(n_product_species);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, copy_species1.begin(),
                              copy_species1.end(), device_copy_species1.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, copy_species2.begin(),
                              copy_species2.end(), device_copy_species2.begin());
        amrex::Gpu::streamSynchronize();
        auto *copy_species1_data = device_copy_species1.data();
        auto *copy_species2_data = device_copy_species2.data();
#else
        auto *copy_species1_data = copy_species1.data();
        auto *copy_species2_data = copy_species2.data();
#endif
        if (m_have_product_species){
            species1.defineAllParticleTiles();
            if (!m_isSameSpecies) { species2.defineAllParticleTiles(); }
        }

        // Enable tiling
        amrex::MFItInfo info;
        if (amrex::Gpu::notInLaunchRegion()) { info.EnableTiling(species1.tile_size); }

        // Loop over refinement levels
        for (int lev = 0; lev <= species1.finestLevel(); ++lev){

        amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

        // Loop over all grids/tiles at this level
#ifdef AMREX_USE_OMP
            info.SetDynamic(true);
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi = species1.MakeMFIter(lev, info); mfi.isValid(); ++mfi){
                if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
                {
                    amrex::Gpu::synchronize();
                }
                auto wt = static_cast<amrex::Real>(amrex::second());

                doCollisionsWithinTile( dt, lev, mfi, species1, species2, product_species_vector,
                                        copy_species1_data, copy_species2_data);

                if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
                {
                    amrex::Gpu::synchronize();
                    wt = static_cast<amrex::Real>(amrex::second()) - wt;
                    amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
                }
            }

            if (m_have_product_species) {
                // The fact that there are product species indicates that particles of
                // the colliding species (`species1` and `species2`) may be removed
                // (i.e., marked as invalid) in the process of creating new product particles.
                species1.deleteInvalidParticles();
                if (!m_isSameSpecies) { species2.deleteInvalidParticles(); }
            }
        }
    }

    /** Perform all binary collisions within a tile
     *
     * \param[in] dt time step size
     * \param[in] lev the mesh-refinement level
     * \param[in] mfi iterator for multifab
     * \param species_1 first species container
     * \param species_2 second species container
     * \param product_species_vector vector of pointers to product species containers
     * \param copy_species1 vector of SmartCopy functors used to copy species 1 to product species
     * \param copy_species2 vector of SmartCopy functors used to copy species 2 to product species
     *
     */
    void doCollisionsWithinTile (
        amrex::Real dt, int const lev, amrex::MFIter const& mfi,
        WarpXParticleContainer& species_1,
        WarpXParticleContainer& species_2,
        amrex::Vector<WarpXParticleContainer*> product_species_vector,
        SmartCopy* copy_species1, SmartCopy* copy_species2)
    {
        using namespace ParticleUtils;
        using namespace amrex::literals;

        const auto& binary_collision_functor = m_binary_collision_functor.executor();
        const bool have_product_species = m_have_product_species;

        // Store product species data in vectors
        const int n_product_species = m_product_species.size();
        amrex::Vector<ParticleTileType*> tile_products;
        amrex::Vector<GetParticlePosition<PIdx>> get_position_products;
        amrex::Vector<index_type> products_np;
        amrex::Vector<amrex::ParticleReal> products_mass;
        constexpr int getpos_offset = 0;
        for (int i = 0; i < n_product_species; i++)
        {
            ParticleTileType& ptile_product = product_species_vector[i]->ParticlesAt(lev, mfi);
            tile_products.push_back(&ptile_product);
            get_position_products.push_back(GetParticlePosition<PIdx>(ptile_product,
                                                                      getpos_offset));
            products_np.push_back(ptile_product.numParticles());
            products_mass.push_back(product_species_vector[i]->getMass());
        }
        auto *tile_products_data = tile_products.data();

        amrex::Geometry const& geom = WarpX::GetInstance().Geom(lev);
        auto const dV = AMREX_D_TERM(geom.CellSize(0), *geom.CellSize(1), *geom.CellSize(2));
#if defined WARPX_DIM_RZ
        amrex::Box const& cbx = mfi.tilebox(amrex::IntVect::TheZeroVector()); //Cell-centered box
        auto const lo = lbound(cbx);
        auto const hi = ubound(cbx);
        int const nz = hi.y - lo.y + 1;
        auto const dr = geom.CellSize(0);
#endif

        auto volume_factor = [=] AMREX_GPU_DEVICE(int i_cell) noexcept {
#if defined WARPX_DIM_RZ
            // Return the radial factor for the volume element, dV
            int const ri = (i_cell - i_cell%nz)/nz;
            return MathConst::pi*(2.0_prt*ri + 1.0_prt)*dr;
#else
            // No factor is needed for Cartesian
            amrex::ignore_unused(i_cell);
            return 1._prt;
#endif
        };

        if ( m_isSameSpecies ) // species_1 == species_2
        {
            // Extract particles in the tile that `mfi` points to
            ParticleTileType& ptile_1 = species_1.ParticlesAt(lev, mfi);

            // Find the particles that are in each cell of this tile
            ParticleBins bins_1 = findParticlesInEachCell( lev, mfi, ptile_1 );

            // Loop over cells, and collide the particles in each cell

            // Extract low-level data
            auto const n_cells = static_cast<int>(bins_1.numBins());
            // - Species 1
            const auto soa_1 = ptile_1.getParticleTileData();
            index_type* AMREX_RESTRICT indices_1 = bins_1.permutationPtr();
            index_type const* AMREX_RESTRICT cell_offsets_1 = bins_1.offsetsPtr();
            const amrex::ParticleReal q1 = species_1.getCharge();
            const amrex::ParticleReal m1 = species_1.getMass();
            auto get_position_1  = GetParticlePosition<PIdx>(ptile_1, getpos_offset);

            /*
              The following calculations are only required when creating product particles
            */
            const int n_cells_products = have_product_species ? n_cells: 0;
            amrex::Gpu::DeviceVector<index_type> n_pairs_in_each_cell(n_cells_products);
            index_type* AMREX_RESTRICT p_n_pairs_in_each_cell = n_pairs_in_each_cell.dataPtr();

            // Compute how many pairs in each cell and store in n_pairs_in_each_cell array
            // For a single species, the number of pair in a cell is half the number of particles
            // in that cell, rounded up to the next higher integer.
            amrex::ParallelFor( n_cells_products,
                [=] AMREX_GPU_DEVICE (int i_cell) noexcept
                {
                    const auto n_part_in_cell = cell_offsets_1[i_cell+1] - cell_offsets_1[i_cell];
                    // Particular case: if there's only 1 particle in a cell, then there's no pair
                    p_n_pairs_in_each_cell[i_cell] = (n_part_in_cell == 1)? 0: (n_part_in_cell+1)/2;
                }
            );

            // Start indices of the pairs in a cell. Will be used for particle creation.
            amrex::Gpu::DeviceVector<index_type> pair_offsets(n_cells_products);
            const index_type n_total_pairs = (n_cells_products == 0) ? 0:
                                                amrex::Scan::ExclusiveSum(n_cells_products,
                                                    p_n_pairs_in_each_cell, pair_offsets.data());
            index_type* AMREX_RESTRICT p_pair_offsets = pair_offsets.dataPtr();

            amrex::Gpu::DeviceVector<index_type> n_ind_pairs_in_each_cell(n_cells+1);
            index_type* AMREX_RESTRICT p_n_ind_pairs_in_each_cell = n_ind_pairs_in_each_cell.dataPtr();

            amrex::ParallelFor( n_cells+1,
                [=] AMREX_GPU_DEVICE (int i_cell) noexcept
                {
                    const auto n_part_in_cell = (i_cell < n_cells)? cell_offsets_1[i_cell+1] - cell_offsets_1[i_cell]: 0;
                    // number of independent collisions in each cell
                    p_n_ind_pairs_in_each_cell[i_cell] = n_part_in_cell/2;
                }
            );

            // start indices of independent collisions.
            amrex::Gpu::DeviceVector<index_type> coll_offsets(n_cells+1);
            // number of total independent collision pairs
            const auto n_independent_pairs =  (int) amrex::Scan::ExclusiveSum(n_cells+1,
                                                    p_n_ind_pairs_in_each_cell, coll_offsets.data(), amrex::Scan::RetSum{true});
            index_type* AMREX_RESTRICT p_coll_offsets = coll_offsets.dataPtr();

            // mask: equal to 1 if particle creation occurs for a given pair, 0 otherwise
            amrex::Gpu::DeviceVector<index_type> mask(n_total_pairs);
            index_type* AMREX_RESTRICT p_mask = mask.dataPtr();
            // Will be filled with the index of the first particle of a given pair
            amrex::Gpu::DeviceVector<index_type> pair_indices_1(n_total_pairs);
            index_type* AMREX_RESTRICT p_pair_indices_1 = pair_indices_1.dataPtr();
            // Will be filled with the index of the second particle of a given pair
            amrex::Gpu::DeviceVector<index_type> pair_indices_2(n_total_pairs);
            index_type* AMREX_RESTRICT p_pair_indices_2 = pair_indices_2.dataPtr();
            // How much weight should be given to the produced particles (and removed from the
            // reacting particles)
            amrex::Gpu::DeviceVector<amrex::ParticleReal> pair_reaction_weight(n_total_pairs);
            amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight =
                                                               pair_reaction_weight.dataPtr();
            /*
              End of calculations only required when creating product particles
            */

            // create vectors to store density and temperature on cell level
            amrex::Gpu::DeviceVector<amrex::ParticleReal> n1_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> T1_vec;
            if (binary_collision_functor.m_computeSpeciesDensities) {
                n1_vec.resize(n_cells);
            }
            if (binary_collision_functor.m_computeSpeciesTemperatures) {
                T1_vec.resize(n_cells);
            }
            amrex::ParticleReal* AMREX_RESTRICT n1_in_each_cell = n1_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT T1_in_each_cell = T1_vec.dataPtr();

            // Loop over cells
            amrex::ParallelForRNG( n_cells,
                [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
                {
                    // The particles from species1 that are in the cell `i_cell` are
                    // given by the `indices_1[cell_start_1:cell_stop_1]`
                    index_type const cell_start_1 = cell_offsets_1[i_cell];
                    index_type const cell_stop_1  = cell_offsets_1[i_cell+1];

                    // Do not collide if there is only one particle in the cell
                    if ( cell_stop_1 - cell_start_1 <= 1 ) { return; }

                    // compute local density [1/m^3]
                    if (binary_collision_functor.m_computeSpeciesDensities) {
                        amrex::ParticleReal wtot1 = 0.0;
                        amrex::ParticleReal * const AMREX_RESTRICT w1 = soa_1.m_rdata[PIdx::w];
                        for (index_type i1=cell_start_1; i1<cell_stop_1; ++i1) {
                            wtot1 += w1[ indices_1[i1] ];
                        }

                        n1_in_each_cell[i_cell] = wtot1/(dV*volume_factor(i_cell));
                    }

                    // compute local temperature [Joules]
                    if (binary_collision_functor.m_computeSpeciesTemperatures) {
                        amrex::ParticleReal * const AMREX_RESTRICT w1  = soa_1.m_rdata[PIdx::w];
                        amrex::ParticleReal * const AMREX_RESTRICT u1x = soa_1.m_rdata[PIdx::ux];
                        amrex::ParticleReal * const AMREX_RESTRICT u1y = soa_1.m_rdata[PIdx::uy];
                        amrex::ParticleReal * const AMREX_RESTRICT u1z = soa_1.m_rdata[PIdx::uz];
                        T1_in_each_cell[i_cell] = ComputeTemperature( cell_start_1, cell_stop_1, indices_1,
                                                                      w1, u1x, u1y, u1z, m1 );
                    }

                    // shuffle
                    ShuffleFisherYates(indices_1, cell_start_1, cell_stop_1, engine);
                }
            );

            // Loop over independent particle pairs
            // To speed up binary collisions on GPU, we try to expose as much parallelism
            // as possible (while avoiding race conditions): Instead of looping with one GPU
            // thread per cell, we loop with one GPU thread per "independent pairs" (i.e. pairs
            // that do not touch the same macroparticles, so that there is no race condition),
            // where the number of independent pairs is determined by the lower number of
            // macroparticles of either species, within each cell.
            amrex::ParallelForRNG( n_independent_pairs,
                [=] AMREX_GPU_DEVICE (int i_coll, amrex::RandomEngine const& engine) noexcept
                {
                    // to avoid type mismatch errors
                    auto ui_coll = (index_type)i_coll;

                    // Use a bisection algorithm to find the index of the cell in which this pair is located
                    const int i_cell = amrex::bisect( p_coll_offsets, 0, n_cells, ui_coll );

                    // The particles from species1 that are in the cell `i_cell` are
                    // given by the `indices_1[cell_start_1:cell_stop_1]`
                    index_type const cell_start_1 = cell_offsets_1[i_cell];
                    index_type const cell_stop_1  = cell_offsets_1[i_cell+1];
                    index_type const cell_half_1 = (cell_start_1+cell_stop_1)/2;

                    // collision number of the cell
                    const index_type coll_idx = ui_coll - p_coll_offsets[i_cell];

                    // Same but for the pairs
                    index_type const cell_start_pair = have_product_species?
                                                           p_pair_offsets[i_cell] : 0;

                    // Get the local density and temperature for this cell
                    amrex::ParticleReal n1 = 0.0;
                    amrex::ParticleReal T1 = 0.0;
                    if (binary_collision_functor.m_computeSpeciesDensities) {
                        n1 = n1_in_each_cell[i_cell];
                    }
                    if (binary_collision_functor.m_computeSpeciesTemperatures) {
                        T1 = T1_in_each_cell[i_cell];
                    }

                    // Call the function in order to perform collisions
                    // If there are product species, mask, p_pair_indices_1/2, and
                    // p_pair_reaction_weight are filled here
                    binary_collision_functor(
                        cell_start_1, cell_half_1,
                        cell_half_1, cell_stop_1,
                        indices_1, indices_1,
                        soa_1, soa_1, get_position_1, get_position_1,
                        n1, n1, T1, T1,
                        q1, q1, m1, m1, dt, dV*volume_factor(i_cell), coll_idx,
                        cell_start_pair, p_mask, p_pair_indices_1, p_pair_indices_2,
                        p_pair_reaction_weight, engine);
                }
            );

            // Create the new product particles and define their initial values
            // num_added: how many particles of each product species have been created
            const amrex::Vector<int> num_added = m_copy_transform_functor(n_total_pairs,
                                                    ptile_1, ptile_1,
                                                    product_species_vector,
                                                    tile_products_data,
                                                    m1, m1,
                                                    products_mass, p_mask, products_np,
                                                    copy_species1, copy_species2,
                                                    p_pair_indices_1, p_pair_indices_2,
                                                    p_pair_reaction_weight);

            for (int i = 0; i < n_product_species; i++)
            {
                setNewParticleIDs(*(tile_products_data[i]), static_cast<int>(products_np[i]), num_added[i]);
            }
        }
        else // species_1 != species_2
        {
            // Extract particles in the tile that `mfi` points to
            ParticleTileType& ptile_1 = species_1.ParticlesAt(lev, mfi);
            ParticleTileType& ptile_2 = species_2.ParticlesAt(lev, mfi);

            // Find the particles that are in each cell of this tile
            ParticleBins bins_1 = findParticlesInEachCell( lev, mfi, ptile_1 );
            ParticleBins bins_2 = findParticlesInEachCell( lev, mfi, ptile_2 );

            // Loop over cells, and collide the particles in each cell

            // Extract low-level data
            auto const n_cells = static_cast<int>(bins_1.numBins());
            // - Species 1
            const auto soa_1 = ptile_1.getParticleTileData();
            index_type* AMREX_RESTRICT indices_1 = bins_1.permutationPtr();
            index_type const* AMREX_RESTRICT cell_offsets_1 = bins_1.offsetsPtr();
            const amrex::ParticleReal q1 = species_1.getCharge();
            const amrex::ParticleReal m1 = species_1.getMass();
            auto get_position_1  = GetParticlePosition<PIdx>(ptile_1, getpos_offset);
            // - Species 2
            const auto soa_2 = ptile_2.getParticleTileData();
            index_type* AMREX_RESTRICT indices_2 = bins_2.permutationPtr();
            index_type const* AMREX_RESTRICT cell_offsets_2 = bins_2.offsetsPtr();
            const amrex::ParticleReal q2 = species_2.getCharge();
            const amrex::ParticleReal m2 = species_2.getMass();
            auto get_position_2  = GetParticlePosition<PIdx>(ptile_2, getpos_offset);

            /*
              The following calculations are only required when creating product particles
            */
            const int n_cells_products = have_product_species ? n_cells: 0;
            amrex::Gpu::DeviceVector<index_type> n_pairs_in_each_cell(n_cells_products);
            index_type* AMREX_RESTRICT p_n_pairs_in_each_cell = n_pairs_in_each_cell.dataPtr();

            // Compute how many pairs in each cell and store in n_pairs_in_each_cell array
            // For different species, the number of pairs in a cell is the number of particles of
            // the species that has the most particles in that cell
            amrex::ParallelFor( n_cells_products,
                [=] AMREX_GPU_DEVICE (int i_cell) noexcept
                {
                    const auto n_part_in_cell_1 = cell_offsets_1[i_cell+1] - cell_offsets_1[i_cell];
                    const auto n_part_in_cell_2 = cell_offsets_2[i_cell+1] - cell_offsets_2[i_cell];
                    // Particular case: no pair if a species has no particle in that cell
                    if (n_part_in_cell_1 == 0 || n_part_in_cell_2 == 0) {
                        p_n_pairs_in_each_cell[i_cell] = 0;
                    } else {
                        p_n_pairs_in_each_cell[i_cell] =
                                                      amrex::max(n_part_in_cell_1,n_part_in_cell_2);
                    }
                }
            );

            // Start indices of the pairs in a cell. Will be used for particle creation
            amrex::Gpu::DeviceVector<index_type> pair_offsets(n_cells_products);
            const index_type n_total_pairs = (n_cells_products == 0) ? 0:
                                                amrex::Scan::ExclusiveSum(n_cells_products,
                                                    p_n_pairs_in_each_cell, pair_offsets.data());
            index_type* AMREX_RESTRICT p_pair_offsets = pair_offsets.dataPtr();

            amrex::Gpu::DeviceVector<index_type> n_ind_pairs_in_each_cell(n_cells+1);
            index_type* AMREX_RESTRICT p_n_ind_pairs_in_each_cell = n_ind_pairs_in_each_cell.dataPtr();

            amrex::ParallelFor( n_cells+1,
                [=] AMREX_GPU_DEVICE (int i_cell) noexcept
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
                }
            );

            // start indices of independent collisions.
            amrex::Gpu::DeviceVector<index_type> coll_offsets(n_cells+1);
            // number of total independent collision pairs
            const auto n_independent_pairs = (int) amrex::Scan::ExclusiveSum(n_cells+1,
                                                    p_n_ind_pairs_in_each_cell, coll_offsets.data(), amrex::Scan::RetSum{true});
            index_type* AMREX_RESTRICT p_coll_offsets = coll_offsets.dataPtr();

            // mask: equal to 1 if particle creation occurs for a given pair, 0 otherwise
            amrex::Gpu::DeviceVector<index_type> mask(n_total_pairs);
            index_type* AMREX_RESTRICT p_mask = mask.dataPtr();
            // Will be filled with the index of the first particle of a given pair
            amrex::Gpu::DeviceVector<index_type> pair_indices_1(n_total_pairs);
            index_type* AMREX_RESTRICT p_pair_indices_1 = pair_indices_1.dataPtr();
            // Will be filled with the index of the second particle of a given pair
            amrex::Gpu::DeviceVector<index_type> pair_indices_2(n_total_pairs);
            index_type* AMREX_RESTRICT p_pair_indices_2 = pair_indices_2.dataPtr();
            // How much weight should be given to the produced particles (and removed from the
            // reacting particles)
            amrex::Gpu::DeviceVector<amrex::ParticleReal> pair_reaction_weight(n_total_pairs);
            amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight =
                                                               pair_reaction_weight.dataPtr();
            /*
              End of calculations only required when creating product particles
            */

            // create vectors to store density and temperature on cell level
            amrex::Gpu::DeviceVector<amrex::ParticleReal> n1_vec, n2_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> T1_vec, T2_vec;
            if (binary_collision_functor.m_computeSpeciesDensities) {
                n1_vec.resize(n_cells);
                n2_vec.resize(n_cells);
            }
            if (binary_collision_functor.m_computeSpeciesTemperatures) {
                T1_vec.resize(n_cells);
                T2_vec.resize(n_cells);
            }
            amrex::ParticleReal* AMREX_RESTRICT n1_in_each_cell = n1_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT n2_in_each_cell = n2_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT T1_in_each_cell = T1_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT T2_in_each_cell = T2_vec.dataPtr();

            // Loop over cells
            amrex::ParallelForRNG( n_cells,
                [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
                {
                    // The particles from species1 that are in the cell `i_cell` are
                    // given by the `indices_1[cell_start_1:cell_stop_1]`
                    index_type const cell_start_1 = cell_offsets_1[i_cell];
                    index_type const cell_stop_1  = cell_offsets_1[i_cell+1];
                    // Same for species 2
                    index_type const cell_start_2 = cell_offsets_2[i_cell];
                    index_type const cell_stop_2  = cell_offsets_2[i_cell+1];

                    // ux from species1 can be accessed like this:
                    // ux_1[ indices_1[i] ], where i is between
                    // cell_start_1 (inclusive) and cell_start_2 (exclusive)

                    // compute local densities [1/m^3]
                    if (binary_collision_functor.m_computeSpeciesDensities) {
                        amrex::ParticleReal w1tot = 0.0;
                        amrex::ParticleReal w2tot = 0.0;
                        amrex::ParticleReal * const AMREX_RESTRICT w1 = soa_1.m_rdata[PIdx::w];
                        amrex::ParticleReal * const AMREX_RESTRICT w2 = soa_2.m_rdata[PIdx::w];
                        for (index_type i1=cell_start_1; i1<cell_stop_1; ++i1) {
                            w1tot += w1[ indices_1[i1] ];
                        }
                        for (index_type i2=cell_start_2; i2<cell_stop_2; ++i2) {
                            w2tot += w2[ indices_2[i2] ];
                        }
                        n1_in_each_cell[i_cell] = w1tot/(dV*volume_factor(i_cell));
                        n2_in_each_cell[i_cell] = w2tot/(dV*volume_factor(i_cell));
                    }

                    // compute local temperatures [Joules]
                    if (binary_collision_functor.m_computeSpeciesTemperatures) {
                        amrex::ParticleReal * const AMREX_RESTRICT w1  = soa_1.m_rdata[PIdx::w];
                        amrex::ParticleReal * const AMREX_RESTRICT u1x = soa_1.m_rdata[PIdx::ux];
                        amrex::ParticleReal * const AMREX_RESTRICT u1y = soa_1.m_rdata[PIdx::uy];
                        amrex::ParticleReal * const AMREX_RESTRICT u1z = soa_1.m_rdata[PIdx::uz];
                        T1_in_each_cell[i_cell] = ComputeTemperature( cell_start_1, cell_stop_1, indices_1,
                                                                      w1, u1x, u1y, u1z, m1 );

                        amrex::ParticleReal * const AMREX_RESTRICT w2  = soa_2.m_rdata[PIdx::w];
                        amrex::ParticleReal * const AMREX_RESTRICT u2x = soa_2.m_rdata[PIdx::ux];
                        amrex::ParticleReal * const AMREX_RESTRICT u2y = soa_2.m_rdata[PIdx::uy];
                        amrex::ParticleReal * const AMREX_RESTRICT u2z = soa_2.m_rdata[PIdx::uz];
                        T2_in_each_cell[i_cell] = ComputeTemperature( cell_start_2, cell_stop_2, indices_2,
                                                                      w2, u2x, u2y, u2z, m2 );
                    }

                    // Do not collide if one species is missing in the cell
                    if ( cell_stop_1 - cell_start_1 < 1 ||
                         cell_stop_2 - cell_start_2 < 1 ) { return; }

                    // shuffle
                    ShuffleFisherYates(indices_1, cell_start_1, cell_stop_1, engine);
                    ShuffleFisherYates(indices_2, cell_start_2, cell_stop_2, engine);
                }
            );

            // Loop over independent particle pairs
            // To speed up binary collisions on GPU, we try to expose as much parallelism
            // as possible (while avoiding race conditions): Instead of looping with one GPU
            // thread per cell, we loop with one GPU thread per "independent pairs" (i.e. pairs
            // that do not touch the same macroparticles, so that there is no race condition),
            // where the number of independent pairs is determined by the lower number of
            // macroparticles of either species, within each cell.
            amrex::ParallelForRNG( n_independent_pairs,
                [=] AMREX_GPU_DEVICE (int i_coll, amrex::RandomEngine const& engine) noexcept
                {
                    // to avoid type mismatch errors
                    auto ui_coll = (index_type)i_coll;

                    // Use a bisection algorithm to find the index of the cell in which this pair is located
                    const int i_cell = amrex::bisect( p_coll_offsets, 0, n_cells, ui_coll );

                    // The particles from species1 that are in the cell `i_cell` are
                    // given by the `indices_1[cell_start_1:cell_stop_1]`
                    index_type const cell_start_1 = cell_offsets_1[i_cell];
                    index_type const cell_stop_1  = cell_offsets_1[i_cell+1];
                    // Same for species 2
                    index_type const cell_start_2 = cell_offsets_2[i_cell];
                    index_type const cell_stop_2  = cell_offsets_2[i_cell+1];

                    // collision number of the cell
                    const index_type coll_idx = ui_coll - p_coll_offsets[i_cell];

                    // Same but for the pairs
                    index_type const cell_start_pair = have_product_species?
                                                           p_pair_offsets[i_cell]: 0;

                    // ux from species1 can be accessed like this:
                    // ux_1[ indices_1[i] ], where i is between
                    // cell_start_1 (inclusive) and cell_start_2 (exclusive)

                    // Get the local densities and temperatures for this cell
                    amrex::ParticleReal n1 = 0.0, n2 = 0.0;
                    amrex::ParticleReal T1 = 0.0, T2 = 0.0;
                    if (binary_collision_functor.m_computeSpeciesDensities) {
                        n1 = n1_in_each_cell[i_cell];
                        n2 = n2_in_each_cell[i_cell];
                    }
                    if (binary_collision_functor.m_computeSpeciesTemperatures) {
                        T1 = T1_in_each_cell[i_cell];
                        T2 = T2_in_each_cell[i_cell];
                    }

                    // Call the function in order to perform collisions
                    // If there are product species, p_mask, p_pair_indices_1/2, and
                    // p_pair_reaction_weight are filled here
                    binary_collision_functor(
                        cell_start_1, cell_stop_1, cell_start_2, cell_stop_2,
                        indices_1, indices_2,
                        soa_1, soa_2, get_position_1, get_position_2,
                        n1, n2, T1, T2,
                        q1, q2, m1, m2, dt, dV*volume_factor(i_cell), coll_idx,
                        cell_start_pair, p_mask, p_pair_indices_1, p_pair_indices_2,
                        p_pair_reaction_weight, engine);
                }
            );

            // Create the new product particles and define their initial values
            // num_added: how many particles of each product species have been created
            const amrex::Vector<int> num_added = m_copy_transform_functor(n_total_pairs,
                                                    ptile_1, ptile_2,
                                                    product_species_vector,
                                                    tile_products_data,
                                                    m1, m2,
                                                    products_mass, p_mask, products_np,
                                                    copy_species1, copy_species2,
                                                    p_pair_indices_1, p_pair_indices_2,
                                                    p_pair_reaction_weight);

            for (int i = 0; i < n_product_species; i++)
            {
                setNewParticleIDs(*(tile_products_data[i]), static_cast<int>(products_np[i]), num_added[i]);
            }

        } // end if ( m_isSameSpecies)

    }

private:

    bool m_isSameSpecies;
    bool m_have_product_species;
    amrex::Vector<std::string> m_product_species;
    // functor that performs collisions within a cell
    CollisionFunctor m_binary_collision_functor;
    // functor that creates new particles and initializes their parameters
    CopyTransformFunctor m_copy_transform_functor;

};

#endif // WARPX_PARTICLES_COLLISION_BINARYCOLLISION_H_
