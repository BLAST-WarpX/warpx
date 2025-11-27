/* Copyright 2020-2021 Yinjian Zhao, David Grote, Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_BINARYCOLLISION_H_
#define WARPX_PARTICLES_COLLISION_BINARYCOLLISION_H_

#include "Particles/Algorithms/KineticEnergy.H"
#include "Particles/Collision/BinaryCollision/Bremsstrahlung/BremsstrahlungFunc.H"
#include "Particles/Collision/BinaryCollision/Coulomb/PairWiseCoulombCollisionFunc.H"
#include "Particles/Collision/BinaryCollision/DSMC/DSMCFunc.H"
#include "Particles/Collision/BinaryCollision/NuclearFusion/NuclearFusionFunc.H"
#include "Particles/Collision/BinaryCollision/LinearBreitWheeler/LinearBreitWheelerCollisionFunc.H"
#include "Particles/Collision/BinaryCollision/LinearCompton/LinearComptonCollisionFunc.H"
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
        using namespace amrex::literals;
        if(m_species_names.size() != 2) {
            WARPX_ABORT_WITH_MESSAGE("Binary collision " + collision_name + " must have exactly two species.");
        }

        const CollisionType collision_type = BinaryCollisionUtils::get_collision_type(collision_name, mypc);

        m_isSameSpecies = (m_species_names[0] == m_species_names[1]);

        m_binary_collision_functor = CollisionFunctor(collision_name, mypc, m_isSameSpecies);

        m_use_global_debye_length = m_binary_collision_functor.use_global_debye_length();

        const amrex::ParmParse pp_collision_name(collision_name);
        pp_collision_name.queryarr("product_species", m_product_species);


        if (collision_type == CollisionType::PairwiseCoulomb) {
            // Input parameter set for all pairwise Coulomb collisions
            const amrex::ParmParse pp_collisions("collisions");
            pp_collisions.query("correct_energy_momentum", m_correct_energy_momentum);
            pp_collisions.query("beta_weight_exponent", m_beta_weight_exponent);
            pp_collisions.query("energy_fraction", m_energy_fraction);
            pp_collisions.query("energy_fraction_max", m_energy_fraction_max);

            // Input parameter set for this collision
            pp_collision_name.query("correct_energy_momentum", m_correct_energy_momentum);
            pp_collision_name.query("beta_weight_exponent", m_beta_weight_exponent);
            pp_collision_name.query("energy_fraction", m_energy_fraction);
            pp_collision_name.query("energy_fraction_max", m_energy_fraction_max);
        }

        // If DSMC the colliding species are also product species.
        // Therefore, we insert the colliding species at the beginning of `m_product_species`.
        if (collision_type == CollisionType::DSMC) {
            // If the scattering process is ionization ensure that the
            // explicitly specified "target" species, i.e., the species that
            // undergoes ionization, is second in the species list for this
            // collision set. The reason for this is that during the collision
            // operation, an outgoing particle of the first species type will
            // be created.
            std::string target_species;
            pp_collision_name.query("ionization_target_species", target_species);
            if (!target_species.empty()) {
                if (m_species_names[0] == target_species) {
                    std::swap(m_species_names[0], m_species_names[1]);
                } else if (m_species_names[1] != target_species) {
                    WARPX_ABORT_WITH_MESSAGE("DSMC: Ionization target species, " + target_species + " must be one of the colliding species.");
                }
            }

            m_product_species.insert( m_product_species.begin(), m_species_names.begin(), m_species_names.end() );
            // Note that, if a species is both a colliding species and a product species
            // (e.g., e- in e- + H -> 2 e- + H+) then it will be listed twice in `m_product_species`.
            // (e.g., m_product_species = [e-, H, e-, H+])
            // The code for ionization in `SplitAndScatterFunc` handles this case correctly.
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

        WARPX_PROFILE("BinaryCollision::doCollisionsWithinTile");

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

        bool const use_global_debye_length = m_use_global_debye_length;
        amrex::Real * global_debye_length_data = nullptr;
        if (use_global_debye_length) {
            WarpX & warpx = WarpX::GetInstance();
            amrex::MultiFab & global_debye_length = *warpx.m_fields.get(warpx::fields::FieldType::global_debye_length, lev);
            amrex::FArrayBox & global_debye_length_fab = global_debye_length[mfi];
            global_debye_length_data = global_debye_length_fab.dataPtr();
        }

        amrex::Geometry const& geom_lev = WarpX::GetInstance().Geom(lev);
        amrex::ParticleReal const dV = AMREX_D_TERM(geom_lev.CellSize(0), *geom_lev.CellSize(1), *geom_lev.CellSize(2));
#if defined(WARPX_DIM_RZ)
        amrex::Box const& cbx = mfi.tilebox(amrex::IntVect::TheZeroVector()); //Cell-centered box
        auto const lo = lbound(cbx);
        auto const hi = ubound(cbx);
        int const nz = hi.y - lo.y + 1;
#endif
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        auto const dr = geom_lev.CellSize(0);
#endif

        auto volume_factor = [=] AMREX_GPU_DEVICE(int i_cell) noexcept {
#if defined(WARPX_DIM_RZ)
            // Return the radial factor for the volume element, dV
            int const ri = (i_cell - i_cell%nz)/nz;
            // rr is radius at the cell center
            amrex::ParticleReal const rr = (ri + 0.5_prt)*dr;
            return 2.0_prt*static_cast<amrex::ParticleReal>(MathConst::pi)*rr;
#elif defined(WARPX_DIM_RCYLINDER)
            int const ri = i_cell;
            // rr is radius at the cell center
            amrex::ParticleReal const rr = (ri + 0.5_prt)*dr;
            return 2.0_prt*static_cast<amrex::ParticleReal>(MathConst::pi)*rr;
#elif defined(WARPX_DIM_RSPHERE)
            // This needs double checking
            int const ri = i_cell;
            // rr is radius at the cell center
            amrex::ParticleReal const rr = (ri + 0.5_prt)*dr;
            return 4.0_prt*static_cast<amrex::ParticleReal>(MathConst::pi)*rr*rr;
#else
            // No factor is needed for Cartesian
            amrex::ignore_unused(i_cell);
            return 1.0_prt;
#endif
        };

        if ( m_isSameSpecies ) // species_1 == species_2
        {
            // Extract particles in the tile that `mfi` points to
            ParticleTileType& ptile_1 = species_1.ParticlesAt(lev, mfi);

            // Find the particles that are in each cell of this tile
            WARPX_PROFILE_VAR("BinaryCollision::doCollisionsWithinTile::findParticlesInEachCell", prof_findParticlesInEachCell);
            ParticleBins bins_1 = findParticlesInEachCell( geom_lev, mfi, ptile_1 );
            WARPX_PROFILE_VAR_STOP(prof_findParticlesInEachCell);

            // Loop over cells, and collide the particles in each cell

            // Extract low-level data
            auto const n_cells = static_cast<int>(bins_1.numBins());
            // - Species 1
            auto np1 = ptile_1.numParticles();
            const auto soa_1 = ptile_1.getParticleTileData();
            index_type* AMREX_RESTRICT indices_1 = bins_1.permutationPtr();
            index_type const* AMREX_RESTRICT cell_offsets_1 = bins_1.offsetsPtr();
            index_type const* AMREX_RESTRICT bins_1_ptr = bins_1.binsPtr();
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
            WARPX_PROFILE_VAR("BinaryCollision::doCollisionsWithinTile::computeNumberOfPairs", prof_computeNumberOfPairs);
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
            WARPX_PROFILE_VAR_STOP(prof_computeNumberOfPairs);

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
            // Extra data needed when products are created
            int const n_product_data = (binary_collision_functor.m_need_product_data ? n_total_pairs : 0);
            amrex::Gpu::DeviceVector<amrex::ParticleReal> product_data(n_product_data);
            amrex::ParticleReal* AMREX_RESTRICT p_product_data = product_data.dataPtr();
            /*
              End of calculations only required when creating product particles
            */

            amrex::ParticleReal * const AMREX_RESTRICT u1x = soa_1.m_rdata[PIdx::ux];
            amrex::ParticleReal * const AMREX_RESTRICT u1y = soa_1.m_rdata[PIdx::uy];
            amrex::ParticleReal * const AMREX_RESTRICT u1z = soa_1.m_rdata[PIdx::uz];
            amrex::ParticleReal * const AMREX_RESTRICT w1 = soa_1.m_rdata[PIdx::w];

            // create vectors to store density and temperature on cell level
            amrex::Gpu::DeviceVector<amrex::ParticleReal> n1_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> T1_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> vx1_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> vy1_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> vz1_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> vs1_vec;

            if (binary_collision_functor.m_computeSpeciesDensities) {
                n1_vec.resize(n_cells, 0.0_prt);
            }
            if (binary_collision_functor.m_computeSpeciesTemperatures) {
                T1_vec.resize(n_cells, 0.0_prt);
                vx1_vec.resize(n_cells, 0.0_prt);
                vy1_vec.resize(n_cells, 0.0_prt);
                vz1_vec.resize(n_cells, 0.0_prt);
                vs1_vec.resize(n_cells, 0.0_prt);
            }
            amrex::ParticleReal* AMREX_RESTRICT n1_in_each_cell = n1_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT T1_in_each_cell = T1_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT vx1_in_each_cell = vx1_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT vy1_in_each_cell = vy1_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT vz1_in_each_cell = vz1_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT vs1_in_each_cell = vs1_vec.dataPtr();

            // Create vectors to store energy and momentum in each cell.
            // This is used to correct the energy and momentum in the
            // cell after the collisions.
            amrex::Gpu::DeviceVector<amrex::ParticleReal> ww_weighted_sum_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> KE_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> px_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> py_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> pz_vec;
            if (m_correct_energy_momentum) {
                ww_weighted_sum_vec.resize(n_cells, 0.0_prt);
                KE_vec.resize(n_cells, 0.0_prt);
                px_vec.resize(n_cells, 0.0_prt);
                py_vec.resize(n_cells, 0.0_prt);
                pz_vec.resize(n_cells, 0.0_prt);
            }
            amrex::ParticleReal* AMREX_RESTRICT ww_weighted_sum_in_each_cell = ww_weighted_sum_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT KE_in_each_cell = KE_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT px_in_each_cell = px_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT py_in_each_cell = py_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT pz_in_each_cell = pz_vec.dataPtr();

            // Save the momentum before the collision since sometimes the correction fails
            // and the only solution is to restore the pre-collision values.
            // The implicit solver already has ux_n etc so use those if available,
            // otherwise create temporaries.
            amrex::Gpu::DeviceVector<amrex::ParticleReal> u1x_before;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> u1y_before;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> u1z_before;
            amrex::ParticleReal* AMREX_RESTRICT u1x_before_ptr = nullptr;
            amrex::ParticleReal* AMREX_RESTRICT u1y_before_ptr = nullptr;
            amrex::ParticleReal* AMREX_RESTRICT u1z_before_ptr = nullptr;
            if (m_correct_energy_momentum) {
                // Check if ux_n was added.
                std::vector<std::string> const & real_names1 = species_1.GetRealSoANames();
                auto const pos1 = std::find(real_names1.begin(), real_names1.end(), "ux_n");
                if (pos1 != real_names1.end()) {
                    int const u1x_ni = std::distance(real_names1.begin(), pos1);
                    u1x_before_ptr = soa_1.m_runtime_rdata[u1x_ni];
                    u1y_before_ptr = soa_1.m_runtime_rdata[u1x_ni + 1];
                    u1z_before_ptr = soa_1.m_runtime_rdata[u1x_ni + 2];
                } else {
                    u1x_before.resize(np1);
                    u1y_before.resize(np1);
                    u1z_before.resize(np1);
                    u1x_before_ptr = u1x_before.dataPtr();
                    u1y_before_ptr = u1y_before.dataPtr();
                    u1z_before_ptr = u1z_before.dataPtr();
                }

            }

            amrex::ParticleReal const beta_weight_exponent = m_beta_weight_exponent;

            WARPX_PROFILE_VAR("BinaryCollision::doCollisionsWithinTile::findDensityTemperatures", prof_findDensityTemperatures);

            if (m_correct_energy_momentum ||
                binary_collision_functor.m_computeSpeciesDensities ||
                binary_collision_functor.m_computeSpeciesTemperatures) {

                bool const correct_energy_momentum = m_correct_energy_momentum;

                WARPX_PROFILE_VAR("BinaryCollision::doCollisionsWithinTile::findDensityTemperatures::atomics", prof_findDensityTemperatures_atomics);
                // Loop over particles and compute quantities needed for energy conservation and the collsion parameters
                amrex::ParallelFor( np1,
                    [=] AMREX_GPU_DEVICE (int ip) noexcept
                    {
                        if (correct_energy_momentum) {
                            u1x_before_ptr[ip] = u1x[ip];
                            u1y_before_ptr[ip] = u1y[ip];
                            u1z_before_ptr[ip] = u1z[ip];

                            const int i_cell = bins_1_ptr[ip];

                            // Compute the local energy and momentum.
                            amrex::ParticleReal const w1_scaled = std::pow(w1[ip], beta_weight_exponent);
                            amrex::Gpu::Atomic::AddNoRet(&ww_weighted_sum_in_each_cell[i_cell], w1_scaled);
                            amrex::Gpu::Atomic::AddNoRet(&px_in_each_cell[i_cell], w1[ip]*u1x[ip]*m1);
                            amrex::Gpu::Atomic::AddNoRet(&py_in_each_cell[i_cell], w1[ip]*u1y[ip]*m1);
                            amrex::Gpu::Atomic::AddNoRet(&pz_in_each_cell[i_cell], w1[ip]*u1z[ip]*m1);
                            amrex::ParticleReal KE = Algorithms::KineticEnergy(u1x[ip], u1y[ip], u1z[ip], m1);
                            amrex::Gpu::Atomic::AddNoRet(&KE_in_each_cell[i_cell], w1[ip]*KE);
                        }

                        // compute local density [1/m^3]
                        if (binary_collision_functor.m_computeSpeciesDensities) {
                            amrex::Gpu::Atomic::AddNoRet(&n1_in_each_cell[bins_1_ptr[ip]],
                                                         w1[ip]/(dV*volume_factor(bins_1_ptr[ip])));
                        }

                        // compute local temperature [Joules]
                        if (binary_collision_functor.m_computeSpeciesTemperatures) {
                            amrex::ParticleReal us = u1x[ip]*u1x[ip] + u1y[ip]*u1y[ip] + u1z[ip]*u1z[ip];
                            amrex::ParticleReal constexpr inv_c2 = 1.0_prt / ( PhysConst::c * PhysConst::c );
                            amrex::ParticleReal gm = std::sqrt( 1.0_prt + us*inv_c2);
                            amrex::Gpu::Atomic::AddNoRet(&T1_in_each_cell[bins_1_ptr[ip]],  w1[ip]);
                            amrex::Gpu::Atomic::AddNoRet(&vx1_in_each_cell[bins_1_ptr[ip]], w1[ip]*u1x[ip]/gm);
                            amrex::Gpu::Atomic::AddNoRet(&vy1_in_each_cell[bins_1_ptr[ip]], w1[ip]*u1y[ip]/gm);
                            amrex::Gpu::Atomic::AddNoRet(&vz1_in_each_cell[bins_1_ptr[ip]], w1[ip]*u1z[ip]/gm);
                            amrex::Gpu::Atomic::AddNoRet(&vs1_in_each_cell[bins_1_ptr[ip]], w1[ip]*us/gm/gm);
                        }
                    }
                );
                WARPX_PROFILE_VAR_STOP(prof_findDensityTemperatures_atomics);

            }

            // Finish temperature calculation
            WARPX_PROFILE_VAR("BinaryCollision::doCollisionsWithinTile::findDensityTemperatures::finishTemperature", prof_findDensityTemperatures_finish);
            amrex::ParallelFor( n_cells, [=] AMREX_GPU_DEVICE (int i_cell) noexcept
                {
                    // The particles from species1 that are in the cell `i_cell` are
                    // given by the `indices_1[cell_start_1:cell_stop_1]`
                    index_type const cell_start_1 = cell_offsets_1[i_cell];
                    index_type const cell_stop_1  = cell_offsets_1[i_cell+1];

                    // Do not need if there is only one particle in the cell
                    if ( cell_stop_1 - cell_start_1 <= 1 ) { return; }

                    // finish temperature calculation if needed
                    if (binary_collision_functor.m_computeSpeciesTemperatures) {
                        const amrex::ParticleReal invsum = 1._prt/T1_in_each_cell[i_cell];
                        auto vx1 = vx1_in_each_cell[i_cell] * invsum;
                        auto vy1 = vy1_in_each_cell[i_cell] * invsum;
                        auto vz1 = vz1_in_each_cell[i_cell] * invsum;
                        auto vs1 = vs1_in_each_cell[i_cell] * invsum;

                        T1_in_each_cell[i_cell] = m1/(3._prt)*(vs1 -(vx1*vx1+vy1*vy1+vz1*vz1));
                    }
                }
            );
            WARPX_PROFILE_VAR_STOP(prof_findDensityTemperatures_finish);

            // shuffle
            WARPX_PROFILE_VAR("BinaryCollision::doCollisionsWithinTile::findDensityTemperatures::shuffle", prof_findDensityTemperatures_shuffle);
            amrex::ParallelForRNG( n_cells,
                [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
                {
                    // The particles from species1 that are in the cell `i_cell` are
                    // given by the `indices_1[cell_start_1:cell_stop_1]`
                    index_type const cell_start_1 = cell_offsets_1[i_cell];
                    index_type const cell_stop_1  = cell_offsets_1[i_cell+1];

                    // Do not shuffle if there is only one particle in the cell
                    if ( cell_stop_1 - cell_start_1 <= 1 ) { return; }

                    ShuffleFisherYates(indices_1, cell_start_1, cell_stop_1, engine);
                }
            );
            WARPX_PROFILE_VAR_STOP(prof_findDensityTemperatures_shuffle);
            WARPX_PROFILE_VAR_STOP(prof_findDensityTemperatures);

            // Loop over independent particle pairs
            // To speed up binary collisions on GPU, we try to expose as much parallelism
            // as possible (while avoiding race conditions): Instead of looping with one GPU
            // thread per cell, we loop with one GPU thread per "independent pairs" (i.e. pairs
            // that do not touch the same macroparticles, so that there is no race condition),
            // where the number of independent pairs is determined by the lower number of
            // macroparticles of either species, within each cell.
            WARPX_PROFILE_VAR("BinaryCollision::doCollisionsWithinTile::LoopOverCollisions", prof_loopOverCollisions);
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

                    amrex::Real global_lamdb = 0.;
                    if (use_global_debye_length) {
                        global_lamdb = global_debye_length_data[i_cell];
                    }

                    // Call the function in order to perform collisions
                    // If there are product species, mask, p_pair_indices_1/2, and
                    // p_pair_reaction_weight and p_product_data are filled here
                    binary_collision_functor(
                        cell_start_1, cell_half_1,
                        cell_half_1, cell_stop_1,
                        indices_1, indices_1,
                        soa_1, soa_1, get_position_1, get_position_1,
                        n1, n1, T1, T1, global_lamdb,
                        q1, q1, m1, m1, dt, dV*volume_factor(i_cell), coll_idx,
                        cell_start_pair, p_mask, p_pair_indices_1, p_pair_indices_2,
                        p_pair_reaction_weight, p_product_data, engine);
                }
            );
            WARPX_PROFILE_VAR_STOP(prof_loopOverCollisions);
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
                                                    p_pair_reaction_weight, p_product_data);

            for (int i = 0; i < n_product_species; i++)
            {
                setNewParticleIDs(*(tile_products_data[i]), static_cast<int>(products_np[i]), num_added[i]);
            }

            if (m_correct_energy_momentum) {
                // Loop over cells and calculate the change in energy and momentum.
                amrex::ParticleReal const energy_fraction = m_energy_fraction;
                amrex::ParticleReal const energy_fraction_max = m_energy_fraction_max;

                WARPX_PROFILE_VAR("BinaryCollision::doCollisionsWithinTile::correctEnergyMomentum", prof_correctEnergyMomentum);
                amrex::ParallelFor( np1,
                    [=] AMREX_GPU_DEVICE (int i1) noexcept
                    {

                        const int i_cell = bins_1_ptr[i1];

                        // Subract the momentum from the initial values.
                        // Whatever is left over, will be distributed among the particles.
                        amrex::ParticleReal const px = w1[i1]*u1x[i1]*m1;
                        amrex::ParticleReal const py = w1[i1]*u1y[i1]*m1;
                        amrex::ParticleReal const pz = w1[i1]*u1z[i1]*m1;
                        amrex::Gpu::Atomic::AddNoRet(&px_in_each_cell[i_cell], -px);
                        amrex::Gpu::Atomic::AddNoRet(&py_in_each_cell[i_cell], -py);
                        amrex::Gpu::Atomic::AddNoRet(&pz_in_each_cell[i_cell], -pz);
                    }
                );

                amrex::ParallelFor( np1,
                    [=] AMREX_GPU_DEVICE (int i1) noexcept
                    {

                        const int i_cell = bins_1_ptr[i1];

                        amrex::ParticleReal const ww_weighted_sum = ww_weighted_sum_in_each_cell[i_cell];
                        amrex::ParticleReal const w_factor = std::pow(w1[i1], beta_weight_exponent - 1._prt)/m1;
                        u1x[i1] += w_factor*px_in_each_cell[i_cell]/ww_weighted_sum;
                        u1y[i1] += w_factor*py_in_each_cell[i_cell]/ww_weighted_sum;
                        u1z[i1] += w_factor*pz_in_each_cell[i_cell]/ww_weighted_sum;

                        amrex::ParticleReal const KE_after = Algorithms::KineticEnergy(u1x[i1], u1y[i1], u1z[i1], m1);
                        amrex::Gpu::Atomic::AddNoRet(&KE_in_each_cell[i_cell], -w1[i1]*KE_after);
                    }
                );

                amrex::Gpu::Buffer<amrex::Long> failed_corrections({0});
                amrex::Gpu::Buffer<amrex::ParticleReal> remaining_energy({0.});
                amrex::Long* failed_corrections_ptr = failed_corrections.data();
                amrex::ParticleReal* remaining_energy_ptr = remaining_energy.data();

                amrex::ParallelFor( n_cells,
                    [=] AMREX_GPU_DEVICE (int i_cell) noexcept
                    {

                        index_type const cell_start_1 = cell_offsets_1[i_cell];
                        index_type const cell_stop_1  = cell_offsets_1[i_cell+1];
                        amrex::ParticleReal deltaEp1 = -KE_in_each_cell[i_cell];

                        if (deltaEp1 != 0.) {
                            // Adjust species 1 particles to absorb deltaEp1.
                            bool correction_failed =
                                 ParticleUtils::ModifyEnergyPairwise(u1x, u1y, u1z, w1, indices_1,
                                                                     cell_start_1, cell_stop_1, m1,
                                                                     energy_fraction, energy_fraction_max, deltaEp1);
                            if (correction_failed) {
                                // If the correction failed, give up on the collision and restore the
                                // momentum to what it was beforehand.
                                for (index_type i1=cell_start_1; i1<cell_stop_1; ++i1) {
                                    u1x[ indices_1[i1] ] = u1x_before_ptr[ indices_1[i1] ];
                                    u1y[ indices_1[i1] ] = u1y_before_ptr[ indices_1[i1] ];
                                    u1z[ indices_1[i1] ] = u1z_before_ptr[ indices_1[i1] ];
                                }
                                amrex::Gpu::Atomic::Add(failed_corrections_ptr, amrex::Long(1));
                                amrex::Gpu::Atomic::Add(remaining_energy_ptr, deltaEp1);
                            }
                        }

                    }
                );

                amrex::Long const num_failed_corrections = *(failed_corrections.copyToHost());
                amrex::ParticleReal const total_remaining_energy = *(remaining_energy.copyToHost());
                if (num_failed_corrections > 0) {
                    ablastr::warn_manager::WMRecordWarning("BinaryCollision::doCollisionsWithinTile",
                        "The energy correction failed for " + std::to_string(num_failed_corrections) + " cells " +
                        "for Coulomb collisions for species " + m_species_names[0] + ". " +
                        "The remaining energy is " + std::to_string(total_remaining_energy/PhysConst::q_e) + " eV. " +
                        "The collisions in those cells was cancelled.");
                }

                WARPX_PROFILE_VAR_STOP(prof_correctEnergyMomentum);
            }

        }
        else // species_1 != species_2
        {
            // Extract particles in the tile that `mfi` points to
            ParticleTileType& ptile_1 = species_1.ParticlesAt(lev, mfi);
            ParticleTileType& ptile_2 = species_2.ParticlesAt(lev, mfi);

            // Find the particles that are in each cell of this tile
            WARPX_PROFILE_VAR("BinaryCollision::doCollisionsWithinTile::findParticlesInEachCell", prof_findParticlesInEachCell);
            ParticleBins bins_1 = findParticlesInEachCell( geom_lev, mfi, ptile_1 );
            ParticleBins bins_2 = findParticlesInEachCell( geom_lev, mfi, ptile_2 );
            WARPX_PROFILE_VAR_STOP(prof_findParticlesInEachCell);

            // Loop over cells, and collide the particles in each cell

            // Extract low-level data
            auto const n_cells = static_cast<int>(bins_1.numBins());
            // - Species 1
            auto np1 = ptile_1.numParticles();
            const auto soa_1 = ptile_1.getParticleTileData();
            index_type* AMREX_RESTRICT indices_1 = bins_1.permutationPtr();
            index_type const* AMREX_RESTRICT cell_offsets_1 = bins_1.offsetsPtr();
            index_type const* AMREX_RESTRICT bins_1_ptr = bins_1.binsPtr();
            const amrex::ParticleReal q1 = species_1.getCharge();
            const amrex::ParticleReal m1 = species_1.getMass();
            auto get_position_1  = GetParticlePosition<PIdx>(ptile_1, getpos_offset);
            // - Species 2
            auto np2 = ptile_2.numParticles();
            const auto soa_2 = ptile_2.getParticleTileData();
            index_type* AMREX_RESTRICT indices_2 = bins_2.permutationPtr();
            index_type const* AMREX_RESTRICT cell_offsets_2 = bins_2.offsetsPtr();
            index_type const* AMREX_RESTRICT bins_2_ptr = bins_2.binsPtr();
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
            WARPX_PROFILE_VAR("BinaryCollision::doCollisionsWithinTile::computeNumberOfPairs", prof_computeNumberOfPairs);
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
            WARPX_PROFILE_VAR_STOP(prof_computeNumberOfPairs);

            // mask: equal to 1 if particle creation occurs for a given pair, 0 otherwise
            amrex::Gpu::DeviceVector<index_type> mask(n_total_pairs, index_type(0));
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
            // Extra data needed when products are created
            int const n_product_data = (binary_collision_functor.m_need_product_data ? n_total_pairs : 0);
            amrex::Gpu::DeviceVector<amrex::ParticleReal> product_data(n_product_data);
            amrex::ParticleReal* AMREX_RESTRICT p_product_data = product_data.dataPtr();
            /*
              End of calculations only required when creating product particles
            */

            amrex::ParticleReal * const AMREX_RESTRICT w1 = soa_1.m_rdata[PIdx::w];
            amrex::ParticleReal * const AMREX_RESTRICT u1x = soa_1.m_rdata[PIdx::ux];
            amrex::ParticleReal * const AMREX_RESTRICT u1y = soa_1.m_rdata[PIdx::uy];
            amrex::ParticleReal * const AMREX_RESTRICT u1z = soa_1.m_rdata[PIdx::uz];

            amrex::ParticleReal * const AMREX_RESTRICT w2 = soa_2.m_rdata[PIdx::w];
            amrex::ParticleReal * const AMREX_RESTRICT u2x = soa_2.m_rdata[PIdx::ux];
            amrex::ParticleReal * const AMREX_RESTRICT u2y = soa_2.m_rdata[PIdx::uy];
            amrex::ParticleReal * const AMREX_RESTRICT u2z = soa_2.m_rdata[PIdx::uz];

            // create vectors to store density and temperature on cell level
            amrex::Gpu::DeviceVector<amrex::ParticleReal> n1_vec, n2_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> T1_vec, T2_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> vx1_vec, vx2_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> vy1_vec, vy2_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> vz1_vec, vz2_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> vs1_vec, vs2_vec;

            if (binary_collision_functor.m_computeSpeciesDensities) {
                n1_vec.resize(n_cells, 0.0_prt);
                n2_vec.resize(n_cells, 0.0_prt);
            }
            if (binary_collision_functor.m_computeSpeciesTemperatures) {
                T1_vec.resize(n_cells, 0.0_prt);
                T2_vec.resize(n_cells, 0.0_prt);
                vx1_vec.resize(n_cells, 0.0_prt);
                vx2_vec.resize(n_cells, 0.0_prt);
                vy1_vec.resize(n_cells, 0.0_prt);
                vy2_vec.resize(n_cells, 0.0_prt);
                vz1_vec.resize(n_cells, 0.0_prt);
                vz2_vec.resize(n_cells, 0.0_prt);
                vs1_vec.resize(n_cells, 0.0_prt);
                vs2_vec.resize(n_cells, 0.0_prt);
            }
            amrex::ParticleReal* AMREX_RESTRICT n1_in_each_cell = n1_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT n2_in_each_cell = n2_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT T1_in_each_cell = T1_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT T2_in_each_cell = T2_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT vx1_in_each_cell = vx1_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT vx2_in_each_cell = vx2_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT vy1_in_each_cell = vy1_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT vy2_in_each_cell = vy2_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT vz1_in_each_cell = vz1_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT vz2_in_each_cell = vz2_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT vs1_in_each_cell = vs1_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT vs2_in_each_cell = vs2_vec.dataPtr();

            // Create vectors to store energy and momentum in each cell.
            // This is used to correct the energy and momentum in the
            // cell after the collisions.
            amrex::Gpu::DeviceVector<amrex::ParticleReal> ww_weighted_sum_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> KE_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> px_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> py_vec;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> pz_vec;
            if (m_correct_energy_momentum) {
                ww_weighted_sum_vec.resize(n_cells, 0.0_prt);
                KE_vec.resize(n_cells, 0.0_prt);
                px_vec.resize(n_cells, 0.0_prt);
                py_vec.resize(n_cells, 0.0_prt);
                pz_vec.resize(n_cells, 0.0_prt);
            }
            amrex::ParticleReal* AMREX_RESTRICT ww_weighted_sum_in_each_cell = ww_weighted_sum_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT KE_in_each_cell = KE_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT px_in_each_cell = px_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT py_in_each_cell = py_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT pz_in_each_cell = pz_vec.dataPtr();

            // Save the momentum before the collision since sometimes the correction fails
            // and the only solution is to restore the pre-collision values.
            // The implicit solver already has ux_n etc so use those if available,
            // otherwise create temporaries.
            amrex::Gpu::DeviceVector<amrex::ParticleReal> u1x_before;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> u1y_before;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> u1z_before;
            amrex::ParticleReal* AMREX_RESTRICT u1x_before_ptr = nullptr;
            amrex::ParticleReal* AMREX_RESTRICT u1y_before_ptr = nullptr;
            amrex::ParticleReal* AMREX_RESTRICT u1z_before_ptr = nullptr;
            if (m_correct_energy_momentum) {
                // Check if ux_n was added.
                std::vector<std::string> const & real_names1 = species_1.GetRealSoANames();
                auto const pos1 = std::find(real_names1.begin(), real_names1.end(), "ux_n");
                if (pos1 != real_names1.end()) {
                    int const u1x_ni = std::distance(real_names1.begin(), pos1);
                    u1x_before_ptr = soa_1.m_runtime_rdata[u1x_ni];
                    u1y_before_ptr = soa_1.m_runtime_rdata[u1x_ni + 1];
                    u1z_before_ptr = soa_1.m_runtime_rdata[u1x_ni + 2];
                } else {
                    u1x_before.resize(np1);
                    u1y_before.resize(np1);
                    u1z_before.resize(np1);
                    u1x_before_ptr = u1x_before.dataPtr();
                    u1y_before_ptr = u1y_before.dataPtr();
                    u1z_before_ptr = u1z_before.dataPtr();
                }
            }

            amrex::Gpu::DeviceVector<amrex::ParticleReal> u2x_before;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> u2y_before;
            amrex::Gpu::DeviceVector<amrex::ParticleReal> u2z_before;
            amrex::ParticleReal* AMREX_RESTRICT u2x_before_ptr = nullptr;
            amrex::ParticleReal* AMREX_RESTRICT u2y_before_ptr = nullptr;
            amrex::ParticleReal* AMREX_RESTRICT u2z_before_ptr = nullptr;
            if (m_correct_energy_momentum) {
                // Check if ux_n was added.
                std::vector<std::string> const & real_names2 = species_2.GetRealSoANames();
                auto const pos2 = std::find(real_names2.begin(), real_names2.end(), "ux_n");
                if (pos2 != real_names2.end()) {
                    int const u2x_ni = std::distance(real_names2.begin(), pos2);
                    u2x_before_ptr = soa_2.m_runtime_rdata[u2x_ni];
                    u2y_before_ptr = soa_2.m_runtime_rdata[u2x_ni + 1];
                    u2z_before_ptr = soa_2.m_runtime_rdata[u2x_ni + 2];
                } else {
                    u2x_before.resize(np2);
                    u2y_before.resize(np2);
                    u2z_before.resize(np2);
                    u2x_before_ptr = u2x_before.dataPtr();
                    u2y_before_ptr = u2y_before.dataPtr();
                    u2z_before_ptr = u2z_before.dataPtr();
                }
            }

            amrex::ParticleReal const beta_weight_exponent = m_beta_weight_exponent;

            WARPX_PROFILE_VAR("BinaryCollision::doCollisionsWithinTile::findDensityTemperatures", prof_findDensityTemperatures);

            if (m_correct_energy_momentum ||
                binary_collision_functor.m_computeSpeciesDensities ||
                binary_collision_functor.m_computeSpeciesTemperatures) {

                bool const correct_energy_momentum = m_correct_energy_momentum;

                // Loop over particles and compute quantities needed for energy conservation and the collsion parameters
                WARPX_PROFILE_VAR("BinaryCollision::doCollisionsWithinTile::findDensityTemperatures::atomics", prof_findDensityTemperatures_atomics);
                amrex::ParallelFor( std::max(np1, np2),
                    [=] AMREX_GPU_DEVICE (int ip) noexcept
                    {
                        if (correct_energy_momentum) {
                            if (ip < np1) {
                                const int i1 = ip;
                                const int i_cell = bins_1_ptr[i1];

                                // Save the momenta before the collision
                                u1x_before_ptr[i1] = u1x[i1];
                                u1y_before_ptr[i1] = u1y[i1];
                                u1z_before_ptr[i1] = u1z[i1];

                                // Compute the local energy and momentum.
                                amrex::ParticleReal const w1_scaled = std::pow(w1[i1], beta_weight_exponent);
                                amrex::Gpu::Atomic::AddNoRet(&ww_weighted_sum_in_each_cell[i_cell], w1_scaled);
                                amrex::Gpu::Atomic::AddNoRet(&px_in_each_cell[i_cell], w1[i1]*u1x[i1]*m1);
                                amrex::Gpu::Atomic::AddNoRet(&py_in_each_cell[i_cell], w1[i1]*u1y[i1]*m1);
                                amrex::Gpu::Atomic::AddNoRet(&pz_in_each_cell[i_cell], w1[i1]*u1z[i1]*m1);
                                amrex::ParticleReal KE1 = Algorithms::KineticEnergy(u1x[i1], u1y[i1], u1z[i1], m1);
                                amrex::Gpu::Atomic::AddNoRet(&KE_in_each_cell[i_cell], w1[i1]*KE1);
                            }

                            if (ip < np2) {
                                const int i2 = ip;
                                const int i_cell = bins_2_ptr[i2];

                                // Save the momenta before the collision
                                u2x_before_ptr[i2] = u2x[i2];
                                u2y_before_ptr[i2] = u2y[i2];
                                u2z_before_ptr[i2] = u2z[i2];

                                // Compute the local energy and momentum.
                                amrex::ParticleReal const w2_scaled = std::pow(w2[i2], beta_weight_exponent);
                                amrex::Gpu::Atomic::AddNoRet(&ww_weighted_sum_in_each_cell[i_cell], w2_scaled);
                                amrex::Gpu::Atomic::AddNoRet(&px_in_each_cell[i_cell], w2[i2]*u2x[i2]*m2);
                                amrex::Gpu::Atomic::AddNoRet(&py_in_each_cell[i_cell], w2[i2]*u2y[i2]*m2);
                                amrex::Gpu::Atomic::AddNoRet(&pz_in_each_cell[i_cell], w2[i2]*u2z[i2]*m2);
                                amrex::ParticleReal KE2 = Algorithms::KineticEnergy(u2x[i2], u2y[i2], u2z[i2], m2);
                                amrex::Gpu::Atomic::AddNoRet(&KE_in_each_cell[i_cell], w2[i2]*KE2);
                            }
                        }

                        // compute local densities [1/m^3]
                        if (binary_collision_functor.m_computeSpeciesDensities) {
                            if (ip < np1) {
                                amrex::Gpu::Atomic::AddNoRet(&n1_in_each_cell[bins_1_ptr[ip]],
                                                             w1[ip]/(dV*volume_factor(bins_1_ptr[ip])));
                            }
                            if (ip < np2) {
                                amrex::Gpu::Atomic::AddNoRet(&n2_in_each_cell[bins_2_ptr[ip]],
                                                             w2[ip]/(dV*volume_factor(bins_2_ptr[ip])));
                            }
                        }

                        // compute local temperatures [Joules]
                        if (binary_collision_functor.m_computeSpeciesTemperatures) {
                            if (ip < np1) {
                                amrex::ParticleReal us = u1x[ip]*u1x[ip] + u1y[ip]*u1y[ip] + u1z[ip]*u1z[ip];
                                amrex::ParticleReal constexpr inv_c2 = 1.0_prt / ( PhysConst::c * PhysConst::c );
                                amrex::ParticleReal gm = std::sqrt( 1.0_prt + us*inv_c2);
                                amrex::Gpu::Atomic::AddNoRet(&T1_in_each_cell[bins_1_ptr[ip]],  w1[ip]);
                                amrex::Gpu::Atomic::AddNoRet(&vx1_in_each_cell[bins_1_ptr[ip]], w1[ip]*u1x[ip]/gm);
                                amrex::Gpu::Atomic::AddNoRet(&vy1_in_each_cell[bins_1_ptr[ip]], w1[ip]*u1y[ip]/gm);
                                amrex::Gpu::Atomic::AddNoRet(&vz1_in_each_cell[bins_1_ptr[ip]], w1[ip]*u1z[ip]/gm);
                                amrex::Gpu::Atomic::AddNoRet(&vs1_in_each_cell[bins_1_ptr[ip]], w1[ip]*us/gm/gm);
                            }
                            if (ip < np2) {
                                amrex::ParticleReal us = u2x[ip]*u2x[ip] + u2y[ip]*u2y[ip] + u2z[ip]*u2z[ip];
                                amrex::ParticleReal constexpr inv_c2 = 1.0_prt / ( PhysConst::c * PhysConst::c );
                                amrex::ParticleReal gm = std::sqrt( 1.0_prt + us*inv_c2);
                                amrex::Gpu::Atomic::AddNoRet(&T2_in_each_cell [bins_2_ptr[ip]], w2[ip]);
                                amrex::Gpu::Atomic::AddNoRet(&vx2_in_each_cell[bins_2_ptr[ip]], w2[ip]*u2x[ip]/gm);
                                amrex::Gpu::Atomic::AddNoRet(&vy2_in_each_cell[bins_2_ptr[ip]], w2[ip]*u2y[ip]/gm);
                                amrex::Gpu::Atomic::AddNoRet(&vz2_in_each_cell[bins_2_ptr[ip]], w2[ip]*u2z[ip]/gm);
                                amrex::Gpu::Atomic::AddNoRet(&vs2_in_each_cell[bins_2_ptr[ip]], w2[ip]*us/gm/gm);
                            }
                        }
                    }
                );
                WARPX_PROFILE_VAR_STOP(prof_findDensityTemperatures_atomics);

            }

            // Finish temperature calculation - we launch 2*n_cells threads to compute both species simultaneously
            WARPX_PROFILE_VAR("BinaryCollision::doCollisionsWithinTile::findDensityTemperatures::finishTemperature", prof_findDensityTemperatures_finish);
            amrex::ParallelFor( 2*n_cells, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                int i_cell = i < n_cells ? i : i - n_cells;

                    // The particles from species1 that are in the cell `i_cell` are
                    // given by the `indices_1[cell_start_1:cell_stop_1]`
                    index_type const cell_start_1 = cell_offsets_1[i_cell];
                    index_type const cell_stop_1  = cell_offsets_1[i_cell+1];

                    // Same for species 2
                    index_type const cell_start_2 = cell_offsets_2[i_cell];
                    index_type const cell_stop_2  = cell_offsets_2[i_cell+1];

                    // Do not need if one species is missing in the cell
                    if ( cell_stop_1 - cell_start_1 < 1 ||
                         cell_stop_2 - cell_start_2 < 1 ) { return; }

                    // finish temperature calculation if needed
                    if (binary_collision_functor.m_computeSpeciesTemperatures) {
              if (i < n_cells) {
                        const amrex::ParticleReal invsum1 = 1._prt/T1_in_each_cell[i_cell];
                        auto vx1 = vx1_in_each_cell[i_cell] * invsum1;
                        auto vy1 = vy1_in_each_cell[i_cell] * invsum1;
                        auto vz1 = vz1_in_each_cell[i_cell] * invsum1;
                        auto vs1 = vs1_in_each_cell[i_cell] * invsum1;

                        T1_in_each_cell[i_cell] = m1/(3._prt)*(vs1 -(vx1*vx1+vy1*vy1+vz1*vz1));
              } else {
                        const amrex::ParticleReal invsum2 = 1._prt/T2_in_each_cell[i_cell];
                        auto vx2 = vx2_in_each_cell[i_cell] * invsum2;
                        auto vy2 = vy2_in_each_cell[i_cell] * invsum2;
                        auto vz2 = vz2_in_each_cell[i_cell] * invsum2;
                        auto vs2 = vs2_in_each_cell[i_cell] * invsum2;

                        T2_in_each_cell[i_cell] = m2/(3._prt)*(vs2 -(vx2*vx2+vy2*vy2+vz2*vz2));
              }
                    }
                }
            );
            WARPX_PROFILE_VAR_STOP(prof_findDensityTemperatures_finish);

            // shuffle - we launch 2*n_cells threads to compute both species simultaneously
        WARPX_PROFILE_VAR("BinaryCollision::doCollisionsWithinTile::findDensityTemperatures::shuffle", prof_findDensityTemperatures_shuffle);
            amrex::ParallelForRNG( 2*n_cells,
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
            ShuffleFisherYates(indices_1, cell_start_1, cell_stop_1, engine);
          } else {
            ShuffleFisherYates(indices_2, cell_start_2, cell_stop_2, engine);
          }
                }
            );
            WARPX_PROFILE_VAR_STOP(prof_findDensityTemperatures_shuffle);

        WARPX_PROFILE_VAR_STOP(prof_findDensityTemperatures);
            // Loop over independent particle pairs
            // To speed up binary collisions on GPU, we try to expose as much parallelism
            // as possible (while avoiding race conditions): Instead of looping with one GPU
            // thread per cell, we loop with one GPU thread per "independent pairs" (i.e. pairs
            // that do not touch the same macroparticles, so that there is no race condition),
            // where the number of independent pairs is determined by the lower number of
            // macroparticles of either species, within each cell.
            WARPX_PROFILE_VAR("BinaryCollision::doCollisionsWithinTile::LoopOverCollisions", prof_loopOverCollisions);
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

                    amrex::Real global_lamdb = 0.;
                    if (use_global_debye_length) {
                        global_lamdb = global_debye_length_data[i_cell];
                    }

                    // Call the function in order to perform collisions
                    // If there are product species, p_mask, p_pair_indices_1/2, and
                    // p_pair_reaction_weight and p_product_data are filled here
                    binary_collision_functor(
                        cell_start_1, cell_stop_1, cell_start_2, cell_stop_2,
                        indices_1, indices_2,
                        soa_1, soa_2, get_position_1, get_position_2,
                        n1, n2, T1, T2, global_lamdb,
                        q1, q2, m1, m2, dt, dV*volume_factor(i_cell), coll_idx,
                        cell_start_pair, p_mask, p_pair_indices_1, p_pair_indices_2,
                        p_pair_reaction_weight, p_product_data, engine);
                }
            );
            WARPX_PROFILE_VAR_STOP(prof_loopOverCollisions);

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
                                                    p_pair_reaction_weight, p_product_data);

            for (int i = 0; i < n_product_species; i++)
            {
                setNewParticleIDs(*(tile_products_data[i]), static_cast<int>(products_np[i]), num_added[i]);
            }

            if (m_correct_energy_momentum) {
                // Loop over cells and calculate the change in energy and momentum.
                amrex::ParticleReal const energy_fraction = m_energy_fraction;
                amrex::ParticleReal const energy_fraction_max = m_energy_fraction_max;

                WARPX_PROFILE_VAR("BinaryCollision::doCollisionsWithinTile::correctEnergyMomentum", prof_correctEnergyMomentum);
                amrex::ParallelFor( std::max(np1, np2),
                    [=] AMREX_GPU_DEVICE (int ip) noexcept
                    {

                        if (ip < np1) {
                            const int i1 = ip;
                            const int i_cell = bins_1_ptr[i1];

                            // Subract the momentum from the initial values.
                            // Whatever is left over, will be distributed among the particles.
                            amrex::ParticleReal const px = w1[i1]*u1x[i1]*m1;
                            amrex::ParticleReal const py = w1[i1]*u1y[i1]*m1;
                            amrex::ParticleReal const pz = w1[i1]*u1z[i1]*m1;
                            amrex::Gpu::Atomic::AddNoRet(&px_in_each_cell[i_cell], -px);
                            amrex::Gpu::Atomic::AddNoRet(&py_in_each_cell[i_cell], -py);
                            amrex::Gpu::Atomic::AddNoRet(&pz_in_each_cell[i_cell], -pz);
                        }

                        if (ip < np2) {
                            const int i2 = ip;
                            const int i_cell = bins_2_ptr[i2];

                            // Subract the momentum from the initial values.
                            // Whatever is left over, will be distributed among the particles.
                            amrex::ParticleReal const px = w2[i2]*u2x[i2]*m2;
                            amrex::ParticleReal const py = w2[i2]*u2y[i2]*m2;
                            amrex::ParticleReal const pz = w2[i2]*u2z[i2]*m2;
                            amrex::Gpu::Atomic::AddNoRet(&px_in_each_cell[i_cell], -px);
                            amrex::Gpu::Atomic::AddNoRet(&py_in_each_cell[i_cell], -py);
                            amrex::Gpu::Atomic::AddNoRet(&pz_in_each_cell[i_cell], -pz);
                        }
                    }
                );

                amrex::ParallelFor( std::max(np1, np2),
                    [=] AMREX_GPU_DEVICE (int ip) noexcept
                    {
                        if (ip < np1) {
                            const int i1 = ip;
                            const int i_cell = bins_1_ptr[i1];

                            amrex::ParticleReal const ww_weighted_sum = ww_weighted_sum_in_each_cell[i_cell];
                            amrex::ParticleReal const w_factor = std::pow(w1[i1], beta_weight_exponent - 1._prt)/m1;
                            u1x[i1] += w_factor*px_in_each_cell[i_cell]/ww_weighted_sum;
                            u1y[i1] += w_factor*py_in_each_cell[i_cell]/ww_weighted_sum;
                            u1z[i1] += w_factor*pz_in_each_cell[i_cell]/ww_weighted_sum;
                        }

                        if (ip < np2) {
                            const int i2 = ip;
                            const int i_cell = bins_2_ptr[i2];

                            amrex::ParticleReal const ww_weighted_sum = ww_weighted_sum_in_each_cell[i_cell];
                            amrex::ParticleReal const w_factor = std::pow(w2[i2], beta_weight_exponent - 1._prt)/m2;
                            u2x[i2] += w_factor*px_in_each_cell[i_cell]/ww_weighted_sum;
                            u2y[i2] += w_factor*py_in_each_cell[i_cell]/ww_weighted_sum;
                            u2z[i2] += w_factor*pz_in_each_cell[i_cell]/ww_weighted_sum;
                        }
                    }
                );

                amrex::Gpu::Buffer<amrex::Long> failed_corrections({0});
                amrex::Gpu::Buffer<amrex::ParticleReal> remaining_energy({0.});
                amrex::Long* failed_corrections_ptr = failed_corrections.data();
                amrex::ParticleReal* remaining_energy_ptr = remaining_energy.data();

                amrex::ParallelFor( n_cells,
                    [=] AMREX_GPU_DEVICE (int i_cell) noexcept
                    {
                        // The particles from species1 that are in the cell `i_cell` are
                        // given by the `indices_1[cell_start_1:cell_stop_1]`.
                        index_type const cell_start_1 = cell_offsets_1[i_cell];
                        index_type const cell_stop_1  = cell_offsets_1[i_cell+1];
                        // Same for species 2
                        index_type const cell_start_2 = cell_offsets_2[i_cell];
                        index_type const cell_stop_2  = cell_offsets_2[i_cell+1];

                        // Do not do the rebalance if one species is missing in the cell.
                        if ( cell_stop_1 - cell_start_1 < 1 ||
                             cell_stop_2 - cell_start_2 < 1 ) { return; }

                        amrex::ParticleReal const psq = px_in_each_cell[i_cell]*px_in_each_cell[i_cell] +
                                                        py_in_each_cell[i_cell]*py_in_each_cell[i_cell] +
                                                        pz_in_each_cell[i_cell]*pz_in_each_cell[i_cell];
                        if (psq > 0.) {
                            amrex::ParticleReal w1_sum = 0.0_prt;
                            amrex::ParticleReal w2_sum = 0.0_prt;
                            amrex::ParticleReal KE1_after = 0._prt;
                            amrex::ParticleReal KE2_after = 0._prt;
                            for (index_type i1=cell_start_1; i1<cell_stop_1; ++i1) {
                                w1_sum += w1[ indices_1[i1] ];
                                KE1_after += w1[ indices_1[i1] ]*Algorithms::KineticEnergy(u1x[ indices_1[i1] ],
                                                                                           u1y[ indices_1[i1] ],
                                                                                           u1z[ indices_1[i1] ], m1);
                            }
                            for (index_type i2=cell_start_2; i2<cell_stop_2; ++i2) {
                                w2_sum += w2[ indices_2[i2] ];
                                KE2_after += w2[ indices_2[i2] ]*Algorithms::KineticEnergy(u2x[ indices_2[i2] ],
                                                                                           u2y[ indices_2[i2] ],
                                                                                           u2z[ indices_2[i2] ], m2);
                            }

                            const amrex::ParticleReal KE_after = KE1_after + KE2_after;
                            const amrex::ParticleReal deltaE = KE_after - KE_in_each_cell[i_cell];
                            amrex::ParticleReal deltaEp1, deltaEp2;
                            int const numCell1 = (cell_stop_1 - cell_start_1);
                            int const numCell2 = (cell_stop_2 - cell_start_2);
                            if (numCell1 == 1) {
                                deltaEp1 = 0.0;
                                deltaEp2 = deltaE;
                            } else if (numCell2 == 1) {
                                deltaEp1 = deltaE;
                                deltaEp2 = 0.0;
                            } else {
                                const amrex::ParticleReal Etotdenom = w1_sum*KE1_after/numCell1 + w2_sum*KE2_after/numCell2;
                                deltaEp1 = w1_sum*KE1_after/Etotdenom*deltaE/numCell1;
                                deltaEp2 = w2_sum*KE2_after/Etotdenom*deltaE/numCell2;
                            }

                            bool correction1_failed = false;
                            bool correction2_failed = false;
                            if (deltaEp1 != 0.) {
                                // Adjust species 1 particles to absorb deltaEp1.
                                correction1_failed =
                                     ParticleUtils::ModifyEnergyPairwise(u1x, u1y, u1z, w1, indices_1,
                                                                         cell_start_1, cell_stop_1, m1,
                                                                         energy_fraction, energy_fraction_max, deltaEp1);
                            }

                            if (deltaEp2 != 0.) {
                                // Adjust species 2 particles to absorb deltaEp2.
                                correction2_failed =
                                     ParticleUtils::ModifyEnergyPairwise(u2x, u2y, u2z, w2, indices_2,
                                                                         cell_start_2, cell_stop_2, m2,
                                                                         energy_fraction, energy_fraction_max, deltaEp2);
                            }

                            if (correction1_failed || correction2_failed) {
                                // If the correction failed, give up on the collision and restore the
                                // momentum to what it was beforehand.
                                for (index_type i1=cell_start_1; i1<cell_stop_1; ++i1) {
                                    u1x[ indices_1[i1] ] = u1x_before_ptr[ indices_1[i1] ];
                                    u1y[ indices_1[i1] ] = u1y_before_ptr[ indices_1[i1] ];
                                    u1z[ indices_1[i1] ] = u1z_before_ptr[ indices_1[i1] ];
                                }
                                for (index_type i2=cell_start_2; i2<cell_stop_2; ++i2) {
                                    u2x[ indices_2[i2] ] = u2x_before_ptr[ indices_2[i2] ];
                                    u2y[ indices_2[i2] ] = u2y_before_ptr[ indices_2[i2] ];
                                    u2z[ indices_2[i2] ] = u2z_before_ptr[ indices_2[i2] ];
                                }
                                amrex::Gpu::Atomic::Add(failed_corrections_ptr, amrex::Long(1));
                                amrex::Gpu::Atomic::Add(remaining_energy_ptr, deltaEp1 + deltaEp2);
                            }

                        }
                    }
                );

                amrex::Long const num_failed_corrections = *(failed_corrections.copyToHost());
                amrex::ParticleReal const total_remaining_energy = *(remaining_energy.copyToHost());
                if (num_failed_corrections > 0) {
                    ablastr::warn_manager::WMRecordWarning("BinaryCollision::doCollisionsWithinTile",
                        "The energy correction failed for " + std::to_string(num_failed_corrections) + " cells " +
                        "for Coulomb collisions between species " + m_species_names[0] + " and " + m_species_names[1] + ". " +
                        "The remaining energy is " + std::to_string(total_remaining_energy/PhysConst::q_e) + " eV. " +
                        "The collisions in those cells was cancelled.");
                }

                WARPX_PROFILE_VAR_STOP(prof_correctEnergyMomentum);
            }

        } // end if ( m_isSameSpecies)

    }

private:

    bool m_isSameSpecies;
    bool m_have_product_species;

    bool m_correct_energy_momentum = false;
    amrex::ParticleReal m_beta_weight_exponent = 1.;
    amrex::ParticleReal m_energy_fraction = 0.05;
    amrex::ParticleReal m_energy_fraction_max = 0.5;

    amrex::Vector<std::string> m_product_species;
    // functor that performs collisions within a cell
    CollisionFunctor m_binary_collision_functor;
    // functor that creates new particles and initializes their parameters
    CopyTransformFunctor m_copy_transform_functor;

};

#endif // WARPX_PARTICLES_COLLISION_BINARYCOLLISION_H_
