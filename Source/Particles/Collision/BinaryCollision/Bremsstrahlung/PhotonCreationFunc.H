/* Copyright 2024 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_PHOTON_CREATION_FUNC_H_
#define WARPX_PHOTON_CREATION_FUNC_H_

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"

#include "Particles/ParticleCreation/SmartCopy.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"

#include <AMReX_DenseBins.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_INT.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

/**
 * \brief This functor creates photons produced from a binary collision and sets their initial
 * properties (position, momentum, weight).
 */
class PhotonCreationFunc
{
    // Define shortcuts for frequently-used type names
    using ParticleType = typename WarpXParticleContainer::ParticleType;
    using ParticleTileType = typename WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = typename ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = typename ParticleBins::index_type;
    using SoaData_type = typename WarpXParticleContainer::ParticleTileType::ParticleTileDataType;

public:
    /**
     * \brief Default constructor of the PhotonCreationFunc class.
     */
    PhotonCreationFunc () = default;

    /**
     * \brief Constructor of the PhotonCreationFunc class
     *
     * @param[in] collision_name the name of the collision
     * @param[in] mypc pointer to the MultiParticleContainer
     */
    PhotonCreationFunc (const std::string& collision_name, MultiParticleContainer const * mypc);

    /**
     * \brief operator() of the PhotonCreationFunc class. It creates new photon particles from binary
     * collisions.
     * One product particle is created at the position of the first parent particle that collided.
     * For example, with the Bremsstrahlung collision between an electron and an ion,
     * the new photon is created at the position of the electron.
     * This function also sets the initial weight of the new photon to be the same as the weight
     * of the first parent.
     * This function sets the initial momentum of the product particles, by calling a
     * function specific to the considered binary collision.
     * Finally, this function subtracts the energy of the new photon from the first parent, i.e. the electron.
     *
     * @param[in] n_total_pairs how many binary collisions have been performed in this tile
     * @param[in, out] ptile1,ptile2 the particle tiles of the two colliding species
     * @param[out] tile_products array containing tile data of the product particles.
     * @param[in] m1 mass of the first colliding particle species
     * @param[in] m2 mass of the second colliding particle species
     * @param[in] products_mass array storing the mass of product particles
     * @param[in] p_mask a mask that is 1 if binary collision has resulted in particle creation
     *            event, 0 otherwise.
     * @param[in] products_np array storing the number of existing product particles in that tile
     * @param[in] copy_species1 array of functors used to copy data from the first colliding
     * particle species to the product particles and to default initialize product particle
     * quantities.
     * @param[in] copy_species2 array of functors used to copy data from the second colliding
     * particle species to the product particles and to default initialize product particle
     * quantities.
     * @param[in] p_pair_indices_1 array that stores the indices of the particles of the first
     * colliding species that were used in the binary collisions (i.e. particle with index
     * p_pair_indices_1[i] took part in collision i)
     * @param[in] p_pair_indices_2 array that stores the indices of the particles of the second
     * colliding species that were used in the binary collisions (i.e. particle with index
     * p_pair_indices_2[i] took part in collision i)
     * @param[in] p_pair_reaction_weight array that stores the photon weight
     * @param[in] p_product_data array that stores the photon energy
     */
    AMREX_INLINE
    amrex::Vector<int> operator() (
                    const index_type& n_total_pairs,
                    ParticleTileType& ptile1, ParticleTileType& /*ptile2*/,
                    const amrex::Vector<WarpXParticleContainer*>& pc_products,
                    ParticleTileType** AMREX_RESTRICT tile_products,
                    const amrex::ParticleReal& /*m1*/, const amrex::ParticleReal& /*m2*/,
                    const amrex::Vector<amrex::ParticleReal>& /*products_mass*/,
                    const index_type* AMREX_RESTRICT p_mask,
                    const amrex::Vector<index_type>& products_np,
                    const SmartCopy* AMREX_RESTRICT copy_species1,
                    const SmartCopy* AMREX_RESTRICT /*copy_species2*/,
                    const index_type* AMREX_RESTRICT p_pair_indices_1,
                    const index_type* AMREX_RESTRICT /*p_pair_indices_2*/,
                    const amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight,
                    const amrex::ParticleReal* AMREX_RESTRICT p_product_data
                    ) const
    {
        using namespace amrex::literals;

        if (n_total_pairs == 0) { return amrex::Vector<int>(m_num_product_species, 0); }

        // Compute offset array and allocate memory for the produced species
        amrex::Gpu::DeviceVector<index_type> offsets(n_total_pairs);
        const auto total = amrex::Scan::ExclusiveSum(n_total_pairs, p_mask, offsets.data());
        const index_type* AMREX_RESTRICT p_offsets = offsets.dataPtr();

        amrex::Vector<int> num_added_vec(m_num_product_species);
        for (int i = 0; i < m_num_product_species; i++)
        {
            const index_type num_added = total;
            num_added_vec[i] = static_cast<int>(num_added);
            tile_products[i]->resize(products_np[i] + num_added);
        }

        const auto soa_1 = ptile1.getParticleTileData();

        // Create necessary GPU vectors, that will be used in the kernel below
        amrex::Vector<SoaData_type> soa_products;
        for (int i = 0; i < m_num_product_species; i++)
        {
            soa_products.push_back(tile_products[i]->getParticleTileData());
        }
#ifdef AMREX_USE_GPU
        amrex::Gpu::DeviceVector<SoaData_type> device_soa_products(m_num_product_species);
        amrex::Gpu::DeviceVector<index_type> device_products_np(m_num_product_species);

        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, soa_products.begin(),
                              soa_products.end(),
                              device_soa_products.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, products_np.begin(),
                              products_np.end(),
                              device_products_np.begin());

        amrex::Gpu::streamSynchronize();
        SoaData_type* AMREX_RESTRICT soa_products_data = device_soa_products.data();
        const index_type* AMREX_RESTRICT products_np_data = device_products_np.data();
#else
        SoaData_type* AMREX_RESTRICT soa_products_data = soa_products.data();
        const index_type* AMREX_RESTRICT products_np_data = products_np.data();
#endif

        amrex::ParallelForRNG(n_total_pairs,
        [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
        {
            if (p_mask[i])
            {
                const auto product_index = products_np_data[0] + p_offsets[i];

                // Create product particle at position of particle 1
                copy_species1[0](soa_products_data[0], soa_1, static_cast<int>(p_pair_indices_1[i]),
                                 static_cast<int>(product_index), engine);

                // Set the weight of the new particle to p_pair_reaction_weight[i]
                soa_products_data[0].m_rdata[PIdx::w][product_index] = p_pair_reaction_weight[i];

                amrex::ParticleReal & ux1 = soa_products_data[0].m_rdata[PIdx::ux][product_index];
                amrex::ParticleReal & uy1 = soa_products_data[0].m_rdata[PIdx::uy][product_index];
                amrex::ParticleReal & uz1 = soa_products_data[0].m_rdata[PIdx::uz][product_index];

                // Normalize out the electron velocity and multiply by the photon momentum, E/c
                // Also, the photon momentum is normalized by m_e
                amrex::ParticleReal const u1 = std::sqrt(ux1*ux1 + uy1*uy1 + uz1*uz1);
                ux1 *= (p_product_data[i]/PhysConst::c)/u1/PhysConst::m_e;
                uy1 *= (p_product_data[i]/PhysConst::c)/u1/PhysConst::m_e;
                uz1 *= (p_product_data[i]/PhysConst::c)/u1/PhysConst::m_e;
            }
        });

        // Initialize the user runtime components
        for (int i = 0; i < m_num_product_species; i++)
        {
            const auto start_index = int(products_np[i]);
            const auto stop_index  = int(products_np[i] + num_added_vec[i]);
            ParticleCreation::DefaultInitializeRuntimeAttributes(*tile_products[i],
                                       0, 0,
                                       pc_products[i]->getUserRealAttribs(), pc_products[i]->getUserIntAttribs(),
                                       pc_products[i]->getParticleComps(), pc_products[i]->getParticleiComps(),
                                       pc_products[i]->getUserRealAttribParser(),
                                       pc_products[i]->getUserIntAttribParser(),
#ifdef WARPX_QED
                                       false, // do not initialize QED quantities, since they were initialized
                                              // when calling the SmartCopy functors
                                       pc_products[i]->get_breit_wheeler_engine_ptr(),
                                       pc_products[i]->get_quantum_sync_engine_ptr(),
#endif
                                       pc_products[i]->getIonizationInitialLevel(),
                                       start_index, stop_index);
        }

        amrex::Gpu::synchronize();

        return num_added_vec;
    }

private:
    // How many different type of species the collision produces
    int m_num_product_species;

    CollisionType m_collision_type;

};

#endif // WARPX_PHOTON_CREATION_FUNC_H_
