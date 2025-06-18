/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SPLIT_AND_SCATTER_FUNC_H_
#define WARPX_SPLIT_AND_SCATTER_FUNC_H_

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/Collision/ScatteringProcess.H"
#include "Particles/ParticleCreation/SmartCopy.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/ParticleUtils.H"

/**
 * \brief This class defines an operator to create product particles from DSMC
 * collisions and sets the particle properties (position, momentum, weight).
 */
class SplitAndScatterFunc
{
    // Define shortcuts for frequently-used type names
    using ParticleType = typename WarpXParticleContainer::ParticleType;
    using ParticleTileType = typename WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = typename ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = typename ParticleBins::index_type;
    using SoaData_type = WarpXParticleContainer::ParticleTileType::ParticleTileDataType;

public:
    /**
     * \brief Default constructor of the SplitAndScatterFunc class.
     */
    SplitAndScatterFunc () = default;

    /**
     * \brief Constructor of the SplitAndScatterFunc class
     *
     * @param[in] collision_name the name of the collision
     * @param[in] mypc pointer to the MultiParticleContainer
     */
    SplitAndScatterFunc (const std::string& collision_name, MultiParticleContainer const * mypc);

    /**
     * \brief Function that performs the particle scattering and injection due
     * to binary collisions.
     *
     * \return num_added the number of particles added to each species.
     */
    AMREX_INLINE
    amrex::Vector<int> operator() (
        const index_type& n_total_pairs,
        ParticleTileType& ptile1, ParticleTileType& ptile2,
        const amrex::Vector<WarpXParticleContainer*>& pc_products,
        ParticleTileType** AMREX_RESTRICT tile_products,
        const amrex::ParticleReal m1, const amrex::ParticleReal m2,
        const amrex::Vector<amrex::ParticleReal>& /*products_mass*/,
        const index_type* AMREX_RESTRICT mask,
        amrex::Vector<index_type>& products_np,
        const SmartCopy* AMREX_RESTRICT copy_species1,
        const SmartCopy* AMREX_RESTRICT copy_species2,
        const index_type* AMREX_RESTRICT p_pair_indices_1,
        const index_type* AMREX_RESTRICT p_pair_indices_2,
        const amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight ) const
    {
        using namespace amrex::literals;

        // Return a vector of zeros, indicating that for all the "product" species
        // there were no new particles added.
        if (n_total_pairs == 0) { return amrex::Vector<int>(m_num_product_species, 0); }

        // The following is used to calculate the appropriate offsets for processes
        // that do not produce macroparticles in new species (i.e., not ionization, not charge exchange).
        // Note that a standard cummulative sum is not appropriate since the
        // mask is also used to specify the type of collision and can therefore
        // have values >1
        amrex::Gpu::DeviceVector<index_type> no_product_offsets(n_total_pairs);
        index_type* AMREX_RESTRICT no_product_offsets_data = no_product_offsets.data();
        const index_type* AMREX_RESTRICT no_product_p_offsets = no_product_offsets.dataPtr();
        auto const no_product_total = amrex::Scan::PrefixSum<index_type>(n_total_pairs,
            [=] AMREX_GPU_DEVICE (index_type i) -> index_type {
                return ((mask[i] > 0) &
                        (mask[i] != int(ScatteringProcessType::IONIZATION)) &
                        (mask[i] != int(ScatteringProcessType::CHARGE_EXCHANGE))) ? 1 : 0;
            },
            [=] AMREX_GPU_DEVICE (index_type i, index_type s) { no_product_offsets_data[i] = s; },
            amrex::Scan::Type::exclusive, amrex::Scan::retSum
        );

        amrex::Vector<int> num_added_vec(m_num_product_species, 0);
        for (int i = 0; i < 2; i++)
        {
            // Record the number of non product producing events lead to new
            // particles for species1 and 2. Only 1 particle is created for
            // each species (the piece that breaks off to have equal weight)
            // particles.
            num_added_vec[i] = static_cast<int>(no_product_total);
        }

        // The following is used to calculate the appropriate offsets for
        // product producing processes (i.e., ionization and charge exchange).
        // Note that a standard cummulative sum is not appropriate since the
        // mask is also used to specify the type of collision and can therefore
        // have values >1
        amrex::Gpu::DeviceVector<index_type> with_product_offsets(n_total_pairs);
        index_type* AMREX_RESTRICT with_product_offsets_data = with_product_offsets.data();
        const index_type* AMREX_RESTRICT with_product_p_offsets = with_product_offsets.dataPtr();
        auto const with_product_total = amrex::Scan::PrefixSum<index_type>(n_total_pairs,
            [=] AMREX_GPU_DEVICE (index_type i) -> index_type {
                return ((mask[i] == int(ScatteringProcessType::IONIZATION)) |
                        (mask[i] == int(ScatteringProcessType::CHARGE_EXCHANGE))) ? 1 : 0;
            },
            [=] AMREX_GPU_DEVICE (index_type i, index_type s) { with_product_offsets_data[i] = s; },
            amrex::Scan::Type::exclusive, amrex::Scan::retSum
        );

        for (int i = 0; i < m_num_product_species; i++)
        {
            // Add the number of product producing events to the species involved in those processes.
            int num_products = m_num_products_host[i];
            const index_type num_added = with_product_total * num_products;
            num_added_vec[i] += static_cast<int>(num_added);
        }

        // resize the particle tiles to accomodate the new particles
        // The code works correctly, even if the same species appears
        // several times in the product species list.
        // (e.g., for e- + H -> 2 e- + H+, the product species list is [e-, H, e-, H+])
        // In that case, the species that appears multiple times is resized several times ;
        // `products_np` keeps track of array positions at which it has been resized.
        for (int i = 0; i < m_num_product_species; i++)
        {
            products_np[i] = tile_products[i]->numParticles();
            tile_products[i]->resize(products_np[i] + num_added_vec[i]);
        }

        const auto soa_1 = ptile1.getParticleTileData();
        const auto soa_2 = ptile2.getParticleTileData();

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

        const int num_product_species = m_num_product_species;
        const auto ionization_energy = m_ionization_energy;

        // Grab the masses of ionization products
        amrex::ParticleReal m_ioniz_product1 = 0;
        amrex::ParticleReal m_ioniz_product2 = 0;
        if (num_product_species > 2) {
            m_ioniz_product1 = pc_products[2]->getMass();
            m_ioniz_product2 = pc_products[3]->getMass();
        }

        // First perform all non-product producing collisions
        amrex::ParallelForRNG(n_total_pairs,
        [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
        {
            if ((mask[i] > 0) & (mask[i] != int(ScatteringProcessType::IONIZATION)) & (mask[i] != int(ScatteringProcessType::CHARGE_EXCHANGE)))
            {
                const auto product1_index = products_np_data[0] + no_product_p_offsets[i];
                // Make a copy of the particle from species 1
                copy_species1[0](soa_products_data[0], soa_1, static_cast<int>(p_pair_indices_1[i]),
                                static_cast<int>(product1_index), engine);
                // Set the weight of the new particles to p_pair_reaction_weight[i]
                soa_products_data[0].m_rdata[PIdx::w][product1_index] = p_pair_reaction_weight[i];

                const auto product2_index = products_np_data[1] + no_product_p_offsets[i];
                // Make a copy of the particle from species 2
                copy_species2[1](soa_products_data[1], soa_2, static_cast<int>(p_pair_indices_2[i]),
                                static_cast<int>(product2_index), engine);
                // Set the weight of the new particles to p_pair_reaction_weight[i]
                soa_products_data[1].m_rdata[PIdx::w][product2_index] = p_pair_reaction_weight[i];

                // Set the child particle properties appropriately
                auto& ux1 = soa_products_data[0].m_rdata[PIdx::ux][product1_index];
                auto& uy1 = soa_products_data[0].m_rdata[PIdx::uy][product1_index];
                auto& uz1 = soa_products_data[0].m_rdata[PIdx::uz][product1_index];
                auto& ux2 = soa_products_data[1].m_rdata[PIdx::ux][product2_index];
                auto& uy2 = soa_products_data[1].m_rdata[PIdx::uy][product2_index];
                auto& uz2 = soa_products_data[1].m_rdata[PIdx::uz][product2_index];

#if (defined WARPX_DIM_RZ)
                /* In RZ geometry, macroparticles can collide with other macroparticles
                * in the same *cylindrical* cell. For this reason, collisions between macroparticles
                * are actually not local in space. In this case, the underlying assumption is that
                * particles within the same cylindrical cell represent a cylindrically-symmetry
                * momentum distribution function. Therefore, here, we temporarily rotate the
                * momentum of one of the macroparticles in agreement with this cylindrical symmetry.
                * (This is technically only valid if we use only the m=0 azimuthal mode in the simulation;
                * there is a corresponding assert statement at initialization.)
                */
                amrex::ParticleReal const theta = (
                    soa_products_data[1].m_rdata[PIdx::theta][product2_index]
                    - soa_products_data[0].m_rdata[PIdx::theta][product1_index]
                );
                amrex::ParticleReal const ux1buf = ux1;
                ux1 = ux1buf*std::cos(theta) - uy1*std::sin(theta);
                uy1 = ux1buf*std::sin(theta) + uy1*std::cos(theta);
#endif

                // for simplicity (for now) we assume non-relativistic particles
                // and simply calculate the center-of-momentum velocity from the
                // rest masses
                auto const uCOM_x = (m1 * ux1 + m2 * ux2) / (m1 + m2);
                auto const uCOM_y = (m1 * uy1 + m2 * uy2) / (m1 + m2);
                auto const uCOM_z = (m1 * uz1 + m2 * uz2) / (m1 + m2);

                // transform to COM frame
                ux1 -= uCOM_x;
                uy1 -= uCOM_y;
                uz1 -= uCOM_z;
                ux2 -= uCOM_x;
                uy2 -= uCOM_y;
                uz2 -= uCOM_z;

                if (mask[i] == int(ScatteringProcessType::ELASTIC)) {
                    // randomly rotate the velocity vector for the first particle
                    ParticleUtils::RandomizeVelocity(
                        ux1, uy1, uz1, std::sqrt(ux1*ux1 + uy1*uy1 + uz1*uz1), engine
                    );
                    // set the second particles velocity so that the total momentum
                    // is zero
                    ux2 = -ux1 * m1 / m2;
                    uy2 = -uy1 * m1 / m2;
                    uz2 = -uz1 * m1 / m2;
                } else if (mask[i] == int(ScatteringProcessType::BACK)) {
                    // reverse the velocity vectors of both particles
                    ux1 *= -1.0_prt;
                    uy1 *= -1.0_prt;
                    uz1 *= -1.0_prt;
                    ux2 *= -1.0_prt;
                    uy2 *= -1.0_prt;
                    uz2 *= -1.0_prt;
                } else if (mask[i] == int(ScatteringProcessType::FORWARD)) {
                    amrex::Abort("Forward scattering with DSMC not implemented yet.");
                }
                else {
                    amrex::Abort("Unknown scattering process.");
                }
                // transform back to labframe
                ux1 += uCOM_x;
                uy1 += uCOM_y;
                uz1 += uCOM_z;
                ux2 += uCOM_x;
                uy2 += uCOM_y;
                uz2 += uCOM_z;

#if (defined WARPX_DIM_RZ)
                /* Undo the earlier velocity rotation. */
                amrex::ParticleReal const ux1buf_new = ux1;
                ux1 = ux1buf_new*std::cos(-theta) - uy1*std::sin(-theta);
                uy1 = ux1buf_new*std::sin(-theta) + uy1*std::cos(-theta);
#endif
            }

            // Next perform all product-producing collisions
            else if (mask[i] == int(ScatteringProcessType::CHARGE_EXCHANGE))
            {
                // create a copy of the first product species at the location of species 1
                const auto product1_index = products_np_data[2] + with_product_p_offsets[i];
                copy_species1[2](soa_products_data[2], soa_1, static_cast<int>(p_pair_indices_1[i]),
                                static_cast<int>(product1_index), engine);
                // Set the weight of the new particle to p_pair_reaction_weight[i]
                soa_products_data[2].m_rdata[PIdx::w][product1_index] = p_pair_reaction_weight[i];

                // create a copy of the other product species at the location of species 2
                const auto product2_index = products_np_data[3] + with_product_p_offsets[i];
                copy_species2[3](soa_products_data[3], soa_2, static_cast<int>(p_pair_indices_2[i]),
                                static_cast<int>(product2_index), engine);
                // Set the weight of the new particle to p_pair_reaction_weight[i]
                soa_products_data[3].m_rdata[PIdx::w][product2_index] = p_pair_reaction_weight[i];
            }
            else if (mask[i] == int(ScatteringProcessType::IONIZATION))
            {
                const auto species1_index = products_np_data[0] + no_product_total + with_product_p_offsets[i];
                // Make a copy of the particle from species 1
                copy_species1[0](soa_products_data[0], soa_1, static_cast<int>(p_pair_indices_1[i]),
                               static_cast<int>(species1_index), engine);
                // Set the weight of the new particles to p_pair_reaction_weight[i]
                soa_products_data[0].m_rdata[PIdx::w][species1_index] = p_pair_reaction_weight[i];

                // create a copy of the first product species at the location of species 2
                // (species2 is the "target species", i.e. the species that is ionized)
                const auto product1_index = products_np_data[2] + with_product_p_offsets[i];
                copy_species2[2](soa_products_data[2], soa_2, static_cast<int>(p_pair_indices_2[i]),
                                static_cast<int>(product1_index), engine);
                // Set the weight of the new particle to p_pair_reaction_weight[i]
                soa_products_data[2].m_rdata[PIdx::w][product1_index] = p_pair_reaction_weight[i];

                // create a copy of the other product species at the location of species 2
                // (species2 is the "target species", i.e. the species that is ionized)
                const auto product2_index = products_np_data[3] + with_product_p_offsets[i];
                copy_species2[3](soa_products_data[3], soa_2, static_cast<int>(p_pair_indices_2[i]),
                                static_cast<int>(product2_index), engine);
                // Set the weight of the new particle to p_pair_reaction_weight[i]
                soa_products_data[3].m_rdata[PIdx::w][product2_index] = p_pair_reaction_weight[i];

                // Grab the colliding particle velocities to calculate the COM
                // Note that the two product particles currently have the same
                // velocity as the "target" particle
                auto& ux1 = soa_products_data[0].m_rdata[PIdx::ux][species1_index];
                auto& uy1 = soa_products_data[0].m_rdata[PIdx::uy][species1_index];
                auto& uz1 = soa_products_data[0].m_rdata[PIdx::uz][species1_index];
                auto& ux_p1 = soa_products_data[2].m_rdata[PIdx::ux][product1_index];
                auto& uy_p1 = soa_products_data[2].m_rdata[PIdx::uy][product1_index];
                auto& uz_p1 = soa_products_data[2].m_rdata[PIdx::uz][product1_index];
                auto& ux_p2 = soa_products_data[3].m_rdata[PIdx::ux][product2_index];
                auto& uy_p2 = soa_products_data[3].m_rdata[PIdx::uy][product2_index];
                auto& uz_p2 = soa_products_data[3].m_rdata[PIdx::uz][product2_index];

#if (defined WARPX_DIM_RZ)
                /* In RZ geometry, macroparticles can collide with other macroparticles
                * in the same *cylindrical* cell. For this reason, collisions between macroparticles
                * are actually not local in space. In this case, the underlying assumption is that
                * particles within the same cylindrical cell represent a cylindrically-symmetry
                * momentum distribution function. Therefore, here, we temporarily rotate the
                * momentum of one of the macroparticles in agreement with this cylindrical symmetry.
                * (This is technically only valid if we use only the m=0 azimuthal mode in the simulation;
                * there is a corresponding assert statement at initialization.)
                */
                amrex::ParticleReal const theta = (
                    soa_products_data[2].m_rdata[PIdx::theta][product1_index]
                    - soa_products_data[0].m_rdata[PIdx::theta][species1_index]
                );
                amrex::ParticleReal const ux1buf = ux1;
                ux1 = ux1buf*std::cos(theta) - uy1*std::sin(theta);
                uy1 = ux1buf*std::sin(theta) + uy1*std::cos(theta);
#endif

                // for simplicity (for now) we assume non-relativistic particles
                // and simply calculate the center-of-momentum velocity from the
                // rest masses
                auto const uCOM_x = (m1 * ux1 + m2 * ux_p2) / (m1 + m2);
                auto const uCOM_y = (m1 * uy1 + m2 * uy_p2) / (m1 + m2);
                auto const uCOM_z = (m1 * uz1 + m2 * uz_p2) / (m1 + m2);

                // transform to COM frame
                ux1 -= uCOM_x;
                uy1 -= uCOM_y;
                uz1 -= uCOM_z;
                ux_p1 -= uCOM_x;
                uy_p1 -= uCOM_y;
                uz_p1 -= uCOM_z;
                ux_p2 -= uCOM_x;
                uy_p2 -= uCOM_y;
                uz_p2 -= uCOM_z;

                // calculate kinetic energy of the collision
                const amrex::ParticleReal p_non_target_in = m1 * std::sqrt(ux1*ux1 + uy1*uy1 + uz1*uz1);
                const amrex::ParticleReal E_coll = 0.5_prt * p_non_target_in * p_non_target_in * (1.0_prt/m1 + 1.0_prt/m2);
                // subtract the energy cost for ionization (converted from eV to J)
                const amrex::ParticleReal E_out = E_coll - ionization_energy * PhysConst::q_e;

                // Momentum and energy division after the ionization event is done as follows:
                // we assume that, in the center of mass frame, the two products (the ion and the electron)
                // have equal velocity and that their velocity vector is aligned with that of the target particle

                // With these assumptions, conservation of momentum in the center of mass frame allows to express the momenta
                // of the products, as a function of the outcoming momentum of the non-target particle:
                // p_product1 = - p_non_target_out * m_ioniz_product1 / m2
                // p_product2 = - p_non_target_out * m_ioniz_product2 / m2
                // (with m2 = m_ioniz_product1 + m_ioniz_product2, by conservation of mass)

                // Amplitude of the momentum of the non-target particle, obtained from conservation of energy
                const amrex::ParticleReal p_non_target_out = std::sqrt( 2.0_prt * E_out / (1.0_prt / m1 + 1.0_prt / m2 ) );

                // Reduce the velocity of the non-target particle, to reflect the loss of energy due to ionization
                const amrex::ParticleReal reduce_factor = p_non_target_out / p_non_target_in;
                ux1 *= reduce_factor;
                uy1 *= reduce_factor;
                uz1 *= reduce_factor;
                // Obtain the other velocity vectors by conservation of momentum
                const amrex::ParticleReal mass_ratio = m1 / m2;
                ux_p1 = -mass_ratio * ux1;
                uy_p1 = -mass_ratio * uy1;
                uz_p1 = -mass_ratio * uz1;
                ux_p2 = -mass_ratio * ux1;
                uy_p2 = -mass_ratio * uy1;
                uz_p2 = -mass_ratio * uz1;

                // transform back to labframe
                ux1 += uCOM_x;
                uy1 += uCOM_y;
                uz1 += uCOM_z;
                ux_p1 += uCOM_x;
                uy_p1 += uCOM_y;
                uz_p1 += uCOM_z;
                ux_p2 += uCOM_x;
                uy_p2 += uCOM_y;
                uz_p2 += uCOM_z;

#if (defined WARPX_DIM_RZ)
                /* Undo the earlier velocity rotation. */
                amrex::ParticleReal const ux1buf_new = ux1;
                ux1 = ux1buf_new*std::cos(-theta) - uy1*std::sin(-theta);
                uy1 = ux1buf_new*std::sin(-theta) + uy1*std::cos(-theta);
#endif
            }
        });

        // Initialize the user runtime components
        for (int i = 0; i < m_num_product_species; i++)
        {
            const int start_index = int(products_np[i]);
            const int stop_index  = int(products_np[i] + num_added_vec[i]);
            ParticleCreation::DefaultInitializeRuntimeAttributes(*tile_products[i],
                                       0, 0,
                                       pc_products[i]->getUserRealAttribs(), pc_products[i]->getUserIntAttribs(),
                                       pc_products[i]->GetRealSoANames(), pc_products[i]->GetIntSoANames(),
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
    // If ionization collisions are included, what is the energy cost
    amrex::ParticleReal m_ionization_energy = 0.0;
    // Vectors of size m_num_product_species storing how many particles of a given species are
    // produced by a collision event.
    amrex::Gpu::HostVector<int> m_num_products_host;
    CollisionType m_collision_type;
};
#endif // WARPX_SPLIT_AND_SCATTER_FUNC_H_
