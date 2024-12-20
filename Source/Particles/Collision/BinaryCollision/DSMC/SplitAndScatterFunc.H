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
        const amrex::Vector<index_type>& products_np,
        const SmartCopy* AMREX_RESTRICT copy_species1,
        const SmartCopy* AMREX_RESTRICT copy_species2,
        const index_type* AMREX_RESTRICT p_pair_indices_1,
        const index_type* AMREX_RESTRICT p_pair_indices_2,
        const amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight ) const
    {
        using namespace amrex::literals;

        if (n_total_pairs == 0) { return amrex::Vector<int>(m_num_product_species, 0); }

        amrex::Gpu::DeviceVector<index_type> offsets(n_total_pairs);
        index_type* AMREX_RESTRICT offsets_data = offsets.data();
        const index_type* AMREX_RESTRICT p_offsets = offsets.dataPtr();

        // The following is used to calculate the appropriate offsets. Note that
        // a standard cummulative sum is not appropriate since the mask is also
        // used to specify the type of collision and can therefore have values >1
        auto const total = amrex::Scan::PrefixSum<index_type>(n_total_pairs,
            [=] AMREX_GPU_DEVICE (index_type i) -> index_type { return mask[i] ? 1 : 0; },
            [=] AMREX_GPU_DEVICE (index_type i, index_type s) { offsets_data[i] = s; },
            amrex::Scan::Type::exclusive, amrex::Scan::retSum
        );

        amrex::Vector<int> num_added_vec(m_num_product_species);
        for (int i = 0; i < m_num_product_species; i++)
        {
            // How many particles of product species i are potentially created.
            // ("potentially" because the actual number of particles created depends on the
            // type of reaction in the DSMC collision, which is stored in the mask. For
            // instance, an IONIZATION reaction will create 2 electrons (the scattered
            // electron, which will be split off from the incident electron, and the
            // ionized electron), while an ELASTIC reaction will only 1 electron (the
            // scattered electron). Therefore, here we allocate enough memory to store
            // all potential particles that can be created (see also the constructor of
            // SplitAndScatterFunc, where `m_num_products_host` is set), but set an
            // invalid ID for the particles that should not actually be created, so that
            // they are removed in the subsequent call to ReDistribute.
            const index_type num_added = total * m_num_products_host[i];
            num_added_vec[i] = static_cast<int>(num_added);
            tile_products[i]->resize(products_np[i] + num_added);
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

        const int* AMREX_RESTRICT p_num_products_device = m_num_products_device.data();
        const bool ionization_flag = m_ionization_flag;

        amrex::ParallelForRNG(n_total_pairs,
        [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
        {
            if (mask[i])
            {
                const auto product1_index = products_np_data[0] +
                                           (p_offsets[i]*p_num_products_device[0] + 0);
                // Make a copy of the particle from species 1
                copy_species1[0](soa_products_data[0], soa_1, static_cast<int>(p_pair_indices_1[i]),
                                static_cast<int>(product1_index), engine);
                // Set the weight of the new particles to p_pair_reaction_weight[i]
                soa_products_data[0].m_rdata[PIdx::w][product1_index] = p_pair_reaction_weight[i];

                const auto product2_index = products_np_data[1] +
                                           (p_offsets[i]*p_num_products_device[1] + 0);
                // Make a copy of the particle from species 2
                copy_species2[1](soa_products_data[1], soa_2, static_cast<int>(p_pair_indices_2[i]),
                                static_cast<int>(product2_index), engine);
                // Set the weight of the new particles to p_pair_reaction_weight[i]
                soa_products_data[1].m_rdata[PIdx::w][product2_index] = p_pair_reaction_weight[i];

                if (ionization_flag) {
                    const auto product3_index = products_np_data[2] +
                                            (p_offsets[i]*p_num_products_device[2] + 0);
                    if (mask[i] == int(ScatteringProcessType::IONIZATION)) {
                        // The electron created from ionization is stored as the second particle in product 1
                        // Copy its properties from the target atoms, i.e. species 2 (momentum, position, etc.)
                        copy_species2[0](soa_products_data[0], soa_2, static_cast<int>(p_pair_indices_2[i]),
                                        static_cast<int>(product1_index+1), engine);
                        // Set the weight of the new particles to p_pair_reaction_weight[i]
                        soa_products_data[0].m_rdata[PIdx::w][product1_index+1] = p_pair_reaction_weight[i];

                        // The ion created from ionization is stored as the first particle in product 3
                        // Copy its properties from the target atoms, i.e. species 2 (momentum, position, etc.)
                        copy_species2[2](soa_products_data[2], soa_2, static_cast<int>(p_pair_indices_2[i]),
                                        static_cast<int>(product3_index), engine);
                        // Set the weight of the new particles to p_pair_reaction_weight[i]
                        soa_products_data[2].m_rdata[PIdx::w][product3_index] = p_pair_reaction_weight[i];

                        // The ion created in product 3 replaces the scattered neutral particle in product 2
                        // (which would have been created by other, elastic scattering processes)
                        // Therefore, we set the ID of the scattered neutral particle to invalid,
                        // so that it gets removed in the next call to `ReDistribute` or `RemoveInvalidParticles`
                        soa_products_data[1].idcpu(product2_index) = amrex::ParticleIdCpus::Invalid;
                    } else {
                        // Set IDs as invalid for slots that have been allocated for potential
                        // ionization events, so that the corresponding particles get removed in the next
                        // call to `ReDistribute` or `RemoveInvalidParticles`
                        soa_products_data[0].idcpu(product1_index+1) = amrex::ParticleIdCpus::Invalid;
                        soa_products_data[2].idcpu(product3_index) = amrex::ParticleIdCpus::Invalid;
                    }
                }

                // Set the child particle properties appropriately
                auto& ux1 = soa_products_data[0].m_rdata[PIdx::ux][product1_index];
                auto& uy1 = soa_products_data[0].m_rdata[PIdx::uy][product1_index];
                auto& uz1 = soa_products_data[0].m_rdata[PIdx::uz][product1_index];
                auto& ux2 = soa_products_data[1].m_rdata[PIdx::ux][product2_index];
                auto& uy2 = soa_products_data[1].m_rdata[PIdx::uy][product2_index];
                auto& uz2 = soa_products_data[1].m_rdata[PIdx::uz][product2_index];

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
                } else if (mask[i] == int(ScatteringProcessType::CHARGE_EXCHANGE)) {
                    if (std::abs(m1 - m2) < 1e-28) {
                        auto const temp_ux = ux1;
                        auto const temp_uy = uy1;
                        auto const temp_uz = uz1;
                        ux1 = ux2;
                        uy1 = uy2;
                        uz1 = uz2;
                        ux2 = temp_ux;
                        uy2 = temp_uy;
                        uz2 = temp_uz;
                    }
                    else {
                        amrex::Abort("Uneven mass charge-exchange not implemented yet.");
                    }
                } else if (mask[i] == int(ScatteringProcessType::IONIZATION)) {
                    // calculate kinetic energy of the incident electron
                    double E_coll;
                    double const energy_cost = 0; // TODO: update this: get from scattering process
                    const amrex::ParticleReal u_coll2 = ux1*ux1 + uy1*uy1 + uz1*uz1;
                    ParticleUtils::getEnergy(u_coll2, m1, E_coll);
                    // each of the two electron (the scattered e) gets half the energy (could change this later)
                    const amrex::ParticleReal E_out = static_cast<amrex::ParticleReal>((E_coll - energy_cost) / 2.0_prt * PhysConst::q_e);
                    constexpr auto c2 = PhysConst::c * PhysConst::c;
                    const auto mc2 = m1*c2;
                    const amrex::ParticleReal up = static_cast<amrex::ParticleReal>(std::sqrt(E_out * (E_out + 2.0_prt*mc2) / c2) / m1);

                    // isotropically scatter electrons
                    // - The incident electron
                    ParticleUtils::RandomizeVelocity(ux1, uy1, uz1, up, engine);
                    // - The newly created electron
                    auto& e_ux1 = soa_products_data[0].m_rdata[PIdx::ux][product1_index+1];
                    auto& e_uy1 = soa_products_data[0].m_rdata[PIdx::uy][product1_index+1];
                    auto& e_uz1 = soa_products_data[0].m_rdata[PIdx::uz][product1_index+1];
                    ParticleUtils::RandomizeVelocity(e_ux1, e_uy1, e_uz1, up, engine);
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
    bool m_ionization_flag = false;
    // Vectors of size m_num_product_species storing how many particles of a given species are
    // produced by a collision event. These vectors are duplicated (one version for host and one
    // for device) which is necessary with GPUs but redundant on CPU.
    amrex::Gpu::DeviceVector<int> m_num_products_device;
    amrex::Gpu::HostVector<int> m_num_products_host;
    CollisionType m_collision_type;
};
#endif // WARPX_SPLIT_AND_SCATTER_FUNC_H_
