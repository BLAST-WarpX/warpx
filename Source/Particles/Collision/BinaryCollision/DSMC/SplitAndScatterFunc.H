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
        const amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight,
        const amrex::ParticleReal* /*p_product_data*/ ) const
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

                // calculate kinetic energy of the collision (in eV)
                const amrex::ParticleReal E1 = (
                    0.5_prt * m1 * (ux1*ux1 + uy1*uy1 + uz1*uz1) / PhysConst::q_e
                );
                const amrex::ParticleReal E2 = (
                    0.5_prt * m2 * (ux_p2*ux_p2 + uy_p2*uy_p2 + uz_p2*uz_p2) / PhysConst::q_e
                );
                const amrex::ParticleReal E_coll = E1 + E2;

                // subtract the energy cost for ionization
                const amrex::ParticleReal E_out = (E_coll - ionization_energy) * PhysConst::q_e;

                // Momentum and energy division after the ionization event
                // is done as follows:
                // Three numbers are generated that satisfy the triangle
                // inequality. These numbers will be scaled to give the
                // momentum for each particle (satisfying the triangle
                // inequality ensures that the momentum vectors can be
                // arranged such that the net linear momentum is zero).
                // Product 2 is definitely an ion, so we first choose its
                // proportion to be n_2 = min(E2 / E_coll, 0.5 * R), where
                // R is a random number between 0 and 1.
                // The other two numbers (n_0 & n_1) are then found
                // by choosing a random point on the ellipse characterized
                // by the semi-major and semi-minor axes
                //    a = (1 - n_0) / 2.0
                //    b = 0.5 * sqrt(1 - 2 * n_0)
                // The numbers are found by randomly sampling an x value
                // between -a and a, and finding the corresponding y value
                // that falls on the ellipse: y^2 = b^2 - b^2/a^2 * x^2.
                // Then n_0 = sqrt(y^2 + (x - n_2/2)^2) and
                // n_1 = 1 - n_2 - n_0.
                // Next, we need to find the number C, such that if
                // p_i = C * n_i, we would have E_0 + E_1 + E_2 = E_out
                // where E_i = p_i^2 / (2 M_i), i.e.,
                // C^2 * \sum_i [n_i^2 / (2 M_i)] = E_out
                // After C is determined the momentum vectors are arranged
                // such that the net linear momentum is zero.

                const amrex::ParticleReal n_2 = std::min(E2 / E_coll, 0.5_prt * static_cast<amrex::ParticleReal>(amrex::Random(engine)));

                // find ellipse semi-major and minor axis
                const amrex::ParticleReal a = 0.5_prt * (1.0_prt - n_2);
                const amrex::ParticleReal b = 0.5_prt * std::sqrt(1.0_prt - 2.0_prt * n_2);

                // sample random x value and calculate y
                const amrex::ParticleReal x = (2._prt * amrex::Random(engine) - 1.0_prt) * a;
                const amrex::ParticleReal y2 = b*b - b*b/(a*a) * x*x;
                const amrex::ParticleReal n_0 = std::sqrt(y2 + x*x - x*n_2 + 0.25_prt*n_2*n_2);
                const amrex::ParticleReal n_1 = 1.0_prt - n_0 - n_2;

                // calculate the value of C
                const amrex::ParticleReal C = std::sqrt(E_out / (
                    n_0*n_0 / (2.0_prt * m1) + n_1*n_1 / (2.0_prt * m_ioniz_product1) + n_2*n_2 / (2.0_prt * m_ioniz_product2)
                ));

                // Now that appropriate momenta are set for each outgoing species
                // the directions for the velocity vectors must be chosen such
                // that the net linear momentum in the current frame is 0.
                // This is achieved by arranging the momentum vectors in
                // a triangle and finding the required angles between the vectors.
                const amrex::ParticleReal cos_alpha = (n_0*n_0 + n_1*n_1 - n_2*n_2) / (2.0_prt * n_0 * n_1);
                const amrex::ParticleReal sin_alpha = std::sqrt(1.0_prt - cos_alpha*cos_alpha);
                const amrex::ParticleReal cos_gamma = (n_0*n_0 + n_2*n_2 - n_1*n_1) / (2.0_prt * n_0 * n_2);
                const amrex::ParticleReal sin_gamma = std::sqrt(1.0_prt - cos_gamma*cos_gamma);

                // choose random theta and phi values (orientation of the triangle)
                const amrex::ParticleReal Theta = amrex::Random(engine) * 2.0_prt * MathConst::pi;
                const amrex::ParticleReal phi = amrex::Random(engine) * MathConst::pi;

                const amrex::ParticleReal cos_theta = std::cos(Theta);
                const amrex::ParticleReal sin_theta = std::sin(Theta);
                const amrex::ParticleReal cos_phi = std::cos(phi);
                const amrex::ParticleReal sin_phi = std::sin(phi);

                // calculate the velocity components for each particle
                ux1 = C * n_0 / m1 * cos_theta * cos_phi;
                uy1 = C * n_0 / m1 * cos_theta * sin_phi;
                uz1 = -C * n_0 / m1 * sin_theta;

                ux_p1 = C * n_1 / m_ioniz_product1 * (-cos_alpha * cos_theta * cos_phi - sin_alpha * sin_phi);
                uy_p1 = C * n_1 / m_ioniz_product1 * (-cos_alpha * cos_theta * sin_phi + sin_alpha * cos_phi);
                uz_p1 = C * n_1 / m_ioniz_product1 * (cos_alpha * sin_theta);

                ux_p2 = C * n_2 / m_ioniz_product2 * (-cos_gamma * cos_theta * cos_phi + sin_gamma * sin_phi);
                uy_p2 = C * n_2 / m_ioniz_product2 * (-cos_gamma * cos_theta * sin_phi - sin_gamma * cos_phi);
                uz_p2 = C * n_2 / m_ioniz_product2 * (cos_gamma * sin_theta);

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
