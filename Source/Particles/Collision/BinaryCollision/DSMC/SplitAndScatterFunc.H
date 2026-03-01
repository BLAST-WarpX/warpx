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

#include "Particles/Collision/BinaryCollision/TwoProductUtil.H"
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
     * Product species are arranged in slots 0 through m_num_product_species-1.
     * Slots 0 and 1 always correspond to the first and second reactant species, respectively. For reaction
     * events that produce new species (ionization, charge exchange, two-product
     * reaction), slots 2 and 3 hold the true output species.
     *
     * @param[in]     n_total_pairs         total number of particle pairs across all cells.
     * @param[in,out] ptile0                particle tile for the first reactant species (slot 0);
     *                                      weights may be reduced by the collision.
     * @param[in,out] ptile1                particle tile for the other reactant species (slot 1);
     *                                      weights may be reduced by the collision.
     * @param[in]     pc_products           pointers to all product species containers
     *                                      (size m_num_product_species); used for mass queries
     *                                      and runtime attribute initialization.
     * @param[in,out] tile_products         array of pointers to product particle tiles
     *                                      (size m_num_product_species); new particles are
     *                                      appended to each tile.
     * @param[in]     m0                    rest mass of the first reactant species (slot 0).
     * @param[in]     m1                    rest mass of the other reactant species (slot 1).
     * @param[in]     mask                  per-pair collision type: 0 = no collision,
     *                                      otherwise cast of ScatteringProcessType to int.
     * @param[in,out] products_np           on input: current particle count of each product
     *                                      tile; updated to the pre-addition count so that
     *                                      newly added particles can be correctly addressed
     *                                      even when the same container appears in multiple
     *                                      slots (e.g. e- + H -> 2 e- + H+).
     * @param[in]     copy_species0         array of SmartCopy functors (size m_num_product_species);
     *                                      copy_species0[k] copies a particle FROM the first
     *                                      reactant species (slot 0) INTO product slot k.
     * @param[in]     copy_species1         same as copy_species0 but copies FROM the other
     *                                      reactant species (slot 1).
     * @param[in]     p_pair_indices_0      per-pair index of the first reactant species (slot 0) particle.
     * @param[in]     p_pair_indices_1      per-pair index of the other reactant species (slot 1) particle.
     * @param[in]     p_pair_reaction_weight per-pair weight assigned to newly created product
     *                                      particles; also the amount subtracted from the
     *                                      colliding macroparticle weights.
     * \return num_added                    number of particles added to each product slot.
     */
    AMREX_INLINE
    amrex::Vector<int> operator() (
        const index_type& n_total_pairs,
        ParticleTileType& ptile0, ParticleTileType& ptile1,
        const amrex::Vector<WarpXParticleContainer*>& pc_products,
        ParticleTileType** AMREX_RESTRICT tile_products,
        const amrex::ParticleReal m0, const amrex::ParticleReal m1,
        const amrex::Vector<amrex::ParticleReal>& /*products_mass*/,
        const index_type* AMREX_RESTRICT mask,
        amrex::Vector<index_type>& products_np,
        const SmartCopy* AMREX_RESTRICT copy_species0,
        const SmartCopy* AMREX_RESTRICT copy_species1,
        const index_type* AMREX_RESTRICT p_pair_indices_0,
        const index_type* AMREX_RESTRICT p_pair_indices_1,
        const amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight,
        const amrex::ParticleReal* /*p_product_data*/ ) const
    {
        using namespace amrex::literals;

        // Product species slot layout (m_num_product_species slots total):
        //   Slot 0: copy of the first reactant species, i.e. the species from ptile0 (always present)
        //   Slot 1: copy of the other reactant species, i.e. the species from ptile1 (always present)
        //   Slot 2: first true output species   (only when m_num_product_species == 4)
        //   Slot 3: second true output species  (only when m_num_product_species == 4)
        //
        // For non-reaction events (elastic, back, forward scattering), only slots 0 and 1
        // receive new particles: one weight-split copy per reactant per colliding pair.
        // These are counted by `no_product_total` and NOT reflected in `m_num_products_host`.
        //
        // For reaction events (ionization, two-product reaction, charge exchange), slots 2
        // and 3 receive one new particle per event. For ionization, slot 0 also receives one
        // particle (the surviving incident species copy). These counts are stored in
        // `m_num_products_host[i]` and multiplied by `with_product_total` to get the totals.

        // Return a vector of zeros, indicating that for all the "product" species
        // there were no new particles added.
        if (n_total_pairs == 0) { return amrex::Vector<int>(m_num_product_species, 0); }

        // The following is used to calculate the appropriate offsets for processes
        // that do not produce macroparticles in new species (i.e., not ionization, not two-product reaction).
        // Note that a standard cummulative sum is not appropriate since the
        // mask is also used to specify the type of collision and can therefore
        // have values >1
        amrex::Gpu::DeviceVector<index_type> no_product_offsets(n_total_pairs);
        index_type* AMREX_RESTRICT no_product_offsets_data = no_product_offsets.data();
        auto const no_product_total = amrex::Scan::PrefixSum<index_type>(n_total_pairs,
            [=] AMREX_GPU_DEVICE (index_type i) -> index_type {
                return ((mask[i] > 0) &
                        (mask[i] != int(ScatteringProcessType::IONIZATION)) &
                        (mask[i] != int(ScatteringProcessType::TWOPRODUCT_REACTION))) ? 1 : 0;
            },
            [=] AMREX_GPU_DEVICE (index_type i, index_type s) { no_product_offsets_data[i] = s; },
            amrex::Scan::Type::exclusive, amrex::Scan::retSum
        );

        amrex::Vector<int> num_added_vec(m_num_product_species, 0);
        for (int i = 0; i < 2; i++)
        {
            // Record the number of non-reaction events that lead to new particles
            // for the first and second reactant species (slots 0 and 1). Only 1
            // weight-split particle is created per reactant per event.
            num_added_vec[i] = static_cast<int>(no_product_total);
        }

        // The following is used to calculate the appropriate offsets for
        // product producing processes (i.e., ionization and two-product reaction).
        // Note that a standard cummulative sum is not appropriate since the
        // mask is also used to specify the type of collision and can therefore
        // have values >1
        amrex::Gpu::DeviceVector<index_type> with_product_offsets(n_total_pairs);
        index_type* AMREX_RESTRICT with_product_offsets_data = with_product_offsets.data();
        auto const with_product_total = amrex::Scan::PrefixSum<index_type>(n_total_pairs,
            [=] AMREX_GPU_DEVICE (index_type i) -> index_type {
                return ((mask[i] == int(ScatteringProcessType::IONIZATION)) |
                        (mask[i] == int(ScatteringProcessType::TWOPRODUCT_REACTION))) ? 1 : 0;
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

        const auto soa_0 = ptile0.getParticleTileData();
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

        const int num_product_species = m_num_product_species;
        const auto reaction_energy = m_reaction_energy;

        // Grab the masses of the true product species (slots 2 and 3)
        amrex::ParticleReal mass_slot2 = 0;
        amrex::ParticleReal mass_slot3 = 0;
        if (num_product_species > 2) {
            mass_slot2 = pc_products[2]->getMass();
            mass_slot3 = pc_products[3]->getMass();
        }

        // First perform all non-product producing collisions
        amrex::ParallelForRNG(n_total_pairs,
        [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
        {
            if ((mask[i] > 0) & (mask[i] != int(ScatteringProcessType::IONIZATION)) & (mask[i] != int(ScatteringProcessType::TWOPRODUCT_REACTION)))
            {
                const auto slot0_idx = products_np_data[0] + no_product_offsets_data[i];
                // Make a copy of the particle from the first reactant species
                copy_species0[0](soa_products_data[0], soa_0, static_cast<int>(p_pair_indices_0[i]),
                                static_cast<int>(slot0_idx), engine);
                // Set the weight of the new particles to p_pair_reaction_weight[i]
                soa_products_data[0].m_rdata[PIdx::w][slot0_idx] = p_pair_reaction_weight[i];

                const auto slot1_idx = products_np_data[1] + no_product_offsets_data[i];
                // Make a copy of the particle from the other reactant species
                copy_species1[1](soa_products_data[1], soa_1, static_cast<int>(p_pair_indices_1[i]),
                                static_cast<int>(slot1_idx), engine);
                // Set the weight of the new particles to p_pair_reaction_weight[i]
                soa_products_data[1].m_rdata[PIdx::w][slot1_idx] = p_pair_reaction_weight[i];

                // Set the child particle properties appropriately
                auto& ux0 = soa_products_data[0].m_rdata[PIdx::ux][slot0_idx];
                auto& uy0 = soa_products_data[0].m_rdata[PIdx::uy][slot0_idx];
                auto& uz0 = soa_products_data[0].m_rdata[PIdx::uz][slot0_idx];
                auto& ux1 = soa_products_data[1].m_rdata[PIdx::ux][slot1_idx];
                auto& uy1 = soa_products_data[1].m_rdata[PIdx::uy][slot1_idx];
                auto& uz1 = soa_products_data[1].m_rdata[PIdx::uz][slot1_idx];

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
                    soa_products_data[1].m_rdata[PIdx::theta][slot1_idx]
                    - soa_products_data[0].m_rdata[PIdx::theta][slot0_idx]
                );
                amrex::ParticleReal const ux0buf = ux0;
                ux0 = ux0buf*std::cos(theta) - uy0*std::sin(theta);
                uy0 = ux0buf*std::sin(theta) + uy0*std::cos(theta);
#endif

                // for simplicity (for now) we assume non-relativistic particles
                // and simply calculate the center-of-momentum velocity from the
                // rest masses
                // TODO: this could be made relativistic by using TwoProductComputeProductMomenta
                auto const uCOM_x = (m0 * ux0 + m1 * ux1) / (m0 + m1);
                auto const uCOM_y = (m0 * uy0 + m1 * uy1) / (m0 + m1);
                auto const uCOM_z = (m0 * uz0 + m1 * uz1) / (m0 + m1);

                // transform to COM frame
                ux0 -= uCOM_x;
                uy0 -= uCOM_y;
                uz0 -= uCOM_z;
                ux1 -= uCOM_x;
                uy1 -= uCOM_y;
                uz1 -= uCOM_z;

                if (mask[i] == int(ScatteringProcessType::ELASTIC)) {
                    // randomly rotate the velocity vector for the first particle
                    ParticleUtils::RandomizeVelocity(
                        ux0, uy0, uz0, std::sqrt(ux0*ux0 + uy0*uy0 + uz0*uz0), engine
                    );
                    // set the second particles velocity so that the total momentum
                    // is zero
                    ux1 = -ux0 * m0 / m1;
                    uy1 = -uy0 * m0 / m1;
                    uz1 = -uz0 * m0 / m1;
                } else if (mask[i] == int(ScatteringProcessType::BACK)) {
                    // reverse the velocity vectors of both particles
                    ux0 *= -1.0_prt;
                    uy0 *= -1.0_prt;
                    uz0 *= -1.0_prt;
                    ux1 *= -1.0_prt;
                    uy1 *= -1.0_prt;
                    uz1 *= -1.0_prt;
                } else if (mask[i] == int(ScatteringProcessType::FORWARD)) {
                    amrex::Abort("Forward scattering with DSMC not implemented yet.");
                }
                else {
                    amrex::Abort("Unknown scattering process.");
                }
                // transform back to labframe
                ux0 += uCOM_x;
                uy0 += uCOM_y;
                uz0 += uCOM_z;
                ux1 += uCOM_x;
                uy1 += uCOM_y;
                uz1 += uCOM_z;

#if (defined WARPX_DIM_RZ)
                /* Undo the earlier velocity rotation. */
                amrex::ParticleReal const ux0buf_new = ux0;
                ux0 = ux0buf_new*std::cos(-theta) - uy0*std::sin(-theta);
                uy0 = ux0buf_new*std::sin(-theta) + uy0*std::cos(-theta);
#endif
            }

            // Next perform all product-producing collisions
            else if (mask[i] == int(ScatteringProcessType::TWOPRODUCT_REACTION))
            {
                // create a copy of the first product species at the location of the first reactant species
                const auto src0_idx = static_cast<int>(p_pair_indices_0[i]);
                const auto slot2_idx = products_np_data[2] + with_product_offsets_data[i];
                copy_species0[2](soa_products_data[2], soa_0, src0_idx,
                                static_cast<int>(slot2_idx), engine);
                // Set the weight of the new particle to p_pair_reaction_weight[i]
                soa_products_data[2].m_rdata[PIdx::w][slot2_idx] = p_pair_reaction_weight[i];

                // create a copy of the other product species at the location of the other reactant species
                const auto src1_idx = static_cast<int>(p_pair_indices_1[i]);
                const auto slot3_idx = products_np_data[3] + with_product_offsets_data[i];
                copy_species1[3](soa_products_data[3], soa_1, src1_idx,
                                static_cast<int>(slot3_idx), engine);
                // Set the weight of the new particle to p_pair_reaction_weight[i]
                soa_products_data[3].m_rdata[PIdx::w][slot3_idx] = p_pair_reaction_weight[i];

                // Update the momenta of the products
                TwoProductComputeProductMomenta(
                    soa_0.m_rdata[PIdx::ux][src0_idx],
                    soa_0.m_rdata[PIdx::uy][src0_idx],
                    soa_0.m_rdata[PIdx::uz][src0_idx], m0,
                    soa_1.m_rdata[PIdx::ux][src1_idx],
                    soa_1.m_rdata[PIdx::uy][src1_idx],
                    soa_1.m_rdata[PIdx::uz][src1_idx], m1,
                    soa_products_data[2].m_rdata[PIdx::ux][slot2_idx],
                    soa_products_data[2].m_rdata[PIdx::uy][slot2_idx],
                    soa_products_data[2].m_rdata[PIdx::uz][slot2_idx], mass_slot2,
                    soa_products_data[3].m_rdata[PIdx::ux][slot3_idx],
                    soa_products_data[3].m_rdata[PIdx::uy][slot3_idx],
                    soa_products_data[3].m_rdata[PIdx::uz][slot3_idx], mass_slot3,
                    -reaction_energy*PhysConst::q_e,
                    // TwoProductComputeProductMomenta expects the *released* energy here, hence the negative sign
                    // We also convert from eV to Joules
                    false,
                    // no angular scattering: the products' momenta have the same direction
                    // as that of the incident particle, in the center-of-mass frame
                    engine);
            }
            else if (mask[i] == int(ScatteringProcessType::IONIZATION))
            {
                const auto slot0_idx = products_np_data[0] + no_product_total + with_product_offsets_data[i];
                // Make a copy of the particle from the first reactant species (incident species, slot 0)
                copy_species0[0](soa_products_data[0], soa_0, static_cast<int>(p_pair_indices_0[i]),
                               static_cast<int>(slot0_idx), engine);
                // Set the weight of the new particles to p_pair_reaction_weight[i]
                soa_products_data[0].m_rdata[PIdx::w][slot0_idx] = p_pair_reaction_weight[i];

                // create a copy of the first product species at the location of the other reactant species
                // (the other reactant species in slot 1 is the "target species", i.e. the species that is ionized)
                const auto slot2_idx = products_np_data[2] + with_product_offsets_data[i];
                copy_species1[2](soa_products_data[2], soa_1, static_cast<int>(p_pair_indices_1[i]),
                                static_cast<int>(slot2_idx), engine);
                // Set the weight of the new particle to p_pair_reaction_weight[i]
                soa_products_data[2].m_rdata[PIdx::w][slot2_idx] = p_pair_reaction_weight[i];

                // create a copy of the other product species at the location of the other reactant species
                // (the other reactant species in slot 1 is the "target species", i.e. the species that is ionized)
                const auto slot3_idx = products_np_data[3] + with_product_offsets_data[i];
                copy_species1[3](soa_products_data[3], soa_1, static_cast<int>(p_pair_indices_1[i]),
                                static_cast<int>(slot3_idx), engine);
                // Set the weight of the new particle to p_pair_reaction_weight[i]
                soa_products_data[3].m_rdata[PIdx::w][slot3_idx] = p_pair_reaction_weight[i];

                // Grab the colliding particle velocities to calculate the COM
                // Note that the two product particles currently have the same
                // velocity as the "target" particle
                auto& ux0 = soa_products_data[0].m_rdata[PIdx::ux][slot0_idx];
                auto& uy0 = soa_products_data[0].m_rdata[PIdx::uy][slot0_idx];
                auto& uz0 = soa_products_data[0].m_rdata[PIdx::uz][slot0_idx];
                auto& ux2 = soa_products_data[2].m_rdata[PIdx::ux][slot2_idx];
                auto& uy2 = soa_products_data[2].m_rdata[PIdx::uy][slot2_idx];
                auto& uz2 = soa_products_data[2].m_rdata[PIdx::uz][slot2_idx];
                auto& ux3 = soa_products_data[3].m_rdata[PIdx::ux][slot3_idx];
                auto& uy3 = soa_products_data[3].m_rdata[PIdx::uy][slot3_idx];
                auto& uz3 = soa_products_data[3].m_rdata[PIdx::uz][slot3_idx];

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
                    soa_products_data[2].m_rdata[PIdx::theta][slot2_idx]
                    - soa_products_data[0].m_rdata[PIdx::theta][slot0_idx]
                );
                amrex::ParticleReal const ux0buf = ux0;
                ux0 = ux0buf*std::cos(theta) - uy0*std::sin(theta);
                uy0 = ux0buf*std::sin(theta) + uy0*std::cos(theta);
#endif

                // for simplicity (for now) we assume non-relativistic particles
                // and simply calculate the center-of-momentum velocity from the
                // rest masses
                auto const uCOM_x = (m0 * ux0 + m1 * ux3) / (m0 + m1);
                auto const uCOM_y = (m0 * uy0 + m1 * uy3) / (m0 + m1);
                auto const uCOM_z = (m0 * uz0 + m1 * uz3) / (m0 + m1);

                // transform to COM frame
                ux0 -= uCOM_x;
                uy0 -= uCOM_y;
                uz0 -= uCOM_z;
                ux2 -= uCOM_x;
                uy2 -= uCOM_y;
                uz2 -= uCOM_z;
                ux3 -= uCOM_x;
                uy3 -= uCOM_y;
                uz3 -= uCOM_z;

                // calculate kinetic energy of the collision (in eV)
                const amrex::ParticleReal E1 = (
                    0.5_prt * m0 * (ux0*ux0 + uy0*uy0 + uz0*uz0) / PhysConst::q_e
                );
                const amrex::ParticleReal E2 = (
                    0.5_prt * m1 * (ux3*ux3 + uy3*uy3 + uz3*uz3) / PhysConst::q_e
                );
                const amrex::ParticleReal E_coll = E1 + E2;

                // subtract the energy cost for ionization
                const amrex::ParticleReal E_out = (E_coll - reaction_energy) * PhysConst::q_e;

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

                const double n_2 = std::min<double>(E2 / E_coll, 0.5 * amrex::Random(engine));

                // find ellipse semi-major and minor axis
                const double a = 0.5 * (1.0 - n_2);
                const double b = 0.5 * std::sqrt(1.0 - 2.0 * n_2);

                // sample random x value and calculate y
                const double x = (2.0 * amrex::Random(engine) - 1.0) * a;
                const double y2 = b*b - b*b/(a*a) * x*x;
                const double n_0 = std::sqrt(y2 + x*x - x*n_2 + 0.25*n_2*n_2);
                const double n_1 = 1.0 - n_0 - n_2;

                // calculate the value of C
                const double C = std::sqrt(E_out / (
                    n_0*n_0 / (2.0 * m0) + n_1*n_1 / (2.0 * mass_slot2) + n_2*n_2 / (2.0 * mass_slot3)
                ));

                // Now that appropriate momenta are set for each outgoing species
                // the directions for the velocity vectors must be chosen such
                // that the net linear momentum in the current frame is 0.
                // This is achieved by arranging the momentum vectors in
                // a triangle and finding the required angles between the vectors.
                const double cos_alpha = (n_0*n_0 + n_1*n_1 - n_2*n_2) / (2.0 * n_0 * n_1);
                const double sin_alpha = std::sqrt(1.0 - cos_alpha*cos_alpha);
                const double cos_gamma = (n_0*n_0 + n_2*n_2 - n_1*n_1) / (2.0 * n_0 * n_2);
                const double sin_gamma = std::sqrt(1.0 - cos_gamma*cos_gamma);

                // choose random theta and phi values (orientation of the triangle)
                const double Theta = amrex::Random(engine) * 2.0 * MathConst::pi;
                const double phi = amrex::Random(engine) * MathConst::pi;

                const double cos_theta = std::cos(Theta);
                const double sin_theta = std::sin(Theta);
                const double cos_phi = std::cos(phi);
                const double sin_phi = std::sin(phi);

                // calculate the velocity components for each particle
                ux0 = static_cast<amrex::ParticleReal>(C * n_0 / m0 * cos_theta * cos_phi);
                uy0 = static_cast<amrex::ParticleReal>(C * n_0 / m0 * cos_theta * sin_phi);
                uz0 = static_cast<amrex::ParticleReal>(-C * n_0 / m0 * sin_theta);

                ux2 = static_cast<amrex::ParticleReal>(C * n_1 / mass_slot2 * (-cos_alpha * cos_theta * cos_phi - sin_alpha * sin_phi));
                uy2 = static_cast<amrex::ParticleReal>(C * n_1 / mass_slot2 * (-cos_alpha * cos_theta * sin_phi + sin_alpha * cos_phi));
                uz2 = static_cast<amrex::ParticleReal>(C * n_1 / mass_slot2 * (cos_alpha * sin_theta));

                ux3 = static_cast<amrex::ParticleReal>(C * n_2 / mass_slot3 * (-cos_gamma * cos_theta * cos_phi + sin_gamma * sin_phi));
                uy3 = static_cast<amrex::ParticleReal>(C * n_2 / mass_slot3 * (-cos_gamma * cos_theta * sin_phi - sin_gamma * cos_phi));
                uz3 = static_cast<amrex::ParticleReal>(C * n_2 / mass_slot3 * (cos_gamma * sin_theta));

                // transform back to labframe
                ux0 += uCOM_x;
                uy0 += uCOM_y;
                uz0 += uCOM_z;
                ux2 += uCOM_x;
                uy2 += uCOM_y;
                uz2 += uCOM_z;
                ux3 += uCOM_x;
                uy3 += uCOM_y;
                uz3 += uCOM_z;

#if (defined WARPX_DIM_RZ)
                /* Undo the earlier velocity rotation. */
                amrex::ParticleReal const ux0buf_new = ux0;
                ux0 = ux0buf_new*std::cos(-theta) - uy0*std::sin(-theta);
                uy0 = ux0buf_new*std::sin(-theta) + uy0*std::cos(-theta);
#endif
            }
        });

        // Initialize the user runtime components
        for (int i = 0; i < m_num_product_species; i++)
        {
            const int start_index = int(products_np[i]);
            const int stop_index  = int(products_np[i] + num_added_vec[i]);
            ParticleCreation::DefaultInitializeRuntimeAttributes(*tile_products[i],
                                                 *pc_products[i], start_index, stop_index);
        }

        amrex::Gpu::synchronize();
        return num_added_vec;
    }

private:
    // How many different type of species the collision produces
    int m_num_product_species;
    // If ionization collisions are included, what is the energy cost
    amrex::ParticleReal m_reaction_energy = 0.0;
    // Vector of size m_num_product_species: for each product slot, the number of new particles
    // created per *reaction* event (ionization, two-product reaction, charge exchange).
    // Weight-split particles added to slots 0 and 1 during non-reaction events are NOT counted
    // here; they are tracked separately via `no_product_total` inside operator().
    amrex::Gpu::HostVector<int> m_num_products_host;
    CollisionType m_collision_type;
};
#endif // WARPX_SPLIT_AND_SCATTER_FUNC_H_
