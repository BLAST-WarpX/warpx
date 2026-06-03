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
#include "Utils/WarpXAlgorithmSelection.H"

#include <array>

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
     * \brief Function that performs binary collision between macroparticle pairs
     *  (referred to as "reactant species" in the code below)
     *
     * During a collision, new macroparticles can be created in different species,
     * which are arranged in slots 0 through m_num_product_species-1.
     *
     * Slots 0 and 1 always correspond to the first and second reactant species, respectively.
     * This is because, physically, some of the reacting particles can still be present after the collision
     * (e.g., if the collision is elastic scattering), but with modified properties (e.g., modified momentum).
     * In the code, this is handled by creating *new* macroparticles in the reactant species, whose weight
     * is subtracted from the reacting macroparticles. (This approach is used in order to handle the fact
     * that the two reactant macroparticles may have different weights.)
     *
     * For collisions that, physically, produce new particles (e.g., ionization, charge exchange, two-product
     * reaction), slots 2 and 3 hold the true product species.
     *
     * @param[in]     n_total_pairs         total number of colliding particle pairs
     *                                      (across all cells) handled by this kernel.
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
     * @param[in,out] products_np           macroparticle count of each product tile; updated in this
     *                                      kernel to reflect the number of new macroparticles added in each species.
     * @param[in]     copy_species0         array of SmartCopy functors (size m_num_product_species);
     *                                      copy_species0[k] copies a particle FROM the first
     *                                      reactant species (slot 0) INTO product slot k.
     * @param[in]     copy_species1         array of SmartCopy functors (size m_num_product_species);
     *                                      copy_species1[k] copies a particle FROM the other
     *                                      reactant species (slot 1) INTO product slot k.
     * @param[in]     p_pair_indices_0      per-pair index of the first reactant species (slot 0) particle.
     * @param[in]     p_pair_indices_1      per-pair index of the other reactant species (slot 1) particle.
     * @param[in]     p_pair_reaction_weight per-pair weight assigned to newly created product
     *                                      macroparticles; also the amount subtracted from the
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
        //   Slot 2: first true product species
        //   Slot 3: second true product species
        //
        // For non-product producing collisions (elastic, back, forward scattering), only slots 0
        // and 1 receive new macroparticles: one weight-split copy per reactant per colliding pair.
        // These are counted by `no_product_total` and NOT reflected in `m_num_products_host`.
        //
        // For product-producing collisions (ionization, two-product reaction, charge exchange),
        // slots 2 and 3 receive one new particle per event. For ionization, slot 0 also receives one
        // particle (the surviving reactant species copy). These counts are stored in
        // `m_num_products_host[i]` and multiplied by `with_product_total` to get the totals.

        // If there are no colliding pairs, skip the rest of this kernel and return a vector of
        // zeros, indicating that for all the "product" species, there were no new macroparticles added.
        if (n_total_pairs == 0) { return amrex::Vector<int>(m_num_product_species, 0); }

        // The following is used to calculate the appropriate offsets for
        // non-product producing collisions (i.e., not ionization, not two-product reaction).
        // Note that a standard cummulative sum is not appropriate since the
        // mask is also used to specify the type of collision and can therefore
        // have values >1
        amrex::Gpu::DeviceVector<index_type> no_product_offsets(n_total_pairs);
        index_type* AMREX_RESTRICT no_product_offsets_data = no_product_offsets.data();
        auto const no_product_total = amrex::Scan::PrefixSum<index_type>(n_total_pairs,
            [=] AMREX_GPU_DEVICE (index_type i) -> index_type {
                return ((mask[i] > 0) &&
                        (mask[i] != int(ScatteringProcessType::IONIZATION)) &&
                        (mask[i] != int(ScatteringProcessType::TWOPRODUCT_REACTION)) &&
                        (mask[i] != int(ScatteringProcessType::CHARGE_EXCHANGE))) ? 1 : 0;
            },
            [=] AMREX_GPU_DEVICE (index_type i, index_type s) { no_product_offsets_data[i] = s; },
            amrex::Scan::Type::exclusive, amrex::Scan::retSum
        );

        amrex::Vector<int> num_added_vec(m_num_product_species, 0);
        for (int i = 0; i < 2; i++)
        {
            // Record the number of non-product producing collisions that lead to new particles
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
                        (mask[i] == int(ScatteringProcessType::TWOPRODUCT_REACTION)) |
                        (mask[i] == int(ScatteringProcessType::CHARGE_EXCHANGE))) ? 1 : 0;
            },
            [=] AMREX_GPU_DEVICE (index_type i, index_type s) { with_product_offsets_data[i] = s; },
            amrex::Scan::Type::exclusive, amrex::Scan::retSum
        );

        for (int i = 0; i < m_num_product_species; i++)
        {
            // Add the number of product producing events to the species involved in those processes.
            const int num_products = m_num_products_host[i];
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
            if ((mask[i] > 0) && (mask[i] != int(ScatteringProcessType::IONIZATION)) && (mask[i] != int(ScatteringProcessType::TWOPRODUCT_REACTION)) && (mask[i] != int(ScatteringProcessType::CHARGE_EXCHANGE)))
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

            }

            // Next perform all product-producing collisions
            else if ((mask[i] == int(ScatteringProcessType::TWOPRODUCT_REACTION)) ||
                     (mask[i] == int(ScatteringProcessType::CHARGE_EXCHANGE)))
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

                // For charge exchange, swap positions at which the products are created
                // to ensure local charge conservation. For instance in the reaction
                // `A+ + B -> A + B+`, B+ will be created at the position of A+, instead of that of B.
                if (mask[i] == int(ScatteringProcessType::CHARGE_EXCHANGE)) {
                    // The SoA components that define the positions depend on the geometry.
#if defined(WARPX_DIM_1D_Z)
                    constexpr std::array<int, 1> position_indices = {PIdx::z};
#elif defined(WARPX_DIM_XZ)
                    constexpr std::array<int, 2> position_indices = {PIdx::x, PIdx::z};
#elif defined(WARPX_DIM_RZ)
                    constexpr std::array<int, 3> position_indices = {PIdx::x, PIdx::z, PIdx::theta};
#elif defined(WARPX_DIM_RCYLINDER)
                    constexpr std::array<int, 2> position_indices = {PIdx::x, PIdx::theta};
#elif defined(WARPX_DIM_RSPHERE)
                    constexpr std::array<int, 3> position_indices = {PIdx::x, PIdx::theta, PIdx::phi};
#elif defined(WARPX_DIM_3D)
                    constexpr std::array<int, 3> position_indices = {PIdx::x, PIdx::y, PIdx::z};
#endif
                    // Loop over the components that define the positions and swap the positions of the products.
                    for (int const idim : position_indices) {
                        soa_products_data[2].m_rdata[idim][slot2_idx] = soa_1.m_rdata[idim][src1_idx];
                        soa_products_data[3].m_rdata[idim][slot3_idx] = soa_0.m_rdata[idim][src0_idx];
                    }
                }

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
                    ScatteringAngleModel::Forward,
                    // no angular scattering: the products' momenta have the same direction
                    // as that of the reactant particle, in the center-of-mass frame
                    engine);
            }
            else if (mask[i] == int(ScatteringProcessType::IONIZATION))
            {
                const auto slot0_idx = products_np_data[0] + no_product_total + with_product_offsets_data[i];
                // Make a copy of the particle from the first reactant species (slot 0)
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

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                /* In RZ and RCYLINDER geometry, macroparticles can collide with other macroparticles
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

                // calculate kinetic energy of the collision
                const amrex::ParticleReal p_non_target_in = m0 * std::sqrt(ux0*ux0 + uy0*uy0 + uz0*uz0);
                const amrex::ParticleReal E_coll = 0.5_prt * p_non_target_in * p_non_target_in * (1.0_prt/m0 + 1.0_prt/m1);
                // subtract the energy cost for ionization (converted from eV to J)
                const amrex::ParticleReal E_out = E_coll - reaction_energy * PhysConst::q_e;

                // The kinematics of the ionization event (i.e. how the energy and momentum are
                // distributed among the products) depend on the nature of the incident particle
                // (first reactant species, slot 0, with mass m0).
                if (m0 > 2.0_prt * PhysConst::m_e)
                {
                    // Ion impact ionization (the incident particle is heavier than an electron).
                    // We use the collinear model: in the center-of-mass frame, the two products
                    // (the ion and the electron) are assumed to have equal velocity, and their
                    // velocity vector is aligned with that of the incident (non-target) particle.

                    // Momentum and energy division after the ionization event is done as follows:
                    // we assume that, in the center of mass frame, the two products (the ion and the electron)
                    // have equal velocity and that their velocity vector is aligned with that of the target particle

                    // With these assumptions, conservation of momentum in the center of mass frame allows to express the momenta
                    // of the products, as a function of the outcoming momentum of the non-target particle:
                    // p2 = - p_non_target_out * mass_slot2 / m1
                    // p3 = - p_non_target_out * mass_slot3 / m1
                    // (with m1 = mass_slot2 + mass_slot3, by conservation of mass)

                    // Amplitude of the momentum of the non-target particle, obtained from conservation of energy
                    const amrex::ParticleReal p_non_target_out = std::sqrt( 2.0_prt * E_out / (1.0_prt / m0 + 1.0_prt / (mass_slot2 + mass_slot3) ) );

                    // Reduce the velocity of the non-target particle, to reflect the loss of energy due to ionization
                    const amrex::ParticleReal reduce_factor = p_non_target_out / p_non_target_in;
                    ux0 *= reduce_factor;
                    uy0 *= reduce_factor;
                    uz0 *= reduce_factor;
                    // Obtain the other velocity vectors by conservation of momentum
                    const amrex::ParticleReal mass_ratio = m0 / (mass_slot2 + mass_slot3);
                    ux2 = -mass_ratio * ux0;
                    uy2 = -mass_ratio * uy0;
                    uz2 = -mass_ratio * uz0;
                    ux3 = -mass_ratio * ux0;
                    uy3 = -mass_ratio * uy0;
                    uz3 = -mass_ratio * uz0;
                }
                else
                {
                    // Electron impact ionization (the incident particle is an electron).
                    // We use the following model: in the center-of-mass frame, the two
                    // electrons (the incident electron and the newly freed electron)
                    // are scattered isotropically and are assumed to carry an equal share
                    // of the energy. All the other unknowns (magnitude of the isotropically
                    // scattered electron momentum, momentum of the ion) are determined by
                    // conservation of energy and momentum.

                    // The freed electron has the same mass as the incident electron (m0);
                    // the ion is the heavier of the two products in slots 2 and 3.
                    const amrex::ParticleReal m_ion = amrex::max(mass_slot2, mass_slot3);

                    // Sample two isotropic unit vectors: one for the incident electron
                    // (slot 0) and one for the newly freed electron.
                    // These units vectors have the right direction, but not the right magnitude ;
                    // they will be multiplied by the right magnitude p_e further down.
                    amrex::ParticleReal scaled_ux0, scaled_uy0, scaled_uz0;
                    amrex::ParticleReal scaled_ux_new, scaled_uy_new, scaled_uz_new;
                    ParticleUtils::getRandomVector(scaled_ux0, scaled_uy0, scaled_uz0, engine);
                    ParticleUtils::getRandomVector(scaled_ux_new, scaled_uy_new, scaled_uz_new, engine);

                    // Sum of the two electron directions; the ion momentum is
                    // proportional to its opposite (conservation of momentum in COM).
                    const amrex::ParticleReal scaled_ux_ion = -(scaled_ux0 + scaled_ux_new);
                    const amrex::ParticleReal scaled_uy_ion = -(scaled_uy0 + scaled_uy_new);
                    const amrex::ParticleReal scaled_uz_ion = -(scaled_uz0 + scaled_uz_new);
                    const amrex::ParticleReal scaled_u2_ion = scaled_ux_ion*scaled_ux_ion + scaled_uy_ion*scaled_uy_ion + scaled_uz_ion*scaled_uz_ion;

                    // Magnitude of the momentum of the two electrons, from conservation
                    // of energy in the center-of-mass frame (both electrons are assumed
                    // to carry an equal share of the energy):
                    //     2 * p_e^2 / (2 m_e) + p_ion^2 / (2 m_ion) = E_out
                    const amrex::ParticleReal p_e = std::sqrt(
                        E_out / (1.0_prt/m0 + 0.5_prt*scaled_u2_ion/m_ion) );

                    // Assign the normalized momenta (still in the COM frame; the surrounding code
                    // transforms them back to the lab frame). The incident electron is in
                    // slot 0; the two products of the ionized target (slots 2 and 3) are
                    // one electron and one ion, the electron being the lighter of the two.
                    ux0 = p_e/m0 * scaled_ux0;
                    uy0 = p_e/m0 * scaled_uy0;
                    uz0 = p_e/m0 * scaled_uz0;
                    if (mass_slot2 < mass_slot3) {
                        // slot 2 is the freed electron, slot 3 is the ion
                        ux2 = p_e/m0 * scaled_ux_new;
                        uy2 = p_e/m0 * scaled_uy_new;
                        uz2 = p_e/m0 * scaled_uz_new;
                        ux3 = p_e/m_ion * scaled_ux_ion;
                        uy3 = p_e/m_ion * scaled_uy_ion;
                        uz3 = p_e/m_ion * scaled_uz_ion;
                    } else {
                        // slot 3 is the freed electron, slot 2 is the ion
                        ux3 = p_e/m0 * scaled_ux_new;
                        uy3 = p_e/m0 * scaled_uy_new;
                        uz3 = p_e/m0 * scaled_uz_new;
                        ux2 = p_e/m_ion * scaled_ux_ion;
                        uy2 = p_e/m_ion * scaled_uy_ion;
                        uz2 = p_e/m_ion * scaled_uz_ion;
                    }
                }

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

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
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
