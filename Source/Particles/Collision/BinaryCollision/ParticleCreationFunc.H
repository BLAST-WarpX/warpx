/* Copyright 2021 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_PARTICLE_CREATION_FUNC_H_
#define WARPX_PARTICLE_CREATION_FUNC_H_

#include "BinaryCollisionUtils.H"

#include "Particles/Collision/BinaryCollision/NuclearFusion/TwoProductFusionInitializeMomentum.H"
#include "Particles/Collision/BinaryCollision/NuclearFusion/ProtonBoronFusionInitializeMomentum.H"
#include "Particles/Collision/BinaryCollision/LinearBreitWheeler/LinearBreitWheelerInitializeMomentum.H"
#include "Particles/Collision/BinaryCollision/LinearCompton/LinearComptonInitializeMomentum.H"
#include "Particles/ParticleCreation/SmartCopy.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/ScatteringUtils.H"

#include <AMReX_DenseBins.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_INT.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>
#include <AMReX_Vector.H>

/**
 * \brief This functor creates particles produced from a binary collision and sets their initial
 * properties (position, momentum, weight).
 */
class ParticleCreationFunc
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
     * \brief Default constructor of the ParticleCreationFunc class.
     */
    ParticleCreationFunc () = default;

    /**
     * \brief Constructor of the ParticleCreationFunc class
     *
     * @param[in] collision_name the name of the collision
     * @param[in] mypc pointer to the MultiParticleContainer
     */
    ParticleCreationFunc (const std::string& collision_name, MultiParticleContainer const * mypc);

    /**
     * \brief operator() of the ParticleCreationFunc class. It creates new particles from binary
     * collisions.
     * For two-product processes in which the charge of each product matches the charge of one
     * of the reactants (linear Compton, D+D -> T+p, D+He3 -> He4+p), one macroparticle per
     * product species is created, at the position of the reactant with the same charge. This
     * conserves charge locally while creating half as many macroparticles.
     * For all other binary processes (linear Breit-Wheeler, D+T -> n+He4, D+D -> n+He3,
     * proton-boron fusion), one product is created at the position of each
     * parent particle that collided (e.g. 2 photons and 2 electrons for Breit-Wheeler), with half
     * of the reaction weight each, allowing for exact charge conservation. For example, in the
     * nuclear reaction "proton + boron -> 3 alpha", we create 6 new alpha particles, 3 at the
     * position of the proton and 3 at the position of the boron.
     * This function also sets the initial weight of the produced particles and subtracts it from
     * the parent particles. If the weight of a parent particle becomes 0, then that particle is
     * deleted.
     * Finally, this function sets the initial momentum of the product particles, by calling a
     * function specific to the considered binary collision.
     *
     * @param[in] n_total_pairs how many binary collisions have been performed in this tile
     * @param[in, out] ptile1,ptile2 the particle tiles of the two colliding species
     * @param[in, out] pc_products particle container of the product particles.
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
     * @param[in] p_pair_reaction_weight array that stores the weight of the binary collisions.
     * This weight is removed from the parent particles and given to the product particles.
     */
    AMREX_INLINE
    amrex::Vector<int> operator() (
                    const index_type& n_total_pairs,
                    ParticleTileType& ptile1, ParticleTileType& ptile2,
                    const amrex::Vector<WarpXParticleContainer*>& pc_products,
                    ParticleTileType** AMREX_RESTRICT tile_products,
                    const amrex::ParticleReal& m1, const amrex::ParticleReal& m2,
                    const amrex::Vector<amrex::ParticleReal>& products_mass,
                    const index_type* AMREX_RESTRICT p_mask,
                    const amrex::Vector<index_type>& products_np,
                    const SmartCopy* AMREX_RESTRICT copy_species1,
                    const SmartCopy* AMREX_RESTRICT copy_species2,
                    const index_type* AMREX_RESTRICT p_pair_indices_1,
                    const index_type* AMREX_RESTRICT p_pair_indices_2,
                    const amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight,
                    const amrex::ParticleReal* /*p_product_data*/
                    ) const
    {
        using namespace amrex::literals;

        if (n_total_pairs == 0) { return amrex::Vector<int>(m_num_product_species, 0); }

        // Compute offset array and allocate memory for the produced species
        amrex::Gpu::DeviceVector<index_type> offsets(n_total_pairs);
        const auto total = amrex::Scan::ExclusiveSum(n_total_pairs, p_mask, offsets.data());
        const index_type* AMREX_RESTRICT p_offsets = offsets.dataPtr();
        // How many particles of each product species are created.
        // If each product has the same charge as one of the reactants (e.g. linear Compton,
        // D+D -> T+p, D+He3 -> He4+p): one macroparticle per product species, created at the
        // position of the reactant with the same charge (factor 1).
        // Otherwise: create one product particle at the position of each reacting
        // particle (factor 2), with half of the reaction weight each.
        // E.g., if a binary collision produces one electron, we create two electrons,
        // one at the position of each particle that collided.
        // Both choices ensure exact local charge conservation.
        const int products_per_reactant_factor = m_products_per_reactant_factor;
        amrex::Vector<int> num_added_vec(m_num_product_species);
        for (int i = 0; i < m_num_product_species; i++)
        {
            const index_type num_added = total * m_num_products_host[i] * products_per_reactant_factor;
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
        amrex::Gpu::DeviceVector<amrex::ParticleReal> device_products_mass(m_num_product_species);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, soa_products.begin(),
                              soa_products.end(),
                              device_soa_products.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, products_np.begin(),
                              products_np.end(),
                              device_products_np.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, products_mass.begin(),
                              products_mass.end(),
                              device_products_mass.begin());
        amrex::Gpu::streamSynchronize();
        SoaData_type* AMREX_RESTRICT soa_products_data = device_soa_products.data();
        const index_type* AMREX_RESTRICT products_np_data = device_products_np.data();
        const amrex::ParticleReal* AMREX_RESTRICT products_mass_data = device_products_mass.data();
#else
        SoaData_type* AMREX_RESTRICT soa_products_data = soa_products.data();
        const index_type* AMREX_RESTRICT products_np_data = products_np.data();
        const amrex::ParticleReal* AMREX_RESTRICT products_mass_data = products_mass.data();
#endif

        const int t_num_product_species = m_num_product_species;
        const int* AMREX_RESTRICT p_num_products_device = m_num_products_device.data();
        const CollisionType t_collision_type = m_collision_type;
        const bool t_swap_product_positions = m_swap_product_positions;
        const ScatteringAngleModel t_scattering_angle_model = m_scattering_angle_model;
        const ScatteringUtils::AnisotropicCoefficientTable t_anisotropic_coefficients{
            m_legendre_angular_distribution_energies.empty() ? nullptr : m_legendre_angular_distribution_energies.data(),
            m_legendre_angular_distribution_coefficients.empty() ? nullptr : m_legendre_angular_distribution_coefficients.data(),
            m_legendre_angular_distribution_num_energies,
            m_legendre_angular_distribution_num_coefficients
        };
        bool const check_anisotropic_energy_range = t_scattering_angle_model == ScatteringAngleModel::Legendre;
        amrex::Gpu::DeviceVector<int> energy_range_status(check_anisotropic_energy_range ? n_total_pairs : 0);
        int* const p_energy_range_status = check_anisotropic_energy_range ? energy_range_status.data() : nullptr;

        amrex::ParallelForRNG(n_total_pairs,
        [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
        {
            int event_energy_range_status = 0;
            if (p_mask[i])
            {
                for (int j = 0; j < t_num_product_species; j++)
                {
                    for (int k = 0; k < p_num_products_device[j]; k++)
                    {
                        if (products_per_reactant_factor == 1)
                        {
                            // One macroparticle per product species (only used for
                            // two-product processes), each created at the position of
                            // the reactant that has the same charge: by default, the first
                            // product at the first reactant and the second product at the
                            // second reactant (swapped if t_swap_product_positions).
                            const auto product_index = products_np_data[j] +
                                (p_offsets[i]*p_num_products_device[j] + k);
                            const bool from_first_reactant = ((j == 0) != t_swap_product_positions);
                            if (from_first_reactant)
                            {
                                copy_species1[j](soa_products_data[j], soa_1,
                                    static_cast<int>(p_pair_indices_1[i]),
                                    static_cast<int>(product_index), engine);
                            }
                            else
                            {
                                copy_species2[j](soa_products_data[j], soa_2,
                                    static_cast<int>(p_pair_indices_2[i]),
                                    static_cast<int>(product_index), engine);
                            }
                            soa_products_data[j].m_rdata[PIdx::w][product_index] =
                                p_pair_reaction_weight[i];
                        }
                        else
                        {
                            // Create one product at the position of each source particle
                            const auto product_index = products_np_data[j] +
                                2*(p_offsets[i]*p_num_products_device[j] + k);
                            copy_species1[j](soa_products_data[j], soa_1,
                                static_cast<int>(p_pair_indices_1[i]),
                                static_cast<int>(product_index), engine);
                            copy_species2[j](soa_products_data[j], soa_2,
                                static_cast<int>(p_pair_indices_2[i]),
                                static_cast<int>(product_index + 1), engine);
                            soa_products_data[j].m_rdata[PIdx::w][product_index] =
                                p_pair_reaction_weight[i]/amrex::ParticleReal(2.);
                            soa_products_data[j].m_rdata[PIdx::w][product_index + 1] =
                                p_pair_reaction_weight[i]/amrex::ParticleReal(2.);
                        }
                    }
                }

                // Initialize the product particles' momentum, using a function depending on the
                // specific collision type
                if (t_collision_type == CollisionType::ProtonBoronToAlphasFusion)
                {
                    const index_type product_start_index = products_np_data[0] + 2*p_offsets[i]*
                                                           p_num_products_device[0];
                    ProtonBoronFusionInitializeMomentum(soa_1, soa_2, soa_products_data[0],
                                                        p_pair_indices_1[i], p_pair_indices_2[i],
                                                        product_start_index, m1, m2,
                                                        t_scattering_angle_model,
                                                        t_anisotropic_coefficients,
                                                        &event_energy_range_status, engine);
                }
                else if (BinaryCollisionUtils::is_two_product_fusion_type(t_collision_type))
                {
                    const amrex::ParticleReal mass_before = m1 + m2;
                    const amrex::ParticleReal mass_after = products_mass_data[0] + products_mass_data[1];
                    const amrex::ParticleReal fusion_energy = (mass_before - mass_after)*PhysConst::c2;
                    TwoProductFusionInitializeMomentum(soa_1, soa_2,
                        soa_products_data[0], soa_products_data[1],
                        p_pair_indices_1[i], p_pair_indices_2[i],
                        products_np_data[0] + products_per_reactant_factor*p_offsets[i]*p_num_products_device[0],
                        products_np_data[1] + products_per_reactant_factor*p_offsets[i]*p_num_products_device[1],
                        products_per_reactant_factor,
                        m1, m2, products_mass_data[0], products_mass_data[1], fusion_energy,
                        t_scattering_angle_model, t_anisotropic_coefficients,
                        &event_energy_range_status, engine);
                }
                else if(t_collision_type == CollisionType::LinearBreitWheeler){
                    LinearBreitWheelerInitializeMomentum(soa_1, soa_2,
                        soa_products_data[0], soa_products_data[1],
                        p_pair_indices_1[i], p_pair_indices_2[i],
                        products_np_data[0] + 2*p_offsets[i]*p_num_products_device[0],
                        products_np_data[1] + 2*p_offsets[i]*p_num_products_device[1],
                        engine);
                }
                else if(t_collision_type == CollisionType::LinearCompton){
                    LinearComptonInitializeMomentum(soa_1, soa_2,
                        soa_products_data[0], soa_products_data[1],
                        p_pair_indices_1[i], p_pair_indices_2[i],
                        products_np_data[0] + p_offsets[i]*p_num_products_device[0],
                        products_np_data[1] + p_offsets[i]*p_num_products_device[1],
                        engine);
                }
            }
            if (p_energy_range_status != nullptr) {
                p_energy_range_status[i] = event_energy_range_status;
            }
        });

        if (check_anisotropic_energy_range) {
            auto const [minimum_status, maximum_status] = amrex::Reduce::MinMax(n_total_pairs, p_energy_range_status);
            RecordEnergyRangeWarnings(minimum_status, maximum_status);
        }

        // Initialize the user runtime components
        for (int i = 0; i < m_num_product_species; i++)
        {
            const auto start_index = int(products_np[i]);
            const auto stop_index  = int(products_np[i] + num_added_vec[i]);
            ParticleCreation::DefaultInitializeRuntimeAttributes(*tile_products[i],
                                                 *pc_products[i], start_index, stop_index);
        }

        amrex::Gpu::synchronize();

        return num_added_vec;
    }

private:
    static void RecordEnergyRangeWarnings (int minimum_status, int maximum_status);

    // How many different type of species the collision produces
    int m_num_product_species;
    // Vectors of size m_num_product_species storing how many particles of a given species are
    // produced by a collision event. These vectors are duplicated (one version for host and one
    // for device) which is necessary with GPUs but redundant on CPU.
    amrex::Gpu::DeviceVector<int> m_num_products_device;
    amrex::Gpu::HostVector<int> m_num_products_host;
    CollisionType m_collision_type;
    // Number of macroparticles created per product, per reaction and per reactant:
    // 1 when each product of a two-product process can be placed at the position of a
    // reactant with the same charge, 2 otherwise (one product at each reactant position).
    int m_products_per_reactant_factor = 2;
    // Only used when m_products_per_reactant_factor == 1: if true, the first product is
    // created at the position of the second reactant and vice versa.
    bool m_swap_product_positions = false;
    ScatteringAngleModel m_scattering_angle_model{ScatteringAngleModel::Default};
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_legendre_angular_distribution_energies;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_legendre_angular_distribution_coefficients;
    int m_legendre_angular_distribution_num_energies = 0;
    int m_legendre_angular_distribution_num_coefficients = 0;
};


/**
 * \brief This class does nothing and is used as second template parameter for binary collisions
 * that do not create particles.
 */
class NoParticleCreationFunc
{
    using ParticleType = typename WarpXParticleContainer::ParticleType;
    using ParticleTileType = typename WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = typename ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = typename ParticleBins::index_type;
    using SoaData_type = typename WarpXParticleContainer::ParticleTileType::ParticleTileDataType;

public:
    NoParticleCreationFunc () = default;

    NoParticleCreationFunc (const std::string& /*collision_name*/,
                            MultiParticleContainer const * const /*mypc*/) {}

    AMREX_INLINE
    amrex::Vector<int> operator() (
                    const index_type& /*n_total_pairs*/,
                    ParticleTileType& /*ptile1*/, ParticleTileType& /*ptile2*/,
                    amrex::Vector<WarpXParticleContainer*>& /*pc_products*/,
                    ParticleTileType** /*tile_products*/,
                    const amrex::ParticleReal& /*m1*/, const amrex::ParticleReal& /*m2*/,
                    const amrex::Vector<amrex::ParticleReal>& /*products_mass*/,
                    const index_type* /*p_mask*/, const amrex::Vector<index_type>& /*products_np*/,
                    const SmartCopy* /*copy_species1*/, const SmartCopy* /*copy_species2*/,
                    const index_type* /*p_pair_indices_1*/, const index_type* /*p_pair_indices_2*/,
                    const amrex::ParticleReal* /*p_pair_reaction_weight*/,
                    const amrex::ParticleReal* AMREX_RESTRICT /*p_product_data*/
                    ) const
    {
        return {};
    }
};

#endif // WARPX_PARTICLE_CREATION_FUNC_H_
