/* Copyright 2021 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_NUCLEAR_FUSION_FUNC_H_
#define WARPX_NUCLEAR_FUSION_FUNC_H_

#include "SingleNuclearFusionEvent.H"

#include "Particles/Collision/CollisionFuncBase.H"
#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

#include <AMReX_Algorithm.H>
#include <AMReX_DenseBins.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

/**
 * \brief This functor does binary nuclear fusions on a single cell.
 *  Particles of the two reacting species are paired with each other and for each pair we compute
 *  if a fusion event occurs. If so, we fill a mask (input parameter p_mask) with true so that
 *  product particles corresponding to a given pair can be effectively created in the particle
 *  creation functor.
 *  This functor also reads and contains the fusion multiplier.
 */
class NuclearFusionFunc
    : public CollisionFuncBase
{
    // Define shortcuts for frequently-used type names
    using ParticleType = WarpXParticleContainer::ParticleType;
    using ParticleTileType = WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = ParticleBins::index_type;
    using SoaData_type = WarpXParticleContainer::ParticleTileType::ParticleTileDataType;

public:
    /**
     * \brief Default constructor of the NuclearFusionFunc class.
     */
    NuclearFusionFunc () = default;

    /**
     * \brief Constructor of the NuclearFusionFunc class
     *
     * @param[in] collision_name the name of the collision
     * @param[in] mypc pointer to the MultiParticleContainer
     * @param[in] isSameSpecies whether the two colliding species are the same
     */
    NuclearFusionFunc (const std::string& collision_name, MultiParticleContainer const * const mypc,
                       const bool isSameSpecies):
        m_fusion_multiplier{amrex::ParticleReal{1.0}}, // default fusion multiplier
        m_probability_threshold{amrex::ParticleReal{0.02}}, // default fusion probability threshold
        m_probability_target_value{amrex::ParticleReal{0.002}}, // default fusion probability target_value
        m_fusion_type{BinaryCollisionUtils::get_nuclear_fusion_type(collision_name, mypc)},
        m_isSameSpecies{isSameSpecies}
    {

#ifdef AMREX_SINGLE_PRECISION_PARTICLES
        WARPX_ABORT_WITH_MESSAGE("Nuclear fusion module does not currently work with single precision");
#endif

        const amrex::ParmParse pp_collision_name(collision_name);
        utils::parser::queryWithParser(
            pp_collision_name, "event_multiplier", m_fusion_multiplier);
        utils::parser::queryWithParser(
            pp_collision_name, "probability_threshold", m_probability_threshold);
        utils::parser::queryWithParser(
            pp_collision_name, "probability_target_value",
            m_probability_target_value);

        bool create_products = true;
        utils::parser::queryWithParser(
            pp_collision_name, "create_products", create_products);
        utils::parser::queryWithParser(
            pp_collision_name, "save_particle_production", m_save_particle_production);

        m_exe.m_fusion_multiplier = m_fusion_multiplier;
        m_exe.m_probability_threshold = m_probability_threshold;
        m_exe.m_probability_target_value = m_probability_target_value;
        m_exe.m_fusion_type = m_fusion_type;
        m_exe.m_isSameSpecies = m_isSameSpecies;
        m_exe.m_create_products = create_products;

        if (m_save_particle_production) {
            m_particle_production_mf_name = collision_name + "_particle_production";
        }
    }

    void AllocData () override {
        if (m_save_particle_production) {
            WarpX & warpx = WarpX::GetInstance();

            int const level = 0;
            amrex::BoxArray const & ba = warpx.boxArray(level);
            amrex::DistributionMapping const & dmap = warpx.DistributionMap(level);
            int const ncomps = 1;
            const amrex::IntVect ng = amrex::IntVect::TheZeroVector();
            amrex::Real const initial_value = 0.;
            bool const remake = true;
            bool const redistribute_on_remake = true;
            warpx.m_fields.alloc_init(m_particle_production_mf_name, level, ba, dmap, ncomps, ng,
                                      initial_value, remake, redistribute_on_remake);
        }
    }

    struct Executor {
        /**
         * \brief Executor of the NuclearFusionFunc class. Performs nuclear fusions at the cell level
         * using the algorithm described in Higginson et al., Journal of Computational Physics 388,
         * 439-453 (2019). Note that this function does not yet create the product particles, but
         * instead fills an array p_mask that stores which collisions result in a fusion event.
         *
         * Also note that there are three main differences between this implementation and the
         * algorithm described in Higginson's paper:
         * - 1) The transformation from the lab frame to the center of mass frame is nonrelativistic
         * in Higginson's paper. Here, we implement a relativistic generalization.
         * - 2) The behaviour when the estimated fusion probability is greater than one is not
         * specified in Higginson's paper. Here, we provide an implementation using two runtime
         * dependent parameters (fusion probability threshold and fusion probability target value). See
         * documentation for more details.
         * - 3) Here, we divide the weight of a particle by the number of times it is paired with other
         * particles. This was not explicitly specified in Higginson's paper.
         *
         * @param[in] I1s,I2s is the start index for I1,I2 (inclusive).
         * @param[in] I1e,I2e is the stop index for I1,I2 (exclusive).
         * @param[in] I1,I2 index arrays. They determine all elements that will be used.
         * @param[in] soa_1,soa_2 contain the struct of array data of the two species
         * @param[in] m1,m2 are masses.
         * @param[in] dt is the time step length between two collision calls.
         * @param[in] dV is the volume of the corresponding cell.
         * @param[in] coll_idx is the collision index offset.
         * @param[in] cell_start_pair is the start index of the pairs in that cell.
         * @param[out] p_mask is a mask that will be set to true if a fusion event occurs for a given
         * pair. It is only needed here to store information that will be used later on when actually
         * creating the product particles.
         * @param[out] p_pair_indices_1,p_pair_indices_2 arrays that store the indices of the
         * particles of a given pair. They are only needed here to store information that will be used
         * later on when actually creating the product particles.
         * @param[out] p_pair_reaction_weight stores the weight of the product particles. It is only
         * needed here to store information that will be used later on when actually creating the
         * product particles.
         * @param[in] engine the random engine.
         */
        AMREX_GPU_HOST_DEVICE AMREX_INLINE
        void operator() (
            index_type const I1s, index_type const I1e,
            index_type const I2s, index_type const I2e,
            index_type const* AMREX_RESTRICT I1,
            index_type const* AMREX_RESTRICT I2,
            const SoaData_type& soa_1, const SoaData_type& soa_2,
            GetParticlePosition<PIdx> /*get_position_1*/, GetParticlePosition<PIdx> /*get_position_2*/,
            amrex::ParticleReal const /*n1*/, amrex::ParticleReal const /*n2*/,
            amrex::ParticleReal const /*T1*/, amrex::ParticleReal const /*T2*/,
            amrex::Real const /*global_lamdb*/,
            amrex::ParticleReal const  /*q1*/, amrex::ParticleReal const  /*q2*/,
            amrex::ParticleReal const  m1, amrex::ParticleReal const  m2,
            amrex::Real const  dt, amrex::Real const dV, index_type coll_idx,
            index_type const cell_start_pair, index_type* AMREX_RESTRICT p_mask,
            index_type* AMREX_RESTRICT p_pair_indices_1, index_type* AMREX_RESTRICT p_pair_indices_2,
            amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight,
            amrex::ParticleReal* /*p_product_data*/,
            int const i_cell,
            amrex::RandomEngine const& engine) const
        {
            amrex::ParticleReal * const AMREX_RESTRICT w1 = soa_1.m_rdata[PIdx::w];
            amrex::ParticleReal * const AMREX_RESTRICT u1x = soa_1.m_rdata[PIdx::ux];
            amrex::ParticleReal * const AMREX_RESTRICT u1y = soa_1.m_rdata[PIdx::uy];
            amrex::ParticleReal * const AMREX_RESTRICT u1z = soa_1.m_rdata[PIdx::uz];
            uint64_t* AMREX_RESTRICT idcpu1 = soa_1.m_idcpu;

            amrex::ParticleReal * const AMREX_RESTRICT w2 = soa_2.m_rdata[PIdx::w];
            amrex::ParticleReal * const AMREX_RESTRICT u2x = soa_2.m_rdata[PIdx::ux];
            amrex::ParticleReal * const AMREX_RESTRICT u2y = soa_2.m_rdata[PIdx::uy];
            amrex::ParticleReal * const AMREX_RESTRICT u2z = soa_2.m_rdata[PIdx::uz];
            uint64_t* AMREX_RESTRICT idcpu2 = soa_2.m_idcpu;

            // Number of macroparticles of each species
            const index_type NI1 = I1e - I1s;
            const index_type NI2 = I2e - I2s;
            const index_type max_N = amrex::max(NI1,NI2);
            const index_type min_N = amrex::min(NI1,NI2);

            index_type pair_index = cell_start_pair + coll_idx;

            // multiplier ratio to take into account unsampled pairs
            const auto multiplier_ratio = static_cast<int>(
                m_isSameSpecies ? min_N + max_N - 1 : min_N);

            index_type i1 = I1s + coll_idx;
            index_type i2 = I2s + coll_idx;
            // we will start from collision number = coll_idx and then add
            // stride (smaller set size) until we do all collisions (larger set size)
            for (index_type k = coll_idx; k < max_N; k += min_N)
            {

                // do not check for collision if a particle's weight was
                // reduced to zero from a previous collision
                if (idcpu1[ I1[i1] ]==amrex::ParticleIdCpus::Invalid ||
                    idcpu2[ I2[i2] ]==amrex::ParticleIdCpus::Invalid ) {
                    p_mask[pair_index] = false;
                }
                else {

                    SingleNuclearFusionEvent(
                        u1x[ I1[i1] ], u1y[ I1[i1] ], u1z[ I1[i1] ],
                        u2x[ I2[i2] ], u2y[ I2[i2] ], u2z[ I2[i2] ],
                        m1, m2, w1[ I1[i1] ], w2[ I2[i2] ],
                        dt, dV, static_cast<int>(pair_index), p_mask, p_pair_reaction_weight,
                        m_fusion_multiplier, multiplier_ratio,
                        m_probability_threshold,
                        m_probability_target_value,
                        m_fusion_type, engine,
                        m_create_products,
                        i_cell,
                        m_particle_production);

                    // Remove pair reaction weight from the colliding particles' weights
                    if (p_mask[pair_index]) {
                        BinaryCollisionUtils::remove_weight_from_colliding_particle(
                            w1[ I1[i1] ], idcpu1[ I1[i1] ], p_pair_reaction_weight[pair_index]);
                        BinaryCollisionUtils::remove_weight_from_colliding_particle(
                            w2[ I2[i2] ], idcpu2[ I2[i2] ], p_pair_reaction_weight[pair_index]);
                    }

                }

                p_pair_indices_1[pair_index] = I1[i1];
                p_pair_indices_2[pair_index] = I2[i2];
                if (max_N == NI1) { i1 += min_N; }
                if (max_N == NI2) { i2 += min_N; }
                pair_index += min_N;
            }
        }

        amrex::ParticleReal m_fusion_multiplier;
        amrex::ParticleReal m_probability_threshold;
        amrex::ParticleReal m_probability_target_value;
        NuclearFusionType m_fusion_type;
        bool m_computeSpeciesDensities = false;
        bool m_computeSpeciesTemperatures = false;
        bool m_need_product_data = false;
        bool m_isSameSpecies;

        bool m_create_products = true;
        amrex::Real * m_particle_production = nullptr;
    };

    [[nodiscard]] Executor const& executor (amrex::MFIter const& mfi) {
        if (m_save_particle_production) {
            WarpX & warpx = WarpX::GetInstance();
            int const level = 0;
            amrex::MultiFab * particle_production_mf = warpx.m_fields.get(m_particle_production_mf_name, level);
            m_exe.m_particle_production = particle_production_mf->array(mfi).dataPtr();
        }
        return m_exe;
    }

    bool use_global_debye_length() { return false; }

private:
    // Factor used to increase the number of fusion reaction by decreasing the weight of the
    // produced particles
    amrex::ParticleReal m_fusion_multiplier;
    // If the fusion multiplier is too high and results in a fusion probability that approaches
    // 1, there is a risk of underestimating the total fusion yield. In these cases, we reduce
    // the fusion multiplier used in a given collision. m_probability_threshold is the fusion
    // probability threshold above which we reduce the fusion multiplier.
    // m_probability_target_value is the target probability used to determine by how much
    // the fusion multiplier should be reduced.
    amrex::ParticleReal m_probability_threshold;
    amrex::ParticleReal m_probability_target_value;
    NuclearFusionType m_fusion_type;
    bool m_isSameSpecies;

    bool m_save_particle_production = false;
    std::string m_particle_production_mf_name;

    Executor m_exe;
};

#endif // WARPX_NUCLEAR_FUSION_FUNC_H_
