/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DSMC_FUNC_H_
#define WARPX_DSMC_FUNC_H_

#include "CollisionFilterFunc.H"

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/Collision/BinaryCollision/ShuffleFisherYates.H"
#include "Particles/Collision/CollisionBase.H"
#include "Particles/Collision/ScatteringProcess.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/ParticleCreation/SmartCopy.H"
#include "Particles/ParticleCreation/SmartUtils.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <AMReX_DenseBins.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>

/**
 * \brief This class performs DSMC (direct simulation Monte Carlo) collisions
 * within a cell. Particles are paired up and for each pair a stochastic process
 * determines whether a collision occurs. The algorithm is similar to the one
 * used for binary Coulomb collisions and the nuclear fusion module.
 */
class DSMCFunc
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
     * \brief Constructor of the DSMCFunc class.
     */
    DSMCFunc () = default;

    /**
     * \brief Constructor of the DSMCFunc class
     *
     * @param[in] collision_name the name of the collision
     * @param[in] mypc pointer to the MultiParticleContainer
     * @param[in] isSameSpecies whether the two colliding species are the same
     */
    DSMCFunc ( const std::string& collision_name,
               MultiParticleContainer const * mypc,
               bool isSameSpecies );

    struct Executor {
        /**
         * \brief Executor of the DSMCFunc class. Performs DSMC collisions at the cell level.
         * Note that this function does not yet create the product particles, but
         * instead fills an array p_mask that stores which pairs result in a collision event.
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
            amrex::ParticleReal const  /*q1*/, amrex::ParticleReal const  /*q2*/,
            amrex::ParticleReal const  m1, amrex::ParticleReal const  m2,
            amrex::Real const  dt, amrex::Real const dV, index_type coll_idx,
            index_type const cell_start_pair, index_type* AMREX_RESTRICT p_mask,
            index_type* AMREX_RESTRICT p_pair_indices_1, index_type* AMREX_RESTRICT p_pair_indices_2,
            amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight,
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

#if (defined WARPX_DIM_RZ)
            amrex::ParticleReal * const AMREX_RESTRICT theta1 = soa_1.m_rdata[PIdx::theta];
            amrex::ParticleReal * const AMREX_RESTRICT theta2 = soa_2.m_rdata[PIdx::theta];
#endif
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
                } else {

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
                    amrex::ParticleReal const theta = theta2[I2[i2]]-theta1[I1[i1]];
                    amrex::ParticleReal const u1xbuf = u1x[I1[i1]];
                    u1x[I1[i1]] = u1xbuf*std::cos(theta) - u1y[I1[i1]]*std::sin(theta);
                    u1y[I1[i1]] = u1xbuf*std::sin(theta) + u1y[I1[i1]]*std::cos(theta);
#endif

                    const int max_process_count = 4; // Pre-defined value, for performance reasons
                    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                        (m_process_count < max_process_count), "Too many scattering processes in DSMC routine (hardcoded to only allow 4). Update the max_process_count value in source code to allow more scattering processes."
                    );
                    CollisionPairFilter<max_process_count>(
                        u1x[ I1[i1] ], u1y[ I1[i1] ], u1z[ I1[i1] ],
                        u2x[ I2[i2] ], u2y[ I2[i2] ], u2z[ I2[i2] ],
                        m1, m2, w1[ I1[i1] ], w2[ I2[i2] ],
                        dt, dV, static_cast<int>(pair_index), p_mask,
                        p_pair_reaction_weight, multiplier_ratio,
                        m_process_count, m_scattering_processes_data, engine);

#if (defined WARPX_DIM_RZ)
                    amrex::ParticleReal const u1xbuf_new = u1x[I1[i1]];
                    u1x[I1[i1]] = u1xbuf_new*std::cos(-theta) - u1y[I1[i1]]*std::sin(-theta);
                    u1y[I1[i1]] = u1xbuf_new*std::sin(-theta) + u1y[I1[i1]]*std::cos(-theta);
#endif

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

        int m_process_count;
        bool m_computeSpeciesDensities = false;
        bool m_computeSpeciesTemperatures = false;
        bool m_isSameSpecies = false;
        ScatteringProcess::Executor* m_scattering_processes_data;
    };

    [[nodiscard]] Executor const& executor () const { return m_exe; }

private:
    amrex::Vector<ScatteringProcess> m_scattering_processes;
    amrex::Gpu::DeviceVector<ScatteringProcess::Executor> m_scattering_processes_exe;
    bool m_isSameSpecies;

    Executor m_exe;
};

#endif // WARPX_DSMC_FUNC_H_
