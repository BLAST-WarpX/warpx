/* Copyright 2021 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_PAIRWISE_COULOMB_COLLISION_FUNC_H_
#define WARPX_PAIRWISE_COULOMB_COLLISION_FUNC_H_

#include "ElasticCollisionPerez.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"

#include <AMReX_DenseBins.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>


/**
 * \brief This functor performs pairwise Coulomb collision on a single cell by calling the function
 *        ElasticCollisionPerez. It also reads and contains the Coulomb logarithm.
 */
class PairWiseCoulombCollisionFunc
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
     * \brief Default constructor of the PairWiseCoulombCollisionFunc class.
     */
    PairWiseCoulombCollisionFunc () = default;

    /**
     * \brief Constructor of the PairWiseCoulombCollisionFunc class
     *
     * @param[in] collision_name the name of the collision
     * @param[in] mypc the particle container (unused)
     * @param[in] isSameSpecies true if this is an intra-species colission
     */
    PairWiseCoulombCollisionFunc (const std::string& collision_name,
                                  [[maybe_unused]] MultiParticleContainer const * const mypc,
                                  const bool isSameSpecies):
        m_isSameSpecies{isSameSpecies}
    {
        using namespace amrex::literals;
        const amrex::ParmParse pp_collision_name(collision_name);
        // default Coulomb log, if < 0, will be computed automatically
        amrex::ParticleReal CoulombLog = -1.0_prt;
        utils::parser::queryWithParser(
            pp_collision_name, "CoulombLog", CoulombLog);
        m_CoulombLog = CoulombLog;

        m_exe.m_CoulombLog = m_CoulombLog;
        if (m_CoulombLog<0.0) { m_exe.m_computeSpeciesTemperatures = true; }
        m_exe.m_isSameSpecies = m_isSameSpecies;
    }

    struct Executor {
        /**
         * \brief Executor of the PairWiseCoulombCollisionFunc class. Performs Coulomb collisions
         * at the cell level by calling ElasticCollisionPerez.
         *
         * @param[in] I1s,I2s is the start index for I1,I2 (inclusive).
         * @param[in] I1e,I2e is the stop index for I1,I2 (exclusive).
         * @param[in] I1,I2 index arrays. They determine all elements that will be used.
         * @param[in,out] soa_1,soa_2 contain the struct of array data of the two species.
         * @param[in] n1,n2 are local densities.
         * @param[in] T1,T2 are local temperatures.
         * @param[in] q1,q2 are charges.
         * @param[in] m1,m2 are masses.
         * @param[in] dt is the time step length between two collision calls.
         * @param[in] dV is the volume of the corresponding cell.
         * @param[in] coll_idx is the collision index offset.
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
            amrex::ParticleReal const  n1, amrex::ParticleReal const  n2,
            amrex::ParticleReal const  T1, amrex::ParticleReal const  T2,
            amrex::ParticleReal const  q1, amrex::ParticleReal const  q2,
            amrex::ParticleReal const  m1, amrex::ParticleReal const  m2,
            amrex::Real const  dt, amrex::Real const dV, index_type coll_idx,
            index_type const /*cell_start_pair*/, index_type* /*p_mask*/,
            index_type* /*p_pair_indices_1*/, index_type* /*p_pair_indices_2*/,
            amrex::ParticleReal* /*p_pair_reaction_weight*/,
            amrex::RandomEngine const& engine) const
        {
            using namespace amrex::literals;

            ElasticCollisionPerez(
                    I1s, I1e, I2s, I2e, I1, I2,
                    soa_1, soa_2, n1, n2, T1, T2,
                    q1, q2, m1, m2,
                    dt, m_CoulombLog, dV, engine, m_isSameSpecies, coll_idx);
        }

        amrex::ParticleReal m_CoulombLog;
        bool m_computeSpeciesDensities = true;
        bool m_computeSpeciesTemperatures = false;
        bool m_isSameSpecies;
    };

    [[nodiscard]] Executor const& executor () const { return m_exe; }

private:
    amrex::ParticleReal m_CoulombLog;
    bool m_isSameSpecies;

    Executor m_exe;
};

#endif // WARPX_PAIRWISE_COULOMB_COLLISION_FUNC_H_
