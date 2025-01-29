/* Copyright 2021 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_SINGLE_NUCLEAR_FUSION_EVENT_H_
#define WARPX_SINGLE_NUCLEAR_FUSION_EVENT_H_

#include "BoschHaleFusionCrossSection.H"
#include "ProtonBoronFusionCrossSection.H"

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>


/**
 * \brief This function computes whether the collision between two particles result in a
 * nuclear fusion event, using the algorithm described in Higginson et al., Journal of
 * Computational Physics 388, 439-453 (2019). If nuclear fusion occurs, the mask is set to true
 * for that given pair of particles and the weight of the produced particles is stored in
 * p_pair_reaction_weight.
 *
 * @tparam index_type type of the index argument
 * @param[in] u1x,u1y,u1z momenta of the first colliding particle
 * @param[in] u2x,u2y,u2z momenta of the second colliding particle
 * @param[in] m1,m2 masses
 * @param[in] w1,w2 effective weight of the colliding particles
 * @param[in] dt is the time step length between two collision calls.
 * @param[in] dV is the volume of the corresponding cell.
 * @param[in] pair_index is the index of the colliding pair
 * @param[out] p_mask is a mask that will be set to true if fusion occurs for that pair
 * @param[out] p_pair_reaction_weight stores the weight of the product particles
 * @param[in] fusion_multiplier factor used to increase the number of fusion events by
 * decreasing the weight of the produced particles
 * @param[in] multiplier_ratio factor used to take into account unsampled pairs (i.e. the fact
 * that a particle only collides with one or few particles of the other species)
 * @param[in] probability_threshold probability threshold above which we decrease the fusion
 * multiplier
 * @param[in] probability_target_value if the probability threshold is exceeded, this is used
 * to determine by how much the fusion multiplier is reduced
 * @param[in] fusion_type the physical fusion process to model
 * @param[in] engine the random engine.
 */
template <typename index_type>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void SingleNuclearFusionEvent (const amrex::ParticleReal& u1x, const amrex::ParticleReal& u1y,
                               const amrex::ParticleReal& u1z, const amrex::ParticleReal& u2x,
                               const amrex::ParticleReal& u2y, const amrex::ParticleReal& u2z,
                               const amrex::ParticleReal& m1, const amrex::ParticleReal& m2,
                               amrex::ParticleReal w1, amrex::ParticleReal w2,
                               const amrex::Real& dt, const amrex::ParticleReal& dV, const int& pair_index,
                               index_type* AMREX_RESTRICT p_mask,
                               amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight,
                               const amrex::ParticleReal& fusion_multiplier,
                               const int& multiplier_ratio,
                               const amrex::ParticleReal& probability_threshold,
                               const amrex::ParticleReal& probability_target_value,
                               const NuclearFusionType& fusion_type,
                               const amrex::RandomEngine& engine)
{
    amrex::ParticleReal E_coll, v_coll, lab_to_COM_factor;

    BinaryCollisionUtils::get_collision_parameters(
        u1x, u1y, u1z, u2x, u2y, u2z, m1, m2,
        E_coll, v_coll, lab_to_COM_factor);

    using namespace amrex::literals;

    const amrex::ParticleReal w_min = amrex::min(w1, w2);
    const amrex::ParticleReal w_max = amrex::max(w1, w2);

    // Compute fusion cross section as a function of kinetic energy in the center of mass frame
    auto fusion_cross_section = amrex::ParticleReal(0.);
    if (fusion_type == NuclearFusionType::ProtonBoronToAlphas)
    {
        fusion_cross_section = ProtonBoronFusionCrossSection(E_coll);
    }
    else if ((fusion_type == NuclearFusionType::DeuteriumTritiumToNeutronHelium)
          || (fusion_type == NuclearFusionType::DeuteriumDeuteriumToProtonTritium)
          || (fusion_type == NuclearFusionType::DeuteriumDeuteriumToNeutronHelium))
    {
        fusion_cross_section = BoschHaleFusionCrossSection(E_coll, fusion_type, m1, m2);
    }

    // First estimate of probability to have fusion reaction
    amrex::ParticleReal probability_estimate = multiplier_ratio * fusion_multiplier *
                                lab_to_COM_factor * w_max * fusion_cross_section * v_coll * dt / dV;

    // Effective fusion multiplier
    amrex::ParticleReal fusion_multiplier_eff = fusion_multiplier;

    // If the fusion probability is too high and the fusion multiplier greater than one, we risk to
    // systematically underestimate the fusion yield. In this case, we reduce the fusion multiplier
    // to reduce the fusion probability
    if (probability_estimate > probability_threshold)
    {
        // We aim for a fusion probability of probability_target_value but take into account
        // the constraint that the fusion_multiplier cannot be smaller than one
        fusion_multiplier_eff  = amrex::max(fusion_multiplier *
                                         probability_target_value / probability_estimate , 1._prt);
        probability_estimate *= fusion_multiplier_eff/fusion_multiplier;
    }

    // Compute actual fusion probability that is always between zero and one
    // In principle this is obtained by computing 1 - exp(-probability_estimate)
    // However, the computation of this quantity can fail numerically when probability_estimate is
    // too small (e.g. exp(-probability_estimate) returns 1 and the computation returns 0).
    // std::expm1 is used since it maintains correctness for small exponent.
    const amrex::ParticleReal probability = -std::expm1(-probability_estimate);

    // Get a random number
    const amrex::ParticleReal random_number = amrex::Random(engine);

    // If we have a fusion event, set the mask the true and fill the product weight array
    if (random_number < probability)
    {
        p_mask[pair_index] = true;
        p_pair_reaction_weight[pair_index] = w_min/fusion_multiplier_eff;
    }
    else
    {
        p_mask[pair_index] = false;
    }

}
#endif // WARPX_SINGLE_NUCLEAR_FUSION_EVENT_H_
