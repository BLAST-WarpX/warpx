/* Copyright 2023 Arianna Formenti
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef SINGLE_LINEAR_COMPTON_COLLISION_EVENT_H_
#define SINGLE_LINEAR_COMPTON_COLLISION_EVENT_H_

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>


/**
 * \brief This function computes whether the collision between a photon and lepton results in a
 * scattering event, using the algorithm described in Higginson et al., Journal of Computational
 * Physics 388, 439-453 (2019). If the scattering occurs, the mask is set to true for that given pair
 * of colliding macroparticles and the weight of the produced particles is stored in p_pair_reaction_weight.
 *
 * @tparam index_type type of the index argument
 * @param[in] u1x,u1y,u1z momentum components of the first colliding particle (photon)
 * @param[in] u2x,u2y,u2z momentum components of the second colliding particle (lepton)
 * @param[in] w1,w2 effective weight of the colliding particles
 * @param[in] dt is the time step length between two collision calls.
 * @param[in] dV is the volume of the corresponding cell.
 * @param[in] pair_index is the index of the colliding pair
 * @param[out] p_mask is a mask that will be set to true if pair (electron-positron) production
 * occurs for that pair of macrophotons
 * @param[out] p_pair_reaction_weight stores the weight of the product particles
 * @param[in] event_multiplier factor used to increase the number of pair-production events by
 * decreasing the weight of the produced particles
 * @param[in] multiplier_ratio factor used to take into account unsampled pairs (i.e. the fact
 * that a particle only collides with one or few particles of the other species)
 * @param[in] probability_threshold probability threshold above which we decrease the event
 * multiplier
 * @param[in] probability_target_value if the probability threshold is exceeded, this is used
 * to determine by how much the event multiplier is reduced
 * @param[in] engine the random engine.
 */
template <typename index_type>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void SingleLinearComptonCollisionEvent (const amrex::ParticleReal& u1x, const amrex::ParticleReal& u1y,
                                       const amrex::ParticleReal& u1z, const amrex::ParticleReal& u2x,
                                       const amrex::ParticleReal& u2y, const amrex::ParticleReal& u2z,
                                       amrex::ParticleReal w1, amrex::ParticleReal w2,
                                       const amrex::Real& dt, const amrex::ParticleReal& dV, const int& pair_index,
                                       index_type* AMREX_RESTRICT p_mask,
                                       amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight,
                                       const amrex::ParticleReal& event_multiplier,
                                       const int& multiplier_ratio,
                                       const amrex::ParticleReal& probability_threshold,
                                       const amrex::ParticleReal& probability_target_value,
                                       const amrex::RandomEngine& engine)
{
    using namespace amrex::literals;
    constexpr amrex::ParticleReal inv_c = 1._prt / PhysConst::c;
    constexpr amrex::ParticleReal inv_c_sq = inv_c * inv_c;

    const amrex::ParticleReal w_min = amrex::min(w1, w2);
    const amrex::ParticleReal w_max = amrex::max(w1, w2);

    // When checking the user input, we ensure that the first particle is the photon
    // and the second particle is the lepton (electron/positron).
    // The cross-section is computed in the rest frame of the lepton.

    // Lepton's Lorentz factor
    const amrex::ParticleReal gamma_lepton = std::sqrt(1._prt + (u2x*u2x + u2y*u2y + u2z*u2z)*inv_c_sq);
    // Photon's normalized momentum in the lab frame
    amrex::ParticleReal const u1_norm = std::sqrt(u1x*u1x + u1y*u1y + u1z*u1z);
    // Photon's normalized momentum in the rest frame of the lepton
    // (obtained from the fact that the product of the photon and lepton's 4-momenta is invariant)
    const amrex::ParticleReal u1_norm_rest = u1_norm * gamma_lepton - (u1x*u2x + u1y*u2y + u1z*u2z)*inv_c;
    // Factor that converts collision rate from rest frame to lab frame: ration of energies in rest frame and lab frame
    // (See L. D. Landau and E. M. Lifshitz. The classical theory of fields, Eq. 12.5, 12.6)
    const amrex::ParticleReal lab_to_rest_frame_factor = u1_norm_rest * 1._prt / (u1_norm * gamma_lepton);
    // sqrt( | v1 - v2 |^2  - | v1 x v2 |^2/c^2 ) evaluated in the rest frame
    const amrex::ParticleReal v_rel =  PhysConst::c;


    // Calculate the Klein-Nishina cross-section (cross-section in the rest frame of the lepton)
    const amrex::ParticleReal k = u1_norm_rest * inv_c;
    // By convention, in WarpX, the photon momentum u is normalized by the electron mass m_e, so k corresponds to p/(m_e*c)
    const amrex::ParticleReal f1 = 2._prt * ( 2._prt + k*(1._prt+k)*(8._prt+k) ) / ( k*k * (1._prt + 2._prt*k)*(1._prt + 2._prt*k) );
    const amrex::ParticleReal f2 = ( 2._prt + k*(2._prt-k) ) * std::log( 1._prt + 2._prt*k ) / (k*k*k);
    const amrex::ParticleReal klein_nishina_cross_section = MathConst::pi * PhysConst::r_e * PhysConst::r_e * ( f1 - f2 );

    // First estimate of probability for scattering
    // (Convert from rest frame to lab frame, using lab_to_rest_frame_factor)
    amrex::ParticleReal probability_estimate = multiplier_ratio * event_multiplier *
                                lab_to_rest_frame_factor * w_max * klein_nishina_cross_section * v_rel * dt / dV;

    // Effective event multiplier
    amrex::ParticleReal event_multiplier_eff = event_multiplier;

    // If the scattering probability is too high and the event multiplier greater than one,
    // we risk to systematically underestimate the Compton scattering yield. In this case,
    // we reduce the event multiplier to reduce the scattering probability
    if (probability_estimate > probability_threshold)
    {
        // We aim for a scattering probability of probability_target_value but take into account
        // the constraint that the event_multiplier cannot be smaller than one
        event_multiplier_eff  = amrex::max(event_multiplier *
                                         probability_target_value / probability_estimate , 1._prt);
        probability_estimate *= event_multiplier_eff/event_multiplier;
    }

    // Compute actual scattering probability that is always between zero and one
    // In principle this is obtained by computing 1 - exp(-probability_estimate)
    // However, the computation of this quantity can fail numerically when probability_estimate is
    // too small (e.g. exp(-probability_estimate) returns 1 and the computation returns 0).
    // In this case, we simply use "probability_estimate" instead of 1 - exp(-probability_estimate)
    const amrex::ParticleReal probability = -std::expm1(-probability_estimate);

    // Get a random number
    const amrex::ParticleReal random_number = amrex::Random(engine);

    // If we have a scattering event, set the mask to true and fill the product weight array
    if (random_number < probability)
    {
        p_mask[pair_index] = true;
        p_pair_reaction_weight[pair_index] = w_min/event_multiplier_eff;
    }
    else
    {
        p_mask[pair_index] = false;
    }
}

#endif // SINGLE_LINEAR_COMPTON_COLLISION_EVENT_H_
