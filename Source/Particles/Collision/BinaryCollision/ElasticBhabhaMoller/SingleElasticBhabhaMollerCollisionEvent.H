/* Copyright 2023 Arianna Formenti, Jinseo Park
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef SINGLE_ELASTIC_BHABHA_COLLISION_EVENT_H_
#define SINGLE_ELASTIC_BHABHA_COLLISION_EVENT_H_

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
void SingleElasticBhabhaCollisionEvent (const amrex::ParticleReal& u1x, const amrex::ParticleReal& u1y,
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

    // Compute Lorentz factor gamma
    amrex::ParticleReal const gb1sq = (u1x*u1x + u1y*u1y + u1z*u1z)*inv_c_sq;
    amrex::ParticleReal const gb2sq = (u2x*u2x + u2y*u2y + u2z*u2z)*inv_c_sq;
    amrex::ParticleReal const g1 = std::sqrt( 1._prt + gb1sq );
    amrex::ParticleReal const g2 = std::sqrt( 1._prt + gb2sq );

    amrex::ParticleReal const diffx = u1x-u2x;
    amrex::ParticleReal const diffy = u1y-u2y;
    amrex::ParticleReal const diffz = u1z-u2z;
    amrex::ParticleReal const diffm = std::sqrt((diffx*diffx+diffy*diffy+diffz*diffz)*inv_c_sq);
    amrex::ParticleReal const summm = std::sqrt(gb1sq) + std::sqrt(gb2sq);
    // If g = u1 - u2 = 0, do not collide.
    // Or if the relative difference is less than 1.0e-10.
    if ( diffm < std::numeric_limits<amrex::ParticleReal>::min() || diffm/summm < 1.0e-10 ) { return; }

    amrex::ParticleReal const m1 = PhysConst::m_e;
    amrex::ParticleReal const m2 = PhysConst::m_e;
    // Compute momenta
    amrex::ParticleReal const p1x = u1x * m1;
    amrex::ParticleReal const p1y = u1y * m1;
    amrex::ParticleReal const p1z = u1z * m1;
    amrex::ParticleReal const p2x = u2x * m2;
    amrex::ParticleReal const p2y = u2y * m2;
    amrex::ParticleReal const p2z = u2z * m2;

    // Compute center-of-momentum (COM) velocity and gamma
    amrex::ParticleReal const mass_g = m1 * g1 + m2 * g2;
    amrex::ParticleReal const vcx    = (p1x+p2x) / mass_g;
    amrex::ParticleReal const vcy    = (p1y+p2y) / mass_g;
    amrex::ParticleReal const vcz    = (p1z+p2z) / mass_g;
    amrex::ParticleReal const vcms   = vcx*vcx + vcy*vcy + vcz*vcz;
    amrex::ParticleReal const gc     = 1._prt / std::sqrt( 1._prt - vcms*inv_c_sq );
    // Factor that converts collision rate from rest frame to lab frame: ration of energies in rest frame and lab frame
    // (See L. D. Landau and E. M. Lifshitz. The classical theory of fields, Eq. 12.5, 12.6)
    const amrex::ParticleReal lab_to_rest_frame_factor = gc
    // sqrt( | v1 - v2 |^2  - | v1 x v2 |^2/c^2 ) evaluated in the rest frame
    const amrex::ParticleReal v_rel =  PhysConst::c;

    // Elastic Bhabha Tree level cross section.
    amrex::ParticleReal numerator = (
        111._prt * std::cos(theta_m) +
        6._prt * std::cos(2._prt * theta_m) +
        std::cos(3 * theta_m) +
        96._prt / (sin(theta_m / 2._prt)*sin(theta_m / 2._prt)) +
        384._prt * std::log(std::sin(theta_m / 2._prt)) +
        10._prt
    );
    amrex::ParticleReal denominator = 1536._prt * MathConst::pi;
    const amrex::ParticleReal bhabha_cross_section = (16*MathConst::pi*MathConst::pi* PhysConst::r_e* PhysConst::r_e)*(p1sm / m1 / PhysConst::c)*(p1sm / m1 / PhysConst::c)* (numerator / denominator);
    // First estimate of probability for scattering
    // (Convert from rest frame to lab frame, using lab_to_rest_frame_factor)
    amrex::ParticleReal probability_estimate = multiplier_ratio * event_multiplier *
                                lab_to_rest_frame_factor * w_max * bhabha_cross_section * v_rel * dt / dV;

    // Effective event multiplier
    amrex::ParticleReal event_multiplier_eff = event_multiplier;

    // If the scattering probability is too high and the event multiplier greater than one,
    // we risk to systematically underestimate the electron-positron yield. In this case,
    // we reduce the event multiplier to reduce the pair-production probability
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

#endif // SINGLE_ELASTIC_BHABHA_COLLISION_EVENT_H_
