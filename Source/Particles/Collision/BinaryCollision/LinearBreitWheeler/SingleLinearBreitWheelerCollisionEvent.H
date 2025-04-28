/* Copyright 2023 Arianna Formenti
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef SINGLE_LINEAR_BREIT_WHEELER_COLLISION_EVENT_H_
#define SINGLE_LINEAR_BREIT_WHEELER_COLLISION_EVENT_H_

#include "LinearBreitWheelerCrossSection.H"

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>


/**
 * \brief This function computes whether the collision between two photons results in a
 * pair-producing (electron and positron) event, using the algorithm described in
 * Higginson et al., Journal of Computational Physics 388, 439-453 (2019).
 * If the electron-positron pair is produced, the mask is set to true for that given pair
 * of macrophotons and the weight of the produced particles is stored in p_pair_reaction_weight.
 *
 * Note that the word <<pair>> sometimes refers to the electron-positron products of the collision,
 * sometimes to the macrophotons that are paired to collide in a cell according to the algorithm.
 * It should be clear from the context.
 *
 * @tparam index_type type of the index argument
 * @param[in] u1x,u1y,u1z momentum components of the first colliding particle
 * @param[in] u2x,u2y,u2z momentum components of the second colliding particle
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
void SingleLinearBreitWheelerCollisionEvent (const amrex::ParticleReal& u1x, const amrex::ParticleReal& u1y,
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
    // General notations in this function:
    //     x_sq denotes the square of x
    //     x_star denotes the value of x in the center of mass frame

    using namespace amrex::literals;

    const amrex::ParticleReal w_min = amrex::min(w1, w2);
    const amrex::ParticleReal w_max = amrex::max(w1, w2);

    constexpr auto one_pr = amrex::ParticleReal(1.);
    constexpr auto two_pr = amrex::ParticleReal(2.);
    constexpr auto one_half_pr = amrex::ParticleReal(1./2.);
    constexpr amrex::ParticleReal c = PhysConst::c;
    constexpr amrex::ParticleReal c_sq = c*c;
    constexpr amrex::ParticleReal me = PhysConst::m_e;
    auto constexpr pow2 = [](double const x) { return x*x; };

    // Compute momenta in the lab frame
    const amrex::ParticleReal p1x = u1x * me;
    const amrex::ParticleReal p1y = u1y * me;
    const amrex::ParticleReal p1z = u1z * me;
    const amrex::ParticleReal p2x = u2x * me;
    const amrex::ParticleReal p2y = u2y * me;
    const amrex::ParticleReal p2z = u2z * me;
    const amrex::ParticleReal p1 = std::sqrt(pow2(p1x)+pow2(p1y)+pow2(p1z));
    const amrex::ParticleReal p2 = std::sqrt(pow2(p2x)+pow2(p2y)+pow2(p2z));

    // Compute Lorentz factor gamma in the lab frame
    const amrex::ParticleReal g1 = p1 / ( me * c );
    const amrex::ParticleReal g2 = p2 / ( me * c );

    // Compute the cosine of the angle between the two photon momenta in the lab frame
    const amrex::ParticleReal cos_ang = (p1x*p2x+p1y*p2y+p1z*p2z)/(p1*p2);

    // Energy squared of each and every particle involved in the collision
    // (both colliding photons and the produced electron and positron)
    // in the center of momentum frame (they are all equal),
    // calculated using the Lorentz invariance of the total four-momentum norm
    const amrex::ParticleReal E_star_sq = one_half_pr*c_sq*p1*p2*(one_pr - cos_ang);

    // Kinetic energy of each particle in the center of momentum frame
    const amrex::ParticleReal E_star = std::sqrt(E_star_sq);

    // Compute linear Breit Wheeler cross section as a function of the kinetic energy
    // in the center of momentum frame
    auto lbw_cross_section = amrex::ParticleReal(0.);
    lbw_cross_section = LinearBreitWheelerCrossSection(E_star);

    // Lorentz factors in the center of momentum frame
    const amrex::ParticleReal g1_star = E_star / (me*c_sq);
    const amrex::ParticleReal g2_star = g1_star;

    // relative velocity in the center of momentum frame
    const amrex::ParticleReal v_rel = two_pr*c;

    // Cross section and relative velocity are computed in the center of momentum frame.
    // On the other hand, the particle densities (weight over volume) in the lab frame are used. To
    // take into account this discrepancy, we need to multiply the collision probability by the ratio
    // between the Lorentz factors in the COM frame and the Lorentz factors in the lab frame
    // (see Perez et al., Phys.Plasmas.19.083104 (2012))
    const amrex::ParticleReal lab_to_COM_factor = g1_star*g2_star/(g1*g2);

    // First estimate of probability to produce an electron-positron pair
    amrex::ParticleReal probability_estimate = multiplier_ratio * event_multiplier *
                                lab_to_COM_factor * w_max * lbw_cross_section * v_rel * dt / dV;

    // Effective event multiplier
    amrex::ParticleReal event_multiplier_eff = event_multiplier;

    // If the pair-production probability is too high and the event multiplier greater than one,
    // we risk to systematically underestimate the electron-positron yield. In this case,
    // we reduce the event multiplier to reduce the pair-production probability
    if (probability_estimate > probability_threshold)
    {
        // We aim for a pair-production probability of probability_target_value but take into account
        // the constraint that the event_multiplier cannot be smaller than one
        event_multiplier_eff  = amrex::max(event_multiplier *
                                         probability_target_value / probability_estimate , one_pr);
        probability_estimate *= event_multiplier_eff/event_multiplier;
    }

    // Compute actual pair-production probability that is always between zero and one
    // In principle this is obtained by computing 1 - exp(-probability_estimate)
    // However, the computation of this quantity can fail numerically when probability_estimate is
    // too small (e.g. exp(-probability_estimate) returns 1 and the computation returns 0).
    // In this case, we simply use "probability_estimate" instead of 1 - exp(-probability_estimate)
    // The threshold exp_threshold at which we switch between the two formulas is determined by the
    // fact that computing the exponential is only useful if it can resolve the x^2/2 term of its
    // Taylor expansion, i.e. the square of probability_estimate should be greater than the
    // machine epsilon.
    const amrex::ParticleReal probability = -std::expm1(-probability_estimate);

    // Get a random number
    const amrex::ParticleReal random_number = amrex::Random(engine);

    // If we have a pair-production event, set the mask to true and fill the product weight array
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

#endif // SINGLE_LINEAR_BREIT_WHEELER_COLLISION_EVENT_H_
