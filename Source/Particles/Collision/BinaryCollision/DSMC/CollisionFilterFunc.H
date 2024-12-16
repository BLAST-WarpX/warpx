/* Copyright 2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies), Neil Zaim
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_COLLISION_FILTER_FUNC_H_
#define WARPX_COLLISION_FILTER_FUNC_H_

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/Collision/ScatteringProcess.H"

#include <AMReX_Random.H>

/**
 * \brief This function determines whether a collision occurs for a given
 * pair of particles.
 *
 * @param[in] u1x,u1y,u1z momenta of the first colliding particle.
 * @param[in] u2x,u2y,u2z momenta of the second colliding particle.
 * @param[in] m1,m2 masses.
 * @param[in] w1,w2 effective weight of the colliding particles.
 * @param[in] dt is the time step length between two collision calls.
 * @param[in] dV is the volume of the corresponding cell.
 * @param[in] pair_index is the index of the colliding pair.
 * @param[out] p_mask is a mask that will be set to a non-zero integer if a
 *             collision occurs. The integer encodes the scattering process.
 * @param[out] p_pair_reaction_weight stores the weight of the product particles.
 * @param[in] multiplier factor by which the collision probability is increased to
 *            account for all other possible binary collision partners.
 * @param[in] process_count number of scattering processes to consider.
 * @param[in] scattering processes an array of scattering processes included for consideration.
 * @param[in] engine the random engine.
 */
template <int max_process_count, typename index_type>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void CollisionPairFilter (const amrex::ParticleReal u1x, const amrex::ParticleReal u1y,
                          const amrex::ParticleReal u1z, const amrex::ParticleReal u2x,
                          const amrex::ParticleReal u2y, const amrex::ParticleReal u2z,
                          const amrex::ParticleReal m1, const amrex::ParticleReal m2,
                          const amrex::ParticleReal w1, const amrex::ParticleReal w2,
                          const amrex::Real dt, const amrex::ParticleReal dV, const int pair_index,
                          index_type* AMREX_RESTRICT p_mask,
                          amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight,
                          const int multiplier,
                          const int process_count,
                          const ScatteringProcess::Executor* scattering_processes,
                          const amrex::RandomEngine& engine)
{
    amrex::ParticleReal E_coll, v_coll, lab_to_COM_factor;

    BinaryCollisionUtils::get_collision_parameters(
        u1x, u1y, u1z, u2x, u2y, u2z, m1, m2,
        E_coll, v_coll, lab_to_COM_factor);

    using namespace amrex::literals;

    const amrex::ParticleReal w_min = amrex::min(w1, w2);
    const amrex::ParticleReal w_max = amrex::max(w1, w2);

    // convert E_coll from Joule to eV
    E_coll /= PhysConst::q_e;

    // Evaluate the cross-section for each scattering process to determine
    // the total collision probability.

    // The size of the arrays below is a compile-time constant (template parameter)
    // for performance reasons: it avoids dynamic memory allocation on the GPU.
    int coll_type[max_process_count] = {0};
    amrex::ParticleReal sigma_sums[max_process_count] = {0._prt};
    for (int ii = 0; ii < process_count; ii++) {
        auto const& scattering_process = scattering_processes[ii];
        coll_type[ii] = int(scattering_process.m_type);
        const amrex::ParticleReal sigma = scattering_process.getCrossSection(E_coll);
        sigma_sums[ii] = sigma + ((ii == 0) ? 0._prt : sigma_sums[ii-1]);
    }
    const auto sigma_tot = sigma_sums[process_count-1];

    // calculate total collision probability
    const amrex::ParticleReal exponent = (
        lab_to_COM_factor * multiplier * w_max * sigma_tot * v_coll * dt / dV
    );

    // Compute actual collision probability that is always between zero and one
    // In principle this is obtained by computing 1 - exp(-probability_estimate)
    // However, the computation of this quantity can fail numerically when probability_estimate is
    // too small (e.g. exp(-probability_estimate) returns 1 and the computation returns 0).
    // std::expm1 is used since it maintains correctness for small exponent.
    const amrex::ParticleReal probability = -std::expm1(-exponent);

    // Now we determine if a collision should occur
    if (amrex::Random(engine) < probability)
    {
        const amrex::ParticleReal random_number = amrex::Random(engine);
        for (int ii = 0; ii < process_count; ii++) {
            if (random_number <= sigma_sums[ii] / sigma_tot)
            {
                p_mask[pair_index] = coll_type[ii];
                p_pair_reaction_weight[pair_index] = w_min;
                break;
            }
        }
    }
    else
    {
        p_mask[pair_index] = false;
    }
}

#endif // WARPX_COLLISION_FILTER_FUNC_H_
