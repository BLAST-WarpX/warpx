/* Copyright 2026 The WarpX Community
 * Authors: Prabhat Kumar, Eric Clark (Helion Energy)
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_HYBRIDRESISTIVEDRAG_H_
#define WARPX_PARTICLES_COLLISION_HYBRIDRESISTIVEDRAG_H_

#include "Particles/MultiParticleContainer.H"
#include "Particles/Collision/CollisionBase.H"

#include <AMReX_REAL.H>

#include <string>

/** Hybrid-PIC electron-ion drag operator (drag only, no heat deposit).
 *
 * Acts on one charged ion species at the eta-derived rate
 *     nu_{s,e} = Z_s e^2 eta_s_eff n_e / m_s,
 *     eta_s_eff = eta_global(rho,|J|,t) + eta_s_per (per-species overlay, if any).
 *
 * The cell bulk velocity relaxes toward the electron fluid V_e,
 *     V_s -> V_e + (V_s - V_e) exp(-nu_{s,e} dt),
 * applied as a uniform per-cell shift to every particle:
 *     v_p -= (V_s - V_e)(1 - exp(-nu_{s,e} dt)).
 * Only the bulk drift is decelerated; the thermal spread (T_i) is preserved.
 * Drag only, no diffusion -- NOT a Q_ei temperature-relaxation channel.
 *
 * No electron heat is deposited here: the eta J^2 source on T_e is the separate
 * HybridPICModel::QDSMCAddJouleHeating (Phys. Plasmas 31, 012902 (2024), Eq. 12).
 * This operator only supplies the Newton's-3rd-law back-reaction to the eta*J
 * term ions feel through Ohm's law, cancelling the hybrid-PIC anti-friction.
 */
class HybridResistiveDrag final
    : public CollisionBase
{
public:
    explicit HybridResistiveDrag (const std::string& collision_name);

    ~HybridResistiveDrag () override = default;

    HybridResistiveDrag ( HybridResistiveDrag const &)             = delete;
    HybridResistiveDrag& operator= ( HybridResistiveDrag const & ) = delete;
    HybridResistiveDrag ( HybridResistiveDrag&& )                  = delete;
    HybridResistiveDrag& operator= ( HybridResistiveDrag&& )       = delete;

    void doCollisions (amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc) override;
};

#endif // WARPX_PARTICLES_COLLISION_HYBRIDRESISTIVEDRAG_H_
