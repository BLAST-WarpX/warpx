/* Copyright 2026 The WarpX Community
 *
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

/** Hybrid-PIC electron-ion drag operator (drag-only, no heat deposit).
 *
 * Acts on a single charged ion species. The per-species collision rate
 * comes from the Ohm's-law eta parser registered on HybridPICModel
 * (`hybrid_pic_model.plasma_resistivity`):
 *
 *     nu_{s,e}  =  Z_s * e^2 * eta(rho, |J|, t) * n_e / m_s
 *
 * Per particle the update is the operator-split bulk-shift form
 *
 *     v_p^{n+1}  =  V_e + (v_p^n - V_e) * exp(-nu_{s,e} * dt)
 *
 * which decelerates the bulk ion drift toward V_e without contracting the
 * thermal moment (uniform per-cell shift; no Langevin diffusion -- this
 * operator does not equilibrate T_i toward T_e and must NOT be used for
 * the Q_ei temperature-relaxation channel).
 *
 * The corresponding electron-heat dissipation is NOT computed here.
 * S_e on T_e is instead built per cell from gridded fields by
 * HybridPICModel::QDSMCAddJouleHeatingKinetic (Topanga Eq. 12), which is
 * a separate concern controlled by `joule_heating_mode`. Registering
 * this drag operator therefore changes only ion dynamics -- it adds the
 * Newton's 3rd-law back-reaction to the eta*J piece of qE that ions feel
 * through Ohm's law, cancelling the well-known hybrid-PIC anti-friction.
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
