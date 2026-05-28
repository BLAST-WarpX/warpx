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

#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

#include <string>

enum class HybridResistiveDragExpressionType {
    CollisionFrequency,
    Resistivity,
};

/** Coulomb electron-ion drag operator for the hybrid-PIC solver.
 *
 * Acts on a single charged ion species. The drag coefficient is supplied as
 * a runtime-parsed expression nu(rho, Vs, Ve, Ts, Te, B) (or, equivalently,
 * a resistivity eta with the same arguments, internally converted via
 * nu = eta * Z_s^2 * e^2 * n_s / m_e). The particle update,
 *   v_p^{n+1} = V_e + (v_p^n - V_e) * exp(-nu * dt),
 * is operator-split into the particle push (wired in a subsequent phase).
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

private:

    HybridResistiveDragExpressionType m_expression_type =
        HybridResistiveDragExpressionType::CollisionFrequency;

    amrex::Parser m_drag_parser;
    amrex::ParserExecutor<6> m_drag_func;
};

#endif // WARPX_PARTICLES_COLLISION_HYBRIDRESISTIVEDRAG_H_
