/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_PLANCKEXCHANGE_H_
#define WARPX_RADIATION_PLANCKEXCHANGE_H_

#include "Utils/WarpXConst.H"

#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace warpx::radiation
{
    using namespace amrex::literals;

    struct PlanckExchangeResult
    {
        amrex::Real exchange_energy = 0.0_rt;
        amrex::Real new_radiation_energy = 0.0_rt;
        bool valid = true;

        // A material-energy limiter only scales net emission. Preserve the
        // independently computed outgoing state for unlimited absorption.
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void limitEmission (amrex::Real old_energy, amrex::Real scale) noexcept
        {
            if (exchange_energy > 0.0_rt && scale < 1.0_rt) {
                exchange_energy *= scale;
                new_radiation_energy = old_energy + exchange_energy;
            }
        }
    };

    /** Exact frozen-state solution of dEr/dt=c*(kappa_e*B-kappa_a*Er).
     *
     * Positive exchange transfers energy from matter to radiation.  Distinct
     * emission and absorption coefficients permit a non-LTE source while the
     * equal-coefficient limit exactly recovers the historical LTE operator.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    PlanckExchangeResult EvaluatePlanckExchange (
        amrex::Real const absorption, amrex::Real const emission,
        amrex::Real const old_radiation_energy,
        amrex::Real const equilibrium_energy, amrex::Real const dt) noexcept
    {
        PlanckExchangeResult result;
        result.new_radiation_energy = old_radiation_energy;
        if (absorption < 0.0_rt || emission < 0.0_rt
            || old_radiation_energy < 0.0_rt || equilibrium_energy < 0.0_rt
            || dt < 0.0_rt || !amrex::Math::isfinite(absorption)
            || !amrex::Math::isfinite(emission)
            || !amrex::Math::isfinite(old_radiation_energy)
            || !amrex::Math::isfinite(equilibrium_energy)
            || !amrex::Math::isfinite(dt))
        {
            result.valid = false;
            return result;
        }

        if (dt == 0.0_rt) { return result; }
        if (absorption > 0.0_rt) {
            amrex::Real const depth = absorption * PhysConst::c * dt;
            amrex::Real const fraction = -std::expm1(-depth);
            amrex::Real const emitted = emission * equilibrium_energy * (fraction / absorption);
            result.exchange_energy = emitted - old_radiation_energy * fraction;
            result.new_radiation_energy = result.exchange_energy < -0.5_rt * old_radiation_energy
                ? old_radiation_energy * std::exp(-depth) + emitted
                : old_radiation_energy + result.exchange_energy;
        } else {
            result.exchange_energy = emission * PhysConst::c * dt * equilibrium_energy;
            result.new_radiation_energy = old_radiation_energy + result.exchange_energy;
        }
        result.valid = result.new_radiation_energy >= 0.0_rt
            && amrex::Math::isfinite(result.exchange_energy)
            && amrex::Math::isfinite(result.new_radiation_energy);
        return result;
    }
}

#endif // WARPX_RADIATION_PLANCKEXCHANGE_H_
