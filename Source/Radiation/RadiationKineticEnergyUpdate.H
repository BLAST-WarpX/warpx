/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_RADIATIONKINETICENERGYUPDATE_H_
#define WARPX_RADIATION_RADIATIONKINETICENERGYUPDATE_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <limits>

namespace warpx::radiation
{
    /** Return the larger of field and particle representation epsilons. */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real KineticPrecisionEpsilon (
        amrex::Real const real_epsilon,
        amrex::Real const particle_epsilon) noexcept
    {
        return real_epsilon > particle_epsilon
            ? real_epsilon
            : particle_epsilon;
    }

    /** Return the active field/particle precision epsilon. */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real KineticPrecisionEpsilon () noexcept
    {
        amrex::Real const real_epsilon =
            std::numeric_limits<amrex::Real>::epsilon();
        auto const particle_epsilon = static_cast<amrex::Real>(
            std::numeric_limits<amrex::ParticleReal>::epsilon());
        return KineticPrecisionEpsilon(real_epsilon, particle_epsilon);
    }

    /** Result of reconciling a requested material-energy transfer [J/cell]
     * with the transfer realized by kinetic particles [J/cell]. */
    struct KineticEnergyUpdateResult
    {
        amrex::Real realized_energy_change = amrex::Real(0.0);
        amrex::Real residual = amrex::Real(0.0);
        bool valid = true;
    };

    /** Classify a kinetic particle-energy realization using a fixed tolerance.
     *
     * q is the requested material-ledger transfer, old is the pre-update
     * particle kinetic energy, target is old+q, and achieved is the measured
     * post-update particle kinetic energy.  A valid result records d=achieved-
     * old and the signed residual R=q-d.  The caller must not replace its
     * material ledger with d when valid is false.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    KineticEnergyUpdateResult EvaluateKineticEnergyUpdate (
        amrex::Real const q,
        amrex::Real const old_energy,
        amrex::Real const target_energy,
        amrex::Real const achieved_energy) noexcept
    {
        KineticEnergyUpdateResult result;
        if (!amrex::Math::isfinite(q)
            || !amrex::Math::isfinite(old_energy)
            || !amrex::Math::isfinite(target_energy)
            || !amrex::Math::isfinite(achieved_energy))
        {
            result.valid = false;
            return result;
        }

        amrex::Real const precision_epsilon = KineticPrecisionEpsilon();
        amrex::Real const coefficient =
            amrex::Real(2.0e-10) > amrex::Real(1024.0) * precision_epsilon
            ? amrex::Real(2.0e-10)
            : amrex::Real(1024.0) * precision_epsilon;
        amrex::Real scale = amrex::Math::abs(q);
        amrex::Real const old_abs = amrex::Math::abs(old_energy);
        amrex::Real const target_abs = amrex::Math::abs(target_energy);
        amrex::Real const achieved_abs = amrex::Math::abs(achieved_energy);
        scale = std::max(scale, old_abs);
        scale = std::max(scale, target_abs);
        scale = std::max(scale, achieved_abs);
        amrex::Real const tolerance = coefficient * scale;
        amrex::Real const mismatch =
            amrex::Math::abs(achieved_energy - target_energy);
        if (!amrex::Math::isfinite(tolerance)
            || !amrex::Math::isfinite(mismatch)
            || mismatch > tolerance)
        {
            result.valid = false;
            return result;
        }

        result.realized_energy_change = achieved_energy - old_energy;
        result.residual = q - result.realized_energy_change;
        result.valid = amrex::Math::isfinite(result.realized_energy_change)
            && amrex::Math::isfinite(result.residual);
        if (!result.valid) {
            result.realized_energy_change = amrex::Real(0.0);
            result.residual = amrex::Real(0.0);
        }
        return result;
    }
}

#endif // WARPX_RADIATION_RADIATIONKINETICENERGYUPDATE_H_
