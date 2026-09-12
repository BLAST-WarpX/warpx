/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_RADIATIONENERGYUPDATE_H_
#define WARPX_RADIATION_RADIATIONENERGYUPDATE_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <limits>

namespace warpx::radiation
{
    /** Result of a positivity-preserving radiation-energy update [J/cell]. */
    struct RadiationEnergyUpdateResult
    {
        amrex::Real stored_energy = amrex::Real(0.0);
        amrex::Real residual = amrex::Real(0.0);
        bool valid = true;
    };

    /** Apply one shared raw-energy update with a precision-scaled tolerance.
     *
     * The signed residual is raw-stored_energy.  Thus a small negative raw
     * energy clipped to zero is recorded as a negative numerical-energy
     * residual, while a rejected update leaves the caller's old value intact.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RadiationEnergyUpdateResult ApplyRadiationEnergyUpdate (
        amrex::Real const old_energy,
        amrex::Real const energy_change) noexcept
    {
        RadiationEnergyUpdateResult result;
        result.stored_energy = old_energy;
        if (!amrex::Math::isfinite(old_energy)
            || !amrex::Math::isfinite(energy_change))
        {
            result.valid = false;
            return result;
        }

        amrex::Real const raw_energy = old_energy + energy_change;
        amrex::Real const old_abs = amrex::Math::abs(old_energy);
        amrex::Real const change_abs = amrex::Math::abs(energy_change);
        amrex::Real const scale = old_abs > change_abs ? old_abs : change_abs;
        amrex::Real const tolerance = amrex::Real(100.0)
            * std::numeric_limits<amrex::Real>::epsilon() * scale;
        if (!amrex::Math::isfinite(raw_energy) || raw_energy < -tolerance) {
            result.valid = false;
            return result;
        }

        result.stored_energy = raw_energy > amrex::Real(0.0)
            ? raw_energy
            : amrex::Real(0.0);
        result.residual = raw_energy - result.stored_energy;
        result.valid = amrex::Math::isfinite(result.residual);
        if (!result.valid) {
            result.stored_energy = old_energy;
            result.residual = amrex::Real(0.0);
        }
        return result;
    }
}

#endif // WARPX_RADIATION_RADIATIONENERGYUPDATE_H_
