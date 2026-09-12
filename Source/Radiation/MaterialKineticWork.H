/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_MATERIALKINETICWORK_H_
#define WARPX_RADIATION_MATERIALKINETICWORK_H_

#include "Utils/WarpXConst.H"

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace warpx::radiation
{
    /** Work of an unrounded requested proper-velocity increment. Unlike
     * subtracting two stored velocities, this retains a sub-ULP request. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    MaterialSpecificRequestedWork (
        amrex::GpuArray<amrex::Real, 3> const& old_u,
        amrex::GpuArray<amrex::Real, 3> const& delta_u) noexcept
    {
        amrex::Real old_squared = 0;
        amrex::Real new_squared = 0;
        amrex::Real delta_squared = 0;
        amrex::Real old_dot_delta = 0;
        for (int d = 0; d < 3; ++d)
        {
            amrex::Real const next = old_u[d] + delta_u[d];
            old_squared += old_u[d] * old_u[d];
            new_squared += next * next;
            delta_squared += delta_u[d] * delta_u[d];
            old_dot_delta += old_u[d] * delta_u[d];
        }
        amrex::Real const gamma_old = std::sqrt(1 + old_squared / PhysConst::c2);
        amrex::Real const gamma_new = std::sqrt(1 + new_squared / PhysConst::c2);
        return (2 * old_dot_delta + delta_squared) / (gamma_old + gamma_new);
    }

    /** Actual finite-kick change of kinetic energy per unit rest mass [m^2/s^2].
     * Inputs are the represented proper velocities before and after the kick,
     * not the requested impulse. Rationalizing gamma_new-gamma_old avoids
     * subtracting rest energies and retains small work on moving particles.
     * This function changes no state; callers must validate the candidate and
     * its paired energy reservoir before committing a transactional update.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    MaterialSpecificKineticWork (
        amrex::GpuArray<amrex::Real, 3> const& old_u,
        amrex::GpuArray<amrex::Real, 3> const& new_u) noexcept
    {
        amrex::Real old_squared = 0;
        amrex::Real new_squared = 0;
        amrex::Real delta_squared = 0;
        amrex::Real old_dot_delta = 0;
        for (int d = 0; d < 3; ++d)
        {
            amrex::Real const delta = new_u[d] - old_u[d];
            old_squared += old_u[d] * old_u[d];
            new_squared += new_u[d] * new_u[d];
            delta_squared += delta * delta;
            old_dot_delta += old_u[d] * delta;
        }
        amrex::Real const gamma_old = std::sqrt(1 + old_squared / PhysConst::c2);
        amrex::Real const gamma_new = std::sqrt(1 + new_squared / PhysConst::c2);
        return (2 * old_dot_delta + delta_squared) / (gamma_old + gamma_new);
    }

    struct MaterialImpulseCandidate
    {
        amrex::GpuArray<amrex::ParticleReal, 3> velocity{};
        // Per unit particle rest mass; multiply by weight*mass for inventories.
        amrex::GpuArray<amrex::Real, 3> momentum_carry{};
        amrex::Real energy_carry = 0;
        amrex::Real requested_work = 0;
        amrex::Real actual_work = 0;
        bool valid = false;
    };

    struct MaterialCarryReflection
    {
        // Per unit rest mass: Cartesian pending momentum and signed pending work.
        amrex::GpuArray<amrex::Real, 4> carry{};
        amrex::GpuArray<amrex::Real, 4> boundary_transfer{};
        bool valid = false;
    };

    /** Candidate for an elastic, stationary Cartesian particle reflection.
     * Use actual boundary-event flags, not signs inferred from particle velocity.
     * The boundary transfer accounts for pending radiation impulse only, not
     * the represented particle-wall impulse already assigned to material.
     * Neither input nor a diagnostic is mutated. Thermalization and loss require
     * different ownership rules and are deliberately rejected here.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE MaterialCarryReflection
    EvaluateMaterialCarryReflection (
        amrex::GpuArray<amrex::Real, 4> const& carry,
        amrex::GpuArray<bool, 3> const& cartesian_reflection,
        bool thermalized, bool lost) noexcept
    {
        MaterialCarryReflection result;
        if (thermalized || lost) {
            return result;
        }
        for (auto value : carry) {
            if (!amrex::Math::isfinite(value)) {
                return result;
            }
        }
        result.carry = carry;
        for (int d = 0; d < 3; ++d) {
            if (cartesian_reflection[d]) {
                result.carry[d] = -carry[d];
                result.boundary_transfer[d] = carry[d] - result.carry[d];
                if (!amrex::Math::isfinite(result.boundary_transfer[d])) {
                    return {};
                }
            }
        }
        // Orthogonal reflection preserves both |u| and |u+delta_u|, so it
        // neither realizes nor discards the signed pending kinetic-work account.
        result.valid = true;
        return result;
    }

    /** Pure candidate for a particle-owned radiation impulse. All vectors are
     * Cartesian proper velocities or impulses divided by particle rest mass.
     * No mesh reservoir, particle state or diagnostic is mutated. The caller
     * must validate aggregate radiation/thermal changes and precision bounds
     * before committing this candidate. This helper alone does not supply
     * migration, boundary export, or coupling to the PIC pusher.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE MaterialImpulseCandidate
    EvaluateMaterialImpulseCandidate (
        amrex::GpuArray<amrex::ParticleReal, 3> const& velocity,
        amrex::GpuArray<amrex::Real, 3> const& momentum_carry,
        amrex::Real const energy_carry,
        amrex::GpuArray<amrex::Real, 3> const& requested_delta_u) noexcept
    {
        MaterialImpulseCandidate result;
        amrex::GpuArray<amrex::Real, 3> before{};
        amrex::GpuArray<amrex::Real, 3> after{};
        if (!amrex::Math::isfinite(energy_carry))
        {
            return result;
        }
        for (int d = 0; d < 3; ++d)
        {
            if (!amrex::Math::isfinite(velocity[d]) ||
                !amrex::Math::isfinite(momentum_carry[d]) ||
                !amrex::Math::isfinite(requested_delta_u[d]))
            {
                return result;
            }
            before[d] = velocity[d];
            amrex::Real const target = momentum_carry[d] + requested_delta_u[d];
            result.velocity[d] = static_cast<amrex::ParticleReal>(before[d] + target);
            after[d] = result.velocity[d];
            result.momentum_carry[d] = target - (after[d] - before[d]);
            if (!amrex::Math::isfinite(after[d]) ||
                !amrex::Math::isfinite(result.momentum_carry[d]))
            {
                return result;
            }
        }
        result.requested_work = MaterialSpecificRequestedWork(before, requested_delta_u);
        result.actual_work = MaterialSpecificKineticWork(before, after);
        result.energy_carry = energy_carry + (result.requested_work - result.actual_work);
        result.valid = amrex::Math::isfinite(result.requested_work)
            && amrex::Math::isfinite(result.actual_work)
            && amrex::Math::isfinite(result.energy_carry);
        return result;
    }
}
#endif
