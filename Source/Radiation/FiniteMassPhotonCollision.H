/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_FINITEMASSPHOTONCOLLISION_H_
#define WARPX_RADIATION_FINITEMASSPHOTONCOLLISION_H_

#include "FrameTransform.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

namespace warpx::radiation
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
    ValidFiniteMassPhoton (FourVector const& photon) noexcept
    {
        for (auto const value : photon)
        {
            if (!amrex::Math::isfinite(value)) { return false; }
        }
        if (!(photon[0] > 0)) { return false; }
        amrex::Real norm_squared = 0;
        for (int d = 1; d < 4; ++d)
        {
            auto const direction = photon[d] / photon[0];
            norm_squared += direction * direction;
        }
        return amrex::Math::isfinite(norm_squared) &&
            std::abs(norm_squared - 1) <= 256 * std::numeric_limits<amrex::Real>::epsilon();
    }

    struct CoherentPhotonCollisionResult
    {
        FourVector outgoing_photon{};
        amrex::GpuArray<amrex::Real, 3> material_impulse{};
        amrex::Real radiation_debit = 0;
        bool valid = false;
    };

    /** Coherent collision in the supplied material secant frame.
     *
     * incoming_photon is a lab-frame [E,c*p] four-vector and frame_direction
     * is the already-drawn unit outgoing direction in that frame. beta is the
     * staged WorkVelocity/c. No random state is consumed here: replay code must
     * keep frame_direction frozen for every fixed-point iteration.
     *
     * On failure the output parameters are deliberately left unchanged. The
     * null and unit-vector checks are relative to the supplied vector scales;
     * there are no velocity, energy, or residual floors.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
    CoherentPhotonCollision (
        FourVector const& incoming_photon,
        amrex::GpuArray<amrex::Real, 3> const& frame_direction,
        amrex::GpuArray<amrex::Real, 3> const& beta,
        FourVector& outgoing_photon, amrex::GpuArray<amrex::Real, 3>& material_impulse,
        amrex::Real& radiation_debit) noexcept
    {
        if (!ValidFiniteMassPhoton(incoming_photon)) { return false; }
        amrex::Real beta_squared = 0;
        amrex::Real beta_dot_direction = 0;
        amrex::Real direction_squared = 0;
        for (int d = 0; d < 3; ++d)
        {
            if (!amrex::Math::isfinite(frame_direction[d]) ||
                !amrex::Math::isfinite(beta[d]))
            {
                return false;
            }
            beta_squared += beta[d] * beta[d];
            beta_dot_direction += beta[d] * frame_direction[d];
            direction_squared += frame_direction[d] * frame_direction[d];
        }
        if (!(beta_squared < 1) || !amrex::Math::isfinite(direction_squared) ||
            !(direction_squared > 0) ||
            std::abs(direction_squared - 1) >
                256 * std::numeric_limits<amrex::Real>::epsilon() * direction_squared ||
            !(1 + beta_dot_direction > 0))
        {
            return false;
        }

        amrex::GpuArray<amrex::Real, 3> inverse_beta{};
        for (int d = 0; d < 3; ++d) { inverse_beta[d] = -beta[d]; }
        FourVector const frame_photon = BoostFourVector(incoming_photon, inverse_beta);
        if (!ValidFiniteMassPhoton(frame_photon)) { return false; }

        FourVector frame_outgoing{};
        for (int d = 0; d < 3; ++d)
        {
            frame_outgoing[d + 1] = frame_photon[0] * frame_direction[d];
        }
        frame_outgoing[0] = frame_photon[0];
        FourVector const lab_outgoing = BoostFourVector(frame_outgoing, beta);
        if (!ValidFiniteMassPhoton(lab_outgoing)) { return false; }

        amrex::GpuArray<amrex::Real, 3> candidate_impulse{};
        for (int d = 0; d < 3; ++d)
        {
            candidate_impulse[d] =
                (incoming_photon[d + 1] - lab_outgoing[d + 1]) / PhysConst::c;
            if (!amrex::Math::isfinite(candidate_impulse[d]))
            {
                return false;
            }
        }
        amrex::Real const candidate_debit = incoming_photon[0] - lab_outgoing[0];
        if (!amrex::Math::isfinite(candidate_debit))
        {
            return false;
        }

        outgoing_photon = lab_outgoing;
        material_impulse = candidate_impulse;
        radiation_debit = candidate_debit;
        return true;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CoherentPhotonCollisionResult
    CoherentPhotonCollision (
        FourVector const& incoming_photon,
        amrex::GpuArray<amrex::Real, 3> const& frame_direction,
        amrex::GpuArray<amrex::Real, 3> const& beta) noexcept
    {
        CoherentPhotonCollisionResult result;
        result.valid = CoherentPhotonCollision(
            incoming_photon, frame_direction, beta, result.outgoing_photon,
            result.material_impulse, result.radiation_debit);
        return result;
    }
}

#endif
