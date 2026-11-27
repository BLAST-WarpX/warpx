/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_CARTESIANPHOTONBOUNDARY_H_
#define WARPX_RADIATION_CARTESIANPHOTONBOUNDARY_H_

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cstring>

namespace warpx::radiation
{
using namespace amrex::literals;

/** Kinetic photon face policy. Integer values are checkpointed. */
enum class PhotonFacePolicy : int
{
    Reflecting = 0,
    Absorbing = 1,
    Periodic = 2,
    None = 3
};

/** Result of one Cartesian domain-leaving face classification.
 *
 * `leaving_side[d]` is -1 at the low face, +1 at the high face, and 0 if
 * direction `d` does not leave the domain on this segment. Periodic and
 * axis (`None`) faces are ignored here: the caller keeps the ghost-cell
 * increment. Absorbing owns a mixed absorbing/reflecting corner so a packet
 * is counted once and is not also reversed.
 */
struct CartesianPhotonDomainCrossing
{
    bool escaped = false;
    bool reflected = false;
    amrex::GpuArray<int, 3> reverse{};
};

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
CartesianPhotonDomainCrossing
ClassifyCartesianPhotonDomainCrossing (
    amrex::GpuArray<int, 3> const& leaving_side,
    amrex::GpuArray<int, 3> const& policy_lo,
    amrex::GpuArray<int, 3> const& policy_hi) noexcept
{
    CartesianPhotonDomainCrossing result{};
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
        if (leaving_side[direction] == 0) { continue; }
        int const policy = leaving_side[direction] > 0
            ? policy_hi[direction] : policy_lo[direction];
        if (policy == static_cast<int>(PhotonFacePolicy::Periodic)
            || policy == static_cast<int>(PhotonFacePolicy::None))
        {
            continue;
        }
        if (policy == static_cast<int>(PhotonFacePolicy::Absorbing)) {
            result.escaped = true;
        } else if (policy == static_cast<int>(PhotonFacePolicy::Reflecting)) {
            result.reverse[direction] = 1;
            result.reflected = true;
        }
    }
    if (result.escaped) {
        result.reflected = false;
        result.reverse = {};
    }
    return result;
}

/** Specular fold of one coordinate through a wall. */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::ParticleReal
SpecularPhotonCoordinate (
    amrex::ParticleReal const position, amrex::Real const wall) noexcept
{
    return static_cast<amrex::ParticleReal>(
        2.0_rt * wall - static_cast<amrex::Real>(position));
}

[[nodiscard]] inline
int
PhotonFacePolicyFromName (char const* const name) noexcept
{
    if (name == nullptr) { return -1; }
    if (std::strcmp(name, "absorbing") == 0 || std::strcmp(name, "open") == 0) {
        return static_cast<int>(PhotonFacePolicy::Absorbing);
    }
    if (std::strcmp(name, "reflecting") == 0) {
        return static_cast<int>(PhotonFacePolicy::Reflecting);
    }
    if (std::strcmp(name, "periodic") == 0) {
        return static_cast<int>(PhotonFacePolicy::Periodic);
    }
    if (std::strcmp(name, "none") == 0) {
        return static_cast<int>(PhotonFacePolicy::None);
    }
    return -1;
}

[[nodiscard]] inline
char const*
PhotonFacePolicyName (int const policy) noexcept
{
    if (policy == static_cast<int>(PhotonFacePolicy::Absorbing)) {
        return "absorbing";
    }
    if (policy == static_cast<int>(PhotonFacePolicy::Reflecting)) {
        return "reflecting";
    }
    if (policy == static_cast<int>(PhotonFacePolicy::Periodic)) {
        return "periodic";
    }
    if (policy == static_cast<int>(PhotonFacePolicy::None)) {
        return "none";
    }
    return "invalid";
}
} // namespace warpx::radiation

#endif
