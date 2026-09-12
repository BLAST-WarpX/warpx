/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_PARTICLEIMPULSEBOUNDARY_H_
#define WARPX_RADIATION_PARTICLEIMPULSEBOUNDARY_H_

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <functional>
#include <string>
#include <vector>

struct ParticleBoundaries;
class WarpXParticleContainer;

namespace warpx::radiation
{
    struct ParticleImpulseBoundaryTransfer {
        std::string path;
        // Global pending-only wall momentum [kg m/s], NOT represented ion impulse.
        amrex::GpuArray<amrex::Real, 3> momentum{};
    };

    std::vector<std::string> RegisteredParticleImpulsePaths(WarpXParticleContainer &species);

    /** Native elastic Cartesian reflection plus all registered radiation carries.
     * Stage the complete container and direct event transfers before committing.
     * Failure leaves particles and the supplied transfer vector unchanged.
     * Success does not redistribute particles or accumulate a persistent ledger.
     * The optional collective accept callback must stage external accounting
     * privately; it can reject the proposed transfers before particle commit.
     */
    bool TryReflectParticleImpulseState(
        WarpXParticleContainer &species, ParticleBoundaries const &boundaries,
        std::vector<ParticleImpulseBoundaryTransfer> &transfers,
        std::function<bool(std::vector<ParticleImpulseBoundaryTransfer> const &)> const &accept =
            {});
} // namespace warpx::radiation
#endif
