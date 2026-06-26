/* Copyright 2026 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLETHERMALIZER_H
#define WARPX_PARTICLETHERMALIZER_H

#include <string>
#include <vector>

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

// Forward declaration to avoid heavy includes in the header
class MultiParticleContainer;
class WarpXParticleContainer;

class ParticleThermalizer {
public:

    // Construct and read parameters from inputs under the `particle_thermalizer` block.
    ParticleThermalizer ();

    // Returns true if the particle_thermalizer parameters were provided
    // and the thermalizer is therefore on.
    [[nodiscard]] bool defined () const;

    // Apply the thermalizer to particles in the provided MultiParticleContainer.
    void applyThermalizer (MultiParticleContainer &mpc) const;

    // Per-species overload; operates on a single WarpXParticleContainer.
    // Should be protected, but can't be because it has an extended __device__ lambda
    void applyThermalizer (WarpXParticleContainer &pcont) const;

private:
    bool m_defined{false};

    // The normal direction as an integer USING THE AMREX CONVENTION (0, 1, or 2)
    // in 1D, "z" is 0
    // in 2D, "x" is 0, "z" is 1
    // in 3D, "x" is 0, "y" is 1, "z" is 2
    int m_normal{-1};

    // The start and end positions of the thermalizer region along the normal direction
    amrex::Real m_start;
    amrex::Real m_end;

    // The threshold for each momentum component above which particles are thermalized in normalized units (i.e. u = gamma*beta)
    amrex::Array<amrex::Real, 3> m_momentum_threshold;

    // dimensionless temperature theta = kB*T/(m*c^2) used to sample thermal momentum distribution
    amrex::Real m_theta;

    // If non-empty, only these species are thermalized; if empty, all species are thermalized.
    std::vector<std::string> m_species_names;
};

#endif // WARPX_PARTICLETHERMALIZER_H
