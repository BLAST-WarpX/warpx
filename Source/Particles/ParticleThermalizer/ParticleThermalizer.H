/* Copyright 2026 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLETHERMALIZER_H
#define WARPX_PARTICLETHERMALIZER_H

#include <string>
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
    bool defined () const;

    // Apply the thermalizer to particles in the provided MultiParticleContainer.
    void applyThermalizer (MultiParticleContainer &mpc);

    // Per-species overload; operates on a single WarpXParticleContainer.
    // Should be protected, but can't be because it has an extended __device__ lambda
    void applyThermalizer (WarpXParticleContainer &pcont);

private:
    bool m_defined;

    // The normal direction of the thermalizer ('x', 'y', or 'z')
    std::string m_normal_str;

    // The normal direction as an integer USING THE AMREX CONVENTION (0, 1, or 2)
    // in 1D, "z" is 0
    // in 2D, "x" is 0, "z" is 1
    // in 3D, "x" is 0, "y" is 1, "z" is 2
    int m_normal;

    // The start and end positions of the thermalizer region along the normal direction
    amrex::Real m_start;
    amrex::Real m_end;

    // The momentum threshold above which particles are thermalized in normalized units (i.e. u = gamma*beta)
    amrex::Real m_momentum_threshold;

    // dimensionless temperature theta = kB*T/(m*c^2) used to sample thermal momentum distribution
    amrex::Real m_theta;
};

#endif // WARPX_PARTICLETHERMALIZER_H
