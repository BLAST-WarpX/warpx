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

class ParticleThermalizer {
public:
    enum class Normal { X, Y, Z };

    // Construct and read parameters from inputs under the `particle_thermalizer` block.
    ParticleThermalizer();

private:
    bool m_defined;
    Normal m_normal;
    amrex::Real m_start;
    amrex::Real m_end;
    amrex::Real m_momentum_threshold;
    amrex::Real m_temperature;
};

#endif // WARPX_PARTICLETHERMALIZER_H
