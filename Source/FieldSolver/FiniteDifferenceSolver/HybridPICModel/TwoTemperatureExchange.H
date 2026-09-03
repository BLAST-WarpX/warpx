/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_TWOTEMPERATUREEXCHANGE_H_
#define WARPX_TWOTEMPERATUREEXCHANGE_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

/** Exact frozen-coefficient exchange between two finite heat capacities. */
struct TwoTemperatureExchangeState
{
    amrex::Real electron_temperature = amrex::Real(0.0);
    amrex::Real ion_temperature = amrex::Real(0.0);
    // Positive when energy is transferred to electrons [J/m^3].
    amrex::Real electron_energy_change_density = amrex::Real(0.0);
    bool valid = true;
};

/**
 * Solve
 *
 *   C_e dT_e/dt = G (T_i-T_e),
 *   C_i dT_i/dt =-G (T_i-T_e)
 *
 * for constant positive volumetric heat capacities and conductance over one
 * source step.  The result is conservative to roundoff, cannot overshoot the
 * common equilibrium temperature, and remains bounded for an arbitrarily
 * stiff step.  Nonlinear EOS callers use the returned energy change with
 * their caloric inverse instead of treating the linearized T_e as authoritative.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
TwoTemperatureExchangeState exactTwoTemperatureExchange (
    amrex::Real const electron_temperature,
    amrex::Real const ion_temperature,
    amrex::Real const electron_heat_capacity_density,
    amrex::Real const ion_heat_capacity_density,
    amrex::Real const exchange_conductance_density,
    amrex::Real const dt) noexcept
{
    if (!amrex::Math::isfinite(electron_temperature)
        || !amrex::Math::isfinite(ion_temperature)
        || !amrex::Math::isfinite(electron_heat_capacity_density)
        || !amrex::Math::isfinite(ion_heat_capacity_density)
        || !amrex::Math::isfinite(exchange_conductance_density)
        || !amrex::Math::isfinite(dt)
        || electron_temperature < amrex::Real(0.0)
        || ion_temperature < amrex::Real(0.0)
        || !(electron_heat_capacity_density > amrex::Real(0.0))
        || !(ion_heat_capacity_density > amrex::Real(0.0))
        || exchange_conductance_density < amrex::Real(0.0)
        || dt < amrex::Real(0.0))
    {
        amrex::Real const invalid =
            std::numeric_limits<amrex::Real>::quiet_NaN();
        return {invalid, invalid, invalid, false};
    }
    if (exchange_conductance_density == amrex::Real(0.0)
        || dt == amrex::Real(0.0))
    {
        return {electron_temperature, ion_temperature, amrex::Real(0.0), true};
    }

    amrex::Real const coupled_rate = exchange_conductance_density
        * (amrex::Real(1.0) / electron_heat_capacity_density
            + amrex::Real(1.0) / ion_heat_capacity_density);
    amrex::Real const exchange_fraction =
        -std::expm1(-coupled_rate * dt);
    amrex::Real const series_heat_capacity =
        electron_heat_capacity_density * ion_heat_capacity_density
        / (electron_heat_capacity_density + ion_heat_capacity_density);
    amrex::Real const electron_energy_change = series_heat_capacity
        * (ion_temperature - electron_temperature) * exchange_fraction;
    return {
        electron_temperature + electron_energy_change
            / electron_heat_capacity_density,
        ion_temperature - electron_energy_change
            / ion_heat_capacity_density,
        electron_energy_change,
        true};
}

#endif // WARPX_TWOTEMPERATUREEXCHANGE_H_
