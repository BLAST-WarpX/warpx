/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcMetricTransport.H"

#include <AMReX_REAL.H>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>

namespace
{
    using namespace amrex::literals;
    bool close (
        amrex::Real actual,
        amrex::Real expected,
        amrex::Real relative_tolerance =
            64.0_rt * std::numeric_limits<amrex::Real>::epsilon())
    {
        amrex::Real const scale = std::max(
            std::abs(actual), std::abs(expected));
        return std::abs(actual - expected)
            <= relative_tolerance * std::max(scale, 1.0_rt);
    }
}

int main ()
{
    constexpr amrex::Real r0 = 1.25_rt;
    constexpr amrex::Real dr = 0.0625_rt;
    constexpr amrex::Real dz = 0.375_rt;
    constexpr amrex::Real rho = 1.7e20_rt;
    constexpr amrex::Real c = 2.4_rt;
    constexpr amrex::Real constant_velocity = 0.3_rt;
    constexpr std::array<amrex::Real, 4> radii{
        r0 + dr, r0 + 2.0_rt * dr, r0 + 3.0_rt * dr,
        r0 + 4.0_rt * dr};

    bool pass = true;
    pass = pass
        && close(
            warpx::hybrid::cylindricalTransportNodeVolume(
                0.0_rt, dr, dz, 1.0_rt / 3.0_rt),
            3.14159265358979323846_rt * dr * dr * dz / 3.0_rt)
        && close(
            warpx::hybrid::cylindricalTransportNodeVolume(
                0.0_rt, dr, dz, 1.0_rt / 4.0_rt),
            3.14159265358979323846_rt * dr * dr * dz / 4.0_rt);

    for (amrex::Real const r : radii) {
        amrex::Real const r_lo = r - 0.5_rt * dr;
        amrex::Real const r_hi = r + 0.5_rt * dr;
        amrex::Real const volume =
            warpx::hybrid::cylindricalTransportNodeVolume(r, dr, dz);
        amrex::Real const expected_volume =
            3.14159265358979323846_rt
            * (r_hi * r_hi - r_lo * r_lo) * dz;
        amrex::Real const area_lo =
            warpx::hybrid::cylindricalRadialFaceArea(r_lo, dz);
        amrex::Real const area_hi =
            warpx::hybrid::cylindricalRadialFaceArea(r_hi, dz);
        amrex::Real const axial_area =
            warpx::hybrid::cylindricalAxialFaceArea(r_lo, r_hi);

        pass = pass && close(volume, expected_volume);

        // V_r=C/r is analytically divergence free. Evaluating its mass flux
        // through the two cylindrical faces must cancel identically.
        amrex::Real const inverse_c_divergence =
            warpx::hybrid::metricFluxDivergence(
                rho * c / r_lo, rho * c / r_hi, area_lo, area_hi,
                1.0_rt / volume);
        amrex::Real const inverse_c_roundoff_bound = 256.0_rt
            * std::numeric_limits<amrex::Real>::epsilon()
            * std::abs(rho * c) * 2.0_rt
            * 3.14159265358979323846_rt * dz / volume;
        pass = pass
            && std::abs(inverse_c_divergence) <= inverse_c_roundoff_bound;

        // A constant radial velocity has div(V)=V/r in cylindrical geometry.
        amrex::Real const constant_divergence =
            warpx::hybrid::metricFluxDivergence(
                rho * constant_velocity, rho * constant_velocity, area_lo,
                area_hi, 1.0_rt / volume) / rho;
        pass = pass && close(constant_divergence, constant_velocity / r);

        // Equal axial areas cancel a constant flux, while a unit gradient in
        // z gives unit divergence.
        amrex::Real const axial_constant_divergence =
            warpx::hybrid::metricFluxDivergence(
                rho * constant_velocity, rho * constant_velocity, axial_area,
                axial_area, 1.0_rt / volume);
        amrex::Real const axial_gradient_divergence =
            warpx::hybrid::metricFluxDivergence(
                0.0_rt, dz, axial_area, axial_area, 1.0_rt / volume);
        pass = pass && close(axial_constant_divergence, 0.0_rt, 1.0e-14_rt)
            && close(axial_gradient_divergence, 1.0_rt, 1.0e-13_rt);

        amrex::Real const interpolated_constant =
            warpx::hybrid::cylindricalRadialFaceVelocity(
                constant_velocity, constant_velocity, r_lo, r_hi,
                0.5_rt * (r_lo + r_hi));
        amrex::Real const interpolated_inverse_radius =
            warpx::hybrid::cylindricalRadialFaceVelocity(
                c / r_lo, c / r_hi, r_lo, r_hi, 0.5_rt * (r_lo + r_hi));
        pass = pass && close(interpolated_constant, constant_velocity)
            && close(
                interpolated_inverse_radius,
                c / (0.5_rt * (r_lo + r_hi)));

    }

    std::cout << "hybrid metric transport: "
              << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}
