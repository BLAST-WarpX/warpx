/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_RADIATION_MATERIAL_OPACITY_TABLE_RESPONSE_H_
#define WARPX_RADIATION_MATERIAL_OPACITY_TABLE_RESPONSE_H_

#include "MaterialOpacityTable.H"
#include "Utils/WarpXConst.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <type_traits>

namespace warpx::radiation {

struct MaterialOpacityTableResponse {
    amrex::Real transport_opacity = 0;     // 1/m
    amrex::Real absorption_rate = 0;       // 1/s
    amrex::Real emission_rate_density = 0; // W/m^3
    amrex::Real scattering_opacity = 0;    // 1/m
    bool valid = false;
};

static_assert(std::is_trivially_copyable_v<MaterialOpacityTableResponse>,
              "MaterialOpacityTableResponse must be trivially copyable");

/**
 * \brief Evaluate physical material opacity responses from an opacity table
 * executor.
 *
 * Validates executor axis counts, interpolation mode, table/mask pointers,
 * group index range, mass density, temperature, and group Planck fraction.
 * Valid vacuum returns zero coefficients with valid = true. Nonvacuum inputs
 * evaluate the table interpolator, convert channel coefficients to SI units,
 * check for finite non-negative values and output overflow, and return the
 * response struct with valid set accordingly. Exact zeros from the explicit
 * schema-1.1 mask pass unchanged and are never floored.
 *
 * \param table Opacity table executor
 * \param mass_density Mass density (kg/m^3)
 * \param temperature Temperature (K)
 * \param group Discrete radiation group index
 * \param planck_fraction Normalized group Planck fraction in [0, 1]
 * \return MaterialOpacityTableResponse
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE MaterialOpacityTableResponse
EvaluateMaterialOpacityTableResponse (MaterialOpacityTableExecutor const& table,
                                      amrex::Real mass_density,
                                      amrex::Real temperature, int group,
                                      amrex::Real planck_fraction) noexcept {
    MaterialOpacityTableResponse response{};

    // 1. Validate positive axis and group counts
    if (table.m_num_density <= 0 || table.m_num_temperature <= 0 ||
        table.m_num_groups <= 0) {
        return response;
    }

    // 2. Validate all six required table pointers before evaluation
    if (table.m_log_density == nullptr || table.m_log_temperature == nullptr ||
        table.m_log_planck_absorption == nullptr ||
        table.m_log_planck_emission == nullptr ||
        table.m_log_rosseland_transport == nullptr ||
        table.m_log_scattering == nullptr) {
        return response;
    }
    if (table.m_interpolation_mode !=
            static_cast<int>(
                MaterialOpacityInterpolation::PositiveLog) &&
        table.m_interpolation_mode !=
            static_cast<int>(
                MaterialOpacityInterpolation::ExactZeroLinear)) {
        return response;
    }
    if (table.m_interpolation_mode ==
            static_cast<int>(
                MaterialOpacityInterpolation::ExactZeroLinear) &&
        table.m_zero_mask == nullptr) {
        return response;
    }

    // 3. Validate discrete group index
    if (group < 0 || group >= table.m_num_groups) {
        return response;
    }

    // 4. Validate density: finite and >= 0
    if (!std::isfinite(mass_density) || mass_density < amrex::Real(0.0)) {
        return response;
    }

    // 5. Validate temperature: finite, T >= 0 (T > 0 for nonvacuum)
    if (!std::isfinite(temperature) || temperature < amrex::Real(0.0)) {
        return response;
    }
    if (mass_density > amrex::Real(0.0) && temperature <= amrex::Real(0.0)) {
        return response;
    }

    // 6. Validate group Planck fraction: finite and in [0, 1]
    if (!std::isfinite(planck_fraction) || planck_fraction < amrex::Real(0.0) ||
        planck_fraction > amrex::Real(1.0)) {
        return response;
    }

    // 7. Valid vacuum: zero density returns zero coefficients with valid = true
    if (mass_density == amrex::Real(0.0)) {
        response.transport_opacity = amrex::Real(0.0);
        response.absorption_rate = amrex::Real(0.0);
        response.emission_rate_density = amrex::Real(0.0);
        response.scattering_opacity = amrex::Real(0.0);
        response.valid = true;
        return response;
    }

    // 8. Evaluate table for nonvacuum input
    MaterialOpacityCoefficients const coeffs =
        table(mass_density, temperature, group);

    // 9. Validate interpolated coefficients: must be finite and non-negative
    if (!std::isfinite(coeffs.planck_absorption) ||
        coeffs.planck_absorption < amrex::Real(0.0) ||
        !std::isfinite(coeffs.planck_emission) ||
        coeffs.planck_emission < amrex::Real(0.0) ||
        !std::isfinite(coeffs.rosseland_transport) ||
        coeffs.rosseland_transport < amrex::Real(0.0) ||
        !std::isfinite(coeffs.scattering) ||
        coeffs.scattering < amrex::Real(0.0)) {
        return response;
    }

    // 10. Compute physical channel responses
    constexpr amrex::Real radiation_constant_a =
        7.565733250280007e-16; // J/(m^3 K^4)
    amrex::Real const c = PhysConst::c;

    amrex::Real const transport_opacity = coeffs.rosseland_transport;
    amrex::Real const absorption_rate = c * coeffs.planck_absorption;

    amrex::Real const T2 = temperature * temperature;
    amrex::Real const T4 = T2 * T2;
    amrex::Real const emission_rate_density = c * coeffs.planck_emission *
                                              radiation_constant_a * T4 *
                                              planck_fraction;

    amrex::Real const scattering_opacity = coeffs.scattering;

    // 11. Check all outputs are finite and non-negative (detects overflow)
    if (!std::isfinite(transport_opacity) ||
        transport_opacity < amrex::Real(0.0) ||
        !std::isfinite(absorption_rate) || absorption_rate < amrex::Real(0.0) ||
        !std::isfinite(emission_rate_density) ||
        emission_rate_density < amrex::Real(0.0) ||
        !std::isfinite(scattering_opacity) ||
        scattering_opacity < amrex::Real(0.0)) {
        return response;
    }

    response.transport_opacity = transport_opacity;
    response.absorption_rate = absorption_rate;
    response.emission_rate_density = emission_rate_density;
    response.scattering_opacity = scattering_opacity;
    response.valid = true;
    return response;
}

} // namespace warpx::radiation

#endif // WARPX_RADIATION_MATERIAL_OPACITY_TABLE_RESPONSE_H_
