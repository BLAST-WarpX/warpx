/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Radiation/MaterialOpacityTable.H"
#include "Radiation/MaterialOpacityTableResponse.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace {

void
TestAssert (bool condition, std::string const& message) {
    if (!condition) {
        amrex::Abort("TEST FAILURE: " + message);
    }
}

bool
CheckNear (
    amrex::Real actual, amrex::Real expected,
    amrex::Real rel_tol = (std::is_same_v<amrex::Real, float> ? 1e-5 : 1e-11),
    amrex::Real abs_tol = (std::is_same_v<amrex::Real, float> ? 1e-6 : 1e-14)) {
    amrex::Real const diff = std::abs(actual - expected);
    amrex::Real const scale = std::max(std::abs(actual), std::abs(expected));
    return (diff <= abs_tol) || (diff <= rel_tol * scale);
}

struct ExpectedResponse {
    amrex::Real transport_opacity = 0;
    amrex::Real absorption_rate = 0;
    amrex::Real emission_rate_density = 0;
    amrex::Real scattering_opacity = 0;
};

ExpectedResponse
ComputeExpected (amrex::Real rho, amrex::Real T, int g, amrex::Real fraction,
                 amrex::Real rho0, amrex::Real rho1, amrex::Real T0,
                 amrex::Real T1) {
    amrex::Real const u =
        (std::log(rho) - std::log(rho0)) / (std::log(rho1) - std::log(rho0));
    amrex::Real const v =
        (std::log(T) - std::log(T0)) / (std::log(T1) - std::log(T0));

    amrex::Real const log_kp = -2.0 + 0.4 * u - 0.2 * v + 0.5 * g;
    amrex::Real const log_kr = -1.0 + 0.2 * u - 0.4 * v + 0.3 * g;
    amrex::Real const log_ks = -3.0 + 0.1 * u + 0.1 * v + 0.2 * g;

    amrex::Real const mass_kp = std::exp(log_kp);
    amrex::Real const mass_kr = std::exp(log_kr);
    amrex::Real const mass_ks = std::exp(log_ks);

    amrex::Real const kappa_p = rho * mass_kp;
    amrex::Real const kappa_r = rho * mass_kr;
    amrex::Real const kappa_s = rho * mass_ks;

    constexpr amrex::Real radiation_constant_a = 7.565733250280007e-16;
    amrex::Real const c = PhysConst::c;
    amrex::Real const T2 = T * T;
    amrex::Real const T4 = T2 * T2;

    ExpectedResponse exp;
    exp.transport_opacity = kappa_r;
    exp.absorption_rate = c * kappa_p;
    exp.emission_rate_density =
        c * kappa_p * radiation_constant_a * T4 * fraction;
    exp.scattering_opacity = kappa_s;
    return exp;
}

void
TestEndpointsAndInterior () {
    amrex::Real const rho0 = 1.0;
    amrex::Real const rho1 = 100.0;
    amrex::Real const T0 = 1000.0;
    amrex::Real const T1 = 10000.0;

    std::vector<amrex::Real> log_density = {std::log(rho0), std::log(rho1)};
    std::vector<amrex::Real> log_temperature = {std::log(T0), std::log(T1)};

    int const num_d = 2;
    int const num_T = 2;
    int const num_g = 2;
    int const total = num_d * num_T * num_g;

    std::vector<amrex::Real> log_p_abs(total);
    std::vector<amrex::Real> log_p_em(total);
    std::vector<amrex::Real> log_ross(total);
    std::vector<amrex::Real> log_scat(total);

    for (int i = 0; i < num_d; ++i) {
        for (int j = 0; j < num_T; ++j) {
            for (int g = 0; g < num_g; ++g) {
                int const idx = (i * num_T + j) * num_g + g;
                amrex::Real const u = static_cast<amrex::Real>(i);
                amrex::Real const v = static_cast<amrex::Real>(j);

                log_p_abs[idx] = -2.0 + 0.4 * u - 0.2 * v + 0.5 * g;
                log_p_em[idx] = log_p_abs[idx]; // LTE Kirchhoff balance
                log_ross[idx] = -1.0 + 0.2 * u - 0.4 * v + 0.3 * g;
                log_scat[idx] = -3.0 + 0.1 * u + 0.1 * v + 0.2 * g;
            }
        }
    }

    warpx::radiation::MaterialOpacityTableExecutor table;
    table.m_num_density = num_d;
    table.m_num_temperature = num_T;
    table.m_num_groups = num_g;
    table.m_log_density = log_density.data();
    table.m_log_temperature = log_temperature.data();
    table.m_log_planck_absorption = log_p_abs.data();
    table.m_log_planck_emission = log_p_em.data();
    table.m_log_rosseland_transport = log_ross.data();
    table.m_log_scattering = log_scat.data();

    // 1. Check all 4 table corners (endpoints) for both groups
    amrex::Real const corner_rhos[4] = {rho0, rho1, rho0, rho1};
    amrex::Real const corner_Ts[4] = {T0, T0, T1, T1};

    for (int k = 0; k < 4; ++k) {
        for (int g = 0; g < num_g; ++g) {
            amrex::Real const f = 0.5;
            auto const res =
                warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table, corner_rhos[k], corner_Ts[k], g, f);
            TestAssert(res.valid, "Endpoint response must be valid");

            auto const exp = ComputeExpected(corner_rhos[k], corner_Ts[k], g, f,
                                             rho0, rho1, T0, T1);
            TestAssert(CheckNear(res.transport_opacity, exp.transport_opacity),
                       "Transport opacity mismatch at endpoint");
            TestAssert(CheckNear(res.absorption_rate, exp.absorption_rate),
                       "Absorption rate mismatch at endpoint");
            TestAssert(
                CheckNear(res.emission_rate_density, exp.emission_rate_density),
                "Emission rate density mismatch at endpoint");
            TestAssert(
                CheckNear(res.scattering_opacity, exp.scattering_opacity),
                "Scattering opacity mismatch at endpoint");
        }
    }

    // 2. Check interior points
    amrex::Real const rho_mid = std::sqrt(rho0 * rho1); // u = 0.5
    amrex::Real const T_mid = std::sqrt(T0 * T1);       // v = 0.5

    for (int g = 0; g < num_g; ++g) {
        amrex::Real const f = 0.75;
        auto const res = warpx::radiation::EvaluateMaterialOpacityTableResponse(
            table, rho_mid, T_mid, g, f);
        TestAssert(res.valid, "Interior midpoint response must be valid");

        auto const exp =
            ComputeExpected(rho_mid, T_mid, g, f, rho0, rho1, T0, T1);
        TestAssert(CheckNear(res.transport_opacity, exp.transport_opacity),
                   "Transport opacity mismatch at interior");
        TestAssert(CheckNear(res.absorption_rate, exp.absorption_rate),
                   "Absorption rate mismatch at interior");
        TestAssert(
            CheckNear(res.emission_rate_density, exp.emission_rate_density),
            "Emission rate density mismatch at interior");
        TestAssert(CheckNear(res.scattering_opacity, exp.scattering_opacity),
                   "Scattering opacity mismatch at interior");
    }

    // 3. Check group fraction scaling and equilibrium relation
    amrex::Real const rho_test = 10.0;
    amrex::Real const T_test = 3000.0;
    int const g_test = 0;

    auto const res_f0 = warpx::radiation::EvaluateMaterialOpacityTableResponse(
        table, rho_test, T_test, g_test, 0.0);
    TestAssert(res_f0.valid, "f=0 response must be valid");
    TestAssert(res_f0.emission_rate_density == 0.0, "f=0 emission must be 0");

    auto const res_f1 = warpx::radiation::EvaluateMaterialOpacityTableResponse(
        table, rho_test, T_test, g_test, 1.0);
    TestAssert(res_f1.valid, "f=1 response must be valid");

    auto const res_fhalf =
        warpx::radiation::EvaluateMaterialOpacityTableResponse(
            table, rho_test, T_test, g_test, 0.5);
    TestAssert(res_fhalf.valid, "f=0.5 response must be valid");
    TestAssert(CheckNear(res_fhalf.emission_rate_density,
                         amrex::Real(0.5) * res_f1.emission_rate_density),
               "Emission must scale linearly with fraction");

    // Kirchhoff equilibrium relation: emission = absorption_rate * a * T^4 when
    // f = 1
    constexpr amrex::Real a = 7.565733250280007e-16;
    amrex::Real const U_rad = a * (T_test * T_test) * (T_test * T_test);
    TestAssert(
        CheckNear(res_f1.emission_rate_density, res_f1.absorption_rate * U_rad),
        "Kirchhoff equilibrium relation violated");

    // Direct SI channel conversion check
    auto const direct_coeffs = table(rho_test, T_test, g_test);
    TestAssert(CheckNear(res_f1.absorption_rate,
                         PhysConst::c * direct_coeffs.planck_absorption),
               "Absorption SI conversion mismatch");
    TestAssert(
        CheckNear(res_f1.transport_opacity, direct_coeffs.rosseland_transport),
        "Transport opacity direct channel mismatch");
    TestAssert(CheckNear(res_f1.scattering_opacity, direct_coeffs.scattering),
               "Scattering opacity direct channel mismatch");

    // 4. Out-of-table queries
    auto const res_low_rho =
        warpx::radiation::EvaluateMaterialOpacityTableResponse(
            table, 0.5 * rho0, T_test, g_test, 1.0);
    TestAssert(!res_low_rho.valid, "Out-of-table low density must be invalid");
    TestAssert(res_low_rho.transport_opacity == 0.0,
               "Invalid output must be zero");

    auto const res_high_rho =
        warpx::radiation::EvaluateMaterialOpacityTableResponse(
            table, 2.0 * rho1, T_test, g_test, 1.0);
    TestAssert(!res_high_rho.valid,
               "Out-of-table high density must be invalid");

    auto const res_low_T =
        warpx::radiation::EvaluateMaterialOpacityTableResponse(
            table, rho_test, 0.5 * T0, g_test, 1.0);
    TestAssert(!res_low_T.valid,
               "Out-of-table low temperature must be invalid");

    auto const res_high_T =
        warpx::radiation::EvaluateMaterialOpacityTableResponse(
            table, rho_test, 2.0 * T1, g_test, 1.0);
    TestAssert(!res_high_T.valid,
               "Out-of-table high temperature must be invalid");

    // 5. Invalid inputs
    auto const nan_val = std::numeric_limits<amrex::Real>::quiet_NaN();
    auto const inf_val = std::numeric_limits<amrex::Real>::infinity();

    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table, -1.0, T_test, g_test, 1.0)
                    .valid,
               "Negative density rejected");
    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table, nan_val, T_test, g_test, 1.0)
                    .valid,
               "NaN density rejected");
    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table, inf_val, T_test, g_test, 1.0)
                    .valid,
               "Inf density rejected");

    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table, rho_test, -10.0, g_test, 1.0)
                    .valid,
               "Negative T rejected");
    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table, rho_test, 0.0, g_test, 1.0)
                    .valid,
               "Zero T nonvacuum rejected");
    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table, rho_test, nan_val, g_test, 1.0)
                    .valid,
               "NaN T rejected");
    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table, rho_test, inf_val, g_test, 1.0)
                    .valid,
               "Inf T rejected");

    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table, rho_test, T_test, g_test, -0.01)
                    .valid,
               "Negative fraction rejected");
    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table, rho_test, T_test, g_test, 1.01)
                    .valid,
               "Fraction > 1 rejected");
    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table, rho_test, T_test, g_test, nan_val)
                    .valid,
               "NaN fraction rejected");
    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table, rho_test, T_test, g_test, inf_val)
                    .valid,
               "Inf fraction rejected");

    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table, rho_test, T_test, -1, 1.0)
                    .valid,
               "Negative group rejected");
    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table, rho_test, T_test, num_g, 1.0)
                    .valid,
               "Group index >= num_groups rejected");

    // 6. Uninitialized and partially initialized executor validation
    warpx::radiation::MaterialOpacityTableExecutor empty_table;
    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    empty_table, rho_test, T_test, 0, 1.0)
                    .valid,
               "Uninitialized table rejected");
    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    empty_table, 0.0, 0.0, 0, 1.0)
                    .valid,
               "Uninitialized table vacuum rejected");

    auto table_bad_ptr = table;
    table_bad_ptr.m_log_scattering = nullptr;
    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table_bad_ptr, rho_test, T_test, 0, 1.0)
                    .valid,
               "Null pointer rejected");

    auto table_zero_groups = table;
    table_zero_groups.m_num_groups = 0;
    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table_zero_groups, rho_test, T_test, 0, 1.0)
                    .valid,
               "Zero groups rejected");

    // 7. Valid vacuum queries
    auto const vac1 = warpx::radiation::EvaluateMaterialOpacityTableResponse(
        table, 0.0, 0.0, 0, 1.0);
    TestAssert(vac1.valid, "Vacuum with T=0 must be valid");
    TestAssert(vac1.transport_opacity == 0.0 && vac1.absorption_rate == 0.0 &&
                   vac1.emission_rate_density == 0.0 &&
                   vac1.scattering_opacity == 0.0,
               "Vacuum outputs must be zero");

    auto const vac2 = warpx::radiation::EvaluateMaterialOpacityTableResponse(
        table, 0.0, 5000.0, 1, 0.5);
    TestAssert(vac2.valid, "Vacuum with positive T must be valid");
    TestAssert(vac2.transport_opacity == 0.0 && vac2.absorption_rate == 0.0 &&
                   vac2.emission_rate_density == 0.0 &&
                   vac2.scattering_opacity == 0.0,
               "Vacuum outputs must be zero");

    // Vacuum with invalid inputs must still return valid = false
    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table, 0.0, -1.0, 0, 1.0)
                    .valid,
               "Vacuum negative T rejected");
    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table, 0.0, 5000.0, 0, 1.5)
                    .valid,
               "Vacuum invalid fraction rejected");
    TestAssert(!warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    table, 0.0, 5000.0, -1, 1.0)
                    .valid,
               "Vacuum invalid group rejected");
}

void
TestOverflowAndNonfinite () {
    // Build a finite singleton table that causes output overflow
    amrex::Real const log_rho = std::log(1.0);
    amrex::Real const log_T = std::log(1000.0);
    amrex::Real const large_log_val =
        705.0; // exp(705) > 1e306, c * exp(705) > 1e314 -> overflow
    amrex::Real const zero_val = 0.0;

    warpx::radiation::MaterialOpacityTableExecutor overflow_table;
    overflow_table.m_num_density = 1;
    overflow_table.m_num_temperature = 1;
    overflow_table.m_num_groups = 1;
    overflow_table.m_log_density = &log_rho;
    overflow_table.m_log_temperature = &log_T;
    overflow_table.m_log_planck_absorption = &large_log_val;
    overflow_table.m_log_planck_emission = &large_log_val;
    overflow_table.m_log_rosseland_transport = &zero_val;
    overflow_table.m_log_scattering = &zero_val;

    auto const of_res = warpx::radiation::EvaluateMaterialOpacityTableResponse(
        overflow_table, 1.0, 1000.0, 0, 1.0);
    TestAssert(!of_res.valid, "Overflow response must be marked invalid");
    TestAssert(of_res.absorption_rate == 0.0,
               "Invalid overflow absorption must stay zero");
    TestAssert(of_res.emission_rate_density == 0.0,
               "Invalid overflow emission must stay zero");

    // Test nonfinite table data
    amrex::Real const nan_val = std::numeric_limits<amrex::Real>::quiet_NaN();
    warpx::radiation::MaterialOpacityTableExecutor nan_table = overflow_table;
    nan_table.m_log_planck_absorption = &nan_val;

    auto const nan_res = warpx::radiation::EvaluateMaterialOpacityTableResponse(
        nan_table, 1.0, 1000.0, 0, 1.0);
    TestAssert(!nan_res.valid,
               "Nonfinite table coefficient must yield valid = false");
}

void
TestDeviceExecution () {
    amrex::Real const rho0 = 1.0;
    amrex::Real const rho1 = 100.0;
    amrex::Real const T0 = 1000.0;
    amrex::Real const T1 = 10000.0;

    int const num_d = 2;
    int const num_T = 2;
    int const num_g = 2;
    int const total = num_d * num_T * num_g;

    std::vector<amrex::Real> h_log_d = {std::log(rho0), std::log(rho1)};
    std::vector<amrex::Real> h_log_T = {std::log(T0), std::log(T1)};
    std::vector<amrex::Real> h_log_p_abs(total);
    std::vector<amrex::Real> h_log_p_em(total);
    std::vector<amrex::Real> h_log_ross(total);
    std::vector<amrex::Real> h_log_scat(total);

    for (int i = 0; i < num_d; ++i) {
        for (int j = 0; j < num_T; ++j) {
            for (int g = 0; g < num_g; ++g) {
                int const idx = (i * num_T + j) * num_g + g;
                amrex::Real const u = static_cast<amrex::Real>(i);
                amrex::Real const v = static_cast<amrex::Real>(j);
                h_log_p_abs[idx] = -2.0 + 0.4 * u - 0.2 * v + 0.5 * g;
                h_log_p_em[idx] = h_log_p_abs[idx];
                h_log_ross[idx] = -1.0 + 0.2 * u - 0.4 * v + 0.3 * g;
                h_log_scat[idx] = -3.0 + 0.1 * u + 0.1 * v + 0.2 * g;
            }
        }
    }

    // Copy table arrays to device vectors
    amrex::Gpu::DeviceVector<amrex::Real> d_log_d(2);
    amrex::Gpu::DeviceVector<amrex::Real> d_log_T(2);
    amrex::Gpu::DeviceVector<amrex::Real> d_log_p_abs(total);
    amrex::Gpu::DeviceVector<amrex::Real> d_log_p_em(total);
    amrex::Gpu::DeviceVector<amrex::Real> d_log_ross(total);
    amrex::Gpu::DeviceVector<amrex::Real> d_log_scat(total);

    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_log_d.begin(), h_log_d.end(),
                     d_log_d.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_log_T.begin(), h_log_T.end(),
                     d_log_T.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_log_p_abs.begin(),
                     h_log_p_abs.end(), d_log_p_abs.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_log_p_em.begin(),
                     h_log_p_em.end(), d_log_p_em.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_log_ross.begin(),
                     h_log_ross.end(), d_log_ross.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_log_scat.begin(),
                     h_log_scat.end(), d_log_scat.begin());

    warpx::radiation::MaterialOpacityTableExecutor d_table;
    d_table.m_num_density = num_d;
    d_table.m_num_temperature = num_T;
    d_table.m_num_groups = num_g;
    d_table.m_log_density = d_log_d.data();
    d_table.m_log_temperature = d_log_T.data();
    d_table.m_log_planck_absorption = d_log_p_abs.data();
    d_table.m_log_planck_emission = d_log_p_em.data();
    d_table.m_log_rosseland_transport = d_log_ross.data();
    d_table.m_log_scattering = d_log_scat.data();

    int const n_probes = 4;
    amrex::Gpu::DeviceVector<warpx::radiation::MaterialOpacityTableResponse>
        d_responses(n_probes);
    auto* const resp_ptr = d_responses.data();

    amrex::ParallelFor(n_probes, [=] AMREX_GPU_DEVICE(int idx) noexcept {
        if (idx == 0) {
            // Interior point
            resp_ptr[idx] =
                warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    d_table, 10.0, 3162.277660168379, 0, 0.75);
        } else if (idx == 1) {
            // Endpoint
            resp_ptr[idx] =
                warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    d_table, 1.0, 1000.0, 1, 1.0);
        } else if (idx == 2) {
            // Valid vacuum
            resp_ptr[idx] =
                warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    d_table, 0.0, 3000.0, 0, 0.5);
        } else {
            // Out of table
            resp_ptr[idx] =
                warpx::radiation::EvaluateMaterialOpacityTableResponse(
                    d_table, 200.0, 3000.0, 0, 1.0);
        }
    });
    amrex::Gpu::streamSynchronize();

    std::vector<warpx::radiation::MaterialOpacityTableResponse> h_responses(
        n_probes);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_responses.begin(),
                     d_responses.end(), h_responses.begin());

    // Verify device probe results on host
    TestAssert(h_responses[0].valid, "Device interior probe must be valid");
    auto const exp0 =
        ComputeExpected(10.0, 3162.277660168379, 0, 0.75, rho0, rho1, T0, T1);
    TestAssert(
        CheckNear(h_responses[0].transport_opacity, exp0.transport_opacity),
        "Device interior transport opacity mismatch");
    TestAssert(CheckNear(h_responses[0].absorption_rate, exp0.absorption_rate),
               "Device interior absorption rate mismatch");

    TestAssert(h_responses[1].valid, "Device endpoint probe must be valid");
    auto const exp1 = ComputeExpected(1.0, 1000.0, 1, 1.0, rho0, rho1, T0, T1);
    TestAssert(
        CheckNear(h_responses[1].transport_opacity, exp1.transport_opacity),
        "Device endpoint transport opacity mismatch");

    TestAssert(h_responses[2].valid, "Device vacuum probe must be valid");
    TestAssert(h_responses[2].transport_opacity == 0.0 &&
                   h_responses[2].absorption_rate == 0.0 &&
                   h_responses[2].emission_rate_density == 0.0 &&
                   h_responses[2].scattering_opacity == 0.0,
               "Device vacuum outputs must be zero");

    TestAssert(!h_responses[3].valid,
               "Device out-of-table probe must be invalid");
}

} // namespace

int
main (int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    {
        amrex::Print()
            << "[TEST] Running MaterialOpacityTableResponse tests...\n";

        TestEndpointsAndInterior();
        amrex::Print()
            << "  PASSED: Manufactured table endpoints, interior, fractions, "
            << "and Kirchhoff equilibrium.\n";

        TestOverflowAndNonfinite();
        amrex::Print() << "  PASSED: Overflow and nonfinite input detection.\n";

        TestDeviceExecution();
        amrex::Print() << "  PASSED: Device execution with device-resident "
                          "table arrays.\n";

        amrex::Print() << "[TEST] All MaterialOpacityTableResponse tests "
                          "passed successfully!\n";
    }
    amrex::Finalize();
    return 0;
}
