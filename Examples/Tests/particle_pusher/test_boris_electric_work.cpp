/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Particles/Pusher/UpdateMomentumBoris.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>

using namespace amrex::literals;

namespace
{
    struct BorisTestInput
    {
        amrex::ParticleReal ux;
        amrex::ParticleReal uy;
        amrex::ParticleReal uz;
        amrex::ParticleReal Ex;
        amrex::ParticleReal Ey;
        amrex::ParticleReal Ez;
        amrex::ParticleReal Eax;
        amrex::ParticleReal Eay;
        amrex::ParticleReal Eaz;
        amrex::ParticleReal Bx;
        amrex::ParticleReal By;
        amrex::ParticleReal Bz;
        amrex::ParticleReal q;
        amrex::ParticleReal m;
        amrex::ParticleReal weight;
        amrex::Real dt;
    };

    struct BorisTestResult
    {
        amrex::ParticleReal baseline_ux;
        amrex::ParticleReal baseline_uy;
        amrex::ParticleReal baseline_uz;
        amrex::ParticleReal instrumented_ux;
        amrex::ParticleReal instrumented_uy;
        amrex::ParticleReal instrumented_uz;
        BorisElectricWorkVelocity work_velocity;
        amrex::ParticleReal magnetic_ux;
        amrex::ParticleReal magnetic_uy;
        amrex::ParticleReal magnetic_uz;
    };

    class DeterministicGenerator
    {
    public:
        double symmetric () noexcept
        {
            m_state = m_state * UINT64_C(6364136223846793005)
                + UINT64_C(1442695040888963407);
            std::uint64_t const mantissa = m_state >> 11;
            constexpr double inverse_53_bits =
                1.0 / static_cast<double>(UINT64_C(1) << 53);
            return 2.0 * static_cast<double>(mantissa) * inverse_53_bits - 1.0;
        }

    private:
        std::uint64_t m_state = UINT64_C(0x6a09e667f3bcc909);
    };

    template<typename T>
    bool same_bits (T const left, T const right)
    {
        using Bits = std::conditional_t<
            sizeof(T) == sizeof(float), std::uint32_t, std::uint64_t>;
        static_assert(sizeof(Bits) == sizeof(T));
        return std::bit_cast<Bits>(left) == std::bit_cast<Bits>(right);
    }

    long double gamma_from_u (
        amrex::ParticleReal const ux,
        amrex::ParticleReal const uy,
        amrex::ParticleReal const uz)
    {
        auto const c = static_cast<long double>(PhysConst::c);
        auto const x = static_cast<long double>(ux);
        auto const y = static_cast<long double>(uy);
        auto const z = static_cast<long double>(uz);
        return std::sqrt(1.0L + (x * x + y * y + z * z) / (c * c));
    }

    long double dot (
        amrex::ParticleReal const ax,
        amrex::ParticleReal const ay,
        amrex::ParticleReal const az,
        BorisElectricWorkVelocity const& b)
    {
        return static_cast<long double>(ax) * static_cast<long double>(b.x)
            + static_cast<long double>(ay) * static_cast<long double>(b.y)
            + static_cast<long double>(az) * static_cast<long double>(b.z);
    }

    bool close_work (
        long double const actual,
        long double const expected,
        long double const rest_energy)
    {
        auto const epsilon = static_cast<long double>(
            std::numeric_limits<amrex::ParticleReal>::epsilon());
        long double const scale = std::max({
            std::abs(actual), std::abs(expected), 0.01L * rest_energy});
        return std::abs(actual - expected) <= 4096.0L * epsilon * scale;
    }
}

int main (int argc, char* argv[])
{
    static_assert(std::is_trivially_copyable_v<BorisElectricWorkVelocity>);
    static_assert(std::is_trivially_copyable_v<BorisTestInput>);
    static_assert(std::is_trivially_copyable_v<BorisTestResult>);

    amrex::Initialize(argc, argv);
    bool pass = true;
    {
        constexpr int num_states = 128;
        std::array<BorisTestInput, num_states> host_inputs{};
        DeterministicGenerator random;
        auto const c = static_cast<amrex::ParticleReal>(PhysConst::c);

        for (int i = 0; i < num_states; ++i) {
            BorisTestInput& input = host_inputs[i];
            input.ux = static_cast<amrex::ParticleReal>(2.5 * random.symmetric()) * c;
            input.uy = static_cast<amrex::ParticleReal>(2.5 * random.symmetric()) * c;
            input.uz = static_cast<amrex::ParticleReal>(2.5 * random.symmetric()) * c;
            input.Ex = static_cast<amrex::ParticleReal>(0.7 * random.symmetric()) * c;
            input.Ey = static_cast<amrex::ParticleReal>(0.7 * random.symmetric()) * c;
            input.Ez = static_cast<amrex::ParticleReal>(0.7 * random.symmetric()) * c;
            input.Eax = 0.25_prt * input.Ex + 0.125_prt * input.Ey;
            input.Eay = -0.375_prt * input.Ex + 0.5_prt * input.Ey;
            input.Eaz = 0.625_prt * input.Ez;
            input.Bx = static_cast<amrex::ParticleReal>(8.0 * random.symmetric());
            input.By = static_cast<amrex::ParticleReal>(8.0 * random.symmetric());
            input.Bz = static_cast<amrex::ParticleReal>(8.0 * random.symmetric());
            input.q = static_cast<amrex::ParticleReal>(
                (i % 2 == 0 ? 1.0 : -1.0) * (0.5 + 0.5 * std::abs(random.symmetric())));
            input.m = static_cast<amrex::ParticleReal>(
                0.75 + 0.5 * std::abs(random.symmetric()));
            input.weight = static_cast<amrex::ParticleReal>(
                0.1 + 9.9 * std::abs(random.symmetric()));
            input.dt = static_cast<amrex::Real>(
                0.1 + 0.15 * std::abs(random.symmetric()));
        }

        amrex::Gpu::DeviceVector<BorisTestInput> device_inputs(num_states);
        amrex::Gpu::DeviceVector<BorisTestResult> device_results(num_states);
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice,
            host_inputs.begin(), host_inputs.end(), device_inputs.begin());

        BorisTestInput const* const inputs = device_inputs.data();
        BorisTestResult* const results = device_results.data();
        amrex::ParallelFor(num_states, [=] AMREX_GPU_DEVICE (int i) noexcept
        {
            BorisTestInput const input = inputs[i];
            BorisTestResult result{};

            result.baseline_ux = input.ux;
            result.baseline_uy = input.uy;
            result.baseline_uz = input.uz;
            UpdateMomentumBoris(
                result.baseline_ux, result.baseline_uy, result.baseline_uz,
                input.Ex, input.Ey, input.Ez,
                input.Bx, input.By, input.Bz,
                input.q, input.m, input.dt, MomentumPushType::Full);

            result.instrumented_ux = input.ux;
            result.instrumented_uy = input.uy;
            result.instrumented_uz = input.uz;
            result.work_velocity = UpdateMomentumBorisWithElectricWorkVelocity(
                result.instrumented_ux,
                result.instrumented_uy,
                result.instrumented_uz,
                input.Ex, input.Ey, input.Ez,
                input.Bx, input.By, input.Bz,
                input.q, input.m, input.dt, MomentumPushType::Full);

            result.magnetic_ux = input.ux;
            result.magnetic_uy = input.uy;
            result.magnetic_uz = input.uz;
            UpdateMomentumBorisWithElectricWorkVelocity(
                result.magnetic_ux, result.magnetic_uy, result.magnetic_uz,
                0.0_prt, 0.0_prt, 0.0_prt,
                input.Bx, input.By, input.Bz,
                input.q, input.m, input.dt, MomentumPushType::Full);

            results[i] = result;
        });

        std::array<BorisTestResult, num_states> host_results{};
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost,
            device_results.begin(), device_results.end(), host_results.begin());

        int trajectory_failures = 0;
        int work_failures = 0;
        int attribution_failures = 0;
        int selection_failures = 0;
        int magnetic_failures = 0;
        int positive_charge_states = 0;
        int negative_charge_states = 0;
        for (int i = 0; i < num_states; ++i) {
            BorisTestInput const& input = host_inputs[i];
            BorisTestResult const& result = host_results[i];
            bool const same_trajectory =
                same_bits(result.baseline_ux, result.instrumented_ux)
                && same_bits(result.baseline_uy, result.instrumented_uy)
                && same_bits(result.baseline_uz, result.instrumented_uz);
            trajectory_failures += same_trajectory ? 0 : 1;

            long double const gamma_initial = gamma_from_u(
                input.ux, input.uy, input.uz);
            long double const gamma_final = gamma_from_u(
                result.instrumented_ux,
                result.instrumented_uy,
                result.instrumented_uz);
            auto const mass = static_cast<long double>(input.m);
            auto const weight = static_cast<long double>(input.weight);
            auto const c_long = static_cast<long double>(PhysConst::c);
            long double const rest_energy = weight * mass * c_long * c_long;
            long double const kinetic_energy_change =
                rest_energy * (gamma_final - gamma_initial);
            long double const electric_work = weight
                * static_cast<long double>(input.q)
                * static_cast<long double>(input.dt)
                * dot(input.Ex, input.Ey, input.Ez, result.work_velocity);
            bool const work_matches = close_work(
                electric_work, kinetic_energy_change, rest_energy);
            work_failures += work_matches ? 0 : 1;

            amrex::ParticleReal const Ebx = input.Ex - input.Eax;
            amrex::ParticleReal const Eby = input.Ey - input.Eay;
            amrex::ParticleReal const Ebz = input.Ez - input.Eaz;
            long double const prefactor = weight
                * static_cast<long double>(input.q)
                * static_cast<long double>(input.dt);
            long double const attributed_work = prefactor
                * (dot(input.Eax, input.Eay, input.Eaz, result.work_velocity)
                    + dot(Ebx, Eby, Ebz, result.work_velocity));
            bool const attribution_matches = close_work(
                attributed_work, electric_work, rest_energy);
            attribution_failures += attribution_matches ? 0 : 1;

            // The selected pressure component can be absent or equal to the
            // entire E field. Both limiting attributions must be exact.
            long double const zero_selected_work = prefactor
                * dot(0.0_prt, 0.0_prt, 0.0_prt, result.work_velocity);
            long double const full_selected_work = prefactor
                * dot(input.Ex, input.Ey, input.Ez, result.work_velocity);
            bool const selection_matches = zero_selected_work == 0.0L
                && close_work(full_selected_work, electric_work, rest_energy);
            selection_failures += selection_matches ? 0 : 1;
            positive_charge_states += input.q > 0.0_prt ? 1 : 0;
            negative_charge_states += input.q < 0.0_prt ? 1 : 0;

            long double const magnetic_gamma = gamma_from_u(
                result.magnetic_ux, result.magnetic_uy, result.magnetic_uz);
            auto const particle_epsilon = static_cast<long double>(
                std::numeric_limits<amrex::ParticleReal>::epsilon());
            bool const magnetic_preserves_gamma =
                std::abs(magnetic_gamma - gamma_initial)
                <= 512.0L * particle_epsilon
                    * std::max(gamma_initial, 1.0L);
            magnetic_failures += magnetic_preserves_gamma ? 0 : 1;
        }

        pass = pass && trajectory_failures == 0;
        pass = pass && work_failures == 0;
        pass = pass && attribution_failures == 0;
        pass = pass && selection_failures == 0;
        pass = pass && magnetic_failures == 0;
        pass = pass && positive_charge_states > 0;
        pass = pass && negative_charge_states > 0;

        // Negative control: for a nonrelativistic particle accelerated from
        // rest with B=0, endpoint-current work is nearly twice the actual
        // kinetic-energy gain. The Boris work velocity must give the gain.
        BorisTestInput rest_input{};
        rest_input.Ex = 0.01_prt * c;
        rest_input.Ey = -0.006_prt * c;
        rest_input.Ez = 0.004_prt * c;
        rest_input.q = 1.0_prt;
        rest_input.m = 1.0_prt;
        rest_input.weight = 3.25_prt;
        rest_input.dt = 1.0_rt;
        amrex::Gpu::DeviceVector<BorisTestInput> rest_device_input(1);
        amrex::Gpu::DeviceVector<BorisTestResult> rest_device_result(1);
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice,
            &rest_input, &rest_input + 1, rest_device_input.begin());
        BorisTestInput const* const rest_input_ptr = rest_device_input.data();
        BorisTestResult* const rest_result_ptr = rest_device_result.data();
        amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) noexcept
        {
            BorisTestInput const input = rest_input_ptr[0];
            BorisTestResult result{};
            result.instrumented_ux = input.ux;
            result.instrumented_uy = input.uy;
            result.instrumented_uz = input.uz;
            result.work_velocity = UpdateMomentumBorisWithElectricWorkVelocity(
                result.instrumented_ux,
                result.instrumented_uy,
                result.instrumented_uz,
                input.Ex, input.Ey, input.Ez,
                0.0_prt, 0.0_prt, 0.0_prt,
                input.q, input.m, input.dt, MomentumPushType::Full);
            rest_result_ptr[0] = result;
        });
        BorisTestResult rest_result{};
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost,
            rest_device_result.begin(), rest_device_result.end(), &rest_result);

        long double const rest_gamma = gamma_from_u(
            rest_result.instrumented_ux,
            rest_result.instrumented_uy,
            rest_result.instrumented_uz);
        auto const c_long = static_cast<long double>(PhysConst::c);
        auto const rest_weight =
            static_cast<long double>(rest_input.weight);
        long double const kinetic_energy_gain = rest_weight * c_long * c_long
            * (rest_gamma - 1.0L);
        long double const ledger_work = rest_weight
            * static_cast<long double>(rest_input.q)
            * static_cast<long double>(rest_input.dt)
            * dot(
                rest_input.Ex, rest_input.Ey, rest_input.Ez,
                rest_result.work_velocity);
        BorisElectricWorkVelocity const endpoint_velocity{
            static_cast<amrex::ParticleReal>(
                static_cast<long double>(rest_result.instrumented_ux)
                / rest_gamma),
            static_cast<amrex::ParticleReal>(
                static_cast<long double>(rest_result.instrumented_uy)
                / rest_gamma),
            static_cast<amrex::ParticleReal>(
                static_cast<long double>(rest_result.instrumented_uz)
                / rest_gamma)};
        long double const endpoint_work = rest_weight
            * static_cast<long double>(rest_input.q)
            * static_cast<long double>(rest_input.dt)
            * dot(
                rest_input.Ex, rest_input.Ey, rest_input.Ez,
                endpoint_velocity);
        long double const endpoint_ratio = endpoint_work / kinetic_energy_gain;
        long double const analytic_ratio = (rest_gamma + 1.0L) / rest_gamma;
        bool const negative_control_pass = close_work(
                ledger_work, kinetic_energy_gain,
                rest_weight * c_long * c_long)
            && std::abs(endpoint_ratio - analytic_ratio)
                <= 512.0L
                    * static_cast<long double>(
                        std::numeric_limits<amrex::ParticleReal>::epsilon())
                    * analytic_ratio
            && endpoint_ratio > 1.99L;
        pass = pass && negative_control_pass;

        std::cout
            << "Boris trajectory bit failures: " << trajectory_failures << '\n'
            << "Boris electric-work failures: " << work_failures << '\n'
            << "Additive-attribution failures: " << attribution_failures << '\n'
            << "Zero/full-selection failures: " << selection_failures << '\n'
            << "Charge signs exercised: +" << positive_charge_states
            << " / -" << negative_charge_states << '\n'
            << "Magnetic-gamma failures: " << magnetic_failures << '\n'
            << "Endpoint-current/kinetic-work ratio: "
            << static_cast<double>(endpoint_ratio)
            << (negative_control_pass ? " PASS\n" : " FAIL\n");
    }
    amrex::Finalize();

    std::cout << (pass ? "PASS\n" : "FAIL\n");
    return pass ? 0 : 1;
}
