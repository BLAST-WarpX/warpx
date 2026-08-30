/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Radiation/RadiationKineticEnergyUpdate.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

using namespace amrex::literals;
using warpx::radiation::EvaluateKineticEnergyUpdate;
using warpx::radiation::KineticEnergyUpdateResult;
using warpx::radiation::KineticPrecisionEpsilon;

namespace
{
    enum class CaseKind : int
    {
        Zero,
        Normal,
        BelowParticleUlp,
        AcceptedRepresentationMismatch,
        Nonfinite,
        ExcessiveMismatch
    };

    struct TestInput
    {
        amrex::Real requested;
        amrex::Real old_energy;
        amrex::Real target_energy;
        amrex::Real achieved_energy;
        CaseKind kind;
    };

    struct TestOutput
    {
        KineticEnergyUpdateResult result;
    };

    bool SameValue (amrex::Real const left, amrex::Real const right)
    {
        if (amrex::Math::isfinite(left) && amrex::Math::isfinite(right)) {
            return left == right;
        }
        if (std::isnan(left) || std::isnan(right)) {
            return std::isnan(left) && std::isnan(right);
        }
        return std::isinf(left) && std::isinf(right)
            && std::signbit(left) == std::signbit(right);
    }

    bool CheckResult (
        std::string const& label,
        TestInput const& input,
        KineticEnergyUpdateResult const& result)
    {
        bool pass = true;
        switch (input.kind) {
        case CaseKind::Zero:
            pass = result.valid
                && result.realized_energy_change == 0.0_rt
                && result.residual == 0.0_rt;
            break;
        case CaseKind::Normal:
            pass = result.valid
                && result.realized_energy_change == input.requested
                && result.residual == 0.0_rt;
            break;
        case CaseKind::BelowParticleUlp:
            pass = result.valid
                && result.realized_energy_change == 0.0_rt
                && result.residual == input.requested
                && result.residual < 0.0_rt;
            break;
        case CaseKind::AcceptedRepresentationMismatch:
            pass = result.valid
                && result.realized_energy_change
                    == input.achieved_energy - input.old_energy
                && result.residual
                    == input.requested - result.realized_energy_change
                && result.residual != 0.0_rt;
            break;
        case CaseKind::Nonfinite:
        case CaseKind::ExcessiveMismatch:
            pass = !result.valid
                && SameValue(result.realized_energy_change, 0.0_rt)
                && SameValue(result.residual, 0.0_rt);
            break;
        }
        if (!pass) {
            std::cerr << label << " failed: valid=" << result.valid
                      << " d=" << result.realized_energy_change
                      << " residual=" << result.residual << '\n';
        }
        return pass;
    }
}

int main (int argc, char* argv[])
{
    static_assert(std::is_trivially_copyable_v<TestInput>);
    static_assert(std::is_trivially_copyable_v<TestOutput>);
    static_assert(std::is_trivially_copyable_v<KineticEnergyUpdateResult>);

    amrex::Initialize(argc, argv);
    bool pass = true;
    {
        amrex::Real const expected_precision_epsilon =
            amrex::max(
                std::numeric_limits<amrex::Real>::epsilon(),
                static_cast<amrex::Real>(
                    std::numeric_limits<amrex::ParticleReal>::epsilon()));
        amrex::Real const host_precision_epsilon = KineticPrecisionEpsilon();
        amrex::Real const synthetic_low = 1.0e-6_rt;
        amrex::Real const synthetic_high = 2.0e-6_rt;
        bool const host_precision_pass =
            host_precision_epsilon == expected_precision_epsilon
            && KineticPrecisionEpsilon(synthetic_high, synthetic_low)
                == synthetic_high
            && KineticPrecisionEpsilon(synthetic_low, synthetic_high)
                == synthetic_high;
        if (!host_precision_pass) {
            std::cerr << "kinetic precision epsilon host semantics failed\n";
        }
        pass = host_precision_pass && pass;

        amrex::Gpu::DeviceVector<amrex::Real> device_precision_epsilon(3);
        amrex::Real* const device_precision_epsilon_ptr =
            device_precision_epsilon.data();
        amrex::ParallelFor(
            3,
            [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                if (i == 0) {
                    device_precision_epsilon_ptr[i] = KineticPrecisionEpsilon();
                } else if (i == 1) {
                    device_precision_epsilon_ptr[i] = KineticPrecisionEpsilon(
                        synthetic_high, synthetic_low);
                } else {
                    device_precision_epsilon_ptr[i] = KineticPrecisionEpsilon(
                        synthetic_low, synthetic_high);
                }
            });
        amrex::Gpu::streamSynchronize();
        std::array<amrex::Real, 3> device_precision_epsilon_host{};
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, device_precision_epsilon.begin(),
            device_precision_epsilon.end(), device_precision_epsilon_host.begin());
        bool const device_precision_pass =
            device_precision_epsilon_host[0] == expected_precision_epsilon
            && device_precision_epsilon_host[1] == synthetic_high
            && device_precision_epsilon_host[2] == synthetic_high;
        if (!device_precision_pass) {
            std::cerr << "kinetic precision epsilon device semantics failed\n";
        }
        pass = device_precision_pass && pass;

        amrex::Real const energy = std::ldexp(amrex::Real(1.0), -40);
        auto const particle_epsilon = static_cast<amrex::Real>(
            std::numeric_limits<amrex::ParticleReal>::epsilon());
        amrex::Real const coefficient = amrex::max(
            2.0e-10_rt, 1024.0_rt * expected_precision_epsilon);
        amrex::Real const old_energy = energy;
        amrex::Real const normal_request = energy / 4.0_rt;
        amrex::Real const normal_target = old_energy + normal_request;
        amrex::Real const accepted_mismatch =
            0.5_rt * coefficient * normal_target;
        amrex::Real const excessive_mismatch =
            2.0_rt * coefficient * normal_target;

        std::array<TestInput, 7> const inputs{
            TestInput{0.0_rt, old_energy, old_energy, old_energy,
                      CaseKind::Zero},
            TestInput{normal_request, old_energy, normal_target, normal_target,
                      CaseKind::Normal},
            TestInput{
                -particle_epsilon * energy / 4.0_rt,
                old_energy,
                old_energy - particle_epsilon * energy / 4.0_rt,
                old_energy,
                CaseKind::BelowParticleUlp},
            TestInput{
                normal_request,
                old_energy,
                normal_target,
                normal_target + accepted_mismatch,
                CaseKind::AcceptedRepresentationMismatch},
            TestInput{
                std::numeric_limits<amrex::Real>::quiet_NaN(),
                old_energy,
                std::numeric_limits<amrex::Real>::quiet_NaN(),
                old_energy,
                CaseKind::Nonfinite},
            TestInput{
                normal_request,
                old_energy,
                normal_target,
                normal_target + excessive_mismatch,
                CaseKind::ExcessiveMismatch},
            TestInput{
                normal_request,
                std::numeric_limits<amrex::Real>::infinity(),
                std::numeric_limits<amrex::Real>::infinity(),
                std::numeric_limits<amrex::Real>::infinity(),
                CaseKind::Nonfinite}};
        std::array<std::string, 7> const labels{
            "zero", "normal", "below_particle_ulp",
            "accepted_representation_mismatch", "nonfinite_request",
            "excessive_mismatch", "nonfinite_old"};

        std::array<KineticEnergyUpdateResult, 7> host_results{};
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            host_results[i] = EvaluateKineticEnergyUpdate(
                inputs[i].requested, inputs[i].old_energy,
                inputs[i].target_energy, inputs[i].achieved_energy);
            pass = CheckResult(labels[i], inputs[i], host_results[i]) && pass;
        }

        amrex::Gpu::DeviceVector<TestInput> device_inputs(inputs.size());
        amrex::Gpu::DeviceVector<TestOutput> device_outputs(inputs.size());
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, inputs.begin(), inputs.end(),
            device_inputs.begin());
        TestInput const* const input_ptr = device_inputs.data();
        TestOutput* const output_ptr = device_outputs.data();
        amrex::ParallelFor(
            static_cast<int>(inputs.size()),
            [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                output_ptr[i].result = EvaluateKineticEnergyUpdate(
                    input_ptr[i].requested, input_ptr[i].old_energy,
                    input_ptr[i].target_energy, input_ptr[i].achieved_energy);
            });
        amrex::Gpu::streamSynchronize();

        std::array<TestOutput, 7> device_results{};
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, device_outputs.begin(),
            device_outputs.end(), device_results.begin());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            pass = CheckResult(labels[i], inputs[i], device_results[i].result)
                && pass;
            pass = (device_results[i].result.valid == host_results[i].valid)
                && SameValue(
                    device_results[i].result.realized_energy_change,
                    host_results[i].realized_energy_change)
                && SameValue(
                    device_results[i].result.residual,
                    host_results[i].residual)
                && pass;
        }

        std::cout << "kinetic energy realization host/device semantics: "
                  << (pass ? "PASS" : "FAIL") << '\n';
    }
    amrex::Finalize();
    return pass ? 0 : 1;
}
