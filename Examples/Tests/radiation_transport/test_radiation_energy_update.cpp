/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Radiation/RadiationEnergyUpdate.H"

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
using warpx::radiation::ApplyRadiationEnergyUpdate;
using warpx::radiation::RadiationEnergyUpdateResult;

namespace
{
    struct TestInput
    {
        amrex::Real old_energy;
        amrex::Real energy_change;
    };

    struct TestOutput
    {
        RadiationEnergyUpdateResult result;
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
        RadiationEnergyUpdateResult const& result)
    {
        amrex::Real const epsilon =
            std::numeric_limits<amrex::Real>::epsilon();
        amrex::Real const scale = amrex::max(
            std::abs(input.old_energy), std::abs(input.energy_change));
        amrex::Real const tolerance =
            amrex::Real(100.0) * epsilon * scale;
        bool pass = true;

        if (label.find("zero") != std::string::npos) {
            pass = result.valid && result.stored_energy == 0.0_rt
                && result.residual == 0.0_rt;
        } else if (label.find("normal") != std::string::npos) {
            pass = result.valid
                && result.stored_energy == amrex::Real(0.75) * input.old_energy
                && result.residual == 0.0_rt;
        } else if (label.find("roundoff_clip") != std::string::npos) {
            amrex::Real const raw =
                input.old_energy + input.energy_change;
            pass = result.valid && result.stored_energy == 0.0_rt
                && result.residual < 0.0_rt
                && result.residual == raw
                && std::abs(result.residual) <= tolerance;
        } else if (label.find("overdraw") != std::string::npos) {
            amrex::Real const raw =
                input.old_energy + input.energy_change;
            pass = !result.valid
                && result.stored_energy == input.old_energy
                && std::abs(raw) > tolerance
                && result.residual == 0.0_rt;
        } else if (label.find("nonfinite") != std::string::npos) {
            pass = !result.valid
                && SameValue(result.stored_energy, input.old_energy)
                && result.residual == 0.0_rt;
        }

        if (!pass) {
            std::cerr << label << " failed: valid=" << result.valid
                      << " stored=" << result.stored_energy
                      << " residual=" << result.residual << '\n';
        }
        return pass;
    }
}

int main (int argc, char* argv[])
{
    static_assert(std::is_trivially_copyable_v<TestInput>);
    static_assert(std::is_trivially_copyable_v<TestOutput>);
    static_assert(std::is_trivially_copyable_v<RadiationEnergyUpdateResult>);

    amrex::Initialize(argc, argv);
    bool pass = true;
    {
        amrex::Real const epsilon =
            std::numeric_limits<amrex::Real>::epsilon();
        amrex::Real const energy = std::ldexp(amrex::Real(1.0), -40);
        auto const zero = amrex::Real(0.0);
        amrex::Real const roundoff_factor =
            amrex::Real(1.0) + amrex::Real(32.0) * epsilon;
        amrex::Real const overdraw_factor =
            amrex::Real(1.0) + amrex::Real(512.0) * epsilon;

        std::array<TestInput, 12> const inputs{
            TestInput{zero, zero},
            TestInput{energy, -energy / amrex::Real(4.0)},
            TestInput{energy, -energy * roundoff_factor},
            TestInput{energy, -energy * overdraw_factor},
            TestInput{energy, std::numeric_limits<amrex::Real>::quiet_NaN()},
            TestInput{std::numeric_limits<amrex::Real>::quiet_NaN(), zero},
            TestInput{zero, zero},
            TestInput{energy, -energy / amrex::Real(4.0)},
            TestInput{energy, -energy * roundoff_factor},
            TestInput{energy, -energy * overdraw_factor},
            TestInput{energy, std::numeric_limits<amrex::Real>::infinity()},
            TestInput{std::numeric_limits<amrex::Real>::infinity(), zero}};
        std::array<std::string, 12> const labels{
            "work/zero",
            "work/normal",
            "work/roundoff_clip",
            "work/overdraw",
            "work/nonfinite_change",
            "work/nonfinite_old",
            "diffusion/zero",
            "diffusion/normal",
            "diffusion/roundoff_clip",
            "diffusion/overdraw",
            "diffusion/nonfinite_change",
            "diffusion/nonfinite_old"};

        std::array<RadiationEnergyUpdateResult, 12> host_results{};
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            host_results[i] = ApplyRadiationEnergyUpdate(
                inputs[i].old_energy, inputs[i].energy_change);
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
                output_ptr[i].result = ApplyRadiationEnergyUpdate(
                    input_ptr[i].old_energy, input_ptr[i].energy_change);
            });
        amrex::Gpu::streamSynchronize();

        std::array<TestOutput, 12> device_results{};
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, device_outputs.begin(),
            device_outputs.end(), device_results.begin());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            pass = CheckResult(labels[i], inputs[i], device_results[i].result)
                && pass;
            pass = (device_results[i].result.valid == host_results[i].valid)
                && SameValue(
                    device_results[i].result.stored_energy,
                    host_results[i].stored_energy)
                && SameValue(
                    device_results[i].result.residual,
                    host_results[i].residual)
                && pass;
        }

        std::cout << "radiation energy update host/device semantics: "
                  << (pass ? "PASS" : "FAIL") << '\n';
    }
    amrex::Finalize();
    return pass ? 0 : 1;
}
