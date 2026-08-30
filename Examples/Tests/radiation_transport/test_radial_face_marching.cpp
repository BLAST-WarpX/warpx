/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Radiation/RadialFaceMarching.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using warpx::radiation::FindNextRadialFace;
using warpx::radiation::RadialFaceMarchResult;
using namespace amrex::literals;

namespace
{
    struct TestInput
    {
        amrex::ParticleReal x;
        amrex::ParticleReal y;
        amrex::ParticleReal nx;
        amrex::ParticleReal ny;
        amrex::ParticleReal nz;
        int current_radial_cell;
        int domain_lo;
        int domain_hi;
        amrex::ParticleReal prob_lo;
        amrex::ParticleReal dr;
        amrex::ParticleReal max_distance;
        amrex::ParticleReal expected_distance;
        int expected_next_radial_cell;
        bool expected_crossed;
        bool expected_outside_domain;
        bool expected_valid_input;
    };

    struct TestOutput
    {
        RadialFaceMarchResult result;
    };

    bool NearlyEqual (
        amrex::ParticleReal const actual,
        amrex::ParticleReal const expected,
        amrex::ParticleReal const scale)
    {
        if (actual == expected) { return true; }
        if (!std::isfinite(actual) || !std::isfinite(expected)) { return false; }
        amrex::ParticleReal const epsilon = amrex::max(
            static_cast<amrex::ParticleReal>(
                std::numeric_limits<amrex::Real>::epsilon()),
            static_cast<amrex::ParticleReal>(
                std::numeric_limits<amrex::ParticleReal>::epsilon()));
        amrex::ParticleReal const tolerance =
            amrex::ParticleReal(128.0) * epsilon * scale;
        return std::abs(actual - expected) <= tolerance;
    }

    amrex::ParticleReal TestScale (TestInput const& input)
    {
        amrex::ParticleReal scale = std::abs(input.x);
        scale = amrex::max(scale, std::abs(input.y));
        scale = amrex::max(scale, std::abs(input.prob_lo));
        scale = amrex::max(scale, std::abs(input.dr));
        scale = amrex::max(scale, std::abs(input.max_distance));
        scale = amrex::max(scale, std::abs(input.expected_distance));
        return scale;
    }

    bool CheckResult (
        std::string const& label,
        TestInput const& input,
        RadialFaceMarchResult const& result)
    {
        bool const pass = NearlyEqual(
                              result.distance,
                              input.expected_distance,
                              TestScale(input))
            && result.next_radial_cell == input.expected_next_radial_cell
            && result.crossed == input.expected_crossed
            && result.outside_domain == input.expected_outside_domain
            && result.valid_input == input.expected_valid_input;
        if (!pass) {
            std::cerr << label << " failed: distance=" << result.distance
                      << " (expected " << input.expected_distance << ")"
                      << ", next=" << result.next_radial_cell
                      << " (expected " << input.expected_next_radial_cell << ")"
                      << ", crossed=" << result.crossed
                      << " (expected " << input.expected_crossed << ")"
                      << ", outside=" << result.outside_domain
                      << " (expected " << input.expected_outside_domain << ")"
                      << ", valid=" << result.valid_input
                      << " (expected " << input.expected_valid_input << ")\n";
        }
        return pass;
    }

    bool CheckHostDevice (
        std::vector<TestInput> const& inputs,
        std::vector<TestOutput> const& host_outputs,
        std::vector<TestOutput> const& device_outputs)
    {
        bool pass = true;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            pass = CheckResult(
                       "device/" + std::to_string(i), inputs[i],
                       device_outputs[i].result)
                && pass;
            pass = device_outputs[i].result.next_radial_cell
                    == host_outputs[i].result.next_radial_cell
                && device_outputs[i].result.crossed
                    == host_outputs[i].result.crossed
                && device_outputs[i].result.outside_domain
                    == host_outputs[i].result.outside_domain
                && device_outputs[i].result.valid_input
                    == host_outputs[i].result.valid_input
                && NearlyEqual(
                    device_outputs[i].result.distance,
                    host_outputs[i].result.distance,
                    TestScale(inputs[i]))
                && pass;
        }
        return pass;
    }

}

int main (int argc, char* argv[])
{
    static_assert(std::is_trivially_copyable_v<TestInput>);
    static_assert(std::is_trivially_copyable_v<TestOutput>);
    static_assert(std::is_trivially_copyable_v<RadialFaceMarchResult>);
    static_assert(std::is_trivial_v<RadialFaceMarchResult>);
    static_assert(std::is_standard_layout_v<RadialFaceMarchResult>);
    amrex::Initialize(argc, argv);
    bool pass = true;
    {
        amrex::ParticleReal const precision_epsilon = amrex::max(
            static_cast<amrex::ParticleReal>(
                std::numeric_limits<amrex::Real>::epsilon()),
            static_cast<amrex::ParticleReal>(
                std::numeric_limits<amrex::ParticleReal>::epsilon()));
        amrex::ParticleReal const just_inside_tangent =
            std::sqrt(amrex::ParticleReal(1.0)
                      - amrex::ParticleReal(0.999) * amrex::ParticleReal(0.999));
        amrex::ParticleReal const direction_at_root =
            amrex::ParticleReal(0.25) + amrex::ParticleReal(0.6);
        amrex::ParticleReal const oblique_distance =
            -amrex::ParticleReal(0.15) + std::sqrt(amrex::ParticleReal(0.96));
        amrex::ParticleReal const threshold_discriminant =
            amrex::ParticleReal(128.0) * precision_epsilon;
        amrex::ParticleReal const threshold_inside_tangent = std::sqrt(
            amrex::ParticleReal(1.0) - threshold_discriminant);
        amrex::ParticleReal const threshold_inside_distance = std::sqrt(
            threshold_discriminant);
        amrex::ParticleReal const threshold_outside_tangent = std::sqrt(
            amrex::ParticleReal(1.0) + threshold_discriminant);
        amrex::ParticleReal const endpoint_below = std::nextafter(
            amrex::ParticleReal(0.75), amrex::ParticleReal(0.0));
        auto const small_transverse_nx =
            amrex::ParticleReal(1.0e-8);
        amrex::ParticleReal const small_transverse_nz = std::sqrt(
            amrex::ParticleReal(1.0)
            - small_transverse_nx * small_transverse_nx);
        amrex::ParticleReal const nan =
            std::numeric_limits<amrex::ParticleReal>::quiet_NaN();
        amrex::ParticleReal const infinity =
            std::numeric_limits<amrex::ParticleReal>::infinity();
        amrex::ParticleReal const finite_max =
            std::numeric_limits<amrex::ParticleReal>::max();

        std::vector<TestInput> const inputs{
            // Ordinary outward and inward annular crossings.
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 0.75_prt, 1,
                      true, false, true},
            TestInput{1.75_prt, 0.0_prt, -1.0_prt, 0.0_prt, 0.0_prt,
                      1, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 0.75_prt, 0,
                      true, false, true},
            // Exact face departures and non-repeating arrivals.
            TestInput{1.0_prt, 0.0_prt, -1.0_prt, 0.0_prt, 0.0_prt,
                      1, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 0.0_prt, 0,
                      true, false, true},
            TestInput{2.0_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      1, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 0.0_prt, 2,
                      true, false, true},
            TestInput{1.0_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      1, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 1.0_prt, 2,
                      true, false, true},
            TestInput{2.0_prt, 0.0_prt, -1.0_prt, 0.0_prt, 0.0_prt,
                      1, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 1.0_prt, 0,
                      true, false, true},
            // Tangency, including points just on either side of the tangent.
            TestInput{0.0_prt, 1.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 2.0_prt, 0,
                      false, false, true},
            TestInput{0.0_prt, 0.999_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, 1.0_prt,
                      just_inside_tangent, 1, true, false, true},
            TestInput{0.0_prt, 1.001_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, 1.0_prt, 1.0_prt, 0,
                      false, false, true},
            // Cases just inside/outside the discriminant tolerance.
            TestInput{0.0_prt, threshold_inside_tangent, 1.0_prt, 0.0_prt,
                      0.0_prt, 0, 0, 3, 0.0_prt, 1.0_prt, 1.0_prt,
                      threshold_inside_distance, 1, true, false, true},
            TestInput{0.0_prt, threshold_outside_tangent, 1.0_prt, 0.0_prt,
                      0.0_prt, 0, 0, 3, 0.0_prt, 1.0_prt, 1.0_prt, 1.0_prt,
                      0, false, false, true},
            // Purely axial motion has zero transverse velocity.
            TestInput{0.5_prt, 0.0_prt, 0.0_prt, 0.0_prt, 1.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 2.0_prt, 0,
                      false, false, true},
            TestInput{0.5_prt, 0.0_prt, 0.0_prt, 0.0_prt, -1.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 2.0_prt, 0,
                      false, false, true},
            // The axis is skipped, so a path traversing it reaches the outer face.
            TestInput{-0.5_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 1.5_prt, 1,
                      true, false, true},
            // Radial direction is evaluated at the root, after closest approach.
            TestInput{0.25_prt, 0.8_prt, -1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt,
                      direction_at_root, 1, true, false, true},
            // An oblique unit vector has the same path-length contract.
            TestInput{0.25_prt, 0.0_prt, 0.6_prt, 0.8_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, oblique_distance, 1,
                      true, false, true},
            // Positive-radius lower and upper domain exits.
            TestInput{1.0_prt, 0.0_prt, -1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 1, 1.0_prt, 1.0_prt, 1.0_prt, 0.0_prt, -1,
                      true, true, true},
            TestInput{1.5_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 0, 1.0_prt, 1.0_prt, 1.0_prt, 0.5_prt, 1,
                      true, true, true},
            // The requested path can end before the next face.
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, 0.5_prt, 0.5_prt, 0,
                      false, false, true},
            // The strict endpoint contract distinguishes an exact endpoint
            // from a path ending at the preceding representable distance.
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, 0.75_prt, 0.75_prt, 1,
                      true, false, true},
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, endpoint_below,
                      endpoint_below, 0, false, false, true},
            // The same crossing at very small and very large length scales.
            TestInput{0.25e-9_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0e-9_prt, 1.0e-9_prt, 1.0e-9_prt,
                      0.75e-9_prt, 1, true, false, true},
            TestInput{0.25e9_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0e9_prt, 1.0e9_prt, 1.0e9_prt,
                      0.75e9_prt, 1, true, false, true},
            // A translated thin annulus exercises the cancellation-safe root.
            TestInput{1000000.125_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 1, 1000000.0_prt, 0.25_prt, 0.25_prt,
                      0.125_prt, 1, true, false, true},
            // Small transverse motion remains a valid, long path crossing.
            TestInput{0.25_prt, 0.0_prt, small_transverse_nx, 0.0_prt,
                      small_transverse_nz, 0, 0, 1, 0.0_prt, 1.0_prt,
                      8.0e7_prt, 7.5e7_prt, 1, true, false, true},
            // Rejected coordinates, directions, and radial geometry.
            TestInput{nan, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 0.0_prt, 0,
                      false, false, false},
            TestInput{infinity, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 0.0_prt, 0,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, nan, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 0.0_prt, 0,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, infinity, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 0.0_prt, 0,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, nan,
                      0, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 0.0_prt, 0,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, infinity,
                      0, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 0.0_prt, 0,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, nan, 1.0_prt, 2.0_prt, 0.0_prt, 0,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, infinity, 1.0_prt, 2.0_prt, 0.0_prt, 0,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, -1.0_prt, 1.0_prt, 2.0_prt, 0.0_prt, 0,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 0.0_prt, 2.0_prt, 0.0_prt, 0,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, -1.0_prt, 2.0_prt, 0.0_prt, 0,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, infinity, 2.0_prt, 0.0_prt, 0,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, nan, 2.0_prt, 0.0_prt, 0,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, -1.0_prt, 0.0_prt, 0,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, infinity, 0.0_prt, 0,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, nan, 0.0_prt, 0,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      -1, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 0.0_prt, -1,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      4, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 0.0_prt, 4,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 3, 0, 0.0_prt, 1.0_prt, 2.0_prt, 0.0_prt, 0,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, 1.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 1, finite_max / 2.0_prt, finite_max, 1.0_prt,
                      0.0_prt, 0, false, false, false},
            // A non-unit vector is rejected even if its transverse part is valid.
            TestInput{0.25_prt, 0.0_prt, 0.6_prt, 0.8_prt, 0.1_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 0.0_prt, 0,
                      false, false, false},
            TestInput{0.25_prt, 0.0_prt, 0.0_prt, 0.0_prt, 0.0_prt,
                      0, 0, 3, 0.0_prt, 1.0_prt, 2.0_prt, 0.0_prt, 0,
                      false, false, false}};

        std::vector<std::string> const labels{
            "outward", "inward", "exact_inner_inward", "exact_outer_outward",
            "exact_inner_outward_nonrepeat", "exact_outer_inward_nonrepeat",
            "exact_tangent", "just_inside_tangent", "just_outside_tangent",
            "threshold_inside_tangent", "threshold_outside_tangent",
            "zero_transverse_velocity_positive_axial",
            "zero_transverse_velocity_negative_axial", "axis_traversal",
            "direction_at_root", "oblique_unit", "lower_domain_exit",
            "outer_exit", "no_crossing", "exact_endpoint", "endpoint_below",
            "small_scale", "large_scale", "translated_thin_annulus",
            "small_transverse_component", "nan_coordinate", "inf_coordinate",
            "nan_direction", "inf_direction", "nan_axial_direction",
            "inf_axial_direction", "nan_prob_lo", "inf_prob_lo",
            "negative_prob_lo", "zero_dr", "negative_dr", "inf_dr", "nan_dr",
            "negative_max_distance", "inf_max_distance", "nan_max_distance",
            "cell_below_domain", "cell_above_domain", "reversed_domain",
            "derived_radius_overflow", "non_unit_direction", "zero_direction"};

        std::vector<TestOutput> host_outputs(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            host_outputs[i].result = FindNextRadialFace(
                inputs[i].x, inputs[i].y, inputs[i].nx, inputs[i].ny,
                inputs[i].nz,
                inputs[i].current_radial_cell, inputs[i].domain_lo,
                inputs[i].domain_hi, inputs[i].prob_lo, inputs[i].dr,
                inputs[i].max_distance);
            pass = CheckResult(labels[i], inputs[i], host_outputs[i].result)
                && pass;
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
                TestInput const& input = input_ptr[i];
                output_ptr[i].result = FindNextRadialFace(
                    input.x, input.y, input.nx, input.ny, input.nz,
                    input.current_radial_cell, input.domain_lo,
                    input.domain_hi, input.prob_lo, input.dr,
                    input.max_distance);
            });
        amrex::Gpu::streamSynchronize();

        std::vector<TestOutput> device_outputs_host(inputs.size());
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, device_outputs.begin(),
            device_outputs.end(), device_outputs_host.begin());
        pass = CheckHostDevice(inputs, host_outputs, device_outputs_host)
            && pass;

        std::cout << "radial face marching host/device semantics: "
                  << (pass ? "PASS" : "FAIL") << '\n';
    }
    amrex::Finalize();
    return pass ? 0 : 1;
}
