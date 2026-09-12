/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/ImplicitDiffusion.H"
#include "Radiation/RadiationTransport.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>

using namespace amrex::literals;

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        constexpr int cells = 32;
        constexpr int groups = 3;
        amrex::Box const domain(amrex::IntVect(0), amrex::IntVect(cells - 1));
        amrex::RealBox const physical(0.0_rt, 1.0_rt);
        int periodic[] = {1};
        amrex::Geometry const geometry(domain, &physical, 0, periodic);
        amrex::BoxArray boxes(domain);
        boxes.maxSize(8);
        amrex::DistributionMapping const distribution(boxes);
        amrex::MultiFab energy(boxes, distribution, groups, 1);
        amrex::MultiFab opacity(boxes, distribution, groups, 1);
        amrex::Real const dx = 1.0_rt / cells;
        amrex::Real const tolerance =
            amrex::max(1.0e-10_rt, 64.0_rt * std::numeric_limits<amrex::Real>::epsilon());
        warpx::radiation::ImplicitDiffusionOptions options{
            tolerance,
            amrex::max(1.0e-13_rt, 8.0_rt * std::numeric_limits<amrex::Real>::epsilon()),
            1.0_rt,
            100,
            100,
            0,
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0, 0, 0, 0},
            nullptr,
            {nullptr, nullptr, groups}};
        for (amrex::Real const factor : {0.01_rt, 10.0_rt, 100.0_rt}) {
            for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi) {
                auto const e = energy.array(mfi);
                auto const a = opacity.array(mfi);
                amrex::ParallelFor(mfi.validbox(), groups,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k, int g) noexcept {
                                       amrex::Real const background =
                                           g == 0 ? 1.0_rt : (g == 1 ? 0.1_rt : 0.001_rt);
                                       e(i, j, k, g) =
                                           dx * background *
                                           (1.0_rt + 0.01_rt * std::cos(2.0_rt * MathConst::pi *
                                                                        (i + 0.5_rt) * dx));
                                       a(i, j, k, g) = 100.0_rt * (g + 1);
                                   });
            }
            opacity.FillBoundary(geometry.periodicity());
            amrex::Real const d0 = PhysConst::c / 300.0_rt;
            amrex::Real const dt = factor * dx * dx / (2.0_rt * d0);
            auto const result = warpx::radiation::AdvanceImplicitDiffusion(
                energy, opacity, geometry, 0.0_rt, dt, options);
            AMREX_ALWAYS_ASSERT(result.maximum_relative_residual <= tolerance);
            AMREX_ALWAYS_ASSERT(result.escaped_energy == 0 && result.injected_energy == 0);
            for (int group = 0; group < groups; ++group) {
                amrex::Real const background =
                    group == 0 ? 1.0_rt : (group == 1 ? 0.1_rt : 0.001_rt);
                amrex::Real const integral = energy.sum(group, false);
                AMREX_ALWAYS_ASSERT(std::abs(integral / background - 1.0_rt) < 10.0_rt * tolerance);
                amrex::ReduceOps<amrex::ReduceOpSum> ops;
                amrex::ReduceData<amrex::Real> data(ops);
                using Tuple = typename decltype(data)::Type;
                for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi) {
                    auto const e = energy.const_array(mfi);
                    ops.eval(mfi.validbox(), data,
                             [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                                 return {2.0_rt * (e(i, j, k, group) - dx * background) *
                                         std::cos(2.0_rt * MathConst::pi * (i + 0.5_rt) * dx)};
                             });
                }
                amrex::Real amplitude = amrex::get<0>(data.value());
                amrex::ParallelDescriptor::ReduceRealSum(amplitude);
                amrex::Real const sine = std::sin(MathConst::pi / cells);
                amrex::Real const eigenvalue = 4.0_rt * sine * sine / (dx * dx);
                amrex::Real const expected =
                    0.01_rt * background / (1.0_rt + dt * d0 / (group + 1) * eigenvalue);
                // The LP limiter differs from 1/3 by O(R^2), below 1e-6 here.
                amrex::Real const allowed = amrex::max(2.0e-6_rt, 100.0_rt * tolerance);
                AMREX_ALWAYS_ASSERT(std::abs(amplitude / expected - 1.0_rt) < allowed);
                amrex::Print() << "implicit FLD factor=" << factor << " group=" << group
                               << " mode relative error=" << amplitude / expected - 1.0_rt
                               << " nonlinear residual=" << result.maximum_relative_residual
                               << '\n';
            }
        }
        // A steep pulse exercises the nonlinear limiter far from the diffusion
        // approximation used by the Fourier-mode oracle above.
        options.minimum_optical_depth = 0.001_rt;
        options.max_iterations = 200;
        for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi) {
            auto const e = energy.array(mfi);
            auto const a = opacity.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), groups, [=] AMREX_GPU_DEVICE(int i, int j, int k, int g) noexcept {
                    amrex::Real const x = (i + 0.5_rt) * dx - 0.5_rt;
                    amrex::Real const scale = g == 0 ? 1.0_rt : (g == 1 ? 0.1_rt : 0.001_rt);
                    e(i, j, k, g) =
                        dx * scale * (1.0e-8_rt + 100.0_rt * std::exp(-x * x / 0.0018_rt));
                    a(i, j, k, g) = static_cast<amrex::Real>(g + 1);
                });
        }
        opacity.FillBoundary(geometry.periodicity());
        amrex::GpuArray<amrex::Real, groups> totals{};
        for (int g = 0; g < groups; ++g) {
            totals[g] = energy.sum(g, false);
        }
        auto const pulse = warpx::radiation::AdvanceImplicitDiffusion(
            energy, opacity, geometry, 0.0_rt, 10.0_rt * dx / PhysConst::c, options);
        AMREX_ALWAYS_ASSERT(pulse.maximum_relative_residual <= tolerance);
        for (int g = 0; g < groups; ++g) {
            AMREX_ALWAYS_ASSERT(energy.min(g) >= 0);
            AMREX_ALWAYS_ASSERT(std::abs(energy.sum(g, false) / totals[g] - 1.0_rt) <
                                10 * tolerance);
        }
        amrex::Print() << "steep-pulse nonlinear iterations=" << pulse.nonlinear_iterations
                       << " residual=" << pulse.maximum_relative_residual << '\n';

        // Group 0 is exactly uniform and can converge immediately; a later
        // nonuniform group must fail a one-Picard-iteration budget. Verify that
        // no group or guard cell changes, then retry the same state successfully.
        energy.setVal(dx, 0, 1, 1);
        energy.FillBoundary(geometry.periodicity());
        amrex::MultiFab saved(boxes, distribution, groups, 1);
        amrex::MultiFab difference(boxes, distribution, groups, 1);
        amrex::MultiFab::Copy(saved, energy, 0, 0, groups, 1);
        int const accepted_iteration_budget = options.max_iterations;
        options.max_iterations = 1;
        auto const rejected = warpx::radiation::TryAdvanceImplicitDiffusion(
            energy, opacity, geometry, 0.0_rt, 10.0_rt * dx / PhysConst::c, options);
        AMREX_ALWAYS_ASSERT(rejected.failure ==
                            warpx::radiation::ImplicitDiffusionFailure::NonlinearConvergence);
        AMREX_ALWAYS_ASSERT(rejected.nonlinear_iterations >= 2);
        AMREX_ALWAYS_ASSERT(rejected.escaped_energy == 0 && rejected.injected_energy == 0 &&
                            rejected.numerical_energy_residual == 0);
        amrex::MultiFab::Copy(difference, energy, 0, 0, groups, 1);
        amrex::MultiFab::Subtract(difference, saved, 0, 0, groups, 1);
        for (int g = 0; g < groups; ++g) {
            AMREX_ALWAYS_ASSERT(difference.norm0(g, 1) == 0);
        }
        options.max_iterations = accepted_iteration_budget;
        auto const retry = warpx::radiation::TryAdvanceImplicitDiffusion(
            energy, opacity, geometry, 0.0_rt, 10.0_rt * dx / PhysConst::c, options);
        AMREX_ALWAYS_ASSERT(retry.failure == warpx::radiation::ImplicitDiffusionFailure::None);
        AMREX_ALWAYS_ASSERT(retry.maximum_relative_residual <= tolerance);
        amrex::Print() << "Rejected late-group solve preserves all groups/guards; retry passes.\n";
        // Frozen-material reaction + spatial transport: independent discrete
        // Fourier solution, including a band initially empty but emitting.
        amrex::MultiFab absorption(boxes, distribution, groups, 0);
        amrex::MultiFab emission(boxes, distribution, groups, 0);
        options.absorption_rate = &absorption;
        options.emissivity = &emission;
        options.tolerance = tolerance;
        opacity.setVal(100.0_rt);
        amrex::Real const reaction_dt = 1.0e-9_rt;
        for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi) {
            auto const e = energy.array(mfi);
            auto const rate = absorption.array(mfi);
            auto const emissivity = emission.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), groups, [=] AMREX_GPU_DEVICE(int i, int j, int k, int g) {
                    e(i, j, k, g) = g == 2
                                        ? 0.0_rt
                                        : dx * (1.0_rt + 0.01_rt * std::cos(2.0_rt * MathConst::pi *
                                                                            (i + 0.5_rt) * dx));
                    rate(i, j, k, g) = (g + 1) / reaction_dt;
                    emissivity(i, j, k, g) = 0.3_rt * (g + 1) / reaction_dt;
                });
        }
        amrex::MultiFab::Copy(saved, energy, 0, 0, groups, 1);
        auto const reaction = warpx::radiation::TryAdvanceImplicitDiffusion(
            energy, opacity, geometry, 0.0_rt, reaction_dt, options);
        AMREX_ALWAYS_ASSERT(reaction.failure == warpx::radiation::ImplicitDiffusionFailure::None);
        for (int g = 0; g < groups; ++g) {
            amrex::Real const initial_mean = g == 2 ? 0.0_rt : 1.0_rt;
            amrex::Real const expected_mean = (initial_mean + 0.3_rt * (g + 1)) / (g + 2);
            AMREX_ALWAYS_ASSERT(std::abs(energy.sum(g, false) - expected_mean) < 10 * tolerance);
            AMREX_ALWAYS_ASSERT(std::abs(energy.sum(g, false) + reaction.group_material_energy[g] -
                                         initial_mean) < 10 * tolerance);
            AMREX_ALWAYS_ASSERT(reaction.group_escaped_energy[g] == 0 &&
                                reaction.group_injected_energy[g] == 0);
            amrex::Real const sine = std::sin(MathConst::pi / cells);
            amrex::Real const eigenvalue = 4.0_rt * sine * sine / (dx * dx);
            amrex::Real const expected_amplitude =
                g == 2 ? 0.0_rt
                       : 0.01_rt / (g + 2 + reaction_dt * PhysConst::c / 300.0_rt * eigenvalue);
            for (amrex::MFIter mfi(difference); mfi.isValid(); ++mfi) {
                auto const e = energy.const_array(mfi);
                auto const err = difference.array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    err(i, j, k, g) =
                        e(i, j, k, g) / dx - expected_mean -
                        expected_amplitude * std::cos(2.0_rt * MathConst::pi * (i + 0.5_rt) * dx);
                });
            }
            AMREX_ALWAYS_ASSERT(difference.norm0(g) <
                                amrex::max(2.0e-6_rt * expected_amplitude, 10.0_rt * tolerance));
        }
        amrex::Print() << "Frozen-material reaction/diffusion and spectral source ledger pass.\n";
        auto const checked = warpx::radiation::EvaluateImplicitDiffusionResidual(
            saved, energy, opacity, geometry, 0.0_rt, reaction_dt, options);
        AMREX_ALWAYS_ASSERT(checked.maximum_relative_residual <= tolerance);
        // A materially changed emitter must invalidate the previously converged
        // radiation state. This prevents checking only lagged coefficients.
        emission.mult(1.1_rt, 0, groups);
        amrex::MultiFab::Copy(difference, energy, 0, 0, groups, 1);
        auto const stale = warpx::radiation::EvaluateImplicitDiffusionResidual(
            saved, energy, opacity, geometry, 0.0_rt, reaction_dt, options);
        AMREX_ALWAYS_ASSERT(stale.maximum_relative_residual > 0.01_rt);
        amrex::MultiFab::Subtract(difference, energy, 0, 0, groups, 1);
        for (int g = 0; g < groups; ++g) {
            AMREX_ALWAYS_ASSERT(difference.norm0(g, 1) == 0);
        }
        if constexpr (sizeof(amrex::Real) == sizeof(double))
        {
            // Stiff diffusion of a small, representable perturbation on a large
            // background. Manufacture the backward-Euler alternating-cell mode.
            // The correction RHS must retain the signal instead of subtracting
            // assembled O(background*diffusive_stiffness) matrix products.
            options.use_incremental_form = true;
            options.nonlinear_relaxation = 1;
            options.tolerance = 1.e-13_rt;
            options.linear_tolerance = 1.e-14_rt;
            absorption.setVal(0);
            emission.setVal(0);
            opacity.setVal(PhysConst::c / 3);
            constexpr amrex::Real stiff_dt = 1024;
            auto const stiffness = 4 * stiff_dt / (dx * dx); // D=1 m^2/s.
            for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi)
            {
                auto const e = energy.array(mfi);
                amrex::ParallelFor(mfi.validbox(), groups,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k, int g)
                                   {
                                       auto const background = std::ldexp(1.0_rt, 40 - 10 * g);
                                       auto const perturbation =
                                           std::ldexp(1.0_rt, -2 * g) * (i % 2 == 0 ? 1 : -1);
                                       e(i, j, k, g) =
                                           dx * (background + (1 + stiffness) * perturbation);
                                   });
            }
            auto const stiff = warpx::radiation::TryAdvanceImplicitDiffusion(
                energy, opacity, geometry, 0, stiff_dt, options);
            AMREX_ALWAYS_ASSERT(stiff.failure == warpx::radiation::ImplicitDiffusionFailure::None);
            AMREX_ALWAYS_ASSERT(stiff.maximum_relative_residual <= options.tolerance);
            for (amrex::MFIter mfi(difference); mfi.isValid(); ++mfi)
            {
                auto const e = energy.const_array(mfi);
                auto const error = difference.array(mfi);
                amrex::ParallelFor(mfi.validbox(), groups,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k, int g)
                                   {
                                       auto const background = std::ldexp(1.0_rt, 40 - 10 * g);
                                       auto const amplitude = std::ldexp(1.0_rt, -2 * g);
                                       auto const target =
                                           background + amplitude * (i % 2 == 0 ? 1 : -1);
                                       error(i, j, k, g) =
                                           (e(i, j, k, g) / dx - target) / amplitude;
                                   });
            }
            for (int g = 0; g < groups; ++g)
            {
                AMREX_ALWAYS_ASSERT(difference.norm0(g) < 0.01_rt);
            }
            amrex::Print() << "Stiff-background correction solve residual="
                           << stiff.maximum_relative_residual << '\n';
        }
    }
    amrex::Finalize();
}
