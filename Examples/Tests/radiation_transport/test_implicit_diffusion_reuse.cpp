/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/ImplicitDiffusion.H"
#include "Radiation/RadiationBoundary.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Parser.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

using namespace amrex::literals;

namespace
{
    void
    initialize (amrex::MultiFab& radiation, amrex::MultiFab& opacity)
    {
        for (amrex::MFIter mfi(radiation); mfi.isValid(); ++mfi) {
            auto const energy = radiation.array(mfi);
            auto const alpha = opacity.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), radiation.nComp(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int g) noexcept {
                    amrex::Real const x = (i + 0.5_rt) / 32.0_rt;
                    energy(i, j, k, g) =
                        (1.0_rt / 32.0_rt) * (1.0_rt + g) *
                        (1.0_rt + 0.35_rt * std::cos(8.0_rt * MathConst::pi * x));
                    alpha(i, j, k, g) = (60.0_rt + 40.0_rt * g) *
                                        (1.0_rt + 0.1_rt * std::cos(4.0_rt * MathConst::pi * x));
                });
        }
    }

    void
    compare (amrex::MultiFab const& fresh, amrex::MultiFab const& reused,
             warpx::radiation::ImplicitDiffusionResult const& fresh_result,
             warpx::radiation::ImplicitDiffusionResult const& reused_result,
             amrex::Real tolerance)
    {
        AMREX_ALWAYS_ASSERT(fresh_result.failure ==
                                warpx::radiation::ImplicitDiffusionFailure::None &&
                            reused_result.failure ==
                                warpx::radiation::ImplicitDiffusionFailure::None);
        for (int group = 0; group < fresh.nComp(); ++group) {
            amrex::MultiFab difference(fresh.boxArray(), fresh.DistributionMap(), 1, 0);
            amrex::MultiFab::Copy(difference, reused, group, 0, 1, 0);
            amrex::MultiFab::Subtract(difference, fresh, group, 0, 1, 0);
            amrex::Real const scale =
                amrex::max(fresh.norm0(group), reused.norm0(group), 1.0e-30_rt);
            AMREX_ALWAYS_ASSERT(difference.norm0(0) <= 64.0_rt * tolerance * scale);
            AMREX_ALWAYS_ASSERT(std::abs(fresh_result.group_escaped_energy[group] -
                                         reused_result.group_escaped_energy[group]) <=
                                64.0_rt * tolerance * scale);
            AMREX_ALWAYS_ASSERT(std::abs(fresh_result.group_injected_energy[group] -
                                         reused_result.group_injected_energy[group]) <=
                                64.0_rt * tolerance * scale);
            AMREX_ALWAYS_ASSERT(std::abs(fresh_result.group_material_energy[group] -
                                         reused_result.group_material_energy[group]) <=
                                64.0_rt * tolerance * scale);
        }
    }
} // namespace

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        static_assert(AMREX_SPACEDIM == 1, "This fixture is a one-dimensional slab.");
        constexpr int cells = 32;
        constexpr int groups = 2;
        amrex::Box const domain(amrex::IntVect(0), amrex::IntVect(cells - 1));
        amrex::RealBox const physical(0.0_rt, 1.0_rt);
        int periodic[] = {0};
        amrex::Geometry const geometry(domain, &physical, 0, periodic);
        amrex::BoxArray boxes(domain);
        boxes.maxSize(8);
        amrex::DistributionMapping const distribution(boxes);
        amrex::MultiFab fresh(boxes, distribution, groups, 1);
        amrex::MultiFab reused(boxes, distribution, groups, 1);
        amrex::MultiFab opacity(boxes, distribution, groups, 1);
        amrex::MultiFab absorption(boxes, distribution, groups, 0);
        amrex::MultiFab emission(boxes, distribution, groups, 0);
        fresh.setVal(7.0_rt);
        reused.setVal(7.0_rt);
        opacity.setVal(1.0_rt);

        amrex::Vector<amrex::ParserExecutor<4>> host_baths(2 * groups);
        // Executors borrow compiled parser storage through the entire solve.
        amrex::Vector<amrex::Parser> parsers;
        parsers.reserve(2 * groups);
        for (int group = 0; group < groups; ++group) {
            parsers.emplace_back("2.0 + 0.5 * " + std::to_string(group) +
                                 " + 0.1 * x + 0.02 * t");
            parsers.emplace_back("1.0 + 0.25 * " + std::to_string(group) +
                                 " + 0.1 * x + 0.02 * t");
            auto& lower = parsers[2 * group];
            auto& upper = parsers[2 * group + 1];
            lower.registerVariables({"x", "y", "z", "t"});
            upper.registerVariables({"x", "y", "z", "t"});
            host_baths[group] = lower.compile<4>();
            host_baths[groups + group] = upper.compile<4>();
        }
        amrex::Gpu::DeviceVector<amrex::ParserExecutor<4>> baths(host_baths.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, host_baths.begin(), host_baths.end(),
                         baths.begin());

        using Boundary = warpx::radiation::DiffusionBoundary;
        int const bath = static_cast<int>(Boundary::MarshakBath);
        warpx::radiation::ImplicitDiffusionOptions options{
            1.0e-10_rt,
            1.0e-13_rt,
            1.0_rt,
            100,
            100,
            0,
            {bath, 0, 0},
            {bath, 0, 0},
            {0, 0, 0, 0, 0, 0},
            baths.dataPtr(),
            {nullptr, nullptr, groups}};
        options.absorption_rate = &absorption;
        options.emissivity = &emission;
        initialize(fresh, opacity);
        opacity.FillBoundary(geometry.periodicity());
        amrex::MultiFab::Copy(reused, fresh, 0, 0, groups, 1);
        for (amrex::MFIter mfi(absorption); mfi.isValid(); ++mfi) {
            auto const rate = absorption.array(mfi);
            auto const source = emission.array(mfi);
            amrex::ParallelFor(mfi.validbox(), groups,
                               [=] AMREX_GPU_DEVICE (int i, int j, int k, int g) noexcept {
                                   rate(i, j, k, g) = (1.0_rt + g) * 1.0e9_rt;
                                   source(i, j, k, g) = 0.2_rt * rate(i, j, k, g);
                               });
        }

        amrex::Real const dt = 1.0e-9_rt;
        amrex::GpuArray<amrex::Real, groups> initial_energy{};
        for (int group = 0; group < groups; ++group) {
            initial_energy[group] = fresh.sum(group, false);
        }
        options.reuse_operator = false;
        auto const fresh_result = warpx::radiation::AdvanceImplicitDiffusion(
            fresh, opacity, geometry, 0.0_rt, dt, options);
        options.reuse_operator = true;
        auto const reused_result = warpx::radiation::AdvanceImplicitDiffusion(
            reused, opacity, geometry, 0.0_rt, dt, options);
        compare(fresh, reused, fresh_result, reused_result, options.tolerance);
        for (int group = 0; group < groups; ++group) {
            amrex::Real const initial = initial_energy[group];
            auto const& result = reused_result;
            AMREX_ALWAYS_ASSERT(std::abs(fresh.sum(group, false) - initial +
                                         result.group_escaped_energy[group] -
                                         result.group_injected_energy[group] +
                                         result.group_material_energy[group]) <
                                64.0_rt * options.tolerance);
        }
        amrex::Print() << "implicit diffusion fresh/reused operator states agree\n";

        // A uniform first group converges in one Picard pass; a later strongly
        // nonlinear group exhausts the deliberately small budget. No scratch
        // state or transfer from either group may become visible to the caller.
        options.absorption_rate = nullptr;
        options.emissivity = nullptr;
        options.boundary_lo = {0, 0, 0};
        options.boundary_hi = {0, 0, 0};
        options.max_iterations = 1;
        options.nonlinear_relaxation = 1.0_rt;
        for (amrex::MFIter mfi(reused); mfi.isValid(); ++mfi) {
            auto const energy = reused.array(mfi);
            amrex::ParallelFor(mfi.validbox(), groups,
                               [=] AMREX_GPU_DEVICE (int i, int j, int k, int g) noexcept {
                                   energy(i, j, k, g) =
                                       g == 0 ? 1.0_rt / 32.0_rt
                                              : (1.0_rt / 32.0_rt) *
                                                    (1.0_rt + 0.8_rt * (i % 2));
                               });
        }
        reused.FillBoundary(geometry.periodicity());
        amrex::MultiFab saved(boxes, distribution, groups, 1);
        amrex::MultiFab difference(boxes, distribution, groups, 1);
        amrex::MultiFab::Copy(saved, reused, 0, 0, groups, 1);
        auto const rejected = warpx::radiation::TryAdvanceImplicitDiffusion(
            reused, opacity, geometry, 0.0_rt, 1.0e-7_rt, options);
        AMREX_ALWAYS_ASSERT(rejected.failure ==
                            warpx::radiation::ImplicitDiffusionFailure::NonlinearConvergence);
        AMREX_ALWAYS_ASSERT(rejected.nonlinear_iterations >= groups);
        AMREX_ALWAYS_ASSERT(rejected.escaped_energy == 0 && rejected.injected_energy == 0 &&
                            rejected.numerical_energy_residual == 0);
        amrex::MultiFab::Copy(difference, reused, 0, 0, groups, 1);
        amrex::MultiFab::Subtract(difference, saved, 0, 0, groups, 1);
        for (int group = 0; group < groups; ++group) {
            AMREX_ALWAYS_ASSERT(difference.norm0(group, 1) == 0);
        }
        amrex::Print() << "implicit diffusion later-group failure leaves no partial commit\n";
    }
    amrex::Finalize();
}
