/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/ImplicitMomentTransport.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <string>

using namespace amrex::literals;

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse inputs("test");
        int cells = 64, steps = 100;
        amrex::Real duration = 1000, speed = 3.e-4_rt, opacity = 1.e4_rt;
        std::string mode = "trapped";
        inputs.query("mode", mode);
        if (mode == "beam") {
            duration = 0.1_rt;
            speed = 0;
            opacity = 0;
        }
        bool const reflecting = mode == "reflecting";
        AMREX_ALWAYS_ASSERT(mode == "beam" || mode == "trapped" || reflecting);
        inputs.query("cells", cells);
        inputs.query("steps", steps);
        inputs.query("duration", duration);
        inputs.query("beta", speed);
        inputs.query("opacity", opacity);
        amrex::IntVect hi(AMREX_D_DECL(cells - 1, 3, 3));
        amrex::Box domain(amrex::IntVect(0), hi);
        amrex::RealBox bounds({AMREX_D_DECL(0., 0., 0.)}, {AMREX_D_DECL(1., 1., 1.)});
        int periodic[AMREX_SPACEDIM] = {AMREX_D_DECL(1, 1, 1)};
        if (reflecting) {
            periodic[0] = 0;
        }
        amrex::Geometry geometry(domain, &bounds, 0, periodic);
        amrex::BoxArray boxes(domain);
        boxes.maxSize(cells / 2);
        amrex::DistributionMapping distribution(boxes);
        amrex::MultiFab radiation(boxes, distribution, 4, 1), beta(boxes, distribution, 3, 0);
        amrex::MultiFab absorption(boxes, distribution, 1, 0),
            scattering(boxes, distribution, 1, 0);
        amrex::MultiFab equilibrium(boxes, distribution, 1, 0), transfer(boxes, distribution, 4, 1);
        absorption.setVal(0);
        scattering.setVal(opacity);
        equilibrium.setVal(0);
        beta.setVal(0);
#if defined(WARPX_DIM_1D_Z)
        constexpr int normal_axis = 2;
#else
        constexpr int normal_axis = 0;
#endif
        int const axis = reflecting ? (normal_axis + 1) % 3 : normal_axis;
        beta.setVal(speed, axis, 1);
        amrex::Real volume = 1;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            volume *= geometry.CellSize(d);
        }
        auto const g2 = 1 / (1 - speed * speed);
        bool const beam = mode == "beam";
        auto const wave_number = reflecting ? 3.141592653589793_rt : 6.283185307179586_rt;
        radiation.setVal(0);
        for (amrex::MFIter iterator(radiation); iterator.isValid(); ++iterator) {
            auto const r = radiation.array(iterator);
            amrex::ParallelFor(iterator.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                auto const rest =
                    volume * (1 + 0.1_rt * std::cos(wave_number * (i + 0.5_rt) / cells));
                for (int d = 0; d < 4; ++d) {
                    r(i, j, k, d) = 0;
                }
                r(i, j, k, 0) = beam ? rest : g2 * (1 + speed * speed / 3) * rest;
                r(i, j, k, axis + 1) = beam ? rest : 4 * g2 * speed * rest / 3;
            });
        }
        radiation.FillBoundary(geometry.periodicity());
        warpx::radiation::ImplicitMomentTransportOptions options;
        options.reflecting_boundaries = reflecting;
        inputs.query("verbose", options.verbose);
        auto const dt = duration / (steps * PhysConst::c);
        amrex::MultiFab snapshot(boxes, distribution, 4, radiation.nGrowVect());
        amrex::MultiFab::Copy(snapshot, radiation, 0, 0, 4, radiation.nGrowVect());
        transfer.setVal(123);
        auto budget = options;
        budget.max_nonlinear_iterations = 1;
        auto const rejected = warpx::radiation::TryImplicitMomentTransport(
            radiation, beta, absorption, scattering, equilibrium, transfer, geometry, dt, budget);
        AMREX_ALWAYS_ASSERT(!rejected.valid);
        amrex::MultiFab::Subtract(snapshot, radiation, 0, 0, 4, radiation.nGrowVect());
        AMREX_ALWAYS_ASSERT(snapshot.norm0(0, 4, radiation.nGrowVect()) == 0);
        for (int d = 0; d < 4; ++d) {
            AMREX_ALWAYS_ASSERT(transfer.min(d, transfer.nGrow()) == 123 &&
                                transfer.max(d, transfer.nGrow()) == 123);
        }
        amrex::Real worst_energy = 0, worst_momentum = 0;
        int maximum_nonlinear = 0, maximum_linear = 0;
        for (int step = 0; step < steps; ++step) {
            auto const result = warpx::radiation::TryImplicitMomentTransport(
                radiation, beta, absorption, scattering, equilibrium, transfer, geometry, dt,
                options);
            if (!result.valid) {
                amrex::Print() << "Transport failed step=" << step
                               << " nonlinear=" << result.nonlinear_iterations
                               << " linear=" << result.linear_iterations
                               << " equation=" << result.equation_residual
                               << " energy=" << result.energy_residual
                               << " momentum=" << result.momentum_residual << '\n';
            }
            AMREX_ALWAYS_ASSERT(result.valid);
            if (reflecting) {
                AMREX_ALWAYS_ASSERT(result.boundary_exchange[0] == 0);
                AMREX_ALWAYS_ASSERT(result.boundary_exchange[axis + 1] == 0);
            }
            worst_energy = amrex::max(worst_energy, result.energy_residual);
            worst_momentum = amrex::max(worst_momentum, result.momentum_residual);
            maximum_nonlinear = amrex::max(maximum_nonlinear, result.nonlinear_iterations);
            maximum_linear = amrex::max(maximum_linear, result.linear_iterations);
        }
        auto const phase = reflecting ? 0 : wave_number * (beam ? 1 : speed) * duration;
        auto const diffusion = beam ? 0 :
            std::pow(1 - speed * speed, reflecting ? 0.5_rt : 1.5_rt) / (3 * opacity);
        auto const amplitude = 0.1_rt * std::exp(-diffusion * wave_number * wave_number * duration);
        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                         amrex::ReduceOpSum> ops;
        amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real> data(ops);
        using Tuple = typename decltype(data)::Type;
        for (amrex::MFIter iterator(radiation); iterator.isValid(); ++iterator) {
            auto const r = radiation.const_array(iterator);
            ops.eval(iterator.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                auto const slow = r(i, j, k, 0) - speed * r(i, j, k, axis + 1);
                auto const location = wave_number * (i + 0.5_rt) / cells;
                auto const exact = volume * (1 + amplitude * std::cos(location - phase));
                return {slow, 2 * slow * std::cos(location),
                        reflecting ? 0 : 2 * slow * std::sin(location), std::abs(slow - exact)};
            });
        }
        auto const values = data.value();
        amrex::Real measured[4] = {amrex::get<0>(values), amrex::get<1>(values),
                                   amrex::get<2>(values), amrex::get<3>(values)};
        amrex::ParallelDescriptor::ReduceRealSum(measured, 4);
        auto const error = std::hypot(measured[1] - amplitude * std::cos(phase),
                                      measured[2] - amplitude * std::sin(phase)) /
                           amplitude;
        AMREX_ALWAYS_ASSERT(std::abs(measured[0] - 1) < 1.e-10_rt);
        amrex::Print() << "Moving transport " << mode << " cells=" << cells << " steps=" << steps
                       << " relative Fourier error=" << error << " real=" << measured[1]
                       << " imag=" << measured[2] << " max nonlinear=" << maximum_nonlinear
                       << " max linear=" << maximum_linear << " energy=" << worst_energy
                       << " momentum=" << worst_momentum
                       << " profile_L1=" << measured[3] / amplitude << '\n';
    }
    amrex::Finalize();
}
