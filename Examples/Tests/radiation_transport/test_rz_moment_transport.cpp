/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/ImplicitMomentTransport.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

using namespace amrex::literals;

namespace
{
double
BesselJ1 (double x)
{
    // Convergent small-argument series on the fixture's [0, first J1 zero]
    // interval; no platform-specific special-function library is required.
    AMREX_ALWAYS_ASSERT(x >= 0 && x < 4);
    double term = x / 2, sum = term;
    for (int k = 1; k < 32; ++k)
    {
        term *= -(x * x / 4) / (k * (k + 1));
        sum += term;
    }
    return sum;
}

struct Problem
{
    amrex::Geometry geometry;
    amrex::BoxArray boxes;
    amrex::DistributionMapping distribution;
    amrex::MultiFab radiation, beta, absorption, scattering, equilibrium, material, heat;

    Problem (int nr, int nz, bool periodic_z)
        : geometry(amrex::Box(amrex::IntVect(0, 0), amrex::IntVect(nr - 1, nz - 1)),
                   amrex::RealBox({0._rt, 0._rt}, {1._rt, 1._rt}), 0,
                   std::array<int, 2>{0, periodic_z ? 1 : 0}),
          boxes(geometry.Domain())
    {
        boxes.maxSize(4);
        distribution.define(boxes);
        radiation.define(boxes, distribution, 4, 1);
        beta.define(boxes, distribution, 3, 0);
        absorption.define(boxes, distribution, 1, 0);
        scattering.define(boxes, distribution, 1, 0);
        equilibrium.define(boxes, distribution, 1, 0);
        material.define(boxes, distribution, 4, 1);
        heat.define(boxes, distribution, 1, 1);
        radiation.setVal(0);
        beta.setVal(0);
        absorption.setVal(0);
        scattering.setVal(0);
        equilibrium.setVal(0);
        material.setVal(0);
        heat.setVal(0);
    }

    void
    Initialize (bool radial)
    {
        auto const spacing = geometry.CellSizeArray();
        int const nr = geometry.Domain().length(0);
        amrex::Gpu::HostVector<amrex::Real> host(nr);
        constexpr double root = 3.8317059702075123156;
        AMREX_ALWAYS_ASSERT(std::abs(BesselJ1(root)) < 1.e-14);
        for (int i = 0; i < nr; ++i)
        {
            double const lo = i * spacing[0], hi = (i + 1) * spacing[0];
            // Exact annular cell average of J0, not a midpoint reference.
            host[i] = radial ? 2 * (hi * BesselJ1(root * hi) - lo * BesselJ1(root * lo)) /
                                   (root * (hi * hi - lo * lo))
                             : 0;
        }
        amrex::Gpu::DeviceVector<amrex::Real> profile(nr);
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, host.begin(), host.end(), profile.begin());
        auto const* values = profile.data();
        for (amrex::MFIter it(radiation); it.isValid(); ++it)
        {
            auto const u = radiation.array(it);
            amrex::ParallelFor(it.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   auto const lo = i * spacing[0], hi = (i + 1) * spacing[0];
                                   auto const volume =
                                       MathConst::pi * (hi * hi - lo * lo) * spacing[1];
                                   u(i, j, k, 0) = volume * (1 + 0.01_rt * values[i]);
                               });
        }
        amrex::Gpu::streamSynchronize();
    }

    void
    InitializeFlow (bool compression)
    {
        auto const dx = geometry.CellSizeArray();
        for (amrex::MFIter it(radiation); it.isValid(); ++it)
        {
            auto const u = radiation.array(it);
            auto const b = beta.array(it);
            amrex::ParallelFor(
                it.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    auto const lo = i * dx[0], hi = (i + 1) * dx[0];
                    auto const r = (i + 0.5_rt) * dx[0], z = (j + 0.5_rt) * dx[1];
                    auto const volume = MathConst::pi * (hi * hi - lo * lo) * dx[1];
                    auto const velocity =
                        compression ? -1.e-3_rt * std::sin(MathConst::pi * r) : 3.e-3_rt;
                    auto const gamma2 = 1 / (1 - velocity * velocity);
                    auto const sinc = std::sin(MathConst::pi * dx[1]) / (MathConst::pi * dx[1]);
                    auto const energy =
                        compression ? 1._rt : 1 + 0.01_rt * sinc * std::cos(2 * MathConst::pi * z);
                    int const direction = compression ? 0 : 2;
                    b(i, j, k, direction) = velocity;
                    u(i, j, k, 0) = volume * gamma2 * (1 + velocity * velocity / 3) * energy;
                    u(i, j, k, direction + 1) = volume * (4._rt / 3) * gamma2 * velocity * energy;
                });
        }
    }
};

void
WriteRadialProfile (Problem const& p, std::string const& filename)
{
    int const nr = p.geometry.Domain().length(0);
    std::vector<amrex::Real> values(4 * nr, 0);
    for (amrex::MFIter it(p.radiation); it.isValid(); ++it)
    {
        auto const& fab = p.radiation[it];
        amrex::FArrayBox host(fab.box(), 4, amrex::The_Pinned_Arena());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, fab.dataPtr(), fab.dataPtr() + fab.size(),
                         host.dataPtr());
        auto const u = host.const_array();
        auto const lo = amrex::lbound(it.validbox()), hi = amrex::ubound(it.validbox());
        for (int j = lo.y; j <= hi.y; ++j)
        {
            for (int i = lo.x; i <= hi.x; ++i)
            {
                for (int d = 0; d < 4; ++d)
                {
                    values[4 * i + d] += u(i, j, 0, d);
                }
            }
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(values.data(), static_cast<int>(values.size()));
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::ofstream output(filename);
        output << "r_lo,r_hi,E,Qr,Qtheta,Qz\n" << std::setprecision(17);
        auto const dr = p.geometry.CellSize(0);
        for (int i = 0; i < nr; ++i)
        {
            auto const lo = i * dr, hi = (i + 1) * dr;
            auto const volume = MathConst::pi * (hi * hi - lo * lo);
            output << lo << ',' << hi;
            for (int d = 0; d < 4; ++d)
            {
                output << ',' << values[4 * i + d] / volume;
            }
            output << '\n';
        }
        AMREX_ALWAYS_ASSERT(output.good());
    }
}

amrex::Real
Flow (bool compression, int cells, std::string const& output)
{
    using namespace warpx::radiation;
    Problem p(compression ? cells : 4, compression ? 4 : cells, true);
    p.InitializeFlow(compression);
    p.scattering.setVal(1000);
    auto const initial_energy = p.radiation.sum(0);
    amrex::Real material_energy = 0, absolute_work = 0;
    constexpr int steps = 128;
    auto const ct = compression ? 30._rt : 100._rt;
    auto const dt = ct / (steps * PhysConst::c);
    ImplicitMomentTransportOptions options;
    options.reflecting_boundaries = true;
    for (int step = 0; step < steps; ++step)
    {
        auto const result =
            TryImplicitMomentTransport(p.radiation, p.beta, p.absorption, p.scattering,
                                       p.equilibrium, p.material, p.geometry, dt, options, &p.heat);
        if (!result.valid)
        {
            amrex::Print() << "RZ moving moment failure: step=" << step
                           << " equation=" << result.equation_residual
                           << " energy=" << result.energy_residual
                           << " momentum=" << result.momentum_residual << '\n';
        }
        AMREX_ALWAYS_ASSERT(result.valid);
        material_energy += p.material.sum(0);
        absolute_work += p.material.norm1(0);
        AMREX_ALWAYS_ASSERT(std::abs(p.radiation.sum(0) + material_energy - initial_energy) <
                            1.e-10_rt * initial_energy);
        AMREX_ALWAYS_ASSERT(p.heat.norm0(0) < 1.e-10_rt * initial_energy);
    }
    AMREX_ALWAYS_ASSERT(absolute_work > 1.e-8_rt * initial_energy);
    amrex::Print() << "RZ moving moments: material work=" << material_energy
                   << " absolute work=" << absolute_work << '\n';
    if (compression)
    {
        AMREX_ALWAYS_ASSERT(material_energy < -1.e-6_rt * initial_energy);
        WriteRadialProfile(p, output);
        return 0;
    }
    auto const dx = p.geometry.CellSizeArray();
    auto const amplitude = std::exp(-4 * MathConst::pi * MathConst::pi * ct / 3000);
    constexpr amrex::Real beta = 3.e-3_rt;
    constexpr amrex::Real lab_factor = (1 + beta * beta / 3) / (1 - beta * beta);
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Real, amrex::Real> data(ops);
    using Tuple = typename decltype(data)::Type;
    for (amrex::MFIter it(p.radiation); it.isValid(); ++it)
    {
        auto const u = p.radiation.const_array(it);
        ops.eval(it.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple
                 {
                     auto const lo = i * dx[0], hi = (i + 1) * dx[0];
                     auto const volume = MathConst::pi * (hi * hi - lo * lo) * dx[1];
                     auto const sinc = std::sin(MathConst::pi * dx[1]) / (MathConst::pi * dx[1]);
                     auto const mode =
                         lab_factor * 0.01_rt * amplitude * sinc *
                         std::cos(2 * MathConst::pi * ((j + 0.5_rt) * dx[1] - beta * ct));
                     auto const error = u(i, j, k, 0) / volume - lab_factor - mode;
                     return {volume * error * error, volume * mode * mode};
                 });
    }
    auto const sums = data.value();
    amrex::Real values[2]{amrex::get<0>(sums), amrex::get<1>(sums)};
    amrex::ParallelDescriptor::ReduceRealSum(values, 2);
    auto const error = std::sqrt(values[0] / values[1]);
    amrex::Print() << "RZ trapped axial mode: cells=" << cells << " relative error=" << error
                   << '\n';
    return error;
}

void
Rejection (Problem& p)
{
    using namespace warpx::radiation;
    amrex::MultiFab before(p.boxes, p.distribution, 4, 1);
    amrex::MultiFab increment(p.boxes, p.distribution, 8, 1);
    for (int kind = 0; kind < 3; ++kind)
    {
        p.radiation.setVal(0);
        p.beta.setVal(0);
        p.Initialize(kind == 2);
        p.scattering.setVal(1000);
        if (kind == 0)
        {
            p.beta.setVal(1.e-3_rt, 1, 1);
        }
        if (kind == 1)
        {
            amrex::MultiFab::Copy(p.radiation, p.radiation, 0, 2, 1, 0);
            p.radiation.mult(1.e-3_rt, 2, 1, 0);
        }
        amrex::MultiFab::Copy(before, p.radiation, 0, 0, 4, 1);
        p.material.setVal(17);
        p.heat.setVal(19);
        increment.setVal(23);
        ImplicitMomentTransportOptions options;
        options.reflecting_boundaries = true;
        if (kind == 2)
        {
            options.max_nonlinear_iterations = 1;
            options.max_linear_iterations = 1;
        }
        auto const result = TryImplicitMomentTransport(
            p.radiation, p.beta, p.absorption, p.scattering, p.equilibrium, p.material, p.geometry,
            1 / PhysConst::c, options, &p.heat);
        AMREX_ALWAYS_ASSERT(!result.valid);
        for (int d = 0; d < 4; ++d)
        {
            AMREX_ALWAYS_ASSERT(result.boundary_exchange[d] == 0);
            AMREX_ALWAYS_ASSERT(result.geometric_exchange[d] == 0);
        }
        if (kind < 2)
        {
            AMREX_ALWAYS_ASSERT(!ComputeMomentTransportIncrement(
                p.radiation, p.beta, p.absorption, p.scattering, p.equilibrium, increment,
                p.geometry, 1 / PhysConst::c, true));
            for (int d = 0; d < 8; ++d)
            {
                AMREX_ALWAYS_ASSERT(increment.min(d, 1) == 23 && increment.max(d, 1) == 23);
            }
        }
        else
        {
            AMREX_ALWAYS_ASSERT(result.nonlinear_iterations == 1);
        }
        amrex::MultiFab::Subtract(before, p.radiation, 0, 0, 4, 1);
        for (int d = 0; d < 4; ++d)
        {
            AMREX_ALWAYS_ASSERT(before.norm0(d, 1) == 0);
            AMREX_ALWAYS_ASSERT(p.material.min(d, 1) == 17 && p.material.max(d, 1) == 17);
        }
        AMREX_ALWAYS_ASSERT(p.heat.min(0, 1) == 19 && p.heat.max(0, 1) == 19);
    }
}

void
ThermalBath (amrex::Real speed)
{
    using namespace warpx::radiation;
    Problem p(8, 4, true);
    p.Initialize(false);
    p.beta.setVal(speed, 2, 1);
    p.absorption.setVal(1000);
    amrex::MultiFab::Copy(p.equilibrium, p.radiation, 0, 0, 1, 0);
    p.equilibrium.mult(2);
    auto const initial_energy = p.radiation.sum(0);
    amrex::Real exchange = 0;
    ImplicitMomentTransportOptions options;
    options.reflecting_boundaries = true;
    for (int step = 0; step < 64; ++step)
    {
        auto const result = TryImplicitMomentTransport(
            p.radiation, p.beta, p.absorption, p.scattering, p.equilibrium, p.material, p.geometry,
            0.25_rt / (1000 * PhysConst::c), options, &p.heat);
        AMREX_ALWAYS_ASSERT(result.valid);
        exchange += p.material.sum(0);
        AMREX_ALWAYS_ASSERT(std::abs(p.radiation.sum(0) + exchange - initial_energy) <
                            1.e-10_rt * initial_energy);
    }
    auto const gamma2 = 1 / (1 - speed * speed);
    auto const expected_energy = 2 * gamma2 * (1 + speed * speed / 3) * initial_energy;
    auto const expected_momentum = 2 * (4._rt / 3) * gamma2 * speed * initial_energy;
    AMREX_ALWAYS_ASSERT(std::abs(p.radiation.sum(0) - expected_energy) <
                        1.e-5_rt * expected_energy);
    AMREX_ALWAYS_ASSERT(std::abs(p.radiation.sum(3) - expected_momentum) <
                        1.e-5_rt *
                            amrex::max(std::abs(expected_momentum), 1.e-12_rt * initial_energy));
    amrex::Print() << "RZ LTE bath: beta_z=" << speed << " material exchange=" << exchange << '\n';
}

void
Run (bool radial, int nr, bool periodic_z)
{
    using namespace warpx::radiation;
    Problem p(nr, 4, periodic_z);
    p.Initialize(radial);
    auto const initial_energy = p.radiation.sum(0);
    constexpr amrex::Real opacity = 1000;
    constexpr amrex::Real root = 3.8317059702075123156_rt;
    int steps = radial ? 64 : 16;
    amrex::ParmParse("test").query("steps", steps);
    auto const ct = radial ? 3 * opacity * std::log(2._rt) / (root * root)
                           : 0.1_rt * p.geometry.CellSize(0) * steps;
    auto const dt = ct / (PhysConst::c * steps);
    if (radial)
    {
        p.scattering.setVal(opacity);
    }
    ImplicitMomentTransportOptions options;
    options.reflecting_boundaries = true;
        amrex::ParmParse("test").query("verbose", options.verbose);
        {
            amrex::MultiFab increment(p.boxes, p.distribution, 8, 0);
            MomentTransportAccounting accounting;
            AMREX_ALWAYS_ASSERT(ComputeMomentTransportIncrement(p.radiation, p.beta,
                p.absorption, p.scattering, p.equilibrium, increment, p.geometry, dt,
                true, &accounting));
            // No axial variation: exact absence of a spurious axial force,
            // including when radial pressure varies and CUDA uses fused math.
            AMREX_ALWAYS_ASSERT(increment.norm0(3) == 0);
            AMREX_ALWAYS_ASSERT(accounting.boundary[0] == 0 && accounting.geometric[0] == 0);
        }
        amrex::Real material_energy = 0;
    for (int step = 0; step < steps; ++step)
    {
        auto const result =
            TryImplicitMomentTransport(p.radiation, p.beta, p.absorption, p.scattering,
                                       p.equilibrium, p.material, p.geometry, dt, options, &p.heat);
        amrex::Print() << "RZ M1 step=" << step << " valid=" << result.valid
                       << " equation=" << result.equation_residual << '\n';
        AMREX_ALWAYS_ASSERT(result.valid);
        material_energy += p.material.sum(0);
        AMREX_ALWAYS_ASSERT(std::abs(p.radiation.sum(0) + material_energy - initial_energy) <
                            1.e-10_rt * initial_energy);
        AMREX_ALWAYS_ASSERT(result.boundary_exchange[0] == 0);
        if (!radial)
        {
            auto const expected = dt * PhysConst::c * 2 * MathConst::pi / 3;
            AMREX_ALWAYS_ASSERT(std::abs(result.geometric_exchange[1] / expected - 1) < 1.e-12_rt);
            AMREX_ALWAYS_ASSERT(std::abs(result.boundary_exchange[1] / expected - 1) < 1.e-12_rt);
            for (int d = 1; d < 4; ++d)
            {
                AMREX_ALWAYS_ASSERT(p.radiation.norm0(d) == 0);
            }
        }
    }
    if (radial)
    {
        auto const spacing = p.geometry.CellSizeArray();
        auto const discriminant = std::sqrt(opacity * opacity - 4 * root * root / 3);
        auto const slow = -(2 * root * root / 3) / (opacity + discriminant);
        auto const fast = -opacity - slow;
        auto const amplitude =
            (-fast * std::exp(slow * ct) + slow * std::exp(fast * ct)) / (slow - fast);
        amrex::MultiFab initial(p.boxes, p.distribution, 4, 1);
        amrex::MultiFab::Copy(initial, p.radiation, 0, 0, 4, 0);
        p.Initialize(true);
        amrex::Real squared_error = 0, squared_reference = 0;
        for (amrex::MFIter it(p.radiation); it.isValid(); ++it)
        {
            auto const final = initial.const_array(it);
            auto const start = p.radiation.const_array(it);
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> ops;
            amrex::ReduceData<amrex::Real, amrex::Real> data(ops);
            using Tuple = typename decltype(data)::Type;
            ops.eval(it.validbox(), data,
                     [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple
                     {
                         auto const lo = i * spacing[0], hi = (i + 1) * spacing[0];
                         auto const volume = MathConst::pi * (hi * hi - lo * lo) * spacing[1];
                         auto const mode = amplitude * (start(i, j, k, 0) / volume - 1);
                         auto const error = final(i, j, k, 0) / volume - 1 - mode;
                         return {volume * error * error, volume * mode * mode};
                     });
            auto const values = data.value();
            squared_error += amrex::get<0>(values);
            squared_reference += amrex::get<1>(values);
        }
        amrex::ParallelDescriptor::ReduceRealSum(squared_error);
        amrex::ParallelDescriptor::ReduceRealSum(squared_reference);
        auto const error = std::sqrt(squared_error / squared_reference);
        amrex::Print() << "RZ radial diffusion mode relative error=" << error << '\n';
        AMREX_ALWAYS_ASSERT(error < 0.03_rt);
    }
    Rejection(p);
}
} // namespace

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        std::string kind = "uniform";
        int nr = 16;
        amrex::ParmParse pp("test");
        pp.query("case", kind);
        pp.query("nr", nr);
        if (kind == "lte")
        {
            ThermalBath(0);
            ThermalBath(3.e-3_rt);
        }
        else if (kind == "axial")
        {
            auto const coarse = Flow(false, 64, "");
            auto const fine = Flow(false, 128, "");
            AMREX_ALWAYS_ASSERT(coarse < 0.15_rt && fine < 0.08_rt && fine < 0.75_rt * coarse);
        }
        else if (kind == "compression")
        {
            std::string output = "rz_m1_compression.csv";
            pp.query("output", output);
            Flow(true, nr, output);
        }
        else
        {
            Run(kind == "radial", nr, true);
            if (kind == "uniform")
            {
                Run(false, nr, false);
            }
        }
    }
    amrex::Finalize();
}
