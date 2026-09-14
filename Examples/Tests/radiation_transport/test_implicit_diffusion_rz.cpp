/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/ImplicitDiffusion.H"
#include "Radiation/RadiationTransport.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>

using namespace amrex::literals;

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Box const domain(amrex::IntVect(0), amrex::IntVect(0));
        amrex::RealBox const physical({0.0_rt, 0.0_rt}, {1.0_rt, 1.0_rt});
        int periodic[] = {0, 0};
        amrex::Geometry const geometry(domain, &physical, 1, periodic);
        amrex::BoxArray const boxes(domain);
        amrex::DistributionMapping const distribution(boxes);
        amrex::MultiFab energy(boxes, distribution, 1, 1);
        amrex::MultiFab opacity(boxes, distribution, 1, 1);
        amrex::Parser parser("1+z");
        parser.registerVariables({"x", "y", "z", "t"});
        amrex::Vector<amrex::ParserExecutor<4>> host(4, parser.compile<4>());
        amrex::Gpu::DeviceVector<amrex::ParserExecutor<4>> device(4);
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, host.begin(), host.end(), device.begin());
        amrex::Real const tolerance =
            amrex::max(1.0e-10_rt, 64.0_rt * std::numeric_limits<amrex::Real>::epsilon());
        using Boundary = RadiationTransport::DiffusionBoundary;
        for (bool const bath : {false, true}) {
            int const code = static_cast<int>(bath ? Boundary::MarshakBath : Boundary::Marshak);
            warpx::radiation::ImplicitDiffusionOptions const options{
                tolerance,
                amrex::max(1.0e-13_rt, 8.0_rt * std::numeric_limits<amrex::Real>::epsilon()),
                1.0_rt,
                100,
                100,
                0,
                {0, code, 0},
                {code, code, 0},
                {0, 0, 0, 0, 0, 0},
                device.dataPtr(),
                {nullptr, nullptr, 1}};
            energy.setVal(MathConst::pi);
            opacity.setVal(20.0_rt);
            amrex::Real const dt = 1.0e-9_rt;
            auto const result = warpx::radiation::AdvanceImplicitDiffusion(
                energy, opacity, geometry, 0.0_rt, dt, options);
            amrex::Real const conductance = bath ? PhysConst::c / 32.0_rt : PhysConst::c / 2.0_rt;
            amrex::Real const expected = MathConst::pi *
                                         (1.0_rt + (bath ? 6.0_rt * dt * conductance : 0.0_rt)) /
                                         (1.0_rt + 4.0_rt * dt * conductance);
            amrex::Real const actual = energy.sum(0, false);
            AMREX_ALWAYS_ASSERT(std::abs(actual / expected - 1.0_rt) < 10 * tolerance);
            AMREX_ALWAYS_ASSERT(std::abs(actual + result.escaped_energy - result.injected_energy +
                                         result.numerical_energy_residual - MathConst::pi) <
                                10 * tolerance);
            AMREX_ALWAYS_ASSERT(result.maximum_relative_residual <= tolerance);
            amrex::Print() << "RZ implicit " << (bath ? "bath" : "escape")
                           << " relative error=" << actual / expected - 1.0_rt << '\n';
        }
    }
    amrex::Finalize();
}
