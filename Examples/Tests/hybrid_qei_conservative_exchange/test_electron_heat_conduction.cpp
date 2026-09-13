/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronHeatConduction.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>
#include <AMReX_iMultiFab.H>

#include <cmath>

using namespace amrex::literals;

namespace
{
constexpr int axial_cells = 32;
constexpr int axial_dir = AMREX_SPACEDIM - 1;
constexpr amrex::Real capacity = 0.5_rt;
constexpr amrex::Real gamma_e = 5.0_rt / 3.0_rt;

amrex::Real
inventory (amrex::MultiFab const& temperature, amrex::MultiFab const& charge,
           amrex::Geometry const& geometry)
{
    auto const dx = geometry.CellSizeArray();
    auto const hi = amrex::ubound(amrex::surroundingNodes(geometry.Domain()));
    auto const periodic = geometry.isPeriodicArray();
    auto const owner = temperature.OwnerMask(geometry.periodicity());
    amrex::MultiFab total(temperature.boxArray(), temperature.DistributionMap(), 1, 0);
    for (amrex::MFIter mfi(total); mfi.isValid(); ++mfi)
    {
        auto const out = total.array(mfi);
        auto const t = temperature.const_array(mfi), rho = charge.const_array(mfi);
        auto const owns = owner->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               amrex::Real volume = 1.0_rt;
#if defined(WARPX_DIM_RZ)
                               volume = MathConst::pi * dx[0] * dx[0] * dx[1] *
                                        (i == 0 ? 1.0_rt / 3.0_rt : 2.0_rt * i);
                               if (i == hi.x)
                               {
                                   volume *= 0.5_rt;
                               }
#else
                for (int d = 0; d < AMREX_SPACEDIM; ++d) { volume *= dx[d]; }
                amrex::ignore_unused(hi);
#endif
                               amrex::ignore_unused(periodic);
                               out(i, j, k) = owns(i, j, k) ? volume * rho(i, j, k) /
                                                                  PhysConst::q_e * PhysConst::kb /
                                                                  (gamma_e - 1.0_rt) * t(i, j, k)
                                                            : 0.0_rt;
                           });
    }
    return total.sum(0);
}
} // namespace

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::IntVect cells(AMREX_D_DECL(8, 8, 8));
        cells[axial_dir] = axial_cells;
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
        cells[0] = 16;
#endif
        amrex::Box const domain(amrex::IntVect::TheZeroVector(), cells - 1);
        amrex::RealBox const physical({AMREX_D_DECL(0, 0, 0)}, {AMREX_D_DECL(1, 1, 1)});
        amrex::Array<int, AMREX_SPACEDIM> periodic;
        periodic.fill(1);
        int coordinates = 0;
#if defined(WARPX_DIM_RZ)
        coordinates = 1;
        periodic[0] = 0;
#endif
        amrex::Geometry const geometry(domain, &physical, coordinates, periodic.data());
        amrex::BoxArray ba(domain);
        ba.maxSize(8);
        ba.convert(amrex::IntVect::TheNodeVector());
        amrex::DistributionMapping const dm(ba);
        amrex::MultiFab temperature(ba, dm, 1, 1), charge(ba, dm, 1, 1), delta(ba, dm, 1, 1);
        amrex::MultiFab saved(ba, dm, 1, 1), error(ba, dm, 1, 0);
        auto const density = capacity * (gamma_e - 1.0_rt) * PhysConst::q_e / PhysConst::kb;
        charge.setVal(density);
        temperature.setVal(-7.0_rt); // Physical ghosts must also survive a no-op unchanged.
        for (amrex::MFIter mfi(temperature); mfi.isValid(); ++mfi)
        {
            auto const t = temperature.array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   amrex::GpuArray<int, 3> const p{i, j, k};
                                   t(i, j, k) =
                                       1000.0_rt *
                                       (1.0_rt + 0.2_rt * std::cos(2.0_rt * MathConst::pi *
                                                                   p[axial_dir] / axial_cells));
                               });
        }
        amrex::Parser zero("0"), constant("2"), large("1000"), contact("1000*rho/rho0");
        for (auto* parser : {&zero, &constant, &large})
        {
            parser->registerVariables({"rho", "Te"});
        }
        contact.setConstant("rho0", density);
        contact.registerVariables({"rho", "Te"});
        bool invalid_conductivity = false;
        amrex::ParmParse("test").query("invalid_conductivity", invalid_conductivity);
        if (invalid_conductivity)
        {
            amrex::Parser invalid("-1");
            invalid.registerVariables({"rho", "Te"});
            warpx::hybrid::advanceElectronHeatConduction(temperature, delta, charge, geometry,
                                                         gamma_e, invalid.compile<2>(), 0.0_rt,
                                                         0.0005_rt, 256, 1.0_rt / 3.0_rt);
            amrex::Abort("Negative conductivity was incorrectly accepted.");
        }

        amrex::MultiFab::Copy(saved, temperature, 0, 0, 1, 1);
        auto const noop = warpx::hybrid::advanceElectronHeatConduction(
            temperature, delta, charge, geometry, gamma_e, zero.compile<2>(), 0.0_rt, 1.0_rt, 256,
            1.0_rt / 3.0_rt);
        amrex::MultiFab::Subtract(saved, temperature, 0, 0, 1, 1);
        AMREX_ALWAYS_ASSERT(noop.substeps == 0 && saved.norm0(0, 1) == 0 && delta.norm0(0, 1) == 0);

        auto const before = inventory(temperature, charge, geometry);
        constexpr amrex::Real dt = 0.0005_rt;
        for (int step = 0; step < 10; ++step)
        {
            auto const result = warpx::hybrid::advanceElectronHeatConduction(
                temperature, delta, charge, geometry, gamma_e, constant.compile<2>(), 0.0_rt, dt,
                256, 1.0_rt / 3.0_rt);
            AMREX_ALWAYS_ASSERT(result.substeps == 1);
        }
        auto const after = inventory(temperature, charge, geometry);
        auto const sine = std::sin(MathConst::pi / axial_cells);
        auto const amplitude =
            0.2_rt * std::pow(1.0_rt + 4.0_rt * 2.0_rt / capacity * dt * axial_cells * axial_cells *
                                           sine * sine,
                              -10);
        AMREX_ALWAYS_ASSERT(amplitude < 0.1_rt); // More than half the original mode decays.
        for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi)
        {
            auto const out = error.array(mfi);
            auto const t = temperature.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   amrex::GpuArray<int, 3> const p{i, j, k};
                                   auto const expected =
                                       1000.0_rt *
                                       (1.0_rt + amplitude * std::cos(2.0_rt * MathConst::pi *
                                                                      p[axial_dir] / axial_cells));
                                   out(i, j, k) = std::abs(t(i, j, k) - expected) / 1000.0_rt;
                               });
        }
        amrex::Print() << "Conduction Fourier mode: error=" << error.norm0(0, 0)
                       << " energy_residual=" << (after - before) / before << '\n';
        AMREX_ALWAYS_ASSERT(error.norm0(0, 0) < 2.e-11_rt);
        AMREX_ALWAYS_ASSERT(std::abs(after - before) < 1.e-11_rt * before);
        AMREX_ALWAYS_ASSERT(temperature.min(0) > 800.0_rt && temperature.max(0) < 1200.0_rt);

        // A two-node thermal contact bounded by exact vacuum. The transverse
        // state is uniform, so the independent two-capacity solution applies
        // in every geometry, including the RZ axis and outer half-volumes.
        for (auto const limiter : {0.0_rt, 0.001_rt})
        {
            for (amrex::MFIter mfi(temperature); mfi.isValid(); ++mfi)
            {
                auto const t = temperature.array(mfi), rho = charge.array(mfi);
                amrex::ParallelFor(mfi.validbox(),
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
                                       amrex::GpuArray<int, 3> const p{i, j, k};
                                       int const z = p[axial_dir];
                                       rho(i, j, k) =
                                           z == 8 ? 2.0_rt * density : (z == 9 ? density : 0.0_rt);
                                       t(i, j, k) = z == 8 ? 2000.0_rt : 1000.0_rt;
                                   });
            }
            auto const initial = inventory(temperature, charge, geometry);
            auto const result = warpx::hybrid::advanceElectronHeatConduction(
                temperature, delta, charge, geometry, gamma_e, contact.compile<2>(), limiter,
                1.e-6_rt, 256, 1.0_rt / 3.0_rt);
            AMREX_ALWAYS_ASSERT(result.substeps == 1);
            auto conductivity = 4000.0_rt / 3.0_rt; // Harmonic mean of 2000 and 1000.
            if (limiter > 0)
            {
                auto const thermal = PhysConst::kb * 1500.0_rt;
                auto const saturated_flux = limiter * density / PhysConst::q_e * thermal *
                                            std::sqrt(thermal / PhysConst::m_e);
                conductivity =
                    1.0_rt / (1.0_rt / conductivity + 1000.0_rt * axial_cells / saturated_flux);
            }
            auto const difference = 1000.0_rt / (1.0_rt + 1.5_rt * conductivity * 1.e-6_rt *
                                                              axial_cells * axial_cells / capacity);
            for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi)
            {
                auto const out = error.array(mfi);
                auto const t = temperature.const_array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::GpuArray<int, 3> const p{i, j, k};
                        int const z = p[axial_dir];
                        auto const expected =
                            z == 8
                                ? (5000.0_rt + difference) / 3.0_rt
                                : (z == 9 ? (5000.0_rt - 2.0_rt * difference) / 3.0_rt : 1000.0_rt);
                        out(i, j, k) = std::abs(t(i, j, k) - expected) / 2000.0_rt;
                    });
            }
            amrex::Print() << "Conduction contact: limiter=" << limiter
                           << " error=" << error.norm0(0, 0) << '\n';
            AMREX_ALWAYS_ASSERT(error.norm0(0, 0) < 2.e-11_rt);
            AMREX_ALWAYS_ASSERT(std::abs(inventory(temperature, charge, geometry) - initial) <
                                1.e-11_rt * initial);
        }
        temperature.setVal(1234.0_rt);
        auto const uniform = warpx::hybrid::advanceElectronHeatConduction(
            temperature, delta, charge, geometry, gamma_e, large.compile<2>(), 0.05_rt, 1.0_rt, 256,
            1.0_rt / 3.0_rt);
        AMREX_ALWAYS_ASSERT(uniform.substeps == 0 && temperature.min(0, 1) == 1234.0_rt &&
                            temperature.max(0, 1) == 1234.0_rt && delta.norm0(0, 1) == 0.0_rt);
        amrex::Print()
            << "Electron conduction: Fourier, contact, limiter, vacuum and exact no-op PASS\n";
    }
    amrex::Finalize();
}
