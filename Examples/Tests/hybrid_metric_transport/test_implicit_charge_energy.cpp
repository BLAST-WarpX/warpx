/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ImplicitChargeEnergyTransport.H"

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>

int
main (int argc, char* argv[])
{
    using namespace amrex::literals;
    amrex::Initialize(argc, argv);
    {
        bool corrupt_metric = false;
        amrex::ParmParse("test").query("corrupt_metric", corrupt_metric);
        amrex::Box const domain(amrex::IntVect(0), amrex::IntVect(3));
        amrex::RealBox const physical({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(1.0, 1.0, 1.0)});
        std::array<int, AMREX_SPACEDIM> const periodic{};
        amrex::Geometry const geometry(domain, physical, 0, periodic);
        amrex::BoxArray boxes(domain);
        boxes.maxSize(2);
        boxes.convert(amrex::IntVect::TheNodeVector());
        amrex::DistributionMapping const distribution(boxes);
        amrex::MultiFab old(boxes, distribution, 1, 0), rho(boxes, distribution, 1, 0);
        amrex::MultiFab outgoing(boxes, distribution, 1, 0);
        amrex::MultiFab volumes(boxes, distribution, 1, 0);
        amrex::MultiFab incoming(boxes, distribution, 2 * AMREX_SPACEDIM, 0);
        amrex::MultiFab result(boxes, distribution, 1, 0), error(boxes, distribution, 1, 0);
        std::array<amrex::Real, 3> const scales{1.e-12_rt, 1.0_rt, 1.e12_rt};
        for (auto const scale : scales) {
            incoming.setVal(0);
            for (amrex::MFIter mfi(old); mfi.isValid(); ++mfi) {
                auto const e = old.array(mfi);
                auto const q = rho.array(mfi);
                auto const out = outgoing.array(mfi);
                auto const in = incoming.array(mfi);
                auto const measure = volumes.array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    // Three nodes with unequal volumes 1,2,3. Initially all
                    // charge/energy is at A. Transfer A->B=1 and B->C=1/2:
                    // B has through-flow despite zero old charge, while A empties.
                    auto const volume = static_cast<amrex::Real>(i + 1);
                    measure(i, j, k) = volume * (corrupt_metric && i == 2 ? 1.01_rt : 1.0_rt);
                    e(i, j, k) = i == 0 ? 2 * scale / volume : 0;
                    q(i, j, k) = (i == 1 || i == 2) ? 0.5_rt * scale / volume : 0;
                    out(i, j, k) = (i == 0 ? scale : (i == 1 ? 0.5_rt * scale : 0)) / volume;
                    in(i, j, k, 0) = (i == 1 ? scale : (i == 2 ? 0.5_rt * scale : 0)) / volume;
                });
            }
            auto const iterations = warpx::hybrid::implicitChargeEnergyRemap(
                result, old, rho, outgoing, incoming, volumes, geometry);
            AMREX_ALWAYS_ASSERT(iterations >= 2);
            for (amrex::MFIter mfi(result); mfi.isValid(); ++mfi) {
                auto const value = result.const_array(mfi);
                auto const discrepancy = error.array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    auto const expected =
                        (i == 1 || i == 2) ? scale / static_cast<amrex::Real>(i + 1) : 0;
                    discrepancy(i, j, k) = std::isfinite(value(i, j, k)) && value(i, j, k) >= 0
                                               ? std::abs(value(i, j, k) - expected) / scale
                                               : 1;
                });
            }
            AMREX_ALWAYS_ASSERT(error.norm0(0, 0, false) < 1.e-13_rt);
        }
        amrex::Print() << "Implicit empty-node through-flow preserves positive energy, "
                          "constant specific energy and the unequal-volume inventory.\n";
    }
    amrex::Finalize();
}
