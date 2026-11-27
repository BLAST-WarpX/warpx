/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/RZPressureInduction.H"
#include "Utils/WarpXConst.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <array>
#include <cmath>

using namespace amrex::literals;

namespace
{
// Independent actual-state inventory; no source ledger or solver residual.
std::array<long double, 4>
inventory (amrex::MultiFab const& b, amrex::MultiFab const& u, amrex::MultiFab const& volume,
           amrex::Geometry const& geom, amrex::Real first_stage_dt = 0)
{
    std::array<long double, 4> total{};
    auto const owner = b.OwnerMask(geom.periodicity());
    int const nr = geom.Domain().length(0), nz = geom.Domain().length(1);
    auto const dr = geom.CellSize(0), dz = geom.CellSize(1);
    for (amrex::MFIter it(b); it.isValid(); ++it)
    {
        auto const copy = [&] (amrex::MultiFab const& field)
        {
            amrex::FArrayBox host(field[it].box(), 1, amrex::The_Pinned_Arena());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, field[it].dataPtr(),
                             field[it].dataPtr() + field[it].box().numPts(), host.dataPtr());
            return host;
        };
        auto const bh = copy(b), uh = copy(u), vh = copy(volume);
        amrex::IArrayBox mask((*owner)[it].box(), 1, amrex::The_Pinned_Arena());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, (*owner)[it].dataPtr(),
                         (*owner)[it].dataPtr() + (*owner)[it].box().numPts(), mask.dataPtr());
        auto const box = it.validbox();
        for (int j = box.smallEnd(1); j <= box.bigEnd(1) && j < nz; ++j)
        {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i)
            {
                amrex::IntVect const node(i, j);
                if (!mask(node))
                {
                    continue;
                }
                long double const magnetic_volume =
                    i == 0 ? MathConst::pi * dr * dr * dz / 4
                           : 2 * MathConst::pi * i * dr * dr * dz * (i == nr ? .5L : 1.L);
                long double const field = bh(node);
                total[0] += magnetic_volume * field * field / (2 * PhysConst::mu0);
                total[1] += static_cast<long double>(vh(node)) * uh(node);
                if (first_stage_dt > 0)
                {
                    long double const r = static_cast<long double>(i) / nr;
                    long double const phase = 2 * MathConst::pi * j / nz;
                    long double const shape = std::max(0.L, 1 - 4 * r * r);
                    long double const pressure_dr = -(64.L / 3) * 1.e9L * r / geom.ProbHi(0) *
                                                    shape * shape * shape *
                                                    (1 + .2L * std::cos(phase));
                    long double const reciprocal_dz =
                        -.2L * 2 * MathConst::pi / geom.ProbLength(1) * std::cos(phase) /
                        (1.e7L * std::pow(1 + .2L * std::sin(phase), 2));
                    long double const expected = first_stage_dt * reciprocal_dz * pressure_dr;
                    long double const error = field - 5 * r * (1 - r * r) - expected;
                    total[2] += magnetic_volume * error * error;
                    total[3] += magnetic_volume * expected * expected;
                }
            }
        }
    }
#ifdef AMREX_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, total.data(), 4, MPI_LONG_DOUBLE, MPI_SUM,
                  amrex::ParallelDescriptor::Communicator());
#endif
    return total;
}
} // namespace

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        int cells = 32, max_grid = 32, steps = 128;
        bool constant_charge = false;
        amrex::ParmParse pp("test");
        pp.query("cells", cells);
        pp.query("max_grid", max_grid);
        pp.query("steps", steps);
        pp.query("constant_charge", constant_charge);
        amrex::Box const domain(amrex::IntVect(0), amrex::IntVect(cells - 1));
        amrex::RealBox const physical({0, 0}, {1.e-3_rt, 1.e-3_rt});
        int periodic[2]{0, 1};
        amrex::Geometry const geom(domain, &physical, 1, periodic);
        amrex::BoxArray boxes(domain);
        boxes.maxSize(max_grid);
        boxes.convert(amrex::IntVect::TheNodeVector());
        amrex::DistributionMapping const mapping(boxes);
        amrex::MultiFab b(boxes, mapping, 1, 1), u(boxes, mapping, 1, 1), p(boxes, mapping, 1, 1);
        amrex::MultiFab rho(boxes, mapping, 1, 1), volume(boxes, mapping, 1, 1);
        for (auto* field : {&b, &u, &p, &rho, &volume})
        {
            field->setVal(-123);
        }
        auto const dr = geom.CellSize(0), dz = geom.CellSize(1);
        for (amrex::MFIter it(b); it.isValid(); ++it)
        {
            auto const bf = b.array(it), uf = u.array(it), pf = p.array(it);
            auto const rf = rho.array(it), vf = volume.array(it);
            amrex::ParallelFor(
                it.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    auto const r = static_cast<amrex::Real>(i) / cells;
                    auto const phase = 2 * MathConst::pi * j / cells;
                    auto const shape = amrex::max(0._rt, 1 - 4 * r * r);
                    uf(i, j, k) = 1.e9_rt * (2 + shape * shape * shape * shape *
                                                     (1 + .2_rt * std::cos(phase)));
                    pf(i, j, k) = (2._rt / 3) * uf(i, j, k);
                    rf(i, j, k) = 1.e7_rt * (constant_charge ? 1 : 1 + .2_rt * std::sin(phase));
                    bf(i, j, k) = constant_charge ? 0 : 5 * r * (1 - r * r);
                    vf(i, j, k) = i == 0 ? MathConst::pi * dr * dr * dz / 3
                                         : 2 * MathConst::pi * i * dr * dr * dz *
                                               (i == cells ? .5_rt : 1._rt);
                });
        }
        auto const initial = inventory(b, u, volume, geom);
        amrex::Real const dt = 1.e-10_rt;
        for (int stage = 0; stage < steps; ++stage)
        {
            AMREX_ALWAYS_ASSERT(
                warpx::hybrid::tryAdvanceRZPressureInduction(b, u, p, rho, volume, geom, dt));
            if (stage == 0 && !constant_charge)
            {
                auto const analytic = inventory(b, u, volume, geom, dt);
                auto const error = std::sqrt(analytic[2] / analytic[3]);
                amrex::Print() << "RZ Biermann first-stage relative L2 error=" << error << '\n';
                AMREX_ALWAYS_ASSERT(error < .05L);
            }
            amrex::MultiFab::Copy(p, u, 0, 0, 1, 0);
            p.mult(2._rt / 3, 0, 1, 0);
        }
        auto const final = inventory(b, u, volume, geom);
        long double const defect = (final[0] - initial[0]) + (final[1] - initial[1]);
        long double const transfer = std::abs(final[0] - initial[0]);
        amrex::Print() << "RZ pressure induction magnetic transfer=" << final[0] - initial[0]
                       << " caloric transfer=" << final[1] - initial[1]
                       << " energy defect=" << defect << '\n';
        AMREX_ALWAYS_ASSERT(std::abs(defect) < 1.e-11L * (initial[0] + initial[1]));
        if (constant_charge)
        {
            AMREX_ALWAYS_ASSERT(b.norm0(0) < 1.e-10_rt);
        }
        else
        {
            AMREX_ALWAYS_ASSERT(transfer > 1.e-6L * initial[1]);
            AMREX_ALWAYS_ASSERT(std::abs(defect) < 1.e-8L * transfer);
            amrex::MultiFab saved_b(boxes, mapping, 1, 1), saved_u(boxes, mapping, 1, 1);
            amrex::MultiFab::Copy(saved_b, b, 0, 0, 1, 1);
            amrex::MultiFab::Copy(saved_u, u, 0, 0, 1, 1);
            AMREX_ALWAYS_ASSERT(!warpx::hybrid::tryAdvanceRZPressureInduction(b, u, p, rho, volume,
                                                                              geom, 1.e6_rt * dt));
            amrex::MultiFab::Subtract(saved_b, b, 0, 0, 1, 1);
            amrex::MultiFab::Subtract(saved_u, u, 0, 0, 1, 1);
            AMREX_ALWAYS_ASSERT(saved_b.norm0(0, 1) == 0 && saved_u.norm0(0, 1) == 0);
        }
    }
    amrex::Finalize();
}
