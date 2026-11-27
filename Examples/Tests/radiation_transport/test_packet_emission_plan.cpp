/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Radiation/FluxRelaxedPackets.H"
#include "Radiation/PhotonPacketTransaction.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>

using namespace amrex::literals;

namespace
{
bool
SameInventory (amrex::GpuArray<amrex::Real, 4> const& lhs,
               amrex::GpuArray<amrex::Real, 4> const& rhs)
{
    for (int d = 0; d < 4; ++d) {
        if (lhs[d] != rhs[d]) { return false; }
    }
    return true;
}

amrex::GpuArray<amrex::Real, 6>
NativePressure (WarpXParticleContainer::Base& photons)
{
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real,
                      amrex::Real, amrex::Real, amrex::Real> data(ops);
    for (WarpXParIter iterator(photons, 0); iterator.isValid(); ++iterator) {
        auto const p = iterator.GetParticleTile().getParticleTileData();
        ops.eval(iterator.numParticles(), data,
            [=] AMREX_GPU_DEVICE(long ip) -> decltype(data)::Type
            {
                if (!amrex::ParticleIDWrapper{p.m_idcpu[ip]}.is_valid() ||
                    !(p.m_rdata[PIdx::w][ip] > 0)) { return {}; }
                auto const ux = p.m_rdata[PIdx::ux][ip];
                auto const uy = p.m_rdata[PIdx::uy][ip];
                auto const uz = p.m_rdata[PIdx::uz][ip];
                auto const speed = std::hypot(std::hypot(ux, uy), uz);
                auto const coefficient =
                    p.m_rdata[PIdx::w][ip] * PhysConst::m_e * PhysConst::c / speed;
                return {coefficient * ux * ux, coefficient * uy * uy,
                        coefficient * uz * uz, coefficient * ux * uy,
                        coefficient * ux * uz, coefficient * uy * uz};
            });
    }
    auto const value = data.value();
    amrex::GpuArray<amrex::Real, 6> result{
        amrex::get<0>(value), amrex::get<1>(value), amrex::get<2>(value),
        amrex::get<3>(value), amrex::get<4>(value), amrex::get<5>(value)};
    amrex::ParallelDescriptor::ReduceRealSum(result.data(), 6);
    return result;
}

void
CheckNearBeamEmission (WarpXParticleContainer& live, amrex::Geometry const& geometry)
{
    using namespace warpx::radiation::detail;
    amrex::GpuArray<amrex::Real, 4> initial{};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(PhotonInventory(live, initial), "Invalid native baseline.");
    auto const initial_pressure = NativePressure(live);
    amrex::Real const total_energy = 1.e8_rt;
    amrex::GpuArray<amrex::Real, 3> const direction{0.6_rt, 0.64_rt, 0.48_rt};
    auto const& boxes = live.ParticleBoxArray(0);
    auto const& distribution = live.ParticleDistributionMap(0);
    amrex::MultiFab radiation(boxes, distribution, 4, 1);
    amrex::MultiFab opacity(boxes, distribution, 1, 0);
    opacity.setVal(0);
    auto const cell_energy = total_energy / static_cast<amrex::Real>(boxes.numPts());
    for (auto fraction : {0.98_rt, 0.999999_rt, 1._rt,
                         1._rt + 4 * std::numeric_limits<amrex::Real>::epsilon()}) {
        radiation.setVal(cell_energy, 0, 1);
        for (int d = 0; d < 3; ++d) {
            radiation.setVal(cell_energy * fraction * direction[d], d + 1, 1);
        }
        EmissionPlan emission;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            BuildThinCellEmissionPlan(radiation, opacity, geometry, geometry.CellSize(0),
                                      1._rt, 0._rt, 1.e-19_rt, emission),
            "Existing M1 reconstruction rejected an oblique near-beam state.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            radiation.norm0(0, 4, amrex::IntVect(1)) == 0,
            "Converted field inventory was not cleared.");
        warpx::radiation::PhotonPacketTransaction trial(live);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(StageEmissionPlan(trial, emission) && trial.ValidateLive(),
                                         "Near-beam native emission staging failed.");
        amrex::GpuArray<amrex::Real, 4> actual{};
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(PhotonInventory(trial.Particles(), actual),
                                         "Invalid actual near-beam native inventory.");
        for (int d = 0; d < 4; ++d) {
            long double const added = d == 0 ? total_energy :
                static_cast<long double>(total_energy) * fraction * direction[d - 1];
            long double const scale = std::abs(initial[d]) + total_energy;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::abs(static_cast<long double>(actual[d]) - initial[d] - added) <=
                    2.e-13L * scale,
                "Actual near-beam native energy/momentum differs from requested moments.");
        }
        auto const actual_pressure = NativePressure(trial.Particles());
        long double const f = fraction;
        long double const eddington = (3 + 4 * f * f) / (5 + 2 * std::sqrt(4 - 3 * f * f));
        int const row[6] = {0, 1, 2, 0, 0, 1};
        int const column[6] = {0, 1, 2, 1, 2, 2};
        for (int component = 0; component < 6; ++component) {
            int const d = row[component], e = column[component];
            long double const desired = total_energy *
                ((d == e ? (1 - eddington) / 2 : 0) +
                 (3 * eddington - 1) / 2 * direction[d] * direction[e]);
            long double const scale = std::abs(initial_pressure[component]) + total_energy;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::abs(static_cast<long double>(actual_pressure[component]) -
                         initial_pressure[component] - desired) <= 2.e-13L * scale,
                "Actual near-beam native angular pressure differs from M1 closure.");
        }
        amrex::Print() << "Native M1 emission E/3P/pressure fraction=" << fraction << " pass.\n";
    }
    amrex::GpuArray<amrex::Real, 4> unchanged{};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(PhotonInventory(live, unchanged) &&
                                     SameInventory(unchanged, initial),
                                     "Discarding near-beam private probes changed live state.");
}
}

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        using warpx::radiation::PhotonPacketTransaction;
        using namespace warpx::radiation::detail;
        using ParticleType = WarpXParticleContainer::ParticleType;
        auto& simulation = WarpX::GetInstance();
        simulation.InitData();
        auto& live = simulation.GetPartContainer().GetParticleContainerFromName("photons");
        CheckNearBeamEmission(live, simulation.Geom(0));
        auto const initial_count = live.TotalNumberOfParticles();
        amrex::GpuArray<amrex::Real, 4> initial{};
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(PhotonInventory(live, initial), "Invalid initial state.");

        EmissionPlan plan;
        if (amrex::ParallelDescriptor::MyProc() == 0) {
            // Stage locally, then migrate into the other box/rank. All physical
            // momentum components are retained even by the 1D spatial fixture.
            plan.x = {0.25_prt, 0.25_prt};
            plan.y = {0.25_prt, 0.25_prt};
            plan.z = {0.8125_prt, 0.875_prt};
            plan.ux = {300.0_prt, -200.0_prt};
            plan.uy = {-400.0_prt, 500.0_prt};
            plan.uz = {500.0_prt, -600.0_prt};
            plan.weight = {2.5e20_prt, 1.25e20_prt};
        }
        long const before_reject = ParticleType::NextID();
        {
            PhotonPacketTransaction trial(live);
            EmissionPlan invalid = plan;
            if (amrex::ParallelDescriptor::MyProc() == 0) { invalid.y.pop_back(); }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!StageEmissionPlan(trial, invalid),
                                             "Malformed local emission must reject collectively.");
            amrex::GpuArray<amrex::Real, 4> unchanged{};
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(PhotonInventory(trial.Particles(), unchanged),
                                             "Invalid rejected candidate.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(SameInventory(unchanged, initial),
                                             "Emission rejection changed candidate inventory.");
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ParticleType::NextID() == before_reject + 1,
                                         "Rejected emission consumed native particle IDs.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(live.TotalNumberOfParticles() == initial_count,
                                         "Rejected emission changed live particle count.");

        long const before_discard = ParticleType::NextID();
        {
            PhotonPacketTransaction trial(live);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(StageEmissionPlan(trial, plan),
                                             "Private emission staging failed.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(trial.Particles().TotalNumberOfParticles() ==
                                                 initial_count + 2,
                                             "Private emission count is incorrect.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(trial.ValidateLive(), "Private validation failed.");
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ParticleType::NextID() == before_discard + 1,
                                         "Discarded emission consumed native IDs.");
        amrex::GpuArray<amrex::Real, 4> unchanged{};
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(PhotonInventory(live, unchanged) &&
                                             SameInventory(unchanged, initial),
                                         "Discarded emission changed live inventory.");
        {
            PhotonPacketTransaction trial(live);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(StageEmissionPlan(trial, plan),
                                             "Accepted emission staging failed.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(trial.ValidateLive(), "Accepted validation failed.");
            trial.Commit();
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(live.TotalNumberOfParticles() == initial_count + 2 &&
                                             live.NumIntComps() == 0,
                                         "Commit lost births or retained the private marker.");
        amrex::GpuArray<amrex::Real, 4> final{};
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(PhotonInventory(live, final), "Invalid final inventory.");
        long double const u[2][3] = {{300, -400, 500}, {-200, 500, -600}};
        long double const w[2] = {2.5e20L, 1.25e20L};
        long double expected[4] = {};
        for (int p = 0; p < 2; ++p) {
            long double speed2 = 0;
            for (int d = 0; d < 3; ++d) {
                speed2 += u[p][d] * u[p][d];
                expected[d + 1] += w[p] * PhysConst::m_e * PhysConst::c * u[p][d];
            }
            expected[0] += w[p] * PhysConst::m_e * PhysConst::c * std::sqrt(speed2);
        }
        for (int d = 0; d < 4; ++d) {
            long double const target = initial[d] + expected[d];
            long double const scale = std::abs(initial[d]) + std::abs(expected[d]);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(final[d] - target) <= 2.e-13L * scale,
                                             "Actual emitted photon energy/momentum is incorrect.");
        }
        amrex::Print() << "Private emission plan: reject/discard/commit and four-momentum pass.\n";
    }
    WarpX::Finalize();
    warpx::initialization::finalize_external_libraries();
    return 0;
}
