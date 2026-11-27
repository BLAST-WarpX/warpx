/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Radiation/NativePhotonPathReplay.H"
#include "Radiation/PhotonPacketTransaction.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace
{
constexpr int cases = 3;
constexpr std::array<int, 7> components{
    PIdx::x, PIdx::y, PIdx::z, PIdx::ux, PIdx::uy, PIdx::uz, PIdx::w};
struct Snapshot
{
    std::array<long double, cases*7> values{};
    std::array<std::uint64_t, cases> ids{};
};

Snapshot Read (WarpXParticleContainer::Base const& photons, int case_component)
{
    Snapshot result;
    for (auto const& [key, tile] : photons.GetParticles(0))
    {
        amrex::ignore_unused(key);
        long const count = tile.numParticles();
        auto const& soa = tile.GetStructOfArrays();
        std::vector<int> labels(count);
        std::vector<std::uint64_t> ids(count);
        auto const& labels_device = soa.GetIntData(case_component);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, labels_device.begin(),
            labels_device.begin()+count, labels.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetIdCPUData().begin(),
            soa.GetIdCPUData().begin()+count, ids.begin());
        for (int c = 0; c < 7; ++c)
        {
            std::vector<amrex::ParticleReal> host(count);
            auto const& values = soa.GetRealData(components[c]);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(), values.begin()+count,
                             host.begin());
            for (long p = 0; p < count; ++p)
            {
                AMREX_ALWAYS_ASSERT(labels[p] >= 0 && labels[p] < cases);
                result.values[7*labels[p]+c] = host[p];
                result.ids[labels[p]] = ids[p];
            }
        }
    }
#ifdef AMREX_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, result.values.data(), cases*7, MPI_LONG_DOUBLE,
                  MPI_SUM, amrex::ParallelDescriptor::Communicator());
    MPI_Allreduce(MPI_IN_PLACE, result.ids.data(), cases, MPI_UINT64_T,
                  MPI_BOR, amrex::ParallelDescriptor::Communicator());
#endif
    return result;
}

std::array<long double, 4> Inventory (Snapshot const& state)
{
    std::array<long double, 4> result{};
    for (int p = 0; p < cases; ++p)
    {
        long double norm2 = 0;
        auto const factor = state.values[7*p+6]*PhysConst::m_e*PhysConst::c;
        for (int d = 0; d < 3; ++d)
        {
            auto const u = state.values[7*p+3+d];
            norm2 += u*u;
            result[d+1] += factor*u;
        }
        result[0] += factor*std::sqrt(norm2);
    }
    return result;
}
}

int main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        using namespace warpx::radiation;
        auto& simulation = WarpX::GetInstance();
        auto& photons = simulation.GetPartContainer().GetParticleContainerFromName("photons");
        photons.AddIntComp("path_corner_case", 1);
        simulation.InitData();
        int const label = photons.GetIntCompIndex("path_corner_case");
        for (auto& [key, tile] : photons.GetParticles(0))
        {
            auto const count = tile.numParticles();
            auto& soa = tile.GetStructOfArrays();
            std::vector<std::uint64_t> ids(count);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetIdCPUData().begin(),
                soa.GetIdCPUData().begin()+count, ids.begin());
            std::vector<int> labels(count, 0);
            std::array<std::vector<amrex::ParticleReal>, 7> values;
            for (int c = 0; c < 7; ++c)
            {
                values[c].resize(count);
                auto const& source = soa.GetRealData(components[c]);
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, source.begin(), source.begin()+count,
                                 values[c].begin());
            }
            for (long p = 0; p < count; ++p)
            {
                if (key.first != 0 || key.second != 0 || p >= cases)
                {
                    ids[p] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }
                labels[p] = static_cast<int>(p);
                auto const position = p == 0 ? -0.002 : p == 1 ? 0.248 : -0.248;
                for (int d = 0; d < 3; ++d)
                {
                    values[d][p] = position;
                    auto const numerator = p == 1 ? 3-d : d+1;
                    values[3+d][p] = (p == 2 ? -1 : 1)*numerator*PhysConst::c/std::sqrt(14.0);
                }
                values[6][p] = 1;
            }
            for (int c = 0; c < 7; ++c)
            {
                amrex::Gpu::copy(amrex::Gpu::hostToDevice, values[c].begin(), values[c].end(),
                    soa.GetRealData(components[c]).begin());
            }
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, ids.begin(), ids.end(),
                             soa.GetIdCPUData().begin());
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, labels.begin(), labels.end(),
                             soa.GetIntData(label).begin());
        }
        photons.Redistribute();
        AMREX_ALWAYS_ASSERT(photons.TotalNumberOfParticles() == cases);
        auto const initial = Read(photons, label);
        auto const before = Inventory(initial);
        auto const& ba = photons.ParticleBoxArray(0);
        auto const& dm = photons.ParticleDistributionMap(0);
        amrex::MultiFab chi(ba, dm, 1, 0), velocity(ba, dm, 3, 0);
        amrex::MultiFab impulse(ba, dm, 3, 0), debit(ba, dm, 1, 0), scale(ba, dm, 1, 0);
        velocity.setVal(0);
        auto const dt = 0.0625/PhysConst::c;
        NativePhotonCollisionOptions options;
        options.seed = 4;
        options.interval = 27;
        options.moving_cross_sections = true;
        for (bool const scattering : {false, true})
        {
            chi.setVal(scattering ? 80 : 0);
            NativePhotonPathReplay replay;
            NativePhotonPathReport report;
            AMREX_ALWAYS_ASSERT(replay.Prepare(photons, chi, velocity, simulation.Geom(0),
                                               dt, options, report));
            PhotonPacketTransaction transaction(photons);
            bool const accepted = replay.Replay(velocity, transaction.Particles(),
                                                impulse, debit, scale, report);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(accepted, report.reason.c_str());
            auto const final = Read(transaction.Particles(), label);
            AMREX_ALWAYS_ASSERT(final.ids == initial.ids);
            auto const after = Inventory(final);
            auto const roundoff = 2048*std::numeric_limits<amrex::Real>::epsilon();
            for (int d = 0; d < 3; ++d)
            {
                AMREX_ALWAYS_ASSERT(std::abs(before[d+1]-after[d+1]-
                    PhysConst::c*impulse.sum(d)) <= roundoff*(before[0]+after[0]));
            }
            AMREX_ALWAYS_ASSERT(std::abs(before[0]-after[0]-debit.sum()) <=
                                roundoff*(before[0]+after[0]));
            if (scattering) { AMREX_ALWAYS_ASSERT(report.collision_events > 0); }
            else
            {
                AMREX_ALWAYS_ASSERT(report.collision_events == 0 && report.face_crossings == 9);
                AMREX_ALWAYS_ASSERT(debit.norm0() == 0 && scale.norm0() == 0);
                for (int d = 0; d < 3; ++d) { AMREX_ALWAYS_ASSERT(impulse.norm0(d) == 0); }
                for (int p = 0; p < cases; ++p)
                {
                    long double norm2 = 0;
                    for (int d = 0; d < 3; ++d)
                    {
                        auto const u = initial.values[7*p+3+d];
                        norm2 += u*u;
                    }
                    for (int d = 0; d < 3; ++d)
                    {
                        auto expected = initial.values[7*p+d]+PhysConst::c*dt*
                            initial.values[7*p+3+d]/std::sqrt(norm2);
                        if (expected < -0.25L) { expected += 0.5L; }
                        if (expected >= 0.25L) { expected -= 0.5L; }
                        AMREX_ALWAYS_ASSERT(std::abs(final.values[7*p+d]-expected) <= roundoff*0.5L);
                    }
                    for (int c = 3; c < 7; ++c)
                    {
                        AMREX_ALWAYS_ASSERT(final.values[7*p+c] == initial.values[7*p+c]);
                    }
                }
            }
        }
        auto const live = Read(photons, label);
        AMREX_ALWAYS_ASSERT(live.ids == initial.ids && live.values == initial.values);
        amrex::Print() << "Three-dimensional corner and periodic native path ownership passed.\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
