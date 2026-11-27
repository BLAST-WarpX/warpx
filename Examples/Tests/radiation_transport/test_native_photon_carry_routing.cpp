/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */

#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Radiation/NativePhotonCarryRouting.H"
#include "Radiation/NativePhotonPathReplay.H"
#include "Radiation/NativePhotonReplayInternals.H"
#include "Radiation/PhotonPacketTransaction.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <sstream>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#endif

using namespace amrex::literals;

namespace
{
using Carry = warpx::radiation::NativePhotonPathCarry;

std::vector<std::uint64_t>
LocalIDs (WarpXParticleContainer::Base const& particles)
{
    std::vector<std::uint64_t> ids;
    for (auto const& [key, tile] : particles.GetParticles(0))
    {
        amrex::ignore_unused(key);
        auto const count = tile.numParticles();
        if (count == 0) { continue; }
        auto const& soa = tile.GetStructOfArrays();
        auto const offset = ids.size();
        ids.resize(offset + static_cast<std::size_t>(count));
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,
            soa.GetIdCPUData().begin(), soa.GetIdCPUData().begin()+count,
            ids.begin()+static_cast<std::ptrdiff_t>(offset));
    }
    return ids;
}

void
AssertPhotonState (WarpXParticleContainer::Base const& particles,
                   amrex::Real expected_z, int real_sentinel, int int_sentinel)
{
    for (auto const& [key, tile] : particles.GetParticles(0))
    {
        amrex::ignore_unused(key);
        auto const count = tile.numParticles();
        if (count == 0) { continue; }
        auto const& soa = tile.GetStructOfArrays();
        std::array<int, 6> const components{
            PIdx::z, PIdx::w, PIdx::ux, PIdx::uy, PIdx::uz, real_sentinel};
        std::array<std::vector<amrex::ParticleReal>, 6> values;
        for (std::size_t c = 0; c < components.size(); ++c)
        {
            values[c].resize(static_cast<std::size_t>(count));
            amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                soa.GetRealData(components[c]).begin(),
                soa.GetRealData(components[c]).begin()+count,
                values[c].begin());
        }
        std::vector<int> markers(static_cast<std::size_t>(count));
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,
            soa.GetIntData(int_sentinel).begin(),
            soa.GetIntData(int_sentinel).begin()+count,
            markers.begin());
        for (long n = 0; n < count; ++n)
        {
            auto const i = static_cast<std::size_t>(n);
            AMREX_ALWAYS_ASSERT(std::abs(
                static_cast<amrex::Real>(values[0][i])-expected_z) <=
                512*std::numeric_limits<amrex::Real>::epsilon()*
                    std::max(amrex::Real(1), std::abs(expected_z)));
            AMREX_ALWAYS_ASSERT(values[1][i] == 1.0_prt);
            AMREX_ALWAYS_ASSERT(values[2][i] == 0.0_prt);
            AMREX_ALWAYS_ASSERT(values[3][i] == 0.0_prt);
            AMREX_ALWAYS_ASSERT(values[4][i] ==
                                static_cast<amrex::ParticleReal>(PhysConst::c));
            AMREX_ALWAYS_ASSERT(values[5][i] == 19.5_prt);
            AMREX_ALWAYS_ASSERT(markers[i] == 37);
        }
    }
}

void
PlacePhotons (WarpXParticleContainer::Base& particles, amrex::ParticleReal z,
              int real_sentinel, int int_sentinel)
{
    for (auto& [key, tile] : particles.GetParticles(0))
    {
        amrex::ignore_unused(key);
        auto const count = tile.numParticles();
        if (count == 0) { continue; }
        auto& soa = tile.GetStructOfArrays();
        auto const put_real = [&] (int component, amrex::ParticleReal value)
        {
            std::vector<amrex::ParticleReal> values(
                static_cast<std::size_t>(count), value);
            amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                values.begin(), values.end(),
                soa.GetRealData(component).begin());
        };
        put_real(PIdx::z, z);
        put_real(PIdx::w, 1.0_prt);
        put_real(PIdx::ux, 0.0_prt);
        put_real(PIdx::uy, 0.0_prt);
        put_real(PIdx::uz, static_cast<amrex::ParticleReal>(PhysConst::c));
        put_real(real_sentinel, 19.5_prt);
        std::vector<int> marker(static_cast<std::size_t>(count), 37);
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
            marker.begin(), marker.end(),
            soa.GetIntData(int_sentinel).begin());
    }
}

Carry
SyntheticCarry (std::uint64_t idcpu)
{
    int const events = static_cast<int>(idcpu % 5);
    return Carry{idcpu, events+1, events,
                 0.5_rt + static_cast<amrex::Real>(idcpu % 13)/16};
}

bool
SameCarries (std::vector<Carry> const& left, std::vector<Carry> const& right)
{
    if (left.size() != right.size()) { return false; }
    for (std::size_t n = 0; n < left.size(); ++n)
    {
        auto const& a = left[n];
        auto const& b = right[n];
        if (a.idcpu != b.idcpu || a.next_draw != b.next_draw ||
            a.completed_events != b.completed_events ||
            a.residual_optical_depth != b.residual_optical_depth)
        {
            return false;
        }
    }
    return true;
}

amrex::Real
ExpectedDepth (warpx::radiation::NativePhotonCollisionOptions const& options,
               std::uint64_t idcpu, int event)
{
    return -std::log(warpx::radiation::photon_replay_detail::OpenUnitInterval(
        warpx::radiation::photon_replay_detail::PhotonDrawKey(
            options.seed, idcpu, options.interval, event, 0)));
}

void
RunDistributedRouting (WarpXParticleContainer& photons)
{
    using namespace warpx::radiation::photon_replay_detail;
    auto const source_ids = LocalIDs(photons);
    auto const global_photons = photons.TotalNumberOfParticles();
    auto const ranks = amrex::ParallelDescriptor::NProcs();
    auto const my_rank = amrex::ParallelDescriptor::MyProc();
    auto const keep_count = (source_ids.size()+1)/2;
    std::vector<std::uint64_t> target_ids(
        source_ids.begin(), source_ids.begin()+
            static_cast<std::ptrdiff_t>(keep_count));
#ifdef AMREX_USE_MPI
    if (ranks > 1)
    {
        AMREX_ALWAYS_ASSERT(source_ids.size()-keep_count <=
                            static_cast<std::size_t>(std::numeric_limits<int>::max()));
        int const send_count = static_cast<int>(source_ids.size()-keep_count);
        int receive_count = 0;
        int const next = (my_rank+1) % ranks;
        int const previous = (my_rank+ranks-1) % ranks;
        MPI_Sendrecv(&send_count, 1, MPI_INT, next, 40,
                     &receive_count, 1, MPI_INT, previous, 40,
                     amrex::ParallelDescriptor::Communicator(), MPI_STATUS_IGNORE);
        std::vector<std::uint64_t> received(
            static_cast<std::size_t>(receive_count));
        std::uint64_t dummy = 0;
        auto const* send_data = send_count > 0 ?
            source_ids.data()+keep_count : &dummy;
        auto* receive_data = receive_count > 0 ? received.data() : &dummy;
        MPI_Sendrecv(send_data, send_count, MPI_UINT64_T,
                     next, 41, receive_data, receive_count, MPI_UINT64_T,
                     previous, 41, amrex::ParallelDescriptor::Communicator(),
                     MPI_STATUS_IGNORE);
        target_ids.insert(target_ids.end(), received.begin(), received.end());
    }
#endif
    std::vector<Carry> source_carries;
    source_carries.reserve(source_ids.size());
    for (auto const idcpu : source_ids)
    {
        source_carries.push_back(SyntheticCarry(idcpu));
    }
    std::vector<Carry> received_carries;
    auto const started = std::chrono::steady_clock::now();
    AMREX_ALWAYS_ASSERT(RoutePhotonCarries(
        source_carries, target_ids, 64, received_carries));
    auto const ended = std::chrono::steady_clock::now();
    AMREX_ALWAYS_ASSERT(received_carries.size() == target_ids.size());
    for (std::size_t n = 0; n < target_ids.size(); ++n)
    {
        AMREX_ALWAYS_ASSERT(SameCarries(
            {received_carries[n]}, {SyntheticCarry(target_ids[n])}));
    }
    amrex::Long source_max = static_cast<amrex::Long>(source_ids.size());
    amrex::Long target_max = static_cast<amrex::Long>(target_ids.size());
    amrex::Long moved = static_cast<amrex::Long>(source_ids.size()-keep_count);
    amrex::ParallelDescriptor::ReduceLongMax(source_max);
    amrex::ParallelDescriptor::ReduceLongMax(target_max);
    amrex::ParallelDescriptor::ReduceLongSum(moved);
    if (ranks > 1) { AMREX_ALWAYS_ASSERT(moved > 0); }
    auto route_seconds = std::chrono::duration<double>(ended-started).count();
    amrex::ParallelDescriptor::ReduceRealMax(route_seconds);
    amrex::Long peak_rss_kib = 0;
    std::vector<amrex::Long> rss_by_rank(static_cast<std::size_t>(ranks), 0);
#if defined(__linux__)
    struct rusage usage{};
    AMREX_ALWAYS_ASSERT(getrusage(RUSAGE_SELF, &usage) == 0);
    peak_rss_kib = static_cast<amrex::Long>(usage.ru_maxrss);
#ifdef AMREX_USE_MPI
    MPI_Gather(&peak_rss_kib, 1, MPI_LONG, rss_by_rank.data(), 1, MPI_LONG,
        amrex::ParallelDescriptor::IOProcessorNumber(),
        amrex::ParallelDescriptor::Communicator());
#else
    rss_by_rank[0] = peak_rss_kib;
#endif
    amrex::ParallelDescriptor::ReduceLongMax(peak_rss_kib);
#endif
    std::ostringstream rss_values;
    for (int rank = 0; rank < ranks; ++rank)
    {
        if (rank != 0) { rss_values << ','; }
        rss_values << rss_by_rank[static_cast<std::size_t>(rank)];
    }
    amrex::Print() << "carry-routing distributed photons=" << global_photons
        << " ranks=" << ranks << " moved=" << moved
        << " source max records=" << source_max
        << " destination max records=" << target_max
        << " route max seconds=" << route_seconds
        << " peak rank RSS KiB=" << peak_rss_kib
        << " rank RSS KiB=" << rss_values.str() << "\n";
}
}

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        using namespace warpx::radiation;
        using namespace warpx::radiation::photon_replay_detail;
        auto& simulation = WarpX::GetInstance();
        auto& particles = simulation.GetPartContainer();
        auto& photons_before_init =
            particles.GetParticleContainerFromName("photons");
        photons_before_init.AddRealComp("routing_real_sentinel", 1);
        photons_before_init.AddIntComp("routing_int_sentinel", 1);
        simulation.InitData();
        auto& photons = particles.GetParticleContainerFromName("photons");
        amrex::ParmParse routing_parameters("routing_test");
        int distributed_mode = 0;
        routing_parameters.query("distributed", distributed_mode);
        if (distributed_mode != 0)
        {
            RunDistributedRouting(photons);
            WarpX::Finalize();
            warpx::initialization::finalize_external_libraries();
            return 0;
        }
        auto const& geometry = simulation.Geom(0);
        auto const dx = geometry.CellSize(0);
        auto const start_z = 0.375_rt-dx/16;
        PlacePhotons(photons, static_cast<amrex::ParticleReal>(start_z),
            photons.GetRealCompIndex("routing_real_sentinel"),
            photons.GetIntCompIndex("routing_int_sentinel"));
        photons.Redistribute();
        auto const source_ids = LocalIDs(photons);
        auto const global_photons = photons.TotalNumberOfParticles();
        AMREX_ALWAYS_ASSERT(global_photons > 0);
        AMREX_ALWAYS_ASSERT(photons.ParticleBoxArray(0).size() == 2);
        auto const& boxes = photons.ParticleBoxArray(0);
        auto const& distribution = photons.ParticleDistributionMap(0);
        AMREX_ALWAYS_ASSERT(boxes[0].bigEnd(0) == 7);
        if (amrex::ParallelDescriptor::MyProc() == distribution[0])
        {
            AMREX_ALWAYS_ASSERT(source_ids.size() ==
                static_cast<std::size_t>(global_photons));
        }
        else
        {
            AMREX_ALWAYS_ASSERT(source_ids.empty());
        }

        amrex::MultiFab opacity(boxes, distribution, 1, 0);
        amrex::MultiFab velocity(boxes, distribution, 3, 0);
        opacity.setVal(16/dx);
        velocity.setVal(0);
        auto const dt = dx/(4*PhysConst::c);
        NativePhotonCollisionOptions options;
        options.seed = 91;
        options.interval = 29;
        options.max_events_per_photon = 64;
        options.max_optical_depth = 16;
        options.angular_law = NativePhotonCollisionAngularLaw::FixedFrameDirection;
        options.fixed_frame_direction = {0, 0, 1};
        NativePhotonPathReplay replay;
        NativePhotonPathReport prepared;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(replay.Prepare(
            photons, opacity, velocity, geometry, dt, options, prepared),
            prepared.reason.c_str());
        PhotonPacketTransaction trial(photons);
        amrex::MultiFab impulse(boxes, distribution, 3, 1);
        amrex::MultiFab debit(boxes, distribution, 1, 1);
        amrex::MultiFab scale(boxes, distribution, 1, 1);
        auto const replay_start = std::chrono::steady_clock::now();
        NativePhotonPathReport report;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(replay.Replay(velocity,
            trial.Particles(), impulse, debit, scale, report),
            report.reason.c_str());
        auto const replay_end = std::chrono::steady_clock::now();
        auto const destination_ids = LocalIDs(trial.Particles());
        AMREX_ALWAYS_ASSERT(report.valid);
        AMREX_ALWAYS_ASSERT(trial.Particles().TotalNumberOfParticles() ==
                            global_photons);
        AMREX_ALWAYS_ASSERT(report.face_crossings == global_photons);
        AssertPhotonState(photons, start_z,
            photons.GetRealCompIndex("routing_real_sentinel"),
            photons.GetIntCompIndex("routing_int_sentinel"));
        AssertPhotonState(trial.Particles(),
            start_z+PhysConst::c*dt,
            photons.GetRealCompIndex("routing_real_sentinel"),
            photons.GetIntCompIndex("routing_int_sentinel"));
        if (amrex::ParallelDescriptor::MyProc() == distribution[1])
        {
            AMREX_ALWAYS_ASSERT(destination_ids.size() ==
                static_cast<std::size_t>(global_photons));
        }
        else
        {
            AMREX_ALWAYS_ASSERT(destination_ids.empty());
        }
        AMREX_ALWAYS_ASSERT(LocalIDs(photons) == source_ids);
        AMREX_ALWAYS_ASSERT(opacity.norm0() == 16/dx);
        for (int component = 0; component < 3; ++component)
        {
            AMREX_ALWAYS_ASSERT(velocity.norm0(component) == 0);
            AMREX_ALWAYS_ASSERT(PhysConst::c*impulse.norm0(component) <=
                512*std::numeric_limits<amrex::Real>::epsilon()*
                    report.incoming_inventory[0]);
        }
        AMREX_ALWAYS_ASSERT(debit.norm0() <=
            512*std::numeric_limits<amrex::Real>::epsilon()*
                report.incoming_inventory[0]);

        std::map<std::uint64_t, Carry> native_carries;
        for (auto const& carry : report.photon_carries)
        {
            AMREX_ALWAYS_ASSERT(native_carries.emplace(carry.idcpu, carry).second);
        }
        AMREX_ALWAYS_ASSERT(native_carries.size() == destination_ids.size());
        amrex::Long local_events = 0;
        amrex::Long local_colliders = 0;
        amrex::Real const total_depth = 16*PhysConst::c*dt/dx;
        for (auto const idcpu : destination_ids)
        {
            auto const found = native_carries.find(idcpu);
            AMREX_ALWAYS_ASSERT(found != native_carries.end());
            auto const& carry = found->second;
            amrex::Real consumed = 0;
            int events = 0;
            amrex::Real next_depth = ExpectedDepth(options, idcpu, events);
            while (consumed + next_depth <= total_depth)
            {
                consumed += next_depth;
                ++events;
                AMREX_ALWAYS_ASSERT(events < options.max_events_per_photon);
                next_depth = ExpectedDepth(options, idcpu, events);
            }
            auto const expected_residual = next_depth-(total_depth-consumed);
            AMREX_ALWAYS_ASSERT(carry.completed_events == events);
            AMREX_ALWAYS_ASSERT(carry.next_draw == events+1);
            AMREX_ALWAYS_ASSERT(std::abs(
                carry.residual_optical_depth-expected_residual) <=
                512*std::numeric_limits<amrex::Real>::epsilon()*
                    std::max(total_depth, next_depth));
            local_events += events;
            local_colliders += events > 0 ? 1 : 0;
        }
        amrex::ParallelDescriptor::ReduceLongSum(local_events);
        amrex::ParallelDescriptor::ReduceLongSum(local_colliders);
        AMREX_ALWAYS_ASSERT(report.collision_events == local_events);
        AMREX_ALWAYS_ASSERT(report.colliding_photons == local_colliders);

        std::vector<Carry> source_carries;
        source_carries.reserve(source_ids.size());
        for (auto const idcpu : source_ids)
        {
            source_carries.push_back(SyntheticCarry(idcpu));
        }
        std::vector<Carry> migrated;
        auto const route_start = std::chrono::steady_clock::now();
        AMREX_ALWAYS_ASSERT(RoutePhotonCarries(
            source_carries, destination_ids, 64, migrated));
        auto const route_end = std::chrono::steady_clock::now();
        AMREX_ALWAYS_ASSERT(migrated.size() == destination_ids.size());
        for (std::size_t n = 0; n < destination_ids.size(); ++n)
        {
            AMREX_ALWAYS_ASSERT(SameCarries(
                {migrated[n]}, {SyntheticCarry(destination_ids[n])}));
        }

        auto const expect_rejection = [&] (
            std::vector<Carry> const& carries,
            std::vector<std::uint64_t> const& ids, int cap)
        {
            std::vector<Carry> unchanged{{42, 3, 2, 0.75_rt}};
            auto const before = unchanged;
            AMREX_ALWAYS_ASSERT(!RoutePhotonCarries(carries, ids, cap, unchanged));
            AMREX_ALWAYS_ASSERT(SameCarries(unchanged, before));
        };
        {
            auto missing = source_carries;
            if (!missing.empty()) { missing.pop_back(); }
            expect_rejection(missing, destination_ids, 64);
            auto extra = destination_ids;
            if (!extra.empty()) { extra.pop_back(); }
            expect_rejection(source_carries, extra, 64);
            auto duplicate_source = source_carries;
            if (!duplicate_source.empty())
            {
                duplicate_source.push_back(duplicate_source.front());
            }
            expect_rejection(duplicate_source, destination_ids, 64);
            auto duplicate_owner = destination_ids;
            if (!duplicate_owner.empty())
            {
                duplicate_owner.push_back(duplicate_owner.front());
            }
            expect_rejection(source_carries, duplicate_owner, 64);
            auto invalid_payload = source_carries;
            if (!invalid_payload.empty())
            {
                invalid_payload.front().residual_optical_depth =
                    std::numeric_limits<amrex::Real>::quiet_NaN();
            }
            expect_rejection(invalid_payload, destination_ids, 64);
            auto overflow_payload = source_carries;
            if (!overflow_payload.empty())
            {
                overflow_payload.front().completed_events =
                    std::numeric_limits<int>::max();
                overflow_payload.front().next_draw =
                    std::numeric_limits<int>::max();
            }
            expect_rejection(overflow_payload, destination_ids, 64);
            expect_rejection(source_carries, destination_ids,
                std::numeric_limits<int>::max());
            if (amrex::ParallelDescriptor::NProcs() > 1)
            {
                int const different_cap =
                    amrex::ParallelDescriptor::MyProc() == 0 ? 64 : 65;
                expect_rejection(source_carries, destination_ids, different_cap);
            }
        }
        constexpr auto int_limit =
            static_cast<std::size_t>(std::numeric_limits<int>::max());
        AMREX_ALWAYS_ASSERT(ValidPackedWordCounts({0, 4, 8}, 4));
        AMREX_ALWAYS_ASSERT(!ValidPackedWordCounts({4}, 0));
        AMREX_ALWAYS_ASSERT(!ValidPackedWordCounts({5}, 4));
        AMREX_ALWAYS_ASSERT(!ValidPackedWordCounts({int_limit+1}, 4));
        AMREX_ALWAYS_ASSERT(!ValidPackedWordCounts(
            {int_limit-3, 4}, 4));

        auto replay_seconds = std::chrono::duration<double>(
            replay_end-replay_start).count();
        auto route_seconds = std::chrono::duration<double>(
            route_end-route_start).count();
        amrex::ParallelDescriptor::ReduceRealMax(replay_seconds);
        amrex::ParallelDescriptor::ReduceRealMax(route_seconds);
        amrex::Long source_max = static_cast<amrex::Long>(source_carries.size());
        amrex::Long destination_max = static_cast<amrex::Long>(destination_ids.size());
        amrex::ParallelDescriptor::ReduceLongMax(source_max);
        amrex::ParallelDescriptor::ReduceLongMax(destination_max);
        amrex::Long peak_rss_kib = 0;
#if defined(__linux__)
        struct rusage usage{};
        AMREX_ALWAYS_ASSERT(getrusage(RUSAGE_SELF, &usage) == 0);
        peak_rss_kib = static_cast<amrex::Long>(usage.ru_maxrss);
        amrex::ParallelDescriptor::ReduceLongMax(peak_rss_kib);
#endif
        amrex::Print() << "carry-routing native photons=" << global_photons
            << " ranks=" << amrex::ParallelDescriptor::NProcs()
            << " events=" << report.collision_events
            << " source max records=" << source_max
            << " destination max records=" << destination_max
            << " replay max seconds=" << replay_seconds
            << " route max seconds=" << route_seconds
            << " peak rank RSS KiB=" << peak_rss_kib << "\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
    return 0;
}
