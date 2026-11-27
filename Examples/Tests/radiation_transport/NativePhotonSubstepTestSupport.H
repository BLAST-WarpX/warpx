#ifndef WARPX_NATIVE_PHOTON_SUBSTEP_TEST_SUPPORT_H_
#define WARPX_NATIVE_PHOTON_SUBSTEP_TEST_SUPPORT_H_
/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Radiation/NativePhotonPathReplay.H"
#include "Radiation/NativePhotonReplayInternals.H"
#include "Radiation/PhotonFourMomentum.H"
#include "Radiation/PhotonPacketTransaction.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

using namespace amrex::literals;

namespace
{
using namespace warpx::radiation;

constexpr int x_position = 0;
constexpr int y_position = 1;
constexpr int z_position = 2;
constexpr int x_momentum = 3;
constexpr int y_momentum = 4;
constexpr int z_momentum = 5;
constexpr int weight = 6;
constexpr int real_sentinel_value = 7;

struct PhotonState
{
    std::array<amrex::Real, 8> real{};
    int int_sentinel = 0;
    std::uint64_t idcpu = 0;
    int owner = 0;
    int box = 0;
};

struct LocalParticleRecord
{
    std::pair<int, int> key;
    std::vector<amrex::ParticleReal> real;
    std::vector<int> integer;
};

using LocalParticleRecords = std::map<std::uint64_t, LocalParticleRecord>;

struct History
{
    PhotonState photon;
    NativePhotonPathCarry carry;
    amrex::Long events = 0;
    amrex::Long face_crossings = 0;
    std::array<amrex::Real, 4> incoming{};
    std::array<amrex::Real, 4> outgoing{};
    std::array<amrex::Real, 3> impulse{};
    amrex::Real debit = 0;
    std::unique_ptr<amrex::MultiFab> impulse_field;
    std::unique_ptr<amrex::MultiFab> debit_field;
    std::unique_ptr<amrex::MultiFab> scale_field;
    int subsegments = 0;
};

void
PrintHistory (char const* name, History const& history)
{
    auto output = amrex::Print();
    output.SetPrecision(17);
    output << "PARTITION_HISTORY " << name << " " << history.photon.idcpu
           << " " << history.carry.next_draw << " " << history.carry.completed_events
           << " " << history.events << " " << history.face_crossings;
    for (auto value : history.photon.real) { output << " " << value; }
    output << " " << history.carry.residual_optical_depth;
    for (auto value : history.incoming) { output << " " << value; }
    for (auto value : history.outgoing) { output << " " << value; }
    for (auto value : history.impulse) { output << " " << value; }
    output << " " << history.debit << "\n";
}

PhotonState
ReadOne (WarpXParticleContainer::Base const& photons, int real_sentinel,
         int int_sentinel)
{
    PhotonState result;
    int found = 0;
    std::array<int, 8> const components{
        PIdx::x, PIdx::y, PIdx::z, PIdx::ux, PIdx::uy, PIdx::uz,
        PIdx::w, real_sentinel};
    for (auto const& [key, tile] : photons.GetParticles(0))
    {
        auto const count = tile.numParticles();
        auto const& soa = tile.GetStructOfArrays();
        for (long particle = 0; particle < count; ++particle)
        {
            std::uint64_t id = 0;
            amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                soa.GetIdCPUData().begin() + particle,
                soa.GetIdCPUData().begin() + particle + 1, &id);
            if (!amrex::ConstParticleIDWrapper(id).is_valid()) { continue; }
            AMREX_ALWAYS_ASSERT(found == 0);
            result.idcpu = id;
            for (int c = 0; c < 8; ++c)
            {
                amrex::ParticleReal value = 0;
                amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                    soa.GetRealData(components[c]).begin() + particle,
                    soa.GetRealData(components[c]).begin() + particle + 1,
                    &value);
                result.real[static_cast<std::size_t>(c)] = value;
            }
            amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                soa.GetIntData(int_sentinel).begin() + particle,
                soa.GetIntData(int_sentinel).begin() + particle + 1,
                &result.int_sentinel);
            result.owner = amrex::ParallelDescriptor::MyProc() + 1;
            result.box = key.first + 1;
            found = 1;
        }
    }
    amrex::ParallelDescriptor::ReduceIntSum(found);
    AMREX_ALWAYS_ASSERT(found == 1);
    amrex::ParallelDescriptor::ReduceRealSum(result.real.data(), 8);
    amrex::ParallelDescriptor::ReduceIntSum(result.int_sentinel);
    amrex::ParallelDescriptor::ReduceIntSum(result.owner);
    amrex::ParallelDescriptor::ReduceIntSum(result.box);
#ifdef AMREX_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &result.idcpu, 1, MPI_UINT64_T, MPI_BOR,
        amrex::ParallelDescriptor::Communicator());
#endif
    --result.owner;
    --result.box;
    return result;
}

void
SetPhoton (WarpXParticleContainer& photons, int real_sentinel,
           int int_sentinel, std::array<amrex::Real, 3> const& position,
           std::array<amrex::Real, 3> const& direction)
{
    std::array<int, 8> const components{
        PIdx::x, PIdx::y, PIdx::z, PIdx::ux, PIdx::uy, PIdx::uz,
        PIdx::w, real_sentinel};
    std::array<amrex::Real, 8> const values{
        position[0], position[1], position[2],
        PhysConst::c * direction[0], PhysConst::c * direction[1],
        PhysConst::c * direction[2], 1.25_rt, 19.5_rt};
    int local_count = 0;
    for (auto& [key, tile] : photons.GetParticles(0))
    {
        amrex::ignore_unused(key);
        auto const count = tile.numParticles();
        auto& soa = tile.GetStructOfArrays();
        for (long particle = 0; particle < count; ++particle)
        {
            std::uint64_t id = 0;
            amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                soa.GetIdCPUData().begin() + particle,
                soa.GetIdCPUData().begin() + particle + 1, &id);
            if (!amrex::ConstParticleIDWrapper(id).is_valid()) { continue; }
            ++local_count;
            for (int c = 0; c < 8; ++c)
            {
                auto const value = static_cast<amrex::ParticleReal>(
                    values[static_cast<std::size_t>(c)]);
                amrex::Gpu::copy(amrex::Gpu::hostToDevice, &value, &value + 1,
                    soa.GetRealData(components[c]).begin() + particle);
            }
            int const marker = 37;
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, &marker, &marker + 1,
                soa.GetIntData(int_sentinel).begin() + particle);
        }
    }
    amrex::ParallelDescriptor::ReduceIntSum(local_count);
    AMREX_ALWAYS_ASSERT(local_count == 1);
    amrex::Gpu::streamSynchronize();
    photons.Redistribute();
    AMREX_ALWAYS_ASSERT(photons.TotalNumberOfParticles() == 1);
}

LocalParticleRecords
LocalRecords (WarpXParticleContainer::Base const& particles)
{
    LocalParticleRecords result;
    for (auto const& [key, tile] : particles.GetParticles(0))
    {
        auto const count = tile.numParticles();
        if (count == 0) { continue; }
        auto const& soa = tile.GetStructOfArrays();
        std::vector<std::uint64_t> ids(static_cast<std::size_t>(count));
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,
            soa.GetIdCPUData().begin(), soa.GetIdCPUData().begin() + count,
            ids.begin());
        std::vector<std::vector<amrex::ParticleReal>> real_components(
            static_cast<std::size_t>(particles.NumRealComps()),
            std::vector<amrex::ParticleReal>(static_cast<std::size_t>(count)));
        std::vector<std::vector<int>> int_components(
            static_cast<std::size_t>(particles.NumIntComps()),
            std::vector<int>(static_cast<std::size_t>(count)));
        for (int component = 0; component < particles.NumRealComps(); ++component)
        {
            amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                soa.GetRealData(component).begin(),
                soa.GetRealData(component).begin() + count,
                real_components[static_cast<std::size_t>(component)].begin());
        }
        for (int component = 0; component < particles.NumIntComps(); ++component)
        {
            amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                soa.GetIntData(component).begin(),
                soa.GetIntData(component).begin() + count,
                int_components[static_cast<std::size_t>(component)].begin());
        }
        for (long particle = 0; particle < count; ++particle)
        {
            auto const id = ids[static_cast<std::size_t>(particle)];
            if (!amrex::ConstParticleIDWrapper(id).is_valid()) { continue; }
            LocalParticleRecord record;
            record.key = key;
            for (auto const& component : real_components)
            {
                record.real.push_back(component[static_cast<std::size_t>(particle)]);
            }
            for (auto const& component : int_components)
            {
                record.integer.push_back(component[static_cast<std::size_t>(particle)]);
            }
            AMREX_ALWAYS_ASSERT(result.emplace(id, std::move(record)).second);
        }
    }
    return result;
}

template <typename T>
bool
SameBits (std::vector<T> const& first, std::vector<T> const& second)
{
    return first.size() == second.size() &&
        (first.empty() || std::memcmp(first.data(), second.data(),
                                      first.size() * sizeof(T)) == 0);
}

bool
SameRecords (LocalParticleRecords const& first, LocalParticleRecords const& second)
{
    if (first.size() != second.size()) { return false; }
    for (auto const& [id, record] : first)
    {
        auto const found = second.find(id);
        if (found == second.end() || record.key != found->second.key ||
            !SameBits(record.real, found->second.real) ||
            !SameBits(record.integer, found->second.integer))
        {
            return false;
        }
    }
    return true;
}

std::vector<amrex::Real>
FieldSnapshot (amrex::MultiFab const& field)
{
    std::vector<amrex::Real> result;
    for (amrex::MFIter iterator(field); iterator.isValid(); ++iterator)
    {
        auto const& fab = field[iterator];
        auto const count = static_cast<std::size_t>(fab.box().numPts()) *
            static_cast<std::size_t>(field.nComp());
        auto const offset = result.size();
        result.resize(offset + count);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, fab.dataPtr(),
            fab.dataPtr() + count, result.data() + offset);
    }
    return result;
}

bool
Close (amrex::Real first, amrex::Real second, amrex::Real scale)
{
    auto const bound = 2048 * std::numeric_limits<amrex::Real>::epsilon() *
        std::max({std::abs(first), std::abs(second), std::abs(scale),
                  std::numeric_limits<amrex::Real>::min()});
    return std::abs(first - second) <= bound;
}

amrex::Real
ExpectedDepth (NativePhotonCollisionOptions const& options,
               std::uint64_t idcpu, int event)
{
    return -std::log(photon_replay_detail::OpenUnitInterval(
        photon_replay_detail::PhotonDrawKey(options.seed, idcpu,
        options.interval, static_cast<std::uint64_t>(event), 0)));
}

std::array<amrex::Real, 2>
SelectModerateDepths (NativePhotonCollisionOptions& options, std::uint64_t idcpu)
{
    for (std::uint64_t interval = 0; interval < 4096; ++interval)
    {
        options.interval = interval;
        auto const first = ExpectedDepth(options, idcpu, 0);
        auto const second = ExpectedDepth(options, idcpu, 1);
        auto const ratio = second / first;
        if (ratio >= 0.5_rt && ratio <= 1.5_rt) { return {first, second}; }
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false,
        "failed to find a bounded deterministic two-event optical-depth pair");
    return {};
}

NativePhotonPathCarry
ReadCarry (std::vector<NativePhotonPathCarry> const& carries, std::uint64_t idcpu)
{
    NativePhotonPathCarry result;
    int found = 0;
    for (auto const& carry : carries)
    {
        if (carry.idcpu != idcpu) { continue; }
        ++found;
        result = carry;
    }
    amrex::ParallelDescriptor::ReduceIntSum(found);
    amrex::ParallelDescriptor::ReduceIntSum(result.next_draw);
    amrex::ParallelDescriptor::ReduceIntSum(result.completed_events);
    amrex::ParallelDescriptor::ReduceRealSum(result.residual_optical_depth);
    AMREX_ALWAYS_ASSERT(found == 1);
    result.idcpu = idcpu;
    return result;
}

void
AllocateHistoryFields (History& history, amrex::BoxArray const& boxes,
                       amrex::DistributionMapping const& distribution)
{
    history.impulse_field = std::make_unique<amrex::MultiFab>(
        boxes, distribution, 3, 0);
    history.debit_field = std::make_unique<amrex::MultiFab>(
        boxes, distribution, 1, 0);
    history.scale_field = std::make_unique<amrex::MultiFab>(
        boxes, distribution, 1, 0);
    history.impulse_field->setVal(0);
    history.debit_field->setVal(0);
    history.scale_field->setVal(0);
}

void
AccumulatePathReport (History& history, NativePhotonPathReport const& report,
                      bool first)
{
    for (int component = 0; component < 4; ++component)
    {
        auto const index = static_cast<std::size_t>(component);
        if (first) { history.incoming[index] = report.incoming_inventory[component]; }
        history.outgoing[index] = report.outgoing_inventory[component];
    }
    history.events += report.collision_events;
    history.face_crossings += report.face_crossings;
}

} // namespace

#endif
