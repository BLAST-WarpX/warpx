/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativePhotonPathReplay.H"

#include "FiniteMassPhotonCollision.H"
#include "FrozenPhotonCollisionRate.H"
#include "NativePhotonCarryRouting.H"
#include "NativePhotonReplayInternals.H"
#include "PhotonFourMomentum.H"
#include "PhotonKeyAudit.H"
#include "PhotonLogicalDrawKey.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Box.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef AMREX_USE_MPI
#include <mpi.h>
#endif

namespace warpx::radiation
{
using namespace amrex::literals;
using photon_replay_detail::CaptureParticles;
using photon_replay_detail::GlobalFailureCode;
using photon_replay_detail::NearestCell;
using photon_replay_detail::OpenUnitInterval;
using photon_replay_detail::PhotonDrawKey;
using photon_replay_detail::SameBits;
using photon_replay_detail::SplitMix64;

namespace
{
constexpr amrex::Real pi = 3.141592653589793238462643383279502884_rt;

void
SetPathRejection (NativePhotonPathReport& report, int code)
{
    report = NativePhotonPathReport{};
    switch (code)
    {
    case 1: report.reason = "invalid collective inputs"; break;
    case 2: report.reason = "invalid native source photon"; break;
    case 3: report.reason = "optical depth exceeded frozen bound"; break;
    case 4: report.reason = "collision event count exceeded frozen bound"; break;
    case 5: report.reason = "invalid frozen random draw"; break;
    case 6: report.reason = "invalid replay layout"; break;
    case 7: report.reason = "private photon clone does not match frozen snapshot"; break;
    case 8: report.reason = "invalid staged material velocity"; break;
    case 9: report.reason = "invalid replay collision"; break;
    case 10: report.reason = "replay conservation identity failed"; break;
    case 11: report.reason = "replay face march failed"; break;
    case 12: report.reason = "negative or nonfinite distance to replay face"; break;
    case 13: report.reason = "replayed coordinate cannot represent its final cell"; break;
    default: report.reason = "unknown native photon path replay rejection"; break;
    }
}

struct HostField
{
    amrex::Box box{};
    int components = 0;
    std::vector<amrex::Real> values;

    [[nodiscard]] bool contains (amrex::IntVect const& cell) const
    {
        return box.contains(cell);
    }

    [[nodiscard]] amrex::Real operator() (
        amrex::IntVect const& cell, int component) const
    {
        auto const offset = static_cast<std::size_t>(
            box.index(cell) + box.numPts()*static_cast<long long>(component));
        return values[offset];
    }

    void add (amrex::IntVect const& cell, int component, amrex::Real value)
    {
        auto const offset = static_cast<std::size_t>(
            box.index(cell) + box.numPts()*static_cast<long long>(component));
        values[offset] += value;
    }
};

std::unordered_map<int, HostField>
CaptureHostFields (amrex::MultiFab const& fields, int number_components)
{
    std::unordered_map<int, HostField> result;
    for (amrex::MFIter iterator(fields); iterator.isValid(); ++iterator)
    {
        amrex::Box const box = iterator.fabbox();
        HostField host;
        host.box = box;
        host.components = number_components;
        host.values.resize(
            static_cast<std::size_t>(box.numPts()) *
            static_cast<std::size_t>(number_components), 0);
        auto const& fab = fields[iterator];
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, fab.dataPtr(),
            fab.dataPtr() + box.numPts() * number_components,
            host.values.begin());
        result.emplace(iterator.index(), std::move(host));
    }
    return result;
}

bool
ReplayStorageOverlaps (
    amrex::MultiFab const& velocity, amrex::MultiFab const& impulse,
    amrex::MultiFab const& debit, amrex::MultiFab const& scale)
{
    std::array<amrex::MultiFab const*, 4> const fields{&velocity, &impulse, &debit, &scale};
    for (amrex::MFIter it(velocity); it.isValid(); ++it)
    {
        for (std::size_t a = 0; a < fields.size(); ++a)
        {
            auto const& first = (*fields[a])[it];
            auto const start = reinterpret_cast<std::uintptr_t>(first.dataPtr());
            auto const bytes = static_cast<std::uintptr_t>(first.nBytes());
            if (!start || bytes > std::numeric_limits<std::uintptr_t>::max() - start)
            {
                return true;
            }
            for (std::size_t b = a + 1; b < fields.size(); ++b)
            {
                auto const& second = (*fields[b])[it];
                auto const other = reinterpret_cast<std::uintptr_t>(second.dataPtr());
                auto const size = static_cast<std::uintptr_t>(second.nBytes());
                if (!other || size > std::numeric_limits<std::uintptr_t>::max() - other ||
                    (start < other + size && other < start + bytes))
                {
                    return true;
                }
            }
        }
    }
    return false;
}

// Counts and displacements are uint64 words, never C++ struct bytes. The two
// collective gates keep ranks with an invalid local count out of Alltoallv.
bool
ExchangeCarryWords (
    std::vector<std::vector<std::uint64_t>> const& buckets, int width,
    std::vector<std::uint64_t>& received, std::vector<int>& recv_counts,
    std::vector<int>& recv_displacements)
{
    int const ranks = amrex::ParallelDescriptor::NProcs();
    std::vector<int> send_counts(static_cast<std::size_t>(ranks), 0);
    std::vector<int> send_displacements(static_cast<std::size_t>(ranks), 0);
    int invalid = buckets.size() != static_cast<std::size_t>(ranks) || width <= 0;
    std::vector<std::size_t> bucket_sizes;
    bucket_sizes.reserve(buckets.size());
    for (auto const& bucket : buckets) { bucket_sizes.push_back(bucket.size()); }
    if (!photon_replay_detail::ValidPackedWordCounts(bucket_sizes, width)) { invalid = 1; }
    int send_total = 0;
    if (!invalid)
    {
        for (int r = 0; r < ranks; ++r)
        {
            auto const size = buckets[static_cast<std::size_t>(r)].size();
            if (size % static_cast<std::size_t>(width) != 0 ||
                size > static_cast<std::size_t>(std::numeric_limits<int>::max() - send_total))
            {
                invalid = 1;
                break;
            }
            send_displacements[static_cast<std::size_t>(r)] = send_total;
            send_counts[static_cast<std::size_t>(r)] = static_cast<int>(size);
            send_total += static_cast<int>(size);
        }
    }
    if (GlobalFailureCode(invalid) != 0) { return false; }

    recv_counts.assign(static_cast<std::size_t>(ranks), 0);
    recv_displacements.assign(static_cast<std::size_t>(ranks), 0);
#ifdef AMREX_USE_MPI
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                 amrex::ParallelDescriptor::Communicator());
#else
    recv_counts[0] = send_counts[0];
#endif
    int recv_total = 0;
    std::vector<std::size_t> received_sizes;
    received_sizes.reserve(recv_counts.size());
    for (int r = 0; r < ranks; ++r)
    {
        int const count = recv_counts[static_cast<std::size_t>(r)];
        if (count < 0 || count % width != 0 ||
            count > std::numeric_limits<int>::max() - recv_total)
        {
            invalid = 1;
            break;
        }
        recv_displacements[static_cast<std::size_t>(r)] = recv_total;
        recv_total += count;
        received_sizes.push_back(static_cast<std::size_t>(count));
    }
    if (!invalid &&
        !photon_replay_detail::ValidPackedWordCounts(received_sizes, width))
    {
        invalid = 1;
    }
    if (GlobalFailureCode(invalid) != 0) { return false; }

    std::vector<std::uint64_t> send_words;
    send_words.reserve(static_cast<std::size_t>(send_total));
    for (auto const& bucket : buckets)
    {
        send_words.insert(send_words.end(), bucket.begin(), bucket.end());
    }
    received.resize(static_cast<std::size_t>(recv_total));
#ifdef AMREX_USE_MPI
    std::uint64_t empty_send = 0;
    std::uint64_t empty_recv = 0;
    MPI_Alltoallv(send_words.empty() ? &empty_send : send_words.data(),
                  send_counts.data(), send_displacements.data(),
                  MPI_UINT64_T, received.empty() ? &empty_recv : received.data(),
                  recv_counts.data(),
                  recv_displacements.data(), MPI_UINT64_T,
                  amrex::ParallelDescriptor::Communicator());
#else
    received = std::move(send_words);
#endif
    return true;
}

bool
ValidCarry (NativePhotonPathCarry const& carry, int event_cap)
{
    return event_cap >= 0 && event_cap < std::numeric_limits<int>::max() &&
        amrex::ConstParticleIDWrapper(carry.idcpu).is_valid() &&
        std::isfinite(carry.residual_optical_depth) &&
        carry.residual_optical_depth > 0 &&
        carry.completed_events >= 0 && carry.completed_events <= event_cap &&
        carry.next_draw == carry.completed_events + 1;
}

using RoutingIdentity = std::pair<std::uint64_t, std::uint64_t>;

RoutingIdentity
CarryIdentity (NativePhotonPathCarry const& carry)
{
    return carry.provisional_draw_key == 0 ? RoutingIdentity{0, carry.idcpu}
                                           : RoutingIdentity{1, carry.provisional_draw_key};
}

bool
ValidIdentity (RoutingIdentity const& identity)
{
    return identity.first == 0 ? amrex::ConstParticleIDWrapper(identity.second).is_valid()
                               : identity.first == 1 && identity.second != 0;
}

constexpr char ProvisionalMarker[] = "__photon_birth_provisional";

bool
SnapshotSchemaMatches (WarpXParticleContainer::Base const& snapshot,
                       WarpXParticleContainer const& owner, NativePhotonPathSnapshotPolicy policy)
{
    auto const names = snapshot.GetIntSoANames();
    auto const owner_names = owner.GetIntSoANames();
    if (snapshot.NumRealComps() != owner.NumRealComps() ||
        snapshot.GetRealSoANames() != owner.GetRealSoANames() ||
        std::find(owner_names.begin(), owner_names.end(), ProvisionalMarker) != owner_names.end()) {
        return false;
    }
    if (names == owner_names && snapshot.NumIntComps() == owner.NumIntComps()) {
        return true;
    }
    return policy == NativePhotonPathSnapshotPolicy::KeyedProvisional &&
           snapshot.NumIntComps() == owner.NumIntComps() + 1 &&
           names.size() == owner_names.size() + 1 && names.back() == ProvisionalMarker &&
           std::equal(owner_names.begin(), owner_names.end(), names.begin());
}

bool
CaptureRoutingIdentities (WarpXParticleContainer::Base const& particles,
                          NativePhotonPathSnapshotPolicy policy,
                          photon_replay_detail::CapturedParticles const& captured,
                          std::vector<RoutingIdentity>& identities,
                          std::vector<std::uint64_t>* draw_keys = nullptr)
{
    auto const names = particles.GetIntSoANames();
    auto const marker = std::find(names.begin(), names.end(), ProvisionalMarker);
    int const marker_component =
        marker == names.end() ? -1 : static_cast<int>(marker - names.begin());
    if (marker_component >= 0 && policy != NativePhotonPathSnapshotPolicy::KeyedProvisional) {
        return false;
    }
    std::array<int, 2> key_components{-1, -1};
    if (!FindPhotonDrawKeyComponents(particles.GetRealSoANames(), key_components)) {
        return false;
    }
    for (auto const& tile : captured.tiles) {
        for (long n = 0; n < tile.count; ++n) {
            auto const idcpu = tile.idcpu[static_cast<std::size_t>(n)];
            if (!amrex::ConstParticleIDWrapper(idcpu).is_valid()) {
                return false;
            }
            int const marked =
                marker_component < 0
                    ? 0
                    : tile.int_data[static_cast<std::size_t>(marker_component) * tile.count + n];
            if (marked != 0 && marked != 1) {
                return false;
            }
            std::uint64_t key = idcpu;
            if (key_components[0] >= 0) {
                auto const low =
                    tile.real_data[static_cast<std::size_t>(key_components[0]) * tile.count + n];
                auto const high =
                    tile.real_data[static_cast<std::size_t>(key_components[1]) * tile.count + n];
                if (!DecodePhotonDrawKey(low, high, idcpu, key) ||
                    (marked == 1 && low == 0 && high == 0)) {
                    return false;
                }
            } else if (marked == 1) {
                return false;
            }
            if (marked == 1 &&
                (key == 0 || static_cast<amrex::Long>(amrex::ConstParticleIDWrapper(idcpu)) != 1)) {
                return false;
            }
            identities.emplace_back(marked == 1 ? 1 : 0, marked == 1 ? key : idcpu);
            if (draw_keys != nullptr) {
                draw_keys->push_back(key);
            }
        }
    }
    return true;
}

bool
SameCapturedTiles (std::vector<photon_replay_detail::CapturedTile> const& a,
                   std::vector<photon_replay_detail::CapturedTile> const& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t n = 0; n < a.size(); ++n) {
        if (a[n].key != b[n].key || a[n].count != b[n].count ||
            !SameBits(a[n].real_data, b[n].real_data) || !SameBits(a[n].int_data, b[n].int_data) ||
            !SameBits(a[n].idcpu, b[n].idcpu)) {
            return false;
        }
    }
    return true;
}

std::array<std::uint64_t, 5>
PackCarry (NativePhotonPathCarry const& carry)
{
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    double const depth = static_cast<double>(carry.residual_optical_depth);
    std::uint64_t bits = 0;
    std::memcpy(&bits, &depth, sizeof(bits));
    return {carry.idcpu, bits, static_cast<std::uint64_t>(carry.next_draw),
            static_cast<std::uint64_t>(carry.completed_events), carry.provisional_draw_key};
}

bool
UnpackCarry (std::uint64_t const* words, int event_cap, NativePhotonPathCarry& carry)
{
    double depth = 0;
    std::memcpy(&depth, words + 1, sizeof(depth));
    if (words[2] > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        words[3] > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    carry.idcpu = words[0];
    carry.residual_optical_depth = static_cast<amrex::Real>(depth);
    carry.next_draw = static_cast<int>(words[2]);
    carry.completed_events = static_cast<int>(words[3]);
    carry.provisional_draw_key = words[4];
    return ValidCarry(carry, event_cap);
}

bool
RoutePhotonCarriesImpl (
    std::vector<NativePhotonPathCarry> const& local_carries,
    std::vector<RoutingIdentity> const& local_ids, int event_cap,
    std::vector<NativePhotonPathCarry>& migrated,
    std::vector<NativePhotonPathCarry> const* fresh_requests = nullptr,
    bool fresh_native = false,
    std::vector<std::uint64_t> const* birth_keys = nullptr,
    std::vector<std::uint64_t>* local_birth_keys = nullptr)
{
    int const ranks = amrex::ParallelDescriptor::NProcs();
    int const my_rank = amrex::ParallelDescriptor::MyProc();
    int invalid =
        sizeof(amrex::Real) != sizeof(double) || event_cap < 0 ||
        event_cap == std::numeric_limits<int>::max() ||
        (fresh_requests && fresh_requests->size() != local_ids.size());
    int minimum_history = fresh_requests != nullptr,
        maximum_history = minimum_history;
    int minimum_fresh = fresh_native, maximum_fresh = minimum_fresh;
    int minimum_births = (birth_keys != nullptr) + 2 * (local_birth_keys != nullptr);
    int maximum_births = minimum_births;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_history);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_history);
    amrex::ParallelDescriptor::ReduceIntMin(minimum_fresh);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_fresh);
    amrex::ParallelDescriptor::ReduceIntMin(minimum_births);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_births);
    invalid |=
        minimum_history != maximum_history || minimum_fresh != maximum_fresh ||
        minimum_births != maximum_births || (birth_keys && !fresh_requests) ||
        (local_birth_keys && (!birth_keys || fresh_native || local_birth_keys == birth_keys));
    int minimum_cap = event_cap;
    int maximum_cap = event_cap;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_cap);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_cap);
    if (minimum_cap != maximum_cap) { invalid = 1; }
    if (GlobalFailureCode(invalid) != 0) { return false; }
    auto const directory = [ranks] (RoutingIdentity const& identity) {
        auto const value =
            identity.first == 0 ? identity.second : identity.second ^ UINT64_C(0x17a68b9e0c731b2d);
        return static_cast<int>(SplitMix64(value) % static_cast<std::uint64_t>(ranks));
    };

    std::vector<std::vector<std::uint64_t>> carry_buckets(
        static_cast<std::size_t>(ranks));
    std::map<RoutingIdentity, bool> source_ids;
    for (auto const& carry : local_carries)
    {
        auto const identity = CarryIdentity(carry);
        if (!ValidCarry(carry, event_cap) || !source_ids.emplace(identity, true).second) {
            invalid = 1;
            continue;
        }
        auto const words = PackCarry(carry);
        auto& bucket = carry_buckets[static_cast<std::size_t>(directory(identity))];
        bucket.insert(bucket.end(), words.begin(), words.end());
    }
    if (GlobalFailureCode(invalid) != 0) { return false; }

    std::vector<std::uint64_t> received;
    std::vector<int> recv_counts, recv_displacements;
    if (!ExchangeCarryWords(carry_buckets, 5, received, recv_counts, recv_displacements)) {
        return false;
    }
    std::map<RoutingIdentity, NativePhotonPathCarry> directory_carries;
    for (std::size_t offset = 0; offset < received.size(); offset += 5) {
        NativePhotonPathCarry carry;
        if (!UnpackCarry(received.data() + offset, event_cap, carry) ||
            !directory_carries.emplace(CarryIdentity(carry), carry).second) {
            invalid = 1;
        }
    }
    if (GlobalFailureCode(invalid) != 0) { return false; }

    // Receipts come from ACTUAL staging, not endpoint set differences. Route
    // them to the same identity directory; each must be new and present once.
    std::map<std::uint64_t, bool> directory_births;
    if (birth_keys)
    {
        std::vector<std::vector<std::uint64_t>> birth_buckets(
            static_cast<std::size_t>(ranks));
        for (auto const key : *birth_keys)
        {
            if (key == 0)
            {
                invalid = 1;
            }
            birth_buckets[static_cast<std::size_t>(directory({1, key}))]
                .push_back(key);
        }
        if (GlobalFailureCode(invalid) != 0)
        {
            return false;
        }
        if (!ExchangeCarryWords(birth_buckets, 1, received, recv_counts,
                                recv_displacements))
        {
            return false;
        }
        for (auto const key : received)
        {
            if (!directory_births.emplace(key, true).second ||
                directory_carries.contains({1, key}))
            {
                invalid = 1;
            }
        }
        if (GlobalFailureCode(invalid) != 0)
        {
            return false;
        }
    }

    // Requests come from the actual owner after Redistribute, in its particle
    // traversal order. No idcpu creation CPU or geometric ownership guess is used.
    std::map<RoutingIdentity, bool> requested_ids;
    std::vector<std::vector<std::uint64_t>> request_buckets(
        static_cast<std::size_t>(ranks));
    std::size_t request_index = 0;
    for (auto const& identity : local_ids) {
        if (!ValidIdentity(identity) || !requested_ids.emplace(identity, true).second) {
            invalid = 1;
            continue;
        }
        auto& bucket = request_buckets[static_cast<std::size_t>(directory(identity))];
        bucket.push_back(identity.first);
        bucket.push_back(identity.second);
        bucket.push_back(static_cast<std::uint64_t>(my_rank));
        if (fresh_requests)
        {
            auto const& fresh = (*fresh_requests)[request_index];
            if (!ValidCarry(fresh, event_cap) ||
                CarryIdentity(fresh) != identity || fresh.next_draw != 1 ||
                fresh.completed_events != 0)
            {
                invalid = 1;
            }
            auto const words = PackCarry(fresh);
            bucket.insert(bucket.end(), words.begin(), words.end());
        }
        ++request_index;
    }
    if (GlobalFailureCode(invalid) != 0) { return false; }
    int const request_width = fresh_requests ? 8 : 3;
    if (!ExchangeCarryWords(request_buckets, request_width, received,
                            recv_counts, recv_displacements))
    {
        return false;
    }

    std::vector<std::vector<std::uint64_t>> reply_buckets(
        static_cast<std::size_t>(ranks));
    std::map<RoutingIdentity, bool> served_history;
    for (int source = 0; source < ranks; ++source)
    {
        int const end = recv_displacements[static_cast<std::size_t>(source)] +
            recv_counts[static_cast<std::size_t>(source)];
        for (int offset = recv_displacements[static_cast<std::size_t>(source)];
             offset < end; offset += request_width)
        {
            RoutingIdentity const identity{received[static_cast<std::size_t>(offset)],
                                           received[static_cast<std::size_t>(offset + 1)]};
            std::uint64_t const owner = received[static_cast<std::size_t>(offset + 2)];
            auto found = directory_carries.find(identity);
            NativePhotonPathCarry selected;
            bool const have_old = found != directory_carries.end();
            bool const valid_fresh =
                fresh_requests &&
                (fresh_native ||
                 (identity.first == 1 &&
                  directory_births.contains(identity.second))) &&
                UnpackCarry(received.data() + offset + 3, event_cap,
                            selected) &&
                CarryIdentity(selected) == identity;
            if (owner != static_cast<std::uint64_t>(source) ||
                (!have_old && !valid_fresh) ||
                (fresh_requests &&
                 !served_history.emplace(identity, true).second))
            {
                invalid = 1;
                continue;
            }
            if (have_old)
            {
                selected = found->second;
            }
            auto const words = PackCarry(selected);
            auto& bucket = reply_buckets[static_cast<std::size_t>(source)];
            bucket.insert(bucket.end(), words.begin(), words.end());
            if (local_birth_keys)
            {
                // This is the SAME actual receipt decision used to initialize
                // the carry. Never infer births from zero events or native IDs.
                bucket.push_back(!have_old);
            }
            if (have_old)
            {
                directory_carries.erase(found);
            }
            else if (!fresh_native)
            {
                directory_births.erase(identity.second);
            }
        }
    }
    if ((!fresh_requests && !directory_carries.empty()) ||
        !directory_births.empty())
    {
        invalid = 1;
    }
    // An unmatched carry or request must reject on every rank before replies.
    if (GlobalFailureCode(invalid) != 0) { return false; }
    int const reply_width = local_birth_keys ? 6 : 5;
    if (!ExchangeCarryWords(reply_buckets, reply_width, received, recv_counts, recv_displacements)) {
        return false;
    }

    std::map<RoutingIdentity, NativePhotonPathCarry> received_by_id;
    std::set<RoutingIdentity> received_births;
    for (std::size_t offset = 0; offset < received.size(); offset += reply_width) {
        NativePhotonPathCarry carry;
        if (!UnpackCarry(received.data() + offset, event_cap, carry) ||
            !received_by_id.emplace(CarryIdentity(carry), carry).second) {
            invalid = 1;
        }
        if (local_birth_keys)
        {
            auto const born = received[offset + 5];
            if (born > 1 || (born == 1 &&
                (carry.provisional_draw_key == 0 || carry.completed_events != 0 ||
                 carry.next_draw != 1 || !received_births.insert(CarryIdentity(carry)).second)))
            {
                invalid = 1;
            }
        }
    }
    std::vector<NativePhotonPathCarry> candidate;
    std::vector<std::uint64_t> candidate_births;
    candidate.reserve(local_ids.size());
    for (auto const& identity : local_ids) {
        auto found = received_by_id.find(identity);
        if (found == received_by_id.end())
        {
            invalid = 1;
            continue;
        }
        candidate.push_back(found->second);
        if (local_birth_keys && received_births.erase(identity) != 0)
        {
            candidate_births.push_back(identity.second);
        }
        received_by_id.erase(found);
    }
    if (!received_by_id.empty() || !received_births.empty() ||
        candidate.size() != local_ids.size()) { invalid = 1; }
    if (GlobalFailureCode(invalid) != 0) { return false; }
    migrated = std::move(candidate);
    if (local_birth_keys) { *local_birth_keys = std::move(candidate_births); }
    return true;
}

bool
MigratePhotonCarries (WarpXParticleContainer::Base const& particles,
                      std::vector<NativePhotonPathCarry> const& local_carries, int event_cap,
                      std::vector<NativePhotonPathCarry>& migrated,
                      NativePhotonPathSnapshotPolicy policy)
{
    auto const captured = CaptureParticles(particles);
    std::vector<RoutingIdentity> local_ids;
    if (GlobalFailureCode(
            CaptureRoutingIdentities(particles, policy, captured, local_ids) ? 0 : 1) != 0) {
        return false;
    }
    return RoutePhotonCarriesImpl(local_carries, local_ids, event_cap, migrated);
}

void
RestoreHostFields (
    amrex::MultiFab& fields,
    std::unordered_map<int, HostField> const& hosts)
{
    for (amrex::MFIter iterator(fields); iterator.isValid(); ++iterator)
    {
        auto found = hosts.find(iterator.index());
        if (found == hosts.end()) { continue; }
        HostField const& host = found->second;
        auto& fab = fields[iterator];
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, host.values.begin(), host.values.end(),
            fab.dataPtr());
    }
}

std::pair<std::unordered_map<int, HostField>,
          std::unordered_map<int, HostField>>
CaptureFrozenGrids (
    amrex::MultiFab const& opacity, amrex::MultiFab const& rate_velocity,
    amrex::Geometry const& geometry)
{
    amrex::MultiFab compact_opacity(
        opacity.boxArray(), opacity.DistributionMap(), 1, 1);
    amrex::MultiFab compact_velocity(
        rate_velocity.boxArray(), rate_velocity.DistributionMap(), 3, 1);
    amrex::MultiFab::Copy(compact_opacity, opacity, 0, 0, 1, 0);
    amrex::MultiFab::Copy(compact_velocity, rate_velocity, 0, 0, 3, 0);
    compact_opacity.FillBoundary(geometry.periodicity());
    compact_velocity.FillBoundary(geometry.periodicity());
    return {CaptureHostFields(compact_opacity, 1),
            CaptureHostFields(compact_velocity, 3)};
}

constexpr int
SpatialIndexOfPhysicalAxis (int physical_axis) noexcept
{
#if defined(WARPX_DIM_1D_Z)
    return physical_axis == 2 ? 0 : -1;
#elif defined(WARPX_DIM_XZ)
    return physical_axis == 0 ? 0 : (physical_axis == 2 ? 1 : -1);
#elif defined(WARPX_DIM_3D)
    return physical_axis;
#else
    static_cast<void>(physical_axis);
    return -1;
#endif
}

constexpr int
PhysicalAxisOfSpatialIndex (int spatial_dimension) noexcept
{
#if defined(WARPX_DIM_1D_Z)
    return spatial_dimension == 0 ? 2 : -1;
#elif defined(WARPX_DIM_XZ)
    return spatial_dimension == 0 ? 0 :
        (spatial_dimension == 1 ? 2 : -1);
#elif defined(WARPX_DIM_3D)
    return spatial_dimension;
#else
    static_cast<void>(spatial_dimension);
    return -1;
#endif
}

[[maybe_unused, nodiscard]] amrex::Real
WrapPeriodicCoordinate (
    amrex::Geometry const& geometry, int dimension, amrex::Real coordinate)
{
    amrex::Real const lower = geometry.ProbLo(dimension);
    amrex::Real const upper = geometry.ProbHi(dimension);
    amrex::Real const length = geometry.ProbHi(dimension) - lower;
    long double const exact = static_cast<long double>(lower) +
        std::fmod(
            static_cast<long double>(coordinate) -
                static_cast<long double>(lower),
            static_cast<long double>(length));
    long double const canonical =
        exact < static_cast<long double>(lower)
            ? exact + static_cast<long double>(length)
            : exact;
    amrex::Real wrapped = static_cast<amrex::Real>(canonical);
    if (wrapped >= upper) { wrapped = std::nextafter(upper, lower); }
    return wrapped;
}

#if defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D)
[[nodiscard]] bool
CanonicalPositionInCell (
    amrex::Geometry const& geometry, int dimension, amrex::Real position,
    int final_cell, amrex::ParticleReal& published)
{
    auto const& domain = geometry.Domain();
    int const first = domain.smallEnd(dimension);
    int const last = domain.bigEnd(dimension);
    int const count = domain.length(dimension);
    if (count <= 0) { return false; }
    bool const periodic = geometry.isPeriodic(dimension);
    if (!periodic && (final_cell < first || final_cell > last)) { return false; }
    int const periodic_cell = first +
        ((final_cell - first) % count + count) % count;
    amrex::Real const lower = geometry.ProbLo(dimension);
    amrex::Real const upper = geometry.ProbHi(dimension);
    amrex::Real const dx = geometry.CellSize(dimension);
    amrex::Real const inverse_dx = geometry.InvCellSize(dimension);
    amrex::Real const wrapped = periodic ?
        WrapPeriodicCoordinate(geometry, dimension, position) : position;
    amrex::Real const particle_lower = static_cast<amrex::Real>(
        geometry.ProbLoArrayInParticleReal()[dimension]);
    amrex::Real const particle_upper = static_cast<amrex::Real>(
        geometry.ProbHiArrayInParticleReal()[dimension]);
    auto const replay_cell = [&](amrex::Real coordinate)
    {
        return first + static_cast<int>(std::floor((coordinate - lower) / dx));
    };
    auto const particle_cell = [&](amrex::Real coordinate)
    {
        return first + static_cast<int>(std::floor((coordinate - lower) * inverse_dx));
    };
    if (!amrex::Math::isfinite(wrapped) || wrapped < lower ||
        (periodic ? wrapped >= upper : wrapped > upper))
    {
        return false;
    }
    auto const fits_cell = [&](amrex::Real coordinate)
    {
        amrex::Real const represented = static_cast<amrex::Real>(
            static_cast<amrex::ParticleReal>(coordinate));
        return represented >= particle_lower && represented <= particle_upper &&
            replay_cell(represented) == periodic_cell &&
            particle_cell(represented) == periodic_cell;
    };
    int const replay_index = replay_cell(wrapped);
    int const particle_index = particle_cell(wrapped);
    if (fits_cell(wrapped))
    {
        published = static_cast<amrex::ParticleReal>(wrapped);
        return true;
    }

    // AMReX's roundoff-safe particle domain can begin above physical ProbLo.
    amrex::Real const coordinate_scale =
        std::max({dx, std::abs(lower), std::abs(upper), std::abs(wrapped)});
    amrex::Real const proximity = std::min(
        8 * std::numeric_limits<amrex::Real>::epsilon() * coordinate_scale,
        dx / 8);
    if (periodic_cell == first && wrapped < particle_lower &&
        particle_lower - wrapped <= proximity && fits_cell(particle_lower))
    {
        published = static_cast<amrex::ParticleReal>(particle_lower);
        return true;
    }

    // An exact face reached from the left can represent the right cell when
    // the physical path finishes in the left one. Preserve that face side in
    // the native coordinate so Redistribute and the next Prepare agree.
    int const right_cell = periodic && periodic_cell == last ? first : periodic_cell + 1;
    auto const periodic_index = [&](int index)
    {
        return periodic && index == last + 1 ? first : index;
    };
    if ((periodic_index(replay_index) != periodic_cell &&
         periodic_index(replay_index) != right_cell) ||
        (periodic_index(particle_index) != periodic_cell &&
         periodic_index(particle_index) != right_cell))
    {
        return false;
    }
    amrex::Real const face = periodic_cell == last ? upper :
        lower + static_cast<amrex::Real>(periodic_cell - first + 1) * dx;
    amrex::Real const face_scale = coordinate_scale;
    amrex::Real const face_error = periodic_cell == last ?
        std::min(std::abs(wrapped - lower), std::abs(wrapped - upper)) :
        std::abs(wrapped - face);
    if (face_error > proximity)
    {
        return false;
    }
    // The AMReX safe upper bound can lie farther inside the domain than a
    // representable face correction. Check the final value after any clamp.
    auto const bounded_candidate = [&](amrex::Real coordinate)
    {
        return coordinate < face && face - coordinate <= proximity &&
            coordinate >= lower && coordinate < upper && fits_cell(coordinate);
    };
    amrex::Real candidate = face;
    for (int ulp = 0; ulp < 16; ++ulp)
    {
        candidate = std::nextafter(candidate, lower);
        candidate = std::min(candidate, particle_upper);
        if (bounded_candidate(candidate))
        {
            published = static_cast<amrex::ParticleReal>(candidate);
            return true;
        }
    }
    // Near zero, stepping the coordinate by a few ULPs can leave
    // coordinate-lower unchanged. Search a bounded inward distance at the
    // scale of that subtraction, and require both index formulas to agree.
    for (int factor = 1; factor <= 8; factor *= 2)
    {
        amrex::Real const inward = std::min(
            static_cast<amrex::Real>(factor) *
                std::numeric_limits<amrex::Real>::epsilon() * face_scale,
            dx / 8);
        candidate = std::min(face - inward, particle_upper);
        if (bounded_candidate(candidate))
        {
            published = static_cast<amrex::ParticleReal>(candidate);
            return true;
        }
    }
    return false;
}
#endif

[[nodiscard]] amrex::Real
InverseRate (
    NativePhotonCollisionOptions const& options, amrex::Real opacity,
    amrex::GpuArray<amrex::Real, 3> const& beta0,
    amrex::GpuArray<amrex::Real, 3> const& direction)
{
    if (!options.moving_cross_sections) { return opacity; }
    auto const rate = EvaluateFrozenPhotonCollisionRate(opacity, beta0, direction);
    return rate.valid ? rate.inverse_length :
                        std::numeric_limits<amrex::Real>::quiet_NaN();
}

[[nodiscard]] amrex::Real
SampleOpticalDepth (
    NativePhotonCollisionOptions const& options, std::uint64_t idcpu,
    int event)
{
    return -std::log(OpenUnitInterval(
        PhotonDrawKey(
            options.seed, idcpu, options.interval,
            static_cast<std::uint64_t>(event), 0)));
}

[[nodiscard]] amrex::GpuArray<amrex::Real, 3>
SampleEventDirection (
    NativePhotonCollisionOptions const& options, std::uint64_t idcpu,
    int event)
{
    if (options.angular_law ==
        NativePhotonCollisionAngularLaw::FixedFrameDirection)
    {
        return options.fixed_frame_direction;
    }
    amrex::Real const cosine =
        2 * OpenUnitInterval(PhotonDrawKey(
            options.seed, idcpu, options.interval,
            static_cast<std::uint64_t>(event), 1)) - 1;
    amrex::Real const azimuth =
        2 * pi * OpenUnitInterval(PhotonDrawKey(
            options.seed, idcpu, options.interval,
            static_cast<std::uint64_t>(event), 2));
    amrex::Real const sine = std::sqrt(1 - cosine * cosine);
    return {sine * std::cos(azimuth), sine * std::sin(azimuth), cosine};
}

[[nodiscard]] bool
SameOptionsAcrossRanks (NativePhotonCollisionOptions const& options, amrex::Real dt)
{
    amrex::Long integers[6] = {
        static_cast<amrex::Long>(options.seed),
        static_cast<amrex::Long>(options.interval),
        static_cast<amrex::Long>(options.max_events_per_photon),
        static_cast<amrex::Long>(options.angular_law ==
                                 NativePhotonCollisionAngularLaw::Isotropic),
        static_cast<amrex::Long>(options.absorption),
        static_cast<amrex::Long>(options.moving_cross_sections)};
    amrex::Long minimum_integers[6];
    amrex::Long maximum_integers[6];
    std::copy(std::begin(integers), std::end(integers), minimum_integers);
    std::copy(std::begin(integers), std::end(integers), maximum_integers);
    amrex::ParallelDescriptor::ReduceLongMin(minimum_integers, 6);
    amrex::ParallelDescriptor::ReduceLongMax(maximum_integers, 6);
    if (!std::equal(
            std::begin(minimum_integers), std::end(minimum_integers),
            std::begin(maximum_integers)))
    {
        return false;
    }
    amrex::Real reals[5] = {
        dt, options.max_optical_depth, options.fixed_frame_direction[0],
        options.fixed_frame_direction[1], options.fixed_frame_direction[2]};
    amrex::Real minimum_reals[5];
    amrex::Real maximum_reals[5];
    std::copy(std::begin(reals), std::end(reals), minimum_reals);
    std::copy(std::begin(reals), std::end(reals), maximum_reals);
    amrex::ParallelDescriptor::ReduceRealMin(minimum_reals, 5);
    amrex::ParallelDescriptor::ReduceRealMax(maximum_reals, 5);
    return std::equal(
        std::begin(minimum_reals), std::end(minimum_reals),
        std::begin(maximum_reals));
}

[[nodiscard]] bool
SameGeometryAcrossRanks (amrex::Geometry const& geometry)
{
    amrex::Long integers[2 + 2 * AMREX_SPACEDIM] = {
        geometry.Coord(), 0};
    for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
    {
        integers[2 + 2 * dimension] =
            geometry.Domain().smallEnd(dimension);
        integers[3 + 2 * dimension] =
            geometry.Domain().bigEnd(dimension);
        integers[1] = 2 * integers[1] + (geometry.isPeriodic(dimension) ? 1 : 0);
    }
    amrex::Long minimum_integers[2 + 2 * AMREX_SPACEDIM];
    amrex::Long maximum_integers[2 + 2 * AMREX_SPACEDIM];
    std::copy(std::begin(integers), std::end(integers), minimum_integers);
    std::copy(std::begin(integers), std::end(integers), maximum_integers);
    amrex::ParallelDescriptor::ReduceLongMin(
        minimum_integers, 2 + 2 * AMREX_SPACEDIM);
    amrex::ParallelDescriptor::ReduceLongMax(
        maximum_integers, 2 + 2 * AMREX_SPACEDIM);
    if (!std::equal(
            std::begin(minimum_integers), std::end(minimum_integers),
            std::begin(maximum_integers)))
    {
        return false;
    }
    amrex::Real cell_sizes[AMREX_SPACEDIM];
    for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
    {
        cell_sizes[dimension] = geometry.CellSize(dimension);
    }
    amrex::Real minimum_sizes[AMREX_SPACEDIM];
    amrex::Real maximum_sizes[AMREX_SPACEDIM];
    std::copy(std::begin(cell_sizes), std::end(cell_sizes), minimum_sizes);
    std::copy(std::begin(cell_sizes), std::end(cell_sizes), maximum_sizes);
    amrex::ParallelDescriptor::ReduceRealMin(minimum_sizes, AMREX_SPACEDIM);
    amrex::ParallelDescriptor::ReduceRealMax(maximum_sizes, AMREX_SPACEDIM);
    return std::equal(
        std::begin(minimum_sizes), std::end(minimum_sizes),
        std::begin(maximum_sizes));
}
}

bool
photon_replay_detail::ValidPackedWordCounts (
    std::vector<std::size_t> const& word_counts, int record_width)
{
    if (record_width <= 0) { return false; }
    std::size_t total = 0;
    constexpr std::size_t limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
    for (auto const count : word_counts)
    {
        if (count % static_cast<std::size_t>(record_width) != 0 || count > limit - total)
        {
            return false;
        }
        total += count;
    }
    return true;
}

bool
photon_replay_detail::RoutePhotonCarries (
    std::vector<NativePhotonPathCarry> const& local_carries,
    std::vector<std::uint64_t> const& local_post_redistribute_ids,
    int event_cap, std::vector<NativePhotonPathCarry>& migrated)
{
    // This public seam remains native-only, including its legacy RZ callers.
    int invalid = 0;
    for (auto const& carry : local_carries) {
        invalid |= carry.provisional_draw_key != 0;
    }
    if (GlobalFailureCode(invalid) != 0) {
        return false;
    }
    std::vector<RoutingIdentity> identities;
    identities.reserve(local_post_redistribute_ids.size());
    for (auto const id : local_post_redistribute_ids) {
        identities.emplace_back(0, id);
    }
    return RoutePhotonCarriesImpl(local_carries, identities, event_cap, migrated);
}

bool
photon_replay_detail::CapturePhotonRoutingKeys (WarpXParticleContainer::Base const& snapshot,
                                                WarpXParticleContainer const& owner,
                                                NativePhotonPathSnapshotPolicy policy,
                                                std::vector<std::uint64_t>& native_ids,
                                                std::vector<std::uint64_t>& provisional_keys,
                                                std::vector<std::uint64_t>& draw_keys)
{
    if ((policy != NativePhotonPathSnapshotPolicy::NativeOnly &&
         policy != NativePhotonPathSnapshotPolicy::KeyedProvisional) ||
        !SnapshotSchemaMatches(snapshot, owner, policy)) {
        return false;
    }
    auto const captured = CaptureParticles(snapshot);
    std::vector<RoutingIdentity> identities;
    std::vector<std::uint64_t> keys, ids, marked;
    if (!CaptureRoutingIdentities(snapshot, policy, captured, identities, &keys)) {
        return false;
    }
    std::size_t n = 0;
    for (auto const& tile : captured.tiles) {
        for (auto const id : tile.idcpu) {
            ids.push_back(id);
            marked.push_back(identities[n].first == 1 ? identities[n].second : 0);
            ++n;
        }
    }
    native_ids = std::move(ids);
    provisional_keys = std::move(marked);
    draw_keys = std::move(keys);
    return true;
}

bool
photon_replay_detail::RouteKeyedPhotonCarries (
    std::vector<NativePhotonPathCarry> const& local_carries,
    std::vector<std::uint64_t> const& local_post_redistribute_ids,
    std::vector<std::uint64_t> const& local_provisional_keys, int event_cap,
    std::vector<NativePhotonPathCarry>& migrated)
{
    int invalid = local_post_redistribute_ids.size() != local_provisional_keys.size();
    if (!invalid) {
        for (std::size_t n = 0; n < local_post_redistribute_ids.size(); ++n) {
            auto const id = amrex::ConstParticleIDWrapper(local_post_redistribute_ids[n]);
            invalid |= !id.is_valid() ||
                       (local_provisional_keys[n] != 0 && static_cast<amrex::Long>(id) != 1);
        }
    }
    if (GlobalFailureCode(invalid) != 0) {
        return false;
    }
    std::vector<RoutingIdentity> identities;
    identities.reserve(local_post_redistribute_ids.size());
    for (std::size_t n = 0; n < local_post_redistribute_ids.size(); ++n) {
        auto const key = local_provisional_keys[n];
        identities.emplace_back(key == 0 ? 0 : 1, key == 0 ? local_post_redistribute_ids[n] : key);
    }
    std::vector<NativePhotonPathCarry> candidate;
    if (!RoutePhotonCarriesImpl(local_carries, identities, event_cap, candidate)) {
        return false;
    }
    invalid = 0;
    for (std::size_t n = 0; n < candidate.size(); ++n) {
        invalid |= candidate[n].idcpu != local_post_redistribute_ids[n] ||
                   candidate[n].provisional_draw_key != local_provisional_keys[n];
    }
    if (GlobalFailureCode(invalid) != 0) {
        return false;
    }
    migrated = std::move(candidate);
    return true;
}

bool
photon_replay_detail::ReconcileKeyedPhotonHistoryCarries (
    std::vector<NativePhotonPathCarry> const& previous,
    WarpXParticleContainer::Base const* preceding,
    WarpXParticleContainer::Base const& endpoint,
    WarpXParticleContainer const& owner,
    NativePhotonCollisionOptions const& options, bool fresh_outer,
    std::vector<NativePhotonPathCarry>& output,
    std::vector<std::uint64_t> const* birth_keys,
    std::vector<std::uint64_t>* local_birth_keys)
{
    int invalid = (fresh_outer ? preceding != nullptr || !previous.empty()
                               : preceding == nullptr) ||
        (local_birth_keys && (fresh_outer || !birth_keys || local_birth_keys == birth_keys));
    int minimum_fresh = fresh_outer + 2 * (local_birth_keys != nullptr);
    int maximum_fresh = minimum_fresh;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_fresh);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_fresh);
    invalid |= minimum_fresh != maximum_fresh;
#ifdef AMREX_USE_MPI
    std::uint64_t low[2]{options.seed, options.interval};
    std::uint64_t high[2]{options.seed, options.interval};
    MPI_Allreduce(MPI_IN_PLACE, low, 2, MPI_UINT64_T, MPI_MIN,
                  amrex::ParallelDescriptor::Communicator());
    MPI_Allreduce(MPI_IN_PLACE, high, 2, MPI_UINT64_T, MPI_MAX,
                  amrex::ParallelDescriptor::Communicator());
    invalid |= low[0] != high[0] || low[1] != high[1];
#endif
    if (GlobalFailureCode(invalid) != 0)
    {
        return false;
    }
    if (!fresh_outer)
    {
        // Prove the COMPLETE preceding population has clocks BEFORE dropping
        // absorptions or initializing new births. An old marked ray with a
        // missing clock must never be mistaken for a fresh thermal birth.
        std::vector<std::uint64_t> old_ids, old_keys, old_draws;
        if (GlobalFailureCode(
                CapturePhotonRoutingKeys(
                    *preceding, owner,
                    NativePhotonPathSnapshotPolicy::KeyedProvisional, old_ids,
                    old_keys, old_draws)
                    ? 0
                    : 1) != 0)
        {
            return false;
        }
        std::vector<NativePhotonPathCarry> exact_previous;
        if (!RouteKeyedPhotonCarries(previous, old_ids, old_keys,
                                     options.max_events_per_photon,
                                     exact_previous))
        {
            return false;
        }
    }
    std::vector<std::uint64_t> ids, keys, draws;
    if (GlobalFailureCode(CapturePhotonRoutingKeys(
                              endpoint, owner,
                              NativePhotonPathSnapshotPolicy::KeyedProvisional,
                              ids, keys, draws)
                              ? 0
                              : 1) != 0)
    {
        return false;
    }
    std::vector<RoutingIdentity> identities;
    std::vector<NativePhotonPathCarry> fresh;
    for (std::size_t n = 0; n < ids.size(); ++n)
    {
        identities.emplace_back(keys[n] == 0 ? 0 : 1,
                                keys[n] == 0 ? ids[n] : keys[n]);
        auto const depth = -std::log(OpenUnitInterval(
            PhotonDrawKey(options.seed, draws[n], options.interval, 0, 0)));
        fresh.push_back({ids[n], 1, 0, depth, keys[n]});
    }
    std::vector<NativePhotonPathCarry> candidate;
    std::vector<std::uint64_t> candidate_births;
    if (!RoutePhotonCarriesImpl(previous, identities,
                                options.max_events_per_photon, candidate,
                                &fresh, fresh_outer, birth_keys,
                                local_birth_keys ? &candidate_births : nullptr))
    {
        return false;
    }
    invalid = 0;
    for (std::size_t n = 0; n < candidate.size(); ++n)
    {
        invalid |= candidate[n].idcpu != ids[n] ||
                   candidate[n].provisional_draw_key != keys[n];
    }
    if (GlobalFailureCode(invalid) != 0)
    {
        return false;
    }
    output = std::move(candidate);
    if (local_birth_keys) { *local_birth_keys = std::move(candidate_births); }
    return true;
}

struct NativePhotonPathReplay::Impl
{
    amrex::Geometry geometry{};
    NativePhotonCollisionOptions options{};
    NativePhotonPathBoundaries boundaries{};
    amrex::Real dt = 0;
    amrex::BoxArray box_array;
    amrex::DistributionMapping distribution_map;
    std::vector<photon_replay_detail::CapturedTile> baseline;
    int number_real_components = 0;
    int number_int_components = 0;
    std::vector<std::string> real_names;
    std::vector<std::string> int_names;

    std::vector<std::uint64_t> idcpu;
    std::vector<std::uint64_t> draw_key;
    std::vector<amrex::ParticleReal> weight;
    std::vector<amrex::Real> incoming;
    std::vector<amrex::ParticleReal> initial_u;
    std::vector<amrex::Real> initial_position;
    std::vector<int> initial_cell;
    std::vector<NativePhotonPathCarry> initial_carries;
    bool resume_carries = false;
    NativePhotonPathSnapshotPolicy snapshot_policy = NativePhotonPathSnapshotPolicy::NativeOnly;
    amrex::MultiFab frozen_opacity;
    amrex::MultiFab frozen_rate_velocity;
    bool prepared = false;
};

NativePhotonPathReplay::NativePhotonPathReplay ()
    : m_impl(std::make_unique<Impl>()) {}

NativePhotonPathReplay::~NativePhotonPathReplay () = default;

bool
NativePhotonPathReplay::Prepare (
    WarpXParticleContainer& photons, amrex::MultiFab const& scattering_opacity,
    amrex::MultiFab const& initial_rate_velocity, amrex::Geometry const& geometry,
    amrex::Real dt, NativePhotonCollisionOptions const& options,
    NativePhotonPathReport& report, NativePhotonPathBoundaries const& boundaries)
{
    return Prepare(
        photons, photons, scattering_opacity, initial_rate_velocity, geometry,
        dt, options, report, boundaries);
}

bool
NativePhotonPathReplay::Prepare (
    WarpXParticleContainer::Base const& private_snapshot,
    WarpXParticleContainer const& photon_species, amrex::MultiFab const& scattering_opacity,
    amrex::MultiFab const& initial_rate_velocity, amrex::Geometry const& geometry, amrex::Real dt,
    NativePhotonCollisionOptions const& options, NativePhotonPathReport& report,
    NativePhotonPathBoundaries const& boundaries, NativePhotonPathSnapshotPolicy snapshot_policy)
{
    return PrepareImpl(private_snapshot, photon_species, scattering_opacity, initial_rate_velocity,
                       geometry, dt, options, {}, false, report, boundaries, snapshot_policy);
}

bool
NativePhotonPathReplay::Prepare (
    WarpXParticleContainer::Base const& private_snapshot,
    WarpXParticleContainer const& photon_species, amrex::MultiFab const& scattering_opacity,
    amrex::MultiFab const& initial_rate_velocity, amrex::Geometry const& geometry, amrex::Real dt,
    NativePhotonCollisionOptions const& options,
    std::vector<NativePhotonPathCarry> const& initial_carries, NativePhotonPathReport& report,
    NativePhotonPathBoundaries const& boundaries, NativePhotonPathSnapshotPolicy snapshot_policy)
{
    return PrepareImpl(private_snapshot, photon_species, scattering_opacity, initial_rate_velocity,
                       geometry, dt, options, initial_carries, true, report, boundaries,
                       snapshot_policy);
}

bool
NativePhotonPathReplay::PrepareImpl (WarpXParticleContainer::Base const& private_snapshot,
                                     WarpXParticleContainer const& photon_species,
                                     amrex::MultiFab const& scattering_opacity,
                                     amrex::MultiFab const& initial_rate_velocity,
                                     amrex::Geometry const& geometry, amrex::Real dt,
                                     NativePhotonCollisionOptions const& options,
                                     std::vector<NativePhotonPathCarry> const& initial_carries,
                                     bool resume_carries, NativePhotonPathReport& report,
                                     NativePhotonPathBoundaries const& boundaries,
                                     NativePhotonPathSnapshotPolicy snapshot_policy)
{
    auto const& photons = private_snapshot;
    m_impl = std::make_unique<Impl>();
    report = NativePhotonPathReport{};

    int invalid_inputs = 0;
    int minimum_policy = static_cast<int>(snapshot_policy);
    int maximum_policy = minimum_policy;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_policy);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_policy);
    if (minimum_policy != maximum_policy || minimum_policy < 0 || maximum_policy > 1) {
        invalid_inputs = 1;
    }
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        int low[2] = {boundaries.escape_lo[d], boundaries.escape_hi[d]};
        int high[2] = {low[0], low[1]};
        amrex::ParallelDescriptor::ReduceIntMin(low, 2);
        amrex::ParallelDescriptor::ReduceIntMax(high, 2);
        for (int side = 0; side < 2; ++side) {
            if (low[side] != high[side] || low[side] < 0 || high[side] > 1 ||
                (geometry.isPeriodic(d) && high[side] != 0)) { invalid_inputs = 1; }
        }
    }
    int minimum_resume = resume_carries ? 1 : 0;
    int maximum_resume = minimum_resume;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_resume);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_resume);
    if (minimum_resume != maximum_resume) { invalid_inputs = 1; }
#if !defined(WARPX_DIM_1D_Z) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_3D)
    invalid_inputs = 1;
#endif
    if constexpr (std::numeric_limits<amrex::Real>::digits < 53 ||
                  std::numeric_limits<amrex::ParticleReal>::digits < 53)
    {
        invalid_inputs = 1;
    }
    amrex::Real direction_squared = 0;
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        if (!amrex::Math::isfinite(options.fixed_frame_direction[dimension]))
        {
            invalid_inputs = 1;
        }
        direction_squared += options.fixed_frame_direction[dimension] *
                             options.fixed_frame_direction[dimension];
    }
    if (options.absorption || options.emission ||
        options.max_events_per_photon <= 0 ||
        options.max_events_per_photon == std::numeric_limits<int>::max() ||
        (options.angular_law !=
             NativePhotonCollisionAngularLaw::Isotropic &&
         options.angular_law !=
             NativePhotonCollisionAngularLaw::FixedFrameDirection) ||
        !amrex::Math::isfinite(options.max_optical_depth) ||
        !(options.max_optical_depth > 0) ||
        !amrex::Math::isfinite(dt) || !(dt > 0) ||
        !amrex::Math::isfinite(direction_squared) || !(direction_squared > 0) ||
        std::abs(direction_squared - 1) >
            256 * std::numeric_limits<amrex::Real>::epsilon() * direction_squared)
    {
        invalid_inputs = 1;
    }
    bool const same_inputs = SameOptionsAcrossRanks(options, dt) &&
        SameGeometryAcrossRanks(geometry);
    int const global_invalid_inputs = GlobalFailureCode(invalid_inputs);
    if (global_invalid_inputs != 0 || !same_inputs)
    {
        SetPathRejection(report, global_invalid_inputs == 0 ? 1 :
                                                              global_invalid_inputs);
        return false;
    }

    int invalid_layout = 0;
    auto const& photon_boxes = photons.ParticleBoxArray(0);
    auto const& photon_distribution = photons.ParticleDistributionMap(0);
    if (photons.finestLevel() != 0 || geometry.Coord() != 0 || photon_species.finestLevel() != 0 ||
        !photon_species.AmIA<PhysicalSpecies::photon>() ||
        photon_species.ParticleBoxArray(0) != photon_boxes ||
        photon_species.ParticleDistributionMap(0) != photon_distribution ||
        !SnapshotSchemaMatches(photons, photon_species, snapshot_policy) ||
        scattering_opacity.nComp() != 1 || initial_rate_velocity.nComp() != 3 ||
        scattering_opacity.boxArray() != photon_boxes ||
        scattering_opacity.DistributionMap() != photon_distribution ||
        initial_rate_velocity.boxArray() != photon_boxes ||
        initial_rate_velocity.DistributionMap() != photon_distribution) {
        invalid_layout = 1;
    }
    amrex::Real maximum_path_length = std::numeric_limits<amrex::Real>::infinity();
    for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
    {
        amrex::Real const cell_size = geometry.CellSize(dimension);
        if (!amrex::Math::isfinite(cell_size) || !(cell_size > 0))
        {
            invalid_layout = 1;
        }
        maximum_path_length =
            std::min(maximum_path_length, 0.5_rt * cell_size);
    }
    amrex::Real const path_length = PhysConst::c * dt;
    if (!amrex::Math::isfinite(path_length) || !(path_length > 0) ||
        path_length > maximum_path_length)
    {
        invalid_layout = 1;
    }
    int const global_invalid_layout = GlobalFailureCode(invalid_layout);
    if (global_invalid_layout != 0)
    {
        SetPathRejection(report, global_invalid_layout);
        return false;
    }

    int local_failure = 0;
    auto const captured = CaptureParticles(photons);
    std::map<RoutingIdentity, NativePhotonPathCarry> carry_by_id;
    std::vector<RoutingIdentity> routing_identities;
    std::vector<std::uint64_t> draw_keys;
    if (!CaptureRoutingIdentities(photons, snapshot_policy, captured, routing_identities,
                                  &draw_keys)) {
        local_failure = 2;
    }
    long captured_particles = 0;
    for (auto const& tile : captured.tiles) { captured_particles += tile.count; }
    if ((resume_carries && initial_carries.size() !=
             static_cast<std::size_t>(captured_particles)) ||
        (!resume_carries && !initial_carries.empty()))
    {
        local_failure = 1;
    }
    for (auto const& carry : initial_carries)
    {
        if (!ValidCarry(carry, options.max_events_per_photon) ||
            !carry_by_id.emplace(CarryIdentity(carry), carry).second) {
            local_failure = 1;
        }
    }
    if (GlobalFailureCode(local_failure) != 0) {
        SetPathRejection(report, 2);
        return false;
    }
    if (snapshot_policy == NativePhotonPathSnapshotPolicy::KeyedProvisional) {
        // Explicit marked snapshots require globally unique tagged identities,
        // checked before any physical replay, including on empty MPI ranks.
        // These used to be constant-depth/draw carry probes whose returned
        // payload was discarded. Physical snapshot/carry/rate validation and
        // actual carry migration remain unchanged and separate below.
        if (!DistributedPhotonIdentitiesUnique(routing_identities))
        {
            SetPathRejection(report, 2);
            return false;
        }
    }
    auto const [opacity_fields, rate_fields] = CaptureFrozenGrids(
        scattering_opacity, initial_rate_velocity, geometry);
    for (amrex::MFIter iterator(scattering_opacity); iterator.isValid();
         ++iterator)
    {
        auto const& opacity_field = opacity_fields.at(iterator.index());
        auto const& rate_field = rate_fields.at(iterator.index());
        for (amrex::BoxIterator cell(iterator.validbox()); cell.ok(); ++cell)
        {
            amrex::Real const chi = opacity_field(*cell, 0);
            amrex::Real beta_squared = 0;
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                amrex::Real const beta =
                    rate_field(*cell, dimension) / PhysConst::c;
                if (!amrex::Math::isfinite(beta)) { local_failure = 1; }
                beta_squared += beta * beta;
            }
            if (!amrex::Math::isfinite(chi) || chi < 0 ||
                !amrex::Math::isfinite(beta_squared) || !(beta_squared < 1))
            {
                local_failure = 1;
            }
            amrex::Real const beta = std::sqrt(beta_squared);
            amrex::Real const gamma = 1 / std::sqrt(1 - beta_squared);
            amrex::Real const hazard =
                chi * PhysConst::c * dt *
                (options.moving_cross_sections ? gamma * (1 + beta) : amrex::Real(1));
            if (!amrex::Math::isfinite(hazard) || hazard < 0 ||
                hazard > options.max_optical_depth)
            {
                local_failure = 3;
            }
        }
    }

    auto plan = std::make_unique<Impl>();
    plan->geometry = geometry;
    plan->options = options;
    plan->dt = dt;
    plan->box_array = photon_boxes;
    plan->distribution_map = photon_distribution;
    plan->baseline = captured.tiles;
    plan->resume_carries = resume_carries;
    plan->snapshot_policy = snapshot_policy;
    plan->number_real_components = photons.NumRealComps();
    plan->number_int_components = photons.NumIntComps();
    plan->real_names = photons.GetRealSoANames();
    plan->int_names = photons.GetIntSoANames();
    plan->frozen_opacity.define(photon_boxes, photon_distribution, 1, 1);
    plan->frozen_rate_velocity.define(photon_boxes, photon_distribution, 3, 1);
    amrex::MultiFab::Copy(
        plan->frozen_opacity, scattering_opacity, 0, 0, 1, 0);
    amrex::MultiFab::Copy(
        plan->frozen_rate_velocity, initial_rate_velocity, 0, 0, 3, 0);

    long global_particle = 0;
    for (auto const& tile : captured.tiles)
    {
        long const count = tile.count;
        auto const value = [&](int component, long particle) {
            return tile.real_data[
                static_cast<std::size_t>(component) *
                    static_cast<std::size_t>(count) +
                static_cast<std::size_t>(particle)];
        };
        for (long particle = 0; particle < count; ++particle, ++global_particle)
        {
            if (local_failure != 0) { continue; }
            std::uint64_t const idcpu = tile.idcpu[
                static_cast<std::size_t>(particle)];
            NativePhotonPathCarry carry{};
            carry.idcpu = idcpu;
            auto const identity = routing_identities[static_cast<std::size_t>(global_particle)];
            carry.provisional_draw_key = identity.first == 1 ? identity.second : 0;
            if (plan->resume_carries)
            {
                auto const found = carry_by_id.find(identity);
                if (found == carry_by_id.end())
                {
                    local_failure = 1;
                    continue;
                }
                carry = found->second;
                if (carry.idcpu != idcpu || CarryIdentity(carry) != identity ||
                    !(carry.residual_optical_depth > 0) ||
                    !amrex::Math::isfinite(carry.residual_optical_depth) ||
                    carry.completed_events < 0 ||
                    carry.completed_events > options.max_events_per_photon ||
                    carry.next_draw != carry.completed_events + 1) {
                    local_failure = 1;
                    continue;
                }
            }
            if (!amrex::ConstParticleIDWrapper(idcpu).is_valid() ||
                photon_distribution[tile.key.first] !=
                    amrex::ParallelDescriptor::MyProc())
            {
                local_failure = 2;
                continue;
            }
            std::uint64_t const draw_key = draw_keys[static_cast<std::size_t>(global_particle)];
#if defined(WARPX_DIM_3D)
            amrex::ParticleReal const x = value(PIdx::x, particle);
            amrex::ParticleReal const y = value(PIdx::y, particle);
#elif defined(WARPX_DIM_XZ)
            amrex::ParticleReal const x = value(PIdx::x, particle);
            amrex::ParticleReal const y = 0;
#else
            amrex::ParticleReal const x = 0;
            amrex::ParticleReal const y = 0;
#endif
#if defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D)
            amrex::ParticleReal const z = value(PIdx::z, particle);
#else
            amrex::ParticleReal const z = 0;
#endif
            amrex::IntVect cell{};
            if (!NearestCell(geometry, x, y, z, cell) ||
                !photon_boxes[tile.key.first].contains(cell))
            {
                local_failure = 2;
                continue;
            }
            auto const represented = PacketConversionFourMomentumFromPhoton(
                value(PIdx::w, particle), value(PIdx::ux, particle),
                value(PIdx::uy, particle), value(PIdx::uz, particle));
            if (!represented.valid)
            {
                local_failure = 2;
                continue;
            }
            plan->idcpu.push_back(idcpu);
            plan->draw_key.push_back(draw_key);
            plan->initial_carries.push_back(carry);
            plan->weight.push_back(value(PIdx::w, particle));
            plan->initial_u.push_back(value(PIdx::ux, particle));
            plan->initial_u.push_back(value(PIdx::uy, particle));
            plan->initial_u.push_back(value(PIdx::uz, particle));
            plan->incoming.insert(
                plan->incoming.end(), represented.value.begin(),
                represented.value.end());
            plan->initial_position.push_back(0);
            plan->initial_position.push_back(0);
            plan->initial_position.push_back(0);
#if defined(WARPX_DIM_3D)
            plan->initial_position[3 * global_particle + 0] = x;
            plan->initial_position[3 * global_particle + 1] = y;
            plan->initial_position[3 * global_particle + 2] = z;
#elif defined(WARPX_DIM_XZ)
            plan->initial_position[3 * global_particle + 0] = x;
            plan->initial_position[3 * global_particle + 2] = z;
#elif defined(WARPX_DIM_1D_Z)
            plan->initial_position[3 * global_particle + 2] = z;
#endif
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                int const spatial_dimension =
                    SpatialIndexOfPhysicalAxis(dimension);
                plan->initial_cell.push_back(
                    spatial_dimension >= 0 ? cell[spatial_dimension] : 0);
            }
        }
    }

    int const global_failure = GlobalFailureCode(local_failure);
    if (global_failure != 0)
    {
        SetPathRejection(report, global_failure);
        return false;
    }
    plan->boundaries = boundaries;
    plan->prepared = true;
    m_impl = std::move(plan);
    report.valid = true;
    return true;
}

bool
NativePhotonPathReplay::Replay (
    amrex::MultiFab const& work_velocity,
    WarpXParticleContainer::Base& private_photons,
    amrex::MultiFab& material_impulse, amrex::MultiFab& photon_energy_debit,
    amrex::MultiFab& uncancelled_energy_scale,
    NativePhotonPathReport& report) const
{
    report = NativePhotonPathReport{};
    if (GlobalFailureCode(m_impl != nullptr && m_impl->prepared ? 0 : 1) != 0)
    {
        SetPathRejection(report, 1);
        return false;
    }

    int invalid_layout = 0;
    if (work_velocity.nComp() != 3 || material_impulse.nComp() != 3 ||
        photon_energy_debit.nComp() != 1 ||
        uncancelled_energy_scale.nComp() != 1 ||
        work_velocity.boxArray() != m_impl->box_array ||
        work_velocity.DistributionMap() != m_impl->distribution_map ||
        material_impulse.boxArray() != m_impl->box_array ||
        material_impulse.DistributionMap() != m_impl->distribution_map ||
        photon_energy_debit.boxArray() != m_impl->box_array ||
        photon_energy_debit.DistributionMap() != m_impl->distribution_map ||
        uncancelled_energy_scale.boxArray() != m_impl->box_array ||
        uncancelled_energy_scale.DistributionMap() !=
            m_impl->distribution_map ||
        private_photons.ParticleBoxArray(0) != m_impl->box_array ||
        private_photons.ParticleDistributionMap(0) != m_impl->distribution_map ||
        private_photons.NumRealComps() != m_impl->number_real_components ||
        private_photons.NumIntComps() != m_impl->number_int_components ||
        private_photons.GetRealSoANames() != m_impl->real_names ||
        private_photons.GetIntSoANames() != m_impl->int_names)
    {
        invalid_layout = 1;
    }
    int const global_invalid_layout = GlobalFailureCode(invalid_layout);
    if (global_invalid_layout != 0)
    {
        SetPathRejection(report, global_invalid_layout);
        return false;
    }

    if (GlobalFailureCode(ReplayStorageOverlaps(work_velocity, material_impulse,
            photon_energy_debit, uncancelled_energy_scale) ? 1 : 0) != 0)
    {
        SetPathRejection(report, 1);
        return false;
    }
    auto const candidate_capture = CaptureParticles(private_photons);
    bool const same_particles = SameCapturedTiles(candidate_capture.tiles, m_impl->baseline);
    if (GlobalFailureCode(same_particles ? 0 : 7) != 0)
    {
        SetPathRejection(report, 7);
        return false;
    }
    if (GlobalFailureCode(work_velocity.is_finite() ? 0 : 8) != 0)
    {
        SetPathRejection(report, 8);
        return false;
    }

    amrex::MultiFab staged_velocity(
        m_impl->box_array, m_impl->distribution_map, 3, 1);
    amrex::MultiFab::Copy(staged_velocity, work_velocity, 0, 0, 3, 0);
    staged_velocity.FillBoundary(m_impl->geometry.periodicity());
    m_impl->frozen_opacity.FillBoundary(m_impl->geometry.periodicity());
    m_impl->frozen_rate_velocity.FillBoundary(m_impl->geometry.periodicity());
    auto const staged_fields = CaptureHostFields(staged_velocity, 3);
    auto const opacity_fields = CaptureHostFields(m_impl->frozen_opacity, 1);
    auto const rate_fields = CaptureHostFields(
        m_impl->frozen_rate_velocity, 3);

    int invalid_velocity = 0;
    for (amrex::MFIter iterator(staged_velocity); iterator.isValid();
         ++iterator)
    {
        auto const& staged = staged_fields.at(iterator.index());
        for (amrex::BoxIterator cell(iterator.validbox()); cell.ok(); ++cell)
        {
            amrex::Real beta_squared = 0;
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                amrex::Real const beta =
                    staged(*cell, dimension) / PhysConst::c;
                if (!amrex::Math::isfinite(beta))
                {
                    invalid_velocity = 1;
                }
                beta_squared += beta * beta;
            }
            if (!amrex::Math::isfinite(beta_squared) ||
                !(beta_squared < 1))
            {
                invalid_velocity = 1;
            }
        }
    }
    if (GlobalFailureCode(invalid_velocity) != 0)
    {
        SetPathRejection(report, 8);
        return false;
    }

    amrex::MultiFab candidate_impulse(
        m_impl->box_array, m_impl->distribution_map, 3, 1);
    amrex::MultiFab candidate_debit(
        m_impl->box_array, m_impl->distribution_map, 1, 1);
    amrex::MultiFab candidate_scale(
        m_impl->box_array, m_impl->distribution_map, 1, 1);
    candidate_impulse.setVal(0);
    candidate_debit.setVal(0);
    candidate_scale.setVal(0);
    auto candidate_impulse_host = CaptureHostFields(candidate_impulse, 3);
    auto candidate_debit_host = CaptureHostFields(candidate_debit, 1);
    auto candidate_scale_host = CaptureHostFields(candidate_scale, 1);

    std::vector<amrex::ParticleReal> final_position(
        m_impl->idcpu.size() * 3, 0);
    std::vector<amrex::ParticleReal> final_u(m_impl->idcpu.size() * 3, 0);
    std::vector<NativePhotonPathCarry> final_carries(m_impl->idcpu.size());
    std::vector<int> escaped(m_impl->idcpu.size(), 0);
    amrex::GpuArray<amrex::Real, 4> incoming_inventory{};
    amrex::GpuArray<amrex::Real, 4> outgoing_inventory{};
    amrex::GpuArray<amrex::Real, 4> escaped_inventory{};
    amrex::GpuArray<amrex::Real, 3> deposited_impulse{};
    amrex::Real deposited_debit = 0;
    amrex::Long collision_events = 0;
    amrex::Long colliding_photons = 0;
    amrex::Long face_crossings = 0;
    amrex::Long wall_reflections = 0;
    amrex::Long escaped_photons = 0;
    amrex::Long escaped_colliding_photons = 0;
    amrex::GpuArray<amrex::Real, 3> wall_momentum{};
    int local_failure = 0;

    long global_particle = 0;
    for (auto const& tile : m_impl->baseline)
    {
        long const count = tile.count;
        for (long particle = 0; particle < count;
             ++particle, ++global_particle)
        {
            if (local_failure != 0) { continue; }
            auto const& staged = staged_fields.at(tile.key.first);
            auto const& opacity_field = opacity_fields.at(tile.key.first);
            auto const& rate_field = rate_fields.at(tile.key.first);
            auto& impulse_field = candidate_impulse_host.at(tile.key.first);
            auto& debit_field = candidate_debit_host.at(tile.key.first);
            auto& scale_field = candidate_scale_host.at(tile.key.first);

            FourVector current{};
            for (int component = 0; component < 4; ++component)
            {
                current[component] =
                    m_impl->incoming[4 * global_particle + component];
                incoming_inventory[component] += current[component];
            }
            amrex::Real position[3] = {};
            std::array<int, 3> cell{};
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                position[dimension] =
                    m_impl->initial_position[3 * global_particle + dimension];
                cell[dimension] =
                    m_impl->initial_cell[3 * global_particle + dimension];
            }
            amrex::Real remaining_time = m_impl->dt;
            std::uint64_t const draw_key =
                m_impl->draw_key[static_cast<std::size_t>(global_particle)];
            NativePhotonPathCarry carry =
                m_impl->initial_carries[static_cast<std::size_t>(global_particle)];
            amrex::Real remaining_depth = carry.residual_optical_depth;
            if (!(remaining_depth > 0))
            {
                remaining_depth = SampleOpticalDepth(
                    m_impl->options, draw_key, carry.next_draw);
                ++carry.next_draw;
            }
            if (!amrex::Math::isfinite(remaining_depth) ||
                !(remaining_depth > 0))
            {
                local_failure = 5;
                break;
            }
            int events_this_segment = 0;
            std::array<int, 3> last_event_cell = cell;
            FourVector last_unrounded{};
            // Reflection parity since the last collision (or interval start).
            // Return final representation roundoff to its physical event frame,
            // with the corresponding correction on the external wall ledger.
            int reflection_parity[3] = {1, 1, 1};
            long crossings = 0;
            long const maximum_crossings =
                3L * (static_cast<long>(m_impl->options.max_events_per_photon) + 1L);

            while (true)
            {
                amrex::GpuArray<amrex::Real, 3> direction{};
                amrex::Real direction_squared = 0;
                for (int dimension = 0; dimension < 3; ++dimension)
                {
                    direction[dimension] = current[dimension + 1] / current[0];
                    direction_squared += direction[dimension] * direction[dimension];
                }
                if (!amrex::Math::isfinite(direction_squared) ||
                    !(direction_squared > 0))
                {
                    local_failure = 9;
                    break;
                }
                amrex::IntVect local_cell{};
                for (int dimension = 0; dimension < AMREX_SPACEDIM;
                     ++dimension)
                {
                    int const physical_axis = PhysicalAxisOfSpatialIndex(
                        dimension);
                    local_cell[dimension] = cell[physical_axis];
                }
                if (!staged.contains(local_cell) ||
                    !opacity_field.contains(local_cell) ||
                    !rate_field.contains(local_cell))
                {
                    local_failure = 11;
                    break;
                }
                amrex::Real const chi = opacity_field(local_cell, 0);
                amrex::GpuArray<amrex::Real, 3> beta0{};
                for (int dimension = 0; dimension < 3; ++dimension)
                {
                    beta0[dimension] = rate_field(local_cell, dimension) /
                                       PhysConst::c;
                }
                amrex::Real const inverse_length = InverseRate(
                    m_impl->options, chi, beta0, direction);
                if (!amrex::Math::isfinite(inverse_length) ||
                    inverse_length < 0)
                {
                    local_failure = 9;
                    break;
                }

                amrex::Real collision_distance = std::numeric_limits<amrex::Real>::
                    infinity();
                if (inverse_length > 0)
                {
                    collision_distance = remaining_depth / inverse_length;
                }
                amrex::Real face_distance = std::numeric_limits<amrex::Real>::
                    infinity();
                int face_dimension = -1;
                int face_step = 0;
                for (int axis = 0; axis < 3; ++axis)
                {
                    int const dimension =
                        SpatialIndexOfPhysicalAxis(axis);
                    if (dimension < 0) { continue; }
                    amrex::Real const step = direction[axis];
                    if (step == 0) { continue; }
                    amrex::Real const dx = m_impl->geometry.CellSize(dimension);
                    amrex::Real const lower = m_impl->geometry.ProbLo(dimension);
                    amrex::Real const unwrapped = position[axis] - lower;
                    amrex::Real const local = unwrapped -
                        static_cast<amrex::Real>(
                            cell[axis] -
                            m_impl->geometry.Domain().smallEnd(dimension)) * dx;
                    amrex::Real signed_normal_distance = step > 0 ? dx-local : local;
                    // At a simultaneous corner hit the preceding axis update
                    // can overshoot this face by coordinate roundoff. Process
                    // that tied face at zero elapsed time, without consuming a
                    // new Poisson draw or charging a negative path length.
                    auto const face_roundoff = std::min(dx/8,
                        8 * std::numeric_limits<amrex::Real>::epsilon() *
                            std::max({dx, std::abs(position[axis]), std::abs(lower)}));
                    if (signed_normal_distance < 0 &&
                        signed_normal_distance >= -face_roundoff) {
                        signed_normal_distance = 0;
                    }
                    amrex::Real distance = 0;
                    int increment = 0;
                    if (step > 0)
                    {
                        distance = signed_normal_distance / step;
                        increment = 1;
                    }
                    else
                    {
                        distance = signed_normal_distance / (-step);
                        increment = -1;
                    }
                    if (!amrex::Math::isfinite(distance) || distance < 0)
                    {
                        local_failure = 12;
                        break;
                    }
                    if (distance < face_distance)
                    {
                        face_distance = distance;
                        face_dimension = axis;
                        face_step = increment;
                    }
                }
                if (local_failure != 0) { break; }
                amrex::Real const end_distance = remaining_time * PhysConst::c;
                if (!std::isfinite(end_distance) || end_distance < 0)
                {
                    local_failure = 9;
                    break;
                }

                if (collision_distance <= face_distance &&
                    collision_distance <= end_distance)
                {
                    if (carry.completed_events ==
                        m_impl->options.max_events_per_photon)
                    {
                        local_failure = 4;
                        break;
                    }
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        position[axis] += collision_distance *
                                          direction[axis];
                    }
                    remaining_time -= collision_distance / PhysConst::c;
                    remaining_depth = 0;
                    if (!std::isfinite(remaining_time) || remaining_time < 0)
                    {
                        local_failure = 9;
                        break;
                    }
                    amrex::GpuArray<amrex::Real, 3> beta{};
                    for (int dimension = 0; dimension < 3; ++dimension)
                    {
                        beta[dimension] = staged(local_cell, dimension) /
                                          PhysConst::c;
                    }
                    amrex::GpuArray<amrex::Real, 3> frame_direction{};
                    frame_direction = SampleEventDirection(
                        m_impl->options, draw_key, carry.next_draw - 1);
                    amrex::Real direction_norm = 0;
                    for (int dimension = 0; dimension < 3; ++dimension)
                    {
                        direction_norm += frame_direction[dimension] *
                            frame_direction[dimension];
                    }
                    if (!amrex::Math::isfinite(direction_norm) ||
                        !(direction_norm > 0))
                    {
                        local_failure = 5;
                        break;
                    }
                    FourVector outgoing{};
                    amrex::GpuArray<amrex::Real, 3> event_impulse{};
                    amrex::Real event_debit = 0;
                    if (!CoherentPhotonCollision(
                            current, frame_direction, beta, outgoing,
                            event_impulse, event_debit))
                    {
                        local_failure = 9;
                        break;
                    }
                    for (int dimension = 0; dimension < 3; ++dimension)
                    {
                        impulse_field.add(local_cell, dimension, event_impulse[dimension]);
                        deposited_impulse[dimension] += event_impulse[dimension];
                    }
                    debit_field.add(local_cell, 0, event_debit);
                    deposited_debit += event_debit;
                    scale_field.add(
                        local_cell, 0, std::abs(current[0]) + std::abs(outgoing[0]));
                    last_event_cell = cell;
                    last_unrounded = outgoing;
                    for (int d = 0; d < 3; ++d) { reflection_parity[d] = 1; }
                    current = outgoing;
                    ++events_this_segment;
                    ++carry.completed_events;
                    ++collision_events;
                    remaining_depth = SampleOpticalDepth(
                        m_impl->options, draw_key, carry.next_draw);
                    ++carry.next_draw;
                    if (!amrex::Math::isfinite(remaining_depth) ||
                        !(remaining_depth > 0))
                    {
                        local_failure = 5;
                        break;
                    }
                    continue;
                }
                if (face_distance <= end_distance)
                {
                    if (inverse_length > 0)
                    {
                        remaining_depth -= inverse_length * face_distance;
                    }
                    if (!amrex::Math::isfinite(remaining_depth) ||
                        remaining_depth < 0)
                    {
                        local_failure = 9;
                        break;
                    }
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        position[axis] += face_distance *
                                          direction[axis];
                    }
                    int const spatial_dimension =
                        SpatialIndexOfPhysicalAxis(face_dimension);
                    int const face_step_axis = face_step;
                    if (spatial_dimension < 0) { local_failure = 11; break; }
                    position[face_dimension] = m_impl->geometry.ProbLo(
                        spatial_dimension) +
                        static_cast<amrex::Real>(
                            cell[face_dimension] -
                            m_impl->geometry.Domain().smallEnd(
                                spatial_dimension) +
                            (face_step > 0 ? 1 : 0)) *
                        m_impl->geometry.CellSize(spatial_dimension);
                    remaining_time -= face_distance / PhysConst::c;
                    int const next_cell = cell[face_dimension] + face_step_axis;
                    bool const wall = !m_impl->geometry.isPeriodic(spatial_dimension) &&
                        (next_cell < m_impl->geometry.Domain().smallEnd(spatial_dimension) ||
                         next_cell > m_impl->geometry.Domain().bigEnd(spatial_dimension));
                    if (wall) {
                        // An open face owns a simultaneous open/reflecting
                        // corner: no artificial wall impulse precedes escape.
                        // Use only coordinate-representation roundoff, not a
                        // physical distance or time tolerance.
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            if (m_impl->geometry.isPeriodic(d)) { continue; }
                            int const axis = PhysicalAxisOfSpatialIndex(d);
                            bool const high = direction[axis] > 0;
                            bool const open = high ? m_impl->boundaries.escape_hi[d] != 0 :
                                                     m_impl->boundaries.escape_lo[d] != 0;
                            if (!open || direction[axis] == 0) { continue; }
                            auto const edge = high ? m_impl->geometry.ProbHi(d) :
                                                     m_impl->geometry.ProbLo(d);
                            auto const dx = m_impl->geometry.CellSize(d);
                            auto const tolerance = std::min(dx/8,
                                8 * std::numeric_limits<amrex::Real>::epsilon() *
                                    std::max({dx, std::abs(position[axis]), std::abs(edge)}));
                            if (std::abs(position[axis]-edge) <= tolerance) {
                                escaped[global_particle] = 1;
                            }
                        }
                        if (escaped[global_particle] != 0) {
                            ++face_crossings;
                            break;
                        }
                        wall_momentum[face_dimension] +=
                            2 * current[face_dimension + 1] / PhysConst::c;
                        current[face_dimension + 1] = -current[face_dimension + 1];
                        reflection_parity[face_dimension] *= -1;
                        ++wall_reflections;
                    } else {
                        cell[face_dimension] = next_cell;
                    }
                    ++crossings;
                    ++face_crossings;
                    if (crossings > maximum_crossings || face_dimension < 0 ||
                        remaining_time < 0)
                    {
                        local_failure = 11;
                        break;
                    }
                    continue;
                }

                for (int axis = 0; axis < 3; ++axis)
                {
                    position[axis] += end_distance * direction[axis];
                }
                if (inverse_length > 0)
                {
                    remaining_depth -= inverse_length * end_distance;
                }
                if (!std::isfinite(remaining_depth) || remaining_depth < 0)
                {
                    local_failure = 9;
                    break;
                }
                remaining_time = 0;
                break;
            }
            if (local_failure != 0) { break; }

            amrex::ParticleReal rounded_u[3] = {};
            amrex::Real const denominator = m_impl->weight[
                static_cast<std::size_t>(global_particle)] *
                PhysConst::m_e * PhysConst::c;
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                rounded_u[dimension] = static_cast<amrex::ParticleReal>(
                    events_this_segment == 0 ?
                        reflection_parity[dimension] *
                            m_impl->initial_u[3 * global_particle + dimension] :
                        current[dimension + 1] / denominator);
                final_u[3 * global_particle + dimension] = rounded_u[dimension];
            }
            auto const represented = PacketConversionFourMomentumFromPhoton(
                m_impl->weight[static_cast<std::size_t>(global_particle)],
                rounded_u[0], rounded_u[1], rounded_u[2]);
            if (!represented.valid)
            {
                local_failure = 9;
                break;
            }
            if (events_this_segment != 0)
            {
                auto& correction_impulse = candidate_impulse_host.at(tile.key.first);
                auto& correction_debit = candidate_debit_host.at(tile.key.first);
                auto& correction_scale = candidate_scale_host.at(tile.key.first);
                amrex::IntVect correction_cell{};
                for (int dimension = 0; dimension < AMREX_SPACEDIM;
                     ++dimension)
                {
                    int const physical_axis = PhysicalAxisOfSpatialIndex(
                        dimension);
                    correction_cell[dimension] = last_event_cell[physical_axis];
                }
                for (int dimension = 0; dimension < 3; ++dimension)
                {
                    amrex::Real const correction =
                        (last_unrounded[dimension + 1] -
                         reflection_parity[dimension] *
                             represented.value[dimension + 1]) / PhysConst::c;
                    wall_momentum[dimension] +=
                        (current[dimension + 1] - represented.value[dimension + 1]) /
                            PhysConst::c - correction;
                    correction_impulse.add(
                        correction_cell, dimension, correction);
                    deposited_impulse[dimension] += correction;
                }
                amrex::Real const energy_correction =
                    last_unrounded[0] - represented.value[0];
                correction_debit.add(correction_cell, 0, energy_correction);
                deposited_debit += energy_correction;
                correction_scale.add(
                    correction_cell, 0,
                    std::abs(last_unrounded[0]) +
                        std::abs(represented.value[0]));
                ++colliding_photons;
            }
            carry.residual_optical_depth = remaining_depth;
            final_carries[static_cast<std::size_t>(global_particle)] = carry;
            for (int component = 0; component < 4; ++component)
            {
                if (escaped[global_particle] != 0) {
                    escaped_inventory[component] += represented.value[component];
                } else {
                    outgoing_inventory[component] += represented.value[component];
                }
            }
            if (escaped[global_particle] != 0) {
                ++escaped_photons;
                if (carry.completed_events > 0) { ++escaped_colliding_photons; }
                continue;
            }
#if defined(WARPX_DIM_3D)
            if (!CanonicalPositionInCell(
                    m_impl->geometry, 0, position[0], cell[0],
                    final_position[3 * global_particle + 0]) ||
                !CanonicalPositionInCell(
                    m_impl->geometry, 1, position[1], cell[1],
                    final_position[3 * global_particle + 1]) ||
                !CanonicalPositionInCell(
                    m_impl->geometry, 2, position[2], cell[2],
                    final_position[3 * global_particle + 2]))
            {
                local_failure = 13;
                break;
            }
#elif defined(WARPX_DIM_XZ)
            if (!CanonicalPositionInCell(
                    m_impl->geometry, 0, position[0], cell[0],
                    final_position[3 * global_particle + 0]) ||
                !CanonicalPositionInCell(
                    m_impl->geometry, 1, position[2], cell[2],
                    final_position[3 * global_particle + 2]))
            {
                local_failure = 13;
                break;
            }
#elif defined(WARPX_DIM_1D_Z)
            if (!CanonicalPositionInCell(
                    m_impl->geometry, 0, position[2], cell[2],
                    final_position[3 * global_particle + 2]))
            {
                local_failure = 13;
                break;
            }
#endif
            constexpr int active_axes[3] = {
#if defined(WARPX_DIM_3D)
                0, 1, 2
#elif defined(WARPX_DIM_XZ)
                0, 2, -1
#elif defined(WARPX_DIM_1D_Z)
                2, -1, -1
#else
                -1, -1, -1
#endif
            };
            for (int axis : active_axes)
            {
                if (axis < 0) { continue; }
                amrex::Real const published = static_cast<amrex::Real>(
                    final_position[
                        3 * static_cast<std::size_t>(global_particle) +
                        static_cast<std::size_t>(axis)]);
                int const spatial_dimension =
                    SpatialIndexOfPhysicalAxis(axis);
                if (spatial_dimension < 0 ||
                    !amrex::Math::isfinite(published) ||
                    published < m_impl->geometry.ProbLo(spatial_dimension) ||
                    published >= m_impl->geometry.ProbHi(spatial_dimension))
                {
                    local_failure = 11;
                }
            }
            if (local_failure != 0) { break; }
        }
        if (local_failure != 0) { break; }
    }

    int const global_failure = GlobalFailureCode(local_failure);
    if (global_failure != 0)
    {
        SetPathRejection(report, global_failure);
        return false;
    }

    RestoreHostFields(candidate_impulse, candidate_impulse_host);
    RestoreHostFields(candidate_debit, candidate_debit_host);
    RestoreHostFields(candidate_scale, candidate_scale_host);
    candidate_impulse.SumBoundary(m_impl->geometry.periodicity());
    candidate_debit.SumBoundary(m_impl->geometry.periodicity());
    candidate_scale.SumBoundary(m_impl->geometry.periodicity());

    amrex::Real field_impulse[3] = {};
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        field_impulse[dimension] = candidate_impulse.sum(dimension);
    }
    amrex::Real const field_debit = candidate_debit.sum(0);
    amrex::Real identity_values[20] = {};
    amrex::Real reduced_values[16] = {};
    for (int component = 0; component < 4; ++component)
    {
        reduced_values[component] = incoming_inventory[component];
        reduced_values[4 + component] = outgoing_inventory[component];
    }
    // Inventories are local; MultiFab::sum above is already global. Reduce
    // inventories first so an empty rank cannot duplicate the global deposit.
    amrex::ParallelDescriptor::ReduceRealSum(reduced_values, 8);
    amrex::ParallelDescriptor::ReduceRealSum(wall_momentum.data(), 3);
    amrex::ParallelDescriptor::ReduceRealSum(escaped_inventory.data(), 4);
    reduced_values[8] = field_debit + escaped_inventory[0] -
        (reduced_values[0]-reduced_values[4]);
    reduced_values[9] = std::abs(field_debit)+reduced_values[0]+reduced_values[4] +
        escaped_inventory[0];
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        reduced_values[10+dimension] =
            PhysConst::c*(field_impulse[dimension] + wall_momentum[dimension]) +
            escaped_inventory[dimension+1] -
            (reduced_values[dimension+1]-reduced_values[dimension+5]);
        // Uncancelled photon energy bounds every Cartesian cp component.
        reduced_values[13+dimension] = std::abs(PhysConst::c*field_impulse[dimension])+
            std::abs(PhysConst::c*wall_momentum[dimension]) +
            reduced_values[0]+reduced_values[4]+escaped_inventory[0];
    }
    amrex::Real deposited_values[4] = {
        deposited_impulse[0], deposited_impulse[1],
        deposited_impulse[2], deposited_debit};
    amrex::ParallelDescriptor::ReduceRealSum(deposited_values, 4);
    for (int component = 0; component < 4; ++component)
    {
        identity_values[component] = reduced_values[component];
        identity_values[4 + component] = reduced_values[4 + component];
    }
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        identity_values[8 + dimension] = field_impulse[dimension];
        identity_values[14 + dimension] =
            reduced_values[10 + dimension];
        identity_values[17 + dimension] =
            reduced_values[13 + dimension];
    }
    identity_values[11] = field_debit;
    identity_values[12] = reduced_values[8];
    identity_values[13] = reduced_values[9];

    bool valid = true;
    amrex::Real const roundoff = 1024 *
        std::numeric_limits<amrex::Real>::epsilon();
    for (auto value : identity_values)
    {
        valid = valid && amrex::Math::isfinite(value);
    }
    valid = valid && std::abs(identity_values[12]) <=
        roundoff * identity_values[13];
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        valid = valid && std::abs(identity_values[14 + dimension]) <=
            roundoff * identity_values[17 + dimension];
    }
    valid = valid && candidate_impulse.is_finite() &&
        candidate_debit.is_finite() && candidate_scale.is_finite();
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        valid = valid && std::abs(PhysConst::c*(field_impulse[dimension] -
            deposited_values[dimension])) <=
            roundoff * identity_values[17 + dimension];
    }
    valid = valid && std::abs(field_debit - deposited_values[3]) <=
        roundoff * identity_values[13];
    if (GlobalFailureCode(valid ? 0 : 10) != 0)
    {
        SetPathRejection(report, 10);
        return false;
    }

    auto replay_rollback =
        std::make_unique<WarpXParticleContainer::Base>(private_photons.make_alike<>());
    replay_rollback->SetArena(private_photons.arena());
    replay_rollback->copyParticles(private_photons, true);

    std::map<std::pair<int, int>, long> tile_offsets;
    long next_offset = 0;
    for (auto const& tile : m_impl->baseline)
    {
        tile_offsets.emplace(tile.key, next_offset);
        next_offset += tile.count;
    }
    amrex::Gpu::DeviceVector<amrex::ParticleReal> device_position(
        final_position.size());
    amrex::Gpu::DeviceVector<amrex::ParticleReal> device_u(final_u.size());
    amrex::GpuArray<int, 3> census_components{-1, -1, -1};
    bool census_available = true;
    for (int d = 0; d < 3; ++d) {
        auto const name = std::string("radiation_diffusion_census_") +
            (d == 0 ? "ux0" : d == 1 ? "uy0" : "uz0");
        auto const found = std::find(m_impl->real_names.begin(), m_impl->real_names.end(), name);
        census_available = census_available && found != m_impl->real_names.end();
        census_components[d] = static_cast<int>(found - m_impl->real_names.begin());
    }
    std::vector<int> event_counts;
    event_counts.reserve(final_carries.size());
    for (auto const& carry : final_carries) { event_counts.push_back(carry.completed_events); }
    amrex::Gpu::DeviceVector<int> device_events(event_counts.size());
    amrex::Gpu::DeviceVector<int> device_escaped(escaped.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, escaped.begin(), escaped.end(),
                     device_escaped.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, event_counts.begin(), event_counts.end(),
                     device_events.begin());
    if (!final_position.empty())
    {
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, final_position.begin(),
            final_position.end(), device_position.begin());
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, final_u.begin(), final_u.end(),
            device_u.begin());
    }
    for (auto const& tile : m_impl->baseline)
    {
        long const count = tile.count;
        if (count == 0) { continue; }
        auto data = private_photons.GetParticles(0).at(tile.key)
                        .getParticleTileData();
        [[maybe_unused]] amrex::ParticleReal* const position = device_position.dataPtr();
        amrex::ParticleReal* const momentum = device_u.dataPtr();
        int const* const events = device_events.dataPtr();
        int const* const leaves = device_escaped.dataPtr();
        amrex::GpuArray<amrex::ParticleReal*, 3> census{};
        if (census_available) {
            for (int d = 0; d < 3; ++d) {
                census[d] = private_photons.GetParticles(0).at(tile.key)
                    .GetStructOfArrays().GetRealData(census_components[d]).dataPtr();
            }
        }
        long const offset = tile_offsets.at(tile.key);
        amrex::For(count, [=] AMREX_GPU_DEVICE (long particle)
        {
            long const source = offset + particle;
            if (leaves[source] != 0) {
                data.m_idcpu[particle] = amrex::ParticleIdCpus::Invalid;
                data.m_rdata[PIdx::w][particle] = 0;
                return;
            }
#if defined(WARPX_DIM_3D)
            data.m_rdata[PIdx::x][particle] = position[3 * source];
            data.m_rdata[PIdx::y][particle] = position[3 * source + 1];
            data.m_rdata[PIdx::z][particle] = position[3 * source + 2];
#elif defined(WARPX_DIM_XZ)
            data.m_rdata[PIdx::x][particle] = position[3 * source];
            data.m_rdata[PIdx::z][particle] = position[3 * source + 2];
#elif defined(WARPX_DIM_1D_Z)
            data.m_rdata[PIdx::z][particle] = position[3 * source + 2];
#endif
            data.m_rdata[PIdx::ux][particle] = momentum[3 * source];
            data.m_rdata[PIdx::uy][particle] = momentum[3 * source + 1];
            data.m_rdata[PIdx::uz][particle] = momentum[3 * source + 2];
            if (census_available && events[source] == 0) {
                // A pure wall reflection is not a scattering collision. Preserve
                // physical-event eligibility across the complete subsegment history.
                for (int d = 0; d < 3; ++d) { census[d][particle] = momentum[3 * source + d]; }
            }
        });
    }
    amrex::Gpu::streamSynchronize();
    private_photons.Redistribute();
    std::vector<NativePhotonPathCarry> survivor_carries;
    survivor_carries.reserve(final_carries.size());
    for (std::size_t n = 0; n < final_carries.size(); ++n) {
        if (escaped[n] == 0) { survivor_carries.push_back(final_carries[n]); }
    }
    std::vector<NativePhotonPathCarry> migrated_carries;
    if (!MigratePhotonCarries(private_photons, survivor_carries,
                              m_impl->options.max_events_per_photon, migrated_carries,
                              m_impl->snapshot_policy)) {
        private_photons.copyParticles(*replay_rollback, true);
        SetPathRejection(report, 7);
        return false;
    }
    final_carries = std::move(migrated_carries);

    // Migration can still reject. Do not publish fields before it succeeds.
    amrex::MultiFab::Copy(material_impulse, candidate_impulse, 0, 0, 3, 0);
    amrex::MultiFab::Copy(photon_energy_debit, candidate_debit, 0, 0, 1, 0);
    amrex::MultiFab::Copy(uncancelled_energy_scale, candidate_scale, 0, 0, 1, 0);

    amrex::ParallelDescriptor::ReduceLongSum(collision_events);
    amrex::ParallelDescriptor::ReduceLongSum(colliding_photons);
    amrex::ParallelDescriptor::ReduceLongSum(face_crossings);
    amrex::ParallelDescriptor::ReduceLongSum(wall_reflections);
    amrex::ParallelDescriptor::ReduceLongSum(escaped_photons);
    amrex::ParallelDescriptor::ReduceLongSum(escaped_colliding_photons);
    report.valid = true;
    report.photon_carries = std::move(final_carries);
    report.colliding_photons = colliding_photons;
    report.collision_events = collision_events;
    report.face_crossings = face_crossings;
    report.wall_reflections = wall_reflections;
    report.wall_momentum = wall_momentum;
    report.escaped_photons = escaped_photons;
    report.escaped_colliding_photons = escaped_colliding_photons;
    report.escaped_inventory = escaped_inventory;
    for (int component = 0; component < 4; ++component)
    {
        report.incoming_inventory[component] = identity_values[component];
        report.outgoing_inventory[component] =
            identity_values[4 + component];
    }
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        report.material_impulse[dimension] =
            identity_values[8 + dimension];
    }
    report.photon_energy_debit = identity_values[11];
    return true;
}

struct NativePhotonPathSubsegmentReplay::Impl
{
    WarpXParticleContainer* photon_species = nullptr;
    std::unique_ptr<amrex::MultiFab> scattering_opacity;
    std::unique_ptr<amrex::MultiFab> initial_rate_velocity;
    amrex::Geometry geometry{};
    NativePhotonCollisionOptions options{};
    NativePhotonPathBoundaries boundaries{};
    amrex::Real dt = 0;
    amrex::Real subsegment_dt = 0;
    int subsegments = 0;
    NativePhotonPathSnapshotPolicy snapshot_policy = NativePhotonPathSnapshotPolicy::NativeOnly;
    bool freeze_private_snapshot = false;
    std::vector<photon_replay_detail::CapturedTile> baseline;
    bool prepared = false;
};

NativePhotonPathSubsegmentReplay::NativePhotonPathSubsegmentReplay ()
    : m_impl(std::make_unique<Impl>()) {}

NativePhotonPathSubsegmentReplay::~NativePhotonPathSubsegmentReplay () = default;

bool
NativePhotonPathSubsegmentReplay::Prepare (
    WarpXParticleContainer& photon_species,
    amrex::MultiFab const& scattering_opacity,
    amrex::MultiFab const& initial_rate_velocity,
    amrex::Geometry const& geometry, amrex::Real dt,
    NativePhotonCollisionOptions const& options, int subsegment_limit,
    NativePhotonPathReport& report, NativePhotonPathBoundaries const& boundaries)
{
    return PrepareImpl(photon_species, photon_species, scattering_opacity, initial_rate_velocity,
                       geometry, dt, options, subsegment_limit, report, boundaries,
                       NativePhotonPathSnapshotPolicy::NativeOnly, false);
}

bool
NativePhotonPathSubsegmentReplay::Prepare (
    WarpXParticleContainer::Base const& private_snapshot, WarpXParticleContainer& photon_species,
    amrex::MultiFab const& scattering_opacity, amrex::MultiFab const& initial_rate_velocity,
    amrex::Geometry const& geometry, amrex::Real dt, NativePhotonCollisionOptions const& options,
    int subsegment_limit, NativePhotonPathReport& report,
    NativePhotonPathBoundaries const& boundaries, NativePhotonPathSnapshotPolicy snapshot_policy)
{
    return PrepareImpl(private_snapshot, photon_species, scattering_opacity, initial_rate_velocity,
                       geometry, dt, options, subsegment_limit, report, boundaries, snapshot_policy,
                       true);
}

bool
NativePhotonPathSubsegmentReplay::PrepareImpl (
    WarpXParticleContainer::Base const& private_snapshot, WarpXParticleContainer& photon_species,
    amrex::MultiFab const& scattering_opacity, amrex::MultiFab const& initial_rate_velocity,
    amrex::Geometry const& geometry, amrex::Real dt, NativePhotonCollisionOptions const& options,
    int subsegment_limit, NativePhotonPathReport& report,
    NativePhotonPathBoundaries const& boundaries, NativePhotonPathSnapshotPolicy snapshot_policy,
    bool freeze_private_snapshot)
{
    m_impl = std::make_unique<Impl>();
    report = NativePhotonPathReport{};
    int invalid = 0;
    bool const same_options = SameOptionsAcrossRanks(options, dt);
    bool const same_geometry = SameGeometryAcrossRanks(geometry);
    int minimum_limit = subsegment_limit;
    int maximum_limit = subsegment_limit;
    int minimum_freeze = freeze_private_snapshot ? 1 : 0;
    int maximum_freeze = minimum_freeze;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_limit);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_limit);
    amrex::ParallelDescriptor::ReduceIntMin(minimum_freeze);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_freeze);
    if (!same_options || !same_geometry || minimum_limit != maximum_limit ||
        minimum_freeze != maximum_freeze) {
        invalid = 1;
    }
#if !defined(WARPX_DIM_1D_Z) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_3D)
    invalid = 1;
#endif
    if constexpr (std::numeric_limits<amrex::Real>::digits < 53 ||
                  std::numeric_limits<amrex::ParticleReal>::digits < 53)
    {
        invalid = 1;
    }
    if (!photon_species.AmIA<PhysicalSpecies::photon>() ||
        photon_species.finestLevel() != 0 || geometry.Coord() != 0 ||
        !amrex::Math::isfinite(dt) || !(dt > 0) ||
        subsegment_limit <= 0 ||
        !amrex::Math::isfinite(options.max_optical_depth) ||
        !(options.max_optical_depth > 0))
    {
        invalid = 1;
    }
    amrex::Real dx = geometry.CellSize(0);
    for (int dimension = 1; dimension < AMREX_SPACEDIM; ++dimension)
    {
        amrex::Real const axis_dx = geometry.CellSize(dimension);
        if (!amrex::Math::isfinite(axis_dx) || !(axis_dx > 0)) { invalid = 1; }
        dx = std::min(dx, axis_dx);
    }
    amrex::Real const path_length = PhysConst::c * dt;
    if (!amrex::Math::isfinite(dx) || !(dx > 0) ||
        !amrex::Math::isfinite(path_length) || !(path_length > 0))
    {
        invalid = 1;
    }
    long subsegments = 0;
    if (invalid == 0)
    {
        long double const half_cell = 0.5L * static_cast<long double>(dx);
        long double const ratio =
            static_cast<long double>(path_length) / half_cell;
        if (!std::isfinite(ratio) || ratio > subsegment_limit)
        {
            invalid = 1;
        }
        else
        {
            subsegments = std::max(1L, static_cast<long>(std::ceil(ratio)));
            // Apply the represented half-cell check independently in each axis.
            auto const represented_safe = [&](long count)
            {
                amrex::Real const segment_length =
                    PhysConst::c * (dt / static_cast<amrex::Real>(count));
                for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
                {
                    if (segment_length > 0.5_rt * geometry.CellSize(dimension))
                    {
                        return false;
                    }
                }
                return true;
            };
            if (!represented_safe(subsegments)) { ++subsegments; }
            if (subsegments > subsegment_limit || !represented_safe(subsegments))
            {
                invalid = 1;
            }
        }
    }
    if (GlobalFailureCode(invalid) != 0)
    {
        report.reason = "invalid collective packet subsegment controls";
        return false;
    }
    // Validate layout/schema/options collectively before any grid copy or halo
    // exchange. This also keeps the original strict half-cell primitive gate.
    NativePhotonPathReplay preflight;
    if (!preflight.Prepare(private_snapshot, photon_species, scattering_opacity,
                           initial_rate_velocity, geometry,
                           dt / static_cast<amrex::Real>(subsegments), options, report, boundaries,
                           snapshot_policy)) {
        return false;
    }
    {
        auto const [opacity_fields, rate_fields] =
            CaptureFrozenGrids(scattering_opacity, initial_rate_velocity, geometry);
        for (amrex::MFIter iterator(scattering_opacity); iterator.isValid();
             ++iterator)
        {
            auto const& opacity_field = opacity_fields.at(iterator.index());
            auto const& rate_field = rate_fields.at(iterator.index());
            for (amrex::BoxIterator cell(iterator.validbox()); cell.ok(); ++cell)
            {
                amrex::Real const chi = opacity_field(*cell, 0);
                amrex::Real beta_squared = 0;
                for (int dimension = 0; dimension < 3; ++dimension)
                {
                    amrex::Real const beta =
                        rate_field(*cell, dimension) / PhysConst::c;
                    if (!amrex::Math::isfinite(beta)) { invalid = 1; }
                    beta_squared += beta * beta;
                }
                if (!amrex::Math::isfinite(chi) || chi < 0 ||
                    !amrex::Math::isfinite(beta_squared) || !(beta_squared < 1))
                {
                    invalid = 1;
                    continue;
                }
                amrex::Real const beta = std::sqrt(beta_squared);
                amrex::Real const gamma = 1 / std::sqrt(1 - beta_squared);
                amrex::Real const hazard =
                    chi * PhysConst::c * dt *
                    (options.moving_cross_sections ? gamma * (1 + beta) :
                                                     amrex::Real(1));
                if (!amrex::Math::isfinite(hazard) || hazard < 0 ||
                    hazard > options.max_optical_depth)
                {
                    invalid = 3;
                }
            }
        }
    }
    if (GlobalFailureCode(invalid) != 0)
    {
        report.valid = false;
        report.reason = "invalid outer packet subsegment optical-depth envelope";
        return false;
    }
    m_impl->photon_species = &photon_species;
    m_impl->scattering_opacity = std::make_unique<amrex::MultiFab>(
        scattering_opacity.boxArray(), scattering_opacity.DistributionMap(), 1, 0);
    m_impl->initial_rate_velocity = std::make_unique<amrex::MultiFab>(
        initial_rate_velocity.boxArray(), initial_rate_velocity.DistributionMap(), 3, 0);
    amrex::MultiFab::Copy(*m_impl->scattering_opacity, scattering_opacity, 0, 0, 1, 0);
    amrex::MultiFab::Copy(*m_impl->initial_rate_velocity, initial_rate_velocity, 0, 0, 3, 0);
    m_impl->geometry = geometry;
    m_impl->options = options;
    m_impl->boundaries = boundaries;
    m_impl->dt = dt;
    m_impl->subsegment_dt = dt / static_cast<amrex::Real>(subsegments);
    m_impl->subsegments = static_cast<int>(subsegments);
    m_impl->snapshot_policy = snapshot_policy;
    m_impl->freeze_private_snapshot = freeze_private_snapshot;
    if (freeze_private_snapshot) {
        m_impl->baseline = CaptureParticles(private_snapshot).tiles;
    }
    m_impl->prepared = true;
    report.valid = true;
    report.reason = "prepared fixed Cartesian 1D/XZ/3D packet subsegments";
    return true;
}

bool
NativePhotonPathSubsegmentReplay::Replay (
    amrex::MultiFab const& work_velocity,
    WarpXParticleContainer::Base& private_photons,
    amrex::MultiFab& material_impulse, amrex::MultiFab& photon_energy_debit,
    amrex::MultiFab& uncancelled_energy_scale,
    NativePhotonPathSubsegmentReport& report) const
{
    report = NativePhotonPathSubsegmentReport{};
    if (GlobalFailureCode(m_impl != nullptr && m_impl->prepared ? 0 : 1) != 0)
    {
        report.path.reason = "unprepared packet subsegment replay";
        return false;
    }
    int invalid_layout = 0;
    if (work_velocity.nComp() != 3 ||
        work_velocity.boxArray() != m_impl->scattering_opacity->boxArray() ||
        work_velocity.DistributionMap() != m_impl->scattering_opacity->DistributionMap() ||
        material_impulse.boxArray() != work_velocity.boxArray() ||
        material_impulse.DistributionMap() != work_velocity.DistributionMap() ||
        private_photons.ParticleBoxArray(0) != work_velocity.boxArray() ||
        private_photons.ParticleDistributionMap(0) != work_velocity.DistributionMap() ||
        !SnapshotSchemaMatches(private_photons, *m_impl->photon_species, m_impl->snapshot_policy) ||
        material_impulse.nComp() != 3 || photon_energy_debit.nComp() != 1 ||
        uncancelled_energy_scale.nComp() != 1 ||
        material_impulse.boxArray() != photon_energy_debit.boxArray() ||
        material_impulse.DistributionMap() != photon_energy_debit.DistributionMap() ||
        material_impulse.boxArray() != uncancelled_energy_scale.boxArray() ||
        material_impulse.DistributionMap() != uncancelled_energy_scale.DistributionMap()) {
        invalid_layout = 1;
    }
    if (GlobalFailureCode(invalid_layout) != 0)
    {
        report.path.reason = "invalid packet subsegment output layout";
        return false;
    }

    if (GlobalFailureCode(ReplayStorageOverlaps(work_velocity, material_impulse,
            photon_energy_debit, uncancelled_energy_scale) ? 1 : 0) != 0)
    {
        report.path.reason = "packet subsegment output storage aliases an input or output";
        return false;
    }
    if (m_impl->freeze_private_snapshot &&
        GlobalFailureCode(
            SameCapturedTiles(CaptureParticles(private_photons).tiles, m_impl->baseline) ? 0 : 7) !=
            0) {
        SetPathRejection(report.path, 7);
        return false;
    }
    // Clone the actual supplied schema, including its communicated private
    // marker. No transaction constructor or native ID allocation is involved.
    auto private_trial =
        std::make_unique<WarpXParticleContainer::Base>(private_photons.make_alike<>());
    private_trial->SetArena(private_photons.arena());
    private_trial->copyParticles(private_photons, true);
    amrex::MultiFab accumulated_impulse(
        material_impulse.boxArray(), material_impulse.DistributionMap(), 3, 0);
    amrex::MultiFab accumulated_debit(
        photon_energy_debit.boxArray(), photon_energy_debit.DistributionMap(),
        1, 0);
    amrex::MultiFab accumulated_scale(
        uncancelled_energy_scale.boxArray(),
        uncancelled_energy_scale.DistributionMap(), 1, 0);
    accumulated_impulse.setVal(0);
    accumulated_debit.setVal(0);
    accumulated_scale.setVal(0);

    std::vector<NativePhotonPathCarry> carries;
    for (int segment = 0; segment < m_impl->subsegments; ++segment)
    {
        NativePhotonPathReplay path;
        NativePhotonPathReport path_report;
        bool const prepared =
            segment == 0
                ? path.Prepare(*private_trial, *m_impl->photon_species, *m_impl->scattering_opacity,
                               *m_impl->initial_rate_velocity, m_impl->geometry,
                               m_impl->subsegment_dt, m_impl->options, path_report,
                               m_impl->boundaries, m_impl->snapshot_policy)
                : path.Prepare(*private_trial, *m_impl->photon_species, *m_impl->scattering_opacity,
                               *m_impl->initial_rate_velocity, m_impl->geometry,
                               m_impl->subsegment_dt, m_impl->options, carries, path_report,
                               m_impl->boundaries, m_impl->snapshot_policy);
        if (!prepared)
        {
            report.path = path_report;
            return false;
        }
        amrex::MultiFab segment_impulse(
            material_impulse.boxArray(), material_impulse.DistributionMap(), 3, 0);
        amrex::MultiFab segment_debit(
            photon_energy_debit.boxArray(), photon_energy_debit.DistributionMap(),
            1, 0);
        amrex::MultiFab segment_scale(
            uncancelled_energy_scale.boxArray(),
            uncancelled_energy_scale.DistributionMap(), 1, 0);
        if (!path.Replay(work_velocity, *private_trial, segment_impulse, segment_debit,
                         segment_scale, path_report)) {
            report.path = path_report;
            return false;
        }
        amrex::MultiFab::Add(
            accumulated_impulse, segment_impulse, 0, 0, 3, 0);
        amrex::MultiFab::Add(
            accumulated_debit, segment_debit, 0, 0, 1, 0);
        amrex::MultiFab::Add(
            accumulated_scale, segment_scale, 0, 0, 1, 0);
        if (segment == 0)
        {
            report.path.incoming_inventory = path_report.incoming_inventory;
        }
        report.path.outgoing_inventory = path_report.outgoing_inventory;
        report.path.collision_events += path_report.collision_events;
        report.path.face_crossings += path_report.face_crossings;
        report.path.wall_reflections += path_report.wall_reflections;
        report.path.escaped_photons += path_report.escaped_photons;
        report.path.escaped_colliding_photons += path_report.escaped_colliding_photons;
        for (int component = 0; component < 4; ++component) {
            report.path.escaped_inventory[component] += path_report.escaped_inventory[component];
        }
        for (int dimension = 0; dimension < 3; ++dimension)
        {
            report.path.material_impulse[dimension] +=
                path_report.material_impulse[dimension];
            report.path.wall_momentum[dimension] += path_report.wall_momentum[dimension];
        }
        report.path.photon_energy_debit += path_report.photon_energy_debit;
        carries = std::move(path_report.photon_carries);
    }

    if (!accumulated_impulse.is_finite() || !accumulated_debit.is_finite() ||
        !accumulated_scale.is_finite())
    {
        report.path.valid = false;
        report.path.reason = "nonfinite accumulated packet subsegment exchange";
        return false;
    }
    amrex::Long colliding_photons = 0;
    for (auto const& carry : carries)
    {
        if (carry.completed_events > 0) { ++colliding_photons; }
    }
    amrex::ParallelDescriptor::ReduceLongSum(colliding_photons);
    private_photons.copyParticles(*private_trial, true);
    private_photons.Redistribute();
    amrex::MultiFab::Copy(
        material_impulse, accumulated_impulse, 0, 0, 3, 0);
    amrex::MultiFab::Copy(
        photon_energy_debit, accumulated_debit, 0, 0, 1, 0);
    amrex::MultiFab::Copy(
        uncancelled_energy_scale, accumulated_scale, 0, 0, 1, 0);
    report.subsegments = m_impl->subsegments;
    report.path.valid = true;
    report.path.colliding_photons = colliding_photons + report.path.escaped_colliding_photons;
    report.path.photon_carries = carries;
    report.photon_carries = std::move(carries);
    return true;
}
}
