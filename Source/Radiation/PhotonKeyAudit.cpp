/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "PhotonKeyAudit.H"

#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

#ifdef AMREX_USE_MPI
#include <mpi.h>
#endif

namespace warpx::radiation
{
namespace
{
std::uint64_t
DirectoryHash (PhotonAuditIdentity const& key) noexcept
{
    // Pure directory arithmetic: never a physical random draw or key rewrite.
    auto value = key.second ^ (key.first == 0 ? 0 : UINT64_C(0x17a68b9e0c731b2d));
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

struct IdentityHash
{
    std::size_t
    operator()(PhotonAuditIdentity const& key) const noexcept
    {
        return static_cast<std::size_t>(DirectoryHash(key));
    }
};

bool
Agree (bool valid)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid;
}
} // namespace

int
PhotonKeyAuditOwner (PhotonAuditIdentity const& key, int ranks) noexcept
{
    return ranks > 0 ? static_cast<int>(DirectoryHash(key) % static_cast<std::uint64_t>(ranks))
                     : -1;
}

bool
DistributedPhotonIdentitiesUnique (std::vector<PhotonAuditIdentity> const& local, bool local_valid,
                                   std::size_t exchange_key_limit, PhotonKeyAuditStats* stats)
{
    if (stats != nullptr)
    {
        *stats = {};
    }
    auto const ranks = amrex::ParallelDescriptor::NProcs();
    auto const limit = exchange_key_limit == 0
                           ? std::max(std::size_t(32768), static_cast<std::size_t>(ranks))
                           : exchange_key_limit;
    local_valid = local_valid && ranks > 0 && limit >= static_cast<std::size_t>(ranks) &&
                  limit <= static_cast<std::size_t>(std::numeric_limits<int>::max() / 2);
    int minimum_limit = local_valid ? static_cast<int>(limit) : 0;
    int maximum_limit = minimum_limit;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_limit);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_limit);
    for (auto const& key : local)
    {
        local_valid = local_valid && key.first <= 1 && key.second != 0;
    }
    if (!Agree(local_valid && minimum_limit == maximum_limit))
    {
        return false;
    }
    auto const per_peer = limit / static_cast<std::size_t>(ranks);
    std::vector<std::vector<PhotonAuditIdentity>> buckets(static_cast<std::size_t>(ranks));
    for (auto const& key : local)
    {
        buckets[static_cast<std::size_t>(PhotonKeyAuditOwner(key, ranks))].push_back(key);
    }
    std::vector<std::size_t> next(static_cast<std::size_t>(ranks), 0);
    std::vector<int> sent(static_cast<std::size_t>(ranks), 0), received(sent.size(), 0);
    std::vector<int> send_offset(sent.size(), 0), receive_offset(sent.size(), 0);
    std::unordered_set<PhotonAuditIdentity, IdentityHash> owned;
    std::vector<std::uint64_t> send_words, receive_words;
    for (;;)
    {
        bool remaining = false;
        for (int rank = 0; rank < ranks; ++rank)
        {
            auto const index = static_cast<std::size_t>(rank);
            remaining = remaining || next[index] < buckets[index].size();
        }
        amrex::ParallelDescriptor::ReduceBoolOr(remaining);
        if (!remaining)
        {
            if (stats != nullptr)
            {
                stats->owned_keys = owned.size();
            }
            return true;
        }
        send_words.clear();
        for (int rank = 0; rank < ranks; ++rank)
        {
            auto const index = static_cast<std::size_t>(rank);
            auto const count = std::min(per_peer, buckets[index].size() - next[index]);
            send_offset[index] = static_cast<int>(send_words.size());
            sent[index] = static_cast<int>(2 * count);
            for (std::size_t n = 0; n < count; ++n)
            {
                auto const& key = buckets[index][next[index]++];
                send_words.push_back(key.first);
                send_words.push_back(key.second);
            }
        }
#ifdef AMREX_USE_MPI
        auto status = MPI_Alltoall(sent.data(), 1, MPI_INT, received.data(), 1, MPI_INT,
                                   amrex::ParallelDescriptor::Communicator());
        if (!Agree(status == MPI_SUCCESS))
        {
            return false;
        }
#else
        received[0] = sent[0];
#endif
        std::size_t receive_count = 0;
        bool valid_counts = send_words.size() <= 2 * limit;
        for (int rank = 0; rank < ranks; ++rank)
        {
            auto const index = static_cast<std::size_t>(rank);
            auto const count = received[index];
            if (count < 0 || count % 2 != 0 || static_cast<std::size_t>(count) > 2 * per_peer ||
                static_cast<std::size_t>(count) > 2 * limit - receive_count)
            {
                valid_counts = false;
                continue;
            }
            receive_offset[index] = static_cast<int>(receive_count);
            receive_count += static_cast<std::size_t>(count);
        }
        valid_counts = valid_counts && receive_count <= 2 * limit &&
                       receive_count <= static_cast<std::size_t>(std::numeric_limits<int>::max());
        if (!Agree(valid_counts))
        {
            return false;
        }
        receive_words.resize(receive_count);
#ifdef AMREX_USE_MPI
        std::uint64_t dummy_send = 0, dummy_receive = 0;
        status = MPI_Alltoallv(send_words.empty() ? &dummy_send : send_words.data(), sent.data(),
                               send_offset.data(), MPI_UINT64_T,
                               receive_words.empty() ? &dummy_receive : receive_words.data(),
                               received.data(), receive_offset.data(), MPI_UINT64_T,
                               amrex::ParallelDescriptor::Communicator());
        if (!Agree(status == MPI_SUCCESS))
        {
            return false;
        }
#else
        receive_words = send_words;
#endif
        bool unique = true;
        for (std::size_t n = 0; n < receive_words.size(); n += 2)
        {
            PhotonAuditIdentity const key{receive_words[n], receive_words[n + 1]};
            unique = owned.insert(key).second && unique;
        }
        if (stats != nullptr)
        {
            ++stats->rounds;
            stats->peak_send_keys = std::max(stats->peak_send_keys, send_words.size() / 2);
            stats->peak_receive_keys = std::max(stats->peak_receive_keys, receive_words.size() / 2);
            stats->owned_keys = owned.size();
        }
        if (!Agree(unique))
        {
            return false;
        }
    }
}
} // namespace warpx::radiation
