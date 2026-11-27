/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_PHOTONKEYAUDIT_H_
#define WARPX_RADIATION_PHOTONKEYAUDIT_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace warpx::radiation
{
// Domain0 is a complete native idcpu; domain1 is an explicit logical birth
// key. Equal numeric values in different domains remain distinct identities.
using PhotonAuditIdentity = std::pair<std::uint64_t, std::uint64_t>;

struct PhotonKeyAuditStats
{
    std::size_t rounds = 0;
    std::size_t peak_send_keys = 0;
    std::size_t peak_receive_keys = 0;
    std::size_t owned_keys = 0;
};

/** Exact collective uniqueness audit, without replicating global identities.
 * Hashing chooses a directory rank only; its set compares both full64-bit
 * words, never just the hash. Every rank participates, including empty ranks.
 * Two uint64 words/key are exchanged in bounded Alltoallv rounds. Default
 * exchange_key_limit=0 uses max(32768,NProcs) keys per send/receive buffer;
 * explicit limits must be collective, >=NProcs and <=INT_MAX/2. Owner storage
 * is O(keys assigned to this rank), not a full global copy on every rank.
 * The caller supplies local_valid for schema/decoding/native-ID validation.
 * False is collective and changes no caller key/particle/field/allocator/RNG
 * state. Numeric overflow is recoverable; allocation/process/MPI failures are
 * not. Stats are local diagnostic counters only, never an acceptance budget.
 */
[[nodiscard]] bool DistributedPhotonIdentitiesUnique (std::vector<PhotonAuditIdentity> const& local,
                                                      bool local_valid = true,
                                                      std::size_t exchange_key_limit = 0,
                                                      PhotonKeyAuditStats* stats = nullptr);

/** Stable metadata-only directory selector, also used by audit tests. */
[[nodiscard]] int PhotonKeyAuditOwner (PhotonAuditIdentity const& key, int ranks) noexcept;
} // namespace warpx::radiation

#endif
