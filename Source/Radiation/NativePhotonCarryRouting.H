/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NATIVEPHOTONCARRYROUTING_H_
#define WARPX_RADIATION_NATIVEPHOTONCARRYROUTING_H_

#include "NativePhotonPathReplay.H"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace warpx::radiation::photon_replay_detail
{
    /** Validate fixed-width record counts and every int count/displacement.
     * Pure helper so overflow can be checked without allocating the payload. */
    bool ValidPackedWordCounts (
        std::vector<std::size_t> const& word_counts, int record_width);

    /** Route private carry state to actual post-Redistribute particle owners.
     * All ranks must call, including empty ranks. A false return leaves migrated
     * unchanged on every rank. IDs must be in each owner's particle traversal
     * order. This is an internal test seam, not a live species attribute. */
    bool RoutePhotonCarries (
        std::vector<NativePhotonPathCarry> const& local_carries,
        std::vector<std::uint64_t> const& local_post_redistribute_ids,
        int event_cap, std::vector<NativePhotonPathCarry>& migrated);

    /** Explicit marked-snapshot routing contract shared by Cartesian and RZ
     * paths. Read-only, local schema/marker validation; output vectors remain
     * unchanged on rejection. Native records have provisional key zero;
     * marked records have placeholder native ID 1 and a nonzero logical key.
     * Draw keys may also be persistent on native records, without changing
     * their native-ID routing identity. Callers perform collective rejection.
     */
    bool CapturePhotonRoutingKeys (
        WarpXParticleContainer::Base const& snapshot, WarpXParticleContainer const& owner,
        NativePhotonPathSnapshotPolicy policy, std::vector<std::uint64_t>& native_ids,
        std::vector<std::uint64_t>& provisional_keys, std::vector<std::uint64_t>& draw_keys);

    /** Additive opt-in tagged route. Parallel native_ids/provisional_keys are
     * actual post-Redistribute traversal order. Tag zero routes native idcpu;
     * tag one routes explicit nonzero provisional keys, never placeholder IDs.
     * Global duplicate identities, missing carries and mismatched actual IDs
     * reject collectively and preserve migrated. No native IDs are allocated.
     */
    bool RouteKeyedPhotonCarries (
        std::vector<NativePhotonPathCarry> const& local_carries,
        std::vector<std::uint64_t> const& local_post_redistribute_ids,
        std::vector<std::uint64_t> const& local_provisional_keys,
        int event_cap, std::vector<NativePhotonPathCarry>& migrated);

    /** History-only endpoint reconciliation after one accepted physical
     * source. Route complete prior path clocks through actual survivors;
     * discard absorbed clocks and initialize ONLY marked births with an
     * actual staging receipt. Each receipt must be new and present once;
     * unknown old-key replacement rejects. Missing native clocks reject,
     * except at an explicitly
     * declared fresh outer start. Uses the SAME keyed exponential event0
     * as path replay, with unchanged outer seed/interval and event counters.
     * Duplicate/malformed state rejects collectively, preserving output.
     * Optional actual_local_birth_keys returns ONLY newly initialized marked
     * births in this endpoint owner's traversal order, from the actual
     * directory receipt decision, not an endpoint key-set difference. Requires
     * non-fresh reconciliation with actual_birth_keys; pointer presence must
     * agree across ranks. Both outputs remain unchanged on rejection. Default
     * null retains the original five-word reply and clock/draw arithmetic.
     * Carries remain private and require birth-ID remapping before any future
     * live checkpoint consumer; this is not a serialized clock schema.
     */
    bool ReconcileKeyedPhotonHistoryCarries (
        std::vector<NativePhotonPathCarry> const& previous_path_clocks,
        WarpXParticleContainer::Base const* preceding_snapshot,
        WarpXParticleContainer::Base const& accepted_endpoint,
        WarpXParticleContainer const& live_owner,
        NativePhotonCollisionOptions const& outer_clock_options,
        bool fresh_outer_start, std::vector<NativePhotonPathCarry>& reconciled,
        std::vector<std::uint64_t> const* actual_birth_keys = nullptr,
        std::vector<std::uint64_t>* actual_local_birth_keys = nullptr);
}

#endif
