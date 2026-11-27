/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_NATIVE_RZ_HISTORY_PHOTON_WEIGHT_PROPOSAL_H_
#define WARPX_NATIVE_RZ_HISTORY_PHOTON_WEIGHT_PROPOSAL_H_

#include "NativeRZPhotonWeightProposal.H"
#include "NativeResolvedRZThermalHistory.H"

namespace warpx::radiation
{
/** Coupling-owned consumer seam; packet owner does NOT depend on history.
 * Bind ONLY real native history constraints E/cPz/physical-cLz (indices
 * 1/4/5), with every count-gamma-sensitive fixed reporting check preflighted
 * at its lower bound. Other guards are explicitly disabled, not replaced by
 * a fixture tolerance. All 13 features/native arithmetic witnesses remain.
 * False leaves options unchanged. No clocks/fields/owners are mutated.
 */
[[nodiscard]] bool MakeNativeRZHistoryPhotonWeightGuards (
    NativeRZHistoryAcceptanceWitness const&, NativeRZPhotonWeightProposalOptions&);

/** Private transaction consumer, including the default-off normal history
 * endpoint policy. Caller declares a fresh outer boundary and supplies
 * in-progress clocks explicitly; accepted-history clocks stay untouched.
 * The separate CPU/CUDA-host original-direction selector is collectively
 * checked before even a fixed-budget floor refusal; its default remains OFF.
 * Invalid/fixed-budget-inadmissible results never draw. This is not a cap.
 */
[[nodiscard]] NativeRZPhotonWeightProposalResult TryProposeNativeRZHistoryPhotonWeights (
    WarpXParticleContainer& live, NativeResolvedRZThermalHistoryResult const& history,
    amrex::Geometry const&, NativeRZPhotonWeightProposalOptions const&,
    std::vector<NativePhotonPathCarry> const& in_progress_clocks = {});
} // namespace warpx::radiation
#endif
