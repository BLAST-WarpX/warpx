/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativeRZHistoryPhotonWeightProposal.H"

#include <AMReX_ParallelDescriptor.H>

namespace warpx::radiation
{
namespace
{
bool All (bool value)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(value);
    return value;
}
}

bool MakeNativeRZHistoryPhotonWeightGuards (
    NativeRZHistoryAcceptanceWitness const& witness, NativeRZPhotonWeightProposalOptions& options)
{
    auto const lower = NativeRZHistoryAcceptanceLowerBound(witness);
    if (!lower.valid) { return false; }
    auto staged = options;
    for (auto& guard : staged.guards)
    { guard.required = false; guard.original_residual = 0; guard.unchanged_absolute_allowance = 0; }
    std::array<int, 3> const component{1, 4, 5};
    for (int d = 0; d < 3; ++d)
    {
        auto& guard = staged.guards[component[d]];
        guard.required = true;
        guard.original_residual = witness.residual[d];
        guard.unchanged_absolute_allowance = lower.endpoint[d];
    }
    options = staged;
    return true;
}

NativeRZPhotonWeightProposalResult TryProposeNativeRZHistoryPhotonWeights (
    WarpXParticleContainer& live, NativeResolvedRZThermalHistoryResult const& history,
    amrex::Geometry const& geometry, NativeRZPhotonWeightProposalOptions const& requested,
    std::vector<NativePhotonPathCarry> const& clocks)
{
    NativeRZPhotonWeightProposalResult unchanged;
    if (!ValidateNativeRZPhotonWeightOriginalDirectionControls(requested))
    {
        unchanged.reason = "invalid native original-direction certificate controls";
        return unchanged;
    }
    if (!ValidateNativeRZPhotonWeightDiagnosticControls(requested))
    { unchanged.reason = "invalid native diagnostic controls"; return unchanged; }
    unchanged.reason = "invalid native history provenance or immutable budget witness";
    if (!All(history.valid && history.photons && history.material &&
             history.completed_private_subintervals > 0 &&
             history.completed_private_subintervals == static_cast<int>(history.steps.size())))
    { return unchanged; }
    if (!history.photons->ValidateSnapshotFor(live) || !history.material->ValidateLive())
    { return unchanged; }
    auto options = requested;
    if (!All(MakeNativeRZHistoryPhotonWeightGuards(history.acceptance_witness, options)))
    {
        unchanged.valid = true;
        unchanged.photons = std::make_unique<PhotonPacketTransaction>(live, *history.photons);
        unchanged.reason = "native history fixed/count/operand lower-bound preflight: unchanged";
        if (requested.collect_diagnostics)
        {
            unchanged.diagnostics.collected = true;
            unchanged.diagnostics.history_lower_bound_refused = true;
        }
        return unchanged;
    }
    return TryProposeNativeRZPhotonWeights(live, *history.photons, geometry, options, clocks);
}
} // namespace warpx::radiation
