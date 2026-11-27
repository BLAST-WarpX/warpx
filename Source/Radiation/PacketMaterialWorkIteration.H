/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_PACKETMATERIALWORKITERATION_H_
#define WARPX_RADIATION_PACKETMATERIALWORKITERATION_H_

#include "ParticleImpulse.H"

#include <AMReX_MultiFab.H>

#include <functional>
#include <memory>
#include <string>

namespace warpx::radiation
{
    struct PacketMaterialWorkIterationOptions
    {
        int max_iterations = 64;
        amrex::Real damping = 0.5;
        amrex::Real impulse_relative_tolerance = 1.e-12;
        amrex::Real work_relative_tolerance = 1.e-12;
        ParticleImpulseAssignment assignment = ParticleImpulseAssignment::NearestCell;
    };

    enum class PacketMaterialWorkIterationReason {
        None, Success, InvalidOptions, InvalidLayout, StageFailed, ReplayRejected,
        InvalidReplayOutput, MaterialWorkMismatch, MaterialImpulseMismatch, NotConverged
    };

    [[nodiscard]] std::string PacketMaterialWorkIterationReasonName (
        PacketMaterialWorkIterationReason reason);

    /** Candidate fields are the values returned by the last replay invocation.
     * Residuals are maximum cell-wise differences divided by the corresponding
     * uncancelled operand scale. RZ radial/theta material impulse residuals
     * share the transverse operand scale because Cartesian/cylindrical
     * rotation mixes those components. A zero scale is converged only when both
     * operands are exactly zero; no absolute floor is added. */
    struct PacketMaterialWorkIterationReport
    {
        bool converged = false;
        int iterations = 0;
        PacketMaterialWorkIterationReason reason = PacketMaterialWorkIterationReason::None;
        amrex::Real max_impulse_residual = 0;
        amrex::Real max_radiation_work_residual = 0;
        amrex::Real max_material_work_residual = 0;
        amrex::Real max_material_impulse_residual = 0;
        /** Excess beyond the independent arithmetic allowance, normalized by
         * the original material scale. The raw residual above is preserved. */
        amrex::Real max_material_work_excess_residual = 0;
        amrex::Real requested_work = 0;
        amrex::Real actual_work = 0;
        amrex::Real energy_carry_change = 0;
        amrex::Real work_partition_residual = 0;
        amrex::MultiFab candidate_total_impulse;
        amrex::MultiFab candidate_radiation_debit;
        amrex::MultiFab candidate_uncancelled_energy_scale;

        PacketMaterialWorkIterationReport () = default;
        ~PacketMaterialWorkIterationReport () = default;
        PacketMaterialWorkIterationReport (PacketMaterialWorkIterationReport const&) = delete;
        PacketMaterialWorkIterationReport& operator= (
            PacketMaterialWorkIterationReport const&) = delete;
        PacketMaterialWorkIterationReport (PacketMaterialWorkIterationReport&&) = default;
        PacketMaterialWorkIterationReport& operator= (
            PacketMaterialWorkIterationReport&&) = default;
    };

    /** Replay contract. It must write every owned cell on every rank for every
     * invocation: new_total_impulse (three components), total_radiation_debit,
     * and an independently accumulated uncancelled_energy_scale. The coordinator
     * poisons all outputs with NaN before each call.
     *
     * Replay must regenerate photons and scalar states from unchanged snapshots
     * and frozen RNG draws. In particular, it may not copy RequestedWork into
     * total_radiation_debit and may not replace either side by v dot I. It also
     * must keep D and collar source cells disjoint. Callback collectives, if
     * any, must be entered consistently on all ranks even when this callback
     * will return false on some rank.
     *
     * A D solve may take RequestedWork restricted to D-owned cells as its work
     * source. Its debit must then be reconstructed from independently advanced
     * scalar energy and boundary/face exchanges. Collar debit is the difference
     * of frozen incoming and replayed outgoing photon energies. The scale is
     * the sum of the magnitudes of these uncancelled energy operands, in joules.
     * The callback owns candidate photon/scalar storage and must discard it
     * whenever this function returns false; no live state may be changed here.
     */
    using PacketMaterialWorkReplay = std::function<bool(
        ParticleImpulseTransaction const& staged, amrex::MultiFab& new_total_impulse,
        amrex::MultiFab& total_radiation_debit,
        amrex::MultiFab& uncancelled_energy_scale)>;

    /** Solve the packet/material secant fixed point without committing.
     *
     * Every iteration stages the current total impulse on the unchanged private
     * material with WorkVelocity enabled, then calls replay using that staged
     * state. On success accepted_transaction owns the exact uncommitted Stage
     * used by the accepted replay call; the report fields are that call's output.
     * A false return leaves material and accepted_transaction unchanged. This
     * implementation supports Cartesian nearest-cell assignment or, in RZ,
     * exclusively RZAngularConservative with reflecting radial/periodic axial
     * geometry. RZ theta impulse is delta Lz/rbar with annular mean radius
     * rbar, conjugate to the angular WorkVelocity; it is not sum(dp_theta).
     * Material impulse/carry checks use that same conjugate basis. Shaped
     * assignment and RZ nearest-cell assignment are rejected. The replay must
     * itself produce the torque-conjugate impulse and independent energy debit;
     * enabling this coordinator does not qualify a native RZ photon replay.
     * All ranks must call this function with the same registered carry path and
     * material species, as required by ParticleImpulseTransaction::Stage.
     * Options are checked for rank agreement. Fields must use the level-zero
     * native grid layout. Material must outlive the returned transaction.
     */
    [[nodiscard]] bool TryIteratePacketMaterialWork (
        ParticleImpulseMaterial& material, std::string const& path,
        amrex::MultiFab const& initial_total_impulse, PacketMaterialWorkReplay const& replay,
        PacketMaterialWorkIterationOptions const& options,
        std::unique_ptr<ParticleImpulseTransaction>& accepted_transaction,
        PacketMaterialWorkIterationReport& report);
}

#endif
