/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_PARTICLEIMPULSE_H_
#define WARPX_RADIATION_PARTICLEIMPULSE_H_

#include "Particles/WarpXParticleContainer.H"

#include <AMReX_MultiFab.H>

#include <memory>
#include <string>
#include <vector>

class MultiParticleContainer;
class WarpXParticleContainer;

namespace warpx::radiation
{
    enum class ParticleImpulseAssignment {
        NearestCell, LinearNodalCellAverage, NativeNodalCellAverage,
        /** Low-level 1D stationary PEC/reflecting adapter; not runtime boundary support. */
        ReflectingNodalCellAverage,
        /** Low-level RZ torque-conjugate cell assignment. Theta input is
         * requested axial torque divided by the annular mean radius, not a
         * request for the physical sum of theta momenta. */
        RZAngularConservative,
        /** Native shapes 1-3 nodal/corner-average RZ support. Theta is torque/rbar;
         * source work overlaps cells while actual kinetic work/carry retain
         * particle-center ownership. Currently periodic z and reflecting r. */
        RZAngularNodalCellAverage,
        /** Periodic Cartesian native shapes 1-3, with separately bounded
         * GLOBAL source/particle-center work and momentum accounting.
         * The legacy native shape assignment retains its original contract. */
        ConservativeNodalCellAverage,
        /** Native RZ shapes 1-3 with the actual reflecting-radial charge
         * fold, including the boundary node. Periodic z only. The legacy
         * unreflected species-deposit assignment above stays unchanged. */
        RZAngularReflectingRadialNodalCellAverage,
        /** Native RZ shapes 1-3 with reflecting radial and axial charge folds.
         * Only physical source cells receive force; wall reaction is external
         * to this fixed-position material impulse transaction. */
        RZAngularReflectingClosedNodalCellAverage
    };
    /** Private kinetic-material copy for a multi-stage radiation interval.
     * Substeps may change velocities and radiation carry only. Live particles
     * must remain untouched and retain their ordering until Commit. Destroying
     * this object discards all trial substeps; no rollback writes are needed.
     */
    class ParticleImpulseMaterial
    {
    public:
      struct SpeciesView
      {
          std::string name;
          amrex::Real rest_mass = 0;
          WarpXParticleContainer::Base const* particles = nullptr;
      };
        ParticleImpulseMaterial (MultiParticleContainer& particles,
                                 std::vector<std::string> const& species_names);
        ~ParticleImpulseMaterial ();
        ParticleImpulseMaterial (ParticleImpulseMaterial const&) = delete;
        ParticleImpulseMaterial& operator= (ParticleImpulseMaterial const&) = delete;
        ParticleImpulseMaterial (ParticleImpulseMaterial&&) = delete;
        ParticleImpulseMaterial& operator= (ParticleImpulseMaterial&&) = delete;
        bool Commit ();
        /** Collective read-only preflight against original live provenance
         * and frozen private layout/identity/positions, without publication. */
        [[nodiscard]] bool ValidateLive () const;
        /** Read-only actual caller-private native species, not a reconstructed
         * inventory or stage report. Returns null for a missing name, a
         * nonprivate owner or a committed owner. Empty MPI ranks still return
         * the valid private Base. This local accessor performs no collectives,
         * mutation, ownership transfer, RNG draws or native ID allocation.
         * The pointer is borrowed only while this material owner is alive and
         * uncommitted. Accepted private substage commits may change its stored
         * velocities/carries; callers must not retain a frozen-data assumption.
         */
        [[nodiscard]] WarpXParticleContainer::Base const*
        CandidateSpecies (std::string const& species_name) const;
        /** Local read-only views of every actual private species, including
         * its frozen native rest mass. Borrowed for this uncommitted owner's
         * lifetime; no reductions, deposition, reconstruction or allocation
         * of native particle IDs. Empty on a nonprivate/committed owner. */
        [[nodiscard]] std::vector<SpeciesView> CandidateSpecies () const;
        /** Recoverably deep-clone THIS actual uncommitted private material,
         * including earlier accepted private velocity/carry updates. Retain
         * the exact original live provenance/baseline, not a fresh live clone.
         * Species/schema/layout/IDs/weights/positions must still be frozen.
         * Collective rejection preserves output and all original owners.
         */
        [[nodiscard]] bool TryCloneCandidate (
            std::unique_ptr<ParticleImpulseMaterial>& output) const;

      private:
        friend class ParticleImpulseTransaction;
        struct Impl;
        ParticleImpulseMaterial (MultiParticleContainer& particles,
                                 std::vector<std::string> const& species_names, bool clone);
        explicit ParticleImpulseMaterial (std::unique_ptr<Impl> state);
        std::unique_ptr<Impl> m_impl;
    };

    /** Register communicated, checkpointed per-rest-mass numerical residuals
     * before particle initialization. Each radiation path/group owns its state.
     * This does not enable a radiation model or migrate old cell-owned state. */
    void RegisterParticleImpulseState (WarpXParticleContainer& species, std::string const& path);

    /** Pending impulse [kg m/s] and pending work [J]. By default momentum is
     * global Cartesian. The optional cylindrical projection sums local (r,theta,z)
     * components at current particle positions; it is not a conserved global vector.
     */
    amrex::GpuArray<amrex::Real, 4> ParticleImpulseInventory (
        MultiParticleContainer& particles, std::vector<std::string> const& species_names,
        std::string const& path, bool local_cylindrical = false);

    /** Stage a material impulse without mutating any live particle or reservoir.
     * The particle layout and velocities must remain unchanged between Stage and
     * Commit. Only Commit writes the accepted, previously evaluated candidates;
     * it does not repeat an atomic work reduction. Destroying the transaction
     * discards the candidate. Caller must validate radiation and native caloric
     * response together with these fields before committing any of them.
     */
    class ParticleImpulseTransaction
    {
    public:
        ParticleImpulseTransaction ();
        ~ParticleImpulseTransaction ();
        ParticleImpulseTransaction (ParticleImpulseTransaction const&) = delete;
        ParticleImpulseTransaction& operator= (ParticleImpulseTransaction const&) = delete;
        ParticleImpulseTransaction (ParticleImpulseTransaction&&) = delete;
        ParticleImpulseTransaction& operator= (ParticleImpulseTransaction&&) = delete;

        bool Stage (MultiParticleContainer& particles,
                    std::vector<std::string> const& species_names, std::string const& path,
                    amrex::MultiFab const& cell_impulse, bool need_work_velocity = false,
                    ParticleImpulseAssignment assignment = ParticleImpulseAssignment::NearestCell,
                    bool need_source_projection_bound = false);
        bool Stage (ParticleImpulseMaterial& material, std::string const& path,
                    amrex::MultiFab const& cell_impulse, bool need_work_velocity = false,
                    ParticleImpulseAssignment assignment = ParticleImpulseAssignment::NearestCell,
                    bool need_source_projection_bound = false);
        void Commit ();

        /** Source-origin work. For overlapping assignment its global sum
         * differs from the independent particle finite-work sum by the bounded
         * WorkPartitionResidual, not by a freely adjustable energy account. */
        [[nodiscard]] amrex::MultiFab const& RequestedWork () const;
        [[nodiscard]] amrex::MultiFab const& ActualWork () const;
        [[nodiscard]] amrex::MultiFab const& ActualImpulse () const;
        /** Actual impulse in the input's work-conjugate coordinates. In angular
         * RZ mode theta is sum(r*dp_theta)/mean_radius; ActualImpulse still
         * reports physical sum(dp_theta). Other assignments return the same field. */
        [[nodiscard]] amrex::MultiFab const& ConjugateImpulse () const;
        [[nodiscard]] amrex::MultiFab const& ConjugateMomentumCarryChange () const;
        /** Native RZ angular basis/readback arithmetic bound [kg m/s] by
         * cell/component. It is derived from every actual particle's old/new
         * Cartesian velocity, requested increment, carries, angular lever,
         * and deposition count. Empty for other assignments. This is not a
         * physical momentum repair or a source tolerance. */
        [[nodiscard]] amrex::MultiFab const& RZConjugateMomentumRoundoffBound () const;
        /** Global-accounting readback bounds for conservative native overlap.
         * Cartesian components are physical momenta; angular RZ theta is
         * torque/rbar with particle-center ownership. Empty on unsupported
         * legacy Cartesian assignments. Not a per-source-cell identity. */
        [[nodiscard]] amrex::MultiFab const& MomentumAccountingRoundoffBound () const;
        [[nodiscard]] amrex::MultiFab const& EnergyCarryChange () const;
        [[nodiscard]] amrex::MultiFab const& MomentumCarryChange () const;
        [[nodiscard]] amrex::MultiFab const& NumericalEnergyResidual () const;
        /** Source-cell work partition minus independent particle finite work,
         * deposited at particle centers. Bounded from the uncancelled operands
         * during staging; zero for nearest-cell assignment. ActualWork and
         * actual/carry impulses always retain particle-center ownership. */
        [[nodiscard]] amrex::MultiFab const& WorkPartitionResidual () const;
        /** Conservative cell-local roundoff bound [J] for nearest-cell and
         * RZAngularConservative assignment's
         * RequestedWork = ActualWork + EnergyCarryChange accounting identity.
         * It is accumulated independently from the per-particle requested and
         * actual work and old/new energy-carry operands. It is deliberately
         * zero (and unsupported) for Cartesian shaped assignment: zero is not a claim
         * that a shaped partition is exact. NumericalEnergyResidual and
         * WorkPartitionResidual remain diagnostics and never feed this bound.
         * When angular work velocity is requested, explicit uncancelled
         * Cartesian/cylindrical rotation and work-conjugate dot operands also
         * bound RequestedWork = dot(requested_conjugate_impulse, WorkVelocity).
         * A zero net transverse work is never assigned a zero conditioning
         * norm merely because its terms cancel. The RZ nodal assignment instead
         * bounds the GLOBAL source-origin/particle-center work identity; it
         * must never be interpreted as a per-cell ownership identity. */
        [[nodiscard]] amrex::MultiFab const& WorkAccountingRoundoffBound () const;
        /** SOURCE-owned arithmetic bound [J] for RequestedWork versus the dot
         * of the same requested impulse and native WorkVelocity. Available for
         * conservative overlapping assignment with both need_work_velocity and
         * need_source_projection_bound=true. Existing callers retain default-OFF.
         * Accumulates uncancelled per-particle source-stencil products and
         * contributor counts, not actual work/carry or a measured defect.
         * This is distinct from the GLOBAL center-owned work identity above.
         */
        [[nodiscard]] amrex::MultiFab const& RequestedWorkProjectionRoundoffBound () const;
        [[nodiscard]] amrex::MultiFab const& MaterialMass () const;
        /** Mass-weighted finite-kick secant velocity, in the impulse field's
         * coordinate basis. Available only when requested during Stage. The
         * scalar RequestedWork remains authoritative for cancellation-sensitive
         * work; dotting this rounded mean with impulse is not its replacement. */
        [[nodiscard]] amrex::MultiFab const& WorkVelocity () const;
        /** Read-only ACTUAL stored post-kick RZ rate frame, not the work
         * secant and not 2*secant-old. Deposits candidate u/gamma using the
         * original positions/weights and exact native mass/angular-inertia
         * normalization of a zero-requested angular rate sample. Requires a
         * valid uncommitted angular RZ stage, using its original assignment
         * support (nearest-cell or native nodal). Only a private MF is
         * allocated/written; no Stage, kick, carry, ID, source
         * or live/private material publication occurs. Current candidate
         * layout/positions must retain the ordinary Stage-to-Commit contract.
         * Collective failure, including a nonfinite endpoint angular rate,
         * returns false and preserves output and every owner/stage exactly.
         */
        [[nodiscard]] bool
        TryCandidateRateVelocity (amrex::MultiFab& output) const;
        /** Read-only ACTUAL input RZ rate frame, using the same cached native
         * mass/angular-inertia normalization as TryCandidateRateVelocity.
         * Reads input u/gamma directly and ignores pending carry and candidate
         * velocities. A zero-requested Stage may release carry into its
         * candidate; this query does not sample that candidate or rely on the
         * implementation/availability of the separate requested-work secant.
         * Requires a valid uncommitted angular RZ stage and
         * preserves output and all owners on collective failure. No publication
         * occurs. */
        [[nodiscard]] bool
        TryInitialRateVelocity (amrex::MultiFab& output) const;
        /** Maximum actual candidate ion beta squared, over every particle and
         * rank. This is a scope guard, not a bulk-velocity approximation. */
        [[nodiscard]] amrex::Real MaximumBetaSquared () const;

    private:
      [[nodiscard]] bool TryRateVelocity (amrex::MultiFab& output,
                                          bool candidate) const;
      struct Impl;
      std::unique_ptr<Impl> m_impl;
    };
}
#endif
