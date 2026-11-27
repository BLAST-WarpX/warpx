/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NATIVEPHOTONPATHREPLAY_H_
#define WARPX_RADIATION_NATIVEPHOTONPATHREPLAY_H_

#include "NativePhotonCollisionReplay.H"

#include <cstdint>
#include <vector>

namespace warpx::radiation
{
    /** Marked births are admitted only as caller-private, explicitly keyed
     * snapshots. Default native species and carry semantics are unchanged. */
    enum class NativePhotonPathSnapshotPolicy
    {
        NativeOnly,
        KeyedProvisional
    };
    /** Independent photon escape on nonperiodic Cartesian faces. Arrays use
     * mesh axes (z in1D, x/z inXZ), not three-vector component indices.
     * False retains specular reflection. Periodic axes must remain false.
     * Escaped four-momentum belongs to the external budget, never material. */
    struct NativePhotonPathBoundaries
    {
        amrex::GpuArray<int, AMREX_SPACEDIM> escape_lo{};
        amrex::GpuArray<int, AMREX_SPACEDIM> escape_hi{};
    };

    /** Persistent Poisson clock state between fixed path subsegments.
     *
     * residual_optical_depth is the unspent depth of the last keyed event
     * draw. next_draw is the first event index whose depth has not been drawn.
     * Draws use (seed, interval, logical draw identity, event index). The
     * optional persisted draw-key columns supply that identity for resolved
     * leakage births; absent or zero columns retain native-idcpu behavior.
     * Native idcpu routes ordinary photons. Explicitly admitted provisional
     * snapshots instead use a separately tagged nonzero logical birth key,
     * never their duplicate placeholder IDs. Segment boundaries neither redraw
     * nor rekey accepted physical events. */
    struct NativePhotonPathCarry
    {
        std::uint64_t idcpu = 0;
        int next_draw = 0;
        int completed_events = 0;
        amrex::Real residual_optical_depth = 0;
        /** Zero retains native idcpu routing. Nonzero selects a separately
         * tagged logical identity for a private provisional ray, avoiding
         * duplicate placeholder IDs. It cannot survive birth finalization. */
        std::uint64_t provisional_draw_key = 0;
    };

    struct NativePhotonPathReport : NativePhotonCollisionReport
    {
        amrex::Long face_crossings = 0;
        amrex::Long wall_reflections = 0;
        /** Signed momentum received by stationary specular walls [kg m/s].
         * External to the material collision impulse; wall energy work is zero. */
        amrex::GpuArray<amrex::Real, 3> wall_momentum{};
        /** RZ stationary axial K walls, low/high signed physical momentum
         * [kg m/s], including terminal representation roundoff. The operand
         * is the uncancelled sum of absolute individual wall transfers.
         * Already globally reduced; consumers must not MPI-sum again. */
        amrex::GpuArray<amrex::Real, 2> axial_wall_momentum_by_side{};
        amrex::Real axial_wall_momentum_operand = 0;
        /** Actual global records and mirror/terminal-correction additions;
         * count the arithmetic of uncancelled closed-wall accumulation. */
        amrex::Long replayed_photons = 0;
        amrex::Long axial_wall_transfers = 0;
        /** Local actual mirror-event logical draw keys (native fallback for
         * original rays), retained solely for private trajectory provenance. */
        std::vector<std::uint64_t> axial_wall_draw_keys;
        amrex::Long escaped_photons = 0;
        amrex::Long escaped_colliding_photons = 0;
        /** Represented outgoing [E,c*p] through open faces, in joules. */
        amrex::GpuArray<amrex::Real, 4> escaped_inventory{};
        std::vector<NativePhotonPathCarry> photon_carries;
    };

    /** Chronological, conservative physical packet transport and scattering.
     *
     * Prepare freezes the native photon snapshot, proper scattering opacity
     * and initial material rate velocity [m/s], plus seed and accepted interval
     * index. Replay uses the same keyed exponential optical-depth and comoving
     * angular variates, but recomputes the path and actual event cells with the
     * staged material WorkVelocity. Optical-depth remainder crosses faces;
     * events occur in chronological order, not before a full streaming step.
     * A changed secant frame can change event topology. The caller's bounded
     * material iteration must reject nonconvergence, never redraw to cure it.
     *
     * With moving_cross_sections=true, rates are c*chi0*gamma0*(1-beta0.n),
     * using frozen initial local rate_velocity and current lab ray direction.
     * With false, opacity is a prescribed laboratory inverse mean free path.
     * In both cases physical scatterings use the staged secant frame in the
     * actual event cell. Absorption and emission remain unsupported.
     *
     * In 3D the direction and face normals use physical x, y, and z. Face
     * distances are tested in that fixed axis order. An exact edge or corner
     * advances each tied cell face once at zero remaining distance, without a
     * new draw or optical-depth decrement; a collision tied with a face belongs
     * to the incident cell. The keyed event draw index and residual depth remain
     * attached to the tagged routing identity through owner migration. The
     * default native-only path retains idcpu routing, while optional persisted
     * draw identity controls only the random variates. Private provisional rays
     * require the explicit KeyedProvisional contract and unique nonzero keys.
     *
     * All photons remain directed native photons, including inside opaque
     * cells. A cell/region crossing produces no momentum kick, angular change,
     * scalar admission or birth. Scalar/vector conversion is not performed.
     * This is an all-packet/collar source, not accelerated diffusion.
     *
     * Scope: Cartesian level zero, DP native photons, c*dt<=min(dx)/2.
     * Periodic axes wrap; both faces of each nonperiodic axis are stationary
     * specular walls unless explicitly selected for photon escape. Escaping
     * photons are removed only from the private accepted candidate; their
     * represented four-momentum is reported separately from live survivors.
     * Reflection preserves the keyed Poisson clock, changes
     * only the normal ray component and never deposits wall force in material.
     * Path length bounds every coordinate displacement by half a cell, so
     * one filled ghost layer covers corner traversals from each initial box.
     * Recoil deposits on ghost cells are summed exactly once to owners.
     * Scalar fields have one component, velocities/impulses three; all use the
     * photon BoxArray and DistributionMapping. No caller ghost layer is needed.
     *
     * Replay requires a fresh unchanged private photon clone. Only positions
     * and proper momenta change. Returned work and impulse derive from actual
     * represented photon four-vectors and are owned by physical event cells;
     * photons without events deposit zero work/impulse/scale. All ranks,
     * including empty ranks, participate. Numerical rejection leaves supplied
     * photons and output fields unchanged. The outer interval alone commits
     * material and live photons. No global RNG or native IDs are consumed.
     */
    class NativePhotonPathReplay
    {
    public:
        NativePhotonPathReplay ();
        ~NativePhotonPathReplay ();
        NativePhotonPathReplay (NativePhotonPathReplay const&) = delete;
        NativePhotonPathReplay& operator= (NativePhotonPathReplay const&) = delete;

        [[nodiscard]] bool Prepare (
            WarpXParticleContainer& photons, amrex::MultiFab const& scattering_opacity,
            amrex::MultiFab const& initial_rate_velocity, amrex::Geometry const& geometry,
            amrex::Real dt, NativePhotonCollisionOptions const& options,
            NativePhotonPathReport& report,
            NativePhotonPathBoundaries const& boundaries = {});

        /** Same physics from an accepted private subsegment snapshot.
         * The live species owner supplies photon-type, level, layout and
         * runtime-schema provenance not available on Base. Positions/momenta
         * need not match the live owner, but layout/schema must; provisional
         * birth markers remain prohibited unless KeyedProvisional is explicit.
         * Then only the reserved extra marker and nonzero unique logical draw
         * keys are admitted; no native IDs are allocated. The owner is unchanged.
         */
        [[nodiscard]] bool Prepare (
            WarpXParticleContainer::Base const& private_snapshot,
            WarpXParticleContainer const& photon_species,
            amrex::MultiFab const& scattering_opacity,
            amrex::MultiFab const& initial_rate_velocity, amrex::Geometry const& geometry,
            amrex::Real dt, NativePhotonCollisionOptions const& options,
            NativePhotonPathReport& report,
            NativePhotonPathBoundaries const& boundaries = {},
            NativePhotonPathSnapshotPolicy snapshot_policy =
                NativePhotonPathSnapshotPolicy::NativeOnly);

        /** Resume the same keyed Poisson history from fixed-segment carries. */
        [[nodiscard]] bool Prepare (
            WarpXParticleContainer::Base const& private_snapshot,
            WarpXParticleContainer const& photon_species,
            amrex::MultiFab const& scattering_opacity,
            amrex::MultiFab const& initial_rate_velocity, amrex::Geometry const& geometry,
            amrex::Real dt, NativePhotonCollisionOptions const& options,
            std::vector<NativePhotonPathCarry> const& initial_carries,
            NativePhotonPathReport& report,
            NativePhotonPathBoundaries const& boundaries = {},
            NativePhotonPathSnapshotPolicy snapshot_policy =
                NativePhotonPathSnapshotPolicy::NativeOnly);

        [[nodiscard]] bool Replay (
            amrex::MultiFab const& work_velocity, WarpXParticleContainer::Base& private_photons,
            amrex::MultiFab& material_impulse, amrex::MultiFab& photon_energy_debit,
            amrex::MultiFab& uncancelled_energy_scale, NativePhotonPathReport& report) const;

    private:
        [[nodiscard]] bool PrepareImpl (
            WarpXParticleContainer::Base const& private_snapshot,
            WarpXParticleContainer const& photon_species,
            amrex::MultiFab const& scattering_opacity,
            amrex::MultiFab const& initial_rate_velocity,
            amrex::Geometry const& geometry, amrex::Real dt,
            NativePhotonCollisionOptions const& options,
            std::vector<NativePhotonPathCarry> const& initial_carries,
            bool resume_carries, NativePhotonPathReport& report,
            NativePhotonPathBoundaries const& boundaries,
            NativePhotonPathSnapshotPolicy snapshot_policy);

        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };

    struct NativePhotonPathSubsegmentReport
    {
        int subsegments = 0;
        NativePhotonPathReport path;
        std::vector<NativePhotonPathCarry> photon_carries;
    };

    /** Conservative fixed Cartesian 1D/XZ/3D packet path partition.
     *
     * The outer constitutive dt is retained by the caller. This helper divides
     * only the physical packet path into uniform segments satisfying
     * c*dt_sub<=dx/2. It does not subcycle fields or material work. Each replay
     * restarts from the supplied private interval-start clone and accumulates
     * no output until every segment succeeds. The existing half-cell guard is
     * applied independently to every segment. Face, edge, and corner hits
     * advance each crossed Cartesian cell face in deterministic axis order;
     * published coordinates are canonicalized to the replayed cell so particle
     * redistribution and the next segment agree on ownership. The helper stages
     * photon positions/momenta, residual optical depths, draw counters, impulse,
     * and work for the entire partition and publishes no caller state if a later
     * segment fails. */
    class NativePhotonPathSubsegmentReplay
    {
    public:
        NativePhotonPathSubsegmentReplay ();
        ~NativePhotonPathSubsegmentReplay ();
        NativePhotonPathSubsegmentReplay (
            NativePhotonPathSubsegmentReplay const&) = delete;
        NativePhotonPathSubsegmentReplay& operator= (
            NativePhotonPathSubsegmentReplay const&) = delete;

        [[nodiscard]] bool Prepare (
            WarpXParticleContainer& photon_species,
            amrex::MultiFab const& scattering_opacity,
            amrex::MultiFab const& initial_rate_velocity,
            amrex::Geometry const& geometry, amrex::Real dt,
            NativePhotonCollisionOptions const& options, int subsegment_limit,
            NativePhotonPathReport& report,
            NativePhotonPathBoundaries const& boundaries = {});

        [[nodiscard]] bool Replay (
            amrex::MultiFab const& work_velocity,
            WarpXParticleContainer::Base& private_photons,
            amrex::MultiFab& material_impulse, amrex::MultiFab& photon_energy_debit,
            amrex::MultiFab& uncancelled_energy_scale,
            NativePhotonPathSubsegmentReport& report) const;

        /** Private interval-start transport with explicitly keyed provisional
         * rays and tagged routing through owner migration. Replay must match
         * the complete supplied snapshot; no native IDs are consumed. */
        [[nodiscard]] bool Prepare (
            WarpXParticleContainer::Base const& private_snapshot,
            WarpXParticleContainer& photon_species,
            amrex::MultiFab const& scattering_opacity,
            amrex::MultiFab const& initial_rate_velocity,
            amrex::Geometry const& geometry, amrex::Real dt,
            NativePhotonCollisionOptions const& options, int subsegment_limit,
            NativePhotonPathReport& report,
            NativePhotonPathBoundaries const& boundaries = {},
            NativePhotonPathSnapshotPolicy snapshot_policy =
                NativePhotonPathSnapshotPolicy::NativeOnly);

    private:
        [[nodiscard]] bool PrepareImpl (
            WarpXParticleContainer::Base const& private_snapshot,
            WarpXParticleContainer& photon_species,
            amrex::MultiFab const& scattering_opacity,
            amrex::MultiFab const& initial_rate_velocity,
            amrex::Geometry const& geometry, amrex::Real dt,
            NativePhotonCollisionOptions const& options, int subsegment_limit,
            NativePhotonPathReport& report, NativePhotonPathBoundaries const& boundaries,
            NativePhotonPathSnapshotPolicy snapshot_policy, bool freeze_private_snapshot);
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };
}

#endif
