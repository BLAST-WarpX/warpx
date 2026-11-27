/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_RZNATIVEPHOTONPATHREPLAY_H_
#define WARPX_RADIATION_RZNATIVEPHOTONPATHREPLAY_H_

#include "NativePhotonPathReplay.H"

namespace warpx::radiation
{
    class NativeRZThermalAllDNoKEventWitness;
    class NativeRZThermalHistoryStepContext;
    class PhotonPacketTransaction;
    using RZNativePhotonPathReport = NativePhotonPathReport;

    /** Chronological native-photon transport and scattering in an RZ annulus.
     *
     * Prepare freezes the private native photon snapshot, scattering opacity,
     * initial local rate velocity and keyed event draws. With
     * moving_cross_sections=true, opacity is proper chi0 and the frozen RZ
     * material frame sets a position-dependent comoving collision rate along
     * each straight physical path. Physical beta at the cell outer radius is
     * bounded by 1e-3; this rate law is a packet-only source contract and does
     * not activate moving opacity in the coupled scalar diffusion closure.
     * Replay propagates actual Cartesian x=r*cos(theta), y=r*sin(theta), z
     * trajectories while publishing native RZ positions as (r,theta,z).  The
     * proper photon momentum remains Cartesian (ux,uy,uz).
     *
     * Radial faces are fixed, non-periodic specular walls at both rmin and
     * rmax; z is periodic or has stationary specular end walls. Axial wall
     * momentum is reported separately from material recoil. The keyed depth is carried
     * across interior faces and wall reflections.  Events use the staged local
     * WorkVelocity (vr, vtheta-conjugate, vz), converted through the annular
     * mean radius to a Cartesian material secant frame.  The keyed direction
     * is Cartesian for isotropic scattering; FixedFrameDirection is supplied
     * in the event's local (r,theta,z) frame and rotated to Cartesian.
     *
     * Material output is in local work-conjugate coordinates.  Its theta row
     * is (x*dpy-y*dpx)/rbar; global angular balance therefore uses
     * rbar-weighted theta.  Global energy, axial momentum and angular momentum
     * are validated.  Radial momentum is deliberately not asserted globally
     * because free cylindrical streaming changes its projection.  Event-free,
     * wall-only packets receive no material deposit.  Their wall invariants are
     * accepted only to roundoff; no fictitious compensating source is made.
     *
     * If all runtime census columns radiation_diffusion_census_ux0/uy0/uz0
     * exist, Replay overwrites exactly those three markers with the final
     * momentum for event-free photons.  Photons with physical events retain
     * their original census markers.  This prevents a wall-only reflection
     * from looking like a collision.
     *
     * Scope: double-precision, level-zero native RZ builds (Coord==0 or 1),
     * one native photon layout, rmin>=0, non-periodic r, closed or periodic z, and
     * c*dt<=min(dr,dz)/2.  One ghost ring covers the half-cell path.
     * The distinct typed all-D/empty-K History entries may omit ONLY this
     * photon duration comparison; they never permit a long photon path.
     * Axial periodic ghost indices are retained during drift and only the final z is
     * wrapped.  Recoil ghost sums are performed exactly once.  All collective,
     * layout, schema, baseline, alias, state and conservation checks complete
     * before any supplied particle or output field is mutated.  Non-RZ builds
     * safely reject through stub implementations.
     */
    class RZNativePhotonPathReplay
    {
    public:
        RZNativePhotonPathReplay ();
        ~RZNativePhotonPathReplay ();
        RZNativePhotonPathReplay (RZNativePhotonPathReplay const&) = delete;
        RZNativePhotonPathReplay& operator= (RZNativePhotonPathReplay const&) = delete;

        [[nodiscard]] bool Prepare (
            WarpXParticleContainer& photons, amrex::MultiFab const& scattering_opacity,
            amrex::MultiFab const& initial_rate_velocity, amrex::Geometry const& geometry,
            amrex::Real dt, NativePhotonCollisionOptions const& options,
            RZNativePhotonPathReport& report);

        /** Prepare an accepted private subsegment snapshot.
         * The live owner contributes photon type, level, layout and runtime
         * schema provenance. It is never modified. NativeOnly retains the old
         * marker prohibition. Explicit KeyedProvisional permits only the
         * communicated transaction marker and valid nonzero unique logical
         * birth keys; routing never uses duplicate placeholder native IDs.
         */
        [[nodiscard]] bool Prepare (
            WarpXParticleContainer::Base const& private_snapshot,
            WarpXParticleContainer const& photon_species,
            amrex::MultiFab const& scattering_opacity,
            amrex::MultiFab const& initial_rate_velocity, amrex::Geometry const& geometry,
            amrex::Real dt, NativePhotonCollisionOptions const& options,
            RZNativePhotonPathReport& report,
            NativePhotonPathSnapshotPolicy policy = NativePhotonPathSnapshotPolicy::NativeOnly);

        [[nodiscard]] bool Prepare (
            WarpXParticleContainer::Base const& private_snapshot,
            WarpXParticleContainer const& photon_species,
            amrex::MultiFab const& scattering_opacity,
            amrex::MultiFab const& initial_rate_velocity, amrex::Geometry const& geometry,
            amrex::Real dt, NativePhotonCollisionOptions const& options,
            std::vector<NativePhotonPathCarry> const& initial_carries,
            RZNativePhotonPathReport& report,
            NativePhotonPathSnapshotPolicy policy = NativePhotonPathSnapshotPolicy::NativeOnly);

        [[nodiscard]] bool Replay (
            amrex::MultiFab const& work_velocity,
            WarpXParticleContainer::Base& private_photons,
            amrex::MultiFab& material_impulse, amrex::MultiFab& photon_energy_debit,
            amrex::MultiFab& uncancelled_energy_scale,
            RZNativePhotonPathReport& report) const;

        /** History-only implicit D duration, NOT extended photon transport.
         * Requires the actual empty transaction/clocks and immutable engine
         * context/witness. Executes the same genuine path preparation graph;
         * removes ONLY the photon half-cell comparison. All field/frame and
         * scattering-depth guards remain. Old Base APIs cannot use this plan.
         * Neither borrowed authority may escape the synchronous History leaf. */
        [[nodiscard]] bool PrepareAllDNoPhoton (
            PhotonPacketTransaction const& private_snapshot,
            WarpXParticleContainer const& photon_species,
            amrex::MultiFab const& scattering_opacity,
            amrex::MultiFab const& initial_rate_velocity, amrex::Geometry const& geometry,
            amrex::Real dt, NativePhotonCollisionOptions const& options,
            std::vector<NativePhotonPathCarry> const& initial_carries,
            NativeRZThermalHistoryStepContext const& context,
            NativeRZThermalAllDNoKEventWitness const& witness,
            RZNativePhotonPathReport& report);

        [[nodiscard]] bool ReplayAllDNoPhoton (
            amrex::MultiFab const& work_velocity, PhotonPacketTransaction& private_photons,
            NativeRZThermalHistoryStepContext const& context,
            NativeRZThermalAllDNoKEventWitness const& witness,
            amrex::MultiFab& material_impulse, amrex::MultiFab& photon_energy_debit,
            amrex::MultiFab& uncancelled_energy_scale,
            RZNativePhotonPathReport& report) const;

    private:
        [[nodiscard]] bool PrepareImpl (
            WarpXParticleContainer::Base const& private_snapshot,
            WarpXParticleContainer const& photon_species,
            amrex::MultiFab const& scattering_opacity,
            amrex::MultiFab const& initial_rate_velocity, amrex::Geometry const& geometry,
            amrex::Real dt, NativePhotonCollisionOptions const& options,
            std::vector<NativePhotonPathCarry> const& initial_carries,
            bool resume_carries, RZNativePhotonPathReport& report,
            NativePhotonPathSnapshotPolicy policy, bool authenticated_empty_duration);
        [[nodiscard]] bool ReplayImpl (
            amrex::MultiFab const& work_velocity,
            WarpXParticleContainer::Base& private_photons,
            amrex::MultiFab& material_impulse, amrex::MultiFab& photon_energy_debit,
            amrex::MultiFab& uncancelled_energy_scale,
            RZNativePhotonPathReport& report, PhotonPacketTransaction* authenticated_empty) const;
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };

    /** Fixed-history RZ packet path partition with whole-interval publication. */
    class RZNativePhotonPathSubsegmentReplay
    {
    public:
        RZNativePhotonPathSubsegmentReplay ();
        ~RZNativePhotonPathSubsegmentReplay ();
        RZNativePhotonPathSubsegmentReplay (
            RZNativePhotonPathSubsegmentReplay const&) = delete;
        RZNativePhotonPathSubsegmentReplay& operator= (
            RZNativePhotonPathSubsegmentReplay const&) = delete;

        [[nodiscard]] bool Prepare (
            WarpXParticleContainer& photon_species,
            amrex::MultiFab const& scattering_opacity,
            amrex::MultiFab const& initial_rate_velocity,
            amrex::Geometry const& geometry, amrex::Real dt,
            NativePhotonCollisionOptions const& options, int subsegment_limit,
            RZNativePhotonPathReport& report);

        /** Freeze an explicit caller-private snapshot, including keyed births.
         * Replay must begin from its exact bytes. Every subsegment preserves
         * the marker and logical routing tags; no native IDs are allocated.
         * Carries with provisional tags are invalid after final birth commit.
         */
        [[nodiscard]] bool Prepare (
            WarpXParticleContainer::Base const& private_snapshot,
            WarpXParticleContainer& photon_species,
            amrex::MultiFab const& scattering_opacity,
            amrex::MultiFab const& initial_rate_velocity,
            amrex::Geometry const& geometry, amrex::Real dt,
            NativePhotonCollisionOptions const& options, int subsegment_limit,
            RZNativePhotonPathReport& report,
            NativePhotonPathSnapshotPolicy policy = NativePhotonPathSnapshotPolicy::NativeOnly);

        [[nodiscard]] bool Replay (
            amrex::MultiFab const& work_velocity,
            WarpXParticleContainer::Base& private_photons,
            amrex::MultiFab& material_impulse, amrex::MultiFab& photon_energy_debit,
            amrex::MultiFab& uncancelled_energy_scale,
            NativePhotonPathSubsegmentReport& report) const;

    private:
        [[nodiscard]] bool PrepareImpl (
            WarpXParticleContainer::Base const& private_snapshot,
            WarpXParticleContainer& photon_species,
            amrex::MultiFab const& scattering_opacity,
            amrex::MultiFab const& initial_rate_velocity,
            amrex::Geometry const& geometry, amrex::Real dt,
            NativePhotonCollisionOptions const& options, int subsegment_limit,
            RZNativePhotonPathReport& report, NativePhotonPathSnapshotPolicy policy,
            bool freeze_private_snapshot);
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };
}

#endif
