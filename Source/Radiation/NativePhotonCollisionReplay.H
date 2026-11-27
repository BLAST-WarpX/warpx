/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NATIVEPHOTONCOLLISIONREPLAY_H_
#define WARPX_RADIATION_NATIVEPHOTONCOLLISIONREPLAY_H_

#include "Particles/WarpXParticleContainer.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <cstdint>
#include <memory>
#include <string>

namespace warpx::radiation
{
    enum class NativePhotonCollisionAngularLaw { Isotropic, FixedFrameDirection };

    struct NativePhotonCollisionOptions
    {
        std::uint64_t seed = 0;
        std::uint64_t interval = 0;
        int max_events_per_photon = 64;
        amrex::Real max_optical_depth = 16;
        NativePhotonCollisionAngularLaw angular_law = NativePhotonCollisionAngularLaw::Isotropic;
        /** Deterministic angular law for controlled beam/recoil benchmarks. */
        amrex::GpuArray<amrex::Real, 3> fixed_frame_direction{0, 0, -1};
        bool absorption = false;
        bool emission = false;
        /** True requires the explicit frozen rate_velocity Prepare overload. */
        bool moving_cross_sections = false;
    };

    struct NativePhotonCollisionReport
    {
        bool valid = false;
        std::string reason;
        amrex::Long colliding_photons = 0;
        amrex::Long collision_events = 0;
        amrex::GpuArray<amrex::Real, 4> incoming_inventory{}; // [E,c*p], J
        amrex::GpuArray<amrex::Real, 4> outgoing_inventory{}; // actual native photons
        amrex::GpuArray<amrex::Real, 3> material_impulse{};   // kg m/s
        amrex::Real photon_energy_debit = 0;                 // J
    };

    /** Frozen-position coherent-scattering source, followed by caller transport.
     * Prepare freezes interval-start native photons and counter-based Poisson
     * events/directions. The original Prepare uses rates chi_K*c in the lab;
     * it rejects moving_cross_sections=true. This lab-rate mode does not
     * reproduce general moving-bath drag/equilibrium. The overload with an
     * explicit rate_velocity supports the frozen initial material-frame law
     * documented below. Absorption and emission reject in both modes.
     * The caller checkpoints seed+interval and advances interval only on accept.
     * Neither AMReX RNG state nor the native particle-ID allocator is consumed.
     * Provisional photon births are rejected: call Prepare before interface
     * transport/births. Previous accepted births have ordinary identities.
     *
     * Each replay requires a fresh, unchanged private clone of the prepared
     * photons. Only proper momenta change; weights, positions and identities
     * remain fixed. The returned impulse/debit use actual represented outgoing
     * native photons, and are deposited only in their original K source cells.
     * There is no virtual D/K kick. The caller combines these fields with D
     * sources and stages one ParticleImpulseTransaction via the work iterator.
     *
     * Scope: level-zero Cartesian geometry, native double precision, nearest
     * cell source ownership. Event/optical-depth bounds reject rather than clip.
     * All methods are collective, including empty ranks. Numerical failure
     * leaves caller photons/output fields unchanged. Material and live photon
     * commits are exclusively owned by the outer accepted interval.
     */
    class NativePhotonCollisionReplay
    {
    public:
        NativePhotonCollisionReplay ();
        ~NativePhotonCollisionReplay ();
        NativePhotonCollisionReplay (NativePhotonCollisionReplay const&) = delete;
        NativePhotonCollisionReplay& operator= (NativePhotonCollisionReplay const&) = delete;

        [[nodiscard]] bool Prepare (
            WarpXParticleContainer& photons, amrex::iMultiFab const& diffusion_mask,
            amrex::MultiFab const& kinetic_scattering_opacity,
            amrex::Geometry const& geometry, amrex::Real dt,
            NativePhotonCollisionOptions const& options, NativePhotonCollisionReport& report);

        /** Frozen initial material-frame rate model; requires moving_cross_sections=true.
         * Opacity is now the proper inverse length chi0. With beta0=rate_velocity/c,
         * event waiting rates are c*chi0*gamma0*(1-beta0.n_lab). After each frozen
         * comoving direction draw, its boost by beta0 sets the next planning
         * direction/rate. Replay uses the staged secant frame with the same
         * prepared event count/directions, never new draws. Bound optical depth
         * by chi0*c*dt*gamma0*(1+|beta0|). This is a frozen-coefficient source
         * schedule within first-order collision/transport splitting; it is not
         * a time-evolving cross-section or general moving-medium transport model.
         */
        [[nodiscard]] bool Prepare (
            WarpXParticleContainer& photons, amrex::iMultiFab const& diffusion_mask,
            amrex::MultiFab const& kinetic_scattering_opacity,
            amrex::Geometry const& geometry, amrex::Real dt,
            amrex::MultiFab const& rate_velocity, NativePhotonCollisionOptions const& options,
            NativePhotonCollisionReport& report);

        [[nodiscard]] bool Replay (
            amrex::MultiFab const& work_velocity, WarpXParticleContainer::Base& private_photons,
            amrex::MultiFab& material_impulse, amrex::MultiFab& photon_energy_debit,
            amrex::MultiFab& uncancelled_energy_scale, NativePhotonCollisionReport& report) const;

    private:
        bool PrepareImpl (
            WarpXParticleContainer& photons, amrex::iMultiFab const& diffusion_mask,
            amrex::MultiFab const& kinetic_scattering_opacity,
            amrex::Geometry const& geometry, amrex::Real dt,
            amrex::MultiFab const* rate_velocity, NativePhotonCollisionOptions const& options,
            NativePhotonCollisionReport& report);
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };
}

#endif
