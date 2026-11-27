/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_MOVING_ASYMPTOTIC_DIFFUSION_FIELD_H_
#define WARPX_RADIATION_MOVING_ASYMPTOTIC_DIFFUSION_FIELD_H_

#include "AsymptoticDiffusionField.H"

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <memory>

namespace warpx::radiation
{

/** Failure reasons for one bounded moving-material scalar diffusion step. */
enum class MovingAsymptoticDiffusionFieldFailure
{
    None,
    InvalidState,
    UnsupportedGeometry,
    EnthalpyCfl,
    AdvectedState,
    DiffusionFailure,
    LabEnergyResidual,
    NonfiniteDiagnostic
};

/** Result of one accepted moving-material step, or no fields on rejection.
 *
 * `diffusion` retains the native asymptotic-diffusion result except that, with
 * kinetic boundaries enabled, face_leaked_energy also includes outward enthalpy.
 * Its
 * diffusive current, pressure, and Densmore impulse contain only F0 terms;
 * enthalpy is never mixed into those outputs or into the material force.
 * The moving fields are reported separately. `lab_radiation_inertia` is a
 * post-step diagnostic and must not be fed back as causal radiation state.
 */
struct MovingAsymptoticDiffusionFieldResult
{
    MovingAsymptoticDiffusionFieldResult () = default;
    MovingAsymptoticDiffusionFieldResult (
        MovingAsymptoticDiffusionFieldResult&&) = default;
    MovingAsymptoticDiffusionFieldResult& operator= (
        MovingAsymptoticDiffusionFieldResult&&) = default;
    MovingAsymptoticDiffusionFieldResult (
        MovingAsymptoticDiffusionFieldResult const&) = delete;
    MovingAsymptoticDiffusionFieldResult& operator= (
        MovingAsymptoticDiffusionFieldResult const&) = delete;

    bool valid = false;
    MovingAsymptoticDiffusionFieldFailure failure =
        MovingAsymptoticDiffusionFieldFailure::None;
    AsymptoticDiffusionFieldFailure diffusion_failure =
        AsymptoticDiffusionFieldFailure::None;
    int linear_iterations = 0;

    AsymptoticDiffusionFieldResult diffusion;

    /** Explicit old-energy enthalpy current (4/3)e_upwind*v_face [W m^-2]. */
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>
        face_enthalpy_current;
    /** Total laboratory current F0 + enthalpy current [W m^-2]. */
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>
        face_lab_energy_current;
    /** Independent DeltaE + dt div(F_lab) + RequestedWork [J]. */
    amrex::MultiFab lab_energy_balance_residual;
    /** V*(F0_cell + 4 e_new v_cell/3)/c [J]; diagnostic only. In RZ the
     * theta entry is c*Lz/mean_radius with the reconstruction documented below. */
    amrex::MultiFab lab_radiation_inertia;
    /** Signed momentum received by all stationary radiation walls [kg m/s]. */
    amrex::GpuArray<amrex::Real, 3> radiation_wall_momentum{};

    amrex::Real global_lab_energy_balance_residual = 0;
    amrex::Real max_lab_energy_balance_residual = 0;

    explicit operator bool () const noexcept { return valid; }
};

/** Advance one scalar field through frozen material motion.
 *
 * The leading nonrelativistic thick-diffusion model uses lab energy e=E/V,
 * P=e/3, and F0=-c grad(e)/(3 chi). One step first applies the conservative
 * explicit enthalpy flux (4/3)e_upwind*v_face, using an arithmetic shared-face
 * velocity, and then calls the native implicit F0 diffusion solver with the
 * caller's authoritative RequestedWork exactly once. The combined discrete
 * ledger is DeltaE + dt div_A(F0 + (4/3)e_upwind*v) + RequestedWork = 0.
 *
 * `work_velocity` has three physical (x,y,z) components in m/s, matching the
 * staged ParticleImpulseTransaction::WorkVelocity layout. It is a cellwise
 * frozen secant, not an exact Lorentz boost or a moving-interface law. Every
 * component must be finite and |v|/c <= 0.01. This leading thick-limit
 * approximation identifies lab and comoving energy density: it omits the
 * 2 v dot F0/c^2 density correction as well as O(beta^2 e) terms. It supplies
 * no exact Lorentz boost, velocity evolution, or frame-transformed opacity.
 *
 * Scope is level 0, double precision, Cartesian with any combination of
 * periodic and stationary reflecting directions, or RZ with reflecting radial
 * walls and periodic or reflecting z. RZ uses annular volumes
 * and face areas; enthalpy current vanishes exactly at both radial walls,
 * including the zero-area axis. RZ vectors are local (r,theta,z), not global
 * Cartesian momentum. Radial and axial velocities are cellwise physical
 * components; theta is the torque-conjugate secant from RZAngularConservative.
 * Its conjugate impulse is delta Lz/rbar, where rbar=<r>_V is the annular mean
 * radius, and its angular velocity reconstruction is vtheta(r)=vtheta*r/rbar.
 * Consequently diagnostic qtheta=c*Lz/rbar=(4/3)E*vtheta*<r^2>_V/(c*rbar^2).
 * The speed bound also applies at the outer radius of this reconstruction.
 * This diagnostic is not an evolved angular momentum equation, and does not
 * qualify a native packet/material adapter or centrifugal balance. Requested
 * work remains authoritative; diagnostic qtheta must not substitute for work.
 * Every valid mask cell must be 1 unless allow_kinetic_boundaries
 * explicitly permits 0/1 masks. K cells then require zero scalar energy, opacity
 * and requested work. Outward D/K enthalpy is added to face_leaked_energy for
 * caller-owned photon emission; it never creates K scalar energy. There is no
 * incoming scalar admission: physical incoming packets are handled elsewhere.
 * K lab residual and diagnostic inertia remain zero. This opt-in is a leading
 * lab-frame energy closure, not a moving Milne interface law.
 * The outgoing D-cell enthalpy CFL
 * dt/V sum_faces A max((4/3)v_out,0) must be at most one. Inputs and their
 * ghost cells are read-only. A globally rejected result has no candidate
 * fields and applies no flooring or clipping.
 */
MovingAsymptoticDiffusionFieldResult
TryAdvanceMovingAsymptoticDiffusionField (
    amrex::MultiFab const& lab_energy,
    amrex::MultiFab const& opacity,
    amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& work_velocity,
    amrex::Geometry const& geometry,
    amrex::Real dt,
    amrex::MultiFab const& requested_lab_work,
    AsymptoticDiffusionFieldOptions const& options,
    bool allow_kinetic_boundaries = false);

} // namespace warpx::radiation

#endif // WARPX_RADIATION_MOVING_ASYMPTOTIC_DIFFUSION_FIELD_H_
