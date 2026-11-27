/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_ASYMPTOTIC_DIFFUSION_FIELD_H_
#define WARPX_RADIATION_ASYMPTOTIC_DIFFUSION_FIELD_H_

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>
#include <AMReX_SPACE.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>

#include <memory>

namespace warpx::radiation
{

/** Default Marshak extrapolation length in Densmore et al. 2005, in mean free paths. */
inline constexpr amrex::Real asymptotic_marshak_lambda = amrex::Real(0.7104);

/** True when P(mu)=4(1+3 mu/2)/(3 tau+6 lambda) is a probability for all mu in [0,1].
 *
 * Invalid optical depths are rejected; they are never clamped into [0,1].
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
ValidAsymptoticInterfaceOpticalDepth (
    amrex::Real tau, amrex::Real lambda = asymptotic_marshak_lambda) noexcept
{
    if (!(tau > 0) || !amrex::Math::isfinite(tau) || !(lambda > 0) ||
        !amrex::Math::isfinite(lambda))
    {
        return false;
    }
    amrex::Real const denom = 3 * tau + 6 * lambda;
    if (!(denom > 0) || !amrex::Math::isfinite(denom))
    {
        return false;
    }
    amrex::Real const p_min = 4 / denom;
    amrex::Real const p_max = 10 / denom;
    return p_min >= 0 && p_max <= 1 && amrex::Math::isfinite(p_min) &&
           amrex::Math::isfinite(p_max);
}

/** Controls for one level-0 scalar asymptotic-diffusion field step. */
struct AsymptoticDiffusionFieldOptions
{
    /** MLMG relative residual tolerance. Default matches the discrete ledger bound. */
    amrex::Real linear_tolerance = amrex::Real(1.e-12);
    /** Independent cell-ledger relative residual bound. */
    amrex::Real residual_tolerance = amrex::Real(1.e-12);
    int max_linear_iterations = 200;
    int verbosity = 0;
    amrex::Real marshak_extrapolation_lambda = asymptotic_marshak_lambda;
};

enum class AsymptoticDiffusionFieldFailure
{
    None,
    InvalidState,
    UnsupportedGeometry,
    LinearConvergence,
    NonnegativeSolution,
    NonfiniteCoefficient,
    ConservationResidual
};

/** Candidate of one accepted field step, or empty MultiFabs when `valid` is false.
 *
 * Inputs are never mutated. On rejection every output MultiFab is undefined
 * (`ok()==false`) so there is no candidate energy, flux, impulse, or residual.
 */
struct AsymptoticDiffusionFieldResult
{
    AsymptoticDiffusionFieldResult () = default;
    AsymptoticDiffusionFieldResult (AsymptoticDiffusionFieldResult&&) = default;
    AsymptoticDiffusionFieldResult& operator= (AsymptoticDiffusionFieldResult&&) = default;
    AsymptoticDiffusionFieldResult (AsymptoticDiffusionFieldResult const&) = delete;
    AsymptoticDiffusionFieldResult& operator= (AsymptoticDiffusionFieldResult const&) = delete;

    bool valid = false;
    AsymptoticDiffusionFieldFailure failure = AsymptoticDiffusionFieldFailure::None;
    int linear_iterations = 0;

    /** Accepted cell-integrated radiation energy [J]. */
    amrex::MultiFab energy;
    /** Composite MLMG residual in the normalized cell equation. The joule
     * ledger allowance includes this explicitly reported algebraic error; it
     * is never used to hide a boundary flux or work term. */
    amrex::Real linear_residual = 0;
    /** Densmore material impulse [kg m/s], physical (x,y,z), or local
     * (r,theta,z) in RZ. RZ uses the arithmetic average of face currents. */
    amrex::MultiFab material_impulse;
    /** Independent scalar ledger DeltaE + sum Qout + work [J] per cell. */
    amrex::MultiFab energy_balance_residual;
    /** Diagnostic Q_A = V/(2c) sum Fout n [J]; not a causal P1/M1 state. */
    amrex::MultiFab radiation_inertia;
    /** R = Q_Anew-Q_Aold + c dt sum A Pi n + c I [J]. Model/omitted-inertia
     * residual, not numerical momentum loss or the material impulse. */
    amrex::MultiFab closure_diagnostic;

    /** Face current F(L->R) [W m^-2], AMReX-direction positive. */
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> face_current;
    /** Isotropic face pressure Pi=e_face/3 [Pa]. */
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> face_pressure;
    /** Weighted or reconstructed face energy density [J m^-3]. */
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> face_energy_density;
    /** Positive D-to-K leakage [J], computed directly without cancellation
     * against incident admission. Zero on D/D, K/K and reflecting faces. */
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> face_leaked_energy;

    amrex::Real global_energy_balance_residual = 0;
    amrex::Real max_energy_balance_residual = 0;

    explicit operator bool () const noexcept { return valid; }
};

/** Advance one frozen-gray scalar asymptotic-diffusion field step.
 *
 * This is a native AMReX MLMG field backend, not photon marching, packet
 * conversion, or a complete radiation runtime. The unknown is cell energy
 * density e=E/V. All inputs are read-only. Kinetic (mask=0) cells must have
 * zero energy; diffusion (mask=1) cells need positive opacity. D/K faces use
 * the Densmore leakage/admission law and require a valid interface
 * probability. Outer non-periodic faces are reflecting (F=0, e_face=e_D).
 * Periodic faces use canonical AMReX ownership with no duplicated sources.
 *
 * D/D face: denom=chi_L dx_L+chi_R dx_R, F(L->R)=-2c(e_R-e_L)/(3 denom),
 * e_face=(chi_R dx_R e_L+chi_L dx_L e_R)/denom, native bcoef=2 c dx/(3 denom).
 * Equal AMReX side widths still keep distinct side opacities.
 *
 * Geometry: Cartesian level 0, double precision, periodic and/or reflecting
 * walls; RZ supports reflecting radial walls and periodic or reflecting z with nonnegative
 * inner radius. Both WarpX's Coord()==0 and explicit Coord()==1 RZ geometry
 * use annular volumes and face areas. The axis has exactly zero face area
 * and current. Radial stress includes the geometric -P_theta_theta/r term.
 * RZ vector diagnostics use local (r,theta,z); theta is zero, and summed
 * radial components are not conserved Cartesian momentum. Axial components
 * retain their lab-frame meaning; this stationary scalar model transports
 * no azimuthal momentum. It does not enable moving RZ, native photon
 * transactions, or material recoil coupling. Not AMR, not LTE/absorption
 * tables, no P1 momentum unknown, no flux-cone gate on the scalar interior.
 */
AsymptoticDiffusionFieldResult
TryAdvanceAsymptoticDiffusionField (
    amrex::MultiFab const& energy,
    amrex::MultiFab const& opacity,
    amrex::iMultiFab const& diffusion_mask,
    amrex::Geometry const& geometry,
    amrex::Real dt,
    amrex::MultiFab const& requested_work,
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> const& admitted_energy,
    amrex::MultiFab const* old_radiation_inertia,
    AsymptoticDiffusionFieldOptions const& options);

} // namespace warpx::radiation

#endif // WARPX_RADIATION_ASYMPTOTIC_DIFFUSION_FIELD_H_
