/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_MOVINGTHERMALDIFFUSIONFIELD_H_
#define WARPX_RADIATION_MOVINGTHERMALDIFFUSIONFIELD_H_

#include "CoupledImplicitDiffusion.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <memory>

namespace warpx::radiation {
struct MovingThermalDiffusionFieldOptions
{
    MovingThermalDiffusionFieldOptions ()
    {
        coupled.diffusion = {};
        auto& diffusion = coupled.diffusion;
        diffusion.tolerance = 1.e-12;
        diffusion.linear_tolerance = 1.e-14;
        diffusion.minimum_optical_depth = 8;
        diffusion.max_iterations = 100;
        diffusion.max_linear_iterations = 200;
        diffusion.nonlinear_relaxation = 1;
        diffusion.use_incremental_form = true;
        diffusion.use_asymptotic_diffusion = true;
    }
    CoupledImplicitOptions coupled;
    amrex::Real transport_opacity = 0;
    amrex::Real absorption_opacity = 0;
    /** Opt in to cell-dependent coefficients from callbacks.coefficients.
     * The scalar opacities above are ignored. The callback returns chi_t,
     * c*chi_a and LTE emission, using frozen density/composition and the
     * supplied trial nodal temperature and time. No coefficients are frozen
     * across nonlinear temperature trials. */
    bool variable_opacity = false;
    /** Require node-local ideal-caloric convergence, including homogeneous
     * initial states that develop nonuniform heating during a trial. */
    bool require_local_ideal_accuracy = true;
};

enum class MovingThermalDiffusionFieldFailure {
    None,
    InvalidState,
    UnsupportedGeometry,
    EnthalpyCfl,
    AdvectedState,
    CoupledFailure,
    NonfiniteDiagnostic,
    PhysicalResidual
};

struct MovingThermalDiffusionFieldResult
{
    bool valid = false;
    MovingThermalDiffusionFieldFailure failure = MovingThermalDiffusionFieldFailure::None;
    CoupledImplicitResult coupled;
    amrex::MultiFab radiation;
    amrex::MultiFab nodal_temperature;
    // Native cell-corner quadrature of actual electron caloric change [J].
    amrex::MultiFab material_energy;
    // Constitutive physical heat sent to the native nodal deposition [J].
    amrex::MultiFab requested_material_energy;
    // Actual minus requested heat: spatial remap, not mechanical work/carry.
    amrex::MultiFab thermal_deposition_redistribution;
    amrex::MultiFab material_impulse;
    amrex::MultiFab mechanical_debit;
    amrex::MultiFab mechanical_operand_scale;
    // Reaction on stationary reflecting walls [kg m/s]. In RZ these are
    // meridional (r, theta, z), not a global Cartesian radial inventory.
    amrex::GpuArray<amrex::Real, 3> radiation_wall_momentum{};
    // Laboratory face flux [W/m^2], including advected enthalpy.
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> face_lab_energy_current;
    explicit
    operator bool () const noexcept
    {
        return valid;
    }
};

/** One private, first-order static-diffusion source-stage trial.
 * E and work are J/cell, velocity is the native three-component secant (m/s),
 * and T is the existing nodal electron state. With r=chi_a/chi_t, the equations
 * are Heat=dt*(c*chi_a*E-V*emission(T))-2*r*Work and
 * Enew-Eold+dt*div_A(F0+4*Eold_upwind*v_face/(3*V))+Work+Heat=0.
 * F0=-c*grad(Enew/V)/(3*chi_t); impulse=-dt*V*grad(Enew/V)/3.
 * Material callbacks retain the native caloric inversion and residual metric.
 * Primary thermal trials eliminate Qstat from the conservative spatial energy
 * equation. Accepted radiation is never corrected by this elimination; final
 * full constitutive heating and native caloric consistency remain mandatory.
 * Coefficients are checked at every temperature trial. Constant mode checks
 * agreement with the two scalar options; variable mode uses callback values.
 * Variable face diffusion uses c/(3*mean_face(chi_t)), matching the spatial
 * operator. Both the (1-2*r)*Work radiation RHS shift and -2*r*Work caloric
 * offset are refreshed at the same trial as absorption/emission. The primary
 * thermal elimination reconstructs that trial's conservative face flux.
 *
 * Supports Cartesian or axisymmetric RZ double precision with periodic faces
 * or stationary reflecting walls; chi_t*min(dx)>=8, beta<=.01.
 * RZ uses annular volumes/areas and a well-balanced isotropic-pressure force.
 * Positive absorption also requires beta*chi_t*max(domain_length)<=.01.
 * Moving absorbing cells must satisfy r*abs(e-B)/max(e,B)<=1/(chi_t*L)^2
 * at incoming and accepted states, with B=emission/(c*chi_a). Both-zero states
 * pass; exactly stationary cells and chi_a=0 bypass this equilibrium bound.
 * Physical qualification is restricted to near-equilibrium static diffusion;
 * arbitrary moving thermal disequilibrium and dynamic diffusion are excluded.
 * The caller must converge work/impulse/secant consistency and check the
 * independently reconstructed mechanical_debit against actual native work.
 * The local radiation equation uses requested_material_energy. Actual native
 * material_energy is redistributed by shared-node deposition; its difference
 * is reported separately and must not be mistaken for mechanical work.
 * No inner subcycling or source reconciliation is performed. Negative shifted
 * RHS is rejected without clipping. Temperature ghosts are rebuilt privately
 * before callbacks; zero absorption requires exactly zero LTE emissivity.
 * All inputs including ghosts are immutable;
 * rejected results contain no candidate fields. Outer retries own subcycling.
 */
MovingThermalDiffusionFieldResult TryAdvanceMovingThermalDiffusionField (
    amrex::MultiFab const& old_radiation, amrex::MultiFab const& old_nodal_temperature,
    amrex::MultiFab const& work_velocity, amrex::MultiFab const& requested_work,
    amrex::Geometry const& geometry, amrex::Real time, amrex::Real dt,
    MovingThermalDiffusionFieldOptions const& options, CoupledImplicitCallbacks const& callbacks);
} // namespace warpx::radiation
#endif
