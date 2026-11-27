/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_MOVINGTHERMALDIFFUSION_H_
#define WARPX_RADIATION_MOVINGTHERMALDIFFUSION_H_

#include "MovingThermalDiffusionField.H"
#include "PacketMaterialWorkIteration.H"

namespace warpx::radiation {
struct MovingThermalDiffusionOptions
{
    MovingThermalDiffusionFieldOptions field;
    PacketMaterialWorkIterationOptions material;
    int max_subdivisions = 0;
    amrex::Real energy_relative_tolerance = 1.e-10;
};

struct MovingThermalDiffusionResult
{
    bool valid = false;
    std::string reason;
    int work_iterations = 0;
    int accepted_substeps = 0;
    int rejected_attempts = 0;
    amrex::Real actual_material_work = 0;
    amrex::Real material_energy_carry_change = 0;
    amrex::Real raw_energy_residual = 0;
    /** Sum of native nodal caloric realization minus cell radiation heat
     * source. Spatial redistribution cancels globally; this is not ion work
     * carry. */
    amrex::Real thermal_realization_residual = 0;
    amrex::GpuArray<amrex::Real, 3> radiation_wall_momentum{};
    amrex::MultiFab actual_material_impulse;
    amrex::MultiFab actual_material_work_field;
    amrex::MultiFab material_momentum_carry_change;
};

/** Closed radiation source stage; positions and density are frozen. Radiation
 * and heat are cell-integrated J, temperature is native nodal K. All callbacks
 * operate only on scratch. Each retry begins from the same outer state; every
 * accepted substep updates only private particles/fields until the entire
 * interval passes. No numerical rejection occurs after live publication.
 * Empty photons, fixed-charge native ions, Cartesian or single-mode RZ with
 * periodic faces or stationary reflecting walls. RZ uses torque-conjugate
 * assignment internally; returned impulse/carry fields are physical (r,theta,z).
 * The RZ global invariants are axial momentum and angular momentum, not the
 * sum of local radial components. Runtime
 * admits ideal, builtin latent, or table electrons with their respective caloric
 * metrics and frozen composition views. Native material transport retains its
 * separate single-table/composition limitations; this source transaction does
 * not generalize that transport operator.
 * See ASTRA_MOVING_THERMAL_MODEL.md for the static diffusion approximation.
 */
MovingThermalDiffusionResult TryAdvanceMovingThermalDiffusion (
    MultiParticleContainer& particles, WarpXParticleContainer& photons,
    std::vector<std::string> const& species, std::string const& path, amrex::MultiFab& radiation,
    amrex::MultiFab& nodal_temperature, amrex::MultiFab& material_energy,
    amrex::Geometry const& geometry, amrex::Real time, amrex::Real dt,
    MovingThermalDiffusionOptions const& options, CoupledImplicitCallbacks const& callbacks);
} // namespace warpx::radiation
#endif
