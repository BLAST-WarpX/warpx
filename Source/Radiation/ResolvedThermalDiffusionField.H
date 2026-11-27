/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_RESOLVEDTHERMALDIFFUSIONFIELD_H_
#define WARPX_RADIATION_RESOLVEDTHERMALDIFFUSIONFIELD_H_

#include "AsymptoticDiffusionField.H"

#include <string>

namespace warpx::radiation
{
struct ResolvedThermalDiffusionFieldOptions
{
    AsymptoticDiffusionFieldOptions linear;
    amrex::Real radiation_constant = 7.565733250280007e-16;
    amrex::Real minimum_optical_depth = 8;
    amrex::Real maximum_beta = 0.01;
    /** New masked-interface mode. Implicit LAB current contains the missing
     * -r*v*(E-E_LTE) absorption drift; LTE centering uses exact Lorentz columns.
     * Existing all-D/static guards remain unchanged when false. */
    bool lte_centered_moving_source = false;
    int max_drift_iterations = 20;
};

struct ResolvedThermalDiffusionFieldResult
{
    bool valid = false;
    std::string reason;
    int linear_iterations = 0;
    amrex::Real linear_residual = 0;
    int drift_iterations = 0;
    amrex::Real drift_equation_residual = 0;
    amrex::MultiFab energy;
    amrex::MultiFab lab_momentum;
    amrex::MultiFab requested_heat;
    amrex::MultiFab physical_energy_debit;
    amrex::MultiFab raw_energy_debit;
    amrex::MultiFab storage_roundoff_bound;
    amrex::MultiFab outward_pressure_stress;
    amrex::MultiFab operand_scale;
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> face_lab_energy_current;
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> face_pressure;
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> face_leaked_energy;
};

/** Frozen-temperature masked gray spatial/thermal trial, never a live update.
 *
 * E is extensive LAB energy. chi_t=chi_s+chi_a and chi_a are proper inverse
 * lengths, refreshed by the caller before this trial. The bounded static
 * diffusion equations are Enew-Eold+dt div(F0+H)+Work+Heat=0,
 * Heat=dt*c*chi_a*(Enew-V*a_R*T^4)-2*(chi_a/chi_t)*Work,
 * F0=-c grad(Enew/V)/(3 chi_t), H=4/3 Eold_upwind*vface/V.
 * One total-opacity constitutive LAB q is reconstructed; no separate
 * absorption damping of that q is permitted. The caller supplies the total
 * conservative D mechanical work, including its retained q relaxation.
 *
 * The native MLMG unknown is a centered increment. Weak absorption centers on
 * old E; stiff absorption centers on the LTE bath. Heat retains that increment
 * separately from rounded field storage and is independently audited against
 * actual stored debit plus explicitly derived arithmetic/linear bounds.
 *
 * Cartesian periodic level zero, double precision, positive gray coefficients
 * in D cells, zero E/coefficients/work in K cells. Mixed D/K faces use the
 * existing zero-admission Densmore leakage law; chronological native packets
 * traverse D and are admitted only at census. Outgoing enthalpy is included
 * in leakage. Requires chi_t*dx>=8, beta<=.01, beta*chi_t*L<=.01 and the
 * MovingThermalDiffusionField incoming/accepted near-LTE moving envelope.
 * Exactly stationary cells bypass the moving near-LTE restriction. Explicit
 * enthalpy has outgoing CFL<=1. This is first-order static diffusion, not
 * arbitrary moving disequilibrium, dynamic diffusion or a causal moment solve.
 * D/K actual angular traction and finite native material/caloric acceptance
 * belong to the outer private coordinator, using these identical face currents.
 * The opt-in LTE-centered mode instead solves the accepted-time upwind current
 * H=c*(M_i0/M_00)*Enew/V-r*v*(Enew-E_LTE)/V at this SAME trial T/secant.
 * The identical accepted face current defines stored q; there is no endpoint
 * q-only repair or old-bath drift lag under stiff temperature changes. Source
 * heat is centered on E_LTE=M_00*V*a_R*T^4; its leading-O(beta) work term is
 * unchanged. Exact boosted homogeneous LTE is therefore a discrete fixed point.
 * Native MLMG Picard solves the bounded implicit drift, with an independently
 * reported equation residual. This new mode requires beta<=.001 and
 * beta*chi_t*L<=.01, positivity and outgoing drift CFL<=1; it does not impose the
 * all-D near-LTE guard whose omitted drift is now included. It remains a
 * leading-O(beta) static-diffusion/Milne-interface model, not exact moving P1.
 */
[[nodiscard]] ResolvedThermalDiffusionFieldResult TryAdvanceResolvedThermalDiffusionField (
    amrex::MultiFab const& old_energy, amrex::MultiFab const& cell_temperature,
    amrex::MultiFab const& transport_opacity, amrex::MultiFab const& absorption_opacity,
    amrex::iMultiFab const& diffusion_mask, amrex::MultiFab const& work_velocity,
    amrex::MultiFab const& requested_work, amrex::Geometry const& geometry, amrex::Real dt,
    ResolvedThermalDiffusionFieldOptions const& options,
    amrex::MultiFab const* incoming_cell_temperature = nullptr,
    amrex::MultiFab const* reciprocal_proper_bath_density = nullptr);
} // namespace warpx::radiation
#endif
