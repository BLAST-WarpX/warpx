/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "RadiationTransport.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "HybridCaloricResidual.H"
#include "HybridRadiationCoefficients.H"
#include "ImplicitMaterialTableCoefficients.H"
#include "MovingThermalDiffusion.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhotonParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Print.H>

#include <memory>

using warpx::fields::FieldType;

void
RadiationTransport::AdvanceMovingThermalDiffusion (MultiParticleContainer& particles,
                                                   ablastr::fields::MultiFabRegister& fields,
                                                   amrex::Real current_time, amrex::Real dt) const
{
    using namespace warpx::radiation;
    auto const& simulation = WarpX::GetInstance();
    auto const& geometry = simulation.Geom(0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_model &&
            (m_hybrid_model->electronThermodynamicsExecutor().isIdealGas() ||
             m_hybrid_model->electronThermodynamicsExecutor().isFixedChargeLatentEnergy() ||
             m_hybrid_model->electronThermodynamicsExecutor().isSingularitySpiner()) &&
            geometry.Coord() == 0 && simulation.finestLevel() == 0 &&
            !EB::enabled() && WarpX::do_moving_window == 0 && !particles.hasHybridIonization(),
        "moving_thermal requires closed level-zero fixed-charge hybrid electrons "
        "with a supported native caloric model.");
    auto* photons = dynamic_cast<PhotonParticleContainer*>(
        &particles.GetParticleContainerFromName(m_photon_species));
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(photons && photons->TotalNumberOfParticles() == 0,
                                     "moving_thermal requires an empty photon species.");
    auto& radiation = *fields.get(FieldType::radiation_diffusion_energy, 0);
    auto& heat = *fields.get(FieldType::radiation_material_energy, 0);
    auto const& live_temperature = *fields.get(FieldType::hybrid_electron_temperature_fp, 0);
    auto const& live_density = *fields.get(FieldType::rho_fp, 0);
    amrex::MultiFab temperature(live_temperature.boxArray(), live_temperature.DistributionMap(), 1,
                                live_temperature.nGrowVect());
    amrex::MultiFab density(live_density.boxArray(), live_density.DistributionMap(),
                            live_density.nComp(), live_density.nGrowVect());
    amrex::MultiFab::Copy(temperature, live_temperature, 0, 0, 1, temperature.nGrowVect());
    amrex::MultiFab::Copy(density, live_density, 0, 0, live_density.nComp(), density.nGrowVect());
    temperature.FillBoundary(geometry.periodicity());
    density.FillBoundary(geometry.periodicity());
    auto const eos = m_hybrid_model->electronThermodynamicsExecutor();
    detail::CaloricMaterialFields material_fields{};
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> material_snapshots(eos.m_num_materials);
    for (int index = 0; index < eos.m_num_materials; ++index)
    {
        auto const& live = *fields.get(
            "ni_charge_fp_" + m_hybrid_model->electronThermodynamicsMaterialSpeciesName(index), 0);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            live.boxArray() == temperature.boxArray() &&
                live.DistributionMap() == temperature.DistributionMap() &&
                live.nGrowVect().allGE(temperature.nGrowVect()),
            "Moving thermal composition must share the native nodal temperature layout.");
        auto& snapshot = material_snapshots[index];
        snapshot = std::make_unique<amrex::MultiFab>(live.boxArray(), live.DistributionMap(),
                                                    live.nComp(), live.nGrowVect());
        amrex::MultiFab::Copy(*snapshot, live, 0, 0, live.nComp(), live.nGrowVect());
        snapshot->FillBoundary(geometry.periodicity());
        material_fields[index] = snapshot.get();
    }
    auto const minimum_density = m_minimum_electron_density;
    auto const thermodynamic_floor = m_hybrid_model->transportsElectronInternalEnergyAtRawDensity()
                                         ? amrex::Real(0)
                                         : m_hybrid_model->electronDensityFloor();
    detail::OpacityEvaluator<1> const planck{false,
                                             0,
                                             {},
                                             {},
                                             {},
                                             {},
                                             m_use_planck_table,
                                             m_use_spectral_planck_table,
                                             m_planck_table.executor(),
                                             m_spectral_planck_table.executor(),
                                             m_planck_absorption_coefficient};
    detail::OpacityEvaluator<1> const transport{false,
                                                0,
                                                {},
                                                {},
                                                {},
                                                {},
                                                m_use_rosseland_table,
                                                m_use_spectral_rosseland_table,
                                                m_rosseland_table.executor(),
                                                m_spectral_rosseland_table.executor(),
                                                m_rosseland_transport_coefficient};
    CoupledImplicitCallbacks callbacks;
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
    MaterialOpacityTableExecutor material_table;
    amrex::Real mass_per_electron = 0;
    if (m_use_registered_material_opacity_tables)
    {
        auto const& carrier = particles.GetParticleContainerFromName(m_opacity_species[0]);
        mass_per_electron = static_cast<amrex::Real>(carrier.getMass()) * PhysConst::q_e /
                            static_cast<amrex::Real>(carrier.getCharge());
        material_table = m_registered_material_opacity_tables[0].executor();
    }
#endif
    callbacks.coefficients = [&] (amrex::MultiFab const& trial, amrex::MultiFab& opacity,
                                  amrex::MultiFab& absorption, amrex::MultiFab& emission,
                                  amrex::Real evaluation_time) {
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
        if (m_use_registered_material_opacity_tables)
        {
            detail::FillCoupledMaterialTableCoefficients(
                density, trial, opacity, absorption, emission, geometry, eos, minimum_density,
                thermodynamic_floor, material_table, mass_per_electron, m_energy_groups,
                material_fields);
            return;
        }
#endif
        detail::FillCoupledHybridCoefficients(
            density, trial, opacity, absorption, emission, geometry, eos, minimum_density,
            thermodynamic_floor, planck, transport, m_energy_groups, evaluation_time,
            material_fields);
    };
    callbacks.material_response = [&] (amrex::MultiFab& trial, amrex::MultiFab& source) {
        return m_hybrid_model->EvaluateElectronEnergySource(0, trial, source, minimum_density);
    };
    if (!eos.isIdealGas()) {
        callbacks.material_residual = [&] (amrex::MultiFab const& old,
            amrex::MultiFab const& candidate, amrex::MultiFab const& check) {
            return detail::CoupledCaloricResidual(density, old, candidate, check, eos,
                                                  minimum_density, thermodynamic_floor,
                                                  material_fields);
        };
        callbacks.material_local_residual = [&] (amrex::MultiFab const& old,
            amrex::MultiFab const& candidate, amrex::MultiFab const& check,
            amrex::MultiFab const& lower, amrex::MultiFab const& upper, amrex::Real tolerance) {
            return detail::CoupledLocalCaloricResidual(density, old, candidate, check, lower, upper,
                eos, minimum_density, thermodynamic_floor, material_fields, tolerance);
        };
    }
    MovingThermalDiffusionOptions options;
#if defined(WARPX_DIM_RZ)
    options.material.assignment = ParticleImpulseAssignment::RZAngularConservative;
#endif
    options.material.max_iterations = m_moving_diffusion_max_iterations;
    options.max_subdivisions = m_coupled_max_subdivisions;
    options.field.transport_opacity = m_moving_thermal_transport;
    options.field.absorption_opacity = m_moving_thermal_absorption;
    options.field.variable_opacity = m_moving_thermal_variable_opacity;
    // Local caloric accuracy must not depend on a domain-wide contrast switch.
    options.field.require_local_ideal_accuracy = true;
    auto& coupled = options.field.coupled;
    coupled.max_iterations = m_lte_exchange_max_iterations;
    coupled.material_tolerance = m_lte_exchange_tolerance;
    coupled.relaxation = 0.5;
    auto& diffusion = coupled.diffusion;
    diffusion = {};
    diffusion.tolerance = 1.e-12;
    diffusion.linear_tolerance = 1.e-14;
    diffusion.minimum_optical_depth = m_minimum_diffusion_optical_depth;
    diffusion.max_iterations = 100;
    diffusion.max_linear_iterations = 200;
    diffusion.verbosity = m_moving_diffusion_verbosity;
    diffusion.energy_groups = m_energy_groups;
    diffusion.use_incremental_form = true;
    diffusion.nonlinear_relaxation = 1;
    auto result = TryAdvanceMovingThermalDiffusion(particles, *photons, m_momentum_species,
                                                   "diffusion_0", radiation, temperature, heat,
                                                   geometry, current_time, dt, options, callbacks);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(result.valid, "Bounded moving thermal source rejected without "
                                                   "committing radiation, electrons or ions: " +
                                                       result.reason);
    m_hybrid_model->CommitElectronTemperature(0, temperature);
    m_last_boundary_energy_loss = 0;
    m_last_diffusion_boundary_energy_loss = 0;
    m_last_boundary_energy_injection = 0;
    m_last_streaming_boundary_energy_loss = 0;
    m_last_diffusion_boundary_momentum_loss = result.radiation_wall_momentum;
    m_last_streaming_boundary_momentum_loss = {0, 0, 0};
    m_last_implicit_diffusion = {};
    auto& momentum = *fields.get(FieldType::radiation_material_momentum, 0);
    auto& kinetic = *fields.get(FieldType::radiation_material_kinetic_energy, 0);
    momentum.setVal(0);
    kinetic.setVal(0);
    amrex::MultiFab::Copy(momentum, result.actual_material_impulse, 0, 0, 3, 0);
    amrex::MultiFab::Copy(kinetic, result.actual_material_work_field, 0, 0, 1, 0);
    m_last_numerical_energy_residual = result.raw_energy_residual;
    m_cumulative_numerical_energy_residual += result.raw_energy_residual;
    m_last_material_carry_energy_change = result.material_energy_carry_change;
    m_last_coupled_solve = {
        result.work_iterations,    result.accepted_substeps, result.rejected_attempts, 0, 0,
        result.raw_energy_residual};
    if (m_moving_diffusion_verbosity > 0) {
        amrex::Print() << "Moving thermal accepted substeps=" << result.accepted_substeps
                       << " retries=" << result.rejected_attempts << " heat=" << heat.sum(0)
                       << " actual_work=" << result.actual_material_work
                       << " thermal_realization=" << result.thermal_realization_residual
                       << " carry=" << result.material_energy_carry_change
                       << " residual=" << result.raw_energy_residual << '\n';
    }
}
