/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "RadiationTransport.H"

#include "CoupledImplicitDiffusion.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "HybridCaloricResidual.H"
#include "HybridRadiationCoefficients.H"
#include "ImplicitMaterialTableCoefficients.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Array4.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>
#include <memory>
#include <string>

using namespace amrex::literals;
using warpx::fields::FieldType;
using warpx::radiation::detail::CoupledCaloricResidual;
using warpx::radiation::detail::FillCoupledHybridCoefficients;
using warpx::radiation::detail::OpacityEvaluator;

void
RadiationTransport::AdvanceCoupledImplicit (ablastr::fields::MultiFabRegister& fields,
                                            amrex::Real const current_time,
                                            amrex::Real const dt) const
{
    auto& warpx = WarpX::GetInstance();
    auto const& geometry = warpx.Geom(0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_model != nullptr &&
            (m_hybrid_model->electronThermodynamicsExecutor().isIdealGas() ||
             m_hybrid_model->electronThermodynamicsExecutor().isFixedChargeLatentEnergy() ||
             (m_hybrid_model->electronThermodynamicsExecutor().isSingularitySpiner() &&
              m_hybrid_model->electronThermodynamicsNumMaterials() == 1)),
        "coupled_implicit requires a native analytic or single-material electron EOS.");
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
    auto& radiation = *fields.get(FieldType::radiation_diffusion_energy, 0);
    auto& material = *fields.get(FieldType::radiation_material_energy, 0);
    warpx::radiation::CoupledImplicitOptions controls;
    controls.diffusion = m_implicit_diffusion_options;
    controls.max_iterations = m_lte_exchange_max_iterations;
    controls.material_tolerance = m_lte_exchange_tolerance;
    controls.relaxation = 1.0_rt;
    controls.max_subdivisions = m_coupled_max_subdivisions;
    auto const eos = m_hybrid_model->electronThermodynamicsExecutor();
    amrex::GpuArray<amrex::MultiFab const*, ElectronThermodynamicsExecutor::max_materials>
        material_fields{};
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> material_snapshots(eos.m_num_materials);
    for (int index = 0; index < eos.m_num_materials; ++index)
    {
        auto const& live = *fields.get(
            "ni_charge_fp_" + m_hybrid_model->electronThermodynamicsMaterialSpeciesName(index), 0);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            live.boxArray() == temperature.boxArray() &&
                live.DistributionMap() == temperature.DistributionMap() &&
                live.nGrowVect().allGE(temperature.nGrowVect()),
            "Coupled EOS material densities must share the native nodal temperature layout.");
        auto& snapshot = material_snapshots[index];
        snapshot = std::make_unique<amrex::MultiFab>(live.boxArray(), live.DistributionMap(),
                                                     live.nComp(), live.nGrowVect());
        amrex::MultiFab::Copy(*snapshot, live, 0, 0, live.nComp(), live.nGrowVect());
        snapshot->FillBoundary(geometry.periodicity());
        material_fields[index] = snapshot.get();
    }
    auto const minimum_density = m_minimum_electron_density;
    auto const thermodynamic_floor = m_hybrid_model->transportsElectronInternalEnergyAtRawDensity()
                                         ? 0.0_rt
                                         : m_hybrid_model->electronDensityFloor();
    // No species composition is implied by electron-state tables. These views
    // reuse the existing interpolation and endpoint policy without allocating
    // the unrelated maximum-species bundle in every coupled GPU kernel.
    OpacityEvaluator<1> const planck{false,
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
    OpacityEvaluator<1> const rosseland{false,
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
    auto const groups = m_energy_groups;
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
    warpx::radiation::MaterialOpacityTableExecutor material_table;
    amrex::Real mass_per_electron = 0;
    if (m_use_registered_material_opacity_tables) {
        // Constructor admission and the metadata audit require exactly one
        // matching fixed-charge carrier. Preserve the native nodal quadrature.
        auto const& species =
            warpx.GetPartContainer().GetParticleContainerFromName(
                m_opacity_species[0]);
        mass_per_electron = static_cast<amrex::Real>(species.getMass()) *
                            PhysConst::q_e /
                            static_cast<amrex::Real>(species.getCharge());
        material_table = m_registered_material_opacity_tables[0].executor();
    }
#endif
    warpx::radiation::CoupledImplicitCallbacks callbacks;
    callbacks.material_response = [&] (amrex::MultiFab& trial, amrex::MultiFab& source) {
        return m_hybrid_model->EvaluateElectronEnergySource(0, trial, source, minimum_density);
    };
    if (!eos.isIdealGas()) {
        callbacks.material_residual = [&] (amrex::MultiFab const& old,
                                           amrex::MultiFab const& candidate,
                                           amrex::MultiFab const& check)
        {
            return CoupledCaloricResidual(density, old, candidate, check, eos, minimum_density,
                                          thermodynamic_floor, material_fields);
        };
    }
    callbacks.coefficients = [&] (amrex::MultiFab const& trial,
                                  amrex::MultiFab& opacity,
                                  amrex::MultiFab& absorption,
                                  amrex::MultiFab& emission,
                                  amrex::Real evaluation_time) {
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
        if (m_use_registered_material_opacity_tables) {
            warpx::radiation::detail::FillCoupledMaterialTableCoefficients(
                density, trial, opacity, absorption, emission, geometry, eos,
                minimum_density, thermodynamic_floor, material_table,
                mass_per_electron, groups, material_fields);
            return;
        }
#endif
        FillCoupledHybridCoefficients(density, trial, opacity, absorption, emission, geometry, eos,
                                      minimum_density, thermodynamic_floor, planck, rosseland,
                                      groups, evaluation_time, material_fields);
    };
    auto const result = warpx::radiation::TryAdvanceCoupledImplicitSubcycled(
        radiation, temperature, material, geometry, current_time, dt, controls, callbacks);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        result.failure == warpx::radiation::CoupledImplicitFailure::None,
        "Coupled implicit radiation/material solve failed (reason " +
            std::to_string(static_cast<int>(result.failure)) +
            "). "
            "No radiation or material candidate was committed; reduce the timestep.");
    m_hybrid_model->CommitElectronTemperature(0, temperature);
    fields.get(FieldType::radiation_material_kinetic_energy, 0)->setVal(0);
    fields.get(FieldType::radiation_material_momentum, 0)->setVal(0);
    m_last_implicit_diffusion = result.diffusion;
    m_last_coupled_solve = {result.iterations,
                            result.accepted_substeps,
                            result.rejected_attempts,
                            result.source_consistency_corrections,
                            result.material_relative_residual,
                            result.raw_energy_relative_residual};
    m_last_diffusion_boundary_energy_loss = result.diffusion.escaped_energy;
    m_last_boundary_energy_loss = result.diffusion.escaped_energy;
    m_last_boundary_energy_injection = result.diffusion.injected_energy;
    m_last_numerical_energy_residual =
        result.diffusion.numerical_energy_residual + result.material_realization_residual;
    if (m_track_energy_balance) {
        m_cumulative_boundary_energy_loss += m_last_boundary_energy_loss;
        m_cumulative_boundary_energy_injection += m_last_boundary_energy_injection;
    }
    m_cumulative_numerical_energy_residual += m_last_numerical_energy_residual;
    if (m_implicit_diffusion_options.verbosity > 0 || result.source_consistency_corrections > 0) {
        amrex::Print() << "Coupled radiation time=" << current_time + dt
            << " accepted_substeps=" << result.accepted_substeps
            << " rejected_attempts=" << result.rejected_attempts
            << " iterations=" << result.iterations
            << " source_consistency_corrections=" << result.source_consistency_corrections
                       << " radiation_residual=" << result.diffusion.maximum_relative_residual
                       << " material_residual=" << result.material_relative_residual
                       << " raw_energy_residual=" << result.raw_energy_relative_residual << '\n';
    }
}
