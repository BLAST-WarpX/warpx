/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "RadiationTransport.H"

#include "CoupledMomentSource.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "FluxRelaxedDiffusion.H"
#include "FluxRelaxedPackets.H"
#include "HybridRadiationCoefficients.H"
#include "LimitedMomentTransport.H"
#include "ParticleImpulse.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhotonParticleContainer.H"
#include "PhotonPacketTransaction.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace amrex::literals;
using warpx::fields::FieldType;
using warpx::radiation::ParticleImpulseMaterial;
using warpx::radiation::ParticleImpulseTransaction;
using warpx::radiation::PhotonPacketTransaction;
using warpx::radiation::detail::AdvanceCandidatePhotons;
using warpx::radiation::detail::BuildThinCellEmissionPlan;
using warpx::radiation::detail::EmissionPlan;
using warpx::radiation::detail::FillCoupledHybridCoefficients;
using warpx::radiation::detail::OpacityEvaluator;
using warpx::radiation::detail::PhotonInventory;
using warpx::radiation::detail::StageEmissionPlan;

namespace
{
// Keep extended CUDA lambdas in a free function, not a private member.
void
ValidateCoherentOpacity (amrex::MultiFab const& density, amrex::MultiFab& scattering,
                         amrex::MultiFab const& true_absorption)
{
    amrex::iMultiFab opacity_status(scattering.boxArray(), scattering.DistributionMap(), 1, 0);
    opacity_status.setVal(0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter iterator(scattering, amrex::TilingIfNotGPU()); iterator.isValid();
         ++iterator)
    {
        auto const rho = density.const_array(iterator);
        auto chi = scattering.array(iterator);
        auto absorption = true_absorption.array(iterator);
        auto status = opacity_status.array(iterator);
        amrex::ParallelFor(
            iterator.validbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                bool empty = true;
                for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner)
                {
                    empty = empty && rho(i + (corner & 1), j + ((corner >> 1) & 1),
                                         k + ((corner >> 2) & 1)) == 0.0_rt;
                }
                amrex::Real const planck_rate = absorption(i, j, k, 0);
                amrex::Real const transport_coefficient = chi(i, j, k, 0);
                bool valid = empty ? planck_rate == 0.0_rt
                                   : (planck_rate == 0.0_rt && transport_coefficient >= 0.0_rt);
                chi(i, j, k, 0) =
                    empty ? 0.0_rt : transport_coefficient - planck_rate / PhysConst::c;
                if (!valid || !amrex::Math::isfinite(chi(i, j, k, 0)) || chi(i, j, k, 0) < 0.0_rt)
                {
                    status(i, j, k, 0) = 1;
                    chi(i, j, k, 0) = 0.0_rt;
                }
            });
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        opacity_status.max(0) == 0,
        "Coherent radiation transport requires zero true absorption and "
        "finite non-negative gray opacity in every nonempty cell.");
}

[[nodiscard]] amrex::Real
SumComponent (amrex::MultiFab const& field, int const component) noexcept
{
    return field.sum(component, /*local=*/false);
}
} // namespace

void
RadiationTransport::AdvanceFluxRelaxed (MultiParticleContainer& particles,
                                        ablastr::fields::MultiFabRegister& fields,
                                        amrex::Real const current_time, amrex::Real const dt) const
{
    auto& simulation = WarpX::GetInstance();
    auto const& geometry = simulation.Geom(0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        geometry.Coord() == 0 && geometry.isAllPeriodic() && simulation.finestLevel() == 0 &&
            !EB::enabled(),
        "flux_relaxed requires level-zero periodic Cartesian geometry without EB.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_num_groups == 1 && m_enable_diffusion && m_enable_particle_conversion &&
            m_enable_momentum_coupling && !m_enable_lte_exchange && !m_has_diffusion_bath &&
            !m_use_species_opacity && !m_use_material_opacity_table &&
            m_momentum_species.size() == 1,
        "The first flux_relaxed mode is gray scattering-only packet/FLD "
        "conversion with native hybrid electrons and one momentum species.");

    auto* const photon_container = dynamic_cast<PhotonParticleContainer*>(
        &particles.GetParticleContainerFromName(m_photon_species));
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(photon_container != nullptr,
                                     "flux_relaxed requires the configured photon species.");
    auto& live_photons = *photon_container;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        live_photons.doNotPush() && !live_photons.DoResampling() && !live_photons.DoQED(),
        "flux_relaxed owns photon streaming and requires do_not_push=1, "
        "no photon resampling and no QED pair conversion.");
    auto& energy = *fields.get(FieldType::radiation_diffusion_energy, 0);
    std::string const q_names[3] = {"radiation_flux_relaxed_qx", "radiation_flux_relaxed_qy",
                                    "radiation_flux_relaxed_qz"};
    amrex::MultiFab* q_fields[3] = {};
    for (int d = 0; d < 3; ++d)
    {
        q_fields[d] = fields.get(q_names[d], 0);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            q_fields[d] != nullptr && q_fields[d]->nComp() == 1 &&
                q_fields[d]->nGrowVect().allGE(1),
            "Missing or incompatible persistent flux-relaxed q field.");
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(energy.nComp() == 1 && energy.nGrowVect().allGE(1) &&
                                         energy.ixType().cellCentered(),
                                     "Persistent flux-relaxed energy has an incompatible layout.");

    amrex::MultiFab initial_radiation(energy.boxArray(), energy.DistributionMap(), 4, 1);
    initial_radiation.setVal(0.0_rt);
    amrex::MultiFab::Copy(initial_radiation, energy, 0, 0, 1, 0);
    for (int d = 0; d < 3; ++d)
    {
        amrex::MultiFab::Copy(initial_radiation, *q_fields[d], 0, d + 1, 1, 0);
    }
    initial_radiation.FillBoundary(geometry.periodicity());
    amrex::GpuArray<amrex::Real, 4> initial_photons{};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        PhotonInventory(live_photons, initial_photons),
        "flux_relaxed received invalid live photon positions or momenta.");

    amrex::MultiFab scattering(energy.boxArray(), energy.DistributionMap(), 1, 1);
    FillFrozenCoherentOpacity(fields, geometry, current_time, scattering);
    auto const dx = geometry.CellSizeArray();
    amrex::Real min_cell_size = dx[0];
    for (int d = 1; d < AMREX_SPACEDIM; ++d)
    {
        min_cell_size = amrex::min(min_cell_size, dx[d]);
    }

    amrex::Real inverse_transport_spacing = 1/min_cell_size;
    if (m_flux_relaxed_limited_m1) {
        inverse_transport_spacing = 0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            inverse_transport_spacing += 1/geometry.CellSize(d);
        }
    }
    amrex::Real const requested_steps =
        amrex::max(1._rt, std::ceil(2 * PhysConst::c * dt * inverse_transport_spacing));
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(requested_steps) &&
                                         requested_steps <= m_max_transport_substeps,
                                     "flux_relaxed photon microstep budget exceeded.");
    int const base_steps = static_cast<int>(requested_steps);

    std::unique_ptr<ParticleImpulseMaterial> accepted_material;
    std::unique_ptr<PhotonPacketTransaction> accepted_photons;
    amrex::MultiFab accepted_radiation;
    amrex::MultiFab accepted_actual_impulse_field;
    amrex::MultiFab accepted_actual_work_field;
    amrex::GpuArray<amrex::Real, 3> accepted_requested_impulse{};
    amrex::GpuArray<amrex::Real, 3> accepted_actual_impulse{};
    amrex::GpuArray<amrex::Real, 3> accepted_momentum_carry_change{};
    amrex::Real accepted_requested_work = 0.0_rt;
    amrex::Real accepted_actual_work = 0.0_rt;
    amrex::Real accepted_energy_carry_change = 0.0_rt;
    amrex::Real accepted_energy_residual = 0.0_rt;
    amrex::Real accepted_momentum_residual = 0.0_rt;
    int accepted_segments = 0;
    int rejected_attempts = 0;
    int total_iterations = 0;

    for (int subdivision = 0; subdivision <= m_flux_relaxed_max_subdivisions; ++subdivision)
    {
        int const refinement = 1 << subdivision;
        if (base_steps > m_max_transport_substeps / refinement)
        {
            break;
        }
        int const segments = base_steps * refinement;
        amrex::Real const segment_dt = dt / segments;
        amrex::MultiFab trial_radiation(initial_radiation.boxArray(),
                                        initial_radiation.DistributionMap(), 4, 1);
        amrex::MultiFab::Copy(trial_radiation, initial_radiation, 0, 0, 4, amrex::IntVect(1));

        auto material = std::make_unique<ParticleImpulseMaterial>(particles, m_momentum_species);
        auto photon_transaction = std::make_unique<PhotonPacketTransaction>(live_photons);
        auto const& live_temperature = *fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        amrex::MultiFab source_temperature(live_temperature.boxArray(),
            live_temperature.DistributionMap(), 1, live_temperature.nGrowVect());
        amrex::MultiFab::Copy(source_temperature, live_temperature, 0, 0, 1,
                              live_temperature.nGrowVect());
        amrex::MultiFab source_heat(trial_radiation.boxArray(),
            trial_radiation.DistributionMap(), 1, 1);
        source_heat.setVal(0);
        amrex::GpuArray<amrex::Real,4> m1_pending_change{};
        amrex::MultiFab zero_impulse(trial_radiation.boxArray(), trial_radiation.DistributionMap(),
                                     3, 0);
        zero_impulse.setVal(0.0_rt);
        ParticleImpulseTransaction initial_velocity;
        bool attempt_valid = initial_velocity.Stage(*material, "diffusion_0", zero_impulse,
                                                    /*need_work_velocity=*/true);
        amrex::MultiFab frozen_beta(trial_radiation.boxArray(), trial_radiation.DistributionMap(),
                                    3, 1);
        frozen_beta.setVal(0.0_rt);
        if (attempt_valid)
        {
            amrex::MultiFab::Copy(frozen_beta, initial_velocity.WorkVelocity(), 0, 0, 3, 0);
            frozen_beta.mult(1.0_rt / PhysConst::c);
            frozen_beta.FillBoundary(geometry.periodicity());
        }

        amrex::MultiFab requested_impulse(trial_radiation.boxArray(),
                                          trial_radiation.DistributionMap(), 3, 0);
        amrex::MultiFab requested_work(trial_radiation.boxArray(),
                                       trial_radiation.DistributionMap(), 1, 0);
        amrex::MultiFab actual_impulse(trial_radiation.boxArray(),
                                       trial_radiation.DistributionMap(), 3, 0);
        amrex::MultiFab actual_work(trial_radiation.boxArray(), trial_radiation.DistributionMap(),
                                    1, 0);
        amrex::MultiFab momentum_carry_change(trial_radiation.boxArray(),
                                              trial_radiation.DistributionMap(), 3, 0);
        amrex::MultiFab energy_carry_change(trial_radiation.boxArray(),
                                            trial_radiation.DistributionMap(), 1, 0);
        requested_impulse.setVal(0.0_rt);
        requested_work.setVal(0.0_rt);
        actual_impulse.setVal(0.0_rt);
        actual_work.setVal(0.0_rt);
        momentum_carry_change.setVal(0.0_rt);
        energy_carry_change.setVal(0.0_rt);

        int completed_segments = 0;
        for (int segment = 0; attempt_valid && segment < segments; ++segment)
        {
            if (m_flux_relaxed_limited_m1) {
                ParticleImpulseTransaction current_velocity;
                attempt_valid = current_velocity.Stage(*material, "diffusion_0", zero_impulse, true);
                if (!attempt_valid) { break; }
                amrex::MultiFab::Copy(frozen_beta, current_velocity.WorkVelocity(), 0, 0, 3, 0);
                frozen_beta.mult(1/PhysConst::c);
                frozen_beta.FillBoundary(geometry.periodicity());
            }
            attempt_valid = AdvanceCandidatePhotons(
                photon_transaction->Particles(), trial_radiation, scattering, frozen_beta, geometry,
                segment_dt, m_minimum_diffusion_optical_depth);
            if (!attempt_valid)
            {
                break;
            }

            if (m_flux_relaxed_limited_m1) {
                auto const transported = warpx::radiation::TryAdvanceLimitedMomentTransport(
                    trial_radiation, frozen_beta, scattering, geometry, segment_dt);
                attempt_valid = transported.valid;
                if (!attempt_valid) {
                    if (m_implicit_diffusion_options.verbosity > 0) {
                        amrex::Print() << "Limited-M1 transport rejected segment=" << segment
                            << " failure=" << static_cast<int>(transported.failure) << '\n';
                    }
                    break;
                }
                warpx::radiation::CoupledMomentCallbacks callbacks;
                callbacks.coefficients = [&] (amrex::MultiFab const& temperature,
                    amrex::MultiFab& absorption, amrex::MultiFab& chi,
                    amrex::MultiFab& equilibrium, amrex::Real time) {
                    amrex::ignore_unused(temperature, time);
                    absorption.setVal(0);
                    equilibrium.setVal(0);
                    amrex::MultiFab::Copy(chi, scattering, 0, 0, 1, 0);
                    chi.FillBoundary(geometry.periodicity());
                };
                // Elastic coherent scattering has exactly zero caloric source.
                // Reject unexpected heating rather than hiding it in electrons.
                callbacks.material_response = [] (amrex::MultiFab& temperature,
                                                    amrex::MultiFab& heat) {
                    amrex::ignore_unused(temperature);
                    return heat.norm0(0) == 0 ? 0._rt :
                        std::numeric_limits<amrex::Real>::quiet_NaN();
                };
                warpx::radiation::CoupledMomentOptions source_options;
                source_options.max_iterations = m_flux_relaxed_max_iterations;
                source_options.tolerance = m_flux_relaxed_tolerance;
                source_options.energy_tolerance = m_flux_relaxed_energy_tolerance;
                source_options.maximum_beta = m_flux_relaxed_maximum_beta;
                source_options.verbose = m_implicit_diffusion_options.verbosity > 1;
                source_options.spatial_transport = false;
                source_options.particle_assignment =
                    warpx::radiation::ParticleImpulseAssignment::NearestCell;
                warpx::radiation::CoupledMomentExchange exchange;
                auto const source = warpx::radiation::TryAdvanceCoupledMomentSource(
                    *material, "diffusion_0", trial_radiation, source_temperature, source_heat,
                    geometry, current_time + segment*segment_dt, segment_dt,
                    source_options, callbacks, &exchange);
                total_iterations += source.iterations;
                attempt_valid = source.valid;
                if (!attempt_valid) {
                    if (m_implicit_diffusion_options.verbosity > 0) {
                        amrex::Print() << "Limited-M1 source rejected segment=" << segment
                            << " failure=" << static_cast<int>(source.failure)
                            << " work_residual=" << source.work_residual << '\n';
                    }
                    break;
                }
                amrex::MultiFab::Add(actual_impulse, exchange.momentum, 0, 0, 3, 0);
                amrex::MultiFab::Add(actual_work, exchange.kinetic_work, 0, 0, 1, 0);
                amrex::MultiFab::Add(requested_impulse, exchange.momentum, 0, 0, 3, 0);
                amrex::MultiFab::Add(requested_work, exchange.kinetic_work, 0, 0, 1, 0);
                for (int d = 0; d < 4; ++d) { m1_pending_change[d] += exchange.pending_change[d]; }
                ++completed_segments;
                continue;
            }

            std::unique_ptr<ParticleImpulseTransaction> staged_transaction;
            auto const work_callback = [&] (amrex::MultiFab const& impulse, amrex::MultiFab& work)
            {
                auto next = std::make_unique<ParticleImpulseTransaction>();
                if (!next->Stage(*material, "diffusion_0", impulse,
                                 /*need_work_velocity=*/true))
                {
                    return false;
                }
                amrex::MultiFab::Copy(work, next->RequestedWork(), 0, 0, 1, 0);
                staged_transaction = std::move(next);
                return true;
            };
            warpx::radiation::FluxRelaxedDiffusionOptions options;
            options.max_iterations = m_flux_relaxed_max_iterations;
            options.tolerance = m_flux_relaxed_tolerance;
            options.energy_tolerance = m_flux_relaxed_energy_tolerance;
            options.maximum_beta = m_flux_relaxed_maximum_beta;
            options.verbose = m_implicit_diffusion_options.verbosity > 1;
            auto result = warpx::radiation::TryAdvanceFluxRelaxedDiffusion(
                trial_radiation, scattering, frozen_beta, geometry, segment_dt, options,
                work_callback);
            total_iterations += result.iterations;
            attempt_valid = result.valid && staged_transaction != nullptr;
            if (!attempt_valid)
            {
                if (options.verbose)
                {
                    amrex::Print()
                        << "Flux-relaxed rejected segment=" << segment
                        << " subdivisions=" << subdivision
                        << " failure_code=" << static_cast<int>(result.failure)
                        << " energy_residual=" << result.energy_equation_residual
                        << " momentum_residual=" << result.momentum_equation_residual << '\n';
                }
                break;
            }

            amrex::MultiFab::Add(requested_impulse, result.requested_impulse, 0, 0, 3, 0);
            amrex::MultiFab::Add(requested_work, result.requested_work, 0, 0, 1, 0);
            amrex::MultiFab::Add(actual_impulse, staged_transaction->ActualImpulse(), 0, 0, 3, 0);
            amrex::MultiFab::Add(actual_work, staged_transaction->ActualWork(), 0, 0, 1, 0);
            amrex::MultiFab::Add(momentum_carry_change, staged_transaction->MomentumCarryChange(),
                                 0, 0, 3, 0);
            amrex::MultiFab::Add(energy_carry_change, staged_transaction->EnergyCarryChange(), 0, 0,
                                 1, 0);
            // This commits only the private kinetic copy. No trial transaction
            // is committed during the backend's Picard iterations above.
            staged_transaction->Commit();
            ++completed_segments;
        }

        EmissionPlan plan;
        if (attempt_valid)
        {
            attempt_valid = BuildThinCellEmissionPlan(
                trial_radiation, scattering, geometry, min_cell_size,
                m_minimum_diffusion_optical_depth, m_particle_conversion_energy_threshold,
                m_emission_photon_energy, plan);
            amrex::ParallelDescriptor::ReduceBoolAnd(attempt_valid);
        }
        if (attempt_valid)
        {
            attempt_valid = StageEmissionPlan(*photon_transaction, plan);
        }
        if (attempt_valid)
        {
            attempt_valid = photon_transaction->ValidateLive();
        }
        if (attempt_valid)
        {
            amrex::GpuArray<amrex::Real, 4> final_photons{};
            attempt_valid = PhotonInventory(photon_transaction->Particles(), final_photons);
            amrex::GpuArray<amrex::Real, 4> final_radiation{};
            amrex::GpuArray<amrex::Real, 4> initial_state{};
            for (int d = 0; d < 4; ++d)
            {
                final_radiation[d] =
                    SumComponent(trial_radiation, d) + final_photons[d];
                initial_state[d] = SumComponent(initial_radiation, d) + initial_photons[d];
            }
            amrex::GpuArray<amrex::Real, 3> impulse_sum{};
            amrex::GpuArray<amrex::Real, 3> actual_impulse_sum{};
            amrex::GpuArray<amrex::Real, 3> momentum_carry_sum{};
            for (int d = 0; d < 3; ++d)
            {
                impulse_sum[d] = SumComponent(requested_impulse, d) + m1_pending_change[d];
                actual_impulse_sum[d] = SumComponent(actual_impulse, d);
                momentum_carry_sum[d] = SumComponent(momentum_carry_change, d) + m1_pending_change[d];
            }
            amrex::Real const work_sum = SumComponent(requested_work, 0) + m1_pending_change[3];
            amrex::Real const kinetic_work_sum = SumComponent(actual_work, 0);
            amrex::Real const carry_energy_sum = SumComponent(energy_carry_change, 0) + m1_pending_change[3];
            amrex::Real const energy_residual = initial_state[0] - final_radiation[0] - work_sum;
            amrex::Real momentum_squared = 0.0_rt;
            amrex::Real momentum_scale = 0.0_rt;
            amrex::Real energy_scale = amrex::Math::abs(initial_state[0]) +
                                       amrex::Math::abs(final_radiation[0]) +
                                       amrex::Math::abs(work_sum);
            for (int d = 0; d < 3; ++d)
            {
                amrex::Real const residual_component =
                    (final_radiation[d + 1] - initial_state[d + 1]) / PhysConst::c + impulse_sum[d];
                momentum_squared += residual_component * residual_component;
                momentum_scale += amrex::Math::abs(initial_state[d + 1]) +
                                  amrex::Math::abs(final_radiation[d + 1]) +
                                  PhysConst::c * amrex::Math::abs(impulse_sum[d]);
            }
            amrex::Real const momentum_residual = std::sqrt(momentum_squared);
            attempt_valid = attempt_valid && std::isfinite(energy_residual) &&
                            std::isfinite(momentum_residual) &&
                            std::abs(energy_residual) <= 1.0e-10_rt * energy_scale &&
                            momentum_residual <= 1.0e-10_rt * energy_scale / PhysConst::c;
            if (attempt_valid)
            {
                accepted_requested_work = work_sum;
                accepted_actual_work = kinetic_work_sum;
                accepted_energy_carry_change = carry_energy_sum;
                accepted_energy_residual = energy_residual;
                accepted_momentum_residual = momentum_residual;
                accepted_requested_impulse = impulse_sum;
                accepted_actual_impulse = actual_impulse_sum;
                accepted_momentum_carry_change = momentum_carry_sum;
                accepted_radiation.define(trial_radiation.boxArray(),
                                          trial_radiation.DistributionMap(), 4, 1);
                amrex::MultiFab::Copy(accepted_radiation, trial_radiation, 0, 0, 4, 0);
                accepted_actual_impulse_field = std::move(actual_impulse);
                accepted_actual_work_field = std::move(actual_work);
                accepted_material = std::move(material);
                accepted_photons = std::move(photon_transaction);
                accepted_segments = completed_segments;
            }
        }
        if (!attempt_valid)
        {
            ++rejected_attempts;
        }
        else
        {
            break;
        }
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        accepted_material != nullptr && accepted_photons != nullptr && accepted_segments > 0,
        "No flux-relaxed packet/FLD attempt passed photon, field, material, "
        "work, and closed energy/momentum validation; live state is unchanged.");

    // All physics and arithmetic gates are collective and have passed. No
    // physics rejection is allowed after the first live commit below.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        accepted_material->Commit(),
        "Live material changed before flux-relaxed commit; photons/fields unchanged.");
    accepted_photons->Commit();
    amrex::MultiFab::Copy(energy, accepted_radiation, 0, 0, 1, 0);
    for (int d = 0; d < 3; ++d)
    {
        amrex::MultiFab::Copy(*q_fields[d], accepted_radiation, d + 1, 0, 1, 0);
        q_fields[d]->FillBoundary(geometry.periodicity());
    }
    energy.FillBoundary(geometry.periodicity());

    auto& material_energy = *fields.get(FieldType::radiation_material_energy, 0);
    auto& material_momentum = *fields.get(FieldType::radiation_material_momentum, 0);
    auto& material_kinetic = *fields.get(FieldType::radiation_material_kinetic_energy, 0);
    material_energy.setVal(0.0_rt);
    material_momentum.setVal(0.0_rt);
    material_kinetic.setVal(0.0_rt);
    amrex::MultiFab::Copy(material_momentum, accepted_actual_impulse_field, 0, 0, 3, 0);
    amrex::MultiFab::Copy(material_kinetic, accepted_actual_work_field, 0, 0, 1, 0);

    m_last_numerical_energy_residual = accepted_energy_residual;
    m_cumulative_numerical_energy_residual += accepted_energy_residual;
    m_last_material_carry_energy_change = accepted_energy_carry_change;
    m_last_coupled_solve = {total_iterations, accepted_segments,       rejected_attempts, 0,
                            0.0_rt,           accepted_energy_residual};
    if (m_implicit_diffusion_options.verbosity > 0)
    {
        amrex::Print() << (m_flux_relaxed_limited_m1 ? "Limited-M1 packet/vector" : "Flux-relaxed packet/FLD")
                       << " accepted_substeps=" << accepted_segments
                       << " rejected_attempts=" << rejected_attempts
                       << " iterations=" << total_iterations << " ion_impulse=["
                       << accepted_actual_impulse[0] << ", " << accepted_actual_impulse[1] << ", "
                       << accepted_actual_impulse[2] << "]"
                       << " requested_impulse=[" << accepted_requested_impulse[0] << ", "
                       << accepted_requested_impulse[1] << ", " << accepted_requested_impulse[2]
                       << "]"
                       << " momentum_carry_change=[" << accepted_momentum_carry_change[0] << ", "
                       << accepted_momentum_carry_change[1] << ", "
                       << accepted_momentum_carry_change[2] << "]"
                       << " requested_work=" << accepted_requested_work
                       << " actual_kinetic_work=" << accepted_actual_work
                       << " energy_carry_change=" << accepted_energy_carry_change
                       << " raw_energy_residual=" << accepted_energy_residual
                       << " raw_momentum_residual=" << accepted_momentum_residual << '\n';
    }

    // Radiation q is a distinct checkpointed state, not material impulse carry.
    // The native particle carry attributes and requested/actual ledgers above
    // remain separately observable for independent conservation checks.
}

void
RadiationTransport::FillFrozenCoherentOpacity (
    ablastr::fields::MultiFabRegister& fields, amrex::Geometry const& geometry,
    amrex::Real current_time, amrex::MultiFab& scattering) const
{
    auto& live_temperature = *fields.get(FieldType::hybrid_electron_temperature_fp, 0);
    auto& live_density = *fields.get(FieldType::rho_fp, 0);
    amrex::MultiFab temperature(live_temperature.boxArray(), live_temperature.DistributionMap(), 1,
                                live_temperature.nGrowVect());
    amrex::MultiFab density(live_density.boxArray(), live_density.DistributionMap(),
                            live_density.nComp(), live_density.nGrowVect());
    amrex::MultiFab::Copy(temperature, live_temperature, 0, 0, 1, temperature.nGrowVect());
    amrex::MultiFab::Copy(density, live_density, 0, 0, density.nComp(), density.nGrowVect());
    temperature.FillBoundary(geometry.periodicity());
    density.FillBoundary(geometry.periodicity());

    auto const eos = m_hybrid_model->electronThermodynamicsExecutor();
    amrex::GpuArray<amrex::MultiFab const*, ElectronThermodynamicsExecutor::max_materials>
        material_fields{};
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> material_snapshots(eos.m_num_materials);
    for (int material = 0; material < eos.m_num_materials; ++material)
    {
        auto& live_material = *fields.get(
            "ni_charge_fp_" + m_hybrid_model->electronThermodynamicsMaterialSpeciesName(material),
            0);
        material_snapshots[material] = std::make_unique<amrex::MultiFab>(
            live_material.boxArray(), live_material.DistributionMap(), live_material.nComp(),
            live_material.nGrowVect());
        amrex::MultiFab::Copy(*material_snapshots[material], live_material, 0, 0,
                              live_material.nComp(), live_material.nGrowVect());
        material_snapshots[material]->FillBoundary(geometry.periodicity());
        material_fields[material] = material_snapshots[material].get();
    }

    OpacityEvaluator<1> const planck{false,
                                     0,
                                     {},
                                     {},
                                     {},
                                     {},
                                     m_use_planck_table,
                                     false,
                                     m_planck_table.executor(),
                                     m_spectral_planck_table.executor(),
                                     m_planck_absorption_coefficient};
    OpacityEvaluator<1> const transport{false,
                                        0,
                                        {},
                                        {},
                                        {},
                                        {},
                                        m_use_rosseland_table,
                                        false,
                                        m_rosseland_table.executor(),
                                        m_spectral_rosseland_table.executor(),
                                        m_rosseland_transport_coefficient};
    amrex::MultiFab true_absorption(scattering.boxArray(), scattering.DistributionMap(), 1, 1);
    amrex::MultiFab emission(scattering.boxArray(), scattering.DistributionMap(), 1, 1);
    FillCoupledHybridCoefficients(density, temperature, scattering, true_absorption, emission,
                                  geometry, eos, m_minimum_electron_density,
                                  m_hybrid_model->transportsElectronInternalEnergyAtRawDensity()
                                      ? 0.0_rt
                                      : m_hybrid_model->electronDensityFloor(),
                                  planck, transport, m_energy_groups, current_time,
                                  material_fields);

    ValidateCoherentOpacity(density, scattering, true_absorption);
    scattering.FillBoundary(geometry.periodicity());
}
