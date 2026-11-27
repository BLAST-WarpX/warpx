/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "RadiationTransport.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "PacketDiffusionNativeInterval.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhotonParticleContainer.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_iMultiFab.H>

#include <cmath>
#include <string>

using namespace amrex::literals;
using warpx::fields::FieldType;

void
RadiationTransport::AdvanceMovingDiffusion (
    MultiParticleContainer& particles, ablastr::fields::MultiFabRegister& fields,
    amrex::Real current_time, amrex::Real dt) const
{
    using namespace warpx::radiation;
    auto const& simulation = WarpX::GetInstance();
    auto const& geometry = simulation.Geom(0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        geometry.Coord() == 0 && simulation.finestLevel() == 0 &&
            !EB::enabled() && WarpX::do_moving_window == 0,
        "moving_asymptotic requires Cartesian level zero without EB or a moving window.");
    auto* photons = dynamic_cast<PhotonParticleContainer*>(
        &particles.GetParticleContainerFromName(m_photon_species));
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        photons && photons->TotalNumberOfParticles() == 0,
        "moving_asymptotic requires the configured photon species and no live photons.");
    for (auto const& name : m_momentum_species) {
        auto const& species = particles.GetParticleContainerFromName(name);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(species.getMass()) && species.getMass() > 0,
            "moving_asymptotic requires native positive-mass recoiling ions.");
    }
    auto& energy = *fields.get(FieldType::radiation_diffusion_energy, 0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        energy.nComp() == 1 && energy.ixType().cellCentered() && energy.nGrowVect().allGE(1),
        "moving_asymptotic requires cell-integrated radiation_diffusion_energy with ghosts.");
    amrex::MultiFab scattering(energy.boxArray(), energy.DistributionMap(), 1, 1);
    FillFrozenCoherentOpacity(fields, geometry, current_time, scattering);
    amrex::iMultiFab mask(energy.boxArray(), energy.DistributionMap(), 1, 1);
    mask.setVal(1);
    PacketDiffusionNativeIntervalOptions options;
    options.moving_diffusion = true;
    options.material.max_iterations = m_moving_diffusion_max_iterations;
    options.diffusion.verbosity = m_moving_diffusion_verbosity;
    // Diagnostic inertia is not causal state and must not use M1/P1 q.
    if (m_moving_diffusion_verbosity > 1) {
        amrex::Print() << "Moving scalar inputs E=" << energy.min(0) << ".." << energy.max(0)
                       << " chi=" << scattering.min(0) << ".." << scattering.max(0)
                       << " mask=" << mask.min(0) << ".." << mask.max(0) << '\n';
    }
    auto result = TryAdvancePacketDiffusionNativeInterval(
        particles, *photons, m_momentum_species, "diffusion_0", energy, scattering, mask,
        geometry, dt, nullptr, options);
    if (!result.valid && m_moving_diffusion_verbosity > 0) {
        amrex::Print() << "Moving scalar diffusion rejected: " << result.reason << '\n';
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        result.valid && result.actual_material_impulse_field.ok() &&
            result.actual_material_work_field.ok() &&
            result.material_momentum_carry_change_field.ok(),
        "Bounded moving scalar diffusion rejected; live photons, material and radiation "
        "energy are unchanged: " +
            result.reason);
    auto& heat = *fields.get(FieldType::radiation_material_energy, 0);
    auto& momentum = *fields.get(FieldType::radiation_material_momentum, 0);
    auto& kinetic = *fields.get(FieldType::radiation_material_kinetic_energy, 0);
    heat.setVal(0);
    momentum.setVal(0);
    kinetic.setVal(0);
    amrex::MultiFab::Copy(momentum, result.actual_material_impulse_field, 0, 0, 3, 0);
    amrex::MultiFab::Copy(kinetic, result.actual_material_work_field, 0, 0, 1, 0);
    m_last_numerical_energy_residual = result.raw_energy_residual;
    m_cumulative_numerical_energy_residual += result.raw_energy_residual;
    m_last_material_carry_energy_change = result.material_energy_carry_change;
    m_last_diffusion_boundary_momentum_loss = result.radiation_wall_momentum;
    m_last_coupled_solve = {result.work_iterations, 1, 0, 0, 0, result.raw_energy_residual};
    if (m_moving_diffusion_verbosity > 0) {
        amrex::Print() << "Moving scalar diffusion accepted iterations="
                       << result.work_iterations << " actual_work=" << result.actual_material_work
                       << " energy_carry_change=" << result.material_energy_carry_change
                       << " raw_energy_residual=" << result.raw_energy_residual << '\n';
    }
}
