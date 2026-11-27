/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "RadiationTransport.H"

#include "CellVolume.H"
#include "CoupledMomentSource.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "HybridRadiationCoefficients.H"
#include "MomentFields.H"
#include "RZMomentGeometry.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <string>

using namespace amrex::literals;
using warpx::fields::FieldType;
using warpx::radiation::detail::FillCoupledHybridCoefficients;
using warpx::radiation::detail::MomentFieldName;
using warpx::radiation::detail::OpacityEvaluator;

namespace
{
/** Convert the gray coefficient adapter to physical inverse lengths and
 * cell-integrated equilibrium energy. The selected gray model interprets the
 * transport extinction as absorption plus isotropic coherent scattering.
 * A truly empty cell has no material source; invalid nonempty material remains
 * invalid instead of silently removing its opacity. */
void
ConvertGrayMomentCoefficients (amrex::MultiFab const& density, amrex::MultiFab& absorption,
                              amrex::MultiFab& scattering, amrex::MultiFab& equilibrium,
                              amrex::Geometry const& geometry)
{
    auto const dx = geometry.CellSizeArray();
    auto const lower = geometry.ProbLoArray();
    auto const domain_lo = amrex::lbound(geometry.Domain());
    for (amrex::MFIter iterator(absorption); iterator.isValid(); ++iterator) {
        auto const rho = density.const_array(iterator);
        auto const a = absorption.array(iterator);
        auto const s = scattering.array(iterator);
        auto const b = equilibrium.array(iterator);
        amrex::ParallelFor(iterator.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto const volume = warpx::radiation::CellVolume(i, dx, lower, domain_lo);
            bool empty = true;
            for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner) {
                empty = empty && rho(i + (corner & 1), j + ((corner >> 1) & 1),
                                     k + ((corner >> 2) & 1)) == 0;
            }
            if (empty) {
                a(i, j, k) = 0;
                s(i, j, k) = 0;
                b(i, j, k) = 0;
            } else {
                b(i, j, k) = a(i, j, k) > 0 ? volume * b(i, j, k) / a(i, j, k) : 0;
                a(i, j, k) /= PhysConst::c;
                s(i, j, k) -= a(i, j, k);
            }
        });
    }
}
}

void
RadiationTransport::AdvanceCoupledMoment (MultiParticleContainer& particles,
                                         ablastr::fields::MultiFabRegister& fields,
                                         amrex::Real current_time, amrex::Real dt) const
{
    auto& simulation = WarpX::GetInstance();
    auto const& geometry = simulation.Geom(0);
#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(geometry.ProbLo(0) == 0 && geometry.isPeriodic(1) &&
        !geometry.isPeriodic(0) && WarpX::n_rz_azimuthal_modes == 1 &&
        WarpX::field_boundary_lo[0] == FieldBoundaryType::None &&
        WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC &&
        WarpX::particle_boundary_lo[0] == ParticleBoundaryType::None &&
        WarpX::particle_boundary_hi[0] == ParticleBoundaryType::Reflecting,
        "Meridional RZ coupled_moment requires an axis, a fixed PEC/reflecting radial wall "
        "and periodic axial boundaries.");
#else
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(geometry.Coord() == 0 && geometry.isAllPeriodic(),
        "coupled_moment currently requires periodic Cartesian geometry.");
#endif
    auto const& live_temperature = *fields.get(FieldType::hybrid_electron_temperature_fp, 0);
    auto const& live_density = *fields.get(FieldType::rho_fp, 0);
    auto& energy = *fields.get(FieldType::radiation_diffusion_energy, 0);
    auto& heat = *fields.get(FieldType::radiation_material_energy, 0);
    amrex::MultiFab temperature(live_temperature.boxArray(), live_temperature.DistributionMap(),
                                1, live_temperature.nGrowVect());
    amrex::MultiFab density(live_density.boxArray(), live_density.DistributionMap(),
                            live_density.nComp(), live_density.nGrowVect());
    amrex::MultiFab radiation(energy.boxArray(), energy.DistributionMap(), 4, 1);
    radiation.setVal(0);
    amrex::MultiFab::Copy(temperature, live_temperature, 0, 0, 1, temperature.nGrowVect());
    amrex::MultiFab::Copy(density, live_density, 0, 0, density.nComp(), density.nGrowVect());
    density.FillBoundary(geometry.periodicity());
    temperature.FillBoundary(geometry.periodicity());
    amrex::MultiFab::Copy(radiation, energy, 0, 0, 1, 0);
    for (int d = 0; d < 3; ++d) {
        amrex::MultiFab::Copy(radiation, *fields.get(MomentFieldName(d), 0), 0, d + 1, 1, 0);
    }
    radiation.FillBoundary(geometry.periodicity());
#if defined(WARPX_DIM_RZ)
    warpx::radiation::FillRZMomentGhosts(radiation, geometry, 1);
#endif
    auto const initial_energy = radiation.sum(0);
    auto const eos = m_hybrid_model->electronThermodynamicsExecutor();
    auto const minimum_density = m_minimum_electron_density;
    auto const thermodynamic_floor = m_hybrid_model->transportsElectronInternalEnergyAtRawDensity()
        ? 0.0_rt : m_hybrid_model->electronDensityFloor();
    OpacityEvaluator<1> const planck{false, 0, {}, {}, {}, {}, m_use_planck_table, false,
        m_planck_table.executor(), m_spectral_planck_table.executor(),
        m_planck_absorption_coefficient};
    OpacityEvaluator<1> const transport{false, 0, {}, {}, {}, {}, m_use_rosseland_table, false,
        m_rosseland_table.executor(), m_spectral_rosseland_table.executor(),
        m_rosseland_transport_coefficient};
    amrex::GpuArray<amrex::MultiFab const*, ElectronThermodynamicsExecutor::max_materials>
        material_fields{};
    warpx::radiation::CoupledMomentCallbacks callbacks;
    callbacks.material_response = [&] (amrex::MultiFab& trial, amrex::MultiFab& source) {
        return m_hybrid_model->EvaluateElectronEnergySource(0, trial, source, minimum_density);
    };
    callbacks.material_at_temperature = [&] (amrex::MultiFab& old, amrex::MultiFab& source,
                                             amrex::MultiFab const& prescribed,
                                             amrex::MultiFab& residual) {
        return m_hybrid_model->EvaluateElectronEnergySource(0, old, source, minimum_density,
                                                            nullptr, &prescribed, &residual);
    };
    callbacks.coefficients = [&] (amrex::MultiFab const& trial, amrex::MultiFab& absorption,
                                  amrex::MultiFab& scattering, amrex::MultiFab& equilibrium,
                                  amrex::Real evaluation_time) {
        FillCoupledHybridCoefficients(density, trial, scattering, absorption, equilibrium,
            geometry, eos, minimum_density, thermodynamic_floor, planck, transport,
            m_energy_groups, evaluation_time, material_fields);
        ConvertGrayMomentCoefficients(density, absorption, scattering, equilibrium, geometry);
    };
    warpx::radiation::CoupledMomentIntervalOptions options;
    options.source.spatial_transport = true;
#if defined(WARPX_DIM_RZ)
    options.source.particle_assignment = m_rz_angular_transport
        ? warpx::radiation::ParticleImpulseAssignment::RZAngularConservative
        : warpx::radiation::ParticleImpulseAssignment::NearestCell;
    options.source.transport.rz_angular_transport = m_rz_angular_transport;
    options.source.transport.reflecting_boundaries = true;
#else
    options.source.particle_assignment =
        warpx::radiation::ParticleImpulseAssignment::NativeNodalCellAverage;
#endif
    options.source.max_iterations = m_lte_exchange_max_iterations;
    options.source.tolerance = m_lte_exchange_tolerance;
    options.source.relaxation = m_moment_relaxation;
    options.source.verbose = m_implicit_diffusion_options.verbosity > 1;
    options.max_refinements = m_coupled_max_subdivisions;
    warpx::radiation::CoupledMomentExchange exchange;
    auto const result = warpx::radiation::TryAdvanceCoupledMomentInterval(particles,
        m_momentum_species, "diffusion_0", radiation, temperature, heat, geometry,
        current_time, dt, options, callbacks, &exchange, &m_moment_transport_ledger);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(result.valid,
        "Coupled moving radiation solve failed; no source candidate committed. Failure=" +
        std::to_string(static_cast<int>(result.last_source.failure)));
    amrex::MultiFab::Copy(energy, radiation, 0, 0, 1, 1);
    for (int d = 0; d < 3; ++d) {
        amrex::MultiFab::Copy(*fields.get(MomentFieldName(d), 0), radiation, d + 1, 0, 1, 1);
    }
    m_hybrid_model->CommitElectronTemperature(0, temperature);
    auto& kinetic = *fields.get(FieldType::radiation_material_kinetic_energy, 0);
    auto& momentum = *fields.get(FieldType::radiation_material_momentum, 0);
    kinetic.setVal(0);
    momentum.setVal(0);
    amrex::MultiFab::Copy(kinetic, exchange.kinetic_work, 0, 0, 1, 0);
    amrex::MultiFab::Copy(momentum, exchange.momentum, 0, 0, 3, 0);
    m_last_material_carry_energy_change = result.carry_energy_change;
    m_last_numerical_energy_residual = initial_energy - radiation.sum(0) - heat.sum(0)
        - result.actual_kinetic_work - result.carry_energy_change;
    m_cumulative_numerical_energy_residual += m_last_numerical_energy_residual;
    m_last_coupled_solve = {result.last_source.iterations, result.substeps, result.attempts - 1,
        0, result.last_source.material_residual, result.raw_energy_residual};
    if (m_implicit_diffusion_options.verbosity > 0) {
        amrex::Print() << "Moving radiation time=" << current_time + dt
            << " accepted_substeps=" << result.substeps
            << " rejected_attempts=" << result.attempts - 1
            << " last_source_iterations=" << result.last_source.iterations
            << " source_residual=" << result.last_source.source_residual
            << " work_residual=" << result.last_source.work_residual
            << " raw_energy_residual=" << result.raw_energy_residual << '\n';
    }
}
