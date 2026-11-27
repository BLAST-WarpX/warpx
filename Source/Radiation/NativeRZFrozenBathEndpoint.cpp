/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativeRZFrozenBathEndpoint.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "HybridCellMaterialState.H"
#include "Particles/MultiParticleContainer.H"
#include "RZMomentGeometry.H"
#include "WarpX.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <limits>

namespace warpx::radiation
{
namespace
{
#if defined(WARPX_DIM_RZ)
amrex::MultiFab
Host (amrex::MultiFab const& source)
{
    amrex::MultiFab target(source.boxArray(), source.DistributionMap(), source.nComp(),
                           source.nGrowVect(), amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::dtoh_memcpy(target, source);
    amrex::Gpu::streamSynchronize();
    return target;
}
#endif
}

NativeRZFrozenBathEndpointResult
TryNativeRZFrozenBathEndpoint (
    WarpXParticleContainer& live, PhotonPacketTransaction const& endpoint,
    ParticleImpulseMaterial& material, std::string const& path,
    amrex::MultiFab const& temperature, amrex::MultiFab const& opacity,
    amrex::MultiFab const& rate, HybridPICModel const& model,
    amrex::Geometry const& geometry, amrex::Real dt,
    NativePacketThermalEventOptions options, amrex::Real minimum_density,
    NativeRZCaloricOptions const& caloric_options)
{
    NativeRZFrozenBathEndpointResult rejected;
    rejected.reason = "unsupported native RZ frozen-bath thermal endpoint";
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(live, endpoint, material, path, temperature, opacity, rate,
                         model, geometry, dt, options, minimum_density, caloric_options);
    return rejected;
#else
    if (!endpoint.ValidateSnapshotFor(live))
    {
        rejected.reason = "invalid private RZ photon endpoint provenance";
        return rejected;
    }
    auto& simulation = WarpX::GetInstance();
    auto const& native_geometry = simulation.Geom(0);
    auto const eos = model.electronThermodynamicsExecutor();
    auto const& live_density = *simulation.m_fields.get(warpx::fields::FieldType::rho_fp, 0);
    auto const& boxes = endpoint.Particles().ParticleBoxArray(0);
    auto const& mapping = endpoint.Particles().ParticleDistributionMap(0);
    int option_min = caloric_options.reflecting_axial_trial ? 1 : 0;
    int option_max = option_min;
    amrex::ParallelDescriptor::ReduceIntMin(option_min);
    amrex::ParallelDescriptor::ReduceIntMax(option_max);
    bool axial = geometry.isPeriodic(1);
#if !defined(AMREX_USE_GPU)
    axial = axial || (caloric_options.reflecting_axial_trial &&
        WarpX::field_boundary_lo[1] == FieldBoundaryType::PEC &&
        WarpX::field_boundary_hi[1] == FieldBoundaryType::PEC &&
        WarpX::particle_boundary_lo[1] == ParticleBoundaryType::Reflecting &&
        WarpX::particle_boundary_hi[1] == ParticleBoundaryType::Reflecting);
#endif
    bool valid = !EB::enabled() && simulation.finestLevel() == 0 &&
        WarpX::do_moving_window == 0 && !simulation.DoFluidSpecies() &&
        !simulation.GetPartContainer().hasHybridIonization() &&
        model.solvesElectronEnergyEquation() &&
        model.transportsElectronInternalEnergyAtRawDensity() &&
        (eos.isIdealGas() || eos.isFixedChargeLatentEnergy()) && eos.m_num_materials == 0 &&
        geometry.Domain() == native_geometry.Domain() &&
        geometry.ProbLo(0) == 0 && !geometry.isPeriodic(0) && axial && option_min == option_max &&
        WarpX::n_rz_azimuthal_modes == 1 &&
        WarpX::field_boundary_lo[0] == FieldBoundaryType::None &&
        WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC &&
        WarpX::particle_boundary_lo[0] == ParticleBoundaryType::None &&
        WarpX::particle_boundary_hi[0] == ParticleBoundaryType::Reflecting &&
        std::isfinite(dt) && dt > 0 && std::isfinite(minimum_density) &&
        minimum_density >= 0 && !path.empty() &&
        opacity.nComp() == 1 && rate.nComp() == 3 &&
        opacity.boxArray() == boxes && rate.boxArray() == boxes &&
        opacity.DistributionMap() == mapping && rate.DistributionMap() == mapping &&
        temperature.nComp() == 1 &&
        temperature.boxArray() == amrex::convert(boxes, amrex::IntVect(1)) &&
        temperature.DistributionMap() == mapping && temperature.nGrowVect().allGE(1) &&
        live_density.nComp() >= 1 && live_density.boxArray() == temperature.boxArray() &&
        live_density.DistributionMap() == mapping && live_density.nGrowVect().allGE(1);
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        valid = valid && geometry.ProbLo(d) == native_geometry.ProbLo(d) &&
                geometry.ProbHi(d) == native_geometry.ProbHi(d) &&
                geometry.isPeriodic(d) == native_geometry.isPeriodic(d);
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { return rejected; }

    options.rz_angular_conjugate = true;
    options.rz_reflecting_axial_frozen_bath = caloric_options.reflecting_axial_trial;
    amrex::iMultiFab kinetic(boxes, mapping, 1, 0);
    kinetic.setVal(1);
    NativePacketThermalEventReplay replay;
    if (!replay.Prepare(endpoint.Particles(), live, kinetic, opacity, rate, geometry,
                        dt, options, rejected.reason))
    {
        return rejected;
    }
    amrex::MultiFab zero(boxes, mapping, 3, 0);
    zero.setVal(0);
    ParticleImpulseTransaction initial;
    if (!initial.Stage(material, path, zero, true,
                       ParticleImpulseAssignment::RZAngularConservative) ||
        initial.MaximumBetaSquared() > 1.e-6)
    {
        rejected.reason = "invalid or high-beta initial RZ material endpoint";
        return rejected;
    }

    amrex::MultiFab frozen_density(live_density.boxArray(), live_density.DistributionMap(),
                                  1, live_density.nGrowVect());
    amrex::MultiFab::Copy(frozen_density, live_density, 0, 0, 1, live_density.nGrowVect());
    frozen_density.FillBoundary(geometry.periodicity());
    auto density = Host(frozen_density);
    amrex::MultiFab frozen_temperature(temperature.boxArray(), mapping, 1,
                                      temperature.nGrowVect());
    amrex::MultiFab::Copy(frozen_temperature, temperature, 0, 0, 1,
                          temperature.nGrowVect());
    frozen_temperature.FillBoundary(geometry.periodicity());
    auto old_temperature = Host(frozen_temperature);
    amrex::MultiFab cell_temperature(boxes, mapping, 1, 0,
                                    amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    auto const spacing = geometry.CellSizeArray();
    auto const lower = geometry.ProbLoArray();
    auto const domain_lo = amrex::lbound(geometry.Domain());
    for (amrex::MFIter it(cell_temperature); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            amrex::Real capacity = 0, weighted = 0;
            for (int corner = 0; corner < 4; ++corner)
            {
                auto node = index;
                node[0] += corner & 1;
                node[1] += (corner >> 1) & 1;
                auto const charge = density[it](node);
                valid = valid && std::isfinite(charge) &&
                        charge > PhysConst::q_e * minimum_density;
                if (!valid) { continue; }
                auto const t = old_temperature[it](node);
                auto const state = eos.stateFromChargeDensityTemperature(charge, t);
                auto const corner_volume = detail::HybridCellCornerVolume(
                    index[0], index[1], 0, corner & 1, (corner >> 1) & 1, 0,
                    lower, spacing, domain_lo);
                auto const weight = state.heat_capacity_density * corner_volume;
                valid = valid && std::isfinite(t + weight) && weight > 0 &&
                        t >= eos.minimumTemperature() && t <= eos.maximumTemperature();
                capacity += weight;
                weighted += weight * t;
            }
            valid = valid && capacity > 0 && std::isfinite(weighted);
            cell_temperature[it](index) = valid ? weighted / capacity : 0;
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        rejected.reason = "invalid frozen RZ electron bath";
        return rejected;
    }
    amrex::MultiFab source_temperature(boxes, mapping, 1, 0);
    amrex::htod_memcpy(source_temperature, cell_temperature);
    amrex::Gpu::streamSynchronize();
    auto result = NativeRZFrozenBathEndpointResult{};
    result.photons = std::make_unique<PhotonPacketTransaction>(live, endpoint);
    result.packet = replay.Replay(source_temperature, initial.WorkVelocity(), *result.photons);
    if (!result.packet.valid)
    {
        rejected.reason = result.packet.reason;
        return rejected;
    }
    if (!result.photons->ValidateSnapshotFor(live))
    {
        rejected.reason = "invalid private RZ thermal photon candidate provenance";
        return rejected;
    }
    auto debit = Host(result.packet.radiation_debit);
    amrex::MultiFab impulse(boxes, mapping, 3, 0,
                            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    for (amrex::MFIter it(impulse); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            for (int d = 0; d < 3; ++d)
            {
                impulse[it](cell(), d) = debit[it](cell(), d + 1) / PhysConst::c;
            }
        }
    }
    amrex::MultiFab device_impulse(boxes, mapping, 3, 0);
    amrex::htod_memcpy(device_impulse, impulse);
    result.material_stage = std::make_unique<ParticleImpulseTransaction>();
    if (!result.material_stage->Stage(material, path, device_impulse, true,
                                      ParticleImpulseAssignment::RZAngularConservative) ||
        result.material_stage->MaximumBetaSquared() > 1.e-6)
    {
        rejected.reason = "invalid or high-beta finite RZ thermal kick";
        return rejected;
    }
    auto work = Host(result.material_stage->ActualWork());
    auto energy_carry = Host(result.material_stage->EnergyCarryChange());
    auto requested_work = Host(result.material_stage->RequestedWork());
    auto work_bound = Host(result.material_stage->WorkAccountingRoundoffBound());
    auto actual_impulse = Host(result.material_stage->ConjugateImpulse());
    auto momentum_carry = Host(result.material_stage->ConjugateMomentumCarryChange());
    result.requested_heat.define(boxes, mapping, 1, 1);
    result.requested_heat.setVal(0);
    auto heat = Host(result.requested_heat);
    auto scale = Host(result.packet.operand_scale);
    amrex::Real requested_total = 0, bound = 0;
    RZMomentMetric const metric(geometry);
    for (amrex::MFIter it(heat); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            auto const request = debit[it](index, 0) - work[it](index) - energy_carry[it](index);
            heat[it](index) = request;
            requested_total += request;
            bound += scale[it](index, 0) + std::abs(work[it](index)) +
                     std::abs(energy_carry[it](index));
            auto const work_defect = requested_work[it](index) - work[it](index) -
                                     energy_carry[it](index);
            valid = valid && std::isfinite(work_defect + work_bound[it](index)) &&
                    std::abs(work_defect) <= work_bound[it](index);
            for (int d = 1; d < 4; ++d)
            {
                auto const closure = debit[it](index, d) / PhysConst::c -
                    actual_impulse[it](index, d - 1) - momentum_carry[it](index, d - 1);
                auto const tolerance = 1.e-10 *
                    (scale[it](index, d) / PhysConst::c +
                     std::abs(actual_impulse[it](index, d - 1)) +
                     std::abs(momentum_carry[it](index, d - 1)));
                valid = valid && std::isfinite(closure) && std::abs(closure) <= tolerance;
            }
            valid = valid && std::isfinite(request) && metric.Volume(index[0]) > 0;
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        rejected.reason = "RZ thermal recoil or energy account rejected";
        return rejected;
    }
    amrex::htod_memcpy(result.requested_heat, heat);
    result.requested_heat.FillBoundary(geometry.periodicity());
    result.nodal_temperature.define(temperature.boxArray(), mapping, 1,
                                    temperature.nGrowVect());
    amrex::MultiFab::Copy(result.nodal_temperature, frozen_temperature, 0, 0, 1,
                          temperature.nGrowVect());
    result.actual_heat.define(boxes, mapping, 1, 1);
    amrex::MultiFab::Copy(result.actual_heat, result.requested_heat, 0, 0, 1, 1);
    auto const residual = model.EvaluateElectronEnergySource(
        0, result.nodal_temperature, result.actual_heat, minimum_density, nullptr,
        nullptr, nullptr, true);
    amrex::ParallelDescriptor::ReduceRealSum(requested_total);
    amrex::ParallelDescriptor::ReduceRealSum(bound);
    auto const allowance = 1.e-9 * std::max(std::abs(requested_total), bound) +
        256 * std::numeric_limits<amrex::Real>::epsilon() * bound;
    valid = std::isfinite(residual) && result.nodal_temperature.is_finite(0, 1, 0) &&
            result.actual_heat.is_finite(0, 1, 0) &&
            std::abs(residual) <= allowance &&
            std::abs(result.actual_heat.sum(0) - requested_total) <= allowance;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        rejected.reason = "unaffordable or unrealized RZ electron cooling/heating: residual=" +
            std::to_string(residual) + " requested=" + std::to_string(requested_total) +
            " realized=" + std::to_string(result.actual_heat.sum(0)) +
            " allowance=" + std::to_string(allowance);
        return rejected;
    }
    result.caloric_residual = residual;
    result.valid = true;
    return result;
#endif
}
} // namespace warpx::radiation
