/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativeRZOwnedNodalCaloric.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/OwnedNodalCaloricDifference.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcMetricTransport.H"
#include "HybridCellMaterialState.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace warpx::radiation
{
using namespace amrex::literals;

namespace
{
#if defined(WARPX_DIM_RZ)
bool
Collective (bool valid)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid;
}

amrex::MultiFab
Copy (amrex::MultiFab const& field, bool host = false)
{
    amrex::MFInfo info;
    if (host) { info.SetArena(amrex::The_Pinned_Arena()); }
    amrex::MultiFab result(field.boxArray(), field.DistributionMap(), field.nComp(),
                           field.nGrowVect(), info);
    if (host)
    {
        amrex::dtoh_memcpy(result, field);
    }
    else
    {
        amrex::MultiFab::Copy(result, field, 0, 0, field.nComp(), field.nGrowVect());
    }
    amrex::Gpu::streamSynchronize();
    return result;
}

bool
Scope (HybridPICModel const& model, NativeRZCaloricOptions const& options)
{
    auto& simulation = WarpX::GetInstance();
    auto const& geometry = simulation.Geom(0);
    auto const eos = model.electronThermodynamicsExecutor();
    int option_min = options.reflecting_axial_trial ? 1 : 0;
    int option_max = option_min;
    amrex::ParallelDescriptor::ReduceIntMin(option_min);
    amrex::ParallelDescriptor::ReduceIntMax(option_max);
    int selected_min = options.single_selected_table_trial, selected_max = selected_min;
    amrex::ParallelDescriptor::ReduceIntMin(selected_min);
    amrex::ParallelDescriptor::ReduceIntMax(selected_max);
    if (selected_min != selected_max) { return false; }
    int vacuum_min = options.single_selected_table_vacuum, vacuum_max = vacuum_min;
    amrex::ParallelDescriptor::ReduceIntMin(vacuum_min);
    amrex::ParallelDescriptor::ReduceIntMax(vacuum_max);
    if (vacuum_min != vacuum_max ||
        (vacuum_min && !options.single_selected_table_trial)) { return false; }
    bool selected = false;
    if (options.single_selected_table_trial)
    {
        selected = options.single_selected_table_vacuum
            ? detail::HasNativeSingleSelectedTableMaterial0OrExactVacuum(model, geometry)
            : detail::HasNativeSingleSelectedTableMaterial0(model, geometry);
    }
    bool axial = geometry.isPeriodic(1);
    bool table = false;
#if defined(WARPX_USE_SINGULARITY_EOS) && !defined(AMREX_USE_GPU)
    table = eos.isSingularitySpiner() && eos.m_num_materials == 1 &&
        ((!eos.m_use_resolved_material_selection && !options.single_selected_table_trial) ||
         selected) && eos.tableStorage() != nullptr &&
        ((geometry.isPeriodic(1) && !options.reflecting_axial_trial) ||
         (options.reflecting_axial_trial && !geometry.isPeriodic(1)));
    if (table)
    {
        auto const& species = simulation.GetPartContainer().GetParticleContainerFromName(
            model.electronThermodynamicsMaterialSpeciesName(0));
        table = !species.HasEvolvingChargeState();
    }
#endif
#if !defined(AMREX_USE_GPU)
    axial = axial || (options.reflecting_axial_trial && !geometry.isPeriodic(1) &&
        WarpX::field_boundary_lo[1] == FieldBoundaryType::PEC &&
        WarpX::field_boundary_hi[1] == FieldBoundaryType::PEC &&
        WarpX::particle_boundary_lo[1] == ParticleBoundaryType::Reflecting &&
        WarpX::particle_boundary_hi[1] == ParticleBoundaryType::Reflecting);
#endif
    bool valid = simulation.finestLevel() == 0 && !EB::enabled() &&
        (!options.single_selected_table_trial || (selected && !options.reflecting_axial_trial)) &&
        WarpX::do_moving_window == 0 && !simulation.DoFluidSpecies() &&
        !simulation.GetPartContainer().hasHybridIonization() &&
        model.solvesElectronEnergyEquation() &&
        model.transportsElectronInternalEnergyAtRawDensity() &&
        (table || ((eos.isIdealGas() || eos.isFixedChargeLatentEnergy()) &&
         eos.m_num_materials == 0 && std::isfinite(eos.m_gamma) && eos.m_gamma > 1)) &&
        std::numeric_limits<amrex::Real>::digits >= 53 &&
        geometry.ProbLo(0) == 0 && !geometry.isPeriodic(0) && axial && option_min == option_max &&
        WarpX::n_rz_azimuthal_modes == 1 &&
        WarpX::field_boundary_lo[0] == FieldBoundaryType::None &&
        WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC &&
        WarpX::particle_boundary_lo[0] == ParticleBoundaryType::None &&
        WarpX::particle_boundary_hi[0] == ParticleBoundaryType::Reflecting;
    return Collective(valid);
}

/** Exact physical FV measure selected by the native RZ electron source. */
amrex::Real
TransportNodeVolume (amrex::IntVect const& node, amrex::Geometry const& geometry,
                     amrex::Real axis_factor)
{
    auto const dr = geometry.CellSize(0), dz = geometry.CellSize(1);
    auto const radius = geometry.ProbLo(0) +
        (node[0] - geometry.Domain().smallEnd(0)) * dr;
    auto volume = warpx::hybrid::cylindricalTransportNodeVolume(
        radius, dr, dz, axis_factor);
    if (node[0] == geometry.Domain().bigEnd(0) + 1) { volume *= 0.5_rt; }
    if (!geometry.isPeriodic(1) &&
        (node[1] == geometry.Domain().smallEnd(1) ||
         node[1] == geometry.Domain().bigEnd(1) + 1))
    {
        volume *= 0.5_rt;
    }
    return volume;
}

/** Physical clipped nodal shell used by the native corner partition. */
amrex::Real
GeometricNodeVolume (amrex::IntVect const& node, amrex::Geometry const& geometry)
{
    auto const dr = geometry.CellSize(0), dz = geometry.CellSize(1);
    auto const radius = geometry.ProbLo(0) +
        (node[0] - geometry.Domain().smallEnd(0)) * dr;
    auto const lo = std::max(geometry.ProbLo(0), radius - 0.5_rt * dr);
    auto const hi = std::min(geometry.ProbHi(0), radius + 0.5_rt * dr);
    auto width = dz;
    if (!geometry.isPeriodic(1) &&
        (node[1] == geometry.Domain().smallEnd(1) ||
         node[1] == geometry.Domain().bigEnd(1) + 1))
    {
        width *= 0.5_rt;
    }
    return MathConst::pi * (hi * hi - lo * lo) * width;
}

amrex::Real
CornerVolume (amrex::IntVect const& cell, int corner,
              amrex::Geometry const& geometry)
{
    return detail::HybridCellCornerVolume(
        cell[0], cell[1], 0, corner & 1, (corner >> 1) & 1, 0,
        geometry.ProbLoArray(), geometry.CellSizeArray(), amrex::lbound(geometry.Domain()));
}

amrex::Real
CopyAllowance (amrex::Real first, amrex::Real second)
{
    return 8 * std::numeric_limits<amrex::Real>::epsilon() *
        std::max({1.0_rt, std::abs(first), std::abs(second)});
}

#endif
} // namespace

NativeRZOwnedNodalCaloricState
PrepareNativeRZOwnedNodalCaloric (
    HybridPICModel const& model, amrex::MultiFab const& frozen_charge,
    amrex::MultiFab const& frozen_temperature, amrex::Real minimum_density,
    NativeRZCaloricOptions const& options)
{
    NativeRZOwnedNodalCaloricState rejected;
    rejected.m_reason = "unsupported native RZ owned-nodal caloric state";
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(model, frozen_charge, frozen_temperature, minimum_density, options);
    return rejected;
#else
    if (!Scope(model, options)) { return rejected; }
    auto& simulation = WarpX::GetInstance();
    auto const& geometry = simulation.Geom(0);
    auto const& native_charge = *simulation.m_fields.get(warpx::fields::FieldType::rho_fp, 0);
    auto const& native_temperature =
        *simulation.m_fields.get(warpx::fields::FieldType::hybrid_electron_temperature_fp, 0);
    bool valid = frozen_charge.nComp() == 1 && frozen_temperature.nComp() == 1 &&
        frozen_charge.ixType() == amrex::IndexType::TheNodeType() &&
        frozen_charge.boxArray() == native_charge.boxArray() &&
        frozen_charge.DistributionMap() == native_charge.DistributionMap() &&
        frozen_temperature.boxArray() == native_temperature.boxArray() &&
        frozen_temperature.DistributionMap() == native_temperature.DistributionMap() &&
        frozen_charge.boxArray() == frozen_temperature.boxArray() &&
        frozen_charge.nGrowVect().allGE(1) && frozen_temperature.nGrowVect().allGE(1) &&
        std::isfinite(minimum_density) && minimum_density >= 0;
    if (!Collective(valid) || !frozen_charge.is_finite(0, 1, 0) ||
        !frozen_temperature.is_finite(0, 1, 0))
    {
        rejected.m_reason = "invalid frozen RZ nodal caloric input";
        return rejected;
    }
    auto const eos = model.electronThermodynamicsExecutor();
    auto const threshold = PhysConst::q_e * minimum_density;
    auto const density_floor = PhysConst::q_e *
        (model.transportsElectronInternalEnergyAtRawDensity() ? 0 : model.electronDensityFloor());
    auto const axis_factor = simulation.verboncoeurAxisCorrection() ? 1.0_rt / 3 : 1.0_rt / 4;

    NativeRZOwnedNodalCaloricState prepared;
    prepared.m_options = options;
    prepared.m_charge = Copy(frozen_charge);
    prepared.m_temperature = Copy(frozen_temperature);
    prepared.m_reference = Copy(frozen_temperature);
    prepared.m_reference.OverrideSync(geometry.periodicity());
    prepared.m_reference.FillBoundary(geometry.periodicity());
    if (eos.isSingularitySpiner())
    {
        detail::NativeTableCornerCaptureOptions capture_options;
        capture_options.minimum_electron_density = minimum_density;
        capture_options.thermodynamic_electron_density_floor = 0;
        capture_options.rz_native_fv = true;
        capture_options.rz_verboncoeur_axis_correction = simulation.verboncoeurAxisCorrection();
        capture_options.rz_reflecting_axial_trial = options.reflecting_axial_trial;
        capture_options.require_single_selected_material0 = options.single_selected_table_trial;
        capture_options.allow_single_selected_vacuum = options.single_selected_table_vacuum;
        auto capture = detail::CaptureNativeCompositeTableCornersFromModel(
            model, frozen_charge, frozen_temperature, geometry, capture_options);
        if (!capture.Valid()) { rejected.m_reason = capture.Reason(); return rejected; }
        prepared.m_table = std::make_unique<detail::NativeCompositeTableCornerCapture>(
            std::move(capture));
        amrex::MultiFab node_host(frozen_charge.boxArray(), frozen_charge.DistributionMap(), 7, 1,
                                 amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
        node_host.setVal(0);
        for (auto const& owner : prepared.m_table->Owners())
        {
            auto& fab = node_host[owner.fab];
            auto const volume = TransportNodeVolume(owner.index, geometry, axis_factor);
            amrex::Real charge = 0, cv = 0, energy = 0;
            for (int c = 0; c < owner.corner_count; ++c)
            {
                auto const& corner = owner.corners[c];
                charge += corner.measure * corner.effective_charge;
                if (corner.raw_charge == 0) { continue; }
                auto const state = eos.stateFromMaterialMassDensitiesTemperature(
                    corner.effective_charge, corner.mass_density, owner.caloric.ReferenceTemperature());
                cv += corner.measure * state.heat_capacity_density;
                energy += corner.measure * state.internal_energy_density;
            }
            fab(owner.index, 0) = charge / volume;
            fab(owner.index, 1) = cv / volume;
            fab(owner.index, 2) = energy / volume;
            fab(owner.index, 3) = owner.caloric.AvailableJoules() / volume;
            fab(owner.index, 4) = volume;
            fab(owner.index, 5) = owner.caloric.OldOffsetJoules() / volume;
            fab(owner.index, 6) = owner.caloric.ReferenceTemperature();
            valid = valid && volume > 0 && std::isfinite(charge + cv + energy + volume);
        }
        if (!Collective(valid)) { rejected.m_reason = "invalid table owner state"; return rejected; }
        prepared.m_nodes.define(frozen_charge.boxArray(), frozen_charge.DistributionMap(), 7, 1);
        amrex::htod_memcpy(prepared.m_nodes, node_host);
        prepared.m_nodes.OverrideSync(geometry.periodicity());
        prepared.m_nodes.FillBoundary(geometry.periodicity());
        prepared.m_owners = prepared.m_reference.OwnerMask(geometry.periodicity());
        prepared.m_valid = true;
        prepared.m_reason.clear();
        return prepared;
    }
    auto canonical_charge = Copy(frozen_charge);
    canonical_charge.OverrideSync(geometry.periodicity());
    canonical_charge.FillBoundary(geometry.periodicity());
    auto const rho = Copy(prepared.m_charge, true);
    auto const old = Copy(prepared.m_temperature, true);
    auto const reference = Copy(prepared.m_reference, true);
    auto const reference_charge = Copy(canonical_charge, true);

    amrex::MultiFab contribution(frozen_charge.boxArray(), frozen_charge.DistributionMap(),
                                 3, 1, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    contribution.setVal(0);
    amrex::Gpu::streamSynchronize();
    for (amrex::MFIter it(rho); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(amrex::enclosedCells(it.validbox())); cell.ok(); ++cell)
        {
            for (int corner = 0; corner < 4; ++corner)
            {
                auto node = cell();
                node[0] += corner & 1;
                node[1] += (corner >> 1) & 1;
                auto const charge = rho[it](node), ref_charge = reference_charge[it](node);
                auto const t = old[it](node), ref_t = reference[it](node);
                valid = valid && std::isfinite(charge + ref_charge + t + ref_t) &&
                    charge >= 0 && ref_charge >= 0 &&
                    (charge == 0 ? ref_charge == 0 && t == ref_t
                                 : charge > threshold && ref_charge > threshold &&
                                       std::abs(t - ref_t) <= CopyAllowance(t, ref_t) &&
                                       t >= eos.minimumTemperature() &&
                                       t <= eos.maximumTemperature() &&
                                       ref_t >= eos.minimumTemperature() &&
                                       ref_t <= eos.maximumTemperature());
                if (!valid) { continue; }
                auto const physical_volume = TransportNodeVolume(node, geometry, axis_factor);
                auto const geometric_volume = GeometricNodeVolume(node, geometry);
                auto const corner_volume = CornerVolume(cell(), corner, geometry) *
                    physical_volume / geometric_volume;
                valid = valid && physical_volume > 0 && geometric_volume > 0 &&
                    corner_volume > 0 && std::isfinite(corner_volume);
                if (!valid) { continue; }
                contribution[it](node, 0) +=
                    (charge == 0 ? 0 : std::max(charge, density_floor)) * corner_volume;
                contribution[it](node, 1) += corner_volume;
                if (charge > 0)
                {
                    contribution[it](node, 2) += corner_volume *
                        warpx::hybrid::CenteredOwnedNodalCaloricDifference(
                            eos, std::max(charge, density_floor), ref_t, t);
                }
            }
        }
    }
    if (!Collective(valid))
    {
        rejected.m_reason = "inconsistent or invalid frozen RZ corner copies";
        return rejected;
    }
    amrex::MultiFab summed(frozen_charge.boxArray(), frozen_charge.DistributionMap(), 3, 1);
    amrex::htod_memcpy(summed, contribution);
    amrex::Gpu::streamSynchronize();
    summed.SumBoundary(geometry.periodicity(), true);
    summed.OverrideSync(geometry.periodicity());
    summed.FillBoundary(geometry.periodicity());
    auto const aggregate = Copy(summed, true);

    amrex::MultiFab node_host(frozen_charge.boxArray(), frozen_charge.DistributionMap(), 7, 1,
                              amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    node_host.setVal(0);
    amrex::Gpu::streamSynchronize();
    for (amrex::MFIter it(node_host); it.isValid(); ++it)
    {
        for (amrex::BoxIterator node(it.validbox()); node.ok(); ++node)
        {
            auto const index = node();
            auto const measure = aggregate[it](index, 1);
            auto const expected = TransportNodeVolume(index, geometry, axis_factor);
            valid = valid && std::isfinite(measure + expected) && expected > 0 &&
                std::abs(measure - expected) <=
                    256 * std::numeric_limits<amrex::Real>::epsilon() * expected;
            if (!valid) { continue; }
            auto const charge = aggregate[it](index, 0) / measure;
            auto const offset = aggregate[it](index, 2) / measure;
            auto const ref_t = reference[it](index);
            valid = valid && std::isfinite(charge + offset + ref_t) && charge >= 0;
            if (!valid) { continue; }
            node_host[it](index, 0) = charge;
            node_host[it](index, 4) = measure;
            node_host[it](index, 5) = offset;
            node_host[it](index, 6) = ref_t;
            if (charge > 0)
            {
                auto const before = eos.stateFromChargeDensityTemperature(charge, ref_t);
                auto const available = warpx::hybrid::CenteredOwnedNodalCaloricDifference(
                    eos, charge, eos.minimumTemperature(), ref_t) + offset;
                valid = valid && std::isfinite(before.heat_capacity_density +
                                                before.internal_energy_density + available) &&
                    before.heat_capacity_density > 0 && available >= 0;
                node_host[it](index, 1) = before.heat_capacity_density;
                node_host[it](index, 2) = before.internal_energy_density;
                node_host[it](index, 3) = available;
            }
            else
            {
                valid = valid && offset == 0;
            }
        }
    }
    if (!Collective(valid))
    {
        rejected.m_reason = "invalid native RZ FV nodal measure or old EOS energy";
        return rejected;
    }
    prepared.m_nodes.define(frozen_charge.boxArray(), frozen_charge.DistributionMap(), 7, 1);
    amrex::htod_memcpy(prepared.m_nodes, node_host);
    amrex::Gpu::streamSynchronize();
    prepared.m_nodes.OverrideSync(geometry.periodicity());
    prepared.m_nodes.FillBoundary(geometry.periodicity());
    prepared.m_owners = prepared.m_reference.OwnerMask(geometry.periodicity());
    prepared.m_valid = true;
    prepared.m_reason.clear();
    return prepared;
#endif
}

NativeRZOwnedNodalCaloricResult
EvaluateNativeRZOwnedNodalCaloric (
    HybridPICModel const& model, NativeRZOwnedNodalCaloricState const& old,
    amrex::MultiFab const& owner_heat_joules,
    amrex::MultiFab const* prescribed_temperature)
{
    NativeRZOwnedNodalCaloricResult rejected;
    rejected.reason = "unsupported native RZ owned-nodal caloric evaluation";
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(model, old, owner_heat_joules, prescribed_temperature);
    return rejected;
#else
    if (!Collective(old.m_valid) || !Scope(model, old.m_options)) { return rejected; }
    auto const& geometry = WarpX::GetInstance().Geom(0);
    auto same_layout = [&] (amrex::MultiFab const& field)
    {
        return field.nComp() == 1 && field.boxArray() == old.m_temperature.boxArray() &&
            field.DistributionMap() == old.m_temperature.DistributionMap();
    };
    bool valid = same_layout(owner_heat_joules) &&
        (!prescribed_temperature || same_layout(*prescribed_temperature));
    if (!Collective(valid) || !owner_heat_joules.is_finite(0, 1, 0) ||
        (prescribed_temperature && !prescribed_temperature->is_finite(0, 1, 0)))
    {
        rejected.reason = "invalid RZ owner heat or prescribed temperature";
        return rejected;
    }
    auto const eos = model.electronThermodynamicsExecutor();
    auto const axis_factor = WarpX::GetInstance().verboncoeurAxisCorrection()
        ? 1.0_rt / 3 : 1.0_rt / 4;
    auto const nodes = Copy(old.m_nodes, true);
    auto const heat = Copy(owner_heat_joules, true);
    auto const frozen_charge = Copy(old.m_charge, true);
    auto const frozen_temperature = Copy(old.m_temperature, true);
    auto const prescribed = prescribed_temperature ? Copy(*prescribed_temperature, true)
                                                 : amrex::MultiFab{};
    amrex::iMultiFab owners(old.m_owners->boxArray(), old.m_owners->DistributionMap(), 1, 0,
                           amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::dtoh_memcpy(owners, *old.m_owners);
    amrex::Gpu::streamSynchronize();

    // A prescribed candidate may have ordinary native copy noise, but it may
    // not contain a different physical temperature on any duplicate node.
    if (prescribed_temperature)
    {
        auto trial_reference = Copy(*prescribed_temperature);
        trial_reference.OverrideSync(geometry.periodicity());
        trial_reference.FillBoundary(geometry.periodicity());
        auto const trial_host = Copy(trial_reference, true);
        for (amrex::MFIter it(prescribed); it.isValid(); ++it)
        {
            for (amrex::BoxIterator node(it.validbox()); node.ok(); ++node)
            {
                auto const index = node();
                auto const t = prescribed[it](index), reference = trial_host[it](index);
                auto const charge = nodes[it](index, 0);
                valid = valid && std::isfinite(t + reference) &&
                    (charge == 0 ? t == reference
                                 : std::abs(t - reference) <= CopyAllowance(t, reference));
            }
        }
        if (!Collective(valid))
        {
            rejected.reason = "grossly inconsistent prescribed RZ temperature copies";
            return rejected;
        }
    }

    auto candidate_host = Copy(old.m_reference, true);
    amrex::MultiFab residual_host(old.m_temperature.boxArray(),
        old.m_temperature.DistributionMap(), 3, 1,
        amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    residual_host.setVal(0);
    amrex::Gpu::streamSynchronize();
    std::map<std::array<int, 3>, detail::NativeTableCapturedNode const*> table_nodes;
    if (old.m_table)
    {
        for (auto const& node : old.m_table->Owners())
        {
            valid = table_nodes.emplace(std::array<int, 3>{node.fab, node.index[0], node.index[1]},
                                        &node).second && valid;
        }
    }
    for (amrex::MFIter it(candidate_host); it.isValid(); ++it)
    {
        for (amrex::BoxIterator node(it.validbox()); node.ok(); ++node)
        {
            auto const index = node();
            auto const request = heat[it](index);
            if (owners[it](index) == 0)
            {
                valid = valid && request == 0;
                continue;
            }
            if (old.m_table)
            {
                auto const found = table_nodes.find({it.index(), index[0], index[1]});
                valid = valid && found != table_nodes.end();
                if (found == table_nodes.end()) { continue; }
                auto const prescribed_t = prescribed_temperature ? prescribed[it](index) : 0;
                auto const response = detail::EvaluateCompositeTableCaloricNode(
                    eos, found->second->caloric, request,
                    prescribed_temperature ? &prescribed_t : nullptr);
                valid = valid && response.Valid() &&
                    (prescribed_temperature || response.converged);
                if (!valid) { continue; }
                candidate_host[it](index) = response.temperature;
                residual_host[it](index, 0) = response.residual_joules;
                residual_host[it](index, 1) = std::abs(response.realized_joules) + std::abs(request);
                residual_host[it](index, 2) = response.arithmetic_bound_joules;
                continue;
            }
            auto const charge = nodes[it](index, 0);
            auto const volume = nodes[it](index, 4);
            auto const offset_joules = nodes[it](index, 5) * volume;
            auto const reference_t = nodes[it](index, 6);
            amrex::Real next_t = reference_t;
            if (charge == 0)
            {
                valid = valid && request == 0 && offset_joules == 0 &&
                    (!prescribed_temperature || prescribed[it](index) == reference_t);
            }
            else if (prescribed_temperature)
            {
                next_t = prescribed[it](index);
            }
            else
            {
                auto const effective_density = request / volume + nodes[it](index, 5);
                valid = valid && request >= -nodes[it](index, 3) * volume &&
                    std::isfinite(effective_density + nodes[it](index, 2));
                if (valid && effective_density != 0)
                {
                    auto const update = eos.applyEnergyDensityIncrement(
                        charge, {}, reference_t, effective_density);
                    valid = valid && update.valid;
                    next_t = update.temperature;
                }
            }
            valid = valid && std::isfinite(next_t) &&
                (charge == 0 || (next_t >= eos.minimumTemperature() &&
                                 next_t <= eos.maximumTemperature()));
            if (!valid) { continue; }
            candidate_host[it](index) = next_t;
            auto const centered_new_joules = charge == 0 ? 0 : volume *
                warpx::hybrid::CenteredOwnedNodalCaloricDifference(
                    eos, charge, reference_t, next_t);
            auto const after = charge == 0 ? 0 :
                eos.stateFromChargeDensityTemperature(charge, next_t).internal_energy_density;
            auto const realized = centered_new_joules - offset_joules;
            residual_host[it](index, 0) = realized - request;
            residual_host[it](index, 1) =
                std::abs(centered_new_joules) + std::abs(offset_joules) + std::abs(request);
            residual_host[it](index, 2) =
                128 * std::numeric_limits<amrex::Real>::epsilon() * volume *
                (std::abs(nodes[it](index, 2)) + std::abs(after));
            valid = valid && std::isfinite(realized + residual_host[it](index, 1) +
                                            residual_host[it](index, 2));
        }
    }
    if (!Collective(valid))
    {
        rejected.reason = "unaffordable or invalid RZ nodal caloric trial";
        return rejected;
    }

    NativeRZOwnedNodalCaloricResult result;
    result.candidate_temperature.define(old.m_temperature.boxArray(),
        old.m_temperature.DistributionMap(), 1, old.m_temperature.nGrowVect());
    amrex::htod_memcpy(result.candidate_temperature, candidate_host);
    result.candidate_temperature.OverrideSync(geometry.periodicity());
    result.candidate_temperature.FillBoundary(geometry.periodicity());
    // The host candidate is needed for independent original-corner energy
    // readback. This copy has matching layout and preserves owner values.
    amrex::dtoh_memcpy(candidate_host, result.candidate_temperature);
    amrex::Gpu::streamSynchronize();
    result.nodal_residual.define(old.m_temperature.boxArray(),
        old.m_temperature.DistributionMap(), 3, 1);
    amrex::htod_memcpy(result.nodal_residual, residual_host);
    result.nodal_residual.FillBoundary(geometry.periodicity());

    auto const cells = amrex::convert(old.m_temperature.boxArray(), amrex::IntVect(0));
    amrex::MultiFab realized_host(cells, old.m_temperature.DistributionMap(), 2, 1,
                                  amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    realized_host.setVal(0);
    amrex::Gpu::streamSynchronize();
    auto const floor = PhysConst::q_e *
        (model.transportsElectronInternalEnergyAtRawDensity() ? 0 : model.electronDensityFloor());
    amrex::MultiFab table_density;
    if (old.m_table) { table_density = Copy(old.m_table->FrozenUnitIonCharge(0), true); }
    for (amrex::MFIter it(realized_host); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            amrex::Real energy = 0, bound = 0;
            for (int corner = 0; corner < 4; ++corner)
            {
                auto node = cell();
                node[0] += corner & 1;
                node[1] += (corner >> 1) & 1;
                auto const charge = frozen_charge[it](node);
                auto const before_t = frozen_temperature[it](node);
                auto const after_t = candidate_host[it](node);
                if (charge == 0)
                {
                    valid = valid && before_t == after_t;
                    continue;
                }
                auto const effective_charge = std::max(charge, floor);
                auto const physical_volume = TransportNodeVolume(node, geometry, axis_factor);
                auto const geometric_volume = GeometricNodeVolume(node, geometry);
                auto const weight = CornerVolume(cell(), corner, geometry) *
                    physical_volume / geometric_volume;
                if (old.m_table)
                {
                    ElectronThermodynamicsExecutor::MaterialMassDensities density{};
                    density[0] = table_density[it](node) * old.m_table->FrozenMassPerUnitCharge(0);
                    auto const before = eos.stateFromMaterialMassDensitiesTemperature(
                        effective_charge, density, before_t);
                    auto const after = eos.stateFromMaterialMassDensitiesTemperature(
                        effective_charge, density, after_t);
                    energy += weight * eos.internalEnergyDensityDifference(
                        effective_charge, density, before_t, after_t);
                    bound += 128 * std::numeric_limits<amrex::Real>::epsilon() * weight *
                        (std::abs(before.internal_energy_density) +
                         std::abs(after.internal_energy_density));
                    continue;
                }
                auto const before = eos.stateFromChargeDensityTemperature(effective_charge,
                                                                           before_t);
                auto const after = eos.stateFromChargeDensityTemperature(effective_charge,
                                                                          after_t);
                energy += weight * warpx::hybrid::CenteredOwnedNodalCaloricDifference(
                    eos, effective_charge, before_t, after_t);
                bound += 128 * std::numeric_limits<amrex::Real>::epsilon() * weight *
                    (std::abs(before.internal_energy_density) +
                     std::abs(after.internal_energy_density));
            }
            realized_host[it](cell(), 0) = energy;
            realized_host[it](cell(), 1) = bound;
            valid = valid && std::isfinite(energy + bound);
        }
    }
    if (!Collective(valid))
    {
        rejected.reason = "invalid independent RZ cell-corner caloric realization";
        return rejected;
    }
    result.cell_realized_energy.define(cells, old.m_temperature.DistributionMap(), 2, 1);
    amrex::htod_memcpy(result.cell_realized_energy, realized_host);
    result.cell_realized_energy.FillBoundary(geometry.periodicity());
    result.valid = true;
    return result;
#endif
}
} // namespace warpx::radiation
