/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativeCompositeTableCornerCapture.H"

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcMetricTransport.H"
#include "HybridCellMaterialState.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/MaterialRegistry.H"
#include "WarpX.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <limits>

namespace warpx::radiation::detail
{
namespace
{
bool
Collective (bool valid)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid;
}

std::unique_ptr<amrex::MultiFab>
Snapshot (amrex::MultiFab const& source, amrex::Geometry const& geometry)
{
    auto result = std::make_unique<amrex::MultiFab>(
        source.boxArray(), source.DistributionMap(), source.nComp(), source.nGrowVect());
    amrex::MultiFab::Copy(*result, source, 0, 0, source.nComp(), source.nGrowVect());
    result->FillBoundary(geometry.periodicity());
    return result;
}

amrex::MultiFab
Host (amrex::MultiFab const& field)
{
    amrex::MultiFab result(field.boxArray(), field.DistributionMap(), field.nComp(),
                           field.nGrowVect(),
                           amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::dtoh_memcpy(result, field);
    amrex::Gpu::streamSynchronize();
    return result;
}

amrex::Real
CornerMeasure (amrex::IntVect const& cell, int corner,
               amrex::Geometry const& geometry,
               NativeTableCornerCaptureOptions const& options)
{
    int di = corner & 1;
    int dj = AMREX_SPACEDIM >= 2 ? (corner >> 1) & 1 : 0;
    int dk = AMREX_SPACEDIM == 3 ? (corner >> 2) & 1 : 0;
    auto const geometric = HybridCellCornerVolume(
        cell[0], AMREX_SPACEDIM >= 2 ? cell[1] : 0,
        AMREX_SPACEDIM == 3 ? cell[2] : 0, di, dj, dk,
        geometry.ProbLoArray(), geometry.CellSizeArray(),
        amrex::lbound(geometry.Domain()));
#if defined(WARPX_DIM_RZ)
    auto const radial_node = cell[0] + di;
    auto const dr = geometry.CellSize(0), dz = geometry.CellSize(1);
    auto const radius = geometry.ProbLo(0) +
        (radial_node - geometry.Domain().smallEnd(0)) * dr;
    auto const axis_factor = options.rz_verboncoeur_axis_correction
        ? amrex::Real(1.0 / 3.0) : amrex::Real(1.0 / 4.0);
    auto transport = warpx::hybrid::cylindricalTransportNodeVolume(
        radius, dr, dz, axis_factor);
    if (radial_node == geometry.Domain().bigEnd(0) + 1) { transport *= 0.5; }
    auto const lo = std::max(geometry.ProbLo(0), radius - 0.5 * dr);
    auto const hi = std::min(geometry.ProbHi(0), radius + 0.5 * dr);
    auto const node_geometric = MathConst::pi * (hi * hi - lo * lo) * dz;
    // At an axial endpoint BOTH native and geometric nodal supports halve.
    // Their ratio is unchanged; the physical cell-corner wedge must NOT be
    // halved again. Only the sum over its incident physical cells is clipped.
    return geometric * transport / node_geometric;
#else
    amrex::ignore_unused(options);
    return geometric;
#endif
}

amrex::Real
ExpectedNodeMeasure (amrex::IntVect const& node, amrex::Geometry const& geometry,
                     NativeTableCornerCaptureOptions const& options)
{
#if defined(WARPX_DIM_RZ)
    auto const dr = geometry.CellSize(0), dz = geometry.CellSize(1);
    auto const radius = geometry.ProbLo(0) +
        (node[0] - geometry.Domain().smallEnd(0)) * dr;
    auto const axis_factor = options.rz_verboncoeur_axis_correction
        ? amrex::Real(1.0 / 3.0) : amrex::Real(1.0 / 4.0);
    auto measure = warpx::hybrid::cylindricalTransportNodeVolume(
        radius, dr, dz, axis_factor);
    if (node[0] == geometry.Domain().bigEnd(0) + 1) { measure *= 0.5; }
    if (!geometry.isPeriodic(1) &&
        (node[1] == geometry.Domain().smallEnd(1) ||
         node[1] == geometry.Domain().bigEnd(1) + 1))
    {
        measure *= 0.5;
    }
    return measure;
#else
    amrex::ignore_unused(node, options);
    auto measure = amrex::Real(1);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { measure *= geometry.CellSize(d); }
    return measure;
#endif
}
} // namespace

NativeCompositeTableCornerCapture
CaptureNativeCompositeTableCorners (
    ElectronThermodynamicsExecutor const& eos,
    amrex::MultiFab const& frozen_charge,
    amrex::MultiFab const& frozen_temperature,
    UnitIonChargeFields const& unit_ion_charge,
    amrex::Geometry const& geometry,
    NativeTableCornerCaptureOptions const& options)
{
    NativeCompositeTableCornerCapture rejected;
    int controls_min[8] = {static_cast<int>(eos.m_model), eos.m_num_materials,
                           eos.m_use_resolved_material_selection ? 1 : 0,
                           options.rz_native_fv ? 1 : 0,
                           options.rz_verboncoeur_axis_correction ? 1 : 0,
                           options.rz_reflecting_axial_trial ? 1 : 0,
                           options.require_single_selected_material0 ? 1 : 0,
                           options.allow_single_selected_vacuum ? 1 : 0};
    int controls_max[8];
    std::copy(std::begin(controls_min), std::end(controls_min), controls_max);
    amrex::Real floors_min[2] = {options.minimum_electron_density,
                                 options.thermodynamic_electron_density_floor};
    amrex::Real floors_max[2] = {floors_min[0], floors_min[1]};
    amrex::ParallelDescriptor::ReduceIntMin(controls_min, 8);
    amrex::ParallelDescriptor::ReduceIntMax(controls_max, 8);
    amrex::ParallelDescriptor::ReduceRealMin(floors_min, 2);
    amrex::ParallelDescriptor::ReduceRealMax(floors_max, 2);
    bool agreed = true;
    for (int n = 0; n < 8; ++n) { agreed = agreed && controls_min[n] == controls_max[n]; }
    for (int n = 0; n < 2; ++n) { agreed = agreed && floors_min[n] == floors_max[n]; }
    if (!Collective(agreed))
    {
        rejected.m_reason = "rank-dependent table corner capture controls";
        return rejected;
    }
    auto const cells = amrex::convert(frozen_charge.boxArray(), amrex::IntVect(0));
    amrex::Long physical_cells = 0;
    for (int box = 0; box < cells.size(); ++box) {
        physical_cells += cells[box].numPts();
    }
    bool valid = eos.isSingularitySpiner() && eos.m_num_materials > 0 &&
        eos.m_num_materials <= eos.max_materials &&
        frozen_charge.nComp() == 1 && frozen_temperature.nComp() == 1 &&
        frozen_charge.ixType() == amrex::IndexType::TheNodeType() &&
        frozen_charge.boxArray() == frozen_temperature.boxArray() &&
        frozen_charge.DistributionMap() == frozen_temperature.DistributionMap() &&
        frozen_charge.nGrowVect().allGE(1) &&
        frozen_temperature.nGrowVect().allGE(1) &&
        cells.isDisjoint() && cells.minimalBox() == geometry.Domain() &&
        physical_cells == geometry.Domain().numPts() &&
        std::isfinite(options.minimum_electron_density) &&
        options.minimum_electron_density >= 0 &&
        std::isfinite(options.thermodynamic_electron_density_floor) &&
        options.thermodynamic_electron_density_floor >= 0;
    valid = valid && (!options.allow_single_selected_vacuum ||
        (options.require_single_selected_material0 &&
         options.thermodynamic_electron_density_floor == 0));
    if (options.require_single_selected_material0)
    {
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU)
        valid = valid && eos.m_num_materials == 1 && eos.m_use_resolved_material_selection &&
            geometry.isPeriodic(1) && !options.rz_reflecting_axial_trial;
#else
        valid = false;
#endif
    }
#if defined(WARPX_DIM_RZ)
    bool axial = geometry.isPeriodic(1) && !options.rz_reflecting_axial_trial;
#if !defined(AMREX_USE_GPU)
    axial = axial || (options.rz_reflecting_axial_trial && !geometry.isPeriodic(1));
#endif
    valid = valid && options.rz_native_fv && geometry.ProbLo(0) == 0 &&
        !geometry.isPeriodic(0) && axial;
#else
    valid = valid && !options.rz_native_fv && !options.rz_reflecting_axial_trial &&
        geometry.Coord() == 0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        valid = valid && geometry.isPeriodic(d);
    }
#endif
    for (int material = 0; material < std::min(eos.m_num_materials, eos.max_materials);
         ++material)
    {
        auto const* field = unit_ion_charge[material];
        valid = valid && field && field->nComp() == 1 &&
            field->ixType() == frozen_charge.ixType() &&
            field->boxArray() == frozen_charge.boxArray() &&
            field->DistributionMap() == frozen_charge.DistributionMap() &&
            field->nGrowVect().allGE(1) &&
            std::isfinite(eos.m_mass_per_charge[material]) &&
            eos.m_mass_per_charge[material] > 0;
    }
    if (!Collective(valid))
    {
        rejected.m_reason = "invalid table material-corner layout, EOS or geometry";
        return rejected;
    }
    auto const finite_charge = frozen_charge.is_finite(0, 1, 0);
    auto const finite_temperature = frozen_temperature.is_finite(0, 1, 0);
    valid = finite_charge && finite_temperature;
    for (int material = 0; material < eos.m_num_materials; ++material)
    {
        valid = unit_ion_charge[material]->is_finite(0, 1, 0) && valid;
    }
    if (!Collective(valid))
    {
        rejected.m_reason = "nonfinite table material-corner input";
        return rejected;
    }

    NativeCompositeTableCornerCapture capture;
    capture.m_material_count = eos.m_num_materials;
    capture.m_charge = Snapshot(frozen_charge, geometry);
    capture.m_temperature = Snapshot(frozen_temperature, geometry);
    capture.m_mass_per_unit_charge = eos.m_mass_per_charge;
    for (int material = 0; material < eos.m_num_materials; ++material)
    {
        capture.m_unit_ion_charge[material] =
            Snapshot(*unit_ion_charge[material], geometry);
    }
    auto const rho = Host(*capture.m_charge);
    auto const old = Host(*capture.m_temperature);
    std::array<std::unique_ptr<amrex::MultiFab>,
               ElectronThermodynamicsExecutor::max_materials> material_host{};
    for (int material = 0; material < eos.m_num_materials; ++material)
    {
        material_host[material] =
            std::make_unique<amrex::MultiFab>(Host(*capture.m_unit_ion_charge[material]));
    }
    auto canonical = Snapshot(*capture.m_temperature, geometry);
    canonical->OverrideSync(geometry.periodicity());
    canonical->FillBoundary(geometry.periodicity());
    auto const reference = Host(*canonical);

    constexpr int slots = 1 << AMREX_SPACEDIM;
    auto const stride = 5 + eos.m_num_materials;
    amrex::MultiFab contribution(frozen_charge.boxArray(), frozen_charge.DistributionMap(),
                                 slots * stride, 1,
                                 amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    contribution.setVal(0);
    amrex::Gpu::streamSynchronize();
    auto const threshold = PhysConst::q_e * options.minimum_electron_density;
    auto const floor = PhysConst::q_e * options.thermodynamic_electron_density_floor;
    for (amrex::MFIter it(rho); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(amrex::enclosedCells(it.validbox())); cell.ok(); ++cell)
        {
            for (int slot = 0; slot < slots; ++slot)
            {
                auto node = cell();
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    node[d] += (slot >> d) & 1;
                }
                auto const raw = rho[it](node), temperature = old[it](node);
                auto const measure = CornerMeasure(cell(), slot, geometry, options);
                if (options.require_single_selected_material0)
                {
                    ElectronThermodynamicsExecutor::MaterialMassDensities masses{};
                    masses[0] = eos.materialMassDensityFromUnitIonChargeDensity(
                        0, (*material_host[0])[it](node));
                    bool const occupied = raw > 0 && eos.resolvedMaterial(masses) == 0;
                    bool const vacant = options.allow_single_selected_vacuum && raw == 0 &&
                        (*material_host[0])[it](node) == 0 && masses[0] == 0 &&
                        eos.resolvedMaterial(masses) == static_cast<int>(
                            warpx::materials::MaterialRegistry::ResolvedCell::Vacuum);
                    valid = valid && (occupied || vacant);
                }
                valid = valid && std::isfinite(raw + temperature + measure) &&
                    raw >= 0 && (raw == 0 || raw > threshold) &&
                    measure > 0;
                if (!valid) { continue; }
                auto const base = slot * stride;
                contribution[it](node, base) += 1;
                contribution[it](node, base + 1) += raw;
                contribution[it](node, base + 2) += raw == 0 ? 0 : std::max(raw, floor);
                contribution[it](node, base + 3) += temperature;
                contribution[it](node, base + 4) += measure;
                for (int material = 0; material < eos.m_num_materials; ++material)
                {
                    contribution[it](node, base + 5 + material) +=
                        (*material_host[material])[it](node);
                }
            }
        }
    }
    if (!Collective(valid))
    {
        rejected.m_reason = "invalid actual native table cell corner";
        return rejected;
    }
    amrex::MultiFab summed(contribution.boxArray(), contribution.DistributionMap(),
                           contribution.nComp(), contribution.nGrowVect());
    amrex::htod_memcpy(summed, contribution);
    amrex::Gpu::streamSynchronize();
    // Sum only distinct cell-corner records, never the already deposited
    // material density field itself. Slots label physical neighboring cells.
    summed.SumBoundary(geometry.periodicity(), true);
    summed.OverrideSync(geometry.periodicity());
    summed.FillBoundary(geometry.periodicity());
    auto const records = Host(summed);
    auto owners = frozen_temperature.OwnerMask(geometry.periodicity());
    amrex::iMultiFab owner_host(owners->boxArray(), owners->DistributionMap(), 1, 0,
                               amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::dtoh_memcpy(owner_host, *owners);
    amrex::Gpu::streamSynchronize();
    for (amrex::MFIter it(records); it.isValid(); ++it)
    {
        for (amrex::BoxIterator node(it.validbox()); node.ok(); ++node)
        {
            if (owner_host[it](node()) == 0) { continue; }
            NativeTableCapturedNode owned;
            owned.index = node();
            owned.fab = it.index();
            amrex::Real measure_sum = 0;
            for (int slot = 0; slot < slots; ++slot)
            {
                auto const base = slot * stride;
                auto const count = records[it](node(), base);
                valid = valid && (count == 0 || count == 1);
                if (count == 0) { continue; }
                auto& corner = owned.corners[owned.corner_count++];
                corner.raw_charge = records[it](node(), base + 1);
                corner.effective_charge = records[it](node(), base + 2);
                corner.old_temperature = records[it](node(), base + 3);
                corner.measure = records[it](node(), base + 4);
                measure_sum += corner.measure;
                for (int material = 0; material < eos.m_num_materials; ++material)
                {
                    auto const unit_charge = records[it](node(), base + 5 + material);
                    corner.mass_density[material] =
                        unit_charge * eos.m_mass_per_charge[material];
                }
            }
            auto const expected = ExpectedNodeMeasure(node(), geometry, options);
            valid = valid && std::isfinite(measure_sum + expected) && expected > 0 &&
                std::abs(measure_sum - expected) <=
                    256 * std::numeric_limits<amrex::Real>::epsilon() * expected;
            if (!valid) { continue; }
            owned.caloric = PrepareCompositeTableCaloricNode(
                eos, owned.corners, owned.corner_count, reference[it](node()));
            valid = valid && owned.caloric.Valid();
            capture.m_owners.push_back(std::move(owned));
        }
    }
    if (!Collective(valid))
    {
        rejected.m_reason = "invalid table owner, duplicated corner or old-copy EOS state";
        return rejected;
    }
    capture.m_global_owners = static_cast<amrex::Long>(capture.m_owners.size());
    amrex::ParallelDescriptor::ReduceLongSum(capture.m_global_owners);
    capture.m_valid = true;
    capture.m_reason.clear();
    return capture;
}

NativeCompositeTableCornerCapture
CaptureNativeCompositeTableCornersFromModel (
    HybridPICModel const& model, amrex::MultiFab const& frozen_charge,
    amrex::MultiFab const& frozen_temperature, amrex::Geometry const& geometry,
    NativeTableCornerCaptureOptions const& options)
{
    auto rejected = NativeCompositeTableCornerCapture{};
    auto& simulation = WarpX::GetInstance();
    auto const eos = model.electronThermodynamicsExecutor();
    auto resolved = options;
#if defined(WARPX_DIM_RZ)
    resolved.rz_verboncoeur_axis_correction = simulation.verboncoeurAxisCorrection();
#endif
    bool valid = simulation.finestLevel() == 0 &&
        model.transportsElectronInternalEnergyAtRawDensity() &&
        eos.isSingularitySpiner() &&
        model.electronThermodynamicsNumMaterials() == eos.m_num_materials &&
        resolved.thermodynamic_electron_density_floor == 0 &&
        simulation.m_fields.has(warpx::fields::FieldType::rho_fp, 0) &&
        simulation.m_fields.has(warpx::fields::FieldType::hybrid_electron_temperature_fp, 0);
    if (resolved.rz_reflecting_axial_trial)
    {
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU)
        valid = valid && resolved.rz_native_fv && !geometry.isPeriodic(1) &&
            WarpX::field_boundary_lo[1] == FieldBoundaryType::PEC &&
            WarpX::field_boundary_hi[1] == FieldBoundaryType::PEC &&
            WarpX::particle_boundary_lo[1] == ParticleBoundaryType::Reflecting &&
            WarpX::particle_boundary_hi[1] == ParticleBoundaryType::Reflecting;
#else
        valid = false;
#endif
    }
    if (!Collective(valid))
    {
        rejected.m_reason = "missing native table charge/temperature or raw-FV scope";
        return rejected;
    }
    auto const& native_geometry = simulation.Geom(0);
    auto const& native_charge = *simulation.m_fields.get(warpx::fields::FieldType::rho_fp, 0);
    auto const& native_temperature = *simulation.m_fields.get(
        warpx::fields::FieldType::hybrid_electron_temperature_fp, 0);
    valid = geometry.Domain() == native_geometry.Domain() &&
        geometry.Coord() == native_geometry.Coord() &&
        frozen_charge.boxArray() == native_charge.boxArray() &&
        frozen_charge.DistributionMap() == native_charge.DistributionMap() &&
        frozen_temperature.boxArray() == native_temperature.boxArray() &&
        frozen_temperature.DistributionMap() == native_temperature.DistributionMap();
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        valid = valid && geometry.ProbLo(d) == native_geometry.ProbLo(d) &&
            geometry.ProbHi(d) == native_geometry.ProbHi(d) &&
            geometry.isPeriodic(d) == native_geometry.isPeriodic(d);
    }
    UnitIonChargeFields fields{};
    for (int material = 0; material < std::min(eos.m_num_materials, eos.max_materials);
         ++material)
    {
        auto const name = "ni_charge_fp_" +
            model.electronThermodynamicsMaterialSpeciesName(material);
        valid = valid && simulation.m_fields.has(name, 0);
        if (valid) { fields[material] = simulation.m_fields.get(name, 0); }
    }
    if (!Collective(valid))
    {
        rejected.m_reason = "missing native table material deposition or raw-FV scope";
        return rejected;
    }
    return CaptureNativeCompositeTableCorners(eos, frozen_charge, frozen_temperature,
                                              fields, geometry, resolved);
}
namespace
{
bool
HasNativeSingleSelectedTableMaterial0Impl (
    HybridPICModel const& model, amrex::Geometry const& geometry, bool allow_exact_vacuum)
{
#if defined(WARPX_DIM_RZ) && defined(WARPX_USE_SINGULARITY_EOS) && !defined(AMREX_USE_GPU)
    auto& sim = WarpX::GetInstance();
    auto const eos = model.electronThermodynamicsExecutor();
    bool valid = eos.isSingularitySpiner() && eos.m_num_materials == 1 &&
        eos.m_use_resolved_material_selection && eos.tableStorage() != nullptr &&
        geometry.isPeriodic(1) && model.transportsElectronInternalEnergyAtRawDensity() &&
        sim.GetPartContainer().nSpecies() == 2;
    if (!Collective(valid)) { return false; }
    auto const& name = model.electronThermodynamicsMaterialSpeciesName(0);
    auto const& species = sim.GetPartContainer().GetParticleContainerFromName(name);
    auto const field_name = "ni_charge_fp_" + name;
    valid = !species.HasEvolvingChargeState() && species.getMass() > 0 &&
        species.getCharge() > 0 && !species.do_not_deposit && sim.m_fields.has(field_name, 0) &&
        eos.m_mass_per_charge[0] == static_cast<amrex::Real>(species.getMass()) / PhysConst::q_e &&
        eos.m_mass_per_physical_charge[0] == static_cast<amrex::Real>(species.getMass()) /
            static_cast<amrex::Real>(species.getCharge());
    for (auto const& other_name : sim.GetPartContainer().GetSpeciesNames())
    {
        if (other_name == name) { continue; }
        auto const& other = sim.GetPartContainer().GetParticleContainerFromName(other_name);
        valid = valid && other.AmIA<PhysicalSpecies::photon>() &&
            other.getCharge() == 0 && other.do_not_deposit;
    }
    if (!Collective(valid)) { return false; }
    auto const unit = Host(*sim.m_fields.get(field_name, 0));
    std::unique_ptr<amrex::MultiFab> raw;
    if (allow_exact_vacuum)
    {
        raw = std::make_unique<amrex::MultiFab>(
            Host(*sim.m_fields.get(warpx::fields::FieldType::rho_fp, 0)));
    }
    for (amrex::MFIter it(unit); it.isValid(); ++it)
    { for (amrex::BoxIterator node(it.validbox()); node.ok(); ++node)
        {
            ElectronThermodynamicsExecutor::MaterialMassDensities mass{};
            mass[0] = eos.materialMassDensityFromUnitIonChargeDensity(0, unit[it](node()));
            bool const occupied = std::isfinite(mass[0]) && mass[0] > 0 &&
                eos.resolvedMaterial(mass) == 0 &&
                (!allow_exact_vacuum || (std::isfinite((*raw)[it](node())) &&
                                        (*raw)[it](node()) > 0));
            bool const vacant = allow_exact_vacuum && (*raw)[it](node()) == 0 &&
                unit[it](node()) == 0 && mass[0] == 0 && eos.resolvedMaterial(mass) ==
                    static_cast<int>(warpx::materials::MaterialRegistry::ResolvedCell::Vacuum);
            valid = valid && (occupied || vacant);
        } }
    return Collective(valid);
#else
    amrex::ignore_unused(model, geometry, allow_exact_vacuum);
    return false;
#endif
}
} // namespace

bool HasNativeSingleSelectedTableMaterial0 (
    HybridPICModel const& model, amrex::Geometry const& geometry)
{
    return HasNativeSingleSelectedTableMaterial0Impl(model, geometry, false);
}

bool HasNativeSingleSelectedTableMaterial0OrExactVacuum (
    HybridPICModel const& model, amrex::Geometry const& geometry)
{
    return HasNativeSingleSelectedTableMaterial0Impl(model, geometry, true);
}
} // namespace warpx::radiation::detail
