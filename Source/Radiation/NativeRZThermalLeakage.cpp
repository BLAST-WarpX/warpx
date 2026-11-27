/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativeRZThermalLeakage.H"

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FluxRelaxedPackets.H"
#include "GroupMomentGeometry.H"
#include "NativeCompositeTableCornerCapture.H"
#include "PacketDiffusionInterfacePhotons.H"
#include "Particles/MultiParticleContainer.H"
#include "PhotonFourMomentum.H"
#include "RZAngularGeometry.H"
#include "WarpX.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

namespace warpx::radiation
{
namespace
{
#if defined(WARPX_DIM_RZ)
bool Collective (bool valid)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid;
}

amrex::MultiFab Host (amrex::MultiFab const& input)
{
    amrex::MultiFab output(input.boxArray(), input.DistributionMap(), input.nComp(),
                            input.nGrowVect(),
                            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::dtoh_memcpy(output, input);
    amrex::Gpu::streamSynchronize();
    return output;
}

amrex::iMultiFab HostMask (amrex::iMultiFab const& input,
                          amrex::Geometry const& geometry)
{
    amrex::iMultiFab staged(input.boxArray(), input.DistributionMap(), 1, 1);
    staged.setVal(0);
    amrex::iMultiFab::Copy(staged, input, 0, 0, 1, 0);
    staged.FillBoundary(geometry.periodicity());
    amrex::iMultiFab output(staged.boxArray(), staged.DistributionMap(), 1, 1,
                             amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::dtoh_memcpy(output, staged);
    amrex::Gpu::streamSynchronize();
    return output;
}

template <typename Data> auto HostVector (Data const& data)
{
    std::vector<typename Data::value_type> values(data.size());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, data.begin(), data.end(), values.begin());
    return values;
}

struct SelectedReadback
{
    bool valid = true;
    amrex::Long count = 0;
    amrex::GpuArray<amrex::Real, 5> inventory{}, positive{};
};

SelectedReadback ReadSelected (
    WarpXParticleContainer::Base const& particles,
    std::unordered_set<std::uint64_t> const& selected,
    amrex::iMultiFab const& mask, amrex::Geometry const& geometry)
{
    SelectedReadback result;
    if (selected.empty()) { return result; }
    std::array<int, 2> components{-1, -1};
    result.valid = FindPhotonDrawKeyComponents(particles.GetRealSoANames(), components) &&
                   components[0] >= 0;
    if (!result.valid) { return result; }
    std::unordered_set<std::uint64_t> seen;
    GroupMomentMetric const metric(geometry);
    for (auto const& [key, tile] : particles.GetParticles(0))
    {
        auto const& soa = tile.GetStructOfArrays();
        auto const ids = HostVector(soa.GetIdCPUData());
        auto const low = HostVector(soa.GetRealData(components[0]));
        auto const high = HostVector(soa.GetRealData(components[1]));
        auto const r = HostVector(soa.GetRealData(PIdx::r));
        auto const theta = HostVector(soa.GetRealData(PIdx::theta));
        auto const z = HostVector(soa.GetRealData(PIdx::z));
        auto const w = HostVector(soa.GetRealData(PIdx::w));
        auto const ux = HostVector(soa.GetRealData(PIdx::ux));
        auto const uy = HostVector(soa.GetRealData(PIdx::uy));
        auto const uz = HostVector(soa.GetRealData(PIdx::uz));
        for (std::size_t p = 0; p < ids.size(); ++p)
        {
            std::uint64_t draw_key = 0;
            result.valid = DecodePhotonDrawKey(low[p], high[p], ids[p], draw_key) &&
                           result.valid;
            if (!selected.contains(draw_key)) { continue; }
            result.valid = result.valid && seen.insert(draw_key).second &&
                std::isfinite(r[p] + theta[p] + z[p]);
            auto const& box = particles.ParticleBoxArray(0)[key.first];
            auto cell = geometry.Domain().smallEnd();
            bool const in_domain = r[p] >= geometry.ProbLo(0) &&
                r[p] < geometry.ProbHi(0) && z[p] >= geometry.ProbLo(1) &&
                z[p] < geometry.ProbHi(1);
            if (in_domain)
            {
                cell[0] += static_cast<int>(std::floor(
                    (r[p] - geometry.ProbLo(0)) / geometry.CellSize(0)));
                cell[1] += static_cast<int>(std::floor(
                    (z[p] - geometry.ProbLo(1)) / geometry.CellSize(1)));
            }
            result.valid = result.valid && in_domain && box.contains(cell);
            if (!in_domain || !box.contains(cell)) { continue; }
            result.valid = result.valid && mask[key.first](cell) == 0;
            auto const packet = PacketConversionFourMomentumFromPhoton(
                w[p], ux[p], uy[p], uz[p]);
            result.valid = result.valid && packet.valid;
            if (!packet.valid) { continue; }
            auto const cosine = std::cos(theta[p]), sine = std::sin(theta[p]);
            amrex::GpuArray<amrex::Real, 5> actual{
                packet.value[0],
                packet.value[1] * cosine + packet.value[2] * sine,
                -packet.value[1] * sine + packet.value[2] * cosine,
                packet.value[3], 0};
            actual[4] = r[p] * actual[2];
            result.valid = result.valid && metric.Angular(cell[0]).mean_radius > 0;
            for (int d = 0; d < 5; ++d)
            {
                result.inventory[d] += actual[d];
                result.positive[d] += std::abs(actual[d]);
            }
            ++result.count;
        }
    }
    return result;
}

std::unordered_set<std::uint64_t> ReceivingKeys (
    amrex::iMultiFab const& mask, amrex::Geometry const& geometry,
    ResolvedRZThermalDiffusionFieldResult const& field,
    LogicalPhotonBirthKeyContext const& context, bool& valid)
{
    std::unordered_set<std::uint64_t> selected;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> leakage;
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
    {
        leakage[axis] = Host(*field.face_leaked_energy[axis]);
    }
    for (amrex::MFIter it(mask); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            if (mask[it](index) != 0) { continue; }
            for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
            {
                for (int side : {-1, 1})
                {
                    if (!geometry.isPeriodic(axis) &&
                        ((side < 0 && index[axis] == geometry.Domain().smallEnd(axis)) ||
                         (side > 0 && index[axis] == geometry.Domain().bigEnd(axis))))
                    {
                        continue;
                    }
                    auto neighbor = index;
                    neighbor[axis] += side;
                    if (mask[it](neighbor) != 1) { continue; }
                    auto face = index;
                    if (side > 0) { ++face[axis]; }
                    auto const energy = leakage[axis][it](face);
                    valid = valid && std::isfinite(energy) && energy >= 0;
                    if (energy == 0) { continue; }
                    auto const outward = side < 0 ? 1 : -1;
                    for (int ray = 0; ray < 8; ++ray)
                    {
                        std::uint64_t key = 0;
                        valid = MakePhotonBirthDrawKey(geometry, face, axis, outward,
                            ray, context, key) && valid;
                        selected.insert(key);
                    }
                }
            }
        }
    }
    return selected;
}
#endif
} // namespace

NativeRZThermalLeakageResult TryStageNativeRZThermalLeakage (
    WarpXParticleContainer& live, PhotonPacketTransaction const& endpoint,
    ResolvedRZThermalDiffusionFieldResult const& field,
    amrex::iMultiFab const& diffusion_mask, amrex::MultiFab const& velocity,
    amrex::Geometry const& geometry, amrex::Real dt, amrex::Real photon_energy,
    LogicalPhotonBirthKeyContext const& context, bool reflecting_axial_endpoint_trial,
    bool reflecting_axial_table_trial, bool single_selected_table_trial,
    bool single_selected_table_vacuum)
{
    NativeRZThermalLeakageResult rejected;
    rejected.reason = "invalid private RZ thermal leakage face account";
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(live, endpoint, field, diffusion_mask, velocity, geometry,
                         dt, photon_energy, context, reflecting_axial_endpoint_trial,
                         reflecting_axial_table_trial, single_selected_table_trial,
                         single_selected_table_vacuum);
    return rejected;
#else
    auto const& boxes = diffusion_mask.boxArray();
    auto const& mapping = diffusion_mask.DistributionMap();
    auto const& native = live.Geom(0);
    // Snapshot validation is collective even when a caller's local field
    // candidate is already invalid. Every rank must enter it before the
    // ordinary collective layout/physics gate below.
    bool const snapshot_valid = endpoint.ValidateSnapshotFor(live);
    int scope_min[] = {reflecting_axial_endpoint_trial, reflecting_axial_table_trial,
                       single_selected_table_trial, single_selected_table_vacuum};
    int scope_max[] = {scope_min[0], scope_min[1], scope_min[2], scope_min[3]};
    amrex::ParallelDescriptor::ReduceIntMin(scope_min, 4);
    amrex::ParallelDescriptor::ReduceIntMax(scope_max, 4);
    if (!std::equal(std::begin(scope_min), std::end(scope_min), scope_max) ||
        (single_selected_table_vacuum && !single_selected_table_trial)) { return rejected; }
    if (single_selected_table_trial)
    {
        auto const* selected_model = WarpX::GetInstance().get_pointer_HybridPICModel();
        bool admitted = selected_model && !reflecting_axial_endpoint_trial &&
            !reflecting_axial_table_trial && geometry.isPeriodic(1);
        amrex::ParallelDescriptor::ReduceBoolAnd(admitted);
        if (!admitted) { return rejected; }
        bool const selected = single_selected_table_vacuum
            ? detail::HasNativeSingleSelectedTableMaterial0OrExactVacuum(*selected_model, geometry)
            : detail::HasNativeSingleSelectedTableMaterial0(*selected_model, geometry);
        if (!selected) { return rejected; }
    }
    bool reflecting_z = false;
#if !defined(AMREX_USE_GPU)
    auto const* model = WarpX::GetInstance().get_pointer_HybridPICModel();
    if (model)
    {
        auto const eos = model->electronThermodynamicsExecutor();
        bool table = false;
#if defined(WARPX_USE_SINGULARITY_EOS)
        table = reflecting_axial_table_trial && eos.isSingularitySpiner() &&
            eos.m_num_materials == 1 && !eos.m_use_resolved_material_selection && eos.tableStorage() != nullptr;
        if (table)
        {
            auto const& species = WarpX::GetInstance().GetPartContainer().GetParticleContainerFromName(
                model->electronThermodynamicsMaterialSpeciesName(0));
            table = !species.HasEvolvingChargeState();
        }
#endif
        reflecting_z = reflecting_axial_endpoint_trial && !geometry.isPeriodic(1) &&
            (((eos.isIdealGas() || eos.isFixedChargeLatentEnergy()) && eos.m_num_materials == 0 &&
              !reflecting_axial_table_trial) || table) &&
            model->transportsElectronInternalEnergyAtRawDensity() &&
            WarpX::field_boundary_lo[1] == FieldBoundaryType::PEC &&
            WarpX::field_boundary_hi[1] == FieldBoundaryType::PEC &&
            WarpX::particle_boundary_lo[1] == ParticleBoundaryType::Reflecting &&
            WarpX::particle_boundary_hi[1] == ParticleBoundaryType::Reflecting &&
            PhysConst::c * dt <= .5 * std::min(geometry.CellSize(0), geometry.CellSize(1));
    }
#endif
    bool valid = field.valid && snapshot_valid &&
        scope_min[0] == scope_max[0] && scope_min[1] == scope_max[1] &&
        (!reflecting_axial_table_trial || reflecting_axial_endpoint_trial) &&
        (!reflecting_axial_endpoint_trial || reflecting_z) &&
        std::isfinite(dt) && dt > 0 && std::isfinite(photon_energy) &&
        photon_energy > 0 && geometry.Domain() == native.Domain() &&
        geometry.ProbLo(0) == 0 && (geometry.isPeriodic(1) || reflecting_z) &&
        !geometry.isPeriodic(0) && boxes == live.ParticleBoxArray(0) &&
        mapping == live.ParticleDistributionMap(0) &&
        diffusion_mask.nComp() == 1 && diffusion_mask.ixType().cellCentered() &&
        velocity.nComp() == 3 && velocity.ixType().cellCentered() &&
        velocity.boxArray() == boxes && velocity.DistributionMap() == mapping;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        valid = valid && geometry.ProbLo(d) == native.ProbLo(d) &&
            geometry.ProbHi(d) == native.ProbHi(d) &&
            geometry.isPeriodic(d) == native.isPeriodic(d);
        auto const face_type = amrex::IntVect::TheDimensionVector(d);
        auto const face_boxes = amrex::convert(boxes, face_type);
        for (auto const* face : {field.face_lab_energy_current[d].get(),
                                 field.face_lab_stress[d].get(),
                                 field.face_leaked_energy[d].get()})
        {
            valid = valid && face != nullptr && face->boxArray() == face_boxes &&
                face->DistributionMap() == mapping &&
                face->ixType().toIntVect() == face_type;
        }
        if (field.face_lab_energy_current[d] && field.face_lab_stress[d] &&
            field.face_leaked_energy[d])
        {
            valid = valid && field.face_lab_energy_current[d]->nComp() == 1 &&
                field.face_lab_stress[d]->nComp() == 3 &&
                field.face_leaked_energy[d]->nComp() == 1;
        }
    }
    if (!Collective(valid)) { return rejected; }
    amrex::iMultiFab mask_device(boxes, mapping, 1, 1);
    mask_device.setVal(0);
    amrex::iMultiFab::Copy(mask_device, diffusion_mask, 0, 0, 1, 0);
    mask_device.FillBoundary(geometry.periodicity());
    auto mask = HostMask(mask_device, geometry);
    auto photons = std::make_unique<PhotonPacketTransaction>(live, endpoint);
    GroupMomentMetric const metric(geometry);
    NativeRZThermalLeakageResult result;
    amrex::MultiFab donor(boxes, mapping, 4, 1,
                          amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::MultiFab operands(boxes, mapping, 4, 1,
                             amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    donor.setVal(0);
    operands.setVal(0);
    amrex::Gpu::streamSynchronize();
    amrex::Long planned_births = 0;
    amrex::GpuArray<amrex::Real, 5> planned{}, planned_positive{};
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
    {
        auto const& leakage = *field.face_leaked_energy[axis];
        auto actual_face = std::make_unique<amrex::MultiFab>(
            leakage.boxArray(), mapping, 5, leakage.nGrowVect());
        detail::EmissionPlan plan;
        if (!BuildRZPacketDiffusionLeakageEmissionPlan(leakage, mask_device, geometry,
                photon_energy, plan, actual_face.get(), &velocity, axis, &context))
        {
            rejected.reason = "RZ leakage face plan or frame rejected";
            return rejected;
        }
        bool const keys_valid = Collective(plan.logical_draw_keys.size() == plan.x.size());
        if (!keys_valid)
        {
            rejected.reason = "RZ leakage face plan has incomplete logical keys";
            return rejected;
        }
        if (!detail::StageEmissionPlan(*photons, plan))
        {
            rejected.reason = "RZ leakage keyed private birth staging rejected";
            return rejected;
        }
        planned_births += static_cast<amrex::Long>(plan.x.size());
        auto const face_values = Host(*actual_face);
        auto const face_energy = Host(leakage);
        auto const current = Host(*field.face_lab_energy_current[axis]);
        auto const stress = Host(*field.face_lab_stress[axis]);
        auto owner = leakage.OwnerMask(geometry.periodicity());
        amrex::iMultiFab host_owner(owner->boxArray(), owner->DistributionMap(), 1, 0,
                                    amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
        amrex::dtoh_memcpy(host_owner, *owner);
        amrex::Gpu::streamSynchronize();
        std::unordered_set<std::uint64_t> plan_keys(plan.logical_draw_keys.begin(),
                                                     plan.logical_draw_keys.end());
        std::unordered_set<std::uint64_t> recorded_keys;
        for (amrex::MFIter it(face_energy); it.isValid(); ++it)
        {
            for (amrex::BoxIterator face(it.validbox()); face.ok(); ++face)
            {
                auto const index = face();
                if (host_owner[it](index) == 0) { continue; }
                auto const energy = face_energy[it](index);
                if (energy == 0) { continue; }
                auto left = index;
                --left[axis];
                auto const right = index;
                auto const sign = mask[it](left) - mask[it](right);
                valid = valid && (sign == 1 || sign == -1) &&
                    std::isfinite(energy) && energy > 0;
                if (!valid) { continue; }
                NativeRZThermalLeakageFaceAccount account;
                account.grid = it.index();
                account.axis = axis;
                account.face = index;
                account.outward_sign = sign;
                account.donor = sign > 0 ? left : right;
                account.receiver = sign > 0 ? right : left;
                valid = valid && donor[it].box().contains(account.donor) &&
                    operands[it].box().contains(account.donor);
                if (!valid) { continue; }
                auto const area = metric.Area(axis, index[0]);
                auto const scale = dt * area * sign;
                auto const expected_energy = scale * current[it](index);
                auto const check_scale = std::abs(energy) +
                    std::abs(expected_energy) +
                    std::abs(dt * area * current[it](index));
                valid = valid && std::isfinite(check_scale + area) && area > 0 &&
                    std::abs(expected_energy - energy) <=
                        64 * std::numeric_limits<amrex::Real>::epsilon() * check_scale;
                account.accepted[0] = energy;
                account.accepted[1] = scale * stress[it](index, 0);
                account.accepted[2] = scale * stress[it](index, 1) *
                    (axis == 0 ? metric.Radius(index[0]) /
                        metric.Angular(account.donor[0]).mean_radius : 1);
                account.accepted[3] = scale * stress[it](index, 2);
                for (int d = 0; d < 5; ++d)
                {
                    account.actual[d] = face_values[it](index, d);
                    planned[d] += account.actual[d];
                    planned_positive[d] += std::abs(account.actual[d]);
                }
                auto const rbar = metric.Angular(account.donor[0]).mean_radius;
                account.correction[0] = account.accepted[0] - account.actual[0];
                account.correction[1] = account.accepted[1] - account.actual[1];
                account.correction[2] = account.accepted[2] - account.actual[4] / rbar;
                account.correction[3] = account.accepted[3] - account.actual[3];
                auto const lever_max = metric.Radius(account.donor[0] + 1) / rbar;
                for (int d = 0; d < 4; ++d)
                {
                    auto const actual = d == 2 ? account.actual[4] / rbar :
                        account.actual[d == 3 ? 3 : d];
                    donor[it](account.donor, d) += account.correction[d];
                    operands[it](account.donor, d) += std::abs(account.accepted[d]) +
                        std::abs(actual) + std::abs(account.correction[d]) +
                        std::abs(account.actual[0]) * (d == 2 ? lever_max : 1);
                    valid = valid && std::isfinite(account.accepted[d] +
                        account.correction[d] + operands[it](account.donor, d));
                }
                for (int ray = 0; ray < 8; ++ray)
                {
                    auto& key = account.keys[ray];
                    valid = MakePhotonBirthDrawKey(geometry, index, axis, sign,
                                ray, context, key) && valid;
                    valid = valid && plan_keys.contains(key) &&
                        recorded_keys.insert(key).second;
                }
                result.face_accounts.push_back(account);
            }
        }
        valid = valid && recorded_keys.size() == plan_keys.size();
        if (!Collective(valid))
        {
            rejected.reason = "late RZ accepted face, stress or key account rejected";
            return rejected;
        }
        result.actual_face_inventory[axis] = std::move(actual_face);
    }
    auto const selected = ReceivingKeys(mask, geometry, field, context, valid);
    if (!Collective(valid))
    {
        rejected.reason = "RZ receiving-face key construction rejected";
        return rejected;
    }
    auto const baseline = ReadSelected(endpoint.Particles(), selected, mask, geometry);
    valid = baseline.valid && baseline.count == 0;
    if (!Collective(valid))
    {
        rejected.reason = "RZ leakage keys collide with immutable endpoint";
        return rejected;
    }
    auto const readback = ReadSelected(photons->Particles(), selected, mask, geometry);
    valid = readback.valid;
    amrex::MultiFab selected_cells(boxes, mapping, 4, 0);
    auto selected_keys = std::vector<std::uint64_t>(selected.begin(), selected.end());
    amrex::Long birth_count = 0;
    valid = photons->ProvisionalBirthInventory(selected_cells, geometry,
                birth_count, &selected_keys) && valid;
    if (!Collective(valid))
    {
        rejected.reason = "RZ selected-key staged packet readback rejected";
        return rejected;
    }
    auto const selected_host = Host(selected_cells);
    amrex::Real received_lz = 0;
    for (amrex::MFIter it(selected_host); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            received_lz += metric.Angular(cell()[0]).mean_radius *
                selected_host[it](cell(), 2);
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(received_lz);
    amrex::Long actual_count = readback.count;
    amrex::ParallelDescriptor::ReduceLongSum(actual_count);
    amrex::ParallelDescriptor::ReduceLongSum(planned_births);
    amrex::Real staged[5], planned_global[5], positive[5];
    for (int d = 0; d < 5; ++d)
    {
        staged[d] = readback.inventory[d];
        planned_global[d] = planned[d];
        positive[d] = readback.positive[d] + planned_positive[d];
    }
    amrex::ParallelDescriptor::ReduceRealSum(staged, 5);
    amrex::ParallelDescriptor::ReduceRealSum(planned_global, 5);
    amrex::ParallelDescriptor::ReduceRealSum(positive, 5);
    auto const operations = 8 * static_cast<amrex::Real>(planned_births) +
        32 * amrex::ParallelDescriptor::NProcs() + 128;
    auto const numerator = operations * std::numeric_limits<amrex::Real>::epsilon();
    valid = actual_count == planned_births && birth_count == planned_births &&
        planned_births >= 0 && numerator < .125 && std::isfinite(numerator);
    if (!Collective(valid))
    {
        rejected.reason = "RZ selected-key face count or reduction bound rejected";
        return rejected;
    }
    auto const gamma_n = numerator / (1 - numerator);
    for (int d = 0; d < 5; ++d)
    {
        auto const scale = positive[d] + std::abs(staged[d]) +
                           std::abs(planned_global[d]);
        valid = valid && std::isfinite(scale) &&
            std::abs(staged[d] - planned_global[d]) <=
                gamma_n * scale;
    }
    valid = valid && std::abs(received_lz - staged[4]) <=
        gamma_n *
            (positive[4] + std::abs(received_lz) + std::abs(staged[4]));
    if (!Collective(valid))
    {
        rejected.reason = "actual staged RZ selected-key E/P/Lz disagrees with faces";
        return rejected;
    }
    amrex::MultiFab device_donor(boxes, mapping, 4, 1);
    amrex::MultiFab device_operands(boxes, mapping, 4, 1);
    amrex::htod_memcpy(device_donor, donor);
    amrex::htod_memcpy(device_operands, operands);
    amrex::Gpu::streamSynchronize();
    device_donor.SumBoundary(geometry.periodicity());
    device_operands.SumBoundary(geometry.periodicity());
    amrex::MultiFab correction(boxes, mapping, 4, 0);
    amrex::MultiFab scale(boxes, mapping, 4, 0);
    amrex::MultiFab bound(boxes, mapping, 4, 0);
    auto donor_host = Host(device_donor), operand_host = Host(device_operands);
    amrex::MultiFab correction_host(boxes, mapping, 4, 0,
                                    amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::MultiFab scale_host(boxes, mapping, 4, 0,
                               amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::MultiFab bound_host(boxes, mapping, 4, 0,
                               amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    for (amrex::MFIter it(bound_host); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            for (int d = 0; d < 4; ++d)
            {
                auto const magnitude = operand_host[it](cell(), d);
                correction_host[it](cell(), d) = donor_host[it](cell(), d);
                scale_host[it](cell(), d) = magnitude;
                bound_host[it](cell(), d) = magnitude == 0 ? 0 :
                    std::nextafter(512 * std::numeric_limits<amrex::Real>::epsilon() *
                        magnitude, std::numeric_limits<amrex::Real>::infinity());
                valid = valid && std::isfinite(donor_host[it](cell(), d) + magnitude +
                    bound_host[it](cell(), d)) && magnitude >= 0;
            }
        }
    }
    if (!Collective(valid))
    {
        rejected.reason = "RZ leakage donor correction or operand bound rejected";
        return rejected;
    }
    amrex::htod_memcpy(correction, correction_host);
    amrex::htod_memcpy(scale, scale_host);
    amrex::htod_memcpy(bound, bound_host);
    amrex::Gpu::streamSynchronize();
    result.donor_radiation_debit_correction = std::move(correction);
    result.donor_operand_scale = std::move(scale);
    result.donor_roundoff_bound = std::move(bound);
    result.staged_births = planned_births;
    result.photons = std::move(photons);
    result.valid = true;
    return result;
#endif
}
} // namespace warpx::radiation
