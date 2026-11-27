/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativeResolvedThermalSource.H"

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "NativeRZIndependentGrayOpacitySlice.H"
#include "Particles/MultiParticleContainer.H"
#include "ResolvedRZThermalDiffusionField.H"
#include "WarpX.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace warpx::radiation
{
namespace
{
#if defined(WARPX_DIM_RZ)
amrex::MultiFab
CopyRZ (amrex::MultiFab const& source)
{
    amrex::MultiFab copy(source.boxArray(), source.DistributionMap(), source.nComp(),
                         source.nGrowVect());
    amrex::MultiFab::Copy(copy, source, 0, 0, source.nComp(), source.nGrowVect());
    return copy;
}

amrex::MultiFab
HostRZ (amrex::MultiFab const& source)
{
    amrex::MultiFab host(source.boxArray(), source.DistributionMap(), source.nComp(),
                         source.nGrowVect(),
                         amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::dtoh_memcpy(host, source);
    amrex::Gpu::streamSynchronize();
    return host;
}

amrex::MultiFab
ComponentRZ (amrex::MultiFab const& source, int first, int count)
{
    amrex::MultiFab output(source.boxArray(), source.DistributionMap(), count, 0);
    amrex::MultiFab::Copy(output, source, first, 0, count, 0);
    return output;
}

bool
BuildDirectBathOnHost (amrex::MultiFab const& nodes, amrex::MultiFab const& weights,
                       amrex::Real radiation_constant, amrex::MultiFab& bath)
{
    bool valid = true;
    for (amrex::MFIter it(bath); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            amrex::Real mixture = 0;
            for (int corner = 0; corner < 4; ++corner)
            {
                auto node = cell();
                node[0] += corner & 1;
                node[1] += (corner >> 1) & 1;
                auto const t = nodes[it](node), t2 = t * t;
                mixture += weights[it](cell(), corner) * radiation_constant * t2 * t2;
            }
            bath[it](cell()) = mixture;
            valid = valid && std::isfinite(mixture) && mixture >= 0;
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid;
}
#endif
} // namespace

bool
TryBuildNativeResolvedRZReciprocalBath (
    amrex::MultiFab const& nodal_temperature, amrex::MultiFab const& corner_weights,
    amrex::Real radiation_constant, amrex::MultiFab& proper_bath)
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(nodal_temperature, corner_weights, radiation_constant, proper_bath);
    return false;
#else
    bool valid = nodal_temperature.ok() && corner_weights.ok() && proper_bath.ok() &&
        nodal_temperature.nComp() == 1 && corner_weights.nComp() == 4 &&
        proper_bath.nComp() == 1 && proper_bath.ixType().cellCentered() &&
        nodal_temperature.boxArray() == amrex::convert(proper_bath.boxArray(),
            amrex::IntVect::TheNodeVector()) &&
        corner_weights.boxArray() == proper_bath.boxArray() &&
        nodal_temperature.DistributionMap() == proper_bath.DistributionMap() &&
        corner_weights.DistributionMap() == proper_bath.DistributionMap() &&
        std::isfinite(radiation_constant) && radiation_constant > 0;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { return false; }
    auto const weights = HostRZ(corner_weights), nodes = HostRZ(nodal_temperature);
    auto bath = HostRZ(proper_bath);
    if (!BuildDirectBathOnHost(nodes, weights, radiation_constant, bath)) { return false; }
    amrex::htod_memcpy(proper_bath, bath);
    amrex::Gpu::streamSynchronize();
    return true;
#endif
}

bool
TryBuildNativeResolvedRZThermalTrial (
    NativeResolvedRZThermalTrialInputs const& inputs,
    amrex::MultiFab const& trial_temperature,
    amrex::MultiFab const& cell_temperature,
    amrex::MultiFab const& velocity,
    amrex::MultiFab const& prior_work,
    NativeThermalCornerEmission const* emission,
    NativeResolvedThermalSourceResult& candidate,
    amrex::MultiFab& projected, std::string& reason)
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(inputs, trial_temperature, cell_temperature, velocity,
                         prior_work, emission, candidate, projected);
    reason = "RZ trial is unavailable in Cartesian geometry";
    return false;
#else
    auto& live = inputs.live_photon_owner;
    auto const& endpoint = inputs.private_endpoint;
    auto const& replay = inputs.prepared_replay;
    auto const& diffusion = inputs.old_diffusion;
    auto const& mask = inputs.diffusion_mask;
    auto const& d_transport = inputs.d_chi_t;
    auto const& d_absorption = inputs.d_chi_a;
    auto const& geometry = inputs.geometry;
    auto const dt = inputs.dt;
    auto const& options = inputs.source_options;
    auto const& field_options = inputs.field_options;
    auto old_energy = ComponentRZ(diffusion, 0, 1);
    auto old_momentum = ComponentRZ(diffusion, 1, 3);
    auto old_momentum_host = HostRZ(old_momentum);
    auto transport_d_host = HostRZ(d_transport);
    auto absorption_d_host = HostRZ(d_absorption);
    bool temporal_valid = true;
    for (amrex::MFIter it(mask); it.isValid(); ++it)
    {
        std::vector<int> flags(static_cast<std::size_t>(mask[it].box().numPts()));
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, mask[it].dataPtr(),
                         mask[it].dataPtr() + flags.size(), flags.begin());
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const flag = flags[static_cast<std::size_t>(mask[it].box().index(cell()))];
            temporal_valid = temporal_valid && (flag == 0 || flag == 1) &&
                (flag == 0 || (std::isfinite(PhysConst::c * dt *
                                             transport_d_host[it](cell())) &&
                               PhysConst::c * dt * transport_d_host[it](cell()) >= 8));
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(temporal_valid);
    if (!temporal_valid)
    {
        reason = "RZ D temporal transport optical depth is below eight";
        return false;
    }
    bool trial_valid = emission && emission->weights && emission->nodal_temperature &&
        emission->nodal_temperature == &trial_temperature;
    if (trial_valid)
    {
        bool const split = options.rz_centered_ideal_temperature_trial;
        trial_valid = split
            ? emission->nodal_reference && emission->nodal_increment &&
              field_options.centered_thermal_difference
            : !emission->nodal_reference && !emission->nodal_increment;
        if (split && trial_valid)
        {
            for (auto const* field : {emission->nodal_reference, emission->nodal_increment})
            {
                trial_valid = trial_valid && field->boxArray() == trial_temperature.boxArray() &&
                    field->DistributionMap() == trial_temperature.DistributionMap() &&
                    field->nComp() == 1 && field->nGrowVect().allGE(1);
            }
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(trial_valid);
    if (!trial_valid)
    {
        reason = "missing reciprocal RZ frozen corner trial";
        return false;
    }
    candidate.photons = std::make_unique<PhotonPacketTransaction>(live, endpoint);
    auto packets = replay.Replay(cell_temperature, velocity, *candidate.photons, emission);
    if (!packets.valid)
    {
        reason = packets.reason;
        return false;
    }
    auto packet_debit = HostRZ(packets.radiation_debit);
    // FIELD and the final work guard must consume the same combined photon
    // momentum. Separate path/endpoint work subtraction loses their cancellation
    // before the small remaining D work is formed.
    auto excluded = HostRZ(packets.radiation_debit);
    int path_low = inputs.path_impulse != nullptr, path_high = path_low;
    amrex::ParallelDescriptor::ReduceIntMin(path_low);
    amrex::ParallelDescriptor::ReduceIntMax(path_high);
    bool path_valid = path_low == path_high;
    if (inputs.path_impulse)
    {
        path_valid = path_valid && inputs.path_impulse->ok() &&
            inputs.path_impulse->nComp() == 3 &&
            inputs.path_impulse->boxArray() == diffusion.boxArray() &&
            inputs.path_impulse->DistributionMap() == diffusion.DistributionMap() &&
            inputs.path_impulse->ixType().cellCentered();
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(path_valid);
    if (!path_valid)
    {
        reason = "invalid actual RZ path impulse in photon work partition";
        return false;
    }
    if (inputs.path_impulse)
    {
        auto const kick = HostRZ(*inputs.path_impulse);
        for (amrex::MFIter it(excluded); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                for (int d = 0; d < 3; ++d)
                {
                    excluded[it](cell(), d + 1) += PhysConst::c * kick[it](cell(), d);
                }
            }
        }
    }
    auto frame = HostRZ(velocity);
    // Reuse the existing pinned fields/transfer count on the default GPU path.
    auto const weights = HostRZ(*emission->weights);
    auto const nodes = HostRZ(trial_temperature);
    amrex::MultiFab reference_nodes, increment_nodes;
    if (options.rz_centered_ideal_temperature_trial)
    {
        reference_nodes = HostRZ(*emission->nodal_reference);
        increment_nodes = HostRZ(*emission->nodal_increment);
    }
    auto bath = HostRZ(d_absorption);
    bath.setVal(0);
    if (!BuildDirectBathOnHost(nodes, weights, options.packets.radiation_constant, bath))
    {
        reason = "nonphysical direct reciprocal RZ bath";
        return false;
    }
    amrex::MultiFab device_bath(bath.boxArray(), bath.DistributionMap(), 1, 0);
    amrex::htod_memcpy(device_bath, bath);
    auto d_work = HostRZ(prior_work);
    for (amrex::MFIter it(d_work); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            if (transport_d_host[it](cell()) == 0)
            {
                d_work[it](cell()) = 0;
            }
            else
            {
                amrex::Real packet_projection = 0;
                for (int component = 0; component < 3; ++component)
                {
                    packet_projection += frame[it](cell(), component) /
                        PhysConst::c * excluded[it](cell(), component + 1);
                }
                d_work[it](cell()) -= packet_projection;
            }
        }
    }
    // Raw host/device copies require identical FAB layout. Native overlapping
    // RequestedWork retains stencil ghosts; packing it into a zero-ghost FAB
    // shifts valid cells on GPU, although CPU's valid-cell Copy hides that bug.
    amrex::MultiFab device_work(d_work.boxArray(), d_work.DistributionMap(), 1,
                                d_work.nGrowVect());
    amrex::htod_memcpy(device_work, d_work);
    amrex::Gpu::streamSynchronize();
    auto field_controls = field_options;
    field_controls.reflecting_axial_table_trial = options.rz_closed_reciprocal_table_trial;
    if (options.rz_closed_reciprocal_endpoint_trial ||
        options.rz_closed_reciprocal_spatial_endpoint_trial)
    {
        field_controls.reflecting_axial_field_trial = true;
    }
    if (options.rz_independent_gray_opacity && !inputs.history_step)
    {
        reason = "independent-gray trial requires its actual History step";
        return false;
    }
    auto field = options.rz_independent_gray_opacity
        ? options.rz_independent_gray_opacity->Advance(*inputs.history_step, endpoint,
              old_energy, old_momentum, device_bath, mask, velocity, device_work)
        : TryAdvanceResolvedRZThermalDiffusionField(old_energy, old_momentum, device_bath,
              d_transport, d_absorption, mask, velocity, device_work, geometry, dt, field_controls);
    if (!field.valid)
    {
        reason = field.reason;
        return false;
    }
    candidate.rz_field_linear_iterations = field.linear_iterations;
    if (options.rz_independent_gray_opacity)
    { candidate.rz_independent_gray_corner_weights = CopyRZ(*emission->weights); }
    candidate.rz_field_linear_residual = field.linear_residual;
    candidate.rz_axial_wall_momentum_by_side = field.axial_wall_momentum_by_side;
    candidate.rz_axial_wall_momentum = field.axial_wall_momentum;
    NativeRZThermalLeakageResult leakage;
    if (options.rz_closed_reciprocal_endpoint_trial)
    {
        // Reuse the real zero-D field path. The leakage adapter is deliberately
        // periodic-only; for enforced all-K its physical event set is empty.
        // Verify exact zero accounts before constructing that empty set.
        auto const low = mask.min(0), high = mask.max(0);
        bool zero = low == 0 && high == 0 && std::isfinite(field.axial_wall_momentum) &&
                    field.axial_wall_momentum == 0;
        for (auto const wall : field.axial_wall_momentum_by_side)
        {
            zero = std::isfinite(wall) && wall == 0 && zero;
        }
        for (auto const* f : {&field.energy, &field.lab_momentum, &field.momentum_debit,
             &field.requested_heat, &field.physical_energy_debit, &field.raw_energy_debit,
             &field.proper_bath_energy_coefficient, &field.outward_pressure_stress,
             &field.theta_connection, &field.operand_scale, &field.storage_roundoff_bound})
        {
            auto const finite = f->is_finite();
            auto const norm = f->norm0(0, f->nComp(), f->nGrowVect());
            zero = finite && norm == 0 && zero;
        }
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            for (auto const* f : {field.face_lab_energy_current[d].get(),
                 field.face_lab_stress[d].get(), field.face_leaked_energy[d].get()})
            {
                auto const finite = f->is_finite();
                auto const norm = f->norm0(0, f->nComp(), f->nGrowVect());
                zero = finite && norm == 0 && zero;
            }
            auto const& face = *field.face_leaked_energy[d];
            leakage.actual_face_inventory[d] = std::make_unique<amrex::MultiFab>(
                face.boxArray(), face.DistributionMap(), 5, face.nGrowVect());
            leakage.actual_face_inventory[d]->setVal(0);
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(zero);
        if (!zero) { reason = "closed reciprocal endpoint is not exact zero D"; return false; }
        for (auto* f : {&leakage.donor_radiation_debit_correction,
                       &leakage.donor_operand_scale, &leakage.donor_roundoff_bound})
        {
            f->define(diffusion.boxArray(), diffusion.DistributionMap(), 4, 1);
            f->setVal(0);
        }
        leakage.photons = std::make_unique<PhotonPacketTransaction>(live, *candidate.photons);
        leakage.valid = true;
    }
    else
    {
        leakage = TryStageNativeRZThermalLeakage(
            live, *candidate.photons, field, mask, velocity, geometry, dt,
            options.packets.photon_energy,
            LogicalPhotonBirthKeyContext{options.packets.seed, options.packets.interval},
            options.rz_closed_reciprocal_spatial_endpoint_trial,
            options.rz_closed_reciprocal_table_trial,
            options.rz_single_selected_table_trial,
            options.rz_single_selected_table_vacuum);
    }
    if (!leakage.valid)
    {
        reason = leakage.reason;
        return false;
    }
    candidate.photons = std::move(leakage.photons);
    auto correction = HostRZ(leakage.donor_radiation_debit_correction);
    auto correction_scale = HostRZ(leakage.donor_operand_scale);
    auto correction_bound = HostRZ(leakage.donor_roundoff_bound);

    auto packet_scale = HostRZ(packets.operand_scale);
    auto births = HostRZ(packets.corner_emission_debit);
    auto next_energy = HostRZ(field.energy);
    auto next_momentum = HostRZ(field.lab_momentum);
    auto field_momentum = HostRZ(field.momentum_debit);
    auto field_physical = HostRZ(field.physical_energy_debit);
    auto field_raw = HostRZ(field.raw_energy_debit);
    auto field_scale = HostRZ(field.operand_scale);
    auto field_bound = HostRZ(field.storage_roundoff_bound);
    auto stress = HostRZ(field.outward_pressure_stress);
    auto coefficient = HostRZ(field.proper_bath_energy_coefficient);
    auto debit = HostRZ(diffusion), raw = HostRZ(diffusion);
    auto scale = HostRZ(diffusion), storage = HostRZ(diffusion);
    auto next = HostRZ(diffusion);
    amrex::MultiFab offset(diffusion.boxArray(), diffusion.DistributionMap(), 4, 0,
                           amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    offset.setVal(0);
    amrex::Gpu::streamSynchronize();
    for (amrex::MFIter it(debit); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const packet_energy = packet_debit[it](cell(), 0);
            debit[it](cell(), 0) = packet_energy + field_physical[it](cell()) +
                                    correction[it](cell(), 0);
            raw[it](cell(), 0) = packet_energy + field_raw[it](cell()) +
                                  correction[it](cell(), 0);
            next[it](cell(), 0) = next_energy[it](cell());
            scale[it](cell(), 0) = packet_scale[it](cell(), 0) +
                                   field_scale[it](cell()) + correction_scale[it](cell(), 0);
            storage[it](cell(), 0) = field_bound[it](cell()) +
                128 * std::numeric_limits<amrex::Real>::epsilon() *
                packet_scale[it](cell(), 0) + correction_bound[it](cell(), 0);
            amrex::Real projection = 0, total_birth = 0;
            for (int corner = 0; corner < 4; ++corner)
            {
                total_birth += births[it](cell(), 4 * corner);
            }
            for (int d = 0; d < 3; ++d)
            {
                auto const momentum = packet_debit[it](cell(), d + 1) +
                                      field_momentum[it](cell(), d) +
                                      correction[it](cell(), d + 1);
                debit[it](cell(), d + 1) = momentum;
                raw[it](cell(), d + 1) = momentum;
                next[it](cell(), d + 1) = next_momentum[it](cell(), d);
                scale[it](cell(), d + 1) = packet_scale[it](cell(), d + 1) +
                    std::abs(old_momentum_host[it](cell(), d)) +
                    std::abs(next_momentum[it](cell(), d)) +
                        std::abs(stress[it](cell(), d)) +
                        std::abs(field_momentum[it](cell(), d)) +
                        correction_scale[it](cell(), d + 1);
                storage[it](cell(), d + 1) =
                        256 * std::numeric_limits<amrex::Real>::epsilon() *
                        scale[it](cell(), d + 1) +
                        correction_bound[it](cell(), d + 1);
                projection += frame[it](cell(), d) / PhysConst::c * momentum;
            }
            projected[it](cell()) = debit[it](cell(), 0) - projection;
            auto const thermal_coefficient = PhysConst::c * dt *
                absorption_d_host[it](cell()) * coefficient[it](cell());
            for (int corner = 0; corner < 4; ++corner)
            {
                auto node = cell();
                node[0] += corner & 1;
                node[1] += (corner >> 1) & 1;
                auto const t = nodes[it](node), t2 = t * t;
                auto const own_bath = options.packets.radiation_constant * t2 * t2;
                auto const w = weights[it](cell(), corner);
                if (!field_controls.centered_thermal_difference)
                {
                    offset[it](cell(), corner) = births[it](cell(), 4 * corner) -
                        w * total_birth + w * thermal_coefficient *
                        (bath[it](cell()) - own_bath);
                }
                else
                {
                    offset[it](cell(), corner) = births[it](cell(), 4 * corner) -
                        w * total_birth;
                }
            }
            if (field_controls.centered_thermal_difference)
            {
                // The normalized-incidence covariance is a sum of paired
                // transfers: wi*wj*(Bj-Bi). Each pair adds to i and subtracts
                // from j; no hot absolute bath subtraction or stencil
                // renormalization supplies a spurious stiff net source.
                for (int first = 0; first < 4; ++first)
                {
                    auto a = cell();
                    a[0] += first & 1;
                    a[1] += (first >> 1) & 1;
                    auto const ta = nodes[it](a);
                    for (int second = first + 1; second < 4; ++second)
                    {
                        auto b = cell();
                        b[0] += second & 1;
                        b[1] += (second >> 1) & 1;
                        auto const tb = nodes[it](b);
                        auto const temperature_difference =
                            options.rz_centered_ideal_temperature_trial
                                ? (reference_nodes[it](b) - reference_nodes[it](a)) +
                                  (increment_nodes[it](b) - increment_nodes[it](a))
                                : tb - ta;
                        auto const difference = options.packets.radiation_constant *
                            temperature_difference * (tb + ta) * (tb * tb + ta * ta);
                        auto const transfer = weights[it](cell(), first) *
                            weights[it](cell(), second) * thermal_coefficient * difference;
                        offset[it](cell(), first) += transfer;
                        offset[it](cell(), second) -= transfer;
                    }
                }
            }
            trial_valid = trial_valid &&
                std::isfinite(projected[it](cell()) + thermal_coefficient);
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(trial_valid);
    if (!trial_valid)
    {
        reason = "nonfinite reciprocal RZ packet+D private account";
        return false;
    }
    candidate.diffusion_radiation = CopyRZ(diffusion);
    candidate.physical_debit = CopyRZ(diffusion);
    candidate.raw_debit = CopyRZ(diffusion);
    candidate.operand_scale = CopyRZ(diffusion);
    candidate.storage_roundoff_bound = CopyRZ(diffusion);
    candidate.reciprocal_corner_heat_offset.define(diffusion.boxArray(),
                                                   diffusion.DistributionMap(), 4, 0);
    amrex::htod_memcpy(candidate.diffusion_radiation, next);
    amrex::htod_memcpy(candidate.physical_debit, debit);
    amrex::htod_memcpy(candidate.raw_debit, raw);
    amrex::htod_memcpy(candidate.operand_scale, scale);
    amrex::htod_memcpy(candidate.storage_roundoff_bound, storage);
    amrex::htod_memcpy(candidate.reciprocal_corner_heat_offset, offset);
    candidate.rz_packet_debit = std::move(packets.radiation_debit);
    candidate.rz_history_birth_keys = std::move(packets.emission_birth_keys);
    for (auto const& face : leakage.face_accounts)
    {
        candidate.rz_history_birth_keys.insert(
            candidate.rz_history_birth_keys.end(), face.keys.begin(),
            face.keys.end());
    }
    candidate.rz_work_excluded_packet_debit = CopyRZ(candidate.rz_packet_debit);
    amrex::htod_memcpy(candidate.rz_work_excluded_packet_debit, excluded);
    candidate.rz_diffusion_work_input = std::move(device_work);
    candidate.rz_leakage =
        std::make_unique<NativeRZThermalLeakageResult>(std::move(leakage));
    candidate.rz_outward_pressure_stress =
        std::move(field.outward_pressure_stress);
    candidate.rz_theta_connection = std::move(field.theta_connection);
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        candidate.rz_face_lab_energy_current[direction] =
            std::move(field.face_lab_energy_current[direction]);
        candidate.rz_face_lab_stress[direction] =
            std::move(field.face_lab_stress[direction]);
        candidate.rz_face_leaked_energy[direction] =
            std::move(field.face_leaked_energy[direction]);
    }
    amrex::Gpu::streamSynchronize();
    return true;
#endif
}

NativeResolvedThermalSourceResult
TryAdvanceNativeResolvedRZThermalSource (
    WarpXParticleContainer& live, PhotonPacketTransaction const& endpoint,
    ParticleImpulseMaterial& material, std::string const& path,
    amrex::MultiFab const& diffusion, amrex::iMultiFab const& mask,
    amrex::MultiFab const& temperature, amrex::MultiFab const& transport_opacity,
    amrex::MultiFab const& absorption_opacity, amrex::MultiFab const& rate,
    HybridPICModel const& model, amrex::Geometry const& geometry, amrex::Real dt,
    NativeResolvedThermalSourceOptions const& provided_options,
    ResolvedRZThermalDiffusionFieldOptions const& field_options)
{
    NativeResolvedThermalSourceResult rejected;
    rejected.reason = "invalid or unsupported reciprocal RZ native thermal source";
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(live, endpoint, material, path, diffusion, mask, temperature,
                         transport_opacity, absorption_opacity, rate, model, geometry, dt,
                         provided_options, field_options);
    return rejected;
#else
    auto options = provided_options;
    int table_min[] = {options.rz_closed_reciprocal_table_trial,
                       field_options.reflecting_axial_table_trial,
                       options.rz_single_selected_table_trial,
                       field_options.single_selected_table_trial};
    int table_max[] = {table_min[0], table_min[1], table_min[2], table_min[3]};
    amrex::ParallelDescriptor::ReduceIntMin(table_min, 4);
    amrex::ParallelDescriptor::ReduceIntMax(table_max, 4);
    int scope_min = (options.rz_closed_reciprocal_endpoint_trial ? 1 : 0) +
                    (options.rz_closed_reciprocal_spatial_endpoint_trial ? 2 : 0);
    int scope_max = scope_min;
    amrex::ParallelDescriptor::ReduceIntMin(scope_min);
    amrex::ParallelDescriptor::ReduceIntMax(scope_max);
    if (scope_min != scope_max || scope_min == 3 ||
        table_min[0] != table_max[0] || table_min[1] != table_max[1] ||
        table_min[0] != table_min[1] ||
        table_min[2] != table_max[2] || table_min[3] != table_max[3] ||
        table_min[2] != table_min[3] || (table_min[2] && scope_min != 0) ||
        (options.rz_closed_reciprocal_table_trial && scope_min != 2)) { return rejected; }
    if (scope_min != 0)
    {
        bool admitted = false;
#if !defined(AMREX_USE_GPU)
        auto const eos = model.electronThermodynamicsExecutor();
        bool table = false;
#if defined(WARPX_USE_SINGULARITY_EOS)
        table = options.rz_closed_reciprocal_table_trial && eos.isSingularitySpiner() &&
            eos.m_num_materials == 1 && !eos.m_use_resolved_material_selection && eos.tableStorage() != nullptr &&
            material.CandidateSpecies().size() == 1 &&
            material.CandidateSpecies().front().name == model.electronThermodynamicsMaterialSpeciesName(0);
        if (table)
        {
            auto const& species = WarpX::GetInstance().GetPartContainer().GetParticleContainerFromName(
                model.electronThermodynamicsMaterialSpeciesName(0));
            table = !species.HasEvolvingChargeState();
        }
#endif
        admitted = !geometry.isPeriodic(1) &&
            (((eos.isIdealGas() || eos.isFixedChargeLatentEnergy()) && eos.m_num_materials == 0 &&
              !options.rz_closed_reciprocal_table_trial) || table) &&
            model.transportsElectronInternalEnergyAtRawDensity() &&
            !options.packets.rz_reflecting_axial_frozen_bath &&
            diffusion.nComp() == 4 && mask.nComp() == 1 &&
            diffusion.ixType().cellCentered() && mask.ixType().cellCentered() &&
            diffusion.boxArray() == endpoint.Particles().ParticleBoxArray(0) &&
            diffusion.DistributionMap() == endpoint.Particles().ParticleDistributionMap(0) &&
            mask.boxArray() == diffusion.boxArray() &&
            mask.DistributionMap() == diffusion.DistributionMap() &&
            std::isfinite(dt) && dt > 0 &&
            PhysConst::c * dt <= .5 * std::min(geometry.CellSize(0), geometry.CellSize(1));
#endif
        amrex::ParallelDescriptor::ReduceBoolAnd(admitted);
        if (!admitted) { return rejected; }
        if (options.rz_closed_reciprocal_endpoint_trial)
        {
            auto const low = mask.min(0), high = mask.max(0);
            auto const finite = diffusion.is_finite();
            auto const zero = diffusion.norm0(0, 4, diffusion.nGrowVect());
            if (low != 0 || high != 0 || !finite || zero != 0) { return rejected; }
        }
        options.packets.rz_reflecting_axial_reciprocal_endpoint = true;
    }
    bool valid = options.rz_reciprocal_implicit &&
        options.caloric_backend == NativeThermalCaloricBackend::ReciprocalNodalEmission &&
        options.reciprocal_composite_old_temperature && options.allow_vacuum_noop &&
        options.packets.rz_angular_conjugate &&
        options.packets.rz_reciprocal_corner_emission &&
        transport_opacity.nComp() == 1 &&
        transport_opacity.boxArray() == diffusion.boxArray() &&
        transport_opacity.DistributionMap() == diffusion.DistributionMap() &&
        absorption_opacity.nComp() == 1 &&
        absorption_opacity.boxArray() == diffusion.boxArray() &&
        absorption_opacity.DistributionMap() == diffusion.DistributionMap() &&
        mask.nComp() == 1 && mask.boxArray() == diffusion.boxArray() &&
        mask.DistributionMap() == diffusion.DistributionMap();
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { return rejected; }

    amrex::iMultiFab kinetic(mask.boxArray(), mask.DistributionMap(), 1, 0);
    amrex::iMultiFab capture(mask.boxArray(), mask.DistributionMap(), 1, 0);
    capture.setVal(1);
    auto old_energy = ComponentRZ(diffusion, 0, 1);
    auto old_momentum = ComponentRZ(diffusion, 1, 3);
    auto transport_host = HostRZ(transport_opacity);
    auto absorption_host = HostRZ(absorption_opacity);
    auto old_energy_host = HostRZ(old_energy);
    auto old_momentum_host = HostRZ(old_momentum);
    amrex::MultiFab d_transport(transport_opacity.boxArray(),
                                transport_opacity.DistributionMap(), 1, 0);
    amrex::MultiFab d_absorption(absorption_opacity.boxArray(),
                                 absorption_opacity.DistributionMap(), 1, 0);
    auto transport_d_host = HostRZ(d_transport);
    auto absorption_d_host = HostRZ(d_absorption);
    transport_d_host.setVal(0);
    absorption_d_host.setVal(0);
    bool temporal_valid = true;
    for (amrex::MFIter it(mask); it.isValid(); ++it)
    {
        std::vector<int> flags(static_cast<std::size_t>(mask[it].box().numPts()));
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, mask[it].dataPtr(),
                         mask[it].dataPtr() + flags.size(), flags.begin());
        std::vector<int> kinetic_flags(static_cast<std::size_t>(it.validbox().numPts()));
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const flag = flags[static_cast<std::size_t>(mask[it].box().index(cell()))];
            valid = valid && (flag == 0 || flag == 1);
            kinetic_flags[static_cast<std::size_t>(it.validbox().index(cell()))] = 1 - flag;
            if (flag != 0)
            {
                transport_d_host[it](cell()) = transport_host[it](cell());
                absorption_d_host[it](cell()) = absorption_host[it](cell());
                // The instantaneous comoving dipole in the D trial is a
                // diffusion-limit closure only across a temporally thick
                // interval, in addition to its spatial optical-depth guard.
                temporal_valid = temporal_valid &&
                    std::isfinite(PhysConst::c * dt * transport_host[it](cell())) &&
                    PhysConst::c * dt * transport_host[it](cell()) >= 8;
            }
            else
            {
                valid = valid && old_energy_host[it](cell()) == 0;
                for (int d = 0; d < 3; ++d)
                {
                    valid = valid && old_momentum_host[it](cell(), d) == 0;
                }
            }
        }
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, kinetic_flags.begin(),
                         kinetic_flags.end(), kinetic[it].dataPtr());
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    amrex::ParallelDescriptor::ReduceBoolAnd(temporal_valid);
    if (!temporal_valid)
    {
        rejected.reason = "RZ D temporal transport optical depth is below eight";
        return rejected;
    }
    if (!valid) { return rejected; }
    amrex::htod_memcpy(d_transport, transport_d_host);
    amrex::htod_memcpy(d_absorption, absorption_d_host);
    amrex::Gpu::streamSynchronize();

    NativePacketThermalEventReplay replay;
    if (!replay.Prepare(endpoint.Particles(), live, kinetic, absorption_opacity, rate,
                        geometry, dt, options.packets, rejected.reason, &capture))
    {
        return rejected;
    }

    NativeResolvedRZThermalTrialInputs trial_inputs{
        live, endpoint, replay, diffusion, mask, d_transport, d_absorption,
        geometry, dt, options, field_options};
    NativeResolvedThermalTrialBackend backend =
        [&] (amrex::MultiFab const& trial_temperature,
             amrex::MultiFab const& cell_temperature,
             amrex::MultiFab const& velocity,
             amrex::MultiFab const& prior_work,
             NativeThermalCornerEmission const* emission,
             NativeResolvedThermalSourceResult& candidate,
             amrex::MultiFab& projected, std::string& reason) -> bool
    {
        return TryBuildNativeResolvedRZThermalTrial(
            trial_inputs, trial_temperature, cell_temperature, velocity,
            prior_work, emission, candidate, projected, reason);
    };
    return TryAdvanceNativeResolvedThermalSource(
        live, endpoint, material, path, diffusion, mask, temperature,
        absorption_opacity, rate, model, geometry, dt, options, &backend);
#endif
}
} // namespace warpx::radiation
