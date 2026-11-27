/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativeResolvedRZThermalInterval.H"

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "NativePhotonCarryRouting.H"
#include "NativePhotonReplayInternals.H"
#include "NativeCompositeTableCornerCapture.H"
#include "NativeRZIndependentGrayOpacitySlice.H"
#include "PhotonLogicalDrawKey.H"
#include "Particles/MultiParticleContainer.H"
#include "RZMomentGeometry.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

namespace warpx::radiation
{
NativeResolvedRZThermalIntervalOptions::NativeResolvedRZThermalIntervalOptions ()
{
    thermal.caloric_backend = NativeThermalCaloricBackend::ReciprocalNodalEmission;
    thermal.reciprocal_composite_old_temperature = true;
    thermal.rz_reciprocal_implicit = true;
    thermal.allow_vacuum_noop = true;
    thermal.packets.rz_angular_conjugate = true;
    thermal.packets.rz_reciprocal_corner_emission = true;
    collisions.moving_cross_sections = true;
}

namespace
{
#if defined(WARPX_DIM_RZ)
amrex::MultiFab
Copy (amrex::MultiFab const& source)
{
    amrex::MultiFab result(source.boxArray(), source.DistributionMap(), source.nComp(),
                           source.nGrowVect());
    amrex::MultiFab::Copy(result, source, 0, 0, source.nComp(), source.nGrowVect());
    return result;
}

amrex::MultiFab
Host (amrex::MultiFab const& source)
{
    amrex::MultiFab result(source.boxArray(), source.DistributionMap(), source.nComp(),
                           source.nGrowVect(),
                           amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::dtoh_memcpy(result, source);
    amrex::Gpu::streamSynchronize();
    return result;
}

bool
Collective (bool valid)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid;
}

bool
ExactZeroKAccount (amrex::MultiFab const& field)
{
    bool valid = field.ok() && field.nComp() > 0;
    if (!Collective(valid)) { return false; }
    bool const finite = field.is_finite(0, field.nComp(), field.nGrowVect());
    auto const norm = field.norm0(0, field.nComp(), field.nGrowVect());
    return finite && norm == 0;
}

bool
EmptyKFlags (amrex::iMultiFab const& kinetic, amrex::iMultiFab const& capture, std::string& reason)
{
    auto const kinetic_low = kinetic.min(0), kinetic_high = kinetic.max(0);
    auto const capture_low = capture.min(0), capture_high = capture.max(0);
    bool const valid = Collective(kinetic_low == 0 && kinetic_high == 0 &&
                                 capture_low == 1 && capture_high == 1);
    if (!valid) { reason = "all-D derived kinetic/capture flags changed"; }
    return valid;
}

bool
EmptyKPathReport (NativePhotonPathSubsegmentReport const& report)
{
    auto const& p = report.path;
    bool valid = p.valid && p.colliding_photons == 0 && p.collision_events == 0 &&
        p.replayed_photons == 0 && p.face_crossings == 0 && p.wall_reflections == 0 &&
        p.axial_wall_transfers == 0 && p.escaped_photons == 0 && p.escaped_colliding_photons == 0 &&
        p.photon_carries.empty() && report.photon_carries.empty() &&
        p.axial_wall_draw_keys.empty() && p.photon_energy_debit == 0 &&
        p.axial_wall_momentum_operand == 0;
    for (auto const& vector : {p.incoming_inventory, p.outgoing_inventory, p.escaped_inventory})
    {
        for (auto value : vector) { valid = std::isfinite(value) && value == 0 && valid; }
    }
    for (auto const& vector : {p.material_impulse, p.wall_momentum})
    {
        for (auto value : vector) { valid = std::isfinite(value) && value == 0 && valid; }
    }
    for (auto value : p.axial_wall_momentum_by_side)
    { valid = std::isfinite(value) && value == 0 && valid; }
    // report.subsegments counts authentic replay calls, not physical events.
    return Collective(valid);
}

bool
EmptyKSourceOutputs (NativeRZThermalAllDNoKEventWitness const& witness,
                     amrex::iMultiFab const& mask,
                     NativeResolvedThermalSourceResult const& candidate, std::string& reason)
{
    bool valid = candidate.photons && candidate.rz_leakage && candidate.rz_leakage->valid &&
        candidate.rz_leakage->staged_births == 0 && candidate.rz_leakage->face_accounts.empty() &&
        candidate.rz_history_birth_keys.empty();
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        valid = candidate.rz_face_leaked_energy[d] &&
            candidate.rz_leakage && candidate.rz_leakage->actual_face_inventory[d] && valid;
    }
    if (!Collective(valid))
    {
        reason = "all-D source has actual packet/leakage birth or incomplete empty accounts";
        return false;
    }
    if (!witness.ValidateOwners(*candidate.photons, mask, reason)) { return false; }
    for (auto const* field : std::array<amrex::MultiFab const*, 5>{&candidate.rz_packet_debit,
         &candidate.rz_work_excluded_packet_debit,
         &candidate.rz_leakage->donor_radiation_debit_correction,
         &candidate.rz_leakage->donor_operand_scale,
         &candidate.rz_leakage->donor_roundoff_bound})
    {
        bool const zero = ExactZeroKAccount(*field);
        valid = zero && valid;
    }
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        bool const leaked = ExactZeroKAccount(*candidate.rz_face_leaked_energy[d]);
        bool const actual = ExactZeroKAccount(*candidate.rz_leakage->actual_face_inventory[d]);
        valid = leaked && actual && valid;
    }
    // Packet local counters/operand scale are not exposed here. Empty stored
    // endpoints plus exact-zero kinetic flags make the original ray loops
    // empty structurally; never fabricate telemetry or zero D/work accounts.
    if (!Collective(valid))
    {
        reason = "all-D source exposed packet or leakage account is not exact zero";
        return false;
    }
    return true;
}

using RayIdentity = std::pair<bool, std::uint64_t>;

bool
RayKey (photon_replay_detail::CapturedTile const& tile, long p,
        std::array<int, 2> const& components, RayIdentity& identity)
{
    auto const low = tile.real_data[components[0] * tile.count + p];
    auto const high = tile.real_data[components[1] * tile.count + p];
    identity.first = low != 0 || high != 0;
    return DecodePhotonDrawKey(low, high, tile.idcpu[p], identity.second);
}

// Preserve physical replay evidence by EXACT stored identities, not native
// provisional IDs alone. Endpoint absorption changes weights, not u; new
// thermal/leakage rays have no chronological collision in this interval.
bool
PhysicalPathCensus (WarpXParticleContainer::Base const& photons,
                    CollidedDiffusionAdmissionComponents const& census,
                    std::set<RayIdentity>& collided)
{
    std::array<int, 2> keys{};
    bool valid = FindPhotonDrawKeyComponents(photons.GetRealSoANames(), keys) &&
                 keys[0] >= 0;
    if (!Collective(valid))
    {
        return false;
    }
    for (auto const& tile :
         photon_replay_detail::CaptureParticles(photons).tiles)
    {
        for (long p = 0; p < tile.count; ++p)
        {
            if (!amrex::ConstParticleIDWrapper(tile.idcpu[p]).is_valid())
            {
                continue;
            }
            bool changed = false;
            for (int d = 0; d < 3; ++d)
            {
                changed =
                    changed ||
                    tile.real_data[(PIdx::ux + d) * tile.count + p] !=
                        tile.real_data[census.initial_u[d] * tile.count + p];
            }
            RayIdentity key{};
            valid = RayKey(tile, p, keys, key) && valid;
            if (changed)
            {
                valid = collided.insert(key).second && valid;
            }
        }
    }
    return Collective(valid);
}

bool
RetainPhysicalPathCensus (WarpXParticleContainer::Base& photons,
                          CollidedDiffusionAdmissionComponents const& census,
                          std::set<RayIdentity> const& collided)
{
    std::array<int, 2> keys{};
    bool valid = FindPhotonDrawKeyComponents(photons.GetRealSoANames(), keys) &&
                 keys[0] >= 0;
    if (!Collective(valid))
    {
        return false;
    }
    for (auto const& tile :
         photon_replay_detail::CaptureParticles(photons).tiles)
    {
        auto& soa = photons.GetParticles(0).at(tile.key).GetStructOfArrays();
        for (int d = 0; d < 3; ++d)
        {
            std::vector<amrex::ParticleReal> markers(
                static_cast<std::size_t>(tile.count));
            for (long p = 0; p < tile.count; ++p)
            {
                RayIdentity key{};
                valid = RayKey(tile, p, keys, key) && valid;
                markers[p] =
                    tile.real_data[(collided.contains(key) ? census.initial_u[d]
                                                           : PIdx::ux + d) *
                                       tile.count +
                                   p];
            }
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, markers.begin(),
                             markers.end(),
                             soa.GetRealData(census.initial_u[d]).begin());
        }
    }
    return Collective(valid);
}

// Direct actual native rays and extensive annular D, using physical Lz rather
// than a summed theta projection. Only three scalar accounts are reduced.
std::array<long double, 3>
RadiationInventory (WarpXParticleContainer::Base const& photons,
                    amrex::MultiFab const& diffusion, amrex::Geometry const& geometry)
{
    std::array<long double, 3> result{};
    auto const state = photon_replay_detail::CaptureParticles(photons);
    for (auto const& tile : state.tiles)
    {
        for (long p = 0; p < tile.count; ++p)
        {
            if (!amrex::ConstParticleIDWrapper(tile.idcpu[p]).is_valid()) { continue; }
            auto value = [&] (int component) -> long double
            {
                return tile.real_data[static_cast<std::size_t>(component) * tile.count + p];
            };
            auto const ux = value(PIdx::ux), uy = value(PIdx::uy), uz = value(PIdx::uz);
            auto const factor = value(PIdx::w) * PhysConst::m_e * PhysConst::c;
            auto const radius = value(PIdx::r), theta = value(PIdx::theta);
            result[0] += factor * std::sqrt(ux * ux + uy * uy + uz * uz);
            result[1] += factor * uz;
            result[2] += factor * radius * (std::cos(theta) * uy - std::sin(theta) * ux);
        }
    }
    auto const host = Host(diffusion);
    for (amrex::MFIter it(host); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const lo = static_cast<long double>(geometry.ProbLo(0)) +
                (cell()[0] - geometry.Domain().smallEnd(0)) *
                static_cast<long double>(geometry.CellSize(0));
            auto const hi = lo + geometry.CellSize(0);
            auto const rbar = (2.L / 3) * (lo * lo + lo * hi + hi * hi) / (lo + hi);
            result[0] += host[it](cell(), 0);
            result[1] += host[it](cell(), 3);
            result[2] += rbar * host[it](cell(), 2);
        }
    }
#ifdef AMREX_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, result.data(), 3, MPI_LONG_DOUBLE, MPI_SUM,
                  amrex::ParallelDescriptor::Communicator());
#endif
    return result;
}
#endif
} // namespace

bool
ValidateNativeResolvedRZThermalIntervalScope (
    HybridPICModel const& model, amrex::Geometry const& geometry,
    NativeResolvedRZThermalIntervalOptions const& options)
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(model, geometry, options);
    return false;
#else
    bool const closed = options.reflecting_axial_fixed_mask_trial;
    int low[] = {closed, options.reflecting_axial_table_trial,
                 options.thermal.rz_closed_reciprocal_table_trial,
                 options.field.reflecting_axial_table_trial,
                 options.single_selected_table_trial,
                 options.thermal.rz_single_selected_table_trial,
                 options.field.single_selected_table_trial,
                 static_cast<int>(options.absorption_depth_policy),
                 options.authenticated_history_step != nullptr,
                 options.all_d_no_k_event_witness != nullptr,
                 static_cast<int>(options.photon_time_policy),
                 options.thermal.rz_independent_gray_opacity != nullptr,
                 options.thermal.rz_single_selected_table_vacuum,
                 options.field.single_selected_table_vacuum,
                 options.thermal.rz_native_nodal_force,
                 options.thermal.rz_centered_ideal_temperature_trial};
    int high[16];
    std::copy(std::begin(low), std::end(low), high);
    amrex::ParallelDescriptor::ReduceIntMin(low, 16);
    amrex::ParallelDescriptor::ReduceIntMax(high, 16);
    bool valid = std::equal(std::begin(low), std::end(low), high) &&
        options.reflecting_axial_table_trial == options.thermal.rz_closed_reciprocal_table_trial &&
        options.reflecting_axial_table_trial == options.field.reflecting_axial_table_trial &&
        (!options.reflecting_axial_table_trial || closed) && geometry.isPeriodic(1) != closed &&
        !options.thermal.rz_closed_reciprocal_endpoint_trial &&
        !options.thermal.packets.rz_reflecting_axial_frozen_bath &&
        options.thermal.rz_closed_reciprocal_spatial_endpoint_trial == closed &&
        options.thermal.packets.rz_reflecting_axial_reciprocal_endpoint == closed &&
        options.field.reflecting_axial_field_trial == closed &&
        !options.field.centered_thermal_difference &&
        !options.thermal.rz_centered_ideal_temperature_trial;
    auto const nodal_eos = model.electronThermodynamicsExecutor();
    valid = valid && (!options.thermal.rz_native_nodal_force ||
        (!options.single_selected_table_trial && !options.reflecting_axial_table_trial &&
         (nodal_eos.isIdealGas() || nodal_eos.isFixedChargeLatentEnergy()) &&
         nodal_eos.m_num_materials == 0 &&
         WarpX::nox >= 1 && WarpX::nox <= 3));
    valid = valid && options.single_selected_table_trial == options.thermal.rz_single_selected_table_trial &&
        options.single_selected_table_trial == options.field.single_selected_table_trial &&
        (!options.single_selected_table_trial || (!closed && options.desired_diffusion_mask == nullptr &&
         options.radiation_representation_carry == nullptr));
    bool const step_depth = options.absorption_depth_policy ==
        NativeRZThermalAbsorptionDepthPolicy::PureAbsorptionActualStep16;
    bool const all_d = options.absorption_depth_policy ==
        NativeRZThermalAbsorptionDepthPolicy::AllDNoPhotonImplicitAbsorption;
    bool const implicit_step = options.photon_time_policy ==
        NativeRZThermalPhotonTimePolicy::AllDNoPhotonImplicitStep;
    valid = valid && (implicit_step || options.photon_time_policy ==
        NativeRZThermalPhotonTimePolicy::PhotonHalfCell) && (!implicit_step || all_d);
    bool const independent = options.thermal.rz_independent_gray_opacity != nullptr;
    valid = valid &&
        options.thermal.rz_single_selected_table_vacuum ==
            options.field.single_selected_table_vacuum &&
        (!options.thermal.rz_single_selected_table_vacuum ||
         (options.single_selected_table_trial && !closed &&
          (!independent || (options.thermal.rz_independent_gray_opacity->IsMixed() &&
           options.thermal.rz_independent_gray_opacity->Options().opacity.allow_exact_vacuum))));
    valid = valid && (!independent ||
        (!implicit_step && options.single_selected_table_trial && !closed &&
         options.initial_photon_clocks != nullptr));
    valid = valid && (step_depth || all_d || options.absorption_depth_policy ==
        NativeRZThermalAbsorptionDepthPolicy::WholeOuter16) &&
        (step_depth || all_d || independent) == (options.authenticated_history_step != nullptr) &&
        all_d == (options.all_d_no_k_event_witness != nullptr);
    if (step_depth)
    {
        valid = valid && options.single_selected_table_trial && !closed &&
            options.initial_photon_clocks != nullptr && options.desired_diffusion_mask == nullptr &&
            options.radiation_representation_carry == nullptr;
#if defined(AMREX_USE_GPU)
        valid = false;
#endif
    }
    if (all_d)
    {
        valid = valid && !closed && options.initial_photon_clocks != nullptr &&
            options.desired_diffusion_mask == nullptr &&
            options.radiation_representation_carry == nullptr &&
            std::numeric_limits<amrex::Real>::digits == 53 &&
            std::numeric_limits<amrex::ParticleReal>::digits == 53;
#if defined(AMREX_USE_GPU)
        valid = false;
#endif
    }
    if (!Collective(valid)) { return false; }
    if (independent)
    {
        bool const mixed = options.thermal.rz_independent_gray_opacity->IsMixed();
        valid = valid && (mixed ? (!all_d && !step_depth) : all_d);
        if (!Collective(valid)) { return false; }
    }
    if (options.single_selected_table_trial)
    {
        valid = options.thermal.rz_single_selected_table_vacuum
            ? detail::HasNativeSingleSelectedTableMaterial0OrExactVacuum(model, geometry)
            : detail::HasNativeSingleSelectedTableMaterial0(model, geometry);
    }
    if (closed)
    {
#if defined(AMREX_USE_GPU)
        valid = false;
#else
        auto const eos = model.electronThermodynamicsExecutor();
        bool table = false;
#if defined(WARPX_USE_SINGULARITY_EOS)
        table = options.reflecting_axial_table_trial && eos.isSingularitySpiner() &&
            eos.m_num_materials == 1 && !eos.m_use_resolved_material_selection && eos.tableStorage() != nullptr;
        if (table)
        {
            auto const& species = WarpX::GetInstance().GetPartContainer().GetParticleContainerFromName(
                model.electronThermodynamicsMaterialSpeciesName(0));
            table = !species.HasEvolvingChargeState();
        }
#endif
        valid = valid && !geometry.isPeriodic(1) &&
            (((eos.isIdealGas() || eos.isFixedChargeLatentEnergy()) && eos.m_num_materials == 0 &&
              !options.reflecting_axial_table_trial) || table) &&
            model.transportsElectronInternalEnergyAtRawDensity() &&
            options.desired_diffusion_mask == nullptr &&
            options.radiation_representation_carry == nullptr &&
            WarpX::field_boundary_lo[1] == FieldBoundaryType::PEC &&
            WarpX::field_boundary_hi[1] == FieldBoundaryType::PEC &&
            WarpX::particle_boundary_lo[1] == ParticleBoundaryType::Reflecting &&
            WarpX::particle_boundary_hi[1] == ParticleBoundaryType::Reflecting;
#endif
    }
    return Collective(valid);
#endif
}

NativeResolvedRZThermalIntervalResult
TryAdvanceNativeResolvedRZThermalInterval (
    WarpXParticleContainer& live, PhotonPacketTransaction const& start,
    ParticleImpulseMaterial& material, std::string const& path,
    amrex::MultiFab const& old_diffusion, amrex::iMultiFab const& mask,
    amrex::MultiFab const& old_temperature, amrex::MultiFab const& scattering,
    amrex::MultiFab const& absorption, amrex::MultiFab const& initial_rate,
    HybridPICModel const& model, amrex::Geometry const& geometry, amrex::Real dt,
    NativeResolvedRZThermalIntervalOptions const& options)
{
    NativeResolvedRZThermalIntervalResult rejected;
    rejected.reason = "invalid or unsupported private chronological RZ thermal interval";
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(live, start, material, path, old_diffusion, mask, old_temperature,
                         scattering, absorption, initial_rate, model, geometry, dt, options);
    return rejected;
#else
    if (!ValidateNativeResolvedRZThermalIntervalScope(model, geometry, options))
    {
        return rejected;
    }
    if (!Collective(!options.thermal.rz_single_selected_table_vacuum ||
                    options.thermal.rz_independent_gray_opacity != nullptr))
    {
        rejected.reason = "exact-vacuum interval lacks its actual independent mixed epoch";
        return rejected;
    }
    if (!start.ValidateSnapshotFor(live)) { return rejected; }
    auto const& boxes = old_diffusion.boxArray();
    auto const& mapping = old_diffusion.DistributionMap();
    bool const adaptive = options.desired_diffusion_mask != nullptr;
    bool const history = options.initial_photon_clocks != nullptr;
    bool const implicit_step = options.photon_time_policy ==
        NativeRZThermalPhotonTimePolicy::AllDNoPhotonImplicitStep;
    bool valid =
        adaptive == (options.radiation_representation_carry != nullptr) &&
        old_diffusion.nComp() == 4 && old_diffusion.ixType().cellCentered() &&
        mask.nComp() == 1 && mask.boxArray() == boxes &&
        mask.DistributionMap() == mapping && scattering.nComp() == 1 &&
        absorption.nComp() == 1 && initial_rate.nComp() == 3 &&
        std::isfinite(dt) && dt > 0 && options.packet_subsegment_limit > 0 &&
        (implicit_step || dt * PhysConst::c <=
            .5 * std::min(geometry.CellSize(0), geometry.CellSize(1))) &&
        (!history || !adaptive) &&
        (history ||
         (options.collisions.seed == options.thermal.packets.seed &&
          options.collisions.interval == options.thermal.packets.interval)) &&
        options.collisions.moving_cross_sections &&
        !options.collisions.absorption && !options.collisions.emission &&
        options.collisions.angular_law ==
            NativePhotonCollisionAngularLaw::Isotropic &&
        options.thermal.packets.survival_weighted_emission &&
        options.thermal.rz_reciprocal_implicit;
    for (auto const* field : {&scattering, &absorption, &initial_rate})
    {
        valid = valid && field->boxArray() == boxes && field->DistributionMap() == mapping &&
            field->ixType().cellCentered();
    }
    if (adaptive && options.radiation_representation_carry)
    {
        auto const& desired = *options.desired_diffusion_mask;
        auto const& carry = *options.radiation_representation_carry;
        valid = valid && desired.nComp() == 1 &&
                desired.ixType().cellCentered() &&
                desired.boxArray() == boxes &&
                desired.DistributionMap() == mapping && carry.nComp() == 4 &&
                carry.ixType().cellCentered() && carry.boxArray() == boxes &&
                carry.DistributionMap() == mapping;
    }
    int mode_low = adaptive, mode_high = mode_low;
    amrex::ParallelDescriptor::ReduceIntMin(mode_low);
    amrex::ParallelDescriptor::ReduceIntMax(mode_high);
    int history_low = history, history_high = history_low;
    amrex::ParallelDescriptor::ReduceIntMin(history_low);
    amrex::ParallelDescriptor::ReduceIntMax(history_high);
    int low = options.packet_subsegment_limit, high = low;
    amrex::ParallelDescriptor::ReduceIntMin(low);
    amrex::ParallelDescriptor::ReduceIntMax(high);
    if (!Collective(valid && low == high && mode_low == mode_high &&
                    history_low == history_high))
    {
        return rejected;
    }
    bool const step_depth = options.absorption_depth_policy ==
        NativeRZThermalAbsorptionDepthPolicy::PureAbsorptionActualStep16;
    bool const all_d = options.absorption_depth_policy ==
        NativeRZThermalAbsorptionDepthPolicy::AllDNoPhotonImplicitAbsorption;
    // Scope already collectively requires actual History context/witness and
    // initial clocks for the extended policy. No standalone Interval authority.
    auto validate_duration = [&] (PhotonPacketTransaction const& photons,
                                  amrex::iMultiFab const& consumed_mask,
                                  std::string& reason)
    {
        return !implicit_step || options.all_d_no_k_event_witness->ValidatePhotonDuration(
            *options.authenticated_history_step, dt, options.collisions.seed,
            options.collisions.interval, live, photons, consumed_mask, geometry, reason);
    };
    if (!validate_duration(start, mask, rejected.reason)) { return rejected; }
    if (step_depth)
    {
        auto const& context = *options.authenticated_history_step;
        if (!ValidateNativeRZThermalHistoryStepContext(context, dt,
                options.collisions.seed, options.collisions.interval, rejected.reason) ||
            !ValidatePureAbsorptionHistoryStep(context, scattering, absorption, initial_rate,
                geometry, options.thermal.packets.max_absorption_optical_depth, rejected.reason))
        {
            return rejected;
        }
    }
    if (all_d &&
        (!options.all_d_no_k_event_witness->ValidateStep(*options.authenticated_history_step,
             dt, options.collisions.seed, options.collisions.interval, start, mask,
             rejected.reason) ||
         !Collective(options.initial_photon_clocks->empty()) ||
         !ValidateAllDImplicitAbsorptionHistoryStep(*options.authenticated_history_step,
             *options.all_d_no_k_event_witness, scattering, absorption, initial_rate, geometry,
             options.collisions.max_optical_depth,
             options.thermal.packets.max_absorption_optical_depth,
             rejected.reason)))
    { return rejected; }

    // Own the exact outer state once. No trial may use the previous candidate
    // as its new baseline, nor publish a preceding transport/material source.
    PhotonPacketTransaction immutable(live, start);
    // Demotion births are now incoming physical rays. Take a fresh PRIVATE
    // census after conversion, never retain stale cohort or zero birth scratch.
    if (adaptive && !SnapshotCollidedDiffusionAdmission(immutable.Particles(),
                                                        options.census)
                         .valid)
    {
        rejected.reason =
            "invalid incoming adaptive RZ physical collision census";
        return rejected;
    }
    auto frozen_diffusion = Copy(old_diffusion), frozen_temperature = Copy(old_temperature);
    auto frozen_scattering = Copy(scattering), frozen_absorption = Copy(absorption);
    auto frozen_rate = Copy(initial_rate);
    if (step_depth && !ValidatePureAbsorptionHistoryStep(*options.authenticated_history_step,
            frozen_scattering, frozen_absorption, frozen_rate, geometry,
            options.thermal.packets.max_absorption_optical_depth, rejected.reason))
    {
        return rejected;
    }
    amrex::iMultiFab frozen_mask(boxes, mapping, 1, mask.nGrowVect());
    amrex::iMultiFab::Copy(frozen_mask, mask, 0, 0, 1, mask.nGrowVect());
    if (!validate_duration(immutable, frozen_mask, rejected.reason)) { return rejected; }
    if (all_d &&
        (!options.all_d_no_k_event_witness->ValidateStep(*options.authenticated_history_step,
             dt, options.collisions.seed, options.collisions.interval, immutable, frozen_mask,
             rejected.reason) ||
         !ValidateAllDImplicitAbsorptionHistoryStep(*options.authenticated_history_step,
             *options.all_d_no_k_event_witness, frozen_scattering, frozen_absorption, frozen_rate,
             geometry, options.collisions.max_optical_depth,
             options.thermal.packets.max_absorption_optical_depth, rejected.reason)))
    { return rejected; }
    amrex::iMultiFab frozen_desired;
    amrex::MultiFab frozen_carry, future_total;
    if (adaptive)
    {
        auto const& desired = *options.desired_diffusion_mask;
        frozen_desired.define(boxes, mapping, 1, desired.nGrowVect());
        amrex::iMultiFab::Copy(frozen_desired, desired, 0, 0, 1,
                               desired.nGrowVect());
        frozen_carry = Copy(*options.radiation_representation_carry);
        future_total = Copy(frozen_scattering);
        amrex::MultiFab::Add(future_total, frozen_absorption, 0, 0, 1,
                             future_total.nGrowVect());
        if (!frozen_carry.is_finite())
        {
            return rejected;
        }
    }
    auto chi_s = Host(frozen_scattering), chi_a = Host(frozen_absorption);
    auto const* independent = options.thermal.rz_independent_gray_opacity;
    if (independent && !independent->Validate(*options.authenticated_history_step,
            immutable, frozen_mask, rejected.reason)) { return rejected; }
    if (independent && independent->IsMixed() && !independent->ValidatePacketFrame(frozen_rate,
            options.collisions.max_optical_depth,
            options.thermal.packets.max_absorption_optical_depth,
            rejected.reason)) { return rejected; }
    amrex::MultiFab total(boxes, mapping, 1, 0), d_absorption(boxes, mapping, 1, 0);
    auto total_host = Host(total), d_absorption_host = Host(d_absorption);
    amrex::iMultiFab host_mask(boxes, mapping, 1, 0,
                               amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::iMultiFab::Copy(host_mask, frozen_mask, 0, 0, 1, 0);
    amrex::Gpu::streamSynchronize();
    amrex::iMultiFab kinetic(boxes, mapping, 1, 0), capture_all(boxes, mapping, 1, 0);
    capture_all.setVal(1);
    for (amrex::MFIter it(total_host); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const s = chi_s[it](cell()), a = chi_a[it](cell());
            auto const flag = host_mask[it](cell());
            auto const transport = independent
                ? independent->Opacity().Snapshot().coefficients.rosseland_transport[it](cell())
                : s + a;
            if (independent)
            {
                auto const& channels = independent->Opacity().Snapshot().coefficients;
                bool const same = s == channels.true_scattering[it](cell()) &&
                    a == channels.proper_absorption[it](cell());
                valid = valid && same;
            }
            total_host[it](cell()) = flag == 1 ? transport : 0;
            d_absorption_host[it](cell()) = flag == 1 ? a : 0;
            valid = valid && s >= 0 && a >= 0 && std::isfinite(s + a);
            valid = valid && (flag == 0 || flag == 1) &&
                (flag == 0 || PhysConst::c * dt * transport >= 8);
        }
        std::vector<int> flags(static_cast<std::size_t>(it.validbox().numPts()));
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            flags[static_cast<std::size_t>(it.validbox().index(cell()))] =
                1 - host_mask[it](cell());
        }
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, flags.begin(), flags.end(),
                         kinetic[it].dataPtr());
    }
    if (!Collective(valid))
    {
        rejected.reason = "RZ frozen packet channels, mask or D temporal depth rejected";
        return rejected;
    }
    if (all_d && !EmptyKFlags(kinetic, capture_all, rejected.reason))
    {
        rejected.reason = "all-D derived kinetic/capture flags lost empty event closure";
        return rejected;
    }
    amrex::htod_memcpy(total, total_host);
    amrex::htod_memcpy(d_absorption, d_absorption_host);
    amrex::Gpu::streamSynchronize();
    RZNativePhotonPathSubsegmentReplay transport;
    RZNativePhotonPathReplay continued_transport;
    std::vector<NativePhotonPathCarry> frozen_clocks;
    if (history)
    {
        frozen_clocks = *options.initial_photon_clocks;
    }
    RZNativePhotonPathReport initial_path;
    bool const prepared = implicit_step
        ? continued_transport.PrepareAllDNoPhoton(
              immutable, live, frozen_scattering, frozen_rate, geometry, dt, options.collisions,
              frozen_clocks, *options.authenticated_history_step,
              *options.all_d_no_k_event_witness, initial_path)
        : history
            ? continued_transport.Prepare(
                  immutable.Particles(), live, frozen_scattering, frozen_rate,
                  geometry, dt, options.collisions, frozen_clocks, initial_path,
                  NativePhotonPathSnapshotPolicy::KeyedProvisional)
            : transport.Prepare(
                  immutable.Particles(), live, frozen_scattering, frozen_rate,
                  geometry, dt, options.collisions,
                  options.packet_subsegment_limit, initial_path,
                  NativePhotonPathSnapshotPolicy::KeyedProvisional);
    if (!prepared)
    {
        rejected.reason = initial_path.reason;
        return rejected;
    }
    auto const incoming = RadiationInventory(immutable.Particles(), frozen_diffusion, geometry);
    NativePhotonPathSubsegmentReport accepted_path;
    auto thermal_controls = options.thermal;
#if !defined(AMREX_USE_GPU)
    thermal_controls.rz_centered_ideal_temperature_trial =
        implicit_step && !independent && model.electronThermodynamicsExecutor().isIdealGas();
#endif
    NativeResolvedThermalTrialBackend backend =
        [&] (amrex::MultiFab const& trial_temperature, amrex::MultiFab const& cell_temperature,
             amrex::MultiFab const& frame, amrex::MultiFab const& previous_work,
             NativeThermalCornerEmission const* emission,
             NativeResolvedThermalSourceResult& candidate, amrex::MultiFab& projected,
             std::string& reason) -> bool
    {
        if (!validate_duration(immutable, frozen_mask, reason)) { return false; }
        if (independent && independent->IsMixed() &&
            (!independent->Validate(*options.authenticated_history_step,
                 immutable, frozen_mask, reason) ||
             !independent->ValidatePacketFrame(frozen_rate, options.collisions.max_optical_depth,
                 options.thermal.packets.max_absorption_optical_depth, reason) ||
             !independent->ValidatePacketFrame(frame, options.collisions.max_optical_depth,
                 options.thermal.packets.max_absorption_optical_depth, reason)))
        { return false; }
        if (all_d &&
            (!options.all_d_no_k_event_witness->ValidateStep(*options.authenticated_history_step,
                 dt, options.collisions.seed, options.collisions.interval, immutable, frozen_mask,
                 reason) ||
             !Collective(frozen_clocks.empty()) ||
             !EmptyKFlags(kinetic, capture_all, reason) ||
             !ValidateAllDImplicitAbsorptionHistoryStep(*options.authenticated_history_step,
                 *options.all_d_no_k_event_witness, frozen_scattering, frozen_absorption,
                 frozen_rate, geometry, options.collisions.max_optical_depth,
                 options.thermal.packets.max_absorption_optical_depth, reason) ||
             !ValidateAllDImplicitAbsorptionHistoryStep(*options.authenticated_history_step,
                 *options.all_d_no_k_event_witness, frozen_scattering, frozen_absorption,
                 frame, geometry, options.collisions.max_optical_depth,
                 options.thermal.packets.max_absorption_optical_depth, reason)))
        { return false; }
        if (step_depth &&
            (!ValidatePureAbsorptionHistoryStep(*options.authenticated_history_step,
                 frozen_scattering, frozen_absorption, frozen_rate, geometry,
                 options.thermal.packets.max_absorption_optical_depth, reason) ||
             !ValidatePureAbsorptionHistoryStep(*options.authenticated_history_step,
                 frozen_scattering, frozen_absorption, frame, geometry,
                 options.thermal.packets.max_absorption_optical_depth, reason)))
        {
            return false;
        }
        PhotonPacketTransaction endpoint(live, immutable);
        amrex::MultiFab impulse(boxes, mapping, 3, 0), energy(boxes, mapping, 1, 0);
        amrex::MultiFab path_scale(boxes, mapping, 1, 0);
        NativePhotonPathSubsegmentReport path_report;
        bool path_valid = false;
        if (history)
        {
            path_valid = implicit_step
                ? continued_transport.ReplayAllDNoPhoton(frame, endpoint,
                      *options.authenticated_history_step, *options.all_d_no_k_event_witness,
                      impulse, energy, path_scale, path_report.path)
                : continued_transport.Replay(frame, endpoint.Particles(),
                                             impulse, energy, path_scale, path_report.path);
            if (path_valid)
            {
                std::vector<std::uint64_t> ids, keys, draws;
                bool routed = photon_replay_detail::CapturePhotonRoutingKeys(
                    endpoint.Particles(), live,
                    NativePhotonPathSnapshotPolicy::KeyedProvisional, ids, keys,
                    draws);
                if (!Collective(routed))
                {
                    reason = "invalid history path routing";
                    return false;
                }
                std::vector<NativePhotonPathCarry> previous;
                if (!photon_replay_detail::RouteKeyedPhotonCarries(
                        frozen_clocks, ids, keys,
                        options.collisions.max_events_per_photon, previous))
                {
                    reason = "history path lost original clocks";
                    return false;
                }
                amrex::Long events = 0, colliding = 0;
                for (std::size_t n = 0; n < previous.size(); ++n)
                {
                    auto const added =
                        path_report.path.photon_carries[n].completed_events -
                        previous[n].completed_events;
                    routed = routed && added >= 0;
                    events += added;
                    colliding += added > 0;
                }
                if (!Collective(routed))
                {
                    reason = "history event counter regressed";
                    return false;
                }
                amrex::ParallelDescriptor::ReduceLongSum(events);
                amrex::ParallelDescriptor::ReduceLongSum(colliding);
                path_report.path.collision_events = events;
                path_report.path.colliding_photons = colliding;
                path_report.photon_carries = path_report.path.photon_carries;
                path_report.subsegments = 1;
            }
        }
        else
        {
            path_valid = transport.Replay(frame, endpoint.Particles(), impulse,
                                          energy, path_scale, path_report);
        }
        if (!path_valid)
        {
            reason = path_report.path.reason;
            return false;
        }
        if (all_d)
        {
            bool const empty_path = EmptyKPathReport(path_report);
            bool const zero_kick = ExactZeroKAccount(impulse);
            bool const zero_energy = ExactZeroKAccount(energy);
            bool const zero_scale = ExactZeroKAccount(path_scale);
            bool const owners = options.all_d_no_k_event_witness->ValidateOwners(
                endpoint, frozen_mask, reason);
            if (!Collective(empty_path && zero_kick && zero_energy && zero_scale && owners))
            {
                reason = "all-D authentic path produced a photon, event or nonzero K account";
                return false;
            }
        }
        std::set<RayIdentity> physical_collisions;
        if (adaptive &&
            !PhysicalPathCensus(endpoint.Particles(), options.census,
                                physical_collisions))
        {
            reason = "invalid exact RZ physical path collision identities";
            return false;
        }
        auto const kick = Host(impulse), debit = Host(energy), scale = Host(path_scale);
        auto const velocity = Host(frame);
        NativePacketThermalEventReplay endpoint_replay;
        if (!endpoint_replay.Prepare(endpoint.Particles(), live, kinetic, frozen_absorption,
                                     frozen_rate, geometry, dt, options.thermal.packets,
                                     reason, &capture_all))
        {
            return false;
        }
        auto field_controls = options.field;
        field_controls.centered_thermal_difference =
            thermal_controls.rz_centered_ideal_temperature_trial;
        NativeResolvedRZThermalTrialInputs const inputs{
            live, endpoint, endpoint_replay, frozen_diffusion, frozen_mask, total, d_absorption,
            geometry, dt, thermal_controls, field_controls, options.authenticated_history_step,
            &impulse};
        if (!TryBuildNativeResolvedRZThermalTrial(
                inputs, trial_temperature, cell_temperature, frame, previous_work,
                emission, candidate, projected, reason))
        {
            return false;
        }
        if (all_d &&
            (!EmptyKFlags(kinetic, capture_all, reason) ||
             !EmptyKSourceOutputs(*options.all_d_no_k_event_witness, frozen_mask,
                 candidate, reason)))
        { return false; }
        if (adaptive)
        {
            if (!RetainPhysicalPathCensus(candidate.photons->Particles(),
                                          options.census, physical_collisions))
            {
                reason =
                    "RZ endpoint births lost physical-only census provenance";
                return false;
            }
        }
        auto physical = Host(candidate.physical_debit), raw = Host(candidate.raw_debit);
        auto packet = Host(candidate.rz_packet_debit), operands = Host(candidate.operand_scale);
        auto excluded = Host(candidate.rz_work_excluded_packet_debit);
        auto bound = Host(candidate.storage_roundoff_bound);
        for (amrex::MFIter it(raw); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                auto path_projection = debit[it](cell());
                for (int d = 0; d < 4; ++d)
                {
                    auto const value = d == 0 ? debit[it](cell())
                                             : PhysConst::c * kick[it](cell(), d - 1);
                    physical[it](cell(), d) += value;
                    raw[it](cell(), d) += value;
                    packet[it](cell(), d) += value;
                    // Momentum was combined before FIELD consumed its work.
                    // Complete only the ancillary energy component here.
                    if (d == 0)
                    {
                        excluded[it](cell(), d) += value;
                    }
                    operands[it](cell(), d) += scale[it](cell());
                    bound[it](cell(), d) += 4096 * std::numeric_limits<amrex::Real>::epsilon() *
                        scale[it](cell());
                    if (d > 0)
                    {
                        path_projection -= velocity[it](cell(), d - 1) *
                                           kick[it](cell(), d - 1);
                    }
                }
                projected[it](cell()) += path_projection;
            }
        }
        amrex::htod_memcpy(candidate.physical_debit, physical);
        amrex::htod_memcpy(candidate.raw_debit, raw);
        amrex::htod_memcpy(candidate.rz_packet_debit, packet);
        amrex::htod_memcpy(candidate.rz_work_excluded_packet_debit, excluded);
        amrex::htod_memcpy(candidate.operand_scale, operands);
        amrex::htod_memcpy(candidate.storage_roundoff_bound, bound);
        amrex::Gpu::streamSynchronize();
        accepted_path = std::move(path_report);
        return true;
    };
    auto candidate = TryAdvanceNativeResolvedThermalSource(
        live, immutable, material, path, frozen_diffusion, frozen_mask, frozen_temperature,
        frozen_absorption, frozen_rate, model, geometry, dt, thermal_controls, &backend);
    if (!candidate.valid)
    {
        rejected.reason = candidate.reason;
        return rejected;
    }
    if (all_d)
    {
        if (!Collective(candidate.material_stage != nullptr && candidate.photons != nullptr))
        {
            rejected.reason = "all-D accepted source lost private material/photon owners";
            return rejected;
        }
        amrex::MultiFab endpoint_rate;
        bool const rate_valid = candidate.material_stage->TryCandidateRateVelocity(endpoint_rate);
        if (!validate_duration(*candidate.photons, frozen_mask, rejected.reason))
        { return rejected; }
        if (!Collective(rate_valid) ||
            !options.all_d_no_k_event_witness->ValidateStep(*options.authenticated_history_step,
                dt, options.collisions.seed, options.collisions.interval, *candidate.photons,
                frozen_mask, rejected.reason) ||
            !ValidateAllDImplicitAbsorptionHistoryStep(*options.authenticated_history_step,
                *options.all_d_no_k_event_witness, frozen_scattering, frozen_absorption,
                endpoint_rate, geometry, options.collisions.max_optical_depth,
                options.thermal.packets.max_absorption_optical_depth, rejected.reason) ||
            !EmptyKSourceOutputs(*options.all_d_no_k_event_witness, frozen_mask,
                candidate, rejected.reason) || !EmptyKPathReport(accepted_path))
        { return rejected; }
    }
    if (independent)
    {
        amrex::MultiFab endpoint_rate;
        if (!candidate.material_stage->TryCandidateRateVelocity(endpoint_rate))
        {
            rejected.reason = "independent opacity lost actual material endpoint frame";
            return rejected;
        }
        if (independent->IsMixed() && !independent->ValidatePacketFrame(endpoint_rate,
                options.collisions.max_optical_depth,
                options.thermal.packets.max_absorption_optical_depth, rejected.reason))
        { return rejected; }
        // The common Source may invert to a slightly different accepted
        // nodal T. Rebuild its genuine reciprocal bath with the same
        // frozen caloric weights, then screen the actual stored post-kick
        // native rate, not merely the work-secant frame.
        auto const& weights = candidate.rz_independent_gray_corner_weights;
        if (!Collective(weights.ok() && weights.nComp() == 4 &&
                weights.boxArray() == boxes && weights.DistributionMap() == mapping))
        {
            rejected.reason = "independent-gray accepted source lost caloric weights";
            return rejected;
        }
        amrex::MultiFab bath(boxes, mapping, 1, 0);
        bath.setVal(0);
        if (!TryBuildNativeResolvedRZReciprocalBath(candidate.nodal_temperature, weights,
                options.thermal.packets.radiation_constant, bath))
        {
            rejected.reason = "invalid actual accepted independent-gray reciprocal bath";
            return rejected;
        }
        auto energy = amrex::MultiFab(boxes, mapping, 1, 0);
        auto momentum = amrex::MultiFab(boxes, mapping, 3, 0);
        amrex::MultiFab::Copy(energy, candidate.diffusion_radiation, 0, 0, 1, 0);
        amrex::MultiFab::Copy(momentum, candidate.diffusion_radiation, 1, 0, 3, 0);
        // Mixed FIELD owns only its consumed D work input (exact zero in K).
        // The common Source has already qualified the total finite native work;
        // passing it here would incorrectly assign packet work to K's empty D account.
        // Keep the original all-D endpoint screen unchanged.
        auto const& screened_work = independent->IsMixed()
            ? candidate.rz_diffusion_work_input : candidate.material_stage->RequestedWork();
        if (!independent->Preflight(*options.authenticated_history_step, *candidate.photons,
                energy, momentum, bath, frozen_mask, endpoint_rate,
                screened_work, rejected.reason))
        { return rejected; }
        candidate.rz_independent_gray_corner_weights.clear();
    }
    NativeResolvedRZThermalIntervalResult result;
    result.diffusion_wall_momentum_by_side = candidate.rz_axial_wall_momentum_by_side;
    result.diffusion_wall_momentum = candidate.rz_axial_wall_momentum;
    result.axial_wall_momentum_operand = accepted_path.path.axial_wall_momentum_operand +
        std::abs(result.diffusion_wall_momentum_by_side[0]) +
        std::abs(result.diffusion_wall_momentum_by_side[1]);
    std::array<amrex::Real, 3> representation_change{};
    if (adaptive)
    {
        // Source convergence may replace the trial T with the independently
        // inverted physical caloric candidate. Use THAT actual accepted T and
        // actual stored post-kick rate, not the work secant or cached trial
        // bath.
        amrex::MultiFab bath_bound(boxes, mapping, 1, 0), endpoint_rate;
        bath_bound.setVal(0);
        if (!TryBuildResolvedRZThermalBathUpperBound(
                candidate.nodal_temperature, bath_bound, geometry,
                options.thermal.packets.radiation_constant) ||
            !candidate.material_stage->TryCandidateRateVelocity(endpoint_rate))
        {
            rejected.reason =
                "invalid actual accepted RZ endpoint rate or corner bath bound";
            return rejected;
        }
        // Source/native finite work has already converged ONCE. Admission is
        // only a change of radiation representation, never another force or
        // thermal-D source. Keep the accepted source faces/debits untouched.
        PhotonPacketTransaction promotion(live, *candidate.photons);
        amrex::MultiFab e(boxes, mapping, 1, 1), q(boxes, mapping, 3, 1);
        e.setVal(0);
        q.setVal(0);
        amrex::MultiFab::Copy(e, candidate.diffusion_radiation, 0, 0, 1, 0);
        amrex::MultiFab::Copy(q, candidate.diffusion_radiation, 1, 0, 3, 0);
        auto converted = TryConvertAdaptivePacketDiffusionRepresentation(
            promotion, e, q, frozen_mask, frozen_desired, frozen_carry,
            geometry, options.census,
            LogicalPhotonBirthKeyContext{options.thermal.packets.seed,
                                         options.thermal.packets.interval},
            options.thermal.packets.photon_energy, options.admission);
        if (!converted.valid)
        {
            rejected.reason =
                "adaptive RZ thermal census transfer: " + converted.reason;
            return rejected;
        }
        bool const admissible =
            converted.promoted_cells == 0 ||
            AdmissibleResolvedRZThermalDiffusionCensus(
                converted.energy, converted.momentum, bath_bound, future_total,
                frozen_absorption, converted.mask, endpoint_rate, geometry, dt,
                options.field);
        result.accepted_diffusion_mask.define(boxes, mapping, 1,
                                              frozen_mask.nGrowVect());
        result.admitted_four_inventory.define(boxes, mapping, 4, 0);
        if (admissible)
        {
            if (!candidate.photons->AdoptCandidate(promotion))
            {
                rejected.reason =
                    "adaptive RZ promotion lost immutable photon provenance";
                return rejected;
            }
            amrex::MultiFab::Copy(candidate.diffusion_radiation,
                                  converted.energy, 0, 0, 1, 0);
            amrex::MultiFab::Copy(candidate.diffusion_radiation,
                                  converted.momentum, 0, 1, 3, 0);
            amrex::iMultiFab::Copy(result.accepted_diffusion_mask,
                                   converted.mask, 0, 0, 1,
                                   frozen_mask.nGrowVect());
            result.radiation_representation_carry =
                std::move(converted.representation_carry);
            result.admitted_four_inventory =
                std::move(converted.realized_promotion_inventory);
            result.promoted_cells = converted.promoted_cells;
            result.admitted_photons = converted.promoted_photons;
            result.deferred_cells = converted.deferred_cells;
            // Remeasure ACTUAL stored old/new carry. The transfer's expected
            // remainder is not authority to pretend an unrepresentable carry
            // addition was stored.
            auto const change = Host(result.radiation_representation_carry);
            auto const previous = Host(frozen_carry);
            auto const metric = RZMomentMetric(geometry);
            for (amrex::MFIter it(change); it.isValid(); ++it)
            {
                for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
                {
                    representation_change[0] +=
                        change[it](cell(), 0) - previous[it](cell(), 0);
                    representation_change[1] +=
                        change[it](cell(), 3) - previous[it](cell(), 3);
                    representation_change[2] +=
                        metric.Angular(cell()[0]).mean_radius *
                        (change[it](cell(), 2) - previous[it](cell(), 2));
                }
            }
            amrex::ParallelDescriptor::ReduceRealSum(
                representation_change.data(), 3);
        }
        else
        {
            // No inadmissible new D cohort is published: actual collided rays
            // remain kinetic, including their endpoint capture and native work.
            amrex::iMultiFab::Copy(result.accepted_diffusion_mask, frozen_mask,
                                   0, 0, 1, frozen_mask.nGrowVect());
            result.radiation_representation_carry = Copy(frozen_carry);
            result.admitted_four_inventory.setVal(0);
            result.deferred_cells =
                converted.deferred_cells + converted.promoted_cells;
        }
        result.accepted_diffusion_mask.FillBoundary(geometry.periodicity());
        result.radiation_representation_carry.FillBoundary(
            geometry.periodicity());
        candidate.diffusion_radiation.FillBoundary(geometry.periodicity());
    }
    auto const outgoing = RadiationInventory(candidate.photons->Particles(),
                                               candidate.diffusion_radiation, geometry);
    auto const actual = Host(candidate.material_stage->ConjugateImpulse());
    auto const carry = Host(candidate.material_stage->ConjugateMomentumCarryChange());
    auto const angular_bound = Host(candidate.material_stage->RZConjugateMomentumRoundoffBound());
    auto const storage = Host(candidate.storage_roundoff_bound);
    std::array<amrex::Real, 4> momentum{}; // cPz, cLz, and their native/storage arithmetic bounds
    auto const metric = RZMomentMetric(geometry);
    for (amrex::MFIter it(actual); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const rbar = metric.Angular(cell()[0]).mean_radius;
            momentum[0] += PhysConst::c * (actual[it](cell(), 2) + carry[it](cell(), 2));
            momentum[1] += PhysConst::c * rbar * (actual[it](cell(), 1) + carry[it](cell(), 1));
            momentum[2] += PhysConst::c * angular_bound[it](cell(), 2) + storage[it](cell(), 3);
            momentum[3] += rbar * (PhysConst::c * angular_bound[it](cell(), 1) +
                                   storage[it](cell(), 2));
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(momentum.data(), 4);
    auto const work = candidate.material_stage->ActualWork().sum(0) +
                      candidate.material_stage->EnergyCarryChange().sum(0);
    auto const heat = candidate.actual_heat.sum(0);
    result.raw_inventory_residual[0] =
        static_cast<amrex::Real>(incoming[0] - outgoing[0] - work - heat) -
        representation_change[0];
    result.raw_inventory_residual[1] =
        static_cast<amrex::Real>(incoming[1] - outgoing[1]) - momentum[0] -
        PhysConst::c * accepted_path.path.wall_momentum[2] -
        representation_change[1];
    if (options.reflecting_axial_fixed_mask_trial)
    {
        result.raw_inventory_residual[1] -= PhysConst::c * result.diffusion_wall_momentum;
    }
    result.raw_inventory_residual[2] =
        static_cast<amrex::Real>(incoming[2] - outgoing[2]) - momentum[1] -
        representation_change[2];
    auto const energy_scale =
        static_cast<amrex::Real>(std::abs(incoming[0]) + std::abs(outgoing[0])) +
        candidate.actual_heat.norm1(0) + candidate.material_stage->ActualWork().norm1(0) +
        candidate.material_stage->EnergyCarryChange().norm1(0);
    auto const tolerance = options.thermal.energy_tolerance;
    amrex::Real wall_arithmetic_bound = 0;
    if (options.reflecting_axial_fixed_mask_trial)
    {
        // A mirror/terminal correction costs <=16 operations (transfer,
        // signed net/side and absolute sum). Each native record costs <=16
        // terminal-account operations, each subsegment <=8 account additions;
        // 64 per cell and 16 per rank cover D-side integration/reduction.
        // Actual counts, never a measured residual or a fitted multiplier.
        auto const count = 256.L + 64.L * geometry.Domain().numPts() +
            16.L * amrex::ParallelDescriptor::NProcs() +
            16.L * accepted_path.path.replayed_photons +
            16.L * accepted_path.path.axial_wall_transfers + 8.L * accepted_path.subsegments;
        auto const neps = count * std::numeric_limits<amrex::Real>::epsilon();
        if (!Collective(std::isfinite(neps) && neps >= 0 && neps < .5 &&
                        accepted_path.path.replayed_photons >= 0 &&
                        accepted_path.path.axial_wall_transfers >= 0 &&
                        accepted_path.subsegments > 0 &&
                        std::isfinite(result.axial_wall_momentum_operand) &&
                        result.axial_wall_momentum_operand >= 0 &&
                        std::isfinite(result.diffusion_wall_momentum)))
        {
            return rejected;
        }
        wall_arithmetic_bound = neps / (1 - neps) * PhysConst::c *
                                result.axial_wall_momentum_operand;
        result.axial_wall_roundoff_bound = wall_arithmetic_bound;
    }
    auto const axial_allowance = options.reflecting_axial_fixed_mask_trial
        ? tolerance * energy_scale + momentum[2] + wall_arithmetic_bound
        : tolerance * energy_scale + momentum[2];
    valid = std::abs(result.raw_inventory_residual[0]) <=
        tolerance * energy_scale + candidate.energy_roundoff_bound &&
        std::abs(result.raw_inventory_residual[1]) <= axial_allowance &&
        std::abs(result.raw_inventory_residual[2]) <=
            tolerance * energy_scale * geometry.ProbHi(0) + momentum[3];
    for (auto const residual : result.raw_inventory_residual)
    {
        valid = valid && std::isfinite(residual);
    }
    if (!Collective(valid))
    {
        rejected.reason = "complete actual chronological RZ thermal E/Pz/Lz ledger";
        return rejected;
    }
    if (all_d &&
        (!options.all_d_no_k_event_witness->ValidateStep(*options.authenticated_history_step,
            dt, options.collisions.seed, options.collisions.interval, *candidate.photons,
            frozen_mask, rejected.reason) ||
         !EmptyKFlags(kinetic, capture_all, rejected.reason) ||
         !EmptyKSourceOutputs(*options.all_d_no_k_event_witness, frozen_mask,
             candidate, rejected.reason) || !EmptyKPathReport(accepted_path)))
    { return rejected; }
    if (implicit_step && !validate_duration(*candidate.photons, frozen_mask, rejected.reason))
    { return rejected; }
    result.candidate = std::move(candidate);
    result.path = std::move(accepted_path);
    result.valid = true;
    return result;
#endif
}
} // namespace warpx::radiation
