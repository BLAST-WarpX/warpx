/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativeResolvedThermalInterval.H"

#include "ConstitutiveDiffusionMomentum.H"
#include "FluxRelaxedPackets.H"
#include "PacketDiffusionInterfacePhotons.H"
#include "PhotonLogicalDrawKey.H"
#include "PureGrayAbsorptionSource.H"
#include "ResolvedDiffusionFront.H"
#include "WarpX.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <limits>

namespace warpx::radiation
{
namespace
{
amrex::MultiFab
HostCopy (amrex::MultiFab const& field)
{
    amrex::MultiFab result(field.boxArray(), field.DistributionMap(), field.nComp(),
                           field.nGrowVect(), amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::dtoh_memcpy(result, field);
    amrex::Gpu::streamSynchronize();
    return result;
}
amrex::MultiFab
DeviceCopy (amrex::MultiFab const& field)
{
    amrex::MultiFab result(field.boxArray(), field.DistributionMap(), field.nComp(),
                           field.nGrowVect());
    amrex::htod_memcpy(result, field);
    amrex::Gpu::streamSynchronize();
    return result;
}
amrex::MultiFab
Pinned (amrex::BoxArray const& boxes, amrex::DistributionMapping const& distribution,
        int components)
{
    amrex::MultiFab field(boxes, distribution, components, 0,
                          amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    field.setVal(0);
    amrex::Gpu::streamSynchronize();
    return field;
}
bool
Collective (bool value)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(value);
    return value;
}
} // namespace

NativeResolvedThermalIntervalResult
TryAdvanceNativeResolvedThermalInterval (
    WarpXParticleContainer& live, PhotonPacketTransaction const& start,
    ParticleImpulseMaterial& material, std::string const& path,
    amrex::MultiFab const& old_diffusion, amrex::iMultiFab const& desired_mask,
    amrex::MultiFab const& old_temperature, amrex::MultiFab const& scattering,
    amrex::MultiFab const& absorption, amrex::MultiFab const& initial_rate,
    HybridPICModel const& model, amrex::Geometry const& geometry, amrex::Real dt,
    NativeResolvedThermalIntervalOptions const& options)
{
    NativeResolvedThermalIntervalResult rejected;
    rejected.reason = "invalid native resolved thermal interval";
    if (!start.ValidateSnapshotFor(live))
    {
        return rejected;
    }
    auto const& boxes = old_diffusion.boxArray();
    auto const& distribution = old_diffusion.DistributionMap();
    bool valid = old_diffusion.nComp() == 4 && scattering.nComp() == 1 && absorption.nComp() == 1 &&
                 desired_mask.nComp() == 1 && desired_mask.boxArray() == boxes &&
                 desired_mask.DistributionMap() == distribution && initial_rate.nComp() == 3 &&
                 options.packet_subsegment_limit > 0 &&
                 options.collisions.seed == options.thermal.packets.seed &&
                 options.collisions.interval == options.thermal.packets.interval &&
                 !options.collisions.absorption && !options.collisions.emission &&
                 options.collisions.angular_law == NativePhotonCollisionAngularLaw::Isotropic &&
                 options.collisions.moving_cross_sections &&
                 options.thermal.packets.survival_weighted_emission &&
                 options.field.radiation_constant == options.thermal.packets.radiation_constant;
    amrex::Real minimum_spacing = geometry.CellSize(0);
    for (int d = 1; d < AMREX_SPACEDIM; ++d)
    {
        minimum_spacing = std::min(minimum_spacing, geometry.CellSize(d));
    }
    valid = valid && options.field.lte_centered_moving_source &&
            dt * PhysConst::c <= .5 * minimum_spacing;
    for (auto const component : options.census.initial_u)
    {
        valid = valid && component >= 0 && component < live.NumRealComps();
    }
    for (auto const* field : {&scattering, &absorption, &initial_rate})
    {
        valid = valid && field->boxArray() == boxes && field->DistributionMap() == distribution;
    }
    if (!Collective(valid))
    {
        return rejected;
    }
    auto old = HostCopy(old_diffusion), chi_s = HostCopy(scattering), chi_a = HostCopy(absorption);
    auto const rho =
        HostCopy(*WarpX::GetInstance().m_fields.get(warpx::fields::FieldType::rho_fp, 0));
    auto const rate_host = HostCopy(initial_rate);
    auto old_energy = Pinned(boxes, distribution, 1), old_q = Pinned(boxes, distribution, 3);
    auto total = Pinned(boxes, distribution, 1), scalar_absorption = Pinned(boxes, distribution, 1);
    auto future_total = Pinned(boxes, distribution, 1);
    amrex::iMultiFab mask(boxes, distribution, 1, 1,
                          amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::iMultiFab::Copy(mask, desired_mask, 0, 0, 1, 0);
    amrex::Gpu::streamSynchronize();
    for (amrex::MFIter it(old); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            bool vacant = true;
            for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner)
            {
                auto node = index;
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    node[d] += (corner >> d) & 1;
                }
                vacant = vacant && rho[it](node) == 0;
            }
            if (vacant)
            {
                valid = valid && options.thermal.allow_vacuum_noop && chi_s[it](index) == 0 &&
                        chi_a[it](index) == 0;
                for (int d = 0; d < 4; ++d)
                {
                    valid = valid && old[it](index, d) == 0;
                }
            }
            valid = valid && (mask[it](index) == 0 || mask[it](index) == 1) &&
                    chi_s[it](index) >= 0 && chi_a[it](index) >= 0 &&
                    std::isfinite(chi_s[it](index) + chi_a[it](index));
            old_energy[it](index) = old[it](index, 0);
            if (mask[it](index) == 1)
            {
                future_total[it](index) = chi_s[it](index) + chi_a[it](index);
            }
            bool empty = old[it](index, 0) == 0;
            for (int d = 0; d < 3; ++d)
            {
                old_q[it](index, d) = old[it](index, d + 1);
                empty = empty && old[it](index, d + 1) == 0;
            }
            if (mask[it](index) == 0)
            {
                valid = valid && empty;
            }
            if (empty)
            {
                mask[it](index) = 0;
            }
            if (mask[it](index) == 1)
            {
                FourVector incoming{};
                amrex::GpuArray<amrex::Real, 3> beta{};
                for (int d = 0; d < 4; ++d)
                {
                    incoming[d] = old[it](index, d);
                }
                for (int d = 0; d < 3; ++d)
                {
                    beta[d] = rate_host[it](index, d) / PhysConst::c;
                }
                valid = valid && TryPureGrayAbsorptionSource(incoming, beta, 0, 0).valid;
                total[it](index) = chi_s[it](index) + chi_a[it](index);
                scalar_absorption[it](index) = chi_a[it](index);
                valid = valid && dt * PhysConst::c * total[it](index) >= 8;
            }
        }
    }
    if (!Collective(valid))
    {
        return rejected;
    }
    mask.FillBoundary(geometry.periodicity());
    amrex::Gpu::streamSynchronize();
    auto device_old_energy = DeviceCopy(old_energy), device_old_q = DeviceCopy(old_q);
    auto device_total = DeviceCopy(total), device_absorption = DeviceCopy(scalar_absorption);
    auto device_future_total = DeviceCopy(future_total);
    auto prepared = std::make_unique<PhotonPacketTransaction>(live, start);
    auto const snapshot = SnapshotCollidedDiffusionAdmission(prepared->Particles(), options.census);
    if (!snapshot.valid)
    {
        rejected.reason = snapshot.reason;
        return rejected;
    }
    NativePhotonPathSubsegmentReplay transport;
    NativePhotonPathReport initial_report;
    if (!transport.Prepare(prepared->Particles(), live, scattering, initial_rate, geometry, dt,
                           options.collisions, options.packet_subsegment_limit, initial_report, {},
                           NativePhotonPathSnapshotPolicy::KeyedProvisional))
    {
        rejected.reason = initial_report.reason;
        return rejected;
    }
    amrex::GpuArray<amrex::Real, 4> initial_photons{};
    if (!detail::PhotonInventory(prepared->Particles(), initial_photons))
    {
        return rejected;
    }
    amrex::iMultiFab kinetic(boxes, distribution, 1, 0), capture_all(boxes, distribution, 1, 0);
    amrex::iMultiFab capture_none(boxes, distribution, 1, 0);
    capture_all.setVal(1);
    capture_none.setVal(0);
    for (amrex::MFIter it(mask); it.isValid(); ++it)
    {
        std::vector<int> flags(static_cast<std::size_t>(it.validbox().numPts()));
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            flags[static_cast<std::size_t>(it.validbox().index(cell()))] = 1 - mask[it](cell());
        }
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, flags.begin(), flags.end(),
                         kinetic[it].dataPtr());
    }
    amrex::MultiFab incoming_temperature;
    NativePhotonPathSubsegmentReport accepted_path;
    CollidedDiffusionAdmissionReport accepted_census;
    ResolvedThermalDiffusionFieldResult accepted_field;
    amrex::Long deferred_admissions = 0;
    NativeResolvedThermalTrialBackend backend =
        [&] (amrex::MultiFab const&, amrex::MultiFab const& cell_temperature,
             amrex::MultiFab const& frame, amrex::MultiFab const& previous_work,
             NativeThermalCornerEmission const* reciprocal,
             NativeResolvedThermalSourceResult& candidate, amrex::MultiFab& projected,
             std::string& reason) -> bool
    {
        if (!incoming_temperature.ok())
        {
            incoming_temperature = DeviceCopy(cell_temperature);
        }
        candidate.photons = std::make_unique<PhotonPacketTransaction>(live, *prepared);
        amrex::MultiFab collision_impulse(boxes, distribution, 3, 0),
            collision_energy(boxes, distribution, 1, 0);
        amrex::MultiFab collision_scale(boxes, distribution, 1, 0);
        NativePhotonPathSubsegmentReport path_report;
        if (!transport.Replay(frame, candidate.photons->Particles(), collision_impulse,
                              collision_energy, collision_scale, path_report))
        {
            reason = path_report.path.reason;
            return false;
        }
        auto capture_options = options.thermal.packets;
        capture_options.emission = false;
        NativePacketThermalEventReplay capture;
        if (!capture.Prepare(candidate.photons->Particles(), live, kinetic, absorption,
                             initial_rate, geometry, dt, capture_options, reason, &capture_all))
        {
            return false;
        }
        auto captured = capture.Replay(cell_temperature, frame, *candidate.photons);
        if (!captured.valid)
        {
            reason = captured.reason;
            return false;
        }
        // Measure the keyed thermal birth batch independently before solving D;
        // it is restaged identically AFTER census, so newborns cannot masquerade
        // as collided incoming cohorts in an eligible-but-empty D region.
        auto emission_probe = std::make_unique<PhotonPacketTransaction>(live, *candidate.photons);
        NativePacketThermalEventReplay emission;
        if (!emission.Prepare(emission_probe->Particles(), live, kinetic, absorption, initial_rate,
                              geometry, dt, options.thermal.packets, reason, &capture_none))
        {
            return false;
        }
        auto emitted = emission.Replay(cell_temperature, frame, *emission_probe, reciprocal);
        if (!emitted.valid)
        {
            reason = emitted.reason;
            return false;
        }
        auto packet = HostCopy(captured.radiation_debit),
             packet_scale = HostCopy(captured.operand_scale);
        auto const births = HostCopy(emitted.radiation_debit),
                   birth_scale = HostCopy(emitted.operand_scale);
        auto const kick = HostCopy(collision_impulse), coherent = HostCopy(collision_energy);
        auto const coherent_scale = HostCopy(collision_scale), velocity_host = HostCopy(frame);
        auto const work_host = HostCopy(previous_work);
        auto diffusion_work = Pinned(boxes, distribution, 1);
        for (amrex::MFIter it(packet); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                amrex::Real kinetic_work = 0;
                for (int d = 0; d < 4; ++d)
                {
                    packet[it](cell(), d) +=
                        births[it](cell(), d) +
                        (d == 0 ? coherent[it](cell()) : PhysConst::c * kick[it](cell(), d - 1));
                    packet_scale[it](cell(), d) +=
                        birth_scale[it](cell(), d) + coherent_scale[it](cell());
                    if (d > 0)
                    {
                        kinetic_work +=
                            velocity_host[it](cell(), d - 1) / PhysConst::c * packet[it](cell(), d);
                    }
                }
                if (mask[it](cell()) == 1)
                {
                    diffusion_work[it](cell()) = work_host[it](cell()) - kinetic_work;
                }
            }
        }
        auto field_options = options.field;
        field_options.linear.linear_tolerance = std::min(
            field_options.linear.linear_tolerance,
            .01 * std::min({options.thermal.material_tolerance, options.thermal.work_tolerance,
                            options.thermal.energy_tolerance}));
        amrex::MultiFab weights, trial_nodes, bath_reference, bath_difference, proper_bath;
        if (reciprocal)
        {
            weights = HostCopy(*reciprocal->weights);
            trial_nodes = HostCopy(*reciprocal->nodal_temperature);
            bath_reference = Pinned(boxes, distribution, 1);
            bath_difference = Pinned(boxes, distribution, 1);
            auto bath = Pinned(boxes, distribution, 1);
            for (amrex::MFIter it(bath); it.isValid(); ++it)
            {
                for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
                {
                    amrex::Real reference = std::numeric_limits<amrex::Real>::infinity();
                    for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner)
                    {
                        auto node = cell();
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            node[d] += (corner >> d) & 1;
                        }
                        if (weights[it](cell(), corner) > 0)
                        {
                            reference = std::min(reference, trial_nodes[it](node));
                        }
                    }
                    if (!std::isfinite(reference))
                    {
                        reference = 0;
                    }
                    auto const square = reference * reference;
                    amrex::Real difference = 0;
                    for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner)
                    {
                        auto node = cell();
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            node[d] += (corner >> d) & 1;
                        }
                        difference +=
                            weights[it](cell(), corner) *
                            ThermalFourthPowerDifference(trial_nodes[it](node), reference);
                    }
                    bath_reference[it](cell()) = reference;
                    bath_difference[it](cell()) = difference;
                    bath[it](cell()) =
                        field_options.radiation_constant * (square * square + difference);
                }
            }
            proper_bath = DeviceCopy(bath);
        }
        auto field = TryAdvanceResolvedThermalDiffusionField(
            device_old_energy, cell_temperature, device_total, device_absorption, mask, frame,
            diffusion_work, geometry, dt, field_options, &incoming_temperature,
            reciprocal ? &proper_bath : nullptr);
        if (!field.valid)
        {
            reason = field.reason;
            return false;
        }
        auto new_energy = HostCopy(field.energy), new_q = HostCopy(field.lab_momentum);
        bool positive_dipoles = true;
        for (amrex::MFIter it(new_energy); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                if (mask[it](cell()) != 1)
                {
                    continue;
                }
                FourVector state{new_energy[it](cell()), new_q[it](cell(), 0), new_q[it](cell(), 1),
                                 new_q[it](cell(), 2)};
                amrex::GpuArray<amrex::Real, 3> beta{};
                for (int d = 0; d < 3; ++d)
                {
                    beta[d] = velocity_host[it](cell(), d) / PhysConst::c;
                }
                positive_dipoles =
                    TryPureGrayAbsorptionSource(state, beta, 0, 0).valid && positive_dipoles;
            }
        }
        if (!Collective(positive_dipoles))
        {
            reason = "nonpositive accepted comoving D dipole";
            return false;
        }
        auto stress = HostCopy(field.outward_pressure_stress);
        auto physical = HostCopy(field.physical_energy_debit),
             raw_energy = HostCopy(field.raw_energy_debit);
        auto const heat = HostCopy(field.requested_heat), dscale = HostCopy(field.operand_scale);
        auto const dbound = HostCopy(field.storage_roundoff_bound);
        amrex::Array<detail::EmissionPlan, AMREX_SPACEDIM> leakage_plans;
        LogicalPhotonBirthKeyContext const birth_context{
            MixPhotonBirthKey(options.thermal.packets.seed ^ UINT64_C(0x54484d4c45414b31)),
            options.thermal.packets.interval};
        for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
        {
            auto const& faces = *field.face_leaked_energy[axis];
            amrex::MultiFab outgoing(faces.boxArray(), distribution, 4, faces.nGrowVect());
            if (!BuildPacketDiffusionLeakageEmissionPlan(
                    faces, mask, geometry, options.thermal.packets.photon_energy,
                    leakage_plans[axis], &outgoing, &frame, axis, &birth_context))
            {
                reason = "private thermal leakage plan";
                return false;
            }
            auto const actual = HostCopy(outgoing), pressure = HostCopy(*field.face_pressure[axis]);
            amrex::Real volume = 1;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                volume *= geometry.CellSize(d);
            }
            auto const face_weight = dt * volume / geometry.CellSize(axis);
            int const normal_component = AMREX_SPACEDIM == 1                ? 2
                                         : AMREX_SPACEDIM == 2 && axis == 1 ? 2
                                                                            : axis;
            for (amrex::MFIter it(stress); it.isValid(); ++it)
            {
                for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
                {
                    if (mask[it](cell()) != 1)
                    {
                        continue;
                    }
                    for (int side : {-1, 1})
                    {
                        auto neighbor = cell();
                        neighbor[axis] += side;
                        if (mask[it](neighbor) != 0)
                        {
                            continue;
                        }
                        auto face = cell();
                        face[axis] += side > 0 ? 1 : 0;
                        for (int d = 0; d < 3; ++d)
                        {
                            stress[it](cell(), d) +=
                                actual[it](face, d + 1) / PhysConst::c -
                                (d == normal_component ? side * face_weight * pressure[it](face)
                                                       : 0);
                        }
                    }
                }
            }
        }
        auto pre_admission = std::make_unique<PhotonPacketTransaction>(live, *candidate.photons);
        auto census =
            TryAdmitCollidedPhotons(candidate.photons->Particles(), options.census, new_energy,
                                    new_q, desired_mask, geometry, options.admission);
        if (!census.valid)
        {
            reason = census.reason;
            return false;
        }
        amrex::Long deferred = 0;
        if (census.admitted > 0)
        {
            bool screen = true;
            auto const trial_te = HostCopy(cell_temperature);
            amrex::Real volume = 1, length = 0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                volume *= geometry.CellSize(d);
                length = std::max(length, geometry.ProbHi(d) - geometry.ProbLo(d));
            }
            for (amrex::MFIter it(new_energy); it.isValid(); ++it)
            {
                for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
                {
                    if (new_energy[it](cell()) == 0)
                    {
                        continue;
                    }
                    amrex::Real beta_squared = 0;
                    for (int d = 0; d < 3; ++d)
                    {
                        beta_squared += velocity_host[it](cell(), d) *
                                        velocity_host[it](cell(), d) / PhysConst::c2;
                    }
                    auto const opacity = future_total[it](cell());
                    FourVector state{new_energy[it](cell()), new_q[it](cell(), 0),
                                     new_q[it](cell(), 1), new_q[it](cell(), 2)};
                    amrex::GpuArray<amrex::Real, 3> beta{};
                    for (int d = 0; d < 3; ++d)
                    {
                        beta[d] = velocity_host[it](cell(), d) / PhysConst::c;
                    }
                    screen = TryPureGrayAbsorptionSource(state, beta, 0, 0).valid && screen;
                    auto const t = trial_te[it](cell()), t2 = t * t;
                    auto const equilibrium = volume * options.field.radiation_constant * t2 * t2;
                    auto const scale = std::max(new_energy[it](cell()), equilibrium);
                    screen = screen && opacity > 0;
                    if (!options.field.lte_centered_moving_source && beta_squared > 0 &&
                        chi_a[it](cell()) > 0 && scale > 0)
                    {
                        screen = screen && chi_a[it](cell()) / opacity *
                                                   std::abs(new_energy[it](cell()) - equilibrium) /
                                                   scale <=
                                               1 / (opacity * length * opacity * length);
                    }
                }
            }
            screen = Collective(screen);
            bool const diffusion_screen = AdmissibleResolvedDiffusionCensus(
                new_energy, new_q, device_future_total, desired_mask, frame, geometry, dt,
                options.field.linear, true);
            if (!screen || !diffusion_screen)
            {
                deferred = census.admitted;
                candidate.photons = std::move(pre_admission);
                new_energy = HostCopy(field.energy);
                new_q = HostCopy(field.lab_momentum);
                census.retained += census.admitted;
                census.admitted = 0;
                census.actual_admitted_four_inventory = {0, 0, 0, 0};
                census.admitted_four_inventory.setVal(0);
            }
        }
        NativePacketThermalEventReplay final_emission;
        if (!final_emission.Prepare(candidate.photons->Particles(), live, kinetic, absorption,
                                    initial_rate, geometry, dt, options.thermal.packets, reason,
                                    &capture_none))
        {
            return false;
        }
        auto final_born =
            final_emission.Replay(cell_temperature, frame, *candidate.photons, reciprocal);
        if (!final_born.valid)
        {
            reason = final_born.reason;
            return false;
        }
        auto const actual_births = HostCopy(final_born.radiation_debit);
        auto const actual_birth_scale = HostCopy(final_born.operand_scale);
        bool same_births = true;
        for (amrex::MFIter it(packet); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                for (int d = 0; d < 4; ++d)
                {
                    auto const difference = actual_births[it](cell(), d) - births[it](cell(), d);
                    auto const bound =
                        128 * std::numeric_limits<amrex::Real>::epsilon() *
                        (birth_scale[it](cell(), d) + actual_birth_scale[it](cell(), d));
                    same_births = same_births && std::abs(difference) <= bound;
                    packet[it](cell(), d) += difference;
                    packet_scale[it](cell(), d) += actual_birth_scale[it](cell(), d);
                }
            }
        }
        if (!Collective(same_births))
        {
            reason = "keyed thermal birth remeasurement changed";
            return false;
        }
        for (auto const& leakage : leakage_plans)
        {
            if (!detail::StageEmissionPlan(*candidate.photons, leakage))
            {
                reason = "private thermal leakage staging";
                return false;
            }
        }
        auto debit = HostCopy(packet), raw = HostCopy(packet), scale = HostCopy(packet_scale);
        auto storage = Pinned(boxes, distribution, 4), radiation = Pinned(boxes, distribution, 4);
        auto const pre_census_q = HostCopy(field.lab_momentum);
        for (amrex::MFIter it(debit); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                auto projection = packet[it](cell(), 0);
                for (int d = 0; d < 3; ++d)
                {
                    projection -=
                        velocity_host[it](cell(), d) / PhysConst::c * packet[it](cell(), d + 1);
                }
                radiation[it](cell(), 0) = new_energy[it](cell());
                for (int d = 0; d < 3; ++d)
                {
                    radiation[it](cell(), d + 1) = new_q[it](cell(), d);
                }
                if (mask[it](cell()) == 1)
                {
                    amrex::Real mechanical_projection = 0;
                    // Use the PRE-census constitutive population. Admission
                    // is a pure representation transfer, never another kick.
                    for (int d = 0; d < 3; ++d)
                    {
                        auto const transfer = old_q[it](cell(), d) - pre_census_q[it](cell(), d) -
                                              PhysConst::c * stress[it](cell(), d);
                        debit[it](cell(), d + 1) += transfer;
                        raw[it](cell(), d + 1) += transfer;
                        scale[it](cell(), d + 1) += std::abs(old_q[it](cell(), d)) +
                                                    std::abs(pre_census_q[it](cell(), d)) +
                                                    PhysConst::c * std::abs(stress[it](cell(), d));
                        mechanical_projection +=
                            velocity_host[it](cell(), d) / PhysConst::c * transfer;
                    }
                    projection +=
                        heat[it](cell()) + (diffusion_work[it](cell()) - mechanical_projection);
                    debit[it](cell(), 0) += physical[it](cell());
                    raw[it](cell(), 0) += raw_energy[it](cell());
                    scale[it](cell(), 0) += dscale[it](cell());
                    storage[it](cell(), 0) = dbound[it](cell());
                }
                projected[it](cell()) = projection;
            }
        }
        candidate.diffusion_radiation = DeviceCopy(radiation);
        candidate.physical_debit = DeviceCopy(debit);
        candidate.raw_debit = DeviceCopy(raw);
        candidate.operand_scale = DeviceCopy(scale);
        candidate.storage_roundoff_bound = DeviceCopy(storage);
        if (reciprocal)
        {
            auto const corner_births = HostCopy(final_born.corner_emission_debit);
            auto offsets = Pinned(boxes, distribution, 1 << AMREX_SPACEDIM);
            bool balanced_origins = true;
            auto const dx = geometry.CellSizeArray();
            auto const volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
            for (amrex::MFIter it(offsets); it.isValid(); ++it)
            {
                for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
                {
                    amrex::GpuArray<amrex::Real, 3> beta{};
                    for (int d = 0; d < 3; ++d)
                    {
                        beta[d] = velocity_host[it](cell(), d) / PhysConst::c;
                    }
                    StressEnergy moment{}, source{};
                    balanced_origins =
                        BuildIsotropicDipoleFrameMatrices(beta, moment, source) && balanced_origins;
                    amrex::Real sum = 0, origin_scale = actual_birth_scale[it](cell(), 0);
                    for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner)
                    {
                        auto node = cell();
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            node[d] += (corner >> d) & 1;
                        }
                        auto const weight = weights[it](cell(), corner);
                        auto const direct_birth = corner_births[it](cell(), 4 * corner);
                        auto const mean_birth = weight * actual_births[it](cell(), 0);
                        auto const origin_bath_difference =
                            field_options.radiation_constant *
                            (ThermalFourthPowerDifference(trial_nodes[it](node),
                                                          bath_reference[it](cell())) -
                             bath_difference[it](cell()));
                        auto const diffusion_origin = mask[it](cell()) == 1
                                                          ? -weight * PhysConst::c * dt *
                                                                chi_a[it](cell()) * moment[0][0] *
                                                                volume * origin_bath_difference
                                                          : 0;
                        offsets[it](cell(), corner) =
                            (direct_birth - mean_birth) + diffusion_origin;
                        sum += offsets[it](cell(), corner);
                        origin_scale += std::abs(direct_birth) + std::abs(mean_birth) +
                                        std::abs(diffusion_origin);
                    }
                    balanced_origins =
                        balanced_origins && std::isfinite(sum + origin_scale) &&
                        std::abs(sum) <=
                            128 * std::numeric_limits<amrex::Real>::epsilon() * origin_scale;
                }
            }
            if (!Collective(balanced_origins))
            {
                reason = "reciprocal source origin arithmetic or frame failed";
                return false;
            }
            candidate.reciprocal_corner_heat_offset = DeviceCopy(offsets);
        }
        accepted_path = std::move(path_report);
        accepted_census = std::move(census);
        accepted_field = std::move(field);
        deferred_admissions = deferred;
        return true;
    };
    auto candidate = TryAdvanceNativeResolvedThermalSource(
        live, start, material, path, old_diffusion, mask, old_temperature, absorption, initial_rate,
        model, geometry, dt, options.thermal, &backend);
    if (!candidate.valid)
    {
        rejected.reason = candidate.reason;
        return rejected;
    }
    amrex::GpuArray<amrex::Real, 4> final_photons{};
    if (!detail::PhotonInventory(candidate.photons->Particles(), final_photons))
    {
        return rejected;
    }
    NativeResolvedThermalIntervalResult result;
    auto const old_radiation_energy = initial_photons[0] + old_diffusion.sum(0);
    auto const new_radiation_energy = final_photons[0] + candidate.diffusion_radiation.sum(0);
    result.raw_energy_residual = old_radiation_energy - new_radiation_energy -
                                 candidate.material_stage->ActualWork().sum(0) -
                                 candidate.material_stage->EnergyCarryChange().sum(0) -
                                 candidate.actual_heat.sum(0);
    auto const energy_scale = std::abs(old_radiation_energy) + std::abs(new_radiation_energy) +
                              candidate.actual_heat.norm1(0) +
                              candidate.material_stage->ActualWork().norm1(0) +
                              candidate.material_stage->EnergyCarryChange().norm1(0);
    valid = std::isfinite(result.raw_energy_residual) &&
            std::abs(result.raw_energy_residual) <=
                options.thermal.energy_tolerance * energy_scale + candidate.energy_roundoff_bound;
    for (int d = 0; d < 3; ++d)
    {
        result.raw_momentum_residual[d] =
            (initial_photons[d + 1] - final_photons[d + 1] + old_diffusion.sum(d + 1) -
             candidate.diffusion_radiation.sum(d + 1)) /
                PhysConst::c -
            candidate.material_stage->ActualImpulse().sum(d) -
            candidate.material_stage->MomentumCarryChange().sum(d);
        auto const momentum_scale = energy_scale / PhysConst::c +
                                    candidate.material_stage->ActualImpulse().norm1(d) +
                                    candidate.material_stage->MomentumCarryChange().norm1(d);
        valid = valid && std::isfinite(result.raw_momentum_residual[d]) &&
                std::abs(result.raw_momentum_residual[d]) <=
                    options.thermal.energy_tolerance * momentum_scale;
    }
    if (!Collective(valid))
    {
        rejected.reason = "complete actual native thermal interval E/P ledger";
        return rejected;
    }
    result.candidate = std::move(candidate);
    result.field = std::move(accepted_field);
    result.path = std::move(accepted_path);
    result.census = std::move(accepted_census);
    result.deferred_admissions = deferred_admissions;
    result.valid = true;
    return result;
}
} // namespace warpx::radiation
