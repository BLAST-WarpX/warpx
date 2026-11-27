/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativePhotonSubstepTestSupport.H"

namespace
{
using namespace warpx::radiation;
History
RunManual (WarpXParticleContainer& photons, amrex::MultiFab const& opacity,
           amrex::MultiFab const& initial_rate_velocity,
           amrex::MultiFab const& work_velocity, amrex::Geometry const& geometry,
           NativePhotonCollisionOptions const& options,
           std::vector<amrex::Real> const& segment_lengths,
           int real_sentinel, int int_sentinel)
{
    AMREX_ALWAYS_ASSERT(!segment_lengths.empty());
    auto const& boxes = photons.ParticleBoxArray(0);
    auto const& distribution = photons.ParticleDistributionMap(0);
    PhotonPacketTransaction trial(photons);
    auto const live_before = LocalRecords(photons);
    History history;
    AllocateHistoryFields(history, boxes, distribution);
    amrex::MultiFab segment_impulse(boxes, distribution, 3, 0);
    amrex::MultiFab segment_debit(boxes, distribution, 1, 0);
    amrex::MultiFab segment_scale(boxes, distribution, 1, 0);
    std::vector<NativePhotonPathCarry> carries;
    for (std::size_t segment = 0; segment < segment_lengths.size(); ++segment)
    {
        NativePhotonPathReplay path;
        NativePhotonPathReport report;
        auto const dt = segment_lengths[segment] / PhysConst::c;
        bool const prepared = segment == 0 ?
            path.Prepare(trial.Particles(), photons, opacity,
                initial_rate_velocity, geometry, dt, options, report) :
            path.Prepare(trial.Particles(), photons, opacity,
                initial_rate_velocity, geometry, dt, options, carries, report);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(prepared, report.reason.c_str());
        segment_impulse.setVal(0);
        segment_debit.setVal(0);
        segment_scale.setVal(0);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(path.Replay(work_velocity,
            trial.Particles(), segment_impulse, segment_debit, segment_scale,
            report), report.reason.c_str());
        AccumulatePathReport(history, report, segment == 0);
        amrex::MultiFab::Add(*history.impulse_field, segment_impulse, 0, 0, 3, 0);
        amrex::MultiFab::Add(*history.debit_field, segment_debit, 0, 0, 1, 0);
        amrex::MultiFab::Add(*history.scale_field, segment_scale, 0, 0, 1, 0);
        carries = std::move(report.photon_carries);
    }
    history.photon = ReadOne(trial.Particles(), real_sentinel, int_sentinel);
    history.carry = ReadCarry(carries, history.photon.idcpu);
    for (int d = 0; d < 3; ++d)
    {
        history.impulse[static_cast<std::size_t>(d)] =
            history.impulse_field->sum(d);
    }
    history.debit = history.debit_field->sum(0);
    AMREX_ALWAYS_ASSERT(SameRecords(live_before, LocalRecords(photons)));
    return history;
}

History
RunHelper (WarpXParticleContainer& photons, amrex::MultiFab const& opacity,
           amrex::MultiFab const& initial_rate_velocity,
           amrex::MultiFab const& work_velocity, amrex::Geometry const& geometry,
           NativePhotonCollisionOptions const& options, amrex::Real path_length,
           int segment_limit, int real_sentinel, int int_sentinel)
{
    auto const& boxes = photons.ParticleBoxArray(0);
    auto const& distribution = photons.ParticleDistributionMap(0);
    NativePhotonPathSubsegmentReplay replay;
    NativePhotonPathReport prepared;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(replay.Prepare(photons, opacity,
        initial_rate_velocity, geometry, path_length / PhysConst::c, options,
        segment_limit, prepared), prepared.reason.c_str());
    PhotonPacketTransaction trial(photons);
    auto const live_before = LocalRecords(photons);
    History history;
    AllocateHistoryFields(history, boxes, distribution);
    NativePhotonPathSubsegmentReport report;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(replay.Replay(work_velocity,
        trial.Particles(), *history.impulse_field, *history.debit_field,
        *history.scale_field, report), report.path.reason.c_str());
    history.photon = ReadOne(trial.Particles(), real_sentinel, int_sentinel);
    history.carry = ReadCarry(report.photon_carries, history.photon.idcpu);
    history.subsegments = report.subsegments;
    history.events = report.path.collision_events;
    history.face_crossings = report.path.face_crossings;
    for (int component = 0; component < 4; ++component)
    {
        auto const index = static_cast<std::size_t>(component);
        history.incoming[index] = report.path.incoming_inventory[component];
        history.outgoing[index] = report.path.outgoing_inventory[component];
    }
    for (int d = 0; d < 3; ++d)
    {
        history.impulse[static_cast<std::size_t>(d)] =
            history.impulse_field->sum(d);
    }
    history.debit = history.debit_field->sum(0);
    AMREX_ALWAYS_ASSERT(SameRecords(live_before, LocalRecords(photons)));
    return history;
}

void
CheckFourLedger (PhotonState const& initial, History const& history)
{
    auto const before = PacketConversionFourMomentumFromPhoton(
        initial.real[weight], initial.real[x_momentum],
        initial.real[y_momentum], initial.real[z_momentum]);
    auto const after = PacketConversionFourMomentumFromPhoton(
        history.photon.real[weight], history.photon.real[x_momentum],
        history.photon.real[y_momentum], history.photon.real[z_momentum]);
    AMREX_ALWAYS_ASSERT(before.valid && after.valid);
    for (int component = 0; component < 4; ++component)
    {
        auto const index = static_cast<std::size_t>(component);
        AMREX_ALWAYS_ASSERT(Close(history.incoming[index], before.value[index],
            before.value[0]));
        AMREX_ALWAYS_ASSERT(Close(history.outgoing[index], after.value[index],
            before.value[0]));
    }
    AMREX_ALWAYS_ASSERT(Close(history.debit,
        before.value[0] - after.value[0], before.value[0]));
    for (int d = 0; d < 3; ++d)
    {
        AMREX_ALWAYS_ASSERT(Close(PhysConst::c * history.impulse[d],
            before.value[d + 1] - after.value[d + 1], before.value[0]));
    }
}

void
CheckSimultaneousHit (PhotonState const& photon,
                      std::array<amrex::Real, 3> const& offsets)
{
    auto const four = PacketConversionFourMomentumFromPhoton(
        photon.real[weight], photon.real[x_momentum],
        photon.real[y_momentum], photon.real[z_momentum]);
    AMREX_ALWAYS_ASSERT(four.valid);
    amrex::Real reference_distance = -1;
    for (int d = 0; d < 3; ++d)
    {
        if (offsets[static_cast<std::size_t>(d)] == 0) { continue; }
        auto const direction = four.value[static_cast<std::size_t>(d + 1)] /
            four.value[0];
        auto const distance = offsets[static_cast<std::size_t>(d)] / direction;
        if (reference_distance < 0)
        {
            reference_distance = distance;
        }
        else
        {
            AMREX_ALWAYS_ASSERT(Close(distance, reference_distance,
                                      reference_distance));
        }
    }
    AMREX_ALWAYS_ASSERT(reference_distance > 0);
}

void
CompareHistories (History const& first, History const& second)
{
    AMREX_ALWAYS_ASSERT(first.photon.idcpu == second.photon.idcpu);
    AMREX_ALWAYS_ASSERT(first.photon.owner == second.photon.owner);
    AMREX_ALWAYS_ASSERT(first.photon.box == second.photon.box);
    AMREX_ALWAYS_ASSERT(first.photon.int_sentinel == second.photon.int_sentinel);
    AMREX_ALWAYS_ASSERT(first.photon.real[weight] == second.photon.real[weight]);
    AMREX_ALWAYS_ASSERT(first.photon.real[real_sentinel_value] ==
                        second.photon.real[real_sentinel_value]);
    AMREX_ALWAYS_ASSERT(first.carry.idcpu == second.carry.idcpu);
    AMREX_ALWAYS_ASSERT(first.carry.next_draw == second.carry.next_draw);
    AMREX_ALWAYS_ASSERT(first.carry.completed_events ==
                        second.carry.completed_events);
    AMREX_ALWAYS_ASSERT(first.events == second.events);
    AMREX_ALWAYS_ASSERT(first.face_crossings == second.face_crossings);
    auto const energy_scale = std::max(std::abs(first.incoming[0]),
                                       std::abs(second.incoming[0]));
    for (int component = 0; component < 8; ++component)
    {
        auto const scale = component < 3 ? 1.0_rt :
            component < 6 ? PhysConst::c : component == weight ? 1.0_rt : 20.0_rt;
        AMREX_ALWAYS_ASSERT(Close(first.photon.real[component],
            second.photon.real[component], scale));
    }
    AMREX_ALWAYS_ASSERT(Close(first.carry.residual_optical_depth,
        second.carry.residual_optical_depth,
        std::max(first.carry.residual_optical_depth,
                 second.carry.residual_optical_depth)));
    AMREX_ALWAYS_ASSERT(Close(first.debit, second.debit, energy_scale));
    for (int component = 0; component < 4; ++component)
    {
        AMREX_ALWAYS_ASSERT(Close(first.incoming[component],
            second.incoming[component], energy_scale));
        AMREX_ALWAYS_ASSERT(Close(first.outgoing[component],
            second.outgoing[component], energy_scale));
    }
    for (int d = 0; d < 3; ++d)
    {
        auto const impulse_scale = energy_scale / PhysConst::c;
        AMREX_ALWAYS_ASSERT(Close(first.impulse[d], second.impulse[d], impulse_scale));
        amrex::MultiFab difference(first.impulse_field->boxArray(),
            first.impulse_field->DistributionMap(), 1, 0);
        amrex::MultiFab::Copy(difference, *first.impulse_field, d, 0, 1, 0);
        amrex::MultiFab::Subtract(difference, *second.impulse_field, d, 0, 1, 0);
        AMREX_ALWAYS_ASSERT(difference.norm0() <=
            2048 * std::numeric_limits<amrex::Real>::epsilon() * impulse_scale);
    }
    amrex::MultiFab debit_difference(first.debit_field->boxArray(),
        first.debit_field->DistributionMap(), 1, 0);
    amrex::MultiFab::Copy(debit_difference, *first.debit_field, 0, 0, 1, 0);
    amrex::MultiFab::Subtract(debit_difference, *second.debit_field, 0, 0, 1, 0);
    AMREX_ALWAYS_ASSERT(debit_difference.norm0() <=
        2048 * std::numeric_limits<amrex::Real>::epsilon() * energy_scale);
}

void
CheckOwner (PhotonState const& state, amrex::BoxArray const& boxes,
            amrex::DistributionMapping const& distribution,
            amrex::IntVect const& cell)
{
    int box = -1;
    for (int index = 0; index < boxes.size(); ++index)
    {
        if (boxes[index].contains(cell)) { box = index; }
    }
    AMREX_ALWAYS_ASSERT(box >= 0);
    AMREX_ALWAYS_ASSERT(state.box == box);
    AMREX_ALWAYS_ASSERT(state.owner == distribution[box]);
}

amrex::Real
CellValue (amrex::MultiFab const& field, amrex::IntVect const& cell, int component)
{
    amrex::Real result = 0;
    for (amrex::MFIter iterator(field); iterator.isValid(); ++iterator)
    {
        if (!iterator.validbox().contains(cell)) { continue; }
        auto const& fab = field[iterator];
        auto const offset = fab.box().index(cell) +
            static_cast<amrex::Long>(component) * fab.box().numPts();
        // The oracle executes on the host; a CUDA FAB is device-resident.
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,
            fab.dataPtr() + offset, fab.dataPtr() + offset + 1, &result);
    }
    amrex::ParallelDescriptor::ReduceRealSum(result);
    return result;
}

void
CheckTransparentCommonTime (WarpXParticleContainer& photons,
                            amrex::Geometry const& geometry,
                            int real_sentinel, int int_sentinel)
{
    auto const& boxes = photons.ParticleBoxArray(0);
    auto const& distribution = photons.ParticleDistributionMap(0);
    auto const initial = ReadOne(photons, real_sentinel, int_sentinel);
    auto const idcpu = initial.idcpu;
    NativePhotonCollisionOptions options;
    options.seed = 8127;
    options.interval = 41;
    auto const tau = ExpectedDepth(options, idcpu, 0);
    auto const path_length = 1.99_rt;
    amrex::MultiFab opacity(boxes, distribution, 1, 0);
    amrex::MultiFab initial_velocity(boxes, distribution, 3, 0);
    amrex::MultiFab work_velocity(boxes, distribution, 3, 0);
    opacity.setVal(0.25_rt * tau / path_length);
    initial_velocity.setVal(0);
    work_velocity.setVal(0);
    options.max_events_per_photon = 64;
    options.max_optical_depth = 1.e6_rt;
    // The frozen homogeneous rate consumes exactly one quarter of the first
    // keyed exponential draw over the common physical interval.
    AMREX_ALWAYS_ASSERT(Close(opacity.norm0(), 0.25_rt * tau / path_length,
                              0.25_rt * tau / path_length));
    auto const helper = RunHelper(photons, opacity, initial_velocity,
        work_velocity, geometry, options, path_length, 40,
        real_sentinel, int_sentinel);
    auto const forty = RunManual(photons, opacity, initial_velocity,
        work_velocity, geometry, options,
        std::vector<amrex::Real>(40, path_length / 40),
        real_sentinel, int_sentinel);
    auto const forty_eight = RunManual(photons, opacity, initial_velocity,
        work_velocity, geometry, options,
        std::vector<amrex::Real>(48, path_length / 48),
        real_sentinel, int_sentinel);
    AMREX_ALWAYS_ASSERT(helper.subsegments == 32);
    CompareHistories(helper, forty);
    CompareHistories(helper, forty_eight);
    PrintHistory("flight", helper);
    AMREX_ALWAYS_ASSERT(helper.events == 0);
    AMREX_ALWAYS_ASSERT(helper.carry.completed_events == 0);
    AMREX_ALWAYS_ASSERT(helper.carry.next_draw == 1);
    AMREX_ALWAYS_ASSERT(Close(helper.carry.residual_optical_depth,
        0.75_rt * tau, tau));
    AMREX_ALWAYS_ASSERT(helper.debit == 0);
    auto const direction_norm = std::sqrt(
        initial.real[x_momentum] * initial.real[x_momentum] +
        initial.real[y_momentum] * initial.real[y_momentum] +
        initial.real[z_momentum] * initial.real[z_momentum]);
    for (int d = 0; d < 3; ++d)
    {
        AMREX_ALWAYS_ASSERT(helper.impulse[d] == 0);
        AMREX_ALWAYS_ASSERT(helper.photon.real[x_momentum + d] ==
            initial.real[x_momentum + d]);
        auto const unwrapped = initial.real[d] + path_length *
            initial.real[x_momentum + d] / direction_norm;
        auto expected = std::fmod(unwrapped, 1.0_rt);
        if (expected < 0) { expected += 1.0_rt; }
        AMREX_ALWAYS_ASSERT(Close(helper.photon.real[d], expected, 1.0_rt));
    }
    CheckFourLedger(initial, helper);
    CheckFourLedger(initial, forty);
    CheckFourLedger(initial, forty_eight);
    char const* checkpoint = "native_3d_substep_checkpoint";
    photons.Checkpoint(checkpoint, "photons");
    photons.clearParticles();
    photons.Restart(checkpoint, "photons");
    photons.Redistribute();
    auto const restarted = ReadOne(photons, real_sentinel, int_sentinel);
    AMREX_ALWAYS_ASSERT(restarted.idcpu == initial.idcpu);
    AMREX_ALWAYS_ASSERT(restarted.real == initial.real);
    AMREX_ALWAYS_ASSERT(restarted.int_sentinel == initial.int_sentinel);
    auto const resumed = RunHelper(photons, opacity, initial_velocity,
        work_velocity, geometry, options, path_length, 40,
        real_sentinel, int_sentinel);
    CompareHistories(helper, resumed);
    amrex::Print() << "3D homogeneous flight common-time partitions passed: "
                   << "helper/40/48 segments, crossings="
                   << helper.face_crossings
                   << "; checkpoint replay matched\n";
}

void
CheckAxisEdgeCornerAndTangent (WarpXParticleContainer& photons,
                               amrex::Geometry const& geometry,
                               int real_sentinel, int int_sentinel)
{
    auto const& boxes = photons.ParticleBoxArray(0);
    auto const& distribution = photons.ParticleDistributionMap(0);
    amrex::MultiFab opacity(boxes, distribution, 1, 0);
    amrex::MultiFab initial_velocity(boxes, distribution, 3, 0);
    amrex::MultiFab work_velocity(boxes, distribution, 3, 0);
    opacity.setVal(0);
    initial_velocity.setVal(0);
    work_velocity.setVal(0);
    NativePhotonCollisionOptions options;
    options.seed = 19;
    options.interval = 8;
    options.max_optical_depth = 64;

    std::array<amrex::Real, 3> const start{0.4375_rt, 0.375_rt, 0.375_rt};
    std::array<amrex::Real, 3> const axis_distances{0.0625_rt, 0.125_rt, 0.125_rt};
    for (int axis = 0; axis < 3; ++axis)
    {
        SetPhoton(photons, real_sentinel, int_sentinel, start,
            {axis == 0 ? 1.0_rt : 0.0_rt,
             axis == 1 ? 1.0_rt : 0.0_rt,
             axis == 2 ? 1.0_rt : 0.0_rt});
        auto const initial = ReadOne(photons, real_sentinel, int_sentinel);
        auto history = RunHelper(photons, opacity, initial_velocity, work_velocity,
            geometry, options, axis_distances[static_cast<std::size_t>(axis)],
            8, real_sentinel, int_sentinel);
        AMREX_ALWAYS_ASSERT(history.events == 0 && history.face_crossings == 1);
        AMREX_ALWAYS_ASSERT(history.photon.real[static_cast<std::size_t>(axis)] == 0.5_rt);
        for (int d = 0; d < 3; ++d)
        {
            if (d != axis)
            {
                AMREX_ALWAYS_ASSERT(history.photon.real[static_cast<std::size_t>(d)] ==
                    initial.real[static_cast<std::size_t>(d)]);
            }
        }
        CheckFourLedger(initial, history);
        if (axis == 0)
        {
            CheckOwner(history.photon, boxes, distribution,
                amrex::IntVect(AMREX_D_DECL(4, 1, 1)));
        }
    }

    auto const diagonal_start = start;
    SetPhoton(photons, real_sentinel, int_sentinel, diagonal_start,
              {1.0_rt/3, 2.0_rt/3, 2.0_rt/3});
    auto const corner_initial = ReadOne(photons, real_sentinel, int_sentinel);
    CheckSimultaneousHit(corner_initial, {0.0625_rt, 0.125_rt, 0.125_rt});
    auto const corner_length = 0.2_rt;
    auto const corner = RunHelper(photons, opacity, initial_velocity,
        work_velocity, geometry, options, corner_length, 8,
        real_sentinel, int_sentinel);
    AMREX_ALWAYS_ASSERT(corner.subsegments == 4);
    AMREX_ALWAYS_ASSERT(corner.face_crossings == 3);
    AMREX_ALWAYS_ASSERT(Close(corner.photon.real[0],
        diagonal_start[0] + corner_length / 3, 1));
    AMREX_ALWAYS_ASSERT(Close(corner.photon.real[1],
        diagonal_start[1] + 2 * corner_length / 3, 1));
    AMREX_ALWAYS_ASSERT(Close(corner.photon.real[2],
        diagonal_start[2] + 2 * corner_length / 3, 1));
    CheckOwner(corner.photon, boxes, distribution,
        amrex::IntVect(AMREX_D_DECL(4, 2, 2)));
    CheckFourLedger(corner_initial, corner);

    SetPhoton(photons, real_sentinel, int_sentinel, diagonal_start,
              {1.0_rt/std::sqrt(5.0_rt), 2.0_rt/std::sqrt(5.0_rt), 0});
    auto const edge_initial = ReadOne(photons, real_sentinel, int_sentinel);
    CheckSimultaneousHit(edge_initial, {0.0625_rt, 0.125_rt, 0});
    auto const edge_length = 0.16_rt;
    auto const edge = RunHelper(photons, opacity, initial_velocity,
        work_velocity, geometry, options, edge_length, 8,
        real_sentinel, int_sentinel);
    AMREX_ALWAYS_ASSERT(edge.face_crossings == 2);
    AMREX_ALWAYS_ASSERT(Close(edge.photon.real[0],
        diagonal_start[0] + edge_length / std::sqrt(5.0_rt), 1));
    AMREX_ALWAYS_ASSERT(Close(edge.photon.real[1],
        diagonal_start[1] + 2 * edge_length / std::sqrt(5.0_rt), 1));
    AMREX_ALWAYS_ASSERT(edge.photon.real[2] == 0.375_rt);
    CheckOwner(edge.photon, boxes, distribution,
        amrex::IntVect(AMREX_D_DECL(4, 2, 1)));
    CheckFourLedger(edge_initial, edge);

    auto const epsilon = 1.e-13_rt;
    SetPhoton(photons, real_sentinel, int_sentinel, start,
              {std::sqrt(1 - epsilon * epsilon), epsilon, 0});
    auto const tangent_initial = ReadOne(photons, real_sentinel, int_sentinel);
    auto const tangent = RunHelper(photons, opacity, initial_velocity,
        work_velocity, geometry, options, 0.25_rt, 8,
        real_sentinel, int_sentinel);
    AMREX_ALWAYS_ASSERT(tangent.face_crossings == 2);
    AMREX_ALWAYS_ASSERT(tangent.photon.real[y_position] >
                        tangent_initial.real[y_position]);
    AMREX_ALWAYS_ASSERT(tangent.photon.real[y_position] < 0.5_rt);
    CheckFourLedger(tangent_initial, tangent);
}

void
CheckPartitionSideScattering (WarpXParticleContainer& photons,
                              amrex::Geometry const& geometry,
                              int real_sentinel, int int_sentinel,
                              bool event_after_partition)
{
    auto const& boxes = photons.ParticleBoxArray(0);
    auto const& distribution = photons.ParticleDistributionMap(0);
    NativePhotonCollisionOptions options;
    options.seed = 9341;
    options.angular_law = NativePhotonCollisionAngularLaw::FixedFrameDirection;
    options.fixed_frame_direction = {0, 1, 0};
    options.max_events_per_photon = 1;
    options.max_optical_depth = 1.e6_rt;

    SetPhoton(photons, real_sentinel, int_sentinel,
        {0.45_rt, 0.3_rt, 0.3_rt}, {1, 0, 0});
    auto const identity = ReadOne(photons, real_sentinel, int_sentinel).idcpu;
    auto const depths = SelectModerateDepths(options, identity);
    auto const tau0 = depths[0];
    auto const tau1 = depths[1];
    auto const ratio = tau1 / tau0;
    constexpr amrex::Real total_length = 0.05_rt;
    auto const event_distance = total_length / (1 + 0.5_rt * ratio);
    auto const chi = tau0 / event_distance;
    amrex::MultiFab opacity(boxes, distribution, 1, 0);
    amrex::MultiFab initial_velocity(boxes, distribution, 3, 0);
    amrex::MultiFab work_velocity(boxes, distribution, 3, 0);
    opacity.setVal(chi);
    initial_velocity.setVal(0);
    work_velocity.setVal(0);
    work_velocity.setVal(0.125_rt * PhysConst::c, 1, 1);
    amrex::Real const distance_to_partition = event_after_partition ?
        0.5_rt * event_distance : event_distance + 0.5_rt * (total_length - event_distance);
    auto const initial_x = 0.5_rt - distance_to_partition;
    SetPhoton(photons, real_sentinel, int_sentinel,
              {initial_x, 0.3_rt, 0.3_rt}, {1, 0, 0});
    auto const initial = ReadOne(photons, real_sentinel, int_sentinel);
    auto const initial_cell = event_after_partition ? 4 : 3;

    auto const full = RunManual(photons, opacity, initial_velocity, work_velocity,
        geometry, options, {total_length}, real_sentinel, int_sentinel);
    auto const at_partition = RunManual(photons, opacity, initial_velocity,
        work_velocity, geometry, options,
        {distance_to_partition, total_length - distance_to_partition},
        real_sentinel, int_sentinel);
    auto const thirds = RunManual(photons, opacity, initial_velocity,
        work_velocity, geometry, options,
        {total_length / 3, total_length / 3, total_length / 3},
        real_sentinel, int_sentinel);
    CompareHistories(full, at_partition);
    CompareHistories(full, thirds);
    PrintHistory(event_after_partition ? "scatter_after" : "scatter_before", full);
    AMREX_ALWAYS_ASSERT(full.events == 1 && full.carry.completed_events == 1);
    AMREX_ALWAYS_ASSERT(full.carry.next_draw == 2);
    AMREX_ALWAYS_ASSERT(Close(full.carry.residual_optical_depth,
                              0.5_rt * tau1, tau1));
    CheckFourLedger(initial, full);
    AMREX_ALWAYS_ASSERT(std::abs(full.debit) >
        1.e-3_rt * std::abs(full.incoming[0]));
    auto const active_cell = amrex::IntVect(AMREX_D_DECL(initial_cell, 1, 1));
    AMREX_ALWAYS_ASSERT(std::abs(CellValue(*full.impulse_field,
        active_cell, 0)) > 0);
    AMREX_ALWAYS_ASSERT(std::abs(CellValue(*full.impulse_field,
        active_cell, 1)) > 0);
    auto const other_side = amrex::IntVect(AMREX_D_DECL(
        event_after_partition ? 3 : 4, 1, 1));
    AMREX_ALWAYS_ASSERT(CellValue(*full.impulse_field, other_side, 0) == 0);
    amrex::Print() << "3D recoil event "
                   << (event_after_partition ? "after" : "before")
                   << " box partition passed: Edebit=" << full.debit
                   << " J, event cell x=" << initial_cell << "\n";
}

void
CheckLaterSegmentRollback (WarpXParticleContainer& photons,
                           amrex::Geometry const& geometry,
                           int real_sentinel, int int_sentinel)
{
    auto const& boxes = photons.ParticleBoxArray(0);
    auto const& distribution = photons.ParticleDistributionMap(0);
    SetPhoton(photons, real_sentinel, int_sentinel,
        {0.45_rt, 0.3_rt, 0.3_rt}, {1, 0, 0});
    auto const initial = ReadOne(photons, real_sentinel, int_sentinel);
    NativePhotonCollisionOptions options;
    options.seed = 3279;
    options.angular_law = NativePhotonCollisionAngularLaw::FixedFrameDirection;
    options.fixed_frame_direction = {0, 1, 0};
    options.max_events_per_photon = 1;
    options.max_optical_depth = 1.e6_rt;
    auto const depths = SelectModerateDepths(options, initial.idcpu);
    auto const first_depth = depths[0];
    auto const second_depth = depths[1];
    constexpr amrex::Real segment_length = 0.05_rt;
    auto const ratio = second_depth / first_depth;
    auto const first_event_distance = segment_length *
        (1 - ratio / (2 * (1 + ratio)));
    auto const opacity_value = first_depth / first_event_distance;
    amrex::MultiFab opacity(boxes, distribution, 1, 0);
    amrex::MultiFab initial_velocity(boxes, distribution, 3, 0);
    amrex::MultiFab work_velocity(boxes, distribution, 3, 0);
    opacity.setVal(opacity_value);
    initial_velocity.setVal(0);
    work_velocity.setVal(0);
    work_velocity.setVal(0.125_rt * PhysConst::c, 1, 1);

    // Establish that an event is accepted in segment zero. The second keyed
    // event is strictly beyond its end, so the helper must reject later.
    PhotonPacketTransaction preview(photons);
    NativePhotonPathReplay first_segment;
    NativePhotonPathReport first_report;
    AMREX_ALWAYS_ASSERT(first_segment.Prepare(preview.Particles(), photons,
        opacity, initial_velocity, geometry, segment_length / PhysConst::c,
        options, first_report));
    amrex::MultiFab preview_impulse(boxes, distribution, 3, 0);
    amrex::MultiFab preview_debit(boxes, distribution, 1, 0);
    amrex::MultiFab preview_scale(boxes, distribution, 1, 0);
    AMREX_ALWAYS_ASSERT(first_segment.Replay(work_velocity,
        preview.Particles(), preview_impulse, preview_debit, preview_scale,
        first_report));
    AMREX_ALWAYS_ASSERT(first_report.collision_events == 1);

    LocalParticleRecords const live_before = LocalRecords(photons);
    PhotonPacketTransaction trial(photons);
    auto const private_before = LocalRecords(trial.Particles());
    auto const read_next_id = [] ()
    {
        auto const value = WarpXParticleContainer::ParticleType::NextID();
        WarpXParticleContainer::ParticleType::NextID(value);
        return value;
    };
    auto const next_id = read_next_id();
    amrex::MultiFab impulse(boxes, distribution, 3, 1);
    amrex::MultiFab debit(boxes, distribution, 1, 1);
    amrex::MultiFab scale(boxes, distribution, 1, 1);
    impulse.setVal(123);
    debit.setVal(123);
    scale.setVal(123);
    auto const impulse_before = FieldSnapshot(impulse);
    auto const debit_before = FieldSnapshot(debit);
    auto const scale_before = FieldSnapshot(scale);
    NativePhotonPathSubsegmentReplay replay;
    NativePhotonPathReport prepared;
    constexpr amrex::Real outer_length = 0.2_rt;
    AMREX_ALWAYS_ASSERT(replay.Prepare(photons, opacity, initial_velocity,
        geometry, outer_length / PhysConst::c, options, 8, prepared));
    NativePhotonPathSubsegmentReport report;
    AMREX_ALWAYS_ASSERT(!replay.Replay(work_velocity, trial.Particles(),
        impulse, debit, scale, report));
    AMREX_ALWAYS_ASSERT(!report.path.valid && report.photon_carries.empty());
    AMREX_ALWAYS_ASSERT(SameRecords(live_before, LocalRecords(photons)));
    AMREX_ALWAYS_ASSERT(SameRecords(private_before, LocalRecords(trial.Particles())));
    AMREX_ALWAYS_ASSERT(read_next_id() == next_id);
    AMREX_ALWAYS_ASSERT(SameBits(impulse_before, FieldSnapshot(impulse)));
    AMREX_ALWAYS_ASSERT(SameBits(debit_before, FieldSnapshot(debit)));
    AMREX_ALWAYS_ASSERT(SameBits(scale_before, FieldSnapshot(scale)));
    auto const restored = ReadOne(trial.Particles(), real_sentinel, int_sentinel);
    AMREX_ALWAYS_ASSERT(restored.idcpu == initial.idcpu);
    AMREX_ALWAYS_ASSERT(restored.real == initial.real);
    AMREX_ALWAYS_ASSERT(restored.int_sentinel == initial.int_sentinel);
    amrex::Print() << "3D later-segment event-cap rejection rolled back particles, "
                   << "counters, valid cells, and ghosts\n";
}
}

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& simulation = WarpX::GetInstance();
        auto& photons = simulation.GetPartContainer().
            GetParticleContainerFromName("photons");
        photons.AddRealComp("three_d_substep_real_sentinel", 1);
        photons.AddIntComp("three_d_substep_int_sentinel", 1);
        simulation.InitData();
        amrex::ParmParse pp_test("substep_test");
        std::string restart_from;
        std::string checkpoint_to;
        pp_test.query("restart_from", restart_from);
        pp_test.query("checkpoint_to", checkpoint_to);
        if (!restart_from.empty()) {
            photons.clearParticles();
            photons.Restart(restart_from, "photons");
            photons.Redistribute();
        }
        if (!checkpoint_to.empty()) { photons.Checkpoint(checkpoint_to, "photons"); }
        auto const real_sentinel = photons.GetRealCompIndex(
            "three_d_substep_real_sentinel");
        auto const int_sentinel = photons.GetIntCompIndex(
            "three_d_substep_int_sentinel");
        AMREX_ALWAYS_ASSERT(photons.TotalNumberOfParticles() == 1);
        auto const identity = ReadOne(photons, real_sentinel, int_sentinel).idcpu;
        auto const& geometry = simulation.Geom(0);
        auto const& boxes = photons.ParticleBoxArray(0);
        int const initial_box_count = boxes.size();
        auto const& distribution = photons.ParticleDistributionMap(0);
        amrex::MultiFab opacity(boxes, distribution, 1, 0);
        amrex::MultiFab velocity(boxes, distribution, 3, 0);
        opacity.setVal(0);
        velocity.setVal(0);
        SetPhoton(photons, real_sentinel, int_sentinel,
            {0.4375_rt, 0.375_rt, 0.375_rt}, {1, 0, 0});

        CheckAxisEdgeCornerAndTangent(photons, geometry,
            real_sentinel, int_sentinel);
        SetPhoton(photons, real_sentinel, int_sentinel,
        {0.4_rt, 0.237_rt, 0.321_rt},
            {2.0_rt/3, 1.0_rt/3, 2.0_rt/3});
        CheckTransparentCommonTime(photons, geometry,
            real_sentinel, int_sentinel);
        CheckPartitionSideScattering(photons, geometry,
            real_sentinel, int_sentinel, false);
        CheckPartitionSideScattering(photons, geometry,
            real_sentinel, int_sentinel, true);
        CheckLaterSegmentRollback(photons, geometry,
            real_sentinel, int_sentinel);
        amrex::Print() << "Cartesian 3D transactional packet substeps passed on "
                       << amrex::ParallelDescriptor::NProcs()
                       << " MPI rank(s); input boxes=" << initial_box_count
                       << "; post-restart boxes=" << boxes.size()
                       << "; photon idcpu=" << identity << "\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
    return 0;
}
