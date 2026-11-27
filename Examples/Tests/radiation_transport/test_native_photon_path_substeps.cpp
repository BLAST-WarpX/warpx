/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Radiation/NativePhotonPathReplay.H"
#include "Radiation/NativePhotonReplayInternals.H"
#include "Radiation/PhotonFourMomentum.H"
#include "Radiation/PhotonPacketTransaction.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

using namespace amrex::literals;

namespace
{
struct PhotonRecord
{
    std::vector<amrex::ParticleReal> real_data;
    std::vector<int> int_data;
    amrex::Real z = 0;
};

std::map<std::uint64_t, PhotonRecord>
PhotonRecords (WarpXParticleContainer::Base const& particles)
{
    std::map<std::uint64_t, PhotonRecord> records;
    for (auto const& [key, tile] : particles.GetParticles(0))
    {
        amrex::ignore_unused(key);
        long const count = tile.numParticles();
        if (count == 0) { continue; }
        auto const& soa = tile.GetStructOfArrays();
        std::vector<std::uint64_t> ids(static_cast<std::size_t>(count));
        std::vector<amrex::ParticleReal> z_values(
            static_cast<std::size_t>(count));
        std::vector<amrex::ParticleReal> real_values(
            static_cast<std::size_t>(count) *
            static_cast<std::size_t>(particles.NumRealComps()));
        std::vector<int> int_values(
            static_cast<std::size_t>(count) *
            static_cast<std::size_t>(particles.NumIntComps()));
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, soa.GetIdCPUData().begin(),
            soa.GetIdCPUData().begin() + count, ids.begin());
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, soa.GetRealData(PIdx::z).begin(),
            soa.GetRealData(PIdx::z).begin() + count, z_values.begin());
        for (int component = 0; component < particles.NumRealComps();
             ++component)
        {
            amrex::Gpu::copy(
                amrex::Gpu::deviceToHost,
                soa.GetRealData(component).begin(),
                soa.GetRealData(component).begin() + count,
                real_values.begin() +
                    static_cast<std::size_t>(component) *
                        static_cast<std::size_t>(count));
        }
        for (int component = 0; component < particles.NumIntComps();
             ++component)
        {
            amrex::Gpu::copy(
                amrex::Gpu::deviceToHost,
                soa.GetIntData(component).begin(),
                soa.GetIntData(component).begin() + count,
                int_values.begin() +
                    static_cast<std::size_t>(component) *
                        static_cast<std::size_t>(count));
        }
        for (long particle = 0; particle < count; ++particle)
        {
            PhotonRecord record;
            record.real_data.resize(
                static_cast<std::size_t>(particles.NumRealComps()));
            record.int_data.resize(
                static_cast<std::size_t>(particles.NumIntComps()));
            record.z = static_cast<amrex::Real>(
                z_values[static_cast<std::size_t>(particle)]);
            for (int component = 0; component < particles.NumRealComps();
                 ++component)
            {
                record.real_data[static_cast<std::size_t>(component)] =
                    real_values[
                        static_cast<std::size_t>(component) *
                            static_cast<std::size_t>(count) +
                        static_cast<std::size_t>(particle)];
            }
            for (int component = 0; component < particles.NumIntComps();
                 ++component)
            {
                record.int_data[static_cast<std::size_t>(component)] =
                    int_values[
                        static_cast<std::size_t>(component) *
                            static_cast<std::size_t>(count) +
                        static_cast<std::size_t>(particle)];
            }
            records.emplace(
                ids[static_cast<std::size_t>(particle)], std::move(record));
        }
    }
    return records;
}

PhotonRecord GlobalPhotonRecord (WarpXParticleContainer::Base const& photons,
                                std::uint64_t identity)
{
    auto const records = PhotonRecords(photons);
    PhotonRecord result;
    result.real_data.resize(photons.NumRealComps(), 0);
    result.int_data.resize(photons.NumIntComps(), 0);
    int found = 0;
    if (auto it = records.find(identity); it != records.end())
    {
        result = it->second;
        found = 1;
    }
    amrex::ParallelDescriptor::ReduceIntSum(found);
    AMREX_ALWAYS_ASSERT(found == 1);
    amrex::ParallelDescriptor::ReduceRealSum(result.real_data.data(), photons.NumRealComps());
    if (!result.int_data.empty())
    {
        amrex::ParallelDescriptor::ReduceIntSum(result.int_data.data(), photons.NumIntComps());
    }
    result.z = result.real_data[PIdx::z];
    return result;
}

amrex::Real
ExpectedOpticalDepth (
    warpx::radiation::NativePhotonCollisionOptions const& options,
    std::uint64_t identity,
    int event)
{
    return -std::log(warpx::radiation::photon_replay_detail::OpenUnitInterval(
        warpx::radiation::photon_replay_detail::PhotonDrawKey(
            options.seed, identity, options.interval, event, 0)));
}

void
CheckSinglePhotonLedgers (
    warpx::radiation::NativePhotonPathReport const& path,
    PhotonRecord const& record,
    warpx::radiation::PacketConversionFourMomentum const& expected_incoming)
{
    auto const outgoing =
        warpx::radiation::PacketConversionFourMomentumFromPhoton(
        record.real_data[PIdx::w], record.real_data[PIdx::ux],
        record.real_data[PIdx::uy], record.real_data[PIdx::uz]);
    AMREX_ALWAYS_ASSERT(outgoing.valid);
    auto const tolerance = [] (amrex::Real scale)
    {
        return 256 * std::numeric_limits<amrex::Real>::epsilon() *
            amrex::max(amrex::Real(1), std::abs(scale));
    };
    for (int component = 0; component < 4; ++component)
    {
        AMREX_ALWAYS_ASSERT(std::abs(
            path.incoming_inventory[component] -
            expected_incoming.value[component]) <=
            tolerance(expected_incoming.value[component]));
        AMREX_ALWAYS_ASSERT(std::abs(
            path.outgoing_inventory[component] -
            outgoing.value[component]) <= tolerance(outgoing.value[component]));
    }
    AMREX_ALWAYS_ASSERT(std::abs(
        path.photon_energy_debit -
        (path.incoming_inventory[0] - path.outgoing_inventory[0])) <=
        tolerance(path.incoming_inventory[0]));
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        AMREX_ALWAYS_ASSERT(std::abs(
            PhysConst::c * path.material_impulse[dimension] -
            (path.incoming_inventory[dimension + 1] -
             path.outgoing_inventory[dimension + 1])) <=
            tolerance(path.incoming_inventory[0]));
    }
}

void
FillRealComponents (
    WarpXParticleContainer::Base& particles,
    std::vector<std::vector<amrex::ParticleReal>> const& values)
{
    for (auto& [key, tile] : particles.GetParticles(0))
    {
        amrex::ignore_unused(key);
        long const count = tile.numParticles();
        if (count == 0) { continue; }
        auto& soa = tile.GetStructOfArrays();
        for (int component = 0; component < particles.NumRealComps();
             ++component)
        {
            amrex::Gpu::copy(
                amrex::Gpu::hostToDevice,
                values[static_cast<std::size_t>(component)].begin(),
                values[static_cast<std::size_t>(component)].end(),
                soa.GetRealData(component).begin());
        }
    }
}

struct PartitionResult
{
    PhotonRecord photon;
    warpx::radiation::NativePhotonPathCarry carry;
    warpx::radiation::NativePhotonPathReport path;
    std::vector<amrex::Long> events_after_segment;
    std::vector<amrex::Real> z_after_segment;
    std::vector<amrex::Real> energy_after_segment;
    std::vector<int> owner_after_segment;
    std::unique_ptr<amrex::MultiFab> impulse;
    std::unique_ptr<amrex::MultiFab> debit;
    std::unique_ptr<amrex::MultiFab> scale;
    int owner = -1;
};

amrex::Real
RoundoffBound (amrex::Real left, amrex::Real right, amrex::Real scale)
{
    // The 128-segment fixtures are transparent axial rays: each segment has
    // one c*dt product and one position add, plus at most 16 exact face snaps
    // and one periodic wrap (<300 rounded contributions to position). The
    // moving comparisons have at most four segments and two events; their
    // rate/work speeds are |beta|=1/8, so 1-beta.n >= 7/8 and the Lorentz
    // factors stay below 1.01. A 512-epsilon allowance covers those bounded
    // operation chains at their physical operand scale. Integer event/cell
    // topology is checked exactly, with no absolute fitted error floor.
    return 512 * std::numeric_limits<amrex::Real>::epsilon() *
        std::max({std::abs(left), std::abs(right), std::abs(scale)});
}

amrex::Real
CellFieldValue (amrex::MultiFab const& field, int cell_index, int component)
{
    amrex::IntVect const cell(AMREX_D_DECL(cell_index, 0, 0));
    amrex::Real value = 0;
    for (amrex::MFIter iterator(field); iterator.isValid(); ++iterator)
    {
        if (!iterator.validbox().contains(cell)) { continue; }
        auto const& fab = field[iterator];
        auto const box = iterator.fabbox();
        std::vector<amrex::Real> host(
            static_cast<std::size_t>(box.numPts()) * field.nComp());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, fab.dataPtr(),
            fab.dataPtr()+host.size(), host.begin());
        value = host[static_cast<std::size_t>(
            box.index(cell) + box.numPts()*component)];
    }
    amrex::ParallelDescriptor::ReduceRealSum(value);
    return value;
}
}

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        using namespace warpx::radiation;
        auto& simulation = WarpX::GetInstance();
        auto& particles = simulation.GetPartContainer();
        auto& photons_before_init =
            particles.GetParticleContainerFromName("photons");
        photons_before_init.AddRealComp("path_real_sentinel", 1);
        photons_before_init.AddIntComp("path_int_sentinel", 1);
        simulation.InitData();
        auto& photons = particles.GetParticleContainerFromName("photons");

        constexpr amrex::Real initial_z = 0.375_rt-0.0009765625_rt;
        std::uint64_t kept_id = 0;
        for (auto& [key, tile] : photons.GetParticles(0))
        {
            amrex::ignore_unused(key);
            long const count = tile.numParticles();
            if (count == 0) { continue; }
            auto& soa = tile.GetStructOfArrays();
            std::vector<std::uint64_t> ids(static_cast<std::size_t>(count));
            amrex::Gpu::copy(
                amrex::Gpu::deviceToHost, soa.GetIdCPUData().begin(),
                soa.GetIdCPUData().begin() + count, ids.begin());
            for (long particle = 0; particle < count; ++particle)
            {
                if (key.first == 0 && key.second == 0 && particle == 0)
                {
                    kept_id = ids[static_cast<std::size_t>(particle)];
                }
                else
                {
                    ids[static_cast<std::size_t>(particle)] =
                        amrex::ParticleIdCpus::Invalid;
                }
            }
            amrex::Gpu::copy(
                amrex::Gpu::hostToDevice, ids.begin(), ids.end(),
                soa.GetIdCPUData().begin());
        }
        photons.Redistribute();
#ifdef AMREX_USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &kept_id, 1, MPI_UINT64_T, MPI_MAX,
                      amrex::ParallelDescriptor::Communicator());
#endif
        AMREX_ALWAYS_ASSERT(photons.TotalNumberOfParticles() == 1);

        std::vector<std::vector<amrex::ParticleReal>> real_values(
            static_cast<std::size_t>(photons.NumRealComps()));
        for (auto& component : real_values) { component.resize(1, 0); }
        real_values[static_cast<std::size_t>(PIdx::z)][0] =
            static_cast<amrex::ParticleReal>(initial_z);
        real_values[static_cast<std::size_t>(PIdx::w)][0] = 1.0_prt;
        real_values[static_cast<std::size_t>(PIdx::ux)][0] = 0.0_prt;
        real_values[static_cast<std::size_t>(PIdx::uy)][0] = 0.0_prt;
        real_values[static_cast<std::size_t>(PIdx::uz)][0] =
            static_cast<amrex::ParticleReal>(PhysConst::c);
        int const sentinel_real =
            photons.GetRealCompIndex("path_real_sentinel");
        real_values[static_cast<std::size_t>(sentinel_real)][0] =
            19.5_prt;
        auto transverse_values = real_values;
        transverse_values[static_cast<std::size_t>(PIdx::ux)][0] =
            static_cast<amrex::ParticleReal>(PhysConst::c);
        transverse_values[static_cast<std::size_t>(PIdx::uz)][0] = 0.0_prt;
        auto reverse_values = real_values;
        reverse_values[static_cast<std::size_t>(PIdx::uz)][0] =
            static_cast<amrex::ParticleReal>(-PhysConst::c);
        FillRealComponents(photons, real_values);
        for (auto& [key, tile] : photons.GetParticles(0))
        {
            amrex::ignore_unused(key);
            if (tile.numParticles() == 0) { continue; }
            auto& soa = tile.GetStructOfArrays();
            std::vector<int> sentinel(1, 37);
            amrex::Gpu::copy(
                amrex::Gpu::hostToDevice, sentinel.begin(), sentinel.end(),
                soa.GetIntData(
                    photons.GetIntCompIndex("path_int_sentinel")).begin());
        }

        photons.Redistribute();
        auto const& box_array = photons.ParticleBoxArray(0);
        auto const& distribution_map = photons.ParticleDistributionMap(0);
        amrex::MultiFab transparent_opacity(
            box_array, distribution_map, 1, 0);
        amrex::MultiFab zero_velocity(box_array, distribution_map, 3, 0);
        transparent_opacity.setVal(0);
        zero_velocity.setVal(0);
        amrex::Real const dt = 0.00390625_rt/PhysConst::c;
        NativePhotonPathReport report;
        NativePhotonCollisionOptions options;
        options.seed = 91;
        options.interval = 13;
        options.max_events_per_photon = 64;
        options.max_optical_depth = 16;
        options.angular_law =
            NativePhotonCollisionAngularLaw::FixedFrameDirection;
        options.fixed_frame_direction = {0, 0, -1};

        {
            NativePhotonPathReplay transparent;
            bool const transparent_prepared = transparent.Prepare(
                photons, transparent_opacity, zero_velocity,
                simulation.Geom(0), dt, options, report);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                transparent_prepared, report.reason.c_str());
            PhotonPacketTransaction transaction(photons);
            amrex::MultiFab impulse(box_array, distribution_map, 3, 0);
            amrex::MultiFab debit(box_array, distribution_map, 1, 0);
            amrex::MultiFab scale(box_array, distribution_map, 1, 0);
            AMREX_ALWAYS_ASSERT(transparent.Replay(
                zero_velocity, transaction.Particles(), impulse, debit,
                scale, report));
            AMREX_ALWAYS_ASSERT(report.valid);
            AMREX_ALWAYS_ASSERT(report.collision_events == 0);
            AMREX_ALWAYS_ASSERT(report.face_crossings == 1);
            AMREX_ALWAYS_ASSERT(report.photon_energy_debit == 0);
            AMREX_ALWAYS_ASSERT(scale.norm0() == 0 && debit.norm0() == 0);
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                AMREX_ALWAYS_ASSERT(impulse.norm0(dimension) == 0);
            }
            auto const record = GlobalPhotonRecord(transaction.Particles(), kept_id);
            {
                AMREX_ALWAYS_ASSERT(std::abs(
                    record.z -
                    (initial_z + PhysConst::c * dt)) <=
                    8 * std::numeric_limits<amrex::Real>::epsilon() *
                        PhysConst::c * dt);
            }
        }

        amrex::MultiFab crossing_opacity(
            box_array, distribution_map, 1, 0);
        for (amrex::MFIter iterator(crossing_opacity); iterator.isValid();
             ++iterator)
        {
            auto chi = crossing_opacity.array(iterator);
            amrex::For(iterator.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::ignore_unused(j, k);
                chi(i, j, k) = i < 8 ? 0.0_rt : 2048.0_rt;
            });
        }
        {
            PhotonPacketTransaction transaction(photons);
            NativePhotonPathReplay crossing;
            AMREX_ALWAYS_ASSERT(crossing.Prepare(
                transaction.Particles(), photons, crossing_opacity,
                zero_velocity, simulation.Geom(0), dt, options, report));
            amrex::MultiFab impulse(box_array, distribution_map, 3, 0);
            amrex::MultiFab debit(box_array, distribution_map, 1, 0);
            amrex::MultiFab scale(box_array, distribution_map, 1, 0);
            amrex::Real const poison =
                std::numeric_limits<amrex::Real>::quiet_NaN();
            impulse.setVal(poison);
            debit.setVal(poison);
            scale.setVal(poison);
            auto const rejection_before =
                PhotonRecords(transaction.Particles());
            amrex::MultiFab fast_velocity(
                box_array, distribution_map, 3, 0);
            fast_velocity.setVal(PhysConst::c);
            AMREX_ALWAYS_ASSERT(!crossing.Replay(
                fast_velocity, transaction.Particles(), impulse, debit,
                scale, report));
            AMREX_ALWAYS_ASSERT(!report.valid);
            AMREX_ALWAYS_ASSERT(!impulse.is_finite());
            AMREX_ALWAYS_ASSERT(!debit.is_finite());
            AMREX_ALWAYS_ASSERT(!scale.is_finite());
            AMREX_ALWAYS_ASSERT(
                PhotonRecords(transaction.Particles()).size() ==
                rejection_before.size());

            AMREX_ALWAYS_ASSERT(crossing.Replay(
                zero_velocity, transaction.Particles(), impulse, debit,
                scale, report));
            AMREX_ALWAYS_ASSERT(report.valid);
            AMREX_ALWAYS_ASSERT(report.colliding_photons == 1);
            AMREX_ALWAYS_ASSERT(report.collision_events >= 1);
            AMREX_ALWAYS_ASSERT(report.face_crossings >= 1);
            AMREX_ALWAYS_ASSERT(impulse.norm0(2) > 0);
            AMREX_ALWAYS_ASSERT(scale.norm0() > 0);
            auto const record = GlobalPhotonRecord(transaction.Particles(), kept_id);
            {
                AMREX_ALWAYS_ASSERT(record.z < initial_z+PhysConst::c*dt);
                AMREX_ALWAYS_ASSERT(record.real_data[PIdx::uz] < 0);
                AMREX_ALWAYS_ASSERT(
                    record.real_data[
                        static_cast<std::size_t>(sentinel_real)] == 19.5_prt);
                AMREX_ALWAYS_ASSERT(
                    record.int_data[
                        static_cast<std::size_t>(
                            photons.GetIntCompIndex("path_int_sentinel"))] ==
                    37);
            }
        }

        auto const same_records = [](
            std::map<std::uint64_t, PhotonRecord> const& left,
            std::map<std::uint64_t, PhotonRecord> const& right)
        {
            if (left.size() != right.size()) { return false; }
            auto left_it = left.begin();
            auto right_it = right.begin();
            for (; left_it != left.end(); ++left_it, ++right_it)
            {
                if (left_it->first != right_it->first ||
                    left_it->second.z != right_it->second.z ||
                    left_it->second.real_data != right_it->second.real_data ||
                    left_it->second.int_data != right_it->second.int_data)
                {
                    return false;
                }
            }
            return true;
        };
        auto const run_substeps = [&](
            amrex::Real outer_dt, amrex::MultiFab const& opacity,
            NativePhotonCollisionOptions const& local_options,
            int subsegments, bool force_failure,
            NativePhotonPathSubsegmentReport& subreport,
            PhotonRecord& final_record, bool transverse = false)
        {
            PhotonPacketTransaction transaction(photons);
            auto const live_before = PhotonRecords(photons);
            if (transverse)
            {
                FillRealComponents(transaction.Particles(), transverse_values);
            }
            auto const candidate_before = PhotonRecords(transaction.Particles());
            NativePhotonPathSubsegmentReplay replay;
            NativePhotonPathReport prepare_report;
            AMREX_ALWAYS_ASSERT(replay.Prepare(
                photons, opacity, zero_velocity, simulation.Geom(0),
                outer_dt, local_options, subsegments, prepare_report));
            amrex::MultiFab impulse(box_array, distribution_map, 3, 0);
            amrex::MultiFab debit(box_array, distribution_map, 1, 0);
            amrex::MultiFab scale(box_array, distribution_map, 1, 0);
            amrex::Real const output_sentinel = force_failure ? 123.0_rt : 0.0_rt;
            impulse.setVal(output_sentinel);
            debit.setVal(output_sentinel);
            scale.setVal(output_sentinel);
            amrex::MultiFab invalid_velocity(
                box_array, distribution_map, 3, 0);
            invalid_velocity.setVal(PhysConst::c);
            bool const accepted = replay.Replay(
                force_failure ? invalid_velocity : zero_velocity,
                transaction.Particles(), impulse, debit, scale, subreport);
            if (force_failure)
            {
                AMREX_ALWAYS_ASSERT(!accepted);
                AMREX_ALWAYS_ASSERT(!subreport.path.valid);
                AMREX_ALWAYS_ASSERT(subreport.photon_carries.empty());
                amrex::Real const expected =
                    123.0_rt * static_cast<amrex::Real>(
                        simulation.Geom(0).Domain().numPts());
                AMREX_ALWAYS_ASSERT(impulse.sum(0) == expected);
                AMREX_ALWAYS_ASSERT(debit.sum(0) == expected);
                AMREX_ALWAYS_ASSERT(scale.sum(0) == expected);
                AMREX_ALWAYS_ASSERT(same_records(
                    PhotonRecords(transaction.Particles()), candidate_before));
                AMREX_ALWAYS_ASSERT(same_records(PhotonRecords(photons), live_before));
            }
            else
            {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    accepted, subreport.path.reason.c_str());
            }
            final_record = GlobalPhotonRecord(transaction.Particles(), kept_id);
            return accepted;
        };

        auto const dx = simulation.Geom(0).CellSize(0);
        // The primitive may legally replay a half-cell interval in one call.
        // Its private-snapshot overload lets independent partitions resume the
        // same optical-depth clock, with identical interval key and frozen
        // fields. The outer live species is never committed by these trials.
        auto const run_partition = [&] (
            amrex::Real outer_dt, std::vector<amrex::Real> const& fractions,
            amrex::MultiFab const& opacity, amrex::MultiFab const& rate_velocity,
            amrex::MultiFab const& work_velocity,
            NativePhotonCollisionOptions const& local_options,
            std::vector<NativePhotonPathCarry> const* start_carries = nullptr,
            bool reverse_start = false)
        {
            PhotonPacketTransaction trial(photons);
            if (reverse_start)
            {
                FillRealComponents(trial.Particles(), reverse_values);
            }
            PartitionResult result;
            result.impulse = std::make_unique<amrex::MultiFab>(
                box_array, distribution_map, 3, 0);
            result.debit = std::make_unique<amrex::MultiFab>(
                box_array, distribution_map, 1, 0);
            result.scale = std::make_unique<amrex::MultiFab>(
                box_array, distribution_map, 1, 0);
            result.impulse->setVal(0);
            result.debit->setVal(0);
            result.scale->setVal(0);
            std::vector<NativePhotonPathCarry> carries;
            for (std::size_t segment = 0; segment < fractions.size(); ++segment)
            {
                auto const fraction = fractions[segment];
                NativePhotonPathReplay replay;
                NativePhotonPathReport step;
                auto const step_dt = outer_dt * fraction;
                bool const prepared = segment == 0 && start_carries == nullptr ?
                    replay.Prepare(trial.Particles(), photons, opacity,
                        rate_velocity, simulation.Geom(0), step_dt,
                        local_options, step) :
                    replay.Prepare(trial.Particles(), photons, opacity,
                        rate_velocity, simulation.Geom(0), step_dt,
                        local_options, segment == 0 ? *start_carries : carries,
                        step);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(prepared, step.reason.c_str());
                amrex::MultiFab step_impulse(box_array, distribution_map, 3, 0);
                amrex::MultiFab step_debit(box_array, distribution_map, 1, 0);
                amrex::MultiFab step_scale(box_array, distribution_map, 1, 0);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(replay.Replay(
                    work_velocity, trial.Particles(), step_impulse,
                    step_debit, step_scale, step), step.reason.c_str());
                amrex::MultiFab::Add(*result.impulse, step_impulse, 0, 0, 3, 0);
                amrex::MultiFab::Add(*result.debit, step_debit, 0, 0, 1, 0);
                amrex::MultiFab::Add(*result.scale, step_scale, 0, 0, 1, 0);
                if (result.events_after_segment.empty())
                {
                    result.path.incoming_inventory = step.incoming_inventory;
                }
                result.path.outgoing_inventory = step.outgoing_inventory;
                result.path.collision_events += step.collision_events;
                result.path.face_crossings += step.face_crossings;
                result.path.photon_energy_debit += step.photon_energy_debit;
                for (int d = 0; d < 3; ++d)
                {
                    result.path.material_impulse[d] += step.material_impulse[d];
                }
                result.events_after_segment.push_back(result.path.collision_events);
                auto const segment_photon =
                    GlobalPhotonRecord(trial.Particles(), kept_id);
                result.z_after_segment.push_back(segment_photon.z);
                auto const four_momentum =
                    PacketConversionFourMomentumFromPhoton(
                        segment_photon.real_data[PIdx::w],
                        segment_photon.real_data[PIdx::ux],
                        segment_photon.real_data[PIdx::uy],
                        segment_photon.real_data[PIdx::uz]);
                AMREX_ALWAYS_ASSERT(four_momentum.valid);
                result.energy_after_segment.push_back(four_momentum.value[0]);
                int step_owner = PhotonRecords(trial.Particles()).count(kept_id) ?
                    amrex::ParallelDescriptor::MyProc() + 1 : 0;
                amrex::ParallelDescriptor::ReduceIntSum(step_owner);
                result.owner_after_segment.push_back(step_owner-1);
                carries = std::move(step.photon_carries);
            }
            result.photon = GlobalPhotonRecord(trial.Particles(), kept_id);
            int local_owner = PhotonRecords(trial.Particles()).count(kept_id) ?
                amrex::ParallelDescriptor::MyProc() + 1 : 0;
            amrex::ParallelDescriptor::ReduceIntSum(local_owner);
            result.owner = local_owner - 1;
            int found_carry = 0;
            for (auto const& carry : carries)
            {
                if (carry.idcpu == kept_id)
                {
                    result.carry = carry;
                    found_carry = 1;
                }
            }
            amrex::ParallelDescriptor::ReduceIntSum(found_carry);
            AMREX_ALWAYS_ASSERT(found_carry == 1);
            amrex::ParallelDescriptor::ReduceIntSum(result.carry.next_draw);
            amrex::ParallelDescriptor::ReduceIntSum(result.carry.completed_events);
            amrex::ParallelDescriptor::ReduceRealSum(result.carry.residual_optical_depth);
            result.carry.idcpu = kept_id;
            result.path.valid = true;
            return result;
        };

        auto const compare_partitions = [&] (
            PartitionResult const& left, PartitionResult const& right,
            char const* label)
        {
            AMREX_ALWAYS_ASSERT(left.owner == right.owner);
            AMREX_ALWAYS_ASSERT(left.path.collision_events ==
                                right.path.collision_events);
            AMREX_ALWAYS_ASSERT(left.path.face_crossings ==
                                right.path.face_crossings);
            AMREX_ALWAYS_ASSERT(left.carry.idcpu == right.carry.idcpu);
            AMREX_ALWAYS_ASSERT(left.carry.next_draw == right.carry.next_draw);
            AMREX_ALWAYS_ASSERT(left.carry.completed_events ==
                                right.carry.completed_events);
            char const* quantity = "position";
            auto const check = [&] (amrex::Real a, amrex::Real b,
                                   amrex::Real scale)
            {
                auto const error = std::abs(a-b);
                auto const bound = RoundoffBound(a, b, scale);
                if (error > bound)
                {
                    amrex::Print() << label << " " << quantity
                                   << " partition mismatch " << a
                                   << " versus " << b << ", error " << error
                                   << ", operand bound " << bound << "\n";
                }
                AMREX_ALWAYS_ASSERT(error <= bound);
            };
            check(left.photon.z, right.photon.z, dx);
            quantity = "real component";
            auto const momentum_scale = std::max({
                std::abs(static_cast<amrex::Real>(left.photon.real_data[PIdx::ux])),
                std::abs(static_cast<amrex::Real>(left.photon.real_data[PIdx::uy])),
                std::abs(static_cast<amrex::Real>(left.photon.real_data[PIdx::uz])),
                std::abs(static_cast<amrex::Real>(right.photon.real_data[PIdx::ux])),
                std::abs(static_cast<amrex::Real>(right.photon.real_data[PIdx::uy])),
                std::abs(static_cast<amrex::Real>(right.photon.real_data[PIdx::uz]))});
            for (int c = 0; c < photons.NumRealComps(); ++c)
            {
                if (c != PIdx::z && c != PIdx::ux &&
                    c != PIdx::uy && c != PIdx::uz)
                {
                    AMREX_ALWAYS_ASSERT(left.photon.real_data[c] ==
                                        right.photon.real_data[c]);
                    continue;
                }
                check(left.photon.real_data[c], right.photon.real_data[c],
                      c == PIdx::ux || c == PIdx::uy || c == PIdx::uz ?
                      momentum_scale : left.photon.real_data[c]);
            }
            AMREX_ALWAYS_ASSERT(left.photon.int_data == right.photon.int_data);
            quantity = "carry depth";
            check(left.carry.residual_optical_depth,
                  right.carry.residual_optical_depth,
                  left.carry.residual_optical_depth);
            for (int c = 0; c < 4; ++c)
            {
                quantity = "incoming inventory";
                check(left.path.incoming_inventory[c],
                      right.path.incoming_inventory[c],
                      left.path.incoming_inventory[0]);
                quantity = "outgoing inventory";
                check(left.path.outgoing_inventory[c],
                      right.path.outgoing_inventory[c],
                      left.path.incoming_inventory[0]);
            }
            quantity = "energy debit";
            check(left.path.photon_energy_debit,
                  right.path.photon_energy_debit,
                  left.path.incoming_inventory[0]);
            for (int c = 0; c < 3; ++c)
            {
                quantity = "material impulse total";
                check(PhysConst::c * left.path.material_impulse[c],
                      PhysConst::c * right.path.material_impulse[c],
                      left.path.incoming_inventory[0]);
            }
            for (int cell = simulation.Geom(0).Domain().smallEnd(0);
                 cell <= simulation.Geom(0).Domain().bigEnd(0); ++cell)
            {
                auto const local_scale = std::max({
                    std::abs(CellFieldValue(*left.scale, cell, 0)),
                    std::abs(CellFieldValue(*right.scale, cell, 0)),
                    std::abs(CellFieldValue(*left.debit, cell, 0)),
                    std::abs(CellFieldValue(*right.debit, cell, 0))});
                for (int c = 0; c < 3; ++c)
                {
                    quantity = "material impulse cell field";
                    check(PhysConst::c*CellFieldValue(*left.impulse, cell, c),
                          PhysConst::c*CellFieldValue(*right.impulse, cell, c),
                          local_scale);
                }
                quantity = "energy debit cell field";
                check(CellFieldValue(*left.debit, cell, 0),
                      CellFieldValue(*right.debit, cell, 0), local_scale);
            }
            // This is a conservative representation-error budget. A final
            // native rounding correction is added once per event-bearing
            // segment, so its magnitude can depend on the partition even
            // when the physical debit/impulse and carry agree.
            for (auto const* run : {&left, &right})
            {
                AMREX_ALWAYS_ASSERT(run->scale->is_finite());
                AMREX_ALWAYS_ASSERT(run->scale->sum(0) >=
                                    std::abs(run->debit->sum(0)));
            }
        };

        {
            auto moving = options;
            moving.moving_cross_sections = true;
            moving.fixed_frame_direction = {0, 0, -1};
            moving.max_optical_depth = 128;
            moving.max_events_per_photon = 128;
            bool found_event_key = false;
            for (std::uint64_t interval = 0; interval < 4096; ++interval)
            {
                moving.interval = interval;
                auto const first = ExpectedOpticalDepth(moving, kept_id, 0);
                auto const second = ExpectedOpticalDepth(moving, kept_id, 1);
                auto const third = ExpectedOpticalDepth(moving, kept_id, 2);
                if (second > 0.4_rt*first && second < 0.7_rt*first &&
                    third > 1.5_rt*first)
                {
                    found_event_key = true;
                    break;
                }
            }
            AMREX_ALWAYS_ASSERT(found_event_key);
            amrex::MultiFab rate(box_array, distribution_map, 3, 0);
            amrex::MultiFab work(box_array, distribution_map, 3, 0);
            rate.setVal(0);
            work.setVal(0);
            rate.setVal(-0.125_rt*PhysConst::c, 2, 1);
            work.setVal(0.125_rt*PhysConst::c, 2, 1);
            auto const tau = ExpectedOpticalDepth(moving, kept_id, 0);
            auto const beta = -0.125_rt;
            auto const rate_factor = (1-beta)/std::sqrt(1-beta*beta);
            amrex::MultiFab opacity(box_array, distribution_map, 1, 0);
            opacity.setVal(tau / (0.1875_rt * dx * rate_factor));
            auto const full = run_partition(dx/(2*PhysConst::c), {1},
                opacity, rate, work, moving);
            auto const halves = run_partition(dx/(2*PhysConst::c),
                {0.5_rt, 0.5_rt}, opacity, rate, work, moving);
            auto const quarters = run_partition(dx/(2*PhysConst::c),
                {0.25_rt, 0.25_rt, 0.25_rt, 0.25_rt},
                opacity, rate, work, moving);
            AMREX_ALWAYS_ASSERT(full.path.collision_events == 2);
            AMREX_ALWAYS_ASSERT(halves.events_after_segment[0] == 1);
            AMREX_ALWAYS_ASSERT(halves.events_after_segment[1] == 2);
            AMREX_ALWAYS_ASSERT(quarters.events_after_segment[0] == 0);
            AMREX_ALWAYS_ASSERT(quarters.events_after_segment[1] == 1);
            AMREX_ALWAYS_ASSERT(quarters.events_after_segment[3] == 2);
            AMREX_ALWAYS_ASSERT(full.path.photon_energy_debit != 0);
            amrex::Print() << std::setprecision(17)
                << "moving partition full/2: |dz|="
                << std::abs(full.photon.z-halves.photon.z)
                << " |duz|="
                << std::abs(full.photon.real_data[PIdx::uz]-
                            halves.photon.real_data[PIdx::uz])
                << " |ddepth|="
                << std::abs(full.carry.residual_optical_depth-
                            halves.carry.residual_optical_depth)
                << " |dEdebit|="
                << std::abs(full.path.photon_energy_debit-
                            halves.path.photon_energy_debit)
                << " events=" << full.path.collision_events
                << " owner=" << full.owner << "\n";
            for (std::size_t segment = 0;
                 segment < halves.z_after_segment.size(); ++segment)
            {
                amrex::Print() << "moving half segment " << segment
                    << " z=" << halves.z_after_segment[segment]
                    << " owner=" << halves.owner_after_segment[segment]
                    << " cumulative events="
                    << halves.events_after_segment[segment] << "\n";
            }
            for (int cell = 7; cell <= 8; ++cell)
            {
                for (int component = 0; component < 3; ++component)
                {
                    auto const a = CellFieldValue(*full.impulse, cell, component);
                    auto const b = CellFieldValue(*halves.impulse, cell, component);
                    amrex::Print() << "impulse cell " << cell
                        << " component " << component << ": " << a
                        << " versus " << b << "\n";
                }
                auto const a = CellFieldValue(*full.debit, cell, 0);
                auto const b = CellFieldValue(*halves.debit, cell, 0);
                amrex::Print() << "debit cell " << cell << ": " << a
                    << " versus " << b << "\n";
            }
            AMREX_ALWAYS_ASSERT(CellFieldValue(*full.debit, 7, 0) == 0);
            AMREX_ALWAYS_ASSERT(CellFieldValue(*full.debit, 8, 0) != 0);
            auto const scale_extra = halves.scale->sum(0)-full.scale->sum(0);
            auto const expected_extra = 2*halves.energy_after_segment[0];
            amrex::Print() << "scale correction extra=" << scale_extra
                << " expected 2E at cut=" << expected_extra << "\n";
            AMREX_ALWAYS_ASSERT(std::abs(scale_extra-expected_extra) <=
                RoundoffBound(scale_extra, expected_extra,
                              full.path.incoming_inventory[0]));
            compare_partitions(full, halves, "moving half-cell");
            compare_partitions(full, quarters, "moving quarter-cell");
        }
        {
            // Start from an explicit carried depth. The optical clock then
            // places one event immediately before, exactly at, or immediately
            // after the box face. The half partition also cuts at that face.
            auto tied = options;
            tied.fixed_frame_direction = {0, 0, -1};
            tied.max_optical_depth = 16;
            bool found_next_depth = false;
            for (std::uint64_t interval = 0; interval < 4096; ++interval)
            {
                tied.interval = interval;
                if (ExpectedOpticalDepth(tied, kept_id, 1) > 2.5_rt)
                {
                    found_next_depth = true;
                    break;
                }
            }
            AMREX_ALWAYS_ASSERT(found_next_depth);
            amrex::MultiFab opacity(box_array, distribution_map, 1, 0);
            amrex::MultiFab work(box_array, distribution_map, 3, 0);
            opacity.setVal(1024);
            work.setVal(0);
            work.setVal(0.125_rt*PhysConst::c, 2, 1);
            auto const face_distance = 0.375_rt-initial_z;
            AMREX_ALWAYS_ASSERT(face_distance == dx/64);
            for (auto const relative_depth : {0.75_rt, 1.0_rt, 1.25_rt})
            {
                std::vector<NativePhotonPathCarry> carried;
                if (PhotonRecords(photons).count(kept_id) != 0)
                {
                    carried.push_back(NativePhotonPathCarry{
                        kept_id, 1, 0, 1024*face_distance*relative_depth});
                }
                auto const full = run_partition(2*face_distance/PhysConst::c,
                    {1}, opacity, zero_velocity, work, tied, &carried);
                auto const split = run_partition(2*face_distance/PhysConst::c,
                    {0.5_rt, 0.5_rt}, opacity,
                    zero_velocity, work, tied, &carried);
                AMREX_ALWAYS_ASSERT(full.path.collision_events == 1);
                AMREX_ALWAYS_ASSERT(split.path.collision_events == 1);
                AMREX_ALWAYS_ASSERT(split.events_after_segment[0] ==
                                    (relative_depth > 1 ? 0 : 1));
                int const event_cell = relative_depth > 1 ? 8 : 7;
                AMREX_ALWAYS_ASSERT(std::abs(CellFieldValue(
                    *full.debit, event_cell, 0)) > 0);
                amrex::Print() << "face/census relative depth "
                    << relative_depth << ": full/split faces "
                    << full.path.face_crossings << "/"
                    << split.path.face_crossings << ", z "
                    << full.photon.z << "/" << split.photon.z
                    << ", owner " << full.owner << "/" << split.owner
                    << ", residual " << full.carry.residual_optical_depth
                    << "/" << split.carry.residual_optical_depth
                    << ", first-cut z " << split.z_after_segment[0]
                    << ", debit cell7 "
                    << CellFieldValue(*full.debit, 7, 0) << "/"
                    << CellFieldValue(*split.debit, 7, 0)
                    << ", cell8 " << CellFieldValue(*full.debit, 8, 0)
                    << "/" << CellFieldValue(*split.debit, 8, 0) << "\n";
                compare_partitions(full, split, "face/census tie");
            }
        }
        {
            // Two controlled events straddle the census cut AND the face.
            // The first is in cell 7; its oblique outgoing ray reaches cell 8
            // before the second. This checks event-cell chronology rather
            // than merely matching the total collision count.
            auto oblique = options;
            oblique.fixed_frame_direction = {0.6_rt, 0, 0.8_rt};
            oblique.max_optical_depth = 16;
            bool found_key = false;
            for (std::uint64_t interval = 0; interval < 4096; ++interval)
            {
                oblique.interval = interval;
                auto const second = ExpectedOpticalDepth(oblique, kept_id, 1);
                auto const third = ExpectedOpticalDepth(oblique, kept_id, 2);
                if (second > 0.5_rt && second < 2.0_rt && third > second)
                {
                    found_key = true;
                    break;
                }
            }
            AMREX_ALWAYS_ASSERT(found_key);
            auto const face_distance = 0.375_rt-initial_z;
            auto const second_depth = ExpectedOpticalDepth(oblique, kept_id, 1);
            auto const chi = second_depth/(1.25_rt*face_distance);
            amrex::MultiFab opacity(box_array, distribution_map, 1, 0);
            amrex::MultiFab work(box_array, distribution_map, 3, 0);
            opacity.setVal(chi);
            work.setVal(0);
            work.setVal(0.125_rt*PhysConst::c, 2, 1);
            std::vector<NativePhotonPathCarry> carried;
            if (PhotonRecords(photons).count(kept_id) != 0)
            {
                carried.push_back(NativePhotonPathCarry{
                    kept_id, 1, 0, chi*face_distance/4});
            }
            auto const full = run_partition(2*face_distance/PhysConst::c,
                {1}, opacity, zero_velocity, work, oblique, &carried);
            auto const split = run_partition(2*face_distance/PhysConst::c,
                {0.5_rt, 0.5_rt}, opacity,
                zero_velocity, work, oblique, &carried);
            AMREX_ALWAYS_ASSERT(full.path.collision_events == 2);
            AMREX_ALWAYS_ASSERT(split.events_after_segment[0] == 1);
            AMREX_ALWAYS_ASSERT(split.events_after_segment[1] == 2);
            AMREX_ALWAYS_ASSERT(split.owner_after_segment[0] ==
                                distribution_map[0]);
            AMREX_ALWAYS_ASSERT(split.owner_after_segment[1] ==
                                distribution_map[1]);
            AMREX_ALWAYS_ASSERT(CellFieldValue(*full.scale, 7, 0) > 0);
            AMREX_ALWAYS_ASSERT(CellFieldValue(*full.scale, 8, 0) > 0);
            AMREX_ALWAYS_ASSERT(CellFieldValue(*full.debit, 7, 0) != 0);
            compare_partitions(full, split, "ordered cross-face events");
        }
        {
            // The same outer interval has two different legal partitions.
            // A transparent path crosses the box/rank boundary and leaves the
            // first optical-depth draw untouched through every census cut.
            auto const two = run_partition(dx/PhysConst::c,
                {0.5_rt, 0.5_rt}, transparent_opacity,
                zero_velocity, zero_velocity, options);
            auto const three = run_partition(dx/PhysConst::c,
                {0.25_rt, 0.5_rt, 0.25_rt}, transparent_opacity,
                zero_velocity, zero_velocity, options);
            AMREX_ALWAYS_ASSERT(two.path.collision_events == 0);
            AMREX_ALWAYS_ASSERT(two.path.face_crossings == 1);
            AMREX_ALWAYS_ASSERT(two.owner_after_segment[0] ==
                                distribution_map[1]);
            AMREX_ALWAYS_ASSERT(three.owner_after_segment[0] ==
                                distribution_map[1]);
            compare_partitions(two, three, "transparent long interval");
            auto const expected_depth = ExpectedOpticalDepth(options, kept_id, 0);
            AMREX_ALWAYS_ASSERT(two.carry.next_draw == 1);
            AMREX_ALWAYS_ASSERT(two.carry.completed_events == 0);
            AMREX_ALWAYS_ASSERT(two.carry.residual_optical_depth == expected_depth);

            amrex::MultiFab thin(box_array, distribution_map, 1, 0);
            thin.setVal(expected_depth/(4*dx));
            auto const thin_two = run_partition(dx/PhysConst::c,
                {0.5_rt, 0.5_rt}, thin, zero_velocity, zero_velocity, options);
            auto const thin_three = run_partition(dx/PhysConst::c,
                {0.25_rt, 0.5_rt, 0.25_rt}, thin,
                zero_velocity, zero_velocity, options);
            AMREX_ALWAYS_ASSERT(thin_two.path.collision_events == 0);
            compare_partitions(thin_two, thin_three, "nonzero-opacity tail");
            AMREX_ALWAYS_ASSERT(std::abs(
                thin_two.carry.residual_optical_depth - 0.75_rt*expected_depth)
                <= RoundoffBound(thin_two.carry.residual_optical_depth,
                                 0.75_rt*expected_depth, expected_depth));

            auto varying = options;
            varying.moving_cross_sections = true;
            varying.max_optical_depth = 512;
            amrex::MultiFab rate(box_array, distribution_map, 3, 0);
            amrex::MultiFab work(box_array, distribution_map, 3, 0);
            rate.setVal(0);
            work.setVal(0);
            rate.setVal(-0.125_rt*PhysConst::c, 2, 1);
            work.setVal(0.125_rt*PhysConst::c, 2, 1);
            amrex::MultiFab varying_opacity(box_array, distribution_map, 1, 0);
            auto const beta = -0.125_rt;
            auto const rate_factor = (1-beta)/std::sqrt(1-beta*beta);
            auto const opacity_after_face = expected_depth/
                (0.125_rt*dx*rate_factor);
            for (amrex::MFIter iterator(varying_opacity); iterator.isValid();
                 ++iterator)
            {
                auto chi = varying_opacity.array(iterator);
                amrex::For(iterator.validbox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    amrex::ignore_unused(j, k);
                    chi(i, j, k) = i < 8 ? 0 : opacity_after_face;
                });
            }
            auto const varying_two = run_partition(dx/PhysConst::c,
                {0.5_rt, 0.5_rt}, varying_opacity, rate, work, varying);
            auto const varying_three = run_partition(dx/PhysConst::c,
                {0.25_rt, 0.5_rt, 0.25_rt}, varying_opacity,
                rate, work, varying);
            AMREX_ALWAYS_ASSERT(varying_two.path.collision_events >= 1);
            AMREX_ALWAYS_ASSERT(varying_two.path.photon_energy_debit != 0);
            compare_partitions(varying_two, varying_three,
                               "varying moving long interval");
        }
        {
            // A full periodic-domain traverse crosses both box/rank ownership
            // and the periodic seam. Dyadic fractions sum to one exactly.
            std::vector<amrex::Real> sixty_four(64, 1.0_rt/64);
            std::vector<amrex::Real> one_twenty_eight(128, 1.0_rt/128);
            auto const coarse = run_partition(1.0_rt/PhysConst::c,
                sixty_four, transparent_opacity,
                zero_velocity, zero_velocity, options);
            auto const fine = run_partition(1.0_rt/PhysConst::c,
                one_twenty_eight, transparent_opacity,
                zero_velocity, zero_velocity, options);
            AMREX_ALWAYS_ASSERT(coarse.path.face_crossings == 16);
            AMREX_ALWAYS_ASSERT(coarse.path.collision_events == 0);
            AMREX_ALWAYS_ASSERT(std::find(
                coarse.owner_after_segment.begin(),
                coarse.owner_after_segment.end(), distribution_map[1]) !=
                coarse.owner_after_segment.end());
            AMREX_ALWAYS_ASSERT(coarse.owner == distribution_map[0]);
            compare_partitions(coarse, fine, "periodic-domain crossing");
            AMREX_ALWAYS_ASSERT(std::abs(coarse.photon.z-initial_z) <=
                RoundoffBound(coarse.photon.z, initial_z, 1));
            auto const reverse_coarse = run_partition(1.0_rt/PhysConst::c,
                sixty_four, transparent_opacity,
                zero_velocity, zero_velocity, options, nullptr, true);
            auto const reverse_fine = run_partition(1.0_rt/PhysConst::c,
                one_twenty_eight, transparent_opacity,
                zero_velocity, zero_velocity, options, nullptr, true);
            AMREX_ALWAYS_ASSERT(reverse_coarse.path.face_crossings == 16);
            AMREX_ALWAYS_ASSERT(std::find(
                reverse_coarse.owner_after_segment.begin(),
                reverse_coarse.owner_after_segment.end(), distribution_map[1]) !=
                reverse_coarse.owner_after_segment.end());
            AMREX_ALWAYS_ASSERT(reverse_coarse.owner == distribution_map[0]);
            compare_partitions(reverse_coarse, reverse_fine,
                               "reverse periodic-domain crossing");
            AMREX_ALWAYS_ASSERT(std::abs(reverse_coarse.photon.z-initial_z) <=
                RoundoffBound(reverse_coarse.photon.z, initial_z, 1));
        }
        {
            NativePhotonPathSubsegmentReport subreport;
            PhotonRecord final_record;
            AMREX_ALWAYS_ASSERT(run_substeps(
                dx / PhysConst::c, transparent_opacity, options, 2, false,
                subreport, final_record));
            AMREX_ALWAYS_ASSERT(subreport.subsegments == 2);
            AMREX_ALWAYS_ASSERT(subreport.path.face_crossings == 1);
            AMREX_ALWAYS_ASSERT(subreport.path.collision_events == 0);
            AMREX_ALWAYS_ASSERT(subreport.path.photon_energy_debit == 0);
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                AMREX_ALWAYS_ASSERT(
                    subreport.path.material_impulse[dimension] == 0);
            }
            AMREX_ALWAYS_ASSERT(std::abs(
                final_record.z - (initial_z + dx)) <=
                16 * std::numeric_limits<amrex::Real>::epsilon() * dx);
            AMREX_ALWAYS_ASSERT(final_record.real_data[PIdx::w] == 1.0_prt);
            AMREX_ALWAYS_ASSERT(final_record.real_data[PIdx::ux] == 0.0_prt);
            AMREX_ALWAYS_ASSERT(final_record.real_data[PIdx::uy] == 0.0_prt);
            AMREX_ALWAYS_ASSERT(
                final_record.real_data[PIdx::uz] == PhysConst::c);
        }

        {
            amrex::MultiFab varying_opacity(
                box_array, distribution_map, 1, 0);
            for (amrex::MFIter iterator(varying_opacity); iterator.isValid();
                 ++iterator)
            {
                auto chi = varying_opacity.array(iterator);
                amrex::For(iterator.validbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::ignore_unused(j, k);
                    chi(i, j, k) = i < 8 ? 0.0_rt : 512.0_rt;
                });
            }
            NativePhotonCollisionOptions scattering = options;
            scattering.max_optical_depth = 1024;
            scattering.max_events_per_photon = 1024;
            NativePhotonPathSubsegmentReport subreport;
            PhotonRecord final_record;
            AMREX_ALWAYS_ASSERT(run_substeps(
                3 * dx / PhysConst::c, varying_opacity, scattering, 8, false,
                subreport, final_record));
            auto const& path = subreport.path;
            AMREX_ALWAYS_ASSERT(path.collision_events >= 1);
            AMREX_ALWAYS_ASSERT(path.colliding_photons == 1);
            amrex::Long local_scatter_carries =
                static_cast<amrex::Long>(subreport.photon_carries.size());
            amrex::ParallelDescriptor::ReduceLongSum(local_scatter_carries);
            AMREX_ALWAYS_ASSERT(local_scatter_carries == 1);
            if (!subreport.photon_carries.empty())
            {
                auto const& carry = subreport.photon_carries.front();
                AMREX_ALWAYS_ASSERT(carry.completed_events >= 1);
                AMREX_ALWAYS_ASSERT(carry.next_draw == carry.completed_events + 1);
                AMREX_ALWAYS_ASSERT(carry.residual_optical_depth > 0);
            }
            AMREX_ALWAYS_ASSERT(std::abs(
                path.incoming_inventory[0] - path.outgoing_inventory[0] -
                path.photon_energy_debit) <=
                1.e-11 * std::abs(path.incoming_inventory[0]));
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                AMREX_ALWAYS_ASSERT(std::abs(
                    path.incoming_inventory[dimension + 1] -
                    path.outgoing_inventory[dimension + 1] -
                    PhysConst::c * path.material_impulse[dimension]) <=
                    1.e-11 * std::abs(path.incoming_inventory[0]));
            }
            AMREX_ALWAYS_ASSERT(final_record.real_data[PIdx::w] == 1.0_prt);
        }

        {
            NativePhotonPathSubsegmentReport rejected;
            PhotonRecord final_record;
            auto const read_next_id = []()
            {
                auto const value =
                    WarpXParticleContainer::ParticleType::NextID();
                WarpXParticleContainer::ParticleType::NextID(value);
                return value;
            };
            auto const id_before = read_next_id();
            AMREX_ALWAYS_ASSERT(!run_substeps(
                dx / PhysConst::c, transparent_opacity, options, 2, true,
                rejected, final_record));
            AMREX_ALWAYS_ASSERT(read_next_id() == id_before);
        }

        {
            NativePhotonPathSubsegmentReport migrated;
            PhotonRecord final_record;
            AMREX_ALWAYS_ASSERT(run_substeps(
                3 * dx / PhysConst::c, transparent_opacity, options, 8, false,
                migrated, final_record));
            AMREX_ALWAYS_ASSERT(migrated.path.face_crossings >= 1);
            amrex::Long local_carries =
                static_cast<amrex::Long>(migrated.photon_carries.size());
            amrex::ParallelDescriptor::ReduceLongSum(local_carries);
            AMREX_ALWAYS_ASSERT(local_carries == 1);
            if (!migrated.photon_carries.empty())
            {
                auto const& carry = migrated.photon_carries.front();
                AMREX_ALWAYS_ASSERT(carry.completed_events == 0);
                AMREX_ALWAYS_ASSERT(carry.next_draw == 1);
                auto const expected_depth =
                    -std::log(warpx::radiation::photon_replay_detail::OpenUnitInterval(
                        warpx::radiation::photon_replay_detail::PhotonDrawKey(
                            options.seed, kept_id, options.interval, 0, 0)));
                AMREX_ALWAYS_ASSERT(std::abs(
                    carry.residual_optical_depth - expected_depth) <= 1.e-12);
            }
        }

        // A nonzero census tail must consume optical depth even without an
        // event. This is the regression missing from the original MPI probe,
        // which used zero opacity and could not detect a restarted clock.
        {
            auto const tau = ExpectedOpticalDepth(options, kept_id, 0);
            amrex::MultiFab thin(box_array, distribution_map, 1, 0);
            thin.setVal(tau / (4 * dx));
            PhotonPacketTransaction trial(photons);
            NativePhotonPathSubsegmentReplay replay;
            NativePhotonPathReport prepared;
            AMREX_ALWAYS_ASSERT(replay.Prepare(photons, thin, zero_velocity,
                simulation.Geom(0), dx / PhysConst::c, options, 2, prepared));
            // Prepared opacity is frozen, not borrowed mutable caller storage.
            thin.setVal(-1);
            amrex::MultiFab impulse(box_array, distribution_map, 3, 1);
            amrex::MultiFab debit(box_array, distribution_map, 1, 1);
            amrex::MultiFab scale(box_array, distribution_map, 1, 1);
            impulse.setVal(123); debit.setVal(123); scale.setVal(123);
            NativePhotonPathSubsegmentReport result;
            AMREX_ALWAYS_ASSERT(replay.Replay(zero_velocity, trial.Particles(),
                impulse, debit, scale, result));
            AMREX_ALWAYS_ASSERT(result.path.collision_events == 0);
            for (auto const& carry : result.photon_carries)
            {
                AMREX_ALWAYS_ASSERT(carry.next_draw == 1);
                AMREX_ALWAYS_ASSERT(std::abs(carry.residual_optical_depth - 0.75_rt*tau)
                    <= 128 * std::numeric_limits<amrex::Real>::epsilon() * tau);
            }
            CheckSinglePhotonLedgers(result.path,
                GlobalPhotonRecord(trial.Particles(), kept_id),
                PacketConversionFourMomentumFromPhoton(1, 0, 0, PhysConst::c));

            PhotonPacketTransaction alias_trial(photons);
            auto const before = PhotonRecords(alias_trial.Particles());
            debit.setVal(123);
            AMREX_ALWAYS_ASSERT(!replay.Replay(zero_velocity, alias_trial.Particles(),
                impulse, debit, debit, result));
            AMREX_ALWAYS_ASSERT(same_records(before, PhotonRecords(alias_trial.Particles())));
            debit.plus(-123, 0, 1, 1);
            AMREX_ALWAYS_ASSERT(debit.norm0(0, 1) == 0);
        }

        // Put the first two events on opposite sides of the half-interval
        // boundary. An event cap is for the OUTER interval, not each segment.
        {
            auto limited = options;
            limited.angular_law = NativePhotonCollisionAngularLaw::FixedFrameDirection;
            limited.fixed_frame_direction = {0, 0, 1};
            limited.max_events_per_photon = 1;
            auto const first = ExpectedOpticalDepth(limited, kept_id, 0);
            auto const second = first + ExpectedOpticalDepth(limited, kept_id, 1);
            auto const depth = (std::max(2*first, second) + 2*second) / 2;
            limited.max_optical_depth = 2*depth;
            amrex::MultiFab opacity(box_array, distribution_map, 1, 0);
            opacity.setVal(depth / dx);
            PhotonPacketTransaction trial(photons);
            auto const before = PhotonRecords(trial.Particles());
            auto const live_before = PhotonRecords(photons);
            auto const read_next_id = []()
            {
                auto const value = WarpXParticleContainer::ParticleType::NextID();
                WarpXParticleContainer::ParticleType::NextID(value);
                return value;
            };
            auto const id_before = read_next_id();
            {
                // Prove the first segment really accepts an event and moves
                // the photon to the second box before the outer replay fails.
                PhotonPacketTransaction preview(photons);
                NativePhotonPathReplay first_segment;
                NativePhotonPathReport first_report;
                AMREX_ALWAYS_ASSERT(first_segment.Prepare(
                    preview.Particles(), photons, opacity, zero_velocity,
                    simulation.Geom(0), dx/(2*PhysConst::c), limited,
                    first_report));
                amrex::MultiFab first_impulse(box_array, distribution_map, 3, 0);
                amrex::MultiFab first_debit(box_array, distribution_map, 1, 0);
                amrex::MultiFab first_scale(box_array, distribution_map, 1, 0);
                AMREX_ALWAYS_ASSERT(first_segment.Replay(zero_velocity,
                    preview.Particles(), first_impulse, first_debit,
                    first_scale, first_report));
                AMREX_ALWAYS_ASSERT(first_report.collision_events == 1);
                int owner = PhotonRecords(preview.Particles()).count(kept_id) ?
                    amrex::ParallelDescriptor::MyProc()+1 : 0;
                amrex::ParallelDescriptor::ReduceIntSum(owner);
                AMREX_ALWAYS_ASSERT(owner-1 == distribution_map[1]);
            }
            NativePhotonPathSubsegmentReplay replay;
            NativePhotonPathReport prepared;
            AMREX_ALWAYS_ASSERT(replay.Prepare(photons, opacity, zero_velocity,
                simulation.Geom(0), dx / PhysConst::c, limited, 2, prepared));
            amrex::MultiFab impulse(box_array, distribution_map, 3, 1);
            amrex::MultiFab debit(box_array, distribution_map, 1, 1);
            amrex::MultiFab scale(box_array, distribution_map, 1, 1);
            impulse.setVal(123); debit.setVal(123); scale.setVal(123);
            NativePhotonPathSubsegmentReport result;
            AMREX_ALWAYS_ASSERT(!replay.Replay(zero_velocity, trial.Particles(),
                impulse, debit, scale, result));
            AMREX_ALWAYS_ASSERT(!result.path.valid);
            AMREX_ALWAYS_ASSERT(result.photon_carries.empty());
            AMREX_ALWAYS_ASSERT(result.path.collision_events == 0);
            AMREX_ALWAYS_ASSERT(result.path.face_crossings == 0);
            AMREX_ALWAYS_ASSERT(result.path.colliding_photons == 0);
            AMREX_ALWAYS_ASSERT(same_records(before, PhotonRecords(trial.Particles())));
            AMREX_ALWAYS_ASSERT(same_records(live_before, PhotonRecords(photons)));
            AMREX_ALWAYS_ASSERT(read_next_id() == id_before);
            impulse.plus(-123, 0, 3, 1);
            debit.plus(-123, 0, 1, 1);
            scale.plus(-123, 0, 1, 1);
            for (int d = 0; d < 3; ++d) { AMREX_ALWAYS_ASSERT(impulse.norm0(d, 1) == 0); }
            AMREX_ALWAYS_ASSERT(debit.norm0(0, 1) == 0 && scale.norm0(0, 1) == 0);
        }

        {
            NativePhotonPathSubsegmentReplay replay;
            NativePhotonPathReport prepared;
            // Guard conversion to integer subsegment counts before overflow.
            AMREX_ALWAYS_ASSERT(!replay.Prepare(photons, transparent_opacity,
                zero_velocity, simulation.Geom(0), 1.e200_rt, options, 2, prepared));
            int const local_components = amrex::ParallelDescriptor::MyProc() == 0 ? 2 : 1;
            amrex::MultiFab invalid_opacity(box_array, distribution_map, local_components, 0);
            invalid_opacity.setVal(0);
            AMREX_ALWAYS_ASSERT(!replay.Prepare(photons, invalid_opacity, zero_velocity,
                simulation.Geom(0), dx / PhysConst::c, options, 2, prepared));
            if (amrex::ParallelDescriptor::NProcs() > 1)
            {
                auto const local_dt = amrex::ParallelDescriptor::MyProc() == 0 ?
                    std::numeric_limits<amrex::Real>::quiet_NaN() : dx / PhysConst::c;
                AMREX_ALWAYS_ASSERT(!replay.Prepare(photons, transparent_opacity,
                    zero_velocity, simulation.Geom(0), local_dt, options, 2, prepared));
                AMREX_ALWAYS_ASSERT(!replay.Prepare(photons, transparent_opacity,
                    zero_velocity, simulation.Geom(0), dx / PhysConst::c, options,
                    amrex::ParallelDescriptor::MyProc() == 0 ? 2 : 3, prepared));
            }
        }

        // Explicit resume must never infer a fresh draw from an absent or
        // malformed carry on the photon owner. Empty ranks participate in
        // all three collective rejections.
        {
            PhotonPacketTransaction trial(photons);
            NativePhotonPathReplay replay;
            NativePhotonPathReport prepared;
            std::vector<NativePhotonPathCarry> carries;
            AMREX_ALWAYS_ASSERT(!replay.Prepare(trial.Particles(), photons,
                transparent_opacity, zero_velocity, simulation.Geom(0),
                dx/(4*PhysConst::c), options, carries, prepared));

            auto huge_cap = options;
            huge_cap.max_events_per_photon = std::numeric_limits<int>::max();
            AMREX_ALWAYS_ASSERT(!replay.Prepare(trial.Particles(), photons,
                transparent_opacity, zero_velocity, simulation.Geom(0),
                dx/(4*PhysConst::c), huge_cap, prepared));

            if (PhotonRecords(trial.Particles()).count(kept_id) != 0)
            {
                carries.push_back(NativePhotonPathCarry{
                    kept_id, std::numeric_limits<int>::max(),
                    std::numeric_limits<int>::max(), 1});
            }
            AMREX_ALWAYS_ASSERT(!replay.Prepare(trial.Particles(), photons,
                transparent_opacity, zero_velocity, simulation.Geom(0),
                dx/(4*PhysConst::c), options, carries, prepared));
        }

        // Material-work retries change the staged secant frame while the
        // prepared opacity, interval key and initial photons stay frozen.
        // Replaying the first frame again must reproduce native outputs and
        // carry counters exactly, with no live particle publication.
        {
            amrex::MultiFab opacity(box_array, distribution_map, 1, 0);
            amrex::MultiFab forward_work(box_array, distribution_map, 3, 0);
            amrex::MultiFab reverse_work(box_array, distribution_map, 3, 0);
            opacity.setVal(8/dx);
            forward_work.setVal(0);
            reverse_work.setVal(0);
            forward_work.setVal(0.125_rt*PhysConst::c, 2, 1);
            reverse_work.setVal(-0.125_rt*PhysConst::c, 2, 1);
            auto retry_options = options;
            retry_options.fixed_frame_direction = {0, 0, -1};
            NativePhotonPathSubsegmentReplay replay;
            NativePhotonPathReport prepared;
            AMREX_ALWAYS_ASSERT(replay.Prepare(photons, opacity,
                zero_velocity, simulation.Geom(0), dx/PhysConst::c,
                retry_options, 4, prepared));
            PhotonPacketTransaction forward(photons);
            PhotonPacketTransaction reverse(photons);
            PhotonPacketTransaction repeated(photons);
            amrex::MultiFab impulse_a(box_array, distribution_map, 3, 0);
            amrex::MultiFab impulse_b(box_array, distribution_map, 3, 0);
            amrex::MultiFab impulse_c(box_array, distribution_map, 3, 0);
            amrex::MultiFab debit_a(box_array, distribution_map, 1, 0);
            amrex::MultiFab debit_b(box_array, distribution_map, 1, 0);
            amrex::MultiFab debit_c(box_array, distribution_map, 1, 0);
            amrex::MultiFab scale_a(box_array, distribution_map, 1, 0);
            amrex::MultiFab scale_b(box_array, distribution_map, 1, 0);
            amrex::MultiFab scale_c(box_array, distribution_map, 1, 0);
            NativePhotonPathSubsegmentReport report_a, report_b, report_c;
            auto const live_before = PhotonRecords(photons);
            AMREX_ALWAYS_ASSERT(replay.Replay(forward_work,
                forward.Particles(), impulse_a, debit_a, scale_a, report_a));
            AMREX_ALWAYS_ASSERT(replay.Replay(reverse_work,
                reverse.Particles(), impulse_b, debit_b, scale_b, report_b));
            AMREX_ALWAYS_ASSERT(replay.Replay(forward_work,
                repeated.Particles(), impulse_c, debit_c, scale_c, report_c));
            AMREX_ALWAYS_ASSERT(report_a.path.collision_events > 0);
            AMREX_ALWAYS_ASSERT(report_a.path.collision_events ==
                                report_c.path.collision_events);
            AMREX_ALWAYS_ASSERT(same_records(
                PhotonRecords(forward.Particles()),
                PhotonRecords(repeated.Particles())));
            AMREX_ALWAYS_ASSERT(same_records(live_before, PhotonRecords(photons)));
            AMREX_ALWAYS_ASSERT(GlobalPhotonRecord(
                forward.Particles(), kept_id).real_data[PIdx::uz] !=
                GlobalPhotonRecord(reverse.Particles(), kept_id).real_data[PIdx::uz]);
            for (int component = 0; component < 3; ++component)
            {
                amrex::MultiFab delta(box_array, distribution_map, 1, 0);
                amrex::MultiFab::Copy(delta, impulse_a, component, 0, 1, 0);
                amrex::MultiFab::Subtract(delta, impulse_c, component, 0, 1, 0);
                AMREX_ALWAYS_ASSERT(delta.norm0() == 0);
            }
            amrex::MultiFab::Subtract(debit_a, debit_c, 0, 0, 1, 0);
            amrex::MultiFab::Subtract(scale_a, scale_c, 0, 0, 1, 0);
            AMREX_ALWAYS_ASSERT(debit_a.norm0() == 0);
            AMREX_ALWAYS_ASSERT(scale_a.norm0() == 0);
            AMREX_ALWAYS_ASSERT(report_a.photon_carries.size() ==
                                report_c.photon_carries.size());
            for (std::size_t n = 0; n < report_a.photon_carries.size(); ++n)
            {
                auto const& a = report_a.photon_carries[n];
                auto const& c = report_c.photon_carries[n];
                AMREX_ALWAYS_ASSERT(a.idcpu == c.idcpu);
                AMREX_ALWAYS_ASSERT(a.next_draw == c.next_draw);
                AMREX_ALWAYS_ASSERT(a.completed_events == c.completed_events);
                AMREX_ALWAYS_ASSERT(a.residual_optical_depth ==
                                    c.residual_optical_depth);
            }
        }

        // Duplicate native identities are detected during carry migration,
        // after tentative transport. Neither particles nor output fields may
        // be published on this late rejection (including cross-rank motion).
        {
            PhotonPacketTransaction trial(photons);
            auto const original = GlobalPhotonRecord(photons, kept_id);
            for (auto& [key, tile] : trial.Particles().GetParticles(0))
            {
                amrex::ignore_unused(key);
                if (tile.numParticles() == 0) { continue; }
                tile.resize(2);
                auto& soa = tile.GetStructOfArrays();
                std::vector<std::uint64_t> ids(2, kept_id);
                amrex::Gpu::copy(amrex::Gpu::hostToDevice, ids.begin(), ids.end(),
                    soa.GetIdCPUData().begin());
                for (int c = 0; c < trial.Particles().NumRealComps(); ++c)
                {
                    std::vector<amrex::ParticleReal> values(2, original.real_data[c]);
                    amrex::Gpu::copy(amrex::Gpu::hostToDevice, values.begin(), values.end(),
                        soa.GetRealData(c).begin());
                }
                for (int c = 0; c < trial.Particles().NumIntComps(); ++c)
                {
                    std::vector<int> values(2, original.int_data[c]);
                    amrex::Gpu::copy(amrex::Gpu::hostToDevice, values.begin(), values.end(),
                        soa.GetIntData(c).begin());
                }
            }
            auto const before = PhotonRecords(trial.Particles());
            NativePhotonPathReplay replay;
            NativePhotonPathReport duplicate_report;
            AMREX_ALWAYS_ASSERT(replay.Prepare(trial.Particles(), photons,
                transparent_opacity, zero_velocity, simulation.Geom(0),
                dx / (4*PhysConst::c), options, duplicate_report));
            amrex::MultiFab impulse(box_array, distribution_map, 3, 1);
            amrex::MultiFab debit(box_array, distribution_map, 1, 1);
            amrex::MultiFab scale(box_array, distribution_map, 1, 1);
            impulse.setVal(123); debit.setVal(123); scale.setVal(123);
            AMREX_ALWAYS_ASSERT(!replay.Replay(zero_velocity, trial.Particles(),
                impulse, debit, scale, duplicate_report));
            AMREX_ALWAYS_ASSERT(same_records(before, PhotonRecords(trial.Particles())));
            AMREX_ALWAYS_ASSERT(trial.Particles().TotalNumberOfParticles() == 2);
            impulse.plus(-123, 0, 3, 1);
            debit.plus(-123, 0, 1, 1);
            scale.plus(-123, 0, 1, 1);
            for (int d = 0; d < 3; ++d) { AMREX_ALWAYS_ASSERT(impulse.norm0(d, 1) == 0); }
            AMREX_ALWAYS_ASSERT(debit.norm0(0, 1) == 0 && scale.norm0(0, 1) == 0);
        }

        if (amrex::ParallelDescriptor::IOProcessor())
        {
            amrex::Print() << "native photon path replay minimal gate passed\n";
        }
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
    return 0;
}
