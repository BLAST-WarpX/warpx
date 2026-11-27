/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Radiation/NativePhotonPathReplay.H"
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
#include <limits>
#include <map>
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

        if (amrex::ParallelDescriptor::IOProcessor())
        {
            amrex::Print() << "native photon path replay minimal gate passed\n";
        }
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
    return 0;
}
