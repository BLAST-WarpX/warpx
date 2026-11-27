/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/ParticleCreation/DefaultInitialization.H"
#include "Particles/PhotonParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Radiation/PhotonPacketTransaction.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

using namespace amrex::literals;

namespace
{
struct StateSnapshot
{
    std::pair<int, int> key{};
    long count = 0;
    std::vector<amrex::ParticleReal> real_data;
    std::vector<int> int_data;
    std::vector<std::uint64_t> id_data;
};

struct BirthInput
{
    long n = 0;
    amrex::Vector<amrex::ParticleReal> x;
    amrex::Vector<amrex::ParticleReal> y;
    amrex::Vector<amrex::ParticleReal> z;
    amrex::Vector<amrex::ParticleReal> ux;
    amrex::Vector<amrex::ParticleReal> uy;
    amrex::Vector<amrex::ParticleReal> uz;
    amrex::Vector<amrex::Vector<amrex::ParticleReal>> attributes_real;
    amrex::Vector<amrex::Vector<int>> attributes_int;
};

struct BirthRecord
{
    std::uint64_t idcpu = 0;
    amrex::ParticleReal weight = 0.0_prt;
    amrex::ParticleReal sentinel_real = 0.0_prt;
    amrex::ParticleReal uz = 0.0_prt;
    int sentinel_int = 0;
    int cpu = 0;
};

template <typename T>
bool
SameBits (std::vector<T> const& a, std::vector<T> const& b)
{
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

std::vector<StateSnapshot>
Snapshot (WarpXParticleContainer::Base const& container)
{
    std::vector<StateSnapshot> snapshots;
    auto const& tiles = container.GetParticles(0);
    snapshots.reserve(tiles.size());
    for (auto const& [key, tile] : tiles)
    {
        StateSnapshot snap;
        snap.key = key;
        snap.count = tile.numParticles();
        if (snap.count > 0)
        {
            auto const& soa = tile.GetStructOfArrays();
            snap.real_data.resize(static_cast<std::size_t>(snap.count) * tile.NumRealComps());
            snap.int_data.resize(static_cast<std::size_t>(snap.count) * tile.NumIntComps());
            snap.id_data.resize(static_cast<std::size_t>(snap.count));
            for (int comp = 0; comp < tile.NumRealComps(); ++comp)
            {
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetRealData(comp).begin(),
                                 soa.GetRealData(comp).begin() + snap.count,
                                 snap.real_data.data() +
                                     static_cast<std::size_t>(comp) * snap.count);
            }
            for (int comp = 0; comp < tile.NumIntComps(); ++comp)
            {
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetIntData(comp).begin(),
                                 soa.GetIntData(comp).begin() + snap.count,
                                 snap.int_data.data() +
                                     static_cast<std::size_t>(comp) * snap.count);
            }
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetIdCPUData().begin(),
                             soa.GetIdCPUData().begin() + snap.count, snap.id_data.begin());
        }
        snapshots.push_back(std::move(snap));
    }
    return snapshots;
}

bool
SameState (std::vector<StateSnapshot> const& a, std::vector<StateSnapshot> const& b)
{
    if (a.size() != b.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (a[i].key != b[i].key || a[i].count != b[i].count ||
            !SameBits(a[i].real_data, b[i].real_data) || !SameBits(a[i].int_data, b[i].int_data) ||
            !SameBits(a[i].id_data, b[i].id_data))
        {
            return false;
        }
    }
    return true;
}

template <typename Tile>
std::vector<int>
HostIntComponent (Tile const& tile, int component)
{
    if (component < 0)
    {
        return std::vector<int>(static_cast<std::size_t>(tile.numParticles()), 0);
    }
    auto const& data = tile.GetStructOfArrays().GetIntData(component);
    std::vector<int> host(static_cast<std::size_t>(tile.numParticles()));
    if (!host.empty())
    {
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, data.begin(), data.end(), host.begin());
    }
    return host;
}

template <typename Tile, typename VectorType>
std::vector<typename VectorType::value_type>
HostComponent (Tile const& tile, VectorType const& data)
{
    std::vector<typename VectorType::value_type> host(
        static_cast<std::size_t>(tile.numParticles()));
    if (!host.empty())
    {
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, data.begin(), data.end(), host.begin());
    }
    return host;
}

bool
HasBirthMarker (WarpXParticleContainer::Base const& particles)
{
    auto const names = particles.GetIntSoANames();
    return std::find(names.begin(), names.end(), "__photon_birth_provisional") != names.end();
}

int
LocalGrid (WarpXParticleContainer::Base const& particles, int rank)
{
    auto const& boxes = particles.ParticleBoxArray(0);
    auto const& distribution = particles.ParticleDistributionMap(0);
    for (int grid = 0; grid < boxes.size(); ++grid)
    {
        if (distribution[grid] == rank)
        {
            return grid;
        }
    }
    return -1;
}

amrex::Real
GridCellCenter (WarpXParticleContainer::Base const& particles, amrex::Geometry const& geometry,
                int grid, long cell = 0)
{
    auto const& boxes = particles.ParticleBoxArray(0);
    return geometry.ProbLo(0) +
           (boxes[grid].smallEnd(0) + 0.5_rt + static_cast<amrex::Real>(cell)) *
               geometry.CellSize(0);
}

BirthInput
MakeNoIntBirthInput (long n, amrex::Real z, amrex::ParticleReal weight,
                     amrex::ParticleReal sentinel_real)
{
    BirthInput input;
    input.n = n;
    input.x.resize(n, 0.25_rt);
    input.y.resize(n, 0.25_rt);
    input.z.resize(n, z);
    input.ux.resize(n, 0.0_rt);
    input.uy.resize(n, 0.0_rt);
    input.uz.resize(n, 701.0_rt);
    input.attributes_real.push_back(amrex::Vector<amrex::ParticleReal>(n, weight));
    input.attributes_real.push_back(amrex::Vector<amrex::ParticleReal>(n, sentinel_real));
    return input;
}

BirthInput
MakeBirthInput (long n, amrex::Real z, amrex::ParticleReal weight,
                amrex::ParticleReal sentinel_real, int sentinel_int)
{
    BirthInput input = MakeNoIntBirthInput(n, z, weight, sentinel_real);
    input.attributes_int.push_back(amrex::Vector<int>(n, sentinel_int));
    return input;
}

template <typename T>
std::vector<T>
BroadcastRecords (std::vector<T> const& local)
{
    std::vector<T> global;
    int const nprocs = amrex::ParallelDescriptor::NProcs();
    for (int rank = 0; rank < nprocs; ++rank)
    {
        int count = static_cast<int>(local.size());
        amrex::ParallelDescriptor::Bcast(&count, 1, rank);
        std::vector<unsigned char> bytes(static_cast<std::size_t>(count) * sizeof(T));
        if (rank == amrex::ParallelDescriptor::MyProc() && count != 0)
        {
            std::memcpy(bytes.data(), local.data(), bytes.size());
        }
        if (!bytes.empty())
        {
            amrex::ParallelDescriptor::Bcast(bytes.data(), bytes.size(), rank);
        }
        global.resize(global.size() + static_cast<std::size_t>(count));
        if (count != 0)
        {
            std::memcpy(global.data() + global.size() - static_cast<std::size_t>(count),
                        bytes.data(), bytes.size());
        }
    }
    return global;
}

std::vector<std::uint64_t>
AllIdCPUs (WarpXParticleContainer::Base const& particles)
{
    std::vector<std::uint64_t> ids;
    for (auto const& [key, tile] : particles.GetParticles(0))
    {
        auto const& idcpu = tile.GetStructOfArrays().GetIdCPUData();
        std::size_t const old_size = ids.size();
        ids.resize(old_size + static_cast<std::size_t>(tile.numParticles()));
        if (tile.numParticles() != 0)
        {
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, idcpu.begin(), idcpu.end(),
                             ids.begin() + old_size);
        }
        amrex::ignore_unused(key);
    }
    return ids;
}

std::vector<BirthRecord>
BirthRecords (WarpXParticleContainer::Base const& particles, int real_component, int int_component,
              amrex::ParticleReal minimum_weight = 0.0_prt)
{
    std::vector<BirthRecord> records;
    for (auto const& [key, tile] : particles.GetParticles(0))
    {
        auto const& soa = tile.GetStructOfArrays();
        auto const ids = HostComponent(tile, soa.GetIdCPUData());
        auto const weights = HostComponent(tile, soa.GetRealData(PIdx::w));
        auto const sentinels_real = HostComponent(tile, soa.GetRealData(real_component));
        auto const uz_values = HostComponent(tile, soa.GetRealData(PIdx::uz));
        auto const sentinels_int = HostIntComponent(tile, int_component);
        for (std::size_t i = 0; i < ids.size(); ++i)
        {
            if (weights[i] < minimum_weight)
            {
                continue;
            }
            if (std::abs(uz_values[i] - 701.0_prt) > 0.5_prt &&
                std::abs(uz_values[i] - 702.0_prt) > 0.5_prt)
            {
                continue;
            }
            BirthRecord record;
            record.idcpu = ids[i];
            record.weight = weights[i];
            record.sentinel_real = sentinels_real[i];
            record.uz = uz_values[i];
            record.sentinel_int = sentinels_int[i];
            std::uint64_t idcpu_value = record.idcpu;
            amrex::ParticleCPUWrapper const cpu = idcpu_value;
            record.cpu = static_cast<int>(cpu);
            records.push_back(record);
        }
        amrex::ignore_unused(key);
    }
    return records;
}

void
DeleteMarkedBirths (WarpXParticleContainer::Base& candidate, amrex::ParticleReal batch_uz,
                    bool delete_all_marked)
{
    auto const& geometry = WarpX::GetInstance().Geom(0);
    for (auto& [key, tile] : candidate.GetParticles(0))
    {
        if (tile.numParticles() == 0)
        {
            continue;
        }
        auto& soa = tile.GetStructOfArrays();
        int const marker_component = tile.NumIntComps() - 1;
        auto const marker = HostIntComponent(tile, marker_component);
        auto ids = HostComponent(tile, soa.GetIdCPUData());
        auto z_values = HostComponent(tile, soa.GetRealData(PIdx::z));
        auto const uz_values = HostComponent(tile, soa.GetRealData(PIdx::uz));
        bool changed = false;
        for (std::size_t i = 0; i < marker.size(); ++i)
        {
            if (marker[i] == 0 ||
                (!delete_all_marked && std::abs(uz_values[i] - batch_uz) > 0.5_prt))
            {
                continue;
            }
            ids[i] = amrex::ParticleIdCpus::Invalid;
            z_values[i] = geometry.ProbLo(0) - 100.0_rt * geometry.CellSize(0);
            changed = true;
        }
        if (changed)
        {
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, ids.begin(), ids.end(),
                             soa.GetIdCPUData().begin());
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, z_values.begin(), z_values.end(),
                             soa.GetRealData(PIdx::z).begin());
            amrex::Gpu::streamSynchronize();
        }
        amrex::ignore_unused(key);
    }
}

void
AssertUniqueIdCPUs (std::vector<std::uint64_t> const& ids)
{
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        for (std::size_t j = i + 1; j < ids.size(); ++j)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ids[i] != ids[j],
                                             "Native photon (id,cpu) identities are not unique.");
        }
    }
}

void
AssertNamedMetadata (WarpXParticleContainer& photons, int real_component, int int_component)
{
    for (auto& [key, tile] : photons.GetParticles(0))
    {
        auto& soa = tile.GetStructOfArrays();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            soa.GetRealNames() == photons.GetRealSoANames() &&
                soa.GetIntNames() == photons.GetIntSoANames(),
            "Committed photon tile metadata does not belong to the live schema.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            &soa.GetRealData("transaction_sentinel") == &soa.GetRealData(real_component) &&
                &soa.GetIntData("transaction_integer_sentinel") == &soa.GetIntData(int_component),
            "Named photon attributes changed after transaction destruction.");
        amrex::ignore_unused(key);
    }
}
} // namespace

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        using namespace warpx::radiation;
        using ParticleType = WarpXParticleContainer::ParticleType;
        auto& sim = WarpX::GetInstance();
        auto& photons_before_init = sim.GetPartContainer().GetParticleContainerFromName("photons");
        photons_before_init.AddRealComp("transaction_sentinel", 1);
        constexpr std::array<char const*, 5> reserved_defaults = {
            "radiation_resolved_draw_key_lo32", "radiation_resolved_draw_key_hi32",
            "radiation_diffusion_census_ux0", "radiation_diffusion_census_uy0",
            "radiation_diffusion_census_uz0"};
        for (auto const* name : reserved_defaults) {
            photons_before_init.AddRealComp(name, 1);
        }
        sim.InitData();

        auto& live_photons = sim.GetPartContainer().GetParticleContainerFromName("photons");
        int const myproc = amrex::ParallelDescriptor::MyProc();
        int const nprocs = amrex::ParallelDescriptor::NProcs();
        int const sentinel_real = live_photons.GetRealCompIndex("transaction_sentinel");
        int const local_grid = LocalGrid(live_photons, myproc);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(live_photons.TotalNumberOfParticles() >= 2,
                                         "Test requires at least two initialized native photons.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(live_photons.NumIntComps() == 0,
                                         "Photon fixture must start with zero runtime integers.");
        for (auto& [key, tile] : live_photons.GetParticles(0))
        {
            auto* real_values = tile.GetStructOfArrays().GetRealData(sentinel_real).data();
            int const tile_base = key.first;
            amrex::ParallelFor(tile.numParticles(), [=] AMREX_GPU_DEVICE(long p)
                               { real_values[p] = 13.25_prt + p + tile_base; });
        }
        amrex::Gpu::synchronize();

        // Bulk injection must zero the reserved attributes, independent of
        // allocator contents. Poison only a later range, then exercise the
        // shared CPU/GPU initializer and its external-component cutoff.
        int initialized_tiles = 0;
        for (auto& [key, tile] : live_photons.GetParticles(0))
        {
            long const count = tile.numParticles();
            if (count < 2) { continue; }
            ++initialized_tiles;
            auto& soa = tile.GetStructOfArrays();
            for (auto const* name : reserved_defaults)
            {
                int const component = live_photons.GetRealCompIndex(name);
                auto const initial = HostComponent(tile, soa.GetRealData(component));
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    std::all_of(initial.begin(), initial.end(),
                                [](amrex::ParticleReal value) { return value == 0.0_prt; }),
                    "Initial photon reserved draw/census attribute was not zeroed.");
                auto* values = soa.GetRealData(component).data();
                amrex::ParallelFor(count - 1, [=] AMREX_GPU_DEVICE(long p) {
                    if (p == 0) { values[0] = 613.0_prt; }
                    values[p + 1] = -321.0_prt;
                });
            }
            amrex::Gpu::synchronize();
            ParticleCreation::InitializeRadiationImpulseAttributes(
                tile, live_photons, 1, count);
            amrex::Gpu::synchronize();
            for (auto const* name : reserved_defaults)
            {
                auto const values = HostComponent(
                    tile, soa.GetRealData(live_photons.GetRealCompIndex(name)));
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    values.front() == 613.0_prt &&
                    std::all_of(values.begin() + 1, values.end(),
                                [](amrex::ParticleReal value) { return value == 0.0_prt; }),
                    "Reserved photon attribute changed a prior particle or missed the new range.");
            }
            int const low = live_photons.GetRealCompIndex(reserved_defaults[0]);
            int const high = live_photons.GetRealCompIndex(reserved_defaults[1]);
            auto* low_values = soa.GetRealData(low).data();
            amrex::ParallelFor(count - 1, [=] AMREX_GPU_DEVICE(long p) {
                low_values[p + 1] = 937.0_prt;
            });
            amrex::Gpu::synchronize();
            ParticleCreation::InitializeRadiationImpulseAttributes(
                tile, live_photons, 1, count, high);
            amrex::Gpu::synchronize();
            auto const skipped = HostComponent(tile, soa.GetRealData(low));
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                skipped.front() == 613.0_prt &&
                    std::all_of(skipped.begin() + 1, skipped.end(),
                                [](amrex::ParticleReal value) { return value == 937.0_prt; }),
                "Externally supplied photon attribute was overwritten.");
            ParticleCreation::InitializeRadiationImpulseAttributes(
                tile, live_photons, 0, count);
            amrex::Gpu::synchronize();
            auto const sentinel = HostComponent(tile, soa.GetRealData(sentinel_real));
            for (long p = 0; p < count; ++p) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    sentinel[static_cast<std::size_t>(p)] ==
                        13.25_prt + p + key.first,
                    "Reserved initialization changed a non-radiation attribute.");
            }
        }
        amrex::ParallelDescriptor::ReduceIntSum(initialized_tiles);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(initialized_tiles > 0,
                                         "Initializer probe had no two-particle tile.");

        // No-int birth: empty ranks participate without a foreign tile.
        std::vector<std::uint64_t> old_ids = BroadcastRecords(AllIdCPUs(live_photons));
        AssertUniqueIdCPUs(old_ids);
        long const no_int_local_births = myproc == 0 ? 1 : 0;
        amrex::Long const no_int_id_probe = ParticleType::NextID();
        int const no_int_staging_grid = LocalGrid(live_photons, 0);
        {
            PhotonPacketTransaction txn(live_photons);
            auto& candidate = txn.Particles();
            amrex::Real const staging_z =
                GridCellCenter(candidate, sim.Geom(0), no_int_staging_grid);
            BirthInput input =
                MakeNoIntBirthInput(no_int_local_births, staging_z, 101.25_prt, 45.5_prt);
            bool const added = txn.AddProvisionalBirths(
                no_int_staging_grid, input.n, input.x, input.y, input.z, input.ux, input.uy,
                input.uz, input.attributes_real, input.attributes_int, true);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(added, "No-int birth staging failed.");

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                candidate.NumIntComps() == 1 &&
                    candidate.GetIntSoANames().back() == "__photon_birth_provisional",
                "No-int birth candidate did not register exactly one private marker.");
            if (myproc != 0)
            {
                auto const foreign =
                    candidate.GetParticles(0).find(std::make_pair(no_int_staging_grid, 0));
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(foreign == candidate.GetParticles(0).end(),
                                                 "An empty rank created a foreign staging tile.");
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(txn.ValidateLive(), "No-int birth validation failed.");
            txn.Commit();
        }
        {
            auto records = BroadcastRecords(
                BirthRecords(live_photons, sentinel_real, live_photons.NumIntComps() - 1));
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(records.size() == 1,
                                             "No-int birth did not commit exactly one photon.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(records[0].weight == 101.25_prt &&
                                                 records[0].sentinel_real == 45.5_prt &&
                                                 records[0].uz == 701.0_prt,
                                             "No-int birth payload changed.");
            std::uint64_t id_value = records[0].idcpu;
            amrex::ParticleIDWrapper const id = id_value;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<amrex::Long>(id) != 1,
                                             "A provisional placeholder ID escaped Commit.");
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(live_photons.NumIntComps() == 0 &&
                                             !HasBirthMarker(live_photons),
                                         "No-int birth marker leaked into live schema.");
        for (auto& [key, tile] : live_photons.GetParticles(0))
        {
            auto& soa = tile.GetStructOfArrays();
            for (auto const* name : reserved_defaults) {
                auto const values = HostComponent(
                    tile, soa.GetRealData(live_photons.GetRealCompIndex(name)));
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    std::all_of(values.begin(), values.end(),
                                [](amrex::ParticleReal value) { return value == 0.0_prt; }),
                    "Provisional birth did not zero reserved photon attributes.");
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                soa.GetRealNames() == live_photons.GetRealSoANames() &&
                    &soa.GetRealData("transaction_sentinel") == &soa.GetRealData(sentinel_real),
                "No-int live metadata or named real access failed after destruction.");
            amrex::ignore_unused(key);
        }
        auto all_ids = BroadcastRecords(AllIdCPUs(live_photons));
        AssertUniqueIdCPUs(all_ids);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(all_ids.size() == old_ids.size() + 1U,
                                         "No-int birth did not add one native identity.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            ParticleType::NextID() == no_int_id_probe + 1 + no_int_local_births,
            "No-int birth consumed the wrong local number of native IDs.");

        // Repeated batches with all births deleted allocate no IDs.
        auto const before_all_deleted = Snapshot(live_photons);
        amrex::Long const deleted_id_probe = ParticleType::NextID();
        {
            PhotonPacketTransaction txn(live_photons);
            auto& candidate = txn.Particles();
            int const staging_grid = local_grid >= 0 ? local_grid : 0;
            amrex::Real const local_z =
                GridCellCenter(candidate, sim.Geom(0), staging_grid >= 0 ? staging_grid : 0);
            long const local_count = local_grid >= 0 ? 1 : 0;
            BirthInput first = MakeNoIntBirthInput(local_count, local_z, 201.0_prt, 31.0_prt);
            BirthInput second = MakeNoIntBirthInput(local_count, local_z, 202.0_prt, 32.0_prt);
            second.uz.assign(local_count, 502.0_prt);
            bool const added_first = txn.AddProvisionalBirths(
                staging_grid, first.n, first.x, first.y, first.z, first.ux, first.uy, first.uz,
                first.attributes_real, first.attributes_int, true);
            candidate.Redistribute();
            int const later_grid = LocalGrid(candidate, myproc);
            amrex::Real const later_z =
                GridCellCenter(candidate, sim.Geom(0), later_grid >= 0 ? later_grid : 0);
            second.z.assign(local_count, later_z);
            bool const added_second = txn.AddProvisionalBirths(
                later_grid >= 0 ? later_grid : 0, second.n, second.x, second.y, second.z, second.ux,
                second.uy, second.uz, second.attributes_real, second.attributes_int, true);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(added_first && added_second,
                                             "Repeated no-int birth staging failed.");
            DeleteMarkedBirths(candidate, 0.0_prt, true);
            candidate.Redistribute();
            for (auto& [key, tile] : candidate.GetParticles(0))
            {
                auto const marker = HostIntComponent(tile, tile.NumIntComps() - 1);
                for (int value : marker)
                {
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(value == 0,
                                                     "Not all marked births were deleted.");
                }
                amrex::ignore_unused(key);
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(txn.ValidateLive(), "All-deleted validation failed.");
            txn.Commit();
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(SameState(Snapshot(live_photons), before_all_deleted),
                                         "All-deleted birth transaction changed survivors.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ParticleType::NextID() == deleted_id_probe + 1,
                                         "Deleted provisional births consumed native IDs.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(live_photons.NumIntComps() == 0 &&
                                             !HasBirthMarker(live_photons),
                                         "All-deleted birth marker leaked into live schema.");

        // Add the named integer gate after proving the zero-int path.
        live_photons.AddIntComp("transaction_integer_sentinel", 1);
        int sentinel_int = live_photons.NumIntComps() - 1;
        live_photons.AddRealComp("radiation_impulse_default_witness", 1);
        int const default_real = live_photons.NumRealComps() - 1;
        live_photons.AddIntComp("ionizationLevel", 1);
        int const default_int = live_photons.NumIntComps() - 1;
        int const native_initial_level = live_photons.getIonizationInitialLevel();
        for (auto& [key, tile] : live_photons.GetParticles(0))
        {
            live_photons.DefineAndReturnParticleTile(0, key.first, key.second);
        }
        for (auto& [key, tile] : live_photons.GetParticles(0))
        {
            auto* int_values = tile.GetStructOfArrays().GetIntData(sentinel_int).data();
            auto* real_defaults = tile.GetStructOfArrays().GetRealData(default_real).data();
            auto* int_defaults = tile.GetStructOfArrays().GetIntData(default_int).data();
            int const tile_base = key.first;
            amrex::ParallelFor(tile.numParticles(),
                               [=] AMREX_GPU_DEVICE(long p)
                               {
                                   int_values[p] = 17 + static_cast<int>(p) + tile_base;
                                   real_defaults[p] = 0.0_prt;
                                   int_defaults[p] = native_initial_level;
                               });
        }
        amrex::Gpu::synchronize();

        // One-rank bad length/stale live reject before marker mutation.
        {
            PhotonPacketTransaction txn(live_photons);
            auto const candidate_before = Snapshot(txn.Particles());
            auto const live_before = Snapshot(live_photons);
            long const invalid_n = myproc == 0 ? 1 : 0;
            BirthInput invalid = MakeBirthInput(invalid_n, 0.25_rt, 1.0_prt, 0.0_prt, 0);
            if (myproc == 0)
            {
                invalid.z.clear();
            }
            int const grid = local_grid >= 0 ? local_grid : 0;
            bool const added = txn.AddProvisionalBirths(
                grid, invalid.n, invalid.x, invalid.y, invalid.z, invalid.ux, invalid.uy,
                invalid.uz, invalid.attributes_real, invalid.attributes_int, true);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!added,
                                             "One-rank invalid input must reject collectively.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(SameState(Snapshot(txn.Particles()), candidate_before),
                                             "Rejected input changed the candidate.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(SameState(Snapshot(live_photons), live_before),
                                             "Rejected input changed live photons.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!HasBirthMarker(txn.Particles()),
                                             "Rejected input registered the birth marker.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!txn.ValidateLive(),
                                             "A failed staging transaction must not validate.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!txn.ValidateSnapshotFor(live_photons),
                "Failed private staging must reject snapshot preflight.");
        }
        {
            PhotonPacketTransaction txn(live_photons);
            auto const candidate_before = Snapshot(txn.Particles());
            amrex::ParticleReal original = 0.0_prt;
            bool mutated = false;
            if (myproc == 0)
            {
                for (auto& [key, tile] : live_photons.GetParticles(0))
                {
                    if (tile.numParticles() != 0)
                    {
                        auto* values = tile.GetStructOfArrays().GetRealData(sentinel_real).data();
                        amrex::Gpu::copy(amrex::Gpu::deviceToHost, values, values + 1, &original);
                        amrex::ParticleReal const changed = original + 42.0_prt;
                        amrex::Gpu::copy(amrex::Gpu::hostToDevice, &changed, &changed + 1, values);
                        mutated = true;
                        break;
                    }
                    amrex::ignore_unused(key);
                }
            }
            BirthInput empty = MakeBirthInput(0, 0.0_rt, 0.0_prt, 0.0_prt, 0);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!txn.ValidateSnapshotFor(live_photons),
                "One-rank stale live state must reject snapshot preflight collectively.");
            bool const added = txn.AddProvisionalBirths(
                local_grid >= 0 ? local_grid : 0, 0, empty.x, empty.y, empty.z, empty.ux, empty.uy,
                empty.uz, empty.attributes_real, empty.attributes_int, true);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!added, "Stale live input must reject collectively.");
            if (mutated)
            {
                for (auto& [key, tile] : live_photons.GetParticles(0))
                {
                    if (tile.numParticles() != 0)
                    {
                        auto* values = tile.GetStructOfArrays().GetRealData(sentinel_real).data();
                        amrex::Gpu::copy(amrex::Gpu::hostToDevice, &original, &original + 1,
                                         values);
                        break;
                    }
                    amrex::ignore_unused(key);
                }
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(SameState(Snapshot(txn.Particles()), candidate_before),
                                             "Stage-live rejection changed the candidate.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!txn.ValidateLive(),
                                             "A stale-live staging transaction must not validate.");
        }

        // Root no-birth gates and post-destruction metadata checks.
        auto const live_snap_rejection = Snapshot(live_photons);
        amrex::Long const rejection_id_probe = ParticleType::NextID();
        {
            PhotonPacketTransaction txn(live_photons);
            auto& candidate = txn.Particles();
            for (auto& [key, tile] : candidate.GetParticles(0))
            {
                if (tile.numParticles() == 0)
                {
                    continue;
                }
                auto& soa = tile.GetStructOfArrays();
                for (int comp = 0; comp < tile.NumRealComps(); ++comp)
                {
                    auto* values = soa.GetRealData(comp).data();
                    amrex::ParallelFor(tile.numParticles(),
                                       [=] AMREX_GPU_DEVICE(long i) { values[i] += 1.0_prt; });
                }
                for (int comp = 0; comp < tile.NumIntComps(); ++comp)
                {
                    auto* values = soa.GetIntData(comp).data();
                    amrex::ParallelFor(tile.numParticles(),
                                       [=] AMREX_GPU_DEVICE(long i) { values[i] += 1; });
                }
                auto* ids = soa.GetIdCPUData().data();
                amrex::ParallelFor(tile.numParticles(),
                                   [=] AMREX_GPU_DEVICE(long i) { ids[i] ^= 0x55555555ULL; });
                amrex::ignore_unused(key);
            }
            amrex::Gpu::synchronize();
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(SameState(Snapshot(live_photons), live_snap_rejection),
                                         "Uncommitted no-birth mutation changed live photons.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ParticleType::NextID() == rejection_id_probe + 1,
                                         "Uncommitted no-birth mutation consumed IDs.");

        long const particles_before_no_birth = live_photons.TotalNumberOfParticles();
        auto const geom = sim.Geom(0);
        uint64_t moved_survivor_id = 0;
        int const target_rank = nprocs > 1 ? 1 : 0;
        {
            PhotonPacketTransaction txn(live_photons);
            auto& candidate = txn.Particles();
            auto const& boxes = candidate.ParticleBoxArray(0);
            auto const& distribution = candidate.ParticleDistributionMap(0);
            int target_box = 0;
            for (int grid = 0; grid < boxes.size(); ++grid)
            {
                if (distribution[grid] == target_rank)
                {
                    target_box = grid;
                    break;
                }
            }
            amrex::Real const target_x =
                geom.ProbLo(0) + (boxes[target_box].smallEnd(0) + 0.5_rt) * geom.CellSize(0);
            if (myproc == 0)
            {
                for (auto& [key, tile] : candidate.GetParticles(0))
                {
                    if (tile.numParticles() < 2)
                    {
                        continue;
                    }
                    auto& soa = tile.GetStructOfArrays();
                    auto* ids = soa.GetIdCPUData().data();
                    auto* positions = soa.GetRealData(PIdx::z).data();
                    amrex::Gpu::copy(amrex::Gpu::deviceToHost, ids, ids + 1, &moved_survivor_id);
                    if (nprocs > 1)
                    {
                        amrex::Gpu::copy(amrex::Gpu::hostToDevice, &target_x, &target_x + 1,
                                         positions);
                    }
                    std::uint64_t const invalid_id = amrex::ParticleIdCpus::Invalid;
                    amrex::Real const outside = geom.ProbLo(0) - 100.0_rt * geom.CellSize(0);
                    amrex::Gpu::copy(amrex::Gpu::hostToDevice, &invalid_id, &invalid_id + 1,
                                     ids + 1);
                    amrex::Gpu::copy(amrex::Gpu::hostToDevice, &outside, &outside + 1,
                                     positions + 1);
                    break;
                }
            }
            amrex::ParallelDescriptor::Bcast(&moved_survivor_id, 1, 0);
            candidate.Redistribute();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(txn.ValidateLive(),
                                             "Redistributed no-birth candidate validation failed.");
            auto const accepted = Snapshot(candidate);
            txn.Commit();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(SameState(Snapshot(live_photons), accepted),
                                             "No-birth Commit changed the accepted candidate.");
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(live_photons.TotalNumberOfParticles() ==
                                             particles_before_no_birth - 1,
                                         "No-birth Commit did not delete the invalid survivor.");
        if (nprocs > 1)
        {
            long found = 0;
            auto const ids = AllIdCPUs(live_photons);
            if (myproc == target_rank)
            {
                found = std::find(ids.begin(), ids.end(), moved_survivor_id) != ids.end() ? 1 : 0;
            }
            amrex::ParallelDescriptor::ReduceLongMax(found);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(found == 1,
                                             "No-birth survivor did not migrate to its target.");
        }
        AssertNamedMetadata(live_photons, sentinel_real, sentinel_int);

        auto const stale_snapshot = Snapshot(live_photons);
        {
            PhotonPacketTransaction txn(live_photons);
            amrex::ParticleReal original = 0.0_prt;
            bool mutated = false;
            if (myproc == 0)
            {
                for (auto& [key, tile] : live_photons.GetParticles(0))
                {
                    if (tile.numParticles() != 0)
                    {
                        auto* values = tile.GetStructOfArrays().GetRealData(0).data();
                        amrex::Gpu::copy(amrex::Gpu::deviceToHost, values, values + 1, &original);
                        amrex::ParticleReal const changed = original + 42.0_prt;
                        amrex::Gpu::copy(amrex::Gpu::hostToDevice, &changed, &changed + 1, values);
                        mutated = true;
                        break;
                    }
                    amrex::ignore_unused(key);
                }
            }
            bool const valid = txn.ValidateLive();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!valid, "Stale live state was accepted.");
            if (mutated)
            {
                for (auto& [key, tile] : live_photons.GetParticles(0))
                {
                    if (tile.numParticles() != 0)
                    {
                        auto* values = tile.GetStructOfArrays().GetRealData(0).data();
                        amrex::Gpu::copy(amrex::Gpu::hostToDevice, &original, &original + 1,
                                         values);
                        break;
                    }
                    amrex::ignore_unused(key);
                }
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(SameState(Snapshot(live_photons), stale_snapshot),
                                             "Stale-live validation changed non-injected state.");
        }

        // Repeated batches, migration/deletion, exact local ID use.
        old_ids = BroadcastRecords(AllIdCPUs(live_photons));
        AssertUniqueIdCPUs(old_ids);
        amrex::Long const repeated_id_probe = ParticleType::NextID();
        long repeated_births_before = live_photons.TotalNumberOfParticles();
        {
            PhotonPacketTransaction txn(live_photons);
            auto& candidate = txn.Particles();
            int const first_grid = local_grid >= 0 ? local_grid : 0;
            amrex::Real const first_z = GridCellCenter(candidate, sim.Geom(0), first_grid);
            long const first_count = local_grid >= 0 ? 1 : 0;
            BirthInput first = MakeBirthInput(first_count, first_z, 301.0_prt, 41.0_prt, 87);
            first.uz.assign(first_count, 501.0_prt);
            bool const added_first = txn.AddProvisionalBirths(
                first_grid, first.n, first.x, first.y, first.z, first.ux, first.uy, first.uz,
                first.attributes_real, first.attributes_int, true);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(added_first, "Defaulted birth staging failed.");
            long defaulted_count = 0;
            for (auto const& snapshot : Snapshot(candidate))
            {
                for (long p = 0; p < snapshot.count; ++p)
                {
                    if (snapshot.int_data[(candidate.NumIntComps() - 1) * snapshot.count + p] == 0)
                    {
                        continue;
                    }
                    ++defaulted_count;
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        snapshot.real_data[default_real * snapshot.count + p] == 0.0_prt &&
                            snapshot.int_data[default_int * snapshot.count + p] ==
                                native_initial_level,
                        "Omitted deterministic runtime attributes lost their native defaults.");
                }
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(defaulted_count == first_count,
                                             "Defaulted birth batch has the wrong local count.");
            candidate.Redistribute();

            int const target_box = [&candidate, target_rank] ()
            {
                auto const& boxes = candidate.ParticleBoxArray(0);
                auto const& distribution = candidate.ParticleDistributionMap(0);
                for (int grid = 0; grid < boxes.size(); ++grid)
                {
                    if (distribution[grid] == target_rank)
                    {
                        return grid;
                    }
                }
                return 0;
            }();
            BirthInput second;
            second.n = 2;
            second.x.resize(2, 0.25_rt);
            second.y.resize(2, 0.25_rt);
            second.z = {GridCellCenter(candidate, sim.Geom(0), target_box, 0),
                        GridCellCenter(candidate, sim.Geom(0), target_box, 1)};
            second.ux.resize(2, 0.0_rt);
            second.uy.resize(2, 0.0_rt);
            second.uz = {701.0_prt, 702.0_prt};
            second.attributes_real = {{601.5_prt, 602.5_prt}, {-12.25_prt, -13.5_prt}};
            second.attributes_int = {{-21, -22}};
            int const second_grid = LocalGrid(candidate, myproc);
            bool const added_second =
                txn.AddProvisionalBirths(second_grid >= 0 ? second_grid : 0, second.n, second.x,
                                         second.y, second.z, second.ux, second.uy, second.uz,
                                         second.attributes_real, second.attributes_int, false);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(added_first && added_second,
                                             "Repeated marker birth staging failed.");
            candidate.Redistribute();
            DeleteMarkedBirths(candidate, 501.0_prt, false);
            candidate.Redistribute();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(txn.ValidateLive(),
                                             "Repeated-birth candidate validation failed.");
            txn.Commit();
        }
        auto local_records = BirthRecords(live_photons, sentinel_real, sentinel_int, 600.0_prt);
        auto repeated_records = BroadcastRecords(local_records);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            repeated_records.size() == 2 &&
                live_photons.TotalNumberOfParticles() == repeated_births_before + 2,
            "Repeated-birth Commit lost old survivors or accepted the wrong new births.");
        for (auto const& record : repeated_records)
        {
            bool const first = std::abs(record.uz - 701.0_prt) < 0.5_prt;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                record.cpu == target_rank && record.weight == (first ? 601.5_prt : 602.5_prt) &&
                    record.sentinel_real == (first ? -12.25_prt : -13.5_prt) &&
                    record.sentinel_int == (first ? -21 : -22),
                "Repeated-birth payload or owner changed.");
            std::uint64_t id_value = record.idcpu;
            amrex::ParticleIDWrapper const id = id_value;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<amrex::Long>(id) != 1,
                                             "A repeated-batch placeholder ID escaped.");
        }
        all_ids = BroadcastRecords(AllIdCPUs(live_photons));
        AssertUniqueIdCPUs(all_ids);
        for (std::uint64_t id : old_ids)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::find(all_ids.begin(), all_ids.end(), id) != all_ids.end(),
                "A repeated-birth transaction dropped a survivor identity.");
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            ParticleType::NextID() ==
                repeated_id_probe + 1 + static_cast<amrex::Long>(local_records.size()),
            "Repeated-birth Commit consumed the wrong local number of native IDs.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(live_photons.NumIntComps() == default_int + 1 &&
                                             !HasBirthMarker(live_photons),
                                         "Repeated-birth marker leaked into live schema.");
        AssertNamedMetadata(live_photons, sentinel_real, sentinel_int);
        // Nested endpoint trials retain previously staged births and the
        // original live baseline. Rejection never allocates a native identity.
        amrex::Long const nested_probe = ParticleType::NextID();
        long const nested_local_births = local_grid >= 0 ? 2 : 0;
        auto const nested_live_before = Snapshot(live_photons);
        {
            PhotonPacketTransaction outer(live_photons);
            auto input = MakeBirthInput(local_grid >= 0 ? 1 : 0,
                local_grid >= 0 ? GridCellCenter(live_photons, sim.Geom(0), local_grid) : 0,
                3._prt, 29._prt, 41);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(outer.AddProvisionalBirths(
                local_grid >= 0 ? local_grid : 0, input.n, input.x, input.y, input.z,
                input.ux, input.uy, input.uz, input.attributes_real, input.attributes_int),
                "Initial nested fixture birth failed.");
            outer.Particles().Redistribute();
            auto const outer_before = Snapshot(outer.Particles());
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(outer.ValidateSnapshotFor(live_photons),
                "Valid marked private endpoint failed read-only snapshot preflight.");
            auto const& readonly_outer = static_cast<PhotonPacketTransaction const&>(outer);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(SameState(Snapshot(readonly_outer.Particles()),
                outer_before), "Read-only private access changed staged state.");
            {
                PhotonPacketTransaction discarded(live_photons, outer);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(SameState(Snapshot(discarded.Particles()),
                    outer_before), "Nested snapshot lost prior births or metadata.");
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(discarded.AddProvisionalBirths(
                    local_grid >= 0 ? local_grid : 0, input.n, input.x, input.y, input.z,
                    input.ux, input.uy, input.uz, input.attributes_real, input.attributes_int),
                    "Rejected nested fixture birth failed.");
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(SameState(Snapshot(outer.Particles()), outer_before) &&
                SameState(Snapshot(live_photons), nested_live_before),
                "Discarded nested trial changed the outer or live candidate.");
            PhotonPacketTransaction accepted(live_photons, outer);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(accepted.AddProvisionalBirths(
                local_grid >= 0 ? local_grid : 0, input.n, input.x, input.y, input.z,
                input.ux, input.uy, input.uz, input.attributes_real, input.attributes_int),
                "Accepted nested fixture birth failed.");
            accepted.Particles().Redistribute();
            auto const accepted_before = Snapshot(accepted.Particles());
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(outer.AdoptCandidate(accepted) &&
                SameState(Snapshot(outer.Particles()), accepted_before),
                "Private adoption lost the accepted nested state.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!accepted.ValidateSnapshotFor(live_photons),
                "Consumed nested candidate must reject recoverably.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ParticleType::NextID() == nested_probe + 1,
                "Nested clone/adoption or rejection consumed a native birth ID.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(outer.ValidateLive(), "Nested live preflight failed.");
            outer.Commit();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!outer.ValidateSnapshotFor(live_photons),
                "Published snapshot must reject recoverably.");
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            ParticleType::NextID() == nested_probe + 2 + nested_local_births,
            "Final nested publication did not allocate exactly the surviving birth count.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!HasBirthMarker(live_photons),
            "Nested publication leaked the provisional marker into live schema.");
        // Honest non-QED random-name policy rejection (no QED claim).
        live_photons.AddRealComp("opticalDepthQSR", 1);
        int const random_component = live_photons.NumRealComps() - 1;
        for (auto& [key, tile] : live_photons.GetParticles(0))
        {
            live_photons.DefineAndReturnParticleTile(0, key.first, key.second);
            auto* values = tile.GetStructOfArrays().GetRealData(random_component).data();
            amrex::ParallelFor(tile.numParticles(),
                               [=] AMREX_GPU_DEVICE(long p) { values[p] = 0.0_prt; });
        }
        amrex::Gpu::synchronize();
        {
            PhotonPacketTransaction txn(live_photons);
            auto const candidate_before = Snapshot(txn.Particles());
            auto const live_before = Snapshot(live_photons);
            BirthInput input = MakeBirthInput(0, 0.0_rt, 0.0_prt, 0.0_prt, 0);
            bool const added = txn.AddProvisionalBirths(
                local_grid >= 0 ? local_grid : 0, 0, input.x, input.y, input.z, input.ux, input.uy,
                input.uz, input.attributes_real, input.attributes_int, true);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !added, "An omitted random native default must be rejected before RNG use.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(SameState(Snapshot(txn.Particles()), candidate_before),
                                             "Random-default rejection changed the candidate.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(SameState(Snapshot(live_photons), live_before),
                                             "Random-default rejection changed live photons.");
        }

        amrex::Print() << "PhotonPacketTransaction native tests passed successfully!\n";
    }
    WarpX::Finalize();
    warpx::initialization::finalize_external_libraries();
    return 0;
}
