/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "PhotonPacketTransaction.H"

#include "NativePhotonReplayInternals.H"
#include "Particles/PhotonParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "PhotonFourMomentum.H"
#include "PhotonLogicalDrawKey.H"
#include "RZAngularGeometry.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace warpx::radiation
{
namespace
{
constexpr amrex::Long kProvisionalBirthId = 1;
constexpr char const* kProvisionalBirthMarker = "__photon_birth_provisional";

bool
IsDiffusionCensusScratch (std::string const& name)
{
    return name == "radiation_diffusion_census_ux0" ||
           name == "radiation_diffusion_census_uy0" ||
           name == "radiation_diffusion_census_uz0";
}

bool
IsResolvedDrawKeyScratch (std::string const& name)
{
    return name == ResolvedDrawKeyLoName || name == ResolvedDrawKeyHiName;
}

using BirthMask = std::map<std::pair<int, int>, std::vector<bool>>;

struct StateSnapshot
{
    std::pair<int, int> key{};
    long count = 0;
    std::vector<amrex::ParticleReal> real_data;
    std::vector<int> int_data;
    std::vector<std::uint64_t> id_data;
};

template <typename T>
bool
SameBits (std::vector<T> const& left, std::vector<T> const& right)
{
    return left.size() == right.size() &&
           (left.empty() || std::memcmp(left.data(), right.data(), left.size() * sizeof(T)) == 0);
}

bool
SameLength (std::size_t vector_size, long expected)
{
    return expected >= 0 && vector_size == static_cast<std::size_t>(expected);
}

bool
CheckedAddLong (long left, long right, long& result)
{
    if (left < 0 || right < 0 || right > std::numeric_limits<long>::max() - left)
    {
        return false;
    }
    result = left + right;
    return true;
}

bool
StorageSizeFits (long count, int components)
{
    if (count < 0 || components < 0)
    {
        return false;
    }
    constexpr auto maximum_elements =
        std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t);
    return components == 0 || static_cast<unsigned long long>(count) <=
                                  maximum_elements / static_cast<std::size_t>(components);
}

std::size_t
CheckedStorageSize (long count, int components)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(StorageSizeFits(count, components),
                                     "Photon particle storage size exceeds the addressable range.");
    return static_cast<std::size_t>(count) * static_cast<std::size_t>(components);
}

template <typename ParticleTile>
void
AppendTileState (ParticleTile const& tile, StateSnapshot& snapshot)
{
    auto const& soa = tile.GetStructOfArrays();
    std::size_t const real_size = CheckedStorageSize(snapshot.count, tile.NumRealComps());
    std::size_t const int_size = CheckedStorageSize(snapshot.count, tile.NumIntComps());
    snapshot.real_data.resize(real_size);
    snapshot.int_data.resize(int_size);
    snapshot.id_data.resize(static_cast<std::size_t>(snapshot.count));
    for (int comp = 0; comp < tile.NumRealComps(); ++comp)
    {
        auto const& source = soa.GetRealData(comp);
        auto* destination =
            snapshot.real_data.data() + static_cast<std::size_t>(comp) * snapshot.count;
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, source.begin(), source.begin() + snapshot.count,
                         destination);
    }
    for (int comp = 0; comp < tile.NumIntComps(); ++comp)
    {
        auto const& source = soa.GetIntData(comp);
        auto* destination =
            snapshot.int_data.data() + static_cast<std::size_t>(comp) * snapshot.count;
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, source.begin(), source.begin() + snapshot.count,
                         destination);
    }
    auto const& source = soa.GetIdCPUData();
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, source.begin(), source.begin() + snapshot.count,
                     snapshot.id_data.begin());
}

std::vector<StateSnapshot>
Snapshot (WarpXParticleContainer::Base const& container)
{
    std::vector<StateSnapshot> snapshots;
    auto const& tiles = container.GetParticles(0);
    snapshots.reserve(tiles.size());
    for (auto const& [key, tile] : tiles)
    {
        StateSnapshot snapshot;
        snapshot.key = key;
        snapshot.count = tile.numParticles();
        if (snapshot.count != 0)
        {
            AppendTileState(tile, snapshot);
        }
        snapshots.push_back(std::move(snapshot));
    }
    return snapshots;
}

bool
SameState (std::vector<StateSnapshot> const& left, std::vector<StateSnapshot> const& right)
{
    if (left.size() != right.size())
    {
        return false;
    }
    for (std::size_t tile = 0; tile < left.size(); ++tile)
    {
        auto const& a = left[tile];
        auto const& b = right[tile];
        if (a.key != b.key || a.count != b.count || !SameBits(a.real_data, b.real_data) ||
            !SameBits(a.int_data, b.int_data) || !SameBits(a.id_data, b.id_data))
        {
            return false;
        }
    }
    return true;
}

bool
SameContainerLayout (WarpXParticleContainer::Base const& live,
                     WarpXParticleContainer::Base const& candidate, bool candidate_has_birth_marker)
{
    auto const live_int_names = live.GetIntSoANames();
    auto const candidate_int_names = candidate.GetIntSoANames();
    bool const int_schema_valid = candidate_has_birth_marker
                                      ? (candidate.NumIntComps() == live.NumIntComps() + 1 &&
                                         candidate_int_names.size() == live_int_names.size() + 1U &&
                                         candidate_int_names.back() == kProvisionalBirthMarker)
                                      : (candidate.NumIntComps() == live.NumIntComps() &&
                                         candidate_int_names == live_int_names);
    auto candidate_live_int_names = candidate_int_names;
    if (candidate_has_birth_marker && !candidate_live_int_names.empty())
    {
        candidate_live_int_names.pop_back();
    }
    return &live != &candidate && live.arena() == candidate.arena() &&
           live.NumRealComps() == candidate.NumRealComps() && int_schema_valid &&
           live.GetRealSoANames() == candidate.GetRealSoANames() &&
           live_int_names == candidate_live_int_names &&
           live.ParticleBoxArray(0) == candidate.ParticleBoxArray(0) &&
           live.ParticleDistributionMap(0) == candidate.ParticleDistributionMap(0);
}

template <typename Tile>
void
SetBirthMarker (Tile& tile, long start, long count, int value)
{
    if (count == 0)
    {
        return;
    }
    auto& marker = tile.GetStructOfArrays().GetIntData(tile.NumIntComps() - 1);
    int* values = marker.data() + start;
    amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(long i) noexcept { values[i] = value; });
    amrex::Gpu::streamSynchronize();
}

template <typename Tile>
std::vector<int>
HostIntComponent (Tile const& tile, int component)
{
    auto const& data = tile.GetStructOfArrays().GetIntData(component);
    std::size_t const count = CheckedStorageSize(tile.numParticles(), 1);
    std::vector<int> host(count);
    if (!host.empty())
    {
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, data.begin(), data.end(), host.begin());
    }
    return host;
}

template <typename Tile>
std::vector<std::uint64_t>
HostIdCPUComponent (Tile const& tile)
{
    auto const& data = tile.GetStructOfArrays().GetIdCPUData();
    std::size_t const count = CheckedStorageSize(tile.numParticles(), 1);
    std::vector<std::uint64_t> host(count);
    if (!host.empty())
    {
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, data.begin(), data.end(), host.begin());
    }
    return host;
}

bool
CollectBirthMask (WarpXParticleContainer::Base const& candidate, BirthMask& mask, long& birth_count)
{
    mask.clear();
    birth_count = 0;
    if (candidate.NumIntComps() == 0 || candidate.GetIntSoANames().empty() ||
        candidate.GetIntSoANames().back() != kProvisionalBirthMarker)
    {
        return false;
    }
    int const marker_component = candidate.NumIntComps() - 1;
    for (auto const& [key, tile] : candidate.GetParticles(0))
    {
        auto const marker = HostIntComponent(tile, marker_component);
        auto const ids = HostIdCPUComponent(tile);
        std::vector<bool> values(marker.size());
        for (std::size_t i = 0; i < marker.size(); ++i)
        {
            if (marker[i] != 0 && marker[i] != 1)
            {
                return false;
            }
            values[i] = marker[i] != 0;
            if (!values[i])
            {
                continue;
            }
            std::uint64_t id_value = ids[i];
            amrex::ParticleIDWrapper const wrapped_id = id_value;
            if (static_cast<amrex::Long>(wrapped_id) != kProvisionalBirthId)
            {
                return false;
            }
            if (!CheckedAddLong(birth_count, 1, birth_count))
            {
                return false;
            }
        }
        mask.emplace(key, std::move(values));
    }
    return true;
}

struct BirthInputPlan
{
    long ibegin = 0;
    long iend = 0;
    std::size_t local_count = 0;
    int external_real_count = 0;
    int external_int_count = 0;
};

bool
PlanBirthInput (WarpXParticleContainer& live, WarpXParticleContainer::Base& candidate,
                int staging_grid, long n, amrex::Vector<amrex::ParticleReal> const& x,
                amrex::Vector<amrex::ParticleReal> const& y,
                amrex::Vector<amrex::ParticleReal> const& z,
                amrex::Vector<amrex::ParticleReal> const& ux,
                amrex::Vector<amrex::ParticleReal> const& uy,
                amrex::Vector<amrex::ParticleReal> const& uz,
                amrex::Vector<amrex::Vector<amrex::ParticleReal>> const& attributes_real,
                amrex::Vector<amrex::Vector<int>> const& attributes_int, bool unique_particles,
                bool candidate_has_marker, BirthInputPlan& plan)
{
    if (n < 0 || attributes_real.empty() ||
        static_cast<unsigned long long>(attributes_real.size()) >
            static_cast<unsigned long long>(std::numeric_limits<int>::max()) ||
        static_cast<unsigned long long>(attributes_int.size()) >
            static_cast<unsigned long long>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    if (!SameLength(x.size(), n) || !SameLength(y.size(), n) || !SameLength(z.size(), n) ||
        !SameLength(ux.size(), n) || !SameLength(uy.size(), n) || !SameLength(uz.size(), n))
    {
        return false;
    }
    for (auto const& values : attributes_real)
    {
        if (!SameLength(values.size(), n))
        {
            return false;
        }
    }
    for (auto const& values : attributes_int)
    {
        if (!SameLength(values.size(), n))
        {
            return false;
        }
    }
    if (static_cast<std::size_t>(attributes_real.size()) - 1U >
            static_cast<std::size_t>(live.NumRuntimeRealComps()) ||
        static_cast<std::size_t>(attributes_int.size()) >
            static_cast<std::size_t>(live.NumRuntimeIntComps()))
    {
        return false;
    }

    auto const& box_array = candidate.ParticleBoxArray(0);
    auto const& distribution_map = candidate.ParticleDistributionMap(0);
    if (staging_grid < 0 || static_cast<long>(staging_grid) >= box_array.size())
    {
        return false;
    }

    long ibegin = 0;
    long iend = n;
    if (!unique_particles)
    {
        int const nprocs = amrex::ParallelDescriptor::NProcs();
        long const average = n / nprocs;
        long const remainder = n - average * nprocs;
        int const rank = amrex::ParallelDescriptor::MyProc();
        ibegin = rank * average + (rank < remainder ? rank : remainder);
        iend = ibegin + average + (rank < remainder ? 1 : 0);
    }
    if (ibegin < 0 || iend < ibegin || iend > n)
    {
        return false;
    }
    plan.ibegin = ibegin;
    plan.iend = iend;
    plan.local_count = static_cast<std::size_t>(iend - ibegin);
    plan.external_real_count = static_cast<int>(attributes_real.size() - 1U);
    plan.external_int_count = static_cast<int>(attributes_int.size());

    if (plan.local_count != 0 &&
        distribution_map[staging_grid] != amrex::ParallelDescriptor::MyProc())
    {
        return false;
    }
    auto const destination = candidate.GetParticles(0).find(std::make_pair(staging_grid, 0));
    if (destination != candidate.GetParticles(0).end())
    {
        long resized_count = 0;
        if (!CheckedAddLong(destination->second.numParticles(), static_cast<long>(plan.local_count),
                            resized_count))
        {
            return false;
        }
        if (!StorageSizeFits(resized_count, candidate.NumRealComps()) ||
            !StorageSizeFits(resized_count, candidate.NumIntComps()))
        {
            return false;
        }
    }

    if (!SameContainerLayout(live, candidate, candidate_has_marker))
    {
        return false;
    }
    if (!candidate_has_marker)
    {
        auto const& live_names = live.GetIntSoANames();
        if (std::find(live_names.begin(), live_names.end(), kProvisionalBirthMarker) !=
            live_names.end())
        {
            return false;
        }
    }

    // DoQED is the public native QED capability.  The two optical-depth names
    // are the public native runtime columns whose default initializer draws
    // random values. Unknown omitted attributes have no native initializer.
    if (live.DoQED())
    {
        return false;
    }
    auto const& real_names = live.GetRealSoANames();
    auto const user_real_names = live.getUserRealAttribs();
    for (std::size_t comp = static_cast<std::size_t>(PIdx::nattribs) +
                            static_cast<std::size_t>(plan.external_real_count);
         comp < real_names.size(); ++comp)
    {
        if (real_names[comp] == "opticalDepthQSR" || real_names[comp] == "opticalDepthBW")
        {
            return false;
        }
        if (!real_names[comp].starts_with("radiation_impulse_") &&
            !IsDiffusionCensusScratch(real_names[comp]) &&
            !IsResolvedDrawKeyScratch(real_names[comp]) &&
            std::find(user_real_names.begin(), user_real_names.end(), real_names[comp]) ==
                user_real_names.end())
        {
            return false;
        }
    }
    auto const int_names = live.GetIntSoANames();
    auto const user_int_names = live.getUserIntAttribs();
    for (std::size_t comp = static_cast<std::size_t>(plan.external_int_count);
         comp < int_names.size(); ++comp)
    {
        if (int_names[comp] != "ionizationLevel" &&
            std::find(user_int_names.begin(), user_int_names.end(), int_names[comp]) ==
                user_int_names.end())
        {
            return false;
        }
    }
    return true;
}

void
AppendProvisionalBirthBatch (
    WarpXParticleContainer& live, WarpXParticleContainer::Base& candidate, int staging_grid,
    BirthInputPlan const& plan, amrex::Vector<amrex::ParticleReal> const& x,
    amrex::Vector<amrex::ParticleReal> const& y, amrex::Vector<amrex::ParticleReal> const& z,
    amrex::Vector<amrex::ParticleReal> const& ux, amrex::Vector<amrex::ParticleReal> const& uy,
    amrex::Vector<amrex::ParticleReal> const& uz,
    amrex::Vector<amrex::Vector<amrex::ParticleReal>> const& attributes_real,
    amrex::Vector<amrex::Vector<int>> const& attributes_int)
{
    using namespace amrex::literals;
    using PinnedTile = typename WarpXParticleContainer::ContainerLike<
        amrex::PolymorphicArenaAllocator>::ParticleTileType;

    PinnedTile pinned_tile;
    auto real_names = live.GetRealSoANames();
    auto int_names = live.GetIntSoANames();
    int_names.push_back(kProvisionalBirthMarker);
    pinned_tile.define(live.NumRuntimeRealComps(), live.NumRuntimeIntComps() + 1, &real_names,
                       &int_names, amrex::The_Pinned_Arena());

    long const local_count = static_cast<long>(plan.local_count);
    for (long i = 0; i < local_count; ++i)
    {
        pinned_tile.GetStructOfArrays().GetIdCPUData().push_back(
            amrex::SetParticleIDandCPU(kProvisionalBirthId, amrex::ParallelDescriptor::MyProc()));
    }

    if (local_count != 0)
    {
        auto const* x_begin = x.data() + plan.ibegin;
        auto const* x_end = x.data() + plan.iend;
        auto const* y_begin = y.data() + plan.ibegin;
        auto const* y_end = y.data() + plan.iend;
        auto const* z_begin = z.data() + plan.ibegin;
        auto const* z_end = z.data() + plan.iend;

#if defined(WARPX_DIM_RZ)
        // Match WarpXParticleContainer::AppendParticleBatch's native RZ
        // encoding, while retaining provisional IDs and physical weights.
        amrex::Vector<amrex::ParticleReal> radius(plan.local_count);
        amrex::Vector<amrex::ParticleReal> theta(plan.local_count);
        for (long i = 0; i < local_count; ++i) {
            radius[i] = std::sqrt(x_begin[i]*x_begin[i] + y_begin[i]*y_begin[i]);
            theta[i] = std::atan2(y_begin[i], x_begin[i]);
        }
#endif
#if defined(WARPX_DIM_3D)
        pinned_tile.push_back_real(PIdx::x, x_begin, x_end);
        pinned_tile.push_back_real(PIdx::y, y_begin, y_end);
        pinned_tile.push_back_real(PIdx::z, z_begin, z_end);
#elif defined(WARPX_DIM_RZ)
        amrex::ignore_unused(x_end, y_end);
        pinned_tile.push_back_real(PIdx::r, radius.data(), radius.data() + local_count);
        pinned_tile.push_back_real(PIdx::z, z_begin, z_end);
#elif defined(WARPX_DIM_XZ)
        amrex::ignore_unused(y_begin, y_end);
        pinned_tile.push_back_real(PIdx::x, x_begin, x_end);
        pinned_tile.push_back_real(PIdx::z, z_begin, z_end);
#elif defined(WARPX_DIM_1D_Z)
        amrex::ignore_unused(x_begin, x_end, y_begin, y_end);
        pinned_tile.push_back_real(PIdx::z, z_begin, z_end);
#else
        amrex::ignore_unused(x_begin, x_end, y_begin, y_end, z_begin, z_end);
        WARPX_ABORT_WITH_MESSAGE("Provisional photon births require Cartesian geometry.");
#endif

        pinned_tile.push_back_real(PIdx::w, attributes_real.at(0).data() + plan.ibegin,
                                   attributes_real.at(0).data() + plan.iend);
        pinned_tile.push_back_real(PIdx::ux, ux.data() + plan.ibegin, ux.data() + plan.iend);
        pinned_tile.push_back_real(PIdx::uy, uy.data() + plan.ibegin, uy.data() + plan.iend);
        pinned_tile.push_back_real(PIdx::uz, uz.data() + plan.ibegin, uz.data() + plan.iend);
        for (int comp = PIdx::uz + 1; comp < PIdx::nattribs; ++comp)
        {
#if defined(WARPX_DIM_RZ)
            if (comp == PIdx::theta) {
                pinned_tile.push_back_real(comp, theta.data(), theta.data() + local_count);
                continue;
            }
#endif
            pinned_tile.push_back_real(comp, plan.local_count, 0.0_prt);
        }
        for (int j = 0; j < plan.external_real_count; ++j)
        {
            auto const& values = attributes_real.at(static_cast<std::size_t>(j) + 1U);
            pinned_tile.push_back_real(PIdx::nattribs + j, values.data() + plan.ibegin,
                                       values.data() + plan.iend);
        }
        for (int j = 0; j < plan.external_int_count; ++j)
        {
            auto const& values = attributes_int.at(static_cast<std::size_t>(j));
            pinned_tile.push_back_int(j, values.data() + plan.ibegin, values.data() + plan.iend);
        }
    }

    pinned_tile.resize(local_count);
    live.DefaultInitializeRuntimeAttributes(pinned_tile, plan.external_real_count,
                                            plan.external_int_count);
    // Only these exact reserved columns may default to zero. An unkeyed
    // low-level birth retains native-ID draws; resolved runtime requires
    // nonzero keys for its leakage plans before entering this stage.
    for (std::size_t comp = static_cast<std::size_t>(PIdx::nattribs) +
                            static_cast<std::size_t>(plan.external_real_count);
         comp < real_names.size(); ++comp)
    {
        if (IsDiffusionCensusScratch(real_names[comp]) ||
            IsResolvedDrawKeyScratch(real_names[comp]))
        {
            auto* values = pinned_tile.GetStructOfArrays()
                .GetRealData(static_cast<int>(comp)).dataPtr();
            amrex::ParallelFor(local_count, [=] AMREX_GPU_DEVICE(long i) noexcept {
                values[i] = 0.0_prt;
            });
        }
    }
    SetBirthMarker(pinned_tile, 0, local_count, 1);

    if (local_count != 0)
    {
        auto& destination = candidate.DefineAndReturnParticleTile(0, staging_grid, 0);
        long const old_count = destination.numParticles();
        destination.resize(old_count + local_count);
        amrex::copyParticles(destination, pinned_tile, long{0}, old_count, local_count);
    }
    amrex::Gpu::streamSynchronize();
}

void
CopyLiveSchemaParticles (WarpXParticleContainer::Base const& staged,
                         WarpXParticleContainer::Base& accepted)
{
    for (auto const& [key, source] : staged.GetParticles(0))
    {
        auto& destination = accepted.DefineAndReturnParticleTile(0, key.first, key.second);
        std::size_t const count = static_cast<std::size_t>(source.numParticles());
        destination.resize(static_cast<long>(count));
        if (count == 0)
        {
            continue;
        }
        auto const& source_soa = source.GetStructOfArrays();
        auto& destination_soa = destination.GetStructOfArrays();
        for (int comp = 0; comp < accepted.NumRealComps(); ++comp)
        {
            amrex::Gpu::copy(amrex::Gpu::deviceToDevice, source_soa.GetRealData(comp).begin(),
                             source_soa.GetRealData(comp).begin() + count,
                             destination_soa.GetRealData(comp).begin());
        }
        // The final staged integer is the temporary birth marker and is
        // intentionally discarded before accepted storage is exposed.
        for (int comp = 0; comp < accepted.NumIntComps(); ++comp)
        {
            amrex::Gpu::copy(amrex::Gpu::deviceToDevice, source_soa.GetIntData(comp).begin(),
                             source_soa.GetIntData(comp).begin() + count,
                             destination_soa.GetIntData(comp).begin());
        }
        amrex::Gpu::copy(amrex::Gpu::deviceToDevice, source_soa.GetIdCPUData().begin(),
                         source_soa.GetIdCPUData().begin() + count,
                         destination_soa.GetIdCPUData().begin());
    }
    amrex::Gpu::streamSynchronize();
}

void
AssignFinalBirthIds (WarpXParticleContainer::Base& accepted, BirthMask const& mask,
                     long local_birth_count)
{
    using ParticleType = WarpXParticleContainer::ParticleType;
    int const owner_cpu = amrex::ParallelDescriptor::MyProc();
    long assigned_count = 0;
    for (auto const& [key, values] : mask)
    {
        auto const destination = accepted.GetParticles(0).find(key);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(destination != accepted.GetParticles(0).end(),
                                         "Internal error finding accepted photon births.");
        auto& idcpu = destination->second.GetStructOfArrays().GetIdCPUData();
        std::size_t const count = static_cast<std::size_t>(destination->second.numParticles());
        std::vector<std::uint64_t> ids(count);
        if (!ids.empty())
        {
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, idcpu.begin(), idcpu.end(), ids.begin());
        }
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            if (!values[i])
            {
                continue;
            }
            // AMReX exposes no non-allocating allocator peek.  Exhaustion is
            // therefore a fatal infrastructure condition inside NextID().
            amrex::Long const native_id = ParticleType::NextID();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                native_id > 0 && native_id <= amrex::LongParticleIds::LastParticleID,
                "Native photon ID allocation returned an invalid value.");
            ids[i] = amrex::SetParticleIDandCPU(native_id, owner_cpu);
            ++assigned_count;
        }
        if (!ids.empty())
        {
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, ids.begin(), ids.end(), idcpu.begin());
        }
    }
    amrex::Gpu::streamSynchronize();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        assigned_count == local_birth_count,
        "Internal error assigning native identities to photon births.");
}
} // namespace

struct PhotonPacketTransaction::Impl
{
    WarpXParticleContainer* live = nullptr;
    std::unique_ptr<WarpXParticleContainer::Base> candidate;
    std::vector<StateSnapshot> live_baseline;
    std::vector<StateSnapshot> accepted_baseline;
    bool has_birth_marker = false;
    bool staging_failed = false;
    bool validated = false;
    bool committed = false;
};

PhotonPacketTransaction::PhotonPacketTransaction (WarpXParticleContainer& live_photons)
    : m_impl(std::make_unique<Impl>())
{
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    WARPX_ABORT_WITH_MESSAGE("PhotonPacketTransaction is not yet qualified for radial builds.");
#endif
    auto const& warpx = WarpX::GetInstance();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(dynamic_cast<PhotonParticleContainer*>(&live_photons) !=
                                         nullptr,
                                     "PhotonPacketTransaction requires a native photon species.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(warpx.finestLevel() == 0,
                                     "PhotonPacketTransaction currently requires level zero.");
#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (warpx.Geom(0).Coord() == 0 || warpx.Geom(0).Coord() == 1) &&
        warpx.Geom(0).ProbLo(0) >= 0,
        "PhotonPacketTransaction requires cylindrical geometry in an RZ build.");
#else
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(warpx.Geom(0).Coord() == 0,
        "PhotonPacketTransaction requires Cartesian geometry in this build.");
#endif
    m_impl->live = &live_photons;
    m_impl->candidate = std::make_unique<WarpXParticleContainer::Base>(live_photons.make_alike<>());
    // make_alike duplicates the SoA schema but not the polymorphic arena.
    m_impl->candidate->SetArena(live_photons.arena());
    m_impl->candidate->copyParticles(live_photons, true);
    m_impl->live_baseline = Snapshot(live_photons);
}

PhotonPacketTransaction::PhotonPacketTransaction (
    WarpXParticleContainer& live_photons, PhotonPacketTransaction const& private_snapshot)
    : m_impl(std::make_unique<Impl>())
{
    auto const& source = *private_snapshot.m_impl;
    bool valid = private_snapshot.ValidateSnapshotFor(live_photons);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(valid,
        "A nested photon trial requires unchanged live provenance and a valid private snapshot.");
    m_impl->live = &live_photons;
    m_impl->live_baseline = source.live_baseline;
    m_impl->has_birth_marker = source.has_birth_marker;
    m_impl->candidate =
        std::make_unique<WarpXParticleContainer::Base>(source.candidate->make_alike<>());
    m_impl->candidate->SetArena(source.candidate->arena());
    m_impl->candidate->copyParticles(*source.candidate, true);
    valid = SameState(Snapshot(*m_impl->candidate), Snapshot(*source.candidate));
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(valid, "Nested photon clone changed private particle state.");
}

PhotonPacketTransaction::~PhotonPacketTransaction () = default;

WarpXParticleContainer::Base&
PhotonPacketTransaction::Particles ()
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_impl->candidate != nullptr,
                                     "PhotonPacketTransaction has no private candidate.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_impl->committed, "The photon packet candidate no longer accepts trial mutation.");
    return *m_impl->candidate;
}

bool
PhotonPacketTransaction::AdoptCandidate (PhotonPacketTransaction& accepted_private_trial)
{
    auto& destination = *m_impl;
    auto& source = *accepted_private_trial.m_impl;
    bool valid = this != &accepted_private_trial && destination.live == source.live &&
        destination.candidate && source.candidate && !destination.committed && !source.committed &&
        !destination.staging_failed && !source.staging_failed &&
        SameState(destination.live_baseline, source.live_baseline);
    if (valid) {
        valid = SameContainerLayout(*destination.live, *source.candidate, source.has_birth_marker) &&
            SameState(Snapshot(*destination.live), destination.live_baseline);
        BirthMask mask;
        long count = 0;
        valid = (!source.has_birth_marker || CollectBirthMask(*source.candidate, mask, count)) && valid;
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { return false; }
    destination.candidate = std::move(source.candidate);
    destination.has_birth_marker = source.has_birth_marker;
    destination.accepted_baseline.clear();
    destination.validated = false;
    source.validated = false;
    source.accepted_baseline.clear();
    return true;
}

WarpXParticleContainer::Base const&
PhotonPacketTransaction::Particles () const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_impl->candidate && !m_impl->committed,
        "The photon transaction has no readable uncommitted private candidate.");
    return *m_impl->candidate;
}

bool
PhotonPacketTransaction::ValidateSnapshotFor (WarpXParticleContainer const& live_owner) const
{
    auto const& source = *m_impl;
    bool valid = source.live == &live_owner && source.candidate &&
        !source.committed && !source.staging_failed;
    if (valid)
    {
        valid = SameContainerLayout(live_owner, *source.candidate, source.has_birth_marker) &&
            SameState(Snapshot(live_owner), source.live_baseline);
        BirthMask mask;
        long count = 0;
        valid = (!source.has_birth_marker || CollectBirthMask(*source.candidate, mask, count)) &&
            valid;
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid;
}

bool
PhotonPacketTransaction::AddProvisionalBirths (
    int staging_grid, long n, amrex::Vector<amrex::ParticleReal> const& x,
    amrex::Vector<amrex::ParticleReal> const& y, amrex::Vector<amrex::ParticleReal> const& z,
    amrex::Vector<amrex::ParticleReal> const& ux, amrex::Vector<amrex::ParticleReal> const& uy,
    amrex::Vector<amrex::ParticleReal> const& uz,
    amrex::Vector<amrex::Vector<amrex::ParticleReal>> const& attributes_real,
    amrex::Vector<amrex::Vector<int>> const& attributes_int, bool unique_particles)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_impl->committed,
                                     "A committed photon transaction cannot stage births.");
    m_impl->validated = false;
    if (m_impl->staging_failed)
    {
        return false;
    }

    // Every check (including the exact stale-live comparison) happens before
    // marker registration, tile definition, destination resize, or native
    // default initialization.  The reductions make rejection collective.
    BirthInputPlan plan;
    bool local_valid = PlanBirthInput(*m_impl->live, *m_impl->candidate, staging_grid, n, x, y, z,
                                      ux, uy, uz, attributes_real, attributes_int, unique_particles,
                                      m_impl->has_birth_marker, plan) &&
                       SameState(Snapshot(*m_impl->live), m_impl->live_baseline);

    long root_n = n;
    amrex::ParallelDescriptor::Bcast(&root_n, 1, 0);
    int const unique_flag = unique_particles ? 1 : 0;
    int root_unique_flag = unique_flag;
    amrex::ParallelDescriptor::Bcast(&root_unique_flag, 1, 0);
    local_valid =
        local_valid && (unique_particles || n == root_n) && (unique_flag == root_unique_flag);

    int component_min = plan.external_real_count;
    int component_max = plan.external_real_count;
    amrex::ParallelDescriptor::ReduceIntMin(component_min);
    amrex::ParallelDescriptor::ReduceIntMax(component_max);
    local_valid = local_valid && component_min == component_max;

    int int_component_min = plan.external_int_count;
    int int_component_max = plan.external_int_count;
    amrex::ParallelDescriptor::ReduceIntMin(int_component_min);
    amrex::ParallelDescriptor::ReduceIntMax(int_component_max);
    local_valid = local_valid && int_component_min == int_component_max;

    amrex::ParallelDescriptor::ReduceBoolAnd(local_valid);
    if (!local_valid)
    {
        m_impl->staging_failed = true;
        return false;
    }

    if (!m_impl->has_birth_marker)
    {
        m_impl->candidate->AddIntComp(kProvisionalBirthMarker, /*communicate=*/1);
        m_impl->has_birth_marker = true;
        for (auto& [key, tile] : m_impl->candidate->GetParticles(0))
        {
            SetBirthMarker(tile, 0, tile.numParticles(), 0);
            amrex::ignore_unused(key);
        }
    }

    AppendProvisionalBirthBatch(*m_impl->live, *m_impl->candidate, staging_grid, plan, x, y, z, ux,
                                uy, uz, attributes_real, attributes_int);
    return true;
}

bool
PhotonPacketTransaction::ProvisionalBirthInventory (
    amrex::MultiFab& output, amrex::Geometry const& geometry, amrex::Long& count,
    std::vector<std::uint64_t> const* selected_keys) const
{
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    amrex::ignore_unused(output, geometry, count, selected_keys);
    return false;
#else
    bool valid = m_impl->candidate != nullptr && !m_impl->committed && !m_impl->staging_failed;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) {
        return false;
    }
    auto const& candidate = *m_impl->candidate;
    valid =
#if defined(WARPX_DIM_RZ)
        (geometry.Coord() == 0 || geometry.Coord() == 1) && geometry.ProbLo(0) >= 0 &&
        !geometry.isPeriodic(0) &&
#else
        geometry.Coord() == 0 &&
#endif
        output.nComp() == 4 && output.ixType().cellCentered() &&
        output.boxArray() == candidate.ParticleBoxArray(0) &&
        output.DistributionMap() == candidate.ParticleDistributionMap(0) &&
        geometry.Domain() == candidate.Geom(0).Domain();
    std::unordered_set<std::uint64_t> selected;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        valid = valid && std::isfinite(geometry.ProbLo(d)) && std::isfinite(geometry.ProbHi(d)) &&
                geometry.ProbHi(d) > geometry.ProbLo(d) && std::isfinite(geometry.CellSize(d)) &&
                geometry.CellSize(d) > 0 && geometry.ProbLo(d) == candidate.Geom(0).ProbLo(d) &&
                geometry.ProbHi(d) == candidate.Geom(0).ProbHi(d) &&
                geometry.isPeriodic(d) == candidate.Geom(0).isPeriodic(d);
    }
    if (selected_keys) {
        for (auto const key : *selected_keys) {
            valid = valid && key != 0 && selected.insert(key).second;
        }
    }
    std::array<int, 2> key_components{-1, -1};
    bool const empty_selection = selected_keys && selected_keys->empty();
    valid = valid && (!selected_keys || empty_selection ||
                      (FindPhotonDrawKeyComponents(candidate.GetRealSoANames(), key_components) &&
                       key_components[0] >= 0));
    valid = valid && (!m_impl->has_birth_marker ||
                      (!candidate.GetIntSoANames().empty() &&
                       candidate.GetIntSoANames().back() == kProvisionalBirthMarker));
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) {
        return false;
    }
    amrex::MultiFab host(output.boxArray(), output.DistributionMap(), 4, output.nGrowVect(),
                         amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    host.setVal(0);
    amrex::Long births = 0;
    if (m_impl->has_birth_marker && !empty_selection) {
        for (auto const& [tile_key, tile] : candidate.GetParticles(0)) {
            auto const marker = HostIntComponent(tile, candidate.NumIntComps() - 1);
            auto const ids = HostIdCPUComponent(tile);
            auto real = [&tile] (int component) {
                auto const& data = tile.GetStructOfArrays().GetRealData(component);
                std::vector<amrex::ParticleReal> values(tile.numParticles());
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, data.begin(),
                                 data.begin() + tile.numParticles(), values.begin());
                return values;
            };
            auto const w = real(PIdx::w), ux = real(PIdx::ux), uy = real(PIdx::uy),
                       uz = real(PIdx::uz);
            auto const z = real(PIdx::z);
#if defined(WARPX_DIM_RZ)
            auto const x = real(PIdx::r);
#elif AMREX_SPACEDIM >= 2
            auto const x = real(PIdx::x);
#endif
#if defined(WARPX_DIM_RZ)
            auto const theta = real(PIdx::theta);
#endif
#if AMREX_SPACEDIM == 3
            auto const y = real(PIdx::y);
#endif
            std::vector<amrex::ParticleReal> low, high;
            if (selected_keys) {
                low = real(key_components[0]);
                high = real(key_components[1]);
            }
            auto const inventory = host.array(tile_key.first);
            auto const& box = host.boxArray()[tile_key.first];
            for (long n = 0; n < tile.numParticles(); ++n) {
                valid = valid && (marker[n] == 0 || marker[n] == 1);
                if (marker[n] == 0 || !amrex::ParticleIDWrapper{ids[n]}.is_valid()) {
                    continue;
                }
                if (selected_keys) {
                    std::uint64_t key = 0;
                    valid = valid && DecodePhotonDrawKey(low[n], high[n], ids[n], key);
                    if (!selected.contains(key)) {
                        continue;
                    }
                }
                auto const packet =
                    PacketConversionFourMomentumFromPhoton(w[n], ux[n], uy[n], uz[n]);
                amrex::IntVect cell;
                amrex::ParticleReal px = 0, py = 0;
#if AMREX_SPACEDIM >= 2
                px = x[n];
#endif
#if AMREX_SPACEDIM == 3
                py = y[n];
#endif
                bool owned = false;
#if defined(WARPX_DIM_RZ)
                amrex::ignore_unused(py);
                bool const in_domain = std::isfinite(px) && std::isfinite(z[n]) &&
                                       px >= geometry.ProbLo(0) && px < geometry.ProbHi(0) &&
                                       z[n] >= geometry.ProbLo(1) && z[n] < geometry.ProbHi(1);
                if (in_domain) {
                    cell = geometry.Domain().smallEnd();
                    cell[0] += static_cast<int>(
                        std::floor((px - geometry.ProbLo(0)) / geometry.CellSize(0)));
                    cell[1] += static_cast<int>(
                        std::floor((z[n] - geometry.ProbLo(1)) / geometry.CellSize(1)));
                    owned = box.contains(cell);
                }
                valid = valid && std::isfinite(theta[n]);
#else
                owned = photon_replay_detail::NearestCell(geometry, px, py, z[n], cell) &&
                        box.contains(cell);
#endif
                valid = valid && owned && packet.valid;
                if (!owned || !packet.valid) {
                    continue;
                }
#if defined(WARPX_DIM_RZ)
                auto const lo = geometry.ProbLo(0) +
                                (cell[0] - geometry.Domain().smallEnd(0)) * geometry.CellSize(0);
                auto const mean = RZAngularGeometry::MeanRadius(lo, lo + geometry.CellSize(0));
                auto const cosine = std::cos(theta[n]), sine = std::sin(theta[n]);
                auto const radial = packet.value[1] * cosine + packet.value[2] * sine;
                auto const tangent = -packet.value[1] * sine + packet.value[2] * cosine;
                inventory(cell, 0) += packet.value[0];
                inventory(cell, 1) += radial;
                inventory(cell, 2) += px * tangent / mean;
                inventory(cell, 3) += packet.value[3];
#else
                for (int d = 0; d < 4; ++d) {
                    inventory(cell, d) += packet.value[d];
                }
#endif
                ++births;
            }
        }
    }
    valid = valid && host.is_finite(0, 4, 0, true);
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) {
        return false;
    }
    amrex::ParallelDescriptor::ReduceLongSum(births);
    amrex::htod_memcpy(output, host);
    amrex::Gpu::streamSynchronize();
    count = births;
    return true;
#endif
}

bool
PhotonPacketTransaction::ValidateLive ()
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_impl->committed, "A committed photon packet transaction cannot be validated again.");
    m_impl->validated = false;
    bool local_valid =
        !m_impl->staging_failed &&
        SameContainerLayout(*m_impl->live, *m_impl->candidate, m_impl->has_birth_marker) &&
        SameState(Snapshot(*m_impl->live), m_impl->live_baseline);

    // Capture the exact candidate state (including deletion, compaction and
    // reordering) after all caller-side native mutation has finished.
    m_impl->accepted_baseline = Snapshot(*m_impl->candidate);
    BirthMask marker_mask;
    long marker_count = 0;
    local_valid = local_valid && (!m_impl->has_birth_marker ||
                                  CollectBirthMask(*m_impl->candidate, marker_mask, marker_count));
    amrex::ParallelDescriptor::ReduceBoolAnd(local_valid);
    if (!local_valid)
    {
        m_impl->accepted_baseline.clear();
        // Keep every rank at the same validation side effect.
        amrex::ParallelDescriptor::Barrier();
        return false;
    }
    m_impl->validated = true;
    return true;
}

void
PhotonPacketTransaction::Commit ()
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_impl->validated && !m_impl->committed,
        "Only a collectively validated, uncommitted photon candidate may commit.");
    auto& live = *m_impl->live;
    auto& candidate = *m_impl->candidate;
    BirthMask marker_mask;
    long marker_count = 0;
    bool unchanged =
        !m_impl->staging_failed && SameContainerLayout(live, candidate, m_impl->has_birth_marker) &&
        SameState(Snapshot(live), m_impl->live_baseline) &&
        SameState(Snapshot(candidate), m_impl->accepted_baseline) &&
        (!m_impl->has_birth_marker || CollectBirthMask(candidate, marker_mask, marker_count));
    amrex::ParallelDescriptor::ReduceBoolAnd(unchanged);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        unchanged, "Live photons or the validated photon candidate changed before Commit.");

    if (m_impl->has_birth_marker)
    {
        auto accepted = std::make_unique<WarpXParticleContainer::Base>(live.make_alike<>());
        accepted->SetArena(live.arena());
        CopyLiveSchemaParticles(candidate, *accepted);
        AssignFinalBirthIds(*accepted, marker_mask, marker_count);
        live.GetParticles(0).swap(accepted->GetParticles(0));
    }
    else
    {
        live.GetParticles(0).swap(candidate.GetParticles(0));
    }

    // Exchange the final live-schema tile map.  SoA name vectors are pointers
    // owned by their container, so every swapped tile must be rebound to live.
    m_impl->committed = true;
    m_impl->validated = false;
    m_impl->live_baseline.clear();
    m_impl->accepted_baseline.clear();
    for (auto const& [key, tile] : live.GetParticles(0))
    {
        live.DefineAndReturnParticleTile(0, key.first, key.second);
        amrex::ignore_unused(tile);
    }
}
} // namespace warpx::radiation
