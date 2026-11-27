/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativePhotonReplayInternals.H"

#include "PhotonKeyAudit.H"
#include "PhotonLogicalDrawKey.H"

#include <AMReX_GpuContainers.H>

#include <cmath>
#include <limits>

namespace warpx::radiation::photon_replay_detail
{
using namespace amrex::literals;

namespace
{
constexpr std::uint64_t split_mix_gamma = 0x9e3779b97f4a7c15ULL;
}

std::uint64_t
SplitMix64 (std::uint64_t value) noexcept
{
    value += split_mix_gamma;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

void
HashCombine (std::uint64_t& state, std::uint64_t value) noexcept
{
    state = SplitMix64(state ^ (value + split_mix_gamma));
}

std::uint64_t
PhotonDrawKey (
    std::uint64_t seed, std::uint64_t idcpu, std::uint64_t interval,
    std::uint64_t event, std::uint64_t draw) noexcept
{
    std::uint64_t state = SplitMix64(seed);
    HashCombine(state, idcpu);
    HashCombine(state, interval);
    HashCombine(state, event);
    HashCombine(state, draw);
    return SplitMix64(state);
}

amrex::Real
OpenUnitInterval (std::uint64_t key) noexcept
{
    std::uint64_t const top_52 = key >> 12U;
    return (static_cast<amrex::Real>(top_52) + 0.5_rt) *
           std::ldexp(1.0_rt, -52);
}

int
GlobalFailureCode (int local_code) noexcept
{
    int const sentinel = std::numeric_limits<int>::max();
    int code = local_code == 0 ? sentinel : local_code;
    amrex::ParallelDescriptor::ReduceIntMin(code);
    return code == sentinel ? 0 : code;
}

CapturedParticles
CaptureParticles (WarpXParticleContainer::Base const& particles)
{
    CapturedParticles captured;
    auto const& tiles = particles.GetParticles(0);
    captured.tiles.reserve(tiles.size());
    for (auto const& [key, tile] : tiles)
    {
        CapturedTile entry;
        entry.key = key;
        entry.count = tile.numParticles();
        if (entry.count != 0)
        {
            auto const& soa = tile.GetStructOfArrays();
            entry.real_data.resize(static_cast<std::size_t>(entry.count) *
                                   particles.NumRealComps());
            entry.int_data.resize(static_cast<std::size_t>(entry.count) *
                                  particles.NumIntComps());
            entry.idcpu.resize(static_cast<std::size_t>(entry.count));
            for (int component = 0; component < particles.NumRealComps();
                 ++component)
            {
                auto const& values = soa.GetRealData(component);
                amrex::Gpu::copy(
                    amrex::Gpu::deviceToHost, values.begin(),
                    values.begin() + entry.count,
                    entry.real_data.begin() +
                        static_cast<std::size_t>(component) *
                            static_cast<std::size_t>(entry.count));
            }
            for (int component = 0; component < particles.NumIntComps();
                 ++component)
            {
                auto const& values = soa.GetIntData(component);
                amrex::Gpu::copy(
                    amrex::Gpu::deviceToHost, values.begin(),
                    values.begin() + entry.count,
                    entry.int_data.begin() +
                        static_cast<std::size_t>(component) *
                            static_cast<std::size_t>(entry.count));
            }
            auto const& ids = soa.GetIdCPUData();
            amrex::Gpu::copy(
                amrex::Gpu::deviceToHost, ids.begin(), ids.begin() + entry.count,
                entry.idcpu.begin());
        }
        captured.tiles.push_back(std::move(entry));
    }
    return captured;
}

bool
UniqueLogicalPhotonKeys (WarpXParticleContainer::Base const& particles)
{
    std::array<int, 2> components{-1, -1};
    bool valid = FindPhotonDrawKeyComponents(particles.GetRealSoANames(), components);
    std::vector<PhotonAuditIdentity> identities;
    if (valid && components[0] >= 0)
    {
        for (auto const& [tile_key, tile] : particles.GetParticles(0))
        {
            amrex::ignore_unused(tile_key);
            auto const count = tile.numParticles();
            if (count < 0 ||
                static_cast<unsigned long long>(count) > std::numeric_limits<std::size_t>::max())
            {
                valid = false;
                continue;
            }
            std::vector<amrex::ParticleReal> low(static_cast<std::size_t>(count));
            std::vector<amrex::ParticleReal> high(low.size());
            auto const& soa = tile.GetStructOfArrays();
            auto const& stored_low = soa.GetRealData(components[0]);
            auto const& stored_high = soa.GetRealData(components[1]);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, stored_low.begin(),
                             stored_low.begin() + count, low.begin());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, stored_high.begin(),
                             stored_high.begin() + count, high.begin());
            for (std::size_t n = 0; n < low.size(); ++n)
            {
                if (low[n] == 0 && high[n] == 0)
                {
                    continue;
                }
                std::uint64_t key = 0;
                // The native fallback is deliberately absent from this audit:
                // only explicit nonzero logical identities were checked before.
                auto const decoded = DecodePhotonDrawKey(low[n], high[n], 0, key) && key != 0;
                valid = valid && decoded;
                if (decoded)
                {
                    identities.emplace_back(1, key);
                }
            }
        }
    }
    return DistributedPhotonIdentitiesUnique(identities, valid);
}

bool
NearestCell (
    amrex::Geometry const& geometry, amrex::ParticleReal x,
    amrex::ParticleReal y, amrex::ParticleReal z, amrex::IntVect& cell)
{
    amrex::ignore_unused(x, y, z);
    amrex::Real coordinate[3] = {0.0_rt, 0.0_rt, 0.0_rt};
#if defined(WARPX_DIM_3D)
    coordinate[0] = static_cast<amrex::Real>(x);
    coordinate[1] = static_cast<amrex::Real>(y);
    coordinate[2] = static_cast<amrex::Real>(z);
#elif defined(WARPX_DIM_XZ)
    coordinate[0] = static_cast<amrex::Real>(x);
    coordinate[1] = static_cast<amrex::Real>(z);
#elif defined(WARPX_DIM_1D_Z)
    coordinate[0] = static_cast<amrex::Real>(z);
#else
    cell = amrex::IntVect::TheUnitVector();
    return false;
#endif
    cell = geometry.Domain().smallEnd();
    for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
    {
        amrex::Real const position = coordinate[dimension];
        amrex::Real const lower = geometry.ProbLo(dimension);
        amrex::Real const upper = geometry.ProbHi(dimension);
        if (!amrex::Math::isfinite(position) || !(position >= lower) ||
            !(position < upper))
        {
            return false;
        }
        long const index = static_cast<long>(std::floor(
            (position - lower) / geometry.CellSize(dimension)));
        if (index < static_cast<long>(std::numeric_limits<int>::min()) ||
            index > static_cast<long>(std::numeric_limits<int>::max()))
        {
            return false;
        }
        cell[dimension] = geometry.Domain().smallEnd(dimension) +
                           static_cast<int>(index);
    }
    return true;
}
}
