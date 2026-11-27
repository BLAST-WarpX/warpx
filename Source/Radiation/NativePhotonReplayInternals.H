/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NATIVEPHOTONREPLAYINTERNALS_H_
#define WARPX_RADIATION_NATIVEPHOTONREPLAYINTERNALS_H_

#include "Particles/WarpXParticleContainer.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace warpx::radiation::photon_replay_detail
{
    std::uint64_t SplitMix64 (std::uint64_t value) noexcept;
    void HashCombine (std::uint64_t& state, std::uint64_t value) noexcept;
    std::uint64_t PhotonDrawKey (
        std::uint64_t seed, std::uint64_t idcpu, std::uint64_t interval,
        std::uint64_t event, std::uint64_t draw) noexcept;
    amrex::Real OpenUnitInterval (std::uint64_t key) noexcept;
    int GlobalFailureCode (int local_code) noexcept;

    struct CapturedTile
    {
        std::pair<int, int> key{};
        long count = 0;
        amrex::Vector<amrex::ParticleReal> real_data;
        amrex::Vector<int> int_data;
        amrex::Vector<std::uint64_t> idcpu;
    };

    struct CapturedParticles
    {
        std::vector<CapturedTile> tiles;
    };

    CapturedParticles CaptureParticles (
        WarpXParticleContainer::Base const& particles);

    /** Collective read-only logical birth-key audit. Copies only the exact
     * two key columns, not all particle state or global keys. Absent columns
     * and zero/zero native-ID fallback retain existing semantics. */
    bool UniqueLogicalPhotonKeys (WarpXParticleContainer::Base const& particles);

    template <typename T>
    bool SameBits (amrex::Vector<T> const& a, amrex::Vector<T> const& b)
    {
        return a.size() == b.size() &&
            (a.empty() ||
             std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
    }

    bool NearestCell (
        amrex::Geometry const& geometry, amrex::ParticleReal x,
        amrex::ParticleReal y, amrex::ParticleReal z, amrex::IntVect& cell);
}

#endif
