/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_PACKETCONVERSIONSAMPLING_H_
#define WARPX_RADIATION_PACKETCONVERSIONSAMPLING_H_

#include "Utils/TextMsg.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace warpx::radiation::detail
{
using namespace amrex::literals;

// Deterministic sampling for the packet-FLD bridge. The coordinator owns when
// conversion is committed; these helpers only compute keys and positions.
[[nodiscard]] inline
std::uint64_t
SplitMix64 (std::uint64_t value) noexcept
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

[[nodiscard]] inline
std::uint64_t
ParticleConversionKey (
    std::uint64_t const seed,
    int const step,
    int const i,
    int const j,
    int const k,
    int const group,
    int const packet) noexcept
{
    std::uint64_t key = SplitMix64(seed);
    auto const combine = [&key] (std::uint64_t const value) noexcept
    {
        key = SplitMix64(key ^ SplitMix64(value));
    };
    combine(static_cast<std::uint64_t>(static_cast<std::int64_t>(step)));
    combine(static_cast<std::uint64_t>(static_cast<std::int64_t>(i)));
    combine(static_cast<std::uint64_t>(static_cast<std::int64_t>(j)));
    combine(static_cast<std::uint64_t>(static_cast<std::int64_t>(k)));
    combine(static_cast<std::uint64_t>(group));
    combine(static_cast<std::uint64_t>(packet));
    return key;
}

[[nodiscard]] inline
amrex::ParticleReal
ParticleConversionUniform (
    std::uint64_t const key, std::uint64_t const draw) noexcept
{
    std::uint64_t const value = SplitMix64(key ^ SplitMix64(draw));
    amrex::ParticleReal const result =
        static_cast<amrex::ParticleReal>(value >> 11U) * 0x1.0p-53_prt;
    return amrex::min(result, std::nextafter(1.0_prt, 0.0_prt));
}

[[nodiscard]] inline
amrex::ParticleReal
ClampParticleConversionPosition (
    amrex::Real const sampled,
    amrex::Real const cell_lo,
    amrex::Real const cell_hi,
    int const inward_steps = 0) noexcept
{
    using ComparisonReal =
        std::common_type_t<amrex::Real, amrex::ParticleReal>;
    auto lo = static_cast<amrex::ParticleReal>(cell_lo);
    auto hi = static_cast<amrex::ParticleReal>(cell_hi);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        amrex::Math::isfinite(sampled)
            && amrex::Math::isfinite(cell_lo)
            && amrex::Math::isfinite(cell_hi)
            && cell_hi > cell_lo
            && amrex::Math::isfinite(lo) && amrex::Math::isfinite(hi)
            && hi > lo && inward_steps >= 0,
        "A diffusion-to-streaming conversion cell is not representable in "
        "particle-position precision.");

    // Casting a Real face to ParticleReal can round out of the original cell.
    // Move each endpoint to the first representable position inside that cell.
    if (static_cast<ComparisonReal>(lo)
        < static_cast<ComparisonReal>(cell_lo))
    {
        lo = std::nextafter(lo, hi);
    }
    if (static_cast<ComparisonReal>(hi)
        >= static_cast<ComparisonReal>(cell_hi))
    {
        hi = std::nextafter(hi, lo);
    }

    // Radial insertion reconstructs r from Cartesian components. Additional
    // symmetric margins keep norm/trigonometric roundoff away from both faces.
    for (int step = 0; step < inward_steps; ++step) {
        auto const next_lo = std::nextafter(lo, hi);
        auto const next_hi = std::nextafter(hi, lo);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            next_lo <= next_hi,
            "A diffusion-to-streaming conversion cell has insufficient "
            "particle-position precision for the requested inward margin.");
        lo = next_lo;
        hi = next_hi;
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<ComparisonReal>(lo)
                >= static_cast<ComparisonReal>(cell_lo)
            && static_cast<ComparisonReal>(hi)
                < static_cast<ComparisonReal>(cell_hi)
            && lo <= hi,
        "A diffusion-to-streaming conversion cell has no representable "
        "particle position in its physical interior.");
    return amrex::max(
        lo, amrex::min(hi,
            static_cast<amrex::ParticleReal>(sampled)));
}

[[maybe_unused, nodiscard]] inline
amrex::ParticleReal
ParticleConversionCoordinate (
    amrex::Real const cell_lo,
    amrex::Real const cell_hi,
    amrex::ParticleReal const unit_interval) noexcept
{
    amrex::Real const sampled =
        cell_lo + static_cast<amrex::Real>(unit_interval)
            * (cell_hi - cell_lo);
    return ClampParticleConversionPosition(sampled, cell_lo, cell_hi);
}

// Radial particle insertion reconstructs r from sampled Cartesian components.
// Leave several ULPs inside both faces so trig/norm roundoff cannot move it out.
[[maybe_unused]] constexpr int particle_conversion_radial_margin_ulps = 8;

} // namespace warpx::radiation::detail

#endif
