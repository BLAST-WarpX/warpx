/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_PHOTONLOGICALDRAWKEY_H_
#define WARPX_RADIATION_PHOTONLOGICALDRAWKEY_H_

#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace warpx::radiation
{
inline constexpr char ResolvedDrawKeyLoName[] = "radiation_resolved_draw_key_lo32";
inline constexpr char ResolvedDrawKeyHiName[] = "radiation_resolved_draw_key_hi32";

struct LogicalPhotonBirthKeyContext
{
    std::uint64_t seed = 0;
    std::uint64_t interval = 0;
};

/** The two exact 32-bit integers are communicated and checkpointed as double
 * particle reals. Both zero means an original/unkeyed photon: retain its
 * native idcpu draw stream. A malformed nonzero pair is a replay rejection. */
inline bool DecodePhotonDrawKey (amrex::ParticleReal low, amrex::ParticleReal high,
                                 std::uint64_t native_idcpu, std::uint64_t& draw_key)
{
    if (std::numeric_limits<amrex::ParticleReal>::digits < 32) { return false; }
    constexpr auto maximum = static_cast<amrex::ParticleReal>(
        std::numeric_limits<std::uint32_t>::max());
    if (!std::isfinite(low) || !std::isfinite(high) || low < 0 || high < 0 ||
        low > maximum || high > maximum || std::floor(low) != low ||
        std::floor(high) != high) {
        return false;
    }
    auto const lo = static_cast<std::uint32_t>(low);
    auto const hi = static_cast<std::uint32_t>(high);
    draw_key = (lo == 0 && hi == 0) ? native_idcpu :
        (static_cast<std::uint64_t>(hi) << 32U) | lo;
    return true;
}

inline std::array<amrex::ParticleReal, 2> EncodePhotonDrawKey (std::uint64_t key)
{
    return {static_cast<amrex::ParticleReal>(static_cast<std::uint32_t>(key)),
            static_cast<amrex::ParticleReal>(static_cast<std::uint32_t>(key >> 32U))};
}

/** Find the exact reserved pair. An absent pair retains legacy native-ID draws;
 * a half-registered pair is invalid, never a fallback. */
inline bool FindPhotonDrawKeyComponents (std::vector<std::string> const& names,
                                         std::array<int, 2>& components)
{
    auto const low = std::find(names.begin(), names.end(), ResolvedDrawKeyLoName);
    auto const high = std::find(names.begin(), names.end(), ResolvedDrawKeyHiName);
    if ((low == names.end()) != (high == names.end())) { return false; }
    components = low == names.end() ? std::array<int, 2>{-1, -1} :
        std::array<int, 2>{static_cast<int>(low - names.begin()),
                           static_cast<int>(high - names.begin())};
    return true;
}

inline std::uint64_t MixPhotonBirthKey (std::uint64_t value) noexcept
{
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

/** Canonical face/ray identity is independent of rank, box, photon weight,
 * momentum, and AMReX's native birth allocator. The normal high periodic
 * endpoint aliases the low endpoint. The mixed 64-bit key is nonzero so a
 * newborn never takes the original-photon native-ID fallback. */
inline bool MakePhotonBirthDrawKey (amrex::Geometry const& geometry,
                                    amrex::IntVect face, int normal_axis, int side,
                                    int ray, LogicalPhotonBirthKeyContext const& context,
                                    std::uint64_t& key)
{
    if (normal_axis < 0 || normal_axis >= AMREX_SPACEDIM ||
        (side != -1 && side != 1) || ray < 0 || ray >= 8) { return false; }
    auto const& domain = geometry.Domain();
    std::uint64_t site = 0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        int const lo = domain.smallEnd(d);
        int const upper = domain.bigEnd(d);
        if (upper == std::numeric_limits<int>::max()) { return false; }
        if (d == normal_axis && geometry.isPeriodic(d) && face[d] == upper + 1) {
            face[d] = lo;
        }
        int const hi = upper + (d == normal_axis && !geometry.isPeriodic(d) ? 1 : 0);
        if (face[d] < lo || face[d] > hi) { return false; }
        auto const extent = static_cast<std::uint64_t>(static_cast<std::int64_t>(hi) - lo) + 1U;
        auto const offset = static_cast<std::uint64_t>(static_cast<std::int64_t>(face[d]) - lo);
        if (site > (std::numeric_limits<std::uint64_t>::max() - offset) / extent) {
            return false;
        }
        site = site * extent + offset;
    }
    auto const label = static_cast<std::uint64_t>(16 * normal_axis +
        (side > 0 ? 8 : 0) + ray);
    auto const labels = static_cast<std::uint64_t>(16 * AMREX_SPACEDIM);
    if (site > (std::numeric_limits<std::uint64_t>::max() - label) / labels) {
        return false;
    }
    site = site * labels + label;
    key = MixPhotonBirthKey(
        MixPhotonBirthKey(context.seed ^ UINT64_C(0x72872fc4a39b65e1)) ^
        MixPhotonBirthKey(context.interval ^ UINT64_C(0x4f1bbcdc676ff3b3)) ^ site);
    if (key == 0) { key = UINT64_C(0x9e3779b97f4a7c15); }
    return true;
}
} // namespace warpx::radiation

#endif
