/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_PACKETMOMENTRECONSTRUCTION_H_
#define WARPX_RADIATION_PACKETMOMENTRECONSTRUCTION_H_

#include "FrameTransform.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>

#include <cmath>
#include <limits>

namespace warpx::radiation
{
struct PacketPhoton
{
    amrex::GpuArray<amrex::ParticleReal, 3> direction{};
    amrex::GpuArray<amrex::ParticleReal, 3> proper_velocity{};
    // Statistical photon number, not an energy fraction.
    amrex::ParticleReal energy_weight = 0;
};

struct PacketMomentReconstructionResult
{
    amrex::GpuArray<PacketPhoton, 8> photons{};
    FourVector reconstructed_four_momentum{};
    // Actual represented output minus the requested state, in joules.
    FourVector signed_residual{};
    bool valid = false;
};

/** Eight rays at one common position, matching E and q=c*p. The cone
 * quadrature also matches the M1 pressure tensor; at q=0 it is isotropic,
 * unlike a single antipodal beam pair. This is an angular reconstruction,
 * not an M1 spatial solve. Its finite-flux pressure differs from a P1
 * transport closure and is an explicit outgoing sampling approximation.
 *
 * photon_energy is the nominal energy of one physical photon [J]. WarpX
 * stores dimensional u=p/m_e [m/s], so photon energy is m_e*c*|u|, NOT
 * m_e*c*c*|u|. Weights use the actual stored u's energy. Floating-point
 * residuals are returned and must be checked/accounted before conversion.
 * Acausal states are rejected outside the same eight-epsilon arithmetic
 * collar as EvaluateM1Closure. A rounded beam inside that collar is sampled
 * using its limiting beam angular distribution; input conserved moments are
 * never modified. The actual represented difference remains signed_residual,
 * and the caller must audit it against its unchanged conversion/step bounds.
 * No impulse is assigned to ions by this helper.
 */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE PacketMomentReconstructionResult
ReconstructPacketMoments (FourVector const& moments, amrex::Real photon_energy) noexcept
{
    PacketMomentReconstructionResult out;
    if (!amrex::Math::isfinite(photon_energy) || photon_energy <= 0)
    {
        return out;
    }
    for (auto value : moments)
    {
        if (!amrex::Math::isfinite(value))
        {
            return out;
        }
    }
    auto const energy = moments[0];
    if (energy < 0)
    {
        return out;
    }
    auto const scale = amrex::max(std::abs(moments[1]), std::abs(moments[2]), std::abs(moments[3]));
    if (energy == 0)
    {
        out.valid = scale == 0;
        return out;
    }
    amrex::Real norm = 0;
    amrex::GpuArray<amrex::Real, 3> n{0, 0, 1}, e1{1, 0, 0}, e2{0, 1, 0};
    if (scale > 0)
    {
        // Match the constitutive M1 norm and its existing arithmetic domain.
        // This is not an inner-cone floor or a change to conserved input q.
        norm = std::hypot(std::hypot(moments[1], moments[2]), moments[3]);
        constexpr amrex::Real roundoff = 8 * std::numeric_limits<amrex::Real>::epsilon();
        if (!amrex::Math::isfinite(norm) || norm / energy > 1 + roundoff || norm == 0)
        {
            return out;
        }
        for (int d = 0; d < 3; ++d)
        {
            n[d] = moments[d + 1] / norm;
        }
        int axis = 0;
        if (std::abs(n[1]) < std::abs(n[axis]))
        {
            axis = 1;
        }
        if (std::abs(n[2]) < std::abs(n[axis]))
        {
            axis = 2;
        }
        amrex::GpuArray<amrex::Real, 3> reference{};
        reference[axis] = 1;
        amrex::Real length = 0;
        for (int d = 0; d < 3; ++d)
        {
            e1[d] =
                n[(d + 1) % 3] * reference[(d + 2) % 3] - n[(d + 2) % 3] * reference[(d + 1) % 3];
            length += e1[d] * e1[d];
        }
        length = std::sqrt(length);
        for (int d = 0; d < 3; ++d)
        {
            e1[d] /= length;
        }
        // All e1 components must be normalized before forming the second basis.
        for (int d = 0; d < 3; ++d)
        {
            e2[d] = n[(d + 1) % 3] * e1[(d + 2) % 3] - n[(d + 2) % 3] * e1[(d + 1) % 3];
        }
    }
    // Above one only a precision-sized limiting beam is representable. Keep
    // the requested moments intact so its complete signed residual is audited.
    auto const f = norm > energy ? amrex::Real(1) : norm / energy;
    auto const chi = (3 + 4 * f * f) / (5 + 2 * std::sqrt(4 - 3 * f * f));
    // Analytically chi>=f^2. Preserve that inequality at the last rounding
    // bit near a beam; this does not change input E or q or clip acausality.
    auto const a = amrex::max(f, std::sqrt(chi));
    auto const b = std::sqrt(1 - a * a);
    auto const u_scale = photon_energy / (PhysConst::m_e * PhysConst::c);
    if (!amrex::Math::isfinite(u_scale) || u_scale <= 0)
    {
        return out;
    }

    for (int ray = 0; ray < 4; ++ray)
    {
        auto const& transverse = ray < 2 ? e1 : e2;
        amrex::Real const sign = ray % 2 == 0 ? 1 : -1;
        auto& upper = out.photons[ray];
        auto& lower = out.photons[ray + 4];
        amrex::Real u2 = 0;
        for (int d = 0; d < 3; ++d)
        {
            upper.direction[d] =
                static_cast<amrex::ParticleReal>(a * n[d] + sign * b * transverse[d]);
            lower.direction[d] = -upper.direction[d];
            upper.proper_velocity[d] =
                static_cast<amrex::ParticleReal>(u_scale * upper.direction[d]);
            lower.proper_velocity[d] = -upper.proper_velocity[d];
            u2 += upper.proper_velocity[d] * upper.proper_velocity[d];
        }
        auto const actual_photon_energy = PhysConst::m_e * PhysConst::c * std::sqrt(u2);
        if (!amrex::Math::isfinite(actual_photon_energy) || actual_photon_energy <= 0)
        {
            return {};
        }
        upper.energy_weight =
            static_cast<amrex::ParticleReal>((energy / 8 / actual_photon_energy) * (1 + f / a));
        lower.energy_weight =
            static_cast<amrex::ParticleReal>((energy / 8 / actual_photon_energy) * (1 - f / a));
        if (!amrex::Math::isfinite(upper.energy_weight) || upper.energy_weight < 0 ||
            !amrex::Math::isfinite(lower.energy_weight) || lower.energy_weight < 0)
        {
            return {};
        }
    }
    for (auto const& photon : out.photons)
    {
        amrex::Real u2 = 0;
        for (int d = 0; d < 3; ++d)
        {
            u2 += photon.proper_velocity[d] * photon.proper_velocity[d];
        }
        out.reconstructed_four_momentum[0] +=
            photon.energy_weight * PhysConst::m_e * PhysConst::c * std::sqrt(u2);
        for (int d = 0; d < 3; ++d)
        {
            out.reconstructed_four_momentum[d + 1] +=
                photon.energy_weight * PhysConst::m_e * PhysConst::c * photon.proper_velocity[d];
        }
    }
    for (int d = 0; d < 4; ++d)
    {
        out.signed_residual[d] = out.reconstructed_four_momentum[d] - moments[d];
        if (!amrex::Math::isfinite(out.signed_residual[d]))
        {
            return {};
        }
    }
    out.valid = true;
    return out;
}
} // namespace warpx::radiation
#endif
