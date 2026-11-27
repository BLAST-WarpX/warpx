/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_PHOTONFOURMOMENTUM_H_
#define WARPX_RADIATION_PHOTONFOURMOMENTUM_H_

#include "FrameTransform.H"
#include "Utils/WarpXConst.H"
#include <AMReX_Math.H>
#include <cmath>
#include <limits>

namespace warpx::radiation
{
/** Actual native packet [E,c*p], in joules. */
struct PacketConversionFourMomentum
{
    FourVector value{};
    bool valid = false;
};

/** Separately scaled native kinematics and rounded extensive accounts. Used
 * only when the original positive weight*m_e intermediate is subnormal/zero.
 * A positive native photon may have a zero rounded extensive account; it still
 * owns its native weight, direction and identity. This is not a cutoff. */
struct ScaledNativePhotonMomentum
{
    FourVector represented{}, physical{}, absolute_error{};
    amrex::Real u_scale = 0;
    bool valid = false;
};

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
NeedsScaledNativePhotonMomentum (amrex::ParticleReal weight) noexcept
{
    return amrex::Math::isfinite(weight) && weight > 0 &&
        weight * PhysConst::m_e < std::numeric_limits<amrex::Real>::min();
}

namespace photon_momentum_detail
{
/** frexp is exact for represented binary inputs. Three normal mantissa
 * multiplications form cP; energy adds scale and a bounded sqrt mantissa. The
 * final ldexp is exact unless its result is subnormal (one final rounding).
 * No small intermediate can destroy a representable final product. */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
Product (amrex::ParticleReal weight, amrex::Real component,
         amrex::Real norm = 1) noexcept
{
    if (component == 0) { return component; }
    int ew = 0, em = 0, ec = 0, eu = 0, en = 0;
    auto const w = std::frexp(static_cast<amrex::Real>(weight), &ew);
    auto const m = std::frexp(PhysConst::m_e, &em);
    auto const c = std::frexp(PhysConst::c, &ec);
    auto const u = std::frexp(component, &eu);
    auto const n = std::frexp(norm, &en);
    return std::ldexp(((w * m) * c) * u * n, ew + em + ec + eu + en);
}

/** γ_k bounds k normal roundings. Re-express its relative error using the
 * represented result: γ_k/(1-γ_k). One denorm_min covers the final half-ULP
 * ldexp rounding, including a result that rounds to zero. The three outward
 * nextafter operations prevent bound arithmetic from rounding inward. */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
FormationBound (amrex::Real value, int operations) noexcept
{
    auto const unit = std::numeric_limits<amrex::Real>::epsilon() / 2;
    auto const infinity = std::numeric_limits<amrex::Real>::infinity();
    // γ_k/(1-γ_k) = k*u/(1-2*k*u). Both numerator and denominator
    // are exact binary values for these fixed k=3/9 counts.
    auto const relative = std::nextafter(
        operations * unit / (1 - 2 * operations * unit), infinity);
    auto const normal = std::nextafter(relative * std::abs(value), infinity);
    return std::nextafter(normal + std::numeric_limits<amrex::Real>::denorm_min(), infinity);
}
} // namespace photon_momentum_detail

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ScaledNativePhotonMomentum
ScaledNativePhotonMomentumFromPhoton (
    amrex::ParticleReal weight, amrex::ParticleReal ux,
    amrex::ParticleReal uy, amrex::ParticleReal uz) noexcept
{
    ScaledNativePhotonMomentum result;
    if (!NeedsScaledNativePhotonMomentum(weight)) { return result; }
    amrex::GpuArray<amrex::Real, 3> const u{ux, uy, uz};
    for (auto value : u)
    {
        if (!amrex::Math::isfinite(value)) { return result; }
        result.u_scale = amrex::max(result.u_scale, std::abs(value));
    }
    if (!(result.u_scale > 0)) { return result; }
    amrex::Real norm2 = 0;
    for (int d = 0; d < 3; ++d)
    {
        result.physical[d + 1] = u[d] / result.u_scale;
        norm2 += result.physical[d + 1] * result.physical[d + 1];
        result.represented[d + 1] = photon_momentum_detail::Product(weight, u[d]);
        // Product includes the exact norm=1 mantissa multiplication by .5,
        // so only its three nontrivial mantissa products incur rounding.
        result.absolute_error[d + 1] =
            photon_momentum_detail::FormationBound(result.represented[d + 1], 3);
    }
    result.physical[0] = std::sqrt(norm2);
    result.represented[0] =
        photon_momentum_detail::Product(weight, result.u_scale, result.physical[0]);
    // Each squared normalized component has γ3 error (division twice and
    // multiplication), followed by two positive additions: γ5 in norm2.
    // sqrt propagation plus <=1 ULP libm error is bounded by γ5; four
    // nontrivial mantissa products then give γ9 for represented energy.
    result.absolute_error[0] =
        photon_momentum_detail::FormationBound(result.represented[0], 9);
    for (int q = 0; q < 4; ++q)
    {
        if (!amrex::Math::isfinite(result.represented[q]) ||
            !amrex::Math::isfinite(result.absolute_error[q])) { return {}; }
    }
    result.valid = true;
    return result;
}

/** Arithmetic consistency of a rounded extensive null vector. The physical
 * null-vector predicate is separately applied unchanged to physical above.
 * Reverse triangle inequality bounds the norm error by sum(delta cP), plus
 * delta E. Normalized norm evaluation has the same derived γ9 formation
 * bound; this local rounding check does not change a physical tolerance. */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
ConsistentScaledNativePhotonAccounts (ScaledNativePhotonMomentum const& photon) noexcept
{
    if (!photon.valid || photon.represented[0] < 0) { return false; }
    amrex::Real scale = photon.represented[0];
    for (int d = 1; d < 4; ++d)
    {
        scale = amrex::max(scale, std::abs(photon.represented[d]));
    }
    if (scale == 0) { return true; }
    amrex::Real norm2 = 0, bound = photon.absolute_error[0] / scale;
    for (int d = 1; d < 4; ++d)
    {
        auto const component = photon.represented[d] / scale;
        norm2 += component * component;
        bound += photon.absolute_error[d] / scale;
    }
    auto const norm = std::sqrt(norm2);
    bound += photon_momentum_detail::FormationBound(norm, 9);
    return amrex::Math::isfinite(norm) && amrex::Math::isfinite(bound) &&
        std::abs(norm - photon.represented[0] / scale) <= bound;
}
/** Native u is dimensional p/m_e [m/s], not beta or u/c. */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE PacketConversionFourMomentum
PacketConversionFourMomentumFromPhoton (amrex::ParticleReal weight, amrex::ParticleReal ux,
                                        amrex::ParticleReal uy, amrex::ParticleReal uz) noexcept
{
    PacketConversionFourMomentum result;
    amrex::GpuArray<amrex::Real, 3> const u{ux, uy, uz};
    if (!amrex::Math::isfinite(weight) || weight <= 0)
    {
        return result;
    }
    amrex::Real scale = 0;
    for (int d = 0; d < 3; ++d)
    {
        if (!amrex::Math::isfinite(u[d]))
        {
            return result;
        }
        scale = amrex::max(scale, std::abs(u[d]));
    }
    if (scale == 0)
    {
        return result;
    }
    amrex::Real norm2 = 0;
    for (int d = 0; d < 3; ++d)
    {
        norm2 += (u[d] / scale) * (u[d] / scale);
    }
    if (NeedsScaledNativePhotonMomentum(weight))
    {
        auto const scaled = ScaledNativePhotonMomentumFromPhoton(weight, ux, uy, uz);
        result.value = scaled.represented;
        result.valid = scaled.valid && result.value[0] > 0;
        return result;
    }
    auto const factor = weight * PhysConst::m_e * PhysConst::c;
    result.value[0] = factor * scale * std::sqrt(norm2);
    for (int d = 0; d < 3; ++d)
    {
        result.value[d + 1] = factor * u[d];
    }
    for (auto value : result.value)
    {
        if (!amrex::Math::isfinite(value))
        {
            return {};
        }
    }
    result.valid = result.value[0] > 0;
    return result;
}
} // namespace warpx::radiation
#endif
