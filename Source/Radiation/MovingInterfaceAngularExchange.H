/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_MOVINGINTERFACEANGULAREXCHANGE_H_
#define WARPX_RADIATION_MOVINGINTERFACEANGULAREXCHANGE_H_

#include "FiniteMassPhotonCollision.H"

#include <cmath>
#include <limits>
#include <vector>

namespace warpx::radiation
{
struct InterfaceAngularRay
{
    amrex::GpuArray<amrex::Real, 3> frame_direction{};
    amrex::Real fraction = 0;
};

struct StoredInterfaceAngularRay
{
    amrex::GpuArray<amrex::ParticleReal, 3> u{};
    amrex::ParticleReal weight = 0;
    FourVector represented{};
};

struct MovingInterfaceAngularExchangeResult
{
    bool valid = false;
    std::vector<StoredInterfaceAngularRay> rays;
    FourVector outgoing_total{};
    amrex::Real radiation_debit = 0;
    amrex::GpuArray<amrex::Real, 3> material_impulse{};
};

/** Host-only coherent angular exchange for one directed lab photon [E,c*p].
 * Fractions partition its energy; they must sum to one and are never normalized.
 * nominal_photon_energy [J] selects native |u|=epsilon/(m_e*c), not macro energy.
 * beta is the staged material secant velocity/c. Outgoing directions are comoving;
 * every stored lab ray must point into the specified Cartesian hemisphere.
 * Actual u/weight inventory, including storage rounding, defines debit and recoil.
 * This is not a moving Milne boundary law, finite-mass work proof, RZ model or
 * permission for scalar admission. Failed results contain no rays or exchanges.
 */
[[nodiscard]] inline MovingInterfaceAngularExchangeResult
TryMovingInterfaceAngularExchange (
    FourVector const& incoming,
    amrex::GpuArray<amrex::Real, 3> const& beta,
    std::vector<InterfaceAngularRay> const& quadrature,
    amrex::Real nominal_photon_energy, int normal_axis, int outgoing_sign)
{
    MovingInterfaceAngularExchangeResult failed;
    if (!ValidFiniteMassPhoton(incoming) || quadrature.empty() || quadrature.size() > 64 ||
        normal_axis < 0 || normal_axis > 2 ||
        (outgoing_sign != -1 && outgoing_sign != 1) ||
        !(nominal_photon_energy > 0) || !std::isfinite(nominal_photon_energy))
    {
        return failed;
    }
    long double const mc = static_cast<long double>(PhysConst::m_e) * PhysConst::c;
    amrex::Real const scale = static_cast<amrex::Real>(nominal_photon_energy / mc);
    if (!(scale > 0) || !std::isfinite(scale)) { return failed; }
    long double sum_fraction = 0;
    for (auto const& ray : quadrature)
    {
        if (!(ray.fraction > 0) || !std::isfinite(ray.fraction)) { return failed; }
        sum_fraction += ray.fraction;
    }
    if (std::abs(sum_fraction - 1) > 256 * std::numeric_limits<amrex::Real>::epsilon())
    {
        return failed;
    }
    MovingInterfaceAngularExchangeResult candidate;
    long double total[4]{};
    for (auto const& ray : quadrature)
    {
        FourVector part{};
        for (int k = 0; k < 4; ++k) { part[k] = incoming[k] * ray.fraction; }
        auto const collision = CoherentPhotonCollision(part, ray.frame_direction, beta);
        if (!collision.valid) { return failed; }
        StoredInterfaceAngularRay stored;
        auto const& lab = collision.outgoing_photon;
        stored.weight = static_cast<amrex::ParticleReal>(lab[0] / (mc * scale));
        if (!(stored.weight > 0) || !std::isfinite(stored.weight)) { return failed; }
        for (int d = 0; d < 3; ++d)
        {
            stored.u[d] = static_cast<amrex::ParticleReal>(scale * (lab[d + 1] / lab[0]));
            if (!std::isfinite(stored.u[d])) { return failed; }
        }
        if (!(outgoing_sign * stored.u[normal_axis] > 0)) { return failed; }
        long double const speed = std::hypot(static_cast<long double>(stored.u[0]),
            static_cast<long double>(stored.u[1]), static_cast<long double>(stored.u[2]));
        long double represented[4]{stored.weight * mc * speed, 0, 0, 0};
        for (int d = 0; d < 3; ++d) { represented[d + 1] = stored.weight * mc * stored.u[d]; }
        for (int k = 0; k < 4; ++k)
        {
            stored.represented[k] = static_cast<amrex::Real>(represented[k]);
            total[k] += represented[k];
        }
        if (!ValidFiniteMassPhoton(stored.represented)) { return failed; }
        candidate.rays.push_back(stored);
    }
    for (int k = 0; k < 4; ++k)
    {
        candidate.outgoing_total[k] = static_cast<amrex::Real>(total[k]);
        if (!std::isfinite(candidate.outgoing_total[k])) { return failed; }
    }
    candidate.radiation_debit = static_cast<amrex::Real>(incoming[0] - total[0]);
    if (!std::isfinite(candidate.radiation_debit)) { return failed; }
    for (int d = 0; d < 3; ++d)
    {
        candidate.material_impulse[d] =
            static_cast<amrex::Real>((incoming[d + 1] - total[d + 1]) / PhysConst::c);
        if (!std::isfinite(candidate.material_impulse[d])) { return failed; }
    }
    candidate.valid = true;
    return candidate;
}
}
#endif
