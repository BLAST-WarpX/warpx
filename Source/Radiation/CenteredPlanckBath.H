/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_CENTEREDPLANCKBATH_H_
#define WARPX_RADIATION_CENTEREDPLANCKBATH_H_

#include "ExactPlanckBinInteraction.H"

namespace warpx::radiation::centered_bath_detail
{
using Pressure = amrex::GpuArray<amrex::Real, 6 * exact_group_maximum>;

AMREX_GPU_HOST_DEVICE inline amrex::Real Density (amrex::Real x)
{
    return exact_group_detail::PlanckDensity(x);
}

// Positive panels, never a subtraction of nearby Planck tails. Width is
// supplied independently of the lower endpoint so narrow Doppler intervals
// retain their physical width. Broad intervals use unit-width Gauss16 panels;
// widths <=1/4 use Gauss8. The exponentially tiny x>64 remainder uses Tail.
AMREX_GPU_HOST_DEVICE inline amrex::Real Interval (amrex::Real low, amrex::Real width)
{
    return exact_group_detail::PositivePlanckInterval(low, width);
}

/** Smooth LTE bath LAB-bin density moments/pressure. COMOVING proper bins
 * jointly form one continuous Planck spectrum; there are no opacity-weighted
 * source terms here. Paired +/-mu angular nodes evaluate the longitudinal
 * flux using positive narrow edge integrals, not cancellation of O(B) angular
 * terms. This is a separately opt-in evaluation of the same LTE spectrum,
 * not a source reconciliation. Invalid outputs remain untouched. */
AMREX_GPU_HOST_DEVICE inline bool Evaluate (EnergyGroupsExecutor const &groups,
                                            amrex::GpuArray<amrex::Real, 3> const &beta,
                                            amrex::Real temperature, ExactGroupVector &moments,
                                            Pressure &pressure, int order = 16)
{
    auto const n = groups.m_num_groups;
    auto const speed2 = exact_group_detail::Dot(beta, beta);
    if (!spectral_detail::GroupsValid(groups) || n > exact_group_maximum || !(temperature > 0) ||
        !amrex::Math::isfinite(temperature) || !(speed2 <= 1.e-4_rt) ||
        !amrex::Math::isfinite(speed2) || (order != 8 && order != 16))
    {
        return false;
    }
    auto const bath = spectral_detail::Blackbody(temperature);
    auto const thermal = PhysConst::kb * temperature;
    if (!(bath > 0) || !amrex::Math::isfinite(bath) || !(thermal > 0) ||
        !amrex::Math::isfinite(thermal))
    {
        return false;
    }
    ExactGroupVector next{};
    Pressure stress{};
    if (speed2 == 0 || n == 1)
    {
        auto const gamma2 = 1 / (1 - speed2);
        for (int g = 0; g < n; ++g)
        {
            auto const energy = bath * spectral_detail::Fraction(groups, g, temperature);
            next[4 * g] = energy * (4 * gamma2 - 1) / 3;
            for (int d = 0; d < 3; ++d)
            {
                next[4 * g + d + 1] = energy * (4 * gamma2 / 3) * beta[d];
                for (int e = d; e < 3; ++e)
                {
                    stress[6 * g + exact_group_detail::PressureComponent(d, e)] =
                        energy * ((d == e ? 1.0_rt / 3 : 0) + 4 * gamma2 * beta[d] * beta[e] / 3);
                }
            }
        }
    }
    else
    {
        auto const speed = std::sqrt(speed2), gamma2 = 1 / (1 - speed2);
        auto const gamma = std::sqrt(gamma2);
        for (int g = 0; g < n; ++g)
        {
            auto const lower = g == 0 ? 0 : groups.m_boundaries[g - 1] / thermal;
            auto const upper = g + 1 == n ? 0 : groups.m_boundaries[g] / thermal;
            auto const width =
                g + 1 == n ? 0
                           : (groups.m_boundaries[g] - (g == 0 ? 0 : groups.m_boundaries[g - 1])) /
                                 thermal;
            amrex::Real energy = 0, flux = 0, parallel = 0, perpendicular = 0;
            for (int p = order / 2; p < order; ++p)
            {
                auto const rule = exact_group_detail::GaussianNodeWeight(order, p);
                auto const mu = rule[0], weight = rule[1] / 2;
                auto const shift = speed * mu;
                auto const plus = gamma * (1 + shift), minus = gamma * (1 - shift);
                auto const positive = g + 1 == n ? spectral_detail::Tail(lower / plus)
                                                 : Interval(lower / plus, width / plus);
                auto const negative = g + 1 == n ? spectral_detail::Tail(lower / minus)
                                                 : Interval(lower / minus, width / minus);
                // 1/D_minus - 1/D_plus, without subtracting adjacent floats.
                auto const inverse_width = 2 * shift / (gamma * (1 - shift * shift));
                auto const difference =
                    Interval(lower / plus, lower * inverse_width) -
                    (g + 1 == n ? 0 : Interval(upper / plus, upper * inverse_width));
                energy += weight * (plus * plus * positive + minus * minus * negative);
                flux += weight * gamma2 *
                        (mu * (1 + speed2) * difference +
                         speed * (1 + mu * mu) * (positive + negative));
                parallel += weight * gamma2 *
                            ((mu + speed) * (mu + speed) * positive +
                             (mu - speed) * (mu - speed) * negative);
                perpendicular += weight * (1 - mu * mu) * (positive + negative) / 2;
            }
            next[4 * g] = bath * energy;
            for (int d = 0; d < 3; ++d)
            {
                next[4 * g + d + 1] = bath * flux * beta[d] / speed;
                for (int e = d; e < 3; ++e)
                {
                    stress[6 * g + exact_group_detail::PressureComponent(d, e)] =
                        bath * ((d == e ? perpendicular : 0) +
                                (parallel - perpendicular) * beta[d] * beta[e] / speed2);
                }
            }
        }
    }
    for (int row = 0; row < 4 * n; ++row)
    {
        if (!amrex::Math::isfinite(next[row]))
        {
            return false;
        }
    }
    for (int row = 0; row < 6 * n; ++row)
    {
        if (!amrex::Math::isfinite(stress[row]))
        {
            return false;
        }
    }
    moments = next;
    pressure = stress;
    return true;
}
} // namespace warpx::radiation::centered_bath_detail
#endif
