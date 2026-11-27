/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_ENERGYGROUPS_H_
#define WARPX_RADIATION_ENERGYGROUPS_H_

#include <AMReX_Algorithm.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

namespace warpx::radiation
{
    using namespace amrex::literals;

    namespace spectral_detail
    {
        constexpr amrex::Real planck_integral = 6.4939394022668291491_rt;

        // Stable tails avoid subtracting two rounded CDFs near one. The low-x series
        // is the integrated Bernoulli expansion; above x=0.5 use the exponential sum.
        AMREX_GPU_HOST_DEVICE inline amrex::Real LowCdf (amrex::Real x)
        {
            auto const x2 = x * x, x3 = x2 * x;
            return (x3 / 3 - x3 * x / 8 + x3 * x2 / 60 - x3 * x2 * x2 / 5040 + x3 * x2 * x2 * x2 / 272160 -
                    x3 * x2 * x2 * x2 * x2 / 13305600 + x3 * x2 * x2 * x2 * x2 * x2 / 622702080) /
                   planck_integral;
        }
        AMREX_GPU_HOST_DEVICE inline amrex::Real Tail (amrex::Real x)
        {
            if (x < 0.5_rt)
            {
                return 1 - LowCdf(x);
            }
            if (x > 800)
            {
                return 0;
            }
            amrex::Real sum = 0;
            for (int n = 1; n <= 128; ++n)
            {
                auto const inverse = 1.0_rt / n;
                // Logarithmic evaluation avoids x^3 overflow in very cold tails.
                auto const term = std::exp(-n * x + 3 * std::log(x)) * inverse *
                                  (1 + 3 * inverse / x + 6 * inverse * inverse / (x * x) +
                                   6 * inverse * inverse * inverse / (x * x * x));
                sum += term;
                if (n > 1 && term <= std::numeric_limits<amrex::Real>::epsilon() * sum)
                {
                    break;
                }
            }
            return sum / planck_integral;
        }
    }

    /** Device view of the photon-energy group structure.
     *
     * N finite boundaries define N+1 groups and leave the low- and high-energy
     * spectral tails implicit. This guarantees that the group-integrated
     * Planck fractions sum to one at every positive material temperature.
     */
    struct EnergyGroupsExecutor
    {
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        int index (amrex::Real const photon_energy) const noexcept
        {
            int group = 0;
            while (group < m_num_groups - 1
                   && photon_energy >= m_boundaries[group])
            {
                ++group;
            }
            return group;
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real representativeEnergy (int const group) const noexcept
        {
            return m_representative_energies[group];
        }

        /** Fraction of blackbody energy in one group.
         *
         * thermal_energy is k_B*T in joules. The semi-infinite Planck
         * integral is evaluated through a rapidly convergent exponential
         * tail series, with a small-x power series to retain accuracy.
         */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real planckFraction (
            int const group, amrex::Real const thermal_energy) const noexcept
        {
            if (m_num_groups == 1) { return 1.0_rt; }
            if (thermal_energy <= 0.0_rt) {
                return group == 0 ? 1.0_rt : 0.0_rt;
            }
            amrex::Real const lower = group == 0
                ? 0.0_rt : m_boundaries[group - 1] / thermal_energy;
            using spectral_detail::LowCdf;
            using spectral_detail::Tail;
            if (group == m_num_groups - 1) {
                return Tail(lower);
            }
            amrex::Real const upper = m_boundaries[group] / thermal_energy;
            // Use cumulative energy in the low-energy range and the remaining
            // tail at high energy. A CDF rounded to one loses a finite band.
            amrex::Real const fraction = upper < 0.5_rt ? LowCdf(upper) - LowCdf(lower)
                : lower < 0.5_rt ? (1 - Tail(upper)) - LowCdf(lower)
                                 : Tail(lower) - Tail(upper);
            return amrex::max(0.0_rt, amrex::min(1.0_rt, fraction));
        }

        amrex::Real const* m_boundaries = nullptr;
        amrex::Real const* m_representative_energies = nullptr;
        int m_num_groups = 1;

    };
}

#endif // WARPX_RADIATION_ENERGYGROUPS_H_
