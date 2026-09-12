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

namespace warpx::radiation
{
    using namespace amrex::literals;

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
            amrex::Real const upper = group == m_num_groups - 1
                ? 1.0e30_rt : m_boundaries[group] / thermal_energy;
            return amrex::max(
                0.0_rt,
                amrex::min(1.0_rt, planckCdf(upper) - planckCdf(lower)));
        }

        amrex::Real const* m_boundaries = nullptr;
        amrex::Real const* m_representative_energies = nullptr;
        int m_num_groups = 1;

    private:
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        static amrex::Real planckCdf (amrex::Real const x) noexcept
        {
            constexpr amrex::Real integral_to_infinity =
                6.4939394022668291491_rt;
            if (x <= 0.0_rt) { return 0.0_rt; }
            if (x >= 80.0_rt) { return 1.0_rt; }

            if (x < 0.5_rt) {
                amrex::Real const x2 = x * x;
                amrex::Real const x3 = x2 * x;
                amrex::Real const x4 = x2 * x2;
                amrex::Real const integral =
                    x3 / 3.0_rt - x4 / 8.0_rt
                    + x4 * x / 60.0_rt
                    - x4 * x3 / 5040.0_rt
                    + x4 * x4 * x / 272160.0_rt
                    - x4 * x4 * x3 / 13305600.0_rt
                    + x4 * x4 * x4 * x / 622702080.0_rt;
                return integral / integral_to_infinity;
            }

            amrex::Real const x2 = x * x;
            amrex::Real const x3 = x2 * x;
            amrex::Real tail = 0.0_rt;
            for (int n = 1; n <= 64; ++n) {
                amrex::Real const inverse_n = 1.0_rt / n;
                amrex::Real const inverse_n2 = inverse_n * inverse_n;
                tail += std::exp(-n * x) * (
                    x3 * inverse_n
                    + 3.0_rt * x2 * inverse_n2
                    + 6.0_rt * x * inverse_n2 * inverse_n
                    + 6.0_rt * inverse_n2 * inverse_n2);
            }
            return amrex::max(
                0.0_rt, amrex::min(1.0_rt, 1.0_rt - tail / integral_to_infinity));
        }
    };
}

#endif // WARPX_RADIATION_ENERGYGROUPS_H_
