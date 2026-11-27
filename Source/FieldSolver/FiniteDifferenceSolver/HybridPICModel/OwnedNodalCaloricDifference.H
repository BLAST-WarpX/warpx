/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_HYBRIDPIC_OWNEDNODALCALORICDIFFERENCE_H_
#define WARPX_HYBRIDPIC_OWNEDNODALCALORICDIFFERENCE_H_

#include "ElectronThermodynamics.H"

#include "Utils/WarpXConst.H"

#include <AMReX_Algorithm.H>

#include <cmath>

namespace warpx::hybrid
{
/** Stable fixed-charge caloric difference for the opt-in composite old-T
 * contract. In the latent model f(T)=x/(1+x), x=(T/T*)^s. For nearby T,
 * form x_new/x_old-1 with expm1(s*log1p(dT/T_old)); its logistic difference
 * avoids subtracting two rounded, nearly equal latent-energy backgrounds.
 * The high-T branch uses inverse x to avoid overflow. Wide temperature
 * changes retain the established EOS difference, where cancellation is small.
 */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
CenteredOwnedNodalCaloricDifference (
    ElectronThermodynamicsExecutor const& eos, amrex::Real charge,
    amrex::Real old_temperature, amrex::Real new_temperature) noexcept
{
    auto const delta = new_temperature - old_temperature;
    if (delta == 0)
    {
        return 0;
    }
    if (!eos.isFixedChargeLatentEnergy() || !(old_temperature > 0) ||
        !(new_temperature > 0) ||
        std::abs(delta) > amrex::Real(0.25) *
                              amrex::max(old_temperature, new_temperature))
    {
        return eos.internalEnergyDensityDifference(
            charge, {}, old_temperature, new_temperature);
    }

    auto const electron_density = charge / PhysConst::q_e;
    auto change = electron_density * PhysConst::kb /
                  (eos.m_gamma - amrex::Real(1)) * delta;
    auto const log_ratio = std::log1p(delta / old_temperature);
    for (int n = 0; n < eos.m_num_latent_transitions; ++n)
    {
        auto const transition = eos.m_latent_transition_temperature[n];
        auto const sharpness = eos.m_latent_transition_sharpness[n];
        amrex::Real fraction_change;
        if (old_temperature < transition)
        {
            auto const x = std::pow(old_temperature / transition, sharpness);
            auto const ratio_change = std::expm1(sharpness * log_ratio);
            fraction_change = x * ratio_change /
                              ((1 + x) * (1 + x * (1 + ratio_change)));
        }
        else
        {
            auto const y = std::pow(transition / old_temperature, sharpness);
            auto const ratio_change = std::expm1(-sharpness * log_ratio);
            fraction_change = -y * ratio_change /
                              ((1 + y) * (1 + y * (1 + ratio_change)));
        }
        change += electron_density * eos.m_latent_energy_per_electron[n] * fraction_change;
    }
    return change;
}
} // namespace warpx::hybrid

#endif
