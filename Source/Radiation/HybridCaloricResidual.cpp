/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "HybridCaloricResidual.H"

#include "Utils/WarpXConst.H"

#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>

using namespace amrex::literals;

namespace warpx::radiation::detail
{
namespace
{
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
Ulp (amrex::Real value)
{
    auto const inf = std::numeric_limits<amrex::Real>::infinity();
    return amrex::max(std::abs(std::nextafter(value, inf) - value),
                      std::abs(value - std::nextafter(value, -inf)));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
CaloricImageRoundoff (ElectronThermodynamicsExecutor const& eos, amrex::Real charge,
                      ElectronThermodynamicsExecutor::MaterialMassDensities const& composition,
                      amrex::Real temperature, amrex::Real energy, amrex::Real capacity)
{
    if (eos.isIdealGas()) {
        return 0.5_rt * (capacity * Ulp(temperature) + Ulp(energy));
    }
    if (eos.isSingularitySpiner()) {
        auto const bounds = eos.tableInverseImageEnergyBounds(charge, composition, temperature);
        if (!bounds.valid || energy < bounds.lower_energy_density ||
            energy > bounds.upper_energy_density) {
            return std::numeric_limits<amrex::Real>::quiet_NaN();
        }
        // The certified image contains the table inverse's finite bracket and
        // endpoint clipping. U_old+Q independently rounds before inversion.
        return amrex::max(std::abs(bounds.upper_energy_density - energy),
                          std::abs(energy - bounds.lower_energy_density)) +
               0.5_rt * amrex::max(Ulp(energy), amrex::max(Ulp(bounds.lower_energy_density),
                                                        Ulp(bounds.upper_energy_density)));
    }
    // The builtin latent inverse bisects [0, U/Cv_ideal] 64 times. Its
    // rounded midpoint can be either endpoint of an adjacent-T bracket:
    // this is NOT a correctly rounded analytic inverse (half a T ULP).
    // Include the finite-iteration width as well, then evaluate its energy
    // spacing through the EOS rather than linearizing a nonlinear Cv.
    auto const ideal_capacity = charge / PhysConst::q_e * PhysConst::kb / (eos.m_gamma - 1);
    auto const radius = Ulp(temperature) + std::ldexp((energy + Ulp(energy)) / ideal_capacity, -64);
    auto const lower = amrex::max(eos.minimumTemperature(), temperature - radius);
    auto const upper = amrex::min(eos.maximumTemperature(), temperature + radius);
    auto const lower_energy = eos.stateFromMaterialMassDensitiesTemperature(
        charge, composition, lower).internal_energy_density;
    auto const upper_energy = eos.stateFromMaterialMassDensitiesTemperature(
        charge, composition, upper).internal_energy_density;
    // U_old + Q also rounds independently before the inverse. Bound that
    // target addition with the largest half-ULP across this local bracket.
    return amrex::max(std::abs(upper_energy - energy), std::abs(energy - lower_energy)) +
           0.5_rt * amrex::max(Ulp(energy), amrex::max(Ulp(lower_energy), Ulp(upper_energy)));
}
} // namespace

amrex::GpuArray<amrex::Real, 2>
CoupledCaloricResidual (amrex::MultiFab const& density, amrex::MultiFab const& old,
                        amrex::MultiFab const& candidate, amrex::MultiFab const& check,
                        ElectronThermodynamicsExecutor const eos, amrex::Real minimum_density,
                        amrex::Real thermodynamic_floor,
                        CaloricMaterialFields const& material_fields)
{
    amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax> ops;
    amrex::ReduceData<amrex::Real, amrex::Real, int> data(ops);
    using Tuple = typename decltype(data)::Type;
    for (amrex::MFIter mfi(candidate); mfi.isValid(); ++mfi) {
        auto const rho = density.const_array(mfi);
        auto const t0 = old.const_array(mfi);
        auto const t1 = candidate.const_array(mfi);
        auto const t2 = check.const_array(mfi);
        ElectronThermodynamicsExecutor::MaterialChargeDensityArrays materials{};
        for (int material = 0; material < eos.m_num_materials; ++material) {
            materials[material] = material_fields[material]->const_array(mfi);
        }
        ops.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
            if (rho(i, j, k) <= PhysConst::q_e * minimum_density) {
                return {0, 0, 0};
            }
            auto const charge = amrex::max(rho(i, j, k), PhysConst::q_e * thermodynamic_floor);
            auto const composition =
                eos.materialMassDensitiesFromChargeDensityArrays(materials, i, j, k);
            auto const u0 =
                eos.stateFromMaterialMassDensitiesTemperature(charge, composition, t0(i, j, k))
                    .internal_energy_density;
            auto const u1 =
                eos.stateFromMaterialMassDensitiesTemperature(charge, composition, t1(i, j, k))
                    .internal_energy_density;
            auto const u2 =
                eos.stateFromMaterialMassDensitiesTemperature(charge, composition, t2(i, j, k))
                    .internal_energy_density;
            if (!amrex::Math::isfinite(u0) || !amrex::Math::isfinite(u1) ||
                !amrex::Math::isfinite(u2)) {
                return {0, 0, 1};
            }
            return {std::abs(u2 - u1), amrex::max(std::abs(u1 - u0), std::abs(u2 - u0)), 0};
        });
    }
    auto const values = data.value();
    amrex::GpuArray<amrex::Real, 2> result{amrex::get<0>(values), amrex::get<1>(values)};
    int invalid = amrex::get<2>(values);
    amrex::ParallelDescriptor::ReduceRealMax(result.data(), 2);
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    if (invalid != 0) {
        result[0] = std::numeric_limits<amrex::Real>::quiet_NaN();
    }
    return result;
}

LocalCaloricConvergence
CoupledLocalCaloricResidual (amrex::MultiFab const& density, amrex::MultiFab const& old,
                             amrex::MultiFab const& candidate, amrex::MultiFab const& check,
                             amrex::MultiFab const& lower_image, amrex::MultiFab const& upper_image,
                             ElectronThermodynamicsExecutor const eos, amrex::Real minimum_density,
                             amrex::Real thermodynamic_floor,
                             CaloricMaterialFields const& material_fields,
                             amrex::Real material_tolerance)
{
    bool valid = density.nComp() >= 1 && density.ixType().nodeCentered() &&
                 std::isfinite(minimum_density) && minimum_density >= 0 &&
                 std::isfinite(thermodynamic_floor) && thermodynamic_floor >= 0 &&
                 std::isfinite(material_tolerance) && material_tolerance > 0 &&
                 (eos.isIdealGas() || eos.isFixedChargeLatentEnergy() ||
                  eos.isSingularitySpiner()) &&
                 eos.m_num_materials >= 0 && eos.m_num_materials <= eos.max_materials;
    if (eos.isSingularitySpiner()) {
#ifdef WARPX_USE_SINGULARITY_EOS
        valid = valid && eos.m_num_materials > 0 && eos.m_tables_host != nullptr &&
                std::isfinite(eos.minimumTemperature()) &&
                std::isfinite(eos.maximumTemperature()) &&
                eos.maximumTemperature() > eos.minimumTemperature();
#ifdef AMREX_USE_GPU
        valid = valid && eos.m_tables_device != nullptr;
#endif
#else
        valid = false;
#endif
    }
    // A locally valid but rank-dependent material count/model would certify
    // different caloric equations on different MPI ranks. Field contents may
    // vary spatially; the executor's model and composition layout may not.
    int parameters_min[3] = {static_cast<int>(eos.m_model), eos.m_num_materials,
                             eos.m_use_resolved_material_selection ? 1 : 0};
    int parameters_max[3] = {parameters_min[0], parameters_min[1], parameters_min[2]};
    amrex::ParallelDescriptor::ReduceIntMin(parameters_min, 3);
    amrex::ParallelDescriptor::ReduceIntMax(parameters_max, 3);
    for (int n = 0; n < 3; ++n) {
        valid = valid && parameters_min[n] == parameters_max[n];
    }
    for (auto const* field : {&old, &candidate, &check, &lower_image, &upper_image}) {
        valid = valid && field->nComp() == 1 && field->boxArray() == density.boxArray() &&
                field->DistributionMap() == density.DistributionMap();
    }
    for (int n = 0; n < amrex::min(eos.m_num_materials, eos.max_materials); ++n) {
        auto const* field = material_fields[n];
        valid = valid && field && field->nComp() >= 1 && field->boxArray() == density.boxArray() &&
                field->DistributionMap() == density.DistributionMap();
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) {
        return {};
    }
    amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax>
        ops;
    amrex::ReduceData<amrex::Real, amrex::Real, int, int> data(ops);
    using Tuple = typename decltype(data)::Type;
    for (amrex::MFIter mfi(candidate); mfi.isValid(); ++mfi) {
        auto const rho = density.const_array(mfi);
        auto const t0 = old.const_array(mfi), t1 = candidate.const_array(mfi);
        auto const t2 = check.const_array(mfi), tl = lower_image.const_array(mfi);
        auto const tu = upper_image.const_array(mfi);
        ElectronThermodynamicsExecutor::MaterialChargeDensityArrays materials{};
        for (int n = 0; n < eos.m_num_materials; ++n) {
            materials[n] = material_fields[n]->const_array(mfi);
        }
        ops.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
            if (!amrex::Math::isfinite(rho(i, j, k))) {
                return {0, 0, 1, 1};
            }
            if (rho(i, j, k) <= PhysConst::q_e * minimum_density) {
                return {0, 0, 0, 0};
            }
            auto const charge = amrex::max(rho(i, j, k), PhysConst::q_e * thermodynamic_floor);
            auto const composition =
                eos.materialMassDensitiesFromChargeDensityArrays(materials, i, j, k);
            amrex::Real const temperatures[5] = {t0(i, j, k), t1(i, j, k), t2(i, j, k), tl(i, j, k),
                                                 tu(i, j, k)};
            amrex::Real energies[5]{};
            amrex::Real capacities[5]{};
            for (int n = 0; n < 5; ++n) {
                auto const state = eos.stateFromMaterialMassDensitiesTemperature(
                    charge, composition, temperatures[n]);
                if (!amrex::Math::isfinite(temperatures[n]) ||
                    temperatures[n] < eos.minimumTemperature() ||
                    temperatures[n] > eos.maximumTemperature() ||
                    !amrex::Math::isfinite(state.internal_energy_density) ||
                    !amrex::Math::isfinite(state.heat_capacity_density) ||
                    !(state.heat_capacity_density > 0)) {
                    return {0, 0, 1, 1};
                }
                energies[n] = state.internal_energy_density;
                capacities[n] = state.heat_capacity_density;
            }
            auto const heat = amrex::max(std::abs(energies[1] - energies[0]),
                                         std::abs(energies[2] - energies[0]));
            auto const mismatch = std::abs(energies[2] - energies[1]);
            // Native EOS/inverse representation plus local heating
            // arithmetic. The independent source bracket supplies the
            // stiff-emission part.
            auto const representation = CaloricImageRoundoff(
                eos, charge, composition, temperatures[1], energies[1], capacities[1]) +
                CaloricImageRoundoff(
                    eos, charge, composition, temperatures[2], energies[2], capacities[2]);
            auto const bracket = amrex::max(std::abs(energies[4] - energies[2]),
                                            std::abs(energies[2] - energies[3]));
            auto const allowance =
                representation + bracket + 8 * std::numeric_limits<amrex::Real>::epsilon() * heat;
            auto const normalized = mismatch / amrex::max(amrex::max(heat, allowance),
                                                          std::numeric_limits<amrex::Real>::min());
            bool const finite =
                amrex::Math::isfinite(allowance) && amrex::Math::isfinite(normalized);
            bool const bracket_valid = temperatures[3] <= temperatures[4] &&
                                       temperatures[2] >= temperatures[3] - Ulp(temperatures[2]) &&
                                       temperatures[2] <= temperatures[4] + Ulp(temperatures[2]);
            bool const accepted =
                finite && bracket_valid && mismatch <= material_tolerance * heat + allowance;
            return {finite ? normalized : amrex::Real(0), finite ? allowance : amrex::Real(0),
                    finite ? 0 : 1, accepted ? 0 : 1};
        });
    }
    auto const values = data.value();
    amrex::Real maxima[2] = {amrex::get<0>(values), amrex::get<1>(values)};
    int flags[2] = {amrex::get<2>(values), amrex::get<3>(values)};
    amrex::ParallelDescriptor::ReduceRealMax(maxima, 2);
    amrex::ParallelDescriptor::ReduceIntMax(flags, 2);
    return {flags[0] == 0, flags[0] == 0 && flags[1] == 0, maxima[0], maxima[1]};
}
} // namespace warpx::radiation::detail
