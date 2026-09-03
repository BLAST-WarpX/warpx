/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_ELECTRONTHERMODYNAMICS_H_
#define WARPX_ELECTRONTHERMODYNAMICS_H_

#include "Utils/WarpXConst.H"
#include "Utils/MaterialRegistry.H"
#include "Utils/TextMsg.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/SingularityEOS.H"

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Gpu.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

enum class ElectronThermodynamicsModel : int
{
    IdealGas,
    FixedChargeLatentEnergy,
    SingularitySpiner
};

/**
 * \brief Device-callable electron thermodynamics used by the hybrid model.
 *
 * The charge-density argument is the positive electron charge-density
 * magnitude in C/m^3. Temperatures are in K, pressures and thermal-energy
 * densities in Pa and J/m^3, and heat-capacity densities in J/(m^3 K).
 * Density floors remain the caller's responsibility.
 *
 * Keeping the caloric and pressure closures behind one small executor lets
 * source operators share a consistent material model when non-ideal and
 * tabulated backends are added. The ideal-gas expressions deliberately retain
 * the operation order of the legacy hybrid pressure reconstruction.
 */
struct ElectronThermodynamicState
{
    amrex::Real pressure = amrex::Real(0.0);
    amrex::Real internal_energy_density = amrex::Real(0.0);
    amrex::Real heat_capacity_density = amrex::Real(0.0);
    amrex::Real ion_number_density = amrex::Real(0.0);
    // Number-density-weighted nuclear charge.  This is composition metadata,
    // not the plasma mean ionization state Zbar(rho,T).
    amrex::Real mean_atomic_number = amrex::Real(0.0);
    // Charge-consistent effective ionization ne/ni.  This reports the charge
    // state represented by the PIC species; it must not be confused with an
    // equilibrium-ionization prediction supplied by an EOS package.
    amrex::Real mean_ionization = amrex::Real(0.0);
};

/** Result of applying a prescribed volumetric electron-energy increment. */
struct ElectronEnergySourceUpdate
{
    amrex::Real temperature = amrex::Real(0.0);
    // Realized U_new-U_old [J/m^3].  Conservative conjugate operators must
    // consume this value, rather than independently reconstructing the source.
    amrex::Real energy_change_density = amrex::Real(0.0);
    // Realized minus requested energy change [J/m^3].
    amrex::Real energy_residual_density = amrex::Real(0.0);
    bool valid = true;
};

struct ElectronThermodynamicsExecutor
{
    static constexpr int max_latent_transitions = 8;
    static constexpr int max_materials = 8;
    static_assert(
        max_materials == warpx::materials::MaterialRegistry::max_materials);
    using MaterialMassDensities =
        amrex::GpuArray<amrex::Real, max_materials>;
    using MaterialChargeDensityArrays =
        amrex::GpuArray<amrex::Array4<amrex::Real const>, max_materials>;

    ElectronThermodynamicsModel m_model =
        ElectronThermodynamicsModel::IdealGas;
    amrex::Real m_gamma = amrex::Real(5.0) / amrex::Real(3.0);
    int m_num_latent_transitions = 0;
    amrex::GpuArray<amrex::Real, max_latent_transitions>
        m_latent_transition_temperature{};
    amrex::GpuArray<amrex::Real, max_latent_transitions>
        m_latent_energy_per_electron{};
    amrex::GpuArray<amrex::Real, max_latent_transitions>
        m_latent_transition_sharpness{};
    int m_num_materials = 0;
    // Conversion for deposited unit-ion charge density q_e n_i [C/m^3].
    amrex::GpuArray<amrex::Real, max_materials> m_mass_per_charge{};
    // Conversion for physical charge density q_i n_i [C/m^3].
    amrex::GpuArray<amrex::Real, max_materials>
        m_mass_per_physical_charge{};
    bool m_use_resolved_material_selection = false;
    warpx::materials::MaterialRegistry::ResolvedCellSelector
        m_resolved_material_selector{};
    amrex::GpuArray<amrex::Real, max_materials> m_atomic_mass{};
    amrex::GpuArray<amrex::Real, max_materials> m_atomic_number{};

#ifdef WARPX_USE_SINGULARITY_EOS
    // Non-owning view. ElectronThermodynamics owns and finalizes this
    // contiguous host table storage after all executors have become unused.
    singularity::SpinerEOSDependsRhoT const* m_tables = nullptr;
    amrex::GpuArray<amrex::Real, max_materials> m_minimum_mass_density{};
    amrex::GpuArray<amrex::Real, max_materials> m_maximum_mass_density{};
    amrex::Real m_table_temperature_floor = amrex::Real(0.0);
    amrex::Real m_table_temperature_ceiling =
        std::numeric_limits<amrex::Real>::max();
#endif

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool isIdealGas () const noexcept
    {
        return m_model == ElectronThermodynamicsModel::IdealGas;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool isFixedChargeLatentEnergy () const noexcept
    {
        return m_model == ElectronThermodynamicsModel::FixedChargeLatentEnergy;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool isSingularitySpiner () const noexcept
    {
        return m_model == ElectronThermodynamicsModel::SingularitySpiner;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real minimumTemperature () const noexcept
    {
#ifdef WARPX_USE_SINGULARITY_EOS
        return isSingularitySpiner()
            ? m_table_temperature_floor : amrex::Real(0.0);
#else
        return amrex::Real(0.0);
#endif
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real maximumTemperature () const noexcept
    {
#ifdef WARPX_USE_SINGULARITY_EOS
        return isSingularitySpiner()
            ? m_table_temperature_ceiling
            : std::numeric_limits<amrex::Real>::max();
#else
        return std::numeric_limits<amrex::Real>::max();
#endif
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real materialMassDensityFromUnitIonChargeDensity (
        int const material,
        amrex::Real const material_charge_density) const noexcept
    {
        return material_charge_density * m_mass_per_charge[material];
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real materialMassDensityFromPhysicalChargeDensity (
        int const material,
        amrex::Real const physical_charge_density) const noexcept
    {
        return physical_charge_density
            * m_mass_per_physical_charge[material];
    }

    /** Convert per-material deposited unit-ion arrays, each containing q_e n_i. */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    MaterialMassDensities materialMassDensitiesFromChargeDensityArrays (
        MaterialChargeDensityArrays const& material_charge_density,
        int const i,
        int const j,
        int const k) const noexcept
    {
        MaterialMassDensities result{};
        for (int material = 0; material < m_num_materials; ++material) {
            result[material] = materialMassDensityFromUnitIonChargeDensity(
                material, material_charge_density[material](i, j, k));
        }
        return result;
    }

    /** Return the selected registered material or a negative ResolvedCell
     * sentinel.
     *
     * A non-negative result is meaningful only when the named-material
     * resolved-single-material policy is enabled. Legacy table configurations
     * deliberately retain their additive partial-density behavior.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int resolvedMaterial (
        MaterialMassDensities const& material_mass_density) const noexcept
    {
        return m_use_resolved_material_selection
            ? m_resolved_material_selector(material_mass_density)
            : static_cast<int>(
                warpx::materials::MaterialRegistry::ResolvedCell::Invalid);
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real latentFraction (
        amrex::Real const temperature,
        int const transition) const noexcept
    {
        if (!(temperature > amrex::Real(0.0))) { return amrex::Real(0.0); }
        amrex::Real const transition_temperature =
            m_latent_transition_temperature[transition];
        amrex::Real const sharpness =
            m_latent_transition_sharpness[transition];
        if (temperature < transition_temperature) {
            amrex::Real const ratio = std::pow(
                temperature / transition_temperature, sharpness);
            return ratio / (amrex::Real(1.0) + ratio);
        }
        amrex::Real const inverse_ratio = std::pow(
            transition_temperature / temperature, sharpness);
        return amrex::Real(1.0) / (amrex::Real(1.0) + inverse_ratio);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real pressureFromChargeDensityTemperature (
        amrex::Real const charge_density,
        amrex::Real const temperature) const noexcept
    {
        if (isSingularitySpiner()) {
            return std::numeric_limits<amrex::Real>::quiet_NaN();
        }
        return charge_density / PhysConst::q_e * PhysConst::kb * temperature;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ElectronThermodynamicState
    stateFromMaterialMassDensitiesTemperature (
        amrex::Real const charge_density,
        MaterialMassDensities const& material_mass_density,
        amrex::Real const temperature) const noexcept
    {
#ifdef WARPX_USE_SINGULARITY_EOS
        if (isSingularitySpiner()) {
            if (m_tables == nullptr
                || !amrex::Math::isfinite(charge_density)
                || charge_density < amrex::Real(0.0))
            {
                amrex::Real const invalid =
                    std::numeric_limits<amrex::Real>::quiet_NaN();
                return {invalid, invalid, invalid, invalid, invalid, invalid};
            }

            int material_begin = 0;
            int material_end = m_num_materials;
            if (m_use_resolved_material_selection) {
                int const resolved_material =
                    resolvedMaterial(material_mass_density);
                if (resolved_material == static_cast<int>(
                        warpx::materials::MaterialRegistry::ResolvedCell::Vacuum))
                {
                    return {};
                }
                if (resolved_material < 0
                    || resolved_material >= m_num_materials)
                {
                    amrex::Real const invalid =
                        std::numeric_limits<amrex::Real>::quiet_NaN();
                    return {invalid, invalid, invalid, invalid, invalid, invalid};
                }
                material_begin = resolved_material;
                material_end = resolved_material + 1;
            }

            ElectronThermodynamicState state;
            amrex::Real atomic_number_weight = amrex::Real(0.0);
            for (int material = material_begin; material < material_end; ++material) {
                amrex::Real const mass_density =
                    material_mass_density[material];
                if (!amrex::Math::isfinite(mass_density)
                    || mass_density < amrex::Real(0.0)
                    || mass_density > m_maximum_mass_density[material])
                {
                    amrex::Real const invalid =
                        std::numeric_limits<amrex::Real>::quiet_NaN();
                    return {invalid, invalid, invalid, invalid, invalid, invalid};
                }
                if (mass_density == amrex::Real(0.0)) { continue; }
                if (temperature < m_table_temperature_floor
                    || temperature > m_table_temperature_ceiling
                    || !amrex::Math::isfinite(temperature))
                {
                    amrex::Real const invalid =
                        std::numeric_limits<amrex::Real>::quiet_NaN();
                    return {invalid, invalid, invalid, invalid, invalid, invalid};
                }

                // Singularity-EOS uses CGS. Convert kg/m^3 to g/cm^3,
                // dyne/cm^2 to Pa, and erg/g(/K) to J/kg(/K).
                // PIC shape functions generate continuous partial-density
                // tails through material/vacuum and mixed-material interfaces.
                // Below rho_min, evaluate specific properties at rho_min but
                // retain the deposited mass. Pressure is scaled by rho/rho_min
                // so P, U, and C_V all approach zero continuously without
                // inventing mass or introducing a threshold discontinuity.
                amrex::Real const evaluation_mass_density = amrex::max(
                    mass_density, m_minimum_mass_density[material]);
                amrex::Real const density_scale =
                    mass_density / evaluation_mass_density;
                amrex::Real const density_cgs =
                    amrex::Real(1.0e-3) * evaluation_mass_density;
                auto const& table = m_tables[material];
                amrex::Real const specific_energy =
                    table.InternalEnergyFromDensityTemperature(
                        density_cgs, temperature);
                state.pressure += amrex::Real(1.0e-1) * density_scale
                    * table.PressureFromDensityTemperature(
                        density_cgs, temperature);
                state.internal_energy_density += mass_density
                    * amrex::Real(1.0e-4)
                    * specific_energy;
                state.heat_capacity_density += mass_density
                    * amrex::Real(1.0e-4)
                    * table.SpecificHeatFromDensityTemperature(
                        density_cgs, temperature);
                amrex::Real const ion_density = mass_density
                    / (m_atomic_mass[material] * PhysConst::m_u);
                amrex::Real const table_mean_atomic_number =
                    m_atomic_number[material];
                if (!(ion_density >= amrex::Real(0.0))
                    || !amrex::Math::isfinite(ion_density)
                    || !(table_mean_atomic_number >= amrex::Real(0.0))
                    || !amrex::Math::isfinite(table_mean_atomic_number))
                {
                    amrex::Real const invalid =
                        std::numeric_limits<amrex::Real>::quiet_NaN();
                    return {invalid, invalid, invalid, invalid, invalid, invalid};
                }
                state.ion_number_density += ion_density;
                atomic_number_weight +=
                    ion_density * table_mean_atomic_number;
            }
            if (state.ion_number_density > amrex::Real(0.0)) {
                state.mean_atomic_number =
                    atomic_number_weight / state.ion_number_density;
                state.mean_ionization = charge_density / PhysConst::q_e
                    / state.ion_number_density;
            }
            return state;
        }
#else
        amrex::ignore_unused(material_mass_density);
#endif

        amrex::Real const pressure =
            charge_density / PhysConst::q_e * PhysConst::kb * temperature;
        amrex::Real const gamma_minus_one = m_gamma - amrex::Real(1.0);
        amrex::Real const electron_density = charge_density / PhysConst::q_e;
        amrex::Real heat_capacity_density =
            electron_density * PhysConst::kb / gamma_minus_one;
        amrex::Real internal_energy_density =
            pressure / gamma_minus_one;

        if (isFixedChargeLatentEnergy()
            && temperature > amrex::Real(0.0))
        {
            for (int transition = 0;
                 transition < m_num_latent_transitions; ++transition)
            {
                amrex::Real const fraction =
                    latentFraction(temperature, transition);
                amrex::Real const latent_energy_density = electron_density
                    * m_latent_energy_per_electron[transition];
                internal_energy_density += latent_energy_density * fraction;
                heat_capacity_density += latent_energy_density * (
                    m_latent_transition_sharpness[transition]
                    * fraction * (amrex::Real(1.0) - fraction) / temperature);
            }
        }

        if (m_num_materials > 0) {
            auto ion_number_density = amrex::Real(0.0);
            auto atomic_number_weight = amrex::Real(0.0);
            for (int material = 0; material < m_num_materials; ++material) {
                amrex::Real const mass_density =
                    material_mass_density[material];
                if (!(mass_density >= amrex::Real(0.0))
                    || !amrex::Math::isfinite(mass_density))
                {
                    amrex::Real const invalid =
                        std::numeric_limits<amrex::Real>::quiet_NaN();
                    return {
                        invalid, invalid, invalid, invalid, invalid, invalid};
                }
                amrex::Real const ion_density = mass_density
                    / (m_atomic_mass[material] * PhysConst::m_u);
                ion_number_density += ion_density;
                atomic_number_weight +=
                    ion_density * m_atomic_number[material];
            }
            if (ion_number_density > amrex::Real(0.0)) {
                return {
                    pressure, internal_energy_density, heat_capacity_density,
                    ion_number_density,
                    atomic_number_weight / ion_number_density,
                    electron_density / ion_number_density};
            }
            return {
                pressure, internal_energy_density, heat_capacity_density,
                amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0)};
        }

        // Charge-only analytic closures do not carry an independent ion-mass
        // state. Their historical contract is one charge unit per effective
        // ion; material-aware states above provide physical n_i and nuclear Z.
        return {
            pressure, internal_energy_density, heat_capacity_density,
            electron_density,
            electron_density > amrex::Real(0.0)
                ? amrex::Real(1.0) : amrex::Real(0.0),
            electron_density > amrex::Real(0.0)
                ? amrex::Real(1.0) : amrex::Real(0.0)};
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ElectronThermodynamicState
    stateFromChargeDensityTemperature (
        amrex::Real const charge_density,
        amrex::Real const temperature) const noexcept
    {
        if (isSingularitySpiner()) {
            amrex::Real const invalid =
                std::numeric_limits<amrex::Real>::quiet_NaN();
            return {invalid, invalid, invalid, invalid, invalid, invalid};
        }
        return stateFromMaterialMassDensitiesTemperature(
            charge_density, MaterialMassDensities{}, temperature);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real thermalEnergyDensityFromChargeDensityTemperature (
        amrex::Real const charge_density,
        amrex::Real const temperature) const noexcept
    {
        return stateFromChargeDensityTemperature(charge_density, temperature)
            .internal_energy_density;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real temperatureFromChargeDensityThermalEnergyDensity (
        amrex::Real const charge_density,
        amrex::Real const thermal_energy_density) const noexcept
    {
        if (isSingularitySpiner()) {
            return std::numeric_limits<amrex::Real>::quiet_NaN();
        }
        return temperatureFromMaterialMassDensitiesThermalEnergyDensity(
            charge_density, MaterialMassDensities{}, thermal_energy_density);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real temperatureFromMaterialMassDensitiesThermalEnergyDensity (
        amrex::Real const charge_density,
        MaterialMassDensities const& material_mass_density,
        amrex::Real const thermal_energy_density) const noexcept
    {
#ifdef WARPX_USE_SINGULARITY_EOS
        if (isSingularitySpiner()) {
            amrex::Real temperature_lo = m_table_temperature_floor;
            amrex::Real temperature_hi = m_table_temperature_ceiling;
            amrex::Real const minimum_energy =
                stateFromMaterialMassDensitiesTemperature(
                    charge_density, material_mass_density, temperature_lo)
                    .internal_energy_density;
            amrex::Real const maximum_energy =
                stateFromMaterialMassDensitiesTemperature(
                    charge_density, material_mass_density, temperature_hi)
                    .internal_energy_density;
            // Endpoint checks must use endpoint-local scales. A tolerance
            // based on max(|U_min|, |U_max|) can erase a perfectly resolvable
            // source increment near U_min when a wide table has U_max much
            // larger than its cold energy.
            amrex::Real const minimum_tolerance = amrex::Real(256.0)
                * std::numeric_limits<amrex::Real>::epsilon()
                * amrex::max(
                    amrex::max(
                        std::abs(thermal_energy_density),
                        std::abs(minimum_energy)),
                    std::numeric_limits<amrex::Real>::min());
            amrex::Real const maximum_tolerance = amrex::Real(256.0)
                * std::numeric_limits<amrex::Real>::epsilon()
                * amrex::max(
                    amrex::max(
                        std::abs(thermal_energy_density),
                        std::abs(maximum_energy)),
                    std::numeric_limits<amrex::Real>::min());
            if (!amrex::Math::isfinite(minimum_energy)
                || !amrex::Math::isfinite(maximum_energy)
                || !amrex::Math::isfinite(thermal_energy_density)
                || thermal_energy_density
                    < minimum_energy - minimum_tolerance
                || thermal_energy_density
                    > maximum_energy + maximum_tolerance)
            {
                return std::numeric_limits<amrex::Real>::quiet_NaN();
            }
            if (thermal_energy_density
                <= minimum_energy + minimum_tolerance) {
                return temperature_lo;
            }
            if (thermal_energy_density
                >= maximum_energy - maximum_tolerance) {
                return temperature_hi;
            }
            for (int iteration = 0; iteration < 64; ++iteration) {
                amrex::Real const trial_temperature =
                    amrex::Real(0.5) * (temperature_lo + temperature_hi);
                amrex::Real const trial_energy =
                    stateFromMaterialMassDensitiesTemperature(
                        charge_density, material_mass_density,
                        trial_temperature)
                        .internal_energy_density;
                if (trial_energy < thermal_energy_density) {
                    temperature_lo = trial_temperature;
                } else {
                    temperature_hi = trial_temperature;
                }
            }
            return amrex::Real(0.5) * (temperature_lo + temperature_hi);
        }
#else
        amrex::ignore_unused(material_mass_density);
#endif

        amrex::Real const heat_capacity_density =
            charge_density / PhysConst::q_e * PhysConst::kb
            / (m_gamma - amrex::Real(1.0));
        if (!(heat_capacity_density > amrex::Real(0.0))
            || !(thermal_energy_density > amrex::Real(0.0)))
        {
            return amrex::Real(0.0);
        }
        if (isIdealGas()) {
            return thermal_energy_density / heat_capacity_density;
        }

        // Latent fractions are monotone, so the corresponding ideal-gas
        // temperature is a strict upper bracket for the nonlinear inverse.
        auto temperature_lo = amrex::Real(0.0);
        amrex::Real temperature_hi =
            thermal_energy_density / heat_capacity_density;
        for (int iteration = 0; iteration < 64; ++iteration) {
            amrex::Real const trial_temperature =
                amrex::Real(0.5) * (temperature_lo + temperature_hi);
            amrex::Real const trial_energy =
                stateFromMaterialMassDensitiesTemperature(
                    charge_density, material_mass_density, trial_temperature)
                    .internal_energy_density;
            if (trial_energy < thermal_energy_density) {
                temperature_lo = trial_temperature;
            } else {
                temperature_hi = trial_temperature;
            }
        }
        return amrex::Real(0.5) * (temperature_lo + temperature_hi);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real heatCapacityDensityFromChargeDensityTemperature (
        amrex::Real const charge_density,
        amrex::Real const temperature) const noexcept
    {
        return stateFromChargeDensityTemperature(charge_density, temperature)
            .heat_capacity_density;
    }

    /**
     * Apply a prescribed electron internal-energy-density increment through
     * the configured caloric inverse.  This is the common conservative source
     * primitive for radiation, Joule and Q_ei operators.  It deliberately
     * does not alter charge state or add ionization energy; callers that evolve
     * charge must maintain their separate binding-energy ledger.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ElectronEnergySourceUpdate applyEnergyDensityIncrement (
        amrex::Real const charge_density,
        MaterialMassDensities const& material_mass_density,
        amrex::Real const old_temperature,
        amrex::Real const requested_energy_change_density) const noexcept
    {
        amrex::Real const invalid =
            std::numeric_limits<amrex::Real>::quiet_NaN();
        ElectronThermodynamicState const old_state =
            stateFromMaterialMassDensitiesTemperature(
                charge_density, material_mass_density, old_temperature);
        if (!amrex::Math::isfinite(old_state.internal_energy_density)
            || !(old_state.heat_capacity_density > amrex::Real(0.0))
            || !amrex::Math::isfinite(old_state.heat_capacity_density)
            || !amrex::Math::isfinite(requested_energy_change_density))
        {
            return {invalid, invalid, invalid, false};
        }

        amrex::Real const requested_energy_density =
            old_state.internal_energy_density
            + requested_energy_change_density;
        if (!amrex::Math::isfinite(requested_energy_density)) {
            return {invalid, invalid, invalid, false};
        }
        amrex::Real const new_temperature =
            temperatureFromMaterialMassDensitiesThermalEnergyDensity(
                charge_density, material_mass_density,
                requested_energy_density);
        ElectronThermodynamicState const new_state =
            stateFromMaterialMassDensitiesTemperature(
                charge_density, material_mass_density, new_temperature);
        amrex::Real const realized_energy_change_density =
            new_state.internal_energy_density
            - old_state.internal_energy_density;
        amrex::Real const energy_residual_density =
            realized_energy_change_density
            - requested_energy_change_density;
        amrex::Real const energy_scale = amrex::max(
            amrex::max(
                std::abs(old_state.internal_energy_density),
                std::abs(requested_energy_density)),
            amrex::max(
                std::abs(new_state.internal_energy_density),
                amrex::max(
                    std::abs(requested_energy_change_density),
                    std::numeric_limits<amrex::Real>::min())));
        // The caloric inverse uses a fixed 64-iteration bracket.  This bound
        // admits unresolved increments below U roundoff while rejecting a
        // material-table inversion that does not reproduce its requested
        // energy state to arithmetic precision.
        amrex::Real const energy_tolerance = amrex::Real(2048.0)
            * std::numeric_limits<amrex::Real>::epsilon() * energy_scale;
        if (!amrex::Math::isfinite(new_temperature)
            || new_temperature < minimumTemperature()
            || new_temperature > maximumTemperature()
            || !amrex::Math::isfinite(new_state.internal_energy_density)
            || !(new_state.heat_capacity_density > amrex::Real(0.0))
            || !amrex::Math::isfinite(new_state.heat_capacity_density)
            || !amrex::Math::isfinite(realized_energy_change_density)
            || !amrex::Math::isfinite(energy_residual_density)
            || std::abs(energy_residual_density) > energy_tolerance)
        {
            return {invalid, invalid, invalid, false};
        }
        return {
            new_temperature, realized_energy_change_density,
            energy_residual_density, true};
    }
};

/**
 * \brief Host-side configuration and lifetime owner for electron
 * thermodynamics.
 *
 * The legacy ideal gas and a fixed-charge analytic latent-energy reservoir are
 * available. An explicit model choice creates a stable attachment point for
 * package-provided tabulated data while making unavailable models fail during
 * input validation instead of silently falling back to an ideal gas.
 */
class ElectronThermodynamics
{
public:
    ElectronThermodynamics () = default;
    ElectronThermodynamics (ElectronThermodynamics const&) = delete;
    ElectronThermodynamics& operator= (ElectronThermodynamics const&) = delete;
    ElectronThermodynamics (ElectronThermodynamics&&) = delete;
    ElectronThermodynamics& operator= (ElectronThermodynamics&&) = delete;

    ~ElectronThermodynamics ()
    {
        Finalize();
    }

    void ReadParameters (
        amrex::ParmParse const& pp_hybrid,
        amrex::Real const gamma,
        warpx::materials::MaterialRegistry const* const material_registry = nullptr)
    {
#ifndef WARPX_USE_SINGULARITY_EOS
        amrex::ignore_unused(material_registry);
#endif
        Finalize();
        std::string model = "ideal_gas";
        pp_hybrid.query("electron_thermodynamics", model);
        m_executor.m_gamma = gamma;
        m_executor.m_num_latent_transitions = 0;
        if (model == "ideal_gas") {
            if (pp_hybrid.contains(
                    "electron_latent_transition_temperature_eV")
                || pp_hybrid.contains("electron_latent_energy_eV")
                || pp_hybrid.contains("electron_latent_sharpness"))
            {
                amrex::Abort(
                    "hybrid_pic_model.electron_latent_* inputs require "
                    "electron_thermodynamics=fixed_charge_latent_energy.");
            }
            m_executor.m_model = ElectronThermodynamicsModel::IdealGas;
            ReadAnalyticComposition(pp_hybrid);
            return;
        }
        if (model == "singularity_spiner") {
#ifndef WARPX_USE_SINGULARITY_EOS
            amrex::Abort(
                "hybrid_pic_model.electron_thermodynamics="
                "singularity_spiner requires a WarpX build configured with "
                "-DWarpX_SINGULARITY_EOS=ON.");
#else
#   ifdef AMREX_USE_FLOAT
            amrex::Abort(
                "hybrid_pic_model.electron_thermodynamics=singularity_spiner "
                "requires WarpX_PRECISION=DOUBLE; SINGLE field precision is "
                "not qualified for tabulated electron-energy source and ledger "
                "updates.");
#   endif
            m_executor.m_model =
                ElectronThermodynamicsModel::SingularitySpiner;
            bool const use_material_registry =
                material_registry != nullptr && material_registry->enabled();
            if (use_material_registry) {
                m_material_species.reserve(
                    static_cast<std::size_t>(material_registry->size()));
                for (int material = 0;
                     material < material_registry->size(); ++material)
                {
                    auto const& definition =
                        material_registry->material(material);
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        definition.species.size() == 1U
                            && definition.electron_eos.has_value(),
                        "Every registered Singularity-EOS material must have "
                        "exactly one carrier species and one electron_eos "
                        "table handle. Invalid material '"
                            + definition.name + "'.");
                    m_material_species.push_back(definition.species.front());
                }
                if (pp_hybrid.contains("electron_eos_species")) {
                    std::vector<std::string> configured_species;
                    pp_hybrid.getarr(
                        "electron_eos_species", configured_species);
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        configured_species == m_material_species,
                        "When materials.names is enabled, "
                        "hybrid_pic_model.electron_eos_species must exactly "
                        "match the registry's material-name-ordered carrier "
                        "species list, or be omitted.");
                }
                m_executor.m_use_resolved_material_selection = true;
                m_executor.m_resolved_material_selector =
                    material_registry->selector();
            } else {
                pp_hybrid.getarr(
                    "electron_eos_species", m_material_species);
            }
            if (m_material_species.empty()
                || m_material_species.size()
                    > ElectronThermodynamicsExecutor::max_materials)
            {
                amrex::Abort(
                    "hybrid_pic_model.electron_eos_species must contain "
                    "between one and "
                    + std::to_string(
                        ElectronThermodynamicsExecutor::max_materials)
                    + " material species.");
            }

            m_tables.reserve(m_material_species.size());
            amrex::Real common_temperature_floor = amrex::Real(0.0);
            amrex::Real common_temperature_ceiling =
                std::numeric_limits<amrex::Real>::max();
            for (std::size_t material = 0;
                 material < m_material_species.size(); ++material)
            {
                std::string const prefix =
                    "electron_eos_" + m_material_species[material];
                std::string table_file;
                int material_id = -1;
                if (use_material_registry) {
                    auto const& definition = material_registry->material(
                        static_cast<int>(material));
                    auto const& handle = *definition.electron_eos;
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        handle.material_id <= static_cast<std::int64_t>(
                            std::numeric_limits<int>::max()),
                        "Singularity-EOS material ID for registered material '"
                            + definition.name
                            + "' exceeds the SP5 integer range.");
                    table_file = handle.file;
                    material_id = static_cast<int>(handle.material_id);

                    bool const legacy_file_is_set = pp_hybrid.query(
                        prefix + "_table_file", table_file);
                    bool const legacy_id_is_set = pp_hybrid.query(
                        prefix + "_material_id", material_id);
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        legacy_file_is_set == legacy_id_is_set,
                        "Legacy Singularity-EOS file and material-ID inputs "
                        "must be specified together for species '"
                            + m_material_species[material]
                            + "'.");
                    if (legacy_file_is_set) {
                        warpx::materials::MaterialTableHandle const configured{
                            table_file, static_cast<std::int64_t>(material_id),
                            handle.material_key};
                        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                            warpx::materials::MaterialRegistry::sameTableHandle(
                                configured, handle),
                            "Singularity-EOS table file/material ID for "
                            "species '" + m_material_species[material]
                                + "' does not match its materials.names "
                                  "metadata.");
                    }
                } else {
                    pp_hybrid.get(prefix + "_table_file", table_file);
                    pp_hybrid.get(prefix + "_material_id", material_id);
                }

                // GetOnDevice() under the supported host-only build makes a
                // separate malloc-backed table marked OnDevice. Its inverse
                // lookups suppress mutable root-counter instrumentation and
                // are consequently safe to share between OpenMP threads.
                singularity::SpinerEOSDependsRhoT host_table(
                    table_file, material_id,
                    singularity::TableSplit::ElectronOnly);
                m_tables.emplace_back(host_table.GetOnDevice());
                host_table.Finalize();

                auto const& table = m_tables.back();
                m_executor.m_atomic_mass[material] =
                    static_cast<amrex::Real>(table.MeanAtomicMass());
                m_executor.m_atomic_number[material] =
                    static_cast<amrex::Real>(table.MeanAtomicNumber());
                common_temperature_floor = amrex::max(
                    common_temperature_floor,
                    static_cast<amrex::Real>(table.MinimumTemperature()));
                common_temperature_ceiling = amrex::min(
                    common_temperature_ceiling,
                    static_cast<amrex::Real>(table.TMax()));
                m_executor.m_minimum_mass_density[material] =
                    amrex::Real(1.0e3)
                    * static_cast<amrex::Real>(table.MinimumDensity());
                m_executor.m_maximum_mass_density[material] =
                    amrex::Real(1.0e3)
                    * static_cast<amrex::Real>(table.MaximumDensity());
            }
            if (!(common_temperature_ceiling
                    > common_temperature_floor))
            {
                amrex::Abort(
                    "Singularity-EOS electron tables do not share a "
                    "non-empty temperature interval.");
            }
            m_executor.m_num_materials =
                static_cast<int>(m_material_species.size());
            m_executor.m_tables = m_tables.data();
            m_executor.m_table_temperature_floor =
                common_temperature_floor;
            m_executor.m_table_temperature_ceiling =
                common_temperature_ceiling;
            return;
#endif
        }
        if (model != "fixed_charge_latent_energy") {
            amrex::Abort(
                "Unknown hybrid_pic_model.electron_thermodynamics='" + model
                + "'. Valid values are ideal_gas and "
                  "fixed_charge_latent_energy, and singularity_spiner.");
        }

        std::vector<amrex::Real> transition_temperature_eV;
        std::vector<amrex::Real> latent_energy_eV;
        std::vector<amrex::Real> transition_sharpness;
        pp_hybrid.getarr(
            "electron_latent_transition_temperature_eV",
            transition_temperature_eV);
        pp_hybrid.getarr("electron_latent_energy_eV", latent_energy_eV);
        if (pp_hybrid.contains("electron_latent_sharpness")) {
            pp_hybrid.getarr(
                "electron_latent_sharpness", transition_sharpness);
        } else {
            transition_sharpness.assign(
                transition_temperature_eV.size(), amrex::Real(4.0));
        }

        if (transition_temperature_eV.empty()
            || transition_temperature_eV.size()
                > ElectronThermodynamicsExecutor::max_latent_transitions
            || latent_energy_eV.size() != transition_temperature_eV.size()
            || transition_sharpness.size() != transition_temperature_eV.size())
        {
            amrex::Abort(
                "hybrid_pic_model analytic latent-energy arrays must have the "
                "same length, between one and "
                + std::to_string(
                    ElectronThermodynamicsExecutor::max_latent_transitions)
                + ".");
        }

        m_executor.m_model =
            ElectronThermodynamicsModel::FixedChargeLatentEnergy;
        m_executor.m_num_latent_transitions =
            static_cast<int>(transition_temperature_eV.size());
        for (int transition = 0;
             transition < m_executor.m_num_latent_transitions; ++transition)
        {
            if (!std::isfinite(transition_temperature_eV[transition])
                || !std::isfinite(latent_energy_eV[transition])
                || !std::isfinite(transition_sharpness[transition])
                || !(transition_temperature_eV[transition] > 0.0)
                || !(latent_energy_eV[transition] > 0.0)
                || !(transition_sharpness[transition] >= 2.0)
                || !(transition_sharpness[transition] <= 64.0)
                || (transition > 0
                    && !(transition_temperature_eV[transition]
                        > transition_temperature_eV[transition - 1])))
            {
                amrex::Abort(
                    "Analytic latent-energy transition temperatures and energies "
                    "must be finite and positive, temperatures must be strictly "
                    "increasing, and sharpness values must be finite and at "
                    "least two and at most 64.");
            }
            amrex::Real const transition_temperature =
                transition_temperature_eV[transition]
                * PhysConst::q_e / PhysConst::kb;
            amrex::Real const latent_energy =
                latent_energy_eV[transition] * PhysConst::q_e;
            if (!std::isfinite(transition_temperature)
                || !std::isfinite(latent_energy)
                || !(transition_temperature > 0.0)
                || !(latent_energy > 0.0))
            {
                amrex::Abort(
                    "Converted analytic latent-energy temperatures and "
                    "energies must be finite and positive in WarpX precision.");
            }
            m_executor.m_latent_transition_temperature[transition] =
                transition_temperature;
            m_executor.m_latent_energy_per_electron[transition] = latent_energy;
            m_executor.m_latent_transition_sharpness[transition] =
                transition_sharpness[transition];
        }
        ReadAnalyticComposition(pp_hybrid);
    }

    [[nodiscard]] ElectronThermodynamicsExecutor executor () const noexcept
    {
        return m_executor;
    }

    [[nodiscard]] bool isFixedChargeLatentEnergy () const noexcept
    {
        return m_executor.isFixedChargeLatentEnergy();
    }

    [[nodiscard]] bool isSingularitySpiner () const noexcept
    {
        return m_executor.isSingularitySpiner();
    }

    [[nodiscard]] int numMaterials () const noexcept
    {
        return m_executor.m_num_materials;
    }

    [[nodiscard]] std::string const& materialSpeciesName (
        int const material) const
    {
        return m_material_species.at(material);
    }

    void setMaterialMassPerUnitCharge (
        int const material,
        amrex::Real const mass_per_charge)
    {
        if (!(mass_per_charge > amrex::Real(0.0))
            || !std::isfinite(mass_per_charge))
        {
            amrex::Abort(
                "Electron-thermodynamics material species require finite "
                "positive mass per unit charge.");
        }
        m_executor.m_mass_per_charge[material] = mass_per_charge;
    }

    void setMaterialMassPerPhysicalCharge (
        int const material,
        amrex::Real const mass_per_physical_charge)
    {
        if (!(mass_per_physical_charge > amrex::Real(0.0))
            || !std::isfinite(mass_per_physical_charge))
        {
            amrex::Abort(
                "Electron-thermodynamics material species require finite "
                "positive mass per physical charge.");
        }
        m_executor.m_mass_per_physical_charge[material] =
            mass_per_physical_charge;
    }

    /** Whether this backend defines a finite positive caloric heat capacity. */
    [[nodiscard]] bool supportsEnergyCoupling () const noexcept
    {
        return m_executor.isSingularitySpiner()
            || (std::isfinite(m_executor.m_gamma)
                && m_executor.m_gamma > amrex::Real(1.0));
    }

    /** Whether the implicit LTE residual may use U(T)=C_V*T exactly. */
    [[nodiscard]] bool supportsConstantHeatCapacityLte () const noexcept
    {
        return supportsEnergyCoupling() && m_executor.isIdealGas();
    }

private:
    void ReadAnalyticComposition (amrex::ParmParse const& pp_hybrid)
    {
        bool const has_species =
            pp_hybrid.contains("electron_composition_species");
        bool const has_atomic_mass =
            pp_hybrid.contains("electron_composition_atomic_mass");
        bool const has_atomic_number =
            pp_hybrid.contains("electron_composition_atomic_number");
        if (!has_species && !has_atomic_mass && !has_atomic_number) { return; }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            has_species && has_atomic_mass && has_atomic_number,
            "hybrid_pic_model.electron_composition_species, "
            "electron_composition_atomic_mass, and "
            "electron_composition_atomic_number must be specified together.");
        pp_hybrid.getarr(
            "electron_composition_species", m_material_species);
        std::vector<amrex::Real> atomic_mass;
        std::vector<amrex::Real> atomic_number;
        pp_hybrid.getarr(
            "electron_composition_atomic_mass", atomic_mass);
        pp_hybrid.getarr(
            "electron_composition_atomic_number", atomic_number);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_material_species.size() == 1U
                && atomic_mass.size() == m_material_species.size()
                && atomic_number.size() == m_material_species.size(),
            "The first analytic composition adapter requires exactly one "
            "species and equally sized atomic-mass/atomic-number arrays.");
        for (std::size_t material = 0;
             material < m_material_species.size(); ++material)
        {
            auto const material_index = static_cast<int>(material);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::isfinite(atomic_mass[material])
                    && atomic_mass[material] > amrex::Real(0.0)
                    && std::isfinite(atomic_number[material])
                    && atomic_number[material] >= amrex::Real(1.0)
                    && atomic_number[material] <= amrex::Real(118.0),
                "Analytic composition atomic masses must be positive and "
                "atomic numbers must lie in [1,118].");
            m_executor.m_atomic_mass[material_index] = atomic_mass[material];
            m_executor.m_atomic_number[material_index] = atomic_number[material];
        }
        m_executor.m_num_materials =
            static_cast<int>(m_material_species.size());
    }

    void Finalize () noexcept
    {
#ifdef WARPX_USE_SINGULARITY_EOS
        if (!m_tables.empty()) { amrex::Gpu::synchronize(); }
        for (auto& table : m_tables) { table.Finalize(); }
        m_tables.clear();
        m_executor.m_tables = nullptr;
#endif
        m_executor.m_num_materials = 0;
        m_executor.m_use_resolved_material_selection = false;
        m_executor.m_resolved_material_selector = {};
        m_material_species.clear();
    }

    ElectronThermodynamicsExecutor m_executor;
    std::vector<std::string> m_material_species;
#ifdef WARPX_USE_SINGULARITY_EOS
    std::vector<singularity::SpinerEOSDependsRhoT> m_tables;
#endif
};

#endif // WARPX_ELECTRONTHERMODYNAMICS_H_
