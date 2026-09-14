/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronThermodynamics.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace amrex::literals;

namespace
{
    using namespace amrex;
    static_assert(std::is_same_v<Real, amrex::Real>);

    /** Manufactured electron EOS in Singularity's CGS units. */
    class NonlinearElectronEOS
    {
    public:
        static constexpr double cv0 = 2.0e8;
        static constexpr double cv_slope = 1.0e5;
        static constexpr double gas_constant = 8.0e7;

        PORTABLE_INLINE_FUNCTION
        double InternalEnergyFromDensityTemperature (
            double const /*density*/,
            double const temperature) const
        {
            return cv0 * temperature
                + 0.5 * cv_slope * temperature * temperature;
        }

        PORTABLE_INLINE_FUNCTION
        double TemperatureFromDensityInternalEnergy (
            double const /*density*/,
            double const energy) const
        {
            return (-cv0 + std::sqrt(cv0 * cv0 + 2.0 * cv_slope * energy))
                / cv_slope;
        }

        PORTABLE_INLINE_FUNCTION
        double PressureFromDensityTemperature (
            double const density,
            double const temperature) const
        {
            return density * gas_constant * temperature;
        }

        PORTABLE_INLINE_FUNCTION
        double SpecificHeatFromDensityTemperature (
            double const /*density*/,
            double const temperature) const
        {
            return cv0 + cv_slope * temperature;
        }

        PORTABLE_INLINE_FUNCTION
        double MeanAtomicMass () const { return 183.84; }

        PORTABLE_INLINE_FUNCTION
        double MeanAtomicNumber () const { return 74.0; }
    };

    bool close (
        double const actual,
        double const expected,
        double const relative_tolerance)
    {
        return std::abs(actual - expected)
            <= relative_tolerance * std::max(std::abs(expected), 1.0);
    }

    double readFixtureParameter (
        char const* const environment_name,
        double const default_value,
        double const minimum,
        double const maximum,
        bool const require_integer,
        bool const strict_minimum)
    {
        char const* const text = std::getenv(environment_name);
        if (text == nullptr) { return default_value; }
        char* end = nullptr;
        double const value = std::strtod(text, &end);
        if (end == text || *end != '\0' || !std::isfinite(value)
            || (strict_minimum ? value <= minimum : value < minimum)
            || value > maximum
            || (require_integer && std::floor(value) != value))
        {
            throw std::runtime_error(
                std::string(environment_name)
                + " must be finite and within its validated range");
        }
        return value;
    }

    /** Write a minimal split SP5 table for the exact ElectronOnly loader. */
    std::string writeSplitSpinerFixture (
        std::filesystem::path const& requested_fixture,
        double const specific_energy_reference = 0.0,
        int const material_id = 7400,
        double const thermodynamic_scale = 1.0,
        double const atomic_mass = 183.84,
        double const atomic_number = 74.0,
        int const num_temperature_points = 128,
        double const maximum_density = 10.0)
    {
        using DataBox = singularity::spiner_common::DataBox;
        using singularity::table_utils::Bounds;
        using singularity::table_utils::SpinerTableGridParams;

        SpinerTableGridParams parameters;
        parameters.rhoMin = 0.1;
        parameters.rhoMax = maximum_density;
        parameters.TMin = 100.0;
        parameters.TMax = 1.0e8;
        parameters.numRho = 48;
        parameters.numT = num_temperature_points;
        parameters.piecewiseRho = false;
        parameters.piecewiseT = false;
        parameters.rhoNormal = 1.0;

        Bounds l_rho;
        Bounds l_temperature;
        singularity::table_utils::constructRhoBounds(parameters, l_rho);
        singularity::table_utils::constructTBounds(parameters, l_temperature);
        int const num_rho = static_cast<int>(l_rho.grid.nPoints());
        int const num_temperature =
            static_cast<int>(l_temperature.grid.nPoints());

        DataBox pressure;
        pressure.resize(num_rho, num_temperature);
        pressure.setRange(0, l_temperature.grid);
        pressure.setRange(1, l_rho.grid);
        DataBox energy;
        DataBox bulk_modulus;
        DataBox dp_drho;
        DataBox dp_de;
        DataBox dt_drho;
        DataBox dt_de;
        DataBox de_drho;
        DataBox de_dt;
        energy.copyMetadata(pressure);
        bulk_modulus.copyMetadata(pressure);
        dp_drho.copyMetadata(pressure);
        dp_de.copyMetadata(pressure);
        dt_drho.copyMetadata(pressure);
        dt_de.copyMetadata(pressure);
        de_drho.copyMetadata(pressure);
        de_dt.copyMetadata(pressure);

        for (int j = 0; j < num_rho; ++j) {
            double const density = singularity::spiner_common::from_log(
                l_rho.grid.x(j), l_rho.offset);
            for (int i = 0; i < num_temperature; ++i) {
                double const temperature = singularity::spiner_common::from_log(
                    l_temperature.grid.x(i), l_temperature.offset);
                double const cv = thermodynamic_scale
                    * (NonlinearElectronEOS::cv0
                        + NonlinearElectronEOS::cv_slope * temperature);
                pressure(j, i) = density
                    * thermodynamic_scale
                    * NonlinearElectronEOS::gas_constant * temperature;
                energy(j, i) = specific_energy_reference
                    + thermodynamic_scale * (
                        NonlinearElectronEOS::cv0 * temperature
                        + 0.5 * NonlinearElectronEOS::cv_slope
                            * temperature * temperature);
                dp_drho(j, i) =
                    thermodynamic_scale
                    * NonlinearElectronEOS::gas_constant * temperature;
                dp_de(j, i) = density
                    * thermodynamic_scale
                    * NonlinearElectronEOS::gas_constant / cv;
                dt_drho(j, i) = 0.0;
                dt_de(j, i) = 1.0 / cv;
                de_drho(j, i) = 0.0;
                de_dt(j, i) = cv;
                bulk_modulus(j, i) = pressure(j, i)
                    + density * dp_de(j, i) * energy(j, i);
            }
        }

        DataBox cold_pressure;
        cold_pressure.resize(num_rho);
        cold_pressure.setRange(0, l_rho.grid);
        DataBox cold_energy;
        DataBox cold_bulk_modulus;
        DataBox cold_dp_drho;
        cold_energy.copyMetadata(cold_pressure);
        cold_bulk_modulus.copyMetadata(cold_pressure);
        cold_dp_drho.copyMetadata(cold_pressure);
        for (int j = 0; j < num_rho; ++j) {
            double const density = singularity::spiner_common::from_log(
                l_rho.grid.x(j), l_rho.offset);
            double const temperature = parameters.TMin;
            double const cv = thermodynamic_scale
                * (NonlinearElectronEOS::cv0
                    + NonlinearElectronEOS::cv_slope * temperature);
            cold_pressure(j) = density
                * thermodynamic_scale
                * NonlinearElectronEOS::gas_constant * temperature;
            cold_energy(j) = specific_energy_reference
                + thermodynamic_scale * (
                    NonlinearElectronEOS::cv0 * temperature
                    + 0.5 * NonlinearElectronEOS::cv_slope
                        * temperature * temperature);
            cold_dp_drho(j) =
                thermodynamic_scale
                * NonlinearElectronEOS::gas_constant * temperature;
            cold_bulk_modulus(j) = cold_pressure(j)
                + density * density
                    * thermodynamic_scale
                    * NonlinearElectronEOS::gas_constant / cv
                    * cold_energy(j);
        }

        if (requested_fixture.empty()) {
            throw std::runtime_error("The split SP5 fixture path is empty");
        }
        std::filesystem::path const fixture =
            std::filesystem::absolute(requested_fixture);
        if (fixture.has_parent_path()) {
            std::filesystem::create_directories(fixture.parent_path());
        }
        hid_t const file = H5Fcreate(
            fixture.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file < 0) {
            throw std::runtime_error(
                "Failed to create split SP5 fixture at '" + fixture.string()
                + "'");
        }
        int const log_type = singularity::FastMath::Settings::log_type;
        int status = H5LTset_attribute_int(
            file, "/", SP5::logType, &log_type, 1);
        std::string const material_group_name =
            std::to_string(material_id);
        hid_t const material_group = H5Gcreate(
            file, material_group_name.c_str(), H5P_DEFAULT,
            H5P_DEFAULT, H5P_DEFAULT);
        double const normal_density = 1.0;
        status += H5LTset_attribute_double(
            file, material_group_name.c_str(), SP5::Offsets::rho,
            &l_rho.offset, 1);
        status += H5LTset_attribute_double(
            file, material_group_name.c_str(), SP5::Offsets::T,
            &l_temperature.offset, 1);
        status += H5LTset_attribute_double(
            file, material_group_name.c_str(),
            SP5::Material::normalDensity, &normal_density, 1);
        status += H5LTset_attribute_double(
            file, material_group_name.c_str(),
            SP5::Material::meanAtomicMass, &atomic_mass, 1);
        status += H5LTset_attribute_double(
            file, material_group_name.c_str(),
            SP5::Material::meanAtomicNumber, &atomic_number, 1);

        hid_t const rho_temperature_group = H5Gcreate(
            material_group, SP5::Depends::logRhoLogT,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t const electron_group = H5Gcreate(
            rho_temperature_group, SP5::SubTable::electronOnly,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status += pressure.saveHDF(electron_group, SP5::Fields::P);
        status += energy.saveHDF(electron_group, SP5::Fields::sie);
        status += bulk_modulus.saveHDF(electron_group, SP5::Fields::bMod);
        status += dp_drho.saveHDF(electron_group, SP5::Fields::dPdRho);
        status += dp_de.saveHDF(electron_group, SP5::Fields::dPdE);
        status += dt_drho.saveHDF(electron_group, SP5::Fields::dTdRho);
        status += dt_de.saveHDF(electron_group, SP5::Fields::dTdE);
        status += de_drho.saveHDF(electron_group, SP5::Fields::dEdRho);
        status += de_dt.saveHDF(electron_group, SP5::Fields::dEdT);

        hid_t const cold_group = H5Gcreate(
            material_group, SP5::Depends::coldCurve,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status += cold_pressure.saveHDF(cold_group, SP5::Fields::P);
        status += cold_energy.saveHDF(cold_group, SP5::Fields::sie);
        status += cold_bulk_modulus.saveHDF(cold_group, SP5::Fields::bMod);
        status += cold_dp_drho.saveHDF(cold_group, SP5::Fields::dPdRho);
        status += H5Gclose(cold_group);
        status += H5Gclose(electron_group);
        status += H5Gclose(rho_temperature_group);
        status += H5Gclose(material_group);
        status += H5Fclose(file);
        if (status != 0) {
            throw std::runtime_error("Failed to create split SP5 fixture");
        }
        return fixture.string();
    }
}

int main (int argc, char* argv[])
{
    static_assert(
        std::is_trivially_copyable_v<ElectronThermodynamicsExecutor>);
    static_assert(
        sizeof(ElectronThermodynamicsExecutor) < 1024,
        "The kernel executor must contain table views, not table storage.");

    double const fixture_atomic_mass = readFixtureParameter(
        "WARPX_SPINER_FIXTURE_ATOMIC_MASS", 183.84, 0.0,
        std::numeric_limits<double>::max(), false, true);
    double const fixture_atomic_number = readFixtureParameter(
        "WARPX_SPINER_FIXTURE_ATOMIC_NUMBER", 74.0, 1.0, 118.0, true,
        false);
    int const fixture_num_temperature_points = static_cast<int>(
        readFixtureParameter(
            "WARPX_SPINER_FIXTURE_NUM_T", 128.0, 16.0, 4096.0, true,
            false));
    double const fixture_maximum_density = readFixtureParameter(
        "WARPX_SPINER_FIXTURE_RHO_MAX", 10.0, 0.1,
        std::numeric_limits<double>::max(), false, true);

    amrex::Initialize(argc, argv);
    bool pass = true;
    {
        singularity::SpinerTableGridParams parameters;
        parameters.rhoMin = 0.1;
        parameters.rhoMax = 10.0;
        parameters.TMin = 100.0;
        parameters.TMax = 1.0e8;
        parameters.numRho = 48;
        parameters.numT = 128;
        parameters.piecewiseRho = false;
        parameters.piecewiseT = false;
        parameters.matid = 7400;

        std::vector<singularity::SpinerEOSDependsRhoT> tables;
        tables.reserve(2);
        for (int material = 0; material < 2; ++material) {
            singularity::SpinerEOSDependsRhoT host_table(
                NonlinearElectronEOS{}, parameters);
            tables.emplace_back(host_table.GetOnDevice());
            host_table.Finalize();
            ++parameters.matid;
        }

        ElectronThermodynamicsExecutor thermodynamics;
        thermodynamics.m_model =
            ElectronThermodynamicsModel::SingularitySpiner;
        thermodynamics.m_num_materials = 2;
        thermodynamics.m_tables = tables.data();
        thermodynamics.m_mass_per_charge[0] = 2.0_rt;
        thermodynamics.m_mass_per_charge[1] = 3.0_rt;
        thermodynamics.m_atomic_mass[0] = 183.84_rt;
        thermodynamics.m_atomic_mass[1] = 183.84_rt;
        thermodynamics.m_atomic_number[0] = 74.0_rt;
        thermodynamics.m_atomic_number[1] = 74.0_rt;
        thermodynamics.m_table_temperature_floor =
            tables.front().MinimumTemperature();
        thermodynamics.m_table_temperature_ceiling = tables.front().TMax();
        for (int material = 0; material < 2; ++material) {
            thermodynamics.m_minimum_mass_density[material] =
                1.0e3_rt * tables[material].MinimumDensity();
            thermodynamics.m_maximum_mass_density[material] =
                1.0e3_rt * tables[material].MaximumDensity();
        }

        // Unit-ion density is q_e n_i for every ion species. Physical charge
        // density is q_i n_i, so its material multiplier must include q_i.
        ElectronThermodynamicsExecutor mapping;
        constexpr amrex::Real particle_mass = 40.0_rt * PhysConst::m_u;
        constexpr amrex::Real ion_number_density = 3.0e20_rt;
        constexpr amrex::Real z_one = 1.0_rt;
        constexpr amrex::Real z_twenty = 20.0_rt;
        mapping.m_mass_per_charge[0] = particle_mass / PhysConst::q_e;
        mapping.m_mass_per_physical_charge[0] =
            particle_mass / (z_one * PhysConst::q_e);
        mapping.m_mass_per_charge[1] = particle_mass / PhysConst::q_e;
        mapping.m_mass_per_physical_charge[1] =
            particle_mass / (z_twenty * PhysConst::q_e);
        amrex::Real const unit_ion_charge_density =
            PhysConst::q_e * ion_number_density;
        amrex::Real const physical_charge_density_z20 =
            z_twenty * unit_ion_charge_density;
        amrex::Real const expected_mass_density =
            particle_mass * ion_number_density;
        pass = pass && close(
            mapping.materialMassDensityFromUnitIonChargeDensity(
                0, unit_ion_charge_density),
            expected_mass_density, 2.0e-14);
        pass = pass && close(
            mapping.materialMassDensityFromPhysicalChargeDensity(
                0, unit_ion_charge_density),
            expected_mass_density, 2.0e-14);
        pass = pass && close(
            mapping.materialMassDensityFromUnitIonChargeDensity(
                1, unit_ion_charge_density),
            expected_mass_density, 2.0e-14);
        pass = pass && close(
            mapping.materialMassDensityFromPhysicalChargeDensity(
                1, physical_charge_density_z20),
            expected_mass_density, 2.0e-14);
        pass = pass
            && mapping.materialMassDensityFromPhysicalChargeDensity(
                   1, -1.0_rt)
                < 0.0_rt;
        pass = pass && std::isnan(
            mapping.materialMassDensityFromPhysicalChargeDensity(
                1, std::numeric_limits<amrex::Real>::quiet_NaN()));

        ElectronThermodynamicsExecutor::MaterialMassDensities mass_density{};
        mass_density[0] = 600.0_rt;
        mass_density[1] = 400.0_rt;
        amrex::Real const charge_density = 1.0e10_rt;
        amrex::Real const temperature = 5000.0_rt;
        ElectronThermodynamicState const state =
            thermodynamics.stateFromMaterialMassDensitiesTemperature(
                charge_density, mass_density, temperature);
        amrex::Real const recovered_temperature = thermodynamics
            .temperatureFromMaterialMassDensitiesThermalEnergyDensity(
                charge_density, mass_density,
                state.internal_energy_density);

        amrex::Real const total_mass_density =
            mass_density[0] + mass_density[1];
        amrex::Real const temperature_floor =
            thermodynamics.minimumTemperature();
        amrex::Real const expected_pressure = 0.1_rt
            * (1.0e-3_rt * total_mass_density)
            * NonlinearElectronEOS::gas_constant * temperature;
        amrex::Real const expected_energy = total_mass_density * 1.0e-4_rt
            * (NonlinearElectronEOS::cv0
                * temperature
                + 0.5_rt * NonlinearElectronEOS::cv_slope
                    * temperature * temperature);
        amrex::Real const expected_heat_capacity =
            total_mass_density * 1.0e-4_rt
            * (NonlinearElectronEOS::cv0
                + NonlinearElectronEOS::cv_slope * temperature);
        amrex::Real const expected_ion_density = total_mass_density
            / (183.84_rt * PhysConst::m_u);
        amrex::Real const expected_mean_ionization = charge_density
            / PhysConst::q_e / expected_ion_density;

        pass = pass && close(state.pressure, expected_pressure, 2.0e-3);
        pass = pass && close(
            state.internal_energy_density, expected_energy, 2.0e-3);
        pass = pass && close(
            state.heat_capacity_density, expected_heat_capacity, 2.0e-2);
        pass = pass && close(
            state.ion_number_density, expected_ion_density, 2.0e-3);
        pass = pass && close(state.mean_atomic_number, 74.0_rt, 2.0e-3);
        pass = pass && close(
            state.mean_ionization, expected_mean_ionization, 2.0e-3);
        pass = pass && close(recovered_temperature, temperature, 2.0e-6);

        // Thousands of concurrent inversions share the same execution tables.
        // This detects the mutable root-counter race present in ordinary
        // host-table handles when OpenMP is enabled.
        constexpr int num_inverses = 4096;
        amrex::Gpu::DeviceVector<amrex::Real> inverse_results(num_inverses);
        amrex::Real* const results = inverse_results.data();
        auto const executor = thermodynamics;
        auto const densities = mass_density;
        amrex::ParallelForOMP(num_inverses, [=] AMREX_GPU_DEVICE (int i) noexcept
        {
            amrex::Real const fraction =
                static_cast<amrex::Real>(i + 1)
                / static_cast<amrex::Real>(num_inverses + 1);
            amrex::Real const trial_temperature = temperature_floor
                + fraction * (temperature - temperature_floor);
            amrex::Real const trial_energy = executor
                .stateFromMaterialMassDensitiesTemperature(
                    charge_density, densities, trial_temperature)
                .internal_energy_density;
            results[i] = executor
                .temperatureFromMaterialMassDensitiesThermalEnergyDensity(
                    charge_density, densities, trial_energy)
                - trial_temperature;
        });
        std::vector<amrex::Real> host_results(num_inverses);
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost,
            inverse_results.begin(), inverse_results.end(),
            host_results.begin());
        for (amrex::Real const error : host_results) {
            pass = pass && std::abs(error)
                <= 2.0e-6_rt * std::max(temperature, 1.0_rt);
        }

        // A sub-table-minimum PIC shape tail uses a continuous dilute
        // extension, while an actually over-dense resolved state fails loudly.
        auto trace_density = mass_density;
        trace_density[0] = 0.5_rt
            * thermodynamics.m_minimum_mass_density[0];
        ElectronThermodynamicState const trace_state =
            thermodynamics.stateFromMaterialMassDensitiesTemperature(
                charge_density, trace_density, temperature);
        amrex::Real const expected_trace_pressure = 0.1_rt
            * (1.0e-3_rt * (trace_density[0] + trace_density[1]))
            * NonlinearElectronEOS::gas_constant * temperature;
        bool const trace_pressure_pass = close(
            trace_state.pressure, expected_trace_pressure, 2.0e-3);
        if (!trace_pressure_pass) {
            std::cerr << "Dilute pressure mismatch: actual="
                      << trace_state.pressure << " expected="
                      << expected_trace_pressure << '\n';
        }
        pass = pass && trace_pressure_pass;

        auto invalid_density = mass_density;
        invalid_density[0] = 2.0_rt
            * thermodynamics.m_maximum_mass_density[0];
        pass = pass && std::isnan(
            thermodynamics.stateFromMaterialMassDensitiesTemperature(
                charge_density, invalid_density, temperature).pressure);
        auto negative_density = mass_density;
        negative_density[0] = -1.0_rt;
        pass = pass
            && thermodynamics
                    .materialMassDensityFromUnitIonChargeDensity(
                        0, -1.0_rt)
                < 0.0_rt;
        pass = pass && std::isnan(
            thermodynamics.stateFromMaterialMassDensitiesTemperature(
                charge_density, negative_density, temperature).pressure);
        auto nan_density = mass_density;
        nan_density[0] = std::numeric_limits<amrex::Real>::quiet_NaN();
        pass = pass && std::isnan(
                thermodynamics.materialMassDensityFromUnitIonChargeDensity(
                    0, std::numeric_limits<amrex::Real>::quiet_NaN()));
        pass = pass && std::isnan(
            thermodynamics.stateFromMaterialMassDensitiesTemperature(
                charge_density, nan_density, temperature).pressure);
        pass = pass && std::isnan(
            thermodynamics.stateFromMaterialMassDensitiesTemperature(
                std::numeric_limits<amrex::Real>::quiet_NaN(),
                mass_density, temperature).pressure);
        pass = pass && std::isnan(
            thermodynamics.stateFromChargeDensityTemperature(
                charge_density, temperature).pressure);

        amrex::Gpu::synchronize();
        for (auto& table : tables) { table.Finalize(); }

        // Exercise the production host owner and exact ElectronOnly file path.
        // An explicit path also requests retention, so an application test can
        // generate a uniquely named fixture without sharing /tmp state.
        char const* const requested_fixture =
            std::getenv("WARPX_SPINER_FIXTURE_PATH");
        if (requested_fixture != nullptr && requested_fixture[0] == '\0') {
            throw std::runtime_error(
                "WARPX_SPINER_FIXTURE_PATH must not be empty");
        }
        std::filesystem::path const fixture_path = requested_fixture != nullptr
            ? std::filesystem::path(requested_fixture)
            : std::filesystem::temp_directory_path()
                / "warpx_electron_only_fixture.sp5";
        std::string const fixture = writeSplitSpinerFixture(
            fixture_path, 0.0, 7400, 1.0,
            fixture_atomic_mass, fixture_atomic_number,
            fixture_num_temperature_points, fixture_maximum_density);
        std::filesystem::path shifted_fixture_path = fixture_path;
        shifted_fixture_path.replace_filename(
            fixture_path.stem().string() + "_negative_reference"
            + fixture_path.extension().string());
        // A caloric EOS may choose an arbitrary energy zero.  This offset
        // makes the test state and table floor negative without changing any
        // pressure, heat capacity, or temperature differences.
        constexpr double negative_specific_energy_reference = -1.0e13;
        std::string const shifted_fixture = writeSplitSpinerFixture(
            shifted_fixture_path, negative_specific_energy_reference, 7400, 1.0,
            fixture_atomic_mass, fixture_atomic_number,
            fixture_num_temperature_points, fixture_maximum_density);
        std::filesystem::path foam_fixture_path = fixture_path;
        foam_fixture_path.replace_filename(
            fixture_path.stem().string() + "_registry_foam"
            + fixture_path.extension().string());
        std::filesystem::path tungsten_fixture_path = fixture_path;
        tungsten_fixture_path.replace_filename(
            fixture_path.stem().string() + "_registry_tungsten"
            + fixture_path.extension().string());
        // Material IDs are local to each file. Distinct files and registry
        // keys disambiguate these two deliberately identical IDs.
        std::string const foam_fixture = writeSplitSpinerFixture(
            foam_fixture_path, 0.0, 7400, 0.5, 12.0, 6.0);
        std::string const tungsten_fixture = writeSplitSpinerFixture(
            tungsten_fixture_path, 0.0, 7400, 2.0, 183.84, 74.0);
        {
            amrex::ParmParse pp("hybrid_pic_model");
            pp.add("electron_thermodynamics", "singularity_spiner");
            pp.addarr(
                "electron_eos_species", std::vector<std::string>{"ions"});
            pp.add("electron_eos_ions_table_file", fixture);
            pp.add("electron_eos_ions_material_id", 7400);

            ElectronThermodynamics owner;
            owner.ReadParameters(pp, 5.0_rt / 3.0_rt);
            owner.setMaterialMassPerUnitCharge(0, 1.0_rt);
            owner.setMaterialMassPerPhysicalCharge(0, 0.5_rt);
            pass = pass && close(
                owner.executor().m_mass_per_physical_charge[0], 0.5_rt,
                2.0e-14);
            auto const file_executor = owner.executor();
            ElectronThermodynamicsExecutor::MaterialMassDensities
                file_mass_density{};
            file_mass_density[0] = 1000.0_rt;
            ElectronThermodynamicState const file_state = file_executor
                .stateFromMaterialMassDensitiesTemperature(
                    charge_density, file_mass_density, temperature);
            ElectronThermodynamicState const file_floor_state = file_executor
                .stateFromMaterialMassDensitiesTemperature(
                    charge_density, file_mass_density,
                    file_executor.minimumTemperature());
            // U(Tmax) is over ten orders of magnitude larger than U(Tmin)
            // for this table. A tolerance scaled by the opposite endpoint
            // used to snap this resolvable near-floor increment back to Tmin.
            amrex::Real const small_energy_increment = 1.0e5_rt;
            amrex::Real const near_floor_temperature = file_executor
                .temperatureFromMaterialMassDensitiesThermalEnergyDensity(
                    charge_density, file_mass_density,
                    file_floor_state.internal_energy_density
                        + small_energy_increment);
            ElectronThermodynamicState const near_floor_state = file_executor
                .stateFromMaterialMassDensitiesTemperature(
                    charge_density, file_mass_density,
                    near_floor_temperature);
            amrex::Real const file_recovered_temperature = file_executor
                .temperatureFromMaterialMassDensitiesThermalEnergyDensity(
                    charge_density, file_mass_density,
                    file_state.internal_energy_density);
            amrex::Real const source_target_temperature = 25000.0_rt;
            ElectronThermodynamicState const source_target_state =
                file_executor.stateFromMaterialMassDensitiesTemperature(
                    charge_density, file_mass_density,
                    source_target_temperature);
            ElectronEnergySourceUpdate const source_update = file_executor
                .applyEnergyDensityIncrement(
                    charge_density, file_mass_density, temperature,
                    source_target_state.internal_energy_density
                        - file_state.internal_energy_density);
            pass = pass && close(
                file_state.pressure,
                0.1_rt * NonlinearElectronEOS::gas_constant * temperature,
                2.0e-3);
            pass = pass && file_state.internal_energy_density
                > file_floor_state.internal_energy_density;
            bool const near_floor_temperature_pass = near_floor_temperature
                > file_executor.minimumTemperature();
            bool const near_floor_energy_pass = close(
                near_floor_state.internal_energy_density
                    - file_floor_state.internal_energy_density,
                small_energy_increment, 1.0e-4);
            if (!near_floor_temperature_pass || !near_floor_energy_pass) {
                std::cerr << "Near-floor inverse mismatch: Tmin="
                          << file_executor.minimumTemperature()
                          << " recovered=" << near_floor_temperature
                          << " delta_U="
                          << near_floor_state.internal_energy_density
                                - file_floor_state.internal_energy_density
                          << " expected_delta_U=" << small_energy_increment
                          << '\n';
            }
            pass = pass && near_floor_temperature_pass
                && near_floor_energy_pass;
            pass = pass && close(
                file_recovered_temperature, temperature, 2.0e-6);
            pass = pass && source_update.valid
                && close(
                    source_update.temperature,
                    source_target_temperature, 2.0e-6)
                && close(
                    source_update.energy_change_density,
                    source_target_state.internal_energy_density
                        - file_state.internal_energy_density,
                    2.0e-6);
        }
        {
            amrex::ParmParse pp("hybrid_pic_model_shifted");
            pp.add("electron_thermodynamics", "singularity_spiner");
            pp.addarr(
                "electron_eos_species", std::vector<std::string>{"ions"});
            pp.add("electron_eos_ions_table_file", shifted_fixture);
            pp.add("electron_eos_ions_material_id", 7400);

            ElectronThermodynamics owner;
            owner.ReadParameters(pp, 5.0_rt / 3.0_rt);
            owner.setMaterialMassPerUnitCharge(0, 1.0_rt);
            auto const file_executor = owner.executor();
            ElectronThermodynamicsExecutor::MaterialMassDensities
                file_mass_density{};
            file_mass_density[0] = 1000.0_rt;
            ElectronThermodynamicState const shifted_state = file_executor
                .stateFromMaterialMassDensitiesTemperature(
                    charge_density, file_mass_density, temperature);
            amrex::Real const shifted_recovered_temperature = file_executor
                .temperatureFromMaterialMassDensitiesThermalEnergyDensity(
                    charge_density, file_mass_density,
                    shifted_state.internal_energy_density);
            amrex::Real const expected_shifted_energy = expected_energy
                + file_mass_density[0] * 1.0e-4_rt
                    * negative_specific_energy_reference;
            pass = pass && shifted_state.internal_energy_density < 0.0_rt;
            pass = pass && close(
                shifted_state.internal_energy_density,
                expected_shifted_energy, 2.0e-3);
            pass = pass && close(
                shifted_recovered_temperature, temperature, 2.0e-6);
        }
        {
            // Declaration order is opposite the canonical registry order.
            // No redundant legacy EOS file/ID parameters are supplied.
            amrex::ParmParse pp_materials("materials");
            pp_materials.addarr(
                "names", std::vector<std::string>{"tungsten", "foam"});
            pp_materials.add("mixed_cell_relative_tolerance", 1.0e-6);
            amrex::ParmParse pp_foam("materials.foam");
            pp_foam.addarr(
                "species", std::vector<std::string>{"foam_ions"});
            pp_foam.add("electron_eos_table_file", foam_fixture);
            pp_foam.add("electron_eos_material_id", 7400);
            amrex::ParmParse pp_tungsten("materials.tungsten");
            pp_tungsten.addarr(
                "species", std::vector<std::string>{"tungsten_ions"});
            pp_tungsten.add("electron_eos_table_file", tungsten_fixture);
            pp_tungsten.add("electron_eos_material_id", 7400);

            warpx::materials::MaterialRegistry registry;
            registry.ReadParameters();
            amrex::ParmParse pp("hybrid_pic_model_registry");
            pp.add("electron_thermodynamics", "singularity_spiner");

            ElectronThermodynamics owner;
            owner.ReadParameters(pp, 5.0_rt / 3.0_rt, &registry);
            owner.setMaterialMassPerUnitCharge(0, 1.0_rt);
            owner.setMaterialMassPerUnitCharge(1, 1.0_rt);
            auto const resolved_executor = owner.executor();
            pass = pass && owner.numMaterials() == 2;
            pass = pass && owner.materialSpeciesName(0) == "foam_ions";
            pass = pass && owner.materialSpeciesName(1) == "tungsten_ions";
            pass = pass
                && resolved_executor.m_use_resolved_material_selection;

            ElectronThermodynamicsExecutor::MaterialMassDensities
                foam_density{};
            foam_density[0] = 1000.0_rt;
            auto const foam_state = resolved_executor
                .stateFromMaterialMassDensitiesTemperature(
                    charge_density, foam_density, temperature);
            amrex::Real const expected_foam_pressure = 0.5_rt * 0.1_rt
                * NonlinearElectronEOS::gas_constant * temperature;
            amrex::Real const expected_foam_energy = 0.5_rt * 1000.0_rt
                * 1.0e-4_rt * (
                    NonlinearElectronEOS::cv0 * temperature
                    + 0.5_rt * NonlinearElectronEOS::cv_slope
                        * temperature * temperature);
            pass = pass && close(
                foam_state.pressure, expected_foam_pressure, 2.0e-3);
            pass = pass && close(
                foam_state.internal_energy_density,
                expected_foam_energy, 2.0e-3);
            pass = pass && close(
                foam_state.mean_atomic_number, 6.0_rt, 2.0e-3);

            ElectronThermodynamicsExecutor::MaterialMassDensities
                tungsten_density{};
            tungsten_density[1] = 1000.0_rt;
            auto const tungsten_state = resolved_executor
                .stateFromMaterialMassDensitiesTemperature(
                    charge_density, tungsten_density, temperature);
            amrex::Real const expected_tungsten_pressure = 2.0_rt * 0.1_rt
                * NonlinearElectronEOS::gas_constant * temperature;
            pass = pass && close(
                tungsten_state.pressure,
                expected_tungsten_pressure, 2.0e-3);
            pass = pass && close(
                tungsten_state.mean_atomic_number, 74.0_rt, 2.0e-3);
            amrex::Real const recovered_tungsten_temperature =
                resolved_executor
                    .temperatureFromMaterialMassDensitiesThermalEnergyDensity(
                        charge_density, tungsten_density,
                        tungsten_state.internal_energy_density);
            pass = pass && close(
                recovered_tungsten_temperature, temperature, 2.0e-6);

            // Sub-tolerance contamination selects only the dominant table;
            // it is not added as an EOS mixture.
            auto trace_contamination = foam_density;
            trace_contamination[1] = 5.0e-4_rt;
            auto const registered_trace_state = resolved_executor
                .stateFromMaterialMassDensitiesTemperature(
                    charge_density, trace_contamination, temperature);
            pass = pass && close(
                registered_trace_state.pressure,
                expected_foam_pressure, 2.0e-3);

            auto mixed_density = foam_density;
            mixed_density[1] = 1000.0_rt;
            pass = pass && resolved_executor.resolvedMaterial(mixed_density)
                == static_cast<int>(warpx::materials::MaterialRegistry::
                    ResolvedCell::Mixed);
            pass = pass && std::isnan(
                resolved_executor.stateFromMaterialMassDensitiesTemperature(
                    charge_density, mixed_density, temperature).pressure);

            ElectronThermodynamicsExecutor::MaterialMassDensities vacuum{};
            auto const vacuum_state = resolved_executor
                .stateFromMaterialMassDensitiesTemperature(
                    0.0_rt, vacuum, temperature);
            pass = pass && vacuum_state.pressure == 0.0_rt;
            pass = pass && vacuum_state.internal_energy_density == 0.0_rt;
            pass = pass && resolved_executor.resolvedMaterial(vacuum)
                == static_cast<int>(warpx::materials::MaterialRegistry::
                    ResolvedCell::Vacuum);

            // Preserve compatibility with registered inputs that redundantly
            // repeat matching legacy handles (including existing one-table
            // configurations).
            amrex::ParmParse pp_redundant(
                "hybrid_pic_model_registry_redundant");
            pp_redundant.add(
                "electron_thermodynamics", "singularity_spiner");
            pp_redundant.addarr(
                "electron_eos_species",
                std::vector<std::string>{"foam_ions", "tungsten_ions"});
            pp_redundant.add(
                "electron_eos_foam_ions_table_file", foam_fixture);
            pp_redundant.add(
                "electron_eos_foam_ions_material_id", 7400);
            pp_redundant.add(
                "electron_eos_tungsten_ions_table_file", tungsten_fixture);
            pp_redundant.add(
                "electron_eos_tungsten_ions_material_id", 7400);
            ElectronThermodynamics redundant_owner;
            redundant_owner.ReadParameters(
                pp_redundant, 5.0_rt / 3.0_rt, &registry);
            auto const redundant_executor = redundant_owner.executor();
            pass = pass && redundant_owner.numMaterials() == 2;
            pass = pass
                && redundant_executor.m_use_resolved_material_selection;
        }
        if (requested_fixture == nullptr
            && std::getenv("WARPX_KEEP_SPINER_FIXTURE") == nullptr)
        {
            std::filesystem::remove(fixture);
            std::filesystem::remove(shifted_fixture);
            std::filesystem::remove(foam_fixture);
            std::filesystem::remove(tungsten_fixture);
        } else {
            std::cout << "Retained split SP5 fixtures: " << fixture
                      << ", " << shifted_fixture << ", " << foam_fixture
                      << ", and " << tungsten_fixture << '\n';
        }

        // Optional manual regression against a separately generated table.
        // The simulations analytic_high_z_sp5 fixture uses material 7420,
        // A=183.84, nuclear Z=74, and represented fixed Zbar=20.  Keeping
        // this opt-in avoids making WarpX CTest depend on another repository,
        // while letting the exact published interface artifact exercise this
        // source-update primitive unchanged.
        char const* const external_table =
            std::getenv("WARPX_EXTERNAL_ELECTRON_ONLY_SP5");
        if (external_table != nullptr) {
            if (external_table[0] == '\0') {
                throw std::runtime_error(
                    "WARPX_EXTERNAL_ELECTRON_ONLY_SP5 must not be empty");
            }
            amrex::ParmParse pp("hybrid_pic_model_external_sp5");
            pp.add("electron_thermodynamics", "singularity_spiner");
            pp.addarr(
                "electron_eos_species", std::vector<std::string>{"W"});
            pp.add("electron_eos_W_table_file", external_table);
            pp.add("electron_eos_W_material_id", 7420);

            ElectronThermodynamics owner;
            owner.ReadParameters(pp, 5.0_rt / 3.0_rt);
            constexpr amrex::Real represented_zbar = 20.0_rt;
            amrex::Real const mass_per_charge = 183.84_rt * PhysConst::m_u
                / (represented_zbar * PhysConst::q_e);
            owner.setMaterialMassPerUnitCharge(0, mass_per_charge);
            auto const external_executor = owner.executor();
            constexpr amrex::Real mass_density_si = 19300.0_rt;
            ElectronThermodynamicsExecutor::MaterialMassDensities
                external_mass_density{};
            external_mass_density[0] = mass_density_si;
            amrex::Real const external_charge_density =
                mass_density_si / mass_per_charge;
            amrex::Real const kelvin_per_eV =
                PhysConst::q_e / PhysConst::kb;
            amrex::Real const external_old_temperature =
                250.0_rt * kelvin_per_eV;
            amrex::Real const external_target_temperature =
                800.0_rt * kelvin_per_eV;
            ElectronThermodynamicState const external_old_state =
                external_executor.stateFromMaterialMassDensitiesTemperature(
                    external_charge_density, external_mass_density,
                    external_old_temperature);
            ElectronThermodynamicState const external_target_state =
                external_executor.stateFromMaterialMassDensitiesTemperature(
                    external_charge_density, external_mass_density,
                    external_target_temperature);
            ElectronEnergySourceUpdate const external_source_update =
                external_executor.applyEnergyDensityIncrement(
                    external_charge_density, external_mass_density,
                    external_old_temperature,
                    external_target_state.internal_energy_density
                        - external_old_state.internal_energy_density);
            bool const external_pass = external_source_update.valid
                && close(
                    external_source_update.temperature,
                    external_target_temperature, 2.0e-6)
                && close(
                    external_source_update.energy_change_density,
                    external_target_state.internal_energy_density
                        - external_old_state.internal_energy_density,
                    2.0e-6)
                && close(
                    external_old_state.mean_atomic_number, 74.0_rt, 2.0e-6)
                && close(
                    external_old_state.mean_ionization,
                    represented_zbar, 2.0e-6);
            pass = pass && external_pass;
            std::cout
                << "External fixed-Zbar latent SP5 source update: "
                << external_old_temperature / kelvin_per_eV << " -> "
                << external_source_update.temperature / kelvin_per_eV
                << " eV, DeltaU="
                << external_source_update.energy_change_density
                << (external_pass ? " PASS\n" : " FAIL\n");
        }

        std::cout << "Spiner nonlinear electron table: P="
                  << state.pressure
                  << " U=" << state.internal_energy_density
                  << " Cv=" << state.heat_capacity_density
                  << " T(U)=" << recovered_temperature
                  << " executor_bytes="
                  << sizeof(ElectronThermodynamicsExecutor)
                  << (pass ? " PASS\n" : " FAIL\n");
    }
    amrex::Finalize();
    return pass ? 0 : 1;
}
