/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/MaterialOpacityTable.H"
#include "Radiation/PlanckExchange.H"

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>

#include <hdf5.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
    using namespace amrex::literals;
    using warpx::radiation::MaterialOpacityCoefficients;
    using warpx::radiation::MaterialOpacityTable;
    using warpx::radiation::EvaluatePlanckExchange;

    class Hdf5Handle
    {
    public:
        using Closer = herr_t (*) (hid_t);
        Hdf5Handle (hid_t id, Closer closer) : m_id(id), m_closer(closer)
        {
            if (id < 0) { throw std::runtime_error("HDF5 create/open failed"); }
        }
        ~Hdf5Handle ()
        {
            if (m_id >= 0 && m_closer != nullptr) { m_closer(m_id); }
        }
        Hdf5Handle (Hdf5Handle const&) = delete;
        Hdf5Handle& operator= (Hdf5Handle const&) = delete;
        Hdf5Handle (Hdf5Handle&& other) noexcept
            : m_id(std::exchange(other.m_id, -1)),
              m_closer(std::exchange(other.m_closer, nullptr))
        {}
        [[nodiscard]] hid_t get () const noexcept { return m_id; }

    private:
        hid_t m_id;
        Closer m_closer;
    };

    void check (herr_t status)
    {
        if (status < 0) { throw std::runtime_error("HDF5 operation failed"); }
    }

    Hdf5Handle scalar_space ()
    {
        return {H5Screate(H5S_SCALAR), H5Sclose};
    }

    Hdf5Handle string_type ()
    {
        Hdf5Handle result{H5Tcopy(H5T_C_S1), H5Tclose};
        check(H5Tset_size(result.get(), H5T_VARIABLE));
        check(H5Tset_cset(result.get(), H5T_CSET_UTF8));
        return result;
    }

    void write_string_attribute (
        hid_t location, char const* name, std::string const& value)
    {
        auto type = string_type();
        auto space = scalar_space();
        Hdf5Handle attribute{
            H5Acreate2(
                location, name, type.get(), space.get(), H5P_DEFAULT,
                H5P_DEFAULT),
            H5Aclose};
        char const* pointer = value.c_str();
        check(H5Awrite(attribute.get(), type.get(), &pointer));
    }

    template <typename T>
    void write_integer_attribute (
        hid_t location, char const* name, T value,
        hid_t file_type, hid_t memory_type)
    {
        auto space = scalar_space();
        Hdf5Handle attribute{
            H5Acreate2(
                location, name, file_type, space.get(), H5P_DEFAULT,
                H5P_DEFAULT),
            H5Aclose};
        check(H5Awrite(attribute.get(), memory_type, &value));
    }

    void write_string_dataset (
        hid_t location, char const* name, std::string const& value)
    {
        auto type = string_type();
        auto space = scalar_space();
        Hdf5Handle dataset{
            H5Dcreate2(
                location, name, type.get(), space.get(), H5P_DEFAULT,
                H5P_DEFAULT, H5P_DEFAULT),
            H5Dclose};
        char const* pointer = value.c_str();
        check(H5Dwrite(
            dataset.get(), type.get(), H5S_ALL, H5S_ALL,
            H5P_DEFAULT, &pointer));
    }

    Hdf5Handle write_double_dataset (
        hid_t location, char const* name, std::vector<hsize_t> const& shape,
        std::vector<double> const& values, std::string const& units)
    {
        Hdf5Handle space{
            H5Screate_simple(
                static_cast<int>(shape.size()), shape.data(), nullptr),
            H5Sclose};
        Hdf5Handle dataset{
            H5Dcreate2(
                location, name, H5T_IEEE_F64LE, space.get(), H5P_DEFAULT,
                H5P_DEFAULT, H5P_DEFAULT),
            H5Dclose};
        check(H5Dwrite(
            dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
            H5P_DEFAULT, values.data()));
        write_string_attribute(dataset.get(), "units", units);
        return dataset;
    }

    void write_int32_dataset (
        hid_t location, char const* name,
        std::vector<std::int32_t> const& values, std::string const& units)
    {
        std::array<hsize_t, 1> const shape{
            static_cast<hsize_t>(values.size())};
        Hdf5Handle space{H5Screate_simple(1, shape.data(), nullptr), H5Sclose};
        Hdf5Handle dataset{
            H5Dcreate2(
                location, name, H5T_STD_I32LE, space.get(), H5P_DEFAULT,
                H5P_DEFAULT, H5P_DEFAULT),
            H5Dclose};
        check(H5Dwrite(
            dataset.get(), H5T_NATIVE_INT32, H5S_ALL, H5S_ALL,
            H5P_DEFAULT, values.data()));
        write_string_attribute(dataset.get(), "units", units);
    }

    Hdf5Handle create_group (hid_t location, char const* name)
    {
        return {
            H5Gcreate2(
                location, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            H5Gclose};
    }

    enum class Variant
    {
        Valid,
        WithinToleranceEmission,
        MismatchedEmission,
        LegacyNoEmission,
        WrongVersion,
        WrongUnitSystem,
        WrongDatasetUnits,
        MaterialIdMismatch,
        ShapeMismatch,
        FiniteTail,
        WrongGroupIndex,
        MissingProvenance,
        SingletonDensity,
        SingletonTemperature,
        SingletonBoth,
        CollapsedDensityReal,
        CollapsedTemperatureReal,
        ChConversion
    };

    struct TableSpecification
    {
        std::int64_t material_id = 1001;
        std::string material_key = "manufactured-ch";
        std::string material_name = "manufactured CH";
        double opacity_scale = 1.0;
        double internal_group_edge = 2.0e-16;
        std::array<double, 2> representative_energies{1.0e-16, 4.0e-16};
        bool constant_mass_opacity = false;
    };

    void write_table (
        std::filesystem::path const& path, Variant variant,
        TableSpecification const& specification = {})
    {
        Hdf5Handle file{
            H5Fcreate(
                path.string().c_str(), H5F_ACC_TRUNC,
                H5P_DEFAULT, H5P_DEFAULT),
            H5Fclose};
        write_string_attribute(
            file.get(), "format", "org.warpx.opacity.multigroup");
        write_string_attribute(
            file.get(), "schema_version",
            variant == Variant::WrongVersion ? "1.1.0" : "1.0.0");
        write_string_attribute(
            file.get(), "unit_system",
            variant == Variant::WrongUnitSystem ? "CGS" : "SI");
        write_string_attribute(
            file.get(), "array_order", "rho,temperature,group");

        auto materials = create_group(file.get(), "materials");
        std::string const material_id =
            std::to_string(specification.material_id);
        auto material = create_group(materials.get(), material_id.c_str());
        write_integer_attribute<std::int64_t>(
            material.get(), "material_id",
            variant == Variant::MaterialIdMismatch
                ? specification.material_id + 1
                : specification.material_id,
            H5T_STD_I64LE, H5T_NATIVE_INT64);
        write_string_attribute(
            material.get(), "material_key", specification.material_key);
        write_string_attribute(
            material.get(), "material_name", specification.material_name);
        write_string_attribute(material.get(), "thermodynamic_regime", "LTE");
        write_string_attribute(material.get(), "composition_model", "fixed");
        write_string_attribute(
            material.get(), "profile", "complete-multigroup-v1");
        write_string_attribute(
            material.get(), "spectral_coverage", "complete_0_inf");

        auto composition = create_group(material.get(), "composition");
        write_int32_dataset(
            composition.get(), "atomic_number", {6, 1}, "1");
        write_double_dataset(
            composition.get(), "atomic_mass", {2},
            {1.994473472893265e-26, 1.673557692882144e-27}, "kg");
        write_double_dataset(
            composition.get(), "number_fraction", {2}, {0.5, 0.5}, "1");

        std::vector<double> const density = variant == Variant::ChConversion
            ? std::vector<double>{100.0}
            : variant == Variant::CollapsedDensityReal
            ? std::vector<double>{
                1.0e7, std::nextafter(
                    1.0e7, std::numeric_limits<double>::infinity())}
            : variant == Variant::SingletonDensity
                || variant == Variant::SingletonBoth
            ? std::vector<double>{10.0}
            : std::vector<double>{1.0, 100.0};
        std::vector<double> const temperature = variant == Variant::ChConversion
            ? std::vector<double>{1.0e7}
            : variant == Variant::CollapsedTemperatureReal
            ? std::vector<double>{
                1.0e7, std::nextafter(
                    1.0e7, std::numeric_limits<double>::infinity())}
            : variant == Variant::SingletonTemperature
                || variant == Variant::SingletonBoth
            ? std::vector<double>{1.0e4}
            : std::vector<double>{1.0e3, 1.0e5};
        auto axes = create_group(material.get(), "axes");
        write_double_dataset(
            axes.get(), "mass_density", {density.size()}, density,
            variant == Variant::WrongDatasetUnits ? "g cm-3" : "kg m-3");
        write_double_dataset(
            axes.get(), "electron_temperature", {temperature.size()},
            temperature, "K");

        auto groups = create_group(material.get(), "groups");
        write_string_attribute(
            groups.get(), "interval_convention", "[edge[g],edge[g+1])");
        write_string_attribute(groups.get(), "first_edge", "zero");
        write_string_attribute(
            groups.get(), "last_edge", "positive_infinity");
        write_double_dataset(
            groups.get(), "edges", {3},
            {0.0, specification.internal_group_edge,
             variant == Variant::FiniteTail
                 ? 8.0e-16 : std::numeric_limits<double>::infinity()},
            "J");
        write_double_dataset(
            groups.get(), "representative_energy", {2},
            {specification.representative_energies[0],
             specification.representative_energies[1]}, "J");
        write_int32_dataset(
            groups.get(), "index",
            variant == Variant::WrongGroupIndex
                ? std::vector<std::int32_t>{0, 2}
                : std::vector<std::int32_t>{0, 1},
            "1");

        auto opacity_parent = create_group(material.get(), "opacity");
        auto opacity = create_group(opacity_parent.get(), "group");
        auto make_values = [&] (double multiplier, double ch_value)
        {
            std::vector<double> values;
            for (double rho : density) {
                for (double temp : temperature) {
                    for (int group = 0; group < 2; ++group) {
                        double const group_scale = group == 0 ? 1.0 : 10.0;
                        values.push_back(variant == Variant::ChConversion
                            ? ch_value
                            : specification.constant_mass_opacity
                            ? specification.opacity_scale * multiplier
                                * group_scale
                            : specification.opacity_scale * multiplier
                                * group_scale * std::sqrt(rho)
                                * std::pow(temp, 0.25));
                    }
                }
            }
            return values;
        };
        std::vector<hsize_t> const table_shape{
            density.size(), temperature.size(), 2};
        std::vector<hsize_t> const shape = variant == Variant::ShapeMismatch
            ? std::vector<hsize_t>{2, 4, 1}
            : table_shape;

        auto planck = write_double_dataset(
            opacity.get(), "planck_absorption", shape,
            make_values(1.0, 0.0078496526),
            "m2 kg-1");
        write_string_attribute(planck.get(), "mean_definition", "planck");
        write_string_attribute(planck.get(), "interaction", "true_absorption");
        write_integer_attribute<int>(
            planck.get(), "includes_scattering", 0,
            H5T_STD_I32LE, H5T_NATIVE_INT);
        write_string_attribute(
            planck.get(), "stimulated_emission", "included");

        if (variant != Variant::LegacyNoEmission) {
            std::vector<double> emission_values =
                make_values(
                    variant == Variant::MismatchedEmission ? 3.0 : 1.0,
                    variant == Variant::MismatchedEmission
                        ? 0.015 : 0.0078496526);
            if (variant == Variant::WithinToleranceEmission) {
                for (double& value : emission_values) {
                    for (int ulp = 0; ulp < 32; ++ulp) {
                        value = std::nextafter(
                            value, std::numeric_limits<double>::infinity());
                    }
                }
            }
            auto emission = write_double_dataset(
                opacity.get(), "planck_emission", table_shape,
                emission_values, "m2 kg-1");
            write_string_attribute(
                emission.get(), "mean_definition", "planck");
            write_string_attribute(
                emission.get(), "interaction", "thermal_emission");
            write_integer_attribute<int>(
                emission.get(), "includes_scattering", 0,
                H5T_STD_I32LE, H5T_NATIVE_INT);
        }

        auto rosseland = write_double_dataset(
            opacity.get(), "rosseland_transport", table_shape,
            make_values(2.0, 0.0231390773), "m2 kg-1");
        write_string_attribute(
            rosseland.get(), "mean_definition", "rosseland");
        write_string_attribute(
            rosseland.get(), "interaction", "total_transport");
        write_integer_attribute<int>(
            rosseland.get(), "includes_scattering", 1,
            H5T_STD_I32LE, H5T_NATIVE_INT);
        write_string_attribute(
            rosseland.get(), "transport_correction", "1-g");

        auto scattering = write_double_dataset(
            opacity.get(), "planck_scattering", table_shape,
            make_values(0.5, 0.001), "m2 kg-1");
        write_string_attribute(
            scattering.get(), "mean_definition", "planck");
        write_string_attribute(
            scattering.get(), "interaction", "scattering");
        write_integer_attribute<int>(
            scattering.get(), "includes_scattering", 1,
            H5T_STD_I32LE, H5T_NATIVE_INT);

        auto provenance = create_group(material.get(), "provenance");
        for (auto const& [name, value] :
             std::vector<std::pair<std::string, std::string>>{
                 {"generator_name", "manufactured-test"},
                 {"generator_version", "1"},
                 {"generator_git_commit", "not-applicable"},
                 {"converter_name", "none"},
                 {"converter_version", "not-applicable"},
                 {"converter_git_commit", "not-applicable"},
                 {"license_spdx", "CC0-1.0"},
                 {"redistribution", "unrestricted"},
                 {"physics_model", "manufactured power law"},
                 {"created_utc", "2026-08-23T00:00:00Z"}})
        {
            write_string_dataset(provenance.get(), name.c_str(), value);
        }
        std::string const digest(64, 'a');
        write_string_dataset(
            provenance.get(), "generator_input_sha256", digest);
        if (variant != Variant::MissingProvenance) {
            write_string_dataset(
                provenance.get(), "source_output_sha256", digest);
        }
    }

    MaterialOpacityCoefficients evaluate (
        MaterialOpacityTable const& table, amrex::Real density,
        amrex::Real temperature, int group)
    {
        amrex::Gpu::DeviceVector<amrex::Real> result(4);
        amrex::Real* output = result.data();
        auto const executor = table.executor();
        amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) noexcept
        {
            auto const coefficients = executor(density, temperature, group);
            output[0] = coefficients.planck_absorption;
            output[1] = coefficients.planck_emission;
            output[2] = coefficients.rosseland_transport;
            output[3] = coefficients.scattering;
        });
        amrex::Gpu::streamSynchronize();
        std::array<amrex::Real, 4> host{};
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, result.begin(), result.end(), host.begin());
        return {host[0], host[1], host[2], host[3]};
    }

    bool close (amrex::Real actual, amrex::Real expected)
    {
        amrex::Real const tolerance = sizeof(amrex::Real) == sizeof(float)
            ? 2.0e-5_rt : 2.0e-12_rt;
        return std::abs(actual - expected)
            <= tolerance * std::abs(expected);
    }

    void require_test (bool condition, std::string const& message)
    {
        if (!condition) { throw std::runtime_error(message); }
    }

    void expect_rejected (
        std::filesystem::path const& path, Variant variant,
        std::string const& expected_message)
    {
        write_table(path, variant);
        try {
            MaterialOpacityTable table(path.string(), 1001);
        } catch (std::runtime_error const& error) {
            require_test(
                std::string(error.what()).find(expected_message)
                    != std::string::npos,
                "unexpected rejection message: " + std::string(error.what()));
            std::filesystem::remove(path);
            return;
        }
        throw std::runtime_error(
            "malformed table was accepted: " + path.string());
    }

    void check_singleton_table (
        std::filesystem::path const& path, Variant variant)
    {
        write_table(path, variant);
        MaterialOpacityTable table(path.string(), 1001);
        amrex::Real const query_density =
            variant == Variant::SingletonDensity
                || variant == Variant::SingletonBoth
            ? 20.0_rt : 10.0_rt;
        amrex::Real const query_temperature =
            variant == Variant::SingletonTemperature
                || variant == Variant::SingletonBoth
            ? 2.0e4_rt : 1.0e4_rt;
        amrex::Real const table_density = variant == Variant::SingletonDensity
                || variant == Variant::SingletonBoth
            ? 10.0_rt : query_density;
        amrex::Real const table_temperature =
            variant == Variant::SingletonTemperature
                || variant == Variant::SingletonBoth
            ? 1.0e4_rt : query_temperature;
        amrex::Real const expected = query_density * std::sqrt(table_density)
            * std::pow(table_temperature, 0.25_rt);
        auto const coefficients = evaluate(
            table, query_density, query_temperature, 0);
        require_test(
            close(coefficients.planck_absorption, expected),
            "singleton-axis interpolation is wrong");
        if (variant == Variant::SingletonDensity
            || variant == Variant::SingletonBoth)
        {
            auto const invalid = evaluate(
                table, std::numeric_limits<amrex::Real>::infinity(),
                query_temperature, 0);
            require_test(
                std::isnan(invalid.planck_absorption),
                "singleton density axis accepted an infinite query");
        }
        if (variant == Variant::SingletonTemperature
            || variant == Variant::SingletonBoth)
        {
            auto const invalid = evaluate(
                table, query_density,
                std::numeric_limits<amrex::Real>::infinity(), 0);
            require_test(
                std::isnan(invalid.planck_absorption),
                "singleton temperature axis accepted an infinite query");
        }
        std::filesystem::remove(path);
    }

    void check_ch_conversion (std::filesystem::path const& path)
    {
        write_table(path, Variant::ChConversion);
        MaterialOpacityTable table(path.string(), 1001);
        auto const coefficients = evaluate(table, 100.0_rt, 1.0e7_rt, 0);
        require_test(
            close(coefficients.planck_absorption, 0.78496526_rt)
                && close(coefficients.rosseland_transport, 2.31390773_rt),
            "CH SI mass-opacity to linear-coefficient conversion is wrong");
        std::filesystem::remove(path);
    }

    void run_test ()
    {
        amrex::Real const absorption = 2.0_rt;
        amrex::Real const emission = 6.0_rt;
        amrex::Real const old_energy = 5.0_rt;
        amrex::Real const blackbody_energy = 4.0_rt;
        amrex::Real const optical_time = 0.4_rt;
        amrex::Real const dt =
            optical_time / (absorption * PhysConst::c);
        auto const independent_exchange = EvaluatePlanckExchange(
            absorption, emission, old_energy, blackbody_energy, dt);
        amrex::Real const expected_independent =
            (emission / absorption * blackbody_energy - old_energy)
            * -std::expm1(-optical_time);
        require_test(
            independent_exchange.valid
                && close(
                    independent_exchange.exchange_energy,
                    expected_independent),
            "independent Planck emission/absorption update is wrong");
        auto const zero_absorption = EvaluatePlanckExchange(
            0.0_rt, emission, old_energy, blackbody_energy, dt);
        require_test(
            zero_absorption.valid
                && close(
                    zero_absorption.exchange_energy,
                    emission * PhysConst::c * dt * blackbody_energy),
            "zero-absorption Planck emission limit is wrong");
        require_test(
            !EvaluatePlanckExchange(
                -1.0_rt, emission, old_energy, blackbody_energy, dt).valid,
            "negative Planck coefficient was accepted");

        auto const directory = std::filesystem::current_path();
        auto const valid_path = directory / "material_opacity_valid.h5";
        write_table(valid_path, Variant::Valid);
        MaterialOpacityTable table(valid_path.string(), 1001);
        require_test(table.initialized(), "valid table was not initialized");
        require_test(table.materialId() == 1001, "wrong selected material ID");
        require_test(
            table.materialKey() == "manufactured-ch", "wrong material key");
        require_test(
            table.groupEdges().size() == 3U
                && table.groupEdges().front() == 0.0_rt
                && std::isinf(table.groupEdges().back()),
            "group edges were not retained");

        amrex::Real const rho = 10.0_rt;
        amrex::Real const temperature = 1.0e4_rt;
        amrex::Real const base =
            rho * std::sqrt(rho) * std::pow(temperature, 0.25_rt);
        auto const group_zero = evaluate(table, rho, temperature, 0);
        auto const group_one = evaluate(table, rho, temperature, 1);
        require_test(
            close(group_zero.planck_absorption, base)
                && group_zero.planck_emission == group_zero.planck_absorption
                && close(group_zero.rosseland_transport, 2.0_rt * base)
                && close(group_zero.scattering, 0.5_rt * base),
            "log-log interpolation or rho*kappa conversion is wrong");
        require_test(
            close(group_one.planck_absorption, 10.0_rt * base)
                && group_one.planck_emission == group_one.planck_absorption,
            "integer photon-group selection is wrong");

        auto const vacuum = evaluate(table, 0.0_rt, -1.0_rt, -1);
        require_test(
            vacuum.planck_absorption == 0.0_rt
                && vacuum.planck_emission == 0.0_rt
                && vacuum.rosseland_transport == 0.0_rt
                && vacuum.scattering == 0.0_rt,
            "zero density must return four exact zeros");
        auto const out_of_range = evaluate(table, 0.5_rt, temperature, 0);
        require_test(
            std::isnan(out_of_range.planck_absorption)
                && std::isnan(out_of_range.planck_emission)
                && std::isnan(out_of_range.rosseland_transport)
                && std::isnan(out_of_range.scattering),
            "out-of-domain query must not silently extrapolate");
        std::filesystem::remove(valid_path);

        auto const legacy_path = directory / "material_opacity_legacy.h5";
        write_table(legacy_path, Variant::LegacyNoEmission);
        MaterialOpacityTable legacy(legacy_path.string(), 1001);
        auto const legacy_coefficients = evaluate(
            legacy, rho, temperature, 0);
        require_test(
            legacy_coefficients.planck_emission
                == legacy_coefficients.planck_absorption,
            "legacy table did not alias Planck emission to absorption");
        std::filesystem::remove(legacy_path);

        auto const within_tolerance_path =
            directory / "material_opacity_within_tolerance_emission.h5";
        write_table(within_tolerance_path, Variant::WithinToleranceEmission);
        MaterialOpacityTable within_tolerance(
            within_tolerance_path.string(), 1001);
        auto const within_tolerance_coefficients = evaluate(
            within_tolerance, rho, temperature, 0);
        require_test(
            within_tolerance_coefficients.planck_emission
                == within_tolerance_coefficients.planck_absorption,
            "accepted LTE emission was not canonicalized to absorption");
        std::filesystem::remove(within_tolerance_path);

        auto const reload_bad_path =
            directory / "material_opacity_reload_bad.h5";
        write_table(reload_bad_path, Variant::MismatchedEmission);
        auto const original_group_zero = group_zero;
        auto const original_group_one = group_one;
        bool reload_rejected = false;
        try {
            table.load(reload_bad_path.string(), 1001);
        } catch (std::runtime_error const& error) {
            reload_rejected = true;
            require_test(
                std::string(error.what()).find(
                    "planck_emission must satisfy LTE detailed balance")
                    != std::string::npos,
                "reload rejection did not report LTE detailed balance");
        }
        require_test(
            reload_rejected, "bad reload table was accepted");
        require_test(
            table.initialized() && table.materialId() == 1001
                && table.materialKey() == "manufactured-ch"
                && table.groupEdges().size() == 3U,
            "failed reload changed initialized table metadata");
        auto const after_failed_reload_zero = evaluate(
            table, rho, temperature, 0);
        auto const after_failed_reload_one = evaluate(
            table, rho, temperature, 1);
        require_test(
            after_failed_reload_zero.planck_absorption
                    == original_group_zero.planck_absorption
                && after_failed_reload_zero.planck_emission
                    == original_group_zero.planck_emission
                && after_failed_reload_zero.rosseland_transport
                    == original_group_zero.rosseland_transport
                && after_failed_reload_zero.scattering
                    == original_group_zero.scattering
                && after_failed_reload_one.planck_absorption
                    == original_group_one.planck_absorption
                && after_failed_reload_one.planck_emission
                    == original_group_one.planck_emission
                && after_failed_reload_one.rosseland_transport
                    == original_group_one.rosseland_transport
                && after_failed_reload_one.scattering
                    == original_group_one.scattering,
            "failed reload changed evaluated coefficients");
        std::filesystem::remove(reload_bad_path);

        check_singleton_table(
            directory / "material_opacity_singleton_density.h5",
            Variant::SingletonDensity);
        check_singleton_table(
            directory / "material_opacity_singleton_temperature.h5",
            Variant::SingletonTemperature);
        check_singleton_table(
            directory / "material_opacity_singleton_both.h5",
            Variant::SingletonBoth);
        check_ch_conversion(directory / "material_opacity_ch_conversion.h5");

        expect_rejected(
            directory / "material_opacity_wrong_version.h5",
            Variant::WrongVersion, "schema_version");
        expect_rejected(
            directory / "material_opacity_mismatched_emission.h5",
            Variant::MismatchedEmission,
            "planck_emission must satisfy LTE detailed balance");
        expect_rejected(
            directory / "material_opacity_wrong_system.h5",
            Variant::WrongUnitSystem, "unit_system");
        expect_rejected(
            directory / "material_opacity_wrong_units.h5",
            Variant::WrongDatasetUnits, "kg m-3");
        expect_rejected(
            directory / "material_opacity_wrong_material.h5",
            Variant::MaterialIdMismatch, "material_id");
        expect_rejected(
            directory / "material_opacity_wrong_shape.h5",
            Variant::ShapeMismatch, "opacity array shape");
        expect_rejected(
            directory / "material_opacity_finite_tail.h5",
            Variant::FiniteTail, "positive infinity");
        expect_rejected(
            directory / "material_opacity_wrong_group_index.h5",
            Variant::WrongGroupIndex, "group indices");
        expect_rejected(
            directory / "material_opacity_missing_provenance.h5",
            Variant::MissingProvenance, "source_output_sha256");
        if constexpr (sizeof(amrex::Real) == sizeof(float)) {
            expect_rejected(
                directory / "material_opacity_collapsed_density.h5",
                Variant::CollapsedDensityReal,
                "axis 'mass_density' is not strictly increasing");
            expect_rejected(
                directory / "material_opacity_collapsed_temperature.h5",
                Variant::CollapsedTemperatureReal,
                "axis 'electron_temperature' is not strictly increasing");
        }
    }

    void write_runtime_fixtures (std::filesystem::path const& directory)
    {
        std::filesystem::create_directories(directory);
        TableSpecification const material_a{
            1001, "material-a", "manufactured material A",
            3.162277660168379e-2, 2.0e-16,
            {1.0e-16, 4.0e-16}, true};
        TableSpecification const material_b{
            1001, "material-b", "manufactured material B",
            6.324555320336758e-2, 2.0e-16,
            {1.0e-16, 4.0e-16}, true};
        write_table(
            directory / "material_a.h5", Variant::Valid, material_a);
        write_table(
            directory / "material_b.h5", Variant::Valid, material_b);

        auto material_lte_a = material_a;
        material_lte_a.internal_group_edge = 2.0e-19;
        material_lte_a.representative_energies = {1.0e-19, 4.0e-19};
        auto material_lte_b = material_b;
        material_lte_b.internal_group_edge = 2.0e-19;
        material_lte_b.representative_energies = {1.0e-19, 4.0e-19};
        write_table(
            directory / "material_lte_a.h5",
            Variant::WithinToleranceEmission, material_lte_a);
        write_table(
            directory / "material_lte_b.h5",
            Variant::WithinToleranceEmission, material_lte_b);

        auto group_mismatch = material_b;
        group_mismatch.internal_group_edge = 3.0e-16;
        write_table(
            directory / "material_b_group_mismatch.h5", Variant::Valid,
            group_mismatch);
        auto key_mismatch = material_b;
        key_mismatch.material_key = "unexpected-material-b";
        write_table(
            directory / "material_b_key_mismatch.h5", Variant::Valid,
            key_mismatch);
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 0;
    try {
        if (argc == 3 && std::string(argv[1]) == "--write-runtime-fixtures") {
            write_runtime_fixtures(argv[2]);
            std::cout << "Material-opacity runtime fixtures written\n";
        } else {
            run_test();
            std::cout << "MaterialOpacityTable manufactured test passed\n";
        }
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    amrex::Finalize();
    return status;
}
