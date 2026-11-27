/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronThermodynamics.H"
#include "Radiation/FrozenPhotonCollisionRate.H"
#include "Radiation/ImplicitDiffusion.H"
#include "Radiation/ImplicitMaterialTableCoefficients.H"
#include "Radiation/MaterialKineticWork.H"
#include "Radiation/MaterialOpacityTable.H"
#include "Radiation/MaterialOpacityTableResponse.H"
#include "Radiation/OpacityEvaluator.H"
#include "Radiation/PlanckExchange.H"
#include "Radiation/RadiationEnergyUpdate.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <hdf5.h>

#include <array>
#include <chrono>
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
    using warpx::radiation::EnergyGroupsExecutor;
    using warpx::radiation::EvaluateFrozenPhotonCollisionRate;
    using warpx::radiation::EvaluateMaterialImpulseCandidate;
    using warpx::radiation::EvaluatePlanckExchange;
    using warpx::radiation::MaterialOpacityCoefficients;
    using warpx::radiation::MaterialOpacityTable;
    using warpx::radiation::MaterialOpacityTableExecutor;
    using warpx::radiation::MaterialOpacityTableResponse;
    using warpx::radiation::EvaluateMaterialOpacityTableResponse;

    constexpr std::int64_t material_id = 1001;
    constexpr int num_groups = 3;

    class Hdf5Handle
    {
    public:
        using Closer = herr_t (*) (hid_t);

        Hdf5Handle (hid_t id, Closer closer) : m_id(id), m_closer(closer)
        {
            if (id < 0) { throw std::runtime_error("HDF5 operation failed"); }
        }
        ~Hdf5Handle ()
        {
            if (m_id >= 0 && m_closer != nullptr) { m_closer(m_id); }
        }
        Hdf5Handle (Hdf5Handle const&) = delete;
        Hdf5Handle& operator= (Hdf5Handle const&) = delete;
        Hdf5Handle& operator= (Hdf5Handle&&) = delete;
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
                location, name, type.get(), space.get(),
                H5P_DEFAULT, H5P_DEFAULT),
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
                location, name, file_type, space.get(),
                H5P_DEFAULT, H5P_DEFAULT),
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
                location, name, type.get(), space.get(),
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            H5Dclose};
        char const* pointer = value.c_str();
        check(H5Dwrite(
            dataset.get(), type.get(), H5S_ALL, H5S_ALL,
            H5P_DEFAULT, &pointer));
    }

    Hdf5Handle write_double_dataset (
        hid_t location, char const* name,
        std::vector<hsize_t> const& shape,
        std::vector<double> const& values, std::string const& units)
    {
        Hdf5Handle space{
            H5Screate_simple(
                static_cast<int>(shape.size()), shape.data(), nullptr),
            H5Sclose};
        Hdf5Handle dataset{
            H5Dcreate2(
                location, name, H5T_IEEE_F64LE, space.get(),
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
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
                location, name, H5T_STD_I32LE, space.get(),
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            H5Dclose};
        check(H5Dwrite(
            dataset.get(), H5T_NATIVE_INT32, H5S_ALL, H5S_ALL,
            H5P_DEFAULT, values.data()));
        write_string_attribute(dataset.get(), "units", units);
    }

    void write_u8_dataset (
        hid_t location, char const* name,
        std::vector<hsize_t> const& shape,
        std::vector<std::uint8_t> const& values, std::string const& units)
    {
        Hdf5Handle space{
            H5Screate_simple(
                static_cast<int>(shape.size()), shape.data(), nullptr),
            H5Sclose};
        Hdf5Handle dataset{
            H5Dcreate2(
                location, name, H5T_STD_U8LE, space.get(),
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            H5Dclose};
        check(H5Dwrite(
            dataset.get(), H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL,
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

    struct Channel
    {
        std::vector<double> values;
        std::vector<std::uint8_t> mask;
    };

    struct TableSpec
    {
        std::vector<double> density{1.0, 100.0};
        std::vector<double> temperature{1000.0, 10000.0};
        std::vector<double> edges{0.0, 1.0e-18, 2.0e-18,
                                  std::numeric_limits<double>::infinity()};
        std::vector<double> representative{5.0e-19, 1.5e-18, 4.0e-18};
        Channel absorption;
        Channel emission;
        Channel rosseland;
        Channel scattering;

        [[nodiscard]] std::size_t size () const
        {
            return density.size() * temperature.size() * representative.size();
        }
    };

    enum class Corruption
    {
        None,
        Negative,
        NotANumber,
        Infinity,
        MaskMismatch,
        MissingMask,
        InvalidMaskValue,
        EmissionMismatch,
        LegacyZero
    };

    std::size_t index (TableSpec const& spec, int i, int j, int group)
    {
        return (static_cast<std::size_t>(i) * spec.temperature.size()
                + static_cast<std::size_t>(j))
            * spec.representative.size() + static_cast<std::size_t>(group);
    }

    Channel make_channel (TableSpec const& spec, bool zero_scattering_only)
    {
        Channel result;
        result.values.resize(spec.size());
        result.mask.resize(spec.size());
        for (std::size_t i = 0; i < spec.density.size(); ++i) {
            for (std::size_t j = 0; j < spec.temperature.size(); ++j) {
                for (int group = 0; group < num_groups; ++group) {
                    auto const k = index(
                        spec, static_cast<int>(i), static_cast<int>(j), group);
                    double value = 1.0 + 0.25 * i + 0.5 * j + group;
                    if (group == 0) {
                        value = 0.0;
                    } else if (group == 1 && i == 0 && j == 0) {
                        value = 0.0;
                    } else if (zero_scattering_only) {
                        value = group == 2 ? 0.0 : value;
                    }
                    result.values[k] = value;
                    result.mask[k] = value == 0.0 ? 1U : 0U;
                }
            }
        }
        return result;
    }

    TableSpec make_spec ()
    {
        TableSpec spec;
        spec.absorption = make_channel(spec, false);
        spec.emission = spec.absorption;
        spec.rosseland = make_channel(spec, false);
        spec.scattering = make_channel(spec, true);
        return spec;
    }

    void apply_corruption (
        TableSpec& spec, Corruption corruption, bool zero_schema)
    {
        if (corruption == Corruption::None) { return; }
        auto const k = index(spec, 0, 0, 1);
        if (corruption == Corruption::Negative) {
            spec.absorption.values[k] = -1.0;
            spec.absorption.mask[k] = 0;
            spec.emission.values[k] = -1.0;
            spec.emission.mask[k] = 0;
        } else if (corruption == Corruption::NotANumber) {
            spec.absorption.values[k] =
                std::numeric_limits<double>::quiet_NaN();
            spec.absorption.mask[k] = 0;
            spec.emission.values[k] =
                std::numeric_limits<double>::quiet_NaN();
            spec.emission.mask[k] = 0;
        } else if (corruption == Corruption::Infinity) {
            spec.absorption.values[k] =
                std::numeric_limits<double>::infinity();
            spec.absorption.mask[k] = 0;
            spec.emission.values[k] =
                std::numeric_limits<double>::infinity();
            spec.emission.mask[k] = 0;
        } else if (corruption == Corruption::MaskMismatch) {
            spec.absorption.values[k] = 2.0;
            spec.absorption.mask[k] = 1;
        } else if (corruption == Corruption::InvalidMaskValue) {
            spec.absorption.mask[k] = 2;
        } else if (corruption == Corruption::EmissionMismatch) {
            spec.emission.values[k] = 2.0;
            spec.emission.mask[k] = 0;
        } else if (corruption == Corruption::LegacyZero || !zero_schema) {
            spec.absorption.values[k] = 0.0;
            spec.absorption.mask[k] = 1;
            spec.emission.values[k] = 0.0;
            spec.emission.mask[k] = 1;
        }
    }

    void write_channel (
        hid_t location, char const* name, Channel const& channel,
        std::vector<hsize_t> const& shape, bool zero_schema,
        bool write_mask, char const* interaction,
        int includes_scattering, char const* mean_definition)
    {
        auto dataset = write_double_dataset(
            location, name, shape, channel.values, "m2 kg-1");
        write_string_attribute(dataset.get(), "mean_definition", mean_definition);
        write_string_attribute(dataset.get(), "interaction", interaction);
        write_integer_attribute<int>(
            dataset.get(), "includes_scattering", includes_scattering,
            H5T_STD_I32LE, H5T_NATIVE_INT);
        if (std::string(name) == "planck_absorption") {
            write_string_attribute(
                dataset.get(), "stimulated_emission", "included");
        }
        if (std::string(name) == "rosseland_transport") {
            write_string_attribute(
                dataset.get(), "transport_correction", "1-g");
        }
        if (zero_schema) {
            write_string_attribute(
                dataset.get(), "value_encoding", "linear_mass_opacity");
            write_string_attribute(
                dataset.get(), "zero_interpolation",
                "bilinear_linear_exact_masked_nodes");
        }
        if (write_mask) {
            write_u8_dataset(
                location, (std::string(name) + "_zero_mask").c_str(),
                shape, channel.mask, "1");
        }
    }

    void write_table (
        std::filesystem::path const& path, TableSpec spec,
        Corruption corruption = Corruption::None,
        bool zero_schema = true)
    {
        apply_corruption(spec, corruption, zero_schema);
        Hdf5Handle file{
            H5Fcreate(
                path.string().c_str(), H5F_ACC_TRUNC,
                H5P_DEFAULT, H5P_DEFAULT),
            H5Fclose};
        write_string_attribute(
            file.get(), "format", "org.warpx.opacity.multigroup");
        write_string_attribute(
            file.get(), "schema_version", zero_schema ? "1.1.0" : "1.0.0");
        if (zero_schema) {
            write_string_attribute(
                file.get(), "zero_policy",
                "explicit_uint8_mask_exact_zero_linear");
        }
        write_string_attribute(file.get(), "unit_system", "SI");
        write_string_attribute(
            file.get(), "array_order", "rho,temperature,group");

        auto materials = create_group(file.get(), "materials");
        auto material = create_group(
            materials.get(), std::to_string(material_id).c_str());
        write_integer_attribute<std::int64_t>(
            material.get(), "material_id", material_id,
            H5T_STD_I64LE, H5T_NATIVE_INT64);
        write_string_attribute(material.get(), "material_key", "zero-ch");
        write_string_attribute(material.get(), "material_name", "zero CH");
        write_string_attribute(material.get(), "thermodynamic_regime", "LTE");
        write_string_attribute(material.get(), "composition_model", "fixed");
        write_string_attribute(
            material.get(), "profile", "complete-multigroup-v1");
        write_string_attribute(
            material.get(), "spectral_coverage", "complete_0_inf");

        auto composition = create_group(material.get(), "composition");
        write_int32_dataset(
            composition.get(), "atomic_number", {6}, "1");
        write_double_dataset(
            composition.get(), "atomic_mass", {1}, {1.2e-26}, "kg");
        write_double_dataset(
            composition.get(), "number_fraction", {1}, {1.0}, "1");

        auto axes = create_group(material.get(), "axes");
        write_double_dataset(
            axes.get(), "mass_density", {spec.density.size()},
            spec.density, "kg m-3");
        write_double_dataset(
            axes.get(), "electron_temperature", {spec.temperature.size()},
            spec.temperature, "K");

        auto groups = create_group(material.get(), "groups");
        write_string_attribute(
            groups.get(), "interval_convention", "[edge[g],edge[g+1])");
        write_string_attribute(groups.get(), "first_edge", "zero");
        write_string_attribute(
            groups.get(), "last_edge", "positive_infinity");
        write_double_dataset(
            groups.get(), "edges", {spec.edges.size()}, spec.edges, "J");
        write_double_dataset(
            groups.get(), "representative_energy",
            {spec.representative.size()}, spec.representative, "J");
        write_int32_dataset(
            groups.get(), "index", {0, 1, 2}, "1");

        auto opacity_parent = create_group(material.get(), "opacity");
        auto opacity = create_group(opacity_parent.get(), "group");
        std::vector<hsize_t> const shape{
            spec.density.size(), spec.temperature.size(),
            spec.representative.size()};
        bool const write_abs_mask =
            zero_schema && corruption != Corruption::MissingMask;
        write_channel(
            opacity.get(), "planck_absorption", spec.absorption, shape,
            zero_schema, write_abs_mask, "true_absorption", 0, "planck");
        write_channel(
            opacity.get(), "planck_emission", spec.emission, shape,
            zero_schema, zero_schema, "thermal_emission", 0, "planck");
        write_channel(
            opacity.get(), "rosseland_transport", spec.rosseland, shape,
            zero_schema, zero_schema, "total_transport", 1, "rosseland");
        write_channel(
            opacity.get(), "planck_scattering", spec.scattering, shape,
            zero_schema, zero_schema, "scattering", 1, "planck");

        auto provenance = create_group(material.get(), "provenance");
        for (auto const& [name, value] :
             std::vector<std::pair<std::string, std::string>>{
                 {"generator_name", "zero-opacity-test"},
                 {"generator_version", "1"},
                 {"generator_git_commit", "not-applicable"},
                 {"converter_name", "none"},
                 {"converter_version", "not-applicable"},
                 {"converter_git_commit", "not-applicable"},
                 {"license_spdx", "CC0-1.0"},
                 {"redistribution", "unrestricted"},
                 {"physics_model", "manufactured exact-zero channels"},
                 {"created_utc", "2026-09-23T00:00:00Z"}})
        {
            write_string_dataset(provenance.get(), name.c_str(), value);
        }
        std::string const digest(64, 'a');
        write_string_dataset(
            provenance.get(), "generator_input_sha256", digest);
        write_string_dataset(
            provenance.get(), "source_output_sha256", digest);
    }

    void require (bool condition, std::string const& message)
    {
        if (!condition) { throw std::runtime_error(message); }
    }

    MaterialOpacityCoefficients evaluate_host (
        MaterialOpacityTable const& table,
        amrex::Real density, amrex::Real temperature, int group)
    {
        return table.hostExecutor()(density, temperature, group);
    }

    MaterialOpacityTableResponse evaluate_response (
        MaterialOpacityTable const& table,
        amrex::Real density, amrex::Real temperature,
        int group, amrex::Real fraction)
    {
        return EvaluateMaterialOpacityTableResponse(
            table.hostExecutor(), density, temperature, group, fraction);
    }

    amrex::Real linear_interpolation_weight (
        std::vector<double> const& axis, amrex::Real value, int& lower)
    {
        if (axis.size() == 1) {
            lower = 0;
            return 0.0_rt;
        }
        amrex::Real const x = std::log(value);
        if (x <= std::log(axis.front())) {
            lower = 0;
            return 0.0_rt;
        }
        if (x >= std::log(axis.back())) {
            lower = static_cast<int>(axis.size()) - 2;
            return 1.0_rt;
        }
        for (std::size_t i = 0; i + 1 < axis.size(); ++i) {
            amrex::Real const lo = std::log(axis[i]);
            amrex::Real const hi = std::log(axis[i + 1]);
            if (x >= lo && x <= hi) {
                lower = static_cast<int>(i);
                return (x - lo) / (hi - lo);
            }
        }
        lower = 0;
        return 0.0_rt;
    }

    amrex::Real expected_mass_opacity (
        TableSpec const& spec, Channel const& channel,
        amrex::Real density, amrex::Real temperature, int group)
    {
        int i = 0;
        int j = 0;
        amrex::Real const tx = linear_interpolation_weight(
            spec.density, density, i);
        amrex::Real const ty = linear_interpolation_weight(
            spec.temperature, temperature, j);
        int const i1 = amrex::min(i + 1, static_cast<int>(spec.density.size()) - 1);
        int const j1 = amrex::min(
            j + 1, static_cast<int>(spec.temperature.size()) - 1);
        auto const value = [&] (int ii, int jj)
        {
            auto const k = index(spec, ii, jj, group);
            return channel.mask[k] == 0U
                ? static_cast<amrex::Real>(channel.values[k]) : 0.0_rt;
        };
        amrex::Real const v0 =
            (1.0_rt - tx) * value(i, j) + tx * value(i1, j);
        amrex::Real const v1 =
            (1.0_rt - tx) * value(i, j1) + tx * value(i1, j1);
        return (1.0_rt - ty) * v0 + ty * v1;
    }

    bool close (amrex::Real actual, amrex::Real expected)
    {
        if (expected == 0.0_rt) { return actual == 0.0_rt; }
        return std::abs(actual - expected)
            <= 2.0e-11_rt * std::abs(expected);
    }

    void check_coefficients (
        MaterialOpacityCoefficients const& actual,
        MaterialOpacityCoefficients const& expected,
        std::string const& label)
    {
        require(
            close(actual.planck_absorption, expected.planck_absorption)
                && close(actual.planck_emission, expected.planck_emission)
                && close(actual.rosseland_transport,
                         expected.rosseland_transport)
                && close(actual.scattering, expected.scattering),
            label);
    }

    MaterialOpacityCoefficients expected_coefficients (
        TableSpec const& spec, amrex::Real density,
        amrex::Real temperature, int group)
    {
        return {
            density * expected_mass_opacity(
                spec, spec.absorption, density, temperature, group),
            density * expected_mass_opacity(
                spec, spec.emission, density, temperature, group),
            density * expected_mass_opacity(
                spec, spec.rosseland, density, temperature, group),
            density * expected_mass_opacity(
                spec, spec.scattering, density, temperature, group)};
    }

    std::filesystem::path make_directory ()
    {
        auto const stamp = std::chrono::steady_clock::now()
            .time_since_epoch().count();
        auto directory = std::filesystem::temp_directory_path()
            / ("warpx-zero-opacity-" + std::to_string(stamp));
        std::filesystem::create_directories(directory);
        return directory;
    }

    void expect_rejected (
        std::filesystem::path const& path, Corruption corruption,
        std::string const& expected)
    {
        write_table(path, make_spec(), corruption);
        try {
            MaterialOpacityTable table(path.string(), material_id);
        } catch (std::runtime_error const& error) {
            require(
                std::string(error.what()).find(expected) != std::string::npos,
                "unexpected rejection: " + std::string(error.what()));
            return;
        }
        throw std::runtime_error("invalid table was accepted: " + path.string());
    }

    void run_contract ()
    {
        auto const directory = make_directory();
        auto const path = directory / "zero.h5";
        auto const spec = make_spec();
        write_table(path, spec);
        MaterialOpacityTable table(path.string(), material_id);

        check_coefficients(
            evaluate_host(table, 100.0_rt, 10000.0_rt, 0),
            expected_coefficients(spec, 100.0_rt, 10000.0_rt, 0),
            "transparent group interpolation");
        check_coefficients(
            evaluate_host(table, 1.0_rt, 1000.0_rt, 1),
            expected_coefficients(spec, 1.0_rt, 1000.0_rt, 1),
            "zero/positive endpoint interpolation");
        check_coefficients(
            evaluate_host(table, 10.0_rt, 3162.277660168379_rt, 1),
            expected_coefficients(
                spec, 10.0_rt, 3162.277660168379_rt, 1),
            "zero/positive interior interpolation");
        check_coefficients(
            evaluate_host(table, 100.0_rt, 10000.0_rt, 2),
            expected_coefficients(spec, 100.0_rt, 10000.0_rt, 2),
            "zero scattering interpolation");

        auto const transparent = evaluate_response(
            table, 10.0_rt, 3000.0_rt, 0, 1.0_rt);
        require(
            transparent.valid
                && transparent.transport_opacity == 0.0_rt
                && transparent.absorption_rate == 0.0_rt
                && transparent.emission_rate_density == 0.0_rt
                && transparent.scattering_opacity == 0.0_rt,
            "wholly transparent group was not exact zero");
        auto const vacuum = evaluate_response(
            table, 0.0_rt, 3000.0_rt, 2, 1.0_rt);
        require(
            vacuum.valid
                && vacuum.transport_opacity == 0.0_rt
                && vacuum.absorption_rate == 0.0_rt
                && vacuum.emission_rate_density == 0.0_rt
                && vacuum.scattering_opacity == 0.0_rt,
            "vacuum response was not exact zero");

        auto const out_of_range = evaluate_response(
            table, 200.0_rt, 3000.0_rt, 1, 1.0_rt);
        require(!out_of_range.valid, "out-of-range query was accepted");

        auto singleton_path = directory / "singleton.h5";
        TableSpec singleton_spec = make_spec();
        singleton_spec.density = {10.0};
        singleton_spec.temperature = {3000.0};
        singleton_spec.absorption.values.assign(singleton_spec.size(), 0.0);
        singleton_spec.absorption.mask.assign(singleton_spec.size(), 1);
        singleton_spec.emission = singleton_spec.absorption;
        write_table(singleton_path, singleton_spec);
        MaterialOpacityTable singleton_table(
            singleton_path.string(), material_id);
        auto const singleton_coefficients =
            evaluate_host(singleton_table, 20.0_rt, 5000.0_rt, 1);
        require(
            singleton_coefficients.planck_absorption == 0.0_rt
                && singleton_coefficients.planck_emission == 0.0_rt,
            "singleton axes did not preserve exact zero");

        expect_rejected(
            directory / "negative.h5", Corruption::Negative,
            "finite, non-negative");
        expect_rejected(
            directory / "nan.h5", Corruption::NotANumber,
            "finite, non-negative");
        expect_rejected(
            directory / "inf.h5", Corruption::Infinity,
            "finite, non-negative");
        expect_rejected(
            directory / "mask-mismatch.h5", Corruption::MaskMismatch,
            "zero mask disagree");
        expect_rejected(
            directory / "missing-mask.h5", Corruption::MissingMask,
            "required for schema 1.1");
        expect_rejected(
            directory / "bad-mask.h5", Corruption::InvalidMaskValue,
            "only zero or one");
        expect_rejected(
            directory / "emission-mismatch.h5", Corruption::EmissionMismatch,
            "detailed balance");

        auto legacy_path = directory / "legacy-zero.h5";
        write_table(
            legacy_path, make_spec(), Corruption::LegacyZero, false);
        try {
            MaterialOpacityTable legacy(legacy_path.string(), material_id);
        } catch (std::runtime_error const& error) {
            require(
                std::string(error.what()).find("strictly positive")
                    != std::string::npos,
                "legacy zero rejection message changed");
        }
    }

    void run_reload ()
    {
        auto const directory = make_directory();
        auto const valid = directory / "valid.h5";
        auto const invalid = directory / "invalid.h5";
        auto const spec = make_spec();
        write_table(valid, spec);
        write_table(invalid, spec, Corruption::Negative);
        MaterialOpacityTable table(valid.string(), material_id);
        auto const before = evaluate_host(table, 10.0_rt, 3000.0_rt, 1);
        bool rejected = false;
        try {
            table.load(invalid.string(), material_id);
        } catch (std::runtime_error const&) {
            rejected = true;
        }
        require(rejected, "invalid reload was accepted");
        require(table.initialized(), "invalid reload cleared the live table");
        check_coefficients(
            evaluate_host(table, 10.0_rt, 3000.0_rt, 1), before,
            "invalid reload mutated the live table");
    }

    void run_consumers ()
    {
        auto const directory = make_directory();
        auto const path = directory / "consumer.h5";
        auto const spec = make_spec();
        write_table(path, spec);
        MaterialOpacityTable table(path.string(), material_id);

        std::vector<amrex::Real> host_boundaries{1.0e-18_rt, 2.0e-18_rt};
        std::vector<amrex::Real> host_representatives{
            5.0e-19_rt, 1.5e-18_rt, 4.0e-18_rt};
        EnergyGroupsExecutor const host_groups{
            host_boundaries.data(), host_representatives.data(), num_groups};
        amrex::Gpu::DeviceVector<amrex::Real> device_boundaries(
            host_boundaries.size());
        amrex::Gpu::DeviceVector<amrex::Real> device_representatives(
            host_representatives.size());
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, host_boundaries.begin(),
            host_boundaries.end(), device_boundaries.begin());
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, host_representatives.begin(),
            host_representatives.end(), device_representatives.begin());
        EnergyGroupsExecutor const device_groups{
            device_boundaries.data(), device_representatives.data(),
            num_groups};

        warpx::radiation::detail::MaterialOpacityEvaluator<1> evaluator;
        evaluator.enabled = true;
        evaluator.num_species = 1;
        evaluator.species_masses = {1.0_rt};
        evaluator.table = table.hostExecutor();
        evaluator.energy_groups = host_groups;
        amrex::GpuArray<amrex::Real, 1> const number_densities{10.0_rt};
        auto const transparent =
            evaluator(number_densities, 5.0e-19_rt, 3000.0_rt);
        require(
            transparent.planck_absorption == 0.0_rt
                && transparent.planck_emission == 0.0_rt
                && transparent.rosseland_transport == 0.0_rt
                && transparent.scattering == 0.0_rt,
            "MaterialOpacityEvaluator created transparent-group interaction");

        warpx::radiation::detail::RegisteredMaterialOpacityEvaluator<1, 1>
            registered;
        registered.enabled = true;
        registered.num_species = 1;
        registered.num_materials = 1;
        registered.species_masses = {1.0_rt};
        registered.species_material_indices = {0};
        registered.tables[0] = table.hostExecutor();
        registered.selector = {1, 0.0_rt, 0.0_rt, 0.0_rt};
        registered.energy_groups = host_groups;
        auto const registered_zero =
            registered(number_densities, 5.0e-19_rt, 3000.0_rt);
        require(
            registered_zero.planck_absorption == 0.0_rt
                && registered_zero.scattering == 0.0_rt,
            "RegisteredMaterialOpacityEvaluator created an interaction");

        auto const response = EvaluateMaterialOpacityTableResponse(
            table.hostExecutor(), 10.0_rt, 3000.0_rt, 0, 1.0_rt);
        require(
            response.valid && response.absorption_rate == 0.0_rt
                && response.emission_rate_density == 0.0_rt,
            "response API turned a transparent group into a source");

        auto const exchange = EvaluatePlanckExchange(
            response.absorption_rate, response.emission_rate_density,
            5.0_rt, 2.0_rt, 1.0e-12_rt);
        require(
            exchange.valid && exchange.exchange_energy == 0.0_rt
                && exchange.new_radiation_energy == 5.0_rt,
            "Planck exchange changed pure-transport energy");
        auto const rate = EvaluateFrozenPhotonCollisionRate(
            response.scattering_opacity,
            amrex::GpuArray<amrex::Real, 3>{0.1_rt, 0.0_rt, 0.0_rt},
            amrex::GpuArray<amrex::Real, 3>{0.0_rt, 0.0_rt, 1.0_rt});
        require(
            rate.valid && rate.inverse_length == 0.0_rt
                && rate.maximum_inverse_length == 0.0_rt,
            "zero scattering produced a packet collision rate");
        auto const energy_update =
            warpx::radiation::ApplyRadiationEnergyUpdate(7.0_rt, 0.0_rt);
        require(
            energy_update.valid && energy_update.stored_energy == 7.0_rt
                && energy_update.residual == 0.0_rt,
            "zero transport changed radiation energy");
        auto const impulse = EvaluateMaterialImpulseCandidate(
            amrex::GpuArray<amrex::ParticleReal, 3>{1.0_rt, 0.0_rt, 0.0_rt},
            amrex::GpuArray<amrex::Real, 3>{0.0_rt, 0.0_rt, 0.0_rt},
            3.0_rt,
            amrex::GpuArray<amrex::Real, 3>{0.0_rt, 0.0_rt, 0.0_rt});
        require(
            impulse.valid && impulse.actual_work == 0.0_rt
                && impulse.energy_carry == 3.0_rt
                && impulse.velocity[0] == 1.0_rt,
            "pure transport created material momentum or work");

        amrex::Box const domain(amrex::IntVect(0), amrex::IntVect(7));
        amrex::RealBox const physical(AMREX_D_DECL(0.0_rt, 0.0_rt, 0.0_rt),
                                      AMREX_D_DECL(1.0_rt, 1.0_rt, 1.0_rt));
        int periodic[] = {AMREX_D_DECL(0, 0, 0)};
        amrex::Geometry const geometry(domain, &physical, 0, periodic);
        amrex::BoxArray boxes(domain);
        boxes.maxSize(8);
        amrex::DistributionMapping const distribution(boxes);
        amrex::MultiFab density(boxes, distribution, 1, 1);
        amrex::MultiFab trial(boxes, distribution, 1, 1);
        amrex::MultiFab material(boxes, distribution, 1, 1);
        amrex::MultiFab opacity(boxes, distribution, num_groups, 0);
        amrex::MultiFab absorption(boxes, distribution, num_groups, 0);
        amrex::MultiFab emission(boxes, distribution, num_groups, 0);
        density.setVal(10.0_rt * PhysConst::q_e);
        trial.setVal(3000.0_rt);
        material.setVal(10.0_rt);
        density.FillBoundary(geometry.periodicity());
        trial.FillBoundary(geometry.periodicity());
        material.FillBoundary(geometry.periodicity());
        ElectronThermodynamicsExecutor eos;
        eos.m_model = ElectronThermodynamicsModel::IdealGas;
        eos.m_gamma = 5.0_rt / 3.0_rt;
        eos.m_num_materials = 1;
        eos.m_mass_per_charge = {1.0_rt};
        eos.m_atomic_mass = {1.0_rt};
        eos.m_atomic_number = {1.0_rt};
        amrex::GpuArray<amrex::MultiFab const*,
                       ElectronThermodynamicsExecutor::max_materials>
            material_fields{};
        material_fields[0] = &material;
        warpx::radiation::detail::FillCoupledMaterialTableCoefficients(
            density, trial, opacity, absorption, emission, geometry, eos,
            1.0e-6_rt, 0.0_rt, table.executor(), 1.0_rt,
            device_groups, material_fields);
        require(
            opacity.min(0) == 0.0_rt && opacity.max(0) == 0.0_rt
                && absorption.min(0) == 0.0_rt
                && emission.min(0) == 0.0_rt,
            "FillCoupledMaterialTableCoefficients created a zero-group source");

        warpx::radiation::ImplicitDiffusionOptions options{
            1.0e-8_rt, 1.0e-10_rt, 1.0_rt,
            100, 100, 0, {0, 0, 0}, {0, 0, 0},
            {0, 0, 0, 0, 0, 0}, nullptr,
            {nullptr, nullptr, 1}};
        require(
            !(response.transport_opacity * geometry.CellSize(0)
                  >= options.minimum_optical_depth),
            "zero transport opacity satisfies the diffusion guard");
    }

    void run_diffusion_guard ()
    {
        amrex::Box const domain(amrex::IntVect(0), amrex::IntVect(7));
        amrex::RealBox const physical(AMREX_D_DECL(0.0_rt, 0.0_rt, 0.0_rt),
                                      AMREX_D_DECL(1.0_rt, 1.0_rt, 1.0_rt));
        int periodic[] = {AMREX_D_DECL(0, 0, 0)};
        amrex::Geometry const geometry(domain, &physical, 0, periodic);
        amrex::BoxArray boxes(domain);
        boxes.maxSize(8);
        amrex::DistributionMapping const distribution(boxes);
        amrex::MultiFab radiation(boxes, distribution, 1, 1);
        amrex::MultiFab diffusion_opacity(boxes, distribution, 1, 1);
        radiation.setVal(1.0_rt);
        diffusion_opacity.setVal(0.0_rt);
        diffusion_opacity.FillBoundary(geometry.periodicity());
        warpx::radiation::ImplicitDiffusionOptions options{
            1.0e-8_rt, 1.0e-10_rt, 1.0_rt,
            100, 100, 0, {0, 0, 0}, {0, 0, 0},
            {0, 0, 0, 0, 0, 0}, nullptr,
            {nullptr, nullptr, 1}};
        warpx::radiation::TryAdvanceImplicitDiffusion(
            radiation, diffusion_opacity, geometry,
            0.0_rt, 1.0e-12_rt, options);
    }

    struct Probe
    {
        amrex::Real density;
        amrex::Real temperature;
        int group;
        amrex::Real fraction;
    };

    void run_device ()
    {
        auto const directory = make_directory();
        auto const path = directory / "device.h5";
        auto const spec = make_spec();
        write_table(path, spec);
        MaterialOpacityTable table(path.string(), material_id);
        std::vector<Probe> const probes{
            {10.0_rt, 3000.0_rt, 0, 1.0_rt},
            {10.0_rt, 3162.277660168379_rt, 1, 0.5_rt},
            {1.0_rt, 1000.0_rt, 1, 1.0_rt},
            {100.0_rt, 10000.0_rt, 2, 1.0_rt},
            {0.0_rt, 3000.0_rt, 2, 1.0_rt},
            {200.0_rt, 3000.0_rt, 1, 1.0_rt}};
        amrex::Gpu::DeviceVector<Probe> device_probes(probes.size());
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, probes.begin(), probes.end(),
            device_probes.begin());
        amrex::Gpu::DeviceVector<MaterialOpacityTableResponse> device(
            probes.size());
        auto const executor = table.executor();
        auto const* const probe_ptr = device_probes.data();
        auto* const output = device.data();
        amrex::ParallelFor(
            static_cast<int>(probes.size()),
            [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                output[i] = EvaluateMaterialOpacityTableResponse(
                    executor, probe_ptr[i].density, probe_ptr[i].temperature,
                    probe_ptr[i].group, probe_ptr[i].fraction);
            });
        amrex::Gpu::streamSynchronize();
        std::vector<MaterialOpacityTableResponse> host(probes.size());
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, device.begin(), device.end(),
            host.begin());
        for (std::size_t i = 0; i < probes.size(); ++i) {
            auto const expected = EvaluateMaterialOpacityTableResponse(
                table.hostExecutor(), probes[i].density,
                probes[i].temperature, probes[i].group, probes[i].fraction);
            require(
                host[i].valid == expected.valid
                    && host[i].transport_opacity
                        == expected.transport_opacity
                    && host[i].absorption_rate == expected.absorption_rate
                    && host[i].emission_rate_density
                        == expected.emission_rate_density
                    && host[i].scattering_opacity
                        == expected.scattering_opacity,
                "host/device response parity failed");
        }
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv, false);
    int status = 0;
    try {
        std::string const test = argc > 1 ? argv[1] : "contract";
        if (test == "contract") {
            run_contract();
        } else if (test == "reload") {
            run_reload();
        } else if (test == "consumers") {
            run_consumers();
        } else if (test == "diffusion-guard") {
            run_diffusion_guard();
        } else if (test == "device") {
            run_device();
        } else {
            throw std::runtime_error("unknown zero-opacity test case");
        }
        std::cout << "material opacity zero case '" << test << "' passed\n";
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    amrex::Finalize();
    return status;
}
