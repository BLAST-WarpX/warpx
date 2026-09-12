/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "MaterialOpacityTable.H"

#include <AMReX_Gpu.H>

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace warpx::radiation
{
    namespace
    {
        constexpr char format_name[] = "org.warpx.opacity.multigroup";
        constexpr char schema_version[] = "1.0.0";

        class Hdf5Handle
        {
        public:
            using Closer = herr_t (*) (hid_t);

            Hdf5Handle () = default;
            Hdf5Handle (hid_t id, Closer closer) : m_id(id), m_closer(closer) {}
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
            hid_t m_id = -1;
            Closer m_closer = nullptr;
        };

        [[noreturn]] void fail (
            std::string const& filename, std::string const& message)
        {
            throw std::runtime_error(
                "Material opacity table '" + filename + "': " + message);
        }

        void require (
            bool condition, std::string const& filename,
            std::string const& message)
        {
            if (!condition) { fail(filename, message); }
        }

        Hdf5Handle open_file (std::string const& filename)
        {
            hid_t id = -1;
            H5E_BEGIN_TRY
            {
                id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            }
            H5E_END_TRY;
            require(id >= 0, filename, "could not open the HDF5 file.");
            return {id, H5Fclose};
        }

        Hdf5Handle open_group (
            hid_t location, std::string const& path,
            std::string const& filename)
        {
            hid_t id = -1;
            H5E_BEGIN_TRY
            {
                id = H5Gopen2(location, path.c_str(), H5P_DEFAULT);
            }
            H5E_END_TRY;
            require(id >= 0, filename, "missing group '" + path + "'.");
            return {id, H5Gclose};
        }

        Hdf5Handle open_dataset (
            hid_t location, std::string const& path,
            std::string const& filename)
        {
            hid_t id = -1;
            H5E_BEGIN_TRY
            {
                id = H5Dopen2(location, path.c_str(), H5P_DEFAULT);
            }
            H5E_END_TRY;
            require(id >= 0, filename, "missing dataset '" + path + "'.");
            return {id, H5Dclose};
        }

        Hdf5Handle open_attribute (
            hid_t location, std::string const& name,
            std::string const& object_name, std::string const& filename)
        {
            hid_t id = -1;
            H5E_BEGIN_TRY
            {
                id = H5Aopen(location, name.c_str(), H5P_DEFAULT);
            }
            H5E_END_TRY;
            require(
                id >= 0, filename,
                "missing attribute '" + name + "' on '" + object_name + "'.");
            return {id, H5Aclose};
        }

        void require_scalar_space (
            hid_t object, bool attribute, std::string const& object_name,
            std::string const& filename)
        {
            hid_t raw_space = attribute
                ? H5Aget_space(object) : H5Dget_space(object);
            require(
                raw_space >= 0, filename,
                "could not inspect the dataspace of '" + object_name + "'.");
            Hdf5Handle space{raw_space, H5Sclose};
            require(
                H5Sget_simple_extent_type(space.get()) == H5S_SCALAR,
                filename, "'" + object_name + "' must be scalar.");
        }

        std::string read_string (
            hid_t object, bool attribute, std::string const& object_name,
            std::string const& filename)
        {
            require_scalar_space(object, attribute, object_name, filename);
            hid_t raw_type = attribute
                ? H5Aget_type(object) : H5Dget_type(object);
            require(
                raw_type >= 0, filename,
                "could not inspect the datatype of '" + object_name + "'.");
            Hdf5Handle type{raw_type, H5Tclose};
            require(
                H5Tget_class(type.get()) == H5T_STRING, filename,
                "'" + object_name + "' must contain a string.");

            std::string result;
            if (H5Tis_variable_str(type.get()) > 0) {
                char* value = nullptr;
                herr_t const status = attribute
                    ? H5Aread(object, type.get(), &value)
                    : H5Dread(
                        object, type.get(), H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, &value);
                require(
                    status >= 0, filename,
                    "could not read string '" + object_name + "'.");
                if (value != nullptr) {
                    result = value;
                    H5free_memory(value);
                }
            } else {
                std::size_t const size = H5Tget_size(type.get());
                require(
                    size > 0, filename,
                    "string '" + object_name + "' has zero storage size.");
                std::vector<char> value(size + 1U, '\0');
                herr_t const status = attribute
                    ? H5Aread(object, type.get(), value.data())
                    : H5Dread(
                        object, type.get(), H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, value.data());
                require(
                    status >= 0, filename,
                    "could not read string '" + object_name + "'.");
                result.assign(value.data(), strnlen(value.data(), size));
            }
            return result;
        }

        std::string read_string_attribute (
            hid_t location, std::string const& name,
            std::string const& object_name, std::string const& filename)
        {
            auto attribute = open_attribute(
                location, name, object_name, filename);
            return read_string(
                attribute.get(), true, object_name + "@" + name, filename);
        }

        std::string read_string_dataset (
            hid_t location, std::string const& path,
            std::string const& filename)
        {
            auto dataset = open_dataset(location, path, filename);
            return read_string(dataset.get(), false, path, filename);
        }

        template <typename T>
        T read_integer_attribute (
            hid_t location, std::string const& name,
            std::string const& object_name, hid_t memory_type,
            std::string const& filename)
        {
            auto attribute = open_attribute(
                location, name, object_name, filename);
            require_scalar_space(
                attribute.get(), true, object_name + "@" + name, filename);
            auto type = Hdf5Handle{H5Aget_type(attribute.get()), H5Tclose};
            require(
                type.get() >= 0 && H5Tget_class(type.get()) == H5T_INTEGER
                    && H5Tget_size(type.get()) == sizeof(T)
                    && H5Tget_sign(type.get()) == H5T_SGN_2,
                filename, "attribute '" + name + "' on '" + object_name
                    + "' has the wrong signed-integer type.");
            T result{};
            require(
                H5Aread(attribute.get(), memory_type, &result) >= 0,
                filename, "could not read attribute '" + name + "' on '"
                    + object_name + "'.");
            return result;
        }

        struct DoubleDataset
        {
            std::vector<double> values;
            std::vector<hsize_t> dimensions;
        };

        std::size_t checked_size (
            std::vector<hsize_t> const& dimensions,
            std::string const& path, std::string const& filename)
        {
            std::size_t result = 1U;
            for (auto const extent : dimensions) {
                require(extent > 0, filename, "dataset '" + path
                    + "' must not contain an empty dimension.");
                require(
                    extent <= std::numeric_limits<std::size_t>::max() / result,
                    filename, "shape of dataset '" + path + "' overflows.");
                result *= static_cast<std::size_t>(extent);
            }
            return result;
        }

        std::vector<hsize_t> read_dimensions (
            hid_t dataset, std::string const& path,
            std::string const& filename)
        {
            auto space = Hdf5Handle{H5Dget_space(dataset), H5Sclose};
            require(
                space.get() >= 0, filename,
                "could not inspect dataset '" + path + "'.");
            int const rank = H5Sget_simple_extent_ndims(space.get());
            require(
                rank > 0, filename,
                "dataset '" + path + "' must be an array.");
            std::vector<hsize_t> dimensions(static_cast<std::size_t>(rank));
            require(
                H5Sget_simple_extent_dims(
                    space.get(), dimensions.data(), nullptr) >= 0,
                filename, "could not read the shape of dataset '" + path + "'.");
            return dimensions;
        }

        void require_units (
            hid_t dataset, std::string const& path,
            std::string const& expected, std::string const& filename)
        {
            require(
                read_string_attribute(dataset, "units", path, filename)
                    == expected,
                filename, "dataset '" + path + "' must use units '"
                    + expected + "'.");
        }

        DoubleDataset read_double_dataset (
            hid_t location, std::string const& path,
            std::string const& units, std::string const& filename)
        {
            auto dataset = open_dataset(location, path, filename);
            auto type = Hdf5Handle{H5Dget_type(dataset.get()), H5Tclose};
            require(
                type.get() >= 0 && H5Tget_class(type.get()) == H5T_FLOAT
                    && H5Tget_size(type.get()) == sizeof(double),
                filename, "dataset '" + path + "' must be float64.");
            require_units(dataset.get(), path, units, filename);
            auto dimensions = read_dimensions(dataset.get(), path, filename);
            std::vector<double> values(
                checked_size(dimensions, path, filename));
            require(
                H5Dread(
                    dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, values.data()) >= 0,
                filename, "could not read dataset '" + path + "'.");
            return {std::move(values), std::move(dimensions)};
        }

        std::vector<std::int32_t> read_int32_vector (
            hid_t location, std::string const& path,
            std::string const& units, std::string const& filename)
        {
            auto dataset = open_dataset(location, path, filename);
            auto type = Hdf5Handle{H5Dget_type(dataset.get()), H5Tclose};
            require(
                type.get() >= 0 && H5Tget_class(type.get()) == H5T_INTEGER
                    && H5Tget_size(type.get()) == sizeof(std::int32_t)
                    && H5Tget_sign(type.get()) == H5T_SGN_2,
                filename, "dataset '" + path + "' must be int32.");
            require_units(dataset.get(), path, units, filename);
            auto const dimensions = read_dimensions(dataset.get(), path, filename);
            require(
                dimensions.size() == 1U, filename,
                "dataset '" + path + "' must be rank one.");
            std::vector<std::int32_t> values(
                checked_size(dimensions, path, filename));
            require(
                H5Dread(
                    dataset.get(), H5T_NATIVE_INT32, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, values.data()) >= 0,
                filename, "could not read dataset '" + path + "'.");
            return values;
        }

        void require_exact_attribute (
            hid_t location, std::string const& name,
            std::string const& object_name, std::string const& expected,
            std::string const& filename)
        {
            require(
                read_string_attribute(
                    location, name, object_name, filename) == expected,
                filename, "attribute '" + name + "' on '" + object_name
                    + "' must be '" + expected + "'.");
        }

        void require_nonempty_attribute (
            hid_t location, std::string const& name,
            std::string const& object_name, std::string const& filename)
        {
            require(
                !read_string_attribute(
                    location, name, object_name, filename).empty(),
                filename, "attribute '" + name + "' on '" + object_name
                    + "' must not be empty.");
        }

        void require_nonempty_dataset (
            hid_t location, std::string const& path,
            std::string const& filename)
        {
            require(
                !read_string_dataset(location, path, filename).empty(),
                filename, "provenance dataset '" + path
                    + "' must not be empty.");
        }

        bool is_sha256 (std::string const& value)
        {
            return value.size() == 64U
                && std::all_of(value.begin(), value.end(), [] (char c)
                {
                    return (c >= '0' && c <= '9')
                        || (c >= 'a' && c <= 'f')
                        || (c >= 'A' && c <= 'F');
                });
        }

        amrex::Gpu::HostVector<amrex::Real> logarithms (
            std::vector<double> const& values, std::string const& path,
            std::string const& filename)
        {
            amrex::Gpu::HostVector<amrex::Real> result(values.size());
            double const log_min = std::log(
                static_cast<double>(std::numeric_limits<amrex::Real>::min()));
            double const log_max = std::log(
                static_cast<double>(std::numeric_limits<amrex::Real>::max()));
            for (std::size_t i = 0; i < values.size(); ++i) {
                require(
                    std::isfinite(values[i]) && values[i] > 0.0,
                    filename, "dataset '" + path
                        + "' must contain finite, strictly positive values for "
                          "log-log interpolation.");
                double const value = std::log(values[i]);
                require(
                    value >= log_min && value <= log_max,
                    filename, "dataset '" + path
                        + "' contains a value outside the configured Real range.");
                result[i] = static_cast<amrex::Real>(value);
            }
            return result;
        }

        void require_strictly_increasing_real_axis (
            amrex::Gpu::HostVector<amrex::Real> const& values,
            std::string const& path, std::string const& filename)
        {
            for (std::size_t i = 1; i < values.size(); ++i) {
                require(
                    values[i] > values[i - 1U], filename,
                    "axis '" + path + "' is not strictly increasing in the "
                    "configured Real precision.");
            }
        }

        struct LoadedMaterial
        {
            amrex::Gpu::HostVector<amrex::Real> log_density;
            amrex::Gpu::HostVector<amrex::Real> log_temperature;
            amrex::Gpu::HostVector<amrex::Real> log_planck_absorption;
            amrex::Gpu::HostVector<amrex::Real> log_planck_emission;
            amrex::Gpu::HostVector<amrex::Real> log_rosseland_transport;
            amrex::Gpu::HostVector<amrex::Real> log_scattering;
            amrex::Vector<amrex::Real> group_edges;
            amrex::Vector<amrex::Real> representative_energy;
            std::string material_key;
            int num_density = 0;
            int num_temperature = 0;
            int num_groups = 0;
        };

        LoadedMaterial read_material (
            std::string const& filename, std::int64_t material_id)
        {
            auto file = open_file(filename);
            require_exact_attribute(
                file.get(), "format", "/", format_name, filename);
            require_exact_attribute(
                file.get(), "schema_version", "/", schema_version, filename);
            require_exact_attribute(
                file.get(), "unit_system", "/", "SI", filename);
            require_exact_attribute(
                file.get(), "array_order", "/",
                "rho,temperature,group", filename);

            auto materials = open_group(file.get(), "/materials", filename);
            std::string const material_path =
                "/materials/" + std::to_string(material_id);
            auto material = open_group(file.get(), material_path, filename);
            require(
                read_integer_attribute<std::int64_t>(
                    material.get(), "material_id", material_path,
                    H5T_NATIVE_INT64, filename) == material_id,
                filename, "selected material path and material_id attribute differ.");
            require_nonempty_attribute(
                material.get(), "material_key", material_path, filename);
            require_nonempty_attribute(
                material.get(), "material_name", material_path, filename);
            require_exact_attribute(
                material.get(), "thermodynamic_regime", material_path,
                "LTE", filename);
            require_exact_attribute(
                material.get(), "composition_model", material_path,
                "fixed", filename);
            require_exact_attribute(
                material.get(), "profile", material_path,
                "complete-multigroup-v1", filename);
            require_exact_attribute(
                material.get(), "spectral_coverage", material_path,
                "complete_0_inf", filename);

            std::string const material_key = read_string_attribute(
                material.get(), "material_key", material_path, filename);

            std::string const composition_path = material_path + "/composition";
            auto composition = open_group(file.get(), composition_path, filename);
            auto const atomic_number = read_int32_vector(
                composition.get(), "atomic_number", "1", filename);
            auto const atomic_mass = read_double_dataset(
                composition.get(), "atomic_mass", "kg", filename);
            auto const number_fraction = read_double_dataset(
                composition.get(), "number_fraction", "1", filename);
            require(
                atomic_mass.dimensions.size() == 1U
                    && number_fraction.dimensions.size() == 1U
                    && atomic_mass.values.size() == atomic_number.size()
                    && number_fraction.values.size() == atomic_number.size(),
                filename, "composition vectors must be rank-one and have "
                    "identical nonzero lengths.");
            for (std::size_t i = 0; i < atomic_number.size(); ++i) {
                require(
                    atomic_number[i] > 0
                        && std::isfinite(atomic_mass.values[i])
                        && atomic_mass.values[i] > 0.0
                        && std::isfinite(number_fraction.values[i])
                        && number_fraction.values[i] >= 0.0
                        && number_fraction.values[i] <= 1.0,
                    filename, "composition contains an invalid element, mass, "
                        "or number fraction.");
            }
            double const fraction_sum = std::accumulate(
                number_fraction.values.begin(), number_fraction.values.end(),
                0.0);
            require(
                std::abs(fraction_sum - 1.0)
                    <= 64.0 * std::numeric_limits<double>::epsilon(),
                filename, "composition number fractions must sum to one.");

            std::string const axes_path = material_path + "/axes";
            auto axes = open_group(file.get(), axes_path, filename);
            auto const density = read_double_dataset(
                axes.get(), "mass_density", "kg m-3", filename);
            auto const temperature = read_double_dataset(
                axes.get(), "electron_temperature", "K", filename);
            require(
                density.dimensions.size() == 1U
                    && !density.values.empty(),
                filename, "mass-density axis must be a nonempty rank-one array.");
            require(
                temperature.dimensions.size() == 1U
                    && !temperature.values.empty(),
                filename, "temperature axis must be a nonempty rank-one array.");
            require(
                density.values.size()
                        <= static_cast<std::size_t>(
                            std::numeric_limits<int>::max())
                    && temperature.values.size()
                        <= static_cast<std::size_t>(
                            std::numeric_limits<int>::max()),
                filename, "axis size exceeds the executor integer range.");
            for (std::size_t i = 1; i < density.values.size(); ++i) {
                require(
                    density.values[i] > density.values[i - 1], filename,
                    "mass-density axis must be strictly increasing.");
            }
            for (std::size_t i = 1; i < temperature.values.size(); ++i) {
                require(
                    temperature.values[i] > temperature.values[i - 1],
                    filename, "temperature axis must be strictly increasing.");
            }

            std::string const groups_path = material_path + "/groups";
            auto groups = open_group(file.get(), groups_path, filename);
            require_exact_attribute(
                groups.get(), "interval_convention", groups_path,
                "[edge[g],edge[g+1])", filename);
            require_exact_attribute(
                groups.get(), "first_edge", groups_path, "zero", filename);
            require_exact_attribute(
                groups.get(), "last_edge", groups_path,
                "positive_infinity", filename);
            auto const edges = read_double_dataset(
                groups.get(), "edges", "J", filename);
            auto const representative = read_double_dataset(
                groups.get(), "representative_energy", "J", filename);
            auto const group_index = read_int32_vector(
                groups.get(), "index", "1", filename);
            require(
                edges.dimensions.size() == 1U && edges.values.size() >= 2U
                    && representative.dimensions.size() == 1U
                    && representative.values.size() + 1U == edges.values.size()
                    && group_index.size() == representative.values.size(),
                filename, "group edges, indices, and representative energies "
                    "have incompatible shapes.");
            require(
                representative.values.size()
                    <= static_cast<std::size_t>(
                        std::numeric_limits<int>::max()),
                filename, "group count exceeds the executor integer range.");
            for (std::size_t group = 0; group < group_index.size(); ++group) {
                require(
                    group_index[group] == static_cast<std::int32_t>(group),
                    filename, "group indices must be contiguous and start at zero.");
            }
            require(
                edges.values.front() == 0.0, filename,
                "the first photon group edge must be exactly zero.");
            require(
                std::isinf(edges.values.back()) && edges.values.back() > 0.0,
                filename, "the last photon group edge must be positive infinity; "
                    "finite-tail tables are incomplete.");
            for (std::size_t i = 1; i + 1U < edges.values.size(); ++i) {
                require(
                    std::isfinite(edges.values[i])
                        && edges.values[i] > edges.values[i - 1],
                    filename, "interior photon group edges must be finite, "
                        "positive, and strictly increasing.");
            }
            for (std::size_t group = 0;
                 group < representative.values.size(); ++group)
            {
                require(
                    std::isfinite(representative.values[group])
                        && representative.values[group] > edges.values[group]
                        && representative.values[group]
                            < edges.values[group + 1U],
                    filename, "each representative energy must lie strictly "
                        "inside its photon group.");
            }

            std::string const opacity_path = material_path + "/opacity/group";
            auto opacity = open_group(file.get(), opacity_path, filename);
            auto const read_opacity = [&] (
                std::string const& name, std::string const& mean_definition,
                std::string const& interaction, int includes_scattering)
            {
                auto dataset = open_dataset(opacity.get(), name, filename);
                require_units(dataset.get(), name, "m2 kg-1", filename);
                require_exact_attribute(
                    dataset.get(), "mean_definition", name,
                    mean_definition, filename);
                require_exact_attribute(
                    dataset.get(), "interaction", name,
                    interaction, filename);
                require(
                    read_integer_attribute<int>(
                        dataset.get(), "includes_scattering", name,
                        H5T_NATIVE_INT, filename) == includes_scattering,
                    filename, "dataset '" + name
                        + "' has inconsistent scattering metadata.");
                auto type = Hdf5Handle{H5Dget_type(dataset.get()), H5Tclose};
                require(
                    type.get() >= 0 && H5Tget_class(type.get()) == H5T_FLOAT
                        && H5Tget_size(type.get()) == sizeof(double),
                    filename, "dataset '" + name + "' must be float64.");
                auto dimensions = read_dimensions(dataset.get(), name, filename);
                std::vector<double> values(
                    checked_size(dimensions, name, filename));
                require(
                    H5Dread(
                        dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, values.data()) >= 0,
                    filename, "could not read dataset '" + name + "'.");
                return DoubleDataset{
                    std::move(values), std::move(dimensions)};
            };

            auto const planck = read_opacity(
                "planck_absorption", "planck", "true_absorption", 0);
            htri_t const has_planck_emission = H5Lexists(
                opacity.get(), "planck_emission", H5P_DEFAULT);
            require(
                has_planck_emission >= 0, filename,
                "could not inspect optional dataset 'planck_emission'.");
            auto const emission = has_planck_emission > 0
                ? read_opacity(
                    "planck_emission", "planck", "thermal_emission", 0)
                : planck;
            auto const rosseland = read_opacity(
                "rosseland_transport", "rosseland", "total_transport", 1);
            auto const scattering = read_opacity(
                "planck_scattering", "planck", "scattering", 1);
            require_exact_attribute(
                open_dataset(opacity.get(), "planck_absorption", filename).get(),
                "stimulated_emission", "planck_absorption", "included", filename);
            require_exact_attribute(
                open_dataset(opacity.get(), "rosseland_transport", filename).get(),
                "transport_correction", "rosseland_transport", "1-g", filename);

            std::vector<hsize_t> const expected_shape{
                static_cast<hsize_t>(density.values.size()),
                static_cast<hsize_t>(temperature.values.size()),
                static_cast<hsize_t>(representative.values.size())};
            require(
                planck.dimensions == expected_shape
                    && emission.dimensions == expected_shape
                    && rosseland.dimensions == expected_shape
                    && scattering.dimensions == expected_shape,
                filename, "opacity array shape must be exactly "
                    "[mass_density,electron_temperature,group].");

            if (has_planck_emission > 0) {
                double const tolerance_factor =
                    64.0 * std::numeric_limits<double>::epsilon();
                for (std::size_t i = 0; i < planck.values.size(); ++i) {
                    double const tolerance = tolerance_factor
                        * std::max(
                            std::abs(planck.values[i]),
                            std::abs(emission.values[i]));
                    require(
                        std::abs(planck.values[i] - emission.values[i])
                            <= tolerance,
                        filename,
                        "planck_emission must satisfy LTE detailed balance "
                        "with planck_absorption.");
                }
            }

            std::string const provenance_path = material_path + "/provenance";
            auto provenance = open_group(
                file.get(), provenance_path, filename);
            for (char const* name : {
                     "generator_name", "generator_version",
                     "generator_git_commit", "converter_name",
                     "converter_version", "converter_git_commit",
                     "license_spdx", "redistribution", "physics_model",
                     "created_utc"})
            {
                require_nonempty_dataset(
                    provenance.get(), name, filename);
            }
            for (char const* name : {
                     "generator_input_sha256", "source_output_sha256"})
            {
                auto const value = read_string_dataset(
                    provenance.get(), name, filename);
                require(
                    is_sha256(value), filename,
                    "provenance dataset '" + std::string(name)
                        + "' must be a 64-digit SHA-256 value.");
            }

            LoadedMaterial result;
            result.log_density = logarithms(
                density.values, "mass_density", filename);
            result.log_temperature = logarithms(
                temperature.values, "electron_temperature", filename);
            require_strictly_increasing_real_axis(
                result.log_density, "mass_density", filename);
            require_strictly_increasing_real_axis(
                result.log_temperature, "electron_temperature", filename);
            result.log_planck_absorption = logarithms(
                planck.values, "planck_absorption", filename);
            // Schema 1 is LTE: retain the separate executor field for ABI
            // stability, but canonicalize both channels to the same stored
            // logarithms after the raw-double detailed-balance check above.
            result.log_planck_emission = result.log_planck_absorption;
            result.log_rosseland_transport = logarithms(
                rosseland.values, "rosseland_transport", filename);
            result.log_scattering = logarithms(
                scattering.values, "planck_scattering", filename);
            result.group_edges.resize(edges.values.size());
            result.representative_energy.resize(representative.values.size());
            for (std::size_t i = 0; i < edges.values.size(); ++i) {
                result.group_edges[i] =
                    static_cast<amrex::Real>(edges.values[i]);
                require(
                    i == 0U || i + 1U == edges.values.size()
                        || (std::isfinite(result.group_edges[i])
                            && result.group_edges[i]
                                > result.group_edges[i - 1U]),
                    filename, "an interior photon group edge is not "
                        "representable in the configured Real precision.");
            }
            for (std::size_t i = 0;
                 i < representative.values.size(); ++i)
            {
                result.representative_energy[i] =
                    static_cast<amrex::Real>(representative.values[i]);
                require(
                    std::isfinite(result.representative_energy[i])
                        && result.representative_energy[i]
                            > result.group_edges[i]
                        && result.representative_energy[i]
                            < result.group_edges[i + 1U],
                    filename, "a representative photon energy is not strictly "
                        "inside its group in the configured Real precision.");
            }
            result.material_key = material_key;
            result.num_density = static_cast<int>(density.values.size());
            result.num_temperature = static_cast<int>(temperature.values.size());
            result.num_groups = static_cast<int>(representative.values.size());
            return result;
        }
    }

    MaterialOpacityTable::MaterialOpacityTable (
        std::string const& filename, std::int64_t material_id)
    {
        load(filename, material_id);
    }

    void MaterialOpacityTable::load (
        std::string const& filename, std::int64_t material_id)
    {
        LoadedMaterial loaded = read_material(filename, material_id);

        // A reload cannot invalidate storage still used by an earlier kernel.
        if (m_initialized) { amrex::Gpu::streamSynchronize(); }

        m_log_density_h = std::move(loaded.log_density);
        m_log_temperature_h = std::move(loaded.log_temperature);
        m_log_planck_absorption_h = std::move(loaded.log_planck_absorption);
        m_log_planck_emission_h = std::move(loaded.log_planck_emission);
        m_log_rosseland_transport_h =
            std::move(loaded.log_rosseland_transport);
        m_log_scattering_h = std::move(loaded.log_scattering);
        m_group_edges_h = std::move(loaded.group_edges);
        m_representative_energy_h =
            std::move(loaded.representative_energy);
        m_material_id = material_id;
        m_material_key = std::move(loaded.material_key);

        m_executor_h = {
            m_log_density_h.data(), m_log_temperature_h.data(),
            m_log_planck_absorption_h.data(),
            m_log_planck_emission_h.data(),
            m_log_rosseland_transport_h.data(), m_log_scattering_h.data(),
            loaded.num_density, loaded.num_temperature, loaded.num_groups};
#ifdef AMREX_USE_GPU
        auto const copy_to_device = [] (auto const& host, auto& device)
        {
            device.resize(host.size());
            amrex::Gpu::copyAsync(
                amrex::Gpu::hostToDevice, host.begin(), host.end(),
                device.begin());
        };
        copy_to_device(m_log_density_h, m_log_density_d);
        copy_to_device(m_log_temperature_h, m_log_temperature_d);
        copy_to_device(
            m_log_planck_absorption_h, m_log_planck_absorption_d);
        copy_to_device(
            m_log_planck_emission_h, m_log_planck_emission_d);
        copy_to_device(
            m_log_rosseland_transport_h, m_log_rosseland_transport_d);
        copy_to_device(m_log_scattering_h, m_log_scattering_d);
        amrex::Gpu::streamSynchronize();
        m_executor_d = {
            m_log_density_d.data(), m_log_temperature_d.data(),
            m_log_planck_absorption_d.data(),
            m_log_planck_emission_d.data(),
            m_log_rosseland_transport_d.data(), m_log_scattering_d.data(),
            loaded.num_density, loaded.num_temperature, loaded.num_groups};
#endif
        m_initialized = true;
    }

    MaterialOpacityTableExecutor
    MaterialOpacityTable::executor () const noexcept
    {
#ifdef AMREX_USE_GPU
        return m_executor_d;
#else
        return m_executor_h;
#endif
    }
}
