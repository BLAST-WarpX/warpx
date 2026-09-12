/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "OpacityTable.H"

#include "Utils/TextMsg.H"

#include <AMReX_Gpu.H>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace warpx::radiation
{
    namespace
    {
        std::vector<std::string> read_tokens (std::string const& filename)
        {
            std::ifstream input(filename);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                input.is_open(),
                "Could not open radiation opacity table '" + filename + "'.");

            std::vector<std::string> tokens;
            std::string line;
            while (std::getline(input, line)) {
                auto const comment = line.find('#');
                if (comment != std::string::npos) { line.erase(comment); }
                std::replace(line.begin(), line.end(), ',', ' ');
                std::istringstream row(line);
                std::string token;
                while (row >> token) { tokens.push_back(token); }
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !input.bad(),
                "Failed while reading radiation opacity table '" + filename + "'.");
            return tokens;
        }

        int read_size (
            std::vector<std::string> const& tokens, std::size_t& cursor,
            std::string const& filename)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                cursor < tokens.size(),
                "Radiation opacity table '" + filename + "' is truncated.");
            std::size_t consumed = 0;
            long long result = 0;
            try {
                result = std::stoll(tokens[cursor], &consumed);
            } catch (std::exception const&) {
                WARPX_ABORT_WITH_MESSAGE(
                    "Radiation opacity table '" + filename
                    + "' has an invalid axis dimension.");
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                consumed == tokens[cursor].size() && result >= 2
                    && result <= std::numeric_limits<int>::max(),
                "Every radiation opacity table axis must contain at least two points in '"
                    + filename + "'.");
            ++cursor;
            return static_cast<int>(result);
        }

        amrex::Real read_real (
            std::vector<std::string> const& tokens, std::size_t& cursor,
            std::string const& filename)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                cursor < tokens.size(),
                "Radiation opacity table '" + filename + "' is truncated.");
            std::size_t consumed = 0;
            double parsed_result = 0.0;
            try {
                parsed_result = std::stod(tokens[cursor], &consumed);
            } catch (std::exception const&) {
                WARPX_ABORT_WITH_MESSAGE(
                    "Radiation opacity table '" + filename
                    + "' contains a non-numeric entry.");
            }
            auto const result = static_cast<amrex::Real>(parsed_result);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                consumed == tokens[cursor].size() && std::isfinite(result),
                "Radiation opacity table '" + filename
                    + "' contains a non-numeric or non-finite entry.");
            ++cursor;
            return result;
        }

        void validate_and_transform_axis (
            amrex::Vector<amrex::Real>& axis,
            OpacityTableInterpolation const mode,
            std::string const& filename,
            std::string const& name)
        {
            std::string increasing_message{"The "};
            increasing_message.append(name)
                .append(" axis in radiation opacity table '")
                .append(filename)
                .append("' must be strictly increasing.");
            std::string positive_message{
                "Log-log radiation opacity tables require positive "};
            positive_message.append(name)
                .append(" coordinates in '")
                .append(filename)
                .append("'.");
            for (amrex::Long i = 0; i < axis.size(); ++i) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    i == 0 || axis[i] > axis[i - 1],
                    increasing_message);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    mode != OpacityTableInterpolation::LogLog || axis[i] > 0.0,
                    positive_message);
            }
            if (mode == OpacityTableInterpolation::LogLog) {
                for (auto& value : axis) { value = std::log(value); }
            }
        }

        void validate_and_transform_values (
            amrex::Gpu::HostVector<amrex::Real>& values,
            OpacityTableInterpolation const mode,
            std::string const& filename)
        {
            for (auto& value : values) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    value >= 0.0,
                    "Radiation opacity coefficients in '" + filename
                        + "' must be non-negative.");
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    mode != OpacityTableInterpolation::LogLog || value > 0.0,
                    "Log-log radiation opacity tables require strictly positive "
                    "coefficients in '" + filename + "'.");
                if (mode == OpacityTableInterpolation::LogLog) {
                    value = std::log(value);
                }
            }
        }

        template <typename Vector>
        void read_vector (
            Vector& values, std::size_t const size,
            std::vector<std::string> const& tokens, std::size_t& cursor,
            std::string const& filename)
        {
            values.resize(size);
            for (auto& value : values) {
                value = read_real(tokens, cursor, filename);
            }
        }

        std::size_t checked_product (
            int const first, int const second, int const third,
            std::string const& filename)
        {
            auto result = static_cast<std::size_t>(first);
            auto const multiply = [&] (int const factor)
            {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    result <= std::numeric_limits<std::size_t>::max()
                        / static_cast<std::size_t>(factor),
                    "Declared dimensions overflow the opacity-table size in '"
                        + filename + "'.");
                result *= static_cast<std::size_t>(factor);
            };
            multiply(second);
            if (third > 0) { multiply(third); }
            return result;
        }
    }

    void OpacityTable2D::load (
        std::string const& filename, OpacityTableInterpolation const mode)
    {
        auto const tokens = read_tokens(filename);
        std::size_t cursor = 0;
        int const num_density = read_size(tokens, cursor, filename);
        int const num_temperature = read_size(tokens, cursor, filename);
        read_vector(m_density_h, num_density, tokens, cursor, filename);
        read_vector(m_temperature_h, num_temperature, tokens, cursor, filename);
        read_vector(
            m_values_h,
            checked_product(
                num_density, num_temperature, 0, filename),
            tokens, cursor, filename);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            cursor == tokens.size(),
            "Radiation opacity table '" + filename
                + "' contains entries beyond its declared dimensions.");

        validate_and_transform_axis(
            m_density_h, mode, filename, "electron-density");
        validate_and_transform_axis(
            m_temperature_h, mode, filename, "temperature");
        validate_and_transform_values(m_values_h, mode, filename);

        m_executor_h = {
            m_density_h.data(), m_temperature_h.data(), m_values_h.data(),
            num_density, num_temperature,
            mode == OpacityTableInterpolation::LogLog};
#ifdef AMREX_USE_GPU
        m_density_d.resize(m_density_h.size());
        m_temperature_d.resize(m_temperature_h.size());
        m_values_d.resize(m_values_h.size());
        amrex::Gpu::copyAsync(
            amrex::Gpu::hostToDevice, m_density_h.begin(), m_density_h.end(),
            m_density_d.begin());
        amrex::Gpu::copyAsync(
            amrex::Gpu::hostToDevice,
            m_temperature_h.begin(), m_temperature_h.end(),
            m_temperature_d.begin());
        amrex::Gpu::copyAsync(
            amrex::Gpu::hostToDevice, m_values_h.begin(), m_values_h.end(),
            m_values_d.begin());
        amrex::Gpu::streamSynchronize();
        m_executor_d = m_executor_h;
        m_executor_d.m_density = m_density_d.data();
        m_executor_d.m_temperature = m_temperature_d.data();
        m_executor_d.m_values = m_values_d.data();
#endif
        m_initialized = true;
    }

    OpacityTable2DExecutor const& OpacityTable2D::executor () const noexcept
    {
#ifdef AMREX_USE_GPU
        return m_executor_d;
#else
        return m_executor_h;
#endif
    }

    void OpacityTable3D::load (
        std::string const& filename, OpacityTableInterpolation const mode)
    {
        auto const tokens = read_tokens(filename);
        std::size_t cursor = 0;
        int const num_density = read_size(tokens, cursor, filename);
        int const num_temperature = read_size(tokens, cursor, filename);
        int const num_energy = read_size(tokens, cursor, filename);
        read_vector(m_density_h, num_density, tokens, cursor, filename);
        read_vector(m_temperature_h, num_temperature, tokens, cursor, filename);
        read_vector(m_photon_energy_h, num_energy, tokens, cursor, filename);
        read_vector(
            m_values_h,
            checked_product(
                num_density, num_temperature, num_energy, filename),
            tokens, cursor, filename);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            cursor == tokens.size(),
            "Radiation opacity table '" + filename
                + "' contains entries beyond its declared dimensions.");

        validate_and_transform_axis(
            m_density_h, mode, filename, "electron-density");
        validate_and_transform_axis(
            m_temperature_h, mode, filename, "temperature");
        validate_and_transform_axis(
            m_photon_energy_h, mode, filename, "photon-energy");
        validate_and_transform_values(m_values_h, mode, filename);

        m_executor_h = {
            m_density_h.data(), m_temperature_h.data(),
            m_photon_energy_h.data(), m_values_h.data(), num_density,
            num_temperature, num_energy,
            mode == OpacityTableInterpolation::LogLog};
#ifdef AMREX_USE_GPU
        m_density_d.resize(m_density_h.size());
        m_temperature_d.resize(m_temperature_h.size());
        m_photon_energy_d.resize(m_photon_energy_h.size());
        m_values_d.resize(m_values_h.size());
        amrex::Gpu::copyAsync(
            amrex::Gpu::hostToDevice, m_density_h.begin(), m_density_h.end(),
            m_density_d.begin());
        amrex::Gpu::copyAsync(
            amrex::Gpu::hostToDevice,
            m_temperature_h.begin(), m_temperature_h.end(),
            m_temperature_d.begin());
        amrex::Gpu::copyAsync(
            amrex::Gpu::hostToDevice,
            m_photon_energy_h.begin(), m_photon_energy_h.end(),
            m_photon_energy_d.begin());
        amrex::Gpu::copyAsync(
            amrex::Gpu::hostToDevice, m_values_h.begin(), m_values_h.end(),
            m_values_d.begin());
        amrex::Gpu::streamSynchronize();
        m_executor_d = m_executor_h;
        m_executor_d.m_density = m_density_d.data();
        m_executor_d.m_temperature = m_temperature_d.data();
        m_executor_d.m_photon_energy = m_photon_energy_d.data();
        m_executor_d.m_values = m_values_d.data();
#endif
        m_initialized = true;
    }

    OpacityTable3DExecutor const& OpacityTable3D::executor () const noexcept
    {
#ifdef AMREX_USE_GPU
        return m_executor_d;
#else
        return m_executor_h;
#endif
    }
}
