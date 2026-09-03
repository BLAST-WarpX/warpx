/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_OPACITYTABLE_H_
#define WARPX_RADIATION_OPACITYTABLE_H_

#include <AMReX_Algorithm.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <string>

namespace warpx::radiation
{
    enum class OpacityTableInterpolation
    {
        Linear,
        LogLog
    };

    /** Device-callable bilinear opacity interpolator.
     *
     * Coordinates outside the supplied table are clamped to the nearest
     * endpoint.  This makes the extrapolation policy explicit and bounded.
     */
    struct OpacityTable2DExecutor
    {
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real operator() (
            amrex::Real density, amrex::Real temperature) const noexcept
        {
            amrex::Real const x = coordinate(density, m_density[0]);
            amrex::Real const y = coordinate(temperature, m_temperature[0]);
            int const ix = interval(m_density, m_num_density, x);
            int const iy = interval(m_temperature, m_num_temperature, y);
            amrex::Real const tx = fraction(m_density, ix, x);
            amrex::Real const ty = fraction(m_temperature, iy, y);

            amrex::Real const v0 = value(ix, iy)
                + tx * (value(ix + 1, iy) - value(ix, iy));
            amrex::Real const v1 = value(ix, iy + 1)
                + tx * (value(ix + 1, iy + 1) - value(ix, iy + 1));
            return result(v0 + ty * (v1 - v0));
        }

        amrex::Real const* m_density = nullptr;
        amrex::Real const* m_temperature = nullptr;
        amrex::Real const* m_values = nullptr;
        int m_num_density = 0;
        int m_num_temperature = 0;
        bool m_log_log = false;

    private:
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real value (int i, int j) const noexcept
        {
            return m_values[i * m_num_temperature + j];
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real coordinate (
            amrex::Real value, amrex::Real lower_endpoint) const noexcept
        {
            return m_log_log
                ? (value > 0.0 ? std::log(value) : lower_endpoint)
                : value;
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real result (amrex::Real value) const noexcept
        {
            return m_log_log ? std::exp(value) : value;
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        static int interval (
            amrex::Real const* axis, int size, amrex::Real value) noexcept
        {
            if (value <= axis[0]) { return 0; }
            if (value >= axis[size - 1]) { return size - 2; }
            return amrex::bisect(axis, 0, size - 1, value);
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        static amrex::Real fraction (
            amrex::Real const* axis, int index, amrex::Real value) noexcept
        {
            amrex::Real const clamped = amrex::max(
                axis[index], amrex::min(value, axis[index + 1]));
            return (clamped - axis[index])
                / (axis[index + 1] - axis[index]);
        }
    };

    /** Device-callable trilinear spectral-opacity interpolator. */
    struct OpacityTable3DExecutor
    {
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real operator() (
            amrex::Real density, amrex::Real temperature,
            amrex::Real photon_energy) const noexcept
        {
            amrex::Real const x = coordinate(density, m_density[0]);
            amrex::Real const y = coordinate(temperature, m_temperature[0]);
            amrex::Real const z = coordinate(
                photon_energy, m_photon_energy[0]);
            int const ix = interval(m_density, m_num_density, x);
            int const iy = interval(m_temperature, m_num_temperature, y);
            int const iz = interval(m_photon_energy, m_num_photon_energy, z);
            amrex::Real const tx = fraction(m_density, ix, x);
            amrex::Real const ty = fraction(m_temperature, iy, y);
            amrex::Real const tz = fraction(m_photon_energy, iz, z);

            amrex::Real const v00 = value(ix, iy, iz)
                + tx * (value(ix + 1, iy, iz) - value(ix, iy, iz));
            amrex::Real const v10 = value(ix, iy + 1, iz)
                + tx * (value(ix + 1, iy + 1, iz) - value(ix, iy + 1, iz));
            amrex::Real const v01 = value(ix, iy, iz + 1)
                + tx * (value(ix + 1, iy, iz + 1) - value(ix, iy, iz + 1));
            amrex::Real const v11 = value(ix, iy + 1, iz + 1)
                + tx * (value(ix + 1, iy + 1, iz + 1)
                        - value(ix, iy + 1, iz + 1));
            amrex::Real const v0 = v00 + ty * (v10 - v00);
            amrex::Real const v1 = v01 + ty * (v11 - v01);
            return result(v0 + tz * (v1 - v0));
        }

        amrex::Real const* m_density = nullptr;
        amrex::Real const* m_temperature = nullptr;
        amrex::Real const* m_photon_energy = nullptr;
        amrex::Real const* m_values = nullptr;
        int m_num_density = 0;
        int m_num_temperature = 0;
        int m_num_photon_energy = 0;
        bool m_log_log = false;

    private:
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real value (int i, int j, int k) const noexcept
        {
            return m_values[
                (i * m_num_temperature + j) * m_num_photon_energy + k];
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real coordinate (
            amrex::Real value, amrex::Real lower_endpoint) const noexcept
        {
            return m_log_log
                ? (value > 0.0 ? std::log(value) : lower_endpoint)
                : value;
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real result (amrex::Real value) const noexcept
        {
            return m_log_log ? std::exp(value) : value;
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        static int interval (
            amrex::Real const* axis, int size, amrex::Real value) noexcept
        {
            if (value <= axis[0]) { return 0; }
            if (value >= axis[size - 1]) { return size - 2; }
            return amrex::bisect(axis, 0, size - 1, value);
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        static amrex::Real fraction (
            amrex::Real const* axis, int index, amrex::Real value) noexcept
        {
            amrex::Real const clamped = amrex::max(
                axis[index], amrex::min(value, axis[index + 1]));
            return (clamped - axis[index])
                / (axis[index + 1] - axis[index]);
        }
    };

    class OpacityTable2D
    {
    public:
        void load (std::string const& filename, OpacityTableInterpolation mode);
        [[nodiscard]] OpacityTable2DExecutor const& executor () const noexcept;
        [[nodiscard]] bool initialized () const noexcept { return m_initialized; }

    private:
        amrex::Vector<amrex::Real> m_density_h;
        amrex::Vector<amrex::Real> m_temperature_h;
        amrex::Gpu::HostVector<amrex::Real> m_values_h;
#ifdef AMREX_USE_GPU
        amrex::Gpu::DeviceVector<amrex::Real> m_density_d;
        amrex::Gpu::DeviceVector<amrex::Real> m_temperature_d;
        amrex::Gpu::DeviceVector<amrex::Real> m_values_d;
        OpacityTable2DExecutor m_executor_d;
#endif
        OpacityTable2DExecutor m_executor_h;
        bool m_initialized = false;
    };

    class OpacityTable3D
    {
    public:
        void load (std::string const& filename, OpacityTableInterpolation mode);
        [[nodiscard]] OpacityTable3DExecutor const& executor () const noexcept;
        [[nodiscard]] bool initialized () const noexcept { return m_initialized; }

    private:
        amrex::Vector<amrex::Real> m_density_h;
        amrex::Vector<amrex::Real> m_temperature_h;
        amrex::Vector<amrex::Real> m_photon_energy_h;
        amrex::Gpu::HostVector<amrex::Real> m_values_h;
#ifdef AMREX_USE_GPU
        amrex::Gpu::DeviceVector<amrex::Real> m_density_d;
        amrex::Gpu::DeviceVector<amrex::Real> m_temperature_d;
        amrex::Gpu::DeviceVector<amrex::Real> m_photon_energy_d;
        amrex::Gpu::DeviceVector<amrex::Real> m_values_d;
        OpacityTable3DExecutor m_executor_d;
#endif
        OpacityTable3DExecutor m_executor_h;
        bool m_initialized = false;
    };
}

#endif // WARPX_RADIATION_OPACITYTABLE_H_
