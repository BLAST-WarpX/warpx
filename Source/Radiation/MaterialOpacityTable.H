/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_MATERIALOPACITYTABLE_H_
#define WARPX_RADIATION_MATERIALOPACITYTABLE_H_

#include <AMReX_Algorithm.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

namespace warpx::radiation
{
    using namespace amrex::literals;

    /** Linear (inverse-length) coefficients for one photon-energy group. */
    struct MaterialOpacityCoefficients
    {
        amrex::Real planck_absorption = 0.0_rt;
        amrex::Real planck_emission = 0.0_rt;
        amrex::Real rosseland_transport = 0.0_rt;
        amrex::Real scattering = 0.0_rt;
    };

    /** Device-callable interpolation view of a material opacity table.
     *
     * The stored quantities are logarithms of SI mass opacities.  Density and
     * temperature interpolation is bilinear in log space.  Photon groups are
     * selected by integer index and are never interpolated.  A query outside
     * the tabulated density/temperature domain returns NaNs so that a caller
     * cannot silently extrapolate.  Exactly zero density returns exactly zero
     * linear coefficients before any other query validation.
     * A singleton density or temperature axis is treated as constant in that
     * coordinate, without accessing a nonexistent neighboring point.
     */
    struct MaterialOpacityTableExecutor
    {
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        MaterialOpacityCoefficients operator() (
            amrex::Real mass_density, amrex::Real temperature,
            int group) const noexcept
        {
            if (mass_density == 0.0_rt) {
                return {};
            }

            if (!(mass_density > 0.0_rt) || !(temperature > 0.0_rt)
                || !amrex::Math::isfinite(mass_density)
                || !amrex::Math::isfinite(temperature)
                || group < 0 || group >= m_num_groups)
            {
                return invalid();
            }

            amrex::Real const x = std::log(mass_density);
            amrex::Real const y = std::log(temperature);
            if ((m_num_density > 1
                    && (x < m_log_density[0]
                        || x > m_log_density[m_num_density - 1]))
                || (m_num_temperature > 1
                    && (y < m_log_temperature[0]
                        || y > m_log_temperature[m_num_temperature - 1])))
            {
                return invalid();
            }

            int const ix = m_num_density == 1
                ? 0 : interval(m_log_density, m_num_density, x);
            int const iy = m_num_temperature == 1
                ? 0 : interval(m_log_temperature, m_num_temperature, y);
            amrex::Real const tx = m_num_density == 1
                ? 0.0_rt : fraction(m_log_density, ix, x);
            amrex::Real const ty = m_num_temperature == 1
                ? 0.0_rt : fraction(m_log_temperature, iy, y);

            return {
                mass_density * std::exp(interpolate(
                    m_log_planck_absorption, ix, iy, group, tx, ty)),
                mass_density * std::exp(interpolate(
                    m_log_planck_emission, ix, iy, group, tx, ty)),
                mass_density * std::exp(interpolate(
                    m_log_rosseland_transport, ix, iy, group, tx, ty)),
                mass_density * std::exp(interpolate(
                    m_log_scattering, ix, iy, group, tx, ty))};
        }

        amrex::Real const* m_log_density = nullptr;
        amrex::Real const* m_log_temperature = nullptr;
        amrex::Real const* m_log_planck_absorption = nullptr;
        amrex::Real const* m_log_planck_emission = nullptr;
        amrex::Real const* m_log_rosseland_transport = nullptr;
        amrex::Real const* m_log_scattering = nullptr;
        int m_num_density = 0;
        int m_num_temperature = 0;
        int m_num_groups = 0;

    private:
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real value (
            amrex::Real const* values, int i, int j, int group) const noexcept
        {
            return values[
                (i * m_num_temperature + j) * m_num_groups + group];
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real interpolate (
            amrex::Real const* values, int i, int j, int group,
            amrex::Real tx, amrex::Real ty) const noexcept
        {
            int const i1 = amrex::min(i + 1, m_num_density - 1);
            int const j1 = amrex::min(j + 1, m_num_temperature - 1);
            amrex::Real const v0 = value(values, i, j, group)
                + tx * (value(values, i1, j, group)
                        - value(values, i, j, group));
            amrex::Real const v1 = value(values, i, j1, group)
                + tx * (value(values, i1, j1, group)
                        - value(values, i, j1, group));
            return v0 + ty * (v1 - v0);
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        static int interval (
            amrex::Real const* axis, int size, amrex::Real value) noexcept
        {
            if (value == axis[size - 1]) { return size - 2; }
            return amrex::bisect(axis, 0, size - 1, value);
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        static amrex::Real fraction (
            amrex::Real const* axis, int index, amrex::Real value) noexcept
        {
            return (value - axis[index])
                / (axis[index + 1] - axis[index]);
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        static MaterialOpacityCoefficients invalid () noexcept
        {
            auto const nan =
                std::numeric_limits<amrex::Real>::quiet_NaN();
            return {nan, nan, nan, nan};
        }
    };

    static_assert(std::is_trivially_copyable_v<MaterialOpacityCoefficients>);
    static_assert(std::is_trivially_copyable_v<MaterialOpacityTableExecutor>);
    static_assert(sizeof(MaterialOpacityTableExecutor) <= 80U);

    /** Synchronized owner of one selected HDF5 material-opacity table.
     *
     * load() validates the complete schema before changing this object.  The
     * object owns both host and accelerator storage and must outlive all uses
     * of an executor returned from executor().
     *
     * Schema 1.0.0 uses root attributes ``format =
     * org.warpx.opacity.multigroup``, ``schema_version = 1.0.0``,
     * ``unit_system = SI``, and ``array_order = rho,temperature,group``.
     * A selected ``/materials/<material_id>`` contains fixed-LTE/completeness
     * attributes and these datasets (each numeric dataset carries the shown
     * exact ``units`` attribute):
     *
     * - ``composition/{atomic_number,atomic_mass,number_fraction}``
     *   (int32 ``1``, float64 ``kg``, float64 ``1``);
     * - ``axes/{mass_density,electron_temperature}``
     *   (float64 ``kg m-3`` and ``K``);
     * - ``groups/{edges,representative_energy}`` (float64 ``J``) and
     *   ``groups/index`` (int32 ``1``), where indices are exactly contiguous
     *   from zero and the full edge array is exactly
     *   ``[0, finite internal edges, +infinity]``;
     * - ``opacity/group/{planck_absorption,rosseland_transport,
     *   planck_scattering}``, and optional ``planck_emission``, float64
     *   ``m2 kg-1`` arrays shaped ``[density,temperature,group]`` with
     *   explicit interaction metadata.  Schema 1.0 is LTE: an explicit
     *   emission array must satisfy detailed balance with absorption within
     *   ``64*epsilon(double)*max(absorption,emission)`` entry by entry, and is
     *   then canonicalized to the absorption array at runtime.  A missing
     *   emission dataset has the same schema-1.0 Kirchhoff behavior;
     * - scalar string provenance datasets naming generator/converter versions
     *   and commits, license/redistribution/physics model/time, plus 64-digit
     *   ``generator_input_sha256`` and ``source_output_sha256`` values.
     */
    class MaterialOpacityTable
    {
    public:
        MaterialOpacityTable () = default;
        MaterialOpacityTable (
            std::string const& filename, std::int64_t material_id);

        MaterialOpacityTable (MaterialOpacityTable const&) = delete;
        MaterialOpacityTable& operator= (MaterialOpacityTable const&) = delete;
        MaterialOpacityTable (MaterialOpacityTable&&) = delete;
        MaterialOpacityTable& operator= (MaterialOpacityTable&&) = delete;

        void load (std::string const& filename, std::int64_t material_id);

        [[nodiscard]] MaterialOpacityTableExecutor executor () const noexcept;
        [[nodiscard]] bool initialized () const noexcept { return m_initialized; }
        [[nodiscard]] std::int64_t materialId () const noexcept
        {
            return m_material_id;
        }
        [[nodiscard]] std::string const& materialKey () const noexcept
        {
            return m_material_key;
        }
        [[nodiscard]] amrex::Vector<amrex::Real> const& groupEdges () const noexcept
        {
            return m_group_edges_h;
        }
        [[nodiscard]] amrex::Vector<amrex::Real> const&
        representativeEnergies () const noexcept
        {
            return m_representative_energy_h;
        }

    private:
        amrex::Gpu::HostVector<amrex::Real> m_log_density_h;
        amrex::Gpu::HostVector<amrex::Real> m_log_temperature_h;
        amrex::Gpu::HostVector<amrex::Real> m_log_planck_absorption_h;
        amrex::Gpu::HostVector<amrex::Real> m_log_planck_emission_h;
        amrex::Gpu::HostVector<amrex::Real> m_log_rosseland_transport_h;
        amrex::Gpu::HostVector<amrex::Real> m_log_scattering_h;
        amrex::Vector<amrex::Real> m_group_edges_h;
        amrex::Vector<amrex::Real> m_representative_energy_h;
#ifdef AMREX_USE_GPU
        amrex::Gpu::DeviceVector<amrex::Real> m_log_density_d;
        amrex::Gpu::DeviceVector<amrex::Real> m_log_temperature_d;
        amrex::Gpu::DeviceVector<amrex::Real> m_log_planck_absorption_d;
        amrex::Gpu::DeviceVector<amrex::Real> m_log_planck_emission_d;
        amrex::Gpu::DeviceVector<amrex::Real> m_log_rosseland_transport_d;
        amrex::Gpu::DeviceVector<amrex::Real> m_log_scattering_d;
#endif
        MaterialOpacityTableExecutor m_executor_h;
#ifdef AMREX_USE_GPU
        MaterialOpacityTableExecutor m_executor_d;
#endif
        std::int64_t m_material_id = 0;
        std::string m_material_key;
        bool m_initialized = false;
    };
}

#endif // WARPX_RADIATION_MATERIALOPACITYTABLE_H_
