/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *          S. Eric Clark (Helion Energy)
 *          Prabhat Kumar (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "HybridPICModel.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/utils/Communication.H>
#include <ablastr/warn_manager/WarnManager.H>

#include "EmbeddedBoundary/Enabled.H"
#include "Python/callbacks.H"
#include "Fields.H"
#include "Fluids/QdsmcParticleContainer.H"
#include "Particles/MultiParticleContainer.H"
#include "ExternalVectorPotential.H"
#include "QdsmcMetricTransport.H"
#include "TwoTemperatureExchange.H"
#include "Utils/MaterialRegistry.H"
#include "WarpX.H"

#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>
#include <AMReX_Random.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <vector>

using namespace amrex;
using warpx::fields::FieldType;

namespace
{
    /** Physical dual volume associated with a nodal hybrid field value. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real hybrid_node_volume (
        int const i,
        int const j,
        int const k,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
        amrex::Dim3 const domain_lo,
        amrex::Dim3 const domain_hi,
        amrex::GpuArray<int, 3> const& periodic) noexcept
    {
        amrex::Real node_volume = 1.0_rt;
#if defined(WARPX_DIM_RCYLINDER)
        amrex::Real const r = problo[0] + (i - domain_lo.x) * dx[0];
        amrex::Real const r_lo = amrex::max(problo[0], r - 0.5_rt * dx[0]);
        amrex::Real const r_hi = amrex::min(probhi[0], r + 0.5_rt * dx[0]);
        node_volume = MathConst::pi * (r_hi * r_hi - r_lo * r_lo);
#elif defined(WARPX_DIM_RSPHERE)
        amrex::Real const r = problo[0] + (i - domain_lo.x) * dx[0];
        amrex::Real const r_lo = amrex::max(problo[0], r - 0.5_rt * dx[0]);
        amrex::Real const r_hi = amrex::min(probhi[0], r + 0.5_rt * dx[0]);
        node_volume = 4.0_rt / 3.0_rt * MathConst::pi
            * (r_hi * r_hi * r_hi - r_lo * r_lo * r_lo);
#elif defined(WARPX_DIM_RZ)
        amrex::Real const r = problo[0] + (i - domain_lo.x) * dx[0];
        amrex::Real const r_lo = amrex::max(problo[0], r - 0.5_rt * dx[0]);
        amrex::Real const r_hi = amrex::min(probhi[0], r + 0.5_rt * dx[0]);
        amrex::Real const dz =
            periodic[1] || (j != domain_lo.y && j != domain_hi.y + 1)
            ? dx[1] : 0.5_rt * dx[1];
        node_volume = MathConst::pi * (r_hi * r_hi - r_lo * r_lo) * dz;
#else
        node_volume *=
            periodic[0] || (i != domain_lo.x && i != domain_hi.x + 1)
            ? dx[0] : 0.5_rt * dx[0];
#if AMREX_SPACEDIM >= 2
        node_volume *=
            periodic[1] || (j != domain_lo.y && j != domain_hi.y + 1)
            ? dx[1] : 0.5_rt * dx[1];
#endif
#if AMREX_SPACEDIM == 3
        node_volume *=
            periodic[2] || (k != domain_lo.z && k != domain_hi.z + 1)
            ? dx[2] : 0.5_rt * dx[2];
#endif
#endif
        amrex::ignore_unused(
            j, k, problo, probhi, domain_lo, domain_hi, periodic);
        return node_volume;
    }

    constexpr int nonlinear_lte_temperature_increment_comp = 0;
    constexpr int nonlinear_lte_represented_energy_comp = 1;
    constexpr int nonlinear_lte_remap_components = 2;

#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
    /** Volume consistent with a radially inverse-volume-scaled nodal charge
     * deposit.  Unlike hybrid_node_volume, this keeps the full exterior
     * half-cell at a non-periodic radial face because the scaled deposit
     * represents a full-cell density even when that face is a wall. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real hybrid_transport_node_volume (
        int const i,
        int const j,
        int const k,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
        amrex::Dim3 const domain_lo,
        amrex::Real const axis_volume_factor) noexcept
    {
#if defined(WARPX_DIM_RCYLINDER)
        amrex::Real const r =
            problo[0] + (i - domain_lo.x) * dx[0];
        amrex::ignore_unused(j, k);
        return warpx::hybrid::cylindricalTransportNodeVolume(
            r, dx[0], 1.0_rt, axis_volume_factor);
#elif defined(WARPX_DIM_RZ)
        amrex::Real const r =
            problo[0] + (i - domain_lo.x) * dx[0];
        amrex::ignore_unused(j, k);
        return warpx::hybrid::cylindricalTransportNodeVolume(
            r, dx[0], dx[1], axis_volume_factor);
#else
        amrex::ignore_unused(i, j, k, problo, dx, domain_lo, axis_volume_factor);
        return std::numeric_limits<amrex::Real>::quiet_NaN();
#endif
    }
#endif

    /** Physical part of one cell represented by one nodal corner. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real hybrid_cell_corner_volume (
        int const i,
        int const j,
        int const k,
        int const di,
        int const dj,
        int const dk,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
        amrex::Dim3 const domain_lo) noexcept
    {
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE) \
    || defined(WARPX_DIM_RZ)
        amrex::Real const r_lo =
            problo[0] + (i - domain_lo.x) * dx[0];
        amrex::Real const r_mid = r_lo + 0.5_rt * dx[0];
        amrex::Real const r_hi = r_lo + dx[0];
        amrex::Real const corner_r_lo = di == 0 ? r_lo : r_mid;
        amrex::Real const corner_r_hi = di == 0 ? r_mid : r_hi;
#endif
#if defined(WARPX_DIM_RCYLINDER)
        amrex::ignore_unused(j, k, dj, dk);
        return MathConst::pi
            * (corner_r_hi * corner_r_hi - corner_r_lo * corner_r_lo);
#elif defined(WARPX_DIM_RSPHERE)
        amrex::ignore_unused(j, k, dj, dk);
        return 4.0_rt / 3.0_rt * MathConst::pi
            * (corner_r_hi * corner_r_hi * corner_r_hi
               - corner_r_lo * corner_r_lo * corner_r_lo);
#elif defined(WARPX_DIM_RZ)
        // Uniform dz makes the two z-halves equal, so dj does not enter.
        amrex::ignore_unused(j, k, dj, dk);
        return MathConst::pi
            * (corner_r_hi * corner_r_hi - corner_r_lo * corner_r_lo)
            * 0.5_rt * dx[1];
#else
        amrex::ignore_unused(i, j, k, di, dj, dk, problo, domain_lo);
        return AMREX_D_TERM(
            0.5_rt * dx[0], * 0.5_rt * dx[1], * 0.5_rt * dx[2]);
#endif
    }

    /** Physical volume of one cell-centered kinetic-ion moment bin. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real hybrid_cell_volume (
        int const i,
        int const j,
        int const k,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
        amrex::Dim3 const domain_lo) noexcept
    {
#if defined(WARPX_DIM_RCYLINDER)
        amrex::Real const r_lo =
            problo[0] + (i - domain_lo.x) * dx[0];
        amrex::Real const r_hi = r_lo + dx[0];
        amrex::ignore_unused(j, k);
        return MathConst::pi * (r_hi * r_hi - r_lo * r_lo);
#elif defined(WARPX_DIM_RZ)
        amrex::Real const r_lo =
            problo[0] + (i - domain_lo.x) * dx[0];
        amrex::Real const r_hi = r_lo + dx[0];
        amrex::ignore_unused(j, k);
        return MathConst::pi * (r_hi * r_hi - r_lo * r_lo) * dx[1];
#elif defined(WARPX_DIM_RSPHERE)
        amrex::Real const r_lo =
            problo[0] + (i - domain_lo.x) * dx[0];
        amrex::Real const r_hi = r_lo + dx[0];
        amrex::ignore_unused(j, k);
        return 4.0_rt / 3.0_rt * MathConst::pi
            * (r_hi * r_hi * r_hi - r_lo * r_lo * r_lo);
#else
        amrex::ignore_unused(i, j, k, problo, domain_lo);
#if AMREX_SPACEDIM == 1
        return dx[0];
#elif AMREX_SPACEDIM == 2
        return dx[0] * dx[1];
#else
        return dx[0] * dx[1] * dx[2];
#endif
#endif
    }

    /** Whether a cell neighbor of a node is physical or periodic. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool hybrid_source_cell_is_addressable (
        int const i,
        int const j,
        int const k,
        amrex::Dim3 const domain_lo,
        amrex::Dim3 const domain_hi,
        amrex::GpuArray<int, 3> const& periodic) noexcept
    {
        bool const addressable =
            (periodic[0] || (i >= domain_lo.x && i <= domain_hi.x))
#if AMREX_SPACEDIM >= 2
            && (periodic[1] || (j >= domain_lo.y && j <= domain_hi.y))
#endif
#if AMREX_SPACEDIM == 3
            && (periodic[2] || (k >= domain_lo.z && k <= domain_hi.z))
#endif
            ;
        amrex::ignore_unused(j, k);
        return addressable;
    }

    /** Nonlinear LTE contribution from one adjacent cell to one node. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real hybrid_nonlinear_lte_node_energy (
        int const i,
        int const j,
        int const k,
        int const ci,
        int const cj,
        int const ck,
        amrex::Array4<amrex::Real const> const& source,
        amrex::Array4<amrex::Real const> const& remap,
        amrex::Array4<amrex::Real const> const& thermodynamic_state,
        amrex::Array4<amrex::Real const> const& rho,
        amrex::Array4<amrex::Real const> const& temperature,
        ElectronThermodynamicsExecutor::MaterialChargeDensityArrays const&
            material_charge_density,
        ElectronThermodynamicsExecutor const thermodynamics,
        amrex::Real const material_rho_threshold,
        amrex::Real const thermodynamic_rho_floor,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
        amrex::Dim3 const domain_lo,
        amrex::Dim3 const domain_hi,
        amrex::GpuArray<int, 3> const& periodic,
        bool& valid) noexcept
    {
        if (!hybrid_source_cell_is_addressable(
                ci, cj, ck, domain_lo, domain_hi, periodic))
        {
            return 0.0_rt;
        }
        amrex::Real const cell_energy = source(ci, cj, ck);
        amrex::Real const temperature_increment = remap(
            ci, cj, ck, nonlinear_lte_temperature_increment_comp);
        amrex::Real const represented_energy = remap(
            ci, cj, ck, nonlinear_lte_represented_energy_comp);
        if (!amrex::Math::isfinite(cell_energy)
            || !amrex::Math::isfinite(temperature_increment)
            || !amrex::Math::isfinite(represented_energy))
        {
            valid = false;
            return 0.0_rt;
        }
        if (cell_energy == 0.0_rt && temperature_increment == 0.0_rt
            && represented_energy == 0.0_rt)
        {
            return 0.0_rt;
        }

        int const di = i - ci;
        int const dj = AMREX_SPACEDIM >= 2 ? j - cj : 0;
        int const dk = AMREX_SPACEDIM == 3 ? k - ck : 0;
        if (di < 0 || di > 1 || dj < 0 || dj > 1 || dk < 0 || dk > 1) {
            valid = false;
            return 0.0_rt;
        }
        amrex::Real const node_corner_volume = hybrid_cell_corner_volume(
            ci, cj, ck, di, dj, dk, problo, dx, domain_lo);
        amrex::Real const node_capacity =
            thermodynamic_state(i, j, k, 1) * node_corner_volume;
        amrex::Real cell_capacity = 0.0_rt;
        for (int corner_dk = 0;
             corner_dk < (AMREX_SPACEDIM == 3 ? 2 : 1); ++corner_dk)
        {
            for (int corner_dj = 0;
                 corner_dj < (AMREX_SPACEDIM >= 2 ? 2 : 1); ++corner_dj)
            {
                for (int corner_di = 0; corner_di < 2; ++corner_di) {
                    int const ni = ci + corner_di;
                    int const nj = cj + corner_dj;
                    int const nk = ck + corner_dk;
                    if (rho(ni, nj, nk) <= material_rho_threshold) {
                        continue;
                    }
                    amrex::Real const corner_volume =
                        hybrid_cell_corner_volume(
                            ci, cj, ck, corner_di, corner_dj, corner_dk,
                            problo, dx, domain_lo);
                    amrex::Real const corner_capacity =
                        thermodynamic_state(ni, nj, nk, 1) * corner_volume;
                    if (!(corner_capacity > 0.0_rt)
                        || !amrex::Math::isfinite(corner_capacity))
                    {
                        valid = false;
                        return 0.0_rt;
                    }
                    cell_capacity += corner_capacity;
                }
            }
        }
        if (!(node_corner_volume > 0.0_rt)
            || !(node_capacity > 0.0_rt)
            || !(cell_capacity > 0.0_rt)
            || !amrex::Math::isfinite(node_corner_volume)
            || !amrex::Math::isfinite(node_capacity)
            || !amrex::Math::isfinite(cell_capacity))
        {
            valid = false;
            return 0.0_rt;
        }

        amrex::Real const trial_temperature =
            temperature(i, j, k) + temperature_increment;
        auto const material_mass_density = thermodynamics
            .materialMassDensitiesFromChargeDensityArrays(
                material_charge_density, i, j, k);
        ElectronThermodynamicState const trial_state = thermodynamics
            .stateFromMaterialMassDensitiesTemperature(
                amrex::max(rho(i, j, k), thermodynamic_rho_floor),
                material_mass_density, trial_temperature);
        amrex::Real const old_internal_energy_density =
            thermodynamic_state(i, j, k, 0);
        amrex::Real const represented_node_energy =
            (trial_state.internal_energy_density - old_internal_energy_density)
            * node_corner_volume;
        amrex::Real const residual_energy =
            cell_energy - represented_energy;
        amrex::Real const node_energy = represented_node_energy
            + residual_energy * node_capacity / cell_capacity;
        if (trial_temperature < thermodynamics.minimumTemperature()
            || trial_temperature > thermodynamics.maximumTemperature()
            || !amrex::Math::isfinite(trial_temperature)
            || trial_state.pressure < 0.0_rt
            || !(trial_state.heat_capacity_density > 0.0_rt)
            || !amrex::Math::isfinite(trial_state.pressure)
            || !amrex::Math::isfinite(
                trial_state.internal_energy_density)
            || !amrex::Math::isfinite(
                trial_state.heat_capacity_density)
            || !amrex::Math::isfinite(represented_node_energy)
            || !amrex::Math::isfinite(residual_energy)
            || !amrex::Math::isfinite(node_energy))
        {
            valid = false;
            return 0.0_rt;
        }
        return node_energy;
    }

    /** Constant-LTE contribution from one adjacent cell to one node. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real hybrid_constant_lte_node_energy (
        int const i,
        int const j,
        int const k,
        int const ci,
        int const cj,
        int const ck,
        amrex::Array4<amrex::Real const> const& source,
        amrex::Array4<amrex::Real const> const& thermodynamic_state,
        amrex::Array4<amrex::Real const> const& rho,
        amrex::Real const material_rho_threshold,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
        amrex::Dim3 const domain_lo,
        amrex::Dim3 const domain_hi,
        amrex::GpuArray<int, 3> const& periodic,
        bool& valid) noexcept
    {
        if (!hybrid_source_cell_is_addressable(
                ci, cj, ck, domain_lo, domain_hi, periodic))
        {
            return 0.0_rt;
        }
        amrex::Real const cell_energy = source(ci, cj, ck);
        if (!amrex::Math::isfinite(cell_energy)) {
            valid = false;
            return 0.0_rt;
        }
        if (cell_energy == 0.0_rt) { return 0.0_rt; }

        int const di = i - ci;
        int const dj = AMREX_SPACEDIM >= 2 ? j - cj : 0;
        int const dk = AMREX_SPACEDIM == 3 ? k - ck : 0;
        if (di < 0 || di > 1 || dj < 0 || dj > 1 || dk < 0 || dk > 1) {
            valid = false;
            return 0.0_rt;
        }
        amrex::Real const node_corner_volume = hybrid_cell_corner_volume(
            ci, cj, ck, di, dj, dk, problo, dx, domain_lo);
        amrex::Real const node_capacity =
            thermodynamic_state(i, j, k, 1) * node_corner_volume;
        amrex::Real cell_capacity = 0.0_rt;
        for (int corner_dk = 0;
             corner_dk < (AMREX_SPACEDIM == 3 ? 2 : 1); ++corner_dk)
        {
            for (int corner_dj = 0;
                 corner_dj < (AMREX_SPACEDIM >= 2 ? 2 : 1); ++corner_dj)
            {
                for (int corner_di = 0; corner_di < 2; ++corner_di) {
                    int const ni = ci + corner_di;
                    int const nj = cj + corner_dj;
                    int const nk = ck + corner_dk;
                    if (rho(ni, nj, nk) <= material_rho_threshold) {
                        continue;
                    }
                    amrex::Real const corner_volume =
                        hybrid_cell_corner_volume(
                            ci, cj, ck, corner_di, corner_dj, corner_dk,
                            problo, dx, domain_lo);
                    amrex::Real const corner_capacity =
                        thermodynamic_state(ni, nj, nk, 1) * corner_volume;
                    if (!(corner_capacity > 0.0_rt)
                        || !amrex::Math::isfinite(corner_capacity))
                    {
                        valid = false;
                        return 0.0_rt;
                    }
                    cell_capacity += corner_capacity;
                }
            }
        }
        if (!(node_corner_volume > 0.0_rt)
            || !(node_capacity > 0.0_rt)
            || !(cell_capacity > 0.0_rt)
            || !amrex::Math::isfinite(node_corner_volume)
            || !amrex::Math::isfinite(node_capacity)
            || !amrex::Math::isfinite(cell_capacity))
        {
            valid = false;
            return 0.0_rt;
        }
        amrex::Real const node_energy =
            cell_energy * node_capacity / cell_capacity;
        if (!amrex::Math::isfinite(node_energy)) {
            valid = false;
            return 0.0_rt;
        }
        return node_energy;
    }

    struct QdsmcCartesianTransportTerms
    {
        amrex::Real energy_flux_divergence = 0.0_rt;
        amrex::Real charge_flux_divergence = 0.0_rt;
        amrex::Real velocity_divergence = 0.0_rt;
        amrex::Real outgoing_charge_flux_per_volume = 0.0_rt;
        amrex::Real absolute_energy_flux_per_volume = 0.0_rt;
        amrex::Real absolute_charge_flux_per_volume = 0.0_rt;
        amrex::GpuArray<amrex::Real, 6>
        outgoing_face_charge_flux_per_volume{};
    };

    struct QdsmcVelocityMetric
    {
        bool radial = false;
        amrex::Real left_radius = 0.0_rt;
        amrex::Real center_radius = 0.0_rt;
        amrex::Real right_radius = 0.0_rt;
        amrex::Real low_face_radius = 0.0_rt;
        amrex::Real high_face_radius = 0.0_rt;
    };

#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) \
    || defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_RCYLINDER) \
    || defined(WARPX_DIM_RZ)
    /** Mass-consistent first-order fluxes through one direction of a nodal
     * dual control volume.  The electron charge flux is
     *
     *     F_rho = J_i - J_plasma = rho_e V_e,
     *
     * on the staggered current face.  The internal-energy flux carries the
     * donor specific energy, F_U = F_rho (U/rho)_upwind.  Thus charge and
     * internal energy use exactly the same transported amount, including at
     * a material/vacuum front.  At a non-periodic domain edge the exterior
     * flux and face velocity are zero, matching the existing impermeable
     * energy boundary; the boundary control volume has half the interior
     * width. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    QdsmcCartesianTransportTerms qdsmc_nodal_direction_terms (
        amrex::Array4<amrex::Real const> const& energy,
        amrex::Array4<amrex::Real const> const& old_charge_density,
        amrex::Array4<amrex::Real const> const& midpoint_charge_density,
        amrex::Array4<amrex::Real const> const& ion_current,
        amrex::Array4<amrex::Real const> const& plasma_current,
        amrex::Array4<amrex::Real const> const& nodal_velocity,
        int const i, int const j, int const k,
        int const direction, int const lo, int const hi,
        bool const periodic,
        bool const plasma_current_is_face_centered,
        amrex::Real const face_area_lo,
        amrex::Real const face_area_hi,
        amrex::Real const inv_control_volume,
        QdsmcVelocityMetric const& velocity_metric) noexcept
    {
        int const di = direction == 0 ? 1 : 0;
        int const dj = direction == 1 ? 1 : 0;
        int const dk = direction == 2 ? 1 : 0;
        int const index = direction == 0 ? i : (direction == 1 ? j : k);
        bool const low_wall = !periodic && index == lo;
        bool const high_wall = !periodic && index == hi;

        amrex::Real const u_center = energy(i, j, k);
        amrex::Real const rho_center = old_charge_density(i, j, k);
        amrex::Real velocity_lo = 0.0_rt;
        amrex::Real velocity_hi = 0.0_rt;
        amrex::Real charge_flux_lo = 0.0_rt;
        amrex::Real charge_flux_hi = 0.0_rt;
        amrex::Real flux_lo = 0.0_rt;
        amrex::Real flux_hi = 0.0_rt;
        if (!low_wall) {
            amrex::Real const u_left = energy(i-di, j-dj, k-dk);
            amrex::Real const rho_left =
                old_charge_density(i-di, j-dj, k-dk);
            // A current component is cell centered in its own direction and
            // nodal in the transverse directions.  Index (i-di,j-dj,k-dk)
            // therefore labels the low face of nodal control volume (i,j,k).
            amrex::Real const plasma_flux_lo =
                plasma_current_is_face_centered
                ? plasma_current(i-di, j-dj, k-dk)
                : 0.5_rt * (
                    plasma_current(i-di, j-dj, k-dk)
                    + plasma_current(i, j, k));
            charge_flux_lo =
                ion_current(i-di, j-dj, k-dk) - plasma_flux_lo;
            amrex::Real const donor_density = charge_flux_lo >= 0.0_rt
                ? rho_left : rho_center;
            amrex::Real const donor_energy = charge_flux_lo >= 0.0_rt
                ? u_left : u_center;
            if (donor_density > 0.0_rt) {
                flux_lo = charge_flux_lo * donor_energy / donor_density;
            } else if (charge_flux_lo != 0.0_rt) {
                flux_lo = std::numeric_limits<amrex::Real>::quiet_NaN();
            }
            amrex::Real const rho_mid_left =
                midpoint_charge_density(i-di, j-dj, k-dk);
            amrex::Real const rho_mid_center =
                midpoint_charge_density(i, j, k);
            bool const left_has_material =
                rho_mid_left > 0.0_rt;
            bool const center_has_material =
                rho_mid_center > 0.0_rt;
            velocity_lo = left_has_material && center_has_material
                ? (velocity_metric.radial
                    ? warpx::hybrid::cylindricalRadialFaceVelocity(
                        nodal_velocity(i-di, j-dj, k-dk),
                        nodal_velocity(i, j, k),
                        velocity_metric.left_radius,
                        velocity_metric.center_radius,
                        velocity_metric.low_face_radius)
                    : 0.5_rt * (
                        nodal_velocity(i-di, j-dj, k-dk)
                        + nodal_velocity(i, j, k)))
                : (left_has_material
                    ? nodal_velocity(i-di, j-dj, k-dk)
                    : (center_has_material
                        ? nodal_velocity(i, j, k) : 0.0_rt));
        }
        if (!high_wall) {
            amrex::Real const u_right = energy(i+di, j+dj, k+dk);
            amrex::Real const rho_right =
                old_charge_density(i+di, j+dj, k+dk);
            // Index (i,j,k) labels the high face of this nodal control volume.
            amrex::Real const plasma_flux_hi =
                plasma_current_is_face_centered
                ? plasma_current(i, j, k)
                : 0.5_rt * (
                    plasma_current(i, j, k)
                    + plasma_current(i+di, j+dj, k+dk));
            charge_flux_hi = ion_current(i, j, k) - plasma_flux_hi;
            amrex::Real const donor_density = charge_flux_hi >= 0.0_rt
                ? rho_center : rho_right;
            amrex::Real const donor_energy = charge_flux_hi >= 0.0_rt
                ? u_center : u_right;
            if (donor_density > 0.0_rt) {
                flux_hi = charge_flux_hi * donor_energy / donor_density;
            } else if (charge_flux_hi != 0.0_rt) {
                flux_hi = std::numeric_limits<amrex::Real>::quiet_NaN();
            }
            amrex::Real const rho_mid_center =
                midpoint_charge_density(i, j, k);
            amrex::Real const rho_mid_right =
                midpoint_charge_density(i+di, j+dj, k+dk);
            bool const center_has_material =
                rho_mid_center > 0.0_rt;
            bool const right_has_material =
                rho_mid_right > 0.0_rt;
            velocity_hi = center_has_material && right_has_material
                ? (velocity_metric.radial
                    ? warpx::hybrid::cylindricalRadialFaceVelocity(
                        nodal_velocity(i, j, k),
                        nodal_velocity(i+di, j+dj, k+dk),
                        velocity_metric.center_radius,
                        velocity_metric.right_radius,
                        velocity_metric.high_face_radius)
                    : 0.5_rt * (
                        nodal_velocity(i, j, k)
                        + nodal_velocity(i+di, j+dj, k+dk)))
                : (center_has_material
                    ? nodal_velocity(i, j, k)
                    : (right_has_material
                        ? nodal_velocity(i+di, j+dj, k+dk) : 0.0_rt));
        }

        QdsmcCartesianTransportTerms result;
        result.energy_flux_divergence =
            warpx::hybrid::metricFluxDivergence(
                flux_lo, flux_hi, face_area_lo, face_area_hi,
                inv_control_volume);
        result.charge_flux_divergence =
            warpx::hybrid::metricFluxDivergence(
                charge_flux_lo, charge_flux_hi, face_area_lo, face_area_hi,
                inv_control_volume);
        result.velocity_divergence =
            warpx::hybrid::metricFluxDivergence(
                velocity_lo, velocity_hi, face_area_lo, face_area_hi,
                inv_control_volume);
        result.outgoing_charge_flux_per_volume = (
            amrex::max(charge_flux_hi, 0.0_rt) * face_area_hi
            + amrex::max(-charge_flux_lo, 0.0_rt) * face_area_lo)
            * inv_control_volume;
        result.absolute_energy_flux_per_volume =
            (std::abs(flux_lo) * face_area_lo
            + std::abs(flux_hi) * face_area_hi) * inv_control_volume;
        result.absolute_charge_flux_per_volume =
            (std::abs(charge_flux_lo) * face_area_lo
            + std::abs(charge_flux_hi) * face_area_hi) * inv_control_volume;
        result.outgoing_face_charge_flux_per_volume[2 * direction] =
            amrex::max(-charge_flux_lo, 0.0_rt) * face_area_lo
            * inv_control_volume;
        result.outgoing_face_charge_flux_per_volume[2 * direction + 1] =
            amrex::max(charge_flux_hi, 0.0_rt) * face_area_hi
            * inv_control_volume;
        return result;
    }
#endif

#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
    /** Metric-aware transport through the radial and axial faces of a
     * cylindrical nodal control volume.  The caller supplies physical current
     * densities; multiplying by these face areas and dividing by the nodal
     * volume gives the correct cylindrical divergence. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    QdsmcCartesianTransportTerms qdsmc_cylindrical_transport_terms (
        amrex::Array4<amrex::Real const> const& energy,
        amrex::Array4<amrex::Real const> const& old_charge_density,
        amrex::Array4<amrex::Real const> const& midpoint_charge_density,
        amrex::Array4<amrex::Real const> const& ion_current_x,
        amrex::Array4<amrex::Real const> const& ion_current_z,
        amrex::Array4<amrex::Real const> const& plasma_current_x,
        amrex::Array4<amrex::Real const> const& plasma_current_z,
        amrex::Array4<amrex::Real const> const& vr,
        amrex::Array4<amrex::Real const> const& vz,
        int const i, int const j, int const k,
        amrex::Dim3 const nodal_lo, amrex::Dim3 const nodal_hi,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
        amrex::GpuArray<int, 3> const& periodic,
        amrex::GpuArray<int, 3> const& plasma_current_face_centered,
        amrex::Real const axis_volume_factor,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi,
        amrex::Dim3 const physical_domain_lo,
        amrex::Dim3 const physical_domain_hi) noexcept
    {
        amrex::ignore_unused(probhi, physical_domain_hi);
        amrex::Real const r =
            problo[0] + (i - physical_domain_lo.x) * dx[0];
        amrex::Real const r_lo =
            amrex::max(problo[0], r - 0.5_rt * dx[0]);
        amrex::Real const r_hi = r + 0.5_rt * dx[0];
        amrex::Real const node_volume = hybrid_transport_node_volume(
            i, j, k, problo, dx, physical_domain_lo, axis_volume_factor);
        QdsmcVelocityMetric const radial_velocity_metric{
            true, r - dx[0], r, r + dx[0], r_lo, r_hi};

#ifdef WARPX_DIM_RCYLINDER
        amrex::ignore_unused(
            ion_current_z, plasma_current_z, vz, nodal_lo, nodal_hi,
            periodic);
        return qdsmc_nodal_direction_terms(
            energy, old_charge_density, midpoint_charge_density,
            ion_current_x, plasma_current_x, vr,
            i, j, k, 0, nodal_lo.x, nodal_hi.x,
            periodic[0] != 0,
            plasma_current_face_centered[0] != 0,
            warpx::hybrid::cylindricalRadialFaceArea(r_lo, 1.0_rt),
            warpx::hybrid::cylindricalRadialFaceArea(r_hi, 1.0_rt),
            1.0_rt / node_volume, radial_velocity_metric);
#else
        amrex::Real const radial_face_area =
            warpx::hybrid::cylindricalAxialFaceArea(r_lo, r_hi);
        QdsmcCartesianTransportTerms const radial = qdsmc_nodal_direction_terms(
            energy, old_charge_density, midpoint_charge_density,
            ion_current_x, plasma_current_x, vr,
            i, j, k, 0, nodal_lo.x, nodal_hi.x,
            periodic[0] != 0,
            plasma_current_face_centered[0] != 0,
            warpx::hybrid::cylindricalRadialFaceArea(r_lo, dx[1]),
            warpx::hybrid::cylindricalRadialFaceArea(r_hi, dx[1]),
            1.0_rt / node_volume, radial_velocity_metric);
        QdsmcCartesianTransportTerms const axial = qdsmc_nodal_direction_terms(
            energy, old_charge_density, midpoint_charge_density,
            ion_current_z, plasma_current_z, vz,
            i, j, k, 1, nodal_lo.y, nodal_hi.y,
            periodic[1] != 0,
            plasma_current_face_centered[1] != 0,
            radial_face_area, radial_face_area, 1.0_rt / node_volume,
            QdsmcVelocityMetric{});
        QdsmcCartesianTransportTerms result;
        result.energy_flux_divergence =
            radial.energy_flux_divergence + axial.energy_flux_divergence;
        result.charge_flux_divergence =
            radial.charge_flux_divergence + axial.charge_flux_divergence;
        result.velocity_divergence =
            radial.velocity_divergence + axial.velocity_divergence;
        result.outgoing_charge_flux_per_volume =
            radial.outgoing_charge_flux_per_volume
            + axial.outgoing_charge_flux_per_volume;
        result.absolute_energy_flux_per_volume =
            radial.absolute_energy_flux_per_volume
            + axial.absolute_energy_flux_per_volume;
        result.absolute_charge_flux_per_volume =
            radial.absolute_charge_flux_per_volume
            + axial.absolute_charge_flux_per_volume;
        for (int face = 0; face < 4; ++face) {
            result.outgoing_face_charge_flux_per_volume[face] =
                face < 2
                    ? radial.outgoing_face_charge_flux_per_volume[face]
                    : axial.outgoing_face_charge_flux_per_volume[face];
        }
        return result;
#endif
    }
#endif

#if !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RZ)
    /** Conservative Cartesian finite-volume transport terms for
     * dU/dt + div(U V_e) = -P_e div(V_e). Energy advection uses the
     * charge-conserving mass flux, while pressure work uses the separately
     * co-deposited physical velocity so rigid translation has exactly zero
     * discrete compression. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    QdsmcCartesianTransportTerms qdsmc_cartesian_transport_terms (
        amrex::Array4<amrex::Real const> const& energy,
        amrex::Array4<amrex::Real const> const& old_charge_density,
        amrex::Array4<amrex::Real const> const& midpoint_charge_density,
        amrex::Array4<amrex::Real const> const& ion_current_x,
        amrex::Array4<amrex::Real const> const& ion_current_y,
        amrex::Array4<amrex::Real const> const& ion_current_z,
        amrex::Array4<amrex::Real const> const& plasma_current_x,
        amrex::Array4<amrex::Real const> const& plasma_current_y,
        amrex::Array4<amrex::Real const> const& plasma_current_z,
        amrex::Array4<amrex::Real const> const& vx,
        amrex::Array4<amrex::Real const> const& vy,
        amrex::Array4<amrex::Real const> const& vz,
        int const i, int const j, int const k,
        amrex::Dim3 const nodal_lo, amrex::Dim3 const nodal_hi,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& inv_dx,
        amrex::GpuArray<int, 3> const& periodic,
        amrex::GpuArray<int, 3> const& plasma_current_face_centered) noexcept
    {
        amrex::ignore_unused(
            ion_current_x, ion_current_y, ion_current_z,
            plasma_current_x, plasma_current_y, plasma_current_z,
            vx, vy, vz);
        QdsmcCartesianTransportTerms result;
#if defined(WARPX_DIM_3D)
        auto const tx = qdsmc_nodal_direction_terms(
            energy, old_charge_density, midpoint_charge_density,
            ion_current_x, plasma_current_x, vx,
            i, j, k, 0, nodal_lo.x, nodal_hi.x,
            periodic[0] != 0,
            plasma_current_face_centered[0] != 0, 1.0_rt, 1.0_rt,
            inv_dx[0] * ((!periodic[0] && (i == nodal_lo.x
                || i == nodal_hi.x)) ? 2.0_rt : 1.0_rt),
            QdsmcVelocityMetric{});
        auto const ty = qdsmc_nodal_direction_terms(
            energy, old_charge_density, midpoint_charge_density,
            ion_current_y, plasma_current_y, vy,
            i, j, k, 1, nodal_lo.y, nodal_hi.y,
            periodic[1] != 0,
            plasma_current_face_centered[1] != 0, 1.0_rt, 1.0_rt,
            inv_dx[1] * ((!periodic[1] && (j == nodal_lo.y
                || j == nodal_hi.y)) ? 2.0_rt : 1.0_rt),
            QdsmcVelocityMetric{});
        auto const tz = qdsmc_nodal_direction_terms(
            energy, old_charge_density, midpoint_charge_density,
            ion_current_z, plasma_current_z, vz,
            i, j, k, 2, nodal_lo.z, nodal_hi.z,
            periodic[2] != 0,
            plasma_current_face_centered[2] != 0, 1.0_rt, 1.0_rt,
            inv_dx[2] * ((!periodic[2] && (k == nodal_lo.z
                || k == nodal_hi.z)) ? 2.0_rt : 1.0_rt),
            QdsmcVelocityMetric{});
        result.energy_flux_divergence = tx.energy_flux_divergence
            + ty.energy_flux_divergence + tz.energy_flux_divergence;
        result.charge_flux_divergence = tx.charge_flux_divergence
            + ty.charge_flux_divergence + tz.charge_flux_divergence;
        result.velocity_divergence = tx.velocity_divergence
            + ty.velocity_divergence + tz.velocity_divergence;
        result.outgoing_charge_flux_per_volume =
            tx.outgoing_charge_flux_per_volume
            + ty.outgoing_charge_flux_per_volume
            + tz.outgoing_charge_flux_per_volume;
        result.absolute_energy_flux_per_volume =
            tx.absolute_energy_flux_per_volume
            + ty.absolute_energy_flux_per_volume
            + tz.absolute_energy_flux_per_volume;
        result.absolute_charge_flux_per_volume =
            tx.absolute_charge_flux_per_volume
            + ty.absolute_charge_flux_per_volume
            + tz.absolute_charge_flux_per_volume;
        for (int face = 0; face < 6; ++face) {
            result.outgoing_face_charge_flux_per_volume[face] =
                tx.outgoing_face_charge_flux_per_volume[face]
                + ty.outgoing_face_charge_flux_per_volume[face]
                + tz.outgoing_face_charge_flux_per_volume[face];
        }
#elif defined(WARPX_DIM_XZ)
        auto const tx = qdsmc_nodal_direction_terms(
            energy, old_charge_density, midpoint_charge_density,
            ion_current_x, plasma_current_x, vx,
            i, j, k, 0, nodal_lo.x, nodal_hi.x,
            periodic[0] != 0,
            plasma_current_face_centered[0] != 0, 1.0_rt, 1.0_rt,
            inv_dx[0] * ((!periodic[0] && (i == nodal_lo.x
                || i == nodal_hi.x)) ? 2.0_rt : 1.0_rt),
            QdsmcVelocityMetric{});
        auto const tz = qdsmc_nodal_direction_terms(
            energy, old_charge_density, midpoint_charge_density,
            ion_current_z, plasma_current_z, vz,
            i, j, k, 1, nodal_lo.y, nodal_hi.y,
            periodic[1] != 0,
            plasma_current_face_centered[1] != 0, 1.0_rt, 1.0_rt,
            inv_dx[1] * ((!periodic[1] && (j == nodal_lo.y
                || j == nodal_hi.y)) ? 2.0_rt : 1.0_rt),
            QdsmcVelocityMetric{});
        result.energy_flux_divergence =
            tx.energy_flux_divergence + tz.energy_flux_divergence;
        result.charge_flux_divergence =
            tx.charge_flux_divergence + tz.charge_flux_divergence;
        result.velocity_divergence =
            tx.velocity_divergence + tz.velocity_divergence;
        result.outgoing_charge_flux_per_volume =
            tx.outgoing_charge_flux_per_volume
            + tz.outgoing_charge_flux_per_volume;
        result.absolute_energy_flux_per_volume =
            tx.absolute_energy_flux_per_volume
            + tz.absolute_energy_flux_per_volume;
        result.absolute_charge_flux_per_volume =
            tx.absolute_charge_flux_per_volume
            + tz.absolute_charge_flux_per_volume;
        for (int face = 0; face < 6; ++face) {
            result.outgoing_face_charge_flux_per_volume[face] =
                tx.outgoing_face_charge_flux_per_volume[face]
                + tz.outgoing_face_charge_flux_per_volume[face];
        }
#elif defined(WARPX_DIM_1D_Z)
        result = qdsmc_nodal_direction_terms(
            energy, old_charge_density, midpoint_charge_density,
            ion_current_z, plasma_current_z, vz,
            i, j, k, 0, nodal_lo.x, nodal_hi.x,
            periodic[0] != 0,
            plasma_current_face_centered[0] != 0, 1.0_rt, 1.0_rt,
            inv_dx[0] * ((!periodic[0] && (i == nodal_lo.x
                || i == nodal_hi.x)) ? 2.0_rt : 1.0_rt),
            QdsmcVelocityMetric{});
#else
        amrex::ignore_unused(
            energy, old_charge_density, midpoint_charge_density,
            ion_current_x, ion_current_y,
            ion_current_z, plasma_current_x, plasma_current_y,
            plasma_current_z, vx, vy, vz, i, j, k, nodal_lo, nodal_hi,
            inv_dx, periodic,
            plasma_current_face_centered);
        result.energy_flux_divergence =
            std::numeric_limits<amrex::Real>::quiet_NaN();
        result.charge_flux_divergence =
            std::numeric_limits<amrex::Real>::quiet_NaN();
        result.velocity_divergence =
            std::numeric_limits<amrex::Real>::quiet_NaN();
        result.outgoing_charge_flux_per_volume =
            std::numeric_limits<amrex::Real>::quiet_NaN();
        result.absolute_energy_flux_per_volume =
            std::numeric_limits<amrex::Real>::quiet_NaN();
        result.absolute_charge_flux_per_volume =
            std::numeric_limits<amrex::Real>::quiet_NaN();
#endif
        return result;
    }
#endif

#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_1D_Z)
    /** Support-aware divergence of a nodal current divided by nodal charge
     * density.  This mirrors the face construction used by the nonlinear FV
     * electron velocity, so subtracting its V_e divergence isolates the
     * retained div(J_plasma/rho) pressure-work channel without changing the
     * reviewed material/vacuum interface rule. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real qdsmc_ratio_velocity_direction_divergence (
        amrex::Array4<amrex::Real const> const& numerator,
        amrex::Array4<amrex::Real const> const& density,
        int const i, int const j, int const k,
        int const direction, int const lo, int const hi,
        amrex::Real const inv_dx, bool const periodic) noexcept
    {
        int const di = direction == 0 ? 1 : 0;
        int const dj = direction == 1 ? 1 : 0;
        int const dk = direction == 2 ? 1 : 0;
        int const index = direction == 0 ? i : (direction == 1 ? j : k);
        bool const low_wall = !periodic && index == lo;
        bool const high_wall = !periodic && index == hi;

        amrex::Real const rho_center = density(i, j, k);
        bool const center_has_material = rho_center > 0.0_rt;
        amrex::Real const velocity_center = center_has_material
            ? numerator(i, j, k) / rho_center : 0.0_rt;
        amrex::Real velocity_lo = 0.0_rt;
        amrex::Real velocity_hi = 0.0_rt;
        if (!low_wall) {
            amrex::Real const rho_left = density(i-di, j-dj, k-dk);
            bool const left_has_material = rho_left > 0.0_rt;
            amrex::Real const velocity_left = left_has_material
                ? numerator(i-di, j-dj, k-dk) / rho_left : 0.0_rt;
            velocity_lo = left_has_material && center_has_material
                ? 0.5_rt * (velocity_left + velocity_center)
                : (left_has_material ? velocity_left
                                     : (center_has_material
                                         ? velocity_center : 0.0_rt));
        }
        if (!high_wall) {
            amrex::Real const rho_right = density(i+di, j+dj, k+dk);
            bool const right_has_material = rho_right > 0.0_rt;
            amrex::Real const velocity_right = right_has_material
                ? numerator(i+di, j+dj, k+dk) / rho_right : 0.0_rt;
            velocity_hi = center_has_material && right_has_material
                ? 0.5_rt * (velocity_center + velocity_right)
                : (center_has_material ? velocity_center
                                       : (right_has_material
                                           ? velocity_right : 0.0_rt));
        }
        amrex::Real const inv_control_width = inv_dx
            * ((!periodic && (index == lo || index == hi)) ? 2.0_rt : 1.0_rt);
        return (velocity_hi - velocity_lo) * inv_control_width;
    }
#endif

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real qdsmc_cartesian_ratio_velocity_divergence (
        amrex::Array4<amrex::Real const> const& numerator_x,
        amrex::Array4<amrex::Real const> const& numerator_y,
        amrex::Array4<amrex::Real const> const& numerator_z,
        amrex::Array4<amrex::Real const> const& density,
        int const i, int const j, int const k,
        amrex::Dim3 const nodal_lo, amrex::Dim3 const nodal_hi,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& inv_dx,
        amrex::GpuArray<int, 3> const& periodic) noexcept
    {
        amrex::ignore_unused(numerator_x, numerator_y);
#if defined(WARPX_DIM_3D)
        return qdsmc_ratio_velocity_direction_divergence(
                   numerator_x, density, i, j, k, 0,
                   nodal_lo.x, nodal_hi.x, inv_dx[0], periodic[0] != 0)
            + qdsmc_ratio_velocity_direction_divergence(
                   numerator_y, density, i, j, k, 1,
                   nodal_lo.y, nodal_hi.y, inv_dx[1], periodic[1] != 0)
            + qdsmc_ratio_velocity_direction_divergence(
                   numerator_z, density, i, j, k, 2,
                   nodal_lo.z, nodal_hi.z, inv_dx[2], periodic[2] != 0);
#elif defined(WARPX_DIM_XZ)
        return qdsmc_ratio_velocity_direction_divergence(
                   numerator_x, density, i, j, k, 0,
                   nodal_lo.x, nodal_hi.x, inv_dx[0], periodic[0] != 0)
            + qdsmc_ratio_velocity_direction_divergence(
                   numerator_z, density, i, j, k, 1,
                   nodal_lo.y, nodal_hi.y, inv_dx[1], periodic[1] != 0);
#elif defined(WARPX_DIM_1D_Z)
        return qdsmc_ratio_velocity_direction_divergence(
            numerator_z, density, i, j, k, 0,
            nodal_lo.x, nodal_hi.x, inv_dx[0], periodic[0] != 0);
#else
        amrex::ignore_unused(
            numerator_x, numerator_y, numerator_z, density,
            i, j, k, nodal_lo, nodal_hi, inv_dx, periodic);
        return std::numeric_limits<amrex::Real>::quiet_NaN();
#endif
    }

#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_1D_Z)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real qdsmc_masked_pressure_work_velocity (
        amrex::Array4<amrex::Real const> const& work_current,
        amrex::Array4<amrex::Real const> const& pressure_work_state,
        int const i, int const j, int const k) noexcept
    {
        return work_current(i, j, k)
            * pressure_work_state(i, j, k, 1);
    }
#endif

    /** D(V_work) for the exact collocated pressure-force adjoint.  The nodal
     * Ohm-law pressure gradient is centered in every active Cartesian
     * direction, hence on a periodic uniform grid D=-G*=G. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real qdsmc_pressure_work_velocity_divergence (
        amrex::Array4<amrex::Real const> const& work_current_x,
        amrex::Array4<amrex::Real const> const& work_current_y,
        amrex::Array4<amrex::Real const> const& work_current_z,
        amrex::Array4<amrex::Real const> const& pressure_work_state,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& inv_dx) noexcept
    {
        amrex::ignore_unused(work_current_x, work_current_y);
        amrex::Real divergence = 0.0_rt;
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
        divergence += 0.5_rt * inv_dx[0] * (
            qdsmc_masked_pressure_work_velocity(
                work_current_x, pressure_work_state, i+1, j, k)
            - qdsmc_masked_pressure_work_velocity(
                work_current_x, pressure_work_state, i-1, j, k));
#endif
#if defined(WARPX_DIM_3D)
        divergence += 0.5_rt * inv_dx[1] * (
            qdsmc_masked_pressure_work_velocity(
                work_current_y, pressure_work_state, i, j+1, k)
            - qdsmc_masked_pressure_work_velocity(
                work_current_y, pressure_work_state, i, j-1, k));
        divergence += 0.5_rt * inv_dx[2] * (
            qdsmc_masked_pressure_work_velocity(
                work_current_z, pressure_work_state, i, j, k+1)
            - qdsmc_masked_pressure_work_velocity(
                work_current_z, pressure_work_state, i, j, k-1));
#elif defined(WARPX_DIM_XZ)
        divergence += 0.5_rt * inv_dx[1] * (
            qdsmc_masked_pressure_work_velocity(
                work_current_z, pressure_work_state, i, j+1, k)
            - qdsmc_masked_pressure_work_velocity(
                work_current_z, pressure_work_state, i, j-1, k));
#elif defined(WARPX_DIM_1D_Z)
        divergence += 0.5_rt * inv_dx[0] * (
            qdsmc_masked_pressure_work_velocity(
                work_current_z, pressure_work_state, i+1, j, k)
            - qdsmc_masked_pressure_work_velocity(
                work_current_z, pressure_work_state, i-1, j, k));
#else
        amrex::ignore_unused(
            work_current_x, work_current_y, work_current_z,
            pressure_work_state, i, j, k, inv_dx);
        divergence = std::numeric_limits<amrex::Real>::quiet_NaN();
#endif
        return divergence;
    }
}

HybridPICModel::HybridPICModel (
    warpx::materials::MaterialRegistry const* const material_registry)
{
    ReadParameters(material_registry);
}

HybridPICModel::~HybridPICModel () = default;

void HybridPICModel::ReadParameters (
    warpx::materials::MaterialRegistry const* const material_registry)
{
    const ParmParse pp_hybrid("hybrid_pic_model");

    // The B-field update is subcycled to improve stability - the number
    // of sub steps can be specified by the user.
    utils::parser::queryWithParser(pp_hybrid, "substeps", m_substeps);
    if (m_substeps % 2 != 0) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPIC",
            "hybrid_pic_model.substeps must be divisible by 2. "
            "The value " + std::to_string(m_substeps) + " is not valid. "
            "Automatically adjusting to " + std::to_string(m_substeps + 1) + ".",
            ablastr::warn_manager::WarnPriority::medium);
        m_substeps += 1;
    }

    // read rkf45 intervals
    std::vector<std::string> rkf45_intervals_string_vec = {"0"};
    pp_hybrid.queryarr("use_rkf45", rkf45_intervals_string_vec);
    m_rkf45_intervals = ablastr::utils::text::IntervalsParser(rkf45_intervals_string_vec);
    utils::parser::queryWithParser(pp_hybrid, "substep_rtol", m_substep_rtol);
    utils::parser::queryWithParser(pp_hybrid, "substep_atol", m_substep_atol);
    utils::parser::queryWithParser(pp_hybrid, "substep_safety", m_substep_safety);
    utils::parser::queryWithParser(pp_hybrid, "substep_max_growth", m_substep_max_growth);
    pp_hybrid.query("max_substep_attempts", m_max_substep_attempts);

    utils::parser::queryWithParser(pp_hybrid, "holmstrom_vacuum_region", m_holmstrom_vacuum_region);

    // The hybrid model requires an electron temperature, reference density
    // and exponent to be given. These values will be used to calculate the
    // electron pressure according to p = n0 * Te * (n/n0)^gamma
    utils::parser::queryWithParser(pp_hybrid, "gamma", m_gamma);
    if (!utils::parser::queryWithParser(pp_hybrid, "elec_temp", m_elec_temp)) {
        Abort("hybrid_pic_model.elec_temp must be specified when using the hybrid solver");
    }
    const bool n0_ref_given = utils::parser::queryWithParser(pp_hybrid, "n0_ref", m_n0_ref);
    if (m_gamma != 1.0 && !n0_ref_given) {
        Abort("hybrid_pic_model.n0_ref should be specified if hybrid_pic_model.gamma != 1");
    }
    m_electron_thermodynamics.ReadParameters(
        pp_hybrid, m_gamma, material_registry);

    pp_hybrid.query("plasma_resistivity(rho,J,t)", m_eta_expression);
    pp_hybrid.query("plasma_hyper_resistivity(rho,B)", m_eta_h_expression);

    utils::parser::queryWithParser(pp_hybrid, "n_floor", m_n_floor);

    // Master gate for the electron-energy equation. Ideal-gas states use the
    // published QDSMC entropy markers; supported nonlinear Cartesian states
    // use conservative finite-volume U_e transport with pressure work.
    // Operator-split sources are then applied and P_e is emitted for Ohm's
    // law. Default off preserves the legacy algebraic adiabatic closure.
    pp_hybrid.query("solve_electron_energy_equation",
                    m_solve_electron_energy_equation);
    pp_hybrid.query("conservative_pressure_work",
                    m_conservative_pressure_work);
#if defined(WARPX_DIM_RSPHERE)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_solve_electron_energy_equation,
        "hybrid_pic_model.solve_electron_energy_equation is not supported in "
        "RSPHERE geometry yet.");
#endif

    // The finite-volume U_e update covers Cartesian and cylindrical nodal
    // control volumes. Spherical geometry and multi-material tables still
    // require their own metric/composition operators; retain the reviewed
    // ideal-entropy fallback for those configurations and warn below.
    m_fv_transport_internal_energy =
        !m_electron_thermodynamics.executor().isIdealGas();
#if defined(WARPX_DIM_RSPHERE)
    m_fv_transport_internal_energy = false;
#endif
    if (m_electron_thermodynamics.isSingularitySpiner()
        && m_electron_thermodynamics.numMaterials() != 1)
    {
        m_fv_transport_internal_energy = false;
    }

    // Resistive electron-heating source (Phys. Plasmas 31, 012902 (2024), Eq. 12):
    //   S_e = Sigma_s nu_{s,e} n_s m_s |V_s - V_e|^2,  nu_{s,e} = Z_s e^2 eta n_e / m_s
    // added per cell to U_e by QDSMCAddJouleHeating, using the e-i relative
    // drift J_plasma/(e n_e) and rho_fp(_s). Reduces to eta J^2 in single species.
    // Default off; only consulted when solve_electron_energy_equation is on.
    pp_hybrid.query("include_joule_heating", m_include_joule_heating);

    // Te-threshold Joule redirection: heat electrons where Te < threshold,
    // deposit the Joule energy to ions where Te >= threshold. Off by default
    // (threshold < 0); specifying a threshold >= 0 enables the redirect.
    utils::parser::queryWithParser(pp_hybrid, "joule_redirect_Te_threshold", m_joule_redirect_Te_eV);
    m_joule_redirect_to_ions = (m_joule_redirect_Te_eV >= 0._rt);

    // Electron-ion thermal equilibration (Q_ei) on T_e:
    //   Q_ei = 3 n_e k_B nu_ei (T_e - T_i),  applied per ion species weighted by
    //   n_s/n_e, cooling T_e toward T_i. nu_ei[1/s] comes from the
    //   electron_ion_relaxation_rate(rho,Te,Ti,t) parser (rho [C/m^3], Te,Ti [eV]).
    //   The exact electron energy change is ledgered for the conjugate ion
    //   operator. A cell-local ion moment projection consumes that ledger
    //   while preserving cell ion momentum. Enabled by specifying the rate
    //   expression (only consulted when solve_electron_energy_equation is on).
    m_include_temperature_relaxation =
        pp_hybrid.query("electron_ion_relaxation_rate(rho,Te,Ti,t)", m_nu_ei_expression);

    if (m_electron_thermodynamics.isFixedChargeLatentEnergy()) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_electron_thermodynamics.supportsEnergyCoupling(),
            "hybrid_pic_model.electron_thermodynamics="
            "fixed_charge_latent_energy requires gamma > 1.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_solve_electron_energy_equation,
            "hybrid_pic_model.electron_thermodynamics="
            "fixed_charge_latent_energy requires "
            "solve_electron_energy_equation=1.");
        if (!m_fv_transport_internal_energy) {
            ablastr::warn_manager::WMRecordWarning(
                "HybridPICModel",
                "fixed_charge_latent_energy is a fixed-charge analytic caloric "
                "reservoir. This unsupported geometry or multi-material state "
                "still uses the reviewed ideal-entropy QDSMC fallback; "
                "compression and expansion are not yet nonlinear-EOS-"
                "consistent until its finite-volume operator is implemented.",
                ablastr::warn_manager::WarnPriority::medium);
        }
    }
    if (m_electron_thermodynamics.isSingularitySpiner()) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_solve_electron_energy_equation,
            "hybrid_pic_model.electron_thermodynamics=singularity_spiner "
            "requires solve_electron_energy_equation=1.");
        if (!m_fv_transport_internal_energy) {
            ablastr::warn_manager::WMRecordWarning(
                "HybridPICModel",
                "singularity_spiner is restricted to fixed-charge host-only "
                "material coupling. Unsupported geometries and multi-material "
                "tables still use the reviewed ideal-entropy QDSMC fallback; "
                "compression and expansion are not table-EOS-consistent in "
                "those configurations.",
                ablastr::warn_manager::WarnPriority::medium);
        }
    }

    // The electron-energy equation's Joule and Q_ei sources read the
    // per-species charge densities; this flag gates their allocation and
    // the per-species deposition path, so hybrid-PIC runs without the
    // energy equation carry zero extra cost.
    m_need_per_species_fields = m_solve_electron_energy_equation;

    // convert electron temperature from eV to J
    m_elec_temp *= PhysConst::q_e;

    // external currents
    pp_hybrid.query("Jx_external_grid_function(x,y,z,t)", m_Jx_ext_grid_function);
    pp_hybrid.query("Jy_external_grid_function(x,y,z,t)", m_Jy_ext_grid_function);
    pp_hybrid.query("Jz_external_grid_function(x,y,z,t)", m_Jz_ext_grid_function);

    // check if external currents are specified
    if ((m_Jx_ext_grid_function == "0.0") &&
        (m_Jy_ext_grid_function == "0.0") &&
        (m_Jz_ext_grid_function == "0.0"))
    {
        m_has_external_current = false;
    }

    // external fields
    pp_hybrid.query("add_external_fields", m_add_external_fields);

    if (m_add_external_fields) {
        m_external_vector_potential = std::make_unique<ExternalVectorPotential>();
    }
}

void HybridPICModel::AllocateLevelMFs (
    ablastr::fields::MultiFabRegister & fields,
    int lev, const BoxArray& ba, const DistributionMapping& dm,
    const int ncomps,
    const IntVect& ngJ, const IntVect& ngRho,
    const IntVect& ngEB,
    const IntVect& jx_nodal_flag,
    const IntVect& jy_nodal_flag,
    const IntVect& jz_nodal_flag,
    const IntVect& rho_nodal_flag,
    const IntVect& Ex_nodal_flag,
    const IntVect& Ey_nodal_flag,
    const IntVect& Ez_nodal_flag,
    const IntVect& Bx_nodal_flag,
    const IntVect& By_nodal_flag,
    const IntVect& Bz_nodal_flag) const
{
    using ablastr::fields::Direction;

    // The "hybrid_electron_pressure_fp" multifab stores the electron pressure
    // consumed by the Ohm's-law E-solve. With solve_electron_energy_equation
    // off, it is computed from the algebraic adiabatic closure each step. With
    // it on, Pe = n_e k_B T_e is emitted by QDSMCFillElectronPressureFromTe
    // at the end of each QDSMC entropy-transport step.
    fields.alloc_init(FieldType::hybrid_electron_pressure_fp,
        lev, amrex::convert(ba, rho_nodal_flag),
        dm, ncomps, ngRho, 0.0_rt);

    if (m_conservative_pressure_work) {
        // Preserve the pressure component that was actually present in the
        // checkpointed total E field.  Reconstructing it from a later rho/Pe
        // state on restart would debit electron energy for a force that did not
        // produce the restored particle trajectory.
        fields.alloc_init(
            FieldType::hybrid_pressure_E_fp, Direction{0}, lev,
            amrex::convert(ba, Ex_nodal_flag), dm, ncomps, ngEB, 0.0_rt,
            /*remake=*/true, /*redistribute_on_remake=*/true,
            /*checkpoint_restart=*/true);
        fields.alloc_init(
            FieldType::hybrid_pressure_E_fp, Direction{1}, lev,
            amrex::convert(ba, Ey_nodal_flag), dm, ncomps, ngEB, 0.0_rt,
            /*remake=*/true, /*redistribute_on_remake=*/true,
            /*checkpoint_restart=*/true);
        fields.alloc_init(
            FieldType::hybrid_pressure_E_fp, Direction{2}, lev,
            amrex::convert(ba, Ez_nodal_flag), dm, ncomps, ngEB, 0.0_rt,
            /*remake=*/true, /*redistribute_on_remake=*/true,
            /*checkpoint_restart=*/true);
        fields.alloc_init(
            FieldType::hybrid_pressure_work_current_fp, Direction{0}, lev,
            amrex::convert(ba, Ex_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        fields.alloc_init(
            FieldType::hybrid_pressure_work_current_fp, Direction{1}, lev,
            amrex::convert(ba, Ey_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        fields.alloc_init(
            FieldType::hybrid_pressure_work_current_fp, Direction{2}, lev,
            amrex::convert(ba, Ez_nodal_flag), dm, ncomps, ngEB, 0.0_rt);
        fields.alloc_init(
            FieldType::hybrid_pressure_work_state_fp, lev,
            amrex::convert(ba, rho_nodal_flag), dm,
            /*ncomp=*/2, ngRho, 0.0_rt,
            /*remake=*/true, /*redistribute_on_remake=*/true,
            /*checkpoint_restart=*/true);
    }

    // Electron temperature T_e (Kelvin). Allocated unconditionally (one
    // cheap scalar field) so the "Te" diagnostic can always read it: with
    // the energy equation on it is the QDSMC state variable, otherwise it
    // mirrors the closure's implied temperature T_e = P_e / (n_e k_B),
    // filled alongside P_e in CalculateElectronPressure.
    fields.alloc_init(FieldType::hybrid_electron_temperature_fp,
        lev, amrex::convert(ba, rho_nodal_flag),
        dm, ncomps, ngRho, 0.0_rt,
        /*remake=*/true,
        /*redistribute_on_remake=*/true,
        /*checkpoint_restart=*/m_solve_electron_energy_equation);

    // Electron-energy-equation working fields, only touched (and
    // therefore only allocated) when the energy equation is solved:
    //   * hybrid_qdsmc_thermodynamic_fp  : K_e or transported U_e scratch
    //   * hybrid_qdsmc_weights_fp        : deposited N_e scratch
    //   * hybrid_electron_velocity_fp    : three-component V_e on a NODAL
    //     grid, computed each step from V_e = -(J_plasma - J_i)/(q_e n_e)
    //     and consumed by either QDSMC marker advection or nonlinear FV
    //     transport.
    if (m_solve_electron_energy_equation) {
        fields.alloc_init(FieldType::hybrid_qdsmc_thermodynamic_fp,
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);
        fields.alloc_init(FieldType::hybrid_qdsmc_weights_fp,
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);
        fields.alloc_init(FieldType::hybrid_electron_velocity_fp, Direction{0},
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);
        fields.alloc_init(FieldType::hybrid_electron_velocity_fp, Direction{1},
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);
        fields.alloc_init(FieldType::hybrid_electron_velocity_fp, Direction{2},
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);

        if (m_include_joule_heating) {
            // Realized electron-side Joule increments [J/m^3].  The step
            // field exposes the exact caloric update; the cumulative field is
            // checkpointed so a run/restart energy audit has one continuous
            // ledger.  Redirected-to-ion energy is intentionally excluded.
            fields.alloc_init(
                "hybrid_joule_electron_energy_fp", lev,
                amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngRho,
                0.0_rt);
            fields.alloc_init(
                "hybrid_joule_electron_energy_cumulative_fp", lev,
                amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngRho,
                0.0_rt, /*remake=*/true,
                /*redistribute_on_remake=*/true,
                /*checkpoint_restart=*/true);
        }

        if (m_include_temperature_relaxation) {
            // Electron-side Q_ei source ledgers [J/m^3]. The step field is
            // the contract for the conjugate kinetic-ion update; cumulative
            // exchange remains auditable across checkpoints and restarts.
            fields.alloc_init(
                "hybrid_qei_electron_energy_fp", lev,
                amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngRho,
                0.0_rt);
            fields.alloc_init(
                "hybrid_qei_electron_energy_cumulative_fp", lev,
                amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngRho,
                0.0_rt, /*remake=*/true,
                /*redistribute_on_remake=*/true,
                /*checkpoint_restart=*/true);

            // The Q_ei operator consumes the shape-aware, cell-centered ion
            // temperature after interpolating it to the nodal electron grid.
            // Expose that exact per-species operand [eV] for diagnostics and
            // manufactured-solution tests.  This is deliberately distinct
            // from the legacy T_<species> diagnostic, which recomputes an NGP
            // temperature and therefore is not an oracle for this operator.
            auto const& mypc = WarpX::GetInstance().GetPartContainer();
            for (auto const& spec : mypc.GetSpeciesNames()) {
                if (mypc.GetParticleContainerFromName(spec).getCharge()
                    == 0.0_prt)
                {
                    continue;
                }
                fields.alloc_init(
                    "hybrid_qei_ion_temperature_fp_" + spec, lev,
                    amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngRho,
                    0.0_rt);
                fields.alloc_init(
                    "hybrid_qei_electron_temperature_before_fp_" + spec,
                    lev, amrex::convert(ba, rho_nodal_flag), dm, ncomps,
                    ngRho, 0.0_rt);
                // Per-species electron-side exchange [J/m^3].  Its negative,
                // integrated over the nodal dual volumes adjoining a particle
                // cell, is the exact kinetic-ion thermal-energy request.
                fields.alloc_init(
                    "hybrid_qei_electron_energy_fp_" + spec, lev,
                    amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngRho,
                    0.0_rt);
                // Conservative particle-cell request after the nodal source
                // is partitioned over adjacent local ion support [J/m^3].
                fields.alloc_init(
                    "hybrid_qei_ion_energy_cc_" + spec, lev,
                    amrex::convert(ba, amrex::IntVect::TheCellVector()), dm,
                    ncomps, amrex::IntVect::TheZeroVector(), 0.0_rt);
            }
        }

        if (m_fv_transport_internal_energy) {
            // Dedicated Yee-face current used solely as the conservative
            // electron charge/mass flux.  It intentionally keeps this
            // staggering when the primary hybrid field grid is collocated.
#if defined(WARPX_DIM_1D_Z)
            IntVect const flux_x_flag(1);
            IntVect const flux_y_flag(1);
            IntVect const flux_z_flag(0);
#elif defined(WARPX_DIM_XZ)
            IntVect const flux_x_flag(0, 1);
            IntVect const flux_y_flag(1, 1);
            IntVect const flux_z_flag(1, 0);
#elif defined(WARPX_DIM_3D)
            IntVect const flux_x_flag(0, 1, 1);
            IntVect const flux_y_flag(1, 0, 1);
            IntVect const flux_z_flag(1, 1, 0);
#elif defined(WARPX_DIM_RZ)
            IntVect const flux_x_flag(0, 1);
            IntVect const flux_y_flag(1, 1);
            IntVect const flux_z_flag(1, 0);
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            IntVect const flux_x_flag(0);
            IntVect const flux_y_flag(1);
            IntVect const flux_z_flag(1);
#else
            IntVect const flux_x_flag = jx_nodal_flag;
            IntVect const flux_y_flag = jy_nodal_flag;
            IntVect const flux_z_flag = jz_nodal_flag;
#endif
            fields.alloc_init(
                FieldType::hybrid_energy_charge_flux_fp, Direction{0}, lev,
                amrex::convert(ba, flux_x_flag), dm, ncomps, ngJ, 0.0_rt);
            fields.alloc_init(
                FieldType::hybrid_energy_charge_flux_fp, Direction{1}, lev,
                amrex::convert(ba, flux_y_flag), dm, ncomps, ngJ, 0.0_rt);
            fields.alloc_init(
                FieldType::hybrid_energy_charge_flux_fp, Direction{2}, lev,
                amrex::convert(ba, flux_z_flag), dm, ncomps, ngJ, 0.0_rt);
            fields.alloc_init(
                FieldType::hybrid_energy_rho_mid_fp, lev,
                amrex::convert(ba, rho_nodal_flag),
                dm, ncomps, ngRho, 0.0_rt);
            fields.alloc_init(
                FieldType::hybrid_energy_velocity_current_fp, Direction{0},
                lev, amrex::convert(ba, rho_nodal_flag),
                dm, ncomps, ngJ, 0.0_rt);
            fields.alloc_init(
                FieldType::hybrid_energy_velocity_current_fp, Direction{1},
                lev, amrex::convert(ba, rho_nodal_flag),
                dm, ncomps, ngJ, 0.0_rt);
            fields.alloc_init(
                FieldType::hybrid_energy_velocity_current_fp, Direction{2},
                lev, amrex::convert(ba, rho_nodal_flag),
                dm, ncomps, ngJ, 0.0_rt);
        }

        if (WarpX::GetInstance().GetPartContainer().hasHybridIonization()) {
            // Old charge density is scratch. The last-step liberated-electron
            // and binding-energy densities are physical diagnostic ledgers and
            // are checkpointed so diagnostics written immediately on restart
            // reproduce the checkpoint state.
            fields.alloc_init("hybrid_ionization_rho_old_fp",
                lev, amrex::convert(ba, rho_nodal_flag),
                dm, 1, ngRho, 0.0_rt);
            fields.alloc_init("hybrid_ionization_electron_source_fp",
                lev, amrex::convert(ba, rho_nodal_flag),
                dm, 1, ngRho, 0.0_rt,
                /*remake=*/true,
                /*redistribute_on_remake=*/true,
                /*checkpoint_restart=*/true);
            fields.alloc_init("hybrid_ionization_binding_energy_fp",
                lev, amrex::convert(ba, rho_nodal_flag),
                dm, 1, ngRho, 0.0_rt,
                /*remake=*/true,
                /*redistribute_on_remake=*/true,
                /*checkpoint_restart=*/true);
        }
    }

    // The "hybrid_rho_fp_temp" multifab is used to store the ion charge density
    // interpolated or extrapolated to appropriate timesteps.
    fields.alloc_init(FieldType::hybrid_rho_fp_temp,
        lev, amrex::convert(ba, rho_nodal_flag),
        dm, ncomps, ngRho, 0.0_rt);

    // The "hybrid_current_fp_temp" multifab is used to store the ion current density
    // interpolated or extrapolated to appropriate timesteps.
    fields.alloc_init(FieldType::hybrid_current_fp_temp, Direction{0},
        lev, amrex::convert(ba, jx_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_current_fp_temp, Direction{1},
        lev, amrex::convert(ba, jy_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_current_fp_temp, Direction{2},
        lev, amrex::convert(ba, jz_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);

    // The "hybrid_current_fp_plasma" multifab stores the total plasma current calculated
    // as the curl of B minus any external current.
    fields.alloc_init(FieldType::hybrid_current_fp_plasma, Direction{0},
        lev, amrex::convert(ba, jx_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_current_fp_plasma, Direction{1},
        lev, amrex::convert(ba, jy_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_current_fp_plasma, Direction{2},
        lev, amrex::convert(ba, jz_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);

    // Per-species charge densities - one per charged species, deposited
    // from particles and accumulated into the global rho_fp. Only
    // allocated when a feature that consumes them is active (see
    // m_need_per_species_fields).
    if (m_need_per_species_fields) {
        auto const & mypc = WarpX::GetInstance().GetPartContainer();
        for (auto const & spec : mypc.GetSpeciesNames()) {
            if (mypc.GetParticleContainerFromName(spec).getCharge() == 0._prt) { continue; }
            fields.alloc_init("rho_fp_" + spec,
                lev, amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngRho, 0.0_rt);
        }
        for (int material = 0;
             material < m_electron_thermodynamics.numMaterials(); ++material)
        {
            fields.alloc_init(
                "ni_charge_fp_"
                    + m_electron_thermodynamics.materialSpeciesName(material),
                lev, amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngRho,
                0.0_rt);
        }
        // Species-summed physical charge density Sigma_s rho_fp_s, filled
        // once per step in HybridPICDepositRhoAndJ (volume-scaled in radial
        // geometries like the totals, but unfiltered: the same processing as
        // the rho_fp_s numerators, so the species fraction
        // f_s = rho_s / Sigma_t rho_t is well-defined and the physical
        // rho_floor applies to it). Shared by the Joule and Q_ei consumers.
        fields.alloc_init("hybrid_rho_species_sum_fp",
            lev, amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngRho, 0.0_rt);
    }

    // the external current density multifab matches the current staggering and
    // one ghost cell is used since we interpolate the current to a nodal grid
    if (m_has_external_current) {
        fields.alloc_init(FieldType::hybrid_current_fp_external, Direction{0},
            lev, amrex::convert(ba, jx_nodal_flag),
            dm, ncomps, IntVect(1), 0.0_rt);
        fields.alloc_init(FieldType::hybrid_current_fp_external, Direction{1},
            lev, amrex::convert(ba, jy_nodal_flag),
            dm, ncomps, IntVect(1), 0.0_rt);
        fields.alloc_init(FieldType::hybrid_current_fp_external, Direction{2},
            lev, amrex::convert(ba, jz_nodal_flag),
            dm, ncomps, IntVect(1), 0.0_rt);
    }

    if (m_add_external_fields) {
        m_external_vector_potential->AllocateLevelMFs(
            fields,
            lev, ba, dm,
            ncomps, ngEB,
            Ex_nodal_flag, Ey_nodal_flag, Ez_nodal_flag,
            Bx_nodal_flag, By_nodal_flag, Bz_nodal_flag
        );
    }

#ifdef WARPX_DIM_RZ
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (ncomps == 1),
        "Ohm's law solver only support m = 0 azimuthal mode at present.");
#endif
}

void HybridPICModel::AllocateAuxiliaryLevelMFs (
    ablastr::fields::MultiFabRegister& fields,
    int const lev) const
{
    if (!m_conservative_pressure_work) { return; }

    using ablastr::fields::Direction;
    auto const E_aux = fields.get_alldirs(FieldType::Efield_aux, lev);
    for (int idim = 0; idim < 3; ++idim) {
        auto const* const layout = E_aux[idim];
        fields.alloc_init(
            FieldType::hybrid_pressure_E_aux, Direction{idim}, lev,
            layout->boxArray(), layout->DistributionMap(), layout->nComp(),
            layout->nGrowVect(), 0.0_rt);
        fields.alloc_init(
            FieldType::hybrid_pressure_work_current_aux,
            Direction{idim}, lev, layout->boxArray(),
            layout->DistributionMap(), layout->nComp(),
            layout->nGrowVect(), 0.0_rt);
    }
}

void HybridPICModel::InitData (const ablastr::fields::MultiFabRegister& fields)
{
    m_resistivity_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_eta_expression, {"rho","J","t"}));
    m_eta = m_resistivity_parser->compile<3>();
    const std::set<std::string> resistivity_symbols = m_resistivity_parser->symbols();
    m_resistivity_has_J_dependence += resistivity_symbols.count("J");

    // Electron-ion energy-equilibration rate nu_ei(rho,Te,Ti,t) for the Q_ei term.
    m_nu_ei_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_nu_ei_expression, {"rho","Te","Ti","t"}));
    m_nu_ei = m_nu_ei_parser->compile<4>();


    // The Te-threshold Joule redirect only acts inside the Joule source.
    if (m_joule_redirect_to_ions &&
        !(m_solve_electron_energy_equation && m_include_joule_heating)) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPICModel",
            "hybrid_pic_model.joule_redirect_Te_threshold is set, but the Joule "
            "heating source is not active (requires both "
            "hybrid_pic_model.solve_electron_energy_equation and "
            "hybrid_pic_model.include_joule_heating), so the redirect has no "
            "effect.",
            ablastr::warn_manager::WarnPriority::medium);
    }

    m_include_hyper_resistivity_term = (m_eta_h_expression != "0.0");
    m_hyper_resistivity_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_eta_h_expression, {"rho","B"}));
    m_eta_h = m_hyper_resistivity_parser->compile<2>();
    const std::set<std::string> hyper_resistivity_symbols = m_hyper_resistivity_parser->symbols();
    m_hyper_resistivity_has_B_dependence += hyper_resistivity_symbols.count("B");

    if (m_has_external_current) {
        m_J_external_parser[0] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_Jx_ext_grid_function,{"x","y","z","t"}));
        m_J_external_parser[1] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_Jy_ext_grid_function,{"x","y","z","t"}));
        m_J_external_parser[2] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_Jz_ext_grid_function,{"x","y","z","t"}));
        m_J_external[0] = m_J_external_parser[0]->compile<4>();
        m_J_external[1] = m_J_external_parser[1]->compile<4>();
        m_J_external[2] = m_J_external_parser[2]->compile<4>();

        // check if the external current parsers depend on time
        for (int i=0; i<3; i++) {
            const std::set<std::string> J_ext_symbols = m_J_external_parser[i]->symbols();
            m_external_current_has_time_dependence += J_ext_symbols.count("t");
        }
    }

    auto& warpx = WarpX::GetInstance();
    using ablastr::fields::Direction;

    if (m_conservative_pressure_work) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_solve_electron_energy_equation
                && m_fv_transport_internal_energy,
            "hybrid_pic_model.conservative_pressure_work requires the "
            "nonlinear finite-volume electron-energy equation.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::numeric_limits<amrex::Real>::digits >= 53,
            "hybrid_pic_model.conservative_pressure_work currently requires "
            "double-precision fields. The moving-vacuum-front cancellation "
            "cleanup has not yet been validated with single-precision "
            "continuity and flux arithmetic.");
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        amrex::Abort(
            "hybrid_pic_model.conservative_pressure_work is initially "
            "restricted to periodic Cartesian geometry. Radial adjoint "
            "metrics require a separate analytic validation.");
#endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::grid_type == GridType::Collocated,
            "hybrid_pic_model.conservative_pressure_work initially requires "
            "warpx.grid_type=collocated so pressure-field gather and work-"
            "current scatter are an exact transpose pair.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !WarpX::galerkin_interpolation,
            "hybrid_pic_model.conservative_pressure_work initially requires "
            "warpx.galerkin_interpolation=0. Galerkin's reduced longitudinal "
            "particle shape needs its matching transpose deposition.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::particle_pusher_algo == ParticlePusherAlgo::Boris,
            "hybrid_pic_model.conservative_pressure_work currently supports "
            "only the full explicit Boris momentum pusher.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::nox >= 1 && WarpX::nox <= 4,
            "hybrid_pic_model.conservative_pressure_work currently supports "
            "particle shape orders 1 through 4.");
        bool synchronize_velocity_for_diagnostics = true;
        amrex::ParmParse const pp_warpx("warpx");
        pp_warpx.query(
            "synchronize_velocity_for_diagnostics",
            synchronize_velocity_for_diagnostics);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !synchronize_velocity_for_diagnostics,
            "hybrid_pic_model.conservative_pressure_work currently requires "
            "warpx.synchronize_velocity_for_diagnostics=0. The diagnostic "
            "half momentum push is temporary and has no matching electron-"
            "energy debit; unsynchronized leapfrog kinetic energy is the "
            "conserved pressure-work ledger quantity.");
        std::vector<std::string> rigid_injected_species;
        amrex::ParmParse const pp_particles("particles");
        pp_particles.queryarr(
            "rigid_injected_species", rigid_injected_species);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            rigid_injected_species.empty(),
            "hybrid_pic_model.conservative_pressure_work does not yet "
            "support particles.rigid_injected_species. Rigid injection "
            "rescales or suppresses gathered fields before the particle "
            "push and therefore needs the same transformation in the "
            "pressure-work adjoint.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_add_external_fields,
            "hybrid_pic_model.conservative_pressure_work initially requires "
            "hybrid_pic_model.add_external_fields=0. Split external fields "
            "need an explicit initialization-time pressure-component audit.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !WarpX::use_fdtd_nci_corr,
            "hybrid_pic_model.conservative_pressure_work requires "
            "warpx.use_fdtd_nci_corr=0 until the identical filter is applied "
            "to the isolated pressure field.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            warpx.Geom(0).isAllPeriodic(),
            "hybrid_pic_model.conservative_pressure_work initially requires "
            "periodic field boundaries; reflecting and conducting boundaries "
            "need an even pressure-energy boundary adjoint.");
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                WarpX::particle_boundary_lo[d]
                        == ParticleBoundaryType::Periodic
                    && WarpX::particle_boundary_hi[d]
                        == ParticleBoundaryType::Periodic,
                "hybrid_pic_model.conservative_pressure_work initially "
                "requires periodic particle boundaries.");
        }
    }

    if (m_fv_transport_internal_energy) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !WarpX::use_filter,
            "Nonlinear finite-volume electron-energy transport currently "
            "requires warpx.use_filter=0 so rho and its auxiliary charge "
            "flux retain the same continuity stencil.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !WarpX::do_single_precision_comms,
            "Nonlinear finite-volume electron-energy transport currently "
            "requires warpx.do_single_precision_comms=0 so its rho/current "
            "continuity ledger is evaluated at one precision.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !WarpX::do_shared_mem_current_deposition,
            "Nonlinear finite-volume electron-energy transport requires a "
            "dedicated Esirkepov auxiliary current, which is incompatible "
            "with shared-memory current deposition.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !WarpX::do_moving_window,
            "Nonlinear finite-volume electron-energy transport does not yet "
            "shift its evolved temperature, old density, and auxiliary "
            "current state with a moving window.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !warpx.DoFluidSpecies(),
            "Nonlinear finite-volume electron-energy transport does not yet "
            "deposit conservative auxiliary moments for cold-fluid species.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !EB::enabled(),
            "Nonlinear finite-volume electron-energy transport does not yet "
            "include embedded-boundary face metrics and masks.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_has_external_current,
            "Nonlinear finite-volume electron-energy transport currently "
            "requires zero parser-defined volume current. A future external-"
            "current path must preserve discrete solenoidality.");

        auto const boundary_is_impermeable_or_periodic = [&warpx] (
            ParticleBoundaryType const boundary,
            int const d,
            bool const is_lower_face) noexcept
        {
            if (boundary == ParticleBoundaryType::Periodic
                || boundary == ParticleBoundaryType::Reflecting) {
                return true;
            }
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
            return boundary == ParticleBoundaryType::None && is_lower_face
                && d == 0 && warpx.Geom(0).ProbLo(0) == 0._rt;
#else
            amrex::ignore_unused(d, is_lower_face, warpx);
            return false;
#endif
        };
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                boundary_is_impermeable_or_periodic(
                    WarpX::particle_boundary_lo[d], d, true)
                    && boundary_is_impermeable_or_periodic(
                        WarpX::particle_boundary_hi[d], d, false),
                "Nonlinear finite-volume electron-energy transport currently "
                "supports periodic or specular-reflecting particle boundaries, "
                "plus ParticleBoundaryType::None at the low radial coordinate "
                "axis in RCYLINDER/RZ. Open, absorbing, thermal, and other "
                "None boundaries need an explicit material-energy flux/source "
                "ledger.");
        }

        auto& particles = warpx.GetPartContainer();
        for (auto const& species_name : particles.GetSpeciesNames()) {
            auto const& species =
                particles.GetParticleContainerFromName(species_name);
            if (species.getCharge() == 0.0_prt || species.do_not_deposit) {
                continue;
            }
            bool const is_fixed_positive_material =
                std::isfinite(static_cast<double>(species.getMass()))
                && std::isfinite(static_cast<double>(species.getCharge()))
                && species.getMass() > 0.0_prt
                && species.getCharge() > 0.0_prt
                && !species.HasEvolvingChargeState();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                is_fixed_positive_material,
                "Every depositing charged species in nonlinear finite-volume "
                "electron-energy transport must be a massive, fixed, "
                "positively charged material carrier. Unsupported species '"
                    + species_name + "'.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !species.doContinuousInjection(),
                "Nonlinear finite-volume electron-energy transport does not "
                "yet include a charge/internal-energy ledger for continuous "
                "injection. Unsupported species '" + species_name + "'.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !species.DoResampling(),
                "Nonlinear finite-volume electron-energy transport does not "
                "yet support particle resampling because it breaks the "
                "old-density/trajectory continuity ledger. Unsupported "
                "species '" + species_name + "'.");
        }
    }

    if (m_electron_thermodynamics.numMaterials() > 0) {
        bool const tabulated_eos =
            m_electron_thermodynamics.isSingularitySpiner();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_solve_electron_energy_equation,
            "Material-aware electron thermodynamics requires "
            "hybrid_pic_model.solve_electron_energy_equation=1.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !warpx.DoFluidSpecies(),
            "Material-aware electron thermodynamics currently requires "
            "particle materials only. Cold-fluid charge "
            "does not yet have a matching per-material mass-density field.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !WarpX::use_filter,
            "Material-aware electron thermodynamics currently requires "
            "warpx.use_filter=0. Total charge and material ion-number "
            "densities must undergo compatible "
            "deposition operators before evaluating the tabulated EOS.");
        auto& particles = warpx.GetPartContainer();
        std::set<std::string> material_species;
        auto const configured_thermodynamics =
            m_electron_thermodynamics.executor();
        for (int material = 0;
             material < m_electron_thermodynamics.numMaterials(); ++material)
        {
            std::string const& species_name =
                m_electron_thermodynamics.materialSpeciesName(material);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                material_species.insert(species_name).second,
                "Electron-thermodynamics material lists contain duplicate "
                "species '" + species_name + "'.");
            auto const& species =
                particles.GetParticleContainerFromName(species_name);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                species.getMass() > 0.0_prt
                    && species.getCharge() > 0.0_prt
                    && !species.do_not_deposit,
                "Electron-thermodynamics material species '" + species_name
                + "' must be a depositing, massive, positively charged ion "
                  "species.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !tabulated_eos || !species.HasEvolvingChargeState(),
                "Singularity-EOS material species '" + species_name
                + "' must have a fixed charge state. ElectronOnly equilibrium "
                  "tables cannot be combined with explicit hybrid ionization "
                  "without double-counting ionization physics.");
            amrex::Real const configured_mass =
                configured_thermodynamics.m_atomic_mass[material]
                * PhysConst::m_u;
            auto const particle_mass =
                static_cast<amrex::Real>(species.getMass());
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::abs(particle_mass - configured_mass)
                    <= amrex::Real(1.0e-6) * configured_mass,
                "Configured electron-thermodynamics atomic mass for species '"
                    + species_name
                    + "' does not match particle mass; effective-mass mappings "
                      "are currently unsupported.");
            m_electron_thermodynamics.setMaterialMassPerUnitCharge(
                material,
                particle_mass / PhysConst::q_e);
            m_electron_thermodynamics.setMaterialMassPerPhysicalCharge(
                material,
                particle_mass / static_cast<amrex::Real>(species.getCharge()));
        }
        for (auto const& species_name : particles.GetSpeciesNames()) {
            auto const& species =
                particles.GetParticleContainerFromName(species_name);
            if (species.getCharge() == 0.0_prt || species.do_not_deposit) {
                continue;
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                species.getMass() > 0.0_prt
                    && species.getCharge() > 0.0_prt
                    && material_species.count(species_name) == 1,
                "Every depositing charged particle in a material-aware "
                "hybrid simulation must be one of the configured positive "
                "material species. Unsupported "
                "species '" + species_name + "'.");
        }
        if (tabulated_eos) {
            auto const thermodynamics =
                m_electron_thermodynamics.executor();
            amrex::Real const initial_temperature =
                m_elec_temp / PhysConst::kb;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                initial_temperature >= thermodynamics.minimumTemperature()
                    && initial_temperature
                        <= thermodynamics.maximumTemperature(),
                "hybrid_pic_model.elec_temp lies outside the common "
                "temperature range of the configured Singularity-EOS "
                "electron tables.");
        }
        if (tabulated_eos
            && m_electron_thermodynamics.numMaterials() > 1) {
            ablastr::warn_manager::WMRecordWarning(
                "HybridPICModel",
                "Registered multi-material Singularity-EOS selects exactly "
                "one dominant material table per resolved node. Vacuum has "
                "zero material state; unresolved mixed nodes are rejected. "
                "This is not an EOS mixture closure.",
                ablastr::warn_manager::WarnPriority::medium);
        }
    }

    if (warpx.GetPartContainer().hasHybridIonization()) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_solve_electron_energy_equation,
            "do_hybrid_ionization requires "
            "hybrid_pic_model.solve_electron_energy_equation=1.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_electron_thermodynamics.executor().isIdealGas(),
            "do_hybrid_ionization currently requires "
            "hybrid_pic_model.electron_thermodynamics=ideal_gas. The "
            "fixed-charge latent reservoir would double-count energy when "
            "the free-electron density changes.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_electron_thermodynamics.supportsEnergyCoupling(),
            "do_hybrid_ionization requires a finite "
            "hybrid_pic_model.gamma > 1 so the ideal-gas electron internal "
            "energy and heat capacity are finite and positive.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !WarpX::use_filter,
            "do_hybrid_ionization currently requires warpx.use_filter=0. "
            "Filtering rho without applying the identical conservative "
            "filter to the ionization binding-energy ledger would break "
            "local electron-energy conservation.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_joule_redirect_to_ions,
            "do_hybrid_ionization currently requires disabling "
            "hybrid_pic_model.joule_redirect_Te_threshold because that "
            "ion-heating path does not yet use each particle's runtime "
            "charge state.");
        amrex::ParmParse const pp_warpx("warpx");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.contains("max_omegac_dt"),
            "do_hybrid_ionization currently cannot be combined with "
            "warpx.max_omegac_dt because that adaptive limiter uses each "
            "species' base charge instead of its runtime charge state.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.contains("max_omegap_dt"),
            "do_hybrid_ionization currently cannot be combined with "
            "warpx.max_omegap_dt because that adaptive limiter uses each "
            "species' base charge instead of its runtime charge state.");
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RSPHERE)
        amrex::Abort(
            "do_hybrid_ionization is initially supported in Cartesian and "
            "RCYLINDER geometry; RZ and RSPHERE validation is pending.");
#endif
    }

    // Get the grid staggering of the fields involved in calculating E
    amrex::IntVect Jx_stag = fields.get(FieldType::current_fp, Direction{0}, 0)->ixType().toIntVect();
    amrex::IntVect Jy_stag = fields.get(FieldType::current_fp, Direction{1}, 0)->ixType().toIntVect();
    amrex::IntVect Jz_stag = fields.get(FieldType::current_fp, Direction{2}, 0)->ixType().toIntVect();
    amrex::IntVect Bx_stag = fields.get(FieldType::Bfield_fp, Direction{0}, 0)->ixType().toIntVect();
    amrex::IntVect By_stag = fields.get(FieldType::Bfield_fp, Direction{1}, 0)->ixType().toIntVect();
    amrex::IntVect Bz_stag = fields.get(FieldType::Bfield_fp, Direction{2}, 0)->ixType().toIntVect();
    amrex::IntVect Ex_stag = fields.get(FieldType::Efield_fp, Direction{0}, 0)->ixType().toIntVect();
    amrex::IntVect Ey_stag = fields.get(FieldType::Efield_fp, Direction{1}, 0)->ixType().toIntVect();
    amrex::IntVect Ez_stag = fields.get(FieldType::Efield_fp, Direction{2}, 0)->ixType().toIntVect();

    // copy data to device
    for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        Jx_IndexType[idim]    = Jx_stag[idim];
        Jy_IndexType[idim]    = Jy_stag[idim];
        Jz_IndexType[idim]    = Jz_stag[idim];
        Bx_IndexType[idim]    = Bx_stag[idim];
        By_IndexType[idim]    = By_stag[idim];
        Bz_IndexType[idim]    = Bz_stag[idim];
        Ex_IndexType[idim]    = Ex_stag[idim];
        Ey_IndexType[idim]    = Ey_stag[idim];
        Ez_IndexType[idim]    = Ez_stag[idim];
    }

    // Below we set all the unused dimensions to have nodal values for J, B & E
    // since these values will be interpolated onto a nodal grid - if this is
    // not done the Interp function returns nonsense values.
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_1D_Z) || \
    defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    Jx_IndexType[2]    = 1;
    Jy_IndexType[2]    = 1;
    Jz_IndexType[2]    = 1;
    Bx_IndexType[2]    = 1;
    By_IndexType[2]    = 1;
    Bz_IndexType[2]    = 1;
    Ex_IndexType[2]    = 1;
    Ey_IndexType[2]    = 1;
    Ez_IndexType[2]    = 1;
#endif
#if defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    Jx_IndexType[1]    = 1;
    Jy_IndexType[1]    = 1;
    Jz_IndexType[1]    = 1;
    Bx_IndexType[1]    = 1;
    By_IndexType[1]    = 1;
    Bz_IndexType[1]    = 1;
    Ex_IndexType[1]    = 1;
    Ey_IndexType[1]    = 1;
    Ez_IndexType[1]    = 1;
#endif

    if (m_has_external_current) {
        // Initialize external current - note that this approach skips the check
        // if the current is time dependent which is what needs to be done to
        // write time independent fields on the first step.
        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            warpx.ComputeExternalFieldOnGridUsingParser(
                FieldType::hybrid_current_fp_external,
                m_J_external[0],
                m_J_external[1],
                m_J_external[2],
                lev, PatchType::fine,
                warpx.GetEBUpdateEFlag());
        }
    }

    if (m_add_external_fields) {
        m_external_vector_potential->InitData();
    }

    // Seed T_e with the uniform value parsed from <hybrid>.elec_temp (in
    // Joules after ReadParameters, so dividing by k_B gives Kelvin). The
    // iter-0 diagnostic dump -- which WarpX::InitData() flushes BEFORE the
    // first field-solve -- then sees a meaningful T_e rather than the
    // zero-initialized allocation. On restart, InitData runs after the field
    // checkpoint has been read, so it must leave the restored QDSMC state
    // untouched.
    std::string restart_checkpoint;
    amrex::ParmParse const pp_amr("amr");
    pp_amr.query("restart", restart_checkpoint);
    if (restart_checkpoint.empty()) {
        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            amrex::MultiFab & Te_mf = *warpx.m_fields.get(
                FieldType::hybrid_electron_temperature_fp, lev);
            Te_mf.setVal(m_elec_temp / PhysConst::kb);
        }
    }

    // QDSMC: lazy-construct the fictitious-particle container and lay one
    // particle per cell.
    if (m_solve_electron_energy_equation) {
        m_qdsmc_pc = std::make_unique<QdsmcParticleContainer>(&warpx);
        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            m_qdsmc_pc->InitParticles(lev);
        }
    }
}

void HybridPICModel::GetCurrentExternal ()
{
    if (!m_external_current_has_time_dependence) { return; }

    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        warpx.ComputeExternalFieldOnGridUsingParser(
            FieldType::hybrid_current_fp_external,
            m_J_external[0],
            m_J_external[1],
            m_J_external[2],
            lev, PatchType::fine,
            warpx.GetEBUpdateEFlag());
    }
}

void HybridPICModel::CalculatePlasmaCurrent (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E) const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        CalculatePlasmaCurrent(Bfield[lev], eb_update_E[lev], lev);
    }
}

void HybridPICModel::CalculatePlasmaCurrent (
    ablastr::fields::VectorField const& Bfield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
    const int lev) const
{
    ABLASTR_PROFILE("HybridPICModel::CalculatePlasmaCurrent()");

    auto& warpx = WarpX::GetInstance();
    ablastr::fields::VectorField current_fp_plasma = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    warpx.get_pointer_fdtd_solver_fp(lev)->CalculateCurrentAmpere(
        current_fp_plasma, Bfield, eb_update_E, lev
    );

    if (m_has_external_current) {
        // Subtract external current from "Ampere" current calculated above. Note
        // we need to include 1 ghost cell since later we will interpolate the
        // plasma current to a nodal grid.
        ablastr::fields::VectorField current_fp_external = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_external, lev);
        for (int i=0; i<3; i++) {
            current_fp_plasma[i]->minus(*current_fp_external[i], 0, 1, 1);
        }
    }
}

void HybridPICModel::HybridPICSolveE (
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    const bool solve_for_Faraday) const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        HybridPICSolveE(
            Efield[lev], Jfield[lev], Bfield[lev], *rhofield[lev],
            eb_update_E[lev], lev, solve_for_Faraday
        );
    }
    // Allow execution of Python callback after E-field push
    ExecutePythonCallback("afterEpush");
}

void HybridPICModel::HybridPICSolveE (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
    const int lev, const bool solve_for_Faraday) const
{
    ABLASTR_PROFILE("WarpX::HybridPICSolveE()");

    HybridPICSolveE(
        Efield, Jfield, Bfield, rhofield, eb_update_E, lev,
        PatchType::fine, solve_for_Faraday
    );
    if (lev > 0)
    {
        amrex::Abort(Utils::TextMsg::Err(
        "HybridPICSolveE: Only one level implemented for hybrid-PIC solver."));
    }
}

void HybridPICModel::HybridPICSolveE (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
    const int lev, PatchType patch_type,
    const bool solve_for_Faraday) const
{
    auto& warpx = WarpX::GetInstance();

    ablastr::fields::VectorField current_fp_plasma = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    auto* const electron_pressure_fp = warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    ablastr::fields::VectorField pressure_Efield{};
    if (m_conservative_pressure_work && !solve_for_Faraday) {
        pressure_Efield =
            warpx.m_fields.get_alldirs(FieldType::hybrid_pressure_E_fp, lev);
    }

    // Solve E field in regular cells
    warpx.get_pointer_fdtd_solver_fp(lev)->HybridPICSolveE(
        Efield, pressure_Efield, current_fp_plasma, Jfield, Bfield, rhofield,
        *electron_pressure_fp, eb_update_E, lev, this, solve_for_Faraday
    );
    amrex::Real const time = warpx.gett_old(0) + warpx.getdt(0);
    warpx.ApplyEfieldBoundary(lev, patch_type, time);
    if (m_conservative_pressure_work && !solve_for_Faraday) {
        auto const& period = warpx.Geom(lev).periodicity();
        for (auto* pressure_component : pressure_Efield) {
            ablastr::utils::communication::FillBoundary(
                *pressure_component, pressure_component->nGrowVect(),
                /*do_single_precision_comms=*/false, period,
                /*nodal_sync=*/true);
        }

        // Freeze the scalar state that generated this pressure component.
        // Checkpointing P and the exact masked reciprocal denominator makes
        // the later adjoint debit independent of roundoff in rho/Pe
        // reconstruction after a rank-changing restart.
        auto& pressure_work_state = *warpx.m_fields.get(
            FieldType::hybrid_pressure_work_state_fp, lev);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            pressure_work_state.nComp() == 2
                && pressure_work_state.boxArray()
                    == electron_pressure_fp->boxArray()
                && pressure_work_state.DistributionMap()
                    == electron_pressure_fp->DistributionMap(),
            "The checkpointed pressure-work state must share the electron-"
            "pressure layout.");
        amrex::Real const rho_floor = PhysConst::q_e * m_n_floor;
        bool const holmstrom_vacuum_region = m_holmstrom_vacuum_region;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(
                 pressure_work_state, amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi)
        {
            amrex::Box const box = mfi.tilebox();
            auto const state = pressure_work_state.array(mfi);
            auto const pressure = electron_pressure_fp->const_array(mfi);
            auto const rho = rhofield.const_array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (
                int i, int j, int k) noexcept
            {
                amrex::Real const rho_value = rho(i, j, k);
                state(i, j, k, 0) = pressure(i, j, k);
                state(i, j, k, 1) =
                    holmstrom_vacuum_region && rho_value < rho_floor
                    ? 0.0_rt
                    : 1.0_rt / amrex::max(rho_value, rho_floor);
            });
        }
        ablastr::utils::communication::FillBoundary(
            pressure_work_state, pressure_work_state.nGrowVect(),
            /*do_single_precision_comms=*/false, period,
            /*nodal_sync=*/true);
    }
}

void HybridPICModel::CalculateElectronPressure(bool const floor_density) const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        CalculateElectronPressure(lev, floor_density);
    }
}

void HybridPICModel::CalculateElectronPressure(const int lev, bool const floor_density) const
{
    ABLASTR_PROFILE("WarpX::CalculateElectronPressure()");

    auto& warpx = WarpX::GetInstance();
    ablastr::fields::ScalarField electron_temperature_fp = warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    ablastr::fields::ScalarField electron_pressure_fp = warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    ablastr::fields::ScalarField rho_fp = warpx.m_fields.get(FieldType::rho_fp, lev);

    // Calculate the electron pressure (and its implied temperature) using rho^{n+1}.
    FillElectronPressureMF(
        *electron_pressure_fp,
        *electron_temperature_fp,
        *rho_fp,
        floor_density
    );
    warpx.ApplyElectronPressureBoundary(lev, PatchType::fine);
    ablastr::utils::communication::FillBoundary(
        *electron_pressure_fp,
        WarpX::do_single_precision_comms,
        warpx.Geom(lev).periodicity(),
        true);
}

void HybridPICModel::FillElectronPressureMF (
    amrex::MultiFab& Pe_field,
    amrex::MultiFab& Te_field,
    amrex::MultiFab const& rho_field,
    bool const floor_density
) const
{
    const auto n0_ref = m_n0_ref;
    const auto elec_temp = m_elec_temp;
    const auto gamma_minus_1 = m_gamma - 1.0_rt;
    // Only bites when floor_density is set: max(rho, 0) leaves every physical
    // rho >= 0 bit-for-bit alone, so the algebraic-closure path is unchanged.
    const auto rho_floor =
        floor_density ? PhysConst::q_e * m_n_floor : amrex::Real(0.0);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(Pe_field, TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {
        // Extract field data for this grid/tile
        Array4<Real const> const& rho = rho_field.const_array(mfi);
        Array4<Real> const& Te = Te_field.array(mfi);
        Array4<Real> const& Pe = Pe_field.array(mfi);

        // Extract tileboxes for which to loop
        Box tilebox = mfi.tilebox();
        // Cover the ghosts too.
        // QDSMC thermodynamic initialization reads T_e over its own
        // ghost-grown box
        // so the seed has to leave T_e's ghosts valid itself.
        // Out-of-domain ghosts are handled at the density floor.
        tilebox.grow(Pe_field.nGrowVect());

        ParallelFor(tilebox, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Polytropic closure: T_e = T0 (n_e/n0)^(gamma-1), in the units of
            // elec_temp (Joules), with P_e = n_e T_e following from it. The
            // "Te" diagnostic wants Kelvin. Flooring n_e once here keeps P_e
            // and T_e consistent with each other.
            const Real ne = std::max(rho(i, j, k), rho_floor) / PhysConst::q_e;
            const Real Te_joule = elec_temp * std::pow(ne/n0_ref, gamma_minus_1);
            Pe(i, j, k) = ne * Te_joule;
            Te(i, j, k) = Te_joule / PhysConst::kb;
        });
    }
}

// =============================================================================
// QDSMC electron-energy-equation orchestration
// =============================================================================
//
// All four methods below are NO-OPs when m_solve_electron_energy_equation is false; they are
// invoked from HybridPICEvolveFields only when QDSMC is enabled. They operate
// on the level-`lev` MultiFabs of WarpX's MultiFabRegister and use the same
// Yee->nodal interpolation (`ablastr::coarsen::sample::Interp`) as the rest
// of the hybrid solver.

void HybridPICModel::QDSMCInitializeUe (int const lev) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCInitializeUe()");

    using ablastr::fields::Direction;

    auto & warpx = WarpX::GetInstance();
    amrex::Geometry const & geom = warpx.Geom(lev);
    amrex::Periodicity const & period = geom.periodicity();

    // V_e and rho live at the nodal grid; J_plasma and J_i are Yee-staggered.
    amrex::MultiFab       & Vex = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{0}, lev);
    amrex::MultiFab       & Vey = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{1}, lev);
    amrex::MultiFab       & Vez = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{2}, lev);

    amrex::MultiFab const & rho_velocity = *warpx.m_fields.get(
        m_fv_transport_internal_energy
            ? FieldType::hybrid_energy_rho_mid_fp
            : FieldType::hybrid_rho_fp_temp,
        lev);

    ablastr::fields::VectorField J_plasma =
        warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    // The nonlinear conservative update runs after the particle push and
    // uses a direct nodal current co-deposited with rho at the particle
    // midpoint.  Retain the reviewed old-current choice for ideal QDSMC.
    ablastr::fields::VectorField J_i = m_fv_transport_internal_energy
        ? warpx.m_fields.get_alldirs(
            FieldType::hybrid_energy_velocity_current_fp, lev)
        : warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_temp, lev);

    amrex::GpuArray<int, 3> const Jpx_stag = Jx_IndexType;
    amrex::GpuArray<int, 3> const Jpy_stag = Jy_IndexType;
    amrex::GpuArray<int, 3> const Jpz_stag = Jz_IndexType;
    amrex::GpuArray<int, 3> Jix_stag = Jx_IndexType;
    amrex::GpuArray<int, 3> Jiy_stag = Jy_IndexType;
    amrex::GpuArray<int, 3> Jiz_stag = Jz_IndexType;
    if (m_fv_transport_internal_energy) {
        Jix_stag = {1, 1, 1};
        Jiy_stag = {1, 1, 1};
        Jiz_stag = {1, 1, 1};
    }
    amrex::Real const rho_floor = PhysConst::q_e * m_n_floor;
    bool const use_positive_support = m_fv_transport_internal_energy;
    amrex::GpuArray<int, 3> const nodal     = {1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen   = {1, 1, 1};

    Vex.setVal(0.0_rt);
    Vey.setVal(0.0_rt);
    Vez.setVal(0.0_rt);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Vex, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real const> const & rho_arr =
            rho_velocity.const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jpx     = J_plasma[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jpy     = J_plasma[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jpz     = J_plasma[2]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jix     = J_i[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jiy     = J_i[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jiz     = J_i[2]->const_array(mfi);
        amrex::Array4<amrex::Real>       const & Vex_arr = Vex.array(mfi);
        amrex::Array4<amrex::Real>       const & Vey_arr = Vey.array(mfi);
        amrex::Array4<amrex::Real>       const & Vez_arr = Vez.array(mfi);

        amrex::Box const & tbox = mfi.tilebox();

        amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real const rho_val = rho_arr(i,j,k);
            // The nonlinear path uses the deposited PIC support, not the
            // Ohm-law density floor, as its material topology.  A swept
            // shape tail below n_floor still carries a finite co-deposited
            // current/rho velocity and must not become an artificial
            // zero-velocity gap at a material/vacuum interface.
            if (use_positive_support
                    ? rho_val <= 0.0_rt
                    : rho_val <= rho_floor)
            {
                return;
            }

            auto const jx  = ablastr::coarsen::sample::Interp(Jpx, Jpx_stag, nodal, coarsen, i, j, k, 0);
            auto const jy  = ablastr::coarsen::sample::Interp(Jpy, Jpy_stag, nodal, coarsen, i, j, k, 0);
            auto const jz  = ablastr::coarsen::sample::Interp(Jpz, Jpz_stag, nodal, coarsen, i, j, k, 0);
            auto const jix = ablastr::coarsen::sample::Interp(Jix, Jix_stag, nodal, coarsen, i, j, k, 0);
            auto const jiy = ablastr::coarsen::sample::Interp(Jiy, Jiy_stag, nodal, coarsen, i, j, k, 0);
            auto const jiz = ablastr::coarsen::sample::Interp(Jiz, Jiz_stag, nodal, coarsen, i, j, k, 0);

            // V_e = -(J_plasma - J_i) / (q_e * n_e) = -(J_plasma - J_i) / rho_val
            Vex_arr(i,j,k) = -(jx - jix) / rho_val;
            Vey_arr(i,j,k) = -(jy - jiy) / rho_val;
            Vez_arr(i,j,k) = -(jz - jiz) / rho_val;
        });
    }

    if (m_fv_transport_internal_energy) {
        // The nonlinear FV path requires one identical face velocity on both
        // sides of every FAB/MPI interface. Synchronize overlapping nodal
        // valid points and guards in full precision so paired fluxes cancel.
        ablastr::utils::communication::FillBoundary(
            Vex, Vex.nGrowVect(), false, period, true);
        ablastr::utils::communication::FillBoundary(
            Vey, Vey.nGrowVect(), false, period, true);
        ablastr::utils::communication::FillBoundary(
            Vez, Vez.nGrowVect(), false, period, true);
    } else {
        // Preserve the reviewed ideal-entropy QDSMC communication path.
        Vex.FillBoundary(Vex.nGrowVect(), period);
        Vey.FillBoundary(Vey.nGrowVect(), period);
        Vez.FillBoundary(Vez.nGrowVect(), period);
    }
}


void HybridPICModel::QDSMCInitializeThermodynamicQuantity (
    int const lev) const
{
    ABLASTR_PROFILE(
        "HybridPICModel::QDSMCInitializeThermodynamicQuantity()");

    auto & warpx = WarpX::GetInstance();

    amrex::MultiFab       & Ke  = *warpx.m_fields.get(FieldType::hybrid_qdsmc_thermodynamic_fp,    lev);
    amrex::MultiFab const & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp,   lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::hybrid_rho_fp_temp,               lev);

    Ke.setVal(0.0_rt);

    if (m_fv_transport_internal_energy) {
        auto const thermodynamics = m_electron_thermodynamics.executor();
        int const num_materials =
            m_electron_thermodynamics.numMaterials();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            num_materials <= 1,
            "Nonlinear finite-volume electron-energy transport currently "
            "requires at most one fixed material table.");

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Ke, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Array4<amrex::Real> const& energy = Ke.array(mfi);
            amrex::Array4<amrex::Real const> const& temperature =
                Te.const_array(mfi);
            amrex::Array4<amrex::Real const> const& charge_density =
                rho.const_array(mfi);

            amrex::Box const box = amrex::convert(
                mfi.tilebox(), Ke.ixType().toIntVect());

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (
                int i, int j, int k) noexcept
            {
                amrex::Real const deposited_rho_node =
                    charge_density(i, j, k);
                amrex::Real const rho_node =
                    amrex::max(deposited_rho_node, 0.0_rt);
                ElectronThermodynamicsExecutor::MaterialMassDensities
                    material_mass_density{};
                if (num_materials == 1) {
                    material_mass_density[0] = thermodynamics
                        .materialMassDensityFromPhysicalChargeDensity(
                            0, rho_node);
                }
                ElectronThermodynamicState const state = thermodynamics
                    .stateFromMaterialMassDensitiesTemperature(
                        rho_node, material_mass_density,
                        temperature(i, j, k));
                energy(i, j, k) =
                    amrex::Math::isfinite(state.internal_energy_density)
                    ? state.internal_energy_density
                    : std::numeric_limits<amrex::Real>::quiet_NaN();
            });
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            Ke.is_finite(0, 1, Ke.nGrowVect()),
            "Nonlinear electron-energy initialization encountered an invalid "
            "electron internal-energy state.");
        // Compute each valid node once, then make all overlapping nodal
        // representations and neighbor guards identical. Full precision is
        // required so paired finite-volume face fluxes cancel exactly.
        auto const& geom = warpx.Geom(lev);
        ablastr::utils::communication::FillBoundary(
            Ke, Ke.nGrowVect(), false, geom.periodicity(), true);
        return;
    }

    auto const gamma     = m_gamma;
    auto const rho_floor = PhysConst::q_e * m_n_floor;
    // Scale K_e to eV-equivalent (multiply T_e[K] by k_B/q_e) to keep it
    // numerically O(1) for common plasma parameters.
    auto const kb_over_qe = PhysConst::kb / PhysConst::q_e;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Ke, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real>       const & Ke_arr  = Ke.array(mfi);
        amrex::Array4<amrex::Real const> const & Te_arr  = Te.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);

        amrex::Box const tbox = amrex::convert(mfi.tilebox(), Ke.ixType().toIntVect());
        amrex::Box       box  = tbox;
        box.grow(Ke.nGrowVect());

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Floor the density instead of skipping low-density cells:
            // leaving K_e = 0 in the floored halo turns it into an absorbing
            // boundary that drains the plasma's electron thermal energy via
            // the remap diffusion (global exponential T_e collapse). With the
            // floor, the halo keeps whatever T_e it holds and is insulating.
            amrex::Real const ne =
                amrex::max(rho_arr(i,j,k), rho_floor) / PhysConst::q_e;
            Ke_arr(i,j,k) = Te_arr(i,j,k) * std::pow(ne, 1.0_rt - gamma) * kb_over_qe;
        });
    }
    // No ghost exchange: the kernel runs on the ghost-grown box and its
    // inputs (Te, rho at n) already have valid ghosts, so Ke's ghost cells
    // are consistent with the neighboring boxes' valid values.
}


void HybridPICModel::QDSMCUpdateThermodynamics (
    int const lev, amrex::Real const dt) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCUpdateThermodynamics()");

    auto & warpx = WarpX::GetInstance();
    // Nonlinear thermodynamics advance U_e with a conservative Cartesian
    // finite-volume operator and invert U_e(rho,T_e). Ideal thermodynamics
    // recover T_e from the entropy and count scattered by QDSMC markers.

    amrex::MultiFab       & Te      = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & Ke      = *warpx.m_fields.get(FieldType::hybrid_qdsmc_thermodynamic_fp,  lev);
    amrex::MultiFab const & weights = *warpx.m_fields.get(FieldType::hybrid_qdsmc_weights_fp,        lev);
    amrex::MultiFab       & rho     = *warpx.m_fields.get(FieldType::rho_fp,                         lev);

    if (m_fv_transport_internal_energy) {
        using ablastr::fields::Direction;
        amrex::MultiFab const& Vex = *warpx.m_fields.get(
            FieldType::hybrid_electron_velocity_fp, Direction{0}, lev);
        amrex::MultiFab const& Vey = *warpx.m_fields.get(
            FieldType::hybrid_electron_velocity_fp, Direction{1}, lev);
        amrex::MultiFab const& Vez = *warpx.m_fields.get(
            FieldType::hybrid_electron_velocity_fp, Direction{2}, lev);
        amrex::MultiFab const& old_rho = *warpx.m_fields.get(
            FieldType::hybrid_rho_fp_temp, lev);
        amrex::MultiFab const& midpoint_rho = *warpx.m_fields.get(
            FieldType::hybrid_energy_rho_mid_fp, lev);
        ablastr::fields::VectorField const ion_current =
            warpx.m_fields.get_alldirs(
                FieldType::hybrid_energy_charge_flux_fp, lev);
        ablastr::fields::VectorField const plasma_current =
            warpx.m_fields.get_alldirs(
                FieldType::hybrid_current_fp_plasma, lev);
        ablastr::fields::VectorField pressure_work_current{};
        if (m_conservative_pressure_work) {
            pressure_work_current = warpx.m_fields.get_alldirs(
                FieldType::hybrid_pressure_work_current_fp, lev);
        }
        amrex::MultiFab const* pressure_work_state = nullptr;
        if (m_conservative_pressure_work) {
            pressure_work_state = warpx.m_fields.get(
                FieldType::hybrid_pressure_work_state_fp, lev);
        }
        auto const thermodynamics = m_electron_thermodynamics.executor();
        int const num_materials =
            m_electron_thermodynamics.numMaterials();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            num_materials <= 1,
            "Nonlinear finite-volume electron-energy transport currently "
            "requires at most one fixed material table.");

        // The dedicated Esirkepov field must remain face staggered even when
        // the primary hybrid fields use the recommended collocated/direct
        // configuration.
#if defined(WARPX_DIM_3D)
        bool const currents_are_dual_face_staggered =
            ion_current[0]->ixType().cellCentered(0)
            && ion_current[1]->ixType().cellCentered(1)
            && ion_current[2]->ixType().cellCentered(2);
        amrex::GpuArray<int, 3> const plasma_current_face_centered{
            Jx_IndexType[0] == 0 ? 1 : 0,
            Jy_IndexType[1] == 0 ? 1 : 0,
            Jz_IndexType[2] == 0 ? 1 : 0};
#elif defined(WARPX_DIM_XZ)
        bool const currents_are_dual_face_staggered =
            ion_current[0]->ixType().cellCentered(0)
            && ion_current[2]->ixType().cellCentered(1);
        amrex::GpuArray<int, 3> const plasma_current_face_centered{
            Jx_IndexType[0] == 0 ? 1 : 0,
            Jz_IndexType[1] == 0 ? 1 : 0, 0};
#elif defined(WARPX_DIM_1D_Z)
        bool const currents_are_dual_face_staggered =
            ion_current[2]->ixType().cellCentered(0);
        amrex::GpuArray<int, 3> const plasma_current_face_centered{
            Jz_IndexType[0] == 0 ? 1 : 0, 0, 0};
#elif defined(WARPX_DIM_RCYLINDER)
        bool const currents_are_dual_face_staggered =
            ion_current[0]->ixType().cellCentered(0);
        amrex::GpuArray<int, 3> const plasma_current_face_centered{
            Jx_IndexType[0] == 0 ? 1 : 0, 0, 0};
#elif defined(WARPX_DIM_RZ)
        bool const currents_are_dual_face_staggered =
            ion_current[0]->ixType().cellCentered(0)
            && ion_current[2]->ixType().cellCentered(1);
        amrex::GpuArray<int, 3> const plasma_current_face_centered{
            Jx_IndexType[0] == 0 ? 1 : 0, 0,
            Jz_IndexType[1] == 0 ? 1 : 0};
#else
        bool const currents_are_dual_face_staggered = false;
        amrex::GpuArray<int, 3> const plasma_current_face_centered{0, 0, 0};
#endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            currents_are_dual_face_staggered,
            "The nonlinear electron-energy auxiliary current lost its "
            "required dual-face staggering.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !WarpX::use_filter,
            "Nonlinear finite-volume electron-energy transport currently "
            "requires warpx.use_filter=0 so rho and the electron charge flux "
            "satisfy the same discrete continuity equation.");

        auto const& geom = warpx.Geom(lev);
        ablastr::utils::communication::FillBoundary(
            rho, rho.nGrowVect(), false, geom.periodicity(), true);
        auto const inv_dx = geom.InvCellSizeArray();
        auto const dx = geom.CellSizeArray();
        auto const problo = geom.ProbLoArray();
        auto const probhi = geom.ProbHiArray();
        amrex::Dim3 const physical_domain_lo =
            amrex::lbound(geom.Domain());
        amrex::Dim3 const physical_domain_hi =
            amrex::ubound(geom.Domain());
        amrex::Box const nodal_domain =
            amrex::convert(geom.Domain(), Te.ixType().toIntVect());
        amrex::Dim3 const nodal_lo = amrex::lbound(nodal_domain);
        amrex::Dim3 const nodal_hi = amrex::ubound(nodal_domain);
        amrex::GpuArray<int, 3> periodic{0, 0, 0};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            periodic[d] = geom.isPeriodic(d) ? 1 : 0;
        }
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
        amrex::Real const axis_volume_factor =
            warpx.verboncoeurAxisCorrection() ? 1.0_rt / 3.0_rt
                                              : 1.0_rt / 4.0_rt;
#endif
        bool const conservative_pressure_work = m_conservative_pressure_work;

        // The exact particle/electron pressure-work pair is evaluated from
        // the frozen old pressure state.  Apply the electron-side increment
        // before the mass-consistent finite-volume remap.  This ordering is
        // essential at a moving material/vacuum front: the shared upwind
        // charge flux transports the increment with its material carrier, so
        // a node that becomes exact vacuum cannot retain an EOS-unrepresentable
        // nonzero energy density.  Periodic FV conservation leaves the global
        // particle/electron work identity unchanged.
        amrex::MultiFab pressure_work_loaded_energy;
        if (conservative_pressure_work) {
            pressure_work_loaded_energy.define(
                Ke.boxArray(), Ke.DistributionMap(), 1, Ke.nGrowVect(),
                amrex::MFInfo(), Ke.Factory());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(
                     pressure_work_loaded_energy, TilingIfNotGPU());
                 mfi.isValid(); ++mfi)
            {
                amrex::Array4<amrex::Real> const& loaded_energy =
                    pressure_work_loaded_energy.array(mfi);
                amrex::Array4<amrex::Real const> const& old_energy =
                    Ke.const_array(mfi);
                amrex::Array4<amrex::Real const> const& old_charge_density =
                    old_rho.const_array(mfi);
                amrex::Array4<amrex::Real const> const& pressure_state =
                    pressure_work_state->const_array(mfi);
                amrex::Array4<amrex::Real const> const& work_current_x =
                    pressure_work_current[0]->const_array(mfi);
                amrex::Array4<amrex::Real const> const& work_current_y =
                    pressure_work_current[1]->const_array(mfi);
                amrex::Array4<amrex::Real const> const& work_current_z =
                    pressure_work_current[2]->const_array(mfi);

                amrex::Box const box = amrex::convert(
                    mfi.tilebox(), Ke.ixType().toIntVect());
                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept
                {
                    amrex::Real const raw_old_rho =
                        old_charge_density(i, j, k);
                    amrex::Real const old_rho_node =
                        amrex::max(raw_old_rho, 0.0_rt);
                    amrex::Real const work_divergence =
                        qdsmc_pressure_work_velocity_divergence(
                            work_current_x, work_current_y, work_current_z,
                            pressure_state, i, j, k, inv_dx);
                    amrex::Real const source_loaded_energy =
                        old_energy(i, j, k) - dt
                            * pressure_state(i, j, k, 0)
                            * work_divergence;

                    ElectronThermodynamicsExecutor::MaterialMassDensities
                        material_mass_density{};
                    if (num_materials == 1) {
                        material_mass_density[0] = thermodynamics
                            .materialMassDensityFromPhysicalChargeDensity(
                                0, old_rho_node);
                    }
                    amrex::Real const source_loaded_temperature =
                        thermodynamics
                            .temperatureFromMaterialMassDensitiesThermalEnergyDensity(
                                old_rho_node, material_mass_density,
                                source_loaded_energy);
                    ElectronThermodynamicState const source_loaded_state =
                        thermodynamics
                            .stateFromMaterialMassDensitiesTemperature(
                                old_rho_node, material_mass_density,
                                source_loaded_temperature);
                    amrex::Real const energy_scale = amrex::max(
                        amrex::max(
                            std::abs(source_loaded_energy),
                            std::abs(
                                source_loaded_state.internal_energy_density)),
                        std::numeric_limits<amrex::Real>::min());
                    amrex::Real const energy_tolerance = 512.0_rt
                        * std::numeric_limits<amrex::Real>::epsilon()
                        * energy_scale;
                    if (raw_old_rho < 0.0_rt
                        || !amrex::Math::isfinite(work_divergence)
                        || !amrex::Math::isfinite(source_loaded_energy)
                        || !amrex::Math::isfinite(source_loaded_temperature)
                        || !amrex::Math::isfinite(
                            source_loaded_state.pressure)
                        || source_loaded_state.pressure < 0.0_rt
                        || !amrex::Math::isfinite(
                            source_loaded_state.internal_energy_density)
                        || std::abs(
                            source_loaded_state.internal_energy_density
                            - source_loaded_energy) > energy_tolerance)
                    {
                        loaded_energy(i, j, k) =
                            std::numeric_limits<amrex::Real>::quiet_NaN();
                        return;
                    }
                    loaded_energy(i, j, k) = source_loaded_energy;
                });
            }

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                pressure_work_loaded_energy.is_finite(0, 1, 0),
                "Conservative hybrid pressure work produced an old-state "
                "electron energy outside the EOS domain before its "
                "mass-consistent finite-volume transport. Reduce the "
                "timestep or check the pressure-work state.");
            ablastr::utils::communication::FillBoundary(
                pressure_work_loaded_energy,
                pressure_work_loaded_energy.nGrowVect(), false,
                geom.periodicity(), true);
        }

        // Keep failure categories separate.  Exact-vacuum fronts exercise
        // several distinct invariants, and a single NaN assertion otherwise
        // obscures whether the remedy is a smaller CFL timestep, a continuity
        // fix, or an EOS-domain change.
        amrex::Gpu::DeviceScalar<int> invalid_cfl(0);
        amrex::Gpu::DeviceScalar<int> invalid_flux_state(0);
        amrex::Gpu::DeviceScalar<int> invalid_continuity(0);
        amrex::Gpu::DeviceScalar<int> invalid_density(0);
        amrex::Gpu::DeviceScalar<int> invalid_vacuum_cleanup_bound(0);
        amrex::Gpu::DeviceScalar<int> invalid_vacuum_cleanup_path(0);
        amrex::Gpu::DeviceScalar<int> invalid_advected_eos(0);
        amrex::Gpu::DeviceScalar<int> invalid_predicted_eos(0);
        amrex::Gpu::DeviceScalar<int> invalid_final_eos(0);
        int* const invalid_cfl_ptr = invalid_cfl.dataPtr();
        int* const invalid_flux_state_ptr = invalid_flux_state.dataPtr();
        int* const invalid_continuity_ptr = invalid_continuity.dataPtr();
        int* const invalid_density_ptr = invalid_density.dataPtr();
        int* const invalid_vacuum_cleanup_bound_ptr =
            invalid_vacuum_cleanup_bound.dataPtr();
        int* const invalid_vacuum_cleanup_path_ptr =
            invalid_vacuum_cleanup_path.dataPtr();
        int* const invalid_advected_eos_ptr = invalid_advected_eos.dataPtr();
        int* const invalid_predicted_eos_ptr = invalid_predicted_eos.dataPtr();
        int* const invalid_final_eos_ptr = invalid_final_eos.dataPtr();

        // First form the conservative FV result without attempting an EOS
        // inverse.  A material node can become exact vacuum while the two
        // floating-point subtractions U* - dt div(F_U) leave a one-ulp
        // residual even though rho^n - dt div(F_rho) and deposited rho^{n+1}
        // are both exactly zero.  Store a deterministic face-paired cleanup
        // state for only that cancellation-sized residue.  Each source records
        // integrated energy removed from itself plus fixed outflow-face shares;
        // a later gather pass transfers those shares to endpoint material
        // support without atomics or decomposition-dependent accumulation.
        amrex::MultiFab advected_energy(
            Ke.boxArray(), Ke.DistributionMap(), 1, 0,
            amrex::MFInfo(), Ke.Factory());
        constexpr int vacuum_cleanup_components = 1 + 2 * AMREX_SPACEDIM;
        amrex::MultiFab vacuum_cleanup_source(
            Ke.boxArray(), Ke.DistributionMap(), vacuum_cleanup_components,
            amrex::IntVect::TheUnitVector(),
            amrex::MFInfo(), Ke.Factory());
        vacuum_cleanup_source.setVal(
            0.0_rt, 0, vacuum_cleanup_source.nComp(),
            vacuum_cleanup_source.nGrowVect());
        amrex::Real const cleanup_roundoff_factor = 4096.0_rt
            * std::numeric_limits<amrex::Real>::epsilon();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(advected_energy, TilingIfNotGPU());
             mfi.isValid(); ++mfi)
        {
            amrex::Array4<amrex::Real> const& advected =
                advected_energy.array(mfi);
            amrex::Array4<amrex::Real> const& cleanup =
                vacuum_cleanup_source.array(mfi);
            amrex::Array4<amrex::Real const> transport_energy =
                Ke.const_array(mfi);
            if (conservative_pressure_work) {
                transport_energy =
                    pressure_work_loaded_energy.const_array(mfi);
            }
            amrex::Array4<amrex::Real const> const& charge_density =
                rho.const_array(mfi);
            amrex::Array4<amrex::Real const> const& old_charge_density =
                old_rho.const_array(mfi);
            amrex::Array4<amrex::Real const> const& midpoint_charge_density =
                midpoint_rho.const_array(mfi);
            amrex::Array4<amrex::Real const> const& ion_current_x =
                ion_current[0]->const_array(mfi);
            amrex::Array4<amrex::Real const> const& ion_current_y =
                ion_current[1]->const_array(mfi);
            amrex::Array4<amrex::Real const> const& ion_current_z =
                ion_current[2]->const_array(mfi);
            amrex::Array4<amrex::Real const> const& plasma_current_x =
                plasma_current[0]->const_array(mfi);
            amrex::Array4<amrex::Real const> const& plasma_current_y =
                plasma_current[1]->const_array(mfi);
            amrex::Array4<amrex::Real const> const& plasma_current_z =
                plasma_current[2]->const_array(mfi);
            amrex::Array4<amrex::Real const> const& vx =
                Vex.const_array(mfi);
            amrex::Array4<amrex::Real const> const& vy =
                Vey.const_array(mfi);
            amrex::Array4<amrex::Real const> const& vz =
                Vez.const_array(mfi);
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
            amrex::ignore_unused(ion_current_y, plasma_current_y, vy);
#endif

            amrex::Box const box = amrex::convert(
                mfi.tilebox(), Te.ixType().toIntVect());
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (
                int i, int j, int k) noexcept
            {
                amrex::Real const deposited_rho_node =
                    charge_density(i, j, k);
                amrex::Real const old_rho_node =
                    old_charge_density(i, j, k);
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
                QdsmcCartesianTransportTerms const transport =
                    qdsmc_cylindrical_transport_terms(
                        transport_energy, old_charge_density,
                        midpoint_charge_density,
                        ion_current_x, ion_current_z,
                        plasma_current_x, plasma_current_z, vx, vz,
                        i, j, k, nodal_lo, nodal_hi, dx, periodic,
                        plasma_current_face_centered, axis_volume_factor,
                        problo, probhi,
                        physical_domain_lo, physical_domain_hi);
#else
                QdsmcCartesianTransportTerms const transport =
                    qdsmc_cartesian_transport_terms(
                        transport_energy, old_charge_density,
                        midpoint_charge_density,
                        ion_current_x, ion_current_y, ion_current_z,
                        plasma_current_x, plasma_current_y,
                        plasma_current_z, vx, vy, vz, i, j, k,
                        nodal_lo, nodal_hi, inv_dx, periodic,
                        plasma_current_face_centered);
#endif
                amrex::Real const outgoing_charge_flux =
                    transport.outgoing_charge_flux_per_volume;
                amrex::Real const cfl = old_rho_node > 0.0_rt
                    ? dt * outgoing_charge_flux / old_rho_node
                    : (outgoing_charge_flux == 0.0_rt
                        ? 0.0_rt
                        : std::numeric_limits<amrex::Real>::quiet_NaN());
                amrex::Real const predicted_charge_density =
                    old_rho_node - dt * transport.charge_flux_divergence;
                amrex::Real const advected_energy_density =
                    transport_energy(i, j, k)
                    - dt * transport.energy_flux_divergence;
                advected(i, j, k) = advected_energy_density;

                amrex::Real const continuity_scale = amrex::max(
                    amrex::max(
                        amrex::max(
                            std::abs(old_rho_node),
                            std::abs(deposited_rho_node)),
                        std::abs(predicted_charge_density)),
                    std::numeric_limits<amrex::Real>::min());
                amrex::Real const continuity_residual = std::abs(
                    predicted_charge_density - deposited_rho_node)
                    / continuity_scale;
                amrex::Real const continuity_tolerance = amrex::max(
                    1.0e-7_rt,
                    1024.0_rt * std::numeric_limits<amrex::Real>::epsilon());
                bool const bad_cfl = !amrex::Math::isfinite(cfl)
                    || cfl > 1.0_rt + 64.0_rt
                        * std::numeric_limits<amrex::Real>::epsilon();
                bool const bad_flux_state = !amrex::Math::isfinite(
                        transport.energy_flux_divergence)
                    || !amrex::Math::isfinite(
                        transport.velocity_divergence)
                    || !amrex::Math::isfinite(
                        transport.charge_flux_divergence)
                    || !amrex::Math::isfinite(
                        transport.absolute_energy_flux_per_volume)
                    || !amrex::Math::isfinite(
                        transport.absolute_charge_flux_per_volume)
                    || !amrex::Math::isfinite(advected_energy_density);
                bool const bad_continuity =
                    !amrex::Math::isfinite(continuity_residual)
                    || continuity_residual > continuity_tolerance;
                bool const bad_density = deposited_rho_node < 0.0_rt;
                if (bad_cfl || bad_flux_state || bad_continuity || bad_density)
                {
                    if (bad_cfl) {
                        amrex::HostDevice::Atomic::Add(invalid_cfl_ptr, 1);
                    }
                    if (bad_flux_state) {
                        amrex::HostDevice::Atomic::Add(
                            invalid_flux_state_ptr, 1);
                    }
                    if (bad_continuity) {
                        amrex::HostDevice::Atomic::Add(
                            invalid_continuity_ptr, 1);
                    }
                    if (bad_density) {
                        amrex::HostDevice::Atomic::Add(
                            invalid_density_ptr, 1);
                    }
                    advected(i, j, k) =
                        std::numeric_limits<amrex::Real>::quiet_NaN();
                    return;
                }

                if (deposited_rho_node != 0.0_rt
                    || advected_energy_density == 0.0_rt)
                {
                    return;
                }

                amrex::Real const charge_cancellation_scale = amrex::max(
                    std::abs(old_rho_node)
                        + dt * transport.absolute_charge_flux_per_volume,
                    std::numeric_limits<amrex::Real>::min());
                amrex::Real const energy_cancellation_scale = amrex::max(
                    std::abs(transport_energy(i, j, k))
                        + dt * transport.absolute_energy_flux_per_volume,
                    std::numeric_limits<amrex::Real>::min());
                bool const roundoff_sized = old_rho_node > 0.0_rt
                    && std::abs(predicted_charge_density)
                        <= cleanup_roundoff_factor
                            * charge_cancellation_scale
                    && std::abs(advected_energy_density)
                        <= cleanup_roundoff_factor
                            * energy_cancellation_scale;
                if (!roundoff_sized) {
                    amrex::HostDevice::Atomic::Add(
                        invalid_vacuum_cleanup_bound_ptr, 1);
                    return;
                }

                amrex::Real total_outflow = 0.0_rt;
                int last_face = -1;
                bool invalid_path = false;
                for (int face = 0; face < 2 * AMREX_SPACEDIM; ++face) {
                    amrex::Real const face_outflow = dt
                        * transport
                            .outgoing_face_charge_flux_per_volume[face];
                    if (!(face_outflow > 0.0_rt)) { continue; }
                    int const direction = face / 2;
                    int const side = (face % 2 == 0) ? -1 : 1;
                    int const ni = i + (direction == 0 ? side : 0);
                    int const nj = j + (direction == 1 ? side : 0);
                    int const nk = k + (direction == 2 ? side : 0);
                    if (!(charge_density(ni, nj, nk) > 0.0_rt)) {
                        invalid_path = true;
                    } else {
                        total_outflow += face_outflow;
                        last_face = face;
                    }
                }
                if (invalid_path || !(total_outflow > 0.0_rt)) {
                    amrex::HostDevice::Atomic::Add(
                        invalid_vacuum_cleanup_path_ptr, 1);
                    return;
                }

                amrex::Real const source_volume =
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
                    hybrid_transport_node_volume(
                        i, j, k, problo, dx, physical_domain_lo,
                        axis_volume_factor);
#else
                    hybrid_node_volume(
                        i, j, k, problo, probhi, dx,
                        physical_domain_lo, physical_domain_hi, periodic);
#endif
                amrex::Real const integrated_residual =
                    advected_energy_density * source_volume;
                cleanup(i, j, k, 0) = integrated_residual;
                amrex::Real assigned_energy = 0.0_rt;
                for (int face = 0; face < 2 * AMREX_SPACEDIM; ++face) {
                    amrex::Real const face_outflow = dt
                        * transport
                            .outgoing_face_charge_flux_per_volume[face];
                    if (!(face_outflow > 0.0_rt)) { continue; }
                    amrex::Real const face_energy = face == last_face
                        ? integrated_residual - assigned_energy
                        : integrated_residual
                            * face_outflow / total_outflow;
                    cleanup(i, j, k, face + 1) = face_energy;
                    assigned_energy += face_energy;
                }
            });
        }

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            invalid_cfl.dataValue() == 0,
            "Nonlinear finite-volume electron-energy transport exceeded its "
            "upwind CFL limit. Reduce the timestep.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            invalid_flux_state.dataValue() == 0,
            "Nonlinear finite-volume electron-energy transport encountered "
            "a non-finite charge, energy, or velocity flux. Check material/"
            "vacuum donor support and deposited currents.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            invalid_continuity.dataValue() == 0,
            "Nonlinear finite-volume electron-energy transport found that "
            "the endpoint charge and dedicated charge flux violate their "
            "discrete continuity equation.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            invalid_density.dataValue() == 0,
            "Nonlinear finite-volume electron-energy transport encountered "
            "a negative deposited electron charge-density magnitude.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            invalid_vacuum_cleanup_bound.dataValue() == 0,
            "An exact-vacuum endpoint retained electron energy larger than "
            "the finite-volume roundoff-cancellation bound. This is a "
            "physical transport/EOS failure, not roundoff cleanup.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            invalid_vacuum_cleanup_path.dataValue() == 0,
            "Finite-volume roundoff energy at an evacuated node has no "
            "positive-support endpoint along every material outflow face.");

        ablastr::utils::communication::FillBoundary(
            vacuum_cleanup_source,
            vacuum_cleanup_source.nGrowVect(), false,
            geom.periodicity(), true);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(advected_energy, TilingIfNotGPU());
             mfi.isValid(); ++mfi)
        {
            amrex::Array4<amrex::Real> const& advected =
                advected_energy.array(mfi);
            amrex::Array4<amrex::Real const> const& cleanup =
                vacuum_cleanup_source.const_array(mfi);
            amrex::Box const box = amrex::convert(
                mfi.tilebox(), Te.ixType().toIntVect());
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (
                int i, int j, int k) noexcept
            {
                amrex::Real const target_volume =
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
                    hybrid_transport_node_volume(
                        i, j, k, problo, dx, physical_domain_lo,
                        axis_volume_factor);
#else
                    hybrid_node_volume(
                        i, j, k, problo, probhi, dx,
                        physical_domain_lo, physical_domain_hi, periodic);
#endif
                amrex::Real const removed_energy = cleanup(i, j, k, 0);
                amrex::Real incoming_energy = 0.0_rt;
#if AMREX_SPACEDIM >= 1
                incoming_energy += cleanup(i-1, j, k, 2);
                incoming_energy += cleanup(i+1, j, k, 1);
#endif
#if AMREX_SPACEDIM >= 2
                incoming_energy += cleanup(i, j-1, k, 4);
                incoming_energy += cleanup(i, j+1, k, 3);
#endif
#if AMREX_SPACEDIM == 3
                incoming_energy += cleanup(i, j, k-1, 6);
                incoming_energy += cleanup(i, j, k+1, 5);
#endif
                // Multiplying U_res by V and dividing it back can leave the
                // same one-ulp residue.  No cleanup path may target an
                // endpoint-vacuum source, so enforce its representable state
                // bitwise and gather only incoming energy onto material.
                if (removed_energy != 0.0_rt) {
                    advected(i, j, k) = 0.0_rt;
                }
                advected(i, j, k) += incoming_energy / target_volume;
            });
        }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Array4<amrex::Real> const& temperature = Te.array(mfi);
            amrex::Array4<amrex::Real const> const& advected_energy_arr =
                advected_energy.const_array(mfi);
            amrex::Array4<amrex::Real const> const& old_energy =
                Ke.const_array(mfi);
            amrex::Array4<amrex::Real const> transport_energy = old_energy;
            if (conservative_pressure_work) {
                transport_energy =
                    pressure_work_loaded_energy.const_array(mfi);
            }
            amrex::Array4<amrex::Real const> const& charge_density =
                rho.const_array(mfi);
            amrex::Array4<amrex::Real const> const& old_charge_density =
                old_rho.const_array(mfi);
            amrex::Array4<amrex::Real const> const& midpoint_charge_density =
                midpoint_rho.const_array(mfi);
            amrex::Array4<amrex::Real const> const& ion_current_x =
                ion_current[0]->const_array(mfi);
            amrex::Array4<amrex::Real const> const& ion_current_y =
                ion_current[1]->const_array(mfi);
            amrex::Array4<amrex::Real const> const& ion_current_z =
                ion_current[2]->const_array(mfi);
            amrex::Array4<amrex::Real const> const& plasma_current_x =
                plasma_current[0]->const_array(mfi);
            amrex::Array4<amrex::Real const> const& plasma_current_y =
                plasma_current[1]->const_array(mfi);
            amrex::Array4<amrex::Real const> const& plasma_current_z =
                plasma_current[2]->const_array(mfi);
            amrex::Array4<amrex::Real const> const& vx =
                Vex.const_array(mfi);
            amrex::Array4<amrex::Real const> const& vy =
                Vey.const_array(mfi);
            amrex::Array4<amrex::Real const> const& vz =
                Vez.const_array(mfi);
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
            amrex::ignore_unused(ion_current_y, plasma_current_y, vy);
#endif

            amrex::Box const box = amrex::convert(
                mfi.tilebox(), Te.ixType().toIntVect());
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (
                int i, int j, int k) noexcept
            {
                amrex::Real const deposited_rho_node =
                    charge_density(i, j, k);
                amrex::Real const rho_node =
                    amrex::max(deposited_rho_node, 0.0_rt);
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
                QdsmcCartesianTransportTerms const transport =
                    qdsmc_cylindrical_transport_terms(
                        transport_energy, old_charge_density,
                        midpoint_charge_density,
                        ion_current_x, ion_current_z,
                        plasma_current_x, plasma_current_z, vx, vz,
                        i, j, k, nodal_lo, nodal_hi, dx, periodic,
                        plasma_current_face_centered, axis_volume_factor,
                        problo, probhi,
                        physical_domain_lo, physical_domain_hi);
#else
                QdsmcCartesianTransportTerms const transport =
                    qdsmc_cartesian_transport_terms(
                        transport_energy, old_charge_density,
                        midpoint_charge_density,
                        ion_current_x, ion_current_y, ion_current_z,
                        plasma_current_x, plasma_current_y,
                        plasma_current_z, vx, vy, vz, i, j, k,
                        nodal_lo, nodal_hi, inv_dx, periodic,
                        plasma_current_face_centered);
#endif
                amrex::Real pressure_heun_divergence =
                    transport.velocity_divergence;
                amrex::Real plasma_ratio_divergence = 0.0_rt;
                if (conservative_pressure_work) {
                    // V_e = V_i - J_plasma/rho.  The ion part is replaced by
                    // the frozen exact Boris work pair source-loaded into
                    // transport_energy before this FV remap.  Evaluate the
                    // sole retained +P div(J_plasma/rho) term directly rather
                    // than subtracting two independently discretized ion and
                    // electron divergences.  The latter leaves a spurious
                    // residual at an exact-vacuum front even when J_plasma=0.
                    plasma_ratio_divergence =
                        qdsmc_cartesian_ratio_velocity_divergence(
                            plasma_current_x, plasma_current_y,
                            plasma_current_z, midpoint_charge_density,
                            i, j, k, nodal_lo, nodal_hi, inv_dx, periodic);
                    pressure_heun_divergence = -plasma_ratio_divergence;
                }
                amrex::Real const advected_energy_density =
                    advected_energy_arr(i, j, k);

                ElectronThermodynamicsExecutor::MaterialMassDensities
                    material_mass_density{};
                if (num_materials == 1) {
                    material_mass_density[0] = thermodynamics
                        .materialMassDensityFromPhysicalChargeDensity(
                            0, rho_node);
                }
                amrex::Real const advected_temperature = thermodynamics
                    .temperatureFromMaterialMassDensitiesThermalEnergyDensity(
                        rho_node, material_mass_density,
                        advected_energy_density);
                ElectronThermodynamicState const advected_state =
                    thermodynamics.stateFromMaterialMassDensitiesTemperature(
                        rho_node, material_mass_density,
                        advected_temperature);
                amrex::Real const advected_energy_scale = amrex::max(
                    amrex::max(
                        std::abs(advected_energy_density),
                        std::abs(advected_state.internal_energy_density)),
                    std::numeric_limits<amrex::Real>::min());
                amrex::Real const advected_energy_tolerance = 512.0_rt
                    * std::numeric_limits<amrex::Real>::epsilon()
                    * advected_energy_scale;
                amrex::Real const predicted_energy_density =
                    advected_energy_density
                    - dt * pressure_heun_divergence
                        * advected_state.pressure;
                amrex::Real const predicted_temperature = thermodynamics
                    .temperatureFromMaterialMassDensitiesThermalEnergyDensity(
                        rho_node, material_mass_density,
                        predicted_energy_density);
                ElectronThermodynamicState const predicted_state =
                    thermodynamics.stateFromMaterialMassDensitiesTemperature(
                        rho_node, material_mass_density,
                        predicted_temperature);
                amrex::Real const predicted_energy_scale = amrex::max(
                    amrex::max(
                        std::abs(predicted_energy_density),
                        std::abs(predicted_state.internal_energy_density)),
                    std::numeric_limits<amrex::Real>::min());
                amrex::Real const predicted_energy_tolerance = 512.0_rt
                    * std::numeric_limits<amrex::Real>::epsilon()
                    * predicted_energy_scale;
                amrex::Real const final_energy_density =
                    advected_energy_density
                    - 0.5_rt * dt * pressure_heun_divergence
                        * (advected_state.pressure + predicted_state.pressure);
                amrex::Real const final_temperature = thermodynamics
                    .temperatureFromMaterialMassDensitiesThermalEnergyDensity(
                        rho_node, material_mass_density,
                        final_energy_density);
                ElectronThermodynamicState const final_state =
                    thermodynamics.stateFromMaterialMassDensitiesTemperature(
                        rho_node, material_mass_density, final_temperature);
                amrex::Real const energy_scale = amrex::max(
                    amrex::max(
                        std::abs(final_energy_density),
                        std::abs(final_state.internal_energy_density)),
                    std::numeric_limits<amrex::Real>::min());
                amrex::Real const energy_tolerance = 512.0_rt
                    * std::numeric_limits<amrex::Real>::epsilon()
                    * energy_scale;
                bool const bad_advected_eos =
                    !amrex::Math::isfinite(advected_temperature)
                    || !amrex::Math::isfinite(advected_state.pressure)
                    || advected_state.pressure < 0.0_rt
                    || !amrex::Math::isfinite(
                        advected_state.internal_energy_density)
                    || std::abs(
                        advected_state.internal_energy_density
                        - advected_energy_density)
                        > advected_energy_tolerance;
                bool const bad_predicted_eos =
                    !amrex::Math::isfinite(predicted_energy_density)
                    || !amrex::Math::isfinite(predicted_temperature)
                    || !amrex::Math::isfinite(predicted_state.pressure)
                    || predicted_state.pressure < 0.0_rt
                    || !amrex::Math::isfinite(
                        predicted_state.internal_energy_density)
                    || std::abs(
                        predicted_state.internal_energy_density
                        - predicted_energy_density)
                        > predicted_energy_tolerance;
                bool const bad_final_eos =
                    !amrex::Math::isfinite(final_energy_density)
                    || !amrex::Math::isfinite(final_temperature)
                    || !amrex::Math::isfinite(final_state.pressure)
                    || final_state.pressure < 0.0_rt
                    || !amrex::Math::isfinite(
                        final_state.internal_energy_density)
                    || std::abs(
                        final_state.internal_energy_density
                        - final_energy_density) > energy_tolerance;
                if (bad_advected_eos || bad_predicted_eos || bad_final_eos)
                {
                    if (bad_advected_eos) {
                        amrex::HostDevice::Atomic::Add(
                            invalid_advected_eos_ptr, 1);
                    }
                    if (bad_predicted_eos) {
                        amrex::HostDevice::Atomic::Add(
                            invalid_predicted_eos_ptr, 1);
                    }
                    if (bad_final_eos) {
                        amrex::HostDevice::Atomic::Add(
                            invalid_final_eos_ptr, 1);
                    }
                    temperature(i, j, k) =
                        std::numeric_limits<amrex::Real>::quiet_NaN();
                    return;
                }
                temperature(i, j, k) = final_temperature;
            });
        }

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            invalid_advected_eos.dataValue() == 0,
            "The mass-consistent nonlinear electron-energy remap produced an "
            "advected state outside the configured EOS domain.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            invalid_predicted_eos.dataValue() == 0,
            "The predicted nonlinear electron pressure-work state is outside "
            "the configured EOS domain. Reduce the timestep.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            invalid_final_eos.dataValue() == 0,
            "The corrected nonlinear electron pressure-work state is outside "
            "the configured EOS domain. Reduce the timestep.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            Te.is_finite(0, 1, 0),
            "Nonlinear finite-volume electron-energy transport exceeded its "
            "upwind CFL limit, violated discrete rho/current continuity, or "
            "encountered an invalid EOS/pressure-work state. Reduce the "
            "timestep and check for particle creation/deletion, injection, "
            "reactive collisions, resampling, or unsupported material "
            "boundary fluxes.");
        ablastr::utils::communication::FillBoundary(
            Te, Te.nGrowVect(), false, geom.periodicity(), true);
        return;
    }

    // Note: T_e is NOT zeroed here. Cells that received no QDSMC weight
    // keep their previous T_e -- zeroing them would erase valid state (and
    // seed a wrong K_e into neighbors on the next step) whenever a cell
    // momentarily receives no deposit.

    auto const gamma      = m_gamma;
    auto const n_floor    = m_n_floor;
    auto const kb_over_qe = PhysConst::kb / PhysConst::q_e;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real>       const & Te_arr      = Te.array(mfi);
        amrex::Array4<amrex::Real const> const & Ke_arr      = Ke.const_array(mfi);
        amrex::Array4<amrex::Real const> const & weights_arr = weights.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr     = rho.const_array(mfi);

        amrex::Box const tbox = amrex::convert(mfi.tilebox(), Te.ixType().toIntVect());
        amrex::Box       box  = tbox;
        box.grow(Te.nGrowVect());

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Guard the division: a cell no QDSMC marker reached has exactly
            // zero deposited weight and keeps its previous T_e. Cells that did
            // receive weight are all updated, however small the deposit -- the
            // (K*N)/N ratio is well conditioned there because numerator and
            // denominator carry the same small factor.
            if (weights_arr(i,j,k) <= 0.0_rt) { return; }
            amrex::Real const w = weights_arr(i,j,k);
            // Floored density, mirroring ideal-gas QDSMC initialization:
            // below-floor
            // cells are updated too (insulating halo), and the K <-> T_e
            // conversion uses the same n_e^(gamma-1) factor on both sides of
            // the step, so a cell whose marker did not move keeps its T_e
            // exactly.
            amrex::Real const ne =
                amrex::max(rho_arr(i,j,k) / PhysConst::q_e, n_floor);
            Te_arr(i,j,k) = Ke_arr(i,j,k)
                          / std::pow(ne, 1.0_rt - gamma)
                          / w
                          / kb_over_qe;
        });
    }
    // No ghost exchange: the kernel runs on the ghost-grown box and its
    // inputs already have valid ghosts (the QDSMC deposits SumBoundary with
    // dst_ng = nGrowVect(), and rho_fp was FillBoundary'd after deposition).
}


void HybridPICModel::QDSMCAddJouleHeating (int const lev, amrex::Real const dt,
                                           amrex::MultiFab * const redirect_E) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCAddJouleHeating()");

    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Per-cell resistive electron-heating source (Phys. Plasmas 31, 012902 (2024), Eq. 12).
    // With the e-i relative drift V_s - V_e = J_plasma/(e n_e) and the
    // eta-derived rate nu_{s,e} = Z_s e^2 eta n_e / m_s, the source is
    //
    //   S_e = e^2 eta n_e Sum_s Z_s n_s |J_plasma/(e n_e)|^2
    //
    // which collapses to eta J^2 for a single species. Computed on the grid from
    // rho_fp, rho_fp_s and the plasma current -- no per-particle scatter.

    auto & warpx = WarpX::GetInstance();
    amrex::Periodicity const & period = warpx.Geom(lev).periodicity();

    amrex::MultiFab       & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp,                         lev);
    amrex::MultiFab& joule_step =
        *warpx.m_fields.get("hybrid_joule_electron_energy_fp", lev);
    amrex::MultiFab& joule_cumulative = *warpx.m_fields.get(
        "hybrid_joule_electron_energy_cumulative_fp", lev);
    joule_step.setVal(0.0_rt);
    ablastr::fields::VectorField J_plasma =
        warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);

    auto const rho_floor     = PhysConst::q_e * m_n_floor;
    auto const eta           = m_eta;
    auto const t_new         = warpx.gett_new(0);
    auto const thermodynamics = m_electron_thermodynamics.executor();
    int const num_materials = m_electron_thermodynamics.numMaterials();
    amrex::GpuArray<amrex::MultiFab const*,
                    ElectronThermodynamicsExecutor::max_materials>
        material_unit_charge_density_fields{};
    for (int material = 0; material < num_materials; ++material) {
        auto& material_field = *warpx.m_fields.get(
            "ni_charge_fp_"
            + m_electron_thermodynamics.materialSpeciesName(material), lev);
        material_field.FillBoundary(
            material_field.nGrowVect(), period);
        material_unit_charge_density_fields[material] = &material_field;
    }

    amrex::GpuArray<int, 3> const & Jx_stag = Jx_IndexType;
    amrex::GpuArray<int, 3> const & Jy_stag = Jy_IndexType;
    amrex::GpuArray<int, 3> const & Jz_stag = Jz_IndexType;
    amrex::GpuArray<int, 3> const nodal     = {1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen   = {1, 1, 1};

    // Te-threshold Joule redirection: in cells with Te >= threshold the Joule
    // heat is written into redirect_E (per charged ion species, the m_i-independent
    // energy E_s [J]) for QDSMCApplyIonHeating to deposit on the ions, rather than
    // added to T_e.
    bool const do_redirect = (redirect_E != nullptr);
    auto const K_per_eV    = PhysConst::q_e / PhysConst::kb;        // T[eV]*this = T[K]
    amrex::Real const Te_thresh_K = m_joule_redirect_Te_eV * K_per_eV;

    auto & mypc = warpx.GetPartContainer();

    // Loop over every charged ion species and apply its per-cell contribution
    // through the configured caloric inverse:
    //   Delta U_e,s = dt Z_s e^2 eta n_e n_s |dV|^2.
    // For an ideal gas this is algebraically the historical direct dT update;
    // for latent and table EOS it deposits the same Joule energy without
    // assuming a constant heat capacity.
    //
    // n_s is recovered from the species charge fraction rather than from
    // rho_fp_s/q_e directly: the per-species deposits are physical (volume-
    // scaled in radial geometries) but unfiltered and not boundary-treated,
    // while n_e comes from the fully processed total rho_fp used by the
    // E-solve. Taking
    //
    //   f_s = rho_fp_s / Sigma_t rho_fp_t   =   Z_s n_s / n_e   (unitless)
    //   n_s = f_s * n_e / Z_s
    //
    // keeps n_s consistent with that n_e in any dimensionality (numerator
    // and denominator of f_s share identical processing).
    auto const species_names = mypc.GetSpeciesNames();

    // Sigma_t rho_fp_t (physical per-species charge densities), used for the
    // species fraction f_s = rho_fp_s / rhos_sum per cell inside the species
    // loop. Filled once per step by HybridPICDepositRhoAndJ.
    amrex::MultiFab const & rhos_sum =
        *warpx.m_fields.get("hybrid_rho_species_sum_fp", lev);

    // Charged-species component index for redirect_E (matches QDSMCApplyIonHeating).
    int ion_comp = -1;
    for (auto const & spec_name : species_names) {
        auto & pc = mypc.GetParticleContainerFromName(spec_name);
        if (pc.getCharge() == 0._prt) { continue; }
        ++ion_comp;

        amrex::Real const Z_s = pc.getCharge() / PhysConst::q_e;

        amrex::MultiFab const & rho_s =
            *warpx.m_fields.get("rho_fp_" + spec_name, lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Array4<amrex::Real>       const & Te_arr     = Te.array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr    = rho.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rhos_arr   = rho_s.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rhosum_arr = rhos_sum.const_array(mfi);
            amrex::Array4<amrex::Real const> const & Jpx        = J_plasma[0]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Jpy        = J_plasma[1]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Jpz        = J_plasma[2]->const_array(mfi);
            amrex::Array4<amrex::Real> const& joule_step_arr =
                joule_step.array(mfi);
            amrex::Array4<amrex::Real> const& joule_cumulative_arr =
                joule_cumulative.array(mfi);
            ElectronThermodynamicsExecutor::MaterialChargeDensityArrays
                material_unit_charge_density{};
            for (int material = 0; material < num_materials; ++material) {
                material_unit_charge_density[material] =
                    material_unit_charge_density_fields[material]
                        ->const_array(mfi);
            }

            // Redirect output (default Array4 when redirect off -> never indexed
            // because do_redirect gates the write).
            amrex::Array4<amrex::Real> redirect_arr;
            if (do_redirect) { redirect_arr = redirect_E->array(mfi); }

            amrex::Box const & tbox = mfi.tilebox();
            amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real const rho_val = rho_arr(i,j,k);
                if (rho_val <= rho_floor) { return; }
                // n_e (m^-3) from the volume-scaled total rho_fp.
                amrex::Real const ne = rho_val / PhysConst::q_e;
                // Species charge fraction f_s = rho_fp_s / Sigma_t rho_fp_t
                // = Z_s n_s / n_e (unitless; both sides physical and
                // identically processed). Then the per-species number
                // density: n_s = f_s * n_e / Z_s
                amrex::Real const rhos_val      = rhos_arr(i,j,k);
                amrex::Real const rhos_sum_val  = std::max(rhosum_arr(i,j,k), rho_floor);
                amrex::Real const f_s           = rhos_val / rhos_sum_val;
                amrex::Real const ns            = f_s * ne / Z_s;

                // |J| at the nodal grid (where Te lives), for the eta parser.
                auto const jx = ablastr::coarsen::sample::Interp(Jpx, Jx_stag, nodal, coarsen, i, j, k, 0);
                auto const jy = ablastr::coarsen::sample::Interp(Jpy, Jy_stag, nodal, coarsen, i, j, k, 0);
                auto const jz = ablastr::coarsen::sample::Interp(Jpz, Jz_stag, nodal, coarsen, i, j, k, 0);
                amrex::Real const Jmag = std::sqrt(jx*jx + jy*jy + jz*jz);

                // eta: same Ohm's-law parser the E-solve uses, evaluated
                // per cell. This makes the per-cell heat reduce to eta J^2
                // exactly in single species.
                amrex::Real const eta_s_eff = eta(rho_val, Jmag, t_new);

                // e-i relative drift = J_plasma/(e n_e), from the nodal plasma
                // current and n_e. Energy-consistent with the eta*J dissipation
                // in Ohm's law; reduces to eta*|J|^2 for a single species.
                amrex::Real const inv_ene = 1.0_rt / (PhysConst::q_e * ne);
                amrex::Real const dvx = jx * inv_ene;
                amrex::Real const dvy = jy * inv_ene;
                amrex::Real const dvz = jz * inv_ene;
                amrex::Real const dv2 = dvx*dvx + dvy*dvy + dvz*dvz;

                // Per-species source integrated over this PIC step [J/m^3].
                amrex::Real const source_energy_density = dt * Z_s
                    * PhysConst::q_e * PhysConst::q_e * eta_s_eff
                    * ne * ns * dv2;
                // Te-threshold redirection: below threshold heat electrons (the
                // usual Joule deposit); at/above it write this species'
                // m_i-independent redirected energy E_s = (2/3) n_e Z_s e^2 eta
                // |dV|^2 dt [J] into its component for the ion-heating step.
                if (do_redirect && Te_arr(i,j,k) >= Te_thresh_K) {
                    redirect_arr(i,j,k,ion_comp) = (2.0_rt/3.0_rt) * ne
                        * Z_s * PhysConst::q_e * PhysConst::q_e * eta_s_eff * dv2 * dt;
                } else {
                    auto const material_mass_density = thermodynamics
                        .materialMassDensitiesFromChargeDensityArrays(
                            material_unit_charge_density, i, j, k);
                    ElectronEnergySourceUpdate const source_update =
                        thermodynamics.applyEnergyDensityIncrement(
                            rho_val, material_mass_density,
                            Te_arr(i,j,k), source_energy_density);
                    if (!source_update.valid) {
                        Te_arr(i,j,k) =
                            std::numeric_limits<amrex::Real>::quiet_NaN();
                        joule_step_arr(i,j,k) =
                            std::numeric_limits<amrex::Real>::quiet_NaN();
                        return;
                    }
                    Te_arr(i,j,k) = source_update.temperature;
                    joule_step_arr(i,j,k) +=
                        source_update.energy_change_density;
                    joule_cumulative_arr(i,j,k) +=
                        source_update.energy_change_density;
                }
            });
        }
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        Te.is_finite(0, 1, 0) && joule_step.is_finite(0, 1, 0)
            && joule_cumulative.is_finite(0, 1, 0),
        "The hybrid Joule source could not realize the requested electron "
        "energy increment inside the configured caloric EOS domain.");
    Te.FillBoundary(Te.nGrowVect(), period);
    joule_step.FillBoundary(joule_step.nGrowVect(), period);
    joule_cumulative.FillBoundary(
        joule_cumulative.nGrowVect(), period);
}


void HybridPICModel::QDSMCAddTemperatureRelaxation (int const lev, amrex::Real const dt,
    std::map<std::string, amrex::MultiFab*> const & Ti_dep_by_species) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCAddTemperatureRelaxation()");

    using warpx::fields::FieldType;

    // Electron-ion thermal-equilibration sink, summed over ion species s:
    //   Q_ei = Sigma_s 3 n_s k_B nu_ei (T_e - T_i_s),    dU_e/dt += -Q_ei.
    // With U_e = n_e k_B T_e/(gamma-1), the per-cell T_e obeys
    //   dT_e/dt = -(gamma-1) 3 Sigma_s (n_s/n_e) nu_ei (T_e - T_i_s),
    // where n_s/n_e = f_s/Z_s, f_s = rho_fp_s/Sigma_t rho_fp_t. T_e is stored in
    // Kelvin; T_i (deposited per species, cell-centered, in eV) is interpolated
    // to the nodal T_e grid and converted to K. This is the electron-side sink;
    // QDSMCApplyIonHeating projects a stochastic ion proposal onto the exact
    // per-species ledger while preserving cell ion momentum.
    auto & warpx = WarpX::GetInstance();
    amrex::Periodicity const & period = warpx.Geom(lev).periodicity();

    amrex::MultiFab       & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp, lev);
    amrex::MultiFab& qei_step =
        *warpx.m_fields.get("hybrid_qei_electron_energy_fp", lev);
    amrex::MultiFab& qei_cumulative = *warpx.m_fields.get(
        "hybrid_qei_electron_energy_cumulative_fp", lev);
    qei_step.setVal(0.0_rt);

    auto const rho_floor     = PhysConst::q_e * m_n_floor;
    auto const nu_ei         = m_nu_ei;
    auto const t_new         = warpx.gett_new(0);
    auto const K_per_eV      = PhysConst::q_e / PhysConst::kb;   // T[eV] * this = T[K]
    // Floor on T_e in the nu_ei rate argument so pow(Te,-1.5) stays finite.
    amrex::Real const Te_floor_eV = 1.e-3_rt;

    amrex::GpuArray<int, 3> const nodal   = {1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};

    auto & mypc = warpx.GetPartContainer();
    auto const species_names = mypc.GetSpeciesNames();
    auto const thermodynamics = m_electron_thermodynamics.executor();
    int const num_materials = m_electron_thermodynamics.numMaterials();
    amrex::GpuArray<amrex::MultiFab const*,
                    ElectronThermodynamicsExecutor::max_materials>
        material_unit_charge_density_fields{};
    for (int material = 0; material < num_materials; ++material) {
        auto& material_field = *warpx.m_fields.get(
            "ni_charge_fp_"
            + m_electron_thermodynamics.materialSpeciesName(material), lev);
        material_field.FillBoundary(
            material_field.nGrowVect(), period);
        material_unit_charge_density_fields[material] = &material_field;
    }

    // Sigma_t rho_fp_t (physical per-species charge densities) -> species
    // fraction. Filled once per step by HybridPICDepositRhoAndJ.
    amrex::MultiFab const & rhos_sum =
        *warpx.m_fields.get("hybrid_rho_species_sum_fp", lev);

    // Cell-centered field box array (for staging the deposited T_i with a guard
    // cell so it can be interpolated to the nodal T_e grid).
    amrex::BoxArray const cc_ba = amrex::convert(Te.boxArray(), amrex::IntVect::TheCellVector());

    for (auto const & spec_name : species_names) {
        auto & pc = mypc.GetParticleContainerFromName(spec_name);
        if (pc.getCharge() == 0._prt) { continue; }
        amrex::Real const Z_s = pc.getCharge() / PhysConst::q_e;
        int material_index = -1;
        for (int material = 0; material < num_materials; ++material) {
            if (spec_name
                == m_electron_thermodynamics.materialSpeciesName(material))
            {
                material_index = material;
            }
        }

        amrex::MultiFab const & rho_s = *warpx.m_fields.get("rho_fp_" + spec_name, lev);
        amrex::MultiFab& qei_ion_temperature = *warpx.m_fields.get(
            "hybrid_qei_ion_temperature_fp_" + spec_name, lev);
        amrex::MultiFab& qei_electron_temperature_before = *warpx.m_fields.get(
            "hybrid_qei_electron_temperature_before_fp_" + spec_name, lev);
        amrex::MultiFab& qei_species_step = *warpx.m_fields.get(
            "hybrid_qei_electron_energy_fp_" + spec_name, lev);
        qei_species_step.setVal(0.0_rt);

        // Per-cell ion temperature [eV] (NGP velocity-variance deposit, done once
        // by the caller and shared via Ti_dep_by_species), moved onto the field's
        // cell-centered grid with one guard cell so the cc->nodal interpolation
        // has its neighbours at box edges.
        amrex::MultiFab const & Ti_dep = *Ti_dep_by_species.at(spec_name);
        amrex::MultiFab Ti_cc(cc_ba, Te.DistributionMap(), 1, 1);
        Ti_cc.setVal(0.0_rt);
        Ti_cc.ParallelCopy(Ti_dep, 0, 0, 1, amrex::IntVect::TheZeroVector(),
                           amrex::IntVect::TheZeroVector());
        Ti_cc.FillBoundary(warpx.Geom(lev).periodicity());
        // Ti_cc is cell-centered in the real dimensions; the unused (2D/1D)
        // dimensions are set NODAL so they match the nodal destination grid in
        // Interp (sf==sc there -> np=1, no out-of-bounds k=-1 read). Mirrors the
        // unused-dimension handling for J/B/E above.
        amrex::GpuArray<int, 3> cc_stag = {0, 0, 0};
        for (int d = AMREX_SPACEDIM; d < 3; ++d) { cc_stag[d] = 1; }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Array4<amrex::Real>       const & Te_arr     = Te.array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr    = rho.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rhos_arr   = rho_s.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rhosum_arr = rhos_sum.const_array(mfi);
            amrex::Array4<amrex::Real const> const & Ti_arr     = Ti_cc.const_array(mfi);
            amrex::Array4<amrex::Real> const& qei_step_arr =
                qei_step.array(mfi);
            amrex::Array4<amrex::Real> const& qei_cumulative_arr =
                qei_cumulative.array(mfi);
            amrex::Array4<amrex::Real> const& qei_ion_temperature_arr =
                qei_ion_temperature.array(mfi);
            amrex::Array4<amrex::Real> const&
                qei_electron_temperature_before_arr =
                    qei_electron_temperature_before.array(mfi);
            amrex::Array4<amrex::Real> const& qei_species_step_arr =
                qei_species_step.array(mfi);
            ElectronThermodynamicsExecutor::MaterialChargeDensityArrays
                material_unit_charge_density{};
            for (int material = 0; material < num_materials; ++material) {
                material_unit_charge_density[material] =
                    material_unit_charge_density_fields[material]
                        ->const_array(mfi);
            }

            amrex::Box const & tbox = mfi.tilebox();
            amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real const rho_val = rho_arr(i,j,k);
                if (rho_val <= rho_floor) { return; }
                amrex::Real const rhos_sum_val =
                    std::max(rhosum_arr(i,j,k), rho_floor);
                amrex::Real const f_s =
                    rhos_arr(i,j,k) / rhos_sum_val;
                amrex::Real const electron_density =
                    rho_val / PhysConst::q_e;
                // A material composition field deposits q_e*n_i without the
                // runtime ionization level. This is required for evolving
                // high-Z charge states. Fixed-charge legacy species retain
                // n_s/n_e=f_s/Z_s exactly.
                amrex::Real const ion_density = material_index >= 0
                    ? material_unit_charge_density[material_index](i,j,k)
                        / PhysConst::q_e
                    : f_s * electron_density / Z_s;
                if (!(ion_density > 0.0_rt)) { return; }

                amrex::Real const Ti_eV = ablastr::coarsen::sample::Interp(
                    Ti_arr, cc_stag, nodal, coarsen, i, j, k, 0);
                qei_ion_temperature_arr(i,j,k) = Ti_eV;
                amrex::Real const Te_K  = Te_arr(i,j,k);
                amrex::Real const Te_eV = Te_K / K_per_eV;
                amrex::Real const Ti_K  = Ti_eV * K_per_eV;
                qei_electron_temperature_before_arr(i, j, k) = Te_eV;

                amrex::Real const nu = nu_ei(rho_val, amrex::max(Te_eV, Te_floor_eV), Ti_eV, t_new);
                auto const material_mass_density = thermodynamics
                    .materialMassDensitiesFromChargeDensityArrays(
                        material_unit_charge_density, i, j, k);
                ElectronThermodynamicState const old_state = thermodynamics
                    .stateFromMaterialMassDensitiesTemperature(
                        rho_val, material_mass_density, Te_K);
                amrex::Real const ion_heat_capacity_density =
                    1.5_rt * ion_density * PhysConst::kb;
                amrex::Real const conductance_density =
                    3.0_rt * ion_density * PhysConst::kb * nu;
                TwoTemperatureExchangeState const exchange =
                    exactTwoTemperatureExchange(
                        Te_K, Ti_K, old_state.heat_capacity_density,
                        ion_heat_capacity_density, conductance_density, dt);
                if (!exchange.valid) {
                    Te_arr(i,j,k) =
                        std::numeric_limits<amrex::Real>::quiet_NaN();
                    return;
                }
                ElectronEnergySourceUpdate const source_update =
                    thermodynamics.applyEnergyDensityIncrement(
                        rho_val, material_mass_density, Te_K,
                        exchange.electron_energy_change_density);
                if (!source_update.valid) {
                    Te_arr(i,j,k) =
                        std::numeric_limits<amrex::Real>::quiet_NaN();
                    return;
                }
                Te_arr(i,j,k) = source_update.temperature;
                qei_step_arr(i,j,k) +=
                    source_update.energy_change_density;
                qei_cumulative_arr(i,j,k) +=
                    source_update.energy_change_density;
                qei_species_step_arr(i, j, k) =
                    source_update.energy_change_density;
            });
        }
        qei_species_step.FillBoundary(qei_species_step.nGrowVect(), period);
        qei_electron_temperature_before.FillBoundary(
            qei_electron_temperature_before.nGrowVect(), period);
    }

    Te.FillBoundary(Te.nGrowVect(), period);
    for (auto const& spec_name : species_names) {
        auto& pc = mypc.GetParticleContainerFromName(spec_name);
        if (pc.getCharge() == 0.0_prt) { continue; }
        auto& qei_ion_temperature = *warpx.m_fields.get(
            "hybrid_qei_ion_temperature_fp_" + spec_name, lev);
        qei_ion_temperature.FillBoundary(
            qei_ion_temperature.nGrowVect(), period);
    }
}


void HybridPICModel::QDSMCApplyIonHeating (int const lev, amrex::Real const dt,
                                           amrex::MultiFab const * const redirect_E,
                                           std::map<std::string, amrex::MultiFab*> const * const Ti_dep_by_species) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCApplyIonHeating()");

    using warpx::fields::FieldType;

    // Ornstein-Uhlenbeck proposal for the kinetic-ion response:
    //   v_p <- u_e + (v_p - u_e) exp(-nu_ei dt) + sig R,
    //   R ~ N(0,1) per component.
    // Q_ei (when do_relax) sets the drag toward the electron fluid u_e and the
    // thermal diffusion sig^2 = k_B T_e/m_i (1 - exp(-2 nu_ei dt)). The
    // realized proposal is then projected cell-locally to preserve the pre-step
    // ion momentum and to make its nonrelativistic thermal-energy change
    // exactly equal the negative electron ledger. The Te-threshold redirect
    // (when do_redir) adds pure-diffusion heating E_s/m_i, with the per-species
    // redirected energy E_s [J] read from redirect_E. Both channels are
    // per-species correct (own mass, own T_i, own redirect_E comp).
    auto & warpx = WarpX::GetInstance();

    bool const do_relax = (Ti_dep_by_species != nullptr);
    bool const do_redir = (redirect_E != nullptr);
    if (!do_relax && !do_redir) { return; }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(do_relax && do_redir),
        "Qei relaxation and stochastic Joule-to-ion redirection must be "
        "applied in separate ion-heating calls so the Qei moment projection "
        "has one unambiguous energy ledger.");

    amrex::MultiFab const & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp, lev);
    ablastr::fields::VectorField Ve =
        warpx.m_fields.get_alldirs(FieldType::hybrid_electron_velocity_fp, lev);

    auto const rho_floor = PhysConst::q_e * m_n_floor;
    auto const nu_ei     = m_nu_ei;
    auto const t_new     = warpx.gett_new(0);
    auto const K_per_eV  = PhysConst::q_e / PhysConst::kb;   // T[eV]*this = T[K]
    // Floor on T_e in the nu_ei rate argument so pow(Te,-1.5) stays finite.
    amrex::Real const Te_floor_eV = 1.e-3_rt;

    // Nodal->cc interpolation staggers (unused dims set cc-like).
    amrex::GpuArray<int, 3> nodal_src = {1, 1, 1};
    for (int d = AMREX_SPACEDIM; d < 3; ++d) { nodal_src[d] = 0; }
    amrex::GpuArray<int, 3> const cc_dst  = {0, 0, 0};
    amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};

    amrex::BoxArray const cc_ba = amrex::convert(Te.boxArray(), amrex::IntVect::TheCellVector());

    auto & mypc = warpx.GetPartContainer();
    auto const species_names = mypc.GetSpeciesNames();

    // Charged-species component index for redirect_E (matches QDSMCAddJouleHeating:
    // incremented for every charged species before the mass check).
    int ion_comp = -1;
    for (auto const & spec_name : species_names) {
        auto & pc = mypc.GetParticleContainerFromName(spec_name);
        if (pc.getCharge() == 0._prt) { continue; }
        ++ion_comp;
        auto const m_i = pc.getMass();
        if (m_i <= 0._prt) { continue; }
        auto const mass_real = static_cast<amrex::Real>(m_i);

        // Ion temperature [eV] (NGP) -- only needed as the nu_ei parser argument
        // (Q_ei drag/diffusion). Skipped when only the redirect is active. When
        // relaxation is on, T_i was deposited once by the caller and is shared via
        // Ti_dep_by_species (QDSMCAddTemperatureRelaxation ran just before with no
        // intervening ion motion).
        amrex::MultiFab Ti_cc(cc_ba, Te.DistributionMap(), 1, 0);
        Ti_cc.setVal(0.0_rt);
        if (do_relax) {
            amrex::MultiFab const & Ti_dep = *(Ti_dep_by_species->at(spec_name));
            Ti_cc.ParallelCopy(Ti_dep, 0, 0, 1, amrex::IntVect::TheZeroVector(),
                               amrex::IntVect::TheZeroVector());
        }

        // Per-cell drag-diffusion coefficients on the cc field grid:
        //   0 = nu_ei [1/s], 1-3 = u_e [m/s], 4 = T_e [K], 5 = redirected dTe [K].
        // Defaults (0) leave inactive / below-floor cells as no-ops.
        amrex::MultiFab coef(cc_ba, Te.DistributionMap(), 6, 0);
        coef.setVal(0.0_rt);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(coef, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Array4<amrex::Real>       const & coef_arr = coef.array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr  = rho.const_array(mfi);
            amrex::Array4<amrex::Real const> const & Te_arr   = Te.const_array(mfi);
            amrex::Array4<amrex::Real const> const & Ti_arr   = Ti_cc.const_array(mfi);
            amrex::Array4<amrex::Real const> const & Vex_arr  = Ve[0]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Vey_arr  = Ve[1]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Vez_arr  = Ve[2]->const_array(mfi);
            amrex::Array4<amrex::Real const> redirect_arr;
            if (do_redir) { redirect_arr = redirect_E->const_array(mfi); }

            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real const rho_val = ablastr::coarsen::sample::Interp(
                    rho_arr, nodal_src, cc_dst, coarsen, i, j, k, 0);
                if (rho_val <= rho_floor) { return; }

                amrex::Real const Te_K = ablastr::coarsen::sample::Interp(
                    Te_arr, nodal_src, cc_dst, coarsen, i, j, k, 0);
                coef_arr(i,j,k,4) = Te_K;

                if (do_relax) {
                    amrex::Real const Ti_eV = Ti_arr(i,j,k);
                    coef_arr(i,j,k,0) = nu_ei(rho_val, amrex::max(Te_K / K_per_eV, Te_floor_eV), Ti_eV, t_new);
                    coef_arr(i,j,k,1) = ablastr::coarsen::sample::Interp(
                        Vex_arr, nodal_src, cc_dst, coarsen, i, j, k, 0);
                    coef_arr(i,j,k,2) = ablastr::coarsen::sample::Interp(
                        Vey_arr, nodal_src, cc_dst, coarsen, i, j, k, 0);
                    coef_arr(i,j,k,3) = ablastr::coarsen::sample::Interp(
                        Vez_arr, nodal_src, cc_dst, coarsen, i, j, k, 0);
                }
                if (do_redir) {
                    // E_s for this species = redirect_E component ion_comp.
                    coef_arr(i,j,k,5) = ablastr::coarsen::sample::Interp(
                        redirect_arr, nodal_src, cc_dst, coarsen, i, j, k, ion_comp);
                }
            });
        }

        // Stage the coefficients on the particle grid for NGP lookup.
        auto const & pba = pc.ParticleBoxArray(lev);
        auto const & pdm = pc.ParticleDistributionMap(lev);
        amrex::MultiFab coef_p(pba, pdm, 6, 0);
        coef_p.setVal(0.0_rt);
        coef_p.ParallelCopy(coef, 0, 0, 6);

        // Components 0:4 are the pre-proposal weighted moments
        // (sum w, sum w*u[3], sum w*|u|^2); 5:9 are the corresponding
        // moments after the OU proposal. NGP bins are the same bins used by
        // the coefficient lookup and by the local correction below.
        std::unique_ptr<amrex::MultiFab> ion_moments;
        if (do_relax) {
            ion_moments = std::make_unique<amrex::MultiFab>(pba, pdm, 10, 1);
            ion_moments->setVal(0.0_rt);
        }

        // Apply the drag-diffusion update to each ion (NGP cell lookup).
        auto const plo = warpx.Geom(lev).ProbLoArray();
        auto const dxi = warpx.Geom(lev).InvCellSizeArray();
        auto const kb  = PhysConst::kb;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (WarpXParIter pti(pc, lev); pti.isValid(); ++pti)
        {
            long const np = pti.numParticles();
            auto & tile = pti.GetParticleTile();
            auto ptd = tile.getParticleTileData();
            amrex::ParticleReal const* AMREX_RESTRICT wp =
                pti.GetAttribs(PIdx::w).dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT uxp =
                pti.GetAttribs(PIdx::ux).dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT uyp =
                pti.GetAttribs(PIdx::uy).dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT uzp =
                pti.GetAttribs(PIdx::uz).dataPtr();
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) ||                   \
    defined(WARPX_DIM_RSPHERE)
            amrex::ParticleReal const* AMREX_RESTRICT thetap =
                pti.GetAttribs(PIdx::theta).dataPtr();
#endif
#if defined(WARPX_DIM_RSPHERE)
            amrex::ParticleReal const* AMREX_RESTRICT phip =
                pti.GetAttribs(PIdx::phi).dataPtr();
#endif

            amrex::Array4<amrex::Real const> const& coef_arr =
                coef_p.const_array(pti);
            amrex::Array4<amrex::Real> moment_arr;
            if (do_relax) {
                moment_arr = ion_moments->array(pti);
            }

            amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE(
                                          long ip,
                                          amrex::RandomEngine const& engine) {
                auto const p = WarpXParticleContainer::ParticleType(ptd, ip);
                const auto [ii, jj, kk] =
                    amrex::getParticleCell(p, plo, dxi).dim3();
                amrex::ParticleReal const nu = coef_arr(ii, jj, kk, 0);
                amrex::ParticleReal const Te_K = coef_arr(ii, jj, kk, 4);
                amrex::ParticleReal const E_s = coef_arr(ii, jj, kk, 5);
                // Ion moments are stored in amrex::Real even when particle
                // attributes use reduced precision.  Promote the weight
                // before forming every atomic contribution so that the
                // atomic value exactly matches the destination type.
                auto const weight = static_cast<amrex::Real>(wp[ip]);
                amrex::ParticleReal const ux_old = uxp[ip];
                amrex::ParticleReal const uy_old = uyp[ip];
                amrex::ParticleReal const uz_old = uzp[ip];
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                amrex::ParticleReal const theta = thetap[ip];
                amrex::ParticleReal const costheta = std::cos(theta);
                amrex::ParticleReal const sintheta = std::sin(theta);
                amrex::ParticleReal const u0_old =
                    ux_old * costheta + uy_old * sintheta;
                amrex::ParticleReal const u1_old =
                    -ux_old * sintheta + uy_old * costheta;
                amrex::ParticleReal const u2_old = uz_old;
#elif defined(WARPX_DIM_RSPHERE)
                amrex::ParticleReal const theta = thetap[ip];
                amrex::ParticleReal const phi = phip[ip];
                amrex::ParticleReal const costheta = std::cos(theta);
                amrex::ParticleReal const sintheta = std::sin(theta);
                amrex::ParticleReal const cosphi = std::cos(phi);
                amrex::ParticleReal const sinphi = std::sin(phi);
                amrex::ParticleReal const u0_old =
                    ux_old * costheta * cosphi
                    + uy_old * sintheta * cosphi + uz_old * sinphi;
                amrex::ParticleReal const u1_old =
                    -ux_old * sintheta + uy_old * costheta;
                amrex::ParticleReal const u2_old =
                    -ux_old * costheta * sinphi
                    - uy_old * sintheta * sinphi + uz_old * cosphi;
#else
                amrex::ParticleReal const u0_old = ux_old;
                amrex::ParticleReal const u1_old = uy_old;
                amrex::ParticleReal const u2_old = uz_old;
#endif
                if (do_relax) {
                    auto const u0_old_real =
                        static_cast<amrex::Real>(u0_old);
                    auto const u1_old_real =
                        static_cast<amrex::Real>(u1_old);
                    auto const u2_old_real =
                        static_cast<amrex::Real>(u2_old);
                    amrex::Gpu::Atomic::Add(&moment_arr(ii, jj, kk, 0), weight);
                    amrex::Gpu::Atomic::Add(&moment_arr(ii, jj, kk, 1),
                                            weight * u0_old_real);
                    amrex::Gpu::Atomic::Add(&moment_arr(ii, jj, kk, 2),
                                            weight * u1_old_real);
                    amrex::Gpu::Atomic::Add(&moment_arr(ii, jj, kk, 3),
                                            weight * u2_old_real);
                    amrex::Gpu::Atomic::Add(&moment_arr(ii, jj, kk, 4),
                                            weight * (u0_old_real * u0_old_real +
                                                      u1_old_real * u1_old_real +
                                                      u2_old_real * u2_old_real));
                }

                // Ornstein-Uhlenbeck drag and variance (Q_ei diffusion +
                // redirect E_s).
                amrex::ParticleReal const nu_dt = nu * dt;
                amrex::ParticleReal const drag = -std::expm1(-nu_dt);            // 1 - exp(-nu dt)
                amrex::ParticleReal const sig2 =
                    (-kb * Te_K * std::expm1(-2._prt * nu_dt) + E_s) / m_i;
                if (drag > 0._prt || sig2 > 0._prt) {
                    amrex::ParticleReal const ue0 = coef_arr(ii, jj, kk, 1);
                    amrex::ParticleReal const ue1 = coef_arr(ii, jj, kk, 2);
                    amrex::ParticleReal const ue2 = coef_arr(ii, jj, kk, 3);
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                    amrex::ParticleReal const uex =
                        ue0 * costheta - ue1 * sintheta;
                    amrex::ParticleReal const uey =
                        ue0 * sintheta + ue1 * costheta;
                    amrex::ParticleReal const uez = ue2;
#elif defined(WARPX_DIM_RSPHERE)
                    amrex::ParticleReal const uex =
                        ue0 * costheta * cosphi - ue1 * sintheta
                        - ue2 * costheta * sinphi;
                    amrex::ParticleReal const uey =
                        ue0 * sintheta * cosphi + ue1 * costheta
                        - ue2 * sintheta * sinphi;
                    amrex::ParticleReal const uez =
                        ue0 * sinphi + ue2 * cosphi;
#else
                    amrex::ParticleReal const uex = ue0;
                    amrex::ParticleReal const uey = ue1;
                    amrex::ParticleReal const uez = ue2;
#endif
                    amrex::ParticleReal const sig =
                        std::sqrt(amrex::max(0._prt, sig2));
                    uxp[ip] +=
                        -drag * (uxp[ip] - uex) +
                        sig * amrex::RandomNormal(0._prt, 1._prt, engine);
                    uyp[ip] +=
                        -drag * (uyp[ip] - uey) +
                        sig * amrex::RandomNormal(0._prt, 1._prt, engine);
                    uzp[ip] +=
                        -drag * (uzp[ip] - uez) +
                        sig * amrex::RandomNormal(0._prt, 1._prt, engine);
                }
                if (do_relax) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                    amrex::ParticleReal const u0 =
                        uxp[ip] * costheta + uyp[ip] * sintheta;
                    amrex::ParticleReal const u1 =
                        -uxp[ip] * sintheta + uyp[ip] * costheta;
                    amrex::ParticleReal const u2 = uzp[ip];
#elif defined(WARPX_DIM_RSPHERE)
                    amrex::ParticleReal const u0 =
                        uxp[ip] * costheta * cosphi
                        + uyp[ip] * sintheta * cosphi + uzp[ip] * sinphi;
                    amrex::ParticleReal const u1 =
                        -uxp[ip] * sintheta + uyp[ip] * costheta;
                    amrex::ParticleReal const u2 =
                        -uxp[ip] * costheta * sinphi
                        - uyp[ip] * sintheta * sinphi + uzp[ip] * cosphi;
#else
                    amrex::ParticleReal const u0 = uxp[ip];
                    amrex::ParticleReal const u1 = uyp[ip];
                    amrex::ParticleReal const u2 = uzp[ip];
#endif
                    auto const u0_real = static_cast<amrex::Real>(u0);
                    auto const u1_real = static_cast<amrex::Real>(u1);
                    auto const u2_real = static_cast<amrex::Real>(u2);
                    amrex::Gpu::Atomic::Add(&moment_arr(ii, jj, kk, 5), weight);
                    amrex::Gpu::Atomic::Add(&moment_arr(ii, jj, kk, 6),
                                            weight * u0_real);
                    amrex::Gpu::Atomic::Add(&moment_arr(ii, jj, kk, 7),
                                            weight * u1_real);
                    amrex::Gpu::Atomic::Add(&moment_arr(ii, jj, kk, 8),
                                            weight * u2_real);
                    amrex::Gpu::Atomic::Add(&moment_arr(ii, jj, kk, 9),
                                            weight *
                                                (u0_real * u0_real +
                                                 u1_real * u1_real +
                                                 u2_real * u2_real));
                }
            });
        }

        if (do_relax) {
            ion_moments->FillBoundary(warpx.Geom(lev).periodicity());
            amrex::MultiFab const& electron_source = *warpx.m_fields.get(
                "hybrid_qei_electron_energy_fp_" + spec_name, lev);
            amrex::BoxArray const source_pba =
                amrex::convert(pba, electron_source.ixType().toIntVect());
            amrex::MultiFab source_on_particles(source_pba, pdm, 1, 0);
            source_on_particles.setVal(0.0_rt);
            source_on_particles.ParallelCopy(electron_source, 0, 0, 1,
                                             amrex::IntVect::TheZeroVector(),
                                             amrex::IntVect::TheZeroVector());

            // Components: thermal scale; old mean u[3]; proposal mean u[3].
            // Applying old_mean + scale*(proposal-proposal_mean) preserves
            // the old cell momentum while setting the requested thermal
            // energy independently of Monte-Carlo sampling noise.
            amrex::MultiFab correction(pba, pdm, 7, 0);
            correction.setVal(0.0_rt);
            amrex::MultiFab ion_request_density(pba, pdm, 1, 0);
            ion_request_density.setVal(0.0_rt);
            amrex::Gpu::DeviceScalar<int> invalid_cells(0);
            int* const invalid_cells_ptr = invalid_cells.dataPtr();
            auto const problo = warpx.Geom(lev).ProbLoArray();
            auto const dx = warpx.Geom(lev).CellSizeArray();
            auto const domain_lo = amrex::lbound(warpx.Geom(lev).Domain());
            auto const domain_hi = amrex::ubound(warpx.Geom(lev).Domain());
            amrex::GpuArray<int, 3> periodic{0, 0, 0};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                periodic[d] = warpx.Geom(lev).isPeriodic(d) ? 1 : 0;
            }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(correction, TilingIfNotGPU()); mfi.isValid();
                 ++mfi) {
                amrex::Array4<amrex::Real> const corr = correction.array(mfi);
                amrex::Array4<amrex::Real> const request =
                    ion_request_density.array(mfi);
                amrex::Array4<amrex::Real const> const moments =
                    ion_moments->const_array(mfi);
                amrex::Array4<amrex::Real const> const source =
                    source_on_particles.const_array(mfi);
                amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(
                                                      int i, int j,
                                                      int k) noexcept {
                    // Partition each nodal dual-volume source only among
                    // its adjacent particle cells.  The weight is local
                    // ion number density times physical corner volume.
                    // Thus a material/vacuum interface assigns the whole
                    // interface-node request to its supported side, while
                    // the sum over supported neighbors remains the exact
                    // nodal energy.  No distant-cell redistribution is
                    // permitted.
                    amrex::Real electron_change = 0.0_rt;
                    amrex::Real const cell_weight =
                        amrex::max(moments(i, j, k, 0), 0.0_rt);
                    amrex::Real const cell_volume =
                        hybrid_cell_volume(i, j, k, problo, dx, domain_lo);
                    amrex::Real const cell_density =
                        cell_volume > 0.0_rt ? cell_weight / cell_volume
                                             : 0.0_rt;
                    for (int dk = 0; dk < (AMREX_SPACEDIM == 3 ? 2 : 1); ++dk) {
                        for (int dj = 0; dj < (AMREX_SPACEDIM >= 2 ? 2 : 1);
                             ++dj) {
                            for (int di = 0; di < 2; ++di) {
                                int const ni = i + di;
                                int const nj = j + dj;
                                int const nk = k + dk;
                                amrex::Real support = 0.0_rt;
                                amrex::Real physical_node_volume = 0.0_rt;
                                for (int neighbor_dk = 0;
                                     neighbor_dk <
                                     (AMREX_SPACEDIM == 3 ? 2 : 1);
                                     ++neighbor_dk) {
                                    for (int neighbor_dj = 0;
                                         neighbor_dj <
                                         (AMREX_SPACEDIM >= 2 ? 2 : 1);
                                         ++neighbor_dj) {
                                        for (int neighbor_di = 0;
                                             neighbor_di < 2; ++neighbor_di) {
                                            int const ci = ni - neighbor_di;
                                            int const cj = nj - neighbor_dj;
                                            int const ck = nk - neighbor_dk;
                                            if (!hybrid_source_cell_is_addressable(
                                                    ci, cj, ck, domain_lo,
                                                    domain_hi, periodic)) {
                                                continue;
                                            }
                                            amrex::Real const corner_volume =
                                                hybrid_cell_corner_volume(
                                                    ci, cj, ck, neighbor_di,
                                                    neighbor_dj, neighbor_dk,
                                                    problo, dx, domain_lo);
                                            amrex::Real const neighbor_volume =
                                                hybrid_cell_volume(ci, cj, ck,
                                                                   problo, dx,
                                                                   domain_lo);
                                            amrex::Real const neighbor_density =
                                                neighbor_volume > 0.0_rt
                                                    ? amrex::max(moments(ci, cj,
                                                                         ck, 0),
                                                                 0.0_rt) /
                                                          neighbor_volume
                                                    : 0.0_rt;
                                            physical_node_volume +=
                                                corner_volume;
                                            support += neighbor_density *
                                                       corner_volume;
                                        }
                                    }
                                }
                                amrex::Real const source_density =
                                    source(ni, nj, nk);
                                if (!(support > 0.0_rt)) {
                                    if (source_density != 0.0_rt) {
                                        amrex::Gpu::Atomic::Add(
                                            invalid_cells_ptr, 1);
                                    }
                                    continue;
                                }
                                amrex::Real const own_corner_volume =
                                    hybrid_cell_corner_volume(i, j, k, di, dj,
                                                              dk, problo, dx,
                                                              domain_lo);
                                electron_change +=
                                    source_density * physical_node_volume *
                                    cell_density * own_corner_volume / support;
                            }
                        }
                    }
                    amrex::Real const requested_ion_change = -electron_change;
                    request(i, j, k) = cell_volume > 0.0_rt
                                           ? requested_ion_change / cell_volume
                                           : 0.0_rt;
                    amrex::Real const weight = moments(i, j, k, 0);
                    amrex::Real const proposal_weight = moments(i, j, k, 5);
                    amrex::Real const request_scale =
                        amrex::max(std::abs(requested_ion_change),
                                   std::numeric_limits<amrex::Real>::min());
                    if (!(weight > 0.0_rt) || !(proposal_weight > 0.0_rt)) {
                        if (std::abs(requested_ion_change) >
                            128.0_rt *
                                std::numeric_limits<amrex::Real>::epsilon() *
                                request_scale) {
                            amrex::Gpu::Atomic::Add(invalid_cells_ptr, 1);
                        }
                        corr(i, j, k, 0) = 1.0_rt;
                        return;
                    }

                    amrex::Real const old_ux = moments(i, j, k, 1) / weight;
                    amrex::Real const old_uy = moments(i, j, k, 2) / weight;
                    amrex::Real const old_uz = moments(i, j, k, 3) / weight;
                    amrex::Real const proposal_ux =
                        moments(i, j, k, 6) / proposal_weight;
                    amrex::Real const proposal_uy =
                        moments(i, j, k, 7) / proposal_weight;
                    amrex::Real const proposal_uz =
                        moments(i, j, k, 8) / proposal_weight;
                    corr(i, j, k, 1) = old_ux;
                    corr(i, j, k, 2) = old_uy;
                    corr(i, j, k, 3) = old_uz;
                    corr(i, j, k, 4) = proposal_ux;
                    corr(i, j, k, 5) = proposal_uy;
                    corr(i, j, k, 6) = proposal_uz;

                    amrex::Real const old_variance = amrex::max(
                        moments(i, j, k, 4) -
                            weight * (old_ux * old_ux + old_uy * old_uy +
                                      old_uz * old_uz),
                        0.0_rt);
                    amrex::Real const proposal_variance = amrex::max(
                        moments(i, j, k, 9) -
                            proposal_weight * (proposal_ux * proposal_ux +
                                               proposal_uy * proposal_uy +
                                               proposal_uz * proposal_uz),
                        0.0_rt);
                    amrex::Real const old_thermal =
                        0.5_rt * mass_real * old_variance;
                    amrex::Real const proposal_thermal =
                        0.5_rt * mass_real * proposal_variance;
                    amrex::Real const target_thermal =
                        old_thermal + requested_ion_change;
                    amrex::Real const energy_scale =
                        amrex::max(amrex::max(std::abs(old_thermal),
                                              std::abs(requested_ion_change)),
                                   std::numeric_limits<amrex::Real>::min());
                    amrex::Real const tolerance =
                        512.0_rt * std::numeric_limits<amrex::Real>::epsilon() *
                        energy_scale;
                    if (target_thermal < -tolerance ||
                        (proposal_thermal <= tolerance &&
                         target_thermal > tolerance)) {
                        amrex::Gpu::Atomic::Add(invalid_cells_ptr, 1);
                        corr(i, j, k, 0) = 1.0_rt;
                        return;
                    }
                    corr(i, j, k, 0) =
                        proposal_thermal > tolerance
                            ? std::sqrt(amrex::max(target_thermal, 0.0_rt) /
                                        proposal_thermal)
                            : 0.0_rt;
                });
            }
            amrex::Gpu::streamSynchronize();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                invalid_cells.dataValue() == 0,
                "The conservative Qei ion update could not realize one or "
                "more local energy requests while preserving cell momentum. "
                "Every exchanging particle cell needs resolved thermal "
                "variance (normally at least two non-identical ion "
                "macroparticles), and an ion-cooling request cannot exceed "
                "that cell's thermal energy. Increase particles per cell or "
                "reduce the Qei source timestep/rate.");

            auto& ion_request_field = *warpx.m_fields.get(
                "hybrid_qei_ion_energy_cc_" + spec_name, lev);
            ion_request_field.setVal(0.0_rt);
            ion_request_field.ParallelCopy(ion_request_density, 0, 0, 1,
                                           amrex::IntVect::TheZeroVector(),
                                           amrex::IntVect::TheZeroVector());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (WarpXParIter pti(pc, lev); pti.isValid(); ++pti) {
                long const np = pti.numParticles();
                auto const ptd = pti.GetParticleTile().getParticleTileData();
                amrex::ParticleReal* AMREX_RESTRICT uxp =
                    pti.GetAttribs(PIdx::ux).dataPtr();
                amrex::ParticleReal* AMREX_RESTRICT uyp =
                    pti.GetAttribs(PIdx::uy).dataPtr();
                amrex::ParticleReal* AMREX_RESTRICT uzp =
                    pti.GetAttribs(PIdx::uz).dataPtr();
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) ||                   \
    defined(WARPX_DIM_RSPHERE)
                amrex::ParticleReal const* AMREX_RESTRICT thetap =
                    pti.GetAttribs(PIdx::theta).dataPtr();
#endif
#if defined(WARPX_DIM_RSPHERE)
                amrex::ParticleReal const* AMREX_RESTRICT phip =
                    pti.GetAttribs(PIdx::phi).dataPtr();
#endif
                amrex::Array4<amrex::Real const> const corr =
                    correction.const_array(pti);
                amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long ip) noexcept {
                    auto const p =
                        WarpXParticleContainer::ParticleType(ptd, ip);
                    const auto [ii, jj, kk] =
                        amrex::getParticleCell(p, plo, dxi).dim3();
                    amrex::ParticleReal const scale = corr(ii, jj, kk, 0);
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                    amrex::ParticleReal const theta = thetap[ip];
                    amrex::ParticleReal const costheta = std::cos(theta);
                    amrex::ParticleReal const sintheta = std::sin(theta);
                    amrex::ParticleReal const proposal_u0 =
                        uxp[ip] * costheta + uyp[ip] * sintheta;
                    amrex::ParticleReal const proposal_u1 =
                        -uxp[ip] * sintheta + uyp[ip] * costheta;
                    amrex::ParticleReal const proposal_u2 = uzp[ip];
#elif defined(WARPX_DIM_RSPHERE)
                        amrex::ParticleReal const theta = thetap[ip];
                        amrex::ParticleReal const phi = phip[ip];
                        amrex::ParticleReal const costheta = std::cos(theta);
                        amrex::ParticleReal const sintheta = std::sin(theta);
                        amrex::ParticleReal const cosphi = std::cos(phi);
                        amrex::ParticleReal const sinphi = std::sin(phi);
                        amrex::ParticleReal const proposal_u0 =
                            uxp[ip] * costheta * cosphi
                            + uyp[ip] * sintheta * cosphi
                            + uzp[ip] * sinphi;
                        amrex::ParticleReal const proposal_u1 =
                            -uxp[ip] * sintheta + uyp[ip] * costheta;
                        amrex::ParticleReal const proposal_u2 =
                            -uxp[ip] * costheta * sinphi
                            - uyp[ip] * sintheta * sinphi
                            + uzp[ip] * cosphi;
#else
                        amrex::ParticleReal const proposal_u0 = uxp[ip];
                        amrex::ParticleReal const proposal_u1 = uyp[ip];
                        amrex::ParticleReal const proposal_u2 = uzp[ip];
#endif
                    amrex::ParticleReal const corrected_u0 =
                        corr(ii, jj, kk, 1) +
                        scale * (proposal_u0 - corr(ii, jj, kk, 4));
                    amrex::ParticleReal const corrected_u1 =
                        corr(ii, jj, kk, 2) +
                        scale * (proposal_u1 - corr(ii, jj, kk, 5));
                    amrex::ParticleReal const corrected_u2 =
                        corr(ii, jj, kk, 3) +
                        scale * (proposal_u2 - corr(ii, jj, kk, 6));
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                    uxp[ip] = corrected_u0 * costheta - corrected_u1 * sintheta;
                    uyp[ip] = corrected_u0 * sintheta + corrected_u1 * costheta;
                    uzp[ip] = corrected_u2;
#elif defined(WARPX_DIM_RSPHERE)
                        uxp[ip] = corrected_u0 * costheta * cosphi
                            - corrected_u1 * sintheta
                            - corrected_u2 * costheta * sinphi;
                        uyp[ip] = corrected_u0 * sintheta * cosphi
                            + corrected_u1 * costheta
                            - corrected_u2 * sintheta * sinphi;
                        uzp[ip] = corrected_u0 * sinphi
                            + corrected_u2 * cosphi;
#else
                        uxp[ip] = corrected_u0;
                        uyp[ip] = corrected_u1;
                        uzp[ip] = corrected_u2;
#endif
                });
            }
        }
    }
}


void HybridPICModel::QDSMCFillElectronPressureFromTe (int const lev) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCFillElectronPressureFromTe()");

    auto & warpx = WarpX::GetInstance();

    amrex::MultiFab       & Pe  = *warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    amrex::MultiFab const & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp, lev);

    auto const rho_floor = PhysConst::q_e * m_n_floor;
    auto const thermodynamics = m_electron_thermodynamics.executor();
    int const num_materials = m_electron_thermodynamics.numMaterials();
    bool const use_raw_material_support = m_fv_transport_internal_energy;
    amrex::GpuArray<amrex::MultiFab const*,
                    ElectronThermodynamicsExecutor::max_materials>
        material_charge_density_fields{};
    for (int material = 0; material < num_materials; ++material) {
        auto& material_field = *warpx.m_fields.get(
            "ni_charge_fp_"
            + m_electron_thermodynamics.materialSpeciesName(material), lev);
        material_field.FillBoundary(
            material_field.nGrowVect(), warpx.Geom(lev).periodicity());
        material_charge_density_fields[material] = &material_field;
    }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Pe, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real>       const & Pe_arr  = Pe.array(mfi);
        amrex::Array4<amrex::Real const> const & Te_arr  = Te.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);
        ElectronThermodynamicsExecutor::MaterialChargeDensityArrays
            material_charge_density{};
        for (int material = 0; material < num_materials; ++material) {
            material_charge_density[material] =
                material_charge_density_fields[material]->const_array(mfi);
        }

        amrex::Box const & tbox = mfi.tilebox();
        amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // The nonlinear FV path transports energy with the deposited
            // material support and therefore needs the homogeneous vacuum
            // limit P(rho=0)=0.  Keep rho_floor solely in the Ohm-law
            // denominator; using it here would invent pressure in exact
            // vacuum and make pressure work locally EOS-unrepresentable.
            // The legacy entropy-marker path retains its historical floor.
            amrex::Real const rho_val = use_raw_material_support
                ? amrex::max(rho_arr(i,j,k), 0.0_rt)
                : amrex::max(rho_arr(i,j,k), rho_floor);
            auto const material_mass_density = thermodynamics
                .materialMassDensitiesFromChargeDensityArrays(
                    material_charge_density, i, j, k);
            ElectronThermodynamicState const state = thermodynamics
                .stateFromMaterialMassDensitiesTemperature(
                    rho_val, material_mass_density, Te_arr(i,j,k));
            if (state.pressure < 0.0_rt
                || !amrex::Math::isfinite(state.pressure))
            {
                Pe_arr(i,j,k) =
                    std::numeric_limits<amrex::Real>::quiet_NaN();
                return;
            }
            Pe_arr(i,j,k) = state.pressure;
        });
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        Pe.is_finite(0, 1, 0),
        "Hybrid electron-pressure reconstruction encountered an EOS state "
        "outside its valid density or temperature range, or an unresolved "
        "registered mixed-material node.");
}


amrex::Real HybridPICModel::ApplyElectronEnergySource (
    int const lev,
    amrex::MultiFab& cell_integrated_energy,
    amrex::Real const minimum_electron_density,
    amrex::MultiFab const* const nonlinear_lte_remap) const
{
    ABLASTR_PROFILE("HybridPICModel::ApplyElectronEnergySource()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_solve_electron_energy_equation,
        "Hybrid electron energy sources require "
        "hybrid_pic_model.solve_electron_energy_equation=1.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        cell_integrated_energy.ixType().cellCentered(),
        "Hybrid electron energy sources must be cell centered.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        cell_integrated_energy.nComp() >= 1,
        "Hybrid electron energy sources require at least one component.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        cell_integrated_energy.nGrowVect().allGE(1),
        "Hybrid electron energy sources require one cell ghost in every "
        "active dimension for the cell-to-node remap.");
    bool const use_nonlinear_lte_remap = nonlinear_lte_remap != nullptr;
    if (use_nonlinear_lte_remap) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            nonlinear_lte_remap->ixType().cellCentered()
                && nonlinear_lte_remap->nComp()
                    == nonlinear_lte_remap_components
                && nonlinear_lte_remap->nGrowVect().allGE(1)
                && nonlinear_lte_remap->boxArray()
                    == cell_integrated_energy.boxArray()
                && nonlinear_lte_remap->DistributionMap()
                    == cell_integrated_energy.DistributionMap(),
            "Nonlinear hybrid LTE remap and material-energy source must "
            "share one two-component cell-centered layout, distribution "
            "map, and ghost extent.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            nonlinear_lte_remap->is_finite(
                0, nonlinear_lte_remap_components,
                nonlinear_lte_remap->nGrowVect()),
            "Nonlinear hybrid LTE remap contains a non-finite temperature "
            "or represented-energy increment.");
    }

    auto& warpx = WarpX::GetInstance();
    amrex::MultiFab& Te =
        *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab& rho = *warpx.m_fields.get(FieldType::rho_fp, lev);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        amrex::convert(cell_integrated_energy.boxArray(), Te.ixType()) == Te.boxArray(),
        "Hybrid electron energy source and electron-temperature layouts must match.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        cell_integrated_energy.DistributionMap() == Te.DistributionMap(),
        "Hybrid electron energy source and electron-temperature distribution "
        "maps must match.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        rho.boxArray() == Te.boxArray()
            && rho.DistributionMap() == Te.DistributionMap()
            && rho.ixType() == Te.ixType(),
        "Hybrid charge density and electron temperature must share one nodal "
        "layout and distribution map.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        rho.nGrowVect().allGE(1) && Te.nGrowVect().allGE(1),
        "Hybrid charge density and electron temperature require one nodal "
        "ghost in every active dimension for radiation energy remapping.");

    auto const& geom = warpx.Geom(lev);
    rho.FillBoundary(geom.periodicity());
    Te.FillBoundary(Te.nGrowVect(), geom.periodicity());
    int const num_materials = m_electron_thermodynamics.numMaterials();
    amrex::GpuArray<amrex::MultiFab const*,
                    ElectronThermodynamicsExecutor::max_materials>
        material_charge_density_fields{};
    for (int material = 0; material < num_materials; ++material) {
        auto& material_field = *warpx.m_fields.get(
            "ni_charge_fp_"
            + m_electron_thermodynamics.materialSpeciesName(material), lev);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            material_field.boxArray() == Te.boxArray()
                && material_field.DistributionMap() == Te.DistributionMap()
                && material_field.ixType() == Te.ixType()
                && material_field.nGrowVect().allGE(Te.nGrowVect()),
            "Tabulated electron-EOS material densities must share the hybrid "
            "temperature layout and ghost extent.");
        material_field.FillBoundary(
            material_field.nGrowVect(), geom.periodicity());
        material_charge_density_fields[material] = &material_field;
    }
    auto const dx = geom.CellSizeArray();
    auto const problo = geom.ProbLoArray();
    auto const probhi = geom.ProbHiArray();
    auto const domain_lo = amrex::lbound(geom.Domain());
    auto const domain_hi = amrex::ubound(geom.Domain());
    amrex::GpuArray<int, 3> periodic{0, 0, 0};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        periodic[d] = geom.isPeriodic(d) ? 1 : 0;
    }

    amrex::Real const material_rho_threshold =
        PhysConst::q_e * minimum_electron_density;
    amrex::Real const thermodynamic_rho_floor = m_fv_transport_internal_energy
        ? 0.0_rt
        : PhysConst::q_e * m_n_floor;
    auto const thermodynamics = m_electron_thermodynamics.executor();
    // Freeze the old nodal caloric state before updating any temperature.
    // This avoids neighbor-read/write races and makes every source weight use
    // one consistent old-state heat capacity. After that conservative spatial
    // allocation, each node applies the exact configured U-to-T inverse.
    amrex::MultiFab node_thermodynamic_state(
        Te.boxArray(), Te.DistributionMap(), 3, Te.nGrowVect());
    node_thermodynamic_state.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(Te); mfi.isValid(); ++mfi)
    {
        amrex::Box const& box = mfi.growntilebox(Te.nGrowVect());
        amrex::Array4<amrex::Real> const state_arr =
            node_thermodynamic_state.array(mfi);
        amrex::Array4<amrex::Real const> const Te_arr = Te.const_array(mfi);
        amrex::Array4<amrex::Real const> const rho_arr = rho.const_array(mfi);
        ElectronThermodynamicsExecutor::MaterialChargeDensityArrays
            material_charge_density{};
        for (int material = 0; material < num_materials; ++material) {
            material_charge_density[material] =
                material_charge_density_fields[material]->const_array(mfi);
        }
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (
            int i, int j, int k) noexcept
        {
            amrex::Real const rho_node = rho_arr(i, j, k);
            if (rho_node <= material_rho_threshold) { return; }
            auto const material_mass_density = thermodynamics
                .materialMassDensitiesFromChargeDensityArrays(
                    material_charge_density, i, j, k);
            ElectronThermodynamicState const state = thermodynamics
                .stateFromMaterialMassDensitiesTemperature(
                    amrex::max(rho_node, thermodynamic_rho_floor),
                    material_mass_density, Te_arr(i, j, k));
            ElectronThermodynamicState const minimum_state = thermodynamics
                .stateFromMaterialMassDensitiesTemperature(
                    amrex::max(rho_node, thermodynamic_rho_floor),
                    material_mass_density,
                    thermodynamics.minimumTemperature());
            amrex::Real const energy_tolerance = 256.0_rt
                * std::numeric_limits<amrex::Real>::epsilon()
                * amrex::max(
                    amrex::max(
                        std::abs(state.internal_energy_density),
                        std::abs(minimum_state.internal_energy_density)),
                    std::numeric_limits<amrex::Real>::min());
            if (!amrex::Math::isfinite(state.internal_energy_density)
                || !(state.heat_capacity_density > 0.0_rt)
                || !amrex::Math::isfinite(state.heat_capacity_density)
                || !amrex::Math::isfinite(
                    minimum_state.internal_energy_density)
                || state.internal_energy_density
                    < minimum_state.internal_energy_density - energy_tolerance)
            {
                state_arr(i, j, k, 0) =
                    std::numeric_limits<amrex::Real>::quiet_NaN();
                return;
            }
            state_arr(i, j, k, 0) = state.internal_energy_density;
            state_arr(i, j, k, 1) = state.heat_capacity_density;
            state_arr(i, j, k, 2) = minimum_state.internal_energy_density;
        });
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        node_thermodynamic_state.is_finite(
            0, 3, node_thermodynamic_state.nGrowVect()),
        "Hybrid electron energy coupling encountered an invalid initial "
        "thermodynamic state.");

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(Te, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Box const& box = mfi.tilebox();
        amrex::Array4<amrex::Real> const Te_arr = Te.array(mfi);
        amrex::Array4<amrex::Real const> const rho_arr = rho.const_array(mfi);
        amrex::Array4<amrex::Real const> const source_arr =
            cell_integrated_energy.const_array(mfi);
        amrex::Array4<amrex::Real const> const state_arr =
            node_thermodynamic_state.const_array(mfi);
        amrex::Array4<amrex::Real const> nonlinear_remap_arr;
        if (use_nonlinear_lte_remap) {
            nonlinear_remap_arr = nonlinear_lte_remap->const_array(mfi);
        }
        ElectronThermodynamicsExecutor::MaterialChargeDensityArrays
            material_charge_density{};
        for (int material = 0; material < num_materials; ++material) {
            material_charge_density[material] =
                material_charge_density_fields[material]->const_array(mfi);
        }

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::Real const rho_node = rho_arr(i, j, k);
            if (rho_node <= material_rho_threshold) { return; }

            amrex::Real const node_volume = hybrid_node_volume(
                i, j, k, problo, probhi, dx, domain_lo, domain_hi, periodic);
            amrex::Real const thermodynamic_rho_node =
                amrex::max(rho_node, thermodynamic_rho_floor);
            amrex::Real const old_internal_energy_density =
                state_arr(i, j, k, 0);
            amrex::Real const minimum_internal_energy_density =
                state_arr(i, j, k, 2);
            amrex::Real const node_capacity =
                state_arr(i, j, k, 1) * node_volume;
            if (!amrex::Math::isfinite(old_internal_energy_density)
                || !amrex::Math::isfinite(minimum_internal_energy_density)
                || !(node_capacity > 0.0_rt)
                || !amrex::Math::isfinite(node_capacity))
            {
                Te_arr(i, j, k) =
                    std::numeric_limits<amrex::Real>::quiet_NaN();
                return;
            }

            // Gather adjacent cell energies. Weight each cell's contribution
            // by old-state C_V*V. This preserves each cell's energy while
            // avoiding extreme heating of sparse interface nodes.
            amrex::Real node_energy = 0.0_rt;
            if (use_nonlinear_lte_remap) {
                bool valid = true;
                for (int ck = k - (AMREX_SPACEDIM == 3 ? 1 : 0);
                     ck <= k; ++ck)
                {
                    for (int cj = j - (AMREX_SPACEDIM >= 2 ? 1 : 0);
                         cj <= j; ++cj)
                    {
                        for (int ci = i - 1; ci <= i; ++ci) {
                            node_energy += hybrid_nonlinear_lte_node_energy(
                                i, j, k, ci, cj, ck, source_arr,
                                nonlinear_remap_arr, state_arr, rho_arr,
                                Te_arr, material_charge_density,
                                thermodynamics, material_rho_threshold,
                                thermodynamic_rho_floor, problo, dx,
                                domain_lo, domain_hi, periodic, valid);
                            if (!valid) {
                                Te_arr(i, j, k) =
                                    std::numeric_limits<amrex::Real>
                                        ::quiet_NaN();
                                return;
                            }
                        }
                    }
                }
            } else {
                bool valid = true;
                for (int ck = k - (AMREX_SPACEDIM == 3 ? 1 : 0);
                     ck <= k; ++ck)
                {
                    for (int cj = j - (AMREX_SPACEDIM >= 2 ? 1 : 0);
                         cj <= j; ++cj)
                    {
                        for (int ci = i - 1; ci <= i; ++ci) {
                            node_energy += hybrid_constant_lte_node_energy(
                                i, j, k, ci, cj, ck, source_arr, state_arr,
                                rho_arr, material_rho_threshold, problo, dx,
                                domain_lo, domain_hi, periodic, valid);
                            if (!valid) {
                                Te_arr(i, j, k) =
                                    std::numeric_limits<amrex::Real>
                                        ::quiet_NaN();
                                return;
                            }
                        }
                    }
                }
            }
            if (node_energy == 0.0_rt) { return; }
            amrex::Real const energy_density_change = node_energy / node_volume;
            amrex::Real new_internal_energy_density =
                old_internal_energy_density + energy_density_change;
            amrex::Real const negative_tolerance = 100.0_rt
                * std::numeric_limits<amrex::Real>::epsilon()
                * amrex::max(
                    amrex::max(
                        std::abs(old_internal_energy_density),
                        std::abs(minimum_internal_energy_density)),
                    amrex::max(
                        std::abs(energy_density_change),
                        std::numeric_limits<amrex::Real>::min()));
            if (!amrex::Math::isfinite(new_internal_energy_density)
                || new_internal_energy_density
                    < minimum_internal_energy_density - negative_tolerance)
            {
                Te_arr(i, j, k) =
                    std::numeric_limits<amrex::Real>::quiet_NaN();
                return;
            }
            new_internal_energy_density = amrex::max(
                minimum_internal_energy_density,
                new_internal_energy_density);
            // A sub-ULP source must preserve the bitwise thermodynamic state so
            // the realized material ledger rebases to exact zero.
            if (new_internal_energy_density == old_internal_energy_density) {
                return;
            }
            auto const material_mass_density = thermodynamics
                .materialMassDensitiesFromChargeDensityArrays(
                    material_charge_density, i, j, k);
            amrex::Real const new_temperature = thermodynamics
                .temperatureFromMaterialMassDensitiesThermalEnergyDensity(
                    thermodynamic_rho_node, material_mass_density,
                    new_internal_energy_density);
            if (new_temperature < thermodynamics.minimumTemperature()
                || !amrex::Math::isfinite(new_temperature))
            {
                Te_arr(i, j, k) =
                    std::numeric_limits<amrex::Real>::quiet_NaN();
                return;
            }
            Te_arr(i, j, k) = new_temperature;
        });
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        Te.is_finite(0, 1, 0),
        "Hybrid electron energy coupling produced an invalid thermodynamic "
        "state or attempted to remove more electron internal energy than "
        "the participating material nodes contain.");

    // The next QDSMC step reconstructs entropy on a ghost-grown nodal box,
    // so radiation-updated temperatures must also be present in grid ghosts.
    Te.FillBoundary(Te.nGrowVect(), geom.periodicity());

    // The nonlinear remap above is only an input to the temperature solve. The
    // authoritative material transfer is the old-to-final EOS energy change at
    // each cell corner. Include every valid cell here: a zero-request cell may
    // still share a node whose temperature changed because of a neighboring
    // request.
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpMax>
        realization_reduce_ops;
    amrex::ReduceData<amrex::Real, int> realization_reduce_data(
        realization_reduce_ops);
    using RealizationReduceTuple =
        typename decltype(realization_reduce_data)::Type;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(
             cell_integrated_energy, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi)
    {
        amrex::Box const& box = mfi.tilebox();
        amrex::Array4<amrex::Real const> const source_arr =
            cell_integrated_energy.const_array(mfi);
        amrex::Array4<amrex::Real> const material_arr =
            cell_integrated_energy.array(mfi);
        amrex::Array4<amrex::Real const> const rho_arr = rho.const_array(mfi);
        amrex::Array4<amrex::Real const> const Te_arr = Te.const_array(mfi);
        amrex::Array4<amrex::Real const> const old_state_arr =
            node_thermodynamic_state.const_array(mfi);
        ElectronThermodynamicsExecutor::MaterialChargeDensityArrays
            material_charge_density{};
        for (int material = 0; material < num_materials; ++material) {
            material_charge_density[material] =
                material_charge_density_fields[material]->const_array(mfi);
        }

        realization_reduce_ops.eval(box, realization_reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
                -> RealizationReduceTuple
        {
            amrex::Real const requested_energy = source_arr(i, j, k);
            if (!amrex::Math::isfinite(requested_energy)) {
                return {0.0_rt, 1};
            }

            amrex::Real realized_energy = 0.0_rt;
            for (int dk = 0;
                 dk < (AMREX_SPACEDIM == 3 ? 2 : 1); ++dk)
            {
                for (int dj = 0;
                     dj < (AMREX_SPACEDIM >= 2 ? 2 : 1); ++dj)
                {
                    for (int di = 0; di < 2; ++di) {
                        int const ni = i + di;
                        int const nj = j + dj;
                        int const nk = k + dk;
                        amrex::Real const rho_node = rho_arr(ni, nj, nk);
                        if (!amrex::Math::isfinite(rho_node)) {
                            return {0.0_rt, 1};
                        }
                        if (rho_node <= material_rho_threshold) {
                            continue;
                        }

                        amrex::Real const old_internal_energy_density =
                            old_state_arr(ni, nj, nk, 0);
                        amrex::Real const final_temperature =
                            Te_arr(ni, nj, nk);
                        auto const material_mass_density = thermodynamics
                            .materialMassDensitiesFromChargeDensityArrays(
                                material_charge_density, ni, nj, nk);
                        ElectronThermodynamicState const final_state =
                            thermodynamics.stateFromMaterialMassDensitiesTemperature(
                                amrex::max(
                                    rho_node, thermodynamic_rho_floor),
                                material_mass_density, final_temperature);
                        amrex::Real const corner_volume =
                            hybrid_cell_corner_volume(
                                i, j, k, di, dj, dk, problo, dx, domain_lo);
                        amrex::Real const corner_energy =
                            (final_state.internal_energy_density
                                - old_internal_energy_density)
                            * corner_volume;
                        if (!amrex::Math::isfinite(
                                old_internal_energy_density)
                            || !amrex::Math::isfinite(final_temperature)
                            || final_temperature
                                < thermodynamics.minimumTemperature()
                            || final_temperature
                                > thermodynamics.maximumTemperature()
                            || !amrex::Math::isfinite(
                                final_state.internal_energy_density)
                            || !amrex::Math::isfinite(final_state.pressure)
                            || final_state.pressure < 0.0_rt
                            || !amrex::Math::isfinite(
                                final_state.heat_capacity_density)
                            || !(final_state.heat_capacity_density > 0.0_rt)
                            || !amrex::Math::isfinite(corner_volume)
                            || !(corner_volume > 0.0_rt)
                            || !amrex::Math::isfinite(corner_energy))
                        {
                            return {0.0_rt, 1};
                        }
                        realized_energy += corner_energy;
                    }
                }
            }

            amrex::Real const residual = requested_energy - realized_energy;
            if (!amrex::Math::isfinite(realized_energy)
                || !amrex::Math::isfinite(residual))
            {
                return {0.0_rt, 1};
            }
            material_arr(i, j, k) = realized_energy;
            return {residual, 0};
        });
    }

    auto const realization_reduction = realization_reduce_data.value();
    amrex::Real residual = amrex::get<0>(realization_reduction);
    int invalid_realization = amrex::get<1>(realization_reduction);
    amrex::ParallelDescriptor::ReduceRealSum(residual);
    amrex::ParallelDescriptor::ReduceIntMax(&invalid_realization, 1);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        invalid_realization == 0,
        "Hybrid electron radiation coupling produced a non-finite or "
        "invalid thermodynamic EOS realization while rebasing the material "
        "ledger.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        amrex::Math::isfinite(residual),
        "Hybrid electron radiation coupling produced a non-finite material "
        "realization residual.");

    cell_integrated_energy.FillBoundary(geom.periodicity());

    QDSMCFillElectronPressureFromTe(lev);
    warpx.ApplyElectronPressureBoundary(lev, PatchType::fine);
    ablastr::utils::communication::FillBoundary(
        *warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev),
        WarpX::do_single_precision_comms,
        geom.periodicity(),
        true);
    return residual;
}

void HybridPICModel::ApplyHybridIonizationEnergySource (
    int const lev,
    amrex::MultiFab const& old_charge_density,
    amrex::MultiFab const& new_charge_density,
    amrex::MultiFab const& binding_energy_density) const
{
    ABLASTR_PROFILE("HybridPICModel::ApplyHybridIonizationEnergySource()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_solve_electron_energy_equation,
        "Hybrid ionization requires the hybrid electron-energy equation.");
    auto& warpx = WarpX::GetInstance();
    amrex::MultiFab& Te = *warpx.m_fields.get(
        FieldType::hybrid_electron_temperature_fp, lev);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        old_charge_density.boxArray() == Te.boxArray()
            && new_charge_density.boxArray() == Te.boxArray()
            && binding_energy_density.boxArray() == Te.boxArray()
            && old_charge_density.DistributionMap() == Te.DistributionMap()
            && new_charge_density.DistributionMap() == Te.DistributionMap()
            && binding_energy_density.DistributionMap()
                == Te.DistributionMap()
            && old_charge_density.ixType() == Te.ixType()
            && new_charge_density.ixType() == Te.ixType()
            && binding_energy_density.ixType() == Te.ixType(),
        "Hybrid ionization rho, Te, and binding-energy fields must share one "
        "nodal layout and distribution map.");

    auto const thermodynamics = m_electron_thermodynamics.executor();
    amrex::Real const rho_floor = PhysConst::q_e * m_n_floor;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(Te, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi)
    {
        amrex::Box const& box = mfi.tilebox();
        amrex::Array4<amrex::Real> const Te_arr = Te.array(mfi);
        amrex::Array4<amrex::Real const> const old_rho_arr =
            old_charge_density.const_array(mfi);
        amrex::Array4<amrex::Real const> const new_rho_arr =
            new_charge_density.const_array(mfi);
        amrex::Array4<amrex::Real const> const binding_arr =
            binding_energy_density.const_array(mfi);

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (
            int i, int j, int k) noexcept
        {
            amrex::Real const old_rho =
                amrex::max(old_rho_arr(i, j, k), rho_floor);
            amrex::Real const new_rho =
                amrex::max(new_rho_arr(i, j, k), rho_floor);
            amrex::Real const binding_energy = binding_arr(i, j, k);
            if (!amrex::Math::isfinite(binding_energy)
                || binding_energy < 0.0_rt)
            {
                Te_arr(i, j, k) =
                    std::numeric_limits<amrex::Real>::quiet_NaN();
                return;
            }

            ElectronThermodynamicState const old_state =
                thermodynamics.stateFromChargeDensityTemperature(
                    old_rho, Te_arr(i, j, k));
            amrex::Real target_internal_energy =
                old_state.internal_energy_density - binding_energy;
            amrex::Real const tolerance = 100.0_rt
                * std::numeric_limits<amrex::Real>::epsilon()
                * amrex::max(
                    std::abs(old_state.internal_energy_density),
                    amrex::max(
                        binding_energy,
                        std::numeric_limits<amrex::Real>::min()));
            if (!amrex::Math::isfinite(old_state.internal_energy_density)
                || !amrex::Math::isfinite(target_internal_energy)
                || target_internal_energy < -tolerance)
            {
                Te_arr(i, j, k) =
                    std::numeric_limits<amrex::Real>::quiet_NaN();
                return;
            }
            target_internal_energy =
                amrex::max(0.0_rt, target_internal_energy);
            amrex::Real const new_temperature = thermodynamics
                .temperatureFromChargeDensityThermalEnergyDensity(
                    new_rho, target_internal_energy);
            ElectronThermodynamicState const new_state =
                thermodynamics.stateFromChargeDensityTemperature(
                    new_rho, new_temperature);
            amrex::Real const residual =
                new_state.internal_energy_density - target_internal_energy;
            amrex::Real const residual_tolerance = 256.0_rt
                * std::numeric_limits<amrex::Real>::epsilon()
                * amrex::max(
                    std::abs(target_internal_energy),
                    std::numeric_limits<amrex::Real>::min());
            if (!amrex::Math::isfinite(new_temperature)
                || new_temperature < 0.0_rt
                || !amrex::Math::isfinite(new_state.internal_energy_density)
                || std::abs(residual) > residual_tolerance)
            {
                Te_arr(i, j, k) =
                    std::numeric_limits<amrex::Real>::quiet_NaN();
                return;
            }
            Te_arr(i, j, k) = new_temperature;
        });
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        Te.is_finite(0, 1, 0),
        "Hybrid ionization encountered an invalid thermodynamic state or "
        "requested more binding energy than the hybrid electrons contain. "
        "Reduce the timestep or ionization rate.");

    auto const& geom = warpx.Geom(lev);
    Te.FillBoundary(Te.nGrowVect(), geom.periodicity());
    QDSMCFillElectronPressureFromTe(lev);
    warpx.ApplyElectronPressureBoundary(lev, PatchType::fine);
    ablastr::utils::communication::FillBoundary(
        *warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev),
        WarpX::do_single_precision_comms,
        geom.periodicity(), true);
}


void HybridPICModel::AdvanceElectronEnergyQDSMC (amrex::Real const dt) const
{
    ABLASTR_PROFILE("HybridPICModel::AdvanceElectronEnergyQDSMC()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_qdsmc_pc != nullptr,
        "AdvanceElectronEnergyQDSMC called with "
        "solve_electron_energy_equation=true but the "
        "QDSMC particle container was not constructed (InitData not run?)");

    auto & warpx = WarpX::GetInstance();

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_fv_transport_internal_energy || warpx.finestLevel() == 0,
        "Nonlinear finite-volume electron-energy transport currently "
        "requires a single AMR level; coarse/fine energy refluxing is not "
        "implemented yet.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_fv_transport_internal_energy || !WarpX::use_filter,
        "Nonlinear finite-volume electron-energy transport currently "
        "requires warpx.use_filter=0 so rho and the electron charge flux "
        "satisfy the same discrete continuity equation.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_fv_transport_internal_energy
            || !WarpX::do_single_precision_comms,
        "Nonlinear finite-volume electron-energy transport currently "
        "requires warpx.do_single_precision_comms=0. Reduced-precision "
        "guard-cell sums break the discrete rho/current continuity ledger.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_fv_transport_internal_energy || !warpx.DoFluidSpecies(),
        "Nonlinear finite-volume electron-energy transport does not yet "
        "support cold-fluid species because their auxiliary conservative "
        "charge and velocity moments are not deposited.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_fv_transport_internal_energy || !EB::enabled(),
        "Nonlinear finite-volume electron-energy transport does not yet "
        "include embedded-boundary face metrics and masks.");

    // J_plasma (at B^n) is needed for V_e. On all but the first step it is
    // already valid: the previous step's final E-solve computed it from B^n
    // (the external-field subtract at the top of this step exactly cancels
    // the re-add at the end of the previous one), and B has not changed
    // since. Only the first step of a run or restart arrives here with an
    // unfilled J_plasma.
    if (!m_qdsmc_J_plasma_valid) {
        CalculatePlasmaCurrent(
            warpx.m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, warpx.finestLevel()),
            warpx.GetEBUpdateEFlag());
        m_qdsmc_J_plasma_valid = true;
    }

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        // Step 1: grid-side initialization at t = n
        QDSMCInitializeUe(lev);
        QDSMCInitializeThermodynamicQuantity(lev);

        if (m_fv_transport_internal_energy) {
            // Nonlinear thermodynamics use a conservative first-order upwind
            // finite-volume update of dU/dt + div(U V_e) = -P div(V_e).
            // This is exactly stationary at V_e=0 and avoids applying a
            // dissipative particle gather/scatter every PIC step.
            QDSMCUpdateThermodynamics(lev, dt);
        } else {
            using ablastr::fields::Direction;
            amrex::MultiFab const& Vex = *warpx.m_fields.get(
                FieldType::hybrid_electron_velocity_fp, Direction{0}, lev);
            amrex::MultiFab const& Vey = *warpx.m_fields.get(
                FieldType::hybrid_electron_velocity_fp, Direction{1}, lev);
            amrex::MultiFab const& Vez = *warpx.m_fields.get(
                FieldType::hybrid_electron_velocity_fp, Direction{2}, lev);
            amrex::MultiFab const& Ke = *warpx.m_fields.get(
                FieldType::hybrid_qdsmc_thermodynamic_fp, lev);
            amrex::MultiFab const& rho = *warpx.m_fields.get(
                FieldType::hybrid_rho_fp_temp, lev);
            amrex::MultiFab& Karr_out = *warpx.m_fields.get(
                FieldType::hybrid_qdsmc_thermodynamic_fp, lev);
            amrex::MultiFab& weights_out = *warpx.m_fields.get(
                FieldType::hybrid_qdsmc_weights_fp, lev);

            // The reviewed ideal-gas algorithm transports K_e with one
            // Lagrangian marker per cell.
            m_qdsmc_pc->SetV(lev, Vex, Vey, Vez);
            m_qdsmc_pc->SetK(lev, Ke, rho);
            m_qdsmc_pc->PushX(lev, dt);
            m_qdsmc_pc->DepositK(lev, Karr_out);
            m_qdsmc_pc->DepositField(lev, weights_out);
            QDSMCUpdateThermodynamics(lev, dt);
        }

        // Step 6: Joule-heating source on U_e (Phys. Plasmas 31, 012902 (2024), Eq. 12), per-cell from
        // rho_fp(_s), the plasma current, and the Ohm's-law eta parser. With the
        // Te-threshold redirect on, the above-threshold heat is staged in
        // ion_redirect_E (per-charged-species energy, J) for the ion-heating step.
        bool redirect_active = m_include_joule_heating && m_joule_redirect_to_ions;
        int n_ion_species = 0;
        if (redirect_active) {
            auto & mpc = warpx.GetPartContainer();
            for (auto const & nm : mpc.GetSpeciesNames()) {
                if (mpc.GetParticleContainerFromName(nm).getCharge() != 0._prt) { ++n_ion_species; }
            }
            if (n_ion_species == 0) { redirect_active = false; }
        }
        amrex::MultiFab ion_redirect_E;
        if (redirect_active) {
            amrex::MultiFab const & Te_mf =
                *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
            ion_redirect_E.define(Te_mf.boxArray(), Te_mf.DistributionMap(), n_ion_species, 0);
            ion_redirect_E.setVal(0.0_rt);
        }
        if (m_include_joule_heating) {
            QDSMCAddJouleHeating(lev, dt, redirect_active ? &ion_redirect_E : nullptr);
        }

        // Steps 6b/6c both need each charged species' T_i when Q_ei relaxation is
        // on. Deposit it ONCE here (the expensive per-particle NGP temperature
        // reduction) and share it: the electron sink (6b) and the ion-heating
        // operator (6c) run back-to-back with no intervening ion motion, so the
        // deposited T_i is identical for both.
        std::map<std::string, amrex::MultiFab*> Ti_dep_by_species;
        // Owns the per-species cell-centered scalar T_i built from the shape-aware
        // deposition below; must outlive the QDSMCAddTemperatureRelaxation /
        // QDSMCApplyIonHeating calls that read it through Ti_dep_by_species.
        std::map<std::string, std::unique_ptr<amrex::MultiFab>> Ti_scalar_owned;
        if (m_include_temperature_relaxation) {
            using ablastr::fields::Direction;
            amrex::GpuArray<int, 3> const Tr_stag = Jx_IndexType;
            amrex::GpuArray<int, 3> const Tt_stag = Jy_IndexType;
            amrex::GpuArray<int, 3> const Tz_stag = Jz_IndexType;
            amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};
            // Cell-centered target in the real dimensions; in collapsed dimensions
            // (index >= AMREX_SPACEDIM, e.g. theta in RZ or y in 2D) match the source
            // staggering so Interp does not read the out-of-bounds neighbour there.
            amrex::GpuArray<int, 3> cc_r = {0, 0, 0};
            amrex::GpuArray<int, 3> cc_t = {0, 0, 0};
            amrex::GpuArray<int, 3> cc_z = {0, 0, 0};
            for (int d = AMREX_SPACEDIM; d < 3; ++d) {
                cc_r[d] = Tr_stag[d]; cc_t[d] = Tt_stag[d]; cc_z[d] = Tz_stag[d];
            }

            auto & mpc_ti = warpx.GetPartContainer();
            for (auto const & nm : mpc_ti.GetSpeciesNames()) {
                auto & pc = mpc_ti.GetParticleContainerFromName(nm);
                if (pc.getCharge() == 0._prt) { continue; }
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(pc.getTemperatureDepositionFlag(),
                    "The Q_ei temperature relaxation requires do_temperature_deposition "
                    "on every charged ion species; it is enabled automatically at species "
                    "construction, so hitting this indicates the species was created "
                    "before the hybrid_pic_model relaxation parameters were readable.");

                // Shape-aware ion temperature (particle-shape order, consistent with
                // charge/current) in the Yee-staggered 3-component T_<nm> vector field
                // (Tr,Tt,Tz), deposited in Kelvin by HybridPICDepositRhoAndJ ->
                // mypc->DepositTemperatures earlier this step and read here. Fill guard
                // cells so the cell-centered interpolation below reads finite
                // neighbours at box/domain edges.
                auto const T_vf = warpx.m_fields.get_mr_levels_alldirs("T_" + nm, warpx.finestLevel());
                for (int idim = 0; idim < 3; ++idim) {
                    T_vf[lev][Direction{idim}]->FillBoundary(warpx.Geom(lev).periodicity());
                }

                // Collapse the staggered vector to a cell-centered scalar
                // T_i = (Tr + Tt + Tz)/3 by interpolating each component to CC
                // (Path A: accept the CC interpolation for shape consistency).
                amrex::MultiFab const & Te_ref =
                    *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
                amrex::BoxArray const cc_ba =
                    amrex::convert(Te_ref.boxArray(), amrex::IntVect::TheCellVector());
                auto Ti_s = std::make_unique<amrex::MultiFab>(
                    cc_ba, Te_ref.DistributionMap(), 1, 0);
                Ti_s->setVal(0.0_rt);

                // AccumulateVelocitiesAndComputeTemperature writes T_<nm> in Kelvin;
                // the Q_ei consumers (and the nu_ei parser) expect T_i in eV, matching
                // the previous NGP deposit. Convert K -> eV below.
                amrex::Real const K_per_eV = PhysConst::q_e / PhysConst::kb;

                amrex::MultiFab const & Tr = *T_vf[lev][Direction{0}];
                amrex::MultiFab const & Tt = *T_vf[lev][Direction{1}];
                amrex::MultiFab const & Tz = *T_vf[lev][Direction{2}];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (amrex::MFIter mfi(*Ti_s, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    amrex::Box const & bx = mfi.tilebox();
                    amrex::Array4<amrex::Real>       const & Ti_arr = Ti_s->array(mfi);
                    amrex::Array4<amrex::Real const> const & Tr_arr = Tr.const_array(mfi);
                    amrex::Array4<amrex::Real const> const & Tt_arr = Tt.const_array(mfi);
                    amrex::Array4<amrex::Real const> const & Tz_arr = Tz.const_array(mfi);
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        amrex::Real const tr = ablastr::coarsen::sample::Interp(
                            Tr_arr, Tr_stag, cc_r, coarsen, i, j, k, 0);
                        amrex::Real const tt = ablastr::coarsen::sample::Interp(
                            Tt_arr, Tt_stag, cc_t, coarsen, i, j, k, 0);
                        amrex::Real const tz = ablastr::coarsen::sample::Interp(
                            Tz_arr, Tz_stag, cc_z, coarsen, i, j, k, 0);
                        Ti_arr(i, j, k) = (tr + tt + tz) / (3._rt * K_per_eV);  // K -> eV
                    });
                }
                Ti_dep_by_species[nm] = Ti_s.get();
                Ti_scalar_owned[nm] = std::move(Ti_s);
            }
        }

        // Step 6b: electron-ion thermal-equilibration (Q_ei) sink on T_e
        // (cools T_e toward each ion species' T_i).
        if (m_include_temperature_relaxation) {
            QDSMCAddTemperatureRelaxation(lev, dt, Ti_dep_by_species);
        }

        // Step 6c: the Qei OU proposal is projected cell-locally so its ion
        // thermal-energy change is exactly the negative electron ledger while
        // preserving each cell's pre-step ion momentum. Joule redirection is
        // still a separate stochastic diffusion channel; separate calls keep
        // its source from contaminating the conservative Qei projection.
        if (m_include_temperature_relaxation) {
            QDSMCApplyIonHeating(lev, dt, nullptr, &Ti_dep_by_species);
        }
        if (redirect_active) {
            QDSMCApplyIonHeating(lev, dt, &ion_redirect_E, nullptr);
        }

        // Step 7: emit P_e = n_e * k_B * T_e for the downstream Ohm's-law
        // solve, with the same boundary treatment the algebraic closure gets
        // in CalculateElectronPressure (grad P_e reads the ghost cells).
        QDSMCFillElectronPressureFromTe(lev);
        warpx.ApplyElectronPressureBoundary(lev, PatchType::fine);
        ablastr::utils::communication::FillBoundary(
            *warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev),
            WarpX::do_single_precision_comms,
            warpx.Geom(lev).periodicity(),
            true);

        // Reset only after the ideal-gas marker path. Nonlinear transport does
        // not mutate the QDSMC particle container.
        if (!m_fv_transport_internal_energy) {
            m_qdsmc_pc->ResetParticles(lev);
        }
    }
}


void HybridPICModel::BfieldEvolve (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    int step, amrex::Real dt_half, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        BfieldEvolve(
            Bfield, Efield, Jfield, rhofield, eb_update_E,
            step, dt_half, lev, subcycling_half, ng, nodal_sync
        );
    }
}

void HybridPICModel::BfieldEvolve (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    int step, amrex::Real dt_half, int lev, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    bool use_rkf45 = DoRKF45(step);
    // Make copies of the current B-field multifabs (at t = n) since the
    // starting B-field is needed for the integration logic.
    // We also store the initial B-field from the start of this integration step
    // (i.e., a static copy) in case we need to fully reset the Bfield (needed
    // for RK4).
    std::array< MultiFab, 3 > B_old;
    for (int ii = 0; ii < 3; ii++)
    {
        B_old[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 2, ng
        );
        MultiFab::Copy(B_old[ii], *Bfield[lev][ii], 0, 0, 1, ng);
        // the values at index 1 will be kept static through the integration steps
        MultiFab::Copy(B_old[ii], *Bfield[lev][ii], 0, 1, 1, ng);
    }

    amrex::Real dt_sub = dt_half / (m_substeps / 2._rt);
    amrex::Real t = 0._rt;
    int n_attempts = 0;
    int n_accepted = 0;

    // Step the magnetic field forward (from t -> t + dt_half) using the user
    // specified integration scheme. The loop is set up such that the timestep
    // for a given step (dt_sub) can be modified within the loop, i.e.,
    // adaptive timestepping.
    while (t < dt_half)
    {
        // Adjust size of the last substep, so as to land exactly at t+dt_half.
        if (t + dt_sub > dt_half) { dt_sub = dt_half - t; }
        bool step_succeeded = true;
        amrex::Real step_change_factor = 1.0_rt;

        if (use_rkf45) {
            const amrex::Real error = BfieldEvolveRKF45(
                Bfield, Efield, Jfield, rhofield, eb_update_E, B_old,
                dt_sub, lev, subcycling_half, ng, nodal_sync
            );

            step_change_factor = m_substep_safety * std::pow(error + 1.e-10_rt, -0.2_rt);
            step_succeeded = (error <= 1._rt);

        } else {
            BfieldEvolveRK4(
                Bfield, Efield, Jfield, rhofield, eb_update_E, B_old,
                dt_sub, lev, subcycling_half, ng, nodal_sync
            );

            // Check that the B-field does not have nan or inf values
            for (int idim = 0; idim < 3; ++idim) {
                step_succeeded = step_succeeded && Bfield[lev][idim]->is_finite(/*local=*/true);
            }
            amrex::ParallelDescriptor::ReduceBoolAnd(step_succeeded);

            if (!step_succeeded) {
                ablastr::warn_manager::WMRecordWarning(
                    "HybridPIC",
                    "NaN or Inf value encountered in the B-field during RK4 "
                    "substepping. Restarting this step using RKF45.",
                    ablastr::warn_manager::WarnPriority::medium);

                // restart this full step and this time use RKF45
                t = 0._rt;
                n_accepted = 0;
                // reset B_old to original one
                for (int ii = 0; ii < 3; ii++) {
                    MultiFab::Copy(B_old[ii], B_old[ii], 1, 0, 1, ng);
                }
                use_rkf45 = true;
            }
        }

        if (step_succeeded) {
            // update time tracker and accepted steps number
            t += dt_sub;
            ++n_accepted;
            // update B_old to the current Bfield
            for (int ii = 0; ii < 3; ii++) {
                MultiFab::Copy(B_old[ii], *Bfield[lev][ii], 0, 0, 1, ng);
            }
            dt_sub *= std::min(m_substep_max_growth, step_change_factor);
        } else {
            // reset Bfield to B_old before trying the integration again
            for (int ii = 0; ii < 3; ii++) {
                MultiFab::Copy(*Bfield[lev][ii], B_old[ii], 0, 0, 1, ng);
            }
            dt_sub *= std::max(0.1_rt, step_change_factor);
        }

        if (++n_attempts > m_max_substep_attempts) { break; }
    }

    // Adjust the number of substeps for the next RKF45/RK4 half-step.
    // Jump up immediately when this half needed more attempts; otherwise
    // slowly relax toward target = 2*n_attempts.
    // Blend on the half-step counts M = m_substeps/2 and N = n_attempts via
    // integer arithmetic: relaxed = 2*((19*M + N)/20). That is exactly the
    // 95/5 blend, stays even, holds when N == M, and actually decays when
    // N < M (e.g. m=40, n_attempts=10 → 38 → … → 20). Floating-point
    // 0.95*M+0.05*N can undershoot M slightly so floor would leak even at
    // equilibrium.
    {
        const int target = 2 * n_attempts;
        if (m_substeps < target) {
            m_substeps = target;
        } else {
            const int M = m_substeps / 2;
            const int N = n_attempts;
            const int relaxed = 2 * ((19 * M + N) / 20);
            m_substeps = std::max(relaxed, 2);
        }
        // Stay within the abort budget so the controller cannot request more
        // substeps than max_substep_attempts allows.
        if (m_substeps > m_max_substep_attempts) {
            m_substeps = m_max_substep_attempts - (m_max_substep_attempts % 2);
            m_substeps = std::max(m_substeps, 2);
        }
    }

    if (WarpX::GetInstance().Verbose()) {
        amrex::Print() << "B-field update "
            << (subcycling_half == SubcyclingHalf::FirstHalf ? "1st" : "2nd") << " half"
            << ": " << n_accepted << " accepted, "
            << (n_attempts - n_accepted) << " rejected substeps"
            << " (dt_sub_final/dt_half = " << dt_sub / dt_half
            << ", m_substeps = " << m_substeps << ")\n";
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        n_attempts <= m_max_substep_attempts,
        "BfieldEvolve: exceeded max substep attempts;"
        "consider relaxing hybrid_pic_model.substep_rtol/substep_atol."
    );
}

void HybridPICModel::BfieldEvolveRK4 (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    std::array<amrex::MultiFab, 3>& B_old,
    amrex::Real dt, int lev, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    // Create multifabs for each direction to store the Runge-Kutta intermediate terms.
    // Each multifab has 2 components for the different terms that need to be stored.
    std::array< MultiFab, 3 > K;
    for (int ii = 0; ii < 3; ii++)
    {
        K[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 2,
            Bfield[lev][ii]->nGrowVect()
        );
    }

    // The Runge-Kutta scheme begins here.
    // Step 1:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, eb_update_E,
        0.5_rt*dt, subcycling_half, ng, nodal_sync
    );

    // The Bfield is now given by:
    // B_new = B_old + 0.5 * dt * [-curl x E(B_old)] = B_old + 0.5 * dt * K0.
    for (int ii = 0; ii < 3; ii++)
    {
        // Extract 0.5 * dt * K0 for each direction into index 0 of K.
        MultiFab::LinComb(
            K[ii], 1._rt, *Bfield[lev][ii], 0, -1._rt, B_old[ii], 0, 0, 1, ng
        );
    }

    // Step 2:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, eb_update_E,
        0.5_rt*dt, subcycling_half, ng, nodal_sync
    );

    // The Bfield is now given by:
    //   B_new = B_old + 0.5 * dt * K0 + 0.5 * dt * [-curl x E(B_old + 0.5 * dt * K1)]
    //         = B_old + 0.5 * dt * K0 + 0.5 * dt * K1
    //
    // Subtract 0.5 * dt * K0 from the Bfield to get
    //   B_new = B_old + 0.5 * dt * K1.
    // Extract 0.5 * dt * K1 and write into index 1 of K.

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const &Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const &By = Bfield[lev][1]->array(mfi);
        Array4<Real> const &Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const &Kx = K[0].array(mfi);
        Array4<Real> const &Ky = K[1].array(mfi);
        Array4<Real> const &Kz = K[2].array(mfi);
        Array4<Real const> const &Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const &By_old = B_old[1].const_array(mfi);
        Array4<Real const> const &Bz_old = B_old[2].const_array(mfi);

        // Extract tileboxes for which to loop
        Box const& tjx  = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy  = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz  = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);

        amrex::ParallelFor(tjx, tjy, tjz,
            // x calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Bx(i, j, k) -= Kx(i, j, k, 0);
                Kx(i, j, k, 1) = Bx(i, j, k) - Bx_old(i, j, k);
            },

            // y calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                By(i, j, k) -= Ky(i, j, k, 0);
                Ky(i, j, k, 1) = By(i, j, k) - By_old(i, j, k);
            },

            // z calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Bz(i, j, k) -= Kz(i, j, k, 0);
                Kz(i, j, k, 1) = Bz(i, j, k) - Bz_old(i, j, k);
            }
        );
    }

    // Step 3:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, eb_update_E,
        dt, subcycling_half, ng, nodal_sync
    );

    // The Bfield is now given by:
    // B_new = B_old + 0.5 * dt * K1 + dt * [-curl  x E(B_old + 0.5 * dt * K1)]
    //       = B_old + 0.5 * dt * K1 + dt * K2
    for (int ii = 0; ii < 3; ii++)
    {
        // Subtract 0.5 * dt * K1 from the Bfield for each direction to get
        // B_new = B_old + dt * K2.
        MultiFab::Subtract(*Bfield[lev][ii], K[ii], 1, 0, 1, ng);
    }

    // Step 4:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, eb_update_E,
        0.5_rt*dt, subcycling_half, ng, nodal_sync
    );

    // The Bfield is now given by:
    //   B_new = B_old + dt * K2 + 0.5 * dt * [-curl x E(B_old + dt * K2)]
    //         = B_old + dt * K2 + 0.5 * dt * K3
    // and
    //   index 0 of K = 0.5 * dt * K0
    //   index 1 of K = 0.5 * dt * K1
    //
    // We calculate:
    //   K = 0.5 * dt * K0 + dt * K1 + dt * K2 + 0.5 * dt * K3
    // then update B with the Runge-Kutta sum:
    //   B = B_old + 1/3 * K

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const &Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const &By = Bfield[lev][1]->array(mfi);
        Array4<Real> const &Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const &Kx = K[0].array(mfi);
        Array4<Real> const &Ky = K[1].array(mfi);
        Array4<Real> const &Kz = K[2].array(mfi);
        Array4<Real const> const &Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const &By_old = B_old[1].const_array(mfi);
        Array4<Real const> const &Bz_old = B_old[2].const_array(mfi);

        // Extract tileboxes for which to loop
        Box const& tjx  = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy  = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz  = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);

        amrex::ParallelFor(tjx, tjy, tjz,
            // Bx calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Kx(i, j, k, 0) += Bx(i, j, k) - Bx_old(i, j, k) + 2.0_rt * Kx(i, j, k, 1);
                Bx(i, j, k) = Bx_old(i, j, k) + Kx(i, j, k, 0) / 3.0_rt;
            },

            // By calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Ky(i, j, k, 0) += By(i, j, k) - By_old(i, j, k) + 2.0_rt * Ky(i, j, k, 1);
                By(i, j, k) = By_old(i, j, k) + Ky(i, j, k, 0) / 3.0_rt;
            },

            // Bz calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Kz(i, j, k, 0) += Bz(i, j, k) - Bz_old(i, j, k) + 2.0_rt * Kz(i, j, k, 1);
                Bz(i, j, k) = Bz_old(i, j, k) + Kz(i, j, k, 0) / 3.0_rt;
            }
        );
    }
}

amrex::Real HybridPICModel::BfieldEvolveRKF45 (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    std::array<amrex::MultiFab, 3>& B_old,
    amrex::Real dt, int lev, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    // Fehlberg RKF45 Butcher tableau coefficients
    constexpr amrex::Real a21 = 1._rt/4._rt;
    constexpr amrex::Real a31 = 3._rt/32._rt,      a32 = 9._rt/32._rt;
    constexpr amrex::Real a41 = 1932._rt/2197._rt,  a42 = -7200._rt/2197._rt, a43 = 7296._rt/2197._rt;
    constexpr amrex::Real a51 = 439._rt/216._rt,    a52 = -8._rt,
                          a53 = 3680._rt/513._rt,    a54 = -845._rt/4104._rt;
    constexpr amrex::Real a61 = -8._rt/27._rt,      a62 = 2._rt,
                          a63 = -3544._rt/2565._rt,  a64 = 1859._rt/4104._rt,  a65 = -11._rt/40._rt;
    // 4th-order solution weights (k2 and k6 terms are zero in Fehlberg's formula)
    constexpr amrex::Real b1 = 25._rt/216._rt,  b3 = 1408._rt/2565._rt,
                          b4 = 2197._rt/4104._rt, b5 = -1._rt/5._rt;
    // Error = B5 - B4 weights: h*(e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6)
    constexpr amrex::Real e1 =  1._rt/360._rt,    e3 = -128._rt/4275._rt,
                          e4 = -2197._rt/75240._rt, e5 = 1._rt/50._rt, e6 = 2._rt/55._rt;

    // K: 5 components per field direction stored as:
    //   comp 0 = h*k1, comp 1 = h*k2 (overwritten with h*k6 after stage 6),
    //   comp 2 = h*k3, comp 3 = h*k4, comp 4 = h*k5
    std::array<MultiFab, 3> K;
    std::array<MultiFab, 3> err_scratch;
    for (int ii = 0; ii < 3; ii++)
    {
        K[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 5,
            Bfield[lev][ii]->nGrowVect()
        );
        err_scratch[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 1,
            amrex::IntVect(0)
        );
    }

    // ---- Stage 1: B = B_old, FieldPush, K[comp0] = h*k1 fused with Stage 2 B-update ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx, tjy, tjz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Bx(i, j, k) - Bx_old(i, j, k);
                Kx(i, j, k, 0) = k1;
                Bx(i, j, k) = Bx_old(i, j, k) + a21*k1;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = By(i, j, k) - By_old(i, j, k);
                Ky(i, j, k, 0) = k1;
                By(i, j, k) = By_old(i, j, k) + a21*k1;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Bz(i, j, k) - Bz_old(i, j, k);
                Kz(i, j, k, 0) = k1;
                Bz(i, j, k) = Bz_old(i, j, k) + a21*k1;
            }
        );
    }

    // ---- Stage 2: FieldPush, K[comp1] = h*k2 fused with Stage 3 B-update ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
    // Stage 2 K[1]-readback fused with Stage 3 B-update.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx, tjy, tjz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kx(i, j, k, 0);
                amrex::Real const k2 = Bx(i, j, k) - Bx_old(i, j, k) - a21*k1;
                Kx(i, j, k, 1) = k2;
                Bx(i, j, k) = Bx_old(i, j, k) + a31*k1 + a32*k2;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Ky(i, j, k, 0);
                amrex::Real const k2 = By(i, j, k) - By_old(i, j, k) - a21*k1;
                Ky(i, j, k, 1) = k2;
                By(i, j, k) = By_old(i, j, k) + a31*k1 + a32*k2;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kz(i, j, k, 0);
                amrex::Real const k2 = Bz(i, j, k) - Bz_old(i, j, k) - a21*k1;
                Kz(i, j, k, 1) = k2;
                Bz(i, j, k) = Bz_old(i, j, k) + a31*k1 + a32*k2;
            }
        );
    }

    // ---- Stage 3: FieldPush, then K[comp2] = h*k3 fused with Stage 4 B-update ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx, tjy, tjz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kx(i, j, k, 0);
                amrex::Real const k2 = Kx(i, j, k, 1);
                amrex::Real const k3 = Bx(i, j, k) - Bx_old(i, j, k) - a31*k1 - a32*k2;
                Kx(i, j, k, 2) = k3;
                Bx(i, j, k) = Bx_old(i, j, k) + a41*k1 + a42*k2 + a43*k3;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Ky(i, j, k, 0);
                amrex::Real const k2 = Ky(i, j, k, 1);
                amrex::Real const k3 = By(i, j, k) - By_old(i, j, k) - a31*k1 - a32*k2;
                Ky(i, j, k, 2) = k3;
                By(i, j, k) = By_old(i, j, k) + a41*k1 + a42*k2 + a43*k3;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kz(i, j, k, 0);
                amrex::Real const k2 = Kz(i, j, k, 1);
                amrex::Real const k3 = Bz(i, j, k) - Bz_old(i, j, k) - a31*k1 - a32*k2;
                Kz(i, j, k, 2) = k3;
                Bz(i, j, k) = Bz_old(i, j, k) + a41*k1 + a42*k2 + a43*k3;
            }
        );
    }

    // ---- Stage 4: FieldPush, then K[comp3] = h*k4 fused with Stage 5 B-update ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx, tjy, tjz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kx(i, j, k, 0);
                amrex::Real const k2 = Kx(i, j, k, 1);
                amrex::Real const k3 = Kx(i, j, k, 2);
                amrex::Real const k4 = Bx(i, j, k) - Bx_old(i, j, k)
                                        - a41*k1 - a42*k2 - a43*k3;
                Kx(i, j, k, 3) = k4;
                Bx(i, j, k) = Bx_old(i, j, k) + a51*k1 + a52*k2 + a53*k3 + a54*k4;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Ky(i, j, k, 0);
                amrex::Real const k2 = Ky(i, j, k, 1);
                amrex::Real const k3 = Ky(i, j, k, 2);
                amrex::Real const k4 = By(i, j, k) - By_old(i, j, k)
                                        - a41*k1 - a42*k2 - a43*k3;
                Ky(i, j, k, 3) = k4;
                By(i, j, k) = By_old(i, j, k) + a51*k1 + a52*k2 + a53*k3 + a54*k4;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kz(i, j, k, 0);
                amrex::Real const k2 = Kz(i, j, k, 1);
                amrex::Real const k3 = Kz(i, j, k, 2);
                amrex::Real const k4 = Bz(i, j, k) - Bz_old(i, j, k)
                                        - a41*k1 - a42*k2 - a43*k3;
                Kz(i, j, k, 3) = k4;
                Bz(i, j, k) = Bz_old(i, j, k) + a51*k1 + a52*k2 + a53*k3 + a54*k4;
            }
        );
    }

    // ---- Stage 5: FieldPush, then K[comp4] = h*k5 fused with Stage 6 B-update ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx, tjy, tjz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kx(i, j, k, 0);
                amrex::Real const k2 = Kx(i, j, k, 1);
                amrex::Real const k3 = Kx(i, j, k, 2);
                amrex::Real const k4 = Kx(i, j, k, 3);
                amrex::Real const k5 = Bx(i, j, k) - Bx_old(i, j, k)
                                        - a51*k1 - a52*k2 - a53*k3 - a54*k4;
                Kx(i, j, k, 4) = k5;
                Bx(i, j, k) = Bx_old(i, j, k)
                            + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Ky(i, j, k, 0);
                amrex::Real const k2 = Ky(i, j, k, 1);
                amrex::Real const k3 = Ky(i, j, k, 2);
                amrex::Real const k4 = Ky(i, j, k, 3);
                amrex::Real const k5 = By(i, j, k) - By_old(i, j, k)
                                        - a51*k1 - a52*k2 - a53*k3 - a54*k4;
                Ky(i, j, k, 4) = k5;
                By(i, j, k) = By_old(i, j, k)
                            + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kz(i, j, k, 0);
                amrex::Real const k2 = Kz(i, j, k, 1);
                amrex::Real const k3 = Kz(i, j, k, 2);
                amrex::Real const k4 = Kz(i, j, k, 3);
                amrex::Real const k5 = Bz(i, j, k) - Bz_old(i, j, k)
                                        - a51*k1 - a52*k2 - a53*k3 - a54*k4;
                Kz(i, j, k, 4) = k5;
                Bz(i, j, k) = Bz_old(i, j, k)
                            + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5;
            }
        );
    }

    // ---- Stage 6: FieldPush, then K[comp1] = h*k6 (overwrites h*k2) fused with B4 + error ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
    // K[comp1] is overwritten here: reads h*k2 (old value) then writes h*k6 in each cell.
    // k6, B4 assembly (b2=0, so k2 is not needed for B4), and error assembly are fused into
    // one ParallelFor per direction. B4 is updated over ghost+valid cells; error is written
    // only for valid cells (err_scratch has no ghost), guarded by a box check.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real> const& error_x = err_scratch[0].array(mfi);
        Array4<Real> const& error_y = err_scratch[1].array(mfi);
        Array4<Real> const& error_z = err_scratch[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect());
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect());
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect());
        Box const& tjx_ng = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy_ng = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz_ng = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx_ng, tjy_ng, tjz_ng,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kx(i, j, k, 0);
                amrex::Real const k2 = Kx(i, j, k, 1);
                amrex::Real const k3 = Kx(i, j, k, 2);
                amrex::Real const k4 = Kx(i, j, k, 3);
                amrex::Real const k5 = Kx(i, j, k, 4);
                amrex::Real const k6 = Bx(i, j, k) - Bx_old(i, j, k)
                                        - a61*k1 - a62*k2 - a63*k3 - a64*k4 - a65*k5;
                Kx(i, j, k, 1) = k6;
                Bx(i, j, k) = Bx_old(i, j, k) + b1*k1 + b3*k3 + b4*k4 + b5*k5;
                if (tjx.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
                    error_x(i, j, k) = e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6;
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Ky(i, j, k, 0);
                amrex::Real const k2 = Ky(i, j, k, 1);
                amrex::Real const k3 = Ky(i, j, k, 2);
                amrex::Real const k4 = Ky(i, j, k, 3);
                amrex::Real const k5 = Ky(i, j, k, 4);
                amrex::Real const k6 = By(i, j, k) - By_old(i, j, k)
                                        - a61*k1 - a62*k2 - a63*k3 - a64*k4 - a65*k5;
                Ky(i, j, k, 1) = k6;
                By(i, j, k) = By_old(i, j, k) + b1*k1 + b3*k3 + b4*k4 + b5*k5;
                if (tjy.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
                    error_y(i, j, k) = e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6;
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kz(i, j, k, 0);
                amrex::Real const k2 = Kz(i, j, k, 1);
                amrex::Real const k3 = Kz(i, j, k, 2);
                amrex::Real const k4 = Kz(i, j, k, 3);
                amrex::Real const k5 = Kz(i, j, k, 4);
                amrex::Real const k6 = Bz(i, j, k) - Bz_old(i, j, k)
                                        - a61*k1 - a62*k2 - a63*k3 - a64*k4 - a65*k5;
                Kz(i, j, k, 1) = k6;
                Bz(i, j, k) = Bz_old(i, j, k) + b1*k1 + b3*k3 + b4*k4 + b5*k5;
                if (tjz.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
                    error_z(i, j, k) = e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6;
                }
            }
        );
    }

    // ---- Error norm and adaptive step control ----
    // Compute local maxima first, then one combined AllReduce for both norms.
    amrex::Real err_norm = 0._rt;
    amrex::Real B4_norm  = 0._rt;
    for (int ii = 0; ii < 3; ii++) {
        err_norm = std::max(err_norm, err_scratch[ii].norm0(/*comp=*/0, /*nghost=*/0, /*local=*/true));
        B4_norm  = std::max(B4_norm,  Bfield[lev][ii]->norm0(/*comp=*/0, /*nghost=*/0, /*local=*/true));
    }
    amrex::ParallelDescriptor::ReduceRealMax({err_norm, B4_norm});
    return err_norm / (m_substep_atol + m_substep_rtol * B4_norm);
}


void HybridPICModel::FieldPush (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    amrex::Real dt, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    auto& warpx = WarpX::GetInstance();

    amrex::Real const t_old = warpx.gett_old(0);

    // Calculate J = curl x B / mu0 - J_ext
    CalculatePlasmaCurrent(Bfield, eb_update_E);
    // Calculate the E-field from Ohm's law
    HybridPICSolveE(Efield, Jfield, Bfield, rhofield, eb_update_E, true);
    // Call FillBoundary if a collocated grid is used
    if (Bz_IndexType[0] == Ez_IndexType[0]) {
        warpx.FillBoundaryE(ng, nodal_sync);
    }

    // Push forward the B-field using Faraday's law
    warpx.EvolveB(dt, subcycling_half, t_old);
    warpx.FillBoundaryB(ng, nodal_sync);
}
