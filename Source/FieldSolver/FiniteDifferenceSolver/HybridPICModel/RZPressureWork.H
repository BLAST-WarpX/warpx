/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace warpx::hybrid
{
using namespace amrex::literals;

/** Opt-in collocated m=0 pressure gradient. The axis is not a material wall:
 * radial pressure force vanishes there, but axial force need not. Stationary
 * PEC faces have zero tangential E and an even scalar pressure extension,
 * hence zero normal pressure gradient. Preserve the local wall-node EOS P.
 *
 * This centered nodal stencil is deliberately distinct from the legacy RZ
 * forward Yee difference. Only the conservative collocated path selects it.
 */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
rzPressureGradient (amrex::Array4<amrex::Real const> const& pressure, int const i, int const j,
                    int const component, int const axis, int const outer, int const zlo,
                    int const zhi, bool const periodic_z,
                    amrex::Real const inverse_spacing) noexcept
{
    if (component == 1 || i == outer || (!periodic_z && (j == zlo || j == zhi)) ||
        (component == 0 && i == axis))
    {
        return 0.0_rt;
    }
    return component == 0
               ? 0.5_rt * inverse_spacing * (pressure(i + 1, j, 0) - pressure(i - 1, j, 0))
               : 0.5_rt * inverse_spacing * (pressure(i, j + 1, 0) - pressure(i, j - 1, 0));
}

/** -W^-1 G^T rho_inverse C^T J_work. Currents here are integrated charge-work
 * velocities [C m/s], NOT inverse-volume-scaled physical current densities.
 * C^T (axis/PEC ghost folding and MPI sums) has already been applied. Divide
 * by the native electron caloric node volume exactly once. The electron
 * energy-density increment is -dt * P_old * this return value.
 */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
rzPressureWorkDivergence (amrex::Array4<amrex::Real const> const& radial_current,
                          amrex::Array4<amrex::Real const> const& axial_current,
                          amrex::Array4<amrex::Real const> const& state, int const i, int const j,
                          int const axis, int const outer, int const zlo, int const zhi,
                          bool const periodic_z, amrex::Real const inverse_dr,
                          amrex::Real const inverse_dz, amrex::Real const node_volume) noexcept
{
    auto const value =
        [=] AMREX_GPU_HOST_DEVICE(int const r, int const z, bool const radial) noexcept
    {
        if (r < axis || r >= outer || (!periodic_z && (z <= zlo || z >= zhi)) ||
            (radial && r == axis))
        {
            return 0.0_rt;
        }
        return (radial ? radial_current(r, z, 0) : axial_current(r, z, 0)) * state(r, z, 0, 1);
    };
    return (0.5_rt * inverse_dr * (value(i + 1, j, true) - value(i - 1, j, true)) +
            0.5_rt * inverse_dz * (value(i, j + 1, false) - value(i, j - 1, false))) /
           node_volume;
}
} // namespace warpx::hybrid
