/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_GROUPMOMENTGEOMETRY_H_
#define WARPX_RADIATION_GROUPMOMENTGEOMETRY_H_

#include "CellVolume.H"
#include "RZAngularGeometry.H"

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

namespace warpx::radiation
{
/** Physical FV measures for LAB bin moments. RZ uses annular volumes and
 * angular-conjugate theta rows; no global radial vector invariant is implied.
 * Face currents are physical (energy,c*momentum) densities [W/m^2]. */
struct GroupMomentMetric
{
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_spacing, m_lower;
    amrex::Dim3 m_lo, m_hi;
    amrex::GpuArray<int, AMREX_SPACEDIM> m_periodic{};

    explicit GroupMomentMetric (amrex::Geometry const &geometry)
        : m_spacing(geometry.CellSizeArray()), m_lower(geometry.ProbLoArray()),
          m_lo(amrex::lbound(geometry.Domain())), m_hi(amrex::ubound(geometry.Domain()))
    {
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            m_periodic[d] = geometry.isPeriodic(d);
        }
    }
    AMREX_GPU_HOST_DEVICE amrex::Real Volume (int i) const
    {
        return CellVolume(i, m_spacing, m_lower, m_lo);
    }
    AMREX_GPU_HOST_DEVICE amrex::Real Area (int axis, int i) const
    {
#if defined(WARPX_DIM_RZ)
        if (axis == 0)
        {
            return 2 * MathConst::pi * Radius(i) * m_spacing[1];
        }
#endif
        return Volume(i) / m_spacing[axis];
    }
    AMREX_GPU_HOST_DEVICE amrex::Real Radius (int face) const
    {
        return m_lower[0] + (face - m_lo.x) * m_spacing[0];
    }
    AMREX_GPU_HOST_DEVICE RZAngularGeometry Angular (int i) const
    {
        return {Radius(i), Radius(i + 1), Area(0, i), Area(0, i + 1)};
    }
    AMREX_GPU_HOST_DEVICE bool Wall (int axis, int i, int j, int k) const
    {
#if defined(WARPX_DIM_RZ)
        int const index[3]{i, j, k}, low[3]{m_lo.x, m_lo.y, m_lo.z},
            high[3]{m_hi.x, m_hi.y, m_hi.z};
        return !m_periodic[axis] && (index[axis] == low[axis] || index[axis] == high[axis] + 1);
#else
        amrex::ignore_unused(axis, i, j, k);
        return false;
#endif
    }
    /** The same balanced isotropic-pressure gradient as the momentum row.
     * Scalar density ghosts are physical mirrors, never integrated energies
     * divided by a fictitious negative-radius volume. */
    AMREX_GPU_HOST_DEVICE amrex::Real Gradient (amrex::Array4<amrex::Real const> const &density,
                                                int component, int i, int j, int k, int axis) const
    {
        int left[3]{i, j, k}, right[3]{i, j, k};
        --left[axis];
        ++right[axis];
#if defined(WARPX_DIM_RZ)
        if (axis == 0)
        {
            return (Area(0, i + 1) * (density(right[0], right[1], right[2], component) -
                                      density(i, j, k, component)) +
                    Area(0, i) * (density(i, j, k, component) -
                                  density(left[0], left[1], left[2], component))) /
                   (2 * Volume(i));
        }
#endif
        return (density(right[0], right[1], right[2], component) -
                density(left[0], left[1], left[2], component)) /
               (2 * m_spacing[axis]);
    }
    /** Conservative LAB increment and independent uncancelled face/geometry
     * operand norm. For RZ row1 this subtracts hoop stress before area scaling,
     * so a constant isotropic pressure gives bit-exact zero. Row2 conserves
     * mean_radius*Qtheta; its ordinary component sum is not an invariant. */
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, 2>
    Increment (amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> const &flux,
               int component, int i, int j, int k, amrex::Real dt,
               amrex::Real theta_pressure = 0) const
    {
        amrex::GpuArray<amrex::Real, 2> result{};
        auto const row = component % 4;
        for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
        {
            int high[3]{i, j, k};
            ++high[axis];
            auto const low_flux = flux[axis](i, j, k, component);
            auto const high_flux = flux[axis](high[0], high[1], high[2], component);
            auto const low_area = Area(axis, i), high_area = Area(axis, high[0]);
#if defined(WARPX_DIM_RZ)
            if (axis == 0 && row == 1)
            {
                auto const hoop = PhysConst::c * theta_pressure;
                result[0] += dt * (high_area * (high_flux - hoop) - low_area * (low_flux - hoop));
                result[1] += std::abs(dt) *
                             (std::abs(high_area * high_flux) + std::abs(low_area * low_flux) +
                              std::abs((high_area - low_area) * hoop));
                continue;
            }
            if (axis == 0 && row == 2)
            {
                auto const angular = Angular(i);
                result[0] += angular.Divergence(low_flux, high_flux, dt);
                result[1] += angular.Magnitude(low_flux, high_flux, dt);
                continue;
            }
#else
            amrex::ignore_unused(row, theta_pressure);
#endif
            // Equal-area directions subtract currents first: no asymmetric
            // fused multiply/subtract of equal pressure products.
            result[0] += axis == 0 ? dt * (high_area * high_flux - low_area * low_flux)
                                   : dt * low_area * (high_flux - low_flux);
            result[1] +=
                std::abs(dt) * (std::abs(high_area * high_flux) + std::abs(low_area * low_flux));
        }
        return result;
    }
};

enum class GroupGhostKind
{
    Scalar,
    FourMoments,
    SymmetricPressure
};

/** Caller-private physical density/opacity ghosts. FourMoments and symmetric
 * pressure have one independent block per group. Axis parity flips r/theta;
 * radial mirror parity flips only r. No caller input or spectral edge is
 * modified, and no photon half-space pressure rule is implied. */
inline void FillGroupMomentDensityGhosts (amrex::MultiFab &field, amrex::Geometry const &geometry,
                                          int groups, GroupGhostKind kind)
{
    field.FillBoundary(geometry.periodicity());
#if defined(WARPX_DIM_RZ)
    int const block = kind == GroupGhostKind::Scalar        ? 1
                      : kind == GroupGhostKind::FourMoments ? 4
                                                            : 6;
    AMREX_ALWAYS_ASSERT(field.nComp() == block * groups && geometry.ProbLo(0) == 0 &&
                        !geometry.isPeriodic(0) && geometry.isPeriodic(1));
    auto const low = geometry.Domain().smallEnd(0), high = geometry.Domain().bigEnd(0);
    for (amrex::MFIter it(field); it.isValid(); ++it)
    {
        auto const values = field.array(it);
        amrex::ParallelFor(it.fabbox(), field.nComp(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k, int component)
                           {
                               if (i >= low && i <= high)
                               {
                                   return;
                               }
                               auto const from = i < low ? 2 * low - i - 1 : 2 * high - i + 1;
                               auto const within = component % block;
                               int sign = 1;
                               if (kind == GroupGhostKind::FourMoments)
                               {
                                   sign = within == 1 || (i < low && within == 2) ? -1 : 1;
                               }
                               else if (kind == GroupGhostKind::SymmetricPressure)
                               {
                                   // xx,yy,zz,xy,xz,yz in (r,theta,z).
                                   sign = i < low ? (within == 4 || within == 5 ? -1 : 1)
                                                  : (within == 3 || within == 4 ? -1 : 1);
                               }
                               values(i, j, k, component) = sign * values(from, j, k, component);
                           });
    }
#else
    amrex::ignore_unused(groups, kind);
#endif
}
} // namespace warpx::radiation
#endif
