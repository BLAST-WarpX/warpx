/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_RZMOMENTGEOMETRY_H_
#define WARPX_RADIATION_RZMOMENTGEOMETRY_H_

#include "CellVolume.H"
#include "FrameTransform.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#if defined(WARPX_DIM_RZ)
namespace warpx::radiation
{
/** Annular radiation-cell measures, distinct from nodal material volumes. */
struct RZMomentMetric
{
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> spacing, lower;
    amrex::Dim3 domain_lo;

    explicit RZMomentMetric (amrex::Geometry const& geometry)
        : spacing(geometry.CellSizeArray()), lower(geometry.ProbLoArray()),
          domain_lo(amrex::lbound(geometry.Domain()))
    {
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE amrex::Real
    Volume (int i) const noexcept
    {
        return CellVolume(i, spacing, lower, domain_lo);
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE amrex::Real
    Area (int direction, int i) const noexcept
    {
        if (direction == 0)
        {
            auto const radius = lower[0] + (i - domain_lo.x) * spacing[0];
            return 2 * MathConst::pi * radius * spacing[1];
        }
        return Volume(i) / spacing[1];
    }
};

/** Synchronize cell-centered scalars/vectors, then apply cylindrical axis
 * parity and stationary optical/material mirror parity. Vector components
 * start at offset (one for E,Q; zero for beta). Physical corners read only
 * synchronized interior/periodic values, never concurrently modified ghosts.
 */
void FillRZMomentGhosts (amrex::MultiFab& field, amrex::Geometry const& geometry,
                         int vector_offset);

/** Convert integrated radiation state to density before constructing fluxes. */
void PrepareRZMomentDensity (amrex::MultiFab& density, amrex::MultiFab const& integrated,
                             amrex::Geometry const& geometry);
void PrepareRZMomentFluxState (amrex::MultiFab& density, amrex::MultiFab& theta_pressure,
                               amrex::MultiFab const& integrated,
                               amrex::MultiFab const& pressure_jacobian,
                               amrex::Geometry const& geometry, bool physical);
[[nodiscard]] FourVector RZGeometricExchange (amrex::MultiFab const& theta_pressure,
                                              amrex::Geometry const& geometry, amrex::Real dt);

using RZMomentFaceArrays = amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM>;

/** Metric transport of (E-beta.Q,Q), with P_theta_theta in energy-density
 * units. Azimuthal flux is outside this meridional adapter's contract.
 * The radial stress difference is written in a well-balanced form.
 */
[[nodiscard]] AMREX_GPU_HOST_DEVICE inline FourVector
RZMomentDivergence (RZMomentFaceArrays const& flux, RZMomentFaceArrays const& parameters,
                    amrex::IntVect const& cell, amrex::GpuArray<amrex::Real, 3> const& beta,
                    RZMomentMetric const& metric, amrex::Real dt, amrex::Real theta_pressure)
{
    FourVector result{};
    auto const pressure_flux = PhysConst::c * theta_pressure;
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        auto upper = cell;
        ++upper[direction];
        auto const low_area = metric.Area(direction, cell[0]);
        auto const high_area = metric.Area(direction, upper[0]);
        auto low = flux[direction](cell, 4);
        auto high = flux[direction](upper, 4);
        for (int component = 0; component < 3; ++component)
        {
            low += (parameters[direction](cell, component) - beta[component]) *
                   flux[direction](cell, component + 1);
            high += (parameters[direction](upper, component) - beta[component]) *
                    flux[direction](upper, component + 1);
            auto const geometric = direction == 0 && component == 0 ? pressure_flux : 0;
            if (direction == 0)
            {
                result[component + 1] +=
                    dt * (high_area * (flux[direction](upper, component + 1) - geometric) -
                          low_area * (flux[direction](cell, component + 1) - geometric));
            }
            else
            {
                // Equal-area axial faces: subtract fluxes before scaling.
                // An asymmetric fused multiply/subtract of two identical
                // area*pressure products would inject spurious axial force.
                result[component + 1] +=
                    dt * low_area *
                    (flux[direction](upper, component + 1) - flux[direction](cell, component + 1));
            }
        }
        result[0] += direction == 0 ? dt * (high_area * high - low_area * low)
                                    : dt * low_area * (high - low);
    }
    result[0] +=
        beta[0] * dt * (metric.Area(0, cell[0] + 1) - metric.Area(0, cell[0])) * pressure_flux;
    return result;
}
} // namespace warpx::radiation
#endif
#endif
