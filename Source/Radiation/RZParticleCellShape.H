/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_RZPARTICLECELLSHAPE_H_
#define WARPX_RADIATION_RZPARTICLECELLSHAPE_H_

#include "ParticleCellShape.H"
#include "RZMomentGeometry.H"

#if defined(WARPX_DIM_RZ)
namespace warpx::radiation
{
    /** Native nodal mass deposition, inverse-volume scaling, then
     * arithmetic four-corner averaging, expressed as source-cell mass weights.
     * M_c = sum_p m_p S_cp = V_c rho_c. These are NOT a partition of unity at
     * a radial wall: native boundary nodes retain their full deposit volume.
     * Source normalization, not renormalizing a particle's stencil, conserves
     * the requested impulse. Radial cells outside the physical domain have
     * no source; axial entries either communicate periodically through MF
     * ghosts or include the actual reflecting-endcap charge fold.
     */
    template<int Order>
    AMREX_GPU_HOST_DEVICE inline void
    InitializeRZNativeCellAverageOrder (
        ParticleCellShape<2>& shape, RZMomentMetric const& metric,
        int radial_hi, bool verboncoeur_axis,
        bool reflecting_radial_density = false,
        bool reflecting_axial_density = false, int axial_hi = 0) noexcept
    {
        static_assert(Order >= 1 && Order <= 3);
        shape.InitializeOrder<Order>();
        amrex::Real nodal[Order + 1]{};
        auto const first = shape.center[0] +
            Compute_shape_factor<Order>{}(nodal, shape.fraction[0]);
        auto const dr = metric.spacing[0], dz = metric.spacing[1];
        auto const axis_factor = verboncoeur_axis ? amrex::Real(1) / 3 : amrex::Real(1) / 4;
        for (int entry = 0; entry < shape.extent; ++entry)
        {
            int const cell = shape.low[0] + entry;
            amrex::Real inverse_volume_sum = 0;
            if (cell >= metric.domain_lo.x && cell <= radial_hi)
            {
                for (int side = 0; side < 2; ++side)
                {
                    int const node = cell + side;
                    int const offset = node - first;
                    amrex::Real deposited = offset >= 0 && offset <= Order ? nodal[offset] : 0;
                    // The m=0 native charge deposit folds negative-radius
                    // nodal contributions onto the matching positive node.
                    // The axis node itself must not be counted twice.
                    if (node > metric.domain_lo.x)
                    {
                        int const reflected = 2 * metric.domain_lo.x - node - first;
                        if (reflected >= 0 && reflected <= Order) {
                            deposited += nodal[reflected];
                        }
                    }
                    if (reflecting_radial_density)
                    {
                        // Live hybrid rho reflects its nodal guard deposit at
                        // the material wall, including the boundary node itself.
                        // The native radial volume ratio cancels the ghost's
                        // inverse-volume scale, leaving a nodal shape sum here.
                        // This is charge-deposit folding, NOT a normalization or
                        // reflection of a requested force into a fictitious cell.
                        int const wall = radial_hi + 1;
                        if (node == wall)
                        {
                            deposited *= 2;
                        }
                        else
                        {
                            int const reflected = 2 * wall - node - first;
                            if (reflected >= 0 && reflected <= Order)
                            {
                                deposited += nodal[reflected];
                            }
                        }
                    }
                    auto const radius = metric.Radius(node);
                    auto const volume = radius == 0
                        ? MathConst::pi * dr * dr * axis_factor * dz
                        : 2 * MathConst::pi * radius * dr * dz;
                    inverse_volume_sum += deposited / volume;
                }
                shape.weights[0][entry] = metric.Volume(cell) * inverse_volume_sum / 2;
            }
            else
            {
                shape.weights[0][entry] = 0;
            }
        }
        if (reflecting_axial_density)
        {
            amrex::Real axial_nodal[Order + 1]{};
            int const axial_first = shape.center[1] +
                Compute_shape_factor<Order>{}(axial_nodal, shape.fraction[1]);
            int const lo = metric.domain_lo.y, wall = axial_hi + 1;
            for (int entry = 0; entry < shape.extent; ++entry)
            {
                int const cell = shape.low[1] + entry;
                amrex::Real deposited_sum = 0;
                if (cell >= lo && cell <= axial_hi)
                {
                    for (int side = 0; side < 2; ++side)
                    {
                        int const node = cell + side, offset = node - axial_first;
                        amrex::Real deposited = offset >= 0 && offset <= Order
                            ? axial_nodal[offset] : 0;
                        // Positive material reflection doubles the boundary
                        // node itself. At a radial/endcap corner the separable
                        // folds therefore reproduce the live fourfold deposit.
                        if (node == lo || node == wall)
                        {
                            deposited *= 2;
                        }
                        else
                        {
                            for (int endcap = 0; endcap < 2; ++endcap)
                            {
                                int const boundary = endcap == 0 ? lo : wall;
                                int const reflected = 2 * boundary - node - axial_first;
                                if (reflected >= 0 && reflected <= Order)
                                {
                                    deposited += axial_nodal[reflected];
                                }
                            }
                        }
                        deposited_sum += deposited;
                    }
                }
                // No reflected force or renormalization of the native shape:
                // source normalization uses the actual receiving-cell mass.
                shape.weights[1][entry] = deposited_sum / 2;
            }
        }
    }

    AMREX_GPU_HOST_DEVICE inline void
    InitializeRZNativeCellAverage (
        ParticleCellShape<2>& shape, RZMomentMetric const& metric,
        int radial_hi, bool verboncoeur_axis, int order,
        bool reflecting_radial_density = false,
        bool reflecting_axial_density = false, int axial_hi = 0) noexcept
    {
        AMREX_ASSERT(order >= 1 && order <= 3);
        if (order == 1) {
            InitializeRZNativeCellAverageOrder<1>(shape, metric, radial_hi, verboncoeur_axis,
                                                 reflecting_radial_density,
                                                 reflecting_axial_density, axial_hi);
        }
        else if (order == 2) {
            InitializeRZNativeCellAverageOrder<2>(shape, metric, radial_hi, verboncoeur_axis,
                                                 reflecting_radial_density,
                                                 reflecting_axial_density, axial_hi);
        }
        else {
            InitializeRZNativeCellAverageOrder<3>(shape, metric, radial_hi, verboncoeur_axis,
                                                 reflecting_radial_density,
                                                 reflecting_axial_density, axial_hi);
        }
    }
}
#endif
#endif
