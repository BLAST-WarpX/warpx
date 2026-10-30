/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SCALAR_PROBE_SAMPLING_H_
#define WARPX_SCALAR_PROBE_SAMPLING_H_

#include <AMReX_Array4.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

namespace warpx::diagnostics
{
    /** Gather a nodal scalar in the containing cell. The tensor-product linear
     * weights are the order-one particle shape used by FieldProbe. All stencil
     * nodes belong to the cell's valid nodal box, including on box boundaries. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real sampleNodalScalar (
        amrex::Array4<amrex::Real const> const& field, amrex::IntVect cell,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& fraction, int order)
    {
        if (order == 0) {
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (fraction[d] >= amrex::Real(0.5)) { ++cell[d]; }
            }
            return field(cell);
        }
        amrex::Real value = 0;
        for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner) {
            auto node = cell;
            amrex::Real weight = 1;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                int const high = (corner >> d) & 1;
                node[d] += high;
                weight *= high ? fraction[d] : amrex::Real(1) - fraction[d];
            }
            // Do not let an unused NaN corner contaminate a sample at a node.
            if (weight != amrex::Real(0)) { value += weight * field(node); }
        }
        return value;
    }
}
#endif
