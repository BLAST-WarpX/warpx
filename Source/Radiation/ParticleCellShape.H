/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_PARTICLECELLSHAPE_H_
#define WARPX_RADIATION_PARTICLECELLSHAPE_H_

#include "Particles/ShapeFactors.H"

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace warpx::radiation
{
    /** Native B-spline nodal shape followed by arithmetic cell-corner averaging.
     * Interior/periodic Cartesian cells only. Fractions must lie in [0,1].
     * Coordinates are mesh coordinates (z in 1D, x/z in 2D).
     */
    template <int Dim> struct ParticleCellShape {
        static_assert(Dim >= 1 && Dim <= 3);
        int size = 0;
        int extent = 0;
        amrex::GpuArray<int, 3> center{};
        amrex::GpuArray<amrex::Real, 3> fraction{};
        amrex::GpuArray<int, 3> low{};
        amrex::GpuArray<amrex::GpuArray<amrex::Real, 6>, Dim> weights{};

        template<int Order>
        AMREX_GPU_HOST_DEVICE
        void InitializeOrder () noexcept
        {
            extent = Order + 2;
            size = 1;
            for (int d = 0; d < Dim; ++d) {
                amrex::Real nodal[Order + 1]{};
                auto const first = Compute_shape_factor<Order>{}(nodal, fraction[d]);
                low[d] = center[d] + first - 1;
                for (int k = 0; k < extent; ++k) {
                    weights[d][k] = ((k > 0 ? nodal[k - 1] : 0) +
                                     (k <= Order ? nodal[k] : 0)) / 2;
                }
                size *= extent;
            }
        }

        AMREX_GPU_HOST_DEVICE
        void Initialize (int order) noexcept
        {
            AMREX_ASSERT(order >= 1 && order <= 4);
            if (order == 1) { InitializeOrder<1>(); }
            else if (order == 2) { InitializeOrder<2>(); }
            else if (order == 3) { InitializeOrder<3>(); }
            else { InitializeOrder<4>(); }
        }

        [[nodiscard]] AMREX_GPU_HOST_DEVICE
        amrex::GpuArray<int, 3>
        Cell (int entry) const noexcept
        {
            auto result = low;
            for (int d = 0; d < Dim; ++d) {
                result[d] += entry % extent;
                entry /= extent;
            }
            return result;
        }

        [[nodiscard]] AMREX_GPU_HOST_DEVICE
        amrex::Real
        Weight (int entry) const noexcept
        {
            amrex::Real result = 1;
            for (int d = 0; d < Dim; ++d) {
                result *= weights[d][entry % extent];
                entry /= extent;
            }
            return result;
        }
    };
} // namespace warpx::radiation
#endif
