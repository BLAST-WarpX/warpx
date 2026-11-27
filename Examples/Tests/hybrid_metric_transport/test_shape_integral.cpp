/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Particles/Deposition/ShapeIntegral.H"
#include "Particles/ShapeFactors.H"

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>

#include <cmath>
#include <limits>

namespace
{
    template <int order>
    void check ()
    {
        amrex::Gpu::DeviceVector<double> errors(32, 0.0);
        auto* const result = errors.data();
        amrex::ParallelFor(16, [=] AMREX_GPU_DEVICE (int sample) {
            constexpr int size = order+3;
            // Same-cell displacements towards a shape's right support edge.
            // Tiny polynomial tails must survive even when the interior
            // shape differences cancel to machine precision.
            double const delta = std::pow(10.0, -2.0-double(sample % 6));
            double const center = order % 2 == 0 ? 10.5 : 10.0;
            double old_shape[size] = {};
            double new_shape[size] = {};
            Compute_shape_factor<order>{}(old_shape+1, center+delta);
            Compute_shape_factor<order>{}(new_shape+1, center+2.0*delta);
            if (sample >= 8) {
                for (int index = 0; index < size; ++index) {
                    double const saved = old_shape[index];
                    old_shape[index] = new_shape[index];
                    new_shape[index] = saved;
                }
            }
            double const tail = new_shape[order+1]-old_shape[order+1];
            double const flux = warpx::deposition::shapeIntegral(
                old_shape, new_shape, order);
            double legacy = 0.0;
            for (int index = 0; index <= order; ++index) {
                legacy += old_shape[index]-new_shape[index];
            }
            result[sample+16] = std::abs((legacy-tail)/tail);
            double worst = std::abs((flux-tail)/tail);
            double previous = 0.0;
            for (int face = 0; face < size; ++face) {
                double const current = warpx::deposition::shapeIntegral(
                    old_shape, new_shape, face);
                double const residual = current-previous
                    -(old_shape[face]-new_shape[face]);
                worst = amrex::max(worst, std::abs(residual));
                previous = current;
            }
            result[sample] = worst;
        });
        amrex::Gpu::HostVector<double> host(32);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, errors.begin(), errors.end(), host.begin());
        double worst_legacy = 0.0;
        for (int sample = 0; sample < 16; ++sample) {
            double const error = host[sample];
            AMREX_ALWAYS_ASSERT(std::isfinite(error)
                && error <= 16.0*std::numeric_limits<double>::epsilon());
            worst_legacy = amrex::max(worst_legacy, host[sample+16]);
        }
        if constexpr (order >= 3) {
            AMREX_ALWAYS_ASSERT(worst_legacy > 0.01);
        }
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        check<1>(); check<2>(); check<3>(); check<4>();
    }
    amrex::Finalize();
}
