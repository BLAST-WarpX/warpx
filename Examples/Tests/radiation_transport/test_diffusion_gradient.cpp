/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/DiffusionGradient.H"

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>

using namespace amrex::literals;

struct ExponentialDensity
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    operator()(int i, int j, int k) const noexcept
    {
        return std::exp(0.001_rt * (i + j + k));
    }
};

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Gpu::DeviceScalar<amrex::GpuArray<amrex::Real, 2>> values;
        auto* output = values.dataPtr();
        amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int) noexcept {
            ExponentialDensity const density;
            amrex::GpuArray<amrex::Real, 3> const dx{0.001_rt, 0.001_rt, 0.001_rt};
            amrex::GpuArray<int, 3> const periodic{0, 0, 0};
            amrex::Real const normal = (density(1, 0, 0) - density(0, 0, 0)) / dx[0];
            amrex::Real const face = 0.5_rt * (density(1, 0, 0) + density(0, 0, 0));
            amrex::Real const full = warpx::radiation::FaceGradientMagnitude(
                density, 0, 0, 0, 0, 1, normal, dx, periodic, {-8, -8, -8}, {8, 8, 8});
            amrex::Real const rn = normal / (0.001_rt * face);
            amrex::Real const r = full / (0.001_rt * face);
            amrex::Real const lambda = (1 + 2 / r) / (r + 3 + 6 / r);
            amrex::Real const directional = (1 + 2 / rn) / (rn + 3 + 6 / rn);
            (*output)[0] = std::sqrt(3.0_rt) * lambda * rn;
            (*output)[1] = std::sqrt(3.0_rt) * directional * rn;
        });
        auto const result = values.dataValue();
        amrex::Real const eps = std::numeric_limits<amrex::Real>::epsilon();
        AMREX_ALWAYS_ASSERT(result[0] <= 1 + 64 * eps && result[0] > .95_rt);
        AMREX_ALWAYS_ASSERT(result[1] > 1.5_rt);
        amrex::Print() << "Diagonal |F|/(c E): full=" << result[0] << ", directional=" << result[1]
                       << '\n';
    }
    amrex::Finalize();
}
