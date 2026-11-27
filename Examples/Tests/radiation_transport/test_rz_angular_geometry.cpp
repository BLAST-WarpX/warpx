/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include <Radiation/RZAngularGeometry.H>

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>

#include <algorithm>
#include <cmath>
#include <limits>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        // Both exactly representable and non-binary spacings. Signed shear
        // stresses deliberately distinguish theta momentum from true torque.
        for (int const cells : {32, 37}) {
            amrex::Gpu::DeviceVector<amrex::Real> output(6 * cells);
            auto* values = output.data();
            amrex::Real const spacing = amrex::Real(1) / cells;
            amrex::ParallelFor(cells, [=] AMREX_GPU_DEVICE(int i) {
                auto const lower = i * spacing;
                auto const upper = (i + 1) * spacing;
                // Unit common factor 2*pi*dz: the metric identity is unchanged.
                warpx::radiation::RZAngularGeometry metric(lower, upper, lower, upper);
                auto const low_flux = 1 + lower - 3 * lower * lower;
                auto const high_flux = 1 + upper - 3 * upper * upper;
                auto const divergence = metric.Divergence(low_flux, high_flux, amrex::Real(0.125));
                auto const geometric = metric.Geometric(low_flux, high_flux, amrex::Real(0.125));
                auto const ordinary = amrex::Real(0.125) * (upper * high_flux - lower * low_flux);
                auto const beta = amrex::Real(0.001) * metric.mean_radius;
                // Here the energy face flux is the radius, independent of shear.
                auto const energy = amrex::Real(0.125) * (upper * upper - lower * lower);
                auto const projected = energy - beta * ordinary + beta * geometric;
                values[6 * i] = metric.mean_radius;
                values[6 * i + 1] = divergence;
                values[6 * i + 2] = geometric;
                values[6 * i + 3] = metric.Magnitude(low_flux, high_flux, amrex::Real(0.125));
                values[6 * i + 4] = ordinary - geometric - divergence;
                values[6 * i + 5] = projected - (energy - beta * divergence);
            });
            amrex::Gpu::HostVector<amrex::Real> host(output.size());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, output.begin(), output.end(), host.begin());
            long double angular = 0, ordinary = 0, magnitude = 0;
            auto const eps = std::numeric_limits<amrex::Real>::epsilon();
            for (int i = 0; i < cells; ++i) {
                long double const lower = static_cast<long double>(i * spacing);
                long double const upper = static_cast<long double>((i + 1) * spacing);
                auto const exact_mean = (2.L / 3) *
                    (upper * upper + upper * lower + lower * lower) / (upper + lower);
                AMREX_ALWAYS_ASSERT(std::abs(host[6 * i] - exact_mean) <= 16 * eps * exact_mean);
                angular += static_cast<long double>(host[6 * i]) * host[6 * i + 1];
                ordinary += host[6 * i + 1];
                magnitude += std::abs(static_cast<long double>(host[6 * i])) * host[6 * i + 3];
                AMREX_ALWAYS_ASSERT(std::abs(host[6 * i + 4]) <=
                    128 * eps * std::max(host[6 * i + 3], amrex::Real(0.125) * spacing));
                AMREX_ALWAYS_ASSERT(std::abs(host[6 * i + 5]) <= 128 * eps * spacing);
            }
            // Outer boundary torque: dt*r^2*(1+r-3*r^2) = -1/8.
            AMREX_ALWAYS_ASSERT(std::abs(angular + 0.125L) <= 128 * eps * magnitude);
            AMREX_ALWAYS_ASSERT(std::abs(ordinary - angular) > 0.01L);
        }
    }
    amrex::Finalize();
}
