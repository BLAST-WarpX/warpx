/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ChargeContinuityRoundoff.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>

#include <limits>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        using amrex::Real;
        Real const nan = std::numeric_limits<Real>::quiet_NaN();
        Real const inf = std::numeric_limits<Real>::infinity();
        // Actual vacuum residual, injected mismatches, and invalid states.
        amrex::GpuArray<Real, 9> const errors{
            0, 1.1802034918686064e-28, 1.e-20, 2.8162080018985854e-16,
            nan, inf, 1.e-28, -1, 1.e-28};
        amrex::GpuArray<Real, 9> const transfers{
            0, 4.6963154061411356e-12, 4.6963154061411356e-12, 3.421402841865532e-10,
            1, 1, inf, 1, -1};
        amrex::Gpu::DeviceVector<int> results(27, 0);
        auto* const result = results.data();
        amrex::ParallelFor(27, [=] AMREX_GPU_DEVICE (int index) {
            int const sample = index % 9;
            Real const scale = index < 9 ? Real(1.e-50)
                : (index < 18 ? Real(1) : Real(1.e50));
            bool const accepted = warpx::hybrid::chargeFluxRoundoffConsistent(
                errors[sample] * scale, transfers[sample] * scale);
            result[index] = accepted == (sample < 2);
        });
        amrex::Gpu::HostVector<int> host(27);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, results.begin(), results.end(), host.begin());
        for (int const passed : host) { AMREX_ALWAYS_ASSERT(passed == 1); }
    }
    amrex::Finalize();
}
