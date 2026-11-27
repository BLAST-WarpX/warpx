/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/OpacityTable.H"

#include <AMReX.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuMemory.H>

#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        using namespace warpx::radiation;
        std::ofstream("gray.txt") << "2 2\n1 2\n1 2\n1e18 1e18 1e18 1\n";
        std::ofstream("spectral.txt") << "2 2 2\n1 2\n1 2\n1 2\n"
                                      << "1e18 1e18 1e18 1e18 1e18 1e18 1e18 1\n";
        OpacityTable2D gray;
        OpacityTable3D spectral;
        gray.load("gray.txt", OpacityTableInterpolation::Linear);
        spectral.load("spectral.txt", OpacityTableInterpolation::Linear);
        auto const two = gray.executor();
        auto const three = spectral.executor();
        amrex::Gpu::DeviceScalar<int> passed(0);
        auto* result = passed.dataPtr();
        amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) {
            *result = two(2, 2) == 1 && two(3, 3) == 1 &&
                three(2, 2, 2) == 1 && three(3, 3, 3) == 1;
        });
        AMREX_ALWAYS_ASSERT(passed.dataValue() == 1);

        amrex::Real const low = 1000000;
        auto const high = std::nextafter(low, std::numeric_limits<amrex::Real>::infinity());
        AMREX_ALWAYS_ASSERT(high > low && std::log(high) == std::log(low));
        std::ofstream("collapsed.txt") << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
            << "2 2\n" << low << ' ' << high << "\n1 2\n2 2 4 4\n";
        bool rejected = false;
        try {
            OpacityTable2D invalid;
            invalid.load("collapsed.txt", OpacityTableInterpolation::LogLog);
        } catch (std::exception const& error) {
            rejected = std::string(error.what()).find("strictly increasing") != std::string::npos;
        }
        AMREX_ALWAYS_ASSERT(rejected);
    }
    amrex::Finalize();
}
