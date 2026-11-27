/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/CartesianPhotonBoundary.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_REAL.H>

#include <iostream>

using warpx::radiation::ClassifyCartesianPhotonDomainCrossing;
using warpx::radiation::PhotonFacePolicy;
using warpx::radiation::SpecularPhotonCoordinate;
using namespace amrex::literals;

namespace
{
    constexpr int kAbsorbing = static_cast<int>(PhotonFacePolicy::Absorbing);
    constexpr int kReflecting = static_cast<int>(PhotonFacePolicy::Reflecting);
    constexpr int kPeriodic = static_cast<int>(PhotonFacePolicy::Periodic);

    bool CheckCrossing (
        char const* const label,
        amrex::GpuArray<int, 3> const& leaving,
        amrex::GpuArray<int, 3> const& lo,
        amrex::GpuArray<int, 3> const& hi,
        bool const expect_escaped,
        bool const expect_reflected,
        amrex::GpuArray<int, 3> const& expect_reverse)
    {
        auto const result = ClassifyCartesianPhotonDomainCrossing(leaving, lo, hi);
        bool const pass = result.escaped == expect_escaped
            && result.reflected == expect_reflected
            && result.reverse[0] == expect_reverse[0]
            && result.reverse[1] == expect_reverse[1]
            && result.reverse[2] == expect_reverse[2];
        if (!pass) {
            std::cout << "FAIL " << label
                      << " escaped=" << result.escaped
                      << " reflected=" << result.reflected
                      << " reverse=" << result.reverse[0] << ','
                      << result.reverse[1] << ',' << result.reverse[2] << '\n';
        }
        return pass;
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    bool pass = true;
    {
        amrex::GpuArray<int, 3> mixed_lo{kAbsorbing, kAbsorbing, kReflecting};
        amrex::GpuArray<int, 3> mixed_hi = mixed_lo;
#if defined(WARPX_DIM_1D_Z)
        // 1D z is index 0. Application analog: absorbing lo, reflecting hi.
        mixed_lo = {kAbsorbing, kAbsorbing, kAbsorbing};
        mixed_hi = {kReflecting, kReflecting, kReflecting};
#elif defined(WARPX_DIM_XZ)
        // XZ stored as (x,z). Open x, reflecting z.
        mixed_lo = {kAbsorbing, kReflecting, kReflecting};
        mixed_hi = mixed_lo;
#endif
        amrex::GpuArray<int, 3> const periodic_all{kPeriodic, kPeriodic, kPeriodic};

        pass = CheckCrossing(
            "low_face_open", {-1, 0, 0}, mixed_lo, mixed_hi, true, false,
            {0, 0, 0}) && pass;
#if defined(WARPX_DIM_1D_Z)
        pass = CheckCrossing(
            "z_hi_reflect", {1, 0, 0}, mixed_lo, mixed_hi, false, true,
            {1, 0, 0}) && pass;
#elif defined(WARPX_DIM_XZ)
        pass = CheckCrossing(
            "x_hi_open", {1, 0, 0}, mixed_lo, mixed_hi, true, false,
            {0, 0, 0}) && pass;
        pass = CheckCrossing(
            "z_hi_reflect", {0, 1, 0}, mixed_lo, mixed_hi, false, true,
            {0, 1, 0}) && pass;
        pass = CheckCrossing(
            "xz_corner_open_owns", {1, 1, 0}, mixed_lo, mixed_hi,
            true, false, {0, 0, 0}) && pass;
#else
        pass = CheckCrossing(
            "x_hi_open", {1, 0, 0}, mixed_lo, mixed_hi, true, false,
            {0, 0, 0}) && pass;
        pass = CheckCrossing(
            "y_hi_open", {0, 1, 0}, mixed_lo, mixed_hi, true, false,
            {0, 0, 0}) && pass;
        pass = CheckCrossing(
            "z_hi_reflect", {0, 0, 1}, mixed_lo, mixed_hi, false, true,
            {0, 0, 1}) && pass;
        pass = CheckCrossing(
            "xz_corner_open_owns", {1, 0, 1}, mixed_lo, mixed_hi,
            true, false, {0, 0, 0}) && pass;
#endif
        pass = CheckCrossing(
            "periodic_ignored", {1, 0, 0}, periodic_all, periodic_all,
            false, false, {0, 0, 0}) && pass;

        // Independent specular identity: 2*wall - x.
        amrex::ParticleReal const folded = SpecularPhotonCoordinate(1.25_prt, 1.0_rt);
        pass = (folded == 0.75_prt) && pass;
        if (folded != 0.75_prt) {
            std::cout << "FAIL specular fold " << folded << '\n';
        }

#ifdef AMREX_USE_GPU
        amrex::Gpu::DeviceScalar<int> gpu_fail(0);
        int* const gpu_fail_ptr = gpu_fail.dataPtr();
        amrex::GpuArray<int, 3> const device_lo = mixed_lo;
        amrex::GpuArray<int, 3> const device_hi = mixed_hi;
        amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) noexcept
        {
            auto const reflected = ClassifyCartesianPhotonDomainCrossing(
#if defined(WARPX_DIM_1D_Z)
                {1, 0, 0},
#elif defined(WARPX_DIM_XZ)
                {0, 1, 0},
#else
                {0, 0, 1},
#endif
                device_lo, device_hi);
            if (reflected.escaped || !reflected.reflected) {
                *gpu_fail_ptr = 1;
            }
            auto const corner = ClassifyCartesianPhotonDomainCrossing(
#if defined(WARPX_DIM_1D_Z)
                {-1, 0, 0},
#elif defined(WARPX_DIM_XZ)
                {1, 1, 0},
#else
                {1, 0, 1},
#endif
                device_lo, device_hi);
            if (!corner.escaped || corner.reflected) {
                *gpu_fail_ptr = 1;
            }
            if (SpecularPhotonCoordinate(1.25_prt, 1.0_rt) != 0.75_prt) {
                *gpu_fail_ptr = 1;
            }
        });
        amrex::Gpu::streamSynchronize();
        pass = (gpu_fail.dataValue() == 0) && pass;
#endif
    }
    std::cout << "cartesian photon boundary host/device semantics: "
              << (pass ? "PASS" : "FAIL") << '\n';
    amrex::Finalize();
    return pass ? 0 : 1;
}
