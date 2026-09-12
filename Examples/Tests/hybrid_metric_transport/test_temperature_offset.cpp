/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Particles/Deposition/TemperatureDeposition.H"

#include <AMReX_FArrayBox.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <limits>

using namespace amrex::literals;

template<int order>
void check_offset ()
{
    // Storage includes both the intended support and the legacy wrong offset,
    // so the red test reports a misplaced deposit without an out-of-bounds write.
    amrex::Box const storage(amrex::IntVect::TheZeroVector(), amrex::IntVect(31));
    amrex::FArrayBox moments(storage, 9);
    amrex::IArrayBox counts(storage, 3);
    moments.setVal(0);
    counts.setVal(0);
    auto const wx = moments.array(0), wy = moments.array(1), wz = moments.array(2);
    auto const w2x = moments.array(3), w2y = moments.array(4), w2z = moments.array(5);
    auto const vx = moments.array(6), vy = moments.array(7), vz = moments.array(8);
    auto const nx = counts.array(0), ny = counts.array(1), nz = counts.array(2);
    amrex::IntVect const nodal(1);
    amrex::Dim3 const lo = amrex::lbound(amrex::Box(
        amrex::IntVect(AMREX_D_DECL(7, 11, 13)), amrex::IntVect(31)));
    amrex::For(1, [=] AMREX_GPU_DEVICE (int) {
        warpx::particles::deposition::doVarianceDepositionShapeNKernel<order>(
            5.25_prt, 0.0_prt,
#if defined(WARPX_DIM_RSPHERE)
            0.0_prt,
#else
            5.25_prt,
#endif
            4.0_prt, 1.0_prt, 2.0_prt, 3.0_prt,
            nx, ny, nz, wx, wy, wz, w2x, w2y, w2z, vx, vy, vz,
            nodal, nodal, nodal,
            warpx::particles::deposition::TemperatureDepositionType::DOUBLE_PASS,
            warpx::particles::deposition::TemperatureDepositionPass::FIRST,
            0.0_rt, amrex::XDim3{1,1,1}, amrex::XDim3{0,0,0}, lo, 1);
    });
    amrex::Gpu::HostVector<amrex::Real> host(moments.size());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, moments.dataPtr(),
                     moments.dataPtr()+moments.size(), host.begin());
    constexpr int ny_cells = AMREX_SPACEDIM >= 2 ? 32 : 1;
    constexpr int nz_cells = AMREX_SPACEDIM == 3 ? 32 : 1;
    constexpr int cells = 32*ny_cells*nz_cells;
    amrex::Real const tolerance = 256*std::numeric_limits<amrex::Real>::epsilon();
    for (int component = 0; component < 3; ++component) {
        std::array<amrex::Real, 4> sum{};
        for (int k = 0; k < nz_cells; ++k) {
            for (int j = 0; j < ny_cells; ++j) {
                for (int i = 0; i < 32; ++i) {
                    auto const w = host[i+32*(j+ny_cells*k)+component*cells];
                    sum[0] += w;
                    sum[1] += i*w;
                    sum[2] += j*w;
                    sum[3] += k*w;
                }
            }
        }
        AMREX_ALWAYS_ASSERT(std::abs(sum[0]-4.0_rt) < tolerance*4);
        std::array<amrex::Real, 3> const expected{
#if defined(WARPX_DIM_3D)
            12.25_rt, 11.0_rt, 18.25_rt
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            12.25_rt, 16.25_rt, 0.0_rt
#else
            12.25_rt, 0.0_rt, 0.0_rt
#endif
        };
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            amrex::Print() << "Shape " << order << ", component " << component
                          << ", direction " << d << ": centroid=" << sum[d+1]/sum[0]
                          << ", expected=" << expected[d] << '\n';
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::abs(sum[d+1]/sum[0]-expected[d]) < tolerance*32,
                "Temperature deposition must use the stored mesh-coordinate offset.");
        }
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    check_offset<1>();
    check_offset<2>();
    check_offset<3>();
    check_offset<4>();
    amrex::Finalize();
}
