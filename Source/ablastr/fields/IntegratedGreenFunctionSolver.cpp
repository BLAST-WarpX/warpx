/* Copyright 2019-2024 Arianna Formenti, Remi Lehe
 *
 * This file is part of ABLASTR.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "IntegratedGreenFunctionSolver.H"

#include <ablastr/constant.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_Array4.H>
#include <AMReX_BaseFab.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FabArray.H>
#include <AMReX_FFT.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

namespace ablastr::fields {

void
computePhiIGF ( amrex::MultiFab const & rho,
                amrex::MultiFab & phi,
                std::array<amrex::Real, 3> const & cell_size,
                bool const is_igf_2d_slices)
{
    using namespace amrex::literals;

    BL_PROFILE("ablastr::fields::computePhiIGF");

    // Define box that encompasses the valid (nodal) domain of the source.
    //   Note: we intentionally do NOT grow by phi.nGrowVect() here.
    //   Guard/ghost values of phi are filled by the caller later on,
    //   e.g., in FillBoundary.
    amrex::Box const domain = rho.boxArray().minimalBox();

    int nprocs = amrex::ParallelDescriptor::NProcs();
    {
        amrex::ParmParse pp("ablastr");
        pp.queryAdd("nprocs_igf_fft", nprocs);
        nprocs = std::max(1,std::min(nprocs, amrex::ParallelDescriptor::NProcs()));
    }

    static std::unique_ptr<amrex::FFT::OpenBCSolver<amrex::Real>> obc_solver;
    if (!obc_solver) {
        amrex::ExecOnFinalize([&] () { obc_solver.reset(); });
    }
    if (!obc_solver || obc_solver->Domain() != domain) {
        amrex::FFT::Info info{};
        if (is_igf_2d_slices) { info.setTwoDMode(true); } // do 2D FFTs
        info.setNumProcs(nprocs);
        obc_solver = std::make_unique<amrex::FFT::OpenBCSolver<amrex::Real>>(domain, info);
    }

    auto const& lo = domain.smallEnd();
    amrex::Real const dx = cell_size[0];
    amrex::Real const dy = cell_size[1];
    amrex::Real const dz = cell_size[2];

    if (!is_igf_2d_slices){
        // fully 3D solver
        obc_solver->setGreensFunction(
        [=] AMREX_GPU_DEVICE (auto i, int j, int k)
        {
            // i is a scalar int (GPU / non-SIMD CPU) or a SIMD int register
            // (CPU with SIMD); WIDTH is deduced accordingly (1 in the scalar case).
            constexpr int WIDTH = amrex::simd::lane_count_v<decltype(i)>;
            using ST = amrex::simd::SIMDReal<WIDTH>;
            ST const x = amrex::simd::index_to_real<ST>(i - lo[0]) * dx;
            amrex::Real const y = (j - lo[1]) * dy;
            amrex::Real const z = (k - lo[2]) * dz;

            return SumOfIntegratedPotential3D<WIDTH>(x, y, z, dx, dy, dz);
        });
    }else{
        // 2D sliced solver
        obc_solver->setGreensFunction(
        [=] AMREX_GPU_DEVICE (auto i, int j, int k)
        {
            // i is a scalar int (GPU / non-SIMD CPU) or a SIMD int register
            // (CPU with SIMD); WIDTH is deduced accordingly (1 in the scalar case).
            constexpr int WIDTH = amrex::simd::lane_count_v<decltype(i)>;
            using ST = amrex::simd::SIMDReal<WIDTH>;
            ST const x = amrex::simd::index_to_real<ST>(i - lo[0]) * dx;
            amrex::Real const y = (j - lo[1]) * dy;
            amrex::ignore_unused(k);

            return SumOfIntegratedPotential2D<WIDTH>(x, y, dx, dy);
        });

    }

    obc_solver->solve(phi, rho);
} // computePhiIGF

} // namespace ablastr::fields
