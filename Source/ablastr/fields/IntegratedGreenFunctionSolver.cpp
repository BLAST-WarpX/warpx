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
                amrex::BoxArray const & ba,
                bool const is_igf_2d_slices)
{
    using namespace amrex::literals;

    BL_PROFILE("ablastr::fields::computePhiIGF");

    // Define box that encompasses the full domain
    amrex::Box domain = ba.minimalBox();
    domain.surroundingNodes(); // get nodal points, since `phi` and `rho` are nodal
    domain.grow( phi.nGrowVect() ); // include guard cells

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
        if (is_igf_2d_slices) { info.setBatchMode(true); } // do 2D FFTs
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
        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> amrex::Real
        {
            int const i0 = i - lo[0];
            int const j0 = j - lo[1];
            int const k0 = k - lo[2];
            amrex::Real const x = i0*dx;
            amrex::Real const y = j0*dy;
            amrex::Real const z = k0*dz;

            return SumOfIntegratedPotential3D(x, y, z, dx, dy, dz);
        });
    }else{
        // 2D sliced solver
        obc_solver->setGreensFunction(
        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> amrex::Real
        {
            int const i0 = i - lo[0];
            int const j0 = j - lo[1];
            amrex::Real const x = i0*dx;
            amrex::Real const y = j0*dy;
            amrex::ignore_unused(k);

            return SumOfIntegratedPotential2D(x, y, dx, dy);
        });

    }

    obc_solver->solve(phi, rho);
} // computePhiIGF

} // namespace ablastr::fields
