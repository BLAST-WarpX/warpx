#include "Interpolate.H"
#include "Interpolate_K.H"

#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_BCRec.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_SPACE.H>

namespace Interpolate
{
    using namespace amrex;

    std::unique_ptr<MultiFab>
    getInterpolatedScalar(
        const MultiFab& F_cp, const MultiFab& F_fp,
        const DistributionMapping& dm, const amrex::IntVect r_ratio,
        const Real* /*dx*/, const IntVect ngrow )
    {
        // Prepare the structure that will contain the returned fields
        std::unique_ptr<MultiFab> interpolated_F;
        interpolated_F = std::make_unique<MultiFab>(F_fp.boxArray(), dm, 1, ngrow);
        interpolated_F->setVal(0.);

        // Loop through the boxes and interpolate the values from the _cp data
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        {
            FArrayBox ffab; // Temporary array ; contains interpolated fields
            for (MFIter mfi(*interpolated_F); mfi.isValid(); ++mfi)
            {
                Box finebx = mfi.fabbox();
                finebx.coarsen(r_ratio).refine(r_ratio); // so that finebx is coarsenable

                const FArrayBox& cfab = (F_cp)[mfi];
                ffab.resize(finebx);

                // - Fully nodal
                if ( F_fp.is_nodal() ){
                    const amrex::IntVect refinement_vector{AMREX_D_DECL(r_ratio[0], r_ratio[1], r_ratio[2])};
                    node_bilinear_interp.interp(cfab, 0, ffab, 0, 1,
                                                finebx, refinement_vector, {}, {}, {}, 0, 0, RunOn::Device);
                } else {
                    WARPX_ABORT_WITH_MESSAGE("Unknown field staggering.");
                }

                // Add temporary array to the returned structure
                const Box& bx = (*interpolated_F)[mfi].box();
                (*interpolated_F)[mfi].plus<RunOn::Device>(ffab, bx, bx, 0, 0, 1);
                amrex::Gpu::streamSynchronize();
            }
        }
        return interpolated_F;
    }

    std::array<std::unique_ptr<MultiFab>, 3>
    getInterpolatedVector(
        const MultiFab* Fx_cp,
        const MultiFab* Fy_cp,
        const MultiFab* Fz_cp,
        const MultiFab* Fx_fp,
        const MultiFab* Fy_fp,
        const MultiFab* Fz_fp,
        const DistributionMapping& dm, const amrex::IntVect r_ratio,
        const Real* /*dx*/, const IntVect ngrow )
    {

        // Prepare the structure that will contain the returned fields
        std::array<std::unique_ptr<MultiFab>, 3> interpolated_F;
        interpolated_F[0] = std::make_unique<MultiFab>(Fx_fp->boxArray(), dm, 1, ngrow);
        interpolated_F[1] = std::make_unique<MultiFab>(Fy_fp->boxArray(), dm, 1, ngrow);
        interpolated_F[2] = std::make_unique<MultiFab>(Fz_fp->boxArray(), dm, 1, ngrow);

        const IntVect fx_type = interpolated_F[0]->ixType().toIntVect();
        const IntVect fy_type = interpolated_F[1]->ixType().toIntVect();
        const IntVect fz_type = interpolated_F[2]->ixType().toIntVect();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*interpolated_F[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& boxx = mfi.tilebox(fx_type);
            Box const& boxy = mfi.tilebox(fy_type);
            Box const& boxz = mfi.tilebox(fz_type);

            Array4<Real      > const& fx = interpolated_F[0]->array(mfi);
            Array4<Real      > const& fy = interpolated_F[1]->array(mfi);
            Array4<Real      > const& fz = interpolated_F[2]->array(mfi);
            Array4<Real const> const& cx = Fx_cp->const_array(mfi);
            Array4<Real const> const& cy = Fy_cp->const_array(mfi);
            Array4<Real const> const& cz = Fz_cp->const_array(mfi);

            amrex::ParallelFor(boxx, boxy, boxz,
                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                {
                    interp(j,k,l,fx,cx,r_ratio,fx_type);
                },
                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                {
                    interp(j,k,l,fy,cy,r_ratio,fy_type);
                },
                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                {
                    interp(j,k,l,fz,cz,r_ratio,fz_type);
                });
        }
        return interpolated_F;
    }
}
