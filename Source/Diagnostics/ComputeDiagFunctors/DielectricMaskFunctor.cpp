/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "DielectricMaskFunctor.H"

#include "FieldSolver/ElectrostaticSolvers/DielectricMaterials.H"
#include "WarpX.H"

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

DielectricMaskFunctor::DielectricMaskFunctor (int const lev, amrex::IntVect crse_ratio)
    : ComputeDiagFunctor(1, crse_ratio), m_lev(lev)
{}

void
DielectricMaskFunctor::operator() (
    amrex::MultiFab& mf_dst,
    int const dcomp,
    int const /*i_buffer*/) const
{
    auto& warpx = WarpX::GetInstance();
    amrex::iMultiFab const* mask = warpx.GetDielectricMaterials().MaterialID(m_lev);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        mask != nullptr,
        "The dielectric_mask diagnostic requires dielectric material data.");
    AMREX_ASSUME(mask != nullptr);

    amrex::MultiFab mask_real(
        mask->boxArray(), mask->DistributionMap(), 1, mask->nGrowVect());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.growntilebox(mask->nGrowVect());
        amrex::Array4<amrex::Real> const& dst = mask_real.array(mfi);
        amrex::Array4<int const> const& src = mask->const_array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            dst(i, j, k) = static_cast<amrex::Real>(src(i, j, k));
        });
    }

    InterpolateMFForDiag(mf_dst, mask_real, dcomp, warpx.DistributionMap(m_lev), false);
}
