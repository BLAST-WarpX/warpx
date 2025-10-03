#include "PartPerCellFunctor.H"

#include "Diagnostics/ComputeDiagFunctors/ComputeDiagFunctor.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX_BLassert.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

using namespace amrex::literals;

PartPerCellFunctor::PartPerCellFunctor(WarpX* warpx, const amrex::MultiFab* mf_src, const int lev, amrex::IntVect crse_ratio, const int ncomp)
    : ComputeDiagFunctor(warpx, ncomp, crse_ratio), m_lev(lev)
{
    // mf_src will not be used, let's make sure it's null.
    AMREX_ALWAYS_ASSERT(mf_src == nullptr);
    // Write only in one output component.
    AMREX_ALWAYS_ASSERT(ncomp == 1);
}

void
PartPerCellFunctor::operator()(amrex::MultiFab& mf_dst, const int dcomp, const int /*i_buffer*/) const
{
    auto& warpx = *m_warpx;
    // Guard cell is set to 1 for generality. However, for a cell-centered
    // output Multifab, mf_dst, the guard-cell data is not needed especially considering
    // the operations performend in the CoarsenAndInterpolate function.
    constexpr int ng = 1;
    // Temporary cell-centered, single-component MultiFab for storing particles per cell.
    amrex::MultiFab ppc_mf(warpx.boxArray(m_lev), warpx.DistributionMap(m_lev), 1, ng);
    // Set value to 0, and increment the value in each cell with ppc.
    ppc_mf.setVal(0._rt);
    // Compute ppc which includes a summation over all species.
    warpx.GetPartContainer().Increment(ppc_mf, m_lev);
    // Coarsen and interpolate from ppc_mf to the output diagnostic MultiFab, mf_dst.
    ablastr::coarsen::sample::Coarsen(mf_dst, ppc_mf, dcomp, 0, nComp(), 0, m_crse_ratio);
}
