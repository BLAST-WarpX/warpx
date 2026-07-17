#include "ProcessNumberFunctor.H"

#include "Diagnostics/ComputeDiagFunctors/ComputeDiagFunctor.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX_BLassert.H>
#include <AMReX_Config.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_GpuControl.H>
#include <AMReX_INT.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <memory>

ProcessNumberFunctor::ProcessNumberFunctor(const amrex::MultiFab * const mf_src, const int lev, const amrex::IntVect crse_ratio, const int ncomp)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_lev(lev)
{
    // mf_src will not be used, let's make sure it's null.
    AMREX_ALWAYS_ASSERT(mf_src == nullptr);
    // Write only in one output component.
    AMREX_ALWAYS_ASSERT(ncomp == 1);
}

void
ProcessNumberFunctor::operator()(amrex::MultiFab& mf_dst, const int dcomp, const int /*i_buffer*/) const
{
    // fill tmp multifab with the proc num
    auto& warpx = WarpX::GetInstance();
    amrex::MultiFab proc_mf(warpx.boxArray(m_lev), warpx.DistributionMap(m_lev), 1, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (amrex::MFIter mfi(proc_mf); mfi.isValid(); ++mfi) {
        proc_mf[mfi].setVal<amrex::RunOn::Device>(static_cast<amrex::Real>(amrex::ParallelDescriptor::MyProc()));
    }

    // Coarsen and interpolate from proc_mf to the output diagnostic MultiFab, mf_dst.
    ablastr::coarsen::sample::Coarsen(mf_dst, proc_mf, dcomp, 0, nComp(), 0, m_crse_ratio);
}
