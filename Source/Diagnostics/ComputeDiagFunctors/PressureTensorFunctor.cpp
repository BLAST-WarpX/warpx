
#include "PressureTensorFunctor.H"

#include "Diagnostics/ComputeDiagFunctors/ComputeDiagFunctor.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX_BLassert.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

PressureTensorFunctor::PressureTensorFunctor (const int lev,
        const amrex::IntVect crse_ratio, const int ispec, const int ncomp)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_lev(lev), m_ispec(ispec)
{
    // Pressure tensor has 6 independent components: xx, xy, xz, yy, yz, zz
    AMREX_ALWAYS_ASSERT(ncomp == 6);
}

void
PressureTensorFunctor::operator() (amrex::MultiFab& mf_dst, const int dcomp, const int /*i_buffer*/) const
{
    auto& warpx = WarpX::GetInstance();

    auto& pc = warpx.GetPartContainer().GetParticleContainer(m_ispec);
    amrex::Real const mass = pc.getMass();

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(mass > 0.,
        "The pressure tensor diagnostic can not be calculated for a massless species.");

    std::unique_ptr<amrex::MultiFab> ptensor = pc.GetAverageNGPPressureTensor(m_lev);

    // Coarsen and interpolate all 6 components from ptensor to the output diagnostic MultiFab.
    ablastr::coarsen::sample::Coarsen(mf_dst, *ptensor, dcomp, 0, nComp(), 0, m_crse_ratio);
}
