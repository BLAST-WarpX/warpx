#include "HeatFluxFunctor.H"

#include "Diagnostics/ComputeDiagFunctors/ComputeDiagFunctor.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX_BLassert.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

HeatFluxFunctor::HeatFluxFunctor (const int lev,
        const amrex::IntVect crse_ratio, const int ispec, const int ncomp)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_lev(lev), m_ispec(ispec)
{
    // Heat flux has 3 components: Qx, Qy, Qz
    AMREX_ALWAYS_ASSERT(ncomp == 3);
}

void
HeatFluxFunctor::operator() (amrex::MultiFab& mf_dst, const int dcomp, const int /*i_buffer*/) const
{
    auto& warpx = WarpX::GetInstance();

    auto& pc = warpx.GetPartContainer().GetParticleContainer(m_ispec);
    amrex::Real const mass = pc.getMass();

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(mass > 0.,
        "The heat flux diagnostic can not be calculated for a massless species.");

    std::unique_ptr<amrex::MultiFab> heatflux = pc.GetAverageNGPHeatFlux(m_lev);

    // Coarsen and interpolate all 3 components from heatflux to the output diagnostic MultiFab.
    ablastr::coarsen::sample::Coarsen(mf_dst, *heatflux, dcomp, 0, nComp(), 0, m_crse_ratio);
}
