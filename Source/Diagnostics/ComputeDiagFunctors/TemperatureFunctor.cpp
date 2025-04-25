
#include "TemperatureFunctor.H"

#include "Diagnostics/ComputeDiagFunctors/ComputeDiagFunctor.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

TemperatureFunctor::TemperatureFunctor (const int lev,
        const amrex::IntVect crse_ratio, const int ispec, const int ncomp)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_lev(lev), m_ispec(ispec)
{
    // Write only in one output component.
    AMREX_ALWAYS_ASSERT(ncomp == 1);
}

void
TemperatureFunctor::operator() (amrex::MultiFab& mf_dst, const int dcomp, const int /*i_buffer*/) const
{
    using namespace amrex::literals;
    auto& warpx = WarpX::GetInstance();

    auto& pc = warpx.GetPartContainer().GetParticleContainer(m_ispec);
    amrex::Real const mass = pc.getMass();  // Note, implicit conversion from ParticleReal

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(mass > 0.,
        "The temperature diagnostic can not be calculated for a massless species.");

    std::unique_ptr<amrex::MultiFab> temperature = pc.GetTemperature(m_lev);

    // Coarsen and interpolate from temperature to the output diagnostic MultiFab, mf_dst.
    ablastr::coarsen::sample::Coarsen(mf_dst, *temperature, dcomp, 0, nComp(), 0, m_crse_ratio);

}
