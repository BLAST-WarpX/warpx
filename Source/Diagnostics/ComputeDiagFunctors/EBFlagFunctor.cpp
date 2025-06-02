#include "EBFlagFunctor.H"

#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_Extension.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

EBFlagFunctor::EBFlagFunctor (
    const int lev,
    const amrex::IntVect crse_ratio,
    const int ncomp
)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_lev(lev)
{
    // TODO: add runtime check that EB are enabled + ifdef to ensure compilation
    // Write only in one output component.
    AMREX_ALWAYS_ASSERT(ncomp == 1);
}

void
EBFlagFunctor::operator()(amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer*/) const
{

    // TODO: add documentation of parameter
    // TODO: change name to eb covered
    // TODO: handle RZ case

    // Extract structures for embedded boundaries
    auto& warpx = WarpX::GetInstance();
    amrex::EBFArrayBoxFactory const& eb_fact = warpx.fieldEBFactory(m_lev);

    ablastr::coarsen::sample::Coarsen(mf_dst, eb_fact.getVolFrac(), dcomp, 0, nComp(), 0, m_crse_ratio);
}
