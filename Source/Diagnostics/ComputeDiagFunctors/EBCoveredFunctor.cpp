#include "EBCoveredFunctor.H"

#include "EmbeddedBoundary/Enabled.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_Extension.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

EBCoveredFunctor::EBCoveredFunctor (
    const int lev,
    const amrex::IntVect crse_ratio,
    const int ncomp
)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_lev(lev)
{
    // Write only in one output component.
    AMREX_ALWAYS_ASSERT(ncomp == 1);
}

void
EBCoveredFunctor::operator()(amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer*/) const
{
    // Extract structures for embedded boundaries
    // TODO: add runtime check that EB are enabled + ifdef to ensure compilation

    if (EB::enabled()) {
        // If EB are enabled, fill the MultiFab with the volume fraction
#if (defined AMREX_USE_EB)
        auto& warpx = WarpX::GetInstance();
        amrex::EBFArrayBoxFactory const& eb_fact = warpx.fieldEBFactory(m_lev);
        ablastr::coarsen::sample::Coarsen(mf_dst, eb_fact.getVolFrac(), dcomp, 0, nComp(), 0, m_crse_ratio);
        // The fraction of the cell that is covered by the EB is 1 - the volume fraction
        mf_dst.mult(-1.0, dcomp, nComp(), 0);
        mf_dst.plus(1.0, dcomp, nComp(), 0);
#endif
    } else {
        // If EB are disabled, set the fraction to 0
        amrex::ignore_unused(m_lev);
        mf_dst.setVal(0.0, dcomp, nComp(), 0);
    }
}
