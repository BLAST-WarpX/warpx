#include "EBFlagFunctor.H"

#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_Extension.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

EBFlagFunctor::EBFlagFunctor (
    const int lev,
    const amrex::IntVect crse_ratio,
    bool convertRZmodes2cartesian,
    const int ncomp
)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_lev(lev),
      m_convertRZmodes2cartesian(convertRZmodes2cartesian)
{}

void
EBFlagFunctor::operator()(amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer*/) const
{
    // For now, this is an empty implementation as requested
    // The actual implementation will be added later
}
