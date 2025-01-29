#include "DivBFunctor.H"

#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

DivBFunctor::DivBFunctor (
    ablastr::fields::VectorField const & arr_mf_src,
    const int lev,
    const amrex::IntVect crse_ratio,
    bool convertRZmodes2cartesian,
    const int ncomp
)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_arr_mf_src(arr_mf_src), m_lev(lev),
      m_convertRZmodes2cartesian(convertRZmodes2cartesian)
{}

void
DivBFunctor::operator()(amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer*/) const
{
    auto& warpx = WarpX::GetInstance();
    // Guard cell is set to 1 for generality. However, for a cell-centered
    // output Multifab, mf_dst, the guard-cell data is not needed especially considering
    // the operations performend in the CoarsenAndInterpolate function.
    constexpr int ng = 1;
    // A cell-centered divB multifab spanning the entire domain is generated
    // and divB is computed on the cell-center, with ng=1.
    amrex::MultiFab divB( warpx.boxArray(m_lev), warpx.DistributionMap(m_lev), WarpX::ncomps, ng );
    WarpX::ComputeDivB(divB, 0, m_arr_mf_src, WarpX::CellSize(m_lev) );
    // // Coarsen and Interpolate from divB to coarsened/reduced_domain mf_dst
    // ablastr::coarsen::sample::Coarsen( mf_dst, divB, dcomp, 0, nComp(), 0, m_crse_ratio);
#ifdef WARPX_DIM_RZ
    if (m_convertRZmodes2cartesian) {
        // In cylindrical geometry, sum real part of all modes of divE in
        // temporary multifab mf_dst_stag, and cell-center it to mf_dst.
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            nComp()==1,
            "The RZ averaging over modes must write into 1 single component");
        amrex::MultiFab mf_dst_stag(divB.boxArray(), warpx.DistributionMap(m_lev), 1, divB.nGrowVect());
        // Mode 0
        amrex::MultiFab::Copy(mf_dst_stag, divB, 0, 0, 1, divB.nGrowVect());
        for (int ic=1 ; ic < divB.nComp() ; ic += 2) {
            // Real part of all modes > 0
            amrex::MultiFab::Add(mf_dst_stag, divB, ic, 0, 1, divB.nGrowVect());
        }
        // TODO check if coarsening is needed, otherwise copy
        ablastr::coarsen::sample::Coarsen( mf_dst, mf_dst_stag, dcomp, 0, nComp(), 0,  m_crse_ratio);
    } else {
        // TODO check if coarsening is needed, otherwise copy
        ablastr::coarsen::sample::Coarsen( mf_dst, divB, dcomp, 0, nComp(), 0, m_crse_ratio);
    }
#else
    // In cartesian geometry, coarsen and interpolate from simulation MultiFab, divE,
    // to output diagnostic MultiFab, mf_dst.
    ablastr::coarsen::sample::Coarsen(mf_dst, divB, dcomp, 0, nComp(), 0, m_crse_ratio);
    amrex::ignore_unused(m_convertRZmodes2cartesian);
#endif
}
