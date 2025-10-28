#include "DivEFunctor.H"

#include "Utils/TextMsg.H"
#ifdef WARPX_DIM_RZ
#   include "Utils/WarpXAlgorithmSelection.H"
#endif
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

DivEFunctor::DivEFunctor (
    ablastr::fields::VectorField const & arr_mf_src,
    const int lev,
    const amrex::IntVect crse_ratio,
    bool convertRZmodes2cartesian,
    const int ncomp
)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_arr_mf_src(arr_mf_src), m_lev(lev),
      m_convertRZmodes2cartesian(convertRZmodes2cartesian)
{
    amrex::ignore_unused(m_arr_mf_src);
}

void
DivEFunctor::operator()(amrex::MultiFab& mf_dst, const int dcomp, const int /*i_buffer*/) const
{
    auto& warpx = WarpX::GetInstance();
    // Guard cell is set to 1 for generality. However, for a cell-centered
    // output Multifab, mf_dst, the guard-cell data is not needed especially considering
    // the operations performend in the CoarsenAndInterpolate function.
    constexpr int ng = 1;

#ifndef WARPX_DIM_RZ
    // For staggered and nodal calculations, divE is computed on the nodes.
    // The temporary divE MultiFab is generated to comply with the location of divE.
    const auto cell_type = amrex::IntVect::TheNodeVector();
#else
    // For RZ spectral, all quantities are cell centered
    const auto cell_type =
        (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD)?
        amrex::IntVect::TheCellVector():amrex::IntVect::TheNodeVector();
#endif

    const amrex::BoxArray& ba = amrex::convert(warpx.boxArray(m_lev), cell_type);
    amrex::MultiFab divE(ba, warpx.DistributionMap(m_lev), WarpX::ncomps, ng );
    warpx.ComputeDivE(divE, m_lev);

#ifdef WARPX_DIM_RZ
    if (m_convertRZmodes2cartesian) {
        // In cylindrical geometry, sum real part of all modes of divE in
        // temporary multifab mf_dst_stag, and cell-center it to mf_dst.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            nComp()==1,
            "The RZ averaging over modes must write into 1 single component");
        amrex::MultiFab mf_dst_stag(divE.boxArray(), warpx.DistributionMap(m_lev), 1, divE.nGrowVect());
        // Mode 0
        amrex::MultiFab::Copy(mf_dst_stag, divE, 0, 0, 1, divE.nGrowVect());
        for (int ic=1 ; ic < divE.nComp() ; ic += 2) {
            // Real part of all modes > 0
            amrex::MultiFab::Add(mf_dst_stag, divE, ic, 0, 1, divE.nGrowVect());
        }
        ablastr::coarsen::sample::Coarsen( mf_dst, mf_dst_stag, dcomp, 0, nComp(), 0,  m_crse_ratio);
    } else {
        ablastr::coarsen::sample::Coarsen( mf_dst, divE, dcomp, 0, nComp(), 0, m_crse_ratio);
    }
#else
    // In cartesian geometry, coarsen and interpolate from simulation MultiFab, divE,
    // to output diagnostic MultiFab, mf_dst.
    ablastr::coarsen::sample::Coarsen(mf_dst, divE, dcomp, 0, nComp(), 0, m_crse_ratio);
    amrex::ignore_unused(m_convertRZmodes2cartesian);
#endif
}
