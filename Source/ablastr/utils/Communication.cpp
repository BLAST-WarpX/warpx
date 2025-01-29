/* Copyright 2021 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Communication.H"

#include <AMReX_BaseFab.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_IntVect.H>
#include <AMReX_FabArray.H>
#include <AMReX_FabArrayUtility.H>
#include <AMReX_FabFactory.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_IndexType.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <memory>
#include <vector>


namespace ablastr::utils::communication
{

void ParallelCopy(amrex::MultiFab &dst, const amrex::MultiFab &src, int src_comp, int dst_comp, int num_comp,
                  const amrex::IntVect &src_nghost, const amrex::IntVect &dst_nghost,
                  bool do_single_precision_comms, const amrex::Periodicity &period,
                  amrex::FabArrayBase::CpOp op)
{
    BL_PROFILE("ablastr::utils::communication::ParallelCopy");

    using ablastr::utils::communication::comm_float_type;

    if (do_single_precision_comms)
    {
        amrex::FabArray<amrex::BaseFab<comm_float_type> > src_tmp(src.boxArray(),
                                                                  src.DistributionMap(),
                                                                  num_comp,
                                                                  src_nghost);
        mixedCopy(src_tmp, src, src_comp, 0, num_comp, src_nghost);

        amrex::FabArray<amrex::BaseFab<comm_float_type> > dst_tmp(dst.boxArray(),
                                                                  dst.DistributionMap(),
                                                                  num_comp,
                                                                  dst_nghost);

        mixedCopy(dst_tmp, dst, dst_comp, 0, num_comp, dst_nghost);

        dst_tmp.ParallelCopy(src_tmp, 0, 0, num_comp,
                             src_nghost, dst_nghost, period, op);

        mixedCopy(dst, dst_tmp, 0, dst_comp, num_comp, dst_nghost);
    }
    else
    {
        dst.ParallelCopy(src, src_comp, dst_comp, num_comp, src_nghost, dst_nghost, period, op);
    }
}

void ParallelAdd(amrex::MultiFab &dst, const amrex::MultiFab &src, int src_comp, int dst_comp, int num_comp,
                 const amrex::IntVect &src_nghost, const amrex::IntVect &dst_nghost,
                 bool do_single_precision_comms, const amrex::Periodicity &period)
{
    ablastr::utils::communication::ParallelCopy(dst, src, src_comp, dst_comp, num_comp, src_nghost, dst_nghost,
                                                do_single_precision_comms, period, amrex::FabArrayBase::ADD);
}

void FillBoundary (amrex::MultiFab &mf,
                   amrex::IntVect ng,
                   bool do_single_precision_comms,
                   const amrex::Periodicity &period,
                   std::optional<bool> nodal_sync)
{
    BL_PROFILE("ablastr::utils::communication::FillBoundary");

    // allow developers to always enforce nodal sync, independent of the
    // nodal_sync argument
    const bool do_nodal_sync_arg = nodal_sync.value_or(false);

    const amrex::ParmParse pp_ablastr("ablastr");
    bool do_nodal_sync_input = false;
    pp_ablastr.query("fillboundary_always_sync", do_nodal_sync_input);

    // logic: inputs overwrite argument unless argument is true
    bool const do_nodal_sync = do_nodal_sync_arg || do_nodal_sync_input;

    if (do_single_precision_comms)
    {
        amrex::FabArray<amrex::BaseFab<comm_float_type> > mf_tmp(mf.boxArray(),
                                                            mf.DistributionMap(),
                                                            mf.nComp(),
                                                            mf.nGrowVect());

        mixedCopy(mf_tmp, mf, 0, 0, mf.nComp(), mf.nGrowVect());

        if (do_nodal_sync) {
            mf_tmp.FillBoundaryAndSync(0, mf.nComp(), ng, period);
        } else {
            mf_tmp.FillBoundary(ng, period);
        }

        mixedCopy(mf, mf_tmp, 0, 0, mf.nComp(), mf.nGrowVect());
    }
    else
    {
        if (do_nodal_sync) {
            mf.FillBoundaryAndSync(0, mf.nComp(), ng, period);
        } else {
            mf.FillBoundary(ng, period);
        }
    }
}

void FillBoundary (amrex::MultiFab &mf, bool do_single_precision_comms, const amrex::Periodicity &period, std::optional<bool> nodal_sync)
{
    amrex::IntVect const ng = mf.n_grow;
    FillBoundary(mf, ng, do_single_precision_comms, period, nodal_sync);
}

void
FillBoundary (amrex::Vector<amrex::MultiFab *> const &mf, bool do_single_precision_comms,
             const amrex::Periodicity &period, std::optional<bool> nodal_sync)
{
    for (auto *x : mf) {
        ablastr::utils::communication::FillBoundary(*x, do_single_precision_comms, period, nodal_sync);
    }
}

void FillBoundary (amrex::iMultiFab &imf, const amrex::Periodicity &period)
{
    BL_PROFILE("ablastr::utils::communication::FillBoundary::iMultiFab");

    imf.FillBoundary(period);
}

void FillBoundary (amrex::iMultiFab&         imf,
                   amrex::IntVect            ng,
                   const amrex::Periodicity& period)
{
    BL_PROFILE("ablastr::utils::communication::FillBoundary::iMultiFab");

    imf.FillBoundary(ng, period);
}

void
SumBoundary (amrex::MultiFab &mf,
             int start_comp,
             int num_comps,
             amrex::IntVect src_ng,
             amrex::IntVect dst_ng,
             bool do_single_precision_comms,
             const amrex::Periodicity &period)
{
    BL_PROFILE("ablastr::utils::communication::SumBoundary");

    if (do_single_precision_comms)
    {
        amrex::FabArray<amrex::BaseFab<comm_float_type> > mf_tmp(mf.boxArray(),
                                                                 mf.DistributionMap(),
                                                                 num_comps,
                                                                 mf.nGrowVect());
        mixedCopy(mf_tmp, mf, start_comp, 0, num_comps, mf.nGrowVect());

        mf_tmp.SumBoundary(0, num_comps, src_ng, dst_ng, period);

        mixedCopy(mf, mf_tmp, 0, start_comp, num_comps, dst_ng);
    }
    else
    {
        mf.SumBoundary(start_comp, num_comps, src_ng, dst_ng, period);
    }
}

void OverrideSync (amrex::MultiFab &mf,
                   bool do_single_precision_comms,
                   const amrex::Periodicity &period)
{
    BL_PROFILE("ablastr::utils::communication::OverrideSync");

    if (mf.ixType().cellCentered()) { return; }

    if (do_single_precision_comms)
    {
        amrex::FabArray<amrex::BaseFab<comm_float_type> > mf_tmp(mf.boxArray(),
                                                                 mf.DistributionMap(),
                                                                 mf.nComp(),
                                                                 mf.nGrowVect());

        mixedCopy(mf_tmp, mf, 0, 0, mf.nComp(), mf.nGrowVect());

        auto msk = mf.OwnerMask(period);
        amrex::OverrideSync(mf_tmp, *msk, period);

        mixedCopy(mf, mf_tmp, 0, 0, mf.nComp(), mf.nGrowVect());
    }
    else
    {
        mf.OverrideSync(period);
    }
}

} // namespace ablastr::utils::communication
