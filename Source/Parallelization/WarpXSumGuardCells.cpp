/* Copyright 2019 Maxence Thevenet, Remi Lehe, Weiqun Zhang
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "WarpXSumGuardCells.H"

#include "Parallelization/Parallelization.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <ablastr/utils/Communication.H>

using namespace warpx;

void
WarpXSumGuardCells(amrex::MultiFab& mf, const amrex::Periodicity& period,
                   const amrex::IntVect& src_ngrow,
                   const int icomp, const int ncomp)
{
    amrex::IntVect const n_updated_guards = mf.nGrowVect();
    const auto do_single_precision_comms = parallelization::comms_in_single_precision_flag();
    ablastr::utils::communication::SumBoundary(mf, icomp, ncomp, src_ngrow, n_updated_guards, do_single_precision_comms, period);
}


void
WarpXSumGuardCells(amrex::MultiFab& dst, amrex::MultiFab& src,
                   const amrex::Periodicity& period,
                   const amrex::IntVect& src_ngrow,
                   const int icomp, const int ncomp)
{
    amrex::IntVect const n_updated_guards = dst.nGrowVect();

    dst.setVal(0., icomp, ncomp, n_updated_guards);
    dst.ParallelAdd(src, 0, icomp, ncomp, src_ngrow, n_updated_guards, period);
}
