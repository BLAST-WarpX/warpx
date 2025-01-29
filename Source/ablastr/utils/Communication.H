/* Copyright 2019-2022 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_UTILS_COMMUNICATION_H_
#define ABLASTR_UTILS_COMMUNICATION_H_

#include <AMReX_FabArrayBase.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Periodicity.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

#include <optional>


namespace ablastr::utils::communication
{

using comm_float_type = float;

template <class FAB1, class FAB2>
void
mixedCopy (amrex::FabArray<FAB1>& dst, amrex::FabArray<FAB2> const& src, int srccomp, int dstcomp, int numcomp, const amrex::IntVect& nghost)
{
    auto const& srcma = src.const_arrays();
    auto const& dstma = dst.arrays();
    ParallelFor(dst, nghost, numcomp,
    [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k, int n) noexcept
    {
        dstma[box_no](i,j,k,dstcomp+n) = (typename FAB1::value_type) srcma[box_no](i,j,k,srccomp+n);
    });
    amrex::Gpu::synchronize();
}

void ParallelCopy(amrex::MultiFab &dst,
                  const amrex::MultiFab &src,
                  int src_comp,
                  int dst_comp,
                  int num_comp,
                  const amrex::IntVect &src_nghost,
                  const amrex::IntVect &dst_nghost,
                  bool do_single_precision_comms,
                  const amrex::Periodicity &period = amrex::Periodicity::NonPeriodic(),
                  amrex::FabArrayBase::CpOp op = amrex::FabArrayBase::COPY);

void ParallelAdd (amrex::MultiFab &dst,
                  const amrex::MultiFab &src,
                  int src_comp,
                  int dst_comp,
                  int num_comp,
                  const amrex::IntVect &src_nghost,
                  const amrex::IntVect &dst_nghost,
                  bool do_single_precision_comms,
                  const amrex::Periodicity &period = amrex::Periodicity::NonPeriodic());

void FillBoundary (amrex::MultiFab &mf,
                   bool do_single_precision_comms,
                   const amrex::Periodicity &period = amrex::Periodicity::NonPeriodic(),
                   std::optional<bool> nodal_sync=std::nullopt);

void FillBoundary (amrex::MultiFab &mf,
                   amrex::IntVect ng,
                   bool do_single_precision_comms,
                   const amrex::Periodicity &period = amrex::Periodicity::NonPeriodic(),
                   std::optional<bool> nodal_sync = std::nullopt);

void FillBoundary (amrex::iMultiFab &mf,
                   const amrex::Periodicity &period = amrex::Periodicity::NonPeriodic());

void FillBoundary (amrex::iMultiFab&         mf,
                   amrex::IntVect            ng,
                   const amrex::Periodicity& period = amrex::Periodicity::NonPeriodic());

void
FillBoundary(amrex::Vector<amrex::MultiFab *> const &mf, bool do_single_precision_comms,
             const amrex::Periodicity &period, std::optional<bool> nodal_sync=std::nullopt);

void
SumBoundary (amrex::MultiFab &mf,
             int start_comp,
             int num_comps,
             amrex::IntVect src_ng,
             amrex::IntVect dst_ng,
             bool do_single_precision_comms,
             const amrex::Periodicity &period = amrex::Periodicity::NonPeriodic());

void OverrideSync (amrex::MultiFab &mf,
                   bool do_single_precision_comms,
                   const amrex::Periodicity &period = amrex::Periodicity::NonPeriodic());
}

#endif // ABLASTR_UTILS_COMMUNICATION_H_
