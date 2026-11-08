/* Copyright 2019-2022 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_UTILS_COMMUNICATION_H_
#define ABLASTR_UTILS_COMMUNICATION_H_

#include <AMReX_Array4.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Periodicity.H>
#include <AMReX_SPACE.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

#include <optional>


namespace ablastr::utils::communication
{

using comm_float_type = float;

/** Copy (and convert) cells of src to dst, within nghost guard cells of dst
 *
 * src and dst must have the same BoxArray and DistributionMapping.
 * With guard_cells_only, only the guard cells of dst within nghost are copied: cells inside
 * the valid box of each FAB of dst are neither read nor written.
 */
template <class FAB1, class FAB2>
void
mixedCopy (amrex::FabArray<FAB1>& dst, amrex::FabArray<FAB2> const& src, int srccomp, int dstcomp,
           int numcomp, const amrex::IntVect& nghost, bool guard_cells_only = false)
{
    auto const& srcma = src.const_arrays();
    auto const& dstma = dst.arrays();
    // the valid box of each FAB of dst is its full box shrunk by the guard cells of dst
    amrex::IntVect const dst_ngrow = dst.nGrowVect();
    ParallelFor(dst, nghost, numcomp,
    [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k, int n) noexcept
    {
        if (guard_cells_only) {
            amrex::Dim3 const lo = amrex::lbound(dstma[box_no]);
            amrex::Dim3 const hi = amrex::ubound(dstma[box_no]);
            bool const is_valid_cell = AMREX_D_TERM(
                   i >= lo.x + dst_ngrow[0] && i <= hi.x - dst_ngrow[0],
                && j >= lo.y + dst_ngrow[1] && j <= hi.y - dst_ngrow[1],
                && k >= lo.z + dst_ngrow[2] && k <= hi.z - dst_ngrow[2]);
            if (is_valid_cell) { return; }
        }
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

/** Like ParallelCopy, but only writes into the guard cells of dst
 *
 * Cells inside the valid box of each FAB of dst, including the outermost valid cells of
 * nodal directions, are not modified. Guard cells of a FAB that overlap the valid box of
 * another FAB are written, though. See amrex::FabArray::ParallelCopyToGhost.
 *
 * With do_single_precision_comms, all guard cells of dst within dst_nghost are converted
 * to comm_float_type and back, i.e., also those that no cell of src overlaps.
 */
void ParallelCopyToGhost (amrex::MultiFab &dst,
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
