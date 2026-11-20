/* Copyright 2019 Andrew Myers, Maxence Thevenet, Weiqun Zhang
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Filter.H"

#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/profiler/ProfilerWrapper.H>
#include <AMReX_Arena.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#include <algorithm>

using namespace amrex;

#ifdef AMREX_USE_GPU

/* \brief Apply stencil on MultiFab (GPU version, 2D/3D).
 * \param dstmf Destination MultiFab
 * \param srcmf source MultiFab
 * \param[in] lev mesh refinement level
 * \param scomp first component of srcmf on which the filter is applied
 * \param dcomp first component of dstmf on which the filter is applied
 * \param ncomp Number of components on which the filter is applied.
 */
void
Filter::ApplyStencil (MultiFab& dstmf, const MultiFab& srcmf, const int lev, int scomp, int dcomp, int ncomp)
{
    ABLASTR_PROFILE("Filter::ApplyStencil(MultiFab)");
    ncomp = std::min(ncomp, srcmf.nComp());

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    for (MFIter mfi(dstmf); mfi.isValid(); ++mfi)
    {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        const auto& src = srcmf.array(mfi);
        const auto& dst = dstmf.array(mfi);
        const Box& tbx = mfi.growntilebox();

        // Apply filter
        DoFilter(tbx, src, dst, scomp, dcomp, ncomp);

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

/* \brief Apply stencil on FArrayBox (GPU version, 2D/3D).
 * \param dstfab Destination FArrayBox
 * \param srcmf source FArrayBox
 * \param tbx Grown box on which srcfab is defined.
 * \param scomp first component of srcfab on which the filter is applied
 * \param dcomp first component of dstfab on which the filter is applied
 * \param ncomp Number of components on which the filter is applied.
 */
void
Filter::ApplyStencil (FArrayBox& dstfab, const FArrayBox& srcfab,
                      const Box& tbx, int scomp, int dcomp, int ncomp)
{
    ABLASTR_PROFILE("Filter::ApplyStencil(FArrayBox)");
    ncomp = std::min(ncomp, srcfab.nComp());
    const auto& src = srcfab.array();
    const auto& dst = dstfab.array();

    // Apply filter
    DoFilter(tbx, src, dst, scomp, dcomp, ncomp);
}

/* \brief Apply stencil (GPU version)
 *
 * The bilinear/binomial stencil is separable, so the multi-dimensional filter
 * is applied as a sequence of one-dimensional passes (one per dimension) that
 * write into local scratch buffers. Compared to evaluating the full
 * tensor-product stencil in a single kernel, this greatly reduces the number
 * of memory accesses per cell and keeps each pass a simple, low-register
 * kernel (better occupancy). The scratch buffers are grown just enough in the
 * not-yet-filtered directions for the subsequent passes, so no ghost-cell
 * communication is required between passes.
 */
void Filter::DoFilter (const Box& tbx,
                       Array4<Real const> const& src,
                       Array4<Real      > const& dst,
                       int scomp, int dcomp, int ncomp)
{
    AMREX_D_TERM(
    amrex::Real const* AMREX_RESTRICT s0 = m_stencil_0.data();,
    amrex::Real const* AMREX_RESTRICT s1 = m_stencil_1.data();,
    amrex::Real const* AMREX_RESTRICT s2 = m_stencil_2.data();
    )
    Dim3 slen_local = slen;

    // Note: only the first pass (reading from src) needs to pad out-of-bound
    // accesses with zeros; subsequent passes read from scratch buffers that are
    // sized to always be in bounds.

#if AMREX_SPACEDIM == 3
    const int rx = slen_local.x - 1;
    const int ry = slen_local.y - 1;
    const int rz = slen_local.z - 1;

    // Scratch buffers, grown in the not-yet-filtered directions so that each
    // subsequent pass finds its neighbors in bounds.
    const Box xbx = amrex::grow(tbx, IntVect(0, ry, rz));
    const Box ybx = amrex::grow(tbx, IntVect(0,  0, rz));
    FArrayBox xfab(xbx, ncomp, The_Async_Arena());
    FArrayBox yfab(ybx, ncomp, The_Async_Arena());
    const Array4<Real> xtmp = xfab.array();
    const Array4<Real> ytmp = yfab.array();

    // Pass in x: read src (zero-padded), write xtmp
    AMREX_PARALLEL_FOR_4D ( xbx, ncomp, i, j, k, n,
    {
        const auto src_zeropad = [src] (const int jj, const int kk, const int ll, const int nn) noexcept
        {
            return src.contains(jj,kk,ll) ? src(jj,kk,ll,nn) : 0.0_rt;
        };
        Real d = 0.0;
        for (int i0=0; i0 <= rx; ++i0) {
            d += s0[i0]*( src_zeropad(i-i0,j,k,scomp+n) + src_zeropad(i+i0,j,k,scomp+n) );
        }
        xtmp(i,j,k,n) = d;
    });

    // Pass in y: read xtmp (in bounds), write ytmp
    AMREX_PARALLEL_FOR_4D ( ybx, ncomp, i, j, k, n,
    {
        Real d = 0.0;
        for (int i1=0; i1 <= ry; ++i1) {
            d += s1[i1]*( xtmp(i,j-i1,k,n) + xtmp(i,j+i1,k,n) );
        }
        ytmp(i,j,k,n) = d;
    });

    // Pass in z: read ytmp (in bounds), write dst
    AMREX_PARALLEL_FOR_4D ( tbx, ncomp, i, j, k, n,
    {
        Real d = 0.0;
        for (int i2=0; i2 <= rz; ++i2) {
            d += s2[i2]*( ytmp(i,j,k-i2,n) + ytmp(i,j,k+i2,n) );
        }
        dst(i,j,k,dcomp+n) = d;
    });
#elif AMREX_SPACEDIM == 2
    const int rx = slen_local.x - 1;
    const int ry = slen_local.y - 1;

    const Box xbx = amrex::grow(tbx, IntVect(0, ry));
    FArrayBox xfab(xbx, ncomp, The_Async_Arena());
    const Array4<Real> xtmp = xfab.array();

    // Pass in x: read src (zero-padded), write xtmp
    AMREX_PARALLEL_FOR_4D ( xbx, ncomp, i, j, k, n,
    {
        const auto src_zeropad = [src] (const int jj, const int kk, const int ll, const int nn) noexcept
        {
            return src.contains(jj,kk,ll) ? src(jj,kk,ll,nn) : 0.0_rt;
        };
        Real d = 0.0;
        for (int i0=0; i0 <= rx; ++i0) {
            d += s0[i0]*( src_zeropad(i-i0,j,k,scomp+n) + src_zeropad(i+i0,j,k,scomp+n) );
        }
        xtmp(i,j,k,n) = d;
    });

    // Pass in y: read xtmp (in bounds), write dst
    AMREX_PARALLEL_FOR_4D ( tbx, ncomp, i, j, k, n,
    {
        Real d = 0.0;
        for (int i1=0; i1 <= ry; ++i1) {
            d += s1[i1]*( xtmp(i,j-i1,k,n) + xtmp(i,j+i1,k,n) );
        }
        dst(i,j,k,dcomp+n) = d;
    });
#elif AMREX_SPACEDIM == 1
    const int rx = slen_local.x - 1;

    // Single pass in x: read src (zero-padded), write dst
    AMREX_PARALLEL_FOR_4D ( tbx, ncomp, i, j, k, n,
    {
        const auto src_zeropad = [src] (const int jj, const int kk, const int ll, const int nn) noexcept
        {
            return src.contains(jj,kk,ll) ? src(jj,kk,ll,nn) : 0.0_rt;
        };
        Real d = 0.0;
        for (int i0=0; i0 <= rx; ++i0) {
            d += s0[i0]*( src_zeropad(i-i0,j,k,scomp+n) + src_zeropad(i+i0,j,k,scomp+n) );
        }
        dst(i,j,k,dcomp+n) = d;
    });
#endif
}

#else

/* \brief Apply stencil on MultiFab (CPU version, 2D/3D).
 * \param dstmf Destination MultiFab
 * \param srcmf source MultiFab
 * \param[in] lev mesh refinement level
 * \param scomp first component of srcmf on which the filter is applied
 * \param dcomp first component of dstmf on which the filter is applied
 * \param ncomp Number of components on which the filter is applied.
 */
void
Filter::ApplyStencil (amrex::MultiFab& dstmf, const amrex::MultiFab& srcmf, const int lev, int scomp, int dcomp, int ncomp)
{
    ABLASTR_PROFILE("Filter::ApplyStencil(MultiFab)");
    ncomp = std::min(ncomp, srcmf.nComp());

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

#ifdef AMREX_USE_OMP
// never runs on GPU since in the else branch of AMREX_USE_GPU
#pragma omp parallel
#endif
    {
        FArrayBox tmpfab;
        for (MFIter mfi(dstmf,true); mfi.isValid(); ++mfi){

            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
            }
            auto wt = static_cast<amrex::Real>(amrex::second());

            const auto& srcfab = srcmf[mfi];
            auto& dstfab = dstmf[mfi];
            const Box& tbx = mfi.growntilebox();
            const Box& gbx = amrex::grow(tbx,stencil_length_each_dir-1);
            // tmpfab has enough ghost cells for the stencil
            tmpfab.resize(gbx,ncomp);
            tmpfab.setVal(0.0, gbx, 0, ncomp);
            // Copy values in srcfab into tmpfab
            const Box& ibx = gbx & srcfab.box();
            tmpfab.copy(srcfab, ibx, scomp, ibx, 0, ncomp);
            // Apply filter
            DoFilter(tbx, tmpfab.array(), dstfab.array(), 0, dcomp, ncomp);

            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
                wt = static_cast<amrex::Real>(amrex::second()) - wt;
                amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
            }
        }
    }
}

/* \brief Apply stencil on FArrayBox (CPU version, 2D/3D).
 * \param dstfab Destination FArrayBox
 * \param srcmf source FArrayBox
 * \param tbx Grown box on which srcfab is defined.
 * \param scomp first component of srcfab on which the filter is applied
 * \param dcomp first component of dstfab on which the filter is applied
 * \param ncomp Number of components on which the filter is applied.
 */
void
Filter::ApplyStencil (amrex::FArrayBox& dstfab, const amrex::FArrayBox& srcfab,
                      const amrex::Box& tbx, int scomp, int dcomp, int ncomp)
{
    ABLASTR_PROFILE("Filter::ApplyStencil(FArrayBox)");
    ncomp = std::min(ncomp, srcfab.nComp());
    FArrayBox tmpfab;
    const Box& gbx = amrex::grow(tbx,stencil_length_each_dir-1);
    // tmpfab has enough ghost cells for the stencil
    tmpfab.resize(gbx,ncomp);
    tmpfab.setVal(0.0, gbx, 0, ncomp);
    // Copy values in srcfab into tmpfab
    const Box& ibx = gbx & srcfab.box();
    tmpfab.copy(srcfab, ibx, scomp, ibx, 0, ncomp);
    // Apply filter
    DoFilter(tbx, tmpfab.array(), dstfab.array(), 0, dcomp, ncomp);
}

void Filter::DoFilter (const Box& tbx,
                       Array4<Real const> const& tmp,
                       Array4<Real      > const& dst,
                       int scomp, int dcomp, int ncomp)
{
    const auto lo = amrex::lbound(tbx);
    const auto hi = amrex::ubound(tbx);
    // tmp and dst are of type Array4 (Fortran ordering)
    AMREX_D_TERM(
    amrex::Real const* AMREX_RESTRICT s0 = m_stencil_0.data();,
    amrex::Real const* AMREX_RESTRICT s1 = m_stencil_1.data();,
    amrex::Real const* AMREX_RESTRICT s2 = m_stencil_2.data();
    )
    for (int n = 0; n < ncomp; ++n) {
        // Set dst value to 0.
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    dst(i,j,k,dcomp+n) = 0.0;
                }
            }
        }
        // 3 nested loop on 3D stencil
        for         (int i2=0; i2 < slen.z; ++i2){
            for     (int i1=0; i1 < slen.y; ++i1){
                for (int i0=0; i0 < slen.x; ++i0){
                    const Real sss = AMREX_D_TERM(s0[i0], *s1[i1], *s2[i2]);
                    // 3 nested loop on 3D array
                    for         (int k = lo.z; k <= hi.z; ++k) {
                        for     (int j = lo.y; j <= hi.y; ++j) {
                            AMREX_PRAGMA_SIMD
                            for (int i = lo.x; i <= hi.x; ++i) {
#if AMREX_SPACEDIM == 3
                                dst(i,j,k,dcomp+n) += sss*(tmp(i-i0,j-i1,k-i2,scomp+n)
                                                          +tmp(i+i0,j-i1,k-i2,scomp+n)
                                                          +tmp(i-i0,j+i1,k-i2,scomp+n)
                                                          +tmp(i+i0,j+i1,k-i2,scomp+n)
                                                          +tmp(i-i0,j-i1,k+i2,scomp+n)
                                                          +tmp(i+i0,j-i1,k+i2,scomp+n)
                                                          +tmp(i-i0,j+i1,k+i2,scomp+n)
                                                          +tmp(i+i0,j+i1,k+i2,scomp+n));
#elif AMREX_SPACEDIM == 2
                                dst(i,j,k,dcomp+n) += sss*(tmp(i-i0,j-i1,k,scomp+n)
                                                          +tmp(i+i0,j-i1,k,scomp+n)
                                                          +tmp(i-i0,j+i1,k,scomp+n)
                                                          +tmp(i+i0,j+i1,k,scomp+n));
#elif AMREX_SPACEDIM == 1
                                dst(i,j,k,dcomp+n) += sss*(tmp(i-i0,j,k,scomp+n)
                                                          +tmp(i+i0,j,k,scomp+n));
#endif
                            }
                        }
                    }
                }
            }
        }
    }
}

#endif // #ifdef AMREX_USE_CUDA
