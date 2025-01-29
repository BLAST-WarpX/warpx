/* Copyright 2019 Maxence Thevenet, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include <AMReX_Dim3.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

#include <AMReX_BaseFwd.H>

#ifndef WARPX_FILTER_H_
#define WARPX_FILTER_H_

class Filter
{
public:
    Filter () = default;

    // Apply stencil on MultiFab.
    // Guard cells are handled inside this function
    void ApplyStencil (amrex::MultiFab& dstmf,
                       const amrex::MultiFab& srcmf, int lev, int scomp=0,
                       int dcomp=0, int ncomp=10000);

    // Apply stencil on a FabArray.
    void ApplyStencil (amrex::FArrayBox& dstfab,
                       const amrex::FArrayBox& srcfab, const amrex::Box& tbx,
                       int scomp=0, int dcomp=0, int ncomp=10000);

    // public for cuda
    void DoFilter (const amrex::Box& tbx,
                   amrex::Array4<amrex::Real const> const& tmp,
                   amrex::Array4<amrex::Real      > const& dst,
                   int scomp, int dcomp, int ncomp);

    // Length of stencil in each included direction
    amrex::IntVect stencil_length_each_dir;

protected:
    // Stencil along each direction.
    amrex::Gpu::DeviceVector<amrex::Real> m_stencil_0, m_stencil_1, m_stencil_2;
    // Length of each stencil, 1 for dimensions not included
    amrex::Dim3 slen;

private:

};
#endif // #ifndef WARPX_FILTER_H_
