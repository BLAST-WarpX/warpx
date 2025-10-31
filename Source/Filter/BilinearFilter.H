/* Copyright 2019 Andrew Myers, Maxence Thevenet, Weiqun Zhang
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_BILIN_FILTER_H_
#define WARPX_BILIN_FILTER_H_

#include "Filter.H"

#include <AMReX_Array.H>
#include <AMReX_Config.H>

class BilinearFilter : public Filter
{
public:

    BilinearFilter(WarpX* warpx,
                   amrex::Array<unsigned int, AMREX_SPACEDIM> const& np)
        : Filter(warpx), npass_each_dir(np) {}

    void ComputeStencils();

    amrex::Array<unsigned int, AMREX_SPACEDIM> npass_each_dir;

};
#endif // #ifndef WARPX_BILIN_FILTER_H_
