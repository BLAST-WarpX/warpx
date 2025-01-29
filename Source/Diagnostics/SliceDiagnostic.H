/* Copyright 2019 Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SliceDiagnostic_H_
#define WARPX_SliceDiagnostic_H_

#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

#include <memory>

std::unique_ptr<amrex::MultiFab> CreateSlice( const amrex::MultiFab& mf,
               const amrex::Vector<amrex::Geometry> &dom_geom,
               amrex::RealBox &slice_realbox,
               amrex::IntVect &slice_cr_ratio );

void CheckSliceInput( amrex::RealBox real_box,
    amrex::RealBox &slice_cc_nd_box, amrex::RealBox &slice_realbox,
    amrex::IntVect &slice_cr_ratio, amrex::Vector<amrex::Geometry> dom_geom,
    amrex::IntVect SliceType, amrex::IntVect &slice_lo,
    amrex::IntVect &slice_hi, amrex::IntVect &interp_lo);

void InterpolateSliceValues( amrex::MultiFab& smf,
     amrex::IntVect interp_lo, amrex::RealBox slice_realbox,
     const amrex::Vector<amrex::Geometry>& geom, int ncomp, int nghost,
     amrex::IntVect slice_lo, amrex::IntVect slice_hi,
     amrex::IntVect SliceType, amrex::RealBox real_box);

void InterpolateLo( const amrex::Box& bx, amrex::FArrayBox &fabox,
    amrex::IntVect slice_lo, amrex::Vector<amrex::Geometry> geom,
    int idir, amrex::IntVect IndType, amrex::RealBox slice_realbox,
    int srccomp, int ncomp, int nghost, amrex::RealBox real_box);

#endif
