/* Copyright 2019-2020 Andrew Myers, Axel Huebl, David Grote, Maxence Thevenet,
 * Remi Lehe, Revathi Jambunathan, Weiqun Zhang, Edoardo Zoni
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldIO.H"

#include "Utils/TextMsg.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_SPACE.H>

#include <algorithm>
#include <cstdint>
#include <memory>

using namespace amrex;

/** \brief
 * Convert an IntVect to a std::vector<std::uint64_t>
 * (used for compatibility with openPMD-api)
 */
std::vector<std::uint64_t>
getVec( const IntVect& v, bool reverse)
{
  // Convert the IntVect v to and std::vector u
  std::vector<std::uint64_t> u = {
    AMREX_D_DECL(
                 static_cast<std::uint64_t>(v[0]),
                 static_cast<std::uint64_t>(v[1]),
                 static_cast<std::uint64_t>(v[2])
                 )
  };
  // Reverse the order of elements, if v corresponds to the indices of a
  // Fortran-order array (like an AMReX FArrayBox)
  // but u is intended to be used with a C-order API (like openPMD)
  if (reverse) {
    std::reverse( u.begin(), u.end() );
  }

  return u;
}
/** \brief
 * Convert Real* pointer to a std::vector<double>,
 * (used for compatibility with the openPMD API)
 */
std::vector<double>
getVec( const Real* v , bool reverse)
{
  // Convert Real* v to and std::vector u
  std::vector<double> u = {
    AMREX_D_DECL(
                 static_cast<double>(v[0]),
                 static_cast<double>(v[1]),
                 static_cast<double>(v[2])
                 )
  };
  // Reverse the order of elements, if v corresponds to the indices of a
  // Fortran-order array (like an AMReX FArrayBox)
  // but u is intended to be used with a C-order API (like openPMD-api)
  if (reverse) {
    std::reverse( u.begin(), u.end() );
  }

  return u;
}

/** \brief
 * Convert an IntVect to a std::vector<std::uint64_t>
 * and reverse the order of the elements
 * (used for compatibility with the openPMD API)
 */
std::vector<std::uint64_t>
getReversedVec( const IntVect& v )
{
  // Convert the IntVect v to and std::vector u
  std::vector<std::uint64_t> u = {
    AMREX_D_DECL(
                 static_cast<std::uint64_t>(v[0]),
                 static_cast<std::uint64_t>(v[1]),
                 static_cast<std::uint64_t>(v[2])
                 )
  };
  // Reverse the order of elements, if v corresponds to the indices of a
  // Fortran-order array (like an AMReX FArrayBox)
  // but u is intended to be used with a C-order API (like openPMD)
  std::reverse( u.begin(), u.end() );

  return u;
}

/** \brief
 * Convert Real* pointer to a std::vector<double>,
 * and reverse the order of the elements
 * (used for compatibility with the openPMD API)
 */
std::vector<double>
getReversedVec( const Real* v )
{
  // Convert Real* v to and std::vector u
  std::vector<double> u = {
    AMREX_D_DECL(
                 static_cast<double>(v[0]),
                 static_cast<double>(v[1]),
                 static_cast<double>(v[2])
                 )
  };
  // Reverse the order of elements, if v corresponds to the indices of a
  // Fortran-order array (like an AMReX FArrayBox)
  // but u is intended to be used with a C-order API (like openPMD)
  std::reverse( u.begin(), u.end() );

  return u;
}

#ifdef WARPX_DIM_RZ
void
ConstructTotalRZVectorField (const std::array< std::unique_ptr<MultiFab>, 3 >& vector_total,
                             const std::array< std::unique_ptr<MultiFab>, 3 >& vector_field)
{
    // Sum over the real components, giving quantity at theta=0
    MultiFab::Copy(*vector_total[0], *vector_field[0], 0, 0, 1, vector_field[0]->nGrowVect());
    MultiFab::Copy(*vector_total[1], *vector_field[1], 0, 0, 1, vector_field[1]->nGrowVect());
    MultiFab::Copy(*vector_total[2], *vector_field[2], 0, 0, 1, vector_field[2]->nGrowVect());
    for (int ic=1 ; ic < vector_field[0]->nComp() ; ic += 2) {
        MultiFab::Add(*vector_total[0], *vector_field[0], ic, 0, 1, vector_field[0]->nGrowVect());
        MultiFab::Add(*vector_total[1], *vector_field[1], ic, 0, 1, vector_field[1]->nGrowVect());
        MultiFab::Add(*vector_total[2], *vector_field[2], ic, 0, 1, vector_field[2]->nGrowVect());
    }
}

void
ConstructTotalRZScalarField (MultiFab& scalar_total,
                            const MultiFab& scalar_field)
{
    // Sum over the real components, giving quantity at theta=0
    MultiFab::Copy(scalar_total, scalar_field, 0, 0, 1, scalar_field.nGrowVect());
    for (int ic=1 ; ic < scalar_field.nComp() ; ic += 2) {
        MultiFab::Add(scalar_total, scalar_field, ic, 0, 1, scalar_field.nGrowVect());
    }
}
#endif

/** \brief Takes an array of 3 MultiFab `vector_field`
 * (representing the x, y, z components of a vector),
 * averages it to the cell center, and stores the
 * resulting MultiFab in mf_avg (in the components dcomp to dcomp+2)
 * Should only be used for BTD now.
 */
void
AverageAndPackVectorField( MultiFab& mf_avg,
                           const std::array< std::unique_ptr<MultiFab>, 3 >& vector_field,
                           const DistributionMapping& dm,
                           const int dcomp, const IntVect ngrow )
{
#ifndef WARPX_DIM_RZ
    (void)dm;
#endif

#ifdef WARPX_DIM_RZ
    // Note that vector_total is declared in the same way as
    // vector_field so that it can be handled the same way.
    std::array<std::unique_ptr<MultiFab>,3> vector_total;
    if (vector_field[0]->nComp() > 1) {
        // With the RZ solver, if there are more than one component, the total
        // fields needs to be constructed in temporary MultiFabs.
        vector_total[0] = std::make_unique<MultiFab>(vector_field[0]->boxArray(), dm, 1, vector_field[0]->nGrowVect());
        vector_total[1] = std::make_unique<MultiFab>(vector_field[1]->boxArray(), dm, 1, vector_field[1]->nGrowVect());
        vector_total[2] = std::make_unique<MultiFab>(vector_field[2]->boxArray(), dm, 1, vector_field[2]->nGrowVect());
        ConstructTotalRZVectorField(vector_total, vector_field);
    } else {
        // Create aliases of the MultiFabs
        vector_total[0] = std::make_unique<MultiFab>(*vector_field[0], amrex::make_alias, 0, 1);
        vector_total[1] = std::make_unique<MultiFab>(*vector_field[1], amrex::make_alias, 0, 1);
        vector_total[2] = std::make_unique<MultiFab>(*vector_field[2], amrex::make_alias, 0, 1);
    }
#else
    const std::array<std::unique_ptr<MultiFab>,3> &vector_total = vector_field;
#endif

    ablastr::coarsen::sample::Coarsen(mf_avg, *(vector_total[0]), dcomp  , 0, 1, ngrow );
    ablastr::coarsen::sample::Coarsen(mf_avg, *(vector_total[1]), dcomp + 1, 0, 1, ngrow );
    ablastr::coarsen::sample::Coarsen(mf_avg, *(vector_total[2]), dcomp + 2, 0, 1, ngrow );
}

/** \brief Take a MultiFab `scalar_field`
 * averages it to the cell center, and stores the
 * resulting MultiFab in mf_avg (in the components dcomp)
 */
void
AverageAndPackScalarField (MultiFab& mf_avg,
                           const MultiFab & scalar_field,
                           const DistributionMapping& dm,
                           const int dcomp, const IntVect ngrow )
{
    const MultiFab *scalar_total = &scalar_field;

#ifdef WARPX_DIM_RZ
    MultiFab tmp;
    if (scalar_field.nComp() > 1) {
        // With the RZ solver, there are more than one component, so the total
        // fields needs to be constructed in temporary a MultiFab.
        tmp.define(scalar_field.boxArray(), dm, 1, scalar_field.nGrowVect());
        ConstructTotalRZScalarField(tmp, scalar_field);
        scalar_total = &tmp;
    }
#else
    amrex::ignore_unused(dm);
#endif

    // Check the type of staggering of the 3-component `vector_field`
    // and average accordingly:
    // - Fully cell-centered field (no average needed; simply copy)
    if ( scalar_total->is_cell_centered() ){
        MultiFab::Copy( mf_avg, *scalar_total, 0, dcomp, 1, ngrow);
    } else if ( scalar_total->is_nodal() ){
        // - Fully nodal
        ablastr::coarsen::sample::Coarsen(mf_avg, *scalar_total, dcomp, 0, 1, ngrow );
    } else {
        WARPX_ABORT_WITH_MESSAGE("Unknown staggering.");
    }
}
