/* Copyright 2019 Axel Huebl, David Grote, Igor Andriyash
 * Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_FieldIO_H_
#define WARPX_FieldIO_H_

#include <AMReX_REAL.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

void
AverageAndPackVectorField( amrex::MultiFab& mf_avg,
                         const std::array< std::unique_ptr<amrex::MultiFab>, 3 >& vector_field,
                         const amrex::DistributionMapping& dm,
                         int dcomp, amrex::IntVect ngrow );

void
AverageAndPackScalarField( amrex::MultiFab& mf_avg,
                         const amrex::MultiFab & scalar_field,
                         const amrex::DistributionMapping& dm,
                         int dcomp, amrex::IntVect ngrow );

/** Convert amrex IntVect to vector of `uint64`, reverse if requested
 *
 * @param[in] v vector to convert
 * @param[in] reverse (optional) boolean indicating whether to reverse ``v``
 * @return converted vector of type `uint64`, reversed if requested
 */
std::vector<std::uint64_t>
getVec( const amrex::IntVect& v, bool reverse = false);

/** Convert vector of `amrex Reals` to vector of `double`, reverse if requested
 *
 * @param[in] v vector to convert
 * @param[in] reverse (optional) boolean indicating whether to reverse ``v``
 * @return converted vector of type `amrex::Real`, reversed if requested
 */
std::vector<double>
getVec( const amrex::Real* v, bool reverse = false );

std::vector<std::uint64_t>
getReversedVec( const amrex::IntVect& v );

std::vector<double>
getReversedVec( const amrex::Real* v );

#endif // WARPX_FieldIO_H_
