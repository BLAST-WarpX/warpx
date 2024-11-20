/* Copyright 2019-2020 Andrew Myers, David Grote, Maxence Thevenet
 * Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_COMPLEX_H_
#define WARPX_COMPLEX_H_

#ifdef WARPX_USE_FFT
#  include <ablastr/math/fft/AnyFFT.H>
#endif

#include <AMReX_Gpu.H>
#include <AMReX_GpuComplex.H>
#include <AMReX_REAL.H>

#include <complex>

// Defines a complex type on GPU & CPU
using Complex = amrex::GpuComplex<amrex::Real>;

#ifdef WARPX_USE_FFT
static_assert(sizeof(Complex) == sizeof(ablastr::math::anyfft::Complex),
    "The complex type in WarpX and the FFT library do not match.");
#endif

static_assert(sizeof(Complex) == sizeof(amrex::Real[2]),
              "Unexpected complex type.");

#endif //WARPX_COMPLEX_H_
