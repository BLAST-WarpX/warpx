/* Copyright 2019-2020 Andrew Myers, David Grote, Maxence Thevenet
 * Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_MATH_COMPLEX_H_
#define ABLASTR_MATH_COMPLEX_H_

#ifdef ABLASTR_USE_FFT
#  include "ablastr/math/fft/AnyFFT.H"
#endif

#include <AMReX_GpuComplex.H>
#include <AMReX_REAL.H>

namespace ablastr::math
{
    // Defines a complex type on GPU & CPU
    using Complex = amrex::GpuComplex<amrex::Real>;

#ifdef ABLASTR_USE_FFT
    static_assert(sizeof(Complex) == sizeof(ablastr::math::anyfft::Complex),
    "The complex types in ablastr/math/Complex.H and ablastr/math/fft/AnyFFT.H do not match.");
#endif

    static_assert(sizeof(Complex) == sizeof(amrex::Real[2]),
    "Unexpected complex type.");
}

#endif //ABLASTR_MATH_COMPLEX_H_
