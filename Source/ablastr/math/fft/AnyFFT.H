/* Copyright 2019-2023
 *
 * This file is part of ABLASTR.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_ANYFFT_H_
#define ABLASTR_ANYFFT_H_

#ifdef ABLASTR_USE_FFT
#   include <AMReX_Config.H>
#   include <AMReX_GpuComplex.H>
#   include <AMReX_LayoutData.H>

#   if defined(AMREX_USE_CUDA)
#       include <cufft.h>
#       include <cuComplex.h>
#   elif defined(AMREX_USE_HIP)
#       if __has_include(<rocfft/rocfft.h>)  // ROCm 5.3+
#           include <rocfft/rocfft.h>
#       else
#           include <rocfft.h>
#       endif
#       include <hip/hip_complex.h>
#   elif defined(AMREX_USE_SYCL)
#       include <oneapi/mkl/dfti.hpp>
#   else
#       include <fftw3.h>
#   endif
#endif


/**
 * Wrapper around FFT libraries. The header file defines the API and the base types
 * (Complex and VendorFFTPlan), and the implementation for different FFT libraries is
 * done in different cpp files. This wrapper only depends on the underlying FFT library
 * AND on AMReX (There is no dependence on WarpX).
 */
namespace ablastr::math::anyfft
{

    /** This function is a wrapper around rocff_setup().
     *  It is a no-op in case rocfft is not used.
    */
    void setup();

    /** This function is a wrapper around rocff_cleanup().
     *  It is a no-op in case rocfft is not used.
    */
    void cleanup();

#ifdef ABLASTR_USE_FFT

    // First, define library-dependent types (complex, FFT plan)

    /** Complex type for FFT, depends on FFT library */
#   if defined(AMREX_USE_CUDA)
#       ifdef AMREX_USE_FLOAT
            using Complex = cuComplex;
#       else
            using Complex = cuDoubleComplex;
#       endif
#   elif defined(AMREX_USE_HIP)
#       ifdef AMREX_USE_FLOAT
            using Complex = float2;
#       else
            using Complex = double2;
#       endif
#   elif defined(AMREX_USE_SYCL)
        using Complex = amrex::GpuComplex<amrex::Real>;
#   else
#       ifdef AMREX_USE_FLOAT
            using Complex = fftwf_complex;
#       else
            using Complex = fftw_complex;
#       endif
#   endif

    /* Library-dependent multiply helpers */
#   if defined(AMREX_USE_CUDA)
#       ifdef AMREX_USE_FLOAT
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void multiply (Complex & c, Complex const & a, Complex const & b) { c = cuCmulf(a, b); }
#       else
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void multiply (Complex & c, Complex const & a, Complex const & b) { c = cuCmul(a, b); }
#       endif
#   elif defined(AMREX_USE_HIP)
    #   ifdef AMREX_USE_FLOAT
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void multiply (Complex & c, Complex const & a, Complex const & b) { c = hipCmulf(a, b); }
#       else
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void multiply (Complex & c, Complex const & a, Complex const & b) { c = hipCmul(a, b); }
#       endif
#   elif defined(AMREX_USE_SYCL)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void multiply (Complex & c, Complex const & a, Complex const & b) { c = a * b; }
#   else
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void multiply (Complex & c, Complex const & a, Complex const & b) {
        c[0] = a[0] * b[0] - a[1] * b[1];
        c[1] = a[0] * b[1] + a[1] * b[0];
    }
#   endif

    /** Library-dependent FFT plans type, which holds one fft plan per box
     * (plans are only initialized for the boxes that are owned by the local MPI rank).
     */
#   if defined(AMREX_USE_CUDA)
        using VendorFFTPlan = cufftHandle;
#   elif defined(AMREX_USE_HIP)
        using VendorFFTPlan = rocfft_plan;
#   elif defined(AMREX_USE_SYCL)
        using VendorFFTPlan = oneapi::mkl::dft::descriptor<
#       ifdef AMREX_USE_FLOAT
            oneapi::mkl::dft::precision::SINGLE,
#       else
            oneapi::mkl::dft::precision::DOUBLE,
#       endif
            oneapi::mkl::dft::domain::REAL> *;
        // dft::descriptor has no default ctor, so we have to use ptr.
#   else
#       ifdef AMREX_USE_FLOAT
            using VendorFFTPlan = fftwf_plan;
#       else
            using VendorFFTPlan = fftw_plan;
#       endif
#   endif

    // Second, define library-independent API

    /** Direction in which the FFT is performed. */
    enum struct direction {R2C, C2R};

    /** This struct contains the vendor FFT plan and additional metadata
     */
    struct FFTplan
    {
        amrex::Real* m_real_array; /**< pointer to real array */
        Complex* m_complex_array; /**< pointer to complex array */
        VendorFFTPlan m_plan; /**< Vendor FFT plan */
        direction m_dir;  /**< direction (C2R or R2C) */
        int m_dim; /**< Dimensionality of the FFT plan */
#ifdef AMREX_USE_SYCL
        amrex::gpuStream_t m_stream;
#endif
    };

    /** Collection of FFT plans, one FFTplan per box */
    using FFTplans = amrex::LayoutData<FFTplan>;

    /** \brief create FFT plan for the backend FFT library.
     * \param[in] real_size Size of the real array, along each dimension.
     *                      Only the first dim elements are used.
     * \param[out] real_array Real array from/to where R2C/C2R FFT is performed
     * \param[out] complex_array Complex array to/from where R2C/C2R FFT is performed
     * \param[in] dir direction, either R2C or C2R
     * \param[in] dim direction, number of dimensions of the arrays. Must be <= AMREX_SPACEDIM.
     */
    FFTplan CreatePlan(const amrex::IntVect& real_size, amrex::Real* real_array,
                       Complex* complex_array, direction dir, int dim);

    /** \brief Destroy library FFT plan.
     * \param[out] fft_plan plan to destroy
     */
    void DestroyPlan(FFTplan& fft_plan);

    /** \brief Perform FFT with backend library.
     * \param[out] fft_plan plan for which the FFT is performed
     */
    void Execute(FFTplan& fft_plan);

#endif

}

#endif // ABLASTR_ANYFFT_H_
