/* Copyright 2019-2023
 *
 * This file is part of ABLASTR.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "AnyFFT.H"

#include "ablastr/utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

namespace ablastr::math::anyfft
{

    void setup(){/*nothing to do*/}

    void cleanup(){/*nothing to do*/}

#ifdef AMREX_USE_FLOAT
    const auto VendorCreatePlanR2C3D = fftwf_plan_dft_r2c_3d;
    const auto VendorCreatePlanC2R3D = fftwf_plan_dft_c2r_3d;
    const auto VendorCreatePlanR2C2D = fftwf_plan_dft_r2c_2d;
    const auto VendorCreatePlanC2R2D = fftwf_plan_dft_c2r_2d;
    const auto VendorCreatePlanR2C1D = fftwf_plan_dft_r2c_1d;
    const auto VendorCreatePlanC2R1D = fftwf_plan_dft_c2r_1d;
#else
    const auto VendorCreatePlanR2C3D = fftw_plan_dft_r2c_3d;
    const auto VendorCreatePlanC2R3D = fftw_plan_dft_c2r_3d;
    const auto VendorCreatePlanR2C2D = fftw_plan_dft_r2c_2d;
    const auto VendorCreatePlanC2R2D = fftw_plan_dft_c2r_2d;
    const auto VendorCreatePlanR2C1D = fftw_plan_dft_r2c_1d;
    const auto VendorCreatePlanC2R1D = fftw_plan_dft_c2r_1d;
#endif

    FFTplan CreatePlan(const amrex::IntVect& real_size, amrex::Real * const real_array,
                       Complex * const complex_array, const direction dir, const int dim)
    {
        FFTplan fft_plan;

#if defined(AMREX_USE_OMP) && defined(WarpX_FFTW_OMP)
#   ifdef AMREX_USE_FLOAT
        fftwf_init_threads();
        fftwf_plan_with_nthreads(omp_get_max_threads());
#   else
        fftw_init_threads();
        fftw_plan_with_nthreads(omp_get_max_threads());
#   endif
#endif

        // Initialize fft_plan.m_plan with the vendor fft plan.
        // Swap dimensions: AMReX FAB are Fortran-order but FFTW is C-order
        if (dir == direction::R2C){
            if (dim == 3) {
                fft_plan.m_plan = VendorCreatePlanR2C3D(
                    real_size[2], real_size[1], real_size[0], real_array, complex_array, FFTW_ESTIMATE);
            } else if (dim == 2) {
                fft_plan.m_plan = VendorCreatePlanR2C2D(
                    real_size[1], real_size[0], real_array, complex_array, FFTW_ESTIMATE);
            } else if (dim == 1) {
                fft_plan.m_plan = VendorCreatePlanR2C1D(
                    real_size[0], real_array, complex_array, FFTW_ESTIMATE);
            } else {
                ABLASTR_ABORT_WITH_MESSAGE(
                    "only dim=1 and dim=2 and dim=3 have been implemented");
            }
        } else if (dir == direction::C2R){
            if (dim == 3) {
                fft_plan.m_plan = VendorCreatePlanC2R3D(
                    real_size[2], real_size[1], real_size[0], complex_array, real_array, FFTW_ESTIMATE);
            } else if (dim == 2) {
                fft_plan.m_plan = VendorCreatePlanC2R2D(
                    real_size[1], real_size[0], complex_array, real_array, FFTW_ESTIMATE);
            } else if (dim == 1) {
                fft_plan.m_plan = VendorCreatePlanC2R1D(
                    real_size[0], complex_array, real_array, FFTW_ESTIMATE);
            } else {
                ABLASTR_ABORT_WITH_MESSAGE(
                    "only dim=1 and dim=2 and dim=3 have been implemented.");
            }
        }

        // Store meta-data in fft_plan
        fft_plan.m_real_array = real_array;
        fft_plan.m_complex_array = complex_array;
        fft_plan.m_dir = dir;
        fft_plan.m_dim = dim;

        return fft_plan;
    }

    void DestroyPlan(FFTplan& fft_plan)
    {
#  ifdef AMREX_USE_FLOAT
        fftwf_destroy_plan( fft_plan.m_plan );
#  else
        fftw_destroy_plan( fft_plan.m_plan );
#  endif
    }

    void Execute(FFTplan& fft_plan){
#  ifdef AMREX_USE_FLOAT
        fftwf_execute( fft_plan.m_plan );
#  else
        fftw_execute( fft_plan.m_plan );
#  endif
    }
}
