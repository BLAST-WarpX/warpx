/* Copyright 2019-2023
 *
 * This file is part of ABLASTR.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "AnyFFT.H"

#include "ablastr/utils/TextMsg.H"
#include "ablastr/profiler/ProfilerWrapper.H"

#include <cstdint>

namespace ablastr::math::anyfft
{

    void setup () {/*nothing to do*/}

    void cleanup () {/*nothing to do*/}

    FFTplan CreatePlan (const amrex::IntVect& real_size, amrex::Real * const real_array,
                        Complex * const complex_array, const direction dir, const int dim)
    {
        FFTplan fft_plan;
        ABLASTR_PROFILE("ablastr::math::anyfft::CreatePlan");

        // Initialize fft_plan.m_plan with the vendor fft plan.
        std::vector<std::int64_t> strides(dim+1);
        if (dim == 3) {
            fft_plan.m_plan = new std::remove_pointer_t<VendorFFTPlan>(
                {std::int64_t(real_size[2]),
                 std::int64_t(real_size[1]),
                 std::int64_t(real_size[0])});
            strides[0] = 0;
            strides[1] = real_size[0] * real_size[1];
            strides[2] = real_size[0];
            strides[3] = 1;
        } else if (dim == 2) {
            fft_plan.m_plan = new std::remove_pointer_t<VendorFFTPlan>(
                {std::int64_t(real_size[1]),
                 std::int64_t(real_size[0])});
            strides[0] = 0;
            strides[1] = real_size[0];
            strides[2] = 1;
        } else if (dim == 1) {
            strides[0] = 0;
            strides[1] = 1;
            fft_plan.m_plan = new std::remove_pointer_t<VendorFFTPlan>(
                std::int64_t(real_size[0]));
        } else {
            ABLASTR_ABORT_WITH_MESSAGE("only dim2 =1, dim=2 and dim=3 have been implemented");
        }

        fft_plan.m_plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                   DFTI_NOT_INPLACE);
        fft_plan.m_plan->set_value(oneapi::mkl::dft::config_param::FWD_STRIDES,
                                   strides.data());
        fft_plan.m_plan->commit(amrex::Gpu::Device::streamQueue());

        // Store meta-data in fft_plan
        fft_plan.m_real_array = real_array;
        fft_plan.m_complex_array = complex_array;
        fft_plan.m_dir = dir;
        fft_plan.m_dim = dim;
        fft_plan.m_stream = amrex::Gpu::gpuStream();

        return fft_plan;
    }

    void DestroyPlan (FFTplan& fft_plan)
    {
        delete fft_plan.m_plan;
    }

    void Execute (FFTplan& fft_plan)
    {
        if (!(fft_plan.m_stream == amrex::Gpu::gpuStream())) {
            amrex::Gpu::streamSynchronize();
        }

        sycl::event r;
        if (fft_plan.m_dir == direction::R2C) {
            r = oneapi::mkl::dft::compute_forward(
                *fft_plan.m_plan,
                fft_plan.m_real_array,
                reinterpret_cast<std::complex<amrex::Real>*>(fft_plan.m_complex_array));
        } else {
            r = oneapi::mkl::dft::compute_backward(
                *fft_plan.m_plan,
                reinterpret_cast<std::complex<amrex::Real>*>(fft_plan.m_complex_array),
                fft_plan.m_real_array);
        }
        r.wait();
    }
}
