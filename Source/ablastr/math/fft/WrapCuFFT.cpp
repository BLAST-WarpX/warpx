/* Copyright 2019-2023
 *
 * This file is part of ABLASTR.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "AnyFFT.H"

#include "ablastr/utils/TextMsg.H"
#include "ablastr/profiler/ProfilerWrapper.H"

namespace ablastr::math::anyfft
{

    void setup(){/*nothing to do*/}

    void cleanup(){/*nothing to do*/}

#ifdef AMREX_USE_FLOAT
    cufftType VendorR2C = CUFFT_R2C;
    cufftType VendorC2R = CUFFT_C2R;
#else
    cufftType VendorR2C = CUFFT_D2Z;
    cufftType VendorC2R = CUFFT_Z2D;
#endif

    std::string cufftErrorToString (const cufftResult& err);

    FFTplan CreatePlan(const amrex::IntVect& real_size, amrex::Real * const real_array,
                       Complex * const complex_array, const direction dir, const int dim)
    {
        FFTplan fft_plan;
        ABLASTR_PROFILE("ablastr::math::anyfft::CreatePlan");

        // Initialize fft_plan.m_plan with the vendor fft plan.
        cufftResult result;
        if (dir == direction::R2C){
            if (dim == 3) {
                result = cufftPlan3d(
                    &(fft_plan.m_plan), real_size[2], real_size[1], real_size[0], VendorR2C);
            } else if (dim == 2) {
                result = cufftPlan2d(
                    &(fft_plan.m_plan), real_size[1], real_size[0], VendorR2C);
            } else if (dim == 1) {
                result = cufftPlan1d(
                    &(fft_plan.m_plan), real_size[0], VendorR2C, 1);
            } else {
                ABLASTR_ABORT_WITH_MESSAGE("only dim=1 and dim=2 and dim=3 have been implemented");
            }
        } else {
            if (dim == 3) {
                result = cufftPlan3d(
                    &(fft_plan.m_plan), real_size[2], real_size[1], real_size[0], VendorC2R);
            } else if (dim == 2) {
                result = cufftPlan2d(
                    &(fft_plan.m_plan), real_size[1], real_size[0], VendorC2R);
            } else if (dim == 1) {
                result = cufftPlan1d(
                    &(fft_plan.m_plan), real_size[0], VendorC2R, 1);
            } else {
                ABLASTR_ABORT_WITH_MESSAGE("only dim=2 and dim=3 have been implemented");
            }
        }

        ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(result == CUFFT_SUCCESS,
            "cufftplan failed! Error: " + cufftErrorToString(result));

        // Store meta-data in fft_plan
        fft_plan.m_real_array = real_array;
        fft_plan.m_complex_array = complex_array;
        fft_plan.m_dir = dir;
        fft_plan.m_dim = dim;

        return fft_plan;
    }

    void DestroyPlan(FFTplan& fft_plan)
    {
        ABLASTR_PROFILE("ablastr::math::anyfft::DestroyPlan");
        cufftDestroy( fft_plan.m_plan );
    }

    void Execute(FFTplan& fft_plan){
        ABLASTR_PROFILE("ablastr::math::anyfft::Execute");
        // make sure that this is done on the same GPU stream as the above copy
        cudaStream_t stream = amrex::Gpu::Device::cudaStream();
        cufftSetStream ( fft_plan.m_plan, stream);
        cufftResult result;
        if (fft_plan.m_dir == direction::R2C){
#ifdef AMREX_USE_FLOAT
            result = cufftExecR2C(fft_plan.m_plan, fft_plan.m_real_array, fft_plan.m_complex_array);
#else
            result = cufftExecD2Z(fft_plan.m_plan, fft_plan.m_real_array, fft_plan.m_complex_array);
#endif
        } else if (fft_plan.m_dir == direction::C2R){
#ifdef AMREX_USE_FLOAT
            result = cufftExecC2R(fft_plan.m_plan, fft_plan.m_complex_array, fft_plan.m_real_array);
#else
            result = cufftExecZ2D(fft_plan.m_plan, fft_plan.m_complex_array, fft_plan.m_real_array);
#endif
        } else {
            ABLASTR_ABORT_WITH_MESSAGE(
                "direction must be FFTplan::direction::R2C or FFTplan::direction::C2R");
        }
        if ( result != CUFFT_SUCCESS ) {
            ABLASTR_ABORT_WITH_MESSAGE(
                "forward transform using cufftExec failed ! Error: "
                +cufftErrorToString(result));
        }
    }

    /** \brief This method converts a cufftResult
     * into the corresponding string
     *
     * @param[in] err a cufftResult
     * @return an std::string
     */
    std::string cufftErrorToString (const cufftResult& err)
    {
        const auto res2string = std::map<cufftResult, std::string>{
            {CUFFT_SUCCESS, "CUFFT_SUCCESS"},
            {CUFFT_INVALID_PLAN,"CUFFT_INVALID_PLAN"},
            {CUFFT_ALLOC_FAILED,"CUFFT_ALLOC_FAILED"},
            {CUFFT_INVALID_TYPE,"CUFFT_INVALID_TYPE"},
            {CUFFT_INVALID_VALUE,"CUFFT_INVALID_VALUE"},
            {CUFFT_INTERNAL_ERROR,"CUFFT_INTERNAL_ERROR"},
            {CUFFT_EXEC_FAILED,"CUFFT_EXEC_FAILED"},
            {CUFFT_SETUP_FAILED,"CUFFT_SETUP_FAILED"},
            {CUFFT_INVALID_SIZE,"CUFFT_INVALID_SIZE"},
            {CUFFT_UNALIGNED_DATA,"CUFFT_UNALIGNED_DATA"}};

        const auto it = res2string.find(err);
        if(it != res2string.end()){
            return it->second;
        }
        else{
            return std::to_string(err) +
                " (unknown error code)";
        }
    }
}
