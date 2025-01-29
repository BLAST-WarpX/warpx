/* Copyright 2019 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_amrex_qed_wrapper_commons_h_
#define WARPX_amrex_qed_wrapper_commons_h_
/*
 * This header contains some common #define directives
 * used by the QED library wrappers and related components.
 * It also provides a vector type suitable to be used
 * as the building block  of the picsar QED lookup tables
 * when managed memory is not used.
 */

#include <AMReX_AmrCore.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Vector.H>

#include <vector>


/**
 * PICSAR uses PXRMP_GPU to decorate methods which should be
 * compiled for GPU. The user has to set it to the right value
 * (AMREX_GPU_DEVICE in this case).
 */
#define PXRMP_WITH_GPU
#define PXRMP_GPU_QUALIFIER AMREX_GPU_HOST_DEVICE
//_________________________


/**
 * PICSAR uses internally some specifiers analogous to
 * AMREX_RESTRICT and AMREX_FORCE_INLINE. These definitions
 * set the aforementioned specifiers to AMREX_RESTRICT and
 * AMREX_FORCE_INLINE.
 */
#define PXRMP_RESTRICT AMREX_RESTRICT
#define PXRMP_FORCE_INLINE AMREX_FORCE_INLINE
//_________________________


#ifdef AMREX_USE_GPU
/**
* This class provides a vector suitable to be used as the building block
* of the picsar QED lookup tables when managed memory is not used. It behaves as a
* std::vector<Real> when data is written into it, but, "under the hood", it copies data to
* an amrex::Gpu::DeviceVector. It provides a pointer to the Device data via the data() function.
*
* @tparam Real the floating point type to be used
*/
template <typename Real>
class PicsarQedVector : public std::vector<Real>
{
    using HV = std::vector<Real>;
    using AGDV = amrex::Gpu::DeviceVector<Real>;

    public:

    /**
    * All the arguments passed to PicsarQedVector constructor
    * are forwarded to the std::vector constructor.
    *
    * @tparam Args the constructor arguments
    */
    template<typename... Args>
    PicsarQedVector(Args&&... args) : HV(std::forward<Args>(args)...)
    {}

    /**
    * This function may be called in some places inside picsar_table classes.
    * It forces a copy of the CPU data to the GPU
    *
    */
    void pxr_sync()
    {
        m_device_data.resize(this->size());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
            this->begin(), this->end(), m_device_data.begin());
        amrex::Gpu::Device::streamSynchronize();
    }

    /**
    * This function returns a pointer to the raw data inside the
    * amrex::Gpu::DeviceVector<Real> m_device_data.
    * It is called to build table_views.
    *
    * @return a pointer to the raw device vector data
    */
    const Real* data() const
    {
        return m_device_data.data();
    }

    private:
    AGDV m_device_data;
};
#else

template <typename Real>
using PicsarQedVector = std::vector<Real>;

#endif

#endif //WARPX_amrex_qed_wrapper_commons_h_
