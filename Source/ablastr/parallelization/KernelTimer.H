/* Copyright 2019-2020 Michael Rowan, Axel Huebl, Kevin Gott
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_KERNELTIMER_H_
#define ABLASTR_KERNELTIMER_H_

#include "ablastr/utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_BLassert.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <climits>

namespace ablastr::parallelization
{

/**
 * \brief Defines a timer object to be used on GPU; measures summed thread cycles.
 */
class KernelTimer
{
public:
    /** Constructor.
     * \param[in] do_timing Controls whether timer is active.
     * \param[in,out] cost Pointer to cost which holds summed thread cycles
     * (for performance, it is recommended to allocate pinned host memory).
     */
    AMREX_GPU_DEVICE
    KernelTimer
    (const bool do_timing, amrex::Real* cost)
#if (defined AMREX_USE_GPU)
        : m_do_timing(do_timing)
#endif
    {
#if (defined AMREX_USE_GPU)
    if (m_do_timing && cost) {
        m_cost = cost;
#   if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
        // Start the timer
        m_wt = clock64();

#   elif defined(AMREX_USE_DPCPP)
        // To be updated
        ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(m_do_timing == false,
            "KernelTimer not yet supported for this hardware.");
#   endif
    }
#else  // AMREX_USE_GPU
        amrex::ignore_unused(do_timing, cost);
#endif // AMREX_USE_GPU
    }

    //! Destructor.
#if (defined AMREX_USE_GPU)
#   if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
        AMREX_GPU_DEVICE
        ~KernelTimer ()
        {
            if (m_do_timing && m_cost) {
                m_wt = clock64() - m_wt;
                amrex::Gpu::Atomic::Add( m_cost, amrex::Real(m_wt));
            }
        }
#   elif defined(AMREX_USE_DPCPP)
        // To be updated
        AMREX_GPU_DEVICE
        ~KernelTimer () = default;
#   endif

#else
~KernelTimer () = default;
#endif //AMREX_USE_GPU


    KernelTimer ( KernelTimer const &)             = default;
    KernelTimer& operator= ( KernelTimer const & ) = default;
    KernelTimer ( KernelTimer&& )                  = default;
    KernelTimer& operator= ( KernelTimer&& )       = default;

#if (defined AMREX_USE_GPU)
private:
    //! Stores whether kernel timer is active.
    bool m_do_timing;

    //! Location in which to accumulate costs from all threads.
    amrex::Real* m_cost = nullptr;

    //! Store the time difference (cost) from a single thread.
    long long int m_wt;
#endif //AMREX_USE_GPU
};

} // namespace ablastr::parallelization

#endif // ABLASTR_KERNELTIMER_H_
