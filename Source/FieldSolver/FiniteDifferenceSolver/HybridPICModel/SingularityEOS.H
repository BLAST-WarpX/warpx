/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_SINGULARITYEOS_H_
#define WARPX_SINGULARITYEOS_H_

#include <type_traits>

#include <AMReX.H>
#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>
#include <memory>
#include <vector>

#ifdef WARPX_USE_SINGULARITY_EOS

#   if defined(_PORTABILITY_HPP_)
#       error "WarpX Singularity-EOS adapter must be included before ports-of-call portability.hpp"
#   endif
#   if defined(Real)
#       error "WarpX Singularity-EOS adapter requires the Real macro to be undefined"
#   endif

// This is WarpX's sole permitted gateway to Singularity-EOS and ports-of-call;
// native vendor headers must not precede it or be mixed with adapted definitions
// across translation units. ports-of-call declares a global Real typedef, so
// contain it while importing Singularity-EOS to avoid colliding with
// AMReX's amrex::Real.
#   define Real WarpxSingularityEosReal
#   include <singularity-eos/eos/eos_spiner_rho_temp.hpp>
#   include <singularity-eos/eos/eos_spiner_common.hpp>
#   include <singularity-utils/fast-math/logs.hpp>
#   include <singularity-utils/sp5/singularity_eos_sp5.hpp>
#   include <singularity-utils/spiner_params.hpp>
#   undef Real

static_assert(std::is_same_v<WarpxSingularityEosReal, double>);

#   if defined(PORTABILITY_STRATEGY_NONE)
       // Host-resident Spiner tables remain supported for CPU/OpenMP builds.
#   elif defined(PORTABILITY_STRATEGY_KOKKOS) && defined(KOKKOS_ENABLE_CUDA)
       // CUDA uses the Kokkos portability strategy so Spiner table methods
       // and data boxes are device-callable. HIP and SYCL are not qualified.
#   else
#       error "WarpX table EOS supports host-only or Kokkos-CUDA Singularity-EOS"
#   endif

#ifdef AMREX_USE_GPU
namespace warpx::materials
{
    inline void InitializeTableDeviceRuntime ()
    {
        // Respect an embedding application's Kokkos lifetime and the device
        // already selected by AMReX (including rank-local GPU assignment).
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize(Kokkos::InitializationSettings{}
                .set_device_id(amrex::Gpu::Device::deviceId()));
            amrex::ExecOnFinalize([] { Kokkos::finalize(); });
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            Kokkos::Cuda().cuda_device() == amrex::Gpu::Device::deviceId(),
            "Table EOS requires Kokkos and AMReX to select the same CUDA device.");
    }

    /** Owning device table data, with host-constructed handles in managed
     * memory. Spiner handles are nontrivial C++ objects: do not memcpy them
     * through a PODVector. Host lookups use separate host-resident tables. */
    class DeviceSpinerTables
    {
    public:
        using Table = singularity::SpinerEOSDependsRhoT;
        DeviceSpinerTables () = default;
        DeviceSpinerTables (DeviceSpinerTables const&) = delete;
        DeviceSpinerTables& operator= (DeviceSpinerTables const&) = delete;
        ~DeviceSpinerTables () { clear(); }

        void assign (std::vector<Table>& host)
        {
            clear();
            InitializeTableDeviceRuntime();
            m_data = static_cast<Table*>(
                amrex::The_Managed_Arena()->alloc(host.size() * sizeof(Table)));
            for (auto& table : host) {
                std::construct_at(m_data + m_size, table.GetOnDevice());
                ++m_size;
            }
            Kokkos::fence();
        }

        void clear () noexcept
        {
            if (!m_data) { return; }
            amrex::Gpu::synchronize();
            for (std::size_t i = 0; i < m_size; ++i) {
                m_data[i].Finalize();
                std::destroy_at(m_data + i);
            }
            amrex::The_Managed_Arena()->free(m_data);
            m_data = nullptr;
            m_size = 0;
        }

        Table const* data () const noexcept { return m_data; }

    private:
        Table* m_data = nullptr;
        std::size_t m_size = 0;
    };
}
#endif

#endif // WARPX_USE_SINGULARITY_EOS

#endif // WARPX_SINGULARITYEOS_H_
