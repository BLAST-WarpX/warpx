/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ScopedTimeTracker.H"

#include "Utils/TextMsg.H"

#include <AMReX_GpuDevice.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_Utility.H>

namespace warpx::load_balancing
{

    bool ScopedTimeTracker::do_tracking = false;
    amrex::Vector<amrex::Vector<amrex::Real>> ScopedTimeTracker::all_times = amrex::Vector<amrex::Vector<amrex::Real>>{};

    [[nodiscard]]
    const amrex::Vector<amrex::Vector<amrex::Real>>& ScopedTimeTracker::view_tracked_times ()
    {
        return ScopedTimeTracker::all_times;
    }

    [[nodiscard]]
    bool ScopedTimeTracker::enabled ()
    {
        return ScopedTimeTracker::do_tracking;
    }

    ScopedTimeTracker::ScopedTimeTracker (
        const int lev, const int mfi_iter_index, const bool enabled_local):
        m_lev{lev},
        m_index{mfi_iter_index},
        m_enabled_local{enabled_local}
    {
        if (ScopedTimeTracker::do_tracking && m_enabled_local)
        {
            amrex::Gpu::synchronize();
            m_start_time = static_cast<amrex::Real>(amrex::second());
        }
    }

    ScopedTimeTracker::~ScopedTimeTracker ()
    {
        record_time();
    }

    void ScopedTimeTracker::stop_and_record ()
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_stopped,
            "ScopedTimeTracker can't be stopped multiple times!");

        record_time();
        m_stopped = true;
    }

    void ScopedTimeTracker::toggle_tracking(const bool do_tracking_flag)
    {
        ScopedTimeTracker::do_tracking = do_tracking_flag;
    }

    void ScopedTimeTracker::resize (const amrex::Vector<std::size_t>& max_index_per_level)
    {
        const auto max_index_per_level_size = max_index_per_level.size();
        all_times.resize(max_index_per_level_size);
        for (int i = 0; i < max_index_per_level_size; ++i)
        {
            all_times[i].resize(max_index_per_level[i]);
        }
    }

    void ScopedTimeTracker::reset_values ()
    {
        for (auto& lev_data : ScopedTimeTracker::all_times){
            for (auto& lev_index_data : lev_data){
                lev_index_data = 0.0;
            }
        }
    }

    void ScopedTimeTracker::reset_values (const int lev)
    {
        for (auto& lev_index_data : ScopedTimeTracker::all_times[lev]){
            lev_index_data = 0.0;
        }
    }

    void ScopedTimeTracker::record_time ()
    {
        if (ScopedTimeTracker::do_tracking && m_enabled_local && (!m_stopped))
        {
            amrex::Gpu::synchronize();
            const auto end_time = static_cast<amrex::Real>(amrex::second());
            const auto wall_time = end_time - m_start_time;
            amrex::HostDevice::Atomic::Add(&all_times[m_lev][m_index], wall_time);
        }
    }

    [[nodiscard]]
    ScopedTimeTracker get_scoped_time_tracker (
        int lev, int mfi_iter_index, bool enabled_local)
    {
        return ScopedTimeTracker {lev, mfi_iter_index, enabled_local};
    }
}
