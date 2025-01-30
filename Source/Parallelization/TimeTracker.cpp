/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "TimeTracker.H"

#include <AMReX_GpuDevice.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_Utility.H>

bool warpx::parallelization::TimeTracker::do_tracking =
    false;
amrex::Vector<amrex::Vector<amrex::Real>> warpx::parallelization::TimeTracker::all_times =
    amrex::Vector<amrex::Vector<amrex::Real>>{};

namespace warpx::parallelization
{
     [[nodiscard]]
    TimeTracker track_time_until_out_of_scope (const int lev, const std::size_t index, const bool enabled_local)
    {
        return TimeTracker{lev, index, enabled_local};
    }

    void TimeTracker::toggle_tracking(const bool do_tracking_flag)
    {
        TimeTracker::do_tracking = do_tracking_flag;
    }

    void TimeTracker::resize (const amrex::Vector<std::size_t>& max_index_per_level)
    {
        const auto max_index_per_level_size = max_index_per_level.size();
        all_times.resize(max_index_per_level_size);
        for (int i = 0; i < max_index_per_level_size; ++i)
        {
            all_times[i].resize(max_index_per_level[i]);
        }
    }

    void TimeTracker::reset_values()
    {
        for (auto& lev_data : TimeTracker::all_times){
            for (auto& lev_index_data : lev_data){
                lev_index_data = 0.0;
            }
        }
    }

    TimeTracker::TimeTracker(const int lev, const std::size_t mfi_iter_index, const bool enabled_local)
        :m_lev{lev}, m_index{mfi_iter_index}, m_enabled_local{enabled_local}
    {
        if (TimeTracker::do_tracking && m_enabled_local)
        {
            amrex::Gpu::synchronize();
            m_start_time = static_cast<amrex::Real>(amrex::second());
        }
    }

    TimeTracker::~TimeTracker()
    {
        if (TimeTracker::do_tracking && m_enabled_local)
        {
            amrex::Gpu::synchronize();
            const auto end_time = static_cast<amrex::Real>(amrex::second());
            const auto wall_time = end_time - m_start_time;
            amrex::HostDevice::Atomic::Add(&all_times[m_lev][m_index], wall_time);
        }
    }

}
