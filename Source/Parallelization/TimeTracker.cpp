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

using namespace warpx::parallelization;

[[nodiscard]]
TimeTracker track_time_until_out_of_scope (const int lev, const std::size_t index)
{
    return TimeTracker{lev, index};
}

void TimeTrackerWarpXInterface::toggle_tracking (const bool do_tracking)
{
    TimeTracker::toggle_tracking(do_tracking);
}

void TimeTrackerWarpXInterface::resize (const amrex::Vector<std::size_t>& max_index_per_level)
{
    TimeTracker::resize(max_index_per_level);
}

void TimeTrackerWarpXInterface::reset_values ()
{
    TimeTracker::reset_values();
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

TimeTracker::TimeTracker(const int lev, const std::size_t mfi_iter_index)
    :m_lev{lev}, m_index{mfi_iter_index}
{
    if (TimeTracker::do_tracking)
    {
        amrex::Gpu::synchronize();
        m_start_time = static_cast<amrex::Real>(amrex::second());
    }
}

TimeTracker::~TimeTracker()
{
    if (TimeTracker::do_tracking)
    {
        amrex::Gpu::synchronize();
        const auto end_time = static_cast<amrex::Real>(amrex::second());
        const auto wall_time = end_time - m_start_time;
        amrex::HostDevice::Atomic::Add(&all_times[m_lev][m_index], wall_time);
    }
}
