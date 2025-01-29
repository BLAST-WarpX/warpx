/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "TimeTracker.H"

#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_Utility.H>

using namespace warpx::parallelization;

[[nodiscard]]
TimeTracker track_time_until_out_of_scope (const int lev, const std::size_t index)
{
    return TimeTracker{lev, index};
}

[[nodiscard]]
std::map<LevelIndex,amrex::Real> get_tracked_times ()
{
    return AllTimes::get_instance().get_times();
}

[[nodiscard]]
bool is_time_tracking_enabled ()
{
    return AllTimes::get_instance().enabled();
}

void TimeTrackerWarpXInterface::toggle_tracking (const bool do_tracking)
{
    AllTimes::get_instance().toggle_tracking(do_tracking);
}

void TimeTrackerWarpXInterface::reset ()
{
    AllTimes::get_instance().reset();
}

void TimeTrackerWarpXInterface::reset_values ()
{
    AllTimes::get_instance().reset_values();
}

[[nodiscard]]
AllTimes& AllTimes::get_instance()
{
    static auto the_unique_instance = AllTimes{};
    return the_unique_instance;
}

[[nodiscard]]
std::map<LevelIndex,amrex::Real> AllTimes::get_times () const
{
    return m_all_times;
}

AllTimes::AllTimes()
{}

void AllTimes::toggle_tracking(const bool do_tracking)
{
    m_do_tracking = do_tracking;
}

void AllTimes::reset()
{
    m_all_times.clear();
}

void AllTimes::reset_values()
{
    for (auto& el : m_all_times){
        el.second = 0.0;
    }
}

TimeTracker::TimeTracker(const int lev, const std::size_t mfi_iter_index)
    :m_lev_index{lev, mfi_iter_index}
{
    if (AllTimes::get_instance().enabled())
    {
        amrex::Gpu::synchronize();
        m_start_time = static_cast<amrex::Real>(amrex::second());
    }
}

TimeTracker::~TimeTracker()
{
    if (auto& all_times = AllTimes::get_instance(); all_times.enabled())
    {
        amrex::Gpu::synchronize();
        const auto end_time = static_cast<amrex::Real>(amrex::second());
        const auto wall_time = end_time - m_start_time;
        amrex::HostDevice::Atomic::Add( &all_times.m_all_times[m_lev_index], wall_time);
    }
}
