/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_LOAD_BALANCING_SCOPED_TIME_TRACKER_H_
#define WARPX_LOAD_BALANCING_SCOPED_TIME_TRACKER_H_

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_LayoutData.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

class WarpX;

namespace warpx::load_balancing
{
    class ScopedTimeTracker
    {

    public:
        [[nodiscard]]
        static const amrex::Vector<amrex::LayoutData<amrex::Real>>& view_tracked_times ();

        [[nodiscard]]
        static bool enabled ();

        ~ScopedTimeTracker();

        void stop_and_record ();

        //TimeTracker must be uncopyable and unmovable
        ScopedTimeTracker(const ScopedTimeTracker&) = delete;
        ScopedTimeTracker(ScopedTimeTracker&&) = delete;
        ScopedTimeTracker& operator=(const ScopedTimeTracker&) = delete;
        ScopedTimeTracker& operator=(ScopedTimeTracker&&) = delete;

    private:
        static void toggle_tracking (bool do_tracking_flag);
        static void resize (int max_num_levels);
        static void resize_level (int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm);
        static void reset_values ();
        static void reset_values (int lev);

        static bool do_tracking;
        static amrex::Vector<amrex::LayoutData<amrex::Real>> all_times;

        ScopedTimeTracker(int lev, int mfi_iter_index, bool enabled_local);

        void record_time ();

        int m_lev;
        int m_index;

        amrex::Real m_start_time;

        bool m_enabled_local;
        bool m_stopped  = false;

        friend class ::WarpX;

        friend
        ScopedTimeTracker get_scoped_time_tracker (int lev, int mfi_iter_index, bool enabled_local);
    };

    [[nodiscard]]
    ScopedTimeTracker get_scoped_time_tracker (int lev, int mfi_iter_index, bool enabled_local = true);
}

#endif //WARPX_LOAD_BALANCING_SCOPED_TIME_TRACKER_H_
