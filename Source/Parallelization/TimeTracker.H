/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_PARALLELIZATION_TIME_TRACKER_H_
#define WARPX_PARALLELIZATION_TIME_TRACKER_H_

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <cstddef>

namespace warpx::parallelization
{
    class WarpX;

    class TimeTrackerWarpXInterface
    {
        private:
        static void toggle_tracking (bool do_tracking);
        static void resize (const amrex::Vector<std::size_t>& max_index_per_level);
        static void reset_values ();

        friend class WarpX;
    };

    class TimeTracker
    {

    public:
        [[nodiscard]]
        static const amrex::Vector<amrex::Vector<amrex::Real>>& view_tracked_times ();

        [[nodiscard]]
        static bool enabled ();

        TimeTracker(int lev, std::size_t mfi_iter_index);
        ~TimeTracker();

        //TimeTracker must be uncopyable and unmovable
        TimeTracker(const TimeTracker&) = delete;
        TimeTracker(TimeTracker&&) = delete;
        TimeTracker& operator=(const TimeTracker&) = delete;
        TimeTracker& operator=(TimeTracker&&) = delete;

        friend class TimeTrackerWarpXInterface;

    private:
        static void toggle_tracking (bool do_tracking_flag);
        static void resize (const amrex::Vector<std::size_t>& max_index_per_level);
        static void reset_values ();

        static bool do_tracking;
        static amrex::Vector<amrex::Vector<amrex::Real>> all_times;

        int m_lev;
        std::size_t m_index;
        amrex::Real m_start_time;
    };

    [[nodiscard]]
    TimeTracker track_time_until_out_of_scope (int lev, std::size_t index);
}

#endif //WARPX_PARALLELIZATION_TIME_TRACKER_H_
