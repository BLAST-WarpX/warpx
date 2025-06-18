/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_LOAD_BALANCING_SCOPED_TIME_TRACKER_H_
#define WARPX_LOAD_BALANCING_SCOPED_TIME_TRACKER_H_

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <cstddef>

namespace warpx::load_balancing
{
    class ScopedTimeTracker
    {

    public:
        [[nodiscard]]
        static const amrex::Vector<amrex::Vector<amrex::Real>>& view_tracked_times ();

        [[nodiscard]]
        static bool enabled ();

        ScopedTimeTracker(int lev, std::size_t mfi_iter_index, bool enabled_local = true);
        ~ScopedTimeTracker();

        void stop_and_record ();

        //TimeTracker must be uncopyable and unmovable
        ScopedTimeTracker(const ScopedTimeTracker&) = delete;
        ScopedTimeTracker(ScopedTimeTracker&&) = delete;
        ScopedTimeTracker& operator=(const ScopedTimeTracker&) = delete;
        ScopedTimeTracker& operator=(ScopedTimeTracker&&) = delete;

        friend class WarpX;

    private:
        static void toggle_tracking (bool do_tracking_flag);
        static void resize (const amrex::Vector<std::size_t>& max_index_per_level);
        static void reset_values ();

        static bool do_tracking;
        static amrex::Vector<amrex::Vector<amrex::Real>> all_times;

        void record_time ();

        int m_lev;
        std::size_t m_index;

        amrex::Real m_start_time;

        bool m_enabled_local;
        bool m_stopped  = false;
    };
}

#endif //WARPX_LOAD_BALANCING_SCOPED_TIME_TRACKER_H_