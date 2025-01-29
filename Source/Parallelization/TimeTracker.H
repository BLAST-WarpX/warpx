/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_PARALLELIZATION_COST_TRACKER_H_
#define WARPX_PARALLELIZATION_COST_TRACKER_H_

#include <AMReX_REAL.H>

#include <cstddef>
#include <map>
#include <memory>
#include <utility>

namespace warpx::parallelization
{
    class WarpX;
    class TimeTracker;

    using LevelIndex = std::pair<int, std::size_t>;

    [[nodiscard]]
    TimeTracker track_time_until_out_of_scope (int lev, std::size_t index);

    [[nodiscard]]
    std::map<LevelIndex,amrex::Real> get_tracked_times ();

    [[nodiscard]]
    bool is_time_tracking_enabled ();

    class TimeTrackerWarpXInterface
    {
        private:
        static void toggle_tracking (bool do_tracking);
        static void reset ();
        static void reset_values ();

        friend class WarpX;
    };

    class TimeTracker
    {
        public:

        TimeTracker(int lev, std::size_t mfi_iter_index);
        ~TimeTracker();

        //TimeTracker must be uncopyable and unmovable
        TimeTracker(const TimeTracker&) = delete;
        TimeTracker(TimeTracker&&) = delete;
        TimeTracker& operator=(const TimeTracker&) = delete;
        TimeTracker& operator=(TimeTracker&&) = delete;

        private:

        LevelIndex m_lev_index;
        amrex::Real m_start_time;
    };

    class AllTimes
    {
    public:
        [[nodiscard]] static AllTimes& get_instance ();

        [[nodiscard]] std::map<LevelIndex, amrex::Real> get_times () const ;
        [[nodiscard]] bool enabled () const;

        //AllTimes is a singleton, so it  must be uncopyable and unmovable
        AllTimes(const AllTimes&) = delete;
        AllTimes(AllTimes&&) = delete;
        AllTimes& operator=(const AllTimes&) = delete;
        AllTimes& operator=(AllTimes&&) = delete;

        friend class TimeTracker;
        friend class TimeTrackerWarpXInterface;

    private:
        AllTimes();

        void toggle_tracking(bool do_tracking);
        void reset();
        void reset_values();

        bool m_do_tracking = false;
        std::map<LevelIndex, amrex::Real> m_all_times;
    };
}

#endif //WARPX_PARALLELIZATION_COST_TRACKER_H_
