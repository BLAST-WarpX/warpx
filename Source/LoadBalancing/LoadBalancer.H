/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_LOAD_BALANCING_LOAD_BALANCER_H_
#define WARPX_LOAD_BALANCING_LOAD_BALANCER_H_

#include "LoadBalancer_fwd.H"

#include "ScopedTimeTracker.H"
#include "Utils/Parser/IntervalsParser.H"

#include <memory>
#include <optional>

namespace warpx::load_balancing
{
    class LoadBalancer
    {

    public:

        static constexpr auto default_load_balance_knapsack_factor            = amrex::Real{1.24};
        static constexpr auto default_load_balance_efficiency_ratio_threshold = amrex::Real{1.1};

        LoadBalancer ();

    private:
        /** Load balancing intervals that reads the "load_balance_intervals" string int the input file
         * for getting steps at which load balancing is performed */
        utils::parser::IntervalsParser m_load_balance_intervals;

        /** Load balance with 'space filling curve' strategy. */
        int m_load_balance_with_sfc = 0;

        /** Controls the maximum number of boxes that can be assigned to a rank during
        * load balance via the 'knapsack' strategy; e.g., if there are 4 boxes per rank,
        * `load_balance_knapsack_factor=2` limits the maximum number of boxes that can
        * be assigned to a rank to 8. */
        amrex::Real m_load_balance_knapsack_factor = default_load_balance_knapsack_factor;

        /** Threshold value that controls whether to adopt the proposed distribution
        * mapping during load balancing.  The new distribution mapping is adopted
        * if the ratio of proposed distribution mapping efficiency to current
        * distribution mapping efficiency is larger than the threshold; 'efficiency'
        * here means the average cost per MPI rank.  */
        amrex::Real m_load_balance_efficiency_ratio_threshold = default_load_balance_efficiency_ratio_threshold;

        /** Current load balance efficiency for each level.  */
        amrex::Vector<amrex::Real> m_load_balance_efficiency;
    };
}

#endif //WARPX_LOAD_BALANCING_LOAD_BALANCER_H_