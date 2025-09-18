/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "LoadBalancer.H"

#include <AMReX_ParmParse.H>

#include <string>
#include <vector>

namespace warpx::load_balancing
{
    LoadBalancer::LoadBalancer(int particle_shape, ElectromagneticSolverAlgo em_solver_id)
    {
        const auto pp_algo = amrex::ParmParse pp_algo{"algo"};

        std::vector<std::string> load_balance_intervals_string_vec = {"0"};
        pp_algo.queryarr("load_balance_intervals", load_balance_intervals_string_vec);
        m_load_balance_intervals = utils::parser::IntervalsParser{
            load_balance_intervals_string_vec};

        if (!this->is_active()){
            return;
        }

        pp_algo.query("load_balance_with_sfc", m_load_balance_with_sfc);

        // Knapsack factor only used with non-SFC strategy
        if (!m_load_balance_with_sfc) {
            pp_algo.query("load_balance_knapsack_factor", m_load_balance_knapsack_factor);
        }

        utils::parser::queryWithParser(pp_algo, "load_balance_efficiency_ratio_threshold",
                        load_balance_efficiency_ratio_threshold);

        pp_algo.query_enum_sloppy("load_balance_costs_update", m_load_balance_costs_update_algo, "-_");

        if (WarpX::load_balance_costs_update_algo==CostsUpdateAlgo::Heuristic) {
            m_heuristic_cost_tracker = HeuristicCostsTracker{particle_shape, em_solver_id};
        }
        else{
            ScopedTimeTracker::toggle_tracking(true);
        }
    }

    void LoadBalancer::resize (const int nlevs_max)
    {
        if (m_costs_update_algo == CostsUpdateAlgo::Timers){
            ScopedTimeTracker::resize(nlevs_max);
        }
        else {
            m_heuristic_cost_tracker.value().resize(nlevs_max);
        }

        m_load_balance_efficiency.resize(nlevs_max);
    }

    void LoadBalancer::clear_level (const int lev)
    {
        if (m_costs_update_algo == CostsUpdateAlgo::Timers){
            ScopedTimeTracker::clear_level(lev);
        }
        else {
            m_heuristic_cost_tracker.value().clear_level(lev);
        }

        m_load_balance_efficiency.at(lev) = -1;
    }

    bool LoadBalancer::is_active ()
    {
        return m_load_balance_intervals.isActivated();
    }

    void LoadBalancer::allocate_level (
        const int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
    {
        if (m_costs_update_algo == CostsUpdateAlgo::Timers){
            ScopedTimeTracker::allocate_level(lev, ba, dm);
        }
        else {
            m_heuristic_cost_tracker.value().allocate_level(lev, ba, dm);
        }
        m_load_balance_efficiency[lev] = -1;
    }
}
