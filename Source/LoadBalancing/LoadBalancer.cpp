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
    }
}