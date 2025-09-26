/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "LoadBalancer.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

#include <AMReX_ParmParse.H>

#include <string>
#include <vector>

using namespace amrex;

namespace warpx::load_balancing
{
    LoadBalancer::LoadBalancer(int particle_shape, ElectromagneticSolverAlgo em_solver_id)
    {
        const auto pp_algo = amrex::ParmParse{"algo"};

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
                        m_load_balance_efficiency_ratio_threshold);

        pp_algo.query_enum_sloppy("load_balance_costs_update", m_costs_update_algo, "-_");

        if (m_costs_update_algo==CostsUpdateAlgo::Heuristic) {
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

    bool LoadBalancer::is_active () const
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

    int LoadBalancer::get_nlevs() const
    {
        return m_load_balance_efficiency.size();
    }

    amrex::Real LoadBalancer::get_load_balance_efficiency(const int lev) const
    {
        return m_load_balance_efficiency.at(lev);
    }

    amrex::LayoutData<amrex::Real>*
    LoadBalancer::get_costs (const int lev)
    {
        if (m_costs_update_algo == CostsUpdateAlgo::Timers){
            return ScopedTimeTracker::all_times.at(lev).get();
        }
        else {
            return m_heuristic_cost_tracker.value().get_costs(lev);
        }
    }

    void LoadBalancer::reset_costs (const int lev)
    {
        if (m_costs_update_algo == CostsUpdateAlgo::Timers){
            return ScopedTimeTracker::reset_times(lev);
        }
        else {
            return m_heuristic_cost_tracker.value().reset_costs(lev);
        }
    }

    void LoadBalancer::reset_efficiency(const int lev)
    {
        m_load_balance_efficiency.at(lev) = -1.0;
    }

    void LoadBalancer::check_and_do_load_balance (const int step)
    {
        if (!this->is_active()){
            return;
        }

        if (step > 0 && m_load_balance_intervals.contains(step+1)) {
            // TODO
            //LoadBalance();

            const int nlevs = this->get_nlevs();
            for (int lev = 0; lev < nlevs; ++lev){
                this->reset_costs(lev);
            }
        }
    }

    void LoadBalancer::rescale_costs (const int step)
    {
        if (m_costs_update_algo != CostsUpdateAlgo::Timers || !this->is_active()){
            return;
        }

        // Perform running average of the costs
        // (Giving more importance to most recent costs; only needed
        // for timers update, heuristic load balance considers the
        // instantaneous costs)
        auto& all_times = ScopedTimeTracker::all_times;
        const int nlevs = all_times.size();
        for (int lev = 0; lev < nlevs; ++lev){
            for (const auto& i : all_times[lev]->IndexArray()){
                 (*all_times[lev])[i] *= (1._rt - 2._rt/m_load_balance_intervals.localPeriod(step+1));
            }
        }

    }

    void LoadBalancer::compute_cost (
        const int finest_level,
        const ablastr::fields::ConstMultiLevelScalarField& Ex,
        MultiParticleContainer& mpc)
    {
        if (m_costs_update_algo != CostsUpdateAlgo::Heuristic || !this->is_active()){
            return;
        }
        m_heuristic_cost_tracker->compute_cost(finest_level, Ex, mpc);
    }
}




