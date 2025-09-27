/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_LOAD_BALANCING_LOAD_BALANCER_H_
#define WARPX_LOAD_BALANCING_LOAD_BALANCER_H_

#include "LoadBalancer_fwd.H"

#include "HeuristicCostsTracker.H"
#include "ScopedTimeTracker.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/Parser/IntervalsParser.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Enum.H>

#include <memory>
#include <optional>

namespace warpx::load_balancing
{

    /** Strategy to compute weights for use in load balance.
    */
    AMREX_ENUM(CostsUpdateAlgo,
           Timers,     //!< load balance according to in-code timer-based weights (i.e., with  `costs`)
           Heuristic,  /**< load balance according to weights computed from number of cells
                          and number of particles per box (i.e., with `costs_heuristic`) */
           Default = Timers);

    class LoadBalancer
    {

    public:

        static constexpr auto default_load_balance_knapsack_factor            = amrex::Real{1.24};
        static constexpr auto default_load_balance_efficiency_ratio_threshold = amrex::Real{1.1};

        LoadBalancer (int particle_shape, ElectromagneticSolverAlgo em_solver_id);

        void resize (int nlevs_max);
        void clear_level (int lev);
        void allocate_level (int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm);
        void reset_costs (int lev);
        void reset_efficiency(int lev);

        [[nodiscard]] int get_nlevs () const;
        [[nodiscard]] amrex::Real get_load_balance_efficiency (int lev) const;
        [[nodiscard]] amrex::LayoutData<amrex::Real>* get_costs (int lev);

        void compute_cost (
            int finest_level,
            const ablastr::fields::ConstMultiLevelScalarField& Ex,
            MultiParticleContainer& mpc);

        void check_and_do_load_balance (int step);

        void rescale_costs (int step);

        [[nodiscard]] bool is_active () const;

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

        /** Records a number corresponding to the load balance cost update strategy
        *  being used (0 or 1 corresponding to timers or heuristic).
        */
        CostsUpdateAlgo m_costs_update_algo = CostsUpdateAlgo::Default;

        std::optional<HeuristicCostsTracker> m_heuristic_cost_tracker = std::nullopt;
    };
}

#endif //WARPX_LOAD_BALANCING_LOAD_BALANCER_H_
