/* Copyright 2024 Andrew Myers, Ann Almgren, Axel Huebl
 * David Grote, Luca Fedeli, Maxence Thevenet, Michael Rowan
 * Remi Lehe, Weiqun Zhang, levinem, Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_LOADBALANCE_LOADBALANCE_H_
#define WARPX_LOADBALANCE_LOADBALANCE_H_

#include "Particles/MultiParticleContainer.H"
#include "Utils/Parser/IntervalsParser.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_Enum.H>
#include <AMReX_LayoutData.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <cstddef>

namespace warpx::load_balance
{

    constexpr auto default_load_balance_knapsack_factor = amrex::Real(1.24);

    constexpr auto default_load_balance_efficiency_ratio_threshold = amrex::Real(1.1);

    /** Strategy to compute weights for use in load balance.
    */
    AMREX_ENUM(
        CostsUpdateAlgo,
        Timers,     //!< load balance according to in-code timer-based weights (i.e., with  `costs`)
        Heuristic,  /**< load balance according to weights computed from number of cells
                          and number of particles per box (i.e., with `costs_heuristic`) */
        Default = Timers);

   AMREX_ENUM(
        LoadBalanceStrategy,
        Knapsack,
        SpaceFillingCurve
   );

    struct LoadBalanceResult
    {
        amrex::DistributionMapping dm;

        // These store efficiency (meaning, the  average 'cost' over all ranks,
        // normalized to max cost) for current and proposed distribution mappings
        amrex::Real currentEfficiency = 0.0;
        amrex::Real proposedEfficiency = 0.0;
    };

    class CostTracker
    {
    public:
        CostTracker (int lev, std::size_t mfi_iter_index);

        void add () const noexcept;

    private:
        int m_lev;
        std::size_t m_mfi_iter_index;
        amrex::Real m_wt;
    };

    class LoadBalance
    {
        friend class CostTracker;

    public:
        static LoadBalance& get_instance();

        LoadBalance (LoadBalance& other) = delete;
        void operator= (const LoadBalance& other) = delete;

        [[nodiscard]]
        CostsUpdateAlgo get_update_algo () const noexcept;

        [[nodiscard]]
        utils::parser::IntervalsParser get_intervals () const;

        void init (int nlevs_max, int nox, ElectromagneticSolverAlgo electromagnetic_solver_id);

        void clear_level (int lev);

        void set_costs (int lev, amrex::Real val);

        void set_efficiency (int lev, amrex::Real val);

        amrex::Real get_efficiency (int lev) const;

        [[nodiscard]]
        bool cost_tracking_required() const;

        [[nodiscard]]
        amrex::Real get_efficiency_ratio_threshold () const;

        [[nodiscard]]
        const std::unique_ptr<amrex::LayoutData<amrex::Real>>& get_costs (int lev) const;

        /** \brief resets costs to zero
        */
        void reset_costs (int finest_level);

        /** Perform running average of the LB costs
        *
        * Only needed for timers cost update, heuristic load balance considers the
        * instantaneous costs.
        * This gives more importance to most recent costs.
        */
        void rescale_costs (int finest_level, int step);

        void allocate (int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm);

        /** \brief adds particle and cell contributions in cells to compute heuristic
        * cost in each box on each level, and records in `costs`
        */
        void compute_costs_if_heuristic (
            int finest_level,
            const amrex::Vector<std::array<amrex::MultiFab*, 3> >& efield_ref,
            const MultiParticleContainer& mypc_ref);

        LoadBalanceResult compute_new_distribution_mapping(int lev) const;

    protected:
        LoadBalance (){}
        ~ LoadBalance (){}

    private:
        void set_weight_values_for_costs_update (int nox, ElectromagneticSolverAlgo electromagnetic_solver_id);

        /**Flag to store if LoadBalance has been initialized*/
        bool m_initialized = false;

        /** Load balancing intervals that reads the "load_balance_intervals" string int the input file
        * for getting steps at which load balancing is performed */
        utils::parser::IntervalsParser m_intervals;

        /** Algorithm used to compute costs for load balancing */
        CostsUpdateAlgo m_update_algo;

        /** Collection of LayoutData to keep track of weights used in load balancing
        * routines. Contains timer-based or heuristic-based costs depending on input option */
        amrex::Vector<std::unique_ptr<amrex::LayoutData<amrex::Real> > > m_costs;

        /** Load balance strategy. */
        LoadBalanceStrategy m_strategy = LoadBalanceStrategy::Knapsack;

        /** Controls the maximum number of boxes that can be assigned to a rank during
        * load balance via the 'knapsack' strategy; e.g., if there are 4 boxes per rank,
        * `load_balance_knapsack_factor=2` limits the maximum number of boxes that can
        * be assigned to a rank to 8. */
        amrex::Real m_knapsack_factor = 0.0;

        /** Threshold value that controls whether to adopt the proposed distribution
        * mapping during load balancing.  The new distribution mapping is adopted
        * if the ratio of proposed distribution mapping efficiency to current
        * distribution mapping efficiency is larger than the threshold; 'efficiency'
        * here means the average cost per MPI rank.  */
        amrex::Real m_efficiency_ratio_threshold = 0.0;

        std::vector<amrex::Real> m_efficiency;

        /** Weight factor for cells in `Heuristic` costs update.
        * Default values on GPU are determined from single-GPU tests on Summit.
        * The problem setup for these tests is an empty (i.e. no particles) domain
        * of size 256 by 256 by 256 cells, from which the average time per iteration
        * per cell is computed. */
        amrex::Real m_costs_heuristic_cells_wt = amrex::Real(-1);

        /** Weight factor for particles in `Heuristic` costs update.
         * Default values on GPU are determined from single-GPU tests on Summit.
        * The problem setup for these tests is a high-ppc (27 particles per cell)
        * uniform plasma on a domain of size 128 by 128 by 128, from which the approximate
        * time per iteration per particle is computed. */
        amrex::Real m_costs_heuristic_particles_wt = amrex::Real(-1);
    };
}


#endif //WARPX_LOADBALANCE_LOADBALANCE_H_
