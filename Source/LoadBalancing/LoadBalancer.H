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
#include "Utils/Parser/IntervalsParser.H"
#include "Utils/WarpXAlgorithmSelection.H"

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

        [[nodiscard]] bool is_active ();

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

/*
amrex::LayoutData<amrex::Real>*
WarpX::getCosts (int lev)
{
    if (m_instance)
    {
        return m_instance->costs[lev].get();
    } else
    {
        return nullptr;
    }
}

void
WarpX::setLoadBalanceEfficiency (const int lev, const amrex::Real efficiency)
{
    if (m_instance)
    {
        m_instance->load_balance_efficiency[lev] = efficiency;
    } else
    {
        return;
    }
}

amrex::Real
WarpX::getLoadBalanceEfficiency (const int lev)
{
    if (m_instance)
    {
        return m_instance->load_balance_efficiency[lev];
    } else
    {
        return -1;
    }
}


void
WarpX::ResetCosts ()
{
    AMREX_ALWAYS_ASSERT(!costs.empty());
    AMREX_ALWAYS_ASSERT(costs[0] != nullptr);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        const auto iarr = costs[lev]->IndexArray();
        for (const auto& i : iarr)
        {
            // Reset costs
            (*costs[lev])[i] = 0.0;
        }
    }
}

void
WarpX::RescaleCosts (int step)
{
    // rescale is only used for timers
    if (WarpX::load_balance_costs_update_algo != LoadBalanceCostsUpdateAlgo::Timers)
    {
        return;
    }

    AMREX_ALWAYS_ASSERT(costs.size() == finest_level + 1);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (costs[lev])
        {
            // Perform running average of the costs
            // (Giving more importance to most recent costs; only needed
            // for timers update, heuristic load balance considers the
            // instantaneous costs)
            for (const auto& i : costs[lev]->IndexArray())
            {
                (*costs[lev])[i] *= (1._rt - 2._rt/load_balance_intervals.localPeriod(step+1));
            }
        }
    }
}

void
WarpX::ComputeCostsHeuristic (amrex::Vector<std::unique_ptr<amrex::LayoutData<amrex::Real> > >& a_costs)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        const auto & mypc_ref = GetPartContainer();
        const auto nSpecies = mypc_ref.nSpecies();

        // Species loop
        for (int i_s = 0; i_s < nSpecies; ++i_s)
        {
            auto & myspc = mypc_ref.GetParticleContainer(i_s);

            // Particle loop
            for (WarpXParIter pti(myspc, lev); pti.isValid(); ++pti)
            {
                (*a_costs[lev])[pti.index()] += costs_heuristic_particles_wt*pti.numParticles();
            }
        }

        // Cell loop
        MultiFab* Ex = m_fields.get(FieldType::Efield_fp, Direction{0}, lev);
        for (MFIter mfi(*Ex, false); mfi.isValid(); ++mfi)
        {
            const Box& gbx = mfi.growntilebox();
            (*a_costs[lev])[mfi.index()] += costs_heuristic_cells_wt*gbx.numPts();
        }
    }
}

    [[nodiscard]] utils::parser::IntervalsParser get_load_balance_intervals () const
    {
        return load_balance_intervals;
    }

void
WarpX::CheckLoadBalance (int step)
{
    if (step > 0 && load_balance_intervals.contains(step+1))
    {
        LoadBalance();

        // Reset the costs to 0
        ResetCosts();
    }
    if (!costs.empty())
    {
        RescaleCosts(step);
    }
}


*/
