/* Copyright 2024 Andrew Myers, Ann Almgren, Axel Huebl
 * David Grote, Luca Fedeli, Maxence Thevenet, Michael Rowan
 * Remi Lehe, Weiqun Zhang, levinem, Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "LoadBalance.H"

#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

#include "ablastr/warn_manager/WarnManager.H"

#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include <string>
#include <vector>

using namespace warpx::load_balance;
using namespace amrex::literals;

namespace
{
    CostsUpdateAlgo parse_cost_update_algo ()
    {
        CostsUpdateAlgo load_balance_costs_update_algo = CostsUpdateAlgo::Default;

        const amrex::ParmParse pp_algo("algo");
        pp_algo.query_enum_sloppy("load_balance_costs_update", load_balance_costs_update_algo, "-_");

        return load_balance_costs_update_algo;
    }

    [[nodiscard]]
    utils::parser::IntervalsParser parse_load_balance_intervals ()
    {
        const auto pp_algo = amrex::ParmParse{"algo"};
        std::vector<std::string> load_balance_intervals_string_vec = {"0"};
        pp_algo.queryarr("load_balance_intervals", load_balance_intervals_string_vec);
        return utils::parser::IntervalsParser{load_balance_intervals_string_vec};
    }

    [[nodiscard]]
    LoadBalanceStrategy parse_load_balance_strategy ()
    {
        const auto pp_algo = amrex::ParmParse{"algo"};
        bool load_balance_with_sfc = false;
        pp_algo.query("load_balance_with_sfc", load_balance_with_sfc);

        return (load_balance_with_sfc)?
            LoadBalanceStrategy::SpaceFillingCurve :
            LoadBalanceStrategy::Knapsack;
    }

    [[nodiscard]]
    amrex::Real parse_load_balance_knapsack_factor ()
    {
        const auto pp_algo = amrex::ParmParse{"algo"};
        auto load_balance_knapsack_factor = default_load_balance_knapsack_factor;
        pp_algo.query("load_balance_knapsack_factor", load_balance_knapsack_factor);
        return load_balance_knapsack_factor;
    }

    [[nodiscard]]
    amrex::Real parse_load_balance_efficiency_ratio_threshold ()
    {
        const auto pp_algo = amrex::ParmParse{"algo"};
        auto load_balance_efficiency_ratio_threshold = default_load_balance_efficiency_ratio_threshold;
        utils::parser::queryWithParser(pp_algo, "load_balance_efficiency_ratio_threshold",
            load_balance_efficiency_ratio_threshold);
        return load_balance_efficiency_ratio_threshold;
    }



#ifdef AMREX_USE_GPU
    constexpr bool amrex_use_gpu = true;
#else
    constexpr bool amrex_use_gpu = false;
#endif
}

CostTracker::CostTracker (int lev, std::size_t mfi_iter_index):
    m_lev{lev},
    m_mfi_iter_index{mfi_iter_index},
    m_wt{0.0_rt}
{
    const auto& load_balance = LoadBalance::get_instance();

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        load_balance.m_initialized,
        "LoadBalance must be initialized before costs are updated");

    if (!load_balance.cost_tracking_required())
        return;

    amrex::Gpu::synchronize();

    m_wt = static_cast<amrex::Real>(amrex::second());
}

void
CostTracker::add () const noexcept
{
    const auto& load_balance = LoadBalance::get_instance();

    if (!load_balance.cost_tracking_required())
        return;

    amrex::Gpu::synchronize();
    const auto time_diff = static_cast<amrex::Real>(amrex::second()) - m_wt;
    auto& cost = LoadBalance::get_instance().m_costs[m_lev];
    amrex::HostDevice::Atomic::Add( &(*cost)[m_mfi_iter_index], time_diff);
}

LoadBalance& LoadBalance::get_instance()
{
    static auto instance = LoadBalance{};
    return instance;
}

[[nodiscard]]
CostsUpdateAlgo LoadBalance::get_update_algo () const noexcept
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_initialized,
        "LoadBalance must be initialized before calling get_update_algo");
    return m_update_algo;
}

[[nodiscard]]
utils::parser::IntervalsParser LoadBalance::get_intervals () const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_initialized,
        "LoadBalance must be initialized before calling get_intervals");
    return m_intervals;
}

void LoadBalance::init (const int nlevs_max, const int nox, const ElectromagneticSolverAlgo electromagnetic_solver_id)
{
    m_intervals = ::parse_load_balance_intervals();

    m_costs.resize(nlevs_max);
    m_efficiency.resize(nlevs_max);
    m_efficiency_ratio_threshold =
        ::parse_load_balance_efficiency_ratio_threshold();

    m_update_algo = ::parse_cost_update_algo();
    if (m_update_algo == CostsUpdateAlgo::Heuristic){
        this->set_weight_values_for_costs_update(nox, electromagnetic_solver_id);
    }

    m_strategy = ::parse_load_balance_strategy();
    if (m_strategy == LoadBalanceStrategy::Knapsack){
        m_knapsack_factor = ::parse_load_balance_knapsack_factor();
    }

    m_initialized = true;
}

void LoadBalance::set_weight_values_for_costs_update (
    const int nox, const ElectromagneticSolverAlgo electromagnetic_solver_id)
{
    if constexpr (amrex_use_gpu){
        if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
            switch (nox)
            {
                case 1:
                    m_costs_heuristic_cells_wt = 0.575_rt;
                    m_costs_heuristic_particles_wt = 0.425_rt;
                    break;
                case 2:
                    m_costs_heuristic_cells_wt = 0.405_rt;
                    m_costs_heuristic_particles_wt = 0.595_rt;
                    break;
                case 3:
                    m_costs_heuristic_cells_wt = 0.250_rt;
                    m_costs_heuristic_particles_wt = 0.750_rt;
                    break;
                default:
                    // this is only a guess
                    m_costs_heuristic_cells_wt = 0.200_rt;
                    m_costs_heuristic_particles_wt = 0.800_rt;
                    break;
            }
        }
        else{
            switch (nox)
            {
                case 1:
                    m_costs_heuristic_cells_wt = 0.401_rt;
                    m_costs_heuristic_particles_wt = 0.599_rt;
                    break;
                case 2:
                    m_costs_heuristic_cells_wt = 0.268_rt;
                    m_costs_heuristic_particles_wt = 0.732_rt;
                    break;
                case 3:
                    m_costs_heuristic_cells_wt = 0.145_rt;
                    m_costs_heuristic_particles_wt = 0.855_rt;
                    break;
                default:
                    // this is only a guess
                    m_costs_heuristic_cells_wt = 0.100_rt;
                    m_costs_heuristic_particles_wt = 0.900_rt;
                    break;
            }
        }
    }
    else{
        m_costs_heuristic_cells_wt = 0.1_rt;
        m_costs_heuristic_particles_wt = 0.9_rt;
    }

    const auto pp_algo = amrex::ParmParse{"algo"};
    const bool has_user_defined_costs_heuristic_cells_wt =
        utils::parser::queryWithParser(
            pp_algo, "costs_heuristic_cells_wt", m_costs_heuristic_cells_wt);
    const bool has_user_defined_costs_heuristic_particles_wt =
        utils::parser::queryWithParser(
            pp_algo, "costs_heuristic_particles_wt", m_costs_heuristic_particles_wt);

    if ((nox > 3) &&
        !has_user_defined_costs_heuristic_cells_wt &&
        !has_user_defined_costs_heuristic_particles_wt){

        ablastr::warn_manager::WMRecordWarning(
            "LoadBalance",
            "Default weights for heuristic load balancing are not tested for "
            "particle shapes > 3.",
            ablastr::warn_manager::WarnPriority::medium);
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_costs_heuristic_cells_wt > 0.0 ||
        m_costs_heuristic_particles_wt > 0.0,
        "At least one bewteen costs_heuristic_cells_wt and costs_heuristic_particles_wt "
        "must be non-negative");
}

void LoadBalance::clear_level (const int lev)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_initialized,
        "LoadBalance must be initialized before calling clear_level");
    m_costs[lev].reset();
    m_efficiency[lev] = -1;
}

void LoadBalance::set_costs (const int lev, const amrex::Real val)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_initialized,
        "LoadBalance must be initialized before calling set_costs");
    if (m_costs[lev]) {
        const auto iarr = m_costs[lev]->IndexArray();
        for (const auto& i : iarr) {
            (*m_costs[lev])[i] = val;
        }
    }
}

[[nodiscard]]
const std::unique_ptr<amrex::LayoutData<amrex::Real>>&
LoadBalance::get_costs (int lev) const
{
    return m_costs[lev];
}

void LoadBalance::set_efficiency (const int lev, const amrex::Real val)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_initialized,
        "LoadBalance must be initialized before calling set_efficiency");
    m_efficiency[lev] = val;
}

[[nodiscard]]
amrex::Real LoadBalance::get_efficiency (const int lev) const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_initialized,
        "LoadBalance must be initialized before calling get_efficiency");
    return  m_efficiency[lev];
}

[[nodiscard]]
amrex::Real LoadBalance::get_efficiency_ratio_threshold () const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_initialized,
        "LoadBalance must be initialized before calling get_efficiency_ratio_threshold");
    return  m_efficiency_ratio_threshold;
}

[[nodiscard]]
bool LoadBalance::cost_tracking_required () const
{
    return (m_intervals.isActivated() && (m_update_algo == CostsUpdateAlgo::Timers));
}

void LoadBalance::reset_costs (const int finest_level)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_initialized,
        "LoadBalance must be initialized before calling reset_costs");

    for (int lev = 0; lev <= finest_level; ++lev){
        this->set_costs(lev, 0.0_rt);
    }

}

void LoadBalance::rescale_costs (const int finest_level, const int step)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_initialized,
        "LoadBalance must be initialized before calling reset_costs");

    // rescale is only used for timers
    if (m_update_algo != CostsUpdateAlgo::Timers){
        return;
    }

    AMREX_ALWAYS_ASSERT(m_costs.size() == finest_level + 1);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (m_costs[lev])
        {
            // Perform running average of the costs
            // (Giving more importance to most recent costs; only needed
            // for timers update, heuristic load balance considers the
            // instantaneous costs)
            for (const auto& i : m_costs[lev]->IndexArray())
            {
                (*m_costs[lev])[i] *= (1._rt - 2._rt/m_intervals.localPeriod(step+1));
            }
        }
    }

}

void LoadBalance::allocate (const int lev,
    const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
{
    if (m_intervals.isActivated())
    {
        m_costs[lev] = std::make_unique<amrex::LayoutData<amrex::Real>>(ba, dm);
        this->set_efficiency(lev,-1.0);
    }
}

void LoadBalance::compute_costs_if_heuristic (
    const int finest_level,
    const amrex::Vector<std::array<amrex::MultiFab*, 3> >& efield_ref,
    const MultiParticleContainer& mypc_ref)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_initialized,
        "LoadBalance must be initialized before calling compute_costs_if_heuristic");
    if (m_update_algo != CostsUpdateAlgo::Heuristic){
        return;
    }

    const auto nSpecies = mypc_ref.nSpecies();
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // Species loop
        for (int i_s = 0; i_s < nSpecies; ++i_s)
        {
            auto & myspc = mypc_ref.GetParticleContainer(i_s);

            // Particle loop
            for (WarpXParIter pti(myspc, lev); pti.isValid(); ++pti)
            {
                (*m_costs[lev])[pti.index()] += m_costs_heuristic_particles_wt*pti.numParticles();
            }
        }

        // Cell loop
        const auto Ex = efield_ref[lev][0];
        for (amrex::MFIter mfi(*Ex, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box& gbx = mfi.growntilebox();
            (*m_costs[lev])[mfi.index()] += m_costs_heuristic_cells_wt*gbx.numPts();
        }
    }
}

LoadBalanceResult LoadBalance::compute_new_distribution_mapping(int lev) const
{
    using namespace amrex;

    const Real nboxes = m_costs[lev]->size();
    const Real nprocs = ParallelContext::NProcsSub();
    const int nmax = static_cast<int>(std::ceil(nboxes/nprocs*m_knapsack_factor));

    // Compute the new distribution mapping
    LoadBalanceResult res;

    const bool broadcast_to_all_false = false;
    res.dm = (m_strategy == LoadBalanceStrategy::SpaceFillingCurve)?
        DistributionMapping::makeSFC(*m_costs[lev],
            res.currentEfficiency, res.proposedEfficiency,
            broadcast_to_all_false,
            ParallelDescriptor::IOProcessorNumber()):
        DistributionMapping::makeKnapSack(*m_costs[lev],
            res.currentEfficiency, res.proposedEfficiency,
            nmax,
            broadcast_to_all_false,
            ParallelDescriptor::IOProcessorNumber());

    return res;
}
