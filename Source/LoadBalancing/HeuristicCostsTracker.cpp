/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "HeuristicCostsTracker.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

#include "AMReX_ParmParse.H"

namespace warpx::load_balancing
{
    HeuristicCostsTracker::HeuristicCostsTracker (
        const int particle_shape, const ElectromagneticSolverAlgo em_solver_id)
    {
        const auto pp_algo = amrex::ParmParse{"algo"};

        const bool has_custom_value_for_cells = utils::parser::queryWithParser(
            pp_algo, "costs_heuristic_cells_wt", m_costs_heuristic_cells_wt);

        const bool has_custom_value_for_particles = utils::parser::queryWithParser(
            pp_algo, "costs_heuristic_particles_wt", m_costs_heuristic_particles_wt);

        // Set default values for particle and cell weights for costs update;
        // Default values listed here for the case AMREX_USE_GPU are determined
        // from single-GPU tests on Summit.
        if ((!has_custom_value_for_cells) && (!has_custom_value_for_particles)){

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (particle_shape > 0) && (particle_shape <= max_supported_order),
                "Only shape function orders between 1 and " +
                std::to_string(max_supported_order) +
                " have default cost heuristics.");

#ifdef AMREX_USE_GPU
            if (em_solver_id == ElectromagneticSolverAlgo::PSATD) {
                m_costs_heuristic_cells_wt     = costs_heuristic_cells_wt_gpu_psatd_default[particle_shape];
                m_costs_heuristic_particles_wt = costs_heuristic_particles_wt_gpu_psatd_default[particle_shape];
            } else { // FDTD
                m_costs_heuristic_cells_wt     = costs_heuristic_cells_wt_gpu_fdtd_default[particle_shape];
                m_costs_heuristic_particles_wt = costs_heuristic_particles_wt_gpu_fdtd_default[particle_shape];
            }
#else // CPU
            m_costs_heuristic_cells_wt     = costs_heuristic_cells_wt_cpu_default[particle_shape];
            m_costs_heuristic_particles_wt = costs_heuristic_particles_wt_cpu_default[particle_shape];
            amrex::ignore_unused(em_solver_id);
#endif // AMREX_USE_GPU
        }
    }

    void HeuristicCostsTracker::resize (const int nlevs_max)
    {
        m_costs.resize(nlevs_max);
    }

    void HeuristicCostsTracker::clear_level (const int lev)
    {
        m_costs.at(lev).reset();
    }

    void HeuristicCostsTracker::allocate_level (const int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
    {
        m_costs.at(lev) = std::make_unique<amrex::LayoutData<amrex::Real>>(ba, dm);
    }

    amrex::LayoutData<amrex::Real>* HeuristicCostsTracker::get_costs (int lev)
    {
        return m_costs.at(lev).get();
    }

    void HeuristicCostsTracker::reset_costs (int lev)
    {
        auto costs = *m_costs.at(lev);
        const auto iarr = costs.IndexArray();
        for (const auto& i : iarr) {
            costs[i] = 0.0;
        }
    }
}
