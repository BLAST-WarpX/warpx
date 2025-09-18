/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_LOAD_BALANCING_HEURISTIC_COSTS_TRACKER_H_
#define WARPX_LOAD_BALANCING_HEURISTIC_COSTS_TRACKER_H_

#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_LayoutData.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <array>
#include <memory>

namespace warpx::load_balancing
{
    class HeuristicCostsTracker
    {

    public:
        // Default values for particle and cell weights for costs update for
        // shape function orders from 1 to 4.
        // Default values listed here for the case AMREX_USE_GPU are determined
        // from single-GPU tests on Summit. Values for order 4 are only a guess.
        static constexpr int max_supported_order = 4;
        static constexpr auto costs_heuristic_cells_wt_gpu_fdtd_default  =
            std::array<amrex::Real, max_supported_order>{0.401, 0.268, 0.145, 0.100};
        static constexpr auto costs_heuristic_cells_wt_gpu_psatd_default =
            std::array<amrex::Real, max_supported_order>{0.575, 0.405, 0.250, 0.200};
        static constexpr auto costs_heuristic_cells_wt_cpu_default       =
            std::array<amrex::Real, max_supported_order>{0.1, 0.1, 0.1, 0.1};

        static constexpr auto costs_heuristic_particles_wt_gpu_fdtd_default  =
            std::array<amrex::Real, max_supported_order>{0.599, 0.732, 0.855, 0.900};
        static constexpr auto costs_heuristic_particles_wt_gpu_psatd_default =
            std::array<amrex::Real, max_supported_order>{0.425, 0.595, 0.750, 0.800};
        static constexpr auto costs_heuristic_particles_wt_cpu_default       =
            std::array<amrex::Real, max_supported_order>{0.9, 0.9, 0.9, 0.9};

        HeuristicCostsTracker (int particle_shape, ElectromagneticSolverAlgo em_solver_id);

        void resize (int nlevs_max);
        void clear_level (int lev);
        void allocate_level (const int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm);

    private:
        /** Weight factor for cells in `Heuristic` costs update.
        * Default values on GPU are determined from single-GPU tests on Summit.
        * The problem setup for these tests is an empty (i.e. no particles) domain
        * of size 256 by 256 by 256 cells, from which the average time per iteration
        * per cell is computed. */
        amrex::Real costs_heuristic_cells_wt = amrex::Real(0);

        /** Weight factor for particles in `Heuristic` costs update.
        * Default values on GPU are determined from single-GPU tests on Summit.
        * The problem setup for these tests is a high-ppc (27 particles per cell)
        * uniform plasma on a domain of size 128 by 128 by 128, from which the approximate
        * time per iteration per particle is computed. */
        amrex::Real costs_heuristic_particles_wt = amrex::Real(0);

        amrex::Vector<std::unique_ptr<amrex::LayoutData<amrex::Real>>> m_costs;
    };
}

#endif //WARPX_LOAD_BALANCING_HEURISTIC_COSTS_TRACKER_H_
