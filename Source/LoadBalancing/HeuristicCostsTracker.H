/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_LOAD_BALANCING_HEURISTIC_COSTS_TRACKER_H_
#define WARPX_LOAD_BALANCING_HEURISTIC_COSTS_TRACKER_H_

#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_REAL.H>

#include <array>

namespace warpx::load_balancing
{
    class HeuristicCostsTracker
    {

    using namespace amrex;

    public:
        // Default values for particle and cell weights for costs update for
        // shape function orders from 1 to 4.
        // Default values listed here for the case AMREX_USE_GPU are determined
        // from single-GPU tests on Summit. Values for order 4 are only a guess.
        static constexpr int max_supported_order = 4;
        static constexpr costs_heuristic_cells_wt_gpu_fdtd_default  =
            std::array<amrex::Real, max_supported_order>{0.401_rt, 0.268_rt, 0.145_rt, 0.100_rt};
        static constexpr costs_heuristic_cells_wt_gpu_psatd_default =
            std::array<amrex::Real, max_supported_order>{0.575_rt, 0.405_rt, 0.250_rt, 0.200_rt};
        static constexpr costs_heuristic_cells_wt_cpu_default       =
            std::array<amrex::Real, max_supported_order>{0.1_rt, 0.1_rt, 0.1_rt, 0.1_rt};

        static constexpr costs_heuristic_particles_wt_gpu_fdtd_default  =
            std::array<amrex::Real, max_supported_order>{0.599_rt, 0.732_rt, 0.855_rt, 0.900_rt};
        static constexpr costs_heuristic_particles_wt_gpu_psatd_default =
            std::array<amrex::Real, max_supported_order>{0.425_rt, 0.595_rt, 0.750_rt, 0.800_rt};
        static constexpr costs_heuristic_particles_wt_cpu_default       =
            std::array<amrex::Real, max_supported_order>{0.9_rt, 0.9_rt, 0.9_rt, 0.9_rt};

        HeuristicCostsTracker (int particle_shape, ElectromagneticSolverAlgo em_solver_id);

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
    };
}

#endif //WARPX_LOAD_BALANCING_HEURISTIC_COSTS_TRACKER_H_
