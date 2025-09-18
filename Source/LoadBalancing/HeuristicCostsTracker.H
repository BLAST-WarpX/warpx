/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_LOAD_BALANCING_HEURISTIC_COSTS_TRACKER_H_
#define WARPX_LOAD_BALANCING_HEURISTIC_COSTS_TRACKER_H_

#include <AMReX_REAL.H>

namespace warpx::load_balancing
{
    class HeuristicCostsTracker
    {

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

