/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_PARALLELIZATION_PARALLELIZATION_H_
#define WARPX_PARALLELIZATION_PARALLELIZATION_H_

namespace warpx::parallelization
{
    /**
    * \brief Returns true if nodal points have to be synchronized
    *
    *  Note: currently this is hardcoded to "true"
    */
    constexpr bool sync_nodal_points_flag ()
    {
        return true;
    }

    /**
    * \brief Returns true if field communications have to be performed in single precision, false otherwise
    *
    * The function returns "false" unless "warpx.do_single_precision_comms"
    * is set to 1 in the inputfile.
    */
    bool comms_in_single_precision_flag ();

    /**
    * \brief Returns true if dynamic scheduling is enabled, false otherwise
    *
    * The function returns "true" unless "warpx.do_dynamic_scheduling"
    * is set to 0 in the inputfile.
    */
    bool dynamic_scheduling_flag ();
}

#endif //WARPX_PARALLELIZATION_PARALLELIZATION_H_
