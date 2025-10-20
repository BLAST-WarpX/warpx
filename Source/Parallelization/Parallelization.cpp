/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Parallelization.H"

#include "ablastr/warn_manager/WarnManager.H"

#include "AMReX_ParmParse.H"

using namespace amrex;

namespace warpx::parallelization
{
    bool comms_in_single_precision_flag()
    {
        bool do_single_precision_comms = false;
        const auto pp_warpx = ParmParse{"warpx"};
        pp_warpx.query("do_single_precision_comms", do_single_precision_comms);
#ifdef AMREX_USE_FLOAT
        if (do_single_precision_comms) {
            do_single_precision_comms = false;
            ablastr::warn_manager::WMRecordWarning(
                "comms",
                "Overwrote warpx.do_single_precision_comms to be 0, since WarpX was built in single precision.",
                ablastr::warn_manager::WarnPriority::low);
        }
#endif
        return do_single_precision_comms;
    }

    bool dynamic_scheduling_flag()
    {
        bool do_dynamic_scheduling = true;
        const auto pp_warpx = ParmParse{"warpx"};
        pp_warpx.query("do_dynamic_scheduling", do_dynamic_scheduling);

        return do_dynamic_scheduling;
    }
}
