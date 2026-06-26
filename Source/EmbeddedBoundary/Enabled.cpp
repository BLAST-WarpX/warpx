/* Copyright 2024 Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Enabled.H"

#ifdef AMREX_USE_EB
#include <AMReX_ParmParse.H>
#endif
#if defined(AMREX_USE_EB) && defined(WARPX_DIM_RZ)
#include <stdexcept>
#endif


namespace EB
{
    bool enabled ()
    {
#ifndef AMREX_USE_EB
        return false;
#else
        amrex::ParmParse const pp_warpx("warpx");
        amrex::ParmParse const pp_eb2("eb2");

        // test various runtime options to enable EBs
        std::string eb_implicit_function;
        bool eb_enabled = pp_warpx.query("eb_implicit_function", eb_implicit_function);

        // https://amrex-codes.github.io/amrex/docs_html/EB.html
        std::string eb_stl;
        eb_enabled |= pp_eb2.query("geom_type", eb_stl);

        return eb_enabled;
#endif
    }

} // namespace EB
