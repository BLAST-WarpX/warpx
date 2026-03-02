/* Copyright 2020-2021 Axel Huebl, Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_PROFILERWRAPPER_H_
#define ABLASTR_PROFILERWRAPPER_H_

#include <AMReX_BLProfiler.H>
#include <AMReX_ParmParse.H>

namespace ablastr::profiler
{
    /** Controls when a profiler region is active */
    enum class when { Always, IfSyncEnabled };

    /** Returns true if tiny_profiler.device_synchronize_around_region is enabled.
     *
     * Result is cached after the first call: call sites are only profiled
     * accurately without GPU sync when they are outside MFIter/ParIter loops
     * (which synchronize automatically). Regions inside such loops should use
     * when::IfSyncEnabled so they are skipped when sync is off.
     */
    inline bool is_sync_enabled ()
    {
        static const bool v = [] {
            int r = 0;
            amrex::ParmParse{"tiny_profiler"}.query("device_synchronize_around_region", r);
            return static_cast<bool>(r);
        }();
        return v;
    }

    inline bool should_profile (when when)
    {
        return when == when::Always || is_sync_enabled();
    }

} // namespace ablastr::profiler

// Internal helper: receives a pre-expanded unique variable name so that
// both the BL_PROFILE_VAR_NS declaration and the conditional start refer
// to the same TinyProfiler object.
#define ABLASTR_PROFILE_IMPL(fname, when, uniq) \
    BL_PROFILE_VAR_NS(fname, uniq); \
    if (ablastr::profiler::should_profile(when)) { BL_PROFILE_VAR_START(uniq); }

/** Profile the enclosing scope.
 *  \param fname  profiler region name (string literal)
 *  \param when   when::Always or when::IfSyncEnabled
 */
#define ABLASTR_PROFILE(fname, when) \
    ABLASTR_PROFILE_IMPL(fname, when, BL_PROFILE_PASTE(ablastr_blp_, __LINE__))

/** Profile a named sub-region with explicit start (immediate start version).
 *  \param fname   profiler region name
 *  \param vname   C++ variable name for the profiler object
 *  \param when    when::Always or when::IfSyncEnabled
 */
#define ABLASTR_PROFILE_VAR(fname, vname, when) \
    BL_PROFILE_VAR_NS(fname, vname); \
    const bool BL_PROFILE_PASTE(vname, _should_profile) = \
        ablastr::profiler::should_profile(when); \
    if (BL_PROFILE_PASTE(vname, _should_profile)) { BL_PROFILE_VAR_START(vname); }

/** Declare a named profiler without starting it yet.
 *  Use ABLASTR_PROFILE_VAR_START / ABLASTR_PROFILE_VAR_STOP to control timing.
 *  \param fname  profiler region name
 *  \param vname  C++ variable name for the profiler object
 *  \param when   when::Always or when::IfSyncEnabled
 */
#define ABLASTR_PROFILE_VAR_NS(fname, vname, when) \
    BL_PROFILE_VAR_NS(fname, vname); \
    const bool BL_PROFILE_PASTE(vname, _should_profile) = \
        ablastr::profiler::should_profile(when)

/** Start a profiler declared with ABLASTR_PROFILE_VAR_NS (respects its when). */
#define ABLASTR_PROFILE_VAR_START(vname) \
    if (BL_PROFILE_PASTE(vname, _should_profile)) { BL_PROFILE_VAR_START(vname); }

/** Stop a profiler declared with ABLASTR_PROFILE_VAR_NS (respects its when). */
#define ABLASTR_PROFILE_VAR_STOP(vname) \
    if (BL_PROFILE_PASTE(vname, _should_profile)) { BL_PROFILE_VAR_STOP(vname); }

/** Profile the enclosing scope as a named region.
 *  \param rname  region name
 *  \param when   when::Always or when::IfSyncEnabled
 */
#define ABLASTR_PROFILE_REGION(rname, when) \
    ABLASTR_PROFILE(rname, when)

#endif // ABLASTR_PROFILERWRAPPER_H_
