/* Copyright 2020-2021 Axel Huebl, Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_PROFILERWRAPPER_H_
#define ABLASTR_PROFILERWRAPPER_H_

#include <AMReX_BLProfiler.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>

#include "ablastr/warn_manager/WarnManager.H"

namespace ablastr::profiler
{
    /** Controls when a profiler region is active.
     *
     * Always:                  unconditionally profiled.
     * OmitUnlessDeviceSynced:  skipped unless the expensive GPU device
     *     synchronization is enabled at runtime via
     *     tiny_profiler.device_synchronize_around_region (or the WarpX
     *     alias warpx.do_device_synchronize).  Turning that option on
     *     makes every profiled region accurate but noticeably slows
     *     execution due to forced GPU stream synchronization.
     */
    enum When { Always, OmitUnlessDeviceSynced };

    /** Returns true if tiny_profiler.device_synchronize_around_region is enabled.
     *
     * Call sites are only profiled accurately without GPU sync when they are
     * outside MFIter/ParIter loops (which synchronize automatically). Regions
     * inside such loops should use OmitUnlessDeviceSynced so they are
     * skipped when sync is off.
     *
     * For CPU-only builds, this unconditionally returns true since there is
     * no GPU stream synchronization concern.
     */
    inline bool is_sync_enabled ()
    {
#ifndef AMREX_USE_GPU
        return true;
#else
        int r = 0;
        amrex::ParmParse{"tiny_profiler"}.query("device_synchronize_around_region", r);
        return static_cast<bool>(r);
#endif
    }

    inline bool should_profile (When w)
    {
        if (w == Always) {
#ifdef AMREX_USE_GPU
            if (amrex::MFIter::currentDepth() > 0 && !is_sync_enabled()) {
                ablastr::warn_manager::WMRecordWarning(
                    "Performance",
                    "Profiler region with Always used inside an MFIter loop "
                    "without device synchronization; timings may be inaccurate.",
                    ablastr::warn_manager::WarnPriority::high);
            }
#endif
            return true;
        }
        return is_sync_enabled();
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
 *  \param when   ablastr::profiler::Always or ablastr::profiler::OmitUnlessDeviceSynced
 */
#define ABLASTR_PROFILE(fname, when) \
    ABLASTR_PROFILE_IMPL(fname, when, BL_PROFILE_PASTE(ablastr_blp_, __LINE__))

/** Profile a named sub-region with explicit start (immediate start version).
 *  \param fname   profiler region name
 *  \param vname   C++ variable name for the profiler object
 *  \param when    ablastr::profiler::Always or ablastr::profiler::OmitUnlessDeviceSynced
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
 *  \param when   ablastr::profiler::Always or ablastr::profiler::OmitUnlessDeviceSynced
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
 *  \param when   ablastr::profiler::Always or ablastr::profiler::OmitUnlessDeviceSynced
 */
#define ABLASTR_PROFILE_REGION(rname, when) \
    ABLASTR_PROFILE(rname, when)

#endif // ABLASTR_PROFILERWRAPPER_H_
