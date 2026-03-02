/* Copyright 2020-2021 Axel Huebl, Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_PROFILERWRAPPER_H_
#define WARPX_PROFILERWRAPPER_H_

#include "ablastr/profiler/ProfilerWrapper.H"

// Internal helper: receives a pre-expanded unique variable name so that
// both the BL_PROFILE_VAR_NS declaration and the conditional start refer
// to the same TinyProfiler object.
#define WARPX_PROFILE_IMPL(fname, when, uniq) \
    BL_PROFILE_VAR_NS(fname, uniq); \
    if (ablastr::profiler::should_profile(when)) { BL_PROFILE_VAR_START(uniq); }

/** Profile the enclosing scope.
 *  \param fname  profiler region name (string literal)
 *  \param when   WhenToProfile::Always or WhenToProfile::IfSyncEnabled
 */
#define WARPX_PROFILE(fname, when) \
    WARPX_PROFILE_IMPL(fname, when, BL_PROFILE_PASTE(warpx_blp_, __LINE__))

/** Profile a named sub-region with explicit start (immediate start version).
 *  \param fname   profiler region name
 *  \param vname   C++ variable name for the profiler object
 *  \param when    WhenToProfile::Always or WhenToProfile::IfSyncEnabled
 */
#define WARPX_PROFILE_VAR(fname, vname, when) \
    BL_PROFILE_VAR_NS(fname, vname); \
    const bool BL_PROFILE_PASTE(vname, _should_profile) = \
        ablastr::profiler::should_profile(when); \
    if (BL_PROFILE_PASTE(vname, _should_profile)) { BL_PROFILE_VAR_START(vname); }

/** Declare a named profiler without starting it yet.
 *  Use WARPX_PROFILE_VAR_START / WARPX_PROFILE_VAR_STOP to control timing.
 *  \param fname  profiler region name
 *  \param vname  C++ variable name for the profiler object
 *  \param when   WhenToProfile::Always or WhenToProfile::IfSyncEnabled
 */
#define WARPX_PROFILE_VAR_NS(fname, vname, when) \
    BL_PROFILE_VAR_NS(fname, vname); \
    const bool BL_PROFILE_PASTE(vname, _should_profile) = \
        ablastr::profiler::should_profile(when)

/** Start a profiler declared with WARPX_PROFILE_VAR_NS (respects its when). */
#define WARPX_PROFILE_VAR_START(vname) \
    if (BL_PROFILE_PASTE(vname, _should_profile)) { BL_PROFILE_VAR_START(vname); }

/** Stop a profiler declared with WARPX_PROFILE_VAR_NS (respects its when). */
#define WARPX_PROFILE_VAR_STOP(vname) \
    if (BL_PROFILE_PASTE(vname, _should_profile)) { BL_PROFILE_VAR_STOP(vname); }

/** Profile the enclosing scope as a named region.
 *  \param rname  region name
 *  \param when   WhenToProfile::Always or WhenToProfile::IfSyncEnabled
 */
#define WARPX_PROFILE_REGION(rname, when) \
    WARPX_PROFILE(rname, when)

#endif // WARPX_PROFILERWRAPPER_H_
