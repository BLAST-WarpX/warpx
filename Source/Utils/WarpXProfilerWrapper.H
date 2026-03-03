/* Copyright 2020-2021 Axel Huebl, Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_PROFILERWRAPPER_H_
#define WARPX_PROFILERWRAPPER_H_

#include "ablastr/profiler/ProfilerWrapper.H"

// Internal: full implementation receiving a pre-expanded unique variable name.
#define WARPX_PROFILE_IMPL_(fname, when, uniq) \
    BL_PROFILE_VAR_NS(fname, uniq); \
    if (ablastr::profiler::should_profile(when)) { BL_PROFILE_VAR_START(uniq); }

/** Profile the enclosing scope.
 *  \param fname  profiler region name (string literal)
 *  \param when   (optional) ablastr::profiler::OmitUnlessDeviceSynced; default: Always
 */
#define WARPX_PROFILE_1(fname) \
    WARPX_PROFILE_IMPL_(fname, ablastr::profiler::Always, BL_PROFILE_PASTE(warpx_blp_, __LINE__))
#define WARPX_PROFILE_2(fname, when) \
    WARPX_PROFILE_IMPL_(fname, when, BL_PROFILE_PASTE(warpx_blp_, __LINE__))
#define WARPX_PROFILE(...) \
    ABLASTR_PROFILE_SELECT_2(__VA_ARGS__, WARPX_PROFILE_2, WARPX_PROFILE_1)(__VA_ARGS__)

/** Profile a named sub-region with explicit start (immediate start version).
 *  \param fname  profiler region name
 *  \param vname  C++ variable name for the profiler object
 *  \param when   (optional) ablastr::profiler::OmitUnlessDeviceSynced; default: Always
 */
#define WARPX_PROFILE_VAR_2(fname, vname) \
    BL_PROFILE_VAR_NS(fname, vname); \
    const bool BL_PROFILE_PASTE(vname, _should_profile) = true; \
    if (BL_PROFILE_PASTE(vname, _should_profile)) { BL_PROFILE_VAR_START(vname); }
#define WARPX_PROFILE_VAR_3(fname, vname, when) \
    BL_PROFILE_VAR_NS(fname, vname); \
    const bool BL_PROFILE_PASTE(vname, _should_profile) = \
        ablastr::profiler::should_profile(when); \
    if (BL_PROFILE_PASTE(vname, _should_profile)) { BL_PROFILE_VAR_START(vname); }
#define WARPX_PROFILE_VAR(...) \
    ABLASTR_PROFILE_SELECT_3(__VA_ARGS__, WARPX_PROFILE_VAR_3, WARPX_PROFILE_VAR_2)(__VA_ARGS__)

/** Declare a named profiler without starting it yet.
 *  Use WARPX_PROFILE_VAR_START / WARPX_PROFILE_VAR_STOP to control timing.
 *  \param fname  profiler region name
 *  \param vname  C++ variable name for the profiler object
 *  \param when   (optional) ablastr::profiler::OmitUnlessDeviceSynced; default: Always
 */
#define WARPX_PROFILE_VAR_NS_2(fname, vname) \
    BL_PROFILE_VAR_NS(fname, vname); \
    const bool BL_PROFILE_PASTE(vname, _should_profile) = true
#define WARPX_PROFILE_VAR_NS_3(fname, vname, when) \
    BL_PROFILE_VAR_NS(fname, vname); \
    const bool BL_PROFILE_PASTE(vname, _should_profile) = \
        ablastr::profiler::should_profile(when)
#define WARPX_PROFILE_VAR_NS(...) \
    ABLASTR_PROFILE_SELECT_3(__VA_ARGS__, WARPX_PROFILE_VAR_NS_3, WARPX_PROFILE_VAR_NS_2)(__VA_ARGS__)

/** Start a profiler declared with WARPX_PROFILE_VAR_NS (respects its when). */
#define WARPX_PROFILE_VAR_START(vname) \
    if (BL_PROFILE_PASTE(vname, _should_profile)) { BL_PROFILE_VAR_START(vname); }

/** Stop a profiler declared with WARPX_PROFILE_VAR_NS (respects its when). */
#define WARPX_PROFILE_VAR_STOP(vname) \
    if (BL_PROFILE_PASTE(vname, _should_profile)) { BL_PROFILE_VAR_STOP(vname); }

/** Profile the enclosing scope as a named region.
 *  \param rname  region name
 *  \param when   (optional) ablastr::profiler::OmitUnlessDeviceSynced; default: Always
 */
#define WARPX_PROFILE_REGION(...) WARPX_PROFILE(__VA_ARGS__)

#endif // WARPX_PROFILERWRAPPER_H_
