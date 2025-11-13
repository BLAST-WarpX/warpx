/* Copyright 2020-2021 Axel Huebl, Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_PROFILERWRAPPER_H_
#define WARPX_PROFILERWRAPPER_H_

#include <AMReX_BLProfiler.H>

#define WARPX_PROFILE(fname) BL_PROFILE(fname)
#define WARPX_PROFILE_VAR(fname, vname) BL_PROFILE_VAR(fname, vname)
#define WARPX_PROFILE_VAR_NS(fname, vname) BL_PROFILE_VAR_NS(fname, vname)
#define WARPX_PROFILE_VAR_START(vname) BL_PROFILE_VAR_START(vname)
#define WARPX_PROFILE_VAR_STOP(vname) BL_PROFILE_VAR_STOP(vname)
#define WARPX_PROFILE_REGION(rname) BL_PROFILE_REGION(rname)

#endif // WARPX_PROFILERWRAPPER_H_
