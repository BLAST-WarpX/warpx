/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_MOMENTFIELDS_H_
#define WARPX_RADIATION_MOMENTFIELDS_H_

#include <string>

namespace warpx::radiation::detail
{
inline std::string
MomentFieldName (int component)
{
    return std::string("radiation_moment_q") + "xyz"[component];
}
}

#endif // WARPX_RADIATION_MOMENTFIELDS_H_
