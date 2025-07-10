/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_TEMPERATUREDEPOSITIONTYPES_H_
#define WARPX_TEMPERATUREDEPOSITIONTYPES_H_

#include <AMReX.H>

namespace warpx::particles::deposition {

AMREX_ENUM(TemperatureDepositionType,
    SINGLE_PASS,
    DOUBLE_PASS
);

AMREX_ENUM(TemperatureDepositionPass,
    FIRST,
    SECOND
);


} // namespace warpx::particles::deposition
#endif // WARPX_TEMPERATUREDEPOSITIONTYPES_H_
