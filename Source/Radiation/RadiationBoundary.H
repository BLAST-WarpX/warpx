/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_RADIATIONBOUNDARY_H_
#define WARPX_RADIATION_RADIATIONBOUNDARY_H_

namespace warpx::radiation
{
/** Shared face-condition contract; independent of the runtime coordinator.
 * Keep the established integer values used by device options and checkpoints.
 */
enum class DiffusionBoundary
{
    Reflecting = 0,
    Vacuum = 1,
    Marshak = 2,
    MarshakBath = 3
};
} // namespace warpx::radiation

#endif
