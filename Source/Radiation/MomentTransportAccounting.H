/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_MOMENTTRANSPORTACCOUNTING_H_
#define WARPX_RADIATION_MOMENTTRANSPORTACCOUNTING_H_

#include "FrameTransform.H"

namespace warpx::radiation
{
/** Independently integrated outward wall exchange and cylindrical basis
 * connection in [J,c*momentum]. Shared transport contract, not an M1 closure
 * or material-collision account. Caller publishes accepted increments once. */
struct MomentTransportAccounting
{
    FourVector boundary{};
    FourVector geometric{};
};
} // namespace warpx::radiation
#endif
