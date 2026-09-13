/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_MOMENTBALANCE_H_
#define WARPX_RADIATION_MOMENTBALANCE_H_

#include "ImplicitMomentTransport.H"

namespace warpx::radiation::detail
{
/** Audit integrated moment inventories against independently evaluated
 * material, boundary and (in RZ) cylindrical stress contributions.
 * A geometric stress contribution is not a material impulse reservoir.
 */
void UpdateMomentGlobalBalance (amrex::MultiFab const& state, amrex::MultiFab const& old,
                                amrex::MultiFab const& transfer,
                                ImplicitMomentTransportResult& result, FourVector const& boundary,
                                FourVector const& geometric);
} // namespace warpx::radiation::detail
#endif
