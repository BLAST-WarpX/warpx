/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_MOMENTBOUNDARYEXCHANGE_H_
#define WARPX_RADIATION_MOMENTBOUNDARYEXCHANGE_H_

#include "MomentClosure.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <array>

namespace warpx::radiation
{
    /** Independently integrate Cartesian physical-boundary face fluxes.
     * Each face MultiFab has four components (E,q=F/c), positive in its
     * coordinate direction, with the same cell-volume normalization as the
     * moment transport operator. The returned outward exchange is dt/dx times
     * the signed boundary-face sum. Its q components divided by c are physical
     * momentum. Periodic directions contribute exactly zero.
     * This performs no time acceptance or diagnostic accumulation. Callers must
     * commit the result only with the corresponding accepted transport stage.
     * Invalid inputs leave exchange unchanged. No inputs or ghosts are changed.
     */
    bool
    ComputeMomentBoundaryExchange(std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& fluxes,
                                  amrex::Geometry const& geometry, amrex::Real dt,
                                  FourVector& exchange);
} // namespace warpx::radiation
#endif
