/* Copyright 2026 The WarpX Community
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_LIMITEDMOMENTTRANSPORT_H_
#define WARPX_RADIATION_LIMITEDMOMENTTRANSPORT_H_

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

namespace warpx::radiation
{

struct LimitedMomentTransportOptions
{
    int maximum_bisections = 32;
    bool verbose = false;
};

enum class LimitedMomentTransportFailure
{
    None,
    InvalidState,
    InvalidBeta,
    InvalidOpacity,
    InvalidGeometry,
    InvalidDt,
    Admissibility,
    GlobalBalance
};

struct LimitedMomentTransportResult
{
    LimitedMomentTransportFailure failure = LimitedMomentTransportFailure::None;
    bool valid = false;

    // Sum of c*dt/dx over spatial directions; acceptance requires <= 0.5.
    amrex::Real maximum_cfl = 0.0;

    // Shared-face limiter range over accepted geometric faces.  A face is
    // limited when its accepted theta is strictly below one.
    amrex::Real minimum_theta = 1.0;
    amrex::Real maximum_theta = 1.0;
    long long accepted_face_count = 0;
    long long limited_face_count = 0;

    // Minimum dimensionless 1-|q|/E over accepted face bars and cells;
    // vacuum is assigned margin one. Existing M1 roundoff collar applies.
    amrex::Real minimum_cone_margin = 0.0;

    // Cell-integrated global inventories and independently evaluated
    // initial-minus-integrated-divergence-minus-final balances. Components
    // are [energy,qx,qy,qz] in J, with q=c*momentum, not momentum itself.
    amrex::GpuArray<amrex::Real, 4> initial_inventory{};
    amrex::GpuArray<amrex::Real, 4> final_inventory{};
    amrex::GpuArray<amrex::Real, 4> raw_balance_residual{};
};

/** Advance radiation moments with an explicitly cone-limited M1 face flux.
 *
 * state is cell-integrated [E,qx,qy,qz], q=F/c=c*p [J].  beta and chi are
 * read-only frozen fields.  This routine performs transport only: it applies
 * no collision or material kick.  Periodic uniform Cartesian 1D and 3D are
 * supported by the first qualification scope.
 *
 * Every geometric face has one owner and uses one four-component shared
 * flux.  The numerical flux is the LLF flux blended with the existing moving
 * moment (source-aware spatial diffusion) flux by one scalar theta.  On any
 * rejection, state and all of its ghost cells are preserved exactly.  On
 * success, state is replaced by the accepted candidate.  A collective global
 * raw-balance audit is required for acceptance.
 */
LimitedMomentTransportResult
TryAdvanceLimitedMomentTransport (amrex::MultiFab& state, amrex::MultiFab const& beta,
                                  amrex::MultiFab const& chi, amrex::Geometry const& geometry,
                                  amrex::Real dt,
                                  LimitedMomentTransportOptions const& options = {});

} // namespace warpx::radiation

#endif // WARPX_RADIATION_LIMITEDMOMENTTRANSPORT_H_
