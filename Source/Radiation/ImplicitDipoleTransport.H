/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_IMPLICITDIPOLETRANSPORT_H_
#define WARPX_RADIATION_IMPLICITDIPOLETRANSPORT_H_

#include "FrameTransform.H"
#include "MomentTransportAccounting.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

namespace warpx::radiation
{
struct ImplicitDipoleTransportOptions
{
    int max_iterations = 500;
    int restart_length = 120;
    int max_refinements = 8;
    amrex::Real linear_tolerance = 1.e-13;
    amrex::Real equation_tolerance = 1.e-10;
    // Explicit optical mirrors in RZ; not a material-wall policy.
    bool reflecting_boundaries = false;
    // Opt-in annular angular transport: conserve sum(rbar*Qtheta), not sum(Qtheta).
    // Requires an RZ build and the same fixed optical-mirror authority.
    bool rz_angular_transport = false;
};

enum class ImplicitDipoleFailure
{
    None, Configuration, Coefficients, LinearSolve, Admissibility, CollisionSource, Equation
};

struct ImplicitDipoleTransportResult
{
    ImplicitDipoleFailure failure = ImplicitDipoleFailure::Configuration;
    int linear_iterations = 0;
    int refinements = 0;
    amrex::Real equation_ratio = 0;
    amrex::Real conservation_ratio = 0;
    amrex::Real angular_conservation_ratio = 0;
    // Outward optical-wall exchange and cylindrical basis connection,
    // respectively, in [J,c*momentum]. Neither is a material collision.
    FourVector boundary_exchange{};
    FourVector geometric_exchange{};
    bool valid = false;
};

/** Joint backward-Euler comoving-P1 transport and gray collision stage.
 * Cell-integrated radiation and collision transfer are [E,F/c] in joules;
 * absorption/scattering are proper inverse lengths, and equilibrium is a
 * cell-integrated proper bath energy. Frozen beta is physical (x,y,z).
 *
 * This M1-independent low-level advance currently requires periodic Cartesian
 * cell-centred fields. It includes flux memory and source-aware AP face
 * transport, with exact stress/source boosts. It has no native material/EOS
 * iteration, packet interface or boundary authority. No beta*optical-depth
 * restriction is inferred here, but |beta| <= 0.01 and the positive comoving
 * dipole cone remain mandatory. An explicit reflecting_boundaries option
 * admits meridional RZ with an axis and fixed radial optical mirror; z can
 * be periodic or have fixed optical mirrors. The separate rz_angular_transport
 * option admits azimuthal state/frame components with torque-conservative
 * radial flux weights. Default meridional admission and open-wall rejection
 * remain. Angular admission is not packet/native/runtime authority.
 * This does not activate native RZ material/runtime or packet interfaces.
 *
 * transfer contains COLLISION ONLY, never pressure-flux divergence. heat is
 * the projected collision energy minus frozen-beta mechanical work. Output
 * valid cells and ghosts are unchanged on failure. On success valid cells
 * are published together; caller-owned ghosts remain unchanged. Outputs may
 * not alias each other or the prescribed input fields; views are checked.
 */
ImplicitDipoleTransportResult TryImplicitDipoleTransport (
    amrex::MultiFab& radiation, amrex::MultiFab const& beta,
    amrex::MultiFab const& absorption, amrex::MultiFab const& scattering,
    amrex::MultiFab const& equilibrium, amrex::MultiFab& transfer,
    amrex::MultiFab& heat, amrex::Geometry const& geometry, amrex::Real dt,
    ImplicitDipoleTransportOptions const& options = {});

/** Read-only spatial equation witness at a prescribed accepted state/frame.
 * Components0:3 are dt*div(LAB flux); 4:7 are uncancelled face arithmetic
 * operand sums. Output must have eight cell-centred components and the same
 * layout. Failure preserves inputs/output including ghosts. This supplies no
 * collision/material authority and has the same geometry/optical scope.
 * Optional accounting is replaced only on success, like the field output. */
bool ComputeDipoleTransportIncrement (
    amrex::MultiFab const& radiation, amrex::MultiFab const& beta,
    amrex::MultiFab const& absorption, amrex::MultiFab const& scattering,
    amrex::MultiFab const& equilibrium, amrex::MultiFab& increment,
    amrex::Geometry const& geometry, amrex::Real dt,
    ImplicitDipoleTransportOptions const& options = {},
    MomentTransportAccounting* accounting = nullptr);
} // namespace warpx::radiation
#endif
