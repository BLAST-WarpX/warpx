/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_IMPLICITGROUPMOMENTTRANSPORT_H_
#define WARPX_RADIATION_IMPLICITGROUPMOMENTTRANSPORT_H_

#include "ExactPlanckBinInteraction.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <memory>
#include <string>

namespace warpx::radiation
{
struct ImplicitGroupMomentTransportOptions
{
    EnergyGroupsExecutor groups;
    amrex::Real tolerance = 1.e-10;
    amrex::Real linear_tolerance = 1.e-13;
    amrex::Real minimum_optical_depth = 8;
    int max_linear_iterations = 500;
    bool verbose = false;
    // Explicit bounded RZ P1 optical mirror, not exact kinetic half-space
    // wall pressure. The native caller separately owns material wall rules.
    bool reflecting_boundaries = false;
    // Explicit research qualification only, initially Cartesian periodic.
    // Keep the production/default absorption beta*chi_t*L <= .01 guard;
    // this opt-in tests the exact beta^2-opacity source terms and moving AP
    // spatial closure, not a claim of qualified sharp trapped fronts.
    bool experimental_dynamic_diffusion = false;
    // Keep the accepted LAB moment/pressure/face reconstruction centered too.
    // This changes floating-point evaluation only; the default is unchanged.
    bool experimental_centered_reconstruction = false;
};

struct ImplicitGroupMomentTransportResult
{
    bool valid = false;
    std::string reason;
    int linear_iterations = 0;
    amrex::Real equation_residual = 0;
    amrex::Real inventory_residual = 0;
    amrex::Real angular_inventory_residual = 0;
    // Signed outward wall inventory and local cylindrical connection terms,
    // summed across groups, in (E,c*momentum) units J. RZ radial/theta rows
    // close only after boundary - geometry; neither ordinary sum is a global
    // Cartesian momentum invariant. The mean-radius theta ledger is separate.
    FourVector boundary_exchange{};
    FourVector geometric_exchange{};
    amrex::MultiFab radiation;
    amrex::MultiFab comoving_radiation;
    // Per-group LAB extensive (energy,c*impulse), J, source only: spatial
    // redistribution is excluded from material heat/recoil.
    amrex::MultiFab material_transfer;
    amrex::MultiFab transfer_roundoff;
    amrex::MultiFab physical_heat;
    // Summed heat/projection/momentum independent arithmetic envelopes.
    amrex::MultiFab arithmetic;
    // 4N conservative LAB face currents in energy units [W/m^2]: energy
    // current followed by c times each momentum current. The same currents
    // define the local residual, periodic inventory and bounded RZ wall ledger.
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> face_current;
};

/** One immutable fixed-temperature/secant spatial+source trial. Radiation is
 * LAB extensive (E_g,V F_g/c); proper chi_t/chi_a have N cell components,
 * beta has three, and positive material/color temperature one. Exact finite-
 * bin LAB moment, pressure and source matrices reconstruct independent
 * COMOVING positive Planck-bin dipoles. LTE-centered backward Euler solves
 * ALL spatial/source rows together, never a separate opacity damping stage.
 * Conservative AP energy-face correction converts the static centered moment
 * gradient into the existing mean-face-opacity five-point diffusion stencil;
 * it vanishes at second order under smooth spatial refinement. Finite-bin
 * boosted enthalpy/pressure carries compression across proper-frequency edges.
 * Native caloric/recoil feedback belongs to the enclosing private stage.
 * Initial scope: double precision, periodic Cartesian single level or explicit
 * axis/reflecting-radial, periodic-z RZ P1 optical mirror, <=8 groups,
 * chi_t*min(dx)>=8, beta<=.01 and, by default with absorption,
 * beta*chi_t*L<=.01. Experimental Cartesian dynamic diffusion requires an
 * explicit opt-in; it is not production runtime activation. Every
 * failure preserves all caller fields, including ghosts. */
ImplicitGroupMomentTransportResult TryImplicitGroupMomentTransport (
    amrex::MultiFab const &old_lab, amrex::MultiFab const &beta,
    amrex::MultiFab const &cell_temperature, amrex::MultiFab const &transport,
    amrex::MultiFab const &absorption, amrex::Geometry const &geometry, amrex::Real dt,
    ImplicitGroupMomentTransportOptions const &options);
} // namespace warpx::radiation
#endif
