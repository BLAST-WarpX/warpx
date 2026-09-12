/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_IMPLICITMOMENTTRANSPORT_H_
#define WARPX_RADIATION_IMPLICITMOMENTTRANSPORT_H_

#include "MomentBoundaryExchange.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

namespace warpx::radiation
{
    using PrescribedMomentFaceFluxes =
        std::array<amrex::MultiFab const*, AMREX_SPACEDIM>;
    struct ImplicitMomentTransportOptions {
        int max_nonlinear_iterations = 50;
        int max_linear_iterations = 500;
        amrex::Real tolerance = 1.e-10;
        amrex::Real linear_tolerance = 1.e-13;
        bool verbose = false;
        // Experimental low-level fixed optical mirrors on all nonperiodic faces.
        // Mutually exclusive with prescribed boundary fluxes; no particle wall.
        bool reflecting_boundaries = false;
    };

    struct ImplicitMomentTransportResult {
        int nonlinear_iterations = 0;
        int linear_iterations = 0;
        amrex::Real equation_residual = 0;
        amrex::Real energy_residual = 0;
        amrex::Real momentum_residual = 0;
        // Accepted outward (energy,c*momentum); zero on rejected stages.
        FourVector boundary_exchange{};
        bool valid = false;
    };

    /** Joint backward-Euler gray M1 transport/source stage with prescribed
     * beta, opacity and equilibrium energy. Radiation and material transfer are
     * cell-integrated (energy,c*momentum). Opacities are proper inverse lengths.
     * Scope is Cartesian geometry and |beta| <= 0.01. Nonperiodic directions
     * require either externally prescribed lab (E,q) face fluxes, fixed throughout
     * the stage with radiation face layout and cell-volume normalization, or
     * opt-in fixed reflecting optical walls. The latter reevaluate their M1
     * half-space pressure on every nonlinear candidate, with zero physical
     * energy/tangential exchange. This low-level interface does not implement
     * vacuum, material-wall rules, or a user-facing native-PIC boundary path. Radiation
     * and material ghosts are not prescribed physical-boundary states.
     * Failure leaves
     * supplied outputs unchanged. This does not yet update native material or
     * establish production qualification for a moving-medium spatial scheme.
     * equation_residual is the maximum ratio to its physical/roundoff bound;
     * energy_residual and momentum_residual are raw relative global balances.
     */
    ImplicitMomentTransportResult TryImplicitMomentTransport (
        amrex::MultiFab& radiation, amrex::MultiFab const& beta, amrex::MultiFab const& absorption,
        amrex::MultiFab const& scattering, amrex::MultiFab const& equilibrium,
        amrex::MultiFab& material_transfer, amrex::Geometry const& geometry, amrex::Real dt,
        ImplicitMomentTransportOptions const& options = {},
        amrex::MultiFab* material_energy_minus_work = nullptr,
        PrescribedMomentFaceFluxes const* prescribed_boundary_fluxes = nullptr);

    /** Evaluate dt*div(flux) in components 0:3 and absolute Jacobian-weighted
     * face operand contributions in 4:7 for material-feedback rounding bounds.
     * The latter measure cancellation sensitivity, not the net flux magnitude.
     * Inputs (including ghosts) and output remain unchanged on failure. */
    bool ComputeMomentTransportIncrement (amrex::MultiFab const& radiation,
                                          amrex::MultiFab const& beta,
                                          amrex::MultiFab const& absorption,
                                          amrex::MultiFab const& scattering,
                                          amrex::MultiFab const& equilibrium,
                                          amrex::MultiFab& increment,
                                          amrex::Geometry const& geometry, amrex::Real dt);
} // namespace warpx::radiation
#endif
