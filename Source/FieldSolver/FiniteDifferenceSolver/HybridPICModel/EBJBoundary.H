/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_EBJBOUNDARY_H_
#define WARPX_EBJBOUNDARY_H_

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <array>
#include <memory>

namespace warpx::hybrid
{
    /** \brief Enforce the PEC boundary condition on embedded-boundary masked
     * edges of an edge-staggered vector field (the hybrid solver's current
     * density and Ohm's-law electric field).
     *
     * On every edge where \c eb_update_E is 0 the field is rewritten from the
     * level-set geometry (nodal signed distance \c distance_to_eb, < 0 inside
     * the conductor, and the boundary normal from its gradient):
     *
     * - edges deeper than a band of one cell inside the conductor are set to 0
     *   (no volume current and no electric field inside a perfect conductor);
     * - edges in the boundary band are filled from a mirror image point in the
     *   plasma: the component normal to the boundary is even across the
     *   surface (zero normal gradient) while the tangential part is odd
     *   (zero at the surface), which enforces the PEC condition at the actual
     *   embedded boundary to second order.
     *
     * Because the interpolation stencil at the image point can itself reach
     * other masked edges, the band is relaxed with Jacobi sweeps (each
     * preceded by a ghost-cell exchange) until the largest change, measured
     * relative to the largest field magnitude in the band, drops below
     * \c rtol or \c max_iters sweeps have been performed.
     *
     * Unmasked (including conformally cut) edges are never modified.
     *
     * \param[in,out] field edge-staggered vector field (all 3 components)
     * \param[in] eb_update_E per-component embedded-boundary update masks
     * \param[in] distance_to_eb nodal signed-distance function to the boundary
     * \param[in] geom level geometry
     * \param[in] rtol relative residual tolerance of the band relaxation
     * \param[in] max_iters maximum number of Jacobi sweeps
     */
    void ApplyPECBoundaryToEdgeField (
        ablastr::fields::VectorField const& field,
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_E,
        amrex::MultiFab const& distance_to_eb,
        amrex::Geometry const& geom,
        amrex::Real rtol,
        int max_iters);
}

#endif // WARPX_EBJBOUNDARY_H_
