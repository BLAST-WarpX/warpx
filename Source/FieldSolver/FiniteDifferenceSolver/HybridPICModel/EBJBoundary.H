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
    /** \brief Cached classification of the embedded-boundary fill for a
     * staggered vector field: for every point of each component, whether it
     * belongs to the solution domain, is a fill target (mirror band or a cut
     * point with covered center), is a fill target whose mirror-image
     * interpolation is ill-posed (resolved by a locked cascade after the
     * direct pass), or lies deep inside the conductor (zeroed).
     *
     * The classification depends only on the embedded-boundary geometry and
     * the update masks: callers own the cache and must reset() it whenever
     * the grids change (regrid / load balance). An empty cache passed to
     * ApplyPECBoundaryToField is built on first use.
     */
    struct EBFillStatus
    {
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> status;
        int n_pending = 0;
        [[nodiscard]] bool empty () const { return !status[0]; }
        void reset () { for (auto& s : status) { s.reset(); } n_pending = 0; }
    };

    /** \brief Enforce the PEC boundary condition on embedded-boundary masked
     * points of a staggered vector field (the hybrid solver's current
     * density, Ohm's-law electric field, and recovered magnetic field).
     *
     * On every point where the update mask is 0 the field is rewritten from
     * the level-set geometry (nodal signed distance \c distance_to_eb, < 0
     * inside the conductor, and the boundary normal from its gradient):
     *
     * - points deeper than a band of one cell inside the conductor are set
     *   to 0 (no fields inside a perfect conductor);
     * - points in the boundary band are filled from a mirror image point in
     *   the plasma. For edge fields (E, J; \c normal_odd = false) the
     *   tangential part is odd across the surface (zero at the wall) and the
     *   normal component is even (zero normal gradient). For the magnetic
     *   field (\c normal_odd = true) the parities are swapped: the normal
     *   component is odd (flux-excluding PEC, B.n = 0 at the wall) and the
     *   tangential part is even. Either way the PEC condition is enforced at
     *   the actual embedded boundary to second order.
     *
     * Two fill strategies are available. With \c direct_fill the mirrored
     * interpolation uses only solution-domain values, with the stencil
     * weights renormalized over those points. Fill targets whose image
     * stencil retains less than half of its weight in the solution domain
     * are classified ill-posed (once, in the cached \c EBFillStatus) and are
     * instead resolved after the direct pass by a deterministic cascade:
     * each sweep fills the pending points whose stencils reach already
     * locked values (direct-pass results or earlier sweeps), and points
     * that remain enclosed by other pending points after \c max_iters
     * sweeps are zeroed. Without \c direct_fill the band is relaxed with
     * Jacobi sweeps (each preceded by a ghost-cell exchange) until the
     * largest change, measured relative to the largest field magnitude in
     * the band, drops below \c rtol or \c max_iters sweeps have been
     * performed.
     *
     * With \c fill_covered_centers (used for the algebraic Ohm's-law E and
     * the current density), unmasked cut edges whose centers lie on or
     * inside the embedded surface are also fill targets: the generalized
     * Ohm's law evaluates them at their (covered) centers, where the
     * interpolated density and current are not meaningful, and the
     * resulting spurious values would otherwise be integrated into B by the
     * conformal Faraday update. Solution-domain points are never modified.
     *
     * \param[in,out] field staggered vector field (all 3 components)
     * \param[in] eb_update per-component embedded-boundary update masks
     * \param[in] distance_to_eb nodal signed-distance function to the boundary
     * \param[in] geom level geometry
     * \param[in] rtol residual tolerance of the band relaxation
     * \param[in] max_iters maximum number of Jacobi sweeps (also caps the
     *            ill-posed resolution cascade of the direct fill)
     * \param[in] direct_fill use the single-pass masked-stencil fill instead
     *            of the Jacobi relaxation
     * \param[in] normal_odd reflect the normal component oddly and the
     *            tangential part evenly (magnetic-field parity) instead of
     *            the converse (electric-field/current parity)
     * \param[in] fill_covered_centers also fill unmasked points whose
     *            centers are on or inside the embedded surface
     * \param[in,out] status_cache caller-owned fill classification, built on
     *            first use and reusable until the grids change (may be
     *            nullptr to classify locally on every call)
     */
    void ApplyPECBoundaryToField (
        ablastr::fields::VectorField const& field,
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update,
        amrex::MultiFab const& distance_to_eb,
        amrex::Geometry const& geom,
        amrex::Real rtol,
        int max_iters,
        bool direct_fill,
        bool normal_odd = false,
        bool fill_covered_centers = false,
        EBFillStatus* status_cache = nullptr);

    /** \brief Enforce an embedded-boundary condition on a nodal scalar field
     * (the hybrid solver's ion charge density and electron pressure).
     *
     * The nodal level set \c distance_to_eb is collocated with the field, so
     * every node with a nonpositive signed distance s (on or inside the
     * conductor surface) is rewritten from a mirror image point in the fluid:
     *
     * - \c odd = true: the value is reflected oddly, vanishing at the surface
     *   (Dirichlet 0; ghost values inside the conductor have the opposite
     *   sign of the fluid values, e.g. a negative mirrored charge density);
     * - \c odd = false: the value is reflected evenly (zero normal gradient,
     *   Neumann; e.g. the wall supports the plasma pressure).
     *
     * Nodes deeper than one cell inside the conductor are set to 0. The
     * mirrored interpolation uses only fluid nodes (positive level set),
     * with the stencil weights renormalized over those nodes; since the
     * written nodes are never part of the gather stencil, a single
     * deterministic pass is exact. Fluid nodes are never modified.
     *
     * \param[in,out] field nodal scalar field
     * \param[in] distance_to_eb nodal signed-distance function to the boundary
     * \param[in] geom level geometry
     * \param[in] odd reflect oddly (Dirichlet 0 at the surface) instead of
     *            evenly (zero normal gradient)
     */
    void ApplyEBBoundaryToNodalScalar (
        amrex::MultiFab& field,
        amrex::MultiFab const& distance_to_eb,
        amrex::Geometry const& geom,
        bool odd);
}

#endif // WARPX_EBJBOUNDARY_H_
