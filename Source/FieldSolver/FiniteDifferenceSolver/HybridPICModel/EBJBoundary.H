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
    /** \brief Cached per-point classification of the embedded-boundary fill
     * for a staggered vector field: solution domain, fill target (mirror
     * band or covered-center cut point), fill target whose mirror-image
     * interpolation is ill-posed, or deep conductor interior (zeroed).
     *
     * The classification depends on the embedded-boundary geometry, the
     * update masks, and the \c fill_covered_centers mode, so a cache must
     * not be shared between calls with different modes. Callers own the
     * cache and must reset() it whenever the grids change (regrid / load
     * balance). An empty cache passed to ApplyPECBoundaryToField or
     * FoldEBDepositToField is built on first use.
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
     * locked values (direct-pass results or earlier sweeps); points still
     * pending when the cascade stalls or reaches \c max_iters sweeps are
     * zeroed. Without \c direct_fill the band is relaxed with Jacobi sweeps
     * (each preceded by a ghost-cell exchange) until the largest change,
     * measured relative to the largest field magnitude in the band, drops
     * below \c rtol or \c max_iters sweeps have been performed.
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
     * \param[in] normal_odd use the magnetic-field parities (normal odd,
     *            tangential even) instead of the edge-field parities
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

    /** \brief Apply the collocated-ECT (C-ECT) conformal correction to the
     * freshly nodal-pushed magnetic field on a collocated grid, so a curved
     * embedded boundary is honored with a finite-volume Faraday update instead
     * of a pointwise level-set mirror fill (which imprints a grid-locked m = 4
     * on circular walls).
     *
     * The core nodal Faraday push (warpx.EvolveB) has already advanced every
     * node by the wide-centered EvolveBCartesian curl
     *   dB_nodal_c = dt*(UpwardD_b(E_a) - UpwardD_a(E_b))
     * with (a, b) the in-plane axes of component c (Bx: (y,z), By: (z,x),
     * Bz: (x,y)); e.g. Bx: dt*(UpwardDz(Ey) - UpwardDy(Ez)). This routine ADDS
     * the per-node correction
     *   dB_corr_c = dB_fv_c - dB_nodal_c,
     * so the net increment is the FV value dB_fv_c on near-wall band nodes and
     * is bit-identically unchanged in the interior.
     *
     * For each node the dual face (+/- h/2 about the node in the (a, b) plane,
     * at the node's c-index) is cut by the nodal level set. Sampling the four
     * dual-face corners of the level set (each the bilinear cell-center average
     * of the four nodal \c distance_to_eb values bracketing it in the (a, b)
     * plane) classifies the node:
     *
     * - all four corners in the fluid (phi > 0): INTERIOR, no correction;
     * - all four corners in the conductor (phi <= 0): DEEP, B_c set to 0;
     * - mixed: BAND, the FV correction is applied.
     *
     * On a band node the cut dual cell has edge-lengths l = h*(phi>0 fraction)
     * (linear root t = phi_pos/(phi_pos - phi_neg)), cut area S_cut from a
     * Sutherland-Hodgman clip of the dual square against {phi >= 0} (shoelace),
     * ECT stable area S^stb = 0.5*h*max(l), and effective area
     * S_eff = max(S_cut, S^stb) (bounds dt*V/S, no small-cut blow-up). The cut
     * circulation V_cut sums the midpoint-averaged in-plane E along the cut
     * edges, oriented per component so the full-cell limit (l = h,
     * S_eff = h^2) reduces dB_fv_c to dB_nodal_c exactly:
     *   V_cut = E_a^top*l_top - E_a^bot*l_bot + E_b^left*l_left - E_b^right*l_right,
     *   dB_fv_c = dt*V_cut/S_eff.
     *
     * 2D XZ applies the correction only to the out-of-plane component By (whose
     * in-plane curl is circ(Ez, Ex)); the in-plane Bx, Bz are left to the core
     * nodal push (their Faraday curl uses the out-of-plane Ey through a 1-D
     * derivative). RZ and 1D are no-ops (the caller keeps the legacy path).
     *
     * The level set must carry at least two valid ghost cells (the caller
     * FillBoundary's it). This routine writes only band/deep nodes; the caller
     * is responsible for the following FillBoundaryB.
     *
     * \param[in,out] Bfield collocated (nodal) magnetic field (all 3 components)
     * \param[in] Efield collocated (nodal) electric field (all 3 components)
     * \param[in] distance_to_eb nodal signed-distance function to the boundary
     * \param[in] geom level geometry
     * \param[in] dt time step of the Faraday push
     */
    void ApplyConformalBCorrection (
        ablastr::fields::VectorField const& Bfield,
        ablastr::fields::VectorField const& Efield,
        amrex::MultiFab const& distance_to_eb,
        amrex::Geometry const& geom,
        amrex::Real dt);

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
     * \param[in] odd reflect oddly (Dirichlet 0) instead of evenly (Neumann)
     */
    void ApplyEBBoundaryToNodalScalar (
        amrex::MultiFab& field,
        amrex::MultiFab const& distance_to_eb,
        amrex::Geometry const& geom,
        bool odd);

    /** \brief Fold the deposit collected by covered points of a nodal scalar
     * field (the charge density) back across the embedded surface, with the
     * PEC image parity, before the covered points are overwritten by the
     * mirror fill.
     *
     * Particles within a shape length of a conducting wall deposit part of
     * their charge onto points on or inside the embedded surface; without a
     * fold that charge is discarded when the boundary condition rewrites the
     * covered points. Matching the domain treatment
     * (PEC::ApplyReflectiveBoundarytoRhofield), every fluid point within the
     * deposit reach of the surface gathers the covered-side deposit at its
     * mirror-image position inside the conductor (raw masked interpolation:
     * fluid stencil points contribute nothing) and SUBTRACTS it (image
     * charge of the opposite sign). With \c pec_images = false the
     * reflecting-wall parity is used instead and the deposit is ADDED
     * (mass-conserving fold for a wall-supported column, matching the
     * reflecting/thermal-particle branch of the domain treatment). In the
     * planar limit this reproduces the domain fold exactly and conserves
     * the folded amount.
     *
     * Call after the deposit (including guard-cell sums) and before
     * ApplyEBBoundaryToNodalScalar.
     *
     * \param[in,out] field nodal scalar field holding the raw deposit
     * \param[in] distance_to_eb nodal signed-distance function to the boundary
     * \param[in] geom level geometry
     * \param[in] pec_images fold with the PEC image parity (subtract)
     *            instead of the reflecting-wall parity (add)
     */
    void FoldEBDepositToNodalScalar (
        amrex::MultiFab& field,
        amrex::MultiFab const& distance_to_eb,
        amrex::Geometry const& geom,
        bool pec_images = true);

    /** \brief Fold the deposit collected by covered points of a staggered
     * vector field (the deposited ion current) back across the embedded
     * surface, with the PEC image parities, before the covered points are
     * overwritten by the mirror fill.
     *
     * Matching the domain treatment (PEC::ApplyReflectiveBoundarytoJfield),
     * the covered-side deposit gathered at the mirror image of each
     * near-surface fluid point is folded with the image-current parities:
     * the tangential part is SUBTRACTED (image current antiparallel) and the
     * normal part is ADDED. With \c pec_images = false the reflecting-wall
     * parities are used instead (tangential added, normal subtracted).
     * Covered points are identified by the fill classification (the write
     * set of ApplyPECBoundaryToField with \c fill_covered_centers), so cut
     * edges with covered centers fold too.
     *
     * Call after the deposit (including guard-cell sums) and before
     * ApplyPECBoundaryToField.
     *
     * \param[in,out] field staggered vector field holding the raw deposit
     * \param[in] eb_update per-component embedded-boundary update masks
     * \param[in] distance_to_eb nodal signed-distance function to the boundary
     * \param[in] geom level geometry
     * \param[in,out] status_cache caller-owned fill classification (see
     *            ApplyPECBoundaryToField); may be nullptr
     * \param[in] pec_images fold with the PEC image parities instead of the
     *            reflecting-wall parities
     */
    void FoldEBDepositToField (
        ablastr::fields::VectorField const& field,
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update,
        amrex::MultiFab const& distance_to_eb,
        amrex::Geometry const& geom,
        EBFillStatus* status_cache = nullptr,
        bool pec_images = true);
}

#endif // WARPX_EBJBOUNDARY_H_
