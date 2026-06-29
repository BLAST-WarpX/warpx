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
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
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

        /** Number of S_CORNER cells flagged by the concave re-entrant-corner
         * detector (the Option-2 covered-B fill skip). Diagnostic count of the
         * measure-zero ring where the fill reverts to its OFF value; 0 unless
         * the corner-skip detector is enabled. */
        int n_corner = 0;

        /** 2nd-order gather weight cache (quadratic_gather only). The gather
         * weights are geometry-only (the image point, the per-component
         * staggering, and which window cells are fluid -- not the field values),
         * so they are built once (\c weights_built) and reused every substage;
         * invalidated on regrid via reset(). For each fill-target component c,
         * \c gslot maps an S_FILL cell to its compact band-sparse slot (or -1).
         * The fill combines the 3 gathered field components at the image point;
         * for each slot and each gathered component g the gather collapses to a
         * sparse dot value_g = sum_t w * field_g(fill_cell + offset). \c gtap_n[c]
         * holds the per-(slot,g) tap count; \c gtap_off[c]/\c gtap_w[c] pack, per
         * (slot,g) and padded to \c maxtap, the (packed neighbour offset, weight)
         * of each tap. The apply therefore does no matrix build, Cholesky, or
         * basis evaluation. Memory scales with the wall band, not the volume. */
        bool weights_built = false;
        int maxtap = 0;
        std::array<int, 3> n_slots = {{0, 0, 0}};
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> gslot;
        std::array<amrex::Gpu::DeviceVector<int>, 3> gtap_n;
        std::array<amrex::Gpu::DeviceVector<int>, 3> gtap_off;
        std::array<amrex::Gpu::DeviceVector<amrex::Real>, 3> gtap_w;

        /** Geometry-only mirror outputs cached per band slot (quadratic_gather
         * only). mirror_geom() reads the static level-set phi (a second
         * scattered field, ~32 reads/cell, plus a sqrt and the normal-gradient
         * stencil) and produces quantities -- signed distance, image distance,
         * unit boundary normal, point/image positions -- that depend ONLY on
         * the embedded geometry, not on the field values. They are therefore
         * identical on every substage and are computed once in the build pass
         * (alongside the tap cache) and read in the apply path, so the apply
         * never touches phi. Invalidated on regrid via reset() (assumes a
         * static EB: the production case has no moving/deforming wall).
         * Layout per slot: [s, d_im, nv[AMREX_SPACEDIM], xe[AMREX_SPACEDIM],
         * xim[AMREX_SPACEDIM]]. */
        static constexpr int geom_nr = 2 + 3*AMREX_SPACEDIM;
        std::array<amrex::Gpu::DeviceVector<amrex::Real>, 3> ggeom;

        [[nodiscard]] bool empty () const { return !status[0]; }
        void reset () {
            for (auto& s : status) { s.reset(); }
            for (auto& s : gslot) { s.reset(); }
            for (auto& g : gtap_n) { g.clear(); }
            for (auto& g : gtap_off) { g.clear(); }
            for (auto& g : gtap_w) { g.clear(); }
            for (auto& g : ggeom) { g.clear(); }
            n_slots = {{0, 0, 0}};
            maxtap = 0;
            weights_built = false;
            n_pending = 0;
            n_corner = 0;
        }
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
     *            nullptr to classify locally on every call). Because the
     *            classification depends on \c band_cells, a cache must not be
     *            shared between calls with different band widths.
     * \param[in] band_cells width of the mirror-fill band in cells (default 1,
     *            the axis-aligned face reach). A field whose downstream stencil
     *            is isotropic (reaches the diagonal/corner neighbors) passes
     *            sqrt(AMREX_SPACEDIM) so the corner edges are mirror-filled
     *            instead of left in the zeroed deep interior.
     * \param[in] eb_cyl apply the cylindrical (surface-of-revolution) radial
     *            metric correction: the radial and azimuthal parts of the
     *            reflected field are scaled by the radial Jacobian
     *            r_image/r_fill (the leading O(h/R) curved-wall correction the
     *            planar mirror misses). 3D Cartesian only; the wall must be a
     *            surface of revolution about \c eb_cyl_axis centered on the
     *            transverse origin. Default false = the planar reflection.
     * \param[in] eb_cyl_axis symmetry axis for \c eb_cyl (0=x, 1=y, 2=z)
     * \param[in] quadratic_gather use the 2nd-order fluid-only quadratic gather
     *            (covered-B b-curl-fill) instead of the linear band-decoupled
     *            image gather
     * \param[in] normal_weight_floor lower clamp on the wall-NORMAL reflection
     *            weight \c w_n, applied only in the magnetic (\c normal_odd)
     *            branch. The unclamped weight is g.s/d_im (= -1 for the
     *            quadratic gather, in (-1,0) for the linear path), which gives
     *            the covered normal component an ODD reflection (a full sign
     *            reversal across the wall). Passing a floor of 0 clamps
     *            w_n -> max(g.s/d_im, 0) = 0, driving the covered normal
     *            component toward 0 (the physical PEC B_normal -> 0) instead of
     *            reversing it, while leaving the tangential even reflection
     *            (\c w_t) untouched. Default -1e30 = no clamp = byte-identical.
     * \param[in] cut_blend near-wall stability blend for the covered-B
     *            \c quadratic_gather fill, active only at CUT faces (update mask
     *            != 0, i.e. the face has a positive area fraction for which the
     *            conformal ECT B push already computed a value). At a cut face
     *            the mirror would DISCARD that stabler ECT-solved B and replace
     *            it with a steeper near-wall extrapolation; on a dense plasma
     *            shell pinned against the wall this injects a near-wall radial-B
     *            feature that stiffens the RKF45 substepper. With cut_blend in
     *            (0,1] the cut face is instead written as
     *            (1-cut_blend)*B_mirror + cut_blend*B_ECT, where B_ECT is the
     *            value already sitting in \c field at the fill point (EvolveB
     *            wrote it); cut_blend = 1 keeps the pure ECT value, cut_blend = 0
     *            keeps the pure mirror. Fully-covered faces (mask == 0, no ECT
     *            value) always take the full mirror. Default 0 = full mirror at
     *            every face = byte-identical. No effect unless \c quadratic_gather.
     * \param[in] cut_clamp relative cap on the cut-face mirror's deviation from
     *            the conformal-ECT value (the quadratic gather can overshoot the
     *            ECT value wildly on a stiff shell). With cut_clamp > 0 the cut
     *            face is written as B_ECT + clamp(B_mirror - B_ECT, +/- cap),
     *            cap = cut_clamp * max(|B_ECT|, |B_image|), so a gentle (small)
     *            mirror correction passes unclamped (2nd order preserved) while a
     *            stiff overshoot is bounded. Composes with \c cut_blend. Default
     *            0 = no clamp = byte-identical. No effect unless \c quadratic_gather.
     * \param[in] corner_skip Option-2 concave re-entrant-corner skip for the
     *            covered-B \c quadratic_gather fill. When true, the fill
     *            classification flags as S_CORNER every near-wall covered/cut
     *            cell that the validated three-clause detector marks as a
     *            concave re-entrant corner (a fiducial wall-normal bend > 30
     *            degrees, a load-bearing level-set radius jump to a fluid
     *            curl-difference neighbour across an opposite-sign phi, and the
     *            mirror image within reach), and the direct pass SKIPS the
     *            covered-B write there so the cell keeps its stable pre-fill
     *            (OFF / conformal-ECT) value. This removes the destabilizing
     *            cross-wall B jump whose Ampere curl crashes the RKF45
     *            substepper at a stepdown corner. Default false = no skip =
     *            byte-identical. No effect unless \c quadratic_gather.
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
        EBFillStatus* status_cache = nullptr,
        amrex::Real band_cells = 1.0,
        bool eb_cyl = false,
        int eb_cyl_axis = 2,
        bool quadratic_gather = false,
        amrex::Real normal_weight_floor = -1.e30,
        amrex::Real cut_blend = 0.0,
        amrex::Real cut_clamp = 0.0,
        bool corner_skip = false);

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
