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
     * for a staggered vector field (solution domain, fill target, ill-posed
     * fill target, or zeroed deep interior). Depends on the geometry, the
     * update masks, and the fill mode, so a cache must not be shared between
     * calls with different modes; callers reset() it on regrid.
     */
    struct EBFillStatus
    {
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> status;
        int n_pending = 0;

        [[nodiscard]] bool empty () const { return !status[0]; }
        void reset () {
            for (auto& s : status) { s.reset(); }
            n_pending = 0;
        }
    };

    /** \brief Enforce the PEC condition on embedded-boundary masked points of
     * a staggered vector field: points deeper than one cell inside the
     * conductor are zeroed, band points are filled from a fluid mirror image
     * with the PEC parities (edge fields: tangential odd / normal even;
     * \c normal_odd swaps them for B), second order at the actual surface.
     *
     * The mirrored interpolation uses only solution-domain values with
     * renormalized weights; targets retaining less than half their stencil
     * weight are resolved by a deterministic cascade (then zeroed if stalled).
     * With \c fill_covered_centers, unmasked cut edges whose centers lie on
     * or inside the surface are also filled (the algebraic Ohm's-law E and
     * the current are not meaningful at covered centers). \c band_cells
     * widens the fill band for fields whose downstream stencils reach deeper
     * covered points. \c status_cache may be nullptr (classify per call).
     */
    void ApplyPECBoundaryToField (
        ablastr::fields::VectorField const& field,
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update,
        amrex::MultiFab const& distance_to_eb,
        amrex::Geometry const& geom,
        bool normal_odd = false,
        bool fill_covered_centers = false,
        EBFillStatus* status_cache = nullptr,
        amrex::Real band_cells = 1.0);

    /** \brief Enforce an embedded-boundary condition on a nodal scalar (ion
     * charge density, electron pressure): nodes on or inside the surface are
     * rewritten from a fluid mirror image -- \c odd reflects oddly (Dirichlet
     * 0 at the wall), even reflection gives Neumann -- and nodes deeper than
     * one cell are zeroed. Fluid nodes are never modified.
     */
    void ApplyEBBoundaryToNodalScalar (
        amrex::MultiFab& field,
        amrex::MultiFab const& distance_to_eb,
        amrex::Geometry const& geom,
        bool odd);

    /** \brief Fold the deposit collected by covered points of a nodal scalar
     * (the charge density) back across the embedded surface with the PEC
     * image parity (subtract: opposite-sign image charge), matching the
     * domain-boundary fold. Call after the deposit and guard sums, before
     * ApplyEBBoundaryToNodalScalar overwrites the covered points.
     */
    void FoldEBDepositToNodalScalar (
        amrex::MultiFab& field,
        amrex::MultiFab const& distance_to_eb,
        amrex::Geometry const& geom);

    /** \brief Fold the deposit collected by covered points of a staggered
     * vector field (the deposited ion current) back across the surface with
     * the PEC image-current parities (tangential subtracted, normal added),
     * matching the domain-boundary fold. Covered points come from the fill
     * classification. Call after the deposit, before ApplyPECBoundaryToField.
     */
    void FoldEBDepositToField (
        ablastr::fields::VectorField const& field,
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update,
        amrex::MultiFab const& distance_to_eb,
        amrex::Geometry const& geom,
        EBFillStatus* status_cache = nullptr);
}

#endif // WARPX_EBJBOUNDARY_H_
