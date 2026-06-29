/* Copyright 2020 Remi Lehe, Lorenzo Giacomel
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FiniteDifferenceSolver.H"

#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
#   include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#endif
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/utils/Communication.H>

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <memory>

using namespace amrex;
using namespace ablastr::fields;

// The helper is used only by EvolveRhoCartesianECT below (same guard), so it is
// scoped identically to avoid an unused-function warning in the RZ/1D TUs.
#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE) && defined(AMREX_USE_EB)
namespace {
    /** Along-edge curvature correction for the conformal-ECT Faraday circulation.
     *
     *  EvolveRhoCartesianECT forms each face EMF by summing E sampled at the
     *  full-edge center, times the cut edge_length. That is a 1st-order midpoint
     *  quadrature of the circulation over the curved contour: on a cut edge, E
     *  should be evaluated at the centroid of the *uncovered* segment, not the
     *  edge center. This applies the cheap along-edge Taylor shift to that
     *  centroid:
     *      E_c = E(center) + 0.5 * delta * (E(+1) - E(-1)),
     *  where (+1,-1) are the parallel edges one cell away along the edge's own
     *  axis (`dir`), and `delta` (the uncovered-segment centroid offset in
     *  full-cell units) is read from `off`. AMReX's edge centroid
     *  (intercept_to_edge_centroid) is already that offset in cell units, so
     *  `delta = off(i,j,k)` directly (the cell-size dx cancels against the
     *  centered slope). `delta` is 0 on uncut/covered edges, so the result is
     *  byte-identical to the unshifted sample there; `apply_curvature == false`
     *  (EM-ECT, or the opt-in flag off) returns the unshifted sample everywhere.
     *
     *  The along-edge slope dE/ds*dx is a centered difference when both neighbor
     *  edges are open. At a convex curved wall the OUTWARD along-edge neighbor of
     *  a cut edge is typically covered, so a centered-only stencil would disable
     *  the correction on exactly the cut edges that need it; we therefore fall
     *  back to a ONE-SIDED slope using whichever neighbor is open (still 2nd-order
     *  accurate for the O(h) sample-point shift). Only a fully isolated cut edge
     *  (both along-edge neighbors covered) gets no shift. */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real ect_edge_E (
        bool apply_curvature, int i, int j, int k, int dir,
        amrex::Array4<amrex::Real const> const& E,
        amrex::Array4<amrex::Real const> const& off,
        amrex::Array4<amrex::Real const> const& len)
    {
        amrex::Real const e = E(i, j, k);
        if (!apply_curvature) { return e; }
        amrex::Real const delta = off(i, j, k);
        if (delta == amrex::Real(0.0)) { return e; }
        // Step one cell along the edge's own physical axis (dir: 0=x, 1=y, 2=z).
        // In 2D XZ the z-axis is the AMReX j index (k is degenerate); dir==1 (y)
        // never occurs there (the XZ circulation uses only Ex and Ez).
        int ip = i, jp = j, kp = k, im = i, jm = j, km = k;
        if (dir == 0) { ip = i + 1; im = i - 1; }
#if defined(WARPX_DIM_3D)
        else if (dir == 1) { jp = j + 1; jm = j - 1; }
        else { kp = k + 1; km = k - 1; }
#elif defined(WARPX_DIM_XZ)
        else { jp = j + 1; jm = j - 1; }
#endif
        bool const plus_open  = len(ip, jp, kp) > amrex::Real(0.0);
        bool const minus_open = len(im, jm, km) > amrex::Real(0.0);
        // slope_dx approximates (dE/ds)*dx at the edge center (cell size cancels
        // against the dimensionless delta).
        amrex::Real slope_dx;
        if (plus_open && minus_open) {
            slope_dx = amrex::Real(0.5) * (E(ip, jp, kp) - E(im, jm, km));
        } else if (plus_open) {
            slope_dx = E(ip, jp, kp) - e;
        } else if (minus_open) {
            slope_dx = e - E(im, jm, km);
        } else {
            return e; // isolated cut edge: no usable along-edge slope
        }
        return e + delta * slope_dx;
    }
}
#endif

/**
 * \brief Update the B field, over one timestep
 */
void FiniteDifferenceSolver::EvolveECTRho (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& edge_lengths,
    ablastr::fields::VectorField const& face_areas,
    ablastr::fields::VectorField const& ECTRhofield,
    const int lev,
    ablastr::fields::VectorField const* edge_cent_offset,
    bool apply_curvature) {

#if !defined(WARPX_DIM_RZ) and !defined(WARPX_DIM_RCYLINDER) and !defined(WARPX_DIM_RSPHERE) and defined(AMREX_USE_EB)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::ECT ||
        (m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC &&
         WarpX::UseConformalEBSolve())) {

        EvolveRhoCartesianECT(Efield, edge_lengths, face_areas, ECTRhofield, lev,
                              edge_cent_offset, apply_curvature);

    }
#else
    amrex::ignore_unused(Efield, edge_lengths, face_areas, ECTRhofield, lev,
                         edge_cent_offset, apply_curvature);
#endif
}

// If we implement ECT in 1D we will need to take care of this #ifndef differently
#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
void FiniteDifferenceSolver::EvolveRhoCartesianECT (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& edge_lengths,
    ablastr::fields::VectorField const& face_areas,
    ablastr::fields::VectorField const& ECTRhofield, const int lev,
    ablastr::fields::VectorField const* edge_cent_offset,
    bool apply_curvature ) {
#ifdef AMREX_USE_EB

    // Opt-in along-edge curvature correction of the circulation (see ect_edge_E).
    // Needs the per-edge uncovered-segment centroid offsets; if they were not
    // provided, fall back to the (byte-identical) unshifted quadrature.
    bool const do_curv = apply_curvature && (edge_cent_offset != nullptr);
    // A valid VectorField to source the offset Array4s from. When the correction
    // is off this aliases edge_lengths (a placeholder never read, since do_curv
    // gates every access), keeping the Array4 types uniform without a null deref.
    ablastr::fields::VectorField const& off_src =
        do_curv ? *edge_cent_offset : edge_lengths;

#if !(defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ))
    WARPX_ABORT_WITH_MESSAGE(
        "EvolveRhoCartesianECT: Embedded Boundaries are only implemented in 3D and XZ");
#endif

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*ECTRhofield[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers) {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Extract field data for this grid/tile
        amrex::Array4<amrex::Real> const &Ex = Efield[0]->array(mfi);
        amrex::Array4<amrex::Real> const &Ey = Efield[1]->array(mfi);
        amrex::Array4<amrex::Real> const &Ez = Efield[2]->array(mfi);
        amrex::Array4<amrex::Real> const &Rhox = ECTRhofield[0]->array(mfi);
        amrex::Array4<amrex::Real> const &Rhoy = ECTRhofield[1]->array(mfi);
        amrex::Array4<amrex::Real> const &Rhoz = ECTRhofield[2]->array(mfi);
        amrex::Array4<amrex::Real> const &lx = edge_lengths[0]->array(mfi);
        amrex::Array4<amrex::Real> const &ly = edge_lengths[1]->array(mfi);
        amrex::Array4<amrex::Real> const &lz = edge_lengths[2]->array(mfi);
        // Uncovered-segment centroid offsets for the along-edge curvature shift
        // (placeholders aliasing edge_lengths when the correction is off).
        amrex::Array4<amrex::Real const> const ox = off_src[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const oy = off_src[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const oz = off_src[2]->const_array(mfi);
        amrex::Array4<amrex::Real> const &Sx = face_areas[0]->array(mfi);
        amrex::Array4<amrex::Real> const &Sy = face_areas[1]->array(mfi);
        amrex::Array4<amrex::Real> const &Sz = face_areas[2]->array(mfi);

        // Extract tileboxes for which to loop
        amrex::Box const &trhox = mfi.tilebox(ECTRhofield[0]->ixType().toIntVect());
        amrex::Box const &trhoy = mfi.tilebox(ECTRhofield[1]->ixType().toIntVect());
        amrex::Box const &trhoz = mfi.tilebox(ECTRhofield[2]->ixType().toIntVect());

        amrex::ParallelFor(trhox, trhoy, trhoz,

            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (Sx(i, j, k) <= 0) { return; }

// If we implement ECT in 1D we will need to take care of this #ifndef differently
#ifndef WARPX_DIM_XZ
                Rhox(i, j, k) = (
                    ect_edge_E(do_curv, i, j,     k,     1, Ey, oy, ly) * ly(i, j,     k    )
                  - ect_edge_E(do_curv, i, j,     k + 1, 1, Ey, oy, ly) * ly(i, j,     k + 1)
                  + ect_edge_E(do_curv, i, j + 1, k,     2, Ez, oz, lz) * lz(i, j + 1, k    )
                  - ect_edge_E(do_curv, i, j,     k,     2, Ez, oz, lz) * lz(i, j,     k    )
                  ) / Sx(i, j, k);
#endif
            },

            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (Sy(i, j, k) <= 0) { return; }

#ifdef WARPX_DIM_XZ
                Rhoy(i, j, k) = (
                    ect_edge_E(do_curv, i,     j,     k, 2, Ez, oz, lz) * lz(i,     j,     k)
                  - ect_edge_E(do_curv, i + 1, j,     k, 2, Ez, oz, lz) * lz(i + 1, j,     k)
                  + ect_edge_E(do_curv, i,     j + 1, k, 0, Ex, ox, lx) * lx(i,     j + 1, k)
                  - ect_edge_E(do_curv, i,     j,     k, 0, Ex, ox, lx) * lx(i,     j,     k)
                  ) / Sy(i, j, k);
#elif defined(WARPX_DIM_3D)
                Rhoy(i, j, k) = (
                    ect_edge_E(do_curv, i,     j, k,     2, Ez, oz, lz) * lz(i,     j, k    )
                  - ect_edge_E(do_curv, i + 1, j, k,     2, Ez, oz, lz) * lz(i + 1, j, k    )
                  + ect_edge_E(do_curv, i,     j, k + 1, 0, Ex, ox, lx) * lx(i,     j, k + 1)
                  - ect_edge_E(do_curv, i,     j, k,     0, Ex, ox, lx) * lx(i,     j, k    )
                  ) / Sy(i, j, k);
#else
                WARPX_ABORT_WITH_MESSAGE("EvolveRhoCartesianECT: Embedded Boundaries are only implemented in 3D and XZ");
#endif
            },

            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (Sz(i, j, k) <= 0) { return; }

// If we implement ECT in 1D we will need to take care of this #ifndef differently
#ifndef WARPX_DIM_XZ
                Rhoz(i, j, k) = (
                    ect_edge_E(do_curv, i,     j,     k, 0, Ex, ox, lx) * lx(i,     j,     k)
                  - ect_edge_E(do_curv, i,     j + 1, k, 0, Ex, ox, lx) * lx(i,     j + 1, k)
                  + ect_edge_E(do_curv, i + 1, j,     k, 1, Ey, oy, ly) * ly(i + 1, j,     k)
                  - ect_edge_E(do_curv, i,     j,     k, 1, Ey, oy, ly) * ly(i,     j,     k)
                  ) / Sz(i, j, k);
#endif
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
#ifdef WARPX_DIM_XZ
        amrex::ignore_unused(Ey, Rhox, Rhoz, ly, oy);
#endif
    }

    // With cut cells at fab seams the enlarged-cell gather in the B push
    // reads neighbor face EMFs across boxes: refresh one ghost layer so
    // those reads see the owner-computed circulations
    auto& warpx = WarpX::GetInstance();
    if (warpx.ECTNeedsSeamSync()) {
        const auto& period = warpx.Geom(lev).periodicity();
        for (int idim = 0; idim < 3; ++idim) {
            // ECTRho components are face-centered (B staggering), so no
            // nodal-seam reconciliation is needed: leave nodal_sync default.
            ablastr::utils::communication::FillBoundary(
                *ECTRhofield[idim], amrex::IntVect(1),
                WarpX::do_single_precision_comms, period);
        }
    }
#else
    amrex::ignore_unused(Efield, edge_lengths, face_areas, ECTRhofield, lev);
#endif
}
#endif
