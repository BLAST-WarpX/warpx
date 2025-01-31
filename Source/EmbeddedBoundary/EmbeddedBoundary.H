/* Copyright 2021-2025 Lorenzo Giacomel, Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_EMBEDDED_BOUNDARY_EMBEDDED_BOUNDARY_H_
#define WARPX_EMBEDDED_BOUNDARY_EMBEDDED_BOUNDARY_H_

#include "Enabled.H"

#ifdef AMREX_USE_EB

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_EBFabFactory.H>
#include <AMReX_REAL.H>

#include <array>

namespace warpx::embedded_boundary
{
    /**
    * \brief Compute the length of the mesh edges. Here the length is a value in [0, 1].
    *        An edge of length 0 is fully covered.
    */
    void ComputeEdgeLengths (
        ablastr::fields::VectorField& edge_lengths,
        const amrex::EBFArrayBoxFactory& eb_fact);
    /**
    * \brief Compute the area of the mesh faces. Here the area is a value in [0, 1].
    *        An edge of area 0 is fully covered.
    */
    void ComputeFaceAreas (
        ablastr::fields::VectorField& face_areas,
        const amrex::EBFArrayBoxFactory& eb_fact);

    /**
    * \brief Scale the edges lengths by the mesh width to obtain the real lengths.
    */
    void ScaleEdges (
        ablastr::fields::VectorField& edge_lengths,
        const std::array<amrex::Real,3>& cell_size);
    /**
    * \brief Scale the edges areas by the mesh width to obtain the real areas.
    */
    void ScaleAreas (
        ablastr::fields::VectorField& face_areas,
        const std::array<amrex::Real,3>& cell_size);
}

#endif

#endif //WARPX_EMBEDDED_BOUNDARY_EMBEDDED_BOUNDARY_H_
