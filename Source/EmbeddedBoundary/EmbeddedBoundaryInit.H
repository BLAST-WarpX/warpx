/* Copyright 2021-2025 Lorenzo Giacomel, Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_EMBEDDED_BOUNDARY_EMBEDDED_BOUNDARY_INIT_H_
#define WARPX_EMBEDDED_BOUNDARY_EMBEDDED_BOUNDARY_INIT_H_

#include "Enabled.H"

#ifdef AMREX_USE_EB

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_EBFabFactory.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Periodicity.H>
#include <AMReX_REAL.H>

#include <array>

namespace warpx::embedded_boundary
{

    /** \brief Set a flag to indicate in which cells a particle should deposit charge/current
     *  with a reduced, order 1 shape.
     *
     * More specifically, the flag is set to 1 if any of the neighboring cells over which the
     * particle shape might extend are either partially or fully covered by an embedded boundary.
     * This ensures that a particle in this cell deposits with an order 1 shape, which in turn
     * makes sure that the particle never deposits any charge in a partially or fully covered cell.
     *
     * \param[in] eb_reduce_particle_shape multifab to be filled with 1s and 0s
     * \param[in] eb_fact EB factory
     * \param[in] particle_shape_order order of the particle shape function
     * \param[in] periodicity TODO Geom(0).periodicity()
     */
    void MarkReducedShapeCells (
        std::unique_ptr<amrex::iMultiFab> & eb_reduce_particle_shape,
        amrex::EBFArrayBoxFactory const & eb_fact,
        int particle_shape_order,
        const amrex::Periodicity& periodicity);

    /** \brief Set a flag to indicate on which grid points the field `field`
     *  should be updated, depending on their position relative to the embedded boundary.
     *
     * This function is used by all finite-difference solvers, except the
     * ECT solver, which instead uses `MarkUpdateECellsECT` and `MarkUpdateBCellsECT`.
     * It uses a stair-case approximation of the embedded boundary:
     * If a grid point touches cells that are either partially or fully covered
     * by the embedded boundary: the corresponding field is not updated.
     *
     * More specifically, this function fills the iMultiFabs in `eb_update`
     * (which have the same indexType as the MultiFabs in `field`) with 1
     * or 0, depending on whether the grid point should be updated or not.
     */
    void MarkUpdateCellsStairCase (
        std::array< std::unique_ptr<amrex::iMultiFab>,3> & eb_update,
        ablastr::fields::VectorField const & field,
        amrex::EBFArrayBoxFactory const & eb_fact );

    /** \brief Set a flag to indicate on which grid points the E field
     *  should be updated, depending on their position relative to the embedded boundary.
     *
     * This function is used by ECT solver. The E field is not updated if
     * the edge on which it is defined is fully covered by the embedded boundary.
     *
     * More specifically, this function fills the iMultiFabs in `eb_update_E`
     * (which have the same indexType as the E field) with 1 or 0, depending
     * on whether the grid point should be updated or not.
     */
    void MarkUpdateECellsECT (
        std::array< std::unique_ptr<amrex::iMultiFab>,3> & eb_update_E,
        ablastr::fields::VectorField const& edge_lengths );

    /** \brief Set a flag to indicate on which grid points the B field
     *  should be updated, depending on their position relative to the embedded boundary.
     *
     * This function is used by ECT solver. The B field is not updated if
     * the face on which it is defined is fully covered by the embedded boundary.
     *
     * More specifically, this function fills the iMultiFabs in `eb_update_B`
     * (which have the same indexType as the B field) with 1 or 0, depending
     * on whether the grid point should be updated or not.
     */
    void MarkUpdateBCellsECT (
        std::array< std::unique_ptr<amrex::iMultiFab>,3> & eb_update_B,
        ablastr::fields::VectorField const& face_areas,
        ablastr::fields::VectorField const& edge_lengths );

   /**
    * \brief Initialize information for cell extensions.
    *        The flags convention for m_flag_info_face is as follows
    *          - 0 for unstable cells
    *          - 1 for stable cells which have not been intruded
    *          - 2 for stable cells which have been intruded
    *        Here we cannot know if a cell is intruded or not so we initialize all stable cells with 1
    */
    void MarkExtensionCells(
        const std::array<amrex::Real,3>& cell_size,
        std::array< std::unique_ptr<amrex::iMultiFab>, 3 > & flag_info_face,
        std::array< std::unique_ptr<amrex::iMultiFab>, 3 > & flag_ext_face,
        const ablastr::fields::VectorField& b_field,
        const ablastr::fields::VectorField& face_areas,
        const ablastr::fields::VectorField& edge_lengths,
        const ablastr::fields::VectorField& area_mod);

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
