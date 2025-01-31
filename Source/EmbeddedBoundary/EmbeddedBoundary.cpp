/* Copyright 2021-2025 Lorenzo Giacomel, Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Enabled.H"

#ifdef AMREX_USE_EB

#include "EmbeddedBoundary.H"

#include "Utils/TextMsg.H"

#include <AMReX_BLProfiler.H>
#include <AMReX_Box.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>

namespace web = warpx::embedded_boundary;

void
web::ScaleEdges (ablastr::fields::VectorField& edge_lengths,
            const std::array<amrex::Real,3>& cell_size)
{
    BL_PROFILE("ScaleEdges");

#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE("ScaleEdges only implemented in 2D and 3D");
#endif

    for (int idim = 0; idim < 3; ++idim){
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        if (idim == 1) { continue; }
#endif
        for (amrex::MFIter mfi(*edge_lengths[0]); mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.tilebox(edge_lengths[idim]->ixType().toIntVect(),
                                                edge_lengths[idim]->nGrowVect() );
            auto const &edge_lengths_dim = edge_lengths[idim]->array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                edge_lengths_dim(i, j, k) *= cell_size[idim];
            });
        }
    }
}


void
web::ScaleAreas (ablastr::fields::VectorField& face_areas,
            const std::array<amrex::Real,3>& cell_size)
{
    BL_PROFILE("ScaleAreas");

#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE("ScaleAreas only implemented in 2D and 3D");
#endif

    for (int idim = 0; idim < 3; ++idim) {
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        if (idim == 0 || idim == 2) { continue; }
#endif
        for (amrex::MFIter mfi(*face_areas[0]); mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.tilebox(face_areas[idim]->ixType().toIntVect(),
                                                face_areas[idim]->nGrowVect() );
            amrex::Real const full_area = cell_size[(idim+1)%3]*cell_size[(idim+2)%3];
            auto const &face_areas_dim = face_areas[idim]->array(mfi);

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                face_areas_dim(i, j, k) *= full_area;
            });

        }
    }
}

#endif
