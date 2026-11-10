/* Copyright 2021 Lorenzo Giacomel
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_SOURCE_EMBEDDEDBOUNDARY_WARPXFACEINFOBOX_H
#define WARPX_SOURCE_EMBEDDEDBOUNDARY_WARPXFACEINFOBOX_H

#include "WarpXFaceInfoBox_fwd.H"

#include <AMReX_Gpu.H>
#include <AMReX_BaseFab.H>

#include <bitset>

/**
* \brief Categories of mesh faces used by the ECT solver, stored in the `flag_info_face`
* iMultiFab (see WarpX::m_flag_info_face). They are initialized in WarpX::MarkExtensionCells
* and then updated in WarpX::ComputeFaceExtensions.
*
* This is deliberately an unscoped enumeration with a fixed underlying type: the values are
* read from and written to an `amrex::iMultiFab`, so they are compared against `int` and a
* scoped enumeration would require a cast at every use site.
*/
namespace FaceInfo
{
    enum Flag : int {
        //! The face is too small to be stable and could be extended neither with the one-way
        //! nor with the eight-ways extension: it is stabilized with the BCK correction instead
        bck_stabilized = -1,
        //! The face is too small to be stable and is extended, i.e. it borrows area from its
        //! neighbors. This is also the value used before the extensions are computed, to mark
        //! the faces that need to be extended.
        extended = 0,
        //! The face is large enough to be stable and can lend area to its neighbors
        available = 1,
        //! The face is large enough to be stable and has lent area to at least one neighbor
        intruded = 2
    };
}

struct FaceInfoBox {
    enum Neighbours : uint8_t {n, s, e, w, nw, ne, sw, se};

    amrex::Gpu::DeviceVector<Neighbours> neigh_faces;
    amrex::Gpu::DeviceVector<amrex::Real> area;
    // this vector stores the indices to access in the previous vectors
    amrex::Gpu::DeviceVector<int> inds;
    //number of entries of inds which correspond to this element
    amrex::BaseFab<int> size;
    // Each entry points to an element inside the inds vector
    amrex::BaseFab<int*> inds_pointer;

    int vecs_size;


    /**
    * \brief add the neighbor i, j to the list of intruded neighbors.
    */
    AMREX_GPU_HOST_DEVICE
    static void addConnectedNeighbor(int i, int j, int ind, Neighbours* neigh_face_ptr){

        if(i == -1 && j == -1){
            *(neigh_face_ptr + ind) = nw;
        }else if(i == -1 && j == 0){
            *(neigh_face_ptr + ind) = w;
        }else if(i == -1 && j == 1){
            *(neigh_face_ptr + ind) = sw;
        }else if(i == 0 && j == -1){
            *(neigh_face_ptr + ind) = n;
        }else if(i == 0 && j == 1){
            *(neigh_face_ptr + ind) = s;
        }else if(i == 1 && j == -1){
            *(neigh_face_ptr + ind) = ne;
        }else if(i == 1 && j == 0){
            *(neigh_face_ptr + ind) = e;
        }else if(i == 1 && j == 1){
            *(neigh_face_ptr + ind) = se;
        }
    }

    /**
    * \brief writes into i_face and j_face the intruded neighbors indices;
    */
    AMREX_GPU_HOST_DEVICE
    static amrex::Array1D<int, 0, 1> uint8_to_inds(Neighbours mask){
        amrex::Array1D<int, 0, 1> res;

        if(mask == Neighbours::nw){
            res(0) = -1;
            res(1) = -1;
        }else if(mask == Neighbours::w){
            res(0) = -1;
            res(1) = 0;
        }else if(mask == Neighbours::sw){
            res(0) = -1;
            res(1) = 1;
        }else if(mask == Neighbours::n){
            res(0) = 0;
            res(1) = -1;
        }else if(mask == Neighbours::s){
            res(0) = 0;
            res(1) = 1;
        }else if(mask == Neighbours::ne){
            res(0) = 1;
            res(1) = -1;
        }else if(mask == Neighbours::e){
            res(0) = 1;
            res(1) = 0;
        }else if(mask == Neighbours::se){
            res(0) = 1;
            res(1) = 1;
        }

        return res;
    }
};

#endif //WARPX_SOURCE_EMBEDDEDBOUNDARY_WARPXFACEINFOBOX_H
