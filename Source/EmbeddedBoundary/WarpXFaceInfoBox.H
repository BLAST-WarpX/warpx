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

struct FaceInfoBox {
    enum Neighbours : uint8_t {n, s, e, w, nw, ne, sw, se};

    amrex::Gpu::DeviceVector<Neighbours> neighbor_faces;
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
    static void addConnectedNeighbor(int i, int j, int ind, Neighbours* neighbor_face_ptr){

        if(i == -1 && j == -1){
            *(neighbor_face_ptr + ind) = nw;
        }else if(i == -1 && j == 0){
            *(neighbor_face_ptr + ind) = w;
        }else if(i == -1 && j == 1){
            *(neighbor_face_ptr + ind) = sw;
        }else if(i == 0 && j == -1){
            *(neighbor_face_ptr + ind) = n;
        }else if(i == 0 && j == 1){
            *(neighbor_face_ptr + ind) = s;
        }else if(i == 1 && j == -1){
            *(neighbor_face_ptr + ind) = ne;
        }else if(i == 1 && j == 0){
            *(neighbor_face_ptr + ind) = e;
        }else if(i == 1 && j == 1){
            *(neighbor_face_ptr + ind) = se;
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
