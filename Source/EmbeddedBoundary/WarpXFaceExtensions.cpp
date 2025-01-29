/* Copyright 2021 Lorenzo Giacomel
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "WarpXFaceInfoBox.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/warn_manager/WarnManager.H>
#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Scan.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>

using namespace ablastr::fields;
using warpx::fields::FieldType;

/**
* \brief Get the value of arr in the neighbor (i_n, j_n) on the plane with normal 'dim'.
*
*        I.E. If dim==0 it return arr(i, j + i_n, k + j_n),
*             if dim==1 it return arr(i + i_n, j, k + j_n),
*             if dim==2 it return arr(i + i_n, j + j_n, k)
*
* \param[in] arr data To be accessed
* \param[in] i, j, k the indices of the "center" cell
* \param[in] i_n the offset of the neighbor in the first direction
* \param[in] j_n the offset of the neighbor in the second direction
* \param[in] dim normal direction to the plane in consideration (0 for x, 1 for y, 2 for z)
*/
template <class T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
constexpr
T
GetNeigh(const amrex::Array4<T>& arr,
         const int i, const int j, const int k,
         const int i_n, const int j_n, const int dim){

    if(dim == 0){
        return arr(i, j + i_n, k + j_n);
    }
#ifdef WARPX_DIM_XZ
    else if(dim == 1 || (dim == 2)){
        return arr(i + i_n, j + j_n, k);
    }
#elif defined(WARPX_DIM_3D)
    else if(dim == 1){
        return arr(i + i_n, j, k + j_n);
    }
    else if(dim == 2){
        return arr(i + i_n, j + j_n, k);
    }
#else
    else if(dim == 1){
        amrex::Abort("GetNeigh: Only implemented in 2D3V and 3D3V");
    }
    else if(dim == 2){
        return arr(i + i_n, j + j_n, k);
    }
#endif

    amrex::Abort("GetNeigh: dim must be 0, 1 or 2");

    return -1;
}


/**
* \brief Set the value of arr in the neighbor (i_n, j_n) on the plane with normal 'dim'.
*
*        I.E. If dim==0 it return arr(i, j + i_n, k + j_n),
*             if dim==1 it return arr(i + i_n, j, k + j_n),
*             if dim==2 it return arr(i + i_n, j + j_n, k)
*
* \param[in] arr data to be modified
* \param[in] val the value to be set
* \param[in] i, j, k the indices of the "center" cell
* \param[in] i_n the offset of the neighbor in the first direction
* \param[in] j_n the offset of the neighbor in the second direction
* \param[in] dim normal direction to the plane in consideration (0 for x, 1 for y, 2 for z)
*/
template <class T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
constexpr
void
SetNeigh(const amrex::Array4<T>& arr, const T val,
         const int i, const int j, const int k,
         const int i_n, const int j_n, const int dim){

    if(dim == 0){
        arr(i, j + i_n, k + j_n) = val;
        return;
    }
#ifdef WARPX_DIM_XZ
    else if(dim == 1 || (dim == 2)){
        arr(i + i_n, j + j_n, k) = val;
        return;
    }
#elif defined(WARPX_DIM_3D)
    else if(dim == 1){
        arr(i + i_n, j, k + j_n) = val;
        return;
    }
    else if(dim == 2){
        arr(i + i_n, j + j_n, k) = val;
        return;
    }
#else
    else if(dim == 1){
        amrex::Abort("SetNeigh: Only implemented in 2D3V and 3D3V");
    }
    else if(dim == 2){
        arr(i + i_n, j + j_n, k) = val;
        return;
    }
#endif

    amrex::Abort("SetNeigh: dim must be 0, 1 or 2");
}


/**
* \brief Compute the minimal area for stability for the face i, j, k with normal 'dim'.
*
* \param[in] i, j, k the indices of the cell
* \param[in] lx, ly, lz arrays containing the edge lengths
* \param[in] dx, dy, dz the mesh with in each direction
* \param[in] dim normal direction to the plane in consideration (0 for x, 1 for y, 2 for z)
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real
ComputeSStab(const int i, const int j, const int k,
             const amrex::Array4<const amrex::Real> lx,
             const amrex::Array4<const amrex::Real> ly,
             const amrex::Array4<const amrex::Real> lz,
             const amrex::Real dx, const amrex::Real dy, const amrex::Real dz,
             const int dim){

    using namespace amrex::literals;

    if(dim == 0) {
        return 0.5_rt * std::max({ly(i, j, k) * dz, ly(i, j, k + 1) * dz,
                                  lz(i, j, k) * dy, lz(i, j + 1, k) * dy});
    }else if(dim == 1){
#ifdef WARPX_DIM_XZ
        return 0.5_rt * std::max({lx(i, j, k) * dz, lx(i, j + 1, k) * dz,
                                  lz(i, j, k) * dx, lz(i + 1, j, k) * dx});
#elif defined(WARPX_DIM_3D)
        return 0.5_rt * std::max({lx(i, j, k) * dz, lx(i, j, k + 1) * dz,
                                  lz(i, j, k) * dx, lz(i + 1, j, k) * dx});
#else
        amrex::Abort("ComputeSStab: Only implemented in 2D3V and 3D3V");
#endif
    }else if(dim == 2){
        return 0.5_rt * std::max({lx(i, j, k) * dy, lx(i, j + 1, k) * dy,
                                  ly(i, j, k) * dx, ly(i + 1, j, k) * dx});
    }

    amrex::Abort("ComputeSStab: dim must be 0, 1 or 2");

    return -1;
}


amrex::Array1D<int, 0, 2>
WarpX::CountExtFaces () {
    amrex::Array1D<int, 0, 2> sums{0, 0, 0};
#ifdef AMREX_USE_EB
    if (EB::enabled()) {
#ifndef WARPX_DIM_RZ

#ifdef WARPX_DIM_XZ
        // In 2D we change the extrema of the for loop so that we only have the case idim=1
        for(int idim = 1; idim < AMREX_SPACEDIM; ++idim) {
#elif defined(WARPX_DIM_3D)
        for(int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
#else
            WARPX_ABORT_WITH_MESSAGE(
                "CountExtFaces: Only implemented in 2D3V and 3D3V");
#endif
            amrex::ReduceOps<amrex::ReduceOpSum> reduce_ops;
            amrex::ReduceData<int> reduce_data(reduce_ops);
            for (amrex::MFIter mfi(*m_flag_ext_face[maxLevel()][idim]); mfi.isValid(); ++mfi) {
                amrex::Box const &box = mfi.validbox();
                auto const &flag_ext_face = m_flag_ext_face[maxLevel()][idim]->array(mfi);
                reduce_ops.eval(box, reduce_data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int> {
                        return flag_ext_face(i, j, k);
                    });
            }

            auto r = reduce_data.value();
            sums(idim) = amrex::get<0>(r);
        }

        amrex::ParallelDescriptor::ReduceIntSum(&(sums(0)), AMREX_SPACEDIM);
#endif
    }
#endif
    return sums;
}

void
WarpX::ComputeFaceExtensions ()
{
    if (!EB::enabled()) {
        throw std::runtime_error("ComputeFaceExtensions only works when EBs are enabled at runtime");
    }
#ifdef AMREX_USE_EB
    amrex::Array1D<int, 0, 2> N_ext_faces = CountExtFaces();
    ablastr::warn_manager::WMRecordWarning("Embedded Boundary",
            "Faces to be extended in x:\t" + std::to_string(N_ext_faces(0)) + "\n"
            +"Faces to be extended in y:\t" + std::to_string(N_ext_faces(1)) + "\n"
            +"Faces to be extended in z:\t" + std::to_string(N_ext_faces(2)),
            ablastr::warn_manager::WarnPriority::low
    );

    InitBorrowing();
    ComputeOneWayExtensions();

    amrex::Array1D<int, 0, 2> N_ext_faces_after_one_way = CountExtFaces();
    ablastr::warn_manager::WMRecordWarning("Embedded Boundary",
            "Faces to be extended after one way extension in x:\t"
            + std::to_string(N_ext_faces_after_one_way(0)) + "\n"
            +"Faces to be extended after one way extension in y:\t"
            + std::to_string(N_ext_faces_after_one_way(1)) + "\n"
            +"Faces to be extended after one way extension in z:\t"
            + std::to_string(N_ext_faces_after_one_way(2)),
            ablastr::warn_manager::WarnPriority::low
    );

    ComputeEightWaysExtensions();
    ShrinkBorrowing();

    amrex::Array1D<int, 0, 2> N_ext_faces_after_eight_ways = CountExtFaces();
    ablastr::warn_manager::WMRecordWarning("Embedded Boundary",
            "Faces to be extended after eight ways extension in x:\t"
            + std::to_string(N_ext_faces_after_eight_ways(0)) + "\n"
            +"Faces to be extended after eight ways extension in y:\t"
            + std::to_string(N_ext_faces_after_eight_ways(1)) + "\n"
            +"Faces to be extended after eight ways extension in z:\t"
            + std::to_string(N_ext_faces_after_eight_ways(2)),
            ablastr::warn_manager::WarnPriority::low
    );

    bool using_bck = false;

    // If any cell could not be extended we use the BCK method to stabilize them
#if !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_RZ)
    if (N_ext_faces_after_eight_ways(0) > 0) {
        ApplyBCKCorrection(0);
        using_bck = true;
    }
#endif

    if (N_ext_faces_after_eight_ways(1) > 0) {
        ApplyBCKCorrection(1);
        using_bck = true;
    }

#if !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_RZ)
    if (N_ext_faces_after_eight_ways(2) > 0) {
        ApplyBCKCorrection(2);
        using_bck = true;
    }
#endif

    if(using_bck) {
        ablastr::warn_manager::WMRecordWarning("Embedded Boundary",
                             "Some faces could not be stabilized with the ECT and the BCK correction was used.\n"
                             "The BCK correction will be used for:\n"
                             "-" + std::to_string(N_ext_faces_after_eight_ways(0)) + " x-faces\n"
                             + "-" + std::to_string(N_ext_faces_after_eight_ways(1)) + " y-faces\n"
                             + "-" + std::to_string(N_ext_faces_after_eight_ways(2)) + " z-faces\n",
                            ablastr::warn_manager::WarnPriority::low
        );
    }
#endif
}


void
WarpX::InitBorrowing() {
    int idim = 0;
    for (amrex::MFIter mfi(*m_fields.get(FieldType::Bfield_fp, Direction{idim}, maxLevel())); mfi.isValid(); ++mfi) {
        amrex::Box const &box = mfi.validbox();
        auto &borrowing_x = (*m_borrowing[maxLevel()][idim])[mfi];
        borrowing_x.inds_pointer.resize(box);
        borrowing_x.size.resize(box);
        borrowing_x.size.setVal<amrex::RunOn::Device>(0);
        const amrex::Long ncells = box.numPts();
        // inds, neigh_faces and area are extended to their largest possible size here, but they are
        // resized to a much smaller size later on, based on the actual number of neighboring
        // intruded faces for each unstable face.
        borrowing_x.inds.resize(8*ncells);
        borrowing_x.neigh_faces.resize(8*ncells);
        borrowing_x.area.resize(8*ncells);
    }

    idim = 1;
    for (amrex::MFIter mfi(*m_fields.get(FieldType::Bfield_fp, Direction{idim}, maxLevel())); mfi.isValid(); ++mfi) {
        amrex::Box const &box = mfi.validbox();
        auto &borrowing_y = (*m_borrowing[maxLevel()][idim])[mfi];
        borrowing_y.inds_pointer.resize(box);
        borrowing_y.size.resize(box);
        borrowing_y.size.setVal<amrex::RunOn::Device>(0);
        const amrex::Long ncells = box.numPts();
        borrowing_y.inds.resize(8*ncells);
        borrowing_y.neigh_faces.resize(8*ncells);
        borrowing_y.area.resize(8*ncells);
    }

    idim = 2;
    for (amrex::MFIter mfi(*m_fields.get(FieldType::Bfield_fp, Direction{idim}, maxLevel())); mfi.isValid(); ++mfi) {
        amrex::Box const &box = mfi.validbox();
        auto &borrowing_z = (*m_borrowing[maxLevel()][idim])[mfi];
        borrowing_z.inds_pointer.resize(box);
        borrowing_z.size.resize(box);
        borrowing_z.size.setVal<amrex::RunOn::Device>(0);
        const amrex::Long ncells = box.numPts();
        borrowing_z.inds.resize(8*ncells);
        borrowing_z.neigh_faces.resize(8*ncells);
        borrowing_z.area.resize(8*ncells);
    }
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int
ComputeNBorrowOneFaceExtension(const amrex::Dim3 cell, const amrex::Real S_ext,
                               const amrex::Array4<amrex::Real>& S_red,
                               const amrex::Array4<int>& flag_info_face,
                               const amrex::Array4<int>& flag_ext_face, const int idim) {
    const int i = cell.x;
    const int j = cell.y;
    const int k = cell.z;
    int n_borrow = 0;
    bool stop = false;
    for (int i_n = -1; i_n < 2; i_n++) {
        for (int j_n = -1; j_n < 2; j_n++) {
            //This if makes sure that we don't visit the "diagonal neighbours"
            if ((i_n != j_n) && (i_n != -j_n)) {
                // Here a face is available if it doesn't need to be extended itself and if its
                // area exceeds Sz_ext. Here we need to take into account if the intruded face
                // has given away already some area, so we use Sz_red rather than Sz.
                // If no face is available we don't do anything and we will need to use the
                // multi-face extensions.
                if (GetNeigh(S_red, i, j, k, i_n, j_n, idim) > S_ext
                    && (GetNeigh(flag_info_face, i, j, k, i_n, j_n, idim) == 1
                    || GetNeigh(flag_info_face, i, j, k, i_n, j_n, idim) == 2)
                    && flag_ext_face(i, j, k) && ! stop) {
                    n_borrow += 1;
                    stop = true;
                }
            }
        }
    }

    return n_borrow;
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int
ComputeNBorrowEightFacesExtension(const amrex::Dim3 cell, const amrex::Real S_ext,
                                  const amrex::Array4<amrex::Real>& S_red,
                                  const amrex::Array4<amrex::Real>& S,
                                  const amrex::Array4<int>& flag_info_face, const int idim) {
    const int i = cell.x;
    const int j = cell.y;
    const int k = cell.z;
    int n_borrow = 0;
    amrex::Array2D<int, 0, 2, 0, 2> local_avail{};

    for(int i_loc = 0; i_loc <= 2; i_loc++){
        for(int j_loc = 0; j_loc <= 2; j_loc++){
            const int flag = GetNeigh(flag_info_face, i, j, k, i_loc - 1, j_loc - 1, idim);
            local_avail(i_loc, j_loc) = flag == 1 || flag == 2;
        }
    }

    amrex::Real denom = local_avail(0, 1) * GetNeigh(S, i, j, k, -1, 0, idim) +
                        local_avail(2, 1) * GetNeigh(S, i, j, k, 1, 0, idim) +
                        local_avail(1, 0) * GetNeigh(S, i, j, k, 0, -1, idim) +
                        local_avail(1, 2) * GetNeigh(S, i, j, k, 0, 1, idim) +
                        local_avail(0, 0) * GetNeigh(S, i, j, k, -1, -1, idim) +
                        local_avail(2, 0) * GetNeigh(S, i, j, k, 1, -1, idim) +
                        local_avail(0, 2) * GetNeigh(S, i, j, k, -1, 1, idim) +
                        local_avail(2, 2) * GetNeigh(S, i, j, k, 1, 1, idim);

    bool neg_face = true;

    while(denom >= S_ext && neg_face && denom > 0){
        neg_face = false;
        for (int i_n = -1; i_n < 2; i_n++) {
            for (int j_n = -1; j_n < 2; j_n++) {
                if(local_avail(i_n + 1, j_n + 1)){
                    const amrex::Real patch = S_ext * GetNeigh(S, i, j, k, i_n, j_n, idim) / denom;
                    if(GetNeigh(S_red, i, j, k, i_n, j_n, idim) - patch <= 0) {
                        neg_face = true;
                        local_avail(i_n + 1, j_n + 1) = false;
                    }
                }
            }
        }

        denom = local_avail(0, 1) * GetNeigh(S, i, j, k, -1, 0, idim) +
                local_avail(2, 1) * GetNeigh(S, i, j, k, 1, 0, idim) +
                local_avail(1, 0) * GetNeigh(S, i, j, k, 0, -1, idim) +
                local_avail(1, 2) * GetNeigh(S, i, j, k, 0, 1, idim) +
                local_avail(0, 0) * GetNeigh(S, i, j, k, -1, -1, idim) +
                local_avail(2, 0) * GetNeigh(S, i, j, k, 1, -1, idim) +
                local_avail(0, 2) * GetNeigh(S, i, j, k, -1, 1, idim) +
                local_avail(2, 2) * GetNeigh(S, i, j, k, 1, 1, idim);
    }

    // We count the number of entries in local_avail which are still True, this is the number of
    // neighboring faces which are intruded
    for(int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
            n_borrow += local_avail(ii, jj);
        }
    }

    return n_borrow;
}


void
WarpX::ComputeOneWayExtensions ()
{
    if (!EB::enabled()) {
        throw std::runtime_error("ComputeOneWayExtensions only works when EBs are enabled at runtime");
    }
#ifdef AMREX_USE_EB
#ifndef WARPX_DIM_RZ
    auto const eb_fact = fieldEBFactory(maxLevel());

    auto const &cell_size = CellSize(maxLevel());

    const amrex::Real dx = cell_size[0];
    const amrex::Real dy = cell_size[1];
    const amrex::Real dz = cell_size[2];

    // Do the extensions
#ifdef WARPX_DIM_XZ
    // In 2D we change the extrema of the for loop so that we only have the case idim=1
    for(int idim = 1; idim < AMREX_SPACEDIM; ++idim) {
#elif defined(WARPX_DIM_3D)
        for(int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
#else
        WARPX_ABORT_WITH_MESSAGE(
            "ComputeOneWayExtensions: Only implemented in 2D3V and 3D3V");
#endif
        for (amrex::MFIter mfi(*m_fields.get(FieldType::Bfield_fp, Direction{idim}, maxLevel())); mfi.isValid(); ++mfi) {

            amrex::Box const &box = mfi.validbox();
            auto const &S = m_fields.get(FieldType::face_areas, Direction{idim}, maxLevel())->array(mfi);
            auto const &flag_ext_face = m_flag_ext_face[maxLevel()][idim]->array(mfi);
            auto const &flag_info_face = m_flag_info_face[maxLevel()][idim]->array(mfi);
            auto &borrowing = (*m_borrowing[maxLevel()][idim])[mfi];
            auto const &borrowing_inds_pointer = borrowing.inds_pointer.array();
            auto const &borrowing_size = borrowing.size.array();
            amrex::Long const ncells = box.numPts();
            int* borrowing_inds = borrowing.inds.data();
            FaceInfoBox::Neighbours* borrowing_neigh_faces = borrowing.neigh_faces.data();
            amrex::Real* borrowing_area = borrowing.area.data();
            int& vecs_size = borrowing.vecs_size;

            auto const &S_mod = m_fields.get(FieldType::area_mod, Direction{idim}, maxLevel())->array(mfi);

            const auto &lx = m_fields.get(FieldType::edge_lengths, Direction{0}, maxLevel())->array(mfi);
            const auto &ly = m_fields.get(FieldType::edge_lengths, Direction{1}, maxLevel())->array(mfi);
            const auto &lz = m_fields.get(FieldType::edge_lengths, Direction{2}, maxLevel())->array(mfi);

            vecs_size = amrex::Scan::PrefixSum<int>(ncells,
                                                    [=] AMREX_GPU_DEVICE (int icell) {
                const amrex::Dim3 cell = box.atOffset(icell).dim3();
                const int i = cell.x;
                const int j = cell.y;
                const int k = cell.z;
                // If the face doesn't need to be extended break the loop
                if (!flag_ext_face(i, j, k)) {
                    return 0;
                }

                const amrex::Real S_stab = ComputeSStab(i, j, k, lx, ly, lz, dx, dy, dz, idim);

                const amrex::Real S_ext = S_stab - S(i, j, k);
                const int n_borrow =
                    ComputeNBorrowOneFaceExtension(cell, S_ext, S_mod, flag_info_face,
                                                   flag_ext_face, idim);


              borrowing_size(i, j, k) = n_borrow;
                return n_borrow;
            },
                                                [=] AMREX_GPU_DEVICE (int icell, int ps){
                const amrex::Dim3 cell = box.atOffset(icell).dim3();
                const int i = cell.x;
                const int j = cell.y;
                const int k = cell.z;
                const int nborrow = borrowing_size(i, j, k);
                if (nborrow == 0) {
                    borrowing_inds_pointer(i, j, k) = nullptr;
                } else{
                    borrowing_inds_pointer(i, j, k) = borrowing_inds + ps;

                    const amrex::Real S_stab = ComputeSStab(i, j, k, lx, ly, lz, dx, dy, dz, idim);

                    const amrex::Real S_ext = S_stab - S(i, j, k);
                    for (int i_n = -1; i_n < 2; i_n++) {
                        for (int j_n = -1; j_n < 2; j_n++) {
                            //This if makes sure that we don't visit the "diagonal neighbours"
                            if (i_n != j_n && i_n != -j_n){
                                // Here a face is available if it doesn't need to be extended itself and if its
                                // area exceeds Sz_ext. Here we need to take into account if the intruded face
                                // has given away already some area, so we use Sz_red rather than Sz.
                                // If no face is available we don't do anything and we will need to use the
                                // multi-face extensions.
                                if (GetNeigh(S_mod, i, j, k, i_n, j_n, idim) > S_ext
                                    && (GetNeigh(flag_info_face, i, j, k, i_n, j_n, idim) == 1
                                         || GetNeigh(flag_info_face, i, j, k, i_n, j_n, idim) == 2)
                                    && flag_ext_face(i, j, k)) {

                                    SetNeigh(S_mod,
                                             GetNeigh(S_mod, i, j, k, i_n, j_n, idim) - S_ext,
                                             i, j, k, i_n, j_n, idim);

                                    // Insert the index of the face info
                                    borrowing_inds[ps] = ps;
                                    // Store the information about the intruded face in the dataset of the
                                    // faces which are borrowing area
                                    FaceInfoBox::addConnectedNeighbor(i_n, j_n, ps,
                                                                      borrowing_neigh_faces);
                                    borrowing_area[ps] = S_ext;

                                    SetNeigh(flag_info_face, 2, i, j, k, i_n, j_n, idim);
                                    // Add the area to the intruding face.
                                    S_mod(i, j, k) = S(i, j, k) + S_ext;
                                    flag_ext_face(i, j, k) = false;
                                }
                            }
                        }
                    }
                }
            }, amrex::Scan::Type::exclusive);
        }
    }

#endif
#endif
}


void
WarpX::ComputeEightWaysExtensions ()
{
    if (!EB::enabled()) {
        throw std::runtime_error("ComputeEightWaysExtensions only works when EBs are enabled at runtime");
    }
#ifdef AMREX_USE_EB
    using namespace amrex::literals;

#ifndef WARPX_DIM_RZ
    auto const &cell_size = CellSize(maxLevel());

    const amrex::Real dx = cell_size[0];
    const amrex::Real dy = cell_size[1];
    const amrex::Real dz = cell_size[2];

    // Do the extensions
#ifdef WARPX_DIM_XZ
    // In 2D we change the extrema of the for loop so that we only have the case idim=1
    for(int idim = 1; idim < AMREX_SPACEDIM; ++idim) {
#elif defined(WARPX_DIM_3D)
        for(int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
#else
        WARPX_ABORT_WITH_MESSAGE(
            "ComputeEightWaysExtensions: Only implemented in 2D3V and 3D3V");
#endif
        for (amrex::MFIter mfi(*m_fields.get(FieldType::Bfield_fp, Direction{idim}, maxLevel())); mfi.isValid(); ++mfi) {

            amrex::Box const &box = mfi.validbox();

            auto const &S = m_fields.get(FieldType::face_areas, Direction{idim}, maxLevel())->array(mfi);
            auto const &flag_ext_face = m_flag_ext_face[maxLevel()][idim]->array(mfi);
            auto const &flag_info_face = m_flag_info_face[maxLevel()][idim]->array(mfi);
            auto &borrowing = (*m_borrowing[maxLevel()][idim])[mfi];
            auto const &borrowing_inds_pointer = borrowing.inds_pointer.array();
            auto const &borrowing_size = borrowing.size.array();
            amrex::Long const ncells = box.numPts();
            int* borrowing_inds = borrowing.inds.data();
            FaceInfoBox::Neighbours* borrowing_neigh_faces = borrowing.neigh_faces.data();
            amrex::Real* borrowing_area = borrowing.area.data();
            int& vecs_size = borrowing.vecs_size;

            auto const &S_mod = m_fields.get(FieldType::area_mod, Direction{idim}, maxLevel())->array(mfi);

            const auto &lx = m_fields.get(FieldType::edge_lengths, Direction{0}, maxLevel())->array(mfi);
            const auto &ly = m_fields.get(FieldType::edge_lengths, Direction{1}, maxLevel())->array(mfi);
            const auto &lz = m_fields.get(FieldType::edge_lengths, Direction{2}, maxLevel())->array(mfi);

            vecs_size += amrex::Scan::PrefixSum<int>(ncells,
                                                     [=] AMREX_GPU_DEVICE (int icell){
                const amrex::Dim3 cell = box.atOffset(icell).dim3();
                const int i = cell.x;
                const int j = cell.y;
                const int k = cell.z;
                // If the face doesn't need to be extended break the loop
                if (!flag_ext_face(i, j, k)) {
                    return 0;
                }
                const amrex::Real S_stab = ComputeSStab(i, j, k, lx, ly, lz, dx, dy, dz, idim);

                const amrex::Real S_ext = S_stab - S(i, j, k);
                const int n_borrow = ComputeNBorrowEightFacesExtension(cell, S_ext, S_mod, S,
                                                                       flag_info_face, idim);

              borrowing_size(i, j, k) = n_borrow;
                return n_borrow;
            },
            [=] AMREX_GPU_DEVICE (int icell, int ps) {

                ps += vecs_size;

                const amrex::Dim3 cell = box.atOffset(icell).dim3();
                const int i = cell.x;
                const int j = cell.y;
                const int k = cell.z;

                if (!flag_ext_face(i, j, k)) {
                    return;
                }

                const int nborrow = borrowing_size(i, j, k);
                if (nborrow == 0) {
                    borrowing_inds_pointer(i, j, k) = nullptr;
                } else {
                    borrowing_inds_pointer(i, j, k) = borrowing_inds + ps;

                    S_mod(i, j, k) = S(i, j, k);
                    const amrex::Real S_stab = ComputeSStab(i, j, k, lx, ly, lz, dx, dy, dz, idim);

                    const amrex::Real S_ext = S_stab - S(i, j, k);
                    amrex::Array2D<amrex::Real, 0, 2, 0, 2> local_avail{};
                    for(int i_loc = 0; i_loc <= 2; i_loc++){
                        for(int j_loc = 0; j_loc <= 2; j_loc++){
                            auto const flag = GetNeigh(flag_info_face, i, j, k, i_loc - 1, j_loc - 1, idim);
                            local_avail(i_loc, j_loc) = flag == 1 || flag == 2;
                        }
                    }

                    amrex::Real denom = local_avail(0, 1) * GetNeigh(S, i, j, k, -1, 0, idim) +
                                        local_avail(2, 1) * GetNeigh(S, i, j, k, 1, 0, idim) +
                                        local_avail(1, 0) * GetNeigh(S, i, j, k, 0, -1, idim) +
                                        local_avail(1, 2) * GetNeigh(S, i, j, k, 0, 1, idim) +
                                        local_avail(0, 0) * GetNeigh(S, i, j, k, -1, -1, idim) +
                                        local_avail(2, 0) * GetNeigh(S, i, j, k, 1, -1, idim) +
                                        local_avail(0, 2) * GetNeigh(S, i, j, k, -1, 1, idim) +
                                        local_avail(2, 2) * GetNeigh(S, i, j, k, 1, 1, idim);

                    bool neg_face = true;

                    while(denom >= S_ext && neg_face && denom > 0){
                        neg_face = false;
                        for (int i_n = -1; i_n < 2; i_n++) {
                            for (int j_n = -1; j_n < 2; j_n++) {
                                if (local_avail(i_n + 1, j_n + 1) != 0_rt){
                                    const amrex::Real patch = S_ext * GetNeigh(S, i, j, k, i_n, j_n, idim) / denom;
                                    if(GetNeigh(S_mod, i, j, k, i_n, j_n, idim) - patch <= 0) {
                                        neg_face = true;
                                        local_avail(i_n + 1, j_n + 1) = false;
                                    }
                                }
                            }
                        }

                        denom = local_avail(0, 1) * GetNeigh(S, i, j, k, -1, 0, idim) +
                                local_avail(2, 1) * GetNeigh(S, i, j, k, 1, 0, idim) +
                                local_avail(1, 0) * GetNeigh(S, i, j, k, 0, -1, idim) +
                                local_avail(1, 2) * GetNeigh(S, i, j, k, 0, 1, idim) +
                                local_avail(0, 0) * GetNeigh(S, i, j, k, -1, -1, idim) +
                                local_avail(2, 0) * GetNeigh(S, i, j, k, 1, -1, idim) +
                                local_avail(0, 2) * GetNeigh(S, i, j, k, -1, 1, idim) +
                                local_avail(2, 2) * GetNeigh(S, i, j, k, 1, 1, idim);
                    }

                    if(denom >= S_ext){
                        S_mod(i, j, k) = S(i, j, k);
                        int count = 0;
                        for (int i_n = -1; i_n < 2; i_n++) {
                            for (int j_n = -1; j_n < 2; j_n++) {
                                if(local_avail(i_n + 1, j_n + 1) != 0_rt){
                                    const amrex::Real patch = S_ext * GetNeigh(S, i, j, k, i_n, j_n, idim) / denom;
                                    borrowing_inds[ps + count] = ps + count;
                                    FaceInfoBox::addConnectedNeighbor(i_n, j_n, ps + count,
                                                                      borrowing_neigh_faces);
                                    borrowing_area[ps + count] = patch;

                                    SetNeigh(flag_info_face, 2, i, j, k, i_n, j_n, idim);

                                    S_mod(i, j, k) += patch;
                                    SetNeigh(S_mod,
                                             GetNeigh(S_mod, i, j, k, i_n, j_n, idim) - patch,
                                             i, j, k, i_n, j_n, idim);
                                    count +=1;
                                }
                            }
                        }
                        flag_ext_face(i, j, k) = false;
                    }
                }
            }, amrex::Scan::Type::exclusive);
        }
    }
#endif
#endif
}

void
WarpX::ApplyBCKCorrection (const int idim)
{
    if (!EB::enabled()) {
        throw std::runtime_error("ApplyBCKCorrection only works when EBs are enabled at runtime");
    }
#if defined(AMREX_USE_EB) and !defined(WARPX_DIM_RZ)
    const std::array<amrex::Real,3> &cell_size = CellSize(maxLevel());

    const amrex::Real dx = cell_size[0];
    const amrex::Real dy = cell_size[1];
    const amrex::Real dz = cell_size[2];

    for (amrex::MFIter mfi(*m_fields.get(FieldType::Bfield_fp, Direction{idim}, maxLevel()), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const amrex::Box &box = mfi.tilebox();
        const amrex::Array4<int> &flag_ext_face = m_flag_ext_face[maxLevel()][idim]->array(mfi);
        const amrex::Array4<int> &flag_info_face = m_flag_info_face[maxLevel()][idim]->array(mfi);
        const amrex::Array4<amrex::Real> &S =  m_fields.get(FieldType::face_areas, Direction{idim}, maxLevel())->array(mfi);
        const amrex::Array4<amrex::Real> &lx = m_fields.get(FieldType::face_areas, Direction{0}, maxLevel())->array(mfi);;
        const amrex::Array4<amrex::Real> &ly = m_fields.get(FieldType::face_areas, Direction{1}, maxLevel())->array(mfi);;
        const amrex::Array4<amrex::Real> &lz = m_fields.get(FieldType::face_areas, Direction{2}, maxLevel())->array(mfi);;

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            if (flag_ext_face(i, j, k)) {
                // Modify the area according to the BCK algorithm
                S(i, j, k) = ComputeSStab(i, j, k, lx, ly, lz, dx, dy, dz, idim);
                // Update the face info so that the solver doesn't think that this face is being extended
                flag_info_face(i, j, k) = -1;
            }
        });
    }
#else
    amrex::ignore_unused(idim);
#endif
}

void
WarpX::ShrinkBorrowing ()
{
    for(int idim = 0; idim < AMREX_SPACEDIM; idim++) {
        for (amrex::MFIter mfi(*m_fields.get(FieldType::Bfield_fp, Direction{idim}, maxLevel())); mfi.isValid(); ++mfi) {
            auto &borrowing = (*m_borrowing[maxLevel()][idim])[mfi];
            borrowing.inds.resize(borrowing.vecs_size);
            borrowing.neigh_faces.resize(borrowing.vecs_size);
            borrowing.area.resize(borrowing.vecs_size);
        }
    }
}
