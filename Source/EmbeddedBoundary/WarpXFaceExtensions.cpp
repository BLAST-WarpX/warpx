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

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/utils/Communication.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_Functional.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_Scan.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>

using namespace ablastr::fields;

namespace
{

#ifdef AMREX_USE_EB
    /**
    * \brief Auxiliary function to count the amount of faces which still need to be extended
    */
    amrex::Array1D<int, 0, 2>
    CountExtFaces (
        [[maybe_unused]] amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>, 3 > >& flag_ext_face,
        [[maybe_unused]] const int max_level)
    {
        amrex::Array1D<int, 0, 2> sums{0, 0, 0};

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
                for (amrex::MFIter mfi(*flag_ext_face[max_level][idim]); mfi.isValid(); ++mfi) {
                    amrex::Box const &box = mfi.validbox();
                    auto const &r_flag_ext_face = flag_ext_face[max_level][idim]->array(mfi);
                    reduce_ops.eval(box, reduce_data,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int> {
                            return r_flag_ext_face(i, j, k);
                        });
                }

                auto r = reduce_data.value();
                sums(idim) = amrex::get<0>(r);
            }

            amrex::ParallelDescriptor::ReduceIntSum(&(sums(0)), AMREX_SPACEDIM);
#endif
        }
        return sums;
    }

#endif


    /**
    * \brief Compute the minimal area for stability for the face i, j, k with normal 'dim'.
    *
    * \tparam dim normal direction to the plane in consideration (0 for x, 1 for y, 2 for z)
    *
    * \param[in] i, j, k the indices of the cell
    * \param[in] lx, ly, lz arrays containing the edge lengths
    * \param[in] dx, dy, dz the mesh with in each direction
    */
    template <int dim>
    [[nodiscard]]
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real
    ComputeSStab (const int i, const int j, const int k,
                  const amrex::Array4<const amrex::Real> lx,
                  const amrex::Array4<const amrex::Real> ly,
                  const amrex::Array4<const amrex::Real> lz,
                  const amrex::Real dx, const amrex::Real dy, const amrex::Real dz)
    {

        using namespace amrex::literals;

        if constexpr (dim == 0) {
            return 0.5_rt * std::max({ly(i, j, k) * dz, ly(i, j, k + 1) * dz,
                                      lz(i, j, k) * dy, lz(i, j + 1, k) * dy});
        }
        else if constexpr (dim == 1)
        {
#ifdef WARPX_DIM_XZ
            return 0.5_rt * std::max({lx(i, j, k) * dz, lx(i, j + 1, k) * dz,
                                      lz(i, j, k) * dx, lz(i + 1, j, k) * dx});
#elif defined(WARPX_DIM_3D)
            return 0.5_rt * std::max({lx(i, j, k) * dz, lx(i, j, k + 1) * dz,
                                      lz(i, j, k) * dx, lz(i + 1, j, k) * dx});
#else
            amrex::Abort("ComputeSStab: Only implemented in 2D3V and 3D3V");
#endif
        }
        else if constexpr(dim == 2){
            return 0.5_rt * std::max({lx(i, j, k) * dy, lx(i, j + 1, k) * dy,
                                      ly(i, j, k) * dx, ly(i + 1, j, k) * dx});
        }

        amrex::Abort("ComputeSStab: dim must be 0, 1 or 2");

        return -1;
    }


    /**
    * \brief Compute the minimal area for stability for the face i, j, k with normal 'dim'.
    * (wrapper to allow using ComputeSStab as a non-templated function, by passing 'dim' as an argument)
    *
    * \param[in] i, j, k the indices of the cell
    * \param[in] lx, ly, lz arrays containing the edge lengths
    * \param[in] dx, dy, dz the mesh with in each direction
    * \param[in] dim normal direction to the plane in consideration (0 for x, 1 for y, 2 for z)
    */
   [[nodiscard]] [[maybe_unused]]
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real
    ComputeSStab (const int i, const int j, const int k,
                  const amrex::Array4<const amrex::Real> lx,
                  const amrex::Array4<const amrex::Real> ly,
                  const amrex::Array4<const amrex::Real> lz,
                  const amrex::Real dx, const amrex::Real dy, const amrex::Real dz,
                  const int dim)
    {
        if (dim == 0) {
            return ::ComputeSStab<0>(i,j,k,lx,ly,lz,dx,dy,dz);
        }
        else if (dim == 1) {
            return ::ComputeSStab<1>(i,j,k,lx,ly,lz,dx,dy,dz);
        }
        else if (dim == 2) {
            return ::ComputeSStab<2>(i,j,k,lx,ly,lz,dx,dy,dz);
        }
        return -1;
    }


    /**
    * \brief Whenever an unstable cell cannot be extended we increase its area to be the minimal for stability.
    *        This is the method Benkler-Chavannes-Kuster method and it is less accurate than the regular ECT but it
    *        still works better than staircasing. (see https://ieeexplore.ieee.org/document/1638381)
    *
    * @tparam      idim Integer indicating the dimension (x->0, y->1, z->2) for which the BCK correction is done
    *
    * @param[in]      cell_size_max_lev The cell size at the maximum refinement level
    * @param[in, out] all_fields The field manager
    * @param[in]      flag_ext_face The extension flag used by the ECT solver
    * @param[out]     flag_info_face The info flag used by the ECT solver
    */
    template <int idim>
    void ApplyBCKCorrection (
        const std::array<amrex::Real,3> &cell_size_max_lev,
        ablastr::fields::MultiFabRegister& all_fields,
        const int max_level,
        const amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>, 3 > >& flag_ext_face,
        const amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>, 3 > >& flag_info_face)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(EB::enabled(),
            "ApplyBCKCorrection only works when EBs are enabled at runtime");

#if defined(AMREX_USE_EB) and !defined(WARPX_DIM_RZ)

        using warpx::fields::FieldType;

        const amrex::Real dx = cell_size_max_lev[0];
        const amrex::Real dy = cell_size_max_lev[1];
        const amrex::Real dz = cell_size_max_lev[2];

        for (amrex::MFIter mfi(*all_fields.get(FieldType::Bfield_fp, Direction{idim}, max_level), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {

            const amrex::Box &box = mfi.tilebox();
            const amrex::Array4<int> &flag_ext_face_max_lev_idim = flag_ext_face[max_level][idim]->array(mfi);
            const amrex::Array4<int> &flag_info_face_max_lev_idim = flag_info_face[max_level][idim]->array(mfi);
            const amrex::Array4<amrex::Real> &S =  all_fields.get(FieldType::face_areas, Direction{idim}, max_level)->array(mfi);
            const amrex::Array4<amrex::Real> &lx = all_fields.get(FieldType::face_areas, Direction{0}, max_level)->array(mfi);
            const amrex::Array4<amrex::Real> &ly = all_fields.get(FieldType::face_areas, Direction{1}, max_level)->array(mfi);
            const amrex::Array4<amrex::Real> &lz = all_fields.get(FieldType::face_areas, Direction{2}, max_level)->array(mfi);

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (flag_ext_face_max_lev_idim(i, j, k)) {
                    // Modify the area according to the BCK algorithm
                    S(i, j, k) = ::ComputeSStab<idim>(i, j, k, lx, ly, lz, dx, dy, dz);
                    // Update the face info so that the solver doesn't think that this face is being extended
                    flag_info_face_max_lev_idim(i, j, k) = -1;
                }
            });
        }
#else
        amrex::ignore_unused(idim, cell_size_max_lev, all_fields, max_level, flag_ext_face, flag_info_face);
#endif
    }

#ifdef AMREX_USE_EB
    /**
    * \brief Initialize the memory for the FaceInfoBoxes
    */
    void init_borrowing(
        std::array< std::unique_ptr<amrex::LayoutData<FaceInfoBox> >, 3 > & borrowing,
        ablastr::fields::VectorField Bfield)
    {
        for (int idim = 0; idim < 3; ++idim){
            for (amrex::MFIter mfi(*Bfield[idim]); mfi.isValid(); ++mfi){
                amrex::Box const &box = mfi.validbox();
                auto& borrowing_dir = (*borrowing[idim])[mfi];
                borrowing_dir.inds_pointer.resize(box);
                borrowing_dir.size.resize(box);
                borrowing_dir.size.setVal<amrex::RunOn::Device>(0);
                const amrex::Long ncells = box.numPts();
                // inds, neighbor_faces and area are extended to their largest possible size here, but they are
                // resized to a much smaller size later on, based on the actual number of neighboring
                // intruded faces for each unstable face.
                borrowing_dir.inds.resize(8*ncells);
                borrowing_dir.neighbor_faces.resize(8*ncells);
                borrowing_dir.area.resize(8*ncells);
            }
        }
    }

    /**
    * \brief Shrink the vectors in the FaceInfoBoxes
    */
    void shrink_borrowing(
        std::array< std::unique_ptr<amrex::LayoutData<FaceInfoBox> >, 3 > & borrowing,
        ablastr::fields::VectorField Bfield)
    {
        using ablastr::fields::Direction;

        for(int idim = 0; idim < AMREX_SPACEDIM; idim++) {
            for (amrex::MFIter mfi(*Bfield[idim]); mfi.isValid(); ++mfi){
                auto& borrowing_dir = (*borrowing[idim])[mfi];
                borrowing_dir.inds.resize(borrowing_dir.vecs_size);
                borrowing_dir.neighbor_faces.resize(borrowing_dir.vecs_size);
                borrowing_dir.area.resize(borrowing_dir.vecs_size);
            }
        }
    }

#endif


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
    GetNeighbor(const amrex::Array4<T>& arr,
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
            amrex::Abort("GetNeighbor: Only implemented in 2D3V and 3D3V");
        }
        else if(dim == 2){
            return arr(i + i_n, j + j_n, k);
        }
    #endif

        amrex::Abort("GetNeighbor: dim must be 0, 1 or 2");

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
    SetNeighbor(const amrex::Array4<T>& arr, const T val,
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
            amrex::Abort("SetNeighbor: Only implemented in 2D3V and 3D3V");
        }
        else if(dim == 2){
            arr(i + i_n, j + j_n, k) = val;
            return;
        }
    #endif

        amrex::Abort("SetNeighbor: dim must be 0, 1 or 2");
    }


    /**
    * \brief Get the address of the value of arr in the neighbor (i_n, j_n) on
    * the plane with normal 'dim' (same indexing convention as GetNeighbor), for
    * atomic updates of the neighbor's value.
    *
    * \param[in] arr data to be accessed
    * \param[in] i, j, k the indices of the "center" cell
    * \param[in] i_n the offset of the neighbor in the first direction
    * \param[in] j_n the offset of the neighbor in the second direction
    * \param[in] dim normal direction to the plane in consideration (0 for x, 1 for y, 2 for z)
    */
    template <class T>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    constexpr
    T*
    GetNeighborPtr(const amrex::Array4<T>& arr,
            const int i, const int j, const int k,
            const int i_n, const int j_n, const int dim){

        if(dim == 0){
            return &arr(i, j + i_n, k + j_n);
        }
    #ifdef WARPX_DIM_XZ
        else if(dim == 1 || (dim == 2)){
            return &arr(i + i_n, j + j_n, k);
        }
    #elif defined(WARPX_DIM_3D)
        else if(dim == 1){
            return &arr(i + i_n, j, k + j_n);
        }
        else if(dim == 2){
            return &arr(i + i_n, j + j_n, k);
        }
    #else
        else if(dim == 1){
            amrex::Abort("GetNeighborPtr: Only implemented in 2D3V and 3D3V");
        }
        else if(dim == 2){
            return &arr(i + i_n, j + j_n, k);
        }
    #endif

        amrex::Abort("GetNeighborPtr: dim must be 0, 1 or 2");

        return nullptr;
    }

#ifdef AMREX_USE_EB

#ifndef WARPX_DIM_RZ

    /**
    * \brief For the face of cell pointing in direction idim, return the number of faces
    * we need to intrude with the one-way extension. Returns only one or zero: one if the
    * face can be extended with the the one-way extension, zeros if it can't.
    *
    * \param[in] cell \c Dim3 storing the indices of the face to extended
    * \param[in] S_ext amount of area needed for the extension
    * \param[in] S_red \c Array4 storing the amount of  area each face can still give away
    * \param[in] flag_info_face \c Array4 storing face information
    * \param[in] flag_ext_face \c Array4 storing face information
    * \param[in] idim normal direction to the face in consideration (0 for x, 1 for y, 2 for z)
    */
    AMREX_GPU_DEVICE
    int ComputeNBorrowOneFaceExtension(
        const amrex::Dim3 cell, const amrex::Real S_ext,
        const amrex::Array4<amrex::Real>& S_red,
        const amrex::Array4<int>& flag_info_face,
        const amrex::Array4<int>& flag_ext_face, const int idim)
    {
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
                    if (GetNeighbor(S_red, i, j, k, i_n, j_n, idim) > S_ext
                        && (GetNeighbor(flag_info_face, i, j, k, i_n, j_n, idim) == 1
                        || GetNeighbor(flag_info_face, i, j, k, i_n, j_n, idim) == 2)
                        && flag_ext_face(i, j, k) && ! stop) {
                        n_borrow += 1;
                        stop = true;
                    }
                }
            }
        }

        return n_borrow;
    }


    /**
    * \brief For the face of cell pointing in direction idim, return the number of faces
    * we need to intrude with the eight-ways extension.
    *
    * \param[in] cell \c Dim3 storing the indices of the face to extended
    * \param[in] S_ext amount of area needed for the extension
    * \param[in] S_red \c Array4 storing the amount of  area each face can still give away
    * \param[in] S \c Array4 storing the area of face
    * \param[in] flag_info_face \c Array4 storing face information
    * \param[in] idim normal direction to the face in consideration (0 for x, 1 for y, 2 for z)
    */
    AMREX_GPU_DEVICE
    int ComputeNBorrowEightFacesExtension(
        const amrex::Dim3 cell, const amrex::Real S_ext,
        const amrex::Array4<amrex::Real> &S_red,
        const amrex::Array4<amrex::Real> &S,
        const amrex::Array4<int> &flag_info_face,
        int idim)
    {
        const int i = cell.x;
        const int j = cell.y;
        const int k = cell.z;
        int n_borrow = 0;
        amrex::Array2D<int, 0, 2, 0, 2> local_avail{};

        for(int i_loc = 0; i_loc <= 2; i_loc++){
            for(int j_loc = 0; j_loc <= 2; j_loc++){
                const int flag = GetNeighbor(flag_info_face, i, j, k, i_loc - 1, j_loc - 1, idim);
                local_avail(i_loc, j_loc) = flag == 1 || flag == 2;
            }
        }

        amrex::Real denom = local_avail(0, 1) * GetNeighbor(S, i, j, k, -1, 0, idim) +
                            local_avail(2, 1) * GetNeighbor(S, i, j, k, 1, 0, idim) +
                            local_avail(1, 0) * GetNeighbor(S, i, j, k, 0, -1, idim) +
                            local_avail(1, 2) * GetNeighbor(S, i, j, k, 0, 1, idim) +
                            local_avail(0, 0) * GetNeighbor(S, i, j, k, -1, -1, idim) +
                            local_avail(2, 0) * GetNeighbor(S, i, j, k, 1, -1, idim) +
                            local_avail(0, 2) * GetNeighbor(S, i, j, k, -1, 1, idim) +
                            local_avail(2, 2) * GetNeighbor(S, i, j, k, 1, 1, idim);

        bool neg_face = true;

        while(denom >= S_ext && neg_face && denom > 0){
            neg_face = false;
            for (int i_n = -1; i_n < 2; i_n++) {
                for (int j_n = -1; j_n < 2; j_n++) {
                    if(local_avail(i_n + 1, j_n + 1)){
                        const amrex::Real patch = S_ext * GetNeighbor(S, i, j, k, i_n, j_n, idim) / denom;
                        if(GetNeighbor(S_red, i, j, k, i_n, j_n, idim) - patch <= 0) {
                            neg_face = true;
                            local_avail(i_n + 1, j_n + 1) = false;
                        }
                    }
                }
            }

            denom = local_avail(0, 1) * GetNeighbor(S, i, j, k, -1, 0, idim) +
                    local_avail(2, 1) * GetNeighbor(S, i, j, k, 1, 0, idim) +
                    local_avail(1, 0) * GetNeighbor(S, i, j, k, 0, -1, idim) +
                    local_avail(1, 2) * GetNeighbor(S, i, j, k, 0, 1, idim) +
                    local_avail(0, 0) * GetNeighbor(S, i, j, k, -1, -1, idim) +
                    local_avail(2, 0) * GetNeighbor(S, i, j, k, 1, -1, idim) +
                    local_avail(0, 2) * GetNeighbor(S, i, j, k, -1, 1, idim) +
                    local_avail(2, 2) * GetNeighbor(S, i, j, k, 1, 1, idim);
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

#endif //WARPX_DIM_RZ

#endif //AMREX_USE_EB
}


void
WarpX::ComputeFaceExtensions ()
{
    if (!EB::enabled()) {
        throw std::runtime_error("ComputeFaceExtensions only works when EBs are enabled at runtime");
    }
#ifdef AMREX_USE_EB
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    amrex::Array1D<int, 0, 2> N_ext_faces = ::CountExtFaces(m_flag_ext_face, maxLevel());
    ablastr::warn_manager::WMRecordWarning("Embedded Boundary",
            "Faces to be extended in x:\t" + std::to_string(N_ext_faces(0)) + "\n"
            +"Faces to be extended in y:\t" + std::to_string(N_ext_faces(1)) + "\n"
            +"Faces to be extended in z:\t" + std::to_string(N_ext_faces(2)),
            ablastr::warn_manager::WarnPriority::low
    );

    const auto Bfield = m_fields.get_alldirs(FieldType::Bfield_fp, maxLevel());
    ::init_borrowing(m_borrowing[maxLevel()], Bfield);

    // Cross-box bookkeeping: each fab decides borrowing only for the faces it
    // owns, but its lenders (and the faces it marks as intruded) can live in
    // ghost entries or in non-owned copies of shared nodal planes. The area
    // each lender gave away (lent_area) and the intruded marks
    // (intruded_mark) are therefore accumulated alongside the direct writes
    // and reduced to the owners between/after the passes. Single-box
    // non-periodic layouts skip every reduction and keep the historical
    // communication-free behavior bit-identically.
    const bool multi_box = (boxArray(maxLevel()).size() > 1)
        || Geom(maxLevel()).isAnyPeriodic();
    m_ect_needs_seam_sync = multi_box;

    std::array< std::unique_ptr<amrex::MultiFab>, 3 > lent_area;
    std::array< std::unique_ptr<amrex::iMultiFab>, 3 > intruded_mark;
    for (int idim = 0; idim < 3; ++idim) {
        auto const& Bmf = *m_fields.get(FieldType::Bfield_fp, Direction{idim}, maxLevel());
        lent_area[idim] = std::make_unique<amrex::MultiFab>(
            Bmf.boxArray(), Bmf.DistributionMap(), 1, amrex::IntVect(1));
        lent_area[idim]->setVal(0.0);
        intruded_mark[idim] = std::make_unique<amrex::iMultiFab>(
            Bmf.boxArray(), Bmf.DistributionMap(), 1, amrex::IntVect(1));
        intruded_mark[idim]->setVal(0);
    }

    // Reduce the lent-area records to the owners: apply only the remote part
    // (total minus this fab's own records, which were already subtracted
    // directly), then make all copies of area_mod owner-consistent and
    // ghost-fresh for the next pass.
    auto const sync_lent_areas = [&] () {
        if (!multi_box) { return; }
        const auto& period = Geom(maxLevel()).periodicity();
        for (int idim = 0; idim < 3; ++idim) {
            auto& lent = *lent_area[idim];
            amrex::MultiFab lent_local(lent.boxArray(), lent.DistributionMap(), 1,
                                       amrex::IntVect(0));
            amrex::MultiFab::Copy(lent_local, lent, 0, 0, 1, 0);
            lent.SumBoundary(0, 1, lent.nGrowVect(), amrex::IntVect(0), period);
            auto* S_mod_mf = m_fields.get(FieldType::area_mod, Direction{idim}, maxLevel());
            for (amrex::MFIter mfi(lent); mfi.isValid(); ++mfi) {
                const amrex::Box bx = mfi.validbox();
                auto const& tot = lent.const_array(mfi);
                auto const& loc = lent_local.const_array(mfi);
                auto const& S_mod = S_mod_mf->array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    const amrex::Real rem = tot(i, j, k) - loc(i, j, k);
                    // only-touch-if-changed keeps untouched faces bit-identical
                    if (rem != amrex::Real(0.)) { S_mod(i, j, k) -= rem; }
                });
            }
            S_mod_mf->OverrideSync(period);
            S_mod_mf->FillBoundary(period);
            lent.setVal(0.0);
        }
    };

    ComputeOneWayExtensions(lent_area, intruded_mark);
    sync_lent_areas();

    amrex::Array1D<int, 0, 2> N_ext_faces_after_one_way = ::CountExtFaces(m_flag_ext_face, maxLevel());
    ablastr::warn_manager::WMRecordWarning("Embedded Boundary",
            "Faces to be extended after one way extension in x:\t"
            + std::to_string(N_ext_faces_after_one_way(0)) + "\n"
            +"Faces to be extended after one way extension in y:\t"
            + std::to_string(N_ext_faces_after_one_way(1)) + "\n"
            +"Faces to be extended after one way extension in z:\t"
            + std::to_string(N_ext_faces_after_one_way(2)),
            ablastr::warn_manager::WarnPriority::low
    );

    ComputeEightWaysExtensions(lent_area, intruded_mark);
    sync_lent_areas();

    // Reduce the intruded marks to the owners and make the flag fields
    // owner-consistent before the BCK correction reads them. Marks only ever
    // target faces with flag 1 or 2 (both lendable), so deferring this to
    // after the second pass does not change any availability decision.
    if (multi_box) {
        const auto& period = Geom(maxLevel()).periodicity();
        for (int idim = 0; idim < 3; ++idim) {
            auto& marks = *intruded_mark[idim];
            marks.SumBoundary(0, 1, marks.nGrowVect(), amrex::IntVect(0), period);
            auto* info_mf = m_flag_info_face[maxLevel()][idim].get();
            for (amrex::MFIter mfi(marks); mfi.isValid(); ++mfi) {
                const amrex::Box bx = mfi.validbox();
                auto const& mk = marks.const_array(mfi);
                auto const& info = info_mf->array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    if (mk(i, j, k) > 0 && info(i, j, k) == 1) { info(i, j, k) = 2; }
                });
            }
            // These are integer flag fields (iMultiFab); the shared-seam
            // reconciliation between owners is done by OverrideSync above, so
            // the ablastr comms Fill interface only needs to propagate ghosts.
            // The iMultiFab overload has no nodal_sync option (no
            // FillBoundaryAndSync for integers), hence the period-only call.
            info_mf->OverrideSync(period);
            ablastr::utils::communication::FillBoundary(*info_mf, period);
            m_flag_ext_face[maxLevel()][idim]->OverrideSync(period);
            ablastr::utils::communication::FillBoundary(
                *m_flag_ext_face[maxLevel()][idim], period);
        }
    }

    ::shrink_borrowing(m_borrowing[maxLevel()], Bfield);

    amrex::Array1D<int, 0, 2> N_ext_faces_after_eight_ways = ::CountExtFaces(m_flag_ext_face, maxLevel());
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
        ::ApplyBCKCorrection<0>(
            CellSize(maxLevel()), m_fields, maxLevel(),
            m_flag_ext_face, m_flag_info_face);
        using_bck = true;
    }
#endif

    if (N_ext_faces_after_eight_ways(1) > 0) {
        ::ApplyBCKCorrection<1>(
            CellSize(maxLevel()), m_fields, maxLevel(),
            m_flag_ext_face, m_flag_info_face);
        using_bck = true;
    }

#if !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_RZ)
    if (N_ext_faces_after_eight_ways(2) > 0) {
        ::ApplyBCKCorrection<2>(
            CellSize(maxLevel()), m_fields, maxLevel(),
            m_flag_ext_face, m_flag_info_face);
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
WarpX::ComputeOneWayExtensions (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& lent_area,
    std::array< std::unique_ptr<amrex::iMultiFab>, 3 >& intruded_mark)
{
    amrex::ignore_unused(lent_area, intruded_mark);
    if (!EB::enabled()) {
        throw std::runtime_error("ComputeOneWayExtensions only works when EBs are enabled at runtime");
    }
#ifdef AMREX_USE_EB
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;
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
            FaceInfoBox::Neighbours* borrowing_neighbor_faces = borrowing.neighbor_faces.data();
            amrex::Real* borrowing_area = borrowing.area.data();
            int& vecs_size = borrowing.vecs_size;

            auto const &S_mod = m_fields.get(FieldType::area_mod, Direction{idim}, maxLevel())->array(mfi);

            auto const &owner = m_ect_face_owner_mask[maxLevel()][idim]->const_array(mfi);
            auto const &lent = lent_area[idim]->array(mfi);
            auto const &intruded = intruded_mark[idim]->array(mfi);

            const auto &lx = m_fields.get(FieldType::edge_lengths, Direction{0}, maxLevel())->array(mfi);
            const auto &ly = m_fields.get(FieldType::edge_lengths, Direction{1}, maxLevel())->array(mfi);
            const auto &lz = m_fields.get(FieldType::edge_lengths, Direction{2}, maxLevel())->array(mfi);

            vecs_size = amrex::Scan::PrefixSum<int>(ncells,
                                                    [=] AMREX_GPU_DEVICE (int icell) {
                const amrex::Dim3 cell = box.atOffset(icell).dim3();
                const int i = cell.x;
                const int j = cell.y;
                const int k = cell.z;
                // Only the owner copy of a face decides its borrowing (faces
                // on shared nodal planes exist in two fabs)
                if (owner(i, j, k) == 0) {
                    return 0;
                }
                // If the face doesn't need to be extended break the loop
                if (!flag_ext_face(i, j, k)) {
                    return 0;
                }

                const amrex::Real S_stab = ::ComputeSStab(i, j, k, lx, ly, lz, dx, dy, dz, idim);

                const amrex::Real S_ext = S_stab - S(i, j, k);
                const int n_borrow =
                    ::ComputeNBorrowOneFaceExtension(cell, S_ext, S_mod, flag_info_face,
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

                    const amrex::Real S_stab = ::ComputeSStab(i, j, k, lx, ly, lz, dx, dy, dz, idim);

                    const amrex::Real S_ext = S_stab - S(i, j, k);
                    int n_borrowed = 0;
                    for (int i_n = -1; i_n < 2; i_n++) {
                        for (int j_n = -1; j_n < 2; j_n++) {
                            //This if makes sure that we don't visit the "diagonal neighbours"
                            if (i_n != j_n && i_n != -j_n){
                                // Here a face is available if it doesn't need to be extended itself and if its
                                // area exceeds Sz_ext. Here we need to take into account if the intruded face
                                // has given away already some area, so we use Sz_red rather than Sz.
                                // If no face is available we don't do anything and we will need to use the
                                // multi-face extensions.
                                // The area is taken with an atomic test-and-subtract: on GPU
                                // several faces can try to borrow from the same intruded face
                                // concurrently, and a plain read-test-write lets the intruded
                                // face give the same area away more than once (issue #2257;
                                // equivalent to the fix proposed in PR #2298)
                                const bool borrowed = amrex::Gpu::Atomic::If(
                                    ::GetNeighborPtr(S_mod, i, j, k, i_n, j_n, idim),
                                    S_ext, amrex::Minus<amrex::Real>(),
                                    [=] (amrex::Real rem) {
                                        return rem > amrex::Real(0.)
                                            && (::GetNeighbor(flag_info_face, i, j, k, i_n, j_n, idim) == 1
                                                || ::GetNeighbor(flag_info_face, i, j, k, i_n, j_n, idim) == 2)
                                            && flag_ext_face(i, j, k);
                                    });

                                if (borrowed) {
                                    // Insert the index of the face info
                                    borrowing_inds[ps] = ps;
                                    // Store the information about the intruded face in the dataset of the
                                    // faces which are borrowing area
                                    FaceInfoBox::addConnectedNeighbor(i_n, j_n, ps,
                                                                      borrowing_neighbor_faces);
                                    borrowing_area[ps] = S_ext;

                                    ::SetNeighbor(flag_info_face, 2, i, j, k, i_n, j_n, idim);
                                    // Record the lent area and the intruded mark for the
                                    // cross-box reduction (the lender may live in a ghost
                                    // entry or a non-owned copy of a shared nodal plane)
                                    amrex::Gpu::Atomic::AddNoRet(
                                        ::GetNeighborPtr(lent, i, j, k, i_n, j_n, idim), S_ext);
                                    ::SetNeighbor(intruded, 1, i, j, k, i_n, j_n, idim);
                                    // Add the area to the intruding face.
                                    S_mod(i, j, k) = S(i, j, k) + S_ext;
                                    flag_ext_face(i, j, k) = false;
                                    n_borrowed += 1;
                                }
                            }
                        }
                    }
                    // A concurrently extended face may have drained the intruded
                    // face between the counting and the borrowing pass: keep the
                    // recorded size consistent with the entries actually written
                    // (the face then remains flagged for the eight-ways extension)
                    borrowing_size(i, j, k) = n_borrowed;
                    if (n_borrowed == 0) {
                        borrowing_inds_pointer(i, j, k) = nullptr;
                    }
                }
            }, amrex::Scan::Type::exclusive);
        }
    }

#endif
#endif
}


void
WarpX::ComputeEightWaysExtensions (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& lent_area,
    std::array< std::unique_ptr<amrex::iMultiFab>, 3 >& intruded_mark)
{
    amrex::ignore_unused(lent_area, intruded_mark);
    if (!EB::enabled()) {
        throw std::runtime_error("ComputeEightWaysExtensions only works when EBs are enabled at runtime");
    }
#ifdef AMREX_USE_EB
    using namespace amrex::literals;
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

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
            FaceInfoBox::Neighbours* borrowing_neighbor_faces = borrowing.neighbor_faces.data();
            amrex::Real* borrowing_area = borrowing.area.data();
            int& vecs_size = borrowing.vecs_size;

            auto const &S_mod = m_fields.get(FieldType::area_mod, Direction{idim}, maxLevel())->array(mfi);

            auto const &owner = m_ect_face_owner_mask[maxLevel()][idim]->const_array(mfi);
            auto const &lent = lent_area[idim]->array(mfi);
            auto const &intruded = intruded_mark[idim]->array(mfi);

            const auto &lx = m_fields.get(FieldType::edge_lengths, Direction{0}, maxLevel())->array(mfi);
            const auto &ly = m_fields.get(FieldType::edge_lengths, Direction{1}, maxLevel())->array(mfi);
            const auto &lz = m_fields.get(FieldType::edge_lengths, Direction{2}, maxLevel())->array(mfi);

            vecs_size += amrex::Scan::PrefixSum<int>(ncells,
                                                     [=] AMREX_GPU_DEVICE (int icell){
                const amrex::Dim3 cell = box.atOffset(icell).dim3();
                const int i = cell.x;
                const int j = cell.y;
                const int k = cell.z;
                // Only the owner copy of a face decides its borrowing
                if (owner(i, j, k) == 0) {
                    return 0;
                }
                // If the face doesn't need to be extended break the loop
                if (!flag_ext_face(i, j, k)) {
                    return 0;
                }
                const amrex::Real S_stab = ::ComputeSStab(i, j, k, lx, ly, lz, dx, dy, dz, idim);

                const amrex::Real S_ext = S_stab - S(i, j, k);
                const int n_borrow = ::ComputeNBorrowEightFacesExtension(cell, S_ext, S_mod, S,
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
                    const amrex::Real S_stab = ::ComputeSStab(i, j, k, lx, ly, lz, dx, dy, dz, idim);

                    const amrex::Real S_ext = S_stab - S(i, j, k);
                    amrex::Array2D<amrex::Real, 0, 2, 0, 2> local_avail{};
                    for(int i_loc = 0; i_loc <= 2; i_loc++){
                        for(int j_loc = 0; j_loc <= 2; j_loc++){
                            auto const flag = ::GetNeighbor(flag_info_face, i, j, k, i_loc - 1, j_loc - 1, idim);
                            local_avail(i_loc, j_loc) = flag == 1 || flag == 2;
                        }
                    }

                    amrex::Real denom = local_avail(0, 1) * ::GetNeighbor(S, i, j, k, -1, 0, idim) +
                                        local_avail(2, 1) * ::GetNeighbor(S, i, j, k, 1, 0, idim) +
                                        local_avail(1, 0) * ::GetNeighbor(S, i, j, k, 0, -1, idim) +
                                        local_avail(1, 2) * ::GetNeighbor(S, i, j, k, 0, 1, idim) +
                                        local_avail(0, 0) * ::GetNeighbor(S, i, j, k, -1, -1, idim) +
                                        local_avail(2, 0) * ::GetNeighbor(S, i, j, k, 1, -1, idim) +
                                        local_avail(0, 2) * ::GetNeighbor(S, i, j, k, -1, 1, idim) +
                                        local_avail(2, 2) * ::GetNeighbor(S, i, j, k, 1, 1, idim);

                    bool neg_face = true;

                    while(denom >= S_ext && neg_face && denom > 0){
                        neg_face = false;
                        for (int i_n = -1; i_n < 2; i_n++) {
                            for (int j_n = -1; j_n < 2; j_n++) {
                                if (local_avail(i_n + 1, j_n + 1) != 0_rt){
                                    const amrex::Real patch = S_ext * ::GetNeighbor(S, i, j, k, i_n, j_n, idim) / denom;
                                    if(::GetNeighbor(S_mod, i, j, k, i_n, j_n, idim) - patch <= 0) {
                                        neg_face = true;
                                        local_avail(i_n + 1, j_n + 1) = false;
                                    }
                                }
                            }
                        }

                        denom = local_avail(0, 1) * ::GetNeighbor(S, i, j, k, -1, 0, idim) +
                                local_avail(2, 1) * ::GetNeighbor(S, i, j, k, 1, 0, idim) +
                                local_avail(1, 0) * ::GetNeighbor(S, i, j, k, 0, -1, idim) +
                                local_avail(1, 2) * ::GetNeighbor(S, i, j, k, 0, 1, idim) +
                                local_avail(0, 0) * ::GetNeighbor(S, i, j, k, -1, -1, idim) +
                                local_avail(2, 0) * ::GetNeighbor(S, i, j, k, 1, -1, idim) +
                                local_avail(0, 2) * ::GetNeighbor(S, i, j, k, -1, 1, idim) +
                                local_avail(2, 2) * ::GetNeighbor(S, i, j, k, 1, 1, idim);
                    }

                    if(denom >= S_ext){
                        S_mod(i, j, k) = S(i, j, k);
                        int count = 0;
                        bool all_borrowed = true;
                        for (int i_n = -1; i_n < 2; i_n++) {
                            for (int j_n = -1; j_n < 2; j_n++) {
                                if(local_avail(i_n + 1, j_n + 1) != 0_rt && count < nborrow){
                                    const amrex::Real patch = S_ext * ::GetNeighbor(S, i, j, k, i_n, j_n, idim) / denom;
                                    // Atomic test-and-subtract, for the same reason as in
                                    // ComputeOneWayExtensions: an intruded face shared by
                                    // concurrently extended faces must not give the same
                                    // area away more than once (issue #2257, PR #2298)
                                    const bool borrowed = amrex::Gpu::Atomic::If(
                                        ::GetNeighborPtr(S_mod, i, j, k, i_n, j_n, idim),
                                        patch, amrex::Minus<amrex::Real>(),
                                        [=] (amrex::Real rem) {
                                            return rem > amrex::Real(0.);
                                        });
                                    if (!borrowed) {
                                        all_borrowed = false;
                                        continue;
                                    }
                                    borrowing_inds[ps + count] = ps + count;
                                    FaceInfoBox::addConnectedNeighbor(i_n, j_n, ps + count,
                                                                      borrowing_neighbor_faces);
                                    borrowing_area[ps + count] = patch;

                                    ::SetNeighbor(flag_info_face, 2, i, j, k, i_n, j_n, idim);
                                    // Record the lent area and the intruded mark for the
                                    // cross-box reduction
                                    amrex::Gpu::Atomic::AddNoRet(
                                        ::GetNeighborPtr(lent, i, j, k, i_n, j_n, idim), patch);
                                    ::SetNeighbor(intruded, 1, i, j, k, i_n, j_n, idim);

                                    S_mod(i, j, k) += patch;
                                    count +=1;
                                }
                            }
                        }
                        // Keep the recorded size consistent with the entries actually
                        // written; only a fully extended face is unflagged (a partially
                        // extended face would not reach its stable area and is reported
                        // by the unstable-faces check)
                        borrowing_size(i, j, k) = count;
                        if (count == 0) {
                            borrowing_inds_pointer(i, j, k) = nullptr;
                        }
                        if (all_borrowed) {
                            flag_ext_face(i, j, k) = false;
                        }
                    }
                    else {
                        // The face could not be extended after all (the area
                        // available shrank between the counting and the
                        // borrowing pass): record that nothing was borrowed
                        borrowing_size(i, j, k) = 0;
                        borrowing_inds_pointer(i, j, k) = nullptr;
                    }
                }
            }, amrex::Scan::Type::exclusive);
        }
    }
#endif
#endif
}
