/* Copyright 2019 Andrew Myers, Aurore Blelly, Axel Huebl
 * David Grote, Maxence Thevenet, Remi Lehe
 * Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#include "BoundaryConditions/PML.H"
#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
#   include "BoundaryConditions/PML_RZ.H"
#endif
#include "Fields.H"
#include "Filter/BilinearFilter.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "WarpXComm_K.H"
#include "WarpXSumGuardCells.H"
#include "Particles/MultiParticleContainer.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/coarsen/average.H>
#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/Communication.H>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MakeType.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

using namespace amrex;
using warpx::fields::FieldType;

namespace
{
    /**
     * \brief This function is called if \c warpx.do_current_centering = 1 and
     * it centers the currents from a nodal grid to a staggered grid (Yee) using
     * finite-order interpolation based on the Fornberg coefficients.
     *
     * \param[in,out] dst destination \c MultiFab where the results of the finite-order centering are stored
     * \param[in] src source \c MultiFab that contains the values of the nodal current to be centered
     * \param[in] cc_nox order of finite-order centering of currents, along x
     * \param[in] cc_noy order of finite-order centering of currents, along y
     * \param[in] cc_noz order of finite-order centering of currents, along z
     * \param[in] device_current_centering_stencil_coeffs_x stencil coefficients for finite-order centering of currents, along x
     * \param[in] device_current_centering_stencil_coeffs_y stencil coefficients for finite-order centering of currents, along y
     * \param[in] device_current_centering_stencil_coeffs_z stencil coefficients for finite-order centering of currents, along z
     */
    void UpdateCurrentNodalToStag (
        amrex::MultiFab& dst, const amrex::MultiFab& src,
        const int cc_nox, const int cc_noy, const int cc_noz,
        const amrex::Gpu::DeviceVector<amrex::Real>& device_current_centering_stencil_coeffs_x,
        const amrex::Gpu::DeviceVector<amrex::Real>& device_current_centering_stencil_coeffs_y,
        const amrex::Gpu::DeviceVector<amrex::Real>& device_current_centering_stencil_coeffs_z)
    {
        // If source and destination MultiFabs have the same index type, a simple copy is enough
        // (for example, this happens with the current along y in 2D, which is always fully nodal)
        if (dst.ixType() == src.ixType())
        {
            amrex::MultiFab::Copy(dst, src, 0, 0, dst.nComp(), dst.nGrowVect());
            return;
        }

        amrex::IntVect const& dst_stag = dst.ixType().toIntVect();

        // Source MultiFab always has nodal index type when this function is called
        amrex::IntVect const& src_stag = amrex::IntVect::TheNodeVector();

#ifdef AMREX_USE_OMP
    #pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(dst, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            // Loop over full box including ghost cells
            // (input arrays will be padded with zeros beyond ghost cells
            // for out-of-bound accesses due to large-stencil operations)
            const Box bx = mfi.growntilebox();

            amrex::Array4<amrex::Real const> const& src_arr = src.const_array(mfi);
            amrex::Array4<amrex::Real>       const& dst_arr = dst.array(mfi);

            // Device vectors of stencil coefficients used for finite-order centering of currents
            amrex::Real const * stencil_coeffs_x = device_current_centering_stencil_coeffs_x.data();
            amrex::Real const * stencil_coeffs_y = device_current_centering_stencil_coeffs_y.data();
            amrex::Real const * stencil_coeffs_z = device_current_centering_stencil_coeffs_z.data();

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
            {
                warpx_interp(j, k, l, dst_arr, src_arr, dst_stag, src_stag, cc_nox, cc_noy, cc_noz,
                             stencil_coeffs_x, stencil_coeffs_y, stencil_coeffs_z);
            });
        }
    }

    /**
     * \brief Copy all three vector-field components with the same source and destination layout.
     *
     * \param[in,out] dst Destination vector field
     * \param[in] src Source vector field
     * \param[in] ng Number of guard cells to copy
     */
    void CopyVectorField (
        ablastr::fields::VectorField const& dst,
        ablastr::fields::VectorField const& src,
        amrex::IntVect const& ng)
    {
        for (int idim = 0; idim < 3; ++idim) {
            amrex::MultiFab::Copy(
                *dst[idim], *src[idim], /*srccomp=*/0, /*dstcomp=*/0,
                dst[idim]->nComp(), ng);
        }
    }

    /**
     * \brief Copy all three components from a registered field type into an auxiliary vector field.
     *
     * \param[in,out] dst Destination vector field
     * \param[in] fields MultiFab register containing the source field
     * \param[in] src_type Field type to copy from
     * \param[in] lev AMR level
     * \param[in] ng Number of guard cells to copy
     */
    void CopyVectorField (
        ablastr::fields::VectorField const& dst,
        ablastr::fields::MultiFabRegister& fields,
        warpx::fields::FieldType const src_type,
        const int lev,
        amrex::IntVect const& ng)
    {
        using ablastr::fields::Direction;

        for (int idim = 0; idim < 3; ++idim) {
            amrex::MultiFab::Copy(
                *dst[idim], *fields.get(src_type, Direction{idim}, lev),
                /*srccomp=*/0, /*dstcomp=*/0, dst[idim]->nComp(), ng);
        }
    }

    /** Filter aux over ng_FieldGather; extra stencil cells stay local. */
    void FilterAuxGather (
        ablastr::fields::VectorField const& field_aux,
        BilinearFilter& bilinear_filter,
        amrex::IntVect const& ng_gather,
        int const lev,
        amrex::Periodicity const& period)
    {
        const amrex::IntVect extra =
            bilinear_filter.stencil_length_each_dir - amrex::IntVect(1);
        const amrex::IntVect ng_src = ng_gather + extra;

        for (int idim = 0; idim < 3; ++idim) {
            amrex::MultiFab& aux = *field_aux[idim];
            const int ncomp = aux.nComp();

            amrex::MultiFab src(aux.boxArray(), aux.DistributionMap(), ncomp, ng_src);
            src.setVal(0.0_rt);
            amrex::MultiFab::Copy(src, aux, 0, 0, ncomp, ng_gather);
            ablastr::utils::communication::FillBoundary(
                src, ng_src, WarpX::do_single_precision_comms, period);

            amrex::MultiFab dst(aux.boxArray(), aux.DistributionMap(), ncomp, ng_gather);
            bilinear_filter.ApplyStencil(dst, src, lev);
            amrex::MultiFab::Copy(aux, dst, 0, 0, ncomp, ng_gather);
        }
    }

    /**
     * \brief Update level-0 aux fields from averaged fine-patch fields or regular fine fields.
     *
     * \param[in] fields MultiFab register used when copying averaged fields
     * \param[in,out] field_aux Aux field to update
     * \param[in] field_fp Fine-patch field used when time averaging is off
     * \param[in] field_avg_fp_type Averaged fine-patch field type used when time averaging is on
     * \param[in] ng_src Number of source guard cells to copy
     */
    void CopyLevelZeroAuxiliaryData (
        ablastr::fields::MultiFabRegister& fields,
        ablastr::fields::MultiLevelVectorField const& field_aux,
        ablastr::fields::MultiLevelVectorField const& field_fp,
        warpx::fields::FieldType const field_avg_fp_type,
        amrex::IntVect const& ng_src)
    {
        if (WarpX::fft_do_time_averaging) {
            CopyVectorField(field_aux[0], fields, field_avg_fp_type, /*lev=*/0, ng_src);
        } else {
            CopyVectorField(field_aux[0], field_fp[0], ng_src);
        }
    }

    /**
     * \brief Center level-0 components onto the nodal aux grid.
     *
     * \param[in,out] field_aux Nodal aux field to update
     * \param[in] field_src Source field with the native component staggering
     * \param[in] device_field_centering_stencil_coeffs_x Centering stencil coefficients in x
     * \param[in] device_field_centering_stencil_coeffs_y Centering stencil coefficients in y
     * \param[in] device_field_centering_stencil_coeffs_z Centering stencil coefficients in z
     */
    void InterpLevelZeroStagToNodal (
        ablastr::fields::VectorField const& field_aux,
        ablastr::fields::VectorField const& field_src,
        amrex::Gpu::DeviceVector<amrex::Real> const& device_field_centering_stencil_coeffs_x,
        amrex::Gpu::DeviceVector<amrex::Real> const& device_field_centering_stencil_coeffs_y,
        amrex::Gpu::DeviceVector<amrex::Real> const& device_field_centering_stencil_coeffs_z)
    {
        amrex::IntVect const& Fx_stag = field_src[0]->ixType().toIntVect();
        amrex::IntVect const& Fy_stag = field_src[1]->ixType().toIntVect();
        amrex::IntVect const& Fz_stag = field_src[2]->ixType().toIntVect();

        // Aux data are always nodal in this update path.
        amrex::IntVect const& dst_stag = amrex::IntVect::TheNodeVector();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*field_aux[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Array4<Real> const& fx_aux = field_aux[0]->array(mfi);
            Array4<Real> const& fy_aux = field_aux[1]->array(mfi);
            Array4<Real> const& fz_aux = field_aux[2]->array(mfi);
            Array4<Real const> const& fx_src = field_src[0]->const_array(mfi);
            Array4<Real const> const& fy_src = field_src[1]->const_array(mfi);
            Array4<Real const> const& fz_src = field_src[2]->const_array(mfi);

            // Include ghost cells; the interpolation kernels zero-pad out-of-bounds reads.
            const Box bx = mfi.growntilebox();

            // Read the field-centering stencil once per tile for the three components.
            const int fg_nox = WarpX::field_centering_nox;
            const int fg_noy = WarpX::field_centering_noy;
            const int fg_noz = WarpX::field_centering_noz;

            amrex::Real const * stencil_coeffs_x =
                device_field_centering_stencil_coeffs_x.data();
            amrex::Real const * stencil_coeffs_y =
                device_field_centering_stencil_coeffs_y.data();
            amrex::Real const * stencil_coeffs_z =
                device_field_centering_stencil_coeffs_z.data();

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
            {
                // Interpolate each component from its source staggering to the nodal aux grid.
                warpx_interp(j, k, l, fx_aux, fx_src, dst_stag, Fx_stag, fg_nox, fg_noy, fg_noz,
                             stencil_coeffs_x, stencil_coeffs_y, stencil_coeffs_z);
                warpx_interp(j, k, l, fy_aux, fy_src, dst_stag, Fy_stag, fg_nox, fg_noy, fg_noz,
                             stencil_coeffs_x, stencil_coeffs_y, stencil_coeffs_z);
                warpx_interp(j, k, l, fz_aux, fz_src, dst_stag, Fz_stag, fg_nox, fg_noy, fg_noz,
                             stencil_coeffs_x, stencil_coeffs_y, stencil_coeffs_z);
            });
        }
    }

    /**
     * \brief Update one vector field on refined levels for staggered-to-nodal aux data.
     *
     * \param[in] fields MultiFab register containing fine, coarse, and coarse-aux fields
     * \param[in] field_fp Fine-patch field
     * \param[in,out] field_aux Aux field to update
     * \param[in] field_fp_type Fine-patch field type
     * \param[in] field_cp_type Coarse-patch field type
     * \param[in] field_cax_type Coarse-aux field type
     * \param[in] lev AMR level
     * \param[in] cnba Coarsened box array for temporary coarse aux data
     * \param[in] dm Distribution mapping for temporary coarse aux data
     * \param[in] cperiod Coarse-level periodicity
     * \param[in] refinement_ratio Refinement ratio between levels \c lev-1 and \c lev
     * \param[in] ng_src Number of source guard cells to copy
     * \param[in] electromagnetic_solver_id Active electromagnetic solver
     */
    void UpdateAuxiliaryDataStagToNodalField (
        ablastr::fields::MultiFabRegister& fields,
        ablastr::fields::MultiLevelVectorField const& field_fp,
        ablastr::fields::MultiLevelVectorField const& field_aux,
        warpx::fields::FieldType const field_fp_type,
        warpx::fields::FieldType const field_cp_type,
        warpx::fields::FieldType const field_cax_type,
        const int lev,
        BoxArray const& cnba,
        DistributionMapping const& dm,
        amrex::Periodicity const& cperiod,
        amrex::IntVect const& refinement_ratio,
        amrex::IntVect const& ng_src,
        ElectromagneticSolverAlgo const electromagnetic_solver_id)
    {
        using ablastr::fields::Direction;

        if (electromagnetic_solver_id != ElectromagneticSolverAlgo::None) {
            Array<std::unique_ptr<MultiFab>,3> Ftmp;
            if (fields.has_vector(field_cax_type, lev)) {
                // Reuse the solver-provided coarse-aux buffers when they already exist on this level.
                for (int idim = 0; idim < 3; ++idim) {
                    Ftmp[idim] = std::make_unique<MultiFab>(
                        *fields.get(field_cax_type, Direction{idim}, lev), amrex::make_alias, 0, 1);
                }
            } else {
                // Otherwise allocate a temporary coarse-aux field with the aux guard-cell footprint.
                const IntVect ngtmp = field_aux[lev-1][0]->nGrowVect();
                for (int idim = 0; idim < 3; ++idim) {
                    Ftmp[idim] = std::make_unique<MultiFab>(cnba, dm, 1, ngtmp);
                }
            }
            for (int idim = 0; idim < 3; ++idim) {
                Ftmp[idim]->setVal(0.0);
                const IntVect ng = Ftmp[idim]->nGrowVect();
                // Rebuild the coarsened aux data, including guards needed by the coarse/fine stencil.
                ablastr::utils::communication::ParallelCopy(
                    *Ftmp[idim], *field_aux[lev - 1][idim], 0, 0, 1,
                    ng_src, ng, WarpX::do_single_precision_comms, cperiod);
            }

            amrex::IntVect const& Fx_fp_stag =
                fields.get(field_fp_type, Direction{0}, lev)->ixType().toIntVect();
            amrex::IntVect const& Fy_fp_stag =
                fields.get(field_fp_type, Direction{1}, lev)->ixType().toIntVect();
            amrex::IntVect const& Fz_fp_stag =
                fields.get(field_fp_type, Direction{2}, lev)->ixType().toIntVect();

            amrex::IntVect const& Fx_cp_stag =
                fields.get(field_cp_type, Direction{0}, lev)->ixType().toIntVect();
            amrex::IntVect const& Fy_cp_stag =
                fields.get(field_cp_type, Direction{1}, lev)->ixType().toIntVect();
            amrex::IntVect const& Fz_cp_stag =
                fields.get(field_cp_type, Direction{2}, lev)->ixType().toIntVect();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(*field_aux[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Array4<Real> const& fx_aux = field_aux[lev][0]->array(mfi);
                Array4<Real> const& fy_aux = field_aux[lev][1]->array(mfi);
                Array4<Real> const& fz_aux = field_aux[lev][2]->array(mfi);
                Array4<Real const> const& fx_fp =
                    fields.get(field_fp_type, Direction{0}, lev)->const_array(mfi);
                Array4<Real const> const& fy_fp =
                    fields.get(field_fp_type, Direction{1}, lev)->const_array(mfi);
                Array4<Real const> const& fz_fp =
                    fields.get(field_fp_type, Direction{2}, lev)->const_array(mfi);
                Array4<Real const> const& fx_cp =
                    fields.get(field_cp_type, Direction{0}, lev)->const_array(mfi);
                Array4<Real const> const& fy_cp =
                    fields.get(field_cp_type, Direction{1}, lev)->const_array(mfi);
                Array4<Real const> const& fz_cp =
                    fields.get(field_cp_type, Direction{2}, lev)->const_array(mfi);
                Array4<Real const> const& fx_c = Ftmp[0]->const_array(mfi);
                Array4<Real const> const& fy_c = Ftmp[1]->const_array(mfi);
                Array4<Real const> const& fz_c = Ftmp[2]->const_array(mfi);

                const Box& bx = mfi.growntilebox();
                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                {
                    // Interpolate fine data together with coarse-patch and coarse-aux information.
                    warpx_interp(j, k, l, fx_aux, fx_fp, fx_cp, fx_c,
                                 Fx_fp_stag, Fx_cp_stag, refinement_ratio);
                    warpx_interp(j, k, l, fy_aux, fy_fp, fy_cp, fy_c,
                                 Fy_fp_stag, Fy_cp_stag, refinement_ratio);
                    warpx_interp(j, k, l, fz_aux, fz_fp, fz_cp, fz_c,
                                 Fz_fp_stag, Fz_cp_stag, refinement_ratio);
                });
            }
        } else {
            amrex::IntVect const& Fx_fp_stag = field_fp[lev][0]->ixType().toIntVect();
            amrex::IntVect const& Fy_fp_stag = field_fp[lev][1]->ixType().toIntVect();
            amrex::IntVect const& Fz_fp_stag = field_fp[lev][2]->ixType().toIntVect();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(*field_aux[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Array4<Real> const& fx_aux = field_aux[lev][0]->array(mfi);
                Array4<Real> const& fy_aux = field_aux[lev][1]->array(mfi);
                Array4<Real> const& fz_aux = field_aux[lev][2]->array(mfi);
                Array4<Real const> const& fx_fp = field_fp[lev][0]->const_array(mfi);
                Array4<Real const> const& fy_fp = field_fp[lev][1]->const_array(mfi);
                Array4<Real const> const& fz_fp = field_fp[lev][2]->const_array(mfi);

                const Box& bx = mfi.growntilebox();
                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                {
                    // Electrostatic fields only interpolate each component from its field layout to nodal aux.
                    warpx_interp(j, k, l, fx_aux, fx_fp, Fx_fp_stag);
                    warpx_interp(j, k, l, fy_aux, fy_fp, Fy_fp_stag);
                    warpx_interp(j, k, l, fz_aux, fz_fp, Fz_fp_stag);
                });
            }
        }
    }

    /**
     * \brief Update one vector field on refined levels when all grids share a layout.
     *
     * \param[in] fields MultiFab register containing coarse and coarse-aux fields
     * \param[in] field_fp Fine-patch field
     * \param[in,out] field_aux Aux field to update
     * \param[in] field_cp_type Coarse-patch field type
     * \param[in] field_cax_type Coarse-aux field type
     * \param[in] lev AMR level
     * \param[in] ng_src Number of source guard cells to copy
     * \param[in] crse_period Coarse-level periodicity
     * \param[in] refinement_ratio Refinement ratio between levels \c lev-1 and \c lev
     * \param[in] electromagnetic_solver_id Active electromagnetic solver
     */
    void UpdateAuxiliaryDataSameTypeField (
        ablastr::fields::MultiFabRegister& fields,
        ablastr::fields::MultiLevelVectorField const& field_fp,
        ablastr::fields::MultiLevelVectorField const& field_aux,
        warpx::fields::FieldType const field_cp_type,
        warpx::fields::FieldType const field_cax_type,
        const int lev,
        amrex::IntVect const& ng_src,
        amrex::Periodicity const& crse_period,
        amrex::IntVect const& refinement_ratio,
        ElectromagneticSolverAlgo const electromagnetic_solver_id)
    {
        using ablastr::fields::Direction;

        if (electromagnetic_solver_id != ElectromagneticSolverAlgo::None)
        {
            const IntVect& ng = fields.get(field_cp_type, Direction{0}, lev)->nGrowVect();
            const DistributionMapping& dm =
                fields.get(field_cp_type, Direction{0}, lev)->DistributionMap();

            Array<std::unique_ptr<MultiFab>,3> dF;
            for (int idim = 0; idim < 3; ++idim) {
                // dF stores the coarse-level correction on the same index type as the level-lev fields.
                dF[idim] = std::make_unique<MultiFab>(
                    fields.get(field_cp_type, Direction{idim}, lev)->boxArray(), dm,
                    fields.get(field_cp_type, Direction{idim}, lev)->nComp(), ng);
                dF[idim]->setVal(0.0);

                // First import the previous level's aux field onto the coarsened layout of this level.
                ablastr::utils::communication::ParallelCopy(
                    *dF[idim], *field_aux[lev - 1][idim], 0, 0,
                    field_aux[lev - 1][idim]->nComp(), ng_src, ng,
                    WarpX::do_single_precision_comms, crse_period);

                if (fields.has_vector(field_cax_type, lev)) {
                    // Keep an explicit copy of that coarsened aux field when the solver requests it.
                    MultiFab::Copy(
                        *fields.get(field_cax_type, Direction{idim}, lev), *dF[idim],
                        0, 0, fields.get(field_cax_type, Direction{idim}, lev)->nComp(), ng);
                }

                // Convert coarse aux into the additive correction relative to the coarse patch field.
                MultiFab::Subtract(
                    *dF[idim], *fields.get(field_cp_type, Direction{idim}, lev),
                    0, 0, fields.get(field_cp_type, Direction{idim}, lev)->nComp(), ng);
            }

            amrex::IntVect const& Fx_stag = field_aux[lev-1][0]->ixType().toIntVect();
            amrex::IntVect const& Fy_stag = field_aux[lev-1][1]->ixType().toIntVect();
            amrex::IntVect const& Fz_stag = field_aux[lev-1][2]->ixType().toIntVect();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(*field_aux[lev][0]); mfi.isValid(); ++mfi)
            {
                Array4<Real> const& fx_aux = field_aux[lev][0]->array(mfi);
                Array4<Real> const& fy_aux = field_aux[lev][1]->array(mfi);
                Array4<Real> const& fz_aux = field_aux[lev][2]->array(mfi);
                Array4<Real const> const& fx_fp = field_fp[lev][0]->const_array(mfi);
                Array4<Real const> const& fy_fp = field_fp[lev][1]->const_array(mfi);
                Array4<Real const> const& fz_fp = field_fp[lev][2]->const_array(mfi);
                Array4<Real const> const& fx_c = dF[0]->const_array(mfi);
                Array4<Real const> const& fy_c = dF[1]->const_array(mfi);
                Array4<Real const> const& fz_c = dF[2]->const_array(mfi);

                amrex::ParallelFor(Box(fx_aux), Box(fy_aux), Box(fz_aux),
                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                {
                    // Add the coarse correction after refining the native fine data onto the aux grid.
                    warpx_interp(j, k, l, fx_aux, fx_fp, fx_c, Fx_stag, refinement_ratio);
                },
                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                {
                    warpx_interp(j, k, l, fy_aux, fy_fp, fy_c, Fy_stag, refinement_ratio);
                },
                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                {
                    warpx_interp(j, k, l, fz_aux, fz_fp, fz_c, Fz_stag, refinement_ratio);
                });
            }
        }
        else // electrostatic
        {
            for (int idim = 0; idim < 3; ++idim) {
                MultiFab::Copy(
                    *field_aux[lev][idim], *field_fp[lev][idim],
                    0, 0, field_aux[lev][idim]->nComp(), field_aux[lev][idim]->nGrowVect());
            }
        }
    }
}

void
WarpX::UpdateAuxiliaryData ()
{
    ABLASTR_PROFILE("WarpX::UpdateAuxiliaryData()");

    using ablastr::fields::Direction;

    amrex::MultiFab *Bfield_aux_lvl0_0 = m_fields.get(FieldType::Bfield_aux, Direction{0}, 0);

    ablastr::fields::MultiLevelVectorField const& Bfield_fp = m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level);

    // Choose the aux update path from the level-0 B-field staggering.
    if (Bfield_aux_lvl0_0->ixType() == Bfield_fp[0][0]->ixType()) {
        UpdateAuxiliaryDataSameType();
    } else {
        UpdateAuxiliaryDataStagToNodal();
    }

    // When loading particle fields from file, add the external fields.
    for (int lev = 0; lev <= finest_level; ++lev) {

        // External particle E-field maps.
        if (mypc->m_E_ext_particle_s == "read_from_file") {
            ablastr::fields::VectorField E_aux = m_fields.get_alldirs(FieldType::Efield_aux, lev);
            const auto& E_ext = m_fields.get_alldirs(FieldType::E_external_particle_field, lev);

            const auto& metaE = mypc->m_external_particle_fields_metadata.m_E_field_metadata;
            const int ncomp_src = E_ext[0]->nComp();

            // The number of external particle fields must match the field metadata.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                ncomp_src == static_cast<int>(metaE.size()),
                "Mismatch: E_external_particle_field nComp != number of E field metadata entries."
            );

            // Apply each external field map with its time-dependent scale factor.
            for (int ic = 0; ic < ncomp_src; ++ic) {
                const amrex::ParticleReal time_factor = metaE[ic].time_executor(t_new[lev]);

                // dst += time_factor * src(component=ic)
                amrex::Saxpy(*E_aux[0], time_factor, *E_ext[0], /*src_comp=*/ic, /*dst_comp=*/0, /*ncomp=*/1,
                            guard_cells.ng_FieldGather);
                amrex::Saxpy(*E_aux[1], time_factor, *E_ext[1], /*src_comp=*/ic, /*dst_comp=*/0, /*ncomp=*/1,
                            guard_cells.ng_FieldGather);
                amrex::Saxpy(*E_aux[2], time_factor, *E_ext[2], /*src_comp=*/ic, /*dst_comp=*/0, /*ncomp=*/1,
                            guard_cells.ng_FieldGather);
            }
        }

        // External particle B-field maps.
        if (mypc->m_B_ext_particle_s == "read_from_file") {
            ablastr::fields::VectorField B_aux = m_fields.get_alldirs(FieldType::Bfield_aux, lev);
            const auto& B_ext = m_fields.get_alldirs(FieldType::B_external_particle_field, lev);

            const auto& metaB = mypc->m_external_particle_fields_metadata.m_B_field_metadata;
            const int ncomp_src = B_ext[0]->nComp();

            // The number of external particle fields must match the field metadata.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                ncomp_src == static_cast<int>(metaB.size()),
                "Mismatch: B_external_particle_field nComp != number of B field metadata entries."
            );

            // Apply each external field map with its time-dependent scale factor.
            for (int ic = 0; ic < ncomp_src; ++ic) {
                const amrex::ParticleReal time_factor = metaB[ic].time_executor(t_new[lev]);

                // dst += time_factor * src(component=ic)
                amrex::Saxpy(*B_aux[0], time_factor, *B_ext[0], /*src_comp=*/ic, /*dst_comp=*/0, /*ncomp=*/1,
                            guard_cells.ng_FieldGather);
                amrex::Saxpy(*B_aux[1], time_factor, *B_ext[1], /*src_comp=*/ic, /*dst_comp=*/0, /*ncomp=*/1,
                            guard_cells.ng_FieldGather);
                amrex::Saxpy(*B_aux[2], time_factor, *B_ext[2], /*src_comp=*/ic, /*dst_comp=*/0, /*ncomp=*/1,
                            guard_cells.ng_FieldGather);
            }
        }
    }
    if (UseFilterEBAux()) {
        const int lev = 0;
        const amrex::Periodicity& period = Geom(lev).periodicity();
        FilterAuxGather(
            m_fields.get_alldirs(FieldType::Bfield_aux, lev),
            bilinear_filter, guard_cells.ng_FieldGather, lev, period);
        FilterAuxGather(
            m_fields.get_alldirs(FieldType::Efield_aux, lev),
            bilinear_filter, guard_cells.ng_FieldGather, lev, period);
    }

}

void
WarpX::UpdateAuxiliaryDataStagToNodal ()
{
#ifndef WARPX_USE_FFT
    if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( false,
            "WarpX::UpdateAuxiliaryDataStagToNodal: PSATD solver requires "
            "WarpX build with spectral solver support.");
    }
#endif
    ablastr::fields::MultiLevelVectorField const& Bfield_fp = m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level);
    ablastr::fields::MultiLevelVectorField const& Efield_fp = m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level);
    ablastr::fields::MultiLevelVectorField const& Efield_aux = m_fields.get_mr_levels_alldirs(FieldType::Efield_aux, finest_level);
    ablastr::fields::MultiLevelVectorField const& Bfield_aux = m_fields.get_mr_levels_alldirs(FieldType::Bfield_aux, finest_level);

    ablastr::fields::MultiLevelVectorField const & Bmf =
        WarpX::fft_do_time_averaging ?
        m_fields.get_mr_levels_alldirs(FieldType::Bfield_avg_fp, finest_level) :
        Bfield_fp;
    ablastr::fields::MultiLevelVectorField const & Emf =
        WarpX::fft_do_time_averaging ?
        m_fields.get_mr_levels_alldirs(FieldType::Efield_avg_fp, finest_level) :
        Efield_fp;

    // Level 0 only needs native-to-nodal centering, optionally from time-averaged fields.
    InterpLevelZeroStagToNodal(
        Bfield_aux[0], Bmf[0],
        device_field_centering_stencil_coeffs_x,
        device_field_centering_stencil_coeffs_y,
        device_field_centering_stencil_coeffs_z);
    InterpLevelZeroStagToNodal(
        Efield_aux[0], Emf[0],
        device_field_centering_stencil_coeffs_x,
        device_field_centering_stencil_coeffs_y,
        device_field_centering_stencil_coeffs_z);

    // Refined levels use the low-order coarse/fine interpolation path for both B and E.
    for (int lev = 1; lev <= finest_level; ++lev)
    {
        BoxArray const& nba = Bfield_aux[lev][0]->boxArray();
        BoxArray const& cnba = amrex::coarsen(nba,2);
        DistributionMapping const& dm = Bfield_aux[lev][0]->DistributionMap();
        amrex::Periodicity const& cperiod = Geom(lev-1).periodicity();

        amrex::IntVect const& refinement_ratio = refRatio(lev-1);
        amrex::IntVect const& ng_src = guard_cells.ng_FieldGather;

        UpdateAuxiliaryDataStagToNodalField(
            m_fields, Bfield_fp, Bfield_aux, FieldType::Bfield_fp, FieldType::Bfield_cp,
            FieldType::Bfield_cax, lev, cnba, dm, cperiod, refinement_ratio, ng_src,
            electromagnetic_solver_id);
        UpdateAuxiliaryDataStagToNodalField(
            m_fields, Efield_fp, Efield_aux, FieldType::Efield_fp, FieldType::Efield_cp,
            FieldType::Efield_cax, lev, cnba, dm, cperiod, refinement_ratio, ng_src,
            electromagnetic_solver_id);
    }
}

void
WarpX::UpdateAuxiliaryDataSameType ()
{
    // Update aux fields, including guard cells, up to ng_FieldGather.
    const amrex::IntVect& ng_src = guard_cells.ng_FieldGather;

    ablastr::fields::MultiLevelVectorField const Efield_fp = m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level);
    ablastr::fields::MultiLevelVectorField const Bfield_fp = m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level);
    ablastr::fields::MultiLevelVectorField const Efield_aux = m_fields.get_mr_levels_alldirs(FieldType::Efield_aux, finest_level);
    ablastr::fields::MultiLevelVectorField const Bfield_aux = m_fields.get_mr_levels_alldirs(FieldType::Bfield_aux, finest_level);

    // Level 0 copies fine to aux. In some configurations aux and fine fields are aliases,
    // and MultiFab::Copy detects that and does nothing.
    CopyLevelZeroAuxiliaryData(m_fields, Efield_aux, Efield_fp, FieldType::Efield_avg_fp, ng_src);
    CopyLevelZeroAuxiliaryData(m_fields, Bfield_aux, Bfield_fp, FieldType::Bfield_avg_fp, ng_src);

    // Refined levels add the coarse-patch correction through the same helper for B and E.
    for (int lev = 1; lev <= finest_level; ++lev)
    {
        const amrex::Periodicity& crse_period = Geom(lev-1).periodicity();
        amrex::IntVect const& refinement_ratio = refRatio(lev-1);

        UpdateAuxiliaryDataSameTypeField(
            m_fields, Bfield_fp, Bfield_aux, FieldType::Bfield_cp, FieldType::Bfield_cax,
            lev, ng_src, crse_period, refinement_ratio, electromagnetic_solver_id);
        UpdateAuxiliaryDataSameTypeField(
            m_fields, Efield_fp, Efield_aux, FieldType::Efield_cp, FieldType::Efield_cax,
            lev, ng_src, crse_period, refinement_ratio, electromagnetic_solver_id);
    }
}

void
WarpX::FillBoundaryB (IntVect ng, std::optional<bool> nodal_sync)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        FillBoundaryB(lev, ng, nodal_sync);
    }
}

void
WarpX::FillBoundaryE (IntVect ng, std::optional<bool> nodal_sync)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        FillBoundaryE(lev, ng, nodal_sync);
    }
}

void
WarpX::FillBoundaryF (IntVect ng, std::optional<bool> nodal_sync)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        FillBoundaryF(lev, ng, nodal_sync);
    }
}

void
WarpX::FillBoundaryG (IntVect ng, std::optional<bool> nodal_sync)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        FillBoundaryG(lev, ng, nodal_sync);
    }
}

void
WarpX::FillBoundaryB_avg (IntVect ng)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        FillBoundaryB_avg(lev, ng);
    }
}

void
WarpX::FillBoundaryE_avg (IntVect ng)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        FillBoundaryE_avg(lev, ng);
    }
}


void
WarpX::FillBoundaryE (int lev, IntVect ng, std::optional<bool> nodal_sync)
{
    FillBoundaryE(lev, PatchType::fine, ng, nodal_sync);
    if (lev > 0) { FillBoundaryE(lev, PatchType::coarse, ng, nodal_sync); }
}

void
WarpX::FillBoundaryE (const int lev, const PatchType patch_type, const amrex::IntVect ng, std::optional<bool> nodal_sync)
{
    std::array<amrex::MultiFab*,3> mf;
    amrex::Periodicity period;

    using ablastr::fields::Direction;

    if (patch_type == PatchType::fine)
    {
        mf     = {m_fields.get(FieldType::Efield_fp, Direction{0}, lev),
                  m_fields.get(FieldType::Efield_fp, Direction{1}, lev),
                  m_fields.get(FieldType::Efield_fp, Direction{2}, lev)};
        period = Geom(lev).periodicity();
    }
    else // coarse patch
    {
        mf     = {m_fields.get(FieldType::Efield_cp, Direction{0}, lev),
                  m_fields.get(FieldType::Efield_cp, Direction{1}, lev),
                  m_fields.get(FieldType::Efield_cp, Direction{2}, lev)};
        period = Geom(lev-1).periodicity();
    }

    // Exchange data between valid domain and PML
    // Fill guard cells in PML
    if (do_pml)
    {
        if (pml[lev] && pml[lev]->ok())
        {
            const std::array<amrex::MultiFab*,3> mf_pml =
                (patch_type == PatchType::fine) ?
                m_fields.get_alldirs(FieldType::pml_E_fp, lev) :
                m_fields.get_alldirs(FieldType::pml_E_cp, lev);

            pml[lev]->Exchange(mf_pml, mf, patch_type, do_pml_in_domain);
            pml[lev]->FillBoundary(mf_pml, patch_type, nodal_sync);
        }

#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
        if (pml_rz[lev])
        {
            pml_rz[lev]->FillBoundaryE(m_fields, patch_type, do_single_precision_comms, nodal_sync);
        }
#endif
    }

    // Fill guard cells in valid domain
    if (do_single_precision_comms)
    {
        for (int i = 0; i < 3; ++i)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng.allLE(mf[i]->nGrowVect()),
                "Error: in FillBoundaryE, requested more guard cells than allocated");

            const amrex::IntVect nghost = (m_safe_guard_cells) ? mf[i]->nGrowVect() : ng;
            ablastr::utils::communication::FillBoundary(*mf[i], nghost, do_single_precision_comms, period, nodal_sync);
        }
    }
    else
    {
        const amrex::Vector<MultiFab*> vec_mf(mf.begin(), mf.end());
        if (nodal_sync)
        {
            amrex::FillBoundaryAndSync_nowait(vec_mf, period);
            amrex::FillBoundaryAndSync_finish(vec_mf);
        }
        else
        {
            amrex::FillBoundary_nowait(vec_mf, period);
            amrex::FillBoundary_finish(vec_mf);
        }
    }
}

void
WarpX::FillBoundaryB (int lev, IntVect ng, std::optional<bool> nodal_sync)
{
    FillBoundaryB(lev, PatchType::fine, ng, nodal_sync);
    if (lev > 0) { FillBoundaryB(lev, PatchType::coarse, ng, nodal_sync); }
}

void
WarpX::FillBoundaryB (const int lev, const PatchType patch_type, const amrex::IntVect ng, std::optional<bool> nodal_sync)
{
    std::array<amrex::MultiFab*,3> mf;
    amrex::Periodicity period;

    using ablastr::fields::Direction;

    if (patch_type == PatchType::fine)
    {
        mf     = {m_fields.get(FieldType::Bfield_fp, Direction{0}, lev),
                  m_fields.get(FieldType::Bfield_fp, Direction{1}, lev),
                  m_fields.get(FieldType::Bfield_fp, Direction{2}, lev)};
        period = Geom(lev).periodicity();
    }
    else // coarse patch
    {
        mf     = {m_fields.get(FieldType::Bfield_cp, Direction{0}, lev),
                  m_fields.get(FieldType::Bfield_cp, Direction{1}, lev),
                  m_fields.get(FieldType::Bfield_cp, Direction{2}, lev)};
        period = Geom(lev-1).periodicity();
    }

    // Exchange data between valid domain and PML
    // Fill guard cells in PML
    if (do_pml)
    {
        if (pml[lev] && pml[lev]->ok())
        {
            const std::array<amrex::MultiFab*,3> mf_pml =
                (patch_type == PatchType::fine) ?
                m_fields.get_alldirs(FieldType::pml_B_fp, lev) :
                m_fields.get_alldirs(FieldType::pml_B_cp, lev);

            pml[lev]->Exchange(mf_pml, mf, patch_type, do_pml_in_domain);
            pml[lev]->FillBoundary(mf_pml, patch_type, nodal_sync);
        }

#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
        if (pml_rz[lev])
        {
            pml_rz[lev]->FillBoundaryB(m_fields, patch_type, do_single_precision_comms, nodal_sync);
        }
#endif
    }

    // Fill guard cells in valid domain
    if (do_single_precision_comms)
    {
        for (int i = 0; i < 3; ++i)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng.allLE(mf[i]->nGrowVect()),
                "Error: in FillBoundaryB, requested more guard cells than allocated");

            const amrex::IntVect nghost = (m_safe_guard_cells) ? mf[i]->nGrowVect() : ng;
            ablastr::utils::communication::FillBoundary(*mf[i], nghost, do_single_precision_comms, period, nodal_sync);
        }
    }
    else
    {
        const amrex::Vector<MultiFab*> vec_mf(mf.begin(), mf.end());
        if (nodal_sync)
        {
            amrex::FillBoundaryAndSync_nowait(vec_mf, period);
            amrex::FillBoundaryAndSync_finish(vec_mf);
        }
        else
        {
            amrex::FillBoundary_nowait(vec_mf, period);
            amrex::FillBoundary_finish(vec_mf);
        }
    }
}

void
WarpX::FillBoundaryE_avg(int lev, IntVect ng)
{
    FillBoundaryE_avg(lev, PatchType::fine, ng);
    if (lev > 0) { FillBoundaryE_avg(lev, PatchType::coarse, ng); }
}

void
WarpX::FillBoundaryE_avg (int lev, PatchType patch_type, IntVect ng)
{
    bool const skip_lev0_coarse_patch = true;

    if (patch_type == PatchType::fine)
    {
        if (do_pml && pml[lev]->ok())
        {
            WARPX_ABORT_WITH_MESSAGE("Averaged Galilean PSATD with PML is not yet implemented");
        }

        ablastr::fields::MultiLevelVectorField Efield_avg_fp = m_fields.get_mr_levels_alldirs(FieldType::Efield_avg_fp, finest_level);

        const amrex::Periodicity& period = Geom(lev).periodicity();
        if ( m_safe_guard_cells ){
            const Vector<MultiFab*> mf{Efield_avg_fp[lev][0],Efield_avg_fp[lev][1],Efield_avg_fp[lev][2]};
            ablastr::utils::communication::FillBoundary(mf, WarpX::do_single_precision_comms, period);
        } else {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng.allLE(Efield_avg_fp[lev][0]->nGrowVect()),
                "Error: in FillBoundaryE_avg, requested more guard cells than allocated");
            ablastr::utils::communication::FillBoundary(*Efield_avg_fp[lev][0], ng, WarpX::do_single_precision_comms, period);
            ablastr::utils::communication::FillBoundary(*Efield_avg_fp[lev][1], ng, WarpX::do_single_precision_comms, period);
            ablastr::utils::communication::FillBoundary(*Efield_avg_fp[lev][2], ng, WarpX::do_single_precision_comms, period);
        }
    }
    else if (patch_type == PatchType::coarse)
    {
        if (do_pml && pml[lev]->ok())
        {
            WARPX_ABORT_WITH_MESSAGE("Averaged Galilean PSATD with PML is not yet implemented");
        }

        ablastr::fields::MultiLevelVectorField Efield_avg_cp = m_fields.get_mr_levels_alldirs(FieldType::Efield_avg_cp, finest_level, skip_lev0_coarse_patch);

        const amrex::Periodicity& cperiod = Geom(lev-1).periodicity();
        if ( m_safe_guard_cells ) {
            const Vector<MultiFab*> mf{Efield_avg_cp[lev][0],Efield_avg_cp[lev][1],Efield_avg_cp[lev][2]};
            ablastr::utils::communication::FillBoundary(mf, WarpX::do_single_precision_comms, cperiod);

        } else {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng.allLE(Efield_avg_cp[lev][0]->nGrowVect()),
                "Error: in FillBoundaryE, requested more guard cells than allocated");
            ablastr::utils::communication::FillBoundary(*Efield_avg_cp[lev][0], ng, WarpX::do_single_precision_comms, cperiod);
            ablastr::utils::communication::FillBoundary(*Efield_avg_cp[lev][1], ng, WarpX::do_single_precision_comms, cperiod);
            ablastr::utils::communication::FillBoundary(*Efield_avg_cp[lev][2], ng, WarpX::do_single_precision_comms, cperiod);
        }
    }
}


void
WarpX::FillBoundaryB_avg (int lev, IntVect ng)
{
    FillBoundaryB_avg(lev, PatchType::fine, ng);
    if (lev > 0) { FillBoundaryB_avg(lev, PatchType::coarse, ng); }
}

void
WarpX::FillBoundaryB_avg (int lev, PatchType patch_type, IntVect ng)
{
    using ablastr::fields::Direction;

    bool const skip_lev0_coarse_patch = true;

    if (patch_type == PatchType::fine)
    {
        if (do_pml && pml[lev]->ok())
        {
            WARPX_ABORT_WITH_MESSAGE("Averaged Galilean PSATD with PML is not yet implemented");
        }

        ablastr::fields::MultiLevelVectorField Bfield_avg_fp = m_fields.get_mr_levels_alldirs(FieldType::Bfield_avg_fp, finest_level);

        const amrex::Periodicity& period = Geom(lev).periodicity();
        if ( m_safe_guard_cells ) {
            const Vector<MultiFab*> mf{Bfield_avg_fp[lev][0],Bfield_avg_fp[lev][1],Bfield_avg_fp[lev][2]};
            ablastr::utils::communication::FillBoundary(mf, WarpX::do_single_precision_comms, period);
        } else {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng.allLE(m_fields.get(FieldType::Bfield_avg_fp, Direction{0}, lev)->nGrowVect()),
                "Error: in FillBoundaryB_avg, requested more guard cells than allocated");
            ablastr::utils::communication::FillBoundary(*Bfield_avg_fp[lev][0], ng, WarpX::do_single_precision_comms, period);
            ablastr::utils::communication::FillBoundary(*Bfield_avg_fp[lev][1], ng, WarpX::do_single_precision_comms, period);
            ablastr::utils::communication::FillBoundary(*Bfield_avg_fp[lev][2], ng, WarpX::do_single_precision_comms, period);
        }
    }
    else if (patch_type == PatchType::coarse)
    {
        if (do_pml && pml[lev]->ok())
        {
            WARPX_ABORT_WITH_MESSAGE("Averaged Galilean PSATD with PML is not yet implemented");
        }

        ablastr::fields::MultiLevelVectorField Bfield_avg_cp = m_fields.get_mr_levels_alldirs(FieldType::Bfield_avg_cp, finest_level, skip_lev0_coarse_patch);

        const amrex::Periodicity& cperiod = Geom(lev-1).periodicity();
        if ( m_safe_guard_cells ){
            const Vector<MultiFab*> mf{Bfield_avg_cp[lev][0],Bfield_avg_cp[lev][1],Bfield_avg_cp[lev][2]};
            ablastr::utils::communication::FillBoundary(mf, WarpX::do_single_precision_comms, cperiod);
        } else {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng.allLE(Bfield_avg_cp[lev][0]->nGrowVect()),
                "Error: in FillBoundaryB_avg, requested more guard cells than allocated");
            ablastr::utils::communication::FillBoundary(*Bfield_avg_cp[lev][0], ng, WarpX::do_single_precision_comms, cperiod);
            ablastr::utils::communication::FillBoundary(*Bfield_avg_cp[lev][1], ng, WarpX::do_single_precision_comms, cperiod);
            ablastr::utils::communication::FillBoundary(*Bfield_avg_cp[lev][2], ng, WarpX::do_single_precision_comms, cperiod);
        }
    }
}

void
WarpX::FillBoundaryF (int lev, IntVect ng, std::optional<bool> nodal_sync)
{
    FillBoundaryF(lev, PatchType::fine, ng, nodal_sync);
    if (lev > 0) { FillBoundaryF(lev, PatchType::coarse, ng, nodal_sync); }
}

void
WarpX::FillBoundaryF (int lev, PatchType patch_type, IntVect ng, std::optional<bool> nodal_sync)
{
    if (patch_type == PatchType::fine)
    {
        if (do_pml && pml[lev] && pml[lev]->ok())
        {
            if (m_fields.has(FieldType::pml_F_fp, lev) && m_fields.has(FieldType::F_fp, lev)) {
                pml[lev]->Exchange(m_fields.get(FieldType::pml_F_fp, lev), m_fields.get(FieldType::F_fp, lev), patch_type, do_pml_in_domain);
            }
            if (m_fields.has(FieldType::pml_F_fp, lev)) {
                pml[lev]->FillBoundary(*m_fields.get(FieldType::pml_F_fp, lev), patch_type, nodal_sync);
            }
        }

        if (m_fields.has(FieldType::F_fp, lev))
        {
            const amrex::Periodicity& period = Geom(lev).periodicity();
            const amrex::IntVect& nghost = (m_safe_guard_cells) ? m_fields.get(FieldType::F_fp, lev)->nGrowVect() : ng;
            ablastr::utils::communication::FillBoundary(*m_fields.get(FieldType::F_fp, lev), nghost, WarpX::do_single_precision_comms, period, nodal_sync);
        }
    }
    else if (patch_type == PatchType::coarse)
    {
        if (do_pml && pml[lev] && pml[lev]->ok())
        {
            if (m_fields.has(FieldType::pml_F_cp, lev) && m_fields.has(FieldType::F_cp, lev)) {
                pml[lev]->Exchange(m_fields.get(FieldType::pml_F_cp, lev), m_fields.get(FieldType::F_cp, lev), patch_type, do_pml_in_domain);
            }
            if (m_fields.has(FieldType::pml_F_cp, lev)) {
                pml[lev]->FillBoundary(*m_fields.get(FieldType::pml_F_cp, lev), patch_type, nodal_sync);
            }
        }

        if (m_fields.has(FieldType::F_cp, lev))
        {
            const amrex::Periodicity& period = Geom(lev-1).periodicity();
            const amrex::IntVect& nghost = (m_safe_guard_cells) ? m_fields.get(FieldType::F_cp, lev)->nGrowVect() : ng;
            ablastr::utils::communication::FillBoundary(*m_fields.get(FieldType::F_cp, lev), nghost, WarpX::do_single_precision_comms, period, nodal_sync);
        }
    }
}

void WarpX::FillBoundaryG (int lev, IntVect ng, std::optional<bool> nodal_sync)
{
    FillBoundaryG(lev, PatchType::fine, ng, nodal_sync);

    if (lev > 0)
    {
        FillBoundaryG(lev, PatchType::coarse, ng, nodal_sync);
    }
}

void WarpX::FillBoundaryG (int lev, PatchType patch_type, IntVect ng, std::optional<bool> nodal_sync)
{
    if (patch_type == PatchType::fine)
    {
        if (do_pml && pml[lev] && pml[lev]->ok())
        {
            if (m_fields.has(FieldType::pml_G_fp,lev) && m_fields.has(FieldType::G_fp,lev)) {
                pml[lev]->Exchange(m_fields.get(FieldType::pml_G_fp, lev), m_fields.get(FieldType::G_fp, lev), patch_type, do_pml_in_domain);
            }
            if (m_fields.has(FieldType::pml_G_fp,lev)) {
                pml[lev]->FillBoundary(*m_fields.get(FieldType::pml_G_fp, lev), patch_type, nodal_sync);
            }
        }

        if (m_fields.has(FieldType::G_fp,lev))
        {
            const amrex::Periodicity& period = Geom(lev).periodicity();
            MultiFab* G_fp = m_fields.get(FieldType::G_fp,lev);
            const amrex::IntVect& nghost = (m_safe_guard_cells) ? G_fp->nGrowVect() : ng;
            ablastr::utils::communication::FillBoundary(*G_fp, nghost, WarpX::do_single_precision_comms, period, nodal_sync);
        }
    }
    else if (patch_type == PatchType::coarse)
    {
        if (do_pml && pml[lev] && pml[lev]->ok())
        {
            if (m_fields.has(FieldType::pml_G_cp,lev) && m_fields.has(FieldType::G_cp,lev)) {
                pml[lev]->Exchange(m_fields.get(FieldType::pml_G_cp, lev), m_fields.get(FieldType::G_cp, lev), patch_type, do_pml_in_domain);
            }
            if (m_fields.has(FieldType::pml_G_cp, lev)) {
                pml[lev]->FillBoundary(*m_fields.get(FieldType::pml_G_cp, lev), patch_type, nodal_sync);
            }
        }

        if (m_fields.has(FieldType::G_cp,lev))
        {
            const amrex::Periodicity& period = Geom(lev-1).periodicity();
            MultiFab* G_cp = m_fields.get(FieldType::G_cp,lev);
            const amrex::IntVect& nghost = (m_safe_guard_cells) ? G_cp->nGrowVect() : ng;
            ablastr::utils::communication::FillBoundary(*G_cp, nghost, WarpX::do_single_precision_comms, period, nodal_sync);
        }
    }
}

void
WarpX::FillBoundaryAux (IntVect ng)
{
    for (int lev = 0; lev <= finest_level-1; ++lev)
    {
        FillBoundaryAux(lev, ng);
    }
}

void
WarpX::FillBoundaryAux (int lev, IntVect ng)
{
    ablastr::fields::MultiLevelVectorField Efield_aux = m_fields.get_mr_levels_alldirs(FieldType::Efield_aux, finest_level);
    ablastr::fields::MultiLevelVectorField Bfield_aux = m_fields.get_mr_levels_alldirs(FieldType::Bfield_aux, finest_level);

    const amrex::Periodicity& period = Geom(lev).periodicity();
    ablastr::utils::communication::FillBoundary(*Efield_aux[lev][0], ng, WarpX::do_single_precision_comms, period);
    ablastr::utils::communication::FillBoundary(*Efield_aux[lev][1], ng, WarpX::do_single_precision_comms, period);
    ablastr::utils::communication::FillBoundary(*Efield_aux[lev][2], ng, WarpX::do_single_precision_comms, period);
    ablastr::utils::communication::FillBoundary(*Bfield_aux[lev][0], ng, WarpX::do_single_precision_comms, period);
    ablastr::utils::communication::FillBoundary(*Bfield_aux[lev][1], ng, WarpX::do_single_precision_comms, period);
    ablastr::utils::communication::FillBoundary(*Bfield_aux[lev][2], ng, WarpX::do_single_precision_comms, period);
}

void
WarpX::SyncCurrent (const std::string& current_fp_string)
{
    using ablastr::fields::Direction;

    ABLASTR_PROFILE("WarpX::SyncCurrent()");

    bool const skip_lev0_coarse_patch = true;

    ablastr::fields::MultiLevelVectorField const& J_fp = m_fields.get_mr_levels_alldirs(current_fp_string, finest_level);

    // If warpx.do_current_centering = 1, center currents from nodal grid to staggered grid
    if (do_current_centering)
    {
        ablastr::fields::MultiLevelVectorField const& J_fp_nodal = m_fields.get_mr_levels_alldirs(FieldType::current_fp_nodal, finest_level);

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(finest_level <= 1,
                                         "warpx.do_current_centering=1 not supported with more than one fine levels");
        for (int lev = 0; lev <= finest_level; lev++)
        {
            constexpr auto all_dirs = std::array{Direction{0}, Direction{1}, Direction{2}};
            for (const auto& dir : all_dirs){
                ::UpdateCurrentNodalToStag(
                    *J_fp[lev][dir], *J_fp_nodal[lev][dir],
                    m_current_centering_nox, m_current_centering_noy, m_current_centering_noz,
                    device_current_centering_stencil_coeffs_x,
                    device_current_centering_stencil_coeffs_y,
                    device_current_centering_stencil_coeffs_z);
            }
        }
    }

    // If there is a single level, we apply the filter on the fp data and
    // then call SumBoundary that adds data from different boxes. This needs
    // to be done because a particle near a box boundary may deposit current
    // at a given (i,j,k) that is on the edge or in the ghost region of that
    // box while at the same time that (i,j,k) is also in the valid region
    // of another box. After SumBoundary, the result is as if there is only
    // a single box on a single process. Also note we need to call
    // SumBoundary even if there is only a single process, because a process
    // may have multiple boxes. Furthermore, even if there is only a single
    // box on a single process, SumBoundary should also be called if there
    // are periodic boundaries. So we always call SumBoundary even if it
    // might be a no-op in some cases, because the function does not perform
    // any communication if not necessary.
    //
    // When there are multiple levels, we need to send data from fine levels
    // to coarse levels. In the implementation below, we loop over levels
    // from the finest to the coarsest. On each level, filtering and
    // SumBoundary are done as the last two things. So the communication
    // data on the sender side are always unfiltered and unsummed. The
    // receivers are responsible for filtering that is dependent on the grid
    // resolution. On the finest level, we coarsen the unsummed fp data onto
    // cp grids, which are the coarsened version of the fp grids with the
    // same DistributionMapping. Then on the level below, We use ParallelAdd
    // to add the finer level's cp data onto the current level's fp
    // data. After that, we apply filter and SumBoundary to the finer
    // level's cp MultiFab. At this time, the finer level's fp and cp data
    // have all been properly filtered and summed. For the current level, if
    // there are levels below this, we need to process this level's cp data
    // just like we have done for the finer level. The iteration continues
    // until we reach level 0. There are however two additional
    // complications.
    //
    // The first complication is that simply calling ParallelAdd to add the
    // finer level's cp data to the current level's fp data does not
    // work. Suppose there are multiple boxes on the current level (or just
    // a single box with periodic boundaries). A given (i,j,k) can be present
    // in more than one box for nodal data in AMReX.
    // At the time of calling ParallelAdd, the current
    // level's fp data have not been summed. Because of how ParallelAdd
    // works, all boxes with that (i,j,k) will receive the data. So there is
    // a double counting issue of those data points existing on multiple boxes. Note
    // that at this time, the current level's fp data have not been summed
    // and we will call SumBoundary on the fp data. That would overcount the
    // finer level's cp data. So we fix this issue by creating a temporary
    // MultiFab to receive the finer level's cp data. We also create a mask
    // that can mark only one instance of the data as the owner if there are
    // overlapping points among boxes. Using the mask, we can add the
    // temporary MultiFab's data to the fp MultiFab only if the source owns
    // the data.
    //
    // The other complication is there might be a current buffer depending a
    // runtime parameter. The current buffer data, if they exist, need to be
    // communicated to the coarser level's fp MultiFab just like the cp
    // data. A simple approach would be to call another ParallelAdd in
    // additional the ParallelAdd for the cp data. But we like to minimize
    // parallel communication. So we add the cp data to the current buffer
    // MultiFab and use the latter as the source of ParallelAdd
    // communication to the coarser level. Note that we can do it this way
    // but not the other way around of adding the current buffer data to the
    // cp MultiFab because we still need to use the original cp data whereas
    // the buffer data are no longer needed once they have been sent to the
    // coarser level. So there are two cases. If there is no current buffer,
    // the cp MultiFab is the source of communication. If there is a current
    // buffer, the buffer MultiFab is the source instead. In the
    // implementation below, we use an alias MultiFab to manage this.

    std::unique_ptr<MultiFab> mf_comm; // for communication between levels
    for (int idim = 0; idim < 3; ++idim)
    {
        for (int lev = finest_level; lev >= 0; --lev)
        {
            const int ncomp = J_fp[lev][Direction{idim}]->nComp();
            auto const& period = Geom(lev).periodicity();

            if (lev < finest_level)
            {
                // On a coarse level, the data in mf_comm comes from the
                // coarse patch of the fine level. They are unfiltered and uncommunicated.
                // We need to add it to the fine patch of the current level.
                MultiFab fine_lev_cp(J_fp[lev][Direction{idim}]->boxArray(),
                                     J_fp[lev][Direction{idim}]->DistributionMap(),
                                     ncomp, 0);
                fine_lev_cp.setVal(0.0);
                fine_lev_cp.ParallelAdd(*mf_comm, 0, 0, ncomp, mf_comm->nGrowVect(),
                                        IntVect(0), period);
                // We now need to create a mask to fix the double counting.
                auto owner_mask = amrex::OwnerMask(fine_lev_cp, period);
                auto const& mma = owner_mask->const_arrays();
                auto const& sma = fine_lev_cp.const_arrays();
                auto const& dma = J_fp[lev][Direction{idim}]->arrays();
                amrex::ParallelFor(fine_lev_cp, IntVect(0), ncomp,
                [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k, int n)
                {
                    if (mma[bno](i,j,k) && sma[bno](i,j,k,n) != 0.0_rt) {
                        dma[bno](i,j,k,n) += sma[bno](i,j,k,n);
                    }
                });
                // Now it's safe to apply filter and sumboundary on J_cp
                ablastr::fields::MultiLevelVectorField const& J_cp = m_fields.get_mr_levels_alldirs(FieldType::current_cp, finest_level, skip_lev0_coarse_patch);
                if (use_filter)
                {
                    ApplyFilterJ(J_cp, lev+1, idim);
                }
                SumBoundaryJ(J_cp, lev+1, idim, period);
            }

            if (lev > 0)
            {
                // On a fine level, we need to coarsen the current onto the
                // coarse level. This needs to be done before filtering because
                // filtering depends on the level. This is also done before any
                // same-level communication because it's easier this way to
                // avoid double counting.
                ablastr::fields::MultiLevelVectorField const& J_cp = m_fields.get_mr_levels_alldirs(FieldType::current_cp, finest_level, skip_lev0_coarse_patch);
                J_cp[lev][Direction{idim}]->setVal(0.0);
                ablastr::coarsen::average::Coarsen(*J_cp[lev][Direction{idim}],
                                                   *J_fp[lev][Direction{idim}],
                                                   refRatio(lev-1));
                if (m_fields.has(FieldType::current_buf, Direction{idim}, lev))
                {
                    ablastr::fields::MultiLevelVectorField const& J_buffer = m_fields.get_mr_levels_alldirs(FieldType::current_buf, finest_level, skip_lev0_coarse_patch);

                    IntVect const& ng = J_cp[lev][Direction{idim}]->nGrowVect();
                    AMREX_ASSERT(ng.allLE(J_buffer[lev][Direction{idim}]->nGrowVect()));
                    MultiFab::Add(*J_buffer[lev][Direction{idim}], *J_cp[lev][Direction{idim}],
                                  0, 0, ncomp, ng);
                    mf_comm = std::make_unique<MultiFab>
                        (*J_buffer[lev][Direction{idim}], amrex::make_alias, 0, ncomp);
                }
                else
                {
                    mf_comm = std::make_unique<MultiFab>
                        (*J_cp[lev][Direction{idim}], amrex::make_alias, 0, ncomp);
                }
            }

            if (use_filter)
            {
                ApplyFilterJ(J_fp, lev, idim);
            }
            SumBoundaryJ(J_fp, lev, idim, period);
        }
    }
}

void
WarpX::SyncMassMatricesPC ()
{
    ABLASTR_PROFILE("WarpX::SyncMassMatricesPC()");

    ablastr::fields::MultiLevelVectorField const& Sigma_fp = m_fields.get_mr_levels_alldirs("MassMatrices_PC", finest_level);

    for (int idim = 0; idim < 3; ++idim)
    {
        for (int lev = finest_level; lev >= 0; --lev)
        {
            auto const& period = Geom(lev).periodicity();
            SumBoundaryJ(Sigma_fp, lev, idim, period);
        }
    }
}

void
WarpX::SyncMassMatrices ()
{
    ABLASTR_PROFILE("WarpX::SyncMassMatrices()");

    for (int lev = finest_level; lev >= 0; --lev)
    {
        auto const& period = Geom(lev).periodicity();
        SumBoundaryJ(m_fields.get_mr_levels_alldirs(FieldType::MassMatrices_X, lev), lev, period);
        SumBoundaryJ(m_fields.get_mr_levels_alldirs(FieldType::MassMatrices_Y, lev), lev, period);
        SumBoundaryJ(m_fields.get_mr_levels_alldirs(FieldType::MassMatrices_Z, lev), lev, period);
    }
}

void
WarpX::SyncRho () {
    bool const skip_lev0_coarse_patch = true;
    const ablastr::fields::MultiLevelScalarField rho_fp = m_fields.has(FieldType::rho_fp, 0) ?
        m_fields.get_mr_levels(FieldType::rho_fp, finest_level) :
        ablastr::fields::MultiLevelScalarField{static_cast<size_t>(finest_level+1)};
    const ablastr::fields::MultiLevelScalarField rho_cp = m_fields.has(FieldType::rho_cp, 1) ?
        m_fields.get_mr_levels(FieldType::rho_cp, finest_level, skip_lev0_coarse_patch) :
        ablastr::fields::MultiLevelScalarField{static_cast<size_t>(finest_level+1)};
    const ablastr::fields::MultiLevelScalarField rho_buf = m_fields.has(FieldType::rho_buf, 1) ?
        m_fields.get_mr_levels(FieldType::rho_buf, finest_level, skip_lev0_coarse_patch) :
        ablastr::fields::MultiLevelScalarField{static_cast<size_t>(finest_level+1)};

    SyncRho(rho_fp, rho_cp, rho_buf);
}

void
WarpX::SyncRho (
    const ablastr::fields::MultiLevelScalarField& charge_fp,
    const ablastr::fields::MultiLevelScalarField& charge_cp,
    ablastr::fields::MultiLevelScalarField const & charge_buffer)
{
    ABLASTR_PROFILE("WarpX::SyncRho()");

    if (!charge_fp[0]) { return; }
    const int ncomp = charge_fp[0]->nComp();

    // See comments in WarpX::SyncCurrent for an explanation of the algorithm.

    std::unique_ptr<MultiFab> mf_comm; // for communication between levels
    for (int lev = finest_level; lev >= 0; --lev)
    {
        if (lev < finest_level)
        {
            auto const& period = Geom(lev).periodicity();

            // On a coarse level, the data in mf_comm comes from the
            // coarse patch of the fine level. They are unfiltered and uncommunicated.
            // We need to add it to the fine patch of the current level.
            MultiFab fine_lev_cp(charge_fp[lev]->boxArray(),
                                 charge_fp[lev]->DistributionMap(),
                                 ncomp, 0);
            fine_lev_cp.setVal(0.0);
            fine_lev_cp.ParallelAdd(*mf_comm, 0, 0, ncomp, mf_comm->nGrowVect(),
                                    IntVect(0), period);
            // We now need to create a mask to fix the double counting.
            auto owner_mask = amrex::OwnerMask(fine_lev_cp, period);
            auto const& mma = owner_mask->const_arrays();
            auto const& sma = fine_lev_cp.const_arrays();
            auto const& dma = charge_fp[lev]->arrays();
            amrex::ParallelFor(fine_lev_cp, IntVect(0), ncomp,
            [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k, int n)
            {
                if (mma[bno](i,j,k) && sma[bno](i,j,k,n) != 0.0_rt) {
                    dma[bno](i,j,k,n) += sma[bno](i,j,k,n);
                }
            });
            // Now it's safe to apply filter and sumboundary on charge_cp
            ApplyFilterandSumBoundaryRho(lev+1, lev, *charge_cp[lev+1], 0, ncomp);
        }

        if (lev > 0)
        {
            // On a fine level, we need to coarsen the data onto the coarse
            // level. This needs to be done before filtering because
            // filtering depends on the level. This is also done before any
            // same-level communication because it's easier this way to
            // avoid double counting.
            charge_cp[lev]->setVal(0.0);
            ablastr::coarsen::average::Coarsen(*charge_cp[lev],
                                               *charge_fp[lev],
                                               refRatio(lev-1));
            if (charge_buffer[lev])
            {
                IntVect const& ng = charge_cp[lev]->nGrowVect();
                AMREX_ASSERT(ng.allLE(charge_buffer[lev]->nGrowVect()));
                MultiFab::Add(*charge_buffer[lev], *charge_cp[lev], 0, 0, ncomp, ng);
                mf_comm = std::make_unique<MultiFab>
                    (*charge_buffer[lev], amrex::make_alias, 0, ncomp);
            }
            else
            {
                mf_comm = std::make_unique<MultiFab>
                    (*charge_cp[lev], amrex::make_alias, 0, ncomp);
            }
        }

        ApplyFilterandSumBoundaryRho(lev, lev, *charge_fp[lev], 0, ncomp);
    }
}

/** \brief Fills the values of the current on the coarse patch by
 *  averaging the values of the current of the fine patch (on the same level).
 */
void WarpX::RestrictCurrentFromFineToCoarsePatch (
    const ablastr::fields::MultiLevelVectorField& J_fp,
    const ablastr::fields::MultiLevelVectorField& J_cp,
    const int lev)
{
    J_cp[lev][0]->setVal(0.0);
    J_cp[lev][1]->setVal(0.0);
    J_cp[lev][2]->setVal(0.0);

    const IntVect& refinement_ratio = refRatio(lev-1);

    std::array<const MultiFab*,3> fine { J_fp[lev][0],
                                         J_fp[lev][1],
                                         J_fp[lev][2] };
    std::array<      MultiFab*,3> crse { J_cp[lev][0],
                                         J_cp[lev][1],
                                         J_cp[lev][2] };
    ablastr::coarsen::average::Coarsen(*crse[0], *fine[0], refinement_ratio );
    ablastr::coarsen::average::Coarsen(*crse[1], *fine[1], refinement_ratio );
    ablastr::coarsen::average::Coarsen(*crse[2], *fine[2], refinement_ratio );
}

void WarpX::ApplyFilterMF (
    const ablastr::fields::MultiLevelVectorField& mfvec,
    const int lev,
    const int idim)
{
    using ablastr::fields::Direction;

    amrex::MultiFab& mf = *mfvec[lev][Direction{idim}];

    const int ncomp = mf.nComp();
    const amrex::IntVect ngrow = mf.nGrowVect();
    amrex::MultiFab mf_filtered(mf.boxArray(), mf.DistributionMap(), ncomp, ngrow);
    bilinear_filter.ApplyStencil(mf_filtered, mf, lev);

    const int srccomp = 0;
    const int dstcomp = 0;
    amrex::MultiFab::Copy(mf, mf_filtered, srccomp, dstcomp, ncomp, ngrow);
}

void WarpX::ApplyFilterMF (
    const ablastr::fields::MultiLevelVectorField& mfvec,
    const int lev)
{
    for (int idim=0; idim<3; ++idim)
    {
        ApplyFilterMF(mfvec, lev, idim);
    }
}

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
amrex::IntVect WarpX::ApplyVolumeWeightedFilter (amrex::MultiFab& dst, const amrex::MultiFab& src_mf,
                                       const int lev,
                                       const int scomp, const int dcomp, const int ncomp)
{
    using namespace amrex::literals;
    constexpr int NODE = amrex::IndexType::NODE;

    const std::array<amrex::Real,3>& dx = CellSize(lev);
    const amrex::Real dr = dx[0];

    // Same volume conventions as ApplyInverseVolumeScalingToChargeDensity
    // and ...ToCurrentDensity (Verboncoeur JCP 174, 421-427 (2001) for the
    // modified axis factor).
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
    const amrex::Real axis_volume_factor = (m_verboncoeur_axis_correction ? 1.0_rt/3.0_rt : 1.0_rt/4.0_rt);
#elif defined(WARPX_DIM_RSPHERE)
    const amrex::Real axis_volume_factor = (m_verboncoeur_axis_correction ? 1.0_rt/4.0_rt : 1.0_rt/8.0_rt);
#endif

    const auto& bf = bilinear_filter;
    const int npass_r = static_cast<int>(bf.npass_each_dir[0]);
#if defined(WARPX_DIM_RZ)
    const int npass_z = static_cast<int>(bf.npass_each_dir[1]);
#else
    const int npass_z = 0;
#endif

    // Each pass consumes one defined guard layer from the outside while
    // pushing mass one layer outward. This runs before any guard-cell
    // sum, so source guard layers hold only local deposits, bounded by
    // the source guard width -- layers beyond it are genuinely zero.
    // Extending the working arrays by 2*npass keeps the defined region
    // at src.ng + npass after all passes, which covers the final mass
    // reach, so every layer a subsequent guard-cell sum folds holds
    // filtered data rather than a stale deposit.
    const amrex::IntVect ng = src_mf.nGrowVect();
    const amrex::IntVect npass_vec(AMREX_D_DECL(npass_r, npass_z, 0));
    const amrex::IntVect ng_tmp = ng + 2*npass_vec;
    amrex::MultiFab tmp_a(src_mf.boxArray(), src_mf.DistributionMap(), ncomp, ng_tmp);
    amrex::MultiFab tmp_b(src_mf.boxArray(), src_mf.DistributionMap(), ncomp, ng_tmp);
    tmp_a.setVal(0.0_rt);
    tmp_b.setVal(0.0_rt);
    amrex::MultiFab::Copy(tmp_a, src_mf, scomp, 0, ncomp, ng);

    // One binomial pass in flux form. Written as the divergence of a
    // diffusive two-point flux with face weights w_f, it conserves the
    // volume integral of u exactly, leaves constants untouched, reduces to
    // the standard (1/4, 1/2, 1/4) stencil where the volume factors are
    // uniform, and has zero flux through the axis face by construction.
    // dir = 0 sweeps radially with the geometric volume factors; dir = 1
    // sweeps axially where the volumes are uniform.
    // ng_avail tracks how many guard layers of the working arrays still
    // hold meaningful data; each pass lowers it by one in its sweep
    // direction, ending at src.ng + npass.
    amrex::IntVect ng_avail = ng_tmp;

    // Physical (non-periodic) domain boundaries: no smoothing flux crosses
    // them, so the filter never exchanges with guard cells that nothing
    // folds back -- the volume integral over the valid domain is conserved
    // exactly. Periodic directions keep the ordinary flux (the guard sum
    // restores it).
    const amrex::Box& domain = Geom(lev).Domain();
    const amrex::Periodicity& period = Geom(lev).periodicity();

    auto sweep = [&](amrex::MultiFab& out, const amrex::MultiFab& in, int dir)
    {
        amrex::IntVect ng_out = ng_avail;
        ng_out[dir] = std::max(0, ng_out[dir] - 1);
        const bool dir_periodic = period.isPeriodic(dir);

        for (amrex::MFIter mfi(in); mfi.isValid(); ++mfi)
        {
            const amrex::Box& valid = mfi.validbox();
            amrex::Box tb = convert(valid, in.ixType().toIntVect());

            const amrex::XDim3 xyzmin = WarpX::LowerCorner(valid, lev, 0._rt);
            const amrex::Real rminx = xyzmin.x + (tb.type(0) == NODE ? 0._rt : 0.5_rt*dr);
            const int irmin = lbound(valid).x;

            tb.grow(ng_out);

            amrex::Array4<amrex::Real const> const& u = in.const_array(mfi);
            amrex::Array4<amrex::Real> const& v = out.array(mfi);

            auto point_weight = [dr, rminx, irmin, axis_volume_factor]
                AMREX_GPU_DEVICE (int i) -> amrex::Real
            {
                const amrex::Real r = amrex::Math::abs(rminx + (i - irmin)*dr);
                if (r == 0._rt) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                    return MathConst::pi*dr*axis_volume_factor;
#elif defined(WARPX_DIM_RSPHERE)
                    return 4.0_rt/3.0_rt*MathConst::pi*dr*dr*axis_volume_factor;
#endif
                }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                return 2.0_rt*MathConst::pi*r;
#elif defined(WARPX_DIM_RSPHERE)
                return 4.0_rt*MathConst::pi*r*r;
#endif
            };

            // Domain edge in this field's own index space: the point at
            // bigEnd owns the outward face on the physical boundary.
            const amrex::Box domain_t = amrex::convert(domain, in.ixType().toIntVect());
            const int dom_lo = domain_t.smallEnd(dir);
            const int dom_hi = domain_t.bigEnd(dir);

            if (dir == 0) {
                amrex::ParallelFor(tb, ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                    const amrex::Real r_signed = rminx + (i - irmin)*dr;
                    const amrex::Real w0 = point_weight(i);
                    // Face weights: arithmetic mean of the point volume
                    // factors, zeroed when the face sits at or below the
                    // axis (nothing crosses r = 0) or at the outer domain
                    // boundary (nothing leaks into wall guard cells).
                    const amrex::Real r_lo_face = r_signed - 0.5_rt*dr;
                    const amrex::Real r_hi_face = r_signed + 0.5_rt*dr;
                    amrex::Real w_lo = (r_lo_face <= 0._rt)
                        ? 0._rt : 0.5_rt*(point_weight(i-1) + w0);
                    amrex::Real w_hi = (r_hi_face <= 0._rt)
                        ? 0._rt : 0.5_rt*(w0 + point_weight(i+1));
                    if (i >= dom_hi) { w_hi = 0._rt; }
                    if (i > dom_hi)  { w_lo = 0._rt; }
                    v(i,j,k,n) = u(i,j,k,n) + 0.25_rt/w0 *
                        ( w_hi*(u(i+1,j,k,n) - u(i,j,k,n))
                        - w_lo*(u(i,j,k,n) - u(i-1,j,k,n)) );
                });
            } else {
                amrex::ParallelFor(tb, ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                    amrex::Real w_lo = 1._rt;
                    amrex::Real w_hi = 1._rt;
                    if (!dir_periodic) {
                        if (j >= dom_hi) { w_hi = 0._rt; }
                        if (j <= dom_lo) { w_lo = 0._rt; }
                        if (j > dom_hi)  { w_lo = 0._rt; }
                        if (j < dom_lo)  { w_hi = 0._rt; }
                    }
                    v(i,j,k,n) = u(i,j,k,n) + 0.25_rt *
                        ( w_hi*(u(i,j+1,k,n) - u(i,j,k,n))
                        - w_lo*(u(i,j,k,n) - u(i,j-1,k,n)) );
                });
            }
        }
        ng_avail = ng_out;
    };

    amrex::MultiFab* in = &tmp_a;
    amrex::MultiFab* out = &tmp_b;
    for (int p = 0; p < npass_r; ++p) {
        sweep(*out, *in, 0);
        std::swap(in, out);
    }
    for (int p = 0; p < npass_z; ++p) {
        sweep(*out, *in, 1);
        std::swap(in, out);
    }

    const amrex::IntVect ng_copy = amrex::min(dst.nGrowVect(), ng_avail);
    amrex::MultiFab::Copy(dst, *in, 0, dcomp, ncomp, ng_copy);
    return ng_copy;
}
#endif

void WarpX::ApplyFilterJ (
    const ablastr::fields::MultiLevelVectorField& current,
    const int lev,
    const int idim)
{
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    using ablastr::fields::Direction;
    amrex::MultiFab& J = *current[lev][Direction{idim}];
    const int ncomp = J.nComp();
    amrex::MultiFab J_filtered(J.boxArray(), J.DistributionMap(), ncomp, J.nGrowVect());
    const amrex::IntVect ng_filled =
        ApplyVolumeWeightedFilter(J_filtered, J, lev, 0, 0, ncomp);
    amrex::MultiFab::Copy(J, J_filtered, 0, 0, ncomp, ng_filled);
#else
    ApplyFilterMF(current, lev, idim);
#endif
}

void WarpX::ApplyFilterJ (
    const ablastr::fields::MultiLevelVectorField& current,
    const int lev)
{
    for (int idim=0; idim<3; ++idim)
    {
        ApplyFilterJ(current, lev, idim);
    }
}

void WarpX::SumBoundaryJ (
    const ablastr::fields::MultiLevelVectorField& current,
    const int lev,
    const int idim,
    const amrex::Periodicity& period)
{
    using ablastr::fields::Direction;

    amrex::MultiFab& J = *current[lev][Direction{idim}];

    const amrex::IntVect ng = J.nGrowVect();
    amrex::IntVect ng_depos_J = get_ng_depos_J();

    if (do_current_centering)
    {
#if   defined(WARPX_DIM_1D_Z)
        ng_depos_J[0] += m_current_centering_noz / 2;
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        ng_depos_J[0] += m_current_centering_nox / 2;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        ng_depos_J[0] += m_current_centering_nox / 2;
        ng_depos_J[1] += m_current_centering_noz / 2;
#elif defined(WARPX_DIM_3D)
        ng_depos_J[0] += m_current_centering_nox / 2;
        ng_depos_J[1] += m_current_centering_noy / 2;
        ng_depos_J[2] += m_current_centering_noz / 2;
#endif
    }

    if (use_filter)
    {
        ng_depos_J += bilinear_filter.stencil_length_each_dir - amrex::IntVect(1);
    }

    ng_depos_J.min(ng);

    const amrex::IntVect src_ngrow = ng_depos_J;
    const int icomp = 0;
    const int ncomp = J.nComp();
    WarpXSumGuardCells(J, period, src_ngrow, icomp, ncomp);
}

void WarpX::SumBoundaryJ (
    const ablastr::fields::MultiLevelVectorField& current,
    const int lev,
    const amrex::Periodicity& period)
{
    for (int idim=0; idim<3; ++idim)
    {
        SumBoundaryJ(current, lev, idim, period);
    }
}

/**
 * \brief Update the currents of `lev` by adding the currents from particles
 *         that are in the mesh refinement patches at `lev+1`
 *
 * More precisely, apply filter and sum boundaries for the current of:
 * - the fine patch of `lev`
 * - the coarse patch of `lev+1` (same resolution)
 * - the buffer regions of the coarse patch of `lev+1` (i.e. for particules
 * that are within the mesh refinement patch, but do not deposit on the
 * mesh refinement patch because they are too close to the boundary)
 *
 * Then update the fine patch of `lev` by adding the currents for the coarse
 * patch (and buffer region) of `lev+1`
 */
void WarpX::AddCurrentFromFineLevelandSumBoundary (
    const ablastr::fields::MultiLevelVectorField& J_fp,
    const ablastr::fields::MultiLevelVectorField& J_cp,
    const ablastr::fields::MultiLevelVectorField& J_buffer,
    const int lev)
{
    const amrex::Periodicity& period = Geom(lev).periodicity();

    if (use_filter)
    {
        ApplyFilterJ(J_fp, lev);
    }
    SumBoundaryJ(J_fp, lev, period);

    if (lev < finest_level)
    {
        // When there are current buffers, unlike coarse patch,
        // we don't care about the final state of them.

        for (int idim=0; idim<3; ++idim)
        {
            MultiFab mf(J_fp[lev][idim]->boxArray(),
                        J_fp[lev][idim]->DistributionMap(), J_fp[lev][idim]->nComp(), 0);
            mf.setVal(0.0);

            const IntVect ng = J_cp[lev+1][idim]->nGrowVect();

            if (use_filter && J_buffer[lev+1][idim])
            {
                ApplyFilterJ(J_cp, lev+1, idim);
                ApplyFilterJ(J_buffer, lev+1, idim);

                MultiFab::Add(
                    *J_buffer[lev+1][idim], *J_cp[lev+1][idim],
                    0, 0, J_buffer[lev+1][idim]->nComp(), ng);

                ablastr::utils::communication::ParallelAdd(
                    mf, *J_buffer[lev+1][idim], 0, 0,
                    J_buffer[lev+1][idim]->nComp(),
                    ng, amrex::IntVect(0),
                    do_single_precision_comms, period);
            }
            else if (use_filter) // but no buffer
            {
                ApplyFilterJ(J_cp, lev+1, idim);

                ablastr::utils::communication::ParallelAdd(
                    mf, *J_cp[lev+1][idim], 0, 0,
                    J_cp[lev+1][idim]->nComp(),
                    ng, amrex::IntVect(0),
                    do_single_precision_comms, period);
            }
            else if (J_buffer[lev+1][idim]) // but no filter
            {
                MultiFab::Add(
                    *J_buffer[lev+1][idim], *J_cp[lev+1][idim],
                    0, 0, J_buffer[lev+1][idim]->nComp(), ng);

                ablastr::utils::communication::ParallelAdd(
                    mf, *J_buffer[lev+1][idim], 0, 0,
                    J_buffer[lev+1][idim]->nComp(),
                    ng, amrex::IntVect(0),
                    do_single_precision_comms, period);
            }
            else // no filter, no buffer
            {
                ablastr::utils::communication::ParallelAdd(
                    mf, *J_cp[lev+1][idim], 0, 0,
                    J_cp[lev+1][idim]->nComp(),
                    ng, amrex::IntVect(0),
                    do_single_precision_comms, period);
            }
            SumBoundaryJ(J_cp, lev+1, idim, period);
            MultiFab::Add(*J_fp[lev][idim], mf, 0, 0, J_fp[lev+1][idim]->nComp(), 0);
        }
    }
}

void WarpX::RestrictRhoFromFineToCoarsePatch ( const int lev )
{
    if (m_fields.has(FieldType::rho_fp, lev)) {
        m_fields.get(FieldType::rho_cp, lev)->setVal(0.0);
        const IntVect& refinement_ratio = refRatio(lev-1);
        ablastr::coarsen::average::Coarsen(*m_fields.get(FieldType::rho_cp, lev), *m_fields.get(FieldType::rho_fp, lev), refinement_ratio );
    }
}

void WarpX::ApplyFilterandSumBoundaryRho (
    const ablastr::fields::MultiLevelScalarField& charge_fp,
    const ablastr::fields::MultiLevelScalarField& charge_cp,
    const int lev,
    PatchType patch_type,
    const int icomp,
    const int ncomp)
{
    const int glev = (patch_type == PatchType::fine) ? lev : lev-1;
    amrex::MultiFab* rho = (patch_type == PatchType::fine) ?
                                                  charge_fp[lev] : charge_cp[lev];
    if (rho == nullptr) { return; }
    ApplyFilterandSumBoundaryRho(lev, glev, *rho, icomp, ncomp);
}

void WarpX::ApplyFilterandSumBoundaryRho (int /*lev*/, int glev, amrex::MultiFab& rho, int icomp, int ncomp)
{
    const amrex::Periodicity& period = Geom(glev).periodicity();
    IntVect ng = rho.nGrowVect();
    IntVect ng_depos_rho = get_ng_depos_rho();
    if (use_filter) {
        ng += bilinear_filter.stencil_length_each_dir-1;
        ng_depos_rho += bilinear_filter.stencil_length_each_dir-1;
        ng_depos_rho.min(ng);
        MultiFab rf(rho.boxArray(), rho.DistributionMap(), ncomp, ng);
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        // In radial geometry, filter the extensive quantity (charge) rather
        // than the density so total charge is conserved. The flux-form
        // passes fill one guard layer less per pass than the stencil form;
        // seed the unfilled layers with the raw deposit and clamp the
        // guard sum to the well-defined region.
        MultiFab::Copy(rf, rho, icomp, 0, ncomp, amrex::min(ng, rho.nGrowVect()));
        const IntVect ng_filled = ApplyVolumeWeightedFilter(rf, rho, glev, icomp, 0, ncomp);
        ng_depos_rho.min(ng_filled);
#else
        bilinear_filter.ApplyStencil(rf, rho, glev, icomp, 0, ncomp);
#endif
        WarpXSumGuardCells(rho, rf, period, ng_depos_rho, icomp, ncomp );
    } else {
        ng_depos_rho.min(ng);
        WarpXSumGuardCells(rho, period, ng_depos_rho, icomp, ncomp);
    }
}

/**
 *  \brief Update the charge density of `lev` by adding the charge density from particles
 *         that are in the mesh refinement patches at `lev+1`
 *
 * More precisely, apply filter and sum boundaries for the charge density of:
 * - the fine patch of `lev`
 * - the coarse patch of `lev+1` (same resolution)
 * - the buffer regions of the coarse patch of `lev+1` (i.e. for particules
 * that are within the mesh refinement patch, but do not deposit on the
 * mesh refinement patch because they are too close to the boundary)
 *
 * Then update the fine patch of `lev` by adding the charge density for the coarse
 * patch (and buffer region) of `lev+1`
 */
void WarpX::AddRhoFromFineLevelandSumBoundary (
    const ablastr::fields::MultiLevelScalarField& charge_fp,
    const ablastr::fields::MultiLevelScalarField& charge_cp,
    ablastr::fields::MultiLevelScalarField const & charge_buffer,
    const int lev,
    const int icomp,
    const int ncomp)
{
    if (!charge_fp[lev]) { return; }

    ApplyFilterandSumBoundaryRho(charge_fp, charge_cp, lev, PatchType::fine, icomp, ncomp);

    if (lev < finest_level){

        const amrex::Periodicity& period = Geom(lev).periodicity();
        MultiFab mf(charge_fp[lev]->boxArray(),
                    charge_fp[lev]->DistributionMap(),
                    ncomp, 0);
        mf.setVal(0.0);
        IntVect ng = charge_cp[lev+1]->nGrowVect();
        IntVect ng_depos_rho = get_ng_depos_rho();
        if (use_filter && charge_buffer[lev+1])
        {
            // coarse patch of fine level
            ng += bilinear_filter.stencil_length_each_dir-1;
            ng_depos_rho += bilinear_filter.stencil_length_each_dir-1;
            ng_depos_rho.min(ng);
            MultiFab rhofc(charge_cp[lev+1]->boxArray(),
                           charge_cp[lev+1]->DistributionMap(), ncomp, ng);
            bilinear_filter.ApplyStencil(rhofc, *charge_cp[lev+1], lev+1, icomp, 0, ncomp);

            // buffer patch of fine level
            MultiFab rhofb(charge_buffer[lev+1]->boxArray(),
                           charge_buffer[lev+1]->DistributionMap(), ncomp, ng);
            bilinear_filter.ApplyStencil(rhofb, *charge_buffer[lev+1], lev+1, icomp, 0, ncomp);

            MultiFab::Add(rhofb, rhofc, 0, 0, ncomp, ng);

            ablastr::utils::communication::ParallelAdd(mf, rhofb, 0, 0, ncomp, ng, IntVect::TheZeroVector(),
                                                       WarpX::do_single_precision_comms, period);
            WarpXSumGuardCells( *charge_cp[lev+1], rhofc, period, ng_depos_rho, icomp, ncomp );
        }
        else if (use_filter) // but no buffer
        {
            ng += bilinear_filter.stencil_length_each_dir-1;
            ng_depos_rho += bilinear_filter.stencil_length_each_dir-1;
            ng_depos_rho.min(ng);
            MultiFab rf(charge_cp[lev+1]->boxArray(), charge_cp[lev+1]->DistributionMap(), ncomp, ng);
            bilinear_filter.ApplyStencil(rf, *charge_cp[lev+1], lev+1, icomp, 0, ncomp);

            ablastr::utils::communication::ParallelAdd(mf, rf, 0, 0, ncomp, ng, IntVect::TheZeroVector(),
                                                       WarpX::do_single_precision_comms, period);
            WarpXSumGuardCells( *charge_cp[lev+1], rf, period, ng_depos_rho, icomp, ncomp );
        }
        else if (charge_buffer[lev+1]) // but no filter
        {
            ng_depos_rho.min(ng);
            MultiFab::Add(*charge_buffer[lev+1],
                          *charge_cp[lev+1], icomp, icomp, ncomp,
                           charge_cp[lev+1]->nGrowVect());

            ablastr::utils::communication::ParallelAdd(mf, *charge_buffer[lev + 1], icomp, 0,
                                                       ncomp,
                                                       charge_buffer[lev + 1]->nGrowVect(),
                                                       IntVect::TheZeroVector(), WarpX::do_single_precision_comms,
                                                       period);
            WarpXSumGuardCells(*(charge_cp[lev+1]), period, ng_depos_rho, icomp, ncomp);
        }
        else // no filter, no buffer
        {
            ng_depos_rho.min(ng);
            ablastr::utils::communication::ParallelAdd(mf, *charge_cp[lev + 1], icomp, 0, ncomp,
                                                       charge_cp[lev + 1]->nGrowVect(),
                                                       IntVect::TheZeroVector(), WarpX::do_single_precision_comms,
                                                       period);
            WarpXSumGuardCells(*(charge_cp[lev+1]), period, ng_depos_rho, icomp, ncomp);
        }
        MultiFab::Add(*charge_fp[lev], mf, 0, icomp, ncomp, 0);
    }
}
