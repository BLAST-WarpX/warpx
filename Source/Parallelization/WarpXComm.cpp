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
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpXComm_K.H"
#include "WarpXSumGuardCells.H"
#include "Particles/MultiParticleContainer.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/coarsen/average.H>
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

void
WarpX::UpdateAuxilaryData ()
{
    WARPX_PROFILE("WarpX::UpdateAuxilaryData()");

    using ablastr::fields::Direction;

    amrex::MultiFab *Bfield_aux_lvl0_0 = m_fields.get(FieldType::Bfield_aux, Direction{0}, 0);

    ablastr::fields::MultiLevelVectorField const& Bfield_fp = m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level);

    if (Bfield_aux_lvl0_0->ixType() == Bfield_fp[0][0]->ixType()) {
        UpdateAuxilaryDataSameType();
    } else {
        UpdateAuxilaryDataStagToNodal();
    }

    // When loading particle fields from file: add the external fields:
    for (int lev = 0; lev <= finest_level; ++lev) {
        if (mypc->m_E_ext_particle_s == "read_from_file") {
            ablastr::fields::VectorField Efield_aux = m_fields.get_alldirs(FieldType::Efield_aux, lev);
            const auto& E_ext_lev = m_fields.get_alldirs(FieldType::E_external_particle_field, lev);
            amrex::MultiFab::Add(*Efield_aux[0], *E_ext_lev[0], 0, 0, E_ext_lev[0]->nComp(), guard_cells.ng_FieldGather);
            amrex::MultiFab::Add(*Efield_aux[1], *E_ext_lev[1], 0, 0, E_ext_lev[1]->nComp(), guard_cells.ng_FieldGather);
            amrex::MultiFab::Add(*Efield_aux[2], *E_ext_lev[2], 0, 0, E_ext_lev[2]->nComp(), guard_cells.ng_FieldGather);
        }
        if (mypc->m_B_ext_particle_s == "read_from_file") {
            ablastr::fields::VectorField Bfield_aux = m_fields.get_alldirs(FieldType::Bfield_aux, lev);
            const auto& B_ext_lev = m_fields.get_alldirs(FieldType::B_external_particle_field, lev);
            amrex::MultiFab::Add(*Bfield_aux[0], *B_ext_lev[0], 0, 0, B_ext_lev[0]->nComp(), guard_cells.ng_FieldGather);
            amrex::MultiFab::Add(*Bfield_aux[1], *B_ext_lev[1], 0, 0, B_ext_lev[1]->nComp(), guard_cells.ng_FieldGather);
            amrex::MultiFab::Add(*Bfield_aux[2], *B_ext_lev[2], 0, 0, B_ext_lev[2]->nComp(), guard_cells.ng_FieldGather);
        }
    }

}

void
WarpX::UpdateAuxilaryDataStagToNodal ()
{
#ifndef WARPX_USE_FFT
    if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( false,
            "WarpX::UpdateAuxilaryDataStagToNodal: PSATD solver requires "
            "WarpX build with spectral solver support.");
    }
#endif
    using ablastr::fields::Direction;

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

    const amrex::IntVect& Bx_stag = Bmf[0][0]->ixType().toIntVect();
    const amrex::IntVect& By_stag = Bmf[0][1]->ixType().toIntVect();
    const amrex::IntVect& Bz_stag = Bmf[0][2]->ixType().toIntVect();

    const amrex::IntVect& Ex_stag = Emf[0][0]->ixType().toIntVect();
    const amrex::IntVect& Ey_stag = Emf[0][1]->ixType().toIntVect();
    const amrex::IntVect& Ez_stag = Emf[0][2]->ixType().toIntVect();

    // Destination MultiFab (aux) always has nodal index type when this function is called
    const amrex::IntVect& dst_stag = amrex::IntVect::TheNodeVector();

    // For level 0, we only need to do the average.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*Bfield_aux[0][0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Array4<Real> const& bx_aux = Bfield_aux[0][0]->array(mfi);
        Array4<Real> const& by_aux = Bfield_aux[0][1]->array(mfi);
        Array4<Real> const& bz_aux = Bfield_aux[0][2]->array(mfi);
        Array4<Real const> const& bx_fp = Bmf[0][0]->const_array(mfi);
        Array4<Real const> const& by_fp = Bmf[0][1]->const_array(mfi);
        Array4<Real const> const& bz_fp = Bmf[0][2]->const_array(mfi);

        Array4<Real> const& ex_aux = Efield_aux[0][0]->array(mfi);
        Array4<Real> const& ey_aux = Efield_aux[0][1]->array(mfi);
        Array4<Real> const& ez_aux = Efield_aux[0][2]->array(mfi);
        Array4<Real const> const& ex_fp = Emf[0][0]->const_array(mfi);
        Array4<Real const> const& ey_fp = Emf[0][1]->const_array(mfi);
        Array4<Real const> const& ez_fp = Emf[0][2]->const_array(mfi);

        // Loop includes ghost cells (`growntilebox`)
        // (input arrays will be padded with zeros beyond ghost cells
        // for out-of-bound accesses due to large-stencil operations)
        const Box bx = mfi.growntilebox();

        // Order of finite-order centering of fields
        const int fg_nox = WarpX::field_centering_nox;
        const int fg_noy = WarpX::field_centering_noy;
        const int fg_noz = WarpX::field_centering_noz;

        // Device vectors of stencil coefficients used for finite-order centering of fields
        amrex::Real const * stencil_coeffs_x = WarpX::device_field_centering_stencil_coeffs_x.data();
        amrex::Real const * stencil_coeffs_y = WarpX::device_field_centering_stencil_coeffs_y.data();
        amrex::Real const * stencil_coeffs_z = WarpX::device_field_centering_stencil_coeffs_z.data();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
        {
            warpx_interp(j, k, l, bx_aux, bx_fp, dst_stag, Bx_stag, fg_nox, fg_noy, fg_noz,
                         stencil_coeffs_x, stencil_coeffs_y, stencil_coeffs_z);

            warpx_interp(j, k, l, by_aux, by_fp, dst_stag, By_stag, fg_nox, fg_noy, fg_noz,
                         stencil_coeffs_x, stencil_coeffs_y, stencil_coeffs_z);

            warpx_interp(j, k, l, bz_aux, bz_fp, dst_stag, Bz_stag, fg_nox, fg_noy, fg_noz,
                         stencil_coeffs_x, stencil_coeffs_y, stencil_coeffs_z);

            warpx_interp(j, k, l, ex_aux, ex_fp, dst_stag, Ex_stag, fg_nox, fg_noy, fg_noz,
                         stencil_coeffs_x, stencil_coeffs_y, stencil_coeffs_z);

            warpx_interp(j, k, l, ey_aux, ey_fp, dst_stag, Ey_stag, fg_nox, fg_noy, fg_noz,
                         stencil_coeffs_x, stencil_coeffs_y, stencil_coeffs_z);

            warpx_interp(j, k, l, ez_aux, ez_fp, dst_stag, Ez_stag, fg_nox, fg_noy, fg_noz,
                         stencil_coeffs_x, stencil_coeffs_y, stencil_coeffs_z);
        });
    }

    // NOTE: high-order interpolation is not implemented for mesh refinement
    for (int lev = 1; lev <= finest_level; ++lev)
    {
        BoxArray const& nba = Bfield_aux[lev][0]->boxArray();
        BoxArray const& cnba = amrex::coarsen(nba,2);
        DistributionMapping const& dm = Bfield_aux[lev][0]->DistributionMap();
        amrex::Periodicity const& cperiod = Geom(lev-1).periodicity();

        // Bfield
        {
            if (electromagnetic_solver_id != ElectromagneticSolverAlgo::None) {
                Array<std::unique_ptr<MultiFab>,3> Btmp;
                if (m_fields.has_vector(FieldType::Bfield_cax, lev)) {
                    for (int i = 0; i < 3; ++i) {
                        Btmp[i] = std::make_unique<MultiFab>(
                            *m_fields.get(FieldType::Bfield_cax, Direction{i}, lev), amrex::make_alias, 0, 1);
                    }
                } else {
                    const IntVect ngtmp = Bfield_aux[lev-1][0]->nGrowVect();
                    for (int i = 0; i < 3; ++i) {
                        Btmp[i] = std::make_unique<MultiFab>(cnba, dm, 1, ngtmp);
                    }
                }
                Btmp[0]->setVal(0.0);
                Btmp[1]->setVal(0.0);
                Btmp[2]->setVal(0.0);
                // ParallelCopy from coarse level
                for (int i = 0; i < 3; ++i) {
                    const IntVect ng = Btmp[i]->nGrowVect();
                    // Guard cells may not be up to date beyond ng_FieldGather
                    const amrex::IntVect& ng_src = guard_cells.ng_FieldGather;
                    // Copy Bfield_aux to Btmp, using up to ng_src (=ng_FieldGather) guard cells from
                    // Bfield_aux and filling up to ng (=nGrow) guard cells in Btmp
                    ablastr::utils::communication::ParallelCopy(*Btmp[i], *Bfield_aux[lev - 1][i], 0, 0, 1,
                                                                ng_src, ng, WarpX::do_single_precision_comms, cperiod);
                }

                const amrex::IntVect& refinement_ratio = refRatio(lev-1);

                const amrex::IntVect& Bx_fp_stag = m_fields.get(FieldType::Bfield_fp, Direction{0}, lev)->ixType().toIntVect();
                const amrex::IntVect& By_fp_stag = m_fields.get(FieldType::Bfield_fp, Direction{1}, lev)->ixType().toIntVect();
                const amrex::IntVect& Bz_fp_stag = m_fields.get(FieldType::Bfield_fp, Direction{2}, lev)->ixType().toIntVect();

                const amrex::IntVect& Bx_cp_stag = m_fields.get(FieldType::Bfield_cp, Direction{0}, lev)->ixType().toIntVect();
                const amrex::IntVect& By_cp_stag = m_fields.get(FieldType::Bfield_cp, Direction{1}, lev)->ixType().toIntVect();
                const amrex::IntVect& Bz_cp_stag = m_fields.get(FieldType::Bfield_cp, Direction{2}, lev)->ixType().toIntVect();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(*Bfield_aux[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    Array4<Real> const& bx_aux = Bfield_aux[lev][0]->array(mfi);
                    Array4<Real> const& by_aux = Bfield_aux[lev][1]->array(mfi);
                    Array4<Real> const& bz_aux = Bfield_aux[lev][2]->array(mfi);
                    Array4<Real const> const& bx_fp = m_fields.get(FieldType::Bfield_fp, Direction{0}, lev)->const_array(mfi);
                    Array4<Real const> const& by_fp = m_fields.get(FieldType::Bfield_fp, Direction{1}, lev)->const_array(mfi);
                    Array4<Real const> const& bz_fp = m_fields.get(FieldType::Bfield_fp, Direction{2}, lev)->const_array(mfi);
                    Array4<Real const> const& bx_cp = m_fields.get(FieldType::Bfield_cp, Direction{0}, lev)->const_array(mfi);
                    Array4<Real const> const& by_cp = m_fields.get(FieldType::Bfield_cp, Direction{1}, lev)->const_array(mfi);
                    Array4<Real const> const& bz_cp = m_fields.get(FieldType::Bfield_cp, Direction{2}, lev)->const_array(mfi);
                    Array4<Real const> const& bx_c = Btmp[0]->const_array(mfi);
                    Array4<Real const> const& by_c = Btmp[1]->const_array(mfi);
                    Array4<Real const> const& bz_c = Btmp[2]->const_array(mfi);

                    const Box& bx = mfi.growntilebox();
                    amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                    {
                        warpx_interp(j, k, l, bx_aux, bx_fp, bx_cp, bx_c, Bx_fp_stag, Bx_cp_stag, refinement_ratio);
                        warpx_interp(j, k, l, by_aux, by_fp, by_cp, by_c, By_fp_stag, By_cp_stag, refinement_ratio);
                        warpx_interp(j, k, l, bz_aux, bz_fp, bz_cp, bz_c, Bz_fp_stag, Bz_cp_stag, refinement_ratio);
                    });
                }
            }
            else { // electrostatic
                const amrex::IntVect& Bx_fp_stag = Bfield_fp[lev][0]->ixType().toIntVect();
                const amrex::IntVect& By_fp_stag = Bfield_fp[lev][1]->ixType().toIntVect();
                const amrex::IntVect& Bz_fp_stag = Bfield_fp[lev][2]->ixType().toIntVect();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(*Bfield_aux[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    Array4<Real> const& bx_aux = Bfield_aux[lev][0]->array(mfi);
                    Array4<Real> const& by_aux = Bfield_aux[lev][1]->array(mfi);
                    Array4<Real> const& bz_aux = Bfield_aux[lev][2]->array(mfi);
                    Array4<Real const> const& bx_fp = Bfield_fp[lev][0]->const_array(mfi);
                    Array4<Real const> const& by_fp = Bfield_fp[lev][1]->const_array(mfi);
                    Array4<Real const> const& bz_fp = Bfield_fp[lev][2]->const_array(mfi);

                    const Box& bx = mfi.growntilebox();
                    amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                    {
                        warpx_interp(j, k, l, bx_aux, bx_fp, Bx_fp_stag);
                        warpx_interp(j, k, l, by_aux, by_fp, By_fp_stag);
                        warpx_interp(j, k, l, bz_aux, bz_fp, Bz_fp_stag);
                    });
                }
            }
        }
        // Efield
        {
            if (electromagnetic_solver_id != ElectromagneticSolverAlgo::None) {
                Array<std::unique_ptr<MultiFab>,3> Etmp;
                if (m_fields.has_vector(FieldType::Efield_cax, lev)) {
                    for (int i = 0; i < 3; ++i) {
                        Etmp[i] = std::make_unique<MultiFab>(
                            *m_fields.get(FieldType::Efield_cax, Direction{i}, lev), amrex::make_alias, 0, 1);
                    }
                } else {
                    const IntVect ngtmp = Efield_aux[lev-1][0]->nGrowVect();
                    for (int i = 0; i < 3; ++i) {
                        Etmp[i] = std::make_unique<MultiFab>(
                            cnba, dm, 1, ngtmp);
                    }
                }
                Etmp[0]->setVal(0.0);
                Etmp[1]->setVal(0.0);
                Etmp[2]->setVal(0.0);
                // ParallelCopy from coarse level
                for (int i = 0; i < 3; ++i) {
                    const IntVect ng = Etmp[i]->nGrowVect();
                    // Guard cells may not be up to date beyond ng_FieldGather
                    const amrex::IntVect& ng_src = guard_cells.ng_FieldGather;
                    // Copy Efield_aux to Etmp, using up to ng_src (=ng_FieldGather) guard cells from
                    // Efield_aux and filling up to ng (=nGrow) guard cells in Etmp
                    ablastr::utils::communication::ParallelCopy(*Etmp[i], *Efield_aux[lev - 1][i], 0, 0, 1,
                                                                ng_src, ng, WarpX::do_single_precision_comms, cperiod);
                }

                const amrex::IntVect& refinement_ratio = refRatio(lev-1);

                const amrex::IntVect& Ex_fp_stag = m_fields.get(FieldType::Efield_fp, Direction{0}, lev)->ixType().toIntVect();
                const amrex::IntVect& Ey_fp_stag = m_fields.get(FieldType::Efield_fp, Direction{1}, lev)->ixType().toIntVect();
                const amrex::IntVect& Ez_fp_stag = m_fields.get(FieldType::Efield_fp, Direction{2}, lev)->ixType().toIntVect();

                const amrex::IntVect& Ex_cp_stag = m_fields.get(FieldType::Efield_cp, Direction{0}, lev)->ixType().toIntVect();
                const amrex::IntVect& Ey_cp_stag = m_fields.get(FieldType::Efield_cp, Direction{1}, lev)->ixType().toIntVect();
                const amrex::IntVect& Ez_cp_stag = m_fields.get(FieldType::Efield_cp, Direction{2}, lev)->ixType().toIntVect();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(*Efield_aux[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    Array4<Real> const& ex_aux = Efield_aux[lev][0]->array(mfi);
                    Array4<Real> const& ey_aux = Efield_aux[lev][1]->array(mfi);
                    Array4<Real> const& ez_aux = Efield_aux[lev][2]->array(mfi);
                    Array4<Real const> const& ex_fp = m_fields.get(FieldType::Efield_fp, Direction{0}, lev)->const_array(mfi);
                    Array4<Real const> const& ey_fp = m_fields.get(FieldType::Efield_fp, Direction{1}, lev)->const_array(mfi);
                    Array4<Real const> const& ez_fp = m_fields.get(FieldType::Efield_fp, Direction{2}, lev)->const_array(mfi);
                    Array4<Real const> const& ex_cp = m_fields.get(FieldType::Efield_cp, Direction{0}, lev)->const_array(mfi);
                    Array4<Real const> const& ey_cp = m_fields.get(FieldType::Efield_cp, Direction{1}, lev)->const_array(mfi);
                    Array4<Real const> const& ez_cp = m_fields.get(FieldType::Efield_cp, Direction{2}, lev)->const_array(mfi);
                    Array4<Real const> const& ex_c = Etmp[0]->const_array(mfi);
                    Array4<Real const> const& ey_c = Etmp[1]->const_array(mfi);
                    Array4<Real const> const& ez_c = Etmp[2]->const_array(mfi);

                    const Box& bx = mfi.growntilebox();
                    amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                    {
                        warpx_interp(j, k, l, ex_aux, ex_fp, ex_cp, ex_c, Ex_fp_stag, Ex_cp_stag, refinement_ratio);
                        warpx_interp(j, k, l, ey_aux, ey_fp, ey_cp, ey_c, Ey_fp_stag, Ey_cp_stag, refinement_ratio);
                        warpx_interp(j, k, l, ez_aux, ez_fp, ez_cp, ez_c, Ez_fp_stag, Ez_cp_stag, refinement_ratio);
                    });
                }
            }
            else { // electrostatic
                const amrex::IntVect& Ex_fp_stag = m_fields.get(FieldType::Efield_fp, Direction{0}, lev)->ixType().toIntVect();
                const amrex::IntVect& Ey_fp_stag = m_fields.get(FieldType::Efield_fp, Direction{1}, lev)->ixType().toIntVect();
                const amrex::IntVect& Ez_fp_stag = m_fields.get(FieldType::Efield_fp, Direction{2}, lev)->ixType().toIntVect();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(*Efield_aux[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    Array4<Real> const& ex_aux = Efield_aux[lev][0]->array(mfi);
                    Array4<Real> const& ey_aux = Efield_aux[lev][1]->array(mfi);
                    Array4<Real> const& ez_aux = Efield_aux[lev][2]->array(mfi);
                    Array4<Real const> const& ex_fp = m_fields.get(FieldType::Efield_fp, Direction{0}, lev)->const_array(mfi);
                    Array4<Real const> const& ey_fp = m_fields.get(FieldType::Efield_fp, Direction{1}, lev)->const_array(mfi);
                    Array4<Real const> const& ez_fp = m_fields.get(FieldType::Efield_fp, Direction{2}, lev)->const_array(mfi);

                    const Box& bx = mfi.growntilebox();
                    amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                    {
                        warpx_interp(j, k, l, ex_aux, ex_fp, Ex_fp_stag);
                        warpx_interp(j, k, l, ey_aux, ey_fp, Ey_fp_stag);
                        warpx_interp(j, k, l, ez_aux, ez_fp, Ez_fp_stag);
                    });
                }
            }
        }
    }
}

void
WarpX::UpdateAuxilaryDataSameType ()
{
    // Update aux field, including guard cells, up to ng_FieldGather
    const amrex::IntVect& ng_src = guard_cells.ng_FieldGather;

    using ablastr::fields::Direction;
    ablastr::fields::MultiLevelVectorField Efield_fp = m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level);
    ablastr::fields::MultiLevelVectorField Bfield_fp = m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level);
    ablastr::fields::MultiLevelVectorField Efield_aux = m_fields.get_mr_levels_alldirs(FieldType::Efield_aux, finest_level);
    ablastr::fields::MultiLevelVectorField Bfield_aux = m_fields.get_mr_levels_alldirs(FieldType::Bfield_aux, finest_level);

    // Level 0: Copy from fine to aux
    // Note: in some configurations, Efield_aux/Bfield_aux and Efield_fp/Bfield_fp are simply aliases to the
    // same MultiFab object. MultiFab::Copy operation automatically detects this and does nothing in this case.
    if (WarpX::fft_do_time_averaging)
    {
        MultiFab::Copy(*Efield_aux[0][0], *m_fields.get(FieldType::Efield_avg_fp, Direction{0}, 0), 0, 0, Efield_aux[0][0]->nComp(), ng_src);
        MultiFab::Copy(*Efield_aux[0][1], *m_fields.get(FieldType::Efield_avg_fp, Direction{1}, 0), 0, 0, Efield_aux[0][1]->nComp(), ng_src);
        MultiFab::Copy(*Efield_aux[0][2], *m_fields.get(FieldType::Efield_avg_fp, Direction{2}, 0), 0, 0, Efield_aux[0][2]->nComp(), ng_src);
        MultiFab::Copy(*Bfield_aux[0][0], *m_fields.get(FieldType::Bfield_avg_fp, Direction{0}, 0), 0, 0, Bfield_aux[0][0]->nComp(), ng_src);
        MultiFab::Copy(*Bfield_aux[0][1], *m_fields.get(FieldType::Bfield_avg_fp, Direction{1}, 0), 0, 0, Bfield_aux[0][1]->nComp(), ng_src);
        MultiFab::Copy(*Bfield_aux[0][2], *m_fields.get(FieldType::Bfield_avg_fp, Direction{2}, 0), 0, 0, Bfield_aux[0][2]->nComp(), ng_src);
    }
    else
    {
        MultiFab::Copy(*Efield_aux[0][0], *Efield_fp[0][0], 0, 0, Efield_aux[0][0]->nComp(), ng_src);
        MultiFab::Copy(*Efield_aux[0][1], *Efield_fp[0][1], 0, 0, Efield_aux[0][1]->nComp(), ng_src);
        MultiFab::Copy(*Efield_aux[0][2], *Efield_fp[0][2], 0, 0, Efield_aux[0][2]->nComp(), ng_src);
        MultiFab::Copy(*Bfield_aux[0][0], *Bfield_fp[0][0], 0, 0, Bfield_aux[0][0]->nComp(), ng_src);
        MultiFab::Copy(*Bfield_aux[0][1], *Bfield_fp[0][1], 0, 0, Bfield_aux[0][1]->nComp(), ng_src);
        MultiFab::Copy(*Bfield_aux[0][2], *Bfield_fp[0][2], 0, 0, Bfield_aux[0][2]->nComp(), ng_src);
    }
    for (int lev = 1; lev <= finest_level; ++lev)
    {
        const amrex::Periodicity& crse_period = Geom(lev-1).periodicity();
        const IntVect& ng = m_fields.get(FieldType::Bfield_cp, Direction{0}, lev)->nGrowVect();
        const DistributionMapping& dm = m_fields.get(FieldType::Bfield_cp, Direction{0}, lev)->DistributionMap();

        // B field
        {
            if (electromagnetic_solver_id != ElectromagneticSolverAlgo::None)
            {
                MultiFab dBx(m_fields.get(FieldType::Bfield_cp, Direction{0}, lev)->boxArray(), dm,
                             m_fields.get(FieldType::Bfield_cp, Direction{0}, lev)->nComp(), ng);
                MultiFab dBy(m_fields.get(FieldType::Bfield_cp, Direction{1}, lev)->boxArray(), dm,
                             m_fields.get(FieldType::Bfield_cp, Direction{1}, lev)->nComp(), ng);
                MultiFab dBz(m_fields.get(FieldType::Bfield_cp, Direction{2}, lev)->boxArray(), dm,
                             m_fields.get(FieldType::Bfield_cp, Direction{2}, lev)->nComp(), ng);
                dBx.setVal(0.0);
                dBy.setVal(0.0);
                dBz.setVal(0.0);

                // Copy Bfield_aux to the dB MultiFabs, using up to ng_src (=ng_FieldGather) guard
                // cells from Bfield_aux and filling up to ng (=nGrow) guard cells in the dB MultiFabs

                ablastr::utils::communication::ParallelCopy(dBx, *Bfield_aux[lev - 1][0], 0, 0,
                                                            Bfield_aux[lev - 1][0]->nComp(), ng_src, ng, WarpX::do_single_precision_comms,
                                                            crse_period);
                ablastr::utils::communication::ParallelCopy(dBy, *Bfield_aux[lev - 1][1], 0, 0,
                                                            Bfield_aux[lev - 1][1]->nComp(), ng_src, ng, WarpX::do_single_precision_comms,
                                                            crse_period);
                ablastr::utils::communication::ParallelCopy(dBz, *Bfield_aux[lev - 1][2], 0, 0,
                                                            Bfield_aux[lev - 1][2]->nComp(), ng_src, ng, WarpX::do_single_precision_comms,
                                                            crse_period);

                if (m_fields.has_vector(FieldType::Bfield_cax, lev))
                {
                    MultiFab::Copy(*m_fields.get(FieldType::Bfield_cax, Direction{0}, lev), dBx, 0, 0, m_fields.get(FieldType::Bfield_cax, Direction{0}, lev)->nComp(), ng);
                    MultiFab::Copy(*m_fields.get(FieldType::Bfield_cax, Direction{1}, lev), dBy, 0, 0, m_fields.get(FieldType::Bfield_cax, Direction{1}, lev)->nComp(), ng);
                    MultiFab::Copy(*m_fields.get(FieldType::Bfield_cax, Direction{2}, lev), dBz, 0, 0, m_fields.get(FieldType::Bfield_cax, Direction{2}, lev)->nComp(), ng);
                }
                MultiFab::Subtract(dBx, *m_fields.get(FieldType::Bfield_cp, Direction{0}, lev),
                                   0, 0, m_fields.get(FieldType::Bfield_cp, Direction{0}, lev)->nComp(), ng);
                MultiFab::Subtract(dBy, *m_fields.get(FieldType::Bfield_cp, Direction{1}, lev),
                                   0, 0, m_fields.get(FieldType::Bfield_cp, Direction{1}, lev)->nComp(), ng);
                MultiFab::Subtract(dBz, *m_fields.get(FieldType::Bfield_cp, Direction{2}, lev),
                                   0, 0, m_fields.get(FieldType::Bfield_cp, Direction{2}, lev)->nComp(), ng);

                const amrex::IntVect& refinement_ratio = refRatio(lev-1);

                const amrex::IntVect& Bx_stag = Bfield_aux[lev-1][0]->ixType().toIntVect();
                const amrex::IntVect& By_stag = Bfield_aux[lev-1][1]->ixType().toIntVect();
                const amrex::IntVect& Bz_stag = Bfield_aux[lev-1][2]->ixType().toIntVect();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(*Bfield_aux[lev][0]); mfi.isValid(); ++mfi)
                {
                    Array4<Real> const& bx_aux = Bfield_aux[lev][0]->array(mfi);
                    Array4<Real> const& by_aux = Bfield_aux[lev][1]->array(mfi);
                    Array4<Real> const& bz_aux = Bfield_aux[lev][2]->array(mfi);
                    Array4<Real const> const& bx_fp = Bfield_fp[lev][0]->const_array(mfi);
                    Array4<Real const> const& by_fp = Bfield_fp[lev][1]->const_array(mfi);
                    Array4<Real const> const& bz_fp = Bfield_fp[lev][2]->const_array(mfi);
                    Array4<Real const> const& bx_c = dBx.const_array(mfi);
                    Array4<Real const> const& by_c = dBy.const_array(mfi);
                    Array4<Real const> const& bz_c = dBz.const_array(mfi);

                    amrex::ParallelFor(Box(bx_aux), Box(by_aux), Box(bz_aux),
                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                    {
                        warpx_interp(j, k, l, bx_aux, bx_fp, bx_c, Bx_stag, refinement_ratio);
                    },
                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                    {
                        warpx_interp(j, k, l, by_aux, by_fp, by_c, By_stag, refinement_ratio);
                    },
                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                    {
                        warpx_interp(j, k, l, bz_aux, bz_fp, bz_c, Bz_stag, refinement_ratio);
                    });
                }
            }
            else // electrostatic
            {
                MultiFab::Copy(*Bfield_aux[lev][0], *Bfield_fp[lev][0], 0, 0, Bfield_aux[lev][0]->nComp(), Bfield_aux[lev][0]->nGrowVect());
                MultiFab::Copy(*Bfield_aux[lev][1], *Bfield_fp[lev][1], 0, 0, Bfield_aux[lev][1]->nComp(), Bfield_aux[lev][1]->nGrowVect());
                MultiFab::Copy(*Bfield_aux[lev][2], *Bfield_fp[lev][2], 0, 0, Bfield_aux[lev][2]->nComp(), Bfield_aux[lev][2]->nGrowVect());
            }
        }
        // E field
        {
            if (electromagnetic_solver_id != ElectromagneticSolverAlgo::None)
            {
                MultiFab dEx(m_fields.get(FieldType::Efield_cp, Direction{0}, lev)->boxArray(), dm,
                             m_fields.get(FieldType::Efield_cp, Direction{0}, lev)->nComp(), ng);
                MultiFab dEy(m_fields.get(FieldType::Efield_cp, Direction{1}, lev)->boxArray(), dm,
                             m_fields.get(FieldType::Efield_cp, Direction{1}, lev)->nComp(), ng);
                MultiFab dEz(m_fields.get(FieldType::Efield_cp, Direction{2}, lev)->boxArray(), dm,
                             m_fields.get(FieldType::Efield_cp, Direction{2}, lev)->nComp(), ng);
                dEx.setVal(0.0);
                dEy.setVal(0.0);
                dEz.setVal(0.0);

                // Copy Efield_aux to the dE MultiFabs, using up to ng_src (=ng_FieldGather) guard
                // cells from Efield_aux and filling up to ng (=nGrow) guard cells in the dE MultiFabs
                ablastr::utils::communication::ParallelCopy(dEx, *Efield_aux[lev - 1][0], 0, 0,
                                                            Efield_aux[lev - 1][0]->nComp(), ng_src, ng,
                                                            WarpX::do_single_precision_comms,
                                                            crse_period);
                ablastr::utils::communication::ParallelCopy(dEy, *Efield_aux[lev - 1][1], 0, 0,
                                                            Efield_aux[lev - 1][1]->nComp(), ng_src, ng,
                                                            WarpX::do_single_precision_comms,
                                                            crse_period);
                ablastr::utils::communication::ParallelCopy(dEz, *Efield_aux[lev - 1][2], 0, 0,
                                                            Efield_aux[lev - 1][2]->nComp(), ng_src, ng,
                                                            WarpX::do_single_precision_comms,
                                                            crse_period);

                if (m_fields.has_vector(FieldType::Efield_cax, lev))
                {
                    MultiFab::Copy(*m_fields.get(FieldType::Efield_cax, Direction{0}, lev), dEx, 0, 0, m_fields.get(FieldType::Efield_cax, Direction{0}, lev)->nComp(), ng);
                    MultiFab::Copy(*m_fields.get(FieldType::Efield_cax, Direction{1}, lev), dEy, 0, 0, m_fields.get(FieldType::Efield_cax, Direction{1}, lev)->nComp(), ng);
                    MultiFab::Copy(*m_fields.get(FieldType::Efield_cax, Direction{2}, lev), dEz, 0, 0, m_fields.get(FieldType::Efield_cax, Direction{2}, lev)->nComp(), ng);
                }
                MultiFab::Subtract(dEx, *m_fields.get(FieldType::Efield_cp, Direction{0}, lev),
                                   0, 0, m_fields.get(FieldType::Efield_cp, Direction{0}, lev)->nComp(), ng);
                MultiFab::Subtract(dEy, *m_fields.get(FieldType::Efield_cp, Direction{1}, lev),
                                   0, 0, m_fields.get(FieldType::Efield_cp, Direction{1}, lev)->nComp(), ng);
                MultiFab::Subtract(dEz, *m_fields.get(FieldType::Efield_cp, Direction{2}, lev),
                                   0, 0, m_fields.get(FieldType::Efield_cp, Direction{2}, lev)->nComp(), ng);

                const amrex::IntVect& refinement_ratio = refRatio(lev-1);

                const amrex::IntVect& Ex_stag = Efield_aux[lev-1][0]->ixType().toIntVect();
                const amrex::IntVect& Ey_stag = Efield_aux[lev-1][1]->ixType().toIntVect();
                const amrex::IntVect& Ez_stag = Efield_aux[lev-1][2]->ixType().toIntVect();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(*Efield_aux[lev][0]); mfi.isValid(); ++mfi)
                {
                    Array4<Real> const& ex_aux = Efield_aux[lev][0]->array(mfi);
                    Array4<Real> const& ey_aux = Efield_aux[lev][1]->array(mfi);
                    Array4<Real> const& ez_aux = Efield_aux[lev][2]->array(mfi);
                    Array4<Real const> const& ex_fp = m_fields.get(FieldType::Efield_fp, Direction{0}, lev)->const_array(mfi);
                    Array4<Real const> const& ey_fp = m_fields.get(FieldType::Efield_fp, Direction{1}, lev)->const_array(mfi);
                    Array4<Real const> const& ez_fp = m_fields.get(FieldType::Efield_fp, Direction{2}, lev)->const_array(mfi);
                    Array4<Real const> const& ex_c = dEx.const_array(mfi);
                    Array4<Real const> const& ey_c = dEy.const_array(mfi);
                    Array4<Real const> const& ez_c = dEz.const_array(mfi);

                    amrex::ParallelFor(Box(ex_aux), Box(ey_aux), Box(ez_aux),
                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                    {
                        warpx_interp(j, k, l, ex_aux, ex_fp, ex_c, Ex_stag, refinement_ratio);
                    },
                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                    {
                        warpx_interp(j, k, l, ey_aux, ey_fp, ey_c, Ey_stag, refinement_ratio);
                    },
                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                    {
                        warpx_interp(j, k, l, ez_aux, ez_fp, ez_c, Ez_stag, refinement_ratio);
                    });
                }
            }
            else // electrostatic
            {
                MultiFab::Copy(*Efield_aux[lev][0], *m_fields.get(FieldType::Efield_fp, Direction{0}, lev), 0, 0, Efield_aux[lev][0]->nComp(), Efield_aux[lev][0]->nGrowVect());
                MultiFab::Copy(*Efield_aux[lev][1], *m_fields.get(FieldType::Efield_fp, Direction{1}, lev), 0, 0, Efield_aux[lev][1]->nComp(), Efield_aux[lev][1]->nGrowVect());
                MultiFab::Copy(*Efield_aux[lev][2], *m_fields.get(FieldType::Efield_fp, Direction{2}, lev), 0, 0, Efield_aux[lev][2]->nComp(), Efield_aux[lev][2]->nGrowVect());
            }
        }
    }
}

void WarpX::UpdateCurrentNodalToStag (amrex::MultiFab& dst, amrex::MultiFab const& src)
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

        // Order of finite-order centering of currents
        const int cc_nox = WarpX::current_centering_nox;
        const int cc_noy = WarpX::current_centering_noy;
        const int cc_noz = WarpX::current_centering_noz;

        // Device vectors of stencil coefficients used for finite-order centering of currents
        amrex::Real const * stencil_coeffs_x = WarpX::device_current_centering_stencil_coeffs_x.data();
        amrex::Real const * stencil_coeffs_y = WarpX::device_current_centering_stencil_coeffs_y.data();
        amrex::Real const * stencil_coeffs_z = WarpX::device_current_centering_stencil_coeffs_z.data();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
        {
            warpx_interp(j, k, l, dst_arr, src_arr, dst_stag, src_stag, cc_nox, cc_noy, cc_noz,
                         stencil_coeffs_x, stencil_coeffs_y, stencil_coeffs_z);
        });
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
            pml_rz[lev]->FillBoundaryE(m_fields, patch_type, nodal_sync);
        }
#endif
    }

    // Fill guard cells in valid domain
    for (int i = 0; i < 3; ++i)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            ng.allLE(mf[i]->nGrowVect()),
            "Error: in FillBoundaryE, requested more guard cells than allocated");

        const amrex::IntVect nghost = (m_safe_guard_cells) ? mf[i]->nGrowVect() : ng;
        ablastr::utils::communication::FillBoundary(*mf[i], nghost, WarpX::do_single_precision_comms, period, nodal_sync);
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
            pml_rz[lev]->FillBoundaryB(m_fields, patch_type, nodal_sync);
        }
#endif
    }

    // Fill guard cells in valid domain
    for (int i = 0; i < 3; ++i)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            ng.allLE(mf[i]->nGrowVect()),
            "Error: in FillBoundaryB, requested more guard cells than allocated");

        const amrex::IntVect nghost = (m_safe_guard_cells) ? mf[i]->nGrowVect() : ng;
        ablastr::utils::communication::FillBoundary(*mf[i], nghost, WarpX::do_single_precision_comms, period, nodal_sync);
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

        ablastr::fields::MultiLevelVectorField Efield_avg_cp = m_fields.get_mr_levels_alldirs(FieldType::Efield_avg_cp, finest_level);

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
                ng.allLE(m_fields.get(FieldType::Bfield_fp, Direction{0}, lev)->nGrowVect()),
                "Error: in FillBoundaryB, requested more guard cells than allocated");
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

        ablastr::fields::MultiLevelVectorField Bfield_avg_cp = m_fields.get_mr_levels_alldirs(FieldType::Bfield_avg_cp, finest_level);

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

    WARPX_PROFILE("WarpX::SyncCurrent()");

    ablastr::fields::MultiLevelVectorField const& J_fp = m_fields.get_mr_levels_alldirs(current_fp_string, finest_level);

    // If warpx.do_current_centering = 1, center currents from nodal grid to staggered grid
    if (do_current_centering)
    {
        ablastr::fields::MultiLevelVectorField const& J_fp_nodal = m_fields.get_mr_levels_alldirs(FieldType::current_fp_nodal, finest_level+1);

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(finest_level <= 1,
                                         "warpx.do_current_centering=1 not supported with more than one fine levels");
        for (int lev = 0; lev <= finest_level; lev++)
        {
            WarpX::UpdateCurrentNodalToStag(*J_fp[lev][Direction{0}], *J_fp_nodal[lev][Direction{0}]);
            WarpX::UpdateCurrentNodalToStag(*J_fp[lev][Direction{1}], *J_fp_nodal[lev][Direction{1}]);
            WarpX::UpdateCurrentNodalToStag(*J_fp[lev][Direction{2}], *J_fp_nodal[lev][Direction{2}]);
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
                ablastr::fields::MultiLevelVectorField const& J_cp = m_fields.get_mr_levels_alldirs(FieldType::current_cp, finest_level);
                if (use_filter)
                {
                    ApplyFilterMF(J_cp, lev+1, idim);
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
                ablastr::fields::MultiLevelVectorField const& J_cp = m_fields.get_mr_levels_alldirs(FieldType::current_cp, finest_level);
                J_cp[lev][Direction{idim}]->setVal(0.0);
                ablastr::coarsen::average::Coarsen(*J_cp[lev][Direction{idim}],
                                                   *J_fp[lev][Direction{idim}],
                                                   refRatio(lev-1));
                if (m_fields.has(FieldType::current_buf, Direction{idim}, lev))
                {
                    ablastr::fields::MultiLevelVectorField const& J_buffer = m_fields.get_mr_levels_alldirs(FieldType::current_buf, finest_level);

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
                ApplyFilterMF(J_fp, lev, idim);
            }
            SumBoundaryJ(J_fp, lev, idim, period);
        }
    }
}

void
WarpX::SyncRho () {
    const ablastr::fields::MultiLevelScalarField rho_fp = m_fields.has(FieldType::rho_fp, 0) ?
        m_fields.get_mr_levels(FieldType::rho_fp, finest_level) :
        ablastr::fields::MultiLevelScalarField{static_cast<size_t>(finest_level+1)};
    const ablastr::fields::MultiLevelScalarField rho_cp = m_fields.has(FieldType::rho_cp, 1) ?
        m_fields.get_mr_levels(FieldType::rho_cp, finest_level) :
        ablastr::fields::MultiLevelScalarField{static_cast<size_t>(finest_level+1)};
    const ablastr::fields::MultiLevelScalarField rho_buf = m_fields.has(FieldType::rho_buf, 1) ?
        m_fields.get_mr_levels(FieldType::rho_buf, finest_level) :
        ablastr::fields::MultiLevelScalarField{static_cast<size_t>(finest_level+1)};

    SyncRho(rho_fp, rho_cp, rho_buf);
}

void
WarpX::SyncRho (
    const ablastr::fields::MultiLevelScalarField& charge_fp,
    const ablastr::fields::MultiLevelScalarField& charge_cp,
    ablastr::fields::MultiLevelScalarField const & charge_buffer)
{
    WARPX_PROFILE("WarpX::SyncRho()");

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
        ng_depos_J[0] += WarpX::current_centering_noz / 2;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        ng_depos_J[0] += WarpX::current_centering_nox / 2;
        ng_depos_J[1] += WarpX::current_centering_noz / 2;
#elif defined(WARPX_DIM_3D)
        ng_depos_J[0] += WarpX::current_centering_nox / 2;
        ng_depos_J[1] += WarpX::current_centering_noy / 2;
        ng_depos_J[2] += WarpX::current_centering_noz / 2;
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

/* /brief Update the currents of `lev` by adding the currents from particles
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
        ApplyFilterMF(J_fp, lev);
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
                ApplyFilterMF(J_cp, lev+1, idim);
                ApplyFilterMF(J_buffer, lev+1, idim);

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
                ApplyFilterMF(J_cp, lev+1, idim);

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
        bilinear_filter.ApplyStencil(rf, rho, glev, icomp, 0, ncomp);
        WarpXSumGuardCells(rho, rf, period, ng_depos_rho, icomp, ncomp );
    } else {
        ng_depos_rho.min(ng);
        WarpXSumGuardCells(rho, period, ng_depos_rho, icomp, ncomp);
    }
}

/* /brief Update the charge density of `lev` by adding the charge density from particles
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

void WarpX::NodalSyncJ (
    const ablastr::fields::MultiLevelVectorField& J_fp,
    const ablastr::fields::MultiLevelVectorField& J_cp,
    const int lev,
    PatchType patch_type)
{
    if (!override_sync_intervals.contains(istep[0])) { return; }

    if (patch_type == PatchType::fine)
    {
        const amrex::Periodicity& period = Geom(lev).periodicity();
        ablastr::utils::communication::OverrideSync(*J_fp[lev][0], WarpX::do_single_precision_comms, period);
        ablastr::utils::communication::OverrideSync(*J_fp[lev][1], WarpX::do_single_precision_comms, period);
        ablastr::utils::communication::OverrideSync(*J_fp[lev][2], WarpX::do_single_precision_comms, period);
    }
    else if (patch_type == PatchType::coarse)
    {
        const amrex::Periodicity& cperiod = Geom(lev-1).periodicity();
        ablastr::utils::communication::OverrideSync(*J_cp[lev][0], WarpX::do_single_precision_comms, cperiod);
        ablastr::utils::communication::OverrideSync(*J_cp[lev][1], WarpX::do_single_precision_comms, cperiod);
        ablastr::utils::communication::OverrideSync(*J_cp[lev][2], WarpX::do_single_precision_comms, cperiod);
    }
}

void WarpX::NodalSyncRho (
    const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_fp,
    const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_cp,
    const int lev,
    PatchType patch_type,
    const int icomp,
    const int ncomp)
{
    if (!override_sync_intervals.contains(istep[0])) { return; }

    if (patch_type == PatchType::fine && charge_fp[lev])
    {
        const amrex::Periodicity& period = Geom(lev).periodicity();
        MultiFab rhof(*charge_fp[lev], amrex::make_alias, icomp, ncomp);
        ablastr::utils::communication::OverrideSync(rhof, WarpX::do_single_precision_comms, period);
    }
    else if (patch_type == PatchType::coarse && charge_cp[lev])
    {
        const amrex::Periodicity& cperiod = Geom(lev-1).periodicity();
        MultiFab rhoc(*charge_cp[lev], amrex::make_alias, icomp, ncomp);
        ablastr::utils::communication::OverrideSync(rhoc, WarpX::do_single_precision_comms, cperiod);
    }
}
