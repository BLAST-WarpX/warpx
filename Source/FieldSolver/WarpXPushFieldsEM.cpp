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
#include "Evolve/WarpXDtType.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#if defined(WARPX_USE_FFT)
#   include "FieldSolver/SpectralSolver/SpectralFieldData.H"
#   ifdef WARPX_DIM_RZ
#       include "FieldSolver/SpectralSolver/SpectralSolverRZ.H"
#       include "BoundaryConditions/PML_RZ.H"
#   else
#       include "FieldSolver/SpectralSolver/SpectralSolver.H"
#   endif
#endif
#include "Python/callbacks.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpXPushFieldsEM_K.H"
#include "WarpX_FDTD.H"

#include <AMReX.H>
#ifdef AMREX_USE_SENSEI_INSITU
#   include <AMReX_AmrMeshInSituBridge.H>
#endif
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_MFIter.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <array>
#include <cmath>
#include <memory>

using namespace amrex;

#ifdef WARPX_USE_FFT
namespace {

    void ForwardTransformVect (
        const int lev,
#ifdef WARPX_DIM_RZ
        SpectralSolverRZ& solver,
#else
        SpectralSolver& solver,
#endif
        const std::array<std::unique_ptr<amrex::MultiFab>,3>& vector_field,
        const int compx, const int compy, const int compz)
    {
#ifdef WARPX_DIM_RZ
        solver.ForwardTransform(lev, *vector_field[0], compx, *vector_field[1], compy);
        solver.ForwardTransform(lev, *vector_field[2], compz);
#else
        solver.ForwardTransform(lev, *vector_field[0], compx);
        solver.ForwardTransform(lev, *vector_field[1], compy);
        solver.ForwardTransform(lev, *vector_field[2], compz);
#endif
    }

    void BackwardTransformVect (
        const int lev,
#ifdef WARPX_DIM_RZ
        SpectralSolverRZ& solver,
#else
        SpectralSolver& solver,
#endif
        const std::array<std::unique_ptr<amrex::MultiFab>,3>& vector_field,
        const int compx, const int compy, const int compz,
        const amrex::IntVect& fill_guards)
    {
#ifdef WARPX_DIM_RZ
        amrex::ignore_unused(fill_guards);
        solver.BackwardTransform(lev, *vector_field[0], compx, *vector_field[1], compy);
        solver.BackwardTransform(lev, *vector_field[2], compz);
#else
        solver.BackwardTransform(lev, *vector_field[0], compx, fill_guards);
        solver.BackwardTransform(lev, *vector_field[1], compy, fill_guards);
        solver.BackwardTransform(lev, *vector_field[2], compz, fill_guards);
#endif
    }
}

void WarpX::PSATDForwardTransformEB (
    const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& E_fp,
    const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& B_fp,
    const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& E_cp,
    const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& B_cp)
{
    const SpectralFieldIndex& Idx = spectral_solver_fp[0]->m_spectral_index;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        ForwardTransformVect(lev, *spectral_solver_fp[lev], E_fp[lev], Idx.Ex, Idx.Ey, Idx.Ez);
        ForwardTransformVect(lev, *spectral_solver_fp[lev], B_fp[lev], Idx.Bx, Idx.By, Idx.Bz);

        if (spectral_solver_cp[lev])
        {
            ForwardTransformVect(lev, *spectral_solver_cp[lev], E_cp[lev], Idx.Ex, Idx.Ey, Idx.Ez);
            ForwardTransformVect(lev, *spectral_solver_cp[lev], B_cp[lev], Idx.Bx, Idx.By, Idx.Bz);
        }
    }
}

void WarpX::PSATDBackwardTransformEB (
    const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& E_fp,
    const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& B_fp,
    const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& E_cp,
    const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& B_cp)
{
    const SpectralFieldIndex& Idx = spectral_solver_fp[0]->m_spectral_index;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        BackwardTransformVect(lev, *spectral_solver_fp[lev], E_fp[lev],
                              Idx.Ex, Idx.Ey, Idx.Ez, m_fill_guards_fields);
        BackwardTransformVect(lev, *spectral_solver_fp[lev], B_fp[lev],
                              Idx.Bx, Idx.By, Idx.Bz, m_fill_guards_fields);

        if (spectral_solver_cp[lev])
        {
            BackwardTransformVect(lev, *spectral_solver_cp[lev], E_cp[lev],
                                  Idx.Ex, Idx.Ey, Idx.Ez, m_fill_guards_fields);
            BackwardTransformVect(lev, *spectral_solver_cp[lev], B_cp[lev],
                                  Idx.Bx, Idx.By, Idx.Bz, m_fill_guards_fields);
        }
    }

    // Damp the fields in the guard cells
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        DampFieldsInGuards(lev, E_fp[lev], B_fp[lev]);
    }
}

void WarpX::PSATDBackwardTransformEBavg (
    const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& E_avg_fp,
    const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& B_avg_fp,
    const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& E_avg_cp,
    const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& B_avg_cp)
{
    const SpectralFieldIndex& Idx = spectral_solver_fp[0]->m_spectral_index;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        BackwardTransformVect(lev, *spectral_solver_fp[lev], E_avg_fp[lev],
                              Idx.Ex_avg, Idx.Ey_avg, Idx.Ez_avg, m_fill_guards_fields);
        BackwardTransformVect(lev, *spectral_solver_fp[lev], B_avg_fp[lev],
                              Idx.Bx_avg, Idx.By_avg, Idx.Bz_avg, m_fill_guards_fields);

        if (spectral_solver_cp[lev])
        {
            BackwardTransformVect(lev, *spectral_solver_cp[lev], E_avg_cp[lev],
                                  Idx.Ex_avg, Idx.Ey_avg, Idx.Ez_avg, m_fill_guards_fields);
            BackwardTransformVect(lev, *spectral_solver_cp[lev], B_avg_cp[lev],
                                  Idx.Bx_avg, Idx.By_avg, Idx.Bz_avg, m_fill_guards_fields);
        }
    }
}

void
WarpX::PSATDForwardTransformF ()
{
    const SpectralFieldIndex& Idx = spectral_solver_fp[0]->m_spectral_index;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (F_fp[lev]) { spectral_solver_fp[lev]->ForwardTransform(lev, *F_fp[lev], Idx.F); }

        if (spectral_solver_cp[lev])
        {
            if (F_cp[lev]) { spectral_solver_cp[lev]->ForwardTransform(lev, *F_cp[lev], Idx.F); }
        }
    }
}

void
WarpX::PSATDBackwardTransformF ()
{
    const SpectralFieldIndex& Idx = spectral_solver_fp[0]->m_spectral_index;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
#ifdef WARPX_DIM_RZ
        if (F_fp[lev]) { spectral_solver_fp[lev]->BackwardTransform(lev, *F_fp[lev], Idx.F); }
#else
        if (F_fp[lev]) { spectral_solver_fp[lev]->BackwardTransform(lev, *F_fp[lev], Idx.F, m_fill_guards_fields); }
#endif

        if (spectral_solver_cp[lev])
        {
#ifdef WARPX_DIM_RZ
            if (F_cp[lev]) { spectral_solver_cp[lev]->BackwardTransform(lev, *F_cp[lev], Idx.F); }
#else
            if (F_cp[lev]) { spectral_solver_cp[lev]->BackwardTransform(lev, *F_cp[lev], Idx.F, m_fill_guards_fields); }
#endif
        }
    }

    // Damp the field in the guard cells
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        DampFieldsInGuards(lev, F_fp[lev]);
    }
}

void
WarpX::PSATDForwardTransformG ()
{
    const SpectralFieldIndex& Idx = spectral_solver_fp[0]->m_spectral_index;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (G_fp[lev]) { spectral_solver_fp[lev]->ForwardTransform(lev, *G_fp[lev], Idx.G); }

        if (spectral_solver_cp[lev])
        {
            if (G_cp[lev]) { spectral_solver_cp[lev]->ForwardTransform(lev, *G_cp[lev], Idx.G); }
        }
    }
}

void
WarpX::PSATDBackwardTransformG ()
{
    const SpectralFieldIndex& Idx = spectral_solver_fp[0]->m_spectral_index;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
#ifdef WARPX_DIM_RZ
        if (G_fp[lev]) { spectral_solver_fp[lev]->BackwardTransform(lev, *G_fp[lev], Idx.G); }
#else
        if (G_fp[lev]) { spectral_solver_fp[lev]->BackwardTransform(lev, *G_fp[lev], Idx.G, m_fill_guards_fields); }
#endif

        if (spectral_solver_cp[lev])
        {
#ifdef WARPX_DIM_RZ
            if (G_cp[lev]) { spectral_solver_cp[lev]->BackwardTransform(lev, *G_cp[lev], Idx.G); }
#else
            if (G_cp[lev]) { spectral_solver_cp[lev]->BackwardTransform(lev, *G_cp[lev], Idx.G, m_fill_guards_fields); }
#endif
        }
    }

    // Damp the field in the guard cells
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        DampFieldsInGuards(lev, G_fp[lev]);
    }
}

void WarpX::PSATDForwardTransformJ (
    const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_fp,
    const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_cp,
    const bool apply_kspace_filter)
{
    SpectralFieldIndex Idx;
    int idx_jx, idx_jy, idx_jz;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        Idx = spectral_solver_fp[lev]->m_spectral_index;

        idx_jx = (J_in_time == JInTime::Linear) ? static_cast<int>(Idx.Jx_new) : static_cast<int>(Idx.Jx_mid);
        idx_jy = (J_in_time == JInTime::Linear) ? static_cast<int>(Idx.Jy_new) : static_cast<int>(Idx.Jy_mid);
        idx_jz = (J_in_time == JInTime::Linear) ? static_cast<int>(Idx.Jz_new) : static_cast<int>(Idx.Jz_mid);

        ForwardTransformVect(lev, *spectral_solver_fp[lev], J_fp[lev], idx_jx, idx_jy, idx_jz);

        if (spectral_solver_cp[lev])
        {
            Idx = spectral_solver_cp[lev]->m_spectral_index;

            idx_jx = (J_in_time == JInTime::Linear) ? static_cast<int>(Idx.Jx_new) : static_cast<int>(Idx.Jx_mid);
            idx_jy = (J_in_time == JInTime::Linear) ? static_cast<int>(Idx.Jy_new) : static_cast<int>(Idx.Jy_mid);
            idx_jz = (J_in_time == JInTime::Linear) ? static_cast<int>(Idx.Jz_new) : static_cast<int>(Idx.Jz_mid);

            ForwardTransformVect(lev, *spectral_solver_cp[lev], J_cp[lev], idx_jx, idx_jy, idx_jz);
        }
    }

#ifdef WARPX_DIM_RZ
    // Apply filter in k space if needed
    if (use_kspace_filter && apply_kspace_filter)
    {
        for (int lev = 0; lev <= finest_level; ++lev)
        {
            Idx = spectral_solver_fp[lev]->m_spectral_index;

            idx_jx = (J_in_time == JInTime::Linear) ? static_cast<int>(Idx.Jx_new) : static_cast<int>(Idx.Jx_mid);
            idx_jy = (J_in_time == JInTime::Linear) ? static_cast<int>(Idx.Jy_new) : static_cast<int>(Idx.Jy_mid);
            idx_jz = (J_in_time == JInTime::Linear) ? static_cast<int>(Idx.Jz_new) : static_cast<int>(Idx.Jz_mid);

            spectral_solver_fp[lev]->ApplyFilter(lev, idx_jx, idx_jy, idx_jz);

            if (spectral_solver_cp[lev])
            {
                Idx = spectral_solver_cp[lev]->m_spectral_index;

                idx_jx = (J_in_time == JInTime::Linear) ? static_cast<int>(Idx.Jx_new) : static_cast<int>(Idx.Jx_mid);
                idx_jy = (J_in_time == JInTime::Linear) ? static_cast<int>(Idx.Jy_new) : static_cast<int>(Idx.Jy_mid);
                idx_jz = (J_in_time == JInTime::Linear) ? static_cast<int>(Idx.Jz_new) : static_cast<int>(Idx.Jz_mid);

                spectral_solver_cp[lev]->ApplyFilter(lev, idx_jx, idx_jy, idx_jz);
            }
        }
    }
#else
    amrex::ignore_unused(apply_kspace_filter);
#endif
}

void WarpX::PSATDBackwardTransformJ (
    const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_fp,
    const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_cp)
{
    SpectralFieldIndex Idx;
    int idx_jx, idx_jy, idx_jz;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        Idx = spectral_solver_fp[lev]->m_spectral_index;

        // Note that these backward FFTs are currently called only
        // with algorithms that do not support J linear in time
        idx_jx = static_cast<int>(Idx.Jx_mid);
        idx_jy = static_cast<int>(Idx.Jy_mid);
        idx_jz = static_cast<int>(Idx.Jz_mid);

        BackwardTransformVect(lev, *spectral_solver_fp[lev], J_fp[lev],
                              idx_jx, idx_jy, idx_jz, m_fill_guards_current);

        if (spectral_solver_cp[lev])
        {
            Idx = spectral_solver_cp[lev]->m_spectral_index;

            // Note that these backward FFTs are currently called only
            // with algorithms that do not support J linear in time
            idx_jx = static_cast<int>(Idx.Jx_mid);
            idx_jy = static_cast<int>(Idx.Jy_mid);
            idx_jz = static_cast<int>(Idx.Jz_mid);

            BackwardTransformVect(lev, *spectral_solver_cp[lev], J_cp[lev],
                                  idx_jx, idx_jy, idx_jz, m_fill_guards_current);
        }
    }
}

void WarpX::PSATDForwardTransformRho (
    const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_fp,
    const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_cp,
    const int icomp, const int dcomp, const bool apply_kspace_filter)
{
    if (charge_fp[0] == nullptr) { return; }

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (charge_fp[lev]) { spectral_solver_fp[lev]->ForwardTransform(lev, *charge_fp[lev], dcomp, icomp); }

        if (spectral_solver_cp[lev])
        {
            if (charge_cp[lev]) { spectral_solver_cp[lev]->ForwardTransform(lev, *charge_cp[lev], dcomp, icomp); }
        }
    }

#ifdef WARPX_DIM_RZ
    // Apply filter in k space if needed
    if (use_kspace_filter && apply_kspace_filter)
    {
        for (int lev = 0; lev <= finest_level; ++lev)
        {
            spectral_solver_fp[lev]->ApplyFilter(lev, dcomp);

            if (spectral_solver_cp[lev])
            {
                spectral_solver_cp[lev]->ApplyFilter(lev, dcomp);
            }
        }
    }
#else
    amrex::ignore_unused(apply_kspace_filter);
#endif
}

void WarpX::PSATDCurrentCorrection ()
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        spectral_solver_fp[lev]->CurrentCorrection();

        if (spectral_solver_cp[lev])
        {
            spectral_solver_cp[lev]->CurrentCorrection();
        }
    }
}

void WarpX::PSATDVayDeposition ()
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        spectral_solver_fp[lev]->VayDeposition();

        if (spectral_solver_cp[lev])
        {
            spectral_solver_cp[lev]->VayDeposition();
        }
    }
}

void WarpX::PSATDSubtractCurrentPartialSumsAvg ()
{
    // Subtraction of cumulative sum for Vay deposition
    // implemented only in 2D and 3D Cartesian geometry
#if !defined (WARPX_DIM_1D_Z) && !defined (WARPX_DIM_RZ)

    // TODO Implementation with coarse patches
    // TODO Implementation with current centering

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        const std::array<amrex::Real,3>& dx = WarpX::CellSize(lev);

        amrex::MultiFab const& Dx = *current_fp_vay[lev][0];
        amrex::MultiFab const& Dy = *current_fp_vay[lev][1];
        amrex::MultiFab const& Dz = *current_fp_vay[lev][2];

#if defined (WARPX_DIM_XZ)
        amrex::ignore_unused(Dy);
#endif

    amrex::MultiFab& Jx = *current_fp[lev][0];


#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        // Subtract average of cumulative sum from Jx
        for (amrex::MFIter mfi(Jx); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.fabbox();

            amrex::Array4<amrex::Real> const& Jx_arr = Jx.array(mfi);
            amrex::Array4<amrex::Real const> const& Dx_arr = Dx.const_array(mfi);

            const amrex::Dim3 lo = amrex::lbound(bx);
            const amrex::Dim3 hi = amrex::ubound(bx);
            const int nx = hi.x - lo.x + 1;
            const amrex::Real facx = dx[0] / static_cast<amrex::Real>(nx);

            // Subtract average of cumulative sum along x only
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                for (int ii = lo.x; ii <= hi.x; ++ii)
                {
                    Jx_arr(i,j,k) -= (nx-ii) * Dx_arr(ii,j,k) * facx;
                }
            });
        }

#if defined (WARPX_DIM_3D)
        // Subtract average of cumulative sum from Jy
        amrex::MultiFab& Jy = *current_fp[lev][1];
        for (amrex::MFIter mfi(Jy); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.fabbox();

            amrex::Array4<amrex::Real> const& Jy_arr = Jy.array(mfi);
            amrex::Array4<amrex::Real const> const& Dy_arr = Dy.const_array(mfi);

            const amrex::Dim3 lo = amrex::lbound(bx);
            const amrex::Dim3 hi = amrex::ubound(bx);
            const int ny = hi.y - lo.y + 1;
            const amrex::Real facy = dx[1] / static_cast<amrex::Real>(ny);

            // Subtract average of cumulative sum along y only
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                for (int jj = lo.y; jj <= hi.y; ++jj)
                {
                    Jy_arr(i,j,k) -= (ny-jj) * Dy_arr(i,jj,k) * facy;
                }
            });
        }
#endif

        // Subtract average of cumulative sum from Jz
        amrex::MultiFab& Jz = *current_fp[lev][2];
        for (amrex::MFIter mfi(Jz); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.fabbox();

            amrex::Array4<amrex::Real> const& Jz_arr = Jz.array(mfi);
            amrex::Array4<amrex::Real const> const& Dz_arr = Dz.const_array(mfi);

            const amrex::Dim3 lo = amrex::lbound(bx);
            const amrex::Dim3 hi = amrex::ubound(bx);
#if defined (WARPX_DIM_XZ)
            const int nz = hi.y - lo.y + 1;
#elif defined (WARPX_DIM_3D)
            const int nz = hi.z - lo.z + 1;
#endif
            const amrex::Real facz = dx[2] / static_cast<amrex::Real>(nz);

            // Subtract average of cumulative sum along z only
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
#if defined (WARPX_DIM_XZ)
                // z direction is in the second component
                for (int jj = lo.y; jj <= hi.y; ++jj)
                {
                    Jz_arr(i,j,k) -= (nz-jj) * Dz_arr(i,jj,k) * facz;
                }
#elif defined (WARPX_DIM_3D)
                // z direction is in the third component
                for (int kk = lo.z; kk <= hi.z; ++kk)
                {
                    Jz_arr(i,j,k) -= (nz-kk) * Dz_arr(i,j,kk) * facz;
                }
#endif
            });
        }
    }
#endif
}

void
WarpX::PSATDPushSpectralFields ()
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        spectral_solver_fp[lev]->pushSpectralFields();

        if (spectral_solver_cp[lev])
        {
            spectral_solver_cp[lev]->pushSpectralFields();
        }
    }
}

void
WarpX::PSATDMoveRhoNewToRhoOld ()
{
    const SpectralFieldIndex& Idx = spectral_solver_fp[0]->m_spectral_index;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        spectral_solver_fp[lev]->CopySpectralDataComp(Idx.rho_new, Idx.rho_old);

        if (spectral_solver_cp[lev])
        {
            spectral_solver_cp[lev]->CopySpectralDataComp(Idx.rho_new, Idx.rho_old);
        }
    }
}

void
WarpX::PSATDMoveJNewToJOld ()
{
    const SpectralFieldIndex& Idx = spectral_solver_fp[0]->m_spectral_index;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        spectral_solver_fp[lev]->CopySpectralDataComp(Idx.Jx_new, Idx.Jx_old);
        spectral_solver_fp[lev]->CopySpectralDataComp(Idx.Jy_new, Idx.Jy_old);
        spectral_solver_fp[lev]->CopySpectralDataComp(Idx.Jz_new, Idx.Jz_old);

        if (spectral_solver_cp[lev])
        {
            spectral_solver_cp[lev]->CopySpectralDataComp(Idx.Jx_new, Idx.Jx_old);
            spectral_solver_cp[lev]->CopySpectralDataComp(Idx.Jy_new, Idx.Jy_old);
            spectral_solver_cp[lev]->CopySpectralDataComp(Idx.Jz_new, Idx.Jz_old);
        }
    }
}

void
WarpX::PSATDEraseAverageFields ()
{
    const SpectralFieldIndex& Idx = spectral_solver_fp[0]->m_spectral_index;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        spectral_solver_fp[lev]->ZeroOutDataComp(Idx.Ex_avg);
        spectral_solver_fp[lev]->ZeroOutDataComp(Idx.Ey_avg);
        spectral_solver_fp[lev]->ZeroOutDataComp(Idx.Ez_avg);
        spectral_solver_fp[lev]->ZeroOutDataComp(Idx.Bx_avg);
        spectral_solver_fp[lev]->ZeroOutDataComp(Idx.By_avg);
        spectral_solver_fp[lev]->ZeroOutDataComp(Idx.Bz_avg);

        if (spectral_solver_cp[lev])
        {
            spectral_solver_cp[lev]->ZeroOutDataComp(Idx.Ex_avg);
            spectral_solver_cp[lev]->ZeroOutDataComp(Idx.Ey_avg);
            spectral_solver_cp[lev]->ZeroOutDataComp(Idx.Ez_avg);
            spectral_solver_cp[lev]->ZeroOutDataComp(Idx.Bx_avg);
            spectral_solver_cp[lev]->ZeroOutDataComp(Idx.By_avg);
            spectral_solver_cp[lev]->ZeroOutDataComp(Idx.Bz_avg);
        }
    }
}

void
WarpX::PSATDScaleAverageFields (const amrex::Real scale_factor)
{
    const SpectralFieldIndex& Idx = spectral_solver_fp[0]->m_spectral_index;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        spectral_solver_fp[lev]->ScaleDataComp(Idx.Ex_avg, scale_factor);
        spectral_solver_fp[lev]->ScaleDataComp(Idx.Ey_avg, scale_factor);
        spectral_solver_fp[lev]->ScaleDataComp(Idx.Ez_avg, scale_factor);
        spectral_solver_fp[lev]->ScaleDataComp(Idx.Bx_avg, scale_factor);
        spectral_solver_fp[lev]->ScaleDataComp(Idx.By_avg, scale_factor);
        spectral_solver_fp[lev]->ScaleDataComp(Idx.Bz_avg, scale_factor);

        if (spectral_solver_cp[lev])
        {
            spectral_solver_cp[lev]->ScaleDataComp(Idx.Ex_avg, scale_factor);
            spectral_solver_cp[lev]->ScaleDataComp(Idx.Ey_avg, scale_factor);
            spectral_solver_cp[lev]->ScaleDataComp(Idx.Ez_avg, scale_factor);
            spectral_solver_cp[lev]->ScaleDataComp(Idx.Bx_avg, scale_factor);
            spectral_solver_cp[lev]->ScaleDataComp(Idx.By_avg, scale_factor);
            spectral_solver_cp[lev]->ScaleDataComp(Idx.Bz_avg, scale_factor);
        }
    }
}
#endif // WARPX_USE_FFT

void
WarpX::PushPSATD ()
{
#ifndef WARPX_USE_FFT
    WARPX_ABORT_WITH_MESSAGE(
        "PushFieldsEM: PSATD solver selected but not built");
#else

    const int rho_old = spectral_solver_fp[0]->m_spectral_index.rho_old;
    const int rho_new = spectral_solver_fp[0]->m_spectral_index.rho_new;

    if (fft_periodic_single_box)
    {
        if (current_correction)
        {
            // FFT of J and rho
            PSATDForwardTransformJ(current_fp, current_cp);
            PSATDForwardTransformRho(rho_fp, rho_cp, 0, rho_old);
            PSATDForwardTransformRho(rho_fp, rho_cp, 1, rho_new);

            // Correct J in k-space
            PSATDCurrentCorrection();

            // Inverse FFT of J
            PSATDBackwardTransformJ(current_fp, current_cp);
        }
        else if (current_deposition_algo == CurrentDepositionAlgo::Vay)
        {
            // FFT of D and rho (if used)
            // TODO Replace current_cp with current_cp_vay once Vay deposition is implemented with MR
            PSATDForwardTransformJ(current_fp_vay, current_cp);
            PSATDForwardTransformRho(rho_fp, rho_cp, 0, rho_old);
            PSATDForwardTransformRho(rho_fp, rho_cp, 1, rho_new);

            // Compute J from D in k-space
            PSATDVayDeposition();

            // Inverse FFT of J, subtract cumulative sums of D
            PSATDBackwardTransformJ(current_fp, current_cp);
            // TODO Cumulative sums need to be fixed with periodic single box
            PSATDSubtractCurrentPartialSumsAvg();

            // FFT of J after subtraction of cumulative sums
            PSATDForwardTransformJ(current_fp, current_cp);
        }
        else // no current correction, no Vay deposition
        {
            // FFT of J and rho (if used)
            PSATDForwardTransformJ(current_fp, current_cp);
            PSATDForwardTransformRho(rho_fp, rho_cp, 0, rho_old);
            PSATDForwardTransformRho(rho_fp, rho_cp, 1, rho_new);
        }
    }
    else // no periodic single box
    {
        if (current_correction)
        {
            // FFT of J and rho
#ifdef WARPX_DIM_RZ
            // In RZ geometry, do not apply filtering here, since it is
            // applied in the subsequent calls to these functions (below)
            const bool apply_kspace_filter = false;
            PSATDForwardTransformJ(current_fp, current_cp, apply_kspace_filter);
            PSATDForwardTransformRho(rho_fp, rho_cp, 0, rho_old, apply_kspace_filter);
            PSATDForwardTransformRho(rho_fp, rho_cp, 1, rho_new, apply_kspace_filter);
#else
            PSATDForwardTransformJ(current_fp, current_cp);
            PSATDForwardTransformRho(rho_fp, rho_cp, 0, rho_old);
            PSATDForwardTransformRho(rho_fp, rho_cp, 1, rho_new);
#endif

            // Correct J in k-space
            PSATDCurrentCorrection();

            // Inverse FFT of J
            PSATDBackwardTransformJ(current_fp, current_cp);

            // Synchronize J and rho
            SyncCurrent(current_fp, current_cp, current_buf);
            SyncRho(rho_fp, rho_cp, charge_buf);
        }
        else if (current_deposition_algo == CurrentDepositionAlgo::Vay)
        {
            // FFT of D
            PSATDForwardTransformJ(current_fp_vay, current_cp);

            // Compute J from D in k-space
            PSATDVayDeposition();

            // Inverse FFT of J, subtract cumulative sums of D
            PSATDBackwardTransformJ(current_fp, current_cp);
            PSATDSubtractCurrentPartialSumsAvg();

            // Synchronize J and rho (if used).
            // Here we call SumBoundaryJ instead of SyncCurrent, because
            // filtering has been already applied to D in OneStep_nosub,
            // by calling SyncCurrentAndRho (see Evolve/WarpXEvolve.cpp).
            // TODO This works only without mesh refinement
            const int lev = 0;
            SumBoundaryJ(current_fp, lev, Geom(lev).periodicity());
            SyncRho(rho_fp, rho_cp, charge_buf);
        }

        // FFT of J and rho (if used)
        PSATDForwardTransformJ(current_fp, current_cp);
        PSATDForwardTransformRho(rho_fp, rho_cp, 0, rho_old);
        PSATDForwardTransformRho(rho_fp, rho_cp, 1, rho_new);
    }

    // FFT of E and B
    PSATDForwardTransformEB(Efield_fp, Bfield_fp, Efield_cp, Bfield_cp);

#ifdef WARPX_DIM_RZ
    if (pml_rz[0]) { pml_rz[0]->PushPSATD(0); }
#endif

    // FFT of F and G
    if (WarpX::do_dive_cleaning) { PSATDForwardTransformF(); }
    if (WarpX::do_divb_cleaning) { PSATDForwardTransformG(); }

    // Update E, B, F, and G in k-space
    PSATDPushSpectralFields();

    // Inverse FFT of E, B, F, and G
    PSATDBackwardTransformEB(Efield_fp, Bfield_fp, Efield_cp, Bfield_cp);
    if (WarpX::fft_do_time_averaging) {
        PSATDBackwardTransformEBavg(Efield_avg_fp, Bfield_avg_fp, Efield_avg_cp, Bfield_avg_cp);
    }
    if (WarpX::do_dive_cleaning) { PSATDBackwardTransformF(); }
    if (WarpX::do_divb_cleaning) { PSATDBackwardTransformG(); }

    // Evolve the fields in the PML boxes
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (pml[lev] && pml[lev]->ok())
        {
            pml[lev]->PushPSATD(lev);
        }
        ApplyEfieldBoundary(lev, PatchType::fine);
        if (lev > 0) { ApplyEfieldBoundary(lev, PatchType::coarse); }
        ApplyBfieldBoundary(lev, PatchType::fine, DtType::FirstHalf);
        if (lev > 0) { ApplyBfieldBoundary(lev, PatchType::coarse, DtType::FirstHalf); }
    }
#endif
}

void
WarpX::EvolveB (amrex::Real a_dt, DtType a_dt_type)
{
    for (int lev = 0; lev <= finest_level; ++lev) {
        EvolveB(lev, a_dt, a_dt_type);
    }

    // Allow execution of Python callback after B-field push
    ExecutePythonCallback("afterBpush");
}

void
WarpX::EvolveB (int lev, amrex::Real a_dt, DtType a_dt_type)
{
    WARPX_PROFILE("WarpX::EvolveB()");
    EvolveB(lev, PatchType::fine, a_dt, a_dt_type);
    if (lev > 0)
    {
        EvolveB(lev, PatchType::coarse, a_dt, a_dt_type);
    }
}

void
WarpX::EvolveB (int lev, PatchType patch_type, amrex::Real a_dt, DtType a_dt_type)
{

    // Evolve B field in regular cells
    if (patch_type == PatchType::fine) {
        m_fdtd_solver_fp[lev]->EvolveB(Bfield_fp[lev], Efield_fp[lev], G_fp[lev],
                                       m_face_areas[lev], m_area_mod[lev], ECTRhofield[lev], Venl[lev],
                                       m_flag_info_face[lev], m_borrowing[lev], lev, a_dt);
    } else {
        m_fdtd_solver_cp[lev]->EvolveB(Bfield_cp[lev], Efield_cp[lev], G_cp[lev],
                                       m_face_areas[lev], m_area_mod[lev], ECTRhofield[lev], Venl[lev],
                                       m_flag_info_face[lev], m_borrowing[lev], lev, a_dt);
    }

    // Evolve B field in PML cells
    if (do_pml && pml[lev]->ok()) {
        if (patch_type == PatchType::fine) {
            m_fdtd_solver_fp[lev]->EvolveBPML(
                pml[lev]->GetB_fp(), pml[lev]->GetE_fp(), a_dt, WarpX::do_dive_cleaning);
        } else {
            m_fdtd_solver_cp[lev]->EvolveBPML(
                pml[lev]->GetB_cp(), pml[lev]->GetE_cp(), a_dt, WarpX::do_dive_cleaning);
        }
    }

    ApplyBfieldBoundary(lev, patch_type, a_dt_type);
}


void
WarpX::EvolveE (amrex::Real a_dt)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        EvolveE(lev, a_dt);
    }

    // Allow execution of Python callback after E-field push
    ExecutePythonCallback("afterEpush");
}

void
WarpX::EvolveE (int lev, amrex::Real a_dt)
{
    WARPX_PROFILE("WarpX::EvolveE()");
    EvolveE(lev, PatchType::fine, a_dt);
    if (lev > 0)
    {
        EvolveE(lev, PatchType::coarse, a_dt);
    }
}

void
WarpX::EvolveE (int lev, PatchType patch_type, amrex::Real a_dt)
{
    // Evolve E field in regular cells
    if (patch_type == PatchType::fine) {
        m_fdtd_solver_fp[lev]->EvolveE(Efield_fp[lev], Bfield_fp[lev],
                                       current_fp[lev], m_edge_lengths[lev],
                                       m_face_areas[lev], ECTRhofield[lev],
                                       F_fp[lev], lev, a_dt );
    } else {
        m_fdtd_solver_cp[lev]->EvolveE(Efield_cp[lev], Bfield_cp[lev],
                                       current_cp[lev], m_edge_lengths[lev],
                                       m_face_areas[lev], ECTRhofield[lev],
                                       F_cp[lev], lev, a_dt );
    }

    // Evolve E field in PML cells
    if (do_pml && pml[lev]->ok()) {
        if (patch_type == PatchType::fine) {
            m_fdtd_solver_fp[lev]->EvolveEPML(
                pml[lev]->GetE_fp(), pml[lev]->GetB_fp(),
                pml[lev]->Getj_fp(), pml[lev]->Get_edge_lengths(),
                pml[lev]->GetF_fp(),
                pml[lev]->GetMultiSigmaBox_fp(),
                a_dt, pml_has_particles );
        } else {
            m_fdtd_solver_cp[lev]->EvolveEPML(
                pml[lev]->GetE_cp(), pml[lev]->GetB_cp(),
                pml[lev]->Getj_cp(), pml[lev]->Get_edge_lengths(),
                pml[lev]->GetF_cp(),
                pml[lev]->GetMultiSigmaBox_cp(),
                a_dt, pml_has_particles );
        }
    }

    ApplyEfieldBoundary(lev, patch_type);

    // ECTRhofield must be recomputed at the very end of the Efield update to ensure
    // that ECTRhofield is consistent with Efield
#ifdef AMREX_USE_EB
    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::ECT) {
        if (patch_type == PatchType::fine) {
            m_fdtd_solver_fp[lev]->EvolveECTRho(Efield_fp[lev], m_edge_lengths[lev],
                                                m_face_areas[lev], ECTRhofield[lev], lev);
        } else {
            m_fdtd_solver_cp[lev]->EvolveECTRho(Efield_cp[lev], m_edge_lengths[lev],
                                                m_face_areas[lev], ECTRhofield[lev], lev);
        }
    }
#endif
}


void
WarpX::EvolveF (amrex::Real a_dt, DtType a_dt_type)
{
    if (!do_dive_cleaning) { return; }

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        EvolveF(lev, a_dt, a_dt_type);
    }
}

void
WarpX::EvolveF (int lev, amrex::Real a_dt, DtType a_dt_type)
{
    if (!do_dive_cleaning) { return; }

    EvolveF(lev, PatchType::fine, a_dt, a_dt_type);
    if (lev > 0) { EvolveF(lev, PatchType::coarse, a_dt, a_dt_type); }
}

void
WarpX::EvolveF (int lev, PatchType patch_type, amrex::Real a_dt, DtType a_dt_type)
{
    if (!do_dive_cleaning) { return; }

    WARPX_PROFILE("WarpX::EvolveF()");

    const int rhocomp = (a_dt_type == DtType::FirstHalf) ? 0 : 1;

    // Evolve F field in regular cells
    if (patch_type == PatchType::fine) {
        m_fdtd_solver_fp[lev]->EvolveF( F_fp[lev], Efield_fp[lev],
                                        rho_fp[lev], rhocomp, a_dt );
    } else {
        m_fdtd_solver_cp[lev]->EvolveF( F_cp[lev], Efield_cp[lev],
                                        rho_cp[lev], rhocomp, a_dt );
    }

    // Evolve F field in PML cells
    if (do_pml && pml[lev]->ok()) {
        if (patch_type == PatchType::fine) {
            m_fdtd_solver_fp[lev]->EvolveFPML(
                pml[lev]->GetF_fp(), pml[lev]->GetE_fp(), a_dt );
        } else {
            m_fdtd_solver_cp[lev]->EvolveFPML(
                pml[lev]->GetF_cp(), pml[lev]->GetE_cp(), a_dt );
        }
    }
}

void
WarpX::EvolveG (amrex::Real a_dt, DtType a_dt_type)
{
    if (!do_divb_cleaning) { return; }

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        EvolveG(lev, a_dt, a_dt_type);
    }
}

void
WarpX::EvolveG (int lev, amrex::Real a_dt, DtType a_dt_type)
{
    if (!do_divb_cleaning) { return; }

    EvolveG(lev, PatchType::fine, a_dt, a_dt_type);

    if (lev > 0)
    {
        EvolveG(lev, PatchType::coarse, a_dt, a_dt_type);
    }
}

void
WarpX::EvolveG (int lev, PatchType patch_type, amrex::Real a_dt, DtType /*a_dt_type*/)
{
    if (!do_divb_cleaning) { return; }

    WARPX_PROFILE("WarpX::EvolveG()");

    // Evolve G field in regular cells
    if (patch_type == PatchType::fine)
    {
        m_fdtd_solver_fp[lev]->EvolveG(G_fp[lev], Bfield_fp[lev], a_dt);
    }
    else // coarse patch
    {
        m_fdtd_solver_cp[lev]->EvolveG(G_cp[lev], Bfield_cp[lev], a_dt);
    }

    // TODO Evolution in PML cells will go here
}

void
WarpX::MacroscopicEvolveE (amrex::Real a_dt)
{
    for (int lev = 0; lev <= finest_level; ++lev ) {
        MacroscopicEvolveE(lev, a_dt);
    }
}

void
WarpX::MacroscopicEvolveE (int lev, amrex::Real a_dt) {

    WARPX_PROFILE("WarpX::MacroscopicEvolveE()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        lev == 0,
        "Macroscopic EvolveE is not implemented for lev>0, yet."
    );

    MacroscopicEvolveE(lev, PatchType::fine, a_dt);
}

void
WarpX::MacroscopicEvolveE (int lev, PatchType patch_type, amrex::Real a_dt) {

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        patch_type == PatchType::fine,
        "Macroscopic EvolveE is not implemented for lev>0, yet."
    );

    m_fdtd_solver_fp[lev]->MacroscopicEvolveE(
        Efield_fp[lev], Bfield_fp[lev],
        current_fp[lev], m_edge_lengths[lev],
        a_dt, m_macroscopic_properties);

    if (do_pml && pml[lev]->ok()) {
        if (patch_type == PatchType::fine) {
            m_fdtd_solver_fp[lev]->EvolveEPML(
                pml[lev]->GetE_fp(), pml[lev]->GetB_fp(),
                pml[lev]->Getj_fp(), pml[lev]->Get_edge_lengths(),
                pml[lev]->GetF_fp(),
                pml[lev]->GetMultiSigmaBox_fp(),
                a_dt, pml_has_particles );
        } else {
            m_fdtd_solver_cp[lev]->EvolveEPML(
                pml[lev]->GetE_cp(), pml[lev]->GetB_cp(),
                pml[lev]->Getj_cp(), pml[lev]->Get_edge_lengths(),
                pml[lev]->GetF_cp(),
                pml[lev]->GetMultiSigmaBox_cp(),
                a_dt, pml_has_particles );
        }
    }

    ApplyEfieldBoundary(lev, patch_type);
}

void
WarpX::DampFieldsInGuards(const int lev,
                          const std::array<std::unique_ptr<amrex::MultiFab>,3>& Efield,
                          const std::array<std::unique_ptr<amrex::MultiFab>,3>& Bfield) {

    // Loop over dimensions
    for (int dampdir = 0 ; dampdir < AMREX_SPACEDIM ; dampdir++)
    {

        // Loop over the lower and upper guards
        for (int iside = 0 ; iside < 2 ; iside++)
        {

            // Only apply to damped boundaries
            if (iside == 0 && WarpX::field_boundary_lo[dampdir] != FieldBoundaryType::Damped) { continue; }
            if (iside == 1 && WarpX::field_boundary_hi[dampdir] != FieldBoundaryType::Damped) { continue; }

            for ( amrex::MFIter mfi(*Efield[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
            {
                amrex::Array4<amrex::Real> const& Ex_arr = Efield[0]->array(mfi);
                amrex::Array4<amrex::Real> const& Ey_arr = Efield[1]->array(mfi);
                amrex::Array4<amrex::Real> const& Ez_arr = Efield[2]->array(mfi);
                amrex::Array4<amrex::Real> const& Bx_arr = Bfield[0]->array(mfi);
                amrex::Array4<amrex::Real> const& By_arr = Bfield[1]->array(mfi);
                amrex::Array4<amrex::Real> const& Bz_arr = Bfield[2]->array(mfi);

                // Get the tileboxes from Efield and Bfield so that they include the guard cells
                // and take the staggering of each MultiFab into account
                const amrex::Box tex = amrex::convert((*Efield[0])[mfi].box(), Efield[0]->ixType().toIntVect());
                const amrex::Box tey = amrex::convert((*Efield[1])[mfi].box(), Efield[1]->ixType().toIntVect());
                const amrex::Box tez = amrex::convert((*Efield[2])[mfi].box(), Efield[2]->ixType().toIntVect());
                const amrex::Box tbx = amrex::convert((*Bfield[0])[mfi].box(), Bfield[0]->ixType().toIntVect());
                const amrex::Box tby = amrex::convert((*Bfield[1])[mfi].box(), Bfield[1]->ixType().toIntVect());
                const amrex::Box tbz = amrex::convert((*Bfield[2])[mfi].box(), Bfield[2]->ixType().toIntVect());

                // Get smallEnd of tileboxes
                const int tex_smallEnd_d = tex.smallEnd(dampdir);
                const int tey_smallEnd_d = tey.smallEnd(dampdir);
                const int tez_smallEnd_d = tez.smallEnd(dampdir);
                const int tbx_smallEnd_d = tbx.smallEnd(dampdir);
                const int tby_smallEnd_d = tby.smallEnd(dampdir);
                const int tbz_smallEnd_d = tbz.smallEnd(dampdir);

                // Get bigEnd of tileboxes
                const int tex_bigEnd_d = tex.bigEnd(dampdir);
                const int tey_bigEnd_d = tey.bigEnd(dampdir);
                const int tez_bigEnd_d = tez.bigEnd(dampdir);
                const int tbx_bigEnd_d = tbx.bigEnd(dampdir);
                const int tby_bigEnd_d = tby.bigEnd(dampdir);
                const int tbz_bigEnd_d = tbz.bigEnd(dampdir);

                // Box for the whole simulation domain
                amrex::Box const& domain = Geom(lev).Domain();
                int const nn_domain = domain.bigEnd(dampdir);

                // Set the tileboxes so that they only cover the lower/upper half of the guard cells
                const amrex::Box tex_guard = constrain_tilebox_to_guards(tex, dampdir, iside, nn_domain, tex_smallEnd_d, tex_bigEnd_d);
                const amrex::Box tey_guard = constrain_tilebox_to_guards(tey, dampdir, iside, nn_domain, tey_smallEnd_d, tey_bigEnd_d);
                const amrex::Box tez_guard = constrain_tilebox_to_guards(tez, dampdir, iside, nn_domain, tez_smallEnd_d, tez_bigEnd_d);
                const amrex::Box tbx_guard = constrain_tilebox_to_guards(tbx, dampdir, iside, nn_domain, tbx_smallEnd_d, tbx_bigEnd_d);
                const amrex::Box tby_guard = constrain_tilebox_to_guards(tby, dampdir, iside, nn_domain, tby_smallEnd_d, tby_bigEnd_d);
                const amrex::Box tbz_guard = constrain_tilebox_to_guards(tbz, dampdir, iside, nn_domain, tbz_smallEnd_d, tbz_bigEnd_d);

                // Do the damping
                amrex::ParallelFor(
                    tex_guard, Efield[0]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
                    {
                        damp_field_in_guards(Ex_arr, i, j, k, icomp, dampdir, nn_domain, tex_smallEnd_d, tex_bigEnd_d);
                    },
                    tey_guard, Efield[1]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
                    {
                        damp_field_in_guards(Ey_arr, i, j, k, icomp, dampdir, nn_domain, tey_smallEnd_d, tey_bigEnd_d);
                    },
                    tez_guard, Efield[2]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
                    {
                        damp_field_in_guards(Ez_arr, i, j, k, icomp, dampdir, nn_domain, tez_smallEnd_d, tez_bigEnd_d);
                    }
                );

                amrex::ParallelFor(
                    tbx_guard, Bfield[0]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
                    {
                        damp_field_in_guards(Bx_arr, i, j, k, icomp, dampdir, nn_domain, tbx_smallEnd_d, tbx_bigEnd_d);
                    },
                    tby_guard, Bfield[1]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
                    {
                        damp_field_in_guards(By_arr, i, j, k, icomp, dampdir, nn_domain, tby_smallEnd_d, tby_bigEnd_d);
                    },
                    tbz_guard, Bfield[2]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
                    {
                        damp_field_in_guards(Bz_arr, i, j, k, icomp, dampdir, nn_domain, tbz_smallEnd_d, tbz_bigEnd_d);
                    }
                );

            }
        }
    }
}

void WarpX::DampFieldsInGuards(const int lev, std::unique_ptr<amrex::MultiFab>& mf)
{
    // Loop over dimensions
    for (int dampdir = 0; dampdir < AMREX_SPACEDIM; dampdir++)
    {
        // Loop over the lower and upper guards
        for (int iside = 0; iside < 2; iside++)
        {
            // Only apply to damped boundaries
            if (iside == 0 && WarpX::field_boundary_lo[dampdir] != FieldBoundaryType::Damped) { continue; }
            if (iside == 1 && WarpX::field_boundary_hi[dampdir] != FieldBoundaryType::Damped) { continue; }

            for (amrex::MFIter mfi(*mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Array4<amrex::Real> const& mf_arr = mf->array(mfi);

                // Get the tilebox from mf so that it includes the guard cells
                // and takes the staggering of mf into account
                const amrex::Box tx = amrex::convert((*mf)[mfi].box(), mf->ixType().toIntVect());

                // Get smallEnd of tilebox
                const int tx_smallEnd_d = tx.smallEnd(dampdir);

                // Get bigEnd of tilebox
                const int tx_bigEnd_d = tx.bigEnd(dampdir);

                // Box for the whole simulation domain
                amrex::Box const& domain = Geom(lev).Domain();
                int const nn_domain = domain.bigEnd(dampdir);

                // Set the tilebox so that it only covers the lower/upper half of the guard cells
                const amrex::Box tx_guard = constrain_tilebox_to_guards(tx, dampdir, iside, nn_domain, tx_smallEnd_d, tx_bigEnd_d);

                // Do the damping
                amrex::ParallelFor(
                    tx_guard, mf->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
                    {
                        damp_field_in_guards(mf_arr, i, j, k, icomp, dampdir, nn_domain, tx_smallEnd_d, tx_bigEnd_d);
                    }
                );
            }
        }
    }
}

#ifdef WARPX_DIM_RZ
// This scales the current by the inverse volume and wraps around the deposition at negative radius.
// It is faster to apply this on the grid than to do it particle by particle.
// It is put here since there isn't another nice place for it.
void
WarpX::ApplyInverseVolumeScalingToCurrentDensity (MultiFab* Jx, MultiFab* Jy, MultiFab* Jz, int lev)
{
    const amrex::IntVect ngJ = Jx->nGrowVect();
    const std::array<Real,3>& dx = WarpX::CellSize(lev);
    const Real dr = dx[0];

    constexpr int NODE = amrex::IndexType::NODE;

    // See Verboncoeur JCP 174, 421-427 (2001) for the modified volume factor
    const amrex::Real axis_volume_factor = (verboncoeur_axis_correction ? 1._rt/3._rt : 1._rt/4._rt);

    for ( MFIter mfi(*Jx, TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {

        Array4<Real> const& Jr_arr = Jx->array(mfi);
        Array4<Real> const& Jt_arr = Jy->array(mfi);
        Array4<Real> const& Jz_arr = Jz->array(mfi);

        Box const & tilebox = mfi.tilebox();
        Box tbr = convert( tilebox, Jx->ixType().toIntVect() );
        Box tbt = convert( tilebox, Jy->ixType().toIntVect() );
        Box tbz = convert( tilebox, Jz->ixType().toIntVect() );

        // Lower corner of tile box physical domain
        // Note that this is done before the tilebox.grow so that
        // these do not include the guard cells.
        const amrex::XDim3 xyzmin = WarpX::LowerCorner(tilebox, lev, 0._rt);
        const Real rmin  = xyzmin.x;
        const Real rminr = xyzmin.x + (tbr.type(0) == NODE ? 0._rt : 0.5_rt*dx[0]);
        const Real rmint = xyzmin.x + (tbt.type(0) == NODE ? 0._rt : 0.5_rt*dx[0]);
        const Real rminz = xyzmin.x + (tbz.type(0) == NODE ? 0._rt : 0.5_rt*dx[0]);
        const Dim3 lo = lbound(tilebox);
        const int irmin = lo.x;

        // For ishift, 1 means cell centered, 0 means node centered
        int const ishift_r = (rminr > rmin ? 1 : 0);
        int const ishift_t = (rmint > rmin ? 1 : 0);
        int const ishift_z = (rminz > rmin ? 1 : 0);

        const int nmodes = n_rz_azimuthal_modes;

        // Grow the tileboxes to include the guard cells, except for the
        // guard cells at negative radius.
        if (rmin > 0._rt) {
           tbr.growLo(0, ngJ[0]);
           tbt.growLo(0, ngJ[0]);
           tbz.growLo(0, ngJ[0]);
        }
        tbr.growHi(0, ngJ[0]);
        tbt.growHi(0, ngJ[0]);
        tbz.growHi(0, ngJ[0]);
        tbr.grow(1, ngJ[1]);
        tbt.grow(1, ngJ[1]);
        tbz.grow(1, ngJ[1]);

        // Rescale current in r-z mode since the inverse volume factor was not
        // included in the current deposition.
        amrex::ParallelFor(tbr, tbt, tbz,
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
        {
            // Wrap the current density deposited in the guard cells around
            // to the cells above the axis.
            // If Jr is node centered, Jr[0] is located on the boundary.
            // If Jr is cell centered, Jr[0] is at 1/2 dr.
            if (rmin == 0. && 1-ishift_r <= i && i < ngJ[0]-ishift_r) {
                Jr_arr(i,j,0,0) -= Jr_arr(-ishift_r-i,j,0,0);
            }
            // Apply the inverse volume scaling
            // Jr is forced to zero on axis
            const amrex::Real r = amrex::Math::abs(rminr + (i - irmin)*dr);
            if (r == 0._rt) {
                Jr_arr(i,j,0,0) = 0._rt;
            } else {
                Jr_arr(i,j,0,0) /= (2._rt*MathConst::pi*r);
            }

            for (int imode=1 ; imode < nmodes ; imode++) {
                // Wrap the current density deposited in the guard cells around
                // to the cells above the axis.
                if (rmin == 0._rt && 1-ishift_r <= i && i < ngJ[0]-ishift_r) {
                    Jr_arr(i,j,0,2*imode-1) += static_cast<amrex::Real>(std::pow(-1, imode+1)*Jr_arr(-ishift_r-i,j,0,2*imode-1));
                    Jr_arr(i,j,0,2*imode) += static_cast<amrex::Real>(std::pow(-1, imode+1)*Jr_arr(-ishift_r-i,j,0,2*imode));
                }
                // Apply the inverse volume scaling
                // Jr is forced to zero on axis.
                if (r == 0._rt) {
                    Jr_arr(i,j,0,2*imode-1) = 0._rt;
                    Jr_arr(i,j,0,2*imode) = 0._rt;
                } else {
                    Jr_arr(i,j,0,2*imode-1) /= (2._rt*MathConst::pi*r);
                    Jr_arr(i,j,0,2*imode) /= (2._rt*MathConst::pi*r);
                }
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
        {
            // Wrap the current density deposited in the guard cells around
            // to the cells above the axis.
            // If Jt is node centered, Jt[0] is located on the boundary.
            // If Jt is cell centered, Jt[0] is at 1/2 dr.
            if (rmin == 0._rt && 1-ishift_t <= i && i <= ngJ[0]-ishift_t) {
                Jt_arr(i,j,0,0) -= Jt_arr(-ishift_t-i,j,0,0);
            }

            // Apply the inverse volume scaling
            // Jt is forced to zero on axis.
            const amrex::Real r = amrex::Math::abs(rmint + (i - irmin)*dr);
            if (r == 0._rt) {
                Jt_arr(i,j,0,0) = 0._rt;
            } else {
                Jt_arr(i,j,0,0) /= (2._rt*MathConst::pi*r);
            }

            for (int imode=1 ; imode < nmodes ; imode++) {
                // Wrap the current density deposited in the guard cells around
                // to the cells above the axis.
                if (rmin == 0._rt && 1-ishift_t <= i && i <= ngJ[0]-ishift_t) {
                    Jt_arr(i,j,0,2*imode-1) += static_cast<amrex::Real>(std::pow(-1, imode+1)*Jt_arr(-ishift_t-i,j,0,2*imode-1));
                    Jt_arr(i,j,0,2*imode) += static_cast<amrex::Real>(std::pow(-1, imode+1)*Jt_arr(-ishift_t-i,j,0,2*imode));
                }

                // Apply the inverse volume scaling
                // Jt is forced to zero on axis.
                if (r == 0._rt) {
                    Jt_arr(i,j,0,2*imode-1) = 0._rt;
                    Jt_arr(i,j,0,2*imode) = 0._rt;
                } else {
                    Jt_arr(i,j,0,2*imode-1) /= (2._rt*MathConst::pi*r);
                    Jt_arr(i,j,0,2*imode) /= (2._rt*MathConst::pi*r);
                }
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
        {
            // Wrap the current density deposited in the guard cells around
            // to the cells above the axis.
            // If Jz is node centered, Jz[0] is located on the boundary.
            // If Jz is cell centered, Jz[0] is at 1/2 dr.
            if (rmin == 0._rt && 1-ishift_z <= i && i <= ngJ[0]-ishift_z) {
                Jz_arr(i,j,0,0) += Jz_arr(-ishift_z-i,j,0,0);
            }

            // Apply the inverse volume scaling
            const amrex::Real r = amrex::Math::abs(rminz + (i - irmin)*dr);
            if (r == 0._rt) {
                Jz_arr(i,j,0,0) /= (MathConst::pi*dr*axis_volume_factor);
            } else {
                Jz_arr(i,j,0,0) /= (2._rt*MathConst::pi*r);
            }

            for (int imode=1 ; imode < nmodes ; imode++) {
                // Wrap the current density deposited in the guard cells around
                // to the cells above the axis.
                if (rmin == 0._rt && 1-ishift_z <= i && i <= ngJ[0]-ishift_z) {
                    Jz_arr(i,j,0,2*imode-1) -= static_cast<amrex::Real>(std::pow(-1, imode+1)*Jz_arr(-ishift_z-i,j,0,2*imode-1));
                    Jz_arr(i,j,0,2*imode) -= static_cast<amrex::Real>(std::pow(-1, imode+1)*Jz_arr(-ishift_z-i,j,0,2*imode));
                }

                // Apply the inverse volume scaling
                if (r == 0.) {
                    Jz_arr(i,j,0,2*imode-1) /= (MathConst::pi*dr*axis_volume_factor);
                    Jz_arr(i,j,0,2*imode) /= (MathConst::pi*dr*axis_volume_factor);
                } else {
                    Jz_arr(i,j,0,2*imode-1) /= (2._rt*MathConst::pi*r);
                    Jz_arr(i,j,0,2*imode) /= (2._rt*MathConst::pi*r);
                }
            }

        });
    }
}

void
WarpX::ApplyInverseVolumeScalingToChargeDensity (MultiFab* Rho, int lev)
{
    const amrex::IntVect ngRho = Rho->nGrowVect();
    const std::array<Real,3>& dx = WarpX::CellSize(lev);
    const Real dr = dx[0];

    constexpr int NODE = amrex::IndexType::NODE;

    // See Verboncoeur JCP 174, 421-427 (2001) for the modified volume factor
    const amrex::Real axis_volume_factor = (verboncoeur_axis_correction ? 1._rt/3._rt : 1._rt/4._rt);

    Box tilebox;

    for ( MFIter mfi(*Rho, TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {

        Array4<Real> const& Rho_arr = Rho->array(mfi);

        tilebox = mfi.tilebox();
        Box tb = convert( tilebox, Rho->ixType().toIntVect() );

        // Lower corner of tile box physical domain
        // Note that this is done before the tilebox.grow so that
        // these do not include the guard cells.
        const amrex::XDim3 xyzmin = WarpX::LowerCorner(tilebox, lev, 0._rt);
        const Dim3 lo = lbound(tilebox);
        const Real rmin = xyzmin.x;
        const Real rminr = xyzmin.x + (tb.type(0) == NODE ? 0._rt : 0.5_rt*dx[0]);
        const int irmin = lo.x;
        const int ishift = (rminr > rmin ? 1 : 0);

        // Grow the tilebox to include the guard cells, except for the
        // guard cells at negative radius.
        if (rmin > 0.) {
           tb.growLo(0, ngRho[0]);
        }
        tb.growHi(0, ngRho[0]);
        tb.grow(1, ngRho[1]);

        // Rescale charge in r-z mode since the inverse volume factor was not
        // included in the charge deposition.
        // Note that the loop is also over ncomps, which takes care of the RZ modes,
        // as well as the old and new rho.
        int const ncomp = Rho->nComp();
        amrex::ParallelFor(tb, ncomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/, int icomp)
        {
            // Wrap the charge density deposited in the guard cells around
            // to the cells above the axis.
            // Rho is located on the boundary
            if (rmin == 0. && 1-ishift <= i && i <= ngRho[0]-ishift) {
                int imode;
                if (icomp == 0 || icomp == ncomp/2) {
                    imode = 0;
                }
                else if (icomp < ncomp/2) {
                    imode = (icomp+1)/2;
                }
                else {
                    imode = (icomp - ncomp/2 + 1)/2;
                }
                Rho_arr(i,j,0,icomp) -= static_cast<amrex::Real>(std::pow(-1, imode+1)*Rho_arr(-ishift-i,j,0,icomp));
            }

            // Apply the inverse volume scaling
            const amrex::Real r = amrex::Math::abs(rminr + (i - irmin)*dr);
            if (r == 0.) {
                Rho_arr(i,j,0,icomp) /= (MathConst::pi*dr*axis_volume_factor);
            } else {
                Rho_arr(i,j,0,icomp) /= (2._rt*MathConst::pi*r);
            }
        });
    }
}
#endif
