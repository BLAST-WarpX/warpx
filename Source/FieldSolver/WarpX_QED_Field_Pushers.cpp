/* Copyright 2019-2020 Glenn Richardson, Maxence Thevenet, Revathi Jambunathan, Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#include "Fields.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "WarpX_QED_K.H"

#include <ablastr/profiler/ProfilerWrapper.H>
#include <AMReX.H>
#ifdef AMREX_USE_SENSEI_INSITU
#   include <AMReX_AmrMeshInSituBridge.H>
#endif
#include <AMReX_Arena.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>

using namespace amrex;


void
WarpX::Hybrid_QED_Push (amrex::Vector<amrex::Real> a_dt)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::grid_type == GridType::Collocated,
        "Error: The Hybrid QED method is "
        "currently only implemented on a collocated grid."
    );
    for (int lev = 0; lev <= finest_level; ++lev) {
        Hybrid_QED_Push(lev, a_dt[lev]);
    }
}

void
WarpX::Hybrid_QED_Push (int lev, amrex::Real a_dt)
{
    ABLASTR_PROFILE("WarpX::Hybrid_QED_Push()");
    Hybrid_QED_Push(lev, PatchType::fine, a_dt);
    if (lev > 0)
    {
        Hybrid_QED_Push(lev, PatchType::coarse, a_dt);
    }
}

void
WarpX::Hybrid_QED_Push (int lev, PatchType patch_type, amrex::Real a_dt)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    const int patch_level = (patch_type == PatchType::fine) ? lev : lev-1;
    const std::array<Real,3>& dx_vec= WarpX::CellSize(patch_level);
    const Real dx = dx_vec[0];
    const Real dy = dx_vec[1];
    const Real dz = dx_vec[2];

    using ablastr::fields::Direction;

    MultiFab *Ex, *Ey, *Ez, *Bx, *By, *Bz, *Jx, *Jy, *Jz;
    if (patch_type == PatchType::fine)
    {
        Ex = m_fields.get(FieldType::Efield_fp, Direction{0}, lev);
        Ey = m_fields.get(FieldType::Efield_fp, Direction{1}, lev);
        Ez = m_fields.get(FieldType::Efield_fp, Direction{2}, lev);
        Bx = m_fields.get(FieldType::Bfield_fp, Direction{0}, lev);
        By = m_fields.get(FieldType::Bfield_fp, Direction{1}, lev);
        Bz = m_fields.get(FieldType::Bfield_fp, Direction{2}, lev);
        Jx = m_fields.get(FieldType::current_fp, Direction{0}, lev);
        Jy = m_fields.get(FieldType::current_fp, Direction{1}, lev);
        Jz = m_fields.get(FieldType::current_fp, Direction{2}, lev);
    }
    else
    {
        Ex = m_fields.get(FieldType::Efield_cp, Direction{0}, lev);
        Ey = m_fields.get(FieldType::Efield_cp, Direction{1}, lev);
        Ez = m_fields.get(FieldType::Efield_cp, Direction{2}, lev);
        Bx = m_fields.get(FieldType::Bfield_cp, Direction{0}, lev);
        By = m_fields.get(FieldType::Bfield_cp, Direction{1}, lev);
        Bz = m_fields.get(FieldType::Bfield_cp, Direction{2}, lev);
        Jx = m_fields.get(FieldType::current_cp, Direction{0}, lev);
        Jy = m_fields.get(FieldType::current_cp, Direction{1}, lev);
        Jz = m_fields.get(FieldType::current_cp, Direction{2}, lev);
    }

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Copy of E, with one guard cell for the finite difference stencil.
    // It is made for the whole level before E is updated in place below:
    // a copy made tile by tile would pick up values of E already updated
    // in the neighboring tiles.
    MultiFab tmpEx_mf(Ex->boxArray(), Ex->DistributionMap(), 1, 1);
    MultiFab tmpEy_mf(Ey->boxArray(), Ey->DistributionMap(), 1, 1);
    MultiFab tmpEz_mf(Ez->boxArray(), Ez->DistributionMap(), 1, 1);
    MultiFab::Copy(tmpEx_mf, *Ex, 0, 0, 1, 1);
    MultiFab::Copy(tmpEy_mf, *Ey, 0, 0, 1, 1);
    MultiFab::Copy(tmpEz_mf, *Ez, 0, 0, 1, 1);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bx, TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        // Get boxes for B

        const Box& tbx = mfi.tilebox(Bx->ixType().toIntVect());

        // Get field arrays
        auto const& Bxfab = Bx->array(mfi);
        auto const& Byfab = By->array(mfi);
        auto const& Bzfab = Bz->array(mfi);
        auto const& Exfab = Ex->array(mfi);
        auto const& Eyfab = Ey->array(mfi);
        auto const& Ezfab = Ez->array(mfi);
        auto const& Jxfab = Jx->array(mfi);
        auto const& Jyfab = Jy->array(mfi);
        auto const& Jzfab = Jz->array(mfi);

        auto const& tmpEx = tmpEx_mf.array(mfi);
        auto const& tmpEy = tmpEy_mf.array(mfi);
        auto const& tmpEz = tmpEz_mf.array(mfi);

        // Make local copy of xi, to use on device.
        const Real xi_c2 = m_quantum_xi_c2;

        // Apply QED correction to electric field, using temporary arrays.
        amrex::ParallelFor(
            tbx,
            [=] AMREX_GPU_DEVICE (int j, int k, int l)
            {
                warpx_hybrid_QED_push(j,k,l, Exfab, Eyfab, Ezfab, Bxfab, Byfab,
                    Bzfab, tmpEx, tmpEy, tmpEz, Jxfab, Jyfab, Jzfab, dx, dy, dz,
                    a_dt, xi_c2);
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}
