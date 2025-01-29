/* Copyright 2019-2020
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldMomentum.H"

#include "Fields.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/coarsen/sample.H>

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>
#include <AMReX_Tuple.H>
#include <AMReX_Vector.H>

#include <ostream>
#include <algorithm>
#include <vector>

using namespace amrex;
using warpx::fields::FieldType;

FieldMomentum::FieldMomentum (const std::string& rd_name)
    : ReducedDiags{rd_name}
{
    // RZ coordinate is not working
#if (defined WARPX_DIM_RZ)
        WARPX_ABORT_WITH_MESSAGE(
            "FieldMomentum reduced diagnostics not implemented in RZ geometry");
#endif

    // Read number of levels
    int nLevel = 0;
    const amrex::ParmParse pp_amr("amr");
    pp_amr.query("max_level", nLevel);
    nLevel += 1;

    // Resize data array
    m_data.resize(nLevel*3, 0.0_rt);

    if (amrex::ParallelDescriptor::IOProcessor())
    {
        if (m_write_header)
        {
            // Open file
            std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};

            int c = 0;

            // Write header row
            ofs << "#";
            ofs << "[" << c++ << "]step()";
            ofs << m_sep;
            ofs << "[" << c++ << "]time(s)";

            // Loop over refinement levels
            for (int lev = 0; lev < nLevel; ++lev)
            {
                ofs << m_sep;
                ofs << "[" << c++ << "]";
                ofs << "momentum_x_lev" << lev << "(kg*m/s)";
                ofs << m_sep;
                ofs << "[" << c++ << "]";
                ofs << "momentum_y_lev" << lev << "(kg*m/s)";
                ofs << m_sep;
                ofs << "[" << c++ << "]";
                ofs << "momentum_z_lev" << lev << "(kg*m/s)";
            }

            ofs << "\n";
            ofs.close();
        }
    }
}

void FieldMomentum::ComputeDiags (int step)
{
    // Check if the diags should be done
    if (!m_intervals.contains(step+1)) { return; }

    // Get a reference to WarpX instance
    auto & warpx = WarpX::GetInstance();

    // Get number of refinement levels
    const auto nLevel = warpx.finestLevel() + 1;

    using ablastr::fields::Direction;

    // Loop over refinement levels
    for (int lev = 0; lev < nLevel; ++lev)
    {
        // Get MultiFab data at given refinement level
        const amrex::MultiFab & Ex = *warpx.m_fields.get(FieldType::Efield_aux, Direction{0}, lev);
        const amrex::MultiFab & Ey = *warpx.m_fields.get(FieldType::Efield_aux, Direction{1}, lev);
        const amrex::MultiFab & Ez = *warpx.m_fields.get(FieldType::Efield_aux, Direction{2}, lev);
        const amrex::MultiFab & Bx = *warpx.m_fields.get(FieldType::Bfield_aux, Direction{0}, lev);
        const amrex::MultiFab & By = *warpx.m_fields.get(FieldType::Bfield_aux, Direction{1}, lev);
        const amrex::MultiFab & Bz = *warpx.m_fields.get(FieldType::Bfield_aux, Direction{2}, lev);

        // Cell-centered index type
        const amrex::GpuArray<int,3> cc{0,0,0};
        // Coarsening ratio (no coarsening)
        const amrex::GpuArray<int,3> cr{1,1,1};
        // Reduction component (fourth component in Array4)
        constexpr int comp = 0;

        // Index type (staggering) of each MultiFab
        // (with third component set to zero in 2D)
        amrex::GpuArray<int,3> Ex_stag{0,0,0};
        amrex::GpuArray<int,3> Ey_stag{0,0,0};
        amrex::GpuArray<int,3> Ez_stag{0,0,0};
        amrex::GpuArray<int,3> Bx_stag{0,0,0};
        amrex::GpuArray<int,3> By_stag{0,0,0};
        amrex::GpuArray<int,3> Bz_stag{0,0,0};
        for (int i = 0; i < AMREX_SPACEDIM; ++i)
        {
            Ex_stag[i] = Ex.ixType()[i];
            Ey_stag[i] = Ey.ixType()[i];
            Ez_stag[i] = Ez.ixType()[i];
            Bx_stag[i] = Bx.ixType()[i];
            By_stag[i] = By.ixType()[i];
            Bz_stag[i] = Bz.ixType()[i];
        }

        amrex::ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_ops;
        amrex::ReduceData<Real, Real, Real> reduce_data(reduce_ops);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        // Loop over boxes, interpolate E,B data to cell centers
        // and compute sum over cells of (E x B) components
        for (amrex::MFIter mfi(Ex, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box & box = enclosedCells(mfi.nodaltilebox());
            const amrex::Array4<const amrex::Real> & Ex_arr = Ex[mfi].array();
            const amrex::Array4<const amrex::Real> & Ey_arr = Ey[mfi].array();
            const amrex::Array4<const amrex::Real> & Ez_arr = Ez[mfi].array();
            const amrex::Array4<const amrex::Real> & Bx_arr = Bx[mfi].array();
            const amrex::Array4<const amrex::Real> & By_arr = By[mfi].array();
            const amrex::Array4<const amrex::Real> & Bz_arr = Bz[mfi].array();

            // Compute E x B
            reduce_ops.eval(box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> amrex::GpuTuple<Real, Real, Real>
                {
                    const amrex::Real Ex_cc = ablastr::coarsen::sample::Interp(Ex_arr, Ex_stag, cc, cr, i, j, k, comp);
                    const amrex::Real Ey_cc = ablastr::coarsen::sample::Interp(Ey_arr, Ey_stag, cc, cr, i, j, k, comp);
                    const amrex::Real Ez_cc = ablastr::coarsen::sample::Interp(Ez_arr, Ez_stag, cc, cr, i, j, k, comp);

                    const amrex::Real Bx_cc = ablastr::coarsen::sample::Interp(Bx_arr, Bx_stag, cc, cr, i, j, k, comp);
                    const amrex::Real By_cc = ablastr::coarsen::sample::Interp(By_arr, By_stag, cc, cr, i, j, k, comp);
                    const amrex::Real Bz_cc = ablastr::coarsen::sample::Interp(Bz_arr, Bz_stag, cc, cr, i, j, k, comp);

                    return {Ey_cc * Bz_cc - Ez_cc * By_cc,
                            Ez_cc * Bx_cc - Ex_cc * Bz_cc,
                            Ex_cc * By_cc - Ey_cc * Bx_cc};
                });
        }

        // MPI reduce
        auto r = reduce_data.value();
        amrex::Real ExB_x = amrex::get<0>(r);
        amrex::Real ExB_y = amrex::get<1>(r);
        amrex::Real ExB_z = amrex::get<2>(r);
        amrex::ParallelDescriptor::ReduceRealSum({ExB_x,ExB_y,ExB_z});

        // Get cell volume
        const std::array<Real, 3> &dx = WarpX::CellSize(lev);
        const amrex::Real dV = dx[0]*dx[1]*dx[2];

        // Save data (offset: 3 values for each refinement level)
        const int offset = lev*3;
        m_data[offset+0] = PhysConst::ep0 * ExB_x * dV;
        m_data[offset+1] = PhysConst::ep0 * ExB_y * dV;
        m_data[offset+2] = PhysConst::ep0 * ExB_z * dV;
    }
}
