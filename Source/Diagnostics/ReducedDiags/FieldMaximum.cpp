/* Copyright 2020 Neil Zaim, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldMaximum.H"

#include "Fields.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/coarsen/sample.H>

#include <AMReX_Algorithm.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ostream>
#include <vector>

using namespace amrex;
using warpx::fields::FieldType;

// constructor
FieldMaximum::FieldMaximum (const std::string& rd_name)
: ReducedDiags{rd_name}
{
    // Non-Cartesian coordinates not working
#if (defined WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(false,
        "FieldMaximum reduced diagnostics does not work for RZ coordinate.");
#endif

    // read number of levels
    int nLevel = 0;
    const ParmParse pp_amr("amr");
    pp_amr.query("max_level", nLevel);
    nLevel += 1;

    constexpr int noutputs = 8;  // max of Ex,Ey,Ez,|E|,Bx,By,Bz and |B|
    // resize data array
    m_data.resize(noutputs*nLevel, 0.0_rt);

#if defined(WARPX_DIM_RCYLINDER)
    std::vector<std::string> field_names = {"r", "t", "z"};
#elif defined(WARPX_DIM_RSPHERE)
    std::vector<std::string> field_names = {"r", "t", "p"};
#else
    std::vector<std::string> field_names = {"x", "y", "z"};
#endif

    if (ParallelDescriptor::IOProcessor())
    {
        if ( m_write_header )
        {
            // open file
            std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};
            // write header row
            int c = 0;
            ofs << "#";
            ofs << "[" << c++ << "]step()";
            ofs << m_sep;
            ofs << "[" << c++ << "]time(s)";
            for (int lev = 0; lev < nLevel; ++lev)
            {
                ofs << m_sep;
                ofs << "[" << c++ << "]max_E"+ field_names[0] + "_lev" + std::to_string(lev) + "(V/m)";
                ofs << m_sep;
                ofs << "[" << c++ << "]max_E"+ field_names[1] + "_lev" + std::to_string(lev) + "(V/m)";
                ofs << m_sep;
                ofs << "[" << c++ << "]max_E"+ field_names[2] + "_lev" + std::to_string(lev) + "(V/m)";
                ofs << m_sep;
                ofs << "[" << c++ << "]max_|E|_lev" + std::to_string(lev) + "(V/m)";
                ofs << m_sep;
                ofs << "[" << c++ << "]max_B"+ field_names[0] + "_lev" + std::to_string(lev) + "(T)";
                ofs << m_sep;
                ofs << "[" << c++ << "]max_B"+ field_names[1] + "_lev" + std::to_string(lev) + "(T)";
                ofs << m_sep;
                ofs << "[" << c++ << "]max_B"+ field_names[2] + "_lev" + std::to_string(lev) + "(T)";
                ofs << m_sep;
                ofs << "[" << c++ << "]max_|B|_lev" + std::to_string(lev) + "(T)";
            }
            ofs << "\n";
            // close file
            ofs.close();
        }
    }
}
// end constructor

// function that computes maximum field values
void FieldMaximum::ComputeDiags (int step)
{
    // Judge if the diags should be done
    if (!m_intervals.contains(step+1)) { return; }

    // get a reference to WarpX instance
    auto & warpx = WarpX::GetInstance();

    // get number of level
    const auto nLevel = warpx.finestLevel() + 1;

    using ablastr::fields::Direction;

    // loop over refinement levels
    for (int lev = 0; lev < nLevel; ++lev)
    {
        // get MultiFab data at lev
        const MultiFab & Ex = *warpx.m_fields.get(FieldType::Efield_aux, Direction{0}, lev);
        const MultiFab & Ey = *warpx.m_fields.get(FieldType::Efield_aux, Direction{1}, lev);
        const MultiFab & Ez = *warpx.m_fields.get(FieldType::Efield_aux, Direction{2}, lev);
        const MultiFab & Bx = *warpx.m_fields.get(FieldType::Bfield_aux, Direction{0}, lev);
        const MultiFab & By = *warpx.m_fields.get(FieldType::Bfield_aux, Direction{1}, lev);
        const MultiFab & Bz = *warpx.m_fields.get(FieldType::Bfield_aux, Direction{2}, lev);

        constexpr int noutputs = 8; // max of Ex,Ey,Ez,|E|,Bx,By,Bz and |B|
        constexpr int index_Ex = 0;
        constexpr int index_Ey = 1;
        constexpr int index_Ez = 2;
        constexpr int index_absE = 3;
        constexpr int index_Bx = 4;
        constexpr int index_By = 5;
        constexpr int index_Bz = 6;
        constexpr int index_absB = 7;

        // General preparation of interpolation and reduction operations
        const GpuArray<int,3> cellCenteredtype{0,0,0};
        const GpuArray<int,3> reduction_coarsening_ratio{1,1,1};
        constexpr int reduction_comp = 0;

        ReduceOps<ReduceOpMax,ReduceOpMax,ReduceOpMax,ReduceOpMax,
                  ReduceOpMax,ReduceOpMax,ReduceOpMax,ReduceOpMax> reduce_op;
        ReduceData<Real,Real,Real,Real,
                   Real,Real,Real,Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        // Prepare interpolation of field components to cell center
        // The arrays below store the index type (staggering) of each MultiFab, with the third
        // component set to zero in the two-dimensional case.
        auto Extype = amrex::GpuArray<int,3>{0,0,0};
        auto Eytype = amrex::GpuArray<int,3>{0,0,0};
        auto Eztype = amrex::GpuArray<int,3>{0,0,0};
        auto Bxtype = amrex::GpuArray<int,3>{0,0,0};
        auto Bytype = amrex::GpuArray<int,3>{0,0,0};
        auto Bztype = amrex::GpuArray<int,3>{0,0,0};
        for (int i = 0; i < AMREX_SPACEDIM; ++i){
            Extype[i] = Ex.ixType()[i];
            Eytype[i] = Ey.ixType()[i];
            Eztype[i] = Ez.ixType()[i];
            Bxtype[i] = Bx.ixType()[i];
            Bytype[i] = By.ixType()[i];
            Bztype[i] = Bz.ixType()[i];
        }

        // MFIter loop to interpolate fields to cell center and get maximum values
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(Ex, TilingIfNotGPU()); mfi.isValid(); ++mfi )
        {
            // Make the box cell centered in preparation for the interpolation (and to avoid
            // including ghost cells in the calculation)
            const Box& box = enclosedCells(mfi.nodaltilebox());
            const auto& arrEx = Ex[mfi].array();
            const auto& arrEy = Ey[mfi].array();
            const auto& arrEz = Ez[mfi].array();
            const auto& arrBx = Bx[mfi].array();
            const auto& arrBy = By[mfi].array();
            const auto& arrBz = Bz[mfi].array();

            reduce_op.eval(box, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                const Real Ex_interp = ablastr::coarsen::sample::Interp(arrEx, Extype, cellCenteredtype,
                                                                        reduction_coarsening_ratio, i, j, k, reduction_comp);
                const Real Ey_interp = ablastr::coarsen::sample::Interp(arrEy, Eytype, cellCenteredtype,
                                                                        reduction_coarsening_ratio, i, j, k, reduction_comp);
                const Real Ez_interp = ablastr::coarsen::sample::Interp(arrEz, Eztype, cellCenteredtype,
                                                                        reduction_coarsening_ratio, i, j, k, reduction_comp);
                const Real Bx_interp = ablastr::coarsen::sample::Interp(arrBx, Bxtype, cellCenteredtype,
                                                                        reduction_coarsening_ratio, i, j, k, reduction_comp);
                const Real By_interp = ablastr::coarsen::sample::Interp(arrBy, Bytype, cellCenteredtype,
                                                                        reduction_coarsening_ratio, i, j, k, reduction_comp);
                const Real Bz_interp = ablastr::coarsen::sample::Interp(arrBz, Bztype, cellCenteredtype,
                                                                        reduction_coarsening_ratio, i, j, k, reduction_comp);
                return {amrex::Math::abs(Ex_interp),
                        amrex::Math::abs(Ey_interp),
                        amrex::Math::abs(Ez_interp),
                        amrex::Math::abs(Bx_interp),
                        amrex::Math::abs(By_interp),
                        amrex::Math::abs(Bz_interp),
                        amrex::Math::powi<2>(Ex_interp) +
                        amrex::Math::powi<2>(Ey_interp) +
                        amrex::Math::powi<2>(Ez_interp),
                        amrex::Math::powi<2>(Bx_interp) +
                        amrex::Math::powi<2>(By_interp) +
                        amrex::Math::powi<2>(Bz_interp)};
            });
        }

        auto hv = reduce_data.value();
        Real hv_Ex = amrex::get<0>(hv); // highest value of |Ex|
        Real hv_Ey = amrex::get<1>(hv); // highest value of |Ey|
        Real hv_Ez = amrex::get<2>(hv); // highest value of |Ez|
        Real hv_Bx = amrex::get<3>(hv); // highest value of |Bx|
        Real hv_By = amrex::get<4>(hv); // highest value of |By|
        Real hv_Bz = amrex::get<5>(hv); // highest value of |Bz|
        Real hv_E  = amrex::get<6>(hv); // highest value of |E|**2
        Real hv_B  = amrex::get<7>(hv); // highest value of |B|**2

        // MPI reduce
        ParallelDescriptor::ReduceRealMax({hv_Ex,hv_Ey,hv_Ez,
                                           hv_Bx,hv_By,hv_Bz, hv_E, hv_B});

        // Fill output array
        m_data[lev*noutputs+index_Ex] = hv_Ex;
        m_data[lev*noutputs+index_Ey] = hv_Ey;
        m_data[lev*noutputs+index_Ez] = hv_Ez;
        m_data[lev*noutputs+index_Bx] = hv_Bx;
        m_data[lev*noutputs+index_By] = hv_By;
        m_data[lev*noutputs+index_Bz] = hv_Bz;
        m_data[lev*noutputs+index_absE] = std::sqrt(hv_E);
        m_data[lev*noutputs+index_absB] = std::sqrt(hv_B);
    }
    // end loop over refinement levels

    /* m_data now contains up-to-date values for:
     *  [max(Ex),max(Ey),max(Ez),max(|E|),
     *   max(Bx),max(By),max(Bz),max(|B|)] */
}
// end void FieldMaximum::ComputeDiags
