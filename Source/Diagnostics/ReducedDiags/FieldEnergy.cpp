/* Copyright 2019-2020 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldEnergy.H"

#include "Fields.H"
#include "Diagnostics/ReducedDiags/ReducedDiags.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Array4.H>
#include <AMReX_Config.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_MFIter.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <fstream>
#include <vector>

using namespace amrex::literals;
using warpx::fields::FieldType;

// constructor
FieldEnergy::FieldEnergy (const std::string& rd_name)
: ReducedDiags{rd_name}
{

    // read number of levels
    int nLevel = 0;
    amrex::ParmParse const pp_amr("amr");
    pp_amr.query("max_level", nLevel);
    nLevel += 1;

    constexpr int noutputs = 3; // total energy, E-field energy and B-field energy
    // resize data array
    m_data.resize(noutputs*nLevel, 0.0_rt);

    if (amrex::ParallelDescriptor::IOProcessor())
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
                ofs << "[" << c++ << "]total_lev" + std::to_string(lev) + "(J)";
                ofs << m_sep;
                ofs << "[" << c++ << "]E_lev" + std::to_string(lev) + "(J)";
                ofs << m_sep;
                ofs << "[" << c++ << "]B_lev" + std::to_string(lev) + "(J)";
            }
            ofs << "\n";
            // close file
            ofs.close();
        }
    }
}
// end constructor

// function that computes field energy
void FieldEnergy::ComputeDiags (int step)
{
    // Judge if the diags should be done
    if (!m_intervals.contains(step+1)) { return; }

    // get a reference to WarpX instance
    auto const & warpx = WarpX::GetInstance();

    // get number of level
    int const nLevel = warpx.finestLevel() + 1;

    using ablastr::fields::Direction;

    // loop over refinement levels
    for (int lev = 0; lev < nLevel; ++lev)
    {
        // get MultiFab data at lev
        amrex::MultiFab const & Ex = *warpx.m_fields.get(FieldType::Efield_aux, Direction{0}, lev);
        amrex::MultiFab const & Ey = *warpx.m_fields.get(FieldType::Efield_aux, Direction{1}, lev);
        amrex::MultiFab const & Ez = *warpx.m_fields.get(FieldType::Efield_aux, Direction{2}, lev);
        amrex::MultiFab const & Bx = *warpx.m_fields.get(FieldType::Bfield_aux, Direction{0}, lev);
        amrex::MultiFab const & By = *warpx.m_fields.get(FieldType::Bfield_aux, Direction{1}, lev);
        amrex::MultiFab const & Bz = *warpx.m_fields.get(FieldType::Bfield_aux, Direction{2}, lev);

        // get cell volume
        std::array<amrex::Real, 3> const &dx = WarpX::CellSize(lev);
        amrex::Real const dV = dx[0]*dx[1]*dx[2];

        // compute E squared
        amrex::Real const tmpEx = ComputeNorm2(Ex, lev);
        amrex::Real const tmpEy = ComputeNorm2(Ey, lev);
        amrex::Real const tmpEz = ComputeNorm2(Ez, lev);

        // compute B squared
        amrex::Real const tmpBx = ComputeNorm2(Bx, lev);
        amrex::Real const tmpBy = ComputeNorm2(By, lev);
        amrex::Real const tmpBz = ComputeNorm2(Bz, lev);

        amrex::Real const Es = tmpEx + tmpEy + tmpEz;
        amrex::Real const Bs = tmpBx + tmpBy + tmpBz;

        constexpr int noutputs = 3; // total energy, E-field energy and B-field energy
        constexpr int index_total = 0;
        constexpr int index_E = 1;
        constexpr int index_B = 2;

        // save data
        m_data[lev*noutputs+index_E] = 0.5_rt * Es * PhysConst::ep0 * dV;
        m_data[lev*noutputs+index_B] = 0.5_rt * Bs / PhysConst::mu0 * dV;
        m_data[lev*noutputs+index_total] = m_data[lev*noutputs+index_E] +
                                           m_data[lev*noutputs+index_B];
    }
    // end loop over refinement levels

    /* m_data now contains up-to-date values for:
     *  [total field energy at level 0,
     *   electric field energy at level 0,
     *   magnetic field energy at level 0,
     *   total field energy at level 1,
     *   electric field energy at level 1,
     *   magnetic field energy at level 1,
     *   ......] */
}
// end void FieldEnergy::ComputeDiags

// Function that computes the sum of the field squared.
// This takes into account the fraction of the cell volumes within the domain
// and the cell volumes in cylindrical coordinates.
amrex::Real
FieldEnergy::ComputeNorm2(amrex::MultiFab const& field, [[maybe_unused]]int lev)
{
    amrex::IntVect const is_nodal = field.ixType().toIntVect();

    amrex::ReduceOps<amrex::ReduceOpSum> reduce_ops;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_ops);
    using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(field, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {

        amrex::Array4<const amrex::Real> const& field_arr = field.array(mfi);

        amrex::Box const tilebox = mfi.tilebox();
        amrex::Box const tb = convert(tilebox, is_nodal);
        amrex::IntVect const tb_lo = tb.smallEnd();
        amrex::IntVect const tb_hi = tb.bigEnd();

#if defined(WARPX_DIM_RZ)
        // Lower corner of tile box physical domain
        auto const & warpx = WarpX::GetInstance();
        amrex::Geometry const & geom = warpx.Geom(lev);
        amrex::Real const dr = geom.CellSize(0);
        amrex::XDim3 const xyzmin = WarpX::LowerCorner(tilebox, lev, 0._rt);
        amrex::Real const rmin = xyzmin.x + (is_nodal[0] ? 0._rt : 0.5_rt*dr);
#endif

        // On the boundaries, if the grid is nodal, use half of the volume.
        // This applies to all boundary conditions, and to the overlap of
        // boxes within the domain.
        // Previously, the code used MultiFab::norm2, but that does not do
        // the half-volume scaling for the domain boundaries when not periodic.

        auto volume_factor = [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            amrex::ignore_unused(i,j,k,n);
#if defined WARPX_DIM_RZ
            amrex::Real const r = rmin + (i - tb_lo[0])*dr;
            amrex::Real v_factor = 2._rt*r;
            if (i == tb_lo[0] && is_nodal[0]) { v_factor = r + dr/4._rt; }
            if (i == tb_hi[0] && is_nodal[0]) { v_factor = r - dr/4._rt; }
            if (j == tb_lo[1] && is_nodal[1]) { v_factor *= 0.5_rt; }
            if (j == tb_hi[1] && is_nodal[1]) { v_factor *= 0.5_rt; }
            amrex::Real const theta_integral = (n == 0 ? 1._rt : 0.5_rt);
            return MathConst::pi*v_factor*theta_integral;
#else
            amrex::Real v_factor = 1._rt;
            AMREX_D_TERM(
            if (i == tb_lo[0] && is_nodal[0]) { v_factor *= 0.5_rt; },
            if (j == tb_lo[1] && is_nodal[1]) { v_factor *= 0.5_rt; },
            if (k == tb_lo[2] && is_nodal[2]) { v_factor *= 0.5_rt; })
            AMREX_D_TERM(
            if (i == tb_hi[0] && is_nodal[0]) { v_factor *= 0.5_rt; },
            if (j == tb_hi[1] && is_nodal[1]) { v_factor *= 0.5_rt; },
            if (k == tb_hi[2] && is_nodal[2]) { v_factor *= 0.5_rt; })
            return v_factor;
#endif
        };

        int const ncomp = field.nComp();

        reduce_ops.eval(tb, ncomp, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) -> ReduceTuple
            {
                return field_arr(i,j,k,n)*field_arr(i,j,k,n)*volume_factor(i,j,k,n);
            });

    }

    amrex::Real result = amrex::get<0>(reduce_data.value());
    amrex::ParallelDescriptor::ReduceRealSum(result);

    return result;
}
