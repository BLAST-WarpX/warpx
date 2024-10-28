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

using namespace amrex;
using warpx::fields::FieldType;

// constructor
FieldEnergy::FieldEnergy (const std::string& rd_name)
: ReducedDiags{rd_name}
{

    // read number of levels
    int nLevel = 0;
    const ParmParse pp_amr("amr");
    pp_amr.query("max_level", nLevel);
    nLevel += 1;

    constexpr int noutputs = 3; // total energy, E-field energy and B-field energy
    // resize data array
    m_data.resize(noutputs*nLevel, 0.0_rt);

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

        // get cell volume
        const std::array<Real, 3> &dx = WarpX::CellSize(lev);
        const amrex::Real dV = dx[0]*dx[1]*dx[2];

#if defined(WARPX_DIM_RZ)
        amrex::Real const tmpEx = ComputeNorm2RZ(Ex, lev);
        amrex::Real const tmpEy = ComputeNorm2RZ(Ey, lev);
        amrex::Real const tmpEz = ComputeNorm2RZ(Ez, lev);
        amrex::Real const Es = tmpEx + tmpEy + tmpEz;

        amrex::Real const tmpBx = ComputeNorm2RZ(Bx, lev);
        amrex::Real const tmpBy = ComputeNorm2RZ(By, lev);
        amrex::Real const tmpBz = ComputeNorm2RZ(Bz, lev);
        amrex::Real const Bs = tmpBx + tmpBy + tmpBz;
#else
        Geometry const & geom = warpx.Geom(lev);

        // compute E squared
        Real const tmpEx = Ex.norm2(0,geom.periodicity());
        Real const tmpEy = Ey.norm2(0,geom.periodicity());
        Real const tmpEz = Ez.norm2(0,geom.periodicity());
        Real const Es = tmpEx*tmpEx + tmpEy*tmpEy + tmpEz*tmpEz;

        // compute B squared
        Real const tmpBx = Bx.norm2(0,geom.periodicity());
        Real const tmpBy = By.norm2(0,geom.periodicity());
        Real const tmpBz = Bz.norm2(0,geom.periodicity());
        Real const Bs = tmpBx*tmpBx + tmpBy*tmpBy + tmpBz*tmpBz;
#endif

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

// Function that computes the sum of the field squared in RZ
amrex::Real
FieldEnergy::ComputeNorm2RZ(const amrex::MultiFab& field, const int lev)
{
    // get a reference to WarpX instance
    auto & warpx = WarpX::GetInstance();

    Geometry const & geom = warpx.Geom(lev);
    const amrex::Real dr = geom.CellSize(0);

    amrex::ReduceOps<amrex::ReduceOpSum> reduce_ops;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_ops);
    using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(field, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {

        amrex::Array4<const amrex::Real> const& field_arr = field.array(mfi);

        const amrex::Box tilebox = mfi.tilebox();
        amrex::Box tb = convert(tilebox, field.ixType().toIntVect());

        // Lower corner of tile box physical domain
        const amrex::XDim3 xyzmin = WarpX::LowerCorner(tilebox, lev, 0._rt);
        const Dim3 lo = lbound(tilebox);
        const Dim3 hi = ubound(tilebox);
        const Real rmin = xyzmin.x + (tb.ixType().nodeCentered(0) ? 0._rt : 0.5_rt*dr);
        const int irmin = lo.x;
        const int irmax = hi.x;

        int const ncomp = field.nComp();

        for (int idir=0 ; idir < AMREX_SPACEDIM ; idir++) {
            if (WarpX::field_boundary_hi[idir] == FieldBoundaryType::Periodic) {
                // For periodic boundaries, do not include the data in the nodes
                // on the upper edge of the domain
                tb.enclosedCells(idir);
            }
        }

        reduce_ops.eval(tb, ncomp, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) -> ReduceTuple
            {
                const amrex::Real r = rmin + (i - irmin)*dr;
                amrex::Real volume_factor = r;
                if (r == 0._rt) {
                    volume_factor = dr/8._rt;
                } else if (rmin == 0._rt && i == irmax) {
                    volume_factor = r/2._rt - dr/8._rt;
                }
                const amrex::Real theta_integral = (n == 0 ? 2._rt : 1._rt);
                return theta_integral*field_arr(i,j,k,n)*field_arr(i,j,k,n)*volume_factor;
            });

    }

    const amrex::Real field_sum = amrex::get<0>(reduce_data.value());
    const amrex::Real result = MathConst::pi*field_sum;
    return result;
}
// end Real FieldEnergy::ComputeNorm2RZ
