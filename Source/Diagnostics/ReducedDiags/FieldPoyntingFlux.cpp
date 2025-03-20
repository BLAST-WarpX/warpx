/* Copyright 2019-2020
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldPoyntingFlux.H"

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

using namespace amrex::literals;

FieldPoyntingFlux::FieldPoyntingFlux (const std::string& rd_name)
    : ReducedDiags{rd_name}
{
    // Resize data array
    // lo and hi is 2
    // space dims is AMREX_SPACEDIM
    // instantaneous and integrated is 2
    // The order will be outward flux for low faces, then high faces,
    // energy loss for low faces, then high faces
    m_data.resize(2*AMREX_SPACEDIM*2, 0.0_rt);

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

            std::vector<std::string> sides = {"lo", "hi"};

#if defined(WARPX_DIM_3D)
            std::vector<std::string> space_coords = {"x", "y", "z"};
#elif defined(WARPX_DIM_XZ)
            std::vector<std::string> space_coords = {"x", "z"};
#elif defined(WARPX_DIM_1D_Z)
            std::vector<std::string> space_coords = {"z"};
#elif defined(WARPX_DIM_RZ)
            std::vector<std::string> space_coords = {"r", "z"};
#endif

            // Only on level 0
            for (int iside = 0; iside < 2; iside++) {
                for (int ic = 0; ic < AMREX_SPACEDIM; ic++) {
                    ofs << m_sep;
                    ofs << "[" << c++ << "]outward_power_" + sides[iside] + "_" + space_coords[ic] +"(W)";
            }}
            for (int iside = 0; iside < 2; iside++) {
                for (int ic = 0; ic < AMREX_SPACEDIM; ic++) {
                    ofs << m_sep;
                    ofs << "[" << c++ << "]integrated_energy_loss_" + sides[iside] + "_" + space_coords[ic] +"(J)";
            }}

            ofs << "\n";
            ofs.close();
        }
    }
}

void FieldPoyntingFlux::ComputeDiags (int /*step*/)
{
    // This will be called at the end of the time step. Only calculate the
    // flux if it had not already been calculated mid step.
    if (!use_mid_step_value) {
        ComputePoyntingFlux();
    }
}

void FieldPoyntingFlux::ComputeDiagsMidStep (int /*step*/)
{
    // If this is called, always use the value calculated here.
    use_mid_step_value = true;
    ComputePoyntingFlux();
}

void FieldPoyntingFlux::ComputePoyntingFlux ()
{
    using warpx::fields::FieldType;
    using ablastr::fields::Direction;

    // Note that this is calculated every step to get the
    // full resolution on the integrated data

    int const lev = 0;

    // Get a reference to WarpX instance
    auto & warpx = WarpX::GetInstance();

    // RZ coordinate only working with one mode
#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(warpx.n_rz_azimuthal_modes == 1,
        "FieldPoyntingFlux reduced diagnostics only implemented in RZ geometry for one mode");
#endif

    amrex::Box domain_box = warpx.Geom(lev).Domain();
    domain_box.surroundingNodes();

    // Get MultiFab data at given refinement level
    amrex::MultiFab const & Ex = *warpx.m_fields.get(FieldType::Efield_fp, Direction{0}, lev);
    amrex::MultiFab const & Ey = *warpx.m_fields.get(FieldType::Efield_fp, Direction{1}, lev);
    amrex::MultiFab const & Ez = *warpx.m_fields.get(FieldType::Efield_fp, Direction{2}, lev);
    amrex::MultiFab const & Bx = *warpx.m_fields.get(FieldType::Bfield_fp, Direction{0}, lev);
    amrex::MultiFab const & By = *warpx.m_fields.get(FieldType::Bfield_fp, Direction{1}, lev);
    amrex::MultiFab const & Bz = *warpx.m_fields.get(FieldType::Bfield_fp, Direction{2}, lev);

    // Coarsening ratio (no coarsening)
    amrex::GpuArray<int,3> const cr{1,1,1};

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

    for (amrex::OrientationIter face; face; ++face) {

        int const face_dir = face().coordDir();

        if (face().isHigh() && WarpX::field_boundary_hi[face_dir] == FieldBoundaryType::Periodic) {
            // For upper periodic boundaries, copy the lower value instead of regenerating it.
            int const iu = int(face());
            int const il = int(face().flip());
            m_data[iu] = -m_data[il];
            m_data[iu + 2*AMREX_SPACEDIM] = -m_data[il + 2*AMREX_SPACEDIM];
            continue;
        }

        amrex::Box const boundary = amrex::bdryNode(domain_box, face());

        // Get cell area
        amrex::Real const *dx = warpx.Geom(lev).CellSize();
        std::array<amrex::Real, AMREX_SPACEDIM> dxtemp = {AMREX_D_DECL(dx[0], dx[1], dx[2])};
        dxtemp[face_dir] = 1._rt;
        amrex::Real const dA = AMREX_D_TERM(dxtemp[0], *dxtemp[1], *dxtemp[2]);

        // Node-centered in the face direction, Cell-centered in other directions
        amrex::GpuArray<int,3> cc{0,0,0};
        cc[face_dir] = 1;

        // Only calculate the ExB term that is normal to the surface.
        // normal_dir is the normal direction relative to the WarpX coordinates
#if (defined WARPX_DIM_XZ) || (defined WARPX_DIM_RZ)
        // For 2D : it is either 0, or 2
        int const normal_dir = 2*face_dir;
#elif (defined WARPX_DIM_1D_Z)
        // For 1D : it is always 2
        int const normal_dir = 2;
#else
        // For 3D : it is the same as the face direction
        int const normal_dir = face_dir;
#endif

        amrex::ReduceOps<amrex::ReduceOpSum> reduce_ops;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_ops);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        // Loop over boxes, interpolate E,B data to cell face centers
        // and compute sum over cells of (E x B) components
        for (amrex::MFIter mfi(Ex, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Array4<const amrex::Real> const & Ex_arr = Ex[mfi].array();
            amrex::Array4<const amrex::Real> const & Ey_arr = Ey[mfi].array();
            amrex::Array4<const amrex::Real> const & Ez_arr = Ez[mfi].array();
            amrex::Array4<const amrex::Real> const & Bx_arr = Bx[mfi].array();
            amrex::Array4<const amrex::Real> const & By_arr = By[mfi].array();
            amrex::Array4<const amrex::Real> const & Bz_arr = Bz[mfi].array();

            amrex::Box box = enclosedCells(mfi.nodaltilebox());
            box.surroundingNodes(face_dir);

            // Find the intersection with the boundary
            // boundary needs to have the same type as box
            amrex::Box const boundary_matched = amrex::convert(boundary, box.ixType());
            box &= boundary_matched;

#if defined(WARPX_DIM_RZ)
            // Lower corner of box physical domain
            amrex::XDim3 const xyzmin = WarpX::LowerCorner(box, lev, 0._rt);
            amrex::Dim3 const lo = amrex::lbound(box);
            amrex::Real const dr = warpx.Geom(lev).CellSize(lev);
            amrex::Real const rmin = xyzmin.x;
            int const irmin = lo.x;
#endif

            auto area_factor = [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::ignore_unused(i,j,k);
#if defined WARPX_DIM_RZ
                amrex::Real r;
                if (normal_dir == 0) {
                    r = rmin + (i - irmin)*dr;
                } else {
                    r = rmin + (i + 0.5_rt - irmin)*dr;
                }
                return 2._rt*MathConst::pi*r;
#else
                return 1._rt;
#endif
            };

            // Compute E x B
            // On GPU, reduce_ops doesn't work with empty boxes.
            if (box.ok()) {
                reduce_ops.eval(box, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> amrex::GpuTuple<amrex::Real>
                    {
                        amrex::Real Ex_cc = 0._rt, Ey_cc = 0._rt, Ez_cc = 0._rt;
                        amrex::Real Bx_cc = 0._rt, By_cc = 0._rt, Bz_cc = 0._rt;

                        if (normal_dir == 1 || normal_dir == 2) {
                            Ex_cc = ablastr::coarsen::sample::Interp(Ex_arr, Ex_stag, cc, cr, i, j, k, comp);
                            Bx_cc = ablastr::coarsen::sample::Interp(Bx_arr, Bx_stag, cc, cr, i, j, k, comp);
                        }

                        if (normal_dir == 0 || normal_dir == 2) {
                            Ey_cc = ablastr::coarsen::sample::Interp(Ey_arr, Ey_stag, cc, cr, i, j, k, comp);
                            By_cc = ablastr::coarsen::sample::Interp(By_arr, By_stag, cc, cr, i, j, k, comp);
                        }
                        if (normal_dir == 0 || normal_dir == 1) {
                            Ez_cc = ablastr::coarsen::sample::Interp(Ez_arr, Ez_stag, cc, cr, i, j, k, comp);
                            Bz_cc = ablastr::coarsen::sample::Interp(Bz_arr, Bz_stag, cc, cr, i, j, k, comp);
                        }

                        amrex::Real const af = area_factor(i,j,k);
                        if      (normal_dir == 0) { return af*(Ey_cc * Bz_cc - Ez_cc * By_cc); }
                        else if (normal_dir == 1) { return af*(Ez_cc * Bx_cc - Ex_cc * Bz_cc); }
                        else                      { return af*(Ex_cc * By_cc - Ey_cc * Bx_cc); }
                    });
            }
        }

        int const sign = (face().isLow() ? -1 : 1);
        auto r = reduce_data.value();
        int const ii = int(face());
        m_data[ii] = sign*amrex::get<0>(r)/PhysConst::mu0*dA;

    }

    amrex::ParallelDescriptor::ReduceRealSum(m_data.data(), 2*AMREX_SPACEDIM);

    amrex::Real const dt = warpx.getdt(lev);
    for (int ii=0 ; ii < 2*AMREX_SPACEDIM ; ii++) {
        m_data[ii + 2*AMREX_SPACEDIM] += m_data[ii]*dt;
    }

}

void
FieldPoyntingFlux::WriteCheckpointData (std::string const & dir)
{
    // Write out the current values of the time integrated data
    std::ofstream chkfile{dir + "/FieldPoyntingFlux_data.txt", std::ofstream::out};
    if (!chkfile.good()) {
        WARPX_ABORT_WITH_MESSAGE("FieldPoyntingFlux::WriteCheckpointData: could not open file for writing checkpoint data");
    }

    chkfile.precision(17);

    for (int i=0; i < 2*AMREX_SPACEDIM; i++) {
        chkfile << m_data[2*AMREX_SPACEDIM + i] << "\n";
    }
}

void
FieldPoyntingFlux::ReadCheckpointData (std::string const & dir)
{
    // Read in the current values of the time integrated data
    std::ifstream chkfile{dir + "/FieldPoyntingFlux_data.txt", std::ifstream::in};
    if (!chkfile.good()) {
        WARPX_ABORT_WITH_MESSAGE("FieldPoyntingFlux::ReadCheckpointData: could not open file for reading checkpoint data");
    }

    for (int i=0; i < 2*AMREX_SPACEDIM; i++) {
        amrex::Real data;
        if (chkfile >> data) {
            m_data[2*AMREX_SPACEDIM + i] = data;
        } else {
            WARPX_ABORT_WITH_MESSAGE("FieldPoyntingFlux::ReadCheckpointData: could not read in time integrated data");
        }
    }
}
