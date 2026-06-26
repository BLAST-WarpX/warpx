/* Copyright 2021 Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "BackTransformFunctor.H"

#include "Diagnostics/ComputeDiagFunctors/ComputeDiagFunctor.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/utils/Communication.H>

#include <AMReX_Array4.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

#include <cmath>
#include <map>
#include <memory>

using namespace amrex;

BackTransformFunctor::BackTransformFunctor (amrex::MultiFab const * mf_src, int lev,
                                            const int ncomp, const int num_buffers,
                                            amrex::Vector< std::string > varnames,
                                            amrex::Vector< std::string > varnames_fields,
                                            const amrex::IntVect crse_ratio
                                            ):
    ComputeDiagFunctor(ncomp, crse_ratio),
    m_mf_src{mf_src}, m_lev{lev}, m_num_buffers{num_buffers},
    m_varnames{std::move(varnames)}, m_varnames_fields{std::move(varnames_fields)}
{
    InitData();
}

void
BackTransformFunctor::operator ()(amrex::MultiFab& mf_dst, int /*dcomp*/, const int i_buffer) const
{
    // Perform back-transformation only if z slice is within the domain stored as 0/1
    // in m_perform_backtransform[i_buffer]
    if ( m_perform_backtransform[i_buffer] == 1) {
        auto& warpx = WarpX::GetInstance();
        auto geom = warpx.Geom(m_lev);
        const amrex::Real gamma_boost = WarpX::gamma_boost;
        const int moving_window_dir = WarpX::moving_window_dir;
        const amrex::Real beta_boost = std::sqrt( 1._rt - 1._rt/( gamma_boost * gamma_boost) );
        const bool interpolate = true;
        std::unique_ptr< amrex::MultiFab > slice = nullptr;
        const int scomp = 0;

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_mf_src != nullptr, "m_mf_src can't be a nullptr.");
        AMREX_ASSUME(m_mf_src != nullptr);

        // Generate slice of the cell-centered multifab containing boosted-frame field-data
        // at current z-boost location for the ith buffer
        slice = amrex::get_slice_data(
            moving_window_dir,
            m_current_z_boost[i_buffer],
            *m_mf_src,
            geom,
            scomp,
            m_mf_src->nComp(),
            interpolate);


        // Perform in-place Lorentz-transform of all the fields stored in the slice.
        LorentzTransformZ( *slice, gamma_boost, beta_boost);

        // Create a 2D box for the slice in the boosted frame
        const amrex::Real dx = geom.CellSize(moving_window_dir);
        // index corresponding to z_boost location in the boost-frame
        const int i_boost = static_cast<int> ( ( m_current_z_boost[i_buffer]
                                            - geom.ProbLo(moving_window_dir) ) / dx );
        // z-Slice at i_boost with x,y indices same as buffer_box
        amrex::Box slice_box = m_buffer_box[i_buffer];
        slice_box.setSmall(moving_window_dir, i_boost);
        slice_box.setBig(moving_window_dir, i_boost);

        // Make it a BoxArray
        const amrex::BoxArray slice_ba(slice_box);
        // Define MultiFab with the distribution map of the destination multifab and
        // containing all ten components that were in the slice generated from m_mf_src.
        std::unique_ptr< amrex::MultiFab > tmp_slice_ptr = nullptr;
        tmp_slice_ptr = std::make_unique<MultiFab> ( slice_ba, mf_dst.DistributionMap(),
                                                     slice->nComp(), 0 );
        tmp_slice_ptr->setVal(0.0);
        // Parallel copy the lab-frame data from "slice" MultiFab with
        // ncomp=10 and boosted-frame dmap to "tmp_slice_ptr" MultiFab with
        // ncomp=10 and dmap of the destination Multifab, which will store the final data
        ablastr::utils::communication::ParallelCopy(*tmp_slice_ptr, *slice, 0, 0, slice->nComp(),
                                                    IntVect(AMREX_D_DECL(0, 0, 0)),
                                                    IntVect(AMREX_D_DECL(0, 0, 0)),
                                                    WarpX::do_single_precision_comms);
        // Now we will cherry pick only the user-defined fields from
        // tmp_slice_ptr to dst_mf
        const int k_lab = m_k_index_zlab[i_buffer];
        const int ncomp_dst = mf_dst.nComp();
        amrex::MultiFab& tmp = *tmp_slice_ptr;
#ifdef AMREX_USE_GPU
        Gpu::DeviceVector<int> d_map_varnames(m_map_varnames.size());
        Gpu::copyAsync(Gpu::hostToDevice,
                       m_map_varnames.begin(), m_map_varnames.end(),
                       d_map_varnames.begin());
        Gpu::synchronize();
        int const* field_map_ptr = d_map_varnames.dataPtr();
#else
        int const* field_map_ptr = m_map_varnames.dataPtr();
#endif
        for (amrex::MFIter mfi(tmp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& tbx = mfi.tilebox();
            const amrex::Array4<amrex::Real> src_arr = tmp[mfi].array();
            const amrex::Array4<amrex::Real> dst_arr = mf_dst[mfi].array();
#ifdef WARPX_DIM_RZ
            const int n_rz_comp = WarpX::ncomps;
#endif
            amrex::ParallelFor( tbx, ncomp_dst,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    // Field id that corresponds to the nth user-requested component
                    const int icomp = field_map_ptr[n];
#if defined(WARPX_DIM_3D)
                    dst_arr(i, j, k_lab, n) = src_arr(i, j, k, icomp);
#elif defined(WARPX_DIM_XZ)
                    dst_arr(i, k_lab, k, n) = src_arr(i, j, k, icomp);
#elif defined(WARPX_DIM_RZ)
                    // rzcomp below gives the component id, 0 to (n_rz_comp-1) for a given field
                    const int rzcomp = n % n_rz_comp;
                    // Accessing the correct rz component from the cell-centered multifab
                    // that has back-transformed fields and storing it for the appropriate user-requested field, icomp
                    // For example, for 2 rz modes, we have three components (n_rz_comp=3) for each field
                    // If n = 4 gives icomp = 1 (for Et) obtained from field_map_ptr,
                    //           rzcomp = 4 - int(floor(4/3))*3 = 4 - 3 = 1
                    // Thus we are accessing real component of mode 1 of Et (note that modes go from 0 to 1)
                    // Since the fields are stored contiguously in src_arr, icomp*n_rz_comp + rz_comp accesses
                    // real part of mode 1 for Et (1*3+1) = 4
                    dst_arr(i, k_lab, k, n) = src_arr(i, j, k, icomp*n_rz_comp+rzcomp);
#else
                    dst_arr(k_lab, j, k, n) = src_arr(i, j, k, icomp);
#endif
                } );
        }

        // Reset the temporary MultiFabs generated
        slice = nullptr;
        tmp_slice_ptr = nullptr;
    }

}

void
BackTransformFunctor::PrepareFunctorData (int i_buffer,
                          bool z_slice_in_domain, amrex::Real current_z_boost,
                          amrex::Box buffer_box, const int k_index_zlab,
                          const int snapshot_full)
{
    m_buffer_box[i_buffer] = buffer_box;
    m_current_z_boost[i_buffer] = current_z_boost;
    m_k_index_zlab[i_buffer] = k_index_zlab;
    m_perform_backtransform[i_buffer] = 0;
    if (z_slice_in_domain && (snapshot_full == 0)) { m_perform_backtransform[i_buffer] = 1; }
}

void
BackTransformFunctor::InitData ()
{

    m_buffer_box.resize( m_num_buffers );
    m_current_z_boost.resize( m_num_buffers );
    m_perform_backtransform.resize( m_num_buffers );
    m_k_index_zlab.resize( m_num_buffers );
    m_map_varnames.resize( m_varnames.size() );

#ifdef WARPX_DIM_RZ
    std::map<std::string, int> m_possible_fields_to_dump = {
        {"Er", 0},
        {"Et", 1},
        {"Ez", 2},
        {"Br", 3},
        {"Bt", 4},
        {"Bz", 5},
        {"jr", 6},
        {"jt", 7},
        {"jz", 8},
        {"rho", 9}
    };
#else
    std::map<std::string, int> m_possible_fields_to_dump = {
        {"Ex", 0},
        {"Ey", 1},
        {"Ez", 2},
        {"Bx", 3},
        {"By", 4},
        {"Bz", 5},
        {"jx", 6},
        {"jy", 7},
        {"jz", 8},
        {"rho", 9}
    };
#endif

    for (int i = 0; i < m_varnames.size(); ++i)
    {
#ifdef WARPX_DIM_RZ
        const int field_id = i / WarpX::ncomps;
        m_map_varnames[i] = m_possible_fields_to_dump[ m_varnames_fields[field_id] ];
#else
        m_map_varnames[i] = m_possible_fields_to_dump[ m_varnames[i] ] ;
#endif
    }

}

void
BackTransformFunctor::LorentzTransformZ (amrex::MultiFab& data, amrex::Real gamma_boost,
                                         amrex::Real beta_boost) const
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(data, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& tbx = mfi.tilebox();
        const amrex::Array4< amrex::Real > arr = data[mfi].array();
        const amrex::Real clight = PhysConst::c;
        const amrex::Real inv_clight = 1.0_rt/clight;
#ifdef WARPX_DIM_RZ
        const int n_rcomps = WarpX::ncomps;
        amrex::ParallelFor( tbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                for (int mode_comp = 0; mode_comp < n_rcomps; ++mode_comp) {
                    // Back-transform the transverse electric and magnetic fields.
                    // Note that the z-components, Ez, Bz, are not changed by the transform.

                    // Transform Er_boost & Bt_boost to lab-frame for corresponding mode (mode_comp)
                    const amrex::Real er_lab = gamma_boost * ( arr(i, j, k, n_rcomps*0 + mode_comp)
                                            + beta_boost * clight * arr(i, j, k, n_rcomps*4+ mode_comp) );
                    const amrex::Real bt_lab = gamma_boost * ( arr(i, j, k, n_rcomps*4 + mode_comp)
                                            + beta_boost * inv_clight * arr(i, j, k, n_rcomps*0 + mode_comp) );
                    // Store lab-frame data in-place
                    arr(i, j, k, n_rcomps*0 + mode_comp) = er_lab;
                    arr(i, j, k, n_rcomps*4 + mode_comp) = bt_lab;

                    // Transform Et_boost & Br_boost to lab-frame for corresponding mode (mode_comp)
                    const amrex::Real et_lab = gamma_boost * ( arr(i, j, k, n_rcomps*1 + mode_comp)
                                            - beta_boost * clight * arr(i, j, k, n_rcomps*3 + mode_comp) );
                    const amrex::Real br_lab = gamma_boost * ( arr(i, j, k, n_rcomps*3 + mode_comp)
                                            - beta_boost * inv_clight * arr(i, j, k, n_rcomps*1 + mode_comp) );
                    // Store lab-frame data in-place
                    arr(i, j, k, n_rcomps*1 + mode_comp) = et_lab;
                    arr(i, j, k, n_rcomps*3 + mode_comp) = br_lab;

                    // Transform charge density z-component of current density
                    const amrex::Real j_lab = gamma_boost * ( arr(i, j, k, n_rcomps*8 + mode_comp)
                                            + beta_boost * clight * arr(i, j, k, n_rcomps*9 + mode_comp) );
                    const amrex::Real rho_lab = gamma_boost * ( arr(i, j, k, n_rcomps*9 + mode_comp)
                                              + beta_boost * inv_clight * arr(i, j, k, n_rcomps*8 + mode_comp) );
                    // Store lab-frame jz and rho in-place
                    arr(i, j, k, n_rcomps*8 + mode_comp) = j_lab;
                    arr(i, j, k, n_rcomps*9 + mode_comp) = rho_lab;
                }
            }
        );
#else
        // arr(x,y,z,comp) has ten-components namely,
        // Ex Ey Ez Bx By Bz jx jy jz rho in that order.
        amrex::ParallelFor( tbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // Back-transform the transverse electric and magnetic fields.
                // Note that the z-components, Ez, Bz, are not changed by the transform.

                // Transform Ex_boost (ncomp=0) & By_boost (ncomp=4) to lab-frame
                const amrex::Real ex_lab = gamma_boost * ( arr(i, j, k, 0)
                                        + beta_boost * clight * arr(i, j, k, 4) );
                const amrex::Real by_lab = gamma_boost * ( arr(i, j, k, 4)
                                        + beta_boost * inv_clight * arr(i, j, k, 0) );
                // Store lab-frame data in-place
                arr(i, j, k, 0) = ex_lab;
                arr(i, j, k, 4) = by_lab;

                // Transform Ey_boost (ncomp=1) & Bx_boost (ncomp=3) to lab-frame
                const amrex::Real ey_lab = gamma_boost * ( arr(i, j, k, 1)
                                        - beta_boost * clight * arr(i, j, k, 3) );
                const amrex::Real bx_lab = gamma_boost * ( arr(i, j, k, 3)
                                        - beta_boost * inv_clight * arr(i, j, k, 1) );
                // Store lab-frame data in-place
                arr(i, j, k, 1) = ey_lab;
                arr(i, j, k, 3) = bx_lab;

                // Transform charge density (ncomp=9)
                // and z-component of current density (ncomp=8)
                const amrex::Real j_lab = gamma_boost * ( arr(i, j, k, 8)
                                        + beta_boost * clight * arr(i, j, k, 9) );
                const amrex::Real rho_lab = gamma_boost * ( arr(i, j, k, 9)
                                          + beta_boost * inv_clight * arr(i, j, k, 8) );
                // Store lab-frame jz and rho in-place
                arr(i, j, k, 8) = j_lab;
                arr(i, j, k, 9) = rho_lab;
            }
        );
#endif
    }

}
