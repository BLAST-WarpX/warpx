/* Copyright 2019 Andrew Myers, Maxence Thevenet, Weiqun Zhang
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "BilinearFilter.H"

#include "Utils/WarpXProfilerWrapper.H"

#include <AMReX_Config.H>
#include <AMReX_Dim3.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <array>
#include <vector>

using namespace amrex;

namespace {
    void compute_stencil(Gpu::DeviceVector<Real> &stencil, unsigned int npass)
    {
        Vector<Real> old_s(1u+npass,0.);
        Vector<Real> new_s(1u+npass,0.);

        old_s.at(0) = 1._rt;
        int jmax = 1;
        // Convolve the filter with itself npass times
        int const lastpass = static_cast<int>(npass+1u);
        for(int ipass=1; ipass< lastpass; ipass++){
            // element 0 has to be treated in its own way
            new_s.at(0) = 0.5_rt * old_s.at(0);
            if (1<jmax) { new_s.at(0) += 0.5_rt * old_s.at(1); }
            amrex::Real loc = 0._rt;
            // For each element j, apply the filter to
            // old_s to get new_s[j]. loc stores the tmp
            // filtered value.
            for(int j=1; j<jmax+1; j++){
                loc = 0.5_rt * old_s[j];
                loc += 0.25_rt * old_s[j-1];
                if (j<jmax) { loc += 0.25_rt * old_s.at(j+1); }
                new_s.at(j) = loc;
            }
            // copy new_s into old_s
            old_s = new_s;
            // extend the stencil length for next iteration
            jmax += 1;
        }
        // we use old_s here to make sure the stencil
        // is corrent even when npass = 0
        old_s.at(0) *= 0.5_rt; // because we will use it twice
        stencil.resize(old_s.size());
        Gpu::copyAsync(Gpu::hostToDevice,old_s.begin(),old_s.end(),stencil.begin());
        amrex::Gpu::synchronize();
    }
}

void BilinearFilter::ComputeStencils(){
    WARPX_PROFILE("BilinearFilter::ComputeStencils()");
    int i = 0;
    for (const auto& el : npass_each_dir ) {
        stencil_length_each_dir[i++] = static_cast<int>(el) + 1;
    }

    m_stencil_0.resize( 1u + npass_each_dir[0] );
    compute_stencil(m_stencil_0, npass_each_dir[0]);
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_3D)
    m_stencil_1.resize( 1u + npass_each_dir[1] );
    compute_stencil(m_stencil_1, npass_each_dir[1]);
#endif
#if defined(WARPX_DIM_3D)
    m_stencil_2.resize( 1u + npass_each_dir[2] );
    compute_stencil(m_stencil_2, npass_each_dir[2]);
#endif

    slen = stencil_length_each_dir.dim3();
#if defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    slen.z = 1;
#endif
#if defined(WARPX_DIM_1D_Z)
    slen.y = 1;
#endif
}
