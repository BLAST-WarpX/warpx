/* Copyright 2019-2020 Luca Fedeli, Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "NCIGodfreyFilter.H"

#include "Utils/NCIGodfreyTables.H"
#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_Algorithm.H>
#include <AMReX_BLassert.H>
#include <AMReX_Config.H>
#include <AMReX_Dim3.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_IntVect.H>
#include <AMReX_Vector.H>

#include <vector>

using namespace amrex;

//NCIGodfreyFilter not implemented in 1D
#if (AMREX_SPACEDIM >= 2)

NCIGodfreyFilter::NCIGodfreyFilter(godfrey_coeff_set coeff_set, amrex::Real cdtodz, bool nodal_gather):
    m_coeff_set{coeff_set}, // Store parameters into class data members
    m_cdtodz{cdtodz},
    m_nodal_gather{nodal_gather}
{
    // NCI Godfrey filter has fixed size, and is applied along z only.
# if defined(WARPX_DIM_3D)
    stencil_length_each_dir = {1,1,5};
    slen = {1,1,5};
# else
    stencil_length_each_dir = {1,5};
    slen = {1,5,1};
# endif
}

void NCIGodfreyFilter::ComputeStencils()
{
    using namespace warpx::nci_godfrey;

    // Sanity checks: filter length should be 5 in z
#  if  defined(WARPX_DIM_3D)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        slen.z==5,"ERROR: NCI filter requires 5 points in z");
#  else
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        slen.y==5,"ERROR: NCI filter requires 5 points in z");
#  endif
    // Interpolate coefficients from the table, and store into prestencil.
    auto index = static_cast<int>(tab_length*m_cdtodz);
    index = min(index, tab_length-2);
    index = max(index, 0);
    Real const weight_right = m_cdtodz - amrex::Real(index)/amrex::Real(tab_length);
    Real prestencil[4];

    // read prestencil coefficients from table (the stencil is computed from
    // these coefficients)
    for(int i=0; i<tab_width; i++){
        if (!m_nodal_gather)
        {
            // If gather from staggered grid, use coefficients for Galerkin gather
            if        (m_coeff_set == godfrey_coeff_set::Ex_Ey_Bz){
                // Set of coefficients for Ex, Ey and Bz
                prestencil[i] = (1_rt-weight_right)*table_nci_godfrey_galerkin_Ex_Ey_Bz[index  ][i] +
                                   weight_right    *table_nci_godfrey_galerkin_Ex_Ey_Bz[index+1][i];
            } else if (m_coeff_set == godfrey_coeff_set::Bx_By_Ez){
                // Set of coefficients for Bx, By and Ez
                prestencil[i] = (1_rt-weight_right)*table_nci_godfrey_galerkin_Bx_By_Ez[index  ][i] +
                                   weight_right    *table_nci_godfrey_galerkin_Bx_By_Ez[index+1][i];
            } else {
                WARPX_ABORT_WITH_MESSAGE(
                    "m_coeff_set must be godfrey_coeff_set::Ex_Ey_Bz or godfrey_coeff_set::Bx_By_Ez");
            }
        }
        else
        {
            // If gather from node-centered grid, use coefficients for momentum-conserving gather
            if        (m_coeff_set == godfrey_coeff_set::Ex_Ey_Bz){
                // Set of coefficients for Ex, Ey and Bz
                prestencil[i] = (1_rt-weight_right)*table_nci_godfrey_momentum_Ex_Ey_Bz[index  ][i] +
                                   weight_right    *table_nci_godfrey_momentum_Ex_Ey_Bz[index+1][i];
            } else if (m_coeff_set == godfrey_coeff_set::Bx_By_Ez) {
                // Set of coefficients for Bx, By and Ez
                prestencil[i] = (1_rt-weight_right)*table_nci_godfrey_momentum_Bx_By_Ez[index  ][i] +
                                   weight_right    *table_nci_godfrey_momentum_Bx_By_Ez[index+1][i];
            } else {
                WARPX_ABORT_WITH_MESSAGE(
                    "m_coeff_set must be godfrey_coeff_set::Ex_Ey_Bz or godfrey_coeff_set::Bx_By_Ez");
            }
        }
    }
    // Compute stencil_z
    Vector<Real> h_stencil_z(5);
    h_stencil_z[0] =  (256 + 128*prestencil[0] + 96*prestencil[1] + 80*prestencil[2] + 70*prestencil[3]) / 256;
    h_stencil_z[1] = -(       64*prestencil[0] + 64*prestencil[1] + 60*prestencil[2] + 56*prestencil[3]) / 256;
    h_stencil_z[2] =  (                          16*prestencil[1] + 24*prestencil[2] + 28*prestencil[3]) / 256;
    h_stencil_z[3] = -(                                              4*prestencil[2] +  8*prestencil[3]) / 256;
    h_stencil_z[4] =  (                                                                 1*prestencil[3]) / 256;

    // Compute h_stencil_x and h_stencil_y (no filter in these directions,
    // so only 1 coeff, equal to 1)
    Vector<Real> h_stencil_x(1);
    h_stencil_x[0] = 1._rt;
#  if defined(WARPX_DIM_3D)
    Vector<Real> h_stencil_y(1);
    h_stencil_y[0] = 1._rt;
#  endif

    // Due to the way Filter::DoFilter() is written,
    // coefficient 0 has to be /2
    h_stencil_x[0] /= 2._rt;
#  if defined(WARPX_DIM_3D)
    h_stencil_y[0] /= 2._rt;
#  endif
    h_stencil_z[0] /= 2._rt;

    m_stencil_0.resize(h_stencil_x.size());
    Gpu::copyAsync(Gpu::hostToDevice,h_stencil_x.begin(),h_stencil_x.end(),m_stencil_0.begin());
#  if defined(WARPX_DIM_3D)
    m_stencil_1.resize(h_stencil_y.size());
    m_stencil_2.resize(h_stencil_z.size());
    Gpu::copyAsync(Gpu::hostToDevice,h_stencil_y.begin(),h_stencil_y.end(),m_stencil_1.begin());
    Gpu::copyAsync(Gpu::hostToDevice,h_stencil_z.begin(),h_stencil_z.end(),m_stencil_2.begin());
#  elif (AMREX_SPACEDIM == 2)
    // In 2D, the filter applies stencil_1 to the 2nd dimension
    m_stencil_1.resize(h_stencil_z.size());
    Gpu::copyAsync(Gpu::hostToDevice,h_stencil_z.begin(),h_stencil_z.end(),m_stencil_1.begin());
#  endif

    Gpu::synchronize();
}

#else

NCIGodfreyFilter::NCIGodfreyFilter(godfrey_coeff_set, amrex::Real, bool)
{
    WARPX_ABORT_WITH_MESSAGE(
        "NCIGodfreyFilter not implemented in 1D!");
}

void NCIGodfreyFilter::ComputeStencils()
{
    WARPX_ABORT_WITH_MESSAGE(
        "NCIGodfreyFilter not implemented in 1D!");
}

#endif
