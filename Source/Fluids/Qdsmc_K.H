/* Copyright 2024 Marco Acciarri (Helion Energy Inc.)
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_Qdsmc_K_h
#define WARPX_Qdsmc_K_h

#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_FArrayBox.H>

#include <cmath>


/**
 * Qdsmc particles x0,y0,z0 positions should at nodes positions
 * at the beginning of every step, so the field gathering
 * does not require any interpolation, it only needs to read
 * the values from the nodes.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void gather_density_entropy(const amrex::ParticleReal x0p,
                            const amrex::ParticleReal y0p,
                            const amrex::ParticleReal z0p,
                            amrex::Real& n_p,
                            amrex::Real& kn_p,
                            amrex::Array4<amrex::Real const> const& rho_arr,
                            amrex::Array4<amrex::Real const> const& K_arr,
                            amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
                            const amrex::XDim3 & dinv,
                            const amrex::Real cell_volume,
                            const amrex::Box& bx)
{
    amrex::Real x = (x0p - plo[0])*dinv.x - 0.5;
    amrex::Real y = (y0p - plo[1])*dinv.y - 0.5;
    amrex::Real z = (z0p - plo[2])*dinv.z - 0.5;

    int i0p = std::floor(x);
    int j0p = std::floor(y);
    int k0p = std::floor(z);

    amrex::IntVect c(i0p, j0p, k0p);
    if (bx.contains(c)){ // Check if the cell is within the tile's box
        n_p = rho_arr(i0p,j0p,k0p)*cell_volume/PhysConst::q_e;
        kn_p = K_arr(i0p,j0p,k0p)*n_p;
    }     
}


/**
 * Qdsmc particles x0,y0,z0 positions should be at nodes positions
 * at the beginning of every step, so the field gathering
 * does not require any interpolation, it only needs to read
 * the values from the nodes.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void gather_vector_field_qdsmc(const amrex::ParticleReal x0p,
                            const amrex::ParticleReal y0p,
                            const amrex::ParticleReal z0p,
                            amrex::Real& field_x_p,
                            amrex::Real& field_y_p,
                            amrex::Real& field_z_p,
                            amrex::Array4<amrex::Real const> const& field_x_arr,
                            amrex::Array4<amrex::Real const> const& field_y_arr,
                            amrex::Array4<amrex::Real const> const& field_z_arr,
                            amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
                            const amrex::XDim3 & dinv,
                            const amrex::Box& bx)
{
    amrex::Real x = (x0p - plo[0])*dinv.x - 0.5;
    amrex::Real y = (y0p - plo[1])*dinv.y - 0.5;
    amrex::Real z = (z0p - plo[2])*dinv.z - 0.5;

    int i0p = std::floor(x);
    int j0p = std::floor(y);
    int k0p = std::floor(z);

    amrex::IntVect c(i0p, j0p, k0p);
    if (bx.contains(c)){ // Check if the cell is within the tile's box
        field_x_p = field_x_arr(i0p,j0p,k0p);
        field_y_p = field_y_arr(i0p,j0p,k0p);
        field_z_p = field_z_arr(i0p,j0p,k0p);
    }    

}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void push_qdsmc_particle(const amrex::ParticleReal x0p,
                        const amrex::ParticleReal y0p,
                        const amrex::ParticleReal z0p,
                        const amrex::ParticleReal vxp,
                        const amrex::ParticleReal vyp,
                        const amrex::ParticleReal vzp,
                        amrex::Real& xp,
                        amrex::Real& yp,
                        amrex::Real& zp,
                        amrex::Real dt)
{
    xp = x0p + vxp*dt;
    yp = y0p + vyp*dt;
    zp = z0p + vzp*dt;
}


/**
 * General kernel for scalar quantity deposition from
 * fictitious qdsmc particles. Receives a scalar property
 * from a fictitious particle and deposits onto a scalar field.
 * Only linear interpolation for now.
 * Using nodal grid.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void do_deposit_scalar(amrex::Array4<amrex::Real> const& scalar_field,
                    const amrex::ParticleReal xp,
                    const amrex::ParticleReal yp,
                    const amrex::ParticleReal zp,
                    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
                    const amrex::XDim3 & dinv,
                    const amrex::ParticleReal valuep,
                    const amrex::Box& bx)
{
    // calculate di, dj, dk for particle in cell
    amrex::Real x = (xp - plo[0])*dinv.x - 0.5;
    amrex::Real y = (yp - plo[1])*dinv.y - 0.5;
    amrex::Real z = (zp - plo[2])*dinv.z - 0.5;

    int i0p = std::floor(x);
    int j0p = std::floor(y);
    int k0p = std::floor(z);

    amrex::Real dip = x - i0p;
    amrex::Real djp = y - j0p;
    amrex::Real dkp = z - k0p;

    amrex:: Real sx[] = {1.-dip, dip};
    amrex:: Real sy[] = {1.-djp, djp};
    amrex:: Real sz[] = {1.-dkp, dkp};

    // do triple for loop, moving over i,j,k
    for         (int ioff = 0; ioff < 2; ++ioff){
        for     (int joff = 0; joff < 2; ++joff){
            for (int koff = 0; koff < 2; ++koff){

                amrex::IntVect c(i0p+ioff, j0p+joff, k0p+koff);
                if (bx.contains(c)){ // Check if the cell is within the tile's box
                    amrex::Gpu::Atomic::AddNoRet(&scalar_field(i0p+ioff,j0p+joff,k0p+koff), valuep*sx[ioff]*sy[joff]*sz[koff]);
                }

            }
        }
    }
}

#endif //WARPX_Qdsmc_K_h