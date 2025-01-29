/* Copyright 2019 Remi Lehe, Revathi Jambunathan, Revathi Jambunathan
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PML_KERNELS_H_
#define WARPX_PML_KERNELS_H_

#include "BoundaryConditions/PMLComponent.H"
#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_FArrayBox.H>

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void warpx_damp_pml_ex (int i, int j, int k, amrex::Array4<amrex::Real> const& Ex,
                        const amrex::IntVect& Ex_stag,
                        const amrex::Real* const sigma_fac_x,
                        const amrex::Real* const sigma_fac_y,
                        const amrex::Real* const sigma_fac_z,
                        const amrex::Real* const sigma_star_fac_x,
                        const amrex::Real* const sigma_star_fac_y,
                        const amrex::Real* const sigma_star_fac_z,
                        int xlo, int ylo, int zlo,
                        const bool dive_cleaning)
{
#if defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(i, j, k, Ex, Ex_stag, sigma_fac_x, sigma_fac_y, sigma_fac_z,
           sigma_star_fac_x, sigma_star_fac_y, sigma_star_fac_z, xlo, ylo, zlo, dive_cleaning);
    amrex::Abort("PML not implemented in Cartesian 1D geometry");
#endif

#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

    amrex::ignore_unused(sigma_fac_y, sigma_star_fac_y, ylo);

    // sx = 0 means that Ex is staggered in x, while sx = 1 means that Ex is nodal in x (same for z)
    const int sx = Ex_stag[0];
    const int sz = Ex_stag[1];

    if (dive_cleaning)
    {
        // Exx
        if (sx == 0) {
            Ex(i,j,k,PMLComp::xx) *= sigma_star_fac_x[i-xlo];
        } else {
            Ex(i,j,k,PMLComp::xx) *= sigma_fac_x[i-xlo];
        }
    }

    // Exz
    if (sz == 0) {
        Ex(i,j,k,PMLComp::xz) *= sigma_star_fac_z[j-zlo];
    } else {
        Ex(i,j,k,PMLComp::xz) *= sigma_fac_z[j-zlo];
    }

#elif defined(WARPX_DIM_3D)

    // sx = 0 means that Ex is staggered in x, while sx = 1 means that Ex is nodal in x (same for y, z)
    const int sx = Ex_stag[0];
    const int sy = Ex_stag[1];
    const int sz = Ex_stag[2];

    if (dive_cleaning)
    {
        // Exx
        if (sx == 0) {
            Ex(i,j,k,PMLComp::xx) *= sigma_star_fac_x[i-xlo];
        } else {
            Ex(i,j,k,PMLComp::xx) *= sigma_fac_x[i-xlo];
        }
    }

    // Exy
    if (sy == 0) {
        Ex(i,j,k,PMLComp::xy) *= sigma_star_fac_y[j-ylo];
    } else {
        Ex(i,j,k,PMLComp::xy) *= sigma_fac_y[j-ylo];
    }

    // Exz
    if (sz == 0) {
        Ex(i,j,k,PMLComp::xz) *= sigma_star_fac_z[k-zlo];
    } else {
        Ex(i,j,k,PMLComp::xz) *= sigma_fac_z[k-zlo];
    }

#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void warpx_damp_pml_ey (int i, int j, int k, amrex::Array4<amrex::Real> const& Ey,
                        const amrex::IntVect& Ey_stag,
                        const amrex::Real* const sigma_fac_x,
                        const amrex::Real* const sigma_fac_y,
                        const amrex::Real* const sigma_fac_z,
                        const amrex::Real* const sigma_star_fac_x,
                        const amrex::Real* const sigma_star_fac_y,
                        const amrex::Real* const sigma_star_fac_z,
                        int xlo, int ylo, int zlo,
                        const bool dive_cleaning)
{
#if defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(i, j, k, Ey, Ey_stag, sigma_fac_x, sigma_fac_y, sigma_fac_z,
           sigma_star_fac_x, sigma_star_fac_y, sigma_star_fac_z, xlo, ylo, zlo, dive_cleaning);
    amrex::Abort("PML not implemented in Cartesian 1D geometry");
#endif

#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

    amrex::ignore_unused(sigma_fac_y, sigma_star_fac_y, ylo, dive_cleaning);

    // sx = 0 means that Ey is staggered in x, while sx = 1 means that Ey is nodal in x (same for z)
    const int sx = Ey_stag[0];
    const int sz = Ey_stag[1];

    // Eyx
    if (sx == 0) {
        Ey(i,j,k,PMLComp::yx) *= sigma_star_fac_x[i-xlo];
    } else {
        Ey(i,j,k,PMLComp::yx) *= sigma_fac_x[i-xlo];
    }

    // Eyz
    if (sz == 0) {
        Ey(i,j,k,PMLComp::yz) *= sigma_star_fac_z[j-zlo];
    } else {
        Ey(i,j,k,PMLComp::yz) *= sigma_fac_z[j-zlo];
    }

#elif defined(WARPX_DIM_3D)

    // sx = 0 means that Ey is staggered in x, while sx = 1 means that Ey is nodal in x (same for y, z)
    const int sx = Ey_stag[0];
    const int sy = Ey_stag[1];
    const int sz = Ey_stag[2];

    // Eyx
    if (sx == 0) {
        Ey(i,j,k,PMLComp::yx) *= sigma_star_fac_x[i-xlo];
    } else {
        Ey(i,j,k,PMLComp::yx) *= sigma_fac_x[i-xlo];
    }

    if (dive_cleaning)
    {
        // Eyy
        if (sy == 0) {
            Ey(i,j,k,PMLComp::yy) *= sigma_star_fac_y[j-ylo];
        } else {
            Ey(i,j,k,PMLComp::yy) *= sigma_fac_y[j-ylo];
        }
    }

    // Eyz
    if (sz == 0) {
        Ey(i,j,k,PMLComp::yz) *= sigma_star_fac_z[k-zlo];
    } else {
        Ey(i,j,k,PMLComp::yz) *= sigma_fac_z[k-zlo];
    }

#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void warpx_damp_pml_ez (int i, int j, int k, amrex::Array4<amrex::Real> const& Ez,
                        const amrex::IntVect& Ez_stag,
                        const amrex::Real* const sigma_fac_x,
                        const amrex::Real* const sigma_fac_y,
                        const amrex::Real* const sigma_fac_z,
                        const amrex::Real* const sigma_star_fac_x,
                        const amrex::Real* const sigma_star_fac_y,
                        const amrex::Real* const sigma_star_fac_z,
                        int xlo, int ylo, int zlo,
                        const bool dive_cleaning)
{
#if defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(i, j, k, Ez, Ez_stag, sigma_fac_x, sigma_fac_y, sigma_fac_z,
           sigma_star_fac_x, sigma_star_fac_y, sigma_star_fac_z, xlo, ylo, zlo, dive_cleaning);
    amrex::Abort("PML not implemented in Cartesian 1D geometry");
#endif

#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

    amrex::ignore_unused(sigma_fac_y, sigma_star_fac_y, ylo);

    // sx = 0 means that Ez is staggered in x, while sx = 1 means that Ez is nodal in x (same for z)
    const int sx = Ez_stag[0];
    const int sz = Ez_stag[1];

    // Ezx
    if (sx == 0) {
        Ez(i,j,k,PMLComp::zx) *= sigma_star_fac_x[i-xlo];
    } else {
        Ez(i,j,k,PMLComp::zx) *= sigma_fac_x[i-xlo];
    }

    if (dive_cleaning)
    {
        // Ezz
        if (sz == 0) {
            Ez(i,j,k,PMLComp::zz) *= sigma_star_fac_z[j-zlo];
        } else {
            Ez(i,j,k,PMLComp::zz) *= sigma_fac_z[j-zlo];
        }
    }

#elif defined(WARPX_DIM_3D)

    // sx = 0 means that Ez is staggered in x, while sx = 1 means that Ez is nodal in x (same for y, z)
    const int sx = Ez_stag[0];
    const int sy = Ez_stag[1];
    const int sz = Ez_stag[2];

    // Ezx
    if (sx == 0) {
        Ez(i,j,k,PMLComp::zx) *= sigma_star_fac_x[i-xlo];
    } else {
        Ez(i,j,k,PMLComp::zx) *= sigma_fac_x[i-xlo];
    }

    // Ezy
    if (sy == 0) {
        Ez(i,j,k,PMLComp::zy) *= sigma_star_fac_y[j-ylo];
    } else {
        Ez(i,j,k,PMLComp::zy) *= sigma_fac_y[j-ylo];
    }

    if (dive_cleaning)
    {
        // Ezz
        if (sz == 0) {
            Ez(i,j,k,PMLComp::zz) *= sigma_star_fac_z[k-zlo];
        } else {
            Ez(i,j,k,PMLComp::zz) *= sigma_fac_z[k-zlo];
        }
    }

#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void warpx_damp_pml_bx (int i, int j, int k, amrex::Array4<amrex::Real> const& Bx,
                        const amrex::IntVect& Bx_stag,
                        const amrex::Real* const sigma_fac_x,
                        const amrex::Real* const sigma_fac_y,
                        const amrex::Real* const sigma_fac_z,
                        const amrex::Real* const sigma_star_fac_x,
                        const amrex::Real* const sigma_star_fac_y,
                        const amrex::Real* const sigma_star_fac_z,
                        int xlo, int ylo, int zlo,
                        const bool divb_cleaning)
{
#if defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(i, j, k, Bx, Bx_stag, sigma_fac_x, sigma_fac_y, sigma_fac_z,
           sigma_star_fac_x, sigma_star_fac_y, sigma_star_fac_z, xlo, ylo, zlo, divb_cleaning);
    amrex::Abort("PML not implemented in Cartesian 1D geometry");
#endif

#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

   amrex::ignore_unused(sigma_fac_y, sigma_star_fac_y, ylo);

    // sx = 0 means that Bx is staggered in x, while sx = 1 means that Bx is nodal in x (same for z)
    const int sx = Bx_stag[0];
    const int sz = Bx_stag[1];

    if (divb_cleaning)
    {
        // Bxx
        if (sx == 0) {
            Bx(i,j,k,PMLComp::xx) *= sigma_star_fac_x[i-xlo];
        } else {
            Bx(i,j,k,PMLComp::xx) *= sigma_fac_x[i-xlo];
        }
    }

    // Bxz
    if (sz == 0) {
        Bx(i,j,k,PMLComp::xz) *= sigma_star_fac_z[j-zlo];
    } else {
        Bx(i,j,k,PMLComp::xz) *= sigma_fac_z[j-zlo];
    }

#elif defined(WARPX_DIM_3D)

    // sx = 0 means that Bx is staggered in x, while sx = 1 means that Bx is nodal in x (same for y, z)
    const int sx = Bx_stag[0];
    const int sy = Bx_stag[1];
    const int sz = Bx_stag[2];

    if (divb_cleaning)
    {
        // Bxx
        if (sx == 0) {
            Bx(i,j,k,PMLComp::xx) *= sigma_star_fac_x[i-xlo];
        } else {
            Bx(i,j,k,PMLComp::xx) *= sigma_fac_x[i-xlo];
        }
    }

    // Bxy
    if (sy == 0) {
        Bx(i,j,k,PMLComp::xy) *= sigma_star_fac_y[j-ylo];
    } else {
        Bx(i,j,k,PMLComp::xy) *= sigma_fac_y[j-ylo];
    }

    // Bxz
    if (sz == 0) {
        Bx(i,j,k,PMLComp::xz) *= sigma_star_fac_z[k-zlo];
    } else {
        Bx(i,j,k,PMLComp::xz) *= sigma_fac_z[k-zlo];
    }

#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void warpx_damp_pml_by (int i, int j, int k, amrex::Array4<amrex::Real> const& By,
                        const amrex::IntVect& By_stag,
                        const amrex::Real* const sigma_fac_x,
                        const amrex::Real* const sigma_fac_y,
                        const amrex::Real* const sigma_fac_z,
                        const amrex::Real* const sigma_star_fac_x,
                        const amrex::Real* const sigma_star_fac_y,
                        const amrex::Real* const sigma_star_fac_z,
                        int xlo, int ylo, int zlo,
                        const bool divb_cleaning)
{
#if defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(i, j, k, By, By_stag, sigma_fac_x, sigma_fac_y, sigma_fac_z,
           sigma_star_fac_x, sigma_star_fac_y, sigma_star_fac_z, xlo, ylo, zlo, divb_cleaning);
    amrex::Abort("PML not implemented in Cartesian 1D geometry");
#endif

#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

    amrex::ignore_unused(sigma_fac_y, sigma_star_fac_y, ylo, divb_cleaning);

    // sx = 0 means that By is staggered in x, while sx = 1 means that By is nodal in x (same for z)
    const int sx = By_stag[0];
    const int sz = By_stag[1];

    // Byx
    if (sx == 0) {
        By(i,j,k,PMLComp::yx) *= sigma_star_fac_x[i-xlo];
    } else {
        By(i,j,k,PMLComp::yx) *= sigma_fac_x[i-xlo];
    }

    // Byz
    if (sz == 0) {
        By(i,j,k,PMLComp::yz) *= sigma_star_fac_z[j-zlo];
    } else {
        By(i,j,k,PMLComp::yz) *= sigma_fac_z[j-zlo];
    }

#elif defined(WARPX_DIM_3D)

    // sx = 0 means that By is staggered in x, while sx = 1 means that By is nodal in x (same for y, z)
    const int sx = By_stag[0];
    const int sy = By_stag[1];
    const int sz = By_stag[2];

    // Byx
    if (sx == 0) {
        By(i,j,k,PMLComp::yx) *= sigma_star_fac_x[i-xlo];
    } else {
        By(i,j,k,PMLComp::yx) *= sigma_fac_x[i-xlo];
    }

    if (divb_cleaning)
    {
        // Byy
        if (sy == 0) {
            By(i,j,k,PMLComp::yy) *= sigma_star_fac_y[j-ylo];
        } else {
            By(i,j,k,PMLComp::yy) *= sigma_fac_y[j-ylo];
        }
    }

    // Byz
    if (sz == 0) {
        By(i,j,k,PMLComp::yz) *= sigma_star_fac_z[k-zlo];
    } else {
        By(i,j,k,PMLComp::yz) *= sigma_fac_z[k-zlo];
    }

#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void warpx_damp_pml_bz (int i, int j, int k, amrex::Array4<amrex::Real> const& Bz,
                        const amrex::IntVect& Bz_stag,
                        const amrex::Real* const sigma_fac_x,
                        const amrex::Real* const sigma_fac_y,
                        const amrex::Real* const sigma_fac_z,
                        const amrex::Real* const sigma_star_fac_x,
                        const amrex::Real* const sigma_star_fac_y,
                        const amrex::Real* const sigma_star_fac_z,
                        int xlo, int ylo, int zlo,
                        const bool divb_cleaning)
{
#if defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(i, j, k, Bz, Bz_stag, sigma_fac_x, sigma_fac_y, sigma_fac_z,
           sigma_star_fac_x, sigma_star_fac_y, sigma_star_fac_z, xlo, ylo, zlo, divb_cleaning);
    amrex::Abort("PML not implemented in Cartesian 1D geometry");
#endif

#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

   amrex::ignore_unused(sigma_fac_y, sigma_star_fac_y, ylo);

    // sx = 0 means that Bz is staggered in x, while sx = 1 means that Bz is nodal in x (same for z)
    const int sx = Bz_stag[0];
    const int sz = Bz_stag[1];

    // Bzx
    if (sx == 0) {
        Bz(i,j,k,PMLComp::zx) *= sigma_star_fac_x[i-xlo];
    } else {
        Bz(i,j,k,PMLComp::zx) *= sigma_fac_x[i-xlo];
    }

    if (divb_cleaning)
    {
        // Bzz
        if (sz == 0) {
            Bz(i,j,k,PMLComp::zz) *= sigma_star_fac_z[j-zlo];
        } else {
            Bz(i,j,k,PMLComp::zz) *= sigma_fac_z[j-zlo];
        }
    }

#elif defined(WARPX_DIM_3D)

    // sx = 0 means that Bz is staggered in x, while sx = 1 means that Bz is nodal in x (same for y, z)
    const int sx = Bz_stag[0];
    const int sy = Bz_stag[1];
    const int sz = Bz_stag[2];

    // Bzx
    if (sx == 0) {
        Bz(i,j,k,PMLComp::zx) *= sigma_star_fac_x[i-xlo];
    } else {
        Bz(i,j,k,PMLComp::zx) *= sigma_fac_x[i-xlo];
    }

    // Bzy
    if (sy == 0) {
        Bz(i,j,k,PMLComp::zy) *= sigma_star_fac_y[j-ylo];
    } else {
        Bz(i,j,k,PMLComp::zy) *= sigma_fac_y[j-ylo];
    }

    if (divb_cleaning)
    {
        // Bzz
        if (sz == 0) {
            Bz(i,j,k,PMLComp::zz) *= sigma_star_fac_z[k-zlo];
        } else {
            Bz(i,j,k,PMLComp::zz) *= sigma_fac_z[k-zlo];
        }
    }

#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void warpx_damp_pml_scalar (int i, int j, int k, amrex::Array4<amrex::Real> const& arr,
                            const amrex::IntVect& arr_stag,
                            const amrex::Real* const sigma_fac_x,
                            const amrex::Real* const sigma_fac_y,
                            const amrex::Real* const sigma_fac_z,
                            const amrex::Real* const sigma_star_fac_x,
                            const amrex::Real* const sigma_star_fac_y,
                            const amrex::Real* const sigma_star_fac_z,
                            int xlo, int ylo, int zlo)
{
#if defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(i, j, k, arr, arr_stag, sigma_fac_x, sigma_fac_y, sigma_fac_z,
           sigma_star_fac_x, sigma_star_fac_y, sigma_star_fac_z, xlo, ylo, zlo);
    amrex::Abort("PML not implemented in Cartesian 1D geometry");
#endif

#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

   amrex::ignore_unused(sigma_fac_y, sigma_star_fac_y, ylo);

    // sx = 0 means that arr is staggered in x, while sx = 1 means that arr is nodal in x (same for z)
    const int sx = arr_stag[0];
    const int sz = arr_stag[1];

    // Component along x
    if (sx == 0) {
        arr(i,j,k,PMLComp::x) *= sigma_star_fac_x[i-xlo];
    } else {
        arr(i,j,k,PMLComp::x) *= sigma_fac_x[i-xlo];
    }

    // Component along z
    if (sz == 0) {
        arr(i,j,k,PMLComp::z) *= sigma_star_fac_z[j-zlo];
    } else {
        arr(i,j,k,PMLComp::z) *= sigma_fac_z[j-zlo];
    }

#elif defined(WARPX_DIM_3D)

    // sx = 0 means that arr is staggered in x, while sx = 1 means that arr is nodal in x (same for y, z)
    const int sx = arr_stag[0];
    const int sy = arr_stag[1];
    const int sz = arr_stag[2];

    // Component along x
    if (sx == 0) {
        arr(i,j,k,PMLComp::x) *= sigma_star_fac_x[i-xlo];
    } else {
        arr(i,j,k,PMLComp::x) *= sigma_fac_x[i-xlo];
    }

    // Component along y
    if (sy == 0) {
        arr(i,j,k,PMLComp::y) *= sigma_star_fac_y[j-ylo];
    } else {
        arr(i,j,k,PMLComp::y) *= sigma_fac_y[j-ylo];
    }

    // Component along z
    if (sz == 0) {
        arr(i,j,k,PMLComp::z) *= sigma_star_fac_z[k-zlo];
    } else {
        arr(i,j,k,PMLComp::z) *= sigma_fac_z[k-zlo];
    }

#endif
}

#endif
