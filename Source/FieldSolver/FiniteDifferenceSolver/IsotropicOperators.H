/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *          S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ISOTROPIC_OPERATORS_H_
#define WARPX_ISOTROPIC_OPERATORS_H_

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Extension.H>
#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

/** Isotropic finite-difference operators for the Hybrid-PIC Ohm's-law solve.
 *
 *  These device-inline helpers isotropize the hyper-resistivity Laplacian,
 *  the resistive diffusion of the out-of-plane B (corner-curl correction),
 *  and the staggered gradient of the electron pressure. They are header-only
 *  because they are evaluated inside GPU lambdas in HybridPICSolveE.cpp. They
 *  are guarded by the same dimensionality macros as the Cartesian solve (no
 *  definitions in RZ / RSPHERE / RCYLINDER builds).
 */
namespace warpx::hybrid_isotropic
{
#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)

    /** Isotropic discrete Laplacian for the hyper-resistivity term.
     *
     *  The cross stencil Dxx+Dyy+Dzz has a leading truncation error whose
     *  angular dependence is cos(4*theta): the damping rate of a mode at
     *  finite kh differs between the grid axes and the diagonals, which
     *  imprints fourfold "squaring" on diffusing structures. The Mehrstellen
     *  9-point (2D) and Patra-Karttunen 27-point (3D) stencils cancel that
     *  term: their leading error is proportional to the isotropic
     *  biharmonic. They are written here as the cross stencil plus compact
     *  cross-derivative corrections, which stays consistent (2nd order) on
     *  non-cubic cells; the error is fully isotropic for cubic cells.
     */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real LaplacianIsotropic (
        amrex::Array4<amrex::Real const> const& F,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2)
    {
        using namespace amrex::literals;
#if defined(WARPX_DIM_3D)
        amrex::Real const d2x = F(i+1,j,k) - 2._rt*F(i,j,k) + F(i-1,j,k);
        amrex::Real const d2y = F(i,j+1,k) - 2._rt*F(i,j,k) + F(i,j-1,k);
        amrex::Real const d2z = F(i,j,k+1) - 2._rt*F(i,j,k) + F(i,j,k-1);

        // undivided cross differences delta_a^2 delta_b^2 F
        auto const cxy_at = [&] (int kk) {
            return F(i+1,j+1,kk) + F(i+1,j-1,kk) + F(i-1,j+1,kk) + F(i-1,j-1,kk)
                 - 2._rt*(F(i+1,j,kk) + F(i-1,j,kk) + F(i,j+1,kk) + F(i,j-1,kk))
                 + 4._rt*F(i,j,kk);
        };
        amrex::Real const cxy = cxy_at(k);
        amrex::Real const cxz =
              F(i+1,j,k+1) + F(i+1,j,k-1) + F(i-1,j,k+1) + F(i-1,j,k-1)
            - 2._rt*(F(i+1,j,k) + F(i-1,j,k) + F(i,j,k+1) + F(i,j,k-1))
            + 4._rt*F(i,j,k);
        amrex::Real const cyz =
              F(i,j+1,k+1) + F(i,j+1,k-1) + F(i,j-1,k+1) + F(i,j-1,k-1)
            - 2._rt*(F(i,j+1,k) + F(i,j-1,k) + F(i,j,k+1) + F(i,j,k-1))
            + 4._rt*F(i,j,k);
        amrex::Real const txyz = cxy_at(k+1) - 2._rt*cxy_at(k) + cxy_at(k-1);

        return d2x*inv_h2[0] + d2y*inv_h2[1] + d2z*inv_h2[2]
             + ((h2[0] + h2[1])/12._rt)*cxy*inv_h2[0]*inv_h2[1]
             + ((h2[0] + h2[2])/12._rt)*cxz*inv_h2[0]*inv_h2[2]
             + ((h2[1] + h2[2])/12._rt)*cyz*inv_h2[1]*inv_h2[2]
             + ((h2[0]*h2[1] + h2[1]*h2[2] + h2[0]*h2[2])/90._rt)
               *txyz*inv_h2[0]*inv_h2[1]*inv_h2[2];
#else
        // 2D XZ: Mehrstellen 9-point (j is the z index)
        amrex::Real const d2x = F(i+1,j,k) - 2._rt*F(i,j,k) + F(i-1,j,k);
        amrex::Real const d2z = F(i,j+1,k) - 2._rt*F(i,j,k) + F(i,j-1,k);
        amrex::Real const cxz =
              F(i+1,j+1,k) + F(i+1,j-1,k) + F(i-1,j+1,k) + F(i-1,j-1,k)
            - 2._rt*(F(i+1,j,k) + F(i-1,j,k) + F(i,j+1,k) + F(i,j-1,k))
            + 4._rt*F(i,j,k);
        return d2x*inv_h2[0] + d2z*inv_h2[1]
             + ((h2[0] + h2[1])/12._rt)*cxz*inv_h2[0]*inv_h2[1];
#endif
    }

    // Corner-curl correction that isotropizes the in-plane resistive
    // diffusion of the out-of-plane B component (Bz in 3D, By in 2D XZ).
    //
    // The resistive E (eta*curl(B)/mu0) advanced by Faraday produces, for
    // divergence-free B, the cross-stencil in-plane Laplacian of the
    // out-of-plane B, whose leading error is cos(4*theta) anisotropic and
    // drives a grid m=4 mode. Adding the corrections below to the two
    // in-plane E components makes the standard Faraday curl emit the missing
    // Mehrstellen corner term (h^2/6) d2a d2b B, i.e. the in-plane isotropic
    // Laplacian. The correction enters only through E, so the Faraday curl is
    // unchanged and div(B) stays exactly zero. Returned values already carry
    // 1/mu0 (matching the resistive current); multiply by eta at the call.
    //
    // Coefficient: the in-plane Mehrstellen corner is
    // ((h2a+h2b)/12) * (d2a d2b B) * inv_h2a * inv_h2b. It is split
    // antisymmetrically between the two E components (half each), with each
    // half pre-differenced (backward) along the in-plane axis so that the
    // Faraday upward difference completes a centered second difference.
#if defined(WARPX_DIM_3D)
    // 3D: Bz corner delivered through Ex and Ey (curl_z). a=x, b=y.
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real CornerResistiveEx_3D (
        amrex::Array4<amrex::Real const> const& Bz,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::GpuArray<amrex::Real, 3> const& dx,
        amrex::Real const inv_mu0)
    {
        using namespace amrex::literals;
        // half the corner coefficient, times dy (undone by Faraday UpwardDy)
        amrex::Real const cf = 0.5_rt*((h2[0]+h2[1])/12._rt)*inv_h2[0]*inv_h2[1]
                             * dx[1] * inv_mu0;
        auto const d2x = [&] (int jj) {
            return Bz(i+1,jj,k) - 2._rt*Bz(i,jj,k) + Bz(i-1,jj,k);
        };
        // backward y-difference of d2x(Bz); UpwardDy in Faraday -> d2y d2x Bz
        return cf * (d2x(j) - d2x(j-1));
    }
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real CornerResistiveEy_3D (
        amrex::Array4<amrex::Real const> const& Bz,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::GpuArray<amrex::Real, 3> const& dx,
        amrex::Real const inv_mu0)
    {
        using namespace amrex::literals;
        amrex::Real const cf = -0.5_rt*((h2[0]+h2[1])/12._rt)*inv_h2[0]*inv_h2[1]
                             * dx[0] * inv_mu0;
        auto const d2y = [&] (int ii) {
            return Bz(ii,j+1,k) - 2._rt*Bz(ii,j,k) + Bz(ii,j-1,k);
        };
        // backward x-difference of d2y(Bz); UpwardDx in Faraday -> d2x d2y Bz
        return cf * (d2y(i) - d2y(i-1));
    }

    // Collocated (nodal) analogues. On a collocated grid the Faraday UpwardD is a
    // wide centered difference (CartesianNodalAlgorithm), so the uncorrected
    // resistive diffusion of the out-of-plane B is the WIDE cross stencil
    // -(sin^2(kx hx) + sin^2(ky hy))/h^2, whose leading cos(4*theta) anisotropy is
    // (h^2/3)(kx^4+ky^4) -- four times the compact-cross value. The Yee
    // backward-pre-difference trick does not apply (UpwardD is no longer one-sided),
    // so the corner is delivered as a compact 2nd difference along one in-plane axis
    // followed by the nodal centered difference along the other; the wide Faraday
    // UpwardD then completes the corner. A half (coefficient 1/3) goes through each
    // in-plane E component, which reproduces the (hxa^2+hxb^2)/3 corner for any cell
    // aspect ratio (isotropic for cubic cells, consistent otherwise). Returned values
    // carry 1/mu0; multiply by eta at the call.
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real CornerResistiveEx_3D_Nodal (
        amrex::Array4<amrex::Real const> const& Bz,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& dx,
        amrex::Real const inv_mu0)
    {
        using namespace amrex::literals;
        // dEx = (1/3)(1/mu0) Dn_y( d2x(Bz) ); Dn_y is the nodal centered difference.
        auto const d2x = [&] (int jj) {
            return Bz(i+1,jj,k) - 2._rt*Bz(i,jj,k) + Bz(i-1,jj,k);
        };
        return (inv_mu0/3._rt) * (0.5_rt/dx[1]) * (d2x(j+1) - d2x(j-1));
    }
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real CornerResistiveEy_3D_Nodal (
        amrex::Array4<amrex::Real const> const& Bz,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& dx,
        amrex::Real const inv_mu0)
    {
        using namespace amrex::literals;
        // dEy = -(1/3)(1/mu0) Dn_x( d2y(Bz) )
        auto const d2y = [&] (int ii) {
            return Bz(ii,j+1,k) - 2._rt*Bz(ii,j,k) + Bz(ii,j-1,k);
        };
        return -(inv_mu0/3._rt) * (0.5_rt/dx[0]) * (d2y(i+1) - d2y(i-1));
    }
#elif defined(WARPX_DIM_XZ)
    // 2D XZ: By corner delivered through Ex and Ez (curl_y). a=x, b=z (j index).
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real CornerResistiveEx_XZ (
        amrex::Array4<amrex::Real const> const& By,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::GpuArray<amrex::Real, 3> const& dx,
        amrex::Real const inv_mu0)
    {
        using namespace amrex::literals;
        // curl_y = UpwardDz(Ex) - UpwardDx(Ez); this half goes via UpwardDz
        amrex::Real const cf = -0.5_rt*((h2[0]+h2[1])/12._rt)*inv_h2[0]*inv_h2[1]
                             * dx[1] * inv_mu0;
        auto const d2x = [&] (int jj) {
            return By(i+1,jj,k) - 2._rt*By(i,jj,k) + By(i-1,jj,k);
        };
        return cf * (d2x(j) - d2x(j-1));
    }
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real CornerResistiveEz_XZ (
        amrex::Array4<amrex::Real const> const& By,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::GpuArray<amrex::Real, 3> const& dx,
        amrex::Real const inv_mu0)
    {
        using namespace amrex::literals;
        amrex::Real const cf = 0.5_rt*((h2[0]+h2[1])/12._rt)*inv_h2[0]*inv_h2[1]
                             * dx[0] * inv_mu0;
        auto const d2z = [&] (int ii) {
            return By(ii,j+1,k) - 2._rt*By(ii,j,k) + By(ii,j-1,k);
        };
        return cf * (d2z(i) - d2z(i-1));
    }

    // Collocated (nodal) analogues for 2D XZ (out-of-plane By, in-plane x and z).
    // Same wide-Faraday reasoning as the 3D nodal pair; the in-plane axes are x (i)
    // and z (j). Sign convention matches the Yee XZ pair (Ex half negative, Ez half
    // positive), set by the curl_y topology dBy/dt = UpwardDx(Ez) - UpwardDz(Ex).
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real CornerResistiveEx_XZ_Nodal (
        amrex::Array4<amrex::Real const> const& By,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& dx,
        amrex::Real const inv_mu0)
    {
        using namespace amrex::literals;
        // dEx = -(1/3)(1/mu0) Dn_z( d2x(By) ); the -UpwardDz(Ex) in Faraday -> +corner
        auto const d2x = [&] (int jj) {
            return By(i+1,jj,k) - 2._rt*By(i,jj,k) + By(i-1,jj,k);
        };
        return -(inv_mu0/3._rt) * (0.5_rt/dx[1]) * (d2x(j+1) - d2x(j-1));
    }
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real CornerResistiveEz_XZ_Nodal (
        amrex::Array4<amrex::Real const> const& By,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& dx,
        amrex::Real const inv_mu0)
    {
        using namespace amrex::literals;
        // dEz = +(1/3)(1/mu0) Dn_x( d2z(By) ); the +UpwardDx(Ez) in Faraday -> +corner
        auto const d2z = [&] (int ii) {
            return By(ii,j+1,k) - 2._rt*By(ii,j,k) + By(ii,j-1,k);
        };
        return (inv_mu0/3._rt) * (0.5_rt/dx[0]) * (d2z(i+1) - d2z(i-1));
    }
#endif

    // Isotropized gradient of a nodal scalar (the electron pressure).
    //
    // The two-point staggered difference D_x has truncation error
    // (dx^2/24) d^3F/dx^3, whose Fourier symbol depends on k_x alone: the
    // relative error of the gradient vector is modulated in angle as
    // cos(4*theta) -- the same fourfold grid imprint the cross-Laplacian puts
    // on diffusion rates. Pre-smoothing the scalar along each TRANSVERSE
    // direction t with weight a_t = dx^2/(24 dt^2) turns the leading error
    // into (dx^2/24) d/dx(laplacian F): the x-component of an isotropic
    // vector field, so the angular modulation cancels (fully isotropic on
    // cubic cells, consistent 2nd order otherwise). On the collocated grid
    // the wide centered difference has truncation error (dx^2/6) d^3F/dx^3,
    // so the nodal variants use a_t = dx^2/(6 dt^2). Either way the result
    // stays on the component's native staggering and the on-axis stencil is
    // unchanged: isotropization equalizes the diagonal error to the
    // (pre-existing) axis error rather than adding accuracy.
    //
    // NOTE (curl(grad) identity): the Yee curl of the PLAIN staggered
    // gradient vanishes identically, which is why the Ohm's-law solve may
    // omit grad(Pe) on the solve_for_Faraday path. The isotropized gradient
    // smooths each component along different (transverse) directions, so its
    // discrete curl is only O(h^2) small, not zero. This is benign here
    // precisely because grad(Pe) is only ever evaluated on the
    // !solve_for_Faraday path (the E used for the particle push and
    // diagnostics) and the B integration never sees the term -- do not route
    // this operator into an E that feeds a curl without revisiting that
    // argument.
#if defined(WARPX_DIM_3D)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real UpwardDxIsotropic_3D (
        amrex::Array4<amrex::Real const> const& F,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::Real const inv_dx)
    {
        using namespace amrex::literals;
        amrex::Real const ay = h2[0]*inv_h2[1]/24._rt;
        amrex::Real const az = h2[0]*inv_h2[2]/24._rt;
        auto const dfx = [&] (int jj, int kk) { return F(i+1,jj,kk) - F(i,jj,kk); };
        return inv_dx * ( (1._rt - 2._rt*(ay + az))*dfx(j,k)
                        + ay*(dfx(j+1,k) + dfx(j-1,k))
                        + az*(dfx(j,k+1) + dfx(j,k-1)) );
    }
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real UpwardDyIsotropic_3D (
        amrex::Array4<amrex::Real const> const& F,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::Real const inv_dy)
    {
        using namespace amrex::literals;
        amrex::Real const ax = h2[1]*inv_h2[0]/24._rt;
        amrex::Real const az = h2[1]*inv_h2[2]/24._rt;
        auto const dfy = [&] (int ii, int kk) { return F(ii,j+1,kk) - F(ii,j,kk); };
        return inv_dy * ( (1._rt - 2._rt*(ax + az))*dfy(i,k)
                        + ax*(dfy(i+1,k) + dfy(i-1,k))
                        + az*(dfy(i,k+1) + dfy(i,k-1)) );
    }
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real UpwardDzIsotropic_3D (
        amrex::Array4<amrex::Real const> const& F,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::Real const inv_dz)
    {
        using namespace amrex::literals;
        amrex::Real const ax = h2[2]*inv_h2[0]/24._rt;
        amrex::Real const ay = h2[2]*inv_h2[1]/24._rt;
        auto const dfz = [&] (int ii, int jj) { return F(ii,jj,k+1) - F(ii,jj,k); };
        return inv_dz * ( (1._rt - 2._rt*(ax + ay))*dfz(i,j)
                        + ax*(dfz(i+1,j) + dfz(i-1,j))
                        + ay*(dfz(i,j+1) + dfz(i,j-1)) );
    }

    // Collocated (nodal) analogues: wide centered difference, a_t = dx^2/(6 dt^2).
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real UpwardDxIsotropic_3D_Nodal (
        amrex::Array4<amrex::Real const> const& F,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::Real const inv_dx)
    {
        using namespace amrex::literals;
        amrex::Real const ay = h2[0]*inv_h2[1]/6._rt;
        amrex::Real const az = h2[0]*inv_h2[2]/6._rt;
        auto const dfx = [&] (int jj, int kk) { return F(i+1,jj,kk) - F(i-1,jj,kk); };
        return 0.5_rt*inv_dx * ( (1._rt - 2._rt*(ay + az))*dfx(j,k)
                               + ay*(dfx(j+1,k) + dfx(j-1,k))
                               + az*(dfx(j,k+1) + dfx(j,k-1)) );
    }
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real UpwardDyIsotropic_3D_Nodal (
        amrex::Array4<amrex::Real const> const& F,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::Real const inv_dy)
    {
        using namespace amrex::literals;
        amrex::Real const ax = h2[1]*inv_h2[0]/6._rt;
        amrex::Real const az = h2[1]*inv_h2[2]/6._rt;
        auto const dfy = [&] (int ii, int kk) { return F(ii,j+1,kk) - F(ii,j-1,kk); };
        return 0.5_rt*inv_dy * ( (1._rt - 2._rt*(ax + az))*dfy(i,k)
                               + ax*(dfy(i+1,k) + dfy(i-1,k))
                               + az*(dfy(i,k+1) + dfy(i,k-1)) );
    }
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real UpwardDzIsotropic_3D_Nodal (
        amrex::Array4<amrex::Real const> const& F,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::Real const inv_dz)
    {
        using namespace amrex::literals;
        amrex::Real const ax = h2[2]*inv_h2[0]/6._rt;
        amrex::Real const ay = h2[2]*inv_h2[1]/6._rt;
        auto const dfz = [&] (int ii, int jj) { return F(ii,jj,k+1) - F(ii,jj,k-1); };
        return 0.5_rt*inv_dz * ( (1._rt - 2._rt*(ax + ay))*dfz(i,j)
                               + ax*(dfz(i+1,j) + dfz(i-1,j))
                               + ay*(dfz(i,j+1) + dfz(i,j-1)) );
    }
#elif defined(WARPX_DIM_XZ)
    // 2D XZ: j is the z index; the single transverse direction of d/dx is z
    // and of d/dz is x. (No y derivative exists, so Ey keeps the plain
    // stencil, which is identically zero in XZ.)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real UpwardDxIsotropic_XZ (
        amrex::Array4<amrex::Real const> const& F,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::Real const inv_dx)
    {
        using namespace amrex::literals;
        amrex::Real const az = h2[0]*inv_h2[1]/24._rt;
        auto const dfx = [&] (int jj) { return F(i+1,jj,k) - F(i,jj,k); };
        return inv_dx * ( (1._rt - 2._rt*az)*dfx(j) + az*(dfx(j+1) + dfx(j-1)) );
    }
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real UpwardDzIsotropic_XZ (
        amrex::Array4<amrex::Real const> const& F,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::Real const inv_dz)
    {
        using namespace amrex::literals;
        amrex::Real const ax = h2[1]*inv_h2[0]/24._rt;
        auto const dfz = [&] (int ii) { return F(ii,j+1,k) - F(ii,j,k); };
        return inv_dz * ( (1._rt - 2._rt*ax)*dfz(i) + ax*(dfz(i+1) + dfz(i-1)) );
    }

    // Collocated (nodal) analogues: wide centered difference, a_t = dx^2/(6 dt^2).
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real UpwardDxIsotropic_XZ_Nodal (
        amrex::Array4<amrex::Real const> const& F,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::Real const inv_dx)
    {
        using namespace amrex::literals;
        amrex::Real const az = h2[0]*inv_h2[1]/6._rt;
        auto const dfx = [&] (int jj) { return F(i+1,jj,k) - F(i-1,jj,k); };
        return 0.5_rt*inv_dx * ( (1._rt - 2._rt*az)*dfx(j) + az*(dfx(j+1) + dfx(j-1)) );
    }
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real UpwardDzIsotropic_XZ_Nodal (
        amrex::Array4<amrex::Real const> const& F,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::Real const inv_dz)
    {
        using namespace amrex::literals;
        amrex::Real const ax = h2[1]*inv_h2[0]/6._rt;
        auto const dfz = [&] (int ii) { return F(ii,j+1,k) - F(ii,j-1,k); };
        return 0.5_rt*inv_dz * ( (1._rt - 2._rt*ax)*dfz(i) + ax*(dfz(i+1) + dfz(i-1)) );
    }
#endif

#endif // !RZ && !RCYLINDER && !RSPHERE
} // namespace warpx::hybrid_isotropic

#endif // WARPX_ISOTROPIC_OPERATORS_H_
