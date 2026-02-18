/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_UPDATEPOSITIONBLENDED_H_
#define WARPX_PARTICLES_PUSHER_UPDATEPOSITIONBLENDED_H_

#include "Utils/WarpXConst.H"

#include <AMReX.H>
//#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>


/** \brief Push the particle position over one time step,
 *         given the value of its momenta `ux`, `uy`, `uz`,
 *         magnetic field components, `Bx`, `By`, `Bz`,
 *         and magnetic field gradients `gradBx`, `gradBy`, `gradBz`,
 *         using the Blended position update, including estimated drifts (gradB, ExB, curvature)
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdatePositionBlended ( amrex::ParticleReal& x, amrex::ParticleReal& y, amrex::ParticleReal& z,
                     const amrex::ParticleReal ux, const amrex::ParticleReal uy, const amrex::ParticleReal uz,
                     const amrex::ParticleReal Ex, const amrex::ParticleReal Ey, const amrex::ParticleReal Ez,
                     const amrex::ParticleReal Bx, const amrex::ParticleReal By, const amrex::ParticleReal Bz,
                     const amrex::ParticleReal gradBx, const amrex::ParticleReal gradBy, const amrex::ParticleReal gradBz,
                     const amrex::ParticleReal kappax, const amrex::ParticleReal kappay, const amrex::ParticleReal kappaz,
                     const amrex::Real dt,
                     amrex::ParticleReal const m, const amrex::ParticleReal q,
                     const amrex::ParticleReal alpha_gc, const amrex::ParticleReal mu_gc)

{
    using namespace amrex::literals;

    // Compute blending coefficient alpha
    // const amrex::ParticleReal alpha = ComputeBlendedAlpha(Bx, By, Bz, q, m, dt);

    amrex::ParticleReal const u2 = ux*ux + uy*uy + uz*uz;

    // massive particles
    if (m > 0.0_rt) {
        
        constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);
        const amrex::ParticleReal inv_gamma = 1._prt/std::sqrt(1._prt + u2*inv_c2);

        // Compute |B|
        const amrex::ParticleReal Bmag = std::sqrt(Bx*Bx + By*By + Bz*Bz);

        // Limit Bmag to threshold value Bmin to avoid division by zero
        constexpr amrex::ParticleReal Bmin  = 1e-12_prt;     // Tesla
        const amrex::Real invB = 1.0_prt / amrex::max(Bmag, Bmin);

        // If B is below threshold, fall back to standard x += v dt
        // otherwise do standard blended position update with drifts
        if (Bmag <= Bmin) {

#if !defined(WARPX_DIM_1D_Z)
            x += ux * inv_gamma * dt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            // Particles pushed in 3D even in 1D cylindrical, 1D spherical, and 2D cylindrical geometry
            y += uy * inv_gamma * dt;
#endif
#if !defined(WARPX_DIM_RCYLINDER)
            z += uz * inv_gamma * dt;
#endif
            return;
        }

        //define unit vector components
        const amrex::ParticleReal bx = Bx * invB;
        const amrex::ParticleReal by = By * invB;
        const amrex::ParticleReal bz = Bz * invB;

        //compute parallel and perpendicular momenta
        const amrex::ParticleReal u_dot_b = ux*bx + uy*by + uz*bz;
        const amrex::ParticleReal upar_x  = u_dot_b * bx;
        const amrex::ParticleReal upar_y  = u_dot_b * by;
        const amrex::ParticleReal upar_z  = u_dot_b * bz;

        const amrex::ParticleReal uperp_x = ux - upar_x;
        const amrex::ParticleReal uperp_y = uy - upar_y;
        const amrex::ParticleReal uperp_z = uz - upar_z;

        // E×B drift: v_E = (c E × B) / B^2
        const amrex::ParticleReal ExB_x = Ey*Bz - Ez*By;
        const amrex::ParticleReal ExB_y = Ez*Bx - Ex*Bz;
        const amrex::ParticleReal ExB_z = Ex*By - Ey*Bx;

        const amrex::ParticleReal vE_x =  ExB_x * invB * invB;
        const amrex::ParticleReal vE_y =  ExB_y * invB * invB;
        const amrex::ParticleReal vE_z =  ExB_z * invB * invB;

        // Grad-B drift: v_gradB = (mu / (q B)) (B × ∇B) / B^2
        const amrex::ParticleReal bXgradB_x = By*gradBz - Bz*gradBy;
        const amrex::ParticleReal bXgradB_y = Bz*gradBx - Bx*gradBz;
        const amrex::ParticleReal bXgradB_z = Bx*gradBy - By*gradBx;

        const amrex::ParticleReal vgradB_x = (mu_gc * invB * invB / q) * bXgradB_x;
        const amrex::ParticleReal vgradB_y = (mu_gc * invB * invB / q) * bXgradB_y;
        const amrex::ParticleReal vgradB_z = (mu_gc * invB * invB / q) * bXgradB_z;


        // Curvature drift: v_c = (γ^2 v_par^2)/((q/m) * Bmag) * (b × kappa)
        const amrex::ParticleReal bXkx = by*kappaz - bz*kappay;
        const amrex::ParticleReal bXky = bz*kappax - bx*kappaz;
        const amrex::ParticleReal bXkz = bx*kappay - by*kappax;

        const amrex::ParticleReal coeff = m*(upar_x*upar_x + upar_y*upar_y + upar_z*upar_z) * invB / q;

        // vcurv_x = (m * (inv_gamma*upar_x)*(inv_gamma*upar_x)  * invB / q * bXkx;
        const amrex::ParticleReal vcurv_x = coeff * bXkx;
        const amrex::ParticleReal vcurv_y = coeff * bXky;
        const amrex::ParticleReal vcurv_z = coeff * bXkz;

        // net drift velocity
        const amrex::ParticleReal vx_drift = vE_x + vgradB_x + vcurv_x;
        const amrex::ParticleReal vy_drift = vE_y + vgradB_y + vcurv_y;
        const amrex::ParticleReal vz_drift = vE_z + vgradB_z + vcurv_z;

        // Update positions over one time step
        // Blended update blends perpendicular and drift motion via alpha
#if !defined(WARPX_DIM_1D_Z)
            x += upar_x * inv_gamma * dt 
            + alpha_gc * uperp_x * inv_gamma * dt 
            + (1._prt - alpha_gc) * vx_drift * dt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            // Particles pushed in 3D even in 1D cylindrical, 1D spherical, and 2D cylindrical geometry
            y += upar_y * inv_gamma * dt 
            + alpha_gc * uperp_y * inv_gamma * dt 
            + (1._prt - alpha_gc) * vy_drift * dt;
#endif
#if !defined(WARPX_DIM_RCYLINDER)
            z += upar_z * inv_gamma * dt 
            + alpha_gc * uperp_z * inv_gamma * dt 
            + (1._prt - alpha_gc) * vz_drift * dt;
#endif
    }
    // massless particles (photons)
    else {
        if (u2 > 0._prt) {
            amrex::ParticleReal const c_over_unorm = PhysConst::c/std::sqrt(u2);

            // Update positions over one time step
#if !defined(WARPX_DIM_1D_Z)
            x += ux * c_over_unorm * dt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            // Particles pushed in 3D even in 1D cylindrical, 1D spherical, and 2D cylindrical geometry
            y += uy * c_over_unorm * dt;
#endif
#if !defined(WARPX_DIM_RCYLINDER)
            z += uz * c_over_unorm * dt;
#endif
        }
    }
}

#endif // WARPX_PARTICLES_PUSHER_UPDATEPOSITIONBLENDED_H_
