/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_UPDATEPOSITION_H_
#define WARPX_PARTICLES_PUSHER_UPDATEPOSITION_H_

#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>



/** \brief Push the particle's positions over one timestep,
 *    given the value of its momenta `ux`, `uy`, `uz`.
 *    This uses the standard leapfrog algorithm
 *    x^{n+1} - x^{n} = dt*u^{n+1/2}/gamma^{n+1/2}
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdatePosition([[maybe_unused]] amrex::ParticleReal& x,
                    [[maybe_unused]] amrex::ParticleReal& y,
                    [[maybe_unused]] amrex::ParticleReal& z,
                    const amrex::ParticleReal ux, const amrex::ParticleReal uy, const amrex::ParticleReal uz,
                    const amrex::Real dt )
{
    using namespace amrex::literals;

    constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);

    // Compute inverse Lorentz factor
    const amrex::ParticleReal inv_gamma = 1._prt/std::sqrt(1._prt + (ux*ux + uy*uy + uz*uz)*inv_c2);
    // Update positions over one time step
#if !defined(WARPX_DIM_1D_Z)
    x += ux * inv_gamma * dt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    // Radial pushes particles in 3D
    y += uy * inv_gamma * dt;
#endif
#if !defined(WARPX_DIM_RCYLINDER)
    z += uz * inv_gamma * dt;
#endif
}

/** \brief Compute the inverse Lorentz factor for the position update in
 *         the implicit methods, which is the average of gamma at time
 *         levels n and n+1.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::ParticleReal GetImplicitGammaInverse (const amrex::ParticleReal uxp_n,
                                             const amrex::ParticleReal uyp_n,
                                             const amrex::ParticleReal uzp_n,
                                             const amrex::ParticleReal uxp_nph,
                                             const amrex::ParticleReal uyp_nph,
                                             const amrex::ParticleReal uzp_nph) noexcept
{
    using namespace amrex::literals;

    constexpr amrex::ParticleReal inv_c2 = 1.0_prt/(PhysConst::c*PhysConst::c);

    const amrex::ParticleReal uxp_np1 = 2._prt*uxp_nph - uxp_n;
    const amrex::ParticleReal uyp_np1 = 2._prt*uyp_nph - uyp_n;
    const amrex::ParticleReal uzp_np1 = 2._prt*uzp_nph - uzp_n;
    const amrex::ParticleReal gamma_n = std::sqrt(1._prt + (uxp_n*uxp_n + uyp_n*uyp_n + uzp_n*uzp_n)*inv_c2);
    const amrex::ParticleReal gamma_np1 = std::sqrt(1._prt + (uxp_np1*uxp_np1 + uyp_np1*uyp_np1 + uzp_np1*uzp_np1)*inv_c2);
    const amrex::ParticleReal gaminv = 2._prt/(gamma_n + gamma_np1);

    return gaminv;
}

/** \brief Push the particle's positions over one timestep,
 *    given the value of its momenta `ux`, `uy`, `uz`.
 *    The implicit version is the Crank-Nicolson scheme,
 *    x^{n+1} - x^{n} = dt*(u^{n+1} + u^{n})/(gamma^{n+1} + gamma^{n})
 *    See Eqs. 15 and 17 in Chen, JCP 407 (2020) 109228.
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdatePositionImplicit ([[maybe_unused]] amrex::ParticleReal& x,
                             [[maybe_unused]] amrex::ParticleReal& y,
                             [[maybe_unused]] amrex::ParticleReal& z,
                             const amrex::ParticleReal ux_n, const amrex::ParticleReal uy_n, const amrex::ParticleReal uz_n,
                             const amrex::ParticleReal ux, const amrex::ParticleReal uy, const amrex::ParticleReal uz,
                             const amrex::Real dt )
{

    // Compute inverse Lorentz factor, the average of gamma at time levels n and n+1
    const amrex::ParticleReal inv_gamma = GetImplicitGammaInverse(ux_n, uy_n, uz_n, ux, uy, uz);

    // Update positions over one time step
#if !defined(WARPX_DIM_1D_Z)
    x += ux * inv_gamma * dt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    // Radial pushes particles in 3D
    y += uy * inv_gamma * dt;
#endif
#if !defined(WARPX_DIM_RCYLINDER)
    z += uz * inv_gamma * dt;
#endif
}

/** \brief Check particle position for convergence. This is used by the theta-implicit
 *    and semi-implicit time solvers to obtain a self-consistent time-centered update
 *    of the particles for given electric and magnetic fields on the grid.
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void PositionNorm ([[maybe_unused]] amrex::ParticleReal dxp,
                   [[maybe_unused]] amrex::ParticleReal dyp,
                   [[maybe_unused]] amrex::ParticleReal dzp,
                   [[maybe_unused]] amrex::ParticleReal& dxp_save,
                   [[maybe_unused]] amrex::ParticleReal& dyp_save,
                   [[maybe_unused]] amrex::ParticleReal& dzp_save,
                   [[maybe_unused]] amrex::ParticleReal idxg2,
                   [[maybe_unused]] amrex::ParticleReal idyg2,
                   [[maybe_unused]] amrex::ParticleReal idzg2,
                   amrex::ParticleReal& step_norm, const int iter )
{
    using namespace amrex::literals;

    if (iter==0) { step_norm = 1.0_prt; }
    else {
#if !defined(WARPX_DIM_RCYLINDER)
        step_norm = (dzp - dzp_save)*(dzp - dzp_save)*idzg2;
#else
        step_norm = 0._prt;
#endif
#if !defined(WARPX_DIM_1D_Z)
        step_norm += (dxp - dxp_save)*(dxp - dxp_save)*idxg2;
#endif
#if defined(WARPX_DIM_3D)
        step_norm += (dyp - dyp_save)*(dyp - dyp_save)*idyg2;
#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        step_norm += (dyp - dyp_save)*(dyp - dyp_save)*idxg2;
#endif
        step_norm = std::sqrt(step_norm);
    }
#if !defined(WARPX_DIM_RCYLINDER)
    dzp_save = dzp;
#endif
#if !defined(WARPX_DIM_1D_Z)
    dxp_save = dxp;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    dyp_save = dyp;
#endif

}

#endif // WARPX_PARTICLES_PUSHER_UPDATEPOSITION_H_
