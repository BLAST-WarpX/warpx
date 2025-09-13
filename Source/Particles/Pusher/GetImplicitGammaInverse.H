/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef GET_IMPLICIT_GAMMA_INVERSE_H_
#define GET_IMPLICIT_GAMMA_INVERSE_H_

#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_REAL.H>

#include <cmath>

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::ParticleReal GetImplicitGammaInverse (const amrex::ParticleReal uxp_n,
                                             const amrex::ParticleReal uyp_n,
                                             const amrex::ParticleReal uzp_n,
                                             const amrex::ParticleReal uxp_nph,
                                             const amrex::ParticleReal uyp_nph,
                                             const amrex::ParticleReal uzp_nph) noexcept
{
    // Compute inverse Lorentz factor for the implicit methods, the average of gamma at time levels n and n+1
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

#endif
