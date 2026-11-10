/* Copyright 2021 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DISTANCETOEB_H_
#define WARPX_DISTANCETOEB_H_

#include "Utils/TextMsg.H"

#include <ablastr/particles/NodalFieldGather.H>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>

namespace DistanceToEB
{

AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real dot_product (const amrex::RealVect& a, const amrex::RealVect& b) noexcept
{
    return AMREX_D_TERM(a[0]*b[0], + a[1]*b[1], + a[2]*b[2]);
}

// This function calculates the unit normal vector to the embedded boundary at
// the particle position (xp, yp, zp), by interpolating the signed distance
// function distance_to_eb. plo/dxi are the geometry data. The normal is
// always returned in 3D Cartesian coordinates.
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::GpuArray<amrex::ParticleReal, 3>
interp_normal (amrex::ParticleReal xp, amrex::ParticleReal yp, amrex::ParticleReal zp,
                               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
                               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
                               amrex::Array4<const amrex::Real> const& distance_to_eb) noexcept
{
    using namespace amrex::literals;

    // Compute the nodal and cell-centered interpolation weights at the position.
    // i,j,k (ic,jc,kc) are the indices of the nearest node (cell-center) to the
    // left of the point; W (Wc) are the corresponding interpolation weights.
    int i, j, k;
    amrex::Real W[AMREX_SPACEDIM][2];
    ablastr::particles::compute_weights<amrex::IndexType::NODE>(
        xp, yp, zp, plo, dxi, i, j, k, W);
    int ic, jc, kc;
    amrex::Real Wc[AMREX_SPACEDIM][2];
    ablastr::particles::compute_weights<amrex::IndexType::CELL>(
        xp, yp, zp, plo, dxi, ic, jc, kc, Wc);

#if (defined WARPX_DIM_3D)
    amrex::RealVect normal{0.0, 0.0, 0.0};
    for (int iic = 0; iic < 2; ++iic) {
        for (int kk = 0; kk < 2; ++kk) {
            for (int jj=0; jj< 2; ++jj) {
                for (int ii = 0; ii < 2; ++ii) {
                    int const icstart = ic + iic;
                    amrex::Real const sign = (ii%2)*2._rt - 1._rt;
                    int const wccomp = static_cast<int>(iic%2);
                    int const w1comp = static_cast<int>(jj%2);
                    int const w2comp = static_cast<int>(kk%2);
                    normal[0] += sign * distance_to_eb(icstart + ii, j + jj, k + kk) * dxi[0] * Wc[0][wccomp] * W[1][w1comp] * W[2][w2comp];
                }
            }
        }
    }
    for (int iic = 0; iic < 2; ++iic) {
        for (int kk = 0; kk < 2; ++kk) {
            for (int ii=0; ii< 2; ++ii) {
                for (int jj = 0; jj < 2; ++jj) {
                    int const jcstart = jc + iic;
                    amrex::Real const sign = (jj%2)*2._rt - 1._rt;
                    int const wccomp = static_cast<int>(iic%2);
                    int const w1comp = static_cast<int>(ii%2);
                    int const w2comp = static_cast<int>(kk%2);
                    normal[1] += sign * distance_to_eb(i + ii, jcstart + jj, k + kk) * dxi[1] * W[0][w1comp] * Wc[1][wccomp] * W[2][w2comp];
                }
            }
        }
    }
    for (int iic = 0; iic < 2; ++iic) {
        for (int jj = 0; jj < 2; ++jj) {
            for (int ii=0; ii< 2; ++ii) {
                for (int kk = 0; kk < 2; ++kk) {
                    int const kcstart = kc + iic;
                    amrex::Real const sign = (kk%2)*2._rt - 1._rt;
                    int const wccomp = static_cast<int>(iic%2);
                    int const w1comp = static_cast<int>(ii%2);
                    int const w2comp = static_cast<int>(jj%2);
                    normal[2] += sign * distance_to_eb(i + ii, j + jj, kcstart + kk) * dxi[2] * W[0][w1comp] * W[1][w2comp] * Wc[2][wccomp];
                }
            }
        }
    }

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    amrex::RealVect normal{0.0, 0.0};
    for (int iic = 0; iic < 2; ++iic) {
        for (int jj=0; jj< 2; ++jj) {
            for (int ii = 0; ii < 2; ++ii) {
                int const icstart = ic + iic;
                amrex::Real const sign = (ii%2)*2._rt - 1._rt;
                int const wccomp = static_cast<int>(iic%2);
                int const w1comp = static_cast<int>(jj%2);
                normal[0] += sign * distance_to_eb(icstart + ii, j + jj, k) * dxi[0] * Wc[0][wccomp] * W[1][w1comp];
            }
        }
    }
    for (int iic = 0; iic < 2; ++iic) {
        for (int ii=0; ii< 2; ++ii) {
            for (int jj = 0; jj < 2; ++jj) {
                int const jcstart = jc + iic;
                amrex::Real const sign = (jj%2)*2._rt - 1._rt;
                int const wccomp = static_cast<int>(iic%2);
                int const w1comp = static_cast<int>(ii%2);
                normal[1] += sign * distance_to_eb(i + ii, jcstart + jj, k) * dxi[1] * W[0][w1comp] * Wc[1][wccomp];
            }
        }
    }
    amrex::ignore_unused(kc);

#else
    amrex::ignore_unused(i, j, k, ic, jc, kc, W, Wc, distance_to_eb, dxi);
    amrex::RealVect normal(0.0);

    AMREX_IF_ON_DEVICE((
        AMREX_DEVICE_ASSERT(0);
    ))
    AMREX_IF_ON_HOST((
        WARPX_ABORT_WITH_MESSAGE("Error: interp_normal not yet implemented in 1D");
    ))

#endif

    // Normalize to unit length
    amrex::Real const inv_norm = 1.0_rt / std::sqrt(dot_product(normal, normal));
    AMREX_D_DECL(normal[0] *= inv_norm,
                 normal[1] *= inv_norm,
                 normal[2] *= inv_norm);

    // Express the unit normal in 3D Cartesian coordinates, using the Cartesian
    // position (xp, yp, zp) to orient the radial / azimuthal components.
#if defined(WARPX_DIM_3D)
    return {amrex::ParticleReal(normal[0]), amrex::ParticleReal(normal[1]), amrex::ParticleReal(normal[2])};
#elif defined(WARPX_DIM_XZ)
    // normal is (n_x, n_z); the y direction is out-of-plane.
    return {amrex::ParticleReal(normal[0]), 0.0_prt, amrex::ParticleReal(normal[1])};
#elif defined(WARPX_DIM_RZ)
    // normal is (n_r, n_z); orient the radial component with the azimuth.
    amrex::ParticleReal const theta = std::atan2(yp, xp);
    return {amrex::ParticleReal(normal[0]) * std::cos(theta),
            amrex::ParticleReal(normal[0]) * std::sin(theta),
            amrex::ParticleReal(normal[1])};
#elif defined(WARPX_DIM_RCYLINDER)
    // normal is radial n_r; orient with the azimuth.
    amrex::ParticleReal const theta = std::atan2(yp, xp);
    return {amrex::ParticleReal(normal[0]) * std::cos(theta),
            amrex::ParticleReal(normal[0]) * std::sin(theta),
            0.0_prt};
#elif defined(WARPX_DIM_RSPHERE)
    // normal is radial n_r; orient with the position direction.
    amrex::ParticleReal const inv_r = 1.0_prt / std::sqrt(xp*xp + yp*yp + zp*zp);
    return {amrex::ParticleReal(normal[0]) * xp * inv_r,
            amrex::ParticleReal(normal[0]) * yp * inv_r,
            amrex::ParticleReal(normal[0]) * zp * inv_r};
#elif defined(WARPX_DIM_1D_Z)
    // normal is along z.
    return {0.0_prt, 0.0_prt, amrex::ParticleReal(normal[0])};
#endif
}
}
#endif // WARPX_DISTANCETOEB_H_
