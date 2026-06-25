/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Marco Acciarri, Prabhat Kumar (Helion Energy Inc.)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_QDSMC_K_H_
#define WARPX_QDSMC_K_H_

#include "Utils/WarpXConst.H"

#include <ablastr/particles/NodalFieldGather.H>

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>


/**
 * @file Qdsmc_K.H
 *
 * Kernel primitives for the QDSMC (Quiet Direct Simulation Method) Lagrangian
 * transport step used to solve the electron-energy equation in the hybrid-PIC
 * solver. The QDSMC particles are fictitious markers (one per grid cell) that
 * carry the entropy K_e and electron weight N_e of the host cell, advect with
 * the electron fluid velocity V_e for one step, and re-scatter to the grid.
 *
 * Particles in WarpX are always 3D (xp, yp, zp). Field staggering depends on
 * the compile-time dimension (1D_Z / XZ / RZ / 3D). All kernels below are
 * written in terms of `ablastr::particles::compute_weights` to inherit the
 * dim-agnostic position-to-index mapping; the corner-loop body is gated by
 * `#if AMREX_SPACEDIM == N` (rather than `if constexpr`) because the AMReX
 * `IntVect` alias is dim-fixed at build time and constructing it with more
 * args than the build dim fails to parse, even inside a discarded branch.
 *
 * Convention: the QDSMC grid fields (n_e, K_e, weights) are stored with NODAL
 * staggering. This is consistent with the original Helion implementation and
 * also avoids a half-cell offset that the original kernel had from mixing
 * cell-centered indexing with nodal field storage.
 */


/**
 * @brief Gather (n_e * V_cell) and (K_e * smeared shape) at a QDSMC particle
 *        position from nodal grid fields, and pack them into the per-particle
 *        attributes used by the QDSMC entropy transport.
 *
 * At the start of the QDSMC step the particle sits at its home location
 * (cell center); compute_weights\<NODE\> with cell-centered home gives
 * shape factors W[d][0] = W[d][1] = 0.5 per dim, so the gather is a
 * proper linear interpolation between surrounding nodes.
 *
 * After the call:
 *   - n_p  = sum_{nodes around xp} shape * (rho / q_e) * V_cell
 *            (i.e. the electron count contributed to the cell by the
 *            interpolated nodal n_e values).
 *   - kn_p = K_e(xp) * n_p
 *            (i.e. the weighted entropy attached to this particle, with
 *            the same weight n_p that DepositField will scatter.)
 *
 * @param[in]  xp,yp,zp particle position (always 3D)
 * @param[out] n_p      electron count weighted attribute
 * @param[out] kn_p     entropy weighted attribute (K_e * n_p)
 * @param[in]  rho_arr  ion charge density Array4 (= q_e * n_e under quasi-neutrality)
 * @param[in]  K_arr    electron entropy K_e Array4
 * @param[in]  plo      domain lower bound (per dim)
 * @param[in]  dxi      inverse cell size (per dim)
 * @param[in]  cell_volume cell volume in physical units (product of dx over field dims)
 * @param[in]  bx       tile box (including ghost cells) used to clip out-of-tile writes
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void gather_density_entropy (const amrex::ParticleReal xp,
                             const amrex::ParticleReal yp,
                             const amrex::ParticleReal zp,
                             amrex::Real & n_p,
                             amrex::Real & kn_p,
                             amrex::Array4<amrex::Real const> const & rho_arr,
                             amrex::Array4<amrex::Real const> const & K_arr,
                             amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & plo,
                             amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & dxi,
                             amrex::Real const cell_volume,
                             amrex::Box const & bx)
{
    int i, j, k;
    amrex::Real W[AMREX_SPACEDIM][2];
    ablastr::particles::compute_weights<amrex::IndexType::CellIndex::NODE>(
        xp, yp, zp, plo, dxi, i, j, k, W);

    amrex::Real n_local = amrex::Real(0);
    amrex::Real K_local = amrex::Real(0);

#if AMREX_SPACEDIM == 3
    for (int ioff = 0; ioff < 2; ++ioff) {
        for (int joff = 0; joff < 2; ++joff) {
            for (int koff = 0; koff < 2; ++koff) {
                amrex::IntVect const c(i + ioff, j + joff, k + koff);
                if (bx.contains(c)) {
                    amrex::Real const w = W[0][ioff] * W[1][joff] * W[2][koff];
                    n_local += w * rho_arr(c);
                    K_local += w * K_arr(c);
                }
            }
        }
    }
#elif AMREX_SPACEDIM == 2
    for (int ioff = 0; ioff < 2; ++ioff) {
        for (int joff = 0; joff < 2; ++joff) {
            amrex::IntVect const c(i + ioff, j + joff);
            if (bx.contains(c)) {
                amrex::Real const w = W[0][ioff] * W[1][joff];
                n_local += w * rho_arr(c);
                K_local += w * K_arr(c);
            }
        }
    }
#elif AMREX_SPACEDIM == 1
    for (int ioff = 0; ioff < 2; ++ioff) {
        amrex::IntVect const c(i + ioff);
        if (bx.contains(c)) {
            n_local += W[0][ioff] * rho_arr(c);
            K_local += W[0][ioff] * K_arr(c);
        }
    }
#else
#   error "gather_density_entropy: unsupported AMREX_SPACEDIM"
#endif

    n_p = n_local * cell_volume / PhysConst::q_e;
    kn_p = K_local * n_p;
}


/**
 * @brief Gather a 3-component vector field (typically V_e) at a QDSMC particle
 *        position via linear shape-factor weighted interpolation.
 *
 * The vector field components live at the same staggering as the scalar fields
 * used by gather_density_entropy. Particles always carry 3 velocity components
 * even in lower-dim simulations; the field gather skips offsets in dimensions
 * not represented by the field's spatial extent.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void gather_vector_field_qdsmc (const amrex::ParticleReal xp,
                                const amrex::ParticleReal yp,
                                const amrex::ParticleReal zp,
                                amrex::Real & fx_p,
                                amrex::Real & fy_p,
                                amrex::Real & fz_p,
                                amrex::Array4<amrex::Real const> const & fx_arr,
                                amrex::Array4<amrex::Real const> const & fy_arr,
                                amrex::Array4<amrex::Real const> const & fz_arr,
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & plo,
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & dxi,
                                amrex::Box const & bx)
{
    int i, j, k;
    amrex::Real W[AMREX_SPACEDIM][2];
    ablastr::particles::compute_weights<amrex::IndexType::CellIndex::NODE>(
        xp, yp, zp, plo, dxi, i, j, k, W);

#if AMREX_SPACEDIM == 3
    for (int ioff = 0; ioff < 2; ++ioff) {
        for (int joff = 0; joff < 2; ++joff) {
            for (int koff = 0; koff < 2; ++koff) {
                amrex::IntVect const c(i + ioff, j + joff, k + koff);
                if (bx.contains(c)) {
                    amrex::Real const w = W[0][ioff] * W[1][joff] * W[2][koff];
                    fx_p += w * fx_arr(c);
                    fy_p += w * fy_arr(c);
                    fz_p += w * fz_arr(c);
                }
            }
        }
    }
#elif AMREX_SPACEDIM == 2
    for (int ioff = 0; ioff < 2; ++ioff) {
        for (int joff = 0; joff < 2; ++joff) {
            amrex::IntVect const c(i + ioff, j + joff);
            if (bx.contains(c)) {
                amrex::Real const w = W[0][ioff] * W[1][joff];
                fx_p += w * fx_arr(c);
                fy_p += w * fy_arr(c);
                fz_p += w * fz_arr(c);
            }
        }
    }
#elif AMREX_SPACEDIM == 1
    for (int ioff = 0; ioff < 2; ++ioff) {
        amrex::IntVect const c(i + ioff);
        if (bx.contains(c)) {
            fx_p += W[0][ioff] * fx_arr(c);
            fy_p += W[0][ioff] * fy_arr(c);
            fz_p += W[0][ioff] * fz_arr(c);
        }
    }
#else
#   error "gather_vector_field_qdsmc: unsupported AMREX_SPACEDIM"
#endif
}


/**
 * @brief Forward-Euler push of a QDSMC particle by one PIC step.
 *
 * Particles are 3D in WarpX, so all three components are advanced; if a
 * spatial dimension is not represented in the field (e.g. y in 2D XZ),
 * the corresponding velocity component is harmless because the deposit
 * never indexes that direction.
 *
 * The caller is responsible for the CFL constraint |v| * dt < dx_field
 * (one cell per step) -- beyond that the deposit aliases.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void push_qdsmc_particle (const amrex::ParticleReal x0p,
                          const amrex::ParticleReal y0p,
                          const amrex::ParticleReal z0p,
                          const amrex::ParticleReal vxp,
                          const amrex::ParticleReal vyp,
                          const amrex::ParticleReal vzp,
                          amrex::Real & xp,
                          amrex::Real & yp,
                          amrex::Real & zp,
                          amrex::Real const dt)
{
    xp = x0p + vxp * dt;
    yp = y0p + vyp * dt;
    zp = z0p + vzp * dt;
}


/**
 * @brief Scatter a scalar particle quantity onto a nodal grid field via
 *        linear shape-factor weighted atomic add.
 *
 * Used twice per QDSMC step: once to scatter the weight N_e (= np_real) and
 * once to scatter the entropy K_e * N_e. After both deposits the new K_e
 * at each node is recovered by HybridQDSMCUpdateTe via the ratio
 *     K_e_new = (deposited K*N) / max(deposited N, floor) / V_cell
 * and T_e is then computed from the polytropic relation.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void do_deposit_scalar (amrex::Array4<amrex::Real> const & field_arr,
                        const amrex::ParticleReal xp,
                        const amrex::ParticleReal yp,
                        const amrex::ParticleReal zp,
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & plo,
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & dxi,
                        const amrex::ParticleReal valuep,
                        amrex::Box const & bx)
{
    int i, j, k;
    amrex::Real W[AMREX_SPACEDIM][2];
    ablastr::particles::compute_weights<amrex::IndexType::CellIndex::NODE>(
        xp, yp, zp, plo, dxi, i, j, k, W);

#if AMREX_SPACEDIM == 3
    for (int ioff = 0; ioff < 2; ++ioff) {
        for (int joff = 0; joff < 2; ++joff) {
            for (int koff = 0; koff < 2; ++koff) {
                amrex::IntVect const c(i + ioff, j + joff, k + koff);
                if (bx.contains(c)) {
                    amrex::Real const w = W[0][ioff] * W[1][joff] * W[2][koff];
                    amrex::Gpu::Atomic::AddNoRet(&field_arr(c), valuep * w);
                }
            }
        }
    }
#elif AMREX_SPACEDIM == 2
    for (int ioff = 0; ioff < 2; ++ioff) {
        for (int joff = 0; joff < 2; ++joff) {
            amrex::IntVect const c(i + ioff, j + joff);
            if (bx.contains(c)) {
                amrex::Real const w = W[0][ioff] * W[1][joff];
                amrex::Gpu::Atomic::AddNoRet(&field_arr(c), valuep * w);
            }
        }
    }
#elif AMREX_SPACEDIM == 1
    for (int ioff = 0; ioff < 2; ++ioff) {
        amrex::IntVect const c(i + ioff);
        if (bx.contains(c)) {
            amrex::Gpu::Atomic::AddNoRet(&field_arr(c), valuep * W[0][ioff]);
        }
    }
#else
#   error "do_deposit_scalar: unsupported AMREX_SPACEDIM"
#endif
}


#endif // WARPX_QDSMC_K_H_
