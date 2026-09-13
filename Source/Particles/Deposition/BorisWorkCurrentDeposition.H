/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_DEPOSITION_BORISWORKCURRENTDEPOSITION_H_
#define WARPX_PARTICLES_DEPOSITION_BORISWORKCURRENTDEPOSITION_H_

#include "Particles/Pusher/UpdateMomentumBoris.H"
#include "Particles/ShapeFactors.H"

#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

#include <cmath>

/** \brief Scatter one particle's exact Boris electric-work current.
 *
 * This is the transpose of the Cartesian, collocated, direct
 * `doGatherShapeN<depos_order, 0>` electric-field gather.  The particle is
 * sampled at its supplied position (normally x^n), and the same full-order
 * shape is applied to all three component grids.  `q_weight` is the signed
 * particle charge multiplied by its macroparticle weight.  The deposited
 * quantity is
 *
 *     J_work,g += q_weight * v_work * S_g(x^n) / cell_volume.
 *
 * The caller owns the outer particle loop and must use `amrex::For`, not
 * `amrex::ParallelFor`: different particles can scatter to the same node.
 * Host/device atomics protect shared nodes across CPU threads and GPU lanes,
 * but do not satisfy a CPU SIMD promise of independent loop iterations.
 * The three destination arrays must be collocated and have `field_type`.
 * In RZ (m=0), rotate the work velocity by the transpose of the gather's
 * cylindrical-to-Cartesian rotation. The native RZ caller passes invvol=1,
 * retaining integrated [C m/s] contributions until the metric-weighted
 * pressure-gradient transpose divides by native caloric node volumes.
 * Higher modes and other radial geometries remain unsupported.
 *
 * \tparam depos_order full particle shape order (0 through 4)
 */
template <int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doBorisWorkCurrentDepositionShapeN (
    [[maybe_unused]] amrex::ParticleReal const xp,
    [[maybe_unused]] amrex::ParticleReal const yp,
    [[maybe_unused]] amrex::ParticleReal const zp,
    amrex::ParticleReal const q_weight,
    BorisElectricWorkVelocity const& work_velocity,
    amrex::Array4<amrex::Real> const& jx_arr,
    amrex::Array4<amrex::Real> const& jy_arr,
    amrex::Array4<amrex::Real> const& jz_arr,
    amrex::IntVect const& field_type,
    amrex::XDim3 const& dinv,
    amrex::XDim3 const& xyzmin,
    amrex::Real const invvol,
    amrex::Dim3 const lo) noexcept
{
    static_assert(depos_order >= 0 && depos_order <= 4,
        "Boris work-current deposition supports particle shape orders 0-4");

#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    static_assert(depos_order < 0,
        "Boris work-current deposition supports Cartesian and m=0 RZ geometry only");
    amrex::ignore_unused(
        q_weight, work_velocity, jx_arr, jy_arr, jz_arr, field_type,
        dinv, xyzmin, invvol, lo);
#else
    constexpr int NODE = amrex::IndexType::NODE;
    Compute_shape_factor<depos_order> const compute_shape_factor;

#if defined(WARPX_DIM_RZ)
    amrex::Real const radius = static_cast<amrex::Real>(std::sqrt(xp*xp + yp*yp));
    amrex::Real const cosine = radius > amrex::Real(0.0)
        ? static_cast<amrex::Real>(xp / radius) : amrex::Real(1.0);
    amrex::Real const sine = radius > amrex::Real(0.0)
        ? static_cast<amrex::Real>(yp / radius) : amrex::Real(0.0);
    amrex::Real const wqvx = static_cast<amrex::Real>(q_weight)
        * (cosine * work_velocity.x + sine * work_velocity.y) * invvol;
    amrex::Real const wqvy = static_cast<amrex::Real>(q_weight)
        * (-sine * work_velocity.x + cosine * work_velocity.y) * invvol;
#else
    amrex::Real const wqvx = static_cast<amrex::Real>(
        q_weight * work_velocity.x) * invvol;
    amrex::Real const wqvy = static_cast<amrex::Real>(
        q_weight * work_velocity.y) * invvol;
#endif
    amrex::Real const wqvz = static_cast<amrex::Real>(
        q_weight * work_velocity.z) * invvol;

#if !defined(WARPX_DIM_1D_Z)
    // Match the gather arithmetic and precision exactly: FieldGather.H forms
    // these coordinates and shape factors in amrex::Real.
#if defined(WARPX_DIM_RZ)
    amrex::Real const x = (radius - xyzmin.x) * dinv.x;
#else
    amrex::Real const x = (xp - xyzmin.x) * dinv.x;
#endif
    amrex::Real sx[depos_order + 1] = {0.0};
    int const j = compute_shape_factor(
        sx, x - (field_type[0] == NODE
            ? amrex::Real(0.0) : amrex::Real(0.5)));
#endif

#if defined(WARPX_DIM_3D)
    amrex::Real const y = (yp - xyzmin.y) * dinv.y;
    amrex::Real sy[depos_order + 1] = {0.0};
    int const k = compute_shape_factor(
        sy, y - (field_type[1] == NODE
            ? amrex::Real(0.0) : amrex::Real(0.5)));
#endif

    constexpr int zdir = WARPX_ZINDEX;
    amrex::Real const z = (zp - xyzmin.z) * dinv.z;
    amrex::Real sz[depos_order + 1] = {0.0};
    int const l = compute_shape_factor(
        sz, z - (field_type[zdir] == NODE
            ? amrex::Real(0.0) : amrex::Real(0.5)));

#if defined(WARPX_DIM_1D_Z)
    for (int iz = 0; iz <= depos_order; ++iz) {
        amrex::Real const shape = sz[iz];
        amrex::HostDevice::Atomic::Add(
            &jx_arr(lo.x + l + iz, 0, 0, 0), shape * wqvx);
        amrex::HostDevice::Atomic::Add(
            &jy_arr(lo.x + l + iz, 0, 0, 0), shape * wqvy);
        amrex::HostDevice::Atomic::Add(
            &jz_arr(lo.x + l + iz, 0, 0, 0), shape * wqvz);
    }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    for (int iz = 0; iz <= depos_order; ++iz) {
        for (int ix = 0; ix <= depos_order; ++ix) {
            amrex::Real const shape = sx[ix] * sz[iz];
            amrex::HostDevice::Atomic::Add(
                &jx_arr(lo.x + j + ix, lo.y + l + iz, 0, 0),
                shape * wqvx);
            amrex::HostDevice::Atomic::Add(
                &jy_arr(lo.x + j + ix, lo.y + l + iz, 0, 0),
                shape * wqvy);
            amrex::HostDevice::Atomic::Add(
                &jz_arr(lo.x + j + ix, lo.y + l + iz, 0, 0),
                shape * wqvz);
        }
    }
#elif defined(WARPX_DIM_3D)
    for (int iz = 0; iz <= depos_order; ++iz) {
        for (int iy = 0; iy <= depos_order; ++iy) {
            for (int ix = 0; ix <= depos_order; ++ix) {
                amrex::Real const shape = sx[ix] * sy[iy] * sz[iz];
                amrex::HostDevice::Atomic::Add(
                    &jx_arr(lo.x + j + ix, lo.y + k + iy, lo.z + l + iz),
                    shape * wqvx);
                amrex::HostDevice::Atomic::Add(
                    &jy_arr(lo.x + j + ix, lo.y + k + iy, lo.z + l + iz),
                    shape * wqvy);
                amrex::HostDevice::Atomic::Add(
                    &jz_arr(lo.x + j + ix, lo.y + k + iy, lo.z + l + iz),
                    shape * wqvz);
            }
        }
    }
#endif
#endif
}

#endif // WARPX_PARTICLES_DEPOSITION_BORISWORKCURRENTDEPOSITION_H_
