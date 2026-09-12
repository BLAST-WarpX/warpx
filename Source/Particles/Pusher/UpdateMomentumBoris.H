/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_H_
#define WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_H_

#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

/** The velocity that makes the two electric Boris kicks an exact discrete
 *  kinetic-energy work pair.  For a full push,
 *
 *      delta K = q dt E dot electric_work_velocity.
 *
 *  The three components are kept as named scalars so this type remains a
 *  trivial host/device kernel value. */
struct BorisElectricWorkVelocity
{
    amrex::ParticleReal x = 0.0;
    amrex::ParticleReal y = 0.0;
    amrex::ParticleReal z = 0.0;
};

/** \brief Push the particle's positions over one timestep,
 *    given the value of its momenta `ux`, `uy`, `uz`
 *    Note that UpdateMomentumBoris algorithm can be split into FirstHalf and SecondHalf
 *    momentum updates. FirstHalf and SecondHalf updates are constructed so that
 *    performing FirstHalf followed by SecondHalf is mathematically
 *    equivalent to a single Full update. */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdateMomentumBoris(
    amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz,
    const amrex::ParticleReal Ex, const amrex::ParticleReal Ey, const amrex::ParticleReal Ez,
    const amrex::ParticleReal Bx, const amrex::ParticleReal By, const amrex::ParticleReal Bz,
    const amrex::ParticleReal q, const amrex::ParticleReal m, const amrex::Real dt, MomentumPushType momentum_push_type = MomentumPushType::Full)
{
    using namespace amrex::literals;

    const amrex::ParticleReal econst = 0.5_prt*q*dt/m;

    if (momentum_push_type == MomentumPushType::FirstHalf || momentum_push_type == MomentumPushType::Full) {
        // First half-push for E
        ux += econst*Ex;
        uy += econst*Ey;
        uz += econst*Ez;
    }
    // Compute temporary gamma factor
    constexpr auto inv_c2 = PhysConst::inv_c2_v< amrex::ParticleReal>;
    const amrex::ParticleReal inv_gamma = 1._prt/std::sqrt(1._prt + (ux*ux + uy*uy + uz*uz)*inv_c2);
    // Magnetic rotation
    // - Compute temporary variables
    amrex::ParticleReal tx = econst*inv_gamma*Bx;
    amrex::ParticleReal ty = econst*inv_gamma*By;
    amrex::ParticleReal tz = econst*inv_gamma*Bz;
    if (momentum_push_type == MomentumPushType::FirstHalf || momentum_push_type == MomentumPushType::SecondHalf) {
        // For a full push, the Boris algorithm rotates the momentum about the vector t,
        // by an angle alpha that satisfies tan(alpha/2) = |t| = dt q B /(2 gamma m).
        // (See Birdsall and Langdon, "Plasma Physics via Computer Simulation", section 4-3, eq. 10)
        // For half pushes, the combination of the first and second half push should result in a rotation by the same alpha.
        // Therefore, each half push should use a modified t that results in a rotation by only alpha/2.
        // Using the identity tan(alpha/2) = 2 tan(alpha/4) / (1 - tan(alpha/4)^2), one can see that t should be
        // rescaled so that |t_half|/|t_full| = tan(alpha/4)/tan(alpha/2) = (sqrt(1 + |t_full|^2) - 1) / |t_full|^2.
        const amrex::ParticleReal tsq = tx*tx + ty*ty + tz*tz;
        const amrex::ParticleReal factor = (tsq > 0._prt) ? (std::sqrt(1._prt + tsq) - 1._prt) / tsq : 0.5_prt;
        tx *= factor;
        ty *= factor;
        tz *= factor;
    }
    const amrex::ParticleReal tsqi = 2._prt/(1._prt + tx*tx + ty*ty + tz*tz);
    const amrex::ParticleReal sx = tx*tsqi;
    const amrex::ParticleReal sy = ty*tsqi;
    const amrex::ParticleReal sz = tz*tsqi;
    const amrex::ParticleReal ux_p = ux + uy*tz - uz*ty;
    const amrex::ParticleReal uy_p = uy + uz*tx - ux*tz;
    const amrex::ParticleReal uz_p = uz + ux*ty - uy*tx;
    // - Update momentum
    ux += uy_p*sz - uz_p*sy;
    uy += uz_p*sx - ux_p*sz;
    uz += ux_p*sy - uy_p*sx;
    if (momentum_push_type == MomentumPushType::SecondHalf || momentum_push_type == MomentumPushType::Full) {
    // Second half-push for E
        ux += econst*Ex;
        uy += econst*Ey;
        uz += econst*Ez;
    }
}

/** \brief Boris momentum push with its exact discrete electric-work velocity.
 *
 * This is an instrumented counterpart of UpdateMomentumBoris.  Its momentum
 * arithmetic and operation ordering intentionally mirror that routine.  The
 * returned velocity is
 *
 *   1/2 [ (u_0 + u_a)/(gamma_0 + gamma_a)
 *       + (u_b + u_1)/(gamma_b + gamma_1) ],
 *
 * where u_a and u_b bracket the magnetic rotation.  Consequently, the
 * relativistic kinetic-energy change of a full push is q dt E dot v_work up
 * to floating-point roundoff from the energy diagnostic itself.  The same
 * velocity can attribute that work additively to any decomposition of E.
 *
 * UpdateMomentumBoris remains unchanged, so callers that do not request work
 * instrumentation retain their original code path. */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
BorisElectricWorkVelocity UpdateMomentumBorisWithElectricWorkVelocity (
    amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz,
    amrex::ParticleReal const Ex, amrex::ParticleReal const Ey,
    amrex::ParticleReal const Ez, amrex::ParticleReal const Bx,
    amrex::ParticleReal const By, amrex::ParticleReal const Bz,
    amrex::ParticleReal const q, amrex::ParticleReal const m,
    amrex::Real const dt,
    MomentumPushType const momentum_push_type = MomentumPushType::Full)
{
    using namespace amrex::literals;

    BorisElectricWorkVelocity work_velocity;
    constexpr auto inv_c2 = PhysConst::inv_c2_v<amrex::ParticleReal>;
    amrex::ParticleReal const econst = 0.5_prt * q * dt / m;

    if (momentum_push_type == MomentumPushType::FirstHalf
        || momentum_push_type == MomentumPushType::Full)
    {
        amrex::ParticleReal const ux_before = ux;
        amrex::ParticleReal const uy_before = uy;
        amrex::ParticleReal const uz_before = uz;
        amrex::ParticleReal const gamma_before = std::sqrt(
            1.0_prt + (ux_before * ux_before + uy_before * uy_before
                + uz_before * uz_before) * inv_c2);

        // First half-push for E. Keep these operations identical to the
        // uninstrumented routine above.
        ux += econst * Ex;
        uy += econst * Ey;
        uz += econst * Ez;

        amrex::ParticleReal const gamma_after = std::sqrt(
            1.0_prt + (ux * ux + uy * uy + uz * uz) * inv_c2);
        amrex::ParticleReal const inverse_gamma_sum =
            0.5_prt / (gamma_before + gamma_after);
        work_velocity.x += (ux_before + ux) * inverse_gamma_sum;
        work_velocity.y += (uy_before + uy) * inverse_gamma_sum;
        work_velocity.z += (uz_before + uz) * inverse_gamma_sum;
    }

    // Magnetic rotation. Keep this block operation-for-operation identical to
    // UpdateMomentumBoris so enabling the work output cannot alter the orbit.
    amrex::ParticleReal const inv_gamma = 1.0_prt / std::sqrt(
        1.0_prt + (ux * ux + uy * uy + uz * uz) * inv_c2);
    amrex::ParticleReal tx = econst * inv_gamma * Bx;
    amrex::ParticleReal ty = econst * inv_gamma * By;
    amrex::ParticleReal tz = econst * inv_gamma * Bz;
    if (momentum_push_type == MomentumPushType::FirstHalf
        || momentum_push_type == MomentumPushType::SecondHalf)
    {
        amrex::ParticleReal const tsq = tx * tx + ty * ty + tz * tz;
        amrex::ParticleReal const factor = tsq > 0.0_prt
            ? (std::sqrt(1.0_prt + tsq) - 1.0_prt) / tsq
            : 0.5_prt;
        tx *= factor;
        ty *= factor;
        tz *= factor;
    }
    amrex::ParticleReal const tsqi =
        2.0_prt / (1.0_prt + tx * tx + ty * ty + tz * tz);
    amrex::ParticleReal const sx = tx * tsqi;
    amrex::ParticleReal const sy = ty * tsqi;
    amrex::ParticleReal const sz = tz * tsqi;
    amrex::ParticleReal const ux_p = ux + uy * tz - uz * ty;
    amrex::ParticleReal const uy_p = uy + uz * tx - ux * tz;
    amrex::ParticleReal const uz_p = uz + ux * ty - uy * tx;
    ux += uy_p * sz - uz_p * sy;
    uy += uz_p * sx - ux_p * sz;
    uz += ux_p * sy - uy_p * sx;

    if (momentum_push_type == MomentumPushType::SecondHalf
        || momentum_push_type == MomentumPushType::Full)
    {
        amrex::ParticleReal const ux_before = ux;
        amrex::ParticleReal const uy_before = uy;
        amrex::ParticleReal const uz_before = uz;
        amrex::ParticleReal const gamma_before = std::sqrt(
            1.0_prt + (ux_before * ux_before + uy_before * uy_before
                + uz_before * uz_before) * inv_c2);

        // Second half-push for E.
        ux += econst * Ex;
        uy += econst * Ey;
        uz += econst * Ez;

        amrex::ParticleReal const gamma_after = std::sqrt(
            1.0_prt + (ux * ux + uy * uy + uz * uz) * inv_c2);
        amrex::ParticleReal const inverse_gamma_sum =
            0.5_prt / (gamma_before + gamma_after);
        work_velocity.x += (ux_before + ux) * inverse_gamma_sum;
        work_velocity.y += (uy_before + uy) * inverse_gamma_sum;
        work_velocity.z += (uz_before + uz) * inverse_gamma_sum;
    }

    return work_velocity;
}
#endif // WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_H_
