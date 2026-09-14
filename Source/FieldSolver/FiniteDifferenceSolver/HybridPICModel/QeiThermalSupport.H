/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include "Particles/WarpXParticleContainer.H"

#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParticleUtil.H>

#include <cmath>
#include <cstdint>
#include <memory>

namespace warpx::hybrid
{
/** Stateless sampling keyed by checkpointed exchange and persistent identity. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE std::uint64_t
qeiMix64 (std::uint64_t value) noexcept
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
qeiNormal (std::uint64_t seed, std::uint64_t counter, std::uint64_t species, std::uint64_t particle,
           std::uint64_t component) noexcept
{
    using namespace amrex::literals;
    auto key = qeiMix64(seed);
    key = qeiMix64(key ^ qeiMix64(counter));
    key = qeiMix64(key ^ qeiMix64(species));
    key = qeiMix64(key ^ qeiMix64(particle));
    key = qeiMix64(key ^ qeiMix64(component));
    // Open-interval uniforms avoid both logarithm endpoints in double precision.
    auto const u = (static_cast<amrex::Real>(qeiMix64(key) >> 12U) + 0.5_rt) * 0x1p-52_rt;
    auto const v = (static_cast<amrex::Real>(qeiMix64(key + 1U) >> 12U) + 0.5_rt) * 0x1p-52_rt;
    return std::sqrt(-2.0_rt * std::log(u)) * std::cos(6.2831853071795864769_rt * v);
}

/** Frozen cell moments for a realizable thermal exchange: N, N*u[3],
 * sum(w*|u-mean|^2), macroparticle count. The two-pass central variance
 * avoids subtracting the large bulk kinetic energy of an imploding shell.
 * These are transient proposal inputs, not independently evolved state. */
inline std::unique_ptr<amrex::MultiFab>
depositResolvedQeiMoments (WarpXParticleContainer& pc, int const lev,
                           amrex::Geometry const& geometry)
{
    using namespace amrex::literals;
    auto result = std::make_unique<amrex::MultiFab>(pc.ParticleBoxArray(lev),
                                                    pc.ParticleDistributionMap(lev), 6, 1);
    result->setVal(0.0_rt);
    auto const lower = geometry.ProbLoArray();
    auto const inverse_dx = geometry.InvCellSizeArray();
    for (int pass = 0; pass < 2; ++pass) {
        for (WarpXParIter pti(pc, lev); pti.isValid(); ++pti) {
            auto const particles = pti.GetParticleTile().getParticleTileData();
            auto const moments = result->array(pti);
            auto const* const weight = pti.GetAttribs(PIdx::w).dataPtr();
            auto const* const ux = pti.GetAttribs(PIdx::ux).dataPtr();
            auto const* const uy = pti.GetAttribs(PIdx::uy).dataPtr();
            auto const* const uz = pti.GetAttribs(PIdx::uz).dataPtr();
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
            auto const* const theta = pti.GetAttribs(PIdx::theta).dataPtr();
#endif
            // Particle scatter-adds must not promise independent SIMD writes.
            amrex::For(pti.numParticles(), [=] AMREX_GPU_DEVICE(long const ip) {
                if (!amrex::ParticleIDWrapper{particles.m_idcpu[ip]}.is_valid()) {
                    return;
                }
                auto const particle = WarpXParticleContainer::ParticleType(particles, ip);
                auto const [i, j, k] = amrex::getParticleCell(particle, lower, inverse_dx).dim3();
                auto const w = static_cast<amrex::Real>(weight[ip]);
                if (!(w > 0.0_rt)) {
                    return;
                }
                amrex::Real const u2 = uz[ip];
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                auto const cosine = std::cos(theta[ip]);
                auto const sine = std::sin(theta[ip]);
                amrex::Real const u0 = ux[ip] * cosine + uy[ip] * sine;
                amrex::Real const u1 = -ux[ip] * sine + uy[ip] * cosine;
#else
                amrex::Real const u0 = ux[ip];
                amrex::Real const u1 = uy[ip];
#endif
                if (pass == 0) {
                    amrex::Gpu::Atomic::Add(&moments(i, j, k, 0), w);
                    amrex::Gpu::Atomic::Add(&moments(i, j, k, 1), w * u0);
                    amrex::Gpu::Atomic::Add(&moments(i, j, k, 2), w * u1);
                    amrex::Gpu::Atomic::Add(&moments(i, j, k, 3), w * u2);
                    amrex::Gpu::Atomic::Add(&moments(i, j, k, 5), 1.0_rt);
                } else {
                    auto const total = moments(i, j, k, 0);
                    auto const d0 = u0 - moments(i, j, k, 1) / total;
                    auto const d1 = u1 - moments(i, j, k, 2) / total;
                    auto const d2 = u2 - moments(i, j, k, 3) / total;
                    amrex::Gpu::Atomic::Add(&moments(i, j, k, 4),
                                            w * (d0 * d0 + d1 * d1 + d2 * d2));
                }
            });
        }
        amrex::Gpu::streamSynchronize();
    }
    result->FillBoundary(geometry.periodicity());
    return result;
}
} // namespace warpx::hybrid
