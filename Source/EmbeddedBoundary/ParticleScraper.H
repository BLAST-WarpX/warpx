/* Copyright 2021 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLESCRAPER_H_
#define WARPX_PARTICLESCRAPER_H_

#include "EmbeddedBoundary/DistanceToEB.H"
#include "EmbeddedBoundary/ParticleBoundaryProcess.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/ParticleSinkContainer.H"
#include <ablastr/particles/NodalFieldGather.H>

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particle.H>
#include <AMReX_RandomEngine.H>
#include <AMReX_REAL.H>
#include <AMReX_TypeTraits.H>
#include <AMReX_Vector.H>

#include <type_traits>
#include <utility>



/**
 * \brief Interact particles with the embedded boundary walls.
 *
 *  This function detects which particles have entered into the region
 *  covered by the embedded boundaries and applies an operation on those
 *  that have. Boundary collision detection is performed using a signed
 *  distance function, which is generated automatically when WarpX is
 *  compiled with EB support.
 *
 *  The operation to be performed is specified by the callable function
 *  passed in to this function as an argument. This function can access the
 *  position at which the particle hit the boundary. Particles can be `absorbed`
 *  by setting their ids to negative
 *  to flag them for removal. Likewise, they can be reflected back into the domain
 *  by modifying their data appropriately and leaving their ids alone.
 *
 *  This version operates only at the specified level.
 *
 * \tparam pc a type of amrex ParticleContainer
 * \tparam F a callable type, e.g. a lambda function or functor
 *
 * \param pc the particle container to test for boundary interactions.
 * \param distance_to_eb a set of MultiFabs that store the signed distance function
 * \param lev the mesh refinement level to work on.
 * \param f the callable that defines what to do when a particle hits the boundary.
 *
 *        The form of the callable should model:
 *        template <typename PData>
 *        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 *        void operator() (PData& ptd, int i,
 *                         const SetParticlePosition<PIdx>& setPosition,
 *                         amrex::ParticleReal xp, amrex::ParticleReal yp, amrex::ParticleReal zp,
 *                         amrex::Array4<const amrex::Real> const& distance_to_eb,
 *                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
 *                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
 *                         amrex::RandomEngine const& engine);
 *
 *        where ptd is the particle tile, i the index of the particle operated on,
 *        setPosition writes a Cartesian position back to the particle (handling the
 *        reduced-dimension conversion), xp/yp/zp the Cartesian particle position,
 *        distance_to_eb the signed distance array, plo/dxi the geometry data, and
 *        engine for random number generation.
 */
template <class PC, class F, std::enable_if_t<amrex::IsParticleContainer<PC>::value, int> foo = 0>
void
scrapeParticlesAtEB (PC& pc, ablastr::fields::MultiLevelScalarField const& distance_to_eb, int lev, F&& f)
{
    scrapeParticlesAtEB(pc, distance_to_eb, lev, lev, std::forward<F>(f));
}

/**
 * \brief Interact particles with the embedded boundary walls.
 *
 *  This function detects which particles have entered into the region
 *  covered by the embedded boundaries and applies an operation on those
 *  that have. Boundary collision detection is performed using a signed
 *  distance function, which is generated automatically when WarpX is
 *  compiled with EB support.
 *
 *  The operation to be performed is specified by the callable function
 *  passed in to this function as an argument. This function can access the
 *  position at which the particle hit the boundary. Particles can be `absorbed`
 *  by setting their ids to negative
 *  to flag them for removal. Likewise, they can be reflected back into the domain
 *  by modifying their data appropriately and leaving their ids alone.
 *
 *  This version operates over all the levels in the pc.
 *
 * \tparam pc a type of amrex ParticleContainer
 * \tparam F a callable type, e.g. a lambda function or functor
 *
 * \param pc the particle container to test for boundary interactions.
 * \param distance_to_eb a set of MultiFabs that store the signed distance function
 * \param f the callable that defines what to do when a particle hits the boundary.
 *
 *        The form of the callable should model:
 *        template <typename PData>
 *        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 *        void operator() (PData& ptd, int i,
 *                         const SetParticlePosition<PIdx>& setPosition,
 *                         amrex::ParticleReal xp, amrex::ParticleReal yp, amrex::ParticleReal zp,
 *                         amrex::Array4<const amrex::Real> const& distance_to_eb,
 *                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
 *                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
 *                         amrex::RandomEngine const& engine);
 *
 *        where ptd is the particle tile, i the index of the particle operated on,
 *        setPosition writes a Cartesian position back to the particle (handling the
 *        reduced-dimension conversion), xp/yp/zp the Cartesian particle position,
 *        distance_to_eb the signed distance array, plo/dxi the geometry data, and
 *        engine for random number generation.
 */
template <class PC, class F, std::enable_if_t<amrex::IsParticleContainer<PC>::value, int> foo = 0>
void
scrapeParticlesAtEB (PC& pc, ablastr::fields::MultiLevelScalarField const& distance_to_eb, F&& f)
{
    scrapeParticlesAtEB(pc, distance_to_eb, 0, pc.finestLevel(), std::forward<F>(f));
}

/**
 * \brief Interact particles with the embedded boundary walls.
 *
 *  This function detects which particles have entered into the region
 *  covered by the embedded boundaries and applies an operation on those
 *  that have. Boundary collision detection is performed using a signed
 *  distance function, which is generated automatically when WarpX is
 *  compiled with EB support.
 *
 *  The operation to be performed is specified by the callable function
 *  passed in to this function as an argument. This function can access the
 *  position at which the particle hit the boundary. Particles can be `absorbed`
 *  by setting their ids to negative
 *  to flag them for removal. Likewise, they can be reflected back into the domain
 *  by modifying their data appropriately and leaving their ids alone.
 *
 *  This version operates only at the specified levels.
 *
 * \tparam pc a type of amrex ParticleContainer
 * \tparam F a callable type, e.g. a lambda function or functor
 *
 * \param pc the particle container to test for boundary interactions.
 * \param distance_to_eb a set of MultiFabs that store the signed distance function
 * \param lev_min the minimum mesh refinement level to work on.
 * \param lev_max the maximum mesh refinement level to work on.
 * \param f the callable that defines what to do when a particle hits the boundary.
 *
 *        The form of the callable should model:
 *        template <typename PData>
 *        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 *        void operator() (PData& ptd, int i,
 *                         const SetParticlePosition<PIdx>& setPosition,
 *                         amrex::ParticleReal xp, amrex::ParticleReal yp, amrex::ParticleReal zp,
 *                         amrex::Array4<const amrex::Real> const& distance_to_eb,
 *                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
 *                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
 *                         amrex::RandomEngine const& engine);
 *
 *        where ptd is the particle tile, i the index of the particle operated on,
 *        setPosition writes a Cartesian position back to the particle (handling the
 *        reduced-dimension conversion), xp/yp/zp the Cartesian particle position,
 *        distance_to_eb the signed distance array, plo/dxi the geometry data, and
 *        engine for random number generation.
 */
template <class PC, class F, std::enable_if_t<amrex::IsParticleContainer<PC>::value, int> foo = 0>
void
scrapeParticlesAtEB (PC& pc, ablastr::fields::MultiLevelScalarField const& distance_to_eb,
                 int lev_min, int lev_max, F&& f)
{
    amrex::Vector<ablastr::fields::MultiLevelScalarField const*> const eb_vec = { &distance_to_eb };
    scrapeParticlesAtEB(pc, eb_vec, lev_min, lev_max, std::forward<F>(f));
}
/**
 * \brief Interact particles with the embedded boundary walls.
 *
 *  This function detects which particles have entered into the region
 *  covered by the embedded boundaries and applies an operation on those
 *  that have. Boundary collision detection is performed using a signed
 *  distance function, which is generated automatically when WarpX is
 *  compiled with EB support.
 *
 *  The operation to be performed is specified by the callable function
 *  passed in to this function as an argument. This function can access the
 *  position at which the particle hit the boundary. Particles can be `absorbed`
 *  by setting their ids to negative
 *  to flag them for removal. Likewise, they can be reflected back into the domain
 *  by modifying their data appropriately and leaving their ids alone.
 *
 *  This version operates only at the specified levels.
 *
 * \tparam pc a type of amrex ParticleContainer
 * \tparam F a callable type, e.g. a lambda function or functor
 *
 * \param pc the particle container to test for boundary interactions.
 * \param distance_to_eb a vector of a set of MultiFabs that store the signed distance function
 * \param lev_min the minimum mesh refinement level to work on.
 * \param lev_max the maximum mesh refinement level to work on.
 * \param f the callable that defines what to do when a particle hits the boundary.
 *
 *        The form of the callable should model:
 *        template <typename PData>
 *        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 *        void operator() (PData& ptd, int i,
 *                         const SetParticlePosition<PIdx>& setPosition,
 *                         amrex::ParticleReal xp, amrex::ParticleReal yp, amrex::ParticleReal zp,
 *                         amrex::Array4<const amrex::Real> const& distance_to_eb,
 *                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
 *                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
 *                         amrex::RandomEngine const& engine);
 *
 *        where ptd is the particle tile, i the index of the particle operated on,
 *        setPosition writes a Cartesian position back to the particle (handling the
 *        reduced-dimension conversion), xp/yp/zp the Cartesian particle position,
 *        distance_to_eb the signed distance array, plo/dxi the geometry data, and
 *        engine for random number generation.
 */
template <class PC, class F, std::enable_if_t<amrex::IsParticleContainer<PC>::value, int> foo = 0>
void
scrapeParticlesAtEB (PC& pc,
                     amrex::Vector<ablastr::fields::MultiLevelScalarField const*> const& distance_to_eb,
                     int lev_min, int lev_max, F&& f)
{
    BL_PROFILE("scrapeParticlesAtEB");

    const amrex::Long num_ebs = distance_to_eb.size();
    if (num_ebs == 0) { return; }

    // Safety check for fixed GpuArray size
    constexpr int MAX_SINKS = 8;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        num_ebs <= MAX_SINKS,
        "Exceeded maximum supported particle sinks (8) in scrapeParticlesAtEB."
    );

    for (int lev = lev_min; lev <= lev_max; ++lev)
    {
        const auto plo = pc.Geom(lev).ProbLoArray();
        const auto dxi = pc.Geom(lev).InvCellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for(WarpXParIter pti(pc, lev); pti.isValid(); ++pti)
        {
            const auto getPosition = GetParticlePosition<PIdx>(pti);
            const auto setPosition = SetParticlePosition<PIdx>(pti);
            auto& tile = pti.GetParticleTile();
            auto ptd = tile.getParticleTileData();
            const auto np = tile.numParticles();
            // 1. Pack Array4 views into fixed-size GpuArray (Stack allocated, captured by value)
            amrex::GpuArray<amrex::Array4<const amrex::Real>, MAX_SINKS> eb_arrs;
            for (int n = 0; n < num_ebs; ++n) {
                eb_arrs[n] = (*(*distance_to_eb[n])[lev])[pti].array();
            }
            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (const int ip, amrex::RandomEngine const& engine) noexcept
            {
                // skip particles that are already flagged for removal
                if (!amrex::ParticleIDWrapper{ptd.m_idcpu[ip]}.is_valid()) { return; }

                amrex::ParticleReal xp, yp, zp;
                getPosition(ip, xp, yp, zp);

                int i, j, k;
                amrex::Real W[AMREX_SPACEDIM][2];
                ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                    xp, yp, zp, plo, dxi, i, j, k, W);
                // 2. Test particle against each distance field in order
                for (int n = 0; n < num_ebs; ++n)
                {
                    auto const& distance_to_eb_arr = eb_arrs[n];
                    amrex::Real const distance_to_eb_value =
                        ablastr::particles::interp_field_nodal(i, j, k, W, distance_to_eb_arr);

                    if (distance_to_eb_value < 0.0)
                    {
                        f(ptd, ip, setPosition, xp, yp, zp, distance_to_eb_arr, plo, dxi, engine);
                        break; // Stop evaluating remaining EBs for this particle
                    }
                }
            });
        }
    }
}

template <class PC, std::enable_if_t<amrex::IsParticleContainer<PC>::value, int> foo = 0>
void
scrapeParticlesAtEB (PC& pc,
                     ParticleSinkContainer const& particle_sinks,
                     int lev_min, int lev_max)
{
    BL_PROFILE("scrapeParticlesAtEB");

    // 1. Collect non-disabled sinks
    const auto active_sinks = particle_sinks.get_active_sinks();

    const amrex::Long num_ebs = active_sinks.size();
    if (num_ebs == 0) { return; }

    constexpr int MAX_SINKS = 8;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        num_ebs <= MAX_SINKS,
        "Exceeded maximum supported particle sinks (8) in scrapeParticlesAtEB."
    );

    // 2. Extract sink types ONCE (Global for all levels/tiles)
    amrex::GpuArray<ParticleBoundaryType, MAX_SINKS> sink_types;
    for (int n = 0; n < num_ebs; ++n) {
        sink_types[n] = active_sinks[n]->type;
    }

    auto& warpx = WarpX::GetInstance();
    amrex::ParticleReal const mass = pc.getMass();
    amrex::ParticleReal const uth = pc.getBoundaryThermalVelocity();

    for (int lev = lev_min; lev <= lev_max; ++lev)
    {
        const auto plo = pc.Geom(lev).ProbLoArray();
        const auto dxi = pc.Geom(lev).InvCellSizeArray();
        amrex::Real const dt_lev = warpx.getdt(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (WarpXParIter pti(pc, lev); pti.isValid(); ++pti)
        {
            const auto getPosition = GetParticlePosition<PIdx>(pti);
            const auto setPosition = SetParticlePosition<PIdx>(pti);
            auto& tile = pti.GetParticleTile();
            auto ptd = tile.getParticleTileData();
            const auto np = tile.numParticles();

            // 3. Extract tile-specific Array4 views (1-line host setup)
            amrex::GpuArray<amrex::Array4<const amrex::Real>, MAX_SINKS> eb_arrs;
            for (int n = 0; n < num_ebs; ++n) {
                eb_arrs[n] = (*active_sinks[n]->distance_field[lev])[pti].array();
            }

            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (const int ip, amrex::RandomEngine const& engine) noexcept
            {
                if (!amrex::ParticleIDWrapper{ptd.m_idcpu[ip]}.is_valid()) { return; }

                amrex::ParticleReal xp, yp, zp;
                getPosition(ip, xp, yp, zp);

                int i, j, k;
                amrex::Real W[AMREX_SPACEDIM][2];
                ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                    xp, yp, zp, plo, dxi, i, j, k, W);

                // 4. Test particle distance & construct functor on-the-fly
                for (int n = 0; n < num_ebs; ++n)
                {
                    auto const& distance_to_eb_arr = eb_arrs[n];
                    amrex::Real const distance_to_eb_value =
                        ablastr::particles::interp_field_nodal(i, j, k, W, distance_to_eb_arr);

                    if (distance_to_eb_value < 0.0)
                    {
                        if (sink_types[n] == ParticleBoundaryType::Absorbing) {
                            ParticleBoundaryProcess::Absorb{}(
                                ptd, ip, setPosition, xp, yp, zp, distance_to_eb_arr, plo, dxi, engine);
                        } else {
                            ParticleBoundaryProcess::ParticleBoundaryInteraction{
                                dt_lev, mass, sink_types[n], uth
                            }(ptd, ip, setPosition, xp, yp, zp, distance_to_eb_arr, plo, dxi, engine);
                        }
                        break;
                    }
                }
            });
        }
    }
}


#endif //WARPX_PARTICLESCRAPER_H_
