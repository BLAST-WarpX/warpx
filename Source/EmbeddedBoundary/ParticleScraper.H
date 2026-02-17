/* Copyright 2021 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLESCRAPER_H_
#define WARPX_PARTICLESCRAPER_H_

#include "EmbeddedBoundary/DistanceToEB.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/UpdatePosition.H"


#include <ablastr/particles/NodalFieldGather.H>

#include <AMReX.H>
#include <AMReX_Algorithm.H>
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
 *  position at which the particle hit the boundary, and also the associated
 *  normal vector. Particles can be `absorbed` by setting their ids to negative
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
 *        void operator() (const PData& ptd, int i,
 *                         const amrex::RealVect& pos, const amrex::RealVect& normal,
 *                         amrex::RandomEngine const& engine);
 *
 *        where ptd is the particle tile, i the index of the particle operated on,
 *        pos and normal the location of the collision and the boundary normal vector.
 *        engine is for random number generation, if needed.
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
 *  position at which the particle hit the boundary, and also the associated
 *  normal vector. Particles can be `absorbed` by setting their ids to negative
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
 *        void operator() (const PData& ptd, int i,
 *                         const amrex::RealVect& pos, const amrex::RealVect& normal,
 *                         amrex::RandomEngine const& engine);
 *
 *        where ptd is the particle tile, i the index of the particle operated on,
 *        pos and normal the location of the collision and the boundary normal vector.
 *        engine is for random number generation, if needed.
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
 *  position at which the particle hit the boundary, and also the associated
 *  normal vector. Particles can be `absorbed` by setting their ids to negative
 *  to flag them for removal. Likewise, the can be reflected back into the domain
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
 *        void operator() (const PData& ptd, int i,
 *                         const amrex::RealVect& pos, const amrex::RealVect& normal,
 *                         amrex::RandomEngine const& engine);
 *
 *        where ptd is the particle tile, i the index of the particle operated on,
 *        pos and normal the location of the collision and the boundary normal vector.
 *        engine is for random number generation, if needed.
 */
template <class PC, class F, std::enable_if_t<amrex::IsParticleContainer<PC>::value, int> foo = 0>
void
scrapeParticlesAtEB (PC& pc, ablastr::fields::MultiLevelScalarField const& distance_to_eb,
                 int lev_min, int lev_max, F&& f)
{
    BL_PROFILE("scrapeParticlesAtEB");

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
            auto& tile = pti.GetParticleTile();
            auto ptd = tile.getParticleTileData();
            const auto np = tile.numParticles();
            auto phi = (*distance_to_eb[lev])[pti].array();  // signed distance function
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
                amrex::Real const phi_value = ablastr::particles::interp_field_nodal(i, j, k, W, phi);

                if (phi_value < 0.0)
                {
                    int ic, jc, kc; // Cell-centered indices
                    amrex::Real Wc[AMREX_SPACEDIM][2]; // Cell-centered weights
                    ablastr::particles::compute_weights<amrex::IndexType::CELL>(
                        xp, yp, zp, plo, dxi, ic, jc, kc, Wc);
                    amrex::RealVect normal = DistanceToEB::interp_normal(i, j, k, W, ic, jc, kc, Wc, phi, dxi);
                    DistanceToEB::normalize(normal);
                    amrex::RealVect pos;
#if (defined WARPX_DIM_3D)
                    pos[0] = xp;
                    pos[1] = yp;
                    pos[2] = zp;
#elif (defined WARPX_DIM_XZ)
                    pos[0] = xp;
                    pos[1] = zp;
#elif (defined WARPX_DIM_RZ)
                    pos[0] = std::sqrt(xp*xp + yp*yp);
                    pos[1] = zp;
#elif (defined WARPX_DIM_1D_Z)
                    pos[0] = zp;
#endif
                    f(ptd, ip, pos, normal, phi_value, engine);

                }
            });
        }
    }
}

/**
 * \brief Reflect particles off embedded boundary walls using bisection.
 *
 *  For each particle that has crossed into the EB (phi < 0), this function:
 *  1. Bisects along the particle trajectory to find the contact point on
 *     the EB surface (where phi = 0).
 *  2. Computes the surface normal at the exact contact point.
 *  3. Performs specular velocity reflection: v_new = v - 2*(v . n)*n.
 *  4. Advances the particle from the contact point with the reflected
 *     velocity for the remaining fraction of the timestep.
 *
 *  This handles curved EB surfaces correctly, unlike the simpler phi-based
 *  approximation (x_new = x - 2*phi*n) which is only exact for planar
 *  surfaces. The bisection approach follows the same algorithm used in
 *  ParticleBoundaryBuffer::FindEmbeddedBoundaryIntersection.
 *
 * \tparam PC a type of amrex ParticleContainer
 *
 * \param pc the particle container to reflect.
 * \param distance_to_eb MultiFabs storing the signed distance function.
 * \param lev the mesh refinement level to work on.
 * \param dt the timestep size at this level.
 * \param mass the particle mass for this species.
 */
template <class PC, std::enable_if_t<amrex::IsParticleContainer<PC>::value, int> foo = 0>
void
reflectParticlesAtEB (PC& pc, ablastr::fields::MultiLevelScalarField const& distance_to_eb,
                      int lev, amrex::Real dt, amrex::ParticleReal mass)
{
    BL_PROFILE("reflectParticlesAtEB");

    const auto plo = pc.Geom(lev).ProbLoArray();
    const auto dxi = pc.Geom(lev).InvCellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for(WarpXParIter pti(pc, lev); pti.isValid(); ++pti)
    {
        const auto getPosition = GetParticlePosition<PIdx>(pti);
        auto setPosition = SetParticlePosition<PIdx>(pti);
        auto& tile = pti.GetParticleTile();
        auto ptd = tile.getParticleTileData();
        const auto np = tile.numParticles();
        auto phi = (*distance_to_eb[lev])[pti].array();

        amrex::ParallelForRNG( np,
        [=] AMREX_GPU_DEVICE (const int ip, amrex::RandomEngine const& engine) noexcept
        {
            amrex::ignore_unused(engine);

            if (!amrex::ParticleIDWrapper{ptd.m_idcpu[ip]}.is_valid()) { return; }

            amrex::ParticleReal xp, yp, zp;
            getPosition(ip, xp, yp, zp);

            int i, j, k;
            amrex::Real W[AMREX_SPACEDIM][2];
            ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                xp, yp, zp, plo, dxi, i, j, k, W);
            amrex::Real const phi_value =
                ablastr::particles::interp_field_nodal(i, j, k, W, phi);

            if (phi_value < 0.0)
            {
                amrex::ParticleReal ux = ptd.m_rdata[PIdx::ux][ip];
                amrex::ParticleReal uy = ptd.m_rdata[PIdx::uy][ip];
                amrex::ParticleReal uz = ptd.m_rdata[PIdx::uz][ip];

                // Step 1: Bisect along trajectory to find the fraction of dt
                // to walk backward from end-of-step to reach the EB surface.
                amrex::Real const dt_fraction = amrex::bisect(0.0, 1.0,
                    [=] (amrex::Real dt_frac) {
                        int bi, bj, bk;
                        amrex::Real bW[AMREX_SPACEDIM][2];
                        amrex::ParticleReal x_temp = xp, y_temp = yp, z_temp = zp;
                        UpdatePosition(x_temp, y_temp, z_temp,
                                       ux, uy, uz, -dt_frac * dt, mass);
                        ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                            x_temp, y_temp, z_temp, plo, dxi, bi, bj, bk, bW);
                        return ablastr::particles::interp_field_nodal(
                            bi, bj, bk, bW, phi);
                    });

                // Step 2: Move to contact point on the EB surface
                amrex::ParticleReal x_c = xp, y_c = yp, z_c = zp;
                UpdatePosition(x_c, y_c, z_c, ux, uy, uz,
                               -dt_fraction * dt, mass);

                // Step 3: Compute normal at the contact point
                int ci, cj, ck;
                amrex::Real cW[AMREX_SPACEDIM][2];
                ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                    x_c, y_c, z_c, plo, dxi, ci, cj, ck, cW);
                int cci, ccj, cck;
                amrex::Real ccW[AMREX_SPACEDIM][2];
                ablastr::particles::compute_weights<amrex::IndexType::CELL>(
                    x_c, y_c, z_c, plo, dxi, cci, ccj, cck, ccW);
                amrex::RealVect normal = DistanceToEB::interp_normal(
                    ci, cj, ck, cW, cci, ccj, cck, ccW, phi, dxi);
                DistanceToEB::normalize(normal);

                // Step 4: Specular velocity reflection  v = v - 2(v.n)n
                // Normal is in reduced-dimension coordinates;
                // velocity is always 3D Cartesian.
#if defined(WARPX_DIM_3D)
                amrex::ParticleReal const v_dot_n =
                    ux * amrex::ParticleReal(normal[0]) +
                    uy * amrex::ParticleReal(normal[1]) +
                    uz * amrex::ParticleReal(normal[2]);
                ux -= amrex::ParticleReal(2.0) * v_dot_n
                    * amrex::ParticleReal(normal[0]);
                uy -= amrex::ParticleReal(2.0) * v_dot_n
                    * amrex::ParticleReal(normal[1]);
                uz -= amrex::ParticleReal(2.0) * v_dot_n
                    * amrex::ParticleReal(normal[2]);
#elif defined(WARPX_DIM_XZ)
                // normal is (n_x, n_z). uy is out-of-plane, unaffected.
                amrex::ParticleReal const v_dot_n =
                    ux * amrex::ParticleReal(normal[0]) +
                    uz * amrex::ParticleReal(normal[1]);
                ux -= amrex::ParticleReal(2.0) * v_dot_n
                    * amrex::ParticleReal(normal[0]);
                uz -= amrex::ParticleReal(2.0) * v_dot_n
                    * amrex::ParticleReal(normal[1]);
#elif defined(WARPX_DIM_RZ)
                // normal is (n_r, n_z). Convert to 3D Cartesian using
                // the particle theta at the contact point.
                amrex::ParticleReal const theta_c = std::atan2(y_c, x_c);
                amrex::ParticleReal const cos_tc = std::cos(theta_c);
                amrex::ParticleReal const sin_tc = std::sin(theta_c);
                amrex::ParticleReal const n3d_x =
                    amrex::ParticleReal(normal[0]) * cos_tc;
                amrex::ParticleReal const n3d_y =
                    amrex::ParticleReal(normal[0]) * sin_tc;
                amrex::ParticleReal const n3d_z =
                    amrex::ParticleReal(normal[1]);
                amrex::ParticleReal const v_dot_n =
                    ux * n3d_x + uy * n3d_y + uz * n3d_z;
                ux -= amrex::ParticleReal(2.0) * v_dot_n * n3d_x;
                uy -= amrex::ParticleReal(2.0) * v_dot_n * n3d_y;
                uz -= amrex::ParticleReal(2.0) * v_dot_n * n3d_z;
#elif defined(WARPX_DIM_1D_Z)
                amrex::ParticleReal const v_dot_n =
                    uz * amrex::ParticleReal(normal[0]);
                uz -= amrex::ParticleReal(2.0) * v_dot_n
                    * amrex::ParticleReal(normal[0]);
#endif

                // Step 5: Store reflected velocity
                ptd.m_rdata[PIdx::ux][ip] = ux;
                ptd.m_rdata[PIdx::uy][ip] = uy;
                ptd.m_rdata[PIdx::uz][ip] = uz;

                // Step 6: Advance from contact point with reflected velocity
                // for the remaining fraction of the timestep
                amrex::ParticleReal x_new = x_c, y_new = y_c, z_new = z_c;
                UpdatePosition(x_new, y_new, z_new, ux, uy, uz,
                               dt_fraction * dt, mass);

                // Step 7: Write final position back to particle
                setPosition(ip, x_new, y_new, z_new);
            }
        });
    }
}

#endif //WARPX_PARTICLESCRAPER_H_
