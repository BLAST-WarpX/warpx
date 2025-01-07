/* Copyright 2024 Revathi Jambunathan
 *
 * This file is part of WarpX
 *
 * License: BSD-3-Clause-LBNL
 */

#ifdef WARPX_SURFACE_PHYSICS

#include "SurfacePhysicsBase.H"
#include "SurfaceParticleMapping.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/UpdatePosition.H"
#include "WarpX.H"

#include <ablastr/particles/NodalFieldGather.H>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_ParIter.H>
#include <AMReX_Particles.H>
#include <AMReX_StructOfArrays.H>
#include <AMReX_REAL.H>
#include <AMReX_Print.H>


FindEmbeddedBoundaryMapAndCounter::FindEmbeddedBoundaryMapAndCounter (const amrex::Real dt,
       amrex::Array4<const amrex::Real> phi_arr,
       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxi,
       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo,
       amrex::Array4<const int> ivec_map_arr,
       int* incident_np)
    : m_dt(dt), m_phi_arr(phi_arr), m_dxi(dxi), m_plo(plo), m_ivec_map_arr(ivec_map_arr), m_incident_np(incident_np)
{
}

void
SurfacePhysicsBase::countParticlesFromEmbeddedBoundaries (
    MultiParticleContainer& mypc, ablastr::fields::MultiLevelScalarField const& distance_to_eb)
{
    amrex::Print() << " in map particles to EB surface \n";
    using PIter = amrex::ParConstIterSoA<PIdx::nattribs, 0>;
    const auto& warpx = WarpX::GetInstance();
    const amrex::Geometry& geom = warpx.Geom(0);
    auto plo = geom.ProbLoArray();

    for (int i = 0; i < num_influx_species; ++i)
    {
        amrex::Print() << " influx sp name " << mypc.GetSpeciesNames()[0] << " " << mypc.GetSpeciesNames()[1] << "\n";
        const auto& pc = mypc.GetParticleContainer(i);
        int* const AMREX_RESTRICT dptr_incident_np = num_in_particles[i].dataPtr();
        for (int lev = 0; lev < pc.numLevels(); ++lev)
        {
            const auto& plevel = pc.GetParticles(lev);
            auto dxi = warpx.Geom(lev).InvCellSizeArray();
//#ifdef AMREX_USE_OMP
//#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
//#endif
            for (PIter pti(pc, lev); pti.isValid(); ++pti)
            {
                auto phi_arr = (*distance_to_eb[lev])[pti].array();
                auto ivec_map_arr = (*ivect_map)[pti].array();
                auto index = std::make_pair(pti.index(), pti.LocalTileIndex());
                if (plevel.find(index) == plevel.end()) {continue;}

                const auto dt = warpx.getdt(pti.GetLevel());

                const auto MapParticleToEB = FindEmbeddedBoundaryMapAndCounter(dt,
                    phi_arr, dxi, plo, ivec_map_arr, dptr_incident_np);

                const auto& ptile = plevel.at(index);
                auto ptile_data = ptile.getConstParticleTileData();
                long const np = ptile.numParticles();
                const auto getPosition = GetParticlePosition<PIdx>(pti);
                
                amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int ip)
                {
                    amrex::ParticleReal xp, yp, zp;
                    getPosition(ip, xp, yp, zp);
                    amrex::Real const phi_value = ablastr::particles::doGatherScalarFieldNodal(
                        xp, yp, zp, phi_arr, dxi, plo
                    );
                    if (phi_value < 0) {
                        MapParticleToEB(ptile_data, ip);
                    }
                });
            }
        }
//        amrex::Print() << " num_inparticles size " << num_in_particles[i].size() << "\n"; 
        for (int j = 0; j < surf_ijk.size(); ++j) {
            amrex::Print() << " num in for sp " << i << " is : " << num_in_particles[i][j] << " at element " << j << "\n";
        }
    }
}


#endif
