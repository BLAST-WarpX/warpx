#ifndef WARPX_SURFACE_PARTICLE_MAPPING_H
#define WARPX_SURFACE_PARTICLE_MAPPING_H

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/UpdatePosition.H"


#include <ablastr/particles/NodalFieldGather.H>

#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_Reduce.H>
#include <AMReX_Tuple.H>
#include <AMReX.H>
#include <AMReX_Algorithm.H>
#include <AMReX_RealVect.H>

struct FilterEBScrapedParticle
{

    FilterEBScrapedParticle (){}

};

struct FindEmbeddedBoundaryMapAndCounter
{
    FindEmbeddedBoundaryMapAndCounter (const amrex::Real dt,
       amrex::Array4<const amrex::Real> phi_arr,
       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxi,
       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo,
       amrex::Array4<const int> ivec_map_arr,
       int* incident_np);

    template <typename SrcData>
    AMREX_GPU_HOST_DEVICE
    void operator() (const SrcData& src, int src_i) const
    {
        const auto& p = src.getSuperParticle(src_i);
        amrex::ParticleReal xp, yp, zp;
        get_particle_position(p, xp, yp, zp);

        amrex::ParticleReal ux = src.m_rdata[PIdx::ux][src_i];
        amrex::ParticleReal uy = src.m_rdata[PIdx::uy][src_i];
        amrex::ParticleReal uz = src.m_rdata[PIdx::uz][src_i];

        // Temporary variables to avoid implicit capture
        amrex::Real const dt = m_dt;
        amrex::Array4<amrex::Real const> const phiarr = m_phi_arr;
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dxi = m_dxi;
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const plo = m_plo;

        // Bisection algorithm to find the point where phi(x,y,z)=0 (i.e. on the embedded boundary)
        amrex::Real const dt_fraction = amrex::bisect( 0.0, 1.0,
            [=] (amrex::Real dt_frac) {
                int i, j, k;
                amrex::Real W[AMREX_SPACEDIM][2];
                amrex::ParticleReal x_temp=xp, y_temp=yp, z_temp=zp;
                UpdatePosition(x_temp, y_temp, z_temp, ux, uy, uz, -dt_frac*dt);
                ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                    x_temp, y_temp, z_temp, plo, dxi, i, j, k, W);
                amrex::Real const phi_value = ablastr::particles::interp_field_nodal(i, j, k, W, phiarr);
                return phi_value;
            } );

         // intersection point
         amrex::ParticleReal xc=xp, yc=yp, zc=zp;
         // tracing the position back by dt_fraction where phi_value was 0 from bisect
         UpdatePosition(xc, yc, zc, ux, uy, uz, -dt_fraction*m_dt);

         // Find cell-center index that contains the intersection point
         int ic = 0, jc = 0, kc = 0;
#if (AMREX_SPACEDIM >= 2)
         ic = static_cast<int>(floor( (xc - plo[0])*dxi[0] ) );
#endif
#if defined(WARPX_DIM_XZ)
         jc = static_cast<int>(floor( (zc - plo[AMREX_SPACEDIM-1])*dxi[AMREX_SPACEDIM-1] ) );
         kc = 0;
#endif
#if defined(WARPX_DIM_3D)
         jc = static_cast<int>(floor( (yc-plo[1])*dxi[1] ) );
         kc = static_cast<int>(floor( (zc - plo[AMREX_SPACEDIM-1])*dxi[AMREX_SPACEDIM-1] ) );
#endif
         int ivec = m_ivec_map_arr(ic,jc,kc);
         amrex::HostDevice::Atomic::Add(&m_incident_np[ivec], 1);
         return;
    }

    amrex::Real m_dt;
    amrex::Array4<const amrex::Real> m_phi_arr;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_dxi;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_plo;
    amrex::Array4<const int> m_ivec_map_arr;
    int* m_incident_np;
};

#endif
