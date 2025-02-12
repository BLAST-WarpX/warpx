/* Copyright 2021 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_VIRTUAL_PHOTONS_H_
#define WARPX_VIRTUAL_PHOTONS_H_

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/ParticleCreation/SmartCopy.H"

#include "WarpX.H"
#include "Utils/TextMsg.H"


#include <AMReX_INT.H>
#include <AMReX_REAL.H>
#include <AMReX_Particle.H>


namespace collision::binarycollision::virtualphotons{

using namespace amrex::literals;
using SoaData_type = typename WarpXParticleContainer::ParticleTileType::ParticleTileDataType;

void GenerateVirtualPhotons (MultiParticleContainer* mypc){

    WARPX_PROFILE("collision::binarycollision::virtualphotons::GenerateVirtualPhotons()");

    // Enable tiling
    amrex::MFItInfo info;
    if (amrex::Gpu::notInLaunchRegion()) { info.EnableTiling(WarpXParticleContainer::tile_size); }

    // TO DO: particle species should be automatically selected
    auto& leptons  = mypc->GetParticleContainerFromName("beam");
    auto& vphotons = mypc->GetParticleContainerFromName("virtual_photons");

    int const nlevs = std::max(0, leptons.finestLevel()+1);
    for (int lev = 0; lev < nlevs; ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi = leptons.MakeMFIter(lev, info); mfi.isValid(); ++mfi)
        {
            // Notation: _vp means virtual photon

            // primary particles (leptons) in the current tile
            ParticleTileType& ptile = leptons.ParticlesAt(lev, mfi);
            const auto soa = ptile.getParticleTileData();

            // number of primary particles in the current tile
            amrex::Long const num = ptile.numParticles();

            // vector that will contain the number of virtual photons for each primary particle
            amrex::Gpu::DeviceVector<amrex::Long> num_vp(num, 0);
            auto* num_vp_data = num_vp.dataPtr();

            // First pass: compute the number of virtual photons for each primary particle
            // and fill the corresponding vector
            amrex::ParallelFor(num,
             [=] AMREX_GPU_DEVICE (amrex::Long i) noexcept
            {
                amrex::ParticleReal ux = soa.m_rdata[PIdx::ux][i];
                amrex::ParticleReal uy = soa.m_rdata[PIdx::uy][i];
                amrex::ParticleReal uz = soa.m_rdata[PIdx::uz][i];

                // same as Xsuite and guinea-pig
                amrex::ParticleReal gamma = std::sqrt( 1.0_rt + ux*ux + uy*uy + uz*uz);
                amrex::ParticleReal mandelstam_s_4 = gamma * gamma; // Mandelstam variable s / 4

                amrex::Real compt_x_min = 1.e-2_rt; // min fractional photon energy
                amrex::ParticleReal xmin = compt_x_min / mandelstam_s_4;

                amrex::Real lnxmin = -std::log(xmin);
                amrex::Real lns4 = std::log(mandelstam_s_4);

                amrex::Real r_photons = PhysConst::alpha / MathConst::pi * lnxmin * (lnxmin + lns4);

                amrex::Long n_photons = (amrex::Long)r_photons;

                num_vp_data[i] = n_photons;
            });

            // compute the offsets vector as the cumulative sum of the elements of num_vp,
            // excluding the current element
            amrex::Gpu::DeviceVector<amrex::Long> offsets_vp(num);
            amrex::Gpu::exclusive_scan(num_vp.begin(), num_vp.end(), offsets_vp.begin());
            auto *const offset_vp_data = offsets_vp.dataPtr();

            // total number of virtual photons to be generated in the current tile
            const amrex::Long total_num_vp = amrex::Scan::ExclusiveSum(num_vp.size(), num_vp.data(), offsets_vp.data());

            // now we can allocate and build the virtual photon species in the current tile
            ParticleTileType& ptile_vp = vphotons.ParticlesAt(lev, mfi);
            ptile_vp.resize(total_num_vp);

            // soa that will contain the virtual photons data
            auto &soa_vp = ptile_vp.GetStructOfArrays();

            // array with the pids of the virtual photons
            uint64_t * AMREX_RESTRICT pid_vp = soa_vp.GetIdCPUData().data();

            // pointers to the arrays that will contain the particle attributes of the virtual photons
            amrex::GpuArray<amrex::ParticleReal*,PIdx::nattribs> pa_vp;
            for (int ia = 0; ia < PIdx::nattribs; ++ia) {
                pa_vp[ia] = soa_vp.GetRealData(ia).data();
            }

            // Second pass: populate the virtual photon species
            amrex::ParallelFor(num,
             [=] AMREX_GPU_DEVICE (amrex::Long i) noexcept
            {
                amrex::ParticleReal ux = soa.m_rdata[PIdx::ux][i];
                amrex::ParticleReal uy = soa.m_rdata[PIdx::uy][i];
                amrex::ParticleReal uz = soa.m_rdata[PIdx::uz][i];
                amrex::ParticleReal x  = soa.m_rdata[PIdx::x][i];
                amrex::ParticleReal y  = soa.m_rdata[PIdx::y][i];
                amrex::ParticleReal z  = soa.m_rdata[PIdx::z][i];
                amrex::ParticleReal w  = soa.m_rdata[PIdx::w][i];
                uint64_t pid = soa.m_idcpu[i];

                for (int j = 0; j < num_vp_data[i]; j++)
                {
                    // photon index for the current primary
                    amrex::Long ip = offset_vp_data[i] + j;
                    pa_vp[PIdx::ux][ip] = ux;
                    pa_vp[PIdx::uy][ip] = uy;
                    pa_vp[PIdx::uz][ip] = uz;
                    pa_vp[PIdx::x][ip] = x;
                    pa_vp[PIdx::y][ip] = y;
                    pa_vp[PIdx::z][ip] = z;
                    pa_vp[PIdx::w][ip] = w;
                    pid_vp[ip] = pid;
                }
            });
        } // mfi
    } // lev

    amrex::Gpu::synchronize();

} // function
} // close namespace
#endif // WARPX_VIRTUAL_PHOTONS_H_
