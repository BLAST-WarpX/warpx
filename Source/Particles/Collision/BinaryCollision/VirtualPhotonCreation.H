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

#include <cmath>


namespace collision::binarycollision::virtualphotons{

using namespace amrex::literals;
using SoaData_type = typename WarpXParticleContainer::ParticleTileType::ParticleTileDataType;

void GenerateVirtualPhotons (MultiParticleContainer* mypc){

    WARPX_PROFILE("collision::binarycollision::virtualphotons::GenerateVirtualPhotons()");

    // Enable tiling
    amrex::MFItInfo info;
    if (amrex::Gpu::notInLaunchRegion()) { info.EnableTiling(WarpXParticleContainer::tile_size); }

    // TO DO: particle species should be automatically selected
    auto& primary  = mypc->GetParticleContainerFromName("beam");
    auto& vphotons = mypc->GetParticleContainerFromName("virtual_photons");
    // TO DO: minimum photon energy should be an input parameter
    amrex::Real const hw_min = PhysConst::m_e * PhysConst::c * PhysConst::c * 0.0001_rt; // 0.1 * m_e * c^2
    //amrex::Real const c_arb = 0.2_rt; // arbitrary constant which should be >~ 0.1
    //amrex::Real const eta_max = 0.5_rt * ( std::pow( std::log(hw_min), 2._rt) + std::sqrt( std::pow( std::log(hw_min), 4._rt) + 4._rt * c_arb * std::pow( std::log(hw_min), 2._rt) ) );
    amrex::Real const alpha_pi = PhysConst::alpha / MathConst::pi;
    amrex::Real const sampling_factor = 1000000._rt;
    amrex::Real const inv_c2 = 1._rt / (PhysConst::c * PhysConst::c);

    amrex::Real const mass = primary.getMass();

    amrex::Real const gamma_em = 0.57721566490153286060_rt; // Euler-Mascheroni constant
    amrex::Real const coeff = + std::log(4._rt) - 2._rt*gamma_em - 1._rt;

    amrex::Real umin = 0.25_rt * coeff * coeff;




    int const nlevs = std::max(0, primary.finestLevel()+1);
    for (int lev = 0; lev < nlevs; ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi = primary.MakeMFIter(lev, info); mfi.isValid(); ++mfi)
        {
            // Notation: _vp means virtual photon

            // primary particles (leptons) in the current tile
            ParticleTileType& ptile = primary.ParticlesAt(lev, mfi);
            const auto soa = ptile.getParticleTileData();

            // number of primary particles in the current tile
            amrex::Long const num = ptile.numParticles();

            // vector that will contain the number of virtual photons for each primary particle
            amrex::Gpu::DeviceVector<amrex::Long> num_vp(num, 0);
            auto* num_vp_data = num_vp.dataPtr();

            // First pass: compute the number of virtual photons for each primary particle
            // and fill the corresponding vector
            amrex::ParallelForRNG(num,
             [=] AMREX_GPU_DEVICE (amrex::Long i, amrex::RandomEngine const& engine) noexcept
            {
                amrex::ParticleReal ux = soa.m_rdata[PIdx::ux][i]; // u=v*gamma=p/m
                amrex::ParticleReal uy = soa.m_rdata[PIdx::uy][i];
                amrex::ParticleReal uz = soa.m_rdata[PIdx::uz][i];

                // same as Cain
                // similar to Jackson's formula 15.58, but neglect longitudinal field, assume relativistic velocities, and integrate in energy
                amrex::ParticleReal gamma = std::sqrt( 1.0_rt +  (ux*ux + uy*uy + uz*uz) * inv_c2 );
                amrex::ParticleReal y_min = hw_min * inv_c2 / (gamma * mass); // min fractional photon energy
                amrex::ParticleReal lny = std::log( y_min );
                amrex::Real r_photons = alpha_pi * (lny * lny - coeff * lny) * sampling_factor; // number of virtual photons per primary particle

                amrex::Real i_photons;
                amrex::Real f_photons = std::modf(r_photons, &i_photons);

                amrex::Long n_photons = (amrex::Long)i_photons;

                amrex::Print(0) << " VPHOTONSSSS = " << r_photons << " ymin " << y_min << "gamma " << gamma << std::endl;

                if(amrex::Random(engine) < f_photons){
                     n_photons += amrex::Long(1);
                }

                num_vp_data[i] = n_photons;
            });

            // compute the offsets vector as the cumulative sum of the elements of num_vp,
            // excluding the current element
            amrex::Gpu::DeviceVector<amrex::Long> offsets_vp(num);
            //amrex::Gpu::exclusive_scan(num_vp.begin(), num_vp.end(), offsets_vp.begin());

            // total number of virtual photons to be generated in the current tile
            const amrex::Long total_num_vp = amrex::Scan::ExclusiveSum(num_vp.size(), num_vp.data(), offsets_vp.data());
            auto *const offset_vp_data = offsets_vp.dataPtr();

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
            amrex::ParallelForRNG (num,
             [=] AMREX_GPU_DEVICE (amrex::Long i,  amrex::RandomEngine const& engine) noexcept
            {
                // primary energy
                amrex::ParticleReal ux_primary = soa.m_rdata[PIdx::ux][i];
                amrex::ParticleReal uy_primary = soa.m_rdata[PIdx::uy][i];
                amrex::ParticleReal uz_primary = soa.m_rdata[PIdx::uz][i];
                amrex::ParticleReal gamma_primary = std::sqrt( 1.0_rt + (ux_primary*ux_primary + uy_primary*uy_primary + uz_primary*uz_primary)*inv_c2 );

                amrex::Real y_min = hw_min / (mass * gamma_primary * PhysConst::c * PhysConst::c);
                // primary position, weight and id
                amrex::ParticleReal x  = soa.m_rdata[PIdx::x][i];
                amrex::ParticleReal y  = soa.m_rdata[PIdx::y][i];
                amrex::ParticleReal z  = soa.m_rdata[PIdx::z][i];
                amrex::ParticleReal w  = soa.m_rdata[PIdx::w][i];
                uint64_t pid = soa.m_idcpu[i];

                amrex::Real umax = std::pow( 0.5_rt * coeff + std::log(y_min), 2._rt);

                // not sure about this:
                //WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                //    std::exp(0.5_rt * coeff) > y_min,
                //    "should never happen, but check"
                //);


                for (int j = 0; j < num_vp_data[i]; j++)
                {



                    amrex::ParticleReal rnd = (umax - umin) * amrex::Random(engine) + umin ;
                    amrex::ParticleReal frac_energy = std::exp( 0.5_rt * coeff - std::sqrt(rnd)); // fractional energy of the photon, often denoted as y (or x)

                    // photon index for the current primary
                    amrex::Long ip = offset_vp_data[i] + j;
                    pa_vp[PIdx::ux][ip] = 0._rt;
                    pa_vp[PIdx::uy][ip] = 0._rt;
                    pa_vp[PIdx::uz][ip] = frac_energy * gamma_primary * PhysConst::c; // this will be multiplied by m_e before damping the outputs
                    pa_vp[PIdx::x][ip] = x;
                    pa_vp[PIdx::y][ip] = y;
                    pa_vp[PIdx::z][ip] = z;
                    pa_vp[PIdx::w][ip] = w / sampling_factor;
                    pid_vp[ip] = pid;
                    //amrex::Print(0) << " INFO ==== >  " << frac_energy << " ----- " << frac_energy * gamma_primary << std::endl;
                }
            });
        } // mfi
    } // lev

    amrex::Gpu::synchronize();

} // function
} // close namespace
#endif // WARPX_VIRTUAL_PHOTONS_H_
