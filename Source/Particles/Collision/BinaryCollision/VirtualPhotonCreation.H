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
#include "Utils/Parser/ParserUtils.H"

#include <AMReX_INT.H>
#include <AMReX_REAL.H>
#include <AMReX_Particle.H>

#include <cmath>


namespace collision::binarycollision::virtualphotons{

using namespace amrex::literals;
using SoaData_type = typename WarpXParticleContainer::ParticleTileType::ParticleTileDataType;


/** Generate virtual photons from primary particles (electrons or positrons)
 * using the equivalent photon approximation, also known as the Weizsacker-Williams method.
 * See for example Jackson, Classical Electrodynamics, 3rd edition, section 15.4
 * or Berestetskii, Lifshitz, Pitaevskii, Quantum Electrodynamics, 2nd edition, section 99.
 *
 * The number of photons per primary particle is computed according to Berestetskii's formula 99.16,
 * integrated over angles and over photon energies from a minimum energy (input parameter)
 * up to the energy of the primary particle.
 * The photons are assumed to be emitted in the direction of motion of the primary particle.
 * The virtual photons are generated in a separate species, which can then be used
 * as input for other processes, e.g., Breit-Wheeler pair production and Compton scattering.
 * The virtual photons are generated at the same position as the primary particle,
 * and inherit the same weight as the primary particle, divided by a sampling factor.
 * The virtual photons are given the same particle ID as the primary particle.
 * The virtual photon species is regenerated from scratch at each time step.
 * In the current implementation, the primary particles are not affected by
 * the possible interactions that the virtual photons may undergo.
 *
 * \param[in] mypc MultiParticleContainer calling this method
 */
void GenerateVirtualPhotons (MultiParticleContainer* mypc){

    WARPX_PROFILE("collision::binarycollision::virtualphotons::GenerateVirtualPhotons()");

    // Loop through the species
    for (int i_s = 0; i_s < mypc->nSpecies(); ++i_s) {

        auto& primary = mypc->GetParticleContainer(i_s);

        if(!primary.has_virtual_photons()){
            continue;
        }

        // Get the virtual photon species corresponding to this primary species
        int vphotons_index = primary.getVirtualPhotonSpeciesIndex();
        auto& vphotons = mypc->GetParticleContainer(vphotons_index);
        const amrex::ParmParse pp_species_name(mypc->GetSpeciesNames()[vphotons_index]);

        // Minimum allowed enrgy of the virtual photons
        amrex::Real vphoton_min_energy;
        utils::parser::getWithParser(pp_species_name, "virtual_photons_min_energy", vphoton_min_energy);

        // Sampling factor:
        // the number of virtual photons generated is multiplied by this factor,
        // the weight of each virtual photon is divided by this factor
        amrex::Real sampling_factor;
        utils::parser::getWithParser(pp_species_name, "sampling_factor", sampling_factor);

        // Enable tiling
        amrex::MFItInfo info;
        if (amrex::Gpu::notInLaunchRegion()) { info.EnableTiling(WarpXParticleContainer::tile_size); }

        amrex::Real const alpha_pi = PhysConst::alpha / MathConst::pi;
        amrex::Real const inv_c2 = 1._rt / (PhysConst::c * PhysConst::c);
        amrex::Real const mass = primary.getMass();
        amrex::Real umin = 0._rt;

        int const nlevs = std::max(0, primary.finestLevel()+1);
        for (int lev = 0; lev < nlevs; ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi = primary.MakeMFIter(lev, info); mfi.isValid(); ++mfi)
            {
                // Notation: _vp means virtual photon
                // Primary particles (leptons) in the current tile
                ParticleTileType& ptile = primary.ParticlesAt(lev, mfi);
                const auto soa = ptile.getParticleTileData();

                // Number of primary particles in the current tile
                amrex::Long const num = ptile.numParticles();

                // Vector that will contain the number of virtual photons for each primary particle
                amrex::Gpu::DeviceVector<amrex::Long> num_vp(num, 0);
                auto* num_vp_data = num_vp.dataPtr();

                // First pass: compute the number of virtual photons for each primary particle
                // and fill the corresponding vector
                amrex::ParallelForRNG(num,
                [=] AMREX_GPU_DEVICE (amrex::Long i, amrex::RandomEngine const& engine) noexcept
                {
                    amrex::ParticleReal ux = soa.m_rdata[PIdx::ux][i]; // u=v*gamma=p/m_e
                    amrex::ParticleReal uy = soa.m_rdata[PIdx::uy][i];
                    amrex::ParticleReal uz = soa.m_rdata[PIdx::uz][i];

                    // Similar to Jackson's formula 15.58, but neglect longitudinal field, assume relativistic velocities, and integrate in energy
                    amrex::ParticleReal gamma = std::sqrt( 1.0_rt +  (ux*ux + uy*uy + uz*uz) * inv_c2 );
                    // Minimum fractional (w.r.t. the primary) photon energy
                    amrex::ParticleReal y_min = vphoton_min_energy * inv_c2 / (gamma * mass);
                    amrex::ParticleReal lny = std::log( y_min );
                    // Number of virtual photons per primary particle
                    amrex::Real r_photons = alpha_pi * lny * lny * sampling_factor;
                    amrex::Real i_photons; // integer part
                    amrex::Real f_photons = std::modf(r_photons, &i_photons); // fractional part
                    amrex::Long n_photons = (amrex::Long)i_photons;

                    // Account for the fractional part of r_photons by
                    // randomly adding an extra virtual photon
                    if(amrex::Random(engine) < f_photons){ n_photons += amrex::Long(1);}

                    num_vp_data[i] = n_photons;
                });

                // Compute the offsets vector as the cumulative sum of the elements of num_vp excluding the current element,
                // i.e., offsets[i] = sum_{j=0}^{i-1} num_vp[j],
                // and return the total number of virtual photons to be generated in the current tile
                // (which is the last element of the offsets vector)
                amrex::Gpu::DeviceVector<amrex::Long> offsets_vp(num);
                const amrex::Long total_num_vp = amrex::Scan::ExclusiveSum(num_vp.size(), num_vp.data(), offsets_vp.data());
                auto *const offset_vp_data = offsets_vp.dataPtr();

                // Now we can allocate and build the virtual photon species in the current tile
                ParticleTileType& ptile_vp = vphotons.ParticlesAt(lev, mfi);
                ptile_vp.resize(total_num_vp);

                // SoA that will contain the virtual photons data
                auto &soa_vp = ptile_vp.GetStructOfArrays();

                // Array with the PIDs of the virtual photons
                uint64_t * AMREX_RESTRICT pid_vp = soa_vp.GetIdCPUData().data();

                // Pointers to the arrays that will contain the particle attributes of the virtual photons
                amrex::GpuArray<amrex::ParticleReal*,PIdx::nattribs> pa_vp;
                for (int ia = 0; ia < PIdx::nattribs; ++ia) {
                    pa_vp[ia] = soa_vp.GetRealData(ia).data();
                }

                // Second pass: populate the virtual photon species
                amrex::ParallelForRNG (num,
                [=] AMREX_GPU_DEVICE (amrex::Long i,  amrex::RandomEngine const& engine) noexcept
                {
                    // Primary particle
                    amrex::ParticleReal ux_primary = soa.m_rdata[PIdx::ux][i];
                    amrex::ParticleReal uy_primary = soa.m_rdata[PIdx::uy][i];
                    amrex::ParticleReal uz_primary = soa.m_rdata[PIdx::uz][i];
                    amrex::ParticleReal u_primary = std::sqrt(ux_primary*ux_primary + uy_primary*uy_primary + uz_primary*uz_primary);
                    amrex::ParticleReal nx = ux_primary / u_primary; // normalized ux
                    amrex::ParticleReal ny = uy_primary / u_primary; // normalized uy
                    amrex::ParticleReal nz = uz_primary / u_primary; // normalized uz
                    amrex::ParticleReal gamma_primary = std::sqrt( 1.0_rt + (ux_primary*ux_primary + uy_primary*uy_primary + uz_primary*uz_primary)*inv_c2 );
#if (defined WARPX_DIM_XZ) || (defined WARPX_DIM_RZ)
                    amrex::ParticleReal x  = soa.m_rdata[PIdx::x][i];
#elif (defined WARPX_DIM_3D)
                    amrex::ParticleReal x  = soa.m_rdata[PIdx::x][i];
                    amrex::ParticleReal y  = soa.m_rdata[PIdx::y][i];
#endif
                    amrex::ParticleReal z  = soa.m_rdata[PIdx::z][i];
                    amrex::ParticleReal w  = soa.m_rdata[PIdx::w][i];
                    uint64_t pid = soa.m_idcpu[i]; // TODO: use this to update the parent lepton if needed

                    // Minimum fractional (wrt primary particle) photon energy
                    amrex::Real y_min = vphoton_min_energy / (mass * gamma_primary * PhysConst::c * PhysConst::c);
                    amrex::Real umax = std::log(y_min) * std::log(y_min);

                    for (int j = 0; j < num_vp_data[i]; j++)
                    {
                        // Draw a random number between umin and umax
                        amrex::ParticleReal rnd = (umax - umin) * amrex::Random(engine) + umin ;
                        // Fractional energy of the photon, often denoted as y (or x)
                        amrex::ParticleReal frac_energy = std::exp( - std::sqrt(rnd) );
                        // Energy of the virtual photon
                        amrex::ParticleReal vphoton_energy = frac_energy * gamma_primary * PhysConst::c;

                        // Photon index for the current primary
                        amrex::Long ip = offset_vp_data[i] + j;
                        pa_vp[PIdx::ux][ip] = vphoton_energy * nx; // will be multiplied by m_e before dumping the outputs
                        pa_vp[PIdx::uy][ip] = vphoton_energy * ny; // will be multiplied by m_e before dumping the outputs
                        pa_vp[PIdx::uz][ip] = vphoton_energy * nz; // will be multiplied by m_e before dumping the outputs

                        // TODO: add beam size effect - displace the photon position
#if (defined WARPX_DIM_XZ) || (defined WARPX_DIM_RZ)
                        pa_vp[PIdx::x][ip] = x;
#elif (defined WARPX_DIM_3D)
                        pa_vp[PIdx::x][ip] = x;
                        pa_vp[PIdx::y][ip] = y;
#endif
                        pa_vp[PIdx::z][ip] = z;

                        pa_vp[PIdx::w][ip] = w / sampling_factor;
                        pid_vp[ip] = pid; // Same pid as the parent lepton - will be useful if we want to update the parent electron
                    }
                });
            } // mfi
        } // lev
    } // species
} // function
} // close namespace
#endif // WARPX_VIRTUAL_PHOTONS_H_
