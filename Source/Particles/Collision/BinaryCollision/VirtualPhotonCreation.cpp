/* Copyright 2025 Arianna Formenti
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */


#include "VirtualPhotonCreation.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/ParticleCreation/SmartCopy.H"

#include "Utils/TextMsg.H"
#include "Utils/Parser/ParserUtils.H"

#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX.H>
#include <AMReX_INT.H>
#include <AMReX_REAL.H>
#include <AMReX_Particle.H>

#include <cmath>

namespace collision::binarycollision::virtualphotons{

using namespace amrex::literals;
using SoaData_type = typename WarpXParticleContainer::ParticleTileType::ParticleTileDataType;

void GenerateVirtualPhotons (MultiParticleContainer* mypc){

#ifdef WARPX_QED

    ABLASTR_PROFILE("collision::binarycollision::virtualphotons::GenerateVirtualPhotons()");

    // Loop through the species
    for (int i_s = 0; i_s < mypc->nSpecies(); ++i_s) {

        auto& primary = mypc->GetParticleContainer(i_s);

        if(!primary.has_virtual_photons()){
            continue;
        }

        // Get the virtual photon species corresponding to this primary species
        const int vphotons_index = primary.getVirtualPhotonSpeciesIndex();
        auto& vphotons = mypc->GetParticleContainer(vphotons_index);
        const amrex::ParmParse pp_species_name(mypc->GetSpeciesNames()[vphotons_index]);

        // Minimum allowed energy of the virtual photons
        amrex::Real vphoton_min_energy = 0.0_rt;
        utils::parser::getWithParser(pp_species_name, "qed_virtual_photons_min_energy", vphoton_min_energy);

        // Sampling factor (a.k.a. multiplier):
        // the number of virtual photons generated is multiplied by this factor,
        // the weight of each virtual photon is divided by this factor
        amrex::Real sampling_factor = 0.0_rt;
        utils::parser::getWithParser(pp_species_name, "qed_virtual_photons_multiplier", sampling_factor);

        amrex::Real const alpha_over_pi = PhysConst::alpha / MathConst::pi;
        amrex::Real const inv_c2 = 1._rt / (PhysConst::c * PhysConst::c);
        amrex::Real const mass = primary.getMass();

        int const nlevs = std::max(0, primary.finestLevel()+1);
        for (int lev = 0; lev < nlevs; ++lev) {

#ifdef AMREX_USE_OMP
            #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi = primary.MakeMFIter(lev); mfi.isValid(); ++mfi)
            {
                // Notation: _vp means virtual photon
                // Primary particles (leptons) in the current tile
                ParticleUtils::ParticleTileType& ptile = primary.ParticlesAt(lev, mfi);
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
                    const amrex::ParticleReal ux = soa.m_rdata[PIdx::ux][i]; // u=v*gamma=p/m_e
                    const amrex::ParticleReal uy = soa.m_rdata[PIdx::uy][i];
                    const amrex::ParticleReal uz = soa.m_rdata[PIdx::uz][i];

                    // Formula 99.16 in Berestetskii et al., Quantum Electrodynamics
                    // integrated over the photon energies from vphoton_min_energy to the energy of the primary particle
                    // A similar formula is 15.58 in Jackson's, Classical Electrodynamics
                    // but neglect longitudinal field, assume relativistic velocities, and integrate in energy
                    const amrex::ParticleReal gamma = std::sqrt( 1.0_rt +  (ux*ux + uy*uy + uz*uz) * inv_c2 );
                    // Minimum fractional (w.r.t. the primary) photon energy
                    const amrex::ParticleReal y_min = vphoton_min_energy * inv_c2 / (gamma * mass);
                    const amrex::ParticleReal lny = std::log( y_min );
                    // Number of virtual photons per primary particle
                    const amrex::Real r_photons = alpha_over_pi * lny * lny * sampling_factor;

                    // `n_photons` must be an integer, but must average to `r_photons` over many realizations
                    // This is achieved by adding a random number between 0 and 1, and taking the integer part.
                    const auto n_photons = static_cast<amrex::Long>( r_photons + amrex::Random(engine) );

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
                // Note that this operation will overwrite any virtual photons that were previously generated by mypc
                // namely the ones that were created in the previous time step.
                ParticleUtils::ParticleTileType& ptile_vp = vphotons.ParticlesAt(lev, mfi);
                ptile_vp.resize(total_num_vp);

                // Get the starting particle ID on CPU and reserve IDs for all virtual photons
                // This must be done on CPU because NextID() is not thread-safe and cannot be called from GPU
                amrex::Long pid;
#ifdef AMREX_USE_OMP
                #pragma omp critical (virtual_photon_nextid)
#endif
                {
                    pid = ParticleUtils::ParticleTileType::ParticleType::NextID();
                    ParticleUtils::ParticleTileType::ParticleType::NextID(pid + total_num_vp);
                }

                const int cpuid = amrex::ParallelDescriptor::MyProc();

                // SoA that will contain the virtual photons data
                auto &soa_vp = ptile_vp.GetStructOfArrays();

                // Array with the PIDs of the virtual photons
                uint64_t * AMREX_RESTRICT pid_vp = soa_vp.GetIdCPUData().data();

                // Pointers to the arrays that will contain the particle attributes of the virtual photons
                amrex::GpuArray<amrex::ParticleReal*,PIdx::nattribs> pa_vp;
                for (int ia = 0; ia < PIdx::nattribs; ++ia) {
                    pa_vp[ia] = soa_vp.GetRealData(ia).data();
                }

                // Capture the starting PID for use in the GPU kernel
                const amrex::Long pid_start = pid;

                // Second pass: populate the virtual photon species
                amrex::ParallelForRNG (num,
                [=] AMREX_GPU_DEVICE (amrex::Long i,  amrex::RandomEngine const& engine) noexcept
                {
                    // Primary particle
                    const amrex::ParticleReal ux_primary = soa.m_rdata[PIdx::ux][i];
                    const amrex::ParticleReal uy_primary = soa.m_rdata[PIdx::uy][i];
                    const amrex::ParticleReal uz_primary = soa.m_rdata[PIdx::uz][i];
                    const amrex::ParticleReal u_primary = std::sqrt(ux_primary*ux_primary + uy_primary*uy_primary + uz_primary*uz_primary);
                    const amrex::ParticleReal nx = ux_primary / u_primary; // normalized ux
                    const amrex::ParticleReal ny = uy_primary / u_primary; // normalized uy
                    const amrex::ParticleReal nz = uz_primary / u_primary; // normalized uz
                    const amrex::ParticleReal gamma_primary = std::sqrt( 1.0_rt + (ux_primary*ux_primary + uy_primary*uy_primary + uz_primary*uz_primary)*inv_c2 );

#if defined (WARPX_DIM_3D)
                    const amrex::ParticleReal x  = soa.m_rdata[PIdx::x][i];
                    const amrex::ParticleReal y  = soa.m_rdata[PIdx::y][i];
                    const amrex::ParticleReal z  = soa.m_rdata[PIdx::z][i];
#elif defined (WARPX_DIM_XZ)
                    const amrex::ParticleReal x  = soa.m_rdata[PIdx::x][i];
                    const amrex::ParticleReal z  = soa.m_rdata[PIdx::z][i];
#elif defined (WARPX_DIM_RZ)
                    const amrex::ParticleReal x  = soa.m_rdata[PIdx::x][i];
                    const amrex::ParticleReal z  = soa.m_rdata[PIdx::z][i];
                    const amrex::ParticleReal theta  = soa.m_rdata[PIdx::theta][i];
#elif defined (WARPX_DIM_1D_Z)
                    const amrex::ParticleReal z  = soa.m_rdata[PIdx::z][i];
#elif defined (WARPX_DIM_RCYLINDER)
                    const amrex::ParticleReal x  = soa.m_rdata[PIdx::x][i];
                    const amrex::ParticleReal theta  = soa.m_rdata[PIdx::theta][i];
#elif defined(WARPX_DIM_RSPHERE)
                    const amrex::ParticleReal x  = soa.m_rdata[PIdx::x][i];
                    const amrex::ParticleReal theta  = soa.m_rdata[PIdx::theta][i];
                    const amrex::ParticleReal phi  = soa.m_rdata[PIdx::phi][i];
#endif
                    const amrex::ParticleReal w  = soa.m_rdata[PIdx::w][i];

                    // TODO: add a runtime attribute to the virtual photon species
                    // that containes the pid of the parent particle = soa.m_idcpu[i]
                    // This will allow to update the parent lepton if needed

                    // Minimum fractional (wrt primary particle) photon energy
                    const amrex::Real y_min = vphoton_min_energy / (mass * gamma_primary * PhysConst::c * PhysConst::c);
                    const amrex::Real umin = 0._rt;
                    const amrex::Real umax = std::log(y_min) * std::log(y_min);

                    for (int j = 0; j < num_vp_data[i]; j++)
                    {
                        // Sample frac_energy from a probability distribution function
                        // that is proportional to log(frac_energy)/frac_energy
                        // (formula 99.16 in Berestetskii et al.)
                        // using the method of the inverse cumulative distributionfunction

                        // Draw a random number between umin and umax
                        const amrex::ParticleReal rnd = (umax - umin) * amrex::Random(engine) + umin ;
                        // Fractional energy of the photon, often denoted as y (or x)
                        const amrex::ParticleReal frac_energy = std::exp( - std::sqrt(rnd) );
                        // Energy of the virtual photon
                        const amrex::ParticleReal vphoton_energy = frac_energy * gamma_primary * PhysConst::c;

                        // Photon index for the current primary
                        const amrex::Long ip = offset_vp_data[i] + j;
                        pa_vp[PIdx::ux][ip] = vphoton_energy * nx; // will be multiplied by m_e before dumping the outputs
                        pa_vp[PIdx::uy][ip] = vphoton_energy * ny; // will be multiplied by m_e before dumping the outputs
                        pa_vp[PIdx::uz][ip] = vphoton_energy * nz; // will be multiplied by m_e before dumping the outputs

                        // TODO: add beam size effect - displace the photon position
#if defined (WARPX_DIM_3D)
                        pa_vp[PIdx::x][ip] = x;
                        pa_vp[PIdx::y][ip] = y;
                        pa_vp[PIdx::z][ip] = z;
#elif defined (WARPX_DIM_XZ)
                        pa_vp[PIdx::x][ip] = x;
                        pa_vp[PIdx::z][ip] = z;
#elif defined (WARPX_DIM_RZ)
                        pa_vp[PIdx::x][ip] = x;
                        pa_vp[PIdx::z][ip] = z;
                        pa_vp[PIdx::theta][ip] = theta;
#elif defined (WARPX_DIM_1D_Z)
                        pa_vp[PIdx::z][ip] = z;
#elif defined (WARPX_DIM_RCYLINDER)
                        pa_vp[PIdx::x][ip] = x;
                        pa_vp[PIdx::theta][ip] = theta;
#elif defined(WARPX_DIM_RSPHERE)
                        pa_vp[PIdx::x][ip] = x;
                        pa_vp[PIdx::theta][ip] = theta;
                        pa_vp[PIdx::phi][ip] = phi;
#endif
                        pa_vp[PIdx::w][ip] = w / sampling_factor;
                        pid_vp[ip] = amrex::SetParticleIDandCPU(pid_start + ip, cpuid);
                    }
                });
            } // mfi
        } // lev
    } // species

#else

WARPX_ABORT_WITH_MESSAGE("Compiling WarpX with QED support is required to call GenerateVirtualPhotons");
amrex::ignore_unused(mypc);

#endif //WARPX_QED

} // function
} // close namespace
