/* Copyright 2025 Arianna Formenti, Peter Kicsiny
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

#include "Utils/ParticleUtils.H"
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
using namespace ParticleUtils;
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
#if defined (WARPX_DIM_3D)
        const bool do_beam_size_effect = primary.has_virtual_photons_beam_size_effect();
#endif
        // Minimum allowed energy of the virtual photons
        amrex::Real vphoton_min_energy = 0.0_rt;
        utils::parser::getWithParser(pp_species_name, "qed_virtual_photons_min_energy", vphoton_min_energy);

        // Sampling factor (a.k.a. multiplier):
        // the number of virtual photons generated is multiplied by this factor,
        // the weight of each virtual photon is divided by this factor
        amrex::Real sampling_factor = 0.0_rt;
        utils::parser::getWithParser(pp_species_name, "qed_virtual_photons_multiplier", sampling_factor);

        amrex::Real const alpha_over_pi = PhysConst::alpha / MathConst::pi;
        amrex::Real constexpr inv_c2 = PhysConst::inv_c2;
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
                ParticleTileType& ptile_vp = vphotons.ParticlesAt(lev, mfi);
                ptile_vp.resize(total_num_vp);

                // Get the starting particle ID on CPU and reserve IDs for all virtual photons
                // This must be done on CPU because NextID() is not thread-safe and cannot be called from GPU
                amrex::Long pid;
#ifdef AMREX_USE_OMP
#pragma omp critical (virtual_photon_nextid)
#endif
                {
                    pid = ParticleTileType::ParticleType::NextID();
                    ParticleTileType::ParticleType::NextID(pid + total_num_vp);
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
                    const amrex::ParticleReal r  = soa.m_rdata[PIdx::r][i];
                    const amrex::ParticleReal z  = soa.m_rdata[PIdx::z][i];
                    const amrex::ParticleReal theta  = soa.m_rdata[PIdx::theta][i];
#elif defined (WARPX_DIM_1D_Z)
                    const amrex::ParticleReal z  = soa.m_rdata[PIdx::z][i];
#elif defined (WARPX_DIM_RCYLINDER)
                    const amrex::ParticleReal r  = soa.m_rdata[PIdx::r][i];
                    const amrex::ParticleReal theta  = soa.m_rdata[PIdx::theta][i];
#elif defined(WARPX_DIM_RSPHERE)
                    const amrex::ParticleReal r  = soa.m_rdata[PIdx::r][i];
                    const amrex::ParticleReal theta  = soa.m_rdata[PIdx::theta][i];
                    const amrex::ParticleReal phi  = soa.m_rdata[PIdx::phi][i];
#endif
                    const amrex::ParticleReal w  = soa.m_rdata[PIdx::w][i];

                    // TODO: add a runtime attribute to the virtual photon species
                    // that containes the pid of the parent particle = soa.m_idcpu[i]
                    // This will allow to update the parent lepton if needed

                    // Minimum fractional (wrt primary particle) photon energy
                    const amrex::Real y_min = vphoton_min_energy / (mass * gamma_primary * PhysConst::c2);
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

#if defined (WARPX_DIM_3D)
                        pa_vp[PIdx::x][ip] = x;
                        pa_vp[PIdx::y][ip] = y;
                        pa_vp[PIdx::z][ip] = z;
                        // Beam size effect: displace the virtual photon position according to the virtuality
                        if (do_beam_size_effect){
                            // Find distance at which the photon should be displaced (radius)
                            // Refer to doi.org/10.1103/PhysRevAccelBeams.27.091001
                            // radius = hbar * c / [ e * sqrt( Q2*(1-y) ) ]
                            // where Q2 is the virtuality, y the fractional energy
                            // log(Q2) is uniformly distributed between Q2_min and Q2_max
                            // where Q2_min = y^2 Q2_max, Q2_max = (m_e * c)^2
                            // which implies that Q2 can be sampled as
                            // Q2 = Q2_min * (Q2_max / Q2_min)^(rnd)
                            // where rnd is a random number uniformly distributed between 0 and 1
                            const amrex::ParticleReal radius = PhysConst::reduced_compton_wavelength * std::pow(frac_energy,  -amrex::Random(engine))  / std::sqrt(1._rt - frac_energy);

                            const amrex::ParticleReal theta = 2.0_rt * MathConst::pi * amrex::Random(engine);
                            const amrex::ParticleReal cos_theta = std::cos(theta);
                            const amrex::ParticleReal sin_theta = std::sin(theta);

                            // The displacement must be perpendicular to the momentum
                            // Find two unit vectors perpendicular to the momentum
                            const amrex::ParticleReal x_dot_n = x * nx + y * ny + z * nz;
                            const amrex::ParticleReal x_perp1 = x - x_dot_n * nx;
                            const amrex::ParticleReal y_perp1 = y - x_dot_n * ny;
                            const amrex::ParticleReal z_perp1 = z - x_dot_n * nz;
                            const amrex::ParticleReal perp1_norm = std::sqrt(x_perp1*x_perp1 + y_perp1*y_perp1 + z_perp1*z_perp1);
                            const amrex::ParticleReal nx_perp1 = x_perp1 / perp1_norm;
                            const amrex::ParticleReal ny_perp1 = y_perp1 / perp1_norm;
                            const amrex::ParticleReal nz_perp1 = z_perp1 / perp1_norm;
                            const amrex::ParticleReal nx_perp2 = ny*nz_perp1-nz*ny_perp1;
                            const amrex::ParticleReal ny_perp2 = nz*nx_perp1-nx*nz_perp1;
                            const amrex::ParticleReal nz_perp2 = nx*ny_perp1-ny*nx_perp1;

                            // Move the photon position by radius along a random direction perpendicular to the momentum
                            pa_vp[PIdx::x][ip] += radius * (cos_theta * nx_perp1 + sin_theta * nx_perp2);
                            pa_vp[PIdx::y][ip] += radius * (cos_theta * ny_perp1 + sin_theta * ny_perp2);
                            pa_vp[PIdx::z][ip] += radius * (cos_theta * nz_perp1 + sin_theta * nz_perp2);
                        } // beam size effect
#elif defined (WARPX_DIM_XZ)
                        pa_vp[PIdx::x][ip] = x;
                        pa_vp[PIdx::z][ip] = z;
#elif defined (WARPX_DIM_RZ)
                        pa_vp[PIdx::r][ip] = r;
                        pa_vp[PIdx::z][ip] = z;
                        pa_vp[PIdx::theta][ip] = theta;
#elif defined (WARPX_DIM_1D_Z)
                        pa_vp[PIdx::z][ip] = z;
#elif defined (WARPX_DIM_RCYLINDER)
                        pa_vp[PIdx::r][ip] = r;
                        pa_vp[PIdx::theta][ip] = theta;
#elif defined(WARPX_DIM_RSPHERE)
                        pa_vp[PIdx::r][ip] = r;
                        pa_vp[PIdx::theta][ip] = theta;
                        pa_vp[PIdx::phi][ip] = phi;
#endif
                        pa_vp[PIdx::w][ip] = w / sampling_factor;
                        pid_vp[ip] = amrex::SetParticleIDandCPU(pid_start + ip, cpuid);
                    } // vphoton loop
                });
            } // mfi
        } // lev
    } // species

    if (do_beam_size_effect) {
        // The beam-size effect displaces virtual photons away from the cell
        // of their parent particle, so they need a Redistribute to be placed
        // in the correct box/tile. This must happen before they are used in
        // collisions (see CollisionHandler::doCollisions) since the collision
        // kernel assumes that particles are already in the correct box/tile.
        vphotons.Redistribute();
    }

#else

    WARPX_ABORT_WITH_MESSAGE("Compiling WarpX with QED support is required to call GenerateVirtualPhotons");
    amrex::ignore_unused(mypc);

#endif //WARPX_QED

} // function
} // close namespace
