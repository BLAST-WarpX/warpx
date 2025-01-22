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


#include <AMReX_INT.H>
#include <AMReX_REAL.H>


namespace collision::binarycollision::virtualphotons{

using namespace amrex::literals;
using SoaData_type = typename WarpXParticleContainer::ParticleTileType::ParticleTileDataType;



void GenerateVirtualPhotons (MultiParticleContainer* mypc){

    //WarpXParticleContainer& lepton_species, WarpXParticleContainer& vphoton_species){

    int num = 0;
    int * n_photons_tot = &num;

    auto& leptons = mypc->GetParticleContainerFromName("beam");
    auto& vphotons = mypc->GetParticleContainerFromName("virtual_photons");


    // if incoherent pair generation, first create and initialize the virtual photons
    //const CollisionType collision_type = BinaryCollisionUtils::get_collision_type(collision_name, mypc);
    //if collision_type == CollisionType::PhotonPhotonToElectronPositron{
    //}

    // Enable tiling
    amrex::MFItInfo info;
    if (amrex::Gpu::notInLaunchRegion()) { info.EnableTiling(WarpXParticleContainer::tile_size); }

    const SmartCopyFactory copy_factory_phot(leptons, vphotons);
    const auto CopyPhot = copy_factory_phot.getSmartCopy();


    int const nlevs = std::max(0, leptons.finestLevel()+1); // species_1 ?
    for (int lev = 0; lev < nlevs; ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi = leptons.MakeMFIter(lev, info); mfi.isValid(); ++mfi)
        {

            const amrex::Box& box = mfi.validbox();
            amrex::Print(0) << "box : " << box << "\n";


            ParticleTileType& ptile_vph = vphotons.ParticlesAt(lev, mfi);
            auto soa_vph = ptile_vph.getParticleTileData();

            ParticleTileType* AMREX_RESTRICT tile_products =& ptile_vph ;


            ParticleTileType& ptile_lep = leptons.ParticlesAt(lev, mfi);
            const auto soa_lep = ptile_lep.getParticleTileData();

            int const np = ptile_lep.numParticles();
            amrex::Print(0) << "there is " << np << " electrons here \n";

            amrex::ParallelForRNG(np,
             [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
            {

                //CopyPhot(soa_vph_data[0], soa_lep, static_cast<int>(i), static_cast<int>(i), engine);
                amrex::ParticleReal ux = soa_lep.m_rdata[PIdx::ux][i];
                amrex::ParticleReal uy = soa_lep.m_rdata[PIdx::uy][i];
                amrex::ParticleReal uz = soa_lep.m_rdata[PIdx::uz][i];

                amrex::ParticleReal gamma = std::sqrt( 1.0_rt + ux*ux + uy*uy + uz*uz);
                amrex::ParticleReal mandelstam_s_4 = gamma * gamma; // Mandelstam variable s / 4

                amrex::Real compt_x_min = 1.e-2_rt; // min fractional photon energy
                amrex::ParticleReal xmin = compt_x_min / mandelstam_s_4;

                amrex::Real lnxmin = -std::log(xmin);
                amrex::Real lns4 = std::log(mandelstam_s_4);

                amrex::Real r_photons = PhysConst::alpha / MathConst::pi * lnxmin * (lnxmin + lns4);

                int n_photons = (int)r_photons;

                tile_products -> resize(*n_photons_tot + n_photons);

                // Create necessary GPU vectors, that will be used in the kernel below
                SoaData_type soa_products = tile_products->getParticleTileData();

                amrex::Print(0)<< "electron number " << i << " generates " << n_photons<< " v photons \n";

                for (int k = *n_photons_tot; k < *n_photons_tot + n_photons; k++){

                    CopyPhot(soa_products, soa_lep, static_cast<int>(i), static_cast<int>(k), engine);
                    soa_products.m_rdata[PIdx::w][k] = soa_lep.m_rdata[PIdx::w][i];
                    soa_products.m_rdata[PIdx::x][k] = soa_lep.m_rdata[PIdx::x][i];
                    soa_products.m_rdata[PIdx::y][k] = soa_lep.m_rdata[PIdx::y][i];
                    soa_products.m_rdata[PIdx::z][k] = soa_lep.m_rdata[PIdx::z][i];

                }

                *n_photons_tot = *n_photons_tot + n_photons;

            });
            amrex::Print(0)<< " total number of photons in current tile and level " << *n_photons_tot  << " " << lev << "\n";
/*
                // Initialize the user runtime components
                const auto start_index = 0;
                const auto stop_index  = * n_photons_tot;
                ParticleCreation::DefaultInitializeRuntimeAttributes(*tile_products,
                                           0, 0,
                                           vphotons.getUserRealAttribs(), vphotons.getUserIntAttribs(),
                                           vphotons.getParticleComps(), vphotons.getParticleiComps(),
                                           vphotons.getUserRealAttribParser(),
                                           vphotons.getUserIntAttribParser(),
#ifdef WARPX_QED
                                          false, // do not initialize QED quantities, since they were initialized
                                              // when calling the SmartCopy functors
                                           vphotons.get_breit_wheeler_engine_ptr(),
                                           vphotons.get_quantum_sync_engine_ptr(),
#endif
                                           vphotons.getIonizationInitialLevel(),
                                           start_index, stop_index);*/


        } // mfi
    } // lev


} // function
} // close namespace
#endif // WARPX_VIRTUAL_PHOTONS_H_





/*

float requiv(LocalParticle *part, const double e_primary, const double compt_x_min){

    Based on:
    GUINEA-PIG
    https://gitlab.cern.ch/clic-software/guinea-pig-legacy/-/blob/master/guinea_pig.c#L1205
    ----
    Returns the number of equivalent virtual (macro)photons for a single primary macroparticle of the beam.

    const double emass2=MELECTRON_GEV*MELECTRON_GEV;
    double lnxmin, s4, xmin, lns4, r_photons, n_photons;

    s4 = e_primary*e_primary;  // [GeV^2] mandelstam s divided by 4: s/4
    xmin = compt_x_min * emass2 / s4;  // [1] virtual photon's energy fraction compared to primary energy

    if (xmin>=1.0) return 0.0;

    lnxmin=-log(xmin);
    lns4 = log(s4 / (MELECTRON_GEV*MELECTRON_GEV));

    r_photons = .00232461*lnxmin*(lnxmin + lns4);  // alpha/pi = 0.00232461

    // account for noninteger photon by randomly emitting n+1 sometimes
    n_photons = (int)floor(r_photons);
    r_photons -= n_photons;
    if(RandomUniform_generate(part) < r_photons) n_photons += 1.0;

    return n_photons;
}

    int const nlevs = std::max(0, lepton_species.finestLevel()+1); // species_1 ?
    for (int lev = 0; lev < nlevs; ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (WarpXParIter pti(lepton_species, lev, info); pti.isValid(); ++pti)
        {

            auto& src_tile = lepton_species.ParticlesAt(lev, pti);
            auto& dst_tile = vphoton_species.ParticlesAt(lev, pti);

            const long np = pti.numParticles();

            amrex::Print(0) << "XXX " << np << "\n";

            const auto np_dst = dst_tile.numParticles();

            const auto num_added =
                filterCopyTransformParticles<1>(vphoton_species, dst_tile, src_tile, np_dst,
                                                Filter, CopyPhot, Transform);

            setNewParticleIDs(dst_tile, np_dst, num_added);

*/




//    const SmartCopyFactory copy_factory_phot(lepton_species, vphoton_species);
//    const auto CopyPhot = copy_factory_phot.getSmartCopy();

//    lepton_species.defineAllParticleTiles();
//    vphoton_species.defineAllParticleTiles();



/*
    int const nlevs = std::max(0, lepton_species.finestLevel(+1);
    for (int lev = 0; lev < nlevs; ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif

        for (WarpXParIter pti(lepton_species, lev); pti.isValid(); ++pti)
        {

            auto& soa = pti.GetAttribs();
            auto& w = soa[PIdx::w];
            auto& x = soa[PIdx::x];
            auto& y = soa[PIdx::y];
            auto& z = soa[PIdx::z];
            auto& ux = soa[PIdx::ux];
            auto& uy = soa[PIdx::uy];
            auto& uz = soa[PIdx::uz];
            const long np = pti.numParticles();

            // Declare temporary vectors on the CPU
            amrex::Gpu::HostVector<amrex::ParticleReal> w_vp;
            amrex::Gpu::HostVector<amrex::ParticleReal> x_vp;
            amrex::Gpu::HostVector<amrex::ParticleReal> y_vp;
            amrex::Gpu::HostVector<amrex::ParticleReal> z_vp;
            amrex::Gpu::HostVector<amrex::ParticleReal> ux_vp;
            amrex::Gpu::HostVector<amrex::ParticleReal> uy_vp;
            amrex::Gpu::HostVector<amrex::ParticleReal> uz_vp;

            amrex::ParallelFor( np,
                [=] AMREX_GPU_DEVICE(long i)
            {
                //amrex::Real gamma = std::sqrt(ux[i]*ux[i] + uy[i]*uy[i] + uz[i]*uz[i] + 1);
                //amrex::Real energy = gamma * PhysConst::m_e * PhysConst::c * PhysConst::c;
                vphoton_species.
                PhysicalParticleContainer::CheckAndAddParticle(x[i], y[i], z[i], ux[i], uy[i], uz[i], w[i],
                                    x_vp, y_vp, z_vp, ux_vp, uy_vp, uz_vp, w_vp); // tlab???
            }); // parallel for
        } // mfi loop
    } //levels
    */
