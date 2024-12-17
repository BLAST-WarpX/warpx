/* Copyright 2024 Revathi Jambunathan
 *
 * This file is part of WarpX
 *
 * License: BSD-3-Clause-LBNL
 */

#ifdef WARPX_SURFACE_PHYSICS

#include "SurfacePhysicsBase.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_Print.H>

SurfacePhysicsBase::SurfacePhysicsBase ()
{
    amrex::Print() << " in surface physics base class \n";
}


void
SurfacePhysicsBase::InitData ()
{
    initializeMapping();
    auto & warpx = WarpX::GetInstance();
    const auto & mpc = warpx.GetPartContainer();
    num_influx_species = mpc.nSpecies();
    std::vector<std::string> influx_species_names = mpc.GetSpeciesNames();
    amrex::Print() << " num_influx species " << mpc.nSpecies() << "\n";
    amrex::Print() << " influx_sp names " << influx_species_names[0] << " " << influx_species_names[1] << "\n";
    num_outflux_species = num_influx_species; //for now
    AllocAndInitInfluxBndVectors();
    AllocAndInitOutfluxBndVectors();
}

void
SurfacePhysicsBase::initializeMapping ()
{
    amrex::Print() << " init mapping \n";

    // get a reference to WarpX instance
    auto & warpx = WarpX::GetInstance();

    const int lev = 0;
//    const auto dx = warpx.Geom(lev).CellSizeArray();
//    const auto problo = warpx.Geom(lev).ProbLoArray();

    // check if EB is enabled
    if (!EB::enabled() ) {
        amrex::Print() << " current mapping works only with EB surfaces \n";
        return;
    }
    //
    amrex::EBFArrayBoxFactory const& eb_box_factory = warpx.fieldEBFactory(lev);
    amrex::FabArray<amrex::EBCellFlagFab> const& eb_flag = eb_box_factory.getMultiEBCellFlagFab();
    amrex::MultiCutFab const& eb_bnd_cent = eb_box_factory.getBndryCent();
    amrex::MultiCutFab const& eb_bnd_normal = eb_box_factory.getBndryNormal();

    ivect_map = std::make_unique< amrex::iMultiFab> (warpx.boxArray(lev), warpx.DistributionMap(lev), 1, 1);
    ivect_map->setVal(0);

    for (amrex::MFIter mfi(eb_flag); mfi.isValid(); ++mfi)
    {
        amrex::Box const box = mfi.tilebox();
        amrex::Print() << " eb box " << box << "\n";
        amrex::FabType const fab_type = eb_flag[mfi].getType(box);
        if (fab_type == amrex::FabType::regular) { continue;}
        else if (fab_type == amrex::FabType::covered) { continue;}
        else {amrex::Print() << " fab types is neither regular nor convered \n";}

        // all cells in fab are open, i.e., outside EB
        if (fab_type == amrex::FabType::regular) {continue;}
        // all cells in fab are enclosed within EB
        if (fab_type == amrex::FabType::covered) {continue;}

        auto const& eb_flag_arr = eb_flag.array(mfi);
        const amrex::Array4<const amrex::Real> & eb_bnd_normal_arr = eb_bnd_normal.array(mfi);
        const amrex::Array4<const amrex::Real> & eb_bnd_cent_arr = eb_bnd_cent.array(mfi);
        auto const ivect_arr = ivect_map->array(mfi);

        amrex::LoopOnCpu( box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

            amrex::IntVect const iv(AMREX_D_DECL(i,j,k));            
            if (eb_flag_arr(i,j,k).isRegular() ) {
                return;
            }
            else if (eb_flag_arr(i,j,k).isCovered() ) {
                return;
                amrex::Print() << " cell " << i << " " << j << " " << k << " is covered \n";
            }
            else {
                amrex::Print() << " cell" << i << " j " << j << " " << k << " is cut!! \n";
                surf_ijk.push_back(iv);
                amrex::Print() << " size : " << surf_ijk.size() << " intvect " << surf_ijk[surf_ijk.size()-1] << "\n";
                ivect_arr(i,j,k) = surf_ijk.size() - 1;
                amrex::Print() << " iMultiFab " << i << " " << j << " is : " << ivect_arr(i,j,k) << "\n";

                surf_normal_x.push_back(eb_bnd_normal_arr(i,j,k,0));
#if (defined WARPX_DIM_XZ)
                surf_normal_z.push_back(eb_bnd_normal_arr(i,j,k,1));
#elif (defined WARPX_DIM_3D)
                surf_normal_y.push_back(eb_bnd_normal_arr(i,j,k,1));
                surf_normal_z.push_back(eb_bnd_normal_arr(i,j,k,2));
#endif
//                amrex::Print() << " surface normal is  " << surf_normal_x[surf_ijk.size()-1] << " " << surf_normal_z[surf_ijk.size()-1] << " \n";
//                amrex::Print() << " what is cent ? " << eb_bnd_cent_arr(i,j,k)  << " " << eb_bnd_cent_arr(i,j,k,1)<< "\n";
//                amrex::Real x_loc = problo[0] + (i + 0.5 + eb_bnd_cent_arr(i,j,k,0) ) * dx[0];
//                amrex::Real z_loc = problo[1] + (j + 0.5 + eb_bnd_cent_arr(i,j,k,1) ) * dx[1];
//                amrex::Print() << " x loc " << x_loc << "  zloc " << z_loc << "\n";
            }
        });
    }
    
}


void
SurfacePhysicsBase::AllocAndInitInfluxBndVectors ()
{
    num_in_particles.resize(num_influx_species);
    bnd_influx.resize(num_influx_species);
    for (int isp = 0; isp < num_influx_species; ++isp)
    {
        num_in_particles[isp].resize(surf_ijk.size());
        bnd_influx[isp].resize(surf_ijk.size());

        nullifyInfluxParticleCounter(isp);
    }
}

void
SurfacePhysicsBase::AllocAndInitOutfluxBndVectors ()
{
    num_out_particles.resize(num_outflux_species);
    bnd_outflux.resize(num_outflux_species);
    for (int isp = 0; isp < num_outflux_species; ++isp)
    {
        num_out_particles[isp].resize(surf_ijk.size());
        bnd_outflux[isp].resize(surf_ijk.size());

        nullifyOutfluxParticleCounter(isp);
    }

}

void
SurfacePhysicsBase::nullifyInfluxParticleCounter ()
{
    for (int isp = 0; isp < num_influx_species; ++isp) {
        nullifyInfluxParticleCounter(isp);
    }
}

void
SurfacePhysicsBase::nullifyInfluxParticleCounter (int isp)
{
    for (int ibnd = 0; ibnd < surf_ijk.size(); ++ibnd)
    {
        num_in_particles[isp][ibnd] = 0;
        bnd_influx[isp][ibnd] = 0.;
    }
}

void
SurfacePhysicsBase::nullifyOutfluxParticleCounter ()
{
    for (int isp = 0; isp < num_outflux_species; ++isp) {
        nullifyOutfluxParticleCounter(isp);
    }
}

void
SurfacePhysicsBase::nullifyOutfluxParticleCounter (int isp)
{    
    for (int ibnd = 0; ibnd < surf_ijk.size(); ++ibnd)
    {
        num_out_particles[isp][ibnd] = 0;
        bnd_outflux[isp][ibnd] = 0.;
    }
}

#endif
