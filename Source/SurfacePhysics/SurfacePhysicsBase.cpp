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
#include "Utils/Parser/ParserUtils.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

SurfacePhysicsBase::SurfacePhysicsBase ()
{
    amrex::Print() << " in surface physics base class \n";
    ReadParameters();
}

void SurfacePhysicsBase::ReadParameters ()
{
    amrex::ParmParse const pp_surface_chemistry("surface_chemistry");
    std::string chemistry_file;
    pp_surface_chemistry.query("input_file", chemistry_file);

    amrex::ParmParse::addfile(chemistry_file);
    amrex::ParmParse const pp_chem("chem");

    // Read gas species that participate in gas-surface physics
    amrex::Vector<std::string> chem_gas_species;
    pp_chem.queryarr("gasphase_species", chem_gas_species);
    amrex::ParmParse const pp_gasphase("gasphase_species");
    for (const auto& species : chem_gas_species) {
        std::string symbol;
        //pp_chem.query(("gasphase_species."+ species + ".symbol").c_str(), symbol);
        utils::parser::query(pp_gasphase, species, "symbol", symbol);
        gas_species[species] = symbol;
    }

    // Read surface species that participate in gas-surface physics
    amrex::Vector<std::string> chem_surface_species;
    pp_chem.queryarr("surface_species", chem_surface_species);
    amrex::ParmParse const pp_surface("surface_species");
    for (const auto& species : chem_surface_species) {
        std::string symbol;
        utils::parser::query(pp_surface, species, "symbol", symbol);
        surface_species[species] = symbol;
    }

    std::set<std::string> known_symbols;
    for (const auto& [_, symbol] : gas_species) known_symbols.insert(symbol);
    for (const auto& [_, symbol] : surface_species) known_symbols.insert(symbol);


    amrex::Vector<std::string> gas_surface_reactions;
    if (pp_chem.queryarr("reactions", gas_surface_reactions)) {
        for (const auto& line : gas_surface_reactions) {
            Reaction rxn;
            std::vector<std::string> equation_params = amrex::split(line,";");

            rxn.equation = amrex::trim(equation_params[0]);
            auto arrow_pos = rxn.equation.find("=>");
            if (arrow_pos == std::string::npos) {
                amrex::Abort( " Reaction eqution must contain '=>' separator.");
            }
            std::string lhs = amrex::trim(rxn.equation.substr(0,arrow_pos));
            std::string rhs = amrex::trim(rxn.equation.substr(arrow_pos+2));
            rxn.reactants = tokenize_reaction(lhs);
            rxn.products = tokenize_reaction(rhs);
            for (const auto& reactant : rxn.reactants) {
                std::string species_type = is_gas_species(reactant) ? "gas" : "surface";
                rxn.reactant_type.push_back(species_type);
            }
            for (const auto& product : rxn.products) {
                std::string species_type = is_gas_species(product) ? "gas" : "surface";
                rxn.product_type.push_back(species_type);
            }

            rxn.P_energy0 = std::stod(amrex::trim(equation_params[1]));
            rxn.P0        = std::stod(amrex::trim(equation_params[2]));
            rxn.E_ref     = std::stod(amrex::trim(equation_params[3]));
            rxn.E_th      = std::stod(amrex::trim(equation_params[4]));
            rxn.exp       = std::stod(amrex::trim(equation_params[5]));

            reactions.push_back(rxn);
        }
    } else {
        amrex::Print() << " no reactions specified for surface physics \n";
    }
}

bool
SurfacePhysicsBase::is_gas_species (std::string species_symbol)
{
    for (const auto& [_,symbol] : gas_species) {
        if (symbol == species_symbol) return true;
    }
    return false;
}


bool
SurfacePhysicsBase::is_surface_species (std::string species_symbol)
{
    for (const auto& [_,symbol] : surface_species) {
        if (symbol == species_symbol) return true;
    }
    return false;
}

amrex::Vector<std::string>
SurfacePhysicsBase::tokenize_reaction (const std::string& input) {

    amrex::Vector<std::string> result;
    std::string modified = input;

    // replacing "+_" with "%_" temporarily
    std::string::size_type pos = 0;
    while ((pos = modified.find("+_", pos)) != std::string::npos) {
        modified.replace(pos, 2, "%_");
        pos += 2;
    }

    std::vector<std::string> terms = amrex::split(modified, "+");
    for (auto& term : terms) {
        std::string restored = amrex::trim(term);
        std::string::size_type p = 0;
        while ((p = restored.find("%_", p)) != std::string::npos) {
            restored.replace(p, 2, "+_");
            p += 2;
        }
        result.push_back(restored);
    }
    return result;
}

void
SurfacePhysicsBase::InitData ()
{
    initializeMapping();
    auto & warpx = WarpX::GetInstance();
    const auto & mpc = warpx.GetPartContainer();
    num_influx_species = mpc.nSpecies();
    std::vector<std::string> influx_species_names = mpc.GetSpeciesNames();
    num_outflux_species = num_influx_species; //for now
    AllocAndInitInfluxBndVectors();
    AllocAndInitOutfluxBndVectors();
}

void
SurfacePhysicsBase::initializeMapping ()
{
    // get a reference to WarpX instance
    auto & warpx = WarpX::GetInstance();

    const int lev = 0;

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
        amrex::FabType const fab_type = eb_flag[mfi].getType(box);
        if (fab_type == amrex::FabType::regular) { continue;}
        else if (fab_type == amrex::FabType::covered) { continue;}

        // all cells in fab are open, i.e., outside EB
        if (fab_type == amrex::FabType::regular) {continue;}
        // all cells in fab are enclosed within EB
        if (fab_type == amrex::FabType::covered) {continue;}

        auto const& eb_flag_arr = eb_flag.array(mfi);
        const amrex::Array4<const amrex::Real> & eb_bnd_normal_arr = eb_bnd_normal.array(mfi);
        auto const ivect_arr = ivect_map->array(mfi);

        amrex::LoopOnCpu( box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

            amrex::IntVect const iv(AMREX_D_DECL(i,j,k));            
            if (eb_flag_arr(i,j,k).isRegular() ) {
                return;
            }
            else if (eb_flag_arr(i,j,k).isCovered() ) {
                return;
            }
            else {
                surf_ijk.push_back(iv);
                ivect_arr(i,j,k) = surf_ijk.size() - 1;

                surf_normal_x.push_back(eb_bnd_normal_arr(i,j,k,0));
#if (defined WARPX_DIM_XZ)
                surf_normal_z.push_back(eb_bnd_normal_arr(i,j,k,1));
#elif (defined WARPX_DIM_3D)
                surf_normal_y.push_back(eb_bnd_normal_arr(i,j,k,1));
                surf_normal_z.push_back(eb_bnd_normal_arr(i,j,k,2));
#endif
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
        bnd_outflux[isp][ibnd] = 5.e5;
    }
}

#endif
