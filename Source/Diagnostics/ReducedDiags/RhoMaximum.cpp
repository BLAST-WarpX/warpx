/* Copyright 2020 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "RhoMaximum.H"

#include "Diagnostics/ComputeDiagFunctors/RhoFunctor.H"
#include "Diagnostics/ReducedDiags/ReducedDiags.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <ostream>
#include <vector>

using namespace amrex::literals;

// constructor
RhoMaximum::RhoMaximum (const std::string& rd_name)
: ReducedDiags{rd_name}
{
    // RZ coordinate is not working
#if (defined WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(false,
        "RhoMaximum reduced diagnostics does not work for RZ coordinate.");
#endif

    // read number of levels
    int nLevel = 0;
    const amrex::ParmParse pp_amr("amr");
    pp_amr.query("max_level", nLevel);
    nLevel += 1;
    m_rho_functors.resize(nLevel);

    // We do not use coarsening in the functors for this diag.
    const amrex::IntVect crse_ratio = amrex::IntVect(1);

    // Initialize functors for the total charge density
    for (int lev = 0; lev < nLevel; ++lev)
    {
        m_rho_functors[lev].push_back(std::make_unique<RhoFunctor>(lev, crse_ratio));
    }

    // get MultiParticleContainer class object
    const auto & mypc = WarpX::GetInstance().GetPartContainer();

    // get number of species (int)
    const auto nSpecies = mypc.nSpecies();

    // A vector to store the indices of charges species in mypc
    amrex::Vector<int> indices_charged_species;

    // number of charged species
    int n_charged_species = 0;

    for (int i = 0; i < nSpecies; ++i)
    {
        // Only charged species are relevant for this diag
        if (mypc.GetParticleContainer(i).getCharge() != 0.0_rt)
        {
            indices_charged_species.push_back(i);
            n_charged_species += 1;
            for (int lev = 0; lev < nLevel; ++lev)
            {
                // Initialize functors for the charge density of each charged species
                m_rho_functors[lev].push_back(std::make_unique<RhoFunctor>(lev, crse_ratio, false, i));
            }
        }
    }

    // get species names (std::vector<std::string>)
    const auto species_names = mypc.GetSpeciesNames();

    // Min and max of total rho + max of |rho| for each species
    const int noutputs_per_level = 2+n_charged_species;
    m_data.resize(static_cast<std::size_t>(nLevel*noutputs_per_level), 0.0_rt);

    if (amrex::ParallelDescriptor::IOProcessor())
    {
        if ( m_write_header )
        {
            // open file
            std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};
            // write header row
            int c = 0;
            ofs << "#";
            ofs << "[" << c++ << "]step()";
            ofs << m_sep;
            ofs << "[" << c++ << "]time(s)";
            for (int lev = 0; lev < nLevel; ++lev)
            {
                ofs << m_sep;
                ofs << "[" << c++ << "]max_rho_lev" + std::to_string(lev) + "(C/m^3)";
                ofs << m_sep;
                ofs << "[" << c++ << "]min_rho_lev" + std::to_string(lev) + "(C/m^3)";
                for (int i = 0; i < n_charged_species; ++i)
                {
                    ofs << m_sep;
                    ofs << "[" << c++ << "]max_" + species_names[indices_charged_species[i]]
                                         + "_|rho|_lev" + std::to_string(lev) + "(C/m^3)";
                }
            }
            ofs << "\n";
            // close file
            ofs.close();
        }
    }
}
// end constructor

// function that computes maximum charge density values
void RhoMaximum::ComputeDiags (int step)
{
    // Judge if the diags should be done
    if (!m_intervals.contains(step+1)) { return; }

    // get a reference to WarpX instance
    auto & warpx = WarpX::GetInstance();

    // get number of levels
    const auto nLevel = warpx.finestLevel() + 1;

    const auto n_charged_species = static_cast<int>(m_rho_functors[0].size() - 1);
    // Min and max of total rho + max of |rho| for each species
    const int noutputs_per_level = 2+n_charged_species;

    // loop over refinement levels
    for (int lev = 0; lev < nLevel; ++lev)
    {
        // Declare a temporary MultiFAB to store the charge densities.
        const amrex::BoxArray ba = warpx.boxArray(lev);
        const amrex::DistributionMapping dmap = warpx.DistributionMap(lev);
        constexpr int ncomp = 1;
        constexpr int ngrow = 0;
        amrex::MultiFab mf_temp(ba, dmap, ncomp, ngrow);

        // Fill temporary MultiFAB with total charge density
        constexpr int idx_total_rho_functor = 0;
        constexpr int idx_first_species_functor = 1;
        constexpr int icomp = 0;
        constexpr int i_buffer = 0;
        m_rho_functors[lev][idx_total_rho_functor]->operator()(mf_temp, icomp, i_buffer);

        constexpr int idx_max_rho_data = 0;
        constexpr int idx_min_rho_data = 1;
        constexpr int idx_first_species_data = 2;

        // Fill output array with min and max of total rho
        m_data[lev*noutputs_per_level + idx_max_rho_data] = mf_temp.max(icomp);
        m_data[lev*noutputs_per_level + idx_min_rho_data] = mf_temp.min(icomp);

        // Loop over all charged species
        for (int i = 0; i < n_charged_species; ++i)
        {
            // Fill temporary MultiFAB with the species charge density
            m_rho_functors[lev][idx_first_species_functor+i]->operator()(mf_temp, icomp, i_buffer);
            // Fill output array with max |rho| of species
            m_data[lev*noutputs_per_level + idx_first_species_data + i] = mf_temp.norm0();
        }
    }
    // end loop over refinement levels

    /* m_data now contains up-to-date values for:
     *  [max(rho), min(rho), max(|rho_charged_species1|), max(|rho_charged_species2|), ...] */
}
// end void RhoMaximum::ComputeDiags
