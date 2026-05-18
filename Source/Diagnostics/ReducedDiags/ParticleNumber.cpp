/* Copyright 2019-2020 Neil Zaim, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ParticleNumber.H"

#include "Diagnostics/ReducedDiags/ReducedDiags.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParticleReduce.H>
#include <AMReX_Particles.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <map>
#include <ostream>
#include <vector>

using namespace amrex::literals;

// constructor
ParticleNumber::ParticleNumber (const std::string& rd_name)
: ReducedDiags{rd_name}
{
    // get a reference to WarpX instance
    auto & warpx = WarpX::GetInstance();

    // get MultiParticleContainer class object
    const auto & mypc = warpx.GetPartContainer();

    // get number of species (int)
    const auto nSpecies = mypc.nSpecies();

    // resize data array to 2*(nSpecies+1) (each species + sum over all species
    // for both number of macroparticles and of physical particles)
    m_data.resize(2*(nSpecies+1), 0.0_rt);

    // get species names (std::vector<std::string>)
    const auto species_names = mypc.GetSpeciesNames();

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
            ofs << m_sep;
            ofs << "[" << c++ << "]total_macroparticles()";
            // Column number of first species macroparticle number
            for (int i = 0; i < nSpecies; ++i)
            {
                ofs << m_sep;
                ofs << "[" << c++ << "]" << species_names[i] + "_macroparticles()";
            }
            // Column number of total weight (summed over all species)
            ofs << m_sep;
            ofs << "[" << c++ << "]total_weight()";
            // Column number of first species weight
            for (int i = 0; i < nSpecies; ++i)
            {
                ofs << m_sep;
                ofs << "[" << c++ << "]" << species_names[i] + "_weight()";
            }
            ofs << "\n";
            // close file
            ofs.close();
        }
    }
}
// end constructor

// function that computes total number of macroparticles and physical particles
void ParticleNumber::ComputeDiags (int step)
{
    // Judge if the diags should be done
    if (!m_intervals.contains(step+1)) { return; }

    // get MultiParticleContainer class object
    const auto & mypc = WarpX::GetInstance().GetPartContainer();

    // get number of species (int)
    const auto nSpecies = mypc.nSpecies();

    // Index of total number of macroparticles (all species) in m_data
    constexpr int idx_total_macroparticles = 0;
    // Index of first species macroparticle number in m_data
    constexpr int idx_first_species_macroparticles = 1;
    // Index of total weight (all species) in m_data
    const int idx_total_sum_weight = idx_first_species_macroparticles + nSpecies;
    // Index of first species weight in m_data
    const int idx_first_species_sum_weight = idx_total_sum_weight + 1;

    // Initialize total number of macroparticles and total weight (all species) to 0
    m_data[idx_total_macroparticles] = 0.0_rt;
    m_data[idx_total_sum_weight] = 0.0_rt;

    // loop over species
    for (int i_s = 0; i_s < nSpecies; ++i_s)
    {
        // get WarpXParticleContainer class object
        auto & myspc = mypc.GetParticleContainer(i_s);

        // Save total number of macroparticles for this species
        m_data[idx_first_species_macroparticles + i_s] = myspc.TotalNumberOfParticles();

        // Save sum of particles weight for this species
        m_data[idx_first_species_sum_weight + i_s] = myspc.sumParticleWeight(false);

        // Increase total number of macroparticles and total weight (all species)
        m_data[idx_total_macroparticles] += m_data[idx_first_species_macroparticles + i_s];
        m_data[idx_total_sum_weight] += m_data[idx_first_species_sum_weight + i_s];
    }
    // end loop over species

    /* m_data now contains up-to-date values for:
     *  [total number of macroparticles (all species),
     *   total number of macroparticles (species 1),
     *   ...,
     *   total number of macroparticles (species n)
     *   sum of particles weight (all species),
     *   sum of particles weight (species 1),
     *   ...,
     *   sum of particles weight (species n)] */
}
// end void ParticleNumber::ComputeDiags
