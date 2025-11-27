/* Copyright 2019-2021 Luca Fedeli, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ParticleEnergy.H"

#include "Diagnostics/ReducedDiags/ReducedDiags.H"
#include "Particles/Algorithms/KineticEnergy.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/SpeciesPhysicalProperties.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParticleReduce.H>
#include <AMReX_Particles.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>
#include <AMReX_Tuple.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <vector>

using namespace amrex;

// constructor
ParticleEnergy::ParticleEnergy (const std::string& rd_name)
: ReducedDiags{rd_name}
{
    // get a reference to WarpX instance
    auto & warpx = WarpX::GetInstance();

    // get MultiParticleContainer class object
    const auto & mypc = warpx.GetPartContainer();

    // get number of species (int)
    const auto nSpecies = mypc.nSpecies();

    // resize data array
    m_data.resize(2*nSpecies+2, 0.0_rt);

    // get species names (std::vector<std::string>)
    const auto species_names = mypc.GetSpeciesNames();

    if (ParallelDescriptor::IOProcessor())
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
            ofs << "[" << c++ << "]total(J)";
            for (int i = 0; i < nSpecies; ++i)
            {
                ofs << m_sep;
                ofs << "[" << c++ << "]" << species_names[i] + "(J)";
            }
            ofs << m_sep;
            ofs << "[" << c++ << "]total_mean(J)";
            for (int i = 0; i < nSpecies; ++i)
            {
                ofs << m_sep;
                ofs << "[" << c++ << "]" << species_names[i] + "_mean(J)";
            }
            ofs << "\n";
            // close file
            ofs.close();
        }
    }
}

void ParticleEnergy::ComputeDiags (int step)
{
    // Check if the diags should be done
    if (!m_intervals.contains(step+1)) { return; }

    // Get MultiParticleContainer class object
    const auto & mypc = WarpX::GetInstance().GetPartContainer();

    // Get number of species
    const int nSpecies = mypc.nSpecies();

    amrex::ParticleReal Wtot = 0.0_rt;

    // Loop over species
    for (int i_s = 0; i_s < nSpecies; ++i_s)
    {
        // Get WarpXParticleContainer class object
        const auto & myspc = mypc.GetParticleContainer(i_s);

        auto [Etot, Ws] = myspc.sumParticleWeightAndEnergy(false);

        // Accumulate sum of weights over all species (must come after MPI reduction of Ws)
        Wtot += Ws;

        // Save results for this species i_s into m_data

        // Offset:
        // 1 value of total energy for all  species +
        // 1 value of total energy for each species
        const int offset_total_species = 1 + i_s;
        m_data[offset_total_species] = Etot;

        // Offset:
        // 1 value of total energy for all  species +
        // 1 value of total energy for each species +
        // 1 value of mean  energy for all  species +
        // 1 value of mean  energy for each species
        const int offset_mean_species = 1 + nSpecies + 1 + i_s;
        if (Ws > std::numeric_limits<Real>::min())
        {
            m_data[offset_mean_species] = Etot / Ws;
        }
        else
        {
            m_data[offset_mean_species] = 0.0_rt;
        }
    }

    // Total energy
    m_data[0] = 0.0_rt;

    // Loop over species
    for (int i_s = 0; i_s < nSpecies; ++i_s)
    {
        // Offset:
        // 1 value of total energy for all  species +
        // 1 value of total energy for each species
        const int offset_total_species = 1 + i_s;
        m_data[0] += m_data[offset_total_species];
    }

    // Total mean energy. Offset:
    // 1 value of total energy for all  species +
    // 1 value of total energy for each species
    const int offset_mean_all = 1 + nSpecies;
    if (Wtot > std::numeric_limits<Real>::min())
    {
        m_data[offset_mean_all] = m_data[0] / Wtot;
    }
    else
    {
        m_data[offset_mean_all] = 0.0_rt;
    }

    // m_data now contains up-to-date values for:
    // [total energy (all species)
    //  total energy (species 1)
    //  ...
    //  total energy (species n)
    //  mean  energy (all species)
    //  mean  energy (species 1)
    //  ...
    //  mean  energy (species n)]
}
