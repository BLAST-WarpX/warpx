/* Copyright 2019-2020
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ParticleMomentum.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/SpeciesPhysicalProperties.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXConst.H"
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
#include <limits>
#include <map>
#include <ostream>
#include <string>
#include <vector>

using namespace amrex;

ParticleMomentum::ParticleMomentum (const std::string& rd_name)
    : ReducedDiags{rd_name}
{
    // Get a reference to WarpX instance
    auto & warpx = WarpX::GetInstance();

    // Get MultiParticleContainer class object
    const auto & mypc = warpx.GetPartContainer();

    // Get number of species
    const int nSpecies = mypc.nSpecies();

    // Resize data array
    m_data.resize(6*nSpecies+6, 0.0_rt);

    // Get species names
    const std::vector<std::string> species_names = mypc.GetSpeciesNames();

    if (ParallelDescriptor::IOProcessor())
    {
        if (m_write_header)
        {
            // Open file
            std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};

            int c = 0;

            // Write header row
            ofs << "#";
            ofs << "[" << c++ << "]step()";
            ofs << m_sep;
            ofs << "[" << c++ << "]time(s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]total_x(kg*m/s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]total_y(kg*m/s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]total_z(kg*m/s)";

            for (int i = 0; i < nSpecies; ++i)
            {
                ofs << m_sep;
                ofs << "[" << c++ << "]";
                ofs << species_names[i] + "_x(kg*m/s)";
                ofs << m_sep;
                ofs << "[" << c++ << "]";
                ofs << species_names[i] + "_y(kg*m/s)";
                ofs << m_sep;
                ofs << "[" << c++ << "]";
                ofs << species_names[i] + "_z(kg*m/s)";
            }

            ofs << m_sep;
            ofs << "[" << c++ << "]";
            ofs << "total_mean_x(kg*m/s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]";
            ofs << "total_mean_y(kg*m/s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]";
            ofs << "total_mean_z(kg*m/s)";

            for (int i = 0; i < nSpecies; ++i)
            {
                ofs << m_sep;
                ofs << "[" << c++ << "]";
                ofs << species_names[i] + "_mean_x(kg*m/s)";
                ofs << m_sep;
                ofs << "[" << c++ << "]";
                ofs << species_names[i] + "_mean_y(kg*m/s)";
                ofs << m_sep;
                ofs << "[" << c++ << "]";
                ofs << species_names[i] + "_mean_z(kg*m/s)";
            }

            ofs << "\n";
            ofs.close();
        }
    }
}

void ParticleMomentum::ComputeDiags (int step)
{
    // Check if the diags should be done
    if (!m_intervals.contains(step+1)) { return; }

    // Get MultiParticleContainer class object
    const auto & mypc = WarpX::GetInstance().GetPartContainer();

    // Get number of species
    const int nSpecies = mypc.nSpecies();

    amrex::Real Wtot = 0.0_rt;

    // Loop over species
    for (int i_s = 0; i_s < nSpecies; ++i_s)
    {
        // Get WarpXParticleContainer class object
        const auto & myspc = mypc.GetParticleContainer(i_s);

        // Get mass
        // (photons must be treated as a special case: they have zero mass,
        // but ux, uy, uz are calculated assuming a mass equal to the electron mass)
        const amrex::Real m = (myspc.AmIA<PhysicalSpecies::photon>()) ? PhysConst::m_e : myspc.getMass();

        using PType = typename WarpXParticleContainer::SuperParticleType;

        // Use amrex::ParticleReduce to compute the sum of the momenta and weights of all particles
        // held by the current MPI rank for this species (loop over all boxes held by this MPI rank):
        // the result r is the tuple (Px, Py, Pz, Ws)
        amrex::ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_ops;
        auto r = amrex::ParticleReduce<amrex::ReduceData<Real, Real, Real, Real>>(
            myspc,
            [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<Real, Real, Real, Real>
            {
                const amrex::Real w  = p.rdata(PIdx::w);
                const amrex::Real ux = p.rdata(PIdx::ux);
                const amrex::Real uy = p.rdata(PIdx::uy);
                const amrex::Real uz = p.rdata(PIdx::uz);
                return {w*m*ux, w*m*uy, w*m*uz, w};
            },
            reduce_ops);

        amrex::Real Px = amrex::get<0>(r);
        amrex::Real Py = amrex::get<1>(r);
        amrex::Real Pz = amrex::get<2>(r);
        amrex::Real Ws = amrex::get<3>(r);

        // Reduced sum over MPI ranks
        ParallelDescriptor::ReduceRealSum({Px,Py,Pz,Ws}, ParallelDescriptor::IOProcessorNumber());

        // Accumulate sum of weights over all species (must come after MPI reduction of Ws)
        Wtot += Ws;

        // Save results for this species i_s into m_data

        // Offset:
        // 3 values of total momentum for all  species +
        // 3 values of total momentum for each species
        const int offset_total_species = 3 + i_s*3;
        m_data[offset_total_species+0] = Px;
        m_data[offset_total_species+1] = Py;
        m_data[offset_total_species+2] = Pz;

        // Offset:
        // 3 values of total momentum for all  species +
        // 3 values of total momentum for each species +
        // 3 values of mean  momentum for all  species +
        // 3 values of mean  momentum for each species
        const int offset_mean_species = 3 + nSpecies*3 + 3 + i_s*3;
        if (Ws > std::numeric_limits<Real>::min())
        {
            m_data[offset_mean_species+0] = Px / Ws;
            m_data[offset_mean_species+1] = Py / Ws;
            m_data[offset_mean_species+2] = Pz / Ws;
        }
        else
        {
            m_data[offset_mean_species+0] = 0.0_rt;
            m_data[offset_mean_species+1] = 0.0_rt;
            m_data[offset_mean_species+2] = 0.0_rt;
        }
    }

    // Total momentum
    m_data[0] = 0.0_rt;
    m_data[1] = 0.0_rt;
    m_data[2] = 0.0_rt;

    // Loop over species
    for (int i_s = 0; i_s < nSpecies; ++i_s)
    {
        // Offset:
        // 3 values of total momentum for all  species +
        // 3 values of total momentum for each species
        const int offset_total_species = 3 + i_s*3;
        m_data[0] += m_data[offset_total_species+0];
        m_data[1] += m_data[offset_total_species+1];
        m_data[2] += m_data[offset_total_species+2];
    }

    // Total mean momentum. Offset:
    // 3 values of total momentum for all  species +
    // 3 values of total momentum for each species
    const int offset_mean_all = 3 + nSpecies*3;
    if (Wtot > std::numeric_limits<Real>::min())
    {
        m_data[offset_mean_all+0] = m_data[0] / Wtot;
        m_data[offset_mean_all+1] = m_data[1] / Wtot;
        m_data[offset_mean_all+2] = m_data[2] / Wtot;
    }
    else
    {
        m_data[offset_mean_all+0] = 0.0_rt;
        m_data[offset_mean_all+1] = 0.0_rt;
        m_data[offset_mean_all+2] = 0.0_rt;
    }

    // m_data contains up-to-date values for:
    // [total momentum along x (all species)
    //  total momentum along y (all species)
    //  total momentum along z (all species)
    //  total momentum along x (species 1)
    //  total momentum along y (species 1)
    //  total momentum along z (species 1)
    //  ...
    //  total momentum along x (species n)
    //  total momentum along y (species n)
    //  total momentum along z (species n)
    //  mean  momentum along x (all species)
    //  mean  momentum along y (all species)
    //  mean  momentum along z (all species)
    //  mean  momentum along x (species 1)
    //  mean  momentum along y (species 1)
    //  mean  momentum along z (species 1)
    //  ...
    //  mean  momentum along x (species n)
    //  mean  momentum along y (species n)
    //  mean  momentum along z (species n)]
}
