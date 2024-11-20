/* Copyright 2019-2020 Andrew Myers, Axel Huebl,
 * Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_SMART_UTILS_H_
#define WARPX_SMART_UTILS_H_

#include "DefaultInitialization.H"

#include <AMReX_Config.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_INT.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Particle.H>

#include <map>
#include <string>
#include <vector>

using NameMap = std::map<std::string, int>;
using PolicyVec = amrex::Gpu::DeviceVector<InitializationPolicy>;

struct SmartCopyTag
{
    std::vector<std::string> common_names;
    amrex::Gpu::DeviceVector<int> src_comps;
    amrex::Gpu::DeviceVector<int> dst_comps;

    [[nodiscard]] int size () const noexcept { return static_cast<int>(common_names.size()); }
};

PolicyVec getPolicies (const NameMap& names) noexcept;

SmartCopyTag getSmartCopyTag (const NameMap& src, const NameMap& dst) noexcept;

/**
 * \brief Sets the ids of newly created particles to the next values.
 *
 * \tparam PTile the particle tile type
 *
 * \param ptile the particle tile
 * \param old_size the index of the first new particle
 * \param num_added the number of particles to set the ids for.
 */
template <typename PTile>
void setNewParticleIDs (PTile& ptile, amrex::Long old_size, amrex::Long num_added)
{
    amrex::Long pid;
#ifdef AMREX_USE_OMP
#pragma omp critical (ionization_nextid)
#endif
    {
        pid = PTile::ParticleType::NextID();
        PTile::ParticleType::NextID(pid + num_added);
    }

    const int cpuid = amrex::ParallelDescriptor::MyProc();
    auto ptd = ptile.getParticleTileData();
    amrex::ParallelFor(num_added, [=] AMREX_GPU_DEVICE (int ip) noexcept
    {
        auto const lip = static_cast<amrex::Long>(ip);
        auto const new_id = lip + old_size;
        ptd.m_idcpu[new_id] = amrex::SetParticleIDandCPU(pid+lip, cpuid);
    });
}

#endif //WARPX_SMART_UTILS_H_
