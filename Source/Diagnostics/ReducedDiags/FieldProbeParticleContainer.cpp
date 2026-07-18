/* Copyright 2021 Elisa Rheaume, Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldProbeParticleContainer.H"

#include "Utils/TextMsg.H"

#include <AMReX_AmrCore.H>
#include <AMReX_AmrParGDB.H>
#include <AMReX_BLassert.H>
#include <AMReX_GpuAllocators.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Particle.H>
#include <AMReX_ParticleContainer.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_ParticleTransformation.H>
#include <AMReX_StructOfArrays.H>

#include <string>


using namespace amrex;

FieldProbeParticleContainer::FieldProbeParticleContainer (AmrCore* amr_core)
    : ParticleContainerPureSoA<FieldProbePIdx::nattribs, 0>(amr_core->GetParGDB())
{
    SetParticleSize();
}

void
FieldProbeParticleContainer::AddNParticles (
    int lev, amrex::Vector<amrex::ParticleReal> const& probe_x,
    amrex::Vector<amrex::ParticleReal> const& probe_y,
    amrex::Vector<amrex::ParticleReal> const& probe_z,
    amrex::Vector<amrex::ParticleReal> const& storage_x,
    amrex::Vector<amrex::ParticleReal> const& storage_y,
    amrex::Vector<amrex::ParticleReal> const& storage_z)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(lev == 0, "AddNParticles: only lev=0 is supported yet.");
    AMREX_ALWAYS_ASSERT(probe_x.size() == probe_y.size());
    AMREX_ALWAYS_ASSERT(probe_x.size() == probe_z.size());
    AMREX_ALWAYS_ASSERT(probe_x.size() == storage_x.size());
    AMREX_ALWAYS_ASSERT(probe_x.size() == storage_y.size());
    AMREX_ALWAYS_ASSERT(probe_x.size() == storage_z.size());

    // number of particles to add
    auto const np = static_cast<int>(probe_x.size());
    if (np <= 0){
        Redistribute();
        return;
    }

    // have to resize here, not in the constructor because grids have not
    // been built when constructor was called.
    reserveData();
    resizeData();

    auto& particle_tile = DefineAndReturnParticleTile(0, 0, 0);

    /*
     * Creates a temporary tile to obtain data from simulation. This data
     * is then coppied to the permament tile which is stored on the particle
     * (particle_tile).
     */
    using PinnedTile = typename ContainerLike<amrex::PolymorphicArenaAllocator>::ParticleTileType;

    PinnedTile pinned_tile;
    auto soa_rdata_names = GetRealSoANames();
    auto soa_idata_names = GetIntSoANames();
    pinned_tile.define(NumRuntimeRealComps(), NumRuntimeIntComps(), &soa_rdata_names, &soa_idata_names, amrex::The_Pinned_Arena());

    for (int i = 0; i < np; i++)
    {
        auto & idcpu_data = pinned_tile.GetStructOfArrays().GetIdCPUData();
        idcpu_data.push_back(amrex::SetParticleIDandCPU(ParticleType::NextID(), ParallelDescriptor::MyProc()));
    }

    // write Real attributes (SoA) to particle initialized zero
    DefineAndReturnParticleTile(0, 0, 0);

    // for RZ write theta value
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    pinned_tile.push_back_real(FieldProbePIdx::theta, np, 0.0);
#endif
#if defined(WARPX_DIM_RSPHERE)
    pinned_tile.push_back_real(FieldProbePIdx::phi, np, 0.0);
#endif
#if !defined (WARPX_DIM_1D_Z)
    pinned_tile.push_back_real(FieldProbePIdx::x, storage_x);
#endif
#if defined (WARPX_DIM_3D)
    pinned_tile.push_back_real(FieldProbePIdx::y, storage_y);
#endif
#if !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    pinned_tile.push_back_real(FieldProbePIdx::z, storage_z);
#endif
    pinned_tile.push_back_real(FieldProbePIdx::Ex, np, 0.0);
    pinned_tile.push_back_real(FieldProbePIdx::Ey, np, 0.0);
    pinned_tile.push_back_real(FieldProbePIdx::Ez, np, 0.0);
    pinned_tile.push_back_real(FieldProbePIdx::Bx, np, 0.0);
    pinned_tile.push_back_real(FieldProbePIdx::By, np, 0.0);
    pinned_tile.push_back_real(FieldProbePIdx::Bz, np, 0.0);
    pinned_tile.push_back_real(FieldProbePIdx::S, np, 0.0);
    pinned_tile.push_back_real(FieldProbePIdx::probe_x, probe_x);
    pinned_tile.push_back_real(FieldProbePIdx::probe_y, probe_y);
    pinned_tile.push_back_real(FieldProbePIdx::probe_z, probe_z);

    const auto old_np = particle_tile.numParticles();
    const auto new_np = old_np + pinned_tile.numParticles();
    particle_tile.resize(new_np);
    amrex::copyParticles(
        particle_tile, pinned_tile, 0, old_np, pinned_tile.numParticles());

    /*
     * Redistributes particles to their appropriate tiles if the box
     * structure of the simulation changes to accommodate data more
     * efficiently.
     */
    Redistribute();
}
