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
FieldProbeParticleContainer::AddNParticles (int lev,
                                            amrex::Vector<amrex::ParticleReal> const & x,
                                            amrex::Vector<amrex::ParticleReal> const & y,
                                            amrex::Vector<amrex::ParticleReal> const & z)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(lev == 0, "AddNParticles: only lev=0 is supported yet.");
    AMREX_ALWAYS_ASSERT(x.size() == y.size());
    AMREX_ALWAYS_ASSERT(x.size() == z.size());

    // number of particles to add
    auto const np = static_cast<int>(x.size());
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
    using PinnedTile = typename ContainerLike<amrex::PinnedArenaAllocator>::ParticleTileType;

    PinnedTile pinned_tile;
    pinned_tile.define(NumRuntimeRealComps(), NumRuntimeIntComps());

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
    pinned_tile.push_back_real(FieldProbePIdx::x, x);
#endif
#if defined (WARPX_DIM_3D)
    pinned_tile.push_back_real(FieldProbePIdx::y, y);
#endif
#if !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    pinned_tile.push_back_real(FieldProbePIdx::z, z);
#endif
    pinned_tile.push_back_real(FieldProbePIdx::Ex, np, 0.0);
    pinned_tile.push_back_real(FieldProbePIdx::Ey, np, 0.0);
    pinned_tile.push_back_real(FieldProbePIdx::Ez, np, 0.0);
    pinned_tile.push_back_real(FieldProbePIdx::Bx, np, 0.0);
    pinned_tile.push_back_real(FieldProbePIdx::By, np, 0.0);
    pinned_tile.push_back_real(FieldProbePIdx::Bz, np, 0.0);
    pinned_tile.push_back_real(FieldProbePIdx::S, np, 0.0);

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
