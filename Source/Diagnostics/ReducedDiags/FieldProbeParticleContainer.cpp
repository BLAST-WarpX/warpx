/* Copyright 2021 Elisa Rheaume, Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldProbeParticleContainer.H"

#include "Particles/Deposition/ChargeDeposition.H"
#include "Particles/Deposition/CurrentDeposition.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/UpdatePosition.H"
#include "Particles/ParticleBoundaries_K.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_AmrCore.H>
#include <AMReX_AmrParGDB.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_Dim3.H>
#include <AMReX_Extension.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuAllocators.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParGDB.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particle.H>
#include <AMReX_ParticleContainerBase.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_ParticleTransformation.H>
#include <AMReX_ParticleUtil.H>
#include <AMReX_Utility.H>


#include <algorithm>
#include <cmath>

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
#ifdef WARPX_DIM_RZ
    pinned_tile.push_back_real(FieldProbePIdx::theta, np, 0.0);
#endif
#if !defined (WARPX_DIM_1D_Z)
    pinned_tile.push_back_real(FieldProbePIdx::x, x);
#endif
#if defined (WARPX_DIM_3D)
    pinned_tile.push_back_real(FieldProbePIdx::y, y);
#endif
    pinned_tile.push_back_real(FieldProbePIdx::z, z);
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
