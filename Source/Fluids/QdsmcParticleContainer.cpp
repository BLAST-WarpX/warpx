/* Copyright 2024 Marco Acciarri (Helion Energy Inc.)
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "QdsmcParticleContainer.H"
#include "Qdsmc_K.H"

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

#include "EmbeddedBoundary/Enabled.H"

#include <ablastr/utils/Communication.H>

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
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

#ifdef AMREX_USE_EB
#   include "EmbeddedBoundary/ParticleBoundaryProcess.H"
#   include "EmbeddedBoundary/ParticleScraper.H"
#endif

#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>

using namespace amrex;

QdsmcParticleContainer::QdsmcParticleContainer (AmrCore* amr_core)
    : ParticleContainerPureSoA<QdsmcPIdx::nattribs, 0>(amr_core->GetParGDB())
{
    SetParticleSize();
}

/*
void QdsmcParticleContainer::AddNParticles (int lev, amrex::Long n,
                        amrex::Vector<amrex::ParticleReal> const & x,
                        amrex::Vector<amrex::ParticleReal> const & y,
                        amrex::Vector<amrex::ParticleReal> const & z)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(lev == 0,
        "QdsmcParticleContainer::AddNParticles: only lev=0 is supported yet.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(x.size() == n,
        "x.size() != # of qdsmc particles to add");

    // In the original version the I/O processor would partition the global array
    // among all MPI ranks. In the multi-GPU, tiled version each MPI rank already has
    // only its own local particles, so we set:
    long ibegin = 0;
    long iend = n;

    // Ensure that the particle container is ready for new data.
    reserveData();
    resizeData();

    // Get the local particle tile. (Here we assume one tile per MPI rank for simplicity.)
    auto& particle_tile = DefineAndReturnParticleTile(0, 0, 0);

    // Create a temporary pinned tile to hold the new particle data.
    using PinnedTile = typename ContainerLike<amrex::PinnedArenaAllocator>::ParticleTileType;
    PinnedTile pinned_tile;
    pinned_tile.define(NumRuntimeRealComps(), NumRuntimeIntComps());

    const std::size_t np = iend - ibegin;

    // Set the IDs and CPU info for the new particles.
    for (auto i = ibegin; i < iend; ++i)
    {
        auto & idcpu_data = pinned_tile.GetStructOfArrays().GetIdCPUData();
        amrex::Long current_id = ParticleType::NextID();
        idcpu_data.push_back(amrex::SetParticleIDandCPU(current_id, ParallelDescriptor::MyProc()));
    }

#if !defined (WARPX_DIM_1D_Z)
    pinned_tile.push_back_real(QdsmcPIdx::x, x.data() + ibegin, x.data() + iend);
    pinned_tile.push_back_real(QdsmcPIdx::x_node, x.data() + ibegin, x.data() + iend);
#endif
#if defined (WARPX_DIM_3D)
    pinned_tile.push_back_real(QdsmcPIdx::y, y.data() + ibegin, y.data() + iend);
    pinned_tile.push_back_real(QdsmcPIdx::y_node, y.data() + ibegin, y.data() + iend);
#endif
    pinned_tile.push_back_real(QdsmcPIdx::z, z.data() + ibegin, z.data() + iend);
    pinned_tile.push_back_real(QdsmcPIdx::z_node, z.data() + ibegin, z.data() + iend);

    // Initialize velocity and other attributes to zero.
    pinned_tile.push_back_real(QdsmcPIdx::vx, np, 0.0_prt);
    pinned_tile.push_back_real(QdsmcPIdx::vy, np, 0.0_prt);
    pinned_tile.push_back_real(QdsmcPIdx::vz, np, 0.0_prt);
    pinned_tile.push_back_real(QdsmcPIdx::entropy, np, 0.0_prt);
    pinned_tile.push_back_real(QdsmcPIdx::np_real, np, 0.0_prt);

    if ((NumRuntimeRealComps() > 0) || (NumRuntimeIntComps() > 0)) {
        DefineAndReturnParticleTile(0, 0, 0);
    }

    pinned_tile.resize(np);

    // Append the new particles to the permanent tile.
    auto old_np = particle_tile.numParticles();
    auto new_np = old_np + pinned_tile.numParticles();
    particle_tile.resize(new_np);
    amrex::copyParticles(
        particle_tile, pinned_tile, 0, old_np, pinned_tile.numParticles()
    );

    Redistribute();
}


void QdsmcParticleContainer::InitParticles_old (int lev)
{
    auto& warpx = WarpX::GetInstance();
    const auto problo = warpx.Geom(lev).ProbLoArray();
    const amrex::Real* dx = warpx.Geom(lev).CellSize();
    // Get the BoxArray and DistributionMapping for the level.
    const amrex::BoxArray& ba = warpx.boxArray(lev);
    const amrex::DistributionMapping& dm = warpx.DistributionMap(lev);

    // Local vectors to hold the fictitious particle coordinates
    amrex::Vector<amrex::ParticleReal> xpos;
    amrex::Vector<amrex::ParticleReal> ypos;
    amrex::Vector<amrex::ParticleReal> zpos;
    int n_to_add = 0;

    // Loop over all boxes (tiles) that belong to this MPI rank.
    for (amrex::MFIter mfi(ba, dm); mfi.isValid(); ++mfi)
    {
        // The valid box for this MPI rank/tile.
        const amrex::Box& bx = mfi.validbox();

        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i){
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j){
                for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
                    amrex::IntVect iv{i, j, k};

                    amrex::ParticleReal x = problo[0] + (iv[0] + 0.5) * dx[0];
                    amrex::ParticleReal y = problo[1] + (iv[1] + 0.5) * dx[1];
                    amrex::ParticleReal z = problo[2] + (iv[2] + 0.5) * dx[2];

                    xpos.push_back(x);
                    ypos.push_back(y);
                    zpos.push_back(z);
                    ++n_to_add;
                }
            }
        }
    }

    AddNParticles(0, n_to_add, xpos, ypos, zpos);
    amrex::Gpu::synchronize();
}
*/
void QdsmcParticleContainer::AllocData()
{
    reserveData();
    resizeData();
}

void QdsmcParticleContainer::InitParticles_2 (int lev)
{
    //WARPX_PROFILE("QdsmcParticleContainer::InitParticles()");

    auto& warpx = WarpX::GetInstance();
    const auto problo = warpx.Geom(lev).ProbLoArray();
    const auto probhi = warpx.Geom(lev).ProbHiArray();
    const auto dx     = warpx.Geom(lev).CellSize();

    // Pointer to tile-based cost data (for load balancing)
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // ---------------------------------------------------------------------
    // 1) Define Particle Tiles for all grids at this level
    // ---------------------------------------------------------------------
    for (auto mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        const int grid_id = mfi.index();
        const int tile_id = mfi.LocalTileIndex();
        DefineAndReturnParticleTile(lev, grid_id, tile_id);
    }

    // ---------------------------------------------------------------------
    // 2) Loop over all tiles again to create new particles
    // ---------------------------------------------------------------------
    MFItInfo info;
    if (do_tiling && amrex::Gpu::notInLaunchRegion()) {
        info.EnableTiling(tile_size);
    }
#ifdef AMREX_USE_OMP
    info.SetDynamic(true);
#pragma omp parallel if (!WarpX::serialize_initial_conditions)
#endif
    for (MFIter mfi = MakeMFIter(lev, info); mfi.isValid(); ++mfi)
    {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        Box domain_box = warpx.Geom(lev).Domain();
        Box tile_box   = mfi.validbox() & domain_box; // intersection
        const RealBox tile_realbox = WarpX::getRealBox(tile_box, lev);

        amrex::AllPrint() << "Rank " << ParallelDescriptor::MyProc()
            << " tile_box: " << tile_box
            << " numPts=" << tile_box.numPts() << std::endl;

        amrex::AllPrint() << "Rank " << ParallelDescriptor::MyProc()
            << " tile_realbox: lo=" << tile_realbox.lo(0) << "," << tile_realbox.lo(1) << "," << tile_realbox.lo(2)
            << " hi=" << tile_realbox.hi(0) << "," << tile_realbox.hi(1) << "," << tile_realbox.hi(2) << std::endl;

        if (tile_box.numPts()==0) {
            continue; // Go to the next tile
        }

        const int grid_id = mfi.index();
        const int tile_id = mfi.LocalTileIndex();

        //Gpu::DeviceVector<amrex::Long> counts(tile_box.numPts(), 0); // original
        Gpu::DeviceVector<amrex::Long> counts(tile_box.numPts(), 1); // for debugging
        Gpu::DeviceVector<amrex::Long> offset(tile_box.numPts());
        auto *pcounts = counts.data();

        amrex::Gpu::synchronize(); // added for debugging

        amrex::AllPrint() << "Rank " << ParallelDescriptor::MyProc()
            << "Before first ParallelFor(commented now) (but with extra synchronize)" << "\n"; 
        
        /*
        amrex::ParallelFor(tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const IntVect iv(AMREX_D_DECL(i, j, k));
            auto index = tile_box.index(iv);

            // Compute the physical coordinates (cell center) for this index.
            amrex::Real x = problo[0] + (iv[0] + 0.5) * dx[0];
            amrex::Real y = problo[1] + (iv[1] + 0.5) * dx[1];
            amrex::Real z = problo[2] + (iv[2] + 0.5) * dx[2];

            // Only create a particle if the computed center is within the domain.
            // (Assuming the physical domain is [problo, probhi).)
            if ( x >= tile_realbox.lo(0) && x < tile_realbox.hi(0) &&
                 y >= tile_realbox.lo(1) && y < tile_realbox.hi(1) &&
                 z >= tile_realbox.lo(2) && z < tile_realbox.hi(2) )
            {
                    pcounts[index] = 1;
            } else {
                    pcounts[index] = 0;
            }

        });
        */
        
        amrex::Gpu::synchronize();

        amrex::AllPrint() << "Rank " << ParallelDescriptor::MyProc()
            << "After first ParallelFor(commented now)" << "\n";  

        const amrex::Long max_new_particles = Scan::ExclusiveSum(counts.size(), counts.data(), offset.data());

        amrex::Gpu::synchronize(); // Added for debugging

        amrex::AllPrint() << "Rank " << ParallelDescriptor::MyProc()
            << "After Scan::ExclusiveSum" << "\n";  

        // Update NextID to include particles created in this function
        amrex::Long pid;
#ifdef AMREX_USE_OMP
#pragma omp critical (qdsmc_nextid)
#endif
        {
            pid = ParticleType::NextID();
            ParticleType::NextID(pid+max_new_particles);
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(pid + max_new_particles < LongParticleIds::LastParticleID,"ERROR: overflow on particle id numbers");

        const int cpuid = ParallelDescriptor::MyProc();

        auto& particle_tile = GetParticles(lev)[std::make_pair(grid_id,tile_id)];

        if ( (NumRuntimeRealComps()>0) || (NumRuntimeIntComps()>0) ) {
            DefineAndReturnParticleTile(lev, grid_id, tile_id);
        }

        auto const old_size = static_cast<amrex::Long>(particle_tile.size());
        auto const new_size = old_size + max_new_particles;
        particle_tile.resize(new_size);

        auto& soa = particle_tile.GetStructOfArrays();

        GpuArray<ParticleReal*,QdsmcPIdx::nattribs> pa;
        for (int ia = 0; ia < QdsmcPIdx::nattribs; ++ia) {
            pa[ia] = soa.GetRealData(ia).data() + old_size;
        }
        uint64_t* AMREX_RESTRICT pa_idcpu = soa.GetIdCPUData().data() + old_size;

        auto *const poffset = offset.data();

        amrex::ParallelFor(tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const IntVect iv = IntVect(AMREX_D_DECL(i, j, k));
            if(tile_box.contains(iv))
            {
                const auto index = tile_box.index(iv);
            
                // Skip if this cell did not count as a valid cell.
                if ( pcounts[index] == 0 ) { return; }
                
                const amrex::Long ip = poffset[index];
                // check that ip is in [0, max_new_particles)
                if (ip < 0 || ip >= max_new_particles) { return; }
                
                const amrex::Real x = problo[0] + (iv[0] + 0.5_rt) * dx[0];
                const amrex::Real y = problo[1] + (iv[1] + 0.5_rt) * dx[1];
                const amrex::Real z = problo[2] + (iv[2] + 0.5_rt) * dx[2];

                
                pa_idcpu[ip] = amrex::SetParticleIDandCPU(pid + ip, cpuid);

                pa[QdsmcPIdx::x][ip] = x;
                pa[QdsmcPIdx::y][ip] = y;
                pa[QdsmcPIdx::z][ip] = z;

                pa[QdsmcPIdx::x_node][ip] = x;
                pa[QdsmcPIdx::y_node][ip] = y;
                pa[QdsmcPIdx::z_node][ip] = z;

                pa[QdsmcPIdx::vx][ip] = 0._rt;
                pa[QdsmcPIdx::vy][ip] = 0._rt;
                pa[QdsmcPIdx::vz][ip] = 0._rt;

                pa[QdsmcPIdx::entropy][ip] = 0._rt;
                pa[QdsmcPIdx::np_real][ip] = 0._rt;
                
            }
        });

        amrex::Gpu::synchronize();

        amrex::AllPrint() << "Rank " << ParallelDescriptor::MyProc()
            << "After second ParallelFor" << "\n";  
        
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }  

    amrex::Gpu::synchronize();

    amrex::AllPrint() << "Rank " << ParallelDescriptor::MyProc()
            << "post_MFIter" << "\n";  

    Redistribute();
}



void QdsmcParticleContainer::InitParticles(int lev){

    WARPX_PROFILE("QdsmcParticleContainer::InitParticles()");

    auto& warpx = WarpX::GetInstance();
    const auto problo = warpx.Geom(lev).ProbLoArray();
    const amrex::Real* dx = warpx.Geom(lev).CellSize();

    // Define all particles tiles
    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        for (auto mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
        {
            const int grid_id = mfi.index();
            const int tile_id = mfi.LocalTileIndex();
            DefineAndReturnParticleTile(lev, grid_id, tile_id);
        }
    }

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    MFItInfo info;
    if (do_tiling && Gpu::notInLaunchRegion()) {
        info.EnableTiling(tile_size);
    }
#ifdef AMREX_USE_OMP
    info.SetDynamic(true);
#pragma omp parallel if (not WarpX::serialize_initial_conditions)
#endif
    for (MFIter mfi = MakeMFIter(lev, info); mfi.isValid(); ++mfi)
    {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        const Box& tile_box = mfi.tilebox();
        //const RealBox tile_realbox = WarpX::getRealBox(tile_box, lev);

        const int grid_id = mfi.index();
        const int tile_id = mfi.LocalTileIndex();

        Gpu::DeviceVector<amrex::Long> counts(tile_box.numPts(), 0);
        Gpu::DeviceVector<amrex::Long> offset(tile_box.numPts());
        auto *pcounts = counts.data();

        amrex::ParallelFor(tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const IntVect iv(AMREX_D_DECL(i, j, k));
            auto index = tile_box.index(iv);
            pcounts[index] = 1;
        });

        const amrex::Long max_new_particles = Scan::ExclusiveSum(counts.size(), counts.data(), offset.data());

        // Update NextID to include particles created in this function
        amrex::Long pid;
#ifdef AMREX_USE_OMP
#pragma omp critical (add_plasma_nextid)
#endif
        {
            pid = ParticleType::NextID();
            ParticleType::NextID(pid+max_new_particles);
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(pid + max_new_particles < LongParticleIds::LastParticleID,"ERROR: overflow on particle id numbers");

        const int cpuid = ParallelDescriptor::MyProc();

        auto& particle_tile = GetParticles(lev)[std::make_pair(grid_id,tile_id)];

        if ( (NumRuntimeRealComps()>0) || (NumRuntimeIntComps()>0) ) {
            DefineAndReturnParticleTile(lev, grid_id, tile_id);
        }

        auto const old_size = static_cast<amrex::Long>(particle_tile.size());
        auto const new_size = old_size + max_new_particles;
        particle_tile.resize(new_size);

        auto& soa = particle_tile.GetStructOfArrays();

        GpuArray<ParticleReal*,QdsmcPIdx::nattribs> pa;
        for (int ia = 0; ia < QdsmcPIdx::nattribs; ++ia) {
            pa[ia] = soa.GetRealData(ia).data() + old_size;
        }
        uint64_t * AMREX_RESTRICT pa_idcpu = soa.GetIdCPUData().data() + old_size;

        auto *const poffset = offset.data();

        amrex::ParallelFor(tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const IntVect iv = IntVect(AMREX_D_DECL(i, j, k));
            const auto index = tile_box.index(iv);

            long ip = poffset[index];
            pa_idcpu[ip] = amrex::SetParticleIDandCPU(pid+ip, cpuid);

            amrex::Real x = problo[0] + (iv[0] + 0.5) * dx[0];
            amrex::Real y = problo[1] + (iv[1] + 0.5) * dx[1];
            amrex::Real z = problo[2] + (iv[2] + 0.5) * dx[2];

            pa[QdsmcPIdx::x][ip] = x;
            pa[QdsmcPIdx::y][ip] = y;
            pa[QdsmcPIdx::z][ip] = z;

            pa[QdsmcPIdx::x_node][ip] = x;
            pa[QdsmcPIdx::y_node][ip] = y;
            pa[QdsmcPIdx::z_node][ip] = z;

            pa[QdsmcPIdx::vx][ip] = 0.0;
            pa[QdsmcPIdx::vy][ip] = 0.0;
            pa[QdsmcPIdx::vz][ip] = 0.0;

            pa[QdsmcPIdx::entropy][ip] = 0.0;
            pa[QdsmcPIdx::np_real][ip] = 0.0;

        });

        amrex::Gpu::synchronize();
        
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }    

    amrex::Gpu::synchronize();
    //Redistribute is not needed anymore?
    //Redistribute();
}


void
QdsmcParticleContainer::SetV (int lev,
                    const amrex::MultiFab &Ux,
                    const amrex::MultiFab &Uy,
                    const amrex::MultiFab &Uz)
{
    WARPX_PROFILE("QdsmcParticleContainer::SetV()");

    const amrex::XDim3 dinv = WarpX::InvCellSize(lev);

    auto& warpx = WarpX::GetInstance();
    const auto plo = warpx.Geom(lev).ProbLoArray();

    const auto ix_type_Uxfield = Ux.ixType().toIntVect();

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        auto const np = pti.numParticles();
        auto& attribs = pti.GetStructOfArrays().GetRealData();

        amrex::Box tilebox = pti.tilebox();
        amrex::Box box = amrex::convert( tilebox, ix_type_Uxfield );
        box.grow(Ux.nGrowVect());

        amrex::ParticleReal* const AMREX_RESTRICT part_x0 = attribs[QdsmcPIdx::x_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_y0 = attribs[QdsmcPIdx::y_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_z0 = attribs[QdsmcPIdx::z_node].dataPtr();

        amrex::ParticleReal* const AMREX_RESTRICT part_vx = attribs[QdsmcPIdx::vx].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_vy = attribs[QdsmcPIdx::vy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_vz = attribs[QdsmcPIdx::vz].dataPtr();

        const auto &arrUxfield = Ux[pti].array();
        const auto &arrUyfield = Uy[pti].array();
        const auto &arrUzfield = Uz[pti].array();

        amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
        {
            amrex::Real vxp = 0;
            amrex::Real vyp = 0;
            amrex::Real vzp = 0;

            gather_vector_field_qdsmc(part_x0[ip], part_y0[ip], part_z0[ip], vxp, vyp, vzp, arrUxfield, arrUyfield, arrUzfield, plo, dinv, box);

            part_vx[ip] = vxp;
            part_vy[ip] = vyp;
            part_vz[ip] = vzp;
        });
    }

    amrex::Gpu::synchronize();
}


void
QdsmcParticleContainer::SetK (int lev,
                const amrex::MultiFab &Kfield,
                const amrex::MultiFab &rhofield)
{
    WARPX_PROFILE("QdsmcParticleContainer::SetK()");

    // get a reference to WarpX instance
    auto & warpx = WarpX::GetInstance();

    const amrex::XDim3 dinv = WarpX::InvCellSize(lev);
    const amrex::Real* dx = warpx.Geom(lev).CellSize();
    amrex::Real cell_volume = dx[0]*dx[1]*dx[2]; // how is this handling different dimensions?

    const auto ix_type_Kfield = Kfield.ixType().toIntVect();

    auto plo = warpx.Geom(lev).ProbLoArray();

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        auto const np = pti.numParticles();
        auto& attribs = pti.GetStructOfArrays().GetRealData();

        amrex::Box tilebox = pti.tilebox();
        amrex::Box box = amrex::convert( tilebox, ix_type_Kfield );
        box.grow(Kfield.nGrowVect());

        amrex::ParticleReal* const AMREX_RESTRICT part_x0 = attribs[QdsmcPIdx::x_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_y0 = attribs[QdsmcPIdx::y_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_z0 = attribs[QdsmcPIdx::z_node].dataPtr();

        amrex::ParticleReal* const AMREX_RESTRICT part_entropy = attribs[QdsmcPIdx::entropy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_np_real = attribs[QdsmcPIdx::np_real].dataPtr();

        const auto &arrKfield = Kfield[pti].array();
        const auto &arrrhofield = rhofield[pti].array();

        amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
        {
            amrex::Real n_p = 0;
            amrex::Real kn_p = 0;

            gather_density_entropy(part_x0[ip], part_y0[ip], part_z0[ip], n_p, kn_p, arrrhofield, arrKfield, plo, dinv, cell_volume, box);

            part_np_real[ip] = n_p;
            part_entropy[ip] = kn_p;
        });
    }

    amrex::Gpu::synchronize();
}


void
QdsmcParticleContainer::PushX (int lev, amrex::Real dt)
{
    WARPX_PROFILE("QdsmcParticleContainer::PushX()");

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        auto const np = pti.numParticles();
        auto& attribs = pti.GetStructOfArrays().GetRealData();

        amrex::ParticleReal* const AMREX_RESTRICT part_x0 = attribs[QdsmcPIdx::x_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_y0 = attribs[QdsmcPIdx::y_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_z0 = attribs[QdsmcPIdx::z_node].dataPtr();

        amrex::ParticleReal* const AMREX_RESTRICT part_x = attribs[QdsmcPIdx::x].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_y = attribs[QdsmcPIdx::y].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_z = attribs[QdsmcPIdx::z].dataPtr();

        amrex::ParticleReal* const AMREX_RESTRICT part_vx = attribs[QdsmcPIdx::vx].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_vy = attribs[QdsmcPIdx::vy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_vz = attribs[QdsmcPIdx::vz].dataPtr();

        amrex::ParticleReal* const AMREX_RESTRICT part_np_real = attribs[QdsmcPIdx::np_real].dataPtr();

        amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
        {
            // avoid launching kernel for "empty" particle
            if(part_np_real[ip]>0)
            {
                amrex::Real xp;
                amrex::Real yp;
                amrex::Real zp;

                push_qdsmc_particle(part_x0[ip], part_y0[ip], part_z0[ip], part_vx[ip], part_vy[ip], part_vz[ip], xp, yp, zp, dt);

                part_x[ip] = xp;
                part_y[ip] = yp;
                part_z[ip] = zp;
            }

        });
    }

    Redistribute();
    // search for maximum part_dx/part_dy/part_dz and assert if larger than dx/dy/dz
    // ...
    // ...

    //WARPX_ALWAYS_ASSERT_WITH_MESSAGE(part_dx_max >= dx[0], "QdsmcParticleContainer::PushX: qdsmc_part_dx >= dx");
    //WARPX_ALWAYS_ASSERT_WITH_MESSAGE(part_dy_max >= dx[1], "QdsmcParticleContainer::PushX: qdsmc_part_dy >= dy");
    //WARPX_ALWAYS_ASSERT_WITH_MESSAGE(part_dz_max >= dx[2], "QdsmcParticleContainer::PushX: qdsmc_part_dz >= dz");
    amrex::Gpu::synchronize();
}


// Do I really need to call this function?
// Test without it, since r0 does not change and r is updated every time qdsmc is called.
void
QdsmcParticleContainer::ResetParticles(int lev)
{
    WARPX_PROFILE("QdsmcParticleContainer::ResetParticles()");

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        auto const np = pti.numParticles();
        auto& attribs = pti.GetStructOfArrays().GetRealData();

        amrex::ParticleReal* const AMREX_RESTRICT part_x0 = attribs[QdsmcPIdx::x_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_y0 = attribs[QdsmcPIdx::y_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_z0 = attribs[QdsmcPIdx::z_node].dataPtr();

        amrex::ParticleReal* const AMREX_RESTRICT part_x = attribs[QdsmcPIdx::x].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_y = attribs[QdsmcPIdx::y].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_z = attribs[QdsmcPIdx::z].dataPtr();

        amrex::ParticleReal* const AMREX_RESTRICT part_vx = attribs[QdsmcPIdx::vx].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_vy = attribs[QdsmcPIdx::vy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_vz = attribs[QdsmcPIdx::vz].dataPtr();

        amrex::ParticleReal* const AMREX_RESTRICT part_entropy = attribs[QdsmcPIdx::entropy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_np_real = attribs[QdsmcPIdx::np_real].dataPtr();

        amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
        {
            part_x[ip] = part_x0[ip];
            part_y[ip] = part_y0[ip];
            part_z[ip] = part_z0[ip];

            part_vx[ip] = 0;
            part_vy[ip] = 0;
            part_vz[ip] = 0;

            part_entropy[ip] = 0;
            part_np_real[ip] = 0;
        });
    }

    Redistribute();
    amrex::Gpu::synchronize();
}


// Generalize this function to --> DepositScalar
void
QdsmcParticleContainer::DepositK(int lev, amrex::MultiFab &Kfield)
{
    WARPX_PROFILE("QdsmcParticleContainer::DepositK()");

    const amrex::XDim3 dinv = WarpX::InvCellSize(lev);

    WarpX &warpx = WarpX::GetInstance();
    const amrex::Geometry &geom = warpx.Geom(lev);
    const amrex::Periodicity &period = geom.periodicity();
    auto plo = geom.ProbLoArray();

    // We need to set the K multifab to 0 before depositing K values from qdsmc particles
    Kfield.setVal(0);

    const auto ix_type = Kfield.ixType().toIntVect();

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        auto const np = pti.numParticles();

        amrex::Box tilebox = pti.tilebox();
        amrex::Box box = amrex::convert( tilebox, ix_type );
        box.grow(Kfield.nGrowVect());

        auto& attribs = pti.GetStructOfArrays().GetRealData();

        amrex::ParticleReal* const AMREX_RESTRICT part_x = attribs[QdsmcPIdx::x].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_y = attribs[QdsmcPIdx::y].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_z = attribs[QdsmcPIdx::z].dataPtr();

        // should change this so that Deposit receives as argument which value to read from the QdsmcPIdx struct
        amrex::ParticleReal* const AMREX_RESTRICT part_entropy = attribs[QdsmcPIdx::entropy].dataPtr();

        // change this to just scalarField
        auto arrKField = Kfield[pti].array();

        // Gather entropy and density directly from nodes
        // since particles are located at the node positions before PushX
        amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
        {
            // avoid launching kernel for "empty" particles
            if(part_entropy[ip]>0)
            {
                do_deposit_scalar(arrKField, part_x[ip], part_y[ip], part_z[ip], plo, dinv, part_entropy[ip], box);
            }
        });
    }

    amrex::Gpu::synchronize();

    ablastr::utils::communication::SumBoundary(
            Kfield, 0, Kfield.nComp(), Kfield.nGrowVect(), Kfield.nGrowVect(),
            WarpX::do_single_precision_comms, period);
}


// Auxiliary function, should generalize the function above
// to deposit a particle property (passed as argument)
// to a multifab passed as argument
void
QdsmcParticleContainer::DepositField(int lev, amrex::MultiFab &Field)
{
     WARPX_PROFILE("QdsmcParticleContainer::DepositField()");

    const amrex::XDim3 dinv = WarpX::InvCellSize(lev);

    WarpX &warpx = WarpX::GetInstance();
    const amrex::Geometry &geom = warpx.Geom(lev);
    const amrex::Periodicity &period = geom.periodicity();
    auto plo = geom.ProbLoArray();

    const amrex::Real* dx = warpx.Geom(lev).CellSize();

    Field.setVal(0);

    const auto ix_type = Field.ixType().toIntVect();

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        auto const np = pti.numParticles();

        amrex::Box tilebox = pti.tilebox();
        amrex::Box box = amrex::convert( tilebox, ix_type );
        box.grow(Field.nGrowVect());

        auto& attribs = pti.GetStructOfArrays().GetRealData();

        amrex::ParticleReal* const AMREX_RESTRICT part_x = attribs[QdsmcPIdx::x].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_y = attribs[QdsmcPIdx::y].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_z = attribs[QdsmcPIdx::z].dataPtr();

        // should change this so that Deposit receives as argument which value to read from the QdsmcPIdx struct
        amrex::ParticleReal* const AMREX_RESTRICT part_np_real = attribs[QdsmcPIdx::np_real].dataPtr();

        // change this to just scalarField
        auto arrField = Field[pti].array();

        // Gather entropy and density directly from nodes
        // since particles are located at the node positions before PushX
        amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
        {
            // avoid launching kernel for "empty" particles
            if(part_np_real[ip]>0)
            {
                amrex::Real val = part_np_real[ip]/(dx[0]*dx[1]*dx[2]);
                do_deposit_scalar(arrField, part_x[ip], part_y[ip], part_z[ip], plo, dinv, val, box);
            }
        });
    }

    amrex::Gpu::synchronize();

    ablastr::utils::communication::SumBoundary(
            Field, 0, Field.nComp(), Field.nGrowVect(), Field.nGrowVect(),
            WarpX::do_single_precision_comms, period);

}
