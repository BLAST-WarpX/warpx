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



void
QdsmcParticleContainer::AddNParticles (int lev, amrex::Long n,
                        amrex::Vector<amrex::ParticleReal> const & x,
                        amrex::Vector<amrex::ParticleReal> const & y,
                        amrex::Vector<amrex::ParticleReal> const & z,
                        const int grid_id,
                        const int tile_id)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(lev == 0, "QdsmcParticleContainer::AddNParticles: only lev=0 is supported yet.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(x.size() == n,"x.size() != # of qdsmc particles to add");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(y.size() == n,"y.size() != # of qdsmc particles to add");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(z.size() == n,"z.size() != # of qdsmc particles to add");

    // have to resize here, not in the constructor because grids have not
    // been built when constructor was called.
    reserveData();
    resizeData();

    long ibegin = 0;
    long iend = n;

    const int myproc = amrex::ParallelDescriptor::MyProc();
    const int nprocs = amrex::ParallelDescriptor::NProcs();
    const auto navg = n/nprocs;
    const auto nleft = n - navg * nprocs;
    if (myproc < nleft) {
        ibegin = myproc*(navg+1);
        iend = ibegin + navg+1;
    } else {
        ibegin = myproc*navg + nleft;
        iend = ibegin + navg;
    }

    if (n <= 0){
        Redistribute();
        return;
    }

    auto& particle_tile = DefineAndReturnParticleTile(0, 0, 0);

    // Creates a temporary tile to obtain data from simulation. This data
    // is then coppied to the permament tile which is stored on the particle
    // (particle_tile).
    using PinnedTile = typename ContainerLike<amrex::PinnedArenaAllocator>::ParticleTileType;
    PinnedTile pinned_tile;
    pinned_tile.define(NumRuntimeRealComps(), NumRuntimeIntComps());

    const std::size_t np = iend-ibegin;

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
    pinned_tile.push_back_real(QdsmcPIdx::vx, np, 0.0_prt);
    pinned_tile.push_back_real(QdsmcPIdx::vy, np, 0.0_prt);
    pinned_tile.push_back_real(QdsmcPIdx::vz, np, 0.0_prt);
    pinned_tile.push_back_real(QdsmcPIdx::entropy, np, 0.0_prt);
    pinned_tile.push_back_real(QdsmcPIdx::np_real, np, 0.0_prt);

    if ( (NumRuntimeRealComps()>0) || (NumRuntimeIntComps()>0) ){
            DefineAndReturnParticleTile(0, 0, 0);
        }

    pinned_tile.resize(np);

    auto old_np = particle_tile.numParticles();
    auto new_np = old_np + pinned_tile.numParticles();

    particle_tile.resize(new_np);
    amrex::copyParticles(
            particle_tile, pinned_tile, 0, old_np, pinned_tile.numParticles()
        );

    Redistribute();
}


/*
// first implementation, works well on 1 gpu
void
QdsmcParticleContainer::InitParticles (int lev)
{
    auto& warpx = WarpX::GetInstance();
    const auto problo = warpx.Geom(lev).ProbLoArray();
    const auto probhi = warpx.Geom(lev).ProbHiArray();

    const amrex::Real* dx = warpx.Geom(lev).CellSize();

    // if is periodic on dimension d, then substract 1 from nd (nx,ny,nz)
    const auto periodic = warpx.Geom(lev).isPeriodic(); // 1 if periodic, 0 if not

    // Number of cells in each direction
    int nx = (probhi[0] - problo[0])/dx[0] - periodic[0];
    int ny = (probhi[1] - problo[1])/dx[1] - periodic[1];
    int nz = (probhi[2] - problo[2])/dx[2] - periodic[2];

    int n_to_add = 0;

    // create 1D vector for X, Y, and Z coordinates of fictitious particles
    amrex::Vector<amrex::ParticleReal> xpos;
    amrex::Vector<amrex::ParticleReal> ypos;
    amrex::Vector<amrex::ParticleReal> zpos;

    // for now, only one MPI rank adds fictitious particles
    // this is done only once in an entire simulation
    if (ParallelDescriptor::IOProcessor())
    {
        for ( int i = 0; i <= nx; i++)
        {
            for ( int j = 0; j <= ny; j++)
            {
                for ( int k = 0; k <= nz; k++)
                {
                    amrex::ParticleReal x = problo[0] + (i+0.5)*dx[0];
                    amrex::ParticleReal y = problo[1] + (j+0.5)*dx[1];
                    amrex::ParticleReal z = problo[2] + (k+0.5)*dx[2];

                    xpos.push_back(x);
                    ypos.push_back(y);
                    zpos.push_back(z);

                    n_to_add++;
                }
            }
        }
    }
    AddNParticles(0, n_to_add, xpos, ypos, zpos);

    amrex::Gpu::synchronize();
}
*/

// new version, divide initialization in z pos using different ranks, then Redistribute
/*
void QdsmcParticleContainer::InitParticles(int lev)
{
    auto& warpx = WarpX::GetInstance();

    const auto problo = warpx.Geom(lev).ProbLoArray();
    const auto probhi = warpx.Geom(lev).ProbHiArray();
    const amrex::Real* dx = warpx.Geom(lev).CellSize();
    const auto periodic = warpx.Geom(lev).isPeriodic();

    int nx = (probhi[0] - problo[0])/dx[0] - periodic[0];
    int ny = (probhi[1] - problo[1])/dx[1] - periodic[1];
    int nz = (probhi[2] - problo[2])/dx[2] - periodic[2];

    int n_to_add_local = 0;

    amrex::Vector<amrex::ParticleReal> xpos;
    amrex::Vector<amrex::ParticleReal> ypos;
    amrex::Vector<amrex::ParticleReal> zpos;

    const int myproc = ParallelDescriptor::MyProc();
    const int nprocs = ParallelDescriptor::NProcs();

    int nz_per_proc = (nz + 1) / nprocs;
    int z_start = myproc * nz_per_proc;
    int z_end = (myproc == nprocs - 1) ? (nz + 1) : z_start + nz_per_proc;

    for (int i = 0; i <= nx; i++){
        for (int j = 0; j <= ny; j++){
            for (int k = z_start; k < z_end; k++){
                amrex::ParticleReal x = problo[0] + (i+0.5)*dx[0];
                amrex::ParticleReal y = problo[1] + (j+0.5)*dx[1];
                amrex::ParticleReal z = problo[2] + (k+0.5)*dx[2];

                xpos.push_back(x);
                ypos.push_back(y);
                zpos.push_back(z);

                n_to_add_local++;
            }
        }
    }

    AddNParticles(0, n_to_add_local, xpos, ypos, zpos);
    amrex::Gpu::synchronize();
}
*/


void QdsmcParticleContainer::InitParticles(int lev)
{
    auto& warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom(lev);
    const auto problo = geom.ProbLoArray();
    const amrex::Real* dx = geom.CellSize();
    const auto& ba = warpx.boxArray(lev); // Grid boxes
    const auto& dm = warpx.DistributionMap(lev); // Rank ownership

    for (amrex::MFIter mfi(ba, dm); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox(); // Owned cells
        const int grid_id = mfi.index();
        const int tile_id = mfi.LocalTileIndex();

        amrex::Vector<amrex::ParticleReal> xpos, ypos, zpos;
        int n_to_add_local = 0;

        for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i){
            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j){
                for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k){

                    amrex::ParticleReal x = problo[0] + (i+0.5)*dx[0];
                    amrex::ParticleReal y = problo[1] + (j+0.5)*dx[1];
                    amrex::ParticleReal z = problo[2] + (k+0.5)*dx[2];
                    xpos.push_back(x); ypos.push_back(y); zpos.push_back(z);
                    n_to_add_local++;
                }
            }
        }

        AddNParticles(0, n_to_add_local, xpos, ypos, zpos, grid_id, tile_id);
    }
    amrex::Gpu::synchronize();
    Redistribute();
}


/*
void QdsmcParticleContainer::InitParticles(int lev)
{
    auto& warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom(lev);
    const auto problo = geom.ProbLoArray();
    const amrex::Real* dx = geom.CellSize();
    const auto& ba = warpx.boxArray(lev);       // Grid boxes
    const auto& dm = warpx.DistributionMap(lev); // Rank ownership

    // Pre-allocate particle memory
    reserveData();
    resizeData();

    // Iterate over tiles owned by this rank
    for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox(); // Cells this rank owns
        auto& particle_tile = DefineAndReturnParticleTile(lev, mfi); // Direct to tile

        int np_old = particle_tile.numParticles();
        int np_add = box.numPts(); // One particle per cell
        int np_new = np_old + np_add;

        particle_tile.resize(np_new);
        auto& soa = particle_tile.GetStructOfArrays();

        amrex::ParticleReal* x_data = soa.GetRealData(QdsmcPIdx::x).data();
        amrex::ParticleReal* y_data = soa.GetRealData(QdsmcPIdx::y).data();
        amrex::ParticleReal* z_data = soa.GetRealData(QdsmcPIdx::z).data();
        amrex::ParticleReal* x_node_data = soa.GetRealData(QdsmcPIdx::x_node).data();
        amrex::ParticleReal* y_node_data = soa.GetRealData(QdsmcPIdx::y_node).data();
        amrex::ParticleReal* z_node_data = soa.GetRealData(QdsmcPIdx::z_node).data();
        amrex::ParticleReal* vx_data = soa.GetRealData(QdsmcPIdx::vx).data();
        amrex::ParticleReal* vy_data = soa.GetRealData(QdsmcPIdx::vy).data();
        amrex::ParticleReal* vz_data = soa.GetRealData(QdsmcPIdx::vz).data();
        amrex::ParticleReal* entropy_data = soa.GetRealData(QdsmcPIdx::entropy).data();
        amrex::ParticleReal* np_real_data = soa.GetRealData(QdsmcPIdx::np_real).data();

        int idx = np_old;

        for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i){
            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j){
                for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k){
                    x_data[idx] = problo[0] + (i+0.5)*dx[0];
                    y_data[idx] = problo[1] + (j+0.5)*dx[1];
                    z_data[idx] = problo[2] + (k+0.5)*dx[2];

                    x_node_data[idx] = x_data[idx];
                    y_node_data[idx] = y_data[idx];
                    z_node_data[idx] = z_data[idx];

                    vx_data[idx] = 0.0;
                    vy_data[idx] = 0.0;
                    vz_data[idx] = 0.0;

                    entropy_data[idx] = 0.0;
                    np_real_data[idx] = 0.0;

                    idx++;
                }
            }
        }
    }
    amrex::Gpu::synchronize();
    Redistribute(); // this call might not be necessary
}
*/


/*
void QdsmcParticleContainer::InitParticles(int lev)
{
    // complete using example from EMParticleContainer:
    BL_PROFILE("QDSMCParticleContainer::InitParticles");

    auto& warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom(lev);
    const auto plo = geom.ProbLoArray();
    const auto phi = geom.ProbHiArray();
    const amrex::Real* dx = geom.CellSize();
    const auto& ba = warpx.boxArray(lev); // Grid boxes
    const auto& dm = warpx.DistributionMap(lev); // Rank ownership
    const auto periodic = warpx.Geom(lev).isPeriodic();

    for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        const Box& tile_box  = mfi.tilebox();

        const auto lo = amrex::lbound(tile_box);
        const auto hi = amrex::ubound(tile_box);

        Gpu::ManagedVector<unsigned int> counts(tile_box.numPts(), 0);
        unsigned int* pcount = counts.dataPtr();

        Gpu::ManagedVector<unsigned int> offsets(tile_box.numPts());
        unsigned int* poffset = offsets.dataPtr();

        amrex::ParallelFor(tile_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // fix in case is periodic in a specific direction
            // is this adding more particles than needed?

            int ix = i - lo.x;
            int iy = j - lo.y;
            int iz = k - lo.z;
            int nx = hi.x-lo.x+1;
            int ny = hi.y-lo.y+1;
            int nz = hi.z-lo.z+1;
            unsigned int uix = amrex::min(nx-1,amrex::max(0,ix));
            unsigned int uiy = amrex::min(ny-1,amrex::max(0,iy));
            unsigned int uiz = amrex::min(nz-1,amrex::max(0,iz));
            unsigned int cellid = (uix * ny + uiy) * nz + uiz;
            pcount[cellid] += 1;

        });

        Gpu::exclusive_scan(counts.begin(), counts.end(), offsets.begin());

        int num_to_add = offsets[tile_box.numPts()-1] + counts[tile_box.numPts()-1];

        auto& particles = GetParticles(lev);
        auto& particle_tile = particles[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

        auto old_size = particle_tile.numParticles();
        auto new_size = old_size + num_to_add;
        particle_tile.resize(new_size);


        if (num_to_add == 0) continue;

        auto arrdata = particle_tile.GetStructOfArrays().GetRealData();

        int procID = ParallelDescriptor::MyProc();

        amrex::ParallelForRNG(tile_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {

            int ix = i - lo.x;
            int iy = j - lo.y;
            int iz = k - lo.z;
            int nx = hi.x-lo.x+1;
            int ny = hi.y-lo.y+1;
            int nz = hi.z-lo.z+1;
            unsigned int uix = amrex::min(nx-1,amrex::max(0,ix));
            unsigned int uiy = amrex::min(ny-1,amrex::max(0,iy));
            unsigned int uiz = amrex::min(nz-1,amrex::max(0,iz));
            unsigned int cellid = (uix * ny + uiy) * nz + uiz;

            int pidx = poffset[cellid] - poffset[0];

            amrex::Real x = plo[0] + (i + 0.5)*dx[0];
            amrex::Real y = plo[1] + (j + 0.5)*dx[1];
            amrex::Real z = plo[2] + (k + 0.5)*dx[2];


            arrdata[QdsmcPIdx::x][pidx] = x;
            arrdata[QdsmcPIdx::y][pidx] = y;
            arrdata[QdsmcPIdx::z][pidx] = z;

            arrdata[QdsmcPIdx::x_node][pidx] = x;
            arrdata[QdsmcPIdx::y_node][pidx] = y;
            arrdata[QdsmcPIdx::z_node][pidx] = z;

            arrdata[QdsmcPIdx::vx][pidx] = 0;
            arrdata[QdsmcPIdx::vy][pidx] = 0;
            arrdata[QdsmcPIdx::vz][pidx] = 0;

            arrdata[QdsmcPIdx::entropy][pidx] = 0;
            arrdata[QdsmcPIdx::np_real][pidx] = 0;


        });

    }

    amrex::Gpu::synchronize();
    Redistribute();
}
*/

/*
void QdsmcParticleContainer::InitParticles(int lev)
{
    // complete using example from EMParticleContainer:
    BL_PROFILE("QDSMCParticleContainer::InitParticles");

    auto& warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom(lev);
    const auto plo = geom.ProbLoArray();
    const auto phi = geom.ProbHiArray();
    const amrex::Real* dx = geom.CellSize();
    const auto& ba = warpx.boxArray(lev); // Grid boxes
    const auto& dm = warpx.DistributionMap(lev); // Rank ownership
    const auto periodic = warpx.Geom(lev).isPeriodic();

    for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        auto& particles = GetParticles(lev)[std::make_pair(mfi.index(),mfi.LocalTileIndex())];

        //std::array<amrex::Real, NReal> real_attribs;


    }


    amrex::Gpu::synchronize();
    Redistribute();
}
*/


void
QdsmcParticleContainer::SetV (int lev,
                    const amrex::MultiFab &Ux,
                    const amrex::MultiFab &Uy,
                    const amrex::MultiFab &Uz)
{
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
