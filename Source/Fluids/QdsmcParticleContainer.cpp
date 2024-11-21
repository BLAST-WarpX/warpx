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

QdsmcParticleContainer::QdsmcParticleContainer (AmrCore* amr_core)
    : ParticleContainerPureSoA<QdsmcPIdx::nattribs, 0>(amr_core->GetParGDB())
{
    SetParticleSize();
    InitParticles(0); // only level 0 is used in HybridSolver
}


void
QdsmcParticleContainer::AddNParticles (int lev, amrex::Long n,
                        amrex::Vector<amrex::ParticleReal> const & x,
                        amrex::Vector<amrex::ParticleReal> const & y,
                        amrex::Vector<amrex::ParticleReal> const & z)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(lev == 0, "QdsmcParticleContainer::AddNParticles: only lev=0 is supported yet.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(x.size() == n,"x.size() != # of qdsmc particles to add");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(y.size() == n,"y.size() != # of qdsmc particles to add");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(z.size() == n,"z.size() != # of qdsmc particles to add");

    if (n <= 0){
        Redistribute();
        return;
    }
    // have to resize here, not in the constructor because grids have not
    // been built when constructor was called.
    reserveData();
    resizeData();

    auto& particle_tile = DefineAndReturnParticleTile(0, 0, 0);

    // Creates a temporary tile to obtain data from simulation. This data
    // is then coppied to the permament tile which is stored on the particle
    // (particle_tile).
    using PinnedTile = typename ContainerLike<amrex::PinnedArenaAllocator>::ParticleTileType;

    PinnedTile pinned_tile;
    pinned_tile.define(NumRuntimeRealComps(), NumRuntimeIntComps());

    for (int i = 0; i < n; i++)
    {
        auto & idcpu_data = pinned_tile.GetStructOfArrays().GetIdCPUData();
        idcpu_data.push_back(amrex::SetParticleIDandCPU(ParticleType::NextID(), ParallelDescriptor::MyProc()));
    }

    // write Real attributes (SoA) to particle initialized zero
    DefineAndReturnParticleTile(0, 0, 0);

    // for RZ write theta value
#ifdef WARPX_DIM_RZ
    pinned_tile.push_back_real(QdsmcPIdx::theta, n, 0.0);
#endif
#if !defined (WARPX_DIM_1D_Z)
    pinned_tile.push_back_real(QdsmcPIdx::x, x);
    pinned_tile.push_back_real(QdsmcPIdx::x_node, x);
#endif
#if defined (WARPX_DIM_3D)
    pinned_tile.push_back_real(QdsmcPIdx::y, y);
    pinned_tile.push_back_real(QdsmcPIdx::y_node, y);
#endif
    pinned_tile.push_back_real(QdsmcPIdx::z, z);
    pinned_tile.push_back_real(QdsmcPIdx::z_node, z);
    pinned_tile.push_back_real(QdsmcPIdx::vx, n, 0.0);
    pinned_tile.push_back_real(QdsmcPIdx::vy, n, 0.0);
    pinned_tile.push_back_real(QdsmcPIdx::vz, n, 0.0);
    pinned_tile.push_back_real(QdsmcPIdx::entropy, n, 0.0);
    pinned_tile.push_back_real(QdsmcPIdx::np_real, n, 0.0);

    const auto old_np = particle_tile.numParticles();
    const auto new_np = old_np + pinned_tile.numParticles();
    particle_tile.resize(new_np);
    amrex::copyParticles(
        particle_tile, pinned_tile, 0, old_np, pinned_tile.numParticles());

    Redistribute();

    // Remove particles that are inside the embedded boundaries here
    /**
     *
     *
     *
     */
}


void
QdsmcParticleContainer::InitParticles (int lev)
{
    auto& warpx = WarpX::GetInstance();
    const auto problo = warpx.Geom(lev).ProbLoArray();
    const auto probhi = warpx.Geom(lev).ProbHiArray();

    const amrex::Real* dx = warpx.Geom(lev).CellSize();

    // Read this from domain ?
    int nx = (probhi[0] - problo[0])/dx[0];
    int ny = (probhi[1] - problo[1])/dx[1];
    int nz = (probhi[2] - problo[2])/dx[2];

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
                    // cell centered
                    //amrex::Real x = problo[0] + (i+0.5)*dx[0];
                    //amrex::Real y = problo[1] + (j+0.5)*dx[1];
                    //amrex::Real z = problo[2] + (k+0.5)*dx[2];

                    // nodal
                    amrex::Real x = problo[0] + i*dx[0];
                    amrex::Real y = problo[1] + j*dx[1];
                    amrex::Real z = problo[2] + k*dx[2];

                    if( x <= probhi[0] && y <= probhi[1] && z <= probhi[2])
                    {
                        xpos.push_back(x);
                        ypos.push_back(y);
                        zpos.push_back(z);

                        n_to_add++;
                    }
                }
            }
        }
    }

    AddNParticles (0, n_to_add, xpos, ypos, zpos);
}


void
QdsmcParticleContainer::SetV (int lev,
                    const amrex::MultiFab &Ux,
                    const amrex::MultiFab &Uy,
                    const amrex::MultiFab &Uz)
{
    const amrex::XDim3 dinv = WarpX::InvCellSize(lev);

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        auto const np = pti.numParticles();
        auto& attribs = pti.GetStructOfArrays().GetRealData();

        // making box cell centered
        //amrex::Box box = pti.tilebox();
        //box.grow(Ux.nGrowVect());

        amrex::Box box = pti.tilebox();
        const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, lev, 0._rt);

        amrex::ParticleReal* const AMREX_RESTRICT part_x0 = attribs[QdsmcPIdx::x_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_y0 = attribs[QdsmcPIdx::y_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_z0 = attribs[QdsmcPIdx::z_node].dataPtr();

        amrex::ParticleReal* const AMREX_RESTRICT part_vx = attribs[QdsmcPIdx::vx].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_vy = attribs[QdsmcPIdx::vy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_vz = attribs[QdsmcPIdx::vz].dataPtr();

        const auto &arrUxfield = Ux[pti].array();
        const auto &arrUyfield = Uy[pti].array();
        const auto &arrUzfield = Uz[pti].array();

        // Gather drift velocity directly from nodes
        // since particles are located at the node positions before PushX
        amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
        {
            amrex::Real vxp;
            amrex::Real vyp;
            amrex::Real vzp;

            gather_vector_field_qdsmc(part_x0[ip], part_y0[ip], part_z0[ip], vxp, vyp, vzp, arrUxfield, arrUyfield, arrUzfield, xyzmin, dinv);

            part_vx[ip] = vxp;
            part_vy[ip] = vyp;
            part_vz[ip] = vzp;
        });
    }
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

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        auto const np = pti.numParticles();
        auto& attribs = pti.GetStructOfArrays().GetRealData();

        // making box cell centered
        //amrex::Box box = pti.tilebox();
        //box.grow(rhofield.nGrowVect());

        amrex::Box box = pti.tilebox();
        const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, lev, 0._rt);

        amrex::ParticleReal* const AMREX_RESTRICT part_x0 = attribs[QdsmcPIdx::x_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_y0 = attribs[QdsmcPIdx::y_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_z0 = attribs[QdsmcPIdx::z_node].dataPtr();

        amrex::ParticleReal* const AMREX_RESTRICT part_entropy = attribs[QdsmcPIdx::entropy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT part_np_real = attribs[QdsmcPIdx::np_real].dataPtr();

        const auto &arrKfield = Kfield[pti].array();
        const auto &arrrhofield = rhofield[pti].array();

        // Gather entropy and density directly from nodes
        // since particles are located at the node positions before PushX
        amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
        {
            amrex::Real n_p;
            amrex::Real kn_p;

            gather_density_entropy(part_x0[ip], part_y0[ip], part_z0[ip], n_p, kn_p, arrrhofield, arrKfield, xyzmin, dinv, cell_volume);

            part_np_real[ip] = n_p;
            part_entropy[ip] = kn_p;
        });
    }
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
    // search for maximum part_dx/part_dy/part_dz and assert if larger than dx/dy/dz
    // ...
    // ...

    //WARPX_ALWAYS_ASSERT_WITH_MESSAGE(part_dx_max >= dx[0], "QdsmcParticleContainer::PushX: qdsmc_part_dx >= dx");
    //WARPX_ALWAYS_ASSERT_WITH_MESSAGE(part_dy_max >= dx[1], "QdsmcParticleContainer::PushX: qdsmc_part_dy >= dy");
    //WARPX_ALWAYS_ASSERT_WITH_MESSAGE(part_dz_max >= dx[2], "QdsmcParticleContainer::PushX: qdsmc_part_dz >= dz");
}


// Generalize this function to --> DepositScalar
void
QdsmcParticleContainer::DepositK(int lev, amrex::MultiFab &Kfield)
{
    const amrex::XDim3 dinv = WarpX::InvCellSize(lev);

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        auto const np = pti.numParticles();

        // making box cell centered
        //amrex::Box box = pti.tilebox();
        //box.grow(Kfield.nGrowVect());

        amrex::Box box = pti.tilebox();
        const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, lev, 0._rt);

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
                do_deposit_scalar(arrKField, part_x[ip], part_y[ip], part_z[ip], xyzmin, dinv, part_entropy[ip]);
            }

        });
    }
}
