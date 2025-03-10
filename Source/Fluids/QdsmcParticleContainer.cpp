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


void QdsmcParticleContainer::InitParticles(int lev){

    WARPX_PROFILE("QdsmcParticleContainer::InitParticles()");

    reserveData();
    resizeData();

    const Geometry& geom = Geom(lev);
    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();

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
            const IntVect iv(AMREX_D_DECL(i, j, k));
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

    const Geometry& geom = Geom(lev);
    const amrex::Periodicity &period = geom.periodicity();

    const auto plo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    const auto cell_volume = dx[0]*dx[1]*dx[2];

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
                amrex::Real val = part_np_real[ip]/cell_volume;
                do_deposit_scalar(arrField, part_x[ip], part_y[ip], part_z[ip], plo, dinv, val, box);
            }
        });
    }

    amrex::Gpu::synchronize();

    ablastr::utils::communication::SumBoundary(
            Field, 0, Field.nComp(), Field.nGrowVect(), Field.nGrowVect(),
            WarpX::do_single_precision_comms, period);

}
