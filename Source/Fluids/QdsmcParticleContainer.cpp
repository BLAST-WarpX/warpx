/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Marco Acciarri, Prabhat Kumar (Helion Energy Inc.)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "QdsmcParticleContainer.H"
#include "Qdsmc_K.H"

#include "Particles/Pusher/GetAndSetPosition.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/Communication.H>

#include <AMReX.H>
#include <AMReX_AmrCore.H>
#include <AMReX_AmrParGDB.H>
#include <AMReX_Box.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Particle.H>
#include <AMReX_ParticleContainer.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_ParticleUtil.H>
#include <AMReX_Scan.H>
#include <AMReX_Utility.H>

#include <cstdint>

using namespace amrex::literals;


QdsmcParticleContainer::QdsmcParticleContainer (amrex::AmrCore* amr_core)
    : amrex::ParticleContainerPureSoA<QdsmcPIdx::nattribs, 0>(amr_core->GetParGDB())
{
    SetParticleSize();
}


void QdsmcParticleContainer::InitParticles (int lev)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::InitParticles()");

    reserveData();
    resizeData();

    amrex::Geometry const & geom = Geom(lev);
    auto const dx_arr = geom.CellSizeArray();
    auto const plo    = geom.ProbLoArray();

    // Define particle tiles for every (grid, tile) pair on this level.
    for (auto mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
        DefineAndReturnParticleTile(lev, mfi.index(), mfi.LocalTileIndex());
    }

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    amrex::MFItInfo info;
    if (do_tiling && amrex::Gpu::notInLaunchRegion()) {
        info.EnableTiling(tile_size);
    }
#ifdef AMREX_USE_OMP
    info.SetDynamic(true);
#pragma omp parallel if (not WarpX::serialize_initial_conditions)
#endif
    for (amrex::MFIter mfi = MakeMFIter(lev, info); mfi.isValid(); ++mfi)
    {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        amrex::Box const & tile_box = mfi.tilebox();
        int const grid_id = mfi.index();
        int const tile_id = mfi.LocalTileIndex();

        // One particle per cell. Use exclusive scan to assign per-cell offsets
        // so the per-cell writes are race-free in parallel.
        amrex::Gpu::DeviceVector<amrex::Long> counts(tile_box.numPts(), 1);
        amrex::Gpu::DeviceVector<amrex::Long> offset(tile_box.numPts());
        amrex::Long const max_new_particles = amrex::Scan::ExclusiveSum(
            counts.size(), counts.data(), offset.data());

        // Reserve a globally-unique ID range for the new particles.
        amrex::Long pid;
#ifdef AMREX_USE_OMP
#pragma omp critical (qdsmc_init_nextid)
#endif
        {
            pid = ParticleType::NextID();
            ParticleType::NextID(pid + max_new_particles);
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            pid + max_new_particles < amrex::LongParticleIds::LastParticleID,
            "QdsmcParticleContainer::InitParticles: overflow on particle id numbers");

        int const cpuid = amrex::ParallelDescriptor::MyProc();

        auto & particle_tile =
            GetParticles(lev)[std::make_pair(grid_id, tile_id)];

        if ((NumRuntimeRealComps() > 0) || (NumRuntimeIntComps() > 0)) {
            DefineAndReturnParticleTile(lev, grid_id, tile_id);
        }

        auto const old_size = static_cast<amrex::Long>(particle_tile.size());
        auto const new_size = old_size + max_new_particles;
        particle_tile.resize(new_size);

        auto & soa = particle_tile.GetStructOfArrays();

        amrex::GpuArray<amrex::ParticleReal*, QdsmcPIdx::nattribs> pa;
        for (int ia = 0; ia < QdsmcPIdx::nattribs; ++ia) {
            pa[ia] = soa.GetRealData(ia).data() + old_size;
        }
        std::uint64_t * AMREX_RESTRICT pa_idcpu =
            soa.GetIdCPUData().data() + old_size;

        auto * const poffset = offset.data();

        amrex::ParallelFor(tile_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
            long const ip = poffset[tile_box.index(iv)];

            pa_idcpu[ip] = amrex::SetParticleIDandCPU(pid + ip, cpuid);

            // Compute the cell-center position in physical units. The field
            // dimension determines which axis indices are physically meaningful;
            // missing axes are set to 0 on the particle's home record.
#if defined(WARPX_DIM_3D)
            amrex::Real const x_pos = plo[0] + (iv[0] + amrex::Real(0.5)) * dx_arr[0];
            amrex::Real const y_pos = plo[1] + (iv[1] + amrex::Real(0.5)) * dx_arr[1];
            amrex::Real const z_pos = plo[2] + (iv[2] + amrex::Real(0.5)) * dx_arr[2];
            pa[QdsmcPIdx::x][ip] = x_pos;
            pa[QdsmcPIdx::y][ip] = y_pos;
            pa[QdsmcPIdx::z][ip] = z_pos;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            // In 2D Cartesian and RZ the second in-plane coord is z; the y
            // axis is the unused out-of-plane direction.
            amrex::Real const x_pos = plo[0] + (iv[0] + amrex::Real(0.5)) * dx_arr[0];
            amrex::Real const y_pos = amrex::Real(0);
            amrex::Real const z_pos = plo[1] + (iv[1] + amrex::Real(0.5)) * dx_arr[1];
            pa[QdsmcPIdx::x][ip] = x_pos;
            pa[QdsmcPIdx::z][ip] = z_pos;
#elif defined(WARPX_DIM_1D_Z)
            amrex::Real const x_pos = amrex::Real(0);
            amrex::Real const y_pos = amrex::Real(0);
            amrex::Real const z_pos = plo[0] + (iv[0] + amrex::Real(0.5)) * dx_arr[0];
            pa[QdsmcPIdx::z][ip] = z_pos;
#else
            // RCYLINDER / RSPHERE intentionally not supported in this PR.
            amrex::Real const x_pos = amrex::Real(0);
            amrex::Real const y_pos = amrex::Real(0);
            amrex::Real const z_pos = amrex::Real(0);
#endif

            // Home position is always stored as a 3D vector.
            pa[QdsmcPIdx::x_node][ip] = x_pos;
            pa[QdsmcPIdx::y_node][ip] = y_pos;
            pa[QdsmcPIdx::z_node][ip] = z_pos;

            // Velocity, entropy and weight are populated each step by SetV/SetK.
            pa[QdsmcPIdx::vx][ip] = amrex::Real(0);
            pa[QdsmcPIdx::vy][ip] = amrex::Real(0);
            pa[QdsmcPIdx::vz][ip] = amrex::Real(0);
            pa[QdsmcPIdx::entropy][ip] = amrex::Real(0);
            pa[QdsmcPIdx::np_real][ip] = amrex::Real(0);
        });

        amrex::Gpu::synchronize();

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add(&(*cost)[mfi.index()], wt);
        }
    }

    amrex::Gpu::synchronize();
}


void
QdsmcParticleContainer::SetV (int lev,
                              const amrex::MultiFab & Ux,
                              const amrex::MultiFab & Uy,
                              const amrex::MultiFab & Uz)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::SetV()");

    auto & warpx = WarpX::GetInstance();
    auto const plo = warpx.Geom(lev).ProbLoArray();
    auto const dxi = warpx.Geom(lev).InvCellSizeArray();
    auto const ix_type = Ux.ixType().toIntVect();

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        long const np = pti.numParticles();
        auto & attribs = pti.GetStructOfArrays().GetRealData();

        amrex::Box const tilebox = pti.tilebox();
        amrex::Box box = amrex::convert(tilebox, ix_type);
        box.grow(Ux.nGrowVect());

        amrex::ParticleReal* const AMREX_RESTRICT x_node =
            attribs[QdsmcPIdx::x_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT y_node =
            attribs[QdsmcPIdx::y_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT z_node =
            attribs[QdsmcPIdx::z_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vx =
            attribs[QdsmcPIdx::vx].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vy =
            attribs[QdsmcPIdx::vy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vz =
            attribs[QdsmcPIdx::vz].dataPtr();

        auto const ux_arr = Ux[pti].array();
        auto const uy_arr = Uy[pti].array();
        auto const uz_arr = Uz[pti].array();

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
        {
            amrex::Real vxp = 0;
            amrex::Real vyp = 0;
            amrex::Real vzp = 0;

            gather_vector_field_qdsmc(
                x_node[ip], y_node[ip], z_node[ip],
                vxp, vyp, vzp,
                ux_arr, uy_arr, uz_arr,
                plo, dxi, box);

            vx[ip] = vxp;
            vy[ip] = vyp;
            vz[ip] = vzp;
        });
    }

    amrex::Gpu::synchronize();
}


void
QdsmcParticleContainer::SetK (int lev,
                              const amrex::MultiFab & Kfield,
                              const amrex::MultiFab & rhofield)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::SetK()");

    auto & warpx = WarpX::GetInstance();
    auto const plo = warpx.Geom(lev).ProbLoArray();
    auto const dxi = warpx.Geom(lev).InvCellSizeArray();
    auto const * dx_arr = warpx.Geom(lev).CellSize();

    amrex::Real cell_volume = 1.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        cell_volume *= dx_arr[d];
    }

    auto const ix_type = Kfield.ixType().toIntVect();

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        long const np = pti.numParticles();
        auto & attribs = pti.GetStructOfArrays().GetRealData();

        amrex::Box const tilebox = pti.tilebox();
        amrex::Box box = amrex::convert(tilebox, ix_type);
        box.grow(Kfield.nGrowVect());

        amrex::ParticleReal* const AMREX_RESTRICT x_node =
            attribs[QdsmcPIdx::x_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT y_node =
            attribs[QdsmcPIdx::y_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT z_node =
            attribs[QdsmcPIdx::z_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT entropy =
            attribs[QdsmcPIdx::entropy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT np_real =
            attribs[QdsmcPIdx::np_real].dataPtr();

        auto const K_arr   = Kfield.const_array(pti);
        auto const rho_arr = rhofield.const_array(pti);

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
        {
            amrex::Real n_p = 0;
            amrex::Real kn_p = 0;

            gather_density_entropy(
                x_node[ip], y_node[ip], z_node[ip],
                n_p, kn_p,
                rho_arr, K_arr,
                plo, dxi, cell_volume, box);

            np_real[ip] = n_p;
            entropy[ip] = kn_p;
        });
    }

    amrex::Gpu::synchronize();
}


void
QdsmcParticleContainer::PushX (int lev, amrex::Real dt)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::PushX()");

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        long const np = pti.numParticles();
        auto & attribs = pti.GetStructOfArrays().GetRealData();

        amrex::ParticleReal* const AMREX_RESTRICT x_node =
            attribs[QdsmcPIdx::x_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT y_node =
            attribs[QdsmcPIdx::y_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT z_node =
            attribs[QdsmcPIdx::z_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vx =
            attribs[QdsmcPIdx::vx].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vy =
            attribs[QdsmcPIdx::vy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vz =
            attribs[QdsmcPIdx::vz].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT np_real =
            attribs[QdsmcPIdx::np_real].dataPtr();

        // Position attributes (only the AMReX-tracked subset). For
        // dimensions that are not represented in the field (y in 2D,
        // x and y in 1D Z), the position attribute does not exist as
        // an enum value, so the corresponding update is omitted.
#if !defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal* const AMREX_RESTRICT pa_x =
            attribs[QdsmcPIdx::x].dataPtr();
#endif
#if defined(WARPX_DIM_3D)
        amrex::ParticleReal* const AMREX_RESTRICT pa_y =
            attribs[QdsmcPIdx::y].dataPtr();
#endif
        amrex::ParticleReal* const AMREX_RESTRICT pa_z =
            attribs[QdsmcPIdx::z].dataPtr();

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
        {
            // Skip particles with no weight (e.g. just-reset particles
            // before a SetK call). They contribute nothing to the deposit.
            if (np_real[ip] <= amrex::Real(0)) { return; }

            amrex::Real xp_new = 0;
            amrex::Real yp_new = 0;
            amrex::Real zp_new = 0;

            push_qdsmc_particle(
                x_node[ip], y_node[ip], z_node[ip],
                vx[ip], vy[ip], vz[ip],
                xp_new, yp_new, zp_new, dt);

            // Write the new position back to the AMReX-tracked position
            // slots. For axes not represented in the field, there is no
            // corresponding enum slot and the position is simply not
            // tracked by AMReX (consistent with field dimensionality).
#if !defined(WARPX_DIM_1D_Z)
            pa_x[ip] = xp_new;
#endif
#if defined(WARPX_DIM_3D)
            pa_y[ip] = yp_new;
#endif
            pa_z[ip] = zp_new;
        });
    }

    Redistribute();
    amrex::Gpu::synchronize();
}


void
QdsmcParticleContainer::ResetParticles (int lev)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::ResetParticles()");

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        long const np = pti.numParticles();
        auto & attribs = pti.GetStructOfArrays().GetRealData();

        amrex::ParticleReal* const AMREX_RESTRICT x_node =
            attribs[QdsmcPIdx::x_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT y_node =
            attribs[QdsmcPIdx::y_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT z_node =
            attribs[QdsmcPIdx::z_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vx =
            attribs[QdsmcPIdx::vx].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vy =
            attribs[QdsmcPIdx::vy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vz =
            attribs[QdsmcPIdx::vz].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT entropy =
            attribs[QdsmcPIdx::entropy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT np_real =
            attribs[QdsmcPIdx::np_real].dataPtr();

#if !defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal* const AMREX_RESTRICT pa_x =
            attribs[QdsmcPIdx::x].dataPtr();
#endif
#if defined(WARPX_DIM_3D)
        amrex::ParticleReal* const AMREX_RESTRICT pa_y =
            attribs[QdsmcPIdx::y].dataPtr();
#endif
        amrex::ParticleReal* const AMREX_RESTRICT pa_z =
            attribs[QdsmcPIdx::z].dataPtr();

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
        {
#if !defined(WARPX_DIM_1D_Z)
            pa_x[ip] = x_node[ip];
#endif
#if defined(WARPX_DIM_3D)
            pa_y[ip] = y_node[ip];
#endif
            pa_z[ip] = z_node[ip];

            vx[ip] = 0;
            vy[ip] = 0;
            vz[ip] = 0;
            entropy[ip] = 0;
            np_real[ip] = 0;
        });
    }

    Redistribute();
    amrex::Gpu::synchronize();
}


void
QdsmcParticleContainer::DepositK (int lev, amrex::MultiFab & Kfield)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::DepositK()");

    auto & warpx = WarpX::GetInstance();
    amrex::Geometry const & geom = warpx.Geom(lev);
    amrex::Periodicity const & period = geom.periodicity();
    auto const plo = geom.ProbLoArray();
    auto const dxi = geom.InvCellSizeArray();

    Kfield.setVal(0);

    auto const ix_type = Kfield.ixType().toIntVect();

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        long const np = pti.numParticles();
        auto & attribs = pti.GetStructOfArrays().GetRealData();

        amrex::Box const tilebox = pti.tilebox();
        amrex::Box box = amrex::convert(tilebox, ix_type);
        box.grow(Kfield.nGrowVect());

        amrex::ParticleReal* const AMREX_RESTRICT entropy =
            attribs[QdsmcPIdx::entropy].dataPtr();

        // The y-home is used in 2D (Cartesian/RZ) as the implicit y=0 coord;
        // the x-home is used in 1D as the implicit x=0 coord. In 3D the
        // full position lives in the AMReX-tracked pa_x/pa_y/pa_z slots, so
        // no home component is needed.
#if defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal* const AMREX_RESTRICT x_node =
            attribs[QdsmcPIdx::x_node].dataPtr();
#endif
#if !defined(WARPX_DIM_3D)
        amrex::ParticleReal* const AMREX_RESTRICT y_node =
            attribs[QdsmcPIdx::y_node].dataPtr();
#endif

        // After PushX the *advected* position lives in the (dim-dependent)
        // position slots. Re-read whichever ones exist; the missing axes
        // remain at their home value (which is 0 for unrepresented dims).
#if !defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal* const AMREX_RESTRICT pa_x =
            attribs[QdsmcPIdx::x].dataPtr();
#endif
#if defined(WARPX_DIM_3D)
        amrex::ParticleReal* const AMREX_RESTRICT pa_y =
            attribs[QdsmcPIdx::y].dataPtr();
#endif
        amrex::ParticleReal* const AMREX_RESTRICT pa_z =
            attribs[QdsmcPIdx::z].dataPtr();

        auto const K_arr = Kfield.array(pti);

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
        {
            if (entropy[ip] <= amrex::Real(0)) { return; }

            // Assemble the 3D scatter position from per-dim slots.
#if defined(WARPX_DIM_3D)
            amrex::Real const xp = pa_x[ip];
            amrex::Real const yp = pa_y[ip];
            amrex::Real const zp = pa_z[ip];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            amrex::Real const xp = pa_x[ip];
            amrex::Real const yp = y_node[ip];
            amrex::Real const zp = pa_z[ip];
#elif defined(WARPX_DIM_1D_Z)
            amrex::Real const xp = x_node[ip];
            amrex::Real const yp = y_node[ip];
            amrex::Real const zp = pa_z[ip];
#else
#   error "QdsmcParticleContainer::DepositK does not support WARPX_DIM_RCYLINDER / WARPX_DIM_RSPHERE yet"
#endif

            do_deposit_scalar(K_arr, xp, yp, zp, plo, dxi, entropy[ip], box);
        });
    }

    amrex::Gpu::synchronize();

    ablastr::utils::communication::SumBoundary(
        Kfield, 0, Kfield.nComp(), Kfield.nGrowVect(), Kfield.nGrowVect(),
        WarpX::do_single_precision_comms, period);
}


void
QdsmcParticleContainer::DepositField (int lev, amrex::MultiFab & Field)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::DepositField()");

    auto & warpx = WarpX::GetInstance();
    amrex::Geometry const & geom = warpx.Geom(lev);
    amrex::Periodicity const & period = geom.periodicity();
    auto const plo = geom.ProbLoArray();
    auto const dxi = geom.InvCellSizeArray();
    auto const * dx_arr = geom.CellSize();

    amrex::Real cell_volume = 1.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        cell_volume *= dx_arr[d];
    }

    Field.setVal(0);

    auto const ix_type = Field.ixType().toIntVect();

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        long const np = pti.numParticles();
        auto & attribs = pti.GetStructOfArrays().GetRealData();

        amrex::Box const tilebox = pti.tilebox();
        amrex::Box box = amrex::convert(tilebox, ix_type);
        box.grow(Field.nGrowVect());

        amrex::ParticleReal* const AMREX_RESTRICT np_real =
            attribs[QdsmcPIdx::np_real].dataPtr();

#if defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal* const AMREX_RESTRICT x_node =
            attribs[QdsmcPIdx::x_node].dataPtr();
#endif
#if !defined(WARPX_DIM_3D)
        amrex::ParticleReal* const AMREX_RESTRICT y_node =
            attribs[QdsmcPIdx::y_node].dataPtr();
#endif

#if !defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal* const AMREX_RESTRICT pa_x =
            attribs[QdsmcPIdx::x].dataPtr();
#endif
#if defined(WARPX_DIM_3D)
        amrex::ParticleReal* const AMREX_RESTRICT pa_y =
            attribs[QdsmcPIdx::y].dataPtr();
#endif
        amrex::ParticleReal* const AMREX_RESTRICT pa_z =
            attribs[QdsmcPIdx::z].dataPtr();

        auto const field_arr = Field.array(pti);

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
        {
            if (np_real[ip] <= amrex::Real(0)) { return; }

#if defined(WARPX_DIM_3D)
            amrex::Real const xp = pa_x[ip];
            amrex::Real const yp = pa_y[ip];
            amrex::Real const zp = pa_z[ip];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            amrex::Real const xp = pa_x[ip];
            amrex::Real const yp = y_node[ip];
            amrex::Real const zp = pa_z[ip];
#elif defined(WARPX_DIM_1D_Z)
            amrex::Real const xp = x_node[ip];
            amrex::Real const yp = y_node[ip];
            amrex::Real const zp = pa_z[ip];
#else
#   error "QdsmcParticleContainer::DepositField does not support WARPX_DIM_RCYLINDER / WARPX_DIM_RSPHERE yet"
#endif

            // np_real already carries the n_e * V_cell weight; divide by V_cell
            // so the scatter accumulates an n_e (density) field.
            amrex::Real const val = np_real[ip] / cell_volume;
            do_deposit_scalar(field_arr, xp, yp, zp, plo, dxi, val, box);
        });
    }

    amrex::Gpu::synchronize();

    ablastr::utils::communication::SumBoundary(
        Field, 0, Field.nComp(), Field.nGrowVect(), Field.nGrowVect(),
        WarpX::do_single_precision_comms, period);
}
