/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Marco Acciarri, Prabhat Kumar (Helion Energy Inc.)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "QdsmcParticleContainer.H"

#include "Particles/Deposition/ChargeDeposition.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/particles/NodalFieldGather.H>
#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/Communication.H>

#include <AMReX.H>
#include <AMReX_Algorithm.H>
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

// The QDSMC grid fields (K_e, the deposited weights and the nodal v_e) are
// stored with NODAL staggering; every gather and scatter below uses the
// matching order-1 (linear) nodal weights, so a marker at rest reproduces
// its cell values exactly.


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
            amrex::ignore_unused(j, k);  // unused below AMREX_SPACEDIM
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
            auto const y_pos = amrex::Real(0);
            amrex::Real const z_pos = plo[1] + (iv[1] + amrex::Real(0.5)) * dx_arr[1];
            pa[QdsmcPIdx::x][ip] = x_pos;
            pa[QdsmcPIdx::z][ip] = z_pos;
#elif defined(WARPX_DIM_1D_Z)
            auto const x_pos = amrex::Real(0);
            auto const y_pos = amrex::Real(0);
            amrex::Real const z_pos = plo[0] + (iv[0] + amrex::Real(0.5)) * dx_arr[0];
            pa[QdsmcPIdx::z][ip] = z_pos;
#else
            // WARPX_DIM_RCYLINDER / WARPX_DIM_RSPHERE: 1D radial; the single
            // AMReX-tracked position slot is x (= r). QDSMC is not validated
            // in these geometries (no radial volume weighting yet) and
            // HybridPICModel::ReadParameters refuses to enable the energy
            // equation there -- this branch only needs to compile and be sane.
            amrex::Real const x_pos = plo[0] + (iv[0] + amrex::Real(0.5)) * dx_arr[0];
            auto const y_pos = amrex::Real(0);
            auto const z_pos = amrex::Real(0);
            pa[QdsmcPIdx::x][ip] = x_pos;
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

        auto const ux_arr = Ux.const_array(pti);
        auto const uy_arr = Uy.const_array(pti);
        auto const uz_arr = Uz.const_array(pti);

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
        {
            // Linear gather of the nodal field at the marker's home position.
            auto const v = ablastr::particles::doGatherVectorFieldNodal(
                x_node[ip], y_node[ip], z_node[ip],
                ux_arr, uy_arr, uz_arr, dxi, plo);

            vx[ip] = v[0];
            vy[ip] = v[1];
            vz[ip] = v[2];
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
        amrex::ParticleReal* const AMREX_RESTRICT entropy =
            attribs[QdsmcPIdx::entropy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT np_real =
            attribs[QdsmcPIdx::np_real].dataPtr();

        auto const K_arr   = Kfield.const_array(pti);
        auto const rho_arr = rhofield.const_array(pti);

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
        {
            // Linear gathers of the nodal charge density and entropy at the
            // marker's home position; the marker then carries the electron
            // count N of its cell and the matching entropy content K*N.
            amrex::Real const n_p = ablastr::particles::doGatherScalarFieldNodal(
                x_node[ip], y_node[ip], z_node[ip], rho_arr, dxi, plo)
                * cell_volume / PhysConst::q_e;
            amrex::Real const k_p = ablastr::particles::doGatherScalarFieldNodal(
                x_node[ip], y_node[ip], z_node[ip], K_arr, dxi, plo);

            np_real[ip] = n_p;
            entropy[ip] = k_p * n_p;
        });
    }

    amrex::Gpu::synchronize();
}


void
QdsmcParticleContainer::PushX (int lev, amrex::Real dt)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::PushX()");

    amrex::Geometry const & geom = Geom(lev);
    auto const plo = geom.ProbLoArray();
    auto const phi = geom.ProbHiArray();
    auto const dx_arr = geom.CellSizeArray();

    // Per-dimension domain clamp bounds. In non-periodic directions the
    // advected position is clamped just inside the domain (positions at or
    // beyond ProbHi count as outside) rather than handed to Redistribute,
    // which would DELETE the marker: since InitParticles runs only once, the
    // home cell would then have no QDSMC marker for the rest of the run and
    // its T_e could never be updated again. Clamping instead accumulates the
    // carried entropy at the boundary nodes and preserves the
    // one-marker-per-cell invariant (ResetParticles returns it home).
    // Periodic directions are left unclamped so Redistribute wraps them.
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lo_bnd;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> hi_bnd;
    amrex::GpuArray<int, AMREX_SPACEDIM> is_periodic;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        lo_bnd[d] = plo[d];
        hi_bnd[d] = phi[d] - 1.e-6_rt * dx_arr[d];
        is_periodic[d] = geom.isPeriodic(d);
    }

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        long const np = pti.numParticles();
        auto & attribs = pti.GetStructOfArrays().GetRealData();

        // Home and velocity components are only needed for the axes with an
        // AMReX-tracked position slot (x everywhere but 1D_Z, y only in 3D).
#if !defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal* const AMREX_RESTRICT x_node =
            attribs[QdsmcPIdx::x_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vx =
            attribs[QdsmcPIdx::vx].dataPtr();
#endif
#if defined(WARPX_DIM_3D)
        amrex::ParticleReal* const AMREX_RESTRICT y_node =
            attribs[QdsmcPIdx::y_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vy =
            attribs[QdsmcPIdx::vy].dataPtr();
#endif
        amrex::ParticleReal* const AMREX_RESTRICT z_node =
            attribs[QdsmcPIdx::z_node].dataPtr();
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

            // Forward-Euler push of one coordinate by one PIC step, clamped
            // just inside the domain in non-periodic directions (see the
            // bound setup above). The caller owns the CFL constraint
            // |v| dt < dx (at most one cell per step).
            auto const push_clamp = [&] (amrex::Real x0, amrex::Real v, int d)
            {
                amrex::Real const xnew = x0 + v * dt;
                return is_periodic[d] ? xnew
                                      : amrex::Clamp(xnew, lo_bnd[d], hi_bnd[d]);
            };

            // Write the new position to the AMReX-tracked position slots.
            // Axes not represented in the field have no enum slot and are
            // simply not tracked (consistent with field dimensionality).
#if defined(WARPX_DIM_3D)
            pa_x[ip] = push_clamp(x_node[ip], vx[ip], 0);
            pa_y[ip] = push_clamp(y_node[ip], vy[ip], 1);
            pa_z[ip] = push_clamp(z_node[ip], vz[ip], 2);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            pa_x[ip] = push_clamp(x_node[ip], vx[ip], 0);
            pa_z[ip] = push_clamp(z_node[ip], vz[ip], 1);
#elif defined(WARPX_DIM_1D_Z)
            pa_z[ip] = push_clamp(z_node[ip], vz[ip], 0);
#else
            // WARPX_DIM_RCYLINDER / WARPX_DIM_RSPHERE: x (= r) is the single
            // tracked position (QDSMC is refused at runtime in these
            // geometries); z is a plain attribute, advanced unclamped.
            pa_x[ip] = push_clamp(x_node[ip], vx[ip], 0);
            pa_z[ip] = z_node[ip] + vz[ip] * dt;
#endif
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

        // The home components are only needed where a matching AMReX-tracked
        // position slot exists (x everywhere but 1D_Z, y only in 3D).
#if !defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal* const AMREX_RESTRICT x_node =
            attribs[QdsmcPIdx::x_node].dataPtr();
#endif
#if defined(WARPX_DIM_3D)
        amrex::ParticleReal* const AMREX_RESTRICT y_node =
            attribs[QdsmcPIdx::y_node].dataPtr();
#endif
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
QdsmcParticleContainer::DepositScalar (int lev, int const attr,
                                       amrex::Real const scale,
                                       amrex::MultiFab & field)
{
    auto & warpx = WarpX::GetInstance();
    amrex::Periodicity const & period = warpx.Geom(lev).periodicity();
    amrex::XDim3 const dinv = WarpX::InvCellSize(lev);

    field.setVal(0);

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        long const np = pti.numParticles();
        auto & attribs = pti.GetStructOfArrays().GetRealData();

        // Assemble the position functor by hand: its constructor indexes the
        // SoA with the physical-particle PIdx layout, which does not match
        // QdsmcPIdx, so it must not be used with this container. The y_node
        // attribute (identically zero outside 3D) stands in for the angle
        // components, so the azimuthal geometries evaluate to (r, 0, z).
        GetParticlePosition<PIdx> GetPosition;
#if defined(WARPX_DIM_3D)
        GetPosition.m_x = attribs[QdsmcPIdx::x].dataPtr();
        GetPosition.m_y = attribs[QdsmcPIdx::y].dataPtr();
        GetPosition.m_z = attribs[QdsmcPIdx::z].dataPtr();
#elif defined(WARPX_DIM_XZ)
        GetPosition.m_x = attribs[QdsmcPIdx::x].dataPtr();
        GetPosition.m_z = attribs[QdsmcPIdx::z].dataPtr();
#elif defined(WARPX_DIM_RZ)
        GetPosition.m_x = attribs[QdsmcPIdx::x].dataPtr();
        GetPosition.m_z = attribs[QdsmcPIdx::z].dataPtr();
        GetPosition.m_theta = attribs[QdsmcPIdx::y_node].dataPtr();
#elif defined(WARPX_DIM_1D_Z)
        GetPosition.m_z = attribs[QdsmcPIdx::z].dataPtr();
#elif defined(WARPX_DIM_RCYLINDER)
        GetPosition.m_x = attribs[QdsmcPIdx::x].dataPtr();
        GetPosition.m_theta = attribs[QdsmcPIdx::y_node].dataPtr();
#elif defined(WARPX_DIM_RSPHERE)
        GetPosition.m_x = attribs[QdsmcPIdx::x].dataPtr();
        GetPosition.m_theta = attribs[QdsmcPIdx::y_node].dataPtr();
        GetPosition.m_phi = attribs[QdsmcPIdx::y_node].dataPtr();
#endif

        amrex::Box tilebox = pti.tilebox();
        tilebox.grow(field.nGrowVect());
        amrex::Dim3 const lo = amrex::lbound(tilebox);
        amrex::XDim3 const xyzmin = WarpX::LowerCorner(tilebox, lev, 0.0_rt);

        doChargeDepositionShapeN<1>(GetPosition, attribs[attr].dataPtr(),
                                    nullptr, field[pti], np, dinv, xyzmin, lo,
                                    scale, WarpX::n_rz_azimuthal_modes);
    }

    amrex::Gpu::synchronize();

    ablastr::utils::communication::SumBoundary(
        field, 0, field.nComp(), field.nGrowVect(), field.nGrowVect(),
        WarpX::do_single_precision_comms, period);
}


void
QdsmcParticleContainer::DepositK (int lev, amrex::MultiFab & Kfield)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::DepositK()");

    DepositScalar(lev, QdsmcPIdx::entropy, 1.0_rt, Kfield);
}


void
QdsmcParticleContainer::DepositField (int lev, amrex::MultiFab & Field)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::DepositField()");

    // np_real carries the electron count n_e * V_cell; the 1/V_cell scale
    // makes the deposited field an electron (number) density.
    auto const * dx_arr = WarpX::GetInstance().Geom(lev).CellSize();
    amrex::Real cell_volume = 1.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        cell_volume *= dx_arr[d];
    }
    DepositScalar(lev, QdsmcPIdx::np_real, 1.0_rt / cell_volume, Field);
}
