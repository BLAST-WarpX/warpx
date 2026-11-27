/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FluxRelaxedPackets.H"

#include "PacketMomentReconstruction.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/UpdatePosition.H"
#include "PhotonFourMomentum.H"
#include "PhotonLogicalDrawKey.H"
#include "PhotonPacketTransaction.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

namespace warpx::radiation::detail
{
using namespace amrex::literals;

bool
StageEmissionPlan (PhotonPacketTransaction& transaction, EmissionPlan const& plan)
{
    bool valid = std::cmp_less_equal(plan.x.size(), std::numeric_limits<long>::max());
    bool const keyed = !plan.logical_draw_keys.empty();
    valid = valid && (!keyed || std::numeric_limits<amrex::ParticleReal>::digits >= 32);
    std::array<int, 2> key_components{-1, -1};
    auto& candidate = transaction.Particles();
    valid = valid && (!keyed ||
        (plan.logical_draw_keys.size() == plan.x.size() &&
         FindPhotonDrawKeyComponents(candidate.GetRealSoANames(), key_components) &&
         key_components[0] >= 0));
    if (keyed) {
        std::unordered_set<std::uint64_t> seen;
        for (auto const key : plan.logical_draw_keys) {
            valid = valid && key != 0 && seen.insert(key).second;
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { return false; }

    auto const& owners = candidate.ParticleDistributionMap(0);
    auto const& boxes = candidate.ParticleBoxArray(0);
    int staging_grid = 0;
    for (int grid = 0; grid < boxes.size(); ++grid) {
        if (owners[grid] == amrex::ParallelDescriptor::MyProc()) {
            staging_grid = grid;
            break;
        }
    }
    // Empty ranks still participate. Grid zero is admissible for an empty
    // local batch; nonempty batches must have a grid owned by this rank.
    long old_count = 0;
    if (keyed && !plan.x.empty()) {
        auto const found = candidate.GetParticles(0).find({staging_grid, 0});
        if (found != candidate.GetParticles(0).end()) {
            old_count = found->second.numParticles();
        }
    }
    amrex::Vector<amrex::Vector<amrex::ParticleReal>> const attributes{plan.weight};
    amrex::Vector<amrex::Vector<int>> const integer_attributes{};
    valid = transaction.AddProvisionalBirths(
        staging_grid, static_cast<long>(plan.x.size()), plan.x, plan.y, plan.z,
        plan.ux, plan.uy, plan.uz, attributes, integer_attributes,
        /*unique_particles=*/true);
    if (!valid) { return false; }
    if (keyed && !plan.x.empty()) {
        auto& tile = candidate.GetParticles(0).at({staging_grid, 0});
        amrex::Vector<amrex::ParticleReal> low(plan.x.size()), high(plan.x.size());
        for (amrex::Long n = 0; n < plan.x.size(); ++n) {
            auto const parts = EncodePhotonDrawKey(plan.logical_draw_keys[n]);
            low[n] = parts[0];
            high[n] = parts[1];
        }
        auto& soa = tile.GetStructOfArrays();
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, low.begin(), low.end(),
                         soa.GetRealData(key_components[0]).begin() + old_count);
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, high.begin(), high.end(),
                         soa.GetRealData(key_components[1]).begin() + old_count);
        amrex::Gpu::streamSynchronize();
    }
    candidate.Redistribute();
    return true;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
RelabelPhotonIntoRadiation (amrex::Array4<amrex::Real> const& radiation,
                            amrex::Array4<amrex::Real const> const& beta, int const i, int const j,
                            int const k, amrex::ParticleReal const weight,
                            amrex::ParticleReal const ux, amrex::ParticleReal const uy,
                            amrex::ParticleReal const uz, amrex::Real const optical_depth,
                            amrex::Real const minimum_optical_depth, int* const invalid)
{
    // Do not read the shared reservoir while other packets scatter into it.
    // Relabel only this packet; the complete cell state is validated after deposition.
    auto const packet =
        warpx::radiation::PacketConversionFourMomentumFromPhoton(weight, ux, uy, uz);
    if (!packet.valid)
    {
        amrex::HostDevice::Atomic::Add(invalid, 1);
        return;
    }
    for (int d = 0; d < 4; ++d)
    {
        amrex::HostDevice::Atomic::Add(&radiation(i, j, k, d), packet.value[d]);
    }
    amrex::ignore_unused(beta, optical_depth, minimum_optical_depth);
}

[[nodiscard]] bool
AdvanceCandidatePhotons (WarpXParticleContainer::Base& photons, amrex::MultiFab& radiation,
                         amrex::MultiFab const& opacity, amrex::MultiFab const& beta,
                         amrex::Geometry const& geometry, amrex::Real const dt,
                         amrex::Real const minimum_optical_depth)
{
    auto const plo = geometry.ProbLoArray();
    auto const phi = geometry.ProbHiArray();
    auto const dxi = geometry.InvCellSizeArray();
    auto const dx = geometry.CellSizeArray();
    amrex::Real min_cell_size = dx[0];
    for (int d = 1; d < AMREX_SPACEDIM; ++d)
    {
        min_cell_size = amrex::min(min_cell_size, dx[d]);
    }
    amrex::Gpu::DeviceScalar<int> invalid(0);
    int* const invalid_ptr = invalid.dataPtr();

    amrex::MFItInfo info;
    if (amrex::Gpu::notInLaunchRegion())
    {
        info.EnableTiling(WarpXParticleContainer::tile_size);
    }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (WarpXParIter iterator(photons, 0, info); iterator.isValid(); ++iterator)
    {
        auto& tile = iterator.GetParticleTile();
        long const np = iterator.numParticles();
        auto const ptd = tile.getParticleTileData();
        auto* const idcpu = ptd.m_idcpu;
        auto* const wp = ptd.m_rdata[PIdx::w];
        auto const* const uxp = ptd.m_rdata[PIdx::ux];
        auto const* const uyp = ptd.m_rdata[PIdx::uy];
        auto const* const uzp = ptd.m_rdata[PIdx::uz];
        auto const get_position = GetParticlePosition<PIdx>(iterator);
        auto set_position = SetParticlePosition<PIdx>(iterator);
        auto const rad = radiation.array(iterator);
        auto const chi = opacity.const_array(iterator);
        auto const material_beta = beta.const_array(iterator);

        amrex::For(np,
                   [=] AMREX_GPU_DEVICE(long const ip) noexcept
                   {
                       if (!amrex::ParticleIDWrapper{idcpu[ip]}.is_valid() || !(wp[ip] > 0.0_prt))
                       {
                           return;
                       }
                       auto const particle = WarpXParticleContainer::ParticleType(ptd, ip);
                       auto const initial_cell = amrex::getParticleCell(particle, plo, dxi);
                       auto const [i0, j0, k0] = initial_cell.dim3();
                       amrex::Real const initial_optical_depth = chi(i0, j0, k0, 0) * min_cell_size;
                       if (initial_optical_depth >= minimum_optical_depth)
                       {
                           RelabelPhotonIntoRadiation(
                               rad, material_beta, i0, j0, k0, wp[ip], uxp[ip], uyp[ip], uzp[ip],
                               initial_optical_depth, minimum_optical_depth, invalid_ptr);
                           idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                           wp[ip] = 0.0_prt;
                           return;
                       }

                       amrex::ParticleReal x = 0.0_prt;
                       amrex::ParticleReal y = 0.0_prt;
                       amrex::ParticleReal z = 0.0_prt;
                       get_position(ip, x, y, z);
                       if (!amrex::Math::isfinite(x) || !amrex::Math::isfinite(y) ||
                           !amrex::Math::isfinite(z))
                       {
                           amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                           return;
                       }
                       amrex::ParticleReal const velocity_scale =
                           amrex::max(amrex::Math::abs(uxp[ip]), amrex::Math::abs(uyp[ip]),
                                      amrex::Math::abs(uzp[ip]));
                       if (!(velocity_scale > 0.0_prt) || !amrex::Math::isfinite(velocity_scale))
                       {
                           amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                           return;
                       }
                       UpdatePosition(x, y, z, uxp[ip], uyp[ip], uzp[ip], dt, 0.0_prt);
#if defined(WARPX_DIM_3D)
                       while (x < plo[0])
                       {
                           x += phi[0] - plo[0];
                       }
                       while (x >= phi[0])
                       {
                           x -= phi[0] - plo[0];
                       }
                       while (y < plo[1])
                       {
                           y += phi[1] - plo[1];
                       }
                       while (y >= phi[1])
                       {
                           y -= phi[1] - plo[1];
                       }
                       while (z < plo[2])
                       {
                           z += phi[2] - plo[2];
                       }
                       while (z >= phi[2])
                       {
                           z -= phi[2] - plo[2];
                       }
#elif defined(WARPX_DIM_XZ)
            while (x < plo[0]) { x += phi[0] - plo[0]; }
            while (x >= phi[0]) { x -= phi[0] - plo[0]; }
            while (z < plo[1]) { z += phi[1] - plo[1]; }
            while (z >= phi[1]) { z -= phi[1] - plo[1]; }
#else
            while (z < plo[0]) { z += phi[0] - plo[0]; }
            while (z >= phi[0]) { z -= phi[0] - plo[0]; }
#endif
                       set_position(ip, x, y, z);

                       // A ray entering a thick cell is converted at the next segment's
                       // start. It must not receive a second full segment of diffusion now.
                   });
    }

    int invalid_count = invalid.dataValue();
    amrex::ParallelDescriptor::ReduceIntMax(invalid_count);
    if (invalid_count != 0)
    {
        return false;
    }

    radiation.FillBoundary(geometry.periodicity());
    photons.Redistribute();
    return true;
}

[[nodiscard]] bool
BuildThinCellEmissionPlan (amrex::MultiFab& radiation, amrex::MultiFab const& opacity,
                           amrex::Geometry const& geometry, amrex::Real const min_cell_size,
                           amrex::Real const minimum_optical_depth,
                           amrex::Real const energy_threshold, amrex::Real const photon_energy,
                           EmissionPlan& plan)
{
    amrex::MFInfo const host_info = amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
    // Raw host/device FAB copies include ghosts and require identical strides.
    // A zero-ghost staging array would mix components and overread on copy-back.
    amrex::MultiFab host_radiation(radiation.boxArray(), radiation.DistributionMap(), 4,
                                   radiation.nGrowVect(), host_info);
    amrex::MultiFab host_opacity(opacity.boxArray(), opacity.DistributionMap(), 1,
                                 opacity.nGrowVect(), host_info);
    amrex::dtoh_memcpy(host_radiation, radiation);
    amrex::dtoh_memcpy(host_opacity, opacity);
    amrex::Gpu::streamSynchronize();

    auto const plo = geometry.ProbLoArray();
    auto const dx = geometry.CellSizeArray();
    auto const domain_lo = amrex::lbound(geometry.Domain());
    // Local planning can reject; no rank may enter FillBoundary until that
    // result is reduced collectively, including ranks without any local cells.
    bool valid = [&] ()
    {
        for (amrex::MFIter iterator(host_radiation); iterator.isValid(); ++iterator)
        {
            auto state = host_radiation.array(iterator);
            auto const chi = host_opacity.const_array(iterator);
            for (amrex::BoxIterator cell(iterator.validbox()); cell.ok(); ++cell)
            {
                auto const [i, j, k] = cell().dim3();
                amrex::Real const optical_depth = chi(i, j, k, 0) * min_cell_size;
                if (optical_depth >= minimum_optical_depth || state(i, j, k, 0) <= energy_threshold)
                {
                    continue;
                }
                amrex::ParticleReal x = 0.0_prt;
                amrex::ParticleReal y = 0.0_prt;
                amrex::ParticleReal z = 0.0_prt;
#if defined(WARPX_DIM_3D)
                x = static_cast<amrex::ParticleReal>(plo[0] + (i - domain_lo.x + 0.5_rt) * dx[0]);
                y = static_cast<amrex::ParticleReal>(plo[1] + (j - domain_lo.y + 0.5_rt) * dx[1]);
                z = static_cast<amrex::ParticleReal>(plo[2] + (k - domain_lo.z + 0.5_rt) * dx[2]);
#elif defined(WARPX_DIM_XZ)
                x = static_cast<amrex::ParticleReal>(plo[0] + (i - domain_lo.x + 0.5_rt) * dx[0]);
                z = static_cast<amrex::ParticleReal>(plo[1] + (j - domain_lo.y + 0.5_rt) * dx[1]);
#else
                z = static_cast<amrex::ParticleReal>(plo[0] + (i - domain_lo.x + 0.5_rt) * dx[0]);
#endif
                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
                {
                    return false;
                }

                warpx::radiation::FourVector before{};
                for (int d = 0; d < 4; ++d)
                {
                    before[d] = state(i, j, k, d);
                }
                auto const reconstruction =
                    warpx::radiation::ReconstructPacketMoments(before, photon_energy);
                if (!reconstruction.valid)
                {
                    return false;
                }
                for (int d = 0; d < 4; ++d)
                {
                    if (std::abs(reconstruction.signed_residual[d]) >
                        128 * std::numeric_limits<amrex::Real>::epsilon() * before[0])
                    {
                        return false;
                    }
                }
                for (auto const& photon : reconstruction.photons)
                {
                    if (photon.energy_weight == 0)
                    {
                        continue;
                    }
                    plan.x.push_back(x);
                    plan.y.push_back(y);
                    plan.z.push_back(z);
                    plan.ux.push_back(photon.proper_velocity[0]);
                    plan.uy.push_back(photon.proper_velocity[1]);
                    plan.uz.push_back(photon.proper_velocity[2]);
                    plan.weight.push_back(photon.energy_weight);
                }
                for (int d = 0; d < 4; ++d)
                {
                    plan.inventory[d] += reconstruction.reconstructed_four_momentum[d];
                    // Accepted only at arithmetic roundoff. The raw whole-step
                    // residual includes this signed conversion error; it is not
                    // assigned to material or left as an acausal tiny reservoir.
                    state(i, j, k, d) = 0;
                }
            }
        }
        return true;
    }();
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        return false;
    }
    amrex::htod_memcpy(radiation, host_radiation);
    amrex::Gpu::streamSynchronize();
    radiation.FillBoundary(geometry.periodicity());
    return true;
}

[[nodiscard]] bool
PhotonInventory (WarpXParticleContainer::Base& photons, amrex::GpuArray<amrex::Real, 4>& result)
{
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum>
        ops;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real> data(ops);
    using Tuple = typename decltype(data)::Type;
    amrex::Gpu::DeviceScalar<int> invalid(0);
    int* const invalid_ptr = invalid.dataPtr();
    for (WarpXParIter iterator(photons, 0); iterator.isValid(); ++iterator)
    {
        auto& tile = iterator.GetParticleTile();
        auto const ptd = tile.getParticleTileData();
        auto const* const idcpu = ptd.m_idcpu;
        auto const* const wp = ptd.m_rdata[PIdx::w];
        auto const* const uxp = ptd.m_rdata[PIdx::ux];
        auto const* const uyp = ptd.m_rdata[PIdx::uy];
        auto const* const uzp = ptd.m_rdata[PIdx::uz];
        auto const get_position = GetParticlePosition<PIdx>(iterator);
        ops.eval(iterator.numParticles(), data,
                 [=] AMREX_GPU_DEVICE(long const ip) noexcept -> Tuple
                 {
                     if (!amrex::ParticleIDWrapper{idcpu[ip]}.is_valid() || !(wp[ip] > 0.0_prt))
                     {
                         return {0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt};
                     }
                     amrex::ParticleReal x = 0.0_prt;
                     amrex::ParticleReal y = 0.0_prt;
                     amrex::ParticleReal z = 0.0_prt;
                     get_position(ip, x, y, z);
                     if (!amrex::Math::isfinite(x) || !amrex::Math::isfinite(y) ||
                         !amrex::Math::isfinite(z))
                     {
                         amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                         return {0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt};
                     }
                     auto const packet = warpx::radiation::PacketConversionFourMomentumFromPhoton(
                         wp[ip], uxp[ip], uyp[ip], uzp[ip]);
                     if (!packet.valid)
                     {
                         amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                         return {0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt};
                     }
                     return {packet.value[0], packet.value[1], packet.value[2], packet.value[3]};
                 });
    }
    auto const local = data.value();
    result = amrex::GpuArray<amrex::Real, 4>{amrex::get<0>(local), amrex::get<1>(local),
                                             amrex::get<2>(local), amrex::get<3>(local)};
    amrex::ParallelDescriptor::ReduceRealSum(result.data(), 4);
    int invalid_count = invalid.dataValue();
    amrex::ParallelDescriptor::ReduceIntMax(invalid_count);
    return invalid_count == 0;
}

} // namespace warpx::radiation::detail
