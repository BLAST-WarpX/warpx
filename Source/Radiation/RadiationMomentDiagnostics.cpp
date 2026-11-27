/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "RadiationTransport.H"

#include "MomentFields.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "RZMomentGeometry.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <string>

using warpx::radiation::detail::MomentFieldName;

amrex::GpuArray<amrex::Real, 4>
RadiationTransport::representationCarryInventory (
    ablastr::fields::MultiFabRegister const& fields) const {
    amrex::GpuArray<amrex::Real, 4> result{};
    if (usesAdaptivePacketRepresentation()) {
        auto const& carry = *fields.get(
            warpx::radiation::ResolvedRepresentationCarryFieldName, 0);
        for (int d = 0; d < 4; ++d) {
            result[d] = carry.sum(d);
        }
    }
    return result;
}

amrex::GpuArray<amrex::Real, 3>
RadiationTransport::momentMomentumInventory (
    ablastr::fields::MultiFabRegister const& fields) const
{
    amrex::GpuArray<amrex::Real, 3> result{};
    if (m_use_coupled_moment_transport) {
        for (int d = 0; d < 3; ++d) {
            result[d] = fields.get(MomentFieldName(d), 0)->sum(0) / PhysConst::c;
        }
    } else if (m_use_flux_relaxed_diffusion) {
        for (int d = 0; d < 3; ++d) {
            result[d] = fields.get(
                warpx::radiation::detail::FluxRelaxedFieldName(d), 0)->sum(0) /
                PhysConst::c;
        }
    } else if (m_use_resolved_packet_fld) {
        auto const carry = representationCarryInventory(fields);
        for (int d = 0; d < 3; ++d) {
            result[d] =
                (fields.get(warpx::radiation::ResolvedMomentumFieldName(d), 0)
                     ->sum(0) +
                 carry[d + 1]) /
                PhysConst::c;
        }
    } else if (m_use_moving_multigroup) {
        for (int group = 0; group < m_num_groups; ++group) {
            for (int d = 0; d < 3; ++d) {
                result[d] +=
                    fields
                        .get(warpx::radiation::MovingGroupMomentumFieldName(
                                 group, d),
                             0)
                        ->sum(0) /
                    PhysConst::c;
            }
        }
    } else {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            false,
            "A radiation moment inventory requires coupled_moment or "
            "flux_relaxed or resolved_packet_fld transport.");
    }
    return result;
}

amrex::GpuArray<amrex::Real, 3>
RadiationTransport::momentAngularInventory (
    MultiParticleContainer& particles, ablastr::fields::MultiFabRegister const& fields) const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_use_coupled_moment_transport || m_use_moving_multigroup || m_use_resolved_packet_fld) &&
            m_particle_momentum_carry,
        "Angular radiation inventory requires coupled_moment/moving_multigroup/resolved packets "
        "and particle-owned carry.");
    amrex::GpuArray<amrex::Real, 3> result{};
#if defined(WARPX_DIM_RZ)
    warpx::radiation::RZMomentMetric const metric(WarpX::GetInstance().Geom(0));
    amrex::ReduceOps<amrex::ReduceOpSum> radiation_ops;
    amrex::ReduceData<amrex::Real> radiation_data(radiation_ops);
    int const radiation_groups = m_use_moving_multigroup ? m_num_groups : 1;
    for (int group = 0; group < radiation_groups; ++group) {
        auto const& moment = *fields.get(
            m_use_moving_multigroup     ? warpx::radiation::MovingGroupMomentumFieldName(group, 1)
            : m_use_resolved_packet_fld ? warpx::radiation::ResolvedMomentumFieldName(1)
                                        : MomentFieldName(1),
            0);
        for (amrex::MFIter it(moment); it.isValid(); ++it) {
            auto const q = moment.const_array(it);
            radiation_ops.eval(it.validbox(), radiation_data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<amrex::Real> {
                    return {metric.Angular(i).mean_radius * q(i, j, k) / PhysConst::c};
                });
        }
    }
    if (usesAdaptivePacketRepresentation())
    {
        auto const& carry = *fields.get(warpx::radiation::ResolvedRepresentationCarryFieldName, 0);
        for (amrex::MFIter it(carry); it.isValid(); ++it)
        {
            auto const q = carry.const_array(it);
            radiation_ops.eval(
                it.validbox(), radiation_data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<amrex::Real>
                { return {metric.Angular(i).mean_radius * q(i, j, k, 2) / PhysConst::c}; });
        }
    }
    if (m_use_resolved_packet_fld)
    {
        auto& photons = particles.GetParticleContainerFromName(m_photon_species);
        for (WarpXParIter it(photons, 0); it.isValid(); ++it)
        {
            auto const data = it.GetParticleTile().getParticleTileData();
            radiation_ops.eval(it.numParticles(), radiation_data,
                               [=] AMREX_GPU_DEVICE(long p) -> amrex::GpuTuple<amrex::Real>
                               {
                                   if (!amrex::ParticleIDWrapper{data.m_idcpu[p]}.is_valid())
                                   {
                                       return {0};
                                   }
                                   auto const angle = data.m_rdata[PIdx::theta][p];
                                   return {PhysConst::m_e * data.m_rdata[PIdx::w][p] *
                                           data.m_rdata[PIdx::r][p] *
                                           (data.m_rdata[PIdx::uy][p] * std::cos(angle) -
                                            data.m_rdata[PIdx::ux][p] * std::sin(angle))};
                               });
        }
    }
    result[0] = amrex::get<0>(radiation_data.value());
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> particle_ops;
    amrex::ReduceData<amrex::Real, amrex::Real> particle_data(particle_ops);
    for (auto const& name : m_momentum_species) {
        auto& species = particles.GetParticleContainerFromName(name);
        auto const mass = static_cast<amrex::Real>(species.getMass());
        amrex::GpuArray<int, 18> carry_components{};
        int const paths = radiation_groups + 1;
        for (int path = 0; path < paths; ++path) {
            auto const label = path == radiation_groups ? std::string("streaming")
                : "diffusion_" + std::to_string(path);
            carry_components[2 * path] = species.GetRealCompIndex("radiation_impulse_" + label + "_ux");
            carry_components[2 * path + 1] = species.GetRealCompIndex("radiation_impulse_" + label + "_uy");
        }
        for (WarpXParIter it(species, 0); it.isValid(); ++it) {
            auto const data = it.GetParticleTile().getParticleTileData();
            amrex::GpuArray<amrex::ParticleReal const*, 18> carry{};
            for (int d = 0; d < 2 * paths; ++d) {
                carry[d] = it.GetStructOfArrays().GetRealData(carry_components[d]).dataPtr();
            }
            particle_ops.eval(it.numParticles(), particle_data,
                [=] AMREX_GPU_DEVICE(long p) -> amrex::GpuTuple<amrex::Real, amrex::Real> {
                    auto const angle = static_cast<amrex::Real>(data.m_rdata[PIdx::theta][p]);
                    auto const cosine = std::cos(angle), sine = std::sin(angle);
                    auto const lever = mass * data.m_rdata[PIdx::w][p] * data.m_rdata[PIdx::r][p];
                    auto const represented = lever * (-sine * data.m_rdata[PIdx::ux][p]
                                                       + cosine * data.m_rdata[PIdx::uy][p]);
                    amrex::Real pending_x = 0, pending_y = 0;
                    for (int path = 0; path < paths; ++path) {
                        pending_x += carry[2 * path][p];
                        pending_y += carry[2 * path + 1][p];
                    }
                    auto const pending = lever * (-sine * pending_x + cosine * pending_y);
                    return {static_cast<amrex::Real>(represented), static_cast<amrex::Real>(pending)};
                });
        }
    }
    auto const values = particle_data.value();
    result[1] = amrex::get<0>(values);
    result[2] = amrex::get<1>(values);
    amrex::ParallelDescriptor::ReduceRealSum(result.data(), 3);
#else
    amrex::ignore_unused(particles, fields);
    WARPX_ABORT_WITH_MESSAGE("Angular radiation inventory requires RZ geometry.");
#endif
    return result;
}
