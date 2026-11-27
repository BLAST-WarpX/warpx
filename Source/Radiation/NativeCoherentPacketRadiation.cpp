/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "RadiationTransport.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "FluxRelaxedPackets.H"
#include "NativePhotonPathReplay.H"
#include "PacketMaterialWorkIteration.H"
#include "ParticleImpulse.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhotonParticleContainer.H"
#if defined(WARPX_DIM_RZ)
#include "Particles/Pusher/GetAndSetPosition.H"
#endif
#include "PhotonPacketTransaction.H"
#if defined(WARPX_DIM_RZ)
#include "RZMomentGeometry.H"
#include "RZNativePhotonPathReplay.H"
#endif
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

using namespace amrex::literals;
using warpx::fields::FieldType;

namespace
{
#if defined(WARPX_DIM_RZ)
using PhysicalPathReplay = warpx::radiation::RZNativePhotonPathReplay;

bool PhotonInventory (WarpXParticleContainer::Base& photons,
                      amrex::GpuArray<amrex::Real, 4>& inventory)
{
    if (!warpx::radiation::detail::PhotonInventory(photons, inventory)) { return false; }
    amrex::ReduceOps<amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Real> data(ops);
    for (WarpXParIter it(photons, 0); it.isValid(); ++it) {
        auto const p = it.GetParticleTile().getParticleTileData();
        auto const position = GetParticlePosition<PIdx>(it);
        ops.eval(it.numParticles(), data, [=] AMREX_GPU_DEVICE(long n)
                 -> amrex::GpuTuple<amrex::Real> {
            if (!amrex::ParticleIDWrapper{p.m_idcpu[n]}.is_valid()) { return {0}; }
            amrex::ParticleReal x = 0, y = 0, z = 0;
            position(n, x, y, z);
            return {p.m_rdata[PIdx::w][n] * PhysConst::m_e * PhysConst::c *
                (x * p.m_rdata[PIdx::uy][n] - y * p.m_rdata[PIdx::ux][n])};
        });
    }
    auto angular = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(angular);
    inventory[1] = 0;  // A global radial momentum is not conserved in RZ.
    inventory[2] = angular;  // c*Lz [J m].
    return std::isfinite(angular);
}

amrex::Real AngularFieldSum (amrex::MultiFab const& field,
                             amrex::Geometry const& geometry)
{
    warpx::radiation::RZMomentMetric const metric(geometry);
    amrex::ReduceOps<amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Real> data(ops);
    for (amrex::MFIter it(field); it.isValid(); ++it) {
        auto const q = field.const_array(it);
        ops.eval(it.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                 -> amrex::GpuTuple<amrex::Real> {
            return {q(i, j, k, 1) * metric.Angular(i).mean_radius};
        });
    }
    auto result = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(result);
    return result;
}
#else
using PhysicalPathReplay = warpx::radiation::NativePhotonPathReplay;

bool PhotonInventory (WarpXParticleContainer::Base& photons,
                      amrex::GpuArray<amrex::Real, 4>& inventory)
{
    return warpx::radiation::detail::PhotonInventory(photons, inventory);
}
#endif
}

void
RadiationTransport::AdvanceCoherentPackets (
    MultiParticleContainer& particles, ablastr::fields::MultiFabRegister& fields,
    amrex::Real current_time, amrex::Real dt) const
{
    using namespace warpx::radiation;
    auto const& simulation = WarpX::GetInstance();
    auto const& geometry = simulation.Geom(0);
    auto* photons = dynamic_cast<PhotonParticleContainer*>(
        &particles.GetParticleContainerFromName(m_photon_species));
    bool const geometry_valid =
#if defined(WARPX_DIM_RZ)
        (geometry.Coord() == 0 || geometry.Coord() == 1) &&
        geometry.ProbLo(0) == 0 && !geometry.isPeriodic(0);
#else
        geometry.Coord() == 0 && geometry.isAllPeriodic();
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(photons && photons->doNotPush() &&
        !photons->DoResampling() && !photons->DoQED() && geometry_valid &&
        simulation.finestLevel() == 0 && !EB::enabled(),
        "Coherent packets require supported level-zero geometry, owned photon pushing, "
        "no EB, resampling or QED.");
    auto const& boxes = photons->ParticleBoxArray(0);
    auto const& distribution = photons->ParticleDistributionMap(0);
    amrex::MultiFab scattering(boxes, distribution, 1, 1);
    FillFrozenCoherentOpacity(fields, geometry, current_time, scattering);
    amrex::Real min_dx = geometry.CellSize(0);
    for (int d = 1; d < AMREX_SPACEDIM; ++d) {
        min_dx = amrex::min(min_dx, geometry.CellSize(d));
    }
    // Deterministic prepartition only. Selecting a new partition after an
    // event-dependent failure would bias the sampled Poisson histories.
    amrex::Real const requested = std::ceil(amrex::max(1._rt,
        amrex::max(2*PhysConst::c*dt/min_dx, scattering.max(0)*PhysConst::c*dt/4)));
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(requested) &&
        requested <= m_max_transport_substeps,
        "Coherent packet path exceeds the configured substep budget.");
    int const initial_segments = static_cast<int>(requested);
    amrex::GpuArray<amrex::Real,4> initial_photons{};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(PhotonInventory(*photons, initial_photons),
                                    "Invalid initial coherent photon inventory.");

    std::unique_ptr<ParticleImpulseMaterial> accepted_material;
    std::unique_ptr<PhotonPacketTransaction> accepted_photons;
    amrex::MultiFab accepted_impulse, accepted_work;
    int accepted_segments = 0, total_iterations = 0, rejected_attempts = 0;
    amrex::Real accepted_energy_residual = 0, accepted_carry = 0;
    amrex::Real accepted_momentum_residual = 0;
    amrex::Long accepted_events = 0;
    amrex::GpuArray<amrex::Real, 3> accepted_wall_momentum{};

    for (int refinement = 0; refinement <= m_coherent_max_subdivisions; ++refinement) {
        int const factor = 1 << refinement;
        if (initial_segments > m_max_transport_substeps/factor) { break; }
        int const segments = initial_segments*factor;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<std::uint64_t>(segments) <=
                std::numeric_limits<std::uint64_t>::max()-m_coherent_segment_counter,
            "Coherent packet accepted-segment counter overflow.");
        auto material = std::make_unique<ParticleImpulseMaterial>(particles, m_momentum_species);
        auto photon_trial = std::make_unique<PhotonPacketTransaction>(*photons);
        amrex::MultiFab impulse_sum(boxes, distribution, 3, 0);
#if defined(WARPX_DIM_RZ)
        amrex::MultiFab conjugate_impulse_sum(boxes, distribution, 3, 0);
        conjugate_impulse_sum.setVal(0);
#endif
        amrex::MultiFab work_sum(boxes, distribution, 1, 0);
        amrex::MultiFab momentum_carry(boxes, distribution, 3, 0);
        amrex::MultiFab energy_carry(boxes, distribution, 1, 0);
        impulse_sum.setVal(0);
        work_sum.setVal(0);
        momentum_carry.setVal(0);
        energy_carry.setVal(0);
        amrex::Long events = 0;
        amrex::GpuArray<amrex::Real, 3> wall_momentum{};
        bool valid = true;
        for (int segment = 0; segment < segments && valid; ++segment) {
            amrex::MultiFab initial_impulse(boxes, distribution, 3, 0);
            amrex::MultiFab debit(boxes, distribution, 1, 0), scale(boxes, distribution, 1, 0);
            initial_impulse.setVal(0);
            ParticleImpulseTransaction initial_velocity;
            valid = initial_velocity.Stage(*material, "streaming", initial_impulse, true,
#if defined(WARPX_DIM_RZ)
                ParticleImpulseAssignment::RZAngularConservative
#else
                ParticleImpulseAssignment::NearestCell
#endif
            );
            if (!valid) { break; }
            NativePhotonCollisionOptions path_options;
            path_options.seed = m_particle_conversion_seed;
            path_options.interval = m_coherent_segment_counter +
                static_cast<std::uint64_t>(segment);
            path_options.moving_cross_sections = true;
            PhysicalPathReplay path;
            NativePhotonPathReport path_report;
            auto& baseline = photon_trial->Particles();
            valid = path.Prepare(baseline, *photons, scattering, initial_velocity.WorkVelocity(),
                                  geometry, dt/segments, path_options, path_report);
            if (!valid) {
                amrex::Print() << "Coherent path preparation rejected segment=" << segment
                    << " reason=" << path_report.reason << '\n';
                break;
            }
            std::unique_ptr<WarpXParticleContainer::Base> accepted_packet_step;
            auto replay = [&] (ParticleImpulseTransaction const& staged,
                               amrex::MultiFab& next_impulse, amrex::MultiFab& actual_debit,
                               amrex::MultiFab& uncancelled_scale)
            {
                auto next = std::make_unique<WarpXParticleContainer::Base>(baseline.make_alike<>());
                next->SetArena(baseline.arena());
                next->copyParticles(baseline, true);
                if (!path.Replay(staged.WorkVelocity(), *next, next_impulse, actual_debit,
                                 uncancelled_scale, path_report)) { return false; }
                accepted_packet_step = std::move(next);
                return true;
            };
            valid = replay(initial_velocity, initial_impulse, debit, scale);
            if (!valid) {
                amrex::Print() << "Coherent initial replay rejected segment=" << segment
                    << " reason=" << path_report.reason << '\n';
                break;
            }
            PacketMaterialWorkIterationOptions work_options;
            work_options.max_iterations = m_coherent_max_iterations;
#if defined(WARPX_DIM_RZ)
            work_options.assignment = ParticleImpulseAssignment::RZAngularConservative;
#endif
            PacketMaterialWorkIterationReport work_report;
            std::unique_ptr<ParticleImpulseTransaction> accepted_step;
            valid = TryIteratePacketMaterialWork(*material, "streaming", initial_impulse,
                replay, work_options, accepted_step, work_report);
            total_iterations += work_report.iterations;
            if (!valid || !accepted_step || !accepted_packet_step) {
                amrex::Print() << "Coherent work iteration rejected segment=" << segment
                    << " reason=" << PacketMaterialWorkIterationReasonName(work_report.reason)
                    << " path_reason=" << path_report.reason
                    << " iterations=" << work_report.iterations
                    << " impulse_residual=" << work_report.max_impulse_residual
                    << " radiation_work_residual=" << work_report.max_radiation_work_residual
                    << " material_work_residual=" << work_report.max_material_work_residual
                    << " material_work_excess_residual="
                    << work_report.max_material_work_excess_residual
                    << " material_impulse_residual=" << work_report.max_material_impulse_residual
                    << '\n';
                valid = false;
                break;
            }
            amrex::MultiFab::Add(impulse_sum, accepted_step->ActualImpulse(), 0, 0, 3, 0);
#if defined(WARPX_DIM_RZ)
            amrex::MultiFab::Add(conjugate_impulse_sum,
                accepted_step->ConjugateImpulse(), 0, 0, 3, 0);
#endif
            amrex::MultiFab::Add(work_sum, accepted_step->ActualWork(), 0, 0, 1, 0);
#if defined(WARPX_DIM_RZ)
            amrex::MultiFab::Add(momentum_carry,
                accepted_step->ConjugateMomentumCarryChange(), 0, 0, 3, 0);
#else
            amrex::MultiFab::Add(momentum_carry,
                accepted_step->MomentumCarryChange(), 0, 0, 3, 0);
#endif
            amrex::MultiFab::Add(energy_carry, accepted_step->EnergyCarryChange(), 0, 0, 1, 0);
            events += path_report.collision_events;
            for (int d = 0; d < 3; ++d) {
                wall_momentum[d] += path_report.wall_momentum[d];
            }
            accepted_step->Commit(); // Private material only.
            baseline.GetParticles(0).swap(accepted_packet_step->GetParticles(0));
            for (auto const& [key, tile] : baseline.GetParticles(0)) {
                baseline.DefineAndReturnParticleTile(0, key.first, key.second);
                amrex::ignore_unused(tile);
            }
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if (valid) { valid = photon_trial->ValidateLive(); }
        if (valid) {
            amrex::GpuArray<amrex::Real,4> final_photons{};
            valid = PhotonInventory(photon_trial->Particles(), final_photons);
            amrex::Real const work = work_sum.sum(0), carry = energy_carry.sum(0);
            amrex::Real const energy_residual =
                final_photons[0]-initial_photons[0]+work+carry;
            amrex::Real const energy_scale = std::abs(initial_photons[0]) +
                std::abs(final_photons[0])+std::abs(work)+std::abs(carry);
            amrex::Real momentum_error = 0;
#if defined(WARPX_DIM_RZ)
            amrex::Real const axial_defect = final_photons[3]-initial_photons[3] +
                PhysConst::c*(conjugate_impulse_sum.sum(2)+momentum_carry.sum(2) +
                              wall_momentum[2]);
            amrex::Real const angular_defect = final_photons[2]-initial_photons[2] +
                PhysConst::c*(AngularFieldSum(conjugate_impulse_sum, geometry) +
                              AngularFieldSum(momentum_carry, geometry));
            auto const radial_extent = geometry.ProbHi(0);
            momentum_error = amrex::max(std::abs(axial_defect),
                std::abs(angular_defect)/radial_extent);
            valid = valid && std::isfinite(axial_defect) && std::isfinite(angular_defect) &&
                std::abs(axial_defect) <= 1.e-10_rt*energy_scale &&
                std::abs(angular_defect) <= 1.e-10_rt*energy_scale*radial_extent;
#else
            for (int d = 0; d < 3; ++d) {
                amrex::Real const defect = final_photons[d+1]-initial_photons[d+1] +
                    PhysConst::c*(impulse_sum.sum(d)+momentum_carry.sum(d));
                momentum_error = amrex::max(momentum_error, std::abs(defect));
                valid = valid && std::isfinite(defect);
            }
#endif
            valid = valid && std::isfinite(energy_residual) && std::isfinite(energy_scale) &&
                std::abs(energy_residual) <= 1.e-10_rt*energy_scale &&
                momentum_error <= 1.e-10_rt*energy_scale;
            amrex::ParallelDescriptor::ReduceBoolAnd(valid);
            if (valid) {
                accepted_material = std::move(material);
                accepted_photons = std::move(photon_trial);
                accepted_impulse = std::move(impulse_sum);
                accepted_work = std::move(work_sum);
                accepted_segments = segments;
                accepted_energy_residual = energy_residual;
                accepted_momentum_residual = momentum_error/PhysConst::c;
                accepted_carry = carry;
                accepted_events = events;
                accepted_wall_momentum = wall_momentum;
                break;
            }
        }
        ++rejected_attempts;
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(accepted_material && accepted_photons,
        "No chronological coherent-packet attempt passed event, finite-work and full-vector "
        "conservation gates. Live photons/material/counter remain unchanged.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(accepted_material->Commit(),
                                     "Live material changed before coherent-packet commit.");
    accepted_photons->Commit();
    m_coherent_segment_counter += static_cast<std::uint64_t>(accepted_segments);
    auto& heat = *fields.get(FieldType::radiation_material_energy, 0);
    auto& momentum = *fields.get(FieldType::radiation_material_momentum, 0);
    auto& kinetic = *fields.get(FieldType::radiation_material_kinetic_energy, 0);
    heat.setVal(0);
    momentum.setVal(0);
    kinetic.setVal(0);
    amrex::MultiFab::Copy(momentum, accepted_impulse, 0, 0, 3, 0);
    amrex::MultiFab::Copy(kinetic, accepted_work, 0, 0, 1, 0);
    m_last_numerical_energy_residual = accepted_energy_residual;
    m_cumulative_numerical_energy_residual += accepted_energy_residual;
    m_last_material_carry_energy_change = accepted_carry;
    // Publish inactive-backend diagnostics only after the whole source commits.
    // Rejection or a zero-duration call must retain the last accepted ledger.
    m_last_boundary_energy_loss = 0;
    m_last_diffusion_boundary_energy_loss = 0;
    m_last_boundary_energy_injection = 0;
    m_last_streaming_boundary_energy_loss = 0;
    m_last_diffusion_boundary_momentum_loss = {0, 0, 0};
    m_last_particle_wall_momentum = {0, 0, 0};
    m_last_implicit_diffusion = {};
    m_last_streaming_boundary_momentum_loss = accepted_wall_momentum;
    m_last_coupled_solve = {total_iterations, accepted_segments, rejected_attempts, 0, 0,
                            accepted_energy_residual};
    if (m_implicit_diffusion_options.verbosity > 0) {
        amrex::Print() << "Coherent packets accepted_substeps=" << accepted_segments
            << " rejected_attempts=" << rejected_attempts << " collisions=" << accepted_events
            << " actual_work=" << accepted_work.sum(0)
            << " raw_energy_residual=" << accepted_energy_residual
            << " raw_momentum_residual=" << accepted_momentum_residual
            << " accepted_segment_counter=" << m_coherent_segment_counter << '\n';
    }
}
