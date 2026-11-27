/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "RadiationTransport.H"

#include "AdaptivePacketDiffusionTransition.H"
#include "CollidedDiffusionAdmission.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "FluxRelaxedPackets.H"
#include "GroupMomentGeometry.H"
#include "PacketDiffusionNativeInterval.H"
#include "Particles/MultiParticleContainer.H"
#include "PhotonFourMomentum.H"
#include "PhotonLogicalDrawKey.H"
#include "ResolvedDiffusionFront.H"
#include "ResolvedMaterialOpacity.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_Print.H>
#include <AMReX_Reduce.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace amrex::literals;
using namespace warpx::radiation;
using warpx::fields::FieldType;

#if defined(WARPX_DIM_RZ)
namespace
{
// Free function keeps CUDA extended lambdas out of private member functions.
void
InitializeOrValidateResolvedFields (
    amrex::MultiFab& mask, amrex::MultiFab& opacity, amrex::MultiFab const& energy,
    amrex::Geometry const& geometry, amrex::ParserExecutor<3> mask_parser,
    ResolvedOpacityExecutor opacity_parser, bool restored = false, amrex::Real opacity_time = 0,
    amrex::MultiFab const* electron_density = nullptr,
    amrex::MultiFab const* electron_temperature = nullptr, bool defer_material_opacity = false,
    bool preserve_adaptive_mask = false, bool allow_uniform_mask = false)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (!defer_material_opacity || (opacity_parser.material_dependent && !restored)) &&
            (!opacity_parser.material_dependent || defer_material_opacity ||
             (electron_density && electron_temperature)),
        "Resolved material opacity requires explicit frozen samples outside fresh allocation.");
    auto const lo = geometry.ProbLoArray(), dx = geometry.CellSizeArray();
    auto const origin = amrex::lbound(geometry.Domain());
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    for (amrex::MFIter it(mask); it.isValid(); ++it)
    {
        auto const m = mask.array(it), chi = opacity.array(it);
        auto const e = energy.const_array(it);
        auto const ne = electron_density ? electron_density->const_array(it)
                                         : amrex::Array4<amrex::Real const>{};
        auto const te = electron_temperature ? electron_temperature->const_array(it)
                                             : amrex::Array4<amrex::Real const>{};
        ops.eval(
            it.validbox(), data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
            {
                amrex::Real x = 0, y = 0, z = 0;
#if defined(WARPX_DIM_3D)
                x = lo[0] + (i - origin.x + 0.5_rt) * dx[0];
                y = lo[1] + (j - origin.y + 0.5_rt) * dx[1];
                z = lo[2] + (k - origin.z + 0.5_rt) * dx[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                x = lo[0] + (i - origin.x + 0.5_rt) * dx[0];
                z = lo[1] + (j - origin.y + 0.5_rt) * dx[1];
#elif defined(WARPX_DIM_1D_Z)
                z = lo[0] + (i - origin.x + 0.5_rt) * dx[0];
#endif
                auto const expected_mask = mask_parser(x, y, z);
                auto const density_value = ne ? ne(i, j, k) : 0;
                auto const temperature_value = te ? te(i, j, k) : 0;
                auto const expected_opacity =
                    defer_material_opacity
                        ? 0
                        : opacity_parser(x, y, z, opacity_time, density_value, temperature_value);
                if (!restored)
                {
                    if (!preserve_adaptive_mask)
                    {
                        m(i, j, k) = expected_mask;
                    }
                    chi(i, j, k) = expected_opacity;
                }
                // Permit only floating evaluation roundoff across CPU/GPU
                // parsers, not a changed binding hidden behind the same text.
                auto const opacity_roundoff =
                    64 * std::numeric_limits<amrex::Real>::epsilon() *
                    amrex::max(std::abs(chi(i, j, k)), std::abs(expected_opacity));
                return {!(std::isfinite(m(i, j, k)) && (m(i, j, k) == 0 || m(i, j, k) == 1) &&
                          std::isfinite(chi(i, j, k)) && chi(i, j, k) >= 0 &&
                          std::isfinite(e(i, j, k)) && e(i, j, k) >= 0 &&
                          (m(i, j, k) == 1 || e(i, j, k) == 0) && std::isfinite(density_value) &&
                          density_value >= 0 && std::isfinite(temperature_value) &&
                          temperature_value >= 0 && std::isfinite(expected_mask) &&
                          (preserve_adaptive_mask || expected_mask == m(i, j, k)) &&
                          std::isfinite(expected_opacity) && expected_opacity >= 0 &&
                          std::abs(expected_opacity - chi(i, j, k)) <= opacity_roundoff)};
            });
    }
    int invalid = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !invalid,
        restored
            ? "Resolved packet-FLD restart changed evaluated mask/opacity or has invalid E state."
            : "resolved_packet_fld requires a binary mask, finite nonnegative opacity/E, "
              "and zero initial diffusion energy in kinetic cells.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        allow_uniform_mask || (mask.min(0) == 0 && mask.max(0) == 1),
        "resolved_packet_fld requires a mixed domain containing both kinetic "
        "and diffusion cells.");
    mask.FillBoundary(geometry.periodicity());
    opacity.FillBoundary(geometry.periodicity());
}

void
CopyResolvedMask (amrex::MultiFab const& source, amrex::iMultiFab& mask)
{
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    for (amrex::MFIter it(mask); it.isValid(); ++it)
    {
        auto const src = source.const_array(it);
        auto const dst = mask.array(it);
        ops.eval(it.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                 {
                     if (!std::isfinite(src(i, j, k)) || (src(i, j, k) != 0 && src(i, j, k) != 1))
                     {
                         return {1};
                     }
                     dst(i, j, k) = static_cast<int>(src(i, j, k));
                     return {0};
                 });
    }
    int invalid = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!invalid, "Invalid persistent resolved diffusion mask.");
}

void
MaskResolvedOpacity (amrex::MultiFab const& opacity, amrex::iMultiFab const& mask,
                     amrex::MultiFab& diffusion_opacity)
{
    for (amrex::MFIter it(diffusion_opacity); it.isValid(); ++it)
    {
        auto const chi = opacity.const_array(it);
        auto const output = diffusion_opacity.array(it);
        auto const active = mask.const_array(it);
        amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           { output(i, j, k) = active(i, j, k) == 1 ? chi(i, j, k) : 0; });
    }
}

void
PublishResolvedMask (amrex::iMultiFab const& mask, amrex::MultiFab& field)
{
    for (amrex::MFIter it(field); it.isValid(); ++it)
    {
        auto const source = mask.const_array(it);
        auto const target = field.array(it);
        amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           { target(i, j, k) = source(i, j, k); });
    }
}

void
ConstrainAdaptiveMask (amrex::iMultiFab const& old, amrex::iMultiFab& desired,
                       amrex::MultiFab const& opacity, amrex::Geometry const& geometry,
                       amrex::Real dt, NativePhotonPathBoundaries const& boundaries,
                       bool demotion_only)
{
    auto const domain = geometry.Domain();
    auto const low = amrex::lbound(domain), high = amrex::ubound(domain);
    auto const escape_lo = boundaries.escape_lo, escape_hi = boundaries.escape_hi;
    for (amrex::MFIter it(desired); it.isValid(); ++it)
    {
        auto const previous = old.const_array(it);
        auto const next = desired.array(it);
        auto const chi = opacity.const_array(it);
        amrex::ParallelFor(it.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               int const position[3]{i, j, k};
                               int const lo[3]{low.x, low.y, low.z}, hi[3]{high.x, high.y, high.z};
                               bool kinetic = chi(i, j, k) * PhysConst::c * dt < 8;
                               for (int d = 0; d < AMREX_SPACEDIM; ++d)
                               {
                                   kinetic = kinetic || (escape_lo[d] && position[d] == lo[d]) ||
                                             (escape_hi[d] && position[d] == hi[d]);
                               }
                               if (kinetic || (demotion_only && previous(i, j, k) == 0))
                               {
                                   next(i, j, k) = 0;
                               }
                           });
    }
}

bool
RZPhotonInventory (WarpXParticleContainer::Base const& photons,
                   amrex::GpuArray<amrex::Real, 4>& inventory)
{
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpMax>
        ops;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, int> reduced(ops);
    for (auto const& [key, tile] : photons.GetParticles(0))
    {
        amrex::ignore_unused(key);
        auto const data = tile.getConstParticleTileData();
        ops.eval(
            tile.numParticles(), reduced,
            [=] AMREX_GPU_DEVICE(
                long p) -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, int>
            {
                if (!amrex::ParticleIDWrapper{data.m_idcpu[p]}.is_valid())
                {
                    return {0, 0, 0, 0};
                }
                auto const ray = PacketConversionFourMomentumFromPhoton(
                    data.m_rdata[PIdx::w][p], data.m_rdata[PIdx::ux][p], data.m_rdata[PIdx::uy][p],
                    data.m_rdata[PIdx::uz][p]);
                auto const radius = data.m_rdata[PIdx::r][p], angle = data.m_rdata[PIdx::theta][p];
                if (!ray.valid || !std::isfinite(radius) || radius < 0 || !std::isfinite(angle))
                {
                    return {0, 0, 0, 1};
                }
                auto const angular =
                    radius * (ray.value[2] * std::cos(angle) - ray.value[1] * std::sin(angle));
                return {ray.value[0], angular, ray.value[3], !std::isfinite(angular)};
            });
    }
    auto const value = reduced.value();
    int invalid = amrex::get<3>(value);
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    inventory = {amrex::get<0>(value), 0, amrex::get<1>(value), amrex::get<2>(value)};
    amrex::ParallelDescriptor::ReduceRealSum(inventory.data(), 4);
    return invalid == 0 && std::isfinite(inventory[0]) && std::isfinite(inventory[2]) &&
           std::isfinite(inventory[3]);
}

amrex::GpuArray<amrex::Real, 4>
RZFieldInventory (amrex::MultiFab const& energy, amrex::MultiFab const& q,
                  amrex::MultiFab const& carry, amrex::Geometry const& geometry)
{
    GroupMomentMetric const metric(geometry);
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real> reduced(ops);
    for (amrex::MFIter it(energy); it.isValid(); ++it)
    {
        auto const e = energy.const_array(it), momentum = q.const_array(it),
                   c = carry.const_array(it);
        ops.eval(it.validbox(), reduced,
                 [=] AMREX_GPU_DEVICE(
                     int i, int j, int k) -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real>
                 {
                     return {e(i, j, k) + c(i, j, k, 0),
                             metric.Angular(i).mean_radius * (momentum(i, j, k, 1) + c(i, j, k, 2)),
                             momentum(i, j, k, 2) + c(i, j, k, 3)};
                 });
    }
    auto const value = reduced.value();
    amrex::GpuArray<amrex::Real, 4> inventory{amrex::get<0>(value), 0, amrex::get<1>(value),
                                              amrex::get<2>(value)};
    amrex::ParallelDescriptor::ReduceRealSum(inventory.data(), 4);
    return inventory;
}

// Actual stored native Cartesian kinetic/work-carry E, cLz and cPz, plus
// uncancelled native energy/angular/axial operands for arithmetic conditioning.
amrex::GpuArray<amrex::Real, 7>
RZMaterialInventory (MultiParticleContainer& particles, std::vector<std::string> const& species,
                     ParticleImpulseMaterial const* candidate)
{
    amrex::GpuArray<amrex::Real, 7> inventory{};
    for (auto const& name : species)
    {
        auto const& live = particles.GetParticleContainerFromName(name);
        auto const* private_species = candidate ? candidate->CandidateSpecies(name) : &live;
        bool present = private_species != nullptr;
        amrex::ParallelDescriptor::ReduceBoolAnd(present);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(present, "Missing actual private RZ adaptive material.");
        auto const& material = *private_species;
        auto const mass_per_particle = static_cast<amrex::Real>(live.getMass());
        std::array<int, 8> indices{};
        for (int path = 0; path < 2; ++path)
        {
            auto const label = path == 0 ? std::string("diffusion_0") : std::string("streaming");
            for (int d = 0; d < 4; ++d)
            {
                std::array<std::string, 4> const suffix{"ux", "uy", "uz", "work"};
                auto const& names = material.GetRealSoANames();
                auto const found = std::find(names.begin(), names.end(),
                                             "radiation_impulse_" + label + "_" + suffix[d]);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(found != names.end(),
                                                 "Missing RZ adaptive native carry.");
                indices[4 * path + d] = static_cast<int>(found - names.begin());
            }
        }
        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                         amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum>
            ops;
        amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                          amrex::Real>
            reduced(ops);
        for (auto const& [key, tile] : material.GetParticles(0))
        {
            amrex::ignore_unused(key);
            auto const data = tile.getConstParticleTileData();
            amrex::GpuArray<amrex::ParticleReal const*, 8> carry{};
            for (int d = 0; d < 8; ++d)
            {
                carry[d] = tile.GetStructOfArrays().GetRealData(indices[d]).dataPtr();
            }
            ops.eval(
                tile.numParticles(), reduced,
                [=] AMREX_GPU_DEVICE(
                    long p) -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                                               amrex::Real, amrex::Real>
                {
                    auto const mass = mass_per_particle * data.m_rdata[PIdx::w][p];
                    auto const ux = data.m_rdata[PIdx::ux][p], uy = data.m_rdata[PIdx::uy][p],
                               uz = data.m_rdata[PIdx::uz][p];
                    auto const square = ux * ux + uy * uy + uz * uz;
                    auto const ke = mass * square / (std::sqrt(1 + square / PhysConst::c2) + 1);
                    auto const radius = data.m_rdata[PIdx::r][p];
                    auto const cosine = std::cos(data.m_rdata[PIdx::theta][p]),
                               sine = std::sin(data.m_rdata[PIdx::theta][p]);
                    auto const x = carry[0][p] + carry[4][p], y = carry[1][p] + carry[5][p],
                               z = carry[2][p] + carry[6][p], work = carry[3][p] + carry[7][p];
                    auto const sx = std::abs(ux) + std::abs(carry[0][p]) + std::abs(carry[4][p]);
                    auto const sy = std::abs(uy) + std::abs(carry[1][p]) + std::abs(carry[5][p]);
                    auto const sz = std::abs(uz) + std::abs(carry[2][p]) + std::abs(carry[6][p]);
                    return {ke + mass * work,
                            mass * PhysConst::c * radius * ((uy + y) * cosine - (ux + x) * sine),
                            mass * PhysConst::c * (uz + z),
                            std::abs(ke) + mass * (std::abs(carry[3][p]) + std::abs(carry[7][p])),
                            mass * PhysConst::c * radius *
                                (sy * std::abs(cosine) + sx * std::abs(sine)),
                            mass * PhysConst::c * sz};
                });
        }
        auto const value = reduced.value();
        amrex::GpuArray<amrex::Real, 6> local{amrex::get<0>(value), amrex::get<1>(value),
                                              amrex::get<2>(value), amrex::get<3>(value),
                                              amrex::get<4>(value), amrex::get<5>(value)};
        amrex::ParallelDescriptor::ReduceRealSum(local.data(), 6);
        inventory[0] += local[0];
        inventory[2] += local[1];
        inventory[3] += local[2];
        inventory[4] += local[3];
        inventory[5] += local[4];
        inventory[6] += local[5];
    }
    return inventory;
}

bool
AuditRZCombined (amrex::GpuArray<amrex::Real, 4> const& old_packets,
                 amrex::GpuArray<amrex::Real, 4> const& old_fields,
                 amrex::GpuArray<amrex::Real, 4> const& new_packets,
                 amrex::GpuArray<amrex::Real, 4> const& new_fields,
                 amrex::GpuArray<amrex::Real, 7> const& old_material,
                 amrex::GpuArray<amrex::Real, 7> const& new_material,
                 PacketDiffusionNativeIntervalResult const& stage, amrex::Geometry const& geometry,
                 amrex::Real tolerance, amrex::Real physical_energy_scale,
                 amrex::Real& energy_residual)
{
    // Representation carry participates in inventory but NEVER enlarges the
    // physical radiation acceptance scale or supplies a material force/work.
    amrex::GpuArray<amrex::Real, 4> residual{};
    for (int d : {0, 2, 3})
    {
        residual[d] = new_packets[d] - old_packets[d] + new_fields[d] - old_fields[d] +
                      new_material[d] - old_material[d];
    }
    residual[3] +=
        PhysConst::c * (stage.packet_wall_momentum[2] + stage.radiation_wall_momentum[2]);
    energy_residual = residual[0];
    auto const epsilon = 128 * std::numeric_limits<amrex::Real>::epsilon();
    amrex::GpuArray<amrex::Real, 4> bound{};
    bound[0] = tolerance * physical_energy_scale +
               epsilon * (physical_energy_scale + old_material[4] + new_material[4]);
    bound[2] =
        tolerance * physical_energy_scale * geometry.ProbHi(0) +
        epsilon * (physical_energy_scale * geometry.ProbHi(0) + old_material[5] + new_material[5]);
    bound[3] = tolerance * physical_energy_scale +
               epsilon * (physical_energy_scale + old_material[6] + new_material[6]);
    bool balanced = stage.escaped_packet_inventory[0] == 0;
    for (int d : {0, 2, 3})
    {
        balanced = balanced && std::isfinite(residual[d]) && std::isfinite(bound[d]) &&
                   std::abs(residual[d]) <= bound[d];
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(balanced);
    return balanced;
}

} // namespace
#endif

void
RadiationTransport::AdvanceResolvedRZAdaptiveRadiation (MultiParticleContainer& particles,
                                                        ablastr::fields::MultiFabRegister& fields,
                                                        amrex::Real current_time,
                                                        amrex::Real dt) const
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(particles, fields, current_time, dt);
    WARPX_ABORT_WITH_MESSAGE("Native resolved RZ adaptive transport requires RZ geometry.");
#else
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        usesAdaptivePacketRepresentation() && !m_enable_lte_exchange,
        "RZ adaptive resolved transport currently requires coherent scattering without LTE.");
    auto const old_material = RZMaterialInventory(particles, m_momentum_species, nullptr);
    auto const& geometry = WarpX::GetInstance().Geom(0);
    auto& photons = particles.GetParticleContainerFromName(m_photon_species);
    auto& energy = *fields.get(FieldType::radiation_diffusion_energy, 0);
    auto& mask_field = *fields.get(ResolvedMaskFieldName, 0);
    auto& opacity_state = *fields.get(ResolvedOpacityFieldName, 0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_resolved_packet.interval <
                                         std::numeric_limits<std::uint64_t>::max(),
                                     "Resolved packet interval counter overflow.");
    amrex::iMultiFab mask(energy.boxArray(), energy.DistributionMap(), 1, 1);
    mask.setVal(0);
    CopyResolvedMask(mask_field, mask);
    mask.FillBoundary(geometry.periodicity());
    amrex::MultiFab opacity(energy.boxArray(), energy.DistributionMap(), 1, 1);
    opacity.setVal(0);
    amrex::MultiFab::Copy(opacity, opacity_state, 0, 0, 1, 0);
    std::unique_ptr<amrex::MultiFab> ne, te;
    if (usesResolvedMaterialOpacity())
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_resolved_packet.material_opacity_initialized,
            "Resolved material opacity requires initialized native material state.");
        ne = std::make_unique<amrex::MultiFab>(energy.boxArray(), energy.DistributionMap(), 1, 1);
        te = std::make_unique<amrex::MultiFab>(energy.boxArray(), energy.DistributionMap(), 1, 1);
        ne->setVal(0);
        te->setVal(0);
        auto const floor = m_hybrid_model->transportsElectronInternalEnergyAtRawDensity()
                               ? 0.0_rt
                               : m_hybrid_model->electronDensityFloor();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            detail::SnapshotResolvedMaterialOpacity(
                *fields.get(FieldType::rho_fp, 0),
                *fields.get(FieldType::hybrid_electron_temperature_fp, 0), *ne, *te, geometry,
                m_hybrid_model->electronThermodynamicsExecutor(), m_minimum_electron_density,
                floor),
            "Resolved opacity requires finite admissible native material samples.");
    }
    amrex::Real const opacity_time =
        m_resolved_packet.time_dependent_opacity ? current_time + dt : 0;
    if (m_resolved_packet.time_dependent_opacity)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(current_time) && std::isfinite(opacity_time) &&
                opacity_time > current_time,
            "Resolved opacity refresh requires a finite advancing endpoint time.");
        // Keep every accepted field untouched if coefficient evaluation or the
        // subsequent coupled interval rejects. Refreshing opacity does not
        // authorize deleting D inventory before conservative demotion.
        amrex::MultiFab mask_trial(mask_field.boxArray(), mask_field.DistributionMap(), 1, 1);
        amrex::MultiFab::Copy(mask_trial, mask_field, 0, 0, 1, 1);
        InitializeOrValidateResolvedFields(
            mask_trial, opacity, energy, geometry, m_resolved_packet.mask_parser.compile<3>(),
            m_resolved_packet.opacityExecutor(), false, opacity_time, ne.get(), te.get(), false,
            usesAdaptivePacketRepresentation(), usesAdaptivePacketRepresentation());
    }
    opacity.FillBoundary(geometry.periodicity());
    // The packet rate covers the whole domain. The scalar operator's K cells
    // are inactive and require exactly zero opacity; active D rates agree.
    amrex::MultiFab diffusion_opacity(energy.boxArray(), energy.DistributionMap(), 1, 1);
    diffusion_opacity.setVal(0);
    amrex::MultiFab::Copy(diffusion_opacity, opacity, 0, 0, 1, 0);
    MaskResolvedOpacity(opacity, mask, diffusion_opacity);
    diffusion_opacity.FillBoundary(geometry.periodicity());
    amrex::MultiFab q(energy.boxArray(), energy.DistributionMap(), 3, 1);
    q.setVal(0);
    for (int d = 0; d < 3; ++d)
    {
        amrex::MultiFab::Copy(q, *fields.get(ResolvedMomentumFieldName(d), 0), 0, d, 1, 0);
    }
    q.FillBoundary(geometry.periodicity());
    PacketDiffusionNativeIntervalOptions options;
    options.resolved_collided_source = true;
    options.quasi_static_angular_traction = true;
    options.retain_constitutive_momentum = true;
    options.moving_diffusion = m_resolved_packet.moving_diffusion;
    options.kinetic_empty_cells = m_resolved_packet.kinetic_empty_cells;
    options.transactional_packet_substeps = m_resolved_packet.packet_substeps;
    options.packet_subsegment_limit = m_resolved_packet.packet_subsegment_limit;
    options.photon_energy = m_resolved_packet.photon_energy;
    options.kinetic_scattering_opacity = &opacity;
    options.material.max_iterations = m_resolved_packet.max_iterations;
    options.diffusion.verbosity = m_resolved_packet.verbosity;
    options.collisions.seed = m_particle_conversion_seed;
    options.collisions.interval = m_resolved_packet.interval;
    options.logical_birth_keys =
        LogicalPhotonBirthKeyContext{m_particle_conversion_seed, m_resolved_packet.interval};
    options.material.assignment = ParticleImpulseAssignment::RZAngularConservative;
    std::unique_ptr<PhotonPacketTransaction> adaptive_photons;
    std::unique_ptr<amrex::MultiFab> adaptive_energy;
    amrex::MultiFab adaptive_carry;
    amrex::iMultiFab desired_mask;
    amrex::GpuArray<amrex::Real, 4> initial_packet_inventory{};
    amrex::GpuArray<amrex::Real, 4> initial_field_inventory{};
    amrex::Long demoted_cells = 0, promoted_cells = 0, deferred_cells = 0;
    CollidedDiffusionAdmissionComponents census;
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            RZPhotonInventory(photons, initial_packet_inventory),
            "Adaptive representation requires a valid initial native photon "
            "inventory.");
        auto const& old_carry = *fields.get(ResolvedRepresentationCarryFieldName, 0);
        initial_field_inventory = RZFieldInventory(energy, q, old_carry, geometry);
        for (int d = 0; d < 3; ++d)
        {
            census.initial_u[d] = photons.GetRealCompIndex(
                std::string("radiation_diffusion_census_u") + "xyz"[d] + "0");
        }
        adaptive_photons = std::make_unique<PhotonPacketTransaction>(photons);
        auto const snapshot =
            SnapshotCollidedDiffusionAdmission(adaptive_photons->Particles(), census);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            snapshot.valid, "Adaptive representation could not snapshot the physical census.");
        desired_mask.define(mask.boxArray(), mask.DistributionMap(), 1, 1);
        desired_mask.setVal(0);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            TryProposeAdaptiveDiffusionMask(mask, opacity, geometry, desired_mask,
                                            m_resolved_packet.adaptive_policy),
            "Adaptive representation rejected its opacity/mask proposal.");
        ConstrainAdaptiveMask(mask, desired_mask, opacity, geometry, dt, options.photon_boundaries,
                              false);
        // Demotion must precede physical streaming. Promotion instead needs
        // the fresh collision census produced by this interval's actual replay.
        amrex::iMultiFab early_mask(mask.boxArray(), mask.DistributionMap(), 1, 1);
        early_mask.setVal(0);
        amrex::iMultiFab::Copy(early_mask, desired_mask, 0, 0, 1, 0);
        ConstrainAdaptiveMask(mask, early_mask, opacity, geometry, dt, options.photon_boundaries,
                              true);
        auto converted = TryConvertAdaptivePacketDiffusionRepresentation(
            *adaptive_photons, energy, q, mask, early_mask, old_carry, geometry, census,
            *options.logical_birth_keys, options.photon_energy, options.admission);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            converted.valid, "Adaptive demotion rejected before transport: " + converted.reason);
        adaptive_energy = std::make_unique<amrex::MultiFab>(std::move(converted.energy));
        q = std::move(converted.momentum);
        mask = std::move(converted.mask);
        adaptive_carry = std::move(converted.representation_carry);
        demoted_cells = converted.demoted_cells;
        mask.FillBoundary(geometry.periodicity());
        q.FillBoundary(geometry.periodicity());
        MaskResolvedOpacity(opacity, mask, diffusion_opacity);
        diffusion_opacity.FillBoundary(geometry.periodicity());
        options.private_photon_snapshot = adaptive_photons.get();
        options.defer_live_commit = true;
        options.adaptive_rz_private_snapshot = true;
        options.radiation_representation_carry = &adaptive_carry;
    }
    auto result = TryAdvancePacketDiffusionNativeInterval(
        particles, photons, m_momentum_species, "diffusion_0",
        adaptive_energy ? *adaptive_energy : energy, diffusion_opacity, mask, geometry, dt, nullptr,
        options, &q);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        result.valid,
        "Resolved packet/FLD interval rejected before radiation-stage commit: " + result.reason);
    {
        AMREX_ALWAYS_ASSERT(result.private_photons && result.private_material &&
                            result.private_radiation_representation_carry.ok());
        adaptive_carry = std::move(result.private_radiation_representation_carry);
        PhotonPacketTransaction promotion(photons, *result.private_photons);
        auto converted = TryConvertAdaptivePacketDiffusionRepresentation(
            promotion, result.private_radiation_energy, result.private_diffusion_momentum, mask,
            desired_mask, adaptive_carry, geometry, census, *options.logical_birth_keys,
            options.photon_energy, options.admission);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            converted.valid,
            "Adaptive promotion rejected before radiation-stage commit: " + converted.reason);
        MaskResolvedOpacity(opacity, converted.mask, diffusion_opacity);
        diffusion_opacity.FillBoundary(geometry.periodicity());
        bool admissible = true;
        if (converted.promoted_cells > 0)
        {
            amrex::MultiFab zero_impulse(energy.boxArray(), energy.DistributionMap(), 3, 0);
            zero_impulse.setVal(0);
            ParticleImpulseTransaction frame;
            admissible = frame.Stage(*result.private_material, "diffusion_0", zero_impulse, true,
                                     options.material.assignment);
            if (admissible)
            {
                admissible = AdmissibleResolvedDiffusionCensus(
                    converted.energy, converted.momentum, diffusion_opacity, converted.mask,
                    frame.WorkVelocity(), geometry, dt, options.diffusion,
                    options.moving_diffusion);
            }
        }
        if (admissible)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(result.private_photons->AdoptCandidate(promotion),
                                             "Adaptive promotion lost its original private photon "
                                             "provenance.");
            result.private_radiation_energy = std::move(converted.energy);
            result.private_diffusion_momentum = std::move(converted.momentum);
            mask = std::move(converted.mask);
            adaptive_carry = std::move(converted.representation_carry);
            promoted_cells = converted.promoted_cells;
            deferred_cells = converted.deferred_cells;
        }
        else
        {
            // Keep the actual physical photons. A failed scalar predictor is
            // not authority to alter the source, relax its bounds, or lose E/q.
            deferred_cells = converted.promoted_cells + converted.deferred_cells;
        }
        amrex::GpuArray<amrex::Real, 4> final_packets{};
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            RZPhotonInventory(result.private_photons->Particles(), final_packets),
            "Invalid private endpoint photons in adaptive combined ledger.");
        amrex::GpuArray<amrex::Real, 4> final_fields{};
        final_fields =
            RZFieldInventory(result.private_radiation_energy, result.private_diffusion_momentum,
                             adaptive_carry, geometry);
        auto const next_material =
            RZMaterialInventory(particles, m_momentum_species, result.private_material.get());
        amrex::Real energy_residual = 0;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            AuditRZCombined(
                initial_packet_inventory, initial_field_inventory, final_packets, final_fields,
                old_material, next_material, result, geometry, options.energy_relative_tolerance,
                initial_packet_inventory[0] + final_packets[0] + energy.sum(0) +
                    result.private_radiation_energy.sum(0) + std::abs(result.actual_material_work) +
                    std::abs(result.material_energy_carry_change),
                energy_residual),
            "Adaptive RZ packet/FLD/actual native KE/carry/Pz/Lz/wall inventory rejected before "
            "publication.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(result.private_photons->ValidateLive(),
                                         "Live photons changed before adaptive joint commit.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(result.private_material->Commit(),
                                         "Live material changed before adaptive joint commit.");
        result.private_photons->Commit();
        amrex::MultiFab::Copy(energy, result.private_radiation_energy, 0, 0, 1, 0);
        energy.FillBoundary(geometry.periodicity());
        amrex::MultiFab::Copy(q, result.private_diffusion_momentum, 0, 0, 3, 0);
        q.FillBoundary(geometry.periodicity());
        PublishResolvedMask(mask, mask_field);
        mask_field.FillBoundary(geometry.periodicity());
        auto& accepted_carry = *fields.get(ResolvedRepresentationCarryFieldName, 0);
        amrex::MultiFab::Copy(accepted_carry, adaptive_carry, 0, 0, 4, accepted_carry.nGrowVect());
        result.raw_energy_residual = energy_residual;
    }
    // All remaining writes publish the already accepted interval. The outer PIC
    // macrostep is not a transactional object and is not retried by this wrapper.
    if (m_resolved_packet.time_dependent_opacity)
    {
        amrex::MultiFab::Copy(opacity_state, opacity, 0, 0, 1, opacity_state.nGrowVect());
        m_resolved_packet.opacity_time = opacity_time;
        if (usesResolvedMaterialOpacity())
        {
            amrex::MultiFab::Copy(*fields.get(ResolvedElectronDensityFieldName, 0), *ne, 0, 0, 1,
                                  ne->nGrowVect());
            amrex::MultiFab::Copy(*fields.get(ResolvedElectronTemperatureFieldName, 0), *te, 0, 0,
                                  1, te->nGrowVect());
        }
    }
    for (int d = 0; d < 3; ++d)
    {
        amrex::MultiFab::Copy(*fields.get(ResolvedMomentumFieldName(d), 0), q, d, 0, 1, 1);
    }
    if (result.admitted_four_inventory.ok())
    {
        amrex::MultiFab::Add(*fields.get(ResolvedAdmissionFieldName, 0),
                             result.admitted_four_inventory, 0, 0, 1, 0);
    }
    fields.get(FieldType::radiation_material_energy, 0)->setVal(0);
    auto& impulse = *fields.get(FieldType::radiation_material_momentum, 0);
    auto& work = *fields.get(FieldType::radiation_material_kinetic_energy, 0);
    impulse.setVal(0);
    work.setVal(0);
    amrex::MultiFab::Copy(impulse, result.actual_material_impulse_field, 0, 0, 3, 0);
    amrex::MultiFab::Copy(work, result.actual_material_work_field, 0, 0, 1, 0);
    m_last_numerical_energy_residual = result.raw_energy_residual;
    m_cumulative_numerical_energy_residual += result.raw_energy_residual;
    m_last_material_carry_energy_change = result.material_energy_carry_change;
    m_last_diffusion_boundary_momentum_loss = result.radiation_wall_momentum;
    m_last_streaming_boundary_momentum_loss = result.packet_wall_momentum;
    for (int d = 0; d < 3; ++d)
    {
        m_last_streaming_boundary_momentum_loss[d] +=
            result.escaped_packet_inventory[d + 1] / PhysConst::c;
    }
    m_last_boundary_energy_loss = result.escaped_packet_inventory[0];
    m_cumulative_boundary_energy_loss += m_last_boundary_energy_loss;
    m_last_diffusion_boundary_energy_loss = 0;
    m_last_streaming_boundary_energy_loss = result.escaped_packet_inventory[0];
    m_last_boundary_energy_injection = 0;
    m_last_particle_wall_momentum = {0, 0, 0};
    m_last_coupled_solve = {result.work_iterations, 1, 0, 0, 0, result.raw_energy_residual};
    ++m_resolved_packet.interval;
    if (m_resolved_packet.verbosity > 0)
    {
        amrex::Print() << "Resolved packet/FLD accepted interval=" << m_resolved_packet.interval
                       << " collisions=" << result.collision_events
                       << " work=" << result.actual_material_work
                       << " raw_energy_residual=" << result.raw_energy_residual
                       << " kinetic_front_cells=" << result.kinetic_front_cells
                       << " deferred_admissions=" << result.deferred_front_admissions
                       << " linear_tolerance=" << result.linear_tolerance << '\n';
        amrex::Print() << "Adaptive representation demoted=" << demoted_cells
                       << " promoted=" << promoted_cells << " deferred=" << deferred_cells << '\n';
    }
#endif
}
