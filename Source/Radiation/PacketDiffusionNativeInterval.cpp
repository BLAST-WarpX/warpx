/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "PacketDiffusionNativeInterval.H"

#include "ConstitutiveDiffusionMomentum.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FluxRelaxedPackets.H"
#include "MovingAsymptoticDiffusionField.H"
#include "NativePhotonPathReplay.H"
#if defined(WARPX_DIM_RZ)
#include "RZNativePhotonPathReplay.H"
#endif
#include "PacketDiffusionInterfacePhotons.H"
#include "ParticleImpulse.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "PhotonPacketTransaction.H"
#include "RZMomentGeometry.H"
#include "ResolvedDiffusionFront.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <utility>

namespace warpx::radiation
{
namespace
{
amrex::GpuArray<amrex::Real, 3> DiffusionWallImpulse (
    AsymptoticDiffusionFieldResult const& field, amrex::iMultiFab const& mask,
    amrex::Geometry const& geometry, amrex::Real dt)
{
    amrex::GpuArray<amrex::Real, 3> result{};
    amrex::Real volume = 1;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { volume *= geometry.CellSize(d); }
#if defined(WARPX_DIM_RZ)
    RZMomentMetric const metric(geometry);
#endif
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
#if defined(WARPX_DIM_RZ)
        // Only axial momentum is a global linear invariant in cylindrical coordinates.
        if (axis == 0) { continue; }
#endif
        if (geometry.isPeriodic(axis)) { continue; }
        int const low = geometry.Domain().smallEnd(axis);
        int const high = geometry.Domain().bigEnd(axis);
        auto const weight = dt * volume / geometry.CellSize(axis);
        amrex::ReduceOps<amrex::ReduceOpSum> ops;
        amrex::ReduceData<amrex::Real> data(ops);
        for (amrex::MFIter it(mask); it.isValid(); ++it) {
            auto const active = mask.const_array(it);
            auto const pressure = field.face_pressure[axis]->const_array(it);
            ops.eval(it.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                     -> amrex::GpuTuple<amrex::Real> {
                if (active(i,j,k) != 1) { return {0}; }
                int face[3] = {i,j,k};
                amrex::Real value = face[axis] == low ? -pressure(i,j,k) : 0;
                if (face[axis] == high) {
                    ++face[axis];
                    value += pressure(face[0],face[1],face[2]);
                }
#if defined(WARPX_DIM_RZ)
                return {weight * metric.Volume(i) / volume * value};
#else
                return {weight * value};
#endif
            });
        }
        auto value = amrex::get<0>(data.value());
        if (mask.local_size() == 0) { value = 0; }
        amrex::ParallelDescriptor::ReduceRealSum(value);
#if defined(WARPX_DIM_1D_Z)
        result[2] = value;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        result[axis == 0 ? 0 : 2] = value;
#else
        result[axis] = value;
#endif
    }
    return result;
}

[[maybe_unused]] void PrintConstitutiveState (
    amrex::MultiFab const& old_energy, amrex::MultiFab const& old_q,
    amrex::MultiFab const& new_energy, amrex::MultiFab const& new_q)
{
    amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> ops;
    amrex::ReduceData<amrex::Real, amrex::Real> data(ops);
    for (amrex::MFIter it(old_energy); it.isValid(); ++it) {
        auto const e0 = old_energy.const_array(it), e1 = new_energy.const_array(it);
        auto const q0 = old_q.const_array(it), q1 = new_q.const_array(it);
        ops.eval(it.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                 -> amrex::GpuTuple<amrex::Real, amrex::Real> {
            auto const norm0 = std::sqrt(q0(i,j,k,0)*q0(i,j,k,0) +
                q0(i,j,k,1)*q0(i,j,k,1) + q0(i,j,k,2)*q0(i,j,k,2));
            auto const norm1 = std::sqrt(q1(i,j,k,0)*q1(i,j,k,0) +
                q1(i,j,k,1)*q1(i,j,k,1) + q1(i,j,k,2)*q1(i,j,k,2));
            auto const infinity = std::numeric_limits<amrex::Real>::infinity();
            return {e0(i,j,k) > 0 ? norm0/e0(i,j,k) : (norm0 == 0 ? 0 : infinity),
                    e1(i,j,k) > 0 ? norm1/e1(i,j,k) : (norm1 == 0 ? 0 : infinity)};
        });
    }
    auto const value = data.value();
    amrex::Real ratio[2] = {amrex::get<0>(value), amrex::get<1>(value)};
    if (old_energy.local_size() == 0) { ratio[0] = 0; ratio[1] = 0; }
    amrex::ParallelDescriptor::ReduceRealMax(ratio, 2);
    amrex::MultiFab worst_cell_energy(old_energy.boxArray(), old_energy.DistributionMap(), 2, 0);
    auto const maximum_ratio = ratio[1];
    for (amrex::MFIter it(worst_cell_energy); it.isValid(); ++it) {
        auto const e0 = old_energy.const_array(it), e1 = new_energy.const_array(it);
        auto const q1 = new_q.const_array(it);
        auto const selected = worst_cell_energy.array(it);
        amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto const norm = std::sqrt(q1(i,j,k,0)*q1(i,j,k,0) +
                q1(i,j,k,1)*q1(i,j,k,1) + q1(i,j,k,2)*q1(i,j,k,2));
            bool const worst = e1(i,j,k) > 0 && norm/e1(i,j,k) == maximum_ratio;
            selected(i,j,k,0) = worst ? e0(i,j,k) : 0;
            selected(i,j,k,1) = worst ? e1(i,j,k) : 0;
        });
    }
    amrex::Print() << "Native constitutive diagnostic max_old_q_over_E=" << ratio[0]
        << " max_new_q_over_E=" << ratio[1] << " min_old_E=" << old_energy.min(0)
        << " min_new_E=" << new_energy.min(0)
        << " worst_cells_old_E=" << worst_cell_energy.sum(0)
        << " worst_cells_new_E=" << worst_cell_energy.sum(1) << '\n';
}

#if defined(WARPX_DIM_RZ)
constexpr int face_inventory_components = 5;
constexpr auto MakeLeakagePlan = BuildRZPacketDiffusionLeakageEmissionPlan;
#else
constexpr int face_inventory_components = 4;
constexpr auto MakeLeakagePlan = BuildPacketDiffusionLeakageEmissionPlan;
#endif
#if defined(WARPX_DIM_RZ)
using PhysicalPathReplay = RZNativePhotonPathReplay;
using PhysicalPathSubsegmentReplay = RZNativePhotonPathSubsegmentReplay;
bool SourcePhotonInventory (WarpXParticleContainer::Base& photons,
                            amrex::GpuArray<amrex::Real,4>& inventory)
{
    if (!detail::PhotonInventory(photons, inventory)) { return false; }
    amrex::ReduceOps<amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Real> data(ops);
    for (WarpXParIter it(photons,0); it.isValid(); ++it) {
        auto const p = it.GetParticleTile().getParticleTileData();
        auto const position = GetParticlePosition<PIdx>(it);
        ops.eval(it.numParticles(), data, [=] AMREX_GPU_DEVICE(long n)
                 -> amrex::GpuTuple<amrex::Real> {
            if (!amrex::ParticleIDWrapper{p.m_idcpu[n]}.is_valid()) { return {0}; }
            amrex::ParticleReal x=0,y=0,z=0;
            position(n,x,y,z);
            return {p.m_rdata[PIdx::w][n]*PhysConst::m_e*PhysConst::c*
                (x*p.m_rdata[PIdx::uy][n]-y*p.m_rdata[PIdx::ux][n])};
        });
    }
    auto angular = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(angular);
    inventory[1] = 0; // No global radial conservation law.
    inventory[2] = angular; // c*Lz, J m (not Cartesian c*py).
    return std::isfinite(angular);
}
amrex::Real AngularFieldSum (amrex::MultiFab const& field, amrex::Geometry const& geometry,
                            int component = 1)
{
    RZMomentMetric const metric(geometry);
    amrex::ReduceOps<amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Real> data(ops);
    for (amrex::MFIter it(field); it.isValid(); ++it) {
        auto const q = field.const_array(it);
        ops.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)
                 -> amrex::GpuTuple<amrex::Real> {
            return {q(i,j,k,component)*metric.Angular(i).mean_radius};
        });
    }
    auto result = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(result);
    return result;
}
#else
using PhysicalPathReplay = NativePhotonPathReplay;
using PhysicalPathSubsegmentReplay = NativePhotonPathSubsegmentReplay;
bool SourcePhotonInventory (WarpXParticleContainer::Base& photons,
                            amrex::GpuArray<amrex::Real,4>& inventory)
{
    return detail::PhotonInventory(photons, inventory);
}
#endif

[[maybe_unused]] bool
StorageOverlaps (amrex::MultiFab const& first, amrex::MultiFab const& second)
{
    for (amrex::MFIter it(first); it.isValid(); ++it) {
        auto const& a = first[it];
        auto const& b = second[it];
        auto const begin_a = reinterpret_cast<std::uintptr_t>(a.dataPtr());
        auto const begin_b = reinterpret_cast<std::uintptr_t>(b.dataPtr());
        auto const bytes_a = static_cast<std::uintptr_t>(a.box().numPts())*a.nComp()*sizeof(amrex::Real);
        auto const bytes_b = static_cast<std::uintptr_t>(b.box().numPts())*b.nComp()*sizeof(amrex::Real);
        auto const maximum = std::numeric_limits<std::uintptr_t>::max();
        if (!begin_a || !begin_b || bytes_a > maximum-begin_a || bytes_b > maximum-begin_b ||
            (begin_a < begin_b+bytes_b && begin_b < begin_a+bytes_a)) { return true; }
    }
    return false;
}

[[maybe_unused]] bool
SameOnAllRanks (std::string const& value)
{
    int size = static_cast<int>(value.size());
    amrex::ParallelDescriptor::Bcast(&size, 1, amrex::ParallelDescriptor::IOProcessorNumber());
    std::string reference = value;
    reference.resize(size);
    amrex::ParallelDescriptor::Bcast(reference.data(), size,
                                    amrex::ParallelDescriptor::IOProcessorNumber());
    bool same = reference == value;
    amrex::ParallelDescriptor::ReduceBoolAnd(same);
    return same;
}
}

PacketDiffusionNativeIntervalResult
TryAdvancePacketDiffusionNativeInterval (
    MultiParticleContainer& particles, WarpXParticleContainer& photons,
    std::vector<std::string> const& material_species, std::string const& impulse_path,
    amrex::MultiFab& radiation_energy, amrex::MultiFab const& opacity,
    amrex::iMultiFab const& diffusion_mask, amrex::Geometry const& geometry,
    amrex::Real dt, amrex::MultiFab const* old_radiation_inertia,
    PacketDiffusionNativeIntervalOptions const& options,
    amrex::MultiFab* conserved_diffusion_momentum)
{
    PacketDiffusionNativeIntervalResult result;
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    amrex::ignore_unused(particles, photons, material_species, impulse_path,
        radiation_energy, opacity, diffusion_mask, geometry, dt,
        old_radiation_inertia, options, conserved_diffusion_momentum);
    result.reason = "Native mixed-interface source does not support radial geometry";
    return result;
#else
    auto const& simulation = WarpX::GetInstance();
    auto const& native_geometry = simulation.Geom(0);
    auto const& boxes = radiation_energy.boxArray();
    auto const& distribution = radiation_energy.DistributionMap();
    bool const coordinates_valid =
#if defined(WARPX_DIM_RZ)
        (geometry.Coord() == 0 || geometry.Coord() == 1) && geometry.ProbLo(0) >= 0 &&
        !geometry.isPeriodic(0) && !native_geometry.isPeriodic(0) &&
        options.resolved_collided_source && !options.collisions.moving_cross_sections &&
        options.material.assignment == ParticleImpulseAssignment::RZAngularConservative;
#else
        geometry.Coord() == 0 && native_geometry.Coord() == 0 &&
        options.material.assignment == ParticleImpulseAssignment::NearestCell;
#endif
    bool valid =
        coordinates_valid && simulation.finestLevel() == 0 && !EB::enabled() &&
        std::numeric_limits<amrex::Real>::digits >= 53 &&
        std::numeric_limits<amrex::ParticleReal>::digits >= 53 &&
        geometry.Domain() == native_geometry.Domain() && std::isfinite(dt) &&
        dt > 0 && boxes == photons.ParticleBoxArray(0) &&
        distribution == photons.ParticleDistributionMap(0) &&
        radiation_energy.nComp() == 1 &&
        radiation_energy.ixType().cellCentered() &&
        opacity.boxArray() == boxes &&
        opacity.DistributionMap() == distribution && opacity.nComp() == 1 &&
        opacity.ixType().cellCentered() && diffusion_mask.boxArray() == boxes &&
        diffusion_mask.DistributionMap() == distribution &&
        diffusion_mask.nComp() == 1 && diffusion_mask.ixType().cellCentered() &&
        !material_species.empty() && !impulse_path.empty() &&
        std::set<std::string>(material_species.begin(), material_species.end())
                .size() == material_species.size() &&
        std::isfinite(options.photon_energy) && options.photon_energy > 0 &&
        std::isfinite(options.energy_relative_tolerance) &&
        options.energy_relative_tolerance > 0 &&
        options.energy_relative_tolerance <= amrex::Real(1.e-10) &&
        !options.collisions.absorption && !options.collisions.emission &&
        (!options.collisions.moving_cross_sections ||
         options.kinetic_scattering_opacity) &&
        (!options.transactional_packet_substeps ||
         options.resolved_collided_source) &&
        (!options.kinetic_empty_cells || options.resolved_collided_source) &&
        (!options.private_photon_snapshot ||
         (options.resolved_collided_source && options.defer_live_commit));
#if defined(WARPX_DIM_RZ)
    valid = valid &&
        (!options.private_photon_snapshot || options.adaptive_rz_private_snapshot) &&
        (!options.adaptive_rz_private_snapshot ||
         (options.private_photon_snapshot && options.radiation_representation_carry &&
          options.defer_live_commit && options.resolved_collided_source &&
          options.retain_constitutive_momentum &&
          WarpX::n_rz_azimuthal_modes == 1)) &&
        (!options.radiation_representation_carry || options.adaptive_rz_private_snapshot);
    if (options.adaptive_rz_private_snapshot) {
        valid = valid && WarpX::particle_boundary_hi[0] == ParticleBoundaryType::Reflecting &&
            WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC;
        valid = valid &&
            (geometry.ProbLo(0) == 0
                 ? WarpX::particle_boundary_lo[0] == ParticleBoundaryType::None &&
                       WarpX::field_boundary_lo[0] == FieldBoundaryType::None
                 : WarpX::particle_boundary_lo[0] == ParticleBoundaryType::Reflecting &&
                       WarpX::field_boundary_lo[0] == FieldBoundaryType::PEC);
    }
#else
    valid = valid && !options.adaptive_rz_private_snapshot &&
        !options.radiation_representation_carry;
#endif
    if (options.radiation_representation_carry) {
        auto const& carry = *options.radiation_representation_carry;
        valid = valid && carry.nComp() == 4 && carry.ixType().cellCentered() &&
            carry.boxArray() == boxes && carry.DistributionMap() == distribution;
    }
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        valid = valid && geometry.ProbLo(d) == native_geometry.ProbLo(d) &&
            geometry.ProbHi(d) == native_geometry.ProbHi(d) &&
            geometry.isPeriodic(d) == native_geometry.isPeriodic(d);
#if defined(WARPX_DIM_RZ)
        if (d == 0) { continue; }
#endif
        bool const periodic = geometry.isPeriodic(d);
        auto const expected_particle = periodic ? ParticleBoundaryType::Periodic
                                                : ParticleBoundaryType::Reflecting;
        valid = valid && WarpX::particle_boundary_lo[d] == expected_particle &&
            WarpX::particle_boundary_hi[d] == expected_particle &&
            (periodic ||
             (WarpX::field_boundary_lo[d] == FieldBoundaryType::PEC &&
              WarpX::field_boundary_hi[d] == FieldBoundaryType::PEC));
    }
    valid = valid && (!old_radiation_inertia ||
        (old_radiation_inertia->nComp() == 3 && old_radiation_inertia->ixType().cellCentered() &&
         old_radiation_inertia->boxArray() == boxes &&
         old_radiation_inertia->DistributionMap() == distribution));
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) {
        result.reason = "Unsupported native mixed-interface options";
        return result;
    }
    std::string species_contract = impulse_path;
    for (auto const& name : material_species) { species_contract += '\n' + name; }
    valid = SameOnAllRanks(species_contract);
    int min_modes[13] = {options.kinetic_scattering_opacity ? 1 : 0,
                         options.moving_diffusion ? 1 : 0,
                         options.collisions.moving_cross_sections ? 1 : 0,
                         options.quasi_static_angular_traction ? 1 : 0,
                         options.retain_constitutive_momentum ? 1 : 0,
                         conserved_diffusion_momentum ? 1 : 0,
                         options.resolved_collided_source ? 1 : 0,
                         options.transactional_packet_substeps ? 1 : 0,
                         options.kinetic_empty_cells ? 1 : 0,
                         options.defer_live_commit ? 1 : 0,
                         options.private_photon_snapshot ? 1 : 0,
                         options.adaptive_rz_private_snapshot ? 1 : 0,
                         options.radiation_representation_carry ? 1 : 0};
    int max_modes[13] = {min_modes[0], min_modes[1], min_modes[2], min_modes[3],
                         min_modes[4], min_modes[5], min_modes[6], min_modes[7],
                         min_modes[8], min_modes[9], min_modes[10], min_modes[11], min_modes[12]};
    amrex::ParallelDescriptor::ReduceIntMin(min_modes, 13);
    amrex::ParallelDescriptor::ReduceIntMax(max_modes, 13);
    valid = valid && min_modes[0] == max_modes[0] &&
            min_modes[1] == max_modes[1] && min_modes[2] == max_modes[2] &&
            min_modes[3] == max_modes[3] && min_modes[4] == max_modes[4] &&
            min_modes[5] == max_modes[5] && min_modes[6] == max_modes[6] &&
            min_modes[7] == max_modes[7] && min_modes[8] == max_modes[8] &&
            min_modes[9] == max_modes[9] && min_modes[10] == max_modes[10] &&
            min_modes[11] == max_modes[11] && min_modes[12] == max_modes[12];
    int subsegment_limits[2] = {
        options.packet_subsegment_limit, options.packet_subsegment_limit};
    amrex::ParallelDescriptor::ReduceIntMin(subsegment_limits[0]);
    amrex::ParallelDescriptor::ReduceIntMax(subsegment_limits[1]);
    valid = valid && options.packet_subsegment_limit > 0 &&
        subsegment_limits[0] == subsegment_limits[1];
    amrex::Real minimum[5] = {options.photon_energy, options.energy_relative_tolerance,
        options.quasi_static_maximum_beta, options.quasi_static_minimum_optical_depth,
        options.minimum_constitutive_relaxation_depth};
    amrex::Real maximum[5] = {minimum[0], minimum[1], minimum[2], minimum[3], minimum[4]};
    amrex::ParallelDescriptor::ReduceRealMin(minimum, 5);
    amrex::ParallelDescriptor::ReduceRealMax(maximum, 5);
    for (int n = 0; n < 5; ++n) {
        valid = valid && std::isfinite(minimum[n]) && minimum[n] == maximum[n];
    }
    if (!valid) {
        result.reason = "Native interval contract differs between ranks";
        return result;
    }
    amrex::MultiFab representation_carry;
    if (options.radiation_representation_carry) {
        auto const& carry = *options.radiation_representation_carry;
        if (!carry.is_finite(0, 4, carry.nGrowVect())) {
            result.reason = "Invalid signed radiation representation carry";
            return result;
        }
        representation_carry.define(boxes, distribution, 4, carry.nGrowVect());
        amrex::MultiFab::Copy(representation_carry, carry, 0, 0, 4, carry.nGrowVect());
    }
    // Check the original request before tightening it: a cap must not sanitize
    // invalid controls or conceal different per-rank requests.
    amrex::Real accuracy_min[3] = {options.diffusion.linear_tolerance,
        options.diffusion.residual_tolerance, options.material.work_relative_tolerance};
    amrex::Real accuracy_max[3] = {accuracy_min[0], accuracy_min[1], accuracy_min[2]};
    for (auto const tolerance : accuracy_min) {
        valid = valid && std::isfinite(tolerance) && tolerance > 0;
    }
    valid = valid && accuracy_min[0] <= amrex::Real(1.e-11) &&
        accuracy_min[1] <= amrex::Real(1.e-11);
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) {
        result.reason = "Invalid native diffusion/work accuracy controls";
        return result;
    }
    amrex::ParallelDescriptor::ReduceRealMin(accuracy_min, 3);
    amrex::ParallelDescriptor::ReduceRealMax(accuracy_max, 3);
    for (int n = 0; n < 3; ++n) { valid = valid && accuracy_min[n] == accuracy_max[n]; }
    auto diffusion_options = options.diffusion;
    auto const outer_accuracy = std::min({options.material.work_relative_tolerance,
        options.diffusion.residual_tolerance, options.energy_relative_tolerance});
    diffusion_options.linear_tolerance = std::min(
        options.diffusion.linear_tolerance, amrex::Real(0.01)*outer_accuracy);
    valid = valid && diffusion_options.linear_tolerance > 0;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) {
        result.reason = "Inconsistent or unrepresentable native accuracy budget";
        return result;
    }
    result.linear_tolerance = diffusion_options.linear_tolerance;
    valid = options.retain_constitutive_momentum == (conserved_diffusion_momentum != nullptr);
    if (conserved_diffusion_momentum) {
        valid = valid && options.quasi_static_angular_traction &&
            options.minimum_constitutive_relaxation_depth >= 8 &&
            conserved_diffusion_momentum != old_radiation_inertia &&
            conserved_diffusion_momentum->ok() &&
            conserved_diffusion_momentum->nComp() == 3 &&
            conserved_diffusion_momentum->ixType().cellCentered() &&
            conserved_diffusion_momentum->boxArray() == boxes &&
            conserved_diffusion_momentum->DistributionMap() == distribution;
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) {
        result.reason = "Missing, aliased, or incompatible constitutive momentum state";
        return result;
    }
    if (conserved_diffusion_momentum) {
        valid = !StorageOverlaps(*conserved_diffusion_momentum, radiation_energy) &&
            !StorageOverlaps(*conserved_diffusion_momentum, opacity) &&
            (!old_radiation_inertia ||
             !StorageOverlaps(*conserved_diffusion_momentum, *old_radiation_inertia));
        if (options.kinetic_scattering_opacity &&
            options.kinetic_scattering_opacity->boxArray() == boxes &&
            options.kinetic_scattering_opacity->DistributionMap() == distribution) {
            valid = valid && !StorageOverlaps(*conserved_diffusion_momentum,
                                             *options.kinetic_scattering_opacity);
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if (!valid) {
            result.reason = "Constitutive momentum storage overlaps another interval field";
            return result;
        }
    }
    bool const all_diffusion = diffusion_mask.min(0) == 1 && diffusion_mask.max(0) == 1;
    // A kinetic collar owns radiation escape. In particular an empty scalar
    // label is insufficient: it might acquire D energy later in this interval.
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        int low[2] = {options.photon_boundaries.escape_lo[d],
                      options.photon_boundaries.escape_hi[d]};
        int high[2] = {low[0], low[1]};
        amrex::ParallelDescriptor::ReduceIntMin(low, 2);
        amrex::ParallelDescriptor::ReduceIntMax(high, 2);
        for (int side = 0; side < 2; ++side) {
            valid = valid && low[side] == high[side] && low[side] >= 0 && high[side] <= 1;
            if (high[side] == 0) { continue; }
#if defined(WARPX_DIM_RZ)
            valid = false;
#else
            valid = valid && options.resolved_collided_source && !geometry.isPeriodic(d);
            auto edge = geometry.Domain();
            int const index = side == 0 ? edge.smallEnd(d) : edge.bigEnd(d);
            edge.setSmall(d, index);
            edge.setBig(d, index);
            valid = valid && diffusion_mask.max(edge, 0) == 0 && diffusion_mask.min(edge, 0) == 0;
#endif
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) {
        result.reason = "Photon escape requires consistent Cartesian faces with kinetic collars";
        return result;
    }
    valid = (!options.moving_diffusion || all_diffusion ||
             options.resolved_collided_source) &&
            (!all_diffusion || options.resolved_collided_source ||
             photons.TotalNumberOfParticles() == 0);
#if !defined(WARPX_DIM_1D_Z)
    valid = valid && (all_diffusion || options.resolved_collided_source);
#endif
    if (!valid) {
        result.reason = "Unsupported mixed moving interface or directed photons in diffusion cells";
        return result;
    }
    if (options.quasi_static_angular_traction) {
        valid =
            (!all_diffusion || options.resolved_collided_source) &&
            (!options.moving_diffusion || options.resolved_collided_source) &&
            options.quasi_static_maximum_beta > 0 &&
            options.quasi_static_maximum_beta <= 1.e-3 &&
            options.quasi_static_minimum_optical_depth >= 8;
        amrex::MultiFab admissibility(boxes, distribution, 1, 0);
        auto const min_tau = options.quasi_static_minimum_optical_depth;
        auto const retain_q = options.retain_constitutive_momentum;
        auto const allow_cold = options.resolved_collided_source;
        auto const min_relaxation = options.minimum_constitutive_relaxation_depth;
        auto dx = geometry.CellSize(0);
        for (int d = 1; d < AMREX_SPACEDIM; ++d) {
            dx = std::min(dx, geometry.CellSize(d));
        }
        for (amrex::MFIter mfi(admissibility); mfi.isValid(); ++mfi) {
            auto const mask = diffusion_mask.const_array(mfi);
            auto const chi = opacity.const_array(mfi);
            auto const energy = radiation_energy.const_array(mfi);
            auto const ok = admissibility.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                ok(i,j,k) = mask(i,j,k) == 0 ||
                    (mask(i,j,k) == 1 && (energy(i,j,k) > 0 || (allow_cold && energy(i,j,k) == 0)) &&
                     chi(i,j,k)*dx >= min_tau &&
                     (!retain_q || chi(i,j,k)*PhysConst::c*dt >= min_relaxation)) ? 1 : 0;
            });
        }
        valid =
            valid && admissibility.min(0) == 1 &&
            (options.resolved_collided_source || diffusion_mask.max(0) == 1);
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if (!valid) {
            result.reason = "Quasi-static traction requires an opaque mixed positive reservoir";
            return result;
        }
    }
    CollidedDiffusionAdmissionComponents admission_components;
    bool require_birth_keys = false;
    if (options.resolved_collided_source) {
        valid = options.retain_constitutive_momentum && options.quasi_static_angular_traction &&
            options.kinetic_scattering_opacity != nullptr;
        auto const& names = photons.GetRealSoANames();
        std::array<int, 2> key_components{-1, -1};
        valid = valid && FindPhotonDrawKeyComponents(names, key_components);
        require_birth_keys = key_components[0] >= 0;
        valid = valid && (!options.logical_birth_keys || require_birth_keys);
        for (int d = 0; d < 3; ++d) {
            std::string const name = std::string("radiation_diffusion_census_") +
                (d == 0 ? "ux0" : d == 1 ? "uy0" : "uz0");
            auto const found = std::find(names.begin(), names.end(), name);
            valid = valid && found != names.end();
            if (found != names.end()) { admission_components.initial_u[d] = photons.GetRealCompIndex(name); }
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if (!valid) {
            result.reason = "Resolved collided source requires registered photon census and full opacity";
            return result;
        }
    }
    // Neighbor tests and face births use canonical private ghosts, never
    // caller-owned ghost values (which may be stale or absent).
    amrex::iMultiFab transport_mask(boxes, distribution, 1, 1);
    transport_mask.setVal(1); // Reflecting radial exterior in RZ.
    amrex::iMultiFab::Copy(transport_mask, diffusion_mask, 0, 0, 1, 0);
    std::unique_ptr<amrex::MultiFab> front_opacity;
    if (options.kinetic_empty_cells) {
        auto front = BuildResolvedDiffusionFront(radiation_energy,
            *conserved_diffusion_momentum, opacity, diffusion_mask, geometry);
        transport_mask = std::move(front.mask);
        front_opacity = std::make_unique<amrex::MultiFab>(std::move(front.opacity));
        result.kinetic_front_cells = front.kinetic_cells;
    }
    transport_mask.FillBoundary(geometry.periodicity());
    auto const& scalar_opacity = front_opacity ? *front_opacity : opacity;
    auto const available_species = particles.GetSpeciesNames();
    for (auto const& name : material_species) {
        if (std::find(available_species.begin(), available_species.end(), name) ==
            available_species.end()) {
            valid = false;
            continue;
        }
        auto const& species = particles.GetParticleContainerFromName(name);
        valid = valid && &species != &photons && std::isfinite(species.getMass()) &&
            species.getMass() > 0 && species.ParticleBoxArray(0) == boxes &&
            species.ParticleDistributionMap(0) == distribution;
        auto const& attributes = species.GetRealSoANames();
        for (auto const* suffix : {"_ux", "_uy", "_uz", "_work"}) {
            std::string const attribute = "radiation_impulse_" + impulse_path + suffix;
            valid = valid && std::find(attributes.begin(), attributes.end(), attribute) !=
                attributes.end();
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) {
        result.reason = "Invalid, photon, or incompatible recoiling material species";
        return result;
    }

    std::unique_ptr<PhotonPacketTransaction> prepared_photons;
    if (options.resolved_collided_source) {
        if (options.private_photon_snapshot &&
            !options.private_photon_snapshot->ValidateSnapshotFor(photons)) {
            result.reason = "Invalid private interval-start photon provenance";
            return result;
        }
        prepared_photons =
            options.private_photon_snapshot
                ? std::make_unique<PhotonPacketTransaction>(
                      photons, *options.private_photon_snapshot)
                : std::make_unique<PhotonPacketTransaction>(photons);
    }
    amrex::GpuArray<amrex::Real, 4> initial_photons{};
    auto& source_photons =
        prepared_photons ? prepared_photons->Particles() : photons;
    if (!SourcePhotonInventory(source_photons, initial_photons)) {
        result.reason = "Invalid initial photons";
        return result;
    }
    auto material = std::make_unique<ParticleImpulseMaterial>(particles, material_species);
    // Capture the initial material frame before sampling any collision events.
    // The same frozen plan is replayed for every later finite-kick secant.
    amrex::MultiFab initial_impulse(boxes, distribution, 3, 0);
    amrex::MultiFab initial_debit(boxes, distribution, 1, 0);
    amrex::MultiFab initial_scale(boxes, distribution, 1, 0);
    initial_impulse.setVal(0);
    ParticleImpulseTransaction initial_work;
    if (!initial_work.Stage(*material, impulse_path, initial_impulse, true,
                            options.material.assignment)) {
        result.reason = "Initial native material frame rejected";
        return result;
    }
    std::unique_ptr<NativePhotonCollisionReplay> collisions;
    std::unique_ptr<PhysicalPathReplay> physical_path;
    std::unique_ptr<PhysicalPathSubsegmentReplay> physical_subsegments;
    NativePhotonPathReport path_report;
    NativePhotonCollisionReport collision_report;
    auto const consistent_active_opacity = [&](amrex::MultiFab const& energy) {
        amrex::MultiFab compatible(boxes, distribution, 1, 0);
        for (amrex::MFIter it(compatible); it.isValid(); ++it) {
            auto const e = energy.const_array(it);
            auto const mask = diffusion_mask.const_array(it);
            auto const scalar_chi = opacity.const_array(it);
            auto const packet_chi = options.kinetic_scattering_opacity->const_array(it);
            auto const ok = compatible.array(it);
            amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                ok(i,j,k) = mask(i,j,k) != 1 || e(i,j,k) == 0 ||
                    scalar_chi(i,j,k) == packet_chi(i,j,k) ? 1 : 0;
            });
        }
        return compatible.min(0) == 1;
    };
    if (options.resolved_collided_source) {
        auto const snapshot_report = SnapshotCollidedDiffusionAdmission(
            prepared_photons->Particles(), admission_components);
        if (!snapshot_report.valid) {
            result.reason = "Invalid physical photon census snapshot";
            return result;
        }
        if (options.transactional_packet_substeps) {
            physical_subsegments =
                std::make_unique<PhysicalPathSubsegmentReplay>();
            bool prepared = false;
            if (options.private_photon_snapshot) {
                prepared = physical_subsegments->Prepare(
                    prepared_photons->Particles(), photons,
                    *options.kinetic_scattering_opacity,
                    initial_work.WorkVelocity(), geometry, dt,
                    options.collisions, options.packet_subsegment_limit,
                    path_report,
#if !defined(WARPX_DIM_RZ)
                    options.photon_boundaries,
#endif
                    NativePhotonPathSnapshotPolicy::KeyedProvisional);
            } else
            {
                prepared = physical_subsegments->Prepare(
                    photons, *options.kinetic_scattering_opacity,
                    initial_work.WorkVelocity(), geometry, dt,
                    options.collisions, options.packet_subsegment_limit,
                    path_report
#if !defined(WARPX_DIM_RZ)
                    ,
                    options.photon_boundaries
#endif
                );
            }
            if (!prepared) {
                result.reason = "Physical packet subsegment preparation: " +
                    path_report.reason;
                return result;
            }
        } else {
            physical_path = std::make_unique<PhysicalPathReplay>();
            if (!physical_path->Prepare(
                    prepared_photons->Particles(), photons,
                    *options.kinetic_scattering_opacity,
                    initial_work.WorkVelocity(), geometry, dt,
                    options.collisions, path_report
#if !defined(WARPX_DIM_RZ)
                    ,
                    options.photon_boundaries
#endif
                    ,
                    options.private_photon_snapshot
                        ? NativePhotonPathSnapshotPolicy::KeyedProvisional
                        : NativePhotonPathSnapshotPolicy::NativeOnly
                    )) {
                result.reason = "Physical collar preparation: " + path_report.reason;
                return result;
            }
        }
        if (!consistent_active_opacity(radiation_energy)) {
            result.reason = "Active diffuse and packet populations require identical transport opacity";
            return result;
        }
    } else if (options.kinetic_scattering_opacity) {
        collisions = std::make_unique<NativePhotonCollisionReplay>();
        bool const prepared = options.collisions.moving_cross_sections ?
            collisions->Prepare(photons, diffusion_mask, *options.kinetic_scattering_opacity,
                geometry, dt, initial_work.WorkVelocity(), options.collisions, collision_report) :
            collisions->Prepare(photons, diffusion_mask, *options.kinetic_scattering_opacity,
                geometry, dt, options.collisions, collision_report);
        if (!prepared) {
            result.reason = "Kinetic collision plan: " + collision_report.reason;
            return result;
        }
    }
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> admissions;
#if defined(WARPX_DIM_RZ)
    RZMomentMetric const metric(geometry);
#endif
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> admitted{};
    amrex::Real volume = 1;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> area{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { volume *= geometry.CellSize(d); }
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        auto const faces = amrex::convert(boxes, amrex::IntVect::TheDimensionVector(d));
        admissions[d] = std::make_unique<amrex::MultiFab>(faces, distribution, 1, 1);
        admitted[d] = admissions[d].get();
        area[d] = volume / geometry.CellSize(d);
    }
    PacketDiffusionInterfacePhotonControls packet_options;
    packet_options.marshak_extrapolation_lambda = options.diffusion.marshak_extrapolation_lambda;
    amrex::MultiFab diffusion_work(boxes, distribution, 1, 0);
    amrex::MultiFab collision_impulse(boxes, distribution, 3, 0);
    amrex::MultiFab collision_debit(boxes, distribution, 1, 0);
    amrex::MultiFab collision_scale(boxes, distribution, 1, 0);
    // With no old diffuse population there can be no diffuse work. Solve the
    // actual packet/material fixed point directly and admit its census only
    // afterwards. A nonconverged total-minus-packet work residual must not be
    // borrowed from a nonexistent scalar reservoir.
    bool const empty_diffuse_population = options.resolved_collided_source &&
        radiation_energy.norm0(0) == 0 &&
        conserved_diffusion_momentum->norm0(0, 3, amrex::IntVect::TheZeroVector()) == 0;
    amrex::MultiFab crossing_inventory;
    if (!all_diffusion) {
        crossing_inventory.define(admissions[0]->boxArray(), distribution, 8, 1);
    }
    std::unique_ptr<PhotonPacketTransaction> photon_trial;
    std::unique_ptr<AsymptoticDiffusionFieldResult> accepted_field;
    std::unique_ptr<amrex::MultiFab> accepted_lab_inertia;
    amrex::GpuArray<amrex::Real, 3> accepted_radiation_wall_momentum{};
    amrex::GpuArray<amrex::Real, 3> accepted_tangential_defect{};
    auto const replay = [&](ParticleImpulseTransaction const& staged,
                            amrex::MultiFab& next_impulse, amrex::MultiFab& debit,
                            amrex::MultiFab& scale) -> bool
    {
        if (options.quasi_static_angular_traction) {
            auto const& velocity = staged.WorkVelocity();
#if defined(WARPX_DIM_RZ)
            amrex::MultiFab speed(boxes, distribution, 1, 0);
            for (amrex::MFIter it(speed); it.isValid(); ++it) {
                auto const v = velocity.const_array(it);
                auto const s = speed.array(it);
                amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    auto const theta = v(i,j,k,1)*metric.Radius(i+1)/metric.Angular(i).mean_radius;
                    s(i,j,k) = std::sqrt(v(i,j,k,0)*v(i,j,k,0) + theta*theta +
                                        v(i,j,k,2)*v(i,j,k,2));
                });
            }
            if (!speed.is_finite()) { return false; }
            auto const bound = speed.norm0(0);
#else
            auto const bound = std::hypot(velocity.norm0(0), velocity.norm0(1), velocity.norm0(2));
#endif
            if (!std::isfinite(bound) || bound > options.quasi_static_maximum_beta*PhysConst::c) {
                return false;
            }
        }
        // Restart from the unchanged interval-start photons on every replay.
        // The finite-mass frame changes both physical collision energies and
        // directions, so interface admissions must also be recomputed.
        auto replay_photons =
            options.private_photon_snapshot
                ? std::make_unique<PhotonPacketTransaction>(photons,
                                                            *prepared_photons)
                : std::make_unique<PhotonPacketTransaction>(photons);
        collision_impulse.setVal(0);
        collision_debit.setVal(0);
        collision_scale.setVal(0);
        if (physical_path) {
            replay_photons->Particles().copyParticles(prepared_photons->Particles(), true);
            if (!physical_path->Replay(staged.WorkVelocity(), replay_photons->Particles(),
                    collision_impulse, collision_debit, collision_scale, path_report)) { return false; }
            collision_report = path_report;
        } else if (physical_subsegments) {
            replay_photons->Particles().copyParticles(prepared_photons->Particles(), true);
            NativePhotonPathSubsegmentReport subsegment_report;
            if (!physical_subsegments->Replay(staged.WorkVelocity(),
                    replay_photons->Particles(), collision_impulse, collision_debit,
                    collision_scale, subsegment_report)) {
                path_report = subsegment_report.path;
                if (options.diffusion.verbosity > 0) {
                    amrex::Print() << "Native packet subsegment replay failure="
                                   << path_report.reason << '\n';
                }
                return false;
            }
            path_report = subsegment_report.path;
            collision_report = path_report;
        }
        if (collisions && !collisions->Replay(staged.WorkVelocity(), replay_photons->Particles(),
                collision_impulse, collision_debit, collision_scale, collision_report)) {
            return false;
        }
        for (auto& face : admissions) { face->setVal(0); }
        detail::EmissionPlan returns;
        PacketDiffusionInterfacePhotonReport packet_report;
        if (!all_diffusion && !options.resolved_collided_source &&
            (!AdvancePacketDiffusionCandidatePhotons(replay_photons->Particles(), diffusion_mask,
                opacity, geometry, dt, *admissions[0], returns, packet_report, packet_options,
                &crossing_inventory) ||
             !detail::StageEmissionPlan(*replay_photons, returns))) {
            return false;
        }
        if (options.resolved_collided_source) { crossing_inventory.setVal(0); }
        // The scalar D state carries no incoming transverse first moment.
        // Report that model defect; do not invent a numerical carry or kick.
        for (int d = 0; d < 2; ++d) {
            accepted_tangential_defect[d] = all_diffusion ? 0 :
                (packet_report.final_inventory[d+1] + packet_report.reflected_inventory[d+1] -
                 packet_report.initial_inventory[d+1]) / PhysConst::c;
        }
        for (amrex::MFIter mfi(diffusion_work); mfi.isValid(); ++mfi) {
            auto const mask = transport_mask.const_array(mfi);
            auto const requested = staged.RequestedWork().const_array(mfi);
            auto const packet_work = collision_debit.const_array(mfi);
            auto const resolved = options.resolved_collided_source;
            auto const empty_diffuse = empty_diffuse_population;
            auto const work = diffusion_work.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                work(i,j,k) = mask(i,j,k) == 1 && !empty_diffuse ?
                    requested(i,j,k) - (resolved ? packet_work(i,j,k) : 0) : 0;
            });
        }
        AsymptoticDiffusionFieldResult trial_field;
        MovingAsymptoticDiffusionFieldResult moving_field;
        if (options.moving_diffusion) {
            moving_field = TryAdvanceMovingAsymptoticDiffusionField(radiation_energy, scalar_opacity,
                transport_mask, staged.WorkVelocity(), geometry, dt, diffusion_work,
                diffusion_options, options.resolved_collided_source);
            if (!moving_field.valid) {
                if (options.diffusion.verbosity > 0) {
                    amrex::Print() << "Native moving field replay failure="
                                   << static_cast<int>(moving_field.failure)
                                   << '/' << static_cast<int>(moving_field.diffusion_failure)
                                   << '\n';
                }
                return false;
            }
            accepted_radiation_wall_momentum = moving_field.radiation_wall_momentum;
            trial_field = std::move(moving_field.diffusion);
        } else {
            trial_field = TryAdvanceAsymptoticDiffusionField(radiation_energy, scalar_opacity,
                transport_mask, geometry, dt, diffusion_work, admitted,
                old_radiation_inertia, diffusion_options);
        }
        if (!trial_field.valid) {
            if (options.diffusion.verbosity > 0) {
                amrex::Print() << "Native scalar field replay failure="
                               << static_cast<int>(trial_field.failure) << '\n';
            }
            return false;
        }
#if !defined(WARPX_DIM_RZ)
        // The same boundary pressure used below for the material stress has an
        // external wall reaction in both stationary and moving scalar modes.
        accepted_radiation_wall_momentum =
            DiffusionWallImpulse(trial_field, transport_mask, geometry, dt);
#else
        accepted_radiation_wall_momentum[2] =
            DiffusionWallImpulse(trial_field, transport_mask, geometry, dt)[2];
#endif
        amrex::MultiFab::Copy(next_impulse, trial_field.material_impulse, 0, 0, 3, 0);
        if (options.quasi_static_angular_traction) {
            // Quasi-static first moment: I = - integral_boundary(P.n) dt dA.
            // At K/D faces this is measured angular crossing traction, not the
            // scalar pressure PLUS a compensating kick. D/D stresses remain
            // shared isotropic values and cancel between adjacent D cells.
            next_impulse.setVal(0);
            for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
#if defined(WARPX_DIM_1D_Z)
            int const physical_axis = 2;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            int const physical_axis = axis == 0 ? 0 : 2;
#else
            int const physical_axis = axis;
#endif
            auto const& leaked = *trial_field.face_leaked_energy[axis];
            amrex::MultiFab outgoing(leaked.boxArray(), distribution,
                                    face_inventory_components, leaked.nGrowVect());
            detail::EmissionPlan planned_leakage;
            if (!MakeLeakagePlan(leaked, transport_mask, geometry,
                    options.photon_energy, planned_leakage, &outgoing,
                    options.resolved_collided_source && options.moving_diffusion ?
                    &staged.WorkVelocity() : nullptr, axis,
                    options.logical_birth_keys ? &*options.logical_birth_keys : nullptr)) {
                if (options.diffusion.verbosity > 0) {
                    amrex::Print() << "Native traction leakage replay failure axis="
                                   << axis << '\n';
                }
                return false;
            }
            for (amrex::MFIter mfi(next_impulse); mfi.isValid(); ++mfi) {
                auto const mask = transport_mask.const_array(mfi);
#if AMREX_SPACEDIM == 1
                auto const cross = crossing_inventory.const_array(mfi);
#endif
                auto const escape = outgoing.const_array(mfi);
                auto const pressure = trial_field.face_pressure[axis]->const_array(mfi);
                auto const impulse = next_impulse.array(mfi);
                auto const face_area_dt = dt*area[axis];
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    for (int d = 0; d < 3; ++d) {
                        amrex::Real traction = 0;
                        if (mask(i,j,k) == 1) {
                            for (int side = -1; side <= 1; side += 2) {
                                int face[3] = {i,j,k}, neighbor[3] = {i,j,k};
                                face[axis] += side > 0 ? 1 : 0;
                                neighbor[axis] += side;
                                if (mask(neighbor[0],neighbor[1],neighbor[2]) == 0) {
                                    amrex::Real escaped_momentum =
                                        escape(face[0],face[1],face[2],d+1);
#if defined(WARPX_DIM_RZ)
                                    if (d == 1) {
                                        escaped_momentum = escape(face[0],face[1],face[2],4)/
                                            metric.Angular(i).mean_radius;
                                    }
#endif
                                    traction += escaped_momentum/PhysConst::c;
#if AMREX_SPACEDIM == 1
                                        traction += (cross(face[0],j,k,d+5) -
                                            cross(face[0],j,k,d+1))/PhysConst::c;
#endif
                                } else if (d == physical_axis) {
                                    amrex::Real pressure_weight = face_area_dt;
#if defined(WARPX_DIM_RZ)
                                    pressure_weight = dt*metric.Area(axis,face[0]);
#endif
                                    traction += pressure_weight *
                                        pressure(face[0],face[1],face[2]) * side;
                                }
                            }
                        }
                        impulse(i,j,k,d) -= traction;
                    }
                });
            }
            }
#if defined(WARPX_DIM_RZ)
            for (amrex::MFIter it(next_impulse); it.isValid(); ++it) {
                auto const mask = transport_mask.const_array(it);
                auto const energy = trial_field.energy.const_array(it);
                auto const impulse = next_impulse.array(it);
                amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    if (mask(i,j,k) == 1) {
                        impulse(i,j,k,0) += dt*(metric.Area(0,i+1)-metric.Area(0,i))*
                            energy(i,j,k)/(3*metric.Volume(i));
                    }
                });
            }
#endif
        }
        if (options.retain_constitutive_momentum) {
            // next_impulse currently equals minus the independently integrated
            // outward stress. Retain the baseline q change exactly once.
            amrex::MultiFab outward_stress(boxes, distribution, 3, 0);
            amrex::MultiFab::Copy(outward_stress, next_impulse, 0, 0, 3, 0);
            outward_stress.mult(-1, 0, 3, 0);
            ConstitutiveDiffusionMomentumReport storage_report;
            if (!TryComputeConstitutiveDiffusionMomentum(radiation_energy,
                    *conserved_diffusion_momentum, trial_field.energy,
                    (options.moving_diffusion ? moving_field.lab_radiation_inertia :
                     trial_field.radiation_inertia), outward_stress, transport_mask,
                    geometry, next_impulse, storage_report)) {
                // State diagnostics contain reductions, even if a library caller
                // configured verbosity on only one rank.
                int diagnostic_requested = options.diffusion.verbosity > 0 ? 1 : 0;
                amrex::ParallelDescriptor::ReduceIntMax(diagnostic_requested);
                if (diagnostic_requested != 0) {
                    amrex::Print() << "Native constitutive momentum replay failure="
                                   << storage_report.reason << '\n';
                    PrintConstitutiveState(radiation_energy, *conserved_diffusion_momentum,
                        trial_field.energy, options.moving_diffusion ?
                        moving_field.lab_radiation_inertia : trial_field.radiation_inertia);
                }
                return false;
            }
        }
        amrex::MultiFab::Add(next_impulse, collision_impulse, 0, 0, 3, 0);
        // Reconstruct actual scalar debit from the solved reservoir and face
        // currents. Copying RequestedWork here would make the test circular.
        // Use the laboratory enthalpy-plus-diffusive flux for moving material.
        // Using F0 alone would incorrectly count transported energy as work.
        for (amrex::MFIter mfi(debit); mfi.isValid(); ++mfi) {
            auto const old_e = radiation_energy.const_array(mfi);
            auto const new_e = trial_field.energy.const_array(mfi);
            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> flux{};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                flux[d] = (options.moving_diffusion ? moving_field.face_lab_energy_current[d] :
                           trial_field.face_current[d])->const_array(mfi);
            }
            auto const mask = transport_mask.const_array(mfi);
            auto const w = debit.array(mfi);
            auto const s = scale.array(mfi);
            auto const k_work = collision_debit.const_array(mfi);
            auto const k_scale = collision_scale.const_array(mfi);
            auto const resolved = options.resolved_collided_source;
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if (mask(i,j,k) == 0) {
                    w(i,j,k) = k_work(i,j,k);
                    s(i,j,k) = k_scale(i,j,k);
                    return;
                }
                amrex::Real divergence = 0;
                amrex::Real uncancelled = std::abs(old_e(i,j,k)) + std::abs(new_e(i,j,k));
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    int upper[3] = {i,j,k};
                    ++upper[d];
                    amrex::Real low_area = area[d], high_area = area[d];
#if defined(WARPX_DIM_RZ)
                    low_area = metric.Area(d,i);
                    high_area = metric.Area(d,upper[0]);
#endif
                    amrex::Real const low = dt * low_area * flux[d](i,j,k);
                    amrex::Real const high = dt * high_area *
                        flux[d](upper[0],upper[1],upper[2]);
                    divergence += high - low;
                    uncancelled += std::abs(low) + std::abs(high);
                }
                w(i,j,k) = (old_e(i,j,k) - new_e(i,j,k)) - divergence +
                    (resolved ? k_work(i,j,k) : 0);
                s(i,j,k) = uncancelled + (resolved ? k_scale(i,j,k) : 0);
            });
        }
        accepted_field = std::make_unique<AsymptoticDiffusionFieldResult>(std::move(trial_field));
        if (options.moving_diffusion) {
            accepted_lab_inertia = std::make_unique<amrex::MultiFab>(
                std::move(moving_field.lab_radiation_inertia));
        }
        photon_trial = std::move(replay_photons);
        return true;
    };
    if (!replay(initial_work, initial_impulse, initial_debit, initial_scale)) {
        result.reason = "Initial collision/interface/diffusion replay rejected";
        return result;
    }
    std::unique_ptr<ParticleImpulseTransaction> accepted_work;
    PacketMaterialWorkIterationReport work_report;
    if (!TryIteratePacketMaterialWork(*material, impulse_path, initial_impulse,
            replay, options.material, accepted_work, work_report)) {
        result.reason = "Finite-mass work iteration: " +
            PacketMaterialWorkIterationReasonName(work_report.reason);
        if (options.diffusion.verbosity > 0) {
            amrex::Print() << "Native work iteration failure iterations="
                           << work_report.iterations
                           << " impulse_residual=" << work_report.max_impulse_residual
                           << " radiation_work_residual="
                           << work_report.max_radiation_work_residual
                           << " material_work_excess_residual="
                           << work_report.max_material_work_excess_residual << '\n';
        }
        return result;
    }
    if (!accepted_work || !photon_trial || !accepted_field || !accepted_field->valid ||
        !accepted_field->face_leaked_energy[0]) {
        result.reason = "Missing accepted work or leakage state";
        return result;
    }
    // The field operator advances the OLD diffuse population. Collided photons
    // are admitted at census afterwards as a pure four-inventory transfer;
    // neither admission nor newly created leakage photons get a second kick.
    amrex::MultiFab accepted_energy(boxes, distribution, 1, 0);
    amrex::MultiFab::Copy(accepted_energy, accepted_field->energy, 0, 0, 1, 0);
    amrex::MultiFab accepted_q;
    if (conserved_diffusion_momentum) {
        accepted_q.define(boxes, distribution, 3, 0);
        amrex::MultiFab::Copy(accepted_q,
            options.moving_diffusion ? *accepted_lab_inertia : accepted_field->radiation_inertia,
            0, 0, 3, 0);
    }
    CollidedDiffusionAdmissionReport admission_report;
    if (options.resolved_collided_source) {
        std::unique_ptr<WarpXParticleContainer::Base> pre_admission;
        if (options.kinetic_empty_cells) {
            pre_admission = std::make_unique<WarpXParticleContainer::Base>(
                photon_trial->Particles().make_alike<>());
            pre_admission->SetArena(photon_trial->Particles().arena());
            pre_admission->copyParticles(photon_trial->Particles(), true);
        }
        admission_report = TryAdmitCollidedPhotons(photon_trial->Particles(), admission_components,
            accepted_energy, accepted_q, diffusion_mask, geometry, options.admission);
        if (!admission_report.valid) {
            result.reason = "Invalid collided-photon census admission";
            return result;
        }
        if (options.kinetic_empty_cells && admission_report.admitted > 0 &&
            !AdmissibleResolvedDiffusionCensus(accepted_energy, accepted_q, opacity,
                diffusion_mask, accepted_work->WorkVelocity(), geometry, dt,
                diffusion_options, options.moving_diffusion)) {
            // Reject representation conversion, not physical transport. The
            // same scattered photons keep their actual four-vectors and keys.
            photon_trial->Particles().copyParticles(*pre_admission, true);
            amrex::MultiFab::Copy(accepted_energy, accepted_field->energy, 0, 0, 1, 0);
            amrex::MultiFab::Copy(accepted_q, options.moving_diffusion ?
                *accepted_lab_inertia : accepted_field->radiation_inertia, 0, 0, 3, 0);
            result.deferred_front_admissions = admission_report.admitted;
            admission_report.retained += admission_report.admitted;
            admission_report.admitted = 0;
            admission_report.actual_admitted_four_inventory = {0, 0, 0, 0};
            admission_report.admitted_four_inventory.setVal(0);
        }
        if (!consistent_active_opacity(accepted_energy)) {
            result.reason = "Admitted population has inconsistent packet/diffusion transport opacity";
            return result;
        }
    }
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> leakage_inventory;
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> traction_defect;
    if (!all_diffusion) {
        for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
            auto const& faces = *accepted_field->face_leaked_energy[axis];
            leakage_inventory[axis] = std::make_unique<amrex::MultiFab>(
                faces.boxArray(), distribution, face_inventory_components, faces.nGrowVect());
            traction_defect[axis] = std::make_unique<amrex::MultiFab>(
                faces.boxArray(), distribution, 3, 0);
            detail::EmissionPlan leakage;
            if (!MakeLeakagePlan(faces, transport_mask, geometry,
                    options.photon_energy, leakage, leakage_inventory[axis].get(),
                    options.resolved_collided_source && options.moving_diffusion ?
                    &accepted_work->WorkVelocity() : nullptr, axis,
                    options.logical_birth_keys ? &*options.logical_birth_keys : nullptr)) {
                result.reason = "Private leakage construction rejected";
                return result;
            }
            if (require_birth_keys && !leakage.x.empty() &&
                leakage.logical_draw_keys.size() != leakage.x.size()) {
                result.reason = "Resolved leakage births lack logical draw keys";
                return result;
            }
            if (!detail::StageEmissionPlan(*photon_trial, leakage)) {
                result.reason = "Private leakage birth staging rejected";
                return result;
            }
#if defined(WARPX_DIM_1D_Z)
            int const physical_axis = 2;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            int const physical_axis = axis == 0 ? 0 : 2;
#else
            int const physical_axis = axis;
#endif
        for (amrex::MFIter mfi(*traction_defect[axis]); mfi.isValid(); ++mfi) {
#if AMREX_SPACEDIM == 1
            auto const cross = crossing_inventory.const_array(mfi);
#endif
            auto const out = leakage_inventory[axis]->const_array(mfi);
            auto const pressure = accepted_field->face_pressure[axis]->const_array(mfi);
            auto const mask = transport_mask.const_array(mfi);
            auto const defect = traction_defect[axis]->array(mfi);
            auto const traction_scale = PhysConst::c * dt * area[axis];
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                int low[3] = {i,j,k};
                --low[axis];
                int const normal = mask(low[0],low[1],low[2]) - mask(i,j,k);
                for (int d = 0; d < 3; ++d) {
                    amrex::Real crossing = 0;
#if AMREX_SPACEDIM == 1
                        crossing = cross(i,j,k,d+5) - cross(i,j,k,d+1);
#endif
                    auto actual = out(i,j,k,d+1);
                    auto stress_scale = traction_scale;
#if defined(WARPX_DIM_RZ)
                    stress_scale = PhysConst::c*dt*metric.Area(axis,i);
                    int const source_i = axis == 0 && normal > 0 ? i-1 : i;
                    if (normal != 0 && d == 1) {
                        actual = out(i,j,k,4)/metric.Angular(source_i).mean_radius;
                    }
#endif
                    defect(i,j,k,d) = normal == 0 ? 0 : crossing + actual -
                        (d == physical_axis ? stress_scale * pressure(i,j,k) * normal : 0);
                }
            });
        }
        if (!traction_defect[axis]->is_finite()) {
            result.reason = "Nonfinite local interface traction diagnostic";
            return result;
        }
        }
    }
    if (!photon_trial->ValidateLive()) {
        result.reason = "Live photon preflight rejected";
        return result;
    }

    amrex::GpuArray<amrex::Real, 4> final_photons{};
    if (!SourcePhotonInventory(photon_trial->Particles(), final_photons)) {
        result.reason = "Invalid actual outgoing photon inventory";
        return result;
    }
    amrex::Real const old_scalar = radiation_energy.sum(0);
    amrex::Real const new_scalar = accepted_energy.sum(0);
    amrex::Real const initial = initial_photons[0] + old_scalar;
    amrex::Real const final = final_photons[0] + new_scalar;
    result.raw_energy_residual = final - initial + work_report.actual_work +
        work_report.energy_carry_change + path_report.escaped_inventory[0];
    if (options.radiation_representation_carry) {
        // Immutable ownership participates explicitly, but never inflates the
        // physical acceptance norm or supplies a material work source.
        result.raw_energy_residual += representation_carry.sum(0) -
            options.radiation_representation_carry->sum(0);
    }
    amrex::Real const ledger_scale = std::abs(initial) + std::abs(final) +
        std::abs(work_report.actual_work) + std::abs(work_report.energy_carry_change) +
        path_report.escaped_inventory[0];
    valid = std::isfinite(result.raw_energy_residual) && std::isfinite(ledger_scale) &&
        std::abs(result.raw_energy_residual) <= options.energy_relative_tolerance * ledger_scale;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) {
        result.reason = "Actual photon/scalar/material combined energy ledger rejected";
        return result;
    }
#if defined(WARPX_DIM_RZ)
    result.rz_axial_momentum_residual =
        (final_photons[3]-initial_photons[3] + accepted_q.sum(2) -
         conserved_diffusion_momentum->sum(2))/PhysConst::c +
        accepted_work->ConjugateImpulse().sum(2) +
        accepted_work->ConjugateMomentumCarryChange().sum(2) +
        path_report.wall_momentum[2] + accepted_radiation_wall_momentum[2];
    result.rz_angular_momentum_residual =
        (final_photons[2]-initial_photons[2] + AngularFieldSum(accepted_q,geometry) -
         AngularFieldSum(*conserved_diffusion_momentum,geometry))/PhysConst::c +
        AngularFieldSum(accepted_work->ConjugateImpulse(),geometry) +
        AngularFieldSum(accepted_work->ConjugateMomentumCarryChange(),geometry);
    if (options.radiation_representation_carry) {
        result.rz_axial_momentum_residual +=
            (representation_carry.sum(3) - options.radiation_representation_carry->sum(3)) /
            PhysConst::c;
        result.rz_angular_momentum_residual +=
            (AngularFieldSum(representation_carry, geometry, 2) -
             AngularFieldSum(*options.radiation_representation_carry, geometry, 2)) / PhysConst::c;
    }
    auto const axial_scale = (std::abs(initial)+std::abs(final))/PhysConst::c;
    auto const angular_scale = axial_scale*geometry.ProbHi(0);
    valid = std::isfinite(result.rz_axial_momentum_residual) &&
        std::isfinite(result.rz_angular_momentum_residual) &&
        std::abs(result.rz_axial_momentum_residual) <= options.energy_relative_tolerance*axial_scale &&
        std::abs(result.rz_angular_momentum_residual) <= options.energy_relative_tolerance*angular_scale;
#else
    for (int d = 0; d < 3; ++d) {
        amrex::Real const old_q = old_radiation_inertia ? old_radiation_inertia->sum(d) : 0;
        result.momentum_model_residual[d] =
            (final_photons[d+1] - initial_photons[d+1] + path_report.escaped_inventory[d+1] +
             (options.moving_diffusion ? accepted_lab_inertia->sum(d) :
              accepted_field->radiation_inertia.sum(d)) - old_q) / PhysConst::c +
            accepted_work->ActualImpulse().sum(d) + accepted_work->MomentumCarryChange().sum(d);
        valid = valid && std::isfinite(result.momentum_model_residual[d]);
        result.quasi_static_momentum_residual[d] =
            (final_photons[d+1] - initial_photons[d+1] +
             path_report.escaped_inventory[d+1])/PhysConst::c +
            accepted_work->ActualImpulse().sum(d) + accepted_work->MomentumCarryChange().sum(d) +
            path_report.wall_momentum[d] + accepted_radiation_wall_momentum[d];
        result.constitutive_momentum_residual[d] = result.quasi_static_momentum_residual[d];
        if (conserved_diffusion_momentum) {
            result.constitutive_momentum_residual[d] +=
                (accepted_q.sum(d) -
                 conserved_diffusion_momentum->sum(d))/PhysConst::c;
        }
        if (options.quasi_static_angular_traction) {
            // Independent actual packet/material inventory. The diagnostic
            // asymptotic inertia is not inserted as a conserved reservoir.
            auto const scale = (std::abs(initial_photons[0]) + std::abs(final_photons[0]) +
                std::abs(old_scalar) + std::abs(new_scalar) +
                path_report.escaped_inventory[0])/PhysConst::c;
            auto const residual = options.retain_constitutive_momentum ?
                result.constitutive_momentum_residual[d] : result.quasi_static_momentum_residual[d];
            valid = valid && std::isfinite(residual) && std::abs(residual) <=
                    options.energy_relative_tolerance * scale;
        }
    }
#endif
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) {
        result.reason = "Nonfinite momentum-model diagnostic";
        return result;
    }
    // Copy actual transaction fields before any material commit. Radiation
    // ledgers are not a substitute for these finite-mass outputs.
    amrex::MultiFab actual_impulse(boxes, distribution, 3, 0);
    amrex::MultiFab actual_work_field(boxes, distribution, 1, 0);
    amrex::MultiFab momentum_carry_change(boxes, distribution, 3, 0);
    amrex::MultiFab::Copy(actual_impulse, accepted_work->ConjugateImpulse(), 0, 0, 3, 0);
    amrex::MultiFab::Copy(actual_work_field, accepted_work->ActualWork(), 0, 0, 1, 0);
    amrex::MultiFab::Copy(momentum_carry_change, accepted_work->ConjugateMomentumCarryChange(),
                         0, 0, 3, 0);
    // Commit work first only to the private material copy. A larger source
    // transaction may retain these exact candidates without live publication.
    accepted_work->Commit();
    if (options.defer_live_commit) {
        result.private_material = std::move(material);
        result.private_photons = std::move(photon_trial);
        result.private_radiation_energy = std::move(accepted_energy);
        result.private_radiation_energy.FillBoundary(geometry.periodicity());
        if (options.adaptive_rz_private_snapshot) {
            result.private_radiation_representation_carry = std::move(representation_carry);
        }
        if (conserved_diffusion_momentum) {
            result.private_diffusion_momentum = std::move(accepted_q);
            result.private_diffusion_momentum.FillBoundary(geometry.periodicity());
        }
    } else {
        // No numerical rejection is permitted after joint live publication.
        if (!material->Commit()) {
            result.reason = "Live material changed before joint commit";
            return result;
        }
        photon_trial->Commit();
        amrex::MultiFab::Copy(radiation_energy, accepted_energy, 0, 0, 1, 0);
        radiation_energy.FillBoundary(geometry.periodicity());
        if (conserved_diffusion_momentum) {
            amrex::MultiFab::Copy(*conserved_diffusion_momentum,
                accepted_q, 0, 0, 3, 0);
            conserved_diffusion_momentum->FillBoundary(geometry.periodicity());
        }
    }
    result.valid = true;
    result.used_quasi_static_angular_traction = options.quasi_static_angular_traction;
    result.used_constitutive_momentum = options.retain_constitutive_momentum;
    result.reason = "accepted bounded packet/diffusion source interval";
    result.work_iterations = work_report.iterations;
    result.collision_events = collision_report.collision_events;
    result.interface_tangential_model_defect = accepted_tangential_defect;
    result.actual_material_work = work_report.actual_work;
    result.material_energy_carry_change = work_report.energy_carry_change;
    result.field = std::move(*accepted_field);
    if (accepted_lab_inertia) { result.lab_radiation_inertia = std::move(*accepted_lab_inertia); }
    result.radiation_wall_momentum = accepted_radiation_wall_momentum;
    result.packet_wall_momentum = path_report.wall_momentum;
    result.escaped_packet_inventory = path_report.escaped_inventory;
    result.actual_material_impulse_field = std::move(actual_impulse);
    result.actual_material_work_field = std::move(actual_work_field);
    result.material_momentum_carry_change_field = std::move(momentum_carry_change);
    if (options.resolved_collided_source) {
        result.admitted_four_inventory = std::move(admission_report.admitted_four_inventory);
    }
    if (!all_diffusion) {
        if constexpr (AMREX_SPACEDIM == 1) {
            result.interface_crossing_inventory = std::move(crossing_inventory);
            result.interface_leakage_inventory = std::move(*leakage_inventory[0]);
            result.interface_traction_model_defect = std::move(*traction_defect[0]);
        } else {
            result.interface_leakage_by_direction = std::move(leakage_inventory);
            result.interface_traction_defect_by_direction = std::move(traction_defect);
        }
    }
    return result;
#endif
}
}
