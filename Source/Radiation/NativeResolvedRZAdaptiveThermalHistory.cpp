/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativeResolvedRZAdaptiveThermalHistory.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "NativePhotonCarryRouting.H"
#include "NativePhotonReplayInternals.H"
#include "NativeRZOwnedNodalCaloric.H"
#include "Particles/MultiParticleContainer.H"
#include "PhotonLogicalDrawKey.H"
#include "RadiationTransport.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace warpx::radiation
{
namespace
{
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU)
bool All (bool value)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(value);
    return value;
}

void Sum (long double* values, int count)
{
#ifdef AMREX_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, values, count, MPI_LONG_DOUBLE, MPI_SUM,
                  amrex::ParallelDescriptor::Communicator());
#else
    amrex::ignore_unused(values, count);
#endif
}

amrex::MultiFab Copy (amrex::MultiFab const& input)
{
    amrex::MultiFab out(input.boxArray(), input.DistributionMap(), input.nComp(),
                        input.nGrowVect());
    amrex::MultiFab::Copy(out, input, 0, 0, input.nComp(), input.nGrowVect());
    return out;
}

amrex::iMultiFab Copy (amrex::iMultiFab const& input)
{
    amrex::iMultiFab out(input.boxArray(), input.DistributionMap(), input.nComp(),
                         input.nGrowVect());
    amrex::iMultiFab::Copy(out, input, 0, 0, input.nComp(), input.nGrowVect());
    return out;
}

amrex::MultiFab Part (amrex::MultiFab const& input, int component, int count)
{
    amrex::MultiFab out(input.boxArray(), input.DistributionMap(), count, 1);
    out.setVal(0);
    amrex::MultiFab::Copy(out, input, component, 0, count, 0);
    return out;
}

amrex::MultiFab Pack (amrex::MultiFab const& energy, amrex::MultiFab const& momentum)
{
    amrex::MultiFab out(energy.boxArray(), energy.DistributionMap(), 4, 1);
    out.setVal(0);
    amrex::MultiFab::Copy(out, energy, 0, 0, 1, 0);
    amrex::MultiFab::Copy(out, momentum, 0, 1, 3, 0);
    return out;
}

bool SameString (std::string const& input)
{
    bool valid = !input.empty() && input.size() <= 1024;
    if (!All(valid)) { return false; }
    int length = amrex::ParallelDescriptor::IOProcessor() ? static_cast<int>(input.size()) : 0;
    amrex::ParallelDescriptor::Bcast(&length, 1,
                                    amrex::ParallelDescriptor::IOProcessorNumber());
    std::vector<char> value(static_cast<std::size_t>(length));
    if (amrex::ParallelDescriptor::IOProcessor())
    { std::copy(input.begin(), input.end(), value.begin()); }
    amrex::ParallelDescriptor::Bcast(value.data(), length,
                                    amrex::ParallelDescriptor::IOProcessorNumber());
    return All(std::string(value.begin(), value.end()) == input);
}

bool CensusSchema (WarpXParticleContainer::Base const& photons,
                    CollidedDiffusionAdmissionComponents const& census)
{
    std::array<char const*, 3> const names{
        "radiation_diffusion_census_ux0", "radiation_diffusion_census_uy0",
        "radiation_diffusion_census_uz0"};
    auto const& registered = photons.GetRealSoANames();
    bool valid = photons.finestLevel() == 0;
    for (int d = 0; d < 3; ++d)
    {
        int low = census.initial_u[d], high = low;
        amrex::ParallelDescriptor::ReduceIntMin(low);
        amrex::ParallelDescriptor::ReduceIntMax(high);
        bool const in_range = low >= PIdx::nattribs && low < static_cast<int>(registered.size());
        valid = valid && low == high && in_range;
        if (in_range) { valid = valid && registered[low] == names[d]; }
    }
    return All(valid);
}

bool StaticHistoryControls (NativeResolvedRZThermalHistoryOptions const& history)
{
    auto const& interval = history.interval;
    auto const& thermal = interval.thermal;
    auto const& packets = thermal.packets;
    auto const& field = interval.field;
    // EXACT existing accepted option domains, moved ahead of the new outer
    // clone/callback/conversion boundary. Physical state checks remain in the
    // existing paths; no trial, field solve, clock or draw is performed here.
    bool valid = interval.packet_subsegment_limit > 0 &&
        thermal.max_temperature_iterations > 0 && thermal.max_linear_iterations > 0 &&
        thermal.max_material_iterations > 0 && thermal.max_line_search > 0 &&
        thermal.material_tolerance > 0 && thermal.material_tolerance < 1 &&
        thermal.work_tolerance > 0 && thermal.work_tolerance < 1 &&
        thermal.energy_tolerance > 0 && thermal.energy_tolerance < 1 &&
        std::isfinite(thermal.minimum_electron_density) && thermal.minimum_electron_density >= 0 &&
        thermal.caloric_backend == NativeThermalCaloricBackend::ReciprocalNodalEmission &&
        thermal.reciprocal_composite_old_temperature && thermal.allow_vacuum_noop &&
        thermal.reciprocal_emission_rays_per_corner > 0 &&
        thermal.reciprocal_emission_rays_per_corner <= 1024 &&
        packets.emission && packets.rz_angular_conjugate && packets.rz_reciprocal_corner_emission &&
        packets.emission_rays_per_cell > 0 && packets.emission_rays_per_cell <= 1024 &&
        std::isfinite(packets.photon_energy) && packets.photon_energy > 0 &&
        std::isfinite(packets.radiation_constant) && packets.radiation_constant > 0 &&
        field.minimum_optical_depth >= 8 && field.maximum_beta > 0 && field.maximum_beta <= .001 &&
        field.max_picard_iterations > 0 && field.linear.max_linear_iterations > 0 &&
        field.linear.linear_tolerance > 0 && field.linear.linear_tolerance <= 1.e-11 &&
        field.linear.residual_tolerance > 0 && field.linear.residual_tolerance <= 1.e-11 &&
        field.linear.marshak_extrapolation_lambda > 0 &&
        std::isfinite(field.minimum_optical_depth+field.maximum_beta+field.linear.linear_tolerance+
            field.linear.residual_tolerance+field.linear.marshak_extrapolation_lambda);
    if (!All(valid)) { return false; }
    int minimum[]{interval.packet_subsegment_limit, interval.collisions.max_events_per_photon,
        static_cast<int>(interval.collisions.angular_law), interval.collisions.absorption,
        interval.collisions.emission, interval.collisions.moving_cross_sections,
        thermal.max_temperature_iterations, thermal.max_linear_iterations,
        thermal.max_material_iterations, thermal.max_line_search, thermal.allow_vacuum_noop,
        static_cast<int>(thermal.caloric_backend), thermal.reciprocal_emission_rays_per_corner,
        thermal.reciprocal_composite_old_temperature, thermal.rz_reciprocal_implicit,
        thermal.rz_closed_reciprocal_endpoint_trial, thermal.rz_closed_reciprocal_spatial_endpoint_trial,
        packets.emission_rays_per_cell, packets.emission, packets.survival_weighted_emission,
        packets.rz_angular_conjugate, packets.rz_reciprocal_corner_emission,
        packets.rz_reflecting_axial_frozen_bath, packets.rz_reflecting_axial_reciprocal_endpoint,
        field.max_picard_iterations, field.linear.max_linear_iterations,
        interval.reflecting_axial_fixed_mask_trial, field.reflecting_axial_field_trial};
    std::array<int,std::size(minimum)> maximum;
    std::copy(std::begin(minimum),std::end(minimum),maximum.begin());
    amrex::ParallelDescriptor::ReduceIntMin(minimum,std::size(minimum));
    amrex::ParallelDescriptor::ReduceIntMax(maximum.data(),maximum.size());
    amrex::Real low[]{interval.collisions.max_optical_depth,
        interval.collisions.fixed_frame_direction[0], interval.collisions.fixed_frame_direction[1],
        interval.collisions.fixed_frame_direction[2], thermal.material_tolerance, thermal.work_tolerance,
        thermal.energy_tolerance, thermal.minimum_electron_density, packets.photon_energy,
        packets.radiation_constant, packets.max_absorption_optical_depth, field.minimum_optical_depth,
        field.maximum_beta, field.linear.linear_tolerance, field.linear.residual_tolerance,
        field.linear.marshak_extrapolation_lambda};
    std::array<amrex::Real,std::size(low)> high;
    std::copy(std::begin(low),std::end(low),high.begin());
    amrex::ParallelDescriptor::ReduceRealMin(low,std::size(low));
    amrex::ParallelDescriptor::ReduceRealMax(high.data(),high.size());
    std::array<std::uint64_t,4> const keys{interval.collisions.seed,interval.collisions.interval,
                                        packets.seed,packets.interval};
    auto expected = keys;
    amrex::ParallelDescriptor::Bcast(expected.data(),expected.size(),
                                    amrex::ParallelDescriptor::IOProcessorNumber());
    return All(std::equal(std::begin(minimum),std::end(minimum),maximum.begin()) &&
        std::equal(std::begin(low),std::end(low),high.begin()) && expected == keys);
}

bool Opacity (NativeRZThermalHistoryOpacityEvaluator const& evaluate,
              amrex::Real time, amrex::MultiFab const& temperature,
              amrex::MultiFab const& rate, amrex::MultiFab& scattering,
              amrex::MultiFab& absorption, std::string& reason)
{
    bool valid = evaluate(time, temperature, rate, scattering, absorption, reason);
    if (!All(valid)) { return false; }
    return scattering.is_finite(0, 1, 0) && absorption.is_finite(0, 1, 0) &&
        scattering.min(0) >= 0 && absorption.min(0) >= 0;
}

amrex::MultiFab Total (amrex::MultiFab const& scattering, amrex::MultiFab const& absorption)
{
    auto out = Copy(scattering);
    amrex::MultiFab::Add(out, absorption, 0, 0, 1, 0);
    return out;
}

void ConstrainDirection (amrex::iMultiFab const& old, amrex::iMultiFab& desired,
                          amrex::MultiFab const& total, amrex::Real leaf_dt, bool entry)
{
    for (amrex::MFIter it(desired); it.isValid(); ++it)
    for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
    {
        int const previous = old[it](cell());
        // Demotion only on entry. Endpoint retains every accepted middle-D
        // cell until the NEXT fresh entry, and proposes only new K->D labels.
        if (entry && (previous == 0 || PhysConst::c*leaf_dt*total[it](cell()) < 8))
        { desired[it](cell()) = 0; }
        if (!entry && previous == 1) { desired[it](cell()) = 1; }
        if (!entry && previous == 0 && PhysConst::c*leaf_dt*total[it](cell()) < 8)
        { desired[it](cell()) = 0; }
    }
}

bool FinalCensus (WarpXParticleContainer::Base const& photons,
                  WarpXParticleContainer const& live,
                  NativeResolvedRZThermalHistoryResult const& history)
{
    std::vector<std::uint64_t> ids, keys, draws;
    bool valid = history.census_receipt.valid &&
        photon_replay_detail::CapturePhotonRoutingKeys(photons, live,
            NativePhotonPathSnapshotPolicy::KeyedProvisional, ids, keys, draws) &&
        ids.size() == history.photon_clocks.size();
    if (!All(valid)) { return false; }
    std::size_t collider = 0;
    for (std::size_t n = 0; n < ids.size(); ++n)
    {
        auto const& clock = history.photon_clocks[n];
        valid = valid && clock.idcpu == ids[n] && clock.provisional_draw_key == keys[n] &&
            clock.completed_events >= 0 && clock.next_draw == clock.completed_events+1;
        if (clock.completed_events > 0)
        {
            valid = valid && collider < history.census_receipt.collided_clocks.size();
            if (collider < history.census_receipt.collided_clocks.size())
            {
                auto const& receipt = history.census_receipt.collided_clocks[collider++];
                valid = valid && receipt.idcpu == clock.idcpu &&
                    receipt.provisional_draw_key == clock.provisional_draw_key &&
                    receipt.next_draw == clock.next_draw &&
                    receipt.completed_events == clock.completed_events &&
                    receipt.residual_optical_depth == clock.residual_optical_depth;
            }
        }
    }
    return All(valid && collider == history.census_receipt.collided_clocks.size());
}

NativeRZAdaptiveHistoryOwnerWitness MeasureOwners (
    ParticleImpulseMaterial const& before, ParticleImpulseMaterial const& after,
    std::string const& path, NativeRZOwnedNodalCaloricResult const& caloric,
    NativeResolvedRZThermalHistoryResult const& history, amrex::Geometry const& geometry)
{
    NativeRZAdaptiveHistoryOwnerWitness out;
    auto const old_species = before.CandidateSpecies(), new_species = after.CandidateSpecies();
    bool valid = !old_species.empty() && old_species.size() == new_species.size() && caloric.valid;
    int low_species = static_cast<int>(old_species.size()), high_species = low_species;
    amrex::ParallelDescriptor::ReduceIntMin(low_species);
    amrex::ParallelDescriptor::ReduceIntMax(high_species);
    if (!All(valid && low_species == high_species)) { return out; }
    for (auto const& species : old_species)
    {
        if (!SameString(species.name)) { return out; }
        auto const* next_species = after.CandidateSpecies(species.name);
        auto const& names = species.particles->GetRealSoANames();
        std::array<int, 4> carry{};
        for (int d = 0; d < 4; ++d)
        {
            auto const label = "radiation_impulse_" + path +
                (d == 3 ? "_work" : std::string("_u") + "xyz"[d]);
            auto const found = std::find(names.begin(), names.end(), label);
            carry[d] = found == names.end() ? -1 : static_cast<int>(found-names.begin());
        }
        valid = valid && next_species && species.rest_mass > 0 &&
            std::isfinite(species.rest_mass) &&
            std::all_of(carry.begin(), carry.end(), [] (int d) { return d >= 0; });
        if (!All(valid)) { return out; }
        auto const previous = photon_replay_detail::CaptureParticles(*species.particles);
        auto const current = photon_replay_detail::CaptureParticles(*next_species);
        valid = previous.tiles.size() == current.tiles.size();
        if (!All(valid)) { return out; }
        for (std::size_t t = 0; t < previous.tiles.size(); ++t)
        {
            auto const& a = previous.tiles[t]; auto const& b = current.tiles[t];
            valid = valid && a.key == b.key && a.count == b.count &&
                photon_replay_detail::SameBits(a.idcpu, b.idcpu);
            if (!valid) { continue; }
            for (long p = 0; p < a.count; ++p)
            {
                auto old = [&] (int d) -> long double { return a.real_data[d*a.count+p]; };
                auto next = [&] (int d) -> long double { return b.real_data[d*b.count+p]; };
                for (int d : {PIdx::w, PIdx::r, PIdx::theta, PIdx::z})
                { valid = valid && old(d) == next(d); }
                long double old_sq = 0, new_sq = 0, paired = 0;
                for (int d = 0; d < 3; ++d)
                {
                    auto const x = old(PIdx::ux+d), y = next(PIdx::ux+d);
                    old_sq += x*x; new_sq += y*y; paired += (y-x)*(y+x);
                }
                auto const old_gamma = std::sqrt(1+old_sq/PhysConst::c2);
                auto const new_gamma = std::sqrt(1+new_sq/PhysConst::c2);
                auto const mass = old(PIdx::w)*species.rest_mass;
                auto const radius = old(PIdx::r), cosine = std::cos(old(PIdx::theta)),
                    sine = std::sin(old(PIdx::theta));
                auto const dx = next(PIdx::ux)-old(PIdx::ux), dy = next(PIdx::uy)-old(PIdx::uy);
                auto const cx = next(carry[0])-old(carry[0]), cy = next(carry[1])-old(carry[1]);
                out.material_change[0] += mass*paired/(old_gamma+new_gamma);
                out.material_change[1] += PhysConst::c*mass*(next(PIdx::uz)-old(PIdx::uz));
                out.material_change[2] += PhysConst::c*mass*radius*(cosine*dy-sine*dx);
                out.material_carry_change[0] += mass*(next(carry[3])-old(carry[3]));
                out.material_carry_change[1] += PhysConst::c*mass*(next(carry[2])-old(carry[2]));
                out.material_carry_change[2] += PhysConst::c*mass*radius*(cosine*cy-sine*cx);
                out.material_operands[0] += mass*(old_sq/(old_gamma+1)+new_sq/(new_gamma+1)+
                    std::abs(old(carry[3]))+std::abs(next(carry[3])));
                out.material_operands[1] += PhysConst::c*mass*(std::abs(old(PIdx::uz))+
                    std::abs(next(PIdx::uz))+std::abs(old(carry[2]))+std::abs(next(carry[2])));
                out.material_operands[2] += PhysConst::c*mass*radius*(std::abs(old(PIdx::ux))+
                    std::abs(old(PIdx::uy))+std::abs(next(PIdx::ux))+std::abs(next(PIdx::uy))+
                    std::abs(old(carry[0]))+std::abs(old(carry[1]))+
                    std::abs(next(carry[0]))+std::abs(next(carry[1])));
                valid = valid && mass >= 0 && radius >= 0 && out.material_records <
                    std::numeric_limits<amrex::Long>::max();
                if (valid) { ++out.material_records; }
            }
        }
    }
    auto const& field = caloric.cell_realized_energy;
    std::array<long double,3> reported_work{}, reported_carry{};
    long double reported_heat = 0;
    for (amrex::MFIter it(field); it.isValid(); ++it)
    for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
    {
        out.eos_heat += field[it](cell(),0);
        out.eos_scale += std::abs(field[it](cell(),0));
        out.eos_roundoff_bound += field[it](cell(),1);
        auto const lo = static_cast<long double>(geometry.ProbLo(0)) +
            (cell()[0]-geometry.Domain().smallEnd(0))*static_cast<long double>(geometry.CellSize(0));
        auto const hi = lo+geometry.CellSize(0);
        auto const mean = (2.L/3)*(lo*lo+lo*hi+hi*hi)/(lo+hi);
        reported_work[0] += history.actual_material_work[it](cell());
        reported_work[1] += PhysConst::c*history.actual_conjugate_impulse[it](cell(),2);
        reported_work[2] += PhysConst::c*mean*history.actual_conjugate_impulse[it](cell(),1);
        reported_carry[0] += history.material_energy_carry_change[it](cell());
        reported_carry[1] += PhysConst::c*history.conjugate_momentum_carry_change[it](cell(),2);
        reported_carry[2] += PhysConst::c*mean*history.conjugate_momentum_carry_change[it](cell(),1);
        reported_heat += history.actual_heat[it](cell());
        valid = valid && out.caloric_cells < std::numeric_limits<amrex::Long>::max();
        if (valid) { ++out.caloric_cells; }
    }
    if (!All(valid)) { return out; }
    auto maximum = std::max(out.material_records, out.caloric_cells);
    amrex::ParallelDescriptor::ReduceLongMax(maximum);
    out.ranks = amrex::ParallelDescriptor::NProcs();
    if (out.ranks <= 0 || maximum > std::numeric_limits<amrex::Long>::max()/out.ranks)
    { return out; }
    Sum(out.material_change.data(),3); Sum(out.material_carry_change.data(),3);
    Sum(out.material_operands.data(),3); Sum(&out.eos_heat,1); Sum(&out.eos_scale,1);
    Sum(&out.eos_roundoff_bound,1);
    Sum(reported_work.data(),3); Sum(reported_carry.data(),3); Sum(&reported_heat,1);
    for (int d = 0; d < 3; ++d)
    {
        out.work_report_residual[d] = reported_work[d]-out.material_change[d];
        out.carry_report_residual[d] = reported_carry[d]-out.material_carry_change[d];
    }
    out.heat_report_residual = reported_heat-out.eos_heat;
    amrex::ParallelDescriptor::ReduceLongSum(out.material_records);
    amrex::ParallelDescriptor::ReduceLongSum(out.caloric_cells);
    auto const product = (128.L*out.material_records+32.L*out.caloric_cells+32.L*out.ranks)*
        std::numeric_limits<amrex::Real>::epsilon();
    out.valid = std::isfinite(product) && product >= 0 && product < .25L;
    if (out.valid)
    {
        for (auto& operand : out.material_operands)
        { if (operand > 0) { operand = std::nextafter(operand/(1-product),
            std::numeric_limits<long double>::infinity()); } }
        if (out.eos_scale > 0)
        { out.eos_scale = std::nextafter(out.eos_scale/(1-product),
            std::numeric_limits<long double>::infinity()); }
    }
    return out;
}
#endif
} // namespace

NativeResolvedRZAdaptiveThermalHistoryResult
TryAdvanceNativeResolvedRZAdaptiveThermalHistory (
    WarpXParticleContainer& live, PhotonPacketTransaction const& start,
    ParticleImpulseMaterial const& material_start, std::string const& path,
    amrex::MultiFab const& old_diffusion, amrex::iMultiFab const& old_mask,
    amrex::MultiFab const& old_carry, amrex::MultiFab const& old_temperature,
    HybridPICModel const& model, amrex::Geometry const& geometry,
    amrex::Real time, amrex::Real outer_dt,
    NativeRZThermalHistoryOpacityEvaluator const& evaluate,
    NativeResolvedRZAdaptiveThermalHistoryOptions const& options)
{
    NativeResolvedRZAdaptiveThermalHistoryResult rejected;
    rejected.reason = "unsupported private fresh-outer adaptive RZ thermal history";
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU)
    amrex::ignore_unused(live, start, material_start, path, old_diffusion, old_mask,
        old_carry, old_temperature, model, geometry, time, outer_dt, evaluate, options);
    return rejected;
#else
    // Static scope/option admission precedes clone/conversion/callback/clocks.
    // Actual-state/constitutive admission remains in the existing paths.
    auto const eos = model.electronThermodynamicsExecutor();
    auto& simulation = WarpX::GetInstance();
    auto const& native_geometry = simulation.Geom(0);
    auto const* native_model = simulation.get_pointer_HybridPICModel();
    auto const& h = options.history;
    bool valid = options.fresh_outer_endpoint &&
        (eos.isIdealGas() || eos.isFixedChargeLatentEnergy()) && eos.m_num_materials == 0 &&
        model.transportsElectronInternalEnergyAtRawDensity() &&
        model.solvesElectronEnergyEquation() && simulation.getdo_moving_window() == 0 &&
        !EB::enabled() && !simulation.DoFluidSpecies() &&
        !simulation.GetPartContainer().hasHybridIonization() && native_model != nullptr &&
        native_model->solvesElectronEnergyEquation() &&
        native_model->transportsElectronInternalEnergyAtRawDensity() &&
        simulation.finestLevel() == 0 && WarpX::n_rz_azimuthal_modes == 1 &&
        WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC &&
        std::numeric_limits<amrex::Real>::digits == 53 &&
        std::numeric_limits<amrex::ParticleReal>::digits == 53 &&
        // The repeated native-owner readback promotes finite binary64 scalars
        // BEFORE products/quotients. Its conservative intermediate exponent
        // envelope (about [-5300,4150], including count summation) requires an
        // extended LD range; LD==double is outside this NEW private scope.
        std::numeric_limits<long double>::min_exponent <= -8192 &&
        std::numeric_limits<long double>::max_exponent >= 8192 &&
        WarpX::GetInstance().GetRadiationTransport().numEnergyGroups() == 1 &&
        !WarpX::GetInstance().GetRadiationTransport().usesHistoryPhotonWeights() &&
        geometry.Domain() == native_geometry.Domain() && geometry.Coord() == native_geometry.Coord() &&
        geometry.ProbLo(0) == 0 && !geometry.isPeriodic(0) && geometry.isPeriodic(1) &&
        WarpX::field_boundary_lo[0] == FieldBoundaryType::None &&
        WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC &&
        WarpX::particle_boundary_lo[0] == ParticleBoundaryType::None &&
        WarpX::particle_boundary_hi[0] == ParticleBoundaryType::Reflecting &&
        !h.interval.reflecting_axial_fixed_mask_trial &&
        !h.interval.thermal.rz_closed_reciprocal_endpoint_trial &&
        !h.interval.thermal.rz_closed_reciprocal_spatial_endpoint_trial &&
        h.interval.desired_diffusion_mask == nullptr &&
        h.interval.radiation_representation_carry == nullptr &&
        h.interval.initial_photon_clocks == nullptr && !h.retain_collided_census &&
        h.interval.collisions.moving_cross_sections && !h.interval.collisions.absorption &&
        !h.interval.collisions.emission && h.interval.collisions.angular_law ==
            NativePhotonCollisionAngularLaw::Isotropic &&
        h.interval.thermal.packets.survival_weighted_emission && h.interval.thermal.rz_reciprocal_implicit &&
        h.subintervals > 0 && h.subintervals <= h.maximum_subintervals &&
        h.maximum_subintervals > 0 && h.maximum_subintervals <= 32 &&
        h.interval.collisions.max_optical_depth > 0 && h.interval.collisions.max_optical_depth <= 16 &&
        h.interval.collisions.max_events_per_photon > 0 && h.interval.collisions.max_events_per_photon <= 64 &&
        h.interval.thermal.packets.max_absorption_optical_depth > 0 &&
        h.interval.thermal.packets.max_absorption_optical_depth <= 16 &&
        std::isfinite(time) && time >= 0 && std::isfinite(outer_dt) && outer_dt > 0 &&
        std::isfinite(time+outer_dt) && time+outer_dt > time && evaluate &&
        PhysConst::c*outer_dt <= .5*std::min(geometry.CellSize(0), geometry.CellSize(1)) &&
        old_diffusion.nComp() == 4 && old_mask.nComp() == 1 && old_carry.nComp() == 4 &&
        old_diffusion.ixType().cellCentered() && old_mask.ixType().cellCentered() &&
        old_carry.ixType().cellCentered() &&
        old_diffusion.boxArray() == old_mask.boxArray() && old_diffusion.boxArray() == old_carry.boxArray() &&
        old_diffusion.DistributionMap() == old_mask.DistributionMap() &&
        old_diffusion.DistributionMap() == old_carry.DistributionMap() &&
        old_diffusion.boxArray() == live.ParticleBoxArray(0) &&
        old_diffusion.DistributionMap() == live.ParticleDistributionMap(0) &&
        old_diffusion.boxArray() == simulation.boxArray(0) &&
        old_diffusion.DistributionMap() == simulation.DistributionMap(0) &&
        old_temperature.nComp() == 1 && old_temperature.ixType().nodeCentered() &&
        old_temperature.nGrowVect().allGE(1) && old_temperature.boxArray() ==
            amrex::convert(old_diffusion.boxArray(), amrex::IntVect::TheNodeVector()) &&
        old_temperature.DistributionMap() == old_diffusion.DistributionMap() &&
        std::isfinite(options.mask_policy.demote_depth) && options.mask_policy.demote_depth >= 0 &&
        std::isfinite(options.mask_policy.promote_depth) &&
        options.mask_policy.promote_depth > options.mask_policy.demote_depth &&
        std::isfinite(options.admission.max_flux_fraction) && options.admission.max_flux_fraction >= 0 &&
        options.admission.max_flux_fraction <= amrex::Real(1)/3 &&
        std::isfinite(options.admission.max_pressure_anisotropy) &&
        options.admission.max_pressure_anisotropy > 0 && options.admission.max_pressure_anisotropy <= .1;
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
    { valid = valid && geometry.ProbLo(axis) == native_geometry.ProbLo(axis) &&
        geometry.ProbHi(axis) == native_geometry.ProbHi(axis) &&
        geometry.isPeriodic(axis) == native_geometry.isPeriodic(axis) &&
        std::isfinite(geometry.CellSize(axis)) && geometry.CellSize(axis) > 0; }
    int fresh_low = options.fresh_outer_endpoint, fresh_high = fresh_low;
    amrex::ParallelDescriptor::ReduceIntMin(fresh_low);
    amrex::ParallelDescriptor::ReduceIntMax(fresh_high);
    if (!All(valid && fresh_low == fresh_high) ||
        !ValidateNativeResolvedRZThermalIntervalScope(model, geometry, h.interval) ||
        !StaticHistoryControls(h))
    { return rejected; }
    amrex::Real controls[]{time, outer_dt, static_cast<amrex::Real>(h.subintervals),
        static_cast<amrex::Real>(h.maximum_subintervals), options.mask_policy.demote_depth,
        options.mask_policy.promote_depth, options.admission.max_flux_fraction,
        options.admission.max_pressure_anisotropy};
    amrex::Real high[8]; std::copy(std::begin(controls), std::end(controls), high);
    amrex::ParallelDescriptor::ReduceRealMin(controls,8);
    amrex::ParallelDescriptor::ReduceRealMax(high,8);
    if (!All(std::equal(std::begin(controls), std::end(controls), high)) ||
        !SameString(path) || !SameString(h.opacity_model_identity) ||
        !CensusSchema(live, options.census) || !CensusSchema(start.Particles(), options.census) ||
        !start.ValidateSnapshotFor(live) || !material_start.ValidateLive()) { return rejected; }
    if (!old_diffusion.is_finite() || !old_carry.is_finite() || !old_temperature.is_finite() ||
        old_mask.min(0) < 0 || old_mask.max(0) > 1) { return rejected; }

    NativeResolvedRZAdaptiveThermalHistoryResult result;
    auto const& boxes = old_diffusion.boxArray(); auto const& dm = old_diffusion.DistributionMap();
    amrex::Real const leaf_dt = outer_dt/h.subintervals;
    auto const frozen_charge = Copy(*WarpX::GetInstance().m_fields.get(warpx::fields::FieldType::rho_fp,0));
    auto const caloric = PrepareNativeRZOwnedNodalCaloric(model,frozen_charge,old_temperature,
        h.interval.thermal.minimum_electron_density);
    if (!caloric.Valid()) { rejected.reason = caloric.Reason(); return rejected; }
    std::unique_ptr<ParticleImpulseMaterial> rate_material;
    if (!material_start.TryCloneCandidate(rate_material)) { return rejected; }
    amrex::MultiFab zero(boxes,dm,3,0); zero.setVal(0);
    ParticleImpulseTransaction rate_sample;
    amrex::MultiFab rate;
    if (!rate_sample.Stage(*rate_material,path,zero,false,ParticleImpulseAssignment::RZAngularConservative) ||
        !rate_sample.TryInitialRateVelocity(rate)) { return rejected; }
    amrex::MultiFab scattering(boxes,dm,1,1), absorption(boxes,dm,1,1), bath(boxes,dm,1,0);
    scattering.setVal(0); absorption.setVal(0); bath.setVal(0);
    if (!Opacity(evaluate,time,old_temperature,rate,scattering,absorption,rejected.reason) ||
        !TryBuildResolvedRZThermalBathUpperBound(old_temperature,bath,geometry,
                                               h.interval.thermal.packets.radiation_constant))
    { return rejected; }
    auto total = Total(scattering,absorption);
    auto energy = Part(old_diffusion,0,1), momentum = Part(old_diffusion,1,3);
    auto desired = Copy(old_mask);
    if (!TryProposeAdaptiveDiffusionMask(old_mask,total,geometry,desired,options.mask_policy))
    { return rejected; }
    ConstrainDirection(old_mask,desired,total,leaf_dt,true);
    if (!TryConstrainResolvedRZThermalDiffusionMask(energy,momentum,bath,total,absorption,
            old_mask,desired,rate,geometry,leaf_dt,h.interval.field)) { return rejected; }
    PhotonPacketTransaction entry(live,start);
    result.entry_transition = TryConvertAdaptivePacketDiffusionRepresentation(entry,energy,momentum,
        old_mask,desired,old_carry,geometry,options.census,
        {h.interval.thermal.packets.seed,h.interval.thermal.packets.interval},
        h.interval.thermal.packets.photon_energy,options.admission,true);
    if (!result.entry_transition.valid)
    { rejected.reason = "adaptive history entry: "+result.entry_transition.reason; return rejected; }
    auto const middle = Pack(result.entry_transition.energy,result.entry_transition.momentum);
    auto history_options = h;
    history_options.retain_collided_census = true;
    history_options.collided_census = options.census;
    result.history = TryAdvanceNativeResolvedRZThermalHistory(live,entry,material_start,path,middle,
        result.entry_transition.mask,old_temperature,model,geometry,time,outer_dt,evaluate,history_options);
    rejected.completed_private_subintervals = result.history.completed_private_subintervals;
    if (!result.history.valid)
    { rejected.reason = "adaptive frozen-mask history: "+result.history.reason; return rejected; }
    if (!FinalCensus(result.history.photons->Particles(),live,result.history))
    { rejected.reason = "invalid completed adaptive history collision receipt"; return rejected; }

    auto const& endpoint_rate = result.history.steps.back().endpoint_rate_velocity;
    scattering.setVal(0); absorption.setVal(0); bath.setVal(0);
    if (!Opacity(evaluate,time+outer_dt,result.history.nodal_temperature,endpoint_rate,
            scattering,absorption,rejected.reason) ||
        !TryBuildResolvedRZThermalBathUpperBound(result.history.nodal_temperature,bath,geometry,
                                               h.interval.thermal.packets.radiation_constant))
    { return rejected; }
    total = Total(scattering,absorption);
    energy = Part(result.history.diffusion_radiation,0,1);
    momentum = Part(result.history.diffusion_radiation,1,3);
    desired = Copy(result.entry_transition.mask);
    if (!TryProposeAdaptiveDiffusionMask(result.entry_transition.mask,total,geometry,desired,
                                        options.mask_policy)) { return rejected; }
    ConstrainDirection(result.entry_transition.mask,desired,total,leaf_dt,false);
    auto promotion = std::make_unique<PhotonPacketTransaction>(live,*result.history.photons);
    auto converted = TryConvertAdaptivePacketDiffusionRepresentation(*promotion,energy,momentum,
        result.entry_transition.mask,desired,result.entry_transition.representation_carry,geometry,
        options.census,{h.interval.thermal.packets.seed,h.interval.thermal.packets.interval},
        h.interval.thermal.packets.photon_energy,options.admission,true);
    if (!converted.valid)
    { rejected.reason = "adaptive history endpoint: "+converted.reason; return rejected; }
    bool const admissible = converted.promoted_cells == 0 ||
        AdmissibleResolvedRZThermalDiffusionCensus(converted.energy,converted.momentum,bath,total,
            absorption,converted.mask,endpoint_rate,geometry,leaf_dt,h.interval.field);
    if (admissible)
    {
        result.photons = std::move(promotion);
        result.diffusion_radiation = Pack(converted.energy,converted.momentum);
        result.accepted_diffusion_mask = Copy(converted.mask);
        result.radiation_representation_carry = Copy(converted.representation_carry);
        result.endpoint_transition = std::move(converted);
    }
    else
    {
        // Only the EXISTING deterministic constitutive screen can defer pure
        // promotion. No accepted physical source or stochastic outcome is redone.
        result.endpoint_promotion_deferred = true;
        result.photons = std::make_unique<PhotonPacketTransaction>(live,*result.history.photons);
        result.diffusion_radiation = Copy(result.history.diffusion_radiation);
        result.accepted_diffusion_mask = Copy(result.entry_transition.mask);
        result.radiation_representation_carry = Copy(result.entry_transition.representation_carry);
        auto& actual = result.endpoint_transition;
        actual.valid = true; actual.energy = std::move(energy); actual.momentum = std::move(momentum);
        actual.mask = Copy(result.accepted_diffusion_mask);
        actual.representation_carry = Copy(result.radiation_representation_carry);
        actual.deferred_cells = converted.deferred_cells+converted.promoted_cells;
        // Actual published endpoint is the exact identity transfer, not the
        // refused attempted conversion residual/fields.
        actual.accounting_witness.valid = true;
        actual.accounting_witness.before = converted.accounting_witness.before;
        actual.accounting_witness.after = converted.accounting_witness.before;
    }
    result.accepted_diffusion_mask.FillBoundary(geometry.periodicity());
    result.diffusion_radiation.FillBoundary(geometry.periodicity());
    result.radiation_representation_carry.FillBoundary(geometry.periodicity());

    amrex::MultiFab owner_zero(old_temperature.boxArray(),old_temperature.DistributionMap(),1,0);
    owner_zero.setVal(0);
    auto const physical = EvaluateNativeRZOwnedNodalCaloric(model,caloric,owner_zero,&result.history.nodal_temperature);
    auto& witness = result.acceptance_witness;
    witness.owners = MeasureOwners(material_start,*result.history.material,path,physical,
                                  result.history,geometry);
    witness.limits = EvaluateNativeRZAdaptiveHistoryAcceptanceBudget(result.history.acceptance_witness,
        result.entry_transition.accounting_witness,result.endpoint_transition.accounting_witness,witness.owners);
    valid = witness.limits.valid && result.photons->ValidateSnapshotFor(live) &&
        result.history.material->ValidateLive();
    auto const& original = result.entry_transition.accounting_witness.before;
    auto const& final = result.endpoint_transition.accounting_witness.after;
    std::array<int,3> const component{0,3,2};
    for (int d = 0; d < 3; ++d)
    {
        int const c = component[d];
        witness.composed_residual[d] = result.history.acceptance_witness.residual[d] -
            result.entry_transition.accounting_witness.carried_residual[c] -
            result.endpoint_transition.accounting_witness.carried_residual[c];
        witness.actual_residual[d] = (original.packet[c]+original.field[c]+original.carry[c]) -
            (final.packet[c]+final.field[c]+final.carry[c]) - witness.owners.material_change[d] -
            witness.owners.material_carry_change[d] - (d == 0 ? witness.owners.eos_heat : 0);
        valid = valid && std::isfinite(witness.actual_residual[d]) &&
            std::isfinite(witness.composed_residual[d]) &&
            std::abs(witness.actual_residual[d]) <= witness.limits.endpoint[d] &&
            std::abs(witness.composed_residual[d]) <= witness.limits.endpoint[d] &&
            std::abs(witness.actual_residual[d]-witness.composed_residual[d]) <=
                witness.limits.representation_roundoff[d];
    }
    if (!All(valid)) { rejected.reason = "independent adaptive outer endpoint ledger rejected"; return rejected; }
    witness.valid = true; result.valid = true;
    result.completed_private_subintervals = result.history.completed_private_subintervals;
    result.policy_identity = result.history.policy_identity+"_private_fresh_outer_adaptive_entry_frozen_history_actual_census_endpoint_v1";
    return result;
#endif
}
} // namespace warpx::radiation
