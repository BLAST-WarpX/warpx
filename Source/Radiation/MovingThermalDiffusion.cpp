/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "MovingThermalDiffusion.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "RZMomentGeometry.H"
#include "WarpX.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>

namespace warpx::radiation {
namespace {
[[maybe_unused]] bool
Agree (bool local)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(local);
    return local;
}

[[maybe_unused]] bool
OverlappingStorage (amrex::MultiFab const& first, amrex::MultiFab const& second)
{
    for (amrex::MFIter mfi(first); mfi.isValid(); ++mfi) {
        auto const begin_a = reinterpret_cast<std::uintptr_t>(first[mfi].dataPtr());
        auto const begin_b = reinterpret_cast<std::uintptr_t>(second[mfi].dataPtr());
        auto const bytes_a = static_cast<std::uintptr_t>(first[mfi].box().numPts()) *
                             first.nComp() * sizeof(amrex::Real);
        auto const bytes_b = static_cast<std::uintptr_t>(second[mfi].box().numPts()) *
                             second.nComp() * sizeof(amrex::Real);
        auto const limit = std::numeric_limits<std::uintptr_t>::max();
        if (!begin_a || !begin_b || bytes_a > limit - begin_a || bytes_b > limit - begin_b ||
            (begin_a < begin_b + bytes_b && begin_b < begin_a + bytes_a)) {
            return true;
        }
    }
    return false;
}

[[maybe_unused]] void
Copy (amrex::MultiFab& target, amrex::MultiFab const& source)
{
    amrex::MultiFab::Copy(target, source, 0, 0, source.nComp(), source.nGrowVect());
}

[[maybe_unused]] bool ClosedMomentumBudget (
    ParticleImpulseTransaction const& work, MovingThermalDiffusionFieldResult const& field,
    amrex::Geometry const& geometry, amrex::Real dt, amrex::Real tolerance)
{
    auto const& impulse = work.ConjugateImpulse();
    auto const& carry = work.ConjugateMomentumCarryChange();
    amrex::Real minimum_spacing = geometry.CellSize(0);
    for (int d = 1; d < AMREX_SPACEDIM; ++d) {
        minimum_spacing = std::min(minimum_spacing, geometry.CellSize(d));
    }
    bool valid = true;
    for (int d = 0; d < 3; ++d) {
#if defined(WARPX_DIM_RZ)
        if (d != 2) { continue; }
#endif
        auto const residual = impulse.sum(d)+carry.sum(d)+field.radiation_wall_momentum[d];
        auto const scale = impulse.norm1(d)+carry.norm1(d)+
            dt*field.radiation.norm1(0)/minimum_spacing;
        valid = valid && std::isfinite(residual) && std::isfinite(scale) &&
            std::abs(residual) <= tolerance*scale;
    }
#if defined(WARPX_DIM_RZ)
    RZMomentMetric const metric(geometry);
    amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Real,amrex::Real> data(ops);
    for (amrex::MFIter it(impulse); it.isValid(); ++it) {
        auto const p = impulse.const_array(it), c = carry.const_array(it);
        ops.eval(it.validbox(), data, [=] AMREX_GPU_DEVICE(int i,int j,int k)
            -> amrex::GpuTuple<amrex::Real,amrex::Real> {
            auto const radius = metric.Angular(i).mean_radius;
            return {radius*(p(i,j,k,1)+c(i,j,k,1)),
                    radius*(std::abs(p(i,j,k,1))+std::abs(c(i,j,k,1)))};
        });
    }
    auto const sums = data.value();
    auto angular = amrex::get<0>(sums), scale = amrex::get<1>(sums);
    amrex::ParallelDescriptor::ReduceRealSum(angular);
    amrex::ParallelDescriptor::ReduceRealSum(scale);
    scale += dt*field.radiation.norm1(0)*geometry.ProbHi(0)/geometry.CellSize(0);
    valid = valid && std::isfinite(angular) && std::isfinite(scale) &&
        std::abs(angular) <= tolerance*scale;
#endif
    return Agree(valid);
}
} // namespace

MovingThermalDiffusionResult
TryAdvanceMovingThermalDiffusion (MultiParticleContainer& particles,
                                  WarpXParticleContainer& photons,
                                  std::vector<std::string> const& species, std::string const& path,
                                  amrex::MultiFab& radiation, amrex::MultiFab& nodal_temperature,
                                  amrex::MultiFab& material_energy, amrex::Geometry const& geometry,
                                  amrex::Real time, amrex::Real dt,
                                  MovingThermalDiffusionOptions const& options,
                                  CoupledImplicitCallbacks const& callbacks)
{
    MovingThermalDiffusionResult result;
    auto fail = [&] (std::string const& why) {
        result.reason = why;
        return std::move(result);
    };
#if defined(WARPX_DIM_RCYLINDER) || \
    defined(WARPX_DIM_RSPHERE) || defined(AMREX_USE_FLOAT)
    amrex::ignore_unused(particles, photons, species, path, radiation, nodal_temperature,
                         material_energy, geometry, time, dt, options, callbacks);
    return fail("Moving thermal diffusion requires double precision Cartesian or RZ geometry");
#else
    auto const& boxes = radiation.boxArray();
    auto const& distribution = radiation.DistributionMap();
#if defined(WARPX_DIM_RZ)
    bool const supported = (geometry.Coord() == 0 || geometry.Coord() == 1) &&
        !geometry.isPeriodic(0) && geometry.ProbLo(0) >= 0 && WarpX::n_rz_azimuthal_modes == 1 &&
        options.material.assignment == ParticleImpulseAssignment::RZAngularConservative;
#else
    bool const supported = geometry.Coord() == 0 &&
        options.material.assignment == ParticleImpulseAssignment::NearestCell;
#endif
    bool valid = supported && std::isfinite(time) &&
        std::isfinite(dt) && dt > 0 && std::isfinite(time + dt) && time + dt > time &&
        WarpX::GetInstance().finestLevel() == 0 &&
        std::numeric_limits<amrex::ParticleReal>::digits >= 53 && options.max_subdivisions >= 0 &&
        options.max_subdivisions <= 10 && std::isfinite(options.energy_relative_tolerance) &&
        options.energy_relative_tolerance > 0 && options.energy_relative_tolerance <= 1.e-10 &&
        !species.empty() && !path.empty() && callbacks.coefficients &&
        callbacks.material_response && radiation.nComp() == 1 &&
        radiation.ixType().cellCentered() && radiation.nGrowVect().allGE(1) &&
        nodal_temperature.nComp() == 1 && nodal_temperature.ixType().nodeCentered() &&
        nodal_temperature.nGrowVect().allGE(1) &&
        nodal_temperature.boxArray() == amrex::convert(boxes, amrex::IntVect(1)) &&
        nodal_temperature.DistributionMap() == distribution && material_energy.nComp() == 1 &&
        material_energy.boxArray() == boxes && material_energy.DistributionMap() == distribution &&
        material_energy.nGrowVect().allGE(1) && &radiation != &material_energy &&
        &radiation != &nodal_temperature && &material_energy != &nodal_temperature;
    auto const& native_geometry = WarpX::GetInstance().Geom(0);
    valid = valid && geometry.Domain() == native_geometry.Domain();
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        valid = valid && geometry.ProbLo(d) == native_geometry.ProbLo(d) &&
            geometry.ProbHi(d) == native_geometry.ProbHi(d) &&
            geometry.isPeriodic(d) == native_geometry.isPeriodic(d);
    }
    if (!Agree(valid)) {
        return fail("Invalid moving thermal source layout or controls");
    }
    if (!Agree(!OverlappingStorage(radiation, material_energy))) {
        return fail("Aliased radiation and material energy storage");
    }
    if (!Agree(!particles.hasHybridIonization())) {
        return fail("Moving thermal source requires fixed ion charge");
    }
    amrex::Real minimum[4] = {time, dt, static_cast<amrex::Real>(options.max_subdivisions),
                              options.energy_relative_tolerance};
    amrex::Real maximum[4];
    std::copy_n(minimum, 4, maximum);
    amrex::ParallelDescriptor::ReduceRealMin(minimum, 4);
    amrex::ParallelDescriptor::ReduceRealMax(maximum, 4);
    for (int i = 0; i < 4; ++i) {
        valid = valid && minimum[i] == maximum[i];
    }
    if (!Agree(valid)) {
        return fail("Rank-inconsistent moving thermal controls");
    }
    std::ostringstream identity_stream;
    identity_stream << path.size() << ':' << path << species.size() << ':';
    for (auto const& name : species) {
        identity_stream << name.size() << ':' << name;
    }
    auto const identity = identity_stream.str();
    if (!Agree(identity.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()))) {
        return fail("Material identity exceeds MPI count bound");
    }
    int identity_size = static_cast<int>(identity.size());
    amrex::ParallelDescriptor::Bcast(&identity_size, 1,
                                     amrex::ParallelDescriptor::IOProcessorNumber());
    std::string reference(static_cast<std::size_t>(identity_size), '\0');
    if (amrex::ParallelDescriptor::IOProcessor()) {
        reference = identity;
    }
    amrex::ParallelDescriptor::Bcast(reference.data(), identity_size,
                                     amrex::ParallelDescriptor::IOProcessorNumber());
    if (!Agree(reference == identity)) {
        return fail("Rank-inconsistent material species/path");
    }
    if (photons.TotalNumberOfParticles() != 0) {
        return fail("Live photons are unsupported");
    }
    auto const names = particles.GetSpeciesNames();
    std::vector<std::string> unique;
    for (auto const& name : species) {
        if (std::find(names.begin(), names.end(), name) == names.end() ||
            std::find(unique.begin(), unique.end(), name) != unique.end()) {
            valid = false;
            continue;
        }
        unique.push_back(name);
        auto const& ions = particles.GetParticleContainerFromName(name);
        valid = valid && &ions != &photons && std::isfinite(ions.getMass()) && ions.getMass() > 0 &&
                ions.ParticleBoxArray(0) == boxes &&
                ions.ParticleDistributionMap(0) == distribution;
        auto const& attributes = ions.GetRealSoANames();
        for (auto const* suffix : {"_ux", "_uy", "_uz", "_work"}) {
            auto const attribute = "radiation_impulse_" + path + suffix;
            valid = valid &&
                    std::find(attributes.begin(), attributes.end(), attribute) != attributes.end();
        }
    }
    if (!Agree(valid)) {
        return fail("Invalid native material or unregistered carry path");
    }

    amrex::MultiFab candidate(boxes, distribution, 1, radiation.nGrowVect());
    amrex::MultiFab temperature(nodal_temperature.boxArray(), distribution, 1,
                                nodal_temperature.nGrowVect());
    amrex::MultiFab total_heat(boxes, distribution, 1, material_energy.nGrowVect());
    amrex::MultiFab total_impulse(boxes, distribution, 3, 0);
    amrex::MultiFab total_work(boxes, distribution, 1, 0);
    amrex::MultiFab total_momentum_carry(boxes, distribution, 3, 0);
    int rejected = 0;
    for (int subdivision = 0; subdivision <= options.max_subdivisions; ++subdivision) {
        int const steps = 1 << subdivision;
        auto const sub_dt = dt / steps;
        if (!(sub_dt > 0 && time + sub_dt > time)) {
            return fail("Unrepresentable moving thermal retry timestep");
        }
        auto material = std::make_unique<ParticleImpulseMaterial>(particles, species);
        Copy(candidate, radiation);
        Copy(temperature, nodal_temperature);
        total_heat.setVal(0);
        total_impulse.setVal(0);
        total_work.setVal(0);
        total_momentum_carry.setVal(0);
        amrex::Real carry_change = 0;
        amrex::Real thermal_realization_residual = 0;
        amrex::GpuArray<amrex::Real,3> wall_momentum{};
        int iterations = 0;
        bool accepted = true;
        for (int step = 0; step < steps; ++step) {
            amrex::MultiFab initial_impulse(boxes, distribution, 3, 0);
            amrex::MultiFab debit(boxes, distribution, 1, 0), scale(boxes, distribution, 1, 0);
            initial_impulse.setVal(0);
            ParticleImpulseTransaction initial;
            if (!initial.Stage(*material, path, initial_impulse, true,
                               options.material.assignment)) {
                accepted = false;
                result.reason = "Initial native material staging rejected";
                break;
            }
            std::unique_ptr<MovingThermalDiffusionFieldResult> field;
            auto replay = [&] (ParticleImpulseTransaction const& staged, amrex::MultiFab& impulse,
                               amrex::MultiFab& mechanical_debit, amrex::MultiFab& operands) {
                auto trial = TryAdvanceMovingThermalDiffusionField(
                    candidate, temperature, staged.WorkVelocity(), staged.RequestedWork(), geometry,
                    time + step * sub_dt, sub_dt, options.field, callbacks);
                if (!trial.valid) {
                    result.reason =
                        "Field failure=" + std::to_string(static_cast<int>(trial.failure)) +
                        " coupled=" + std::to_string(static_cast<int>(trial.coupled.failure));
                    return false;
                }
                amrex::MultiFab::Copy(impulse, trial.material_impulse, 0, 0, 3, 0);
                amrex::MultiFab::Copy(mechanical_debit, trial.mechanical_debit, 0, 0, 1, 0);
                amrex::MultiFab::Copy(operands, trial.mechanical_operand_scale, 0, 0, 1, 0);
                field = std::make_unique<MovingThermalDiffusionFieldResult>(std::move(trial));
                return true;
            };
            if (!Agree(replay(initial, initial_impulse, debit, scale))) {
                accepted = false;
                result.reason = "Initial moving thermal field trial rejected: " + result.reason;
                break;
            }
            std::unique_ptr<ParticleImpulseTransaction> work;
            PacketMaterialWorkIterationReport report;
            if (!TryIteratePacketMaterialWork(*material, path, initial_impulse, replay,
                                              options.material, work, report)) {
                accepted = false;
                if (options.field.coupled.diffusion.verbosity > 0) {
                    amrex::Print() << "Moving thermal rejected work iteration reason="
                        << PacketMaterialWorkIterationReasonName(report.reason)
                        << " impulse_residual=" << report.max_impulse_residual
                        << " radiation_work_residual=" << report.max_radiation_work_residual
                        << " material_work_residual=" << report.max_material_work_excess_residual
                        << " field_detail=" << result.reason << '\n';
                }
                result.reason = "Moving thermal work iteration: " +
                                PacketMaterialWorkIterationReasonName(report.reason);
                break;
            }
            auto const raw = field->radiation.sum(0) - candidate.sum(0) +
                             field->material_energy.sum(0) + report.actual_work +
                             report.energy_carry_change;
            auto const magnitude = field->radiation.norm1(0) + candidate.norm1(0) +
                                   field->material_energy.norm1(0) + work->ActualWork().norm1(0) +
                                   work->EnergyCarryChange().norm1(0);
            if (!Agree(std::isfinite(raw) && std::isfinite(magnitude) &&
                       std::abs(raw) <= options.energy_relative_tolerance * magnitude)) {
                accepted = false;
                result.reason = "Moving thermal substep energy ledger rejected";
                break;
            }
            if (!ClosedMomentumBudget(*work, *field, geometry, sub_dt,
                                      options.energy_relative_tolerance)) {
                accepted = false;
                result.reason = "Moving thermal closed momentum ledger rejected";
                break;
            }
            amrex::MultiFab::Add(total_heat, field->material_energy, 0, 0, 1, 0);
            amrex::MultiFab::Add(total_impulse, work->ActualImpulse(), 0, 0, 3, 0);
            amrex::MultiFab::Add(total_work, work->ActualWork(), 0, 0, 1, 0);
            amrex::MultiFab::Add(total_momentum_carry, work->MomentumCarryChange(), 0, 0,
                                 3, 0);
            for (int d = 0; d < 3; ++d) { wall_momentum[d] += field->radiation_wall_momentum[d]; }
            carry_change += report.energy_carry_change;
            thermal_realization_residual += field->thermal_deposition_redistribution.sum(0);
            iterations += report.iterations;
            amrex::MultiFab::Copy(candidate, field->radiation, 0, 0, 1, 0);
            candidate.FillBoundary(geometry.periodicity());
            amrex::MultiFab::Copy(temperature, field->nodal_temperature, 0, 0, 1, 0);
            temperature.FillBoundary(geometry.periodicity());
            work->Commit(); // Private particles only. A later substep can still
                            // reject.
        }
        if (!accepted) {
            if (options.field.coupled.diffusion.verbosity > 0) {
                amrex::Print() << "Moving thermal retry subdivisions=" << subdivision
                               << " reason=" << result.reason << '\n';
            }
            ++rejected;
            continue;
        }
        auto const raw = candidate.sum(0) - radiation.sum(0) + total_heat.sum(0) +
                         total_work.sum(0) + carry_change;
        auto const magnitude = candidate.norm1(0) + radiation.norm1(0) + total_heat.norm1(0) +
                               total_work.norm1(0) + std::abs(carry_change);
        if (!Agree(std::isfinite(raw) && std::isfinite(magnitude) &&
                   std::abs(raw) <= options.energy_relative_tolerance * magnitude)) {
            ++rejected;
            result.reason = "Moving thermal complete interval ledger rejected";
            continue;
        }
        total_heat.FillBoundary(geometry.periodicity());
        bool finite_wall = true;
        for (auto const value : wall_momentum) { finite_wall = finite_wall && std::isfinite(value); }
        if (!Agree(finite_wall)) {
            return fail("Nonfinite whole-interval thermal wall momentum");
        }
        if (!material->Commit()) {
            return fail("Live material changed before joint commit");
        }
        // All numerical checks and allocations precede this publication block.
        Copy(radiation, candidate);
        Copy(nodal_temperature, temperature);
        Copy(material_energy, total_heat);
        result.valid = true;
        result.reason = "Accepted bounded moving thermal source interval";
        result.work_iterations = iterations;
        result.accepted_substeps = steps;
        result.rejected_attempts = rejected;
        result.actual_material_work = total_work.sum(0);
        result.material_energy_carry_change = carry_change;
        result.raw_energy_residual = raw;
        result.thermal_realization_residual = thermal_realization_residual;
        result.radiation_wall_momentum = wall_momentum;
        result.actual_material_impulse = std::move(total_impulse);
        result.actual_material_work_field = std::move(total_work);
        result.material_momentum_carry_change = std::move(total_momentum_carry);
        return result;
    }
    result.rejected_attempts = rejected;
    return result;
#endif
}
} // namespace warpx::radiation
