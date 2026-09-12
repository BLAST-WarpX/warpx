/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "CoupledImplicitDiffusion.H"

#include "CellVolume.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>
#include <memory>

using namespace amrex::literals;

namespace warpx::radiation
{
    CoupledImplicitResult
    TryAdvanceCoupledImplicitDiffusion (amrex::MultiFab& radiation,
                                        amrex::MultiFab& nodal_temperature,
                                        amrex::MultiFab& realized_material_energy,
                                        amrex::Geometry const& geometry, amrex::Real time,
                                        amrex::Real dt, CoupledImplicitOptions const& options,
                                        CoupledImplicitCallbacks const& callbacks)
    {
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        WARPX_ABORT_WITH_MESSAGE(
            "Coupled implicit radiation currently requires Cartesian or RZ geometry.");
#endif
#if defined(WARPX_DIM_RZ)
        // WarpX can store its RZ grid in a Cartesian AMReX Geometry and
        // supplies the cylindrical measures explicitly, as the spatial solver
        // does. Standalone AMReX RZ fixtures may instead use Coord()==1.
        bool const compatible_geometry = geometry.Coord() == 0 || geometry.Coord() == 1;
#else
        bool const compatible_geometry = geometry.Coord() == 0;
#endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            compatible_geometry && sizeof(amrex::Real) == sizeof(double) &&
                options.max_iterations > 0 && options.material_tolerance > 0 &&
                std::isfinite(options.material_tolerance) && options.energy_tolerance > 0 &&
                std::isfinite(options.energy_tolerance) && options.relaxation > 0 &&
                options.relaxation <= 1 && callbacks.coefficients && callbacks.material_response &&
                radiation.ixType().cellCentered() && nodal_temperature.ixType().nodeCentered() &&
                nodal_temperature.nComp() == 1 && nodal_temperature.nGrowVect().allGE(1) &&
                amrex::convert(radiation.boxArray(), nodal_temperature.ixType()) ==
                    nodal_temperature.boxArray() &&
                radiation.DistributionMap() == nodal_temperature.DistributionMap() &&
                realized_material_energy.boxArray() == radiation.boxArray() &&
                realized_material_energy.DistributionMap() == radiation.DistributionMap() &&
                realized_material_energy.nComp() == 1 &&
                realized_material_energy.nGrowVect().allGE(1),
            "Invalid stationary coupled radiation/material configuration.");
        auto const& boxes = radiation.boxArray();
        auto const& distribution = radiation.DistributionMap();
        int const groups = radiation.nComp();
        auto const dx = geometry.CellSizeArray();
        auto const lower = geometry.ProbLoArray();
        auto const domain_lo = amrex::lbound(geometry.Domain());
        auto temperature = [&] {
            auto result = std::make_unique<amrex::MultiFab>(
                nodal_temperature.boxArray(), distribution, 1, nodal_temperature.nGrowVect());
            amrex::MultiFab::Copy(*result, nodal_temperature, 0, 0, 1,
                                  nodal_temperature.nGrowVect());
            return result;
        };
        auto trial = temperature();
        auto candidate_temperature = temperature();
        auto material_check = temperature();
        amrex::MultiFab candidate(boxes, distribution, groups, radiation.nGrowVect());
        amrex::MultiFab opacity(boxes, distribution, groups, 1);
        amrex::MultiFab absorption(boxes, distribution, groups, 0);
        amrex::MultiFab emission(boxes, distribution, groups, 0);
        amrex::MultiFab source(boxes, distribution, 1, realized_material_energy.nGrowVect());
        amrex::MultiFab requested_source(boxes, distribution, 1,
                                         realized_material_energy.nGrowVect());
        amrex::MultiFab final_source(boxes, distribution, 1, realized_material_energy.nGrowVect());
        auto controls = options.diffusion;
        controls.absorption_rate = &absorption;
        controls.emissivity = &emission;
        // The inner solve must leave room for the nonlinear material update;
        // its stopping tolerance is stricter than the final equation gate.
        auto spatial_controls = controls;
        spatial_controls.tolerance = 0.1_rt * controls.tolerance;
        // Keep the requested linear solve stricter than the independently
        // checked inner equation, but do not impose a second decade of
        // accuracy: refined-grid double-precision operators can reach their
        // linear residual floor there. The inner and final equation, material,
        // positivity and raw-energy acceptance gates remain unchanged.
        spatial_controls.linear_tolerance =
            amrex::min(controls.linear_tolerance, 0.5_rt * spatial_controls.tolerance);
        auto make_source = [&] (amrex::MultiFab& output) {
            output.setVal(0);
            for (amrex::MFIter mfi(output); mfi.isValid(); ++mfi) {
                auto const energy = candidate.const_array(mfi);
                auto const rate = absorption.const_array(mfi);
                auto const emissivity = emission.const_array(mfi);
                auto const q = output.array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    amrex::Real transfer = 0;
                    amrex::Real const volume = CellVolume(i, dx, lower, domain_lo);
                    for (int g = 0; g < groups; ++g) {
                        transfer += dt * (rate(i, j, k, g) * energy(i, j, k, g) -
                                          volume * emissivity(i, j, k, g));
                    }
                    q(i, j, k) = transfer;
                });
            }
            output.FillBoundary(geometry.periodicity());
        };
        auto reconcile_source = [&] {
            // At fixed candidate T the source is affine in group energy. Make
            // that source agree with the request that actually produced T.
            // This resolves discrete U-to-T roundoff cycles without relaxing
            // either the material or radiation equation gates. Prefer a strong
            // absorbing group with modest diagonal stiffness, not a weak band.
            for (amrex::MFIter mfi(candidate); mfi.isValid(); ++mfi) {
                auto const energy = candidate.array(mfi);
                auto const rate = absorption.const_array(mfi);
                auto const transport = opacity.const_array(mfi);
                auto const target = requested_source.const_array(mfi);
                auto const current = final_source.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    amrex::Real const mismatch = target(i, j, k) - current(i, j, k);
                    int selected = -1;
                    amrex::Real best = 0;
                    for (int g = 0; g < groups; ++g) {
                        amrex::Real const coefficient = dt * rate(i, j, k, g);
                        amrex::Real const capacity = coefficient * energy(i, j, k, g);
                        if (!(coefficient > 0) || capacity + mismatch < 0) {
                            continue;
                        }
                        amrex::Real stiffness = 1 + coefficient;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            stiffness +=
                                2 * dt * PhysConst::c / (3 * transport(i, j, k, g) * dx[d] * dx[d]);
                        }
                        amrex::Real const score = capacity / stiffness;
                        if (score > best && amrex::Math::isfinite(score)) {
                            best = score;
                            selected = g;
                        }
                    }
                    if (selected < 0 || mismatch == 0) {
                        return;
                    }
                    auto const previous = energy(i, j, k, selected);
                    auto next = previous + mismatch / (dt * rate(i, j, k, selected));
                    if (next == previous) {
                        next = std::nextafter(previous,
                                              mismatch > 0 ? std::numeric_limits<amrex::Real>::max()
                                                           : 0.0_rt);
                    }
                    if (next >= 0 && amrex::Math::isfinite(next)) {
                        energy(i, j, k, selected) = next;
                    }
                });
            }
            candidate.FillBoundary(geometry.periodicity());
        };
        CoupledImplicitResult result;
        int spatial_nonlinear_iterations = 0;
        int spatial_linear_iterations = 0;
        auto fail = [&] (CoupledImplicitFailure failure) {
            result.failure = failure;
            result.diffusion = {};
            result.material_realization_residual = 0;
            result.native_source_realization_residual = 0;
            return result;
        };
        auto material_valid = [] (amrex::Real residual, amrex::MultiFab const& te,
                                  amrex::MultiFab const& q) {
            // Do not short-circuit collective field checks on a rank-local
            // callback failure. Every rank must reject the same attempt.
            int invalid = std::isfinite(residual) ? 0 : 1;
            if (!te.is_finite(0, 1, te.nGrowVect())) {
                invalid = 1;
            }
            if (!q.is_finite(0, 1, 0)) {
                invalid = 1;
            }
            if (!(te.min(0) >= 0)) {
                invalid = 1;
            }
            amrex::ParallelDescriptor::ReduceIntMax(invalid);
            return invalid == 0;
        };
        for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
            result.iterations = iteration + 1;
            callbacks.coefficients(*trial, opacity, absorption, emission, time + dt);
            opacity.FillBoundary(geometry.periodicity());
            amrex::MultiFab::Copy(candidate, radiation, 0, 0, groups, radiation.nGrowVect());
            auto const spatial = TryAdvanceImplicitDiffusion(candidate, opacity, geometry, time, dt,
                                                             spatial_controls);
            spatial_nonlinear_iterations += spatial.nonlinear_iterations;
            spatial_linear_iterations += spatial.linear_iterations;
            if (spatial.failure != ImplicitDiffusionFailure::None) {
                amrex::Print() << "Coupled spatial attempt failed: code="
                    << static_cast<int>(spatial.failure)
                    << " nonlinear iterations=" << spatial.nonlinear_iterations
                    << " linear iterations=" << spatial.linear_iterations << '\n';
                return fail(CoupledImplicitFailure::Spatial);
            }
            make_source(source);
            amrex::MultiFab::Copy(requested_source, source, 0, 0, 1, source.nGrowVect());
            amrex::MultiFab::Copy(*candidate_temperature, nodal_temperature, 0, 0, 1,
                                  nodal_temperature.nGrowVect());
            auto const realization = callbacks.material_response(*candidate_temperature, source);
            if (!material_valid(realization, *candidate_temperature, source)) {
                return fail(CoupledImplicitFailure::Material);
            }
            // Re-evaluate BOTH equations at the actual native material response,
            // never accept the spatial residual from the lagged trial temperature.
            callbacks.coefficients(*candidate_temperature, opacity, absorption, emission, time + dt);
            opacity.FillBoundary(geometry.periodicity());
            auto equation = EvaluateImplicitDiffusionResidual(radiation, candidate, opacity,
                                                              geometry, time, dt, controls);
            if (equation.failure != ImplicitDiffusionFailure::None) {
                return fail(CoupledImplicitFailure::Spatial);
            }
            amrex::Real norms[2] = {0, 0};
            for (int reconciliation = 0; reconciliation < 4; ++reconciliation) {
                make_source(final_source);
                amrex::MultiFab::Copy(*material_check, nodal_temperature, 0, 0, 1,
                                      nodal_temperature.nGrowVect());
                auto const final_realization =
                    callbacks.material_response(*material_check, final_source);
                if (!material_valid(final_realization, *material_check, final_source)) {
                    return fail(CoupledImplicitFailure::Material);
                }
                amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> ops;
                amrex::ReduceData<amrex::Real, amrex::Real> data(ops);
                using Tuple = typename decltype(data)::Type;
                for (amrex::MFIter mfi(*candidate_temperature); mfi.isValid(); ++mfi) {
                    auto const old = nodal_temperature.const_array(mfi);
                    auto const current = candidate_temperature->const_array(mfi);
                    auto const check = material_check->const_array(mfi);
                    ops.eval(mfi.validbox(), data,
                             [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                                 return {std::abs(check(i, j, k) - current(i, j, k)),
                                         amrex::max(std::abs(check(i, j, k) - old(i, j, k)),
                                                    std::abs(current(i, j, k) - old(i, j, k)))};
                             });
                }
                auto const values = data.value();
                norms[0] = amrex::get<0>(values);
                norms[1] = amrex::get<1>(values);
                amrex::ParallelDescriptor::ReduceRealMax(norms, 2);
                if (callbacks.material_residual) {
                    auto const caloric = callbacks.material_residual(
                        nodal_temperature, *candidate_temperature, *material_check);
                    norms[0] = caloric[0];
                    norms[1] = caloric[1];
                    if (!std::isfinite(norms[0]) || !std::isfinite(norms[1]) ||
                        norms[0] < 0 || norms[1] < 0) {
                        return fail(CoupledImplicitFailure::Material);
                    }
                }
                result.material_relative_residual = norms[1] > 0 ? norms[0] / norms[1] : norms[0];
                if (result.material_relative_residual <= options.material_tolerance ||
                    equation.maximum_relative_residual > controls.tolerance || iteration < 2 ||
                    reconciliation == 3) {
                    break;
                }
                // The material callback rebases final_source to realized energy;
                // restore the actual radiation request before correcting it.
                make_source(final_source);
                reconcile_source();
                ++result.source_consistency_corrections;
                equation = EvaluateImplicitDiffusionResidual(radiation, candidate, opacity,
                                                             geometry, time, dt, controls);
                if (equation.failure != ImplicitDiffusionFailure::None) {
                    return fail(CoupledImplicitFailure::Spatial);
                }
            }
            amrex::Real raw =
                source.sum(0, false) + equation.escaped_energy - equation.injected_energy;
            amrex::Real scale = source.norm1(0);
            for (int g = 0; g < groups; ++g) {
                auto const previous = radiation.sum(g, false);
                auto const next = candidate.sum(g, false);
                raw += next - previous;
                scale += amrex::max(previous, next);
            }
            scale += equation.escaped_energy + equation.injected_energy;
            if (!std::isfinite(raw) || !std::isfinite(scale)) {
                return fail(CoupledImplicitFailure::Material);
            }
            result.raw_energy_relative_residual = scale > 0 ? std::abs(raw) / scale : std::abs(raw);
            if (iteration + 1 == options.max_iterations) {
                amrex::Print() << "Coupled final trial: radiation residual="
                               << equation.maximum_relative_residual
                               << " material residual=" << result.material_relative_residual
                               << " temperature mismatch=" << norms[0]
                               << " heating temperature scale=" << norms[1]
                               << " raw energy residual=" << result.raw_energy_relative_residual
                               << '\n';
            }
            if (equation.maximum_relative_residual <= controls.tolerance &&
                result.material_relative_residual <= options.material_tolerance &&
                result.raw_energy_relative_residual <= options.energy_tolerance) {
                amrex::MultiFab::Copy(radiation, candidate, 0, 0, groups, radiation.nGrowVect());
                amrex::MultiFab::Copy(nodal_temperature, *candidate_temperature, 0, 0, 1,
                                      nodal_temperature.nGrowVect());
                amrex::MultiFab::Copy(realized_material_energy, source, 0, 0, 1,
                                      realized_material_energy.nGrowVect());
                result.diffusion = equation;
                result.accepted_substeps = 1;
                result.diffusion.nonlinear_iterations = spatial_nonlinear_iterations;
                result.diffusion.linear_iterations = spatial_linear_iterations;
                // Separate the native EOS rounding residual from the remaining
                // nonlinear mismatch between final radiation sources and the
                // material state actually committed.
                result.native_source_realization_residual = realization;
                result.material_realization_residual = -source.sum(0, false);
                for (auto transfer : equation.group_material_energy) {
                    result.material_realization_residual += transfer;
                }
                return result;
            }
            amrex::MultiFab::LinComb(*trial, options.relaxation, *candidate_temperature, 0,
                                     1 - options.relaxation, *trial, 0, 0, 1,
                                     nodal_temperature.nGrowVect());
        }
        return fail(CoupledImplicitFailure::IterationBudget);
    }

    CoupledImplicitResult
    TryAdvanceCoupledImplicitSubcycled (
        amrex::MultiFab& radiation, amrex::MultiFab& nodal_temperature,
        amrex::MultiFab& realized_material_energy, amrex::Geometry const& geometry,
        amrex::Real time, amrex::Real dt, CoupledImplicitOptions const& options,
        CoupledImplicitCallbacks const& callbacks)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            options.max_subdivisions >= 0 && options.max_subdivisions <= 10,
            "Coupled radiation subdivisions must be between zero and ten.");
        if (options.max_subdivisions == 0) {
            return TryAdvanceCoupledImplicitDiffusion(radiation, nodal_temperature,
                realized_material_energy, geometry, time, dt, options, callbacks);
        }
        amrex::MultiFab candidate(radiation.boxArray(), radiation.DistributionMap(),
                                  radiation.nComp(), radiation.nGrowVect());
        amrex::MultiFab temperature(nodal_temperature.boxArray(),
            nodal_temperature.DistributionMap(), 1, nodal_temperature.nGrowVect());
        amrex::MultiFab source(realized_material_energy.boxArray(),
            realized_material_energy.DistributionMap(), 1, realized_material_energy.nGrowVect());
        amrex::MultiFab total_source(source.boxArray(), source.DistributionMap(), 1,
                                     source.nGrowVect());
        CoupledImplicitResult last;
        int rejected = 0;
        auto add_groups = [] (amrex::Vector<amrex::Real>& total,
                              amrex::Vector<amrex::Real> const& next) {
            if (total.empty()) { total.resize(next.size(), 0); }
            for (amrex::Long g = 0; g < next.size(); ++g) { total[g] += next[g]; }
        };
        for (int subdivision = 0; subdivision <= options.max_subdivisions; ++subdivision) {
            int const steps = 1 << subdivision;
            amrex::Real const sub_dt = dt / steps;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(sub_dt > 0 && time + sub_dt > time,
                "Coupled radiation retry timestep is not representable.");
            amrex::MultiFab::Copy(candidate, radiation, 0, 0, radiation.nComp(),
                                  radiation.nGrowVect());
            amrex::MultiFab::Copy(temperature, nodal_temperature, 0, 0, 1,
                                  temperature.nGrowVect());
            total_source.setVal(0);
            CoupledImplicitResult aggregate;
            for (int step = 0; step < steps; ++step) {
                source.setVal(0);
                last = TryAdvanceCoupledImplicitDiffusion(candidate, temperature, source,
                    geometry, time + step * sub_dt, sub_dt, options, callbacks);
                if (last.failure != CoupledImplicitFailure::None) {
                    ++rejected;
                    break;
                }
                amrex::MultiFab::Add(total_source, source, 0, 0, 1, source.nGrowVect());
                aggregate.iterations += last.iterations;
                aggregate.source_consistency_corrections += last.source_consistency_corrections;
                aggregate.material_relative_residual = amrex::max(
                    aggregate.material_relative_residual, last.material_relative_residual);
                aggregate.raw_energy_relative_residual = amrex::max(
                    aggregate.raw_energy_relative_residual, last.raw_energy_relative_residual);
                aggregate.material_realization_residual += last.material_realization_residual;
                aggregate.native_source_realization_residual += last.native_source_realization_residual;
                auto& sum = aggregate.diffusion;
                auto const& next = last.diffusion;
                sum.escaped_energy += next.escaped_energy;
                sum.injected_energy += next.injected_energy;
                sum.numerical_energy_residual += next.numerical_energy_residual;
                sum.maximum_relative_residual = amrex::max(
                    sum.maximum_relative_residual, next.maximum_relative_residual);
                sum.nonlinear_iterations += next.nonlinear_iterations;
                sum.linear_iterations += next.linear_iterations;
                add_groups(sum.group_escaped_energy, next.group_escaped_energy);
                add_groups(sum.group_injected_energy, next.group_injected_energy);
                add_groups(sum.group_material_energy, next.group_material_energy);
            }
            if (last.failure == CoupledImplicitFailure::None) {
                // In addition to every substep gate, check the whole interval
                // using the actual staged state and summed material transfer.
                amrex::Real raw = total_source.sum(0, false) +
                    aggregate.diffusion.escaped_energy - aggregate.diffusion.injected_energy;
                amrex::Real scale = total_source.norm1(0) +
                    aggregate.diffusion.escaped_energy + aggregate.diffusion.injected_energy;
                for (int g = 0; g < radiation.nComp(); ++g) {
                    auto const old = radiation.sum(g, false);
                    auto const next = candidate.sum(g, false);
                    raw += next - old;
                    scale += amrex::max(old, next);
                }
                auto const residual = scale > 0 ? std::abs(raw) / scale : std::abs(raw);
                if (!std::isfinite(residual) || residual > options.energy_tolerance) {
                    last = {};
                    last.failure = CoupledImplicitFailure::IntervalEnergyBalance;
                    last.raw_energy_relative_residual = residual;
                    ++rejected;
                    continue;
                }
                aggregate.raw_energy_relative_residual = amrex::max(
                    aggregate.raw_energy_relative_residual, residual);
                amrex::MultiFab::Copy(radiation, candidate, 0, 0, radiation.nComp(),
                                      radiation.nGrowVect());
                amrex::MultiFab::Copy(nodal_temperature, temperature, 0, 0, 1,
                                      temperature.nGrowVect());
                amrex::MultiFab::Copy(realized_material_energy, total_source, 0, 0, 1,
                                      source.nGrowVect());
                aggregate.accepted_substeps = steps;
                aggregate.rejected_attempts = rejected;
                return aggregate;
            }
        }
        last.rejected_attempts = rejected;
        return last;
    }
} // namespace warpx::radiation
