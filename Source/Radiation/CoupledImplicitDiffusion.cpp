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
#include <AMReX_iMultiFab.H>

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
                (!callbacks.material_local_residual || callbacks.material_residual) &&
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
        int rhs_min = callbacks.radiation_rhs ? 1 : 0;
        int rhs_max = rhs_min;
        amrex::ParallelDescriptor::ReduceIntMin(rhs_min);
        amrex::ParallelDescriptor::ReduceIntMax(rhs_max);
        if (rhs_min != rhs_max) {
            CoupledImplicitResult rejected;
            rejected.failure = CoupledImplicitFailure::Material;
            return rejected;
        }
        int rebase_min = callbacks.rebase_material_source ? 1 : 0;
        int rebase_max = rebase_min;
        amrex::ParallelDescriptor::ReduceIntMin(rebase_min);
        amrex::ParallelDescriptor::ReduceIntMax(rebase_max);
        if (rebase_min != rebase_max || rebase_max != rhs_max) {
            CoupledImplicitResult rejected;
            rejected.failure = CoupledImplicitFailure::Material;
            return rejected;
        }
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
        bool const local_caloric_accuracy = static_cast<bool>(callbacks.material_local_residual);
        bool const need_source_images = options.require_local_ideal_accuracy || local_caloric_accuracy;
        auto lower_roundoff_image = need_source_images ? temperature() : nullptr;
        auto upper_roundoff_image = need_source_images ? temperature() : nullptr;
        amrex::MultiFab candidate(boxes, distribution, groups, radiation.nGrowVect());
        amrex::MultiFab opacity(boxes, distribution, groups, 1);
        amrex::MultiFab absorption(boxes, distribution, groups, 0);
        amrex::MultiFab emission(boxes, distribution, groups, 0);
        auto trial_rhs = callbacks.radiation_rhs
                             ? std::make_unique<amrex::MultiFab>(boxes, distribution, groups,
                                                                 radiation.nGrowVect())
                             : nullptr;
        auto const& spatial_rhs = trial_rhs ? *trial_rhs : radiation;
        auto previous_rhs = callbacks.rebase_material_source
                                ? std::make_unique<amrex::MultiFab>(boxes, distribution, groups,
                                                                    radiation.nGrowVect())
                                : nullptr;
        bool have_previous_rhs = false;
        amrex::MultiFab source(boxes, distribution, 1, realized_material_energy.nGrowVect());
        amrex::MultiFab requested_source(boxes, distribution, 1,
                                         realized_material_energy.nGrowVect());
        amrex::MultiFab final_source(boxes, distribution, 1, realized_material_energy.nGrowVect());
        auto before_rebase =
            previous_rhs ? std::make_unique<amrex::MultiFab>(boxes, distribution, 1,
                                                             realized_material_energy.nGrowVect())
                         : nullptr;
        if (previous_rhs) {
            source.setVal(0);
            requested_source.setVal(0);
            final_source.setVal(0);
        }
        auto roundoff_source = need_source_images
            ? std::make_unique<amrex::MultiFab>(boxes, distribution, 1,
                                               realized_material_energy.nGrowVect())
            : nullptr;
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
        auto coefficients_valid = [&] {
            // A trial EOS/table state may be inadmissible even when the live
            // state is valid. Reject that coupled attempt collectively before
            // entering the spatial solver's fatal input-configuration checks.
            // In particular this permits subcycling to retry a table-domain
            // overshoot without committing any radiation/material candidate.
            // The optically-thick admission limit is part of the same trial
            // domain. A finite positive opacity may still cross this limit.
            amrex::Real min_dx = dx[0];
            for (int d = 1; d < AMREX_SPACEDIM; ++d) {
                min_dx = amrex::min(min_dx, dx[d]);
            }
            auto const minimum_optical_depth = controls.minimum_optical_depth;
            amrex::ReduceOps<amrex::ReduceOpMax> ops;
            amrex::ReduceData<int> data(ops);
            using Tuple = typename decltype(data)::Type;
            for (amrex::MFIter mfi(opacity); mfi.isValid(); ++mfi) {
                auto const kr = opacity.const_array(mfi);
                auto const rate = absorption.const_array(mfi);
                auto const emit = emission.const_array(mfi);
                ops.eval(
                    mfi.validbox(), groups, data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int g) -> Tuple {
                        return {!(kr(i, j, k, g) >= 0 &&
                                  amrex::Math::isfinite(kr(i, j, k, g)) &&
                                  kr(i, j, k, g) * min_dx >=
                                      minimum_optical_depth &&
                                  rate(i, j, k, g) >= 0 &&
                                  amrex::Math::isfinite(rate(i, j, k, g)) &&
                                  emit(i, j, k, g) >= 0 &&
                                  amrex::Math::isfinite(emit(i, j, k, g)))};
                    });
            }
            int invalid = amrex::get<0>(data.value());
            amrex::ParallelDescriptor::ReduceIntMax(invalid);
            return invalid == 0;
        };
        auto evaluate_coefficients = [&] (amrex::MultiFab const& te) {
            callbacks.coefficients(te, opacity, absorption, emission, time + dt);
            if (!coefficients_valid()) {
                return false;
            }
            if (trial_rhs) {
                amrex::MultiFab::Copy(*trial_rhs, radiation, 0, 0, groups, radiation.nGrowVect());
                callbacks.radiation_rhs(radiation, *trial_rhs);
                // The spatial API treats invalid RHS as fatal configuration.
                // A nonlinear callback-domain failure must remain a rollback.
                bool const finite = trial_rhs->is_finite(0, groups, 0);
                int invalid = !finite;
                for (int g = 0; g < groups; ++g) {
                    if (!(trial_rhs->min(g) >= 0)) {
                        invalid = 1;
                    }
                }
                amrex::ParallelDescriptor::ReduceIntMax(invalid);
                if (invalid) {
                    return false;
                }
                trial_rhs->FillBoundary(geometry.periodicity());
                if (previous_rhs) {
                    if (have_previous_rhs) {
                        for (auto* q : {&source, &requested_source, &final_source}) {
                            amrex::MultiFab::Copy(*before_rebase, *q, 0, 0, 1, q->nGrowVect());
                            callbacks.rebase_material_source(*previous_rhs, *trial_rhs, *q);
                            if (!q->is_finite(0, 1, 0)) {
                                return false;
                            }
                            amrex::ReduceOps<amrex::ReduceOpMax> ops;
                            amrex::ReduceData<int> data(ops);
                            using Tuple = typename decltype(data)::Type;
                            for (amrex::MFIter mfi(*q); mfi.isValid(); ++mfi) {
                                auto const before = before_rebase->const_array(mfi);
                                auto const after = q->const_array(mfi);
                                auto const old_rhs = previous_rhs->const_array(mfi);
                                auto const new_rhs = trial_rhs->const_array(mfi);
                                ops.eval(mfi.validbox(), data,
                                         [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                                             amrex::Real delta = 0;
                                             for (int g = 0; g < groups; ++g) {
                                                 delta += new_rhs(i, j, k, g) - old_rhs(i, j, k, g);
                                             }
                                             return {after(i, j, k) != before(i, j, k) + delta ? 1
                                                                                               : 0};
                                         });
                            }
                            int corrupt = amrex::get<0>(data.value());
                            amrex::ParallelDescriptor::ReduceIntMax(corrupt);
                            if (corrupt) {
                                return false;
                            }
                            q->FillBoundary(geometry.periodicity());
                        }
                    }
                    amrex::MultiFab::Copy(*previous_rhs, *trial_rhs, 0, 0, groups,
                                          radiation.nGrowVect());
                    have_previous_rhs = true;
                }
            }
            return true;
        };
        amrex::Real iteration_relaxation = options.relaxation;
        bool reduced_check_image_step = false;
        auto reduce_search_step = [&] {
            // Convex nonlinear search damping; this changes no physical source
            // or acceptance equation. Persist the smaller step across trials so
            // a stiff caloric image does not repeat the same overshoot cycle.
            iteration_relaxation = amrex::max(0.5_rt * iteration_relaxation,
                16 * std::numeric_limits<amrex::Real>::epsilon());
        };
        auto previous_residual = options.safeguard_material_iteration ? temperature() : nullptr;
        auto current_residual = options.safeguard_material_iteration ? temperature() : nullptr;
        std::unique_ptr<amrex::iMultiFab> residual_owner;
        if (options.safeguard_material_iteration) {
            residual_owner = nodal_temperature.OwnerMask(geometry.periodicity());
        }
        bool residual_history = false;
        amrex::Real previous_relaxation = iteration_relaxation;
        int primary_safeguards = 0;
        int check_safeguards = 0;
        auto update_temperature_trial = [&] (bool full_caloric_image) {
            if (options.safeguard_material_iteration && full_caloric_image) {
                amrex::MultiFab::Copy(*current_residual, *candidate_temperature, 0, 0, 1,
                                      nodal_temperature.nGrowVect());
                amrex::MultiFab::Subtract(*current_residual, *trial, 0, 0, 1,
                                          nodal_temperature.nGrowVect());
                if (residual_history &&
                    (!options.require_local_ideal_accuracy || callbacks.material_residual)) {
                    // Vector Aitken delta-squared relaxation of the full fixed-
                    // point residual. Count each periodic node exactly once so
                    // decomposition changes do not alter its dot-product metric.
                    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> ops;
                    amrex::ReduceData<amrex::Real, amrex::Real> data(ops);
                    using Tuple = typename decltype(data)::Type;
                    for (amrex::MFIter mfi(*current_residual); mfi.isValid(); ++mfi) {
                        auto const current = current_residual->const_array(mfi);
                        auto const previous = previous_residual->const_array(mfi);
                        auto const owner = residual_owner->const_array(mfi);
                        ops.eval(mfi.validbox(), data,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                                if (!owner(i,j,k)) { return {0.0_rt, 0.0_rt}; }
                                auto const delta = current(i,j,k)-previous(i,j,k);
                                return {previous(i,j,k)*delta, delta*delta};
                            });
                    }
                    auto const values = data.value();
                    amrex::Real dots[2] = {amrex::get<0>(values), amrex::get<1>(values)};
                    amrex::ParallelDescriptor::ReduceRealSum(dots, 2);
                    if (dots[1] > 0 && std::isfinite(dots[0]) && std::isfinite(dots[1])) {
                        auto const aitken = -previous_relaxation * dots[0] / dots[1];
                        if (aitken > 0 && std::isfinite(aitken)) {
                            auto const arithmetic_floor =
                                16 * std::numeric_limits<amrex::Real>::epsilon();
                            iteration_relaxation = amrex::min(options.relaxation,
                                amrex::max(aitken, arithmetic_floor));
                        }
                    }
                }
                amrex::MultiFab::Copy(*previous_residual, *current_residual, 0, 0, 1,
                                      nodal_temperature.nGrowVect());
                previous_relaxation = iteration_relaxation;
                residual_history = true;
            } else {
                // A source-backtracked image is only an admissible initializer;
                // it must never enter the secant history of the physical map.
                residual_history = false;
            }
            if (options.safeguard_material_iteration) {
                // Add the small search increment directly to avoid rounding two
                // separately scaled background temperatures near convergence.
                auto const relaxation = iteration_relaxation;
                for (amrex::MFIter mfi(*trial); mfi.isValid(); ++mfi) {
                    auto const current = trial->array(mfi);
                    auto const image = candidate_temperature->const_array(mfi);
                    amrex::ParallelFor(mfi.fabbox(),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                            auto const before = current(i,j,k);
                            auto const target = image(i,j,k);
                            auto next = before + relaxation*(target-before);
                            if (next == before && target != before) {
                                // A nonzero relaxed direction can round away.
                                // Advance the scratch search by one representable
                                // value within [before,target]; accepted fields
                                // and the original equation gates are untouched.
                                next = std::nextafter(before, target);
                            }
                            current(i,j,k) = next;
                        });
                }
            } else {
                amrex::MultiFab::LinComb(*trial, iteration_relaxation, *candidate_temperature, 0,
                    1 - iteration_relaxation, *trial, 0, 0, 1, nodal_temperature.nGrowVect());
            }
        };
        for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
            result.iterations = iteration + 1;
            if (!evaluate_coefficients(*trial)) {
                return fail(CoupledImplicitFailure::Material);
            }
            opacity.FillBoundary(geometry.periodicity());
            if (callbacks.conservative_material_request) {
                // Energy elimination prevents amplification of the inner
                // equation defect in the thermal request. This is an inexact
                // nonlinear trial solve: reserve half the final equation budget
                // for it instead of imposing an artificial stiff-source floor
                // that can differ across decompositions. Final constitutive,
                // material and global-energy gates below are unchanged.
                spatial_controls.tolerance = 0.5_rt * controls.tolerance;
                spatial_controls.linear_tolerance = amrex::min(controls.linear_tolerance,
                    0.5_rt * spatial_controls.tolerance);
            } else if (options.safeguard_material_iteration) {
                // A spatial equation defect perturbs the caloric image, then
                // gets amplified by the stiff source at the new temperature.
                // Tighten the inner solve with source stiffness, while avoiding
                // a requested tolerance below one diagonal energy-ULP spacing.
                amrex::Real maximum_rate = 0;
                for (int g = 0; g < groups; ++g) {
                    maximum_rate = amrex::max(maximum_rate, absorption.norm0(g));
                }
                auto const stiffness = 1 + dt * maximum_rate;
                auto const nominal = 0.1_rt * controls.tolerance;
                auto const energy_spacing =
                    std::numeric_limits<amrex::Real>::epsilon() * stiffness;
                spatial_controls.tolerance = amrex::min(nominal,
                    amrex::max(energy_spacing, nominal / stiffness));
                spatial_controls.linear_tolerance = amrex::min(controls.linear_tolerance,
                    0.5_rt * spatial_controls.tolerance);
            }
            amrex::MultiFab::Copy(candidate, spatial_rhs, 0, 0, groups, radiation.nGrowVect());
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
            if (callbacks.conservative_material_request) {
                // Eliminate the source using the spatial energy equation for
                // this nonlinear trial. This prevents its small inner equation
                // defect from being amplified through a stiff caloric image.
                // The radiation candidate is immutable; final constitutive and
                // material residual checks below remain independent and intact.
                callbacks.conservative_material_request(candidate, source);
                if (!source.is_finite(0, 1, 0)) {
                    return fail(CoupledImplicitFailure::Material);
                }
                source.FillBoundary(geometry.periodicity());
            } else {
                make_source(source);
            }
            amrex::MultiFab::Copy(requested_source, source, 0, 0, 1, source.nGrowVect());
            amrex::MultiFab::Copy(*candidate_temperature, nodal_temperature, 0, 0, 1,
                                  nodal_temperature.nGrowVect());
            auto realization = callbacks.material_response(*candidate_temperature, source);
            if (!material_valid(realization, *candidate_temperature, source)) {
                if (!options.safeguard_material_iteration) {
                    return fail(CoupledImplicitFailure::Material);
                }
                // A frozen-temperature stiff emission trial can request more
                // electron energy than exists even when the coupled root is
                // positive. A backtracked request supplies an admissible search
                // direction, never an accepted physical source or update.
                bool admissible_trial = false;
                amrex::Real fraction = 1;
                for (int backtrack = 0; backtrack < 32; ++backtrack) {
                    fraction *= 0.5_rt;
                    amrex::MultiFab::Copy(source, requested_source, 0, 0, 1,
                                          source.nGrowVect());
                    source.mult(fraction);
                    source.FillBoundary(geometry.periodicity());
                    amrex::MultiFab::Copy(*candidate_temperature, nodal_temperature, 0, 0, 1,
                                          nodal_temperature.nGrowVect());
                    auto const trial_realization =
                        callbacks.material_response(*candidate_temperature, source);
                    if (material_valid(trial_realization, *candidate_temperature, source)) {
                        admissible_trial = true;
                        break;
                    }
                }
                if (!admissible_trial) { return fail(CoupledImplicitFailure::Material); }
                reduce_search_step();
                ++primary_safeguards;
                update_temperature_trial(false);
                continue;
            }
            if (options.safeguard_material_iteration && controls.verbosity > 0 &&
                (iteration < 8 || iteration + 8 >= options.max_iterations)) {
                amrex::MultiFab::Copy(*current_residual, *candidate_temperature, 0, 0, 1, 0);
                amrex::MultiFab::Subtract(*current_residual, *trial, 0, 0, 1, 0);
                auto const image_delta = current_residual->norm0(0);
                amrex::Print() << "Coupled primary image iteration=" << iteration + 1
                    << " temperature_delta=" << image_delta
                    << " inner_residual=" << spatial.maximum_relative_residual
                    << " inner_tolerance=" << spatial_controls.tolerance
                    << " relaxation=" << iteration_relaxation << '\n';
            }
            // Re-evaluate BOTH equations at the actual native material response,
            // never accept the spatial residual from the lagged trial temperature.
            if (!evaluate_coefficients(*candidate_temperature)) {
                return fail(CoupledImplicitFailure::Material);
            }
            opacity.FillBoundary(geometry.periodicity());
            auto equation = EvaluateImplicitDiffusionResidual(spatial_rhs, candidate, opacity,
                                                              geometry, time, dt, controls);
            if (equation.failure != ImplicitDiffusionFailure::None) {
                return fail(CoupledImplicitFailure::Spatial);
            }
            amrex::Real norms[6] = {0, 0, 0, 0, 0, 0};
            bool material_converged = false;
            bool const temperature_roundoff = options.allow_ideal_temperature_roundoff &&
                !callbacks.material_residual;
            bool const local_temperature_accuracy = options.require_local_ideal_accuracy &&
                temperature_roundoff;
            bool restart_material_trial = false;
            for (int reconciliation = 0; reconciliation < 4; ++reconciliation) {
                make_source(final_source);
                amrex::MultiFab::Copy(*material_check, nodal_temperature, 0, 0, 1,
                                      nodal_temperature.nGrowVect());
                auto const final_realization =
                    callbacks.material_response(*material_check, final_source);
                if (!material_valid(final_realization, *material_check, final_source)) {
                    if (!options.safeguard_material_iteration) {
                        return fail(CoupledImplicitFailure::Material);
                    }
                    // The first material image is admissible but its newly
                    // evaluated emission can overshoot. Advance only the
                    // nonlinear temperature trial; accept no physical fields.
                    restart_material_trial = true;
                    break;
                }
                if (local_temperature_accuracy || local_caloric_accuracy) {
                    // Propagate the arithmetic error of absorption minus
                    // emission locally through the native caloric map. Use the
                    // bracket around the check image. Its consistency is
                    // checked separately: a corrupted check must never supply
                    // its own roundoff allowance. Rounded endpoints need not
                    // be symmetric about the unperturbed image.
                    for (int sign : {-1, 1}) {
                        make_source(*roundoff_source);
                        for (amrex::MFIter mfi(*roundoff_source); mfi.isValid(); ++mfi) {
                            auto const q = roundoff_source->array(mfi);
                            auto const e = candidate.const_array(mfi);
                            auto const a = absorption.const_array(mfi);
                            auto const eta = emission.const_array(mfi);
                            amrex::ParallelFor(mfi.validbox(),
                                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                                    amrex::Real magnitude = 0;
                                    auto const volume = CellVolume(i, dx, lower, domain_lo);
                                    for (int g = 0; g < groups; ++g) {
                                        magnitude += dt * (std::abs(a(i,j,k,g)*e(i,j,k,g)) +
                                                          volume*std::abs(eta(i,j,k,g)));
                                    }
                                    q(i,j,k) += sign * 16 *
                                        std::numeric_limits<amrex::Real>::epsilon() * magnitude;
                                });
                        }
                        roundoff_source->FillBoundary(geometry.periodicity());
                        auto& image = sign < 0 ? *lower_roundoff_image : *upper_roundoff_image;
                        amrex::MultiFab::Copy(image, nodal_temperature, 0, 0, 1,
                                              nodal_temperature.nGrowVect());
                        auto const probe = callbacks.material_response(image, *roundoff_source);
                        if (!material_valid(probe, image, *roundoff_source)) {
                            return fail(CoupledImplicitFailure::Material);
                        }
                    }
                }
                amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax,
                                 amrex::ReduceOpMax, amrex::ReduceOpMax,
                                 amrex::ReduceOpMax, amrex::ReduceOpMax> ops;
                amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                                  amrex::Real, amrex::Real> data(ops);
                using Tuple = typename decltype(data)::Type;
                for (amrex::MFIter mfi(*candidate_temperature); mfi.isValid(); ++mfi) {
                    auto const old = nodal_temperature.const_array(mfi);
                    auto const current = candidate_temperature->const_array(mfi);
                    auto const check = material_check->const_array(mfi);
                    auto const lower_image = local_temperature_accuracy
                        ? lower_roundoff_image->const_array(mfi) : check;
                    auto const upper_image = local_temperature_accuracy
                        ? upper_roundoff_image->const_array(mfi) : check;
                    ops.eval(mfi.validbox(), data,
                             [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                                 auto const old_temperature = old(i, j, k);
                                 auto const candidate_t_value = current(i, j, k);
                                 auto const check_temperature = check(i, j, k);
                                 auto const local_scale = amrex::max(
                                     std::abs(check_temperature - old_temperature),
                                     std::abs(candidate_t_value - old_temperature));
                                 auto const infinity =
                                     std::numeric_limits<amrex::Real>::infinity();
                                 auto const candidate_ulp = amrex::max(
                                     std::abs(std::nextafter(candidate_t_value, infinity) -
                                              candidate_t_value),
                                     std::abs(candidate_t_value -
                                         std::nextafter(candidate_t_value, -infinity)));
                                 auto const check_ulp = amrex::max(
                                     std::abs(std::nextafter(check_temperature, infinity) -
                                              check_temperature),
                                     std::abs(check_temperature -
                                         std::nextafter(check_temperature, -infinity)));
                                 // The ideal caloric map adds a local increment to T.
                                 // Allow one half-ULP from each independently rounded
                                 // image and 8 eps for the local increment/remap
                                 // arithmetic. Do not scale this allowance by the
                                 // hottest node's change elsewhere in the domain.
                                 auto const local_roundoff = local_temperature_accuracy
                                     ? 0.5_rt * (candidate_ulp + check_ulp) +
                                           8 * std::numeric_limits<amrex::Real>::epsilon() *
                                               local_scale +
                                           amrex::max(std::abs(upper_image(i,j,k) - check_temperature),
                                                      std::abs(check_temperature - lower_image(i,j,k)))
                                     : 0.0_rt;
                                 // Keep the original global map allowance for
                                 // homogeneous callers. Hot/cold callers use the
                                 // separate node-local ULP allowance above.
                                 auto const legacy_roundoff = temperature_roundoff
                                     ? 128 * std::numeric_limits<amrex::Real>::epsilon() *
                                           std::abs(old_temperature) +
                                           64 * std::numeric_limits<amrex::Real>::epsilon() *
                                               std::abs(candidate_t_value) +
                                           64 * std::numeric_limits<amrex::Real>::epsilon() *
                                               std::abs(check_temperature)
                                     : 0.0_rt;
                                 auto const mismatch =
                                     std::abs(check_temperature - candidate_t_value);
                                 auto const local_relative = local_temperature_accuracy &&
                                         local_scale > local_roundoff
                                     ? mismatch / local_scale
                                     : local_temperature_accuracy
                                         ? mismatch / amrex::max(local_roundoff,
                                               std::numeric_limits<amrex::Real>::min())
                                         : 0.0_rt;
                                 auto const local_not_converged = local_temperature_accuracy &&
                                     (mismatch > options.material_tolerance * local_scale +
                                                     local_roundoff ||
                                      check_temperature < lower_image(i,j,k) - check_ulp ||
                                      check_temperature > upper_image(i,j,k) + check_ulp);
                                 return {std::abs(check(i, j, k) - current(i, j, k)),
                                         amrex::max(std::abs(check(i, j, k) - old(i, j, k)),
                                                    std::abs(current(i, j, k) - old(i, j, k))),
                                         legacy_roundoff, local_roundoff, local_relative,
                                         local_not_converged ? 1.0_rt : 0.0_rt};
                             });
                }
                auto const values = data.value();
                norms[0] = amrex::get<0>(values);
                norms[1] = amrex::get<1>(values);
                norms[2] = amrex::get<2>(values);
                norms[3] = amrex::get<3>(values);
                norms[4] = amrex::get<4>(values);
                norms[5] = amrex::get<5>(values);
                amrex::ParallelDescriptor::ReduceRealMax(norms, 6);
                if (callbacks.material_residual) {
                    auto const caloric = callbacks.material_residual(
                        nodal_temperature, *candidate_temperature, *material_check);
                    norms[0] = caloric[0];
                    norms[1] = caloric[1];
                    int invalid_metric = !std::isfinite(norms[0]) ||
                        !std::isfinite(norms[1]) || norms[0] < 0 || norms[1] < 0;
                    amrex::ParallelDescriptor::ReduceIntMax(invalid_metric);
                    if (invalid_metric) {
                        return fail(CoupledImplicitFailure::Material);
                    }
                    amrex::ParallelDescriptor::ReduceRealMax(norms, 2);
                }
                if (local_caloric_accuracy) {
                    auto const local = callbacks.material_local_residual(
                        nodal_temperature, *candidate_temperature, *material_check,
                        *lower_roundoff_image, *upper_roundoff_image, options.material_tolerance);
                    int invalid = !local.valid ||
                        !std::isfinite(local.maximum_relative_residual) ||
                        !std::isfinite(local.maximum_roundoff_bound) ||
                        local.maximum_relative_residual < 0 || local.maximum_roundoff_bound < 0;
                    amrex::ParallelDescriptor::ReduceIntMax(invalid);
                    if (invalid != 0) { return fail(CoupledImplicitFailure::Material); }
                    norms[3] = local.maximum_roundoff_bound;
                    norms[4] = local.maximum_relative_residual;
                    norms[5] = local.converged ? 0 : 1;
                    amrex::ParallelDescriptor::ReduceRealMax(norms+3, 3);
                }
                result.material_relative_residual = norms[1] > 0 ? norms[0] / norms[1] : norms[0];
                result.material_absolute_residual = norms[0];
                result.material_roundoff_bound = local_temperature_accuracy || local_caloric_accuracy
                    ? norms[3] : norms[2];
                result.material_maximum_local_relative_residual = norms[4];
                material_converged = local_temperature_accuracy || local_caloric_accuracy
                    ? norms[5] == 0
                    : result.material_relative_residual <= options.material_tolerance ||
                          (temperature_roundoff && norms[0] <=
                               options.material_tolerance * norms[1] + norms[2]);
                if (local_temperature_accuracy && !material_converged &&
                    iteration >= 2 && reconciliation < 3 && norms[0] <= norms[2] &&
                    equation.maximum_relative_residual <= controls.tolerance) {
                    // Energy elimination can lose a few temperature ULPs when
                    // subtracting background radiation energies. Near that
                    // arithmetic floor, try the independently evaluated physical
                    // caloric image instead. This is a new scratch candidate,
                    // not a residual correction: recompute coefficients and
                    // recheck BOTH equations and the raw energy ledger before
                    // accepting it. The global bound only triggers a trial;
                    // final acceptance remains node-local.
                    bool polished = false;
                    amrex::Real fraction = options.relaxation;
                    for (int search = 0; search < 16; ++search, fraction *= 0.5_rt) {
                        if (!evaluate_coefficients(*candidate_temperature)) {
                            return fail(CoupledImplicitFailure::Material);
                        }
                        make_source(*roundoff_source);
                        amrex::MultiFab::LinComb(final_source, fraction, *roundoff_source, 0,
                            1 - fraction, requested_source, 0, 0, 1, source.nGrowVect());
                        amrex::MultiFab::Copy(*roundoff_source, final_source,
                            0, 0, 1, source.nGrowVect());
                        amrex::MultiFab::Copy(*upper_roundoff_image, nodal_temperature,
                            0, 0, 1, nodal_temperature.nGrowVect());
                        auto probe = callbacks.material_response(*upper_roundoff_image,
                                                                 *roundoff_source);
                        if (!material_valid(probe, *upper_roundoff_image, *roundoff_source)) {
                            continue;
                        }
                        if (!evaluate_coefficients(*upper_roundoff_image)) {
                            continue;
                        }
                        make_source(*roundoff_source);
                        amrex::MultiFab::Copy(*lower_roundoff_image, nodal_temperature,
                            0, 0, 1, nodal_temperature.nGrowVect());
                        probe = callbacks.material_response(*lower_roundoff_image, *roundoff_source);
                        if (!material_valid(probe, *lower_roundoff_image, *roundoff_source)) {
                            continue;
                        }
                        amrex::MultiFab::Subtract(*lower_roundoff_image, *upper_roundoff_image,
                                                  0, 0, 1, 0);
                        if (lower_roundoff_image->norm0(0) > 0.9_rt * norms[0]) { continue; }
                        amrex::MultiFab::Copy(requested_source, final_source,
                            0, 0, 1, source.nGrowVect());
                        amrex::MultiFab::Copy(source, final_source, 0, 0, 1, source.nGrowVect());
                        amrex::MultiFab::Copy(*candidate_temperature, nodal_temperature,
                            0, 0, 1, nodal_temperature.nGrowVect());
                        realization = callbacks.material_response(*candidate_temperature, source);
                        if (!material_valid(realization, *candidate_temperature, source)) {
                            return fail(CoupledImplicitFailure::Material);
                        }
                        polished = true;
                        break;
                    }
                    if (!evaluate_coefficients(*candidate_temperature)) {
                        return fail(CoupledImplicitFailure::Material);
                    }
                    opacity.FillBoundary(geometry.periodicity());
                    equation = EvaluateImplicitDiffusionResidual(spatial_rhs, candidate, opacity,
                                                                 geometry, time, dt, controls);
                    if (equation.failure != ImplicitDiffusionFailure::None) {
                        return fail(CoupledImplicitFailure::Spatial);
                    }
                    if (polished) { continue; }
                }
                if (material_converged ||
                    equation.maximum_relative_residual > controls.tolerance || iteration < 2 ||
                    reconciliation == 3 || !options.reconcile_material_source) {
                    break;
                }
                // The material callback rebases final_source to realized energy;
                // restore the actual radiation request before correcting it.
                make_source(final_source);
                reconcile_source();
                ++result.source_consistency_corrections;
                equation = EvaluateImplicitDiffusionResidual(spatial_rhs, candidate, opacity,
                                                             geometry, time, dt, controls);
                if (equation.failure != ImplicitDiffusionFailure::None) {
                    return fail(CoupledImplicitFailure::Spatial);
                }
            }
            if (restart_material_trial) {
                // This check image applies the stiff source a second time at
                // the new temperature. Repeated inadmissibility of that check
                // alone does not prove the primary fixed-point step diverges.
                if (!reduced_check_image_step) {
                    reduce_search_step();
                    reduced_check_image_step = true;
                }
                ++check_safeguards;
                update_temperature_trial(true);
                continue;
            }
            amrex::Real raw =
                source.sum(0, false) + equation.escaped_energy - equation.injected_energy;
            amrex::Real scale = source.norm1(0);
            for (int g = 0; g < groups; ++g) {
                auto const previous = spatial_rhs.sum(g, false);
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
                               << " local material residual="
                               << result.material_maximum_local_relative_residual
                               << " material mismatch=" << norms[0]
                               << " global material heating scale=" << norms[1]
                               << " legacy material roundoff=" << norms[2]
                               << " local material roundoff=" << norms[3]
                               << " raw energy residual=" << result.raw_energy_relative_residual
                               << " inner equation residual=" << spatial.maximum_relative_residual
                               << " inner tolerance=" << spatial_controls.tolerance
                               << " search relaxation=" << iteration_relaxation
                               << " primary safeguards=" << primary_safeguards
                               << " check safeguards=" << check_safeguards << '\n';
            }
            if (equation.maximum_relative_residual <= controls.tolerance &&
                material_converged &&
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
            update_temperature_trial(true);
        }
        if (options.safeguard_material_iteration) {
            amrex::Print() << "Coupled nonlinear search exhausted: relaxation="
                << iteration_relaxation << " primary safeguards=" << primary_safeguards
                << " check safeguards=" << check_safeguards << '\n';
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
        int shifted_rhs = callbacks.radiation_rhs ? 1 : 0;
        amrex::ParallelDescriptor::ReduceIntMax(shifted_rhs);
        if (shifted_rhs) {
            CoupledImplicitResult rejected;
            rejected.failure = CoupledImplicitFailure::Material;
            return rejected;
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
                aggregate.material_absolute_residual = amrex::max(
                    aggregate.material_absolute_residual, last.material_absolute_residual);
                aggregate.material_roundoff_bound = amrex::max(
                    aggregate.material_roundoff_bound, last.material_roundoff_bound);
                aggregate.material_maximum_local_relative_residual = amrex::max(
                    aggregate.material_maximum_local_relative_residual,
                    last.material_maximum_local_relative_residual);
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
