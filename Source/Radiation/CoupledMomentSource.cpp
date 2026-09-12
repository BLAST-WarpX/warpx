/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "CoupledMomentSource.H"

#include "ImplicitMomentSource.H"
#include "ParticleImpulse.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

using namespace amrex::literals;

namespace warpx::radiation
{
    namespace
    {
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
        RelativeSourceError (amrex::Real error, amrex::Real scale, amrex::Real noise) noexcept
        {
            return scale + noise > 0 ? error / (scale + noise) : error;
        }

        bool
        ThermalNewtonProposal (
            amrex::MultiFab const& x, amrex::MultiFab const& f, amrex::MultiFab& target,
            amrex::Geometry const& geometry,
            std::function<bool(amrex::MultiFab const&, amrex::MultiFab&)> const& evaluate)
        {
            auto make = [&] () {
                return std::make_unique<amrex::MultiFab>(x.boxArray(), x.DistributionMap(), 1,
                                                         x.nGrowVect());
            };
            auto dot = [] (amrex::MultiFab const& a, amrex::MultiFab const& b) {
                return amrex::MultiFab::Dot(a, 0, b, 0, 1, 0);
            };
            auto r = make();
            amrex::MultiFab::Copy(*r, f, 0, 0, 1, 0);
            amrex::MultiFab::Subtract(*r, x, 0, 0, 1, 0);
            auto const initial_norm = std::sqrt(dot(*r, *r));
            if (!(initial_norm > 0) || !std::isfinite(initial_norm)) {
                return false;
            }
            constexpr int depth = 20;
            amrex::Real h[depth + 1][depth]{};
            amrex::Real cosine[depth]{}, sine[depth]{}, g[depth + 1]{};
            g[0] = initial_norm;
            r->mult(1 / initial_norm);
            std::vector<std::unique_ptr<amrex::MultiFab>> basis;
            basis.push_back(std::move(r));
            auto perturbed = make();
            auto response = make();
            int columns = 0;
            for (int column = 0; column < depth; ++column) {
                auto const step = std::sqrt(std::numeric_limits<amrex::Real>::epsilon()) *
                                  amrex::max(1._rt, x.norm0(0)) / basis[column]->norm0(0);
                amrex::MultiFab::Copy(*perturbed, x, 0, 0, 1, 0);
                amrex::MultiFab::Saxpy(*perturbed, step, *basis[column], 0, 0, 1, 0);
                perturbed->FillBoundary(geometry.periodicity());
                if (!evaluate(*perturbed, *response)) {
                    return false;
                }
                auto vector = make();
                amrex::MultiFab::Copy(*vector, *response, 0, 0, 1, 0);
                amrex::MultiFab::Subtract(*vector, f, 0, 0, 1, 0);
                vector->mult(-1 / step);
                amrex::MultiFab::Add(*vector, *basis[column], 0, 0, 1, 0);
                for (int pass = 0; pass < 2; ++pass) {
                    for (int row = 0; row <= column; ++row) {
                        auto const projection = dot(*basis[row], *vector);
                        h[row][column] += projection;
                        amrex::MultiFab::Saxpy(*vector, -projection, *basis[row], 0, 0, 1, 0);
                    }
                }
                auto const next_norm = std::sqrt(dot(*vector, *vector));
                if (!std::isfinite(next_norm)) {
                    return false;
                }
                h[column + 1][column] = next_norm;
                for (int row = 0; row < column; ++row) {
                    auto const first = h[row][column];
                    h[row][column] = cosine[row] * first + sine[row] * h[row + 1][column];
                    h[row + 1][column] = -sine[row] * first + cosine[row] * h[row + 1][column];
                }
                auto const diagonal = std::hypot(h[column][column], h[column + 1][column]);
                if (!(diagonal > 0) || !std::isfinite(diagonal)) {
                    return false;
                }
                cosine[column] = h[column][column] / diagonal;
                sine[column] = h[column + 1][column] / diagonal;
                h[column][column] = diagonal;
                g[column + 1] = -sine[column] * g[column];
                g[column] *= cosine[column];
                columns = column + 1;
                if (std::abs(g[column + 1]) < 1.e-3_rt * initial_norm || next_norm == 0) {
                    break;
                }
                vector->mult(1 / next_norm);
                basis.push_back(std::move(vector));
            }
            for (int row = columns - 1; row >= 0; --row) {
                for (int column = row + 1; column < columns; ++column) {
                    g[row] -= h[row][column] * g[column];
                }
                g[row] /= h[row][row];
                if (!std::isfinite(g[row])) {
                    return false;
                }
            }
            amrex::MultiFab::Copy(target, x, 0, 0, 1, 0);
            for (int column = 0; column < columns; ++column) {
                amrex::MultiFab::Saxpy(target, g[column], *basis[column], 0, 0, 1, 0);
            }
            return target.is_finite();
        }

        CoupledMomentResult
        AdvanceSource (
            std::function<bool(ParticleImpulseTransaction&, amrex::MultiFab const&)> const& stage,
            amrex::MultiFab& radiation, amrex::MultiFab& nodal_temperature,
            amrex::MultiFab& realized_material_energy, amrex::Geometry const& geometry,
            amrex::Real time, amrex::Real dt, CoupledMomentOptions const& options,
            CoupledMomentCallbacks const& callbacks, CoupledMomentExchange* exchange)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                geometry.isAllPeriodic() && radiation.nComp() == 4 &&
                    radiation.ixType().cellCentered() && nodal_temperature.nComp() == 1 &&
                    nodal_temperature.ixType().nodeCentered() &&
                    nodal_temperature.nGrowVect().allGE(1) &&
                    amrex::convert(radiation.boxArray(), nodal_temperature.ixType()) ==
                        nodal_temperature.boxArray() &&
                    radiation.DistributionMap() == nodal_temperature.DistributionMap() &&
                    realized_material_energy.boxArray() == radiation.boxArray() &&
                    realized_material_energy.DistributionMap() == radiation.DistributionMap() &&
                    realized_material_energy.nComp() == 1 &&
                    realized_material_energy.nGrowVect().allGE(1) && callbacks.coefficients &&
                    callbacks.material_response && options.max_iterations > 0 &&
                    options.tolerance > 0 && options.tolerance < 1 && options.energy_tolerance > 0 &&
                    options.energy_tolerance < 1 && options.relaxation > 0 && options.relaxation <= 1 &&
                    options.maximum_beta > 0 && options.maximum_beta <= 0.01 && std::isfinite(time) &&
                    std::isfinite(dt) && dt > 0,
                "Invalid coupled gray moment source configuration.");
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            amrex::Abort("Coupled moment source is initially qualified only in Cartesian geometry.");
#endif
            auto const& boxes = radiation.boxArray();
            auto const& distribution = radiation.DistributionMap();
            auto const temperature_ghosts = nodal_temperature.nGrowVect();
            auto const source_ghosts = realized_material_energy.nGrowVect();
            amrex::MultiFab trial_temperature(nodal_temperature.boxArray(), distribution, 1,
                                              temperature_ghosts);
            amrex::MultiFab candidate_temperature(nodal_temperature.boxArray(), distribution, 1,
                                                  temperature_ghosts);
            amrex::MultiFab base_temperature(nodal_temperature.boxArray(), distribution, 1,
                                             temperature_ghosts);
            amrex::MultiFab accepted_temperature(nodal_temperature.boxArray(), distribution, 1,
                                                 temperature_ghosts);
            amrex::MultiFab native_equation(nodal_temperature.boxArray(), distribution, 3, 0);
            amrex::MultiFab previous_response(nodal_temperature.boxArray(),distribution,1,0);
            amrex::MultiFab target_temperature(nodal_temperature.boxArray(), distribution, 1,
                                               temperature_ghosts);
            amrex::MultiFab candidate_radiation(boxes, distribution, 4, radiation.nGrowVect());
            amrex::MultiFab absorption(boxes, distribution, 1, 0);
            amrex::MultiFab scattering(boxes, distribution, 1, 0);
            amrex::MultiFab equilibrium(boxes, distribution, 1, 0);
            amrex::MultiFab requested_heat(boxes, distribution, 1, source_ghosts);
            amrex::MultiFab actual_heat(boxes, distribution, 1, source_ghosts);
            amrex::MultiFab impulse(boxes, distribution, 3, 0);
            amrex::MultiFab beta(boxes, distribution, 3, 0);
            amrex::MultiFab next_beta(boxes, distribution, 3, 0);
            amrex::MultiFab base_beta(boxes, distribution, 3, 0);
            amrex::MultiFab target_beta(boxes, distribution, 3, 0);
            amrex::MultiFab verification_radiation(boxes, distribution, 4, 0);
            amrex::MultiFab verification_heat(boxes, distribution, 1, source_ghosts);
            amrex::MultiFab verification_impulse(boxes, distribution, 3, 0);
            amrex::MultiFab precision(boxes, distribution, 2, 0);
            amrex::MultiFab spatial_transfer;
            amrex::MultiFab spatial_flux_precision;
            amrex::MultiFab transport_increment(boxes, distribution, 8, 0);
            transport_increment.setVal(0);
            if (options.spatial_transport) {
                spatial_transfer.define(boxes, distribution, 4, 0);
                spatial_flux_precision.define(boxes, distribution, 8, 0);
            }
            amrex::MultiFab::Copy(trial_temperature, nodal_temperature, 0, 0, 1, temperature_ghosts);
            impulse.setVal(0);
            CoupledMomentResult result;
            auto fail = [&] (CoupledMomentFailure failure) {
                result.failure = failure;
                result.valid = false;
                result.actual_kinetic_work = 0;
                result.carry_energy_change = 0;
                result.native_realization_residual = 0;
                return result;
            };
            {
                ParticleImpulseTransaction initial_particle;
                if (!stage(initial_particle, impulse)) {
                    return fail(CoupledMomentFailure::Particle);
                }
                amrex::MultiFab::Copy(beta, initial_particle.WorkVelocity(), 0, 0, 3, 0);
            }
            beta.mult(1 / PhysConst::c);

            auto source = [&] (amrex::MultiFab const& temperature, amrex::MultiFab const& material_beta,
                               amrex::MultiFab& trial_radiation, amrex::MultiFab& heat,
                               amrex::MultiFab& momentum) {
                if (!temperature.is_finite() || !(temperature.min(0) >= 0)) {
                    return false;
                }
                callbacks.coefficients(temperature, absorption, scattering, equilibrium, time + dt);
                heat.setVal(0);
                if (options.spatial_transport) {
                    amrex::MultiFab::Copy(trial_radiation, radiation, 0, 0, 4, 0);
                    auto transport_options = options.transport;
                    transport_options.verbose = transport_options.verbose || options.verbose;
                    transport_options.tolerance =
                        amrex::min(transport_options.tolerance, 0.1_rt * options.tolerance);
                    transport_options.linear_tolerance = amrex::min(
                        transport_options.linear_tolerance, 0.01_rt * transport_options.tolerance);
                    auto const solved = TryImplicitMomentTransport(
                        trial_radiation, material_beta, absorption, scattering, equilibrium, spatial_transfer, geometry,
                        dt, transport_options, &heat);
                    if (!solved.valid) {
                        if (options.verbose) {
                            amrex::Print()
                                << "Material-coupled transport failure: nonlinear="
                                << solved.nonlinear_iterations << " linear=" << solved.linear_iterations
                                << " equation=" << solved.equation_residual
                                << " energy=" << solved.energy_residual
                                << " momentum=" << solved.momentum_residual << '\n';
                        }
                        return false;
                    }
                    if (!ComputeMomentTransportIncrement(trial_radiation, material_beta, absorption, scattering,
                                                         equilibrium, spatial_flux_precision, geometry,
                                                         dt)) {
                        return false;
                    }
                }
                amrex::ReduceOps<amrex::ReduceOpMax> ops;
                amrex::ReduceData<int> data(ops);
                using Tuple = typename decltype(data)::Type;
                auto const maximum_beta = options.maximum_beta;
                auto const inner_tolerance = amrex::min(1.e-12_rt, 0.01_rt * options.tolerance);
                auto const spatial = options.spatial_transport;
                for (amrex::MFIter iterator(trial_radiation); iterator.isValid(); ++iterator) {
                    auto const old = radiation.const_array(iterator);
                    auto const b = material_beta.const_array(iterator);
                    auto const a = absorption.const_array(iterator);
                    auto const s = scattering.const_array(iterator);
                    auto const bath = equilibrium.const_array(iterator);
                    auto const output = trial_radiation.array(iterator);
                    auto const h = heat.array(iterator);
                    auto const p = momentum.array(iterator);
                    auto const allowance = precision.array(iterator);
                    amrex::Array4<amrex::Real const> spatial_source;
                    amrex::Array4<amrex::Real const> flux_precision;
                    if (spatial) {
                        spatial_source = spatial_transfer.const_array(iterator);
                        flux_precision = spatial_flux_precision.const_array(iterator);
                    }
                    ops.eval(
                        iterator.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                            amrex::GpuArray<amrex::Real, 3> drift{b(i, j, k, 0), b(i, j, k, 1),
                                                                  b(i, j, k, 2)};
                            amrex::Real b2 = 0;
                            for (auto v : drift) {
                                b2 += v * v;
                            }
                            if (!(b2 <= maximum_beta * maximum_beta)) {
                                return {1};
                            }
                            FourVector before{};
                            for (int d = 0; d < 4; ++d) {
                                before[d] = old(i, j, k, d);
                            }
                            auto const rate_a = PhysConst::c * dt * a(i, j, k);
                            auto const rate_s = PhysConst::c * dt * s(i, j, k);
                            if (spatial) {
                                for (int d = 0; d < 3; ++d) {
                                    p(i, j, k, d) = spatial_source(i, j, k, d + 1) / PhysConst::c;
                                }
                            } else {
                                auto const update = TryImplicitGreyMomentSource(
                                    before, drift, rate_a, rate_s, bath(i, j, k), 60, inner_tolerance);
                                if (!update.valid) {
                                    return {1};
                                }
                                for (int d = 0; d < 4; ++d) {
                                    output(i, j, k, d) = update.radiation[d];
                                }
                                h(i, j, k) = update.material_energy_minus_work;
                                for (int d = 0; d < 3; ++d) {
                                    p(i, j, k, d) = update.material_transfer[d + 1] / PhysConst::c;
                                }
                            }
                            auto const scale = amrex::max(before[0], bath(i, j, k));
                            auto const roundoff = 64 * std::numeric_limits<amrex::Real>::epsilon();
                            amrex::Real heat_scale = scale;
                            amrex::Real momentum_scale = scale;
                            if (spatial) {
                                // The source responds to an algebraic transport
                                // RHS. Include its face-sum rounding uncertainty,
                                // attenuated by the source response (O(rate) when
                                // weak, bounded when stiff), in the fixed-point
                                // comparison. The actual equation and raw ledger
                                // gates below are still independently required.
                                heat_scale += flux_precision(i, j, k, 4);
                                for (int d = 0; d < 3; ++d) {
                                    heat_scale += std::abs(drift[d]) * flux_precision(i, j, k, d + 5);
                                    momentum_scale += flux_precision(i, j, k, d + 5);
                                }
                            }
                            allowance(i, j, k, 0) = roundoff * heat_scale * amrex::min(1._rt, rate_a);
                            allowance(i, j, k, 1) =
                                roundoff * momentum_scale * amrex::min(1._rt, rate_a + rate_s);
                            return {0};
                        });
                }
                int invalid = amrex::get<0>(data.value());
                amrex::ParallelDescriptor::ReduceIntMax(invalid);
                heat.FillBoundary(geometry.periodicity());
                return invalid == 0;
            };

            bool have_base = false;
            bool thermal_stiff = false;
            std::vector<amrex::Real> search_history;
            auto const temperature_scale = amrex::max(1._rt, nodal_temperature.norm0(0));
            amrex::MultiFab temperature_residual(nodal_temperature.boxArray(), distribution, 1, 0);
            amrex::MultiFab beta_residual(boxes, distribution, 3, 0);
            amrex::Real fraction = options.relaxation;
            amrex::Real base_merit = std::numeric_limits<amrex::Real>::max();
            auto interpolate = [&] (amrex::MultiFab& trial, amrex::MultiFab const& base,
                                    amrex::MultiFab const& target) {
                auto const step_fraction = fraction;
                for (amrex::MFIter iterator(trial); iterator.isValid(); ++iterator) {
                    auto const output = trial.array(iterator);
                    auto const previous = base.const_array(iterator);
                    auto const next = target.const_array(iterator);
                    amrex::ParallelFor(iterator.validbox(), trial.nComp(),
                                       [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                                           auto const value = previous(i, j, k, n);
                                           auto const candidate =
                                               value + step_fraction * (next(i, j, k, n) - value);
                                           output(i, j, k, n) =
                                               candidate == value && step_fraction >= 0.25_rt
                                                   ? next(i, j, k, n)
                                                   : candidate;
                                       });
                }
                trial.FillBoundary(geometry.periodicity());
            };
            auto backtrack = [&] () {
                if (!have_base || fraction <= 16 * std::numeric_limits<amrex::Real>::epsilon()) {
                    return false;
                }
                fraction *= 0.5_rt;
                interpolate(trial_temperature, base_temperature, target_temperature);
                interpolate(beta, base_beta, target_beta);
                return true;
            };
            for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
                result.iterations = iteration + 1;
                if (!source(trial_temperature, beta, candidate_radiation, requested_heat, impulse)) {
                    if (backtrack()) {
                        continue;
                    }
                    return fail(CoupledMomentFailure::Radiation);
                }
                ParticleImpulseTransaction particle_candidate;
                if (!stage(particle_candidate, impulse)) {
                    return fail(CoupledMomentFailure::Particle);
                }
                amrex::MultiFab::Copy(candidate_temperature, nodal_temperature, 0, 0, 1,
                                      temperature_ghosts);
                amrex::MultiFab::Copy(actual_heat, requested_heat, 0, 0, 1, source_ghosts);
                auto realization = callbacks.material_response(candidate_temperature, actual_heat);
                // Collective checks cannot be skipped on a rank-local callback failure.
                int invalid = std::isfinite(realization) ? 0 : 1;
                if (!candidate_temperature.is_finite()) {
                    invalid = 1;
                }
                if (!actual_heat.is_finite()) {
                    invalid = 1;
                }
                if (!(candidate_temperature.min(0) >= 0)) {
                    invalid = 1;
                }
                amrex::ParallelDescriptor::ReduceIntMax(invalid);
                if (invalid != 0) {
                    if (backtrack()) {
                        continue;
                    }
                    if (!have_base) {
                        // A hot old-state emission trial can remove more than the
                        // available native energy. Search for an admissible initial
                        // iterate; this changes only coefficient-evaluation scratch
                        // temperature, never the physical old material state.
                        trial_temperature.mult(0.5_rt);
                        trial_temperature.FillBoundary(geometry.periodicity());
                        continue;
                    }
                    return fail(CoupledMomentFailure::Material);
                }
                amrex::MultiFab::Copy(next_beta, particle_candidate.WorkVelocity(), 0, 0, 3, 0);
                next_beta.mult(1 / PhysConst::c);
                bool const prescribed_material =
                    options.spatial_transport && static_cast<bool>(callbacks.material_at_temperature);
                amrex::Real native_error = 0;
                if (prescribed_material) {
                    amrex::MultiFab::Copy(accepted_temperature, nodal_temperature, 0, 0, 1,
                                          temperature_ghosts);
                    amrex::MultiFab::Copy(actual_heat, requested_heat, 0, 0, 1, source_ghosts);
                    realization = callbacks.material_at_temperature(accepted_temperature, actual_heat,
                                                                    trial_temperature, native_equation);
                    if (!std::isfinite(realization) || !accepted_temperature.is_finite() ||
                        !actual_heat.is_finite() || !native_equation.is_finite() ||
                        accepted_temperature.min(0) < 0 || native_equation.min(1) < 0 ||
                        native_equation.min(2) < 0) {
                        if (backtrack()) {
                            continue;
                        }
                        return fail(CoupledMomentFailure::Material);
                    }
                    auto const tol = options.tolerance;
                    for (amrex::MFIter iterator(native_equation); iterator.isValid(); ++iterator) {
                        auto const values = native_equation.array(iterator);
                        amrex::ParallelFor(
                            iterator.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                values(i, j, k, 0) =
                                    RelativeSourceError(std::abs(values(i, j, k, 0)),
                                                        values(i, j, k, 1), values(i, j, k, 2) / tol);
                            });
                    }
                    native_error = native_equation.norm0(0);
                }
                auto const& physical_temperature =
                    prescribed_material ? accepted_temperature : candidate_temperature;
                if (!source(physical_temperature, next_beta, verification_radiation, verification_heat,
                            verification_impulse)) {
                    if (backtrack()) {
                        continue;
                    }
                    return fail(CoupledMomentFailure::Radiation);
                }
                if (options.spatial_transport &&
                    !ComputeMomentTransportIncrement(candidate_radiation, next_beta, absorption,
                                                     scattering, equilibrium, transport_increment,
                                                     geometry, dt)) {
                    if (backtrack()) {
                        continue;
                    }
                    return fail(CoupledMomentFailure::Radiation);
                }
                // The native response has already evaluated the accepted heat
                // source. Test that source against the equation at its actual
                // candidate temperature, not the change from the lagged iterate.
                // The latter can stall by one temperature ULP despite converged
                // physical source equations. Native realization and raw balances
                // remain independently measured below.
                amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax,
                                 amrex::ReduceOpMax, amrex::ReduceOpMax>
                    ops;
                amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real> data(
                    ops);
                using Tuple = typename decltype(data)::Type;
                auto const tolerance = options.tolerance;
                for (amrex::MFIter iterator(impulse); iterator.isValid(); ++iterator) {
                    auto const p = impulse.const_array(iterator);
                    auto const pc = verification_impulse.const_array(iterator);
                    auto const h = requested_heat.const_array(iterator);
                    auto const hc = verification_heat.const_array(iterator);
                    auto const b = beta.const_array(iterator);
                    auto const w = particle_candidate.RequestedWork().const_array(iterator);
                    auto const allowance = precision.const_array(iterator);
                    auto const old_radiation = radiation.const_array(iterator);
                    auto const candidate = candidate_radiation.const_array(iterator);
                    auto const actual_beta = next_beta.const_array(iterator);
                    auto const a = absorption.const_array(iterator);
                    auto const s = scattering.const_array(iterator);
                    auto const bath = equilibrium.const_array(iterator);
                    auto const transported = transport_increment.const_array(iterator);
                    ops.eval(
                        iterator.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                            auto heat_error = RelativeSourceError(
                                std::abs(hc(i, j, k) - h(i, j, k)),
                                amrex::max(std::abs(hc(i, j, k)), std::abs(h(i, j, k))),
                                allowance(i, j, k, 0) / tolerance);
                            amrex::Real projected_work = 0;
                            amrex::Real worst = 0;
                            amrex::Real work_scale = std::abs(w(i, j, k));
                            amrex::Real absolute_error =
                                amrex::max(0._rt, std::abs(hc(i, j, k) - h(i, j, k)) -
                                                      tolerance * amrex::max(std::abs(hc(i, j, k)),
                                                                             std::abs(h(i, j, k))) -
                                                      allowance(i, j, k, 0));
                            for (int d = 0; d < 3; ++d) {
                                absolute_error = amrex::max(
                                    absolute_error,
                                    PhysConst::c * (std::abs(pc(i, j, k, d) - p(i, j, k, d)) -
                                                    tolerance * amrex::max(std::abs(pc(i, j, k, d)),
                                                                           std::abs(p(i, j, k, d)))) -
                                        allowance(i, j, k, 1));
                                worst = amrex::max(
                                    worst, RelativeSourceError(
                                               PhysConst::c * std::abs(pc(i, j, k, d) - p(i, j, k, d)),
                                               PhysConst::c * amrex::max(std::abs(pc(i, j, k, d)),
                                                                         std::abs(p(i, j, k, d))),
                                               allowance(i, j, k, 1) / tolerance));
                                auto const work = b(i, j, k, d) * PhysConst::c * p(i, j, k, d);
                                projected_work += work;
                                work_scale += std::abs(work);
                            }
                            auto const work_error = RelativeSourceError(
                                std::abs(w(i, j, k) - projected_work), work_scale, 0);
                            absolute_error =
                                amrex::max(absolute_error, std::abs(w(i, j, k) - projected_work) -
                                                               tolerance * work_scale);
                            // A converged source fixed point is not itself an
                            // unscaled backward-Euler equation gate: stiffness can
                            // amplify the remaining iterate error. Evaluate the
                            // actual accepted radiation state at candidate T/beta.
                            FourVector moments{};
                            amrex::GpuArray<amrex::Real, 3> drift{};
                            for (int d = 0; d < 4; ++d) {
                                moments[d] = candidate(i, j, k, d);
                            }
                            for (int d = 0; d < 3; ++d) {
                                drift[d] = actual_beta(i, j, k, d);
                            }
                            auto const closure = EvaluateM1Closure(moments);
                            auto const rate_a = PhysConst::c * dt * a(i, j, k);
                            auto const rate_s = PhysConst::c * dt * s(i, j, k);
                            auto const force = EvaluateGreyFourForce(closure.tensor, drift, rate_a,
                                                                     rate_s, bath(i, j, k));
                            if (!closure.valid || !force.valid) {
                                auto const invalid_value = std::numeric_limits<amrex::Real>::max();
                                return {invalid_value, invalid_value, invalid_value, invalid_value,
                                        invalid_value};
                            }
                            // Independently enforce the local caloric source at
                            // the actual material state. A transport-divergence
                            // rounding allowance must not hide inconsistent heat.
                            auto const local_heat =
                                GreyEnergyMinusWork(closure.tensor, drift, rate_a, bath(i, j, k));
                            auto const caloric_noise = 64 *
                                                       std::numeric_limits<amrex::Real>::epsilon() *
                                                       rate_a * (std::abs(moments[0]) + bath(i, j, k));
                            heat_error = amrex::max(
                                heat_error, RelativeSourceError(
                                                std::abs(local_heat - h(i, j, k)),
                                                amrex::max(std::abs(local_heat), std::abs(h(i, j, k))),
                                                caloric_noise / tolerance));
                            auto const equation_noise =
                                64 * std::numeric_limits<amrex::Real>::epsilon() *
                                amrex::max(old_radiation(i, j, k, 0), bath(i, j, k)) *
                                (1 + rate_a + rate_s);
                            auto search_noise = equation_noise;
                            for (int d = 0; d < 4; ++d) {
                                auto const change =
                                    old_radiation(i, j, k, d) - moments[d] - transported(i, j, k, d);
                                auto const residual = std::abs(change - force.material_force[d]);
                                auto const equation_scale =
                                    amrex::max(std::abs(change), std::abs(force.material_force[d]));
                                auto const noise =
                                    equation_noise + 64 * std::numeric_limits<amrex::Real>::epsilon() *
                                                         transported(i, j, k, d + 4);
                                search_noise = amrex::max(search_noise, noise);
                                auto const error =
                                    RelativeSourceError(residual, equation_scale, noise / tolerance);
                                if (d == 0) {
                                    heat_error = amrex::max(heat_error, error);
                                } else {
                                    worst = amrex::max(worst, error);
                                }
                                absolute_error = amrex::max(
                                    absolute_error, residual - tolerance * equation_scale - noise);
                            }
                            // Line search needs a fixed scale: normalizing by the
                            // changing trial source can reward a larger residual
                            // when heating and cooling straddle equilibrium.
                            auto const fixed_scale =
                                old_radiation(i, j, k, 0) > 0 ? old_radiation(i, j, k, 0) : 1._rt;
                            return {heat_error, worst, work_error, absolute_error / fixed_scale,
                                    search_noise / fixed_scale};
                        });
                }
                auto const norms = data.value();
                amrex::Real errors[5] = {amrex::get<0>(norms), amrex::get<1>(norms),
                                         amrex::get<2>(norms), amrex::get<3>(norms),
                                         amrex::get<4>(norms)};
                amrex::ParallelDescriptor::ReduceRealMax(errors, 5);
                for (auto error : errors) {
                    if (!std::isfinite(error)) {
                        return fail(CoupledMomentFailure::Radiation);
                    }
                }
                result.material_residual = amrex::max(errors[0], native_error);
                result.source_residual = amrex::max(errors[1], 0._rt);
                result.work_residual = amrex::max(errors[2], 0._rt);
                auto const convergence = amrex::max(
                    result.material_residual, amrex::max(result.source_residual, result.work_residual));
                auto merit = errors[3];
                auto search_noise = errors[4];
                if (options.spatial_transport) {
                    // Use the natural fixed-point residual for searching, not the
                    // composition of the stiff source with that residual. This
                    // changes no final source, work or conservation gate.
                    amrex::MultiFab::Copy(temperature_residual, candidate_temperature, 0, 0, 1, 0);
                    amrex::MultiFab::Subtract(temperature_residual, trial_temperature, 0, 0, 1, 0);
                    amrex::MultiFab::Copy(beta_residual, next_beta, 0, 0, 3, 0);
                    amrex::MultiFab::Subtract(beta_residual, beta, 0, 0, 3, 0);
                    merit = std::sqrt(
                        amrex::MultiFab::Dot(temperature_residual, 0, 1, 0) /
                            (temperature_scale * temperature_scale *
                             static_cast<amrex::Real>(nodal_temperature.boxArray().numPts())) +
                        amrex::MultiFab::Dot(beta_residual, 0, 3, 0) /
                            (options.maximum_beta * options.maximum_beta *
                             static_cast<amrex::Real>(boxes.numPts())));
                    search_noise = 64 * std::numeric_limits<amrex::Real>::epsilon() *
                                   (1 + candidate_temperature.norm0(0) / temperature_scale);
                }
                if (options.verbose) {
                    amrex::Print() << "Coupled source iteration=" << iteration << " merit=" << merit
                                   << " fraction=" << fraction
                                   << " heat_error=" << result.material_residual
                                   << " native_error=" << native_error
                                   << " impulse_error=" << result.source_residual
                                   << " work_error=" << result.work_residual
                                   << " trial_T=" << trial_temperature.min(0)
                                   << " candidate_T=" << candidate_temperature.min(0) << '\n';
                }
                // Do not require a monotone decrease smaller than the rounding
                // uncertainty of the equation evaluation. This affects search
                // direction only; all final physical acceptance gates remain.
                auto search_ceiling = base_merit;
                if (options.spatial_transport) {
                    for (auto previous : search_history) {
                        search_ceiling = amrex::max(search_ceiling, previous);
                    }
                }
                if (convergence > tolerance && have_base && merit > search_ceiling + search_noise &&
                    backtrack()) {
                    continue;
                }
                result.actual_kinetic_work = particle_candidate.ActualWork().sum(0);
                result.carry_energy_change = particle_candidate.EnergyCarryChange().sum(0);
                result.native_realization_residual = realization;
                auto const raw = candidate_radiation.sum(0) - radiation.sum(0) + actual_heat.sum(0) +
                                 result.actual_kinetic_work + result.carry_energy_change;
                auto const scale = radiation.norm1(0) + candidate_radiation.norm1(0) +
                                   actual_heat.norm1(0) + particle_candidate.ActualWork().norm1(0);
                if (!std::isfinite(raw) || !std::isfinite(scale)) {
                    return fail(CoupledMomentFailure::Radiation);
                }
                result.raw_energy_residual = scale > 0 ? std::abs(raw) / scale : std::abs(raw);
                result.raw_momentum_residual = 0;
                for (int d = 0; d < 3; ++d) {
                    auto const imbalance =
                        candidate_radiation.sum(d + 1) - radiation.sum(d + 1) +
                        PhysConst::c * (particle_candidate.ActualImpulse().sum(d) +
                                        particle_candidate.MomentumCarryChange().sum(d));
                    auto const momentum_scale =
                        candidate_radiation.norm1(d + 1) + radiation.norm1(d + 1) +
                        PhysConst::c * (impulse.norm1(d) + particle_candidate.ActualImpulse().norm1(d) +
                                        particle_candidate.MomentumCarryChange().norm1(d));
                    if (!std::isfinite(imbalance) || !std::isfinite(momentum_scale)) {
                        return fail(CoupledMomentFailure::Radiation);
                    }
                    result.raw_momentum_residual =
                        amrex::max(result.raw_momentum_residual,
                                   momentum_scale > 0 ? std::abs(imbalance) / momentum_scale
                                                      : std::abs(imbalance));
                }
                if (result.material_residual <= tolerance && result.source_residual <= tolerance &&
                    result.work_residual <= tolerance &&
                    result.raw_energy_residual <= options.energy_tolerance &&
                    result.raw_momentum_residual <= options.energy_tolerance) {
                    CoupledMomentExchange accepted_exchange;
                    if (exchange != nullptr) {
                        accepted_exchange.kinetic_work.define(radiation.boxArray(),
                            radiation.DistributionMap(), 1, 0);
                        accepted_exchange.momentum.define(radiation.boxArray(),
                            radiation.DistributionMap(), 3, 0);
                        amrex::MultiFab::Copy(accepted_exchange.kinetic_work,
                            particle_candidate.ActualWork(), 0, 0, 1, 0);
                        amrex::MultiFab::Copy(accepted_exchange.momentum,
                            particle_candidate.ActualImpulse(), 0, 0, 3, 0);
                    }
                    candidate_radiation.FillBoundary(geometry.periodicity());
                    particle_candidate.Commit();
                    amrex::MultiFab::Copy(radiation, candidate_radiation, 0, 0, 4,
                                          radiation.nGrowVect());
                    amrex::MultiFab::Copy(nodal_temperature, physical_temperature, 0, 0, 1,
                                          temperature_ghosts);
                    amrex::MultiFab::Copy(realized_material_energy, actual_heat, 0, 0, 1,
                                          source_ghosts);
                    if (exchange != nullptr) { *exchange = std::move(accepted_exchange); }
                    result.valid = true;
                    return result;
                }
                if (options.spatial_transport && have_base) {
                    // Opacity alone does not determine caloric stiffness. Detect
                    // noncontractive feedback from accepted temperature secants,
                    // excluding changes at the input's arithmetic-noise scale.
                    amrex::MultiFab::Copy(temperature_residual,candidate_temperature,0,0,1,0);
                    amrex::MultiFab::Subtract(temperature_residual,previous_response,0,0,1,0);
                    amrex::MultiFab::Copy(target_temperature,trial_temperature,0,0,1,0);
                    amrex::MultiFab::Subtract(target_temperature,base_temperature,0,0,1,0);
                    auto const input_change = target_temperature.norm0(0);
                    if (input_change > 64*std::numeric_limits<amrex::Real>::epsilon()*temperature_scale &&
                        temperature_residual.norm0(0) > input_change) { thermal_stiff = true; }
                }
                amrex::MultiFab::Copy(previous_response,candidate_temperature,0,0,1,0);
                amrex::MultiFab::Copy(base_temperature, trial_temperature, 0, 0, 1, temperature_ghosts);
                // Once the thermal equation is resolved, do not throttle the
                // remaining velocity iteration with its stiff thermal step size.
                // Re-evaluation at every trial still reopens the thermal block if
                // the changed velocity makes its source residual inadmissible.
                // Momentum can be more temperature-sensitive than net heat. If
                // work/velocity is already resolved but momentum is not, continue
                // the temperature iteration instead of freezing it at its own gate.
                bool const thermal_resolved =
                    result.material_residual <= tolerance &&
                    (result.source_residual <= tolerance || result.work_residual > tolerance);
                amrex::MultiFab::Copy(target_temperature,
                                      thermal_resolved ? trial_temperature : candidate_temperature, 0,
                                      0, 1, temperature_ghosts);
                if (options.spatial_transport && !thermal_resolved &&
                    (thermal_stiff || PhysConst::c * dt * absorption.norm0(0) > 1)) {
                    auto evaluate_temperature = [&] (amrex::MultiFab const& t, amrex::MultiFab& out) {
                        if (!source(t, beta, verification_radiation, verification_heat,
                                    verification_impulse)) {
                            return false;
                        }
                        amrex::MultiFab::Copy(out, nodal_temperature, 0, 0, 1, temperature_ghosts);
                        auto const residual = callbacks.material_response(out, verification_heat);
                        return std::isfinite(residual) && out.is_finite() && out.min(0) >= 0;
                    };
                    if (!ThermalNewtonProposal(trial_temperature, candidate_temperature,
                                               target_temperature, geometry, evaluate_temperature)) {
                        amrex::MultiFab::Copy(target_temperature, candidate_temperature, 0, 0, 1,
                                              temperature_ghosts);
                    }
                }
                amrex::MultiFab::Copy(base_beta, beta, 0, 0, 3, 0);
                amrex::MultiFab::Copy(target_beta, next_beta, 0, 0, 3, 0);
                have_base = true;
                base_merit = merit;
                if (options.spatial_transport) {
                    if (search_history.size() == 6) {
                        search_history.erase(search_history.begin());
                    }
                    search_history.push_back(merit);
                }
                fraction = (thermal_resolved || options.spatial_transport)
                               ? options.relaxation
                               : amrex::min(options.relaxation, 1.5_rt * fraction);
                interpolate(trial_temperature, base_temperature, target_temperature);
                interpolate(beta, base_beta, target_beta);
            }
            return fail(CoupledMomentFailure::IterationBudget);
        }
    } // namespace

    CoupledMomentResult
    TryAdvanceCoupledMomentSource (MultiParticleContainer& particles,
                                   std::vector<std::string> const& momentum_species,
                                   std::string const& carry_path, amrex::MultiFab& radiation,
                                   amrex::MultiFab& nodal_temperature,
                                   amrex::MultiFab& realized_material_energy,
                                   amrex::Geometry const& geometry, amrex::Real time,
                                   amrex::Real dt, CoupledMomentOptions const& options,
                                   CoupledMomentCallbacks const& callbacks,
                                   CoupledMomentExchange* exchange)
    {
        auto stage = [&] (ParticleImpulseTransaction& trial, amrex::MultiFab const& impulse) {
            return trial.Stage(particles, momentum_species, carry_path, impulse, true,
                               options.particle_assignment);
        };
        return AdvanceSource(stage, radiation, nodal_temperature, realized_material_energy,
                             geometry, time, dt, options, callbacks, exchange);
    }

    CoupledMomentResult
    TryAdvanceCoupledMomentSource (ParticleImpulseMaterial& material, std::string const& carry_path,
                                   amrex::MultiFab& radiation, amrex::MultiFab& nodal_temperature,
                                   amrex::MultiFab& realized_material_energy,
                                   amrex::Geometry const& geometry, amrex::Real time,
                                   amrex::Real dt, CoupledMomentOptions const& options,
                                   CoupledMomentCallbacks const& callbacks,
                                   CoupledMomentExchange* exchange)
    {
        auto stage = [&] (ParticleImpulseTransaction& trial, amrex::MultiFab const& impulse) {
            return trial.Stage(material, carry_path, impulse, true, options.particle_assignment);
        };
        return AdvanceSource(stage, radiation, nodal_temperature, realized_material_energy,
                             geometry, time, dt, options, callbacks, exchange);
    }

    CoupledMomentIntervalResult
    TryAdvanceCoupledMomentInterval (MultiParticleContainer& particles,
                                     std::vector<std::string> const& momentum_species,
                                     std::string const& carry_path, amrex::MultiFab& radiation,
                                     amrex::MultiFab& nodal_temperature,
                                     amrex::MultiFab& realized_material_energy,
                                     amrex::Geometry const& geometry, amrex::Real time,
                                     amrex::Real dt, CoupledMomentIntervalOptions const& options,
                                     CoupledMomentCallbacks const& callbacks,
                                     CoupledMomentExchange* exchange)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            options.initial_substeps > 0 && options.max_refinements >= 0 &&
                options.max_refinements <= 20 && std::isfinite(time) && std::isfinite(dt) &&
                dt > 0 && std::isfinite(time + dt) && time + dt > time,
            "Invalid coupled radiation source interval or refinement budget.");
        CoupledMomentIntervalResult result;
        amrex::MultiFab trial_radiation(radiation.boxArray(), radiation.DistributionMap(),
                                        radiation.nComp(), radiation.nGrowVect());
        amrex::MultiFab trial_temperature(nodal_temperature.boxArray(),
                                          nodal_temperature.DistributionMap(),
                                          nodal_temperature.nComp(), nodal_temperature.nGrowVect());
        amrex::MultiFab heat(realized_material_energy.boxArray(),
                             realized_material_energy.DistributionMap(), 1,
                             realized_material_energy.nGrowVect());
        amrex::MultiFab total_heat(heat.boxArray(), heat.DistributionMap(), 1, heat.nGrowVect());
        int substeps = options.initial_substeps;
        for (int attempt = 0; attempt <= options.max_refinements; ++attempt) {
            ++result.attempts;
            ParticleImpulseMaterial material(particles, momentum_species);
            amrex::MultiFab::Copy(trial_radiation, radiation, 0, 0, radiation.nComp(),
                                  radiation.nGrowVect());
            amrex::MultiFab::Copy(trial_temperature, nodal_temperature, 0, 0,
                                  nodal_temperature.nComp(), nodal_temperature.nGrowVect());
            total_heat.setVal(0);
            CoupledMomentExchange total_exchange;
            if (exchange != nullptr) {
                total_exchange.kinetic_work.define(radiation.boxArray(),
                    radiation.DistributionMap(), 1, 0);
                total_exchange.momentum.define(radiation.boxArray(),
                    radiation.DistributionMap(), 3, 0);
                total_exchange.kinetic_work.setVal(0);
                total_exchange.momentum.setVal(0);
            }
            amrex::Real kinetic_work = 0;
            amrex::Real carry_change = 0;
            bool accepted = true;
            for (int step = 0; step < substeps; ++step) {
                auto const begin = time + dt * (static_cast<amrex::Real>(step) / substeps);
                auto const end = time + dt * (static_cast<amrex::Real>(step + 1) / substeps);
                if (!(end > begin)) {
                    accepted = false;
                    result.last_source = {};
                    result.last_source.failure = CoupledMomentFailure::IterationBudget;
                    break;
                }
                CoupledMomentExchange step_exchange;
                result.last_source = TryAdvanceCoupledMomentSource(
                    material, carry_path, trial_radiation, trial_temperature, heat, geometry, begin,
                    end - begin, options.source, callbacks,
                    exchange != nullptr ? &step_exchange : nullptr);
                if (!result.last_source.valid) {
                    accepted = false;
                    break;
                }
                ++result.completed_trial_substeps;
                kinetic_work += result.last_source.actual_kinetic_work;
                carry_change += result.last_source.carry_energy_change;
                amrex::MultiFab::Add(total_heat, heat, 0, 0, 1, heat.nGrowVect());
                if (exchange != nullptr) {
                    amrex::MultiFab::Add(total_exchange.kinetic_work,
                        step_exchange.kinetic_work, 0, 0, 1, 0);
                    amrex::MultiFab::Add(total_exchange.momentum,
                        step_exchange.momentum, 0, 0, 3, 0);
                }
            }
            if (accepted) {
                auto const raw = trial_radiation.sum(0) - radiation.sum(0) + total_heat.sum(0) +
                                 kinetic_work + carry_change;
                auto const scale = trial_radiation.norm1(0) + radiation.norm1(0) +
                                   total_heat.norm1(0) + std::abs(kinetic_work) +
                                   std::abs(carry_change);
                bool const finite =
                    total_heat.is_finite() && std::isfinite(raw) && std::isfinite(scale) &&
                    (exchange == nullptr || (total_exchange.kinetic_work.is_finite() &&
                                             total_exchange.momentum.is_finite()));
                result.raw_energy_residual =
                    finite ? (scale > 0 ? std::abs(raw) / scale : std::abs(raw))
                           : std::numeric_limits<amrex::Real>::infinity();
                if (!(result.raw_energy_residual <= options.source.energy_tolerance)) {
                    accepted = false;
                    result.last_source = {};
                    result.last_source.failure = CoupledMomentFailure::Material;
                }
            }
            if (accepted) {
                if (!material.Commit()) {
                    result.last_source = {};
                    result.last_source.failure = CoupledMomentFailure::Particle;
                    return result;
                }
                amrex::MultiFab::Copy(radiation, trial_radiation, 0, 0, radiation.nComp(),
                                      radiation.nGrowVect());
                amrex::MultiFab::Copy(nodal_temperature, trial_temperature, 0, 0,
                                      nodal_temperature.nComp(), nodal_temperature.nGrowVect());
                amrex::MultiFab::Copy(realized_material_energy, total_heat, 0, 0, 1,
                                      realized_material_energy.nGrowVect());
                if (exchange != nullptr) { *exchange = std::move(total_exchange); }
                result.actual_kinetic_work = kinetic_work;
                result.carry_energy_change = carry_change;
                result.substeps = substeps;
                result.valid = true;
                return result;
            }
            if (substeps > std::numeric_limits<int>::max() / 2) {
                break;
            }
            substeps *= 2;
        }
        return result;
    }
} // namespace warpx::radiation
