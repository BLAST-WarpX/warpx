/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "Radiation/CoupledImplicitDiffusion.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>

using namespace amrex::literals;

namespace
{
    std::unique_ptr<amrex::MultiFab>
    copy (amrex::MultiFab const& field)
    {
        auto result = std::make_unique<amrex::MultiFab>(field.boxArray(), field.DistributionMap(),
                                                        field.nComp(), field.nGrowVect());
        amrex::MultiFab::Copy(*result, field, 0, 0, field.nComp(), field.nGrowVect());
        return result;
    }

    void
    unchanged (amrex::MultiFab const& field, amrex::MultiFab const& snapshot)
    {
        auto difference = copy(field);
        amrex::MultiFab::Subtract(*difference, snapshot, 0, 0, field.nComp(), field.nGrowVect());
        for (int component = 0; component < field.nComp(); ++component) {
            AMREX_ALWAYS_ASSERT(difference->norm0(component, field.nGrow()) == 0);
        }
    }
} // namespace

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& simulation = WarpX::GetInstance();
        simulation.InitData();
        simulation.HybridPICPrepareElectronStateForDiagnostics();
        auto& model = *simulation.get_pointer_HybridPICModel();
        using warpx::fields::FieldType;
        auto& temperature = *simulation.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto& density = *simulation.m_fields.get(FieldType::rho_fp, 0);
        auto& pressure = *simulation.m_fields.get(FieldType::hybrid_electron_pressure_fp, 0);
        auto const& geometry = simulation.Geom(0);
        auto const eos = model.electronThermodynamicsExecutor();
        constexpr int cells = 8;
        AMREX_ALWAYS_ASSERT(geometry.Domain().length(0) == cells && geometry.isPeriodic(0));
        amrex::Real const dx = geometry.CellSize(0);
        amrex::Real const old_temperature = 1.0e4_rt;
        amrex::Real const rho0 = 1.0e20_rt * PhysConst::q_e;
        auto rho_at = [=] AMREX_GPU_HOST_DEVICE(int i) {
            return rho0 * (1.0_rt + 0.25_rt * std::cos(2.0_rt * MathConst::pi * i / cells));
        };
        for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
            auto const rho = density.array(mfi);
            auto const te = temperature.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho(i, j, k) = rho_at(i);
                te(i, j, k) = old_temperature;
            });
        }
        density.FillBoundary(geometry.periodicity());
        temperature.FillBoundary(geometry.periodicity());
        auto const old_te = copy(temperature);
        auto const old_rho = copy(density);
        auto const old_pe = copy(pressure);
        amrex::MultiFab source(amrex::convert(temperature.boxArray(), amrex::IntVect(0)),
                               temperature.DistributionMap(), 1, 1);
        amrex::Real const scale =
            0.01_rt * dx *
            eos.stateFromChargeDensityTemperature(rho0, old_temperature).internal_energy_density;
        auto request = [=] AMREX_GPU_HOST_DEVICE(int i) {
            int const cell = (i % cells + cells) % cells;
            return scale * (cell == 0 ? 1.0_rt : (cell == 3 ? -0.1_rt : 0.0_rt));
        };
        for (amrex::MFIter mfi(source); mfi.isValid(); ++mfi) {
            auto const q = source.array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                q(i, j, k) = request(i);
            });
        }
        auto const original_source = copy(source);
        auto candidate = copy(temperature);
        auto const residual = model.EvaluateElectronEnergySource(0, *candidate, source, 1.0_rt);
        amrex::Real const tolerance = 2048.0_rt * std::numeric_limits<amrex::Real>::epsilon();
        AMREX_ALWAYS_ASSERT(std::isfinite(residual) && std::abs(residual) < tolerance * scale);
        auto delta_u = [=] AMREX_GPU_HOST_DEVICE(int i) {
            auto cv = [=] AMREX_GPU_HOST_DEVICE(int n) {
                return eos.stateFromChargeDensityTemperature(rho_at(n), old_temperature)
                    .heat_capacity_density;
            };
            return (request(i - 1) * cv(i) / (cv(i - 1) + cv(i)) +
                    request(i) * cv(i) / (cv(i) + cv(i + 1))) /
                   dx;
        };
        auto error = copy(*candidate);
        for (amrex::MFIter mfi(*candidate); mfi.isValid(); ++mfi) {
            auto const te = candidate->const_array(mfi);
            auto const err = error->array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                auto const old = eos.stateFromChargeDensityTemperature(rho_at(i), old_temperature);
                auto const expected = eos.temperatureFromChargeDensityThermalEnergyDensity(
                    rho_at(i), old.internal_energy_density + delta_u(i));
                err(i, j, k) = std::abs(te(i, j, k) - expected) / old_temperature;
            });
        }
        AMREX_ALWAYS_ASSERT(error->norm0(0) < tolerance);
        auto source_error = copy(source);
        for (amrex::MFIter mfi(source); mfi.isValid(); ++mfi) {
            auto const realized = source.const_array(mfi);
            auto const err = source_error->array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                err(i, j, k) =
                    (realized(i, j, k) - 0.5_rt * dx * (delta_u(i) + delta_u(i + 1))) / scale;
            });
        }
        AMREX_ALWAYS_ASSERT(source_error->norm0(0) < tolerance);
        // The zero-request neighbor receives realized energy through shared nodes.
        AMREX_ALWAYS_ASSERT(source.sum(0, false) > 0);
        unchanged(temperature, *old_te);
        unchanged(density, *old_rho);
        unchanged(pressure, *old_pe);
        auto failed = copy(temperature);
        auto failed_source = copy(*original_source);
        failed_source->setVal(-1.0e30_rt);
        AMREX_ALWAYS_ASSERT(
            !std::isfinite(model.EvaluateElectronEnergySource(0, *failed, *failed_source, 1.0_rt)));
        unchanged(temperature, *old_te);
        unchanged(density, *old_rho);
        unchanged(pressure, *old_pe);
        auto retry = copy(temperature);
        auto retry_source = copy(*original_source);
        AMREX_ALWAYS_ASSERT(model.EvaluateElectronEnergySource(0, *retry, *retry_source, 1.0_rt) ==
                            residual);
        unchanged(*retry, *candidate);
        unchanged(*retry_source, source);
        // A nonlinear solver must be able to validate its own trial T without
        // replacing it with a second, stiffness-amplified inverse-EOS image.
        amrex::MultiFab nodal_residual(temperature.boxArray(), temperature.DistributionMap(), 3, 0);
        auto prescribed_state = copy(temperature);
        auto prescribed_source = copy(*original_source);
        auto const prescribed_residual =
            model.EvaluateElectronEnergySource(0, *prescribed_state, *prescribed_source, 1._rt,
                                               nullptr, candidate.get(), &nodal_residual);
        AMREX_ALWAYS_ASSERT(prescribed_residual == residual);
        unchanged(*prescribed_state, *candidate);
        unchanged(*prescribed_source, source);
        AMREX_ALWAYS_ASSERT(nodal_residual.norm0(0) < tolerance * scale / dx);
        AMREX_ALWAYS_ASSERT(nodal_residual.min(1) >= 0 && nodal_residual.min(2) > 0);
        // Finite but inconsistent temperatures produce a measurable nodal
        // residual. A caller must reject them; finite is not convergence.
        auto inconsistent = copy(*candidate);
        inconsistent->mult(1.01_rt);
        inconsistent->FillBoundary(geometry.periodicity());
        prescribed_state = copy(temperature);
        prescribed_source = copy(*original_source);
        AMREX_ALWAYS_ASSERT(std::isfinite(
            model.EvaluateElectronEnergySource(0, *prescribed_state, *prescribed_source, 1._rt,
                                               nullptr, inconsistent.get(), &nodal_residual)));
        AMREX_ALWAYS_ASSERT(nodal_residual.norm0(0) > 0.1_rt * scale / dx);
        unchanged(*prescribed_state, *inconsistent);
        inconsistent->setVal(-1);
        prescribed_state = copy(temperature);
        prescribed_source = copy(*original_source);
        AMREX_ALWAYS_ASSERT(!std::isfinite(
            model.EvaluateElectronEnergySource(0, *prescribed_state, *prescribed_source, 1._rt,
                                               nullptr, inconsistent.get(), &nodal_residual)));
        unchanged(temperature, *old_te);
        unchanged(density, *old_rho);
        unchanged(pressure, *old_pe);
        amrex::Print() << "Native shared-node material response, rejection and retry passed\n";
        amrex::MultiFab radiation(source.boxArray(), source.DistributionMap(), 1, 1);
        radiation.setVal(0);
        for (amrex::MFIter mfi(radiation); mfi.isValid(); ++mfi) {
            auto const energy = radiation.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                energy(i, j, k) =
                    dx * 20.0_rt *
                    (1.0_rt + 0.1_rt * std::cos(2.0_rt * MathConst::pi * (i + 0.5_rt) / cells));
            });
        }
        radiation.FillBoundary(geometry.periodicity());
        auto const old_radiation = copy(radiation);
        source.setVal(0);
        auto coupled_temperature = copy(temperature);
        warpx::radiation::CoupledImplicitOptions controls;
        controls.diffusion = {
            1.e-10_rt, 1.e-13_rt, 1.0_rt, 100, 100, 0, {}, {}, {}, nullptr, {nullptr, nullptr, 1},
            true};
        controls.material_tolerance = 1.e-9_rt;
        controls.energy_tolerance = 1.e-10_rt;
        warpx::radiation::CoupledImplicitCallbacks callbacks;
        callbacks.material_response = [&] (amrex::MultiFab& te, amrex::MultiFab& q) {
            return model.EvaluateElectronEnergySource(0, te, q, 1.0_rt);
        };
        callbacks.coefficients = [&] (amrex::MultiFab const& te, amrex::MultiFab& opacity,
                                      amrex::MultiFab& absorption, amrex::MultiFab& emission,
                                      amrex::Real) {
            opacity.setVal(100.0_rt);
            int const bands = absorption.nComp();
            for (amrex::MFIter mfi(absorption); mfi.isValid(); ++mfi) {
                auto const t = te.const_array(mfi);
                auto const rate = absorption.array(mfi);
                auto const emit = emission.array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    auto const cell_t = 0.5_rt * (t(i, j, k) + t(i + 1, j, k));
                    auto const t2 = cell_t * cell_t;
                    for (int g = 0; g < bands; ++g) {
                        rate(i, j, k, g) = 1.e8_rt * t2 / (old_temperature * old_temperature);
                        emit(i, j, k, g) =
                            rate(i, j, k, g) * 7.565733250280007e-16_rt * t2 * t2 / bands;
                    }
                });
            }
        };
        controls.max_iterations = 1;
        auto const denied = warpx::radiation::TryAdvanceCoupledImplicitDiffusion(
            radiation, *coupled_temperature, source, geometry, 0.0_rt, 1.e-9_rt, controls,
            callbacks);
        AMREX_ALWAYS_ASSERT(denied.failure ==
                            warpx::radiation::CoupledImplicitFailure::IterationBudget);
        unchanged(radiation, *old_radiation);
        unchanged(*coupled_temperature, *old_te);
        AMREX_ALWAYS_ASSERT(source.norm0(0, source.nGrow()) == 0);
        controls.max_iterations = 100;
        auto const coupled = warpx::radiation::TryAdvanceCoupledImplicitDiffusion(
            radiation, *coupled_temperature, source, geometry, 0.0_rt, 1.e-9_rt, controls,
            callbacks);
        amrex::Print() << "Coupled iterations=" << coupled.iterations
                       << " failure=" << static_cast<int>(coupled.failure)
                       << " material residual=" << coupled.material_relative_residual
                       << " raw energy residual=" << coupled.raw_energy_relative_residual << '\n';
        AMREX_ALWAYS_ASSERT(coupled.failure == warpx::radiation::CoupledImplicitFailure::None);
        AMREX_ALWAYS_ASSERT(coupled.diffusion.maximum_relative_residual <=
                            controls.diffusion.tolerance);
        AMREX_ALWAYS_ASSERT(std::abs(radiation.sum(0, false) + source.sum(0, false) -
                                     old_radiation->sum(0, false)) <
                            1.e-10_rt * old_radiation->sum(0, false));
        // A successful retry must reproduce an independently accepted update.
        auto replay_radiation = copy(*old_radiation);
        auto replay_temperature = copy(*old_te);
        auto replay_source = copy(source);
        replay_source->setVal(0);
        auto const replay = warpx::radiation::TryAdvanceCoupledImplicitDiffusion(
            *replay_radiation, *replay_temperature, *replay_source, geometry, 0.0_rt, 1.e-9_rt,
            controls, callbacks);
        AMREX_ALWAYS_ASSERT(replay.failure == warpx::radiation::CoupledImplicitFailure::None);
        unchanged(radiation, *replay_radiation);
        unchanged(*coupled_temperature, *replay_temperature);
        unchanged(source, *replay_source);
        // Material and spatial rejection must preserve all caller-owned fields.
        auto failing_callbacks = callbacks;
        failing_callbacks.material_response = [] (amrex::MultiFab& te, amrex::MultiFab& q) {
            if (amrex::ParallelDescriptor::MyProc() == 0) {
                te.setVal(-1);
                q.setVal(-2);
                return std::numeric_limits<amrex::Real>::quiet_NaN();
            }
            return 0.0_rt;
        };
        auto const material_failure = warpx::radiation::TryAdvanceCoupledImplicitDiffusion(
            radiation, *coupled_temperature, source, geometry, 0.0_rt, 1.e-9_rt, controls,
            failing_callbacks);
        AMREX_ALWAYS_ASSERT(material_failure.failure ==
                            warpx::radiation::CoupledImplicitFailure::Material);
        unchanged(radiation, *replay_radiation);
        unchanged(*coupled_temperature, *replay_temperature);
        unchanged(source, *replay_source);
        controls.diffusion.max_iterations = 1;
        auto const spatial_failure = warpx::radiation::TryAdvanceCoupledImplicitDiffusion(
            radiation, *coupled_temperature, source, geometry, 0.0_rt, 1.e-9_rt, controls,
            callbacks);
        AMREX_ALWAYS_ASSERT(spatial_failure.failure ==
                            warpx::radiation::CoupledImplicitFailure::Spatial);
        unchanged(radiation, *replay_radiation);
        unchanged(*coupled_temperature, *replay_temperature);
        unchanged(source, *replay_source);
        controls.diffusion.max_iterations = 100;
        unchanged(temperature, *old_te);
        unchanged(density, *old_rho);
        unchanged(pressure, *old_pe);

        // Inject a first-attempt rank-local material failure, then require the
        // transactional retry to exactly reproduce two independently accepted
        // half steps, including their summed native caloric source.
        auto split_radiation = copy(*old_radiation);
        auto split_temperature = copy(*old_te);
        auto split_source = copy(source);
        auto first_source = copy(source);
        auto const half1 = warpx::radiation::TryAdvanceCoupledImplicitDiffusion(
            *split_radiation, *split_temperature, *split_source, geometry, 0, 5.e-10_rt, controls,
            callbacks);
        AMREX_ALWAYS_ASSERT(half1.failure == warpx::radiation::CoupledImplicitFailure::None);
        amrex::MultiFab::Copy(*first_source, *split_source, 0, 0, 1, source.nGrowVect());
        auto const half2 = warpx::radiation::TryAdvanceCoupledImplicitDiffusion(
            *split_radiation, *split_temperature, *split_source, geometry, 5.e-10_rt, 5.e-10_rt,
            controls, callbacks);
        AMREX_ALWAYS_ASSERT(half2.failure == warpx::radiation::CoupledImplicitFailure::None);
        amrex::MultiFab::Add(*split_source, *first_source, 0, 0, 1, source.nGrowVect());
        auto retried_radiation = copy(*old_radiation);
        auto retried_temperature = copy(*old_te);
        auto retried_source = copy(source);
        int material_calls = 0;
        auto fault_once = callbacks;
        fault_once.material_response = [&] (amrex::MultiFab& te, amrex::MultiFab& q) {
            ++material_calls;
            auto const value = callbacks.material_response(te, q);
            if (material_calls == 1 && amrex::ParallelDescriptor::MyProc() == 0) {
                return std::numeric_limits<amrex::Real>::quiet_NaN();
            }
            return value;
        };
        controls.max_subdivisions = 1;
        auto const recovered = warpx::radiation::TryAdvanceCoupledImplicitSubcycled(
            *retried_radiation, *retried_temperature, *retried_source, geometry, 0, 1.e-9_rt,
            controls, fault_once);
        AMREX_ALWAYS_ASSERT(recovered.failure == warpx::radiation::CoupledImplicitFailure::None);
        AMREX_ALWAYS_ASSERT(recovered.rejected_attempts == 1 && recovered.accepted_substeps == 2);
        unchanged(*retried_radiation, *split_radiation);
        unchanged(*retried_temperature, *split_temperature);
        unchanged(*retried_source, *split_source);
        AMREX_ALWAYS_ASSERT(recovered.diffusion.group_material_energy[0] ==
                            half1.diffusion.group_material_energy[0] +
                                half2.diffusion.group_material_energy[0]);
        // A second-half failure must roll back even an already accepted first
        // half, and must not publish its energy/boundary/group ledger.
        amrex::Real endpoint = 0;
        auto late_fault = callbacks;
        late_fault.coefficients = [&] (amrex::MultiFab const& te, amrex::MultiFab& opacity,
                                       amrex::MultiFab& absorption, amrex::MultiFab& emission,
                                       amrex::Real time) {
            endpoint = time;
            callbacks.coefficients(te, opacity, absorption, emission, time);
        };
        late_fault.material_response = [&] (amrex::MultiFab& te, amrex::MultiFab& q) {
            auto const value = callbacks.material_response(te, q);
            if (endpoint > 5.e-10_rt && amrex::ParallelDescriptor::MyProc() == 0) {
                return std::numeric_limits<amrex::Real>::quiet_NaN();
            }
            return value;
        };
        auto const late = warpx::radiation::TryAdvanceCoupledImplicitSubcycled(
            *retried_radiation, *retried_temperature, *retried_source, geometry, 0, 1.e-9_rt,
            controls, late_fault);
        AMREX_ALWAYS_ASSERT(late.failure == warpx::radiation::CoupledImplicitFailure::Material);
        AMREX_ALWAYS_ASSERT(late.rejected_attempts == 2 && late.accepted_substeps == 0);
        AMREX_ALWAYS_ASSERT(late.diffusion.group_material_energy.empty());
        AMREX_ALWAYS_ASSERT(late.diffusion.escaped_energy == 0 &&
                            late.diffusion.injected_energy == 0);
        unchanged(*retried_radiation, *split_radiation);
        unchanged(*retried_temperature, *split_temperature);
        unchanged(*retried_source, *split_source);
        controls.max_subdivisions = 0;
        amrex::Print() << "Substep recovery and late-failure full rollback passed\n";

        // Independent uniform constant-CV LTE root with temperature-dependent
        // opacity, not the solver's own residual used as an oracle.
        density.setVal(rho0);
        coupled_temperature->setVal(old_temperature);
        radiation.setVal(dx * 20.0_rt);
        source.setVal(0);
        auto const uniform = warpx::radiation::TryAdvanceCoupledImplicitDiffusion(
            radiation, *coupled_temperature, source, geometry, 0.0_rt, 1.e-9_rt, controls,
            callbacks);
        AMREX_ALWAYS_ASSERT(uniform.failure == warpx::radiation::CoupledImplicitFailure::None);
        auto const cv =
            eos.stateFromChargeDensityTemperature(rho0, old_temperature).heat_capacity_density;
        amrex::Real lower = old_temperature;
        amrex::Real upper = old_temperature + 20.0_rt / cv;
        for (int iteration = 0; iteration < 100; ++iteration) {
            auto const t = 0.5_rt * (lower + upper);
            auto const e = 20.0_rt - cv * (t - old_temperature);
            auto const rate_dt = 0.1_rt * t * t / (old_temperature * old_temperature);
            auto const equation =
                e - 20.0_rt + rate_dt * (e - 7.565733250280007e-16_rt * t * t * t * t);
            if (equation > 0) {
                lower = t;
            } else {
                upper = t;
            }
        }
        auto const reference_temperature = 0.5_rt * (lower + upper);
        auto const reference_energy = 20.0_rt - cv * (reference_temperature - old_temperature);
        amrex::Print() << "Uniform LTE temperature=" << coupled_temperature->max(0)
                       << " independent root=" << reference_temperature << '\n';
        AMREX_ALWAYS_ASSERT(std::abs(coupled_temperature->max(0) - reference_temperature) <
                            1.e-8_rt * (reference_temperature - old_temperature));
        AMREX_ALWAYS_ASSERT(std::abs(coupled_temperature->min(0) - reference_temperature) <
                            1.e-8_rt * (reference_temperature - old_temperature));
        AMREX_ALWAYS_ASSERT(std::abs(radiation.sum(0, false) - reference_energy) < 1.e-8_rt);

        // Manufacture a two-band spatial/LTE solution with opposite group
        // exchanges and exactly zero net material heating. Face fluxes here
        // are computed independently from the analytic target values.
        constexpr amrex::Real manufactured_dt = 1.e-9_rt;
        amrex::Real const background = 0.5_rt * 7.565733250280007e-16_rt * old_temperature *
                                       old_temperature * old_temperature * old_temperature;
        auto target = [=] AMREX_GPU_HOST_DEVICE(int i, int g) {
            return background * (1 + (g == 0 ? 0.01_rt : -0.01_rt) *
                                         std::cos(2.0_rt * MathConst::pi * (i + 0.5_rt) / cells));
        };
        auto face_flux = [=] AMREX_GPU_HOST_DEVICE(int i, int neighbor, int g) {
            auto const left = target(i, g);
            auto const right = target(neighbor, g);
            auto const r = std::abs(right - left) / (dx * 100.0_rt * 0.5_rt * (right + left));
            auto const lambda = (2 + r) / (6 + 3 * r + r * r);
            return PhysConst::c * lambda / 100.0_rt * (left - right) / dx;
        };
        amrex::MultiFab manufactured(source.boxArray(), source.DistributionMap(), 2, 1);
        for (amrex::MFIter mfi(manufactured); mfi.isValid(); ++mfi) {
            auto const e = manufactured.array(mfi);
            amrex::ParallelFor(mfi.validbox(), 2, [=] AMREX_GPU_DEVICE(int i, int j, int k, int g) {
                auto const q = manufactured_dt * dx * 1.e8_rt * (target(i, g) - background);
                e(i, j, k, g) = dx * target(i, g) + q +
                                manufactured_dt * (face_flux(i, i - 1, g) + face_flux(i, i + 1, g));
            });
        }
        manufactured.FillBoundary(geometry.periodicity());
        controls.diffusion.energy_groups.m_num_groups = 2;
        // Exercise the refined-grid inner linear budget without changing the
        // independent manufactured-solution or final physical residual gates.
        auto const previous_linear_tolerance = controls.diffusion.linear_tolerance;
        controls.diffusion.linear_tolerance = 5.e-12_rt;
        controls.relaxation = 1.0_rt;
        controls.diffusion.nonlinear_relaxation = 1.0_rt;
        coupled_temperature->setVal(old_temperature);
        source.setVal(0);
        auto const manufactured_result = warpx::radiation::TryAdvanceCoupledImplicitDiffusion(
            manufactured, *coupled_temperature, source, geometry, 0.0_rt, manufactured_dt, controls,
            callbacks);
        amrex::Print() << "Opposing-band manufactured solution failure="
                       << static_cast<int>(manufactured_result.failure)
                       << " material residual=" << manufactured_result.material_relative_residual
                       << '\n';
        AMREX_ALWAYS_ASSERT(manufactured_result.failure ==
                            warpx::radiation::CoupledImplicitFailure::None);
        for (amrex::MFIter mfi(manufactured); mfi.isValid(); ++mfi) {
            auto const e = manufactured.array(mfi);
            amrex::ParallelFor(mfi.validbox(), 2, [=] AMREX_GPU_DEVICE(int i, int j, int k, int g) {
                e(i, j, k, g) = e(i, j, k, g) / dx - target(i, g);
            });
        }
        for (int g = 0; g < 2; ++g) {
            AMREX_ALWAYS_ASSERT(manufactured.norm0(g) < 1.e-9_rt * background);
        }
        AMREX_ALWAYS_ASSERT(std::abs(coupled_temperature->max(0) - old_temperature) < 1.e-7_rt);
        AMREX_ALWAYS_ASSERT(manufactured_result.diffusion.maximum_relative_residual <=
                            controls.diffusion.tolerance);
        AMREX_ALWAYS_ASSERT(manufactured_result.raw_energy_relative_residual <=
                            controls.energy_tolerance);
        controls.diffusion.linear_tolerance = previous_linear_tolerance;
        // Exercise source reconciliation with a deliberately stricter material
        // equation gate, retaining the independent caloric root and raw-energy
        // checks. No acceptance tolerance is waived for this corrective step.
        controls.diffusion.energy_groups.m_num_groups = 1;
        controls.material_tolerance = 1.e-14_rt;
        coupled_temperature->setVal(old_temperature);
        radiation.setVal(dx * 20.0_rt);
        source.setVal(0);
        auto const reconciled = warpx::radiation::TryAdvanceCoupledImplicitDiffusion(
            radiation, *coupled_temperature, source, geometry, 0.0_rt, 1.e-9_rt, controls,
            callbacks);
        AMREX_ALWAYS_ASSERT(reconciled.failure == warpx::radiation::CoupledImplicitFailure::None);
        AMREX_ALWAYS_ASSERT(reconciled.source_consistency_corrections > 0);
        AMREX_ALWAYS_ASSERT(reconciled.material_relative_residual <= controls.material_tolerance);
        AMREX_ALWAYS_ASSERT(reconciled.raw_energy_relative_residual <= controls.energy_tolerance);
        AMREX_ALWAYS_ASSERT(std::abs(coupled_temperature->max(0) - reference_temperature) <
                            1.e-8_rt * (reference_temperature - old_temperature));
        amrex::Print() << "Source-consistency corrections="
                       << reconciled.source_consistency_corrections
                       << " material residual=" << reconciled.material_relative_residual << '\n';
        // Many small ideal-gas sources must not repeatedly round-trip the large
        // background through U(T)/C_V. Both the initial T and each increment
        // are exactly representable; this is an independent caloric trajectory.
        density.setVal(rho0);
        constexpr int small_steps = 4096;
        amrex::Real const small_initial = 262144.0_rt;
        amrex::Real const small_increment = 0.000244140625_rt;
        auto small_temperature = copy(temperature);
        small_temperature->setVal(small_initial);
        auto const capacity =
            eos.stateFromChargeDensityTemperature(rho0, small_initial).heat_capacity_density;
        for (int step = 0; step < small_steps; ++step) {
            source.setVal(dx * capacity * small_increment);
            AMREX_ALWAYS_ASSERT(std::isfinite(
                model.EvaluateElectronEnergySource(0, *small_temperature, source, 1.0_rt)));
        }
        auto const expected_small = small_initial + small_steps * small_increment;
        auto const small_error = amrex::max(std::abs(small_temperature->max(0) - expected_small),
                                            std::abs(small_temperature->min(0) - expected_small));
        amrex::Print() << "Repeated small ideal sources: temperature error=" << small_error << '\n';
        AMREX_ALWAYS_ASSERT(small_error <
                            4 * std::numeric_limits<amrex::Real>::epsilon() * small_initial);
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
