/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Radiation/CoupledMomentSource.H"
#include "Radiation/ParticleImpulse.H"
#include "WarpX.H"
#include "nonuniform_moment_state.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>

using namespace amrex::literals;

namespace
{
    constexpr amrex::Real initial_u = 1.e5_rt;
    constexpr amrex::Real radiation_constant = 7.565733250280007e-16_rt;

    std::array<long double, 8>
    RadiationInventory (amrex::MultiFab const& radiation)
    {
        std::array<long double, 8> total{};
        for (amrex::MFIter iterator(radiation); iterator.isValid(); ++iterator) {
            amrex::FArrayBox host(iterator.validbox(), 4, amrex::The_Pinned_Arena());
            amrex::FArrayBox packed(iterator.validbox(), 4, amrex::The_Arena());
            amrex::Box const box = iterator.validbox();
            auto const input = radiation.const_array(iterator);
            auto const output = packed.array();
            amrex::ParallelFor(box, 4, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                output(i, j, k, n) = input(i, j, k, n);
            });
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, packed.dataPtr(),
                             packed.dataPtr() + packed.size(), host.dataPtr());
            for (int d = 0; d < 4; ++d) {
                for (long p = 0; p < box.numPts(); ++p) {
                    total[d] += host.dataPtr(d)[p];
                    total[d + 4] += std::abs(static_cast<long double>(host.dataPtr(d)[p]));
                }
            }
        }
#ifdef AMREX_USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, total.data(), 8, MPI_LONG_DOUBLE, MPI_SUM,
                      amrex::ParallelDescriptor::Communicator());
#endif
        return total;
    }

    // Independent extended-precision particle inventory. The source producer's
    // work fields are not used as the kinetic-energy or momentum oracle.
    std::array<long double, 12>
    ParticleInventory (WarpXParticleContainer& ions)
    {
        std::array<long double, 12> total{};
        std::array<int, 8> components{PIdx::w,
                                      PIdx::ux,
                                      PIdx::uy,
                                      PIdx::uz,
                                      ions.GetRealCompIndex("radiation_impulse_coupled_ux"),
                                      ions.GetRealCompIndex("radiation_impulse_coupled_uy"),
                                      ions.GetRealCompIndex("radiation_impulse_coupled_uz"),
                                      ions.GetRealCompIndex("radiation_impulse_coupled_work")};
        long double const c2 = PhysConst::c2;
        auto const old_gamma = std::sqrt(1 + initial_u * initial_u / c2);
        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
            std::array<amrex::Gpu::HostVector<amrex::ParticleReal>, 8> host;
            for (int d = 0; d < 8; ++d) {
                auto const& data = iterator.GetStructOfArrays().GetRealData(components[d]);
                host[d].resize(iterator.numParticles());
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, data.begin(),
                                 data.begin() + iterator.numParticles(), host[d].begin());
            }
            for (long p = 0; p < iterator.numParticles(); ++p) {
                long double const mass = static_cast<long double>(host[0][p]) * ions.getMass();
                long double squared = 0;
                long double difference_squared = 0;
                for (int d = 0; d < 3; ++d) {
                    long double const u = host[d + 1][p];
                    long double const old = d == 2 ? initial_u : 0;
                    squared += u * u;
                    difference_squared += (u - old) * (u + old);
                    total[1 + d] += mass * (u - old);
                    total[9 + d] += std::abs(mass * (u - old));
                    total[5 + d] += mass * host[4 + d][p];
                }
                total[0] += mass;
                total[4] += mass * difference_squared / (old_gamma + std::sqrt(1 + squared / c2));
                total[8] += mass * host[7][p];
            }
        }
#ifdef AMREX_USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, total.data(), 12, MPI_LONG_DOUBLE, MPI_SUM,
                      amrex::ParallelDescriptor::Communicator());
#endif
        return total;
    }
} // namespace

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        using namespace warpx::radiation;
        using warpx::fields::FieldType;
        std::string kind = "drag";
        amrex::ParmParse test("test");
        test.query("kind", kind);
        AMREX_ALWAYS_ASSERT(kind == "drag" || kind == "thermal");
        bool const drag = kind == "drag";
        bool nonuniform = false;
        test.query("nonuniform", nonuniform);
        bool interval = false;
        test.query("interval", interval);
        bool spatial = false;
        test.query("spatial", spatial);
        bool shape = false;
        test.query("shape", shape);
        auto const assignment = shape ? ParticleImpulseAssignment::LinearNodalCellAverage
            : ParticleImpulseAssignment::NearestCell;
        int progress_interval = 0;
        test.query("progress_interval",progress_interval);
        amrex::Real absorption_step = 0.1_rt;
        test.query("absorption_step", absorption_step);
        auto& simulation = WarpX::GetInstance();
        auto& particles = simulation.GetPartContainer();
        auto& ions = particles.GetParticleContainerFromName("ions");
        RegisterParticleImpulseState(ions, "coupled");
        simulation.InitData();
        simulation.HybridPICPrepareElectronStateForDiagnostics();
        auto& model = *simulation.get_pointer_HybridPICModel();
        auto const eos = model.electronThermodynamicsExecutor();
        auto const& live_t = *simulation.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto const& geometry = simulation.Geom(0);
        amrex::Real volume = 1;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            volume *= geometry.CellSize(d);
        }
        amrex::MultiFab radiation(ions.ParticleBoxArray(0), ions.ParticleDistributionMap(0), 4, 1);
        amrex::MultiFab temperature(live_t.boxArray(), live_t.DistributionMap(), 1,
                                    live_t.nGrowVect());
        amrex::MultiFab heat(radiation.boxArray(), radiation.DistributionMap(), 1, 1);
        amrex::MultiFab::Copy(temperature, live_t, 0, 0, 1, live_t.nGrowVect());
        radiation.setVal(0);
        radiation.setVal(drag ? 0.1_rt * 1.e20_rt * ions.getMass() * PhysConst::c2 * volume
                              : 100 * volume,
                         0, 1);
        if (drag) {
            amrex::MultiFab::Copy(radiation, radiation, 0, 1, 1, 0);
            radiation.mult(1.e-4_rt, 1, 1, 0);
        }
        radiation.FillBoundary(geometry.periodicity());
        constexpr amrex::Real dt = 1.e-10_rt;
        CoupledMomentCallbacks callbacks;
        callbacks.coefficients = [&] (amrex::MultiFab const& trial, amrex::MultiFab& absorption,
                                      amrex::MultiFab& scattering, amrex::MultiFab& equilibrium,
                                      amrex::Real) {
            absorption.setVal(drag ? 0 : absorption_step / (PhysConst::c * dt));
            scattering.setVal(drag ? 20 / (PhysConst::c * dt) : 0);
            for (amrex::MFIter iterator(equilibrium); iterator.isValid(); ++iterator) {
                auto const t = trial.const_array(iterator);
                auto const b = equilibrium.array(iterator);
                amrex::ParallelFor(iterator.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    amrex::Real sum = 0;
                    for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner) {
                        auto const value =
                            t(i + (corner & 1), j + ((corner >> 1) & 1), k + ((corner >> 2) & 1));
                        sum += value * value * value * value;
                    }
                    b(i, j, k) = radiation_constant * volume * sum / (1 << AMREX_SPACEDIM);
                });
            }
        };
        callbacks.material_response = [&] (amrex::MultiFab& trial, amrex::MultiFab& source) {
            return model.EvaluateElectronEnergySource(0, trial, source, 1.e18_rt);
        };
        callbacks.material_at_temperature = [&] (amrex::MultiFab& old, amrex::MultiFab& source,
                                                 amrex::MultiFab const& prescribed,
                                                 amrex::MultiFab& residual) {
            return model.EvaluateElectronEnergySource(0, old, source, 1.e18_rt, nullptr,
                                                      &prescribed, &residual);
        };
        if (nonuniform) {
            AMREX_ALWAYS_ASSERT(!drag);
            auto const nx = geometry.Domain().length(0);
            auto const ny = AMREX_SPACEDIM == 2 ? geometry.Domain().length(1) : 1;
            auto const base_t = temperature.min(0);
            for (amrex::MFIter iterator(temperature); iterator.isValid(); ++iterator) {
                auto const t = temperature.array(iterator);
                amrex::ParallelFor(iterator.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    t(i, j, k) = base_t * (1 + 0.2_rt * std::sin(6.283185307179586_rt * i / nx) +
                                           0.15_rt * std::sin(6.283185307179586_rt * j / ny));
                });
            }
            temperature.FillBoundary(geometry.periodicity());
            for (amrex::MFIter iterator(radiation); iterator.isValid(); ++iterator) {
                auto const r = radiation.array(iterator);
                amrex::ParallelFor(iterator.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    r(i, j, k, 0) =
                        100 * volume *
                        (1 + 0.3_rt * std::cos(6.283185307179586_rt * (i + 0.5_rt) / nx));
                    r(i, j, k, 1) = 0.2_rt * r(i, j, k, 0);
                    r(i, j, k, 2) = 0;
                    r(i, j, k, 3) = -0.1_rt * r(i, j, k, 0);
                });
            }
            radiation.FillBoundary(geometry.periodicity());
            auto const before = qualification::CollectState(radiation, temperature, ions, geometry);
            if (shape) { qualification::WriteParticles(ions, "particles_before.txt"); }
            CoupledMomentOptions options;
            options.spatial_transport = spatial;
            options.particle_assignment = assignment;
            options.max_iterations = 150;
            test.query("verbose", options.verbose);
            auto const result = TryAdvanceCoupledMomentSource(particles, {"ions"}, "coupled",
                                                              radiation, temperature, heat,
                                                              geometry, 0, dt, options, callbacks);
            amrex::Print() << "Nonuniform coupled source valid=" << result.valid
                           << " iterations=" << result.iterations
                           << " heat=" << result.material_residual
                           << " momentum=" << result.source_residual << '\n';
            AMREX_ALWAYS_ASSERT(result.valid);
            auto const after = qualification::CollectState(radiation, temperature, ions, geometry);
            if (shape) { qualification::WriteParticles(ions, "particles_after.txt"); }
            qualification::WriteState(before, after, geometry, volume, absorption_step,
                                      eos.isFixedChargeLatentEnergy(), dt, spatial);
        } else if (interval) {
            auto const original_particles = ParticleInventory(ions);
            auto const original_radiation = RadiationInventory(radiation);
            auto const original_t = temperature.min(0);
            amrex::MultiFab reference_radiation(radiation.boxArray(), radiation.DistributionMap(),
                                                4, radiation.nGrowVect());
            amrex::MultiFab reference_temperature(
                temperature.boxArray(), temperature.DistributionMap(), 1, temperature.nGrowVect());
            amrex::MultiFab::Copy(reference_radiation, radiation, 0, 0, 4, radiation.nGrowVect());
            amrex::MultiFab::Copy(reference_temperature, temperature, 0, 0, 1,
                                  temperature.nGrowVect());
            bool faulted = false;
            auto guarded = callbacks;
            guarded.coefficients = [&] (amrex::MultiFab const& trial, amrex::MultiFab& a,
                                        amrex::MultiFab& s, amrex::MultiFab& bath,
                                        amrex::Real end_time) {
                // Observe live state while a previous substep has already
                // accepted into the private copy. Test-only fault injection
                // makes the second substep fail on its first source evaluation.
                AMREX_ALWAYS_ASSERT(ParticleInventory(ions) == original_particles);
                AMREX_ALWAYS_ASSERT(RadiationInventory(radiation) == original_radiation);
                AMREX_ALWAYS_ASSERT(temperature.min(0) == original_t);
                callbacks.coefficients(trial, a, s, bath, end_time);
                if (!faulted && end_time > 0.5_rt * dt) {
                    faulted = true;
                    a.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
                }
            };
            CoupledMomentIntervalOptions options;
            options.source.spatial_transport = spatial;
            options.source.particle_assignment = assignment;
            options.initial_substeps = 2;
            options.max_refinements = 0;
            CoupledMomentExchange exchange;
            exchange.kinetic_work.define(radiation.boxArray(), radiation.DistributionMap(), 1, 1);
            exchange.momentum.define(radiation.boxArray(), radiation.DistributionMap(), 3, 1);
            exchange.kinetic_work.setVal(77);
            exchange.momentum.setVal(88);
            heat.setVal(123);
            auto const rejected = TryAdvanceCoupledMomentInterval(
                particles, {"ions"}, "coupled", radiation, temperature, heat, geometry, 0, dt,
                options, guarded, &exchange);
            AMREX_ALWAYS_ASSERT(!rejected.valid && rejected.completed_trial_substeps == 1);
            AMREX_ALWAYS_ASSERT(rejected.actual_kinetic_work == 0 &&
                                rejected.carry_energy_change == 0 && rejected.substeps == 0);
            AMREX_ALWAYS_ASSERT(ParticleInventory(ions) == original_particles);
            AMREX_ALWAYS_ASSERT(RadiationInventory(radiation) == original_radiation);
            AMREX_ALWAYS_ASSERT(heat.min(0, heat.nGrow()) == 123 &&
                                heat.max(0, heat.nGrow()) == 123);
            AMREX_ALWAYS_ASSERT(exchange.kinetic_work.nGrow() == 1 &&
                exchange.kinetic_work.min(0, 1) == 77 && exchange.kinetic_work.max(0, 1) == 77);
            for (int d = 0; d < 3; ++d) {
                AMREX_ALWAYS_ASSERT(exchange.momentum.min(d, 1) == 88 &&
                    exchange.momentum.max(d, 1) == 88);
            }
            amrex::MultiFab::Subtract(reference_radiation, radiation, 0, 0, 4,
                                      radiation.nGrowVect());
            amrex::MultiFab::Subtract(reference_temperature, temperature, 0, 0, 1,
                                      temperature.nGrowVect());
            AMREX_ALWAYS_ASSERT(reference_radiation.norm0(0, 4, radiation.nGrowVect()) == 0);
            AMREX_ALWAYS_ASSERT(reference_temperature.norm0(0, 1, temperature.nGrowVect()) == 0);
            amrex::MultiFab::Copy(reference_radiation, radiation, 0, 0, 4, radiation.nGrowVect());
            amrex::MultiFab::Copy(reference_temperature, temperature, 0, 0, 1,
                                  temperature.nGrowVect());
            // Fixed four-substep reference, itself never committed to live ions.
            amrex::Real reference_work = 0;
            amrex::Real reference_carry = 0;
            {
                ParticleImpulseMaterial private_material(particles, {"ions"});
                amrex::MultiFab reference_heat(heat.boxArray(), heat.DistributionMap(), 1,
                                               heat.nGrowVect());
                for (int step = 0; step < 4; ++step) {
                    auto const begin = dt * (static_cast<amrex::Real>(step) / 4);
                    auto const end = dt * (static_cast<amrex::Real>(step + 1) / 4);
                    auto const source = TryAdvanceCoupledMomentSource(
                        private_material, "coupled", reference_radiation, reference_temperature,
                        reference_heat, geometry, begin, end - begin, options.source, callbacks);
                    AMREX_ALWAYS_ASSERT(source.valid);
                    reference_work += source.actual_kinetic_work;
                    reference_carry += source.carry_energy_change;
                    AMREX_ALWAYS_ASSERT(ParticleInventory(ions) == original_particles);
                }
            }
            faulted = false;
            options.max_refinements = 1;
            auto const accepted = TryAdvanceCoupledMomentInterval(
                particles, {"ions"}, "coupled", radiation, temperature, heat, geometry, 0, dt,
                options, guarded, &exchange);
            AMREX_ALWAYS_ASSERT(accepted.valid && accepted.attempts == 2 &&
                                accepted.substeps == 4 && accepted.completed_trial_substeps == 5);
            auto const measured = ParticleInventory(ions);
            AMREX_ALWAYS_ASSERT(std::abs(measured[4]) > 0);
            auto const work_scale = std::abs(reference_work);
            AMREX_ALWAYS_ASSERT(std::abs(exchange.kinetic_work.sum(0) - measured[4])
                < 1.e-10L * work_scale);
            for (int d = 0; d < 3; ++d) {
                AMREX_ALWAYS_ASSERT(PhysConst::c *
                    std::abs(exchange.momentum.sum(d) - measured[1 + d])
                    < 1.e-10L * original_radiation[0]);
            }
            AMREX_ALWAYS_ASSERT(std::abs(measured[4] - reference_work) < 1.e-10L * work_scale);
            AMREX_ALWAYS_ASSERT(std::abs(measured[8] - reference_carry) < 1.e-10L * work_scale);
            auto const radiation_scale = radiation.norm0(0);
            amrex::MultiFab::Subtract(reference_radiation, radiation, 0, 0, 4,
                                      radiation.nGrowVect());
            amrex::MultiFab::Subtract(reference_temperature, temperature, 0, 0, 1,
                                      temperature.nGrowVect());
            AMREX_ALWAYS_ASSERT(reference_radiation.norm0(0, 4, radiation.nGrowVect()) <=
                                1.e-12_rt * radiation_scale);
            AMREX_ALWAYS_ASSERT(reference_temperature.norm0(0, 1, temperature.nGrowVect()) == 0);
            auto const measured_radiation = RadiationInventory(radiation);
            auto const number = original_particles[0] / ions.getMass();
            auto caloric = [&] (long double t) {
                auto const ev = t * PhysConst::kb / PhysConst::q_e;
                auto const power = ev * ev * ev * ev;
                return number * PhysConst::q_e *
                       (1.5L * ev +
                        (eos.isFixedChargeLatentEnergy() ? 2 * power / (1 + power) : 0));
            };
            auto const native_heat = caloric(temperature.min(0)) - caloric(original_t);
            AMREX_ALWAYS_ASSERT(std::abs(native_heat - heat.sum(0)) <
                                1.e-10L * original_radiation[0]);
            auto const balance_energy = measured_radiation[0] - original_radiation[0] +
                                        measured[4] + measured[8] + native_heat;
            AMREX_ALWAYS_ASSERT(std::abs(balance_energy) < 1.e-10L * original_radiation[0]);
            for (int d = 0; d < 3; ++d) {
                auto const balance = measured_radiation[1 + d] - original_radiation[1 + d] +
                                     PhysConst::c * (measured[1 + d] + measured[5 + d]);
                AMREX_ALWAYS_ASSERT(std::abs(balance) < 1.e-10L * original_radiation[0]);
            }
            amrex::Print() << "Private interval: late failure discarded, retry equals four-substep "
                           << "reference; actual energy=" << accepted.raw_energy_residual << '\n';
            // A stale interval must never overwrite an intervening live kick.
            ParticleImpulseMaterial stale(particles, {"ions"});
            auto const before_external = ParticleInventory(ions);
            for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
                auto* ux = iterator.GetStructOfArrays().GetRealData(PIdx::ux).dataPtr();
                amrex::ParallelFor(iterator.numParticles(),
                                   [=] AMREX_GPU_DEVICE(long p) { ux[p] += 1; });
            }
            auto const externally_changed = ParticleInventory(ions);
            AMREX_ALWAYS_ASSERT(externally_changed != before_external);
            AMREX_ALWAYS_ASSERT(!stale.Commit());
            AMREX_ALWAYS_ASSERT(ParticleInventory(ions) == externally_changed);
        } else {
            auto const initial = ParticleInventory(ions);
            long double const mass = initial[0];
            long double const particles_count = mass / ions.getMass();
            auto caloric = [&] (long double t) {
                auto const ev = t * PhysConst::kb / PhysConst::q_e;
                auto const power = ev * ev * ev * ev;
                return particles_count * PhysConst::q_e *
                       (1.5L * ev +
                        (eos.isFixedChargeLatentEnergy() ? 2 * power / (1 + power) : 0));
            };
            long double const t0 = temperature.min(0);
            long double const u0 = caloric(t0);
            auto const initial_radiation = RadiationInventory(radiation);
            long double const e0 = initial_radiation[0];
            std::array<long double, 3> q0{};
            for (int d = 0; d < 3; ++d) {
                q0[d] = initial_radiation[d + 1];
            }
            CoupledMomentOptions options;
            options.spatial_transport = spatial;
            options.particle_assignment = assignment;
            test.query("verbose", options.verbose);
            options.max_iterations = 1;
            amrex::MultiFab saved_radiation(radiation.boxArray(), radiation.DistributionMap(), 4,
                                            radiation.nGrowVect());
            amrex::MultiFab saved_temperature(temperature.boxArray(), temperature.DistributionMap(),
                                              1, temperature.nGrowVect());
            amrex::MultiFab::Copy(saved_radiation, radiation, 0, 0, 4, radiation.nGrowVect());
            amrex::MultiFab::Copy(saved_temperature, temperature, 0, 0, 1, temperature.nGrowVect());
            heat.setVal(123);
            auto const rejected = TryAdvanceCoupledMomentSource(
                particles, {"ions"}, "coupled", radiation, temperature, heat, geometry, 0, dt,
                options, callbacks);
            AMREX_ALWAYS_ASSERT(!rejected.valid &&
                                RadiationInventory(radiation) == initial_radiation &&
                                temperature.min(0) == t0);
            AMREX_ALWAYS_ASSERT(rejected.actual_kinetic_work == 0 &&
                                rejected.carry_energy_change == 0);
            AMREX_ALWAYS_ASSERT(heat.min(0) == 123 && heat.max(0) == 123);
            amrex::MultiFab::Subtract(saved_radiation, radiation, 0, 0, 4, radiation.nGrowVect());
            amrex::MultiFab::Subtract(saved_temperature, temperature, 0, 0, 1,
                                      temperature.nGrowVect());
            AMREX_ALWAYS_ASSERT(saved_radiation.norm0(0, 4, radiation.nGrowVect()) == 0);
            AMREX_ALWAYS_ASSERT(saved_temperature.norm0(0, 1, temperature.nGrowVect()) == 0);
            auto const unchanged = ParticleInventory(ions);
            for (int d = 1; d < 9; ++d) {
                AMREX_ALWAYS_ASSERT(unchanged[d] == initial[d]);
            }
            options.max_iterations = 150;
            long double worst_energy = 0;
            long double worst_momentum = 0;
            int maximum_iterations = 0;
            for (int step = 0; step < 400; ++step) {
                if (progress_interval > 0 && step % progress_interval == 0) {
                    amrex::Print() << "Coupled source starting step=" << step << "/400" << std::endl;
                }
                auto const result = TryAdvanceCoupledMomentSource(
                    particles, {"ions"}, "coupled", radiation, temperature, heat, geometry,
                    step * dt, dt, options, callbacks);
                if (!result.valid) {
                    amrex::Print()
                        << "Coupled failure step=" << step
                        << " code=" << static_cast<int>(result.failure)
                        << " iterations=" << result.iterations
                        << " material=" << result.material_residual
                        << " source=" << result.source_residual << " work=" << result.work_residual
                        << " energy=" << result.raw_energy_residual
                        << " momentum=" << result.raw_momentum_residual << '\n';
                }
                AMREX_ALWAYS_ASSERT(result.valid);
                maximum_iterations = amrex::max(maximum_iterations, result.iterations);
                model.CommitElectronTemperature(0, temperature);
                ions.PushX(0, dt);
                ions.Redistribute();
                auto const measured = ParticleInventory(ions);
                auto const measured_radiation = RadiationInventory(radiation);
                auto const t = 0.5L * (live_t.min(0) + live_t.max(0));
                AMREX_ALWAYS_ASSERT((live_t.max(0) - live_t.min(0)) / t < 1.e-11L);
                if (drag) {
                    AMREX_ALWAYS_ASSERT(live_t.min(0) == t0 && live_t.max(0) == t0);
                }
                auto const balance =
                    measured_radiation[0] - e0 + measured[4] + caloric(t) - u0 + measured[8];
                worst_energy = std::max(worst_energy, std::abs(balance) / e0);
                for (int d = 0; d < 3; ++d) {
                    auto const q = measured_radiation[d + 1];
                    auto const balance_p =
                        q - q0[d] + PhysConst::c * (measured[1 + d] + measured[5 + d]);
                    // Spatial fluxes can produce cancelling transverse changes
                    // even for nominally uniform data. Normalize their global
                    // balance by actual local inventories, not two near-zero
                    // net sums. The physical 1e-10 gate is unchanged.
                    auto const scale =
                        (spatial ? measured_radiation[d + 5] : std::abs(q)) + std::abs(q0[d]) +
                        PhysConst::c * (spatial ? measured[9 + d] : std::abs(measured[1 + d]));
                    worst_momentum =
                        std::max(worst_momentum,
                                 scale > 0 ? std::abs(balance_p) / scale : std::abs(balance_p));
                }
                if (!(worst_energy < 1.e-10L && worst_momentum < 1.e-10L)) {
                    amrex::Print() << "Trajectory ledger failure step=" << step
                                   << " energy=" << static_cast<double>(worst_energy)
                                   << " momentum=" << static_cast<double>(worst_momentum) << '\n';
                    for (int d = 0; d < 3; ++d) {
                        amrex::Print()
                            << "component=" << d
                            << " radiation=" << static_cast<double>(measured_radiation[d + 1])
                            << " initial=" << static_cast<double>(q0[d])
                            << " ion change=" << static_cast<double>(PhysConst::c * measured[1 + d])
                            << " carry=" << static_cast<double>(PhysConst::c * measured[5 + d])
                            << " radiation L1=" << radiation.norm1(d + 1) << '\n';
                    }
                }
                AMREX_ALWAYS_ASSERT(worst_energy < 1.e-10L && worst_momentum < 1.e-10L);
            }
            // Independent homogeneous equilibrium: total momentum fixes the common
            // material/rest-isotropic radiation velocity. No source-solver routines
            // enter this oracle; conservation alone would also pass a zero update.
            long double const c = PhysConst::c;
            long double const gamma0 = std::sqrt(1 + initial_u * initial_u / (c * c));
            long double const k0 = mass * initial_u * initial_u / (gamma0 + 1);
            std::array<long double, 3> total_p{};
            long double pnorm = 0;
            for (int d = 0; d < 3; ++d) {
                total_p[d] = q0[d] / c + (d == 2 ? mass * initial_u : 0);
                pnorm += total_p[d] * total_p[d];
            }
            pnorm = std::sqrt(pnorm);
            auto bisect = [] (auto residual, long double lo, long double hi) {
                AMREX_ALWAYS_ASSERT(residual(lo) <= 0 && residual(hi) >= 0);
                for (int iteration = 0; iteration < 160; ++iteration) {
                    auto const mid = (lo + hi) / 2;
                    if (residual(mid) > 0) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                return (lo + hi) / 2;
            };
            auto kinetic = [=] (long double beta) {
                auto const gamma = 1 / std::sqrt(1 - beta * beta);
                return mass * c * c * gamma * gamma * beta * beta / (gamma + 1);
            };
            long double domain_volume = volume * radiation.boxArray().numPts();
            auto equilibrium_energy = [=] (long double t, long double beta) {
                if (drag) {
                    return e0 + k0 - kinetic(beta);
                }
                return radiation_constant * domain_volume * t * t * t * t * (1 + beta * beta / 3) /
                       (1 - beta * beta);
            };
            auto equilibrium_beta = [&] (long double t) {
                return bisect(
                    [&] (long double beta) {
                        return mass * c * beta / std::sqrt(1 - beta * beta) +
                               4 * beta / (3 + beta * beta) * equilibrium_energy(t, beta) / c -
                               pnorm;
                    },
                    0.L, 0.009L);
            };
            long double expected_t = t0;
            if (!drag) {
                expected_t = bisect(
                    [&] (long double t) {
                        auto const beta = equilibrium_beta(t);
                        return kinetic(beta) + caloric(t) + equilibrium_energy(t, beta) - k0 - u0 -
                               e0;
                    },
                    0.L, 100 * PhysConst::q_e / PhysConst::kb);
            }
            auto const expected_beta = equilibrium_beta(expected_t);
            auto const expected_e = equilibrium_energy(expected_t, expected_beta);
            auto const measured = ParticleInventory(ions);
            auto const measured_radiation = RadiationInventory(radiation);
            long double worst_endpoint = std::abs(temperature.min(0) - expected_t) / expected_t;
            auto const energy_change = expected_e - e0;
            AMREX_ALWAYS_ASSERT(std::abs(energy_change) > 1.e-9L * e0);
            worst_endpoint = std::max(worst_endpoint, std::abs(measured_radiation[0] - expected_e) /
                                                          std::abs(energy_change));
            for (int d = 0; d < 3; ++d) {
                auto const direction = total_p[d] / pnorm;
                auto const expected_q = 4 * expected_beta / (3 + expected_beta * expected_beta) *
                                        expected_e * direction;
                auto const expected_dp = mass * c * expected_beta /
                                             std::sqrt(1 - expected_beta * expected_beta) *
                                             direction -
                                         (d == 2 ? mass * initial_u : 0);
                if (expected_dp != 0) {
                    worst_endpoint =
                        std::max(worst_endpoint,
                                 std::abs(measured[1 + d] - expected_dp) / std::abs(expected_dp));
                }
                if (expected_q != 0) {
                    worst_endpoint =
                        std::max(worst_endpoint, std::abs(measured_radiation[d + 1] - expected_q) /
                                                     std::abs(expected_q));
                }
            }
            amrex::Print() << "Independent equilibrium relative error="
                           << static_cast<double>(worst_endpoint) << '\n';
            AMREX_ALWAYS_ASSERT(worst_endpoint < 1.e-8L);
            amrex::Print() << "Finite-mass/native-electron source " << kind
                           << ": 400 steps, max iterations=" << maximum_iterations
                           << " actual energy=" << static_cast<double>(worst_energy)
                           << " actual momentum=" << static_cast<double>(worst_momentum) << '\n';
        }
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
