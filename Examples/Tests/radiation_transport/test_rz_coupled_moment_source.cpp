/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Radiation/CoupledMomentSource.H"
#include "Radiation/MomentTransportLedger.H"
#include "Radiation/ParticleImpulse.H"
#include "Radiation/RZMomentGeometry.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>

using namespace amrex::literals;

namespace
{
// Independent native-particle inventories, not the source's deposited work
// or impulse fields. Positions do not drift during this source transaction.
std::array<long double, 8>
ParticleInventory (WarpXParticleContainer& ions, bool angular = false)
{
    std::array<long double, 8> total{};
    std::array<int, 9> components{PIdx::w,
                                  PIdx::ux,
                                  PIdx::uy,
                                  PIdx::uz,
                                  PIdx::theta,
                                  ions.GetRealCompIndex("radiation_impulse_rz_coupled_ux"),
                                  ions.GetRealCompIndex("radiation_impulse_rz_coupled_uy"),
                                  ions.GetRealCompIndex("radiation_impulse_rz_coupled_uz"),
                                  ions.GetRealCompIndex("radiation_impulse_rz_coupled_work")};
    for (WarpXParIter it(ions, 0); it.isValid(); ++it)
    {
        std::array<amrex::Gpu::HostVector<amrex::ParticleReal>, 10> host;
        for (int d = 0; d < 9; ++d)
        {
            auto const& values = it.GetStructOfArrays().GetRealData(components[d]);
            host[d].resize(it.numParticles());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(),
                             values.begin() + it.numParticles(), host[d].begin());
        }
        if (angular) {
            auto const& radius = it.GetStructOfArrays().GetRealData(PIdx::r);
            host[9].resize(it.numParticles());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, radius.begin(),
                             radius.begin() + it.numParticles(), host[9].begin());
        }
        for (long p = 0; p < it.numParticles(); ++p)
        {
            long double const mass = static_cast<long double>(host[0][p]) * ions.getMass();
            long double const x = host[1][p], y = host[2][p], z = host[3][p];
            long double const square = x * x + y * y + z * z;
            total[0] += mass * square / (1 + std::sqrt(1 + square / PhysConst::c2));
            long double const cosine = std::cos(static_cast<long double>(host[4][p]));
            long double const sine = std::sin(static_cast<long double>(host[4][p]));
            total[1] += mass * (x * cosine + y * sine);
            long double const lever = angular ? host[9][p] : 1;
            total[2] += mass * lever * (-x * sine + y * cosine);
            total[3] += mass * z;
            total[4] += mass * host[8][p];
            total[5] += mass * (host[5][p] * cosine + host[6][p] * sine);
            total[6] += mass * lever * (-host[5][p] * sine + host[6][p] * cosine);
            total[7] += mass * host[7][p];
        }
    }
#ifdef AMREX_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, total.data(), 8, MPI_LONG_DOUBLE, MPI_SUM,
                  amrex::ParallelDescriptor::Communicator());
#endif
    return total;
}

amrex::Real
RadiationAngularInventory (amrex::MultiFab const& radiation, amrex::Geometry const& geometry)
{
    amrex::MultiFab weighted(radiation.boxArray(), radiation.DistributionMap(), 1, 0);
    auto const dx = geometry.CellSizeArray();
    for (amrex::MFIter it(weighted); it.isValid(); ++it) {
        auto const u = radiation.const_array(it);
        auto const out = weighted.array(it);
        amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto const midpoint = (i + 0.5_rt) * dx[0];
            auto const mean_radius = midpoint + dx[0] * dx[0] / (12 * midpoint);
            out(i, j, k) = mean_radius * u(i, j, k, 2);
        });
    }
    return weighted.sum(0);
}

amrex::Real
ElectronEnergy (WarpX const& simulation, amrex::MultiFab const& temperature)
{
    using warpx::fields::FieldType;
    auto const& geometry = simulation.Geom(0);
    auto const dx = geometry.CellSizeArray();
    auto const hi = amrex::ubound(geometry.Domain());
    auto const& charge = *simulation.m_fields.get(FieldType::rho_fp, 0);
    auto const owner = temperature.OwnerMask(geometry.periodicity());
    amrex::MultiFab energy(temperature.boxArray(), temperature.DistributionMap(), 1, 0);
    auto const axis = simulation.verboncoeurAxisCorrection() ? 1._rt / 3 : 1._rt / 4;
    bool const periodic = geometry.isPeriodic(1);
    for (amrex::MFIter it(energy); it.isValid(); ++it)
    {
        auto const out = energy.array(it);
        auto const t = temperature.const_array(it);
        auto const rho = charge.const_array(it);
        auto const owned = owner->const_array(it);
        amrex::ParallelFor(it.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               auto volume = MathConst::pi * dx[0] * dx[0] * dx[1] *
                                             (i == 0 ? axis : 2._rt * i);
                               if (i == hi.x + 1)
                               {
                                   volume *= 0.5_rt;
                               }
                               if (!periodic && (j == 0 || j == hi.y + 1))
                               {
                                   volume *= 0.5_rt;
                               }
                               out(i, j, k) = owned(i, j, k)
                                                  ? 1.5_rt * rho(i, j, k) / PhysConst::q_e *
                                                        PhysConst::kb * t(i, j, k) * volume
                                                  : 0._rt;
                           });
    }
    return energy.sum(0);
}
} // namespace

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        using namespace warpx::radiation;
        using warpx::fields::FieldType;
        auto& simulation = WarpX::GetInstance();
        auto& particles = simulation.GetPartContainer();
        auto& ions = particles.GetParticleContainerFromName("ions");
        RegisterParticleImpulseState(ions, "rz_coupled");
        simulation.InitData();
        simulation.HybridPICPrepareElectronStateForDiagnostics();
        auto& model = *simulation.get_pointer_HybridPICModel();
        auto const& geometry = simulation.Geom(0);
        auto const dx = geometry.CellSizeArray();
        auto const& live = *simulation.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        amrex::MultiFab temperature(live.boxArray(), live.DistributionMap(), 1, live.nGrowVect());
        amrex::MultiFab radiation(ions.ParticleBoxArray(0), ions.ParticleDistributionMap(0), 4, 1);
        amrex::MultiFab heat(radiation.boxArray(), radiation.DistributionMap(), 1, 1);
        amrex::MultiFab::Copy(temperature, live, 0, 0, 1, live.nGrowVect());
        radiation.setVal(0);
        bool thermal = false, spatial = false, rollback = false, angular = false;
        int steps = 64;
        amrex::ParmParse pp("test");
        pp.query("thermal", thermal);
        pp.query("spatial", spatial);
        pp.query("rollback", rollback);
        pp.query("angular", angular);
        if (angular) {
            for (WarpXParIter it(ions, 0); it.isValid(); ++it) {
                auto const data = it.GetParticleTile().getParticleTileData();
                amrex::ParallelFor(it.numParticles(), [=] AMREX_GPU_DEVICE(long p) {
                    auto const theta = data.m_rdata[PIdx::theta][p];
                    auto const speed = 1.e8_rt * data.m_rdata[PIdx::r][p];
                    data.m_rdata[PIdx::ux][p] = -speed * std::sin(theta);
                    data.m_rdata[PIdx::uy][p] = speed * std::cos(theta);
                    data.m_rdata[PIdx::uz][p] = 0;
                });
            }
        }
        pp.query("steps", steps);
        auto const density =
            thermal ? 100._rt : 0.01_rt * 1.e20_rt * ions.getMass() * PhysConst::c2;
        for (amrex::MFIter it(radiation); it.isValid(); ++it)
        {
            auto const u = radiation.array(it);
            amrex::ParallelFor(it.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   auto const lo = i * dx[0], hi = (i + 1) * dx[0];
                                   auto const volume = MathConst::pi * (hi * hi - lo * lo) * dx[1];
                                   u(i, j, k, 0) = density * volume;
                                   if (!thermal && !angular)
                                   {
                                       u(i, j, k, 1) = 1.e-3_rt * u(i, j, k, 0);
                                   }
                               });
        }
        radiation.FillBoundary(geometry.periodicity());
        constexpr amrex::Real dt = 1.e-12_rt;
        CoupledMomentCallbacks callbacks;
        callbacks.coefficients = [&] (amrex::MultiFab const& trial, amrex::MultiFab& absorption,
                                      amrex::MultiFab& scattering, amrex::MultiFab& equilibrium,
                                      amrex::Real)
        {
            absorption.setVal(thermal ? 0.2_rt / (PhysConst::c * dt) : 0);
            scattering.setVal(thermal ? 0 : 2 / (PhysConst::c * dt));
            for (amrex::MFIter it(equilibrium); it.isValid(); ++it)
            {
                auto const t = trial.const_array(it);
                auto const out = equilibrium.array(it);
                amrex::ParallelFor(it.validbox(),
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
                                       auto const lo = i * dx[0], hi = (i + 1) * dx[0];
                                       auto const volume =
                                           MathConst::pi * (hi * hi - lo * lo) * dx[1];
                                       amrex::Real average = 0;
                                       for (int a = 0; a < 2; ++a)
                                       {
                                           for (int b = 0; b < 2; ++b)
                                           {
                                               auto const value = t(i + a, j + b, k);
                                               average += value * value * value * value / 4;
                                           }
                                       }
                                       out(i, j, k) = 7.565733250280007e-16_rt * volume * average;
                                   });
            }
        };
        callbacks.material_response = [&] (amrex::MultiFab& t, amrex::MultiFab& source)
        { return model.EvaluateElectronEnergySource(0, t, source, 1.e18_rt); };
        callbacks.material_at_temperature = [&] (amrex::MultiFab& t, amrex::MultiFab& source,
                                                 amrex::MultiFab const& prescribed,
                                                 amrex::MultiFab& residual)
        {
            return model.EvaluateElectronEnergySource(0, t, source, 1.e18_rt, nullptr, &prescribed,
                                                      &residual);
        };
        CoupledMomentOptions options;
        options.spatial_transport = spatial;
        options.transport.reflecting_boundaries = true;
        pp.query("verbose", options.verbose);
        pp.query("relaxation", options.relaxation);
        if (angular) {
            options.particle_assignment = ParticleImpulseAssignment::RZAngularConservative;
            options.transport.rz_angular_transport = true;
        }
        auto const original = ParticleInventory(ions, angular);
        auto const initial_electrons = ElectronEnergy(simulation, temperature);
        auto const initial_radiation = radiation.sum(0);
        {
            amrex::MultiFab beta(radiation.boxArray(), radiation.DistributionMap(), 3, 0);
            beta.setVal(0);
            beta.setVal(1.e-3_rt,0,1);
            beta.setVal(1.e-6_rt,1,1);
            AMREX_ALWAYS_ASSERT(!TryCanonicalizeRZMeridionalVelocity(beta));
            AMREX_ALWAYS_ASSERT(beta.min(1) == 1.e-6_rt && beta.max(1) == 1.e-6_rt);
            beta.setVal(std::numeric_limits<amrex::Real>::epsilon()*1.e-3_rt,1,1);
            AMREX_ALWAYS_ASSERT(TryCanonicalizeRZMeridionalVelocity(beta));
            AMREX_ALWAYS_ASSERT(beta.norm0(1) == 0 && beta.min(0) == 1.e-3_rt
                && beta.max(0) == 1.e-3_rt && beta.norm0(2) == 0);
        }
        long double const initial_energy =
            original[0] + original[4] + initial_electrons + initial_radiation;
        std::array<amrex::Real, 3> initial_momentum{};
        for (int d = 0; d < 3; ++d)
        {
            initial_momentum[d] = radiation.sum(d + 1);
        }
        long double work = 0, caloric = 0;
        long double worst_energy = 0;
        MomentTransportAccounting total;
        if (rollback)
        {
            auto const initial_angular = RadiationAngularInventory(radiation, geometry);
            amrex::MultiFab old_radiation(radiation.boxArray(), radiation.DistributionMap(), 4, 1);
            amrex::MultiFab old_temperature(temperature.boxArray(), temperature.DistributionMap(),
                                            1, temperature.nGrowVect());
            amrex::MultiFab::Copy(old_radiation, radiation, 0, 0, 4, 1);
            amrex::MultiFab::Copy(old_temperature, temperature, 0, 0, 1, temperature.nGrowVect());
            CoupledMomentExchange exchange;
            exchange.kinetic_work.define(radiation.boxArray(), radiation.DistributionMap(), 1, 1);
            exchange.momentum.define(radiation.boxArray(), radiation.DistributionMap(), 3, 1);
            exchange.kinetic_work.setVal(31);
            exchange.momentum.setVal(37);
            exchange.transport.boundary = {41, 43, 47, 53};
            exchange.transport.geometric = {59, 61, 67, 71};
            auto const original_accounting = exchange.transport;
            MomentTransportLedger ledger;
            AMREX_ALWAYS_ASSERT(ledger.TryAccumulate(original_accounting));
            std::ostringstream original_ledger;
            AMREX_ALWAYS_ASSERT(ledger.Write(original_ledger));
            heat.setVal(79);
            bool faulted = false;
            auto guarded = callbacks;
            guarded.coefficients = [&] (amrex::MultiFab const& t, amrex::MultiFab& a,
                                        amrex::MultiFab& s, amrex::MultiFab& bath, amrex::Real end)
            {
                AMREX_ALWAYS_ASSERT(ParticleInventory(ions, angular) == original);
                AMREX_ALWAYS_ASSERT(radiation.sum(0) == initial_radiation);
                callbacks.coefficients(t, a, s, bath, end);
                if (!faulted && end > dt / 2)
                {
                    faulted = true;
                    a.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
                }
            };
            CoupledMomentIntervalOptions interval;
            interval.source = options;
            interval.initial_substeps = 2;
            interval.max_refinements = 0;
            auto const rejected = TryAdvanceCoupledMomentInterval(
                particles, {"ions"}, "rz_coupled", radiation, temperature, heat, geometry, 0, dt,
                interval, guarded, &exchange, &ledger);
            AMREX_ALWAYS_ASSERT(!rejected.valid && rejected.completed_trial_substeps == 1);
            std::ostringstream rejected_ledger;
            AMREX_ALWAYS_ASSERT(ledger.Write(rejected_ledger));
            AMREX_ALWAYS_ASSERT(rejected_ledger.str() == original_ledger.str());
            AMREX_ALWAYS_ASSERT(ParticleInventory(ions, angular) == original);
            AMREX_ALWAYS_ASSERT(heat.min(0, 1) == 79 && heat.max(0, 1) == 79);
            AMREX_ALWAYS_ASSERT(exchange.kinetic_work.min(0, 1) == 31 &&
                                exchange.kinetic_work.max(0, 1) == 31);
            for (int d = 0; d < 3; ++d)
            {
                AMREX_ALWAYS_ASSERT(exchange.momentum.min(d, 1) == 37 &&
                                    exchange.momentum.max(d, 1) == 37);
            }
            for (int d = 0; d < 4; ++d)
            {
                AMREX_ALWAYS_ASSERT(exchange.transport.boundary[d] ==
                                    original_accounting.boundary[d]);
                AMREX_ALWAYS_ASSERT(exchange.transport.geometric[d] ==
                                    original_accounting.geometric[d]);
            }
            amrex::MultiFab::Subtract(old_radiation, radiation, 0, 0, 4, 1);
            amrex::MultiFab::Subtract(old_temperature, temperature, 0, 0, 1,
                                      temperature.nGrowVect());
            AMREX_ALWAYS_ASSERT(old_radiation.norm0(0, 4, amrex::IntVect(1)) == 0);
            AMREX_ALWAYS_ASSERT(old_temperature.norm0(0, 1, temperature.nGrowVect()) == 0);

            // Four-step reference on private particles, with independently
            // accumulated physical optical-wall/geometric exchange.
            amrex::MultiFab::Copy(old_radiation, radiation, 0, 0, 4, 1);
            amrex::MultiFab::Copy(old_temperature, temperature, 0, 0, 1, temperature.nGrowVect());
            MomentTransportAccounting reference;
            auto reference_ledger = ledger;
            {
                ParticleImpulseMaterial material(particles, {"ions"});
                amrex::MultiFab reference_heat(heat.boxArray(), heat.DistributionMap(), 1, 1);
                for (int step = 0; step < 4; ++step)
                {
                    CoupledMomentExchange part;
                    // Match the interval's representable time endpoints; dt/4
                    // need not equal end-begin for the last floating-point slice.
                    auto const begin = dt * (static_cast<amrex::Real>(step) / 4);
                    auto const end = dt * (static_cast<amrex::Real>(step + 1) / 4);
                    auto const result = TryAdvanceCoupledMomentSource(
                        material, "rz_coupled", old_radiation, old_temperature, reference_heat,
                        geometry, begin, end - begin, options, callbacks, &part);
                    AMREX_ALWAYS_ASSERT(result.valid);
                    AMREX_ALWAYS_ASSERT(reference_ledger.TryAccumulate(part.transport));
                    for (int d = 0; d < 4; ++d)
                    {
                        reference.boundary[d] += part.transport.boundary[d];
                        reference.geometric[d] += part.transport.geometric[d];
                    }
                }
            }
            AMREX_ALWAYS_ASSERT(ParticleInventory(ions, angular) == original);
            faulted = false;
            interval.max_refinements = 1;
            auto const accepted = TryAdvanceCoupledMomentInterval(
                particles, {"ions"}, "rz_coupled", radiation, temperature, heat, geometry, 0, dt,
                interval, guarded, &exchange, &ledger);
            AMREX_ALWAYS_ASSERT(accepted.valid && accepted.attempts == 2 &&
                                accepted.substeps == 4 && accepted.completed_trial_substeps == 5);
            std::ostringstream accepted_ledger, expected_ledger;
            AMREX_ALWAYS_ASSERT(ledger.Write(accepted_ledger));
            AMREX_ALWAYS_ASSERT(reference_ledger.Write(expected_ledger));
            AMREX_ALWAYS_ASSERT(accepted_ledger.str() == expected_ledger.str());
            for (int d = 0; d < 4; ++d)
            {
                AMREX_ALWAYS_ASSERT(
                    std::abs(exchange.transport.boundary[d] - reference.boundary[d]) <=
                    1.e-12_rt * initial_radiation);
                AMREX_ALWAYS_ASSERT(
                    std::abs(exchange.transport.geometric[d] - reference.geometric[d]) <=
                    1.e-12_rt * initial_radiation);
            }
            auto const scale = old_radiation.norm0(0);
            amrex::MultiFab::Subtract(old_radiation, radiation, 0, 0, 4, 1);
            amrex::MultiFab::Subtract(old_temperature, temperature, 0, 0, 1,
                                      temperature.nGrowVect());
            AMREX_ALWAYS_ASSERT(old_radiation.norm0(0, 4, amrex::IntVect(1)) <= 1.e-12_rt * scale);
            AMREX_ALWAYS_ASSERT(old_temperature.norm0(0, 1, temperature.nGrowVect()) == 0);
            auto const current = ParticleInventory(ions, angular);
            auto const final_energy = current[0] + current[4] +
                                      ElectronEnergy(simulation, temperature) + radiation.sum(0);
            AMREX_ALWAYS_ASSERT(std::abs(final_energy - initial_energy) < 1.e-10L * initial_energy);
            if (angular) {
                auto const change = RadiationAngularInventory(radiation, geometry) - initial_angular
                    + PhysConst::c * ((current[2] - original[2]) + (current[6] - original[6]));
                auto const angular_scale = std::abs(initial_angular)
                    + PhysConst::c * (std::abs(original[2]) + std::abs(original[6]));
                AMREX_ALWAYS_ASSERT(std::abs(change) <= 1.e-10L * angular_scale);
            }
            amrex::Print() << "RZ source interval rollback and accepted-only exchange passed.\n";
        }
        if (!rollback)
        {
            for (int step = 0; step < steps; ++step)
            {
                CoupledMomentExchange exchange;
                auto const result = TryAdvanceCoupledMomentSource(
                    particles, {"ions"}, "rz_coupled", radiation, temperature, heat, geometry,
                    step * dt, dt, options, callbacks, &exchange);
                amrex::Print() << "RZ native source step=" << step << " valid=" << result.valid
                               << " failure=" << static_cast<int>(result.failure)
                               << " raw_energy=" << result.raw_energy_residual
                               << " raw_momentum=" << result.raw_momentum_residual << '\n';
                AMREX_ALWAYS_ASSERT(result.valid);
                work += result.actual_kinetic_work;
                caloric += heat.sum(0);
                for (int d = 0; d < 4; ++d)
                {
                    total.boundary[d] += exchange.transport.boundary[d];
                    total.geometric[d] += exchange.transport.geometric[d];
                }
                auto const current = ParticleInventory(ions, angular);
                auto const electron_energy = ElectronEnergy(simulation, temperature);
                auto const energy = current[0] + current[4] + electron_energy + radiation.sum(0);
                worst_energy = std::max(worst_energy, std::abs(energy-initial_energy)/initial_energy);
                AMREX_ALWAYS_ASSERT(worst_energy < 1.e-10L);
                AMREX_ALWAYS_ASSERT(std::abs(current[0] - original[0] - work) <
                                    1.e-10L * (std::abs(work) + original[0]));
                AMREX_ALWAYS_ASSERT(std::abs(electron_energy - initial_electrons - caloric) <
                                    1.e-10L * (initial_electrons + std::abs(caloric)));
                for (int d = 0; d < 3; ++d)
                {
                    auto const change = (angular && d == 1 ? RadiationAngularInventory(radiation, geometry)
                                                           : radiation.sum(d + 1) - initial_momentum[d]) +
                                        PhysConst::c * (current[d + 1] - original[d + 1] +
                                                        current[d + 5] - original[d + 5]) +
                                        (angular && d == 1 ? 0 : total.boundary[d + 1] - total.geometric[d + 1]);
                    auto const length = angular && d == 1 ? geometry.ProbHi(0) : 1;
                    AMREX_ALWAYS_ASSERT(std::abs(change) < 1.e-10L * initial_radiation * length);
                }
            }
            AMREX_ALWAYS_ASSERT(std::abs(work) > 1.e-8L * original[0]);
            if (thermal)
            {
                AMREX_ALWAYS_ASSERT(std::abs(caloric) > 0.01L * initial_radiation);
            }
            amrex::Print() << "Native RZ source: actual kinetic work=" << work
                           << " caloric=" << caloric << " spatial=" << spatial
                           << " independent_energy_residual=" << worst_energy << '\n';
        }
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
