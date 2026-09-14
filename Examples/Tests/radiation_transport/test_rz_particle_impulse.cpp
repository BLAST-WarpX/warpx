/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/ParticleBoundaries.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Radiation/ParticleImpulse.H"
#include "Radiation/ParticleImpulseBoundary.H"
#include "Radiation/RadiationTransport.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>

using namespace amrex::literals;

namespace
{
using Snapshot = std::vector<std::array<amrex::Real, 11>>;

std::array<amrex::Real, 4>
NativeInventory (WarpX& simulation)
{
    using warpx::fields::FieldType;
    auto const& geometry = simulation.Geom(0);
    auto const dx = geometry.CellSizeArray();
    auto const hi = amrex::ubound(geometry.Domain());
    auto const& temperature =
        *simulation.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
    auto const& charge = *simulation.m_fields.get(FieldType::rho_fp, 0);
    auto const owner = temperature.OwnerMask(geometry.periodicity());
    amrex::MultiFab energy(temperature.boxArray(), temperature.DistributionMap(), 1, 0);
    auto const axis = simulation.verboncoeurAxisCorrection() ? 1._rt / 3 : 1._rt / 4;
    bool const periodic = geometry.isPeriodic(1);
    for (amrex::MFIter it(energy); it.isValid(); ++it)
    {
        auto const output = energy.array(it);
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
                               output(i, j, k) = owned(i, j, k)
                                                     ? 1.5_rt * rho(i, j, k) / PhysConst::q_e *
                                                           PhysConst::kb * t(i, j, k) * volume
                                                     : 0._rt;
                           });
    }
    auto& particles = simulation.GetPartContainer();
    auto const pending = simulation.GetRadiationTransport().pendingMaterialImpulse(particles, true);
    auto radiation = particles.GetParticleContainerFromName("photons").sumParticleEnergy();
    if (simulation.m_fields.has(FieldType::radiation_diffusion_energy, 0)) {
        radiation += simulation.m_fields.get(FieldType::radiation_diffusion_energy, 0)->sum(0);
    }
    return {particles.GetParticleContainerFromName("ions").sumParticleEnergy(), energy.sum(0),
            radiation, pending[3]};
}

Snapshot
ReadParticles (WarpXParticleContainer& ions)
{
    Snapshot result;
    for (WarpXParIter it(ions, 0); it.isValid(); ++it)
    {
        auto const count = it.numParticles();
        auto const first = result.size();
        result.resize(first + count);
        std::array<int, 11> const components{
            PIdx::w,
            PIdx::ux,
            PIdx::uy,
            PIdx::uz,
            PIdx::theta,
            PIdx::r,
            PIdx::z,
            ions.GetRealCompIndex("radiation_impulse_rz_probe_ux"),
            ions.GetRealCompIndex("radiation_impulse_rz_probe_uy"),
            ions.GetRealCompIndex("radiation_impulse_rz_probe_uz"),
            ions.GetRealCompIndex("radiation_impulse_rz_probe_work")};
        for (int d = 0; d < 11; ++d)
        {
            auto const& values = it.GetStructOfArrays().GetRealData(components[d]);
            amrex::Gpu::HostVector<amrex::ParticleReal> host(count);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(), values.begin() + count,
                             host.begin());
            for (int p = 0; p < count; ++p)
            {
                result[first + p][d] = host[p];
            }
        }
    }
    return result;
}
void
CheckWalls (MultiParticleContainer& particles, WarpXParticleContainer& ions)
{
    using namespace warpx::radiation;
    ParticleBoundaries boundaries;
    boundaries.SetAll(ParticleBoundaryType::Periodic);
    boundaries.SetBoundsX(ParticleBoundaryType::None, ParticleBoundaryType::Reflecting);
    bool const axial_walls = !ions.Geom(0).isPeriodic(1);
    if (axial_walls)
    {
        boundaries.SetBoundsZ(ParticleBoundaryType::Reflecting, ParticleBoundaryType::Reflecting);
    }
    boundaries.BuildReflectionModelParsers();
    for (bool reflect_all : {false, true})
    {
        boundaries.Set_reflect_all_velocities(reflect_all);
        for (WarpXParIter it(ions, 0); it.isValid(); ++it)
        {
            auto const data = it.GetParticleTile().getParticleTileData();
            amrex::GpuArray<int, 4> const components{
                ions.GetRealCompIndex("radiation_impulse_rz_probe_ux"),
                ions.GetRealCompIndex("radiation_impulse_rz_probe_uy"),
                ions.GetRealCompIndex("radiation_impulse_rz_probe_uz"),
                ions.GetRealCompIndex("radiation_impulse_rz_probe_work")};
            amrex::GpuArray<amrex::ParticleReal*, 4> carries{};
            for (int d = 0; d < 4; ++d)
            {
                carries[d] = it.GetStructOfArrays().GetRealData(components[d]).data();
            }
            amrex::ParallelFor(it.numParticles(),
                               [=] AMREX_GPU_DEVICE(long p)
                               {
                                   data.m_rdata[PIdx::r][p] = p % 2 == 0 ? 1.125_prt : 0.5_prt;
                                   if (axial_walls)
                                   {
                                       data.m_rdata[PIdx::z][p] =
                                           p % 3 == 0 ? -0.125_prt
                                                      : (p % 3 == 1 ? 1.125_prt : 0.5_prt);
                                   }
                                   for (int d = 0; d < 4; ++d)
                                   {
                                       carries[d][p] = (d + 1) * 1.e-30_prt;
                                   }
                               });
        }
        auto const before = ReadParticles(ions);
        auto const old_inventory = ParticleImpulseInventory(particles, {"ions"}, "rz_probe");
        std::vector<ParticleImpulseBoundaryTransfer> transfers;
        AMREX_ALWAYS_ASSERT(!TryReflectParticleImpulseState(ions, boundaries, transfers,
                                                            [] (auto const&) { return false; }));
        AMREX_ALWAYS_ASSERT(transfers.empty() && ReadParticles(ions) == before);
        AMREX_ALWAYS_ASSERT(TryReflectParticleImpulseState(ions, boundaries, transfers));
        AMREX_ALWAYS_ASSERT(transfers.size() == 1 && transfers[0].path == "rz_probe");
        auto const after = ReadParticles(ions);
        amrex::Real mass = 0;
        for (std::size_t p = 0; p < before.size(); ++p)
        {
            mass += before[p][0] * ions.getMass();
            bool const radial_wall = before[p][5] > 1;
            bool const axial_wall = before[p][6] < 0 || before[p][6] > 1;
            if (!radial_wall && !axial_wall)
            {
                AMREX_ALWAYS_ASSERT(before[p] == after[p]);
                continue;
            }
            AMREX_ALWAYS_ASSERT(after[p][5] == (radial_wall ? 0.875_rt : before[p][5]));
            auto const expected_z = before[p][6] < 0
                                        ? -before[p][6]
                                        : (before[p][6] > 1 ? 2 - before[p][6] : before[p][6]);
            AMREX_ALWAYS_ASSERT(after[p][6] == expected_z);
            AMREX_ALWAYS_ASSERT(after[p][4] == before[p][4]);
            AMREX_ALWAYS_ASSERT(after[p][10] == before[p][10]);
            long double const cosine = std::cos(static_cast<long double>(before[p][4]));
            long double const sine = std::sin(static_cast<long double>(before[p][4]));
            for (int offset : {1, 7})
            {
                long double const x = before[p][offset];
                long double const y = before[p][offset + 1];
                long double const radial =
                    (radial_wall || reflect_all ? -1 : 1) * (x * cosine + y * sine);
                long double const azimuthal = (reflect_all ? -1 : 1) * (-x * sine + y * cosine);
                auto const expected_x = radial * cosine - azimuthal * sine;
                auto const expected_y = radial * sine + azimuthal * cosine;
                auto const scale = std::abs(x) + std::abs(y);
                AMREX_ALWAYS_ASSERT(std::abs(after[p][offset] - expected_x) < 2.e-14L * scale);
                AMREX_ALWAYS_ASSERT(std::abs(after[p][offset + 1] - expected_y) < 2.e-14L * scale);
                AMREX_ALWAYS_ASSERT(after[p][offset + 2] ==
                                    (axial_wall || reflect_all ? -1 : 1) * before[p][offset + 2]);
            }
        }
        amrex::ParallelDescriptor::ReduceRealSum(mass);
        auto const inventory = ParticleImpulseInventory(particles, {"ions"}, "rz_probe");
        for (int d = 0; d < 3; ++d)
        {
            AMREX_ALWAYS_ASSERT(std::abs(old_inventory[d] - inventory[d] -
                                         transfers[0].momentum[d]) < 2.e-12_rt * mass * 1.e-30_rt);
        }
        AMREX_ALWAYS_ASSERT(inventory[3] == old_inventory[3]);
        AMREX_ALWAYS_ASSERT(TryReflectParticleImpulseState(ions, boundaries, transfers));
        AMREX_ALWAYS_ASSERT(ReadParticles(ions) == after);
        for (auto const transfer : transfers[0].momentum)
        {
            AMREX_ALWAYS_ASSERT(transfer == 0);
        }
    }
    // Exercise the actual particle-boundary entry point and its persistent,
    // compensated wall ledger, then migrate the reflected particles between FABs.
    for (WarpXParIter it(ions, 0); it.isValid(); ++it)
    {
        auto const data = it.GetParticleTile().getParticleTileData();
        amrex::ParallelFor(it.numParticles(),
                           [=] AMREX_GPU_DEVICE(long p) { data.m_rdata[PIdx::r][p] = 1.125_prt; });
    }
    auto& radiation = WarpX::GetInstance().GetRadiationTransport();
    auto const before = ParticleImpulseInventory(particles, {"ions"}, "rz_probe");
    auto const old_wall = radiation.particleCarryWallMomentum("ions", "rz_probe");
    ions.ApplyBoundaryConditions();
    ions.Redistribute();
    auto const after = ParticleImpulseInventory(particles, {"ions"}, "rz_probe");
    auto const wall = radiation.particleCarryWallMomentum("ions", "rz_probe");
    amrex::Real mass = 0;
    for (auto const& particle : ReadParticles(ions))
    {
        mass += particle[0] * ions.getMass();
    }
    amrex::ParallelDescriptor::ReduceRealSum(mass);
    for (int d = 0; d < 3; ++d)
    {
        AMREX_ALWAYS_ASSERT(std::abs(before[d] - after[d] - (wall[d] - old_wall[d])) <
                            2.e-12_rt * mass * 1.e-30_rt);
    }
    AMREX_ALWAYS_ASSERT(std::abs(before[3] - after[3]) < 2.e-12_rt * mass * 1.e-30_rt);
    amrex::GpuArray<amrex::Real, 4> expected{};
    for (auto const& particle : ReadParticles(ions))
    {
        auto const weight_mass = particle[0] * ions.getMass();
        auto const cosine = std::cos(particle[4]);
        auto const sine = std::sin(particle[4]);
        expected[0] += weight_mass * (particle[7] * cosine + particle[8] * sine);
        expected[1] += weight_mass * (-particle[7] * sine + particle[8] * cosine);
        expected[2] += weight_mass * particle[9];
        expected[3] += weight_mass * particle[10];
    }
    amrex::ParallelDescriptor::ReduceRealSum(expected.data(), 4);
    auto const local = ParticleImpulseInventory(particles, {"ions"}, "rz_probe", true);
    for (int d = 0; d < 4; ++d)
    {
        AMREX_ALWAYS_ASSERT(std::abs(local[d] - expected[d]) < 2.e-12_rt * mass * 1.e-30_rt);
    }
    std::string const checkpoint = "rz_carry_wall_checkpoint";
    ions.Checkpoint(checkpoint, "ions");
    radiation.WriteParticleCarryWallCheckpoint(checkpoint);
    amrex::ParallelDescriptor::Barrier();
    ions.clearParticles();
    ions.Restart(checkpoint, "ions");
    ions.Redistribute();
    radiation.ReadParticleCarryWallCheckpoint(checkpoint);
    auto const restored = ParticleImpulseInventory(particles, {"ions"}, "rz_probe");
    for (int d = 0; d < 4; ++d)
    {
        AMREX_ALWAYS_ASSERT(std::abs(restored[d] - after[d]) < 2.e-12_rt * mass * 1.e-30_rt);
    }
    auto const restored_wall = radiation.particleCarryWallMomentum("ions", "rz_probe");
    for (int d = 0; d < 3; ++d)
    {
        AMREX_ALWAYS_ASSERT(restored_wall[d] == wall[d]);
    }
}
} // namespace

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        using namespace warpx::radiation;
        auto& simulation = WarpX::GetInstance();
        auto& particles = simulation.GetPartContainer();
        auto& ions = particles.GetParticleContainerFromName("ions");
        bool runtime = false;
        amrex::ParmParse("test").query("radiation_runtime", runtime);
        bool moment_runtime = false;
        amrex::ParmParse("test").query("moment_runtime", moment_runtime);
        if (runtime)
        {
            simulation.InitData();
            simulation.HybridPICPrepareElectronStateForDiagnostics();
            auto const initial = NativeInventory(simulation);
            simulation.Evolve();
            auto const final = NativeInventory(simulation);
            amrex::Real total = 0, difference = 0;
            for (int d = 0; d < 4; ++d)
            {
                total += initial[d];
                difference += final[d] - initial[d];
            }
            amrex::Print() << "Native RZ radiation/material energy residual=" << difference / total
                           << (moment_runtime ? " moment loss=" : " photon loss=")
                           << initial[2] - final[2] << '\n';
            if (moment_runtime) {
                AMREX_ALWAYS_ASSERT(initial[2] > 0 && final[2] > 0);
                AMREX_ALWAYS_ASSERT(std::abs(final[0] - initial[0]) > 1.e-8_rt * initial[0]);
                if (amrex::ParallelDescriptor::IOProcessor()) {
                    std::ofstream inventory("native_inventory.txt");
                    inventory << std::setprecision(17);
                    for (auto const& state : {initial, final}) {
                        for (auto const value : state) { inventory << value << ' '; }
                        inventory << '\n';
                    }
                    AMREX_ALWAYS_ASSERT(inventory.good());
                }
            } else {
                AMREX_ALWAYS_ASSERT(initial[2] - final[2] > 0.5_rt * initial[2]);
            }
            AMREX_ALWAYS_ASSERT(std::abs(difference) < 1.e-10_rt * total);
            WarpX::Finalize();
            warpx::initialization::finalize_external_libraries();
            return 0;
        }
        RegisterParticleImpulseState(ions, "rz_probe");
        simulation.InitData();
        amrex::MultiFab impulse(ions.ParticleBoxArray(0), ions.ParticleDistributionMap(0), 3, 0);
        impulse.setVal(0);
        ParticleImpulseTransaction zero;
        AMREX_ALWAYS_ASSERT(zero.Stage(particles, {"ions"}, "rz_probe", impulse, true));
        auto const total_mass = zero.MaterialMass().sum(0);
        AMREX_ALWAYS_ASSERT(total_mass > 0);
        auto const unchanged = ReadParticles(ions);
        impulse.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
        ParticleImpulseTransaction invalid;
        AMREX_ALWAYS_ASSERT(!invalid.Stage(particles, {"ions"}, "rz_probe", impulse, true));
        AMREX_ALWAYS_ASSERT(ReadParticles(ions) == unchanged);

        // Repeated finite, non-collinear kicks exercise the rotating cylindrical
        // basis and the actual relativistic work, not merely a stationary force.
        for (int step = 0; step < 128; ++step)
        {
            std::array<amrex::Real, 3> const kick{step % 2 == 0 ? -13000._rt : 7000._rt,
                                                  step % 3 == 0 ? 9000._rt : -2000._rt, 3000._rt};
            for (int d = 0; d < 3; ++d)
            {
                amrex::MultiFab::Copy(impulse, zero.MaterialMass(), 0, d, 1, 0);
                impulse.mult(kick[d], d, 1);
            }
            auto const before = ReadParticles(ions);
            ParticleImpulseTransaction trial;
            AMREX_ALWAYS_ASSERT(trial.Stage(particles, {"ions"}, "rz_probe", impulse, true));
            AMREX_ALWAYS_ASSERT(ReadParticles(ions) == before);
            for (int d = 0; d < 3; ++d)
            {
                auto const represented =
                    trial.ActualImpulse().sum(d) + trial.MomentumCarryChange().sum(d);
                AMREX_ALWAYS_ASSERT(std::abs(represented / (total_mass * kick[d]) - 1) < 2.e-12_rt);
            }
            trial.Commit();
            auto const after = ReadParticles(ions);
            AMREX_ALWAYS_ASSERT(before.size() == after.size());
            amrex::Real work = 0;
            amrex::Real work_scale = 0;
            for (std::size_t p = 0; p < before.size(); ++p)
            {
                auto const cosine = std::cos(before[p][4]);
                auto const sine = std::sin(before[p][4]);
                std::array<amrex::Real, 3> const cartesian{
                    kick[0] * cosine - kick[1] * sine, kick[0] * sine + kick[1] * cosine, kick[2]};
                long double old_squared = 0;
                long double new_squared = 0;
                long double difference = 0;
                for (int d = 0; d < 3; ++d)
                {
                    long double const old_u = before[p][d + 1];
                    long double const new_u = after[p][d + 1];
                    AMREX_ALWAYS_ASSERT(std::abs(static_cast<amrex::Real>(new_u - old_u) -
                                                 cartesian[d]) < 2.e-9_rt);
                    old_squared += old_u * old_u;
                    new_squared += new_u * new_u;
                    difference += (new_u - old_u) * (new_u + old_u);
                }
                auto const denominator = std::sqrt(1 + old_squared / PhysConst::c2) +
                                         std::sqrt(1 + new_squared / PhysConst::c2);
                auto const energy = static_cast<amrex::Real>(before[p][0] * ions.getMass() *
                                                             difference / denominator);
                work += energy;
                work_scale += std::abs(energy);
            }
            amrex::ParallelDescriptor::ReduceRealSum(work);
            amrex::ParallelDescriptor::ReduceRealSum(work_scale);
            AMREX_ALWAYS_ASSERT(std::abs(work - trial.ActualWork().sum(0)) <
                                2.e-12_rt * work_scale);
            auto const accounted = trial.ActualWork().sum(0) + trial.EnergyCarryChange().sum(0) +
                                   trial.NumericalEnergyResidual().sum(0);
            AMREX_ALWAYS_ASSERT(std::abs(accounted - trial.RequestedWork().sum(0)) <
                                2.e-12_rt * work_scale);
        }
        CheckWalls(particles, ions);
        amrex::Print() << "RZ finite impulse and carry-wall transactions pass\n";
    }
    WarpX::Finalize();
    warpx::initialization::finalize_external_libraries();
}
