/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Diagnostics/FlushFormats/FlushFormatPlotfile.H"
#ifdef WARPX_USE_OPENPMD
#include "Diagnostics/FlushFormats/FlushFormatOpenPMD.H"
#endif
#include "Diagnostics/ParticleDiag/ParticleDiag.H"
#include "Fields.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Radiation/ParticleImpulse.H"
#include "Radiation/ParticleImpulseBoundary.H"
#include "Radiation/RadiationTransport.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>
#include <AMReX_VisMF.H>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace amrex::literals;

namespace
{
    constexpr amrex::Real speed = 1024;
    constexpr amrex::Real increment = speed * std::numeric_limits<amrex::Real>::epsilon() / 8;

    std::vector<amrex::ParticleReal>
    ParticleSnapshot (WarpXParticleContainer& ions)
    {
        std::vector<amrex::ParticleReal> snapshot;
        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
            for (int component = 0; component < ions.NumRealComps(); ++component) {
                auto const& values = iterator.GetStructOfArrays().GetRealData(component);
                auto const start = snapshot.size();
                snapshot.resize(start + iterator.numParticles());
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(),
                    values.begin() + iterator.numParticles(), snapshot.begin() + start);
            }
        }
        return snapshot;
    }

    bool SameSnapshot (std::vector<amrex::ParticleReal> const& a,
                       std::vector<amrex::ParticleReal> const& b)
    {
        return a.size() == b.size() && (a.empty() ||
            std::memcmp(a.data(), b.data(), a.size() * sizeof(amrex::ParticleReal)) == 0);
    }

    void CheckBirthCarry (WarpXParticleContainer& ions)
    {
        std::vector<int> components;
        for (auto const& path : warpx::radiation::RegisteredParticleImpulsePaths(ions)) {
            for (auto const* suffix : {"_ux", "_uy", "_uz", "_work"}) {
                components.push_back(ions.GetRealCompIndex("radiation_impulse_" + path + suffix));
            }
        }
        AMREX_ALWAYS_ASSERT(components.size() == 8);
        auto const initial_count = ions.TotalNumberOfParticles();
        AMREX_ALWAYS_ASSERT(initial_count > 0);
        std::map<std::pair<int, int>, amrex::Long> counts;
        for (auto& [key, tile] : ions.GetParticles(0)) {
            auto const count = static_cast<amrex::Long>(tile.numParticles());
            counts.emplace(key, count);
            // Check initial native births before deliberately filling retained
            // capacity. No allocator-zeroing assumption is allowed below.
            for (int component : components) {
                auto const& values = tile.GetStructOfArrays().GetRealData(component);
                amrex::Gpu::HostVector<amrex::ParticleReal> host(count);
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(), values.begin() + count,
                                 host.begin());
                for (auto value : host) { AMREX_ALWAYS_ASSERT(value == 0); }
            }
            tile.resize(2 * count);
            for (int component : components) {
                auto* const values = tile.GetStructOfArrays().GetRealData(component).data();
                auto const old_value = amrex::ParticleReal(17 + component);
                auto const poison = amrex::ParticleReal(91 + component);
                amrex::ParallelFor(2 * count, [=] AMREX_GPU_DEVICE(amrex::Long ip) {
                    values[ip] = ip < count ? old_value : poison;
                });
            }
            amrex::Gpu::streamSynchronize();
            tile.resize(count);
        }
        // Reuse the pre-filled capacity through the actual bulk-plasma path.
        // Existing particles must retain their accounts; only newborns start at zero.
        dynamic_cast<PhysicalParticleContainer&>(ions).AddParticles(0);
        AMREX_ALWAYS_ASSERT(ions.TotalNumberOfParticles() == 2 * initial_count);
        for (auto const& [key, tile] : ions.GetParticles(0)) {
            auto const old_count = counts.at(key);
            AMREX_ALWAYS_ASSERT(tile.numParticles() == 2 * old_count);
            for (int component : components) {
                auto const& values = tile.GetStructOfArrays().GetRealData(component);
                amrex::Gpu::HostVector<amrex::ParticleReal> host(2 * old_count);
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(),
                                 values.begin() + 2 * old_count, host.begin());
                for (amrex::Long ip = 0; ip < 2 * old_count; ++ip) {
                    auto const expected = ip < old_count ? amrex::ParticleReal(17 + component)
                                                         : 0._prt;
                    AMREX_ALWAYS_ASSERT(host[ip] == expected);
                }
            }
        }
        amrex::Print() << "Native births zero both carry owners and preserve existing accounts\n";
    }

    void CheckBoundaryCarry (WarpX& simulation, WarpXParticleContainer& ions)
    {
#if !defined(WARPX_DIM_1D_Z)
        amrex::Abort("The native carry-boundary fixture is 1D.");
#endif
        using namespace warpx::radiation;
        std::array<std::string, 2> const paths{"test", "boundary_second"};
        std::array<std::string, 4> const suffix{"ux", "uy", "uz", "work"};
        std::string restart;
        amrex::ParmParse("test").query("boundary_carry_restart", restart);
        if (!restart.empty()) {
            ions.clearParticles();
            ions.Restart(restart, "ions");
            simulation.GetRadiationTransport().ReadCheckpointData(restart);
            ions.Redistribute();
            AMREX_ALWAYS_ASSERT(ions.TotalNumberOfParticles() == 64);
            for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
                auto const data = iterator.GetParticleTile().getParticleTileData();
                auto* pending = iterator.GetStructOfArrays().GetRealData(
                    ions.GetRealCompIndex("radiation_impulse_test_uz")).data();
                amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                    auto p = WarpXParticleContainer::ParticleType(data, ip);
                    p.pos(0) = -0.125;
                    pending[ip] = -1.e-6;
                });
            }
            ions.ApplyBoundaryConditions();
            auto const expected = 8 * 1.e20 * ions.getMass() * 1.e-30;
            AMREX_ALWAYS_ASSERT(std::abs(simulation.GetRadiationTransport()
                .particleCarryWallMomentum("ions", "test")[2] - expected)
                < 1.e-12 * std::abs(expected));
            amrex::Print() << "Changed-rank compensated carry-wall continuation passed\n";
            return;
        }
        ParticleBoundaries boundaries;
        boundaries.SetAll(ParticleBoundaryType::Reflecting);
        boundaries.BuildReflectionModelParsers();
        for (bool reflect_all : {false, true}) {
            boundaries.Set_reflect_all_velocities(reflect_all);
            for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
                auto const data = iterator.GetParticleTile().getParticleTileData();
                amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                    auto p = WarpXParticleContainer::ParticleType(data, ip);
                    p.pos(0) = ip % 4 == 0 ? 0.5 : (ip % 4 == 2 ? 1.125 : -0.125);
                    data.m_rdata[PIdx::ux][ip] = 1;
                    data.m_rdata[PIdx::uy][ip] = 2;
                    data.m_rdata[PIdx::uz][ip] = ip % 4 == 3 ? 0 : 3;
                });
                for (int group = 0; group < 2; ++group) {
                    for (int d = 0; d < 4; ++d) {
                        auto* value = iterator.GetStructOfArrays().GetRealData(
                            ions.GetRealCompIndex("radiation_impulse_" + paths[group] + "_" + suffix[d])).data();
                        amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                            value[ip] = (group + 1) * (d + 1) * (d % 2 ? -1 : 1) * 1.e-30;
                        });
                    }
                }
            }
            std::array<amrex::GpuArray<amrex::Real, 4>, 2> before;
            for (int group = 0; group < 2; ++group) {
                before[group] = ParticleImpulseInventory(simulation.GetPartContainer(), {"ions"}, paths[group]);
            }
            std::vector<ParticleImpulseBoundaryTransfer> transfer;
            AMREX_ALWAYS_ASSERT(TryReflectParticleImpulseState(ions, boundaries, transfer));
            AMREX_ALWAYS_ASSERT(transfer.size() == 2);
            amrex::ReduceOps<amrex::ReduceOpMax> point_ops;
            amrex::ReduceData<amrex::Real> point_data(point_ops);
            using PointTuple = typename decltype(point_data)::Type;
            for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
                auto const data = iterator.GetParticleTile().getParticleTileData();
                point_ops.eval(iterator.numParticles(), point_data,
                    [=] AMREX_GPU_DEVICE(long ip) -> PointTuple {
                        auto p = WarpXParticleContainer::ParticleType(data, ip);
                        bool const hit = ip % 4 != 0;
                        auto const expected_position = ip % 4 == 0 ? 0.5
                            : (ip % 4 == 2 ? 0.875 : 0.125);
                        amrex::Real error = std::abs(p.pos(0) - expected_position);
                        for (int d = 0; d < 3; ++d) {
                            amrex::Real initial = d + 1;
                            if (d == 2 && ip % 4 == 3) { initial = 0; }
                            auto const expected = hit && (reflect_all || d == 2) ? -initial : initial;
                            error = amrex::max(error,
                                std::abs(data.m_rdata[PIdx::ux + d][ip] - expected));
                        }
                        return {error};
                    });
                for (int group = 0; group < 2; ++group) {
                    for (int d = 0; d < 4; ++d) {
                        auto const* carry = iterator.GetStructOfArrays().GetRealData(
                            ions.GetRealCompIndex("radiation_impulse_" + paths[group]
                                + "_" + suffix[d])).data();
                        auto const initial = (group + 1) * (d + 1) * (d % 2 ? -1 : 1) * 1.e-30;
                        point_ops.eval(iterator.numParticles(), point_data,
                            [=] AMREX_GPU_DEVICE(long ip) -> PointTuple {
                                bool const flip = ip % 4 != 0 && d < 3 && (reflect_all || d == 2);
                                auto const expected = flip ? -initial : initial;
                                return {amrex::Real(carry[ip] != expected)};
                            });
                    }
                }
            }
            auto point_error = amrex::get<0>(point_data.value());
            amrex::ParallelDescriptor::ReduceRealMax(point_error);
            AMREX_ALWAYS_ASSERT(point_error == 0);
            for (int group = 0; group < 2; ++group) {
                AMREX_ALWAYS_ASSERT(transfer[group].path == paths[group]);
                auto after = ParticleImpulseInventory(simulation.GetPartContainer(), {"ions"}, paths[group]);
                for (int d = 0; d < 3; ++d) {
                    auto const expected = (reflect_all || d == 2) ? 1.5 * before[group][d] : 0;
                    auto const scale = std::abs(before[group][d]);
                    AMREX_ALWAYS_ASSERT(std::abs(transfer[group].momentum[d] - expected) < 1.e-12 * scale);
                    AMREX_ALWAYS_ASSERT(std::abs(after[d] + transfer[group].momentum[d]
                        - before[group][d]) < 1.e-12 * scale);
                }
                AMREX_ALWAYS_ASSERT(std::abs(after[3] - before[group][3])
                    < 1.e-12 * std::abs(before[group][3]));
            }
            // Communication must migrate the already-reflected attributes.
            ions.Redistribute();
            for (int group = 0; group < 2; ++group) {
                auto const after = ParticleImpulseInventory(simulation.GetPartContainer(), {"ions"}, paths[group]);
                for (int d = 0; d < 3; ++d) {
                    AMREX_ALWAYS_ASSERT(std::abs(after[d] + transfer[group].momentum[d]
                        - before[group][d]) < 1.e-12 * std::abs(before[group][d]));
                }
            }
        }
        int const last_grid = ions.ParticleBoxArray(0).size() - 1;
        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
            auto const data = iterator.GetParticleTile().getParticleTileData();
            bool const poison = iterator.index() == last_grid;
            auto* work = iterator.GetStructOfArrays().GetRealData(
                ions.GetRealCompIndex("radiation_impulse_boundary_second_work")).data();
            amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                auto p = WarpXParticleContainer::ParticleType(data, ip);
                p.pos(0) = -0.125;
                if (poison && ip == 0) { work[ip] = std::numeric_limits<amrex::Real>::quiet_NaN(); }
            });
        }
        auto const before = ParticleSnapshot(ions);
        std::vector<ParticleImpulseBoundaryTransfer> sentinel{{"sentinel", {7, 8, 9}}};
        AMREX_ALWAYS_ASSERT(!TryReflectParticleImpulseState(ions, boundaries, sentinel));
        AMREX_ALWAYS_ASSERT(SameSnapshot(before, ParticleSnapshot(ions)));
        AMREX_ALWAYS_ASSERT(sentinel.size() == 1 && sentinel[0].path == "sentinel"
            && sentinel[0].momentum[0] == 7 && sentinel[0].momentum[1] == 8
            && sentinel[0].momentum[2] == 9);
        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
            auto const data = iterator.GetParticleTile().getParticleTileData();
            auto* work = iterator.GetStructOfArrays().GetRealData(
                ions.GetRealCompIndex("radiation_impulse_boundary_second_work")).data();
            amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                auto p = WarpXParticleContainer::ParticleType(data, ip);
                p.pos(0) = -0.125;
                work[ip] = -8.e-30;
            });
        }
        auto const clean = ParticleSnapshot(ions);
        for (auto type : {ParticleBoundaryType::Open, ParticleBoundaryType::Thermal,
                          ParticleBoundaryType::Periodic}) {
            boundaries.SetAll(type);
            AMREX_ALWAYS_ASSERT(!TryReflectParticleImpulseState(ions, boundaries, sentinel));
            AMREX_ALWAYS_ASSERT(SameSnapshot(clean, ParticleSnapshot(ions)));
            AMREX_ALWAYS_ASSERT(sentinel.size() == 1 && sentinel[0].path == "sentinel"
                && sentinel[0].momentum[0] == 7 && sentinel[0].momentum[1] == 8
                && sentinel[0].momentum[2] == 9);
        }
        boundaries.SetAll(ParticleBoundaryType::Reflecting);
        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
            auto const data = iterator.GetParticleTile().getParticleTileData();
            auto* pending = iterator.GetStructOfArrays().GetRealData(
                ions.GetRealCompIndex("radiation_impulse_boundary_second_uz")).data();
            amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                data.m_rdata[PIdx::w][ip] = 1.e126;
                pending[ip] = 1.e208;
            });
        }
        auto const overflow = ParticleSnapshot(ions);
        // Individual transfers are finite; the second path's sum overflows.
        AMREX_ALWAYS_ASSERT(!TryReflectParticleImpulseState(ions, boundaries, sentinel));
        AMREX_ALWAYS_ASSERT(SameSnapshot(overflow, ParticleSnapshot(ions)));
        AMREX_ALWAYS_ASSERT(sentinel.size() == 1 && sentinel[0].path == "sentinel"
            && sentinel[0].momentum[0] == 7 && sentinel[0].momentum[1] == 8
            && sentinel[0].momentum[2] == 9);
        // Exercise the actual native dispatcher and its persistent ledger.
        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
            auto const data = iterator.GetParticleTile().getParticleTileData();
            amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                auto p = WarpXParticleContainer::ParticleType(data, ip);
                p.pos(0) = -0.125;
                data.m_rdata[PIdx::w][ip] = 1.e20 / 64;
            });
            for (int group = 0; group < 2; ++group) {
                for (int d = 0; d < 4; ++d) {
                    auto* value = iterator.GetStructOfArrays().GetRealData(
                        ions.GetRealCompIndex("radiation_impulse_" + paths[group]
                            + "_" + suffix[d])).data();
                    amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                        value[ip] = (group + 1) * (d + 1) * (d % 2 ? -1 : 1) * 1.e-30;
                    });
                }
            }
        }
        auto& radiation = simulation.GetRadiationTransport();
        std::array<amrex::GpuArray<amrex::Real, 4>, 2> native_before;
        for (int group = 0; group < 2; ++group) {
            native_before[group] = ParticleImpulseInventory(simulation.GetPartContainer(),
                {"ions"}, paths[group]);
        }
        ions.ApplyBoundaryConditions();
        ions.Redistribute();
        std::array<amrex::GpuArray<amrex::Real, 3>, 2> saved;
        for (int group = 0; group < 2; ++group) {
            saved[group] = radiation.particleCarryWallMomentum("ions", paths[group]);
            AMREX_ALWAYS_ASSERT(saved[group][0] == 0 && saved[group][1] == 0);
            AMREX_ALWAYS_ASSERT(std::abs(saved[group][2] - 2 * native_before[group][2])
                < 1.e-12 * std::abs(native_before[group][2]));
        }
        std::string const checkpoint = "native_carry_wall_checkpoint";
        ions.Checkpoint(checkpoint, "ions");
        if (amrex::ParallelDescriptor::IOProcessor()) { radiation.WriteCheckpointData(checkpoint); }
        amrex::ParallelDescriptor::Barrier();
        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
            auto const data = iterator.GetParticleTile().getParticleTileData();
            amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                auto p = WarpXParticleContainer::ParticleType(data, ip);
                p.pos(0) = -0.125;
            });
        }
        ions.ApplyBoundaryConditions();
        AMREX_ALWAYS_ASSERT(std::abs(radiation.particleCarryWallMomentum("ions", paths[0])[2])
            < 1.e-12 * std::abs(saved[0][2]));
        ions.clearParticles(); // The low-level reader appends, unlike fresh WarpX startup.
        ions.Restart(checkpoint, "ions");
        AMREX_ALWAYS_ASSERT(ions.TotalNumberOfParticles() == 64);
        radiation.ReadCheckpointData(checkpoint);
        ions.Redistribute();
        for (int group = 0; group < 2; ++group) {
            auto const restored = radiation.particleCarryWallMomentum("ions", paths[group]);
            for (int d = 0; d < 3; ++d) { AMREX_ALWAYS_ASSERT(restored[d] == saved[group][d]); }
            auto const inventory = ParticleImpulseInventory(simulation.GetPartContainer(),
                {"ions"}, paths[group]);
            AMREX_ALWAYS_ASSERT(std::abs(inventory[2] + restored[2] - native_before[group][2])
                < 1.e-12 * std::abs(native_before[group][2]));
        }
        int sequence = 0;
        for (auto pending_value : {1.e-6, 1.e-30, -1.e-6}) {
            for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
                auto const data = iterator.GetParticleTile().getParticleTileData();
                auto* pending = iterator.GetStructOfArrays().GetRealData(
                    ions.GetRealCompIndex("radiation_impulse_test_uz")).data();
                amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                    auto p = WarpXParticleContainer::ParticleType(data, ip);
                    p.pos(0) = -0.125;
                    pending[ip] = pending_value;
                });
            }
            ions.ApplyBoundaryConditions();
            if (sequence++ == 1) {
                std::string const compensated = "compensated_carry_wall_checkpoint";
                ions.Checkpoint(compensated, "ions");
                if (amrex::ParallelDescriptor::IOProcessor()) {
                    radiation.WriteCheckpointData(compensated);
                }
                amrex::ParallelDescriptor::Barrier();
                ions.clearParticles();
                ions.Restart(compensated, "ions");
                AMREX_ALWAYS_ASSERT(ions.TotalNumberOfParticles() == 64);
                radiation.ReadCheckpointData(compensated);
                ions.Redistribute();
            }
        }
        auto const tiny_transfer = 2 * 1.e20 * ions.getMass() * 1.e-30;
        AMREX_ALWAYS_ASSERT(std::abs(radiation.particleCarryWallMomentum("ions", paths[0])[2]
            - saved[0][2] - tiny_transfer) < 1.e-12 * std::abs(tiny_transfer));
        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
            auto const data = iterator.GetParticleTile().getParticleTileData();
            amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                auto p = WarpXParticleContainer::ParticleType(data, ip);
                p.pos(0) = -0.125;
            });
        }
        auto const vetoed = ParticleSnapshot(ions);
        bool consulted = false;
        AMREX_ALWAYS_ASSERT(!TryReflectParticleImpulseState(ions, boundaries, sentinel,
            [&] (auto const&) { consulted = true; return false; }));
        AMREX_ALWAYS_ASSERT(consulted && SameSnapshot(vetoed, ParticleSnapshot(ions)));
        AMREX_ALWAYS_ASSERT(sentinel.size() == 1 && sentinel[0].path == "sentinel"
            && sentinel[0].momentum[0] == 7 && sentinel[0].momentum[1] == 8
            && sentinel[0].momentum[2] == 9);
        amrex::Print() << "Native multi-path carry reflection, migration and rejection passed\n";
    }

    void CheckPlotImmutability (WarpX& simulation, WarpXParticleContainer& ions)
    {
        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
            auto& soa = iterator.GetStructOfArrays();
            auto* ux = soa.GetRealData(PIdx::ux).data();
            auto* uy = soa.GetRealData(PIdx::uy).data();
            auto* uz = soa.GetRealData(PIdx::uz).data();
            amrex::GpuArray<amrex::ParticleReal*, 4> carry{};
            std::array<std::string, 4> const names{"ux", "uy", "uz", "work"};
            for (int d = 0; d < 4; ++d) {
                carry[d] = soa.GetRealData(ions.GetRealCompIndex(
                    "radiation_impulse_test_" + names[d])).data();
            }
            amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                ux[ip] = 12345.6789123 + ip * 0.314159;
                uy[ip] = -87654.32198 - ip * 0.192837;
                uz[ip] = (ip % 2 ? -1 : 1) * (90000.1234 + ip * 0.271828);
                for (int d = 0; d < 4; ++d) {
                    carry[d][ip] = (d % 2 ? -1 : 1) * (d + 1) * (ip + 1) * 1.e-30;
                }
            });
        }
        auto const before = ParticleSnapshot(ions);
        amrex::Vector<ParticleDiag> diagnostics;
        diagnostics.emplace_back("readonly", "ions", &ions);
        diagnostics[0].m_particle_filter_parser =
            std::make_unique<amrex::Parser>("(uz>0)*(uz<0.001)");
        diagnostics[0].m_particle_filter_parser->registerVariables(
            {"t", "x", "y", "z", "ux", "uy", "uz"});
        amrex::Vector<amrex::MultiFab> fields(1);
        fields[0].define(ions.ParticleBoxArray(0), ions.ParticleDistributionMap(0), 1, 0);
        fields[0].setVal(0);
        amrex::Vector<amrex::Geometry> geometry{simulation.Geom(0)};
        bool openpmd = false;
        amrex::ParmParse("test").query("openpmd", openpmd);
        for (int mode = 0; mode < 3; ++mode) {
            diagnostics[0].m_do_uniform_filter = mode == 1;
            diagnostics[0].m_uniform_stride = 2;
            diagnostics[0].m_do_parser_filter = mode == 2;
            std::unique_ptr<FlushFormat> writer;
            if (openpmd) {
#ifdef WARPX_USE_OPENPMD
                writer = std::make_unique<FlushFormatOpenPMD>("readonly");
#else
                amrex::Abort("The openPMD immutability test requires openPMD support.");
#endif
            } else {
                writer = std::make_unique<FlushFormatPlotfile>();
            }
            writer->WriteToFile({"rho"}, fields, geometry, {0}, 0, diagnostics, 1,
                "readonly_" + std::to_string(mode) + "_", 6, false, false, 0);
            AMREX_ALWAYS_ASSERT(SameSnapshot(ParticleSnapshot(ions), before));
        }
        amrex::Print() << "Diagnostic output leaves all live particle real attributes unchanged\n";
    }

    AMREX_GPU_HOST_DEVICE amrex::Real
    CardinalSpline (amrex::Real x, int order)
    {
        if (std::abs(x) >= (order + 1) / 2._rt) { return 0; }
        amrex::Real sum = 0;
        int choose = 1;
        for (int k = 0; k <= order + 1; ++k) {
            auto const value = amrex::max(0._rt, x + (order + 1) / 2._rt - k);
            amrex::Real power = 1;
            for (int p = 0; p < order; ++p) { power *= value; }
            sum += (k % 2 ? -choose : choose) * power;
            choose = choose * (order + 1 - k) / (k + 1);
        }
        for (int p = 2; p <= order; ++p) { sum /= p; }
        return sum;
    }

    // Independent reference: explicitly deposit to the two charge nodes, then
    // average the two corners of each target cell, including periodic wrapping.
    AMREX_GPU_HOST_DEVICE amrex::Real
    ReferenceWeight (amrex::GpuArray<int, 3> cell, amrex::GpuArray<int, 3> base,
                     amrex::GpuArray<int, 3> length, amrex::Real fraction, int order = 1,
                     bool reflecting = false)
    {
        amrex::Real weight = 1;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            amrex::Real one = 0;
            for (int node = -3; node <= 4; ++node) {
                if (reflecting) {
                    // Independent nodal-charge reference: mirror nodes, double
                    // physical endpoint density, then average physical corners.
                    int index = base[d] + node;
                    if (index < 0) { index = -index; }
                    else if (index > length[d]) { index = 2 * length[d] - index; }
                    if (index == cell[d] || index == cell[d] + 1) {
                        amrex::Real const multiplicity =
                            index == 0 || index == length[d] ? 2 : 1;
                        one += multiplicity * CardinalSpline(node - fraction, order) / 2;
                    }
                    continue;
                }
                for (int corner = 0; corner < 2; ++corner) {
                    if (((base[d] + node) % length[d] + length[d]) % length[d] ==
                        (cell[d] + corner) % length[d]) {
                        one += CardinalSpline(node - fraction, order) / 2;
                    }
                }
            }
            weight *= one;
        }
        return weight;
    }

    void
    CheckShapeAssignment (WarpX& simulation, WarpXParticleContainer& ions,
                          bool reflecting = false, int component = 2)
    {
        using namespace warpx::radiation;
        auto const& geometry = simulation.Geom(0);
        auto const dx = geometry.CellSizeArray();
        auto const plo = geometry.ProbLoArray();
        amrex::GpuArray<int, 3> length{1, 1, 1};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            length[d] = geometry.Domain().length(d);
        }
        amrex::Real const total_mass = 1.e20_rt * ions.getMass();
        int const order = WarpX::nox;
        AMREX_ALWAYS_ASSERT(component >= 0 && component < 3);
        if (component != 2) {
            for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
                auto const data = iterator.GetParticleTile().getParticleTileData();
                amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                    data.m_rdata[PIdx::uz][ip] = 0;
                    data.m_rdata[PIdx::ux + component][ip] = speed;
                });
            }
        }
        amrex::Vector<amrex::Real> fractions{0._rt, 0.125_rt, 0.173_rt,
                                           0.5_rt, 0.875_rt, 0.9999_rt};
        if (reflecting) { fractions.push_back(1); }
        amrex::MultiFab request(ions.ParticleBoxArray(0), ions.ParticleDistributionMap(0), 3, 0);
        amrex::MultiFab native_charge;
        amrex::Real const mass_to_charge = ions.getMass() / ions.getCharge();
        if (reflecting) {
            native_charge.define(amrex::convert(ions.ParticleBoxArray(0),
                amrex::IntVect::TheNodeVector()), ions.ParticleDistributionMap(0), 1, order + 2);
        }
        int cases = 0;
        for (int location : {0, 7, 8, 15}) {
            amrex::GpuArray<int, 3> base{};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                base[d] = location % length[d];
            }
            for (amrex::Real fraction : fractions) {
                // Collapse the cloud to leave opacity-bearing neighbor cells
                // without centers and, with MPI, an entirely empty rank.
                for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
                    auto const data = iterator.GetParticleTile().getParticleTileData();
                    amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                        auto p = WarpXParticleContainer::ParticleType(data, ip);
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            p.pos(d) = plo[d] + (base[d] + fraction) * dx[d];
                        }
                    });
                }
                // The exact upper face is tested before redistribution, with
                // particles still owned by their preceding last-cell tile.
                if (!(reflecting && base[0] == length[0] - 1 && fraction == 1)) {
                    ions.Redistribute();
                }
                if (reflecting) {
                    ions.DepositCharge(&native_charge, 0, false, true, true);
                }
                for (amrex::Real amplitude : {32._rt, increment}) {
                    for (amrex::Real bias : {0._rt, 0.25_rt}) {
                        request.setVal(0);
                        for (amrex::MFIter iterator(request); iterator.isValid(); ++iterator) {
                            auto const field = request.array(iterator);
                            amrex::ParallelFor(iterator.validbox(), [=] AMREX_GPU_DEVICE(
                                                                        int i, int j, int k) {
                                auto const w = ReferenceWeight({i, j, k}, base, length, fraction,
                                                               order, reflecting);
                                // Opposing neighboring forces, including exact cancellation.
                                field(i, j, k, component) =
                                    total_mass * w * amplitude * ((i % 2 ? -1 : 1) + bias);
                            });
                        }
                        long double expected_increment = 0;
                        for (int k = 0; k < length[2]; ++k) {
                            for (int j = 0; j < length[1]; ++j) {
                                for (int i = 0; i < length[0]; ++i) {
                                    expected_increment +=
                                        ReferenceWeight({i, j, k}, base, length, fraction,
                                                        order, reflecting) *
                                        amplitude * ((i % 2 ? -1 : 1) + bias);
                                }
                            }
                        }
                        if (!reflecting) {
                            ParticleImpulseTransaction nearest;
                            AMREX_ALWAYS_ASSERT(!nearest.Stage(
                                simulation.GetPartContainer(), {"ions"}, "test", request));
                        }
                        ParticleImpulseTransaction shaped;
                        AMREX_ALWAYS_ASSERT(
                            shaped.Stage(simulation.GetPartContainer(), {"ions"}, "test", request,
                                         true, reflecting
                                             ? ParticleImpulseAssignment::ReflectingNodalCellAverage
                                             : ParticleImpulseAssignment::NativeNodalCellAverage));
                        amrex::ReduceOps<amrex::ReduceOpMax> mass_ops;
                        amrex::ReduceData<amrex::Real> mass_data(mass_ops);
                        using MassTuple = typename decltype(mass_data)::Type;
                        for (amrex::MFIter iterator(request); iterator.isValid(); ++iterator) {
                            auto const actual_mass = shaped.MaterialMass().const_array(iterator);
                            auto const native = reflecting ? native_charge.const_array(iterator)
                                : amrex::Array4<amrex::Real const>{};
                            mass_ops.eval(iterator.validbox(), mass_data,
                                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> MassTuple {
                                    auto const expected = ReferenceWeight(
                                        {i, j, k}, base, length, fraction, order, reflecting);
                                    auto error = std::abs(
                                        actual_mass(i, j, k) / total_mass - expected);
                                    if (reflecting) {
                                        auto const native_mass = 0.5_rt * dx[0] * mass_to_charge
                                            * (native(i, j, k) + native(i + 1, j, k));
                                        error = amrex::max(error,
                                            std::abs(actual_mass(i, j, k) - native_mass)
                                                / total_mass);
                                    }
                                    return {error};
                                });
                        }
                        auto mass_error = amrex::get<0>(mass_data.value());
                        amrex::ParallelDescriptor::ReduceRealMax(mass_error);
                        AMREX_ALWAYS_ASSERT(mass_error < 1.e-12_rt);
                        AMREX_ALWAYS_ASSERT(
                            std::abs(shaped.MaterialMass().sum(0) / total_mass - 1) < 1.e-12_rt);
                        auto const received =
                            shaped.ActualImpulse().sum(component) +
                            shaped.MomentumCarryChange().sum(component);
                        AMREX_ALWAYS_ASSERT(std::abs(received / total_mass - expected_increment) <
                                            1.e-12_rt * amplitude);
                        auto const new_speed = speed + expected_increment;
                        auto const denominator =
                            std::sqrt(1.L + speed * speed / PhysConst::c2) +
                            std::sqrt(1.L + new_speed * new_speed / PhysConst::c2);
                        auto const expected_work =
                            (2 * speed + expected_increment) * expected_increment / denominator;
                        auto const source_work = shaped.RequestedWork().sum(0);
                        auto const particle_work = shaped.ActualWork().sum(0) +
                                                   shaped.EnergyCarryChange().sum(0) +
                                                   shaped.NumericalEnergyResidual().sum(0);
                        AMREX_ALWAYS_ASSERT(std::abs(source_work / total_mass - expected_work) <
                                            1.e-12_rt * speed * amplitude);
                        AMREX_ALWAYS_ASSERT(std::abs(source_work - particle_work -
                                                     shaped.WorkPartitionResidual().sum(0)) <
                                            1.e-12_rt * total_mass * speed * amplitude);
                        // Staging is private; committing must apply the gathered
                        // total exactly once to each real particle, including carry.
                        shaped.Commit();
                        amrex::ReduceOps<amrex::ReduceOpMax> ops;
                        amrex::ReduceData<amrex::Real> data(ops);
                        using Tuple = typename decltype(data)::Type;
                        auto const expected = static_cast<amrex::Real>(expected_increment);
                        std::array<std::string, 3> const suffix{"ux", "uy", "uz"};
                        auto const carry_index = ions.GetRealCompIndex(
                            "radiation_impulse_test_" + suffix[component]);
                        auto const energy_index =
                            ions.GetRealCompIndex("radiation_impulse_test_work");
                        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
                            auto const values = iterator.GetParticleTile().getParticleTileData();
                            auto* carry =
                                iterator.GetStructOfArrays().GetRealData(carry_index).dataPtr();
                            auto* energy =
                                iterator.GetStructOfArrays().GetRealData(energy_index).dataPtr();
                            ops.eval(iterator.numParticles(), data,
                                     [=] AMREX_GPU_DEVICE(long ip) -> Tuple {
                                         auto const change =
                                             values.m_rdata[PIdx::ux + component][ip] - speed;
                                         return {std::abs(change + carry[ip] - expected) / amplitude};
                                     });
                            amrex::ParallelFor(iterator.numParticles(),
                                               [=] AMREX_GPU_DEVICE(long ip) {
                                                   values.m_rdata[PIdx::ux + component][ip] = speed;
                                                   carry[ip] = 0;
                                                   energy[ip] = 0;
                                               });
                        }
                        auto error = amrex::get<0>(data.value());
                        amrex::ParallelDescriptor::ReduceRealMax(error);
                        AMREX_ALWAYS_ASSERT(error < 1.e-12_rt);
                        ++cases;
                    }
                }
            }
        }
        if (reflecting) {
            for (auto const position : {plo[0] - dx[0], geometry.ProbHi(0) + dx[0],
                                         std::numeric_limits<amrex::Real>::quiet_NaN()}) {
                for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
                    auto const data = iterator.GetParticleTile().getParticleTileData();
                    amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                        auto p = WarpXParticleContainer::ParticleType(data, ip);
                        p.pos(0) = position;
                    });
                }
                request.setVal(0);
                auto const before = ParticleSnapshot(ions);
                ParticleImpulseTransaction rejected;
                AMREX_ALWAYS_ASSERT(!rejected.Stage(simulation.GetPartContainer(), {"ions"},
                    "test", request, true, ParticleImpulseAssignment::ReflectingNodalCellAverage));
                AMREX_ALWAYS_ASSERT(SameSnapshot(ParticleSnapshot(ions), before));
            }
            auto const upper = geometry.ProbHi(0);
            for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
                auto const data = iterator.GetParticleTile().getParticleTileData();
                amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                    auto p = WarpXParticleContainer::ParticleType(data, ip);
                    p.pos(0) = upper;
                });
            }
        }
        if (component != 2) {
            // Restore the canonical fixture orientation after the tangential
            // candidate checks, for the existing final owner-state assertion.
            for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
                auto const data = iterator.GetParticleTile().getParticleTileData();
                amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                    data.m_rdata[PIdx::ux + component][ip] = 0;
                    data.m_rdata[PIdx::uz][ip] = speed;
                });
            }
        }
        amrex::Print() << cases << " overlapping-cloud assignment cases passed.\n";
    }

    amrex::Real
    CheckOwners (WarpXParticleContainer& ions, bool moved, bool kicked, bool realized = false,
                 bool shaped = false)
    {
        int const nz = WarpX::GetInstance().Geom(0).Domain().length(AMREX_SPACEDIM - 1);
        int const carry_index = ions.GetRealCompIndex("radiation_impulse_test_uz");
        int const work_index = ions.GetRealCompIndex("radiation_impulse_test_work");
        amrex::ReduceOps<amrex::ReduceOpMax> ops;
        amrex::ReduceData<amrex::Real> data(ops);
        using Tuple = typename decltype(data)::Type;
        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator)
        {
            auto const position = GetParticlePosition<PIdx>(iterator);
            auto const* carry = iterator.GetStructOfArrays().GetRealData(carry_index).dataPtr();
            auto const* work = iterator.GetStructOfArrays().GetRealData(work_index).dataPtr();
            auto const* uz = iterator.GetStructOfArrays().GetRealData(PIdx::uz).dataPtr();
            ops.eval(iterator.numParticles(), data, [=] AMREX_GPU_DEVICE (long ip) -> Tuple
            {
                amrex::ParticleReal x, y, z;
                position(ip, x, y, z);
                amrex::ignore_unused(x, y);
                if (shaped)
                {
                    auto const old_z = moved ? (z < 0.5_rt ? z + 0.5_rt : z - 0.5_rt) : z;
                    int const base = static_cast<int>(std::floor(old_z * nz));
                    auto const fraction = old_z * nz - base;
                    amrex::Real overlap = 0;
                    for (int node = 0; node < 2; ++node)
                    {
                        for (int corner = 0; corner < 2; ++corner)
                        {
                            int const cell = (base + node - corner + nz) % nz;
                            if (cell < nz / 2)
                            {
                                overlap += (node == 0 ? 1 - fraction : fraction) / 2;
                            }
                        }
                    }
                    auto const requested = kicked ? (realized ? 9 : 1) * increment * overlap : 0;
                    auto const actual = uz[ip] - speed;
                    auto const gamma0 = std::sqrt(1 + speed * speed / PhysConst::c2);
                    auto const gamma1 = std::sqrt(1 + uz[ip] * uz[ip] / PhysConst::c2);
                    auto const kinetic = actual * (uz[ip] + speed) / (gamma0 + gamma1);
                    auto const requested_work = requested * (2 * speed + requested)
                        / (gamma0 + std::sqrt(1 + (speed + requested) * (speed + requested)
                            / PhysConst::c2));
                    return {amrex::max(std::abs(actual + carry[ip] - requested) / increment,
                        std::abs(kinetic + work[ip] - requested_work) / (speed * increment))};
                }
                bool const owns = kicked && (moved ? z >= 0.5_rt : z < 0.5_rt);
                auto const expected = owns ? increment : 0;
                auto const gamma = std::sqrt(1 + speed * speed / PhysConst::c2);
                auto const expected_work = owns ? speed * increment / gamma : 0;
                auto const expected_velocity = speed + (owns && realized ? 8 * increment : 0);
                auto const error = amrex::max(std::abs(carry[ip] - expected) / increment,
                    std::abs(work[ip] - expected_work) / (speed * increment));
                return {amrex::max(error, std::abs(uz[ip] - expected_velocity) / increment)};
            });
        }
        auto error = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceRealMax(error);
        return amrex::max(error, 0._rt);
    }
}

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        using namespace warpx::radiation;
        std::string checkpoint, restart;
        amrex::ParmParse options("test");
        options.query("checkpoint", checkpoint);
        options.query("restart", restart);
        auto& simulation = WarpX::GetInstance();
        auto& particles = simulation.GetPartContainer();
        auto& ions = particles.GetParticleContainerFromName("ions");
        bool hybrid_restore_probe = false;
        bool boundary_carry = false;
        bool birth_carry = false;
        options.query("boundary_carry", boundary_carry);
        options.query("birth_carry", birth_carry);
        options.query("hybrid_restore_probe", hybrid_restore_probe);
        if (!hybrid_restore_probe) {
            RegisterParticleImpulseState(ions, "test");
            if (boundary_carry) { RegisterParticleImpulseState(ions, "boundary_second"); }
            if (birth_carry) { RegisterParticleImpulseState(ions, "birth_second"); }
        }
        simulation.InitData();
        if (birth_carry) {
            CheckBirthCarry(ions);
            WarpX::Finalize();
            warpx::initialization::finalize_external_libraries();
            return 0;
        }
        if (boundary_carry) {
            CheckBoundaryCarry(simulation, ions);
            WarpX::Finalize();
            warpx::initialization::finalize_external_libraries();
            return 0;
        }
        if (hybrid_restore_probe) {
            // Qualification probe of the actual restart bootstrap, without a
            // subsequent particle push. Do not add carry attributes to an
            // existing native-material checkpoint in this diagnostic mode.
            auto const before = ParticleSnapshot(ions);
            simulation.HybridPICInitializeRhoJandB();
            AMREX_ALWAYS_ASSERT(SameSnapshot(ParticleSnapshot(ions), before));
            using warpx::fields::FieldType;
            amrex::VisMF::Write(*simulation.m_fields.get(FieldType::rho_fp, 0), "probe_rho");
            amrex::VisMF::Write(*simulation.m_fields.get(
                FieldType::hybrid_electron_temperature_fp, 0), "probe_temperature");
            for (int d = 0; d < 3; ++d) {
                amrex::VisMF::Write(*simulation.m_fields.get(
                    FieldType::current_fp, ablastr::fields::Direction{d}, 0),
                    "probe_current_" + std::to_string(d));
            }
            WarpX::Finalize();
            warpx::initialization::finalize_external_libraries();
            return 0;
        }
        bool plot_immutable = false;
        options.query("plot_immutable", plot_immutable);
        if (plot_immutable) {
            CheckPlotImmutability(simulation, ions);
            WarpX::Finalize();
            warpx::initialization::finalize_external_libraries();
            return 0;
        }
        bool shape_test = false;
        options.query("shape", shape_test);
        if (shape_test)
        {
            bool reflecting = false;
            int component = 2;
            options.query("reflecting", reflecting);
            options.query("component", component);
            CheckShapeAssignment(simulation, ions, reflecting, component);
            AMREX_ALWAYS_ASSERT(CheckOwners(ions, false, false) < 1.e-12_rt);
            WarpX::Finalize();
            warpx::initialization::finalize_external_libraries();
            return 0;
        }
        bool shape_ownership = false;
        options.query("shape_ownership", shape_ownership);
        auto const assignment = shape_ownership ? ParticleImpulseAssignment::LinearNodalCellAverage
            : ParticleImpulseAssignment::NearestCell;
        auto check = [&] (bool moved, bool kicked, bool realized = false) {
            return CheckOwners(ions, moved, kicked, realized, shape_ownership);
        };
        auto stage = [&] (ParticleImpulseTransaction& trial, amrex::MultiFab const& impulse,
                          bool velocity = false) {
            return trial.Stage(particles, {"ions"}, "test", impulse, velocity, assignment);
        };
        auto const expected_particles = 4 * simulation.Geom(0).Domain().numPts();
        AMREX_ALWAYS_ASSERT(ions.TotalNumberOfParticles() == expected_particles);
        AMREX_ALWAYS_ASSERT(check(false, false) < 1.e-12_rt);
        auto const& geometry = simulation.Geom(0);
        auto const dx = geometry.CellSizeArray();
        auto const plo = geometry.ProbLoArray();
        amrex::Real volume = 1;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { volume *= dx[d]; }
        amrex::Real const mass = 1.e20_rt * ions.getMass() * volume;
        amrex::MultiFab request(ions.ParticleBoxArray(0), ions.ParticleDistributionMap(0), 3, 0);
        request.setVal(0);
        for (amrex::MFIter iterator(request); iterator.isValid(); ++iterator)
        {
            auto const field = request.array(iterator);
            amrex::ParallelFor(iterator.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
#if AMREX_SPACEDIM == 1
                amrex::Real const z = plo[0] + (i + 0.5_rt) * dx[0];
#else
                amrex::Real const z = plo[1] + (j + 0.5_rt) * dx[1];
#endif
                field(i, j, k, 2) = z < 0.5_rt ? mass * increment : 0;
            });
        }
        if (restart.empty())
        {
            // A discarded candidate must leave both momentum and work state unchanged.
            {
                ParticleImpulseTransaction discarded;
                AMREX_ALWAYS_ASSERT(stage(discarded, request, true));
                AMREX_ALWAYS_ASSERT(check(false, false) < 1.e-12_rt);
                auto const requested = discarded.RequestedWork().sum(0);
                auto const actual = discarded.ActualWork().sum(0);
                auto const deferred = discarded.EnergyCarryChange().sum(0);
                AMREX_ALWAYS_ASSERT(requested > 0 && actual == 0);
                auto const gamma = std::sqrt(1 + speed * speed / PhysConst::c2);
                AMREX_ALWAYS_ASSERT(std::abs(discarded.WorkVelocity().max(2) / speed
                                            - 1 / gamma) < 1.e-12_rt);
                AMREX_ALWAYS_ASSERT(std::abs(discarded.WorkVelocity().min(2) / speed
                                            - 1 / gamma) < 1.e-12_rt);
                AMREX_ALWAYS_ASSERT(std::abs(discarded.MaterialMass().sum(0)
                                            / (1.e20_rt * ions.getMass()) - 1) < 1.e-12_rt);
                AMREX_ALWAYS_ASSERT(std::abs(requested - deferred) < 1.e-12_rt * requested);
            }
            AMREX_ALWAYS_ASSERT(check(false, false) < 1.e-12_rt);
            ParticleImpulseTransaction accepted;
            AMREX_ALWAYS_ASSERT(stage(accepted, request));
            accepted.Commit();
            AMREX_ALWAYS_ASSERT(check(false, true) < 1.e-12_rt);
            // Use WarpX's real ballistic position pusher and MPI redistribution.
            auto const gamma = std::sqrt(1 + speed * speed / PhysConst::c2);
            ions.PushX(0, 0.5_rt * gamma / speed);
            ions.Redistribute();
            AMREX_ALWAYS_ASSERT(ions.TotalNumberOfParticles() == expected_particles);
            AMREX_ALWAYS_ASSERT(check(true, true) < 1.e-12_rt);
            if (!checkpoint.empty()) { ions.Checkpoint(checkpoint, "ions"); }
        }
        else
        {
            // The low-level AMReX reader appends; unlike WarpX's normal restart
            // path this driver first created a fresh fixture population.
            ions.clearParticles();
            ions.Restart(restart, "ions");
            AMREX_ALWAYS_ASSERT(ions.TotalNumberOfParticles() == expected_particles);
            AMREX_ALWAYS_ASSERT(check(true, true) < 1.e-12_rt);
        }
        // A rejected non-finite request must not modify relocated owners.
        request.setVal(std::numeric_limits<amrex::Real>::infinity());
        ParticleImpulseTransaction rejected;
        AMREX_ALWAYS_ASSERT(!stage(rejected, request));
        AMREX_ALWAYS_ASSERT(check(true, true) < 1.e-12_rt);
        // Continue after migration or changed-rank restart: the same material
        // now receives a representable kick in its new cells. Its prior work
        // account is retained while the new realized work is measured.
        request.setVal(0);
        for (amrex::MFIter iterator(request); iterator.isValid(); ++iterator)
        {
            auto const field = request.array(iterator);
            amrex::ParallelFor(iterator.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
#if AMREX_SPACEDIM == 1
                amrex::Real const z = plo[0] + (i + 0.5_rt) * dx[0];
#else
                amrex::Real const z = plo[1] + (j + 0.5_rt) * dx[1];
#endif
                field(i, j, k, 2) = z >= 0.5_rt ? 8 * mass * increment : 0;
            });
        }
        ParticleImpulseTransaction continued;
        AMREX_ALWAYS_ASSERT(stage(continued, request));
        auto const work = continued.RequestedWork().sum(0);
        auto const accounted = continued.ActualWork().sum(0)
            + continued.EnergyCarryChange().sum(0) + continued.NumericalEnergyResidual().sum(0);
        AMREX_ALWAYS_ASSERT(work > 0 && std::abs(work - accounted) < 1.e-12_rt * work);
        continued.Commit();
        AMREX_ALWAYS_ASSERT(check(true, true, true) < 1.e-12_rt);
        amrex::Print() << "Particle impulse staging, discard, ballistic ownership, "
                          "redistribution, rejection and continued kick passed. Restart="
                       << !restart.empty() << '\n';
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
