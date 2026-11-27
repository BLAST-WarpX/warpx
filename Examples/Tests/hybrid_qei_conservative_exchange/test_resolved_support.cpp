/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QeiThermalSupport.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>

using namespace amrex::literals;

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        // Fixed-key ensemble: sanity-check proposal moments and independent
        // component/counter streams without weakening any energy tolerances.
        std::array<double, 4> sampler_moments{};
        double component_cross = 0.0;
        double counter_cross = 0.0;
        constexpr std::uint64_t samples = 1000000;
        for (std::uint64_t particle = 0; particle < samples; ++particle) {
            auto const x = warpx::hybrid::qeiNormal(1, 0, 0, particle, 0);
            auto const y = warpx::hybrid::qeiNormal(1, 0, 0, particle, 1);
            auto const next = warpx::hybrid::qeiNormal(1, 1, 0, particle, 0);
            AMREX_ALWAYS_ASSERT(std::isfinite(x) && std::isfinite(y) && std::isfinite(next));
            auto power = x;
            for (auto& moment : sampler_moments) {
                moment += power / static_cast<double>(samples);
                power *= x;
            }
            component_cross += x * y / static_cast<double>(samples);
            counter_cross += x * next / static_cast<double>(samples);
        }
        AMREX_ALWAYS_ASSERT(std::abs(sampler_moments[0]) < 0.01);
        AMREX_ALWAYS_ASSERT(std::abs(sampler_moments[1] - 1.0) < 0.02);
        AMREX_ALWAYS_ASSERT(std::abs(sampler_moments[2]) < 0.04);
        AMREX_ALWAYS_ASSERT(std::abs(sampler_moments[3] - 3.0) < 0.1);
        AMREX_ALWAYS_ASSERT(std::abs(component_cross) < 0.01);
        AMREX_ALWAYS_ASSERT(std::abs(counter_cross) < 0.01);
        auto& simulation = WarpX::GetInstance();
        simulation.InitData();
        simulation.HybridPICPrepareElectronStateForDiagnostics();
        auto& model = *simulation.get_pointer_HybridPICModel();
        auto& pc = simulation.GetPartContainer().GetParticleContainerFromName("ions");
        bool expect_no_exchange = false;
        amrex::ParmParse("test").query("expect_no_exchange", expect_no_exchange);
        using Velocities = std::array<amrex::Gpu::HostVector<amrex::ParticleReal>, 3>;
        std::map<std::pair<int, int>, Velocities> initial_velocities;
        for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
            auto& saved = initial_velocities[{pti.index(), pti.LocalTileIndex()}];
            for (int d = 0; d < 3; ++d) {
                auto const& values = pti.GetAttribs(PIdx::ux + d);
                saved[d].resize(values.size());
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(), values.end(),
                                 saved[d].begin());
            }
        }
        auto const& geom = simulation.Geom(0);
        auto const dx = geom.CellSizeArray();
        auto const domain = amrex::ubound(geom.Domain());
        using warpx::fields::FieldType;
        auto& te = *simulation.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto const& rho = *simulation.m_fields.get(FieldType::rho_fp, 0);
        auto owner = te.OwnerMask(geom.periodicity());
        auto const initial = warpx::hybrid::depositResolvedQeiMoments(pc, 0, geom);
        amrex::MultiFab dummy(amrex::convert(te.boxArray(), amrex::IntVect::TheCellVector()),
                              te.DistributionMap(), 1, 0);
        dummy.setVal(0);
        std::map<std::string, amrex::MultiFab*> temperatures{{"ions", &dummy}};
        auto const mass = static_cast<amrex::Real>(pc.getMass());
        auto inventory = [&] () {
            amrex::MultiFab energy(te.boxArray(), te.DistributionMap(), 1, 0);
            for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi) {
                auto const out = energy.array(mfi);
                auto const t = te.const_array(mfi);
                auto const charge = rho.const_array(mfi);
                auto const owned = owner->const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    amrex::ignore_unused(k);
                    auto volume = (i == 0 ? MathConst::pi * dx[0] * dx[0] / 3
                                          : 2 * MathConst::pi * i * dx[0] * dx[0]) *
                                  dx[1];
                    if (i == domain.x + 1) {
                        volume *= 0.5_rt;
                    }
                    if (j == 0 || j == domain.y + 1) {
                        volume *= 0.5_rt;
                    }
                    out(i, j, k) = owned(i, j, k) ? 1.5_rt * charge(i, j, k) / PhysConst::q_e *
                                                        PhysConst::kb * t(i, j, k) * volume
                                                  : 0.0_rt;
                });
            }
            auto moments = warpx::hybrid::depositResolvedQeiMoments(pc, 0, geom);
            amrex::MultiFab kinetic(moments->boxArray(), moments->DistributionMap(), 1, 0);
            for (amrex::MFIter mfi(kinetic); mfi.isValid(); ++mfi) {
                auto const out = kinetic.array(mfi);
                auto const m = moments->const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    auto const n = m(i, j, k, 0);
                    out(i, j, k) = n > 0 ? 0.5_rt * mass *
                                               (m(i, j, k, 4) + (m(i, j, k, 1) * m(i, j, k, 1) +
                                                                 m(i, j, k, 2) * m(i, j, k, 2) +
                                                                 m(i, j, k, 3) * m(i, j, k, 3)) /
                                                                    n)
                                         : 0.0_rt;
                });
            }
            return energy.sum(0) + kinetic.sum(0);
        };
        auto const before = inventory();
        amrex::Real maximum_energy_error = 0;
        for (int step = 0; step < 100; ++step) {
            model.QDSMCAddTemperatureRelaxation(0, 1.e-10_rt, temperatures);
            model.QDSMCApplyIonHeating(0, 1.e-10_rt, nullptr, &temperatures);
            auto const after = inventory();
            maximum_energy_error =
                std::max(maximum_energy_error, std::abs(after - before) / before);
        }
        auto const final = warpx::hybrid::depositResolvedQeiMoments(pc, 0, geom);
        amrex::MultiFab errors(final->boxArray(), final->DistributionMap(), 3, 0);
        for (amrex::MFIter mfi(errors); mfi.isValid(); ++mfi) {
            auto const out = errors.array(mfi);
            auto const a = initial->const_array(mfi);
            auto const b = final->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                amrex::Real error = 0;
                amrex::Real singleton = 0;
                for (int c = 0; c < 4; ++c) {
                    error = std::max(error, std::abs(a(i, j, k, c) - b(i, j, k, c)) /
                                                std::max(a(i, j, k, 0) * PhysConst::c, 1.0_rt));
                    if (a(i, j, k, 5) == 1 && a(i, j, k, c) != b(i, j, k, c)) {
                        singleton = 1;
                    }
                }
                out(i, j, k, 0) = error;
                out(i, j, k, 1) = singleton;
                out(i, j, k, 2) = 0.5_rt * mass * (b(i, j, k, 4) - a(i, j, k, 4));
            });
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(maximum_energy_error < 1.e-10_rt,
                                         "Resolved Qei exchange failed total "
                                         "electron/ion energy conservation.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            errors.norm0(0) < 1.e-12_rt && errors.norm0(1) == 0,
            "Resolved Qei exchange changed cell momentum or an unresolved "
            "singleton.");
        if (expect_no_exchange) {
            bool unchanged = maximum_energy_error == 0;
            for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
                auto const& saved = initial_velocities.at({pti.index(), pti.LocalTileIndex()});
                for (int d = 0; d < 3; ++d) {
                    auto const& values = pti.GetAttribs(PIdx::ux + d);
                    amrex::Gpu::HostVector<amrex::ParticleReal> current(values.size());
                    amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(), values.end(),
                                     current.begin());
                    unchanged =
                        unchanged && std::equal(current.begin(), current.end(), saved[d].begin());
                }
            }
            amrex::ParallelDescriptor::ReduceBoolAnd(unchanged);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(unchanged,
                                             "Zero Qei rate must preserve every particle "
                                             "momentum and the energy inventory exactly.");
        } else {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                errors.max(2) > 1.e-3_rt * before && errors.min(2) < -1.e-3_rt * before,
                "Sparse support test must exercise both ion heating and ion "
                "cooling.");
        }
        amrex::Print() << "100 stiff sparse-cell exchanges: maximum energy error = "
                       << maximum_energy_error << ", momentum error = " << errors.norm0(0) << '\n';
        bool check_source = false;
        amrex::ParmParse("test").query("check_radiation_source", check_source);
        if (check_source) {
            amrex::MultiFab source(dummy.boxArray(), dummy.DistributionMap(), 1, 1);
            source.setVal(0);
            for (amrex::MFIter mfi(source); mfi.isValid(); ++mfi) {
                auto const out = source.array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    out(i, j, k) = i == 0 && j == 2 ? 1.0_rt : 0.0_rt;
                });
            }
            source.FillBoundary(geom.periodicity());
            auto const old_inventory = inventory();
            auto const residual = model.ApplyElectronEnergySource(0, source, 1.e10_rt);
            auto const realized = inventory() - old_inventory;
            amrex::Print() << "Axis source: native energy increment=" << realized
                           << ", reported residual=" << residual << '\n';
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::abs(realized - (1.0_rt - residual)) < 1.e-10_rt,
                "Radiation source must conserve the evolved native caloric "
                "inventory.");
            for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                auto const temperature = te.array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    temperature(i, j, k) =
                        (i == 0 ? 0.01_rt : 100.0_rt) * PhysConst::q_e / PhysConst::kb;
                });
            }
            te.FillBoundary(geom.periodicity());
            source.setVal(0);
            for (amrex::MFIter mfi(source); mfi.isValid(); ++mfi) {
                auto const out = source.array(mfi);
                auto const temperature = te.const_array(mfi);
                auto const charge = rho.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if (i != 0 || j != 2) {
                        return;
                    }
                    amrex::Real available = 0;
                    for (int di = 0; di < 2; ++di) {
                        for (int dj = 0; dj < 2; ++dj) {
                            auto const corner = MathConst::pi * dx[0] * dx[0] * dx[1] *
                                                (di == 0 ? 0.25_rt : 0.75_rt) * 0.5_rt;
                            available += 1.5_rt * charge(i + di, j + dj, k) / PhysConst::q_e *
                                         PhysConst::kb * temperature(i + di, j + dj, k) * corner;
                        }
                    }
                    out(i, j, k) = -0.5_rt * available;
                });
            }
            source.FillBoundary(geom.periodicity());
            auto const requested_cooling = source.sum(0);
            auto const before_cooling = inventory();
            auto const cooling_residual = model.ApplyElectronEnergySource(0, source, 1.e10_rt);
            auto const realized_cooling = inventory() - before_cooling;
            amrex::Print() << "Cold/hot cell cooling: requested=" << requested_cooling
                           << ", realized=" << realized_cooling << '\n';
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                requested_cooling < 0 && te.min(0) > 0 &&
                    std::abs(realized_cooling - (requested_cooling - cooling_residual)) <
                        1.e-10_rt * std::abs(requested_cooling),
                "Cold/hot source remap must preserve positivity and actual "
                "energy.");
        }
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
