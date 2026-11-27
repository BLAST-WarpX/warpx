/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "Radiation/RadiationTransport.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <cmath>

using namespace amrex::literals;

int main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        std::string table_root;
        amrex::ParmParse test_options("test");
        if (test_options.query("opacity_table_root", table_root)) {
            // Select a backend before initialization; do not alter live opacity
            // state. The resulting complete inputs are recorded by WarpX.
            amrex::ParmParse opacity_options("radiation_transport");
            opacity_options.remove("planck_absorption_coefficient");
            opacity_options.remove(
                "rosseland_transport_coefficient(x,y,z,t,ne,Te)");
            opacity_options.add("opacity_table_interpolation",
                                std::string("log_log"));
#if defined(WARPX_DIM_RZ)
            opacity_options.add(
                "planck_absorption_coefficient_spectral_table_file",
                table_root + "/opacity_coupled_planck_3d.dat");
            opacity_options.add(
                "rosseland_transport_coefficient_spectral_table_file",
                table_root + "/opacity_coupled_rosseland_3d.dat");
#else
            opacity_options.add("planck_absorption_coefficient_table_file",
                                table_root + "/opacity_coupled_planck_2d.dat");
            opacity_options.add("rosseland_transport_coefficient_table_file",
                                table_root +
                                    "/opacity_coupled_rosseland_2d.dat");
#endif
        }
        auto& simulation = WarpX::GetInstance();
        simulation.InitData();
        simulation.HybridPICPrepareElectronStateForDiagnostics();
        using warpx::fields::FieldType;
        auto& model = *simulation.get_pointer_HybridPICModel();
        auto& te = *simulation.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto& rho = *simulation.m_fields.get(FieldType::rho_fp, 0);
        auto& radiation = *simulation.m_fields.get(FieldType::radiation_diffusion_energy, 0);
        auto const& geometry = simulation.Geom(0);
        auto const dx = geometry.CellSizeArray();
        auto const lower = geometry.ProbLoArray();
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lengths{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { lengths[d] = geometry.ProbLength(d); }
        auto const lo = amrex::lbound(geometry.Domain());
        bool const latent = model.electronThermodynamicsExecutor().isFixedChargeLatentEnergy();
#if defined(WARPX_DIM_RZ)
        auto const hi = amrex::ubound(geometry.Domain());
        auto const upper = geometry.ProbHiArray();
        auto const native_measure = model.transportsElectronInternalEnergyAtRawDensity();
        auto const axis_factor = simulation.verboncoeurAxisCorrection() ? 1.0_rt/3 : 1.0_rt/4;
        AMREX_ALWAYS_ASSERT(geometry.isPeriodic(1));
        auto geometric_node_volume = [=] AMREX_GPU_HOST_DEVICE(int node) {
            auto const r = lower[0] + (node-lo.x)*dx[0];
            auto const a = amrex::max(lower[0], r-dx[0]/2);
            auto const b = amrex::min(upper[0], r+dx[0]/2);
            return MathConst::pi*(b*b-a*a)*dx[1];
        };
        auto node_volume = [=] AMREX_GPU_HOST_DEVICE(int node) {
            if (!native_measure) { return geometric_node_volume(node); }
            auto const r = lower[0] + (node-lo.x)*dx[0];
            if (r == 0) { return MathConst::pi*dx[0]*dx[0]*dx[1]*axis_factor; }
            auto v = 2*MathConst::pi*r*dx[0]*dx[1];
            if (node == hi.x+1 || (node == lo.x && r > 0)) { v *= 0.5_rt; }
            return v;
        };
#endif
        constexpr amrex::Real number_density = 1.e25_rt;
        constexpr amrex::Real dt = 1.e-13_rt;
        rho.setVal(number_density * PhysConst::q_e);
        // Nonuniform physical native nodes, including the axis and walls.
        for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
            auto const t = te.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                amrex::ignore_unused(j, k);
                auto const x = (i - lo.x) * dx[0] / lengths[0];
                amrex::Real modulation = std::cos(2 * MathConst::pi * x);
#if AMREX_SPACEDIM >= 2
                modulation *= std::cos(2 * MathConst::pi * (j - lo.y) * dx[1] / lengths[1]);
#endif
                t(i, j, k) = 100 * PhysConst::q_e / PhysConst::kb * (1 + 0.05_rt * modulation);
            });
        }
        te.FillBoundary(geometry.periodicity());
        // Independent physical subcell quadrature and analytic caloric EOS.
        // Do not reuse the producer's volume helper, pressure or source ledger
        // as the caloric oracle. Each valid cell owns disjoint corner volumes.
        auto caloric_energy = [&] {
            amrex::ReduceOps<amrex::ReduceOpSum> ops;
            amrex::ReduceData<amrex::Real> data(ops);
            using Tuple = typename decltype(data)::Type;
            for (amrex::MFIter mfi(radiation); mfi.isValid(); ++mfi) {
                auto const t = te.const_array(mfi);
                ops.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                    amrex::Real energy = 0;
                    for (int dj = 0; dj < (AMREX_SPACEDIM > 1 ? 2 : 1); ++dj) {
                        for (int di = 0; di < 2; ++di) {
                            amrex::Real volume = 0.5_rt * dx[0];
#if defined(WARPX_DIM_RZ)
                            auto const r0 = lower[0] + (i - lo.x + 0.5_rt * di) * dx[0];
                            auto const r1 = r0 + 0.5_rt * dx[0];
                            volume = MathConst::pi * (r1 * r1 - r0 * r0);
#else
                            amrex::ignore_unused(lower);
#endif
#if AMREX_SPACEDIM >= 2
                            volume *= 0.5_rt * dx[1];
#endif
#if defined(WARPX_DIM_RZ)
                            volume *= node_volume(i+di) / geometric_node_volume(i+di);
#endif
                            auto const ev = t(i + di, j + dj, k) * PhysConst::kb / PhysConst::q_e;
                            auto const ratio = ev / 100.0_rt;
                            auto const power = ratio * ratio * ratio * ratio;
                            auto const u = number_density * PhysConst::q_e *
                                (1.5_rt * ev + (latent ? 1000.0_rt * power / (1 + power) : 0));
                            energy += volume * u;
                        }
                    }
                    return {energy};
                });
            }
            auto value = amrex::get<0>(data.value());
            amrex::ParallelDescriptor::ReduceRealSum(value);
            return value;
        };
        auto radiation_energy = [&] {
            amrex::Real sum = 0;
            for (int g = 0; g < radiation.nComp(); ++g) { sum += radiation.sum(g, false); }
            return sum;
        };
        auto const initial_material = caloric_energy();
        auto const initial_total = initial_material + radiation_energy();
        amrex::Real worst = 0;
        for (int step = 0; step < 100; ++step) {
            auto const old_material = caloric_energy();
            simulation.GetRadiationTransport().Advance(simulation.GetPartContainer(),
                simulation.m_fields, step * dt, dt);
            auto const material = caloric_energy();
            auto const source = simulation.m_fields.get(FieldType::radiation_material_energy, 0)
                                    ->sum(0, false);
            auto const residual = std::abs(material + radiation_energy() - initial_total) / initial_total;
            worst = amrex::max(worst, residual);
            AMREX_ALWAYS_ASSERT(residual < 1.e-10_rt);
            AMREX_ALWAYS_ASSERT(std::abs(material - old_material - source) < 1.e-10_rt * initial_total);
            AMREX_ALWAYS_ASSERT(te.min(0) > 0 && te.is_finite());
            for (int g = 0; g < radiation.nComp(); ++g) { AMREX_ALWAYS_ASSERT(radiation.min(g) >= 0); }
        }
        AMREX_ALWAYS_ASSERT(std::abs(caloric_energy() - initial_material) > 0.01_rt * initial_material);
        amrex::Print() << "100-stage native caloric/metric trajectory: latent="
                       << latent << " tables=" << !table_root.empty()
                       << " worst raw energy residual=" << worst << '\n';
        // Independent uniform nonlinear backward-Euler root, not a ledger oracle.
        te.setVal(100 * PhysConst::q_e / PhysConst::kb);
        for (amrex::MFIter mfi(radiation); mfi.isValid(); ++mfi) {
            auto const e = radiation.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                amrex::Real volume = dx[0];
#if defined(WARPX_DIM_RZ)
                auto const r0 = lower[0] + (i - lo.x) * dx[0];
                auto const r1 = r0 + dx[0];
                volume = MathConst::pi * (r1 * r1 - r0 * r0);
#endif
#if AMREX_SPACEDIM >= 2
                volume *= dx[1];
#endif
                e(i, j, k) = 2.e9_rt * volume;
            });
        }
        radiation.FillBoundary(geometry.periodicity());
        auto u_of = [=] AMREX_GPU_HOST_DEVICE (amrex::Real ev) {
            auto const ratio = ev / 100;
            auto const power = ratio * ratio * ratio * ratio;
            return number_density * PhysConst::q_e *
                (1.5_rt * ev + (latent ? 1000.0_rt * power / (1 + power) : 0));
        };
#if !defined(WARPX_DIM_RZ)
        amrex::Real low = 100;
        amrex::Real high = 100 + 2.e9_rt / (1.5_rt * number_density * PhysConst::q_e);
        for (int iteration = 0; iteration < 100; ++iteration) {
            auto const ev = 0.5_rt * (low + high);
            auto const t = ev * PhysConst::q_e / PhysConst::kb;
            auto const e = 2.e9_rt - (u_of(ev) - u_of(100));
            auto const equation = e - 2.e9_rt + dt * PhysConst::c * 1000 *
                (e - 7.565733250280007e-16_rt * t * t * t * t);
            if (equation > 0) { low = ev; } else { high = ev; }
        }
        auto const expected = 0.5_rt * (low + high) * PhysConst::q_e / PhysConst::kb;
#else
        auto const reset_total = caloric_energy() + radiation_energy();
#endif
        simulation.GetRadiationTransport().Advance(simulation.GetPartContainer(),
            simulation.m_fields, 100 * dt, dt);
#if defined(WARPX_DIM_RZ)
        // Uniform physical radiation does not imply uniform T at native FV
        // axis/wall nodes. Check their actual nonlinear BE material equation
        // instead of imposing the Cartesian zero-dimensional boundary root.
        amrex::MultiFab requested(radiation.boxArray(), radiation.DistributionMap(), 1, 1);
        requested.setVal(0);
        int const groups = radiation.nComp();
        for (amrex::MFIter mfi(requested); mfi.isValid(); ++mfi) {
            auto const t = te.const_array(mfi), e = radiation.const_array(mfi);
            auto const q = requested.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                amrex::Real cv_sum = 0, cv_t = 0, volume = 0, energy = 0;
                for (int di = 0; di < 2; ++di) {
                    auto const a = lower[0] + (i-lo.x+0.5_rt*di)*dx[0];
                    auto const b = a + dx[0]/2;
                    auto const v = MathConst::pi*(b*b-a*a)*dx[1]/2;
                    for (int dj = 0; dj < 2; ++dj) {
                        auto const ev = t(i+di,j+dj,k)*PhysConst::kb/PhysConst::q_e;
                        auto const x = ev/100, x3 = x*x*x, x4 = x3*x;
                        auto const cv = 1.5_rt + (latent ? 40*x3/((1+x4)*(1+x4)) : 0);
                        cv_sum += cv*v;
                        cv_t += cv*v*ev;
                        volume += v;
                    }
                }
                for (int g = 0; g < groups; ++g) { energy += e(i,j,k,g); }
                auto const tc = cv_t/cv_sum*PhysConst::q_e/PhysConst::kb;
                q(i,j,k) = dt*PhysConst::c*1000*(energy-volume*7.565733250280007e-16_rt*tc*tc*tc*tc);
            });
        }
        requested.FillBoundary(geometry.periodicity());
        amrex::ReduceOps<amrex::ReduceOpMax> check_ops;
        amrex::ReduceData<amrex::Real> check_data(check_ops);
        for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
            auto const t = te.const_array(mfi), q = requested.const_array(mfi);
            check_ops.eval(mfi.validbox(), check_data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<amrex::Real> {
                    amrex::Real received = 0;
                    for (int ci = i-1; ci <= i; ++ci) {
                        if (ci < lo.x || ci > hi.x) { continue; }
                        auto const a = lower[0]+(ci-lo.x)*dx[0], b = a+dx[0];
                        auto const ca = a+(i-ci)*dx[0]/2, cb = ca+dx[0]/2;
                        auto const weight = (cb*cb-ca*ca)/(b*b-a*a)/2;
                        for (int cj = j-1; cj <= j; ++cj) { received += weight*q(ci,cj,k); }
                    }
                    auto const ev = t(i,j,k)*PhysConst::kb/PhysConst::q_e;
                    auto const realized = (u_of(ev)-u_of(100))*node_volume(i);
                    auto const scale = amrex::max(std::abs(received), std::abs(realized));
                    return {scale > 0 ? std::abs(realized-received)/scale : 0};
                });
        }
        auto error = amrex::get<0>(check_data.value());
        amrex::ParallelDescriptor::ReduceRealMax(error);
        amrex::Print() << "RZ nonlinear nodal BE relative source error=" << error << '\n';
        AMREX_ALWAYS_ASSERT(error < 1.e-8_rt);
        AMREX_ALWAYS_ASSERT(std::abs(caloric_energy()+radiation_energy()-reset_total) < 1.e-10_rt*reset_total);
#else
        auto const heating = expected - 100 * PhysConst::q_e / PhysConst::kb;
        amrex::Print() << "Nonlinear uniform root relative heating error="
            << (te.max(0) - expected) / heating << '\n';
        AMREX_ALWAYS_ASSERT(std::abs(te.max(0) - expected) < 1.e-8_rt * heating);
        AMREX_ALWAYS_ASSERT(std::abs(te.min(0) - expected) < 1.e-8_rt * heating);
#endif
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
