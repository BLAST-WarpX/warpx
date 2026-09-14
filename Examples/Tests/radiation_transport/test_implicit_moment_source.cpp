/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/ImplicitMomentSource.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>

using namespace amrex::literals;
using namespace warpx::radiation;

AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, 3>
case_beta (int index)
{
    auto const velocity_scale = index < 210 ? 1._rt : 0.01_rt;
    index %= 210;
    amrex::Real const speed = velocity_scale * 0.6_rt * ((index / 7) % 5) / 4;
    int const axis = (index / 35) % 3;
    amrex::Real const sign = index % 2 == 0 ? 1 : -1;
    amrex::GpuArray<amrex::Real, 3> beta{};
    beta[axis] = sign * speed * (index < 105 ? 1 : 0.6_rt);
    if (index >= 105) {
        beta[(axis + 1) % 3] = sign * speed * 0.8_rt;
    }
    return beta;
}

AMREX_GPU_HOST_DEVICE amrex::Real
check_source (int index)
{
    amrex::Real const rates[] = {1.e-20_rt, 1.e-8_rt, 1.e-3_rt, 1, 100, 1.e4_rt, 1.e8_rt};
    amrex::Real const rate = rates[index % 7];
    amrex::Real const speed = (index < 210 ? 1._rt : 0.01_rt) * 0.6_rt * ((index / 7) % 5) / 4;
    auto const beta = case_beta(index);
    amrex::Real error = 0;
    auto maximum = [&] (amrex::Real value) { error = amrex::max(error, std::abs(value)); };

    // Exact stationary backward-Euler solution, both energy and flux damping.
    FourVector const initial{3, 0.6_rt, -0.3_rt, 0.9_rt};
    auto const stationary = TryImplicitGreyMomentSource(initial, {}, rate, 2 * rate, 1);
    if (!stationary.valid) {
        return 1;
    }
    maximum((stationary.radiation[0] - (3 + rate) / (1 + rate)) / 3);
    for (int d = 1; d < 4; ++d) {
        maximum((stationary.radiation[d] - initial[d] / (1 + 3 * rate)) /
                (std::abs(initial[d]) / (1 + 3 * rate)));
    }
    // A moving material attenuates a beam by gamma*(1-beta.n), independently
    // of the moment code. Check the implicit amplitude, including stiff beams.
    FourVector const beam{3, 1.8_rt, 0, 2.4_rt};
    auto const moving = TryImplicitGreyMomentSource(beam, beta, rate, 0, 0);
    if (!moving.valid) {
        return 2;
    }
    auto const gamma = 1 / std::sqrt(1 - speed * speed);
    auto const extinction = gamma * (1 - 0.6_rt * beta[0] - 0.8_rt * beta[2]);
    if (index % 7 == 0) {
        maximum(stationary.material_energy_minus_work / (2 * rate) - 1);
        maximum(moving.material_transfer[0] / (3 * rate * extinction) - 1);
    }
    for (int d = 0; d < 4; ++d) {
        maximum((moving.radiation[d] - beam[d] / (1 + rate * extinction)) /
                (3 / (1 + rate * extinction)));
    }
    // Boosted LTE is stationary at any stiffness, not just at beta=0.
    StressEnergy equilibrium{};
    equilibrium[0][0] = 3;
    for (int d = 1; d < 4; ++d) {
        equilibrium[d][d] = 1;
    }
    auto const lab = BoostStressEnergy(equilibrium, beta);
    auto const lte = TryImplicitGreyMomentSource(lab[0], beta, rate, 2 * rate, 3);
    if (!lte.valid) {
        return 3;
    }
    for (int d = 0; d < 4; ++d) {
        maximum((lte.radiation[d] - lab[0][d]) / lab[0][0]);
    }
    // Verify the analytic closure Jacobian independently by centered
    // differences away from the realizability boundary. Its O(h^2) truncation
    // has a separate 3e-9 gate; the analytic physics gate remains 1e-10.
    StressEnergy reconstructed{};
    for (int column = 0; column < 4; ++column) {
        auto const derivative = M1ClosureDerivative(lab[0], column);
        auto plus = lab[0];
        auto minus = lab[0];
        auto const h = 1.e-5_rt * lab[0][0];
        plus[column] += h;
        minus[column] -= h;
        auto const p = EvaluateM1Closure(plus);
        auto const m = EvaluateM1Closure(minus);
        if (!p.valid || !m.valid) {
            return 6;
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                auto const difference = (p.tensor[i][j] - m.tensor[i][j]) / (2 * h);
                if (std::abs(difference - derivative[i][j]) > 3.e-9_rt) {
                    return 7;
                }
                reconstructed[i][j] += derivative[i][j] * lab[0][column];
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            maximum((reconstructed[i][j] - lab[i][j]) / lab[0][0]);
        }
    }

    // Fixed-velocity coherent scattering exchanges lab work, but satisfies
    // delta E_material = beta dot (c*delta p_material) exactly in this model.
    auto const scattering = TryImplicitGreyMomentSource(initial, beta, 0, rate, 0);
    if (!scattering.valid) {
        return 4 + static_cast<int>(scattering.failure) * 0.1_rt;
    }
    amrex::Real work = 0;
    for (int d = 0; d < 3; ++d) {
        work += beta[d] * scattering.material_transfer[d + 1];
    }
    maximum((scattering.material_transfer[0] - work) / initial[0]);
    if (scattering.material_energy_minus_work != 0) {
        return 5;
    }
    if (index % 7 == 6) {
        auto const weak_heat = TryImplicitGreyMomentSource(initial, beta, 1.e-20_rt, rate, 0);
        if (!weak_heat.valid || !(weak_heat.material_energy_minus_work > 1.e-20_rt)) {
            return 8;
        }
    }

    for (int kind = 0; kind < 3; ++kind) {
        auto const& result = kind == 0 ? stationary : (kind == 1 ? moving : scattering);
        auto const& old = kind == 1 ? beam : initial;
        for (int d = 0; d < 4; ++d) {
            maximum((old[d] - result.radiation[d] - result.material_transfer[d] -
                     result.numerical_residual[d]) /
                    3);
        }
    }
    return error;
}

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        constexpr int cases = 420;
        amrex::Gpu::DeviceVector<amrex::Real> device(cases);
        auto* output = device.dataPtr();
        amrex::ParallelFor(cases, [=] AMREX_GPU_DEVICE(int i) { output[i] = check_source(i); });
        amrex::Gpu::HostVector<amrex::Real> host(cases);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, device.begin(), device.end(), host.begin());
        amrex::Real worst = 0;
        for (int i = 0; i < cases; ++i) {
            auto const cpu = check_source(i);
            worst = amrex::max(worst, amrex::max(cpu, host[i]));
            if (cpu >= 1.e-10_rt || host[i] >= 1.e-10_rt) {
                amrex::Print() << "Source case " << i << " host=" << cpu << " device=" << host[i]
                               << '\n';
                if (cpu >= 4 && cpu < 5) {
                    amrex::Real const rates[] = {1.e-20_rt, 1.e-8_rt, 1.e-3_rt, 1,
                                                 100,       1.e4_rt,  1.e8_rt};
                    auto const diagnostic = TryImplicitGreyMomentSource(
                        {3, 0.6_rt, -0.3_rt, 0.9_rt}, case_beta(i), 0, rates[i % 7], 0);
                    amrex::Print() << "iterations=" << diagnostic.iterations << '\n';
                    for (int d = 0; d < 4; ++d) {
                        amrex::Print()
                            << "row " << d << " residual=" << diagnostic.equation_residual[d]
                            << " roundoff=" << diagnostic.roundoff_bound[d] << '\n';
                    }
                }
            }
        }
        AMREX_ALWAYS_ASSERT(worst < 1.e-10_rt);
        auto const weak = TryImplicitGreyMomentSource({1, 0, 0, 0}, {}, 1.e-20_rt, 0, 0);
        AMREX_ALWAYS_ASSERT(weak.valid && weak.radiation[0] == 1);
        AMREX_ALWAYS_ASSERT(std::abs(weak.material_transfer[0] / 1.e-20_rt - 1) < 1.e-12_rt);
        AMREX_ALWAYS_ASSERT(weak.numerical_residual[0] == -weak.material_transfer[0]);
        auto const weak_heat = TryImplicitGreyMomentSource(
            {3, 0.6_rt, -0.3_rt, 0.9_rt}, {0.2_rt, -0.1_rt, 0}, 1.e-20_rt, 1.e4_rt, 0);
        AMREX_ALWAYS_ASSERT(weak_heat.valid && weak_heat.material_energy_minus_work > 1.e-20_rt);
        FourVector const before{1, 0.1_rt, 0, 0};
        auto const failed = TryImplicitGreyMomentSource(before, {0.2_rt, 0, 0}, 1, 2, 3, 1);
        AMREX_ALWAYS_ASSERT(!failed.valid);
        for (int d = 0; d < 4; ++d) {
            AMREX_ALWAYS_ASSERT(failed.radiation[d] == before[d]);
        }
        for (auto value : failed.material_transfer) {
            AMREX_ALWAYS_ASSERT(value == 0);
        }
        AMREX_ALWAYS_ASSERT(failed.material_energy_minus_work == 0);
        AMREX_ALWAYS_ASSERT(failed.energy_projection_residual == 0);
        AMREX_ALWAYS_ASSERT(!TryImplicitGreyMomentSource(before, {1, 0, 0}, 1, 0, 1).valid);
        AMREX_ALWAYS_ASSERT(!TryImplicitGreyMomentSource(before, {}, -1, 0, 1).valid);
        // Reject invalid tolerances without altering the caller's radiation
        // guess or accepting an energy/momentum transfer, including for NaN.
        for (amrex::Real const tolerance : {
                 0._rt, 1._rt, -1._rt, std::numeric_limits<amrex::Real>::infinity(),
                 std::numeric_limits<amrex::Real>::quiet_NaN()}) {
            auto const rejected = TryDrivenImplicitGreyMomentSource(
                before, before, {}, 1, 0, 1, 60, tolerance);
            AMREX_ALWAYS_ASSERT(!rejected.valid);
            for (int d = 0; d < 4; ++d) {
                AMREX_ALWAYS_ASSERT(rejected.radiation[d] == before[d]);
                AMREX_ALWAYS_ASSERT(rejected.material_transfer[d] == 0);
            }
            AMREX_ALWAYS_ASSERT(rejected.material_energy_minus_work == 0);
            AMREX_ALWAYS_ASSERT(rejected.energy_projection_residual == 0);
        }
        // A joint solve can have a non-realizable algebraic transport RHS
        // while its physical old state and accepted final state are valid.
        auto const driven_scattering = TryImplicitGreyMomentSourceWithTransport(
            {1, 0, 0, 0}, {0, -2, 0, 0}, {1, 0, 0, 0}, {}, 0, 3, 0);
        AMREX_ALWAYS_ASSERT(driven_scattering.valid);
        AMREX_ALWAYS_ASSERT(std::abs(driven_scattering.radiation[1] - 0.5_rt) < 1.e-12_rt);
        AMREX_ALWAYS_ASSERT(std::abs(driven_scattering.material_transfer[1] - 1.5_rt) < 1.e-12_rt);
        auto const driven_emission = TryImplicitGreyMomentSourceWithTransport(
            {1, 0, 0, 0}, {2, 0, 0, 0}, {1, 0, 0, 0}, {}, 3, 0, 2);
        AMREX_ALWAYS_ASSERT(driven_emission.valid);
        AMREX_ALWAYS_ASSERT(std::abs(driven_emission.radiation[0] - 1.25_rt) < 1.e-12_rt);
        AMREX_ALWAYS_ASSERT(std::abs(driven_emission.material_energy_minus_work + 2.25_rt) <
                            1.e-12_rt);
        auto const driven_weak = TryImplicitGreyMomentSourceWithTransport(
            {1, 0, 0, 0}, {0.25_rt, 0, 0, 0}, {0.75_rt, 0, 0, 0}, {}, 1.e-20_rt, 0, 0);
        AMREX_ALWAYS_ASSERT(driven_weak.valid && driven_weak.radiation[0] == 0.75_rt);
        AMREX_ALWAYS_ASSERT(std::abs(driven_weak.material_energy_minus_work / 7.5e-21_rt - 1) <
                            1.e-12_rt);
        AMREX_ALWAYS_ASSERT(!TryImplicitGreyMomentSourceWithTransport({1, 0, 0, 0}, {0, -2, 0, 0},
                                                                      {1, 0, 0, 0}, {}, 0, 0, 0)
                                 .valid);
        AMREX_ALWAYS_ASSERT(!TryImplicitGreyMomentSource({1, 2, 0, 0}, {}, 0, 3, 0).valid);
        amrex::Print() << "Implicit moving gray source: " << cases
                       << " cases; worst analytic/ledger error=" << worst << '\n';
    }
    amrex::Finalize();
}
