/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/FrameTransform.H"
#include "Radiation/MomentClosure.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>

using namespace amrex::literals;
using namespace warpx::radiation;

AMREX_GPU_HOST_DEVICE amrex::Real
check_frame (int index)
{
    // Both velocity signs, all axes and speeds from zero through beta=0.6.
    amrex::Real const speed = (index < 102 ? 0.6_rt : 0.01_rt) * (index % 17) / 16;
    amrex::GpuArray<amrex::Real, 3> beta{};
    beta[index % 3] = index % 2 == 0 ? speed : -speed;
    auto inverse = beta;
    for (int d = 0; d < 3; ++d) {
        inverse[d] = -beta[d];
    }
    StressEnergy rest{};
    rest[0][0] = 3;
    for (int d = 1; d < 4; ++d) {
        rest[d][d] = 1;
    }
    auto const lab = BoostStressEnergy(rest, beta);
    amrex::Real error = 0;
    auto maximum = [&] (amrex::Real value) { error = amrex::max(error, std::abs(value)); };
    auto const gamma2 = 1 / (1 - speed * speed);
    // Independently boost an isotropic stress tensor, then reconstruct its
    // pressure from only the lab energy and flux. This checks the moving
    // closure rather than a second copy of its chi formula.
    auto const closure = EvaluateM1Closure(lab[0]);
    if (!closure.valid) {
        return 2;
    }
    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            maximum((closure.tensor[a][b] - lab[a][b]) / lab[0][0]);
        }
    }
    maximum((lab[0][0] - gamma2 * (3 + speed * speed)) / 3);
    for (int d = 0; d < 3; ++d) {
        maximum((lab[0][d + 1] - 4 * gamma2 * beta[d]) / 3);
    }
    auto const roundtrip = BoostStressEnergy(lab, inverse);
    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            maximum((roundtrip[a][b] - rest[a][b]) / 3);
        }
    }
    // Boosted LTE must have zero four-force, including beta^2 terms.
    auto const lte = EvaluateGreyFourForce(lab, beta, 7, 11, 3);
    if (!lte.valid) {
        return 1;
    }
    for (auto value : lte.material_force) {
        maximum(value / 54);
    }
    // Pure scattering exchanges lab work but zero rest-frame heat.
    rest[0][1] = rest[1][0] = 0.2_rt;
    auto const scattering = EvaluateGreyFourForce(BoostStressEnergy(rest, beta), beta, 0, 5, 0);
    if (!scattering.valid) {
        return 1;
    }
    amrex::Real work = 0;
    for (int d = 0; d < 3; ++d) {
        work += beta[d] * scattering.material_force[d + 1];
    }
    maximum((scattering.material_force[0] - work) / 15);
    // A lab beam has the independent extinction factor gamma*(1-beta.n).
    FourVector const direction{1, 0.6_rt, 0, 0.8_rt};
    StressEnergy beam{};
    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            beam[a][b] = 3 * direction[a] * direction[b];
        }
    }
    auto const absorbed = EvaluateGreyFourForce(beam, beta, 7, 0, 0);
    auto const beam_closure = EvaluateM1Closure(beam[0]);
    if (!beam_closure.valid) {
        return 3;
    }
    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            maximum((beam_closure.tensor[a][b] - beam[a][b]) / 3);
        }
    }
    if (!absorbed.valid) {
        return 1;
    }
    auto const factor = 21 * std::sqrt(gamma2) * (1 - 0.6_rt * beta[0] - 0.8_rt * beta[2]);
    for (int a = 0; a < 4; ++a) {
        maximum((absorbed.material_force[a] - factor * direction[a]) / 21);
    }
    // An actual finite packet sample need not have zero sampled momentum.
    FourVector sampled{};
    FourVector sampled_lab{};
    for (int packet = 0; packet < 7; ++packet) {
        FourVector value{};
        value[0] = 1 + 0.1_rt * packet;
        value[1 + packet % 3] = packet % 2 == 0 ? value[0] : -value[0];
        auto const boosted = BoostFourVector(value, beta);
        auto const back = BoostFourVector(boosted, inverse);
        for (int a = 0; a < 4; ++a) {
            sampled[a] += value[a];
            sampled_lab[a] += boosted[a];
            maximum((back[a] - value[a]) / value[0]);
        }
    }
    auto const aggregate = BoostFourVector(sampled, beta);
    for (int a = 0; a < 4; ++a) {
        maximum((aggregate[a] - sampled_lab[a]) / sampled_lab[0]);
    }
    if (sampled[1] == 0 && sampled[2] == 0 && sampled[3] == 0) {
        return 1;
    }
    auto const emitted = EvaluateGreyFourForce({}, beta, 7, 0, 3);
    maximum((emitted.material_force[0] + 21 * std::sqrt(gamma2)) / 21);
    for (int d = 0; d < 3; ++d) {
        maximum((emitted.material_force[d + 1] + 21 * std::sqrt(gamma2) * beta[d]) / 21);
    }
    // Rounded streaming states must retain their exact input momentum, while
    // a genuinely excessive flux is rejected rather than clipped to E.
    auto const eps = std::numeric_limits<amrex::Real>::epsilon();
    auto const rounded_beam = EvaluateM1Closure({1, 1 + 2 * eps, 0, 0});
    if (!rounded_beam.valid || rounded_beam.tensor[0][1] != 1 + 2 * eps ||
        EvaluateM1Closure({1, 1 + 64 * eps, 0, 0}).valid) {
        return 4;
    }
    if (EvaluateGreyFourForce(beam, {1, 0, 0}, 7, 0, 0).valid ||
        EvaluateGreyFourForce(beam, beta, -1, 0, 0).valid ||
        !EvaluateM1Closure({0, 0, 0, 0}).valid || EvaluateM1Closure({0, 1, 0, 0}).valid ||
        EvaluateM1Closure({-1, 0, 0, 0}).valid || EvaluateM1Closure({1, 1, 0.01_rt, 0}).valid) {
        return 1;
    }
    return error;
}

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        constexpr int cases = 204;
        amrex::Gpu::DeviceVector<amrex::Real> device(cases);
        auto* output = device.dataPtr();
        amrex::ParallelFor(cases, [=] AMREX_GPU_DEVICE(int i) { output[i] = check_frame(i); });
        amrex::Gpu::HostVector<amrex::Real> host(cases);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, device.begin(), device.end(), host.begin());
        amrex::Real worst = 0;
        for (int i = 0; i < cases; ++i) {
            worst = amrex::max(worst, amrex::max(host[i], check_frame(i)));
            if (host[i] >= 1.e-12_rt || check_frame(i) >= 1.e-12_rt) {
                amrex::Print() << "Frame case " << i << ": device=" << host[i]
                               << " host=" << check_frame(i) << '\n';
            }
        }
        AMREX_ALWAYS_ASSERT(worst < 1.e-12_rt);
        amrex::Print() << "Frame, M1 closure, moving LTE, beam extinction, scattering work "
                          "and finite-sample momentum: "
                       << worst << '\n';
    }
    amrex::Finalize();
}
