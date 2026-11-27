/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/MovingMomentFlux.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace warpx::radiation;

namespace
{
    struct Case {
        FourVector left{}, right{};
        amrex::GpuArray<amrex::Real, 3> beta{}, gradient{};
        amrex::Real opacity = 0;
        amrex::Real spacing = 0.25;
        int normal = 0;
        bool equilibrium = true;
    };

    long double
    Pressure (FourVector const& radiation, int i, int j)
    {
        long double const energy = radiation[0];
        long double q2 = 0;
        for (int d = 1; d < 4; ++d) {
            long double const momentum = radiation[d];
            q2 += momentum * momentum;
        }
        if (q2 == 0) {
            return i == j ? energy / 3 : 0;
        }
        auto const f2 = q2 / (energy * energy);
        auto const chi = (3 + 4 * f2) / (5 + 2 * std::sqrt(4 - 3 * f2));
        return energy * ((i == j ? (1 - chi) / 2 : 0) +
                         (3 * chi - 1) / 2 * radiation[i + 1] * radiation[j + 1] / q2);
    }
} // namespace

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Gpu::HostVector<Case> cases;
        for (auto opacity : {0., 1.e-20, 1.e-6, 1., 1.e3, 1.e8, 1.e14, 1.e200}) {
            for (auto speed : {0., 1.e-3, 1.e-2, 0.2, 0.6}) {
                for (int normal = 0; normal < 3; ++normal) {
                    for (auto sign : {-1., 1.}) {
                        Case input;
                        input.opacity = opacity;
                        input.spacing = opacity == 1.e200 ? 1.e200 : 0.25;
                        input.normal = normal;
                        input.beta = {0.6 * sign * speed, -0.8 * sign * speed, 0};
                        input.gradient = {0.3, -0.2, 0.4};
                        long double b2 = 0;
                        for (auto b : input.beta) {
                            b2 += static_cast<long double>(b) * b;
                        }
                        auto const g2 = 1 / (1 - b2);
                        for (int side = 0; side < 2; ++side) {
                            auto& value = side == 0 ? input.left : input.right;
                            long double const rest = side == 0 ? 1.7L : 2.3L;
                            value[0] = static_cast<amrex::Real>(g2 * (1 + b2 / 3) * rest);
                            for (int d = 0; d < 3; ++d) {
                                value[d + 1] =
                                    static_cast<amrex::Real>(4 * g2 * input.beta[d] * rest / 3);
                            }
                        }
                        cases.push_back(input);
                        if (opacity == 0) {
                            input.equilibrium = false;
                            input.left = {2, 0.4, -0.1, 0.3};
                            input.right = {3, -0.2, 0.5, 0.1};
                            cases.push_back(input);
                        }
                    }
                }
            }
        }
        amrex::Gpu::DeviceVector<Case> device(cases.size());
        amrex::Gpu::DeviceVector<MovingMomentFluxResult> output(cases.size());
        amrex::Gpu::HostVector<MovingMomentFluxResult> results(cases.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, cases.begin(), cases.end(), device.begin());
        auto const* input = device.dataPtr();
        auto* result = output.dataPtr();
        amrex::ParallelFor(static_cast<int>(cases.size()), [=] AMREX_GPU_DEVICE(int i) {
            auto const& test = input[i];
            result[i] = EvaluateMovingMomentFlux(test.left, test.right, test.beta, test.normal,
                                                 test.opacity, test.spacing, test.gradient);
        });
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, output.begin(), output.end(), results.begin());
        long double worst = 0;
        for (std::size_t i = 0; i < cases.size(); ++i) {
            auto const& test = cases[i];
            auto const& measured = results[i];
            AMREX_ALWAYS_ASSERT(measured.valid);
            long double const c = PhysConst::c;
            long double b2 = 0;
            for (auto b : test.beta) {
                b2 += static_cast<long double>(b) * b;
            }
            auto const gamma = 1 / std::sqrt(1 - b2);
            long double const bn = test.beta[test.normal];
            auto const tau = 1.5L * test.opacity * gamma * test.spacing / (1 - bn * bn);
            auto const alpha = 1 / (1 + tau);
            auto const blend = tau / (1 + tau);
            long double expected_slow = 0;
            long double expected_diffusion = 0;
            if (test.equilibrium) {
                expected_diffusion = -c * alpha * (2.3L - 1.7L) / 2;
                if (test.opacity > 0) {
                    for (int d = 0; d < 3; ++d) {
                        if (d != test.normal) {
                            expected_diffusion += c * blend * blend * bn * test.beta[d] *
                                                  test.gradient[d] / (3 * test.opacity * gamma);
                        }
                    }
                }
                expected_slow =
                    c * (2 * bn - blend * std::abs(bn) * (2.3L - 1.7L) / 2) + expected_diffusion;
                if (std::abs(expected_diffusion) > 100 * std::numeric_limits<amrex::Real>::min()) {
                    worst = std::max(worst, std::abs(measured.projected_nonequilibrium_flux -
                                                     expected_diffusion) /
                                                std::abs(expected_diffusion));
                }
                worst =
                    std::max(worst, std::abs(measured.projected_flux - expected_slow) / (3 * c));
            }
            if (test.opacity == 0) {
                auto const energy_flux = c *
                                         (test.left[test.normal + 1] + test.right[test.normal + 1] -
                                          test.right[0] + test.left[0]) /
                                         2;
                worst = std::max(worst, std::abs(measured.flux[0] - energy_flux) / (3 * c));
            }
            for (int d = 0; d < 3; ++d) {
                auto const expected =
                    c *
                    (Pressure(test.left, d, test.normal) + Pressure(test.right, d, test.normal) -
                     alpha * (test.right[d + 1] - test.left[d + 1])) /
                    2;
                worst = std::max(worst, std::abs(measured.flux[d + 1] - expected) / (3 * c));
            }
            long double projected = measured.flux[0];
            for (int d = 0; d < 3; ++d) {
                projected -= static_cast<long double>(test.beta[d]) * measured.flux[d + 1];
            }
            worst = std::max(worst, std::abs(projected - measured.projected_flux) / (3 * c));
        }
        AMREX_ALWAYS_ASSERT(worst < 1.e-10L);
        AMREX_ALWAYS_ASSERT(
            !EvaluateMovingMomentFlux({1, 2, 0, 0}, {1, 0, 0, 0}, {}, 0, 1, 1).valid);
        AMREX_ALWAYS_ASSERT(
            !EvaluateMovingMomentFlux({1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0}, 0, 1, 1).valid);
        AMREX_ALWAYS_ASSERT(
            !EvaluateMovingMomentFlux({1, 0, 0, 0}, {1, 0, 0, 0}, {}, 0, -1, 1).valid);
        AMREX_ALWAYS_ASSERT(
            !EvaluateMovingMomentFlux({1, 0, 0, 0}, {1, 0, 0, 0}, {}, 3, 1, 1).valid);
        amrex::Print() << "Moving face flux: " << cases.size()
                       << " cases; worst independent flux error=" << static_cast<double>(worst)
                       << '\n';
    }
    amrex::Finalize();
}
