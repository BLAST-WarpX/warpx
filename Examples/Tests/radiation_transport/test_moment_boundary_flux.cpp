/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/MomentBoundaryFlux.H"
#include "Radiation/MovingMomentFlux.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>

using namespace warpx::radiation;

namespace {
struct Case {
    FourVector state{};
    int normal = 0;
    int side = 1;
};
struct Result {
    MomentBoundaryFluxResult outgoing, opposite, mirror;
    MomentMirrorJacobianResult jacobian;
    FourVector finite_difference{};
};
} // namespace

int
main (int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    {
        amrex::Gpu::HostVector<Case> cases;
        for (double energy : {1.e-200, 1., 1.e200}) {
            for (double f :
                 {0., 0.2, 0.7, 0.95, 1 - 1.e-6, 1 - 1.e-12, 1 - 1.e-15, 1.}) {
                if (f > 0.95 && f < 1 && energy != 1) {
                    continue;
                }
                for (int orientation = 0; orientation < (f <= 0.95 ? 4 : 1);
                     ++orientation) {
                    FourVector state{energy, f * energy, 0, 0};
                    if (orientation == 1) {
                        state = {energy, 0.6 * f * energy, 0.8 * f * energy, 0};
                    }
                    if (orientation == 2) {
                        state = {energy, -0.6 * f * energy, 0.8 * f * energy,
                                 0};
                    }
                    if (orientation == 3) {
                        state = {energy, 0.3 * f * energy, -0.4 * f * energy,
                                 std::sqrt(0.75) * f * energy};
                    }
                    for (int normal = 0; normal < 3; ++normal) {
                        for (int side : {-1, 1}) {
                            cases.push_back({state, normal, side});
                        }
                    }
                }
            }
        }
        cases.push_back({{0, 0, 0, 0}, 0, 1});
        auto const collar_case = cases.size();
        auto const epsilon = std::numeric_limits<amrex::Real>::epsilon();
        cases.push_back({{1, 1 + 2 * epsilon, 0, 0}, 0, 1});
        auto const valid_count = cases.size();
        cases.push_back({{1, 1 + 64 * epsilon, 0, 0}, 0, 1});
        cases.push_back({{-1, 0, 0, 0}, 0, 1});
        cases.push_back({{1, 2, 0, 0}, 0, 1});
        cases.push_back({{0, 1, 0, 0}, 0, 1});
        cases.push_back(
            {{std::numeric_limits<amrex::Real>::infinity(), 0, 0, 0}, 0, 1});
        cases.push_back({{1, 0, 0, 0}, 3, 1});
        cases.push_back({{1, 0, 0, 0}, 0, 0});
        cases.push_back(
            {{std::numeric_limits<amrex::Real>::quiet_NaN(), 0, 0, 0}, 0, 1});
        amrex::Gpu::DeviceVector<Case> device_cases(cases.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, cases.begin(), cases.end(),
                         device_cases.begin());
        amrex::Gpu::DeviceVector<Result> device_results(cases.size());
        auto const* input = device_cases.data();
        auto* output = device_results.data();
        amrex::ParallelFor(
            static_cast<int>(cases.size()), [=] AMREX_GPU_DEVICE(int i) {
                auto const& test = input[i];
                output[i] = {
                    EvaluateM1OutgoingFlux(test.state, test.normal, test.side),
                    EvaluateM1OutgoingFlux(test.state, test.normal, -test.side),
                    EvaluateM1MirrorFlux(test.state, test.normal, test.side),
                    EvaluateM1MirrorJacobian(test.state, test.normal, test.side)};
                if (output[i].jacobian.valid && test.state[0] > 0) {
                    auto const f = std::hypot(std::hypot(test.state[1], test.state[2]),
                                              test.state[3]) / test.state[0];
                    if (f < 0.951) {
                        auto const step = 1.e-5 * test.state[0];
                        for (int column = 0; column < 4; ++column) {
                            auto plus = test.state;
                            auto minus = test.state;
                            plus[column] += step;
                            minus[column] -= step;
                            auto const upper = EvaluateM1MirrorFlux(plus, test.normal, test.side);
                            auto const lower = EvaluateM1MirrorFlux(minus, test.normal, test.side);
                            output[i].finite_difference[column] =
                                (upper.flux[test.normal + 1] - lower.flux[test.normal + 1]) /
                                (2 * step);
                        }
                    }
                }
            });
        amrex::Gpu::HostVector<Result> results(cases.size());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_results.begin(),
                         device_results.end(), results.begin());
        bool const writer = amrex::ParallelDescriptor::IOProcessor();
        std::ofstream data;
        if (writer) {
            data.open("boundary_flux.txt");
        }
        data.precision(17);
        for (std::size_t i = 0; i < cases.size(); ++i) {
            auto const& test = cases[i];
            auto const& result = results[i];
            bool const expected_valid = i < valid_count;
            AMREX_ALWAYS_ASSERT(result.outgoing.valid == expected_valid);
            AMREX_ALWAYS_ASSERT(result.opposite.valid == expected_valid);
            AMREX_ALWAYS_ASSERT(result.mirror.valid == expected_valid);
            AMREX_ALWAYS_ASSERT(result.jacobian.valid == expected_valid);
            if (i == collar_case) {
                AMREX_ALWAYS_ASSERT(result.outgoing.flux[0] ==
                                    PhysConst::c * test.state[1]);
            }
            if (!expected_valid || test.state[0] == 0) {
                continue;
            }
            auto const closure = EvaluateM1Closure(test.state);
            auto const scale =
                static_cast<long double>(PhysConst::c) * test.state[0];
            long double euler_sum = 0;
            auto const f = std::hypot(std::hypot(test.state[1], test.state[2]),
                                      test.state[3]) / test.state[0];
            for (int column = 0; column < 4; ++column) {
                euler_sum += static_cast<long double>(result.jacobian.derivative[column]) *
                             test.state[column];
                if (f < 0.951) {
                    AMREX_ALWAYS_ASSERT(std::abs(result.jacobian.derivative[column] -
                                                result.finite_difference[column]) /
                                        PhysConst::c < 2.e-8);
                }
            }
            AMREX_ALWAYS_ASSERT(std::abs(euler_sum - result.mirror.flux[test.normal + 1]) /
                                scale < 2.e-13L);
            if (test.state[0] == 1 && f > 0.99 && f < 1 &&
                test.normal == 0 && test.side == -1) {
                // Independent axial backward-pressure derivative, retaining
                // relative accuracy in a tiny nonzero tail near the light cone.
                long double const axial_f = test.state[1];
                auto const root = std::sqrt(4 - 3 * axial_f * axial_f);
                auto const b = 3 * axial_f / (2 + root);
                auto const delta = 12 * (1 - axial_f) * (1 + axial_f) /
                                   ((root + 1) * (root + 2));
                auto const gap = delta / (1 + b);
                auto const denominator = 3 + b * b;
                auto const h = gap * gap * gap / denominator;
                auto const bp = 3 / (2 + root) + 9 * axial_f * axial_f /
                                (root * (2 + root) * (2 + root));
                auto const positive_slope = bp * (3 * gap * gap + 2 * b * h) / denominator;
                auto const expected_q = PhysConst::c * positive_slope;
                auto const expected_e = -PhysConst::c * (h + axial_f * positive_slope);
                AMREX_ALWAYS_ASSERT(
                    std::abs(result.jacobian.derivative[1] / expected_q - 1) < 1.e-10L);
                AMREX_ALWAYS_ASSERT(
                    std::abs(result.jacobian.derivative[0] / expected_e - 1) < 1.e-10L);
            }
            for (int d = 0; d < 4; ++d) {
                long double const full =
                    PhysConst::c * static_cast<long double>(test.side) *
                    (d == 0 ? test.state[test.normal + 1]
                            : closure.tensor[d][test.normal + 1]);
                AMREX_ALWAYS_ASSERT(std::abs((result.outgoing.flux[d] -
                                              result.opposite.flux[d] - full) /
                                             scale) < 2.e-13L);
                if (d != test.normal + 1) {
                    AMREX_ALWAYS_ASSERT(result.mirror.flux[d] == 0);
                }
            }
            AMREX_ALWAYS_ASSERT(result.outgoing.flux[0] >= 0);
            AMREX_ALWAYS_ASSERT(
                test.side * result.outgoing.flux[test.normal + 1] >= 0);
            AMREX_ALWAYS_ASSERT(result.mirror.flux[test.normal + 1] ==
                                2 * result.outgoing.flux[test.normal + 1]);
            if (!writer) {
                continue;
            }
            for (auto value : test.state) {
                data << value << ' ';
            }
            data << test.normal << ' ' << test.side;
            for (auto value : result.outgoing.flux) {
                data << ' ' << value / scale;
            }
            data << '\n';
        }
        AMREX_ALWAYS_ASSERT(!writer || data.good());
        // A generic dissipative interior flux is not a positive wall-pressure
        // closure for radiation moving away from a fixed mirror.
        auto const artificial_wall = EvaluateMovingMomentFlux(
            {1, -0.7, 0, 0}, {1, 0.7, 0, 0}, {}, 0, 0, 1);
        auto const physical_wall = EvaluateM1MirrorFlux({1, 0.7, 0, 0}, 0, -1);
        AMREX_ALWAYS_ASSERT(artificial_wall.valid && artificial_wall.flux[1] < 0);
        AMREX_ALWAYS_ASSERT(physical_wall.valid && physical_wall.flux[1] < 0);
        // Here coordinate flux < 0 is outward-positive pressure on the low wall;
        // the artificial interior face is expressed in the +x direction and
        // therefore has the opposite (unphysical tensile) pressure sign.
        amrex::Print() << cases.size() << " boundary kernel cases passed\n";
    }
    amrex::Finalize();
}
