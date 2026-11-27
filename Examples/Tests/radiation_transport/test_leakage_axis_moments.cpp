/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/MovingLeakageAngularQuadrature.H"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>

int main ()
{
    int failures = 0;
    auto check = [&failures](bool condition, char const* message) {
        if (!condition) { ++failures; std::cerr << message << '\n'; }
    };
    auto moments = [&check](std::array<double, 3> const& beta, int sign, int axis) {
        auto const quadrature = warpx::radiation::ComputeMovingLeakageAngularQuadrature(
            beta, sign, axis);
        check(quadrature.valid, "oriented quadrature rejected");
        std::array<long double, 3> mean{};
        long double sum = 0;
        for (int ray = 0; ray < 8; ++ray) {
            auto const& direction = quadrature.direction[ray];
            check(sign*direction[axis] > 0, "ray points into emitting material");
            sum += quadrature.fraction[ray];
            for (int component = 0; component < 3; ++component) {
                mean[component] += quadrature.fraction[ray]*direction[component];
            }
        }
        check(std::abs(sum-1) <= 16*std::numeric_limits<long double>::epsilon(),
              "oriented energy is not normalized");
        return mean;
    };
    for (int axis = 0; axis < 3; ++axis) {
        for (int sign : {-1, 1}) {
            auto const rest = moments({0,0,0}, sign, axis);
            for (int component = 0; component < 3; ++component) {
                auto const expected = component == axis ? sign*2.L/3 : 0.L;
                check(std::abs(rest[component]-expected) <
                          32*std::numeric_limits<long double>::epsilon(),
                      "rest momentum rotates incorrectly");
            }
            // The continuum cosine hemisphere has covariance 1/18 along its
            // normal, 1/4 tangentially, and zero off-diagonal. Differentiating
            // normalized D^4 intensity therefore gives diag(2/9,1,1).
            for (int boost = 0; boost < 3; ++boost) {
                constexpr double h = 5.e-5;
                std::array<double, 3> positive{}, negative{};
                positive[boost] = h;
                negative[boost] = -h;
                auto const plus = moments(positive, sign, axis);
                auto const minus = moments(negative, sign, axis);
                long double const bound = 2200.L*h*h/6 +
                    256*std::numeric_limits<long double>::epsilon()/h;
                for (int component = 0; component < 3; ++component) {
                    auto const expected = component != boost ? 0.L :
                        (component == axis ? 2.L/9 : 1.L);
                    check(std::abs((plus[component]-minus[component])/(2*h)-expected)
                              <= bound,
                          "normal or tangential continuum response is incorrectly oriented");
                }
            }
        }
    }
    return failures == 0 ? 0 : 1;
}
