/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/MovingLeakageAngularQuadrature.H"
#include <iostream>
#include <limits>

using warpx::radiation::ComputeMovingLeakageAngularQuadrature;
namespace
{
int failures = 0;
void check (bool ok, char const* message)
{
    if (!ok) { ++failures; std::cerr << message << '\n'; }
}
long double mean (std::array<double, 3> beta, int sign, int component)
{
    auto const q = ComputeMovingLeakageAngularQuadrature(beta, sign);
    check(q.valid, "valid quadrature");
    long double value = 0, sum = 0;
    for (int n = 0; n < 8; ++n) {
        check(q.fraction[n] > 0 && sign*q.direction[n][2] > 0, "positive outward rays");
        value += q.fraction[n]*q.direction[n][component];
        sum += q.fraction[n];
    }
    check(std::abs(sum-1) <= 16*std::numeric_limits<long double>::epsilon(), "normalization");
    return value;
}
}
int main ()
{
    long double const eps = std::numeric_limits<long double>::epsilon();
    for (int sign : {-1,1}) {
        auto const rest = ComputeMovingLeakageAngularQuadrature({0,0,0},sign);
        constexpr long double root6 = 2.4494897427831780981972840747058913919659474806567L;
        check(rest.fraction[0] == ((9-root6)/18)/4 &&
              rest.fraction[4] == ((9+root6)/18)/4, "original rest quadrature weights");
        check(std::abs(mean({0,0,0},sign,2)-sign*2.L/3) < 16*eps, "rest normal moment");
        check(mean({0,0,0},sign,0) == 0, "rest tangential moment");
        for (double h : {1.e-4, 5.e-5}) {
            // For weights (1-beta.n)^-4 at |beta|<=.001, raw derivatives up to
            // order3 are bounded by 1.005,4.021,20.13,120.85. Quotient-rule
            // Z>.996, |mean|<=1; quotient recursion bounds derivatives by
            // 8.1,106,2200. Central difference error <=2200*h²/6.
            long double const bound = 2200.L*h*h/6 + 256*eps/h;
            auto const tangent = (mean({h,0,0},sign,0)-mean({-h,0,0},sign,0))/(2*h);
            auto const normal = (mean({0,0,h},sign,2)-mean({0,0,-h},sign,2))/(2*h);
            check(std::abs(tangent-1) <= bound, "continuum tangential derivative 1");
            check(std::abs(normal-2.L/9) <= bound, "continuum normal derivative 2/9");
        }
    }
    check(std::abs(mean({.0002,-.0003,.0004},1,2)+
                   mean({.0002,-.0003,-.0004},-1,2)) < 64*eps, "normal reflection symmetry");
    check(!ComputeMovingLeakageAngularQuadrature({.0011,0,0},1).valid, "speed bound");
    check(!ComputeMovingLeakageAngularQuadrature({0,0,0},0).valid, "invalid sign");
    check(!ComputeMovingLeakageAngularQuadrature(
        {std::numeric_limits<double>::quiet_NaN(),0,0},1).valid, "invalid frame");
    return failures ? 1 : 0;
}
