/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/MovingInterfaceAngularExchange.H"

#include <cmath>
#include <iostream>
#include <limits>

namespace
{
int failures = 0;
void check (bool condition, char const* description)
{
    if (!condition) { ++failures; std::cerr << description << '\n'; }
}
bool near (long double a, long double b)
{
    return std::abs(a - b) < 2048 * std::numeric_limits<amrex::ParticleReal>::epsilon() *
        std::max(1.0L, std::max(std::abs(a), std::abs(b)));
}
}

int main ()
{
    using namespace warpx::radiation;
    using V3 = amrex::GpuArray<amrex::Real, 3>;
    FourVector const incoming{1, .3, .4, std::sqrt(.75)};
    amrex::Real const nominal = PhysConst::m_e * PhysConst::c * PhysConst::c;
    std::vector<InterfaceAngularRay> rays{{V3{.6, 0, -.8}, .25},
                                         {V3{-.6, 0, -.8}, .75}};
    for (V3 const beta : {V3{0, 0, 0}, V3{0, 0, .001}, V3{.002, -.003, .001}})
    {
        auto const result = TryMovingInterfaceAngularExchange(incoming, beta, rays, nominal, 2, -1);
        check(result.valid && result.rays.size() == 2, "valid angular exchange");
        if (!result.valid) { continue; }
        long double actual[4]{};
        for (auto const& ray : result.rays)
        {
            long double const factor = static_cast<long double>(ray.weight) * PhysConst::m_e *
                                      PhysConst::c;
            actual[0] += factor * std::hypot(static_cast<long double>(ray.u[0]),
                static_cast<long double>(ray.u[1]), static_cast<long double>(ray.u[2]));
            for (int d = 0; d < 3; ++d) { actual[d + 1] += factor * ray.u[d]; }
        }
        check(near(result.radiation_debit, incoming[0] - actual[0]), "actual energy debit");
        long double work = 0;
        for (int d = 0; d < 3; ++d)
        {
            check(near(result.material_impulse[d] * PhysConst::c,
                       incoming[d + 1] - actual[d + 1]), "actual xyz recoil");
            work += beta[d] * (incoming[d + 1] - actual[d + 1]);
        }
        check(near(result.radiation_debit, work), "secant frame work up to storage error");
        V3 const inverse{-beta[0], -beta[1], -beta[2]};
        auto const frame = BoostFourVector(incoming, inverse);
        auto const roundtrip = BoostFourVector(frame, beta);
        for (int k = 0; k < 4; ++k) { check(near(incoming[k], roundtrip[k]), "boost roundtrip"); }
        FourVector expected{};
        for (auto const& ray : rays)
        {
            FourVector const outgoing{frame[0] * ray.fraction,
                frame[0] * ray.fraction * ray.frame_direction[0],
                frame[0] * ray.fraction * ray.frame_direction[1],
                frame[0] * ray.fraction * ray.frame_direction[2]};
            auto const lab = BoostFourVector(outgoing, beta);
            for (int k = 0; k < 4; ++k) { expected[k] += lab[k]; }
        }
        for (int k = 0; k < 4; ++k) { check(near(actual[k], expected[k]), "independent boost sum"); }
    }
    auto rejected = [&] (FourVector photon, V3 beta, std::vector<InterfaceAngularRay> q,
                        amrex::Real energy = 1, int axis = 2, int sign = -1)
    {
        auto const r = TryMovingInterfaceAngularExchange(photon, beta, q, energy, axis, sign);
        bool empty = !r.valid && r.rays.empty() && r.radiation_debit == 0;
        for (auto value : r.material_impulse) { empty = empty && value == 0; }
        for (auto value : r.outgoing_total) { empty = empty && value == 0; }
        check(empty, "atomic rejection");
    };
    rejected(FourVector{1, 0, 0, 0}, V3{}, rays);
    rejected(incoming, V3{0, 0, 1}, rays);
    rejected(incoming, V3{std::numeric_limits<amrex::Real>::quiet_NaN(), 0, 0}, rays);
    rejected(incoming, V3{}, {});
    rejected(incoming, V3{}, std::vector<InterfaceAngularRay>(65, {V3{0, 0, -1}, 1./65}));
    rejected(incoming, V3{}, {{V3{0, 0, -2}, 1}});
    rejected(incoming, V3{}, {{V3{0, 0, -1}, -1}});
    rejected(incoming, V3{}, {{V3{0, 0, -1}, .5}});
    rejected(incoming, V3{}, rays, 0);
    rejected(incoming, V3{}, rays, std::numeric_limits<amrex::Real>::max());
    rejected(incoming, V3{}, rays, nominal, 3);
    rejected(incoming, V3{}, rays, nominal, 2, 0);
    rejected(incoming, V3{0, 0, .9}, {{V3{.6, 0, -.8}, 1}}, nominal);
    check(TryMovingInterfaceAngularExchange(incoming, V3{}, {{V3{0, 0, 1}, 1}},
        nominal, 2, 1).valid, "positive lab hemisphere");
    check(incoming[1] == .3 && rays[0].fraction == .25, "inputs unchanged");
    return failures == 0 ? 0 : 1;
}
