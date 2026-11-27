/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */

#include "Radiation/QuasiStaticInterfaceTraction.H"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace
{

using warpx::radiation::QuasiStaticInterfaceFace;
using warpx::radiation::QuasiStaticInterfaceFaceKind;
using warpx::radiation::QuasiStaticInterfaceStatus;

bool near_enough (double actual, double expected)
{
    double const scale = std::max(std::abs(actual), std::abs(expected));
    return std::abs(actual-expected)
        <= 64.0*std::numeric_limits<double>::epsilon()*scale;
}

void check_vector (std::array<double, 3> const& actual,
                   std::array<double, 3> const& expected)
{
    for (int d = 0; d < 3; ++d)
    {
        if (!near_enough(actual[d], expected[d]))
        {
            std::fprintf(stderr, "component=%d actual=%.17g expected=%.17g\n",
                d, actual[d], expected[d]);
            std::abort();
        }
    }
}

QuasiStaticInterfaceFace face (std::uint64_t id, int side,
    QuasiStaticInterfaceFaceKind kind, std::array<double, 4> incoming,
    std::array<double, 4> outgoing, double pressure, double area)
{
    return {id, side, kind, incoming, outgoing, pressure, area};
}

} // namespace

int main ()
{
    namespace qsi = warpx::radiation;

    // Two half-space inventories with mean |mu|=2/3 give one scalar pressure
    // face each. Every factor is deliberately different from one.
    {
        double const dt = 0.13;
        double const c = 0.29;
        double const area = 2.1;
        double const pressure = 0.71;
        double const energy = 0.75*c*pressure*dt*area;
        double const normal_momentum = (2.0/3.0)*energy;
        std::vector<QuasiStaticInterfaceFace> faces{
            face(10, -1, QuasiStaticInterfaceFaceKind::RayBoundary,
                 {energy, 0.0, 0.0, normal_momentum},
                 {energy, 0.0, 0.0, -normal_momentum}, 0.0, area),
            face(11, 1, QuasiStaticInterfaceFaceKind::RayBoundary,
                 {energy, 0.0, 0.0, -normal_momentum},
                 {energy, 0.0, 0.0, normal_momentum}, 0.0, area)};
        auto const traction = qsi::ComputeQuasiStaticInterfaceTraction(faces, dt, c);
        if (!traction.valid || traction.status != QuasiStaticInterfaceStatus::Valid)
        {
            std::abort();
        }
        // Equal scalar pressure on opposite faces has zero surface divergence.
        check_vector(traction.outer_ray_traction, {0.0, 0.0, 0.0});
        check_vector(traction.inner_pressure_traction, {0.0, 0.0, 0.0});
        check_vector(traction.material_impulse, {0.0, 0.0, 0.0});

        // An angular face must also match a scalar face, not just cancel its
        // own duplicate; this independently fixes the hemisphere normalization.
        faces[1].kind = QuasiStaticInterfaceFaceKind::IsotropicInterior;
        faces[1].ray_incoming = {};
        faces[1].ray_outgoing = {};
        faces[1].interior_pressure = pressure;
        auto const mixed = qsi::ComputeQuasiStaticInterfaceTraction(faces, dt, c);
        if (!mixed.valid) { std::abort(); }
        double const stress = pressure*dt*area;
        check_vector(mixed.outer_ray_traction, {0.0, 0.0, -stress});
        check_vector(mixed.inner_pressure_traction, {0.0, 0.0, stress});
        if (std::abs(mixed.material_impulse[2]) >
            64.0*std::numeric_limits<double>::epsilon()*stress) { std::abort(); }
    }

    // Oblique left and normal right crossings have exact 3-2-4 and 5-12-13
    // direction cosines. The analytic result contains no dt or area multiplier.
    {
        double const c = 0.4;
        std::vector<QuasiStaticInterfaceFace> faces{
            face(20, -1, QuasiStaticInterfaceFaceKind::RayBoundary,
                 {10.0, 6.0, 0.0, 8.0}, {13.0, -5.0, 0.0, -12.0}, 0.0, 7.0),
            face(21, 1, QuasiStaticInterfaceFaceKind::RayBoundary,
                 {5.0, 0.0, 0.0, -5.0}, {11.0, 0.0, 0.0, 11.0}, 0.0, 3.0)};
        auto const traction = qsi::ComputeQuasiStaticInterfaceTraction(
            faces, 0.37, c);
        if (!traction.valid) { std::abort(); }
        check_vector(traction.outer_ray_traction, {-27.5, 0.0, -10.0});
        check_vector(traction.inner_pressure_traction, {0.0, 0.0, 0.0});
        check_vector(traction.material_impulse, {27.5, 0.0, 10.0});
    }

    // The same shared interior face has opposite orientation in adjacent calls.
    {
        double const dt = 0.23;
        double const area = 1.9;
        double const pressure = 3.7;
        std::array<double, 4> const none{0.0, 0.0, 0.0, 0.0};
        std::vector<QuasiStaticInterfaceFace> left_cell{
            face(30, -1, QuasiStaticInterfaceFaceKind::RayBoundary,
                 none, none, 0.0, area),
            face(31, 1, QuasiStaticInterfaceFaceKind::IsotropicInterior,
                 none, none, pressure, area)};
        std::vector<QuasiStaticInterfaceFace> right_cell{
            face(31, -1, QuasiStaticInterfaceFaceKind::IsotropicInterior,
                 none, none, pressure, area),
            face(32, 1, QuasiStaticInterfaceFaceKind::RayBoundary,
                 none, none, 0.0, area)};
        auto const a = qsi::ComputeQuasiStaticInterfaceTraction(left_cell, dt, 0.6);
        auto const b = qsi::ComputeQuasiStaticInterfaceTraction(right_cell, dt, 0.6);
        if (!a.valid || !b.valid) { std::abort(); }
        double const shared = 1.6169; // 3.7*0.23*1.9
        check_vector(a.inner_pressure_traction, {0.0, 0.0, shared});
        check_vector(b.inner_pressure_traction, {0.0, 0.0, -shared});
        check_vector(a.material_impulse, {0.0, 0.0, -shared});
        check_vector(b.material_impulse, {0.0, 0.0, shared});
        for (int d = 0; d < 3; ++d)
        {
            if (a.inner_pressure_traction[d] + b.inner_pressure_traction[d] != 0.0 ||
                a.material_impulse[d] + b.material_impulse[d] != 0.0)
            {
                std::abort();
            }
        }
    }

    // A RayBoundary and IsotropicInterior face coexist with nonunit factors.
    {
        double const dt = 0.17;
        double const c = 0.53;
        double const area = 2.3;
        double const pressure = 5.7;
        std::vector<QuasiStaticInterfaceFace> faces{
            face(40, -1, QuasiStaticInterfaceFaceKind::RayBoundary,
                 {1.0, 0.0, 0.0, 1.0}, {0.5, 0.0, 0.0, -0.5}, 0.0, area),
            face(41, 1, QuasiStaticInterfaceFaceKind::IsotropicInterior,
                 {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, pressure, area)};
        auto const traction = qsi::ComputeQuasiStaticInterfaceTraction(faces, dt, c);
        if (!traction.valid) { std::abort(); }
        double const ray_z = -1.5/0.53;
        double const pressure_z = 5.7*0.17*2.3;
        check_vector(traction.outer_ray_traction, {0.0, 0.0, ray_z});
        check_vector(traction.inner_pressure_traction, {0.0, 0.0, pressure_z});
        check_vector(traction.material_impulse, {0.0, 0.0, -(ray_z + pressure_z)});
    }

    auto rejected = [] (std::vector<QuasiStaticInterfaceFace> const& faces,
                        double dt, double c,
                        QuasiStaticInterfaceStatus expected = QuasiStaticInterfaceStatus::Valid)
    {
        auto const traction = qsi::ComputeQuasiStaticInterfaceTraction(faces, dt, c);
        if (traction.valid || traction.status == QuasiStaticInterfaceStatus::Valid)
        {
            std::abort();
        }
        if (expected != QuasiStaticInterfaceStatus::Valid && traction.status != expected)
        {
            std::abort();
        }
        check_vector(traction.material_impulse, {0.0, 0.0, 0.0});
        check_vector(traction.outer_ray_traction, {0.0, 0.0, 0.0});
        check_vector(traction.inner_pressure_traction, {0.0, 0.0, 0.0});
    };

    auto valid_face = [&] (std::uint64_t id, int side)
    {
        return face(id, side, QuasiStaticInterfaceFaceKind::RayBoundary,
                    {1.0, 0.0, 0.0, -static_cast<double>(side)},
                    {1.0, 0.0, 0.0, static_cast<double>(side)}, 0.0, 1.0);
    };
    std::vector<QuasiStaticInterfaceFace> pair{valid_face(50, -1), valid_face(51, 1)};
    if (!qsi::ComputeQuasiStaticInterfaceTraction(pair, 1.0, 1.0).valid)
    {
        std::abort();
    }
    auto bad = pair;

    rejected({}, 1.0, 1.0);
    rejected({pair[0]}, 1.0, 1.0);
    rejected({pair[0], pair[1], valid_face(52, -1)}, 1.0, 1.0);
    rejected({valid_face(60, -1), valid_face(60, 1)}, 1.0, 1.0);
    rejected({valid_face(61, 1), valid_face(62, 1)}, 1.0, 1.0);
    bad = pair;
    bad[1].side = 0;
    rejected(bad, 1.0, 1.0);
    rejected(pair, 0.0, 1.0);
    rejected(pair, -1.0, 1.0);
    rejected(pair, std::numeric_limits<double>::quiet_NaN(), 1.0);
    rejected(pair, 1.0, 0.0);
    rejected(pair, 1.0, -1.0);
    rejected(pair, 1.0, std::numeric_limits<double>::infinity());

    auto reject_area = [&] (double area)
    {
        auto bad_area = pair;
        bad_area[1].face_area = area;
        rejected(bad_area, 1.0, 1.0);
    };
    reject_area(0.0);
    reject_area(-1.0);
    reject_area(std::numeric_limits<double>::quiet_NaN());

    bad = pair;
    bad[1].kind = QuasiStaticInterfaceFaceKind::IsotropicInterior;
    bad[1].interior_pressure = -0.1;
    rejected(bad, 1.0, 1.0);
    bad[1].interior_pressure = std::numeric_limits<double>::quiet_NaN();
    rejected(bad, 1.0, 1.0);

    for (int component = 0; component < 4; ++component)
    {
        for (bool const outgoing_inventory : {false, true})
        {
            auto nonfinite = pair;
            (outgoing_inventory ? nonfinite[0].ray_outgoing
                                : nonfinite[0].ray_incoming)[component]
                = std::numeric_limits<double>::quiet_NaN();
            rejected(nonfinite, 1.0, 1.0);
        }
    }

    bad = pair;
    bad[0].ray_outgoing = {1.0, 2.0, 0.0, -1.0};
    rejected(bad, 1.0, 1.0); // aggregate momentum lies outside the photon cone
    bad = pair;
    bad[0].ray_incoming = {1.0, 0.0, 0.0, -1.0};
    rejected(bad, 1.0, 1.0); // left incoming photon points out of the cell
    bad = pair;
    bad[1].ray_outgoing = {1.0, 0.0, 0.0, -1.0};
    rejected(bad, 1.0, 1.0); // right outgoing photon points into the cell

    bad = pair;
    bad[0].kind = static_cast<QuasiStaticInterfaceFaceKind>(99);
    rejected(bad, 1.0, 1.0, QuasiStaticInterfaceStatus::InvalidFaceKind);
    bad = pair;
    bad[0].ray_incoming[0] = -1.0;
    rejected(bad, 1.0, 1.0, QuasiStaticInterfaceStatus::NegativeRayEnergy);

    // A bright opposite stream must not hide a wrong-sign dim stream.
    bad = pair;
    bad[0].ray_incoming = {1.0, 0.0, 0.0, -1.0};
    bad[0].ray_outgoing = {1.e100, 0.0, 0.0, -1.e100};
    rejected(bad, 1.0, 1.0, QuasiStaticInterfaceStatus::IncomingOrientationViolation);
    bad = pair;
    bad[0].ray_incoming = {1.e100, 0.0, 0.0, 1.e100};
    bad[0].ray_outgoing = {1.0, 0.0, 0.0, 1.0};
    rejected(bad, 1.0, 1.0, QuasiStaticInterfaceStatus::OutgoingOrientationViolation);

    double const maximum = std::numeric_limits<double>::max();
    bad = pair;
    bad[0].ray_incoming = {maximum, maximum, maximum, 0.0};
    rejected(bad, 1.0, 1.0, QuasiStaticInterfaceStatus::PhotonConeViolation);
    bad = pair;
    bad[0].ray_incoming = {maximum, 0.0, 0.0, maximum};
    bad[0].ray_outgoing = {maximum, 0.0, 0.0, -maximum};
    rejected(bad, 1.0, 1.0, QuasiStaticInterfaceStatus::NonfiniteTraction);
    bad = pair;
    bad[0].kind = QuasiStaticInterfaceFaceKind::IsotropicInterior;
    bad[0].interior_pressure = maximum;
    rejected(bad, 2.0, 1.0, QuasiStaticInterfaceStatus::NonfiniteTraction);

    pair[0].ray_incoming = {13.0, 3.0, 4.0, 12.0};
    pair[0].ray_outgoing = {13.0, -3.0, -4.0, -12.0};
    // A ray face owns the traction exclusively, even when a caller also
    // supplies a finite scalar pressure. That pressure must not be added.
    pair[0].interior_pressure = 1.e80;
    pair[1].ray_incoming = {};
    pair[1].ray_outgoing = {};
    auto const full_vector = qsi::ComputeQuasiStaticInterfaceTraction(pair, 0.7, 2.0);
    if (!full_vector.valid) { std::abort(); }
    check_vector(full_vector.material_impulse, {3.0, 4.0, 12.0});

    return 0;
}
