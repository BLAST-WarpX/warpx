/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/RZSpecularTrajectory.H"

#include <AMReX.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        using warpx::radiation::ReflectRZTrajectory;
        using Real = amrex::Real;
        Real const eps = std::numeric_limits<Real>::epsilon();
        // Include near-grazing rays, genuine multiple collisions and small SI
        // radii.
        for (Real radius : {Real(1), Real(0.001)})
        {
            for (Real impact : {Real(0), Real(0.3), Real(0.99)})
            {
                for (Real distance : {Real(0.01), Real(0.5), Real(3)})
                {
                    amrex::GpuArray<Real, 2> const previous{0, impact * radius};
                    amrex::GpuArray<Real, 2> const trial{distance * radius, previous[1]};
                    auto const path = ReflectRZTrajectory(previous, trial, radius, 32);
                    AMREX_ALWAYS_ASSERT(path.valid);
                    // Incoming velocity (1,0); reflected velocity is first
                    // matrix column.
                    auto const vx = path.reflection[0], vy = path.reflection[2];
                    Real const old_l = -previous[1];
                    Real const new_l = path.position[0] * vy - path.position[1] * vx;
                    AMREX_ALWAYS_ASSERT(std::abs(new_l - old_l) <= 1024 * eps * radius);
                    AMREX_ALWAYS_ASSERT(std::abs(vx * vx + vy * vy - 1) <= 1024 * eps);
                    auto const reversed =
                        ReflectRZTrajectory(path.position,
                                            {path.position[0] - distance * radius * vx,
                                             path.position[1] - distance * radius * vy},
                                            radius, 32);
                    AMREX_ALWAYS_ASSERT(reversed.valid);
                    for (int d = 0; d < 2; ++d)
                    {
                        AMREX_ALWAYS_ASSERT(std::abs(reversed.position[d] - previous[d]) <=
                                            8192 * eps * radius);
                    }
                    Real const radius_squared =
                        path.position[0] * path.position[0] + path.position[1] * path.position[1];
                    AMREX_ALWAYS_ASSERT(radius_squared <= radius * radius * (1 + 8 * eps));
                    // The same orthogonal map must preserve arbitrary
                    // pending-vector norm.
                    auto const cx = 2 * path.reflection[0] - 3 * path.reflection[1];
                    auto const cy = 2 * path.reflection[2] - 3 * path.reflection[3];
                    AMREX_ALWAYS_ASSERT(std::abs(cx * cx + cy * cy - 13) <= 16384 * eps);
                }
            }
        }
        auto const normal = ReflectRZTrajectory({0, 0}, {Real(1.2), 0}, 1);
        AMREX_ALWAYS_ASSERT(normal.valid && normal.collisions == 1);
        AMREX_ALWAYS_ASSERT(std::abs(normal.position[0] - Real(0.8)) < 8 * eps);
        AMREX_ALWAYS_ASSERT(normal.reflection[0] == -1 && normal.reflection[3] == 1);
        AMREX_ALWAYS_ASSERT(!ReflectRZTrajectory({0, 0}, {3, 0}, 1, 0).valid);
        AMREX_ALWAYS_ASSERT(!ReflectRZTrajectory({2, 0}, {3, 0}, 1).valid);
        AMREX_ALWAYS_ASSERT(!ReflectRZTrajectory({0, 0}, {1, 0}, 0).valid);
        AMREX_ALWAYS_ASSERT(
            !ReflectRZTrajectory({0, 0}, {std::numeric_limits<Real>::quiet_NaN(), 0}, 1).valid);
        amrex::Print() << "Circular collision-point trajectories: angular/work "
                          "invariants passed.\n";
    }
    amrex::Finalize();
}
