/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/MomentBoundaryExchange.H"

#include <AMReX.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <limits>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        using warpx::radiation::ComputeMomentBoundaryExchange;
        using warpx::radiation::FourVector;
        amrex::Box domain(amrex::IntVect(0), amrex::IntVect(7));
        amrex::RealBox bounds({AMREX_D_DECL(0., 0., 0.)}, {AMREX_D_DECL(2., 3., 4.)});
        int periodic[AMREX_SPACEDIM] = {AMREX_D_DECL(0, 0, 0)};
        amrex::Geometry geometry(domain, &bounds, 0, periodic);
        amrex::BoxArray boxes(domain);
        boxes.maxSize(2);
        amrex::DistributionMapping mapping(boxes);
        std::array<amrex::MultiFab, AMREX_SPACEDIM> storage;
        std::array<amrex::MultiFab const*, AMREX_SPACEDIM> faces;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            storage[d].define(amrex::convert(boxes, amrex::IntVect::TheDimensionVector(d)), mapping,
                              4, 1);
            faces[d] = &storage[d];
            // Poison ghosts: only physical valid faces may contribute.
            storage[d].setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
            for (amrex::MFIter iterator(storage[d]); iterator.isValid(); ++iterator) {
                auto const output = storage[d].array(iterator);
                amrex::ParallelFor(iterator.validbox(), 4,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) {
                                       amrex::IntVect index(AMREX_D_DECL(i, j, k));
                                       amrex::ignore_unused(j, k);
                                       // Nonzero equal incoming/outgoing offset must cancel.
                                       output(i, j, k, component) =
                                           100 + (component + 1) * (d + 1) * index[d];
                                   });
            }
        }
        constexpr amrex::Real dt = 0.125;
        FourVector exchange{123, 123, 123, 123};
        AMREX_ALWAYS_ASSERT(ComputeMomentBoundaryExchange(faces, geometry, dt, exchange));
        for (int component = 0; component < 4; ++component) {
            amrex::Real expected = 0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                auto const count = domain.numPts() / domain.length(d);
                expected += dt / geometry.CellSize(d) * count * 8 * (component + 1) * (d + 1);
            }
            AMREX_ALWAYS_ASSERT(std::abs(exchange[component] - expected) < 1.e-13 * expected);
        }
        // Independently omit each periodic direction; in 2D/3D this exercises
        // mixed boundaries without discarding the remaining physical faces.
        for (int periodic_axis = 0; periodic_axis < AMREX_SPACEDIM; ++periodic_axis) {
            periodic[periodic_axis] = 1;
            amrex::Geometry mixed(domain, &bounds, 0, periodic);
            FourVector mixed_exchange{};
            AMREX_ALWAYS_ASSERT(ComputeMomentBoundaryExchange(faces, mixed, dt, mixed_exchange));
            for (int component = 0; component < 4; ++component) {
                auto const count = domain.numPts() / domain.length(periodic_axis);
                auto const omitted = dt / geometry.CellSize(periodic_axis) * count * 8 *
                                     (component + 1) * (periodic_axis + 1);
                AMREX_ALWAYS_ASSERT(
                    std::abs(mixed_exchange[component] - (exchange[component] - omitted)) <
                    1.e-13 * exchange[component]);
            }
            periodic[periodic_axis] = 0;
        }
        auto const sentinel = exchange;
        auto unchanged = [&] () {
            return std::equal(exchange.begin(), exchange.end(), sentinel.begin());
        };
        AMREX_ALWAYS_ASSERT(!ComputeMomentBoundaryExchange(faces, geometry, -dt, exchange));
        AMREX_ALWAYS_ASSERT(unchanged());
        auto invalid_faces = faces;
        invalid_faces[0] = nullptr;
        AMREX_ALWAYS_ASSERT(!ComputeMomentBoundaryExchange(invalid_faces, geometry, dt, exchange));
        AMREX_ALWAYS_ASSERT(unchanged());
        // A nonfinite physical face rejects without partially committing other components.
        storage[0].setVal(std::numeric_limits<amrex::Real>::infinity(), 3, 1, 0);
        AMREX_ALWAYS_ASSERT(!ComputeMomentBoundaryExchange(faces, geometry, dt, exchange));
        AMREX_ALWAYS_ASSERT(unchanged());
        for (auto& flag : periodic) {
            flag = 1;
        }
        amrex::Geometry closed(domain, &bounds, 0, periodic);
        AMREX_ALWAYS_ASSERT(ComputeMomentBoundaryExchange(faces, closed, dt, exchange));
        for (auto value : exchange) {
            AMREX_ALWAYS_ASSERT(value == 0);
        }
        amrex::Print() << "Independent boundary exchange and rejection checks passed\n";
    }
    amrex::Finalize();
}
