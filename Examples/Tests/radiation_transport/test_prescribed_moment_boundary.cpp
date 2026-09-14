/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/ImplicitMomentTransport.H"
#include "Radiation/MomentBoundaryFlux.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        using namespace warpx::radiation;
        amrex::Box domain(amrex::IntVect(0), amrex::IntVect(7));
        amrex::RealBox bounds({AMREX_D_DECL(0., 0., 0.)}, {AMREX_D_DECL(1., 1., 1.)});
        int periodic[AMREX_SPACEDIM] = {AMREX_D_DECL(0, 1, 1)};
        amrex::Geometry geometry(domain, &bounds, 0, periodic);
        amrex::BoxArray boxes(domain);
        boxes.maxSize(4);
        amrex::DistributionMapping mapping(boxes);
        amrex::MultiFab state(boxes, mapping, 4, 1), transfer(boxes, mapping, 4, 1);
        amrex::MultiFab beta(boxes, mapping, 3, 0), absorption(boxes, mapping, 1, 0);
        amrex::MultiFab scattering(boxes, mapping, 1, 0), equilibrium(boxes, mapping, 1, 0);
        amrex::MultiFab snapshot(boxes, mapping, 4, 1), heat(boxes, mapping, 1, 1);
#if defined(WARPX_DIM_1D_Z)
        constexpr int axis = 2;
#else
        constexpr int axis = 0;
#endif
        constexpr amrex::Real speed = 0.002;
        amrex::Real injection = 0.1;
        amrex::ParmParse("test").query("injection", injection);
        bool reflecting = false;
        amrex::ParmParse("test").query("reflecting", reflecting);
        AMREX_ALWAYS_ASSERT(std::abs(injection) == 0.1);
        amrex::Real const volume = 1. / domain.numPts();
        amrex::Real const dt = 0.05 / PhysConst::c;
        FourVector initial{volume * (1 + speed * speed / 3) / (1 - speed * speed), 0, 0, 0};
        int const drift_axis = reflecting ? (axis + 1) % 3 : axis;
        initial[drift_axis + 1] = volume * 4 * speed / (3 * (1 - speed * speed));
        if (reflecting) {
            initial[axis + 1] = 0.25 * initial[0];
        }
        for (int c = 0; c < 4; ++c) {
            state.setVal(initial[c], c, 1, 1);
        }
        beta.setVal(0);
        beta.setVal(speed, drift_axis, 1);
        absorption.setVal(0);
        scattering.setVal(2);
        equilibrium.setVal(volume);
        std::array<amrex::MultiFab, AMREX_SPACEDIM> face_storage;
        PrescribedMomentFaceFluxes faces{};
        auto const closure = EvaluateM1Closure(initial);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            face_storage[d].define(amrex::convert(boxes, amrex::IntVect::TheDimensionVector(d)),
                                   mapping, 4, 0);
            face_storage[d].setVal(0);
            faces[d] = &face_storage[d];
        }
        for (amrex::MFIter iterator(face_storage[0]); iterator.isValid(); ++iterator) {
            auto const face = face_storage[0].array(iterator);
            amrex::ParallelFor(iterator.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                face(i, j, k, 0) = PhysConst::c * initial[axis + 1];
                for (int d = 0; d < 3; ++d) {
                    face(i, j, k, d + 1) = PhysConst::c * closure.tensor[d + 1][axis + 1];
                }
                if (i == 0) {
                    // Inject a beam: energy and c*momentum enter together.
                    face(i, j, k, 0) += injection * PhysConst::c * volume;
                    face(i, j, k, axis + 1) += injection * PhysConst::c * volume;
                }
            });
        }
        auto const assert_unchanged = [&] () {
            amrex::MultiFab::Subtract(snapshot, state, 0, 0, 4, 1);
            AMREX_ALWAYS_ASSERT(snapshot.norm0(0, 4, amrex::IntVect(1)) == 0);
            for (int c = 0; c < 4; ++c) {
                AMREX_ALWAYS_ASSERT(transfer.min(c, 1) == 123 && transfer.max(c, 1) == 123);
            }
            AMREX_ALWAYS_ASSERT(heat.min(0, 1) == 456 && heat.max(0, 1) == 456);
        };
        ImplicitMomentTransportOptions options;
        options.reflecting_boundaries = reflecting;
        auto const* boundary = reflecting ? nullptr : &faces;
        auto budget = options;
        budget.max_nonlinear_iterations = 1;
        transfer.setVal(123);
        heat.setVal(456);
        amrex::MultiFab::Copy(snapshot, state, 0, 0, 4, 1);
        auto rejected = TryImplicitMomentTransport(state, beta, absorption, scattering, equilibrium,
                                                   transfer, geometry, dt, budget, &heat, boundary);
        AMREX_ALWAYS_ASSERT(!rejected.valid);
        for (auto value : rejected.boundary_exchange) {
            AMREX_ALWAYS_ASSERT(value == 0);
        }
        assert_unchanged();
        FourVector accumulated{};
        amrex::Real wall_impulse = 0;
        for (int step = 0; step < 20; ++step) {
            auto const result =
                TryImplicitMomentTransport(state, beta, absorption, scattering, equilibrium,
                                           transfer, geometry, dt, options, &heat, boundary);
            AMREX_ALWAYS_ASSERT(result.valid);
            if (reflecting) {
                // Reevaluate pressure on the accepted stored state, outside
                // the operator, to detect a stale candidate boundary ledger.
                amrex::ReduceOps<amrex::ReduceOpSum> ops;
                amrex::ReduceData<amrex::Real> data(ops);
                using Tuple = typename decltype(data)::Type;
                auto const spacing = geometry.CellSize(0);
                for (amrex::MFIter iterator(state); iterator.isValid(); ++iterator) {
                    auto const values = state.const_array(iterator);
                    ops.eval(iterator.validbox(), data,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                            if (i != 0 && i != 7) {
                                return {0};
                            }
                            FourVector moments{values(i,j,k,0), values(i,j,k,1),
                                               values(i,j,k,2), values(i,j,k,3)};
                            auto const wall = EvaluateM1MirrorFlux(moments, axis, i == 0 ? -1 : 1);
                            return {dt / spacing * wall.flux[axis + 1]};
                        });
                }
                auto pressure = amrex::get<0>(data.value());
                amrex::ParallelDescriptor::ReduceRealSum(pressure);
                AMREX_ALWAYS_ASSERT(
                    std::abs(pressure - result.boundary_exchange[axis + 1]) < 1.e-12);
            }
            for (int c = 0; c < 4; ++c) {
                if (reflecting) {
                    if (c != axis + 1) {
                        AMREX_ALWAYS_ASSERT(result.boundary_exchange[c] == 0);
                    }
                } else {
                    auto const expected = (c == 0 || c == axis + 1) ? -0.05 * injection : 0;
                    AMREX_ALWAYS_ASSERT(std::abs(result.boundary_exchange[c] - expected) < 1.e-15);
                }
                accumulated[c] += transfer.sum(c) + result.boundary_exchange[c];
                auto const residual = state.sum(c) - initial[c] * domain.numPts() + accumulated[c];
                AMREX_ALWAYS_ASSERT(std::abs(residual) < 1.e-10);
            }
            wall_impulse += result.boundary_exchange[axis + 1];
        }
        if (reflecting) {
            AMREX_ALWAYS_ASSERT(wall_impulse > 0.01);
            AMREX_ALWAYS_ASSERT(state.sum(axis + 1) < initial[axis + 1] * domain.numPts());
        } else {
            AMREX_ALWAYS_ASSERT(injection * (state.sum(0) - 1) > 0.005);
        }
        // Nonfinite prescribed forcing must reject before touching output state.
        transfer.setVal(123);
        heat.setVal(456);
        amrex::MultiFab::Copy(snapshot, state, 0, 0, 4, 1);
        if (reflecting) {
            scattering.setVal(std::numeric_limits<amrex::Real>::infinity());
        } else {
            face_storage[0].setVal(std::numeric_limits<amrex::Real>::infinity(), 3, 1);
        }
        rejected = TryImplicitMomentTransport(state, beta, absorption, scattering, equilibrium,
                                              transfer, geometry, dt, options, &heat, boundary);
        AMREX_ALWAYS_ASSERT(!rejected.valid);
        assert_unchanged();
        amrex::Print() << (reflecting ? "Reflecting" : "Prescribed")
                       << " moving-boundary flux balances and rollback passed\n";
    }
    amrex::Finalize();
}
