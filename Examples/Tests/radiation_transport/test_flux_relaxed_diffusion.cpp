/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/FluxRelaxedDiffusion.H"
#include "Radiation/MaterialKineticWork.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <cmath>

using namespace amrex::literals;

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        static_assert(AMREX_SPACEDIM == 1);
        constexpr int count = 32;
        constexpr amrex::Real h = 1._rt / count, dt = 1.e-9_rt, opacity_value = 100._rt;
        amrex::Box const domain(amrex::IntVect(0), amrex::IntVect(count - 1));
        amrex::RealBox const extent(0._rt, 1._rt);
        int periodic[] = {1};
        amrex::Geometry const geometry(domain, &extent, 0, periodic);
        amrex::BoxArray boxes(domain);
        boxes.maxSize(8);
        amrex::DistributionMapping distribution(boxes);
        amrex::MultiFab state(boxes, distribution, 4, 1), opacity(boxes, distribution, 1, 1);
        amrex::MultiFab beta(boxes, distribution, 3, 1), saved(boxes, distribution, 4, 1);
        opacity.setVal(opacity_value);
        beta.setVal(0);
        warpx::radiation::FluxRelaxedDiffusionOptions options;
        options.relaxation = 1;
        auto no_work = [] (amrex::MultiFab const&, amrex::MultiFab& work)
        {
            work.setVal(0);
            return true;
        };
        auto initialize = [&] (int mode)
        {
            state.setVal(-17); // Deliberately distinct caller ghosts for
                               // rejection checks.
            for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi)
            {
                auto const r = state.array(mfi);
                amrex::ParallelFor(mfi.validbox(),
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
                                       auto const phase = 2 * MathConst::pi * (i + 0.5_rt) * h;
                                       r(i, j, k, 0) = mode == 0 ? 1 + 0.1_rt * std::cos(phase) : 1;
                                       r(i, j, k, 1) = 0;
                                       r(i, j, k, 2) = 0;
                                       r(i, j, k, 3) = mode == 1
                                                           ? 0.2_rt + 0.05_rt * std::cos(phase)
                                                           : (mode == 2 ? 0.2_rt : 0);
                                   });
            }
        };
        auto assert_unchanged = [&] ()
        {
            amrex::MultiFab difference(boxes, distribution, 4, 1);
            amrex::MultiFab::Copy(difference, state, 0, 0, 4, 1);
            amrex::MultiFab::Subtract(difference, saved, 0, 0, 4, 1);
            for (int d = 0; d < 4; ++d)
            {
                AMREX_ALWAYS_ASSERT(difference.norm0(d, 1) == 0);
            }
        };
        auto const x = PhysConst::c * opacity_value * dt;
        auto const theta = 2 * MathConst::pi * h;
        auto const denominator = 1 + 4 * PhysConst::c2 * dt * dt / (3 * (1 + x) * h * h) *
                                         std::sin(theta / 2) * std::sin(theta / 2);
        for (int mode = 0; mode < 2; ++mode)
        {
            initialize(mode);
            auto result = warpx::radiation::TryAdvanceFluxRelaxedDiffusion(
                state, opacity, beta, geometry, dt, options, no_work);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(result.valid,
                                             "Nontrivial Fourier/old-flux case was rejected.");
            amrex::ReduceOps<amrex::ReduceOpMax> ops;
            amrex::ReduceData<amrex::Real> data(ops);
            for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi)
            {
                auto const r = state.const_array(mfi);
                ops.eval(mfi.validbox(), data,
                         [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<amrex::Real>
                         {
                             auto expected_energy = [=] AMREX_GPU_HOST_DEVICE(amrex::Real phase)
                             {
                                 return mode == 0
                                            ? 1 + 0.1_rt * std::cos(phase) / denominator
                                            : 1 - PhysConst::c * dt * 0.05_rt /
                                                      ((1 + x) * h * denominator) *
                                                      (std::cos(phase) - std::cos(phase - theta));
                             };
                             auto const phase = theta * (i + 0.5_rt);
                             auto const old_q = mode == 0 ? 0 : 0.2_rt + 0.05_rt * std::cos(phase);
                             auto const q = (old_q - PhysConst::c * dt / (6 * h) *
                                                         (expected_energy(phase + theta) -
                                                          expected_energy(phase - theta))) /
                                            (1 + x);
                             return {amrex::max(std::abs(r(i, j, k, 0) - expected_energy(phase)),
                                                std::abs(r(i, j, k, 3) - q),
                                                std::abs(r(i, j, k, 1)), std::abs(r(i, j, k, 2)))};
                         });
            }
            auto error = amrex::get<0>(data.value());
            amrex::ParallelDescriptor::ReduceRealMax(error);
            amrex::Print() << "mode=" << mode << " independent E/q error=" << error << '\n';
            AMREX_ALWAYS_ASSERT(error < 2.e-11_rt);
            AMREX_ALWAYS_ASSERT(result.energy_equation_residual <= 1.e-11_rt);
        }

        initialize(2);
        constexpr amrex::Real mass = 1.e-12_rt;
        auto finite_mass_work = [] (amrex::MultiFab const& impulse, amrex::MultiFab& work)
        {
            for (amrex::MFIter mfi(work); mfi.isValid(); ++mfi)
            {
                auto const request = impulse.const_array(mfi);
                auto const value = work.array(mfi);
                amrex::ParallelFor(mfi.validbox(),
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
                                       amrex::GpuArray<amrex::Real, 3> zero{}, u{};
                                       for (int d = 0; d < 3; ++d)
                                       {
                                           u[d] = request(i, j, k, d) / mass;
                                       }
                                       value(i, j, k) =
                                           mass *
                                           warpx::radiation::MaterialSpecificKineticWork(zero, u);
                                   });
            }
            return true;
        };
        auto collision = warpx::radiation::TryAdvanceFluxRelaxedDiffusion(
            state, opacity, beta, geometry, dt, options, finite_mass_work);
        AMREX_ALWAYS_ASSERT(collision.valid);
        long double const impulse =
            (0.2L - 0.2L / (1 + static_cast<long double>(x))) / PhysConst::c;
        long double const m = mass, c = PhysConst::c;
        long double const expected_work =
            impulse * impulse / (m + std::sqrt(m * m + impulse * impulse / (c * c)));
        auto const measured_work = collision.requested_work.sum(0) / count;
        AMREX_ALWAYS_ASSERT(expected_work > 1.e-8L); // Work is resolved, not a no-op.
        AMREX_ALWAYS_ASSERT(std::abs(measured_work - static_cast<amrex::Real>(expected_work)) <
                            2.e-11_rt * static_cast<amrex::Real>(expected_work));
        AMREX_ALWAYS_ASSERT(std::abs(state.sum(0) / count - 1 + measured_work) < 2.e-11_rt);
        AMREX_ALWAYS_ASSERT(std::abs(state.sum(3) / count - 0.2_rt / (1 + x)) < 2.e-13_rt);
        amrex::Print() << "finite-mass work=" << measured_work << " J/cell\n";

        state.setVal(0);
        state.setVal(1, 0, 1, 0);
        constexpr amrex::Real drift = 1.e-4_rt;
        state.setVal((4._rt / 3._rt) * drift, 3, 1, 0);
        beta.setVal(drift, 2, 1, 0);
        auto bath = warpx::radiation::TryAdvanceFluxRelaxedDiffusion(state, opacity, beta, geometry,
                                                                     dt, options, no_work);
        AMREX_ALWAYS_ASSERT(bath.valid);
        AMREX_ALWAYS_ASSERT(bath.requested_impulse.norm0(2) < 1.e-24_rt);
        AMREX_ALWAYS_ASSERT(std::abs(state.sum(0) / count - 1) < 2.e-13_rt);
        beta.setVal(0);

        initialize(0);
        amrex::MultiFab::Copy(saved, state, 0, 0, 4, 1);
        auto one_rank_reject = [] (amrex::MultiFab const&, amrex::MultiFab& work)
        {
            work.setVal(0);
            return amrex::ParallelDescriptor::MyProc() != 0;
        };
        auto rejected = warpx::radiation::TryAdvanceFluxRelaxedDiffusion(
            state, opacity, beta, geometry, dt, options, one_rank_reject);
        AMREX_ALWAYS_ASSERT(!rejected.valid);
        AMREX_ALWAYS_ASSERT(rejected.requested_work.norm0() == 0);
        assert_unchanged();
        auto excessive_work = [] (amrex::MultiFab const&, amrex::MultiFab& work)
        {
            work.setVal(3);
            return true;
        };
        auto unphysical = warpx::radiation::TryAdvanceFluxRelaxedDiffusion(
            state, opacity, beta, geometry, dt, options, excessive_work);
        AMREX_ALWAYS_ASSERT(!unphysical.valid);
        assert_unchanged();
        // Known model-domain failure, not a successful transport case:
        // P1 cannot yet admit a directed cold thick/vacuum interface. It
        // must reject rather than clip the photon cone or commit a trial.
        state.setVal(-17);
        for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi)
        {
            auto const r = state.array(mfi), chi = opacity.array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   r(i, j, k, 0) = i < count / 2 ? 1 : 0;
                                   r(i, j, k, 1) = r(i, j, k, 2) = 0;
                                   r(i, j, k, 3) = r(i, j, k, 0);
                                   chi(i, j, k) = i < count / 2 ? 200 : 0;
                               });
        }
        amrex::MultiFab::Copy(saved, state, 0, 0, 4, 1);
        auto cold = warpx::radiation::TryAdvanceFluxRelaxedDiffusion(state, opacity, beta, geometry,
                                                                     1.e-11_rt, options, no_work);
        AMREX_ALWAYS_ASSERT(!cold.valid);
        AMREX_ALWAYS_ASSERT(cold.failure ==
                            warpx::radiation::FluxRelaxedDiffusionFailure::Admissibility);
        AMREX_ALWAYS_ASSERT(cold.requested_work.norm0() == 0);
        for (int d = 0; d < 3; ++d)
        {
            AMREX_ALWAYS_ASSERT(cold.requested_impulse.norm0(d) == 0);
        }
        assert_unchanged();
        amrex::Print() << "Cold interface remains unsupported; rejection/rollback verified.\n";
        amrex::Print() << "Flux-relaxed Fourier, drift, work, and collective "
                          "rollback gates passed.\n";
    }
    amrex::Finalize();
}
