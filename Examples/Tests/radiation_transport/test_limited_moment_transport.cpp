/* Copyright 2026 The WarpX Community
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/LimitedMomentTransport.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

using namespace amrex::literals;

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse inputs("test");
        bool verbose = false;
        inputs.query("verbose", verbose);

#if defined(WARPX_DIM_1D_Z)
        int const longitudinal_cells = 24;
        amrex::IntVect const hi(AMREX_D_DECL(longitudinal_cells - 1, 0, 0));
        constexpr int normal = 2;
#elif defined(WARPX_DIM_XZ)
        int const longitudinal_cells = 16;
        amrex::IntVect const hi(AMREX_D_DECL(longitudinal_cells - 1, 0, 0));
        constexpr int normal = 0;
#else
        int const longitudinal_cells = 12;
        amrex::IntVect const hi(AMREX_D_DECL(longitudinal_cells - 1, 3, 3));
        constexpr int normal = 0;
#endif
        amrex::Box const domain(amrex::IntVect::TheZeroVector(), hi);
        amrex::RealBox const bounds({AMREX_D_DECL(0., 0., 0.)},
                                    {AMREX_D_DECL(1., 1., 1.)});
        int const periodic[AMREX_SPACEDIM] = {AMREX_D_DECL(1, 1, 1)};
        amrex::Geometry const geometry(domain, &bounds, 0, periodic);

        amrex::BoxArray boxes(domain);
        int const ranks = amrex::ParallelDescriptor::NProcs();
        amrex::IntVect split_size = amrex::IntVect::TheUnitVector();
        split_size[0] = ranks == 1 ? 4 : longitudinal_cells / 2;
        for (int dimension = 1; dimension < AMREX_SPACEDIM; ++dimension)
        {
            split_size[dimension] = domain.length(dimension);
        }
        boxes.maxSize(split_size);
        amrex::DistributionMapping const distribution(boxes);
        amrex::MultiFab state(boxes, distribution, 4, 1);
        amrex::MultiFab beta(boxes, distribution, 3, 0);
        amrex::MultiFab chi(boxes, distribution, 1, 0);
        beta.setVal(0.0);
        chi.setVal(0.0);

        amrex::Real minimum_spacing = std::numeric_limits<amrex::Real>::infinity();
        amrex::Real volume = 1.0;
        for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
        {
            minimum_spacing = amrex::min(minimum_spacing, geometry.CellSize(dimension));
            volume *= geometry.CellSize(dimension);
        }
        amrex::Real const safe_dt = 0.15_rt * minimum_spacing / PhysConst::c;
        amrex::GpuArray<amrex::Real, 3> direction{};
#if defined(WARPX_DIM_1D_Z)
        direction[0] = 0.6;
        direction[2] = 0.8;
#else
        direction[0] = 0.6;
        direction[1] = 0.64;
        direction[2] = 0.48;
#endif

        auto initialize = [&] (std::string const& mode, amrex::Real optical_depth)
        {
            int const profile = mode == "uniform" ? 0 : (mode == "cold" ? 1 : 2);
            state.setVal(0.0);
            chi.setVal(0.0);
            auto const opacity = optical_depth / geometry.CellSize(0);
            if (mode == "opaque")
            {
                chi.setVal(opacity);
            }
            auto const wave_number = 6.283185307179586_rt;
            for (amrex::MFIter iterator(state); iterator.isValid(); ++iterator)
            {
                auto const field = state.array(iterator);
                auto const transport_opacity = chi.array(iterator);
                amrex::ParallelFor(iterator.validbox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        auto const location =
                            wave_number * (i + 0.5_rt) / longitudinal_cells;
                        amrex::Real energy_density = 0.0;
                        amrex::GpuArray<amrex::Real, 3> momentum_density{};
                        if (profile == 0)
                        {
                            energy_density = 1.0_rt;
                            for (int component = 0; component < 3; ++component)
                            {
                                momentum_density[component] =
                                    0.2_rt * direction[component] * energy_density;
                            }
                        }
                        else if (profile == 1)
                        {
                            auto const first = longitudinal_cells / 4;
                            auto const last = longitudinal_cells / 2;
                            transport_opacity(i, j, k) =
                                i >= first && i <= last ? 200._rt : 0._rt;
                            if (i >= first && i <= last)
                            {
                                energy_density = 1.0_rt;
                                for (int component = 0; component < 3; ++component)
                                {
                                    momentum_density[component] =
                                        direction[component] * energy_density;
                                }
                            }
                        }
                        else
                        {
                            energy_density = 1.0_rt + 0.1_rt * std::cos(location);
                        }
                        field(i, j, k, 0) = volume * energy_density;
                        for (int component = 0; component < 3; ++component)
                        {
                            field(i, j, k, component + 1) =
                                volume * momentum_density[component];
                        }
                    });
            }
            state.FillBoundary(geometry.periodicity());
        };

        auto rollback_is_exact = [&] (amrex::MultiFab const& snapshot)
        {
            amrex::MultiFab difference(boxes, distribution, 4, 1);
            amrex::MultiFab::Copy(difference, snapshot, 0, 0, 4, 1);
            amrex::MultiFab::Subtract(difference, state, 0, 0, 4, 1);
            return difference.norm0(0, 4, amrex::IntVect(1)) == 0.0;
        };

        warpx::radiation::LimitedMomentTransportOptions options;
        options.verbose = verbose;

        initialize("uniform", 0.0);
        amrex::MultiFab uniform_before(boxes, distribution, 4, 1);
        amrex::MultiFab::Copy(uniform_before, state, 0, 0, 4, 1);
        auto const uniform_result = warpx::radiation::TryAdvanceLimitedMomentTransport(
            state, beta, chi, geometry, safe_dt, options);
        if (!uniform_result.valid)
        {
            amrex::Print() << "Uniform limited transport failure="
                           << static_cast<int>(uniform_result.failure)
                           << " balance=(";
            for (int component = 0; component < 4; ++component)
            {
                amrex::Print() << uniform_result.raw_balance_residual[component] << ',';
            }
            amrex::Print() << ") initial=(";
            for (int component = 0; component < 4; ++component)
            {
                amrex::Print() << uniform_result.initial_inventory[component] << ',';
            }
            amrex::Print() << ") final=(";
            for (int component = 0; component < 4; ++component)
            {
                amrex::Print() << uniform_result.final_inventory[component] << ',';
            }
            amrex::Print() << ")\n";
        }
        AMREX_ALWAYS_ASSERT(uniform_result.valid);
        amrex::Print() << "Uniform result theta=(" << uniform_result.minimum_theta << ','
                       << uniform_result.maximum_theta << ") cone="
                       << uniform_result.minimum_cone_margin << " state_range=(";
        for (int component = 0; component < 4; ++component)
        {
            amrex::Print() << (component == 0 ? "" : ",")
                           << state.min(component, 0) << ':'
                           << state.max(component, 0);
        }
        amrex::Print() << ")\n";
        amrex::Print() << "Uniform before range=(";
        for (int component = 0; component < 4; ++component)
        {
            amrex::Print() << (component == 0 ? "" : ",")
                           << uniform_before.min(component, 1) << ':'
                           << uniform_before.max(component, 1);
        }
        amrex::Print() << ")\n";
        AMREX_ALWAYS_ASSERT(uniform_result.failure ==
                            warpx::radiation::LimitedMomentTransportFailure::None);
        AMREX_ALWAYS_ASSERT(uniform_result.maximum_theta == 1.0);
        AMREX_ALWAYS_ASSERT(uniform_result.minimum_theta == 1.0);
        AMREX_ALWAYS_ASSERT(uniform_result.limited_face_count == 0);
        {
            amrex::MultiFab difference(boxes, distribution, 4, 1);
            amrex::MultiFab::Copy(difference, uniform_before, 0, 0, 4, 1);
            amrex::MultiFab::Subtract(difference, state, 0, 0, 4, 1);
            amrex::Print() << "Uniform difference range=(";
            for (int component = 0; component < 4; ++component)
            {
                amrex::Print() << (component == 0 ? "" : ",")
                               << difference.min(component, 1) << ':'
                               << difference.max(component, 1);
            }
            amrex::Print() << ")\n";
            auto const relative_change = difference.norm0(0, 4, amrex::IntVect(1)) /
                (uniform_before.norm1(0) + state.norm1(0));
            amrex::Print() << "Uniform norms difference=" << difference.norm0(0, 4, amrex::IntVect(1))
                           << " denominator=" << uniform_before.norm1(0) + state.norm1(0)
                           << '\n';
            amrex::Print() << "Uniform relative change=" << relative_change
                           << " interior=" << difference.norm0(0, 4, amrex::IntVect(0))
                           << " ghosts=" << difference.norm0(0, 4, amrex::IntVect(1)) << '\n';
            AMREX_ALWAYS_ASSERT(relative_change <=
                                128 * std::numeric_limits<amrex::Real>::epsilon());
        }

        initialize("cold", 0.0);
        amrex::GpuArray<amrex::Real, 4> initial_inventory{};
        for (int component = 0; component < 4; ++component)
        {
            initial_inventory[component] = state.sum(component, false);
        }
        amrex::Real minimum_theta = 1.0;
        amrex::Real minimum_cone_margin = 1.0;
        long long limited_faces = 0;
        constexpr int cold_steps = 8;
        for (int step = 0; step < cold_steps; ++step)
        {
            auto const result = warpx::radiation::TryAdvanceLimitedMomentTransport(
                state, beta, chi, geometry, safe_dt, options);
            AMREX_ALWAYS_ASSERT(result.valid);
            AMREX_ALWAYS_ASSERT(result.maximum_cfl <= 0.5);
            AMREX_ALWAYS_ASSERT(result.accepted_face_count ==
                                static_cast<long long>(domain.numPts()) *
                                    AMREX_SPACEDIM);
            for (int component = 0; component < 4; ++component)
            {
                auto const residual = std::abs(
                    initial_inventory[component] - result.final_inventory[component]);
                auto const scale = state.norm1(component) +
                    std::abs(initial_inventory[component]);
                AMREX_ALWAYS_ASSERT(residual <=
                                    128 * std::numeric_limits<amrex::Real>::epsilon() *
                                        scale);
                AMREX_ALWAYS_ASSERT(std::abs(result.raw_balance_residual[component]) <=
                                    128 * std::numeric_limits<amrex::Real>::epsilon() *
                                        scale);
            }
            minimum_theta = amrex::min(minimum_theta, result.minimum_theta);
            minimum_cone_margin = amrex::min(
                minimum_cone_margin, result.minimum_cone_margin);
            limited_faces += result.limited_face_count;
        }
        AMREX_ALWAYS_ASSERT(minimum_theta < 1.0);
        AMREX_ALWAYS_ASSERT(limited_faces > 0);
        AMREX_ALWAYS_ASSERT(minimum_cone_margin >=
                            -8 * std::numeric_limits<amrex::Real>::epsilon());

        amrex::MultiFab snapshot(boxes, distribution, 4, 1);
        initialize("uniform", 0.0);
        amrex::MultiFab::Copy(snapshot, state, 0, 0, 4, 1);
        amrex::Real inverse_spacing_sum = 0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            inverse_spacing_sum += 1 / geometry.CellSize(d);
        }
        auto const invalid_cfl = warpx::radiation::TryAdvanceLimitedMomentTransport(
            state, beta, chi, geometry, .6_rt / (PhysConst::c * inverse_spacing_sum), options);
        AMREX_ALWAYS_ASSERT(!invalid_cfl.valid);
        AMREX_ALWAYS_ASSERT(invalid_cfl.failure ==
                            warpx::radiation::LimitedMomentTransportFailure::InvalidDt);
        AMREX_ALWAYS_ASSERT(rollback_is_exact(snapshot));

        chi.setVal(-1.0);
        auto const invalid_opacity =
            warpx::radiation::TryAdvanceLimitedMomentTransport(
                state, beta, chi, geometry, safe_dt, options);
        AMREX_ALWAYS_ASSERT(!invalid_opacity.valid);
        AMREX_ALWAYS_ASSERT(invalid_opacity.failure ==
                            warpx::radiation::LimitedMomentTransportFailure::InvalidOpacity);
        AMREX_ALWAYS_ASSERT(rollback_is_exact(snapshot));
        chi.setVal(0.0);

        initialize("uniform", 0.0);
        for (amrex::MFIter iterator(state); iterator.isValid(); ++iterator)
        {
            auto const field = state.array(iterator);
            amrex::ParallelFor(iterator.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (i == domain.smallEnd(0))
                    {
                        field(i, j, k, normal + 1) =
                            2.0 * field(i, j, k, 0);
                    }
                });
        }
        amrex::MultiFab::Copy(snapshot, state, 0, 0, 4, 1);
        auto const invalid_state =
            warpx::radiation::TryAdvanceLimitedMomentTransport(
                state, beta, chi, geometry, safe_dt, options);
        AMREX_ALWAYS_ASSERT(!invalid_state.valid);
        AMREX_ALWAYS_ASSERT(invalid_state.failure ==
                            warpx::radiation::LimitedMomentTransportFailure::InvalidState);
        AMREX_ALWAYS_ASSERT(rollback_is_exact(snapshot));

        initialize("uniform", 0.0);
        amrex::MultiFab::Copy(snapshot, state, 0, 0, 4, 1);
        amrex::MultiFab wrong_beta(boxes, distribution, 2, 0);
        wrong_beta.setVal(0.0);
        auto const invalid_layout =
            warpx::radiation::TryAdvanceLimitedMomentTransport(
                state, wrong_beta, chi, geometry, safe_dt, options);
        AMREX_ALWAYS_ASSERT(!invalid_layout.valid);
        AMREX_ALWAYS_ASSERT(invalid_layout.failure ==
                            warpx::radiation::LimitedMomentTransportFailure::InvalidGeometry);
        AMREX_ALWAYS_ASSERT(rollback_is_exact(snapshot));

        amrex::MultiFab alias_beta(state, amrex::make_alias, 0, 3);
        auto const invalid_alias = warpx::radiation::TryAdvanceLimitedMomentTransport(
            state, alias_beta, chi, geometry, safe_dt, options);
        AMREX_ALWAYS_ASSERT(!invalid_alias.valid);
        AMREX_ALWAYS_ASSERT(rollback_is_exact(snapshot));
        auto invalid_options = options;
        invalid_options.maximum_bisections = 0;
        AMREX_ALWAYS_ASSERT(!warpx::radiation::TryAdvanceLimitedMomentTransport(
            state, beta, chi, geometry, safe_dt, invalid_options).valid);
        AMREX_ALWAYS_ASSERT(rollback_is_exact(snapshot));
        if (ranks > 1) {
            auto const inconsistent_dt = amrex::ParallelDescriptor::MyProc() == 0
                ? safe_dt : 0.5_rt * safe_dt;
            AMREX_ALWAYS_ASSERT(!warpx::radiation::TryAdvanceLimitedMomentTransport(
                state, beta, chi, geometry, inconsistent_dt, options).valid);
            AMREX_ALWAYS_ASSERT(rollback_is_exact(snapshot));
            auto const nan_dt = amrex::ParallelDescriptor::MyProc() == 0
                ? std::numeric_limits<amrex::Real>::quiet_NaN() : safe_dt;
            AMREX_ALWAYS_ASSERT(!warpx::radiation::TryAdvanceLimitedMomentTransport(
                state, beta, chi, geometry, nan_dt, options).valid);
            AMREX_ALWAYS_ASSERT(rollback_is_exact(snapshot));
        }

        std::vector<amrex::Real> diffusion_errors;
        std::vector<amrex::Real> measured_diffusions;
        for (auto optical_depth : {8.0_rt, 32.0_rt, 128.0_rt})
        {
            initialize("opaque", optical_depth);
            amrex::MultiFab old_state(boxes, distribution, 4, 1);
            amrex::MultiFab::Copy(old_state, state, 0, 0, 4, 0);
            auto const result = warpx::radiation::TryAdvanceLimitedMomentTransport(
                state, beta, chi, geometry, safe_dt, options);
            AMREX_ALWAYS_ASSERT(result.valid);
            AMREX_ALWAYS_ASSERT(result.minimum_theta == 1.0);
            AMREX_ALWAYS_ASSERT(result.maximum_theta == 1.0);
            AMREX_ALWAYS_ASSERT(result.limited_face_count == 0);

            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> ops;
            amrex::ReduceData<amrex::Real, amrex::Real> data(ops);
            auto const wave_number = 6.283185307179586_rt;
            auto const spacing = geometry.CellSize(0);
            for (amrex::MFIter iterator(state); iterator.isValid(); ++iterator)
            {
                auto const next = state.const_array(iterator);
                auto const old = old_state.const_array(iterator);
                ops.eval(iterator.validbox(), data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        -> amrex::GpuTuple<amrex::Real, amrex::Real>
                    {
                        auto const phase = std::cos(
                            wave_number * (i + 0.5_rt) / longitudinal_cells);
                        auto const change = (next(i, j, k, 0) - old(i, j, k, 0)) /
                            volume;
                        return {-phase * change, phase * phase};
                    });
            }
            auto const values = data.value();
            amrex::Real reduction[2] = {amrex::get<0>(values), amrex::get<1>(values)};
            amrex::ParallelDescriptor::ReduceRealSum(reduction, 2);
            auto const discrete_laplacian =
                4.0 * std::pow(std::sin(0.5 * wave_number * spacing), 2) /
                    (spacing * spacing);
            auto const measured = reduction[0] /
                (reduction[1] * .1_rt * discrete_laplacian * safe_dt);
            auto const target = PhysConst::c * spacing / (3.0 * optical_depth);
            auto const relative_error = std::abs(measured / target - 1.0);
            measured_diffusions.push_back(measured);
            diffusion_errors.push_back(relative_error);
        }
        AMREX_ALWAYS_ASSERT(diffusion_errors[0] < 0.10);
        AMREX_ALWAYS_ASSERT(diffusion_errors[1] < 0.025);
        AMREX_ALWAYS_ASSERT(diffusion_errors[2] < 0.010);
        AMREX_ALWAYS_ASSERT(diffusion_errors[2] < diffusion_errors[0]);

        if (amrex::ParallelDescriptor::IOProcessor())
        {
            amrex::Print() << "Limited M1 transport ranks=" << ranks
                           << " dimensions=" << AMREX_SPACEDIM
                           << " cold_steps=" << cold_steps
                           << " limited_faces=" << limited_faces
                           << " minimum_theta=" << minimum_theta
                           << " minimum_cone_margin=" << minimum_cone_margin
                           << " opaque_D_errors=";
            for (auto error : diffusion_errors)
            {
                amrex::Print() << error << ',';
            }
            amrex::Print() << '\n';
        }
    }
    amrex::Finalize();
}
