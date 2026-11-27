/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "CoupledGroupSource.H"

#include "CellVolume.H"
#include "Particles/MultiParticleContainer.H"
#include "RZMomentGeometry.H"
#include "WarpX.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

namespace warpx::radiation
{
namespace
{
[[maybe_unused]] bool Agree (bool value)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(value);
    return value;
}
[[maybe_unused]] void Copy (amrex::MultiFab &out, amrex::MultiFab const &in)
{
    amrex::MultiFab::Copy(out, in, 0, 0, in.nComp(), in.nGrowVect());
}
[[maybe_unused]] bool Overlaps (amrex::MultiFab const &a, amrex::MultiFab const &b)
{
    for (amrex::MFIter it(a); it.isValid(); ++it)
    {
        auto const start_a = reinterpret_cast<std::uintptr_t>(a[it].dataPtr());
        auto const start_b = reinterpret_cast<std::uintptr_t>(b[it].dataPtr());
        auto const bytes_a = static_cast<std::uintptr_t>(a[it].size()) * sizeof(amrex::Real);
        auto const bytes_b = static_cast<std::uintptr_t>(b[it].size()) * sizeof(amrex::Real);
        auto const limit = std::numeric_limits<std::uintptr_t>::max();
        if (!start_a || !start_b || bytes_a > limit - start_a || bytes_b > limit - start_b ||
            (start_a < start_b + bytes_b && start_b < start_a + bytes_a))
        {
            return true;
        }
    }
    return false;
}
[[maybe_unused]] amrex::Real Relative (amrex::Real error, amrex::Real scale, amrex::Real roundoff,
                                       amrex::Real tolerance)
{
    auto const bound = tolerance * scale + roundoff;
    return bound == 0 ? (error == 0 ? 0 : std::numeric_limits<amrex::Real>::infinity())
                      : tolerance * error / bound;
}

template <int MaximumGroups>
bool EvaluateSource (EnergyGroupsExecutor const &groups, amrex::MultiFab const &radiation,
                     amrex::MultiFab const &beta, amrex::MultiFab const &temperature,
                     amrex::MultiFab const &transport, amrex::MultiFab const &absorption,
                     amrex::Geometry const &geometry, amrex::Real dt, amrex::MultiFab &state,
                     amrex::MultiFab &momentum, amrex::MultiFab &energy, amrex::MultiFab &physical,
                     amrex::MultiFab &arithmetic, amrex::MultiFab &transfers,
                     amrex::MultiFab &transfer_noise)
{
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    auto const n = groups.m_num_groups;
    auto const spacing = geometry.CellSizeArray(), lower = geometry.ProbLoArray();
    auto const domain_lo = amrex::lbound(geometry.Domain());
    for (amrex::MFIter it(radiation); it.isValid(); ++it)
    {
        auto const old = radiation.const_array(it), velocity = beta.const_array(it);
        auto const chi_t = transport.const_array(it), chi_a = absorption.const_array(it);
        auto const local_t = temperature.const_array(it);
        auto const out = state.array(it), p = momentum.array(it), e = energy.array(it);
        auto const q = physical.array(it), noise = arithmetic.array(it);
        auto const group_q = transfers.array(it), group_noise = transfer_noise.array(it);
        ops.eval(it.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                 {
                     ExactGroupVector before{};
                     amrex::GpuArray<amrex::Real, MaximumGroups> ct{}, ca{};
                     amrex::GpuArray<amrex::Real, 3> drift{};
                     for (int d = 0; d < 3; ++d)
                     {
                         drift[d] = velocity(i, j, k, d);
                     }
                     for (int g = 0; g < n; ++g)
                     {
                         ct[g] = chi_t(i, j, k, g);
                         ca[g] = chi_a(i, j, k, g);
                         for (int d = 0; d < 4; ++d)
                         {
                             before[4 * g + d] = old(i, j, k, 4 * g + d);
                         }
                     }
                     auto const solved = TryImplicitExactPlanckBinSource<MaximumGroups>(
                         before, groups, drift, local_t(i, j, k),
                         CellVolume(i, spacing, lower, domain_lo), dt, ct.data(), ca.data());
                     if (!solved.valid)
                     {
                         return {1};
                     }
                     e(i, j, k) = 0;
                     for (int d = 0; d < 3; ++d)
                     {
                         p(i, j, k, d) = 0;
                     }
                     for (int g = 0; g < n; ++g)
                     {
                         for (int d = 0; d < 4; ++d)
                         {
                             out(i, j, k, 4 * g + d) = solved.radiation[4 * g + d];
                             group_q(i, j, k, 4 * g + d) = solved.material_transfer[4 * g + d];
                             group_noise(i, j, k, 4 * g + d) = solved.transfer_roundoff[4 * g + d];
                         }
                         e(i, j, k) += solved.material_transfer[4 * g];
                         for (int d = 0; d < 3; ++d)
                         {
                             p(i, j, k, d) +=
                                 solved.material_transfer[4 * g + d + 1] / PhysConst::c;
                         }
                     }
                     q(i, j, k) = solved.physical_heat;
                     noise(i, j, k, 0) = solved.physical_heat_roundoff;
                     noise(i, j, k, 1) = solved.projection_roundoff;
                     noise(i, j, k, 2) = solved.momentum_roundoff;
                     return {0};
                 });
    }
    int invalid = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    return invalid == 0;
}

#if defined(WARPX_DIM_RZ)
amrex::GpuArray<amrex::Real, 2>
AngularBalance (amrex::MultiFab const &after, amrex::MultiFab const &before,
                amrex::MultiFab const &requested, ParticleImpulseTransaction const &kick,
                amrex::Geometry const &geometry, int groups, amrex::Real tolerance)
{
    RZMomentMetric const metric(geometry);
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real> data(ops);
    for (amrex::MFIter it(after); it.isValid(); ++it)
    {
        auto const old = before.const_array(it), next = after.const_array(it);
        auto const want = requested.const_array(it),
                   actual = kick.ConjugateImpulse().const_array(it);
        auto const carry = kick.ConjugateMomentumCarryChange().const_array(it);
        ops.eval(it.validbox(), data,
                 [=] AMREX_GPU_DEVICE(
                     int i, int j, int k) -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real>
                 {
                     auto change = PhysConst::c * (actual(i, j, k, 1) + carry(i, j, k, 1));
                     auto scale =
                         PhysConst::c * (std::abs(want(i, j, k, 1)) + std::abs(actual(i, j, k, 1)) +
                                         std::abs(carry(i, j, k, 1)));
                     amrex::Real rotation = 0;
                     for (int d = 0; d < 2; ++d)
                     {
                         rotation += PhysConst::c *
                                     (std::abs(want(i, j, k, d)) + std::abs(actual(i, j, k, d)) +
                                      std::abs(carry(i, j, k, d)));
                     }
                     for (int g = 0; g < groups; ++g)
                     {
                         change += next(i, j, k, 4 * g + 2) - old(i, j, k, 4 * g + 2);
                         scale +=
                             std::abs(next(i, j, k, 4 * g + 2)) + std::abs(old(i, j, k, 4 * g + 2));
                     }
                     auto const radius = metric.Angular(i).mean_radius;
                     return {radius * change, radius * scale, radius * rotation};
                 });
    }
    auto const values = data.value();
    amrex::Real sums[3]{amrex::get<0>(values), amrex::get<1>(values), amrex::get<2>(values)};
    amrex::ParallelDescriptor::ReduceRealSum(sums, 3);
    auto const bound =
        tolerance * sums[1] + 64 * std::numeric_limits<amrex::Real>::epsilon() * sums[2];
    return {sums[1] == 0 ? std::abs(sums[0]) : std::abs(sums[0]) / sums[1],
            bound == 0 ? (sums[0] == 0 ? 0 : std::numeric_limits<amrex::Real>::infinity())
                       : std::abs(sums[0]) / bound};
}

void FillGroupGhosts (amrex::MultiFab &field, amrex::Geometry const &geometry, int groups)
{
    for (int group = 0; group < groups; ++group)
    {
        amrex::MultiFab alias(field, amrex::make_alias, 4 * group, 4);
        FillRZMomentGhosts(alias, geometry, 1);
    }
}
#endif
} // namespace

CoupledGroupSourceResult TryAdvanceCoupledGroupSource (
    MultiParticleContainer &particles, std::vector<std::string> const &species,
    std::string const &path, amrex::MultiFab &radiation, amrex::MultiFab &temperature,
    amrex::MultiFab &realized_heat, amrex::Geometry const &geometry, amrex::Real time,
    amrex::Real dt, CoupledGroupSourceOptions const &options,
    CoupledGroupSourceCallbacks const &callbacks)
{
    CoupledGroupSourceResult result;
    auto fail = [&] (std::string const &reason)
    {
        result.reason = reason;
        return std::move(result);
    };
#if defined(AMREX_USE_FLOAT) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    amrex::ignore_unused(particles, species, path, radiation, temperature, realized_heat, geometry,
                         time, dt, options, callbacks);
    return fail("Native group source requires double precision periodic Cartesian geometry");
#else
    int const n = options.groups.m_num_groups;
#if defined(WARPX_DIM_RZ)
    // WarpX uses a Cartesian logical AMReX mesh and explicit annular
    // measures; do not require AMReX's separate cylindrical-operator mode.
    auto const &native_geometry = WarpX::GetInstance().Geom(0);
    bool const supported_geometry =
        geometry.Coord() == native_geometry.Coord() &&
        geometry.Domain() == native_geometry.Domain() &&
        geometry.CellSize(0) == native_geometry.CellSize(0) &&
        geometry.CellSize(1) == native_geometry.CellSize(1) && geometry.ProbLo(0) == 0 &&
        !geometry.isPeriodic(0) && geometry.isPeriodic(1) && WarpX::n_rz_azimuthal_modes == 1 &&
        (!options.spatial_transport || options.spatial.reflecting_boundaries) &&
        radiation.nGrowVect() == amrex::IntVect(1) &&
        options.assignment == ParticleImpulseAssignment::RZAngularConservative &&
        WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC &&
        WarpX::particle_boundary_hi[0] == ParticleBoundaryType::Reflecting;
#else
    bool const supported_geometry = geometry.isAllPeriodic() && geometry.Coord() == 0 &&
                                    options.assignment == ParticleImpulseAssignment::NearestCell;
#endif
    auto const &boxes = radiation.boxArray();
    auto const &distribution = radiation.DistributionMap();
    bool valid = n > 0 && n <= exact_group_maximum && radiation.nComp() == 4 * n &&
                 radiation.ixType().cellCentered() && radiation.nGrowVect().allGE(1) &&
                 temperature.nComp() == 1 && temperature.ixType().nodeCentered() &&
                 temperature.nGrowVect().allGE(1) && realized_heat.nComp() == 1 &&
                 realized_heat.nGrowVect().allGE(1) && realized_heat.boxArray() == boxes &&
                 realized_heat.DistributionMap() == distribution &&
                 amrex::convert(boxes, temperature.ixType()) == temperature.boxArray() &&
                 temperature.DistributionMap() == distribution && supported_geometry &&
                 WarpX::GetInstance().finestLevel() == 0 && options.max_iterations > 0 &&
                 options.tolerance > 0 && options.tolerance < 1 && options.energy_tolerance > 0 &&
                 options.energy_tolerance < 1 && options.relaxation > 0 &&
                 options.relaxation <= 1 && std::isfinite(time) && std::isfinite(dt) && dt > 0 &&
                 std::isfinite(time + dt) && time + dt > time && callbacks.coefficients &&
                 callbacks.caloric.material_response && callbacks.caloric.material_residual &&
                 callbacks.caloric.material_local_residual &&
                 std::numeric_limits<amrex::ParticleReal>::digits >= 53;
    valid = valid && (n == 1 || options.groups.m_boundaries);
    if (!Agree(valid))
    {
        return fail("Invalid native multigroup source state or callbacks");
    }
    if (!Agree(!Overlaps(radiation, temperature) && !Overlaps(radiation, realized_heat) &&
               !Overlaps(temperature, realized_heat)))
    {
        return fail("Native group source fields overlap");
    }
    int integer_min[5]{n, options.max_iterations, static_cast<int>(options.assignment),
                       static_cast<int>(options.spatial_transport),
                       static_cast<int>(options.defer_live_commit)};
    int integer_max[5]{n, options.max_iterations, static_cast<int>(options.assignment),
                       static_cast<int>(options.spatial_transport),
                       static_cast<int>(options.defer_live_commit)};
    amrex::ParallelDescriptor::ReduceIntMin(integer_min, 5);
    amrex::ParallelDescriptor::ReduceIntMax(integer_max, 5);
    for (int d = 0; d < 5; ++d)
    {
        if (integer_min[d] != integer_max[d])
        {
            return fail("Native group options differ across ranks");
        }
    }
    amrex::Real real_min[5]{time, dt, options.tolerance, options.energy_tolerance,
                            options.relaxation};
    amrex::Real real_max[5]{time, dt, options.tolerance, options.energy_tolerance,
                            options.relaxation};
    amrex::ParallelDescriptor::ReduceRealMin(real_min, 5);
    amrex::ParallelDescriptor::ReduceRealMax(real_max, 5);
    for (int d = 0; d < 5; ++d)
    {
        if (real_min[d] != real_max[d])
        {
            return fail("Native group real options differ across ranks");
        }
    }
    amrex::GpuArray<amrex::Real, exact_group_maximum - 1> boundary_min{}, boundary_max{};
    if (n > 1)
    {
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, options.groups.m_boundaries,
                         options.groups.m_boundaries + n - 1, boundary_min.begin());
        boundary_max = boundary_min;
    }
    amrex::ParallelDescriptor::ReduceRealMin(boundary_min.data(), exact_group_maximum - 1);
    amrex::ParallelDescriptor::ReduceRealMax(boundary_max.data(), exact_group_maximum - 1);
    amrex::Real previous_boundary = 0;
    for (int d = 0; d < n - 1; ++d)
    {
        if (boundary_min[d] != boundary_max[d] || !(boundary_min[d] > previous_boundary) ||
            !std::isfinite(boundary_min[d]))
        {
            return fail("Native group frequency grids disagree or are invalid");
        }
        previous_boundary = boundary_min[d];
    }
    auto cell = [&] (int components, int ghosts = 0)
    { return std::make_unique<amrex::MultiFab>(boxes, distribution, components, ghosts); };
    auto node = [&] ()
    {
        return std::make_unique<amrex::MultiFab>(temperature.boxArray(), distribution, 1,
                                                 temperature.nGrowVect());
    };
    auto trial_t = node(), candidate_t = node(), check_t = node(), lower_t = node(),
         upper_t = node();
    auto previous_t_residual = node(), t_residual = node();
    auto candidate_r = cell(4 * n, 1), check_r = cell(4 * n, 1);
    auto group_transfer = cell(4 * n), check_group_transfer = cell(4 * n);
    auto group_precision = cell(4 * n), check_group_precision = cell(4 * n);
    auto transport = cell(n), absorption = cell(n), cell_t = cell(1);
    auto beta = cell(3), next_beta = cell(3), zero_impulse = cell(3);
    auto impulse = cell(3), check_impulse = cell(3), debit = cell(1), check_debit = cell(1);
    auto heat = cell(1, 1), actual_heat = cell(1, 1), check_heat = cell(1, 1),
         probe_heat = cell(1, 1);
    auto physical_heat = cell(1), check_physical_heat = cell(1), precision = cell(3),
         check_precision = cell(3);
    std::unique_ptr<ImplicitGroupMomentTransportResult> latest_spatial;
    Copy(*trial_t, temperature);
    zero_impulse->setVal(0);
    auto private_material = std::make_unique<ParticleImpulseMaterial>(particles, species);
    auto& material = *private_material;
    {
        ParticleImpulseTransaction initial;
        if (!initial.Stage(material, path, *zero_impulse, true, options.assignment))
        {
            return fail("Initial private material stage rejected");
        }
        amrex::MultiFab::Copy(*beta, initial.WorkVelocity(), 0, 0, 3, 0);
        beta->mult(1 / PhysConst::c);
    }
    auto source = [&] (amrex::MultiFab const &t, amrex::MultiFab const &b, amrex::MultiFab &state,
                       amrex::MultiFab &momentum, amrex::MultiFab &energy,
                       amrex::MultiFab &physical, amrex::MultiFab &arithmetic,
                       amrex::MultiFab &transfers, amrex::MultiFab &transfer_noise)
    {
        if (!Agree(t.is_finite() && t.min(0) > 0 && b.is_finite()))
        {
            return false;
        }
        transport->setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
        absorption->setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
        cell_t->setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
        callbacks.coefficients(t, *transport, *absorption, *cell_t, time + dt);
        if (options.spatial_transport)
        {
            auto controls = options.spatial;
            controls.groups = options.groups;
            auto stage = TryImplicitGroupMomentTransport(radiation, b, *cell_t, *transport,
                                                         *absorption, geometry, dt, controls);
            if (!stage.valid)
            {
                if (options.verbose)
                {
                    amrex::Print() << "Native group spatial rejection: " << stage.reason << '\n';
                }
                return false;
            }
            amrex::MultiFab::Copy(state, stage.radiation, 0, 0, 4 * n, 0);
            amrex::MultiFab::Copy(transfers, stage.material_transfer, 0, 0, 4 * n, 0);
            amrex::MultiFab::Copy(transfer_noise, stage.transfer_roundoff, 0, 0, 4 * n, 0);
            amrex::MultiFab::Copy(physical, stage.physical_heat, 0, 0, 1, 0);
            amrex::MultiFab::Copy(arithmetic, stage.arithmetic, 0, 0, 3, 0);
            for (amrex::MFIter it(momentum); it.isValid(); ++it)
            {
                auto const group = stage.material_transfer.const_array(it);
                auto const p = momentum.array(it), e = energy.array(it);
                amrex::ParallelFor(it.validbox(),
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
                                       e(i, j, k) = 0;
                                       for (int d = 0; d < 3; ++d)
                                       {
                                           p(i, j, k, d) = 0;
                                       }
                                       for (int g = 0; g < n; ++g)
                                       {
                                           e(i, j, k) += group(i, j, k, 4 * g);
                                           for (int d = 0; d < 3; ++d)
                                           {
                                               p(i, j, k, d) +=
                                                   group(i, j, k, 4 * g + d + 1) / PhysConst::c;
                                           }
                                       }
                                   });
            }
            latest_spatial = std::make_unique<ImplicitGroupMomentTransportResult>(std::move(stage));
            return true;
        }
        // Host dispatch produces distinct CUDA kernels with right-sized local
        // matrices. A one/two-group run never allocates an eight-group stack.
        if (n <= 1)
        {
            return EvaluateSource<1>(options.groups, radiation, b, *cell_t, *transport, *absorption,
                                     geometry, dt, state, momentum, energy, physical, arithmetic,
                                     transfers, transfer_noise);
        }
        if (n <= 2)
        {
            return EvaluateSource<2>(options.groups, radiation, b, *cell_t, *transport, *absorption,
                                     geometry, dt, state, momentum, energy, physical, arithmetic,
                                     transfers, transfer_noise);
        }
        if (n <= 4)
        {
            return EvaluateSource<4>(options.groups, radiation, b, *cell_t, *transport, *absorption,
                                     geometry, dt, state, momentum, energy, physical, arithmetic,
                                     transfers, transfer_noise);
        }
        return EvaluateSource<8>(options.groups, radiation, b, *cell_t, *transport, *absorption,
                                 geometry, dt, state, momentum, energy, physical, arithmetic,
                                 transfers, transfer_noise);
    };
    auto make_heat = [&] (ParticleImpulseTransaction const &kick, amrex::MultiFab const &energy,
                          amrex::MultiFab &q)
    {
        q.setVal(0);
        for (amrex::MFIter it(q); it.isValid(); ++it)
        {
            auto const debit_a = energy.const_array(it), w = kick.ActualWork().const_array(it);
            auto const carry = kick.EnergyCarryChange().const_array(it);
            auto const out = q.array(it);
            amrex::ParallelFor(
                it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
                { out(i, j, k) = (debit_a(i, j, k) - w(i, j, k)) - carry(i, j, k); });
        }
        q.FillBoundary(geometry.periodicity());
    };
    auto caloric = [&] (amrex::MultiFab const &q, amrex::MultiFab &t, amrex::MultiFab &realized)
    {
        Copy(t, temperature);
        Copy(realized, q);
        auto const residual = callbacks.caloric.material_response(t, realized);
        auto const finite_t = t.is_finite();
        auto const minimum_t = t.min(0);
        auto const finite_heat = realized.is_finite();
        return Agree(std::isfinite(residual) && finite_t && minimum_t > 0 && finite_heat);
    };
    amrex::Real relaxation = options.relaxation;
    bool have_previous = false;
    for (int iteration = 0; iteration < options.max_iterations; ++iteration)
    {
        result.iterations = iteration + 1;
        trial_t->FillBoundary(geometry.periodicity());
        if (!source(*trial_t, *beta, *candidate_r, *impulse, *debit, *physical_heat, *precision,
                    *group_transfer, *group_precision))
        {
            return fail("Exact multigroup source solve rejected");
        }
        auto candidate_spatial = std::move(latest_spatial);
        ParticleImpulseTransaction kick;
        if (!kick.Stage(material, path, *impulse, true, options.assignment))
        {
            return fail("Native group recoil rejected");
        }
        amrex::MultiFab::Copy(*next_beta, kick.WorkVelocity(), 0, 0, 3, 0);
        next_beta->mult(1 / PhysConst::c);
        make_heat(kick, *debit, *heat);
        if (!caloric(*heat, *candidate_t, *actual_heat))
        {
            return fail("Native group caloric response rejected");
        }
        candidate_t->FillBoundary(geometry.periodicity());
        if (!source(*candidate_t, *next_beta, *check_r, *check_impulse, *check_debit,
                    *check_physical_heat, *check_precision, *check_group_transfer,
                    *check_group_precision))
        {
            return fail("Exact native group physical check rejected");
        }
        ParticleImpulseTransaction check_kick;
        if (!check_kick.Stage(material, path, *check_impulse, true, options.assignment))
        {
            return fail("Native group verification recoil rejected");
        }
        make_heat(check_kick, *check_debit, *check_heat);
        if (!caloric(*check_heat, *check_t, *probe_heat))
        {
            return fail("Native group caloric check rejected");
        }
        auto const raw_caloric =
            callbacks.caloric.material_residual(temperature, *candidate_t, *check_t);
        if (!Agree(std::isfinite(raw_caloric[0]) && std::isfinite(raw_caloric[1])))
        {
            return fail("Nonfinite native group caloric metric");
        }
        for (amrex::Real sign : {-1._rt, 1._rt})
        {
            Copy(*probe_heat, *check_heat);
            for (amrex::MFIter it(*probe_heat); it.isValid(); ++it)
            {
                auto const q = probe_heat->array(it);
                auto const original = check_heat->const_array(it);
                auto const noise = check_precision->const_array(it);
                auto const work_noise = check_kick.WorkAccountingRoundoffBound().const_array(it);
                auto const tolerance = options.tolerance;
                amrex::ParallelFor(it.validbox(),
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
                                       q(i, j, k) +=
                                           sign * (tolerance * std::abs(original(i, j, k)) +
                                                   noise(i, j, k, 0) + noise(i, j, k, 1) +
                                                   work_noise(i, j, k));
                                   });
            }
            probe_heat->FillBoundary(geometry.periodicity());
            auto &image = sign < 0 ? *lower_t : *upper_t;
            Copy(image, temperature);
            auto const probe = callbacks.caloric.material_response(image, *probe_heat);
            auto const finite_image = image.is_finite();
            auto const minimum_image = image.min(0);
            if (!Agree(std::isfinite(probe) && finite_image && minimum_image >= 0))
            {
                return fail("Native group local caloric bracket rejected");
            }
        }
        auto const local = callbacks.caloric.material_local_residual(
            temperature, *candidate_t, *check_t, *lower_t, *upper_t, options.tolerance);
        result.caloric_residual = local.maximum_relative_residual;
        amrex::Real worst_source = 0, worst_work = 0;
        // A sum over groups can conceal equal and opposite spectral errors.
        // Check every retained four-transfer at the accepted native T/secant.
        amrex::ReduceOps<amrex::ReduceOpMax> group_ops;
        amrex::ReduceData<amrex::Real> group_data(group_ops);
        auto const tolerance = options.tolerance;
        auto const spatial_transport = options.spatial_transport;
        for (amrex::MFIter it(*group_transfer); it.isValid(); ++it)
        {
            auto const q = group_transfer->const_array(it);
            auto const qc = check_group_transfer->const_array(it);
            auto const noise = group_precision->const_array(it);
            auto const check_noise = check_group_precision->const_array(it);
            auto const state = candidate_r->const_array(it), check_state = check_r->const_array(it);
            group_ops.eval(
                it.validbox(), group_data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<amrex::Real>
                {
                    amrex::Real worst = 0;
                    for (int component = 0; component < 4 * n; ++component)
                    {
                        auto const scale = amrex::max(std::abs(q(i, j, k, component)),
                                                      std::abs(qc(i, j, k, component)));
                        auto const bound = tolerance * scale + noise(i, j, k, component) +
                                           check_noise(i, j, k, component);
                        auto const error = std::abs(q(i, j, k, component) - qc(i, j, k, component));
                        auto const residual =
                            bound == 0
                                ? (error == 0 ? 0 : std::numeric_limits<amrex::Real>::infinity())
                                : tolerance * error / bound;
                        worst = amrex::max(worst, residual);
                        if (spatial_transport)
                        {
                            auto const group = component / 4;
                            auto const state_scale = std::abs(state(i, j, k, 4 * group)) +
                                                     std::abs(check_state(i, j, k, 4 * group));
                            auto const state_error = std::abs(state(i, j, k, component) -
                                                              check_state(i, j, k, component));
                            auto const state_bound =
                                tolerance * state_scale +
                                128 * std::numeric_limits<amrex::Real>::epsilon() *
                                    (std::abs(state(i, j, k, component)) +
                                     std::abs(check_state(i, j, k, component)));
                            worst = amrex::max(
                                worst, state_bound == 0
                                           ? (state_error == 0
                                                  ? 0
                                                  : std::numeric_limits<amrex::Real>::infinity())
                                           : tolerance * state_error / state_bound);
                        }
                    }
                    return {worst};
                });
        }
        worst_source = amrex::get<0>(group_data.value());
        for (amrex::MFIter it(*impulse); it.isValid(); ++it)
        {
            auto const p = impulse->const_array(it), pc = check_impulse->const_array(it);
            auto const h = heat->const_array(it), hc = check_heat->const_array(it);
            auto const q = physical_heat->const_array(it), precision_a = precision->const_array(it);
            auto const velocity = beta->const_array(it);
            auto const work = kick.RequestedWork().const_array(it);
            auto const noise = kick.WorkAccountingRoundoffBound().const_array(it);
            amrex::FArrayBox host(it.validbox(), 12, amrex::The_Pinned_Arena());
            amrex::FArrayBox values(it.validbox(), 12, amrex::The_Arena());
            auto const v = values.array();
            amrex::ParallelFor(
                it.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    v(i, j, k, 0) = h(i, j, k);
                    v(i, j, k, 1) = hc(i, j, k);
                    v(i, j, k, 2) = q(i, j, k);
                    v(i, j, k, 3) =
                        precision_a(i, j, k, 0) + precision_a(i, j, k, 1) + noise(i, j, k);
                    v(i, j, k, 4) = work(i, j, k);
                    v(i, j, k, 5) = 0;
                    v(i, j, k, 6) = noise(i, j, k);
                    v(i, j, k, 10) = 0;
                    for (int d = 0; d < 3; ++d)
                    {
                        v(i, j, k, 5) += PhysConst::c * velocity(i, j, k, d) * p(i, j, k, d);
                        v(i, j, k, 7 + d) = PhysConst::c * (pc(i, j, k, d) - p(i, j, k, d));
                        v(i, j, k, 10) +=
                            PhysConst::c * (std::abs(pc(i, j, k, d)) + std::abs(p(i, j, k, d)));
                    }
                    v(i, j, k, 11) = precision_a(i, j, k, 2);
                });
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.dataPtr(),
                             values.dataPtr() + values.size(), host.dataPtr());
            for (long point = 0; point < it.validbox().numPts(); ++point)
            {
                auto const value = [&] (int component) { return host.dataPtr(component)[point]; };
                worst_source = std::max(worst_source,
                                        Relative(std::abs(value(0) - value(1)),
                                                 std::max(std::abs(value(0)), std::abs(value(1))),
                                                 value(3), options.tolerance));
                worst_work =
                    std::max(worst_work, Relative(std::abs(value(0) - value(2)),
                                                  std::max(std::abs(value(0)), std::abs(value(2))),
                                                  value(3), options.tolerance));
                worst_work = std::max(worst_work, Relative(std::abs(value(4) - value(5)),
                                                           std::abs(value(4)) + std::abs(value(5)),
                                                           value(6), options.tolerance));
                for (int d = 0; d < 3; ++d)
                {
                    worst_source =
                        std::max(worst_source, Relative(std::abs(value(7 + d)), value(10),
                                                        value(11), options.tolerance));
                }
            }
        }
        amrex::ParallelDescriptor::ReduceRealMax(worst_source);
        amrex::ParallelDescriptor::ReduceRealMax(worst_work);
        result.source_residual = worst_source;
        result.work_residual = worst_work;
        if (options.verbose)
        {
            amrex::Print() << "Native groups iteration=" << result.iterations
                           << " source=" << worst_source << " work=" << worst_work
                           << " caloric=" << result.caloric_residual << " T=" << trial_t->min(0)
                           << " candidate=" << candidate_t->min(0) << '\n';
        }
        if (Agree(local.valid && local.converged && worst_source <= options.tolerance &&
                  worst_work <= options.tolerance))
        {
            result.actual_kinetic_work = kick.ActualWork().sum(0);
            result.energy_carry_change = kick.EnergyCarryChange().sum(0);
            amrex::Real change =
                actual_heat->sum(0) + result.actual_kinetic_work + result.energy_carry_change;
            amrex::Real scale = actual_heat->norm1(0) + kick.ActualWork().norm1(0) +
                                kick.EnergyCarryChange().norm1(0);
            for (int g = 0; g < n; ++g)
            {
                change += candidate_r->sum(4 * g) - radiation.sum(4 * g);
                scale += candidate_r->norm1(4 * g) + radiation.norm1(4 * g);
            }
            if (candidate_spatial)
            {
                change += candidate_spatial->boundary_exchange[0];
                scale += std::abs(candidate_spatial->boundary_exchange[0]);
            }
            result.raw_energy_residual = scale == 0 ? std::abs(change) : std::abs(change) / scale;
            result.energy_inventory_defect = change;
#if defined(WARPX_DIM_RZ)
            amrex::Real rotation_scale = 0;
            for (int d = 0; d < 2; ++d)
            {
                rotation_scale +=
                    PhysConst::c * (impulse->norm1(d) + kick.ConjugateImpulse().norm1(d) +
                                    kick.ConjugateMomentumCarryChange().norm1(d));
            }
#endif
            for (int d = 0; d < 3; ++d)
            {
                auto momentum_change = PhysConst::c * (kick.ConjugateImpulse().sum(d) +
                                                       kick.ConjugateMomentumCarryChange().sum(d));
                auto momentum_scale = PhysConst::c * (kick.ConjugateImpulse().norm1(d) +
                                                      kick.ConjugateMomentumCarryChange().norm1(d));
                for (int g = 0; g < n; ++g)
                {
                    momentum_change +=
                        candidate_r->sum(4 * g + d + 1) - radiation.sum(4 * g + d + 1);
                    momentum_scale +=
                        candidate_r->norm1(4 * g + d + 1) + radiation.norm1(4 * g + d + 1);
                }
                if (candidate_spatial)
                {
                    auto const boundary = candidate_spatial->boundary_exchange[d + 1];
                    auto const geometric = candidate_spatial->geometric_exchange[d + 1];
                    momentum_change += boundary - geometric;
                    momentum_scale += std::abs(boundary) + std::abs(geometric);
                }
                result.raw_momentum_residual =
                    std::max(result.raw_momentum_residual,
                             momentum_scale == 0 ? std::abs(momentum_change)
                                                 : std::abs(momentum_change) / momentum_scale);
#if defined(WARPX_DIM_RZ)
                auto const bound =
                    options.energy_tolerance * momentum_scale +
                    (d < 2 ? 64 * std::numeric_limits<amrex::Real>::epsilon() * rotation_scale : 0);
                result.momentum_balance_residual = std::max(
                    result.momentum_balance_residual,
                    bound == 0 ? std::abs(momentum_change) : std::abs(momentum_change) / bound);
#endif
            }
#if defined(WARPX_DIM_RZ)
            auto const angular = AngularBalance(*candidate_r, radiation, *impulse, kick, geometry,
                                                n, options.energy_tolerance);
            result.raw_angular_momentum_residual = angular[0];
            result.angular_balance_residual = angular[1];
            bool const momentum_balanced = std::isfinite(result.momentum_balance_residual) &&
                                           std::isfinite(angular[0]) && std::isfinite(angular[1]) &&
                                           result.momentum_balance_residual <= 1 &&
                                           result.angular_balance_residual <= 1;
#else
            bool const momentum_balanced = result.raw_momentum_residual <= options.energy_tolerance;
#endif
            if (!Agree(std::isfinite(result.raw_energy_residual) &&
                       std::isfinite(result.raw_momentum_residual) &&
                       result.raw_energy_residual <= options.energy_tolerance && momentum_balanced))
            {
                return fail("Native group source inventory rejected");
            }
            result.requested_material_energy.define(boxes, distribution, 1, 1);
            result.actual_material_impulse.define(boxes, distribution, 3, 0);
            result.actual_material_work.define(boxes, distribution, 1, 0);
            result.conjugate_material_impulse.define(boxes, distribution, 3, 0);
            Copy(result.requested_material_energy, *heat);
            amrex::MultiFab::Copy(result.actual_material_impulse, kick.ActualImpulse(), 0, 0, 3, 0);
            amrex::MultiFab::Copy(result.actual_material_work, kick.ActualWork(), 0, 0, 1, 0);
            amrex::MultiFab::Copy(result.conjugate_material_impulse, kick.ConjugateImpulse(), 0, 0,
                                  3, 0);
            kick.Commit();
            if (!options.defer_live_commit && !material.Commit())
            {
                return fail("Native group private material commit rejected");
            }
#if defined(WARPX_DIM_RZ)
            FillGroupGhosts(*candidate_r, geometry, n);
#else
            candidate_r->FillBoundary(geometry.periodicity());
#endif
            candidate_t->FillBoundary(geometry.periodicity());
            Copy(radiation, *candidate_r);
            Copy(temperature, *candidate_t);
            Copy(realized_heat, *actual_heat);
            result.spatial_trial = std::move(candidate_spatial);
            if (options.defer_live_commit)
            {
                result.private_material = std::move(private_material);
            }
            radiation.FillBoundary(geometry.periodicity());
            realized_heat.FillBoundary(geometry.periodicity());
            result.valid = true;
            return result;
        }
        Copy(*t_residual, *candidate_t);
        amrex::MultiFab::Subtract(*t_residual, *trial_t, 0, 0, 1, 0);
        if (have_previous)
        {
            auto difference = node();
            Copy(*difference, *t_residual);
            amrex::MultiFab::Subtract(*difference, *previous_t_residual, 0, 0, 1, 0);
            auto const denominator = amrex::MultiFab::Dot(*difference, 0, *difference, 0, 1, 0);
            auto const numerator =
                amrex::MultiFab::Dot(*previous_t_residual, 0, *difference, 0, 1, 0);
            if (denominator > 0 && std::isfinite(denominator) && std::isfinite(numerator))
            {
                relaxation = std::clamp(-relaxation * numerator / denominator, 0.01_rt, 1.0_rt);
            }
        }
        Copy(*previous_t_residual, *t_residual);
        have_previous = true;
        amrex::MultiFab::Saxpy(*trial_t, relaxation, *t_residual, 0, 0, 1, 0);
        for (amrex::MFIter it(*beta); it.isValid(); ++it)
        {
            auto const b = beta->array(it);
            auto const target = next_beta->const_array(it);
            auto const damping = options.relaxation;
            amrex::ParallelFor(
                it.validbox(), 3, [=] AMREX_GPU_DEVICE(int i, int j, int k, int d)
                { b(i, j, k, d) += damping * (target(i, j, k, d) - b(i, j, k, d)); });
        }
    }
    return fail("Native multigroup source iteration budget exhausted");
#endif
}
} // namespace warpx::radiation
