/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "PacketMaterialWorkIteration.H"

#include "WarpX.H"

#include <AMReX_GpuLaunch.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>
#include <utility>

namespace warpx::radiation
{
    namespace
    {
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
        RelativeResidual (amrex::Real difference, amrex::Real scale) noexcept
        {
            if (!amrex::Math::isfinite(difference) || !amrex::Math::isfinite(scale))
            {
                return std::numeric_limits<amrex::Real>::infinity();
            }
            if (scale > 0) { return difference / scale; }
            return difference == 0 ? 0 : std::numeric_limits<amrex::Real>::infinity();
        }

        bool
        ValidReplayOutputs (
            amrex::MultiFab const& total_impulse, amrex::MultiFab const& radiation_debit,
            amrex::MultiFab const& uncancelled_energy_scale)
        {
            amrex::Gpu::DeviceScalar<int> invalid(0);
            auto* const invalid_ptr = invalid.dataPtr();
            for (amrex::MFIter iterator(total_impulse); iterator.isValid(); ++iterator)
            {
                auto const impulse = total_impulse.const_array(iterator);
                auto const debit = radiation_debit.const_array(iterator);
                auto const scale = uncancelled_energy_scale.const_array(iterator);
                amrex::For(iterator.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    bool valid = amrex::Math::isfinite(debit(i, j, k)) &&
                        amrex::Math::isfinite(scale(i, j, k)) && scale(i, j, k) >= 0;
                    for (int d = 0; d < 3; ++d)
                    {
                        valid = valid && amrex::Math::isfinite(impulse(i, j, k, d));
                    }
                    if (!valid) { amrex::HostDevice::Atomic::Add(invalid_ptr, 1); }
                });
            }
            int invalid_count = invalid.dataValue();
            amrex::ParallelDescriptor::ReduceIntMax(invalid_count);
            return invalid_count == 0;
        }

        amrex::GpuArray<amrex::Real, 5>
        PacketMaterialResiduals (
            amrex::MultiFab const& current_impulse, amrex::MultiFab const& candidate_impulse,
            amrex::MultiFab const& radiation_debit,
            amrex::MultiFab const& uncancelled_energy_scale,
            ParticleImpulseTransaction const& staged)
        {
            amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax,
                             amrex::ReduceOpMax, amrex::ReduceOpMax,
                             amrex::ReduceOpMax> ops;
            amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                              amrex::Real> data(ops);
            using Tuple = typename decltype(data)::Type;
            for (amrex::MFIter iterator(candidate_impulse); iterator.isValid(); ++iterator)
            {
                auto const current = current_impulse.const_array(iterator);
                auto const candidate = candidate_impulse.const_array(iterator);
                auto const debit = radiation_debit.const_array(iterator);
                auto const scale = uncancelled_energy_scale.const_array(iterator);
                auto const requested = staged.RequestedWork().const_array(iterator);
                auto const actual = staged.ActualWork().const_array(iterator);
                auto const carry = staged.EnergyCarryChange().const_array(iterator);
                auto const accounting_bound =
                    staged.WorkAccountingRoundoffBound().const_array(iterator);
                auto const actual_impulse = staged.ConjugateImpulse().const_array(iterator);
                auto const impulse_carry = staged.ConjugateMomentumCarryChange().const_array(iterator);
                ops.eval(iterator.validbox(), data, [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) -> Tuple
                {
                    amrex::Real impulse_residual = 0;
                    amrex::Real material_impulse_residual = 0;
                    amrex::Real impulse_scale = 0;
#if defined(WARPX_DIM_RZ)
                    // Rotation between Cartesian particle kicks and local
                    // cylindrical components mixes radial/theta operands.
                    // A pure torque has zero requested radial component, but
                    // its radial cancellation is conditioned by both transverse
                    // components, not by the tiny residual alone.
                    amrex::Real transverse_material_scale = 0;
                    for (int d = 0; d < 2; ++d) {
                        transverse_material_scale += std::abs(current(i,j,k,d)) +
                            std::abs(actual_impulse(i,j,k,d)) + std::abs(impulse_carry(i,j,k,d));
                    }
#endif
                    for (int d = 0; d < 3; ++d)
                    {
                        impulse_scale += std::abs(candidate(i, j, k, d))
                                       + std::abs(current(i, j, k, d));
                    }
                    for (int d = 0; d < 3; ++d)
                    {
                        amrex::Real const difference =
                            std::abs(candidate(i, j, k, d) - current(i, j, k, d));
                        impulse_residual = amrex::max(
                            impulse_residual, RelativeResidual(difference, impulse_scale));
                        auto const material_difference = std::abs(current(i, j, k, d)
                            - actual_impulse(i, j, k, d) - impulse_carry(i, j, k, d));
                        auto material_scale = std::abs(current(i, j, k, d))
                            + std::abs(actual_impulse(i, j, k, d))
                            + std::abs(impulse_carry(i, j, k, d));
#if defined(WARPX_DIM_RZ)
                        if (d < 2) { material_scale = transverse_material_scale; }
#endif
                        material_impulse_residual = amrex::max(material_impulse_residual,
                            RelativeResidual(material_difference, material_scale));
                    }

                    amrex::Real const energy_difference =
                        std::abs(debit(i, j, k) - requested(i, j, k));
                    amrex::Real const energy_residual = RelativeResidual(
                        energy_difference, scale(i, j, k));

                    // Numerical/partition residuals are diagnostics, not accounts.
                    // Nearest-cell ownership makes this a cell-wise identity.
                    amrex::Real const accounted = actual(i, j, k) + carry(i, j, k);
                    amrex::Real const material_difference =
                        std::abs(requested(i, j, k) - accounted);
                    amrex::Real const material_scale = std::abs(requested(i, j, k)) +
                        std::abs(actual(i, j, k)) + std::abs(carry(i, j, k));
                    amrex::Real const material_residual =
                        RelativeResidual(material_difference, material_scale);
                    amrex::Real bound_excess = std::numeric_limits<amrex::Real>::infinity();
                    if (amrex::Math::isfinite(material_difference) &&
                        amrex::Math::isfinite(material_scale) &&
                        accounting_bound(i, j, k) >= 0 &&
                        amrex::Math::isfinite(accounting_bound(i, j, k)))
                    {
                        bound_excess = RelativeResidual(amrex::max(amrex::Real(0),
                            material_difference - accounting_bound(i, j, k)), material_scale);
                    }
                    return {impulse_residual, energy_residual, material_residual,
                            material_impulse_residual, bound_excess};
                });
            }
            auto const values = data.value();
            amrex::GpuArray<amrex::Real, 5> result{
                amrex::get<0>(values), amrex::get<1>(values), amrex::get<2>(values),
                amrex::get<3>(values), amrex::get<4>(values)
            };
            amrex::ParallelDescriptor::ReduceRealMax(result.data(), 5);
            return result;
        }
    }

    std::string
    PacketMaterialWorkIterationReasonName (PacketMaterialWorkIterationReason reason)
    {
        switch (reason)
        {
        case PacketMaterialWorkIterationReason::None: return "none";
        case PacketMaterialWorkIterationReason::Success: return "success";
        case PacketMaterialWorkIterationReason::InvalidOptions: return "invalid options";
        case PacketMaterialWorkIterationReason::InvalidLayout: return "invalid layout";
        case PacketMaterialWorkIterationReason::StageFailed: return "stage failed";
        case PacketMaterialWorkIterationReason::ReplayRejected: return "replay rejected";
        case PacketMaterialWorkIterationReason::InvalidReplayOutput:
            return "invalid replay output";
        case PacketMaterialWorkIterationReason::MaterialWorkMismatch:
            return "material work mismatch";
        case PacketMaterialWorkIterationReason::MaterialImpulseMismatch:
            return "material impulse mismatch";
        case PacketMaterialWorkIterationReason::NotConverged: return "not converged";
        }
        return "unknown";
    }

    bool
    TryIteratePacketMaterialWork (
        ParticleImpulseMaterial& material, std::string const& path,
        amrex::MultiFab const& initial_total_impulse, PacketMaterialWorkReplay const& replay,
        PacketMaterialWorkIterationOptions const& options,
        std::unique_ptr<ParticleImpulseTransaction>& accepted_transaction,
        PacketMaterialWorkIterationReport& report)
    {
        int invalid_options = 0;
#if defined(WARPX_DIM_RZ)
        bool const valid_assignment =
            options.assignment == ParticleImpulseAssignment::RZAngularConservative;
#else
        bool const valid_assignment = options.assignment == ParticleImpulseAssignment::NearestCell;
#endif
        if (options.max_iterations <= 0 || !replay ||
            !(options.damping > 0) || !(options.damping <= 1) ||
            !amrex::Math::isfinite(options.damping) ||
            !(options.impulse_relative_tolerance > 0) ||
            !amrex::Math::isfinite(options.impulse_relative_tolerance) ||
            !(options.work_relative_tolerance > 0) ||
            !amrex::Math::isfinite(options.work_relative_tolerance) ||
            !valid_assignment ||
            std::numeric_limits<amrex::Real>::digits < 53 ||
            std::numeric_limits<amrex::ParticleReal>::digits < 53)
        {
            invalid_options = 1;
        }
        int minimum_iterations = options.max_iterations;
        int maximum_iterations = options.max_iterations;
        amrex::ParallelDescriptor::ReduceIntMin(minimum_iterations);
        amrex::ParallelDescriptor::ReduceIntMax(maximum_iterations);
        int minimum_assignment = static_cast<int>(options.assignment);
        int maximum_assignment = minimum_assignment;
        amrex::ParallelDescriptor::ReduceIntMin(minimum_assignment);
        amrex::ParallelDescriptor::ReduceIntMax(maximum_assignment);
        if (minimum_assignment != maximum_assignment) { invalid_options = 1; }
        amrex::Real minimum_options[3] = {options.damping, options.impulse_relative_tolerance,
                                         options.work_relative_tolerance};
        amrex::Real maximum_options[3] = {options.damping, options.impulse_relative_tolerance,
                                         options.work_relative_tolerance};
        amrex::ParallelDescriptor::ReduceRealMin(minimum_options, 3);
        amrex::ParallelDescriptor::ReduceRealMax(maximum_options, 3);
        if (minimum_iterations != maximum_iterations) { invalid_options = 1; }
        for (int d = 0; d < 3; ++d)
        {
            if (minimum_options[d] != maximum_options[d]) { invalid_options = 1; }
        }
        amrex::ParallelDescriptor::ReduceIntMax(invalid_options);
        if (invalid_options != 0)
        {
            report = PacketMaterialWorkIterationReport{};
            report.reason = PacketMaterialWorkIterationReason::InvalidOptions;
            return false;
        }

        int invalid_layout = 0;
        auto const& geometry = WarpX::GetInstance().Geom(0);
#if defined(WARPX_DIM_RZ)
        bool const valid_geometry = (geometry.Coord() == 0 || geometry.Coord() == 1) &&
            !geometry.isPeriodic(0) && geometry.ProbLo(0) >= 0;
#else
        bool const valid_geometry = geometry.Coord() == 0;
#endif
        if (path.empty() || initial_total_impulse.nComp() != 3 ||
            !initial_total_impulse.ixType().cellCentered() ||
            WarpX::GetInstance().finestLevel() != 0 ||
            !valid_geometry ||
            initial_total_impulse.boxArray() != WarpX::GetInstance().boxArray(0) ||
            initial_total_impulse.DistributionMap() != WarpX::GetInstance().DistributionMap(0))
        {
            invalid_layout = 1;
        }
        amrex::ParallelDescriptor::ReduceIntMax(invalid_layout);
        if (invalid_layout != 0)
        {
            report = PacketMaterialWorkIterationReport{};
            report.reason = PacketMaterialWorkIterationReason::InvalidLayout;
            return false;
        }
        if (!initial_total_impulse.is_finite())
        {
            report = PacketMaterialWorkIterationReport{};
            report.reason = PacketMaterialWorkIterationReason::InvalidLayout;
            return false;
        }

        auto const& box_array = initial_total_impulse.boxArray();
        auto const& distribution_map = initial_total_impulse.DistributionMap();
        amrex::MultiFab current_impulse(box_array, distribution_map, 3, 0);
        amrex::MultiFab::Copy(current_impulse, initial_total_impulse, 0, 0, 3, 0);

        PacketMaterialWorkIterationReport next_report;
        next_report.candidate_total_impulse.define(box_array, distribution_map, 3, 0);
        next_report.candidate_radiation_debit.define(box_array, distribution_map, 1, 0);
        next_report.candidate_uncancelled_energy_scale.define(
            box_array, distribution_map, 1, 0);

        amrex::Real const epsilon = std::numeric_limits<amrex::Real>::epsilon();
        amrex::Real const roundoff = 256 * epsilon;
        for (int iteration = 0; iteration < options.max_iterations; ++iteration)
        {
            auto staged = std::make_unique<ParticleImpulseTransaction>();
            bool const stage_valid = staged->Stage(
                material, path, current_impulse, /*need_work_velocity=*/true,
                options.assignment);
            if (!stage_valid)
            {
                next_report.iterations = iteration + 1;
                next_report.reason = PacketMaterialWorkIterationReason::StageFailed;
                report = std::move(next_report);
                return false;
            }

            amrex::MultiFab candidate_impulse(box_array, distribution_map, 3, 0);
            amrex::MultiFab radiation_debit(box_array, distribution_map, 1, 0);
            amrex::MultiFab uncancelled_energy_scale(box_array, distribution_map, 1, 0);
            amrex::Real const poison = std::numeric_limits<amrex::Real>::quiet_NaN();
            candidate_impulse.setVal(poison);
            radiation_debit.setVal(poison);
            uncancelled_energy_scale.setVal(poison);

            bool const replay_accepted = replay(
                *staged, candidate_impulse, radiation_debit, uncancelled_energy_scale);
            int replay_rejected = replay_accepted ? 0 : 1;
            amrex::ParallelDescriptor::ReduceIntMax(replay_rejected);
            if (replay_rejected != 0)
            {
                next_report.iterations = iteration + 1;
                next_report.reason = PacketMaterialWorkIterationReason::ReplayRejected;
                report = std::move(next_report);
                return false;
            }

            int invalid_replay_layout = 0;
            int expected_components = 3;
            for (auto const* field : {&candidate_impulse, &radiation_debit,
                                     &uncancelled_energy_scale})
            {
                if (field->nComp() != expected_components || field->boxArray() != box_array ||
                    field->DistributionMap() != distribution_map)
                {
                    invalid_replay_layout = 1;
                }
                expected_components = 1;
            }
            amrex::ParallelDescriptor::ReduceIntMax(invalid_replay_layout);
            if (invalid_replay_layout != 0)
            {
                next_report.iterations = iteration + 1;
                next_report.reason = PacketMaterialWorkIterationReason::InvalidReplayOutput;
                report = std::move(next_report);
                return false;
            }

            if (!ValidReplayOutputs(
                    candidate_impulse, radiation_debit, uncancelled_energy_scale))
            {
                next_report.iterations = iteration + 1;
                next_report.reason = PacketMaterialWorkIterationReason::InvalidReplayOutput;
                report = std::move(next_report);
                return false;
            }

            auto const residuals = PacketMaterialResiduals(
                current_impulse, candidate_impulse, radiation_debit,
                uncancelled_energy_scale, *staged);
            next_report.iterations = iteration + 1;
            next_report.max_impulse_residual = residuals[0];
            next_report.max_radiation_work_residual = residuals[1];
            next_report.max_material_work_residual = residuals[2];
            next_report.max_material_impulse_residual = residuals[3];
            next_report.max_material_work_excess_residual = residuals[4];
            next_report.requested_work = staged->RequestedWork().sum(0);
            next_report.actual_work = staged->ActualWork().sum(0);
            next_report.energy_carry_change = staged->EnergyCarryChange().sum(0);
            next_report.work_partition_residual = staged->WorkPartitionResidual().sum(0);
            amrex::MultiFab::Copy(
                next_report.candidate_total_impulse, candidate_impulse, 0, 0, 3, 0);
            amrex::MultiFab::Copy(
                next_report.candidate_radiation_debit, radiation_debit, 0, 0, 1, 0);
            amrex::MultiFab::Copy(
                next_report.candidate_uncancelled_energy_scale, uncancelled_energy_scale,
                0, 0, 1, 0);

            bool const impulse_converged =
                residuals[0] <= options.impulse_relative_tolerance + roundoff;
            bool const radiation_work_converged =
                residuals[1] <= options.work_relative_tolerance + roundoff;
            bool const material_work_converged =
                residuals[4] <= options.work_relative_tolerance + roundoff;
            if (!material_work_converged)
            {
                next_report.reason =
                    PacketMaterialWorkIterationReason::MaterialWorkMismatch;
                report = std::move(next_report);
                return false;
            }
            if (residuals[3] > options.impulse_relative_tolerance + roundoff)
            {
                next_report.reason = PacketMaterialWorkIterationReason::MaterialImpulseMismatch;
                report = std::move(next_report);
                return false;
            }
            if (impulse_converged && radiation_work_converged)
            {
                next_report.converged = true;
                next_report.reason = PacketMaterialWorkIterationReason::Success;
                report = std::move(next_report);
                accepted_transaction = std::move(staged);
                return true;
            }
            if (iteration + 1 == options.max_iterations)
            {
                next_report.reason = PacketMaterialWorkIterationReason::NotConverged;
                report = std::move(next_report);
                return false;
            }

            amrex::MultiFab::LinComb(
                current_impulse, options.damping, candidate_impulse, 0,
                1 - options.damping, current_impulse, 0, 0, 3, 0);
        }

        next_report.reason = PacketMaterialWorkIterationReason::NotConverged;
        report = std::move(next_report);
        return false;
    }
}
