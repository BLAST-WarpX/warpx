/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "MovingThermalDiffusionField.H"

#include "CellVolume.H"

#include "Utils/WarpXConst.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

using namespace amrex::literals;

namespace warpx::radiation {
namespace {
// Cell energies are extensive; face currents and pressure use energy density.
// Never evaluate a radial ghost volume across the axis or a physical wall.
struct ThermalMetric
{
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> spacing;
    amrex::GpuArray<int, AMREX_SPACEDIM> periodic, low, high;
    amrex::Real radial_low, cartesian_volume;

    AMREX_GPU_HOST_DEVICE amrex::Real Volume (int i) const noexcept
    {
#if defined(WARPX_DIM_RZ)
        return CellVolume(i, spacing, {radial_low, 0}, {low[0], low[1], 0});
#else
        amrex::ignore_unused(i);
        return cartesian_volume;
#endif
    }
    AMREX_GPU_HOST_DEVICE amrex::Real Area (int d, int i) const noexcept
    {
#if defined(WARPX_DIM_RZ)
        if (d == 0) {
            return 2 * MathConst::pi *
                (radial_low + (i - low[0]) * spacing[0]) * spacing[1];
        }
#endif
        return Volume(i) / spacing[d];
    }
    AMREX_GPU_HOST_DEVICE bool Wall (int d, int i, int j, int k) const noexcept
    {
        int const index[3] = {i, j, k};
        return !periodic[d] && (index[d] == low[d] || index[d] == high[d] + 1);
    }
};

void FillClosedTemperatureGhosts (amrex::MultiFab& temperature, amrex::Geometry const& geometry)
{
    temperature.FillBoundary(geometry.periodicity());
    if (geometry.isAllPeriodic()) { return; }
    auto const domain = amrex::convert(geometry.Domain(), temperature.ixType());
    auto const lo = amrex::lbound(domain), hi = amrex::ubound(domain);
    amrex::GpuArray<int, AMREX_SPACEDIM> periodic{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { periodic[d] = geometry.isPeriodic(d); }
    for (amrex::MFIter mfi(temperature); mfi.isValid(); ++mfi) {
        auto const t = temperature.array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            int index[3] = {i,j,k};
            int const lower[3] = {lo.x,lo.y,lo.z}, upper[3] = {hi.x,hi.y,hi.z};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (!periodic[d]) { index[d] = amrex::min(upper[d], amrex::max(lower[d],index[d])); }
            }
            if (index[0] != i || index[1] != j || index[2] != k) {
                t(i,j,k) = t(index[0],index[1],index[2]);
            }
        });
    }
}
} // namespace

MovingThermalDiffusionFieldResult
TryAdvanceMovingThermalDiffusionField (
    amrex::MultiFab const& old_radiation, amrex::MultiFab const& old_nodal_temperature,
    amrex::MultiFab const& work_velocity, amrex::MultiFab const& requested_work,
    amrex::Geometry const& geometry, amrex::Real time, amrex::Real dt,
    MovingThermalDiffusionFieldOptions const& options, CoupledImplicitCallbacks const& callbacks)
{
    using Failure = MovingThermalDiffusionFieldFailure;
    auto fail = [] (Failure reason) {
        MovingThermalDiffusionFieldResult result;
        result.failure = reason;
        return result;
    };
    int invalid_geometry = geometry.Coord() != 0 || sizeof(amrex::Real) != sizeof(double);
#if defined(WARPX_DIM_RZ)
    invalid_geometry = (geometry.Coord() != 0 && geometry.Coord() != 1) ||
        geometry.isPeriodic(0) || !std::isfinite(geometry.ProbLo(0)) || geometry.ProbLo(0) < 0 ||
        sizeof(amrex::Real) != sizeof(double);
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    invalid_geometry = 1;
#endif
    amrex::ParallelDescriptor::ReduceIntMax(invalid_geometry);
    if (invalid_geometry) {
        return fail(Failure::UnsupportedGeometry);
    }

    auto const& boxes = old_radiation.boxArray();
    auto const& distribution = old_radiation.DistributionMap();
    auto const& coupled = options.coupled;
    auto const& diffusion = coupled.diffusion;
    amrex::Real const chi_t = options.transport_opacity;
    amrex::Real const chi_a = options.absorption_opacity;
    bool const variable_opacity = options.variable_opacity;
    auto const spacing = geometry.CellSizeArray();
    amrex::Real dx = spacing[0];
    amrex::Real length = geometry.ProbHi(0) - geometry.ProbLo(0);
    amrex::Real volume = 1;
    bool valid_metric = true;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        auto const extent = geometry.ProbHi(d) - geometry.ProbLo(d);
        valid_metric = valid_metric && std::isfinite(spacing[d]) && spacing[d] > 0 &&
            std::isfinite(extent) && extent > 0;
        dx = std::min(dx, spacing[d]);
        length = std::max(length, extent);
        volume *= spacing[d];
    }
    ThermalMetric metric{};
    metric.spacing = spacing;
    metric.radial_low = geometry.ProbLo(0);
    metric.cartesian_volume = volume;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        metric.periodic[d] = geometry.isPeriodic(d);
        metric.low[d] = geometry.Domain().smallEnd(d);
        metric.high[d] = geometry.Domain().bigEnd(d);
    }
    int invalid = !(
        std::isfinite(time) && dt > 0 && std::isfinite(dt) && std::isfinite(time + dt) &&
        time + dt > time &&
        (variable_opacity || (chi_t > 0 && std::isfinite(chi_t) && chi_a >= 0 && chi_a <= chi_t &&
                              chi_t * dx >= amrex::max(8.0_rt, diffusion.minimum_optical_depth))) &&
        valid_metric && volume > 0 && std::isfinite(volume) && diffusion.tolerance > 0 &&
        std::isfinite(diffusion.tolerance) && diffusion.linear_tolerance > 0 &&
        diffusion.linear_tolerance < diffusion.tolerance && diffusion.minimum_optical_depth > 0 &&
        std::isfinite(diffusion.minimum_optical_depth) && diffusion.max_iterations > 0 &&
        diffusion.max_linear_iterations > 0 && diffusion.nonlinear_relaxation > 0 &&
        diffusion.nonlinear_relaxation <= 1 && diffusion.energy_groups.m_num_groups == 1 &&
        coupled.max_iterations > 0 && coupled.material_tolerance > 0 &&
        std::isfinite(coupled.material_tolerance) && coupled.energy_tolerance > 0 &&
        std::isfinite(coupled.energy_tolerance) && coupled.relaxation > 0 &&
        coupled.relaxation <= 1 && callbacks.coefficients && callbacks.material_response &&
        old_radiation.ixType().cellCentered() && old_radiation.nComp() == 1 &&
        old_nodal_temperature.ixType().nodeCentered() && old_nodal_temperature.nComp() == 1 &&
        old_nodal_temperature.nGrowVect().allGE(1) &&
        amrex::convert(boxes, old_nodal_temperature.ixType()) == old_nodal_temperature.boxArray() &&
        old_nodal_temperature.DistributionMap() == distribution &&
        work_velocity.boxArray() == boxes && work_velocity.DistributionMap() == distribution &&
        work_velocity.nComp() == 3 && requested_work.boxArray() == boxes &&
        requested_work.DistributionMap() == distribution && requested_work.nComp() == 1);
    invalid = invalid || !boxes.ok() || !boxes.isDisjoint() ||
        boxes.numPts() != geometry.Domain().numPts();
    for (int n = 0; n < boxes.size(); ++n) {
        invalid = invalid || !geometry.Domain().contains(boxes[n]);
    }
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    if (invalid) {
        return fail(Failure::InvalidState);
    }
    // All branch/loop controls agree before any rank enters callback
    // collectives.
    amrex::Real parameters[] = {time,
                                dt,
                                variable_opacity ? 0.0_rt : chi_t,
                                variable_opacity ? 0.0_rt : chi_a,
                                variable_opacity ? 1.0_rt : 0.0_rt,
                                options.require_local_ideal_accuracy ? 1.0_rt : 0.0_rt,
                                dx,
                                length,
                                diffusion.tolerance,
                                diffusion.linear_tolerance,
                                diffusion.minimum_optical_depth,
                                amrex::Real(diffusion.max_iterations),
                                amrex::Real(diffusion.max_linear_iterations),
                                diffusion.nonlinear_relaxation,
                                amrex::Real(coupled.max_iterations),
                                coupled.material_tolerance,
                                coupled.energy_tolerance,
                                coupled.relaxation,
                                amrex::Real(geometry.Coord()),
                                callbacks.material_residual ? 1.0_rt : 0.0_rt,
                                callbacks.material_local_residual ? 1.0_rt : 0.0_rt,
                                diffusion.use_incremental_form ? 1.0_rt : 0.0_rt,
                                diffusion.reuse_operator ? 1.0_rt : 0.0_rt};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        for (auto value : {spacing[d], geometry.ProbLo(d), geometry.ProbHi(d),
                           amrex::Real(metric.periodic[d]), amrex::Real(metric.low[d]),
                           amrex::Real(metric.high[d])}) {
            auto lo = value;
            auto hi = value;
            amrex::ParallelDescriptor::ReduceRealMin(lo);
            amrex::ParallelDescriptor::ReduceRealMax(hi);
            if (lo != hi) { invalid = 1; }
        }
    }
    for (auto value : parameters) {
        auto lo = value;
        auto hi = value;
        amrex::ParallelDescriptor::ReduceRealMin(lo);
        amrex::ParallelDescriptor::ReduceRealMax(hi);
        if (lo != hi) {
            invalid = 1;
        }
    }
    if (invalid) {
        return fail(Failure::InvalidState);
    }
    bool const finite_energy = old_radiation.is_finite(0, 1, 0);
    bool const finite_temperature = old_nodal_temperature.is_finite(0, 1, 0);
    bool const finite_velocity = work_velocity.is_finite(0, 3, 0);
    bool const finite_work = requested_work.is_finite(0, 1, 0);
    auto const min_energy = old_radiation.min(0);
    auto const min_temperature = old_nodal_temperature.min(0);
    if (!finite_energy || !finite_temperature || !finite_velocity || !finite_work ||
        min_energy < 0 || min_temperature < 0) {
        return fail(Failure::InvalidState);
    }

    amrex::MultiFab old(boxes, distribution, 1, 1);
    amrex::MultiFab velocity(boxes, distribution, 3, 1);
    amrex::MultiFab shifted(boxes, distribution, 1, 1);
    amrex::MultiFab advected(boxes, distribution, 1, 1);
    amrex::MultiFab trial_shifted(boxes, distribution, 1, 1);
    amrex::MultiFab trial_opacity(boxes, distribution, 1, 1);
    amrex::MultiFab trial_ratio(boxes, distribution, 1, 1);
    amrex::MultiFab heat(boxes, distribution, 1, 1);
    amrex::MultiFab temperature(old_nodal_temperature.boxArray(), distribution, 1,
                                old_nodal_temperature.nGrowVect());
    old.setVal(0);
    velocity.setVal(0);
    shifted.setVal(0);
    advected.setVal(0);
    trial_shifted.setVal(0);
    trial_opacity.setVal(0);
    trial_ratio.setVal(variable_opacity ? 0 : chi_a / chi_t);
    amrex::MultiFab::Copy(old, old_radiation, 0, 0, 1, 0);
    amrex::MultiFab::Copy(velocity, work_velocity, 0, 0, 3, 0);
    amrex::MultiFab::Copy(temperature, old_nodal_temperature, 0, 0, 1,
                          old_nodal_temperature.nGrowVect());
    FillClosedTemperatureGhosts(temperature, geometry);
    old.FillBoundary(geometry.periodicity());
    velocity.FillBoundary(geometry.periodicity());
    // Match the other native diffusion APIs: laboratory face flux is W/m^2,
    // while radiation and work are J/cell. Apply physical face areas explicitly.
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> enthalpy;
    amrex::Real const ratio = variable_opacity ? 0 : chi_a / chi_t;
    amrex::ReduceOps<amrex::ReduceOpMax> checks;
    amrex::ReduceData<int> check_data(checks);
    using CheckTuple = typename decltype(check_data)::Type;
    for (amrex::MFIter mfi(old); mfi.isValid(); ++mfi) {
        auto const v = velocity.const_array(mfi);
        checks.eval(
            mfi.validbox(), check_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> CheckTuple {
                amrex::Real angular_velocity = v(i, j, k, 1);
#if defined(WARPX_DIM_RZ)
                // The secant is conjugate to Lz/rbar. Bound the reconstructed
                // tangential speed at the outer radius of the annulus too.
                auto const radius = metric.radial_low +
                    (i - metric.low[0] + 0.5_rt) * spacing[0];
                auto const mean_radius = radius + spacing[0] * spacing[0] / (12 * radius);
                angular_velocity *= (radius + 0.5_rt * spacing[0]) / mean_radius;
#endif
                amrex::Real const beta =
                    std::sqrt(v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1) +
                              v(i, j, k, 2) * v(i, j, k, 2)) /
                    PhysConst::c;
                amrex::Real const angular_beta = std::sqrt(
                    v(i, j, k, 0) * v(i, j, k, 0) + angular_velocity * angular_velocity +
                    v(i, j, k, 2) * v(i, j, k, 2)) / PhysConst::c;
                auto const bound = amrex::max(beta, angular_beta);
                return {!(bound <= 0.01_rt &&
                          (variable_opacity || chi_a == 0 || bound * chi_t * length <= 0.01_rt))};
            });
    }
    invalid = amrex::get<0>(check_data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    if (invalid) {
        return fail(Failure::InvalidState);
    }
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        enthalpy[d] = std::make_unique<amrex::MultiFab>(
            amrex::convert(boxes, amrex::IntVect::TheDimensionVector(d)), distribution, 1, 0);
        int const physical = AMREX_SPACEDIM == 1 ? 2 :
            (AMREX_SPACEDIM == 2 && d == 1 ? 2 : d);
        for (amrex::MFIter mfi(*enthalpy[d]); mfi.isValid(); ++mfi) {
            auto const e = old.const_array(mfi);
            auto const v = velocity.const_array(mfi);
            auto const h = enthalpy[d]->array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (metric.Wall(d, i, j, k)) {
                    h(i, j, k) = 0;
                    return;
                }
                int left[3] = {i,j,k};
                --left[d];
                auto const speed = 0.5_rt * (v(left[0],left[1],left[2],physical) +
                                            v(i,j,k,physical));
                h(i,j,k) = 4.0_rt/3.0_rt * speed *
                    (speed >= 0 ? e(left[0],left[1],left[2]) / metric.Volume(left[0]) :
                                  e(i,j,k) / metric.Volume(i));
            });
        }
    }
    amrex::ReduceOps<amrex::ReduceOpMax> cfl_check;
    amrex::ReduceData<int> cfl_data(cfl_check);
    for (amrex::MFIter mfi(shifted); mfi.isValid(); ++mfi) {
        auto const e = old.const_array(mfi);
        auto const v = velocity.const_array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> h{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { h[d] = enthalpy[d]->const_array(mfi); }
        auto const w = requested_work.const_array(mfi);
        auto const rhs = shifted.array(mfi);
        auto const base = advected.array(mfi);
        cfl_check.eval(
            mfi.validbox(), cfl_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> CheckTuple {
                amrex::Real cfl = 0, divergence = 0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    int left[3] = {i,j,k}, right[3] = {i,j,k};
                    --left[d];
                    ++right[d];
                    int const physical = AMREX_SPACEDIM == 1 ? 2 :
                        (AMREX_SPACEDIM == 2 && d == 1 ? 2 : d);
                    auto const vl = metric.Wall(d, i, j, k) ? 0 :
                        0.5_rt*(v(left[0],left[1],left[2],physical)+v(i,j,k,physical));
                    auto const vr = metric.Wall(d, right[0], right[1], right[2]) ? 0 : 0.5_rt *
                        (v(right[0],right[1],right[2],physical)+v(i,j,k,physical));
                    auto const al = metric.Area(d, i), ar = metric.Area(d, right[0]);
                    cfl += dt/metric.Volume(i) * 4.0_rt/3.0_rt *
                        (al*amrex::max(-vl,0.0_rt)+ar*amrex::max(vr,0.0_rt));
                    divergence += ar*h[d](right[0],right[1],right[2])-al*h[d](i,j,k);
                }
                base(i, j, k) = e(i, j, k) - dt * divergence;
                rhs(i, j, k) = variable_opacity
                                   ? base(i, j, k)
                                   : e(i, j, k) - dt * divergence - (1 - 2 * ratio) * w(i, j, k);
                return {cfl > 1 ? 1 : 0};
            });
    }
    invalid = amrex::get<0>(cfl_data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    if (invalid) {
        return fail(Failure::EnthalpyCfl);
    }
    bool const finite_shifted = shifted.is_finite(0, 1, 0);
    auto const min_shifted = shifted.min(0);
    if (!finite_shifted || min_shifted < 0) {
        return fail(Failure::AdvectedState);
    }
    shifted.FillBoundary(geometry.periodicity());
    advected.FillBoundary(geometry.periodicity());
    amrex::MultiFab::Copy(trial_shifted, shifted, 0, 0, 1, 1);
    heat.setVal(0);

    CoupledImplicitCallbacks wrapped = callbacks;
    wrapped.radiation_rhs = {};
    wrapped.rebase_material_source = {};
    wrapped.coefficients = [&] (amrex::MultiFab const& te, amrex::MultiFab& opacity,
                                amrex::MultiFab& absorption, amrex::MultiFab& emission,
                                amrex::Real trial_time) {
        callbacks.coefficients(te, opacity, absorption, emission, trial_time);
        // Poison invalid trials for the coupled solver's collective nonfatal
        // preflight.
        amrex::Real const tolerance = 32 * std::numeric_limits<amrex::Real>::epsilon();
        amrex::Real const bad = std::numeric_limits<amrex::Real>::quiet_NaN();
        for (amrex::MFIter mfi(opacity); mfi.isValid(); ++mfi) {
            auto const transport = opacity.array(mfi);
            auto const rate = absorption.const_array(mfi);
            auto const emit = emission.const_array(mfi);
            auto const rhs = advected.const_array(mfi);
            auto const constant_rhs = trial_shifted.const_array(mfi);
            auto const work = requested_work.const_array(mfi);
            auto const trial_chi = trial_opacity.array(mfi);
            auto const trial_r = trial_ratio.array(mfi);
            auto const minimum_tau = amrex::max(8.0_rt, diffusion.minimum_optical_depth);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                auto const chi = transport(i, j, k);
                auto const absorption_rate = rate(i, j, k);
                auto const r = variable_opacity ? absorption_rate / (PhysConst::c * chi) : ratio;
                auto const shifted_density =
                    variable_opacity
                        ? (rhs(i, j, k) - (1 - 2 * r) * work(i, j, k)) / metric.Volume(i)
                        : constant_rhs(i, j, k) / metric.Volume(i);
                if (!(chi > 0 && amrex::Math::isfinite(chi) && chi * dx >= minimum_tau &&
                      absorption_rate >= 0 && amrex::Math::isfinite(absorption_rate) &&
                      absorption_rate <= PhysConst::c * chi &&
                      (variable_opacity || (std::abs(chi - chi_t) <= tolerance * chi_t &&
                                            std::abs(absorption_rate - PhysConst::c * chi_a) <=
                                                tolerance * PhysConst::c * chi_a)) &&
                      emit(i, j, k) >= 0 && amrex::Math::isfinite(emit(i, j, k)) &&
                      (absorption_rate > 0 || emit(i, j, k) == 0) &&
                      amrex::Math::isfinite(1 + dt * absorption_rate) &&
                      amrex::Math::isfinite(shifted_density + dt * emit(i, j, k)))) {
                    transport(i, j, k) = bad;
                }
                trial_chi(i, j, k) = variable_opacity ? transport(i, j, k) : chi_t;
                trial_r(i, j, k) = r;
            });
        }
        trial_opacity.FillBoundary(geometry.periodicity());
        trial_ratio.FillBoundary(geometry.periodicity());
    };
    if (variable_opacity) {
        wrapped.radiation_rhs = [&] (amrex::MultiFab const&, amrex::MultiFab& rhs) {
            for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi) {
                auto const base = advected.const_array(mfi);
                auto const w = requested_work.const_array(mfi);
                auto const r = trial_ratio.const_array(mfi);
                auto const q = rhs.array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    q(i, j, k) = base(i, j, k) - (1 - 2 * r(i, j, k)) * w(i, j, k);
                });
            }
            rhs.FillBoundary(geometry.periodicity());
            amrex::MultiFab::Copy(trial_shifted, rhs, 0, 0, 1, 1);
        };
        wrapped.rebase_material_source = [] (amrex::MultiFab const& before,
                                             amrex::MultiFab const& after,
                                             amrex::MultiFab& source) {
            // RHS_new-RHS_old = 2*(r_new-r_old)*Work. Apply the same
            // offset change to Qstat, preserving the native physical heat.
            for (amrex::MFIter mfi(source); mfi.isValid(); ++mfi) {
                auto const old_rhs = before.const_array(mfi);
                auto const new_rhs = after.const_array(mfi);
                auto const q = source.array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    q(i, j, k) += new_rhs(i, j, k) - old_rhs(i, j, k);
                });
            }
        };
    }
    // Check physical endpoints, not the artificial shifted radiation RHS or
    // intermediate nonlinear temperatures. This excludes an initially far
    // from LTE moving state even if a stiff step would end close to LTE.
    auto physical_domain_valid = [&] (amrex::MultiFab const& energy, amrex::MultiFab const& te,
                                      amrex::Real endpoint_time) {
        amrex::MultiFab opacity(boxes, distribution, 1, 1);
        amrex::MultiFab absorption(boxes, distribution, 1, 0);
        amrex::MultiFab emission(boxes, distribution, 1, 0);
        wrapped.coefficients(te, opacity, absorption, emission, endpoint_time);
        amrex::ReduceOps<amrex::ReduceOpMax> ops;
        amrex::ReduceData<int> data(ops);
        for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi) {
            auto const e = energy.const_array(mfi);
            auto const v = velocity.const_array(mfi);
            auto const transport = opacity.const_array(mfi);
            auto const rate = absorption.const_array(mfi);
            auto const emit = emission.const_array(mfi);
            ops.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> CheckTuple {
                if (!amrex::Math::isfinite(transport(i, j, k))) {
                    return {1};
                }
                if (rate(i, j, k) == 0) {
                    return {0};
                }
                // Include out-of-plane motion even in reduced-dimensional geometry.
                if (v(i, j, k, 0) == 0 && v(i, j, k, 1) == 0 && v(i, j, k, 2) == 0) {
                    return {0};
                }
                amrex::Real const equilibrium = emit(i, j, k) / rate(i, j, k) * metric.Volume(i);
                auto const chi = variable_opacity ? transport(i, j, k) : chi_t;
                auto const r = variable_opacity ? rate(i, j, k) / (PhysConst::c * chi) : ratio;
                auto angular_velocity = v(i, j, k, 1);
#if defined(WARPX_DIM_RZ)
                auto const radius = metric.radial_low + (i - metric.low[0] + 0.5_rt) * spacing[0];
                auto const mean_radius = radius + spacing[0] * spacing[0] / (12 * radius);
                angular_velocity *= (radius + 0.5_rt * spacing[0]) / mean_radius;
#endif
                auto const beta =
                    std::sqrt(v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1) +
                              v(i, j, k, 2) * v(i, j, k, 2)) /
                    PhysConst::c;
                auto const angular_beta =
                    std::sqrt(v(i, j, k, 0) * v(i, j, k, 0) + angular_velocity * angular_velocity +
                              v(i, j, k, 2) * v(i, j, k, 2)) /
                    PhysConst::c;
                amrex::Real const inverse_tau = 1 / (chi * length);
                amrex::Real const equilibrium_bound = inverse_tau * inverse_tau;
                amrex::Real const scale = amrex::max(e(i, j, k), equilibrium);
                if (!(equilibrium >= 0) || !amrex::Math::isfinite(equilibrium)) {
                    return {1};
                }
                amrex::Real const mismatch =
                    scale > 0 ? r * (std::abs(e(i, j, k) - equilibrium) / scale) : 0;
                return {!(amrex::max(beta, angular_beta) * chi * length <= 0.01_rt &&
                          mismatch <= equilibrium_bound)};
            });
        }
        int rejected = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceIntMax(rejected);
        return rejected == 0;
    };
    if (!physical_domain_valid(old, temperature, time)) {
        return fail(Failure::InvalidState);
    }
    // A prescribed opacity update may leave the supported motion regime
    // independently of temperature convergence. Reject that unchanged bound
    // before a stiff solve can fail for a less informative numerical reason.
    // Malformed coefficients retain the coupled solver's existing rejection.
    amrex::MultiFab endpoint_transport(boxes, distribution, 1, 1);
    amrex::MultiFab endpoint_absorption(boxes, distribution, 1, 0);
    amrex::MultiFab endpoint_emission(boxes, distribution, 1, 0);
    wrapped.coefficients(temperature, endpoint_transport, endpoint_absorption,
                         endpoint_emission, time + dt);
    amrex::ReduceOps<amrex::ReduceOpMax> endpoint_ops;
    amrex::ReduceData<int> endpoint_data(endpoint_ops);
    for (amrex::MFIter it(endpoint_transport); it.isValid(); ++it) {
        auto const chi = endpoint_transport.const_array(it);
        auto const rate = endpoint_absorption.const_array(it);
        auto const v = velocity.const_array(it);
        endpoint_ops.eval(
            it.validbox(), endpoint_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> CheckTuple {
                // A malformed endpoint evaluation must reject now: a later
                // nonlinear coefficient callback can otherwise overwrite it.
                if (!amrex::Math::isfinite(chi(i, j, k)) ||
                    !amrex::Math::isfinite(rate(i, j, k))) {
                    return {2};
                }
                if (rate(i, j, k) < 0) {
                    return {2};
                }
                if (rate(i, j, k) == 0) {
                    return {0};
                }
                auto angular_velocity = v(i, j, k, 1);
#if defined(WARPX_DIM_RZ)
                auto const radius = metric.radial_low +
                                    (i - metric.low[0] + 0.5_rt) * spacing[0];
                auto const mean_radius =
                    radius + spacing[0] * spacing[0] / (12 * radius);
                angular_velocity *=
                    (radius + 0.5_rt * spacing[0]) / mean_radius;
#endif
                auto const transverse =
                    amrex::max(v(i, j, k, 1) * v(i, j, k, 1),
                               angular_velocity * angular_velocity);
                auto const beta =
                    std::sqrt(v(i, j, k, 0) * v(i, j, k, 0) + transverse +
                              v(i, j, k, 2) * v(i, j, k, 2)) /
                    PhysConst::c;
                return {beta * chi(i, j, k) * length > 0.01_rt};
            });
    }
    int endpoint_invalid = amrex::get<0>(endpoint_data.value());
    amrex::ParallelDescriptor::ReduceIntMax(endpoint_invalid);
    // Malformed material data retain the coupled-solver failure taxonomy;
    // only a finite endpoint beta*chi*L violation is a motion-domain error.
    if (endpoint_invalid == 2) {
        auto result = fail(Failure::CoupledFailure);
        result.coupled.failure = CoupledImplicitFailure::Material;
        return result;
    }
    if (endpoint_invalid) {
        return fail(Failure::InvalidState);
    }
    auto offset = [&] (amrex::MultiFab& source, amrex::Real sign) {
        for (amrex::MFIter mfi(source); mfi.isValid(); ++mfi) {
            auto const q = source.array(mfi);
            auto const w = requested_work.const_array(mfi);
            auto const r = trial_ratio.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                q(i, j, k) += sign * 2 * r(i, j, k) * w(i, j, k);
            });
        }
        source.FillBoundary(geometry.periodicity());
    };
    wrapped.material_response = [&] (amrex::MultiFab& te, amrex::MultiFab& source) {
        offset(source, -1);
        if (!geometry.isAllPeriodic()) { FillClosedTemperatureGhosts(te, geometry); }
        auto const residual = callbacks.material_response(te, source);
        if (!geometry.isAllPeriodic()) { FillClosedTemperatureGhosts(te, geometry); }
        offset(source, 1);
        return residual;
    };
    wrapped.conservative_material_request = [&] (amrex::MultiFab const& energy,
                                                 amrex::MultiFab& source) {
        if (!variable_opacity && chi_a == 0) {
            // With no absorptive exchange, zero work correction, and the
            // already-enforced zero emissivity, the exact constitutive request
            // is zero. Do not reconstruct this no-op by subtracting radiation
            // energy states.
            source.setVal(0);
            return;
        }
        // Eliminate Qstat using the current trial RHS and the same
        // face-averaged Fick flux as the spatial solve.
        // This only generates a thermal trial; it never corrects energy or
        // replaces the independent final constitutive-source checks.
        source.setVal(0);
        for (amrex::MFIter mfi(source); mfi.isValid(); ++mfi) {
            auto const before = trial_shifted.const_array(mfi);
            auto const after = energy.const_array(mfi);
            auto const chi = trial_opacity.const_array(mfi);
            auto const r = trial_ratio.const_array(mfi);
            auto const q = source.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // An exactly nonabsorbing cell has no constitutive heat even
                // if neighbors absorb. Avoid subtracting its radiation states.
                if (r(i, j, k) == 0) {
                    q(i, j, k) = 0;
                    return;
                }
                amrex::Real laplacian = 0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    int left[3] = {i, j, k}, right[3] = {i, j, k};
                    --left[d];
                    ++right[d];
                    auto const center = after(i, j, k) / metric.Volume(i);
                    auto const lo =
                        metric.Wall(d, i, j, k)
                            ? 0
                            : metric.Area(d, i) * (center - after(left[0], left[1], left[2]) /
                                                                metric.Volume(left[0]));
                    auto const hi =
                        metric.Wall(d, right[0], right[1], right[2])
                            ? 0
                            : metric.Area(d, right[0]) *
                                  (after(right[0], right[1], right[2]) / metric.Volume(right[0]) -
                                   center);
                    if (variable_opacity) {
                        auto const low_flux =
                            metric.Wall(d, i, j, k)
                                ? 0
                                : lo / (0.5_rt * (chi(i, j, k) + chi(left[0], left[1], left[2])));
                        auto const high_flux =
                            metric.Wall(d, right[0], right[1], right[2])
                                ? 0
                                : hi /
                                      (0.5_rt * (chi(i, j, k) + chi(right[0], right[1], right[2])));
                        laplacian += dt * PhysConst::c / (3 * spacing[d]) * (high_flux - low_flux);
                    } else {
                        laplacian += dt * PhysConst::c / (3 * chi_t * spacing[d]) * (hi - lo);
                    }
                }
                q(i, j, k) = before(i, j, k) - after(i, j, k) + laplacian;
            });
        }
    };
    auto controls = coupled;
    controls.max_subdivisions = 0;
    controls.reconcile_material_source = false;
    controls.safeguard_material_iteration = true;
    controls.allow_ideal_temperature_roundoff = true;
    controls.require_local_ideal_accuracy = options.require_local_ideal_accuracy;
    controls.diffusion.use_asymptotic_diffusion = true;
    controls.diffusion.use_full_gradient = false;
    // Closed stage: periodic faces or stationary reflecting walls, never baths.
    controls.diffusion.boundary_lo = {0, 0, 0};
    controls.diffusion.boundary_hi = {0, 0, 0};
    auto const solve = TryAdvanceCoupledImplicitDiffusion(shifted, temperature, heat, geometry,
                                                          time, dt, controls, wrapped);
    if (solve.failure != CoupledImplicitFailure::None) {
        auto result = fail(Failure::CoupledFailure);
        result.coupled = solve;
        return result;
    }
    if (!physical_domain_valid(shifted, temperature, time + dt)) {
        return fail(Failure::InvalidState);
    }
    if (!geometry.isAllPeriodic()) { FillClosedTemperatureGhosts(temperature, geometry); }
    amrex::MultiFab final_opacity(boxes, distribution, 1, 1);
    amrex::MultiFab final_absorption(boxes, distribution, 1, 0);
    amrex::MultiFab final_emission(boxes, distribution, 1, 0);
    wrapped.coefficients(temperature, final_opacity, final_absorption, final_emission, time + dt);
    if (!final_opacity.is_finite(0, 1, 0)) {
        return fail(Failure::InvalidState);
    }
    offset(heat, -1);
    MovingThermalDiffusionFieldResult result;
    result.coupled = solve;
    result.radiation = std::move(shifted);
    result.nodal_temperature = std::move(temperature);
    result.material_energy = std::move(heat);
    result.requested_material_energy.define(boxes, distribution, 1, 1);
    result.thermal_deposition_redistribution.define(boxes, distribution, 1, 1);
    result.material_impulse.define(boxes, distribution, 3, 1);
    result.mechanical_debit.define(boxes, distribution, 1, 1);
    result.mechanical_operand_scale.define(boxes, distribution, 1, 1);
    result.material_impulse.setVal(0);
    result.requested_material_energy.setVal(0);
    result.thermal_deposition_redistribution.setVal(0);
    result.mechanical_debit.setVal(0);
    result.mechanical_operand_scale.setVal(0);
    result.radiation.FillBoundary(geometry.periodicity());
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        result.face_lab_energy_current[d] =
            std::make_unique<amrex::MultiFab>(enthalpy[d]->boxArray(), distribution, 1, 0);
        for (amrex::MFIter mfi(*result.face_lab_energy_current[d]); mfi.isValid(); ++mfi) {
            auto const e = result.radiation.const_array(mfi);
            auto const chi = trial_opacity.const_array(mfi);
            auto const h = enthalpy[d]->const_array(mfi);
            auto const f = result.face_lab_energy_current[d]->array(mfi);
            auto const axis_dx = spacing[d];
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (metric.Wall(d, i, j, k)) {
                    f(i, j, k) = 0;
                    return;
                }
                int left[3] = {i,j,k};
                --left[d];
                auto const face_chi = 0.5_rt * (chi(i, j, k) + chi(left[0], left[1], left[2]));
                f(i, j, k) =
                    h(i, j, k) - PhysConst::c / (3 * face_chi * axis_dx) *
                                     (e(i, j, k) / metric.Volume(i) -
                                      e(left[0], left[1], left[2]) / metric.Volume(left[0]));
            });
        }
    }
    for (amrex::MFIter mfi(result.radiation); mfi.isValid(); ++mfi) {
        auto const before = old.const_array(mfi);
        auto const after = result.radiation.const_array(mfi);
        auto const actual_heat = result.material_energy.const_array(mfi);
        auto const requested_heat = result.requested_material_energy.array(mfi);
        auto const redistribution = result.thermal_deposition_redistribution.array(mfi);
        auto const rate = final_absorption.const_array(mfi);
        auto const emission = final_emission.const_array(mfi);
        auto const w = requested_work.const_array(mfi);
        auto const r = trial_ratio.const_array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> f{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            f[d] = result.face_lab_energy_current[d]->const_array(mfi);
        }
        auto const p = result.material_impulse.array(mfi);
        auto const debit = result.mechanical_debit.array(mfi);
        auto const scale = result.mechanical_operand_scale.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto const q =
                dt * (rate(i, j, k) * after(i, j, k) - metric.Volume(i) * emission(i, j, k)) -
                2 * r(i, j, k) * w(i, j, k);
            requested_heat(i, j, k) = q;
            redistribution(i, j, k) = actual_heat(i, j, k) - q;
            amrex::Real transport = 0, transport_scale = 0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                int left[3] = {i, j, k}, right[3] = {i, j, k};
                --left[d];
                ++right[d];
                int const physical =
                    AMREX_SPACEDIM == 1 ? 2 : (AMREX_SPACEDIM == 2 && d == 1 ? 2 : d);
                auto const pressure = after(i, j, k) / (3 * metric.Volume(i));
                auto const pl = metric.Wall(d, i, j, k)
                                    ? pressure
                                    : 0.5_rt * (pressure + after(left[0], left[1], left[2]) /
                                                               (3 * metric.Volume(left[0])));
                auto const pr = metric.Wall(d, right[0], right[1], right[2])
                                    ? pressure
                                    : 0.5_rt * (pressure + after(right[0], right[1], right[2]) /
                                                               (3 * metric.Volume(right[0])));
                auto const al = metric.Area(d, i), ar = metric.Area(d, right[0]);
#if defined(WARPX_DIM_RZ)
                // Subtract cylindrical isotropic stress geometrically. This
                // well-balanced form gives zero force for constant pressure.
                if (d == 0) {
                    p(i, j, k, physical) = -dt * (ar * (pr - pressure) - al * (pl - pressure));
                } else
#endif
                {
                    p(i, j, k, physical) = -dt * al * (pr - pl);
                }
                transport += dt * (ar * f[d](right[0], right[1], right[2]) - al * f[d](i, j, k));
                transport_scale += dt * (ar * std::abs(f[d](right[0], right[1], right[2])) +
                                         al * std::abs(f[d](i, j, k)));
            }
            debit(i, j, k) = before(i, j, k) - after(i, j, k) - transport - q;
            scale(i, j, k) = std::abs(before(i, j, k)) + std::abs(after(i, j, k)) +
                             transport_scale + std::abs(q);
        });
    }
    for (auto* field :
         {&result.material_impulse, &result.mechanical_debit, &result.mechanical_operand_scale,
          &result.requested_material_energy, &result.thermal_deposition_redistribution}) {
        bool const finite = field->is_finite(0, field->nComp(), 0);
        if (!finite) {
            return fail(Failure::NonfiniteDiagnostic);
        }
        field->FillBoundary(geometry.periodicity());
    }
    for (auto const& flux : result.face_lab_energy_current) {
        if (!flux->is_finite(0,1,0)) { return fail(Failure::NonfiniteDiagnostic); }
    }
    // Independently reconstruct the full physical equation. In particular,
    // a temperature-dependent work split must not survive acceptance with a
    // lagged r in either the heat or mechanical-debit representation.
    amrex::ReduceOps<amrex::ReduceOpMax> physical_ops;
    amrex::ReduceData<int> physical_data(physical_ops);
    auto const equation_tolerance = diffusion.tolerance;
    for (amrex::MFIter mfi(result.radiation); mfi.isValid(); ++mfi) {
        auto const debit = result.mechanical_debit.const_array(mfi);
        auto const scale = result.mechanical_operand_scale.const_array(mfi);
        auto const work = requested_work.const_array(mfi);
        physical_ops.eval(
            mfi.validbox(), physical_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> CheckTuple {
                auto const magnitude = scale(i, j, k) + std::abs(work(i, j, k));
                auto const allowance =
                    (equation_tolerance + 64 * std::numeric_limits<amrex::Real>::epsilon()) *
                    magnitude;
                return {std::abs(debit(i, j, k) - work(i, j, k)) > allowance ? 1 : 0};
            });
    }
    int physical_invalid = amrex::get<0>(physical_data.value());
    amrex::ParallelDescriptor::ReduceIntMax(physical_invalid);
    auto const physical_balance = old.sum(0) - result.radiation.sum(0) -
                                  result.material_energy.sum(0) - requested_work.sum(0);
    auto const physical_scale = old.norm1(0) + result.radiation.norm1(0) +
                                result.material_energy.norm1(0) + requested_work.norm1(0);
    if (physical_invalid || !std::isfinite(physical_balance) || !std::isfinite(physical_scale) ||
        std::abs(physical_balance) > coupled.energy_tolerance * physical_scale) {
        return fail(Failure::PhysicalResidual);
    }
    // Integrate from boundary-adjacent cells, counted once even when face
    // MultiFabs have aliases on box edges. A stationary wall performs no work.
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> wall_ops;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real> wall_data(wall_ops);
    for (amrex::MFIter mfi(result.radiation); mfi.isValid(); ++mfi) {
        auto const e = result.radiation.const_array(mfi);
        wall_ops.eval(mfi.validbox(), wall_data, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real> {
            amrex::Real impulse[3] = {0,0,0};
            auto const pressure = e(i,j,k)/(3*metric.Volume(i));
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                int high[3] = {i,j,k};
                ++high[d];
                int const physical = AMREX_SPACEDIM == 1 ? 2 :
                    (AMREX_SPACEDIM == 2 && d == 1 ? 2 : d);
                if (metric.Wall(d,i,j,k)) { impulse[physical] -= dt*metric.Area(d,i)*pressure; }
                if (metric.Wall(d,high[0],high[1],high[2])) {
                    impulse[physical] += dt*metric.Area(d,high[0])*pressure;
                }
            }
            return {impulse[0],impulse[1],impulse[2]};
        });
    }
    auto const wall = wall_data.value();
    result.radiation_wall_momentum = {
        amrex::get<0>(wall), amrex::get<1>(wall), amrex::get<2>(wall)};
    amrex::ParallelDescriptor::ReduceRealSum(result.radiation_wall_momentum.data(), 3);
    for (auto const impulse : result.radiation_wall_momentum) {
        if (!std::isfinite(impulse)) { return fail(Failure::NonfiniteDiagnostic); }
    }
    result.valid = true;
    return result;
}
} // namespace warpx::radiation
