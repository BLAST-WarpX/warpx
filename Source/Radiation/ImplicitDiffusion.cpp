/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "ImplicitDiffusion.H"

#include "CellVolume.H"
#include "DiffusionGradient.H"
#include "RadiationTransport.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Geometry.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

using namespace amrex::literals;

namespace
{
    using Boundary = RadiationTransport::DiffusionBoundary;

    struct Metrics
    {
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx;
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lower;
        amrex::Dim3 domain_lo;
        amrex::Dim3 domain_hi;

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
        volume (int i) const noexcept
        {
            return warpx::radiation::CellVolume(i, dx, lower, domain_lo);
        }

        // i is the face index when direction==0, otherwise a cell index.
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
        area (int i, int direction) const noexcept
        {
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RSPHERE)
            amrex::Real const r = lower[0] + (i - domain_lo.x) * dx[0];
#if defined(WARPX_DIM_RSPHERE)
            amrex::ignore_unused(direction);
            return 4.0_rt * MathConst::pi * r * r;
#elif defined(WARPX_DIM_RZ)
            return direction == 0 ? 2.0_rt * MathConst::pi * r * dx[1] : volume(i) / dx[1];
#else
            amrex::ignore_unused(direction);
            return 2.0_rt * MathConst::pi * r;
#endif
#else
            return volume(i) / dx[direction];
#endif
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
        boundary (int i, int j, int k, int direction, int side) const noexcept
        {
            amrex::ignore_unused(j, k);
            amrex::GpuArray<int, AMREX_SPACEDIM> const index{AMREX_D_DECL(i, j, k)};
            amrex::GpuArray<int, AMREX_SPACEDIM> const lo{
                AMREX_D_DECL(domain_lo.x, domain_lo.y, domain_lo.z)};
            amrex::GpuArray<int, AMREX_SPACEDIM> const hi{
                AMREX_D_DECL(domain_hi.x, domain_hi.y, domain_hi.z)};
            return index[direction] == (side < 0 ? lo[direction] : hi[direction]);
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, 3>
        position (int i, int j, int k, int direction, int side) const noexcept
        {
            amrex::ignore_unused(j, k);
            amrex::GpuArray<int, AMREX_SPACEDIM> const index{AMREX_D_DECL(i, j, k)};
            amrex::GpuArray<int, AMREX_SPACEDIM> const lo{
                AMREX_D_DECL(domain_lo.x, domain_lo.y, domain_lo.z)};
            amrex::GpuArray<amrex::Real, 3> result{0, 0, 0};
            for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
#if defined(WARPX_DIM_1D_Z)
                int const component = 2;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                int const component = axis == 1 ? 2 : 0;
#else
                int const component = axis;
#endif
                result[component] =
                    lower[axis] + dx[axis] * (index[axis] - lo[axis] + 0.5_rt +
                                              (axis == direction ? 0.5_rt * side : 0.0_rt));
            }
            return result;
        }
    };

    struct Face
    {
        amrex::Real conductance = 0;
        amrex::Real bath = 0;
    };

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Face
    boundaryFace (int direction, int side, int group, amrex::Real opacity, amrex::Real time,
                  amrex::GpuArray<amrex::Real, 3> const& position, Metrics const& metric,
                  warpx::radiation::ImplicitDiffusionOptions const& options)
    {
        int const code = side < 0 ? options.boundary_lo[direction] : options.boundary_hi[direction];
        Face face;
        if (code == static_cast<int>(Boundary::Vacuum)) {
            face.conductance = PhysConst::c;
        } else if (code == static_cast<int>(Boundary::Marshak)) {
            face.conductance = 0.5_rt * PhysConst::c;
        } else if (code == static_cast<int>(Boundary::MarshakBath)) {
            face.conductance = PhysConst::c / (2.0_rt + 1.5_rt * opacity * metric.dx[direction]);
            int const index = 2 * direction + (side > 0 ? 1 : 0);
            face.bath = options.bath_executors[index * options.energy_groups.m_num_groups + group](
                position[0], position[1], position[2], time);
            if (!(face.bath >= 0) || !amrex::Math::isfinite(face.bath)) {
                face.bath = -1;
                return face;
            }
            if (options.bath_is_temperature[index] != 0) {
                amrex::Real const t2 = face.bath * face.bath;
                face.bath = 7.565733250280007e-16_rt * t2 * t2 *
                            options.energy_groups.planckFraction(group, PhysConst::kb * face.bath);
            }
        }
        return face;
    }

    struct CellResidual
    {
        amrex::Real equation;
        amrex::Real boundary_out;
        amrex::Real boundary_in;
    };

    /** One authoritative cell-integrated equation for both correction RHS and
     * acceptance/ledgers. Form face differences before multiplying by stiff
     * conductances; do not subtract two large assembled matrix products. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CellResidual
    cellResidual (
        int i, int j, int k, int group, amrex::Array4<amrex::Real const> const& energy,
        amrex::Array4<amrex::Real const> const& initial_energy,
        amrex::Array4<amrex::Real const> const& alpha,
        amrex::Array4<amrex::Real const> const& absorption,
        amrex::Array4<amrex::Real const> const& emission,
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> const& coefficients,
        amrex::GpuArray<int, AMREX_SPACEDIM> const& periodic, Metrics const& metric,
        amrex::Real volume_scale, amrex::Real time, amrex::Real dt, bool has_source,
        warpx::radiation::ImplicitDiffusionOptions const& options) noexcept
    {
        amrex::Real out = 0, in = 0, transfer = 0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            for (int side = -1; side <= 1; side += 2)
            {
                amrex::Real outward;
                int const ni = i + (d == 0 ? side : 0);
                int const nj = j + (d == 1 ? side : 0);
                int const nk = k + (d == 2 ? side : 0);
                if (periodic[d] == 0 && metric.boundary(i, j, k, d, side))
                {
                    Face const face =
                        boundaryFace(d, side, group, alpha(i, j, k, group), time + 0.5_rt * dt,
                                     metric.position(i, j, k, d, side), metric, options);
                    outward = dt * metric.area(i + (d == 0 && side > 0 ? 1 : 0), d) *
                              face.conductance * (energy(i, j, k) - face.bath);
                    out += amrex::max(outward, 0.0_rt);
                    in += amrex::max(-outward, 0.0_rt);
                }
                else
                {
                    int const fi = i + (d == 0 && side > 0 ? 1 : 0);
                    int const fj = j + (d == 1 && side > 0 ? 1 : 0);
                    int const fk = k + (d == 2 && side > 0 ? 1 : 0);
                    outward = dt * volume_scale * coefficients[d](fi, fj, fk) *
                              (energy(i, j, k) - energy(ni, nj, nk)) /
                              (metric.dx[d] * metric.dx[d]);
                }
                transfer += outward;
            }
        }
        amrex::Real const stored_energy = energy(i, j, k) * metric.volume(i);
        amrex::Real const material_transfer =
            has_source ? dt * (absorption(i, j, k, group) * stored_energy -
                               metric.volume(i) * emission(i, j, k, group))
                       : 0.0_rt;
        return {stored_energy - initial_energy(i, j, k, group) + transfer + material_transfer, out,
                in};
    }
} // namespace

namespace warpx::radiation
{
    namespace
    {
        ImplicitDiffusionResult
        SolveImplicitDiffusion (amrex::MultiFab& radiation, amrex::MultiFab const& opacity,
                                amrex::Geometry const& geometry, amrex::Real time, amrex::Real dt,
                                ImplicitDiffusionOptions const& options,
                                amrex::MultiFab const* residual_candidate)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                dt >= 0 && amrex::Math::isfinite(dt) && amrex::Math::isfinite(time) &&
                    options.tolerance > 0 && amrex::Math::isfinite(options.tolerance) &&
                    options.linear_tolerance > 0 && options.linear_tolerance < options.tolerance &&
                    options.minimum_optical_depth > 0 &&
                    amrex::Math::isfinite(options.minimum_optical_depth) &&
                    options.max_iterations > 0 && options.max_linear_iterations > 0 &&
                    options.nonlinear_relaxation > 0 && options.nonlinear_relaxation <= 1 &&
                    opacity.nComp() == radiation.nComp() &&
                    opacity.boxArray() == radiation.boxArray() &&
                    opacity.DistributionMap() == radiation.DistributionMap() &&
                    opacity.nGrowVect().min() >= 1 && radiation.ixType().cellCentered() &&
                    options.energy_groups.m_num_groups == radiation.nComp(),
                "Invalid implicit radiation diffusion controls or group layout.");
            Metrics const metric{geometry.CellSizeArray(), geometry.ProbLoArray(),
                                 amrex::lbound(geometry.Domain()), amrex::ubound(geometry.Domain())};
            auto const periodic = geometry.isPeriodicArray();
            bool const use_full_gradient = options.use_full_gradient;
            bool const has_source = options.absorption_rate != nullptr;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(has_source == (options.emissivity != nullptr),
                                             "Implicit radiation source requires both rate fields.");
            if (has_source) {
                for (auto const* field : {options.absorption_rate, options.emissivity}) {
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        field->boxArray() == radiation.boxArray() &&
                            field->DistributionMap() == radiation.DistributionMap() &&
                            field->nComp() == radiation.nComp(),
                        "Implicit radiation source fields must match the radiation layout.");
                }
            }
            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
                bool const bath =
                    options.boundary_lo[direction] == static_cast<int>(Boundary::MarshakBath) ||
                    options.boundary_hi[direction] == static_cast<int>(Boundary::MarshakBath);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!bath || periodic[direction] == 0,
                                                 "Implicit radiation bath faces must be nonperiodic.");
            }
            amrex::Real const volume_scale = metric.volume(metric.domain_hi.x);
            amrex::Real const min_dx =
                AMREX_D_PICK(metric.dx[0], amrex::min(metric.dx[0], metric.dx[1]),
                             amrex::min(metric.dx[0], amrex::min(metric.dx[1], metric.dx[2])));
            auto const& boxes = radiation.boxArray();
            auto const& distribution = radiation.DistributionMap();
            amrex::MultiFab accepted(boxes, distribution, radiation.nComp(), 0);
            amrex::MultiFab old(boxes, distribution, 1, 0);
            amrex::MultiFab iterate(boxes, distribution, 1, 1);
            amrex::MultiFab candidate(boxes, distribution, 1, 1);
            amrex::MultiFab a(boxes, distribution, 1, 0);
            amrex::MultiFab rhs(boxes, distribution, 1, 0);
            amrex::MultiFab correction_rhs;
            if (options.use_incremental_form && residual_candidate == nullptr)
            {
                correction_rhs.define(boxes, distribution, 1, 0);
            }
            amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> b;
            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
                b[direction] = std::make_unique<amrex::MultiFab>(
                    amrex::convert(boxes, amrex::IntVect::TheDimensionVector(direction)), distribution,
                    1, 0);
            }
            // Metrics are explicit in a and b. Do not let MLMG apply radial metrics
            // twice.
            amrex::RealBox const physical_box = geometry.ProbDomain();
            amrex::Geometry const cartesian(geometry.Domain(), &physical_box, 0, periodic.data());
            amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> boundary;
            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
                boundary[direction] = periodic[direction] != 0 ? amrex::LinOpBCType::Periodic
                                                               : amrex::LinOpBCType::Neumann;
            }
            ImplicitDiffusionResult result;
            result.group_escaped_energy.resize(radiation.nComp(), 0);
            result.group_injected_energy.resize(radiation.nComp(), 0);
            result.group_material_energy.resize(radiation.nComp(), 0);
            auto failed = [&result] (ImplicitDiffusionFailure reason) {
                // Earlier groups are only scratch candidates. Their fluxes must not
                // escape through a failed attempt's accepted-transfer ledger either.
                result.escaped_energy = 0;
                result.injected_energy = 0;
                result.numerical_energy_residual = 0;
                std::fill(result.group_escaped_energy.begin(), result.group_escaped_energy.end(), 0);
                std::fill(result.group_injected_energy.begin(), result.group_injected_energy.end(), 0);
                std::fill(result.group_material_energy.begin(), result.group_material_energy.end(), 0);
                result.failure = reason;
                return result;
            };
            for (int group = 0; group < radiation.nComp(); ++group) {
                amrex::ReduceOps<amrex::ReduceOpMax> check_ops;
                amrex::ReduceData<int> check_data(check_ops);
                using CheckTuple = typename decltype(check_data)::Type;
                for (amrex::MFIter mfi(old); mfi.isValid(); ++mfi) {
                    auto const source = radiation.const_array(mfi);
                    auto const alpha = opacity.const_array(mfi);
                    auto const u0 = old.array(mfi);
                    auto const ac = a.array(mfi);
                    auto const rc = rhs.array(mfi);
                    amrex::Array4<amrex::Real const> absorption, emission;
                    if (has_source) {
                        absorption = options.absorption_rate->const_array(mfi);
                        emission = options.emissivity->const_array(mfi);
                    }
                    check_ops.eval(
                        mfi.validbox(), check_data,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) -> CheckTuple {
                            amrex::Real const volume = metric.volume(i);
                            amrex::Real const opacity_value = alpha(i, j, k, group);
                            u0(i, j, k) = source(i, j, k, group) / volume;
                            ac(i, j, k) = volume / volume_scale;
                            rc(i, j, k) = source(i, j, k, group) / volume_scale;
                            if (has_source) {
                                auto const rate = absorption(i, j, k, group);
                                auto const emissivity = emission(i, j, k, group);
                                if (!(rate >= 0) || !amrex::Math::isfinite(rate) ||
                                    !(emissivity >= 0) || !amrex::Math::isfinite(emissivity)) {
                                    return {2};
                                }
                                ac(i, j, k) += dt * rate * volume / volume_scale;
                                rc(i, j, k) += dt * emissivity * volume / volume_scale;
                            }
                            if (!(u0(i, j, k) >= 0) || !amrex::Math::isfinite(u0(i, j, k)) ||
                                !(opacity_value * min_dx >= options.minimum_optical_depth) ||
                                !amrex::Math::isfinite(opacity_value)) {
                                return {1};
                            }
                            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                                for (int side = -1; side <= 1; side += 2) {
                                    if (periodic[d] != 0 || !metric.boundary(i, j, k, d, side)) {
                                        continue;
                                    }
                                    Face const face = boundaryFace(
                                        d, side, group, opacity_value, time + 0.5_rt * dt,
                                        metric.position(i, j, k, d, side), metric, options);
                                    if (!(face.bath >= 0) || !amrex::Math::isfinite(face.bath)) {
                                        return {2};
                                    }
                                    amrex::Real const area =
                                        metric.area(i + (d == 0 && side > 0 ? 1 : 0), d);
                                    amrex::Real const coefficient =
                                        dt * face.conductance * area / volume_scale;
                                    ac(i, j, k) += coefficient;
                                    rc(i, j, k) += coefficient * face.bath;
                                }
                            }
                            return {amrex::Math::isfinite(rc(i, j, k)) &&
                                            amrex::Math::isfinite(ac(i, j, k))
                                        ? 0
                                        : 2};
                        });
                }
                int invalid = amrex::get<0>(check_data.value());
                amrex::ParallelDescriptor::ReduceIntMax(invalid);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    invalid == 0, "Implicit radiation diffusion requires finite nonnegative "
                                  "radiation/bath "
                                  "states and all cells above minimum_diffusion_optical_depth.");
                // Only valid values enter the operator; initialize physical guard
                // storage as well before copying an initial guess into MLMG.
                iterate.setVal(0.0_rt);
                candidate.setVal(0.0_rt);
                amrex::MultiFab::Copy(iterate, old, 0, 0, 1, 0);
                if (residual_candidate != nullptr) {
                    for (amrex::MFIter mfi(iterate); mfi.isValid(); ++mfi) {
                        auto const trial = residual_candidate->const_array(mfi);
                        auto const target = iterate.array(mfi);
                        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                            target(i, j, k) = trial(i, j, k, group) / metric.volume(i);
                        });
                    }
                }
                amrex::Real const initial_norm = old.norm0(0);
                auto fill_coefficients = [&] (amrex::MultiFab& state) {
                    state.FillBoundary(geometry.periodicity());
                    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
                        for (amrex::MFIter mfi(*b[direction]); mfi.isValid(); ++mfi) {
                            auto const energy = state.const_array(mfi);
                            auto const alpha = opacity.const_array(mfi);
                            auto const coefficient = b[direction]->array(mfi);
                            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                                    int k) noexcept {
                                int const pi = i - (direction == 0 ? 1 : 0);
                                int const pj = j - (direction == 1 ? 1 : 0);
                                int const pk = k - (direction == 2 ? 1 : 0);
                                if (periodic[direction] == 0 &&
                                    (metric.boundary(i, j, k, direction, -1) ||
                                     metric.boundary(pi, pj, pk, direction, 1))) {
                                    coefficient(i, j, k) = 0;
                                    return;
                                }
                                amrex::Real const left = energy(pi, pj, pk);
                                amrex::Real const right = energy(i, j, k);
                                amrex::Real const face_opacity =
                                    0.5_rt * (alpha(pi, pj, pk, group) + alpha(i, j, k, group));
                                amrex::Real const jump =
                                    left + right > 0 ? 2.0_rt * std::abs(right - left) / (left + right)
                                                     : 0;
                                amrex::Real r = jump / (metric.dx[direction] * face_opacity);
                                if (use_full_gradient && left + right > 0) {
                                    DiffusionDensityView<false> const density{
                                        energy, 0, metric.dx, metric.lower, metric.domain_lo};
                                    amrex::Real const magnitude = FaceGradientMagnitude(
                                        density, pi, pj, pk, direction, 1,
                                        (right - left) / metric.dx[direction], metric.dx, periodic,
                                        metric.domain_lo, metric.domain_hi);
                                    r = magnitude / (face_opacity * (0.5_rt * left + 0.5_rt * right));
                                }
                                amrex::Real const limiter = r > 1 ? (1 + 2 / r) / (r + 3 + 6 / r)
                                                                  : (2 + r) / (6 + 3 * r + r * r);
                                coefficient(i, j, k) = metric.area(i, direction) *
                                                       metric.dx[direction] / volume_scale *
                                                       PhysConst::c * limiter / face_opacity;
                            });
                        }
                    }
                };
                bool converged = false;
                amrex::Real group_out = 0, group_in = 0, group_defect = 0;
                // A posteriori residual checks need face coefficients, not an
                // unused multigrid hierarchy. In particular, avoid its many GPU
                // allocations and initialization kernels for every outer trial.
                std::unique_ptr<amrex::MLABecLaplacian> op;
                std::unique_ptr<amrex::MLMG> solver;
                if (residual_candidate == nullptr) {
                    fill_coefficients(iterate);
                    op = std::make_unique<amrex::MLABecLaplacian>(
                        amrex::Vector<amrex::Geometry>{cartesian}, amrex::Vector<amrex::BoxArray>{boxes},
                        amrex::Vector<amrex::DistributionMapping>{distribution});
                    op->setDomainBC(boundary, boundary);
                    op->setLevelBC(0, nullptr);
                    op->setScalars(1.0_rt, dt);
                    op->setACoeffs(0, a);
                    op->setBCoeffs(0, amrex::GetArrOfConstPtrs(b));
                    solver = std::make_unique<amrex::MLMG>(*op);
                    solver->setVerbose(options.verbosity);
                    solver->setMaxIter(options.max_linear_iterations);
                    solver->setThrowException(true);
                }
                for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
                    if (iteration > 0 && op) {
                        fill_coefficients(iterate);
                        op->setBCoeffs(0, amrex::GetArrOfConstPtrs(b));
                    }
                    amrex::MultiFab::Copy(candidate, iterate, 0, 0, 1, 1);
                    try {
                        if (residual_candidate == nullptr) {
                            if (options.use_incremental_form)
                            {
                                for (amrex::MFIter mfi(correction_rhs); mfi.isValid(); ++mfi)
                                {
                                    auto const energy = iterate.const_array(mfi);
                                    auto const original = radiation.const_array(mfi);
                                    auto const alpha = opacity.const_array(mfi);
                                    auto const output = correction_rhs.array(mfi);
                                    amrex::Array4<amrex::Real const> absorption, emission;
                                    if (has_source)
                                    {
                                        absorption = options.absorption_rate->const_array(mfi);
                                        emission = options.emissivity->const_array(mfi);
                                    }
                                    amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM>
                                        coefficients;
                                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                    {
                                        coefficients[d] = b[d]->const_array(mfi);
                                    }
                                    amrex::ParallelFor(
                                        mfi.validbox(),
                                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                        {
                                            output(i, j, k) =
                                                -cellResidual(i, j, k, group, energy, original, alpha,
                                                              absorption, emission, coefficients,
                                                              periodic, metric, volume_scale, time, dt,
                                                              has_source, options)
                                                     .equation /
                                                volume_scale;
                                        });
                                }
                                auto const scale = amrex::max(initial_norm, iterate.norm0(0));
                                auto const absolute = options.linear_tolerance * scale *
                                                      metric.volume(metric.domain_lo.x) / volume_scale;
                                candidate.setVal(0);
                                solver->solve({&candidate}, {&correction_rhs}, options.linear_tolerance,
                                              absolute);
                                amrex::MultiFab::Add(candidate, iterate, 0, 0, 1, 0);
                            }
                            else
                            {
                                solver->solve({&candidate}, {&rhs}, options.linear_tolerance, 0.0_rt);
                            }
                            result.linear_iterations += solver->getNumIters();
                        }
                    } catch (amrex::MLMG::error const&) {
                        // MLMG's convergence failure is decided from collective
                        // residual norms; every rank takes the same return path.
                        result.linear_iterations += solver->getNumIters();
                        return failed(ImplicitDiffusionFailure::LinearConvergence);
                    }
                    if (!(candidate.min(0) >= 0) || candidate.contains_nan() ||
                        candidate.contains_inf()) {
                        return failed(ImplicitDiffusionFailure::NonnegativeIterate);
                    }
                    // Under-relaxed Picard iteration preserves nonnegativity.
                    if (residual_candidate == nullptr) {
                        amrex::MultiFab::LinComb(candidate, options.nonlinear_relaxation, candidate, 0,
                                                 1 - options.nonlinear_relaxation, iterate, 0, 0, 1, 0);
                    }
                    fill_coefficients(candidate);
                    amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpSum, amrex::ReduceOpSum,
                                     amrex::ReduceOpSum>
                        residual_ops;
                    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real> residual_data(
                        residual_ops);
                    using ResidualTuple = typename decltype(residual_data)::Type;
                    for (amrex::MFIter mfi(candidate); mfi.isValid(); ++mfi) {
                        auto const energy = candidate.const_array(mfi);
                        auto const initial_energy = radiation.const_array(mfi);
                        auto const alpha = opacity.const_array(mfi);
                        amrex::Array4<amrex::Real const> absorption, emission;
                        if (has_source) {
                            absorption = options.absorption_rate->const_array(mfi);
                            emission = options.emissivity->const_array(mfi);
                        }
                        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> coefficients;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            coefficients[d] = b[d]->const_array(mfi);
                        }
                        residual_ops.eval(mfi.validbox(), residual_data,
                                          [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ResidualTuple
                                          {
                                              auto const residual = cellResidual(
                                                  i, j, k, group, energy, initial_energy, alpha,
                                                  absorption, emission, coefficients, periodic, metric,
                                                  volume_scale, time, dt, has_source, options);
                                              return {std::abs(residual.equation) / metric.volume(i),
                                                      residual.boundary_out, residual.boundary_in,
                                                      -residual.equation};
                                          });
                    }
                    auto const values = residual_data.value();
                    amrex::Real norm = amrex::get<0>(values);
                    amrex::ParallelDescriptor::ReduceRealMax(norm);
                    amrex::Real const scale = amrex::max(initial_norm, candidate.norm0(0));
                    amrex::Real const relative = scale > 0 ? norm / scale : norm;
                    amrex::MultiFab::Copy(iterate, candidate, 0, 0, 1, 0);
                    if (residual_candidate == nullptr) { ++result.nonlinear_iterations; }
                    if (relative <= options.tolerance || residual_candidate != nullptr) {
                        result.maximum_relative_residual =
                            amrex::max(result.maximum_relative_residual, relative);
                        group_out = amrex::get<1>(values);
                        group_in = amrex::get<2>(values);
                        group_defect = amrex::get<3>(values);
                        amrex::ParallelDescriptor::ReduceRealSum(group_out);
                        amrex::ParallelDescriptor::ReduceRealSum(group_in);
                        amrex::ParallelDescriptor::ReduceRealSum(group_defect);
                        converged = true;
                        break;
                    }
                }
                if (!converged) {
                    return failed(ImplicitDiffusionFailure::NonlinearConvergence);
                }
                result.escaped_energy += group_out;
                result.injected_energy += group_in;
                result.numerical_energy_residual += group_defect;
                result.group_escaped_energy[group] = group_out;
                result.group_injected_energy[group] = group_in;
                for (amrex::MFIter mfi(accepted); mfi.isValid(); ++mfi) {
                    auto const energy = iterate.const_array(mfi);
                    auto const output = accepted.array(mfi);
                    amrex::ParallelFor(mfi.validbox(),
                                       [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                           output(i, j, k, group) = energy(i, j, k) * metric.volume(i);
                                       });
                }
                if (has_source) {
                    amrex::ReduceOps<amrex::ReduceOpSum> source_ops;
                    amrex::ReduceData<amrex::Real> source_data(source_ops);
                    using SourceTuple = typename decltype(source_data)::Type;
                    for (amrex::MFIter mfi(accepted); mfi.isValid(); ++mfi) {
                        auto const energy = accepted.const_array(mfi);
                        auto const absorption = options.absorption_rate->const_array(mfi);
                        auto const emission = options.emissivity->const_array(mfi);
                        source_ops.eval(
                            mfi.validbox(), source_data,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> SourceTuple {
                                return {dt * (absorption(i, j, k, group) * energy(i, j, k, group) -
                                              metric.volume(i) * emission(i, j, k, group))};
                            });
                    }
                    auto transfer = amrex::get<0>(source_data.value());
                    amrex::ParallelDescriptor::ReduceRealSum(transfer);
                    result.group_material_energy[group] = transfer;
                }
            }
            if (residual_candidate == nullptr) {
                amrex::MultiFab::Copy(radiation, accepted, 0, 0, radiation.nComp(), 0);
                radiation.FillBoundary(geometry.periodicity());
            }
            return result;
        }
    } // namespace

    ImplicitDiffusionResult
    TryAdvanceImplicitDiffusion (amrex::MultiFab& radiation, amrex::MultiFab const& opacity,
                                 amrex::Geometry const& geometry, amrex::Real time, amrex::Real dt,
                                 ImplicitDiffusionOptions const& options)
    {
        return SolveImplicitDiffusion(radiation, opacity, geometry, time, dt, options, nullptr);
    }

    ImplicitDiffusionResult
    EvaluateImplicitDiffusionResidual (amrex::MultiFab const& old_radiation,
                                       amrex::MultiFab const& candidate,
                                       amrex::MultiFab const& opacity,
                                       amrex::Geometry const& geometry, amrex::Real time,
                                       amrex::Real dt, ImplicitDiffusionOptions const& options)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            candidate.boxArray() == old_radiation.boxArray() &&
                candidate.DistributionMap() == old_radiation.DistributionMap() &&
                candidate.nComp() == old_radiation.nComp(),
            "Implicit residual candidate must match the old radiation layout.");
        amrex::MultiFab scratch(old_radiation.boxArray(), old_radiation.DistributionMap(),
                                old_radiation.nComp(), 0);
        amrex::MultiFab::Copy(scratch, old_radiation, 0, 0, old_radiation.nComp(), 0);
        return SolveImplicitDiffusion(scratch, opacity, geometry, time, dt, options, &candidate);
    }

    ImplicitDiffusionResult
    AdvanceImplicitDiffusion (amrex::MultiFab& radiation, amrex::MultiFab const& opacity,
                              amrex::Geometry const& geometry, amrex::Real time, amrex::Real dt,
                              ImplicitDiffusionOptions const& options)
    {
        auto const result =
            TryAdvanceImplicitDiffusion(radiation, opacity, geometry, time, dt, options);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            result.failure == ImplicitDiffusionFailure::None,
            "Implicit radiation diffusion failed its linear/nonlinear convergence or "
            "positivity gate. No radiation groups or boundary transfers have been committed. "
            "The runtime adapter does not yet retry the PIC step.");
        return result;
    }
} // namespace warpx::radiation
