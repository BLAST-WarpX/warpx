/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "ResolvedThermalDiffusionField.H"

#include "IsotropicDipoleFrameMatrices.H"
#include "Utils/WarpXConst.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <limits>

namespace warpx::radiation
{
#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE) && \
    !defined(AMREX_USE_FLOAT)
// Compile Cartesian implementation helpers only where the public operator
// is supported. The other dimensionalities retain the explicit reject path.
namespace
{
amrex::MultiFab
PrivateHost (amrex::MultiFab const& input, amrex::Geometry const& geometry)
{
    amrex::MultiFab field(input.boxArray(), input.DistributionMap(), input.nComp(), 1,
                          amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::MultiFab::Copy(field, input, 0, 0, input.nComp(), 0);
    field.FillBoundary(geometry.periodicity());
    amrex::Gpu::streamSynchronize();
    return field;
}

amrex::MultiFab
PrivateHost (amrex::BoxArray const& boxes, amrex::DistributionMapping const& distribution,
             int components, int ghosts = 0)
{
    amrex::MultiFab field(boxes, distribution, components, ghosts,
                          amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    field.setVal(0);
    amrex::Gpu::streamSynchronize();
    return field;
}

amrex::MultiFab
DeviceCopy (amrex::MultiFab const& field)
{
    amrex::MultiFab result(field.boxArray(), field.DistributionMap(), field.nComp(),
                           field.nGrowVect());
    amrex::htod_memcpy(result, field);
    amrex::Gpu::streamSynchronize();
    return result;
}

int
PhysicalAxis (int axis)
{
    return AMREX_SPACEDIM == 1 ? 2 : AMREX_SPACEDIM == 2 && axis == 1 ? 2 : axis;
}

struct FaceValue
{
    amrex::Real current = 0;
    amrex::Real density = 0;
    amrex::Real leakage_rate = 0;
};

FaceValue
DiffusiveFace (amrex::FArrayBox const& density, amrex::FArrayBox const& opacity,
               amrex::IArrayBox const& mask, amrex::IntVect face, int axis, amrex::Real spacing,
               amrex::Real extrapolation)
{
    auto left = face;
    --left[axis];
    bool const dleft = mask(left) == 1, dright = mask(face) == 1;
    FaceValue result;
    if (dleft && dright)
    {
        auto const denominator = opacity(left) + opacity(face);
        result.current =
            -2 * PhysConst::c * (density(face) - density(left)) / (3 * spacing * denominator);
        result.density =
            (opacity(face) * density(left) + opacity(left) * density(face)) / denominator;
    }
    else if (dleft || dright)
    {
        auto const source = dleft ? left : face;
        auto const denominator = opacity(source) * spacing + 2 * extrapolation;
        result.leakage_rate = 2 * PhysConst::c * density(source) / (3 * denominator);
        result.current = (dleft ? 1 : -1) * result.leakage_rate;
        result.density = 2 * extrapolation * density(source) / denominator;
    }
    return result;
}

bool
Collective (bool valid)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid;
}

void
UpdateImplicitDrift (amrex::MultiFab const& energy_density, amrex::MultiFab const& proper_bath,
                     amrex::MultiFab const& opacity, amrex::MultiFab const& absorption,
                     amrex::MultiFab const& velocity, amrex::iMultiFab const& mask,
                     amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>& current)
{
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
    {
        for (amrex::MFIter it(*current[axis]); it.isValid(); ++it)
        {
            for (amrex::BoxIterator face(it.validbox()); face.ok(); ++face)
            {
                auto left = face();
                --left[axis];
                amrex::GpuArray<amrex::Real, 3> beta{};
                for (int d = 0; d < 3; ++d)
                {
                    beta[d] = .5 * (velocity[it](left, d) + velocity[it](face(), d)) / PhysConst::c;
                }
                auto const component = PhysicalAxis(axis);
                auto const donor = beta[component] >= 0 ? left : face();
                (*current[axis])[it](face()) = 0;
                if (mask[it](donor) == 0)
                {
                    continue;
                }
                StressEnergy moment{}, source{};
                bool const frame_valid = BuildIsotropicDipoleFrameMatrices(beta, moment, source);
                auto const ratio = absorption[it](donor) / opacity[it](donor);
                auto const equilibrium = moment[0][0] * proper_bath[it](donor);
                auto const speed = PhysConst::c * beta[component];
                auto const advection = PhysConst::c * moment[component + 1][0] / moment[0][0];
                (*current[axis])[it](face()) =
                    frame_valid ? advection * energy_density[it](donor) -
                                      ratio * speed * (energy_density[it](donor) - equilibrium)
                                : std::numeric_limits<amrex::Real>::quiet_NaN();
            }
        }
    }
}
} // namespace
#endif

ResolvedThermalDiffusionFieldResult
TryAdvanceResolvedThermalDiffusionField (
    amrex::MultiFab const& old_energy, amrex::MultiFab const& temperature,
    amrex::MultiFab const& transport, amrex::MultiFab const& absorption,
    amrex::iMultiFab const& diffusion_mask, amrex::MultiFab const& velocity,
    amrex::MultiFab const& work, amrex::Geometry const& geometry, amrex::Real dt,
    ResolvedThermalDiffusionFieldOptions const& options,
    amrex::MultiFab const* incoming_temperature, amrex::MultiFab const* reciprocal_bath)
{
    ResolvedThermalDiffusionFieldResult rejected;
    rejected.reason = "invalid or unsupported resolved thermal diffusion trial";
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE) ||         \
    defined(AMREX_USE_FLOAT)
    amrex::ignore_unused(old_energy, temperature, transport, absorption, diffusion_mask, velocity,
                         work, geometry, dt, options, incoming_temperature, reciprocal_bath);
    return rejected;
#else
    auto const& boxes = old_energy.boxArray();
    auto const& distribution = old_energy.DistributionMap();
    bool valid =
        geometry.Coord() == 0 && geometry.isAllPeriodic() && old_energy.ixType().cellCentered() &&
        std::isfinite(dt) && dt > 0 && options.radiation_constant > 0 &&
        std::isfinite(options.radiation_constant) && options.minimum_optical_depth >= 8 &&
        options.maximum_beta > 0 && options.maximum_beta <= .01 &&
        options.linear.linear_tolerance > 0 && options.linear.linear_tolerance < 1 &&
        options.linear.residual_tolerance > 0 && options.linear.residual_tolerance < 1 &&
        options.linear.max_linear_iterations > 0 && options.max_drift_iterations > 0 &&
        (!options.lte_centered_moving_source || options.maximum_beta <= .001) &&
        options.linear.marshak_extrapolation_lambda > 0 && old_energy.nComp() == 1 &&
        temperature.nComp() == 1 && transport.nComp() == 1 && absorption.nComp() == 1 &&
        work.nComp() == 1 && velocity.nComp() == 3 && diffusion_mask.nComp() == 1 &&
        diffusion_mask.boxArray() == boxes && diffusion_mask.DistributionMap() == distribution;
    for (auto const* field : {&temperature, &transport, &absorption, &velocity, &work})
    {
        valid = valid && field->boxArray() == boxes && field->DistributionMap() == distribution;
    }
    int old_temperature_low = incoming_temperature ? 1 : 0;
    int old_temperature_high = old_temperature_low;
    amrex::ParallelDescriptor::ReduceIntMin(old_temperature_low);
    amrex::ParallelDescriptor::ReduceIntMax(old_temperature_high);
    valid = valid && old_temperature_low == old_temperature_high;
    if (incoming_temperature)
    {
        valid = valid && incoming_temperature->nComp() == 1 &&
                incoming_temperature->boxArray() == boxes &&
                incoming_temperature->DistributionMap() == distribution;
    }
    int bath_low = reciprocal_bath ? 1 : 0, bath_high = bath_low;
    amrex::ParallelDescriptor::ReduceIntMin(bath_low);
    amrex::ParallelDescriptor::ReduceIntMax(bath_high);
    valid = valid && bath_low == bath_high;
    if (reciprocal_bath)
    {
        valid = valid && options.lte_centered_moving_source && reciprocal_bath->nComp() == 1 &&
                reciprocal_bath->boxArray() == boxes &&
                reciprocal_bath->DistributionMap() == distribution;
    }
    // Compare every collective branch/solver control before any callbacks or solve.
    amrex::Real minimum[] = {dt,
                             options.radiation_constant,
                             options.minimum_optical_depth,
                             options.maximum_beta,
                             options.linear.linear_tolerance,
                             options.linear.residual_tolerance,
                             options.linear.marshak_extrapolation_lambda,
                             static_cast<amrex::Real>(options.linear.max_linear_iterations),
                             static_cast<amrex::Real>(options.linear.verbosity),
                             options.lte_centered_moving_source ? 1. : 0.,
                             static_cast<amrex::Real>(options.max_drift_iterations)};
    amrex::Real maximum[11];
    std::copy(std::begin(minimum), std::end(minimum), maximum);
    amrex::ParallelDescriptor::ReduceRealMin(minimum, 11);
    amrex::ParallelDescriptor::ReduceRealMax(maximum, 11);
    valid = valid && std::equal(std::begin(minimum), std::end(minimum), maximum);
    if (!Collective(valid))
    {
        return rejected;
    }
    auto old = PrivateHost(old_energy, geometry), te = PrivateHost(temperature, geometry);
    amrex::MultiFab direct_bath;
    if (reciprocal_bath)
    {
        direct_bath = PrivateHost(*reciprocal_bath, geometry);
    }
    auto old_te = PrivateHost(incoming_temperature ? *incoming_temperature : temperature, geometry);
    auto chi = PrivateHost(transport, geometry), chi_a = PrivateHost(absorption, geometry);
    auto v = PrivateHost(velocity, geometry), mechanical = PrivateHost(work, geometry);
    amrex::iMultiFab mask(boxes, distribution, 1, 1,
                          amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::iMultiFab::Copy(mask, diffusion_mask, 0, 0, 1, 0);
    mask.FillBoundary(geometry.periodicity());
    amrex::Gpu::streamSynchronize();
    auto const dx = geometry.CellSizeArray();
    amrex::Real volume = 1, length = 0, minimum_spacing = dx[0];
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        volume *= dx[d];
        length = std::max(length, geometry.ProbHi(d) - geometry.ProbLo(d));
        minimum_spacing = std::min(minimum_spacing, dx[d]);
    }
    auto bath = PrivateHost(boxes, distribution, 1, 1);
    auto proper_bath = PrivateHost(boxes, distribution, 1, 1);
    auto reference = PrivateHost(boxes, distribution, 1, 1);
    amrex::Real normalization = 1;
    for (amrex::MFIter it(old); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            auto const flag = mask[it](index);
            amrex::Real squared = 0;
            for (int d = 0; d < 3; ++d)
            {
                squared += v[it](index, d) * v[it](index, d);
            }
            auto const beta = std::sqrt(squared) / PhysConst::c;
            auto const t = te[it](index), t2 = t * t;
            auto equilibrium = reciprocal_bath ? volume * direct_bath[it](index)
                                               : volume * options.radiation_constant * t2 * t2;
            valid = valid && equilibrium >= 0;
            proper_bath[it](index) = equilibrium / volume;
            if (options.lte_centered_moving_source)
            {
                amrex::GpuArray<amrex::Real, 3> beta_vector{};
                for (int d = 0; d < 3; ++d)
                {
                    beta_vector[d] = v[it](index, d) / PhysConst::c;
                }
                StressEnergy moment{}, source{};
                valid = BuildIsotropicDipoleFrameMatrices(beta_vector, moment, source) && valid;
                equilibrium *= moment[0][0];
            }
            auto const rate = dt * PhysConst::c * chi_a[it](index);
            valid = valid && (flag == 0 || flag == 1) && old[it](index) >= 0 && t >= 0 &&
                    std::isfinite(old[it](index) + equilibrium + squared + mechanical[it](index) +
                                  rate) &&
                    beta <= options.maximum_beta;
            if (flag == 0)
            {
                valid = valid && old[it](index) == 0 && chi[it](index) == 0 &&
                        chi_a[it](index) == 0 && mechanical[it](index) == 0;
                continue;
            }
            auto const total = chi[it](index), absorb = chi_a[it](index);
            valid = valid && total * minimum_spacing >= options.minimum_optical_depth &&
                    absorb >= 0 && absorb <= total && std::isfinite(total) &&
                    ((absorb == 0 && !options.lte_centered_moving_source) ||
                     beta * total * length <= .01);
            auto const initial_t = old_te[it](index), initial_t2 = initial_t * initial_t;
            auto const initial_bath = volume * options.radiation_constant * initial_t2 * initial_t2;
            valid = valid && initial_t >= 0 && std::isfinite(initial_bath);
            auto const scale = std::max(old[it](index), initial_bath);
            if (!options.lte_centered_moving_source && beta > 0 && absorb > 0 && scale > 0)
            {
                valid = valid && absorb / total * std::abs(old[it](index) - initial_bath) / scale <=
                                     1 / (total * length * total * length);
            }
            bath[it](index) = equilibrium / volume;
            reference[it](index) = (rate > 1 ? equilibrium : old[it](index)) / volume;
            normalization = std::max(normalization, 1 + rate);
        }
    }
    amrex::ParallelDescriptor::ReduceRealMax(normalization);
    if (!Collective(valid))
    {
        rejected.reason = "resolved static-diffusion envelope";
        return rejected;
    }
    bath.FillBoundary(geometry.periodicity());
    proper_bath.FillBoundary(geometry.periodicity());
    reference.FillBoundary(geometry.periodicity());
    amrex::Gpu::streamSynchronize();
    auto a = PrivateHost(boxes, distribution, 1);
    auto rhs = PrivateHost(boxes, distribution, 1);
    auto rhs_base = PrivateHost(boxes, distribution, 1);
    auto delta = PrivateHost(boxes, distribution, 1, 1);
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> b, enthalpy;
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
    {
        auto const faces = amrex::convert(boxes, amrex::IntVect::TheDimensionVector(axis));
        b[axis] = std::make_unique<amrex::MultiFab>(PrivateHost(faces, distribution, 1));
        enthalpy[axis] = std::make_unique<amrex::MultiFab>(PrivateHost(faces, distribution, 1));
        for (amrex::MFIter it(*b[axis]); it.isValid(); ++it)
        {
            for (amrex::BoxIterator face(it.validbox()); face.ok(); ++face)
            {
                auto left = face();
                --left[axis];
                auto const right = face();
                if (mask[it](left) == 1 && mask[it](right) == 1)
                {
                    (*b[axis])[it](right) =
                        2 * PhysConst::c / (3 * (chi[it](left) + chi[it](right)));
                }
                auto const speed =
                    .5 * (v[it](left, PhysicalAxis(axis)) + v[it](right, PhysicalAxis(axis)));
                auto const donor = speed >= 0 ? left : right;
                (*enthalpy[axis])[it](right) = 4 * old[it](donor) * speed / (3 * volume);
            }
        }
    }
    if (options.lte_centered_moving_source)
    {
        auto guess = PrivateHost(old_energy, geometry);
        for (amrex::MFIter it(guess); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(guess[it].box()); cell.ok(); ++cell)
            {
                guess[it](cell()) /= volume;
            }
        }
        UpdateImplicitDrift(guess, proper_bath, chi, chi_a, v, mask, enthalpy);
    }
    for (amrex::MFIter it(rhs); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            if (mask[it](index) == 0)
            {
                a[it](index) = 1;
                continue;
            }
            auto const rate = dt * PhysConst::c * chi_a[it](index);
            auto const ratio = chi_a[it](index) / chi[it](index);
            amrex::Real leakage = 0, div_reference = 0, div_enthalpy = 0, outgoing = 0;
            for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
            {
                auto upper = index;
                ++upper[axis];
                auto const low =
                    DiffusiveFace(reference[it], chi[it], mask[it], index, axis, dx[axis],
                                  options.linear.marshak_extrapolation_lambda);
                auto const high =
                    DiffusiveFace(reference[it], chi[it], mask[it], upper, axis, dx[axis],
                                  options.linear.marshak_extrapolation_lambda);
                div_reference += (high.current - low.current) / dx[axis];
                div_enthalpy +=
                    ((*enthalpy[axis])[it](upper) - (*enthalpy[axis])[it](index)) / dx[axis];
                for (int side : {-1, 1})
                {
                    auto neighbor = index;
                    neighbor[axis] += side;
                    if (mask[it](neighbor) == 0)
                    {
                        leakage += 2 * PhysConst::c /
                                   (3 *
                                    (chi[it](index) * dx[axis] +
                                     2 * options.linear.marshak_extrapolation_lambda) *
                                    dx[axis]);
                    }
                    auto face = index;
                    face[axis] += side > 0 ? 1 : 0;
                    auto left = face;
                    --left[axis];
                    auto const speed =
                        .5 * (v[it](left, PhysicalAxis(axis)) + v[it](face, PhysicalAxis(axis)));
                    outgoing += 4 * std::max(side * speed, amrex::Real(0)) / (3 * dx[axis]);
                }
            }
            valid = valid && dt * outgoing <= 1;
            a[it](index) = (1 + rate + dt * leakage) / normalization;
            rhs_base[it](index) =
                ((old[it](index) / volume - reference[it](index)) -
                 rate * (reference[it](index) - bath[it](index)) -
                 (1 - 2 * ratio) * mechanical[it](index) / volume - dt * div_reference) /
                normalization;
            rhs[it](index) = rhs_base[it](index) - dt * div_enthalpy / normalization;
            valid = valid && std::isfinite(a[it](index) + rhs[it](index)) && a[it](index) > 0;
        }
    }
    if (!Collective(valid))
    {
        rejected.reason = "resolved centered RHS or enthalpy CFL";
        return rejected;
    }
    auto device_a = DeviceCopy(a), device_rhs = DeviceCopy(rhs), device_delta = DeviceCopy(delta);
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> device_b;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        device_b[d] = std::make_unique<amrex::MultiFab>(DeviceCopy(*b[d]));
    }
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> boundary;
    boundary.fill(amrex::LinOpBCType::Periodic);
    amrex::MLABecLaplacian op({geometry}, {boxes}, {distribution});
    op.setMaxOrder(2);
    op.setDomainBC(boundary, boundary);
    op.setLevelBC(0, nullptr);
    op.setScalars(1, dt / normalization);
    op.setACoeffs(0, device_a);
    op.setBCoeffs(0, amrex::GetArrOfConstPtrs(device_b));
    amrex::MLMG solver(op);
    solver.setVerbose(options.linear.verbosity);
    solver.setMaxIter(options.linear.max_linear_iterations);
    solver.setThrowException(true);
    ResolvedThermalDiffusionFieldResult result;
    auto drift_bound = PrivateHost(boxes, distribution, 1);
    bool drift_converged = false;
    int const drift_limit = options.lte_centered_moving_source ? options.max_drift_iterations : 1;
    for (int iteration = 0; iteration < drift_limit; ++iteration)
    {
        try
        {
            result.linear_residual =
                solver.solve({&device_delta}, {&device_rhs}, options.linear.linear_tolerance, 0);
            result.linear_iterations += solver.getNumIters();
        }
        catch (amrex::MLMG::error const&)
        {
            rejected.reason = "resolved centered thermal MLMG solve";
            return rejected;
        }
        delta = PrivateHost(device_delta, geometry);
        if (!options.lte_centered_moving_source)
        {
            drift_converged = true;
            break;
        }
        result.drift_iterations = iteration + 1;
        auto guess = PrivateHost(boxes, distribution, 1, 1);
        for (amrex::MFIter it(guess); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(guess[it].box()); cell.ok(); ++cell)
            {
                guess[it](cell()) = reference[it](cell()) + delta[it](cell());
            }
        }
        amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> next_current;
        for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
        {
            next_current[axis] = std::make_unique<amrex::MultiFab>(
                PrivateHost(enthalpy[axis]->boxArray(), distribution, 1));
        }
        UpdateImplicitDrift(guess, proper_bath, chi, chi_a, v, mask, next_current);
        bool converged = true;
        amrex::Real maximum_defect = 0;
        for (amrex::MFIter it(rhs); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                if (mask[it](cell()) == 0)
                {
                    continue;
                }
                amrex::Real div_new = 0, div_change = 0, operands = 0;
                for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
                {
                    auto upper = cell();
                    ++upper[axis];
                    auto const lo = (*next_current[axis])[it](cell());
                    auto const hi = (*next_current[axis])[it](upper);
                    div_new += (hi - lo) / dx[axis];
                    div_change += ((hi - (*enthalpy[axis])[it](upper)) -
                                   (lo - (*enthalpy[axis])[it](cell()))) /
                                  dx[axis];
                    operands += dt * volume / dx[axis] * (std::abs(hi) + std::abs(lo));
                }
                auto const defect = std::abs(dt * volume * div_change);
                auto const scale =
                    std::abs(old[it](cell())) +
                    volume * (std::abs(reference[it](cell())) + std::abs(delta[it](cell()))) +
                    operands + std::abs(mechanical[it](cell()));
                auto const bound = (options.linear.residual_tolerance +
                                    128 * std::numeric_limits<amrex::Real>::epsilon()) *
                                   scale;
                drift_bound[it](cell()) = defect;
                converged = converged && std::isfinite(defect + bound) && defect <= bound &&
                            guess[it](cell()) >= 0;
                maximum_defect = std::max(maximum_defect, defect);
                rhs[it](cell()) = rhs_base[it](cell()) - dt * div_new / normalization;
            }
        }
        amrex::ParallelDescriptor::ReduceRealMax(maximum_defect);
        result.drift_equation_residual = maximum_defect;
        enthalpy = std::move(next_current);
        if (Collective(converged))
        {
            drift_converged = true;
            break;
        }
        device_rhs = DeviceCopy(rhs);
    }
    if (!drift_converged)
    {
        rejected.reason = "implicit gray source drift iteration budget";
        return rejected;
    }
    auto energy = PrivateHost(boxes, distribution, 1);
    auto momentum = PrivateHost(boxes, distribution, 3);
    auto heat = PrivateHost(boxes, distribution, 1);
    auto physical = PrivateHost(boxes, distribution, 1);
    auto raw = PrivateHost(boxes, distribution, 1);
    auto storage = PrivateHost(boxes, distribution, 1);
    auto stress = PrivateHost(boxes, distribution, 3);
    auto scale = PrivateHost(boxes, distribution, 1);
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> flux, pressure, leakage;
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
    {
        auto const faces = b[axis]->boxArray();
        flux[axis] = std::make_unique<amrex::MultiFab>(PrivateHost(faces, distribution, 1));
        pressure[axis] = std::make_unique<amrex::MultiFab>(PrivateHost(faces, distribution, 1));
        leakage[axis] = std::make_unique<amrex::MultiFab>(PrivateHost(faces, distribution, 1));
        for (amrex::MFIter it(*flux[axis]); it.isValid(); ++it)
        {
            for (amrex::BoxIterator face(it.validbox()); face.ok(); ++face)
            {
                auto const ref =
                    DiffusiveFace(reference[it], chi[it], mask[it], face(), axis, dx[axis],
                                  options.linear.marshak_extrapolation_lambda);
                auto const increment =
                    DiffusiveFace(delta[it], chi[it], mask[it], face(), axis, dx[axis],
                                  options.linear.marshak_extrapolation_lambda);
                (*flux[axis])[it](face()) =
                    ref.current + increment.current + (*enthalpy[axis])[it](face());
                (*pressure[axis])[it](face()) = (ref.density + increment.density) / 3;
                auto left = face();
                --left[axis];
                int const normal = mask[it](left) - mask[it](face());
                if (normal != 0)
                {
                    (*leakage[axis])[it](face()) = dt * volume / dx[axis] *
                                                   (ref.leakage_rate + increment.leakage_rate +
                                                    normal * (*enthalpy[axis])[it](face()));
                }
                valid = valid && (*leakage[axis])[it](face()) >= 0 &&
                        (*pressure[axis])[it](face()) >= 0 &&
                        std::isfinite((*flux[axis])[it](face()) + (*leakage[axis])[it](face()) +
                                      (*pressure[axis])[it](face()));
            }
        }
    }
    auto const epsilon = std::numeric_limits<amrex::Real>::epsilon();
    auto const linear_bound = std::abs(result.linear_residual) * normalization * volume;
    for (amrex::MFIter it(energy); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            if (mask[it](index) == 0)
            {
                continue;
            }
            auto const rate = dt * PhysConst::c * chi_a[it](index);
            auto const ratio = chi_a[it](index) / chi[it](index);
            auto const retained_change =
                volume * ((reference[it](index) - old[it](index) / volume) + delta[it](index));
            energy[it](index) = volume * (reference[it](index) + delta[it](index));
            heat[it](index) =
                rate * volume * ((reference[it](index) - bath[it](index)) + delta[it](index)) -
                2 * ratio * mechanical[it](index);
            physical[it](index) = mechanical[it](index) + heat[it](index);
            amrex::Real transfer = 0, operands = 0;
            for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
            {
                auto upper = index;
                ++upper[axis];
                auto const low = dt * volume / dx[axis] * (*flux[axis])[it](index);
                auto const high = dt * volume / dx[axis] * (*flux[axis])[it](upper);
                transfer += high - low;
                operands += std::abs(high) + std::abs(low);
                auto const component = PhysicalAxis(axis);
                stress[it](index, component) =
                    dt * volume / dx[axis] *
                    ((*pressure[axis])[it](upper) - (*pressure[axis])[it](index));
                auto const low0 =
                    (*flux[axis])[it](index) -
                    (options.lte_centered_moving_source ? 0 : (*enthalpy[axis])[it](index));
                auto const high0 =
                    (*flux[axis])[it](upper) -
                    (options.lte_centered_moving_source ? 0 : (*enthalpy[axis])[it](upper));
                momentum[it](index, component) = volume * (low0 + high0) / (2 * PhysConst::c);
            }
            for (int d = 0; d < 3; ++d)
            {
                bool spatial_component = false;
                for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
                {
                    spatial_component |= d == PhysicalAxis(axis);
                }
                if (!options.lte_centered_moving_source)
                {
                    momentum[it](index, d) +=
                        4 * energy[it](index) * v[it](index, d) / (3 * PhysConst::c);
                }
                else if (!spatial_component)
                {
                    amrex::GpuArray<amrex::Real, 3> beta{};
                    for (int component = 0; component < 3; ++component)
                    {
                        beta[component] = v[it](index, component) / PhysConst::c;
                    }
                    StressEnergy moment{}, source{};
                    valid = BuildIsotropicDipoleFrameMatrices(beta, moment, source) && valid;
                    momentum[it](index, d) =
                        moment[d + 1][0] / moment[0][0] * energy[it](index) -
                        ratio * beta[d] * (energy[it](index) - volume * bath[it](index));
                }
            }
            raw[it](index) = (old[it](index) - energy[it](index)) - transfer;
            scale[it](index) =
                std::abs(old[it](index)) + std::abs(energy[it](index)) + operands +
                std::abs(mechanical[it](index)) + std::abs(heat[it](index)) +
                volume * (std::abs(reference[it](index)) + std::abs(delta[it](index)));
            storage[it](index) =
                128 * epsilon * scale[it](index) + 2 * linear_bound + drift_bound[it](index);
            auto const residual = retained_change + transfer + physical[it](index);
            auto const equation_bound =
                options.linear.residual_tolerance *
                    (std::abs(retained_change) + operands + std::abs(physical[it](index))) +
                storage[it](index);
            valid = valid && energy[it](index) >= 0 && std::isfinite(residual + equation_bound) &&
                    std::abs(residual) <= equation_bound &&
                    std::abs(raw[it](index) - physical[it](index)) <= equation_bound;
            amrex::Real squared = 0;
            for (int d = 0; d < 3; ++d)
            {
                squared += v[it](index, d) * v[it](index, d);
            }
            auto const equilibrium = volume * bath[it](index);
            auto const magnitude = std::max(energy[it](index), equilibrium);
            if (!options.lte_centered_moving_source && squared > 0 && rate > 0 && magnitude > 0)
            {
                valid = valid && ratio * std::abs(energy[it](index) - equilibrium) / magnitude <=
                                     1 / (chi[it](index) * length * chi[it](index) * length);
            }
        }
    }
    if (!Collective(valid))
    {
        rejected.reason = "resolved physical spatial/thermal ledger or envelope";
        return rejected;
    }
    result.energy = DeviceCopy(energy);
    result.lab_momentum = DeviceCopy(momentum);
    result.requested_heat = DeviceCopy(heat);
    result.physical_energy_debit = DeviceCopy(physical);
    result.raw_energy_debit = DeviceCopy(raw);
    result.storage_roundoff_bound = DeviceCopy(storage);
    result.outward_pressure_stress = DeviceCopy(stress);
    result.operand_scale = DeviceCopy(scale);
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        result.face_lab_energy_current[d] = std::make_unique<amrex::MultiFab>(DeviceCopy(*flux[d]));
        result.face_pressure[d] = std::make_unique<amrex::MultiFab>(DeviceCopy(*pressure[d]));
        result.face_leaked_energy[d] = std::make_unique<amrex::MultiFab>(DeviceCopy(*leakage[d]));
    }
    result.valid = true;
    return result;
#endif
}
} // namespace warpx::radiation
