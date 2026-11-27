/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "ResolvedRZThermalDiffusionField.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FrameTransform.H"
#include "GroupMomentGeometry.H"
#include "NativeCompositeTableCornerCapture.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>

namespace warpx::radiation
{
namespace
{
#if defined(WARPX_DIM_RZ)
using Faces = amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>;

bool Collective (bool valid)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid;
}

bool
SupportedNativeThermalContext (amrex::Geometry const& geometry,
                               amrex::BoxArray const& boxes,
                               amrex::DistributionMapping const& mapping,
                               bool reflecting_axial_field_trial = false,
                               bool reflecting_axial_table_trial = false,
                               bool single_selected_table_trial = false,
                               bool single_selected_table_vacuum = false)
{
    auto& simulation = WarpX::GetInstance();
    auto const& native = simulation.Geom(0);
    auto const* model = simulation.get_pointer_HybridPICModel();
    bool selected = false;
    if (single_selected_table_trial)
    {
        if (!Collective(model != nullptr)) { return false; }
        selected = single_selected_table_vacuum
            ? detail::HasNativeSingleSelectedTableMaterial0OrExactVacuum(*model, geometry)
            : detail::HasNativeSingleSelectedTableMaterial0(*model, geometry);
    }
    bool reflecting_z = false;
#if !defined(AMREX_USE_GPU)
    reflecting_z = reflecting_axial_field_trial && !geometry.isPeriodic(1) &&
        WarpX::field_boundary_lo[1] == FieldBoundaryType::PEC &&
        WarpX::field_boundary_hi[1] == FieldBoundaryType::PEC &&
        WarpX::particle_boundary_lo[1] == ParticleBoundaryType::Reflecting &&
        WarpX::particle_boundary_hi[1] == ParticleBoundaryType::Reflecting;
#else
    amrex::ignore_unused(reflecting_axial_field_trial);
#endif
    bool valid =
        (!single_selected_table_vacuum || single_selected_table_trial) &&
        (!single_selected_table_trial || (selected && !reflecting_axial_field_trial &&
                                         !reflecting_axial_table_trial)) &&
        !EB::enabled() && simulation.finestLevel() == 0 &&
        WarpX::do_moving_window == 0 && !simulation.DoFluidSpecies() &&
        !simulation.GetPartContainer().hasHybridIonization() &&
        model != nullptr && model->solvesElectronEnergyEquation() &&
        model->transportsElectronInternalEnergyAtRawDensity() &&
        WarpX::n_rz_azimuthal_modes == 1 &&
        geometry.Domain() == native.Domain() &&
        geometry.Coord() == native.Coord() && geometry.ProbLo(0) == 0 &&
        boxes == simulation.boxArray(0) &&
        mapping == simulation.DistributionMap(0) && !geometry.isPeriodic(0) &&
        (geometry.isPeriodic(1) || reflecting_z) &&
        WarpX::field_boundary_lo[0] == FieldBoundaryType::None &&
        WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC &&
        WarpX::particle_boundary_lo[0] == ParticleBoundaryType::None &&
        WarpX::particle_boundary_hi[0] == ParticleBoundaryType::Reflecting;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        valid = valid && geometry.ProbLo(d) == native.ProbLo(d) &&
                geometry.ProbHi(d) == native.ProbHi(d) &&
                geometry.isPeriodic(d) == native.isPeriodic(d);
    }
    if (model)
    {
        auto const eos = model->electronThermodynamicsExecutor();
        bool supported_caloric =
            (eos.isIdealGas() || eos.isFixedChargeLatentEnergy()) &&
            eos.m_num_materials == 0;
#if !defined(AMREX_USE_GPU)
        supported_caloric = supported_caloric || (!reflecting_z &&
            (eos.isSingularitySpiner() && eos.m_num_materials == 1 &&
             ((!eos.m_use_resolved_material_selection && !single_selected_table_trial) || selected)));
#endif
        bool closed_table = false;
#if defined(WARPX_USE_SINGULARITY_EOS) && !defined(AMREX_USE_GPU)
        closed_table = reflecting_axial_table_trial && reflecting_z &&
            eos.isSingularitySpiner() && eos.m_num_materials == 1 &&
            !eos.m_use_resolved_material_selection && eos.tableStorage() != nullptr;
        if (closed_table)
        {
            auto const& species = simulation.GetPartContainer().GetParticleContainerFromName(
                model->electronThermodynamicsMaterialSpeciesName(0));
            closed_table = !species.HasEvolvingChargeState();
        }
#endif
        supported_caloric = supported_caloric || closed_table;
        valid = valid && (!reflecting_axial_table_trial || closed_table);
        valid = valid && supported_caloric;
    }
    return valid;
}

bool
SupportedThermalControls (amrex::Real dt,
                          ResolvedRZThermalDiffusionFieldOptions const& options)
{
    return std::isfinite(dt) && dt > 0 && std::isfinite(PhysConst::c * dt) &&
           options.minimum_optical_depth >= 8 && options.maximum_beta > 0 &&
           options.maximum_beta <= .001 && options.max_picard_iterations > 0 &&
           options.linear.max_linear_iterations > 0 &&
           options.linear.linear_tolerance > 0 &&
           options.linear.linear_tolerance <= 1.e-11 &&
           options.linear.residual_tolerance > 0 &&
           options.linear.residual_tolerance <= 1.e-11 &&
           options.linear.marshak_extrapolation_lambda > 0 &&
           std::isfinite(options.minimum_optical_depth + options.maximum_beta +
                         options.linear.linear_tolerance +
                         options.linear.residual_tolerance +
                         options.linear.marshak_extrapolation_lambda);
}

bool
SameThermalControls (amrex::Real dt,
                     ResolvedRZThermalDiffusionFieldOptions const& options)
{
    amrex::Real minimum[13]{
        dt,
        options.minimum_optical_depth,
        options.maximum_beta,
        options.linear.linear_tolerance,
        options.linear.residual_tolerance,
        options.linear.marshak_extrapolation_lambda,
        static_cast<amrex::Real>(options.linear.max_linear_iterations),
        static_cast<amrex::Real>(options.max_picard_iterations),
        static_cast<amrex::Real>(options.reflecting_axial_field_trial),
        static_cast<amrex::Real>(options.reflecting_axial_table_trial),
        static_cast<amrex::Real>(options.single_selected_table_trial),
        static_cast<amrex::Real>(options.single_selected_table_vacuum),
        static_cast<amrex::Real>(options.centered_thermal_difference)};
    amrex::Real maximum[13];
    std::copy(std::begin(minimum), std::end(minimum), maximum);
    amrex::ParallelDescriptor::ReduceRealMin(minimum, 13);
    amrex::ParallelDescriptor::ReduceRealMax(maximum, 13);
    return Collective(
        std::equal(std::begin(minimum), std::end(minimum), maximum));
}

amrex::MultiFab Host (amrex::MultiFab const& input, amrex::Geometry const& geometry)
{
    amrex::MultiFab result(input.boxArray(), input.DistributionMap(), input.nComp(), 1,
                           amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    result.setVal(0);
    amrex::MultiFab::Copy(result, input, 0, 0, input.nComp(), 0);
    result.FillBoundary(geometry.periodicity());
    amrex::Gpu::streamSynchronize();
    return result;
}

amrex::iMultiFab Host (amrex::iMultiFab const& input, amrex::Geometry const& geometry)
{
    amrex::iMultiFab device(input.boxArray(), input.DistributionMap(), 1, 1);
    device.setVal(0);
    amrex::iMultiFab::Copy(device, input, 0, 0, 1, 0);
    device.FillBoundary(geometry.periodicity());
    amrex::iMultiFab result(input.boxArray(), input.DistributionMap(), 1, 1,
                            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::dtoh_memcpy(result, device);
    amrex::Gpu::streamSynchronize();
    return result;
}

amrex::MultiFab HostZero (amrex::BoxArray const& boxes,
                          amrex::DistributionMapping const& mapping,
                          int components, int ghosts = 0)
{
    amrex::MultiFab result(boxes, mapping, components, ghosts,
                           amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    result.setVal(0);
    amrex::Gpu::streamSynchronize();
    return result;
}

amrex::MultiFab Device (amrex::MultiFab const& host)
{
    amrex::MultiFab result(host.boxArray(), host.DistributionMap(), host.nComp(),
                           host.nGrowVect());
    amrex::htod_memcpy(result, host);
    amrex::Gpu::streamSynchronize();
    return result;
}

Faces FaceFields (amrex::BoxArray const& cells, amrex::DistributionMapping const& mapping,
                  int components)
{
    Faces fields;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        auto const boxes = amrex::convert(cells, amrex::IntVect::TheDimensionVector(d));
        fields[d] = std::make_unique<amrex::MultiFab>(HostZero(boxes, mapping, components));
    }
    return fields;
}

struct Quadrature
{
    amrex::Real radius[2]{};
    amrex::Real lever[2]{};
};

Quadrature AnnularQuadrature (int i, GroupMomentMetric const& metric)
{
    auto const lo = metric.Radius(i), hi = metric.Radius(i + 1);
    auto const s_lo = lo * lo, ds = (hi - lo) * (hi + lo);
    auto const mid = s_lo + ds / 2;
    auto const offset = ds / (2 * std::sqrt(3.0));
    auto const mean_radius = metric.Angular(i).mean_radius;
    Quadrature result;
    result.radius[0] = std::sqrt(mid - offset);
    result.radius[1] = std::sqrt(mid + offset);
    result.lever[0] = result.radius[0] / mean_radius;
    result.lever[1] = result.radius[1] / mean_radius;
    return result;
}

/** Necessary, not sufficient, causal cone for an annular LAB inventory.
 * A ray's conjugate theta row is (r/rbar)*E*n_theta. Dividing the summed
 * row by lambda_max=r_hi/rbar leaves (r/r_hi)*E*n_theta for each ray, so
 * the triangle inequality gives the scaled norm <= total nonnegative E.
 * The 64-epsilon gamma covers the geometry, theta division, three-component
 * hypot and comparison on stored operands, not a model or solver tolerance. */
bool NecessaryCausalMoment (amrex::Real energy,
                            amrex::GpuArray<amrex::Real, 3> const& momentum,
                            int radial_cell, GroupMomentMetric const& metric)
{
    if (!std::isfinite(energy) || energy < 0) { return false; }
    for (auto const component : momentum)
    {
        if (!std::isfinite(component)) { return false; }
    }
    if (energy == 0)
    {
        return momentum[0] == 0 && momentum[1] == 0 && momentum[2] == 0;
    }
    auto const upper = metric.Radius(radial_cell + 1);
    auto const mean = metric.Angular(radial_cell).mean_radius;
    if (!(upper > 0 && mean > 0 && std::isfinite(upper + mean))) { return false; }
    auto const lambda_max = upper / mean;
    if (!(lambda_max >= 1 && std::isfinite(lambda_max))) { return false; }
    auto const theta = momentum[1] / lambda_max;
    auto const magnitude = std::hypot(momentum[0], theta, momentum[2]);
    if (!std::isfinite(magnitude)) { return false; }
    if (magnitude <= energy) { return true; }
    auto const scale = std::max({energy, std::abs(momentum[0]),
                                 std::abs(theta), std::abs(momentum[2])});
    auto const numerator = 64 * std::numeric_limits<amrex::Real>::epsilon();
    auto const arithmetic_bound = numerator / (1 - numerator) * scale;
    return magnitude - energy <= arithmetic_bound;
}

amrex::GpuArray<amrex::Real, 3>
PhysicalBeta (amrex::FArrayBox const& velocity, amrex::IntVect const& cell,
              amrex::Real radius, GroupMomentMetric const& metric)
{
    auto const mean_radius = metric.Angular(cell[0]).mean_radius;
    return {velocity(cell, 0) / PhysConst::c,
            velocity(cell, 1) * radius / (mean_radius * PhysConst::c),
            velocity(cell, 2) / PhysConst::c};
}

StressEnergy BoostedDipole (amrex::Real proper_energy,
                            amrex::GpuArray<amrex::Real, 3> const& proper_flux_over_c,
                            amrex::GpuArray<amrex::Real, 3> const& beta)
{
    StressEnergy rest{};
    rest[0][0] = proper_energy;
    for (int d = 0; d < 3; ++d)
    {
        rest[0][d + 1] = proper_flux_over_c[d];
        rest[d + 1][0] = proper_flux_over_c[d];
        rest[d + 1][d + 1] = proper_energy / 3;
    }
    return BoostStressEnergy(rest, beta);
}

struct CellMoments
{
    amrex::Real energy_density = 0;
    amrex::Real offset_density = 0;
    amrex::Real equilibrium_density = 0;
    amrex::Real nonequilibrium_density = 0;
    amrex::Real transport_offset_density = 0;
    amrex::Real theta_conjugate_density = 0;
    amrex::Real theta_pressure = 0;
    amrex::Real m00 = 0;
};

CellMoments CellMoment (
    int radial_cell, amrex::IntVect const& cell,
    amrex::FArrayBox const& velocity, GroupMomentMetric const& metric,
    amrex::Real proper_energy, amrex::Real proper_bath,
    amrex::Real proper_fr_over_c, amrex::Real proper_fz_over_c,
    amrex::Real absorption_ratio, bool centered_difference = false,
    amrex::Real proper_deviation = 0)
{
    CellMoments result;
    auto const quadrature = AnnularQuadrature(radial_cell, metric);
    for (int point = 0; point < 2; ++point)
    {
        auto const beta = PhysicalBeta(velocity, cell, quadrature.radius[point], metric);
        amrex::GpuArray<amrex::Real, 3> total_flux{
            proper_fr_over_c, 0, proper_fz_over_c};
        amrex::GpuArray<amrex::Real, 3> offset_flux{
            proper_fr_over_c, 0, proper_fz_over_c};
        amrex::GpuArray<amrex::Real, 3> unit_flux{};
        for (int d = 0; d < 3; ++d)
        {
            total_flux[d] -= absorption_ratio * beta[d] *
                (centered_difference ? proper_deviation : proper_energy - proper_bath);
            offset_flux[d] += absorption_ratio * beta[d] * proper_bath;
            unit_flux[d] = -absorption_ratio * beta[d];
        }
        auto const tensor = BoostedDipole(proper_energy,
            total_flux, beta);
        auto const offset = BoostedDipole(0, offset_flux, beta);
        auto const equilibrium = BoostedDipole(proper_bath, {0, 0, 0}, beta);
        auto const unit = BoostedDipole(1, unit_flux, beta);
        auto const weight = 0.5;
        if (centered_difference)
        {
            auto const transport = BoostedDipole(
                0, {proper_fr_over_c, 0, proper_fz_over_c}, beta);
            // Linearity of the boost gives E-E_LTE = m00*(U-B) +
            // the boosted transport dipole. Do not form two rounded LAB
            // energies and amplify their subtraction error by a stiff rate.
            result.nonequilibrium_density += weight *
                (unit[0][0] * proper_deviation + transport[0][0]);
            result.transport_offset_density += weight * transport[0][0];
        }
        result.offset_density += weight * offset[0][0];
        result.equilibrium_density += weight * equilibrium[0][0];
        result.theta_pressure += weight * tensor[2][2];
        result.m00 += weight * unit[0][0];
        result.theta_conjugate_density += weight * quadrature.lever[point] *
            tensor[0][2];
    }
    result.energy_density = result.m00 * proper_energy + result.offset_density;
    return result;
}

struct CellPreflight
{
    bool valid = false;
    CellMoments moments;
    CellMoments unit_bath;
    amrex::Real proper_energy = 0;
    amrex::Real normalization = 0;
};

CellPreflight PreflightDiffusionCell (
    amrex::IntVect const& index, amrex::Real energy, amrex::Real source,
    amrex::Real total, amrex::Real absorption, amrex::FArrayBox const& velocity,
    GroupMomentMetric const& metric, amrex::Geometry const& geometry, amrex::Real dt,
    ResolvedRZThermalDiffusionFieldOptions const& options, bool independent_rest = false)
{
    CellPreflight result;
    auto const radius = metric.Radius(index[0] + 1);
    auto const beta = PhysicalBeta(velocity, index, radius, metric);
    auto const beta2 = beta[0] * beta[0] + beta[1] * beta[1] + beta[2] * beta[2];
    auto const speed = std::sqrt(beta2);
    auto const width = std::max(geometry.ProbHi(0) - geometry.ProbLo(0),
                                geometry.ProbHi(1) - geometry.ProbLo(1));
    result.valid = std::isfinite(energy + source + total + absorption + beta2) &&
        energy >= 0 && source >= 0 && total > 0 && absorption >= 0 &&
        (absorption <= total || independent_rest) &&
        speed <= options.maximum_beta &&
        total * std::min(geometry.CellSize(0), geometry.CellSize(1)) >=
            options.minimum_optical_depth && speed * total * width <= .01;
    if (!result.valid) { return result; }
    auto const ratio = absorption / total;
    result.moments = CellMoment(index[0], index, velocity, metric, 1, 0, 0, 0, ratio);
    auto const old_bath = CellMoment(index[0], index, velocity, metric, 0, source, 0, 0, ratio);
    result.unit_bath = CellMoment(index[0], index, velocity, metric, 0, 1, 0, 0, ratio);
    auto const volume = metric.Volume(index[0]);
    result.valid = result.moments.m00 > 0 &&
        std::isfinite(result.moments.m00 + old_bath.offset_density +
                      old_bath.equilibrium_density + result.unit_bath.equilibrium_density) &&
        result.unit_bath.equilibrium_density > 0;
    if (!result.valid) { return result; }
    result.proper_energy = (energy / volume - old_bath.offset_density) / result.moments.m00;
    result.valid = std::isfinite(result.proper_energy) && result.proper_energy >= 0;
    result.normalization = volume * result.moments.m00 * (1 + dt * PhysConst::c * absorption);
    amrex::Real outgoing = 0;
    for (int d = 0; d < 2; ++d)
    {
        auto const area_lo = metric.Area(d, index[0]);
        auto const area_hi = metric.Area(d, index[0] + (d == 0 ? 1 : 0));
        auto const beta_normal = d == 0 ? beta[0] : beta[2];
        auto const speed_bound = (4.0 / 3 + ratio) * PhysConst::c * std::abs(beta_normal);
        if (d != 0 || index[0] > geometry.Domain().smallEnd(0)) { outgoing += area_lo * speed_bound; }
        if (d != 0 || index[0] < geometry.Domain().bigEnd(0)) { outgoing += area_hi * speed_bound; }
    }
    result.valid = result.valid && dt * outgoing / volume <= 1;
    return result;
}

struct FaceMoment
{
    amrex::Real proper_current = 0;
    amrex::Real lab_current = 0;
    amrex::Real leakage_rate = 0;
    amrex::GpuArray<amrex::Real, 3> stress{};
    bool valid = true;
};

bool ReflectingFace (amrex::IntVect const& face, int axis,
                      amrex::Geometry const& geometry)
{
    return !geometry.isPeriodic(axis) &&
        (face[axis] == geometry.Domain().smallEnd(axis) ||
         face[axis] == geometry.Domain().bigEnd(axis) + 1);
}

FaceMoment FaceMomentAt (
    amrex::IntVect const& face, int axis, amrex::FArrayBox const& proper_energy,
    amrex::FArrayBox const& bath, amrex::FArrayBox const& opacity,
    amrex::FArrayBox const& absorption, amrex::FArrayBox const& velocity,
    amrex::IArrayBox const& mask, GroupMomentMetric const& metric,
    amrex::Geometry const& geometry, amrex::Real lambda,
    amrex::FArrayBox const* proper_deviation = nullptr)
{
    FaceMoment result;
    auto left = face;
    --left[axis];
    auto const right = face;
    bool const wall = ReflectingFace(face, axis, geometry);
    bool const left_d = !wall && mask(left) == 1;
    bool const right_d = !wall && mask(right) == 1;
    if (!wall && !left_d && !right_d) { return result; }
    auto const source = wall ? (face[axis] == geometry.Domain().smallEnd(axis) ? right : left)
                             : (left_d ? left : right);
    if (wall && mask(source) == 0) { return result; }
    auto const dx = geometry.CellSize(axis);
    amrex::Real face_energy = 0, face_bath = 0, ratio = 0, face_deviation = 0;
    if (wall)
    {
        face_energy = proper_energy(source);
        face_bath = bath(source);
        if (proper_deviation) { face_deviation = (*proper_deviation)(source); }
        ratio = absorption(source) / opacity(source);
    }
    else if (left_d && right_d)
    {
        auto const denominator = opacity(left) + opacity(right);
        result.valid = denominator > 0;
        if (!result.valid) { return result; }
        result.proper_current = -2 * PhysConst::c *
            (proper_deviation
                 ? (bath(right) - bath(left)) +
                   ((*proper_deviation)(right) - (*proper_deviation)(left))
                 : proper_energy(right) - proper_energy(left)) / (3 * dx * denominator);
        face_energy = (opacity(right) * proper_energy(left) +
                       opacity(left) * proper_energy(right)) / denominator;
        face_bath = (opacity(right) * bath(left) +
                     opacity(left) * bath(right)) / denominator;
        if (proper_deviation)
        {
            face_deviation = (opacity(right) * (*proper_deviation)(left) +
                              opacity(left) * (*proper_deviation)(right)) / denominator;
            face_energy = face_bath + face_deviation;
        }
        ratio = (opacity(right) * absorption(left) / opacity(left) +
                 opacity(left) * absorption(right) / opacity(right)) / denominator;
    }
    else
    {
        auto const denominator = opacity(source) * dx + 2 * lambda;
        result.valid = denominator > 0;
        if (!result.valid) { return result; }
        result.leakage_rate = 2 * PhysConst::c * proper_energy(source) /
                              (3 * denominator);
        result.proper_current = (left_d ? 1 : -1) * result.leakage_rate;
        face_energy = 2 * lambda * proper_energy(source) / denominator;
        face_bath = bath(source);
        ratio = absorption(source) / opacity(source);
        if (proper_deviation) { face_deviation = face_energy - face_bath; }
    }

    int const normal = axis == 0 ? 0 : 2;
    auto const mean_velocity = [&] (amrex::Real radius)
    {
        auto beta = PhysicalBeta(velocity, source, radius, metric);
        if (!wall && left_d && right_d)
        {
            auto const other = source == left ? right : left;
            auto const second = PhysicalBeta(velocity, other, radius, metric);
            for (int d = 0; d < 3; ++d) { beta[d] = 0.5 * (beta[d] + second[d]); }
        }
        if (wall) { beta[normal] = 0; }
        return beta;
    };
    auto sample = [&] (amrex::Real radius, amrex::Real weight,
                       amrex::Real angular_lever)
    {
        auto const beta = mean_velocity(radius);
        auto const flux = result.proper_current / PhysConst::c;
        amrex::GpuArray<amrex::Real, 3> proper_dipole{};
        proper_dipole[normal] = flux;
        for (int d = 0; d < 3; ++d)
        {
            proper_dipole[d] -= ratio * beta[d] *
                (proper_deviation ? face_deviation : face_energy - face_bath);
        }
        auto const tensor = BoostedDipole(face_energy, proper_dipole, beta);
        result.lab_current += weight * PhysConst::c * tensor[0][normal + 1];
        for (int d = 0; d < 3; ++d)
        {
            result.stress[d] += weight * (axis == 1 && d == 1 ? angular_lever : 1) *
                PhysConst::c * tensor[d + 1][normal + 1];
        }
    };
    if (axis == 0)
    {
        sample(metric.Radius(face[0]), 1, 1);
    }
    else
    {
        auto const quadrature = AnnularQuadrature(face[0], metric);
        sample(quadrature.radius[0], 0.5, quadrature.lever[0]);
        sample(quadrature.radius[1], 0.5, quadrature.lever[1]);
    }
    if (wall)
    {
        result.lab_current = 0;
        result.proper_current = 0;
        for (int d = 0; d < 3; ++d)
        {
            if (d != normal) { result.stress[d] = 0; }
        }
    }
    result.valid = std::isfinite(result.lab_current + result.proper_current +
                                 result.stress[0] + result.stress[1] + result.stress[2]) &&
                   face_energy >= 0 && result.leakage_rate >= 0;
    return result;
}

bool BuildFaces (amrex::MultiFab const& proper, amrex::MultiFab const& bath,
                 amrex::MultiFab const& opacity, amrex::MultiFab const& absorption,
                 amrex::MultiFab const& velocity, amrex::iMultiFab const& mask,
                 GroupMomentMetric const& metric, amrex::Geometry const& geometry,
                 amrex::Real lambda, Faces& proper_current, Faces& lab_current,
                 Faces& stress, Faces& leakage,
                 amrex::MultiFab const* proper_deviation = nullptr)
{
    bool valid = true;
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
    {
        for (amrex::MFIter it(*lab_current[axis]); it.isValid(); ++it)
        {
            for (amrex::BoxIterator face(it.validbox()); face.ok(); ++face)
            {
                auto const value = FaceMomentAt(face(), axis, proper[it], bath[it],
                    opacity[it], absorption[it], velocity[it], mask[it], metric, geometry, lambda,
                    proper_deviation ? &(*proper_deviation)[it] : nullptr);
                valid = valid && value.valid;
                (*proper_current[axis])[it](face()) = value.proper_current;
                (*lab_current[axis])[it](face()) = value.lab_current;
                (*leakage[axis])[it](face()) = value.leakage_rate;
                for (int d = 0; d < 3; ++d)
                {
                    (*stress[axis])[it](face(), d) = value.stress[d];
                }
            }
        }
    }
    return Collective(valid);
}

struct IndependentChannelConditioning
{
    bool valid = false;
    long double transport_error = 0;
    long double absorption_error = 0;
};

// This is the strict-rest entry's original channel/inverse/ratio operation
// graph, factored without changing its order or its resolution ceiling.
template <typename Proof>
IndependentChannelConditioning ConditionIndependentChannels (
    Proof const& proof, amrex::Real r, amrex::Real p, amrex::Real dt)
{
    auto const upward = [] (long double x)
    {
        return std::nextafter(x, std::numeric_limits<long double>::infinity());
    };
    constexpr long double unit_roundoff = std::numeric_limits<amrex::Real>::epsilon() / 2;
    constexpr long double resolution = 0x1p-26L;
    auto const er = upward(proof.radius[0] / proof.ideal[0].lo);
    auto const ep = upward(proof.radius[1] / proof.ideal[1].lo);
    auto const denominator = std::nextafter(1 - er, 0.L);
    auto const inverse_error = upward(upward(er + unit_roundoff) / denominator);
    auto const numerator = upward(upward(upward(ep + er) + unit_roundoff) +
                                   upward(ep * unit_roundoff));
    auto const ratio_error = upward(numerator / denominator);
    auto const ratio = p / r;
    return {er <= resolution && ep <= resolution && denominator > 0 &&
            inverse_error <= resolution && ratio_error <= resolution &&
            std::isnormal(1 / r) && std::isnormal(ratio) && ratio > 0 &&
            std::isfinite(2 * ratio) && std::isfinite(1 - 2 * ratio) &&
            std::isfinite((4.0 / 3 + ratio) * PhysConst::c) &&
            std::isfinite(dt * PhysConst::c * p), er, ep};
}

#if !defined(AMREX_USE_GPU) && defined(WARPX_USE_SINGULARITY_EOS) && \
    defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
// Errors below are relative to the authentic ideal channel, not to a fitted
// solution. Every positive operation follows the actual eager face DAG.
long double ErrorUp (long double value)
{
    return std::nextafter(value, std::numeric_limits<long double>::infinity());
}

long double ProductError (long double a, long double b)
{
    constexpr long double u = std::numeric_limits<amrex::Real>::epsilon() / 2;
    return ErrorUp(ErrorUp(ErrorUp(ErrorUp(1 + a) * ErrorUp(1 + b)) * (1 + u)) - 1);
}

long double QuotientError (long double a, long double b)
{
    constexpr long double u = std::numeric_limits<amrex::Real>::epsilon() / 2;
    auto const denominator = std::nextafter(1 - b, 0.L);
    return denominator > 0 ? ErrorUp(ErrorUp(ErrorUp(ErrorUp(1 + a) * (1 + u)) /
                                                        denominator) - 1)
                           : std::numeric_limits<long double>::infinity();
}

long double SumError (long double a, long double b)
{
    constexpr long double u = std::numeric_limits<amrex::Real>::epsilon() / 2;
    return ErrorUp(ErrorUp(ErrorUp(1 + std::max(a, b)) * (1 + u)) - 1);
}

struct IndependentFaceConditioning
{
    bool valid = false;
    amrex::Real ratio = 0;
    amrex::Real transport = 0;
    amrex::Real absorption = 0;
};

IndependentFaceConditioning ConditionIndependentFace (
    amrex::Real rl, amrex::Real rr, amrex::Real pl, amrex::Real pr,
    amrex::FArrayBox const& errors, amrex::IntVect const& left,
    amrex::IntVect const& right)
{
    auto const a = rr * pl, b = rl * pr;
    auto const denominator = rl + rr;
    auto const first = a / rl, second = b / rr;
    auto const sum = first + second;
    auto const ratio = sum / denominator;
    auto const r = denominator / 2;
    auto const p = ratio * r;
    bool valid = std::isnormal(a) && a > 0 && std::isnormal(b) && b > 0 &&
        std::isnormal(denominator) && denominator > 0 &&
        std::isnormal(first) && first > 0 && std::isnormal(second) && second > 0 &&
        std::isnormal(sum) && sum > 0 && std::isnormal(ratio) && ratio > 0 &&
        std::isnormal(r) && r > 0 && std::isnormal(p) && p > 0;
    auto const erl = static_cast<long double>(errors(left, 0));
    auto const err = static_cast<long double>(errors(right, 0));
    auto const epl = static_cast<long double>(errors(left, 1));
    auto const epr = static_cast<long double>(errors(right, 1));
    auto const ea = ProductError(err, epl), eb = ProductError(erl, epr);
    auto const ed = SumError(erl, err);
    auto const ef = QuotientError(ea, erl), es = QuotientError(eb, err);
    auto const eq = QuotientError(SumError(ef, es), ed);
    // Division by two is exact for the required normal represented R_f.
    auto const ep = ProductError(eq, ed);
    constexpr long double resolution = 0x1p-26L;
    for (auto const error : {ea, eb, ed, ef, es, eq, ep})
    {
        valid = valid && error >= 0 && std::isfinite(error) && error <= resolution;
    }
    return {valid, ratio, r, p};
}

bool IndependentFrameSample (
    amrex::FArrayBox const& velocity, amrex::IntVect const& cell,
    amrex::Real radius, GroupMomentMetric const& metric,
    amrex::Real r, amrex::Real p, amrex::Real width, amrex::Real maximum_beta)
{
    auto const mean = metric.Angular(cell[0]).mean_radius;
    auto const beta = PhysicalBeta(velocity, cell, radius, metric);
    bool valid = mean > 0 && radius >= 0 && std::isfinite(mean + radius);
    for (int d = 0; d < 3; ++d)
    {
        valid = valid && std::isfinite(velocity(cell, d)) && std::isfinite(beta[d]);
    }
    if (!valid) { return false; }
    // An outward L1 bound dominates the Euclidean speed without squaring a
    // tiny component or assuming a libm hypot error bound. Extended range
    // (checked by the entry) covers all products of represented double inputs.
    // Audit the represented frame too: that is what the unchanged kernel uses.
    long double const c = PhysConst::c;
    auto const radial = ErrorUp(std::abs(static_cast<long double>(velocity(cell, 0))) / c);
    auto const denominator = std::nextafter(mean * c, 0.L);
    auto const theta = ErrorUp(ErrorUp(
        std::abs(static_cast<long double>(velocity(cell, 1))) * radius) / denominator);
    auto const axial = ErrorUp(std::abs(static_cast<long double>(velocity(cell, 2))) / c);
    auto const physical_speed = ErrorUp(ErrorUp(radial + theta) + axial);
    auto const represented_speed = ErrorUp(ErrorUp(std::abs(static_cast<long double>(beta[0])) +
        std::abs(static_cast<long double>(beta[1]))) + std::abs(static_cast<long double>(beta[2])));
    auto const speed = std::max(physical_speed, represented_speed);
    return std::isfinite(speed) && speed <= maximum_beta &&
        ErrorUp(ErrorUp(speed * r) * width) <= .01L &&
        ErrorUp(ErrorUp(speed * p) * width) <= .01L;
}
#endif
#endif
} // namespace

bool AdmissibleResolvedRZThermalDiffusionCensus (
    amrex::MultiFab const& energy, amrex::MultiFab const& momentum,
    amrex::MultiFab const& proper_bath, amrex::MultiFab const& transport,
    amrex::MultiFab const& absorption, amrex::iMultiFab const& mask,
    amrex::MultiFab const& work_velocity, amrex::Geometry const& geometry, amrex::Real dt,
    ResolvedRZThermalDiffusionFieldOptions const& options)
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(energy, momentum, proper_bath, transport, absorption, mask,
                         work_velocity, geometry, dt, options);
    return false;
#else
    auto const& boxes = energy.boxArray();
    auto const& mapping = energy.DistributionMap();
    bool valid =
        SupportedNativeThermalContext(geometry, boxes, mapping) &&
        SupportedThermalControls(dt, options) && energy.nComp() == 1 &&
        momentum.nComp() == 3 && proper_bath.nComp() == 1 &&
        transport.nComp() == 1 && absorption.nComp() == 1 &&
        mask.nComp() == 1 && work_velocity.nComp() == 3 &&
        energy.ixType().cellCentered() && mask.boxArray() == boxes &&
        mask.DistributionMap() == mapping && mask.ixType().cellCentered() &&
        boxes.isDisjoint() && boxes.contains(geometry.Domain()) &&
        boxes.numPts() == geometry.Domain().numPts() &&
        geometry.ProbLo(0) == 0 && !geometry.isPeriodic(0) &&
        geometry.isPeriodic(1) && !options.reflecting_axial_table_trial && std::isfinite(dt) && dt > 0 &&
        dt * PhysConst::c <=
            .5 * std::min(geometry.CellSize(0), geometry.CellSize(1)) &&
        std::isfinite(options.minimum_optical_depth + options.maximum_beta) &&
        options.minimum_optical_depth >= 8 && options.maximum_beta > 0 &&
        options.maximum_beta <= .001;
    for (auto const* field : {&momentum, &proper_bath, &transport, &absorption, &work_velocity})
    {
        valid = valid && field->boxArray() == boxes && field->DistributionMap() == mapping &&
            field->ixType().cellCentered();
    }
    if (!Collective(valid)) { return false; }
    if (!SameThermalControls(dt, options))
    {
        return false;
    }
    auto old = Host(energy, geometry), q = Host(momentum, geometry);
    auto bath = Host(proper_bath, geometry), chi = Host(transport, geometry);
    auto absorb = Host(absorption, geometry), velocity = Host(work_velocity, geometry);
    auto flags = Host(mask, geometry);
    GroupMomentMetric const metric(geometry);
    auto const volume = metric.Volume(geometry.Domain().bigEnd(0));
    auto const width = std::max(geometry.ProbHi(0) - geometry.ProbLo(0),
                                geometry.ProbHi(1) - geometry.ProbLo(1));
    if (!Collective(volume > 0 && std::isfinite(volume + width)))
    {
        return false;
    }
    for (amrex::MFIter it(old); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            auto const flag = flags[it](index);
            auto const beta = PhysicalBeta(velocity[it], index,
                                           metric.Radius(index[0] + 1), metric);
            auto const speed = std::sqrt(beta[0] * beta[0] + beta[1] * beta[1] +
                                         beta[2] * beta[2]);
            valid = valid && (flag == 0 || flag == 1) &&
                    std::isfinite(old[it](index) + q[it](index, 0) +
                                  q[it](index, 1) + q[it](index, 2) +
                                  bath[it](index) + chi[it](index) +
                                  absorb[it](index) + speed) &&
                    old[it](index) >= 0 && bath[it](index) >= 0 &&
                    chi[it](index) >= 0 && absorb[it](index) >= 0 &&
                    absorb[it](index) <= chi[it](index) &&
                    speed <= options.maximum_beta;
            if (flag == 0)
            {
                valid = valid && old[it](index) == 0 && q[it](index, 0) == 0 &&
                    q[it](index, 1) == 0 && q[it](index, 2) == 0;
                continue;
            }
            auto const check = PreflightDiffusionCell(index, old[it](index), bath[it](index),
                chi[it](index), absorb[it](index), velocity[it], metric, geometry, dt, options);
            valid = valid && check.valid && PhysConst::c * dt * chi[it](index) >= 8 &&
                NecessaryCausalMoment(old[it](index),
                    {q[it](index, 0), q[it](index, 1), q[it](index, 2)}, index[0], metric);
        }
    }
    return Collective(valid);
#endif
}

bool
TryBuildResolvedRZThermalBathUpperBound (amrex::MultiFab const& temperature,
                                         amrex::MultiFab& bound,
                                         amrex::Geometry const& geometry,
                                         amrex::Real radiation_constant)
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(temperature, bound, geometry, radiation_constant);
    return false;
#else
    bool valid =
        bound.nComp() == 1 && bound.ixType().cellCentered() &&
        temperature.nComp() == 1 && temperature.ixType().nodeCentered() &&
        amrex::convert(temperature.boxArray(),
                       amrex::IntVect::TheZeroVector()) == bound.boxArray() &&
        temperature.DistributionMap() == bound.DistributionMap() &&
        SupportedNativeThermalContext(geometry, bound.boxArray(),
                                      bound.DistributionMap()) &&
        std::isfinite(radiation_constant) &&
        radiation_constant >= std::numeric_limits<amrex::Real>::min();
    if (!Collective(valid))
    {
        return false;
    }
    amrex::Real minimum = radiation_constant, maximum = minimum;
    amrex::ParallelDescriptor::ReduceRealMin(minimum);
    amrex::ParallelDescriptor::ReduceRealMax(maximum);
    if (!Collective(minimum == maximum))
    {
        return false;
    }
    auto const nodes = Host(temperature, geometry);
    auto candidate = HostZero(bound.boxArray(), bound.DistributionMap(), 1);
    auto const epsilon = std::numeric_limits<amrex::Real>::epsilon();
    auto const collar = 16 * epsilon / (1 - 16 * epsilon);
    for (amrex::MFIter it(candidate); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            amrex::Real maximum_bath = 0;
            for (int corner = 0; corner < 4; ++corner)
            {
                auto node = cell();
                node[0] += corner & 1;
                node[1] += (corner >> 1) & 1;
                auto const t = nodes[it](node), t2 = t * t;
                auto const value = radiation_constant * t2 * t2;
                // A relative gamma bound does not cover under-resolved
                // subnormal products or four-term weighted sums. Reject
                // nonzero subnormal baths rather than alter the physical RHS.
                valid = valid && std::isfinite(t + value) && t >= 0 &&
                        ((t == 0 && value == 0) ||
                         value >= std::numeric_limits<amrex::Real>::min());
                maximum_bath = std::max(maximum_bath, value);
            }
            candidate[it](cell()) = maximum_bath * (1 + collar);
            valid = valid && std::isfinite(candidate[it](cell()));
        }
    }
    if (!Collective(valid))
    {
        return false;
    }
    auto device = Device(candidate);
    amrex::MultiFab::Copy(bound, device, 0, 0, 1, 0);
    return true;
#endif
}

bool
TryConstrainResolvedRZThermalDiffusionMask (
    amrex::MultiFab const& energy, amrex::MultiFab const& momentum,
    amrex::MultiFab const& proper_bath, amrex::MultiFab const& transport,
    amrex::MultiFab const& absorption, amrex::iMultiFab const& old_mask,
    amrex::iMultiFab& proposed, amrex::MultiFab const& rate,
    amrex::Geometry const& geometry, amrex::Real dt,
    ResolvedRZThermalDiffusionFieldOptions const& options)
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(energy, momentum, proper_bath, transport, absorption,
                         old_mask, proposed, rate, geometry, dt, options);
    return false;
#else
    auto const& boxes = energy.boxArray();
    auto const& mapping = energy.DistributionMap();
    bool valid = SupportedNativeThermalContext(geometry, boxes, mapping) &&
                 !options.reflecting_axial_table_trial &&
                 SupportedThermalControls(dt, options) && energy.nComp() == 1 &&
                 energy.ixType().cellCentered() && momentum.nComp() == 3 &&
                 proper_bath.nComp() == 1 && transport.nComp() == 1 &&
                 absorption.nComp() == 1 && rate.nComp() == 3 &&
                 old_mask.nComp() == 1 && proposed.nComp() == 1;
    for (auto const* field :
         {&momentum, &proper_bath, &transport, &absorption, &rate})
    {
        valid = valid && field->boxArray() == boxes &&
                field->DistributionMap() == mapping &&
                field->ixType().cellCentered();
    }
    for (auto const* flags : {&old_mask, static_cast<amrex::iMultiFab const*>(&proposed)})
    {
        valid = valid && flags->boxArray() == boxes &&
                flags->DistributionMap() == mapping &&
                flags->ixType().cellCentered();
    }
    if (!Collective(valid))
    {
        return false;
    }
    if (!SameThermalControls(dt, options))
    {
        return false;
    }
    auto e = Host(energy, geometry), q = Host(momentum, geometry);
    auto bath = Host(proper_bath, geometry), chi = Host(transport, geometry);
    auto absorb = Host(absorption, geometry), velocity = Host(rate, geometry);
    auto old = Host(old_mask, geometry), next = Host(proposed, geometry);
    GroupMomentMetric const metric(geometry);
    auto const volume = metric.Volume(geometry.Domain().bigEnd(0));
    auto const width = std::max(geometry.ProbHi(0) - geometry.ProbLo(0),
                                geometry.ProbHi(1) - geometry.ProbLo(1));
    if (!Collective(volume > 0 && std::isfinite(volume + width)))
    {
        return false;
    }
    for (amrex::MFIter it(e); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            valid = valid && (old[it](index) == 0 || old[it](index) == 1) &&
                    (next[it](index) == 0 || next[it](index) == 1) &&
                    std::isfinite(
                        e[it](index) + q[it](index, 0) + q[it](index, 1) +
                        q[it](index, 2) + bath[it](index) + chi[it](index) +
                        absorb[it](index) + velocity[it](index, 0) +
                        velocity[it](index, 1) + velocity[it](index, 2)) &&
                    e[it](index) >= 0 && bath[it](index) >= 0 &&
                    chi[it](index) >= 0 && absorb[it](index) >= 0 &&
                    absorb[it](index) <= chi[it](index);
            if (old[it](index) == 0)
            {
                valid = valid && e[it](index) == 0 && q[it](index, 0) == 0 &&
                        q[it](index, 1) == 0 && q[it](index, 2) == 0;
            }
            if (old[it](index) != 1 || next[it](index) != 1)
            {
                continue;
            }
            auto const check = PreflightDiffusionCell(
                index, e[it](index), bath[it](index), chi[it](index),
                absorb[it](index), velocity[it], metric, geometry, dt, options);
            bool const admissible =
                check.valid && PhysConst::c * dt * chi[it](index) >= 8 &&
                NecessaryCausalMoment(
                    e[it](index),
                    {q[it](index, 0), q[it](index, 1), q[it](index, 2)},
                    index[0], metric);
            if (!admissible)
            {
                next[it](index) = 0;
            }
        }
    }
    if (!Collective(valid))
    {
        return false;
    }
    amrex::iMultiFab device(boxes, mapping, 1, next.nGrowVect());
    amrex::htod_memcpy(device, next);
    amrex::iMultiFab::Copy(proposed, device, 0, 0, 1, 0);
    return true;
#endif
}

static ResolvedRZThermalDiffusionFieldResult
TryAdvanceThermalFieldImplementation (
    amrex::MultiFab const& old_lab_energy, amrex::MultiFab const& old_lab_momentum,
    amrex::MultiFab const& proper_bath, amrex::MultiFab const& proper_transport_opacity,
    amrex::MultiFab const& proper_absorption_opacity, amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& work_velocity, amrex::MultiFab const& actual_work,
    amrex::Geometry const& geometry, amrex::Real dt,
    ResolvedRZThermalDiffusionFieldOptions const& options, bool independent_rest)
{
    ResolvedRZThermalDiffusionFieldResult rejected;
    rejected.reason = "unsupported private RZ thermal diffusion trial";
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(old_lab_energy, old_lab_momentum, proper_bath,
        proper_transport_opacity, proper_absorption_opacity, diffusion_mask,
        work_velocity, actual_work, geometry, dt, options, independent_rest);
    return rejected;
#else
    auto const& boxes = old_lab_energy.boxArray();
    auto const& mapping = old_lab_energy.DistributionMap();
    int selected_min = options.single_selected_table_trial, selected_max = selected_min;
    amrex::ParallelDescriptor::ReduceIntMin(selected_min);
    amrex::ParallelDescriptor::ReduceIntMax(selected_max);
    if (selected_min != selected_max) { return rejected; }
    bool valid =
        SupportedNativeThermalContext(geometry, boxes, mapping,
                                      options.reflecting_axial_field_trial,
                                      options.reflecting_axial_table_trial,
                                      options.single_selected_table_trial,
                                      options.single_selected_table_vacuum) &&
        SupportedThermalControls(dt, options) && old_lab_energy.nComp() == 1 &&
        old_lab_energy.ixType().cellCentered() &&
        old_lab_momentum.nComp() == 3 && proper_bath.nComp() == 1 &&
        proper_transport_opacity.nComp() == 1 &&
        proper_absorption_opacity.nComp() == 1 && diffusion_mask.nComp() == 1 &&
        work_velocity.nComp() == 3 && actual_work.nComp() == 1 &&
        (geometry.isPeriodic(1) || PhysConst::c * dt <=
            .5 * std::min(geometry.CellSize(0), geometry.CellSize(1)));
    for (auto const* field : {&old_lab_momentum, &proper_bath,
                              &proper_transport_opacity, &proper_absorption_opacity,
                              &work_velocity, &actual_work})
    {
        valid = valid && field->boxArray() == boxes && field->DistributionMap() == mapping &&
            field->ixType().cellCentered();
    }
    valid = valid && diffusion_mask.boxArray() == boxes &&
        diffusion_mask.DistributionMap() == mapping && diffusion_mask.ixType().cellCentered();
    if (!Collective(valid)) { return rejected; }
    if (!SameThermalControls(dt, options))
    {
        return rejected;
    }

    GroupMomentMetric const metric(geometry);
    auto const volume_scale = metric.Volume(geometry.Domain().bigEnd(0));
    auto const width = std::max(geometry.ProbHi(0) - geometry.ProbLo(0),
                                geometry.ProbHi(1) - geometry.ProbLo(1));
    valid = volume_scale > 0 && std::isfinite(volume_scale + width);
    if (!Collective(valid)) { return rejected; }
    auto old = Host(old_lab_energy, geometry);
    auto old_q = Host(old_lab_momentum, geometry);
    auto bath = Host(proper_bath, geometry);
    auto chi = Host(proper_transport_opacity, geometry);
    auto absorb = Host(proper_absorption_opacity, geometry);
    auto mask = Host(diffusion_mask, geometry);
    auto velocity = Host(work_velocity, geometry);
    auto work = Host(actual_work, geometry);
    auto guess = HostZero(boxes, mapping, 1, 1);
    auto m00 = HostZero(boxes, mapping, 1, 1);
    auto bath_coefficient = HostZero(boxes, mapping, 1);
    amrex::Real normalization = volume_scale;
    bool causal_old = true;
    for (amrex::MFIter it(old); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            auto const flag = mask[it](index);
            auto const total = chi[it](index), absorption = absorb[it](index);
            auto const energy = old[it](index), source = bath[it](index);
            auto const radius = metric.Radius(index[0] + 1);
            auto const beta = PhysicalBeta(velocity[it], index, radius, metric);
            auto const beta2 = beta[0] * beta[0] + beta[1] * beta[1] + beta[2] * beta[2];
            auto const speed = std::sqrt(beta2);
            valid = valid && (flag == 0 || flag == 1) &&
                std::isfinite(energy + source + total + absorption + work[it](index) +
                              beta2 + old_q[it](index, 0) + old_q[it](index, 1) +
                              old_q[it](index, 2)) && energy >= 0 && source >= 0 &&
                total >= 0 && absorption >= 0 && (absorption <= total || independent_rest) &&
                speed <= options.maximum_beta;
            if (flag == 1)
            {
                causal_old = NecessaryCausalMoment(energy,
                    {old_q[it](index, 0), old_q[it](index, 1), old_q[it](index, 2)},
                    index[0], metric) && causal_old;
            }
            if (!valid) { continue; }
            if (flag == 0)
            {
                valid = valid && !options.centered_thermal_difference &&
                    energy == 0 && total == 0 && absorption == 0 &&
                    work[it](index) == 0 && old_q[it](index, 0) == 0 &&
                    old_q[it](index, 1) == 0 && old_q[it](index, 2) == 0;
                continue;
            }
            auto const check = PreflightDiffusionCell(index, energy, source, total, absorption,
                velocity[it], metric, geometry, dt, options, independent_rest);
            valid = valid && check.valid;
            if (!valid) { continue; }
            m00[it](index) = check.moments.m00;
            bath_coefficient[it](index) = metric.Volume(index[0]) * check.unit_bath.equilibrium_density;
            guess[it](index) = options.centered_thermal_difference
                ? check.proper_energy - source : check.proper_energy;
            normalization = std::max(normalization, check.normalization);
        }
    }
    amrex::ParallelDescriptor::ReduceRealMax(normalization);
    if (!Collective(causal_old))
    {
        rejected.reason = "noncausal old RZ LAB radiation moment";
        return rejected;
    }
    if (!Collective(valid))
    {
        amrex::MultiFab direct_work(actual_work.boxArray(), actual_work.DistributionMap(),
                                    1, actual_work.nGrowVect(),
                                    amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
        amrex::dtoh_memcpy(direct_work, actual_work);
        amrex::Gpu::streamSynchronize();
        std::array<amrex::Real, 5> maxima{};
        amrex::Real minimum_d_depth = std::numeric_limits<amrex::Real>::max();
        for (amrex::MFIter it(guess); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                auto const index = cell();
                auto const beta = PhysicalBeta(velocity[it], index,
                                                metric.Radius(index[0] + 1), metric);
                auto const speed = std::sqrt(beta[0] * beta[0] + beta[1] * beta[1] +
                                              beta[2] * beta[2]);
                maxima[0] = std::max(maxima[0], speed);
                if (mask[it](index) == 1)
                {
                    maxima[1] = std::max(maxima[1], speed * chi[it](index) * width);
                    minimum_d_depth = std::min(minimum_d_depth, chi[it](index) *
                        std::min(geometry.CellSize(0), geometry.CellSize(1)));
                }
                else
                {
                    maxima[2] = std::max(maxima[2], std::abs(work[it](index)));
                    maxima[4] = std::max(maxima[4], std::abs(direct_work[it](index)));
                    maxima[3] = std::max(maxima[3], std::abs(chi[it](index)) +
                        std::abs(absorb[it](index)) + std::abs(old[it](index)) +
                        std::abs(old_q[it](index, 0)) + std::abs(old_q[it](index, 1)) +
                        std::abs(old_q[it](index, 2)));
                }
            }
        }
        amrex::ParallelDescriptor::ReduceRealMax(maxima.data(), maxima.size());
        amrex::ParallelDescriptor::ReduceRealMin(minimum_d_depth);
        std::ostringstream message;
        message << "invalid RZ static-diffusion source, opacity, beta or outgoing CFL"
                << " (max_beta=" << maxima[0] << ", max_beta_chi_L=" << maxima[1]
                << ", min_D_chi_dx=" << minimum_d_depth << ", max_abs_K_work=" << maxima[2]
                << ", max_K_state=" << maxima[3] << ", max_abs_K_input_work="
                << maxima[4] << ')';
        rejected.reason = message.str();
        return rejected;
    }
    guess.FillBoundary(geometry.periodicity());
    m00.FillBoundary(geometry.periodicity());

    auto acoef = HostZero(boxes, mapping, 1);
    auto rhs = HostZero(boxes, mapping, 1);
    auto bcoef = FaceFields(boxes, mapping, 1);
    auto proper_face = FaceFields(boxes, mapping, 1);
    auto lab_face = FaceFields(boxes, mapping, 1);
    auto stress_face = FaceFields(boxes, mapping, 3);
    auto leak_face = FaceFields(boxes, mapping, 1);
    auto bath_proper_face = options.centered_thermal_difference
        ? FaceFields(boxes, mapping, 1) : Faces{};
    amrex::MultiFab physical_proper;
    if (options.centered_thermal_difference)
    {
        physical_proper = HostZero(boxes, mapping, 1, 1);
    }
    auto const reconstruct_proper = [&] (amrex::MultiFab const& deviation)
    {
        for (amrex::MFIter it(physical_proper); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                physical_proper[it](cell()) = bath[it](cell()) + deviation[it](cell());
            }
        }
        physical_proper.FillBoundary(geometry.periodicity());
    };
    if (options.centered_thermal_difference)
    {
        for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
        {
            for (amrex::MFIter it(*bath_proper_face[axis]); it.isValid(); ++it)
            {
                for (amrex::BoxIterator face(it.validbox()); face.ok(); ++face)
                {
                    auto const value = FaceMomentAt(face(), axis, bath[it], bath[it], chi[it],
                        absorb[it], velocity[it], mask[it], metric, geometry,
                        options.linear.marshak_extrapolation_lambda);
                    valid = valid && value.valid;
                    (*bath_proper_face[axis])[it](face()) = value.proper_current;
                }
            }
        }
        if (!Collective(valid))
        {
            rejected.reason = "invalid centered RZ thermal bath faces";
            return rejected;
        }
    }
    for (amrex::MFIter it(acoef); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            if (mask[it](index) == 0)
            {
                acoef[it](index) = 1;
                continue;
            }
            auto const volume = metric.Volume(index[0]);
            auto const rate = dt * PhysConst::c * absorb[it](index);
            amrex::Real diagonal = volume * m00[it](index) * (1 + rate);
            for (int d = 0; d < 2; ++d)
            {
                for (int side : {-1, 1})
                {
                    if (!geometry.isPeriodic(d) &&
                        ((side < 0 && index[d] == geometry.Domain().smallEnd(d)) ||
                         (side > 0 && index[d] == geometry.Domain().bigEnd(d))))
                    {
                        continue;
                    }
                    auto neighbor = index;
                    neighbor[d] += side;
                    if (mask[it](neighbor) != 0) { continue; }
                    auto face = index;
                    if (side > 0) { ++face[d]; }
                    auto const h = 2 * PhysConst::c /
                        (3 * (chi[it](index) * geometry.CellSize(d) +
                              2 * options.linear.marshak_extrapolation_lambda));
                    diagonal += dt * metric.Area(d, face[0]) * h;
                }
            }
            acoef[it](index) = diagonal / normalization;
            valid = valid && acoef[it](index) > 0 && std::isfinite(acoef[it](index));
        }
    }
    for (int d = 0; d < 2; ++d)
    {
        for (amrex::MFIter it(*bcoef[d]); it.isValid(); ++it)
        {
            for (amrex::BoxIterator face(it.validbox()); face.ok(); ++face)
            {
                auto const right = face();
                auto left = right;
                --left[d];
                if (ReflectingFace(right, d, geometry))
                {
                    continue;
                }
                if (mask[it](left) != 1 || mask[it](right) != 1) { continue; }
                auto const denominator = chi[it](left) + chi[it](right);
                auto const value = 2 * PhysConst::c * metric.Area(d, right[0]) *
                    geometry.CellSize(d) / (3 * denominator * normalization);
                (*bcoef[d])[it](right) = value;
                valid = valid && value > 0 && std::isfinite(value);
            }
        }
    }
    if (!Collective(valid))
    {
        rejected.reason = "invalid RZ native annular diffusion coefficients";
        return rejected;
    }

    auto device_a = Device(acoef);
    auto device_b0 = Device(*bcoef[0]);
    auto device_b1 = Device(*bcoef[1]);
    amrex::Array<amrex::MultiFab const*, 2> device_b{&device_b0, &device_b1};
    auto device_u = Device(guess);
    amrex::MultiFab device_rhs(boxes, mapping, 1, 0);
    amrex::Array<amrex::LinOpBCType, 2> boundary{
        amrex::LinOpBCType::Neumann, geometry.isPeriodic(1)
            ? amrex::LinOpBCType::Periodic : amrex::LinOpBCType::Neumann};
    std::array<int, 2> periodic{0, geometry.isPeriodic(1) ? 1 : 0};
    auto const physical_box = geometry.ProbDomain();
    amrex::Geometry const cartesian(geometry.Domain(), &physical_box, 0, periodic.data());
    amrex::MLABecLaplacian op({cartesian}, {boxes}, {mapping});
    op.setMaxOrder(2);
    op.setDomainBC(boundary, boundary);
    op.setLevelBC(0, nullptr);
    op.setScalars(1, dt);
    op.setACoeffs(0, device_a);
    op.setBCoeffs(0, device_b);
    amrex::MLMG solver(op);
    solver.setVerbose(options.linear.verbosity);
    solver.setMaxIter(options.linear.max_linear_iterations);
    solver.setThrowException(true);

    ResolvedRZThermalDiffusionFieldResult result;
    bool converged = false;
    amrex::Real last_defect_ratio = 0;
    auto solved = HostZero(boxes, mapping, 1, 1);
    for (int iteration = 0; iteration < options.max_picard_iterations; ++iteration)
    {
        // A rejected Picard iterate moves the already checked `solved` into `guess`.
        // Its face moments remain in these four fields; only the first iterate
        // needs an entry-side face construction.
        if (iteration == 0 && options.centered_thermal_difference) { reconstruct_proper(guess); }
        if (iteration == 0 &&
            !BuildFaces(options.centered_thermal_difference ? physical_proper : guess,
                        bath, chi, absorb, velocity, mask, metric, geometry,
                        options.linear.marshak_extrapolation_lambda,
                        proper_face, lab_face, stress_face, leak_face,
                        options.centered_thermal_difference ? &guess : nullptr))
        {
            rejected.reason = "invalid RZ boosted face current during Picard iteration";
            return rejected;
        }
        for (amrex::MFIter it(rhs); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                auto const index = cell();
                if (mask[it](index) == 0) { rhs[it](index) = 0; continue; }
                auto upper_r = index, upper_z = index;
                ++upper_r[0];
                ++upper_z[1];
                auto const fr = ((*proper_face[0])[it](index) +
                                 (*proper_face[0])[it](upper_r)) / (2 * PhysConst::c);
                auto const fz = ((*proper_face[1])[it](index) +
                                 (*proper_face[1])[it](upper_z)) / (2 * PhysConst::c);
                auto const ratio = absorb[it](index) / chi[it](index);
                auto const moments = CellMoment(index[0], index, velocity[it], metric,
                    options.centered_thermal_difference ? bath[it](index) + guess[it](index)
                                                       : guess[it](index),
                    bath[it](index), fr, fz, ratio,
                    options.centered_thermal_difference, guess[it](index));
                auto const cross = options.centered_thermal_difference
                    ? moments.transport_offset_density : moments.offset_density;
                amrex::Real divergence = 0, bath_transport = 0;
                for (int d = 0; d < 2; ++d)
                {
                    auto upper = index;
                    ++upper[d];
                    auto const low_h = (*lab_face[d])[it](index) -
                                       (*proper_face[d])[it](index);
                    auto const high_h = (*lab_face[d])[it](upper) -
                                        (*proper_face[d])[it](upper);
                    auto const area_lo = metric.Area(d, index[0]);
                    auto const area_hi = metric.Area(d, upper[0]);
                    divergence += d == 0 ? area_hi * high_h - area_lo * low_h
                                         : area_lo * (high_h - low_h);
                    if (options.centered_thermal_difference)
                    {
                        bath_transport += area_hi * (*bath_proper_face[d])[it](upper) -
                                          area_lo * (*bath_proper_face[d])[it](index);
                    }
                }
                auto const volume = metric.Volume(index[0]);
                auto const rate = dt * PhysConst::c * absorb[it](index);
                auto const equilibrium = bath_coefficient[it](index) * bath[it](index);
                auto const bath_rhs = options.centered_thermal_difference
                    ? -equilibrium
                    : rate * bath_coefficient[it](index) * bath[it](index);
                auto const advective_transport = options.centered_thermal_difference
                    ? dt * (divergence + bath_transport) : dt * divergence;
                rhs[it](index) = (old[it](index) - (1 - 2 * ratio) * work[it](index) +
                    bath_rhs - (1 + rate) * volume * cross - advective_transport) /
                    normalization;
                valid = valid && std::isfinite(rhs[it](index));
            }
        }
        if (!Collective(valid))
        {
            rejected.reason = "invalid RZ thermal diffusion Picard RHS";
            return rejected;
        }
        // The layout is fixed for the whole Picard solve. Refresh the RHS in
        // place instead of allocating a device MultiFab on every iteration.
        amrex::htod_memcpy(device_rhs, rhs);
        amrex::Gpu::streamSynchronize();
        try
        {
            result.linear_residual = solver.solve(
                {&device_u}, {&device_rhs}, options.linear.linear_tolerance, 0);
            result.linear_iterations += solver.getNumIters();
        }
        catch (amrex::MLMG::error const&)
        {
            rejected.reason = "RZ thermal annular MLMG convergence";
            return rejected;
        }
        // Copy valid cells into the reusable pinned buffer. Its only needed
        // ghost fill is after the inactive cells below have been reset to zero.
        solved.setVal(0);
        amrex::MultiFab::Copy(solved, device_u, 0, 0, 1, 0);
        amrex::Gpu::streamSynchronize();
        for (amrex::MFIter it(solved); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                auto const index = cell();
                if (mask[it](index) == 0) { solved[it](index) = 0; }
                auto const physical = options.centered_thermal_difference
                    ? bath[it](index) + solved[it](index) : solved[it](index);
                valid = valid && std::isfinite(physical) && physical >= 0;
            }
        }
        if (!Collective(valid))
        {
            rejected.reason = "negative or nonfinite private RZ proper radiation energy";
            return rejected;
        }
        solved.FillBoundary(geometry.periodicity());
        if (options.centered_thermal_difference) { reconstruct_proper(solved); }
        if (!BuildFaces(options.centered_thermal_difference ? physical_proper : solved,
                        bath, chi, absorb, velocity, mask, metric, geometry,
                        options.linear.marshak_extrapolation_lambda,
                        proper_face, lab_face, stress_face, leak_face,
                        options.centered_thermal_difference ? &solved : nullptr))
        {
            rejected.reason = "invalid accepted RZ boosted face current";
            return rejected;
        }
        amrex::Real maximum_defect = 0;
        last_defect_ratio = 0;
        bool iteration_valid = true;
        for (amrex::MFIter it(solved); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                auto const index = cell();
                if (mask[it](index) == 0) { continue; }
                auto upper_r = index, upper_z = index;
                ++upper_r[0];
                ++upper_z[1];
                auto const fr = ((*proper_face[0])[it](index) +
                                 (*proper_face[0])[it](upper_r)) / (2 * PhysConst::c);
                auto const fz = ((*proper_face[1])[it](index) +
                                 (*proper_face[1])[it](upper_z)) / (2 * PhysConst::c);
                auto const ratio = absorb[it](index) / chi[it](index);
                auto const moments = CellMoment(index[0], index, velocity[it], metric,
                    options.centered_thermal_difference ? bath[it](index) + solved[it](index)
                                                       : solved[it](index),
                    bath[it](index), fr, fz, ratio,
                    options.centered_thermal_difference, solved[it](index));
                auto const volume = metric.Volume(index[0]);
                auto const equilibrium = bath_coefficient[it](index) * bath[it](index);
                auto const energy = options.centered_thermal_difference
                    ? equilibrium + volume * moments.nonequilibrium_density
                    : volume * moments.energy_density;
                auto const heat = dt * PhysConst::c * absorb[it](index) *
                    (options.centered_thermal_difference
                         ? volume * moments.nonequilibrium_density : energy - equilibrium) -
                    2 * ratio * work[it](index);
                amrex::Real transport = 0, operands = 0;
                for (int d = 0; d < 2; ++d)
                {
                    auto upper = index;
                    ++upper[d];
                    auto const low = dt * metric.Area(d, index[0]) *
                                     (*lab_face[d])[it](index);
                    auto const high = dt * metric.Area(d, upper[0]) *
                                      (*lab_face[d])[it](upper);
                    transport += high - low;
                    operands += std::abs(high) + std::abs(low);
                }
                auto const residual = (energy - old[it](index)) + transport +
                                      work[it](index) + heat;
                auto const scale = std::abs(energy) + std::abs(old[it](index)) +
                    std::abs(work[it](index)) + std::abs(heat) + operands +
                    volume * (std::abs(moments.m00 * solved[it](index)) +
                              std::abs(moments.equilibrium_density));
                auto const bound = (options.linear.residual_tolerance +
                    256 * std::numeric_limits<amrex::Real>::epsilon()) * scale +
                    2 * std::abs(result.linear_residual) * normalization;
                maximum_defect = std::max(maximum_defect, std::abs(residual));
                last_defect_ratio = std::max(last_defect_ratio,
                    bound > 0 ? std::abs(residual) / bound :
                    residual == 0 ? 0 : std::numeric_limits<amrex::Real>::infinity());
                iteration_valid = iteration_valid && std::isfinite(residual + bound) &&
                    energy >= 0 && std::abs(residual) <= bound;
            }
        }
        amrex::ParallelDescriptor::ReduceRealMax(maximum_defect);
        result.equation_residual = maximum_defect;
        result.picard_iterations = iteration + 1;
        if (Collective(iteration_valid)) { converged = true; break; }
        std::swap(guess, solved);
    }
    if (!converged)
    {
        amrex::ParallelDescriptor::ReduceRealMax(last_defect_ratio);
        std::ostringstream message;
        message << "RZ boosted spatial/thermal Picard equation did not converge"
                << " (iterations=" << result.picard_iterations
                << ", max_equation_defect=" << result.equation_residual
                << ", max_defect_over_bound=" << last_defect_ratio
                << ", normalized_linear_residual=" << result.linear_residual << ')';
        rejected.reason = message.str();
        return rejected;
    }
    auto energy_host = HostZero(boxes, mapping, 1);
    auto q_host = HostZero(boxes, mapping, 3);
    auto debit_host = HostZero(boxes, mapping, 3);
    auto heat_host = HostZero(boxes, mapping, 1);
    auto physical_host = HostZero(boxes, mapping, 1);
    auto raw_host = HostZero(boxes, mapping, 1);
    auto bound_host = HostZero(boxes, mapping, 1);
    auto scale_host = HostZero(boxes, mapping, 1);
    auto stress_host = HostZero(boxes, mapping, 3);
    auto connection_host = HostZero(boxes, mapping, 1);
    bool causal_accepted = true;
    for (amrex::MFIter it(energy_host); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            if (mask[it](index) == 0) { continue; }
            auto upper_r = index, upper_z = index;
            ++upper_r[0];
            ++upper_z[1];
            auto const fr = ((*proper_face[0])[it](index) +
                             (*proper_face[0])[it](upper_r)) / (2 * PhysConst::c);
            auto const fz = ((*proper_face[1])[it](index) +
                             (*proper_face[1])[it](upper_z)) / (2 * PhysConst::c);
            auto const ratio = absorb[it](index) / chi[it](index);
            auto const moments = CellMoment(index[0], index, velocity[it], metric,
                options.centered_thermal_difference ? bath[it](index) + solved[it](index)
                                                   : solved[it](index),
                bath[it](index), fr, fz, ratio,
                options.centered_thermal_difference, solved[it](index));
            auto const volume = metric.Volume(index[0]);
            auto const equilibrium = bath_coefficient[it](index) * bath[it](index);
            auto const energy = options.centered_thermal_difference
                ? equilibrium + volume * moments.nonequilibrium_density
                : volume * moments.energy_density;
            auto const heat = dt * PhysConst::c * absorb[it](index) *
                (options.centered_thermal_difference
                     ? volume * moments.nonequilibrium_density : energy - equilibrium) -
                2 * ratio * work[it](index);
            energy_host[it](index) = energy;
            heat_host[it](index) = heat;
            physical_host[it](index) = work[it](index) + heat;
            q_host[it](index, 0) = volume *
                ((*lab_face[0])[it](index) + (*lab_face[0])[it](upper_r)) /
                (2 * PhysConst::c);
            q_host[it](index, 1) = volume * moments.theta_conjugate_density;
            q_host[it](index, 2) = volume *
                ((*lab_face[1])[it](index) + (*lab_face[1])[it](upper_z)) /
                (2 * PhysConst::c);
            causal_accepted = NecessaryCausalMoment(energy,
                {q_host[it](index, 0), q_host[it](index, 1), q_host[it](index, 2)},
                index[0], metric) && causal_accepted;

            auto const ar_lo = metric.Area(0, index[0]);
            auto const ar_hi = metric.Area(0, upper_r[0]);
            auto const az = metric.Area(1, index[0]);
            auto const low_r = dt * ar_lo * (*lab_face[0])[it](index);
            auto const high_r = dt * ar_hi * (*lab_face[0])[it](upper_r);
            auto const low_z = dt * az * (*lab_face[1])[it](index);
            auto const high_z = dt * az * (*lab_face[1])[it](upper_z);
            auto const transfer = (high_r - low_r) + (high_z - low_z);
            raw_host[it](index) = (old[it](index) - energy) - transfer;
            auto const hoop = PhysConst::c * moments.theta_pressure;
            stress_host[it](index, 0) = dt *
                (ar_hi * ((*stress_face[0])[it](upper_r, 0) - hoop) -
                 ar_lo * ((*stress_face[0])[it](index, 0) - hoop)) +
                dt * az * ((*stress_face[1])[it](upper_z, 0) -
                           (*stress_face[1])[it](index, 0));
            stress_host[it](index, 1) = metric.Angular(index[0]).Divergence(
                (*stress_face[0])[it](index, 1),
                (*stress_face[0])[it](upper_r, 1), dt) +
                dt * az * ((*stress_face[1])[it](upper_z, 1) -
                           (*stress_face[1])[it](index, 1));
            connection_host[it](index) = metric.Angular(index[0]).Geometric(
                (*stress_face[0])[it](index, 1),
                (*stress_face[0])[it](upper_r, 1), dt);
            stress_host[it](index, 2) = dt *
                (ar_hi * (*stress_face[0])[it](upper_r, 2) -
                 ar_lo * (*stress_face[0])[it](index, 2)) +
                dt * az * ((*stress_face[1])[it](upper_z, 2) -
                           (*stress_face[1])[it](index, 2));
            for (int d = 0; d < 3; ++d)
            {
                debit_host[it](index, d) = old_q[it](index, d) -
                    q_host[it](index, d) - stress_host[it](index, d);
            }
            auto const operands = std::abs(low_r) + std::abs(high_r) +
                                  std::abs(low_z) + std::abs(high_z);
            auto const scale = std::abs(old[it](index)) + std::abs(energy) +
                operands + std::abs(work[it](index)) + std::abs(heat) +
                std::abs(equilibrium) + volume *
                    (std::abs(moments.m00 * solved[it](index)) +
                     std::abs(moments.offset_density));
            auto const bound = 256 * std::numeric_limits<amrex::Real>::epsilon() * scale +
                               2 * std::abs(result.linear_residual) * normalization;
            scale_host[it](index) = scale;
            bound_host[it](index) = bound;
            auto const equation = (energy - old[it](index)) + transfer +
                                  physical_host[it](index);
            auto const allowance = options.linear.residual_tolerance * scale + bound;
            valid = valid && std::isfinite(energy + heat + transfer + scale + bound +
                                            q_host[it](index, 0) + q_host[it](index, 1) +
                                            q_host[it](index, 2) +
                                            debit_host[it](index, 0) +
                                            debit_host[it](index, 1) +
                                            debit_host[it](index, 2)) &&
                    energy >= 0 && std::abs(equation) <= allowance &&
                    std::abs(raw_host[it](index) - physical_host[it](index)) <= allowance;
        }
    }
    if (!Collective(causal_accepted))
    {
        rejected.reason = "noncausal accepted RZ LAB radiation moment";
        return rejected;
    }
    for (int d = 0; d < 2; ++d)
    {
        for (amrex::MFIter it(*leak_face[d]); it.isValid(); ++it)
        {
            for (amrex::BoxIterator face(it.validbox()); face.ok(); ++face)
            {
                auto const index = face();
                auto left = index;
                --left[d];
                if (ReflectingFace(index, d, geometry))
                {
                    (*leak_face[d])[it](index) = 0;
                    continue;
                }
                auto const normal = mask[it](left) - mask[it](index);
                auto const accepted = dt * metric.Area(d, index[0]) * normal *
                    (*lab_face[d])[it](index);
                (*leak_face[d])[it](index) = normal == 0 ? 0 : accepted;
                valid = valid && std::isfinite(accepted) &&
                    (normal == 0 || accepted >= 0);
            }
        }
    }
    if (!Collective(valid))
    {
        rejected.reason = "RZ thermal field debit, angular stress, or D/K leakage rejected";
        return rejected;
    }
    if (!geometry.isPeriodic(1))
    {
        // Count each endface from its disjoint boundary-adjacent cell, not
        // duplicated nodal face owners. Stress stores c*Pzz [W/m^2].
        for (amrex::MFIter it(mask); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                auto const index = cell();
                if (mask[it](index) == 0) { continue; }
                for (int side = 0; side < 2; ++side)
                {
                    if (index[1] != (side == 0 ? geometry.Domain().smallEnd(1)
                                              : geometry.Domain().bigEnd(1)))
                    {
                        continue;
                    }
                    auto face = index;
                    face[1] += side;
                    auto const impulse = (side == 0 ? -1 : 1) * dt *
                        metric.Area(1, index[0]) * (*stress_face[1])[it](face, 2) /
                        PhysConst::c;
                    result.axial_wall_momentum_by_side[side] += impulse;
                    valid = valid && std::isfinite(impulse) &&
                        (*lab_face[1])[it](face) == 0 &&
                        (*proper_face[1])[it](face) == 0 &&
                        (*leak_face[1])[it](face) == 0 &&
                        (*stress_face[1])[it](face, 0) == 0 &&
                        (*stress_face[1])[it](face, 1) == 0;
                }
            }
        }
        amrex::ParallelDescriptor::ReduceRealSum(
            result.axial_wall_momentum_by_side.data(), 2);
        result.axial_wall_momentum = result.axial_wall_momentum_by_side[0] +
                                     result.axial_wall_momentum_by_side[1];
        valid = valid && std::isfinite(result.axial_wall_momentum);
        if (!Collective(valid))
        {
            rejected.reason = "RZ reflecting axial field wall account rejected";
            return rejected;
        }
    }
    result.energy = Device(energy_host);
    result.lab_momentum = Device(q_host);
    result.momentum_debit = Device(debit_host);
    result.requested_heat = Device(heat_host);
    result.proper_bath_energy_coefficient = Device(bath_coefficient);
    result.physical_energy_debit = Device(physical_host);
    result.raw_energy_debit = Device(raw_host);
    result.storage_roundoff_bound = Device(bound_host);
    result.operand_scale = Device(scale_host);
    result.outward_pressure_stress = Device(stress_host);
    result.theta_connection = Device(connection_host);
    for (int d = 0; d < 2; ++d)
    {
        result.face_lab_energy_current[d] =
            std::make_unique<amrex::MultiFab>(Device(*lab_face[d]));
        result.face_lab_stress[d] =
            std::make_unique<amrex::MultiFab>(Device(*stress_face[d]));
        result.face_leaked_energy[d] =
            std::make_unique<amrex::MultiFab>(Device(*leak_face[d]));
    }
    result.valid = true;
    return result;
#endif
}

ResolvedRZThermalDiffusionFieldResult
TryAdvanceResolvedRZThermalDiffusionField (
    amrex::MultiFab const& old_lab_energy, amrex::MultiFab const& old_lab_momentum,
    amrex::MultiFab const& proper_bath, amrex::MultiFab const& proper_transport_opacity,
    amrex::MultiFab const& proper_absorption_opacity, amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& work_velocity, amrex::MultiFab const& actual_work,
    amrex::Geometry const& geometry, amrex::Real dt,
    ResolvedRZThermalDiffusionFieldOptions const& options)
{
    return TryAdvanceThermalFieldImplementation(old_lab_energy, old_lab_momentum,
        proper_bath, proper_transport_opacity, proper_absorption_opacity, diffusion_mask,
        work_velocity, actual_work, geometry, dt, options, false);
}

ResolvedRZThermalDiffusionFieldResult
TryAdvanceIndependentGrayRestAllDField (
    amrex::MultiFab const& old_lab_energy, amrex::MultiFab const& old_lab_momentum,
    amrex::MultiFab const& proper_bath, amrex::MultiFab const& raw_charge_density,
    amrex::MultiFab const& nodal_temperature,
    detail::ResolvedMaterialChargeFields const& unit_ion_charge,
    MaterialOpacityTable const& table, amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& work_velocity, amrex::MultiFab const& actual_work,
    amrex::Geometry const& geometry, amrex::Real dt,
    IndependentGrayRestAllDFieldOptions const& options)
{
    ResolvedRZThermalDiffusionFieldResult rejected;
    rejected.reason = "unsupported independent-gray rest ALL-D FIELD trial";
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU) || \
    !defined(WARPX_USE_SINGULARITY_EOS) || !defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
    amrex::ignore_unused(old_lab_energy, old_lab_momentum, proper_bath, raw_charge_density,
        nodal_temperature, unit_ion_charge, table, diffusion_mask, work_velocity,
        actual_work, geometry, dt, options);
    return rejected;
#else
    int low = options.enabled, high = low;
    amrex::ParallelDescriptor::ReduceIntMin(low);
    amrex::ParallelDescriptor::ReduceIntMax(high);
    if (low != high || low == 0) { return rejected; }
    auto const& boxes = old_lab_energy.boxArray();
    auto const& mapping = old_lab_energy.DistributionMap();
    bool valid = SupportedThermalControls(dt, options.field) &&
        !options.field.reflecting_axial_field_trial &&
        !options.field.reflecting_axial_table_trial && geometry.isPeriodic(1) &&
        PhysConst::c * dt <= .5 * std::min(geometry.CellSize(0), geometry.CellSize(1)) &&
        old_lab_energy.nComp() == 1 && old_lab_energy.ixType().cellCentered() &&
        old_lab_momentum.nComp() == 3 && proper_bath.nComp() == 1 &&
        work_velocity.nComp() == 3 && actual_work.nComp() == 1 && diffusion_mask.nComp() == 1;
    for (auto const* field : {&old_lab_momentum, &proper_bath, &work_velocity, &actual_work})
    {
        valid = valid && field->ok() && field->boxArray() == boxes &&
            field->DistributionMap() == mapping && field->ixType().cellCentered();
    }
    valid = valid && diffusion_mask.boxArray() == boxes &&
        diffusion_mask.DistributionMap() == mapping && diffusion_mask.ixType().cellCentered();
    if (!Collective(valid) || !SameThermalControls(dt, options.field)) { return rejected; }
    // Context and EOS come from the actual native model, never a rewritten
    // caller executor. This prerequisite collectively checks the TRUE sole
    // TABLE selector/fixed charge and native geometry/layout/boundaries.
    if (!Collective(SupportedNativeThermalContext(geometry, boxes, mapping, false, false, true)))
    {
        return rejected;
    }
    auto const eos = WarpX::GetInstance().get_pointer_HybridPICModel()->
        electronThermodynamicsExecutor();
    auto snapshot = detail::TrySnapshotResolvedIndependentGrayMaterialTableOpacity(
        raw_charge_density, nodal_temperature, unit_ion_charge, eos, geometry, table,
        options.opacity);
    if (!snapshot.valid) { rejected.reason = snapshot.reason; return rejected; }
    if (!Collective(snapshot.coefficients.rosseland_transport.boxArray() == boxes &&
                    snapshot.coefficients.rosseland_transport.DistributionMap() == mapping))
    {
        return rejected;
    }
    auto const linear = options.opacity.encoding ==
        detail::IndependentGrayMaterialOpacityEncoding::ExactZeroLinearAllScatteringZero;
    auto velocity = Host(work_velocity, geometry), work = Host(actual_work, geometry);
    auto mask = Host(diffusion_mask, geometry);
    auto const& transport = snapshot.coefficients.rosseland_transport;
    auto const& absorption = snapshot.coefficients.proper_absorption;
    for (amrex::MFIter it(transport); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            auto const rho = snapshot.electron_density[it](index) * options.opacity.mass_per_electron;
            auto const t = snapshot.electron_temperature[it](index);
            auto const proof = linear
                ? detail::BoundExactZeroScatteringGrayOpacityResponse(table, rho, t,
                                                                      options.opacity.axis_policy)
                : detail::BoundPositiveLogGrayOpacityResponse(table, rho, t,
                                                              options.opacity.axis_policy);
            auto const r = transport[it](index), p = absorption[it](index);
            valid = valid && proof.valid && proof.resolved && proof.ideal[0].lo > 0 &&
                proof.ideal[1].lo > 0 && r == proof.response.transport_opacity &&
                p == proof.recovered_alpha && std::isnormal(r) && std::isnormal(p) &&
                r > 0 && p > 0 && mask[it](index) == 1 && work[it](index) == 0 &&
                velocity[it](index, 0) == 0 && velocity[it](index, 1) == 0 &&
                velocity[it](index, 2) == 0 && PhysConst::c * dt * r >= 8;
            if (!valid) { continue; }
            // Separate channel, inversion and ratio conditioning. The global
            // provider metric alone can hide a weak R beneath a large S.
            valid = valid && ConditionIndependentChannels(proof, r, p, dt).valid;
        }
    }
    if (!Collective(valid))
    {
        rejected.reason = "independent-gray rest ALL-D scope or R/P/inverse arithmetic rejected";
        return rejected;
    }
    // Check the authentic eager D-D ratio graph before BuildFaces. Rest does
    // not license inf*0: products can overflow even when P/R itself is finite.
    for (int d = 0; d < 2; ++d)
    {
        for (amrex::MFIter it(transport); it.isValid(); ++it)
        {
            for (amrex::BoxIterator face(amrex::surroundingNodes(it.validbox(), d));
                 face.ok(); ++face)
            {
                if (ReflectingFace(face(), d, geometry)) { continue; }
                auto left = face(); --left[d];
                auto const right = face();
                auto const a = transport[it](right) * absorption[it](left);
                auto const b = transport[it](left) * absorption[it](right);
                auto const denominator = transport[it](left) + transport[it](right);
                auto const first = a / transport[it](left);
                auto const second = b / transport[it](right);
                auto const ratio = (first + second) / denominator;
                valid = valid && a > 0 && b > 0 && std::isfinite(a) && std::isfinite(b) &&
                    denominator > 0 && std::isfinite(denominator) && first > 0 && second > 0 &&
                    std::isfinite(first + second) && ratio > 0 && std::isfinite(ratio);
            }
        }
    }
    if (!Collective(valid))
    {
        rejected.reason = "independent-gray eager face-ratio arithmetic rejected";
        return rejected;
    }
    auto field_options = options.field;
    field_options.single_selected_table_trial = true;
    return TryAdvanceThermalFieldImplementation(old_lab_energy, old_lab_momentum, proper_bath,
        transport, absorption, diffusion_mask, work_velocity, actual_work, geometry, dt,
        field_options, true);
#endif
}

IndependentGrayMovingFieldOpacity::IndependentGrayMovingFieldOpacity (
    detail::ResolvedIndependentGrayMaterialTableOpacitySnapshot&& snapshot,
    amrex::MultiFab&& errors, amrex::Geometry const& geometry, amrex::Real dt,
    ResolvedRZThermalDiffusionFieldOptions const& options,
    std::unique_ptr<amrex::iMultiFab const> mixed_mask,
    std::unique_ptr<amrex::MultiFab const> masked_transport,
    std::unique_ptr<amrex::MultiFab const> masked_absorption)
    : m_snapshot(std::move(snapshot)), m_errors(std::move(errors)),
      m_geometry(geometry), m_dt(dt), m_options(options),
      m_mixed_mask(std::move(mixed_mask)), m_masked_transport(std::move(masked_transport)),
      m_masked_absorption(std::move(masked_absorption))
{}

std::unique_ptr<IndependentGrayMovingFieldOpacity const>
IndependentGrayMovingFieldOpacity::Issue (
    amrex::MultiFab const& raw_charge_density,
    amrex::MultiFab const& nodal_temperature,
    detail::ResolvedMaterialChargeFields const& unit_ion_charge,
    MaterialOpacityTable const& table,
    amrex::Geometry const& geometry, amrex::Real dt,
    IndependentGrayMovingAllDFieldOptions const& options, std::string& reason)
{
    return IssueEpoch(raw_charge_density, nodal_temperature, unit_ion_charge,
                      table, nullptr, geometry, dt, options, reason);
}

std::unique_ptr<IndependentGrayMovingFieldOpacity const>
IndependentGrayMovingFieldOpacity::IssueMixed (
    amrex::MultiFab const& raw_charge_density, amrex::MultiFab const& nodal_temperature,
    detail::ResolvedMaterialChargeFields const& unit_ion_charge,
    MaterialOpacityTable const& table, amrex::iMultiFab const& fixed_mask,
    amrex::Geometry const& geometry, amrex::Real dt,
    IndependentGrayMovingAllDFieldOptions const& options, std::string& reason)
{
    return IssueEpoch(raw_charge_density, nodal_temperature, unit_ion_charge,
                      table, &fixed_mask, geometry, dt, options, reason);
}

std::unique_ptr<IndependentGrayMovingFieldOpacity const>
IndependentGrayMovingFieldOpacity::IssueEpoch (
    amrex::MultiFab const& raw_charge_density, amrex::MultiFab const& nodal_temperature,
    detail::ResolvedMaterialChargeFields const& unit_ion_charge,
    MaterialOpacityTable const& table, amrex::iMultiFab const* fixed_mask,
    amrex::Geometry const& geometry, amrex::Real dt,
    IndependentGrayMovingAllDFieldOptions const& options, std::string& reason)
{
    reason = "unsupported independent-gray moving ALL-D FIELD trial";
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU) || \
    !defined(WARPX_USE_SINGULARITY_EOS) || !defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
    amrex::ignore_unused(raw_charge_density, nodal_temperature, unit_ion_charge,
                        table, fixed_mask, geometry, dt, options);
    return nullptr;
#else
    int low = options.enabled, high = low;
    amrex::ParallelDescriptor::ReduceIntMin(low);
    amrex::ParallelDescriptor::ReduceIntMax(high);
    if (low != high || low == 0) { return nullptr; }
    auto const boxes = amrex::convert(raw_charge_density.boxArray(), amrex::IntVect(0));
    auto const& mapping = raw_charge_density.DistributionMap();
    int mask_low = fixed_mask != nullptr, mask_high = mask_low;
    amrex::ParallelDescriptor::ReduceIntMin(mask_low);
    amrex::ParallelDescriptor::ReduceIntMax(mask_high);
    if (mask_low != mask_high) { return nullptr; }
    bool valid = std::numeric_limits<amrex::Real>::is_iec559 &&
        (!options.opacity.allow_exact_vacuum ||
         (fixed_mask && options.field.single_selected_table_vacuum)) &&
        std::numeric_limits<amrex::Real>::radix == 2 &&
        std::numeric_limits<amrex::Real>::digits == 53 &&
        std::numeric_limits<long double>::is_iec559 &&
        std::numeric_limits<long double>::radix == 2 &&
        std::numeric_limits<long double>::digits >= 64 &&
        std::numeric_limits<long double>::min_exponent <= -8192 &&
        std::numeric_limits<long double>::max_exponent >= 8192 &&
        std::fegetround() == FE_TONEAREST &&
        SupportedThermalControls(dt, options.field) &&
        !options.field.reflecting_axial_field_trial &&
        !options.field.reflecting_axial_table_trial && geometry.isPeriodic(1) &&
        PhysConst::c * dt <= .5 * std::min(geometry.CellSize(0), geometry.CellSize(1));
    if (!Collective(valid) || !SameThermalControls(dt, options.field)) { return nullptr; }
    std::unique_ptr<amrex::iMultiFab> mixed_mask;
    if (fixed_mask)
    {
        if (!Collective(fixed_mask->ok() && fixed_mask->nComp() == 1 &&
                fixed_mask->boxArray() == boxes && fixed_mask->DistributionMap() == mapping &&
                fixed_mask->nGrowVect().allGE(1))) { return nullptr; }
        mixed_mask = std::make_unique<amrex::iMultiFab>(boxes, mapping, 1,
                                                       fixed_mask->nGrowVect());
        amrex::iMultiFab::Copy(*mixed_mask, *fixed_mask, 0, 0, 1, fixed_mask->nGrowVect());
    }
    // Same native selector/EOS authority as the strict-rest seam. No caller
    // executor, forged coefficient pair or inferred S=R-P enters this call.
    if (!Collective(SupportedNativeThermalContext(geometry, boxes, mapping, false, false, true,
            options.field.single_selected_table_vacuum)))
    {
        return nullptr;
    }
    auto const eos = WarpX::GetInstance().get_pointer_HybridPICModel()->
        electronThermodynamicsExecutor();
    auto snapshot = detail::TrySnapshotResolvedIndependentGrayMaterialTableOpacity(
        raw_charge_density, nodal_temperature, unit_ion_charge, eos, geometry, table,
        options.opacity);
    if (!snapshot.valid) { reason = snapshot.reason; return nullptr; }
    auto const& transport = snapshot.coefficients.rosseland_transport;
    auto const& absorption = snapshot.coefficients.proper_absorption;
    if (!Collective(transport.boxArray() == boxes && transport.DistributionMap() == mapping))
    {
        return nullptr;
    }
    auto errors = HostZero(boxes, mapping, 2, 1);
    auto const linear = options.opacity.encoding ==
        detail::IndependentGrayMaterialOpacityEncoding::ExactZeroLinearAllScatteringZero;
    for (amrex::MFIter it(transport); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            auto const rho = snapshot.electron_density[it](index) * options.opacity.mass_per_electron;
            auto const t = snapshot.electron_temperature[it](index);
            if (options.opacity.allow_exact_vacuum && snapshot.electron_density[it](index) == 0)
            {
                bool const vacant = mixed_mask && (*mixed_mask)[it](index) == 0 && t == 0 &&
                    transport[it](index) == 0 && absorption[it](index) == 0 &&
                    snapshot.coefficients.true_scattering[it](index) == 0;
                valid = valid && vacant;
                continue;
            }
            auto const proof = linear
                ? detail::BoundExactZeroScatteringGrayOpacityResponse(table, rho, t,
                                                                      options.opacity.axis_policy)
                : detail::BoundPositiveLogGrayOpacityResponse(table, rho, t,
                                                              options.opacity.axis_policy);
            auto const r = transport[it](index), p = absorption[it](index);
            bool cell_valid = proof.valid && proof.resolved && proof.ideal[0].lo > 0 &&
                proof.ideal[1].lo > 0 && r == proof.response.transport_opacity &&
                p == proof.recovered_alpha && std::isnormal(r) && std::isnormal(p) &&
                r > 0 && p > 0 &&
                ((mixed_mask && (*mixed_mask)[it](index) == 0) || PhysConst::c * dt * r >= 8);
            cell_valid = cell_valid && (!mixed_mask ||
                ((*mixed_mask)[it](index) == 0 || (*mixed_mask)[it](index) == 1));
            if (cell_valid)
            {
                auto const conditioned = ConditionIndependentChannels(proof, r, p, dt);
                cell_valid = conditioned.valid;
                // The stored double bounds are rounded UP, then communicated
                // through exactly the canonical one-ghost interface epoch.
                errors[it](index, 0) = std::nextafter(
                    static_cast<amrex::Real>(conditioned.transport_error),
                    std::numeric_limits<amrex::Real>::infinity());
                errors[it](index, 1) = std::nextafter(
                    static_cast<amrex::Real>(conditioned.absorption_error),
                    std::numeric_limits<amrex::Real>::infinity());
            }
            valid = valid && cell_valid;
        }
    }
    if (!Collective(valid))
    {
        reason = "independent-gray moving ALL-D scope, frame or R/P/inverse arithmetic rejected";
        return nullptr;
    }
    // No local validity shortcut may skip this collective halo exchange.
    // Empty ranks participate, and periodic aliases use the same real owner
    // errors as coefficient/velocity halos, not fresh uncorrelated witnesses.
    errors.FillBoundary(geometry.periodicity());
    std::unique_ptr<amrex::MultiFab> masked_transport, masked_absorption;
    if (mixed_mask)
    {
        masked_transport = std::make_unique<amrex::MultiFab>(boxes, mapping, 1, 1);
        masked_absorption = std::make_unique<amrex::MultiFab>(boxes, mapping, 1, 1);
        masked_transport->setVal(0);
        masked_absorption->setVal(0);
        for (amrex::MFIter it(*masked_transport); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                if ((*mixed_mask)[it](cell()) == 1)
                {
                    (*masked_transport)[it](cell()) = transport[it](cell());
                    (*masked_absorption)[it](cell()) = absorption[it](cell());
                }
            }
        }
        masked_transport->FillBoundary(geometry.periodicity());
        masked_absorption->FillBoundary(geometry.periodicity());
    }
    return std::unique_ptr<IndependentGrayMovingFieldOpacity const>(
        new IndependentGrayMovingFieldOpacity(std::move(snapshot), std::move(errors),
            geometry, dt, options.field, std::move(mixed_mask),
            std::move(masked_transport), std::move(masked_absorption)));
#endif
}

bool IndependentGrayMovingFieldOpacity::Preflight (
    amrex::MultiFab const& old_lab_energy, amrex::MultiFab const& old_lab_momentum,
    amrex::MultiFab const& proper_bath, amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& work_velocity, amrex::MultiFab const& actual_work,
    std::string& reason) const
{
    reason = "independent-gray moving ALL-D scope, frame or R/P/inverse arithmetic rejected";
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU) || \
    !defined(WARPX_USE_SINGULARITY_EOS) || !defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
    amrex::ignore_unused(old_lab_energy, old_lab_momentum, proper_bath, diffusion_mask,
                        work_velocity, actual_work);
    return false;
#else
    auto const& geometry = m_geometry;
    auto const dt = m_dt;
    auto const& options = m_options;
    auto const& transport = m_snapshot.coefficients.rosseland_transport;
    auto const& absorption = m_snapshot.coefficients.proper_absorption;
    auto const& boxes = transport.boxArray();
    auto const& mapping = transport.DistributionMap();
    bool valid = std::fegetround() == FE_TONEAREST &&
        old_lab_energy.ok() && old_lab_energy.nComp() == 1 &&
        old_lab_momentum.nComp() == 3 && proper_bath.nComp() == 1 &&
        work_velocity.nComp() == 3 && actual_work.nComp() == 1 && diffusion_mask.nComp() == 1;
    for (auto const* field : {&old_lab_energy, &old_lab_momentum, &proper_bath,
                              &work_velocity, &actual_work})
    {
        valid = valid && field->ok() && field->boxArray() == boxes &&
            field->DistributionMap() == mapping && field->ixType().cellCentered();
    }
    valid = valid && diffusion_mask.ok() && diffusion_mask.boxArray() == boxes &&
        diffusion_mask.DistributionMap() == mapping && diffusion_mask.ixType().cellCentered();
    if (!Collective(valid)) { return false; }
    if (m_mixed_mask)
    {
        if (!Collective(diffusion_mask.nGrowVect() == m_mixed_mask->nGrowVect()))
        { return false; }
        for (amrex::MFIter it(diffusion_mask); it.isValid(); ++it)
        {
            auto const bytes =
                static_cast<std::size_t>(diffusion_mask[it].box().numPts()) * sizeof(int);
            valid = valid && std::memcmp(diffusion_mask[it].dataPtr(),
                                         (*m_mixed_mask)[it].dataPtr(), bytes) == 0;
        }
        if (!Collective(valid))
        {
            reason = "independent-gray frozen diffusion mask changed";
            return false;
        }
    }
    if (!Collective(SupportedNativeThermalContext(geometry, boxes, mapping, false, false, true,
            options.single_selected_table_vacuum)))
    { return false; }
    auto velocity = Host(work_velocity, geometry), work = Host(actual_work, geometry);
    auto energy = Host(old_lab_energy, geometry), momentum = Host(old_lab_momentum, geometry);
    auto bath = Host(proper_bath, geometry);
    auto mask = Host(diffusion_mask, geometry);
    GroupMomentMetric const metric(geometry);
    auto const width = std::max(geometry.ProbHi(0) - geometry.ProbLo(0),
                                geometry.ProbHi(1) - geometry.ProbLo(1));
    valid = width > 0 && std::isfinite(width);
    for (amrex::MFIter it(transport); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            auto const r = transport[it](index), p = absorption[it](index);
            auto const flag = mask[it](index);
            if (m_mixed_mask && flag == 0)
            {
                valid = valid && energy[it](index) == 0 && momentum[it](index, 0) == 0 &&
                    momentum[it](index, 1) == 0 && momentum[it](index, 2) == 0 &&
                    work[it](index) == 0 && std::isfinite(bath[it](index)) && bath[it](index) >= 0;
                continue;
            }
            bool const frame_valid = IndependentFrameSample(velocity[it], index,
                metric.Radius(index[0] + 1), metric, r, p, width, options.maximum_beta);
            auto const check = PreflightDiffusionCell(index, energy[it](index), bath[it](index),
                r, p, velocity[it], metric, geometry, dt, options, true);
            amrex::GpuArray<amrex::Real, 3> q{};
            for (int j = 0; j < 3; ++j) { q[j] = momentum[it](index, j); }
            valid = valid && mask[it](index) == 1 && std::isfinite(work[it](index)) &&
                frame_valid && check.valid && NecessaryCausalMoment(energy[it](index), q,
                                                                   index[0], metric);
        }
    }
    if (!Collective(valid)) { return false; }
    for (int d = 0; d < 2; ++d)
    {
        for (amrex::MFIter it(transport); it.isValid(); ++it)
        {
            for (amrex::BoxIterator face(amrex::surroundingNodes(it.validbox(), d));
                 face.ok(); ++face)
            {
                auto left = face(); --left[d];
                auto const right = face();
                bool const wall = ReflectingFace(face(), d, geometry);
                bool const left_d = !wall && mask[it](left) == 1;
                bool const right_d = !wall && mask[it](right) == 1;
                auto const source = wall
                    ? (face()[d] == geometry.Domain().smallEnd(d) ? right : left)
                    : (left_d ? left : right);
                if ((wall && mask[it](source) == 0) || (!wall && !left_d && !right_d))
                { continue; }
                IndependentFaceConditioning conditioned;
                if (wall || !(left_d && right_d))
                {
                    conditioned = {true, absorption[it](source) / transport[it](source),
                                   transport[it](source), absorption[it](source)};
                }
                else
                {
                    conditioned = ConditionIndependentFace(transport[it](left), transport[it](right),
                        absorption[it](left), absorption[it](right), m_errors[it], left, right);
                }
                bool face_valid = conditioned.valid;
                auto const sample = [&] (amrex::Real radius)
                {
                    bool accepted = IndependentFrameSample(velocity[it], source, radius, metric,
                        conditioned.transport, conditioned.absorption, width, options.maximum_beta);
                    auto beta = PhysicalBeta(velocity[it], source, radius, metric);
                    if (!wall && left_d && right_d)
                    {
                        accepted = IndependentFrameSample(velocity[it], right, radius, metric,
                            conditioned.transport, conditioned.absorption, width,
                            options.maximum_beta) && accepted;
                        auto const other = PhysicalBeta(velocity[it], right, radius, metric);
                        for (int j = 0; j < 3; ++j) { beta[j] = .5 * (beta[j] + other[j]); }
                    }
                    else if (wall) { beta[d == 0 ? 0 : 2] = 0; }
                    amrex::GpuArray<amrex::Real, 3> dipole{};
                    amrex::Real unit_energy = 1;
                    if (!wall && !(left_d && right_d))
                    {
                        auto const lambda = options.linear.marshak_extrapolation_lambda;
                        auto const denominator =
                            conditioned.transport * geometry.CellSize(d) + 2 * lambda;
                        unit_energy = 2 * lambda / denominator;
                        dipole[d == 0 ? 0 : 2] = (left_d ? 1 : -1) * 2 / (3 * denominator);
                    }
                    for (int j = 0; j < 3; ++j)
                    { dipole[j] -= conditioned.ratio * beta[j] * unit_energy; }
                    auto const unit = BoostedDipole(unit_energy, dipole, beta);
                    accepted = accepted && unit[0][0] > 0;
                    for (auto const& row : unit)
                    {
                        for (auto const value : row) { accepted = accepted && std::isfinite(value); }
                    }
                    return accepted;
                };
                if (d == 0) { face_valid = sample(metric.Radius(face()[0])) && face_valid; }
                else
                {
                    auto const quadrature = AnnularQuadrature(face()[0], metric);
                    face_valid = sample(quadrature.radius[0]) && face_valid;
                    face_valid = sample(quadrature.radius[1]) && face_valid;
                }
                valid = valid && face_valid;
            }
        }
    }
    if (!Collective(valid))
    {
        reason = "independent-gray moving eager face-ratio, conditioning or frame rejected";
        return false;
    }
    reason.clear();
    return true;
#endif
}

ResolvedRZThermalDiffusionFieldResult
IndependentGrayMovingFieldOpacity::Advance (
    amrex::MultiFab const& old_lab_energy, amrex::MultiFab const& old_lab_momentum,
    amrex::MultiFab const& proper_bath, amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& work_velocity, amrex::MultiFab const& actual_work) const
{
    ResolvedRZThermalDiffusionFieldResult rejected;
    if (!Preflight(old_lab_energy, old_lab_momentum, proper_bath, diffusion_mask,
                   work_velocity, actual_work, rejected.reason)) { return rejected; }
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU) && \
    defined(WARPX_USE_SINGULARITY_EOS) && defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
    auto field_options = m_options;
    field_options.single_selected_table_trial = true;
    return TryAdvanceThermalFieldImplementation(old_lab_energy, old_lab_momentum, proper_bath,
        m_masked_transport ? *m_masked_transport : m_snapshot.coefficients.rosseland_transport,
        m_masked_absorption ? *m_masked_absorption : m_snapshot.coefficients.proper_absorption,
        diffusion_mask, work_velocity, actual_work, m_geometry, m_dt, field_options, true);
#else
    return rejected;
#endif
}

ResolvedRZThermalDiffusionFieldResult
TryAdvanceIndependentGrayMovingAllDField (
    amrex::MultiFab const& old_lab_energy, amrex::MultiFab const& old_lab_momentum,
    amrex::MultiFab const& proper_bath, amrex::MultiFab const& raw_charge_density,
    amrex::MultiFab const& nodal_temperature,
    detail::ResolvedMaterialChargeFields const& unit_ion_charge,
    MaterialOpacityTable const& table, amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& work_velocity, amrex::MultiFab const& actual_work,
    amrex::Geometry const& geometry, amrex::Real dt,
    IndependentGrayMovingAllDFieldOptions const& options)
{
    ResolvedRZThermalDiffusionFieldResult rejected;
    auto const opacity = IndependentGrayMovingFieldOpacity::Issue(
        raw_charge_density, nodal_temperature, unit_ion_charge, table, geometry, dt,
        options, rejected.reason);
    if (!opacity) { return rejected; }
    return opacity->Advance(old_lab_energy, old_lab_momentum, proper_bath,
                            diffusion_mask, work_velocity, actual_work);
}
} // namespace warpx::radiation
