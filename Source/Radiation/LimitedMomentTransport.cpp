/* Copyright 2026 The WarpX Community
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/LimitedMomentTransport.H"

#include "Radiation/MomentClosure.H"
#include "Radiation/MovingMomentFlux.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace warpx::radiation
{
using namespace amrex::literals;

namespace
{

constexpr amrex::Real cone_roundoff =
    8 * std::numeric_limits<amrex::Real>::epsilon();
constexpr amrex::Real balance_roundoff =
    128 * std::numeric_limits<amrex::Real>::epsilon();

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
PhysicalDirection (int dimension) noexcept
{
#if defined(WARPX_DIM_1D_Z)
    (void) dimension;
    return 2;
#elif defined(WARPX_DIM_XZ)
    return dimension == 0 ? 0 : 2;
#else
    return dimension;
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE FourVector
LoadDensity (amrex::Array4<amrex::Real const> const& state, int const (&cell)[3],
             amrex::Real inverse_volume) noexcept
{
    FourVector value{};
    for (int component = 0; component < 4; ++component)
    {
        value[component] = state(cell[0], cell[1], cell[2], component) * inverse_volume;
    }
    return value;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE FourVector
PhysicalFlux (FourVector const& state, int normal) noexcept
{
    auto const closure = EvaluateM1Closure(state);
    FourVector flux{};
    if (!closure.valid)
    {
        return flux;
    }
    flux[0] = PhysConst::c * state[normal + 1];
    for (int component = 0; component < 3; ++component)
    {
        flux[component + 1] =
            PhysConst::c * closure.tensor[component + 1][normal + 1];
    }
    return flux;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
ConeMargin (FourVector const& state) noexcept
{
    if (!EvaluateM1Closure(state).valid)
    {
        return -std::numeric_limits<amrex::Real>::infinity();
    }
    if (state[0] == 0)
    {
        return 1.0;
    }
    return 1.0 - std::hypot(std::hypot(state[1], state[2]), state[3]) / state[0];
}

struct LimitedFace
{
    FourVector flux{};
    FourVector bar_left{};
    FourVector bar_right{};
    amrex::Real theta = 0.0;
    amrex::Real cone_margin = -std::numeric_limits<amrex::Real>::infinity();
    bool valid = false;
};

#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_RCYLINDER) && \
    !defined(WARPX_DIM_RSPHERE)
bool
FabsOverlap (amrex::FArrayBox const& first, amrex::FArrayBox const& second) noexcept
{
    auto const* const first_data = first.dataPtr();
    auto const* const second_data = second.dataPtr();
    if (first_data == nullptr || second_data == nullptr)
    {
        return true;
    }
    auto const first_begin = reinterpret_cast<std::uintptr_t>(first_data);
    auto const second_begin = reinterpret_cast<std::uintptr_t>(second_data);
    auto const first_end = first_begin + first.nBytes();
    auto const second_end = second_begin + second.nBytes();
    return first_begin < second_end && second_begin < first_end;
}
#endif

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE LimitedFace
LimitFaceFlux (FourVector const& left, FourVector const& right,
               amrex::GpuArray<amrex::Real, 3> const& beta, int normal,
               amrex::Real chi, amrex::Real spacing,
               amrex::GpuArray<amrex::Real, 3> const& transverse_gradient,
               int maximum_bisections) noexcept
{
    LimitedFace result;
    auto const left_flux = PhysicalFlux(left, normal);
    auto const right_flux = PhysicalFlux(right, normal);
    FourVector llf{};
    for (int component = 0; component < 4; ++component)
    {
        llf[component] = 0.5_rt * (left_flux[component] + right_flux[component]) -
            0.5_rt * PhysConst::c * (right[component] - left[component]);
    }
    auto const asymptotic = EvaluateMovingMomentFlux(
        left, right, beta, normal, chi, spacing, transverse_gradient);
    if (!asymptotic.valid)
    {
        return result;
    }

    auto blended = [=] (amrex::Real theta) noexcept
    {
        FourVector flux{};
        for (int component = 0; component < 4; ++component)
        {
            flux[component] = llf[component] +
                theta * (asymptotic.flux[component] - llf[component]);
        }
        return flux;
    };
    auto endpoints = [=] (amrex::Real theta, FourVector const& flux)
    {
        FourVector bar_left{};
        FourVector bar_right{};
        if (theta == 0.0)
        {
            for (int component = 0; component < 4; ++component)
            {
                auto const common = 0.5_rt *
                    ((left[component] + left_flux[component] / PhysConst::c) +
                     (right[component] - right_flux[component] / PhysConst::c));
                bar_left[component] = common;
                bar_right[component] = common;
            }
        }
        else
        {
            for (int component = 0; component < 4; ++component)
            {
                bar_left[component] = left[component] -
                    (flux[component] - left_flux[component]) / PhysConst::c;
                bar_right[component] = right[component] +
                    (flux[component] - right_flux[component]) / PhysConst::c;
            }
        }
        return amrex::GpuArray<FourVector, 2>{bar_left, bar_right};
    };
    auto admissible = [&] (amrex::Real theta) noexcept
    {
        auto const endpoint = endpoints(theta, blended(theta));
        return EvaluateM1Closure(endpoint[0]).valid &&
               EvaluateM1Closure(endpoint[1]).valid;
    };

    amrex::Real low = 0.0;
    amrex::Real high = 1.0;
    if (admissible(high))
    {
        low = high;
    }
    else if (!admissible(low))
    {
        return result;
    }
    else
    {
        for (int iteration = 0; iteration < maximum_bisections; ++iteration)
        {
            auto const middle = 0.5_rt * (low + high);
            if (middle <= low || middle >= high)
            {
                break;
            }
            if (admissible(middle))
            {
                low = middle;
            }
            else
            {
                high = middle;
            }
        }
    }

    result.theta = low;
    result.flux = blended(low);
    auto const endpoint = endpoints(low, result.flux);
    result.bar_left = endpoint[0];
    result.bar_right = endpoint[1];
    result.cone_margin = amrex::min(
        ConeMargin(endpoint[0]), ConeMargin(endpoint[1]));
    for (auto value : result.flux)
    {
        if (!amrex::Math::isfinite(value))
        {
            return LimitedFace{};
        }
    }
    result.valid = amrex::Math::isfinite(result.theta) && result.theta >= 0.0 &&
        result.theta <= 1.0 && amrex::Math::isfinite(result.cone_margin);
    return result;
}

} // namespace

LimitedMomentTransportResult
TryAdvanceLimitedMomentTransport (amrex::MultiFab& state, amrex::MultiFab const& beta,
                                  amrex::MultiFab const& chi, amrex::Geometry const& geometry,
                                  amrex::Real dt, LimitedMomentTransportOptions const& options)
{
    LimitedMomentTransportResult result;

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || \
    defined(WARPX_DIM_RSPHERE)
    (void) state;
    (void) beta;
    (void) chi;
    (void) geometry;
    (void) dt;
    (void) options;
    result.failure = LimitedMomentTransportFailure::InvalidGeometry;
    return result;
#else
    bool boxes_inside_domain = true;
    for (int box_index = 0; box_index < state.boxArray().size(); ++box_index)
    {
        amrex::Box const& box = state.boxArray()[box_index];
        boxes_inside_domain = boxes_inside_domain && geometry.Domain().contains(box);
    }
    bool const compatible_layout =
        state.nComp() == 4 && beta.nComp() == 3 && chi.nComp() == 1 &&
        state.ixType().cellCentered() && beta.ixType().cellCentered() &&
        chi.ixType().cellCentered() && state.nGrowVect().max() <= 1 &&
        beta.nGrowVect().max() <= 1 && chi.nGrowVect().max() <= 1 &&
        beta.boxArray() == state.boxArray() && chi.boxArray() == state.boxArray() &&
        beta.DistributionMap() == state.DistributionMap() &&
        chi.DistributionMap() == state.DistributionMap() && boxes_inside_domain &&
        state.boxArray().numPts() == geometry.Domain().numPts() &&
        geometry.Domain().numPts() > 0 && geometry.Coord() == amrex::CoordSys::cartesian &&
        geometry.isAllPeriodic();
    bool overlapping_fabs = false;
    if (compatible_layout)
    {
        for (amrex::MFIter iterator(state); iterator.isValid(); ++iterator)
        {
            overlapping_fabs = overlapping_fabs ||
                FabsOverlap(state[iterator], beta[iterator]) ||
                FabsOverlap(state[iterator], chi[iterator]) ||
                FabsOverlap(beta[iterator], chi[iterator]);
        }
    }
    bool const layout_valid = compatible_layout && !overlapping_fabs &&
        state.boxArray().isDisjoint();
    int local_layout_failure = layout_valid ? 0 : 1;
    amrex::ParallelDescriptor::ReduceIntMax(local_layout_failure);
    if (local_layout_failure != 0)
    {
        result.failure = LimitedMomentTransportFailure::InvalidGeometry;
        return result;
    }

    int controls_invalid = !std::isfinite(dt) || dt <= 0 ||
        options.maximum_bisections < 1 || options.maximum_bisections > 64;
    amrex::ParallelDescriptor::ReduceIntMax(controls_invalid);
    if (controls_invalid != 0)
    {
        result.failure = LimitedMomentTransportFailure::InvalidDt;
        return result;
    }
    amrex::Real minimum_dt = dt;
    amrex::Real maximum_dt = dt;
    amrex::ParallelDescriptor::ReduceRealMin(minimum_dt);
    amrex::ParallelDescriptor::ReduceRealMax(maximum_dt);
    int minimum_budget = options.maximum_bisections;
    int maximum_budget = options.maximum_bisections;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_budget);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_budget);
    if (minimum_dt != maximum_dt || minimum_budget != maximum_budget)
    {
        result.failure = std::isfinite(dt) && minimum_dt == maximum_dt
            ? LimitedMomentTransportFailure::InvalidGeometry
            : LimitedMomentTransportFailure::InvalidDt;
        return result;
    }

    auto const cell_size = geometry.CellSizeArray();
    amrex::Real volume = 1.0;
    bool sizes_valid = true;
    for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
    {
        sizes_valid = sizes_valid && std::isfinite(cell_size[dimension]) &&
            cell_size[dimension] > 0.0;
        volume *= cell_size[dimension];
    }
    amrex::Real local_cfl = 0.0;
    for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
    {
        local_cfl += PhysConst::c * dt / cell_size[dimension];
    }
    bool const time_valid = std::isfinite(dt) && dt > 0.0 && sizes_valid &&
        std::isfinite(volume) && volume > 0.0 && std::isfinite(local_cfl);
    result.maximum_cfl = time_valid ? local_cfl : std::numeric_limits<amrex::Real>::quiet_NaN();
    int local_time_failure = time_valid ? 0 : 1;
    amrex::ParallelDescriptor::ReduceIntMax(local_time_failure);
    amrex::ParallelDescriptor::ReduceRealMax(result.maximum_cfl);
    if (local_time_failure != 0)
    {
        result.failure = LimitedMomentTransportFailure::InvalidDt;
        return result;
    }
    if (result.maximum_cfl > 0.5)
    {
        result.failure = LimitedMomentTransportFailure::InvalidDt;
        return result;
    }

    amrex::Real const inverse_volume = 1.0 / volume;
    int local_state_failure = 0;
    amrex::Real local_minimum_margin = std::numeric_limits<amrex::Real>::infinity();
    {
        amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMin> ops;
        amrex::ReduceData<int, amrex::Real> data(ops);
        for (amrex::MFIter iterator(state); iterator.isValid(); ++iterator)
        {
            auto const field = state.const_array(iterator);
            ops.eval(iterator.validbox(), data,
                     [=] AMREX_GPU_DEVICE (int i, int j, int k)
                         -> amrex::GpuTuple<int, amrex::Real>
                     {
                         int const cell[3] = {i, j, k};
                         auto const density = LoadDensity(field, cell, inverse_volume);
                         auto const margin = ConeMargin(density);
                         return {margin == -std::numeric_limits<amrex::Real>::infinity(),
                                 margin};
                     });
        }
        auto const values = data.value();
        local_state_failure = amrex::get<0>(values);
        local_minimum_margin = amrex::get<1>(values);
    }
    amrex::ParallelDescriptor::ReduceIntMax(local_state_failure);
    amrex::ParallelDescriptor::ReduceRealMin(local_minimum_margin);
    if (local_state_failure != 0 || local_minimum_margin < -cone_roundoff)
    {
        result.failure = LimitedMomentTransportFailure::InvalidState;
        return result;
    }

    amrex::Real local_field_failure[3] = {0.0, 0.0, 0.0};
    {
        amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax,
                         amrex::ReduceOpMax> ops;
        amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real> data(ops);
        for (amrex::MFIter iterator(state); iterator.isValid(); ++iterator)
        {
            auto const velocity = beta.const_array(iterator);
            auto const opacity = chi.const_array(iterator);
            ops.eval(iterator.validbox(), data,
                     [=] AMREX_GPU_DEVICE (int i, int j, int k)
                         -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real>
                     {
                         amrex::Real velocity_failure = 0.0;
                         amrex::Real velocity_norm_failure = 0.0;
                         for (int component = 0; component < 3; ++component)
                         {
                             if (!amrex::Math::isfinite(velocity(i, j, k, component)))
                             {
                                 velocity_failure = 1.0;
                             }
                         }
                         auto const velocity_norm = std::hypot(std::hypot(
                             velocity(i, j, k, 0), velocity(i, j, k, 1)),
                             velocity(i, j, k, 2));
                         if (!amrex::Math::isfinite(velocity_norm) ||
                             velocity_norm >= 1.0)
                         {
                             velocity_norm_failure = 1.0;
                         }
                         auto const opacity_value = opacity(i, j, k);
                         return {velocity_failure, velocity_norm_failure,
                                 (!amrex::Math::isfinite(opacity_value) ||
                                  opacity_value < 0.0) ? 1.0 : 0.0};
                     });
        }
        auto const values = data.value();
        local_field_failure[0] = amrex::get<0>(values);
        local_field_failure[1] = amrex::get<1>(values);
        local_field_failure[2] = amrex::get<2>(values);
    }
    amrex::ParallelDescriptor::ReduceRealMax(local_field_failure, 3);
    if (local_field_failure[0] != 0.0)
    {
        result.failure = LimitedMomentTransportFailure::InvalidBeta;
        return result;
    }
    if (local_field_failure[1] != 0.0)
    {
        result.failure = LimitedMomentTransportFailure::InvalidBeta;
        return result;
    }
    if (local_field_failure[2] != 0.0)
    {
        result.failure = LimitedMomentTransportFailure::InvalidOpacity;
        return result;
    }

    amrex::IntVect const ghost = amrex::IntVect::TheUnitVector();
    amrex::MultiFab candidate(state.boxArray(), state.DistributionMap(), 4, ghost);
    amrex::MultiFab beta_copy(state.boxArray(), state.DistributionMap(), 3, ghost);
    amrex::MultiFab chi_copy(state.boxArray(), state.DistributionMap(), 1, ghost);
    amrex::MultiFab::Copy(candidate, state, 0, 0, 4, 0);
    amrex::MultiFab::Copy(beta_copy, beta, 0, 0, 3, 0);
    amrex::MultiFab::Copy(chi_copy, chi, 0, 0, 1, 0);
    candidate.FillBoundary(geometry.periodicity());
    beta_copy.FillBoundary(geometry.periodicity());
    chi_copy.FillBoundary(geometry.periodicity());

    std::vector<std::unique_ptr<amrex::MultiFab>> face_flux(AMREX_SPACEDIM);
    for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
    {
        face_flux[dimension] = std::make_unique<amrex::MultiFab>(
            state.boxArray(), state.DistributionMap(), 12, ghost);
        face_flux[dimension]->setVal(0.0);
    }

    amrex::Real local_minimum_theta = std::numeric_limits<amrex::Real>::infinity();
    amrex::Real local_maximum_theta = 0.0;
    amrex::Real local_minimum_cone_margin =
        std::numeric_limits<amrex::Real>::infinity();
    amrex::Long local_accepted_faces = 0;
    amrex::Long local_limited_faces = 0;
    int const maximum_bisections = options.maximum_bisections;
    for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
    {
        int const normal = PhysicalDirection(dimension);
        amrex::ReduceOps<amrex::ReduceOpMin, amrex::ReduceOpMax, amrex::ReduceOpMin,
                         amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpMax>
            ops;
        amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, long long, long long, int>
            data(ops);
        auto const spacing = cell_size[dimension];
        for (amrex::MFIter iterator(state); iterator.isValid(); ++iterator)
        {
            auto const radiation = candidate.const_array(iterator);
            auto const velocity = beta_copy.const_array(iterator);
            auto const opacity = chi_copy.const_array(iterator);
            auto const flux = face_flux[dimension]->array(iterator);
            ops.eval(iterator.validbox(), data,
                     [=] AMREX_GPU_DEVICE (int i, int j, int k)
                         -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real,
                                           long long, long long, int>
                     {
                         int left_cell[3] = {i, j, k};
                         int right_cell[3] = {i, j, k};
                         ++right_cell[dimension];
                         auto const left = LoadDensity(radiation, left_cell, inverse_volume);
                         auto const right = LoadDensity(radiation, right_cell, inverse_volume);
                         amrex::GpuArray<amrex::Real, 3> face_beta{};
                         for (int component = 0; component < 3; ++component)
                         {
                             face_beta[component] = 0.5_rt *
                                 (velocity(left_cell[0], left_cell[1], left_cell[2],
                                           component) +
                                  velocity(right_cell[0], right_cell[1], right_cell[2],
                                           component));
                         }
                         auto const face_chi = 0.5_rt *
                             (opacity(left_cell[0], left_cell[1], left_cell[2]) +
                              opacity(right_cell[0], right_cell[1], right_cell[2]));

                         amrex::GpuArray<amrex::Real, 3> transverse_gradient{};
                             for (int transverse_dimension = 0;
                                  transverse_dimension < AMREX_SPACEDIM;
                                  ++transverse_dimension)
                         {
                             if (transverse_dimension == dimension)
                             {
                                 continue;
                             }
                             auto const component = PhysicalDirection(transverse_dimension);
                             amrex::Real gradient = 0.0;
                             for (int side = 0; side < 2; ++side)
                             {
                                 for (int normal_neighbor = 0;
                                      normal_neighbor < 2; ++normal_neighbor)
                                 {
                                     int cell[3] = {i, j, k};
                                     cell[dimension] += normal_neighbor;
                                     if (side == 0)
                                     {
                                         ++cell[transverse_dimension];
                                     }
                                     else
                                     {
                                         --cell[transverse_dimension];
                                     }
                                     auto const value = LoadDensity(
                                         radiation, cell, inverse_volume);
                                     amrex::Real reduced = value[0];
                                     for (int momentum = 0; momentum < 3;
                                          ++momentum)
                                     {
                                         reduced -= face_beta[momentum] *
                                             value[momentum + 1];
                                     }
                                     gradient += (side == 0 ? 1.0_rt : -1.0_rt) *
                                         reduced /
                                         (2.0_rt * cell_size[transverse_dimension]);
                                 }
                             }
                             transverse_gradient[component] = 0.5_rt * gradient;
                         }

                         auto const limited = LimitFaceFlux(
                             left, right, face_beta, normal, face_chi, spacing,
                             transverse_gradient, maximum_bisections);
                         if (!limited.valid)
                         {
                             return {std::numeric_limits<amrex::Real>::infinity(), 0.0,
                                     -std::numeric_limits<amrex::Real>::infinity(),
                                     0, 0, 1};
                         }
                         for (int component = 0; component < 4; ++component)
                         {
                             flux(i, j, k, component) = limited.flux[component];
                             flux(i, j, k, component + 4) =
                                 limited.bar_left[component];
                             flux(i, j, k, component + 8) =
                                 limited.bar_right[component];
                         }
                         return {limited.theta, limited.theta, limited.cone_margin, 1,
                                 limited.theta < 1.0 ? 1 : 0, 0};
                     });
        }
        auto const values = data.value();
        local_minimum_theta = amrex::min(local_minimum_theta, amrex::get<0>(values));
        local_maximum_theta = amrex::max(local_maximum_theta, amrex::get<1>(values));
        local_minimum_cone_margin = amrex::min(
            local_minimum_cone_margin, amrex::get<2>(values));
        local_accepted_faces += amrex::get<3>(values);
        local_limited_faces += amrex::get<4>(values);
        // An empty rank retains ReduceOpMax's negative identity.
        if (amrex::get<5>(values) > 0)
        {
            result.failure = LimitedMomentTransportFailure::Admissibility;
        }
    }
    amrex::ParallelDescriptor::ReduceRealMin(local_minimum_theta);
    amrex::ParallelDescriptor::ReduceRealMax(local_maximum_theta);
    amrex::ParallelDescriptor::ReduceRealMin(local_minimum_cone_margin);
    amrex::ParallelDescriptor::ReduceLongSum(local_accepted_faces);
    amrex::ParallelDescriptor::ReduceLongSum(local_limited_faces);
    int local_face_failure =
        result.failure != LimitedMomentTransportFailure::None ? 1 : 0;
    amrex::ParallelDescriptor::ReduceIntMax(local_face_failure);
    if (local_face_failure != 0)
    {
        result.failure = LimitedMomentTransportFailure::Admissibility;
        return result;
    }
    for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
    {
        face_flux[dimension]->FillBoundary(geometry.periodicity());
    }

    amrex::Real total_courant = 0.0;
    for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
    {
        total_courant += PhysConst::c * dt / cell_size[dimension];
    }
    amrex::Real const convex_weight = 1.0 - 2.0 * total_courant;
    for (amrex::MFIter iterator(state); iterator.isValid(); ++iterator)
    {
        auto const old_state = state.const_array(iterator);
        auto const next = candidate.array(iterator);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> flux;
        for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
        {
            flux[dimension] = face_flux[dimension]->const_array(iterator);
        }
        amrex::ParallelFor(iterator.validbox(),
                           [=] AMREX_GPU_DEVICE (int i, int j, int k)
                           {
                               int const cell[3] = {i, j, k};
                               auto const old_density =
                                   LoadDensity(old_state, cell, inverse_volume);
                               FourVector update{};
                               FourVector compensation{};
                               for (int component = 0; component < 4; ++component)
                               {
                                   update[component] =
                                       convex_weight * old_density[component];
                               }
                               for (int dimension = 0; dimension < AMREX_SPACEDIM;
                                    ++dimension)
                               {
                                   auto const weight =
                                       PhysConst::c * dt / cell_size[dimension];
                                   for (int component = 0; component < 4; ++component)
                                   {
                                       int low_face[3] = {i, j, k};
                                       --low_face[dimension];
                                       auto const low_bar = flux[dimension](
                                           low_face[0], low_face[1], low_face[2],
                                           component + 8);
                                       auto const high_bar = flux[dimension](
                                           i, j, k, component + 4);
                                       auto const term = weight *
                                           (low_bar + high_bar);
                                       auto const corrected =
                                           term - compensation[component];
                                       auto const next_sum =
                                           corrected + update[component];
                                       compensation[component] =
                                           (next_sum - update[component]) - corrected;
                                       update[component] = next_sum;
                                   }
                               }
                               for (int component = 0; component < 4; ++component)
                               {
                                   next(i, j, k, component) =
                                       update[component] * volume;
                               }
                           });
    }

    int local_candidate_failure = 0;
    amrex::Real local_candidate_margin =
        std::numeric_limits<amrex::Real>::infinity();
    {
        amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMin> ops;
        amrex::ReduceData<int, amrex::Real> data(ops);
        for (amrex::MFIter iterator(candidate); iterator.isValid(); ++iterator)
        {
            auto const field = candidate.const_array(iterator);
            ops.eval(iterator.validbox(), data,
                     [=] AMREX_GPU_DEVICE (int i, int j, int k)
                         -> amrex::GpuTuple<int, amrex::Real>
                     {
                         int const cell[3] = {i, j, k};
                         auto const density = LoadDensity(field, cell, inverse_volume);
                         auto const margin = ConeMargin(density);
                         return {margin == -std::numeric_limits<amrex::Real>::infinity(),
                                 margin};
                     });
        }
        auto const values = data.value();
        local_candidate_failure = amrex::get<0>(values);
        local_candidate_margin = amrex::get<1>(values);
    }
    amrex::ParallelDescriptor::ReduceIntMax(local_candidate_failure);
    amrex::ParallelDescriptor::ReduceRealMin(local_candidate_margin);
    if (local_candidate_failure != 0 || local_candidate_margin < -cone_roundoff)
    {
        result.failure = LimitedMomentTransportFailure::Admissibility;
        return result;
    }

    for (int component = 0; component < 4; ++component)
    {
        result.initial_inventory[component] = state.sum(component, false);
        result.final_inventory[component] = candidate.sum(component, false);
    }

    amrex::Real local_balance[12] = {};
    {
        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                         amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                         amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                         amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum>
            ops;
        amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                          amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                          amrex::Real, amrex::Real, amrex::Real, amrex::Real>
            data(ops);
        for (amrex::MFIter iterator(state); iterator.isValid(); ++iterator)
        {
            auto const old_field = state.const_array(iterator);
            auto const next_field = candidate.const_array(iterator);
            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> flux;
            for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
            {
                flux[dimension] = face_flux[dimension]->const_array(iterator);
            }
            ops.eval(iterator.validbox(), data,
                     [=] AMREX_GPU_DEVICE (int i, int j, int k)
                         -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real,
                                           amrex::Real, amrex::Real, amrex::Real,
                                           amrex::Real, amrex::Real, amrex::Real,
                                           amrex::Real, amrex::Real, amrex::Real>
                     {
                         amrex::Real residual[4] = {};
                         amrex::Real scale[4] = {};
                         amrex::Real equation[4] = {};
                         for (int component = 0; component < 4; ++component)
                         {
                             auto const old_value = old_field(i, j, k, component);
                             auto const next_value = next_field(i, j, k, component);
                             residual[component] = old_value - next_value;
                             equation[component] = residual[component];
                             scale[component] =
                                 std::abs(old_value) + std::abs(next_value);
                         }
                         for (int dimension = 0; dimension < AMREX_SPACEDIM;
                              ++dimension)
                         {
                             int low_face[3] = {i, j, k};
                             --low_face[dimension];
                             auto const face_weight =
                                 dt * volume / cell_size[dimension];
                             for (int component = 0; component < 4; ++component)
                             {
                                 auto const high = face_weight *
                                     flux[dimension](i, j, k, component);
                                 auto const low = face_weight *
                                     flux[dimension](low_face[0], low_face[1],
                                                     low_face[2], component);
                                 equation[component] -= high - low;
                                 scale[component] += std::abs(high) + std::abs(low);
                             }
                         }
                         return {residual[0], residual[1], residual[2], residual[3],
                                 scale[0], scale[1], scale[2], scale[3],
                                 std::abs(equation[0]), std::abs(equation[1]),
                                 std::abs(equation[2]), std::abs(equation[3])};
                     });
        }
        auto const values = data.value();
        local_balance[0] = amrex::get<0>(values);
        local_balance[1] = amrex::get<1>(values);
        local_balance[2] = amrex::get<2>(values);
        local_balance[3] = amrex::get<3>(values);
        local_balance[4] = amrex::get<4>(values);
        local_balance[5] = amrex::get<5>(values);
        local_balance[6] = amrex::get<6>(values);
        local_balance[7] = amrex::get<7>(values);
        local_balance[8] = amrex::get<8>(values);
        local_balance[9] = amrex::get<9>(values);
        local_balance[10] = amrex::get<10>(values);
        local_balance[11] = amrex::get<11>(values);
    }
    amrex::ParallelDescriptor::ReduceRealSum(local_balance, 12);
    for (int component = 0; component < 4; ++component)
    {
        result.raw_balance_residual[component] = local_balance[component];
    }
    bool balance_valid = true;
    for (int component = 0; component < 4; ++component)
    {
        balance_valid = balance_valid &&
            std::isfinite(result.initial_inventory[component]) &&
            std::isfinite(result.final_inventory[component]) &&
            std::isfinite(result.raw_balance_residual[component]) &&
            std::isfinite(local_balance[component + 4]) &&
            std::isfinite(local_balance[component + 8]) &&
            local_balance[component + 8] <=
                balance_roundoff * local_balance[component + 4] &&
            std::abs(result.initial_inventory[component] - result.final_inventory[component]) <=
                balance_roundoff * local_balance[component + 4] &&
            std::abs(result.raw_balance_residual[component]) <=
                balance_roundoff * local_balance[component + 4];
    }
    int local_balance_failure = balance_valid ? 0 : 1;
    amrex::ParallelDescriptor::ReduceIntMax(local_balance_failure);
    if (local_balance_failure != 0)
    {
        result.failure = LimitedMomentTransportFailure::GlobalBalance;
        return result;
    }

    result.minimum_theta = local_minimum_theta;
    result.maximum_theta = local_maximum_theta;
    result.minimum_cone_margin = amrex::min(
        local_minimum_cone_margin, local_candidate_margin);
    result.accepted_face_count = local_accepted_faces;
    result.limited_face_count = local_limited_faces;
    result.valid = true;

    amrex::MultiFab::Copy(state, candidate, 0, 0, 4, state.nGrowVect());
    if (state.nGrowVect().max() > 0)
    {
        state.FillBoundary(geometry.periodicity());
    }
    return result;
#endif
}

} // namespace warpx::radiation
