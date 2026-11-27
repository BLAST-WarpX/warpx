/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "MovingAsymptoticDiffusionField.H"
#include "DiffusionBoundaryFlux.H"

#include "Utils/WarpXConst.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>
#include <memory>
#include <utility>

namespace warpx::radiation
{
using namespace amrex::literals;
namespace
{
using Faces = amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>;

struct MovingFieldMetric
{
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx;
    amrex::GpuArray<int, AMREX_SPACEDIM> periodic;
    amrex::GpuArray<int, AMREX_SPACEDIM> domain_lo;
    amrex::GpuArray<int, AMREX_SPACEDIM> domain_hi;
    amrex::Real lower;
    int first, last;
    amrex::Real cartesian_volume;

    AMREX_GPU_HOST_DEVICE amrex::Real Radius (int i) const noexcept
    {
        return lower + (i - first + amrex::Real(0.5)) * dx[0];
    }
    AMREX_GPU_HOST_DEVICE amrex::Real Volume (int i) const noexcept
    {
#if defined(WARPX_DIM_RZ)
        return 2 * MathConst::pi * Radius(i) * dx[0] * dx[1];
#else
        amrex::ignore_unused(i);
        return cartesian_volume;
#endif
    }
    AMREX_GPU_HOST_DEVICE amrex::Real Area (int d, int i) const noexcept
    {
#if defined(WARPX_DIM_RZ)
        if (d == 0) {
            return 2 * MathConst::pi * (lower + (i - first) * dx[0]) * dx[1];
        }
#endif
        return Volume(i) / dx[d];
    }
    AMREX_GPU_HOST_DEVICE bool Wall (int d, int i, int j, int k) const noexcept
    {
        int const coordinates[3] = {i, j, k};
        int const face = coordinates[d];
        return periodic[d] == 0 &&
               (face == domain_lo[d] || face == domain_hi[d] + 1);
    }
    AMREX_GPU_HOST_DEVICE amrex::Real MeanRadius (int i) const noexcept
    {
        auto const center = Radius(i);
        return center + dx[0] * dx[0] / (12 * center);
    }
    AMREX_GPU_HOST_DEVICE amrex::Real AngularFactor (int i) const noexcept
    {
        auto const center = Radius(i);
        auto const mean = MeanRadius(i);
        return (center * center + dx[0] * dx[0] / 4) / (mean * mean);
    }
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
physicalDirection (int d) noexcept
{
#if defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(d);
    return 2;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    return d == 0 ? 0 : 2;
#else
    return d;
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
shiftIndex (int& i, int& j, int& k, int d, int s) noexcept
{
    int idx[3] = {i, j, k};
    idx[d] += s;
    i = idx[0];
    j = idx[1];
    k = idx[2];
}

bool
collectiveValid (bool local)
{
    int invalid = local ? 0 : 1;
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    return invalid == 0;
}

int
localFlag (int value, amrex::FabArrayBase const& field)
{
    return field.local_size() == 0 ? 0 : value;
}

amrex::Real
localReal (amrex::Real value, amrex::FabArrayBase const& field)
{
    return field.local_size() == 0 ? 0 : value;
}

void
checkFaceAliases (amrex::MultiFab& faces, amrex::Geometry const& geometry, bool& valid)
{
    amrex::MultiFab canonical(faces.boxArray(), faces.DistributionMap(), 1, 0);
    amrex::MultiFab::Copy(canonical, faces, 0, 0, 1, 0);
    canonical.OverrideSync(geometry.periodicity());
    amrex::MultiFab mismatch(faces.boxArray(), faces.DistributionMap(), 1, 0);
    amrex::MultiFab::Copy(mismatch, canonical, 0, 0, 1, 0);
    amrex::MultiFab::Subtract(mismatch, faces, 0, 0, 1, 0);
    valid = valid && !mismatch.contains_nan() && !mismatch.contains_inf() && mismatch.norm0() == 0;
    faces = std::move(canonical);
}
} // namespace

MovingAsymptoticDiffusionFieldResult
TryAdvanceMovingAsymptoticDiffusionField (amrex::MultiFab const& lab_energy,
                                          amrex::MultiFab const& opacity,
                                          amrex::iMultiFab const& diffusion_mask,
                                          amrex::MultiFab const& work_velocity,
                                          amrex::Geometry const& geometry, amrex::Real dt,
                                          amrex::MultiFab const& requested_lab_work,
                                          AsymptoticDiffusionFieldOptions const& options,
                                          bool allow_kinetic_boundaries)
{
    MovingAsymptoticDiffusionFieldResult result;
    auto fail = [&result] (MovingAsymptoticDiffusionFieldFailure reason)
    {
        result.failure = reason;
        result.valid = false;
        return std::move(result);
    };

#if defined(AMREX_USE_FLOAT)
    return fail(MovingAsymptoticDiffusionFieldFailure::InvalidState);
#endif
#if defined(WARPX_DIM_RZ)
    bool const geometry_supported = (geometry.Coord() == 0 || geometry.Coord() == 1) &&
        !geometry.isPeriodic(0) &&
        std::isfinite(geometry.ProbLo(0)) && geometry.ProbLo(0) >= 0;
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    constexpr bool geometry_supported = false;
#else
    bool const geometry_supported = geometry.Coord() == 0;
#endif

    auto const& boxes = lab_energy.boxArray();
    auto const& distribution = lab_energy.DistributionMap();
    bool layout_valid =
        geometry_supported && boxes.ok() &&
        boxes.isDisjoint() && boxes.numPts() == geometry.Domain().numPts() &&
        lab_energy.nComp() == 1 && lab_energy.ixType().cellCentered() && opacity.nComp() == 1 &&
        opacity.ixType().cellCentered() && opacity.boxArray() == boxes &&
        opacity.DistributionMap() == distribution && diffusion_mask.nComp() == 1 &&
        diffusion_mask.ixType().cellCentered() && diffusion_mask.boxArray() == boxes &&
        diffusion_mask.DistributionMap() == distribution && work_velocity.nComp() == 3 &&
        work_velocity.ixType().cellCentered() && work_velocity.boxArray() == boxes &&
        work_velocity.DistributionMap() == distribution && requested_lab_work.nComp() == 1 &&
        requested_lab_work.ixType().cellCentered() && requested_lab_work.boxArray() == boxes &&
        requested_lab_work.DistributionMap() == distribution && std::isfinite(dt) && dt > 0 &&
        std::isfinite(PhysConst::c * dt) && PhysConst::c * dt > 0 &&
        options.max_linear_iterations > 0 && std::isfinite(options.linear_tolerance) &&
        options.linear_tolerance > 0 && options.linear_tolerance <= 1.e-11_rt &&
        std::isfinite(options.residual_tolerance) && options.residual_tolerance > 0 &&
        options.residual_tolerance <= 1.e-11_rt &&
        std::isfinite(options.marshak_extrapolation_lambda) &&
        options.marshak_extrapolation_lambda > 0;
    for (int n = 0; n < boxes.size(); ++n)
    {
        layout_valid = layout_valid && geometry.Domain().contains(boxes[n]);
    }
    if (!collectiveValid(layout_valid))
    {
        return fail(geometry_supported ? MovingAsymptoticDiffusionFieldFailure::InvalidState
                                    : MovingAsymptoticDiffusionFieldFailure::UnsupportedGeometry);
    }

    amrex::Real minimum_options[6] = {dt, options.linear_tolerance, options.residual_tolerance,
                                      options.marshak_extrapolation_lambda,
                                      static_cast<amrex::Real>(options.max_linear_iterations),
                                      allow_kinetic_boundaries ? amrex::Real(1) : amrex::Real(0)};
    amrex::Real maximum_options[6];
    for (int n = 0; n < 6; ++n)
    {
        maximum_options[n] = minimum_options[n];
    }
    amrex::ParallelDescriptor::ReduceRealMin(minimum_options, 6);
    amrex::ParallelDescriptor::ReduceRealMax(maximum_options, 6);
    for (int n = 0; n < 6; ++n)
    {
        if (minimum_options[n] != maximum_options[n])
        {
            return fail(MovingAsymptoticDiffusionFieldFailure::InvalidState);
        }
    }

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = geometry.CellSizeArray();
    amrex::Real volume = 1;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        if (!(dx[d] > 0) || !amrex::Math::isfinite(dx[d]))
        {
            volume = 0;
        }
        volume *= dx[d];
    }
    if (!collectiveValid(volume > 0 && amrex::Math::isfinite(volume)))
    {
        return fail(MovingAsymptoticDiffusionFieldFailure::InvalidState);
    }
    auto const geometry_periodic = geometry.isPeriodicArray();
    amrex::GpuArray<int, AMREX_SPACEDIM> domain_lo{};
    amrex::GpuArray<int, AMREX_SPACEDIM> domain_hi{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        domain_lo[d] = geometry.Domain().smallEnd(d);
        domain_hi[d] = geometry.Domain().bigEnd(d);
    }
    MovingFieldMetric const metric{dx, geometry_periodic, domain_lo, domain_hi,
                                   geometry.ProbLo(0),
                                   geometry.Domain().smallEnd(0),
                                   geometry.Domain().bigEnd(0), volume};
    if (!collectiveValid(metric.Volume(metric.last) > 0 &&
                         std::isfinite(metric.Volume(metric.last)))) {
        return fail(MovingAsymptoticDiffusionFieldFailure::InvalidState);
    }

    amrex::MultiFab energy_old(boxes, distribution, 1, 1);
    amrex::MultiFab chi(boxes, distribution, 1, 1);
    amrex::iMultiFab mask(boxes, distribution, 1, 1);
    amrex::MultiFab velocity(boxes, distribution, 3, 1);
    amrex::MultiFab work(boxes, distribution, 1, 0);
    energy_old.setVal(0);
    chi.setVal(0);
    mask.setVal(0);
    velocity.setVal(0);
    work.setVal(0);
    amrex::MultiFab::Copy(energy_old, lab_energy, 0, 0, 1, 0);
    amrex::MultiFab::Copy(chi, opacity, 0, 0, 1, 0);
    amrex::MultiFab::Copy(velocity, work_velocity, 0, 0, 3, 0);
    amrex::MultiFab::Copy(work, requested_lab_work, 0, 0, 1, 0);
    for (amrex::MFIter mfi(mask); mfi.isValid(); ++mfi)
    {
        auto const src = diffusion_mask.const_array(mfi);
        auto const dst = mask.array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               dst(i, j, k) = src(i, j, k);
                           });
    }
    energy_old.FillBoundary(geometry.periodicity());
    chi.FillBoundary(geometry.periodicity());
    mask.FillBoundary(geometry.periodicity());
    velocity.FillBoundary(geometry.periodicity());

    amrex::Real const c_light = PhysConst::c;
    amrex::Real const speed_limit = 1.e-2_rt * c_light;
    amrex::ReduceOps<amrex::ReduceOpMax> input_ops;
    amrex::ReduceData<int> input_data(input_ops);
    for (amrex::MFIter mfi(energy_old); mfi.isValid(); ++mfi)
    {
        auto const e = energy_old.const_array(mfi);
        auto const alpha = chi.const_array(mfi);
        auto const m = mask.const_array(mfi);
        auto const v = velocity.const_array(mfi);
        auto const w = work.const_array(mfi);
        input_ops.eval(mfi.validbox(), input_data,
                       [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                       {
                           amrex::Real speed2 = 0;
                           bool const kinetic = allow_kinetic_boundaries && m(i,j,k) == 0;
                           bool bad = (!kinetic && m(i,j,k) != 1) ||
                                      !amrex::Math::isfinite(e(i,j,k)) || e(i,j,k) < 0 ||
                                      !amrex::Math::isfinite(alpha(i,j,k)) ||
                                      !amrex::Math::isfinite(w(i,j,k)) ||
                                      (kinetic ? (e(i,j,k) != 0 || alpha(i,j,k) != 0 ||
                                                  w(i,j,k) != 0) : !(alpha(i,j,k) > 0));
                           for (int p = 0; p < 3; ++p)
                           {
                               auto component = v(i, j, k, p);
#if defined(WARPX_DIM_RZ)
                               if (p == 1) {
                                   component *= (metric.Radius(i) + dx[0] / 2) /
                                                metric.MeanRadius(i);
                               }
#endif
                               speed2 += component * component;
                               bad = bad || !amrex::Math::isfinite(component);
                           }
                           bad = bad || !(speed2 <= speed_limit * speed_limit);
                           return {bad ? 1 : 0};
                       });
    }
    if (!collectiveValid(localFlag(amrex::get<0>(input_data.value()), energy_old) == 0))
    {
        return fail(MovingAsymptoticDiffusionFieldFailure::InvalidState);
    }

    amrex::ReduceOps<amrex::ReduceOpMax> cfl_ops;
    amrex::ReduceData<amrex::Real> cfl_data(cfl_ops);
    for (amrex::MFIter mfi(energy_old); mfi.isValid(); ++mfi)
    {
        auto const v = velocity.const_array(mfi);
        auto const region = mask.const_array(mfi);
        cfl_ops.eval(mfi.validbox(), cfl_data,
                     [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<amrex::Real>
                     {
                         if (allow_kinetic_boundaries && region(i,j,k) == 0) { return {0}; }
                         amrex::Real outgoing = 0;
                         for (int d = 0; d < AMREX_SPACEDIM; ++d)
                         {
                             int const p = physicalDirection(d);
                             int li = i, lj = j, lk = k;
                             int hi = i, hj = j, hk = k;
                             shiftIndex(li, lj, lk, d, -1);
                             shiftIndex(hi, hj, hk, d, 1);
                             amrex::Real const vlo = metric.Wall(d, i, j, k) ? 0 :
                                 amrex::Real(0.5) * (v(li, lj, lk, p) + v(i, j, k, p));
                             amrex::Real const vhi = metric.Wall(d, hi, hj, hk) ? 0 :
                                 amrex::Real(0.5) * (v(i, j, k, p) + v(hi, hj, hk, p));
                             outgoing += metric.Area(d, i) * amrex::max(-vlo, amrex::Real(0)) +
                                         metric.Area(d, hi) * amrex::max(vhi, amrex::Real(0));
                         }
                         return {dt / metric.Volume(i) * (amrex::Real(4) / 3) * outgoing};
                     });
    }
    amrex::Real max_cfl = localReal(amrex::get<0>(cfl_data.value()), energy_old);
    amrex::ParallelDescriptor::ReduceRealMax(max_cfl);
    if (!std::isfinite(max_cfl) || max_cfl > 1)
    {
        return fail(MovingAsymptoticDiffusionFieldFailure::EnthalpyCfl);
    }

    Faces enthalpy;
    amrex::ReduceOps<amrex::ReduceOpMax> face_ops;
    amrex::ReduceData<int> face_data(face_ops);
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        auto const faces = amrex::convert(boxes, amrex::IntVect::TheDimensionVector(d));
        enthalpy[d] = std::make_unique<amrex::MultiFab>(faces, distribution, 1, 0);
        enthalpy[d]->setVal(0);
        for (amrex::MFIter mfi(*enthalpy[d]); mfi.isValid(); ++mfi)
        {
            auto const e = energy_old.const_array(mfi);
            auto const v = velocity.const_array(mfi);
            auto const h = enthalpy[d]->array(mfi);
            face_ops.eval(mfi.validbox(), face_data,
                          [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                          {
                              if (metric.Wall(d, i, j, k)) {
                                  h(i, j, k) = 0;
                                  return {0};
                              }
                              int const p = physicalDirection(d);
                              int li = i, lj = j, lk = k;
                              shiftIndex(li, lj, lk, d, -1);
                              amrex::Real const face_velocity =
                                  amrex::Real(0.5) * (v(li, lj, lk, p) + v(i, j, k, p));
                              amrex::Real const e_upwind =
                                  face_velocity >= 0 ? e(li, lj, lk) / metric.Volume(li)
                                                     : e(i, j, k) / metric.Volume(i);
                              amrex::Real const current =
                                  (amrex::Real(4) / 3) * e_upwind * face_velocity;
                              h(i, j, k) = current;
                              bool const bad = !amrex::Math::isfinite(current);
                              return {bad ? 1 : 0};
                          });
        }
    }
    if (!collectiveValid(localFlag(amrex::get<0>(face_data.value()), chi) == 0))
    {
        return fail(MovingAsymptoticDiffusionFieldFailure::NonfiniteDiagnostic);
    }

    bool aliases_valid = true;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        checkFaceAliases(*enthalpy[d], geometry, aliases_valid);
    }
    if (!collectiveValid(aliases_valid))
    {
        return fail(MovingAsymptoticDiffusionFieldFailure::InvalidState);
    }

    amrex::MultiFab advected_energy(boxes, distribution, 1, 0);
    advected_energy.setVal(0);
    amrex::ReduceOps<amrex::ReduceOpMax> advect_ops;
    amrex::ReduceData<int> advect_data(advect_ops);
    for (amrex::MFIter mfi(advected_energy); mfi.isValid(); ++mfi)
    {
        auto const e = energy_old.const_array(mfi);
        auto const candidate = advected_energy.array(mfi);
        auto const region = mask.const_array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> h;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            h[d] = enthalpy[d]->const_array(mfi);
        }
        advect_ops.eval(mfi.validbox(), advect_data,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                        {
                            if (allow_kinetic_boundaries && region(i,j,k) == 0) {
                                candidate(i,j,k) = 0;
                                return {0};
                            }
                            amrex::Real divergence = 0;
                            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            {
                                int hi = i, hj = j, hk = k;
                                shiftIndex(hi, hj, hk, d, 1);
                                // The reported wall current is zero. Omit--rather
                                // than count--that zero in the cell budget so a
                                // reflecting wall returns upwind enthalpy instead
                                // of silently cooling the adjacent cell.
                                amrex::Real const low = metric.Wall(d, i, j, k) ? 0 :
                                    metric.Area(d, i) * h[d](i, j, k);
                                amrex::Real const high = metric.Wall(d, hi, hj, hk) ? 0 :
                                    metric.Area(d, hi) * h[d](hi, hj, hk);
                                divergence += high - low;
                            }
                            amrex::Real const value = e(i, j, k) - dt * divergence;
                            candidate(i, j, k) = value;
                            bool const bad = !amrex::Math::isfinite(value) || value < 0 ||
                                             !amrex::Math::isfinite(divergence);
                            return {bad ? 1 : 0};
                        });
    }
    if (!collectiveValid(localFlag(amrex::get<0>(advect_data.value()), advected_energy) == 0))
    {
        return fail(MovingAsymptoticDiffusionFieldFailure::AdvectedState);
    }

    Faces admitted;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        auto const faces = amrex::convert(boxes, amrex::IntVect::TheDimensionVector(d));
        admitted[d] = std::make_unique<amrex::MultiFab>(faces, distribution, 1, 0);
        admitted[d]->setVal(0);
    }
    auto const admitted_ptrs = amrex::GetArrOfConstPtrs(admitted);
    auto field = TryAdvanceAsymptoticDiffusionField(advected_energy, chi, mask, geometry, dt, work,
                                                    admitted_ptrs, nullptr, options);
    result.linear_iterations = field.linear_iterations;
    if (!field.valid)
    {
        result.diffusion_failure = field.failure;
        return fail(MovingAsymptoticDiffusionFieldFailure::DiffusionFailure);
    }

    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum>
        wall_ops;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real> wall_data(wall_ops);
    int const reflecting = static_cast<int>(DiffusionBoundary::Reflecting);
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        if (geometry.isPeriodic(d)) { continue; }
        for (amrex::MFIter mfi(*field.face_pressure[d]); mfi.isValid(); ++mfi)
        {
            auto const pressure = field.face_pressure[d]->const_array(mfi);
            auto const box = mfi.validbox();
            wall_ops.eval(
                box, wall_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real>
                {
                    int const face[3] = {i, j, k};
                    int const axis = face[d];
                    int side = 0;
                    if (axis == metric.domain_lo[d])
                    {
                        side = -1;
                    }
                    else if (axis == metric.domain_hi[d] + 1)
                    {
                        side = 1;
                    }
                    if (side == 0) { return {0, 0, 0}; }
                    int const p = physicalDirection(d);
                    amrex::Real const value = DiffusionBoundaryWallMomentum(
                        reflecting, pressure(i, j, k), metric.Area(d, i), dt, side);
                    amrex::Real component[3] = {0, 0, 0};
                    component[p] = value;
                    return {component[0], component[1], component[2]};
                });
        }
    }
    auto const wall_sum = wall_data.value();
    result.radiation_wall_momentum = {
        amrex::get<0>(wall_sum), amrex::get<1>(wall_sum), amrex::get<2>(wall_sum)};
    amrex::ParallelDescriptor::ReduceRealSum(
        result.radiation_wall_momentum.data(), 3);
    bool wall_valid = true;
    for (int d = 0; d < 3; ++d)
    {
        wall_valid = wall_valid && std::isfinite(result.radiation_wall_momentum[d]);
    }
    if (!collectiveValid(wall_valid))
    {
        return fail(MovingAsymptoticDiffusionFieldFailure::NonfiniteDiagnostic);
    }

    Faces lab_current;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        lab_current[d] =
            std::make_unique<amrex::MultiFab>(enthalpy[d]->boxArray(), distribution, 1, 0);
        lab_current[d]->setVal(0);
        for (amrex::MFIter mfi(*lab_current[d]); mfi.isValid(); ++mfi)
        {
            auto const f0 = field.face_current[d]->const_array(mfi);
            auto const h = enthalpy[d]->const_array(mfi);
            auto const total = lab_current[d]->array(mfi);
            auto const region = mask.const_array(mfi);
            auto const leaked = field.face_leaked_energy[d]->array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   total(i, j, k) = metric.Wall(d, i, j, k)
                                       ? DiffusionBoundaryNormalEnergyFlux(
                                           reflecting, f0(i, j, k) + h(i, j, k))
                                       : f0(i, j, k) + h(i, j, k);
                                   if (allow_kinetic_boundaries && !metric.Wall(d, i, j, k)) {
                                       int li = i, lj = j, lk = k;
                                       shiftIndex(li, lj, lk, d, -1);
                                       int const sign = region(li,lj,lk)-region(i,j,k);
                                       if (sign != 0) {
                                           // Upwind kinetic scalar energy is zero, so this
                                           // increment is outward or zero, never admission.
                                           amrex::Real const outgoing = sign*h(i,j,k);
                                           leaked(i,j,k) = outgoing >= 0
                                               ? leaked(i,j,k)+dt*metric.Area(d,i)*outgoing
                                               : std::numeric_limits<amrex::Real>::quiet_NaN();
                                       }
                                   }
                               });
        }
        checkFaceAliases(*lab_current[d], geometry, aliases_valid);
        if (allow_kinetic_boundaries) {
            checkFaceAliases(*field.face_leaked_energy[d], geometry, aliases_valid);
            bool const nonnegative_leakage = field.face_leaked_energy[d]->min(0) >= 0;
            aliases_valid = nonnegative_leakage && aliases_valid;
        }
    }
    if (!collectiveValid(aliases_valid))
    {
        return fail(MovingAsymptoticDiffusionFieldFailure::InvalidState);
    }

    amrex::MultiFab residual(boxes, distribution, 1, 0);
    amrex::MultiFab inertia(boxes, distribution, 3, 0);
    residual.setVal(0);
    inertia.setVal(0);
    amrex::Real const eps = std::numeric_limits<amrex::Real>::epsilon();
    amrex::Real const residual_tol = options.residual_tolerance;
    amrex::Real const linear_residual = field.linear_residual;
    amrex::Real const linear_energy_scale = metric.Volume(metric.last);
    amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpSum> ledger_ops;
    amrex::ReduceData<amrex::Real, int, amrex::Real> ledger_data(ledger_ops);
    for (amrex::MFIter mfi(residual); mfi.isValid(); ++mfi)
    {
        auto const e0 = energy_old.const_array(mfi);
        auto const en = field.energy.const_array(mfi);
        auto const v = velocity.const_array(mfi);
        auto const w = work.const_array(mfi);
        auto const res = residual.array(mfi);
        auto const region = mask.const_array(mfi);
        auto const q = inertia.array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> flux;
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> diffusive;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            flux[d] = lab_current[d]->const_array(mfi);
            diffusive[d] = field.face_current[d]->const_array(mfi);
        }
        ledger_ops.eval(
            mfi.validbox(), ledger_data,
            [=] AMREX_GPU_DEVICE(int i, int j,
                                 int k) -> amrex::GpuTuple<amrex::Real, int, amrex::Real>
            {
                if (allow_kinetic_boundaries && region(i,j,k) == 0) {
                    res(i,j,k) = 0;
                    for (int p = 0; p < 3; ++p) { q(i,j,k,p) = 0; }
                    return {0, 0, 0};
                }
                amrex::Real divergence = 0;
                amrex::Real term_scale = std::abs(en(i, j, k)) +
                    std::abs(e0(i, j, k)) + std::abs(w(i, j, k));
                amrex::GpuArray<amrex::Real, 3> f0_cell{};
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    int hi = i, hj = j, hk = k;
                    shiftIndex(hi, hj, hk, d, 1);
                    amrex::Real const q_low = metric.Wall(d, i, j, k) ? 0 :
                        metric.Area(d, i) * flux[d](i, j, k);
                    amrex::Real const q_high = metric.Wall(d, hi, hj, hk) ? 0 :
                        metric.Area(d, hi) * flux[d](hi, hj, hk);
                    divergence += q_high - q_low;
                    term_scale += dt * (std::abs(q_low) + std::abs(q_high));
                    int const p = physicalDirection(d);
                    f0_cell[p] =
                        amrex::Real(0.5) * (diffusive[d](i, j, k) + diffusive[d](hi, hj, hk));
                }
                amrex::Real const value = en(i, j, k) - e0(i, j, k) + dt * divergence + w(i, j, k);
                res(i, j, k) = value;
                for (int p = 0; p < 3; ++p)
                {
                    q(i, j, k, p) = metric.Volume(i) *
                                    (f0_cell[p] + (amrex::Real(4) / 3) * (en(i, j, k) / metric.Volume(i)) *
                                                      v(i, j, k, p)) /
                                    c_light;
#if defined(WARPX_DIM_RZ)
                    if (p == 1) { q(i, j, k, p) *= metric.AngularFactor(i); }
#endif
                }
                amrex::Real const roundoff =
                    amrex::Real(512) * eps * (term_scale + std::abs(value));
                amrex::Real const allowed = amrex::max(
                    residual_tol * term_scale,
                    amrex::max(roundoff, 2 * linear_residual * linear_energy_scale));
                bool bad = !amrex::Math::isfinite(value) || std::abs(value) > allowed;
                for (int p = 0; p < 3; ++p)
                {
                    bad = bad || !amrex::Math::isfinite(q(i, j, k, p));
                }
                return {std::abs(value), bad ? 1 : 0, value};
            });
    }
    auto const reduced = ledger_data.value();
    amrex::Real max_abs = localReal(amrex::get<0>(reduced), field.energy);
    int bad_ledger = localFlag(amrex::get<1>(reduced), field.energy);
    amrex::Real global = localReal(amrex::get<2>(reduced), field.energy);
    amrex::ParallelDescriptor::ReduceRealMax(max_abs);
    amrex::ParallelDescriptor::ReduceIntMax(bad_ledger);
    amrex::ParallelDescriptor::ReduceRealSum(global);
    if (bad_ledger != 0 || !std::isfinite(max_abs) || !std::isfinite(global))
    {
        return fail(MovingAsymptoticDiffusionFieldFailure::LabEnergyResidual);
    }

    result.diffusion = std::move(field);
    result.lab_energy_balance_residual = std::move(residual);
    result.lab_radiation_inertia = std::move(inertia);
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        result.face_enthalpy_current[d] = std::move(enthalpy[d]);
        result.face_lab_energy_current[d] = std::move(lab_current[d]);
    }
    result.global_lab_energy_balance_residual = global;
    result.max_lab_energy_balance_residual = max_abs;
    result.valid = true;
    result.failure = MovingAsymptoticDiffusionFieldFailure::None;
    return result;
}
} // namespace warpx::radiation
