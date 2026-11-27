/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "RZPressureInduction.H"

#include "RZPressureWork.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <array>
#include <cmath>

namespace warpx::hybrid
{
using namespace amrex::literals;

namespace
{
// Each temporary has one radial ghost. Periodic exchange also resolves nodal
// duplicates before physical reflection; inputs never supply trusted ghosts.
void
fillGhosts (amrex::MultiFab& field, amrex::Geometry const& geometry, bool electric, bool magnetic)
{
    field.OverrideSync(geometry.periodicity());
    field.FillBoundary(geometry.periodicity());
    auto const domain = amrex::convert(geometry.Domain(), field.ixType());
    int const axis = domain.smallEnd(0), outer = domain.bigEnd(0);
    int const components = field.nComp();
    for (amrex::MFIter it(field); it.isValid(); ++it)
    {
        auto const data = field.array(it);
        amrex::ParallelFor(it.fabbox(), components,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                           {
                               if (i >= axis && i <= outer)
                               {
                                   return;
                               }
                               bool const low = i < axis;
                               int const source = low ? 2 * axis - i : 2 * outer - i;
                               bool const odd =
                                   low ? (magnetic || (electric && n == 0)) : (electric && n == 1);
                               data(i, j, k, n) = (odd ? -1._rt : 1._rt) * data(source, j, k, n);
                           });
    }
}
} // namespace

bool
tryAdvanceRZPressureInduction (amrex::MultiFab& magnetic_theta, amrex::MultiFab& internal_energy,
                               amrex::MultiFab const& pressure, amrex::MultiFab const& charge,
                               amrex::MultiFab const& caloric_volume,
                               amrex::Geometry const& geometry, amrex::Real dt)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        &magnetic_theta != &internal_energy && geometry.Coord() == 1 && geometry.ProbLo(0) == 0 &&
            !geometry.isPeriodic(0) && geometry.isPeriodic(1) &&
            magnetic_theta.ixType().nodeCentered() && std::isfinite(dt) && dt > 0,
        "RZ pressure induction requires an axis/PEC, periodic-z nodal domain and positive dt.");
    std::array<amrex::MultiFab const*, 5> const fields{&magnetic_theta, &internal_energy, &pressure,
                                                       &charge, &caloric_volume};
    for (auto const* field : fields)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            field->nComp() == 1 && field->boxArray() == magnetic_theta.boxArray() &&
                field->DistributionMap() == magnetic_theta.DistributionMap(),
            "RZ pressure-induction fields must have matching scalar nodal layouts.");
    }
    auto const& boxes = magnetic_theta.boxArray();
    auto const& mapping = magnetic_theta.DistributionMap();
    amrex::MultiFab state(boxes, mapping, 2, 1), electric(boxes, mapping, 2, 1);
    amrex::MultiFab next_b(boxes, mapping, 1, 1), midpoint(boxes, mapping, 1, 1);
    amrex::MultiFab work_current(boxes, mapping, 2, 1), next_energy(boxes, mapping, 1, 0);
    auto const domain = amrex::convert(geometry.Domain(), magnetic_theta.ixType());
    int const axis = domain.smallEnd(0), outer = domain.bigEnd(0);
    int const zlo = domain.smallEnd(1), zhi = domain.bigEnd(1);
    auto const dr = geometry.CellSize(0), dz = geometry.CellSize(1);

    amrex::ReduceOps<amrex::ReduceOpMax> input_ops;
    amrex::ReduceData<int> input_data(input_ops);
    for (amrex::MFIter it(state); it.isValid(); ++it)
    {
        auto const p = pressure.const_array(it), rho = charge.const_array(it);
        auto const u = internal_energy.const_array(it), b = magnetic_theta.const_array(it);
        auto const volume = caloric_volume.const_array(it);
        input_ops.eval(it.validbox(), input_data,
                       [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                       {
                           return {!std::isfinite(p(i, j, k)) || p(i, j, k) < 0 ||
                                   !std::isfinite(rho(i, j, k)) || !(rho(i, j, k) > 0) ||
                                   !std::isfinite(u(i, j, k)) || !(u(i, j, k) > 0) ||
                                   !std::isfinite(volume(i, j, k)) || !(volume(i, j, k) > 0) ||
                                   !std::isfinite(b(i, j, k)) || (i == axis && b(i, j, k) != 0)};
                       });
    }
    int invalid = amrex::get<0>(input_data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    if (invalid != 0)
    {
        return false;
    }
    for (amrex::MFIter it(state); it.isValid(); ++it)
    {
        auto const p = pressure.const_array(it), rho = charge.const_array(it);
        auto const output = state.array(it);
        amrex::ParallelFor(it.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               output(i, j, k, 0) = p(i, j, k);
                               output(i, j, k, 1) = 1 / rho(i, j, k);
                           });
    }
    fillGhosts(state, geometry, false, false);
    for (amrex::MFIter it(electric); it.isValid(); ++it)
    {
        auto const s = state.const_array(it);
        auto const e = electric.array(it);
        amrex::ParallelFor(
            it.validbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                e(i, j, k, 0) = -s(i, j, k, 1) *
                                rzPressureGradient(s, i, j, 0, axis, outer, zlo, zhi, true, 1 / dr);
                e(i, j, k, 1) = -s(i, j, k, 1) *
                                rzPressureGradient(s, i, j, 2, axis, outer, zlo, zhi, true, 1 / dz);
            });
    }
    fillGhosts(electric, geometry, true, false);
    for (amrex::MFIter it(next_b); it.isValid(); ++it)
    {
        auto const e = electric.const_array(it), old = magnetic_theta.const_array(it);
        auto const next = next_b.array(it), mid = midpoint.array(it);
        amrex::ParallelFor(it.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               auto const change =
                                   i == axis
                                       ? 0
                                       : dt * ((e(i + 1, j, k, 1) - e(i - 1, j, k, 1)) / (2 * dr) -
                                               (e(i, j + 1, k, 0) - e(i, j - 1, k, 0)) / (2 * dz));
                               next(i, j, k) = old(i, j, k) + change;
                               mid(i, j, k) = old(i, j, k) + .5_rt * change;
                           });
    }
    fillGhosts(midpoint, geometry, false, true);
    for (amrex::MFIter it(work_current); it.isValid(); ++it)
    {
        auto const b = midpoint.const_array(it);
        auto const current = work_current.array(it);
        amrex::ParallelFor(
            it.validbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                // FieldEnergy's actual nodal magnetic quadrature, not the electron
                // control volume. The axis B_theta row is identically zero.
                auto const volume = [=] AMREX_GPU_HOST_DEVICE(int r)
                {
                    return 2 * MathConst::pi * (r - axis) * dr * dr * dz *
                           (r == outer ? .5_rt : 1._rt);
                };
                current(i, j, k, 0) = i == axis ? 0
                                                : -volume(i) * (b(i, j + 1, k) - b(i, j - 1, k)) /
                                                      (2 * dz * PhysConst::mu0);
                auto const lower = i > axis + 1 ? volume(i - 1) * b(i - 1, j, k) : 0;
                auto const upper =
                    i < outer ? volume(i + 1) * b(i + 1, j, k) * (i + 1 == outer ? 2._rt : 1._rt)
                              : 0;
                current(i, j, k, 1) = i == outer ? 0 : (upper - lower) / (2 * dr * PhysConst::mu0);
            });
    }
    fillGhosts(work_current, geometry, false, false);
    amrex::ReduceOps<amrex::ReduceOpMax> candidate_ops;
    amrex::ReduceData<int> candidate_data(candidate_ops);
    for (amrex::MFIter it(next_energy); it.isValid(); ++it)
    {
        auto const s = state.const_array(it), current = work_current.const_array(it);
        auto const old = internal_energy.const_array(it), volume = caloric_volume.const_array(it);
        auto const next = next_energy.array(it);
        auto const b = next_b.const_array(it);
        candidate_ops.eval(it.validbox(), candidate_data,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                           {
                               // The axial component lives in component one; use an offset view.
                               auto const axial = amrex::Array4<amrex::Real const>(current, 1);
                               auto const paired = rzPressureWorkDivergence(
                                   current, axial, s, i, j, axis, outer, zlo, zhi, true, 1 / dr,
                                   1 / dz, volume(i, j, k));
                               next(i, j, k) = old(i, j, k) + dt * s(i, j, k, 0) * paired;
                               return {!std::isfinite(next(i, j, k)) || !(next(i, j, k) > 0) ||
                                       !std::isfinite(b(i, j, k))};
                           });
    }
    invalid = amrex::get<0>(candidate_data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    if (invalid != 0)
    {
        return false;
    }
    amrex::MultiFab::Copy(magnetic_theta, next_b, 0, 0, 1, 0);
    amrex::MultiFab::Copy(internal_energy, next_energy, 0, 0, 1, 0);
    return true;
}
} // namespace warpx::hybrid
