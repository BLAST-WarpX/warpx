/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "RZMomentGeometry.H"

#include "MomentClosure.H"

#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>

#if defined(WARPX_DIM_RZ)
namespace warpx::radiation
{
    bool TryCanonicalizeRZMeridionalVelocity (amrex::MultiFab& beta)
    {
        AMREX_ALWAYS_ASSERT(beta.nComp() == 3 && beta.ixType().cellCentered());
        amrex::ReduceOps<amrex::ReduceOpMax> ops;
        amrex::ReduceData<int> data(ops);
        using Tuple = typename decltype(data)::Type;
        for (amrex::MFIter it(beta); it.isValid(); ++it) {
            auto const b = beta.const_array(it);
            ops.eval(it.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                auto const scale = std::abs(b(i,j,k,0)) + std::abs(b(i,j,k,2));
                auto const allowance = 64 * std::numeric_limits<amrex::Real>::epsilon() * scale;
                return {!std::isfinite(scale) || !std::isfinite(b(i,j,k,1))
                    || std::abs(b(i,j,k,1)) > allowance};
            });
        }
        int invalid = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceIntMax(invalid);
        if (invalid) { return false; }
        beta.setVal(0, 1, 1, 0);
        return true;
    }

void
FillRZMomentGhosts (amrex::MultiFab& field, amrex::Geometry const& geometry, int vector_offset)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        field.ixType().cellCentered() && field.nGrowVect() == amrex::IntVect(1) &&
            (vector_offset == 0 || vector_offset == 1) && field.nComp() == vector_offset + 3 &&
            geometry.ProbLo(0) == 0 && !geometry.isPeriodic(0),
        "RZ moment ghost extension requires an axis domain and one cell-centered ghost layer.");
    field.FillBoundary(geometry.periodicity());
    auto const lo = amrex::lbound(geometry.Domain());
    auto const hi = amrex::ubound(geometry.Domain());
    bool const periodic_z = geometry.isPeriodic(1);
    for (amrex::MFIter it(field); it.isValid(); ++it)
    {
        auto const values = field.array(it);
        amrex::ParallelFor(it.fabbox(), field.nComp(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k, int component)
                           {
                               int ii = i, jj = j;
                               int sign = 1;
                               if (i < lo.x)
                               {
                                   ii = 2 * lo.x - i - 1;
                                   if (component == vector_offset || component == vector_offset + 1)
                                   {
                                       sign = -sign;
                                   }
                               }
                               else if (i > hi.x)
                               {
                                   ii = 2 * hi.x - i + 1;
                                   if (component == vector_offset)
                                   {
                                       sign = -sign;
                                   }
                               }
                               if (!periodic_z && (j < lo.y || j > hi.y))
                               {
                                   jj = j < lo.y ? 2 * lo.y - j - 1 : 2 * hi.y - j + 1;
                                   if (component == vector_offset + 2)
                                   {
                                       sign = -sign;
                                   }
                               }
                               if (ii != i || jj != j)
                               {
                                   values(i, j, k, component) = sign * values(ii, jj, k, component);
                               }
                           });
    }
}

void
PrepareRZMomentDensity (amrex::MultiFab& density, amrex::MultiFab const& integrated,
                        amrex::Geometry const& geometry)
{
    RZMomentMetric const metric(geometry);
    for (amrex::MFIter it(density); it.isValid(); ++it)
    {
        auto const source = integrated.const_array(it);
        auto const output = density.array(it);
        amrex::ParallelFor(
            it.validbox(), 4, [=] AMREX_GPU_DEVICE(int i, int j, int k, int component)
            { output(i, j, k, component) = source(i, j, k, component) / metric.Volume(i); });
    }
    FillRZMomentGhosts(density, geometry, 1);
}

void
PrepareRZMomentFluxState (amrex::MultiFab& density, amrex::MultiFab& theta_pressure,
                          amrex::MultiFab const& integrated,
                          amrex::MultiFab const& pressure_jacobian, amrex::Geometry const& geometry,
                          bool physical)
{
    PrepareRZMomentDensity(density, integrated, geometry);
    for (amrex::MFIter it(theta_pressure); it.isValid(); ++it)
    {
        auto const u = density.const_array(it);
        auto const jacobian = pressure_jacobian.const_array(it);
        auto const output = theta_pressure.array(it);
        amrex::ParallelFor(
            it.validbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                FourVector const values{u(i, j, k, 0), u(i, j, k, 1), u(i, j, k, 2), u(i, j, k, 3)};
                amrex::Real stress = 0, scale = 0;
                for (int column = 0; column < 4; ++column)
                {
                    auto const term = jacobian(i, j, k, 9 * column + 4) * values[column];
                    stress += term;
                    scale += std::abs(term);
                }
                output(i, j, k, 0) = physical ? EvaluateM1Closure(values).tensor[2][2] : stress;
                output(i, j, k, 1) = scale;
            });
    }
}

FourVector
RZGeometricExchange (amrex::MultiFab const& theta_pressure, amrex::Geometry const& geometry,
                     amrex::Real dt, amrex::MultiFab const* radial_flux)
{
    RZMomentMetric const metric(geometry);
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Real, amrex::Real> data(ops);
    using Tuple = typename decltype(data)::Type;
    for (amrex::MFIter it(theta_pressure); it.isValid(); ++it)
    {
        auto const pressure = theta_pressure.const_array(it);
        bool const angular = radial_flux != nullptr;
        auto const flux = angular ? radial_flux->const_array(it) : amrex::Array4<amrex::Real const>{};
        ops.eval(it.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple
                 {
                     return {dt * PhysConst::c * (metric.Area(0, i + 1) - metric.Area(0, i)) *
                                 pressure(i, j, k, 0),
                             angular ? metric.Angular(i).Geometric(flux(i, j, k, 2),
                                                                   flux(i + 1, j, k, 2), dt) : 0};
                 });
    }
    auto const values = data.value();
    amrex::Real sum[2]{amrex::get<0>(values), amrex::get<1>(values)};
    amrex::ParallelDescriptor::ReduceRealSum(sum, 2);
    return {0, sum[0], sum[1], 0};
}

amrex::Real
RZAngularBalance (amrex::MultiFab const& state, amrex::MultiFab const& old,
                  amrex::MultiFab const& transfer, amrex::Geometry const& geometry)
{
    RZMomentMetric const metric(geometry);
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Real, amrex::Real> data(ops);
    using Tuple = typename decltype(data)::Type;
    for (amrex::MFIter it(state); it.isValid(); ++it) {
        auto const current = state.const_array(it);
        auto const initial = old.const_array(it);
        auto const source = transfer.const_array(it);
        ops.eval(it.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
            auto const radius = metric.Angular(i).mean_radius;
            auto const a = current(i, j, k, 2), b = initial(i, j, k, 2), c = source(i, j, k, 2);
            return {radius * ((a - b) + c), radius * (std::abs(a) + std::abs(b) + std::abs(c))};
        });
    }
    auto const values = data.value();
    amrex::Real sum[2]{amrex::get<0>(values), amrex::get<1>(values)};
    amrex::ParallelDescriptor::ReduceRealSum(sum, 2);
    if (!std::isfinite(sum[0]) || !std::isfinite(sum[1])) {
        return std::numeric_limits<amrex::Real>::infinity();
    }
    return sum[1] > 0 ? std::abs(sum[0]) / sum[1] : std::abs(sum[0]);
}
} // namespace warpx::radiation
#endif
