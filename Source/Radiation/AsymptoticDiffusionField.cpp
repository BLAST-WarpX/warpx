/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "AsymptoticDiffusionField.H"

#include "Utils/WarpXConst.H"

#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>
#include <memory>

namespace warpx::radiation
{
using namespace amrex::literals;
namespace
{
using Faces = amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>;

/** Explicit finite-volume metrics, as in ImplicitDiffusion. MLMG itself uses
 * Cartesian coordinates so these factors are applied exactly once. */
struct FieldMetric
{
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx;
    amrex::Real lower;
    int first;
    amrex::Real cartesian_volume;

    AMREX_GPU_HOST_DEVICE amrex::Real Volume (int i) const noexcept
    {
#if defined(WARPX_DIM_RZ)
        auto const radius = lower + (i - first + amrex::Real(0.5)) * dx[0];
        return 2 * MathConst::pi * radius * dx[0] * dx[1];
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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
axisIndex (int i, int j, int k, int d) noexcept
{
    int const idx[3] = {i, j, k};
    return idx[d];
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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
isDiffusionCell (int mask) noexcept
{
    return mask == 1;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
isKineticCell (int mask) noexcept
{
    return mask == 0;
}
} // namespace

AsymptoticDiffusionFieldResult
TryAdvanceAsymptoticDiffusionField (
    amrex::MultiFab const& energy,
    amrex::MultiFab const& opacity,
    amrex::iMultiFab const& diffusion_mask,
    amrex::Geometry const& geometry,
    amrex::Real dt,
    amrex::MultiFab const& requested_work,
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> const& admitted_energy,
    amrex::MultiFab const* old_radiation_inertia,
    AsymptoticDiffusionFieldOptions const& options)
{
    AsymptoticDiffusionFieldResult result;
    auto fail = [&result] (AsymptoticDiffusionFieldFailure reason)
    {
        result.failure = reason;
        result.valid = false;
        return std::move(result);
    };

#if defined(AMREX_USE_FLOAT)
    return fail(AsymptoticDiffusionFieldFailure::InvalidState);
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

    auto const& boxes = energy.boxArray();
    auto const& distribution = energy.DistributionMap();
    auto const periodic = geometry.isPeriodicArray();
    bool layout_valid = geometry_supported && boxes.ok() &&
                        energy.nComp() == 1 && energy.ixType().cellCentered() &&
                        opacity.nComp() == 1 && opacity.ixType().cellCentered() &&
                        opacity.boxArray() == boxes &&
                        opacity.DistributionMap() == distribution &&
                        diffusion_mask.nComp() == 1 &&
                        diffusion_mask.ixType().cellCentered() &&
                        diffusion_mask.boxArray() == boxes &&
                        diffusion_mask.DistributionMap() == distribution &&
                        requested_work.nComp() == 1 &&
                        requested_work.ixType().cellCentered() &&
                        requested_work.boxArray() == boxes &&
                        requested_work.DistributionMap() == distribution &&
                        std::isfinite(dt) && dt > 0 &&
                        std::isfinite(PhysConst::c * dt) && PhysConst::c * dt > 0 &&
                        options.max_linear_iterations > 0 &&
                        std::isfinite(options.linear_tolerance) &&
                        options.linear_tolerance > 0 &&
                        options.linear_tolerance <= 1.e-11_rt &&
                        std::isfinite(options.residual_tolerance) &&
                        options.residual_tolerance > 0 &&
                        options.residual_tolerance <= 1.e-11_rt &&
                        std::isfinite(options.marshak_extrapolation_lambda) &&
                        options.marshak_extrapolation_lambda > 0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        auto const* faces = admitted_energy[d];
        auto const expected = amrex::convert(boxes, amrex::IntVect::TheDimensionVector(d));
        layout_valid = layout_valid && faces != nullptr && faces->nComp() == 1 &&
                       faces->ixType() == amrex::IndexType(amrex::IntVect::TheDimensionVector(d)) &&
                       faces->boxArray() == expected &&
                       faces->DistributionMap() == distribution;
    }
    if (old_radiation_inertia != nullptr)
    {
        layout_valid = layout_valid && old_radiation_inertia->nComp() == 3 &&
                       old_radiation_inertia->ixType().cellCentered() &&
                       old_radiation_inertia->boxArray() == boxes &&
                       old_radiation_inertia->DistributionMap() == distribution;
    }
    if (!collectiveValid(layout_valid))
    {
        return fail(geometry_supported ? AsymptoticDiffusionFieldFailure::InvalidState
                                    : AsymptoticDiffusionFieldFailure::UnsupportedGeometry);
    }

    amrex::Real minimum_options[5] = {dt, options.linear_tolerance,
        options.residual_tolerance, options.marshak_extrapolation_lambda,
        static_cast<amrex::Real>(options.max_linear_iterations)};
    amrex::Real maximum_options[5];
    for (int n = 0; n < 5; ++n) { maximum_options[n] = minimum_options[n]; }
    amrex::ParallelDescriptor::ReduceRealMin(minimum_options, 5);
    amrex::ParallelDescriptor::ReduceRealMax(maximum_options, 5);
    for (int n = 0; n < 5; ++n) {
        if (minimum_options[n] != maximum_options[n]) {
            return fail(AsymptoticDiffusionFieldFailure::InvalidState);
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
        return fail(AsymptoticDiffusionFieldFailure::InvalidState);
    }
    FieldMetric const metric{dx, geometry.ProbLo(0), geometry.Domain().smallEnd(0), volume};
    amrex::Real const volume_scale = metric.Volume(geometry.Domain().bigEnd(0));
    if (!collectiveValid(volume_scale > 0 && std::isfinite(volume_scale))) {
        return fail(AsymptoticDiffusionFieldFailure::InvalidState);
    }

    amrex::GpuArray<int, AMREX_SPACEDIM> domain_lo{};
    amrex::GpuArray<int, AMREX_SPACEDIM> domain_hi{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        domain_lo[d] = geometry.Domain().smallEnd(d);
        domain_hi[d] = geometry.Domain().bigEnd(d);
    }

    amrex::MultiFab energy_old(boxes, distribution, 1, 1);
    amrex::MultiFab chi(boxes, distribution, 1, 1);
    amrex::iMultiFab mask(boxes, distribution, 1, 1);
    amrex::MultiFab work(boxes, distribution, 1, 0);
    amrex::MultiFab qa_old(boxes, distribution, 3, 0);
    energy_old.setVal(0);
    chi.setVal(0);
    mask.setVal(0);
    work.setVal(0);
    qa_old.setVal(0);
    amrex::MultiFab::Copy(energy_old, energy, 0, 0, 1, 0);
    amrex::MultiFab::Copy(chi, opacity, 0, 0, 1, 0);
    amrex::MultiFab::Copy(work, requested_work, 0, 0, 1, 0);
    for (amrex::MFIter mfi(mask); mfi.isValid(); ++mfi)
    {
        auto const src = diffusion_mask.const_array(mfi);
        auto const dst = mask.array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE (int i, int j, int k)
                           {
                               dst(i, j, k) = src(i, j, k);
                           });
    }
    if (old_radiation_inertia != nullptr)
    {
        amrex::MultiFab::Copy(qa_old, *old_radiation_inertia, 0, 0, 3, 0);
    }
    energy_old.FillBoundary(geometry.periodicity());
    chi.FillBoundary(geometry.periodicity());
    mask.FillBoundary(geometry.periodicity());

    Faces admitted;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        admitted[d] = std::make_unique<amrex::MultiFab>(
            admitted_energy[d]->boxArray(), distribution, 1, 0);
        amrex::MultiFab::Copy(*admitted[d], *admitted_energy[d], 0, 0, 1, 0);
        // A complete face admission must agree on shared and periodic aliases.
        // Inspect a private canonical copy; never silently choose an owner of
        // inconsistent input or count an unsynchronized contribution twice.
        amrex::MultiFab canonical(admitted[d]->boxArray(), distribution, 1, 0);
        amrex::MultiFab::Copy(canonical, *admitted[d], 0, 0, 1, 0);
        canonical.OverrideSync(geometry.periodicity());
        amrex::MultiFab::Subtract(canonical, *admitted[d], 0, 0, 1, 0);
        if (canonical.contains_nan() || canonical.contains_inf() || canonical.norm0() != 0) {
            return fail(AsymptoticDiffusionFieldFailure::InvalidState);
        }
    }

    auto const lambda = options.marshak_extrapolation_lambda;
    amrex::ReduceOps<amrex::ReduceOpMax> input_ops;
    amrex::ReduceData<int> input_data(input_ops);
    for (amrex::MFIter mfi(energy_old); mfi.isValid(); ++mfi)
    {
        auto const e = energy_old.const_array(mfi);
        auto const alpha = chi.const_array(mfi);
        auto const m = mask.const_array(mfi);
        auto const w = work.const_array(mfi);
        auto const qa = qa_old.const_array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> admit;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            admit[d] = admitted[d]->const_array(mfi);
        }
        input_ops.eval(
            mfi.validbox(), input_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> amrex::GpuTuple<int>
            {
                int const flag = m(i, j, k);
                bool bad = flag != 0 && flag != 1;
                auto const cell_energy = e(i, j, k);
                auto const cell_chi = alpha(i, j, k);
                auto const cell_work = w(i, j, k);
                bad = bad || !amrex::Math::isfinite(cell_energy) || cell_energy < 0 ||
                      !amrex::Math::isfinite(cell_chi) || cell_chi < 0 ||
                      !amrex::Math::isfinite(cell_work);
                for (int p = 0; p < 3; ++p)
                {
                    bad = bad || !amrex::Math::isfinite(qa(i, j, k, p));
                }
                if (isKineticCell(flag))
                {
                    bad = bad || cell_energy != 0 || cell_work != 0;
                }
                if (isDiffusionCell(flag))
                {
                    bad = bad || !(cell_chi > 0);
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        auto const tau = cell_chi * dx[d];
                        bad = bad || !(tau > 0) || !amrex::Math::isfinite(tau);
                        for (int side = -1; side <= 1; side += 2)
                        {
                            bool const at_wall =
                                periodic[d] == 0 &&
                                axisIndex(i, j, k, d) ==
                                    (side < 0 ? domain_lo[d] : domain_hi[d]);
                            int ni = i, nj = j, nk = k;
                            shiftIndex(ni, nj, nk, d, side);
                            int fi = i, fj = j, fk = k;
                            if (side > 0)
                            {
                                shiftIndex(fi, fj, fk, d, 1);
                            }
                            auto const face_admit = admit[d](fi, fj, fk);
                            bad = bad || !amrex::Math::isfinite(face_admit) ||
                                  face_admit < 0;
                            if (at_wall)
                            {
                                bad = bad || face_admit != 0;
                                continue;
                            }
                            if (isKineticCell(m(ni, nj, nk)))
                            {
                                bad = bad ||
                                      !ValidAsymptoticInterfaceOpticalDepth(tau, lambda);
                            }
                            else if (isDiffusionCell(m(ni, nj, nk)))
                            {
                                bad = bad || face_admit != 0;
                            }
                            else
                            {
                                bad = true;
                            }
                        }
                    }
                }
                else if (isKineticCell(flag))
                {
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        for (int side = -1; side <= 1; side += 2)
                        {
                            bool const at_wall =
                                periodic[d] == 0 &&
                                axisIndex(i, j, k, d) ==
                                    (side < 0 ? domain_lo[d] : domain_hi[d]);
                            int fi = i, fj = j, fk = k;
                            if (side > 0)
                            {
                                shiftIndex(fi, fj, fk, d, 1);
                            }
                            auto const face_admit = admit[d](fi, fj, fk);
                            bad = bad || !amrex::Math::isfinite(face_admit) ||
                                  face_admit < 0;
                            if (at_wall)
                            {
                                bad = bad || face_admit != 0;
                                continue;
                            }
                            int ni = i, nj = j, nk = k;
                            shiftIndex(ni, nj, nk, d, side);
                            if (isKineticCell(m(ni, nj, nk)))
                            {
                                bad = bad || face_admit != 0;
                            }
                        }
                    }
                }
                return {bad ? 1 : 0};
            });
    }
    if (!collectiveValid(localFlag(amrex::get<0>(input_data.value()), energy_old) == 0))
    {
        return fail(AsymptoticDiffusionFieldFailure::InvalidState);
    }

    amrex::MultiFab acoef(boxes, distribution, 1, 0);
    amrex::MultiFab rhs(boxes, distribution, 1, 0);
    amrex::MultiFab phi(boxes, distribution, 1, 1);
    acoef.setVal(1);
    rhs.setVal(0);
    phi.setVal(0);
    Faces bcoef;
    Faces current;
    Faces pressure;
    Faces face_density;
    Faces leaked_energy;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        auto const faces = amrex::convert(boxes, amrex::IntVect::TheDimensionVector(d));
        bcoef[d] = std::make_unique<amrex::MultiFab>(faces, distribution, 1, 0);
        current[d] = std::make_unique<amrex::MultiFab>(faces, distribution, 1, 0);
        pressure[d] = std::make_unique<amrex::MultiFab>(faces, distribution, 1, 0);
        face_density[d] = std::make_unique<amrex::MultiFab>(faces, distribution, 1, 0);
        leaked_energy[d] = std::make_unique<amrex::MultiFab>(faces, distribution, 1, 0);
        bcoef[d]->setVal(0);
        current[d]->setVal(0);
        pressure[d]->setVal(0);
        face_density[d]->setVal(0);
        leaked_energy[d]->setVal(0);
    }

    auto const c = PhysConst::c;
    amrex::ReduceOps<amrex::ReduceOpMax> coef_ops;
    amrex::ReduceData<int> coef_data(coef_ops);
    for (amrex::MFIter mfi(acoef); mfi.isValid(); ++mfi)
    {
        auto const e = energy_old.const_array(mfi);
        auto const alpha = chi.const_array(mfi);
        auto const m = mask.const_array(mfi);
        auto const w = work.const_array(mfi);
        auto const a = acoef.array(mfi);
        auto const r = rhs.array(mfi);
        auto const guess = phi.array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> admit;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            admit[d] = admitted[d]->const_array(mfi);
        }
        coef_ops.eval(
            mfi.validbox(), coef_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> amrex::GpuTuple<int>
            {
                amrex::Real const cell_volume = metric.Volume(i);
                guess(i, j, k) = e(i, j, k) / cell_volume;
                if (isKineticCell(m(i, j, k)))
                {
                    a(i, j, k) = 1;
                    r(i, j, k) = 0;
                    guess(i, j, k) = 0;
                    return {0};
                }
                amrex::Real diagonal = cell_volume / volume_scale;
                amrex::Real admission = 0;
                bool bad = !amrex::Math::isfinite(guess(i, j, k));
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    for (int side = -1; side <= 1; side += 2)
                    {
                        bool const at_wall =
                            periodic[d] == 0 &&
                            axisIndex(i, j, k, d) ==
                                (side < 0 ? domain_lo[d] : domain_hi[d]);
                        if (at_wall)
                        {
                            continue;
                        }
                        int ni = i, nj = j, nk = k;
                        shiftIndex(ni, nj, nk, d, side);
                        if (!isKineticCell(m(ni, nj, nk)))
                        {
                            continue;
                        }
                        amrex::Real const tau = alpha(i, j, k) * dx[d];
                        amrex::Real const h = 2 * c / (3 * (tau + 2 * lambda));
                        int fi = i, fj = j, fk = k;
                        if (side > 0)
                        {
                            shiftIndex(fi, fj, fk, d, 1);
                        }
                        diagonal += dt * h * metric.Area(d, fi) / volume_scale;
                        admission += admit[d](fi, fj, fk);
                        bad = bad || !amrex::Math::isfinite(h) || !(h > 0) ||
                              !amrex::Math::isfinite(diagonal);
                    }
                }
                a(i, j, k) = diagonal;
                r(i, j, k) = (e(i, j, k) + admission - w(i, j, k)) / volume_scale;
                bad = bad || !amrex::Math::isfinite(a(i, j, k)) ||
                      !amrex::Math::isfinite(r(i, j, k)) || !(a(i, j, k) > 0);
                return {bad ? 1 : 0};
            });
    }

    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        for (amrex::MFIter mfi(*bcoef[d]); mfi.isValid(); ++mfi)
        {
            auto const alpha = chi.const_array(mfi);
            auto const m = mask.const_array(mfi);
            auto const b = bcoef[d]->array(mfi);
            coef_ops.eval(
                mfi.validbox(), coef_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> amrex::GpuTuple<int>
                {
                    bool const lo_wall = periodic[d] == 0 &&
                                         axisIndex(i, j, k, d) == domain_lo[d];
                    int li = i, lj = j, lk = k;
                    shiftIndex(li, lj, lk, d, -1);
                    bool const hi_wall = periodic[d] == 0 &&
                                         axisIndex(li, lj, lk, d) == domain_hi[d];
                    if (lo_wall || hi_wall)
                    {
                        b(i, j, k) = 0;
                        return {0};
                    }
                    if (!isDiffusionCell(m(li, lj, lk)) || !isDiffusionCell(m(i, j, k)))
                    {
                        b(i, j, k) = 0;
                        return {0};
                    }
                    amrex::Real const denom =
                        alpha(li, lj, lk) * dx[d] + alpha(i, j, k) * dx[d];
                    amrex::Real const value = 2 * c * dx[d] / (3 * denom);
                    b(i, j, k) = value * metric.Area(d, i) * dx[d] / volume_scale;
                    bool const bad = !(denom > 0) || !amrex::Math::isfinite(denom) ||
                                     !amrex::Math::isfinite(value) || !(value > 0) ||
                                     !amrex::Math::isfinite(b(i, j, k)) || !(b(i, j, k) > 0);
                    return {bad ? 1 : 0};
                });
        }
    }
    if (!collectiveValid(localFlag(amrex::get<0>(coef_data.value()), acoef) == 0))
    {
        return fail(AsymptoticDiffusionFieldFailure::NonfiniteCoefficient);
    }

    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> boundary;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        boundary[d] = periodic[d] != 0 ? amrex::LinOpBCType::Periodic
                                       : amrex::LinOpBCType::Neumann;
    }
    auto const physical_box = geometry.ProbDomain();
    amrex::Geometry const cartesian(geometry.Domain(), &physical_box, 0, periodic.data());
    amrex::MLABecLaplacian op({cartesian}, {boxes}, {distribution});
    op.setMaxOrder(2);
    op.setDomainBC(boundary, boundary);
    op.setLevelBC(0, nullptr);
    op.setScalars(1, dt);
    op.setACoeffs(0, acoef);
    op.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoef));
    amrex::MLMG solver(op);
    solver.setVerbose(options.verbosity);
    solver.setMaxIter(options.max_linear_iterations);
    solver.setThrowException(true);
    try
    {
        auto const linear_residual = solver.solve(
            {&phi}, {&rhs}, options.linear_tolerance, 0.0_rt);
        result.linear_iterations = solver.getNumIters();
        result.linear_residual = linear_residual;
    } catch (amrex::MLMG::error const&)
    {
        result.linear_iterations = solver.getNumIters();
        return fail(AsymptoticDiffusionFieldFailure::LinearConvergence);
    }

    // K rows are analytically independent equations phi_K=0 (a=1, rhs=0,
    // and every adjacent b=0). Solve those auxiliary degrees of freedom
    // exactly before diagnostics. They are never physical radiation energy;
    // do not apply a dimensionless tolerance or clip any D-cell solution.
    for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto const value = phi.array(mfi);
        auto const m = mask.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            if (isKineticCell(m(i,j,k))) { value(i,j,k) = 0; }
        });
    }
    phi.FillBoundary(geometry.periodicity());
    if (phi.contains_nan() || phi.contains_inf() || !(phi.min(0) >= 0))
    {
        return fail(AsymptoticDiffusionFieldFailure::NonnegativeSolution);
    }

    amrex::ReduceOps<amrex::ReduceOpMax> recon_ops;
    amrex::ReduceData<int> recon_data(recon_ops);
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        for (amrex::MFIter mfi(*current[d]); mfi.isValid(); ++mfi)
        {
            auto const e = phi.const_array(mfi);
            auto const alpha = chi.const_array(mfi);
            auto const m = mask.const_array(mfi);
            auto const admit = admitted[d]->const_array(mfi);
            auto const flux = current[d]->array(mfi);
            auto const pi = pressure[d]->array(mfi);
            auto const ef = face_density[d]->array(mfi);
            auto const leak = leaked_energy[d]->array(mfi);
            recon_ops.eval(
                mfi.validbox(), recon_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> amrex::GpuTuple<int>
                {
                    amrex::Real const area = metric.Area(d, i);
                    int li = i, lj = j, lk = k;
                    shiftIndex(li, lj, lk, d, -1);
                    bool const lo_wall = periodic[d] == 0 &&
                                         axisIndex(i, j, k, d) == domain_lo[d];
                    bool const hi_wall = periodic[d] == 0 &&
                                         axisIndex(li, lj, lk, d) == domain_hi[d];
                    amrex::Real F = 0;
                    amrex::Real eface = 0;
                    if (lo_wall)
                    {
                        if (isDiffusionCell(m(i, j, k)))
                        {
                            eface = e(i, j, k);
                        }
                    }
                    else if (hi_wall)
                    {
                        if (isDiffusionCell(m(li, lj, lk)))
                        {
                            eface = e(li, lj, lk);
                        }
                    }
                    else if (isDiffusionCell(m(li, lj, lk)) &&
                             isDiffusionCell(m(i, j, k)))
                    {
                        amrex::Real const denom =
                            alpha(li, lj, lk) * dx[d] + alpha(i, j, k) * dx[d];
                        F = -2 * c * (e(i, j, k) - e(li, lj, lk)) / (3 * denom);
                        eface = (alpha(i, j, k) * dx[d] * e(li, lj, lk) +
                                 alpha(li, lj, lk) * dx[d] * e(i, j, k)) /
                                denom;
                    }
                    else if (isDiffusionCell(m(li, lj, lk)) &&
                             isKineticCell(m(i, j, k)))
                    {
                        amrex::Real const tau = alpha(li, lj, lk) * dx[d];
                        amrex::Real const h = 2 * c / (3 * (tau + 2 * lambda));
                        leak(i,j,k) = dt * area * h * e(li,lj,lk);
                        amrex::Real const Fout =
                            h * e(li, lj, lk) - admit(i, j, k) / (dt * area);
                        F = Fout;
                        eface = 2 * lambda * e(li,lj,lk) / (tau + 2 * lambda) +
                            (3 * tau / (2 * c)) * (admit(i,j,k) / (dt * area));
                    }
                    else if (isKineticCell(m(li, lj, lk)) &&
                             isDiffusionCell(m(i, j, k)))
                    {
                        amrex::Real const tau = alpha(i, j, k) * dx[d];
                        amrex::Real const h = 2 * c / (3 * (tau + 2 * lambda));
                        leak(i,j,k) = dt * area * h * e(i,j,k);
                        amrex::Real const Fout =
                            h * e(i, j, k) - admit(i, j, k) / (dt * area);
                        F = -Fout;
                        eface = 2 * lambda * e(i,j,k) / (tau + 2 * lambda) +
                            (3 * tau / (2 * c)) * (admit(i,j,k) / (dt * area));
                    }
                    flux(i, j, k) = F;
                    ef(i, j, k) = eface;
                    pi(i, j, k) = eface / 3;
                    bool const bad = !amrex::Math::isfinite(F) ||
                                     !amrex::Math::isfinite(eface) ||
                                     !amrex::Math::isfinite(pi(i, j, k)) ||
                                     !amrex::Math::isfinite(leak(i,j,k)) || leak(i,j,k) < 0 ||
                                     pi(i, j, k) < 0;
                    return {bad ? 1 : 0};
                });
        }
    }
    if (!collectiveValid(localFlag(amrex::get<0>(recon_data.value()), phi) == 0))
    {
        return fail(AsymptoticDiffusionFieldFailure::NonfiniteCoefficient);
    }

    amrex::MultiFab energy_new(boxes, distribution, 1, 0);
    amrex::MultiFab impulse(boxes, distribution, 3, 0);
    amrex::MultiFab ledger(boxes, distribution, 1, 0);
    amrex::MultiFab qa_new(boxes, distribution, 3, 0);
    amrex::MultiFab closure(boxes, distribution, 3, 0);
    energy_new.setVal(0);
    impulse.setVal(0);
    ledger.setVal(0);
    qa_new.setVal(0);
    closure.setVal(0);

    amrex::Real const roundoff_mult = 256;
    amrex::Real const eps = std::numeric_limits<amrex::Real>::epsilon();
    auto const residual_tol = options.residual_tolerance;
    auto const linear_residual = result.linear_residual;
    amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpSum>
        ledger_ops;
    amrex::ReduceData<amrex::Real, int, amrex::Real> ledger_data(ledger_ops);
    for (amrex::MFIter mfi(energy_new); mfi.isValid(); ++mfi)
    {
        auto const e = phi.const_array(mfi);
        auto const e0 = energy_old.const_array(mfi);
        auto const alpha = chi.const_array(mfi);
        auto const m = mask.const_array(mfi);
        auto const w = work.const_array(mfi);
        auto const qa0 = qa_old.const_array(mfi);
        auto const En = energy_new.array(mfi);
        auto const I = impulse.array(mfi);
        auto const res = ledger.array(mfi);
        auto const qa = qa_new.array(mfi);
        auto const R = closure.array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> flux, pi;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            flux[d] = current[d]->const_array(mfi);
            pi[d] = pressure[d]->const_array(mfi);
        }
        ledger_ops.eval(
            mfi.validbox(), ledger_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
                -> amrex::GpuTuple<amrex::Real, int, amrex::Real>
            {
                if (isKineticCell(m(i, j, k)))
                {
                    bool const bad = e(i, j, k) != 0;
                    En(i, j, k) = 0;
                    return {0, bad ? 1 : 0, 0};
                }
                amrex::Real const cell_volume = metric.Volume(i);
                amrex::Real const Enew = cell_volume * e(i, j, k);
                En(i, j, k) = Enew;
                amrex::Real sum_q = 0;
                amrex::Real term_scale = std::abs(Enew) + std::abs(e0(i, j, k)) +
                                         std::abs(w(i, j, k));
                amrex::GpuArray<amrex::Real, 3> impulse_cell{};
                amrex::GpuArray<amrex::Real, 3> qa_cell{};
                amrex::GpuArray<amrex::Real, 3> pressure_term{};
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    int const p = physicalDirection(d);
                    int hi[3] = {i, j, k};
                    hi[d] += 1;
                    amrex::Real const area_lo = metric.Area(d, i);
                    amrex::Real const area_hi = metric.Area(d, hi[0]);
                    amrex::Real const F_lo = flux[d](i, j, k);
                    amrex::Real const F_hi = flux[d](hi[0], hi[1], hi[2]);
                    bool const lo_wall = periodic[d] == 0 &&
                                         axisIndex(i, j, k, d) == domain_lo[d];
                    bool const hi_wall = periodic[d] == 0 &&
                                         axisIndex(hi[0], hi[1], hi[2], d) ==
                                             domain_hi[d] + 1;
                    amrex::Real const Fout_lo = lo_wall ? 0 : -F_lo;
                    amrex::Real const Fout_hi = hi_wall ? 0 : F_hi;
                    amrex::Real const Q_lo = dt * area_lo * Fout_lo;
                    amrex::Real const Q_hi = dt * area_hi * Fout_hi;
                    sum_q += Q_lo + Q_hi;
                    term_scale += std::abs(Q_lo) + std::abs(Q_hi);
                    amrex::Real const pre =
                        dt * cell_volume / (2 * c) * alpha(i, j, k);
                    impulse_cell[p] += pre * (Fout_lo * (-1) + Fout_hi);
                    qa_cell[p] += cell_volume / (2 * c) * (Fout_lo * (-1) + Fout_hi);
#if defined(WARPX_DIM_RZ)
                    if (d == 0) {
                        // Local radial stress includes -P_theta_theta/r. Equal
                        // isotropic cell and face pressure must cancel exactly.
                        auto const theta_pressure = e(i, j, k) / 3;
                        pressure_term[p] += c * dt *
                            (area_hi * (pi[d](hi[0], hi[1], hi[2]) - theta_pressure) -
                             area_lo * (pi[d](i, j, k) - theta_pressure));
                    } else
#endif
                    {
                        pressure_term[p] += c * dt * area_lo *
                            (pi[d](hi[0], hi[1], hi[2]) - pi[d](i, j, k));
                    }
                }
                res(i, j, k) = Enew - e0(i, j, k) + sum_q + w(i, j, k);
                for (int p = 0; p < 3; ++p)
                {
                    I(i, j, k, p) = impulse_cell[p];
                    qa(i, j, k, p) = qa_cell[p];
                    R(i, j, k, p) = qa_cell[p] - qa0(i, j, k, p) + pressure_term[p] +
                                    c * impulse_cell[p];
                }
                // 256*eps times ledger operands is a documented roundoff
                // scale, not an energy floor and not a clipped residual.
                amrex::Real const roundoff =
                    roundoff_mult * eps * (term_scale + std::abs(Enew));
                amrex::Real const allowed =
                    amrex::max(residual_tol * term_scale,
                               amrex::max(roundoff,
                                          2 * linear_residual * volume_scale));
                bool bad = !amrex::Math::isfinite(Enew) || Enew < 0 ||
                           !amrex::Math::isfinite(res(i, j, k)) ||
                           std::abs(res(i, j, k)) > allowed;
                for (int p = 0; p < 3; ++p)
                {
                    bad = bad || !amrex::Math::isfinite(I(i, j, k, p)) ||
                          !amrex::Math::isfinite(qa(i, j, k, p)) ||
                          !amrex::Math::isfinite(R(i, j, k, p));
                }
                return {std::abs(res(i, j, k)), bad ? 1 : 0, res(i, j, k)};
            });
    }
    auto const reduced = ledger_data.value();
    amrex::Real max_abs = localReal(amrex::get<0>(reduced), energy_new);
    int bad_ledger = localFlag(amrex::get<1>(reduced), energy_new);
    amrex::Real global = localReal(amrex::get<2>(reduced), energy_new);
    amrex::ParallelDescriptor::ReduceRealMax(max_abs);
    amrex::ParallelDescriptor::ReduceIntMax(bad_ledger);
    amrex::ParallelDescriptor::ReduceRealSum(global);
    if (bad_ledger != 0 || !std::isfinite(global) || !std::isfinite(max_abs))
    {
        return fail(AsymptoticDiffusionFieldFailure::ConservationResidual);
    }

    result.energy.define(boxes, distribution, 1, 0);
    result.material_impulse.define(boxes, distribution, 3, 0);
    result.energy_balance_residual.define(boxes, distribution, 1, 0);
    result.radiation_inertia.define(boxes, distribution, 3, 0);
    result.closure_diagnostic.define(boxes, distribution, 3, 0);
    amrex::MultiFab::Copy(result.energy, energy_new, 0, 0, 1, 0);
    amrex::MultiFab::Copy(result.material_impulse, impulse, 0, 0, 3, 0);
    amrex::MultiFab::Copy(result.energy_balance_residual, ledger, 0, 0, 1, 0);
    amrex::MultiFab::Copy(result.radiation_inertia, qa_new, 0, 0, 3, 0);
    amrex::MultiFab::Copy(result.closure_diagnostic, closure, 0, 0, 3, 0);
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        result.face_current[d] = std::move(current[d]);
        result.face_pressure[d] = std::move(pressure[d]);
        result.face_energy_density[d] = std::move(face_density[d]);
        result.face_leaked_energy[d] = std::move(leaked_energy[d]);
    }
    result.global_energy_balance_residual = global;
    result.max_energy_balance_residual = max_abs;
    result.valid = true;
    result.failure = AsymptoticDiffusionFieldFailure::None;
    return result;
}
} // namespace warpx::radiation
