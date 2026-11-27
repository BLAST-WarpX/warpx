/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "ImplicitGroupMomentTransport.H"

#include "CenteredPlanckBath.H"
#include "GroupMomentGeometry.H"
#include "RZMomentGeometry.H"

#include <AMReX_GMRES.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <limits>

namespace warpx::radiation
{
namespace
{
[[maybe_unused]] bool Agree (bool value)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(value);
    return value;
}

AMREX_GPU_HOST_DEVICE inline int PhysicalAxis (int d)
{
    return AMREX_SPACEDIM == 1 ? 2 : (AMREX_SPACEDIM == 2 && d == 1 ? 2 : d);
}

// Public top-level operator: CUDA device lambdas do not live in a private
// nested Impl. Signed Krylov tangents are algebra, not admitted physical states.
class GroupOperator
{
  public:
    using RT = amrex::Real;
    GroupOperator (amrex::MultiFab const &old, amrex::Geometry const &geometry, amrex::Real dt,
                   int groups)
        : m_geometry(geometry), m_metric(geometry), m_dt(dt), m_groups(groups), m_size(4 * groups),
          m_moment(old.boxArray(), old.DistributionMap(), m_size * m_size, 0),
          m_source(old.boxArray(), old.DistributionMap(), m_size * m_size, 0),
          m_source_operands(old.boxArray(), old.DistributionMap(), m_size * m_size, 0),
          m_pressure(old.boxArray(), old.DistributionMap(), 6 * groups * m_size, 0),
          m_equilibrium(old.boxArray(), old.DistributionMap(), m_size, 2),
          m_transport(old.boxArray(), old.DistributionMap(), groups, 1),
          m_lab(old.boxArray(), old.DistributionMap(), m_size, 1),
          m_stress(old.boxArray(), old.DistributionMap(), 6 * groups, 1)
    {
#if defined(WARPX_DIM_RZ)
        m_density.define(old.boxArray(), old.DistributionMap(), m_size, 2);
        m_lab_density.define(old.boxArray(), old.DistributionMap(), m_size, 1);
#endif
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            m_flux[d].define(amrex::convert(old.boxArray(), amrex::IntVect::TheDimensionVector(d)),
                             old.DistributionMap(), m_size, 0);
        }
    }

    bool build (amrex::MultiFab const &beta, amrex::MultiFab const &temperature,
                amrex::MultiFab const &transport, amrex::MultiFab const &absorption,
                ImplicitGroupMomentTransportOptions const &options)
    {
        m_centered = options.experimental_centered_reconstruction;
        if (m_centered)
        {
            m_bath_lab.define(m_lab.boxArray(), m_lab.DistributionMap(), m_size, 0);
            m_bath_stress.define(m_stress.boxArray(), m_stress.DistributionMap(), 6 * m_groups, 0);
        }
        if (m_groups <= 1)
        {
            return buildSized<1>(beta, temperature, transport, absorption, options);
        }
        if (m_groups <= 2)
        {
            return buildSized<2>(beta, temperature, transport, absorption, options);
        }
        if (m_groups <= 4)
        {
            return buildSized<4>(beta, temperature, transport, absorption, options);
        }
        return buildSized<8>(beta, temperature, transport, absorption, options);
    }

    template <int MaximumGroups>
    bool buildSized (amrex::MultiFab const &beta, amrex::MultiFab const &temperature,
                     amrex::MultiFab const &transport, amrex::MultiFab const &absorption,
                     ImplicitGroupMomentTransportOptions const &options)
    {
        amrex::MultiFab::Copy(m_transport, transport, 0, 0, m_groups, 0);
        FillGroupMomentDensityGhosts(m_transport, m_geometry, m_groups, GroupGhostKind::Scalar);
        m_equilibrium.setVal(0);
        amrex::Real volume = 1, dx = m_geometry.CellSize(0), length = 0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            volume *= m_geometry.CellSize(d);
            dx = std::min(dx, m_geometry.CellSize(d));
            length = std::max(length, m_geometry.ProbHi(d) - m_geometry.ProbLo(d));
        }
        amrex::ReduceOps<amrex::ReduceOpMax> ops;
        amrex::ReduceData<int> data(ops);
        auto const groups = options.groups;
        auto const optical = std::max(8.0_rt, options.minimum_optical_depth);
        auto const dynamic = options.experimental_dynamic_diffusion;
        auto const centered = m_centered;
        auto const n = m_groups, size = m_size;
        auto const step = m_dt;
        auto const metric = m_metric;
        for (amrex::MFIter it(m_equilibrium); it.isValid(); ++it)
        {
            auto const b = beta.const_array(it), t = temperature.const_array(it);
            auto const ct = transport.const_array(it), ca = absorption.const_array(it);
            auto const m = m_moment.array(it), s = m_source.array(it);
            auto const noise = m_source_operands.array(it), p = m_pressure.array(it);
            auto const eq = m_equilibrium.array(it);
            auto const bath_lab = centered ? m_bath_lab.array(it) : amrex::Array4<amrex::Real>{};
            auto const bath_stress =
                centered ? m_bath_stress.array(it) : amrex::Array4<amrex::Real>{};
            ops.eval(
                it.validbox(), data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                {
                    amrex::GpuArray<amrex::Real, 3> drift{};
                    amrex::GpuArray<amrex::Real, exact_group_maximum> chi_t{}, chi_a{};
                    for (int d = 0; d < 3; ++d)
                    {
                        drift[d] = b(i, j, k, d);
                    }
                    auto const speed = std::sqrt(exact_group_detail::Dot(drift, drift));
                    for (int g = 0; g < n; ++g)
                    {
                        chi_t[g] = ct(i, j, k, g);
                        chi_a[g] = ca(i, j, k, g);
                        if (!(chi_t[g] * dx >= optical) ||
                            (chi_a[g] > 0 && !dynamic && !(speed * chi_t[g] * length <= 0.01_rt)))
                        {
                            return {1};
                        }
                    }
                    if (centered)
                    {
                        ExactGroupVector bath{};
                        centered_bath_detail::Pressure stress{};
                        if (!centered_bath_detail::Evaluate(groups, drift, t(i, j, k), bath,
                                                            stress))
                        {
                            return {1};
                        }
                        auto const cell_volume =
#if defined(WARPX_DIM_RZ)
                            metric.Volume(i);
#else
                            volume;
#endif
                        for (int row = 0; row < size; ++row)
                        {
                            bath_lab(i, j, k, row) = bath[row] * cell_volume;
                        }
                        for (int row = 0; row < 6 * n; ++row)
                        {
                            bath_stress(i, j, k, row) = stress[row] * cell_volume;
                        }
                    }
                    exact_group_detail::SizedMatrix<MaximumGroups> moment{}, source{}, operands{};
                    exact_group_detail::SizedPressureMatrix<MaximumGroups> pressure{};
                    if (!exact_group_detail::Assemble(groups, drift, t(i, j, k), step, chi_t.data(),
                                                      chi_a.data(), moment, source, 16, &operands,
                                                      &pressure, centered, centered))
                    {
                        return {1};
                    }
                    auto const blackbody = spectral_detail::Blackbody(t(i, j, k)) *
#if defined(WARPX_DIM_RZ)
                                           metric.Volume(i);
                    amrex::ignore_unused(volume);
#else
                                           volume;
                    amrex::ignore_unused(metric);
#endif
                    for (int g = 0; g < n; ++g)
                    {
                        eq(i, j, k, 4 * g) =
                            blackbody * spectral_detail::Fraction(groups, g, t(i, j, k));
                    }
                    for (int row = 0; row < size; ++row)
                    {
                        for (int column = 0; column < size; ++column)
                        {
                            auto const index = size * row + column;
                            m(i, j, k, index) = moment[row][column];
                            s(i, j, k, index) = source[row][column];
                            noise(i, j, k, index) = operands[row][column];
                        }
                    }
                    for (int row = 0; row < 6 * n; ++row)
                    {
                        for (int column = 0; column < size; ++column)
                        {
                            p(i, j, k, size * row + column) = pressure[row][column];
                        }
                    }
                    return {0};
                });
        }
        int invalid = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceIntMax(invalid);
        m_equilibrium.FillBoundary(m_geometry.periodicity());
        return invalid == 0 && m_equilibrium.is_finite();
    }

    amrex::MultiFab makeVecRHS () const
    {
        return {m_equilibrium.boxArray(), m_equilibrium.DistributionMap(), m_size, 0};
    }
    amrex::MultiFab makeVecLHS () const
    {
        return {m_equilibrium.boxArray(), m_equilibrium.DistributionMap(), m_size, 2};
    }
    void assign (amrex::MultiFab &lhs, amrex::MultiFab const &rhs) const
    {
        amrex::MultiFab::Copy(lhs, rhs, 0, 0, m_size, 0);
    }
    RT dotProduct (amrex::MultiFab const &a, amrex::MultiFab const &b) const
    {
        return amrex::MultiFab::Dot(a, 0, b, 0, m_size, 0);
    }
    RT norm2 (amrex::MultiFab const &a) const { return std::sqrt(dotProduct(a, a)); }
    void increment (amrex::MultiFab &lhs, amrex::MultiFab const &rhs, RT factor) const
    {
        amrex::MultiFab::Saxpy(lhs, factor, rhs, 0, 0, m_size, 0);
    }
    void linComb (amrex::MultiFab &lhs, RT a, amrex::MultiFab const &x, RT b,
                  amrex::MultiFab const &y) const
    {
        amrex::MultiFab::LinComb(lhs, a, x, 0, b, y, 0, 0, m_size, 0);
    }
    void scale (amrex::MultiFab &v, RT factor) const { v.mult(factor, 0, m_size, 0); }
    void setToZero (amrex::MultiFab &v) const { v.setVal(0); }

    void precond (amrex::MultiFab &lhs, amrex::MultiFab const &rhs) const
    {
        if (m_groups <= 1)
        {
            precondSized<1>(lhs, rhs);
        }
        else if (m_groups <= 2)
        {
            precondSized<2>(lhs, rhs);
        }
        else if (m_groups <= 4)
        {
            precondSized<4>(lhs, rhs);
        }
        else
        {
            precondSized<8>(lhs, rhs);
        }
    }

    template <int MaximumGroups>
    void precondSized (amrex::MultiFab &lhs, amrex::MultiFab const &rhs) const
    {
        auto const size = m_size;
        for (amrex::MFIter it(lhs); it.isValid(); ++it)
        {
            auto const m = m_moment.const_array(it), s = m_source.const_array(it);
            auto const r = rhs.const_array(it);
            auto const out = lhs.array(it);
            amrex::ParallelFor(
                it.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    exact_group_detail::SizedMatrix<MaximumGroups> local{};
                    ExactGroupVector input{}, answer{};
                    for (int row = 0; row < size; ++row)
                    {
                        input[row] = r(i, j, k, row);
                        for (int column = 0; column < size; ++column)
                        {
                            auto const index = size * row + column;
                            local[row][column] = m(i, j, k, index) + s(i, j, k, index);
                        }
                    }
                    auto const solved = exact_group_detail::Solve(local, input, size, answer);
                    for (int row = 0; row < size; ++row)
                    {
                        out(i, j, k, row) =
                            solved ? answer[row] : std::numeric_limits<amrex::Real>::quiet_NaN();
                    }
                });
        }
    }

    void
    formFluxes (amrex::MultiFab const& rest, bool bath = false)
    {
#if !defined(WARPX_DIM_RZ)
        const_cast<amrex::MultiFab &>(rest).FillBoundary(m_geometry.periodicity());
#endif
        auto const size = m_size, n = m_groups;
        auto const metric = m_metric;
        auto const centered = bath && m_centered;
        for (amrex::MFIter it(m_lab); it.isValid(); ++it)
        {
            auto const r = rest.const_array(it), m = m_moment.const_array(it);
            auto const p = m_pressure.const_array(it);
            auto const lab = m_lab.array(it), stress = m_stress.array(it);
            auto const bath_lab =
                centered ? m_bath_lab.const_array(it) : amrex::Array4<amrex::Real const>{};
            auto const bath_stress =
                centered ? m_bath_stress.const_array(it) : amrex::Array4<amrex::Real const>{};
#if defined(WARPX_DIM_RZ)
            auto const density = m_density.array(it), lab_density = m_lab_density.array(it);
#endif
            amrex::ParallelFor(
                it.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
#if defined(WARPX_DIM_RZ)
                    ExactGroupVector local_density{};
                    for (int column = 0; column < size; ++column)
                    {
                        local_density[column] = r(i, j, k, column) / metric.Volume(i);
                        density(i, j, k, column) = local_density[column];
                    }
#else
                    amrex::ignore_unused(metric);
#endif
                    for (int row = 0; row < size; ++row)
                    {
                        lab(i, j, k, row) = 0;
#if defined(WARPX_DIM_RZ)
                        lab_density(i, j, k, row) = 0;
#endif
                        for (int column = 0; column < size; ++column)
                        {
                            lab(i, j, k, row) +=
                                m(i, j, k, size * row + column) * r(i, j, k, column);
#if defined(WARPX_DIM_RZ)
                            lab_density(i, j, k, row) +=
                                m(i, j, k, size * row + column) * local_density[column];
#endif
                        }
                    }
                    for (int row = 0; row < 6 * n; ++row)
                    {
                        stress(i, j, k, row) = 0;
                        for (int column = 0; column < size; ++column)
                        {
                            stress(i, j, k, row) += p(i, j, k, size * row + column) *
#if defined(WARPX_DIM_RZ)
                                                    local_density[column];
#else
                                                    r(i, j, k, column);
#endif
                        }
                    }
                    if (centered)
                    {
                        for (int row = 0; row < size; ++row)
                        {
                            lab(i, j, k, row) = bath_lab(i, j, k, row);
#if defined(WARPX_DIM_RZ)
                            lab_density(i, j, k, row) = bath_lab(i, j, k, row) / metric.Volume(i);
#endif
                        }
                        for (int row = 0; row < 6 * n; ++row)
                        {
                            stress(i, j, k, row) = bath_stress(i, j, k, row)
#if defined(WARPX_DIM_RZ)
                                                   / metric.Volume(i)
#endif
                                ;
                        }
                    }
                });
        }
#if defined(WARPX_DIM_RZ)
        FillGroupMomentDensityGhosts(m_density, m_geometry, n, GroupGhostKind::FourMoments);
        FillGroupMomentDensityGhosts(m_lab_density, m_geometry, n, GroupGhostKind::FourMoments);
        FillGroupMomentDensityGhosts(m_stress, m_geometry, n, GroupGhostKind::SymmetricPressure);
#else
        m_lab.FillBoundary(m_geometry.periodicity());
        m_stress.FillBoundary(m_geometry.periodicity());
#endif
        auto const step = m_dt;
        for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
        {
            auto const physical = PhysicalAxis(axis);
            auto const dx = m_geometry.CellSize(axis);
            for (amrex::MFIter it(m_flux[axis]); it.isValid(); ++it)
            {
#if defined(WARPX_DIM_RZ)
                auto const lab = m_lab_density.const_array(it), r = m_density.const_array(it);
#else
                auto const lab = m_lab.const_array(it), r = rest.const_array(it);
#endif
                auto const p = m_stress.const_array(it), ct = m_transport.const_array(it);
                auto const f = m_flux[axis].array(it);
                amrex::ParallelFor(
                    it.validbox(), n,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int g)
                    {
                        int left[3]{i, j, k};
                        --left[axis];
#if !defined(WARPX_DIM_RZ)
                        int left_left[3]{i, j, k}, right_right[3]{i, j, k};
                        left_left[axis] -= 2;
                        ++right_right[axis];
#endif
                        auto const d_left =
                            PhysConst::c * PhysConst::c * step /
                            (3 * (1 + PhysConst::c * step * ct(left[0], left[1], left[2], g)));
                        auto const d_right = PhysConst::c * PhysConst::c * step /
                                             (3 * (1 + PhysConst::c * step * ct(i, j, k, g)));
                        auto const d_face =
                            PhysConst::c * PhysConst::c * step /
                            (3 * (1 + PhysConst::c * step *
                                          (ct(left[0], left[1], left[2], g) + ct(i, j, k, g)) / 2));
#if defined(WARPX_DIM_RZ)
                        auto const grad_left =
                            metric.Gradient(r, 4 * g, left[0], left[1], left[2], axis);
                        auto const grad_right = metric.Gradient(r, 4 * g, i, j, k, axis);
#else
                        auto const grad_left = (r(i, j, k, 4 * g) - r(left_left[0], left_left[1],
                                                                      left_left[2], 4 * g)) /
                                               (2 * dx);
                        auto const grad_right =
                            (r(right_right[0], right_right[1], right_right[2], 4 * g) -
                             r(left[0], left[1], left[2], 4 * g)) /
                            (2 * dx);
#endif
                        auto const grad_face =
                            (r(i, j, k, 4 * g) - r(left[0], left[1], left[2], 4 * g)) / dx;
                        f(i, j, k, 4 * g) =
                            PhysConst::c *
                                (lab(i, j, k, 4 * g + physical + 1) +
                                 lab(left[0], left[1], left[2], 4 * g + physical + 1)) /
                                2 +
                            (d_left * grad_left + d_right * grad_right) / 2 - d_face * grad_face;
#if defined(WARPX_DIM_RZ)
                        if (metric.Wall(axis, i, j, k))
                        {
                            f(i, j, k, 4 * g) = 0;
                        }
#else
                        amrex::ignore_unused(metric);
#endif
                        for (int d = 0; d < 3; ++d)
                        {
                            auto const component =
                                6 * g + exact_group_detail::PressureComponent(physical, d);
                            f(i, j, k, 4 * g + d + 1) =
                                PhysConst::c *
                                (p(i, j, k, component) + p(left[0], left[1], left[2], component)) /
                                2;
                        }
                    });
            }
        }
    }

    void
    formCenteredFluxes (amrex::MultiFab const& delta)
    {
        // The map and spatial flux are linear at fixed T/secant. Evaluate
        // the small correction before storing it beside the LTE bath rather
        // than rounding it into every rest component before matrix products.
        formFluxes(delta);
        amrex::MultiFab lab(m_lab.boxArray(), m_lab.DistributionMap(), m_size, 1);
        amrex::MultiFab stress(m_stress.boxArray(), m_stress.DistributionMap(), 6 * m_groups, 1);
        amrex::MultiFab::Copy(lab, m_lab, 0, 0, m_size, 1);
        amrex::MultiFab::Copy(stress, m_stress, 0, 0, 6 * m_groups, 1);
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> flux;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            flux[d].define(m_flux[d].boxArray(), m_flux[d].DistributionMap(), m_size, 0);
            amrex::MultiFab::Copy(flux[d], m_flux[d], 0, 0, m_size, 0);
        }
        formFluxes(m_equilibrium, true);
        amrex::MultiFab::Saxpy(m_lab, 1, lab, 0, 0, m_size, 1);
        amrex::MultiFab::Saxpy(m_stress, 1, stress, 0, 0, 6 * m_groups, 1);
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            amrex::MultiFab::Saxpy(m_flux[d], 1, flux[d], 0, 0, m_size, 0);
        }
    }

    void apply (amrex::MultiFab &out, amrex::MultiFab const &rest)
    {
        formFluxes(rest);
        auto const size = m_size;
        auto const step = m_dt;
        auto const spacing = m_geometry.CellSizeArray();
        auto const metric = m_metric;
        for (amrex::MFIter it(out); it.isValid(); ++it)
        {
            auto const r = rest.const_array(it), lab = m_lab.const_array(it);
            auto const s = m_source.const_array(it);
            auto const result = out.array(it);
#if defined(WARPX_DIM_RZ)
            auto const stress = m_stress.const_array(it);
#endif
            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> flux{};
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                flux[d] = m_flux[d].const_array(it);
            }
            amrex::ParallelFor(it.validbox(), size,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k, int row)
                               {
                                   auto value = lab(i, j, k, row);
                                   for (int column = 0; column < size; ++column)
                                   {
                                       value +=
                                           s(i, j, k, size * row + column) * r(i, j, k, column);
                                   }
#if defined(WARPX_DIM_RZ)
                                   value += metric.Increment(flux, row, i, j, k, step,
                                                             stress(i, j, k, 6 * (row / 4) + 1))[0];
                                   amrex::ignore_unused(spacing);
#else
                                   amrex::ignore_unused(metric);
                                   for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                   {
                                       int right[3]{i, j, k};
                                       ++right[d];
                                       value += step / spacing[d] *
                                                (flux[d](right[0], right[1], right[2], row) -
                                                 flux[d](i, j, k, row));
                                   }
#endif
                                   result(i, j, k, row) = value;
                               });
        }
    }

    amrex::Geometry const &m_geometry;
    GroupMomentMetric m_metric;
    amrex::Real m_dt;
    int m_groups, m_size;
    amrex::MultiFab m_moment, m_source, m_source_operands, m_pressure, m_equilibrium, m_transport;
    amrex::MultiFab m_lab, m_stress;
    bool m_centered = false;
    amrex::MultiFab m_bath_lab, m_bath_stress;
#if defined(WARPX_DIM_RZ)
    amrex::MultiFab m_density, m_lab_density;
#endif
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_flux;
};
} // namespace

ImplicitGroupMomentTransportResult
TryImplicitGroupMomentTransport (amrex::MultiFab const &old, amrex::MultiFab const &beta,
                                 amrex::MultiFab const &temperature,
                                 amrex::MultiFab const &transport,
                                 amrex::MultiFab const &absorption, amrex::Geometry const &geometry,
                                 amrex::Real dt, ImplicitGroupMomentTransportOptions const &options)
{
    ImplicitGroupMomentTransportResult result;
    auto fail = [&] (std::string const &reason)
    {
        result.reason = reason;
        return std::move(result);
    };
#if defined(AMREX_USE_FLOAT) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    amrex::ignore_unused(old, beta, temperature, transport, absorption, geometry, dt, options);
    return fail("Group moment spatial transport requires a supported double precision geometry");
#else
    auto const n = options.groups.m_num_groups, size = 4 * n;
    auto same = [&] (amrex::MultiFab const &field, int components)
    {
        return field.boxArray() == old.boxArray() &&
               field.DistributionMap() == old.DistributionMap() && field.nComp() == components &&
               field.ixType().cellCentered();
    };
    bool geometry_valid = geometry.isAllPeriodic();
#if defined(WARPX_DIM_RZ)
    geometry_valid = options.reflecting_boundaries && !options.experimental_dynamic_diffusion &&
                     !options.experimental_centered_reconstruction && geometry.ProbLo(0) == 0 &&
                     !geometry.isPeriodic(0) && geometry.isPeriodic(1) &&
                     old.boxArray().minimalBox() == geometry.Domain() &&
                     old.boxArray().numPts() == geometry.Domain().numPts();
#endif
    if (!Agree(n > 0 && n <= exact_group_maximum && (n == 1 || options.groups.m_boundaries) &&
               old.nComp() == size && old.ixType().cellCentered() && same(beta, 3) &&
               same(temperature, 1) && same(transport, n) && same(absorption, n) &&
               geometry_valid && geometry.Coord() == 0 && dt > 0 && std::isfinite(dt) &&
               options.tolerance > 0 && options.tolerance < 1 && options.linear_tolerance > 0 &&
               options.linear_tolerance < options.tolerance && options.max_linear_iterations > 0 &&
               std::isfinite(options.minimum_optical_depth)))
    {
        return fail("Invalid frozen multigroup transport state or options");
    }
    int integer_min[5]{n, options.max_linear_iterations,
                       static_cast<int>(options.reflecting_boundaries),
                       static_cast<int>(options.experimental_dynamic_diffusion),
                       static_cast<int>(options.experimental_centered_reconstruction)};
    int integer_max[5]{n, options.max_linear_iterations,
                       static_cast<int>(options.reflecting_boundaries),
                       static_cast<int>(options.experimental_dynamic_diffusion),
                       static_cast<int>(options.experimental_centered_reconstruction)};
    amrex::ParallelDescriptor::ReduceIntMin(integer_min, 5);
    amrex::ParallelDescriptor::ReduceIntMax(integer_max, 5);
    amrex::Real real_min[4]{dt, options.tolerance, options.linear_tolerance,
                            options.minimum_optical_depth};
    amrex::Real real_max[4]{dt, options.tolerance, options.linear_tolerance,
                            options.minimum_optical_depth};
    amrex::ParallelDescriptor::ReduceRealMin(real_min, 4);
    amrex::ParallelDescriptor::ReduceRealMax(real_max, 4);
    for (int d = 0; d < 4; ++d)
    {
        if (real_min[d] != real_max[d])
        {
            return fail("Frozen group transport controls disagree across ranks");
        }
    }
    for (int d = 0; d < 5; ++d)
    {
        if (integer_min[d] != integer_max[d])
        {
            return fail("Frozen group transport controls disagree across ranks");
        }
    }
    amrex::GpuArray<amrex::Real, exact_group_maximum - 1> edge_min{}, edge_max{};
    if (n > 1)
    {
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, options.groups.m_boundaries,
                         options.groups.m_boundaries + n - 1, edge_min.begin());
        edge_max = edge_min;
    }
    amrex::ParallelDescriptor::ReduceRealMin(edge_min.data(), exact_group_maximum - 1);
    amrex::ParallelDescriptor::ReduceRealMax(edge_max.data(), exact_group_maximum - 1);
    amrex::Real previous_edge = 0;
    for (int d = 0; d < n - 1; ++d)
    {
        if (edge_min[d] != edge_max[d] || !(edge_min[d] > previous_edge) ||
            !std::isfinite(edge_min[d]))
        {
            return fail("Frozen group transport spectral grids disagree or are invalid");
        }
        previous_edge = edge_min[d];
    }
    GroupOperator op(old, geometry, dt, n);
    if (!op.build(beta, temperature, transport, absorption, options))
    {
        return fail("Frozen multigroup transport coefficients or envelope rejected");
    }
    auto rhs = op.makeVecRHS();
    auto delta = op.makeVecLHS(), rest = op.makeVecLHS();
    // The LTE source cancels analytically; its spatial flux is retained.
    op.formFluxes(op.m_equilibrium, true);
    amrex::Real volume = 1;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        volume *= geometry.CellSize(d);
    }
    auto const spacing = geometry.CellSizeArray();
    auto const metric = op.m_metric;
#if !defined(WARPX_DIM_RZ)
    auto const centered = options.experimental_centered_reconstruction;
#endif
    for (amrex::MFIter it(rhs); it.isValid(); ++it)
    {
        auto const incoming = old.const_array(it), baseline = op.m_lab.const_array(it);
        auto const out = rhs.array(it);
        auto const stress = op.m_stress.const_array(it);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> flux{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            flux[d] = op.m_flux[d].const_array(it);
        }
        amrex::ParallelFor(it.validbox(), size,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k, int row)
                           {
                               auto value = incoming(i, j, k, row) - baseline(i, j, k, row);
#if defined(WARPX_DIM_RZ)
                               value -= metric.Increment(flux, row, i, j, k, dt,
                                                         stress(i, j, k, 6 * (row / 4) + 1))[0];
                               amrex::ignore_unused(spacing);
#else
                               amrex::ignore_unused(metric);
                               for (int d = 0; d < AMREX_SPACEDIM; ++d)
                               {
                                   int right[3]{i, j, k};
                                   ++right[d];
                                   if (centered && row % 4 != 0)
                                   {
                                       int left[3]{i, j, k};
                                       --left[d];
                                       auto const component =
                                           6 * (row / 4) + exact_group_detail::PressureComponent(
                                                               PhysicalAxis(d), row % 4 - 1);
                                       value -= dt * PhysConst::c / (2 * spacing[d]) *
                                                (stress(right[0], right[1], right[2], component) -
                                                 stress(left[0], left[1], left[2], component));
                                   }
                                   else
                                   {
                                       value -= dt / spacing[d] *
                                                (flux[d](right[0], right[1], right[2], row) -
                                                 flux[d](i, j, k, row));
                                   }
                               }
#endif
                               out(i, j, k, row) = value;
                           });
    }
    if (!rhs.is_finite())
    {
        return fail("Nonfinite multigroup transport right-hand side");
    }
    op.precond(delta, rhs);
    amrex::GMRES<amrex::MultiFab, GroupOperator> solver;
    solver.define(op);
    solver.setRestartLength(40);
    solver.setVerbose(options.verbose ? 1 : 0);
    solver.setMaxIters(options.max_linear_iterations);
    solver.solve(delta, rhs, options.linear_tolerance, 0);
    result.linear_iterations = solver.getNumIters();
    if (!delta.is_finite())
    {
        return fail("Nonfinite multigroup spatial correction");
    }
    op.assign(rest, op.m_equilibrium);
    op.increment(rest, delta, 1);
    if (options.experimental_centered_reconstruction)
    {
        op.formCenteredFluxes(delta);
    }
    else
    {
        op.formFluxes(rest);
    }
    result.radiation.define(old.boxArray(), old.DistributionMap(), size, 1);
    result.comoving_radiation.define(old.boxArray(), old.DistributionMap(), size, 1);
    result.material_transfer.define(old.boxArray(), old.DistributionMap(), size, 0);
    result.transfer_roundoff.define(old.boxArray(), old.DistributionMap(), size, 0);
    result.physical_heat.define(old.boxArray(), old.DistributionMap(), 1, 0);
    result.arithmetic.define(old.boxArray(), old.DistributionMap(), 3, 0);
    op.assign(result.radiation, op.m_lab);
    op.assign(result.comoving_radiation, rest);
    amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> ops;
    amrex::ReduceData<int, amrex::Real> data(ops);
    auto const tolerance = options.tolerance;
    for (amrex::MFIter it(rest); it.isValid(); ++it)
    {
        auto const incoming = old.const_array(it), current = op.m_lab.const_array(it);
        auto const r = rest.const_array(it), change = delta.const_array(it);
        auto const baseline = op.m_equilibrium.const_array(it), b = beta.const_array(it);
        auto const ct = transport.const_array(it), ca = absorption.const_array(it);
        auto const s = op.m_source.const_array(it), s_abs = op.m_source_operands.const_array(it);
        auto const transfer = result.material_transfer.array(it);
        auto const noise = result.transfer_roundoff.array(it),
                   heat = result.physical_heat.array(it);
        auto const arithmetic = result.arithmetic.array(it);
#if defined(WARPX_DIM_RZ)
        auto const stress = op.m_stress.const_array(it);
#endif
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> flux{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            flux[d] = op.m_flux[d].const_array(it);
        }
        ops.eval(
            it.validbox(), data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int, amrex::Real>
            {
                amrex::GpuArray<amrex::Real, 3> drift{}, recoil{};
                for (int d = 0; d < 3; ++d)
                {
                    drift[d] = b(i, j, k, d);
                }
                auto const gamma = 1 / std::sqrt(1 - exact_group_detail::Dot(drift, drift));
                amrex::Real total_debit = 0, total_heat = 0, projection_operands = 0;
                amrex::Real heat_noise = 0, momentum_noise = 0, maximum = 0;
                for (int g = 0; g < n; ++g)
                {
                    GroupSpectralMoments comoving, lab, before;
                    comoving.energy = r(i, j, k, 4 * g);
                    lab.energy = current(i, j, k, 4 * g);
                    before.energy = incoming(i, j, k, 4 * g);
                    for (int d = 0; d < 3; ++d)
                    {
                        comoving.flux[d] = PhysConst::c * r(i, j, k, 4 * g + d + 1);
                        lab.flux[d] = PhysConst::c * current(i, j, k, 4 * g + d + 1);
                        before.flux[d] = PhysConst::c * incoming(i, j, k, 4 * g + d + 1);
                    }
                    if (!spectral_detail::MomentValid(comoving, 1.0_rt / 3) ||
                        !spectral_detail::MomentValid(lab, 1) ||
                        !spectral_detail::MomentValid(before, 1))
                    {
                        return {1, 0};
                    }
                    auto const rest_heat =
                        PhysConst::c * dt * ca(i, j, k, g) * change(i, j, k, 4 * g) / gamma;
                    total_heat += rest_heat;
                    projection_operands += std::abs(rest_heat);
                    auto const scale =
                        std::abs(incoming(i, j, k, 4 * g)) + std::abs(baseline(i, j, k, 4 * g));
                    heat_noise += 64 * std::numeric_limits<amrex::Real>::epsilon() *
                                  amrex::min(1.0_rt, PhysConst::c * dt * ca(i, j, k, g)) * scale;
                    momentum_noise += 64 * std::numeric_limits<amrex::Real>::epsilon() *
                                      amrex::min(1.0_rt, PhysConst::c * dt * ct(i, j, k, g)) *
                                      scale;
                }
                for (int row = 0; row < size; ++row)
                {
                    amrex::Real q = 0, source_scale = 0, transfer_operands = 0;
                    for (int column = 0; column < size; ++column)
                    {
                        auto const value =
                            s(i, j, k, size * row + column) * change(i, j, k, column);
                        q += value;
                        source_scale += std::abs(value);
                        auto const g = column / 4;
                        auto const rate =
                            PhysConst::c * dt * (column % 4 == 0 ? ca(i, j, k, g) : ct(i, j, k, g));
                        transfer_operands += s_abs(i, j, k, size * row + column) *
                                             (std::abs(change(i, j, k, column)) +
                                              (std::abs(incoming(i, j, k, 4 * g)) +
                                               std::abs(baseline(i, j, k, 4 * g))) /
                                                  (1 + rate));
                        projection_operands += (row % 4 == 0 ? 1 : std::abs(drift[row % 4 - 1])) *
                                               s_abs(i, j, k, size * row + column) *
                                               std::abs(change(i, j, k, column));
                    }
                    transfer(i, j, k, row) = q;
                    noise(i, j, k, row) =
                        128 * std::numeric_limits<amrex::Real>::epsilon() * transfer_operands;
                    auto balance = (current(i, j, k, row) - incoming(i, j, k, row)) + q;
                    auto scale = std::abs(current(i, j, k, 4 * (row / 4))) +
                                 std::abs(incoming(i, j, k, 4 * (row / 4))) + source_scale;
                    amrex::Real operands = std::abs(current(i, j, k, row)) +
                                           std::abs(incoming(i, j, k, row)) + source_scale;
#if defined(WARPX_DIM_RZ)
                    auto const increment = metric.Increment(flux, row, i, j, k, dt,
                                                            stress(i, j, k, 6 * (row / 4) + 1));
                    balance += increment[0];
                    operands += increment[1];
                    amrex::ignore_unused(spacing);
#else
                    amrex::ignore_unused(metric);
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        int right[3]{i, j, k};
                        ++right[d];
                        auto const high =
                            dt / spacing[d] * flux[d](right[0], right[1], right[2], row);
                        auto const low = dt / spacing[d] * flux[d](i, j, k, row);
                        balance += high - low;
                        operands += std::abs(high) + std::abs(low);
                    }
#endif
                    auto const bound = tolerance * scale +
                                       128 * std::numeric_limits<amrex::Real>::epsilon() * operands;
                    auto const relative =
                        bound == 0
                            ? (balance == 0 ? 0 : std::numeric_limits<amrex::Real>::infinity())
                            : std::abs(balance) / bound;
                    if (!amrex::Math::isfinite(q) || !amrex::Math::isfinite(relative))
                    {
                        return {1, 0};
                    }
                    maximum = amrex::max(maximum, relative);
                    if (row % 4 == 0)
                    {
                        total_debit += q;
                        projection_operands += std::abs(q);
                    }
                    else
                    {
                        recoil[row % 4 - 1] += q;
                        projection_operands += std::abs(drift[row % 4 - 1] * q);
                    }
                }
                auto const projection_noise =
                    128 * std::numeric_limits<amrex::Real>::epsilon() * projection_operands;
                auto const projection =
                    total_debit - exact_group_detail::Dot(drift, recoil) - total_heat;
                if (!amrex::Math::isfinite(total_heat) || std::abs(projection) > projection_noise)
                {
                    return {1, maximum};
                }
                heat(i, j, k) = total_heat;
                arithmetic(i, j, k, 0) = heat_noise;
                arithmetic(i, j, k, 1) = projection_noise;
                arithmetic(i, j, k, 2) = momentum_noise;
                return {0, maximum};
            });
    }
    int invalid = amrex::get<0>(data.value());
    result.equation_residual = amrex::get<1>(data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    amrex::ParallelDescriptor::ReduceRealMax(result.equation_residual);
    if (invalid || result.equation_residual > 1)
    {
        return fail("Multigroup spatial cone, source projection or equation rejected");
    }
    for (int row = 0; row < size; ++row)
    {
        auto change = result.radiation.sum(row) - old.sum(row) + result.material_transfer.sum(row);
        auto scale =
            result.radiation.norm1(row) + old.norm1(row) + result.material_transfer.norm1(row);
#if defined(WARPX_DIM_RZ)
        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                         amrex::ReduceOpSum, amrex::ReduceOpSum>
            inventory_ops;
        amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real>
            inventory_data(inventory_ops);
        for (amrex::MFIter it(result.radiation); it.isValid(); ++it)
        {
            auto const flux = op.m_flux[0].const_array(it), stress = op.m_stress.const_array(it);
            auto const a = result.radiation.const_array(it), b = old.const_array(it);
            auto const source = result.material_transfer.const_array(it);
            inventory_ops.eval(
                it.validbox(), inventory_data,
                [=] AMREX_GPU_DEVICE(
                    int i, int j, int k) -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real,
                                                            amrex::Real, amrex::Real>
                {
                    amrex::Real boundary = 0, boundary_scale = 0, angular_boundary = 0;
                    auto const low = dt * metric.Area(0, i) * flux(i, j, k, row);
                    auto const high = dt * metric.Area(0, i + 1) * flux(i + 1, j, k, row);
                    if (i == metric.m_lo.x)
                    {
                        boundary -= low;
                        boundary_scale += std::abs(low);
                        angular_boundary -= metric.Radius(i) * low;
                    }
                    if (i == metric.m_hi.x)
                    {
                        boundary += high;
                        boundary_scale += std::abs(high);
                        angular_boundary += metric.Radius(i + 1) * high;
                    }
                    amrex::Real geometric = 0;
                    if (row % 4 == 1)
                    {
                        geometric = dt * PhysConst::c *
                                    (metric.Area(0, i + 1) - metric.Area(0, i)) *
                                    stress(i, j, k, 6 * (row / 4) + 1);
                    }
                    if (row % 4 == 2)
                    {
                        geometric = metric.Angular(i).Geometric(flux(i, j, k, row),
                                                                flux(i + 1, j, k, row), dt);
                    }
                    auto const radius = metric.Angular(i).mean_radius;
                    auto const angular_change =
                        radius * ((a(i, j, k, row) - b(i, j, k, row)) + source(i, j, k, row)) +
                        angular_boundary;
                    auto const angular_scale =
                        radius * (std::abs(a(i, j, k, row)) + std::abs(b(i, j, k, row)) +
                                  std::abs(source(i, j, k, row))) +
                        metric.Radius(metric.m_hi.x + 1) * boundary_scale;
                    return {boundary, geometric, boundary_scale + std::abs(geometric),
                            angular_change, angular_scale};
                });
        }
        auto const inventory = inventory_data.value();
        amrex::Real values[5]{amrex::get<0>(inventory), amrex::get<1>(inventory),
                              amrex::get<2>(inventory), amrex::get<3>(inventory),
                              amrex::get<4>(inventory)};
        amrex::ParallelDescriptor::ReduceRealSum(values, 5);
        result.boundary_exchange[row % 4] += values[0];
        result.geometric_exchange[row % 4] += values[1];
        change += values[0] - values[1];
        scale += values[2];
        if (row % 4 == 2)
        {
            auto const angular_relative =
                values[4] == 0 ? std::abs(values[3]) : std::abs(values[3]) / values[4];
            result.angular_inventory_residual =
                std::max(result.angular_inventory_residual, angular_relative);
        }
#endif
        auto const relative = scale == 0 ? std::abs(change) : std::abs(change) / scale;
        result.inventory_residual = std::max(result.inventory_residual, relative);
    }
    if (!std::isfinite(result.inventory_residual) || result.inventory_residual > tolerance ||
        !std::isfinite(result.angular_inventory_residual) ||
        result.angular_inventory_residual > tolerance)
    {
        return fail("Multigroup spatial closed inventory rejected");
    }
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        result.face_current[d] = std::make_unique<amrex::MultiFab>(op.m_flux[d].boxArray(),
                                                                   old.DistributionMap(), size, 0);
        amrex::MultiFab::Copy(*result.face_current[d], op.m_flux[d], 0, 0, size, 0);
#if !defined(WARPX_DIM_RZ)
        result.face_current[d]->mult(1 / volume);
#else
        amrex::ignore_unused(volume);
#endif
    }
#if defined(WARPX_DIM_RZ)
    for (int g = 0; g < n; ++g)
    {
        amrex::MultiFab lab(result.radiation, amrex::make_alias, 4 * g, 4);
        amrex::MultiFab comoving(result.comoving_radiation, amrex::make_alias, 4 * g, 4);
        FillRZMomentGhosts(lab, geometry, 1);
        FillRZMomentGhosts(comoving, geometry, 1);
    }
#else
    result.radiation.FillBoundary(geometry.periodicity());
    result.comoving_radiation.FillBoundary(geometry.periodicity());
#endif
    result.valid = true;
    return result;
#endif
}
} // namespace warpx::radiation
