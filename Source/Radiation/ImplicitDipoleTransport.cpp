/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "ImplicitDipoleTransport.H"

#include "ComovingDipolePressure.H"
#include "MovingMomentFlux.H"
#include "RZMomentGeometry.H"

#include <AMReX_GMRES.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace warpx::radiation
{
namespace
{
AMREX_GPU_HOST_DEVICE amrex::IntVect Cell (int i, int j, int k)
{
    amrex::ignore_unused(j, k);
    return amrex::IntVect(AMREX_D_DECL(i, j, k));
}

AMREX_GPU_HOST_DEVICE constexpr int Axis (int d)
{
#if defined(WARPX_DIM_1D_Z)
    static_cast<void>(d);
    return 2;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    return d == 0 ? 0 : 2;
#else
    return d;
#endif
}

AMREX_GPU_HOST_DEVICE FourVector Values (
    amrex::Array4<amrex::Real const> const& a, amrex::IntVect const& cell)
{
    return {a(cell, 0), a(cell, 1), a(cell, 2), a(cell, 3)};
}

AMREX_GPU_HOST_DEVICE amrex::Real Project (
    FourVector const& u, amrex::GpuArray<amrex::Real, 3> const& beta)
{
    auto h = u[0];
    for (int d = 0; d < 3; ++d) { h -= beta[d] * u[d + 1]; }
    return h;
}

bool GeometryAdmitted (amrex::Geometry const& geometry, bool mirrors)
{
#if defined(WARPX_DIM_RZ)
    // WarpX's native RZ mesh has AMReX coord_sys=0; standalone cylindrical
    // fixtures can use1. The compile-time RZ backend owns the annular metric
    // explicitly in either case. Do not admit spherical coordinate metadata.
    return mirrors && (geometry.Coord() == 0 || geometry.Coord() == 1) &&
        geometry.ProbLo(0) == 0 &&
        !geometry.isPeriodic(0);
#else
    amrex::ignore_unused(mirrors);
    return geometry.Coord() == 0 && geometry.isAllPeriodic();
#endif
}

#if defined(WARPX_DIM_RZ)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void AccumulatePressureProduct (amrex::Real coefficient, amrex::Real input,
                                amrex::Real& sum, amrex::Real& correction)
{
    auto const product = coefficient * input;
    auto const next = sum + product;
    correction += std::abs(sum) >= std::abs(product)
        ? (sum - next) + product : (product - next) + sum;
    correction += std::fma(coefficient, input, -product);
    sum = next;
}

void FillScalarMirrors (amrex::MultiFab& field, amrex::Geometry const& geometry)
{
    field.FillBoundary(geometry.periodicity());
    auto const lo = amrex::lbound(geometry.Domain());
    auto const hi = amrex::ubound(geometry.Domain());
    bool const periodic_z = geometry.isPeriodic(1);
    for (amrex::MFIter it(field); it.isValid(); ++it)
    {
        auto const a = field.array(it);
        amrex::ParallelFor(it.fabbox(), field.nComp(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                int ii = i, jj = j;
                if (i < lo.x) { ii = 2 * lo.x - i - 1; }
                else if (i > hi.x) { ii = 2 * hi.x - i + 1; }
                if (!periodic_z && (j < lo.y || j > hi.y))
                { jj = j < lo.y ? 2 * lo.y - j - 1 : 2 * hi.y - j + 1; }
                if (ii != i || jj != j) { a(i, j, k, n) = a(ii, jj, k, n); }
            });
    }
}
#endif

/** Linear operator with frozen physical coefficients, not an M1 iteration.
 * Cache exact signed pressure columns and shared face flux columns once.
 * The source rows have separate absorption/total-rate normalization. */
class DipoleOperator
{
public:
    using RT = amrex::Real;
    using Arrays = amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM>;

    DipoleOperator (amrex::MultiFab const& old, amrex::MultiFab const& beta,
                    amrex::MultiFab const& absorption, amrex::MultiFab const& scattering,
                    amrex::MultiFab const& bath, amrex::Geometry const& geom, RT dt,
                    bool angular_transport = false)
        : m_geom(geom), m_dt(dt), m_angular_transport(angular_transport),
          m_beta(old.boxArray(), old.DistributionMap(), 3, 1),
          m_coeff(old.boxArray(), old.DistributionMap(), 3, 1),
          m_map(old.boxArray(), old.DistributionMap(), 36, 1),
          m_local(old.boxArray(), old.DistributionMap(), 16, 0)
    {
        amrex::MultiFab::Copy(m_beta, beta, 0, 0, 3, 0);
        amrex::MultiFab::Copy(m_coeff, absorption, 0, 0, 1, 0);
        amrex::MultiFab::Copy(m_coeff, scattering, 0, 1, 1, 0);
        amrex::MultiFab::Copy(m_coeff, bath, 0, 2, 1, 0);
        m_beta.FillBoundary(m_geom.periodicity());
        m_coeff.FillBoundary(m_geom.periodicity());
#if defined(WARPX_DIM_RZ)
        FillRZMomentGhosts(m_beta, m_geom, 0);
        FillScalarMirrors(m_coeff, m_geom);
        m_density.define(old.boxArray(), old.DistributionMap(), 4, 1);
        m_geometry_terms.define(old.boxArray(), old.DistributionMap(), 4, 0);
        m_boundary_terms.define(old.boxArray(), old.DistributionMap(), 4, 0);
#endif
    }

    bool initialize (amrex::MultiFab const& old)
    {
        amrex::ReduceOps<amrex::ReduceOpMax> ops;
        amrex::ReduceData<int> data(ops);
        using Tuple = typename decltype(data)::Type;
        auto const dt = m_dt;
        [[maybe_unused]] bool const angular = m_angular_transport;
        for (amrex::MFIter it(old); it.isValid(); ++it)
        {
            auto const b = m_beta.const_array(it), c = m_coeff.const_array(it);
            auto const u = old.const_array(it);
            ops.eval(it.validbox(), data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Tuple
                {
                    auto const p = Cell(i, j, k);
                    amrex::GpuArray<RT, 3> const beta{b(p, 0), b(p, 1), b(p, 2)};
#if defined(WARPX_DIM_RZ)
                    if (!angular && (beta[1] != 0 || u(p, 2) != 0)) { return {1}; }
#endif
                    if (!EvaluateComovingDipolePressure(Values(u, p), beta).valid) { return {1}; }
                    for (int d = 0; d < 3; ++d)
                    {
                        if (!(c(p, d) >= 0) || !amrex::Math::isfinite(c(p, d))) { return {1}; }
                    }
                    return {!amrex::Math::isfinite(PhysConst::c * dt * (c(p, 0) + c(p, 1)))};
                });
        }
        int bad = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceIntMax(bad);
        if (bad) { return false; }
        for (amrex::MFIter it(m_map); it.isValid(); ++it)
        {
            auto const b = m_beta.const_array(it);
            auto const out = m_map.array(it);
            amrex::ParallelFor(it.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                auto const p = Cell(i, j, k);
                amrex::GpuArray<RT, 3> const beta{b(p, 0), b(p, 1), b(p, 2)};
                for (int col = 0; col < 4; ++col)
                {
                    FourVector unit{};
                    unit[col] = 1;
                    auto const pressure = EvaluateComovingDipolePressure(unit, beta, false);
                    for (int row = 0; row < 3; ++row)
                    {
                        for (int d = 0; d < 3; ++d)
                        { out(p, 9 * col + 3 * row + d) = pressure.tensor[row + 1][d + 1]; }
                    }
                }
            });
        }
        buildLocal();
        buildFaces();
        bool valid = m_local.is_finite();
        for (auto const& weights : m_weights) { valid = weights.is_finite() && valid; }
        return valid;
    }

    void buildLocal ()
    {
        auto const dt = m_dt;
        for (amrex::MFIter it(m_local); it.isValid(); ++it)
        {
            auto const b = m_beta.const_array(it), c = m_coeff.const_array(it);
            auto const out = m_local.array(it);
            amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                auto const p = Cell(i, j, k);
                amrex::GpuArray<RT, 3> const beta{b(p, 0), b(p, 1), b(p, 2)};
                StressEnergy moment{}, force{};
                BuildIsotropicDipoleFrameMatrices(beta, moment, force);
                auto const a = PhysConst::c * dt * c(p, 0);
                auto const t = PhysConst::c * dt * (c(p, 0) + c(p, 1));
                auto const inverse_a = 1 / (1 + a), inverse_t = 1 / (1 + t);
                RT beta2 = 0;
                for (auto v : beta) { beta2 += v * v; }
                for (int col = 0; col < 4; ++col)
                {
                    FourVector unit{};
                    unit[col] = 1;
                    auto const proper = EvaluateComovingDipolePressure(unit, beta, false).proper;
                    out(p, col) = inverse_a * Project(unit, beta) +
                        (a * inverse_a) * std::sqrt(1 - beta2) * proper[0];
                    for (int row = 1; row < 4; ++row)
                    {
                        RT value = inverse_t * unit[row] +
                            (a * inverse_t) * force[row][0] * proper[0];
                        for (int d = 1; d < 4; ++d)
                        { value += (t * inverse_t) * force[row][d] * proper[d]; }
                        out(p, 4 * row + col) = value;
                    }
                }
            });
        }
    }

    AMREX_GPU_HOST_DEVICE static StressEnergy Pressure (
        amrex::Array4<RT const> const& map, amrex::IntVect const& p, int col)
    {
        StressEnergy tensor{};
        for (int row = 0; row < 3; ++row)
        {
            for (int d = 0; d < 3; ++d)
            { tensor[row + 1][d + 1] = map(p, 9 * col + 3 * row + d); }
        }
        return tensor;
    }

    void buildFaces ()
    {
        auto const dx = m_geom.CellSizeArray();
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
        {
            auto const boxes = amrex::convert(m_beta.boxArray(),
                                              amrex::IntVect::TheDimensionVector(direction));
            m_weights[direction].define(boxes, m_beta.DistributionMap(), 35, 0);
            m_flux[direction].define(boxes, m_beta.DistributionMap(), 8, 0);
            auto const normal = Axis(direction);
            for (amrex::MFIter it(m_weights[direction]); it.isValid(); ++it)
            {
                auto const b = m_beta.const_array(it), c = m_coeff.const_array(it);
                auto const map = m_map.const_array(it);
                auto const w = m_weights[direction].array(it);
                amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    auto const right = Cell(i, j, k);
                    auto left = right;
                    --left[direction];
                    amrex::GpuArray<RT, 3> beta{};
                    for (int d = 0; d < 3; ++d) { beta[d] = (b(left, d) + b(right, d)) / 2; }
                    auto const opacity = RT(.5) * c(left, 0) + RT(.5) * c(left, 1) +
                        RT(.5) * c(right, 0) + RT(.5) * c(right, 1);
                    for (int side = 0; side < 2; ++side)
                    {
                        for (int col = 0; col < 4; ++col)
                        {
                            FourVector unit{};
                            unit[col] = 1;
                            auto const pressure = Pressure(map, side == 0 ? left : right, col);
                            auto const face = detail::MovingMomentFluxWithPressure(
                                side == 0 ? unit : FourVector{}, side == 1 ? unit : FourVector{},
                                side == 0 ? pressure : StressEnergy{},
                                side == 1 ? pressure : StressEnergy{}, beta, normal,
                                opacity, dx[direction]);
                            for (int row = 0; row < 4; ++row)
                            {
                                w(right, 16 * side + 4 * row + col) = face.valid ? face.flux[row] :
                                    std::numeric_limits<RT>::quiet_NaN();
                            }
                        }
                    }
                    for (int d = 0; d < 3; ++d)
                    {
                        amrex::GpuArray<RT, 3> gradient{};
                        gradient[d] = 1;
                        w(right, 32 + d) = detail::MovingMomentFluxWithPressure(
                            {}, {}, {}, {}, beta, normal, opacity, dx[direction], gradient).flux[0];
                    }
                });
            }
        }
    }

    void transport (amrex::MultiFab& out, amrex::MultiFab const& state)
    {
        // All states passed here are caller-private GMRES/candidate buffers.
        const_cast<amrex::MultiFab&>(state).FillBoundary(m_geom.periodicity());
        auto const dx = m_geom.CellSizeArray();
#if defined(WARPX_DIM_RZ)
        PrepareRZMomentDensity(m_density, state, m_geom);
        auto const domain_lo = amrex::lbound(m_geom.Domain());
        auto const domain_hi = amrex::ubound(m_geom.Domain());
        bool const periodic_z = m_geom.isPeriodic(1);
#endif
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
        {
            for (amrex::MFIter it(m_flux[direction]); it.isValid(); ++it)
            {
#if defined(WARPX_DIM_RZ)
                auto const u = m_density.const_array(it);
#else
                auto const u = state.const_array(it);
#endif
                auto const w = m_weights[direction].const_array(it);
                auto const b = m_beta.const_array(it);
                auto const f = m_flux[direction].array(it);
                amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    auto const right = Cell(i, j, k);
                    auto left = right;
                    --left[direction];
                    amrex::GpuArray<RT, 3> beta{};
                    for (int d = 0; d < 3; ++d) { beta[d] = (b(left, d) + b(right, d)) / 2; }
                    for (int row = 0; row < 4; ++row)
                    {
#if defined(WARPX_DIM_RZ)
                        bool const mirror = direction == 0
                            ? (i == domain_lo.x || i == domain_hi.x + 1)
                            : (!periodic_z && (j == domain_lo.y || j == domain_hi.y + 1));
                        if (mirror && row != Axis(direction) + 1)
                        {
                            // Exact specular parity: no energy or tangential
                            // stress crosses a stationary optical mirror.
                            f(right, row) = f(right, row + 4) = 0;
                            continue;
                        }
#endif
                        RT value = 0, scale = 0;
                        for (int col = 0; col < 4; ++col)
                        {
                            auto const low = w(right, 4 * row + col) * u(left, col);
                            auto const high = w(right, 16 + 4 * row + col) * u(right, col);
                            value += low + high;
                            scale += std::abs(low) + std::abs(high);
                        }
                        if (row == 0)
                        {
                            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            {
                                if (d == direction) { continue; }
                                auto const coefficient = w(right, 32 + Axis(d)) / (4 * dx[d]);
                                for (int side = 0; side < 2; ++side)
                                {
                                    for (int sign = -1; sign <= 1; sign += 2)
                                    {
                                        auto p = side == 0 ? left : right;
                                        p[d] += sign;
                                        auto const v = Values(u, p);
                                        value += sign * coefficient * Project(v, beta);
                                        scale += std::abs(coefficient * v[0]);
                                        for (int e = 0; e < 3; ++e)
                                        { scale += std::abs(coefficient * beta[e] * v[e + 1]); }
                                    }
                                }
                            }
                        }
                        f(right, row) = value;
                        f(right, row + 4) = scale;
                    }
                });
            }
        }
        auto const dt = m_dt;
#if defined(WARPX_DIM_RZ)
        RZMomentMetric const metric(m_geom);
        bool const angular = m_angular_transport;
        m_geometry_terms.setVal(0);
        m_boundary_terms.setVal(0);
#endif
        for (amrex::MFIter it(out); it.isValid(); ++it)
        {
            Arrays faces{};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) { faces[d] = m_flux[d].const_array(it); }
            auto const output = out.array(it);
#if defined(WARPX_DIM_RZ)
            auto const density = m_density.const_array(it), map = m_map.const_array(it);
            auto const radial_weights = m_weights[0].const_array(it);
            auto const geometry_terms = m_geometry_terms.array(it);
            auto const boundary_terms = m_boundary_terms.array(it);
#endif
            amrex::ParallelFor(it.validbox(), 4,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int row)
                {
                    auto const p = Cell(i, j, k);
                    RT value = 0, scale = 0;
#if defined(WARPX_DIM_RZ)
                    RT connection = 0, connection_scale = 0;
                    if (row == 1)
                    {
                        for (int col = 0; col < 4; ++col)
                        {
                            auto const term = PhysConst::c * map(p, 9 * col + 4) * density(p, col);
                            connection += term;
                            connection_scale += std::abs(term);
                        }
                        geometry_terms(p, row) = dt *
                            (metric.Area(0, i + 1) - metric.Area(0, i)) * connection;
                    }
#endif
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        auto high = p;
                        ++high[d];
#if defined(WARPX_DIM_RZ)
                        auto const low_area = metric.Area(d, i);
                        auto const high_area = metric.Area(d, high[0]);
                        if (d == 0 && row == 2 && angular)
                        {
                            // Qtheta is a local cylindrical component. The
                            // conserved axial torque is rbar*Qtheta, so radial
                            // transport needs r_face/rbar in its face areas.
                            // Keep the ordinary-minus-angular basis exchange
                            // separate from collision/material momentum.
                            auto const weights = metric.Angular(i);
                            value += weights.Divergence(faces[d](p, row),
                                                         faces[d](high, row), dt);
                            scale += weights.Magnitude(faces[d](p, row + 4),
                                                        faces[d](high, row + 4), dt);
                            geometry_terms(p, row) = weights.Geometric(
                                faces[d](p, row), faces[d](high, row), dt);
                        }
                        else if (d == 0)
                        {
                            // Subtract the cell's Ptheta before area products:
                            // an isotropic resting bath is exactly balanced at
                            // the axis, rather than cancelling large forces.
                            if (row == 1 && angular)
                            {
                                // Evaluate the identical cached linear face minus
                                // cell-connection products before rounding either
                                // O(c*E/3) inventory to one scalar. Keep full faces
                                // unchanged for boundary and geometric ledgers.
                                RT difference[2]{};
                                for (int side = 0; side < 2; ++side)
                                {
                                    auto const face = side == 0 ? p : high;
                                    auto left = face;
                                    --left[0];
                                    RT sum = 0, correction = 0;
                                    for (int col = 0; col < 4; ++col)
                                    {
                                        AccumulatePressureProduct(
                                            radial_weights(face, 4 * row + col),
                                            density(left, col), sum, correction);
                                        AccumulatePressureProduct(
                                            radial_weights(face, 16 + 4 * row + col),
                                            density(face, col), sum, correction);
                                        AccumulatePressureProduct(
                                            -(PhysConst::c * map(p, 9 * col + 4)),
                                            density(p, col), sum, correction);
                                    }
                                    difference[side] = sum + correction;
                                }
                                value += dt * (high_area * difference[1] -
                                               low_area * difference[0]);
                            }
                            else
                            {
                                value += dt * (high_area * (faces[d](high, row) - connection) -
                                               low_area * (faces[d](p, row) - connection));
                            }
                        }
                        else
                        {
                            value += dt * low_area *
                                (faces[d](high, row) - faces[d](p, row));
                        }
                        if (!(d == 0 && row == 2 && angular))
                        {
                            scale += dt * (high_area * faces[d](high, row + 4) +
                                           low_area * faces[d](p, row + 4));
                        }
                        if (d == 0) { scale += dt * (high_area + low_area) * connection_scale; }
                        auto const low_index = d == 0 ? domain_lo.x : domain_lo.y;
                        auto const high_index = d == 0 ? domain_hi.x : domain_hi.y;
                        bool const physical = d == 0 || !periodic_z;
                        if (physical && p[d] == low_index)
                        { boundary_terms(p, row) -= dt * low_area * faces[d](p, row); }
                        if (physical && p[d] == high_index)
                        { boundary_terms(p, row) += dt * high_area * faces[d](high, row); }
#else
                        value += (dt / dx[d]) * (faces[d](high, row) - faces[d](p, row));
                        scale += (dt / dx[d]) *
                            (faces[d](high, row + 4) + faces[d](p, row + 4));
#endif
                    }
                    output(p, row) = value;
                    if (output.nComp() >= 8) { output(p, row + 4) = scale; }
                });
        }
    }

    /** Apply the frozen linear transport to a retained old + increment pair.
     * Preserve sub-ULP state response until the source/transport transaction;
     * published radiation remains the ordinary represented old + increment.
     * Face, wall and geometric accounts have the same linear decomposition. */
    void transportIncrement (amrex::MultiFab& out, amrex::MultiFab const& old,
                             amrex::MultiFab const& increment)
    {
        transport(out, old);
#if defined(WARPX_DIM_RZ)
        amrex::MultiFab old_geometry(m_geometry_terms.boxArray(),
            m_geometry_terms.DistributionMap(), 4, 0);
        amrex::MultiFab old_boundary(m_boundary_terms.boxArray(),
            m_boundary_terms.DistributionMap(), 4, 0);
        amrex::MultiFab::Copy(old_geometry, m_geometry_terms, 0, 0, 4, 0);
        amrex::MultiFab::Copy(old_boundary, m_boundary_terms, 0, 0, 4, 0);
#endif
        amrex::MultiFab increment_transport(out.boxArray(), out.DistributionMap(),
                                           out.nComp(), 0);
        transport(increment_transport, increment);
        // Components4:7 are uncancelled operand sums. Their sum remains a
        // conservative bound for this actual split evaluation, not a defect.
        amrex::MultiFab::Add(out, increment_transport, 0, 0, out.nComp(), 0);
#if defined(WARPX_DIM_RZ)
        amrex::MultiFab::Add(m_geometry_terms, old_geometry, 0, 0, 4, 0);
        amrex::MultiFab::Add(m_boundary_terms, old_boundary, 0, 0, 4, 0);
#endif
    }

    void apply (amrex::MultiFab& out, amrex::MultiFab const& state)
    {
        transport(out, state);
        auto const dt = m_dt;
        for (amrex::MFIter it(out); it.isValid(); ++it)
        {
            auto const u = state.const_array(it), b = m_beta.const_array(it);
            auto const c = m_coeff.const_array(it), matrix = m_local.const_array(it);
            auto const output = out.array(it);
            amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                auto const p = Cell(i, j, k);
                amrex::GpuArray<RT, 3> const beta{b(p, 0), b(p, 1), b(p, 2)};
                auto const flux = Values(output, p);
                auto const a = PhysConst::c * dt * c(p, 0);
                auto const t = PhysConst::c * dt * (c(p, 0) + c(p, 1));
                for (int row = 0; row < 4; ++row)
                {
                    RT value = row == 0 ? Project(flux, beta) / (1 + a) : flux[row] / (1 + t);
                    for (int col = 0; col < 4; ++col)
                    { value += matrix(p, 4 * row + col) * u(p, col); }
                    output(p, row) = value;
                }
            });
        }
    }

    /** Initial increment RHS with the old-state identity cancelled analytically.
     * This is rhs - A(old) in the same absorption/total-rate normalization,
     * without subtracting two extensive old-energy values. */
    bool initialIncrementRhs (amrex::MultiFab& out, amrex::MultiFab const& old)
    {
        transport(out, old);
        auto const dt = m_dt;
        for (amrex::MFIter it(out); it.isValid(); ++it)
        {
            auto const u = old.const_array(it), b = m_beta.const_array(it);
            auto const c = m_coeff.const_array(it);
            auto const output = out.array(it);
            amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                auto const p = Cell(i, j, k);
                amrex::GpuArray<RT, 3> const beta{b(p, 0), b(p, 1), b(p, 2)};
                auto const flux = Values(output, p);
                auto const pressure = EvaluateComovingDipolePressure(Values(u, p), beta);
                auto const a = PhysConst::c * dt * c(p, 0);
                auto const t = PhysConst::c * dt * (c(p, 0) + c(p, 1));
                auto const inverse_a = 1 / (1 + a), inverse_t = 1 / (1 + t);
                RT beta2 = 0;
                for (auto value : beta) { beta2 += value * value; }
                auto const inverse_gamma = std::sqrt(1 - beta2);
                // Absorption and coherent scattering have distinct normalized
                // rows. The projected energy row has no scattering term.
                auto const contrast = pressure.proper[0] - c(p, 2);
                FourVector rest_force{};
                rest_force[0] = (a * inverse_t) * contrast;
                for (int d = 0; d < 3; ++d)
                { rest_force[d + 1] = (t * inverse_t) * pressure.proper[d + 1]; }
                auto const force = BoostFourVector(rest_force, beta);
                output(p, 0) = -inverse_a * Project(flux, beta) -
                    (a * inverse_a) * inverse_gamma * contrast;
                for (int row = 1; row < 4; ++row)
                { output(p, row) = -inverse_t * flux[row] - force[row]; }
                if (!pressure.valid)
                {
                    for (int row = 0; row < 4; ++row)
                    { output(p, row) = std::numeric_limits<RT>::quiet_NaN(); }
                }
            });
        }
        return out.is_finite();
    }

    void rhs (amrex::MultiFab& out, amrex::MultiFab const& old)
    {
        auto const dt = m_dt;
        for (amrex::MFIter it(out); it.isValid(); ++it)
        {
            auto const u = old.const_array(it), b = m_beta.const_array(it);
            auto const c = m_coeff.const_array(it);
            auto const output = out.array(it);
            amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                auto const p = Cell(i, j, k);
                amrex::GpuArray<RT, 3> const beta{b(p, 0), b(p, 1), b(p, 2)};
                RT beta2 = 0;
                for (auto v : beta) { beta2 += v * v; }
                auto const a = PhysConst::c * dt * c(p, 0);
                auto const t = PhysConst::c * dt * (c(p, 0) + c(p, 1));
                auto const inverse_a = 1 / (1 + a), inverse_t = 1 / (1 + t);
                auto const inverse_gamma = std::sqrt(1 - beta2);
                output(p, 0) = inverse_a * Project(Values(u, p), beta) +
                    (a * inverse_a) * inverse_gamma * c(p, 2);
                for (int row = 1; row < 4; ++row)
                {
                    output(p, row) = inverse_t * u(p, row) +
                        (a * inverse_t) * beta[row - 1] * c(p, 2) / inverse_gamma;
                }
            });
        }
    }

    void precond (amrex::MultiFab& out, amrex::MultiFab const& rhs)
    {
        for (amrex::MFIter it(out); it.isValid(); ++it)
        {
            auto const matrix = m_local.const_array(it), forcing = rhs.const_array(it);
            auto const output = out.array(it);
            amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                StressEnergy block{};
                FourVector solution{};
                for (int row = 0; row < 4; ++row)
                {
                    for (int col = 0; col < 4; ++col)
                    { block[row][col] = matrix(i, j, k, 4 * row + col); }
                }
                auto const input = Values(forcing, Cell(i, j, k));
                bool const solved = gray_absorption_detail::Solve(block, input, solution);
                for (int row = 0; row < 4; ++row)
                { output(i, j, k, row) = solved ? solution[row] : input[row]; }
            });
        }
    }

    amrex::MultiFab makeVecLHS () const
    {
        amrex::MultiFab result(m_beta.boxArray(), m_beta.DistributionMap(), 4, 1);
#if defined(WARPX_DIM_RZ)
        // Physical ghost moments are not the face authority: annular density
        // has its own metric/parity cache. Initialize these private, unused
        // ghosts nevertheless, because MultiFab::is_finite() checks nGrowVect
        // by default. Periodic FillBoundary alone cannot initialize a radial
        // physical ghost and must not make a valid solve depend on allocator
        // residue. Keep the full finiteness screen and caller ghost contract.
        result.setVal(0);
#endif
        return result;
    }
    amrex::MultiFab makeVecRHS () const
    { return amrex::MultiFab(m_beta.boxArray(), m_beta.DistributionMap(), 4, 0); }
    static void assign (amrex::MultiFab& a, amrex::MultiFab const& b)
    { amrex::MultiFab::Copy(a, b, 0, 0, 4, 0); }
    static RT dotProduct (amrex::MultiFab const& a, amrex::MultiFab const& b)
    { return amrex::MultiFab::Dot(a, 0, b, 0, 4, 0); }
    static RT norm2 (amrex::MultiFab const& a) { return std::sqrt(dotProduct(a, a)); }
    static void increment (amrex::MultiFab& a, amrex::MultiFab const& b, RT f)
    { amrex::MultiFab::Saxpy(a, f, b, 0, 0, 4, 0); }
    static void linComb (amrex::MultiFab& a, RT f, amrex::MultiFab const& b,
                         RT g, amrex::MultiFab const& c)
    { amrex::MultiFab::LinComb(a, f, b, 0, g, c, 0, 0, 4, 0); }
    static void scale (amrex::MultiFab& a, RT f) { a.mult(f, 0, 4, 0); }
    static void setToZero (amrex::MultiFab& a) { a.setVal(0); }

    FourVector boundaryExchange () const
    {
        FourVector result{};
#if defined(WARPX_DIM_RZ)
        for (int row = 0; row < 4; ++row) { result[row] = m_boundary_terms.sum(row); }
#endif
        return result;
    }
    FourVector geometricExchange () const
    {
        FourVector result{};
#if defined(WARPX_DIM_RZ)
        for (int row = 0; row < 4; ++row) { result[row] = m_geometry_terms.sum(row); }
#endif
        return result;
    }

    amrex::Geometry m_geom;
    RT m_dt;
    bool m_angular_transport;
    amrex::MultiFab m_beta, m_coeff, m_map, m_local;
    std::array<amrex::MultiFab, AMREX_SPACEDIM> m_weights, m_flux;
#if defined(WARPX_DIM_RZ)
    amrex::MultiFab m_density, m_geometry_terms, m_boundary_terms;
#endif
};

bool Layout (amrex::MultiFab const& a, amrex::MultiFab const& reference, int components)
{
    return a.nComp() == components && a.ixType().cellCentered() &&
        a.boxArray() == reference.boxArray() && a.DistributionMap() == reference.DistributionMap();
}

bool SharesStorage (amrex::MultiFab const& a, amrex::MultiFab const& b)
{
    // Same-layout make_alias views may be distinct objects and start at a
    // nonzero component. Compare allocated ranges, not object addresses.
    for (amrex::MFIter it(a); it.isValid(); ++it)
    {
        auto const low_a = reinterpret_cast<std::uintptr_t>(a[it].dataPtr());
        auto const low_b = reinterpret_cast<std::uintptr_t>(b[it].dataPtr());
        auto const high_a = low_a + a[it].box().numPts() * a.nComp() * sizeof(amrex::Real);
        auto const high_b = low_b + b[it].box().numPts() * b.nComp() * sizeof(amrex::Real);
        if (low_a < high_b && low_b < high_a) { return true; }
    }
    return false;
}
} // namespace

ImplicitDipoleTransportResult TryImplicitDipoleTransport (
    amrex::MultiFab& radiation, amrex::MultiFab const& beta,
    amrex::MultiFab const& absorption, amrex::MultiFab const& scattering,
    amrex::MultiFab const& equilibrium, amrex::MultiFab& transfer,
    amrex::MultiFab& heat, amrex::Geometry const& geometry, amrex::Real dt,
    ImplicitDipoleTransportOptions const& options)
{
    ImplicitDipoleTransportResult result;
    result.equation_ratio = result.conservation_ratio =
        std::numeric_limits<amrex::Real>::infinity();
    bool const valid_layout = radiation.nComp() == 4 && radiation.ixType().cellCentered() &&
        !radiation.boxArray().empty() && radiation.boxArray().isDisjoint() &&
        Layout(beta, radiation, 3) && Layout(absorption, radiation, 1) &&
        Layout(scattering, radiation, 1) && Layout(equilibrium, radiation, 1) &&
        Layout(transfer, radiation, 4) && Layout(heat, radiation, 1) &&
        &radiation != &transfer && &heat != &equilibrium && &heat != &absorption &&
        &heat != &scattering && radiation.boxArray().minimalBox() == geometry.Domain() &&
        radiation.boxArray().numPts() == geometry.Domain().numPts();
    int invalid = !valid_layout || !GeometryAdmitted(geometry, options.reflecting_boundaries) ||
        !(dt > 0) || !std::isfinite(dt) || options.max_iterations < 1 ||
        options.restart_length < 1 ||
        options.max_refinements < 0 || !(options.linear_tolerance > 0) ||
        !(options.equation_tolerance > 0) || !std::isfinite(options.linear_tolerance) ||
        !std::isfinite(options.equation_tolerance);
#if !defined(WARPX_DIM_RZ)
    invalid = invalid || options.rz_angular_transport;
#endif
    if (valid_layout)
    {
        std::array<amrex::MultiFab const*, 3> const outputs{&radiation, &transfer, &heat};
        std::array<amrex::MultiFab const*, 4> const inputs{
            &beta, &absorption, &scattering, &equilibrium};
        for (std::size_t a = 0; a < outputs.size(); ++a)
        {
            for (std::size_t b = a + 1; b < outputs.size(); ++b)
            { invalid = SharesStorage(*outputs[a], *outputs[b]) || invalid; }
            for (auto const* input : inputs)
            { invalid = SharesStorage(*outputs[a], *input) || invalid; }
        }
    }
    // A rank-local solver policy is not authority to diverge collective flow.
    amrex::Real const configuration[] = {dt, options.linear_tolerance,
        options.equation_tolerance, amrex::Real(options.max_iterations),
        amrex::Real(options.max_refinements), amrex::Real(options.restart_length),
        amrex::Real(options.reflecting_boundaries), amrex::Real(options.rz_angular_transport)};
    for (auto value : configuration)
    {
        auto low = value, high = value;
        amrex::ParallelDescriptor::ReduceRealMin(low);
        amrex::ParallelDescriptor::ReduceRealMax(high);
        invalid = invalid || !std::isfinite(value) || low != high;
    }
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    if (invalid) { return result; }
    DipoleOperator op(radiation, beta, absorption, scattering, equilibrium, geometry, dt,
                       options.rz_angular_transport);
    result.failure = ImplicitDipoleFailure::Coefficients;
    if (!op.initialize(radiation)) { return result; }
    auto candidate = op.makeVecLHS();
    DipoleOperator::assign(candidate, radiation);
    auto rhs = op.makeVecRHS(), defect = op.makeVecRHS();
    auto correction = op.makeVecLHS();
    op.rhs(rhs, radiation);
    amrex::GMRES<amrex::MultiFab, DipoleOperator> solver;
    solver.define(op);
    // The pinned GMRES first cycle consults setMaxIters, not solve's a_its.
    // Set both so a caller's budget is a hard cap even within that cycle.
    solver.setMaxIters(options.max_iterations);
    // Preserve more spatial modes between restarts without increasing the
    // caller's total iteration cap. Small-budget refusal remains a hard cap.
    solver.setRestartLength(amrex::min(options.restart_length, options.max_iterations));
    // GMRES starts from zero. Solve the physical increment, not the large
    // stored inventory: weak late-time exchange must not inherit a relative
    // solve error scaled to the boosted LTE reservoir on every native step.
    amrex::MultiFab frozen_old;
    if (options.rz_angular_transport)
    {
        // transport refreshes private ghosts. Never pass the caller's
        // radiation, whose valid cells and ghosts must survive rejection.
        frozen_old.define(radiation.boxArray(), radiation.DistributionMap(), 4, 1);
        frozen_old.setVal(0);
        DipoleOperator::assign(frozen_old, radiation);
        if (!op.initialIncrementRhs(defect, frozen_old)) { return result; }
    }
    else
    {
        op.apply(defect, candidate);
        DipoleOperator::linComb(defect, 1, rhs, -1, defect);
    }
    amrex::MultiFab initial_defect, total_correction;
    if (options.rz_angular_transport)
    {
        initial_defect.define(radiation.boxArray(), radiation.DistributionMap(), 4, 0);
        total_correction.define(radiation.boxArray(), radiation.DistributionMap(), 4, 1);
        total_correction.setVal(0);
        DipoleOperator::assign(initial_defect, defect);
    }
    solver.solve(correction, defect, options.linear_tolerance, 0, options.max_iterations);
    if (options.rz_angular_transport)
    { DipoleOperator::assign(total_correction, correction); }
    DipoleOperator::increment(candidate, correction, 1);
    result.linear_iterations = solver.getNumIters();
    result.failure = ImplicitDipoleFailure::LinearSolve;
    if (solver.getStatus() != 0 || !candidate.is_finite()) { return result; }
    amrex::MultiFab transport(radiation.boxArray(), radiation.DistributionMap(), 8, 0);
    auto source = op.makeVecRHS();
    amrex::MultiFab thermal(radiation.boxArray(), radiation.DistributionMap(), 1, 0);
    amrex::MultiFab balances(radiation.boxArray(), radiation.DistributionMap(), 8, 0);
    for (int refinement = 0; refinement <= options.max_refinements; ++refinement)
    {
        result.refinements = refinement;
        if (options.rz_angular_transport)
        { op.transportIncrement(transport, frozen_old, total_correction); }
        else
        { op.transport(transport, candidate); }
        amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> ops;
        amrex::ReduceData<int, amrex::Real> data(ops);
        using Tuple = typename decltype(data)::Type;
        auto const tolerance = options.equation_tolerance;
        auto const epsilon = std::numeric_limits<amrex::Real>::epsilon();
        for (amrex::MFIter it(candidate); it.isValid(); ++it)
        {
            auto const u = candidate.const_array(it), old = radiation.const_array(it);
            auto const b = op.m_beta.const_array(it), c = op.m_coeff.const_array(it);
            auto const flux = transport.const_array(it);
            auto const out = source.array(it), heat_out = thermal.array(it);
            auto const balance = balances.array(it);
            ops.eval(it.validbox(), data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Tuple
                {
                    auto const p = Cell(i, j, k);
                    amrex::GpuArray<amrex::Real, 3> const velocity{b(p, 0), b(p, 1), b(p, 2)};
                    if (!EvaluateComovingDipolePressure(Values(u, p), velocity).valid)
                    { return {1, 0}; }
                    FourVector physical{}, storage_bound{};
                    amrex::Real projected_heat = 0;
                    auto const rate_a = PhysConst::c * dt * c(p, 0);
                    auto const rate_s = PhysConst::c * dt * c(p, 1);
                    if (rate_a + rate_s <= 1)
                    {
                        // The actual accepted state is the constitutive source
                        // authority in the non-stiff regime. A second local BE
                        // solve with old-minus-transport can lose a near-LTE
                        // proper flux and reject an otherwise valid spatial
                        // solve. The full independent physical balances below
                        // still audit this assignment, with no extra local-solve
                        // storage allowance. Stiff collisions need the separate
                        // increment so a stored LTE ULP is not amplified by rate.
                        auto const pressure = EvaluateComovingDipolePressure(Values(u, p), velocity);
                        auto const force = EvaluateGreyFourForce(
                            pressure.tensor, velocity, rate_a, rate_s, c(p, 2));
                        if (!force.valid) { return {2, 0}; }
                        physical = force.material_force;
                        projected_heat = GreyEnergyMinusWork(
                            pressure.tensor, velocity, rate_a, c(p, 2));
                        if (!amrex::Math::isfinite(projected_heat)) { return {2, 0}; }
                    }
                    else
                    {
                        auto const update = TryImplicitDipoleCollisionSource(
                            Values(old, p), velocity, rate_a, rate_a + rate_s,
                            c(p, 2), Values(flux, p));
                        if (!update.valid) { return {2, 0}; }
                        physical = update.physical_transfer;
                        projected_heat = update.physical_energy_minus_work;
                        storage_bound = update.storage_bound;
                    }
                    amrex::Real worst = 0;
                    for (int row = 0; row < 4; ++row)
                    {
                        auto const change = u(p, row) - old(p, row);
                        auto const residual = change + flux(p, row) + physical[row];
                        auto const physical_scale = std::abs(change) + std::abs(flux(p, row)) +
                            std::abs(physical[row]);
                        auto const noise = 2048 * epsilon *
                            (std::abs(old(p, 0)) + std::abs(u(p, 0)) + flux(p, row + 4));
                        auto const bound = tolerance * physical_scale + noise +
                            storage_bound[row];
                        if (!amrex::Math::isfinite(residual) || !amrex::Math::isfinite(bound))
                        { return {3, 0}; }
                        worst = amrex::max(worst,
                            bound > 0 ? std::abs(residual) / bound : std::abs(residual));
                        out(p, row) = physical[row];
                        balance(p, row) = change + physical[row];
                        balance(p, row + 4) = bound;
                    }
                    heat_out(p) = projected_heat;
                    return {0, worst};
                });
        }
        int bad = amrex::get<0>(data.value());
        auto worst = amrex::get<1>(data.value());
        amrex::ParallelDescriptor::ReduceIntMax(bad);
        amrex::ParallelDescriptor::ReduceRealMax(worst);
        if (bad)
        {
            result.failure = bad == 1 ? ImplicitDipoleFailure::Admissibility :
                (bad == 2 ? ImplicitDipoleFailure::CollisionSource : ImplicitDipoleFailure::Equation);
            return result;
        }
        result.failure = ImplicitDipoleFailure::Equation;
        result.equation_ratio = worst;
        result.conservation_ratio = 0;
        result.boundary_exchange = op.boundaryExchange();
        result.geometric_exchange = op.geometricExchange();
        for (int row = 0; row < 4; ++row)
        {
            auto const residual = std::abs(balances.sum(row) + result.boundary_exchange[row] -
                                           result.geometric_exchange[row]);
            auto const bound = balances.sum(row + 4);
            if (!std::isfinite(residual) || !std::isfinite(bound)) { return result; }
            result.conservation_ratio = amrex::max(result.conservation_ratio,
                bound > 0 ? residual / bound : residual);
        }
#if defined(WARPX_DIM_RZ)
        if (options.rz_angular_transport)
        {
            // Independent inventory gate: weighted cell changes/collision
            // transfers must telescope without wall torque. No geometric
            // vector correction is used to repair this physical Lz identity.
            RZMomentMetric const metric(geometry);
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> angular_ops;
            amrex::ReduceData<amrex::Real, amrex::Real> angular_data(angular_ops);
            using AngularTuple = typename decltype(angular_data)::Type;
            for (amrex::MFIter it(balances); it.isValid(); ++it)
            {
                auto const values = balances.const_array(it);
                angular_ops.eval(it.validbox(), angular_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> AngularTuple
                    {
                        auto const radius = metric.Angular(i).mean_radius;
                        return {radius * values(i, j, k, 2), radius * values(i, j, k, 6)};
                    });
            }
            auto const values = angular_data.value();
            amrex::Real sums[2]{amrex::get<0>(values), amrex::get<1>(values)};
            amrex::ParallelDescriptor::ReduceRealSum(sums, 2);
            if (!std::isfinite(sums[0]) || !std::isfinite(sums[1])) { return result; }
            result.angular_conservation_ratio = sums[1] > 0
                ? std::abs(sums[0]) / sums[1] : std::abs(sums[0]);
        }
#endif
        if (result.equation_ratio <= 1 && result.conservation_ratio <= 1 &&
            result.angular_conservation_ratio <= 1)
        {
            amrex::MultiFab::Copy(radiation, candidate, 0, 0, 4, 0);
            amrex::MultiFab::Copy(transfer, source, 0, 0, 4, 0);
            amrex::MultiFab::Copy(heat, thermal, 0, 0, 1, 0);
            result.valid = true;
            result.failure = ImplicitDipoleFailure::None;
            return result;
        }
        if (options.rz_angular_transport)
        {
            op.apply(defect, total_correction);
            DipoleOperator::linComb(defect, 1, initial_defect, -1, defect);
        }
        else
        {
            op.apply(defect, candidate);
            DipoleOperator::linComb(defect, 1, rhs, -1, defect);
        }
        correction.setVal(0);
        solver.solve(correction, defect, options.linear_tolerance, 0, options.max_iterations);
        result.linear_iterations += solver.getNumIters();
        if (solver.getStatus() != 0 || !correction.is_finite()) { return result; }
        if (options.rz_angular_transport)
        {
            DipoleOperator::increment(total_correction, correction, 1);
            DipoleOperator::assign(candidate, radiation);
            DipoleOperator::increment(candidate, total_correction, 1);
        }
        else
        { DipoleOperator::increment(candidate, correction, 1); }
    }
    return result;
}

bool ComputeDipoleTransportIncrement (
    amrex::MultiFab const& radiation, amrex::MultiFab const& beta,
    amrex::MultiFab const& absorption, amrex::MultiFab const& scattering,
    amrex::MultiFab const& equilibrium, amrex::MultiFab& increment,
    amrex::Geometry const& geometry, amrex::Real dt,
    ImplicitDipoleTransportOptions const& options,
    MomentTransportAccounting* accounting)
{
    bool const layout = radiation.nComp() == 4 && radiation.ixType().cellCentered() &&
        !radiation.boxArray().empty() && radiation.boxArray().isDisjoint() &&
        radiation.boxArray().minimalBox() == geometry.Domain() &&
        radiation.boxArray().numPts() == geometry.Domain().numPts() &&
        Layout(beta, radiation, 3) && Layout(absorption, radiation, 1) &&
        Layout(scattering, radiation, 1) && Layout(equilibrium, radiation, 1) &&
        Layout(increment, radiation, 8);
    int invalid = !layout || !GeometryAdmitted(geometry, options.reflecting_boundaries) ||
        !(dt > 0) || !std::isfinite(dt);
#if !defined(WARPX_DIM_RZ)
    invalid = invalid || options.rz_angular_transport;
#endif
    if (layout)
    {
        for (auto const* input : {&radiation, &beta, &absorption, &scattering, &equilibrium})
        { invalid = SharesStorage(increment, *input) || invalid; }
    }
    auto low = dt, high = dt;
    amrex::ParallelDescriptor::ReduceRealMin(low);
    amrex::ParallelDescriptor::ReduceRealMax(high);
    invalid = invalid || low != high;
    int mirror_low = options.reflecting_boundaries, mirror_high = mirror_low;
    amrex::ParallelDescriptor::ReduceIntMin(mirror_low);
    amrex::ParallelDescriptor::ReduceIntMax(mirror_high);
    invalid = invalid || mirror_low != mirror_high;
    int angular_low = options.rz_angular_transport, angular_high = angular_low;
    amrex::ParallelDescriptor::ReduceIntMin(angular_low);
    amrex::ParallelDescriptor::ReduceIntMax(angular_high);
    invalid = invalid || angular_low != angular_high;
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    if (invalid) { return false; }
    DipoleOperator op(radiation, beta, absorption, scattering, equilibrium, geometry, dt,
                       options.rz_angular_transport);
    if (!op.initialize(radiation)) { return false; }
    auto state = op.makeVecLHS();
    DipoleOperator::assign(state, radiation);
    amrex::MultiFab trial(radiation.boxArray(), radiation.DistributionMap(), 8, 0);
    op.transport(trial, state);
    if (!trial.is_finite()) { return false; }
    MomentTransportAccounting accepted;
    if (accounting != nullptr)
    {
        accepted.boundary = op.boundaryExchange();
        accepted.geometric = op.geometricExchange();
        for (int d = 0; d < 4; ++d)
        {
            if (!std::isfinite(accepted.boundary[d]) || !std::isfinite(accepted.geometric[d]))
            { return false; }
        }
    }
    amrex::MultiFab::Copy(increment, trial, 0, 0, 8, 0);
    if (accounting != nullptr) { *accounting = accepted; }
    return true;
}
} // namespace warpx::radiation
