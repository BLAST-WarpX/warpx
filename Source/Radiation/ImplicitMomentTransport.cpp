/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "ImplicitMomentTransport.H"

#include "ImplicitMomentSource.H"
#include "MomentBoundaryFlux.H"
#include "MovingMomentFlux.H"
#include "Utils/TextMsg.H"

#include <AMReX_BCUtil.H>
#include <AMReX_GMRES.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <array>
#include <cmath>
#include <limits>

using namespace amrex::literals;

namespace warpx::radiation
{
    namespace
    {
        void
        FillMomentGhosts (amrex::MultiFab& field, amrex::Geometry const& geometry)
        {
            field.FillBoundary(geometry.periodicity());
            if (!geometry.isAllPeriodic()) {
                amrex::Vector<amrex::BCRec> boundaries(field.nComp());
                for (auto& bc : boundaries) {
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        int const type = geometry.isPeriodic(d) ? amrex::BCType::int_dir
                                                                : amrex::BCType::foextrap;
                        bc.setLo(d, type);
                        bc.setHi(d, type);
                    }
                }
                amrex::FillDomainBoundary(field, geometry, boundaries);
            }
        }

        AMREX_GPU_HOST_DEVICE constexpr int
        PhysicalAxis (int direction)
        {
#if defined(WARPX_DIM_1D_Z)
            static_cast<void>(direction);
            return 2;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            return direction == 0 ? 0 : 2;
#else
            return direction;
#endif
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::IntVect
        Cell (int i, int j, int k)
        {
            amrex::ignore_unused(j, k);
            return amrex::IntVect(AMREX_D_DECL(i, j, k));
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE FourVector
        State (amrex::Array4<amrex::Real const> const& state, amrex::IntVect const& cell)
        {
            return {state(cell, 0), state(cell, 1), state(cell, 2), state(cell, 3)};
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE StressEnergy
        LinearizedTensor (FourVector const& state,
                          amrex::Array4<amrex::Real const> const& pressure_jacobian,
                          amrex::IntVect const& cell)
        {
            StressEnergy tensor{};
            tensor[0][0] = state[0];
            for (int d = 0; d < 3; ++d) {
                tensor[0][d + 1] = tensor[d + 1][0] = state[d + 1];
                for (int e = 0; e < 3; ++e) {
                    for (int column = 0; column < 4; ++column) {
                        tensor[d + 1][e + 1] +=
                            pressure_jacobian(cell, 9 * column + 3 * d + e) * state[column];
                    }
                }
            }
            return tensor;
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
        Projection (FourVector const& state, amrex::GpuArray<amrex::Real, 3> const& beta)
        {
            auto value = state[0];
            for (int d = 0; d < 3; ++d) {
                value -= beta[d] * state[d + 1];
            }
            return value;
        }

        class MomentOperator
        {
          public:
            using RT = amrex::Real;
            using FaceArrays = amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM>;

            MomentOperator (amrex::MultiFab const& state, amrex::MultiFab const& velocity,
                            amrex::MultiFab const& absorption, amrex::MultiFab const& scattering,
                            amrex::MultiFab const& equilibrium, amrex::Geometry const& geometry,
                            amrex::Real step,
                            PrescribedMomentFaceFluxes const* boundary = nullptr,
                            bool reflecting = false)
                : m_geom(geometry), m_dt(step),
                  m_beta(state.boxArray(), state.DistributionMap(), 3, 1),
                  m_coefficients(state.boxArray(), state.DistributionMap(), 3, 1),
                  m_pressure_jacobian(state.boxArray(), state.DistributionMap(), 36, 1),
                  m_diagonal(state.boxArray(), state.DistributionMap(), 16, 0),
                  m_boundary(boundary), m_reflecting(reflecting)
            {
                amrex::MultiFab::Copy(m_beta, velocity, 0, 0, 3, 0);
                amrex::MultiFab::Copy(m_coefficients, absorption, 0, 0, 1, 0);
                amrex::MultiFab::Copy(m_coefficients, scattering, 0, 1, 1, 0);
                amrex::MultiFab::Copy(m_coefficients, equilibrium, 0, 2, 1, 0);
                FillMomentGhosts(m_beta, m_geom);
                FillMomentGhosts(m_coefficients, m_geom);
                buildFaceParameters();
            }

            void
            buildFaceParameters ()
            {
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    auto face_boxes =
                        amrex::convert(m_beta.boxArray(), amrex::IntVect::TheDimensionVector(d));
                    m_parameters[d].define(face_boxes, m_beta.DistributionMap(), 4, 0);
                    m_flux[d].define(face_boxes, m_beta.DistributionMap(), 5, 0);
                    m_flux_precision[d].define(face_boxes, m_beta.DistributionMap(), 5, 0);
                    if (m_reflecting && !m_geom.isPeriodic(d)) {
                        m_mirror_jacobian[d].define(face_boxes, m_beta.DistributionMap(), 4, 0);
                    }
                    for (amrex::MFIter iterator(m_parameters[d]); iterator.isValid(); ++iterator) {
                        auto const b = m_beta.const_array(iterator);
                        auto const c = m_coefficients.const_array(iterator);
                        auto const p = m_parameters[d].array(iterator);
                        amrex::ParallelFor(
                            iterator.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                auto const right = Cell(i, j, k);
                                auto left = right;
                                --left[d];
                                for (int e = 0; e < 3; ++e) {
                                    p(right, e) = 0.5_rt * b(left, e) + 0.5_rt * b(right, e);
                                }
                                p(right, 3) = 0.5_rt * c(left, 0) + 0.5_rt * c(left, 1) +
                                              0.5_rt * c(right, 0) + 0.5_rt * c(right, 1);
                            });
                    }
                }
            }

            bool
            validCoefficients () const
            {
                amrex::ReduceOps<amrex::ReduceOpMax> ops;
                amrex::ReduceData<int> data(ops);
                using Tuple = typename decltype(data)::Type;
                auto const step = m_dt;
                for (amrex::MFIter iterator(m_beta); iterator.isValid(); ++iterator) {
                    auto const b = m_beta.const_array(iterator);
                    auto const c = m_coefficients.const_array(iterator);
                    ops.eval(iterator.validbox(), data,
                             [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                                 amrex::Real b2 = 0;
                                 for (int d = 0; d < 3; ++d) {
                                     b2 += b(i, j, k, d) * b(i, j, k, d);
                                 }
                                 if (!(b2 <= 1.e-4_rt)) {
                                     return {1};
                                 }
                                 for (int d = 0; d < 3; ++d) {
                                     if (!(c(i, j, k, d) >= 0) ||
                                         !amrex::Math::isfinite(c(i, j, k, d))) {
                                         return {1};
                                     }
                                 }
                                 return {!amrex::Math::isfinite(PhysConst::c * step *
                                                                (c(i, j, k, 0) + c(i, j, k, 1)))};
                             });
                }
                int invalid = amrex::get<0>(data.value());
                amrex::ParallelDescriptor::ReduceIntMax(invalid);
                return invalid == 0;
            }

            bool
            updateClosure (amrex::MultiFab const& state)
            {
                amrex::ReduceOps<amrex::ReduceOpMax> ops;
                amrex::ReduceData<int> data(ops);
                using Tuple = typename decltype(data)::Type;
                for (amrex::MFIter iterator(state); iterator.isValid(); ++iterator) {
                    auto const u = state.const_array(iterator);
                    auto const out = m_pressure_jacobian.array(iterator);
                    ops.eval(iterator.validbox(), data,
                             [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                                 FourVector const values{u(i, j, k, 0), u(i, j, k, 1), u(i, j, k, 2),
                                                   u(i, j, k, 3)};
                                 auto const closure = EvaluateM1Closure(values);
                                 if (!closure.valid) {
                                     return {1};
                                 }
                                 for (int column = 0; column < 4; ++column) {
                                     auto const derivative = M1ClosureDerivative(values, column);
                                     for (int d = 0; d < 3; ++d) {
                                         for (int e = 0; e < 3; ++e) {
                                             out(i, j, k, 9 * column + 3 * d + e) =
                                                 derivative[d + 1][e + 1];
                                         }
                                     }
                                 }
                                 return {0};
                             });
                }
                int invalid = amrex::get<0>(data.value());
                amrex::ParallelDescriptor::ReduceIntMax(invalid);
                if (invalid != 0) {
                    return false;
                }
                FillMomentGhosts(m_pressure_jacobian, m_geom);
                if (m_reflecting) {
                    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
                        if (m_geom.isPeriodic(direction)) {
                            continue;
                        }
                        auto const domain = m_geom.Domain();
                        int const normal = PhysicalAxis(direction);
                        amrex::ReduceOps<amrex::ReduceOpMax> wall_ops;
                        amrex::ReduceData<int> wall_data(wall_ops);
                        using WallTuple = typename decltype(wall_data)::Type;
                        for (amrex::MFIter iterator(m_mirror_jacobian[direction]);
                             iterator.isValid(); ++iterator) {
                            auto const u = state.const_array(iterator);
                            auto const jacobian = m_mirror_jacobian[direction].array(iterator);
                            wall_ops.eval(iterator.validbox(), wall_data,
                                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> WallTuple {
                                    auto const face = Cell(i, j, k);
                                    if (face[direction] != domain.smallEnd(direction) &&
                                        face[direction] != domain.bigEnd(direction) + 1) {
                                        return {0};
                                    }
                                    int const side =
                                        face[direction] == domain.smallEnd(direction) ? -1 : 1;
                                    auto interior = face;
                                    if (side > 0) {
                                        --interior[direction];
                                    }
                                    auto const derivative =
                                        EvaluateM1MirrorJacobian(State(u, interior), normal, side);
                                    if (!derivative.valid) {
                                        return {1};
                                    }
                                    for (int column = 0; column < 4; ++column) {
                                        jacobian(face, column) =
                                            side * derivative.derivative[column];
                                    }
                                    return {0};
                                });
                        }
                        int wall_invalid = amrex::get<0>(wall_data.value());
                        amrex::ParallelDescriptor::ReduceIntMax(wall_invalid);
                        if (wall_invalid) {
                            return false;
                        }
                    }
                }
                return true;
            }

            void
            buildFlux (amrex::MultiFab const& state, bool physical = false)
            {
                FillMomentGhosts(const_cast<amrex::MultiFab&>(state), m_geom);
                auto const spacing = m_geom.CellSizeArray();
                auto const domain = m_geom.Domain();
                for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
                    int const normal = PhysicalAxis(direction);
                    bool const prescribed = m_boundary && !m_geom.isPeriodic(direction);
                    bool const reflecting = m_reflecting && !m_geom.isPeriodic(direction);
                    for (amrex::MFIter iterator(m_flux[direction]); iterator.isValid();
                         ++iterator) {
                        auto const u = state.const_array(iterator);
                        auto const pressure = m_pressure_jacobian.const_array(iterator);
                        auto const p = m_parameters[direction].const_array(iterator);
                        auto const output = m_flux[direction].array(iterator);
                        auto const precision = m_flux_precision[direction].array(iterator);
                        auto const imposed = prescribed ? (*m_boundary)[direction]->const_array(iterator)
                                                        : amrex::Array4<amrex::Real const>{};
                        auto const mirror_jacobian = reflecting
                            ? m_mirror_jacobian[direction].const_array(iterator)
                            : amrex::Array4<amrex::Real const>{};
                        amrex::ParallelFor(iterator.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                                     int k) {
                            auto const right = Cell(i, j, k);
                            auto left = right;
                            --left[direction];
                            amrex::GpuArray<amrex::Real, 3> velocity{p(right, 0), p(right, 1),
                                                                     p(right, 2)};
                            if (reflecting && (right[direction] == domain.smallEnd(direction) ||
                                               right[direction] == domain.bigEnd(direction) + 1)) {
                                int const side =
                                    right[direction] == domain.smallEnd(direction) ? -1 : 1;
                                auto const interior = side < 0 ? right : left;
                                auto const values = State(u, interior);
                                auto wall_flux = amrex::Real(0);
                                auto sensitivity = amrex::Real(0);
                                if (physical) {
                                    auto const wall = EvaluateM1MirrorFlux(values, normal, side);
                                    wall_flux = wall.valid ? side * wall.flux[normal + 1]
                                        : std::numeric_limits<amrex::Real>::quiet_NaN();
                                    for (int column = 0; column < 4; ++column) {
                                        sensitivity += std::abs(mirror_jacobian(right, column) *
                                                                values[column]);
                                    }
                                } else {
                                    for (int column = 0; column < 4; ++column) {
                                        wall_flux +=
                                            mirror_jacobian(right, column) * values[column];
                                    }
                                }
                                for (int d = 0; d < 4; ++d) {
                                    output(right, d) = d == normal + 1 ? wall_flux : 0;
                                    if (physical) {
                                        precision(right, d) = d == normal + 1 ? sensitivity : 0;
                                    }
                                }
                                output(right, 4) = -velocity[normal] * wall_flux;
                                if (physical) {
                                    precision(right, 4) = std::abs(velocity[normal]) * sensitivity;
                                }
                                return;
                            }
                            if (prescribed && (right[direction] == domain.smallEnd(direction) ||
                                               right[direction] == domain.bigEnd(direction) + 1)) {
                                // Fixed physical forcing has zero derivative in
                                // the Krylov correction operator. Ghost closure
                                // extrapolation is not the boundary flux rule.
                                for (int d = 0; d < 4; ++d) {
                                    output(right, d) = physical ? imposed(right, d) : 0;
                                    if (physical) {
                                        precision(right, d) = std::abs(imposed(right, d));
                                    }
                                }
                                auto projected = output(right, 0);
                                auto bound = physical ? std::abs(imposed(right, 0)) : 0;
                                for (int d = 0; d < 3; ++d) {
                                    projected -= velocity[d] * output(right, d + 1);
                                    if (physical) {
                                        bound += std::abs(velocity[d] * imposed(right, d + 1));
                                    }
                                }
                                output(right, 4) = projected;
                                if (physical) {
                                    precision(right, 4) = bound;
                                }
                                return;
                            }
                            auto const l = State(u, left);
                            auto const r = State(u, right);
                            amrex::GpuArray<amrex::Real, 3> gradient{};
                            for (int t = 0; t < AMREX_SPACEDIM; ++t) {
                                if (t == direction) {
                                    continue;
                                }
                                auto lp = left, lm = left, rp = right, rm = right;
                                ++lp[t];
                                --lm[t];
                                ++rp[t];
                                --rm[t];
                                gradient[PhysicalAxis(t)] = (Projection(State(u, lp), velocity) -
                                                             Projection(State(u, lm), velocity) +
                                                             Projection(State(u, rp), velocity) -
                                                             Projection(State(u, rm), velocity)) /
                                                            (4 * spacing[t]);
                            }
                            auto const face = detail::MovingMomentFluxWithPressure(
                                l, r,
                                physical ? EvaluateM1Closure(l).tensor
                                         : LinearizedTensor(l, pressure, left),
                                physical ? EvaluateM1Closure(r).tensor
                                         : LinearizedTensor(r, pressure, right),
                                velocity, normal, p(right, 3), spacing[direction], gradient);
                            for (int d = 0; d < 4; ++d) {
                                output(right, d) =
                                    face.valid ? face.flux[d]
                                               : std::numeric_limits<amrex::Real>::quiet_NaN();
                            }
                            output(right, 4) = face.valid
                                                   ? face.projected_flux
                                                   : std::numeric_limits<amrex::Real>::quiet_NaN();
                            if (physical) {
                                // Bound input-rounding propagation through the
                                // face Jacobian, not merely the small net flux
                                // after cancellation of its large operands.
                                amrex::GpuArray<amrex::Real, 5> sensitivity{};
                                for (int column = 0; column < 4; ++column) {
                                    FourVector unit{};
                                    unit[column] = 1;
                                    auto const dl = detail::MovingMomentFluxWithPressure(
                                        unit, {}, LinearizedTensor(unit, pressure, left), {},
                                        velocity, normal, p(right, 3), spacing[direction]);
                                    auto const dr = detail::MovingMomentFluxWithPressure(
                                        {}, unit, {}, LinearizedTensor(unit, pressure, right),
                                        velocity, normal, p(right, 3), spacing[direction]);
                                    for (int row = 0; row < 4; ++row) {
                                        sensitivity[row] += std::abs(dl.flux[row] * l[column]) +
                                                            std::abs(dr.flux[row] * r[column]);
                                    }
                                    sensitivity[4] += std::abs(dl.projected_flux * l[column]) +
                                                      std::abs(dr.projected_flux * r[column]);
                                }
                                for (int t = 0; t < AMREX_SPACEDIM; ++t) {
                                    if (t == direction) {
                                        continue;
                                    }
                                    amrex::GpuArray<amrex::Real, 3> unit_gradient{};
                                    unit_gradient[PhysicalAxis(t)] = 1;
                                    auto const derivative = detail::MovingMomentFluxWithPressure(
                                        {}, {}, {}, {}, velocity, normal, p(right, 3),
                                        spacing[direction], unit_gradient);
                                    amrex::Real scale = 0;
                                    for (int side = 0; side < 2; ++side) {
                                        for (int sign = -1; sign <= 1; sign += 2) {
                                            auto neighbor = side == 0 ? left : right;
                                            neighbor[t] += sign;
                                            auto const values = State(u, neighbor);
                                            scale += std::abs(values[0]);
                                            for (int d = 0; d < 3; ++d) {
                                                scale += std::abs(velocity[d] * values[d + 1]);
                                            }
                                        }
                                    }
                                    sensitivity[0] +=
                                        std::abs(derivative.flux[0]) * scale / (4 * spacing[t]);
                                    sensitivity[4] += std::abs(derivative.projected_flux) * scale /
                                                      (4 * spacing[t]);
                                }
                                for (int row = 0; row < 5; ++row) {
                                    precision(right, row) = sensitivity[row];
                                }
                            }
                        });
                    }
                }
            }

            void
            updateBoundaryExchange ()
            {
                if (m_reflecting) {
                    PrescribedMomentFaceFluxes faces{};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        faces[d] = &m_flux[d];
                    }
                    if (!ComputeMomentBoundaryExchange(faces, m_geom, m_dt, m_boundary_exchange)) {
                        for (auto& value : m_boundary_exchange) {
                            value = std::numeric_limits<amrex::Real>::quiet_NaN();
                        }
                    }
                }
            }

            AMREX_GPU_HOST_DEVICE static FourVector
            Divergence (FaceArrays const& f, FaceArrays const& p, amrex::IntVect const& cell,
                        amrex::GpuArray<amrex::Real, 3> const& velocity,
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& spacing,
                        amrex::Real step)
            {
                FourVector change{};
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    auto upper = cell;
                    ++upper[d];
                    auto lo = f[d](cell, 4);
                    auto hi = f[d](upper, 4);
                    for (int e = 0; e < 3; ++e) {
                        lo += (p[d](cell, e) - velocity[e]) * f[d](cell, e + 1);
                        hi += (p[d](upper, e) - velocity[e]) * f[d](upper, e + 1);
                        change[e + 1] +=
                            step * (f[d](upper, e + 1) - f[d](cell, e + 1)) / spacing[d];
                    }
                    change[0] += step * (hi - lo) / spacing[d];
                }
                return change;
            }

            void
            apply (amrex::MultiFab& output, amrex::MultiFab const& state)
            {
                buildFlux(state);
                auto const step = m_dt;
                auto const spacing = m_geom.CellSizeArray();
                for (amrex::MFIter iterator(output); iterator.isValid(); ++iterator) {
                    auto const u = state.const_array(iterator);
                    auto const b = m_beta.const_array(iterator);
                    auto const c = m_coefficients.const_array(iterator);
                    auto const pressure = m_pressure_jacobian.const_array(iterator);
                    auto const out = output.array(iterator);
                    FaceArrays f{}, p{};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        f[d] = m_flux[d].const_array(iterator);
                        p[d] = m_parameters[d].const_array(iterator);
                    }
                    amrex::ParallelFor(iterator.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                                 int k) {
                        auto const cell = Cell(i, j, k);
                        auto const values = State(u, cell);
                        amrex::GpuArray<amrex::Real, 3> velocity{}, inverse{};
                        amrex::Real b2 = 0;
                        for (int d = 0; d < 3; ++d) {
                            velocity[d] = b(cell, d);
                            inverse[d] = -velocity[d];
                            b2 += velocity[d] * velocity[d];
                        }
                        auto const a = PhysConst::c * step * c(cell, 0);
                        auto const s = PhysConst::c * step * c(cell, 1);
                        auto const rest =
                            BoostStressEnergy(LinearizedTensor(values, pressure, cell), inverse);
                        auto const source = GreyFourForceFromComoving(rest, velocity, a, s, 0);
                        auto const transport = Divergence(f, p, cell, velocity, spacing, step);
                        out(cell, 0) = Projection(values, velocity) + transport[0] +
                                       a * std::sqrt(1 - b2) * rest[0][0];
                        for (int d = 1; d < 4; ++d) {
                            out(cell, d) = (values[d] + transport[d] + source[d]) / (1 + a + s);
                        }
                    });
                }
            }

            void
            makeRHS (amrex::MultiFab& rhs, amrex::MultiFab const& old)
            {
                auto const step = m_dt;
                for (amrex::MFIter iterator(rhs); iterator.isValid(); ++iterator) {
                    auto const u = old.const_array(iterator);
                    auto const b = m_beta.const_array(iterator);
                    auto const c = m_coefficients.const_array(iterator);
                    auto const out = rhs.array(iterator);
                    amrex::ParallelFor(
                        iterator.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                            auto const cell = Cell(i, j, k);
                            amrex::GpuArray<amrex::Real, 3> velocity{b(cell, 0), b(cell, 1),
                                                                     b(cell, 2)};
                            amrex::Real b2 = 0;
                            for (auto v : velocity) {
                                b2 += v * v;
                            }
                            auto const gamma = 1 / std::sqrt(1 - b2);
                            auto const a = PhysConst::c * step * c(cell, 0);
                            auto const s = PhysConst::c * step * c(cell, 1);
                            out(cell, 0) =
                                Projection(State(u, cell), velocity) + a * c(cell, 2) / gamma;
                            for (int d = 0; d < 3; ++d) {
                                out(cell, d + 1) =
                                    (u(cell, d + 1) + a * gamma * velocity[d] * c(cell, 2)) /
                                    (1 + a + s);
                            }
                        });
                }
            }

            void
            buildPreconditioner ()
            {
                auto const step = m_dt;
                auto const spacing = m_geom.CellSizeArray();
                for (amrex::MFIter iterator(m_diagonal); iterator.isValid(); ++iterator) {
                    auto const b = m_beta.const_array(iterator);
                    auto const c = m_coefficients.const_array(iterator);
                    auto const pressure = m_pressure_jacobian.const_array(iterator);
                    auto const out = m_diagonal.array(iterator);
                    FaceArrays p{};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        p[d] = m_parameters[d].const_array(iterator);
                    }
                    amrex::ParallelFor(iterator.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                                 int k) {
                        auto const cell = Cell(i, j, k);
                        amrex::GpuArray<amrex::Real, 3> velocity{}, inverse{};
                        amrex::Real b2 = 0;
                        for (int d = 0; d < 3; ++d) {
                            velocity[d] = b(cell, d);
                            inverse[d] = -velocity[d];
                            b2 += velocity[d] * velocity[d];
                        }
                        amrex::Real q_diagonal = 1, advection = 0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            for (int side = 0; side < 2; ++side) {
                                auto face = cell;
                                face[d] += side;
                                amrex::GpuArray<amrex::Real, 3> v{p[d](face, 0), p[d](face, 1),
                                                                  p[d](face, 2)};
                                auto const weight = detail::MovingMomentFluxWithPressure(
                                                        {}, {}, {}, {}, v, PhysicalAxis(d),
                                                        p[d](face, 3), spacing[d])
                                                        .asymptotic_weight;
                                q_diagonal += 0.5_rt * step * PhysConst::c * weight / spacing[d];
                                advection += 0.5_rt * step * PhysConst::c * (1 - weight) *
                                             std::abs(v[PhysicalAxis(d)]) / spacing[d];
                            }
                        }
                        auto const a = PhysConst::c * step * c(cell, 0);
                        auto const s = PhysConst::c * step * c(cell, 1);
                        for (int col = 0; col < 4; ++col) {
                            FourVector unit{};
                            unit[col] = 1;
                            auto const rest =
                                BoostStressEnergy(LinearizedTensor(unit, pressure, cell), inverse);
                            auto const source = GreyFourForceFromComoving(rest, velocity, a, s, 0);
                            out(cell, col) = (q_diagonal + advection) * Projection(unit, velocity) +
                                             a * std::sqrt(1 - b2) * rest[0][0];
                            for (int row = 1; row < 4; ++row) {
                                out(cell, 4 * row + col) =
                                    (q_diagonal * unit[row] + source[row]) / (1 + a + s);
                            }
                        }
                    });
                }
            }

            void
            precond (amrex::MultiFab& lhs, amrex::MultiFab const& rhs)
            {
                for (amrex::MFIter iterator(lhs); iterator.isValid(); ++iterator) {
                    auto const matrix = m_diagonal.const_array(iterator);
                    auto const b = rhs.const_array(iterator);
                    auto const out = lhs.array(iterator);
                    amrex::ParallelFor(
                        iterator.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                            StressEnergy block{};
                            FourVector forcing{}, solution{};
                            for (int row = 0; row < 4; ++row) {
                                forcing[row] = b(i, j, k, row);
                                for (int col = 0; col < 4; ++col) {
                                    block[row][col] = matrix(i, j, k, 4 * row + col);
                                }
                            }
                            bool const solved =
                                detail::SolveMomentLinearSystem(block, forcing, solution);
                            for (int d = 0; d < 4; ++d) {
                                out(i, j, k, d) = solved ? solution[d] : forcing[d];
                            }
                        });
                }
            }

            amrex::MultiFab
            makeVecRHS () const
            {
                return amrex::MultiFab(m_beta.boxArray(), m_beta.DistributionMap(), 4, 0);
            }
            amrex::MultiFab
            makeVecLHS () const
            {
                return amrex::MultiFab(m_beta.boxArray(), m_beta.DistributionMap(), 4, 1);
            }
            static void
            assign (amrex::MultiFab& a, amrex::MultiFab const& b)
            {
                amrex::MultiFab::Copy(a, b, 0, 0, 4, 0);
            }
            static RT
            dotProduct (amrex::MultiFab const& a, amrex::MultiFab const& b)
            {
                return amrex::MultiFab::Dot(a, 0, b, 0, 4, 0);
            }
            static RT
            norm2 (amrex::MultiFab const& a)
            {
                return std::sqrt(dotProduct(a, a));
            }
            static void
            increment (amrex::MultiFab& a, amrex::MultiFab const& b, RT f)
            {
                amrex::MultiFab::Saxpy(a, f, b, 0, 0, 4, 0);
            }
            static void
            linComb (amrex::MultiFab& a, RT f, amrex::MultiFab const& b, RT g,
                     amrex::MultiFab const& c)
            {
                amrex::MultiFab::LinComb(a, f, b, 0, g, c, 0, 0, 4, 0);
            }
            static void
            scale (amrex::MultiFab& a, RT f)
            {
                a.mult(f);
            }
            static void
            setToZero (amrex::MultiFab& a)
            {
                a.setVal(0);
            }

            void
            writeTransportIncrement (amrex::MultiFab& output) const
            {
                auto const spacing = m_geom.CellSizeArray();
                auto const step = m_dt;
                for (amrex::MFIter iterator(output); iterator.isValid(); ++iterator) {
                    FaceArrays f{}, precision{};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        f[d] = m_flux[d].const_array(iterator);
                        precision[d] = m_flux_precision[d].const_array(iterator);
                    }
                    auto const out = output.array(iterator);
                    amrex::ParallelFor(
                        iterator.validbox(), 4, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                            auto const cell = Cell(i, j, k);
                            amrex::Real value = 0, absolute = 0;
                            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                                auto upper = cell;
                                ++upper[d];
                                value += step * (f[d](upper, n) - f[d](cell, n)) / spacing[d];
                                absolute += step *
                                            (precision[d](upper, n) + precision[d](cell, n)) /
                                            spacing[d];
                            }
                            out(cell, n) = value;
                            out(cell, n + 4) = absolute;
                        });
                }
            }

            amrex::Geometry m_geom;
            amrex::Real m_dt;
            amrex::MultiFab m_beta, m_coefficients, m_pressure_jacobian, m_diagonal;
            std::array<amrex::MultiFab, AMREX_SPACEDIM> m_parameters, m_flux, m_flux_precision;
            PrescribedMomentFaceFluxes const* m_boundary;
            bool m_reflecting;
            std::array<amrex::MultiFab, AMREX_SPACEDIM> m_mirror_jacobian;
            FourVector m_boundary_exchange{};
        };

        void
        UpdateGlobalBalance (amrex::MultiFab const& state, amrex::MultiFab const& old,
                             amrex::MultiFab const& transfer, ImplicitMomentTransportResult& result,
                             FourVector const& boundary)
        {
            result.momentum_residual = 0;
            for (int d = 0; d < 4; ++d) {
                // Collect the same six inventories in one device pass and MPI
                // collective. Separate sum/norm calls dominated small-grid GPU
                // qualification time; no conservation scale or gate changes.
                amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
                                 amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum> ops;
                amrex::ReduceData<amrex::Real,amrex::Real,amrex::Real,
                                  amrex::Real,amrex::Real,amrex::Real> data(ops);
                using Tuple = typename decltype(data)::Type;
                for (amrex::MFIter iterator(state); iterator.isValid(); ++iterator) {
                    auto const current = state.const_array(iterator);
                    auto const initial = old.const_array(iterator);
                    auto const source = transfer.const_array(iterator);
                    ops.eval(iterator.validbox(),data,
                        [=] AMREX_GPU_DEVICE(int i,int j,int k) -> Tuple {
                            auto const a = current(i,j,k,d);
                            auto const b = initial(i,j,k,d);
                            auto const c = source(i,j,k,d);
                            return {a,b,c,std::abs(a),std::abs(b),std::abs(c)};
                        });
                }
                auto const values = data.value();
                amrex::Real totals[6] = {amrex::get<0>(values),amrex::get<1>(values),
                    amrex::get<2>(values),amrex::get<3>(values),amrex::get<4>(values),
                    amrex::get<5>(values)};
                amrex::ParallelDescriptor::ReduceRealSum(totals,6);
                auto const imbalance = totals[0]-totals[1]+totals[2]+boundary[d];
                auto const scale = totals[3]+totals[4]+totals[5]+std::abs(boundary[d]);
                auto const error =
                    std::isfinite(imbalance) && std::isfinite(scale)
                        ? (scale > 0 ? std::abs(imbalance) / scale : std::abs(imbalance))
                        : std::numeric_limits<amrex::Real>::infinity();
                if (d == 0) {
                    result.energy_residual = error;
                } else {
                    result.momentum_residual = amrex::max(result.momentum_residual, error);
                }
            }
        }

        bool
        ProjectConservedTotals (amrex::MultiFab const& state, amrex::MultiFab const& old,
                                amrex::MultiFab const& transfer, amrex::MultiFab& trial,
                                FourVector const& boundary)
        {
            FourVector correction{};
            // Sum small cell balances, not a difference of large global
            // inventories. The latter loses the weak energy/work exchange we
            // are attempting to preserve, and can bias repeated projections.
            for (amrex::MFIter iterator(state); iterator.isValid(); ++iterator) {
                auto const u = state.const_array(iterator);
                auto const initial = old.const_array(iterator);
                auto const source = transfer.const_array(iterator);
                auto const out = trial.array(iterator);
                amrex::ParallelFor(
                    iterator.validbox(), 4, [=] AMREX_GPU_DEVICE(int i, int j, int k, int d) {
                        auto const a = initial(i, j, k, d);
                        auto const b = -u(i, j, k, d);
                        auto const sum = a + b;
                        auto const recovered = sum - a;
                        auto const residual = (a - (sum - recovered)) + (b - recovered);
                        out(i, j, k, d) = (sum - source(i, j, k, d)) + residual;
                    });
            }
            bool nonzero = false;
            for (int d = 0; d < 4; ++d) {
                correction[d] = trial.sum(d) - boundary[d];
                if (!std::isfinite(correction[d])) {
                    return false;
                }
                nonzero = nonzero || correction[d] != 0;
            }
            auto const energy = state.sum(0);
            if (!(energy > 0) && nonzero) {
                return false;
            }
            for (amrex::MFIter iterator(state); iterator.isValid(); ++iterator) {
                auto const u = state.const_array(iterator);
                auto const out = trial.array(iterator);
                amrex::ParallelFor(iterator.validbox(), 4,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k, int d) {
                                       auto const weight = energy > 0 ? u(i, j, k, 0) / energy : 0;
                                       out(i, j, k, d) = u(i, j, k, d) + weight * correction[d];
                                   });
            }
            return true;
        }

        bool
        FinalizeSource (MomentOperator& op, amrex::MultiFab const& old,
                        amrex::MultiFab const& guess, amrex::MultiFab& corrected,
                        amrex::MultiFab& transfer, amrex::MultiFab& heat,
                        amrex::MultiFab& transport, amrex::Real tolerance)
        {
            op.writeTransportIncrement(transport);
            if (!transport.is_finite()) {
                return false;
            }
            amrex::ReduceOps<amrex::ReduceOpMax> ops;
            amrex::ReduceData<int> data(ops);
            using Tuple = typename decltype(data)::Type;
            auto const dt = op.m_dt;
            for (amrex::MFIter iterator(old); iterator.isValid(); ++iterator) {
                auto const initial = old.const_array(iterator);
                auto const g = guess.const_array(iterator);
                auto const flux = transport.const_array(iterator);
                auto const b = op.m_beta.const_array(iterator);
                auto const c = op.m_coefficients.const_array(iterator);
                auto const out = corrected.array(iterator);
                auto const source = transfer.array(iterator);
                auto const thermal = heat.array(iterator);
                ops.eval(
                    iterator.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                        auto const cell = Cell(i, j, k);
                        amrex::GpuArray<amrex::Real, 3> const beta{b(cell, 0), b(cell, 1), b(cell, 2)};
                        auto const absorption = PhysConst::c * dt * c(cell, 0);
                        auto const scattering = PhysConst::c * dt * c(cell, 1);
                        if (absorption + scattering <= 1) {
                            // In the non-stiff regime, direct force evaluation
                            // on the accepted stored state is well conditioned.
                            // Inferring a tiny source from old-minus-transport
                            // can instead amplify cancellation in that RHS.
                            // For stiff sources below, retain the independent
                            // implicit increment rather than amplifying an LTE
                            // state's rounding error by a large opacity.
                            auto const moments = State(g, cell);
                            auto const tensor = EvaluateM1Closure(moments).tensor;
                            auto const force = EvaluateGreyFourForce(tensor, beta, absorption,
                                                                     scattering, c(cell, 2));
                            if (!force.valid) {
                                return {1};
                            }
                            for (int d = 0; d < 4; ++d) {
                                out(cell, d) = moments[d];
                                source(cell, d) = force.material_force[d];
                            }
                            thermal(cell) =
                                GreyEnergyMinusWork(tensor, beta, absorption, c(cell, 2));
                            return {amrex::Math::isfinite(thermal(cell)) ? 0 : 1};
                        }
                        auto const update = TryImplicitGreyMomentSourceWithTransport(
                            State(initial, cell), State(flux, cell), State(g, cell), beta,
                            PhysConst::c * dt * c(cell, 0), PhysConst::c * dt * c(cell, 1),
                            c(cell, 2), 60, tolerance);
                        if (!update.valid) {
                            return {1};
                        }
                        for (int d = 0; d < 4; ++d) {
                            out(cell, d) = update.radiation[d];
                            source(cell, d) = update.material_transfer[d];
                        }
                        thermal(cell) = update.material_energy_minus_work;
                        return {0};
                    });
            }
            int invalid = amrex::get<0>(data.value());
            amrex::ParallelDescriptor::ReduceIntMax(invalid);
            return invalid == 0;
        }

        amrex::Real
        AssignedResidual (amrex::MultiFab const& state, amrex::MultiFab const& old,
                          amrex::MultiFab const& transfer, amrex::MultiFab const& transport,
                          amrex::Real tolerance)
        {
            amrex::ReduceOps<amrex::ReduceOpMax> ops;
            amrex::ReduceData<amrex::Real> data(ops);
            using Tuple = typename decltype(data)::Type;
            for (amrex::MFIter iterator(state); iterator.isValid(); ++iterator) {
                auto const u = state.const_array(iterator);
                auto const before = old.const_array(iterator);
                auto const source = transfer.const_array(iterator);
                auto const flux = transport.const_array(iterator);
                ops.eval(
                    iterator.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                        amrex::Real worst = 0;
                        for (int d = 0; d < 4; ++d) {
                            auto const change = u(i, j, k, d) - before(i, j, k, d);
                            auto const residual =
                                std::abs(change + flux(i, j, k, d) + source(i, j, k, d));
                            auto const scale = std::abs(change) + std::abs(flux(i, j, k, d)) +
                                               std::abs(source(i, j, k, d));
                            auto const noise =
                                64 * std::numeric_limits<amrex::Real>::epsilon() *
                                (u(i, j, k, 0) + before(i, j, k, 0) + flux(i, j, k, d + 4));
                            auto const bound = tolerance * scale + noise;
                            if (!amrex::Math::isfinite(residual) || !amrex::Math::isfinite(bound)) {
                                return {std::numeric_limits<amrex::Real>::max()};
                            }
                            worst = amrex::max(worst, bound > 0 ? residual / bound : residual);
                        }
                        return {worst};
                    });
            }
            auto result = amrex::get<0>(data.value());
            amrex::ParallelDescriptor::ReduceRealMax(result);
            return result;
        }

        amrex::Real
        EvaluateResidual (MomentOperator& op, amrex::MultiFab const& state,
                          amrex::MultiFab const& old, amrex::MultiFab& transfer,
                          ImplicitMomentTransportOptions const& options,
                          ImplicitMomentTransportResult& result,
                          amrex::MultiFab* correction_rhs = nullptr)
        {
            if (!op.updateClosure(state)) {
                return std::numeric_limits<amrex::Real>::infinity();
            }
            op.buildFlux(state, true);
            op.updateBoundaryExchange();
            amrex::ReduceOps<amrex::ReduceOpMax> ops;
            amrex::ReduceData<amrex::Real> data(ops);
            using Tuple = typename decltype(data)::Type;
            auto const step = op.m_dt;
            auto const spacing = op.m_geom.CellSizeArray();
            auto const tolerance = options.tolerance;
            for (amrex::MFIter iterator(state); iterator.isValid(); ++iterator) {
                auto const u = state.const_array(iterator);
                auto const initial = old.const_array(iterator);
                auto const b = op.m_beta.const_array(iterator);
                auto const c = op.m_coefficients.const_array(iterator);
                auto const output = transfer.array(iterator);
                amrex::Array4<amrex::Real> correction;
                if (correction_rhs) {
                    correction = correction_rhs->array(iterator);
                }
                bool const write_correction = correction_rhs != nullptr;
                MomentOperator::FaceArrays f{}, p{}, precision{};
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    f[d] = op.m_flux[d].const_array(iterator);
                    p[d] = op.m_parameters[d].const_array(iterator);
                    precision[d] = op.m_flux_precision[d].const_array(iterator);
                }
                ops.eval(
                    iterator.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                        auto const cell = Cell(i, j, k);
                        auto const values = State(u, cell);
                        auto const before = State(initial, cell);
                        amrex::GpuArray<amrex::Real, 3> velocity{};
                        for (int d = 0; d < 3; ++d) {
                            velocity[d] = b(cell, d);
                        }
                        auto const a = PhysConst::c * step * c(cell, 0);
                        auto const s = PhysConst::c * step * c(cell, 1);
                        auto const source = EvaluateGreyFourForce(EvaluateM1Closure(values).tensor,
                                                                  velocity, a, s, c(cell, 2))
                                                .material_force;
                        for (int d = 0; d < 4; ++d) {
                            output(cell, d) = source[d];
                        }
                        auto const transport =
                            MomentOperator::Divergence(f, p, cell, velocity, spacing, step);
                        FourVector difference{};
                        for (int d = 0; d < 4; ++d) {
                            difference[d] = values[d] - before[d];
                        }
                        difference[0] = Projection(difference, velocity);
                        FourVector forcing = source;
                        forcing[0] = GreyEnergyMinusWork(EvaluateM1Closure(values).tensor, velocity,
                                                         a, c(cell, 2));
                        if (write_correction) {
                            for (int d = 0; d < 4; ++d) {
                                correction(cell, d) = -(difference[d] + transport[d] + forcing[d]) /
                                                      (d == 0 ? 1 : 1 + a + s);
                            }
                        }
                        amrex::Real worst = 0;
                        for (int component = 0; component < 4; ++component) {
                            auto const residual = std::abs(
                                difference[component] + transport[component] + forcing[component]);
                            auto const scale = std::abs(difference[component]) +
                                               std::abs(transport[component]) +
                                               std::abs(forcing[component]);
                            auto noise_scale =
                                std::abs(before[0]) + std::abs(values[0]) +
                                (component == 0 ? a : a + s) * (std::abs(values[0]) + c(cell, 2));
                            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                                auto upper = cell;
                                ++upper[d];
                                int const fc = component == 0 ? 4 : component;
                                auto lo = precision[d](cell, fc);
                                auto hi = precision[d](upper, fc);
                                if (component == 0) {
                                    for (int e = 0; e < 3; ++e) {
                                        lo += std::abs(p[d](cell, e) - velocity[e]) *
                                              precision[d](cell, e + 1);
                                        hi += std::abs(p[d](upper, e) - velocity[e]) *
                                              precision[d](upper, e + 1);
                                    }
                                }
                                noise_scale += step * (lo + hi) / spacing[d];
                            }
                            auto const bound =
                                tolerance * scale +
                                64 * std::numeric_limits<amrex::Real>::epsilon() * noise_scale;
                            if (!amrex::Math::isfinite(residual) || !amrex::Math::isfinite(bound)) {
                                return {std::numeric_limits<amrex::Real>::max()};
                            }
                            worst = amrex::max(worst, bound > 0 ? residual / bound : residual);
                        }
                        return {worst};
                    });
            }
            result.equation_residual = amrex::get<0>(data.value());
            amrex::ParallelDescriptor::ReduceRealMax(result.equation_residual);
            UpdateGlobalBalance(state, old, transfer, result, op.m_boundary_exchange);
            return amrex::max(result.equation_residual,
                              amrex::max(result.energy_residual, result.momentum_residual) /
                                  tolerance);
        }
    } // namespace

    ImplicitMomentTransportResult
    TryImplicitMomentTransport (amrex::MultiFab& radiation, amrex::MultiFab const& beta,
                                amrex::MultiFab const& absorption,
                                amrex::MultiFab const& scattering,
                                amrex::MultiFab const& equilibrium,
                                amrex::MultiFab& material_transfer, amrex::Geometry const& geometry,
                                amrex::Real dt, ImplicitMomentTransportOptions const& options,
                                amrex::MultiFab* material_energy_minus_work,
                                PrescribedMomentFaceFluxes const* prescribed_boundary_fluxes)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::numeric_limits<amrex::Real>::digits >= 53,
            "Implicit moving moment transport requires double precision.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            radiation.nComp() == 4 && radiation.ixType().cellCentered() &&
                (geometry.isAllPeriodic() || prescribed_boundary_fluxes ||
                 options.reflecting_boundaries) &&
                !(prescribed_boundary_fluxes && options.reflecting_boundaries) &&
                beta.nComp() == 3 && absorption.nComp() == 1 &&
                scattering.nComp() == 1 && equilibrium.nComp() == 1 &&
                material_transfer.nComp() == 4 && std::isfinite(dt) && dt > 0 &&
                options.max_nonlinear_iterations > 0 && options.max_linear_iterations > 0 &&
                options.tolerance > 0 && options.tolerance < 1 && options.linear_tolerance > 0 &&
                options.linear_tolerance < options.tolerance,
            "Invalid implicit gray moment transport configuration.");
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        amrex::Abort(
            "Implicit moving moment transport initially supports Cartesian geometry only.");
#endif
        for (auto const* field : {&beta, &absorption, &scattering, &equilibrium,
                                  static_cast<amrex::MultiFab const*>(&material_transfer)}) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                field->boxArray() == radiation.boxArray() &&
                    field->DistributionMap() == radiation.DistributionMap(),
                "Implicit moment transport fields must have matching cell-centered layouts.");
        }
        if (material_energy_minus_work) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                material_energy_minus_work->nComp() == 1 &&
                    material_energy_minus_work->boxArray() == radiation.boxArray() &&
                    material_energy_minus_work->DistributionMap() == radiation.DistributionMap(),
                "Implicit moment caloric output must match the radiation layout.");
        }
        ImplicitMomentTransportResult result;
        FourVector boundary_exchange{};
        if (prescribed_boundary_fluxes) {
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                auto const* face = (*prescribed_boundary_fluxes)[d];
                if (!face || face->DistributionMap() != radiation.DistributionMap() ||
                    face->boxArray() != amrex::convert(radiation.boxArray(),
                                                      amrex::IntVect::TheDimensionVector(d))) {
                    return result;
                }
            }
            if (!ComputeMomentBoundaryExchange(*prescribed_boundary_fluxes, geometry, dt,
                                                boundary_exchange)) {
                return result;
            }
        }
        MomentOperator op(radiation, beta, absorption, scattering, equilibrium, geometry, dt,
                          prescribed_boundary_fluxes, options.reflecting_boundaries);
        op.m_boundary_exchange = boundary_exchange;
        if (!op.validCoefficients()) {
            return result;
        }
        auto current = op.makeVecLHS();
        auto candidate = op.makeVecLHS();
        auto trial = op.makeVecLHS();
        auto transfer = op.makeVecRHS();
        auto stable_transfer = op.makeVecRHS();
        amrex::MultiFab heat(radiation.boxArray(), radiation.DistributionMap(), 1, 0);
        amrex::MultiFab transport(radiation.boxArray(), radiation.DistributionMap(), 8, 0);
        auto rhs = op.makeVecRHS();
        auto linear_rhs = op.makeVecRHS();
        MomentOperator::assign(current, radiation);
        // A realizable local-source solution is an initial guess, not a split
        // update. It is already the full solution for spatially uniform data,
        // avoiding a poorly conditioned solve of that constant mode. Retain
        // the old physical state wherever the optional prediction fails; every
        // candidate still passes the complete transport/source equations below.
        for (amrex::MFIter iterator(current); iterator.isValid(); ++iterator) {
            auto const original = radiation.const_array(iterator);
            auto const velocity = op.m_beta.const_array(iterator);
            auto const coefficients = op.m_coefficients.const_array(iterator);
            auto const output = current.array(iterator);
            amrex::ParallelFor(iterator.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                auto const cell = Cell(i,j,k);
                amrex::GpuArray<amrex::Real,3> const drift{
                    velocity(cell,0),velocity(cell,1),velocity(cell,2)};
                auto const predicted = TryImplicitGreyMomentSource(State(original,cell),drift,
                    PhysConst::c*dt*coefficients(cell,0),PhysConst::c*dt*coefficients(cell,1),
                    coefficients(cell,2));
                if (predicted.valid) {
                    for (int d = 0; d < 4; ++d) { output(cell,d) = predicted.radiation[d]; }
                }
            });
        }
        op.makeRHS(rhs, radiation);
        if (!rhs.is_finite()) {
            return result;
        }
        amrex::GMRES<amrex::MultiFab, MomentOperator> solver;
        solver.define(op);
        solver.setRestartLength(40);
        solver.setVerbose(0);
        // Leave accuracy headroom for material feedback and source finalization.
        // This is a stricter iteration target, not a larger acceptance bound.
        auto const iteration_target = material_energy_minus_work ? 0.0625_rt : 1._rt;
        for (int iteration = 0; iteration < options.max_nonlinear_iterations; ++iteration) {
            result.nonlinear_iterations = iteration + 1;
            auto const merit =
                EvaluateResidual(op, current, radiation, transfer, options, result, &linear_rhs);
            if (!std::isfinite(merit)) {
                return result;
            }
            if (options.verbose) {
                amrex::Print() << "Implicit moment iteration=" << iteration
                               << " equation=" << result.equation_residual
                               << " energy=" << result.energy_residual
                               << " momentum=" << result.momentum_residual << '\n';
            }
            if (result.equation_residual <= iteration_target) {
                // Finalize independent source increments against the actual
                // transport flux, then recheck the corrected physical state.
                // Raw evaluation of a rounded LTE state must not generate a
                // spurious force, or erase a genuinely weak caloric source.
                if (!FinalizeSource(op, radiation, current, trial, stable_transfer, heat, transport,
                                    amrex::min(1.e-12_rt, 0.01_rt * options.tolerance))) {
                    return result;
                }
                // A source increment can be more precise than either rounded
                // radiation state. Retain the already equation-qualified state
                // if it also satisfies the separately checked assigned-source
                // balance. Forcing a different rounded state can cycle a stiff
                // solve around one ULP without improving its physical equation.
                auto assigned = AssignedResidual(current, radiation, stable_transfer, transport,
                                                 options.tolerance);
                ImplicitMomentTransportResult checked = result;
                UpdateGlobalBalance(current, radiation, stable_transfer, checked,
                                    op.m_boundary_exchange);
                bool const keep_current = assigned <= 1 && std::isfinite(checked.energy_residual) &&
                                          std::isfinite(checked.momentum_residual);
                MomentOperator::assign(candidate, trial);
                // Enforce the integral equations, including independently
                // counted prescribed boundary exchange, in the state rather
                // than allowing a small accepted zero-mode error to accumulate
                // over many steps. This projection is attempted only after the
                // original local equations pass, and the resulting state
                // is independently rechecked below. It is not a diagnostic
                // residual reservoir or an inferred boundary contribution.
                // Do not require the unprojected global balance to pass first:
                // correcting its small zero mode is the purpose of this step.
                // Every final local and global gate must still pass.
                if (keep_current) {
                    if (!ProjectConservedTotals(current, radiation, stable_transfer, trial,
                                                op.m_boundary_exchange)) {
                        MomentOperator::assign(trial, candidate);
                    }
                }
                auto checked_merit =
                    EvaluateResidual(op, trial, radiation, transfer, options, checked);
                op.writeTransportIncrement(transport);
                assigned = AssignedResidual(trial, radiation, stable_transfer, transport,
                                            options.tolerance);
                UpdateGlobalBalance(trial, radiation, stable_transfer, checked,
                                    op.m_boundary_exchange);
                if (keep_current &&
                    (!std::isfinite(checked_merit) || checked.equation_residual > 1 ||
                     assigned > 1 || checked.energy_residual > options.tolerance ||
                     checked.momentum_residual > options.tolerance)) {
                    MomentOperator::assign(trial, candidate);
                    checked_merit =
                        EvaluateResidual(op, trial, radiation, transfer, options, checked);
                    op.writeTransportIncrement(transport);
                    assigned = AssignedResidual(trial, radiation, stable_transfer, transport,
                                                options.tolerance);
                    UpdateGlobalBalance(trial, radiation, stable_transfer, checked,
                                        op.m_boundary_exchange);
                }
                if (std::isfinite(checked_merit) && checked.equation_residual <= 1 &&
                    assigned <= 1 && checked.energy_residual <= options.tolerance &&
                    checked.momentum_residual <= options.tolerance) {
                    amrex::MultiFab::Copy(radiation, trial, 0, 0, 4, 0);
                    amrex::MultiFab::Copy(material_transfer, stable_transfer, 0, 0, 4, 0);
                    if (material_energy_minus_work) {
                        amrex::MultiFab::Copy(*material_energy_minus_work, heat, 0, 0, 1, 0);
                        material_energy_minus_work->FillBoundary(geometry.periodicity());
                    }
                    radiation.FillBoundary(geometry.periodicity());
                    material_transfer.FillBoundary(geometry.periodicity());
                    result.equation_residual = amrex::max(checked.equation_residual, assigned);
                    result.energy_residual = checked.energy_residual;
                    result.momentum_residual = checked.momentum_residual;
                    result.boundary_exchange = op.m_boundary_exchange;
                    result.valid = true;
                    return result;
                }
                result.equation_residual = checked.equation_residual;
                result.energy_residual = checked.energy_residual;
                result.momentum_residual = checked.momentum_residual;
                MomentOperator::assign(current, trial);
                continue;
            }
            op.buildPreconditioner();
            // Solve for the correction. A relative tolerance on the full-state
            // RHS can stop while a small physical transport increment still
            // violates its equation gate underneath a large background.
            // Use the actual nonlinear residual, not rhs-J(current)*current.
            // M1 pressure is homogeneous, but reconstructing it via Euler's
            // identity loses rounding accuracy after a large-CFL divergence.
            if (!linear_rhs.is_finite()) {
                return result;
            }
            candidate.setVal(0);
            solver.solve(candidate, linear_rhs, options.linear_tolerance, 0,
                         options.max_linear_iterations);
            result.linear_iterations += solver.getNumIters();
            if (!candidate.is_finite()) {
                return result;
            }
            if (options.verbose && solver.getStatus() != 0) {
                amrex::Print() << "Inexact moment linear step: status=" << solver.getStatus()
                               << " residual=" << solver.getResidualNorm() << '\n';
            }
            // A finite inexact Krylov step still has to reduce the independently
            // evaluated nonlinear residual and pass every final physical gate.
            MomentOperator::increment(candidate, current, 1);
            bool accepted = false;
            amrex::Real fraction = 1;
            for (int line = 0; line < 25; ++line) {
                for (amrex::MFIter iterator(trial); iterator.isValid(); ++iterator) {
                    auto const old = current.const_array(iterator);
                    auto const next = candidate.const_array(iterator);
                    auto const out = trial.array(iterator);
                    amrex::ParallelFor(
                        iterator.validbox(), 4, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                            out(i, j, k, n) =
                                old(i, j, k, n) + fraction * (next(i, j, k, n) - old(i, j, k, n));
                        });
                }
                ImplicitMomentTransportResult evaluation;
                auto const next_merit =
                    EvaluateResidual(op, trial, radiation, transfer, options, evaluation);
                if (next_merit < merit || next_merit <= iteration_target) {
                    MomentOperator::assign(current, trial);
                    accepted = true;
                    break;
                }
                fraction *= 0.5_rt;
            }
            if (!accepted) {
                return result;
            }
        }
        return result;
    }

    bool
    ComputeMomentTransportIncrement (amrex::MultiFab const& radiation, amrex::MultiFab const& beta,
                                     amrex::MultiFab const& absorption,
                                     amrex::MultiFab const& scattering,
                                     amrex::MultiFab const& equilibrium, amrex::MultiFab& increment,
                                     amrex::Geometry const& geometry, amrex::Real dt)
    {
        if (std::numeric_limits<amrex::Real>::digits < 53 || radiation.nComp() != 4 ||
            !radiation.ixType().cellCentered() || !geometry.isAllPeriodic() || beta.nComp() != 3 ||
            absorption.nComp() != 1 || scattering.nComp() != 1 || equilibrium.nComp() != 1 ||
            increment.nComp() != 8 || !std::isfinite(dt) || !(dt > 0)) {
            return false;
        }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        return false;
#else
        for (auto const* field : {&beta, &absorption, &scattering, &equilibrium,
                                  static_cast<amrex::MultiFab const*>(&increment)}) {
            if (field->boxArray() != radiation.boxArray() ||
                field->DistributionMap() != radiation.DistributionMap()) {
                return false;
            }
        }
        MomentOperator op(radiation, beta, absorption, scattering, equilibrium, geometry, dt);
        if (!op.validCoefficients()) {
            return false;
        }
        auto state = op.makeVecLHS();
        MomentOperator::assign(state, radiation);
        if (!op.updateClosure(state)) {
            return false;
        }
        op.buildFlux(state, true);
        amrex::MultiFab trial(radiation.boxArray(), radiation.DistributionMap(), 8, 0);
        op.writeTransportIncrement(trial);
        if (!trial.is_finite()) {
            return false;
        }
        amrex::MultiFab::Copy(increment, trial, 0, 0, 8, 0);
        increment.FillBoundary(geometry.periodicity());
        return true;
#endif
    }
} // namespace warpx::radiation
