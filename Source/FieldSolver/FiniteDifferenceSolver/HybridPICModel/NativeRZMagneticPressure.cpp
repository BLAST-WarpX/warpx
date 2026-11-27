/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativeRZMagneticPressure.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "HybridPICModel.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace warpx::hybrid {
using namespace amrex::literals;
using warpx::fields::FieldType;

namespace {
int
globalInvalid (int invalid)
{
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    return invalid;
}

/** Reciprocal logarithmic mean. The series avoids cancellation near a=b.
 * A^-1=(log b-log a)/(b-a) = 1/rho_mid + O(dx^2). Thus
 * -A^-1 G P is a second-order approximation of -grad(P)/rho, while
 * P=c*rho gives exactly -c*G(log rho), whose Yee curl vanishes. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
inverseLogMean (amrex::Real const a, amrex::Real const b)
{
    amrex::Real const x = (b - a) / (a + b);
    if (amrex::Math::abs(x) < 1.e-3_rt) {
        amrex::Real const x2 = x * x;
        return 2._rt / (a + b) * (1._rt + x2 * (1._rt / 3._rt + x2 * (1._rt / 5._rt + x2 / 7._rt)));
    }
    return (std::log(b) - std::log(a)) / (b - a);
}

struct Snapshot
{
    amrex::MultiFab& live;
    amrex::MultiFab saved;
    explicit Snapshot (amrex::MultiFab& field)
        : live(field),
          saved(field.boxArray(), field.DistributionMap(), field.nComp(), field.nGrowVect())
    {
        amrex::MultiFab::Copy(saved, live, 0, 0, live.nComp(), live.nGrowVect());
    }
    void
    restore () const
    {
        amrex::MultiFab::Copy(live, saved, 0, 0, live.nComp(), live.nGrowVect());
    }
};
} // namespace

NativeRZMagneticPressure::NativeRZMagneticPressure (
    WarpX& sim, amrex::MultiFab& U, amrex::MultiFab& T, amrex::MultiFab const& charge,
    amrex::MultiFab const& volume, ElectronThermodynamicsExecutor eos,
    NativeRZMagneticPressureDiagnostics& diagnostics)
    : m_sim(sim), m_U(U), m_T(T), m_charge(charge.boxArray(), charge.DistributionMap(), 1, 1),
      m_volume(volume), m_eos(eos), m_diagnostics(diagnostics),
      m_pressure(U.boxArray(), U.DistributionMap(), 1, amrex::IntVect(1)),
      m_collocated(WarpX::grid_type == ablastr::utils::enums::GridType::Collocated)
{
    amrex::MultiFab::Copy(m_charge, charge, 0, 0, 1, 0);
    fillScalar(m_charge);
}

void
FillNativeMagneticPressureScalar (NativeRZMagneticPressure const& state, amrex::MultiFab& field)
{
    auto const& m_sim = state.m_sim;
    auto const& geom = m_sim.Geom(0);
    field.OverrideSync(geom.periodicity());
    field.FillBoundary(geom.periodicity());
    int const lo = geom.Domain().smallEnd(0);
    int const hi = geom.Domain().bigEnd(0) + 1;
    for (amrex::MFIter it(field); it.isValid(); ++it) {
        auto const arr = field.array(it);
        amrex::ParallelFor(it.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            if (i < lo) {
                arr(i, j, k) = arr(2 * lo - i, j, k);
            }
            if (i > hi) {
                arr(i, j, k) = arr(2 * hi - i, j, k);
            }
        });
    }
}

bool
ReconstructNativeMagneticPressure (NativeRZMagneticPressure& state, bool const check_temperature)
{
    auto& m_U = state.m_U;
    auto& m_T = state.m_T;
    auto& m_charge = state.m_charge;
    auto& m_pressure = state.m_pressure;
    auto const& m_eos = state.m_eos;
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> invalid(ops);
    auto const eos = m_eos;
    for (amrex::MFIter it(m_U); it.isValid(); ++it) {
        auto const U = m_U.const_array(it), rho = m_charge.const_array(it);
        auto const T = m_T.array(it), P = m_pressure.array(it);
        ops.eval(
            it.validbox(), invalid,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int> {
                amrex::Real const u = U(i, j, k), q = rho(i, j, k);
                if (!(q > 0 && u > 0) || !amrex::Math::isfinite(q + u)) {
                    return {1};
                }
                amrex::Real const t = eos.temperatureFromChargeDensityThermalEnergyDensity(q, u);
                auto const eos_state = eos.stateFromChargeDensityTemperature(q, t);
                amrex::Real const eps = std::numeric_limits<amrex::Real>::epsilon();
                bool const valid =
                    t > 0 && amrex::Math::isfinite(t) && t >= eos.minimumTemperature() &&
                    t <= eos.maximumTemperature() && eos_state.pressure > 0 &&
                    eos_state.heat_capacity_density > 0 &&
                    amrex::Math::isfinite(eos_state.pressure + eos_state.heat_capacity_density) &&
                    amrex::Math::abs(eos_state.internal_energy_density - u) <= 2048 * eps * u &&
                    (!check_temperature || (amrex::Math::isfinite(T(i, j, k)) &&
                                            amrex::Math::abs(T(i, j, k) - t) <= 256 * eps * t));
                if (valid) {
                    T(i, j, k) = t;
                    P(i, j, k) = eos_state.pressure;
                }
                return {!valid};
            });
    }
    if (globalInvalid(amrex::get<0>(invalid.value())) != 0) {
        return false;
    }
    state.fillScalar(m_U);
    state.fillScalar(m_T);
    state.fillScalar(m_pressure);
    return true;
}

void
NativeRZMagneticPressure::fillElectricField ()
{
    auto const E = m_sim.m_fields.get_alldirs(FieldType::Efield_fp, 0);
    auto const& geom = m_sim.Geom(0);
    amrex::Real const dr = geom.CellSize(0), dz = geom.CellSize(1);
    int const outer = geom.Domain().bigEnd(0) + 1;
    int const axis = geom.Domain().smallEnd(0);
    bool const collocated = m_collocated;
    for (auto* field : E) {
        field->setVal(0);
    }
    for (amrex::MFIter it(*E[0]); it.isValid(); ++it) {
        auto const P = m_pressure.const_array(it), rho = m_charge.const_array(it);
        auto const Er = E[0]->array(it), Ez = E[2]->array(it);
        amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            if (collocated) {
                Er(i, j, k) = i == axis    ? 0._rt
                              : i == outer ? -(P(i, j, k) - P(i - 1, j, k)) / dr *
                                                 inverseLogMean(rho(i - 1, j, k), rho(i, j, k))
                                           : -(P(i + 1, j, k) - P(i - 1, j, k)) / (2 * dr) *
                                                 inverseLogMean(rho(i - 1, j, k), rho(i + 1, j, k));
            } else {
                Er(i, j, k) = -(P(i + 1, j, k) - P(i, j, k)) / dr *
                              inverseLogMean(rho(i, j, k), rho(i + 1, j, k));
            }
        });
        amrex::ParallelFor(amrex::convert(it.validbox(), E[2]->ixType()), [=] AMREX_GPU_DEVICE(
                                                                              int i, int j, int k) {
            Ez(i, j, k) = i == outer   ? 0._rt
                          : collocated ? -(P(i, j + 1, k) - P(i, j - 1, k)) / (2 * dz) *
                                             inverseLogMean(rho(i, j - 1, k), rho(i, j + 1, k))
                                       : -(P(i, j + 1, k) - P(i, j, k)) / dz *
                                             inverseLogMean(rho(i, j, k), rho(i, j + 1, k));
        });
    }
    m_sim.ApplyEfieldBoundary(0, PatchType::fine, 0._rt);
    m_sim.ApplyFieldBoundaryOnAxis(E[0], E[1], E[2], 0);
}

void
EvaluateNativeMagneticPressureWork (
    NativeRZMagneticPressure const& state, amrex::MultiFab const& pressure,
    amrex::MultiFab const& magnetic, amrex::MultiFab& derivative)
{
    auto const& m_sim = state.m_sim;
    auto const& m_charge = state.m_charge;
    auto const& m_volume = state.m_volume;
    auto const m_collocated = state.m_collocated;
    auto const& geom = m_sim.Geom(0);
    int const axis = geom.Domain().smallEnd(0), outer = geom.Domain().bigEnd(0) + 1;
    amrex::Real const dr = geom.CellSize(0), dz = geom.CellSize(1);
    if (m_collocated) {
        // Material transpose spans two centered operators, but native B only
        // needs one ghost: form C^T first, then exchange its own one-cell halo.
        amrex::MultiFab current(pressure.boxArray(), pressure.DistributionMap(), 2, 1);
        for (amrex::MFIter it(current); it.isValid(); ++it) {
            auto const b = magnetic.const_array(it), q = m_charge.const_array(it);
            auto const c = current.array(it);
            amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                auto const wb = [=] AMREX_GPU_HOST_DEVICE(int r, int z) {
                    return r <= axis || r > outer
                               ? 0._rt
                               : 2 * MathConst::pi * (r - axis) * dr * dr * dz *
                                     (r == outer ? .5_rt : 1._rt) * b(r, z, k) / PhysConst::mu0;
                };
                amrex::Real const kr = (wb(i, j + 1) - wb(i, j - 1)) / (2 * dz);
                c(i, j, k, 0) =
                    i == axis ? 0._rt
                              : kr * (i == outer ? inverseLogMean(q(i - 1, j, k), q(i, j, k))
                                                 : inverseLogMean(q(i - 1, j, k), q(i + 1, j, k)));
                // Odd PEC Ez ghosts double the outer magnetic-row coefficient.
                amrex::Real const kz =
                    (wb(i - 1, j) - (i + 1 == outer ? 2._rt : 1._rt) * wb(i + 1, j)) / (2 * dr);
                c(i, j, k, 1) =
                    i == outer ? 0._rt : kz * inverseLogMean(q(i, j - 1, k), q(i, j + 1, k));
            });
        }
        current.OverrideSync(geom.periodicity());
        current.FillBoundary(geom.periodicity());
        for (amrex::MFIter it(derivative); it.isValid(); ++it) {
            auto const c = current.const_array(it), p = pressure.const_array(it);
            auto const v = m_volume.const_array(it);
            auto const output = derivative.array(it);
            amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                amrex::Real const left = i > axis + 1 ? c(i - 1, j, k, 0) : 0._rt;
                amrex::Real const right = i + 1 < outer ? c(i + 1, j, k, 0) : 0._rt;
                amrex::Real radial = (left - right) / (2 * dr);
                if (i == outer) {
                    radial += c(i, j, k, 0) / dr;
                }
                if (i + 1 == outer) {
                    radial -= c(i + 1, j, k, 0) / dr;
                }
                amrex::Real const axial = (c(i, j - 1, k, 1) - c(i, j + 1, k, 1)) / (2 * dz);
                output(i, j, k) = p(i, j, k) * (radial + axial) / v(i, j, k);
            });
        }
        return;
    }
    for (amrex::MFIter it(derivative); it.isValid(); ++it) {
        auto const P = pressure.const_array(it), q = m_charge.const_array(it);
        auto const B = magnetic.const_array(it), V = m_volume.const_array(it);
        auto const output = derivative.array(it);
        amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto const weighted_b = [=] AMREX_GPU_HOST_DEVICE(int r, int z) {
                return r < axis || r >= outer ? 0._rt
                                              : 2 * MathConst::pi * (r - axis + 0.5_rt) * dr * dr *
                                                    dz * B(r, z, k) / PhysConst::mu0;
            };
            // These are C^T W B/mu0, not the physical Ampere current.
            auto const edge_r = [=] AMREX_GPU_HOST_DEVICE(int r, int z) {
                return r < axis || r >= outer ? 0._rt
                                              : (weighted_b(r, z) - weighted_b(r, z - 1)) / dz *
                                                    inverseLogMean(q(r, z, k), q(r + 1, z, k));
            };
            auto const edge_z = [=] AMREX_GPU_HOST_DEVICE(int r, int z) {
                // M^T: the outer PEC edge was zeroed before Faraday.
                return r == outer ? 0._rt
                                  : (weighted_b(r - 1, z) - weighted_b(r, z)) / dr *
                                        inverseLogMean(q(r, z, k), q(r, z + 1, k));
            };
            amrex::Real const transpose =
                (edge_r(i - 1, j) - edge_r(i, j)) / dr + (edge_z(i, j - 1) - edge_z(i, j)) / dz;
            output(i, j, k) = P(i, j, k) * transpose / V(i, j, k);
        });
    }
}

bool
TrialNativeMagneticPressure (
    NativeRZMagneticPressure& state, HybridPICModel& hybrid, amrex::Real const dt,
    bool const adaptive, amrex::Real& error)
{
    auto& m_sim = state.m_sim;
    auto& m_U = state.m_U;
    auto& m_pressure = state.m_pressure;
    auto& m_diagnostics = state.m_diagnostics;
    auto const& m_volume = state.m_volume;
    auto const m_collocated = state.m_collocated;
    // The bounded branch uses the same RK4/Fehlberg tableaux as native BfieldEvolve.
    constexpr amrex::Real rk4_a[6][6] = {{}, {.5_rt}, {0, .5_rt}, {0, 0, 1._rt}, {}, {}};
    constexpr amrex::Real rkf_a[6][6] = {
        {},
        {1._rt / 4},
        {3._rt / 32, 9._rt / 32},
        {1932._rt / 2197, -7200._rt / 2197, 7296._rt / 2197},
        {439._rt / 216, -8, 3680._rt / 513, -845._rt / 4104},
        {-8._rt / 27, 2, -3544._rt / 2565, 1859._rt / 4104, -11._rt / 40}};
    constexpr amrex::Real rk4_b[6] = {1._rt / 6, 1._rt / 3, 1._rt / 3, 1._rt / 6, 0, 0};
    constexpr amrex::Real rkf_b[6] = {25._rt / 216,    0,          1408._rt / 2565,
                                      2197._rt / 4104, -1._rt / 5, 0};
    constexpr amrex::Real rkf_high[6] = {16._rt / 135,      0,           6656._rt / 12825,
                                         28561._rt / 56430, -9._rt / 50, 2._rt / 55};
    auto const& a = adaptive ? rkf_a : rk4_a;
    auto const& weights = adaptive ? rkf_b : rk4_b;
    int const stages = adaptive ? 6 : 4;
    auto Bfields = m_sim.m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, 0);
    auto Efields = m_sim.m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, 0);
    auto currents = m_sim.m_fields.get_mr_levels_alldirs(FieldType::current_fp, 0);
    auto density = m_sim.m_fields.get_mr_levels(FieldType::rho_fp, 0);
    auto& B = *Bfields[0][1];
    auto const& geom = m_sim.Geom(0);
    Snapshot b0(B), u0(m_U);
    amrex::MultiFab Kb(B.boxArray(), B.DistributionMap(), stages, 1);
    amrex::MultiFab Ku(m_U.boxArray(), m_U.DistributionMap(), stages, 1);
    amrex::MultiFab Kwork(B.boxArray(), B.DistributionMap(), stages, 1);
    amrex::MultiFab derivative(m_U.boxArray(), m_U.DistributionMap(), 1, 0);
    amrex::MultiFab stage_b(B.boxArray(), B.DistributionMap(), 1, B.nGrowVect());

    for (int stage = 0; stage < stages; ++stage) {
        amrex::MultiFab::Copy(B, b0.saved, 0, 0, 1, 0);
        amrex::MultiFab::Copy(m_U, u0.saved, 0, 0, 1, 0);
        for (int previous = 0; previous < stage; ++previous) {
            amrex::MultiFab::Saxpy(B, a[stage][previous], Kb, previous, 0, 1, 0);
            amrex::MultiFab::Saxpy(m_U, a[stage][previous], Ku, previous, 0, 1, 0);
        }
        if (!state.reconstruct()) {
            return false;
        }
        m_sim.ApplyBfieldBoundary(0, PatchType::fine, SubcyclingHalf::FirstHalf, 0._rt);
        m_sim.FillBoundaryB(B.nGrowVect(), true);
        state.work(m_pressure, B, derivative);
        amrex::MultiFab::Copy(Ku, derivative, 0, stage, 1, 0);
        Ku.mult(dt, stage, 1, 0);
        amrex::MultiFab::Copy(stage_b, B, 0, 0, 1, B.nGrowVect());
        // FieldPush calculates actual Ampere J, fills the pressure edges and
        // invokes native EvolveB. No non-pressure Ohm channel enters this call.
        hybrid.FieldPush(Bfields, Efields, currents, density, m_sim.GetEBUpdateEFlag_fp(), dt,
                         SubcyclingHalf::FirstHalf, B.nGrowVect(), true);
        auto const plasma = m_sim.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0);
        for (auto* current : plasma) {
            m_diagnostics.max_ampere_current =
                std::max(m_diagnostics.max_ampere_current, current->norm0(0, 0));
        }
        amrex::MultiFab::LinComb(Kb, 1._rt, B, 0, -1._rt, stage_b, 0, stage, 1, 0);
        for (amrex::MFIter it(Kwork); it.isValid(); ++it) {
            auto const bs = stage_b.const_array(it), increment = Kb.const_array(it);
            auto const w = Kwork.array(it);
            amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                w(i, j, k, stage) = bs(i, j, k) * increment(i, j, k, stage);
            });
        }
        if (!B.is_finite(0, 1, 0)) {
            return false;
        }
    }

    auto candidate = [&] (amrex::Real const* w, amrex::MultiFab& candidate_B,
                          amrex::MultiFab& candidate_U) {
        amrex::MultiFab work_sum(B.boxArray(), B.DistributionMap(), 1, 1);
        amrex::MultiFab defect(B.boxArray(), B.DistributionMap(), 1, 1);
        work_sum.setVal(0);
        amrex::MultiFab::Copy(candidate_B, b0.saved, 0, 0, 1, 0);
        amrex::MultiFab::Copy(candidate_U, u0.saved, 0, 0, 1, 0);
        for (int stage = 0; stage < stages; ++stage) {
            amrex::MultiFab::Saxpy(candidate_B, w[stage], Kb, stage, 0, 1, 0);
            amrex::MultiFab::Saxpy(candidate_U, w[stage], Ku, stage, 0, 1, 0);
            amrex::MultiFab::Saxpy(work_sum, w[stage], Kwork, stage, 0, 1, 0);
        }
        // Each cell's RK quadrature approximates d(B^2/2)/dt to RK order.
        // The difference below is O(dt^5) for RK4, unlike replacing the local
        // thermal endpoint by the product of separately averaged P and B.
        auto const dr = geom.CellSize(0), dz = geom.CellSize(1);
        int const axis = geom.Domain().smallEnd(0), outer = geom.Domain().bigEnd(0) + 1;
        bool const collocated = m_collocated;
        for (amrex::MFIter it(defect); it.isValid(); ++it) {
            auto const old = b0.saved.const_array(it), next = candidate_B.const_array(it);
            auto const sum = work_sum.const_array(it);
            auto const d = defect.array(it);
            amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                amrex::Real const volume =
                    2 * MathConst::pi * dr * dr * dz *
                    (collocated ? (i - axis) * (i == outer ? .5_rt : 1._rt) : i - axis + .5_rt);
                d(i, j, k) =
                    volume / PhysConst::mu0 *
                    (.5_rt * (next(i, j, k) - old(i, j, k)) * (next(i, j, k) + old(i, j, k)) -
                     sum(i, j, k));
            });
        }
        if (m_collocated) {
            defect.OverrideSync(geom.periodicity());
            for (amrex::MFIter it(candidate_U); it.isValid(); ++it) {
                auto const d = defect.const_array(it), v = m_volume.const_array(it);
                auto const u = candidate_U.array(it);
                amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    u(i, j, k) -= d(i, j, k) / v(i, j, k);
                });
            }
            return;
        }
        defect.FillBoundary(geom.periodicity());
        for (amrex::MFIter it(candidate_U); it.isValid(); ++it) {
            auto const d = defect.const_array(it), volume = m_volume.const_array(it);
            auto const output = candidate_U.array(it);
            amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                amrex::Real local = 0;
                if (i > axis) {
                    local += d(i - 1, j - 1, k) + d(i - 1, j, k);
                }
                if (i < outer) {
                    local += d(i, j - 1, k) + d(i, j, k);
                }
                output(i, j, k) -= .25_rt * local / volume(i, j, k);
            });
        }
    };

    amrex::MultiFab low_B(B.boxArray(), B.DistributionMap(), 1, B.nGrowVect());
    amrex::MultiFab low_U(m_U.boxArray(), m_U.DistributionMap(), 1, m_U.nGrowVect());
    candidate(weights, low_B, low_U);
    error = 0;
    if (adaptive) {
        amrex::MultiFab high_B(B.boxArray(), B.DistributionMap(), 1, B.nGrowVect());
        amrex::MultiFab high_U(m_U.boxArray(), m_U.DistributionMap(), 1, m_U.nGrowVect());
        candidate(rkf_high, high_B, high_U);
        if (!high_B.is_finite(0, 1, 0) || !high_U.is_finite(0, 1, 0)) {
            return false;
        }
        amrex::MultiFab::Subtract(high_B, low_B, 0, 0, 1, 0);
        amrex::MultiFab::Subtract(high_U, low_U, 0, 0, 1, 0);
        amrex::Real const bscale = low_B.norm0(0, 0);
        amrex::Real const uscale = low_U.norm0(0, 0);
        error =
            std::max(high_B.norm0(0, 0) / (hybrid.m_substep_atol + hybrid.m_substep_rtol * bscale),
                     high_U.norm0(0, 0) / (hybrid.m_substep_atol + hybrid.m_substep_rtol * uscale));
    }
    amrex::MultiFab::Copy(B, low_B, 0, 0, 1, 0);
    amrex::MultiFab::Copy(m_U, low_U, 0, 0, 1, 0);
    if (!state.reconstruct() || !std::isfinite(error)) {
        return false;
    }
    m_sim.ApplyBfieldBoundary(0, PatchType::fine, SubcyclingHalf::FirstHalf, 0._rt);
    m_sim.FillBoundaryB(B.nGrowVect(), true);
    amrex::ignore_unused(geom);
    return true;
}

void
NativeRZMagneticPressure::fillScalar (amrex::MultiFab& field) const
{
    FillNativeMagneticPressureScalar(*this, field);
}

bool
NativeRZMagneticPressure::reconstruct (bool const check_temperature)
{
    return ReconstructNativeMagneticPressure(*this, check_temperature);
}

void
NativeRZMagneticPressure::work (
    amrex::MultiFab const& pressure, amrex::MultiFab const& magnetic,
    amrex::MultiFab& derivative) const
{
    EvaluateNativeMagneticPressureWork(*this, pressure, magnetic, derivative);
}

bool
NativeRZMagneticPressure::trial (
    HybridPICModel& hybrid, amrex::Real const dt, bool const adaptive, amrex::Real& error)
{
    return TrialNativeMagneticPressure(*this, hybrid, dt, adaptive, error);
}

bool
NativeRZMagneticPressure::refresh (HybridPICModel& hybrid)
{
    auto const B = m_sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
    if (!B[1]->is_finite(0, 1, 0) || B[0]->norm0(0, 0) != 0 || B[2]->norm0(0, 0) != 0) {
        return false;
    }
    if (m_collocated) {
        amrex::ReduceOps<amrex::ReduceOpMax> ops;
        amrex::ReduceData<int> invalid(ops);
        int const axis = m_sim.Geom(0).Domain().smallEnd(0);
        for (amrex::MFIter it(*B[1]); it.isValid(); ++it) {
            auto const b = B[1]->const_array(it);
            ops.eval(it.validbox(), invalid,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int> {
                    return {i == axis && b(i, j, k) != 0};
                });
        }
        if (globalInvalid(amrex::get<0>(invalid.value())) != 0) { return false; }
    }
    if (!reconstruct(true)) {
        return false;
    }
    m_sim.ApplyBfieldBoundary(0, PatchType::fine, SubcyclingHalf::FirstHalf, 0._rt);
    m_sim.FillBoundaryB(B[1]->nGrowVect(), true);
    hybrid.CalculatePlasmaCurrent(m_sim.m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, 0),
                                  m_sim.GetEBUpdateEFlag_fp());
    fillElectricField();
    m_sim.FillBoundaryE(B[1]->nGrowVect(), true);
    return true;
}

bool
NativeRZMagneticPressure::evolve (HybridPICModel& hybrid, amrex::Real const half_dt, int const step)
{
    m_accepted = false;
    if (!reconstruct(true)) {
        return false;
    }
    auto B = m_sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
    std::array<std::unique_ptr<Snapshot>, 3> initial_B;
    for (int d = 0; d < 3; ++d) {
        initial_B[d] = std::make_unique<Snapshot>(*B[d]);
    }
    Snapshot initial_U(m_U), initial_T(m_T);
    auto reset_half = [&] {
        for (auto const& saved : initial_B) {
            saved->restore();
        }
        initial_U.restore();
        initial_T.restore();
    };
    bool adaptive = hybrid.DoRKF45(step);
    amrex::Real dt = half_dt / (hybrid.m_substeps / 2._rt), elapsed = 0;
    int attempts = 0;
    while (elapsed < half_dt && attempts < hybrid.m_max_substep_attempts) {
        dt = std::min(dt, half_dt - elapsed);
        if (!(dt > 0) || elapsed + dt == elapsed) {
            reset_half();
            return false;
        }
        Snapshot accepted_B(*B[1]), accepted_U(m_U), accepted_T(m_T);
        amrex::Real error = 0;
        bool const admissible = trial(hybrid, dt, adaptive, error);
        bool const success = admissible && (!adaptive || error <= 1);
        ++attempts;
        if (success) {
            elapsed += dt;
            ++m_diagnostics.accepted;
        } else {
            accepted_B.restore();
            accepted_U.restore();
            accepted_T.restore();
            ++m_diagnostics.rejected;
            if (!adaptive) {
                reset_half();
                elapsed = 0;
                m_diagnostics.accepted = 0;
                ++m_diagnostics.rk4_restarts;
                adaptive = true;
            }
        }
        amrex::Real const factor =
            !admissible ? .25_rt
            : adaptive  ? hybrid.m_substep_safety * std::pow(error + 1.e-10_rt, -.2_rt)
                        : 1._rt;
        dt *= success ? std::min(hybrid.m_substep_max_growth, factor)
                      : std::max(.1_rt, std::min(.5_rt, factor));
    }
    if (elapsed < half_dt) {
        reset_half();
        return false;
    }
    // Rebuild caches from the accepted bundle, never the last RK trial state.
    hybrid.CalculatePlasmaCurrent(m_sim.m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, 0),
                                  m_sim.GetEBUpdateEFlag_fp());
    fillElectricField();
    m_sim.FillBoundaryE(B[1]->nGrowVect(), true);
    hybrid.m_substeps = std::max(2, 2 * attempts);
    m_accepted = true;
    return true;
}
} // namespace warpx::hybrid

bool
HybridPICModel::tryBfieldEvolveNativePressure (
    amrex::MultiFab& U, amrex::MultiFab& T, amrex::MultiFab const& charge,
    amrex::MultiFab const& volume, amrex::Real const half_dt, int const step,
    warpx::hybrid::NativeRZMagneticPressureDiagnostics& diagnostics)
{
    using namespace amrex::literals;
    using warpx::fields::FieldType;
    using warpx::hybrid::Snapshot;
    auto& sim = WarpX::GetInstance();
    auto const& geom = sim.Geom(0);
    auto const B = sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
    auto const E = sim.m_fields.get_alldirs(FieldType::Efield_fp, 0);
    auto const J = sim.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0);
    auto const eos = electronThermodynamicsExecutor();
    diagnostics = {};
    bool layout = U.nComp() == 1 && U.nGrowVect().min() >= 1;
    for (auto const* scalar : std::array<amrex::MultiFab const*, 3>{&T, &charge, &volume}) {
        layout = layout && scalar->nComp() == 1 && scalar->ixType() == U.ixType() &&
                 scalar->boxArray() == U.boxArray() &&
                 scalar->DistributionMap() == U.DistributionMap() && scalar->nGrowVect().min() >= 1;
    }
    for (int d = 0; d < 3; ++d) {
        layout = layout && B[d]->nComp() == 1 && E[d]->nComp() == 1 &&
                 B[d]->nGrowVect().min() >= 1 && E[d]->nGrowVect().min() >= 1;
    }
    bool const collocated = WarpX::grid_type == ablastr::utils::enums::GridType::Collocated;
    layout = layout && U.ixType().nodeCentered() &&
             (collocated ? B[1]->ixType().nodeCentered() : B[1]->ixType().cellCentered()) &&
             amrex::convert(U.boxArray(), B[1]->ixType()) == B[1]->boxArray() &&
             U.DistributionMap() == B[1]->DistributionMap();
    bool const supported =
        m_native_rz_pressure == nullptr && layout && sim.finestLevel() == 0 &&
        (WarpX::grid_type == ablastr::utils::enums::GridType::Staggered || collocated) &&
        WarpX::n_rz_azimuthal_modes == 1 && geom.ProbLo(0) == 0 && !geom.isPeriodic(0) &&
        geom.isPeriodic(1) && WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC &&
        !EB::enabled() && !sim.DoPML() && eos.isIdealGas() && eos.m_num_latent_transitions == 0 &&
        !eos.m_use_resolved_material_selection && m_electron_thermodynamics.numMaterials() == 0 &&
        !m_add_external_fields && !m_has_external_current && !WarpX::use_filter &&
        !WarpX::do_single_precision_comms && std::numeric_limits<amrex::Real>::digits >= 53 &&
        std::isfinite(half_dt) && half_dt > 0 && m_substeps >= 2 && m_max_substep_attempts >= 1 &&
        std::isfinite(m_substep_rtol) && m_substep_rtol > 0 && std::isfinite(m_substep_atol) &&
        m_substep_atol >= 0 && std::isfinite(m_substep_safety) && m_substep_safety > 0 &&
        std::isfinite(m_substep_max_growth) && m_substep_max_growth >= 1;
    if (warpx::hybrid::globalInvalid(!supported) != 0) {
        return false;
    }
    bool const finite = charge.is_finite(0, 1, 0) && volume.is_finite(0, 1, 0) &&
                        B[0]->norm0(0, 0) == 0 && B[2]->norm0(0, 0) == 0 && charge.min(0) > 0 &&
                        volume.min(0) > 0;
    if (!finite) {
        return false;
    }
    if (collocated) {
        amrex::ReduceOps<amrex::ReduceOpMax> ops;
        amrex::ReduceData<int> bad_axis(ops);
        int const axis = geom.Domain().smallEnd(0);
        for (amrex::MFIter it(*B[1]); it.isValid(); ++it) {
            auto const b = B[1]->const_array(it);
            ops.eval(it.validbox(), bad_axis,
                     [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int> {
                         return {i == axis && b(i, j, k) != 0};
                     });
        }
        if (warpx::hybrid::globalInvalid(amrex::get<0>(bad_axis.value())) != 0) {
            return false;
        }
    }
    std::vector<std::unique_ptr<Snapshot>> saved;
    for (int d = 0; d < 3; ++d) {
        saved.push_back(std::make_unique<Snapshot>(*B[d]));
        saved.push_back(std::make_unique<Snapshot>(*E[d]));
        saved.push_back(std::make_unique<Snapshot>(*J[d]));
    }
    saved.push_back(std::make_unique<Snapshot>(U));
    saved.push_back(std::make_unique<Snapshot>(T));
    int const old_substeps = m_substeps;
    auto restore = [&] {
        for (auto const& snapshot : saved) {
            snapshot->restore();
        }
        m_substeps = old_substeps;
    };
    warpx::hybrid::NativeRZMagneticPressure bundle(sim, U, T, charge, volume, eos, diagnostics);
    m_native_rz_pressure = &bundle;
    try {
        BfieldEvolve(sim.m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, 0),
                     sim.m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, 0),
                     sim.m_fields.get_mr_levels_alldirs(FieldType::current_fp, 0),
                     sim.m_fields.get_mr_levels(FieldType::rho_fp, 0), sim.GetEBUpdateEFlag_fp(), step,
                     half_dt, SubcyclingHalf::FirstHalf, B[1]->nGrowVect(), true);
    } catch (...) {
        m_native_rz_pressure = nullptr;
        restore();
        throw;
    }
    m_native_rz_pressure = nullptr;
    if (!bundle.accepted()) {
        restore();
    }
    return bundle.accepted();
}
