/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Diagnostics/ReducedDiags/FieldEnergy.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/NativeRZMagneticPressure.H"
#include "Fields.H"
#include "Initialization/WarpXInit.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>
#include <AMReX_VisMF.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

using namespace amrex::literals;
using warpx::fields::FieldType;
using warpx::hybrid::NativeRZMagneticPressureDiagnostics;

namespace {
std::vector<amrex::Real>
gather (amrex::MultiFab const& field, amrex::Geometry const& geom)
{
    amrex::Box const box = amrex::convert(geom.Domain(), field.ixType());
    amrex::BoxArray boxes(box);
    amrex::DistributionMapping mapping(amrex::Vector<int>{0});
    amrex::MultiFab single(boxes, mapping, 1, 0);
    single.ParallelCopy(field, 0, 0, 1);
    std::vector<amrex::Real> values(static_cast<std::size_t>(box.numPts()));
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, single[0].dataPtr(),
                         single[0].dataPtr() + box.numPts(), values.begin());
    }
    amrex::ParallelDescriptor::Bcast(values.data(), values.size(), 0);
    return values;
}

amrex::Real
thermalEnergy (amrex::MultiFab const& U, amrex::MultiFab const& V, amrex::Geometry const& geom)
{
    auto const mask = U.OwnerMask(geom.periodicity());
    amrex::ReduceOps<amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Real> sum(ops);
    for (amrex::MFIter it(U); it.isValid(); ++it) {
        auto const u = U.const_array(it), v = V.const_array(it);
        auto const owner = mask->const_array(it);
        ops.eval(it.validbox(), sum,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<amrex::Real> {
                     return {owner(i, j, k) ? u(i, j, k) * v(i, j, k) : 0._rt};
                 });
    }
    auto result = amrex::get<0>(sum.value());
    amrex::ParallelDescriptor::ReduceRealSum(result);
    return result;
}

struct Backup
{
    amrex::MultiFab& field;
    amrex::MultiFab saved;
    explicit Backup (amrex::MultiFab& f)
        : field(f), saved(f.boxArray(), f.DistributionMap(), f.nComp(), f.nGrowVect())
    {
        amrex::MultiFab::Copy(saved, f, 0, 0, f.nComp(), f.nGrowVect());
    }
    void
    restore () const
    {
        amrex::MultiFab::Copy(field, saved, 0, 0, field.nComp(), field.nGrowVect());
    }
    bool
    identical () const
    {
        amrex::MultiFab delta(field.boxArray(), field.DistributionMap(), field.nComp(),
                              field.nGrowVect());
        amrex::MultiFab::LinComb(delta, 1._rt, field, 0, -1._rt, saved, 0, 0, field.nComp(),
                                 field.nGrowVect());
        return delta.norm0(0, field.nComp(), field.nGrowVect()) == 0;
    }
};

/** Independent host ODE: assemble C^T and G^T by scatter over complete cells
 * and edges. It shares no stencil/work/RK helper with the native implementation.
 * Periodic endpoint nodes are stored once here, unlike the native MultiFabs. */
struct Oracle
{
    int nr, nz;
    double dr, dz, gamma;
    bool collocated;
    std::vector<double> rho, volume;
    int
    node (int i, int j) const
    {
        return i + (nr + 1) * ((j + nz) % nz);
    }
    int
    cell (int i, int j) const
    {
        return i + nr * ((j + nz) % nz);
    }
    static double
    invmean (double a, double b)
    {
        // Symmetric collocated endpoints can differ by one ulp. log1p keeps
        // their logarithmic mean accurate without using the native series.
        return a == b ? 1 / a : std::log1p((b - a) / a) / (b - a);
    }

    std::vector<double>
    collocatedRhs (std::vector<double> const& state) const
    {
        int const nn = (nr + 1) * nz;
        std::vector<double> output(state.size(), 0), er(nn, 0), ez(nn, 0);
        std::vector<double> kr(nn, 0), kz(nn, 0), div(nn, 0);
        for (int j = 0; j < nz; ++j) {
            for (int i = 1; i <= nr; ++i) {
                int const lower = i - 1;
                int const upper = i == nr ? i : i + 1;
                double const length = i == nr ? dr : 2 * dr;
                er[node(i, j)] = -(gamma - 1) *
                                 (state[nn + node(upper, j)] - state[nn + node(lower, j)]) *
                                 invmean(rho[node(lower, j)], rho[node(upper, j)]) / length;
            }
            for (int i = 0; i < nr; ++i) {
                ez[node(i, j)] = -(gamma - 1) *
                                 (state[nn + node(i, j + 1)] - state[nn + node(i, j - 1)]) *
                                 invmean(rho[node(i, j - 1)], rho[node(i, j + 1)]) / (2 * dz);
            }
        }
        for (int j = 0; j < nz; ++j) {
            for (int i = 1; i <= nr; ++i) {
                int const n = node(i, j);
                output[n] = (i == nr ? -ez[node(i - 1, j)] / dr
                                     : (ez[node(i + 1, j)] - ez[node(i - 1, j)]) / (2 * dr)) -
                            (er[node(i, j + 1)] - er[node(i, j - 1)]) / (2 * dz);
                double const wb = 2 * MathConst::pi * i * dr * dr * dz * (i == nr ? .5 : 1) *
                                  state[n] / PhysConst::mu0;
                kr[node(i, j - 1)] += wb / (2 * dz);
                kr[node(i, j + 1)] -= wb / (2 * dz);
                if (i == nr) {
                    kz[node(i - 1, j)] -= wb / dr;
                } else {
                    kz[node(i + 1, j)] += wb / (2 * dr);
                    kz[node(i - 1, j)] -= wb / (2 * dr);
                }
            }
        }
        for (int j = 0; j < nz; ++j) {
            for (int i = 1; i <= nr; ++i) {
                int const lower = i - 1, upper = i == nr ? i : i + 1;
                double const length = i == nr ? dr : 2 * dr;
                double const f =
                    kr[node(i, j)] * invmean(rho[node(lower, j)], rho[node(upper, j)]) / length;
                div[node(lower, j)] -= f;
                div[node(upper, j)] += f;
            }
            for (int i = 0; i < nr; ++i) {
                double const f =
                    kz[node(i, j)] * invmean(rho[node(i, j - 1)], rho[node(i, j + 1)]) / (2 * dz);
                div[node(i, j - 1)] -= f;
                div[node(i, j + 1)] += f;
            }
        }
        for (int n = 0; n < nn; ++n) {
            output[nn + n] = (gamma - 1) * state[nn + n] * div[n] / volume[n];
        }
        return output;
    }

    std::vector<double>
    rhs (std::vector<double> const& state) const
    {
        if (collocated) {
            return collocatedRhs(state);
        }
        int const nb = nr * nz, nn = (nr + 1) * nz;
        std::vector<double> output(state.size(), 0), er(nb), ez(nn), kr(nb, 0), kz(nn, 0),
            div(nn, 0);
        for (int j = 0; j < nz; ++j) {
            for (int i = 0; i < nr; ++i) {
                er[cell(i, j)] = -(gamma - 1) *
                                 (state[nb + node(i + 1, j)] - state[nb + node(i, j)]) *
                                 invmean(rho[node(i, j)], rho[node(i + 1, j)]) / dr;
            }
            for (int i = 0; i <= nr; ++i) {
                ez[node(i, j)] =
                    i == nr ? 0
                            : -(gamma - 1) * (state[nb + node(i, j + 1)] - state[nb + node(i, j)]) *
                                  invmean(rho[node(i, j)], rho[node(i, j + 1)]) / dz;
            }
        }
        for (int j = 0; j < nz; ++j) {
            for (int i = 0; i < nr; ++i) {
                int const c = cell(i, j);
                output[c] =
                    (ez[node(i + 1, j)] - ez[node(i, j)]) / dr - (er[cell(i, j + 1)] - er[c]) / dz;
                double const wb =
                    2 * MathConst::pi * (i + .5) * dr * dr * dz * state[c] / PhysConst::mu0;
                kr[c] += wb / dz;
                kr[cell(i, j + 1)] -= wb / dz;
                kz[node(i + 1, j)] += wb / dr;
                kz[node(i, j)] -= wb / dr;
            }
        }
        for (int j = 0; j < nz; ++j) {
            for (int i = 0; i < nr; ++i) {
                double const f =
                    kr[cell(i, j)] * invmean(rho[node(i, j)], rho[node(i + 1, j)]) / dr;
                div[node(i, j)] -= f;
                div[node(i + 1, j)] += f;
            }
            for (int i = 0; i < nr; ++i) {
                double const f =
                    kz[node(i, j)] * invmean(rho[node(i, j)], rho[node(i, j + 1)]) / dz;
                div[node(i, j)] -= f;
                div[node(i, j + 1)] += f;
            }
        }
        for (int n = 0; n < nn; ++n) {
            output[nb + n] = (gamma - 1) * state[nb + n] * div[n] / volume[n];
        }
        return output;
    }
    std::vector<double>
    integrate (std::vector<double> state, double dt, int steps) const
    {
        for (int s = 0; s < steps; ++s) {
            auto const k1 = rhs(state);
            auto temp = state;
            for (std::size_t n = 0; n < state.size(); ++n) {
                temp[n] = state[n] + .5 * dt * k1[n];
            }
            auto const k2 = rhs(temp);
            for (std::size_t n = 0; n < state.size(); ++n) {
                temp[n] = state[n] + .5 * dt * k2[n];
            }
            auto const k3 = rhs(temp);
            for (std::size_t n = 0; n < state.size(); ++n) {
                temp[n] = state[n] + dt * k3[n];
            }
            auto const k4 = rhs(temp);
            for (std::size_t n = 0; n < state.size(); ++n) {
                state[n] += dt * (k1[n] + 2 * k2[n] + 2 * k3[n] + k4[n]) / 6;
            }
        }
        return state;
    }
};
} // namespace

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        auto* hybrid = sim.get_pointer_HybridPICModel();
        auto const& geom = sim.Geom(0);
        auto B = sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
        auto E = sim.m_fields.get_alldirs(FieldType::Efield_fp, 0);
        auto J = sim.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0);
        amrex::BoxArray ba = amrex::convert(B[1]->boxArray(), amrex::IntVect(1));
        auto const& dm = B[1]->DistributionMap();
        amrex::MultiFab U(ba, dm, 1, 1), T(ba, dm, 1, 1), rho(ba, dm, 1, 1), V(ba, dm, 1, 1);
        auto const eos = hybrid->electronThermodynamicsExecutor();
        amrex::Real const dr = geom.CellSize(0), dz = geom.CellSize(1);
        int const nr = geom.Domain().length(0), nz = geom.Domain().length(1);
        bool const collocated = B[1]->ixType().nodeCentered();
        int const nb = (collocated ? nr + 1 : nr) * nz;
        FieldEnergy field_energy("native_pressure_energy");
        auto energy = [&] {
            return thermalEnergy(U, V, geom) +
                   .5_rt * field_energy.ComputeNorm2(*B[1], 0) * dr * dz / PhysConst::mu0;
        };
        auto seed = [&] (int control, bool magnetic) {
            for (auto* field : B) {
                field->setVal(0);
            }
            for (auto* field : E) {
                field->setVal(17);
            }
            for (auto* field : J) {
                field->setVal(23);
            }
            for (amrex::MFIter it(U); it.isValid(); ++it) {
                auto const u = U.array(it), t = T.array(it), q = rho.array(it), v = V.array(it);
                amrex::ParallelFor(it.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    amrex::Real const r = i * dr, z = j * dz,
                                      shape = std::pow(std::cos(.5_rt * MathConst::pi * r), 2);
                    amrex::Real const density =
                        1.e18_rt *
                        (1 + (control == 2 ? 0 : .25_rt * shape * std::cos(2 * MathConst::pi * z)));
                    amrex::Real const temperature =
                        1.e5_rt *
                        (1 + (control == 1 ? 0 : .2_rt * shape * std::sin(2 * MathConst::pi * z)));
                    auto const state = eos.stateFromChargeDensityTemperature(
                        density * PhysConst::q_e, temperature);
                    q(i, j, k) = density * PhysConst::q_e;
                    t(i, j, k) = temperature;
                    u(i, j, k) = state.internal_energy_density;
                    // Declared native caloric support: axis and reflecting wall
                    // use current FV inverse-deposition volumes, not W_B.
                    v(i, j, k) =
                        i == 0 ? MathConst::pi * dr * dr * dz / 3
                               : 2 * MathConst::pi * i * dr * dr * dz * (i == nr ? .5_rt : 1._rt);
                });
            }
            if (magnetic) {
                for (amrex::MFIter it(*B[1]); it.isValid(); ++it) {
                    auto const b = B[1]->array(it);
                    amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        amrex::Real const r = (i + (collocated ? 0._rt : .5_rt)) * dr,
                                          z = (j + (collocated ? 0._rt : .5_rt)) * dz;
                        b(i, j, k) = .002_rt * r * (1 - .4_rt * r * r) *
                                     (1 + .1_rt * std::cos(2 * MathConst::pi * z));
                    });
                }
            }
            sim.ApplyBfieldBoundary(0, PatchType::fine, SubcyclingHalf::FirstHalf, 0._rt);
            sim.FillBoundaryB(B[1]->nGrowVect(), true);
            hybrid->m_substeps = 2;
        };

        bool spatial_only = false;
        amrex::ParmParse("test").query("spatial_only", spatial_only);
        if (spatial_only) {
            seed(0, false);
            NativeRZMagneticPressureDiagnostics diag;
            constexpr amrex::Real dt = 1.e-12_rt;
            AMREX_ALWAYS_ASSERT(hybrid->tryBfieldEvolveNativePressure(U, T, rho, V, dt, 0, diag));
            auto const numerical = gather(*B[1], geom);
            double norm = 0, scale = 0;
            for (int j = 0; j < nz; ++j) {
                for (int i = 0; i < nr + (collocated ? 1 : 0); ++i) {
                    double const r = (i + (collocated ? 0 : .5)) * dr,
                                 z = (j + (collocated ? 0 : .5)) * dz;
                    double const shape = std::pow(std::cos(.5 * MathConst::pi * r), 2);
                    double const derivative = -.5 * MathConst::pi * std::sin(MathConst::pi * r);
                    // Analytic curl(-grad(P)/rho), with P=rho*k_B*T/q_e.
                    double const exact = PhysConst::kb / PhysConst::q_e * 1.e5 * .25 * .2 * 2 *
                                         MathConst::pi * shape * derivative /
                                         (1 + .25 * shape * std::cos(2 * MathConst::pi * z));
                    double const weight = r * (collocated && i == nr ? .5 : 1);
                    norm += weight *
                            std::pow(numerical[i + (collocated ? nr + 1 : nr) * j] / dt - exact, 2);
                    scale += weight * exact * exact;
                }
            }
            amrex::Print() << "spatial_relative_error=" << std::sqrt(norm / scale) << '\n';
        } else {

            for (int control = 1; control <= 2; ++control) {
                seed(control, false);
                amrex::Real const initial = energy();
                NativeRZMagneticPressureDiagnostics diag;
                AMREX_ALWAYS_ASSERT(
                    hybrid->tryBfieldEvolveNativePressure(U, T, rho, V, 1.e-6_rt, 0, diag));
                AMREX_ALWAYS_ASSERT(B[1]->norm0(0, 0) < 2.e-13_rt);
                AMREX_ALWAYS_ASSERT(std::abs(energy() - initial) < 2.e-13_rt * initial);
                amrex::Print() << "null control=" << control << " B=" << B[1]->norm0(0, 0) << '\n';
            }

            seed(0, true);
            Backup initial_B(*B[1]), initial_U(U), initial_T(T);
            auto const rhos = gather(rho, geom), volumes = gather(V, geom);
            auto const bvalues = gather(*B[1], geom), uvalues = gather(U, geom);
            Oracle oracle{nr, nz, dr, dz, eos.m_gamma, collocated, {}, {}};
            oracle.rho.assign(rhos.begin(), rhos.begin() + (nr + 1) * nz);
            oracle.volume.assign(volumes.begin(), volumes.begin() + (nr + 1) * nz);
            std::vector<double> initial(bvalues.begin(), bvalues.begin() + nb);
            initial.insert(initial.end(), uvalues.begin(), uvalues.begin() + (nr + 1) * nz);
            constexpr double total_time = 4.e-6;
            auto const reference = oracle.integrate(initial, total_time / 2048, 2048);
            std::array<double, 3> errors{};
            amrex::Real const initial_energy = energy();
            for (int refinement = 0; refinement < 3; ++refinement) {
                initial_B.restore();
                initial_U.restore();
                initial_T.restore();
                int const steps = 2 << refinement;
                for (int s = 0; s < steps; ++s) {
                    hybrid->m_substeps = 2;
                    NativeRZMagneticPressureDiagnostics diag;
                    AMREX_ALWAYS_ASSERT(hybrid->tryBfieldEvolveNativePressure(
                        U, T, rho, V, total_time / steps, 0, diag));
                    AMREX_ALWAYS_ASSERT(diag.max_ampere_current > 100);
                    AMREX_ALWAYS_ASSERT(std::abs(energy() - initial_energy) <
                                        4.e-13_rt * initial_energy);
                }
                auto const native_b = gather(*B[1], geom), native_u = gather(U, geom);
                double error = 0;
                for (int n = 0; n < nb; ++n) {
                    error = std::max(error, std::abs(native_b[n] - reference[n]) / .002);
                }
                for (int n = 0; n < (nr + 1) * nz; ++n) {
                    error = std::max(error, std::abs(native_u[n] - reference[n + nb]) / 2);
                }
                errors[refinement] = error;
                amrex::Print() << "time convergence steps=" << steps << " oracle_error=" << error
                               << " energy_relative_error="
                               << std::abs(energy() - initial_energy) / initial_energy << '\n';
            }
            AMREX_ALWAYS_ASSERT(errors[0] > 8 * errors[1] && errors[1] > 8 * errors[2]);

            // Native RKF45 rejects a large finite trial and then retries the same
            // B/U/T bundle. A one-attempt budget must reject the whole half-step.
            seed(0, true);
            std::vector<std::unique_ptr<Backup>> backups;
            for (int d = 0; d < 3; ++d) {
                backups.push_back(std::make_unique<Backup>(*B[d]));
                backups.push_back(std::make_unique<Backup>(*E[d]));
                backups.push_back(std::make_unique<Backup>(*J[d]));
            }
            backups.push_back(std::make_unique<Backup>(U));
            backups.push_back(std::make_unique<Backup>(T));
            hybrid->m_max_substep_attempts = 1;
            hybrid->m_rkf45_intervals = ablastr::utils::text::IntervalsParser({"0"});
            NativeRZMagneticPressureDiagnostics restarted;
            AMREX_ALWAYS_ASSERT(
                !hybrid->tryBfieldEvolveNativePressure(U, T, rho, V, 1.e-2_rt, 0, restarted));
            AMREX_ALWAYS_ASSERT(restarted.rk4_restarts == 1 && hybrid->m_substeps == 2);
            for (auto const& backup : backups) {
                AMREX_ALWAYS_ASSERT(backup->identical());
            }
            hybrid->m_rkf45_intervals = ablastr::utils::text::IntervalsParser({"1"});
            hybrid->m_max_substep_attempts = 1;
            hybrid->m_substep_rtol = 1.e-12_rt;
            NativeRZMagneticPressureDiagnostics rejected;
            AMREX_ALWAYS_ASSERT(
                !hybrid->tryBfieldEvolveNativePressure(U, T, rho, V, 2.e-5_rt, 1, rejected));
            AMREX_ALWAYS_ASSERT(rejected.rejected == 1 && hybrid->m_substeps == 2);
            for (auto const& backup : backups) {
                AMREX_ALWAYS_ASSERT(backup->identical());
            }
            hybrid->m_max_substep_attempts = 250;
            hybrid->m_substep_rtol = 1.e-8_rt;
            amrex::Real const retry_initial_energy = energy();
            NativeRZMagneticPressureDiagnostics retried;
            AMREX_ALWAYS_ASSERT(
                hybrid->tryBfieldEvolveNativePressure(U, T, rho, V, 2.e-5_rt, 1, retried));
            AMREX_ALWAYS_ASSERT(retried.rejected > 0 && retried.accepted > 1);
            AMREX_ALWAYS_ASSERT(std::abs(energy() - retry_initial_energy) <
                                5.e-13_rt * retry_initial_energy);
            auto const adaptive_reference = oracle.integrate(initial, 2.e-5 / 2048, 2048);
            auto const adaptive_B = gather(*B[1], geom), adaptive_U = gather(U, geom);
            double adaptive_error = 0;
            for (int n = 0; n < nb; ++n) {
                adaptive_error = std::max(adaptive_error,
                                          std::abs(adaptive_B[n] - adaptive_reference[n]) / .002);
            }
            for (int n = 0; n < (nr + 1) * nz; ++n) {
                adaptive_error = std::max(adaptive_error,
                                          std::abs(adaptive_U[n] - adaptive_reference[n + nb]) / 2);
            }
            amrex::Print() << "RKF45 oracle_error=" << adaptive_error
                           << " accepted=" << retried.accepted << " rejected=" << retried.rejected
                           << '\n';
            AMREX_ALWAYS_ASSERT(adaptive_error < 5.e-6);

            // Native field checkpoint serialization of the accepted bundle; a new
            // scoped solver has no persistent trial scratch to restore on replay.
            amrex::VisMF::Write(*B[1], "pressure_checkpoint_B");
            amrex::VisMF::Write(U, "pressure_checkpoint_U");
            amrex::VisMF::Write(T, "pressure_checkpoint_T");
            int const checkpoint_substeps = hybrid->m_substeps;
            AMREX_ALWAYS_ASSERT(
                hybrid->tryBfieldEvolveNativePressure(U, T, rho, V, 1.e-6_rt, 1, retried));
            Backup replay_B(*B[1]), replay_U(U), replay_T(T);
            amrex::VisMF::Read(*B[1], "pressure_checkpoint_B");
            amrex::VisMF::Read(U, "pressure_checkpoint_U");
            amrex::VisMF::Read(T, "pressure_checkpoint_T");
            hybrid->m_substeps = checkpoint_substeps;
            AMREX_ALWAYS_ASSERT(
                hybrid->tryBfieldEvolveNativePressure(U, T, rho, V, 1.e-6_rt, 1, retried));
            AMREX_ALWAYS_ASSERT(replay_B.identical() && replay_U.identical() &&
                                replay_T.identical());
            if (collocated) {
                Backup regular_B(*B[1]);
                for (amrex::MFIter it(*B[1]); it.isValid(); ++it) {
                    auto const b = B[1]->array(it);
                    amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        if (i == 0) {
                            b(i, j, k) = 1;
                        }
                    });
                }
                Backup invalid_axis(*B[1]), untouched_U(U), untouched_T(T);
                AMREX_ALWAYS_ASSERT(
                    !hybrid->tryBfieldEvolveNativePressure(U, T, rho, V, 1.e-6_rt, 1, retried));
                AMREX_ALWAYS_ASSERT(invalid_axis.identical() && untouched_U.identical() &&
                                    untouched_T.identical());
                regular_B.restore();
            }

            // A bad initial caloric state on only one rank rejects every rank
            // and preserves even the offending state and all preexisting ghosts.
            if (amrex::ParallelDescriptor::IOProcessor()) {
                T.setVal(-1, 0, 1, 0);
            }
            Backup invalid_T(T), unchanged_U(U), unchanged_B(*B[1]);
            std::vector<std::unique_ptr<Backup>> caches;
            for (int d = 0; d < 3; ++d) {
                caches.push_back(std::make_unique<Backup>(*E[d]));
                caches.push_back(std::make_unique<Backup>(*J[d]));
            }
            AMREX_ALWAYS_ASSERT(
                !hybrid->tryBfieldEvolveNativePressure(U, T, rho, V, 1.e-6_rt, 1, retried));
            AMREX_ALWAYS_ASSERT(invalid_T.identical() && unchanged_U.identical() &&
                                unchanged_B.identical());
            for (auto const& cache : caches) {
                AMREX_ALWAYS_ASSERT(cache->identical());
            }
            amrex::Print() << "native " << (collocated ? "collocated" : "Yee")
                           << " pressure RK/null/nonzero-current/rollback/retry/replay passed\n";
        }
        WarpX::ResetInstance();
    }
    warpx::initialization::finalize_external_libraries();
}
