/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/NativeRZPressureStage.H"
#include "Fields.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_VisMF.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

using namespace amrex::literals;
using warpx::fields::FieldType;

namespace
{
struct State
{
    amrex::MultiFab B;
    amrex::MultiFab U;
    amrex::MultiFab T;
};

amrex::FArrayBox
copyToHost (amrex::MultiFab const& field, amrex::MFIter const& it)
{
    amrex::FArrayBox host(field[it].box(), 1, amrex::The_Pinned_Arena());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, field[it].dataPtr(),
                     field[it].dataPtr() + field[it].box().numPts(), host.dataPtr());
    return host;
}

bool
bitwiseEqual (amrex::MultiFab const& left, amrex::MultiFab const& right)
{
    int different = 0;
    for (amrex::MFIter it(left); it.isValid(); ++it)
    {
        auto const a = copyToHost(left, it);
        auto const b = copyToHost(right, it);
        auto const bytes = static_cast<std::size_t>(left[it].box().numPts()) * sizeof(amrex::Real);
        different |= std::memcmp(a.dataPtr(), b.dataPtr(), bytes) != 0;
    }
    amrex::ParallelDescriptor::ReduceIntMax(different);
    return different == 0;
}

State
makeState (amrex::BoxArray const& boxes, amrex::DistributionMapping const& mapping,
           int const ghosts)
{
    State state;
    state.B.define(boxes, mapping, 1, ghosts);
    state.U.define(boxes, mapping, 1, ghosts);
    state.T.define(boxes, mapping, 1, ghosts);
    state.B.setVal(-123._rt);
    state.U.setVal(-123._rt);
    state.T.setVal(-123._rt);
    return state;
}

void
initializeState (State& state, amrex::MultiFab const& charge, amrex::Geometry const& geometry,
                 ElectronThermodynamicsExecutor eos, bool const null_curl)
{
    auto const dr = geometry.CellSize(0), dz = geometry.CellSize(1);
    for (amrex::MFIter it(state.B); it.isValid(); ++it)
    {
        auto const B = state.B.array(it);
        auto const U = state.U.array(it);
        auto const T = state.T.array(it);
        auto const rho = charge.const_array(it);
        amrex::ParallelFor(
            it.validbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::Real const r = i * dr, z = j * dz;
                amrex::Real const kelvin =
                    null_curl ? 1.e5_rt
                              : 1.e5_rt * (1._rt + 0.15_rt * std::sin(2 * MathConst::pi * z));
                ElectronThermodynamicState const thermal =
                    eos.stateFromChargeDensityTemperature(rho(i, j, k), kelvin);
                T(i, j, k) = kelvin;
                U(i, j, k) = thermal.internal_energy_density;
                B(i, j, k) = 1.e-4_rt * r * (1 - r * r) *
                             (1 + (null_curl ? 0 : 0.1_rt * std::cos(2 * MathConst::pi * z)));
            });
    }
    state.B.OverrideSync(geometry.periodicity());
    state.B.FillBoundary(geometry.periodicity());
}

void
fillVolume (amrex::MultiFab& volume, amrex::Geometry const& geometry)
{
    auto const dr = geometry.CellSize(0), dz = geometry.CellSize(1);
    int const outer = geometry.Domain().bigEnd(0) + 1;
    for (amrex::MFIter it(volume); it.isValid(); ++it)
    {
        auto const V = volume.array(it);
        amrex::ParallelFor(it.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               V(i, j, k) = i == 0 ? MathConst::pi * dr * dr * dz / 3
                                                   : 2 * MathConst::pi * i * dr * dr * dz *
                                                         (i == outer ? 0.5_rt : 1._rt);
                           });
    }
}

void
installState (WarpX& sim, State const& state)
{
    auto B = sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
    for (auto* field : B)
    {
        field->setVal(0._rt);
    }
    amrex::MultiFab::Copy(*B[1], state.B, 0, 0, 1, state.B.nGrowVect());
}

State
captureLiveState (WarpX& sim, amrex::MultiFab const& U, amrex::MultiFab const& T)
{
    auto const B = sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0)[1];
    State snapshot;
    snapshot.B.define(B->boxArray(), B->DistributionMap(), 1, B->nGrowVect());
    snapshot.U.define(U.boxArray(), U.DistributionMap(), 1, U.nGrowVect());
    snapshot.T.define(T.boxArray(), T.DistributionMap(), 1, T.nGrowVect());
    amrex::MultiFab::Copy(snapshot.B, *B, 0, 0, 1, B->nGrowVect());
    amrex::MultiFab::Copy(snapshot.U, U, 0, 0, 1, U.nGrowVect());
    amrex::MultiFab::Copy(snapshot.T, T, 0, 0, 1, T.nGrowVect());
    return snapshot;
}

amrex::Real
maximumAbsoluteChange (amrex::MultiFab const& candidate, amrex::MultiFab const& old)
{
    amrex::MultiFab difference(candidate.boxArray(), candidate.DistributionMap(), 1, 0);
    amrex::MultiFab::Copy(difference, candidate, 0, 0, 1, 0);
    amrex::MultiFab::Subtract(difference, old, 0, 0, 1, 0);
    return difference.norm0(0, 0);
}

std::array<long double, 2>
energies (amrex::MultiFab const& B, amrex::MultiFab const& U, amrex::MultiFab const& volume,
          amrex::Geometry const& geometry)
{
    std::array<long double, 2> total{};
    auto const owner = B.OwnerMask(geometry.periodicity());
    int const outer = geometry.Domain().bigEnd(0) + 1;
    auto const dr = geometry.CellSize(0), dz = geometry.CellSize(1);
    for (amrex::MFIter it(B); it.isValid(); ++it)
    {
        auto const Bh = copyToHost(B, it);
        auto const Uh = copyToHost(U, it);
        auto const Vh = copyToHost(volume, it);
        amrex::IArrayBox mask((*owner)[it].box(), 1, amrex::The_Pinned_Arena());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, (*owner)[it].dataPtr(),
                         (*owner)[it].dataPtr() + (*owner)[it].box().numPts(), mask.dataPtr());
        auto const box = it.validbox();
        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j)
        {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i)
            {
                amrex::IntVect const node(i, j);
                if (mask(node) == 0)
                {
                    continue;
                }
                if (i > 0)
                {
                    long double const magnetic_volume =
                        2.0L * MathConst::pi * i * dr * dr * dz * (i == outer ? 0.5L : 1.0L);
                    total[0] += magnetic_volume * static_cast<long double>(Bh(node)) *
                                static_cast<long double>(Bh(node)) / (2.0L * PhysConst::mu0);
                }
                total[1] += static_cast<long double>(Vh(node)) * static_cast<long double>(Uh(node));
            }
        }
    }
#ifdef AMREX_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, total.data(), 2, MPI_LONG_DOUBLE, MPI_SUM,
                  amrex::ParallelDescriptor::Communicator());
#endif
    return total;
}

bool
runStage (WarpX& sim, HybridPICModel& hybrid, State& state, amrex::MultiFab const& charge,
          amrex::MultiFab const& volume, amrex::Real const dt,
          warpx::hybrid::NativeRZPressureDiagnostics& diagnostics)
{
    auto& U = *sim.m_fields.get(FieldType::hybrid_qdsmc_thermodynamic_fp, 0);
    auto& T = *sim.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
    amrex::MultiFab::Copy(U, state.U, 0, 0, 1, 0);
    amrex::MultiFab::Copy(T, state.T, 0, 0, 1, 0);
    State const before = captureLiveState(sim, U, T);
    bool const accepted = warpx::hybrid::tryAdvanceNativeRZPressureStage(
        sim, hybrid, U, T, charge, volume, dt, hybrid.electronThermodynamicsExecutor(),
        &diagnostics);
    if (accepted)
    {
        amrex::MultiFab::Copy(state.U, U, 0, 0, 1, 0);
        amrex::MultiFab::Copy(state.T, T, 0, 0, 1, 0);
    }
    else
    {
        auto const B = sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0)[1];
        AMREX_ALWAYS_ASSERT(bitwiseEqual(*B, before.B));
        AMREX_ALWAYS_ASSERT(bitwiseEqual(U, before.U));
        AMREX_ALWAYS_ASSERT(bitwiseEqual(T, before.T));
    }
    return accepted;
}

// The columns are measured through native Ohm/BC/Faraday, independently of
// the stage's work stencil. Use unique owner nodes, including periodic copies.
std::vector<amrex::Real>
gatherNodes (amrex::MultiFab const& field, amrex::Geometry const& geom)
{
    int const nz = geom.Domain().length(1);
    std::vector<amrex::Real> values((geom.Domain().length(0) + 1) * nz, 0._rt);
    auto const owner = field.OwnerMask(geom.periodicity());
    for (amrex::MFIter it(field); it.isValid(); ++it)
    {
        auto const host = copyToHost(field, it);
        amrex::IArrayBox mask((*owner)[it].box(), 1, amrex::The_Pinned_Arena());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, (*owner)[it].dataPtr(),
                         (*owner)[it].dataPtr() + (*owner)[it].box().numPts(), mask.dataPtr());
        auto const box = it.validbox();
        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j)
        {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i)
            {
                amrex::IntVect const node(i, j);
                if (mask(node))
                {
                    values[i * nz + j % nz] += host(node);
                }
            }
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(values.data(), static_cast<int>(values.size()));
    return values;
}

void
checkLocalNativeOracle (WarpX& sim, HybridPICModel const& hybrid, State const& initial,
                        State const& final, amrex::MultiFab const& charge,
                        amrex::MultiFab const& volume, amrex::Real const dt)
{
    auto const& geom = sim.Geom(0);
    int const nr = geom.Domain().length(0), nz = geom.Domain().length(1);
    int const count = (nr + 1) * nz;
    auto B = sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
    auto E = sim.m_fields.get_alldirs(FieldType::Efield_fp, 0);
    std::array<amrex::MultiFab, 6> saved;
    std::array<amrex::MultiFab*, 6> fields{B[0], B[1], B[2], E[0], E[1], E[2]};
    for (int f = 0; f < 6; ++f)
    {
        saved[f].define(fields[f]->boxArray(), fields[f]->DistributionMap(), 1,
                        fields[f]->nGrowVect());
        amrex::MultiFab::Copy(saved[f], *fields[f], 0, 0, 1, fields[f]->nGrowVect());
    }
    amrex::MultiFab basis(charge.boxArray(), charge.DistributionMap(), 1, 1);
    std::array<amrex::MultiFab, 3> current, pressure_e;
    ablastr::fields::VectorField J{}, Ep{};
    for (int d = 0; d < 3; ++d)
    {
        current[d].define(E[d]->boxArray(), E[d]->DistributionMap(), 1, E[d]->nGrowVect());
        pressure_e[d].define(E[d]->boxArray(), E[d]->DistributionMap(), 1, E[d]->nGrowVect());
        current[d].setVal(0._rt);
        J[d] = &current[d];
        Ep[d] = &pressure_e[d];
    }
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> no_eb;
    std::vector<std::vector<amrex::Real>> columns(count);
    for (int n = 0; n < count; ++n)
    {
        for (auto* f : fields)
        {
            f->setVal(0._rt);
        }
        basis.setVal(0._rt);
        int const pi = n / nz, pj = n % nz;
        for (amrex::MFIter it(basis); it.isValid(); ++it)
        {
            auto const p = basis.array(it);
            amrex::ParallelFor(it.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   if (i == pi && j % nz == pj)
                                   {
                                       p(i, j, k) = 1._rt;
                                   }
                               });
        }
        basis.OverrideSync(geom.periodicity());
        basis.FillBoundary(geom.periodicity());
        sim.get_pointer_fdtd_solver_fp(0)->HybridPICSolveE(E, Ep, J, J, B, charge, basis, no_eb, 0,
                                                           &hybrid, false);
        sim.ApplyEfieldBoundary(0, PatchType::fine, 0);
        for (auto* e : E)
        {
            e->OverrideSync(geom.periodicity());
            e->FillBoundary(geom.periodicity());
        }
        sim.EvolveB(0, PatchType::fine, 1._rt, SubcyclingHalf::FirstHalf, 0._rt);
        columns[n] = gatherNodes(*B[1], geom);
    }
    auto const b0 = gatherNodes(initial.B, geom), b1 = gatherNodes(final.B, geom);
    auto const u0 = gatherNodes(initial.U, geom), u1 = gatherNodes(final.U, geom);
    auto const rho = gatherNodes(charge, geom), vol = gatherNodes(volume, geom);
    auto const eos = hybrid.electronThermodynamicsExecutor();
    auto const dr = geom.CellSize(0), dz = geom.CellSize(1);
    std::vector<long double> p0(count), pmid(count), bh(count), wb(count);
    for (int n = 0; n < count; ++n)
    {
        auto const t = eos.temperatureFromChargeDensityThermalEnergyDensity(rho[n], u0[n]);
        p0[n] = eos.stateFromChargeDensityTemperature(rho[n], t).pressure;
        int const i = n / nz;
        wb[n] = 2.L * MathConst::pi * i * dr * dr * dz * (i == nr ? .5L : 1.L);
        bh[n] = b0[n];
    }
    for (int n = 0; n < count; ++n)
    {
        for (int k = 0; k < count; ++k)
        {
            bh[k] += .5L * dt * columns[n][k] * p0[n];
        }
    }
    for (int n = 0; n < count; ++n)
    {
        long double dot = 0;
        for (int k = 0; k < count; ++k)
        {
            dot += columns[n][k] * wb[k] * (b0[k] + bh[k]) / (2 * PhysConst::mu0);
        }
        amrex::Real const uh = static_cast<amrex::Real>(u0[n] - .5L * dt * p0[n] * dot / vol[n]);
        auto const th = eos.temperatureFromChargeDensityThermalEnergyDensity(rho[n], uh);
        pmid[n] = eos.stateFromChargeDensityTemperature(rho[n], th).pressure;
    }
    long double max_work = 0, max_error = 0, max_thermal = 0, max_b_error = 0;
    for (int n = 0; n < count; ++n)
    {
        long double dot = 0, expected_b = b0[n];
        for (int k = 0; k < count; ++k)
        {
            dot += columns[n][k] * wb[k] * (b0[k] + b1[k]) / (2 * PhysConst::mu0);
            expected_b += dt * columns[k][n] * pmid[k];
        }
        long double const expected = -dt * pmid[n] * dot;
        long double const actual = (static_cast<long double>(u1[n]) - u0[n]) * vol[n];
        max_work = std::max(max_work, std::abs(expected));
        max_error = std::max(max_error, std::abs(expected - actual));
        max_thermal = std::max(max_thermal, static_cast<long double>(u0[n]) * vol[n]);
        max_b_error = std::max(max_b_error, std::abs(expected_b - b1[n]));
    }
    amrex::Print() << "wave5_native_local_work=" << max_work << " error=" << max_error
                   << " midpoint_B_error=" << max_b_error << '\n';
    AMREX_ALWAYS_ASSERT(max_work > 1.e-9L * max_thermal);
    AMREX_ALWAYS_ASSERT(max_error <
                        1.e-9L * max_work +
                            128.L * std::numeric_limits<amrex::Real>::epsilon() * max_thermal);
    AMREX_ALWAYS_ASSERT(max_b_error < 512.L * std::numeric_limits<amrex::Real>::epsilon() *
                                          std::max(initial.B.norm0(0, 0), final.B.norm0(0, 0)));
    for (int f = 0; f < 6; ++f)
    {
        amrex::MultiFab::Copy(*fields[f], saved[f], 0, 0, 1, fields[f]->nGrowVect());
    }
}
} // namespace

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        auto const& geometry = sim.Geom(0);
        auto* hybrid = sim.get_pointer_HybridPICModel();
        AMREX_ALWAYS_ASSERT(hybrid != nullptr && AMREX_SPACEDIM == 2);
        auto const eos = hybrid->electronThermodynamicsExecutor();

        auto const live_B = sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
        auto const& boxes = live_B[1]->boxArray();
        auto const& mapping = live_B[1]->DistributionMap();
        amrex::MultiFab charge(boxes, mapping, 1, 1);
        amrex::MultiFab volume(boxes, mapping, 1, 1);
        amrex::MultiFab deposited_charge(boxes, mapping, 1,
                                         sim.m_fields.get(FieldType::rho_fp, 0)->nGrowVect());
        sim.GetPartContainer().GetParticleContainerFromName("ions").DepositChargeOnLevel(
            &deposited_charge, 0, false, true);
        sim.FinalizeRhoOnLevel(0, &deposited_charge, PatchType::fine);
        amrex::MultiFab::Copy(charge, deposited_charge, 0, 0, 1, 0);
        charge.OverrideSync(geometry.periodicity());
        charge.FillBoundary(geometry.periodicity());
        fillVolume(volume, geometry);
        AMREX_ALWAYS_ASSERT(charge.min(0) >
                            std::abs(PhysConst::q_e) * hybrid->electronDensityFloor());

        for (int control = 0; control < 2; ++control)
        {
            bool const null_curl = control == 1;
            State state = makeState(boxes, mapping, 1);
            initializeState(state, charge, geometry, eos, null_curl);
            installState(sim, state);
            State const original = captureLiveState(sim, state.U, state.T);
            warpx::hybrid::NativeRZPressureDiagnostics diagnostics{};
            AMREX_ALWAYS_ASSERT(
                runStage(sim, *hybrid, state, charge, volume, 1.e-7_rt, diagnostics));
            AMREX_ALWAYS_ASSERT(diagnostics.max_abs_ampere_current > 1.e-4_rt);

            State const candidate = captureLiveState(sim, state.U, state.T);
            amrex::Real const B_change = maximumAbsoluteChange(candidate.B, original.B);
            amrex::Real const U_change = maximumAbsoluteChange(candidate.U, original.U);
            amrex::Print() << "wave5_native_pressure control=" << control
                           << " ampere=" << diagnostics.max_abs_ampere_current << " dB=" << B_change
                           << " dU=" << U_change
                           << " operator_B_error=" << diagnostics.max_native_transpose_B_error
                           << '\n';
            if (null_curl)
            {
                AMREX_ALWAYS_ASSERT(B_change <= 64._rt *
                                                    std::numeric_limits<amrex::Real>::epsilon() *
                                                    state.B.norm0(0, 0));
                AMREX_ALWAYS_ASSERT(U_change <= 64._rt *
                                                    std::numeric_limits<amrex::Real>::epsilon() *
                                                    state.U.norm0(0, 0));
            }
            else
            {
                AMREX_ALWAYS_ASSERT(B_change > 1.e-10_rt);
                AMREX_ALWAYS_ASSERT(U_change > 1.e-6_rt);
                AMREX_ALWAYS_ASSERT(diagnostics.max_native_transpose_B_error <=
                                    512._rt * std::numeric_limits<amrex::Real>::epsilon() *
                                        diagnostics.max_native_transpose_B_scale);
                AMREX_ALWAYS_ASSERT(diagnostics.max_native_transpose_U_error <=
                                    512._rt * std::numeric_limits<amrex::Real>::epsilon() *
                                        diagnostics.max_native_transpose_U_scale);

                auto const before = energies(original.B, original.U, volume, geometry);
                auto const after = energies(candidate.B, candidate.U, volume, geometry);
                long double const ledger_error =
                    std::abs((after[0] - before[0]) + (after[1] - before[1]));
                amrex::Print() << "wave5_material_ledger magnetic_dE=" << (after[0] - before[0])
                               << " material_dE=" << (after[1] - before[1])
                               << " error=" << ledger_error << '\n';
                AMREX_ALWAYS_ASSERT(ledger_error <=
                                    512.0L * std::numeric_limits<amrex::Real>::epsilon() *
                                        (before[0] + before[1]));
                checkLocalNativeOracle(sim, *hybrid, original, candidate, charge, volume, 1.e-7_rt);

                // A huge invalid candidate must reject on every rank and
                // restore every valid and ghost cell, including scratch E.
                State rejected = makeState(boxes, mapping, 1);
                initializeState(rejected, charge, geometry, eos, false);
                for (amrex::MFIter it(rejected.U); it.isValid(); ++it)
                {
                    auto const U = rejected.U.array(it);
                    auto const T = rejected.T.array(it);
                    auto const rho = charge.const_array(it);
                    amrex::ParallelFor(it.validbox(),
                                       [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                       {
                                           U(i, j, k) *= 1.e-8_rt;
                                           T(i, j, k) =
                                               eos.temperatureFromChargeDensityThermalEnergyDensity(
                                                   rho(i, j, k), U(i, j, k));
                                       });
                }
                installState(sim, rejected);
                State const before_reject = captureLiveState(sim, rejected.U, rejected.T);
                auto old_B = sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
                auto E = sim.m_fields.get_alldirs(FieldType::Efield_fp, 0);
                std::array<amrex::MultiFab, 3> saved_B;
                std::array<amrex::MultiFab, 3> old_E;
                for (int d = 0; d < 3; ++d)
                {
                    saved_B[d].define(old_B[d]->boxArray(), old_B[d]->DistributionMap(), 1,
                                      old_B[d]->nGrowVect());
                    old_E[d].define(E[d]->boxArray(), E[d]->DistributionMap(), 1,
                                    E[d]->nGrowVect());
                    amrex::MultiFab::Copy(saved_B[d], *old_B[d], 0, 0, 1, old_B[d]->nGrowVect());
                    amrex::MultiFab::Copy(old_E[d], *E[d], 0, 0, 1, E[d]->nGrowVect());
                }
                warpx::hybrid::NativeRZPressureDiagnostics rejection{};
                AMREX_ALWAYS_ASSERT(
                    !runStage(sim, *hybrid, rejected, charge, volume, 1._rt, rejection));
                State const restored = captureLiveState(sim, rejected.U, rejected.T);
                AMREX_ALWAYS_ASSERT(bitwiseEqual(restored.B, before_reject.B));
                AMREX_ALWAYS_ASSERT(bitwiseEqual(restored.U, before_reject.U));
                AMREX_ALWAYS_ASSERT(bitwiseEqual(restored.T, before_reject.T));
                for (int d = 0; d < 3; ++d)
                {
                    AMREX_ALWAYS_ASSERT(bitwiseEqual(*old_B[d], saved_B[d]));
                }
                for (int d = 0; d < 3; ++d)
                {
                    AMREX_ALWAYS_ASSERT(bitwiseEqual(*E[d], old_E[d]));
                }

                // A rank-local invalid volume must reject collectively.
                amrex::MultiFab invalid_volume(boxes, mapping, 1, 1);
                amrex::MultiFab::Copy(invalid_volume, volume, 0, 0, 1, 1);
                for (amrex::MFIter it(invalid_volume); it.isValid(); ++it)
                {
                    auto const v = invalid_volume.array(it);
                    amrex::ParallelFor(it.validbox(),
                                       [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                       {
                                           if (i == 1 && j == 1)
                                           {
                                               v(i, j, k) = -1._rt;
                                           }
                                       });
                }
                AMREX_ALWAYS_ASSERT(
                    !runStage(sim, *hybrid, rejected, charge, invalid_volume, 1.e-7_rt, rejection));
                for (int d = 0; d < 3; ++d)
                {
                    AMREX_ALWAYS_ASSERT(bitwiseEqual(*old_B[d], saved_B[d]));
                    AMREX_ALWAYS_ASSERT(bitwiseEqual(*E[d], old_E[d]));
                }

                auto& native_U = *sim.m_fields.get(FieldType::hybrid_qdsmc_thermodynamic_fp, 0);
                auto& native_T = *sim.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
                // Save/reload the actual native Btheta/U/T bundle between
                // equal steps. This is state I/O, not WarpX checkpoint schema
                // or production RK restart qualification.
                initializeState(state, charge, geometry, eos, false);
                installState(sim, state);
                warpx::hybrid::NativeRZPressureDiagnostics unused{};
                for (int n = 0; n < 4; ++n)
                {
                    AMREX_ALWAYS_ASSERT(
                        runStage(sim, *hybrid, state, charge, volume, 1.e-7_rt, unused));
                }
                State const full = captureLiveState(sim, native_U, native_T);
                initializeState(state, charge, geometry, eos, false);
                installState(sim, state);
                for (int n = 0; n < 2; ++n)
                {
                    AMREX_ALWAYS_ASSERT(
                        runStage(sim, *hybrid, state, charge, volume, 1.e-7_rt, unused));
                }
                State checkpoint = captureLiveState(sim, native_U, native_T);
                amrex::VisMF::Write(checkpoint.B, "wave5_native_B");
                amrex::VisMF::Write(checkpoint.U, "wave5_native_U");
                amrex::VisMF::Write(checkpoint.T, "wave5_native_T");
                initializeState(state, charge, geometry, eos, false);
                installState(sim, state);
                amrex::VisMF::Read(checkpoint.B, "wave5_native_B");
                amrex::VisMF::Read(checkpoint.U, "wave5_native_U");
                amrex::VisMF::Read(checkpoint.T, "wave5_native_T");
                state = std::move(checkpoint);
                installState(sim, state);
                for (int n = 0; n < 2; ++n)
                {
                    AMREX_ALWAYS_ASSERT(
                        runStage(sim, *hybrid, state, charge, volume, 1.e-7_rt, unused));
                }
                State const resumed = captureLiveState(sim, native_U, native_T);
                AMREX_ALWAYS_ASSERT(bitwiseEqual(full.B, resumed.B));
                AMREX_ALWAYS_ASSERT(bitwiseEqual(full.U, resumed.U));
                AMREX_ALWAYS_ASSERT(bitwiseEqual(full.T, resumed.T));
                amrex::Print() << "wave5 native state reload: B/U/T identical including ghosts\n";

                // Resolved pressure feedback must converge at second order,
                // separately from the algebraic energy cancellation.
                auto const integrate = [&] (int const steps)
                {
                    initializeState(state, charge, geometry, eos, false);
                    installState(sim, state);
                    for (int n = 0; n < steps; ++n)
                    {
                        AMREX_ALWAYS_ASSERT(runStage(sim, *hybrid, state, charge, volume,
                                                     1.e-5_rt / steps, unused));
                    }
                    return captureLiveState(sim, native_U, native_T);
                };
                State const coarse = integrate(32), fine = integrate(64),
                            reference = integrate(256);
                auto const b_coarse = maximumAbsoluteChange(coarse.B, reference.B);
                auto const b_fine = maximumAbsoluteChange(fine.B, reference.B);
                auto const u_coarse = maximumAbsoluteChange(coarse.U, reference.U);
                auto const u_fine = maximumAbsoluteChange(fine.U, reference.U);
                amrex::Print() << "wave5 temporal B ratio=" << b_coarse / b_fine
                               << " U ratio=" << u_coarse / u_fine << " B errors=" << b_coarse
                               << "," << b_fine << " U errors=" << u_coarse << "," << u_fine
                               << '\n';
                AMREX_ALWAYS_ASSERT(b_coarse > 3._rt * b_fine && b_coarse < 5._rt * b_fine);
                AMREX_ALWAYS_ASSERT(u_coarse > 3._rt * u_fine && u_coarse < 5._rt * u_fine);
            }
        }

        amrex::Print() << "Wave-5 native pressure transaction passed; this is "
                          "not production RZ pressure activation.\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
