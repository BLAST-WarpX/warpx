/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "NativeRZPressureStage.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Fields.H"
#include "HybridPICModel.H"
#include "RZPressureInduction.H"
#include "RZPressureWork.H"
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

namespace warpx::hybrid
{
using namespace amrex::literals;

namespace
{
constexpr amrex::Real temperature_tolerance_multiplier = amrex::Real(256.0);
constexpr amrex::Real energy_tolerance_multiplier = amrex::Real(2048.0);
constexpr amrex::Real operator_tolerance_multiplier = amrex::Real(512.0);

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
finiteReal (amrex::Real const value)
{
    return amrex::Math::isfinite(value);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
positiveFinite (amrex::Real const value)
{
    return finiteReal(value) && value > amrex::Real(0.0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
near (amrex::Real const value, amrex::Real const expected, amrex::Real const multiplier)
{
    amrex::Real const scale =
        amrex::max(amrex::max(amrex::Math::abs(value), amrex::Math::abs(expected)),
                   std::numeric_limits<amrex::Real>::min());
    return amrex::Math::abs(value - expected) <=
           multiplier * std::numeric_limits<amrex::Real>::epsilon() * scale;
}

int
globalInvalid (int invalid)
{
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    return invalid;
}

void
fillRadialGhosts (amrex::MultiFab& field, amrex::Geometry const& geometry, bool const odd)
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
                               data(i, j, k, n) =
                                   (odd && low ? -1._rt : 1._rt) * data(source, j, k, n);
                           });
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
admissibleThermodynamicState (ElectronThermodynamicsExecutor const& eos, amrex::Real const charge,
                              amrex::Real const U, amrex::Real const T,
                              ElectronThermodynamicState& state)
{
    if (!positiveFinite(charge) || !positiveFinite(U) || !positiveFinite(T))
    {
        return false;
    }
    amrex::Real const inverted_temperature =
        eos.temperatureFromChargeDensityThermalEnergyDensity(charge, U);
    if (!positiveFinite(inverted_temperature) || inverted_temperature < eos.minimumTemperature() ||
        inverted_temperature > eos.maximumTemperature() ||
        !near(T, inverted_temperature, temperature_tolerance_multiplier))
    {
        return false;
    }
    state = eos.stateFromChargeDensityTemperature(charge, inverted_temperature);
    amrex::Real const energy_tolerance =
        energy_tolerance_multiplier * std::numeric_limits<amrex::Real>::epsilon() *
        amrex::max(amrex::Math::abs(U), std::numeric_limits<amrex::Real>::min());
    return positiveFinite(state.pressure) && positiveFinite(state.internal_energy_density) &&
           positiveFinite(state.heat_capacity_density) &&
           amrex::Math::abs(state.internal_energy_density - U) <= energy_tolerance;
}

int
reconstructPressure (amrex::MultiFab const& U, amrex::MultiFab const& T,
                     amrex::MultiFab const& charge, ElectronThermodynamicsExecutor const& eos,
                     bool const require_input_temperature, amrex::MultiFab& pressure)
{
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    for (amrex::MFIter it(pressure); it.isValid(); ++it)
    {
        auto const input_U = U.const_array(it);
        auto const input_T = T.const_array(it);
        auto const rho = charge.const_array(it);
        auto const output = pressure.array(it);
        ops.eval(it.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                 {
                     amrex::Real const temperature =
                         eos.temperatureFromChargeDensityThermalEnergyDensity(rho(i, j, k),
                                                                              input_U(i, j, k));
                     ElectronThermodynamicState state{};
                     bool const valid = admissibleThermodynamicState(
                         eos, rho(i, j, k), input_U(i, j, k),
                         require_input_temperature ? input_T(i, j, k) : temperature, state);
                     output(i, j, k) =
                         valid ? state.pressure : std::numeric_limits<amrex::Real>::quiet_NaN();
                     return {!valid};
                 });
    }
    return globalInvalid(amrex::get<0>(data.value()));
}

int
fillTemperatureFromEnergy (amrex::MultiFab const& U, amrex::MultiFab const& charge,
                           ElectronThermodynamicsExecutor const& eos,
                           amrex::Geometry const& geometry, amrex::MultiFab& T)
{
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    for (amrex::MFIter it(T); it.isValid(); ++it)
    {
        auto const U_arr = U.const_array(it);
        auto const rho = charge.const_array(it);
        auto const T_arr = T.array(it);
        ops.eval(it.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                 {
                     amrex::Real const value = eos.temperatureFromChargeDensityThermalEnergyDensity(
                         rho(i, j, k), U_arr(i, j, k));
                     T_arr(i, j, k) = value;
                     return {!positiveFinite(value)};
                 });
    }
    if (globalInvalid(amrex::get<0>(data.value())) != 0)
    {
        return 1;
    }
    fillRadialGhosts(T, geometry, false);
    return 0;
}

void
setLivePressureTrialState (WarpX& sim, HybridPICModel const& hybrid, amrex::MultiFab const& charge,
                           amrex::MultiFab const& B0, amrex::MultiFab const& pressure)
{
    auto B = sim.m_fields.get_alldirs(warpx::fields::FieldType::Bfield_fp, 0);
    auto E = sim.m_fields.get_alldirs(warpx::fields::FieldType::Efield_fp, 0);
    for (int d = 0; d < 3; ++d)
    {
        B[d]->setVal(0._rt);
        E[d]->setVal(0._rt);
    }
    amrex::MultiFab::Copy(*B[1], B0, 0, 0, 1, 0);
    fillRadialGhosts(*B[1], sim.Geom(0), true);

    auto const& boxes = E[0]->boxArray();
    auto const& mapping = E[0]->DistributionMap();
    std::array<amrex::MultiFab, 3> zero_plasma_current;
    std::array<amrex::MultiFab, 3> zero_ion_current;
    std::array<amrex::MultiFab, 3> pressure_E;
    for (int d = 0; d < 3; ++d)
    {
        zero_plasma_current[d].define(boxes, mapping, 1, 1);
        zero_ion_current[d].define(boxes, mapping, 1, 1);
        pressure_E[d].define(boxes, mapping, 1, E[d]->nGrowVect());
        zero_plasma_current[d].setVal(0._rt);
        zero_ion_current[d].setVal(0._rt);
        pressure_E[d].setVal(0._rt);
    }
    ablastr::fields::VectorField plasma{};
    ablastr::fields::VectorField ion{};
    ablastr::fields::VectorField isolated_pressure_E{};
    for (int d = 0; d < 3; ++d)
    {
        plasma[d] = &zero_plasma_current[d];
        ion[d] = &zero_ion_current[d];
        isolated_pressure_E[d] = &pressure_E[d];
    }

    std::array<std::unique_ptr<amrex::iMultiFab>, 3> no_eb;
    // The native Faraday Ohm entry still zeros the pressure gradient. Obtain
    // its isolated push-field pressure output, then pass that channel to Faraday.
    sim.get_pointer_fdtd_solver_fp(0)->HybridPICSolveE(E, isolated_pressure_E, plasma, ion, B,
                                                       charge, pressure, no_eb, 0, &hybrid,
                                                       /*solve_for_Faraday=*/false);

    // Use only the independently output pressure channel in Faraday.  This is
    // stronger than zeroing inputs: even an accidental non-pressure term in
    // total E cannot enter the bounded candidate.
    for (int d = 0; d < 3; ++d)
    {
        amrex::MultiFab::Copy(*E[d], pressure_E[d], 0, 0, 1, pressure_E[d].nGrowVect());
    }
    sim.ApplyEfieldBoundary(0, PatchType::fine, 0);
    sim.ApplyFieldBoundaryOnAxis(E[0], E[1], E[2], 0);
    for (auto* field : E)
    {
        field->OverrideSync(sim.Geom(0).periodicity());
        field->FillBoundary(sim.Geom(0).periodicity());
    }

    for (int d = 0; d < 3; ++d)
    {
        B[d]->setVal(0._rt);
    }
    amrex::MultiFab::Copy(*B[1], B0, 0, 0, 1, 0);
    fillRadialGhosts(*B[1], sim.Geom(0), true);
}

void
nativeFaradayCandidate (WarpX& sim, HybridPICModel const& hybrid, amrex::MultiFab const& charge,
                        amrex::MultiFab const& B0, amrex::MultiFab const& pressure,
                        amrex::Real const dt, amrex::MultiFab& candidate)
{
    setLivePressureTrialState(sim, hybrid, charge, B0, pressure);
    sim.EvolveB(0, PatchType::fine, dt, SubcyclingHalf::FirstHalf, 0._rt);
    auto B = sim.m_fields.get_alldirs(warpx::fields::FieldType::Bfield_fp, 0);
    amrex::MultiFab::Copy(candidate, *B[1], 0, 0, 1, 0);
    fillRadialGhosts(candidate, sim.Geom(0), true);
}

int
updateEnergyFromActualCandidate (amrex::MultiFab const& B0, amrex::MultiFab const& B1,
                                 amrex::MultiFab const& old_pressure, amrex::MultiFab const& charge,
                                 amrex::MultiFab const& caloric_volume,
                                 amrex::Geometry const& geometry, amrex::Real const dt,
                                 amrex::MultiFab& next_energy)
{
    auto const& boxes = B0.boxArray();
    auto const& mapping = B0.DistributionMap();
    amrex::MultiFab midpoint(boxes, mapping, 1, 1);
    amrex::MultiFab work_current(boxes, mapping, 2, 1);
    amrex::MultiFab::Copy(midpoint, B0, 0, 0, 1, 0);
    amrex::MultiFab::Add(midpoint, B1, 0, 0, 1, 0);
    midpoint.mult(0.5_rt, 0, 1, 0);
    fillRadialGhosts(midpoint, geometry, true);

    auto const domain = amrex::convert(geometry.Domain(), B0.ixType());
    int const axis = domain.smallEnd(0), outer = domain.bigEnd(0);
    auto const dr = geometry.CellSize(0), dz = geometry.CellSize(1);
    for (amrex::MFIter it(work_current); it.isValid(); ++it)
    {
        auto const b = midpoint.const_array(it);
        auto const current = work_current.array(it);
        amrex::ParallelFor(
            it.validbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                auto const magnetic_volume = [=] AMREX_GPU_HOST_DEVICE(int r)
                {
                    return 2 * MathConst::pi * (r - axis) * dr * dr * dz *
                           (r == outer ? 0.5_rt : 1._rt);
                };
                current(i, j, k, 0) = i == axis ? 0
                                                : -magnetic_volume(i) *
                                                      (b(i, j + 1, k) - b(i, j - 1, k)) /
                                                      (2 * dz * PhysConst::mu0);
                auto const lower = i > axis + 1 ? magnetic_volume(i - 1) * b(i - 1, j, k) : 0;
                auto const upper = i < outer ? magnetic_volume(i + 1) * b(i + 1, j, k) *
                                                   (i + 1 == outer ? 2._rt : 1._rt)
                                             : 0;
                current(i, j, k, 1) = i == outer ? 0 : (upper - lower) / (2 * dr * PhysConst::mu0);
            });
    }
    fillRadialGhosts(work_current, geometry, false);

    amrex::MultiFab state(boxes, mapping, 2, 1);
    for (amrex::MFIter it(state); it.isValid(); ++it)
    {
        auto const p = old_pressure.const_array(it);
        auto const rho = charge.const_array(it);
        auto const output = state.array(it);
        amrex::ParallelFor(it.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               output(i, j, k, 0) = p(i, j, k);
                               output(i, j, k, 1) = 1 / rho(i, j, k);
                           });
    }
    fillRadialGhosts(state, geometry, false);

    int const zlo = domain.smallEnd(1), zhi = domain.bigEnd(1);
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    for (amrex::MFIter it(next_energy); it.isValid(); ++it)
    {
        auto const s = state.const_array(it);
        auto const current = work_current.const_array(it);
        auto const old = next_energy.const_array(it);
        auto const volume = caloric_volume.const_array(it);
        auto const next = next_energy.array(it);
        ops.eval(it.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                 {
                     auto const axial = amrex::Array4<amrex::Real const>(current, 1);
                     auto const paired =
                         rzPressureWorkDivergence(current, axial, s, i, j, axis, outer, zlo, zhi,
                                                  true, 1 / dr, 1 / dz, volume(i, j, k));
                     next(i, j, k) = old(i, j, k) + dt * s(i, j, k, 0) * paired;
                     return {!finiteReal(next(i, j, k)) || !(next(i, j, k) > amrex::Real(0.0))};
                 });
    }
    return globalInvalid(amrex::get<0>(data.value()));
}

void
fillCandidateMaterial (amrex::MultiFab const& old_state, amrex::MultiFab const& valid_energy,
                       amrex::Geometry const& geometry, amrex::MultiFab& candidate)
{
    amrex::MultiFab::Copy(candidate, old_state, 0, 0, 1, old_state.nGrowVect());
    amrex::MultiFab::Copy(candidate, valid_energy, 0, 0, 1, 0);
    fillRadialGhosts(candidate, geometry, false);
}

struct Transaction
{
    std::array<amrex::MultiFab, 3> B;
    std::array<amrex::MultiFab, 3> E;
    amrex::MultiFab U;
    amrex::MultiFab T;
};

void
restoreTransaction (WarpX& sim, Transaction const& saved, amrex::MultiFab& U, amrex::MultiFab& T)
{
    auto B = sim.m_fields.get_alldirs(warpx::fields::FieldType::Bfield_fp, 0);
    auto E = sim.m_fields.get_alldirs(warpx::fields::FieldType::Efield_fp, 0);
    for (int d = 0; d < 3; ++d)
    {
        amrex::MultiFab::Copy(*B[d], saved.B[d], 0, 0, 1, saved.B[d].nGrowVect());
        amrex::MultiFab::Copy(*E[d], saved.E[d], 0, 0, 1, saved.E[d].nGrowVect());
    }
    amrex::MultiFab::Copy(U, saved.U, 0, 0, 1, saved.U.nGrowVect());
    amrex::MultiFab::Copy(T, saved.T, 0, 0, 1, saved.T.nGrowVect());
}

class TransactionGuard
{
  public:
    TransactionGuard (WarpX& sim, Transaction const& saved, amrex::MultiFab& U, amrex::MultiFab& T)
        : m_sim(sim), m_saved(saved), m_U(U), m_T(T)
    {
    }
    ~TransactionGuard ()
    {
        if (m_armed)
        {
            restoreTransaction(m_sim, m_saved, m_U, m_T);
        }
    }
    TransactionGuard (TransactionGuard const&) = delete;
    TransactionGuard& operator=(TransactionGuard const&) = delete;
    void
    disarm ()
    {
        m_armed = false;
    }

  private:
    WarpX& m_sim;
    Transaction const& m_saved;
    amrex::MultiFab& m_U;
    amrex::MultiFab& m_T;
    bool m_armed = true;
};
} // namespace

bool
tryAdvanceNativeRZPressureStage (WarpX& sim, HybridPICModel const& hybrid,
                                 amrex::MultiFab& internal_energy, amrex::MultiFab& temperature,
                                 amrex::MultiFab const& charge,
                                 amrex::MultiFab const& caloric_volume, amrex::Real dt,
                                 ElectronThermodynamicsExecutor eos,
                                 NativeRZPressureDiagnostics* diagnostics)
{
    NativeRZPressureDiagnostics local_diagnostics{};
    if (!eos.isIdealGas() || eos.m_num_latent_transitions != 0 || eos.m_num_materials != 0 ||
        eos.m_use_resolved_material_selection || hybrid.electronThermodynamicsNumMaterials() != 0 ||
        !finiteReal(dt) || dt <= amrex::Real(0.0) || EB::enabled() ||
        WarpX::grid_type != ablastr::utils::enums::GridType::Collocated ||
        WarpX::n_rz_azimuthal_modes != 1 || !sim.get_ng_fieldgather().allGE(amrex::IntVect(1)))
    {
        if (diagnostics != nullptr)
        {
            *diagnostics = local_diagnostics;
        }
        return false;
    }

    auto B = sim.m_fields.get_alldirs(warpx::fields::FieldType::Bfield_fp, 0);
    auto E = sim.m_fields.get_alldirs(warpx::fields::FieldType::Efield_fp, 0);
    auto const& geometry = sim.Geom(0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        AMREX_SPACEDIM == 2 && sim.finestLevel() == 0 && !sim.DoPML() &&
            WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC &&
            WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC &&
            WarpX::field_boundary_lo[0] == FieldBoundaryType::None && B[0] != nullptr &&
            B[1] != nullptr && B[2] != nullptr && E[0] != nullptr && E[1] != nullptr &&
            E[2] != nullptr && geometry.ProbLo(0) == 0 && !geometry.isPeriodic(0) &&
            geometry.isPeriodic(1) && B[1]->ixType().nodeCentered(),
        "Native RZ pressure requires an axis/PEC, periodic-z collocated "
        "nodal native mesh.");

    std::array<amrex::MultiFab const*, 4> const scalar_fields{&internal_energy, &temperature,
                                                              &charge, &caloric_volume};
    for (auto const* field : scalar_fields)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            field->nComp() == 1 && field->boxArray() == B[1]->boxArray() &&
                field->DistributionMap() == B[1]->DistributionMap() &&
                field->ixType() == B[1]->ixType(),
            "Native RZ pressure fields must match the scalar Btheta layout.");
        for (int d = 0; d < 3; ++d)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                field != B[d] && field != E[d],
                "Native RZ pressure material fields must be distinct from B/E fields.");
        }
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        &internal_energy != &temperature && &internal_energy != &charge &&
            &internal_energy != &caloric_volume && &temperature != &charge &&
            &temperature != &caloric_volume && &charge != &caloric_volume,
        "Native RZ pressure state arrays must be independently owned.");
    for (int d = 0; d < 3; ++d)
    {
        for (auto const* field : {B[d], E[d]})
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                field->nComp() == 1 && field->ixType() == B[1]->ixType() &&
                    field->boxArray() == B[1]->boxArray() &&
                    field->DistributionMap() == B[1]->DistributionMap() &&
                    field->nGrowVect().allGE(amrex::IntVect(1)),
                "Native RZ pressure admits matching regular mode-0 field layouts with ghosts.");
        }
    }

    Transaction saved{};
    auto const& boxes = B[1]->boxArray();
    auto const& mapping = B[1]->DistributionMap();
    auto const B_ghosts = B[1]->nGrowVect();
    for (int d = 0; d < 3; ++d)
    {
        saved.B[d].define(B[d]->boxArray(), B[d]->DistributionMap(), 1, B[d]->nGrowVect());
        saved.E[d].define(E[d]->boxArray(), E[d]->DistributionMap(), 1, E[d]->nGrowVect());
        amrex::MultiFab::Copy(saved.B[d], *B[d], 0, 0, 1, B[d]->nGrowVect());
        amrex::MultiFab::Copy(saved.E[d], *E[d], 0, 0, 1, E[d]->nGrowVect());
    }
    saved.U.define(boxes, mapping, 1, internal_energy.nGrowVect());
    saved.T.define(boxes, mapping, 1, temperature.nGrowVect());
    amrex::MultiFab::Copy(saved.U, internal_energy, 0, 0, 1, internal_energy.nGrowVect());
    amrex::MultiFab::Copy(saved.T, temperature, 0, 0, 1, temperature.nGrowVect());
    // Propagate recoverable native-call exceptions with the original state
    // restored. A process/device failure or failure of restoration itself is
    // outside this in-memory transaction's contract.
    TransactionGuard transaction_guard(sim, saved, internal_energy, temperature);

    auto const restore_and_return = [&] (bool const accepted)
    {
        if (diagnostics != nullptr)
        {
            *diagnostics = local_diagnostics;
        }
        return accepted;
    };

    amrex::MultiFab old_B(boxes, mapping, 1, B_ghosts);
    amrex::MultiFab old_pressure(boxes, mapping, 1, 1);
    amrex::MultiFab::Copy(old_B, saved.B[1], 0, 0, 1, 0);
    fillRadialGhosts(old_B, geometry, true);
    if (reconstructPressure(internal_energy, temperature, charge, eos, true, old_pressure) != 0)
    {
        return restore_and_return(false);
    }
    fillRadialGhosts(old_pressure, geometry, false);

    // Require every node to lie above the model's density floor.  Thus the
    // native Ohm implementation's safety denominator never silently replaces
    // the frozen positive charge in this bounded trial.
    amrex::Real const minimum_charge = charge.min(0);
    amrex::Real const charge_floor = std::abs(PhysConst::q_e) * hybrid.electronDensityFloor();
    amrex::Real const minimum_volume = caloric_volume.min(0);
    if (!(minimum_charge > charge_floor) || !positiveFinite(minimum_volume))
    {
        return restore_and_return(false);
    }

    auto const nodal_domain = amrex::convert(geometry.Domain(), B[1]->ixType());
    int const axis = nodal_domain.smallEnd(0);
    {
        amrex::ReduceOps<amrex::ReduceOpMax> ops;
        amrex::ReduceData<int> data(ops);
        for (amrex::MFIter it(old_B); it.isValid(); ++it)
        {
            auto const b = old_B.const_array(it);
            auto const volume = caloric_volume.const_array(it);
            ops.eval(it.validbox(), data,
                     [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                     {
                         return {!positiveFinite(volume(i, j, k)) || !finiteReal(b(i, j, k)) ||
                                 (i == axis && b(i, j, k) != amrex::Real(0.0))};
                     });
        }
        if (globalInvalid(amrex::get<0>(data.value())) != 0)
        {
            return restore_and_return(false);
        }
    }

    // Qualify the actual native Ampere curl, but do not use this pointwise
    // current as pressure work.  The weighted transpose below is the conjugate
    // magnetic current for the Faraday operator.
    std::array<amrex::MultiFab, 3> ampere_B;
    std::array<amrex::MultiFab, 3> ampere_J;
    ablastr::fields::VectorField ampere_B_field{};
    ablastr::fields::VectorField ampere_J_field{};
    for (int d = 0; d < 3; ++d)
    {
        // Ampere evaluates one ghost layer of J, which reads two B layers.
        ampere_B[d].define(B[d]->boxArray(), B[d]->DistributionMap(), 1, 2);
        ampere_J[d].define(E[d]->boxArray(), E[d]->DistributionMap(), 1, 1);
        ampere_B[d].setVal(0._rt);
        ampere_J[d].setVal(0._rt);
        ampere_B_field[d] = &ampere_B[d];
        ampere_J_field[d] = &ampere_J[d];
    }
    amrex::MultiFab::Copy(ampere_B[1], old_B, 0, 0, 1, 1);
    fillRadialGhosts(ampere_B[1], geometry, true);
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> no_eb;
    sim.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(ampere_J_field, ampere_B_field, no_eb,
                                                              0);
    amrex::Real max_ampere = 0;
    for (auto const& current : ampere_J)
    {
        max_ampere = std::max(max_ampere, current.max(0));
        max_ampere = std::max(max_ampere, -current.min(0));
    }
    amrex::ParallelDescriptor::ReduceRealMax(max_ampere);
    local_diagnostics.max_abs_ampere_current = max_ampere;
    if (!finiteReal(max_ampere))
    {
        return restore_and_return(false);
    }

    // A half-step predictor gives P(U(t + dt/2)) to O(dt^2). The final
    // forward curl and work transpose use that same predicted pressure.
    amrex::Real const predictor_dt = 0.5_rt * dt;
    amrex::MultiFab trial_B(boxes, mapping, 1, B_ghosts);
    nativeFaradayCandidate(sim, hybrid, charge, old_B, old_pressure, predictor_dt, trial_B);
    amrex::MultiFab trial_energy(boxes, mapping, 1, 0);
    amrex::MultiFab::Copy(trial_energy, internal_energy, 0, 0, 1, 0);
    if (updateEnergyFromActualCandidate(old_B, trial_B, old_pressure, charge, caloric_volume,
                                        geometry, predictor_dt, trial_energy) != 0)
    {
        return restore_and_return(false);
    }
    amrex::MultiFab trial_U(boxes, mapping, 1, internal_energy.nGrowVect());
    amrex::MultiFab trial_T(boxes, mapping, 1, temperature.nGrowVect());
    fillCandidateMaterial(internal_energy, trial_energy, geometry, trial_U);
    if (fillTemperatureFromEnergy(trial_U, charge, eos, geometry, trial_T) != 0)
    {
        return restore_and_return(false);
    }
    amrex::MultiFab trial_pressure(boxes, mapping, 1, 1);
    if (reconstructPressure(trial_U, trial_T, charge, eos, true, trial_pressure) != 0)
    {
        return restore_and_return(false);
    }
    fillRadialGhosts(trial_pressure, geometry, false);

    // Final native Faraday candidate uses pressure evaluated from trial U.
    amrex::MultiFab final_B(boxes, mapping, 1, B_ghosts);
    nativeFaradayCandidate(sim, hybrid, charge, old_B, trial_pressure, dt, final_B);
    amrex::MultiFab final_energy(boxes, mapping, 1, 0);
    amrex::MultiFab::Copy(final_energy, internal_energy, 0, 0, 1, 0);
    if (updateEnergyFromActualCandidate(old_B, final_B, trial_pressure, charge, caloric_volume,
                                        geometry, dt, final_energy) != 0)
    {
        return restore_and_return(false);
    }
    amrex::MultiFab final_U(boxes, mapping, 1, internal_energy.nGrowVect());
    amrex::MultiFab final_T(boxes, mapping, 1, temperature.nGrowVect());
    fillCandidateMaterial(internal_energy, final_energy, geometry, final_U);
    if (fillTemperatureFromEnergy(final_U, charge, eos, geometry, final_T) != 0)
    {
        return restore_and_return(false);
    }
    amrex::MultiFab operator_pressure(boxes, mapping, 1, 1);
    amrex::MultiFab::Copy(operator_pressure, trial_pressure, 0, 0, 1, 1);
    if (reconstructPressure(final_U, final_T, charge, eos, true, trial_pressure) != 0)
    {
        return restore_and_return(false);
    }

    // Independently qualify the actual candidate against the already-tested
    // collocated operator.  The native candidate remains the state used by
    // work and commit; any disagreement rejects it on every rank.
    amrex::MultiFab primitive_B(boxes, mapping, 1, 1);
    amrex::MultiFab primitive_U(boxes, mapping, 1, 1);
    amrex::MultiFab::Copy(primitive_B, old_B, 0, 0, 1, 0);
    amrex::MultiFab::Copy(primitive_U, internal_energy, 0, 0, 1, 0);
    amrex::RealBox const extent(geometry.ProbLo(), geometry.ProbHi());
    int const periodic[2] = {geometry.isPeriodic(0), geometry.isPeriodic(1)};
    amrex::Geometry const cylindrical_geometry(geometry.Domain(), &extent, 1, periodic);
    if (!tryAdvanceRZPressureInduction(primitive_B, primitive_U, operator_pressure, charge,
                                       caloric_volume, cylindrical_geometry, dt))
    {
        return restore_and_return(false);
    }
    amrex::MultiFab difference(boxes, mapping, 1, 0);
    amrex::MultiFab::Copy(difference, final_B, 0, 0, 1, 0);
    amrex::MultiFab::Subtract(difference, primitive_B, 0, 0, 1, 0);
    amrex::Real const B_error = difference.norm0(0, 0);
    amrex::Real const B_scale =
        std::max({old_B.norm0(0, 0), final_B.norm0(0, 0), primitive_B.norm0(0, 0),
                  std::numeric_limits<amrex::Real>::min()});
    amrex::MultiFab::Copy(difference, final_U, 0, 0, 1, 0);
    amrex::MultiFab::Subtract(difference, primitive_U, 0, 0, 1, 0);
    amrex::Real const U_error = difference.norm0(0, 0);
    amrex::Real const U_scale =
        std::max({internal_energy.norm0(0, 0), final_U.norm0(0, 0), primitive_U.norm0(0, 0),
                  std::numeric_limits<amrex::Real>::min()});
    local_diagnostics.max_native_transpose_B_error = B_error;
    local_diagnostics.max_native_transpose_B_scale = B_scale;
    local_diagnostics.max_native_transpose_U_error = U_error;
    local_diagnostics.max_native_transpose_U_scale = U_scale;
    if (!(B_error <=
          operator_tolerance_multiplier * std::numeric_limits<amrex::Real>::epsilon() * B_scale) ||
        !(U_error <=
          operator_tolerance_multiplier * std::numeric_limits<amrex::Real>::epsilon() * U_scale))
    {
        return restore_and_return(false);
    }

    // Commit only after all global checks.  E remains scratch and is restored;
    // the B transaction commits actual native Btheta plus frozen Br/Bz.
    restoreTransaction(sim, saved, internal_energy, temperature);
    auto final_B_live = sim.m_fields.get_alldirs(warpx::fields::FieldType::Bfield_fp, 0)[1];
    amrex::MultiFab::Copy(*final_B_live, final_B, 0, 0, 1, final_B.nGrowVect());
    amrex::MultiFab::Copy(internal_energy, final_U, 0, 0, 1, final_U.nGrowVect());
    amrex::MultiFab::Copy(temperature, final_T, 0, 0, 1, final_T.nGrowVect());
    if (diagnostics != nullptr)
    {
        *diagnostics = local_diagnostics;
    }
    transaction_guard.disarm();
    return true;
}
} // namespace warpx::hybrid
