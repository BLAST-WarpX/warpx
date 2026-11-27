/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "PacketThermalExchange.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FluxRelaxedPackets.H"
#include "MaterialKineticWork.H"
#include "NativePhotonReplayInternals.H"
#include "ParticleImpulse.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhotonParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "PhotonFourMomentum.H"
#include "PhotonPacketTransaction.H"
#include "WarpX.H"

#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

using namespace amrex::literals;

namespace warpx::radiation
{
namespace
{
amrex::MultiFab
CopyField (amrex::MultiFab const& source)
{
    amrex::MultiFab result(source.boxArray(), source.DistributionMap(), source.nComp(),
                          source.nGrowVect());
    amrex::MultiFab::Copy(result, source, 0, 0, source.nComp(), source.nGrowVect());
    return result;
}

bool
SameField (amrex::MultiFab const& current, amrex::MultiFab const& baseline)
{
    bool valid = current.boxArray() == baseline.boxArray() &&
        current.DistributionMap() == baseline.DistributionMap() &&
        current.nComp() == baseline.nComp() && current.nGrowVect() == baseline.nGrowVect();
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { return false; }
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    using Tuple = typename decltype(data)::Type;
    for (amrex::MFIter it(current); it.isValid(); ++it) {
        auto const now = current.const_array(it), old = baseline.const_array(it);
        auto const components = current.nComp();
        ops.eval(it.fabbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
            for (int n = 0; n < components; ++n) {
                if (now(i, j, k, n) != old(i, j, k, n)) { return {1}; }
            }
            return {0};
        });
    }
    int invalid = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    return invalid == 0;
}

bool
SameParticles (WarpXParticleContainer::Base const& current,
               photon_replay_detail::CapturedParticles const& old)
{
    auto const now = photon_replay_detail::CaptureParticles(current);
    if (now.tiles.size() != old.tiles.size()) { return false; }
    for (std::size_t tile = 0; tile < now.tiles.size(); ++tile) {
        auto const& a = now.tiles[tile];
        auto const& b = old.tiles[tile];
        if (a.key != b.key || a.count != b.count ||
            !photon_replay_detail::SameBits(a.real_data, b.real_data) ||
            !photon_replay_detail::SameBits(a.int_data, b.int_data) ||
            !photon_replay_detail::SameBits(a.idcpu, b.idcpu)) { return false; }
    }
    return true;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
Ulp (amrex::Real value)
{
    auto const inf = std::numeric_limits<amrex::Real>::infinity();
    return amrex::max(std::abs(std::nextafter(value, inf) - value),
                      std::abs(value - std::nextafter(value, -inf)));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
CaloricBound (ElectronThermodynamicsExecutor eos, amrex::Real rho, amrex::Real temperature)
{
    auto const state = eos.stateFromChargeDensityTemperature(rho, temperature);
    if (eos.isIdealGas()) {
        return state.heat_capacity_density * Ulp(temperature) +
               0.5_rt * Ulp(state.internal_energy_density);
    }
    auto const capacity = rho / PhysConst::q_e * PhysConst::kb / (eos.m_gamma - 1);
    auto const radius = Ulp(temperature) +
        std::ldexp((state.internal_energy_density + Ulp(state.internal_energy_density)) /
                   capacity, -64);
    auto const lo = eos.stateFromChargeDensityTemperature(
        rho, amrex::max(0._rt, temperature - radius)).internal_energy_density;
    auto const hi = eos.stateFromChargeDensityTemperature(
        rho, temperature + radius).internal_energy_density;
    return amrex::max(std::abs(hi - state.internal_energy_density),
                      std::abs(state.internal_energy_density - lo)) +
           0.5_rt * amrex::max(Ulp(lo), Ulp(hi));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
Gamma (amrex::Real operations)
{
    auto const neps = operations * std::numeric_limits<amrex::Real>::epsilon();
    return neps / (1 - neps);
}
} // namespace

namespace
{
struct PacketThermalState
{
    MultiParticleContainer* particles;
    HybridPICModel const* hybrid;
    WarpXParticleContainer* photons;
    amrex::MultiFab *live_temperature, *live_density, *live_pressure;
    amrex::Geometry geometry;
    ElectronThermodynamicsExecutor eos;
    amrex::Real minimum_density;
    amrex::MultiFab old_temperature, old_density, old_pressure, temperature;
    amrex::MultiFab photon_debit, photon_operands, impulse, requested_heat, returned_heat, actual_heat;
    amrex::MultiFab redistribution, photon_count, caloric_bound, momentum_bound;
    std::vector<std::string> species_names;
    std::string path;
    std::vector<photon_replay_detail::CapturedParticles> material_baseline;
    photon_replay_detail::CapturedParticles photon_baseline;
    std::unique_ptr<PhotonPacketTransaction> photon_trial;
    std::unique_ptr<ParticleImpulseMaterial> material_trial;
    std::unique_ptr<ParticleImpulseTransaction> material_stage;
    PacketThermalReport report;
    bool supported = false, attempted = false, committed = false;

    PacketThermalState (MultiParticleContainer& pc, std::vector<std::string> names, std::string carry,
          WarpXParticleContainer& ph, HybridPICModel const& model, amrex::Real minimum)
        : particles(&pc), hybrid(&model), photons(&ph),
          live_temperature(WarpX::GetInstance().m_fields.get(
              warpx::fields::FieldType::hybrid_electron_temperature_fp, 0)),
          live_density(WarpX::GetInstance().m_fields.get(warpx::fields::FieldType::rho_fp, 0)),
          live_pressure(WarpX::GetInstance().m_fields.get(
              warpx::fields::FieldType::hybrid_electron_pressure_fp, 0)),
          geometry(WarpX::GetInstance().Geom(0)), eos(model.electronThermodynamicsExecutor()),
          minimum_density(minimum), old_temperature(CopyField(*live_temperature)),
          old_density(CopyField(*live_density)), old_pressure(CopyField(*live_pressure)),
          temperature(CopyField(old_temperature)),
          photon_debit(amrex::convert(old_temperature.boxArray(), amrex::IntVect(0)),
                       old_temperature.DistributionMap(), 4, 0),
          photon_operands(photon_debit.boxArray(), photon_debit.DistributionMap(), 4, 0),
          impulse(photon_debit.boxArray(), photon_debit.DistributionMap(), 3, 0),
          requested_heat(photon_debit.boxArray(), photon_debit.DistributionMap(), 1, 1),
          returned_heat(photon_debit.boxArray(), photon_debit.DistributionMap(), 1, 1),
          actual_heat(photon_debit.boxArray(), photon_debit.DistributionMap(), 1, 0),
          redistribution(photon_debit.boxArray(), photon_debit.DistributionMap(), 1, 0),
          photon_count(photon_debit.boxArray(), photon_debit.DistributionMap(), 1, 0),
          caloric_bound(photon_debit.boxArray(), photon_debit.DistributionMap(), 1, 0),
          momentum_bound(photon_debit.boxArray(), photon_debit.DistributionMap(), 3, 0),
          species_names(std::move(names)), path(std::move(carry)),
          photon_baseline(photon_replay_detail::CaptureParticles(ph))
    {
        supported = WarpX::GetInstance().finestLevel() == 0 && geometry.Coord() == 0 &&
            !EB::enabled() && !particles->hasHybridIonization() &&
            model.solvesElectronEnergyEquation() && std::isfinite(eos.m_gamma) && eos.m_gamma > 1 &&
            std::numeric_limits<amrex::Real>::digits >= 53 &&
            std::numeric_limits<amrex::ParticleReal>::digits >= 53 &&
            (eos.isIdealGas() || eos.isFixedChargeLatentEnergy()) && eos.m_num_materials == 0 &&
            std::isfinite(minimum_density) && minimum_density >= 0 && !path.empty() &&
            !species_names.empty() &&
            std::set<std::string>(species_names.begin(), species_names.end()).size() ==
                species_names.size() && old_temperature.ixType().nodeCentered() &&
            old_temperature.nGrowVect().allGE(1) && old_density.boxArray() ==
                old_temperature.boxArray() && old_density.nGrowVect().allGE(1) &&
            dynamic_cast<PhotonParticleContainer*>(photons) != nullptr &&
            photons->ParticleBoxArray(0) == photon_debit.boxArray() &&
            photons->ParticleDistributionMap(0) == photon_debit.DistributionMap();
#if !defined(WARPX_DIM_1D_Z) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_3D)
        supported = false;
#endif
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { supported &= geometry.isPeriodic(d); }
        for (auto const& name : species_names) {
            auto& species = pc.GetParticleContainerFromName(name);
            supported &= !species.HasEvolvingChargeState() && species.getMass() > 0 &&
                species.ParticleBoxArray(0) == photon_debit.boxArray() &&
                species.ParticleDistributionMap(0) == photon_debit.DistributionMap();
            auto const& columns = species.GetRealSoANames();
            for (auto const* suffix : {"ux", "uy", "uz", "work"}) {
                auto const column = "radiation_impulse_" + path + "_" + suffix;
                supported &= std::find(columns.begin(), columns.end(), column) != columns.end();
            }
            material_baseline.push_back(photon_replay_detail::CaptureParticles(species));
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(supported);
        photon_debit.setVal(0);
        photon_operands.setVal(0);
        photon_count.setVal(0);
        requested_heat.setVal(0);
        momentum_bound.setVal(0);
        if (supported) {
            photon_trial = std::make_unique<PhotonPacketTransaction>(ph);
            material_trial = std::make_unique<ParticleImpulseMaterial>(pc, species_names);
        }
    }

    PacketThermalReport reject (PacketThermalFailure failure)
    {
        report.valid = false;
        report.failure = failure;
        return report;
    }

    bool baselineUnchanged () const
    {
        // Evaluate every collective even after a local participant fails.
        bool valid = SameField(*live_temperature, old_temperature);
        valid = SameField(*live_density, old_density) && valid;
        valid = SameField(*live_pressure, old_pressure) && valid;
        valid = SameParticles(*photons, photon_baseline) && valid;
        for (std::size_t n = 0; n < species_names.size(); ++n) {
            valid = SameParticles(particles->GetParticleContainerFromName(species_names[n]),
                                  material_baseline[n]) && valid;
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        return valid;
    }
};

bool
LowBeta (PacketThermalState& s)
{
    amrex::Gpu::DeviceScalar<int> invalid(0);
    auto* bad = invalid.dataPtr();
    auto const lo = s.geometry.ProbLoArray(), dxi = s.geometry.InvCellSizeArray();
    auto const first = amrex::lbound(s.geometry.Domain());
    for (auto const& name : s.species_names) {
        auto& species = s.particles->GetParticleContainerFromName(name);
        amrex::GpuArray<int, 3> carry{};
        for (int d = 0; d < 3; ++d) {
            carry[d] = species.GetRealCompIndex("radiation_impulse_" + s.path + "_" +
                std::string(d == 0 ? "ux" : d == 1 ? "uy" : "uz"));
        }
        auto const rest_mass = static_cast<amrex::Real>(species.getMass());
        for (WarpXParIter it(species, 0); it.isValid(); ++it) {
            auto const data = it.GetParticleTile().getParticleTileData();
            amrex::GpuArray<amrex::ParticleReal const*, 3> carry_values{};
            for (int d = 0; d < 3; ++d) {
                carry_values[d] = it.GetStructOfArrays().GetRealData(carry[d]).dataPtr();
            }
            auto const pos = GetParticlePosition<PIdx>(it);
            auto const mass = s.material_stage->MaterialMass().const_array(it);
            auto const force = s.impulse.const_array(it);
            auto const bound = s.momentum_bound.array(it);
            amrex::For(it.numParticles(), [=] AMREX_GPU_DEVICE(long p) {
                amrex::ParticleReal x = 0, y = 0, z = 0;
                pos(p, x, y, z);
                if (!amrex::Math::isfinite(x) || !amrex::Math::isfinite(y) ||
                    !amrex::Math::isfinite(z)) {
                    amrex::HostDevice::Atomic::Add(bad, 1); return;
                }
                int i = first.x + static_cast<int>(std::floor((z - lo[0]) * dxi[0]));
                int j = 0, k = 0;
#if AMREX_SPACEDIM >= 2
                i = first.x + static_cast<int>(std::floor((x - lo[0]) * dxi[0]));
                j = first.y + static_cast<int>(std::floor((z - lo[1]) * dxi[1]));
#endif
#if AMREX_SPACEDIM == 3
                j = first.y + static_cast<int>(std::floor((y - lo[1]) * dxi[1]));
                k = first.z + static_cast<int>(std::floor((z - lo[2]) * dxi[2]));
#endif
                if (!mass.contains(i, j, k)) {
                    amrex::HostDevice::Atomic::Add(bad, 1); return;
                }
                amrex::GpuArray<amrex::ParticleReal, 3> u{};
                amrex::GpuArray<amrex::Real, 3> pending{}, du{};
                for (int d = 0; d < 3; ++d) {
                    u[d] = data.m_rdata[PIdx::ux + d][p];
                    pending[d] = carry_values[d][p];
                    du[d] = mass(i, j, k) > 0 ? force(i, j, k, d) / mass(i, j, k) : 0;
                }
                auto const candidate = EvaluateMaterialImpulseCandidate(u, pending, 0, du);
                amrex::Real old_u2 = 0, new_u2 = 0;
                for (int d = 0; d < 3; ++d) {
                    old_u2 += u[d] * u[d];
                    new_u2 += candidate.velocity[d] * candidate.velocity[d];
                    // Include the old/new carry operands, not only their
                    // potentially cancelled difference or bulk momentum.
                    auto const weighted_mass = data.m_rdata[PIdx::w][p] * rest_mass;
                    auto const allowance = weighted_mass *
                        (Ulp(pending[d]) + Ulp(candidate.momentum_carry[d]) +
                         Gamma(16) * (std::abs(du[d]) + std::abs(pending[d]) +
                                       std::abs(candidate.momentum_carry[d])));
                    amrex::Gpu::Atomic::AddNoRet(&bound(i, j, k, d), allowance);
                }
                if (!candidate.valid || !amrex::Math::isfinite(old_u2 + new_u2) ||
                    old_u2 / (PhysConst::c2 + old_u2) > 1.e-4_rt ||
                    new_u2 / (PhysConst::c2 + new_u2) > 1.e-4_rt) {
                    amrex::HostDevice::Atomic::Add(bad, 1);
                }
            });
        }
    }
    int flag = invalid.dataValue();
    amrex::ParallelDescriptor::ReduceIntMax(flag);
    return flag == 0;
}

PacketThermalReport
FinishThermal (PacketThermalState& s, PacketThermalEvent event)
{
    for (amrex::MFIter it(s.impulse); it.isValid(); ++it) {
        auto const debit = s.photon_debit.const_array(it);
        auto const force = s.impulse.array(it);
        amrex::For(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            for (int d = 0; d < 3; ++d) { force(i, j, k, d) = debit(i, j, k, d + 1) / PhysConst::c; }
        });
    }
    s.material_stage = std::make_unique<ParticleImpulseTransaction>();
    if (!s.material_stage->Stage(*s.material_trial, s.path, s.impulse, true,
                               ParticleImpulseAssignment::NearestCell)) {
        return s.reject(PacketThermalFailure::MaterialStage);
    }
    if (!LowBeta(s)) { return s.reject(PacketThermalFailure::LowBeta); }
    amrex::Gpu::DeviceScalar<int> invalid(0);
    auto* bad = invalid.dataPtr();
    for (amrex::MFIter it(s.requested_heat); it.isValid(); ++it) {
        auto const debit = s.photon_debit.const_array(it);
        auto const q = s.requested_heat.array(it);
        auto const work = s.material_stage->ActualWork().const_array(it);
        auto const carry = s.material_stage->EnergyCarryChange().const_array(it);
        amrex::For(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto const partition = EvaluatePacketThermalPartition(
                event, debit(i, j, k), work(i, j, k), carry(i, j, k));
            q(i, j, k) = partition.requested_heat;
            if (!partition.valid) { amrex::HostDevice::Atomic::Add(bad, 1); }
        });
    }
    int flag = invalid.dataValue();
    amrex::ParallelDescriptor::ReduceIntMax(flag);
    if (flag != 0) { return s.reject(PacketThermalFailure::NegativeAbsorptionHeat); }
    s.requested_heat.FillBoundary(s.geometry.periodicity());
    amrex::MultiFab::Copy(s.returned_heat, s.requested_heat, 0, 0, 1, 1);
    // The native source evaluator reads live rho. It may only do so while
    // that field still equals the frozen density used by our EOS oracle.
    if (!s.baselineUnchanged()) { return s.reject(PacketThermalFailure::LiveChanged); }
    auto const residual = s.hybrid->EvaluateElectronEnergySource(0, s.temperature, s.returned_heat,
                                                              s.minimum_density);
    bool valid = std::isfinite(residual) && s.temperature.is_finite() && s.returned_heat.is_finite();
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { return s.reject(PacketThermalFailure::CaloricResponse); }
    s.temperature.FillBoundary(s.geometry.periodicity());
    auto const dx = s.geometry.CellSizeArray();
    auto const volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
    auto const closure = s.eos;
    auto const threshold = s.minimum_density;
    auto const floor = s.hybrid->transportsElectronInternalEnergyAtRawDensity()
        ? 0._rt : s.hybrid->electronDensityFloor();
    for (amrex::MFIter it(s.actual_heat); it.isValid(); ++it) {
        auto const rho = s.old_density.const_array(it), old = s.old_temperature.const_array(it);
        auto const next = s.temperature.const_array(it);
        auto const actual = s.actual_heat.array(it);
        auto const bound = s.caloric_bound.array(it);
        auto const requested = s.requested_heat.const_array(it);
        auto const realized = s.returned_heat.const_array(it);
        auto const remap = s.redistribution.array(it);
        amrex::For(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real heat = 0, allowance = 0;
            for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner) {
                int ni = i + (corner & 1), nj = j, nk = k;
#if AMREX_SPACEDIM >= 2
                nj += (corner >> 1) & 1;
#endif
#if AMREX_SPACEDIM == 3
                nk += (corner >> 2) & 1;
#endif
                if (rho(ni, nj, nk) <= PhysConst::q_e * threshold) { continue; }
                auto const charge = amrex::max(rho(ni, nj, nk), PhysConst::q_e * floor);
                auto const before = old(ni, nj, nk), after = next(ni, nj, nk);
                auto const increment = closure.internalEnergyDensityDifference(charge, {}, before, after);
                heat += volume / (1 << AMREX_SPACEDIM) * increment;
                if (before != after) {
                    allowance += volume / (1 << AMREX_SPACEDIM) *
                        (CaloricBound(closure, charge, after) + Gamma(8) * std::abs(increment));
                }
            }
            actual(i, j, k) = heat;
            bound(i, j, k) = allowance;
            remap(i, j, k) = heat - requested(i, j, k);
            if (!amrex::Math::isfinite(heat + allowance) ||
                std::abs(heat - realized(i, j, k)) > allowance +
                Gamma(8) * (std::abs(heat) + std::abs(realized(i, j, k)))) {
                amrex::HostDevice::Atomic::Add(bad, 1);
            }
        });
    }
    flag = invalid.dataValue();
    amrex::ParallelDescriptor::ReduceIntMax(flag);
    if (flag != 0) { return s.reject(PacketThermalFailure::Conservation); }
    s.report.photon_debit = s.photon_debit.sum(0);
    s.report.actual_ion_work = s.material_stage->ActualWork().sum(0);
    s.report.energy_carry_change = s.material_stage->EnergyCarryChange().sum(0);
    s.report.requested_heat = s.requested_heat.sum(0);
    s.report.actual_heat = s.actual_heat.sum(0);
    auto const cells = static_cast<amrex::Real>(s.geometry.Domain().numPts());
    auto const operations = 128 + 8 * cells + 4 * amrex::ParallelDescriptor::NProcs();
    s.report.energy_roundoff_bound = s.caloric_bound.sum(0) +
        s.material_stage->WorkAccountingRoundoffBound().sum(0) + Gamma(operations) *
        (s.photon_operands.norm1(0) + s.material_stage->ActualWork().norm1(0) +
         s.material_stage->EnergyCarryChange().norm1(0) + s.requested_heat.norm1(0) +
         s.actual_heat.norm1(0));
    s.report.energy_residual = s.report.photon_debit - s.report.actual_ion_work -
                            s.report.energy_carry_change - s.report.actual_heat;
    valid = std::isfinite(s.report.energy_residual + s.report.energy_roundoff_bound) &&
            std::abs(s.report.energy_residual) <= s.report.energy_roundoff_bound;
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    using Tuple = typename decltype(data)::Type;
    for (amrex::MFIter it(s.impulse); it.isValid(); ++it) {
        auto const force = s.impulse.const_array(it), bound = s.momentum_bound.const_array(it);
        auto const actual = s.material_stage->ActualImpulse().const_array(it);
        auto const carry = s.material_stage->MomentumCarryChange().const_array(it);
        auto const count = s.photon_count.const_array(it);
        auto const photon_scale = s.photon_operands.const_array(it);
        ops.eval(it.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
            for (int d = 0; d < 3; ++d) {
                auto const error = force(i, j, k, d) - actual(i, j, k, d) - carry(i, j, k, d);
                auto const scale = std::abs(force(i, j, k, d)) + std::abs(actual(i, j, k, d)) +
                                   std::abs(carry(i, j, k, d)) +
                                   photon_scale(i, j, k, d + 1) / PhysConst::c;
                if (!amrex::Math::isfinite(error + scale + bound(i, j, k, d)) ||
                    std::abs(error) > bound(i, j, k, d) + Gamma(8 + 4 * count(i, j, k)) * scale) {
                    return {1};
                }
            }
            return {0};
        });
    }
    flag = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceIntMax(flag);
    for (int d = 0; d < 3; ++d) {
        s.report.momentum_residual[d] = s.impulse.sum(d) -
            s.material_stage->ActualImpulse().sum(d) -
            s.material_stage->MomentumCarryChange().sum(d);
    }
    valid = valid && flag == 0;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { return s.reject(PacketThermalFailure::Conservation); }
    s.report.valid = true;
    return s.report;
}
} // namespace

struct PacketThermalExchangeTransaction::Impl : PacketThermalState
{
    using PacketThermalState::PacketThermalState;
};

PacketThermalExchangeTransaction::PacketThermalExchangeTransaction (
    MultiParticleContainer& pc, std::vector<std::string> const& species, std::string const& path,
    WarpXParticleContainer& photons, HybridPICModel const& hybrid, amrex::Real minimum)
    : m_impl(std::make_unique<Impl>(pc, species, path, photons, hybrid, minimum)) {}

PacketThermalExchangeTransaction::~PacketThermalExchangeTransaction () = default;

PacketThermalReport
PacketThermalExchangeTransaction::StageAbsorption (std::vector<std::uint64_t> const& local_keys)
{
    auto& s = *m_impl;
    if (!s.supported || s.attempted) { return s.reject(PacketThermalFailure::UnsupportedState); }
    s.attempted = true;
    if (!s.baselineUnchanged()) { return s.reject(PacketThermalFailure::LiveChanged); }
    auto keys = local_keys;
    std::sort(keys.begin(), keys.end());
    bool valid = std::adjacent_find(keys.begin(), keys.end()) == keys.end();
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { return s.reject(PacketThermalFailure::InvalidPhotonSelection); }
    amrex::Gpu::DeviceVector<std::uint64_t> device_keys(keys.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, keys.begin(), keys.end(), device_keys.begin());
    auto const* selected = device_keys.data();
    auto const count = keys.size();
    amrex::Gpu::DeviceScalar<int> matched(0), invalid(0);
    auto* found = matched.dataPtr();
    auto* bad = invalid.dataPtr();
    auto& candidate = s.photon_trial->Particles();
    auto const lo = s.geometry.ProbLoArray(), dxi = s.geometry.InvCellSizeArray();
    auto const first = amrex::lbound(s.geometry.Domain());
    for (WarpXParIter it(candidate, 0); it.isValid(); ++it) {
        auto const data = it.GetParticleTile().getParticleTileData();
        auto const pos = GetParticlePosition<PIdx>(it);
        auto const debit = s.photon_debit.array(it), numbers = s.photon_count.array(it);
        auto const operands = s.photon_operands.array(it);
        amrex::For(it.numParticles(), [=] AMREX_GPU_DEVICE(long p) {
            std::size_t left = 0, right = count;
            while (left < right) {
                auto const middle = left + (right - left) / 2;
                if (selected[middle] < data.m_idcpu[p]) { left = middle + 1; }
                else { right = middle; }
            }
            if (left == count || selected[left] != data.m_idcpu[p]) { return; }
            auto const photon = PacketConversionFourMomentumFromPhoton(data.m_rdata[PIdx::w][p],
                data.m_rdata[PIdx::ux][p], data.m_rdata[PIdx::uy][p], data.m_rdata[PIdx::uz][p]);
            if (!photon.valid || !ValidFiniteMassPhoton(photon.value)) {
                amrex::HostDevice::Atomic::Add(bad, 1); return;
            }
            amrex::ParticleReal x = 0, y = 0, z = 0;
            pos(p, x, y, z);
            if (!amrex::Math::isfinite(x) || !amrex::Math::isfinite(y) ||
                !amrex::Math::isfinite(z)) {
                amrex::HostDevice::Atomic::Add(bad, 1); return;
            }
            int i = first.x + static_cast<int>(std::floor((z - lo[0]) * dxi[0]));
            int j = 0, k = 0;
#if AMREX_SPACEDIM >= 2
            i = first.x + static_cast<int>(std::floor((x - lo[0]) * dxi[0]));
            j = first.y + static_cast<int>(std::floor((z - lo[1]) * dxi[1]));
#endif
#if AMREX_SPACEDIM == 3
            j = first.y + static_cast<int>(std::floor((y - lo[1]) * dxi[1]));
            k = first.z + static_cast<int>(std::floor((z - lo[2]) * dxi[2]));
#endif
            if (!debit.contains(i, j, k)) {
                amrex::HostDevice::Atomic::Add(bad, 1); return;
            }
            for (int d = 0; d < 4; ++d) {
                amrex::Gpu::Atomic::AddNoRet(&debit(i, j, k, d), photon.value[d]);
                amrex::Gpu::Atomic::AddNoRet(&operands(i, j, k, d), std::abs(photon.value[d]));
            }
            amrex::Gpu::Atomic::AddNoRet(&numbers(i, j, k), 1._rt);
            amrex::ParticleIDWrapper(data.m_idcpu[p]) = -1;
            amrex::HostDevice::Atomic::Add(found, 1);
        });
    }
    valid = invalid.dataValue() == 0 && static_cast<std::size_t>(matched.dataValue()) == count;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { return s.reject(PacketThermalFailure::InvalidPhotonSelection); }
    candidate.Redistribute();
    return FinishThermal(s, PacketThermalEvent::Absorption);
}

PacketThermalReport
PacketThermalExchangeTransaction::StageEmission (detail::EmissionPlan const& plan)
{
    auto& s = *m_impl;
    if (!s.supported || s.attempted) { return s.reject(PacketThermalFailure::UnsupportedState); }
    s.attempted = true;
    if (!s.baselineUnchanged()) { return s.reject(PacketThermalFailure::LiveChanged); }
    bool valid = plan.logical_draw_keys.empty();
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { return s.reject(PacketThermalFailure::InvalidBirths); }
    auto& candidate = s.photon_trial->Particles();
    int grid = 0;
    for (int n = 0; n < candidate.ParticleBoxArray(0).size(); ++n) {
        if (candidate.ParticleDistributionMap(0)[n] == amrex::ParallelDescriptor::MyProc()) {
            grid = n; break;
        }
    }
    auto const prior = candidate.GetParticles(0).find({grid, 0});
    auto const first_birth = prior == candidate.GetParticles(0).end() ? 0 : prior->second.numParticles();
    amrex::Vector<amrex::Vector<amrex::ParticleReal>> attributes{plan.weight};
    if (!s.photon_trial->AddProvisionalBirths(grid, static_cast<long>(plan.weight.size()),
        plan.x, plan.y, plan.z, plan.ux, plan.uy, plan.uz, attributes, {}, true)) {
        return s.reject(PacketThermalFailure::InvalidBirths);
    }
    // Re-read the exact stored native birth batch before redistribution. This
    // avoids subtracting an arbitrarily large untouched photon background.
    auto const born = candidate.GetParticles(0).find({grid, 0});
    if (born != candidate.GetParticles(0).end() && !plan.weight.empty()) {
        auto const data = born->second.getParticleTileData();
        amrex::Gpu::HostVector<amrex::ParticleReal> weight(plan.weight.size());
        amrex::GpuArray<amrex::Gpu::HostVector<amrex::ParticleReal>, 3> momentum;
        auto const& soa = born->second.GetStructOfArrays();
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetRealData(PIdx::w).begin() + first_birth,
            soa.GetRealData(PIdx::w).end(), weight.begin());
        for (int d = 0; d < 3; ++d) {
            momentum[d].resize(weight.size());
            auto const& values = soa.GetRealData(PIdx::ux + d);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin() + first_birth,
                values.end(), momentum[d].begin());
        }
        amrex::ignore_unused(data);
        auto host = amrex::MultiFab(s.photon_debit.boxArray(), s.photon_debit.DistributionMap(),
            4, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
        auto counts = amrex::MultiFab(s.photon_count.boxArray(), s.photon_count.DistributionMap(),
            1, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
        auto operands = amrex::MultiFab(s.photon_operands.boxArray(),
            s.photon_operands.DistributionMap(), 4, 0,
            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
        host.setVal(0); counts.setVal(0); operands.setVal(0);
        for (std::size_t p = 0; p < weight.size(); ++p) {
            amrex::IntVect cell;
            auto const photon = PacketConversionFourMomentumFromPhoton(weight[p],
                momentum[0][p], momentum[1][p], momentum[2][p]);
            valid &= photon.valid && ValidFiniteMassPhoton(photon.value) &&
                photon_replay_detail::NearestCell(s.geometry, plan.x[p], plan.y[p], plan.z[p], cell);
            bool deposited = false;
            if (valid) {
                for (amrex::MFIter it(host); it.isValid(); ++it) {
                    if (!it.validbox().contains(cell)) { continue; }
                    for (int d = 0; d < 4; ++d) {
                        host[it](cell, d) -= photon.value[d];
                        operands[it](cell, d) += std::abs(photon.value[d]);
                    }
                    counts[it](cell, 0) += 1;
                    deposited = true; break;
                }
            }
            valid &= deposited;
        }
        amrex::htod_memcpy(s.photon_debit, host);
        amrex::htod_memcpy(s.photon_count, counts);
        amrex::htod_memcpy(s.photon_operands, operands);
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { return s.reject(PacketThermalFailure::InvalidBirths); }
    candidate.Redistribute();
    return FinishThermal(s, PacketThermalEvent::Emission);
}

bool
PacketThermalExchangeTransaction::ValidateLive () const
{
    auto const& s = *m_impl;
    bool valid = s.report.valid && !s.committed;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { return false; }
    return s.baselineUnchanged();
}

bool
PacketThermalExchangeTransaction::Commit ()
{
    auto& s = *m_impl;
    if (!ValidateLive()) { return false; }
    if (!s.photon_trial->ValidateLive()) { return false; }
    s.material_stage->Commit(); // Only the private material clone changes here.
    if (!s.material_trial->Commit()) { return false; }
    s.hybrid->CommitElectronTemperature(0, s.temperature);
    s.photon_trial->Commit();
    s.committed = true;
    return true;
}

amrex::MultiFab const& PacketThermalExchangeTransaction::CandidateTemperature () const
{ return m_impl->temperature; }
amrex::MultiFab const& PacketThermalExchangeTransaction::PhotonDebit () const
{ return m_impl->photon_debit; }
amrex::MultiFab const& PacketThermalExchangeTransaction::PhotonOperandScale () const
{ return m_impl->photon_operands; }
amrex::MultiFab const& PacketThermalExchangeTransaction::RequestedHeat () const
{ return m_impl->requested_heat; }
amrex::MultiFab const& PacketThermalExchangeTransaction::ActualHeat () const
{ return m_impl->actual_heat; }
amrex::MultiFab const& PacketThermalExchangeTransaction::ThermalRedistribution () const
{ return m_impl->redistribution; }
ParticleImpulseTransaction const& PacketThermalExchangeTransaction::MaterialStage () const
{ return *m_impl->material_stage; }
} // namespace warpx::radiation
