/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_PACKETTHERMALEXCHANGE_H_
#define WARPX_RADIATION_PACKETTHERMALEXCHANGE_H_

#include "FiniteMassPhotonCollision.H"

#include <AMReX_BaseFwd.H>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class HybridPICModel;
class MultiParticleContainer;
class WarpXParticleContainer;

namespace warpx::radiation
{
namespace detail { struct EmissionPlan; }
class ParticleImpulseTransaction;

enum class PacketThermalEvent { Absorption, Emission };

/** Signed radiation debit [E_before-E_after,c*(P_before-P_after)], in J.
 * Construct from actually represented native photons, never a desired bath
 * energy or an assumed isotropic momentum inventory. */
struct PacketThermalExchange
{
    FourVector radiation_debit{};
    amrex::GpuArray<amrex::Real, 3> material_impulse{};
    PacketThermalEvent event = PacketThermalEvent::Absorption;
    bool valid = false;
};

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE PacketThermalExchange
EvaluatePacketThermalAbsorption (FourVector const& incoming) noexcept
{
    PacketThermalExchange result;
    if (!ValidFiniteMassPhoton(incoming)) { return result; }
    result.radiation_debit = incoming;
    for (int d = 0; d < 3; ++d) {
        result.material_impulse[d] = incoming[d + 1] / PhysConst::c;
    }
    result.valid = true;
    return result;
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE PacketThermalExchange
EvaluatePacketThermalEmission (FourVector const& outgoing) noexcept
{
    auto result = EvaluatePacketThermalAbsorption(outgoing);
    result.event = PacketThermalEvent::Emission;
    if (!result.valid) { return result; }
    for (auto& value : result.radiation_debit) { value = -value; }
    for (auto& value : result.material_impulse) { value = -value; }
    return result;
}

struct PacketThermalPartition
{
    amrex::Real requested_heat = 0;
    amrex::Real energy_operand_scale = 0;
    bool valid = false;
};

/** Actual finite native ion work and its independently measured numerical
 * energy-carry change are both material energy accounts. Subtract both before
 * prescribing electron heat. Negative absorption heat is rejected, without
 * clipping; an emission source may have either sign if prior carry is realized.
 * Low-beta and native impulse/carry conservation are separate stage guards. */
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE PacketThermalPartition
EvaluatePacketThermalPartition (PacketThermalEvent event, amrex::Real photon_debit,
                               amrex::Real actual_work,
                               amrex::Real energy_carry_change) noexcept
{
    PacketThermalPartition result;
    if (!amrex::Math::isfinite(photon_debit) || !amrex::Math::isfinite(actual_work) ||
        !amrex::Math::isfinite(energy_carry_change) ||
        (event == PacketThermalEvent::Absorption && photon_debit < 0) ||
        (event == PacketThermalEvent::Emission && photon_debit > 0)) {
        return result;
    }
    result.requested_heat = photon_debit - (actual_work + energy_carry_change);
    result.energy_operand_scale = std::abs(photon_debit) + std::abs(actual_work) +
                                  std::abs(energy_carry_change) +
                                  std::abs(result.requested_heat);
    result.valid = amrex::Math::isfinite(result.requested_heat) &&
                   amrex::Math::isfinite(result.energy_operand_scale) &&
                   (event != PacketThermalEvent::Absorption || result.requested_heat >= 0);
    return result;
}

enum class PacketThermalFailure {
    None, UnsupportedState, InvalidPhotonSelection, InvalidBirths, MaterialStage,
    LowBeta, NegativeAbsorptionHeat, CaloricResponse, Conservation, LiveChanged
};

struct PacketThermalReport
{
    bool valid = false;
    PacketThermalFailure failure = PacketThermalFailure::None;
    amrex::Real photon_debit = 0;
    amrex::Real actual_ion_work = 0;
    amrex::Real energy_carry_change = 0;
    amrex::Real requested_heat = 0;
    amrex::Real actual_heat = 0;
    amrex::Real energy_residual = 0;
    amrex::Real energy_roundoff_bound = 0;
    amrex::GpuArray<amrex::Real, 3> momentum_residual{};
};

/** One privately owned native absorption or emission trial.
 *
 * Owns photons/births, material impulse/carry and native nodal caloric candidate.
 * Stage uses frozen particle identities and actual stored packet momenta. The
 * hybrid source map retains its conservative shared-node spatial allocation;
 * ActualHeat-RequestedHeat is reported separately as thermal redistribution.
 * Independent EOS corner quadrature validates actual electron heat.
 *
 * Initial qualification: Cartesian periodic level zero, double field/particle
 * precision, fixed-charge material, ideal or builtin latent electrons with no
 * table/composition, nearest-cell recoil, and every represented ion before and
 * after the finite kick at beta<=0.01. No transport, RNG, charge changes or bath
 * are owned here. Emission consumes already prescribed ParticleReal births;
 * it does not claim to sample a moving thermal spectrum or emissivity.
 *
 * All ranks participate, including ranks with no selected photons/births.
 * Exactly one Stage method may be called. Rejection/destruction discards all
 * private candidates and consumes no native birth IDs. Commit rechecks live
 * particles and density/temperature snapshots before publishing exact accepted
 * state. No concurrent live mutation is permitted between preflight and commit.
 */
class PacketThermalExchangeTransaction
{
public:
    PacketThermalExchangeTransaction (
        MultiParticleContainer& particles, std::vector<std::string> const& material_species,
        std::string const& carry_path, WarpXParticleContainer& photons,
        HybridPICModel const& hybrid, amrex::Real minimum_electron_density = 1);
    ~PacketThermalExchangeTransaction ();
    PacketThermalExchangeTransaction (PacketThermalExchangeTransaction const&) = delete;
    PacketThermalExchangeTransaction& operator=(PacketThermalExchangeTransaction const&) = delete;

    /** Full absorption of selected resident native (id,cpu) keys. Local keys
     * must be unique and each must name one photon currently owned by this rank. */
    [[nodiscard]] PacketThermalReport StageAbsorption (
        std::vector<std::uint64_t> const& local_idcpu);
    /** Stage one native birth batch; no logical-key/default RNG initialization
     * is supplied by this initial primitive. Rejection retains no proposed birth. */
    [[nodiscard]] PacketThermalReport StageEmission (detail::EmissionPlan const& plan);
    [[nodiscard]] bool ValidateLive () const;
    [[nodiscard]] bool Commit ();
    [[nodiscard]] amrex::MultiFab const& CandidateTemperature () const;
    [[nodiscard]] amrex::MultiFab const& PhotonDebit () const;
    [[nodiscard]] amrex::MultiFab const& PhotonOperandScale () const;
    [[nodiscard]] amrex::MultiFab const& RequestedHeat () const;
    [[nodiscard]] amrex::MultiFab const& ActualHeat () const;
    [[nodiscard]] amrex::MultiFab const& ThermalRedistribution () const;
    [[nodiscard]] ParticleImpulseTransaction const& MaterialStage () const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace warpx::radiation

#endif
