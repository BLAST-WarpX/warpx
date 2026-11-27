/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NATIVERZTHERMALHISTORYADMISSION_H_
#define WARPX_RADIATION_NATIVERZTHERMALHISTORYADMISSION_H_

#include "RZAngularGeometry.H"
#include "Utils/WarpXConst.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_iMultiFab.H>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>

class HybridPICModel;
class WarpXParticleContainer;

namespace warpx::radiation
{
class ParticleImpulseMaterial;
class PhotonPacketTransaction;
struct NativeResolvedRZThermalHistoryOptions;
struct NativeResolvedRZThermalHistoryResult;

enum class NativeRZThermalAbsorptionDepthPolicy : int
{
    WholeOuter16 = 0,
    PureAbsorptionActualStep16 = 1,
    AllDNoPhotonImplicitAbsorption = 2
};

/** Independent default-bounded packet duration policy. The opt-in is ONLY
 * an engine-witnessed empty-K implicit D step, not long photon transport. */
enum class NativeRZThermalPhotonTimePolicy : int
{
    PhotonHalfCell = 0,
    AllDNoPhotonImplicitStep = 1
};

/** Synchronous, immutable History-issued duration/clock authority. No caller
 * constructor, setter, copy or inferred dt is available. Never return/store
 * this context in a candidate owner or checkpoint. */
class NativeRZThermalHistoryStepContext
{
public:
    NativeRZThermalHistoryStepContext (NativeRZThermalHistoryStepContext const&) = delete;
    NativeRZThermalHistoryStepContext& operator= (
        NativeRZThermalHistoryStepContext const&) = delete;

    [[nodiscard]] amrex::Real dt () const noexcept { return m_dt; }
    [[nodiscard]] amrex::Real time () const noexcept { return m_time; }
    [[nodiscard]] amrex::Real end () const noexcept { return m_end; }
    [[nodiscard]] amrex::Real outerTime () const noexcept { return m_outer_time; }
    [[nodiscard]] amrex::Real outerEnd () const noexcept { return m_outer_end; }
    [[nodiscard]] amrex::Real outerDt () const noexcept { return m_outer_dt; }
    [[nodiscard]] int index () const noexcept { return m_index; }
    [[nodiscard]] int count () const noexcept { return m_count; }
    [[nodiscard]] std::uint64_t seed () const noexcept { return m_seed; }
    [[nodiscard]] std::uint64_t interval () const noexcept { return m_interval; }

private:
    NativeRZThermalHistoryStepContext (
        amrex::Real time, amrex::Real end, amrex::Real dt,
        amrex::Real outer_time, amrex::Real outer_end, amrex::Real outer_dt,
        int index, int count, std::uint64_t seed, std::uint64_t interval)
        : m_time(time), m_end(end), m_dt(dt), m_outer_time(outer_time),
          m_outer_end(outer_end), m_outer_dt(outer_dt), m_index(index), m_count(count),
          m_seed(seed), m_interval(interval)
    {}

    friend NativeResolvedRZThermalHistoryResult TryAdvanceNativeResolvedRZThermalHistory (
        WarpXParticleContainer&, PhotonPacketTransaction const&, ParticleImpulseMaterial const&,
        std::string const&, amrex::MultiFab const&, amrex::iMultiFab const&,
        amrex::MultiFab const&, HybridPICModel const&, amrex::Geometry const&,
        amrex::Real, amrex::Real,
        std::function<bool(amrex::Real, amrex::MultiFab const&, amrex::MultiFab const&,
                           amrex::MultiFab&, amrex::MultiFab&, std::string&)> const&,
        NativeResolvedRZThermalHistoryOptions const&);

    amrex::Real const m_time, m_end, m_dt, m_outer_time, m_outer_end, m_outer_dt;
    int const m_index, m_count;
    std::uint64_t const m_seed, m_interval;
};

/** Read-only structural preflight, NOT absorption-depth authority. Count every
 * stored native record, including invalid IDs and zero represented accounts.
 * Canonical internal/periodic mask halos are D; physical radial ghosts are0. */
[[nodiscard]] bool ValidateNativeRZAllDNoPhotonState (
    WarpXParticleContainer const& live, PhotonPacketTransaction const& snapshot,
    amrex::iMultiFab const& mask, amrex::Geometry const& geometry, std::string& reason);

/** Immutable synchronous authority issued ONLY by History. Owns the canonical
 * consumed mask and an exact original-mask snapshot; binds native live/start
 * provenance and the enclosing clock namespace. Never checkpoint or return it. */
class NativeRZThermalAllDNoKEventWitness
{
public:
    ~NativeRZThermalAllDNoKEventWitness () = default;
    NativeRZThermalAllDNoKEventWitness (NativeRZThermalAllDNoKEventWitness const&) = delete;
    NativeRZThermalAllDNoKEventWitness& operator= (
        NativeRZThermalAllDNoKEventWitness const&) = delete;

    [[nodiscard]] amrex::iMultiFab const& CanonicalMask () const noexcept
    { return *m_canonical_mask; }
    [[nodiscard]] bool ValidateOwners (
        PhotonPacketTransaction const& snapshot, amrex::iMultiFab const& consumed_mask,
        std::string& reason) const;
    [[nodiscard]] bool ValidateStep (
        NativeRZThermalHistoryStepContext const& context, amrex::Real dt,
        std::uint64_t seed, std::uint64_t interval, PhotonPacketTransaction const& snapshot,
        amrex::iMultiFab const& consumed_mask, std::string& reason) const;
    /** Extended duration is valid only for this History's actual context,
     * original live owner, complete empty transaction and exact geometry.
     * This read-only collective validates authority; it never issues it. */
    [[nodiscard]] bool ValidatePhotonDuration (
        NativeRZThermalHistoryStepContext const& context, amrex::Real dt,
        std::uint64_t seed, std::uint64_t interval, WarpXParticleContainer const& live,
        PhotonPacketTransaction const& snapshot, amrex::iMultiFab const& consumed_mask,
        amrex::Geometry const& geometry, std::string& reason) const;

private:
    NativeRZThermalAllDNoKEventWitness (
        WarpXParticleContainer const& live, PhotonPacketTransaction const& start,
        amrex::iMultiFab const& mask, amrex::Geometry const& geometry,
        std::uint64_t seed, std::uint64_t interval, NativeRZThermalPhotonTimePolicy time_policy,
        amrex::Real outer_time, amrex::Real outer_end, amrex::Real outer_dt, int count);
    friend NativeResolvedRZThermalHistoryResult TryAdvanceNativeResolvedRZThermalHistory (
        WarpXParticleContainer&, PhotonPacketTransaction const&, ParticleImpulseMaterial const&,
        std::string const&, amrex::MultiFab const&, amrex::iMultiFab const&,
        amrex::MultiFab const&, HybridPICModel const&, amrex::Geometry const&,
        amrex::Real, amrex::Real,
        std::function<bool(amrex::Real, amrex::MultiFab const&, amrex::MultiFab const&,
                           amrex::MultiFab&, amrex::MultiFab&, std::string&)> const&,
        NativeResolvedRZThermalHistoryOptions const&);

    WarpXParticleContainer const* const m_live;
    PhotonPacketTransaction const* const m_start;
    amrex::iMultiFab const* const m_original_mask;
    std::unique_ptr<amrex::iMultiFab const> const m_mask_snapshot, m_canonical_mask;
    amrex::Geometry const m_geometry;
    std::uint64_t const m_seed, m_interval;
    amrex::Long const m_allocator;
    NativeRZThermalPhotonTimePolicy const m_time_policy;
    amrex::Real const m_outer_time, m_outer_end, m_outer_dt;
    int const m_count;
};

/** Bind the actual leaf argument to the engine's stored dt bits and current
 * outer namespace, collectively including empty ranks. No endpoint subtraction
 * or quotient is used to reconstruct the authoritative duration. */
[[nodiscard]] inline bool ValidateNativeRZThermalHistoryStepContext (
    NativeRZThermalHistoryStepContext const& context, amrex::Real dt,
    std::uint64_t seed, std::uint64_t interval, std::string& reason)
{
    std::array<amrex::Real, 7> const local{
        context.time(), context.end(), context.dt(), context.outerTime(),
        context.outerEnd(), context.outerDt(), dt};
    auto reference = local;
    auto const root = amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::Bcast(reference.data(), reference.size(), root);
    std::array<std::uint64_t, 4> const clocks{context.seed(), context.interval(), seed, interval};
    auto clock_reference = clocks;
    amrex::ParallelDescriptor::Bcast(clock_reference.data(), clock_reference.size(), root);
    std::array<int, 2> const indices{context.index(), context.count()};
    auto index_reference = indices;
    amrex::ParallelDescriptor::Bcast(index_reference.data(), index_reference.size(), root);
    auto const actual_dt = context.dt();
    bool valid = std::memcmp(local.data(), reference.data(),
                             local.size() * sizeof(amrex::Real)) == 0 &&
        clocks == clock_reference && indices == index_reference &&
        std::memcmp(&actual_dt, &dt, sizeof(dt)) == 0 &&
        context.seed() == seed && context.interval() == interval &&
        std::isfinite(context.time()) && std::isfinite(context.end()) &&
        std::isfinite(context.outerTime()) && std::isfinite(context.outerEnd()) &&
        std::isfinite(context.outerDt()) && context.outerDt() > 0 &&
        std::isfinite(dt) && dt > 0 && context.outerTime() >= 0 &&
        context.time() >= context.outerTime() && context.end() > context.time() &&
        context.end() <= context.outerEnd() && context.outerEnd() > context.outerTime() &&
        context.index() >= 0 && context.index() < context.count() &&
        context.count() > 0 && context.count() <= 32;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        reason = "invalid authenticated actual history step duration or namespace";
    }
    return valid;
}

/** CPU all-cell/rank pure-absorption envelope. Retains the original radial
 * tangential-velocity bound and depth expression, with ONLY History's actual
 * stored step dt as duration. Actual ray, D and native material guards remain
 * independent and unchanged; this is not depth<=16*N admission. */
[[nodiscard]] inline bool ValidatePureAbsorptionHistoryStep (
    NativeRZThermalHistoryStepContext const& context,
    amrex::MultiFab const& scattering, amrex::MultiFab const& absorption,
    amrex::MultiFab const& rate, amrex::Geometry const& geometry,
    amrex::Real depth_limit, std::string& reason)
{
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU)
    amrex::ignore_unused(context, scattering, absorption, rate, geometry, depth_limit);
    reason = "pure absorption actual-step depth requires CPU RZ";
    return false;
#else
    bool valid = scattering.nComp() == 1 && absorption.nComp() == 1 && rate.nComp() == 3 &&
        scattering.ixType().cellCentered() && absorption.ixType().cellCentered() &&
        rate.ixType().cellCentered() && scattering.boxArray() == absorption.boxArray() &&
        scattering.boxArray() == rate.boxArray() &&
        scattering.DistributionMap() == absorption.DistributionMap() &&
        scattering.DistributionMap() == rate.DistributionMap() &&
        std::isfinite(depth_limit) && depth_limit > 0 && depth_limit <= 16;
    auto minimum_limit = depth_limit, maximum_limit = depth_limit;
    amrex::ParallelDescriptor::ReduceRealMin(minimum_limit);
    amrex::ParallelDescriptor::ReduceRealMax(maximum_limit);
    valid = valid && minimum_limit == maximum_limit;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        reason = "invalid pure absorption history coefficient fields or limit";
        return false;
    }
    auto const radial_lo = geometry.ProbLo(0);
    auto const dr = geometry.CellSize(0);
    int const radial_first = geometry.Domain().smallEnd(0);
    for (amrex::MFIter it(scattering); it.isValid(); ++it)
    {
        auto const chi = scattering.const_array(it), alpha = absorption.const_array(it);
        auto const velocity = rate.const_array(it);
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            auto const s = chi(index), a = alpha(index);
            amrex::Real beta_squared = 0;
            for (int d = 0; d < 3; ++d)
            {
                auto value = velocity(index, d);
                if (d == 1)
                {
                    auto const lower = radial_lo + (index[0] - radial_first) * dr;
                    value *= (lower + dr) / RZAngularGeometry::MeanRadius(lower, lower + dr);
                }
                beta_squared += value * value / PhysConst::c2;
            }
            bool const cell_valid = s == 0 && !std::signbit(s) && a > 0 && std::isfinite(a) &&
                std::isfinite(beta_squared) && beta_squared >= 0 && beta_squared < 1;
            valid = valid && cell_valid;
            if (!cell_valid) { continue; }
            auto const beta = std::sqrt(beta_squared);
            auto const maximum_doppler = (1 + beta) / std::sqrt(1 - beta_squared);
            auto const absorption_depth = a * PhysConst::c * context.dt() * maximum_doppler;
            valid = valid && std::isfinite(absorption_depth) && absorption_depth >= 0 &&
                absorption_depth <= depth_limit;
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        reason = "pure absorption actual history step exceeds its all-cell envelope";
    }
    return valid;
#endif
}

/** Witnessed all-D absorption has no packet exponential/event set. Keep the
 * original finite Dmax/depth expressions and WHOLE-outer scattering16 in every
 * actual frame; omit only absorption<=16. Actual D/work/caloric guards remain. */
[[nodiscard]] inline bool ValidateAllDImplicitAbsorptionHistoryStep (
    NativeRZThermalHistoryStepContext const& context,
    NativeRZThermalAllDNoKEventWitness const& witness,
    amrex::MultiFab const& scattering, amrex::MultiFab const& absorption,
    amrex::MultiFab const& rate, amrex::Geometry const& geometry,
    amrex::Real scattering_limit, amrex::Real absorption_limit, std::string& reason)
{
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU)
    amrex::ignore_unused(context, witness, scattering, absorption, rate, geometry,
                        scattering_limit, absorption_limit);
    reason = "all-D implicit absorption requires CPU RZ";
    return false;
#else
    auto const& mask = witness.CanonicalMask();
    bool valid = scattering.nComp() == 1 && absorption.nComp() == 1 && rate.nComp() == 3 &&
        scattering.ixType().cellCentered() && absorption.ixType().cellCentered() &&
        rate.ixType().cellCentered() && scattering.boxArray() == mask.boxArray() &&
        absorption.boxArray() == mask.boxArray() && rate.boxArray() == mask.boxArray() &&
        scattering.DistributionMap() == mask.DistributionMap() &&
        absorption.DistributionMap() == mask.DistributionMap() &&
        rate.DistributionMap() == mask.DistributionMap() &&
        std::isfinite(scattering_limit) && scattering_limit > 0 && scattering_limit <= 16 &&
        std::isfinite(absorption_limit) && absorption_limit > 0 && absorption_limit <= 16;
    std::array<amrex::Real, 2> limits{scattering_limit, absorption_limit}, maximum = limits;
    amrex::ParallelDescriptor::ReduceRealMin(limits.data(), limits.size());
    amrex::ParallelDescriptor::ReduceRealMax(maximum.data(), maximum.size());
    valid = valid && limits == maximum;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        reason = "invalid all-D implicit absorption fields or unchanged event limits";
        return false;
    }
    auto const radial_lo = geometry.ProbLo(0), dr = geometry.CellSize(0);
    int const radial_first = geometry.Domain().smallEnd(0);
    for (amrex::MFIter it(scattering); it.isValid(); ++it)
    {
        auto const chi = scattering.const_array(it), alpha = absorption.const_array(it);
        auto const velocity = rate.const_array(it);
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            auto const s = chi(index), a = alpha(index);
            amrex::Real beta_squared = 0;
            for (int d = 0; d < 3; ++d)
            {
                auto value = velocity(index, d);
                if (d == 1)
                {
                    auto const lower = radial_lo + (index[0] - radial_first) * dr;
                    value *= (lower + dr) / RZAngularGeometry::MeanRadius(lower, lower + dr);
                }
                beta_squared += value * value / PhysConst::c2;
            }
            bool const cell_valid = s >= 0 && a >= 0 && std::isfinite(s + a) &&
                std::isfinite(beta_squared) && beta_squared >= 0 && beta_squared <= 1.e-6;
            valid = valid && cell_valid;
            if (!cell_valid) { continue; }
            auto const beta = std::sqrt(beta_squared);
            auto const maximum_doppler = (1 + beta) / std::sqrt(1 - beta_squared);
            auto const scattering_depth = s * PhysConst::c * context.outerDt() * maximum_doppler;
            auto const absorption_depth = a * PhysConst::c * context.outerDt() * maximum_doppler;
            auto const step_depth = a * PhysConst::c * context.dt() * maximum_doppler;
            valid = valid && std::isfinite(scattering_depth + absorption_depth) &&
                scattering_depth >= 0 && scattering_depth <= scattering_limit &&
                absorption_depth >= 0 && std::isfinite(step_depth) && step_depth >= 0;
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { reason = "all-D implicit absorption finite frame or scattering16 rejected"; }
    return valid;
#endif
}
} // namespace warpx::radiation
#endif
