/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativeResolvedRZThermalHistory.H"

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "NativePhotonCarryRouting.H"
#include "NativePhotonReplayInternals.H"
#include "NativeRZIndependentGrayOpacitySlice.H"
#include "NativeRZOwnedNodalCaloric.H"
#include "PhotonLogicalDrawKey.H"
#include "PhotonPacketTransaction.H"
#include "WarpX.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <utility>

namespace warpx::radiation
{
namespace
{
#if defined(WARPX_DIM_RZ)
bool
Collective (bool value)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(value);
    return value;
}

amrex::MultiFab
Copy (amrex::MultiFab const& input)
{
    amrex::MultiFab copy(input.boxArray(), input.DistributionMap(),
                         input.nComp(), input.nGrowVect());
    amrex::MultiFab::Copy(copy, input, 0, 0, input.nComp(), input.nGrowVect());
    return copy;
}

amrex::MultiFab
Host (amrex::MultiFab const& input)
{
    amrex::MultiFab copy(input.boxArray(), input.DistributionMap(),
                         input.nComp(), input.nGrowVect(),
                         amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::dtoh_memcpy(copy, input);
    amrex::Gpu::streamSynchronize();
    return copy;
}

void
Sum (long double* values, int count)
{
#ifdef AMREX_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, values, count, MPI_LONG_DOUBLE, MPI_SUM,
                  amrex::ParallelDescriptor::Communicator());
#else
    amrex::ignore_unused(values, count);
#endif
}

#if !defined(AMREX_USE_GPU)
bool
HistoryCensusSchema (WarpXParticleContainer::Base const& photons,
                     CollidedDiffusionAdmissionComponents const& census)
{
    std::array<char const*, 3> const names{
        "radiation_diffusion_census_ux0", "radiation_diffusion_census_uy0",
        "radiation_diffusion_census_uz0"};
    auto const& registered = photons.GetRealSoANames();
    bool valid = photons.finestLevel() == 0;
    for (int d = 0; d < 3; ++d)
    {
        int minimum = census.initial_u[d], maximum = minimum;
        amrex::ParallelDescriptor::ReduceIntMin(minimum);
        amrex::ParallelDescriptor::ReduceIntMax(maximum);
        bool const in_range = minimum >= PIdx::nattribs &&
            minimum < static_cast<int>(registered.size());
        valid = valid && minimum == maximum && in_range;
        if (in_range)
        {
            valid = valid && registered[minimum] == names[d];
        }
    }
    return Collective(valid);
}

bool
InitializeAcceptedHistoryBirthCensus (
    WarpXParticleContainer::Base& photons, WarpXParticleContainer const& live,
    CollidedDiffusionAdmissionComponents const& census,
    std::vector<NativePhotonPathCarry> const& clocks,
    std::vector<std::uint64_t> const& owner_birth_keys)
{
    std::vector<std::uint64_t> ids, keys, draws;
    bool valid = photon_replay_detail::CapturePhotonRoutingKeys(
        photons, live, NativePhotonPathSnapshotPolicy::KeyedProvisional,
        ids, keys, draws);
    valid = valid && ids.size() == clocks.size();
    if (!Collective(valid)) { return false; }
    std::map<std::uint64_t, std::size_t> marked;
    for (std::size_t n = 0; n < clocks.size(); ++n)
    {
        valid = valid && clocks[n].idcpu == ids[n] &&
            clocks[n].provisional_draw_key == keys[n];
        if (keys[n] != 0)
        {
            valid = marked.emplace(keys[n], n).second && valid;
        }
    }
    if (!Collective(valid)) { return false; }
    std::vector<bool> newborn(clocks.size(), false);
    for (auto const key : owner_birth_keys)
    {
        auto const found = marked.find(key);
        valid = valid && key != 0 && found != marked.end();
        if (found != marked.end())
        {
            auto const n = found->second;
            valid = valid && !newborn[n] && clocks[n].completed_events == 0 &&
                clocks[n].next_draw == 1;
            newborn[n] = true;
        }
    }
    if (!Collective(valid)) { return false; }
    auto const captured = photon_replay_detail::CaptureParticles(photons);
    std::size_t count = 0;
    for (auto const& tile : captured.tiles)
    {
        valid = valid && tile.count >= 0 && static_cast<std::size_t>(tile.count) <=
            std::numeric_limits<std::size_t>::max() - count;
        if (valid) { count += static_cast<std::size_t>(tile.count); }
    }
    if (!Collective(valid && count == newborn.size())) { return false; }
    std::size_t offset = 0;
    for (auto const& tile : captured.tiles)
    {
        for (long p = 0; p < tile.count; ++p, ++offset)
        {
            for (int d = 0; d < 3; ++d)
            {
                auto const component = newborn[offset] ? PIdx::ux + d : census.initial_u[d];
                valid = valid && std::isfinite(tile.real_data[component * tile.count + p]);
            }
        }
    }
    if (!Collective(valid && offset == newborn.size())) { return false; }
    offset = 0;
    for (auto const& tile : captured.tiles)
    {
        auto& soa = photons.GetParticles(0).at(tile.key).GetStructOfArrays();
        for (int d = 0; d < 3; ++d)
        {
            for (long p = 0; p < tile.count; ++p)
            {
                if (newborn[offset + static_cast<std::size_t>(p)])
                {
                    soa.GetRealData(census.initial_u[d])[p] =
                        tile.real_data[(PIdx::ux + d) * tile.count + p];
                }
            }
        }
        offset += static_cast<std::size_t>(tile.count);
    }
    return true;
}
#endif

struct RadiationEndpoint
{
    std::array<long double, 3> inventory{};
    std::array<long double, 3> operands{};
    amrex::Long records = 0;
    std::array<long double, 3> diffusion_operands{};
    amrex::Long diffusion_records = 0;
};

RadiationEndpoint
Radiation (WarpXParticleContainer::Base const& photons,
           amrex::MultiFab const& diffusion, amrex::Geometry const& geometry)
{
    RadiationEndpoint result;
    for (auto const& tile :
         photon_replay_detail::CaptureParticles(photons).tiles)
    {
        for (long p = 0; p < tile.count; ++p)
        {
            if (!amrex::ConstParticleIDWrapper(tile.idcpu[p]).is_valid())
            {
                continue;
            }
            auto get = [&] (int d) -> long double
            { return tile.real_data[d * tile.count + p]; };
            auto const factor = get(PIdx::w) * PhysConst::m_e * PhysConst::c;
            auto const x = get(PIdx::ux), y = get(PIdx::uy), z = get(PIdx::uz);
            auto const radius = get(PIdx::r), theta = get(PIdx::theta);
            std::array<long double, 3> value{
                factor * std::sqrt(x * x + y * y + z * z), factor * z,
                factor * radius * (std::cos(theta) * y - std::sin(theta) * x)};
            for (int d = 0; d < 3; ++d)
            {
                result.inventory[d] += value[d];
                result.operands[d] += std::abs(value[d]);
            }
            ++result.records;
        }
    }
    auto const cells = Host(diffusion);
    for (amrex::MFIter it(cells); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const lo = static_cast<long double>(geometry.ProbLo(0)) +
                            (cell()[0] - geometry.Domain().smallEnd(0)) *
                                static_cast<long double>(geometry.CellSize(0));
            auto const hi = lo + geometry.CellSize(0);
            auto const mean =
                (2.L / 3) * (lo * lo + lo * hi + hi * hi) / (lo + hi);
            std::array<long double, 3> value{cells[it](cell(), 0),
                                             cells[it](cell(), 3),
                                             mean * cells[it](cell(), 2)};
            for (int d = 0; d < 3; ++d)
            {
                result.inventory[d] += value[d];
                result.operands[d] += std::abs(value[d]);
                result.diffusion_operands[d] += std::abs(value[d]);
            }
            ++result.records;
            ++result.diffusion_records;
        }
    }
    Sum(result.inventory.data(), 3);
    Sum(result.operands.data(), 3);
    Sum(result.diffusion_operands.data(), 3);
    amrex::ParallelDescriptor::ReduceLongSum(result.records);
    amrex::ParallelDescriptor::ReduceLongSum(result.diffusion_records);
    return result;
}

struct MaterialAnchor
{
    std::string name;
    amrex::Real mass = 0;
    std::array<int, 4> carry{};
    photon_replay_detail::CapturedParticles particles;
};

std::vector<MaterialAnchor>
CaptureMaterial (ParticleImpulseMaterial const& material,
                 std::string const& path)
{
    std::vector<MaterialAnchor> anchors;
    for (auto const& view : material.CandidateSpecies())
    {
        MaterialAnchor anchor;
        anchor.name = view.name;
        anchor.mass = view.rest_mass;
        auto const& names = view.particles->GetRealSoANames();
        for (int d = 0; d < 4; ++d)
        {
            auto const name = "radiation_impulse_" + path +
                              (d == 3 ? "_work" : std::string("_u") + "xyz"[d]);
            auto const found = std::find(names.begin(), names.end(), name);
            anchor.carry[d] = found == names.end()
                                  ? -1
                                  : static_cast<int>(found - names.begin());
        }
        anchor.particles =
            photon_replay_detail::CaptureParticles(*view.particles);
        anchors.push_back(std::move(anchor));
    }
    return anchors;
}

struct MaterialChange
{
    std::array<long double, 3> physical{};
    std::array<long double, 3> carry{};
    std::array<long double, 3> operands{};
    amrex::Long records = 0;
    bool valid = true;
};

MaterialChange
MeasureMaterial (std::vector<MaterialAnchor> const& anchors,
                 ParticleImpulseMaterial const& material)
{
    MaterialChange result;
    for (auto const& anchor : anchors)
    {
        auto const* species = material.CandidateSpecies(anchor.name);
        result.valid = result.valid && species != nullptr && anchor.mass > 0 &&
                       std::all_of(anchor.carry.begin(), anchor.carry.end(),
                                   [] (int d) { return d >= 0; });
        if (!result.valid)
        {
            continue;
        }
        auto const current = photon_replay_detail::CaptureParticles(*species);
        result.valid = current.tiles.size() == anchor.particles.tiles.size();
        if (!result.valid)
        {
            continue;
        }
        for (std::size_t n = 0; n < current.tiles.size(); ++n)
        {
            auto const& before = anchor.particles.tiles[n];
            auto const& after = current.tiles[n];
            result.valid =
                result.valid && before.key == after.key &&
                before.count == after.count &&
                photon_replay_detail::SameBits(before.idcpu, after.idcpu);
            if (!result.valid)
            {
                continue;
            }
            for (long p = 0; p < before.count; ++p)
            {
                auto old = [&] (int d) -> long double
                { return before.real_data[d * before.count + p]; };
                auto next = [&] (int d) -> long double
                { return after.real_data[d * after.count + p]; };
                result.valid = result.valid && old(PIdx::w) == next(PIdx::w) &&
                               old(PIdx::r) == next(PIdx::r) &&
                               old(PIdx::theta) == next(PIdx::theta) &&
                               old(PIdx::z) == next(PIdx::z);
                long double old_sq = 0, new_sq = 0, paired = 0;
                for (int d = 0; d < 3; ++d)
                {
                    auto const a = old(PIdx::ux + d), b = next(PIdx::ux + d);
                    old_sq += a * a;
                    new_sq += b * b;
                    paired += (b - a) * (b + a);
                }
                auto const old_gamma = std::sqrt(1 + old_sq / PhysConst::c2);
                auto const new_gamma = std::sqrt(1 + new_sq / PhysConst::c2);
                auto const mass = old(PIdx::w) * anchor.mass;
                auto const radius = old(PIdx::r),
                           c = std::cos(old(PIdx::theta)),
                           s = std::sin(old(PIdx::theta));
                auto const dx = next(PIdx::ux) - old(PIdx::ux),
                           dy = next(PIdx::uy) - old(PIdx::uy);
                auto const cx = next(anchor.carry[0]) - old(anchor.carry[0]);
                auto const cy = next(anchor.carry[1]) - old(anchor.carry[1]);
                result.physical[0] += mass * paired / (old_gamma + new_gamma);
                result.physical[1] +=
                    PhysConst::c * mass * (next(PIdx::uz) - old(PIdx::uz));
                result.physical[2] +=
                    PhysConst::c * mass * radius * (c * dy - s * dx);
                result.carry[0] +=
                    mass * (next(anchor.carry[3]) - old(anchor.carry[3]));
                result.carry[1] +=
                    PhysConst::c * mass *
                    (next(anchor.carry[2]) - old(anchor.carry[2]));
                result.carry[2] +=
                    PhysConst::c * mass * radius * (c * cy - s * cx);
                result.operands[0] += mass * (old_sq / (old_gamma + 1) +
                                              new_sq / (new_gamma + 1) +
                                              std::abs(next(anchor.carry[3])) +
                                              std::abs(old(anchor.carry[3])));
                result.operands[1] +=
                    PhysConst::c * mass *
                    (std::abs(old(PIdx::uz)) + std::abs(next(PIdx::uz)) +
                     std::abs(old(anchor.carry[2])) +
                     std::abs(next(anchor.carry[2])));
                result.operands[2] +=
                    PhysConst::c * mass * radius *
                    (std::abs(old(PIdx::ux)) + std::abs(old(PIdx::uy)) +
                     std::abs(next(PIdx::ux)) + std::abs(next(PIdx::uy)) +
                     std::abs(old(anchor.carry[0])) +
                     std::abs(old(anchor.carry[1])) +
                     std::abs(next(anchor.carry[0])) +
                     std::abs(next(anchor.carry[1])));
                ++result.records;
            }
        }
    }
    result.valid = Collective(result.valid);
    Sum(result.physical.data(), 3);
    Sum(result.carry.data(), 3);
    Sum(result.operands.data(), 3);
    amrex::ParallelDescriptor::ReduceLongSum(result.records);
    return result;
}
#endif
} // namespace

namespace
{
std::unique_ptr<amrex::iMultiFab const>
SnapshotAllDMask (amrex::iMultiFab const& mask, amrex::Geometry const& geometry,
                  bool canonical)
{
    auto owned = std::make_unique<amrex::iMultiFab>(mask.boxArray(), mask.DistributionMap(),
        1, canonical ? amrex::IntVect(1) : mask.nGrowVect());
    owned->setVal(0);
    amrex::iMultiFab::Copy(*owned, mask, 0, 0, 1,
                         canonical ? amrex::IntVect(0) : mask.nGrowVect());
    if (canonical) { owned->FillBoundary(geometry.periodicity()); }
    return owned;
}

#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU)
bool
AllDMaskBitsMatch (amrex::iMultiFab const& left, amrex::iMultiFab const& right)
{
    bool valid = left.nComp() == 1 && right.nComp() == 1 &&
        left.boxArray() == right.boxArray() && left.DistributionMap() == right.DistributionMap() &&
        left.nGrowVect() == right.nGrowVect() && left.ixType() == right.ixType();
    if (!valid) { return false; }
    for (amrex::MFIter it(left); it.isValid(); ++it)
    {
        auto const count = static_cast<std::size_t>(left[it].box().numPts());
        valid = std::memcmp(left[it].dataPtr(), right[it].dataPtr(), count * sizeof(int)) == 0 &&
            valid;
    }
    return valid;
}

bool
AllStoredPhotonRecordsEmpty (WarpXParticleContainer::Base const& photons)
{
    bool empty = photons.finestLevel() == 0;
    for (auto const& [key, tile] : photons.GetParticles(0))
    {
        amrex::ignore_unused(key);
        empty = tile.numParticles() == 0 && empty;
    }
    return empty;
}

bool
CanonicalAllDMaskValid (amrex::iMultiFab const& mask, amrex::Geometry const& geometry)
{
    if (mask.nComp() != 1 || mask.nGrowVect() != amrex::IntVect(1)) { return false; }
    bool valid = true;
    for (amrex::MFIter it(mask); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(mask[it].box()); cell.ok(); ++cell)
        {
            auto const radial = cell()[0];
            bool const wall_ghost = radial < geometry.Domain().smallEnd(0) ||
                                    radial > geometry.Domain().bigEnd(0);
            valid = mask[it](cell()) == (wall_ghost ? 0 : 1) && valid;
        }
    }
    return valid;
}
#endif
} // namespace

bool
ValidateNativeRZAllDNoPhotonState (
    WarpXParticleContainer const& live, PhotonPacketTransaction const& snapshot,
    amrex::iMultiFab const& mask, amrex::Geometry const& geometry, std::string& reason)
{
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU)
    amrex::ignore_unused(live, snapshot, mask, geometry);
    reason = "all-D no-photon state requires CPU RZ";
    return false;
#else
    bool const provenance = snapshot.ValidateSnapshotFor(live);
    bool valid = provenance && geometry.ProbLo(0) == 0 && !geometry.isPeriodic(0) &&
        geometry.isPeriodic(1) && mask.nComp() == 1 && mask.ixType().cellCentered() &&
        mask.boxArray() == live.ParticleBoxArray(0) &&
        mask.DistributionMap() == live.ParticleDistributionMap(0) &&
        mask.boxArray().isDisjoint() && mask.boxArray().contains(geometry.Domain()) &&
        mask.boxArray().numPts() == geometry.Domain().numPts() &&
        AllStoredPhotonRecordsEmpty(live) && AllStoredPhotonRecordsEmpty(snapshot.Particles());
    if (!Collective(valid))
    {
        reason = "all-D state has stored photons or invalid native mask provenance";
        return false;
    }
    auto canonical = SnapshotAllDMask(mask, geometry, true);
    valid = CanonicalAllDMaskValid(*canonical, geometry) && valid;
    if (!Collective(valid))
    {
        reason = "all-D canonical valid/internal/periodic mask or radial wall ghost rejected";
        return false;
    }
    return true;
#endif
}

NativeRZThermalAllDNoKEventWitness::NativeRZThermalAllDNoKEventWitness (
    WarpXParticleContainer const& live, PhotonPacketTransaction const& start,
    amrex::iMultiFab const& mask, amrex::Geometry const& geometry,
    std::uint64_t seed, std::uint64_t interval, NativeRZThermalPhotonTimePolicy time_policy,
    amrex::Real outer_time, amrex::Real outer_end, amrex::Real outer_dt, int count)
    : m_live(&live), m_start(&start), m_original_mask(&mask),
      m_mask_snapshot(SnapshotAllDMask(mask, geometry, false)),
      m_canonical_mask(SnapshotAllDMask(mask, geometry, true)), m_geometry(geometry),
      m_seed(seed), m_interval(interval),
      m_allocator(WarpXParticleContainer::ParticleType::the_next_id),
      m_time_policy(time_policy), m_outer_time(outer_time), m_outer_end(outer_end),
      m_outer_dt(outer_dt), m_count(count)
{}

bool
NativeRZThermalAllDNoKEventWitness::ValidateOwners (
    PhotonPacketTransaction const& snapshot, amrex::iMultiFab const& consumed_mask,
    std::string& reason) const
{
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU)
    amrex::ignore_unused(snapshot, consumed_mask);
    reason = "all-D no-K witness requires CPU RZ";
    return false;
#else
    bool const original = m_start->ValidateSnapshotFor(*m_live);
    bool const current = snapshot.ValidateSnapshotFor(*m_live);
    bool const empty = original && current && AllStoredPhotonRecordsEmpty(*m_live) &&
        AllStoredPhotonRecordsEmpty(m_start->Particles()) &&
        AllStoredPhotonRecordsEmpty(snapshot.Particles());
    bool const masks = AllDMaskBitsMatch(*m_original_mask, *m_mask_snapshot) &&
        AllDMaskBitsMatch(consumed_mask, *m_canonical_mask) &&
        CanonicalAllDMaskValid(*m_canonical_mask, m_geometry) &&
        CanonicalAllDMaskValid(consumed_mask, m_geometry);
    bool const allocator = WarpXParticleContainer::ParticleType::the_next_id == m_allocator;
    if (!Collective(empty && masks && allocator))
    {
        reason = "all-D no-K witness lost exact mask, empty owner or allocator baseline";
        return false;
    }
    return true;
#endif
}

bool
NativeRZThermalAllDNoKEventWitness::ValidateStep (
    NativeRZThermalHistoryStepContext const& context, amrex::Real dt,
    std::uint64_t seed, std::uint64_t interval, PhotonPacketTransaction const& snapshot,
    amrex::iMultiFab const& consumed_mask, std::string& reason) const
{
    bool const clock = ValidateNativeRZThermalHistoryStepContext(
        context, dt, seed, interval, reason);
    bool const owners = ValidateOwners(snapshot, consumed_mask, reason);
    bool valid = clock && owners && seed == m_seed && interval == m_interval;
    if (m_time_policy == NativeRZThermalPhotonTimePolicy::AllDNoPhotonImplicitStep)
    {
        std::array<amrex::Real, 3> const actual{
            context.outerTime(), context.outerEnd(), context.outerDt()};
        std::array<amrex::Real, 3> const issued{m_outer_time, m_outer_end, m_outer_dt};
        valid = valid && std::memcmp(actual.data(), issued.data(), sizeof(actual)) == 0 &&
            context.count() == m_count;
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { reason = "all-D no-K witness actual duration/namespace/owner binding rejected"; }
    return valid;
}

bool
NativeRZThermalAllDNoKEventWitness::ValidatePhotonDuration (
    NativeRZThermalHistoryStepContext const& context, amrex::Real dt,
    std::uint64_t seed, std::uint64_t interval, WarpXParticleContainer const& live,
    PhotonPacketTransaction const& snapshot, amrex::iMultiFab const& consumed_mask,
    amrex::Geometry const& geometry, std::string& reason) const
{
    bool const step = ValidateStep(context, dt, seed, interval, snapshot, consumed_mask, reason);
    // Step/owner failure is collective. Do not access a transaction which has
    // lost its readable uncommitted candidate after provenance rejection.
    if (!step) { return false; }
    bool valid = m_time_policy == NativeRZThermalPhotonTimePolicy::AllDNoPhotonImplicitStep &&
        &live == m_live;
    for (auto const* actual_geometry : {&geometry, &live.Geom(0), &snapshot.Particles().Geom(0)})
    {
        valid = valid && actual_geometry->Domain() == m_geometry.Domain() &&
            actual_geometry->Coord() == m_geometry.Coord();
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            std::array<amrex::Real, 3> const actual{
                actual_geometry->ProbLo(d), actual_geometry->ProbHi(d),
                actual_geometry->CellSize(d)};
            std::array<amrex::Real, 3> const issued{
                m_geometry.ProbLo(d), m_geometry.ProbHi(d), m_geometry.CellSize(d)};
            valid = valid && actual_geometry->isPeriodic(d) == m_geometry.isPeriodic(d) &&
                std::memcmp(actual.data(), issued.data(), sizeof(actual)) == 0;
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        reason = "all-D implicit photon duration policy/owner/geometry/context rejected";
    }
    return valid;
}

NativeResolvedRZThermalHistoryResult
TryAdvanceNativeResolvedRZThermalHistory (
    WarpXParticleContainer& live, PhotonPacketTransaction const& start,
    ParticleImpulseMaterial const& material_start, std::string const& path,
    amrex::MultiFab const& old_diffusion, amrex::iMultiFab const& mask,
    amrex::MultiFab const& old_temperature, HybridPICModel const& model,
    amrex::Geometry const& geometry, amrex::Real time, amrex::Real outer_dt,
    NativeRZThermalHistoryOpacityEvaluator const& evaluate,
    NativeResolvedRZThermalHistoryOptions const& options)
{
    NativeResolvedRZThermalHistoryResult rejected;
    rejected.reason = "unsupported private fixed-mask RZ thermal history";
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(live, start, material_start, path, old_diffusion, mask,
                         old_temperature, model, geometry, time, outer_dt,
                         evaluate, options);
    return rejected;
#else
    int minimum_policy = static_cast<int>(options.absorption_depth_policy);
    int maximum_policy = minimum_policy;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_policy);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_policy);
    if (!Collective(minimum_policy == maximum_policy &&
                    minimum_policy >= 0 && maximum_policy <= 2))
    {
        rejected.reason = "invalid or rank-dependent history absorption depth policy";
        return rejected;
    }
    bool const step_depth = options.absorption_depth_policy ==
        NativeRZThermalAbsorptionDepthPolicy::PureAbsorptionActualStep16;
    bool const all_d = options.absorption_depth_policy ==
        NativeRZThermalAbsorptionDepthPolicy::AllDNoPhotonImplicitAbsorption;
    bool const independent = options.independent_gray_table != nullptr;
    bool const independent_mixed = options.independent_gray_mixed;
    int independent_low = independent, independent_high = independent_low;
    amrex::ParallelDescriptor::ReduceIntMin(independent_low);
    amrex::ParallelDescriptor::ReduceIntMax(independent_high);
    if (!Collective(independent_low == independent_high))
    {
        rejected.reason = "independent-gray History policy differs between ranks";
        return rejected;
    }
    int mixed_low = independent_mixed, mixed_high = mixed_low;
    amrex::ParallelDescriptor::ReduceIntMin(mixed_low);
    amrex::ParallelDescriptor::ReduceIntMax(mixed_high);
    if (!Collective(mixed_low == mixed_high && (!independent_mixed || independent)))
    {
        rejected.reason = "independent mixed-opacity History policy differs or lacks a table";
        return rejected;
    }
    int vacuum_low = options.independent_gray_opacity.allow_exact_vacuum, vacuum_high = vacuum_low;
    amrex::ParallelDescriptor::ReduceIntMin(vacuum_low);
    amrex::ParallelDescriptor::ReduceIntMax(vacuum_high);
    if (!Collective(vacuum_low == vacuum_high &&
            (!vacuum_low || (independent_mixed &&
             options.interval.thermal.rz_single_selected_table_vacuum &&
             options.interval.field.single_selected_table_vacuum))))
    {
        rejected.reason = "independent exact-vacuum policy differs or lacks its mixed owners";
        return rejected;
    }
    int minimum_time_policy = static_cast<int>(options.photon_time_policy);
    int maximum_time_policy = minimum_time_policy;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_time_policy);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_time_policy);
    bool const implicit_step = options.photon_time_policy ==
        NativeRZThermalPhotonTimePolicy::AllDNoPhotonImplicitStep;
    if (!Collective(minimum_time_policy == maximum_time_policy &&
            minimum_time_policy >= 0 && maximum_time_policy <= 1 && (!implicit_step || all_d)))
    {
        rejected.reason = "invalid or rank-dependent history photon time policy";
        return rejected;
    }
    int minimum_census = options.retain_collided_census, maximum_census = minimum_census;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_census);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_census);
    if (!Collective(minimum_census == maximum_census))
    {
        rejected.reason = "private history census controls differ between ranks";
        return rejected;
    }
    if (options.retain_collided_census)
    {
        bool supported = false;
#if !defined(AMREX_USE_GPU)
        auto const census_eos = model.electronThermodynamicsExecutor();
        supported = (census_eos.isIdealGas() || census_eos.isFixedChargeLatentEnergy()) &&
            census_eos.m_num_materials == 0 &&
            model.transportsElectronInternalEnergyAtRawDensity() &&
            WarpX::GetInstance().finestLevel() == 0 && WarpX::n_rz_azimuthal_modes == 1 &&
            WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC &&
            geometry.ProbLo(0) == 0 && !geometry.isPeriodic(0) && geometry.isPeriodic(1) &&
            WarpX::field_boundary_lo[0] == FieldBoundaryType::None &&
            WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC &&
            WarpX::particle_boundary_lo[0] == ParticleBoundaryType::None &&
            WarpX::particle_boundary_hi[0] == ParticleBoundaryType::Reflecting &&
            !options.interval.reflecting_axial_fixed_mask_trial;
#endif
        if (!Collective(supported))
        {
            rejected.reason = "private history census requires CPU built-in periodic RZ";
            return rejected;
        }
#if !defined(AMREX_USE_GPU)
        if (!HistoryCensusSchema(live, options.collided_census) ||
            !HistoryCensusSchema(start.Particles(), options.collided_census))
        {
            rejected.reason = "invalid private history census schema";
            return rejected;
        }
#endif
    }
#if !defined(AMREX_USE_GPU)
    auto const preflight_eos = model.electronThermodynamicsExecutor();
    // The table caller's closed selector is rejected collectively before any
    // capture, clone, callback or clock initialization, including empty ranks.
    if (!Collective(!preflight_eos.isSingularitySpiner() ||
                    (geometry.isPeriodic(1) &&
                     !options.interval.reflecting_axial_fixed_mask_trial) ||
                    options.interval.reflecting_axial_table_trial))
    {
        rejected.reason = "table+closed RZ thermal history is unsupported";
        return rejected;
    }
#endif
    if (!ValidateNativeResolvedRZThermalIntervalScope(model, geometry, options.interval))
    {
        return rejected;
    }
    auto const& native_geometry = WarpX::GetInstance().Geom(0);
    auto const eos = model.electronThermodynamicsExecutor();
    bool table = false;
#if defined(WARPX_USE_SINGULARITY_EOS) && !defined(AMREX_USE_GPU)
    table = eos.isSingularitySpiner() && eos.m_num_materials == 1 &&
        ((!eos.m_use_resolved_material_selection && !options.interval.single_selected_table_trial) ||
         options.interval.single_selected_table_trial) && eos.tableStorage() != nullptr &&
        model.transportsElectronInternalEnergyAtRawDensity() &&
        ((geometry.isPeriodic(1) && !options.interval.reflecting_axial_fixed_mask_trial &&
          !options.interval.reflecting_axial_table_trial) ||
         (!geometry.isPeriodic(1) && options.interval.reflecting_axial_fixed_mask_trial &&
          options.interval.reflecting_axial_table_trial)) &&
        material_start.CandidateSpecies().size() == 1 &&
        material_start.CandidateSpecies().front().name ==
            model.electronThermodynamicsMaterialSpeciesName(0);
#endif
    bool valid =
        WarpX::GetInstance().finestLevel() == 0 &&
        (eos.isIdealGas() || eos.isFixedChargeLatentEnergy() || table) &&
        old_diffusion.boxArray() == live.ParticleBoxArray(0) &&
        old_diffusion.DistributionMap() == live.ParticleDistributionMap(0) &&
        geometry.Domain() == native_geometry.Domain() &&
        geometry.Coord() == native_geometry.Coord() &&
        options.subintervals > 0 && options.maximum_subintervals > 0 &&
        options.maximum_subintervals <= 32 &&
        options.subintervals <= options.maximum_subintervals &&
        options.interval.collisions.max_optical_depth > 0 &&
        options.interval.collisions.max_optical_depth <= 16 &&
        options.interval.collisions.max_events_per_photon > 0 &&
        options.interval.collisions.max_events_per_photon <= 64 &&
        options.interval.thermal.packets.max_absorption_optical_depth > 0 &&
        options.interval.thermal.packets.max_absorption_optical_depth <= 16 &&
        !options.opacity_model_identity.empty() &&
        options.opacity_model_identity.size() <= 1024 && (independent || evaluate) &&
        (!options.interval.reflecting_axial_fixed_mask_trial ||
         PhysConst::c * outer_dt <=
             .5 * std::min(geometry.CellSize(0), geometry.CellSize(1))) &&
        std::isfinite(time + outer_dt) && time >= 0 && outer_dt > 0 &&
        time + outer_dt > time && !path.empty() && path.size() <= 1024 &&
        old_diffusion.nComp() == 4 && mask.nComp() == 1 &&
        old_diffusion.ixType().cellCentered() && mask.ixType().cellCentered() &&
        old_temperature.nComp() == 1 &&
        old_temperature.ixType().nodeCentered() &&
        old_temperature.nGrowVect().allGE(1) &&
        mask.boxArray() == old_diffusion.boxArray() &&
        mask.DistributionMap() == old_diffusion.DistributionMap() &&
        old_temperature.boxArray() ==
            amrex::convert(old_diffusion.boxArray(),
                           amrex::IntVect::TheNodeVector()) &&
        old_temperature.DistributionMap() == old_diffusion.DistributionMap() &&
        options.interval.desired_diffusion_mask == nullptr &&
        options.interval.radiation_representation_carry == nullptr &&
        options.interval.initial_photon_clocks == nullptr &&
        options.interval.authenticated_history_step == nullptr &&
        options.interval.all_d_no_k_event_witness == nullptr &&
        options.interval.thermal.rz_independent_gray_opacity == nullptr &&
        options.interval.photon_time_policy == NativeRZThermalPhotonTimePolicy::PhotonHalfCell &&
        options.interval.absorption_depth_policy ==
            NativeRZThermalAbsorptionDepthPolicy::WholeOuter16;
    if (step_depth)
    {
        valid = valid && std::numeric_limits<amrex::Real>::digits == 53 &&
            table && options.interval.single_selected_table_trial &&
            !options.retain_collided_census && geometry.isPeriodic(1) &&
            !options.interval.reflecting_axial_fixed_mask_trial &&
            !options.interval.reflecting_axial_table_trial &&
            WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC &&
            PhysConst::c * outer_dt <=
                .5 * std::min(geometry.CellSize(0), geometry.CellSize(1));
#if defined(AMREX_USE_GPU)
        valid = false;
#endif
    }
    if (all_d)
    {
        valid = valid && std::numeric_limits<amrex::Real>::digits == 53 &&
            std::numeric_limits<amrex::ParticleReal>::digits == 53 &&
            model.transportsElectronInternalEnergyAtRawDensity() &&
            !options.retain_collided_census && geometry.isPeriodic(1) &&
            !options.interval.reflecting_axial_fixed_mask_trial &&
            !options.interval.reflecting_axial_table_trial &&
            WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC &&
            (implicit_step || PhysConst::c * outer_dt <=
                .5 * std::min(geometry.CellSize(0), geometry.CellSize(1)));
#if defined(AMREX_USE_GPU)
        valid = false;
#endif
    }
    if (independent)
    {
        valid = valid && (independent_mixed ? (!all_d && !step_depth) : all_d) &&
            !implicit_step && table &&
            options.interval.single_selected_table_trial &&
            options.independent_gray_opacity.enabled && !options.retain_collided_census &&
            PhysConst::c * outer_dt <=
                .5 * std::min(geometry.CellSize(0), geometry.CellSize(1));
    }
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
    {
        valid = valid &&
                geometry.ProbLo(axis) == native_geometry.ProbLo(axis) &&
                geometry.ProbHi(axis) == native_geometry.ProbHi(axis) &&
                geometry.isPeriodic(axis) == native_geometry.isPeriodic(axis);
    }
    if (!Collective(valid))
    {
        return rejected;
    }
    amrex::Real controls[4]{
        time, outer_dt, static_cast<amrex::Real>(options.subintervals),
        static_cast<amrex::Real>(options.maximum_subintervals)};
    amrex::Real maximum[4];
    std::copy(std::begin(controls), std::end(controls), maximum);
    amrex::ParallelDescriptor::ReduceRealMin(controls, 4);
    amrex::ParallelDescriptor::ReduceRealMax(maximum, 4);
    if (!Collective(
            std::equal(std::begin(controls), std::end(controls), maximum)))
    {
        return rejected;
    }
    int length = amrex::ParallelDescriptor::IOProcessor()
                     ? static_cast<int>(options.opacity_model_identity.size())
                     : 0;
    amrex::ParallelDescriptor::Bcast(
        &length, 1, amrex::ParallelDescriptor::IOProcessorNumber());
    std::vector<char> identity(static_cast<std::size_t>(length));
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::copy(options.opacity_model_identity.begin(),
                  options.opacity_model_identity.end(), identity.begin());
    }
    amrex::ParallelDescriptor::Bcast(
        identity.data(), length,
        amrex::ParallelDescriptor::IOProcessorNumber());
    if (!Collective(std::string(identity.begin(), identity.end()) ==
                    options.opacity_model_identity))
    {
        return rejected;
    }
    length = amrex::ParallelDescriptor::IOProcessor()
                 ? static_cast<int>(path.size())
                 : 0;
    amrex::ParallelDescriptor::Bcast(
        &length, 1, amrex::ParallelDescriptor::IOProcessorNumber());
    identity.resize(static_cast<std::size_t>(length));
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::copy(path.begin(), path.end(), identity.begin());
    }
    amrex::ParallelDescriptor::Bcast(
        identity.data(), length,
        amrex::ParallelDescriptor::IOProcessorNumber());
    if (!Collective(std::string(identity.begin(), identity.end()) == path))
    {
        return rejected;
    }
    if (!start.ValidateSnapshotFor(live))
    {
        return rejected;
    }
    std::unique_ptr<NativeRZThermalAllDNoKEventWitness const> all_d_witness;
    if (all_d)
    {
        if (!ValidateNativeRZAllDNoPhotonState(live, start, mask, geometry, rejected.reason))
        { return rejected; }
        all_d_witness.reset(new NativeRZThermalAllDNoKEventWitness(
            live, start, mask, geometry, options.interval.collisions.seed,
            options.interval.collisions.interval, options.photon_time_policy,
            time, time + outer_dt, outer_dt, options.subintervals));
        if (!all_d_witness->ValidateOwners(start, all_d_witness->CanonicalMask(), rejected.reason))
        { return rejected; }
    }
    auto const& physical_mask = all_d ? all_d_witness->CanonicalMask() : mask;
    NativeResolvedRZThermalHistoryResult result;
    if (!material_start.TryCloneCandidate(result.material))
    {
        return rejected;
    }
    result.photons = std::make_unique<PhotonPacketTransaction>(live, start);
    if (all_d && !all_d_witness->ValidateOwners(
            *result.photons, physical_mask, rejected.reason)) { return rejected; }
#if !defined(AMREX_USE_GPU)
    if (options.retain_collided_census &&
        !SnapshotCollidedDiffusionAdmission(result.photons->Particles(),
                                            options.collided_census).valid)
    {
        rejected.reason = "invalid post-entry private history census snapshot";
        return rejected;
    }
#endif
    result.diffusion_radiation = Copy(old_diffusion);
    result.nodal_temperature = Copy(old_temperature);
    auto const material_anchor = CaptureMaterial(material_start, path);
    valid = !material_anchor.empty();
    for (auto const& anchor : material_anchor)
    {
        valid = valid && anchor.mass > 0 && std::isfinite(anchor.mass) &&
                std::all_of(anchor.carry.begin(), anchor.carry.end(),
                            [] (int d) { return d >= 0; });
    }
    if (!Collective(valid))
    {
        rejected.reason = "invalid private history material/carry schema";
        return rejected;
    }
    auto const radiation_before =
        Radiation(start.Particles(), old_diffusion, geometry);
    auto const frozen_charge = Copy(*WarpX::GetInstance().m_fields.get(
        warpx::fields::FieldType::rho_fp, 0));
    auto caloric = PrepareNativeRZOwnedNodalCaloric(
        model, frozen_charge, old_temperature,
        options.interval.thermal.minimum_electron_density,
        NativeRZCaloricOptions{options.interval.reflecting_axial_fixed_mask_trial,
                              options.interval.single_selected_table_trial,
                              options.interval.thermal.rz_single_selected_table_vacuum});
    if (!caloric.Valid())
    {
        rejected.reason = caloric.Reason();
        return rejected;
    }
    detail::ResolvedMaterialChargeFields independent_units{};
    if (independent)
    {
        auto const name = "ni_charge_fp_" + model.electronThermodynamicsMaterialSpeciesName(0);
        if (!Collective(WarpX::GetInstance().m_fields.has(name, 0)))
        {
            rejected.reason = "independent-gray History has no native unit-ion owner";
            return rejected;
        }
        independent_units[0] = WarpX::GetInstance().m_fields.get(name, 0);
    }
    auto const& boxes = old_diffusion.boxArray();
    auto const& mapping = old_diffusion.DistributionMap();
    for (auto* field :
         {&result.actual_material_work, &result.material_energy_carry_change,
          &result.actual_heat})
    {
        field->define(boxes, mapping, 1, 0);
        field->setVal(0);
    }
    for (auto* field : {&result.actual_material_impulse,
                        &result.actual_conjugate_impulse,
                        &result.conjugate_momentum_carry_change})
    {
        field->define(boxes, mapping, 3, 0);
        field->setVal(0);
    }
    amrex::MultiFab zero(boxes, mapping, 3, 0);
    zero.setVal(0);
    ParticleImpulseTransaction initial_frame;
    auto const assignment = options.interval.thermal.RZForceAssignment();
    if (!initial_frame.Stage(*result.material, path, zero, false,
                             assignment))
    {
        rejected.reason = "invalid initial actual private RZ rate frame";
        return rejected;
    }
    amrex::MultiFab rate;
    if (!initial_frame.TryInitialRateVelocity(rate))
    {
        rejected.reason = "nonfinite actual initial private RZ rate frame";
        return rejected;
    }
    if (!photon_replay_detail::ReconcileKeyedPhotonHistoryCarries(
            {}, nullptr, result.photons->Particles(), live,
            options.interval.collisions, true, result.photon_clocks))
    {
        rejected.reason = "invalid fresh outer photon clocks";
        return rejected;
    }
    if (options.retain_collided_census && !Collective(std::all_of(
        result.photon_clocks.begin(), result.photon_clocks.end(),
        [] (NativePhotonPathCarry const& clock)
        { return clock.completed_events == 0 && clock.next_draw == 1; })))
    {
        rejected.reason = "invalid fresh outer history census provenance";
        return rejected;
    }
    if (all_d && !Collective(result.photon_clocks.empty()))
    {
        rejected.reason = "all-D no-photon history has nonempty fresh clocks";
        return rejected;
    }
    std::ostringstream policy;
    policy << "rz_private_material_history_v1_uniform_left_actual_T_rate_"
              "frozen_density_positions"
           << "_same_outer_clock_distinct_birth_seed_n=" << options.subintervals
           << "_cap=" << options.maximum_subintervals
           << "_clock_seed=" << options.interval.collisions.seed
           << "_clock_interval=" << options.interval.collisions.interval
           << "_event_cap=" << options.interval.collisions.max_events_per_photon
           << "_carry_path_len=" << path.size() << ':' << path
           << "_opacity_len=" << options.opacity_model_identity.size() << ':'
           << options.opacity_model_identity;
    result.policy_identity = policy.str();
    if (options.interval.thermal.rz_native_nodal_force)
    {
        result.policy_identity +=
            (options.interval.reflecting_axial_fixed_mask_trial
                ? "_builtin_native_reflected_closed_nodal_source_work_center_carry_v3_shape="
                : "_builtin_native_reflected_radial_nodal_source_work_center_carry_v2_shape=") +
            std::to_string(WarpX::nox);
    }
    if (independent)
    { result.policy_identity += independent_mixed
        ? "_independent_fixed_mask_packet_fld_R_P_true_S_frozen_left_outer_optical16_v1"
        : "_independent_R_P_true_S_frozen_accepted_left_slice_v1"; }
    if (options.independent_gray_opacity.allow_exact_vacuum)
    { result.policy_identity += "_exact_vacuum_K_raw_unit_zero_no_table_query_v1"
                                "_rz_native_linear_source_work_center_carry_v1"; }
    if (options.independent_gray_opacity.allow_exact_vacuum && WarpX::nox > 1)
    { result.policy_identity += "_native_shape=" + std::to_string(WarpX::nox); }
    if (options.interval.single_selected_table_trial)
    { result.policy_identity += "_private_single_registered_table_material0_true_selector_v1"; }
    if (options.interval.reflecting_axial_fixed_mask_trial)
    {
        result.policy_identity += "_private_fixed_mask_closed_z_Kwall_Dwall_v1";
    }
    if (options.interval.reflecting_axial_table_trial)
    {
        result.policy_identity += "_private_closed_table_actual_corner_caloric_v1";
    }
    if (options.retain_collided_census)
    {
        result.policy_identity += "_private_current_outer_collided_census_v1";
    }
    if (step_depth)
    {
        result.policy_identity +=
            "_pure_absorption_actual_history_step_dt_depth16_v1_allcell_every_trial";
    }
    if (all_d)
    {
        result.policy_identity +=
            "_all_d_no_photon_implicit_absorption_v1_actual_step_every_trial_scattering16";
    }
    if (implicit_step)
    {
        result.policy_identity += "_all_d_no_photon_implicit_step_without_photon_halfcell_v1";
    }
    std::array<amrex::Real, 3> step_bounds{};
    amrex::Real accepted_wall_bound = 0;
    std::array<amrex::Real, 3> step_operands{};
    long double physical_source_budget = 0;
    long double radiation_history_budget = 0;
    std::array<long double, 3> accumulated_operands{};
    auto previous_radiation = radiation_before;
    std::unique_ptr<NativeRZThermalHistoryStepContext const> final_duration_context;
    for (int n = 0; n < options.subintervals; ++n)
    {
        rejected.completed_private_subintervals =
            result.completed_private_subintervals;
        auto const begin = time + outer_dt * (static_cast<amrex::Real>(n) /
                                              options.subintervals);
        auto const end =
            n + 1 == options.subintervals
                ? time + outer_dt
                : time + outer_dt * (static_cast<amrex::Real>(n + 1) /
                                     options.subintervals);
        NativeResolvedRZThermalHistoryStep step;
        step.time = begin;
        step.dt = end - begin;
        std::unique_ptr<NativeRZThermalHistoryStepContext const> step_context;
        if (step_depth || all_d || independent_mixed)
        {
            std::string context_reason;
            step_context.reset(new NativeRZThermalHistoryStepContext(
                step.time, end, step.dt, time, time + outer_dt, outer_dt,
                n, options.subintervals, options.interval.collisions.seed,
                options.interval.collisions.interval));
            if (!ValidateNativeRZThermalHistoryStepContext(*step_context, step.dt,
                    options.interval.collisions.seed, options.interval.collisions.interval,
                    context_reason))
            {
                rejected.completed_private_subintervals = n;
                rejected.reason = context_reason;
                return rejected;
            }
        }
        step.initial_rate_velocity = Copy(rate);
        step.scattering_opacity.define(boxes, mapping, 1, 1);
        step.absorption_opacity.define(boxes, mapping, 1, 1);
        step.scattering_opacity.setVal(0);
        step.absorption_opacity.setVal(0);
        std::string reason;
        if (all_d && !all_d_witness->ValidateStep(*step_context, step.dt,
                options.interval.collisions.seed, options.interval.collisions.interval,
                *result.photons, physical_mask, reason))
        {
            rejected.reason = "history all-D pre-sample witness rejected: " + reason;
            return rejected;
        }
        if (implicit_step && !all_d_witness->ValidatePhotonDuration(*step_context, step.dt,
                options.interval.collisions.seed, options.interval.collisions.interval, live,
                *result.photons, physical_mask, geometry, reason))
        {
            rejected.reason = "history implicit photon duration rejected: " + reason;
            return rejected;
        }
        std::unique_ptr<NativeRZIndependentGrayOpacitySlice const> opacity_slice;
        if (independent)
        {
            IndependentGrayMovingAllDFieldOptions independent_options;
            independent_options.enabled = true;
            independent_options.field = options.interval.field;
            independent_options.opacity = options.independent_gray_opacity;
            opacity_slice = independent_mixed
                ? NativeRZIndependentGrayOpacitySlice::IssueMixed(*step_context, live,
                      *result.photons, physical_mask, frozen_charge, result.nodal_temperature,
                      independent_units, *options.independent_gray_table, geometry,
                      options.photon_time_policy, independent_options, reason)
                : NativeRZIndependentGrayOpacitySlice::Issue(*step_context,
                      *all_d_witness, *result.photons, frozen_charge, result.nodal_temperature,
                      independent_units, *options.independent_gray_table, geometry,
                      options.photon_time_policy, independent_options, reason);
            valid = opacity_slice != nullptr;
            if (valid)
            {
                auto const& channels = opacity_slice->Opacity().Snapshot().coefficients;
                amrex::MultiFab::Copy(step.scattering_opacity, channels.true_scattering,
                                     0, 0, 1, 1);
                amrex::MultiFab::Copy(step.absorption_opacity, channels.proper_absorption,
                                     0, 0, 1, 1);
            }
            if (valid && independent_mixed)
            { valid = opacity_slice->ValidatePacketFrame(rate,
                options.interval.collisions.max_optical_depth,
                options.interval.thermal.packets.max_absorption_optical_depth, reason); }
        }
        else
        {
            valid = evaluate(begin, result.nodal_temperature, rate,
                             step.scattering_opacity, step.absorption_opacity, reason);
        }
        if (!Collective(valid))
        {
            rejected.completed_private_subintervals = n;
            rejected.reason = "history opacity sample rejected: " + reason;
            return rejected;
        }
        step.scattering_opacity.FillBoundary(geometry.periodicity());
        step.absorption_opacity.FillBoundary(geometry.periodicity());
        if (step_depth && !ValidatePureAbsorptionHistoryStep(*step_context,
                step.scattering_opacity, step.absorption_opacity, step.initial_rate_velocity,
                geometry, options.interval.thermal.packets.max_absorption_optical_depth, reason))
        {
            rejected.completed_private_subintervals = n;
            rejected.reason = "history initial-rate depth rejected: " + reason;
            return rejected;
        }
        if (all_d && !ValidateAllDImplicitAbsorptionHistoryStep(*step_context,
                *all_d_witness, step.scattering_opacity, step.absorption_opacity,
                step.initial_rate_velocity, geometry, options.interval.collisions.max_optical_depth,
                options.interval.thermal.packets.max_absorption_optical_depth, reason))
        {
            rejected.reason = "history all-D initial-rate admission rejected: " + reason;
            return rejected;
        }
        auto leaf = options.interval;
        leaf.initial_photon_clocks = &result.photon_clocks;
        if (step_depth || all_d || independent_mixed)
        {
            leaf.absorption_depth_policy = options.absorption_depth_policy;
            leaf.authenticated_history_step = step_context.get();
        }
        if (all_d) { leaf.all_d_no_k_event_witness = all_d_witness.get(); }
        leaf.thermal.rz_independent_gray_opacity = opacity_slice.get();
        if (implicit_step) { leaf.photon_time_policy = options.photon_time_policy; }
        leaf.thermal.packets.seed = MixPhotonBirthKey(
            options.interval.collisions.seed ^ UINT64_C(0x6fd0aa15619e23b7) ^
            static_cast<std::uint64_t>(n));
        leaf.thermal.packets.interval = options.interval.collisions.interval;
        step.accepted = TryAdvanceNativeResolvedRZThermalInterval(
            live, *result.photons, *result.material, path,
            result.diffusion_radiation, physical_mask, result.nodal_temperature,
            step.scattering_opacity, step.absorption_opacity, rate, model,
            geometry, step.dt, leaf);
        if (!step.accepted.valid)
        {
            rejected.completed_private_subintervals = n;
            rejected.reason =
                "history physical slice rejected: " + step.accepted.reason;
            return rejected;
        }
        // The epoch borrows accepted-left T/photons; destroy it before adopting
        // the new material/temperature/photon state. No pointer escapes the leaf.
        opacity_slice.reset();
        leaf.thermal.rz_independent_gray_opacity = nullptr;
        auto& candidate = step.accepted.candidate;
        if (!candidate.material_stage->TryCandidateRateVelocity(
                step.endpoint_rate_velocity))
        {
            rejected.completed_private_subintervals = n;
            rejected.reason = "nonfinite actual history endpoint rate";
            return rejected;
        }
        if (step_depth && !ValidatePureAbsorptionHistoryStep(*step_context,
                step.scattering_opacity, step.absorption_opacity, step.endpoint_rate_velocity,
                geometry, options.interval.thermal.packets.max_absorption_optical_depth, reason))
        {
            rejected.completed_private_subintervals = n;
            rejected.reason = "history accepted endpoint depth rejected: " + reason;
            return rejected;
        }
        if (all_d &&
            (!all_d_witness->ValidateStep(*step_context, step.dt, leaf.collisions.seed,
                leaf.collisions.interval, *candidate.photons, physical_mask, reason) ||
             !ValidateAllDImplicitAbsorptionHistoryStep(*step_context, *all_d_witness,
                step.scattering_opacity, step.absorption_opacity, step.endpoint_rate_velocity,
                geometry, leaf.collisions.max_optical_depth,
                leaf.thermal.packets.max_absorption_optical_depth, reason)))
        {
            rejected.reason = "history all-D accepted endpoint admission rejected: " + reason;
            return rejected;
        }
        if (!photon_replay_detail::UniqueLogicalPhotonKeys(
                candidate.photons->Particles()))
        {
            rejected.reason = "duplicate history photon birth keys";
            return rejected;
        }
        std::vector<NativePhotonPathCarry> next_clocks;
        std::vector<std::uint64_t> owner_birth_keys;
        if (!photon_replay_detail::ReconcileKeyedPhotonHistoryCarries(
                step.accepted.path.photon_carries, &result.photons->Particles(),
                candidate.photons->Particles(), live, leaf.collisions, false,
                next_clocks, &candidate.rz_history_birth_keys,
                options.retain_collided_census ? &owner_birth_keys : nullptr))
        {
            rejected.reason = "accepted source lost clock/birth provenance";
            return rejected;
        }
        if (all_d && !Collective(next_clocks.empty() && candidate.rz_history_birth_keys.empty()))
        {
            rejected.reason = "all-D accepted history lost empty clock/birth closure";
            return rejected;
        }
#if !defined(AMREX_USE_GPU)
        if (options.retain_collided_census)
        {
            if (!InitializeAcceptedHistoryBirthCensus(candidate.photons->Particles(), live,
                    options.collided_census, next_clocks, owner_birth_keys))
            {
                rejected.reason = "accepted history birth lost census provenance";
                return rejected;
            }
            auto& keys = result.census_receipt.actual_birth_keys;
            keys.insert(keys.end(), candidate.rz_history_birth_keys.begin(),
                        candidate.rz_history_birth_keys.end());
        }
#endif
        amrex::MultiFab::Add(result.actual_material_work,
                             candidate.material_stage->ActualWork(), 0, 0, 1,
                             0);
        amrex::MultiFab::Add(result.material_energy_carry_change,
                             candidate.material_stage->EnergyCarryChange(), 0,
                             0, 1, 0);
        amrex::MultiFab::Add(result.actual_heat, candidate.actual_heat, 0, 0, 1,
                             0);
        amrex::MultiFab::Add(result.actual_material_impulse,
                             candidate.material_stage->ActualImpulse(), 0, 0,
                             3, 0);
        amrex::MultiFab::Add(result.actual_conjugate_impulse,
                             candidate.material_stage->ConjugateImpulse(), 0, 0,
                             3, 0);
        amrex::MultiFab::Add(
            result.conjugate_momentum_carry_change,
            candidate.material_stage->ConjugateMomentumCarryChange(), 0, 0, 3,
            0);
        step_bounds[0] += candidate.energy_roundoff_bound;
        auto const actual_work = Host(candidate.material_stage->ActualWork());
        auto const carry_work =
            Host(candidate.material_stage->EnergyCarryChange());
        auto const actual_heat = Host(candidate.actual_heat);
        auto const actual_impulse =
            Host(candidate.material_stage->ConjugateImpulse());
        auto const momentum_carry =
            Host(candidate.material_stage->ConjugateMomentumCarryChange());
        auto const bound = Host(candidate.storage_roundoff_bound);
        auto const impulse_bound =
            Host(candidate.material_stage->RZConjugateMomentumRoundoffBound());
        auto const operands = Host(candidate.operand_scale);
        for (amrex::MFIter it(bound); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                auto const lo = geometry.ProbLo(0) +
                                (cell()[0] - geometry.Domain().smallEnd(0)) *
                                    geometry.CellSize(0);
                auto const hi = lo + geometry.CellSize(0);
                auto const mean =
                    (2. / 3) * (lo * lo + lo * hi + hi * hi) / (lo + hi);
                step_bounds[1] += bound[it](cell(), 3) +
                                  PhysConst::c * impulse_bound[it](cell(), 2);
                step_bounds[2] +=
                    mean * (bound[it](cell(), 2) +
                            PhysConst::c * impulse_bound[it](cell(), 1));
                step_operands[0] += operands[it](cell(), 0);
                step_operands[1] += operands[it](cell(), 3);
                step_operands[2] += mean * operands[it](cell(), 2);
                physical_source_budget += std::abs(actual_work[it](cell())) +
                                          std::abs(carry_work[it](cell())) +
                                          std::abs(actual_heat[it](cell()));
                accumulated_operands[0] += std::abs(actual_work[it](cell())) +
                                           std::abs(carry_work[it](cell()));
                accumulated_operands[1] +=
                    PhysConst::c * (std::abs(actual_impulse[it](cell(), 2)) +
                                    std::abs(momentum_carry[it](cell(), 2)));
                accumulated_operands[2] +=
                    PhysConst::c * mean *
                    (std::abs(actual_impulse[it](cell(), 1)) +
                     std::abs(momentum_carry[it](cell(), 1)));
            }
        }
        for (int d = 0; d < 3; ++d)
        {
            result.wall_momentum[d] += step.accepted.path.path.wall_momentum[d];
        }
        for (int side = 0; side < 2; ++side)
        {
            result.kinetic_wall_momentum_by_side[side] +=
                step.accepted.path.path.axial_wall_momentum_by_side[side];
            result.diffusion_wall_momentum_by_side[side] +=
                step.accepted.diffusion_wall_momentum_by_side[side];
        }
        result.diffusion_wall_momentum += step.accepted.diffusion_wall_momentum;
        result.axial_wall_momentum_operand += step.accepted.axial_wall_momentum_operand;
        accepted_wall_bound += step.accepted.axial_wall_roundoff_bound;
        result.diffusion_radiation = Copy(candidate.diffusion_radiation);
        result.nodal_temperature = Copy(candidate.nodal_temperature);
        auto const accepted_radiation =
            Radiation(candidate.photons->Particles(),
                      candidate.diffusion_radiation, geometry);
        radiation_history_budget +=
            previous_radiation.operands[0] + accepted_radiation.operands[0];
        previous_radiation = accepted_radiation;
        rate = Copy(step.endpoint_rate_velocity);
        candidate.material_stage->Commit(); // PRIVATE only; next source sees
                                            // actual accepted u/carry.
        if (!result.photons->AdoptCandidate(*candidate.photons))
        {
            rejected.reason =
                "history photon adoption lost original live provenance";
            return rejected;
        }
        candidate.photons.reset();
        result.photon_clocks = std::move(next_clocks);
        result.steps.push_back(std::move(step));
        ++result.completed_private_subintervals;
        if (implicit_step) { final_duration_context = std::move(step_context); }
    }
    rejected.completed_private_subintervals =
        result.completed_private_subintervals;
    amrex::ParallelDescriptor::ReduceRealSum(step_bounds.data() + 1, 2);
    if (options.interval.reflecting_axial_fixed_mask_trial)
    {
        // These leaf wall accounts/bounds are ALREADY GLOBAL. Never place
        // them in the local cell ledger before its MPI reduction.
        step_bounds[1] += accepted_wall_bound;
    }
    amrex::ParallelDescriptor::ReduceRealSum(step_operands.data(), 3);
    Sum(&physical_source_budget, 1);
    Sum(accumulated_operands.data(), 3);
    auto const radiation_after = Radiation(
        result.photons->Particles(), result.diffusion_radiation, geometry);
    auto const material = MeasureMaterial(material_anchor, *result.material);
    amrex::MultiFab owner_zero(old_temperature.boxArray(),
                               old_temperature.DistributionMap(), 1, 0);
    owner_zero.setVal(0);
    auto physical_eos = EvaluateNativeRZOwnedNodalCaloric(
        model, caloric, owner_zero, &result.nodal_temperature);
    if (!physical_eos.valid || !material.valid)
    {
        rejected.reason = "independent history endpoint state rejected";
        return rejected;
    }
    auto const eos_field = Host(physical_eos.cell_realized_energy);
    long double eos_heat = 0, eos_scale = 0, eos_bound = 0;
    for (amrex::MFIter it(eos_field); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            eos_heat += eos_field[it](cell(), 0);
            eos_scale += std::abs(eos_field[it](cell(), 0));
            eos_bound += eos_field[it](cell(), 1);
        }
    }
    Sum(&eos_heat, 1);
    Sum(&eos_scale, 1);
    Sum(&eos_bound, 1);
    auto& witness = result.acceptance_witness;
    auto& budget_input = witness.original;
    budget_input.material_records = material.records;
    budget_input.radiation_before_records = radiation_before.records;
    budget_input.radiation_after_records = radiation_after.records;
    budget_input.cells = geometry.Domain().numPts();
    budget_input.subintervals = options.subintervals;
    budget_input.ranks = amrex::ParallelDescriptor::NProcs();
    auto const count_budget = EvaluateNativeRZHistoryCountBudget(budget_input);
    valid = count_budget.valid && result.material->ValidateLive() &&
            result.photons->ValidateSnapshotFor(live);
    result.endpoint_physical_energy_scale = static_cast<amrex::Real>(
        radiation_history_budget + physical_source_budget);
    budget_input.before_operands = radiation_before.operands;
    budget_input.after_operands = radiation_after.operands;
    budget_input.material_operands = material.operands;
    budget_input.accumulated_operands = accumulated_operands;
    budget_input.eos_scale = eos_scale;
    budget_input.eos_bound = eos_bound;
    budget_input.physical_source_budget = physical_source_budget;
    budget_input.physical_energy_scale = result.endpoint_physical_energy_scale;
    budget_input.radial_lever = geometry.ProbHi(0);
    budget_input.energy_tolerance = options.interval.thermal.energy_tolerance;
    for (int d = 0; d < 3; ++d)
    {
        budget_input.step_operands[d] = step_operands[d];
        budget_input.step_bounds[d] = step_bounds[d];
        budget_input.wall_operands[d] = d == 1 ?
            std::abs(PhysConst::c * result.wall_momentum[2]) : 0;
        if (options.interval.reflecting_axial_fixed_mask_trial && d == 1)
        {
            budget_input.wall_operands[d] = PhysConst::c * result.axial_wall_momentum_operand;
        }
    }
    // Pair ACTUAL outer material/EOS endpoints independently with accumulated
    // cell accounts. Summing reported leaf residuals would not establish this.
    auto const accumulated_work = Host(result.actual_material_work);
    auto const accumulated_carry = Host(result.material_energy_carry_change);
    auto const accumulated_heat = Host(result.actual_heat);
    auto const accumulated_impulse = Host(result.actual_conjugate_impulse);
    auto const accumulated_momentum_carry =
        Host(result.conjugate_momentum_carry_change);
    std::array<long double, 3> reported_work{}, reported_carry{},
        reported_scale{};
    long double reported_heat = 0;
    for (amrex::MFIter it(accumulated_work); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const lo = static_cast<long double>(geometry.ProbLo(0)) +
                            (cell()[0] - geometry.Domain().smallEnd(0)) *
                                static_cast<long double>(geometry.CellSize(0));
            auto const hi = lo + geometry.CellSize(0);
            auto const mean =
                (2.L / 3) * (lo * lo + lo * hi + hi * hi) / (lo + hi);
            std::array<long double, 3> work{
                accumulated_work[it](cell()),
                PhysConst::c * accumulated_impulse[it](cell(), 2),
                PhysConst::c * mean * accumulated_impulse[it](cell(), 1)};
            std::array<long double, 3> carry{
                accumulated_carry[it](cell()),
                PhysConst::c * accumulated_momentum_carry[it](cell(), 2),
                PhysConst::c * mean *
                    accumulated_momentum_carry[it](cell(), 1)};
            for (int d = 0; d < 3; ++d)
            {
                reported_work[d] += work[d];
                reported_carry[d] += carry[d];
                reported_scale[d] += std::abs(work[d]) + std::abs(carry[d]);
            }
            reported_heat += accumulated_heat[it](cell());
        }
    }
    Sum(reported_work.data(), 3);
    Sum(reported_carry.data(), 3);
    Sum(reported_scale.data(), 3);
    Sum(&reported_heat, 1);
    budget_input.reported_scale = reported_scale;
    auto const budget = EvaluateNativeRZHistoryAcceptanceBudget(budget_input);
    valid = valid && budget.valid;
    for (int d = 0; d < 3; ++d)
    {
        auto const bound = budget.material_report[d];
        witness.work_report_residual[d] = reported_work[d] - material.physical[d];
        witness.carry_report_residual[d] = reported_carry[d] - material.carry[d];
        valid = valid &&
                std::abs(witness.work_report_residual[d]) <= bound &&
                std::abs(witness.carry_report_residual[d]) <= bound;
    }
    witness.heat_report_residual = reported_heat - eos_heat;
    valid = valid &&
            std::abs(witness.heat_report_residual) <= budget.heat_report;
    for (int d = 0; d < 3; ++d)
    {
        auto const wall = d == 1 ? PhysConst::c * result.wall_momentum[2] : 0;
        auto const diffusion_wall =
            d == 1 ? PhysConst::c * result.diffusion_wall_momentum : 0;
        auto const heat = d == 0 ? eos_heat : 0;
        auto residual =
            radiation_before.inventory[d] - radiation_after.inventory[d] -
            material.physical[d] - material.carry[d] - heat - wall;
        if (options.interval.reflecting_axial_fixed_mask_trial && d == 1)
        {
            residual -= diffusion_wall;
        }
        // Uncancelled material background KE/EOS operands condition arithmetic
        // only. The physical tolerance scales radiation and actual source
        // changes, never the potentially much larger background inventory.
        // Retain qualified leaf E/Pz/Lz source allowances using actual
        // radiation/source operands, with a physical lever for torque.
        auto const allowance = budget.endpoint[d];
        witness.residual[d] = residual;
        result.endpoint_residual[d] = static_cast<amrex::Real>(residual);
        result.endpoint_roundoff_bound[d] = static_cast<amrex::Real>(allowance);
        valid = valid && std::isfinite(residual + allowance) &&
                allowance >= 0 && std::abs(residual) <= allowance;
    }
    if (!Collective(valid))
    {
        rejected.reason =
            "independent cumulative history endpoint ledger rejected";
        return rejected;
    }
    if (all_d &&
        (!all_d_witness->ValidateOwners(*result.photons, physical_mask, rejected.reason) ||
         !Collective(result.photon_clocks.empty())))
    {
        rejected.reason = "all-D history prepublication owner/clock witness rejected";
        return rejected;
    }
    if (implicit_step)
    {
        // Retain the last ACTUAL engine-issued context, never reconstruct a
        // shorter/publication step from floating times or dt/N.
        if (!Collective(final_duration_context != nullptr) ||
            !all_d_witness->ValidatePhotonDuration(*final_duration_context,
                final_duration_context->dt(), options.interval.collisions.seed,
                options.interval.collisions.interval, live, *result.photons, physical_mask,
                geometry, rejected.reason))
        { return rejected; }
    }
    result.valid = true;
    witness.minimum_after_records = radiation_after.diffusion_records;
    // A separately summed immutable D operand can round above a later mixed
    // photon+D sum. Downward bound independent of reduction association.
    auto const floor_product = 256.L *
        (static_cast<long double>(radiation_after.records) + budget_input.cells +
         budget_input.ranks + 1) * std::numeric_limits<long double>::epsilon();
    witness.valid = floor_product < .25L;
    for (int d = 0; d < 3; ++d)
    {
        witness.minimum_after_operands[d] = std::max(0.L,
            radiation_after.diffusion_operands[d] * (1 - floor_product));
        witness.minimum_after_operands[d] = std::min(witness.minimum_after_operands[d],
                                                    radiation_after.operands[d]);
        if (witness.minimum_after_operands[d] > 0)
        { witness.minimum_after_operands[d] = std::nextafter(witness.minimum_after_operands[d], 0.L); }
    }
    if (options.retain_collided_census)
    {
        for (auto const& clock : result.photon_clocks)
        {
            if (clock.completed_events > 0)
            {
                result.census_receipt.collided_clocks.push_back(clock);
            }
        }
        result.census_receipt.valid = true;
    }
    return result;
#endif
}
} // namespace warpx::radiation
