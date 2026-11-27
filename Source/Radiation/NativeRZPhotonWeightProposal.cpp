/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativeRZPhotonWeightProposal.H"

#include "ConservativePhotonWeightProposal.H"
#include "ConservativePhotonWeightConditioning.H"
#include "ConservativePhotonWeightTree.H"
#include "NativePhotonCarryRouting.H"
#include "NativePhotonReplayInternals.H"
#include "PhotonKeyAudit.H"
#include "PhotonLogicalDrawKey.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Gpu.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <utility>

namespace warpx::radiation
{
namespace
{
bool All (bool value)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(value);
    return value;
}

#if defined(WARPX_DIM_RZ)
void Sum (NativePhotonWeightInventory& value)
{
#ifdef AMREX_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, value.data(), NativePhotonWeightMoments,
                  MPI_LONG_DOUBLE, MPI_SUM, amrex::ParallelDescriptor::Communicator());
#else
    amrex::ignore_unused(value);
#endif
}

void Maximum (NativePhotonWeightInventory& value)
{
#ifdef AMREX_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, value.data(), NativePhotonWeightMoments,
                  MPI_LONG_DOUBLE, MPI_MAX, amrex::ParallelDescriptor::Communicator());
#else
    amrex::ignore_unused(value);
#endif
}
#endif

bool SameOptions (NativeRZPhotonWeightProposalOptions const& options)
{
    bool valid = true;
#ifdef AMREX_USE_MPI
    std::uint64_t guard_mask = 0;
    for (int d = 0; d < NativePhotonWeightMoments; ++d)
    { guard_mask |= static_cast<std::uint64_t>(options.guards[d].required) << d; }
    std::array<std::uint64_t, 13> words{options.seed, options.outer_interval,
        static_cast<std::uint64_t>(options.fresh_outer_endpoint),
        static_cast<std::uint64_t>(options.fixed_gray_cohort), guard_mask,
        options.canonical_cell, options.maximum_blocks_per_cohort,
        options.maximum_elimination_attempts,
        static_cast<std::uint64_t>(options.require_well_resolved_direction),
        static_cast<std::uint64_t>(options.direction_method),
        static_cast<std::uint64_t>(options.collect_diagnostics),
        options.maximum_captured_blocks_per_rank,
        static_cast<std::uint64_t>(options.use_original_direction_certificate)};
    auto words_low = words, words_high = words;
    MPI_Allreduce(MPI_IN_PLACE, words_low.data(), 13, MPI_UINT64_T, MPI_MIN,
                  amrex::ParallelDescriptor::Communicator());
    MPI_Allreduce(MPI_IN_PLACE, words_high.data(), 13, MPI_UINT64_T, MPI_MAX,
                  amrex::ParallelDescriptor::Communicator());
    valid = words_low == words_high;
#endif
    for (auto const& guard : options.guards)
    {
        valid = valid && std::isfinite(guard.original_residual) &&
            std::isfinite(guard.unchanged_absolute_allowance) &&
            guard.unchanged_absolute_allowance >= 0;
    }
    valid = valid && (options.direction_method == PhotonWeightDirectionMethod::Legacy ||
        (options.direction_method == PhotonWeightDirectionMethod::RankRevealingOriginalRows &&
         options.require_well_resolved_direction && options.maximum_elimination_attempts == 1));
    valid = valid && options.maximum_captured_blocks_per_rank <= 64 &&
        (options.collect_diagnostics || options.maximum_captured_blocks_per_rank == 0);
    // Reject malformed values before min/max, including NaN on an empty rank.
    if (!All(valid)) { return false; }
#ifdef AMREX_USE_MPI
    for (auto const& guard : options.guards)
    {
        std::array<long double, 2> low{guard.original_residual,
                                     guard.unchanged_absolute_allowance};
        auto high = low;
        MPI_Allreduce(MPI_IN_PLACE, low.data(), 2, MPI_LONG_DOUBLE, MPI_MIN,
                      amrex::ParallelDescriptor::Communicator());
        MPI_Allreduce(MPI_IN_PLACE, high.data(), 2, MPI_LONG_DOUBLE, MPI_MAX,
                      amrex::ParallelDescriptor::Communicator());
        valid = valid && low == high;
    }
#endif
    return All(valid);
}

#if defined(WARPX_DIM_RZ)
void FinalizeDiagnostics (NativePhotonWeightProposalDiagnostics& diagnostics)
{
    if (!diagnostics.collected) { return; }
    std::array<amrex::Long, 6 + 15 + NativePhotonWeightDiagnosticStages> counts{};
    counts[0] = diagnostics.input_records;
    counts[1] = diagnostics.positive_records;
    counts[2] = diagnostics.cohorts;
    counts[3] = diagnostics.visited_cohorts;
    counts[4] = diagnostics.scheduled_blocks;
    counts[5] = diagnostics.eligible_positive_rays;
    std::copy(diagnostics.active_size.begin(), diagnostics.active_size.end(), counts.begin() + 6);
    std::copy(diagnostics.stages.begin(), diagnostics.stages.end(), counts.begin() + 21);
    amrex::ParallelDescriptor::ReduceLongSum(counts.data(), static_cast<int>(counts.size()));
    diagnostics.input_records = counts[0];
    diagnostics.positive_records = counts[1];
    diagnostics.cohorts = counts[2];
    diagnostics.visited_cohorts = counts[3];
    diagnostics.scheduled_blocks = counts[4];
    diagnostics.eligible_positive_rays = counts[5];
    std::copy_n(counts.begin() + 6, 15, diagnostics.active_size.begin());
    std::copy_n(counts.begin() + 21, NativePhotonWeightDiagnosticStages, diagnostics.stages.begin());
}

struct Ray
{
    std::size_t tile = 0;
    long index = 0;
    PhotonWeightRecord row;
    long double r = 0, theta = 0, ux = 0, uy = 0, uz = 0;
};

// Deliberately evaluate actual native operands with w first, exactly as the
// history endpoint's physical photon accounts, not row.features * w. Fixed
// phase-space makes the sqrt/trig results identical at both endpoints. Standard
// long-double product/sum guards below cover their weight-dependent arithmetic.
NativePhotonWeightInventory NativeValues (Ray const& ray, double weight)
{
    auto const norm = std::sqrt(ray.ux * ray.ux + ray.uy * ray.uy + ray.uz * ray.uz);
    auto const c = std::cos(ray.theta), s = std::sin(ray.theta);
    auto const ur = c * ray.ux + s * ray.uy;
    auto const ut = c * ray.uy - s * ray.ux;
    auto const f = static_cast<long double>(weight) * PhysConst::m_e * PhysConst::c;
    auto const e = f * norm;
    auto const nr = ur / norm, nt = ut / norm, nz = ray.uz / norm;
    return {weight, e, f * ray.ux, f * ray.uy, f * ray.uz, f * ray.r * ut,
        f * ur, e * nr * nr, e * nt * nt, e * nz * nz,
        e * nr * nt, e * nr * nz, e * nt * nz};
}

bool Cell (amrex::Geometry const& geometry, Ray const& ray, long double z,
           amrex::IntVect& cell)
{
    std::array<long double, 2> position{ray.r, z};
    cell = geometry.Domain().smallEnd();
    for (int d = 0; d < 2; ++d)
    {
        if (!std::isfinite(position[d]) || position[d] < geometry.ProbLo(d) ||
            !(position[d] < geometry.ProbHi(d))) { return false; }
        auto const offset = std::floor((position[d] - geometry.ProbLo(d)) /
                                      geometry.CellSize(d));
        if (offset > std::numeric_limits<int>::max()) { return false; }
        cell[d] += static_cast<int>(offset);
    }
    return geometry.Domain().contains(cell);
}

struct Pair
{
    std::vector<std::size_t> rays;
    std::array<PhotonWeightProposal, 2> endpoints;
    long double probability = 0;
    std::uint64_t bucket = 0;
    PhotonWeightTree tree;
};

std::uint64_t Bucket (std::vector<std::uint64_t> const& cohort, std::size_t block)
{
    auto value = photon_weight_detail::Mix(UINT64_C(0x4e4154495645525a));
    for (auto word : cohort) { value = photon_weight_detail::Mix(value ^ word); }
    return photon_weight_detail::Mix(value ^ photon_weight_detail::Mix(block));
}

double Counter (Pair const& pair, std::vector<Ray> const& rays,
                NativeRZPhotonWeightProposalOptions const& options, std::size_t path = 0)
{
    using photon_weight_detail::Mix;
    auto value = Mix(options.seed ^ UINT64_C(0x706f70756c617469));
    value = Mix(value ^ Mix(options.outer_interval ^ UINT64_C(0xa14587125bf62687)));
    value = Mix(value ^ Mix(pair.bucket ^ UINT64_C(0x87f4c9b62704a145)));
    for (auto index : pair.rays)
    {
        value = Mix(value ^ Mix(rays[index].row.key.first));
        value = Mix(value ^ Mix(rays[index].row.key.second));
    }
    return photon_weight_detail::CounterDraw{value}(path);
}
#endif
} // namespace

bool ValidateNativeRZPhotonWeightOriginalDirectionControls (
    NativeRZPhotonWeightProposalOptions const& options)
{
#ifdef AMREX_USE_MPI
    std::uint64_t low = static_cast<std::uint64_t>(options.use_original_direction_certificate);
    auto high = low;
    MPI_Allreduce(MPI_IN_PLACE, &low, 1, MPI_UINT64_T, MPI_MIN,
                  amrex::ParallelDescriptor::Communicator());
    MPI_Allreduce(MPI_IN_PLACE, &high, 1, MPI_UINT64_T, MPI_MAX,
                  amrex::ParallelDescriptor::Communicator());
    if (low != high) { return false; }
#endif
    if (!options.use_original_direction_certificate) { return true; }
    bool valid =
        options.direction_method == PhotonWeightDirectionMethod::RankRevealingOriginalRows &&
        options.require_well_resolved_direction && options.maximum_elimination_attempts == 1 &&
        std::numeric_limits<amrex::Real>::digits == 53 &&
        std::numeric_limits<amrex::ParticleReal>::digits == 53 &&
        std::numeric_limits<amrex::Real>::is_iec559 &&
        std::numeric_limits<amrex::ParticleReal>::is_iec559 &&
        std::numeric_limits<amrex::Real>::radix == 2 &&
        std::numeric_limits<amrex::ParticleReal>::radix == 2 &&
        std::numeric_limits<double>::is_iec559 &&
        std::numeric_limits<double>::radix == 2 &&
        std::numeric_limits<double>::digits == 53 &&
        std::numeric_limits<long double>::is_iec559 &&
        std::numeric_limits<long double>::radix == 2 &&
        std::numeric_limits<long double>::digits >= 64 &&
        std::numeric_limits<long double>::min_exponent <= -8192 &&
        std::numeric_limits<long double>::max_exponent >= 8192 &&
        std::fegetround() == FE_TONEAREST;
    // CUDA storage is captured synchronously before this ordinary host-only
    // proposal path. Its coefficient bounds and certificate use host extended
    // long double, never device long double. Other GPU backends remain guarded.
#if !defined(WARPX_DIM_RZ) || \
    (defined(AMREX_USE_GPU) && !defined(AMREX_USE_CUDA)) || defined(__FAST_MATH__)
    valid = false;
#endif
    return All(valid);
}

bool ValidateNativeRZPhotonWeightDiagnosticControls (
    NativeRZPhotonWeightProposalOptions const& options)
{
    bool valid = options.maximum_captured_blocks_per_rank <= 64 &&
        (options.collect_diagnostics || options.maximum_captured_blocks_per_rank == 0);
#ifdef AMREX_USE_MPI
    std::array<std::uint64_t, 2> low{
        static_cast<std::uint64_t>(options.collect_diagnostics),
        options.maximum_captured_blocks_per_rank};
    auto high = low;
    MPI_Allreduce(MPI_IN_PLACE, low.data(), 2, MPI_UINT64_T, MPI_MIN,
                  amrex::ParallelDescriptor::Communicator());
    MPI_Allreduce(MPI_IN_PLACE, high.data(), 2, MPI_UINT64_T, MPI_MAX,
                  amrex::ParallelDescriptor::Communicator());
    valid = valid && low == high;
#endif
    return All(valid);
}

NativeRZPhotonWeightProposalResult TryProposeNativeRZPhotonWeights (
    WarpXParticleContainer& live, PhotonPacketTransaction const& endpoint,
    amrex::Geometry const& geometry, NativeRZPhotonWeightProposalOptions const& options,
    std::vector<NativePhotonPathCarry> const& clocks)
{
    NativeRZPhotonWeightProposalResult result;
    if (!ValidateNativeRZPhotonWeightOriginalDirectionControls(options))
    { result.reason = "invalid native original-direction certificate controls"; return result; }
    result.reason = "invalid private photon provenance or collective controls";
    if (!endpoint.ValidateSnapshotFor(live) || !SameOptions(options)) { return result; }
    result.photons = std::make_unique<PhotonPacketTransaction>(live, endpoint);
    result.valid = true;
    result.reason = "unsupported endpoint/gray/clock/precision/geometry: unchanged";
    bool supported = options.fresh_outer_endpoint && options.fixed_gray_cohort && clocks.empty() &&
        std::numeric_limits<amrex::Real>::digits == 53 &&
        std::numeric_limits<amrex::ParticleReal>::digits == 53;
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(geometry);
    supported = false;
    All(supported);
    return result;
#else
    auto const& input = endpoint.Particles();
    if (!All(options.maximum_blocks_per_cohort > 0 &&
        options.maximum_elimination_attempts >= 1 && options.maximum_elimination_attempts <= 3 &&
        (!options.require_well_resolved_direction || options.maximum_elimination_attempts == 1) &&
        options.maximum_captured_blocks_per_rank <= 64 &&
        (options.collect_diagnostics || options.maximum_captured_blocks_per_rank == 0) &&
        (options.canonical_cell == std::numeric_limits<std::uint64_t>::max() ||
         options.canonical_cell < static_cast<std::uint64_t>(geometry.Domain().numPts()))))
    { result.valid = false; result.reason = "invalid canonical pre-draw scope"; return result; }
    auto const& native_geometry = input.Geom(0);
    supported = supported && live.AmIA<PhysicalSpecies::photon>() &&
        input.finestLevel() == 0 && (geometry.Coord() == 0 || geometry.Coord() == 1) &&
        geometry.Domain() == native_geometry.Domain() &&
        geometry.Coord() == native_geometry.Coord();
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        supported = supported && geometry.ProbLo(d) == native_geometry.ProbLo(d) &&
            geometry.ProbHi(d) == native_geometry.ProbHi(d) &&
            geometry.isPeriodic(d) == native_geometry.isPeriodic(d);
    }
    for (auto const& [key, tile] : input.GetParticles(0))
    {
        amrex::ignore_unused(key);
        supported = supported && tile.numNeighborParticles() == 0;
    }
    if (!All(supported)) { return result; }
    std::vector<std::uint64_t> ids, provisional, draw_keys;
    bool valid = photon_replay_detail::CapturePhotonRoutingKeys(input, live,
        NativePhotonPathSnapshotPolicy::KeyedProvisional, ids, provisional, draw_keys);
    std::array<int, 2> key_components{-1, -1};
    valid = FindPhotonDrawKeyComponents(input.GetRealSoANames(), key_components) && valid;
    std::array<int, 3> census{-1, -1, -1};
    for (int d = 0; d < 3; ++d)
    {
        auto const& names = input.GetRealSoANames();
        auto const found = std::find(names.begin(), names.end(),
            std::string("radiation_diffusion_census_u") + "xyz"[d] + "0");
        if (found != names.end()) { census[d] = static_cast<int>(found - names.begin()); }
    }
    valid = valid && ((census[0] < 0 && census[1] < 0 && census[2] < 0) ||
                     (census[0] >= 0 && census[1] >= 0 && census[2] >= 0));
    auto const captured = photon_replay_detail::CaptureParticles(input);
    std::vector<Ray> rays;
    std::vector<PhotonAuditIdentity> routing_keys, stochastic_keys;
    std::map<std::vector<std::uint64_t>, std::vector<std::size_t>> cohorts;
    std::size_t traversal = 0;
    for (std::size_t t = 0; t < captured.tiles.size() && valid; ++t)
    {
        auto const& tile = captured.tiles[t];
        for (long p = 0; p < tile.count; ++p, ++traversal)
        {
            auto get = [&] (int d) { return tile.real_data[d * tile.count + p]; };
            Ray ray;
            ray.tile = t; ray.index = p;
            ray.r = get(PIdx::r); ray.theta = get(PIdx::theta);
            ray.ux = get(PIdx::ux); ray.uy = get(PIdx::uy); ray.uz = get(PIdx::uz);
            ray.row.weight = get(PIdx::w);
            valid = traversal < ids.size() && traversal < provisional.size() &&
                traversal < draw_keys.size() && std::isfinite(ray.row.weight) &&
                ray.row.weight >= 0 && std::isfinite(ray.theta) &&
                std::isfinite(ray.ux + ray.uy + ray.uz) &&
                ray.ux * ray.ux + ray.uy * ray.uy + ray.uz * ray.uz > 0;
            if (!valid) { break; }
            bool const explicit_key = key_components[0] >= 0 &&
                (get(key_components[0]) != 0 || get(key_components[1]) != 0);
            ray.row.key = {explicit_key ? 1 : 0, draw_keys[traversal]};
            routing_keys.emplace_back(provisional[traversal] == 0 ? 0 : 1,
                provisional[traversal] == 0 ? ids[traversal] : provisional[traversal]);
            stochastic_keys.push_back(ray.row.key);
            auto const values = NativeValues(ray, 1);
            for (auto value : values)
            {
                ray.row.features.push_back(static_cast<double>(value));
                valid = valid && std::isfinite(value) &&
                    std::isfinite(ray.row.features.back());
            }
            amrex::IntVect cell;
            valid = Cell(geometry, ray, get(PIdx::z), cell) && valid;
            valid = valid && tile.key.first >= 0 &&
                tile.key.first < input.ParticleBoxArray(0).size() &&
                input.ParticleBoxArray(0)[tile.key.first].contains(cell);
            if (!valid) { break; }
            bool collided = false;
            for (int d = 0; d < 3 && census[0] >= 0; ++d)
            {
                valid = valid && std::isfinite(get(census[d]));
                collided = collided || get(PIdx::ux + d) != get(census[d]);
            }
            std::vector<std::uint64_t> cohort{
                static_cast<std::uint64_t>(geometry.Domain().index(cell)),
                static_cast<std::uint64_t>(provisional[traversal] != 0),
                static_cast<std::uint64_t>(collided)};
            // Opaque attributes could affect unresolved physics: only combine
            // equal bit payloads. Actual energy is NOT a cohort discriminator.
            for (int d = PIdx::nattribs; d < input.NumRealComps(); ++d)
            {
                if (d == key_components[0] || d == key_components[1] ||
                    std::find(census.begin(), census.end(), d) != census.end()) { continue; }
                cohort.push_back(std::bit_cast<std::uint64_t>(static_cast<double>(get(d))));
            }
            for (int d = 0; d < input.NumIntComps(); ++d)
            { cohort.push_back(static_cast<std::uint64_t>(tile.int_data[d * tile.count + p])); }
            cohorts[cohort].push_back(rays.size());
            rays.push_back(std::move(ray));
        }
    }
    valid = valid && traversal == ids.size();
    // Both exact audits are collective and separate routing from RNG identity.
    valid = DistributedPhotonIdentitiesUnique(routing_keys, valid);
    valid = DistributedPhotonIdentitiesUnique(stochastic_keys, valid);
    if (!valid)
    { result.valid = false; result.reason = "invalid native schema/pose/identity"; return result; }
    auto& diagnostics = result.diagnostics;
    diagnostics.collected = options.collect_diagnostics;
    if (diagnostics.collected)
    {
        diagnostics.input_records = static_cast<amrex::Long>(rays.size());
        diagnostics.positive_records = static_cast<amrex::Long>(std::count_if(
            rays.begin(), rays.end(), [] (auto const& ray) { return ray.row.weight > 0; }));
        diagnostics.cohorts = static_cast<amrex::Long>(cohorts.size());
    }
    std::vector<Pair> pairs;
    NativePhotonWeightInventory original_operands{};
    NativePhotonWeightInventory maximum_per_energy{};
    for (auto const& ray : rays)
    {
        auto const values = NativeValues(ray, ray.row.weight);
        auto const unit = NativeValues(ray, 1);
        for (int d = 0; d < NativePhotonWeightMoments; ++d)
        {
            original_operands[d] += std::abs(values[d]);
            if (d != 0)
            { maximum_per_energy[d] = std::max(maximum_per_energy[d],
                std::abs(unit[d]) / unit[1]); }
        }
    }
    for (auto& [cohort, members] : cohorts)
    {
        if (options.canonical_cell != std::numeric_limits<std::uint64_t>::max() &&
            cohort.front() != options.canonical_cell) { continue; }
        if (diagnostics.collected) { ++diagnostics.visited_cohorts; }
        std::sort(members.begin(), members.end(), [&] (auto a, auto b)
                  { return rays[a].row.key < rays[b].row.key; });
        for (std::size_t start = 0; start < members.size() &&
             start / 14 < options.maximum_blocks_per_cohort; start += 14)
        {
            Pair pair;
            auto const end = std::min(start + 14, members.size());
            pair.rays.assign(members.begin() + start, members.begin() + end);
            std::vector<PhotonWeightRecord> rows;
            for (auto index : pair.rays) { rows.push_back(rays[index].row); }
            NativePhotonWeightBlockCapture* capture = nullptr;
            std::size_t positive = 0;
            if (diagnostics.collected)
            {
                positive = static_cast<std::size_t>(std::count_if(
                    rows.begin(), rows.end(), [] (auto const& row) { return row.weight > 0; }));
                ++diagnostics.scheduled_blocks;
                ++diagnostics.active_size[positive];
                if (positive >= 2)
                {
                    diagnostics.eligible_positive_rays += static_cast<amrex::Long>(positive);
                    if (diagnostics.captures.size() < options.maximum_captured_blocks_per_rank)
                    {
                        diagnostics.captures.emplace_back();
                        capture = &diagnostics.captures.back();
                        capture->cohort = cohort; capture->block = start / 14;
                        capture->rows = rows;
                        capture->certificate_kind = options.use_original_direction_certificate ?
                            NativePhotonWeightCertificateKind::OriginalDirection :
                            (options.require_well_resolved_direction ?
                                NativePhotonWeightCertificateKind::RetainedConditioning :
                                NativePhotonWeightCertificateKind::None);
                        for (auto index : pair.rays)
                        {
                            auto const& ray = rays[index];
                            capture->unit_poses.push_back({static_cast<double>(ray.r),
                                static_cast<double>(ray.theta), static_cast<double>(ray.ux),
                                static_cast<double>(ray.uy), static_cast<double>(ray.uz)});
                        }
                        for (std::size_t p = 0; p < rows.size(); ++p)
                        { if (rows[p].weight > 0) { capture->active.push_back(p); } }
                    }
                }
            }
            auto record_stage = [&] (NativePhotonWeightDiagnosticStage stage)
            {
                if (!diagnostics.collected) { return; }
                ++diagnostics.stages[static_cast<std::size_t>(stage)];
                if (capture) { capture->stage = stage; }
            };
            std::vector<std::vector<double> const*> outcome_weights;
            if (options.maximum_elimination_attempts > 1)
            {
                pair.tree = PrepareConservativePhotonWeightTree(
                    rows, options.maximum_elimination_attempts);
                if (!pair.tree.valid || pair.tree.nodes.empty())
                {
                    record_stage(positive < 2 ? NativePhotonWeightDiagnosticStage::TooFewPositive :
                                 NativePhotonWeightDiagnosticStage::RepeatedTreeRefused);
                    continue;
                }
                for (auto const& leaf : pair.tree.leaves)
                { outcome_weights.push_back(&leaf.weights); }
            }
            else
            {
                // Preserve the qualified default1 endpoint preparation exactly.
                std::vector<std::size_t> active;
                for (std::size_t p = 0; p < rows.size(); ++p)
                { if (rows[p].weight > 0) { active.push_back(p); } }
                if (active.size() < 2)
                { record_stage(NativePhotonWeightDiagnosticStage::TooFewPositive); continue; }
                PhotonWeightOriginalRowBasis basis;
                if (options.direction_method ==
                    PhotonWeightDirectionMethod::RankRevealingOriginalRows)
                {
                    basis = SelectPhotonWeightOriginalRowBasis(rows, active);
                    if (capture) { capture->selected_rows = basis.rows; }
                    if (!basis.valid)
                    { record_stage(NativePhotonWeightDiagnosticStage::OriginalRowBasisRefused); continue; }
                }
                photon_weight_detail::NullDirectionPivots retained;
                auto const h = photon_weight_detail::NullDirection(rows, active,
                    options.require_well_resolved_direction ? &retained : nullptr,
                    options.direction_method == PhotonWeightDirectionMethod::Legacy ?
                        nullptr : &basis.rows);
                if (capture)
                {
                    capture->retained_rows = retained.rows;
                    capture->retained_columns = retained.columns;
                    capture->direction = h;
                }
                if (h.empty())
                { record_stage(NativePhotonWeightDiagnosticStage::NullDirectionRefused); continue; }
                if (options.require_well_resolved_direction)
                {
                    if (options.use_original_direction_certificate)
                    {
                        bool formation_valid = true;
                        std::vector<std::vector<long double>> coefficient_radius;
                        for (std::size_t p = 0; p < rows.size(); ++p)
                        {
                            auto const& ray = rays[pair.rays[p]];
                            NativePhotonWeightUnitPose const pose{static_cast<double>(ray.r),
                                static_cast<double>(ray.theta), static_cast<double>(ray.ux),
                                static_cast<double>(ray.uy), static_cast<double>(ray.uz)};
                            auto const formation =
                                BoundNativeRZPhotonWeightFeatureFormation(pose, rows[p]);
                            formation_valid = formation_valid && formation.valid;
                            coefficient_radius.emplace_back(formation.absolute_radius.begin(),
                                                            formation.absolute_radius.end());
                            if (capture) { capture->feature_formation.push_back(formation); }
                        }
                        if (capture)
                        {
                            capture->formation_observed = true;
                            capture->formation_valid = formation_valid;
                        }
                        if (!formation_valid)
                        {
                            record_stage(NativePhotonWeightDiagnosticStage::CertificateInvalid);
                            continue;
                        }
                        auto const certificate = CertifyPhotonWeightOriginalDirection(
                            rows, active, retained, h, coefficient_radius);
                        if (capture)
                        {
                            capture->original_certificate = certificate;
                            capture->original_certificate_observed = true;
                        }
                        if (!certificate.admitted)
                        {
                            record_stage(certificate.valid ?
                                NativePhotonWeightDiagnosticStage::CertificateUnresolved :
                                NativePhotonWeightDiagnosticStage::CertificateInvalid);
                            continue;
                        }
                    }
                    else
                    {
                        auto const certificate =
                            CertifyPhotonWeightDirection(rows, active, retained);
                        if (capture)
                        {
                            capture->certificate = certificate;
                            capture->certificate_observed = true;
                        }
                        if (!certificate.admitted)
                        {
                            record_stage(certificate.valid ?
                                NativePhotonWeightDiagnosticStage::CertificateUnresolved :
                                NativePhotonWeightDiagnosticStage::CertificateInvalid);
                            continue;
                        }
                    }
                }
                long double plus = std::numeric_limits<long double>::infinity(), minus = plus;
                for (std::size_t p = 0; p < h.size(); ++p)
                {
                    if (h[p] > 0) { plus = std::min(plus, rows[active[p]].weight / h[p]); }
                    if (h[p] < 0) { minus = std::min(minus, -rows[active[p]].weight / h[p]); }
                }
                pair.probability = minus / (plus + minus);
                if (capture)
                { capture->plus = plus; capture->minus = minus; capture->probability = pair.probability; }
                // Restrict endpoint enumeration to two branches reachable on the
                // 53-bit counter lattice. Extreme ratios safely remain unchanged.
                if (!(pair.probability >= 0x1.0p-52L &&
                      pair.probability <= 1 - 0x1.0p-52L))
                { record_stage(NativePhotonWeightDiagnosticStage::ProbabilityLatticeRefused); continue; }
                PhotonWeightProposalOptions core;
                core.target = active.size() - 1;
                core.direction_method = options.direction_method;
                pair.endpoints[0] = ProposeConservativePhotonWeights(rows, core,
                    [] (std::size_t) { return 0.; });
                pair.endpoints[1] = ProposeConservativePhotonWeights(rows, core,
                    [] (std::size_t) { return std::nextafter(1., 0.); });
                if (capture)
                {
                    for (int side = 0; side < 2; ++side)
                    {
                        capture->endpoint_valid[side] = pair.endpoints[side].valid;
                        capture->endpoint_eliminations[side] = pair.endpoints[side].eliminations;
                        capture->endpoint_reasons[side] = pair.endpoints[side].reason;
                    }
                }
                if (!pair.endpoints[0].valid || !pair.endpoints[1].valid ||
                    pair.endpoints[0].eliminations != 1 || pair.endpoints[1].eliminations != 1)
                { record_stage(NativePhotonWeightDiagnosticStage::PureEndpointRefused); continue; }
                for (auto const& prepared : pair.endpoints)
                { outcome_weights.push_back(&prepared.weights); }
            }
            NativePhotonWeightInventory worst{}, operands{};
            for (auto const* endpoint_weights : outcome_weights)
            {
                NativePhotonWeightInventory change{}, scale{};
                for (std::size_t p = 0; p < rows.size(); ++p)
                {
                    auto const& ray = rays[pair.rays[p]];
                    auto const before = NativeValues(ray, rows[p].weight);
                    auto const after = NativeValues(ray, (*endpoint_weights)[p]);
                    for (int d = 0; d < NativePhotonWeightMoments; ++d)
                    {
                        change[d] += after[d] - before[d];
                        scale[d] += std::abs(after[d]) + std::abs(before[d]);
                    }
                }
                for (int d = 0; d < NativePhotonWeightMoments; ++d)
                {
                    worst[d] = std::max(worst[d], std::abs(change[d]));
                    operands[d] = std::max(operands[d], scale[d]);
                }
            }
            auto const product = 256.L * (rows.size() + 1) *
                                 std::numeric_limits<long double>::epsilon();
            auto const gamma = product / (1 - product);
            for (int d = 0; d < NativePhotonWeightMoments; ++d)
            { result.all_outcome_change_bound[d] += worst[d] + gamma * operands[d]; }
            pair.bucket = Bucket(cohort, start / 14);
            result.weight_expectation_roundoff_bound +=
                options.maximum_elimination_attempts == 1 ?
                    std::max(pair.endpoints[0].weight_expectation_roundoff_bound,
                             pair.endpoints[1].weight_expectation_roundoff_bound) :
                    pair.tree.weight_expectation_roundoff_bound;
            pairs.push_back(std::move(pair));
            record_stage(NativePhotonWeightDiagnosticStage::Prepared);
        }
    }
    for (auto const& pair : pairs)
    { result.preflight_pairs += pair.tree.nodes.empty() ? 1 : pair.tree.nodes.size(); }
    amrex::ParallelDescriptor::ReduceLongSum(result.preflight_pairs);
    FinalizeDiagnostics(diagnostics);
    result.reason = "irreducible or unsupported cohorts: unchanged";
    if (result.preflight_pairs == 0) { return result; }
    Sum(result.all_outcome_change_bound);
    Sum(original_operands);
    Maximum(maximum_per_energy);
    amrex::Long global_records = static_cast<amrex::Long>(rays.size());
    amrex::ParallelDescriptor::ReduceLongSum(global_records);
    auto const product = 256.L * (global_records +
        amrex::ParallelDescriptor::NProcs() + result.preflight_pairs + 1) *
        std::numeric_limits<long double>::epsilon();
    bool fits = product < .25L;
    auto const gamma = product / (1 - product);
    // Signed components can gain large ABSOLUTE operands while their signed
    // moments remain conserved. Bound every possible post-proposal operand
    // by positive E, with an actual immutable coefficient/E maximum (physical
    // cLz includes its radius lever). Never use old signed-component operands
    // alone as a surrogate for all selected-outcome summation conditioning.
    auto const post_energy = (original_operands[1] + result.all_outcome_change_bound[1]) /
                             (1 - gamma);
    auto const post_number = (original_operands[0] + result.all_outcome_change_bound[0]) /
                             (1 - gamma);
    auto const coefficient_product = 128 * std::numeric_limits<long double>::epsilon();
    for (int d = 0; d < NativePhotonWeightMoments; ++d)
    {
        auto& envelope = result.all_outcome_change_bound[d];
        auto& operands = result.all_outcome_operand_bound[d];
        operands = (d == 0 ? post_number : maximum_per_energy[d] * post_energy) /
                   (1 - coefficient_product);
        if (operands > 0)
        { operands = std::nextafter(operands, std::numeric_limits<long double>::infinity()); }
        // Sum and MPI-reduction arithmetic, including untouched native terms
        // whose traversal summation changes after compaction. Outward inflate.
        envelope = (envelope + gamma * (original_operands[d] + operands + envelope)) /
                   (1 - gamma);
        if (envelope > 0)
        { envelope = std::nextafter(envelope, std::numeric_limits<long double>::infinity()); }
        auto const& guard = options.guards[d];
        fits = fits && std::isfinite(envelope);
        if (!guard.required) { continue; }
        auto const headroom = guard.unchanged_absolute_allowance -
                              std::abs(guard.original_residual);
        auto const subtraction_guard = 4 * std::numeric_limits<long double>::epsilon() *
            (guard.unchanged_absolute_allowance + std::abs(guard.original_residual));
        fits = fits && headroom >= subtraction_guard &&
               envelope <= headroom - subtraction_guard;
    }
    if (!All(fits))
    {
        if (diagnostics.collected)
        { diagnostics.native_headroom_refused_blocks = diagnostics.stages[
            static_cast<std::size_t>(NativePhotonWeightDiagnosticStage::Prepared)]; }
        result.reason = "both-outcome native headroom preflight: unchanged";
        return result;
    }
    // All physical guards and collective identity/schema checks are complete.
    // No acceptance branch, target filtering, or rollback is allowed below.
    std::vector<double> weights;
    std::vector<bool> admitted(rays.size(), false);
    for (auto const& ray : rays) { weights.push_back(ray.row.weight); }
    for (auto const& pair : pairs)
    {
        std::vector<double> selected;
        if (pair.tree.nodes.empty())
        {
            auto const side = Counter(pair, rays, options) < pair.probability ? 0 : 1;
            selected = pair.endpoints[side].weights;
            ++result.counter_choices;
        }
        else
        {
            auto chosen = SampleConservativePhotonWeightTree(pair.tree,
                [&] (std::size_t path) { return Counter(pair, rays, options, path); });
            // Counter is a total pure53-bit function; the preflighted tree has
            // no rejection/rollback/target branches during this traversal.
            selected = std::move(chosen.weights);
            result.counter_choices += chosen.choices;
        }
        for (std::size_t p = 0; p < pair.rays.size(); ++p)
        {
            weights[pair.rays[p]] = selected[p];
            admitted[pair.rays[p]] = true;
        }
    }
    std::size_t offset = 0;
    auto& candidate = result.photons->Particles();
    for (auto const& before : captured.tiles)
    {
        std::vector<long> keep;
        for (long p = 0; p < before.count; ++p)
        {
            if (!admitted[offset + p] || weights[offset + p] != 0) { keep.push_back(p); }
            else { ++result.removed; }
        }
        auto& tile = candidate.GetParticles(0).at(before.key);
        tile.resize(keep.size());
        auto& soa = tile.GetStructOfArrays();
        for (int d = 0; d < candidate.NumRealComps(); ++d)
        {
            amrex::Vector<amrex::ParticleReal> values;
            for (auto p : keep)
            { values.push_back(d == PIdx::w ? weights[offset + p] :
                before.real_data[d * before.count + p]); }
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, values.begin(), values.end(),
                             soa.GetRealData(d).begin());
        }
        for (int d = 0; d < candidate.NumIntComps(); ++d)
        {
            amrex::Vector<int> values;
            for (auto p : keep) { values.push_back(before.int_data[d * before.count + p]); }
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, values.begin(), values.end(),
                             soa.GetIntData(d).begin());
        }
        amrex::Vector<std::uint64_t> values;
        for (auto p : keep) { values.push_back(before.idcpu[p]); }
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, values.begin(), values.end(),
                         soa.GetIdCPUData().begin());
        offset += before.count;
    }
    amrex::Gpu::streamSynchronize();
    for (std::size_t p = 0; p < rays.size(); ++p)
    {
        auto const before = NativeValues(rays[p], rays[p].row.weight);
        auto const after = NativeValues(rays[p], weights[p]);
        for (int d = 0; d < NativePhotonWeightMoments; ++d)
        { result.actual_change[d] += after[d] - before[d]; }
    }
    Sum(result.actual_change);
    NativePhotonWeightInventory expectation{};
    expectation[0] = result.weight_expectation_roundoff_bound;
    Sum(expectation);
    result.weight_expectation_roundoff_bound = expectation[0];
    amrex::ParallelDescriptor::ReduceLongSum(result.removed);
    amrex::ParallelDescriptor::ReduceLongSum(result.counter_choices);
    result.applied = true;
    result.reason = "private bounded both-outcome native weight proposal";
    return result;
#endif
}

} // namespace warpx::radiation
