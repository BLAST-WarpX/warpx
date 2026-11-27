/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_NATIVE_RZ_PHOTON_WEIGHT_PROPOSAL_H_
#define WARPX_NATIVE_RZ_PHOTON_WEIGHT_PROPOSAL_H_

#include "ConservativePhotonWeightConditioning.H"
#include "ConservativePhotonWeightOriginalDirectionCertificate.H"
#include "ConservativePhotonWeightOriginalRowBasis.H"
#include "NativePhotonPathReplay.H"
#include "NativeRZPhotonWeightFeatureFormation.H"
#include "PhotonPacketTransaction.H"

#include <array>
#include <cstdint>
#include <memory>
#include <limits>
#include <string>
#include <vector>

namespace warpx::radiation
{
/** Independent physical native accounts, NOT conjugate RZ rate components.
 * Number is statistical photon number; E/cP/pressure have units J; cLz J m.
 * Pressure is in the physical local r/theta/z basis, including all six entries.
 */
inline constexpr int NativePhotonWeightMoments = 13;
using NativePhotonWeightInventory = std::array<long double, NativePhotonWeightMoments>;

struct NativePhotonWeightGuard
{
    bool required = true; // Standalone API remains guarded by default.
    long double original_residual = 0;
    // Outcome-independent lower bound of the UNCHANGED caller allowance.
    // If that allowance depends on record counts/operands, the caller must
    // minimize it over all proposed endpoints, not pass only its old value.
    long double unchanged_absolute_allowance = 0;
};

enum class NativePhotonWeightDiagnosticStage : std::size_t
{
    TooFewPositive, OriginalRowBasisRefused, NullDirectionRefused,
    CertificateInvalid, CertificateUnresolved, ProbabilityLatticeRefused,
    PureEndpointRefused, RepeatedTreeRefused, Prepared, Count
};
inline constexpr std::size_t NativePhotonWeightDiagnosticStages =
    static_cast<std::size_t>(NativePhotonWeightDiagnosticStage::Count);
inline constexpr std::array<char const*, NativePhotonWeightDiagnosticStages>
    NativePhotonWeightDiagnosticStageNames{
        "too_few_positive", "original_row_basis_refused", "null_direction_refused",
        "certificate_invalid", "certificate_unresolved", "probability_lattice_refused",
        "pure_endpoint_refused", "repeated_tree_refused", "prepared"};

enum class NativePhotonWeightCertificateKind : std::size_t
{
    None, RetainedConditioning, OriginalDirection
};
inline constexpr std::array<char const*, 3> NativePhotonWeightCertificateKindNames{
    "none", "retained_conditioning", "original_direction"};

/** Exact OWNER-LOCAL diagnostic copy of an original scheduled block, BEFORE
 * any real counter draw. Keys/cohort words stay uint64; rows retain binary64
 * weights and all original features, including inactive records. Consumers
 * must print doubles/long doubles with round-trip precision or hex, and must
 * never serialize keys through a floating-point/JavaScript-number intermediary.
 */
struct NativePhotonWeightBlockCapture
{
    std::vector<std::uint64_t> cohort;
    std::size_t block = 0;
    std::vector<PhotonWeightRecord> rows;
    // Exact original binary64 r,theta,ux,uy,uz; no feature recomputation.
    std::vector<std::array<double, 5>> unit_poses;
    std::vector<std::size_t> active, selected_rows, retained_rows, retained_columns;
    std::vector<long double> direction;
    PhotonWeightDirectionCertificate certificate;
    bool certificate_observed = false;
    long double plus = 0, minus = 0, probability = 0;
    std::array<bool, 2> endpoint_valid{};
    std::array<std::size_t, 2> endpoint_eliminations{};
    std::array<std::string, 2> endpoint_reasons;
    NativePhotonWeightDiagnosticStage stage = NativePhotonWeightDiagnosticStage::TooFewPositive;
    // Additive policy-specific evidence. Established fields above are unchanged.
    NativePhotonWeightCertificateKind certificate_kind = NativePhotonWeightCertificateKind::None;
    bool formation_observed = false, formation_valid = false;
    std::vector<NativePhotonWeightFeatureFormation> feature_formation;
    PhotonWeightOriginalDirectionCertificate original_certificate;
    bool original_certificate_observed = false;
};

/** ON-only, MPI-SUM GLOBAL integer census; captures remain OWNER-LOCAL.
 * active_size[n] counts scheduled blocks with n positive records (0..14).
 * Sum(active_size)==sum(stages)==scheduled_blocks; eligible_positive_rays is
 * sum(n*active_size[n],n>=2), so bins0+1 equal TooFewPositive. Current key blocks
 * do not overlap. Headroom refusal is a SECOND phase: zero or all Prepared
 * blocks, never part of the mutually exclusive block-stage closure. A history
 * lower-bound refusal precedes the native census; its counts remain zero.
 */
struct NativePhotonWeightProposalDiagnostics
{
    bool collected = false;
    bool history_lower_bound_refused = false; // Collective call flag, not SUM.
    amrex::Long input_records = 0, positive_records = 0;
    amrex::Long cohorts = 0, visited_cohorts = 0;
    amrex::Long scheduled_blocks = 0, eligible_positive_rays = 0;
    std::array<amrex::Long, 15> active_size{};
    std::array<amrex::Long, NativePhotonWeightDiagnosticStages> stages{};
    amrex::Long native_headroom_refused_blocks = 0;
    std::vector<NativePhotonWeightBlockCapture> captures;
};

struct NativeRZPhotonWeightProposalOptions
{
    // Explicit caller contract, never inferred from empty clocks alone.
    bool fresh_outer_endpoint = false;
    bool fixed_gray_cohort = false;
    std::uint64_t seed = 0;
    std::uint64_t outer_interval = 0;
    // Explicit deterministic pre-draw work scope, NOT a realized population
    // target. Sentinel selects all cells; defaults preserve the original pass.
    std::uint64_t canonical_cell = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t maximum_blocks_per_cohort = std::numeric_limits<std::uint64_t>::max();
    // Deterministic bounded work; not a target or a post-draw stop. Default1
    // retains the exact established one-pair path. Repeated mode supports 1..3.
    std::uint64_t maximum_elimination_attempts = 1;
    // Optional retained-system resolution admission BEFORE any draws. Default
    // false preserves the established prototype bit path. Only one attempt
    // is certified; true plus repeated attempts is collectively invalid.
    bool require_well_resolved_direction = false;
    // Optional row selection, including the guarded normal history policy.
    // Requires the unchanged resolution certificate and exactly one attempt.
    PhotonWeightDirectionMethod direction_method = PhotonWeightDirectionMethod::Legacy;
    // Separate default-OFF CPU policy: certify the ACTUAL unchanged original-
    // equation direction with caller-derived fixed-pose formation intervals.
    // Requires RRQR original rows, well-resolved admission and exactly1 attempt.
    // All original13 BOTH endpoints and the unchanged native law remain required.
    bool use_original_direction_certificate = false;
    // Diagnostic-only collective controls. No direction/endpoint/counter work
    // is added or changed. Captures store the first eligible (>=2 positive)
    // blocks per rank, capped at 64 per call, with their actual refusal stage.
    bool collect_diagnostics = false;
    std::uint64_t maximum_captured_blocks_per_rank = 0;
    // Original caller constraints in order N,E,cPx,cPy,cPz,cLz,cPr,
    // Prr,Ptt,Pzz,Prt,Prz,Ptz. No tolerance is created/enlarged here.
    std::array<NativePhotonWeightGuard, NativePhotonWeightMoments> guards{};
};

struct NativeRZPhotonWeightProposalResult
{
    bool valid = false;
    bool applied = false;
    std::string reason;
    std::unique_ptr<PhotonPacketTransaction> photons;
    NativePhotonWeightInventory all_outcome_change_bound{};
    NativePhotonWeightInventory all_outcome_operand_bound{};
    NativePhotonWeightInventory actual_change{}; // Diagnostic, not outcome selection.
    long double weight_expectation_roundoff_bound = 0;
    amrex::Long preflight_pairs = 0; // Complete-tree internal nodes in repeated mode.
    amrex::Long counter_choices = 0;
    amrex::Long removed = 0;
    NativePhotonWeightProposalDiagnostics diagnostics;
};

/** Validate ONLY the new diagnostic controls before a history-floor no-op.
 * No physical guard, direction, counter or owner is evaluated or changed.
 */
[[nodiscard]] bool ValidateNativeRZPhotonWeightDiagnosticControls (
    NativeRZPhotonWeightProposalOptions const&);

/** Validate ONLY the new certificate selector before any history-floor no-op
 * or clone. All-OFF retains old option semantics. ON requires collective CPU
 * RZ/RRQR/well-resolved/one-attempt and supported nearest extended-LD arithmetic.
 */
[[nodiscard]] bool ValidateNativeRZPhotonWeightOriginalDirectionControls (
    NativeRZPhotonWeightProposalOptions const&);

/** PRIVATE, bounded prototype; NOT normal runtime or a population cap.
 * Every canonical cell/cohort block (at most 14 rays) attempts one elimination
 * by default, or explicitly 2..3 via a complete precomputed outcome tree.
 * All represented terminal/prefix native endpoints are remeasured before ANY
 * stochastic choice. Their worst-case envelopes compose over all
 * blocks/ranks and must fit the original caller headroom before drawing.
 * There is no selected-outcome fallback, target filter, or final rejection.
 * Fixed gray means frequency-independent native hazard, not equal lab energy:
 * moving thermal rays retain their distinct actual energies and momenta.
 * Cohorts separate birth/native status, actual collision-census eligibility,
 * and every opaque runtime value; draw keys and census vectors remain on rays.
 * Every survivor retains ALL non-weight native real/int/idcpu bits. Zero rows
 * are compacted only in admitted blocks of the nested candidate. Live owner,
 * original candidate, native allocator, clocks and global RNG are untouched.
 * In-progress clocks/non-gray/unsupported precision or geometry are safe no-ops.
 * All ranks participate, including empty ranks. The caller must retain the
 * certified candidate even if an optional population target was not reached.
 */
[[nodiscard]] NativeRZPhotonWeightProposalResult TryProposeNativeRZPhotonWeights (
    WarpXParticleContainer& live, PhotonPacketTransaction const& endpoint,
    amrex::Geometry const& geometry, NativeRZPhotonWeightProposalOptions const& options,
    std::vector<NativePhotonPathCarry> const& in_progress_clocks = {});

} // namespace warpx::radiation

#endif
