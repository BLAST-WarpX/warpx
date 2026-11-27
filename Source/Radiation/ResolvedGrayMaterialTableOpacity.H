/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_RESOLVEDGRAYMATERIALTABLEOPACITY_H_
#define WARPX_RADIATION_RESOLVEDGRAYMATERIALTABLEOPACITY_H_

#include "MaterialOpacityTable.H"
#include "MaterialOpacityTableResponse.H"
#include "ResolvedMaterialOpacity.H"

#include <AMReX_MultiFab.H>

#include <array>
#include <string>

namespace warpx::radiation::detail
{
enum class ResolvedGrayMaterialOpacityAxisPolicy : int
{
    StrictInterior = 0,
    ExactRawNodes = 1
};

enum class ResolvedGrayMaterialOpacityChannelPolicy : int
{
    PositiveLogMatched = 0,
    ExactZeroLinearAllScatteringZero = 1
};

struct GrayOpacityArithmeticBox
{
    long double lo = 0, hi = 0;
};

/** Conditional binary64/long-double enclosure of the authentic positive-log
 * table response. Requires round-to-nearest, no unsafe reassociation, and
 * <=1 ULP host log/exp (including long-double endpoint exp). The propagated
 * boxes are not fitted to channel differences. The optional 2^-26 relative
 * width ceiling is an engineering admission policy, not a physics tolerance.
 * A matched represented closure has ideal channel defect <=2*bound_upper;
 * it does NOT assert exact equality of independently interpolated channels. */
struct GrayOpacityArithmeticResult
{
    bool valid = false, resolved = false, matched = false;
    std::string reason;
    MaterialOpacityTableResponse response;
    double recovered_alpha = 0;
    std::array<GrayOpacityArithmeticBox, 3> ideal{}, floating{};
    std::array<long double, 3> radius{};
    long double residual_upper = 0, bound_upper = 0, relative_bound_upper = 0;
    std::array<long double, 2> gap_lower{};
    std::array<GrayOpacityArithmeticBox, 2> ideal_fraction{};
    std::array<int, 2> selected{-1, -1};
    std::array<int, 2> exact_axis_node{-1, -1};
};

[[nodiscard]] GrayOpacityArithmeticResult BoundPositiveLogGrayOpacityResponse (
    MaterialOpacityTableExecutor const& table, double mass_density, double temperature);

/** Derives raw-axis identities only from the actual initialized loaded owner.
 * ExactRawNodes correlates an ideal fraction only when BOTH query/raw bits and
 * authentic query/stored logs agree. All other queries keep strict-interior
 * admission. The authentic floating affine graph, including upper-node
 * cancellation, exp, density scaling and c*P/c, is never replaced by a corner.
 * The owner must remain immutable through the call. HDF5-disabled builds
 * return an unsupported-owner result without referencing hostExecutor(). */
[[nodiscard]] GrayOpacityArithmeticResult BoundPositiveLogGrayOpacityResponse (
    MaterialOpacityTable const& table, double mass_density, double temperature,
    ResolvedGrayMaterialOpacityAxisPolicy policy);

/** Loaded-owner-only schema-1.1 linear certificate for one complete gray group.
 * Every packed node mask must be exactly ScatteringZero, with canonical +0 S
 * and unmasked normal-positive, bit-canonical P/emission and R. The ideal uses
 * the original represented linear mass opacities and true-log coordinate
 * weights; the floating box encloses the authentic linear interpolation,
 * density scaling and c*P/c graph. S boxes/radius are exactly zero, never an
 * exp-underflow surrogate or R-P reconstruction. Existing axis policy and the
 * full derived bound/2^-26 resolution policy remain in force. */
[[nodiscard]] GrayOpacityArithmeticResult BoundExactZeroScatteringGrayOpacityResponse (
    MaterialOpacityTable const& table, double mass_density, double temperature,
    ResolvedGrayMaterialOpacityAxisPolicy policy);

/** Private CPU RZ matched one-gray LTE provider. Default OFF. No Source,
 * history, constructor or normal checkpoint guard is activated by this type.
 * Units: photon_energy [J], mass_per_electron [kg], density threshold [m^-3].
 * The physical mass-per-electron must be audited against the sole fixed-charge
 * native carrier before the cell-sample seam is called. */
struct ResolvedGrayMaterialTableOpacityOptions
{
    bool enabled = false;
    amrex::Real photon_energy = 0;
    amrex::Real mass_per_electron = 0;
    amrex::Real minimum_electron_density = 0;
    amrex::Real thermodynamic_floor = 0;
    ResolvedGrayMaterialOpacityAxisPolicy axis_policy =
        ResolvedGrayMaterialOpacityAxisPolicy::StrictInterior;
    ResolvedGrayMaterialOpacityChannelPolicy channel_policy =
        ResolvedGrayMaterialOpacityChannelPolicy::PositiveLogMatched;
};

struct ResolvedGrayMaterialTableOpacityReport
{
    amrex::Real maximum_channel_residual = 0;
    amrex::Real maximum_channel_bound = 0;
};

/** Existing immutable cell ne [m^-3]/Te [K] inputs, including persisted native
 * accepted-left samples at restart. Never regather live deposition here.
 * The actual loaded owner must have one complete [0,+infinity] gray group and
 * matching representative energy. PositiveLogMatched is unchanged by default;
 * ExactZeroLinearAllScatteringZero separately requires the schema-1.1 global
 * exact-S-zero contract above. Existing strict-domain table
 * response and conditioned matched-channel arithmetic are used. StrictInterior
 * is unchanged by default; ExactRawNodes additionally admits proven original
 * raw-axis identities, not rounded-log neighbors. Other near-knot or unresolved
 * arithmetic rejects, distinct from out-of-domain rejection.
 * Outputs [m^-1] plus report publish only after complete collective acceptance;
 * all output/caller fields remain unchanged on failure. No draw/counter/ID is
 * read or modified. Table ownership/contents must outlive and remain immutable
 * through this call. */
[[nodiscard]] bool TryFillResolvedGrayMaterialTableOpacity (
    amrex::MultiFab& scattering, amrex::MultiFab& absorption,
    amrex::MultiFab const& electron_density, amrex::MultiFab const& electron_temperature,
    amrex::Geometry const& geometry, MaterialOpacityTable const& table,
    ResolvedGrayMaterialTableOpacityOptions const& options, std::string& reason,
    ResolvedGrayMaterialTableOpacityReport* report = nullptr);

struct ResolvedGrayMaterialTableOpacitySnapshot
{
    bool valid = false;
    std::string reason;
    amrex::MultiFab electron_density, electron_temperature, scattering, absorption;
    ResolvedGrayMaterialTableOpacityReport report;
};

/** Actual immutable nodal raw charge [C/m^3], Te [K] and frozen unit-ion charge
 * fields with the TRUE EOS executor/selector. Reuses the existing native
 * four-corner ne/Cv-weighted Te gather, followed by the cell-sample seam above.
 * Requires actual single selected TABLE material0; never rewrites its executor.
 * mass_per_electron must equal qe*eos.m_mass_per_physical_charge[0].
 * Rejection returns no partially published sample/coefficient fields. */
[[nodiscard]] ResolvedGrayMaterialTableOpacitySnapshot
TrySnapshotResolvedGrayMaterialTableOpacity (
    amrex::MultiFab const& raw_charge_density, amrex::MultiFab const& nodal_temperature,
    ResolvedMaterialChargeFields const& unit_ion_charge, ElectronThermodynamicsExecutor const& eos,
    amrex::Geometry const& geometry, MaterialOpacityTable const& table,
    ResolvedGrayMaterialTableOpacityOptions const& options);

/** Encoding selector for COEFFICIENT INFRASTRUCTURE ONLY. This is not a
 * runtime/history/field policy and does not weaken either matched provider. */
enum class IndependentGrayMaterialOpacityEncoding : int
{
    PositiveLogPositiveScattering = 0,
    ExactZeroLinearAllScatteringZero = 1
};

struct ResolvedIndependentGrayMaterialTableOpacityOptions
{
    bool enabled = false;
    /** Exact ne=Te=0 has no table query and canonical zero R/P/S. Native
     * consumers must also prove zero raw/unit material corners and K-only
     * classification. Small positive density is never rounded to vacuum. */
    bool allow_exact_vacuum = false;
    amrex::Real photon_energy = 0;
    amrex::Real mass_per_electron = 0;
    amrex::Real minimum_electron_density = 0;
    amrex::Real thermodynamic_floor = 0;
    ResolvedGrayMaterialOpacityAxisPolicy axis_policy =
        ResolvedGrayMaterialOpacityAxisPolicy::StrictInterior;
    IndependentGrayMaterialOpacityEncoding encoding =
        IndependentGrayMaterialOpacityEncoding::PositiveLogPositiveScattering;
};

/** One caller-owned coefficient epoch. All three cell/ghost fields have
 * units [1/m]. R is authentic Rosseland transport, S is authentic TRUE
 * scattering, and P is the authentic absorption-rate/c graph. No S=R-P,
 * clipping, R=S+P substitution, runtime registry or physics admission exists.
 * A consumer must retain this owner immutably while using its fields. */
struct ResolvedIndependentGrayMaterialOpacityFields
{
    amrex::MultiFab rosseland_transport, true_scattering, proper_absorption;
};

struct ResolvedIndependentGrayMaterialTableOpacityReport
{
    // Arithmetic-channel order R, recovered P, S; not a closure defect bound.
    std::array<amrex::Real, 3> maximum_channel_radius{};
    // Existing resolution metric: SUM of radii / MAX(R,P,S), not a
    // separately conditioned relative R/P/S accuracy or D-inversion bound.
    amrex::Real maximum_relative_arithmetic_bound = 0;
};

/** Separate CPU RZ/true TABLE coefficient seam. Reuses the EXISTING loaded
 * owner enclosure functions' valid+resolved result, including their unchanged
 * normal-positive, axis/domain and global sum(radius)/max(R,P,S) <= 2^-26
 * resolution guards, without requiring
 * R=P+S. The old two-field entry point remains matched-only. Raw axes,
 * encodings, channel/mask data, material metadata, controls and report-pointer
 * presence agree collectively, including empty ranks. Failure preserves all
 * three caller FABs/ghosts, report and input fields exactly. No stochastic
 * draws, material/photon owners, force/work, field advance or checkpoint is
 * touched. A future consumer must separately prove its channel/inversion
 * conditioning, especially R much smaller than P/S. mass_per_electron still
 * requires the native fixed-charge audit. */
[[nodiscard]] bool TryFillResolvedIndependentGrayMaterialTableOpacity (
    ResolvedIndependentGrayMaterialOpacityFields& output,
    amrex::MultiFab const& electron_density, amrex::MultiFab const& electron_temperature,
    amrex::Geometry const& geometry, MaterialOpacityTable const& table,
    ResolvedIndependentGrayMaterialTableOpacityOptions const& options, std::string& reason,
    ResolvedIndependentGrayMaterialTableOpacityReport* report = nullptr);

struct ResolvedIndependentGrayMaterialTableOpacitySnapshot
{
    bool valid = false;
    std::string reason;
    amrex::MultiFab electron_density, electron_temperature;
    ResolvedIndependentGrayMaterialOpacityFields coefficients;
    ResolvedIndependentGrayMaterialTableOpacityReport report;
};

/** Actual native TRUE-selected sole TABLE ne/Cv-weighted Te snapshot plus
 * three independent coefficients. The original four-corner gather and EOS
 * selector are reused unchanged. Failure publishes no partial epoch and never
 * mutates raw charge/T/unit fields. Coefficients alone do NOT qualify distinct
 * channels for an existing History/Interval/D/K interface or runtime. */
[[nodiscard]] ResolvedIndependentGrayMaterialTableOpacitySnapshot
TrySnapshotResolvedIndependentGrayMaterialTableOpacity (
    amrex::MultiFab const& raw_charge_density, amrex::MultiFab const& nodal_temperature,
    ResolvedMaterialChargeFields const& unit_ion_charge, ElectronThermodynamicsExecutor const& eos,
    amrex::Geometry const& geometry, MaterialOpacityTable const& table,
    ResolvedIndependentGrayMaterialTableOpacityOptions const& options);
} // namespace warpx::radiation::detail
#endif
