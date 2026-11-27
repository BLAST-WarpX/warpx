/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_RESOLVEDRZTHERMALDIFFUSIONFIELD_H_
#define WARPX_RADIATION_RESOLVEDRZTHERMALDIFFUSIONFIELD_H_

#include "AsymptoticDiffusionField.H"
#include "ResolvedGrayMaterialTableOpacity.H"

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <memory>
#include <string>

namespace warpx::radiation
{
struct ResolvedRZThermalDiffusionFieldOptions
{
    AsymptoticDiffusionFieldOptions linear;
    int max_picard_iterations = 30;
    amrex::Real minimum_optical_depth = 8;
    amrex::Real maximum_beta = 0.001;
    /** Solve for the proper-energy deviation U-B and evaluate boosted
     * E-E_LTE in its centered proper-energy/flux basis.
     * This is the same constitutive law, without subtracting separately
     * rounded large LAB energies before multiplication by c*dt*chi_a.
     * The normal long-step all-D coordinator selects it after authenticating
     * the duration; no equation, ledger tolerance or admission guard changes. */
    bool centered_thermal_difference = false;
    /** Opt-in CPU built-in-caloric FIELD trial with stationary reflecting z
     * endcaps. This does not admit native source, caloric, census or runtime
     * coupling at those boundaries. Periodic callers keep the default. */
    bool reflecting_axial_field_trial = false;
    /** PRIVATE default-OFF CPU one-fixed-charge TABLE context for that
     * closed FIELD trial. Source/helper composition must opt in separately;
     * this does not activate adaptive admission or normal runtime. */
    bool reflecting_axial_table_trial = false;
    bool single_selected_table_trial = false;
    /** Explicit periodic CPU selected-table context with exact vacuum in K.
     * This flag never admits a vacuum D cell or creates a density floor. */
    bool single_selected_table_vacuum = false;
};

struct ResolvedRZThermalDiffusionFieldResult
{
    bool valid = false;
    std::string reason;
    int linear_iterations = 0;
    amrex::Real linear_residual = 0;
    int picard_iterations = 0;
    amrex::Real equation_residual = 0;
    /** External stationary D-field endcap impulse [kg m/s], signed into each
     * wall (low z, high z). No wall energy/work or angular torque is implied. */
    amrex::GpuArray<amrex::Real, 2> axial_wall_momentum_by_side{};
    amrex::Real axial_wall_momentum = 0;

    /** Accepted private LAB energy [J/cell] and local conjugate
     * [cPr,cLz/rbar,cPz] [J/cell]. These are not published live states. */
    amrex::MultiFab energy;
    amrex::MultiFab lab_momentum;
    /** Old minus new radiation momentum minus stress transport [J/cell].
     * The theta row is conjugate; multiply by rbar for a global Lz audit. */
    amrex::MultiFab momentum_debit;
    /** Actual work is authoritative: physical debit=work+heat [J/cell]. */
    amrex::MultiFab requested_heat;
    /** Cell LAB LTE energy per unit direct proper bath [m^3]. This exact
     * coefficient is used by the accepted thermal RHS, is zero in K cells,
     * and permits corner allocation without recomputing frame quadrature. */
    amrex::MultiFab proper_bath_energy_coefficient;
    amrex::MultiFab physical_energy_debit;
    amrex::MultiFab raw_energy_debit;
    amrex::MultiFab storage_roundoff_bound;
    amrex::MultiFab operand_scale;
    /** c*dt*divergence of face stresses. Radial hoop and angular-conjugate
     * geometry are included; the theta connection is also exposed. */
    amrex::MultiFab outward_pressure_stress;
    amrex::MultiFab theta_connection;
    /** Signed AMReX-positive physical LAB energy current c*T0n [W/m^2],
     * stress current [cP_rn,cP_thetan,cP_zn] [W/m^2], and positive D->K
     * leaked energy [J]. On z faces the theta stress is the angular-conjugate
     * full-annulus <(r/rbar)cP_theta,z>; on radial faces it is physical
     * cP_theta,r and RZAngularGeometry applies the face lever. Both current
     * and lab_momentum boost one transport-plus-absorption dipole. */
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>
        face_lab_energy_current;
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>
        face_lab_stress;
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>
        face_leaked_energy;
};

/** Read-only immediate endpoint/frozen-input screen for collided admission
 * cohorts. Uses the SAME full-annulus initial proper-energy, causal-moment,
 * opacity, beta*chi*L and outgoing drift-CFL arithmetic as the implicit field.
 * It also requires the outer half-cell and temporal c*dt*chi_t>=8 accuracy
 * guards. The same native L0/m0/raw-FV/ideal-latent/context, exact
 * geometry/layout and stationary radial/periodic axial boundary guards as the
 * field are required. No field advance, material kick, heat, carry, birth or
 * publication occurs. Caller must separately enforce the actual physical census
 * angular/pressure admission screen: the necessary annular moment cone is not
 * sufficient closure suitability. Opacities/bath may be unmasked; K E/Q must be
 * zero.
 */
[[nodiscard]] bool AdmissibleResolvedRZThermalDiffusionCensus (
    amrex::MultiFab const& lab_energy, amrex::MultiFab const& lab_momentum,
    amrex::MultiFab const& proper_bath, amrex::MultiFab const& proper_transport_opacity,
    amrex::MultiFab const& proper_absorption_opacity, amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& work_velocity, amrex::Geometry const& geometry, amrex::Real dt,
    ResolvedRZThermalDiffusionFieldOptions const& options);

/** Private conservative scope bound, NOT the physical direct bath/RHS.
 * Positive normalized native Cv incidence makes sum(W*a*T^4) <= max(a*T^4),
 * with an operand-derived 16-epsilon collar for four-term normalization.
 * Exact zero bath remains zero; nonzero subnormal corner baths are rejected
 * because the relative collar cannot bound under-resolved weighted sums.
 * Evaluate from ACTUAL accepted/current nodal T, never a prior nonlinear trial.
 * Failure preserves the caller's output exactly; success writes valid cells.
 */
[[nodiscard]] bool TryBuildResolvedRZThermalBathUpperBound (
    amrex::MultiFab const& nodal_temperature,
    amrex::MultiFab& private_bath_bound, amrex::Geometry const& geometry,
    amrex::Real radiation_constant = 7.565733250280007e-16);

/** Conservatively clear only retained old-D proposals that fail the exact
 * current annular constitutive preflight. Old-K proposals remain proposals:
 * only physical collided census can admit them after the ONE physical source.
 * This handles ordinary macrostep changes of T/Cv/density/rate without trapping
 * stored D in an inadmissible source frame. It does not erase E/Q: the caller
 * MUST constructively demote cleared cells before paths. No live owner changes;
 * collective invalid-input rejection preserves proposed_mask including ghosts.
 */
[[nodiscard]] bool TryConstrainResolvedRZThermalDiffusionMask (
    amrex::MultiFab const& old_energy, amrex::MultiFab const& old_momentum,
    amrex::MultiFab const& proper_bath_bound, amrex::MultiFab const& transport,
    amrex::MultiFab const& absorption, amrex::iMultiFab const& old_mask,
    amrex::iMultiFab& proposed_mask,
    amrex::MultiFab const& current_rate_velocity,
    amrex::Geometry const& geometry, amrex::Real dt,
    ResolvedRZThermalDiffusionFieldOptions const& options);

/** Opt-in caller-private m0 RZ static-diffusion spatial/thermal trial.
 * All fields are cell centered except returned faces. old_momentum and result
 * momentum use (cPr,cLz/rbar,cPz). proper_bath is the caller's DIRECT
 * reciprocal sum of old-Cv/FV-weighted W_corner*a*Tnode^4 [J/m^3]. Every
 * thermodynamic origin samples the same full-annulus radial rule; W_corner
 * is caloric incidence, not localized spatial emissivity. The operator does
 * not substitute a cell-temperature average. work_velocity uses physical vr,
 * torque-conjugate vtheta at annular mean radius, and vz [m/s]. actual_work
 * includes the staged finite native kick and carry [J/cell] and is used
 * exactly, never reconciled to a predicted force. K cells have zero E,Q,
 * opacities and work. Failure leaves only discardable private outputs.
 * Old and accepted D LAB inventories obey the necessary annular ray cone
 * hypot(cPr,(cLz/rbar)/(rhi/rbar),cPz)<=E. A nonnegative ray at radius r
 * contributes (r/rbar)*E*n_theta to the conjugate theta row, so dividing by
 * rhi/rbar and applying the triangle inequality gives this bound. It is not
 * a sufficient realizability test or a Cartesian M1/dipole inversion.
 *
 * The bounded model solves a native annular FV implicit scalar energy with
 * absorption and accepted Densmore D/K leakage. The two-point Gauss rule in
 * s=r^2 and vtheta(r)=vtheta_star*r/rbar matches reciprocal RZ photon births.
 * Both LAB density and face/current/stress boost the same comoving
 * isotropic-plus-dipole closure: F0/c=Fdiff/c-(chi_a/chi_t)*beta*(u-B).
 * This retains first-order mixed beta_i*F0_j/c shear. Radial walls
 * reflect energy and angular shear; radial pressure traction and cylindrical
 * hoop/connection terms are reported. No global radial-momentum invariant,
 * chronological flight, runtime publication, M1, or general implicit K+D is
 * implied by a valid result.
 * The explicit CPU reflecting_axial_field_trial option additionally admits
 * stationary specular z endcaps for this FIELD trial only, under an outer
 * half-cell bound. It reports their external axial impulse separately.
 */
[[nodiscard]] ResolvedRZThermalDiffusionFieldResult
TryAdvanceResolvedRZThermalDiffusionField (
    amrex::MultiFab const& old_lab_energy, amrex::MultiFab const& old_lab_momentum,
    amrex::MultiFab const& proper_bath, amrex::MultiFab const& proper_transport_opacity,
    amrex::MultiFab const& proper_absorption_opacity, amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& work_velocity, amrex::MultiFab const& actual_work,
    amrex::Geometry const& geometry, amrex::Real dt,
    ResolvedRZThermalDiffusionFieldOptions const& options);

/** Separate default-OFF FIELD infrastructure, never a History/Interval or
 * runtime policy. The old options type and matched entry remain unchanged. */
struct IndependentGrayRestAllDFieldOptions
{
    bool enabled = false;
    ResolvedRZThermalDiffusionFieldOptions field;
    detail::ResolvedIndependentGrayMaterialTableOpacityOptions opacity;
};

/** CPU RZ periodic, strict-rest ALL-D caller-private trial. Obtains the actual
 * native sole TRUE-selected TABLE EOS internally and snapshots the supplied
 * physical nodal owners through the unchanged independent provider. R drives
 * diffusion and recovered P drives absorption; authentic S is metadata only.
 * Every cell separately conditions R/P and inverse R from loaded-owner boxes
 * at the existing 2^-26 resolution ceiling. Every velocity/work component
 * must be exactly zero, with temporal/spatial optical depth >=8 and the outer
 * half-cell bound. No mixed D/K, moving, closed, census, adaptive, material
 * kick, packet/source, restart or live publication is admitted. The table and
 * all inputs must remain immutable during this synchronous call. */
[[nodiscard]] ResolvedRZThermalDiffusionFieldResult
TryAdvanceIndependentGrayRestAllDField (
    amrex::MultiFab const& old_lab_energy, amrex::MultiFab const& old_lab_momentum,
    amrex::MultiFab const& proper_bath, amrex::MultiFab const& raw_charge_density,
    amrex::MultiFab const& nodal_temperature,
    detail::ResolvedMaterialChargeFields const& unit_ion_charge,
    MaterialOpacityTable const& table, amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& work_velocity, amrex::MultiFab const& actual_work,
    amrex::Geometry const& geometry, amrex::Real dt,
    IndependentGrayRestAllDFieldOptions const& options);

/** Separate default-OFF moving FIELD infrastructure. This does not activate
 * independent opacities in History, Interval, census or normal runtime. */
struct IndependentGrayMovingAllDFieldOptions
{
    bool enabled = false;
    ResolvedRZThermalDiffusionFieldOptions field;
    detail::ResolvedIndependentGrayMaterialTableOpacityOptions opacity;
};

class NativeRZIndependentGrayOpacitySlice;

/** Immutable numerical opacity epoch, internally issued from actual loaded
 * raw owners. This carries no History duration/no-K authority. No public
 * constructor, channel-field factory, caller executor or validity setter is
 * available. FIELD-only calls may issue it synchronously; History slice
 * owners add their separate engine-issued source/namespace authority. */
class IndependentGrayMovingFieldOpacity
{
public:
    ~IndependentGrayMovingFieldOpacity () = default;
    IndependentGrayMovingFieldOpacity (IndependentGrayMovingFieldOpacity const&) = delete;
    IndependentGrayMovingFieldOpacity& operator= (IndependentGrayMovingFieldOpacity const&) = delete;
    [[nodiscard]] detail::ResolvedIndependentGrayMaterialTableOpacitySnapshot const&
    Snapshot () const noexcept { return m_snapshot; }
    [[nodiscard]] amrex::MultiFab const& ChannelRelativeErrors () const noexcept
    { return m_errors; }
    [[nodiscard]] bool IsMixed () const noexcept { return m_mixed_mask != nullptr; }
    /** Read-only initial/trial/post-kick constitutive screen. Uses this ONE
     * frozen opacity epoch and actual supplied bath/frame, not a new opacity
     * query or physical FIELD advance. All calls are collective. */
    [[nodiscard]] bool Preflight (
        amrex::MultiFab const& lab_energy, amrex::MultiFab const& lab_momentum,
        amrex::MultiFab const& proper_bath, amrex::iMultiFab const& diffusion_mask,
        amrex::MultiFab const& velocity, amrex::MultiFab const& actual_work,
        std::string& reason) const;
    /** Advance using this frozen epoch. Trial bath/frame/work may change;
     * the accepted-left opacity query must not change during the solve. */
    [[nodiscard]] ResolvedRZThermalDiffusionFieldResult Advance (
        amrex::MultiFab const& lab_energy, amrex::MultiFab const& lab_momentum,
        amrex::MultiFab const& proper_bath, amrex::iMultiFab const& diffusion_mask,
        amrex::MultiFab const& velocity, amrex::MultiFab const& actual_work) const;

private:
    IndependentGrayMovingFieldOpacity (
        detail::ResolvedIndependentGrayMaterialTableOpacitySnapshot&& snapshot,
        amrex::MultiFab&& errors, amrex::Geometry const& geometry, amrex::Real dt,
        ResolvedRZThermalDiffusionFieldOptions const& options,
        std::unique_ptr<amrex::iMultiFab const> mixed_mask,
        std::unique_ptr<amrex::MultiFab const> masked_transport,
        std::unique_ptr<amrex::MultiFab const> masked_absorption);
    [[nodiscard]] static std::unique_ptr<IndependentGrayMovingFieldOpacity const> Issue (
        amrex::MultiFab const& raw_charge_density, amrex::MultiFab const& nodal_temperature,
        detail::ResolvedMaterialChargeFields const& unit_ion_charge,
        MaterialOpacityTable const& table, amrex::Geometry const& geometry, amrex::Real dt,
        IndependentGrayMovingAllDFieldOptions const& options, std::string& reason);
    [[nodiscard]] static std::unique_ptr<IndependentGrayMovingFieldOpacity const> IssueMixed (
        amrex::MultiFab const& raw_charge_density, amrex::MultiFab const& nodal_temperature,
        detail::ResolvedMaterialChargeFields const& unit_ion_charge,
        MaterialOpacityTable const& table, amrex::iMultiFab const& fixed_mask,
        amrex::Geometry const& geometry, amrex::Real dt,
        IndependentGrayMovingAllDFieldOptions const& options, std::string& reason);
    [[nodiscard]] static std::unique_ptr<IndependentGrayMovingFieldOpacity const> IssueEpoch (
        amrex::MultiFab const& raw_charge_density, amrex::MultiFab const& nodal_temperature,
        detail::ResolvedMaterialChargeFields const& unit_ion_charge,
        MaterialOpacityTable const& table, amrex::iMultiFab const* fixed_mask,
        amrex::Geometry const& geometry, amrex::Real dt,
        IndependentGrayMovingAllDFieldOptions const& options, std::string& reason);
    friend class NativeRZIndependentGrayOpacitySlice;
    friend ResolvedRZThermalDiffusionFieldResult TryAdvanceIndependentGrayMovingAllDField (
        amrex::MultiFab const&, amrex::MultiFab const&, amrex::MultiFab const&,
        amrex::MultiFab const&, amrex::MultiFab const&, detail::ResolvedMaterialChargeFields const&,
        MaterialOpacityTable const&, amrex::iMultiFab const&, amrex::MultiFab const&,
        amrex::MultiFab const&, amrex::Geometry const&, amrex::Real,
        IndependentGrayMovingAllDFieldOptions const&);
    detail::ResolvedIndependentGrayMaterialTableOpacitySnapshot const m_snapshot;
    amrex::MultiFab const m_errors;
    amrex::Geometry const m_geometry;
    amrex::Real const m_dt;
    ResolvedRZThermalDiffusionFieldOptions const m_options;
    std::unique_ptr<amrex::iMultiFab const> const m_mixed_mask;
    std::unique_ptr<amrex::MultiFab const> const m_masked_transport, m_masked_absorption;
};

/** CPU RZ periodic ALL-D, native sole TRUE-selected fixed TABLE trial. Uses
 * authentic R for diffusion and P for absorption; S remains metadata, never
 * R-P. Retains the old half-cell, optical-depth, causal, outgoing-CFL and
 * solver guards. Additionally conditions each R/P channel, inverse R and
 * authentic eager face ratio at 2^-26, and requires beta*R*L<=.01 and
 * beta*P*L<=.01 in every cell and at both physical neighboring face samples.
 * These new guards conservatively bound beta by an outward physical and
 * represented L1 norm, on IEEE double with extended-range long double.
 * The actual annular moment map must retain positive m00/proper energy.
 * work_velocity and actual_work are immutable caller-private inputs with
 * the existing finite-kick/carry meanings; this FIELD seam does not certify
 * their MaterialStage origin or publish any live material/photon state.
 * No mixed D/K, closed, adaptive, macro-duration or GPU scope is admitted. */
[[nodiscard]] ResolvedRZThermalDiffusionFieldResult
TryAdvanceIndependentGrayMovingAllDField (
    amrex::MultiFab const& old_lab_energy, amrex::MultiFab const& old_lab_momentum,
    amrex::MultiFab const& proper_bath, amrex::MultiFab const& raw_charge_density,
    amrex::MultiFab const& nodal_temperature,
    detail::ResolvedMaterialChargeFields const& unit_ion_charge,
    MaterialOpacityTable const& table, amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& work_velocity, amrex::MultiFab const& actual_work,
    amrex::Geometry const& geometry, amrex::Real dt,
    IndependentGrayMovingAllDFieldOptions const& options);
} // namespace warpx::radiation

#endif
