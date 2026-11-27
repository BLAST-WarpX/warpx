/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_PACKETDIFFUSIONNATIVEINTERVAL_H_
#define WARPX_RADIATION_PACKETDIFFUSIONNATIVEINTERVAL_H_

#include "AsymptoticDiffusionField.H"
#include "CollidedDiffusionAdmission.H"
#include "NativePhotonCollisionReplay.H"
#include "NativePhotonPathReplay.H"
#include "PhotonLogicalDrawKey.H"
#include "PhotonPacketTransaction.H"
#include "PacketMaterialWorkIteration.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <string>
#include <memory>
#include <optional>
#include <vector>

class MultiParticleContainer;
class WarpXParticleContainer;

namespace warpx::radiation
{
struct PacketDiffusionNativeIntervalOptions
{
    /** First-normal-moment-compatible extrapolation for the cosine interface.
     * This does not fix its unretained tangential momentum. The standalone
     * scalar backend retains its separate asymptotic lambda=0.7104 default. */
    AsymptoticDiffusionFieldOptions diffusion{
        .marshak_extrapolation_lambda = amrex::Real(2)/3};
    /** The native coordinator treats diffusion.linear_tolerance as an upper
     * bound. Its inner solve uses at most one percent of the tightest outer
     * work, scalar-ledger, or combined-energy tolerance. No acceptance bound
     * is changed; the independent work and conservation checks remain final. */
    PacketMaterialWorkIterationOptions material;
    amrex::Real photon_energy = amrex::Real(1.e-16);
    amrex::Real energy_relative_tolerance = amrex::Real(1.e-11);
    /** Optional frozen laboratory K-cell scattering coefficient [1/m].
     * It is separate from the D-cell diffusion opacity. */
    amrex::MultiFab const* kinetic_scattering_opacity = nullptr;
    NativePhotonCollisionOptions collisions;
    /** Resolved Cartesian path only. Open faces require a permanently kinetic
     * boundary cell layer; scalar diffusion never touches an escape face. */
    NativePhotonPathBoundaries photon_boundaries;
    /** Low-velocity enthalpy transport. Mixed domains require the explicit
     * resolved_collided_source mode; the default still requires all-D. */
    bool moving_diffusion = false;
    /** Experimental fixed-interface, quasi-static boundary-layer approximation.
     * Replace outer isotropic pressure traction by actual angular crossing
     * traction, retaining isotropic D/D pressure. No second interface kick.
     * Alone, radiation momentum storage is neglected and the angular law is static:
     * this is leading beta->0 physics, not exact transient/moving scattering.
     * Caller must establish slow forcing relative to layer residence time;
     * optical-depth and velocity guards alone do not establish that ordering.
     * Alone, requires mixed 1D, nonmoving diffusion and a nonzero D reservoir.
     * retain_constitutive_momentum adds stiff stored q; resolved_collided_source
     * replaces the instantaneous face split and permits cold/moving populations. */
    bool quasi_static_angular_traction = false;
    amrex::Real quasi_static_maximum_beta = amrex::Real(1.e-3);
    amrex::Real quasi_static_minimum_optical_depth = 8;
    /** Explicit conservative constitutive-FLD model, NOT causal layer storage.
     * Persist q=V*F/c [J,3] and include its interval change in material force.
     * The supplied conserved_diffusion_momentum is distinct from the legacy
     * diagnostic inertia. Angular traction and stiff D relaxation are required;
     * the constitutive projection is not a finite-rate transport solution. */
    bool retain_constitutive_momentum = false;
    amrex::Real minimum_constitutive_relaxation_depth = 8;
    /** Collided-source decomposition: chronological physical packets traverse
     * K/collar and D cells. Only a scattered, angularly admissible census cohort
     * converts volumetrically at interval end, transferring its full inventory.
     * Requires explicit photon census registration and retained momentum. The
     * kinetic_scattering_opacity then covers the COMPLETE domain, including D.
     * Active or newly admitted D populations require identical packet/scalar
     * transport opacity for this gray isotropic scattering medium. RZ currently
     * requires prescribed laboratory opacity (moving_cross_sections=false).
     * An empty D label
     * alone does not impose that constraint on purely kinetic photons.
     * The caller chooses a resolved kinetic collar; no instantaneous reflection
     * is applied at its outer edge. */
    bool resolved_collided_source = false;
    /** Keep exactly empty eligible cells kinetic for this source interval.
     * The frozen geometric diffusion_mask remains the census-admission region;
     * only cells with zero old E and all three stored q components zero are
     * excluded from the private scalar solve. A frozen-frame scalar predictor
     * screens proposed census admissions; a failed screen retains all those
     * photons kinetically, without changing the actual material work.
     * Existing diffuse inventories
     * are never converted, clipped, or removed. Requires resolved source mode.
     * This is an empty-front treatment, not a general realizability guarantee
     * for steep nonempty profiles or an adaptive material/opacity partition. */
    bool kinetic_empty_cells = false;
    /** Opt-in fixed packet-only path subsegments for conservative Cartesian
     * 1D, XZ, 3D and native RZ transport. The outer constitutive interval and material work remain
     * unsubcycled; only physical packet replay is partitioned so every native
     * path call retains c*dt_sub <= half the smallest spatial cell width.
     * Requires resolved_collided_source. */
    bool transactional_packet_substeps = false;
    /** Explicit bound on the fixed number of packet path subsegments. */
    int packet_subsegment_limit = 1024;
    /** Runtime-only nonzero draw identities for D->K births. The context is
     * fixed for the accepted outer interval and reused across work replay. */
    std::optional<LogicalPhotonBirthKeyContext> logical_birth_keys;
    CollidedDiffusionAdmissionControls admission;
    /** Additive staging mode for a larger radiation transaction. On success,
     * return privately advanced material, photons and post-admission E/q,
     * without publishing any live particle, input field, birth ID or counter.
     * The caller may follow spatial/coherent transport by a local thermal
     * source, then jointly publish all representations after full acceptance.
     * This is an explicit sequential source split, not a simultaneous source
     * solve. Default false preserves the established commit path. */
    bool defer_live_commit = false;
    /** Optional caller-private interval-start photons after conservative
     * representation changes. Requires resolved/deferred Cartesian transport,
     * or the narrow native RZ opt-in below. The snapshot retains its original live provenance;
     * marked newborns require unique logical draw keys and remain provisional
     * through physical transport. No early native IDs or live commit occurs.
     * The caller owns the conversion's separate E/q/carry inventory audit. */
    PhotonPacketTransaction const* private_photon_snapshot = nullptr;
    /** Narrow native RZ adaptive prerequisite. Requires a caller-private
     * snapshot, deferred publication, resolved source, retained conjugate q,
     * RZAngularConservative material assignment, m=0, and the established
     * laboratory-opacity/scalar bounds. No open radial/axial escape is enabled.
     * False leaves the old RZ snapshot prohibition and Cartesian API unchanged.
     */
    bool adaptive_rz_private_snapshot = false;
    /** Required only by adaptive_rz_private_snapshot: immutable signed native
     * radiation [E,cPr,cLz/rbar,cPz] ownership through transport. The caller's
     * actual stored-birth conversion audited its origin. This interval neither
     * spends it as material work nor creates a carry from a global residual.
     * It is copied privately, including ghosts, and enters E/Pz/rbar-Lz ledgers
     * without enlarging physical acceptance scales. The caller must persist it
     * through the same joint publication and all future transfers/restarts. */
    amrex::MultiFab const* radiation_representation_carry = nullptr;
};

struct PacketDiffusionNativeIntervalResult
{
    bool valid = false;
    std::string reason;
    int work_iterations = 0;
    /** Actual inner relative solve tolerance after native accuracy budgeting. */
    amrex::Real linear_tolerance = 0;
    /** Eligible empty cells left kinetic by the opt-in front policy. */
    amrex::Long kinetic_front_cells = 0;
    /** Physical census photons retained when the prospective scalar state fails. */
    amrex::Long deferred_front_admissions = 0;
    amrex::Long collision_events = 0;
    amrex::Real actual_material_work = 0;
    amrex::Real material_energy_carry_change = 0;
    amrex::Real raw_energy_residual = 0;
    /** Actual photon/material change plus the change of asymptotic radiation
     * inertia [kg m/s]. This model diagnostic is NOT a numerical acceptance
     * condition or an additional material kick. */
    amrex::GpuArray<amrex::Real, 3> momentum_model_residual{};
    /** Momentum discarded by the static 1D diffuse interface's tangential
     * isotropization [kg m/s]. Diagnostic only: neither a material impulse nor
     * a conserved carry. Nonzero values preclude a vector-conservative claim. */
    amrex::GpuArray<amrex::Real, 3> interface_tangential_model_defect{};
    /** Original scalar-backend candidate and diagnostics. In angular-traction
     * mode field.material_impulse and field.closure_diagnostic still describe
     * the unmodified scalar closure, NOT the applied mixed angular stress.
     * Use actual_material_impulse_field plus its momentum carry change for
     * the accepted material exchange in every mode. In resolved-source mode
     * field.energy and both field/lab inertia diagnostics are PRE-admission;
     * the committed caller E/q include admitted_four_inventory afterwards.
     * Admission is a representation transfer, not another material force. */
    AsymptoticDiffusionFieldResult field;
    bool used_quasi_static_angular_traction = false;
    bool used_constitutive_momentum = false;
    /** Separate lab-frame diagnostic for moving_diffusion, otherwise undefined.
     * Do not confuse it with field.radiation_inertia's comoving diagnostic. */
    amrex::MultiFab lab_radiation_inertia;
    /** Signed stationary radiation-wall reaction for moving_diffusion [kg m/s]. */
    amrex::GpuArray<amrex::Real, 3> radiation_wall_momentum{};
    /** Signed native packet momentum received by stationary Cartesian walls.
     * External to material recoil; energy work is exactly zero. */
    amrex::GpuArray<amrex::Real, 3> packet_wall_momentum{};
    /** External escape [E,c*p], distinct from stationary wall reaction. */
    amrex::GpuArray<amrex::Real, 4> escaped_packet_inventory{};
    /** Actual finite-mass impulse [kg m/s, 3]. Copied from the accepted
     * transaction before commit. Defined only when valid. */
    amrex::MultiFab actual_material_impulse_field;
    /** Actual finite-mass kinetic work [J, 1]. Copied from the accepted
     * transaction before commit. Defined only when valid. */
    amrex::MultiFab actual_material_work_field;
    /** Particle-owned momentum-carry change [kg m/s, 3]. Copied from the
     * accepted transaction before commit. Defined only when valid. */
    amrex::MultiFab material_momentum_carry_change_field;
    /** Defined only for a valid deferred interval. The finite transport kick
     * has already been committed into private_material, never into live ions.
     * private_photons retains the original live baseline and provisional births.
     * Both owners must survive any subsequent source stages. The accepted
     * E/q include census admission, unlike the pre-admission field diagnostics.
     * Destroying these candidates discards the complete transport interval. */
    std::unique_ptr<ParticleImpulseMaterial> private_material;
    std::unique_ptr<PhotonPacketTransaction> private_photons;
    amrex::MultiFab private_radiation_energy;
    amrex::MultiFab private_diffusion_momentum;
    /** Immutable conjugate radiation carry for a valid adaptive RZ deferred
     * interval. Undefined for legacy/Cartesian calls. Its ghosts are exact
     * copies, not independently filled or projected. */
    amrex::MultiFab private_radiation_representation_carry;
    /** Mixed 1D interfaces only; undefined for all-D and on rejection.
     * Face values [J] are synchronized copies; sum unique OwnerMask faces.
     * Crossing components 0:3 are incident [E,c*p], 4:7 actual diffuse returns.
     * Leakage components 0:3 are actual outgoing [E,c*p]. */
    amrex::MultiFab interface_crossing_inventory;
    amrex::MultiFab interface_leakage_inventory;
    /** Actual outgoing-minus-incoming c*p minus the scalar boundary traction
     * c*dt*A*Pi*n, in joules. Three physical xyz components per face. This is
     * a LOCAL closure diagnostic, never a deposited impulse or a work carry. */
    amrex::MultiFab interface_traction_model_defect;
    /** Multidimensional resolved-source face diagnostics. Cartesian inventories
     * have four components, RZ has [E,cPr,cPtheta,cPz,cLz]. The old scalar face
     * members remain the 1D compatibility outputs. */
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>
        interface_leakage_by_direction;
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>
        interface_traction_defect_by_direction;
    /** Packet plus native material/carry momentum change [kg m/s], excluding
     * asymptotic diagnostic inertia. Enforced only for the explicit quasi-static
     * angular-traction model. Its neglected storage remains a model error. */
    amrex::GpuArray<amrex::Real, 3> quasi_static_momentum_residual{};
    amrex::GpuArray<amrex::Real, 3> constitutive_momentum_residual{};
    /** RZ conserved ledgers; the Cartesian residual arrays above are not used.
     * Axial residual [kg m/s]; angular residual [kg m^2/s]. No global radial
     * conservation condition is asserted. Material field outputs in RZ use
     * local radial, torque/rbar, and axial work-conjugate components. */
    amrex::Real rz_axial_momentum_residual = 0;
    amrex::Real rz_angular_momentum_residual = 0;
    amrex::MultiFab admitted_four_inventory;
};

/** One coordinated native packet/diffusion/material source interval.
 *
 * Bounded source API: Cartesian 1D/XZ/3D with periodic or stationary reflecting
 * faces, or native RZ with reflecting radial and periodic or stationary
 * reflecting axial faces. Fixed
 * D/K masks, gray coherent scattering
 * and c*dt<=min(dx)/2. Material positions are frozen but finite-mass work and
 * momenta are native. Cartesian uses nearest-cell assignment; RZ requires
 * angular-conjugate assignment and prescribed laboratory opacity. RZ has no
 * global radial momentum conservation condition.
 * Multidimensional mixed domains require resolved_collided_source and retained
 * constitutive q; the old instantaneous scalar interface remains 1D only.
 * Event choices are frozen across work iteration. Moving diffusion/leakage
 * is a leading thick/low-beta model, omitting v*F0/c^2 stress corrections and
 * O(beta^2) stresses. This does not activate a complete WarpX Evolve timestep,
 * qualify moving geometric masks, or resolve Wave 10's transient closure bias.
 *
 * Radiation energy is cell-integrated J. Old inertia is an optional [J,3]
 * diagnostic field (lab for moving_diffusion; comoving otherwise), not causal
 * radiation momentum. Caller species must be
 * valid positive-mass material and must exclude the photon species. The caller
 * must register the material impulse path before particle initialization.
 *
 * Numerical rejection leaves live photons/material and energy/ghosts untouched.
 * Only after actual emitted photons, scalar energy, actual kinetic work and
 * carry pass a combined ledger are all three representations committed.
 * Process/allocation failure and concurrent live mutations are not recoverable
 * transaction semantics. Caller stores returned inertia only on success.
 */
PacketDiffusionNativeIntervalResult
TryAdvancePacketDiffusionNativeInterval (
    MultiParticleContainer& particles, WarpXParticleContainer& photons,
    std::vector<std::string> const& material_species, std::string const& impulse_path,
    amrex::MultiFab& radiation_energy, amrex::MultiFab const& opacity,
    amrex::iMultiFab const& diffusion_mask, amrex::Geometry const& geometry,
    amrex::Real dt, amrex::MultiFab const* old_radiation_inertia,
    PacketDiffusionNativeIntervalOptions const& options = {},
    amrex::MultiFab* conserved_diffusion_momentum = nullptr);
}

#endif
