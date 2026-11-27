/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_RADIATION_PACKET_DIFFUSION_INTERFACE_PHOTONS_H_
#define WARPX_RADIATION_PACKET_DIFFUSION_INTERFACE_PHOTONS_H_

#include "FluxRelaxedPackets.H"
#include "PhotonLogicalDrawKey.H"

#include <AMReX_Geometry.H>
#include <AMReX_INT.H>
#include <AMReX_MultiFab.H>

namespace warpx::radiation
{
/** Controls for the bounded native D/K packet-interface trial.
 *
 * All quantities use SI units.  The extrapolation constant is dimensionless.
 */
struct PacketDiffusionInterfacePhotonControls
{
    double marshak_extrapolation_lambda = 0.7104;
};

/** Exact-once global diagnostics for a private interface trial. */
struct PacketDiffusionInterfacePhotonReport
{
    long incident_packets = 0;      ///< packets that crossed into a D cell
    long transported_packets = 0;   ///< packets remaining in transparent K
    long reflected_rays = 0;        ///< nonzero reflected quadrature rays
    double admitted_energy = 0.0;   ///< integrated face admissions [J]
    double reflected_energy = 0.0;  ///< reflected emission-plan energy [J]
    amrex::GpuArray<amrex::Real, 4> initial_inventory{};  ///< [E,c*p] [J]
    amrex::GpuArray<amrex::Real, 4> final_inventory{};    ///< [E,c*p] [J]
    amrex::GpuArray<amrex::Real, 4> reflected_inventory{}; ///< [E,c*p] [J]
};

/** Stream one private level-zero photon candidate through transparent K cells.
 *
 * The fixed cell-centered mask uses one component with 1=D and 0=K.  Both the
 * mask and one-component transport opacity must have at least one filled ghost
 * cell, identical boxes and distribution maps, and Cartesian periodic topology.
 * The coordinates are physical xyz directions even for WarpX's 1D-z spatial
 * storage.  A K cell must have zero transport opacity; the helper performs no
 * K collisions or material callbacks.  `dt` must be positive and satisfy
 * c*dt <= 0.5*dx.  A directed photon initially classified as D is rejected.
 * The half-open convention maps [cell_lo,cell_hi) to that cell, including
 * periodic wrap of the domain's upper endpoint to its lower cell.
 *
 * A ray can encounter at most one geometric face.  If the adjacent cell is K,
 * it is analytically placed at the segment endpoint.  If it is D, tau=chi_D*dx
 * (the full D-cell width) and mu=|u_z|/|u| are passed to
 * PacketDiffusionInterfaceProbability without clipping.  Packet energy means
 * the actual photon energy `w*m_e*c*|u|` [J].  The already-crossed incident
 * energy is multiplied by P once; no extra incident cosine is applied.  The
 * admitted part is deposited to `face_admissions` as positive energy into D.
 * The original private photon is invalidated and its (1-P) energy is returned
 * in `diffuse_returns` using eight positive-energy outward cosine-hemisphere
 * rays.  Each ray is analytically advanced only through its remaining segment
 * time in transparent K.  Thus `report.final_inventory` is survivors-only,
 * before deferred return plans are staged; actual native inventory must be
 * remeasured by the caller after `StageEmissionPlan`.
 *
 * `face_admissions` must be x-face/nodal, have one component, at least one
 * ghost node, and the mask BoxArray converted to faces.  Its units are
 * integrated joules admitted through each geometric face (not a flux).  D/D
 * and K/K values remain exactly zero.  Local ghost deposition is followed by
 * AMReX SumBoundary and OverrideSync, including periodic boundaries, so a
 * duplicated shared/periodic node represents one event.
 *
 * This call is collective over every rank, including ranks with no particle
 * grids.  False is the same numerical rejection on every rank.  On false the
 * report, emission plan, and all valid+ghost face components are cleared, and
 * the caller must discard the entire private transaction.  Live photons, IDs,
 * RNG state and material state are never touched by this helper.  On success
 * the private survivor container is redistributed.
 *
 * Optional face_inventory records actual crossed incident [E,c*p] in components
 * 0:3 and actual diffuse-return [E,c*p] in 4:7, all in joules. It has the same
 * layout/ghost extent as face_admissions and eight components. Shared/periodic
 * copies are synchronized, not additional events: use an OwnerMask to sum them.
 * This is an angular boundary ledger, NOT a material impulse or a new closure.
 * It is cleared on rejection, including ghosts. All ranks must opt in together.
 */
[[nodiscard]] bool
AdvancePacketDiffusionCandidatePhotons (
    WarpXParticleContainer::Base& private_photons, amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& opacity, amrex::Geometry const& geometry, amrex::Real dt,
    amrex::MultiFab& face_admissions, detail::EmissionPlan& diffuse_returns,
    PacketDiffusionInterfacePhotonReport& report,
    PacketDiffusionInterfacePhotonControls const& controls = {},
    amrex::MultiFab* face_inventory = nullptr);

/** Convert accepted D-to-K face leakage to a private photon emission plan.
 *
 * `accepted_face_leakage` has the selected mesh-direction face/nodal layout and
 * integrated-energy [J] units as the admission field, with nonnegative energy
 * and zero or more face ghosts (only owned valid face values are read),
 * directed out of D.  It is an outgoing leakage integral, not a signed net
 * current.  D/D and K/K faces must be identically zero.  Shared and periodic
 * nodes are read exactly once through an AMReX owner mask.  The selected owner
 * may be either equivalent periodic endpoint, so neighbor reads and births use
 * that owner's local face index.
 *
 * Each positive face event becomes eight deterministic outward
 * cosine-hemisphere rays with nodes mu_pm=(6_pm sqrt(6))/10 and weights
 * w_pm=(9_pm sqrt(6))/18, each with four equal azimuths.  This integrates
 * density 2 mu with <mu>=2/3, <mu^2>=1/2 and zero transverse mean.  A ray is
 * born at segment end exactly on the owned K side (one `nextafter` from the
 * geometric face), using the configured one-photon energy [J]; there is no
 * additional full-segment streaming.  `photon_energy` must be strictly
 * positive.  Empty and identically-zero leakage return true with an empty plan.
 *
 * The call is collective.  False is identical on every rank and clears the
 * plan; no live state is accessed.
 * Optional face_inventory contains actual outgoing [E,c*p] [J] in four
 * components on the leakage face layout with the same ghost extent. Values
 * are synchronized across shared/periodic copies; sum unique owners only.
 * Optional cell-centered work_velocity supplies the emitting D-cell material
 * secant velocity [m/s], three physical components, |v|/c<=0.001. Its ghosts
 * are not read or changed. Lab-fixed hemisphere nodes then use normalized
 * weights w*D^4, D=1/[gamma*(1-beta.n_lab)], from isotropic comoving intensity.
 * This integrates leading O(beta) angular moments; finite higher-order
 * quadrature errors and the Milne-layer approximation remain. It is not an
 * exact moving boundary law. Prescribed LAB leakage energy stays authoritative:
 * angular normalization creates no second Doppler debit. Actual stored u/w
 * inventory defines the returned ledger. Null or zero velocity retains the
 * original rays; all ranks must agree on pointer presence.
 * mesh_direction selects z in 1D, x/z in 2D, and x/y/z in 3D. All ranks
 * must agree on it. Births use transverse cell centers. Nonperiodic Cartesian
 * faces are stationary reflecting walls: leakage must be exactly zero there,
 * and exterior mask ghosts are not consulted. RZ uses the separate API below.
 */
[[nodiscard]] bool
BuildPacketDiffusionLeakageEmissionPlan (
    amrex::MultiFab const& accepted_face_leakage, amrex::iMultiFab const& diffusion_mask,
    amrex::Geometry const& geometry, amrex::Real photon_energy,
    detail::EmissionPlan& leakage_plan, amrex::MultiFab* face_inventory = nullptr,
    amrex::MultiFab const* work_velocity = nullptr, int mesh_direction = 0,
    LogicalPhotonBirthKeyContext const* key_context = nullptr);

/** RZ counterpart with a distinct five-component face ledger.
 * A native RZ build accepts AMReX coordinate tags 0 or 1 with explicit annular
 * measures, radially nonperiodic and axially periodic or reflecting geometry;
 * radial exterior faces represent reflecting boundaries and must have zero
 * leakage. Accepted energies already include the annular face measure [J].
 * Each macro-photon represents its full physical energy, with no additional
 * radius or volume factor. Births lie at theta=0: radial faces use their
 * radius nudged into K, axial faces use the annular RMS radius
 * sqrt((rlo^2+rhi^2)/2). This single radial node preserves leading angular
 * flux for rigid rotation, proportional to Omega*<r^2>; it does not resolve
 * higher radial moments or provide an exact moving Milne law. Momentum
 * components are Cartesian at theta=0, i.e. local (r,theta,z).
 * Optional face_inventory contains [E,cPr,cPtheta,cPz,cLz], the last component
 * in J*m, computed from actual stored native radius, momentum and weight.
 * Divide cLz by the emitting D-cell annular mean radius to obtain the
 * torque-conjugate azimuthal inventory. EmissionPlan.inventory retains its
 * ordinary four physical Cartesian components; it does not contain torque.
 * work_velocity uses (vr,vtheta_conjugate,vz), where the conjugate azimuthal
 * velocity is evaluated at the D-cell annular mean radius. At birth radius r,
 * the intensity uses vtheta=vtheta_conjugate*r/rbar_D. The resulting physical
 * speed must satisfy |v|/c<=0.001. mesh_direction is 0=radial or 1=axial.
 * Non-RZ builds reject this API. Other ownership/rollback contracts match the
 * Cartesian builder. This representative-angle construction makes no claim
 * of global radial momentum conservation.
 */
[[nodiscard]] bool
BuildRZPacketDiffusionLeakageEmissionPlan (
    amrex::MultiFab const& accepted_face_leakage, amrex::iMultiFab const& diffusion_mask,
    amrex::Geometry const& geometry, amrex::Real photon_energy,
    detail::EmissionPlan& leakage_plan, amrex::MultiFab* face_inventory = nullptr,
    amrex::MultiFab const* work_velocity = nullptr, int mesh_direction = 0,
    LogicalPhotonBirthKeyContext const* key_context = nullptr);

} // namespace warpx::radiation

#endif // WARPX_RADIATION_PACKET_DIFFUSION_INTERFACE_PHOTONS_H_
