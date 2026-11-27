/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_ADAPTIVEPACKETDIFFUSIONTRANSITION_H_
#define WARPX_RADIATION_ADAPTIVEPACKETDIFFUSIONTRANSITION_H_

#include "CollidedDiffusionAdmission.H"
#include "PhotonLogicalDrawKey.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <array>
#include <string>

namespace warpx::radiation
{
class PhotonPacketTransaction;

/** Optical-depth hysteresis is a representation policy, not a moving face.
 * The candidate may label a cell D only above promote_depth; an existing D
 * label survives down to demote_depth. Anisotropic photons remain kinetic.
 */
struct AdaptiveDiffusionMaskPolicy
{
    amrex::Real demote_depth = 8;
    amrex::Real promote_depth = 16;
};

/** Pure mask proposal from current material/opacity. Validation is collective;
 * rejection leaves candidate_mask (including ghosts) unchanged. This does not
 * move radiation inventory and does not make the mask safe for a field solve.
 */
bool TryProposeAdaptiveDiffusionMask (amrex::iMultiFab const& old_mask,
                                      amrex::MultiFab const& opacity,
                                      amrex::Geometry const& geometry,
                                      amrex::iMultiFab& candidate_mask,
                                      AdaptiveDiffusionMaskPolicy const& policy = {});

/** Optional CPU RZ readback, GLOBAL exactly once. Currency is physical
 * [E,0,cLz,cPz], not local conjugate q. Every owned cell/valid native ray is
 * counted, including zero inventories; carry participates even in K cells.
 * These are actual immutable stored-state operands, never an allowance
 * inferred from a measured residual. */
struct AdaptiveRepresentationInventoryWitness
{
    bool valid = false;
    amrex::Long packet_records = 0, field_records = 0, carry_records = 0;
    int ranks = 0;
    std::array<long double, 4> packet{}, field{}, carry{};
    std::array<long double, 4> packet_operands{}, field_operands{}, carry_operands{};
    // Bounds BOTH original Real and new LD readback, including upper operand
    // formation: <=64/native ray, <=32/field or carry cell and <=32/rank
    // for their floating reductions (27 relevant scalar reductions/rank).
    // Basic operations round to nearest; sqrt relative and sin/cos absolute
    // error must be <=epsReal (CPU libm/no fast-math). Opt-in capture rejects
    // nonzero subnormal inputs/products/quotients and non-nearest rounding:
    // no relative-gamma underflow claim. Angular operands use
    // |w*m_e*c*r|*(|ux|+|uy|), not the cancelled angular momentum.
    // This is a readback bridge, not the constructive conversion's budget.
    long double readback_operations = 0;
};

/** Capture-only CPU RZ helper. No owner/ID/RNG/clock changes. GLOBAL arrays
 * and counts must not be MPI-reduced again by an enclosing coordinator. */
[[nodiscard]] AdaptiveRepresentationInventoryWitness CaptureAdaptiveRepresentationInventory (
    WarpXParticleContainer::Base const& photons, amrex::MultiFab const& energy,
    amrex::MultiFab const& momentum, amrex::MultiFab const& representation_carry,
    amrex::Geometry const& geometry);

struct AdaptiveRepresentationConversionWitness
{
    bool valid = false;
    AdaptiveRepresentationInventoryWitness before, after;
    // Same declared old converter bound: 128*eps*(packet_before_E+field_before_E),
    // with ProbHi(r) for physical cLz. Additional witness readback uses its
    // explicit count/uncancelled operands; no observed residual supplies bounds.
    std::array<long double, 4> declared_conversion_roundoff_bound{};
    std::array<long double, 4> carried_residual{}; // GLOBAL after-minus-before.
};

struct AdaptivePacketDiffusionTransitionResult
{
    bool valid = false;
    std::string reason;
    amrex::Long demoted_cells = 0;
    amrex::Long demoted_photons = 0;
    amrex::Long promoted_cells = 0;
    amrex::Long promoted_photons = 0;
    amrex::Long deferred_cells = 0;
    /** Private physical E [J] and q [J,3]. Cartesian q=c*p_xyz;
     * native RZ q=[cPr,cLz/rbar,cPz] in the annular conjugate basis. */
    amrex::MultiFab energy;
    amrex::MultiFab momentum;
    amrex::iMultiFab mask;
    /** Signed radiation [E,q] participant in the field's Cartesian or native
     * RZ conjugate basis. It is not a material impulse or
     * energy source. Caller must persist it in diagnostics, normal restart and
     * all future inventory transfers, even in cells whose D label is cleared.
     * Only actual stored-ray reconstruction/addition roundoff enters changes.
     */
    amrex::MultiFab representation_carry;
    amrex::MultiFab carry_change;
    /** Independently deposited from actual stored marked newborns after
     * native staging and redistribution, selected by this batch's draw keys.
     */
    amrex::MultiFab realized_demotion_inventory;
    amrex::MultiFab realized_promotion_inventory;
    /** Cartesian E/c*p residuals; RZ [E,0,cLz,cPz], with no radial global claim. */
    amrex::GpuArray<amrex::Real, 4> raw_inventory_residual{};
    amrex::GpuArray<amrex::Real, 4> carried_inventory_residual{};
    // Opt-in immutable accounting metadata; disabled preserves old arithmetic.
    AdaptiveRepresentationConversionWitness accounting_witness;
};

/** Caller-private conservative representation conversion on a fixed Cartesian or native RZ
 * level-zero mesh. No propagation, wall work, material force, field transport,
 * opacity update or time step is performed. One-dimensional radial geometry is unqualified.
 * Native RZ births use the annular volume-mean radius and deterministic theta=0
 * (axisymmetric m=0). Actual stored r/theta and Cartesian momentum realize the
 * local conjugate E/q. Global ledgers are E/Pz/rbar-weighted Lz only; no summed
 * radial conservation law or geometric moving-boundary claim is made.
 *
 * Demote every old D cell requested K by constructive nonnegative 8-ray E/q
 * reconstruction. Strict |q|<=E is required before birth; the arithmetic collar
 * of the angular helper cannot admit an acausal input. Only after actual native
 * newborn E/q pass the existing 128-epsilon cell arithmetic bound can E/q be
 * cleared, with its signed representational remainder retained in carry.
 *
 * Promote only physically collided/isotropized census in old K cells requested
 * D, using the existing angular admission screen and actual four-inventory.
 * Empty or inadmissible proposed D cells retain their old K label. Census
 * markers must describe the current physical replay; wall-only changes and
 * previously born photons without a fresh census snapshot are not admissible
 * evidence. Demotion births are staged after census promotion.
 *
 * Inputs are read-only. photon_trial must be a private native transaction;
 * a rejected call requires discarding that complete photon trial and result.
 * Live photons, material, fields, masks/carry, native IDs and RNG remain
 * intact. The caller audits all coupled physics before ValidateLive/Commit and
 * copies accepted fields only after that final boundary. Native
 * process/allocation failure and concurrent mutation are outside recoverable
 * numerical semantics. Root runtime activation requires persistent ownership of
 * representation_carry.
 */
AdaptivePacketDiffusionTransitionResult TryConvertAdaptivePacketDiffusionRepresentation (
    PhotonPacketTransaction& photon_trial, amrex::MultiFab const& old_energy,
    amrex::MultiFab const& old_momentum, amrex::iMultiFab const& old_mask,
    amrex::iMultiFab const& desired_mask, amrex::MultiFab const& representation_carry,
    amrex::Geometry const& geometry, CollidedDiffusionAdmissionComponents const& census,
    LogicalPhotonBirthKeyContext const& birth_keys, amrex::Real photon_energy,
    CollidedDiffusionAdmissionControls const& admission = {},
    bool capture_accounting_witness = false);
} // namespace warpx::radiation
#endif
