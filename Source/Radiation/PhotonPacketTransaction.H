/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PHOTON_PACKET_TRANSACTION_H_
#define WARPX_PHOTON_PACKET_TRANSACTION_H_

#include "Particles/WarpXParticleContainer.H"

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <cstdint>
#include <memory>
#include <vector>

namespace warpx::radiation
{
/** Private native packet-to-FLD photon conversion trial.
 *
 * The constructor clones the live level-zero photon container with
 * make_alike and copyParticles. The clone has the same SoA schema, arena,
 * global IDs and complete real/int particle state as the live container.
 * The caller may redistribute, compact/delete or reorder the candidate;
 * ValidateLive captures that exact candidate layout for Commit.
 *
 * ValidateLive is collective on every rank and checks the unchanged live
 * baseline while capturing the staged candidate.  Commit performs a final
 * collective check, assigns native identities to surviving births in a
 * marker-free accepted candidate, and then swaps live storage. Destruction
 * or rejection destroys the private clone and cannot change live particles or
 * the native ID allocator.  This class stores no fields and performs no
 * conversion physics.
 * Numerical rejection is recoverable; process/GPU failures are not. No
 * concurrent mutation of the live species is permitted during the call.
 *
 * Provisional births use duplicate placeholder IDs while marked only in the
 * private candidate.  Native identity is the complete (id, cpu) pair.  Commit
 * consumes exactly one native NextID value per surviving local birth and never
 * rewinds the allocator. Cartesian and RZ geometry at level zero are supported.
 * RZ births accept Cartesian x/y positions and momenta, encoding position as
 * native radius/theta while retaining physical packet weight (no annular
 * density factor). Snapshots include theta and every other native component.
 * Marked candidates may only undergo identity-independent trial operations.
 * Never expose placeholder identities to diagnostics, checkpoints, QED,
 * resampling, or ID-seeded random draws; a caller needing random trial events
 * must supply its own stable trial keys and random-number accounting.
 * The exact live/candidate baselines cost host memory of the order of two
 * complete particle copies; they are released when no longer needed.
 */
class PhotonPacketTransaction
{
  public:
    explicit PhotonPacketTransaction (WarpXParticleContainer& live_photons);
    /** Restart from an existing private candidate, including provisional
     * births, while retaining that transaction's original live baseline.
     * Both objects must have the same live owner. No native IDs are consumed. */
    PhotonPacketTransaction (WarpXParticleContainer& live_photons,
                             PhotonPacketTransaction const& private_snapshot);
    ~PhotonPacketTransaction ();

    PhotonPacketTransaction (PhotonPacketTransaction const&) = delete;
    PhotonPacketTransaction& operator=(PhotonPacketTransaction const&) = delete;
    PhotonPacketTransaction (PhotonPacketTransaction&&) = delete;
    PhotonPacketTransaction& operator=(PhotonPacketTransaction&&) = delete;

    /** Private trial container. The reference remains valid until the
     * transaction is destroyed or committed, but is invalid for further
     * mutation after a successful ValidateLive followed by Commit. */
    [[nodiscard]] WarpXParticleContainer::Base& Particles ();
    [[nodiscard]] WarpXParticleContainer::Base const& Particles () const;

    /** Collective, recoverable, read-only preflight for a nested clone.
     * Checks this uncommitted private snapshot's original live provenance,
     * schema and provisional markers. False changes neither owner, candidate
     * nor native ID allocator. Call before the snapshot constructor when
     * invalid caller provenance must be a numerical rejection, not an assert. */
    [[nodiscard]] bool ValidateSnapshotFor (WarpXParticleContainer const& live_owner) const;

    /** Collectively adopt a nested private result with identical original live
     * provenance. This consumes the source candidate and invalidates previous
     * validation of this candidate; a fresh ValidateLive is required before
     * final live Commit. False changes neither private candidate. */
    [[nodiscard]] bool AdoptCandidate (PhotonPacketTransaction& accepted_private_trial);

    /** Stage one collective birth batch without using the native allocator.
     *
     * attributes_real[0] is weight, followed by runtime real values;
     * attributes_int contains runtime integer values.  Omitted deterministic
     * runtime columns retain native AppendParticleBatch semantics.  Random and
     * QED default initialization is not supported by this MVP and is rejected
     * along with omitted custom columns that have no native initializer,
     * collectively before initialization. The three exact diffusion census
     * columns radiation_diffusion_census_ux0/uy0/uz0 and two exact resolved
     * draw-key halves have a deterministic zero default for omitted newborn
     * values; arbitrary columns with a similar prefix remain unsupported.
     * A false return is the same numerical
     * rejection on every rank and leaves live and the candidate unchanged.
     * Multiple batches are allowed before validation.
     */
    [[nodiscard]] bool AddProvisionalBirths (
        int staging_grid, long n, amrex::Vector<amrex::ParticleReal> const& x,
        amrex::Vector<amrex::ParticleReal> const& y, amrex::Vector<amrex::ParticleReal> const& z,
        amrex::Vector<amrex::ParticleReal> const& ux, amrex::Vector<amrex::ParticleReal> const& uy,
        amrex::Vector<amrex::ParticleReal> const& uz,
        amrex::Vector<amrex::Vector<amrex::ParticleReal>> const& attributes_real,
        amrex::Vector<amrex::Vector<int>> const& attributes_int, bool unique_particles = true);

    /** Read actual stored E/c*p of surviving private newborns, deposited once
     * per native cell after redistribution. Cartesian components are lab xyz;
     * native RZ uses [E,cPr,cLz/rbar,cPz], projected from actual stored r/theta
     * and Cartesian proper momentum. rbar is the annular volume-mean radius.
     * Global RZ balance uses E/Pz/rbar-weighted Lz, never summed radial momentum.
     * Optional exact logical
     * draw keys isolate one batch from other staged births without subtracting
     * a large background census. No particle/ID state changes; numerical
     * rejection preserves the supplied output field and count.
     */
    [[nodiscard]] bool
    ProvisionalBirthInventory (amrex::MultiFab& cell_four_inventory,
                               amrex::Geometry const& geometry, amrex::Long& count,
                               std::vector<std::uint64_t> const* selected_keys = nullptr) const;

    /** Collective, read-only preflight. Every rank must call it exactly
     * once after candidate mutation/deletion. False means no Commit. */
    bool ValidateLive ();

    /** Replace live photons by the validated candidate. Caller acceptance
     * of all coupled native material, fields and this candidate must
     * precede this call; no coupled physics rejection is allowed after it. */
    void Commit ();

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace warpx::radiation

#endif // WARPX_PHOTON_PACKET_TRANSACTION_H_
