/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_FLUX_RELAXED_PACKETS_H_
#define WARPX_FLUX_RELAXED_PACKETS_H_

#include "Particles/WarpXParticleContainer.H"
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <cstdint>

namespace warpx::radiation
{
class PhotonPacketTransaction;
}

namespace warpx::radiation::detail
{
/** Deferred native births, with actual [E,c*p] inventory [J].
 * Trial code never allocates photon IDs or touches live particles.
 */
struct EmissionPlan
{
    amrex::Vector<amrex::ParticleReal> x;
    amrex::Vector<amrex::ParticleReal> y;
    amrex::Vector<amrex::ParticleReal> z;
    amrex::Vector<amrex::ParticleReal> ux;
    amrex::Vector<amrex::ParticleReal> uy;
    amrex::Vector<amrex::ParticleReal> uz;
    amrex::Vector<amrex::ParticleReal> weight;
    /** Optional per-ray logical RNG identity. Empty retains the legacy
     * native-ID stream for generic low-level emission plans. */
    amrex::Vector<std::uint64_t> logical_draw_keys;
    amrex::GpuArray<amrex::Real, 4> inventory{};
};

/** Stage an emission plan in the private photon candidate, collectively.
 * No native IDs are consumed and no live particles change. The candidate is
 * redistributed before returning so its actual inventory can be validated.
 * Failure requires discarding the entire caller-owned trial.
 */
bool StageEmissionPlan (PhotonPacketTransaction& transaction, EmissionPlan const& plan);

/** Relabel initially thick packets, then stream remaining private packets.
 * Collective. No material impulse is applied by representation conversion.
 */
bool AdvanceCandidatePhotons (WarpXParticleContainer::Base& photons, amrex::MultiFab& radiation,
                              amrex::MultiFab const& opacity, amrex::MultiFab const& beta,
                              amrex::Geometry const& geometry, amrex::Real dt,
                              amrex::Real minimum_optical_depth);

/** Collective outgoing reconstruction. The caller discards all private state
 * and this plan if any rank rejects; no new photon is created here.
 */
bool BuildThinCellEmissionPlan (amrex::MultiFab& radiation, amrex::MultiFab const& opacity,
                                amrex::Geometry const& geometry, amrex::Real min_cell_size,
                                amrex::Real minimum_optical_depth, amrex::Real energy_threshold,
                                amrex::Real photon_energy, EmissionPlan& plan);

/** Actual native packet energy and c times Cartesian momentum [J]. */
bool PhotonInventory (WarpXParticleContainer::Base& photons,
                      amrex::GpuArray<amrex::Real, 4>& result);
} // namespace warpx::radiation::detail
#endif
