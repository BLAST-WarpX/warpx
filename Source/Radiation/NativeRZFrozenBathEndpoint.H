/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NATIVERZFROZENBATHENDPOINT_H_
#define WARPX_RADIATION_NATIVERZFROZENBATHENDPOINT_H_

#include "NativePacketThermalEventReplay.H"
#include "NativeRZOwnedNodalCaloric.H"
#include "ParticleImpulse.H"
#include "PhotonPacketTransaction.H"

class HybridPICModel;

namespace warpx::radiation
{
struct NativeRZFrozenBathEndpointResult
{
    bool valid = false;
    std::string reason;
    NativePacketThermalEventResult packet;
    amrex::MultiFab requested_heat;
    amrex::MultiFab actual_heat;
    amrex::MultiFab nodal_temperature;
    std::unique_ptr<PhotonPacketTransaction> photons;
    std::unique_ptr<ParticleImpulseTransaction> material_stage;
    amrex::Real caloric_residual = 0;
};

/** Caller-private, first-order RZ thermal source at a fixed transport endpoint.
 * The temperature and proper absorption rate are frozen during this one
 * nonstiff step. One native angular-conservative ion kick consumes the ACTUAL
 * stored photon debit; the native FV electron inverse consumes debit minus
 * finite ion work and energy carry. No photon flight, reciprocal corner law,
 * implicit relaxation, D field or public mutation is performed here.
 * Returned stages must be accepted by the caller's outer joint transaction.
 * The optional closed-z caloric helper scope is CPU/PEC/reflecting only and
 * does not activate normal radiation, native source/history or census paths.
 */
[[nodiscard]] NativeRZFrozenBathEndpointResult TryNativeRZFrozenBathEndpoint (
    WarpXParticleContainer& live_photons, PhotonPacketTransaction const& endpoint,
    ParticleImpulseMaterial& material, std::string const& carry_path,
    amrex::MultiFab const& nodal_temperature, amrex::MultiFab const& proper_absorption,
    amrex::MultiFab const& initial_rate_velocity, HybridPICModel const& hybrid,
    amrex::Geometry const& geometry, amrex::Real dt,
    NativePacketThermalEventOptions options, amrex::Real minimum_electron_density = 1,
    NativeRZCaloricOptions const& caloric_options = {});
} // namespace warpx::radiation

#endif
