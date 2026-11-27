/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_CAUSALSCATTERINGLAYER_H_
#define WARPX_RADIATION_CAUSALSCATTERINGLAYER_H_

#include "FiniteMassPhotonCollision.H"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace warpx::radiation
{
    enum class LayerDisposition { Resident, EscapedLeft, EscapedRight };

    struct CausalLayerPacket
    {
        std::uint64_t id = 0;
        amrex::Real z = 0;
        FourVector photon{}; // Actual [E,c*p], J; no ParticleReal quantization.
        amrex::Real remaining_tau = -1; // -1: uninitialized; zero is a valid event clock.
        std::uint64_t rng_counter = 0;
        LayerDisposition disposition = LayerDisposition::Resident;
        std::uint64_t collision_count = 0;
    };

    struct CausalLayerSlab
    {
        amrex::Real lo = 0, hi = 1, chi = 0;
        amrex::GpuArray<amrex::Real, 3> initial_rate_beta{}, work_beta{};
    };

    struct CausalLayerResult
    {
        bool valid = false;
        std::string reason;
        std::vector<CausalLayerPacket> candidate;
        FourVector collision_transfer{}, escaped_inventory{}, resident_inventory{}; // [E,c*p], J
        std::uint64_t collisions = 0;
        std::vector<amrex::Real> escape_times; // Relative seconds; -1 means no NEW escape.
        std::vector<std::uint64_t> collision_counts; // New events, aligned with candidate IDs.
    };

    inline constexpr char CausalLayerRngVersion[] = "causal-layer-splitmix-v1";

    namespace causal_layer_detail
    {
        inline bool Uniform (std::uint64_t seed, CausalLayerPacket& packet, amrex::Real& u)
        {
            if (packet.rng_counter == std::numeric_limits<std::uint64_t>::max()) { return false; }
            auto value = (seed ^ (packet.id * 0xD2B74407B1CE6E93ULL) ^
                (packet.rng_counter * 0xCA5A826395121157ULL)) + 0x9E3779B97F4A7C15ULL;
            value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
            value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
            value ^= value >> 31;
            ++packet.rng_counter;
            constexpr int bits = std::numeric_limits<amrex::Real>::digits - 1;
            u = std::ldexp(static_cast<amrex::Real>(value >> (64-bits)) + amrex::Real(0.5), -bits);
            return u > 0 && u < 1;
        }

        inline CausalLayerResult Reject (std::string const& reason)
        {
            CausalLayerResult result;
            result.reason = reason;
            return result;
        }
    }

    /** Host-only homogeneous layer trial, 1D position/full xyz photon momentum.
     * Input is immutable; numerical/event-budget failure returns no candidate
     * and zero inventories. Work scattering reuses CoherentPhotonCollision;
     * rates use the separately frozen initial frame. No material kick is committed.
     * Caller must serialize RNG version, real precision, seed and ALL packet fields. Neither dt,
     * census nor retry changes the draw key. An escaped packet can be reactivated
     * at a valid entry face, preserving clock/counters and identity. Escape times
     * allow continuation outside this slab during the same timestep.
     * Not a native-particle adapter, moving geometric surface or FLD admission.
     */
    inline CausalLayerResult AdvanceCausalScatteringLayer (
        std::vector<CausalLayerPacket> const& packets, CausalLayerSlab const& slab,
        amrex::Real dt, std::uint64_t seed, std::uint64_t max_events_per_packet = 10000)
    {
        using causal_layer_detail::Reject;
        using causal_layer_detail::Uniform;
        auto const flight = PhysConst::c * dt;
        if (!std::isfinite(dt) || dt < 0 || !std::isfinite(flight) ||
            !std::isfinite(slab.lo) || !std::isfinite(slab.hi) || !(slab.hi > slab.lo) ||
            !std::isfinite(slab.hi-slab.lo) || !std::isfinite(slab.chi) || slab.chi < 0 ||
            max_events_per_packet == 0) { return Reject("Invalid layer controls"); }
        amrex::Real rate_beta2 = 0, work_beta2 = 0;
        for (int d = 0; d < 3; ++d) {
            if (!std::isfinite(slab.initial_rate_beta[d]) || !std::isfinite(slab.work_beta[d])) {
                return Reject("Nonfinite layer frame");
            }
            rate_beta2 += slab.initial_rate_beta[d]*slab.initial_rate_beta[d];
            work_beta2 += slab.work_beta[d]*slab.work_beta[d];
        }
        if (!(rate_beta2 < 1) || !(work_beta2 < 1)) { return Reject("Superluminal frame"); }
        std::unordered_set<std::uint64_t> ids;
        for (auto const& p : packets) {
            bool const disposition = p.disposition == LayerDisposition::Resident ||
                (p.disposition == LayerDisposition::EscapedLeft && p.z == slab.lo) ||
                (p.disposition == LayerDisposition::EscapedRight && p.z == slab.hi);
            if (!ids.insert(p.id).second || !ValidFiniteMassPhoton(p.photon) ||
                !std::isfinite(p.z) || p.z < slab.lo || p.z > slab.hi || !disposition ||
                !std::isfinite(p.remaining_tau) || (p.remaining_tau < 0 &&
                    !(p.remaining_tau == -1 && p.rng_counter == 0 && p.collision_count == 0))) {
                return Reject("Malformed packet/census");
            }
        }
        CausalLayerResult result;
        result.candidate = packets;
        result.escape_times.assign(packets.size(), -1);
        result.collision_counts.assign(packets.size(), 0);
        auto const gamma_rate = 1/std::sqrt(1-rate_beta2);
        for (std::size_t index = 0; index < packets.size(); ++index) {
            auto& p = result.candidate[index];
            auto& events = result.collision_counts[index];
            amrex::Real left = p.disposition == LayerDisposition::Resident ? flight : 0;
            if (left > 0 && p.remaining_tau == -1) {
                amrex::Real u;
                if (!Uniform(seed, p, u)) { return Reject("Clock RNG exhausted"); }
                p.remaining_tau = -std::log(u);
            }
            while (left > 0) {
                amrex::Real projection = 0;
                for (int d = 0; d < 3; ++d) {
                    projection += p.photon[d+1]/p.photon[0]*slab.initial_rate_beta[d];
                }
                auto const rate = slab.chi*gamma_rate*(1-projection);
                if (!std::isfinite(rate) || rate < 0) { return Reject("Invalid flight rate"); }
                auto const nz = p.photon[3]/p.photon[0];
                auto const collision_distance = rate > 0 ? p.remaining_tau/rate :
                    std::numeric_limits<amrex::Real>::infinity();
                auto const boundary_distance = nz != 0 ? ((nz > 0 ? slab.hi : slab.lo)-p.z)/nz :
                    std::numeric_limits<amrex::Real>::infinity();
                if (boundary_distance < 0) { return Reject("Negative boundary distance"); }
                bool const boundary = boundary_distance <= collision_distance &&
                    boundary_distance <= left;
                bool const scatter = !boundary && collision_distance <= left;
                auto const distance = boundary ? boundary_distance :
                    (scatter ? collision_distance : left);
                if (boundary) {
                    result.escape_times[index] = (flight-left+distance)/PhysConst::c;
                    p.z = nz > 0 ? slab.hi : slab.lo;
                    p.disposition = nz > 0 ? LayerDisposition::EscapedRight :
                        LayerDisposition::EscapedLeft;
                } else {
                    p.z += distance*nz;
                }
                left -= distance;
                if (!std::isfinite(p.z) || p.z < slab.lo || p.z > slab.hi || left < 0) {
                    return Reject("Unrepresentable flight");
                }
                if (!scatter) {
                    p.remaining_tau = std::fma(-rate, distance, p.remaining_tau);
                    if (!std::isfinite(p.remaining_tau) || p.remaining_tau < 0) {
                        return Reject("Unrepresentable optical clock");
                    }
                    break;
                }
                if (events == max_events_per_packet ||
                    p.collision_count == std::numeric_limits<std::uint64_t>::max() ||
                    result.collisions == std::numeric_limits<std::uint64_t>::max()) {
                    return Reject("Collision budget/counter exhausted");
                }
                amrex::Real u, v;
                if (!Uniform(seed, p, u) || !Uniform(seed, p, v)) {
                    return Reject("Angle RNG exhausted");
                }
                auto const mu = 2*u-1;
                auto const phi = 2*MathConst::pi*v;
                auto const transverse = std::sqrt(1-mu*mu);
                amrex::GpuArray<amrex::Real, 3> direction{
                    transverse*std::cos(phi), transverse*std::sin(phi), mu};
                auto const collision = CoherentPhotonCollision(p.photon, direction, slab.work_beta);
                if (!collision.valid) { return Reject("Coherent collision rejected"); }
                for (int d = 0; d < 4; ++d) {
                    result.collision_transfer[d] += p.photon[d]-collision.outgoing_photon[d];
                }
                p.photon = collision.outgoing_photon;
                ++events;
                ++p.collision_count;
                ++result.collisions;
                if (!Uniform(seed, p, u)) { return Reject("Clock RNG exhausted"); }
                p.remaining_tau = -std::log(u);
            }
            auto& inventory = p.disposition == LayerDisposition::Resident
                ? result.resident_inventory : result.escaped_inventory;
            for (int d = 0; d < 4; ++d) { inventory[d] += p.photon[d]; }
        }
        for (int d = 0; d < 4; ++d) {
            if (!std::isfinite(result.collision_transfer[d]) ||
                !std::isfinite(result.resident_inventory[d]) ||
                !std::isfinite(result.escaped_inventory[d])) {
                return Reject("Nonfinite layer inventory");
            }
        }
        result.valid = true;
        return result;
    }
}
#endif
