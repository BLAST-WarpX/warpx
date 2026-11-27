/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NATIVEPACKETTHERMALEVENTREPLAY_H_
#define WARPX_RADIATION_NATIVEPACKETTHERMALEVENTREPLAY_H_

#include "FrameTransform.H"
#include "NativeThermalCornerEmission.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace warpx::radiation
{
class PhotonPacketTransaction;

/** Optional OWNER-LOCAL reporting only. Caller owns/reset this single sample;
 * no control, reduction, predicate, counter, result or checkpoint uses it.
 * Records the first local true-to-false gray-capture transition before any
 * photon publication. Identities remain typed uint64, never float metadata.
 */
struct NativePacketThermalEventRefusalTrace
{
    bool captured = false;
    char const* stage = "none";
    amrex::Long record = -1;
    std::uint64_t idcpu = 0, logical_key = 0;
    amrex::IntVect cell{};
    // RZ coordinates are r,z,theta; Cartesian coordinates are x,y,z.
    bool cylindrical_coordinates = false;
    std::array<amrex::Real, 3> coordinates{}, momentum_u{}, rate_beta{}, work_beta{};
    amrex::Real dt = 0, alpha = 0, old_weight = 0, next_weight = 0;
    amrex::Real depth = 0, depth_limit = 0, temperature = 0, incidence = 0;
    amrex::Real source_energy = 0;
    FourVector incoming{}, outgoing{};
    bool incoming_valid = false, outgoing_valid = false;
};

struct NativePacketThermalEventOptions
{
    std::uint64_t seed = 0;
    std::uint64_t interval = 0;
    int emission_rays_per_cell = 8;
    amrex::Real photon_energy = PhysConst::q_e;
    amrex::Real radiation_constant = 7.565733250280007e-16;
    amrex::Real max_absorption_optical_depth = 16;
    bool emission = true;
    /** Exact constant-trial-temperature endpoint source: births include their
     * continuous survival over the source duration. Required for finite-tau
     * stationary LTE and the coupled stiff source. */
    bool survival_weighted_emission = false;
    /** Explicit source-only RZ endpoint contract. Defaults false, preserving
     * Cartesian guards. Uses annular birth volumes and local angular-conjugate
     * debit [E,cPr,cLz/rbar,cPz]; reciprocal corner emission requires its
     * separate selector below.
     * The initial/trial physical speed at each cell's outer radius is <=1e-3 c. */
    bool rz_angular_conjugate = false;
    /** Explicit reciprocal RZ corner bath and two-point annular births.
     * Requires rz_angular_conjugate and frozen OLD Cv times corrected native-FV
     * corner incidence on every Replay. Each corner bath is uniform across the
     * full cell annulus; W is caloric ownership, not radial emissivity support.
     * Defaults false; the mean-radius endpoint is unchanged. */
    bool rz_reciprocal_corner_emission = false;
    /** CPU-only fixed-position mean-radius frozen-bath helper with stationary
     * z PEC/reflecting endcaps. Defaults false and forbids reciprocal corner
     * emission. This is not a flight, census, coupled-source/history or normal
     * runtime boundary activation; their independent guards remain periodic. */
    bool rz_reflecting_axial_frozen_bath = false;
    /** Separate CPU zero-flight reciprocal endpoint selector. Only the closed
     * reciprocal source preflight forwards this; no flight/runtime activation.
     * Mutually exclusive with the mean-radius frozen-bath selector. */
    bool rz_reflecting_axial_reciprocal_endpoint = false;
    // Default-null, caller-owned first-refusal observation only. No new MPI
    // controls/collectives or serialized identity; NULL keeps old computation.
    NativePacketThermalEventRefusalTrace* refusal_trace = nullptr;
};

struct NativePacketThermalEventResult
{
    bool valid = false;
    std::string reason;
    amrex::Long attenuated_packets = 0;
    amrex::Long emission_rays = 0;
    amrex::Real absorbed_energy = 0;
    amrex::Real emitted_energy = 0;
    /** Actual staged emission-plan keys, one receipt per newborn. Private
     * history uses these to distinguish births from illegal old-key changes. */
    std::vector<std::uint64_t> emission_birth_keys;
    /** Signed actual native [E_before-E_after,c*(P_before-P_after)] in J/cell.
     * Explicit RZ mode stores local [E,cPr,cLz/rbar,cPz], not global Pr. */
    amrex::MultiFab radiation_debit;
    /** Absolute before/after operands for each represented interaction, in J. */
    amrex::MultiFab operand_scale;
    /** New reciprocal model only: negative ACTUAL stored birth four-vector
     * by immutable cell-corner origin, four components per corner. RZ stores
     * local [E,cPr,cLz/rbar,cPz] at the actual birth radius. Capture
     * is not assigned here; its single common caloric allocation is separate. */
    amrex::MultiFab corner_emission_debit;
};

/** Continuous gray packet capture and keyed comoving thermal birth source.
 *
 * This is a fixed-position source operator of laboratory duration dt, applied
 * to the exact supplied private snapshot. The chronological path replay owns
 * all streaming, cell crossings, scattering clocks and collision histories;
 * the caller chooses its first-order source/transport splitting. This helper does not
 * integrate absorption along the preceding trajectory. Source-cell ownership,
 * proper chi_a and initial rate velocity are frozen by Prepare. Capture uses
 * w_new=ParticleReal(w_old*exp(-c*dt*chi_a*gamma0*(1-beta0.n_lab))). Actual
 * debit and impulse come from the stored before/after weight and momenta,
 * including zero debit when the change cannot be represented. No whole-packet
 * Bernoulli selection, optical-depth clipping or photon threshold is used.
 *
 * Emission uses frame energy c*dt*V*chi_a*a_R*T^4. Uniform comoving directions
 * are keyed by (seed, interval, canonical cell, ray); repeated temperature and
 * secant-velocity trials reuse the same directions and birth keys. Frame rays
 * are boosted to the supplied work frame. Gray photon_energy sets a tracer's
 * comoving energy and its statistical weight, not a Planck spectrum. Births
 * occur at cell centers at this source endpoint. Actual stored native births
 * are independently remeasured; sampled momentum is never discarded or
 * replaced by its isotropic expectation. Finite native ion recoil and heat
 * are owned by the caller's single coupled material/caloric transaction.
 *
 * Scope: level-zero periodic Cartesian, double field and particle precision,
 * fixed gray proper opacity, beta0 and trial beta <= .01. Temperature is a
 * supplied nonnegative cell-centered K field; no EOS is owned here. Photon
 * logical-key columns must already be registered if emission is enabled.
 * All calls are collective, including empty ranks. Prepare and Replay consume
 * no global RNG, native birth IDs, live fields or material state. Replay changes
 * only the caller-owned photon candidate. A failure requires discarding that
 * entire private transaction, including any prior caller stages; success does
 * not authorize a standalone commit. The outer kernel combines these actual
 * accounts with coherent packet and diffusion sources, stages one native ion
 * kick, and requests Q_total=D_packet+Q_diffusion-(actual_work+delta_carry).
 * The separately selected RZ endpoint option instead permits m0 axis-containing
 * native RZ, stationary radial PEC/reflecting material wall and periodic z.
 * Births occur at the annular mean radius with azimuth zero; Cartesian native
 * photon directions and actual sampled torque are retained. Local rate velocity
 * is (vr,vtheta-conjugate,vz), with physical tangential speed vtheta*r/rbar.
 * This is still frozen-temperature endpoint splitting, not path absorption,
 * implicit thermal relaxation or a K+D runtime. A separate explicit selector
 * permits reciprocal corner emission with two annular radial birth points;
 * it requires frozen old native-FV corner incidence supplied by its caller.
 */
class NativePacketThermalEventReplay
{
public:
    NativePacketThermalEventReplay ();
    ~NativePacketThermalEventReplay ();
    NativePacketThermalEventReplay (NativePacketThermalEventReplay const&) = delete;
    NativePacketThermalEventReplay& operator=(NativePacketThermalEventReplay const&) = delete;

    [[nodiscard]] bool Prepare (
        WarpXParticleContainer::Base const& private_snapshot,
        WarpXParticleContainer const& photon_species, amrex::iMultiFab const& kinetic_mask,
        amrex::MultiFab const& proper_absorption_opacity,
        amrex::MultiFab const& initial_rate_velocity, amrex::Geometry const& geometry,
        amrex::Real dt, NativePacketThermalEventOptions const& options, std::string& reason,
        amrex::iMultiFab const* capture_mask = nullptr);

    [[nodiscard]] NativePacketThermalEventResult
    Replay (amrex::MultiFab const& cell_temperature, amrex::MultiFab const& work_velocity,
            PhotonPacketTransaction& private_photons,
            NativeThermalCornerEmission const* reciprocal_emission = nullptr) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace warpx::radiation

#endif
