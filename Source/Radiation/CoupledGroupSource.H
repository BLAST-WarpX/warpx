/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_COUPLEDGROUPSOURCE_H_
#define WARPX_RADIATION_COUPLEDGROUPSOURCE_H_

#include "CoupledImplicitDiffusion.H"
#include "ExactPlanckBinInteraction.H"
#include "ImplicitGroupMomentTransport.H"
#include "ParticleImpulse.H"

namespace warpx::radiation
{
struct CoupledGroupSourceCallbacks
{
    /** At a supplied nodal temperature trial, fill proper chi_t/chi_a [1/m]
     * in N cell components and positive material/color temperature [K] in
     * one cell component. Density/composition stay frozen across the stage.
     * The same cell temperature defines LTE Planck emission and the declared
     * comoving within-bin shape. Only explicit outputs may be mutated. */
    std::function<void(amrex::MultiFab const &, amrex::MultiFab &, amrex::MultiFab &,
                       amrex::MultiFab &, amrex::Real)>
        coefficients;
    // Actual native caloric response, raw EOS-energy metric and per-node
    // independently perturbed source bracket. coefficients in this adapter
    // is unused: proper coefficients above have different units.
    CoupledImplicitCallbacks caloric;
};

struct CoupledGroupSourceOptions
{
    EnergyGroupsExecutor groups;
    int max_iterations = 100;
    amrex::Real tolerance = 1.e-10;
    amrex::Real energy_tolerance = 1.e-10;
    amrex::Real relaxation = 0.5;
    ParticleImpulseAssignment assignment = ParticleImpulseAssignment::NearestCell;
    bool verbose = false;
    // Opt in to a joint spatial+source trial INSIDE the same native caloric /
    // finite-kick iteration. This is not operator-split source damping.
    bool spatial_transport = false;
    ImplicitGroupMomentTransportOptions spatial;
    // Caller-private explicit radiation/temperature/heat fields only. Retain
    // the accepted actual native material candidate for one later publication.
    // Defaults preserve the established immediate native commit.
    bool defer_live_commit = false;
};

struct CoupledGroupSourceResult
{
    bool valid = false;
    std::string reason;
    int iterations = 0;
    amrex::Real source_residual = 0;
    amrex::Real caloric_residual = 0;
    amrex::Real work_residual = 0;
    amrex::Real raw_energy_residual = 0;
    // Signed accepted radiation + realized electron + actual ion kinetic +
    // energy-carry + outward wall inventory defect [J], before normalization.
    // A diagnostic for the native numerical-energy ledger, never a source
    // reconciliation or physical transfer.
    amrex::Real energy_inventory_defect = 0;
    amrex::Real raw_momentum_residual = 0;
    // RZ: local/conjugate components including outward wall - cylindrical
    // connection inventory when spatial, not global radial momentum.
    amrex::Real momentum_balance_residual = 0;
    amrex::Real raw_angular_momentum_residual = 0;
    amrex::Real angular_balance_residual = 0;
    amrex::Real actual_kinetic_work = 0;
    amrex::Real energy_carry_change = 0;
    amrex::MultiFab requested_material_energy;
    amrex::MultiFab actual_material_impulse;
    amrex::MultiFab actual_material_work;
    amrex::MultiFab conjugate_material_impulse;
    std::unique_ptr<ImplicitGroupMomentTransportResult> spatial_trial;
    std::unique_ptr<ParticleImpulseMaterial> private_material;
};

/** Private native multigroup heat/recoil/work stage (source-only by default). Radiation
 * stores N LAB extensive (E_g,V F_g/c) blocks. Exact Lorentz spectral assembly
 * closes independent COMOVING Planck-bin dipoles; physical chi_a/chi_t refresh
 * at every material-temperature trial. One summed impulse stages an actual
 * finite-mass particle kick on private material. Q_total is retained physical
 * radiation energy debit minus actual native ion work and energy carry.
 * Local EOS, physical heat/work/secant, source and raw inventory gates must
 * all pass before commit. Every failure preserves caller fields/particles.
 * With defer_live_commit=true, all three explicit fields must be caller-private;
 * acceptance updates them and returns the actual staged private_material owner,
 * leaving all native species/IDs untouched until its final Commit. Destroying
 * that owner abandons the accepted trial. Default immediate publication is unchanged.
 * With spatial_transport=true, a joint all-group spatial/source solve runs
 * inside EVERY same caloric/secant trial; spatial redistribution is excluded
 * from Q_total, and accepted LAB states as well as transfers are checked at
 * the updated native temperature/secant. There is no split opacity damping.
 * This API performs no drift, checkpoint or live runtime activation. Initial
 * scope is periodic Cartesian level zero, double precision,
 * nearest-cell material assignment and at most eight proper spectral groups.
 * RZ admission additionally requires an axis, reflecting radial
 * optical/material boundary, periodic z, one mode and RZAngularConservative
 * assignment. Its LAB basis is (r,theta,z), and angular inventory is
 * sum(mean_annular_radius * Qtheta_g/c); actual physical theta impulse and
 * torque-conjugate impulse remain distinct. Joint RZ spatial transport also
 * requires spatial.reflecting_boundaries=true, declaring the bounded P1
 * optical mirror. Its accepted spatial_trial retains separate external wall
 * and cylindrical-connection ledgers. No global radial vector balance or
 * exact kinetic half-space wall pressure is claimed by this adapter. */
CoupledGroupSourceResult TryAdvanceCoupledGroupSource (
    MultiParticleContainer &particles, std::vector<std::string> const &species,
    std::string const &carry_path, amrex::MultiFab &lab_radiation,
    amrex::MultiFab &nodal_temperature, amrex::MultiFab &realized_material_energy,
    amrex::Geometry const &geometry, amrex::Real time, amrex::Real dt,
    CoupledGroupSourceOptions const &options, CoupledGroupSourceCallbacks const &callbacks);
} // namespace warpx::radiation
#endif
