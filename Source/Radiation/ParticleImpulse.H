/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_PARTICLEIMPULSE_H_
#define WARPX_RADIATION_PARTICLEIMPULSE_H_

#include <AMReX_MultiFab.H>

#include <memory>
#include <string>
#include <vector>

class MultiParticleContainer;
class WarpXParticleContainer;

namespace warpx::radiation
{
    enum class ParticleImpulseAssignment {
        NearestCell, LinearNodalCellAverage, NativeNodalCellAverage,
        /** Low-level 1D stationary PEC/reflecting adapter; not runtime boundary support. */
        ReflectingNodalCellAverage
    };
    /** Private kinetic-material copy for a multi-stage radiation interval.
     * Substeps may change velocities and radiation carry only. Live particles
     * must remain untouched and retain their ordering until Commit. Destroying
     * this object discards all trial substeps; no rollback writes are needed.
     */
    class ParticleImpulseMaterial
    {
    public:
        ParticleImpulseMaterial (MultiParticleContainer& particles,
                                 std::vector<std::string> const& species_names);
        ~ParticleImpulseMaterial ();
        ParticleImpulseMaterial (ParticleImpulseMaterial const&) = delete;
        ParticleImpulseMaterial& operator= (ParticleImpulseMaterial const&) = delete;
        ParticleImpulseMaterial (ParticleImpulseMaterial&&) = delete;
        ParticleImpulseMaterial& operator= (ParticleImpulseMaterial&&) = delete;
        bool Commit ();

    private:
        friend class ParticleImpulseTransaction;
        ParticleImpulseMaterial (MultiParticleContainer& particles,
                                 std::vector<std::string> const& species_names, bool clone);
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };

    /** Register communicated, checkpointed per-rest-mass numerical residuals
     * before particle initialization. Each radiation path/group owns its state.
     * This does not enable a radiation model or migrate old cell-owned state. */
    void RegisterParticleImpulseState (WarpXParticleContainer& species, std::string const& path);

    /** Global Cartesian pending impulse [kg m/s] and pending work [J]. */
    amrex::GpuArray<amrex::Real, 4> ParticleImpulseInventory (
        MultiParticleContainer& particles, std::vector<std::string> const& species_names,
        std::string const& path);

    /** Stage a material impulse without mutating any live particle or reservoir.
     * The particle layout and velocities must remain unchanged between Stage and
     * Commit. Only Commit writes the accepted, previously evaluated candidates;
     * it does not repeat an atomic work reduction. Destroying the transaction
     * discards the candidate. Caller must validate radiation and native caloric
     * response together with these fields before committing any of them.
     */
    class ParticleImpulseTransaction
    {
    public:
        ParticleImpulseTransaction ();
        ~ParticleImpulseTransaction ();
        ParticleImpulseTransaction (ParticleImpulseTransaction const&) = delete;
        ParticleImpulseTransaction& operator= (ParticleImpulseTransaction const&) = delete;
        ParticleImpulseTransaction (ParticleImpulseTransaction&&) = delete;
        ParticleImpulseTransaction& operator= (ParticleImpulseTransaction&&) = delete;

        bool Stage (MultiParticleContainer& particles,
                    std::vector<std::string> const& species_names, std::string const& path,
                    amrex::MultiFab const& cell_impulse, bool need_work_velocity = false,
                    ParticleImpulseAssignment assignment = ParticleImpulseAssignment::NearestCell);
        bool Stage (ParticleImpulseMaterial& material, std::string const& path,
                    amrex::MultiFab const& cell_impulse, bool need_work_velocity = false,
                    ParticleImpulseAssignment assignment = ParticleImpulseAssignment::NearestCell);
        void Commit ();

        /** Source-origin work. For overlapping assignment its global sum
         * differs from the independent particle finite-work sum by the bounded
         * WorkPartitionResidual, not by a freely adjustable energy account. */
        [[nodiscard]] amrex::MultiFab const& RequestedWork () const;
        [[nodiscard]] amrex::MultiFab const& ActualWork () const;
        [[nodiscard]] amrex::MultiFab const& ActualImpulse () const;
        [[nodiscard]] amrex::MultiFab const& EnergyCarryChange () const;
        [[nodiscard]] amrex::MultiFab const& MomentumCarryChange () const;
        [[nodiscard]] amrex::MultiFab const& NumericalEnergyResidual () const;
        /** Source-cell work partition minus independent particle finite work,
         * deposited at particle centers. Bounded from the uncancelled operands
         * during staging; zero for nearest-cell assignment. ActualWork and
         * actual/carry impulses always retain particle-center ownership. */
        [[nodiscard]] amrex::MultiFab const& WorkPartitionResidual () const;
        [[nodiscard]] amrex::MultiFab const& MaterialMass () const;
        /** Mass-weighted finite-kick secant velocity, in the impulse field's
         * coordinate basis. Available only when requested during Stage. The
         * scalar RequestedWork remains authoritative for cancellation-sensitive
         * work; dotting this rounded mean with impulse is not its replacement. */
        [[nodiscard]] amrex::MultiFab const& WorkVelocity () const;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };
}
#endif
