/* Copyright 2019 Andrew Myers, David Grote, Maxence Thevenet
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RigidInjectedParticleContainer_H_
#define WARPX_RigidInjectedParticleContainer_H_

#include "Evolve/WarpXDtType.H"
#include "Evolve/WarpXPushType.H"
#include "Particles/Gather/ScaleFields.H"
#include "PhysicalParticleContainer.H"

#include "WarpXParticleContainer_fwd.H"

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>
#include <AMReX_AmrCoreFwd.H>

#include <iosfwd>
#include <string>

/**
 * When injecting a particle beam (typically for a plasma wakefield
 * acceleration simulation), say propagating in the z direction, it can
 * be necessary to make particles propagate in a straight line up to a given
 * location z=z0. This is of particular importance when running in a boosted
 * frame, where the beam may evolve due to its space charge fields before
 * entering the plasma, causing the actual injected beam, and hence the whole
 * simulation result, to depend on the Lorentz factor of the boost.
 *
 * This feature is implemented in RigidInjectedParticleContainer: At each
 * iteration, for each particle, if z<z0 the particle moves in a straight line,
 * and if z>z0 the particle evolves as a regular PhysicalParticleContainer.
 *
 * Note: This option is also useful to build self-consistent space charge
 * fields for the particle beam.
 *
 * RigidInjectedParticleContainer derives from PhysicalParticleContainer.
 */
class RigidInjectedParticleContainer
    : public PhysicalParticleContainer
{
public:
    RigidInjectedParticleContainer (amrex::AmrCore* amr_core,
                                    const std::string& name);
    ~RigidInjectedParticleContainer () override = default;

    RigidInjectedParticleContainer ( RigidInjectedParticleContainer const &)             = delete;
    RigidInjectedParticleContainer& operator= ( RigidInjectedParticleContainer const & ) = delete;
    RigidInjectedParticleContainer ( RigidInjectedParticleContainer&& )                  = default;
    RigidInjectedParticleContainer& operator= ( RigidInjectedParticleContainer&& )       = default;


    void InitData() override;

    virtual void RemapParticles();

    void Evolve (ablastr::fields::MultiFabRegister& fields,
                 int lev,
                 const std::string& current_fp_string,
                 amrex::Real t,
                 amrex::Real dt,
                 DtType a_dt_type=DtType::Full,
                 bool skip_deposition=false,
                 PushType push_type=PushType::Explicit) override;

    void PushPX (WarpXParIter& pti,
                         amrex::FArrayBox const * exfab,
                         amrex::FArrayBox const * eyfab,
                         amrex::FArrayBox const * ezfab,
                         amrex::FArrayBox const * bxfab,
                         amrex::FArrayBox const * byfab,
                         amrex::FArrayBox const * bzfab,
                         amrex::IntVect ngEB, int /*e_is_nodal*/,
                         long offset,
                         long np_to_push,
                         int lev, int gather_lev,
                         amrex::Real dt, ScaleFields scaleFields,
                         DtType a_dt_type=DtType::Full) override;

    void PushP (int lev, amrex::Real dt,
                        const amrex::MultiFab& Ex,
                        const amrex::MultiFab& Ey,
                        const amrex::MultiFab& Ez,
                        const amrex::MultiFab& Bx,
                        const amrex::MultiFab& By,
                        const amrex::MultiFab& Bz) override;

    void ReadHeader (std::istream& is) override;

    void WriteHeader (std::ostream& os) const override;

private:

    // User input quantities
    amrex::ParticleReal zinject_plane = 0.;
    bool rigid_advance = true; // When true, particles are advance with vzbar before injection

    amrex::ParticleReal vzbeam_ave_boosted;

    amrex::Vector<amrex::ParticleReal> zinject_plane_levels;

    // Temporary quantities
    amrex::ParticleReal zinject_plane_lev;
    amrex::ParticleReal zinject_plane_lev_previous;
    bool done_injecting_lev;

};

#endif
