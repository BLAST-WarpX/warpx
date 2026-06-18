/* Copyright 2026 WarpX contributors
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PrescribedCurrentParticleContainer_H_
#define WARPX_PrescribedCurrentParticleContainer_H_

#include "WarpXParticleContainer.H"

#include "FieldSolver/ImplicitSolvers/ImplicitOptions.H"

#include <AMReX_AmrCoreFwd.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_REAL.H>

#include <string>
#include <vector>

/**
 * PrescribedCurrentParticleContainer imposes a user-prescribed, time-varying
 * current density J = I(t)/A over one or more axis-aligned rectangular boxes,
 * using artificial macro-particles deposited through the standard WarpX
 * current-deposition path (like LaserParticleContainer).
 *
 * Compared with writing J directly into `current_fp`, depositing through the
 * particle path means the imposed current (a) shows up in the Jx/Jy/Jz
 * diagnostics and (b) is handled identically by the explicit and implicit
 * field solvers.
 *
 * Each filled cell carries a +/- weight particle pair (charge=1). The two
 * partners move in opposite directions, so the pair deposits a net current
 * with **zero net charge** -- no explicit "return" face is required to keep
 * the source charge-neutral. Particle positions are reset every step, so the
 * antenna does not drift even for a sustained unidirectional waveform.
 *
 * Configured via the public `warpx.current_injection.*` input block.
 *
 * Current scope (first port of PR #6766's public feature): all drive boxes
 * share a single current direction `dir`, sign = +1, and a single global
 * waveform file. Return faces / sign = -1 / mixed directions abort with a
 * message (these need per-particle metadata and are a follow-up).
 */
class PrescribedCurrentParticleContainer
    : public WarpXParticleContainer
{
public:
    PrescribedCurrentParticleContainer (amrex::AmrCore* amr_core, int ispecies);
    ~PrescribedCurrentParticleContainer () override = default;

    PrescribedCurrentParticleContainer ( PrescribedCurrentParticleContainer const &)             = delete;
    PrescribedCurrentParticleContainer& operator= ( PrescribedCurrentParticleContainer const & ) = delete;
    PrescribedCurrentParticleContainer ( PrescribedCurrentParticleContainer&& )                  = default;
    PrescribedCurrentParticleContainer& operator= ( PrescribedCurrentParticleContainer&& )       = default;

    void InitData () final;

    void DefaultInitializeRuntimeAttributes (
        typename ContainerLike<amrex::PolymorphicArenaAllocator>::ParticleTileType& /*pinned_tile*/,
        int /*n_external_attr_real*/,
        int /*n_external_attr_int*/) final {}

    void ReadHeader (std::istream& /*is*/) final {}
    void WriteHeader (std::ostream& /*os*/) const final {}

    void Evolve (ablastr::fields::MultiFabRegister& fields,
                 int lev,
                 const std::string& current_fp_string,
                 amrex::Real t, amrex::Real dt, SubcyclingHalf subcycling_half=SubcyclingHalf::None,
                 bool skip_deposition=false,
                 PositionPushType position_push_type=PositionPushType::Full,
                 MomentumPushType momentum_push_type=MomentumPushType::Full,
                 ImplicitOptions const * implicit_options = nullptr) final;

    void PushP (int /*lev*/, amrex::Real /*dt*/,
                const amrex::MultiFab&, const amrex::MultiFab&, const amrex::MultiFab&,
                const amrex::MultiFab&, const amrex::MultiFab&, const amrex::MultiFab&,
                MomentumPushType /*momentum_push_type*/ = MomentumPushType::Full) final {}

    void PostRestart () final {}

    /** Whether `warpx.current_injection = 1` is set in the inputs. */
    static bool is_enabled ();

private:
    //! One axis-aligned drive box.
    struct Face {
        amrex::Real lo[3] = {0.,0.,0.};
        amrex::Real hi[3] = {0.,0.,0.};
        amrex::Real A     = 1.;   //!< Face area perpendicular to dir [m^2]
        int dir           = 0;    //!< Current component: 0=Jx, 1=Jy, 2=Jz
        int sign          = 1;    //!< Current sign (only +1 supported for now)
    };

    std::vector<Face>        m_faces;
    std::vector<amrex::Real> m_time;     //!< waveform sample times [s]
    std::vector<amrex::Real> m_current;  //!< waveform currents [A]

    int        m_dir        = 0;   //!< Common current direction of all faces
    amrex::Real m_vel_coeff = 0.;  //!< C such that speed = C * I(t) [m/s/A]
    bool       m_enabled    = true;

    //! Linear interpolation of the waveform; returns 0 outside the table.
    [[nodiscard]] amrex::Real interpolate_current (amrex::Real t) const;
};

#endif // WARPX_PrescribedCurrentParticleContainer_H_
