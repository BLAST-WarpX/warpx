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
 * Each filled cell carries a single charge=1 particle of signed weight; with
 * DIRECT current deposition the antenna deposits only J (no charge), so it
 * reproduces the current_fp box injection exactly. Particle positions are
 * reset every step, so the antenna does not drift for a sustained waveform.
 *
 * NOTE: this does NOT remove the need for a closed current path. The injected
 * J is localized, so div(J) != 0 at the box faces and charge conservation
 * still forces a return (locally, via displacement current / the nearest
 * conductor). Prescribed current injection only sustains a current along a
 * conductor when the circuit is closed -- a ring/coil (div J = 0 around the
 * loop) or an explicit drive + return face pair.
 *
 * Configured via the public `warpx.current_injection.*` input block:
 * n_pairs drive faces, each with a box, area A, sign (+/-1), and an optional
 * per-pair waveform file (pair_N.file) that overrides the global
 * current_injection.file. A "return" is just a drive face with sign = -1, so
 * no dedicated return block exists. Each face has its own current direction
 * `dir`. Requires algo.particle_shape >= 1. Works with algo.current_deposition =
 * direct or a charge-conserving scheme (esirkepov/villasenor): positions are
 * pushed by v*dt before deposition (then reset), so the displacement yields
 * J = q w v either way.
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
    //! One axis-aligned drive box with its own waveform.
    struct Face {
        amrex::Real lo[3] = {0.,0.,0.};
        amrex::Real hi[3] = {0.,0.,0.};
        amrex::Real A     = 1.;   //!< Face area perpendicular to dir [m^2]
        int dir           = 0;    //!< Current component: 0=Jx, 1=Jy, 2=Jz
        int sign          = 1;    //!< Current sign: +1 or -1 (J = sign*I(t)/A)
        //! Per-face waveform (from pair_N.file if given, else the global file).
        std::vector<amrex::Real> t;   //!< sample times [s]
        std::vector<amrex::Real> I;   //!< currents [A]
    };

    std::vector<Face> m_faces;

    amrex::Real m_vel_coeff = 0.;  //!< C such that speed = C * I(t) [m/s/A]
    bool       m_enabled    = true;

    //! Linear interpolation of a (t,I) table; returns 0 outside the table.
    [[nodiscard]] static amrex::Real interpolate (
        const std::vector<amrex::Real>& t, const std::vector<amrex::Real>& I, amrex::Real tt);
};

#endif // WARPX_PrescribedCurrentParticleContainer_H_
