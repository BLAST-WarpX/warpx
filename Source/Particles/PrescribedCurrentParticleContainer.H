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
#include "Utils/CurrentWaveform.H"

#include <AMReX_AmrCoreFwd.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_REAL.H>

#include <string>
#include <vector>

/**
 * PrescribedCurrentParticleContainer is an impressed-current antenna that
 * imposes a user-prescribed, time-varying current density J = I(t)/A over one
 * or more axis-aligned rectangular boxes, using artificial macro-particles
 * deposited through the standard WarpX current-deposition path (like
 * LaserParticleContainer).
 *
 * Compared with writing J directly into `current_fp`, depositing through the
 * particle path means the imposed current (a) shows up in the Jx/Jy/Jz
 * diagnostics and (b) is handled identically by the explicit and implicit
 * field solvers.
 *
 * Each filled cell carries a coincident pair of opposite signed weights.  The
 * pair members move in opposite directions, so their currents add while their
 * charge separation supplies the rho required by the discrete continuity
 * equation.  Their maximum excursion is kept below a fraction of one cell by
 * choosing the particle weight from the time-integrated waveform.
 *
 * This is not a terminal or circuit boundary.  A finite box generally has
 * div(J) != 0, so it polarizes its two ends and behaves as an antenna.  It
 * cannot enforce the same current through a separate conductor or plasma.
 *
 * Configured via the public `warpx.current_injection.*` input block:
 * n_pairs antenna regions, each with a box, area A, sign (+/-1), and an
 * optional per-pair waveform file (pair_N.file) that overrides the global
 * current_injection.file. The sign reverses the current vector; it does not
 * identify a circuit return terminal. Each region has its own current direction
 * `dir`. Requires algo.particle_shape >= 1 and a charge-conserving current
 * deposition scheme (Esirkepov or Villasenor).
 *
 * Dimensionality:
 *  - 3D and 2D (XZ): J = sign * I(t) / A is uniform over the drive box; the
 *    two pair members have weights +/-dV/(2*vel_coeff*A), and the requested
 *    sign is applied to their opposite velocities.
 *  - RZ: dir = 0/1/2 -> Jr/Jtheta/Jz. Antenna particles are seeded at theta = 0
 *    so the Cartesian push is a radial/azimuthal/axial step. The RZ current
 *    path deposits a raw J that ApplyInverseVolumeScalingToCurrentDensity later
 *    divides by 2*pi*r, so the per-cell weight is set to give the physical
 *    profile: dir = 0 (Jr) uses a uniform weight ->
 *    Jr = I(t)/(2*pi*r*Nz*dz), the 1/r coaxial profile with total radial
 *    current I conserved across shells and distributed across the axial extent
 *    of the box (A is unused/optional); dir = 1,2 (Jtheta/Jz) use a weight ~ r
 *    -> uniform J = sign * I(t) / A over the r-z box (A is the physical
 *    cross-section, e.g. pi*(r_o^2-r_i^2) for Jz, (r_o-r_i)*dz for Jtheta).
 */
class PrescribedCurrentParticleContainer : public WarpXParticleContainer {
  public:
    PrescribedCurrentParticleContainer (amrex::AmrCore* amr_core, int ispecies);
    ~PrescribedCurrentParticleContainer () override = default;

    PrescribedCurrentParticleContainer (
        PrescribedCurrentParticleContainer const&) = delete;
    PrescribedCurrentParticleContainer&
    operator=(PrescribedCurrentParticleContainer const&) = delete;
    PrescribedCurrentParticleContainer (PrescribedCurrentParticleContainer&&) =
        default;
    PrescribedCurrentParticleContainer&
    operator=(PrescribedCurrentParticleContainer&&) = default;

    void InitData () final;

    void
    DefaultInitializeRuntimeAttributes (
        ContainerLike<amrex::PolymorphicArenaAllocator>::ParticleTileType&
        /*pinned_tile*/,
        int /*n_external_attr_real*/, int /*n_external_attr_int*/) final {}

    void
    ReadHeader (std::istream& /*is*/) final {}
    void
    WriteHeader (std::ostream& /*os*/) const final {}

    void Evolve (ablastr::fields::MultiFabRegister& fields, int lev,
                 const std::string& current_fp_string, amrex::Real t,
                 amrex::Real dt,
                 SubcyclingHalf subcycling_half = SubcyclingHalf::None,
                 bool skip_deposition = false,
                 PositionPushType position_push_type = PositionPushType::Full,
                 MomentumPushType momentum_push_type = MomentumPushType::Full,
                 ImplicitOptions const* implicit_options = nullptr) final;

    void
    PushP (int /*lev*/, amrex::Real /*dt*/, const amrex::MultiFab&,
           const amrex::MultiFab&, const amrex::MultiFab&,
           const amrex::MultiFab&, const amrex::MultiFab&,
           const amrex::MultiFab&,
           MomentumPushType /*momentum_push_type*/ =
               MomentumPushType::Full) final {}

    void PostRestart () final;

    /** Whether the impressed-current antenna is enabled in the inputs. */
    static bool is_enabled ();

  private:
    //! One axis-aligned drive box with its own waveform.
    struct Face {
        amrex::Real lo[3] = {0., 0., 0.};
        amrex::Real hi[3] = {0., 0., 0.};
        amrex::Real A = 1.; //!< Face area perpendicular to dir [m^2]
        int dir = 0;        //!< Current component: 0=Jx, 1=Jy, 2=Jz
        int sign = 1;       //!< Current sign: +1 or -1 (J = sign*I(t)/A)
        //! Per-region waveform (from pair_N.file if given, else the global
        //! file).
        warpx::utils::CurrentWaveform waveform;
    };

    std::vector<Face> m_faces;

    amrex::Real m_vel_coeff = 0.; //!< C such that speed = C * I(t) [m/s/A]
    bool m_enabled = true;

    //! Set the velocity scale from the largest waveform sample.
    void compute_velocity_scale ();
};

#endif // WARPX_PrescribedCurrentParticleContainer_H_
