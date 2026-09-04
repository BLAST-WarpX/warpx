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

/** An impressed-current antenna driven by external waveform files.
 *
 * Each selected cell contains a coincident pair of opposite charge weights.
 * The pair members move in opposite directions, so their currents add while
 * their separation deposits the polarization charge required by discrete
 * continuity.  Deposition uses the standard WarpX particle path for both
 * explicit and implicit Maxwell solvers.
 *
 * A finite region is a local antenna, not a circuit terminal: it can polarize
 * at its ends but cannot enforce through-current in a separate conductor.
 * Configure independent regions with `warpx.current_injection.*`.  Esirkepov or
 * Villasenor current deposition and a particle shape of at least one are
 * required.
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
    //! One axis-aligned antenna region with its own waveform.
    struct Drive {
        amrex::Real lo[3] = {0., 0., 0.};
        amrex::Real hi[3] = {0., 0., 0.};
        amrex::Real A = 1.; //!< Cross-sectional area normal to dir [m^2]
        int dir = 0;        //!< Current component: 0=Jx, 1=Jy, 2=Jz
        int sign = 1;       //!< Current sign: +1 or -1 (J = sign*I(t)/A)
        //! Per-region waveform, falling back to the global waveform file.
        warpx::utils::CurrentWaveform waveform;
    };

    std::vector<Drive> m_drives;

    amrex::Real m_vel_coeff = 0.; //!< C such that speed = C * I(t) [m/s/A]
    bool m_enabled = true;

    //! Set the velocity scale from the largest waveform sample.
    void compute_velocity_scale ();
};

#endif // WARPX_PrescribedCurrentParticleContainer_H_
