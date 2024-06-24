/* Copyright 2019 Andrew Myers, Axel Huebl, David Grote
 * Luca Fedeli, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_LaserParticleContainer_H_
#define WARPX_LaserParticleContainer_H_

#include "Evolve/WarpXDtType.H"
#include "Evolve/WarpXPushType.H"
#include "Laser/LaserProfiles.H"
#include "WarpXParticleContainer.H"

#include <AMReX_Extension.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>
#include <AMReX_AmrCoreFwd.H>

#include <limits>
#include <memory>
#include <string>

/**
 * The main method to inject a laser pulse in WarpX is to use an artificial
 * antenna: particles evenly distributed in a given plane (one particle per
 * cell) move at each iteration and deposit a current J onto the grid, which
 * in turns creates an electromagnetic field on the grid. The particles'
 * displacements are prescribed to create the field requested by the user.
 *
 * These artificial particles are contained in the LaserParticleContainer.
 * LaserParticleContainer derives directly from WarpXParticleContainer. It
 * requires a DepositCurrent function, but no FieldGather function.
 */
class LaserParticleContainer
    : public WarpXParticleContainer
{
public:
    LaserParticleContainer (amrex::AmrCore* amr_core, const std::string& name);
    ~LaserParticleContainer () override = default;

    LaserParticleContainer ( LaserParticleContainer const &)             = delete;
    LaserParticleContainer& operator= ( LaserParticleContainer const & ) = delete;
    LaserParticleContainer ( LaserParticleContainer&& )                  = default;
    LaserParticleContainer& operator= ( LaserParticleContainer&& )       = default;

    void InitData () final;

    /**
     * \brief Method to initialize runtime attributes. Does nothing for LaserParticleContainer.
     */
    void DefaultInitializeRuntimeAttributes (
        typename ContainerLike<amrex::PinnedArenaAllocator>::ParticleTileType& /*pinned_tile*/,
        int /*n_external_attr_real*/,
        int /*n_external_attr_int*/) final {}

    void ReadHeader (std::istream& is) final;

    void WriteHeader (std::ostream& os) const final;

    void Evolve (ablastr::fields::MultiFabRegister& fields,
                 int lev,
                 const std::string& current_fp_string,
                 amrex::Real t, amrex::Real dt, DtType a_dt_type=DtType::Full,
                 bool skip_deposition=false, PushType push_type=PushType::Explicit) final;

    void PushP (int lev, amrex::Real dt,
                        const amrex::MultiFab& ,
                        const amrex::MultiFab& ,
                        const amrex::MultiFab& ,
                        const amrex::MultiFab& ,
                        const amrex::MultiFab& ,
                        const amrex::MultiFab& ) final;

    void PostRestart () final;

    void calculate_laser_plane_coordinates (const WarpXParIter& pti, int np,
                                            amrex::Real * AMREX_RESTRICT pplane_Xp,
                                            amrex::Real * AMREX_RESTRICT pplane_Yp);

    void update_laser_particle (WarpXParIter& pti, int np, amrex::ParticleReal * AMREX_RESTRICT puxp,
                                amrex::ParticleReal * AMREX_RESTRICT puyp,
                                amrex::ParticleReal * AMREX_RESTRICT puzp,
                                amrex::ParticleReal const * AMREX_RESTRICT pwp,
                                amrex::Real const * AMREX_RESTRICT amplitude,
                                amrex::Real dt, PushType push_type=PushType::Explicit);

protected:

    std::string m_laser_name;

private:
    // runtime paramters
    amrex::Vector<amrex::Real> m_position; //! Coordinates of one of the point of the antenna
    amrex::Vector<amrex::Real> m_nvec; //! Normal of the plane of the antenna
    amrex::Vector<amrex::Real> m_p_X;// ! Polarization

    amrex::Real               m_e_max       = std::numeric_limits<amrex::Real>::quiet_NaN();
    amrex::Real               m_wavelength  = std::numeric_limits<amrex::Real>::quiet_NaN();

    amrex::Real               m_Z0_lab = 0; // Position of the antenna in the lab frame

    long m_min_particles_per_mode = 4;

    // computed using runtime parameters
    amrex::Vector<amrex::Real> m_p_Y;
    amrex::Vector<amrex::Real> m_u_X;
    amrex::Vector<amrex::Real> m_u_Y;
    amrex::Real m_weight   = std::numeric_limits<amrex::Real>::quiet_NaN();
    amrex::Real m_mobility = std::numeric_limits<amrex::Real>::quiet_NaN();


    // laser particle domain
    amrex::RealBox m_laser_injection_box;
    // Theoretical position of the antenna. Used if do_continuous_injection=1.
    // Track the position of the antenna until it enters the simulation domain.
    amrex::Vector<amrex::Real> m_updated_position;

    void ComputeSpacing (int lev, amrex::Real& Sx, amrex::Real& Sy) const;
    void ComputeWeightMobility (amrex::Real Sx, amrex::Real Sy);
    void InitData (int lev);
    // Inject the laser antenna during the simulation, if it started
    // outside of the simulation domain and enters it.
    void ContinuousInjection(const amrex::RealBox& injection_box) override;

    /**
     * \brief  Update antenna position for continuous injection of lasers
     *         in a boosted frame
     */
    void UpdateAntennaPosition (amrex::Real dt) override;

    // Unique (smart) pointer to the laser profile
    std::unique_ptr<WarpXLaserProfiles::ILaserProfile> m_up_laser_profile;

    // Flag to disable the laser (e.g., if e_max is 0)
    bool m_enabled = true;
};

#endif
