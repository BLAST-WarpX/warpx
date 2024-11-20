/* Copyright 2019 Andrew Myers, David Grote, Luca Fedeli
 * Maxence Thevenet, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PhotonParticleContainer_H_
#define WARPX_PhotonParticleContainer_H_

#include "Evolve/WarpXDtType.H"
#include "Evolve/WarpXPushType.H"
#include "Particles/Gather/ScaleFields.H"
#include "PhysicalParticleContainer.H"

#include "Particles/WarpXParticleContainer_fwd.H"

#include <AMReX_Particles.H>
#include <AMReX_REAL.H>

#include <AMReX_BaseFwd.H>
#include <AMReX_AmrCoreFwd.H>

#include <string>

/**
 * Photon particles have no mass, they deposit no charge, and see specific QED
 * effects. For these reasons, they are stored in the separate particle
 * container PhotonParticleContainer, that inherits from
 * PhysicalParticleContainer. The particle pusher and current deposition, in
 * particular, are overriden in this container.
 */
class PhotonParticleContainer
    : public PhysicalParticleContainer
{
public:
    PhotonParticleContainer (amrex::AmrCore* amr_core,
                                    int ispecies,
                                    const std::string& name);
    ~PhotonParticleContainer () override = default;

    PhotonParticleContainer ( PhotonParticleContainer const &)             = delete;
    PhotonParticleContainer& operator= ( PhotonParticleContainer const & ) = delete;
    PhotonParticleContainer ( PhotonParticleContainer&& )                  = default;
    PhotonParticleContainer& operator= ( PhotonParticleContainer&& )       = default;

    void InitData() override;

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
                        DtType a_dt_type) override;

    // Do nothing
    void PushP (int /*lev*/,
                        amrex::Real /*dt*/,
                        const amrex::MultiFab& /*Ex*/,
                        const amrex::MultiFab& /*Ey*/,
                        const amrex::MultiFab& /*Ez*/,
                        const amrex::MultiFab& /*Bx*/,
                        const amrex::MultiFab& /*By*/,
                        const amrex::MultiFab& /*Bz*/) override {}


    // DepositCharge should do nothing for photons
    void DepositCharge (WarpXParIter& /*pti*/,
                                RealVector const & /*wp*/,
                                const int * const /*ion_lev*/,
                                amrex::MultiFab* /*rho*/,
                                int /*icomp*/,
                                const long /*offset*/,
                                const long /*np_to_deposit*/,
                                int /*thread_num*/,
                                int /*lev*/,
                                int /*depos_lev*/) override {}

    // DepositCurrent should do nothing for photons
    void DepositCurrent (WarpXParIter& /*pti*/,
                                 RealVector const & /*wp*/,
                                 RealVector const & /*uxp*/,
                                 RealVector const & /*uyp*/,
                                 RealVector const & /*uzp*/,
                                 int const * const /*ion_lev*/,
                                 amrex::MultiFab * const /*jx*/,
                                 amrex::MultiFab * const /*jy*/,
                                 amrex::MultiFab * const /*jz*/,
                                 long const /*offset*/,
                                 long const /*np_to_deposit*/,
                                 int const /*thread_num*/,
                                 int const /*lev*/,
                                 int const /*depos_lev*/,
                                 amrex::Real const /*dt*/,
                                 amrex::Real const /*relative_time*/,
                                 PushType /*push_type*/) override {}
};

#endif // #ifndef WARPX_PhotonParticleContainer_H_
