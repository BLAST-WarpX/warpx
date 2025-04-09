/* Copyright 2021 Elisa Rheaume, Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_FieldProbeParticleContainer_H_
#define WARPX_FieldProbeParticleContainer_H_

#include <AMReX_ParIter.H>
#include <AMReX_Particles.H>

#include <AMReX_BaseFwd.H>
#include <AMReX_AmrCoreFwd.H>
#include <AMReX_Vector.H>

/**
 * This enumerated struct is used to index the field probe particle
 * values that are being stored as SoA data. Nattribs
 * is enumerated to give the number of attributes stored.
 */
struct FieldProbePIdx
{
    enum
    {
#if !defined (WARPX_DIM_1D_Z)
        x,
#endif
#if defined (WARPX_DIM_3D)
        y,
#endif
        z,
        Ex, Ey, Ez,
        Bx, By, Bz,
        S, //!< the Poynting vector
#ifdef WARPX_DIM_RZ
        theta,      ///< RZ needs all three position components
#endif
        nattribs
    };
};

/**
 * This class defines the FieldProbeParticleContainer
 * which is branched from the amrex::ParticleContainer.
 * nattribs tells the particle container to allot 7 SOA values.
 */
class FieldProbeParticleContainer
    : public amrex::ParticleContainerPureSoA<FieldProbePIdx::nattribs, 0>
{
public:
    static constexpr int NStructReal = 0;
    static constexpr int NStructInt = 0;
    static constexpr int NReal = FieldProbePIdx::nattribs;
    static constexpr int NInt = 0;

    FieldProbeParticleContainer (amrex::AmrCore* amr_core);
    ~FieldProbeParticleContainer() override = default;

    FieldProbeParticleContainer ( FieldProbeParticleContainer const &)             = delete;
    FieldProbeParticleContainer& operator= ( FieldProbeParticleContainer const & ) = delete;
    FieldProbeParticleContainer ( FieldProbeParticleContainer&& )                  = default;
    FieldProbeParticleContainer& operator= ( FieldProbeParticleContainer&& )       = default;

    //! amrex iterator for our number of attributes
    using iterator = amrex::ParIterSoA<FieldProbePIdx::nattribs, 0>;
    //! amrex iterator for our number of attributes (read-only)
    using const_iterator = amrex::ParConstIterSoA<FieldProbePIdx::nattribs, 0>;

    //! similar to WarpXParticleContainer::AddNParticles but does not include u(x,y,z)
    void AddNParticles (int lev, amrex::Vector<amrex::ParticleReal> const & x, amrex::Vector<amrex::ParticleReal> const & y, amrex::Vector<amrex::ParticleReal> const & z);
};

#endif // WARPX_FieldProbeParticleContainer_H_
