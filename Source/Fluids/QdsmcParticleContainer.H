/* Copyright 2024 Marco Acciarri (Helion Energy Inc.)
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_QdsmcParticleContainer_H_
#define WARPX_QdsmcParticleContainer_H_

#include <AMReX_ParIter.H>
#include <AMReX_Particles.H>

#include <AMReX_BaseFwd.H>
#include <AMReX_AmrCoreFwd.H>
#include <AMReX_Vector.H>

//#include "Particles/ParticleBoundaries.H"
//#include <ablastr/fields/MultiFabRegister.H>

//#include "QdsmcParticleContainer_fwd.H"

/**
 * This enumerated struct is used to index the qdsmc fictitious
 * particle values that are being stored as SoA data. Nattribs
 * is enumerated to give the number of attributes stored.
 */
struct QdsmcPIdx
{
    enum
    {
#if !defined (WARPX_DIM_1D_Z) // change this, this algorithm will only be used for the dimensions the Hybrid solver is compatible with
        x,
#endif
#if defined (WARPX_DIM_3D)
        y,
#endif
        z,
        x_node, y_node, z_node,
        vx, vy, vz,
        entropy, np_real,
        nattribs
    };
};

/**
 * This class defines the QdsmcParticleContainer
 * which is branched from the amrex::ParticleContainer.
 */
class QdsmcParticleContainer
    : public amrex::ParticleContainerPureSoA<QdsmcPIdx::nattribs, 0>
{
public:
    static constexpr int NStructReal = 0;
    static constexpr int NStructInt = 0;
    static constexpr int NReal = QdsmcPIdx::nattribs;
    static constexpr int NInt = 0;

    QdsmcParticleContainer (amrex::AmrCore* amr_core);
    ~QdsmcParticleContainer() override = default;

    QdsmcParticleContainer ( QdsmcParticleContainer const &)             = delete;
    QdsmcParticleContainer& operator= ( QdsmcParticleContainer const & ) = delete;
    QdsmcParticleContainer ( QdsmcParticleContainer&& )                  = default;
    QdsmcParticleContainer& operator= ( QdsmcParticleContainer&& )       = default;

    //! amrex iterator for our number of attributes
    using iterator = amrex::ParIterSoA<QdsmcPIdx::nattribs, 0>;
    //! amrex iterator for our number of attributes (read-only)
    using const_iterator = amrex::ParConstIterSoA<QdsmcPIdx::nattribs, 0>;

    //! similar to WarpXParticleContainer::AddNParticles

    void AddNParticles (int lev, amrex::Long np,
                        amrex::Vector<amrex::ParticleReal> const & x,
                        amrex::Vector<amrex::ParticleReal> const & y,
                        amrex::Vector<amrex::ParticleReal> const & z);
    /*
    * Function that initializes the particles (only positions)
    * current version only adds one particle per cell (qdsmc for electron energy equation)
    * future PR will extend this implementation in order to use qdsmc for a neutral gas too
    * and will require gaussian quadrature order used in general QDSMC algorithm
    * (nppc = order^dim), or nppc = 1 for electron energy equation used in HybridSolver
    * calls AddNParticles
    */
    void InitParticles (int lev);

    // Function that set the fictitious particles velocities (only drift for now)
    // check if U or NU is going to be used, current version written for U
    // pass vector field instead
    void SetV (int lev,
        const amrex::MultiFab &Ux,
        const amrex::MultiFab &Uy,
        const amrex::MultiFab &Uz);

    void SetK (int lev,
        const amrex::MultiFab &Kfield,
        const amrex::MultiFab &rhofield);

    // Function that push particle positions by one time step
    void PushX (int lev, amrex::Real dt);

    // Change to general function that deposits some fictitious particle
    // property/value valuep in a generic scalar multifab
    // using linear interpolation
    void DepositK(int lev, amrex::MultiFab &Kfield);

};

#endif // WARPX_QdsmcParticleContainer_H_
