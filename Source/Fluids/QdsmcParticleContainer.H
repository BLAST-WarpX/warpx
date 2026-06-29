/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Marco Acciarri, Prabhat Kumar (Helion Energy Inc.)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_QDSMCPARTICLECONTAINER_H_
#define WARPX_QDSMCPARTICLECONTAINER_H_

#include "QdsmcParticleContainer_fwd.H"

#include <AMReX_AmrCoreFwd.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_ParIter.H>
#include <AMReX_Particles.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>


/**
 * @brief Indexing for the QDSMC fictitious-particle SoA storage.
 *
 * QDSMC particles are one-per-cell Lagrangian markers used to transport the
 * electron entropy K_e by the electron fluid velocity V_e over one PIC step
 * (see Belyaev et al., Phys. Plasmas 31, 012902 (2024), Sec. III.A).
 *
 * Layout: AMReX's `ParticleContainerPureSoA\<NReal=nattribs, NInt=0\>` treats the
 * first AMREX_SPACEDIM real attributes as the AMReX particle positions used
 * for tile-assignment and `Redistribute()`. The remaining attributes are
 * user data.
 *
 *   Position attributes (AMReX-tracked, dim-dependent slot count):
 *     1D Z : [z]               -> AMReX position 0
 *     XZ/RZ: [x, z]            -> AMReX positions 0, 1
 *     3D   : [x, y, z]         -> AMReX positions 0, 1, 2
 *
 *   User attributes (always present, 3D semantic regardless of field dim):
 *     x_node, y_node, z_node : home position (cell center at start of QDSMC step)
 *     vx, vy, vz             : electron-fluid velocity gathered at home (3 components)
 *     entropy                : weighted entropy K_e * N_e carried by this particle
 *     np_real                : weight N_e (electrons in this cell)
 *
 * The home positions and velocities are always stored as 3 components even when
 * the field dimension is lower. In 2D the y-home is set to 0 at init; in 1D
 * both x-home and y-home are 0. The dim-agnostic kernels in Qdsmc_K.H consume
 * (x, y, z) as a 3-vector and the underlying compute_weights\<NODE\> routes them
 * to the right field axes per dim (e.g. r = sqrt(x^2 + y^2) for RZ).
 */
struct QdsmcPIdx
{
    enum
    {
        // AMReX position slots (the first AMREX_SPACEDIM entries).
#if !defined(WARPX_DIM_1D_Z)
        x,
#endif
#if defined(WARPX_DIM_3D)
        y,
#endif
        z,
        // Home position (3D, used for cell-centered gather at start of step).
        x_node, y_node, z_node,
        // Electron-fluid velocity gathered at home (3 components, always).
        vx, vy, vz,
        // QDSMC transported scalars: weighted entropy and electron count.
        entropy, np_real,
        nattribs
    };
};


/**
 * @brief Container for QDSMC fictitious electron-energy-equation particles.
 *
 * One particle is initialized per cell at its center; the particle's velocity
 * is set to V_e (gathered at home), its entropy slot to K_e * N_e, its weight
 * slot to N_e, and the particle is then pushed by one PIC step. After the
 * push, entropy and weight are scattered back to the grid; the new K_e is
 * recovered as (deposited entropy) / (deposited weight), and the electron
 * temperature is then computed from K_e and the updated ion-derived n_e via
 * the polytropic relation.
 *
 * Inherits from `ParticleContainerPureSoA` so all attributes live in
 * struct-of-arrays storage; positions are the first AMREX_SPACEDIM slots.
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

    using iterator = amrex::ParIterSoA<QdsmcPIdx::nattribs, 0>;
    using const_iterator = amrex::ParConstIterSoA<QdsmcPIdx::nattribs, 0>;

    /**
     * @brief Initialize one particle per cell at the cell center.
     *
     * For dim D < 3 the missing position components of the home are set to 0.
     * Particle velocity, entropy and np_real are all set to 0; subsequent
     * SetV/SetK calls will populate them.
     */
    void InitParticles (int lev);

    /**
     * @brief Gather a 3-component vector field (e.g. V_e) at the particle's
     *        home position into the particle's velocity attributes.
     *
     * The field staggering and dim are inferred from AMREX_SPACEDIM; the
     * underlying kernel uses ablastr::particles::compute_weights\<NODE\>.
     */
    void SetV (int lev,
               const amrex::MultiFab & Ux,
               const amrex::MultiFab & Uy,
               const amrex::MultiFab & Uz);

    /**
     * @brief Gather K_e and n_e at the particle's home position and store as
     *        (entropy = K_e * N_e, np_real = N_e).
     */
    void SetK (int lev,
               const amrex::MultiFab & Kfield,
               const amrex::MultiFab & rhofield);

    /**
     * @brief Advance the particle position by x = x_home + v*dt (forward Euler).
     *        Calls Redistribute() at the end so particles end up in the
     *        correct tile for their new position.
     */
    void PushX (int lev, amrex::Real dt);

    /**
     * @brief Scatter the per-particle entropy (K_e * N_e) onto Kfield via
     *        linear shape-factor weighted atomic add.
     *        Kfield is zeroed at the start of this call.
     */
    void DepositK (int lev, amrex::MultiFab & Kfield);

    /**
     * @brief Scatter the per-particle weight np_real/V_cell (= n_e) onto
     *        Field via linear shape-factor weighted atomic add.
     *        Field is zeroed at the start of this call.
     */
    void DepositField (int lev, amrex::MultiFab & Field);

    /**
     * @brief Reset particle position to home, zero velocity / entropy / weight.
     *        Calls Redistribute() afterward.
     */
    void ResetParticles (int lev);
};

#endif // WARPX_QDSMCPARTICLECONTAINER_H_
