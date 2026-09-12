/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLE_BATCH_INJECTION_H_
#define WARPX_PARTICLE_BATCH_INJECTION_H_

#include "WarpXParticleContainer_fwd.H"

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

namespace warpx::particles
{
    /** Append host-supplied particles to a level-zero staging grid locally.
     *
     * No MPI collective or EB scraping is performed. The caller must arrange
     * one collective Redistribute after all ranks finish their local batches.
     * The pinned staging allocation is released before returning. Choosing
     * the local destination grid avoids staging every new packet on grid zero.
     */
    void AppendParticleBatch (
        WarpXParticleContainer& particles, int staging_grid, long n,
        amrex::Vector<amrex::ParticleReal> const& x,
        amrex::Vector<amrex::ParticleReal> const& y,
        amrex::Vector<amrex::ParticleReal> const& z,
        amrex::Vector<amrex::ParticleReal> const& ux,
        amrex::Vector<amrex::ParticleReal> const& uy,
        amrex::Vector<amrex::ParticleReal> const& uz,
        int nattr_real,
        amrex::Vector<amrex::Vector<amrex::ParticleReal>> const& attr_real,
        int nattr_int, amrex::Vector<amrex::Vector<int>> const& attr_int,
        int uniqueparticles, amrex::Long id);
}

#endif // WARPX_PARTICLE_BATCH_INJECTION_H_
