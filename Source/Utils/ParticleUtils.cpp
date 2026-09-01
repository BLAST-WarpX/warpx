/* Copyright 2019-2020 Neil Zaim, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ParticleUtils.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_Dim3.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_REAL.H>
#include <AMReX_SPACE.H>

namespace ParticleUtils
{

    using namespace amrex;

    // Define shortcuts for frequently-used type names
    using ParticleType = typename WarpXParticleContainer::ParticleType;
    using ParticleBins = DenseBins<ParticleTileDataType>;

    /* Find the particles and count the particles that are in each supercell.
       Note that this does *not* rearrange particle arrays */
    amrex::DenseBins<ParticleTileDataType>
    findParticlesInEachSuperCell (amrex::Geometry const& geom_lev,
                  amrex::MFIter const & mfi,
                  ParticleTileType & ptile,
                  const amrex::IntVect& supercell_size) {

        // Extract number of particles for this tile
        int const np = ptile.numParticles();

        // Extract box properties
        Box const& box = mfi.tilebox(IntVect::TheZeroVector()); // Cell-centered box
        const auto dxi = geom_lev.InvCellSizeArray();
        const auto plo = geom_lev.ProbLoArray();
        const auto domain = geom_lev.Domain();

    int ntiles = amrex::numTilesInBox(box, true, supercell_size);

    // Find particles that are in each cell;
        // results are stored in the object `bins`.
        ParticleBins bins;
    bins.build(np, ptile.getParticleTileData(), ntiles,
           amrex::GetParticleBin{plo, dxi, domain, supercell_size, box});
        return bins;
    }

    /* Find the particles and count the particles that are in each cell.
       Note that this does *not* rearrange particle arrays */
    amrex::DenseBins<ParticleTileDataType>
    findParticlesInEachCell (amrex::Geometry const& geom_lev,
                 amrex::MFIter const & mfi,
                 ParticleTileType & ptile) {
    return findParticlesInEachSuperCell(geom_lev, mfi, ptile, amrex::IntVect(AMREX_D_DECL(1, 1, 1)));
    }

} // namespace ParticleUtils
