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

    /* Find the particles and count the particles that are in each bin.
       Note that this does *not* rearrange particle arrays */
    amrex::DenseBins<ParticleTileDataType>
    findParticlesInEachBin (amrex::Geometry const& geom_lev,
                            amrex::MFIter const & mfi,
                            ParticleTileType & ptile,
                            BinSizeRatio const& bin_ratio) {

        // Extract particle structures for this tile
        int const np = ptile.numParticles();
        auto ptd = ptile.getParticleTileData();

        // Extract box properties
        Box const& cbx = mfi.tilebox(IntVect::TheZeroVector()); //Cell-centered box
        Box const bin_box = bin_ratio.binBox(cbx);
        IntVect const bin_lo = bin_box.smallEnd();
        IntVect const domain_lo = geom_lev.Domain().smallEnd();
        IntVect const crse = bin_ratio.coarsen;
        IntVect const ref = bin_ratio.refine;
        const auto dxi = geom_lev.InvCellSizeArray();
        const auto plo = geom_lev.ProbLoArray();

        // Find particles that are in each bin;
        // results are stored in the object `bins`.
        ParticleBins bins;
        bins.build(np, ptd, bin_box,
            // Pass lambda function that returns the bin index, relative to the lower corner
            // of `bin_box`. Particles outside of `bin_box` are put in the closest bin by `build`.
            [=] AMREX_GPU_DEVICE (ParticleType const & p) noexcept -> amrex::IntVect
            {
                IntVect iv;
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                    // Index of the particle on the grid refined by `ref`, in the global index space
                    int const i_fine = static_cast<int>(
                        amrex::Math::floor((p.pos(idim)-plo[idim])*dxi[idim]*ref[idim]))
                        + domain_lo[idim]*ref[idim];
                    iv[idim] = amrex::coarsen(i_fine, crse[idim]) - bin_lo[idim];
                }
                return iv;
            });

        return bins;
    }

    /* Find the particles and count the particles that are in each cell.
       Note that this does *not* rearrange particle arrays */
    amrex::DenseBins<ParticleTileDataType>
    findParticlesInEachCell (amrex::Geometry const& geom_lev,
                             amrex::MFIter const & mfi,
                             ParticleTileType & ptile) {
        return findParticlesInEachBin(geom_lev, mfi, ptile, BinSizeRatio{});
    }

} // namespace ParticleUtils
