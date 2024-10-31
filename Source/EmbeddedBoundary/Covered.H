/* Copyright 2024 S. Eric Clark (Helion Energy)
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_EB_COVERED_H_
#define WARPX_EB_COVERED_H_


#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"

#  include <AMReX.H>
#  include <AMReX_Array.H>
#  include <AMReX_Array4.H>
#  include <AMReX_Config.H>
#  include <AMReX_GpuControl.H>
#  include <AMReX_GpuDevice.H>
#  include <AMReX_GpuQualifiers.H>
#  include <AMReX_MFIter.H>
#  include <AMReX_MultiFab.H>
#  include <AMReX_REAL.H>
#  include <AMReX_SPACE.H>

namespace EB {

AMREX_ENUM(CoverTopology,
    none,
    face,
    edge
);

class Covered
{
public:
    amrex::Array4<amrex::Real> lx, ly, lz, Sx, Sy, Sz;

#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    amrex::Dim3 lx_lo, lx_hi, lz_lo, lz_hi;
#endif

    Covered (amrex::MFIter &mfi, int lev);

    // Can have topology of f or e for 'face' or 'edge'
    [[nodiscard]] AMREX_GPU_HOST_DEVICE
    bool isCovered (
        [[maybe_unused]] const int idir,
        [[maybe_unused]] const CoverTopology topology,
        [[maybe_unused]] const int i,
        [[maybe_unused]] const int j,
        [[maybe_unused]] const int k) const noexcept
    {
#ifdef AMREX_USE_EB
        bool is_covered = false;

        switch (idir)
        {
        case 0:
#ifdef WARPX_DIM_3D
            if((topology==CoverTopology::edge and lx(i, j, k)<=0) or (topology==CoverTopology::face and Sx(i, j, k)<=0))
            {
                is_covered = true;
            }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            //In XZ and RZ Ex is associated with a x-edge, while Bx is associated with a z-edge
            if((topology==CoverTopology::edge and lx(i, j, k)<=0) or (topology==CoverTopology::face and lz(i, j, k)<=0))
            {
                is_covered = true;
            }
#endif
            break;

        case 1:
#ifdef WARPX_DIM_3D
            if((topology==CoverTopology::edge and ly(i, j, k)<=0) or (topology==CoverTopology::face and Sy(i, j, k)<=0))
            {
                is_covered = true;
            }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            //In XZ and RZ Ey is associated with a mesh node, so we need to check if  the mesh node is covered
            if( (topology==CoverTopology::edge and (lx(std::min(i  , lx_hi.x), std::min(j  , lx_hi.y), k)<=0
                            || lx(std::max(i-1, lx_lo.x), std::min(j  , lx_hi.y), k)<=0
                            || lz(std::min(i  , lz_hi.x), std::min(j  , lz_hi.y), k)<=0
                            || lz(std::min(i  , lz_hi.x), std::max(j-1, lz_lo.y), k)<=0)) or
                (topology==CoverTopology::face and Sy(i,j,k)<=0))
            {
                is_covered = true;
            }
#endif
            break;

        case 2:
#ifdef WARPX_DIM_3D
            if((topology==CoverTopology::edge and lz(i, j, k)<=0) or (topology==CoverTopology::face and Sz(i, j, k)<=0))
            {
                is_covered = true;
            }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            //In XZ and RZ Ez is associated with a z-edge, while Bz is associated with a x-edge
            if((topology==CoverTopology::edge and lz(i, j, k)<=0) or (topology==CoverTopology::face and lx(i, j, k)<=0))
            {
                is_covered = true;
            }
#endif
            break;
        }
        return is_covered;
#else
        return false;
#endif //AMREX_USE_EB
    }
};

} // namespace EB
#endif
