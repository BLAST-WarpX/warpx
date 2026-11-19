#ifndef PEC_INSULATOR_K_H_
#define PEC_INSULATOR_K_H_

#include <AMReX_Config.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_SPACE.H>

#include <array>

// Coordinate helpers shared by PEC-insulator field BCs and PC mass-matrix masking,
// so both evaluate the area parsers with the same component staggering.
namespace PEC_InsulatorUtils
{
    // \brief Converts the grid indices to spatial coordinates
    //
    // \param[in] iv      vector of indices
    // \param[in] xyzmin  vector of grid mins
    // \param[in] dx      vector of grid sizes
    // \param[in] lo      vector of grid index mins
    // \param[in] nodal   vector of nodel flags to each dimension
    // \return the coordinates

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::XDim3 ConvertIndexToCoordinate(amrex::IntVect const & iv,
                                          amrex::XDim3 const & xyzmin,
                                          std::array<amrex::Real,3> const & dx,
                                          amrex::IntVect const & lo,
                                          amrex::IntVect const & nodal)
    {
        using namespace amrex::literals;
        amrex::XDim3 result = {0._rt, 0._rt, 0._rt};

#if defined(WARPX_DIM_3D) || defined(WARPX_ZINDEX)
        if constexpr (AMREX_SPACEDIM > 1){
            amrex::Real const shiftx = (nodal[0] ? 0._rt : 0.5_rt);
            result.x = xyzmin.x + (iv[0] - lo[0] + shiftx)*dx[0];
        }
        else{
            result.x = 0._rt;
        }

#endif
        if constexpr (AMREX_SPACEDIM == 3){
            amrex::Real const shifty = nodal[1] ? 0._rt : 0.5_rt;
            result.y = xyzmin.y + (iv[1] - lo[1] + shifty)*dx[1];
        }
        else{
            result.y = 0._rt;
        }

#ifndef WARPX_DIM_1D_Z
#if defined(WARPX_ZINDEX)
        amrex::Real const shiftz = (nodal[WARPX_ZINDEX] ? 0._rt : 0.5_rt);
        result.z = xyzmin.z + (iv[WARPX_ZINDEX] - lo[WARPX_ZINDEX] + shiftz)*dx[2];
#endif
#endif

        return result;
    }

    // Convenient structure to the hold the two transverse coordinates
    struct XDimTransverse { amrex::Real t1; amrex::Real t2; };

    // \brief Returns the two coordinates transverse from the specified dimension
    //
    // \param[im] idim    the dimension number
    // \param[in] coords  the coordinate of each dimension
    // \return the transverse coordinates
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    XDimTransverse GetTransverseCoordinates(int idim, amrex::XDim3 const & coords)
    {
        // Transverse coordinates
        amrex::Real t1 = 0.;
        amrex::Real t2 = 0.;

#ifndef WARPX_DIM_1D_Z
        if (idim == 0) {
            t1 = coords.y;
            t2 = coords.z;
        }
#endif
#if defined(WARPX_DIM_3D)
        if (idim == 1) {
            t1 = coords.x;
            t2 = coords.z;
        }
#endif
#if defined(WARPX_ZINDEX)
        if (idim == WARPX_ZINDEX) {
            t1 = coords.x;
            t2 = coords.y;
        }
#endif
        return XDimTransverse {t1, t2};
    }

}

#endif // PEC_INSULATOR_K_H_
