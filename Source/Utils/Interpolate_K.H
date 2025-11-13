#ifndef WARPX_INTERP_K_H_
#define WARPX_INTERP_K_H_

#include <AMReX_FArrayBox.H>

namespace Interpolate {

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void interp (int j, int k, int l,
             amrex::Array4<amrex::Real      > const& fine,
             amrex::Array4<amrex::Real const> const& crse,
             const amrex::IntVect r_ratio, IntVect const& type) noexcept
{
    using amrex::Real;
    Real const rrx = 1.0_rt/static_cast<Real>(r_ratio[0]);

    int const jg = amrex::coarsen(j,r_ratio[0]);
    Real const wx = static_cast<Real>(type[0]) * static_cast<amrex::Real>(j-jg*r_ratio[0]) * rrx;
    Real const owx = 1.0_rt-wx;

#if (AMREX_SPACEDIM >= 2)
    Real const rry = 1.0_rt/static_cast<Real>(r_ratio[1]);
    int const kg = amrex::coarsen(k,r_ratio[1]);
    Real const wy = static_cast<Real>(type[1]) * static_cast<amrex::Real>(k-kg*r_ratio[1]) * rry;
    Real const owy = 1.0_rt-wy;
#endif

#if AMREX_SPACEDIM == 1
    fine(j,k,l) = owx * crse(jg  ,0,0)
        +          wx * crse(jg+1,0,0);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    fine(j,k,l) = owx * owy * crse(jg  ,kg  ,0)
        +         owx *  wy * crse(jg  ,kg+1,0)
        +          wx * owy * crse(jg+1,kg  ,0)
        +          wx *  wy * crse(jg+1,kg+1,0);
#else
    Real const rrz = 1.0_rt/static_cast<Real>(r_ratio[2]);
    int const lg = amrex::coarsen(l,r_ratio[2]);
    Real const wz = static_cast<Real>(type[2]) * static_cast<amrex::Real>(l-lg*r_ratio[2]) * rrz;
    Real const owz = 1.0_rt-wz;
    fine(j,k,l) = owx * owy * owz * crse(jg  ,kg  ,lg  )
        +          wx * owy * owz * crse(jg+1,kg  ,lg  )
        +         owx *  wy * owz * crse(jg  ,kg+1,lg  )
        +          wx *  wy * owz * crse(jg+1,kg+1,lg  )
        +         owx * owy *  wz * crse(jg  ,kg  ,lg+1)
        +          wx * owy *  wz * crse(jg+1,kg  ,lg+1)
        +         owx *  wy *  wz * crse(jg  ,kg+1,lg+1)
        +          wx *  wy *  wz * crse(jg+1,kg+1,lg+1);
#endif
}

}

#endif
