#ifndef WARPX_INTERPOLATE_H_
#define WARPX_INTERPOLATE_H_

#include <AMReX_REAL.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <memory>

namespace Interpolate
{
    using namespace amrex;

    /** \brief Samples/Interpolates the coarse scalar multifab `F_cp` on the
     * fine grid associated with the fine multifab `F_fp`.
     */
    std::unique_ptr<MultiFab>
        getInterpolatedScalar(
            const MultiFab& F_cp, const MultiFab& F_fp,
            const DistributionMapping& dm, amrex::IntVect r_ratio,
            const Real* /*dx*/, IntVect ngrow );

    /** \brief Samples/Interpolates the coarse vector multifab `F*_cp` on the
     * fine grid associated with the fine multifab `F*_fp`.
     */
    std::array<std::unique_ptr<MultiFab>, 3>
        getInterpolatedVector(
            const MultiFab* Fx_cp,
            const MultiFab* Fy_cp,
            const MultiFab* Fz_cp,
            const MultiFab* Fx_fp,
            const MultiFab* Fy_fp,
            const MultiFab* Fz_fp,
            const DistributionMapping& dm, amrex::IntVect r_ratio,
            const Real* dx, IntVect ngrow );

}

#endif // WARPX_INTERPOLATE_H_
