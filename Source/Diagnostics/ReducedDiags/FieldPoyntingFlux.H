/* Copyright 2019-2020
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_FIELDPOYTINGFLUX_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_FIELDPOYTINGFLUX_H_

#include "ReducedDiags.H"

#include <AMReX_Array4.H>
#include <AMReX_GpuControl.H>
#include <AMReX_Box.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>

#include <string>

using namespace amrex::literals;

struct PoyntingStaggered {
    // This requires the arrays to have Yee centering
    // The procedure (in 3D) is to interpolate the B (which is centered on faces normal
    // to the plane of the calculaton) to the edge in the plane of the calculation to match
    // the location of the E. Then the product E*B is interpolated to the face center in
    // the plane of the calculation.
    // The procedure is mapped to lower dimensions, and in some cases the second interpolation
    // is not needed.

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real EyBx(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ey_arr,
                            amrex::Array4<const amrex::Real> const & Bx_arr) {
#if defined(WARPX_DIM_3D)
        amrex::Real EyBx_dn = Ey_arr(i,j,k,comp)*0.5_rt*(Bx_arr(i,j,k-1,comp) + Bx_arr(i,j,k,comp));
        amrex::Real EyBx_up = Ey_arr(i+1,j,k,comp)*0.5_rt*(Bx_arr(i+1,j,k-1,comp) + Bx_arr(i+1,j,k,comp));
        return 0.5_rt*(EyBx_dn + EyBx_up);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        amrex::Real EyBx_dn = Ey_arr(i,j,k,comp)*0.5_rt*(Bx_arr(i,j-1,k,comp) + Bx_arr(i,j,k,comp));
        amrex::Real EyBx_up = Ey_arr(i+1,j,k,comp)*0.5_rt*(Bx_arr(i+1,j-1,k,comp) + Bx_arr(i+1,j,k,comp));
        return 0.5_rt*(EyBx_dn + EyBx_up);
#elif defined(WARPX_DIM_1D_Z)
        return Ey_arr(i,j,k,comp)*0.5_rt*(Bx_arr(i-1,j,k,comp) + Bx_arr(i,j,k,comp));
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        amrex::ignore_unused(i, j, k, comp, Ey_arr, Bx_arr);
        return 0._rt;
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real ExBy(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ex_arr,
                            amrex::Array4<const amrex::Real> const & By_arr) {
#if defined(WARPX_DIM_3D)
        amrex::Real ExBy_dn = Ex_arr(i,j,k,comp)*0.5_rt*(By_arr(i,j,k-1,comp) + By_arr(i,j,k,comp));
        amrex::Real ExBy_up = Ex_arr(i,j+1,k,comp)*0.5_rt*(By_arr(i,j+1,k-1,comp) + By_arr(i,j+1,k,comp));
        return 0.5_rt*(ExBy_dn + ExBy_up);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_1D_Z)
        return Ex_arr(i,j,k,comp)*0.5_rt*(By_arr(i-1,j,k,comp) + By_arr(i,j,k,comp));
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        amrex::ignore_unused(i, j, k, comp, Ex_arr, By_arr);
        return 0._rt;
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real EzBx(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ez_arr,
                            amrex::Array4<const amrex::Real> const & Bx_arr) {
#if defined(WARPX_DIM_3D)
        amrex::Real EzBx_dn = Ez_arr(i,j,k,comp)*0.5_rt*(Bx_arr(i,j-1,k,comp) + Bx_arr(i,j,k,comp));
        amrex::Real EzBx_up = Ez_arr(i+1,j,k,comp)*0.5_rt*(Bx_arr(i+1,j-1,k,comp) + Bx_arr(i+1,j,k,comp));
        return 0.5_rt*(EzBx_dn + EzBx_up);
#else
        amrex::ignore_unused(i, j, k, comp, Ez_arr, Bx_arr);
        return 0._rt;
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real ExBz(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ex_arr,
                            amrex::Array4<const amrex::Real> const & Bz_arr) {
#if defined(WARPX_DIM_3D)
        amrex::Real ExBz_dn = Ex_arr(i,j,k,comp)*0.5_rt*(Bz_arr(i,j-1,k,comp) + Bz_arr(i,j,k,comp));
        amrex::Real ExBz_up = Ex_arr(i,j,k+1,comp)*0.5_rt*(Bz_arr(i,j-1,k+1,comp) + Bz_arr(i,j,k+1,comp));
        return 0.5_rt*(ExBz_dn + ExBz_up);
#else
        amrex::ignore_unused(i, j, k, comp, Ex_arr, Bz_arr);
        return 0._rt;
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real EyBz(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ey_arr,
                            amrex::Array4<const amrex::Real> const & Bz_arr) {
#if defined(WARPX_DIM_3D)
        amrex::Real EyBz_dn = Ey_arr(i,j,k,comp)*0.5_rt*(Bz_arr(i-1,j,k,comp) + Bz_arr(i,j,k,comp));
        amrex::Real EyBz_up = Ey_arr(i,j,k+1,comp)*0.5_rt*(Bz_arr(i-1,j,k+1,comp) + Bz_arr(i,j,k+1,comp));
        return 0.5_rt*(EyBz_dn + EyBz_up);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        amrex::Real EyBz_dn = Ey_arr(i,j,k,comp)*0.5_rt*(Bz_arr(i-1,j,k,comp) + Bz_arr(i,j,k,comp));
        amrex::Real EyBz_up = Ey_arr(i,j+1,k,comp)*0.5_rt*(Bz_arr(i-1,j+1,k,comp) + Bz_arr(i,j+1,k,comp));
        return 0.5_rt*(EyBz_dn + EyBz_up);
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        return Ey_arr(i,j,k,comp)*0.5_rt*(Bz_arr(i-1,j,k,comp) + Bz_arr(i,j,k,comp));
#elif defined(WARPX_DIM_1D_Z)
        amrex::ignore_unused(i, j, k, comp, Ey_arr, Bz_arr);
        return 0._rt;
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real EzBy(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ez_arr,
                            amrex::Array4<const amrex::Real> const & By_arr) {
#if defined(WARPX_DIM_3D)
        amrex::Real EzBy_dn = Ez_arr(i,j,k,comp)*0.5_rt*(By_arr(i-1,j,k,comp) + By_arr(i,j,k,comp));
        amrex::Real EzBy_up = Ez_arr(i,j+1,k,comp)*0.5_rt*(By_arr(i-1,j+1,k,comp) + By_arr(i,j+1,k,comp));
        return 0.5_rt*(EzBy_dn + EzBy_up);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        return Ez_arr(i,j,k,comp)*0.5_rt*(By_arr(i-1,j,k,comp) + By_arr(i,j,k,comp));
#elif defined(WARPX_DIM_1D_Z)
        amrex::ignore_unused(i, j, k, comp, Ez_arr, By_arr);
        return 0._rt;
#endif
    }

};

struct PoyntingCellCentered {

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real EyBx(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ey_arr,
                            amrex::Array4<const amrex::Real> const & Bx_arr) {
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_1D_Z)
        return 0.5_rt*(Ey_arr(i,j,k-1,comp)*Bx_arr(i,j,k-1,comp)
                     + Ey_arr(i,j,k,comp)*Bx_arr(i,j,k,comp));
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        amrex::ignore_unused(i, j, k, comp, Ey_arr, Bx_arr);
        return 0._rt;
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real ExBy(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ex_arr,
                            amrex::Array4<const amrex::Real> const & By_arr) {
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_1D_Z)
        return 0.5_rt*(Ex_arr(i,j,k-1,comp)*By_arr(i,j,k-1,comp)
                     + Ex_arr(i,j,k,comp)*By_arr(i,j,k,comp));
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        amrex::ignore_unused(i, j, k, comp, Ex_arr, By_arr);
        return 0._rt;
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real EzBx(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ez_arr,
                            amrex::Array4<const amrex::Real> const & Bx_arr) {
#if defined(WARPX_DIM_3D)
        return 0.5_rt*(Ez_arr(i,j-1,k,comp)*Bx_arr(i,j-1,k,comp)
                     + Ez_arr(i,j,k,comp)*Bx_arr(i,j,k,comp));
#else
        amrex::ignore_unused(i, j, k, comp, Ez_arr, Bx_arr);
        return 0._rt;
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real ExBz(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ex_arr,
                            amrex::Array4<const amrex::Real> const & Bz_arr) {
#if defined(WARPX_DIM_3D)
        return 0.5_rt*(Ex_arr(i,j-1,k,comp)*Bz_arr(i,j-1,k,comp)
                     + Ex_arr(i,j,k,comp)*Bz_arr(i,j,k,comp));
#else
        amrex::ignore_unused(i, j, k, comp, Ex_arr, Bz_arr);
        return 0._rt;
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real EyBz(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ey_arr,
                            amrex::Array4<const amrex::Real> const & Bz_arr) {
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        return 0.5_rt*(Ey_arr(i-1,j,k,comp)*Bz_arr(i-1,j,k,comp)
                     + Ey_arr(i,j,k,comp)*Bz_arr(i,j,k,comp));
#elif defined(WARPX_DIM_1D_Z)
        amrex::ignore_unused(i, j, k, comp, Ey_arr, Bz_arr);
        return 0._rt;
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real EzBy(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ez_arr,
                            amrex::Array4<const amrex::Real> const & By_arr) {
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        return 0.5_rt*(Ez_arr(i-1,j,k,comp)*By_arr(i-1,j,k,comp)
                     + Ez_arr(i,j,k,comp)*By_arr(i,j,k,comp));
#elif defined(WARPX_DIM_1D_Z)
        amrex::ignore_unused(i, j, k, comp, Ez_arr, By_arr);
        return 0._rt;
#endif
    }

};

struct PoyntingNodal {

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real EyBx(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ey_arr,
                            amrex::Array4<const amrex::Real> const & Bx_arr) {
#if defined(WARPX_DIM_3D)
        return 0.25_rt*(Ey_arr(i,j,k,comp)*Bx_arr(i,j,k,comp)
                      + Ey_arr(i+1,j,k,comp)*Bx_arr(i+1,j,k,comp)
                      + Ey_arr(i,j+1,k,comp)*Bx_arr(i,j+1,k,comp)
                      + Ey_arr(i+1,j+1,k,comp)*Bx_arr(i+1,j+1,k,comp));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        return 0.5_rt*(Ey_arr(i,j,k,comp)*Bx_arr(i,j,k,comp) +
                       Ey_arr(i+1,j,k,comp)*Bx_arr(i+1,j,k,comp));
#elif defined(WARPX_DIM_1D_Z)
        return Ey_arr(i,j,k,comp)*Bx_arr(i,j,k,comp);
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        amrex::ignore_unused(i, j, k, comp, Ey_arr, Bx_arr);
        return 0._rt;
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real ExBy(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ex_arr,
                            amrex::Array4<const amrex::Real> const & By_arr) {
#if defined(WARPX_DIM_3D)
        return 0.25_rt*(Ex_arr(i,j,k,comp)*By_arr(i,j,k,comp)
                      + Ex_arr(i+1,j,k,comp)*By_arr(i+1,j,k,comp)
                      + Ex_arr(i,j+1,k,comp)*By_arr(i,j+1,k,comp)
                      + Ex_arr(i+1,j+1,k,comp)*By_arr(i+1,j+1,k,comp));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        return 0.5_rt*(Ex_arr(i,j,k,comp)*By_arr(i,j,k,comp)
                     + Ex_arr(i+1,j,k,comp)*By_arr(i+1,j,k,comp));
#elif defined(WARPX_DIM_1D_Z)
        return Ex_arr(i,j,k,comp)*By_arr(i,j,k,comp);
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        amrex::ignore_unused(i, j, k, comp, Ex_arr, By_arr);
        return 0._rt;
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real EzBx(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ez_arr,
                            amrex::Array4<const amrex::Real> const & Bx_arr) {
#if defined(WARPX_DIM_3D)
        return 0.25_rt*(Ez_arr(i,j,k,comp)*Bx_arr(i,j,k,comp)
                      + Ez_arr(i+1,j,k,comp)*Bx_arr(i+1,j,k,comp)
                      + Ez_arr(i,j,k+1,comp)*Bx_arr(i,j,k+1,comp)
                      + Ez_arr(i+1,j,k+1,comp)*Bx_arr(i+1,j,k+1,comp));
#else
        amrex::ignore_unused(i, j, k, comp, Ez_arr, Bx_arr);
        return 0._rt;
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real ExBz(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ex_arr,
                            amrex::Array4<const amrex::Real> const & Bz_arr) {
#if defined(WARPX_DIM_3D)
        return 0.25_rt*(Ex_arr(i,j,k,comp)*Bz_arr(i,j,k,comp)
                      + Ex_arr(i+1,j,k,comp)*Bz_arr(i+1,j,k,comp)
                      + Ex_arr(i,j,k+1,comp)*Bz_arr(i,j,k+1,comp)
                      + Ex_arr(i+1,j,k+1,comp)*Bz_arr(i+1,j,k+1,comp));
#else
        amrex::ignore_unused(i, j, k, comp, Ex_arr, Bz_arr);
        return 0._rt;
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real EyBz(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ey_arr,
                            amrex::Array4<const amrex::Real> const & Bz_arr) {
#if defined(WARPX_DIM_3D)
        return 0.25_rt*(Ey_arr(i,j,k,comp)*Bz_arr(i,j,k,comp)
                      + Ey_arr(i,j+1,k,comp)*Bz_arr(i,j+1,k,comp)
                      + Ey_arr(i,j,k+1,comp)*Bz_arr(i,j,k+1,comp)
                      + Ey_arr(i,j+1,k+1,comp)*Bz_arr(i,j+1,k+1,comp));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        return 0.5_rt*(Ey_arr(i,j,k,comp)*Bz_arr(i,j,k,comp)
                     + Ey_arr(i,j+1,k,comp)*Bz_arr(i,j+1,k,comp));
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        return Ey_arr(i,j,k,comp)*Bz_arr(i,j,k,comp);
#elif defined(WARPX_DIM_1D_Z)
        amrex::ignore_unused(i, j, k, comp, Ey_arr, Bz_arr);
        return 0._rt;
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    static amrex::Real EzBy(int i, int j, int k, int comp,
                            amrex::Array4<const amrex::Real> const & Ez_arr,
                            amrex::Array4<const amrex::Real> const & By_arr) {
#if defined(WARPX_DIM_3D)
        return 0.25_rt*(Ez_arr(i,j,k,comp)*By_arr(i,j,k,comp)
                      + Ez_arr(i,j+1,k,comp)*By_arr(i,j+1,k,comp)
                      + Ez_arr(i,j,k+1,comp)*By_arr(i,j,k+1,comp)
                      + Ez_arr(i,j+1,k+1,comp)*By_arr(i,j+1,k+1,comp));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        return 0.5_rt*(Ez_arr(i,j,k,comp)*By_arr(i,j,k,comp)
                     + Ez_arr(i,j+1,k,comp)*By_arr(i,j+1,k,comp));
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        return Ez_arr(i,j,k,comp)*By_arr(i,j,k,comp);
#elif defined(WARPX_DIM_1D_Z)
        amrex::ignore_unused(i, j, k, comp, Ez_arr, By_arr);
        return 0._rt;
#endif
    }

};

namespace Poynting {

    template<int normal_dir, typename T_Algo, typename AreaFunc>
    amrex::Real Kernel (amrex::Box const & box,
                        amrex::Array4<const amrex::Real> const & Ex_arr,
                        amrex::Array4<const amrex::Real> const & Ey_arr,
                        amrex::Array4<const amrex::Real> const & Ez_arr,
                        amrex::Array4<const amrex::Real> const & Bx_arr,
                        amrex::Array4<const amrex::Real> const & By_arr,
                        amrex::Array4<const amrex::Real> const & Bz_arr,
                        AreaFunc area_factor)
    {

        amrex::ReduceOps<amrex::ReduceOpSum> reduce_ops;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_ops);

        reduce_ops.eval(box, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> amrex::GpuTuple<amrex::Real>
            {

                constexpr int comp = 0;

                // This is needed by the GPU compilers where captured variables cannot first appear
                // in a constexpr-if context
                amrex::ignore_unused(Ez_arr, By_arr, Ex_arr, Bz_arr, Ey_arr, Bx_arr);

                amrex::Real const af = area_factor(i,j,k);
                if constexpr (normal_dir == 0) {
                    return af*(T_Algo::EyBz(i,j,k,comp,Ey_arr,Bz_arr) - T_Algo::EzBy(i,j,k,comp,Ez_arr,By_arr));
                }

                else if constexpr (normal_dir == 1) {
                    return af*(T_Algo::EzBx(i,j,k,comp,Ez_arr,Bx_arr) - T_Algo::ExBz(i,j,k,comp,Ex_arr,Bz_arr));
                }

                else if constexpr (normal_dir == 2) {
                    return af*(T_Algo::ExBy(i,j,k,comp,Ex_arr,By_arr) - T_Algo::EyBx(i,j,k,comp,Ey_arr,Bx_arr));
                }

            });

        auto r = reduce_data.value();
        return amrex::get<0>(r);
    }
}

/**
 * \brief This class mainly contains a function that computes the field Poynting flux,
 *        S = E cross B, integrated over each face of the domain.
 */
class FieldPoyntingFlux : public ReducedDiags
{
public:

    /**
     * \brief Constructor
     *
     * \param[in] rd_name reduced diags names
     */
    FieldPoyntingFlux (const std::string& rd_name);

    /**
     * \brief Call the routine to compute the Poynting flux if needed
     *
     * \param[in] step current time step
     */
    void ComputeDiags (int step) final;

    /**
     * \brief Call the routine to compute the Poynting flux at the mid step time level
     *
     * \param[in] step current time step
     */
    void ComputeDiagsMidStep (int step) final;

    /**
     * \brief This function computes the electromagnetic Poynting flux,
     * obtained by integrating the electromagnetic Poynting flux density g = eps0 * (E x B)
     * on the surface of the domain.
     */
    void ComputePoyntingFlux ();

    void WriteCheckpointData (std::string const & dir) final;

    void ReadCheckpointData (std::string const & dir) final;

private:

    bool use_mid_step_value = false;

};

#endif
