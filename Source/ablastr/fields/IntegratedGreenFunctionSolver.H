/* Copyright 2019-2024 Arianna Formenti, Remi Lehe
 *
 * This file is part of ABLASTR.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_IGF_SOLVER_H
#define ABLASTR_IGF_SOLVER_H

#include <ablastr/constant.H>

#include <AMReX_BoxArray.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <array>
#include <cmath>


namespace ablastr::fields
{
    using namespace amrex::literals;


    /** @brief Implements equation 2 in https://doi.org/10.1103/PhysRevSTAB.10.129901
     *         with some modification to symmetrize the function.
     *
     * @param[in] x x-coordinate of given location
     * @param[in] y y-coordinate of given location
     * @param[in] z z-coordinate of given location
     *
     * @return the integrated Green function G in 3D
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real
    IntegratedPotential3D (amrex::Real x, amrex::Real y, amrex::Real z)
    {
        amrex::Real const r = std::sqrt( x*x + y*y + z*z );
        amrex::Real const G = - 0.5_rt * z*z * std::atan( x*y/(z*r) )
                              - 0.5_rt * y*y * std::atan( x*z/(y*r) )
                              - 0.5_rt * x*x * std::atan( y*z/(x*r) )
                              + y*z*std::asinh( x/std::sqrt(y*y + z*z) )
                              + x*z*std::asinh( y/std::sqrt(x*x + z*z) )
                              + x*y*std::asinh( z/std::sqrt(x*x + y*y) );
        return G;
    }


    /** @brief Implements equation 58 in https://doi.org/10.1016/j.jcp.2004.01.008
     *
     * @param[in] x x-coordinate of given location
     * @param[in] y y-coordinate of given location
     *
     * @return the integrated Green function G in 2D
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real
    IntegratedPotential2D (amrex::Real x, amrex::Real y)
    {
        amrex::Real const G = 3_rt*x*y
                              - x*x * std::atan(y/x)
                              - y*y * std::atan(x/y)
                              - x*y * std::log(x*x + y*y);
        return G;
    }


    /** @brief add
     *
     * @param[in] x x-coordinate of given location
     * @param[in] y y-coordinate of given location
     * @param[in] z z-coordinate of given location
     * @param[in] dx cell size along x
     * @param[in] dy cell size along y
     * @param[in] dz cell size along z
     *
     * @return the sum of integrated Green function G in 3D
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real
    SumOfIntegratedPotential3D (amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real dx, amrex::Real dy, amrex::Real dz)
    {
        return 1._rt/(4._rt*ablastr::constant::math::pi*ablastr::constant::SI::ep0) * (
                  IntegratedPotential3D( x+0.5_rt*dx, y+0.5_rt*dy, z+0.5_rt*dz )
                - IntegratedPotential3D( x-0.5_rt*dx, y+0.5_rt*dy, z+0.5_rt*dz )
                - IntegratedPotential3D( x+0.5_rt*dx, y-0.5_rt*dy, z+0.5_rt*dz )
                + IntegratedPotential3D( x-0.5_rt*dx, y-0.5_rt*dy, z+0.5_rt*dz )
                - IntegratedPotential3D( x+0.5_rt*dx, y+0.5_rt*dy, z-0.5_rt*dz )
                + IntegratedPotential3D( x-0.5_rt*dx, y+0.5_rt*dy, z-0.5_rt*dz )
                + IntegratedPotential3D( x+0.5_rt*dx, y-0.5_rt*dy, z-0.5_rt*dz )
                - IntegratedPotential3D( x-0.5_rt*dx, y-0.5_rt*dy, z-0.5_rt*dz )
                );
    }


    /** @brief add
     *
     * @param[in] x x-coordinate of given location
     * @param[in] y y-coordinate of given location
     * @param[in] dx cell size along x
     * @param[in] dy cell size along y
     *
     * @return the sum of integrated Green function G in 2D
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real
    SumOfIntegratedPotential2D (amrex::Real x, amrex::Real y, amrex::Real dx, amrex::Real dy)
    {
        return 1._rt/(4._rt*ablastr::constant::math::pi*ablastr::constant::SI::ep0) * (
                  IntegratedPotential2D( x+0.5_rt*dx, y+0.5_rt*dy )
                - IntegratedPotential2D( x+0.5_rt*dx, y-0.5_rt*dy )
                - IntegratedPotential2D( x-0.5_rt*dx, y+0.5_rt*dy )
                + IntegratedPotential2D( x-0.5_rt*dx, y-0.5_rt*dy )
                );
    }


    /** @brief Compute the electrostatic potential using the Integrated Green Function method
     *         as in http://dx.doi.org/10.1103/PhysRevSTAB.9.044204
     *
     * @param[in] rho the charge density amrex::MultiFab
     * @param[out] phi the electrostatic potential amrex::MultiFab
     * @param[in] cell_size an arreay of 3 reals dx dy dz
     * @param[in] ba amrex::BoxArray with the grid of a given level
     */
    void
    computePhiIGF (amrex::MultiFab const & rho,
                   amrex::MultiFab & phi,
                   std::array<amrex::Real, 3> const & cell_size,
                   amrex::BoxArray const & ba,
                   bool is_igf_2d_slices);

} // namespace ablastr::fields

#endif // ABLASTR_IGF_SOLVER_H
