/* Copyright 2019-2020 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_HANKEL_TRANSFORM_H_
#define WARPX_HANKEL_TRANSFORM_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuContainers.H>

#ifdef AMREX_USE_GPU
#   include <blas.hh>
#endif

#include <memory>

/* \brief This defines the class that performs the Hankel transform.
 * Original authors: Remi Lehe, Manuel Kirchen
 *
 *
 * Definition of the Hankel forward and backward transform of order p:
 * g(kr) = \int_0^\infty f(r) J_p(kr r) r dr
 * f(r ) = \int_0^\infty g(kr) J_p(kr r) kr dkr
*/
class HankelTransform
{
    public:

        using RealVector = amrex::Gpu::DeviceVector<amrex::Real>;

        // Constructor
        HankelTransform(int hankel_order,
                        int azimuthal_mode,
                        int nr,
                        amrex::Real rmax);

        const RealVector & getSpectralWavenumbers() {return m_kr;}

        void HankelForwardTransform(amrex::FArrayBox const& F, int F_icomp,
                                    amrex::FArrayBox      & G, int G_icomp);

        void HankelInverseTransform(amrex::FArrayBox const& G, int G_icomp,
                                    amrex::FArrayBox      & F, int F_icomp);

    private:
        // Even though nk == nr always, use a separate variable for clarity.
        int m_nr, m_nk;

        RealVector m_kr;

        RealVector m_invM;
        RealVector m_M;

#ifdef AMREX_USE_GPU
        std::unique_ptr<blas::Queue> m_queue;
#endif
};

#endif
