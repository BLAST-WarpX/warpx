/* Copyright 2020 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SPECTRAL_BINOMIAL_FILTER_H_
#define WARPX_SPECTRAL_BINOMIAL_FILTER_H_

#include "SpectralHankelTransform/HankelTransform.H"
#include "SpectralKSpace.H"

/**
 * \brief Class that sets up binomial filtering in k space
 *
 * Contains the filter arrays for r and z.
 */
class SpectralBinomialFilter
{
    public:

        // `KFilterArray` holds a one 1D array ("DeviceVector") that
        // implements the filter.
        using KFilterArray = amrex::Gpu::DeviceVector<amrex::Real>;

        SpectralBinomialFilter () = default;
        void InitFilterArray (RealKVector const & kvec,
                              amrex::Real dels,
                              int npasses,
                              bool compensation,
                              KFilterArray & filter);
        void InitFilterArray (RealKVector const & kr,
                              RealKVector const & kz,
                              amrex::RealVect dx,
                              amrex::IntVect filter_npass_each_dir,
                              bool compensation);

        KFilterArray const & getFilterArrayR () {return filter_r;}
        KFilterArray const & getFilterArrayZ () {return filter_z;}

    protected:

        KFilterArray filter_r;
        KFilterArray filter_z;

};

#endif
