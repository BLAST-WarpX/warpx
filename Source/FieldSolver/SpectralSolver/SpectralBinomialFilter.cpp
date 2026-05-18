/* Copyright 2020 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "SpectralBinomialFilter.H"

#include <AMReX_REAL.H>

#include <cmath>

using namespace amrex::literals;

/* \brief Initialize the general filter array */
void
SpectralBinomialFilter::InitFilterArray (HankelTransform::RealVector const & kvec,
                                         amrex::Real const dels,
                                         int const npasses,
                                         bool const compensation,
                                         KFilterArray & filter)
{
    const auto N = static_cast<int>(kvec.size());
    filter.resize(N);
    amrex::Real* p_filter = filter.data();
    amrex::Real const* p_kvec = kvec.data();

    amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        amrex::Real const ss = std::sin(0.5_rt*p_kvec[i]*dels);
        amrex::Real const ss2 = ss*ss;
        auto filt = static_cast<amrex::Real>(std::pow(1._rt - ss2, npasses));
        if (compensation) {
            filt *= (1._rt + npasses*ss2);
        }
        p_filter[i] = filt;
    });
}

/* \brief Initialize the radial and longitudinal filter arrays */
void
SpectralBinomialFilter::InitFilterArray (RealKVector const & kr,
                                         RealKVector const & kz,
                                         amrex::RealVect const dx,
                                         amrex::IntVect const filter_npass_each_dir,
                                         bool const compensation)
{
    // Note that this includes the kr values for all modes
    InitFilterArray(kr, dx[0], filter_npass_each_dir[0], compensation, filter_r);
    InitFilterArray(kz, dx[1], filter_npass_each_dir[1], compensation, filter_z);
}
