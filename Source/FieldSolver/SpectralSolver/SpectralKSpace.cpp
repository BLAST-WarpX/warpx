/* Copyright 2019-2020 Andrew Myers, Maxence Thevenet, Remi Lehe
 * Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SpectralKSpace.H"

#include "WarpX.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxList.H>
#include <AMReX_GpuComplex.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>

#include <array>
#include <cmath>
#include <vector>

using namespace amrex;

/* \brief Initialize k space object.
 *
 * \param realspace_ba Box array that corresponds to the decomposition
 * of the fields in real space (cell-centered ; includes guard cells)
 * \param dm Indicates which MPI proc owns which box, in realspace_ba.
 * \param realspace_dx Cell size of the grid in real space
 */
SpectralKSpace::SpectralKSpace( const BoxArray& realspace_ba,
                                const DistributionMapping& dm,
                                const RealVect realspace_dx )
    : dx(realspace_dx)  // Store the cell size as member `dx`
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        realspace_ba.ixType()==IndexType::TheCellType(),
        "SpectralKSpace expects a cell-centered box.");

    // Create the box array that corresponds to spectral space
    BoxList spectral_bl; // Create empty box list
    // Loop over boxes and fill the box list
    for (int i=0; i < realspace_ba.size(); i++ ) {
        // For local FFTs, boxes in spectral space start at 0 in
        // each direction and have the same number of points as the
        // (cell-centered) real space box
        const Box realspace_bx = realspace_ba[i];
        IntVect fft_size = realspace_bx.length();
        // Because the spectral solver uses real-to-complex FFTs, we only
        // need the positive k values along the fastest axis
        // (first axis for AMReX Fortran-order arrays) in spectral space.
        // This effectively reduces the size of the spectral space by half
        // see e.g. the FFTW documentation for real-to-complex FFTs
        IntVect spectral_bx_size = fft_size;
        spectral_bx_size[0] = fft_size[0]/2 + 1;
        // Define the corresponding box
        const Box spectral_bx = Box( IntVect::TheZeroVector(),
                               spectral_bx_size - IntVect::TheUnitVector() );
        spectral_bl.push_back( spectral_bx );
    }
    spectralspace_ba.define( spectral_bl );

    // Allocate the components of the k vector: kx, ky (only in 3D), kz
    for (int i_dim=0; i_dim<AMREX_SPACEDIM; i_dim++) {
        // Real-to-complex FFTs: first axis contains only the positive k
        const auto only_positive_k = (i_dim==0);
        k_vec[i_dim] = getKComponent(dm, realspace_ba, i_dim, only_positive_k);
    }
}

/* For each box, in `spectralspace_ba`, which is owned by the local MPI rank
 * (as indicated by the argument `dm`), compute the values of the
 * corresponding k coordinate along the dimension specified by `i_dim`
 */
KVectorComponent
SpectralKSpace::getKComponent( const DistributionMapping& dm,
                               const BoxArray& realspace_ba,
                               const int i_dim,
                               const bool only_positive_k ) const
{
    // Initialize an empty DeviceVector in each box
    KVectorComponent k_comp(spectralspace_ba, dm);
    // Loop over boxes and allocate the corresponding DeviceVector
    // for each box owned by the local MPI proc
    for ( MFIter mfi(spectralspace_ba, dm); mfi.isValid(); ++mfi ){
        const Box bx = spectralspace_ba[mfi];
        Gpu::DeviceVector<Real>& k = k_comp[mfi];

        // Allocate k to the right size
        const int N = bx.length( i_dim );
        k.resize( N );
        Real* pk = k.data();

        // Fill the k vector
        IntVect fft_size = realspace_ba[mfi].length();
        const Real dk = 2*MathConst::pi/(fft_size[i_dim]*dx[i_dim]);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( bx.smallEnd(i_dim) == 0,
            "Expected box to start at 0, in spectral space.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( bx.bigEnd(i_dim) == N-1,
            "Expected different box end index in spectral space.");
        if (only_positive_k){
            // Fill the full axis with positive k values
            // (typically: first axis, in a real-to-complex FFT)
            amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                pk[i] = i*dk;
            });
        } else {
            const int mid_point = (N+1)/2;
            amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                if (i < mid_point) {
                    // Fill positive values of k
                    // (FFT conventions: first half is positive)
                    pk[i] = i*dk;
                } else {
                    // Fill negative values of k
                    // (FFT conventions: second half is negative)
                    pk[i] = (i-N)*dk;
                }
            });
        }
    }
    return k_comp;
}

/* For each box, in `spectralspace_ba`, which is owned by the local MPI rank
 * (as indicated by the argument `dm`), compute the values of the
 * corresponding correcting "shift" factor, along the dimension
 * specified by `i_dim`.
 *
 * (By default, we assume the FFT is done from/to a collocated grid in real space
 * If the FFT is performed from/to a cell-centered grid in real space,
 * a correcting "shift" factor must be applied in spectral space.)
 */
SpectralShiftFactor
SpectralKSpace::getSpectralShiftFactor( const DistributionMapping& dm,
                                        const int i_dim,
                                        const int shift_type ) const
{
    // Initialize an empty DeviceVector in each box
    SpectralShiftFactor shift_factor( spectralspace_ba, dm );
    // Loop over boxes and allocate the corresponding DeviceVector
    // for each box owned by the local MPI proc
    for ( MFIter mfi(spectralspace_ba, dm); mfi.isValid(); ++mfi ){
        const Gpu::DeviceVector<Real>& k = k_vec[i_dim][mfi];
        Gpu::DeviceVector<Complex>& shift = shift_factor[mfi];

        // Allocate shift coefficients
        const auto N = static_cast<int>(k.size());
        shift.resize(N);
        Real const* pk = k.data();
        Complex* pshift = shift.data();

        // Fill the shift coefficients
        Real sign = 0;
        switch (shift_type){
            case ShiftType::TransformFromCellCentered: sign = -1.; break;
            case ShiftType::TransformToCellCentered: sign = 1.;
        }
        const Complex I{0,1};
        const auto t_dx_idim = dx[i_dim];
        amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
        {
            pshift[i] = amrex::exp( I*sign*pk[i]*0.5_rt*t_dx_idim);
        });
    }
    return shift_factor;
}

/* \brief For each box, in `spectralspace_ba`, which is owned by the local MPI
 * rank (as indicated by the argument `dm`), compute the values of the
 * corresponding finite-order modified k vector, along the
 * dimension specified by `i_dim`
 *
 * The finite-order modified k vector is the spectral-space representation
 * of a finite-order stencil in real space.
 *
 * \param n_order Order of accuracy of the stencil, in discretizing
 *                a spatial derivative
 * \param grid_type type of grid (collocated or not)
 */
KVectorComponent
SpectralKSpace::getModifiedKComponent (const DistributionMapping& dm,
                                       const int i_dim,
                                       const int n_order,
                                       ablastr::utils::enums::GridType grid_type) const
{
    // Initialize an empty DeviceVector in each box
    KVectorComponent modified_k_comp(spectralspace_ba, dm);

    if (n_order == -1) { // Infinite-order case
        for ( MFIter mfi(spectralspace_ba, dm); mfi.isValid(); ++mfi ){
            const Gpu::DeviceVector<Real>& k = k_vec[i_dim][mfi];
            Gpu::DeviceVector<Real>& modified_k = modified_k_comp[mfi];

            // Allocate modified_k to the same size as k
            const auto N = static_cast<int>(k.size());;
            modified_k.resize(N);

            // Fill the modified k vector
            Gpu::copyAsync(Gpu::deviceToDevice, k.begin(), k.end(), modified_k.begin());
        }
    } else {

        // Compute real-space stencil coefficients
        Vector<Real> h_stencil_coef = WarpX::getFornbergStencilCoefficients(n_order, grid_type);
        Gpu::DeviceVector<Real> d_stencil_coef(h_stencil_coef.size());
        Gpu::copyAsync(Gpu::hostToDevice, h_stencil_coef.begin(), h_stencil_coef.end(),
                       d_stencil_coef.begin());
        Gpu::synchronize();
        const auto nstencil = static_cast<int>(d_stencil_coef.size());
        Real const* p_stencil_coef = d_stencil_coef.data();

        // Loop over boxes and allocate the corresponding DeviceVector
        // for each box owned by the local MPI proc
        for ( MFIter mfi(spectralspace_ba, dm); mfi.isValid(); ++mfi ){
            const Real delta_x = dx[i_dim];
            const Gpu::DeviceVector<Real>& k = k_vec[i_dim][mfi];
            Gpu::DeviceVector<Real>& modified_k = modified_k_comp[mfi];

            // Allocate modified_k to the same size as k
            const auto N = static_cast<int>(k.size());;
            modified_k.resize(N);
            Real const* p_k = k.data();
            Real * p_modified_k = modified_k.data();

            // Fill the modified k vector
            amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                p_modified_k[i] = 0;
                for (int n=0; n<nstencil; n++){
                    if (grid_type == GridType::Collocated){
                        p_modified_k[i] += p_stencil_coef[n]*
                            std::sin( p_k[i]*(n+1)*delta_x )/( (n+1)*delta_x );
                    } else {
                        p_modified_k[i] += p_stencil_coef[n]* \
                            std::sin( p_k[i]*(n+0.5_rt)*delta_x )/( (n+0.5_rt)*delta_x );
                    }
                }

                // By construction, at finite order and for a collocated grid,
                // the *modified* k corresponding to the Nyquist frequency
                // (i.e. highest *real* k) is 0. However, the above calculation
                // based on stencil coefficients does not give 0 to machine precision.
                // Therefore, we need to enforce the fact that the modified k be 0 here.
                if (grid_type == GridType::Collocated){
                    if (i_dim == 0){
                        // Because of the real-to-complex FFTs, the first axis (idim=0)
                        // contains only the positive k, and the Nyquist frequency is
                        // the last element of the array.
                        if (i == N-1) {
                            p_modified_k[i] = 0.0_rt;
                        }
                    } else {
                        // The other axes contains both positive and negative k ;
                        // the Nyquist frequency is in the middle of the array.
                        if ( (N%2==0) && (i == N/2) ){
                            p_modified_k[i] = 0.0_rt;
                        }
                    }
                }
            });
        }
    }
    return modified_k_comp;
}
