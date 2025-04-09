/* Copyright 2021 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PSATD_ALGORITHM_PML_RZ_H_
#define WARPX_PSATD_ALGORITHM_PML_RZ_H_

#include "SpectralBaseAlgorithmRZ.H"

#include <ablastr/utils/Enums.H>


/* \brief Class that updates the field in spectral space
 * and stores the coefficients of the corresponding update equation.
 */
class PsatdAlgorithmPmlRZ : public SpectralBaseAlgorithmRZ
{

    public:
        PsatdAlgorithmPmlRZ (SpectralKSpaceRZ const & spectral_kspace,
                             amrex::DistributionMapping const & dm,
                             const SpectralFieldIndex& spectral_index,
                             int n_rz_azimuthal_modes,
                             int norder_z,
                             ablastr::utils::enums::GridType grid_type,
                             amrex::Real dt_step);

        // Redefine functions from base class
        void pushSpectralFields (SpectralFieldDataRZ & f) final;

        void InitializeSpectralCoefficients (SpectralFieldDataRZ const & f);

        /**
         * \brief Virtual function for current correction in Fourier space
         * (<a href="https://doi.org/10.1016/j.jcp.2013.03.010"> Vay et al, 2013</a>).
         * This function overrides the virtual function \c CurrentCorrection in the
         * base class \c SpectralBaseAlgorithmRZ and cannot be overridden by further
         * derived classes.
         *
         * \param[in,out] field_data All fields in Fourier space
         */
        void CurrentCorrection (SpectralFieldDataRZ& field_data) final;

        /**
         * \brief Virtual function for Vay current deposition in Fourier space
         * (<a href="https://doi.org/10.1016/j.jcp.2013.03.010"> Vay et al, 2013</a>).
         * This function overrides the virtual function \c VayDeposition in the
         * base class \c SpectralBaseAlgorithmRZ and cannot be overridden by further
         * derived classes.
         *
         * \param[in,out] field_data All fields in Fourier space
         */
        void VayDeposition (SpectralFieldDataRZ& field_data) final;

    private:

        bool coefficients_initialized = false;
        // Note that dt is saved to use in InitializeSpectralCoefficients
        amrex::Real m_dt;
        SpectralRealCoefficients C_coef, S_ck_coef;
};

#endif // WARPX_PSATD_ALGORITHM_PML_RZ_H_
