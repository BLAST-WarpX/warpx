/* Copyright 2019 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PSATD_ALGORITHM_RZ_H_
#define WARPX_PSATD_ALGORITHM_RZ_H_

#include "SpectralBaseAlgorithmRZ.H"

#include <ablastr/utils/Enums.H>


/* \brief Class that updates the field in spectral space
 * and stores the coefficients of the corresponding update equation.
 */
class PsatdAlgorithmRZ : public SpectralBaseAlgorithmRZ
{

    public:
        PsatdAlgorithmRZ(SpectralKSpaceRZ const & spectral_kspace,
                         amrex::DistributionMapping const & dm,
                         const SpectralFieldIndex& spectral_index,
                         int n_rz_azimuthal_modes, int norder_z,
                         ablastr::utils::enums::GridType grid_type,
                         amrex::Real dt_step,
                         bool update_with_rho,
                         bool time_averaging,
                         JInTime J_in_time,
                         RhoInTime rho_in_time,
                         bool dive_cleaning,
                         bool divb_cleaning);
        // Redefine functions from base class
        void pushSpectralFields(SpectralFieldDataRZ & f) final;

        void InitializeSpectralCoefficients(SpectralFieldDataRZ const & f);

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

        bool coefficients_initialized;
        // Note that dt is saved to use in InitializeSpectralCoefficients
        amrex::Real m_dt;
        bool m_update_with_rho;
        bool m_time_averaging;
        JInTime m_J_in_time;
        bool m_dive_cleaning;
        bool m_divb_cleaning;
        SpectralRealCoefficients C_coef, S_ck_coef, X1_coef, X2_coef, X3_coef;
        SpectralRealCoefficients X5_coef, X6_coef;
};

#endif // WARPX_PSATD_ALGORITHM_RZ_H_
